"""
Live Price Agent — Multi-source real-time price fetcher for IDX stocks

Problem: Yahoo Finance (and Google Finance) IDX data can go stale after
IDX holidays. This agent tries multiple sources with freshness validation.

Sources (in priority order):
  1. Google Finance scraper     — HTML scrape, proven to return values
  2. Yahoo Finance v8 chart     — REST API, with freshness timestamp check
  3. Yahoo Finance v7 quote     — REST API with session/crumb
  4. Stockbit web scraper       — HTML scrape fallback

Freshness logic:
  - IDX market hours: 08:55–15:00 WIB (01:55–08:00 UTC)
  - During market hours: data older than 30 min → STALE
  - Outside market hours: any data is OK (last close is fine)

Usage:
    from agents.live_price import get_live_price, get_live_prices, run_live_price_check
"""

import re
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ─── Constants ───────────────────────────────────────────────────────────────

# IDX market hours in UTC
IDX_OPEN_HOUR_UTC = 1    # 08:55 WIB = 01:55 UTC → use 1
IDX_OPEN_MIN_UTC = 55
IDX_CLOSE_HOUR_UTC = 8   # 15:00 WIB = 08:00 UTC
IDX_CLOSE_MIN_UTC = 0

STALE_THRESHOLD_MINUTES = 30    # during market hours
REQUEST_TIMEOUT = 10             # seconds per HTTP request

WATCHLIST = [
    "BBCA", "BBRI", "BMRI", "BBNI",
    "TLKM", "EXCL",
    "ANTM", "MDKA", "MEDC",
    "GOTO", "BUKA",
    "ASII", "AALI",
    "UNVR", "ICBP", "INDF",
    "ADRO", "PTBA",
    "SMGR", "INTP",
    "AKRA", "ELSA",
]

_GOOGLE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

_YF_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; YFetcher/1.0)",
    "Accept": "application/json",
}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _ts_to_wib_str(ts_utc: Optional[int]) -> str:
    """Convert Unix timestamp (UTC) to human-readable WIB string."""
    if not ts_utc:
        return "unknown"
    try:
        dt_utc = datetime.fromtimestamp(ts_utc, tz=timezone.utc)
        dt_wib = dt_utc + timedelta(hours=7)
        return dt_wib.strftime("%Y-%m-%d %H:%M WIB")
    except Exception:
        return "unknown"


def is_market_hours(now: Optional[datetime] = None) -> bool:
    """
    Return True if current time falls within IDX trading hours (UTC).
    IDX: Mon–Fri 08:55–15:00 WIB → 01:55–08:00 UTC
    Note: does not account for public holidays.
    """
    if now is None:
        now = _now_utc()
    # IDX only trades Mon–Fri
    if now.weekday() >= 5:  # 5=Sat, 6=Sun
        return False
    open_minutes = IDX_OPEN_HOUR_UTC * 60 + IDX_OPEN_MIN_UTC   # 115
    close_minutes = IDX_CLOSE_HOUR_UTC * 60 + IDX_CLOSE_MIN_UTC  # 480
    current_minutes = now.hour * 60 + now.minute
    return open_minutes <= current_minutes < close_minutes


def is_fresh(timestamp_utc: Optional[int], max_age_minutes: int = STALE_THRESHOLD_MINUTES) -> bool:
    """
    Return True if timestamp is recent enough.

    Rules:
      - If outside market hours: always True (last close price is fine)
      - If inside market hours: True only if age < max_age_minutes
      - If timestamp is None/0: always False
    """
    if not timestamp_utc:
        return False

    now = _now_utc()

    # Outside market hours → last close price is acceptable regardless of age
    if not is_market_hours(now):
        return True

    age_seconds = now.timestamp() - timestamp_utc
    return age_seconds < max_age_minutes * 60


def _parse_google_price(text: str) -> Optional[float]:
    """Parse Google Finance price string like 'Rp 3,050.00' → 3050.0"""
    try:
        # Remove currency prefix (Rp, IDR, etc.) and whitespace
        cleaned = re.sub(r'[^\d.,]', '', text).strip()
        # Remove thousand separators (commas) then parse
        cleaned = cleaned.replace(',', '')
        return float(cleaned) if cleaned else None
    except Exception:
        return None


def _parse_google_pct(text: str) -> Optional[float]:
    """Parse Google Finance change pct like '-5.73%' → -5.73"""
    try:
        cleaned = text.strip().replace('%', '').replace(',', '.')
        return float(cleaned)
    except Exception:
        return None


# ─── Source 1: Google Finance ────────────────────────────────────────────────

def _fetch_google_finance(ticker: str) -> Optional[dict]:
    """
    Scrape Google Finance for IDX stock price.
    Returns partial dict or None on failure.
    ticker should be bare (e.g. 'TLKM', not 'TLKM.JK')
    """
    url = f"https://www.google.com/finance/quote/{ticker}:IDX"
    try:
        resp = requests.get(url, headers=_GOOGLE_HEADERS, timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200:
            logger.debug(f"[GFinance] {ticker} HTTP {resp.status_code}")
            return None

        # Lazy import to keep dependency optional in tests
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")

        # Price: div class 'YMlKec fxKbKc'
        price_el = soup.find(class_="YMlKec fxKbKc")
        if not price_el:
            logger.debug(f"[GFinance] {ticker} price element not found")
            return None

        price = _parse_google_price(price_el.text)
        if not price:
            return None

        # Change %: first occurrence of class 'JwB6zf'
        # Note: first element on page is usually the stock's own change %
        pct_el = soup.find(class_="JwB6zf")
        change_pct = _parse_google_pct(pct_el.text) if pct_el else None

        # Timestamp: look for sibling of price container
        # Google Finance shows "Mar 17, 4:40 PM GMT+7" near price
        ts_utc = None
        try:
            price_container = price_el
            for _ in range(6):
                price_container = price_container.parent
            for sibling in price_container.next_siblings:
                sibling_text = sibling.text.strip() if hasattr(sibling, "text") else ""
                if sibling_text:
                    # Try to parse "Mar 17, 4:40:00 PM GMT+7"
                    # or "Mar 17, 4:40 PM GMT+7 · IDR · IDX"
                    ts_match = re.search(
                        r"(\w{3}\s+\d+,\s+\d+:\d+(?::\d+)?\s+[AP]M\s+GMT[+-]\d+)",
                        sibling_text
                    )
                    if ts_match:
                        ts_str = ts_match.group(1)
                        # Parse with timezone offset
                        for fmt in ["%b %d, %I:%M:%S %p GMT%z", "%b %d, %I:%M %p GMT%z"]:
                            try:
                                # Python strptime doesn't handle 'GMT+7' well → replace
                                ts_str_fixed = ts_str.replace("GMT+", "+").replace("GMT-", "-")
                                dt = datetime.strptime(ts_str_fixed, fmt.replace("GMT%z", "%z"))
                                # Add year (Google doesn't show year)
                                dt = dt.replace(year=_now_utc().year)
                                ts_utc = int(dt.timestamp())
                                break
                            except ValueError:
                                continue
                    break
        except Exception:
            pass

        # If we couldn't parse timestamp from page, use 0 (unknown)
        # Freshness will be False if ts=0
        return {
            "price": price,
            "change_pct": change_pct,
            "timestamp_utc": ts_utc,
            "source": "google_finance",
        }

    except ImportError:
        logger.error("[GFinance] BeautifulSoup not installed")
        return None
    except requests.RequestException as e:
        logger.debug(f"[GFinance] {ticker} request error: {e}")
        return None
    except Exception as e:
        logger.debug(f"[GFinance] {ticker} error: {e}")
        return None


# ─── Source 2: Yahoo Finance v8 chart ────────────────────────────────────────

def _fetch_yf_v8(ticker: str) -> Optional[dict]:
    """
    Fetch price from Yahoo Finance v8 chart endpoint.
    ticker format: 'TLKM' (we append .JK internally)
    Returns partial dict or None.
    """
    yf_symbol = f"{ticker}.JK"
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yf_symbol}"
    try:
        resp = requests.get(url, headers=_YF_HEADERS, timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200:
            logger.debug(f"[YFv8] {ticker} HTTP {resp.status_code}")
            return None

        data = resp.json()
        result = data.get("chart", {}).get("result", [])
        if not result:
            return None

        meta = result[0].get("meta", {})
        price = meta.get("regularMarketPrice")
        ts_utc = meta.get("regularMarketTime")
        change_pct = meta.get("regularMarketChangePercent")

        if not price:
            return None

        return {
            "price": float(price),
            "change_pct": float(change_pct) if change_pct is not None else None,
            "timestamp_utc": ts_utc,
            "source": "yf_v8",
        }

    except requests.RequestException as e:
        logger.debug(f"[YFv8] {ticker} request error: {e}")
        return None
    except Exception as e:
        logger.debug(f"[YFv8] {ticker} error: {e}")
        return None


# ─── Source 3: Yahoo Finance v7 quote (with session crumb) ───────────────────

_yf_session: Optional[requests.Session] = None
_yf_crumb: Optional[str] = None
_yf_crumb_ts: float = 0.0


def _get_yf_crumb() -> Optional[str]:
    """Obtain Yahoo Finance crumb cookie for authenticated requests."""
    global _yf_session, _yf_crumb, _yf_crumb_ts

    # Reuse crumb if fresh (< 30 min)
    if _yf_crumb and (time.time() - _yf_crumb_ts) < 1800:
        return _yf_crumb

    try:
        _yf_session = requests.Session()
        _yf_session.get(
            "https://finance.yahoo.com",
            headers=_GOOGLE_HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        crumb_resp = _yf_session.get(
            "https://query2.finance.yahoo.com/v1/test/getcrumb",
            headers=_GOOGLE_HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        if crumb_resp.status_code == 200:
            _yf_crumb = crumb_resp.text.strip()
            _yf_crumb_ts = time.time()
            return _yf_crumb
    except Exception as e:
        logger.debug(f"[YFv7] crumb error: {e}")
    return None


def _fetch_yf_v7(ticker: str) -> Optional[dict]:
    """
    Fetch price via Yahoo Finance v7 quote API with crumb auth.
    ticker format: 'TLKM' (we append .JK)
    """
    global _yf_session

    yf_symbol = f"{ticker}.JK"
    crumb = _get_yf_crumb()
    if not crumb or _yf_session is None:
        logger.debug(f"[YFv7] {ticker} no crumb available")
        return None

    url = "https://query1.finance.yahoo.com/v7/finance/quote"
    params = {"symbols": yf_symbol, "crumb": crumb}
    try:
        resp = _yf_session.get(
            url, params=params, headers=_GOOGLE_HEADERS, timeout=REQUEST_TIMEOUT
        )
        if resp.status_code != 200:
            logger.debug(f"[YFv7] {ticker} HTTP {resp.status_code}")
            return None

        data = resp.json()
        results = data.get("quoteResponse", {}).get("result", [])
        if not results:
            return None

        q = results[0]
        price = q.get("regularMarketPrice")
        ts_utc = q.get("regularMarketTime")
        change_pct = q.get("regularMarketChangePercent")

        if not price:
            return None

        return {
            "price": float(price),
            "change_pct": float(change_pct) if change_pct is not None else None,
            "timestamp_utc": ts_utc,
            "source": "yf_v7",
        }

    except requests.RequestException as e:
        logger.debug(f"[YFv7] {ticker} request error: {e}")
        return None
    except Exception as e:
        logger.debug(f"[YFv7] {ticker} error: {e}")
        return None


# ─── Source 4: Stockbit web scraper ──────────────────────────────────────────

def _fetch_stockbit(ticker: str) -> Optional[dict]:
    """
    Scrape Stockbit for IDX stock price.
    Tries the Stockbit API summary endpoint.
    """
    # Stockbit uses SPA (#/symbol/TICKER) which doesn't return HTML content directly
    # Try undocumented API endpoint
    url = f"https://api.stockbit.com/v2.4/company/{ticker}/summary"
    try:
        resp = requests.get(
            url,
            headers={**_GOOGLE_HEADERS, "Referer": f"https://stockbit.com/#/symbol/{ticker}"},
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code != 200:
            logger.debug(f"[Stockbit] {ticker} HTTP {resp.status_code}")
            return None

        data = resp.json()
        # Try common response structures
        price = (
            data.get("data", {}).get("last_price")
            or data.get("data", {}).get("close")
            or data.get("last_price")
            or data.get("close")
        )
        if not price:
            return None

        change_pct = (
            data.get("data", {}).get("change_percent")
            or data.get("change_percent")
        )
        ts_utc = data.get("data", {}).get("timestamp") or data.get("timestamp")

        return {
            "price": float(price),
            "change_pct": float(change_pct) if change_pct is not None else None,
            "timestamp_utc": ts_utc,
            "source": "stockbit",
        }

    except requests.RequestException as e:
        logger.debug(f"[Stockbit] {ticker} request error: {e}")
        return None
    except Exception as e:
        logger.debug(f"[Stockbit] {ticker} error: {e}")
        return None


def get_live_price(ticker: str) -> dict:
    """
    Fetch live price for a single IDX ticker using multi-source fallback.

    Args:
        ticker: bare ticker string, e.g. 'TLKM' or 'TLKM.JK' (JK suffix stripped)

    Returns:
        {
            'ticker': 'TLKM',
            'price': 3050.0,
            'change_pct': -5.73,        # None if unavailable
            'source': 'google_finance',  # which source succeeded
            'timestamp': '2026-03-24 09:15 WIB',
            'timestamp_utc': 1742792100,  # raw unix ts; None if unknown
            'is_fresh': True,             # False if data is stale
            'error': None,                # str if all sources failed
        }
    """
    # Normalize ticker
    clean_ticker = ticker.upper().replace(".JK", "").strip()

    last_error: Optional[str] = None
    best_result: Optional[dict] = None  # fallback: use freshest stale result

    # Build source list at call time so patches work correctly in tests
    sources = [
        ("google_finance", _fetch_google_finance),
        ("yf_v8", _fetch_yf_v8),
        ("yf_v7", _fetch_yf_v7),
        ("stockbit", _fetch_stockbit),
    ]

    for source_name, fetch_fn in sources:
        try:
            raw = fetch_fn(clean_ticker)
        except Exception as e:
            last_error = str(e)
            logger.debug(f"[LivePrice] {clean_ticker} source={source_name} exception: {e}")
            continue

        if raw is None:
            logger.debug(f"[LivePrice] {clean_ticker} source={source_name} returned None")
            continue

        price = raw.get("price")
        if not price or price <= 0:
            continue

        ts_utc = raw.get("timestamp_utc")
        fresh = is_fresh(ts_utc)

        result = {
            "ticker": clean_ticker,
            "price": float(price),
            "change_pct": raw.get("change_pct"),
            "source": raw.get("source", source_name),
            "timestamp": _ts_to_wib_str(ts_utc),
            "timestamp_utc": ts_utc,
            "is_fresh": fresh,
            "error": None,
        }

        if fresh:
            # Fresh data — return immediately
            logger.info(f"[LivePrice] {clean_ticker} fresh from {source_name} @ {result['timestamp']}")
            return result

        # Stale — keep as best fallback but continue trying other sources
        if best_result is None:
            best_result = result
        logger.debug(
            f"[LivePrice] {clean_ticker} source={source_name} stale "
            f"(ts={_ts_to_wib_str(ts_utc)}), trying next..."
        )

    # All sources tried. Return best stale result if available.
    if best_result is not None:
        best_result["is_fresh"] = False
        logger.warning(
            f"[LivePrice] {clean_ticker} all sources stale — "
            f"returning best from {best_result['source']}"
        )
        return best_result

    # Total failure
    logger.error(f"[LivePrice] {clean_ticker} all sources failed")
    return {
        "ticker": clean_ticker,
        "price": None,
        "change_pct": None,
        "source": None,
        "timestamp": "unknown",
        "timestamp_utc": None,
        "is_fresh": False,
        "error": last_error or "All sources failed",
    }


def get_live_prices(tickers: list) -> dict:
    """
    Batch fetch live prices for multiple IDX tickers.

    Args:
        tickers: list of ticker strings (e.g. ['TLKM', 'BBCA'] or ['TLKM.JK'])

    Returns:
        {ticker: price_dict, ...}  — keyed by normalized ticker (no .JK suffix)
    """
    results = {}
    for ticker in tickers:
        clean = ticker.upper().replace(".JK", "").strip()
        results[clean] = get_live_price(clean)
        # Small delay between requests to avoid rate limiting
        time.sleep(0.2)
    return results


# ─── Formatting ──────────────────────────────────────────────────────────────

def format_live_prices(prices: dict) -> str:
    """
    Format live price data as a terminal-friendly table.

    Args:
        prices: output from get_live_prices()

    Returns:
        Multi-line string ready for print()
    """
    lines = ["─" * 65]
    lines.append(f"{'TICKER':<8} {'PRICE':>10} {'CHG%':>8} {'FRESH':>6}  {'SOURCE':<15} TIMESTAMP")
    lines.append("─" * 65)

    for ticker in sorted(prices.keys()):
        d = prices[ticker]
        if d.get("error"):
            lines.append(f"{ticker:<8} {'ERROR':>10}  {'N/A':>8}  {'❌':>6}  {'—':<15} {d.get('error', '')[:20]}")
            continue

        price = d.get("price")
        price_str = f"{price:>10,.0f}" if price else f"{'N/A':>10}"

        pct = d.get("change_pct")
        if pct is not None:
            sign = "+" if pct >= 0 else ""
            pct_str = f"{sign}{pct:.2f}%"
        else:
            pct_str = "N/A"

        fresh = "✅" if d.get("is_fresh") else "⚠️"
        source = (d.get("source") or "—")[:14]
        ts = (d.get("timestamp") or "unknown")[:22]

        lines.append(f"{ticker:<8} {price_str}  {pct_str:>8}  {fresh:>6}  {source:<15} {ts}")

    lines.append("─" * 65)

    # Summary
    total = len(prices)
    fresh_count = sum(1 for d in prices.values() if d.get("is_fresh"))
    stale_count = total - fresh_count
    errors = sum(1 for d in prices.values() if d.get("error"))

    in_mh = is_market_hours()
    market_status = "OPEN" if in_mh else "CLOSED"
    now_wib = (_now_utc() + timedelta(hours=7)).strftime("%H:%M WIB")

    lines.append(
        f"Market: {market_status} | {now_wib} | "
        f"Fresh: {fresh_count}/{total} | Stale: {stale_count} | Errors: {errors}"
    )
    return "\n".join(lines)


# ─── Convenience runner ───────────────────────────────────────────────────────

def run_live_price_check(watchlist: Optional[list] = None) -> dict:
    """
    Fetch prices for all 22 watchlist tickers and print status.

    Args:
        watchlist: list of tickers; defaults to WATCHLIST constant

    Returns:
        dict of {ticker: price_dict}
    """
    if watchlist is None:
        watchlist = WATCHLIST

    now_wib = (_now_utc() + timedelta(hours=7)).strftime("%Y-%m-%d %H:%M:%S WIB")
    in_mh = is_market_hours()

    print(f"\n{'═' * 65}")
    print(f"  LIVE PRICE CHECK — {now_wib}")
    print(f"  Market: {'🟢 OPEN' if in_mh else '🔴 CLOSED'} | Fetching {len(watchlist)} tickers...")
    print(f"{'═' * 65}\n")

    prices = get_live_prices(watchlist)
    output = format_live_prices(prices)
    print(output)

    # Source summary
    source_counts: dict = {}
    for d in prices.values():
        src = d.get("source") or "error"
        source_counts[src] = source_counts.get(src, 0) + 1

    print("\nSource breakdown:")
    for src, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"  {src:<20}: {count} tickers")

    return prices


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import logging as _logging
    _logging.basicConfig(level=_logging.WARNING)

    tickers = sys.argv[1:] if len(sys.argv) > 1 else None
    run_live_price_check(tickers)
