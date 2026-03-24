"""
Intraday Scanner Agent — 5-minute VWAP/BB/Stoch/OBV/Volume/MACD signals
Hedge fund discipline: STRONG BUY only when 5/6 intraday conditions met
Runs at 09:05 WIB and 09:30 WIB during market hours

Indicators (5-minute intraday data from yfinance):
  1. VWAP — price vs VWAP
  2. Bollinger Bands (20, 2) — squeeze + bounce detection
  3. Stochastic (5,3,3) — %K/%D crossup zone
  4. OBV — rising trend = accumulation
  5. Volume spike — current candle vs 20-period avg
  6. MACD (12,26,9) — intraday bullish crossover

Signal levels:
  STRONG BUY: score >= 5
  WATCH:      score >= 3
  AVOID:      score < 3
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
import time
from datetime import datetime, timedelta
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# Reuse WATCHLIST from scanner
WATCHLIST = [
    "BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK",
    "TLKM.JK", "EXCL.JK",
    "ANTM.JK", "MDKA.JK", "MEDC.JK",
    "GOTO.JK", "BUKA.JK",
    "ASII.JK", "AALI.JK",
    "UNVR.JK", "ICBP.JK", "INDF.JK",
    "ADRO.JK", "PTBA.JK",
    "SMGR.JK", "INTP.JK",
    "AKRA.JK", "ELSA.JK",
]

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "5922770410")

# Intraday params
BB_PERIOD = 20
BB_STD = 2
STOCH_K = 5
STOCH_D = 3
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
STRONG_BUY_THRESHOLD = 5
WATCH_THRESHOLD = 3
TP_PCT = 0.025  # +2.5% intraday TP


def send_telegram(msg: str, retries: int = 3) -> bool:
    """Send Telegram message."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    for attempt in range(retries):
        try:
            resp = requests.post(url, json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": msg,
                "parse_mode": "HTML"
            }, timeout=10)
            data = resp.json()
            if data.get("ok"):
                return True
            print(f"[Intraday] Telegram ERROR: {data}")
        except Exception as e:
            print(f"[Intraday] Telegram attempt {attempt+1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(2)
    return False


def get_intraday_data(ticker: str, period: str = "1d", interval: str = "5m", retries: int = 2) -> pd.DataFrame:
    """Fetch 5-minute intraday data from Yahoo Finance."""
    for attempt in range(retries):
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval=interval)
            if not df.empty:
                return df
        except Exception as e:
            print(f"[Intraday] Error fetching {ticker} 5m (attempt {attempt+1}): {e}")
            if attempt < retries - 1:
                time.sleep(1)
    return pd.DataFrame()


def compute_vwap(hist_5m: pd.DataFrame) -> float:
    """
    Compute VWAP from 5-minute intraday data.
    VWAP = cumsum(typical_price * volume) / cumsum(volume)
    typical_price = (high + low + close) / 3
    Returns current VWAP value (float).
    """
    if hist_5m is None or hist_5m.empty:
        return 0.0

    try:
        high = hist_5m["High"]
        low = hist_5m["Low"]
        close = hist_5m["Close"]
        volume = hist_5m["Volume"]

        # Guard against all-zero volume
        if volume.sum() == 0:
            return float(close.iloc[-1]) if not close.empty else 0.0

        typical_price = (high + low + close) / 3
        cum_tp_vol = (typical_price * volume).cumsum()
        cum_vol = volume.cumsum()

        # Avoid division by zero at each step
        vwap_series = cum_tp_vol / cum_vol.replace(0, np.nan)
        vwap = float(vwap_series.iloc[-1])
        return round(vwap, 2) if not np.isnan(vwap) else float(close.iloc[-1])
    except Exception as e:
        print(f"[Intraday] VWAP compute error: {e}")
        return 0.0


def compute_bollinger(hist_5m: pd.DataFrame) -> dict:
    """
    Compute Bollinger Bands (20, 2) from 5-minute data.

    Returns:
        {
            'upper': float, 'middle': float, 'lower': float,
            'band_width': float,
            'squeeze': bool,   # band_width < 20-period avg bw * 0.8
            'bounce': bool,    # price < lower_band * 1.02 (bouncing from lower)
        }
    """
    empty = {"upper": 0.0, "middle": 0.0, "lower": 0.0,
             "band_width": 0.0, "squeeze": False, "bounce": False}

    if hist_5m is None or hist_5m.empty or len(hist_5m) < BB_PERIOD:
        return empty

    try:
        close = hist_5m["Close"]
        ma = close.rolling(BB_PERIOD).mean()
        std = close.rolling(BB_PERIOD).std()

        upper = ma + BB_STD * std
        lower = ma - BB_STD * std
        band_width = upper - lower

        current_price = float(close.iloc[-1])
        current_upper = float(upper.iloc[-1])
        current_middle = float(ma.iloc[-1])
        current_lower = float(lower.iloc[-1])
        current_bw = float(band_width.iloc[-1])

        # Squeeze: current band_width < 20-period avg of band_width * 0.8
        avg_bw = float(band_width.rolling(BB_PERIOD).mean().iloc[-1])
        squeeze = bool(current_bw < avg_bw * 0.8) if not np.isnan(avg_bw) and avg_bw > 0 else False

        # Bounce: price is near or below lower band (price < lower * 1.02)
        bounce = bool(current_price < current_lower * 1.02) if current_lower > 0 else False

        return {
            "upper": round(current_upper, 2),
            "middle": round(current_middle, 2),
            "lower": round(current_lower, 2),
            "band_width": round(current_bw, 2),
            "squeeze": squeeze,
            "bounce": bounce,
        }
    except Exception as e:
        print(f"[Intraday] Bollinger compute error: {e}")
        return empty


def compute_stochastic(hist_5m: pd.DataFrame, k_period: int = STOCH_K, d_period: int = STOCH_D) -> dict:
    """
    Compute Stochastic Oscillator (5,3,3).
    %K = (close - lowest_low) / (highest_high - lowest_low) * 100
    %D = 3-period SMA of %K

    Returns:
        {'k': float, 'd': float}
    """
    empty = {"k": 50.0, "d": 50.0}

    if hist_5m is None or hist_5m.empty or len(hist_5m) < k_period + d_period:
        return empty

    try:
        high = hist_5m["High"]
        low = hist_5m["Low"]
        close = hist_5m["Close"]

        lowest_low = low.rolling(k_period).min()
        highest_high = high.rolling(k_period).max()
        range_hl = highest_high - lowest_low

        # Avoid division by zero
        k_raw = ((close - lowest_low) / range_hl.replace(0, np.nan)) * 100
        k_smooth = k_raw.rolling(d_period).mean()  # Smoothed %K (slow stochastic)
        d_line = k_smooth.rolling(d_period).mean()   # %D

        k_val = float(k_smooth.iloc[-1])
        d_val = float(d_line.iloc[-1])

        k_val = round(k_val, 2) if not np.isnan(k_val) else 50.0
        d_val = round(d_val, 2) if not np.isnan(d_val) else 50.0

        return {"k": k_val, "d": d_val}
    except Exception as e:
        print(f"[Intraday] Stochastic compute error: {e}")
        return empty


def compute_obv_trend(hist_5m: pd.DataFrame) -> bool:
    """
    Compute OBV (On Balance Volume) and detect rising trend.
    OBV rises if close > prev close, falls if close < prev close.
    Returns True if OBV is trending up over last 5 candles.
    """
    if hist_5m is None or hist_5m.empty or len(hist_5m) < 6:
        return False

    try:
        close = hist_5m["Close"]
        volume = hist_5m["Volume"]

        # Compute OBV
        obv = [0]
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i - 1]:
                obv.append(obv[-1] + volume.iloc[i])
            elif close.iloc[i] < close.iloc[i - 1]:
                obv.append(obv[-1] - volume.iloc[i])
            else:
                obv.append(obv[-1])

        obv_series = pd.Series(obv, index=close.index)

        # Check if last 5 candles OBV is rising (each value > previous)
        last_5 = obv_series.iloc[-5:].values
        rising = all(last_5[i] >= last_5[i - 1] for i in range(1, len(last_5)))
        return bool(rising)
    except Exception as e:
        print(f"[Intraday] OBV compute error: {e}")
        return False


def compute_intraday_macd(hist_5m: pd.DataFrame) -> dict:
    """
    Compute MACD (12,26,9) on 5-minute intraday data.

    Returns:
        {'macd': float, 'signal': float, 'bullish': bool}
    """
    empty = {"macd": 0.0, "signal": 0.0, "bullish": False}

    if hist_5m is None or hist_5m.empty or len(hist_5m) < MACD_SLOW + MACD_SIGNAL:
        return empty

    try:
        close = hist_5m["Close"]
        ema_fast = close.ewm(span=MACD_FAST, adjust=False).mean()
        ema_slow = close.ewm(span=MACD_SLOW, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()

        macd_val = float(macd_line.iloc[-1])
        signal_val = float(signal_line.iloc[-1])

        if np.isnan(macd_val) or np.isnan(signal_val):
            return empty

        bullish = bool(macd_val > signal_val)

        return {
            "macd": round(macd_val, 4),
            "signal": round(signal_val, 4),
            "bullish": bullish,
        }
    except Exception as e:
        print(f"[Intraday] MACD compute error: {e}")
        return empty


def compute_volume_spike(hist_5m: pd.DataFrame) -> dict:
    """
    Compute volume spike: current 5m candle vs 20-period avg volume.

    Returns:
        {'current_vol': int, 'avg_vol': float, 'ratio': float, 'spike': bool}
    """
    empty = {"current_vol": 0, "avg_vol": 0.0, "ratio": 1.0, "spike": False}

    if hist_5m is None or hist_5m.empty or len(hist_5m) < 2:
        return empty

    try:
        volume = hist_5m["Volume"]
        current_vol = int(volume.iloc[-1])

        # 20-period avg excluding current candle
        window = min(20, len(volume) - 1)
        avg_vol = float(volume.iloc[-window - 1:-1].mean()) if window > 0 else 0.0

        if avg_vol <= 0:
            return {**empty, "current_vol": current_vol}

        ratio = round(current_vol / avg_vol, 2)
        spike = bool(ratio >= 2.0)

        return {
            "current_vol": current_vol,
            "avg_vol": round(avg_vol, 0),
            "ratio": ratio,
            "spike": spike,
        }
    except Exception as e:
        print(f"[Intraday] Volume spike compute error: {e}")
        return empty


def _get_live_price_for_ticker(ticker: str) -> Optional[float]:
    """
    Fetch live/current price for a ticker using live_price agent.
    Falls back to yfinance intraday close if live_price unavailable.
    Returns None on total failure.
    """
    try:
        from agents.live_price import get_live_price
        result = get_live_price(ticker)
        if result and result.get("price") and not result.get("error"):
            return float(result["price"])
    except Exception as e:
        print(f"[Intraday] live_price fetch error for {ticker}: {e}")
    return None


def compute_intraday_score(ticker: str, hist_5m: pd.DataFrame) -> dict:
    """
    Compute all intraday indicators and score for a ticker.

    Conditions:
      cond1: price > VWAP
      cond2: bollinger squeeze OR price bouncing from lower band
      cond3: stochastic %K < 40 AND %K > %D (crossup zone)
      cond4: OBV trend rising (last 5 candles OBV increasing)
      cond5: volume spike > 2x avg
      cond6: MACD bullish (macd_line > signal_line)

    Returns:
        dict with ticker, price, score, signal, conditions, indicators, TP, CL
    """
    if hist_5m is None or hist_5m.empty:
        return {"ticker": ticker, "error": "No data", "score": 0, "signal": "AVOID"}

    try:
        # Try live price first (multi-source, handles stale Yahoo Finance)
        live_px = _get_live_price_for_ticker(ticker)
        current_price = live_px if live_px else float(hist_5m["Close"].iloc[-1])

        # Compute all indicators
        vwap = compute_vwap(hist_5m)
        bb = compute_bollinger(hist_5m)
        stoch = compute_stochastic(hist_5m)
        obv_rising = compute_obv_trend(hist_5m)
        vol_data = compute_volume_spike(hist_5m)
        macd_data = compute_intraday_macd(hist_5m)

        # 6 conditions
        cond1 = bool(current_price > vwap) if vwap > 0 else False
        cond2 = bool(bb["squeeze"] or bb["bounce"])
        cond3 = bool(stoch["k"] < 40 and stoch["k"] > stoch["d"])
        cond4 = bool(obv_rising)
        cond5 = bool(vol_data["spike"])
        cond6 = bool(macd_data["bullish"])

        conditions = {
            "vwap":   cond1,
            "bb":     cond2,
            "stoch":  cond3,
            "obv":    cond4,
            "volume": cond5,
            "macd":   cond6,
        }

        score = sum(conditions.values())

        if score >= STRONG_BUY_THRESHOLD:
            signal = "STRONG BUY"
        elif score >= WATCH_THRESHOLD:
            signal = "WATCH"
        else:
            signal = "AVOID"

        # TP = +2.5% from entry (intraday)
        tp = round(current_price * (1 + TP_PCT), 0)
        tp_pct = round(TP_PCT * 100, 1)

        # CL = VWAP level (dynamic — exit if candle closes below VWAP)
        cl = round(vwap, 0)

        return {
            "ticker": ticker,
            "price": round(current_price, 0),
            "vwap": vwap,
            "score": score,
            "signal": signal,
            "conditions": conditions,
            "indicators": {
                "vwap": vwap,
                "bb_upper": bb["upper"],
                "bb_middle": bb["middle"],
                "bb_lower": bb["lower"],
                "bb_squeeze": bb["squeeze"],
                "bb_bounce": bb["bounce"],
                "stoch_k": stoch["k"],
                "stoch_d": stoch["d"],
                "obv_rising": obv_rising,
                "vol_ratio": vol_data["ratio"],
                "vol_spike": vol_data["spike"],
                "macd": macd_data["macd"],
                "macd_signal": macd_data["signal"],
                "macd_bullish": macd_data["bullish"],
            },
            "tp": tp,
            "tp_pct": tp_pct,
            "cl": cl,
        }

    except Exception as e:
        print(f"[Intraday] Score compute error for {ticker}: {e}")
        return {"ticker": ticker, "error": str(e), "score": 0, "signal": "AVOID"}


def run_intraday_scan(watchlist: list = None) -> list:
    """
    Scan intraday signals for all tickers in watchlist.
    Returns list of scored results, sorted by score descending.
    """
    if watchlist is None:
        watchlist = WATCHLIST

    results = []
    print(f"[Intraday] Scanning {len(watchlist)} stocks (5m data)...")

    for ticker in watchlist:
        try:
            hist_5m = get_intraday_data(ticker)
            if hist_5m.empty:
                print(f"[Intraday] No 5m data for {ticker}")
                continue

            result = compute_intraday_score(ticker, hist_5m)
            if "error" not in result:
                ind = result.get("indicators", {})
                print(
                    f"  {ticker}: {result['signal']} (score={result['score']}/6, "
                    f"VWAP={'✅' if result['conditions']['vwap'] else '❌'} "
                    f"BB={'✅' if result['conditions']['bb'] else '❌'} "
                    f"Stoch={'✅' if result['conditions']['stoch'] else '❌'} "
                    f"OBV={'✅' if result['conditions']['obv'] else '❌'} "
                    f"Vol={'✅' if result['conditions']['volume'] else '❌'} "
                    f"MACD={'✅' if result['conditions']['macd'] else '❌'})"
                )
                results.append(result)
        except Exception as e:
            print(f"[Intraday] Error processing {ticker}: {e}")

    # Sort: STRONG BUY first, then by score descending
    results.sort(key=lambda x: -x.get("score", 0))
    return results


def _get_market_info() -> dict:
    """Get current market gate + breadth info for alert footer."""
    info = {"gate": "OPEN", "breadth": None, "mode": "NORMAL"}
    try:
        from agents.market_context import get_context
        ctx = get_context()
        info["gate"] = ctx.get("market_gate", "OPEN")
        info["mode"] = ctx.get("market_mode", "NORMAL")
        breadth = ctx.get("breadth_pct")
        if breadth is not None:
            info["breadth"] = round(float(breadth), 0)
    except Exception:
        pass
    return info


def format_intraday_alert(results: list) -> str:
    """
    Format intraday scan results as Telegram alert.

    Format:
        📊 INTRADAY SCAN — 09:15 WIB

        🔥 STRONG BUY:
        BBNI  | 4.390 | Score: 5/6 | VWAP✅ BB✅ Stoch✅ OBV✅ Vol❌ MACD✅
        Entry: 4.390 | TP: 4.500 (+2.5%) | CL: VWAP (4.350)

        👀 WATCH:
        ADRO  | 2.450 | Score: 4/6 | ...

        ⏸ Total scanned: 22 | Strong Buy: 1 | Watch: 3
        Market: Gate=OPEN | Breadth=64% | Mode=NORMAL
    """
    now_wib = datetime.utcnow() + timedelta(hours=7)
    scan_time = now_wib.strftime("%H:%M")

    strong_buy = [r for r in results if r.get("signal") == "STRONG BUY"]
    watch = [r for r in results if r.get("signal") == "WATCH"]
    total = len(results)

    lines = [f"📊 <b>INTRADAY SCAN — {scan_time} WIB</b>", ""]

    def cond_tag(r: dict, key: str) -> str:
        return "✅" if r.get("conditions", {}).get(key) else "❌"

    def format_entry(r: dict) -> list:
        ticker = r["ticker"].replace(".JK", "")
        price = r.get("price", 0)
        score = r.get("score", 0)
        tp = r.get("tp", 0)
        tp_pct = r.get("tp_pct", 2.5)
        cl = r.get("cl", 0)
        vwap = r.get("vwap", 0)

        conds = (
            f"VWAP{cond_tag(r, 'vwap')} "
            f"BB{cond_tag(r, 'bb')} "
            f"Stoch{cond_tag(r, 'stoch')} "
            f"OBV{cond_tag(r, 'obv')} "
            f"Vol{cond_tag(r, 'volume')} "
            f"MACD{cond_tag(r, 'macd')}"
        )
        return [
            f"<b>{ticker:<6}</b> | {price:,.0f} | Score: {score}/6 | {conds}",
            f"Entry: {price:,.0f} | TP: {tp:,.0f} (+{tp_pct}%) | CL: VWAP ({cl:,.0f})",
        ]

    if strong_buy:
        lines.append("🔥 <b>STRONG BUY:</b>")
        for r in strong_buy:
            lines.extend(format_entry(r))
            lines.append("")
    else:
        lines.append("🔥 <b>STRONG BUY:</b> Tidak ada sinyal kuat")
        lines.append("")

    if watch:
        lines.append("👀 <b>WATCH:</b>")
        for r in watch[:5]:
            ticker = r["ticker"].replace(".JK", "")
            price = r.get("price", 0)
            score = r.get("score", 0)
            conds = (
                f"VWAP{cond_tag(r, 'vwap')} "
                f"BB{cond_tag(r, 'bb')} "
                f"Stoch{cond_tag(r, 'stoch')} "
                f"OBV{cond_tag(r, 'obv')} "
                f"Vol{cond_tag(r, 'volume')} "
                f"MACD{cond_tag(r, 'macd')}"
            )
            lines.append(f"<b>{ticker:<6}</b> | {price:,.0f} | Score: {score}/6 | {conds}")
        lines.append("")

    # Footer summary
    market = _get_market_info()
    gate = market.get("gate", "OPEN")
    mode = market.get("mode", "NORMAL")
    breadth = market.get("breadth")
    breadth_str = f"{breadth:.0f}%" if breadth is not None else "N/A"

    lines.append(
        f"⏸ Total scanned: {total} | "
        f"Strong Buy: {len(strong_buy)} | "
        f"Watch: {len(watch)}"
    )
    lines.append(
        f"Market: Gate={gate} | Breadth={breadth_str} | Mode={mode}"
    )
    lines.append("")
    lines.append("<i>⚠️ BUKAN saran investasi. DYOR.</i>")

    return "\n".join(lines)


def send_intraday_alert(results: list) -> bool:
    """Format and send intraday scan alert via Telegram."""
    if not results:
        print("[Intraday] No results to send.")
        return False

    msg = format_intraday_alert(results)
    ok = send_telegram(msg)
    print(f"[Intraday] Alert sent: {ok}")
    return ok


def run_intraday_scan_and_alert(watchlist: list = None) -> list:
    """Full pipeline: scan + send alert. Used by scheduler."""
    print(f"[Intraday] Running intraday scan at {(datetime.utcnow() + timedelta(hours=7)).strftime('%H:%M')} WIB")
    results = run_intraday_scan(watchlist)
    send_intraday_alert(results)
    return results


if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("INTRADAY SCANNER — Standalone Test")
    print("=" * 60)

    # Use small subset for quick test
    test_tickers = sys.argv[1:] if len(sys.argv) > 1 else WATCHLIST[:5]

    print(f"\nScanning {len(test_tickers)} tickers with 5m data...")
    results = run_intraday_scan(test_tickers)

    print(f"\n{'=' * 60}")
    print(f"RESULTS: {len(results)} stocks scanned")
    print(f"Strong Buy: {len([r for r in results if r.get('signal') == 'STRONG BUY'])}")
    print(f"Watch: {len([r for r in results if r.get('signal') == 'WATCH'])}")
    print(f"Avoid: {len([r for r in results if r.get('signal') == 'AVOID'])}")

    print(f"\n{'=' * 60}")
    print("FORMATTED ALERT:")
    print("=" * 60)
    if results:
        alert = format_intraday_alert(results)
        # Strip HTML for terminal readability
        import re
        alert_clean = re.sub(r"<[^>]+>", "", alert)
        print(alert_clean)
    else:
        print("No results — check yfinance connectivity")
