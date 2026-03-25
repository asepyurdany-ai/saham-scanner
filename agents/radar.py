"""
Radar Agent — Geopolitical + Commodity price monitor
Monitors: Oil (WTI/Brent), Gold, Global news (Reuters, Al Jazeera)
"""

import yfinance as yf
import feedparser
import anthropic
import requests
import json
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Commodity tickers
COMMODITIES = {
    "Gold": "GC=F",
    "WTI Oil": "CL=F",
    "Brent Oil": "BZ=F",
    "USD/IDR": "IDR=X",
    "Copper": "HG=F",
}

# IHSG sectors impacted by commodities
COMMODITY_IMPACT = {
    "Gold": ["ANTM.JK", "MDKA.JK"],
    "WTI Oil": ["MEDC.JK", "AKRA.JK", "ELSA.JK"],
    "Brent Oil": ["MEDC.JK", "AKRA.JK", "ELSA.JK"],
    "Nickel": ["ANTM.JK", "MDKA.JK", "INCO.JK"],
    "Copper": ["ANTM.JK", "MDKA.JK"],
    "USD/IDR": ["BBCA.JK", "BBRI.JK", "BMRI.JK"],  # Banks sensitive to rupiah
}

# Geopolitical RSS
GEO_FEEDS = [
    {"name": "Reuters", "url": "https://feeds.reuters.com/reuters/businessNews"},
    {"name": "Al Jazeera", "url": "https://www.aljazeera.com/xml/rss/all.xml"},
    {"name": "BBC Business", "url": "https://feeds.bbci.co.uk/news/business/rss.xml"},
    {"name": "CNBC", "url": "https://www.cnbc.com/id/100003114/device/rss/rss.html"},
    {"name": "MarketWatch", "url": "https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines"},
]

GEO_KEYWORDS = [
    # Central banks & monetary policy
    "fed", "federal reserve", "interest rate", "rate hike", "rate cut",
    "powell", "fomc", "tapering", "quantitative", "monetary policy",
    "inflation", "cpi", "deflation",

    # Middle East conflict
    "iran", "israel", "hamas", "hezbollah", "middle east", "gaza",
    "war", "conflict", "missile", "strike", "ceasefire",

    # Energy & commodities
    "opec", "opec+", "oil", "crude", "brent", "energy",
    "gold", "commodity", "copper", "nickel", "coal",

    # Geopolitical powers
    "china", "trade war", "tariff", "sanctions", "ukraine", "russia",
    "north korea", "taiwan", "us dollar", "dollar",

    # Global economy
    "recession", "imf", "world bank", "gdp", "indonesia",
    "asean", "emerging market", "debt", "default",
]

# Context map: event type → IHSG sector impact
# Used to enrich Haiku prompt with domain knowledge
GEO_IMPACT_CONTEXT = """
PANDUAN DAMPAK KE IHSG:

Fed naikan suku bunga:
→ NEGATIF: perbankan (BBCA, BBRI, BMRI, BBNI) — cost of fund naik, NIM tertekan
→ NEGATIF: semua saham — capital outflow dari emerging market, rupiah melemah
→ NEGATIF: GOTO, BUKA — valuasi growth stock turun

Fed turunkan suku bunga / dovish:
→ POSITIF: perbankan (BBCA, BBRI, BMRI, BBNI)
→ POSITIF: semua saham — capital inflow ke emerging market
→ POSITIF: GOTO, BUKA — risk-on sentiment

Konflik Iran-Israel / Middle East memanas:
→ NEGATIF global — risk-off, investor jual aset berisiko
→ POSITIF: ANTM, MDKA — gold naik (safe haven)
→ POSITIF: MEDC, AKRA, ELSA — oil naik
→ NEGATIF: BBRI, BMRI, GOTO — likuiditas ketat

Oil naik (WTI/Brent):
→ POSITIF: MEDC, AKRA, ELSA, ELSA
→ NEGATIF: ASII, ICBP, INDF, UNVR — biaya produksi naik

Gold naik:
→ POSITIF: ANTM, MDKA

Rupiah melemah (USD/IDR naik):
→ NEGATIF: BBCA, BBRI, BMRI — valuta asing jadi mahal
→ NEGATIF: ASII, UNVR — impor jadi mahal
→ POSITIF: ADRO, PTBA — ekspor komoditas dalam USD

China ekonomi melemah / trade war:
→ NEGATIF: ADRO, PTBA, ANTM — ekspor komoditas ke China
→ NEGATIF: AALI — sawit ke China

Perang/konflik global baru:
→ NEGATIF: semua saham IHSG — risk-off global
→ POSITIF: ANTM, MDKA — safe haven gold
"""

LAST_COMMODITY_FILE = "data/radar_commodities.json"
LAST_GEO_SEEN_FILE = "data/radar_geo_seen.json"


def _extract_json(text: str):
    """
    Robustly extract JSON from Haiku response.
    Handles: pure JSON, markdown code blocks, leading/trailing text.
    Returns parsed object or None.
    """
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    import re
    match = re.search(r'```(?:json)?\s*([\s\S]+?)\s*```', text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    for pattern in (r'(\[[\s\S]+\])', r'(\{[\s\S]+\})'):
        match = re.search(pattern, text)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
    return None


def send_telegram(msg: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    resp = requests.post(url, json={
        "chat_id": TELEGRAM_CHAT_ID,
        "text": msg,
        "parse_mode": "HTML"
    })
    data = resp.json()
    if not data.get("ok"):
        print(f"[Radar][Telegram ERROR] {data}")
    return data.get("ok", False)


def get_commodity_prices() -> dict:
    """Fetch current commodity prices"""
    prices = {}
    for name, ticker in COMMODITIES.items():
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="2d")
            if not hist.empty:
                current = hist["Close"].iloc[-1]
                prev = hist["Close"].iloc[-2] if len(hist) > 1 else current
                change_pct = ((current - prev) / prev) * 100
                prices[name] = {
                    "current": round(current, 2),
                    "prev": round(prev, 2),
                    "change_pct": round(change_pct, 2),
                }
        except Exception as e:
            print(f"[Radar] Error fetching {name}: {e}")
    return prices


def load_prev_commodities() -> dict:
    os.makedirs("data", exist_ok=True)
    if os.path.exists(LAST_COMMODITY_FILE):
        with open(LAST_COMMODITY_FILE) as f:
            return json.load(f)
    return {}


def save_commodities(data: dict):
    with open(LAST_COMMODITY_FILE, "w") as f:
        json.dump(data, f)


def check_commodity_alerts(prices: dict) -> list:
    """Alert if commodity moves >2% — significant market signal"""
    alerts = []
    for name, data in prices.items():
        if abs(data["change_pct"]) >= 2.0:
            direction = "naik" if data["change_pct"] > 0 else "turun"
            affected = COMMODITY_IMPACT.get(name, [])
            alerts.append({
                "commodity": name,
                "change_pct": data["change_pct"],
                "direction": direction,
                "current": data["current"],
                "affected_stocks": [t.replace(".JK", "") for t in affected],
            })
    return alerts


def fetch_geopolitical_news() -> list:
    """Fetch global news, filter for market-relevant geopolitics"""
    articles = []
    for feed in GEO_FEEDS:
        try:
            d = feedparser.parse(feed["url"])
            for entry in d.entries[:10]:
                title = entry.get("title", "").lower()
                if any(kw in title for kw in GEO_KEYWORDS):
                    articles.append({
                        "source": feed["name"],
                        "title": entry.get("title", ""),
                        "summary": entry.get("summary", "")[:200],
                        "id": entry.get("id", entry.get("link", "")),
                    })
        except Exception as e:
            print(f"[Radar] Error fetching {feed['name']}: {e}")
    return articles


def load_geo_seen() -> set:
    if os.path.exists(LAST_GEO_SEEN_FILE):
        with open(LAST_GEO_SEEN_FILE) as f:
            return set(json.load(f))
    return set()


def save_geo_seen(seen: set):
    os.makedirs("data", exist_ok=True)
    with open(LAST_GEO_SEEN_FILE, "w") as f:
        json.dump(list(seen)[-200:], f)


def analyze_geo_impact(articles: list) -> list:
    """Use Haiku to assess geopolitical impact on IHSG"""
    if not articles:
        return []

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    news_text = "\n".join([
        f"{i+1}. [{a['source']}] {a['title']}: {a['summary']}"
        for i, a in enumerate(articles[:6])
    ])

    prompt = f"""Kamu adalah hedge fund manager yang spesialis pasar saham Indonesia (IHSG).
Tugasmu: analisa dampak berita global berikut ke IHSG dengan presisi tinggi.

{GEO_IMPACT_CONTEXT}

Watchlist IHSG: BBCA, BBRI, BMRI, BBNI, TLKM, EXCL, ANTM, MDKA, MEDC, GOTO, BUKA, ASII, AALI, UNVR, ICBP, INDF, ADRO, PTBA, SMGR, INTP, AKRA, ELSA

Format JSON array — satu objek per berita yang relevan:
[{{
  "judul": "judul singkat berita",
  "event_type": "FED/GEOPOLITIK/KOMODITAS/EKONOMI/LAINNYA",
  "dampak_ihsg": "POSITIF/NEGATIF/NETRAL",
  "level": "TINGGI/SEDANG/RENDAH",
  "saham_terdampak": ["BBCA", "MEDC"],
  "alasan": "penjelasan singkat mengapa saham ini terdampak",
  "analisa": "1 kalimat ringkasan dalam Bahasa Indonesia",
  "reasoning": "rantai logika detail: dari event → mekanisme transmisi → dampak ke sektor/saham IHSG"
}}]

Berita:
{news_text}

Rules:
- Hanya include berita yang RELEVAN ke IHSG (skip berita tidak relevan)
- level TINGGI = dampak besar dalam 1-3 hari trading
- level SEDANG = dampak ada tapi tidak langsung
- level RENDAH = skip saja, jangan include
- Hanya JSON, tidak ada teks lain."""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.content[0].text.strip()
        result = _extract_json(raw)
        if result is None:
            print(f"[Radar] Could not parse JSON from Sonnet response: {raw[:200]}")
            return []
        return result
    except Exception as e:
        print(f"[Radar] Haiku error: {e}")
        return []


def format_commodity_alert(alerts: list, prices: dict) -> str:
    now_wib = datetime.utcnow() + timedelta(hours=7)
    lines = [f"🌍 <b>RADAR ALERT — {now_wib.strftime('%H:%M')} WIB</b>", ""]

    lines.append("⚡ <b>PERGERAKAN KOMODITAS SIGNIFIKAN:</b>")
    for a in alerts:
        emoji = "🟢" if a["change_pct"] > 0 else "🔴"
        affected = ", ".join(a["affected_stocks"])
        lines.append(
            f"{emoji} <b>{a['commodity']}</b> {a['direction']} "
            f"{abs(a['change_pct']):.1f}% → Harga: {a['current']}"
        )
        if affected:
            lines.append(f"   📊 Watch: <b>{affected}</b>")

    lines.append("")
    lines.append("<i>⚠️ Konfirmasi sebelum trading.</i>")
    return "\n".join(lines)


def format_geo_alert(analyses: list) -> str:
    now_wib = datetime.utcnow() + timedelta(hours=7)
    lines = [f"🌐 <b>GEOPOLITIK ALERT — {now_wib.strftime('%H:%M')} WIB</b>", ""]

    EVENT_EMOJI = {
        "FED": "🏦",
        "GEOPOLITIK": "⚔️",
        "KOMODITAS": "🛢",
        "EKONOMI": "📉",
        "LAINNYA": "📌",
    }

    for a in analyses:
        if a.get("level") == "RENDAH":
            continue
        dampak = a.get("dampak_ihsg", "NETRAL")
        emoji = "🟢" if dampak == "POSITIF" else "🔴" if dampak == "NEGATIF" else "⚪"
        event_emoji = EVENT_EMOJI.get(a.get("event_type", "LAINNYA"), "📌")
        level_tag = "🔥" if a.get("level") == "TINGGI" else "📌"
        saham = ", ".join(a.get("saham_terdampak", [])[:5])
        alasan = a.get("alasan", "")
        analisa = a.get("analisa", "")

        lines.append(f"{emoji}{event_emoji}{level_tag} <b>{a.get('judul', '')[:70]}</b>")
        if alasan:
            lines.append(f"   💡 {alasan}")
        if analisa and analisa != alasan:
            lines.append(f"   📝 {analisa}")
        if saham:
            lines.append(f"   📊 Watch: <b>{saham}</b>")
        lines.append("")

    if len(lines) <= 3:
        return None
    lines.append("<i>⚠️ Konfirmasi sebelum trading. DYOR.</i>")
    return "\n".join(lines)


MACRO_SHOCK_PRIORITY_STOCKS = ["GOTO", "BUKA", "BMRI"]


def check_macro_shock() -> list:
    """
    Check for intraday macro shocks:
    - USD/IDR moves > 1.5% intraday → MACRO SHOCK
    - WTI Oil moves > 3% intraday → MACRO SHOCK

    Uses today's open vs current price for intraday move.
    Returns list of shock dicts (may be empty).
    """
    shocks = []
    now_wib = datetime.utcnow() + timedelta(hours=7)
    scan_time = now_wib.strftime("%H:%M")

    shock_checks = [
        ("USD/IDR", "IDR=X", 1.5),
        ("WTI Oil", "CL=F", 3.0),
    ]

    for name, ticker_sym, threshold in shock_checks:
        try:
            t = yf.Ticker(ticker_sym)
            hist = t.history(period="1d", interval="5m")
            if hist is None or hist.empty:
                # Fallback: use daily data
                hist = t.history(period="2d")
                if hist is None or hist.empty:
                    continue
                open_price = float(hist["Open"].iloc[-1])
                current_price = float(hist["Close"].iloc[-1])
            else:
                open_price = float(hist["Open"].iloc[0])
                current_price = float(hist["Close"].iloc[-1])

            if open_price <= 0:
                continue

            change_pct = ((current_price - open_price) / open_price) * 100

            if abs(change_pct) >= threshold:
                shocks.append({
                    "name": name,
                    "current": round(current_price, 2),
                    "open": round(open_price, 2),
                    "change_pct": round(change_pct, 2),
                    "threshold": threshold,
                    "scan_time": scan_time,
                })
        except Exception as e:
            print(f"[Radar] Macro shock check error {name}: {e}")

    return shocks


def format_macro_shock_alert(shocks: list) -> str:
    """Format macro shock alert for Telegram."""
    if not shocks:
        return None

    now_wib = datetime.utcnow() + timedelta(hours=7)
    scan_time = now_wib.strftime("%H:%M")

    lines = [f"🌍 <b>MACRO SHOCK — {scan_time} WIB</b>", ""]

    for shock in shocks:
        sign = "+" if shock["change_pct"] > 0 else ""
        name = shock["name"]
        current = shock["current"]
        change_pct = shock["change_pct"]

        if name == "USD/IDR":
            lines.append(f"💵 {name}: {current:,.0f} ({sign}{change_pct:.1f}% intraday) ‼️")
            if change_pct > 0:
                lines.append("Rupiah melemah tajam → potensi capital outflow.")
            else:
                lines.append("Rupiah menguat tajam → potensi capital inflow.")
        else:
            lines.append(f"🛢 {name}: {current:.2f} ({sign}{change_pct:.1f}% intraday) ‼️")
            if change_pct > 0:
                lines.append("Minyak naik tajam → tekanan inflasi. Watch: MEDC, AKRA.")
            else:
                lines.append("Minyak turun tajam → tekanan sektor energi.")

    priority = ", ".join(MACRO_SHOCK_PRIORITY_STOCKS)
    lines.append("")
    lines.append("⚠️ Pertimbangkan reduce exposure.")
    lines.append(f"Prioritas exit: {priority}")

    return "\n".join(lines)


_last_macro_alert_hash = None
_last_macro_alert_time = 0

def run_macro_shock_check() -> list:
    """Check macro shock — called every 5 min during market hours."""
    global _last_macro_alert_hash, _last_macro_alert_time
    if is_notifications_paused():
        return []
    print("[Radar] Checking macro shock...")
    shocks = check_macro_shock()
    if shocks:
        msg = format_macro_shock_alert(shocks)
        if msg:
            import hashlib, time as _time
            msg_hash = hashlib.md5(msg.encode()).hexdigest()
            now = _time.time()
            # Dedup: skip if same alert sent within 15 minutes
            if msg_hash == _last_macro_alert_hash and (now - _last_macro_alert_time) < 900:
                print("[Radar] Duplicate macro shock suppressed.")
                return shocks
            ok = send_telegram(msg)
            if ok:
                _last_macro_alert_hash = msg_hash
                _last_macro_alert_time = now
            print(f"[Radar] Macro shock alert sent: {ok}")
    else:
        print("[Radar] No macro shock detected")
    return shocks


def run_commodity_check():
    """Check commodity prices — alert if >2% move"""
    print("[Radar] Checking commodities...")
    prices = get_commodity_prices()
    save_commodities(prices)

    alerts = check_commodity_alerts(prices)
    if alerts:
        msg = format_commodity_alert(alerts, prices)
        ok = send_telegram(msg)
        print(f"[Radar] Commodity alert sent: {ok}")
    else:
        print(f"[Radar] No significant commodity moves")

    # --- Shared Intelligence: update market context ---
    try:
        from agents.market_context import update_macro
        rupiah_pct = prices.get("USD/IDR", {}).get("change_pct", 0.0)
        oil_pct = prices.get("WTI Oil", {}).get("change_pct", 0.0)
        gold_pct = prices.get("Gold", {}).get("change_pct", 0.0)
        # Macro shock: rupiah moves >1.5% or oil moves >3%
        shock_active = abs(rupiah_pct) >= 1.5 or abs(oil_pct) >= 3.0
        update_macro(rupiah_pct, oil_pct, gold_pct, shock_active)
    except Exception as e:
        print(f"[Radar] MarketContext macro update error: {e}")

    return prices, alerts


def run_geo_check():
    """Check geopolitical news — alert if relevant"""
    print("[Radar] Checking geopolitical news...")
    seen = load_geo_seen()

    articles = fetch_geopolitical_news()
    new_articles = [a for a in articles if a["id"] not in seen]

    if not new_articles:
        print("[Radar] No new geopolitical news")
        # Update geo context with neutral if no new data (keep existing)
        return

    seen.update(a["id"] for a in new_articles)
    save_geo_seen(seen)

    analyses = analyze_geo_impact(new_articles)
    high_impact = [a for a in analyses if a.get("level") in ["TINGGI", "SEDANG"]]

    if high_impact:
        msg = format_geo_alert(analyses)
        if msg:
            ok = send_telegram(msg)
            print(f"[Radar] Geo alert sent: {ok}")
    else:
        print(f"[Radar] {len(new_articles)} new articles, no high/medium impact")

    # --- Shared Intelligence: update geo context ---
    try:
        from agents.market_context import update_geo
        # Derive risk level from high-impact analyses
        high_neg = [a for a in analyses if a.get("level") == "TINGGI" and a.get("dampak_ihsg") == "NEGATIF"]
        med_neg = [a for a in analyses if a.get("level") == "SEDANG" and a.get("dampak_ihsg") == "NEGATIF"]

        if high_neg:
            risk_level = "HIGH"
            geo_signal = "RISK_OFF"
        elif med_neg:
            risk_level = "MEDIUM"
            geo_signal = "CAUTIOUS"
        else:
            risk_level = "LOW"
            geo_signal = "NEUTRAL"

        active_events = [
            a.get("judul", "")[:60] for a in (high_neg + med_neg)[:5]
        ]
        update_geo(risk_level, active_events, geo_signal)
    except Exception as e:
        print(f"[Radar] MarketContext geo update error: {e}")


def is_notifications_paused() -> bool:
    """Check if notifications are paused for today."""
    try:
        ctx_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'market_context.json')
        with open(ctx_path, 'r') as f:
            ctx = json.load(f)
        paused_until = ctx.get('notifications_paused_until')
        if paused_until:
            from datetime import timezone
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            until = datetime.fromisoformat(paused_until)
            if now < until:
                print(f"[Radar] Notifications paused until {paused_until}. Skipping.")
                return True
    except Exception:
        pass
    return False


def run_radar():
    if is_notifications_paused():
        return
    run_commodity_check()
    run_geo_check()


if __name__ == "__main__":
    run_radar()
