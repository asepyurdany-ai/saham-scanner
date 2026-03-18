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
]

GEO_KEYWORDS = [
    "iran", "israel", "war", "conflict", "opec", "oil", "crude",
    "fed", "interest rate", "inflation", "china", "trade war",
    "sanctions", "indonesia", "asean", "imf", "world bank",
    "recession", "gold", "commodity"
]

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

    prompt = f"""Kamu analis pasar saham Indonesia (IHSG).
Analisa dampak berita global berikut ke IHSG dan komoditas.

Format JSON array:
[{{"judul": "...", "dampak_ihsg": "POSITIF/NEGATIF/NETRAL", "level": "TINGGI/SEDANG/RENDAH", "saham_terdampak": ["ANTM", "MEDC"], "analisa": "1 kalimat Indonesia"}}]

Berita:
{news_text}

Hanya JSON, tidak ada teks lain."""

    try:
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.content[0].text.strip()
        result = _extract_json(raw)
        if result is None:
            print(f"[Radar] Could not parse JSON from Haiku response: {raw[:200]}")
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

    for a in analyses:
        if a.get("level") == "RENDAH":
            continue
        emoji = "🟢" if a["dampak_ihsg"] == "POSITIF" else "🔴" if a["dampak_ihsg"] == "NEGATIF" else "⚪"
        saham = ", ".join(a.get("saham_terdampak", [])[:4])
        lines.append(f"{emoji} <b>{a.get('judul', '')[:60]}</b>")
        lines.append(f"   {a.get('analisa', '')}")
        if saham:
            lines.append(f"   📊 Watch: <b>{saham}</b>")
        lines.append("")

    if len(lines) <= 3:
        return None
    lines.append("<i>⚠️ Konfirmasi sebelum trading.</i>")
    return "\n".join(lines)


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

    return prices, alerts


def run_geo_check():
    """Check geopolitical news — alert if relevant"""
    print("[Radar] Checking geopolitical news...")
    seen = load_geo_seen()

    articles = fetch_geopolitical_news()
    new_articles = [a for a in articles if a["id"] not in seen]

    if not new_articles:
        print("[Radar] No new geopolitical news")
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


def run_radar():
    run_commodity_check()
    run_geo_check()


if __name__ == "__main__":
    run_radar()
