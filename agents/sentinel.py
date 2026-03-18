"""
Sentinel Agent — News sentiment for IHSG
Monitors: Kontan, Bisnis.com, CNBC Indonesia
Uses Claude Haiku for analysis
"""

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

# RSS feeds — Indonesian financial news
RSS_FEEDS = [
    {"name": "Kontan", "url": "https://rss.kontan.co.id/feeds/market"},
    {"name": "Bisnis.com", "url": "https://rss.bisnis.com/market/30/rss.xml"},
    {"name": "CNBC Indonesia", "url": "https://www.cnbcindonesia.com/rss"},
    {"name": "Detik Finance", "url": "https://finance.detik.com/rss"},
]

# Stock mapping — keyword → tickers affected
KEYWORD_MAP = {
    "bank indonesia": ["BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK"],
    "suku bunga": ["BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK"],
    "bi rate": ["BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK"],
    "rupiah": ["BBCA.JK", "BBRI.JK", "BMRI.JK"],
    "nikel": ["ANTM.JK", "MDKA.JK", "INCO.JK"],
    "emas": ["ANTM.JK", "MDKA.JK"],
    "gold": ["ANTM.JK", "MDKA.JK"],
    "batu bara": ["ADRO.JK", "PTBA.JK", "ITMG.JK"],
    "coal": ["ADRO.JK", "PTBA.JK"],
    "minyak": ["MEDC.JK", "AKRA.JK", "ELSA.JK"],
    "oil": ["MEDC.JK", "AKRA.JK", "ELSA.JK"],
    "telkom": ["TLKM.JK"],
    "gojek": ["GOTO.JK"],
    "bukalapak": ["BUKA.JK"],
    "astra": ["ASII.JK"],
    "ojk": ["BBCA.JK", "BBRI.JK", "BMRI.JK"],
    "inflasi": ["UNVR.JK", "ICBP.JK", "INDF.JK"],
    "ekspor": ["ADRO.JK", "ANTM.JK", "MDKA.JK"],
    "sawit": ["AALI.JK", "LSIP.JK"],
}

LAST_SEEN_FILE = "data/sentinel_seen.json"


def _extract_json(text: str):
    """
    Robustly extract JSON from Haiku response.
    Handles: pure JSON, markdown code blocks, leading/trailing text.
    Returns parsed object or None.
    """
    if not text:
        return None
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Strip markdown code blocks
    import re
    match = re.search(r'```(?:json)?\s*([\s\S]+?)\s*```', text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Find first [...] or {...} in text
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
        print(f"[Sentinel][Telegram ERROR] {data}")
    return data.get("ok", False)


def load_seen() -> set:
    os.makedirs("data", exist_ok=True)
    if os.path.exists(LAST_SEEN_FILE):
        with open(LAST_SEEN_FILE) as f:
            return set(json.load(f))
    return set()


def save_seen(seen: set):
    with open(LAST_SEEN_FILE, "w") as f:
        json.dump(list(seen)[-200:], f)  # Keep last 200


def fetch_news() -> list:
    """Fetch latest news from all RSS feeds"""
    articles = []
    for feed in RSS_FEEDS:
        try:
            d = feedparser.parse(feed["url"])
            for entry in d.entries[:5]:  # Latest 5 per source
                articles.append({
                    "source": feed["name"],
                    "title": entry.get("title", ""),
                    "summary": entry.get("summary", "")[:300],
                    "link": entry.get("link", ""),
                    "id": entry.get("id", entry.get("link", "")),
                })
        except Exception as e:
            print(f"[Sentinel] Error fetching {feed['name']}: {e}")
    return articles


def get_affected_tickers(title: str, summary: str) -> list:
    """Map news keywords to affected tickers"""
    text = (title + " " + summary).lower()
    tickers = set()
    for keyword, stocks in KEYWORD_MAP.items():
        if keyword in text:
            tickers.update(stocks)
    return list(tickers)


def analyze_with_haiku(articles: list) -> list:
    """Use Claude Haiku to analyze sentiment of important news"""
    if not articles:
        return []

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    news_text = "\n".join([
        f"{i+1}. [{a['source']}] {a['title']}: {a['summary']}"
        for i, a in enumerate(articles)
    ])

    prompt = f"""Kamu adalah analis pasar saham Indonesia (IHSG).
Analisa berita berikut dan tentukan:
1. Sentimen: POSITIF / NEGATIF / NETRAL
2. Dampak ke pasar: TINGGI / SEDANG / RENDAH
3. Saham yang terdampak (dari IHSG)
4. Ringkasan 1 kalimat dalam Bahasa Indonesia

Format JSON array:
[{{"judul": "...", "sentimen": "POSITIF/NEGATIF/NETRAL", "dampak": "TINGGI/SEDANG/RENDAH", "confidence": "TINGGI/SEDANG", "saham": ["BBCA", "BMRI"], "ringkasan": "..."}}]

Field confidence:
- TINGGI: berita jelas, dampak terukur, sumber kredibel, korelasi kuat ke saham
- SEDANG: ada ketidakpastian atau dampak tidak langsung

Berita:
{news_text}

Hanya kembalikan JSON, tidak ada teks lain."""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.content[0].text.strip()
        result = _extract_json(raw)
        if result is None:
            print(f"[Sentinel] Could not parse JSON from Haiku response: {raw[:200]}")
            return []
        return result
    except Exception as e:
        print(f"[Sentinel] Haiku error: {e}")
        return []


def format_news_alert(analyses: list, articles: list) -> str:
    """Format news alert for Telegram"""
    now_wib = datetime.utcnow() + timedelta(hours=7)
    lines = [
        f"📰 <b>NEWS ALERT — {now_wib.strftime('%H:%M')} WIB</b>",
        "",
    ]

    for a in analyses:
        if a.get("dampak") == "RENDAH":
            continue  # Skip low impact
        if a.get("confidence", "TINGGI") != "TINGGI":
            continue  # Only alert TINGGI confidence

        emoji = "🟢" if a["sentimen"] == "POSITIF" else "🔴" if a["sentimen"] == "NEGATIF" else "⚪"
        impact_emoji = "🔥" if a.get("dampak") == "TINGGI" else "📌"
        saham = ", ".join(a.get("saham", [])[:4])

        lines.append(f"{emoji}{impact_emoji} <b>{a.get('judul', '')[:60]}</b>")
        lines.append(f"   {a.get('ringkasan', '')}")
        if saham:
            lines.append(f"   📊 Dampak: <b>{saham}</b>")
        lines.append("")

    if len(lines) <= 3:
        return None  # No significant news

    lines.append("<i>⚠️ Konfirmasi sebelum trading.</i>")
    return "\n".join(lines)


def run_sentinel():
    """Main sentinel run — call every 30 minutes during market hours"""
    print("[Sentinel] Fetching news...")
    seen = load_seen()

    articles = fetch_news()
    new_articles = [a for a in articles if a["id"] not in seen]

    if not new_articles:
        print("[Sentinel] No new articles")
        return

    # Update seen
    seen.update(a["id"] for a in new_articles)
    save_seen(seen)

    # Filter articles with relevant keywords
    relevant = []
    for a in new_articles:
        tickers = get_affected_tickers(a["title"], a["summary"])
        if tickers:
            a["tickers"] = tickers
            relevant.append(a)

    if not relevant:
        print(f"[Sentinel] {len(new_articles)} new articles, none relevant to watchlist")
        return

    print(f"[Sentinel] {len(relevant)} relevant articles, analyzing...")
    analyses = analyze_with_haiku(relevant[:8])  # Max 8 per call

    if analyses:
        msg = format_news_alert(analyses, relevant)
        if msg:
            ok = send_telegram(msg)
            print(f"[Sentinel] Alert sent: {ok}")
        else:
            print("[Sentinel] No high/medium impact news")

    # --- Shared Intelligence: update market context ---
    try:
        from agents.market_context import update_sentiment
        _update_context_from_analyses(analyses, relevant)
    except Exception as e:
        print(f"[Sentinel] MarketContext sentiment update error: {e}")


# Sector mapping for market context (ticker → sector)
_TICKER_SECTOR_MAP = {
    "BBCA.JK": "Perbankan", "BBRI.JK": "Perbankan",
    "BMRI.JK": "Perbankan", "BBNI.JK": "Perbankan",
    "TLKM.JK": "Telco", "EXCL.JK": "Telco",
    "ANTM.JK": "Mining", "MDKA.JK": "Mining", "MEDC.JK": "Mining",
    "GOTO.JK": "Tech", "BUKA.JK": "Tech",
    "ASII.JK": "Industri", "AALI.JK": "Industri",
    "UNVR.JK": "Consumer", "ICBP.JK": "Consumer", "INDF.JK": "Consumer",
    "ADRO.JK": "Coal", "PTBA.JK": "Coal",
    "SMGR.JK": "Semen", "INTP.JK": "Semen",
    "AKRA.JK": "Oil", "ELSA.JK": "Oil",
    "INCO.JK": "Mining", "ITMG.JK": "Coal",
    "LSIP.JK": "Industri",
}


def _update_context_from_analyses(analyses: list, relevant: list):
    """Extract dominant sentiment and update market context."""
    from agents.market_context import update_sentiment

    if not analyses:
        # No analyses — keep neutral
        update_sentiment("NETRAL", "SEDANG", [], [], "Tidak ada berita relevan")
        return

    # Filter to high-confidence, non-low-impact
    significant = [
        a for a in analyses
        if a.get("dampak", a.get("level", "RENDAH")) != "RENDAH"
        and a.get("confidence", "SEDANG") == "TINGGI"
    ]

    if not significant:
        significant = analyses  # Fall back to all

    # Count sentiments
    counts = {"POSITIF": 0, "NEGATIF": 0, "NETRAL": 0}
    for a in significant:
        s = a.get("sentimen", "NETRAL")
        if s in counts:
            counts[s] += 1

    # Dominant sentiment
    dominant = max(counts, key=lambda k: counts[k])

    # Most impactful article (TINGGI priority, then SEDANG)
    def impact_order(a):
        lvl = a.get("dampak", a.get("level", "RENDAH"))
        return {"TINGGI": 0, "SEDANG": 1, "RENDAH": 2}.get(lvl, 2)

    top = sorted(significant, key=impact_order)[0] if significant else {}

    # Extract confidence
    confidence = top.get("confidence", "SEDANG")

    # Extract affected tickers from all significant articles
    affected_tickers = []
    for a in significant:
        for t in a.get("saham", []):
            ticker_jk = t + ".JK" if not t.endswith(".JK") else t
            if ticker_jk not in affected_tickers:
                affected_tickers.append(ticker_jk)

    # Map tickers to sectors
    affected_sectors = list({
        _TICKER_SECTOR_MAP.get(t, _TICKER_SECTOR_MAP.get(t + ".JK", ""))
        for t in affected_tickers
        if _TICKER_SECTOR_MAP.get(t, _TICKER_SECTOR_MAP.get(t + ".JK", ""))
    })

    summary = top.get("ringkasan", top.get("analisa", ""))

    update_sentiment(dominant, confidence, affected_sectors, affected_tickers, summary)


if __name__ == "__main__":
    run_sentinel()
