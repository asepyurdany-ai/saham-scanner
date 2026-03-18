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
[{{"judul": "...", "sentimen": "...", "dampak": "...", "saham": ["BBCA", "BMRI"], "ringkasan": "..."}}]

Berita:
{news_text}

Hanya kembalikan JSON, tidak ada teks lain."""

    try:
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        result = json.loads(response.content[0].text)
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


if __name__ == "__main__":
    run_sentinel()
