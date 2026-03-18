"""
Pre-market Briefing Agent — Global market context before IDX opens.
Runs at 07:00 WIB every weekday.

fetch_global_markets() -> dict
analyze_premarket_with_sonnet(global_data) -> str
format_premarket_briefing(global_data, analysis) -> str
run_premarket_briefing() -> bool
"""

import os

import anthropic
import requests
import yfinance as yf
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "5922770410")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Tickers to fetch
GLOBAL_TICKERS = {
    "S&P500":   "^GSPC",
    "Nasdaq":   "^IXIC",
    "Dow":      "^DJI",
    "Nikkei":   "^N225",
    "HangSeng": "^HSI",
    "USD_IDR":  "IDR=X",
    "Gold":     "GC=F",
    "Oil":      "CL=F",
}


def fetch_global_markets() -> dict:
    """
    Fetch global market data from yfinance (period="2d").

    Returns:
        {
            market_name: {
                "current": float,
                "change_pct": float,
                "direction": "UP"|"DOWN"|"FLAT"
            }
        }
        Skips markets that fail — never raises.
    """
    result = {}
    for name, symbol in GLOBAL_TICKERS.items():
        try:
            t = yf.Ticker(symbol)
            hist = t.history(period="2d")
            if hist is None or hist.empty:
                continue

            current = float(hist["Close"].iloc[-1])
            prev = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else current
            change_pct = ((current - prev) / prev) * 100 if prev != 0 else 0.0

            if change_pct > 0.1:
                direction = "UP"
            elif change_pct < -0.1:
                direction = "DOWN"
            else:
                direction = "FLAT"

            result[name] = {
                "current": round(current, 2),
                "change_pct": round(change_pct, 2),
                "direction": direction,
            }
        except Exception as e:
            print(f"[Premarket] Error fetching {name} ({symbol}): {e}")

    return result


def analyze_premarket_with_sonnet(global_data: dict) -> str:
    """
    Use Claude Sonnet to analyze global markets and predict IHSG open tone.

    Args:
        global_data: result from fetch_global_markets()

    Returns:
        3-4 sentence analysis in Indonesian.
        Returns fallback string if API unavailable.
    """
    if not ANTHROPIC_API_KEY:
        return "API key tidak tersedia untuk analisa AI."

    if not global_data:
        return "Data global tidak tersedia untuk analisa."

    try:
        lines = []
        for name, data in global_data.items():
            pct = data.get("change_pct", 0)
            curr = data.get("current", 0)
            display_name = name.replace("_", "/")
            lines.append(f"- {display_name}: {curr:,.2f} ({pct:+.2f}%)")
        market_text = "\n".join(lines)

        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=400,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Kamu Dexter, senior analis hedge fund Indonesia. "
                        f"Berikan analisa singkat (3-4 kalimat, bahasa Indonesia) "
                        f"berdasarkan data pasar global berikut:\n\n"
                        f"{market_text}\n\n"
                        f"Prediksi: tone pembukaan IHSG hari ini (POSITIF/NETRAL/NEGATIF) "
                        f"dan sektor apa yang berpotensi leading. "
                        f"Singkat, presisi, actionable."
                    ),
                }
            ],
        )
        return response.content[0].text.strip()

    except Exception as e:
        print(f"[Premarket] Claude analysis error: {e}")
        return "Analisa AI tidak tersedia saat ini."


def _direction_emoji(direction: str) -> str:
    if direction == "UP":
        return "🟢"
    elif direction == "DOWN":
        return "🔴"
    return "⚪"


def format_premarket_briefing(global_data: dict, analysis: str) -> str:
    """
    Format pre-market briefing message for Telegram.

    Args:
        global_data: result from fetch_global_markets()
        analysis: string from analyze_premarket_with_sonnet()

    Returns:
        HTML-formatted Telegram message string.
    """
    now_wib = datetime.utcnow() + timedelta(hours=7)
    date_str = now_wib.strftime("%d %b %Y")

    lines = [f"🌏 <b>PRE-MARKET BRIEFING — {date_str}</b>", ""]

    # Wall Street
    ws_names = ["S&P500", "Nasdaq", "Dow"]
    ws_data = {k: v for k, v in global_data.items() if k in ws_names}
    if ws_data:
        lines.append("🇺🇸 <b>Wall Street (kemarin):</b>")
        for name in ws_names:
            if name not in ws_data:
                continue
            d = ws_data[name]
            em = _direction_emoji(d.get("direction", "FLAT"))
            pct = d.get("change_pct", 0)
            curr = d.get("current", 0)
            lines.append(f"  {name:<8}: {curr:>10,.0f} ({pct:+.2f}%) {em}")
        lines.append("")

    # Asia
    asia_names = ["Nikkei", "HangSeng"]
    asia_data = {k: v for k, v in global_data.items() if k in asia_names}
    if asia_data:
        lines.append("🌏 <b>Asia pagi ini:</b>")
        for name in asia_names:
            if name not in asia_data:
                continue
            d = asia_data[name]
            em = _direction_emoji(d.get("direction", "FLAT"))
            pct = d.get("change_pct", 0)
            curr = d.get("current", 0)
            display = "Hang Seng" if name == "HangSeng" else name
            lines.append(f"  {display:<10}: {curr:>8,.0f} ({pct:+.2f}%) {em}")
        lines.append("")

    # Macro
    macro_names = ["USD_IDR", "Gold", "Oil"]
    macro_data = {k: v for k, v in global_data.items() if k in macro_names}
    if macro_data:
        lines.append("💵 <b>Macro:</b>")
        for name in macro_names:
            if name not in macro_data:
                continue
            d = macro_data[name]
            em = _direction_emoji(d.get("direction", "FLAT"))
            pct = d.get("change_pct", 0)
            curr = d.get("current", 0)
            if name == "USD_IDR":
                lines.append(f"  USD/IDR : {curr:>8,.0f} ({pct:+.2f}%) {em}")
            elif name == "Gold":
                lines.append(f"  Gold    : ${curr:>7,.0f} ({pct:+.2f}%) {em}")
            elif name == "Oil":
                lines.append(f"  Oil WTI : ${curr:>7.1f} ({pct:+.2f}%) {em}")
        lines.append("")

    # Analysis
    if analysis:
        lines.append("🧠 <b>Analisa Dexter:</b>")
        # Indent each sentence
        for sentence in analysis.split(". "):
            sentence = sentence.strip()
            if sentence:
                lines.append(f"  <i>{sentence}{'.' if not sentence.endswith('.') else ''}</i>")
        lines.append("")

    lines.append("⏰ Market buka 09:00 WIB")
    return "\n".join(lines)


def _infer_us_signal(global_data: dict) -> str:
    """Infer US market signal from Wall Street data."""
    ws_changes = [
        global_data.get("S&P500", {}).get("change_pct", 0),
        global_data.get("Nasdaq", {}).get("change_pct", 0),
        global_data.get("Dow", {}).get("change_pct", 0),
    ]
    # Filter out zeros (missing data)
    valid = [c for c in ws_changes if c != 0]
    avg = sum(valid) / len(valid) if valid else 0
    if avg > 0.3:
        return "BULLISH"
    elif avg < -0.3:
        return "BEARISH"
    return "NEUTRAL"


def _infer_asia_signal(global_data: dict) -> str:
    """Infer Asia market signal from Nikkei/HangSeng data."""
    asia_changes = [
        global_data.get("Nikkei", {}).get("change_pct", 0),
        global_data.get("HangSeng", {}).get("change_pct", 0),
    ]
    valid = [c for c in asia_changes if c != 0]
    avg = sum(valid) / len(valid) if valid else 0
    if avg > 0.3:
        return "BULLISH"
    elif avg < -0.3:
        return "BEARISH"
    return "NEUTRAL"


def _infer_ihsg_prediction(global_data: dict) -> str:
    """Infer IHSG open prediction from global signals."""
    us_signal = _infer_us_signal(global_data)
    asia_signal = _infer_asia_signal(global_data)

    # Both positive → POSITIF
    if us_signal == "BULLISH" and asia_signal in ("BULLISH", "NEUTRAL"):
        return "POSITIF"
    # Both negative → NEGATIF
    if us_signal == "BEARISH" and asia_signal in ("BEARISH", "NEUTRAL"):
        return "NEGATIF"
    # US strongly bullish with mixed Asia → POSITIF
    if us_signal == "BULLISH" and asia_signal == "BEARISH":
        return "NETRAL"
    return "NETRAL"


def run_premarket_briefing() -> bool:
    """
    Fetch global market data + analyze + send pre-market briefing to Telegram.

    Returns:
        True if Telegram message sent successfully.
    """
    try:
        print("[Premarket] Running pre-market briefing...")

        global_data = fetch_global_markets()
        analysis = analyze_premarket_with_sonnet(global_data)
        msg = format_premarket_briefing(global_data, analysis)

        # Update market context premarket section
        try:
            from agents.market_context import update_premarket
            us_signal = _infer_us_signal(global_data)
            asia_signal = _infer_asia_signal(global_data)
            prediction = _infer_ihsg_prediction(global_data)
            update_premarket(us_signal, asia_signal, prediction)
        except Exception as e:
            print(f"[Premarket] Context update error: {e}")

        # Send to Telegram
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        resp = requests.post(
            url,
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": msg,
                "parse_mode": "HTML",
            },
            timeout=15,
        )
        ok = resp.json().get("ok", False)
        print(f"[Premarket] Briefing sent: {ok}")
        return ok

    except Exception as e:
        print(f"[Premarket] run_premarket_briefing error: {e}")
        return False
