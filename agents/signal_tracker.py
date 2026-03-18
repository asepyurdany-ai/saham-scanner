"""
Signal Accuracy Tracker
Tracks BUY signals and evaluates their accuracy over time.

- 08:45 WIB (after morning scan): log signals via log_signals_open()
- 15:35 WIB (after closing delta): update results via log_signals_close()
- Friday 15:40 WIB: send weekly accuracy report via send_weekly_accuracy_report()

Signal outcome rules (using daily H/L as proxy for intraday):
  HIT     : day's High >= TP (+8%)
  MISS    : day's Low  <= SL (-4%)
  NEUTRAL : neither
  If both triggered: HIT takes priority (optimistic, hedge fund tracks upside)
"""

import json
import os
import requests
import time
from datetime import datetime, timedelta

import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

SIGNAL_LOG_FILE = "data/signal_log.json"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "5922770410")


# ─── Helpers ────────────────────────────────────────────────────────────────

def _send_telegram(msg: str, retries: int = 3) -> bool:
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
            print(f"[SignalTracker][Telegram ERROR] {data}")
        except Exception as e:
            print(f"[SignalTracker][Telegram] Attempt {attempt+1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(2)
    return False


def _today_str() -> str:
    """Today's date in WIB (UTC+7) as YYYY-MM-DD string."""
    wib = datetime.utcnow() + timedelta(hours=7)
    return wib.strftime("%Y-%m-%d")


def load_signal_log() -> dict:
    """Load signal log from file. Returns empty dict if not found."""
    os.makedirs("data", exist_ok=True)
    if os.path.exists(SIGNAL_LOG_FILE):
        with open(SIGNAL_LOG_FILE) as f:
            return json.load(f)
    return {}


def save_signal_log(data: dict):
    """Persist signal log to file."""
    os.makedirs("data", exist_ok=True)
    with open(SIGNAL_LOG_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ─── Core Functions ─────────────────────────────────────────────────────────

def log_signals_open(signals: list):
    """
    Called at 08:45 WIB after morning scan.
    Logs all STRONG BUY signals with entry price to signal_log.json.
    Skips if today already logged (idempotent).
    """
    today = _today_str()
    log = load_signal_log()

    if today in log:
        print(f"[SignalTracker] Already logged for {today}, skipping")
        return

    buy_signals = [s for s in signals if s.get("signal") in ("STRONG BUY", "BUY")]

    entries = []
    for s in buy_signals:
        entry_price = float(s["current"])
        tp = round(entry_price * 1.08, 0)
        sl = round(entry_price * 0.96, 0)
        entries.append({
            "ticker": s["ticker"].replace(".JK", ""),
            "entry": entry_price,
            "score": s.get("score", 0),
            "tp": s.get("tp", tp),
            "sl": s.get("sl", sl),
            "result": "PENDING",
            "close_price": None,
            "high": None,
            "low": None,
        })

    log[today] = {
        "signals": entries,
        "summary": {
            "total": len(entries),
            "hit": 0,
            "miss": 0,
            "neutral": 0,
            "win_rate": 0.0,
        }
    }

    save_signal_log(log)
    print(f"[SignalTracker] Logged {len(entries)} signals for {today}")


def log_signals_close():
    """
    Called at 15:35 WIB after market close.
    Updates each signal with closing price and result (HIT/MISS/NEUTRAL).
    Uses intraday High/Low to determine if TP or SL was touched.
    """
    today = _today_str()
    log = load_signal_log()

    if today not in log:
        print(f"[SignalTracker] No signals logged for {today}")
        return

    day_data = log[today]
    signals = day_data["signals"]

    if not signals:
        print(f"[SignalTracker] No signals to evaluate for {today}")
        return

    hit = miss = neutral = 0

    for entry in signals:
        if entry["result"] != "PENDING":
            # Already evaluated (idempotent)
            if entry["result"] == "HIT":
                hit += 1
            elif entry["result"] == "MISS":
                miss += 1
            else:
                neutral += 1
            continue

        ticker_yf = entry["ticker"] + ".JK"
        try:
            t = yf.Ticker(ticker_yf)
            hist = t.history(period="2d")
            if hist.empty:
                entry["result"] = "NEUTRAL"
                neutral += 1
                continue

            today_row = hist.iloc[-1]
            high = float(today_row["High"])
            low = float(today_row["Low"])
            close_price = float(today_row["Close"])

            entry["close_price"] = round(close_price, 0)
            entry["high"] = round(high, 0)
            entry["low"] = round(low, 0)

            tp = float(entry["tp"])
            sl = float(entry["sl"])

            if high >= tp:
                entry["result"] = "HIT"
                hit += 1
            elif low <= sl:
                entry["result"] = "MISS"
                miss += 1
            else:
                entry["result"] = "NEUTRAL"
                neutral += 1

        except Exception as e:
            print(f"[SignalTracker] Error evaluating {ticker_yf}: {e}")
            entry["result"] = "NEUTRAL"
            neutral += 1

    total = len(signals)
    win_rate = round((hit / total) * 100, 1) if total > 0 else 0.0

    day_data["summary"] = {
        "total": total,
        "hit": hit,
        "miss": miss,
        "neutral": neutral,
        "win_rate": win_rate,
    }

    log[today] = day_data
    save_signal_log(log)
    print(f"[SignalTracker] Evaluated {today}: HIT={hit} MISS={miss} NEUTRAL={neutral} WR={win_rate}%")


def send_weekly_accuracy_report():
    """
    Called on Friday at 15:40 WIB.
    Sends weekly accuracy summary to Telegram.
    Covers Mon-Fri of the current week.
    """
    wib_now = datetime.utcnow() + timedelta(hours=7)
    log = load_signal_log()

    # Get Mon-Fri of current week
    today = wib_now.date()
    monday = today - timedelta(days=today.weekday())
    week_dates = [(monday + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(5)]

    total_signals = 0
    total_hit = 0
    total_miss = 0
    total_neutral = 0
    daily_lines = []

    for date_str in week_dates:
        if date_str not in log:
            continue
        day = log[date_str]
        s = day.get("summary", {})
        if s.get("total", 0) == 0:
            continue

        total_signals += s.get("total", 0)
        total_hit += s.get("hit", 0)
        total_miss += s.get("miss", 0)
        total_neutral += s.get("neutral", 0)

        wr = s.get("win_rate", 0.0)
        hits = s.get("hit", 0)
        misses = s.get("miss", 0)
        tickers = [sig["ticker"] for sig in day.get("signals", []) if sig.get("result") == "HIT"]
        hit_str = f" ({', '.join(tickers[:3])})" if tickers else ""

        daily_lines.append(
            f"  {date_str}: {s['total']} sinyal | "
            f"HIT {hits} MISS {misses} | WR {wr:.1f}%{hit_str}"
        )

    if total_signals == 0:
        print("[SignalTracker] No signals this week to report")
        return

    weekly_wr = round((total_hit / total_signals) * 100, 1) if total_signals > 0 else 0.0
    grade = "🏆" if weekly_wr >= 60 else "✅" if weekly_wr >= 40 else "⚠️" if weekly_wr >= 20 else "🔴"

    lines = [
        f"📈 <b>WEEKLY ACCURACY REPORT</b>",
        f"<i>Minggu {monday.strftime('%d %b')} – {today.strftime('%d %b %Y')}</i>",
        "",
        f"📊 Total Sinyal : {total_signals}",
        f"✅ HIT          : {total_hit}",
        f"❌ MISS         : {total_miss}",
        f"⚪ NEUTRAL      : {total_neutral}",
        f"🎯 Win Rate     : {weekly_wr:.1f}% {grade}",
        "",
        "<b>Detail Harian:</b>",
    ] + daily_lines + [
        "",
        "<i>Gunakan data ini untuk kalibrasi threshold sinyal minggu depan.</i>",
        "<i>⚠️ BUKAN saran investasi.</i>",
    ]

    msg = "\n".join(lines)
    ok = _send_telegram(msg)
    print(f"[SignalTracker] Weekly report sent: {ok}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "weekly":
            send_weekly_accuracy_report()
        elif sys.argv[1] == "close":
            log_signals_close()
    else:
        print("Usage: python -m agents.signal_tracker [weekly|close]")
