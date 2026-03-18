"""
Self-Improvement Agent
Tracks signal performance over time, generates weekly insights, logs errors.

- log_signal_result: append single result to performance_log.json
- analyze_performance: analyze win rates after 10+ trading days
- generate_improvement_report: weekly Friday 15:40 WIB report via Telegram
- log_error: append errors to error_log.json, alert on repeat errors (3x)
"""

import json
import os
import requests
import time
from collections import defaultdict
from datetime import datetime, timedelta

from dotenv import load_dotenv

load_dotenv()

PERFORMANCE_LOG_FILE = "data/performance_log.json"
ERROR_LOG_FILE = "data/error_log.json"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "5922770410")

# In-memory cache: error key → count since last reset
_error_count_cache: dict = {}


# ─── Telegram ───────────────────────────────────────────────────────────────

def _send_telegram(msg: str, retries: int = 3) -> bool:
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    for attempt in range(retries):
        try:
            resp = requests.post(url, json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": msg,
                "parse_mode": "HTML",
            }, timeout=10)
            data = resp.json()
            if data.get("ok"):
                return True
            print(f"[SelfImprover][Telegram ERROR] {data}")
        except Exception as e:
            print(f"[SelfImprover][Telegram] Attempt {attempt+1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(2)
    return False


# ─── Core Functions ─────────────────────────────────────────────────────────

def log_signal_result(date: str, ticker: str, entry: float, close: float, result: str):
    """
    Append a signal result to data/performance_log.json.

    Args:
        date: Trading date (YYYY-MM-DD)
        ticker: Stock ticker (e.g. "BBNI")
        entry: Entry price
        close: Close/exit price
        result: One of "HIT_TP", "HIT_CL", "NEUTRAL"
    """
    valid_results = {"HIT_TP", "HIT_CL", "NEUTRAL"}
    if result not in valid_results:
        raise ValueError(f"result must be one of {valid_results}, got: {result!r}")

    os.makedirs("data", exist_ok=True)

    log = []
    if os.path.exists(PERFORMANCE_LOG_FILE):
        try:
            with open(PERFORMANCE_LOG_FILE) as f:
                log = json.load(f)
        except (json.JSONDecodeError, IOError):
            log = []

    log.append({
        "date": date,
        "ticker": ticker,
        "entry": float(entry),
        "close": float(close),
        "result": result,
        "timestamp": datetime.utcnow().isoformat(),
    })

    with open(PERFORMANCE_LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)

    print(f"[SelfImprover] Logged {ticker} {result} for {date}")


def analyze_performance() -> dict:
    """
    Analyze performance log for insights.
    Requires 10+ trading days of data.

    Returns an insights dict:
    - status: "ok" | "insufficient_data" | "error"
    - overall_win_rate
    - per-condition win rates
    """
    if not os.path.exists(PERFORMANCE_LOG_FILE):
        return {"status": "error", "error": "No performance log found"}

    try:
        with open(PERFORMANCE_LOG_FILE) as f:
            log = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        return {"status": "error", "error": str(e)}

    if not log:
        return {"status": "error", "error": "Empty performance log"}

    # Count unique trading days
    days = set(entry.get("date", "") for entry in log if entry.get("date"))
    if len(days) < 10:
        return {
            "status": "insufficient_data",
            "days": len(days),
            "needed": 10,
            "total_signals": len(log),
        }

    total = len(log)
    hits = sum(1 for e in log if e.get("result") == "HIT_TP")
    losses = sum(1 for e in log if e.get("result") == "HIT_CL")
    neutrals = sum(1 for e in log if e.get("result") == "NEUTRAL")
    win_rate = round((hits / total) * 100, 1) if total > 0 else 0.0

    # Per-ticker analysis
    ticker_stats = defaultdict(lambda: {"hits": 0, "total": 0})
    for e in log:
        t = e.get("ticker", "UNKNOWN")
        ticker_stats[t]["total"] += 1
        if e.get("result") == "HIT_TP":
            ticker_stats[t]["hits"] += 1

    ticker_win_rates = {
        t: round((v["hits"] / v["total"]) * 100, 1)
        for t, v in ticker_stats.items()
        if v["total"] >= 3
    }

    return {
        "status": "ok",
        "total_signals": total,
        "trading_days": len(days),
        "hits": hits,
        "losses": losses,
        "neutrals": neutrals,
        "overall_win_rate": win_rate,
        "ticker_win_rates": ticker_win_rates,
    }


def generate_improvement_report() -> str:
    """
    Generate and send weekly performance report.
    Called every Friday 15:40 WIB.
    Returns the message string (or empty string if nothing to report).
    """
    os.makedirs("data", exist_ok=True)

    if not os.path.exists(PERFORMANCE_LOG_FILE):
        print("[SelfImprover] No performance log for improvement report")
        return ""

    try:
        with open(PERFORMANCE_LOG_FILE) as f:
            log = json.load(f)
    except (json.JSONDecodeError, IOError):
        return ""

    # Filter to last 7 days
    now_wib = datetime.utcnow() + timedelta(hours=7)
    week_ago = now_wib - timedelta(days=7)

    weekly = []
    for e in log:
        try:
            ts = datetime.fromisoformat(e.get("timestamp", "2000-01-01"))
            if ts >= week_ago:
                weekly.append(e)
        except (ValueError, TypeError):
            pass

    if not weekly:
        weekly = log  # Use all data if no recent weekly data

    total = len(weekly)
    if total == 0:
        print("[SelfImprover] No signals for improvement report")
        return ""

    hits = sum(1 for e in weekly if e.get("result") == "HIT_TP")
    losses = sum(1 for e in weekly if e.get("result") == "HIT_CL")
    neutrals = sum(1 for e in weekly if e.get("result") == "NEUTRAL")

    hit_pct = round((hits / total) * 100, 1) if total > 0 else 0.0
    loss_pct = round((losses / total) * 100, 1) if total > 0 else 0.0
    neutral_pct = round((neutrals / total) * 100, 1) if total > 0 else 0.0

    lines = [
        "📊 <b>WEEKLY PERFORMANCE REPORT</b>",
        "",
        f"Sinyal minggu ini: {total}",
        f"✅ Hit TP: {hits} ({hit_pct}%)",
        f"❌ Hit CL: {losses} ({loss_pct}%)",
        f"➖ Neutral: {neutrals} ({neutral_pct}%)",
        "",
        "🔍 <b>Insight:</b>",
    ]

    if total >= 5:
        lines.append(f"• Overall win rate: {hit_pct}%")

    # Ticker-level insights
    ticker_stats = defaultdict(lambda: {"hits": 0, "total": 0})
    for e in weekly:
        t = e.get("ticker", "UNKNOWN")
        ticker_stats[t]["total"] += 1
        if e.get("result") == "HIT_TP":
            ticker_stats[t]["hits"] += 1

    best_tickers = sorted(
        [(t, round(v["hits"] / v["total"] * 100, 1)) for t, v in ticker_stats.items() if v["total"] >= 2],
        key=lambda x: -x[1]
    )[:3]

    if best_tickers:
        lines.append("• Top performers: " + ", ".join(f"{t} ({wr}%)" for t, wr in best_tickers))

    lines.append("")
    lines.append("💡 <b>Auto-adjustment:</b>")

    if hit_pct >= 70:
        lines.append("• Sinyal akurat — pertahankan parameter saat ini")
    elif hit_pct >= 50:
        lines.append("• Win rate cukup — pertimbangkan RSI range lebih ketat (35→55)")
    else:
        lines.append("• Win rate rendah — review threshold sinyal minggu depan")

    lines.append("")
    lines.append("⚠️ <i>BUKAN saran investasi. DYOR.</i>")

    msg = "\n".join(lines)
    ok = _send_telegram(msg)
    print(f"[SelfImprover] Improvement report sent: {ok}")
    return msg


def log_error(agent: str, error: str, context: str = ""):
    """
    Append error to data/error_log.json.
    If same agent+error occurs 3+ times in the last 20 entries → send Telegram alert.

    Args:
        agent: Agent name (e.g. "Scanner", "Radar")
        error: Error message string
        context: Optional extra context
    """
    os.makedirs("data", exist_ok=True)

    error_log = []
    if os.path.exists(ERROR_LOG_FILE):
        try:
            with open(ERROR_LOG_FILE) as f:
                error_log = json.load(f)
        except (json.JSONDecodeError, IOError):
            error_log = []

    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "agent": str(agent),
        "error": str(error),
        "context": str(context),
    }
    error_log.append(entry)

    with open(ERROR_LOG_FILE, "w") as f:
        json.dump(error_log, f, indent=2)

    # Check for repeated errors in recent entries
    error_str = str(error)
    recent = [
        e for e in error_log[-20:]
        if e.get("agent") == str(agent) and e.get("error") == error_str
    ]

    if len(recent) >= 3:
        try:
            _send_telegram(
                f"⚠️ <b>ERROR BERULANG</b>\n"
                f"Agent <b>{agent}</b> mengalami error berulang\n"
                f"Error: <code>{error_str[:200]}</code>\n"
                f"Sudah {len(recent)}x terjadi"
            )
        except Exception:
            pass

    print(f"[SelfImprover] Error logged: {agent} — {error_str[:100]}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        generate_improvement_report()
    elif len(sys.argv) > 1 and sys.argv[1] == "analyze":
        result = analyze_performance()
        print(json.dumps(result, indent=2))
    else:
        print("Usage: python -m agents.self_improver [report|analyze]")
