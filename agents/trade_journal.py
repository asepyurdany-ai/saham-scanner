"""
Trade Journal Agent — Track completed trades and statistics.
Saves trades to data/trade_journal.json and generates reports.
"""

import json
import os
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "5922770410")

JOURNAL_FILE = "data/trade_journal.json"


def send_telegram(msg: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        resp = requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": msg,
            "parse_mode": "HTML"
        }, timeout=10)
        data = resp.json()
        if not data.get("ok"):
            print(f"[TradeJournal][Telegram ERROR] {data}")
        return data.get("ok", False)
    except Exception as e:
        print(f"[TradeJournal][Telegram EXCEPTION] {e}")
        return False


def load_journal(journal_file: str = JOURNAL_FILE) -> list:
    """Load trade journal from file. Returns list of trades."""
    os.makedirs(os.path.dirname(journal_file) if os.path.dirname(journal_file) else "data", exist_ok=True)
    if os.path.exists(journal_file):
        with open(journal_file) as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            # Support dict wrapper
            return data.get("trades", [])
    return []


def save_journal(trades: list, journal_file: str = JOURNAL_FILE):
    """Save trade journal to file."""
    os.makedirs(os.path.dirname(journal_file) if os.path.dirname(journal_file) else "data", exist_ok=True)
    with open(journal_file, "w") as f:
        json.dump(trades, f, indent=2)


def _generate_trade_id(ticker: str, date_str: str, trades: list) -> str:
    """Generate a unique trade ID like BBCA-20260318-001."""
    prefix = f"{ticker}-{date_str}-"
    count = sum(1 for t in trades if t.get("id", "").startswith(prefix)) + 1
    return f"{prefix}{count:03d}"


def save_trade(trade_dict: dict, journal_file: str = JOURNAL_FILE) -> dict:
    """
    Append a completed trade to trade_journal.json.
    Generates trade ID if not present.
    Returns the final trade dict.
    """
    trades = load_journal(journal_file)

    # Generate ID if missing
    if "id" not in trade_dict or not trade_dict["id"]:
        ticker = trade_dict.get("ticker", "UNK")
        entry_time = trade_dict.get("entry_time", datetime.utcnow().isoformat())
        try:
            dt = datetime.fromisoformat(entry_time)
        except Exception:
            dt = datetime.utcnow()
        date_str = dt.strftime("%Y%m%d")
        trade_dict["id"] = _generate_trade_id(ticker, date_str, trades)

    # Determine WIN/LOSS/NEUTRAL if not set
    if "result" not in trade_dict:
        profit_pct = trade_dict.get("profit_pct", 0)
        if profit_pct > 0:
            trade_dict["result"] = "WIN"
        elif profit_pct < 0:
            trade_dict["result"] = "LOSS"
        else:
            trade_dict["result"] = "NEUTRAL"

    trades.append(trade_dict)
    save_journal(trades, journal_file)
    print(f"[TradeJournal] Saved trade: {trade_dict['id']} — {trade_dict.get('result', '?')}")
    return trade_dict


def get_journal_stats(journal_file: str = JOURNAL_FILE) -> dict:
    """
    Calculate trading statistics from the journal.
    Returns dict with stats.
    """
    trades = load_journal(journal_file)

    if not trades:
        return {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "neutrals": 0,
            "win_rate_pct": 0.0,
            "total_profit_rp": 0,
            "best_trade": None,
            "worst_trade": None,
            "signal_win_rate": 0.0,
            "non_signal_win_rate": 0.0,
            "avg_hold_minutes": 0,
        }

    total = len(trades)
    wins = [t for t in trades if t.get("result") == "WIN"]
    losses = [t for t in trades if t.get("result") == "LOSS"]
    neutrals = [t for t in trades if t.get("result") == "NEUTRAL"]

    win_rate_pct = round(len(wins) / total * 100, 1) if total > 0 else 0.0
    total_profit_rp = sum(t.get("profit_rp", 0) for t in trades)

    # Best / worst
    sorted_by_pct = sorted(trades, key=lambda t: t.get("profit_pct", 0))
    best = sorted_by_pct[-1] if sorted_by_pct else None
    worst = sorted_by_pct[0] if sorted_by_pct else None

    best_trade = {"ticker": best["ticker"], "profit_pct": best.get("profit_pct", 0)} if best else None
    worst_trade = {"ticker": worst["ticker"], "profit_pct": worst.get("profit_pct", 0)} if worst else None

    # Signal vs non-signal win rates
    signal_trades = [t for t in trades if t.get("followed_signal")]
    non_signal_trades = [t for t in trades if not t.get("followed_signal")]

    signal_wins = [t for t in signal_trades if t.get("result") == "WIN"]
    non_signal_wins = [t for t in non_signal_trades if t.get("result") == "WIN"]

    signal_win_rate = round(len(signal_wins) / len(signal_trades) * 100, 1) if signal_trades else 0.0
    non_signal_win_rate = round(len(non_signal_wins) / len(non_signal_trades) * 100, 1) if non_signal_trades else 0.0

    # Average hold time
    hold_minutes_list = [t.get("hold_minutes", 0) for t in trades if t.get("hold_minutes") is not None]
    avg_hold_minutes = round(sum(hold_minutes_list) / len(hold_minutes_list)) if hold_minutes_list else 0

    return {
        "total_trades": total,
        "wins": len(wins),
        "losses": len(losses),
        "neutrals": len(neutrals),
        "win_rate_pct": win_rate_pct,
        "total_profit_rp": int(total_profit_rp),
        "best_trade": best_trade,
        "worst_trade": worst_trade,
        "signal_win_rate": signal_win_rate,
        "non_signal_win_rate": non_signal_win_rate,
        "signal_trades_count": len(signal_trades),
        "non_signal_trades_count": len(non_signal_trades),
        "signal_wins_count": len(signal_wins),
        "non_signal_wins_count": len(non_signal_wins),
        "avg_hold_minutes": avg_hold_minutes,
    }


def _format_hold_time(minutes: int) -> str:
    """Format hold time as 'Xj Ym'"""
    if minutes < 60:
        return f"{minutes}m"
    h = minutes // 60
    m = minutes % 60
    if m == 0:
        return f"{h}j"
    return f"{h}j {m}m"


def format_journal_report(journal_file: str = JOURNAL_FILE) -> str:
    """Format trading journal as a Telegram message."""
    stats = get_journal_stats(journal_file)

    if stats["total_trades"] == 0:
        return (
            "📔 <b>TRADING JOURNAL — Asep</b>\n\n"
            "Belum ada trade yang tercatat.\n"
            "Gunakan /beli dan /jual untuk mulai tracking."
        )

    total = stats["total_trades"]
    win_rate = stats["win_rate_pct"]
    total_profit = stats["total_profit_rp"]
    profit_sign = "+" if total_profit >= 0 else ""

    best = stats.get("best_trade")
    worst = stats.get("worst_trade")

    signal_win_rate = stats["signal_win_rate"]
    signal_wins = stats["signal_wins_count"]
    signal_total = stats["signal_trades_count"]

    non_signal_win_rate = stats["non_signal_win_rate"]
    non_signal_wins = stats["non_signal_wins_count"]
    non_signal_total = stats["non_signal_trades_count"]

    avg_hold = _format_hold_time(stats["avg_hold_minutes"])

    lines = [
        "📔 <b>TRADING JOURNAL — Asep</b>",
        "",
        f"Total trades : {total}",
        f"Win rate     : {win_rate}%",
        f"Total profit : {profit_sign}Rp {abs(total_profit):,.0f}",
        "",
    ]

    if best:
        best_sign = "+" if best["profit_pct"] >= 0 else ""
        lines.append(f"📈 Best  : {best['ticker']} {best_sign}{best['profit_pct']:.1f}%")
    if worst:
        worst_sign = "+" if worst["profit_pct"] >= 0 else ""
        lines.append(f"📉 Worst : {worst['ticker']} {worst_sign}{worst['profit_pct']:.1f}%")

    lines.append("")

    if signal_total > 0:
        lines.append(f"🎯 Follow sinyal Dexter: {signal_win_rate}% win rate ({signal_wins}/{signal_total})")
    if non_signal_total > 0:
        lines.append(f"🎲 Di luar sinyal      : {non_signal_win_rate}% win rate ({non_signal_wins}/{non_signal_total})")

    lines.append("")
    lines.append(f"⏱ Avg hold: {avg_hold}")

    return "\n".join(lines)


def send_journal_summary(journal_file: str = JOURNAL_FILE) -> bool:
    """Send journal summary to Telegram. Returns True if successful."""
    msg = format_journal_report(journal_file)
    return send_telegram(msg)
