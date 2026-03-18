"""
Saham Scanner — Main Daemon
Schedule (WIB):
  08:45 - Morning scan (Scanner) with macro + sector context
  08:55-15:00 - Real-time monitoring every 10 min (Scanner + PositionTracker + SellScan)
  08:55-15:00 - Macro shock check every 5 min (Radar)
  09:00 - Market open commodity check (Radar)
  09:30, 10:30, 11:30, 13:00, 14:00 - Sentinel + Radar check
  15:35 - Closing delta report (Scanner) — new EOD format
  15:40 (Fridays) - Self-improvement weekly report

Self-healing: failed agents are auto-restarted on next run.
Error logging via SelfImprover.log_error on all agent failures.
"""

import time
import schedule
import threading
import json
import os
import requests
from datetime import datetime, timedelta
from agents.scanner import (
    run_morning_scan, run_closing_report, run_realtime_scan, run_sell_scan
)
from agents.sentinel import run_sentinel
from agents.radar import run_radar, run_macro_shock_check
from agents.market_breadth import run_breadth_check
from agents.premarket import run_premarket_briefing
from agents.position_tracker import (
    run_position_monitor, send_position_update, load_positions,
    add_position, add_position_idr, close_position, parse_buy_command,
    format_portfolio_summary, DEFAULT_POSITIONS_FILE
)
from agents.trade_journal import save_trade, send_journal_summary, format_journal_report
from agents.signal_tracker import log_signals_open, log_signals_close, send_weekly_accuracy_report
from agents.self_improver import generate_improvement_report, log_error
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "5922770410")
TELEGRAM_OFFSET_FILE = "data/telegram_offset.json"


def now_wib():
    return datetime.utcnow() + timedelta(hours=7)


IDX_HOLIDAYS_2026 = {
    "2026-01-01",  # Tahun Baru
    "2026-01-27",  # Isra Mi'raj
    "2026-01-29",  # Tahun Baru Imlek
    "2026-03-19",  # Nyepi
    "2026-03-20",  # Wafat Isa Almasih (Good Friday)
    "2026-04-01",  # Hari Raya Idul Fitri
    "2026-04-02",  # Hari Raya Idul Fitri
    "2026-04-03",  # Cuti Bersama
    "2026-05-01",  # Hari Buruh
    "2026-05-14",  # Kenaikan Isa Almasih
    "2026-05-23",  # Hari Raya Waisak
    "2026-06-01",  # Hari Lahir Pancasila
    "2026-06-06",  # Idul Adha
    "2026-06-26",  # Tahun Baru Islam
    "2026-08-17",  # HUT RI
    "2026-09-04",  # Maulid Nabi
    "2026-12-25",  # Natal
}


def is_market_day() -> bool:
    """Returns True if IDX is open today (weekday + not a public holiday)."""
    wib = now_wib()
    if wib.weekday() >= 5:  # weekend
        return False
    date_str = wib.strftime("%Y-%m-%d")
    if date_str in IDX_HOLIDAYS_2026:
        print(f"[Scheduler] IDX holiday today ({date_str}) — skipping market tasks")
        return False
    return True


def is_market_hours_utc() -> bool:
    """
    Check if current UTC time is within IDX market hours.
    IDX: 08:55-15:00 WIB = 01:55-08:00 UTC
    """
    now = datetime.utcnow()
    start = now.replace(hour=1, minute=55, second=0, microsecond=0)
    end = now.replace(hour=8, minute=0, second=0, microsecond=0)
    return start <= now <= end


# Track agent health
_agent_failures = {}
MAX_CONSECUTIVE_FAILURES = 3


def safe_run(fn, name: str):
    """Run an agent function safely with error handling and self-healing."""
    if not is_market_day():
        print(f"[Main] Weekend — skipping {name}")
        return

    # Check if agent is in cooldown (too many consecutive failures)
    failures = _agent_failures.get(name, 0)
    if failures >= MAX_CONSECUTIVE_FAILURES:
        print(f"[Main] {name} in cooldown ({failures} failures). Resetting...")
        _agent_failures[name] = 0  # Reset and retry

    try:
        print(f"[Main] Running {name} at {now_wib().strftime('%H:%M')} WIB")
        fn()
        _agent_failures[name] = 0  # Reset on success
    except Exception as e:
        _agent_failures[name] = _agent_failures.get(name, 0) + 1
        error_msg = str(e)
        print(f"[Main] ERROR in {name} (fail #{_agent_failures[name]}): {error_msg}")
        try:
            log_error(name, error_msg, context=f"safe_run at {now_wib().strftime('%H:%M')} WIB")
        except Exception:
            pass


def _is_context_stale(max_age_minutes: int = 30) -> bool:
    """Check if market context is older than max_age_minutes."""
    try:
        from agents.market_context import load_context
        ctx = load_context()
        updated_at = ctx.get("updated_at")
        if not updated_at:
            return True
        from datetime import timezone
        dt = datetime.fromisoformat(updated_at)
        age = (datetime.utcnow() - dt).total_seconds() / 60
        return age > max_age_minutes
    except Exception:
        return True


def safe_run_realtime():
    """Real-time monitoring job (every 10 min) — only runs during market hours."""
    if not is_market_day():
        return
    if not is_market_hours_utc():
        print(f"[Main] Outside market hours — skipping real-time scan")
        return

    # Refresh context if stale (>30 min) — keep scanner well-informed
    if _is_context_stale(30):
        print("[Main] Context stale >30min — refreshing radar + sentinel before realtime scan")
        safe_run(run_radar, "Radar (Context Refresh)")
        safe_run(run_sentinel, "Sentinel (Context Refresh)")

    safe_run(run_realtime_scan, "Real-time Scan")
    safe_run(run_position_monitor, "Position Monitor")
    safe_run(run_sell_scan, "Sell Signal Scan")


def safe_run_macro_shock():
    """Macro shock check (every 5 min) — only during market hours."""
    if not is_market_day():
        return
    if not is_market_hours_utc():
        return

    safe_run(run_macro_shock_check, "Macro Shock Check")


# --- Scheduled Jobs ---

def run_morning_sequence():
    """
    CRITICAL ORDER for Shared Intelligence System:
    1. run_radar()    → updates macro + geo context FIRST
    2. run_sentinel() → updates sentiment context SECOND
    3. run_morning_scan() → reads full context, uses dynamic threshold THIRD
    """
    print(f"[Main] Starting morning sequence at {now_wib().strftime('%H:%M')} WIB")

    # Step 1: Radar — macro + geo context
    try:
        print("[Main] Step 1/3: Radar (macro + geo context)...")
        run_radar()
    except Exception as e:
        log_error("Morning Sequence Radar", str(e), "run_morning_sequence")
        print(f"[Main] Radar error in morning sequence: {e}")

    # Step 2: Sentinel — news sentiment context
    try:
        print("[Main] Step 2/3: Sentinel (sentiment context)...")
        run_sentinel()
    except Exception as e:
        log_error("Morning Sequence Sentinel", str(e), "run_morning_sequence")
        print(f"[Main] Sentinel error in morning sequence: {e}")

    # Step 3: Scanner — reads full context, dynamic threshold
    run_morning_scan_with_tracker()


def run_morning_scan_with_tracker():
    """Morning scan + log signals for accuracy tracking."""
    try:
        signals = run_morning_scan()
    except Exception as e:
        log_error("Morning Scan", str(e), "run_morning_scan_with_tracker")
        signals = []
    try:
        log_signals_open(signals)
    except Exception as e:
        log_error("SignalTracker", str(e), "log_signals_open")
        print(f"[Main] SignalTracker open log error: {e}")


def run_closing_report_with_tracker():
    """Closing delta + evaluate signal accuracy."""
    try:
        run_closing_report()
    except Exception as e:
        log_error("Closing Report", str(e), "run_closing_report")
    try:
        log_signals_close()
    except Exception as e:
        log_error("SignalTracker", str(e), "log_signals_close")
        print(f"[Main] SignalTracker close log error: {e}")


def run_weekly_accuracy_report():
    """Only runs on Fridays."""
    if now_wib().weekday() != 4:  # 4 = Friday
        return
    try:
        send_weekly_accuracy_report()
    except Exception as e:
        log_error("WeeklyAccuracy", str(e), "send_weekly_accuracy_report")
        print(f"[Main] Weekly accuracy report error: {e}")


def run_weekly_improvement_report():
    """Self-improvement report — only runs on Fridays."""
    if now_wib().weekday() != 4:  # 4 = Friday
        return
    try:
        generate_improvement_report()
    except Exception as e:
        log_error("SelfImprover", str(e), "generate_improvement_report")
        print(f"[Main] Improvement report error: {e}")


def safe_run_position_update():
    """Send 30-min position P&L update during market hours (only if positions exist)."""
    if not is_market_day():
        return
    if not is_market_hours_utc():
        return
    try:
        positions = load_positions(DEFAULT_POSITIONS_FILE)
        if positions:
            send_position_update(positions)
    except Exception as e:
        log_error("PositionUpdate", str(e), "safe_run_position_update")
        print(f"[Main] Position update error: {e}")


def safe_run_breadth_check():
    """Run market breadth check every 30 min during market hours."""
    if not is_market_day():
        return
    if not is_market_hours_utc():
        return
    safe_run(run_breadth_check, "Market Breadth Check")


def safe_run_premarket():
    """Run pre-market briefing at 07:00 WIB on weekdays — including holidays (global markets still open)."""
    if now_wib().weekday() >= 5:  # skip weekend only
        return
    safe_run(run_premarket_briefing, "Pre-market Briefing")


def run_friday_journal():
    """Send journal summary on Fridays EOD."""
    if now_wib().weekday() != 4:  # 4 = Friday
        return
    try:
        send_journal_summary()
    except Exception as e:
        log_error("JournalSummary", str(e), "run_friday_journal")
        print(f"[Main] Journal summary error: {e}")


# ─── Macro Shock: every 5 min during market hours ──────────────────────────
# 01:55–08:00 UTC = 08:55–15:00 WIB, every 5 minutes
MACRO_SHOCK_SLOTS = [
    "01:55", "02:00", "02:05", "02:10", "02:15", "02:20", "02:25", "02:30",
    "02:35", "02:40", "02:45", "02:50", "02:55", "03:00", "03:05", "03:10",
    "03:15", "03:20", "03:25", "03:30", "03:35", "03:40", "03:45", "03:50",
    "03:55", "04:00", "04:05", "04:10", "04:15", "04:20", "04:25", "04:30",
    "04:35", "04:40", "04:45", "04:50", "04:55", "05:00", "05:05", "05:10",
    "05:15", "05:20", "05:25", "05:30", "05:35", "05:40", "05:45", "05:50",
    "05:55", "06:00", "06:05", "06:10", "06:15", "06:20", "06:25", "06:30",
    "06:35", "06:40", "06:45", "06:50", "06:55", "07:00", "07:05", "07:10",
    "07:15", "07:20", "07:25", "07:30", "07:35", "07:40", "07:45", "07:50",
    "07:55", "08:00",
]
for slot in MACRO_SHOCK_SLOTS:
    schedule.every().day.at(slot).do(safe_run_macro_shock)

# ─── Morning sequence (UTC 01:45 = WIB 08:45) ──────────────────────────────
# CRITICAL: Radar → Sentinel → Scanner (in order, so Scanner gets full context)
schedule.every().day.at("01:45").do(safe_run, run_morning_sequence, "Morning Sequence (Radar→Sentinel→Scan)")

# ─── Market open commodity check (UTC 02:00 = WIB 09:00) ───────────────────
schedule.every().day.at("02:00").do(safe_run, run_radar, "Radar (Market Open)")

# ─── Real-time + Sell scan: every 10 min during market hours ───────────────
REALTIME_SLOTS = [
    "01:55", "02:05", "02:15", "02:25", "02:35", "02:45", "02:55",
    "03:05", "03:15", "03:25", "03:35", "03:45", "03:55",
    "04:05", "04:15", "04:25", "04:35", "04:45", "04:55",
    "05:05", "05:15", "05:25", "05:35", "05:45", "05:55",
    "06:05", "06:15", "06:25", "06:35", "06:45", "06:55",
    "07:05", "07:15", "07:25", "07:35", "07:45", "07:55",
]
for slot in REALTIME_SLOTS:
    schedule.every().day.at(slot).do(safe_run_realtime)

# ─── Sentinel + Radar checks ───────────────────────────────────────────────
schedule.every().day.at("02:30").do(safe_run, run_sentinel, "Sentinel")
schedule.every().day.at("03:30").do(safe_run, run_sentinel, "Sentinel")
schedule.every().day.at("03:30").do(safe_run, run_radar, "Radar")
schedule.every().day.at("04:30").do(safe_run, run_sentinel, "Sentinel")
schedule.every().day.at("06:00").do(safe_run, run_sentinel, "Sentinel")
schedule.every().day.at("06:00").do(safe_run, run_radar, "Radar")
schedule.every().day.at("07:00").do(safe_run, run_sentinel, "Sentinel")
schedule.every().day.at("07:00").do(safe_run, run_radar, "Radar")

# ─── EOD: Closing delta report + signal accuracy (UTC 08:35 = WIB 15:35) ───
schedule.every().day.at("08:35").do(safe_run, run_closing_report_with_tracker, "Closing Delta Report + Tracker")

# ─── Friday: Weekly accuracy + improvement report (UTC 08:40 = WIB 15:40) ──
schedule.every().day.at("08:40").do(safe_run, run_weekly_accuracy_report, "Weekly Accuracy Report")
schedule.every().day.at("08:40").do(safe_run, run_weekly_improvement_report, "Weekly Improvement Report")

# ─── Friday EOD: Journal summary (UTC 08:40 = WIB 15:40) ───────────────────
schedule.every().day.at("08:40").do(safe_run, run_friday_journal, "Friday Journal Summary")

# ─── Position P&L update every 30 min during market hours ──────────────────
POSITION_UPDATE_SLOTS = [
    "01:55", "02:25", "02:55", "03:25", "03:55",
    "04:25", "04:55", "05:25", "05:55",
    "06:25", "06:55", "07:25", "07:55",
]
for slot in POSITION_UPDATE_SLOTS:
    schedule.every().day.at(slot).do(safe_run_position_update)

# ─── Market breadth check every 30 min during market hours ─────────────────
# (same slots as position update — piggyback on market hours window)
BREADTH_CHECK_SLOTS = [
    "01:55", "02:25", "02:55", "03:25", "03:55",
    "04:25", "04:55", "05:25", "05:55",
    "06:25", "06:55", "07:25", "07:55",
]
for slot in BREADTH_CHECK_SLOTS:
    schedule.every().day.at(slot).do(safe_run_breadth_check)

# ─── Pre-market briefing at 07:00 WIB = 00:00 UTC (weekdays only) ──────────
schedule.every().day.at("00:00").do(safe_run_premarket)


def _send_telegram(msg: str):
    """Send a Telegram message from main."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        resp = requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": msg,
            "parse_mode": "HTML"
        }, timeout=10)
        return resp.json().get("ok", False)
    except Exception as e:
        print(f"[Main] Telegram send error: {e}")
        return False


def _load_telegram_offset() -> int:
    """Load last Telegram update offset."""
    os.makedirs("data", exist_ok=True)
    if os.path.exists(TELEGRAM_OFFSET_FILE):
        try:
            with open(TELEGRAM_OFFSET_FILE) as f:
                return json.load(f).get("offset", 0)
        except Exception:
            pass
    return 0


def _save_telegram_offset(offset: int):
    """Save Telegram update offset."""
    os.makedirs("data", exist_ok=True)
    with open(TELEGRAM_OFFSET_FILE, "w") as f:
        json.dump({"offset": offset}, f)


def _handle_telegram_command(text: str, chat_id: str):
    """Handle a single Telegram command. Returns reply message."""
    text = text.strip()
    lower = text.lower()

    try:
        # /beli TICKER PRICE IDR_OR_LOTS
        if lower.startswith("/beli"):
            parsed = parse_buy_command(text)
            if not parsed:
                return "❌ Format: /beli TICKER HARGA LOT\nContoh: /beli BBCA 9000 5 atau /beli BBCA 9000 3000000"
            ticker = parsed["ticker"]
            price = parsed["price"]
            if "total_idr" in parsed:
                pos = add_position_idr(ticker, price, parsed["total_idr"])
                lots = pos["lots"]
                actual_cost = pos.get("actual_cost", price * lots * 100)
                tp = pos["tp_price"]
                cl = pos["cl_price"]
                return (
                    f"✅ <b>Posisi IDR ditambah: {ticker}</b>\n\n"
                    f"📌 Entry: Rp {price:,.0f}\n"
                    f"💰 Modal: Rp {parsed['total_idr']:,.0f} → {lots} lot (Rp {actual_cost:,.0f})\n"
                    f"📦 {lots * 100:,} lembar\n"
                    f"🎯 TP: Rp {tp:,.0f} (+8%)\n"
                    f"🛑 CL: Rp {cl:,.0f} (-4%)\n\n"
                    f"<i>Monitor otomatis setiap 10 menit jam pasar.</i>"
                )
            else:
                lots = parsed["lots"]
                from agents.position_tracker import add_position
                pos = add_position(ticker, price, lots)
                tp = pos["tp_price"]
                cl = pos["cl_price"]
                total_value = price * lots * 100
                return (
                    f"✅ <b>Posisi ditambah: {ticker}</b>\n\n"
                    f"📌 Entry: Rp {price:,.0f}\n"
                    f"📦 {lots} lot ({lots * 100:,} saham) = Rp {total_value:,.0f}\n"
                    f"🎯 TP: Rp {tp:,.0f} (+8%)\n"
                    f"🛑 CL: Rp {cl:,.0f} (-4%)\n\n"
                    f"<i>Monitor otomatis setiap 10 menit jam pasar.</i>"
                )

        # /jual TICKER PRICE
        elif lower.startswith("/jual"):
            parts = text.split()
            if len(parts) < 3:
                return "❌ Format: /jual TICKER HARGA\nContoh: /jual BBCA 7010"
            ticker = parts[1].upper()
            try:
                sell_price = float(parts[2])
            except ValueError:
                return "❌ Harga tidak valid."
            closed = close_position(ticker, sell_price)
            if closed is None:
                return f"❌ Posisi {ticker} tidak ditemukan. Cek /posisi"
            # Save to journal
            save_trade(closed)
            return f"✅ Trade {ticker} disimpan ke journal."

        # /posisi
        elif lower.startswith("/posisi"):
            positions = load_positions(DEFAULT_POSITIONS_FILE)
            return format_portfolio_summary(positions)

        # /journal
        elif lower.startswith("/journal"):
            return format_journal_report()

        # /help
        elif lower.startswith("/help"):
            return (
                "🤖 <b>Dexter Commands</b>\n\n"
                "/beli TICKER HARGA LOT — tambah posisi (lots)\n"
                "/beli TICKER HARGA MODAL — tambah posisi (IDR)\n"
                "/jual TICKER HARGA — tutup posisi & catat ke journal\n"
                "/posisi — lihat portfolio terkini\n"
                "/journal — lihat trading journal & statistik\n"
                "/help — tampilkan perintah ini\n\n"
                "<i>Contoh: /beli BBCA 6070 3000000</i>"
            )

    except Exception as e:
        print(f"[Main] Command error: {e}")
        log_error("TelegramCommand", str(e), text)
        return f"⚠️ Error memproses perintah: {e}"

    return None  # not a recognized command


def check_telegram_commands():
    """
    Long-poll Telegram getUpdates and process commands.
    Called in a background thread.
    """
    offset = _load_telegram_offset()

    while True:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
            params = {"offset": offset, "timeout": 30, "allowed_updates": ["message"]}
            resp = requests.get(url, params=params, timeout=40)
            data = resp.json()

            if not data.get("ok"):
                time.sleep(5)
                continue

            updates = data.get("result", [])
            for update in updates:
                update_id = update["update_id"]
                offset = update_id + 1
                _save_telegram_offset(offset)

                msg = update.get("message", {})
                text = msg.get("text", "")
                chat_id = str(msg.get("chat", {}).get("id", ""))

                if not text or not text.startswith("/"):
                    continue

                print(f"[Main] Telegram command: {text} from {chat_id}")
                reply = _handle_telegram_command(text, chat_id)
                if reply:
                    _send_telegram(reply)

        except requests.exceptions.Timeout:
            # Normal for long-polling
            pass
        except Exception as e:
            print(f"[Main] Telegram poll error: {e}")
            time.sleep(10)


def run_daemon():
    print(f"[Saham Scanner] Daemon started at {now_wib().strftime('%Y-%m-%d %H:%M')} WIB")
    print("[Saham Scanner] Schedule:")
    print("  07:00 WIB - Pre-market briefing (Wall Street + Asia + Macro + AI analysis)")
    print("  08:45 WIB - Morning Sequence: Radar → Sentinel → Scanner (full context)")
    print("  08:45 WIB - Shared Intelligence: agents saling connected via market context")
    print("  08:55-15:00 WIB - Macro shock check every 5 min")
    print("  08:55-15:00 WIB - Real-time every 10 min (Scanner + PositionTracker + SellScan)")
    print("  08:55-15:00 WIB - Position P&L update every 30 min")
    print("  08:55-15:00 WIB - Market breadth gate check every 30 min")
    print("  09:00+ WIB - Sentinel & Radar checks")
    print("  15:35 WIB - Closing delta report (EOD format)")
    print("  15:40 WIB (Fri) - Weekly accuracy + improvement report + journal")
    print("[Saham Scanner] Self-healing: agents auto-restart on failure")
    print("[Saham Scanner] Error logging: via SelfImprover.log_error")
    # NOTE: Telegram command handler DISABLED in daemon.
    # Commands (/beli, /jual, /posisi, /journal) are handled by OpenClaw (Dexter).
    # Polling getUpdates here causes 409 conflict with OpenClaw gateway.
    print("[Saham Scanner] Telegram commands: handled by OpenClaw (not daemon)")

    while True:
        try:
            schedule.run_pending()
        except Exception as e:
            print(f"[Main] Scheduler error (will continue): {e}")
        time.sleep(30)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "scan":
            run_morning_scan_with_tracker()
        elif cmd == "close":
            run_closing_report_with_tracker()
        elif cmd == "sentinel":
            run_sentinel()
        elif cmd == "radar":
            run_radar()
        elif cmd == "realtime":
            run_realtime_scan()
        elif cmd == "sell":
            run_sell_scan()
        elif cmd == "macro":
            run_macro_shock_check()
        elif cmd == "positions":
            run_position_monitor()
        elif cmd == "weekly":
            send_weekly_accuracy_report()
        elif cmd == "improve":
            generate_improvement_report()
        elif cmd == "daemon":
            run_daemon()
        elif cmd == "journal":
            send_journal_summary()
        elif cmd == "positions_update":
            positions = load_positions(DEFAULT_POSITIONS_FILE)
            send_position_update(positions)
        elif cmd == "premarket":
            run_premarket_briefing()
        elif cmd == "breadth":
            run_breadth_check()
        else:
            print("Usage: python main.py [scan|close|sentinel|radar|realtime|sell|macro|positions|positions_update|weekly|improve|journal|premarket|breadth|daemon]")
    else:
        run_daemon()
