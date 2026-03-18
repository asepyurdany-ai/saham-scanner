"""
Saham Scanner — Main Daemon
Schedule (WIB):
  08:45 - Morning scan (Scanner) with macro + sector context
  08:55-15:00 - Real-time monitoring every 10 min (Scanner + PositionTracker)
  09:00 - Market open commodity check (Radar)
  09:30, 10:30, 11:30, 13:00, 14:00 - Sentinel + Radar check
  15:35 - Closing delta report (Scanner)
  Weekend/holiday: skip

Self-healing: failed agents are auto-restarted on next run.
"""

import time
import schedule
import threading
from datetime import datetime, timedelta
from agents.scanner import run_morning_scan, run_closing_report, run_realtime_scan
from agents.sentinel import run_sentinel
from agents.radar import run_radar
from agents.position_tracker import run_position_monitor
from agents.signal_tracker import log_signals_open, log_signals_close, send_weekly_accuracy_report


def now_wib():
    return datetime.utcnow() + timedelta(hours=7)


def is_market_day() -> bool:
    """Skip weekends. TODO: add IDX holiday calendar"""
    wib = now_wib()
    return wib.weekday() < 5  # Mon-Fri


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
        print(f"[Main] ERROR in {name} (fail #{_agent_failures[name]}): {e}")
        # Self-healing: log and continue, will retry next scheduled run


def safe_run_realtime():
    """Real-time monitoring job — only runs during market hours."""
    if not is_market_day():
        return
    if not is_market_hours_utc():
        print(f"[Main] Outside market hours — skipping real-time scan")
        return

    safe_run(run_realtime_scan, "Real-time Scan")
    safe_run(run_position_monitor, "Position Monitor")


# --- Scheduled Jobs ---

def run_morning_scan_with_tracker():
    """Morning scan + log signals for accuracy tracking."""
    signals = run_morning_scan()
    try:
        log_signals_open(signals)
    except Exception as e:
        print(f"[Main] SignalTracker open log error: {e}")


def run_closing_report_with_tracker():
    """Closing delta + evaluate signal accuracy."""
    run_closing_report()
    try:
        log_signals_close()
    except Exception as e:
        print(f"[Main] SignalTracker close log error: {e}")


def run_weekly_accuracy_report():
    """Only runs on Fridays."""
    if now_wib().weekday() != 4:  # 4 = Friday
        return
    try:
        send_weekly_accuracy_report()
    except Exception as e:
        print(f"[Main] Weekly accuracy report error: {e}")


# Morning scan (UTC 01:45 = WIB 08:45)
schedule.every().day.at("01:45").do(safe_run, run_morning_scan_with_tracker, "Morning Scan + Tracker")

# Market open commodity check (UTC 02:00 = WIB 09:00)
schedule.every().day.at("02:00").do(safe_run, run_radar, "Radar (Market Open)")

# Real-time monitoring every 10 min during market hours (01:55-08:00 UTC = 08:55-15:00 WIB)
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

# Sentinel checks (UTC = WIB-7)
schedule.every().day.at("02:30").do(safe_run, run_sentinel, "Sentinel")
schedule.every().day.at("03:30").do(safe_run, run_sentinel, "Sentinel")
schedule.every().day.at("03:30").do(safe_run, run_radar, "Radar")
schedule.every().day.at("04:30").do(safe_run, run_sentinel, "Sentinel")
schedule.every().day.at("06:00").do(safe_run, run_sentinel, "Sentinel")
schedule.every().day.at("06:00").do(safe_run, run_radar, "Radar")
schedule.every().day.at("07:00").do(safe_run, run_sentinel, "Sentinel")
schedule.every().day.at("07:00").do(safe_run, run_radar, "Radar")

# Closing delta report + signal accuracy (UTC 08:35 = WIB 15:35)
schedule.every().day.at("08:35").do(safe_run, run_closing_report_with_tracker, "Closing Delta Report + Tracker")

# Friday weekly accuracy report (UTC 08:40 = WIB 15:40)
schedule.every().day.at("08:40").do(safe_run, run_weekly_accuracy_report, "Weekly Accuracy Report")


def run_daemon():
    print(f"[Saham Scanner] Daemon started at {now_wib().strftime('%Y-%m-%d %H:%M')} WIB")
    print("[Saham Scanner] Schedule:")
    print("  08:45 WIB - Morning Scan (macro + sector context)")
    print("  08:55-15:00 WIB - Real-time every 10 min (Scanner + PositionTracker)")
    print("  09:00+ WIB - Sentinel & Radar checks")
    print("  15:35 WIB - Closing delta report")
    print("[Saham Scanner] Self-healing: agents auto-restart on failure")

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
        elif cmd == "positions":
            run_position_monitor()
        elif cmd == "weekly":
            send_weekly_accuracy_report()
        elif cmd == "daemon":
            run_daemon()
        else:
            print("Usage: python main.py [scan|close|sentinel|radar|realtime|positions|weekly|daemon]")
    else:
        run_daemon()
