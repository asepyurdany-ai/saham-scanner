"""
Saham Scanner — Main Daemon
Schedule (WIB):
  08:45 - Morning scan (Scanner)
  09:00 - Market open commodity check (Radar)
  09:30, 10:30, 11:30, 13:00, 14:00 - Sentinel + Radar check
  15:35 - Closing delta report (Scanner)
  Weekend/holiday: skip
"""

import time
import schedule
from datetime import datetime, timedelta
from agents.scanner import run_morning_scan, run_closing_report
from agents.sentinel import run_sentinel
from agents.radar import run_radar


def now_wib():
    return datetime.utcnow() + timedelta(hours=7)


def is_market_day() -> bool:
    """Skip weekends. TODO: add IDX holiday calendar"""
    wib = now_wib()
    return wib.weekday() < 5  # Mon-Fri


def safe_run(fn, name: str):
    if not is_market_day():
        print(f"[Main] Weekend — skipping {name}")
        return
    try:
        print(f"[Main] Running {name} at {now_wib().strftime('%H:%M')} WIB")
        fn()
    except Exception as e:
        print(f"[Main] ERROR in {name}: {e}")


# Schedule (all times WIB = UTC+7, schedule library uses local time)
# Server is UTC, so we offset: WIB 08:45 = UTC 01:45
schedule.every().day.at("01:45").do(safe_run, run_morning_scan, "Morning Scan")
schedule.every().day.at("02:00").do(safe_run, run_radar, "Radar (Market Open)")
schedule.every().day.at("02:30").do(safe_run, run_sentinel, "Sentinel")
schedule.every().day.at("03:30").do(safe_run, run_sentinel, "Sentinel")
schedule.every().day.at("03:30").do(safe_run, run_radar, "Radar")
schedule.every().day.at("04:30").do(safe_run, run_sentinel, "Sentinel")
schedule.every().day.at("06:00").do(safe_run, run_sentinel, "Sentinel")
schedule.every().day.at("06:00").do(safe_run, run_radar, "Radar")
schedule.every().day.at("07:00").do(safe_run, run_sentinel, "Sentinel")
schedule.every().day.at("07:00").do(safe_run, run_radar, "Radar")
schedule.every().day.at("08:35").do(safe_run, run_closing_report, "Closing Delta Report")


def run_daemon():
    print(f"[Saham Scanner] Daemon started at {now_wib().strftime('%Y-%m-%d %H:%M')} WIB")
    print("[Saham Scanner] Schedule: Morning 08:45, Sentinel every ~1h, Close 15:35 WIB")
    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "scan":
            run_morning_scan()
        elif cmd == "close":
            run_closing_report()
        elif cmd == "sentinel":
            run_sentinel()
        elif cmd == "radar":
            run_radar()
        elif cmd == "daemon":
            run_daemon()
        else:
            print("Usage: python main.py [scan|close|sentinel|radar|daemon]")
    else:
        run_daemon()
