"""
Market Breadth Agent — IHSG gate + market breadth calculation.

CLOSED/CAUTIOUS gate blocks or restricts buy signals via market_context.
Integrates with scanner.py validate_signal() and compute_signals().
"""

import json
import os

import requests
import yfinance as yf
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "5922770410")


def fetch_ihsg_data() -> dict:
    """
    Fetch IHSG (^JKSE) current data from yfinance.

    Returns:
        {
            "current": float,
            "open": float,
            "prev_close": float,
            "change_from_open_pct": float,
            "change_from_prev_pct": float,
        }
        Returns None on failure.
    """
    try:
        ticker = yf.Ticker("^JKSE")
        hist = ticker.history(period="2d")
        if hist is None or hist.empty:
            print("[MarketBreadth] IHSG: empty data")
            return None

        current = float(hist["Close"].iloc[-1])
        open_today = float(hist["Open"].iloc[-1])
        prev_close = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else current

        change_from_open_pct = ((current - open_today) / open_today) * 100 if open_today else 0.0
        change_from_prev_pct = ((current - prev_close) / prev_close) * 100 if prev_close else 0.0

        return {
            "current": round(current, 2),
            "open": round(open_today, 2),
            "prev_close": round(prev_close, 2),
            "change_from_open_pct": round(change_from_open_pct, 2),
            "change_from_prev_pct": round(change_from_prev_pct, 2),
        }
    except Exception as e:
        print(f"[MarketBreadth] IHSG fetch error: {e}")
        return None


def calculate_breadth(watchlist_signals: list) -> dict:
    """
    Calculate market breadth from scanner signal list.

    Args:
        watchlist_signals: list of signal dicts with "daily_change_pct" key

    Returns:
        {
            "advancing": int,
            "declining": int,
            "breadth_pct": float,
            "breadth_signal": "STRONG"|"NEUTRAL"|"WEAK"
        }
    """
    if not watchlist_signals:
        return {
            "advancing": 0,
            "declining": 0,
            "breadth_pct": 0.0,
            "breadth_signal": "NEUTRAL",
        }

    advancing = sum(
        1 for s in watchlist_signals if s.get("daily_change_pct", 0) > 0
    )
    total = len(watchlist_signals)
    declining = total - advancing
    breadth_pct = (advancing / total) * 100 if total > 0 else 0.0

    if breadth_pct > 65:
        breadth_signal = "STRONG"
    elif breadth_pct < 35:
        breadth_signal = "WEAK"
    else:
        breadth_signal = "NEUTRAL"

    return {
        "advancing": advancing,
        "declining": declining,
        "breadth_pct": round(breadth_pct, 1),
        "breadth_signal": breadth_signal,
    }


def check_market_gate(ihsg_data: dict, breadth: dict) -> dict:
    """
    Determine market gate status based on IHSG movement and breadth.

    Gate logic:
    - CLOSED:   IHSG down >1.5% from open  OR  breadth WEAK (<35%)
    - CAUTIOUS: IHSG down 0.5-1.5% from open  OR  breadth 35-50%
    - OPEN:     normal conditions

    Args:
        ihsg_data: result from fetch_ihsg_data() (can be None)
        breadth: result from calculate_breadth() (can be None)

    Returns:
        {"gate": "OPEN"|"CAUTIOUS"|"CLOSED", "reason": str}
    """
    if ihsg_data is None and breadth is None:
        return {"gate": "OPEN", "reason": "Data tidak tersedia — default OPEN"}

    ihsg_change = ihsg_data.get("change_from_open_pct", 0) if ihsg_data else 0.0
    breadth_pct = breadth.get("breadth_pct", 50.0) if breadth else 50.0
    breadth_signal = breadth.get("breadth_signal", "NEUTRAL") if breadth else "NEUTRAL"

    reasons = []

    # CLOSED conditions (highest priority)
    if ihsg_change <= -1.5 or breadth_signal == "WEAK":
        if ihsg_change <= -1.5:
            reasons.append(f"IHSG turun {ihsg_change:.1f}% dari open")
        if breadth_signal == "WEAK":
            reasons.append(f"Breadth lemah ({breadth_pct:.0f}% hijau)")
        return {"gate": "CLOSED", "reason": " | ".join(reasons)}

    # CAUTIOUS conditions
    if ihsg_change <= -0.5 or (35.0 <= breadth_pct <= 50.0):
        if ihsg_change <= -0.5:
            reasons.append(f"IHSG turun {ihsg_change:.1f}% dari open")
        if 35.0 <= breadth_pct <= 50.0:
            reasons.append(f"Breadth moderat ({breadth_pct:.0f}% hijau)")
        return {"gate": "CAUTIOUS", "reason": " | ".join(reasons)}

    return {"gate": "OPEN", "reason": "Kondisi market normal"}


def format_breadth_alert(ihsg_data: dict, breadth: dict, gate: dict) -> str:
    """
    Format market breadth alert for Telegram.

    Returns None if gate is OPEN (no alert needed).
    """
    gate_status = gate.get("gate", "OPEN") if gate else "OPEN"

    if gate_status == "OPEN":
        return None

    now_wib = datetime.utcnow() + timedelta(hours=7)
    time_str = now_wib.strftime("%H:%M WIB")

    ihsg_val = ihsg_data.get("current", 0) if ihsg_data else 0
    ihsg_change = ihsg_data.get("change_from_open_pct", 0) if ihsg_data else 0
    advancing = breadth.get("advancing", 0) if breadth else 0
    declining = breadth.get("declining", 0) if breadth else 0
    total = advancing + declining
    breadth_pct = breadth.get("breadth_pct", 0) if breadth else 0

    gate_emoji = "🚫" if gate_status == "CLOSED" else "🚧"

    if gate_status == "CLOSED":
        advice = "🔴 Market melemah — tahan SEMUA BUY baru"
    else:
        advice = "💡 Market lemah — tahan BUY baru, jaga posisi existing"

    msg = (
        f"⚠️ MARKET BREADTH ALERT — {time_str}\n"
        f"\n"
        f"📊 IHSG: {ihsg_val:,.0f} ({ihsg_change:+.1f}% dari open)\n"
        f"📉 Breadth: {advancing}/{total} saham hijau ({breadth_pct:.0f}%)\n"
        f"\n"
        f"{gate_emoji} GATE: {gate_status}\n"
        f"{advice}"
    )
    return msg


def send_breadth_alert(ihsg_data: dict, breadth: dict, gate: dict) -> bool:
    """Send breadth alert to Telegram. Returns True on success."""
    msg = format_breadth_alert(ihsg_data, breadth, gate)
    if not msg:
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        resp = requests.post(
            url,
            json={"chat_id": TELEGRAM_CHAT_ID, "text": msg},
            timeout=10,
        )
        return resp.json().get("ok", False)
    except Exception as e:
        print(f"[MarketBreadth] Telegram send error: {e}")
        return False


def run_breadth_check() -> dict:
    """
    Run market breadth check. Called every 30 min during market hours.
    Updates market_context and sends alert if gate is not OPEN.

    Returns:
        {"ihsg_data": dict, "breadth": dict, "gate": dict}
    """
    try:
        from agents.market_context import update_market_breadth

        ihsg_data = fetch_ihsg_data()

        # Load today's scanner signals for breadth calculation
        signals = []
        try:
            signals_path = "data/signals_today.json"
            if os.path.exists(signals_path):
                with open(signals_path) as f:
                    data = json.load(f)
                signals = data.get("signals", [])
        except Exception as e:
            print(f"[MarketBreadth] Could not load signals: {e}")

        breadth = calculate_breadth(signals)
        gate = check_market_gate(ihsg_data, breadth)

        # Update market context
        ihsg_change = ihsg_data.get("change_from_open_pct", 0) if ihsg_data else 0
        update_market_breadth(gate["gate"], breadth["breadth_pct"], ihsg_change)

        # Send alert if gate is not OPEN
        if gate["gate"] != "OPEN":
            send_breadth_alert(ihsg_data, breadth, gate)

        print(
            f"[MarketBreadth] Gate={gate['gate']}, "
            f"Breadth={breadth['breadth_pct']:.0f}%, "
            f"IHSG={ihsg_change:+.2f}%"
        )
        return {"ihsg_data": ihsg_data, "breadth": breadth, "gate": gate}

    except Exception as e:
        print(f"[MarketBreadth] run_breadth_check error: {e}")
        return {}
