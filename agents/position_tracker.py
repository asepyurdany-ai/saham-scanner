"""
Position Tracker Agent — Track user stock positions
Commands: /beli TICKER PRICE LOTS  OR  /beli TICKER PRICE IDR_AMOUNT
          /jual TICKER PRICE
TP: +8%, CL: -4%, Trailing stop activates at +5%

Usage: /beli BBCA 9000 5        (5 lots @ 9000)
       /beli BBCA 9000 3000000  (IDR-based entry)
Monitoring: every 10 min during market hours
"""

import math
import yfinance as yf
import json
import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "5922770410")

DEFAULT_POSITIONS_FILE = "data/positions.json"
TP_PCT = 0.08       # +8% take profit
CL_PCT = -0.04      # -4% cut loss
TRAILING_ACTIVATE_PCT = 0.05  # activate trailing stop at +5%
TRAILING_STOP_PCT = 0.05      # trailing stop = 5% below highest


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
            print(f"[PositionTracker][Telegram ERROR] {data}")
        return data.get("ok", False)
    except Exception as e:
        print(f"[PositionTracker][Telegram EXCEPTION] {e}")
        return False


def load_positions(positions_file: str = DEFAULT_POSITIONS_FILE) -> dict:
    os.makedirs(os.path.dirname(positions_file) if os.path.dirname(positions_file) else "data", exist_ok=True)
    if os.path.exists(positions_file):
        with open(positions_file) as f:
            return json.load(f)
    return {}


def save_positions(positions: dict, positions_file: str = DEFAULT_POSITIONS_FILE):
    os.makedirs(os.path.dirname(positions_file) if os.path.dirname(positions_file) else "data", exist_ok=True)
    with open(positions_file, "w") as f:
        json.dump(positions, f, indent=2)


def add_position(ticker: str, entry_price: float, lots: int,
                 positions: dict = None, positions_file: str = DEFAULT_POSITIONS_FILE) -> dict:
    """
    Add or update a position.
    ticker: stock ticker without .JK (e.g., 'BBCA')
    entry_price: entry price per share
    lots: number of lots (1 lot = 100 shares)
    Returns the position dict.
    """
    if positions is None:
        positions = load_positions(positions_file)

    ticker = ticker.upper()
    tp_price = entry_price * (1 + TP_PCT)
    cl_price = entry_price * (1 + CL_PCT)

    position = {
        "ticker": ticker,
        "entry_price": float(entry_price),
        "lots": int(lots),
        "shares": int(lots) * 100,
        "tp_price": round(tp_price, 0),
        "cl_price": round(cl_price, 0),
        "trailing_activated": False,
        "trailing_stop": None,
        "highest_price": float(entry_price),
        "current_price": float(entry_price),
        "pnl_pct": 0.0,
        "pnl_rp": 0.0,
        "added_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
    }

    positions[ticker] = position
    save_positions(positions, positions_file)
    print(f"[PositionTracker] Added {ticker}: {lots} lots @ Rp{entry_price:,.0f} | TP: {tp_price:,.0f} | CL: {cl_price:,.0f}")
    return position


def update_position(ticker: str, current_price: float,
                    positions: dict = None, positions_file: str = DEFAULT_POSITIONS_FILE) -> dict:
    """
    Update position with new price, recalculate P&L and trailing stop.
    Returns updated position dict.
    """
    if positions is None:
        positions = load_positions(positions_file)

    ticker = ticker.upper()
    if ticker not in positions:
        print(f"[PositionTracker] {ticker} not in positions")
        return None

    pos = positions[ticker]
    entry = pos["entry_price"]

    # Update P&L
    pnl_pct = ((current_price - entry) / entry) * 100
    pnl_rp = (current_price - entry) * pos["shares"]

    # Track highest
    if current_price > pos["highest_price"]:
        pos["highest_price"] = float(current_price)

    # Trailing stop logic
    profit_pct = (pos["highest_price"] - entry) / entry
    if profit_pct >= TRAILING_ACTIVATE_PCT:
        pos["trailing_activated"] = True
        new_trailing = pos["highest_price"] * (1 - TRAILING_STOP_PCT)
        # Only move trailing stop up, never down
        if pos["trailing_stop"] is None or new_trailing > pos["trailing_stop"]:
            pos["trailing_stop"] = round(new_trailing, 0)

    pos["current_price"] = float(current_price)
    pos["pnl_pct"] = round(pnl_pct, 2)
    pos["pnl_rp"] = round(pnl_rp, 0)
    pos["updated_at"] = datetime.utcnow().isoformat()

    positions[ticker] = pos
    save_positions(positions, positions_file)
    return pos


def check_tp_cl(position: dict) -> str:
    """
    Check if a position triggers TP, CL, or trailing stop.
    Returns: 'TAKE_PROFIT', 'STOP_LOSS', 'TRAILING_STOP', or None
    """
    current = position["current_price"]
    tp = position["tp_price"]
    cl = position["cl_price"]
    trailing_activated = position.get("trailing_activated", False)
    trailing_stop = position.get("trailing_stop")

    # Check TP first
    if current >= tp:
        return "TAKE_PROFIT"

    # Check trailing stop (takes priority over CL when activated)
    if trailing_activated and trailing_stop and current <= trailing_stop:
        return "TRAILING_STOP"

    # Check CL
    if current <= cl:
        return "STOP_LOSS"

    return None


def format_position_alert(position: dict, alert_type: str) -> str:
    """Format position alert message for Telegram."""
    ticker = position["ticker"]
    current = position["current_price"]
    entry = position["entry_price"]
    pnl_pct = position["pnl_pct"]
    pnl_rp = position["pnl_rp"]
    lots = position["lots"]

    if alert_type == "TAKE_PROFIT":
        emoji = "🎯"
        title = "TAKE PROFIT!"
        action = "Saatnya jual & booking profit 💰"
    elif alert_type == "STOP_LOSS":
        emoji = "🛑"
        title = "CUT LOSS!"
        action = "Jual sekarang, lindungi modal 🚨"
    elif alert_type == "TRAILING_STOP":
        emoji = "🔔"
        title = "TRAILING STOP!"
        action = f"Harga balik dari puncak. Trailing stop: Rp{position.get('trailing_stop', 0):,.0f}"
    else:
        emoji = "📊"
        title = "UPDATE POSISI"
        action = "Pantau terus."

    sign = "+" if pnl_pct >= 0 else ""
    lines = [
        f"{emoji} <b>{title} — {ticker}</b>",
        f"",
        f"📌 Entry: <b>Rp{entry:,.0f}</b> | Current: <b>Rp{current:,.0f}</b>",
        f"📦 Lot: {lots} | Shares: {lots * 100:,}",
        f"💵 P&L: <b>{sign}{pnl_pct:.2f}%</b> ({sign}Rp{pnl_rp:,.0f})",
        f"",
        f"💡 {action}",
        f"<i>⚠️ Keputusan final tetap di tangan kamu.</i>",
    ]
    return "\n".join(lines)


def format_portfolio_summary(positions: dict) -> str:
    """Format portfolio summary for Telegram."""
    now_wib = datetime.utcnow() + timedelta(hours=7)
    if not positions:
        return "📊 <b>Portfolio kosong.</b>\nGunakan /beli TICKER HARGA LOT untuk tambah posisi."

    lines = [
        f"📊 <b>PORTFOLIO — {now_wib.strftime('%H:%M')} WIB</b>",
        "",
    ]

    total_pnl_rp = 0
    for ticker, pos in positions.items():
        pnl_pct = pos.get("pnl_pct", 0)
        pnl_rp = pos.get("pnl_rp", 0)
        current = pos.get("current_price", pos["entry_price"])
        sign = "+" if pnl_pct >= 0 else ""
        emoji = "🟢" if pnl_pct > 0 else "🔴" if pnl_pct < 0 else "⚪"
        trailing = " 🔔" if pos.get("trailing_activated") else ""
        lines.append(
            f"{emoji} <b>{ticker}</b> Rp{current:,.0f} "
            f"| {sign}{pnl_pct:.2f}% ({sign}Rp{pnl_rp:,.0f}){trailing}"
        )
        total_pnl_rp += pnl_rp

    lines.append("")
    sign = "+" if total_pnl_rp >= 0 else ""
    lines.append(f"💼 <b>Total P&L: {sign}Rp{total_pnl_rp:,.0f}</b>")
    return "\n".join(lines)


def get_current_price(ticker: str) -> float:
    """Fetch current price for a ticker (IDX format: BBCA → BBCA.JK)"""
    yf_ticker = ticker if ticker.endswith(".JK") else f"{ticker}.JK"
    try:
        t = yf.Ticker(yf_ticker)
        hist = t.history(period="1d", interval="1m")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
        # fallback to daily
        hist = t.history(period="2d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception as e:
        print(f"[PositionTracker] Error fetching price for {ticker}: {e}")
    return None


def monitor_positions(positions_file: str = DEFAULT_POSITIONS_FILE):
    """Monitor all positions, send alerts if TP/CL/trailing triggered."""
    positions = load_positions(positions_file)
    if not positions:
        print("[PositionTracker] No positions to monitor")
        return

    print(f"[PositionTracker] Monitoring {len(positions)} positions...")
    alerts_sent = []

    for ticker, pos in list(positions.items()):
        current_price = get_current_price(ticker)
        if current_price is None:
            print(f"[PositionTracker] Could not fetch price for {ticker}")
            continue

        pos = update_position(ticker, current_price, positions, positions_file)
        if pos is None:
            continue

        alert_type = check_tp_cl(pos)
        if alert_type:
            msg = format_position_alert(pos, alert_type)
            ok = send_telegram(msg)
            print(f"[PositionTracker] {alert_type} alert for {ticker}: {ok}")
            alerts_sent.append(ticker)
        else:
            print(f"[PositionTracker] {ticker}: Rp{current_price:,.0f} | P&L: {pos['pnl_pct']:+.2f}%")

    return alerts_sent


def parse_beli_command(text: str) -> tuple:
    """
    Parse /beli command.
    Format: /beli TICKER PRICE LOTS  OR  /beli TICKER PRICE IDR_AMOUNT
    Returns: (ticker, price, lots) or None if invalid
    Detection: if 4th arg > 50000 → treat as IDR, else treat as lots
    """
    parts = text.strip().split()
    if len(parts) < 4:
        return None
    try:
        cmd, ticker, price, fourth = parts[0], parts[1], parts[2], parts[3]
        if cmd.lower() not in ["/beli", "beli"]:
            return None
        price_val = float(price)
        fourth_val = float(fourth)
        if fourth_val > 50000:
            # IDR-based: calculate lots
            lots = math.floor(fourth_val / (price_val * 100))
        else:
            lots = int(fourth_val)
        return ticker.upper(), price_val, lots
    except (ValueError, IndexError):
        return None


def parse_buy_command(text: str) -> dict:
    """
    Parse /beli command and return a structured dict.
    - /beli BBCA 6070 3000000 → {ticker: "BBCA", price: 6070, total_idr: 3000000}
    - /beli BBCA 6070 10 → {ticker: "BBCA", price: 6070, lots: 10}
    Detection: if 4th arg > 50000 → treat as IDR, else treat as lots
    Returns None if invalid.
    """
    parts = text.strip().split()
    if len(parts) < 4:
        return None
    try:
        cmd, ticker, price, fourth = parts[0], parts[1], parts[2], parts[3]
        if cmd.lower() not in ["/beli", "beli"]:
            return None
        price_val = float(price)
        fourth_val = float(fourth)
        ticker = ticker.upper()
        if fourth_val > 50000:
            return {"ticker": ticker, "price": price_val, "total_idr": int(fourth_val)}
        else:
            return {"ticker": ticker, "price": price_val, "lots": int(fourth_val)}
    except (ValueError, IndexError):
        return None


def add_position_idr(ticker: str, price: float, total_idr: int,
                     positions: dict = None, positions_file: str = DEFAULT_POSITIONS_FILE) -> dict:
    """
    Add a position using IDR total amount.
    lots = floor(total_idr / (price * 100))
    Stores both lots AND total_idr fields.
    Returns the position dict.
    """
    lots = math.floor(total_idr / (price * 100))
    if lots < 1:
        raise ValueError(f"Modal Rp{total_idr:,.0f} tidak cukup untuk membeli 1 lot {ticker} @ Rp{price:,.0f}")

    actual_cost = lots * price * 100

    if positions is None:
        positions = load_positions(positions_file)

    ticker = ticker.upper()
    tp_price = price * (1 + TP_PCT)
    cl_price = price * (1 + CL_PCT)

    position = {
        "ticker": ticker,
        "entry_price": float(price),
        "lots": int(lots),
        "shares": int(lots) * 100,
        "total_idr": int(total_idr),
        "actual_cost": int(actual_cost),
        "tp_price": round(tp_price, 0),
        "cl_price": round(cl_price, 0),
        "trailing_activated": False,
        "trailing_stop": None,
        "highest_price": float(price),
        "current_price": float(price),
        "pnl_pct": 0.0,
        "pnl_rp": 0.0,
        "added_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
    }

    positions[ticker] = position
    save_positions(positions, positions_file)
    print(f"[PositionTracker] Added IDR {ticker}: {lots} lots @ Rp{price:,.0f} | Modal: Rp{actual_cost:,.0f} | TP: {tp_price:,.0f} | CL: {cl_price:,.0f}")
    return position


def _get_today_strong_buy_tickers(signals_file: str = "data/signals_today.json") -> set:
    """Return set of tickers (without .JK) that are STRONG BUY today."""
    try:
        if not os.path.exists(signals_file):
            return set()
        with open(signals_file) as f:
            data = json.load(f)
        tickers = set()
        today = datetime.utcnow().strftime("%Y-%m-%d")
        # Accept signals regardless of date, or only today's
        signals = data.get("signals", [])
        for sig in signals:
            if sig.get("signal") == "STRONG BUY":
                ticker = sig.get("ticker", "")
                # Strip .JK suffix
                ticker = ticker.replace(".JK", "").upper()
                tickers.add(ticker)
        return tickers
    except Exception as e:
        print(f"[PositionTracker] Error reading signals: {e}")
        return set()


def close_position(ticker: str, sell_price: float,
                   positions: dict = None, positions_file: str = DEFAULT_POSITIONS_FILE) -> dict:
    """
    Close an open position and calculate final P&L.
    - Calculates profit_rp and profit_pct
    - Marks position as closed
    - Saves closed trade info (used by trade_journal)
    - Sends Telegram close alert
    Returns closed trade dict, or None if ticker not found.
    """
    if positions is None:
        positions = load_positions(positions_file)

    ticker = ticker.upper()
    if ticker not in positions:
        print(f"[PositionTracker] {ticker} not found in positions")
        return None

    pos = positions[ticker]
    entry_price = pos["entry_price"]
    lots = pos["lots"]
    shares = pos["shares"]
    entry_time_str = pos.get("added_at", datetime.utcnow().isoformat())
    exit_time = datetime.utcnow()

    # Parse entry time
    try:
        entry_time = datetime.fromisoformat(entry_time_str)
    except Exception:
        entry_time = exit_time

    # P&L
    profit_rp = round((sell_price - entry_price) * shares, 0)
    profit_pct = round((sell_price - entry_price) / entry_price * 100, 2)

    # Hold duration
    hold_delta = exit_time - entry_time
    hold_minutes = int(hold_delta.total_seconds() / 60)

    # Format times in WIB
    entry_wib = entry_time + timedelta(hours=7)
    exit_wib = exit_time + timedelta(hours=7)

    entry_fmt = entry_wib.strftime("%d %b %H:%M")
    exit_fmt = exit_wib.strftime("%d %b %H:%M")

    # Hold duration string
    h = hold_minutes // 60
    m = hold_minutes % 60
    if h > 0:
        hold_str = f"{h}j {m}m" if m > 0 else f"{h}j"
    else:
        hold_str = f"{m}m"

    # Check if followed signal
    strong_buy_tickers = _get_today_strong_buy_tickers()
    followed_signal = ticker in strong_buy_tickers

    # Signal score
    signal_score = None
    try:
        if os.path.exists("data/signals_today.json"):
            with open("data/signals_today.json") as f:
                sig_data = json.load(f)
            for sig in sig_data.get("signals", []):
                if sig.get("ticker", "").replace(".JK", "").upper() == ticker:
                    signal_score = sig.get("score")
                    break
    except Exception:
        pass

    # Modal
    modal_idr = pos.get("actual_cost", pos.get("total_idr", int(entry_price * shares)))

    # Build closed trade dict
    closed_trade = {
        "id": None,  # will be set by save_trade
        "ticker": ticker,
        "entry_price": entry_price,
        "exit_price": float(sell_price),
        "lots": lots,
        "shares": shares,
        "modal_idr": int(modal_idr),
        "profit_rp": int(profit_rp),
        "profit_pct": profit_pct,
        "entry_time": entry_time.isoformat(),
        "exit_time": exit_time.isoformat(),
        "hold_minutes": hold_minutes,
        "followed_signal": followed_signal,
        "signal_score": signal_score,
        "result": "WIN" if profit_rp > 0 else ("LOSS" if profit_rp < 0 else "NEUTRAL"),
    }

    # Remove from open positions
    del positions[ticker]
    save_positions(positions, positions_file)

    # Telegram close alert
    profit_sign = "+" if profit_rp >= 0 else ""
    signal_line = f"🎯 vs Sinyal: SESUAI sinyal pagi ✅" if followed_signal else f"🎯 vs Sinyal: Di luar sinyal Dexter"

    msg_lines = [
        f"✅ <b>POSISI DITUTUP — {ticker}</b>",
        "",
        f"📍 Entry : Rp {entry_price:,.0f} ({entry_fmt})",
        f"📤 Exit  : Rp {sell_price:,.0f} ({exit_fmt})",
        f"📦 Lots  : {lots} lots ({shares:,} lembar)",
        "",
        f"💰 Profit: {profit_sign}Rp {abs(int(profit_rp)):,.0f} ({profit_sign}{profit_pct:.1f}%)",
        f"⏱ Hold  : {hold_str}",
        "",
        signal_line,
    ]
    msg = "\n".join(msg_lines)
    send_telegram(msg)

    print(f"[PositionTracker] Closed {ticker}: {profit_sign}{profit_pct:.1f}% | {profit_sign}Rp{abs(int(profit_rp)):,.0f}")
    return closed_trade


def send_position_update(positions: dict = None, positions_file: str = DEFAULT_POSITIONS_FILE):
    """
    Send intraday P&L update for all open positions.
    Called every 30 minutes during market hours.
    """
    if positions is None:
        positions = load_positions(positions_file)

    if not positions:
        return

    now_wib = datetime.utcnow() + timedelta(hours=7)
    time_str = now_wib.strftime("%H:%M")

    lines = [f"📊 <b>UPDATE POSISI — {time_str} WIB</b>", ""]

    total_pnl_rp = 0

    for ticker, pos in positions.items():
        entry = pos["entry_price"]
        current = pos.get("current_price", entry)
        pnl_pct = pos.get("pnl_pct", 0)
        pnl_rp = pos.get("pnl_rp", 0)
        tp = pos.get("tp_price", entry * 1.08)
        cl = pos.get("cl_price", entry * 0.96)

        sign = "+" if pnl_pct >= 0 else ""
        emoji = "🟢" if pnl_pct > 0 else "🔴" if pnl_pct < 0 else "⚪"

        line = (
            f"{emoji} <b>{ticker}</b>: Rp {entry:,.0f} → Rp {current:,.0f} "
            f"| {sign}{pnl_pct:.1f}% | {sign}Rp {abs(int(pnl_rp)):,.0f}"
        )
        lines.append(line)

        # TP/SL proximity hints
        tp_distance_pct = ((tp - current) / current) * 100 if current > 0 else 0
        if 0 < tp_distance_pct <= 1.0:
            lines.append(f"      🎯 TP Intraday: Rp {tp:,.0f} (hampir!)")
        elif tp_distance_pct > 0:
            lines.append(f"      🎯 TP Intraday: Rp {tp:,.0f} (+{tp_distance_pct:.1f}%)")

        trailing_stop = pos.get("trailing_stop")
        if trailing_stop:
            lines.append(f"      🔔 Trailing: Rp {trailing_stop:,.0f}")
        else:
            lines.append(f"      🛡 SL: Rp {cl:,.0f}")

        total_pnl_rp += pnl_rp

    lines.append("")
    total_sign = "+" if total_pnl_rp >= 0 else ""
    lines.append(f"💼 Total P&L hari ini: {total_sign}Rp {abs(int(total_pnl_rp)):,.0f}")

    msg = "\n".join(lines)
    send_telegram(msg)
    print(f"[PositionTracker] Sent position update for {len(positions)} positions")


def handle_beli_command(text: str, positions_file: str = DEFAULT_POSITIONS_FILE) -> str:
    """Handle /beli command, return response message."""
    result = parse_beli_command(text)
    if not result:
        return "❌ Format salah. Gunakan: /beli TICKER HARGA LOT\nContoh: /beli BBCA 9000 5 atau /beli BBCA 9000 3000000"

    ticker, price, lots = result
    position = add_position(ticker, price, lots, positions_file=positions_file)

    tp = position["tp_price"]
    cl = position["cl_price"]
    total_value = price * lots * 100

    msg = (
        f"✅ <b>Posisi ditambah: {ticker}</b>\n\n"
        f"📌 Entry: Rp{price:,.0f}\n"
        f"📦 {lots} lot ({lots * 100:,} saham) = Rp{total_value:,.0f}\n"
        f"🎯 TP: Rp{tp:,.0f} (+8%)\n"
        f"🛑 CL: Rp{cl:,.0f} (-4%)\n"
        f"🔔 Trailing stop aktif saat profit ≥5%\n\n"
        f"<i>Monitor otomatis setiap 10 menit jam pasar.</i>"
    )
    return msg


def run_position_monitor():
    """Entry point for scheduled monitoring."""
    try:
        print("[PositionTracker] Running position monitor...")
        alerts = monitor_positions()
        print(f"[PositionTracker] Done. Alerts: {alerts}")
    except Exception as e:
        print(f"[PositionTracker] ERROR: {e}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "monitor":
            run_position_monitor()
        elif cmd == "portfolio":
            positions = load_positions()
            print(format_portfolio_summary(positions))
        elif cmd == "beli" and len(sys.argv) >= 5:
            text = " ".join(sys.argv[1:])
            msg = handle_beli_command(text)
            print(msg)
            send_telegram(msg)
        else:
            print("Usage: python position_tracker.py [monitor|portfolio|beli TICKER PRICE LOTS]")
    else:
        run_position_monitor()
