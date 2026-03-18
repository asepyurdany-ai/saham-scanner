"""
Scanner Agent — Technical analysis + Whale detection
Hedge fund discipline: only strong signals pass
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import json
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# IDX30 + LQ45 watchlist — liquid, tight spread
WATCHLIST = [
    "BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK",  # Perbankan
    "TLKM.JK", "EXCL.JK",                           # Telco
    "ANTM.JK", "MDKA.JK", "MEDC.JK",               # Mining/Energy
    "GOTO.JK", "BUKA.JK",                            # Tech
    "ASII.JK", "AALI.JK",                            # Industri
    "UNVR.JK", "ICBP.JK", "INDF.JK",               # Consumer
    "ADRO.JK", "PTBA.JK",                            # Coal
    "SMGR.JK", "INTP.JK",                            # Semen
    "AKRA.JK", "ELSA.JK",                            # Oil related
]

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


def send_telegram(msg: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    resp = requests.post(url, json={
        "chat_id": TELEGRAM_CHAT_ID,
        "text": msg,
        "parse_mode": "HTML"
    })
    data = resp.json()
    if not data.get("ok"):
        print(f"[Telegram ERROR] {data}")
    return data.get("ok", False)


def get_stock_data(ticker: str, period: str = "20d") -> pd.DataFrame:
    """Fetch OHLCV from Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        return df
    except Exception as e:
        print(f"[Scanner] Error fetching {ticker}: {e}")
        return pd.DataFrame()


def compute_signals(ticker: str, df: pd.DataFrame) -> dict:
    """
    Compute technical signals.
    Hedge fund rule: need 4/5 conditions for strong signal.
    """
    if df.empty or len(df) < 20:
        return None

    close = df["Close"]
    volume = df["Volume"]

    # MAs
    ma20 = close.rolling(20).mean().iloc[-1]
    ma50 = close.rolling(min(50, len(df))).mean().iloc[-1]
    current = close.iloc[-1]
    prev_close = close.iloc[-2]

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    rsi = (100 - (100 / (1 + rs))).iloc[-1]

    # Volume analysis (whale detection)
    avg_vol_5d = volume.iloc[-6:-1].mean()
    today_vol = volume.iloc[-1]
    vol_ratio = today_vol / avg_vol_5d if avg_vol_5d > 0 else 1

    # Daily change
    daily_change_pct = ((current - prev_close) / prev_close) * 100

    # ARA/ARB detection (simplified — IDX rules vary by price range)
    if current < 200:
        ara_limit = 0.35
        arb_limit = -0.07
    elif current < 5000:
        ara_limit = 0.25
        arb_limit = -0.07
    else:
        ara_limit = 0.20
        arb_limit = -0.07

    approaching_ara = daily_change_pct >= (ara_limit * 100 * 0.7)  # 70% of ARA
    near_arb = daily_change_pct <= (arb_limit * 100 * 0.5)         # 50% of ARB

    # Signal scoring (hedge fund discipline)
    conditions = {
        "uptrend": ma20 > ma50,
        "volume_spike": vol_ratio >= 2.0,  # 2x average = whale activity
        "rsi_ok": 30 <= rsi <= 65,         # Not overbought, not oversold
        "momentum_positive": daily_change_pct > 0,
        "not_approaching_limit": not approaching_ara,
    }

    score = sum(conditions.values())

    return {
        "ticker": ticker,
        "current": round(current, 0),
        "prev_close": round(prev_close, 0),
        "daily_change_pct": round(daily_change_pct, 2),
        "ma20": round(ma20, 0),
        "ma50": round(ma50, 0),
        "rsi": round(rsi, 1),
        "vol_ratio": round(vol_ratio, 2),
        "score": score,
        "conditions": conditions,
        "approaching_ara": approaching_ara,
        "near_arb": near_arb,
        "signal": "BUY" if score >= 4 else "WATCH" if score >= 3 else "AVOID",
    }


def get_foreign_flow() -> dict:
    """
    Fetch IDX foreign net buy/sell data.
    Returns summary of foreign flow today.
    """
    try:
        # IDX foreign flow via public endpoint
        url = "https://www.idx.co.id/umbraco/Surface/StockData/GetSecuritiesStock"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(
            "https://www.idx.co.id/umbraco/Surface/Helper/GetLastTradingDate",
            headers=headers, timeout=5
        )
        # Simplified: use Yahoo Finance foreign proxy via volume patterns
        # Full IDX API requires session — will enhance later
        return {"status": "limited", "note": "Using volume proxy for whale detection"}
    except Exception as e:
        return {"status": "error", "note": str(e)}


def scan_all() -> list:
    """Scan entire watchlist, return ranked signals"""
    results = []
    print(f"[Scanner] Scanning {len(WATCHLIST)} stocks...")

    for ticker in WATCHLIST:
        df = get_stock_data(ticker)
        signal = compute_signals(ticker, df)
        if signal:
            results.append(signal)
            print(f"  {ticker}: {signal['signal']} (score={signal['score']}, RSI={signal['rsi']}, vol={signal['vol_ratio']}x)")

    # Sort: BUY first, then WATCH, then by score
    results.sort(key=lambda x: (-x["score"], x["signal"]))
    return results


def format_morning_alert(signals: list) -> str:
    """Format morning briefing Telegram message"""
    buy_signals = [s for s in signals if s["signal"] == "BUY"]
    watch_signals = [s for s in signals if s["signal"] == "WATCH"]

    now_wib = datetime.utcnow() + timedelta(hours=7)
    lines = [
        f"🏦 <b>MORNING SCAN — {now_wib.strftime('%d %b %Y %H:%M')} WIB</b>",
        f"<i>Hedge fund mode: hanya sinyal kuat yang lolos</i>",
        "",
    ]

    if buy_signals:
        lines.append("🟢 <b>STRONG BUY SIGNALS:</b>")
        for s in buy_signals[:5]:
            ticker_clean = s["ticker"].replace(".JK", "")
            whale = "🐋 " if s["vol_ratio"] >= 2.0 else ""
            lines.append(
                f"{whale}<b>{ticker_clean}</b> Rp{s['current']:,.0f} "
                f"({s['daily_change_pct']:+.2f}%) "
                f"| RSI {s['rsi']} | Vol {s['vol_ratio']}x | Score {s['score']}/5"
            )
        lines.append("")

    if watch_signals:
        lines.append("🟡 <b>WATCH LIST:</b>")
        for s in watch_signals[:3]:
            ticker_clean = s["ticker"].replace(".JK", "")
            lines.append(
                f"<b>{ticker_clean}</b> Rp{s['current']:,.0f} "
                f"({s['daily_change_pct']:+.2f}%) | Score {s['score']}/5"
            )
        lines.append("")

    if not buy_signals and not watch_signals:
        lines.append("⚠️ <b>Tidak ada sinyal kuat hari ini.</b>")
        lines.append("Market conditions tidak ideal — hold cash lebih aman.")

    lines.append("⚠️ <i>BUKAN saran investasi. DYOR.</i>")
    return "\n".join(lines)


def save_signals(signals: list):
    """Save today's signals for delta report at close"""
    path = "data/signals_today.json"
    os.makedirs("data", exist_ok=True)
    # Convert numpy types to native Python
    def convert(obj):
        if hasattr(obj, 'item'):
            return obj.item()
        return obj
    clean = []
    for s in signals:
        clean.append({k: (convert(v) if not isinstance(v, dict) else {kk: convert(vv) for kk, vv in v.items()}) for k, v in s.items()})
    with open(path, "w") as f:
        json.dump({
            "date": datetime.utcnow().strftime("%Y-%m-%d"),
            "signals": clean
        }, f, indent=2)
    print(f"[Scanner] Signals saved to {path}")


def run_morning_scan():
    """Run at 08:45 WIB — before market open"""
    print("[Scanner] Running morning scan...")
    signals = scan_all()
    save_signals(signals)
    msg = format_morning_alert(signals)
    ok = send_telegram(msg)
    print(f"[Scanner] Morning alert sent: {ok}")
    return signals


def run_closing_report():
    """Run at 15:35 WIB — after market close. Show delta."""
    path = "data/signals_today.json"
    if not os.path.exists(path):
        print("[Scanner] No morning signals found for delta report")
        return

    with open(path) as f:
        data = json.load(f)

    morning_signals = {s["ticker"]: s for s in data["signals"] if s["signal"] in ["BUY", "WATCH"]}

    if not morning_signals:
        return

    now_wib = datetime.utcnow() + timedelta(hours=7)
    lines = [
        f"📊 <b>DAILY DELTA REPORT — {now_wib.strftime('%d %b %Y')} WIB</b>",
        "",
    ]

    for ticker, morning in morning_signals.items():
        df = get_stock_data(ticker, period="2d")
        if df.empty:
            continue
        close_price = df["Close"].iloc[-1]
        open_price = df["Open"].iloc[-1]
        delta = ((close_price - open_price) / open_price) * 100
        from_morning = ((close_price - morning["current"]) / morning["current"]) * 100
        emoji = "🟢" if delta > 0 else "🔴"
        ticker_clean = ticker.replace(".JK", "")
        lines.append(
            f"{emoji} <b>{ticker_clean}</b> "
            f"Open: {open_price:,.0f} → Close: {close_price:,.0f} "
            f"| Delta: {delta:+.2f}% "
            f"| vs Pagi: {from_morning:+.2f}%"
        )

    lines.append("")
    lines.append("📈 <i>Track record untuk evaluasi sinyal ke depan.</i>")
    send_telegram("\n".join(lines))
    print("[Scanner] Closing delta report sent")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "close":
        run_closing_report()
    else:
        run_morning_scan()
