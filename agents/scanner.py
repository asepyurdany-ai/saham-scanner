"""
Scanner Agent — Technical analysis + Whale detection + Macro context
Hedge fund discipline: STRONG BUY only when 5/6 conditions met
Real-time monitoring every 10 min during market hours (08:55-15:00 WIB)
"""

import yfinance as yf
import pandas as pd
import numpy as np
import anthropic
import requests
import json
import os
import time
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

# Sector grouping for context
SECTOR_MAP = {
    "Perbankan": ["BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK"],
    "Telco": ["TLKM.JK", "EXCL.JK"],
    "Mining": ["ANTM.JK", "MDKA.JK", "MEDC.JK"],
    "Tech": ["GOTO.JK", "BUKA.JK"],
    "Industri": ["ASII.JK", "AALI.JK"],
    "Consumer": ["UNVR.JK", "ICBP.JK", "INDF.JK"],
    "Coal": ["ADRO.JK", "PTBA.JK"],
    "Semen": ["SMGR.JK", "INTP.JK"],
    "Oil": ["AKRA.JK", "ELSA.JK"],
}

# Macro tickers for pre-market briefing
MACRO_TICKERS = {
    "Oil (WTI)": "CL=F",
    "Gold": "GC=F",
    "USD/IDR": "IDR=X",
}

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "5922770410")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


def send_telegram(msg: str, retries: int = 3) -> bool:
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
            print(f"[Telegram ERROR] {data}")
        except Exception as e:
            print(f"[Telegram] Attempt {attempt+1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(2)
    return False


def get_stock_data(ticker: str, period: str = "60d", retries: int = 2) -> pd.DataFrame:
    """Fetch OHLCV from Yahoo Finance with retry. 60d for MACD accuracy."""
    for attempt in range(retries):
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            if not df.empty:
                return df
        except Exception as e:
            print(f"[Scanner] Error fetching {ticker} (attempt {attempt+1}): {e}")
            if attempt < retries - 1:
                time.sleep(1)
    return pd.DataFrame()


def get_macro_data() -> dict:
    """Fetch macro data: oil, gold, rupiah"""
    macro = {}
    for name, ticker in MACRO_TICKERS.items():
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="2d")
            if not hist.empty and len(hist) >= 2:
                current = hist["Close"].iloc[-1]
                prev = hist["Close"].iloc[-2]
                change_pct = ((current - prev) / prev) * 100
                macro[name] = {
                    "current": round(current, 2),
                    "change_pct": round(change_pct, 2),
                }
            elif not hist.empty:
                macro[name] = {
                    "current": round(hist["Close"].iloc[-1], 2),
                    "change_pct": 0.0,
                }
        except Exception as e:
            print(f"[Scanner] Macro fetch error {name}: {e}")
    return macro


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Compute MACD line and signal line."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def compute_signals(ticker: str, df: pd.DataFrame) -> dict:
    """
    Compute technical signals.
    Hedge fund rule: need 5/6 conditions for STRONG BUY.

    Conditions:
      1. Trend naik: MA20 > MA50
      2. Volume spike: vol > 2x avg 5 hari (whale)
      3. RSI sehat: RSI < 70 AND > 25 (tidak oversold extreme)
      4. Price above MA20
      5. MACD bullish (MACD line > Signal line OR crossover in last 3 candles)
      6. Momentum: daily change > 0%

    Signal levels:
      STRONG BUY: score 5-6
      WATCH:      score 3-4
      AVOID:      score < 3
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

    # RSI (14)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    rsi_series = (100 - (100 / (1 + rs)))
    rsi = rsi_series.iloc[-1]

    # Volume analysis (whale detection) — avg 5 days before today
    avg_vol_5d = volume.iloc[-6:-1].mean()
    today_vol = volume.iloc[-1]
    vol_ratio = today_vol / avg_vol_5d if avg_vol_5d > 0 else 1.0

    # Daily change
    daily_change_pct = ((current - prev_close) / prev_close) * 100

    # ARA/ARB detection (simplified)
    if current < 200:
        ara_limit = 0.35
        arb_limit = -0.07
    elif current < 5000:
        ara_limit = 0.25
        arb_limit = -0.07
    else:
        ara_limit = 0.20
        arb_limit = -0.07

    approaching_ara = daily_change_pct >= (ara_limit * 100 * 0.7)
    near_arb = daily_change_pct <= (arb_limit * 100 * 0.5)

    # Bollinger Bands
    bb_std = close.rolling(20).std().iloc[-1]
    bb_upper = ma20 + 2 * bb_std
    bb_lower = ma20 - 2 * bb_std

    # MACD (12/26/9)
    macd_line, signal_line = compute_macd(close)
    macd_now = macd_line.iloc[-1]
    signal_now = signal_line.iloc[-1]
    macd_above_signal = bool(macd_now > signal_now)

    # Check bullish crossover in last 3 candles
    crossover_bullish = False
    if len(macd_line) >= 4:
        macd_vals = macd_line.iloc[-4:].values
        sig_vals = signal_line.iloc[-4:].values
        for i in range(1, len(macd_vals)):
            if not np.isnan(macd_vals[i]) and not np.isnan(sig_vals[i]):
                if macd_vals[i] > sig_vals[i] and macd_vals[i-1] <= sig_vals[i-1]:
                    crossover_bullish = True
                    break

    macd_bullish = macd_above_signal or crossover_bullish

    # Handle NaN MACD
    if np.isnan(macd_now) or np.isnan(signal_now):
        macd_bullish = False

    # TP and SL (hedge fund standard: +8% TP, -4% SL)
    tp = round(current * 1.08, 0)
    sl = round(current * 0.96, 0)
    tp_pct = 8.0
    sl_pct = -4.0

    # === 6 CONDITIONS ===
    conditions = {
        "uptrend":          bool(ma20 > ma50),
        "volume_spike":     bool(vol_ratio >= 2.0),
        "rsi_ok":           bool(rsi > 25 and rsi < 70),
        "price_above_ma20": bool(current > ma20),
        "macd_bullish":     bool(macd_bullish),
        "momentum_positive": bool(daily_change_pct > 0),
    }

    score = sum(conditions.values())

    # Signal levels
    if score >= 5:
        signal = "STRONG BUY"
    elif score >= 3:
        signal = "WATCH"
    else:
        signal = "AVOID"

    # Richer reasoning
    reasons = []
    if conditions["uptrend"]:
        reasons.append(f"MA20 ({ma20:.0f}) > MA50 ({ma50:.0f}) → uptrend")
    if conditions["volume_spike"]:
        reasons.append(f"Volume spike {vol_ratio:.1f}x avg → potensi whale")
    if conditions["rsi_ok"]:
        reasons.append(f"RSI {rsi:.1f} — zona aman")
    elif rsi >= 70:
        reasons.append(f"RSI {rsi:.1f} — overbought, hati-hati")
    elif rsi <= 25:
        reasons.append(f"RSI {rsi:.1f} — oversold extreme")
    if crossover_bullish:
        reasons.append("MACD bullish crossover")
    elif macd_above_signal:
        reasons.append("MACD bullish (line > signal)")
    if near_arb:
        reasons.append("⚠️ Mendekati ARB")
    if approaching_ara:
        reasons.append("⚠️ Mendekati ARA — risiko jebakan")
    if conditions["price_above_ma20"]:
        reasons.append(f"Harga di atas MA20")

    return {
        "ticker": ticker,
        "current": round(float(current), 0),
        "prev_close": round(float(prev_close), 0),
        "daily_change_pct": round(float(daily_change_pct), 2),
        "ma20": round(float(ma20), 0),
        "ma50": round(float(ma50), 0),
        "rsi": round(float(rsi), 1),
        "vol_ratio": round(float(vol_ratio), 2),
        "macd_bullish": bool(macd_bullish),
        "macd_crossover": bool(crossover_bullish),
        "tp": round(float(tp), 0),
        "sl": round(float(sl), 0),
        "tp_pct": tp_pct,
        "sl_pct": sl_pct,
        "score": int(score),
        "conditions": conditions,
        "approaching_ara": bool(approaching_ara),
        "near_arb": bool(near_arb),
        "signal": signal,
        "reasons": reasons,
    }


def get_sector_performance(signals: list) -> dict:
    """Compute sector-level performance summary."""
    sector_data = {}
    ticker_to_signal = {s["ticker"]: s for s in signals}

    for sector, tickers in SECTOR_MAP.items():
        sector_signals = [ticker_to_signal[t] for t in tickers if t in ticker_to_signal]
        if not sector_signals:
            continue
        avg_change = sum(s["daily_change_pct"] for s in sector_signals) / len(sector_signals)
        buy_count = sum(1 for s in sector_signals if s["signal"] == "STRONG BUY")
        sector_data[sector] = {
            "avg_change_pct": round(avg_change, 2),
            "buy_count": buy_count,
            "total": len(sector_signals),
        }

    return sector_data


def get_claude_market_context(signals: list, macro: dict) -> str:
    """Use Claude Sonnet for richer sector rotation narrative."""
    if not ANTHROPIC_API_KEY:
        return None

    try:
        buy_signals = [s for s in signals if s["signal"] == "STRONG BUY"]
        watch_signals = [s for s in signals if s["signal"] == "WATCH"]

        macro_text = "\n".join([
            f"- {name}: {d['current']} ({d['change_pct']:+.2f}%)"
            for name, d in macro.items()
        ])

        buy_text = "\n".join([
            f"- {s['ticker'].replace('.JK', '')}: RSI {s['rsi']}, Vol {s['vol_ratio']}x, "
            f"MACD {'✅' if s['macd_bullish'] else '❌'}, {s['daily_change_pct']:+.2f}%"
            for s in buy_signals[:5]
        ]) or "Tidak ada"

        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=400,
            messages=[{
                "role": "user",
                "content": f"""Kamu senior analis saham Indonesia di hedge fund. Berikan 3-4 kalimat konteks pasar yang tajam (bahasa Indonesia) berdasarkan:

Makro hari ini:
{macro_text}

Sinyal STRONG BUY teratas:
{buy_text}

Fokus: rotasi sektor, dampak makro ke IHSG, peluang konkret. Singkat, presisi, actionable."""
            }]
        )
        return response.content[0].text.strip()
    except Exception as e:
        print(f"[Scanner] Claude context error: {e}")
        return None


def scan_all() -> list:
    """Scan entire watchlist, return ranked signals"""
    results = []
    print(f"[Scanner] Scanning {len(WATCHLIST)} stocks...")

    for ticker in WATCHLIST:
        try:
            df = get_stock_data(ticker)
            signal = compute_signals(ticker, df)
            if signal:
                results.append(signal)
                print(f"  {ticker}: {signal['signal']} (score={signal['score']}/6, "
                      f"RSI={signal['rsi']}, vol={signal['vol_ratio']}x, "
                      f"MACD={'✅' if signal['macd_bullish'] else '❌'})")
        except Exception as e:
            print(f"[Scanner] Error processing {ticker}: {e}")

    # Sort: STRONG BUY first, then WATCH, then by score
    results.sort(key=lambda x: (-x["score"], x["signal"]))
    return results


def format_morning_alert(signals: list, macro: dict = None, market_context: str = None) -> str:
    """Format morning briefing Telegram message with macro + trade setup"""
    strong_buy_signals = [s for s in signals if s["signal"] == "STRONG BUY"]
    watch_signals = [s for s in signals if s["signal"] == "WATCH"]

    now_wib = datetime.utcnow() + timedelta(hours=7)
    lines = [
        f"🏦 <b>MORNING SCAN — {now_wib.strftime('%d %b %Y %H:%M')} WIB</b>",
        f"<i>Hedge fund mode: STRONG BUY hanya 5-6/6 konfirmasi</i>",
        "",
    ]

    # Macro context
    if macro:
        lines.append("🌍 <b>MAKRO:</b>")
        for name, data in macro.items():
            sign = "+" if data["change_pct"] > 0 else ""
            emoji = "🟢" if data["change_pct"] > 0 else "🔴" if data["change_pct"] < 0 else "⚪"
            lines.append(f"  {emoji} {name}: {data['current']} ({sign}{data['change_pct']:.2f}%)")
        lines.append("")

    # Claude market context
    if market_context:
        lines.append(f"💡 <i>{market_context}</i>")
        lines.append("")

    if strong_buy_signals:
        lines.append(f"🟢 <b>STRONG BUY SIGNALS ({len(strong_buy_signals)} saham):</b>")
        for s in strong_buy_signals[:5]:
            ticker_clean = s["ticker"].replace(".JK", "")
            whale = "🐋 " if s["vol_ratio"] >= 2.0 else ""
            macd_tag = "✅" if s.get("macd_bullish") else "❌"
            ma_tag = "✅" if s["conditions"].get("uptrend") else "❌"

            lines.append(f"")
            lines.append(f"{whale}<b>{ticker_clean}</b> — STRONG BUY ({s['score']}/6)")
            lines.append(f"📍 Entry : Rp {s['current']:,.0f}")
            lines.append(f"🎯 TP    : Rp {s['tp']:,.0f} (+{s.get('tp_pct', 8.0):.0f}%)")
            lines.append(f"🛡 SL    : Rp {s['sl']:,.0f} ({s.get('sl_pct', -4.0):.0f}%)")
            lines.append(
                f"📊 RSI {s['rsi']} | Vol {s['vol_ratio']}x | MACD {macd_tag} | MA {ma_tag}"
            )
            # Key reason
            key_reasons = [r for r in s.get("reasons", []) if "warning" not in r.lower() and "⚠️" not in r]
            if key_reasons:
                lines.append(f"💡 {' + '.join(key_reasons[:2])}")
        lines.append("")

    if watch_signals:
        lines.append("🟡 <b>WATCH LIST:</b>")
        for s in watch_signals[:3]:
            ticker_clean = s["ticker"].replace(".JK", "")
            lines.append(
                f"<b>{ticker_clean}</b> Rp{s['current']:,.0f} "
                f"({s['daily_change_pct']:+.2f}%) | Score {s['score']}/6"
            )
        lines.append("")

    if not strong_buy_signals and not watch_signals:
        lines.append("⚠️ <b>Tidak ada sinyal kuat hari ini.</b>")
        lines.append("Market conditions tidak ideal — hold cash lebih aman.")
        lines.append("")

    lines.append("⚠️ <i>BUKAN saran investasi. DYOR.</i>")
    return "\n".join(lines)


def format_realtime_alert(signals: list, scan_time: str) -> str:
    """Format real-time monitoring alert (condensed)"""
    buy_signals = [s for s in signals if s["signal"] == "STRONG BUY"]
    if not buy_signals:
        return None

    lines = [f"⚡ <b>REAL-TIME SCAN — {scan_time} WIB</b>", ""]
    for s in buy_signals[:3]:
        ticker_clean = s["ticker"].replace(".JK", "")
        whale = "🐋 " if s["vol_ratio"] >= 2.0 else ""
        macd_tag = "✅" if s.get("macd_bullish") else "❌"
        lines.append(
            f"{whale}<b>{ticker_clean}</b> {s['daily_change_pct']:+.2f}% "
            f"| RSI {s['rsi']} | Vol {s['vol_ratio']}x | MACD {macd_tag} | {s['score']}/6"
        )
        if s.get("reasons"):
            lines.append(f"   <i>{s['reasons'][0]}</i>")

    lines.append("")
    lines.append("<i>⚠️ BUKAN saran investasi.</i>")
    return "\n".join(lines)


def save_signals(signals: list, path: str = "data/signals_today.json"):
    """Save today's signals for delta report at close"""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else "data", exist_ok=True)

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    clean = []
    for s in signals:
        clean_s = {}
        for k, v in s.items():
            if isinstance(v, dict):
                clean_s[k] = {kk: convert(vv) for kk, vv in v.items()}
            elif isinstance(v, list):
                clean_s[k] = v
            else:
                clean_s[k] = convert(v)
        clean.append(clean_s)

    with open(path, "w") as f:
        json.dump({
            "date": datetime.utcnow().strftime("%Y-%m-%d"),
            "signals": clean
        }, f, indent=2)
    print(f"[Scanner] Signals saved to {path}")


def run_morning_scan():
    """Run at 08:45 WIB — before market open with macro + sector context"""
    print("[Scanner] Running morning scan...")
    signals = scan_all()
    save_signals(signals)

    macro = {}
    try:
        macro = get_macro_data()
    except Exception as e:
        print(f"[Scanner] Macro data error: {e}")

    market_context = None
    try:
        market_context = get_claude_market_context(signals, macro)
    except Exception as e:
        print(f"[Scanner] Claude context error: {e}")

    msg = format_morning_alert(signals, macro, market_context)
    ok = send_telegram(msg)
    print(f"[Scanner] Morning alert sent: {ok}")
    return signals


def run_realtime_scan():
    """Real-time scan during market hours — alert only on new STRONG BUY signals"""
    try:
        now_wib = datetime.utcnow() + timedelta(hours=7)
        scan_time = now_wib.strftime("%H:%M")
        print(f"[Scanner] Real-time scan at {scan_time} WIB...")

        signals = scan_all()
        msg = format_realtime_alert(signals, scan_time)

        if msg:
            ok = send_telegram(msg)
            print(f"[Scanner] Real-time alert sent: {ok}")
        else:
            print(f"[Scanner] No STRONG BUY signals at {scan_time}")

        return signals
    except Exception as e:
        print(f"[Scanner] Real-time scan ERROR: {e}")
        return []


def run_closing_report():
    """Run at 15:35 WIB — after market close. Show delta."""
    path = "data/signals_today.json"
    if not os.path.exists(path):
        print("[Scanner] No morning signals found for delta report")
        return

    with open(path) as f:
        data = json.load(f)

    morning_signals = {
        s["ticker"]: s for s in data["signals"]
        if s["signal"] in ["STRONG BUY", "BUY", "WATCH"]
    }

    if not morning_signals:
        return

    now_wib = datetime.utcnow() + timedelta(hours=7)
    lines = [
        f"📊 <b>DAILY DELTA REPORT — {now_wib.strftime('%d %b %Y')} WIB</b>",
        "",
    ]

    for ticker, morning in morning_signals.items():
        try:
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
        except Exception as e:
            print(f"[Scanner] Error in closing report for {ticker}: {e}")

    lines.append("")
    lines.append("📈 <i>Track record untuk evaluasi sinyal ke depan.</i>")
    send_telegram("\n".join(lines))
    print("[Scanner] Closing delta report sent")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "close":
        run_closing_report()
    elif len(sys.argv) > 1 and sys.argv[1] == "realtime":
        run_realtime_scan()
    else:
        run_morning_scan()
