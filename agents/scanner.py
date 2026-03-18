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


def get_ticker_sector(ticker: str) -> str:
    """Return sector name for a given ticker."""
    for sector, tickers in SECTOR_MAP.items():
        if ticker in tickers:
            return sector
    return "Unknown"


def format_context_line(ctx: dict, ticker_sector: str) -> str:
    """
    Return 2-3 lines of macro/sentiment/geo context for a ticker's sector.
    Used in morning alert injection.
    """
    if not ctx:
        return ""

    macro = ctx.get("macro", {})
    sentiment = ctx.get("sentiment", {})
    geo = ctx.get("geo", {})

    lines = []

    # Macro line
    rupiah_pct = macro.get("rupiah_change_pct", 0.0)
    rupiah_emoji = "✅" if abs(rupiah_pct) < 0.5 else ("⚠️" if rupiah_pct < -1.0 else "⚠️")
    rupiah_label = "stabil" if abs(rupiah_pct) < 0.5 else ("melemah" if rupiah_pct > 0 else "menguat")
    lines.append(f"🌍 Macro: Rupiah {rupiah_pct:+.1f}% {rupiah_label} {rupiah_emoji}")

    # Sentiment line — only if sector is affected
    news_sentiment = sentiment.get("news_sentiment", "NETRAL")
    affected_sectors = sentiment.get("affected_sectors", [])
    if ticker_sector in affected_sectors:
        sent_emoji = "✅" if news_sentiment == "POSITIF" else ("❌" if news_sentiment == "NEGATIF" else "➖")
        lines.append(
            f"📰 Berita: {ticker_sector} — sentimen {news_sentiment} {sent_emoji}"
        )

    # Geo line
    geo_risk = geo.get("risk_level", "LOW")
    geo_emoji = "✅" if geo_risk == "LOW" else ("⚠️" if geo_risk == "MEDIUM" else "🔴")
    lines.append(f"🌐 Geo: Risk {geo_risk} {geo_emoji}")

    return "\n".join(lines)


def validate_signal(signal: dict, ctx: dict) -> tuple:
    """
    Validate a signal against market context.

    Returns:
        (is_valid: bool, reason: str)

    Rules:
    - market_gate CLOSED → False (blocks ALL buy signals)
    - RISK_OFF + score < 6 → False
    - macro_shock_active + score < 6 → False
    - geo risk HIGH → True with warning
    - else → True
    """
    if not ctx:
        return True, "No context available"

    score = signal.get("score", 0)
    market_mode = ctx.get("market_mode", "NORMAL")
    macro_shock = ctx.get("macro", {}).get("macro_shock_active", False)
    geo_risk = ctx.get("geo", {}).get("risk_level", "LOW")

    # Gap 2: Market gate CLOSED blocks ALL buy signals
    market_gate = ctx.get("market_gate", "OPEN")
    if market_gate == "CLOSED":
        return False, "Market gate CLOSED — pasar lemah, tahan semua BUY"

    if market_mode == "RISK_OFF" and score < 6:
        return False, f"RISK_OFF mode aktif — butuh skor 6/6 (saat ini {score}/7)"

    if macro_shock and score < 6:
        return False, f"Macro shock aktif — butuh skor 6/6 (saat ini {score}/7)"

    if geo_risk == "HIGH":
        return True, "⚠️ Geo Risk HIGH — hati-hati, konfirmasi ekstra diperlukan"

    reasons = []
    if market_mode == "CAUTIOUS":
        reasons.append("mode CAUTIOUS")
    if market_gate == "CAUTIOUS":
        reasons.append("gate CAUTIOUS — threshold dinaikkan")
    if market_mode == "RISK_ON":
        reasons.append("mode RISK_ON — momentum bagus")

    reason = " | ".join(reasons) if reasons else "OK"
    return True, reason


def compute_signals(
    ticker: str,
    df: pd.DataFrame,
    ctx: dict = None,
    threshold: int = None,
    foreign_flow: dict = None,
) -> dict:
    """
    Compute technical signals.
    Hedge fund rule: need 5/7 conditions for STRONG BUY (dynamic threshold).

    Conditions:
      1. Trend naik: MA20 > MA50
      2. Volume spike: vol > 2x avg 5 hari (whale)
      3. RSI sehat: RSI < 70 AND > 25 (tidak oversold extreme)
      4. Price above MA20
      5. MACD bullish (MACD line > Signal line OR crossover in last 3 candles)
      6. Momentum: daily change > 0%
      7. Foreign net buy (asing beli neto)

    Signal levels:
      STRONG BUY: score >= threshold (default 5/7)
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

    # === 7 CONDITIONS (including foreign flow) ===
    # Condition 7: Foreign net buy
    foreign_net_buy = False
    if foreign_flow:
        clean_ticker = ticker.replace(".JK", "").upper()
        flow = foreign_flow.get(clean_ticker, {})
        net = float(flow.get("net_foreign", 0))
        if net > 0:
            foreign_net_buy = True

    conditions = {
        "uptrend":           bool(ma20 > ma50),
        "volume_spike":      bool(vol_ratio >= 2.0),
        "rsi_ok":            bool(rsi > 25 and rsi < 70),
        "price_above_ma20":  bool(current > ma20),
        "macd_bullish":      bool(macd_bullish),
        "momentum_positive": bool(daily_change_pct > 0),
        "foreign_net_buy":   bool(foreign_net_buy),
    }

    score = sum(conditions.values())

    # --- Context-aware threshold ---
    if threshold is None:
        # Load context if not provided
        if ctx is None:
            try:
                from agents.market_context import get_context, get_dynamic_threshold
                ctx = get_context()
                buy_threshold = get_dynamic_threshold()
            except Exception:
                buy_threshold = 5
        else:
            try:
                from agents.market_context import get_dynamic_threshold
                buy_threshold = get_dynamic_threshold()
            except Exception:
                buy_threshold = 5
    else:
        buy_threshold = threshold

    # Sector downgrade: if ticker's sector has negative sentiment, require +1
    if ctx:
        try:
            ticker_sector = get_ticker_sector(ticker)
            sentiment_sectors = ctx["sentiment"].get("affected_sectors", [])
            news_sentiment = ctx["sentiment"].get("news_sentiment", "NETRAL")
            if ticker_sector in sentiment_sectors and news_sentiment == "NEGATIF":
                buy_threshold = min(buy_threshold + 1, 7)
        except Exception:
            pass

    # Gap 2: Market gate CAUTIOUS → threshold +1
    if ctx:
        try:
            market_gate = ctx.get("market_gate", "OPEN")
            if market_gate == "CAUTIOUS":
                buy_threshold = min(buy_threshold + 1, 7)
        except Exception:
            pass

    # Gap 3: Premarket prediction NEGATIF → threshold +1
    if ctx:
        try:
            premarket = ctx.get("premarket", {})
            prediction = premarket.get("ihsg_open_prediction", "NETRAL")
            if prediction == "NEGATIF":
                buy_threshold = min(buy_threshold + 1, 7)
        except Exception:
            pass

    # If no foreign flow data available, cap threshold at 6 (max non-foreign score).
    # Spec: "5/7 ideally, but 5/6 still valid if foreign data unavailable"
    if not foreign_flow:
        buy_threshold = min(buy_threshold, 6)

    # Signal levels
    if score >= buy_threshold:
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
        "buy_threshold": int(buy_threshold),
        "conditions": conditions,
        "approaching_ara": bool(approaching_ara),
        "near_arb": bool(near_arb),
        "signal": signal,
        "reasons": reasons,
        "bb_upper": round(float(bb_upper), 0),
        "bb_lower": round(float(bb_lower), 0),
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

    # Load context once for all tickers in this scan
    ctx = None
    threshold = 5
    try:
        from agents.market_context import get_context, get_dynamic_threshold
        ctx = get_context()
        threshold = get_dynamic_threshold()
        print(f"[Scanner] Market mode: {ctx.get('market_mode', 'NORMAL')}, threshold: {threshold}/7")
    except Exception as e:
        print(f"[Scanner] Could not load market context: {e}")

    # Gap 1: Fetch foreign flow ONCE for all tickers
    foreign_flow = {}
    try:
        from agents.foreign_flow import fetch_foreign_flow, save_flow_data
        from agents.market_context import update_foreign_flow
        foreign_flow = fetch_foreign_flow()
        if foreign_flow:
            save_flow_data(foreign_flow)
            # Compute overall foreign signal for context
            watchlist_clean = [t.replace(".JK", "") for t in WATCHLIST]
            net_values = [
                foreign_flow[t]["net_foreign"]
                for t in watchlist_clean
                if t in foreign_flow
            ]
            if net_values:
                avg_net = sum(net_values) / len(net_values)
                if avg_net > 0:
                    ff_signal = "BULLISH"
                elif avg_net < 0:
                    ff_signal = "BEARISH"
                else:
                    ff_signal = "NEUTRAL"
                top_bought = sorted(
                    [t for t in watchlist_clean if t in foreign_flow and foreign_flow[t]["net_foreign"] > 0],
                    key=lambda t: -foreign_flow[t]["net_foreign"],
                )[:3]
                top_sold = sorted(
                    [t for t in watchlist_clean if t in foreign_flow and foreign_flow[t]["net_foreign"] < 0],
                    key=lambda t: foreign_flow[t]["net_foreign"],
                )[:3]
                update_foreign_flow(ff_signal, top_bought, top_sold)
    except Exception as e:
        print(f"[Scanner] Foreign flow fetch error: {e}")

    # Gap 2: Calculate breadth after scanning (use results from previous scan if available)
    # Breadth will be calculated after the loop with fresh results

    for ticker in WATCHLIST:
        try:
            df = get_stock_data(ticker)
            signal = compute_signals(ticker, df, ctx=ctx, threshold=threshold, foreign_flow=foreign_flow)
            if signal:
                results.append(signal)
                ff_tag = "✅" if signal["conditions"].get("foreign_net_buy") else "❌"
                print(f"  {ticker}: {signal['signal']} (score={signal['score']}/{signal.get('buy_threshold', 5)}, "
                      f"RSI={signal['rsi']}, vol={signal['vol_ratio']}x, "
                      f"MACD={'✅' if signal['macd_bullish'] else '❌'}, "
                      f"Asing={ff_tag})")
        except Exception as e:
            print(f"[Scanner] Error processing {ticker}: {e}")

    # Gap 2: Calculate breadth from fresh scan results
    try:
        from agents.market_breadth import calculate_breadth, fetch_ihsg_data, check_market_gate
        from agents.market_context import update_market_breadth
        breadth = calculate_breadth(results)
        ihsg_data = fetch_ihsg_data()
        gate = check_market_gate(ihsg_data, breadth)
        ihsg_change = ihsg_data.get("change_from_open_pct", 0) if ihsg_data else 0
        update_market_breadth(gate["gate"], breadth["breadth_pct"], ihsg_change)
        print(f"[Scanner] Market gate: {gate['gate']}, breadth: {breadth['breadth_pct']:.0f}%")
    except Exception as e:
        print(f"[Scanner] Market breadth calculation error: {e}")

    # Sort: STRONG BUY first, then WATCH, then by score
    results.sort(key=lambda x: (-x["score"], x["signal"]))
    return results


def format_morning_alert(signals: list, macro: dict = None, market_context: str = None, foreign_flow: dict = None) -> str:
    """Format morning briefing Telegram message with macro + trade setup + market context"""
    strong_buy_signals = [s for s in signals if s["signal"] == "STRONG BUY"]
    watch_signals = [s for s in signals if s["signal"] == "WATCH"]

    now_wib = datetime.utcnow() + timedelta(hours=7)

    # Load shared market context for enriched output
    ctx = None
    try:
        from agents.market_context import get_context, format_context_summary
        ctx = get_context()
        mode = ctx.get("market_mode", "NORMAL")
    except Exception:
        mode = "NORMAL"

    mode_label = {
        "RISK_OFF": "🔴 RISK OFF",
        "CAUTIOUS": "🟡 CAUTIOUS",
        "NORMAL": "⚪ NORMAL",
        "RISK_ON": "🟢 RISK ON",
    }.get(mode, "⚪ NORMAL")

    threshold = strong_buy_signals[0].get("buy_threshold", 5) if strong_buy_signals else 5

    lines = [
        f"🏦 <b>MORNING SCAN — {now_wib.strftime('%d %b %Y %H:%M')} WIB</b>",
        f"<i>Hedge fund mode: {mode_label} | STRONG BUY threshold: {threshold}/7</i>",
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
            buy_thr = s.get("buy_threshold", 5)

            # Trading style
            style_info = classify_trading_style(s)
            style_tag = f"\n{style_info['style']}" if style_info.get("style") else ""

            lines.append(f"")
            lines.append(f"{whale}<b>{ticker_clean}</b> — STRONG BUY ({s['score']}/{buy_thr}){style_tag}")
            lines.append(f"📍 Entry : Rp {s['current']:,.0f}")

            # TP/SL: show intraday + swing if available, else default
            if style_info.get("intraday_tp") and style_info.get("swing_tp"):
                lines.append(
                    f"🎯 TP Intraday: Rp {style_info['intraday_tp']:,.0f} | "
                    f"TP Swing: Rp {style_info['swing_tp']:,.0f}"
                )
                lines.append(
                    f"🛡 SL Intraday: Rp {style_info['intraday_sl']:,.0f} | "
                    f"SL Swing: Rp {style_info['swing_sl']:,.0f}"
                )
            else:
                lines.append(f"🎯 TP    : Rp {s['tp']:,.0f} (+{s.get('tp_pct', 8.0):.0f}%)")
                lines.append(f"🛡 SL    : Rp {s['sl']:,.0f} ({s.get('sl_pct', -4.0):.0f}%)")

            lines.append(
                f"📊 RSI {s['rsi']} | Vol {s['vol_ratio']}x | MACD {macd_tag} | MA {ma_tag}"
            )

            # Gap 1: Foreign flow line
            if foreign_flow:
                clean_t = s["ticker"].replace(".JK", "").upper()
                flow = foreign_flow.get(clean_t, {})
                net = flow.get("net_foreign", 0)
                if net != 0:
                    net_juta = net / 1_000_000
                    if net > 0:
                        lines.append(f"🏦 Asing: Net Buy Rp {net_juta:.1f} juta ✅")
                    else:
                        lines.append(f"🏦 Asing: Net Sell Rp {abs(net_juta):.1f} juta ⚠️")

            # Context lines for this ticker's sector
            if ctx:
                ticker_sector = get_ticker_sector(s["ticker"])
                ctx_line = format_context_line(ctx, ticker_sector)
                if ctx_line:
                    lines.append(ctx_line)

                # Signal validation
                is_valid, reason = validate_signal(s, ctx)
                if not is_valid:
                    lines.append(f"⛔ FILTERED: {reason}")
                elif reason and reason != "OK" and "No context" not in reason:
                    lines.append(f"ℹ️ {reason}")

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


def classify_trading_style(signal: dict) -> dict:
    """
    Classify a signal as INTRADAY and/or SWING trade.

    INTRADAY criteria (need 3/4):
    - Volume > 1.5x avg
    - RSI between 35–62
    - Daily change > 0%
    - Price NOT at upper Bollinger Band (price < upper_band * 0.99)

    SWING criteria (INTRADAY + 2 more):
    - MACD bullish
    - MA20 > MA50 (uptrend)

    Tags only assigned when score >= 5.
    """
    empty = {
        "style": None,
        "intraday_met": False,
        "swing_met": False,
        "intraday_tp": None,
        "intraday_sl": None,
        "swing_tp": None,
        "swing_sl": None,
    }

    if not signal or signal.get("score", 0) < 5:
        return empty

    entry = float(signal.get("current", 0))
    if entry <= 0:
        return empty

    bb_upper = float(signal.get("bb_upper", float("inf")))
    if bb_upper <= 0:
        bb_upper = float("inf")

    # INTRADAY criteria (4 checks, need ≥3)
    intraday_checks = {
        "volume_1_5x": float(signal.get("vol_ratio", 0)) > 1.5,
        "rsi_range": 35 <= float(signal.get("rsi", 0)) <= 62,
        "positive_change": float(signal.get("daily_change_pct", 0)) > 0,
        "not_near_upper_bb": entry < bb_upper * 0.99,
    }
    intraday_met = sum(intraday_checks.values()) >= 3

    # SWING extra criteria (need INTRADAY + both below)
    swing_extra = {
        "macd_bullish": bool(signal.get("macd_bullish", False)),
        "uptrend": bool(signal.get("conditions", {}).get("uptrend", False)),
    }
    swing_met = intraday_met and all(swing_extra.values())

    # Tag assignment
    if intraday_met and swing_met:
        style = "🏃 INTRADAY + 📅 SWING"
    elif intraday_met:
        style = "🏃 INTRADAY"
    elif swing_met:
        style = "📅 SWING"
    else:
        style = None

    result = {
        "style": style,
        "intraday_met": intraday_met,
        "swing_met": swing_met,
        "intraday_checks": intraday_checks,
        "swing_extra": swing_extra,
        "intraday_tp": None,
        "intraday_sl": None,
        "swing_tp": None,
        "swing_sl": None,
    }

    if intraday_met:
        result["intraday_tp"] = round(entry * 1.025, 0)
        result["intraday_sl"] = round(entry * 0.98, 0)

    if swing_met:
        result["swing_tp"] = round(entry * 1.08, 0)
        result["swing_sl"] = round(entry * 0.96, 0)

    return result


def scan_for_sell_signals(prev_signals: list = None) -> list:
    """
    Scan for SELL signals — runs every 10 min during market hours.

    STRONG SELL triggers (need 3/4):
    - RSI > 72 (overbought, whale distributing)
    - MACD bearish crossover (last 3 candles)
    - Price < MA20 (support broken)
    - Volume > 2x avg (distribution volume)

    SECTOR COLLAPSE:
    - 3+ stocks in same sector all down > 1.5% from open
    """
    tickers_to_scan = [s["ticker"] for s in prev_signals] if prev_signals else WATCHLIST

    sell_signals = []
    sector_changes = {sector: [] for sector in SECTOR_MAP}

    now_wib = datetime.utcnow() + timedelta(hours=7)
    scan_time = now_wib.strftime("%H:%M")

    for ticker in tickers_to_scan:
        try:
            df = get_stock_data(ticker)
            if df is None or df.empty or len(df) < 20:
                continue

            close = df["Close"]
            volume = df["Volume"]

            current = float(close.iloc[-1])
            open_today = float(df["Open"].iloc[-1])
            change_from_open = ((current - open_today) / open_today) * 100

            # MA20
            ma20 = float(close.rolling(20).mean().iloc[-1])

            # RSI (14)
            delta_c = close.diff()
            gain = delta_c.clip(lower=0).rolling(14).mean()
            loss = (-delta_c.clip(upper=0)).rolling(14).mean()
            rs = gain / loss
            rsi = float((100 - (100 / (1 + rs))).iloc[-1])

            # Volume
            avg_vol_5d = float(volume.iloc[-6:-1].mean()) if len(volume) >= 6 else float(volume.mean())
            vol_ratio = float(volume.iloc[-1]) / avg_vol_5d if avg_vol_5d > 0 else 1.0

            # MACD bearish crossover (last 3 candles)
            macd_line, signal_line = compute_macd(close)
            macd_bearish_cross = False
            if len(macd_line) >= 4:
                ml = macd_line.iloc[-4:].values
                sl_v = signal_line.iloc[-4:].values
                for i in range(1, len(ml)):
                    if not np.isnan(ml[i]) and not np.isnan(sl_v[i]):
                        if ml[i] < sl_v[i] and ml[i - 1] >= sl_v[i - 1]:
                            macd_bearish_cross = True
                            break

            # Sector tracking
            for sector, sector_tickers in SECTOR_MAP.items():
                if ticker in sector_tickers:
                    sector_changes[sector].append({
                        "ticker": ticker,
                        "change_from_open": change_from_open,
                    })

            # STRONG SELL conditions
            from_ma20_pct = ((current - ma20) / ma20) * 100
            sell_conditions = {
                "rsi_overbought": bool(rsi > 72),
                "macd_bearish": bool(macd_bearish_cross),
                "price_below_ma20": bool(current < ma20),
                "distribution_volume": bool(vol_ratio > 2.0),
            }

            if sum(sell_conditions.values()) >= 3:
                sell_signals.append({
                    "ticker": ticker,
                    "current": round(current, 0),
                    "from_ma20_pct": round(from_ma20_pct, 1),
                    "scan_time": scan_time,
                    "rsi": round(rsi, 1),
                    "vol_ratio": round(vol_ratio, 2),
                    "macd_bearish": bool(macd_bearish_cross),
                    "conditions": sell_conditions,
                    "sell_type": "STRONG_SELL",
                })

        except Exception as e:
            print(f"[Scanner] Sell scan error {ticker}: {e}")

    # Sector collapse check
    for sector, changes in sector_changes.items():
        down_stocks = [c for c in changes if c["change_from_open"] <= -1.5]
        if len(down_stocks) >= 3:
            sell_signals.append({
                "sector": sector,
                "stocks": down_stocks,
                "sell_type": "SECTOR_COLLAPSE",
            })

    return sell_signals


def format_sell_alert(sell_signals: list) -> str:
    """Format SELL signal alerts for Telegram."""
    if not sell_signals:
        return None

    parts = []
    for s in sell_signals:
        if s["sell_type"] == "STRONG_SELL":
            ticker_clean = s["ticker"].replace(".JK", "")
            cond = s["conditions"]
            lines = [
                f"🔴 <b>STRONG SELL — {ticker_clean}</b>",
                "",
                f"📍 Harga: Rp {s['current']:,.0f}",
                f"📉 Dari MA20: {s['from_ma20_pct']:+.1f}%",
                f"⏰ {s['scan_time']} WIB",
                "",
            ]
            if cond.get("rsi_overbought"):
                lines.append(f"❌ RSI {s['rsi']} (distribusi)")
            if cond.get("macd_bearish"):
                lines.append("❌ MACD bearish crossover")
            if cond.get("price_below_ma20"):
                lines.append("❌ Break di bawah MA20")
            if cond.get("distribution_volume"):
                lines.append(f"❌ Volume {s['vol_ratio']}x (whale jual)")
            lines.append("")
            lines.append("💡 Konfirmasi kuat: keluar sekarang.")
            lines.append("⚠️ BUKAN saran investasi. DYOR.")
            parts.append("\n".join(lines))

        elif s["sell_type"] == "SECTOR_COLLAPSE":
            sector = s.get("sector", "Unknown")
            tickers_str = ", ".join(
                t["ticker"].replace(".JK", "") for t in s.get("stocks", [])
            )
            lines = [
                f"🔴 <b>SECTOR COLLAPSE — {sector}</b>",
                "",
                "📉 3+ saham turun >1.5% dari open:",
                f"   {tickers_str}",
                "",
                "⚠️ Pertimbangkan reduce exposure sektor ini.",
                "⚠️ BUKAN saran investasi. DYOR.",
            ]
            parts.append("\n".join(lines))

    return "\n\n".join(parts) if parts else None


def run_sell_scan(prev_signals: list = None) -> list:
    """Run sell signal scan and send alert if any found."""
    try:
        # Load morning signals if not provided
        if prev_signals is None:
            path = "data/signals_today.json"
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                prev_signals = data.get("signals", [])

        sell_signals = scan_for_sell_signals(prev_signals)

        if sell_signals:
            msg = format_sell_alert(sell_signals)
            if msg:
                ok = send_telegram(msg)
                print(f"[Scanner] Sell alert sent: {ok} ({len(sell_signals)} signals)")
        else:
            now_wib = datetime.utcnow() + timedelta(hours=7)
            print(f"[Scanner] No sell signals at {now_wib.strftime('%H:%M')} WIB")

        return sell_signals
    except Exception as e:
        print(f"[Scanner] run_sell_scan ERROR: {e}")
        return []


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

    # Load foreign flow data for alert enrichment
    foreign_flow = {}
    try:
        flow_path = "data/foreign_flow_today.json"
        if os.path.exists(flow_path):
            with open(flow_path) as f:
                flow_data = json.load(f)
            foreign_flow = flow_data.get("data", {})
    except Exception as e:
        print(f"[Scanner] Foreign flow load error: {e}")

    msg = format_morning_alert(signals, macro, market_context, foreign_flow=foreign_flow)
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
    """
    Run at 15:35 WIB — after market close.
    EOD Delta Report: exact format with Win/Loss/Accuracy summary.
    """
    path = "data/signals_today.json"
    if not os.path.exists(path):
        print("[Scanner] No morning signals found for delta report")
        return

    with open(path) as f:
        data = json.load(f)

    morning_signals = {
        s["ticker"]: s for s in data["signals"]
        if s["signal"] in ["STRONG BUY", "WATCH"]
    }

    if not morning_signals:
        return

    now_wib = datetime.utcnow() + timedelta(hours=7)
    date_str = now_wib.strftime("%d %b %Y")

    lines = [
        f"📊 <b>DAILY DELTA REPORT — {date_str}</b>",
        "",
    ]

    wins = 0
    losses = 0

    for ticker, morning in morning_signals.items():
        try:
            df = get_stock_data(ticker, period="2d")
            if df.empty:
                continue
            close_price = float(df["Close"].iloc[-1])
            open_price = float(df["Open"].iloc[-1])
            entry = float(morning["current"])

            delta = ((close_price - open_price) / open_price) * 100
            vs_pagi = ((close_price - entry) / entry) * 100

            emoji = "🟢" if delta > 0 else "🔴"
            if delta > 0:
                wins += 1
            else:
                losses += 1

            ticker_clean = ticker.replace(".JK", "")
            lines.append(
                f"{emoji} <b>{ticker_clean}</b> "
                f"Open: {open_price:,.0f} → Close: {close_price:,.0f} "
                f"| Delta: {delta:+.2f}% "
                f"| vs Pagi: {vs_pagi:+.2f}%"
            )
        except Exception as e:
            print(f"[Scanner] Error in closing report for {ticker}: {e}")

    total = wins + losses
    accuracy = round((wins / total) * 100, 1) if total > 0 else 0.0

    lines.append("")
    lines.append(
        f"📈 Win: {wins} sinyal profit | Loss: {losses} | Akurasi: {accuracy:.1f}%"
    )
    lines.append("")
    lines.append("⚠️ <i>BUKAN saran investasi. DYOR.</i>")

    send_telegram("\n".join(lines))
    print("[Scanner] Closing delta report sent")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "close":
        run_closing_report()
    elif len(sys.argv) > 1 and sys.argv[1] == "realtime":
        run_realtime_scan()
    elif len(sys.argv) > 1 and sys.argv[1] == "sell":
        run_sell_scan()
    else:
        run_morning_scan()
