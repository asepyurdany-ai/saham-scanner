"""
Market Context — Shared Intelligence System for all agents.
Thread-safe read/write to data/market_context.json.

All agents share state via this module:
  - Radar      → update_macro(), update_geo()
  - Sentinel   → update_sentiment()
  - SelfImprover → update_performance()
  - Scanner    → get_context(), get_dynamic_threshold()
"""

import copy
import json
import os
import tempfile
import threading
from datetime import datetime

CONTEXT_FILE = "data/market_context.json"

DEFAULT_CONTEXT = {
    "updated_at": None,
    "market_mode": "NORMAL",  # RISK_OFF | CAUTIOUS | NORMAL | RISK_ON
    "macro": {
        "rupiah_change_pct": 0.0,
        "oil_change_pct": 0.0,
        "gold_change_pct": 0.0,
        "macro_shock_active": False,
        "macro_signal": "NEUTRAL",  # BULLISH | NEUTRAL | BEARISH
    },
    "sentiment": {
        "news_sentiment": "NETRAL",  # POSITIF | NEGATIF | NETRAL
        "confidence": "SEDANG",
        "affected_sectors": [],
        "affected_tickers": [],
        "summary": "",
    },
    "geo": {
        "risk_level": "LOW",  # LOW | MEDIUM | HIGH
        "active_events": [],
        "geo_signal": "NEUTRAL",  # NEUTRAL | CAUTIOUS | RISK_OFF
    },
    "performance": {
        "recent_win_rate": None,
        "buy_threshold_override": None,  # override 4/5/6
        "best_rsi_zone": [35, 62],
        "macd_weight": 1.0,
        "volume_weight": 1.0,
    },
}

# Module-level lock for thread safety
_lock = threading.Lock()


def load_context() -> dict:
    """
    Load context from file. Returns deep copy of DEFAULT_CONTEXT if file
    is missing or corrupt. Never raises — graceful degradation.
    """
    with _lock:
        os.makedirs("data", exist_ok=True)
        if not os.path.exists(CONTEXT_FILE):
            return copy.deepcopy(DEFAULT_CONTEXT)
        try:
            with open(CONTEXT_FILE) as f:
                ctx = json.load(f)
            # Merge with defaults to fill any missing keys (forward compat)
            merged = copy.deepcopy(DEFAULT_CONTEXT)
            for key in merged:
                if key in ctx:
                    if isinstance(merged[key], dict) and isinstance(ctx[key], dict):
                        merged[key].update(ctx[key])
                    else:
                        merged[key] = ctx[key]
            return merged
        except Exception:
            return copy.deepcopy(DEFAULT_CONTEXT)


def save_context(ctx: dict):
    """
    Save context with updated_at timestamp.
    Uses atomic write (write tmp → rename) to prevent partial reads.
    """
    with _lock:
        os.makedirs("data", exist_ok=True)
        ctx["updated_at"] = datetime.utcnow().isoformat()
        # Atomic write
        abs_path = os.path.abspath(CONTEXT_FILE)
        dir_name = os.path.dirname(abs_path)
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(ctx, f, indent=2)
            os.replace(tmp_path, abs_path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            raise


def update_macro(rupiah_pct: float, oil_pct: float, gold_pct: float, shock_active: bool):
    """
    Update macro commodity data and persist.
    Called by Radar after run_commodity_check().

    Derives macro_signal:
    - oil < -2% AND rupiah < -1% → BEARISH
    - else → NEUTRAL
    """
    # Derive macro_signal
    if float(oil_pct) < -2.0 and float(rupiah_pct) < -1.0:
        macro_signal = "BEARISH"
    else:
        macro_signal = "NEUTRAL"

    ctx = load_context()
    ctx["macro"]["rupiah_change_pct"] = float(rupiah_pct)
    ctx["macro"]["oil_change_pct"] = float(oil_pct)
    ctx["macro"]["gold_change_pct"] = float(gold_pct)
    ctx["macro"]["macro_shock_active"] = bool(shock_active)
    ctx["macro"]["macro_signal"] = macro_signal
    ctx["market_mode"] = compute_market_mode(ctx)
    save_context(ctx)
    print(
        f"[MarketContext] Macro updated: rupiah={rupiah_pct:+.2f}%, "
        f"oil={oil_pct:+.2f}%, gold={gold_pct:+.2f}%, "
        f"shock={shock_active}, signal={macro_signal}"
    )


def update_sentiment(
    sentiment: str,
    confidence: str,
    sectors: list,
    tickers: list,
    summary: str,
):
    """
    Update news sentiment and persist.
    Called by Sentinel after run_sentinel().
    """
    ctx = load_context()
    ctx["sentiment"]["news_sentiment"] = str(sentiment)
    ctx["sentiment"]["confidence"] = str(confidence)
    ctx["sentiment"]["affected_sectors"] = list(sectors)
    ctx["sentiment"]["affected_tickers"] = list(tickers)
    ctx["sentiment"]["summary"] = str(summary)
    ctx["market_mode"] = compute_market_mode(ctx)
    save_context(ctx)
    print(
        f"[MarketContext] Sentiment updated: {sentiment} ({confidence}), "
        f"sectors={sectors}"
    )


def update_geo(risk_level: str, active_events: list, geo_signal: str):
    """
    Update geopolitical risk data and persist.
    Called by Radar after run_geo_check().
    """
    ctx = load_context()
    ctx["geo"]["risk_level"] = str(risk_level)
    ctx["geo"]["active_events"] = list(active_events)
    ctx["geo"]["geo_signal"] = str(geo_signal)
    ctx["market_mode"] = compute_market_mode(ctx)
    save_context(ctx)
    print(
        f"[MarketContext] Geo updated: risk={risk_level}, "
        f"signal={geo_signal}, events={len(active_events)}"
    )


def update_performance(win_rate, threshold_override, rsi_zone=None):
    """
    Update performance metrics and persist.
    Called by SelfImprover after analyze_performance().

    Args:
        win_rate: float 0.0–1.0 or None
        threshold_override: int (4/5/6) or None
        rsi_zone: [low, high] or None
    """
    ctx = load_context()
    ctx["performance"]["recent_win_rate"] = win_rate
    ctx["performance"]["buy_threshold_override"] = threshold_override
    if rsi_zone is not None:
        ctx["performance"]["best_rsi_zone"] = list(rsi_zone)
    save_context(ctx)
    print(
        f"[MarketContext] Performance updated: win_rate={win_rate}, "
        f"threshold_override={threshold_override}"
    )


def compute_market_mode(ctx: dict = None) -> str:
    """
    Aggregate all signals to compute market mode.

    Rules (priority order):
    1. macro_shock_active OR geo risk HIGH → RISK_OFF
    2. geo MEDIUM + sentiment NEGATIF → CAUTIOUS
    3. sentiment POSITIF + macro BULLISH → RISK_ON
    4. else → NORMAL
    """
    if ctx is None:
        ctx = load_context()

    macro_shock = bool(ctx["macro"].get("macro_shock_active", False))
    geo_risk = ctx["geo"].get("risk_level", "LOW")
    sentiment = ctx["sentiment"].get("news_sentiment", "NETRAL")
    macro_signal = ctx["macro"].get("macro_signal", "NEUTRAL")

    if macro_shock or geo_risk == "HIGH":
        return "RISK_OFF"
    if geo_risk == "MEDIUM" and sentiment == "NEGATIF":
        return "CAUTIOUS"
    if sentiment == "POSITIF" and macro_signal == "BULLISH":
        return "RISK_ON"
    return "NORMAL"


def get_dynamic_threshold() -> int:
    """
    Returns BUY threshold based on market_mode.
    Respects performance.buy_threshold_override if set.

    RISK_OFF  → 6
    CAUTIOUS  → 6
    NORMAL    → 5
    RISK_ON   → 4
    """
    ctx = load_context()

    # Performance override takes priority
    perf_override = ctx["performance"].get("buy_threshold_override")
    if perf_override is not None:
        return int(perf_override)

    mode = ctx.get("market_mode", "NORMAL")
    if mode in ("RISK_OFF", "CAUTIOUS"):
        return 6
    if mode == "RISK_ON":
        return 4
    return 5  # NORMAL


def get_context() -> dict:
    """
    Returns current context with fresh market_mode computation.
    Safe to call frequently — loads from file each time.
    """
    ctx = load_context()
    ctx["market_mode"] = compute_market_mode(ctx)
    return ctx


def format_context_summary() -> str:
    """Short HTML summary of current market context for alert injection."""
    ctx = get_context()
    mode = ctx.get("market_mode", "NORMAL")
    macro = ctx["macro"]
    sentiment = ctx["sentiment"]
    geo = ctx["geo"]

    mode_emoji = {
        "RISK_OFF": "🔴",
        "CAUTIOUS": "🟡",
        "NORMAL": "⚪",
        "RISK_ON": "🟢",
    }.get(mode, "⚪")

    lines = [
        f"{mode_emoji} Market Mode: <b>{mode}</b>",
        (
            f"🌍 Macro: Rupiah {macro['rupiah_change_pct']:+.2f}% | "
            f"Oil {macro['oil_change_pct']:+.2f}% | "
            f"Gold {macro['gold_change_pct']:+.2f}%"
        ),
        f"📰 Sentimen: {sentiment['news_sentiment']} ({sentiment['confidence']})",
        f"🌐 Geo Risk: {geo['risk_level']} ({geo['geo_signal']})",
    ]

    if macro.get("macro_shock_active"):
        lines.append("⚠️ <b>MACRO SHOCK AKTIF</b>")

    if sentiment.get("summary"):
        lines.append(f"💬 {sentiment['summary'][:100]}")

    return "\n".join(lines)
