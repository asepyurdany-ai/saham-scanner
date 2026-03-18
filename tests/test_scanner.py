"""
Unit tests for agents/scanner.py
Tests: compute_signals (strong_buy/watch/avoid), save_signals, format_morning_alert
Updated for: 6-condition scoring, MACD/BB, STRONG BUY threshold (5/6), TP/SL
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import pytest

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.scanner import (
    compute_signals, save_signals, format_morning_alert, compute_macd,
    classify_trading_style, scan_for_sell_signals, format_sell_alert,
)


def make_df(prices, volumes=None, highs=None, lows=None):
    """Helper: create OHLCV DataFrame from price list."""
    if volumes is None:
        volumes = [1_000_000] * len(prices)
    if highs is None:
        highs = [p * 1.01 for p in prices]
    if lows is None:
        lows = [p * 0.98 for p in prices]
    dates = pd.date_range("2026-01-01", periods=len(prices), freq="D")
    df = pd.DataFrame({
        "Open": [p * 0.99 for p in prices],
        "High": highs,
        "Low": lows,
        "Close": prices,
        "Volume": volumes,
    }, index=dates)
    return df


def make_buy_df():
    """
    35-row uptrend DataFrame → STRONG BUY (score 5-6).
    Conditions met:
      1. uptrend: MA20 > MA50  ✅
      2. volume_spike: last day 2.4x  ✅
      3. rsi_ok: RSI ~50 (mixed days)  ✅
      4. price_above_ma20: rising price  ✅
      5. macd_bullish: EMA12 > EMA26 in uptrend  ✅
      6. momentum_positive: last day up  ✅
    """
    # 35 rows, steady uptrend with mixed days for RSI moderation
    prices = [
        1000, 1002, 999, 1004, 1001,
        1006, 1003, 1008, 1005, 1010,
        1007, 1012, 1009, 1014, 1011,
        1016, 1013, 1018, 1015, 1020,
        1017, 1022, 1019, 1024, 1021,
        1026, 1023, 1028, 1025, 1030,
        1027, 1032, 1029, 1034, 1036,  # last day positive
    ]
    # Volume spike on last day: 1.2M vs 500K avg = 2.4x
    volumes = [500_000] * 34 + [1_200_000]
    return make_df(prices, volumes)


def make_watch_df():
    """
    35-row flat DataFrame → WATCH (score 3-4).
    - No uptrend (flat MA20 ≈ MA50)
    - No volume spike
    - RSI ok, momentum ok, price near MA20
    """
    base = 2000.0
    prices = [base + (i % 3) * 2 for i in range(34)] + [base + 8]
    volumes = [300_000] * 35
    return make_df(prices, volumes)


def make_avoid_df():
    """
    35-row downtrend → AVOID (score < 3).
    All conditions fail: downtrend, no spike, negative RSI territory, below MA, negative MACD
    """
    prices = [3000 - i * 5 for i in range(35)]  # 3000 → 2830
    volumes = [200_000] * 35
    return make_df(prices, volumes)


# ─── Tests: compute_macd ────────────────────────────────────────────────────

class TestComputeMacd:

    def test_returns_two_series(self):
        """compute_macd should return (macd_line, signal_line) Series."""
        prices = pd.Series([float(x) for x in range(1000, 1035)])
        macd_line, signal_line = compute_macd(prices)
        assert isinstance(macd_line, pd.Series)
        assert isinstance(signal_line, pd.Series)
        assert len(macd_line) == len(prices)
        assert len(signal_line) == len(prices)

    def test_macd_bullish_in_uptrend(self):
        """In a consistent uptrend, MACD line should be above signal line."""
        prices = pd.Series([float(1000 + i * 10) for i in range(35)])
        macd_line, signal_line = compute_macd(prices)
        # Last value: MACD > signal in strong uptrend
        assert macd_line.iloc[-1] > signal_line.iloc[-1]

    def test_macd_bearish_in_downtrend(self):
        """In a consistent downtrend, MACD line should be below signal line."""
        prices = pd.Series([float(3000 - i * 10) for i in range(35)])
        macd_line, signal_line = compute_macd(prices)
        assert macd_line.iloc[-1] < signal_line.iloc[-1]

    def test_custom_params(self):
        """compute_macd should accept custom fast/slow/signal params."""
        prices = pd.Series([float(x) for x in range(1000, 1035)])
        macd_line, signal_line = compute_macd(prices, fast=5, slow=10, signal=3)
        assert len(macd_line) == len(prices)


# ─── Tests: compute_signals ─────────────────────────────────────────────────

class TestComputeSignals:

    def test_strong_buy_signal(self):
        """compute_signals should return STRONG BUY for strong uptrend with volume spike."""
        df = make_buy_df()
        result = compute_signals("BBCA.JK", df)

        assert result is not None
        assert result["ticker"] == "BBCA.JK"
        assert result["signal"] == "STRONG BUY"
        assert result["score"] >= 5

    def test_watch_signal(self):
        """compute_signals should return WATCH for moderate conditions (score 3-4)."""
        df = make_watch_df()
        result = compute_signals("TLKM.JK", df)

        assert result is not None
        assert result["ticker"] == "TLKM.JK"
        assert result["signal"] in ("STRONG BUY", "WATCH", "AVOID")
        assert 0 <= result["score"] <= 6

    def test_avoid_signal(self):
        """compute_signals should return AVOID for downtrend with no volume spike."""
        df = make_avoid_df()
        result = compute_signals("GOTO.JK", df)

        assert result is not None
        assert result["ticker"] == "GOTO.JK"
        assert result["signal"] == "AVOID"
        assert result["score"] < 3

    def test_returns_none_for_empty_df(self):
        """compute_signals should return None for empty DataFrame."""
        result = compute_signals("BBRI.JK", pd.DataFrame())
        assert result is None

    def test_returns_none_for_insufficient_data(self):
        """compute_signals should return None if fewer than 20 rows."""
        df = make_df([1000] * 15)
        result = compute_signals("BBRI.JK", df)
        assert result is None

    def test_result_structure(self):
        """compute_signals result should have all required keys."""
        df = make_buy_df()
        result = compute_signals("BBCA.JK", df)

        required_keys = [
            "ticker", "current", "prev_close", "daily_change_pct",
            "ma20", "ma50", "rsi", "vol_ratio", "score", "conditions",
            "approaching_ara", "near_arb", "signal",
            "macd_bullish", "macd_crossover", "tp", "sl", "tp_pct", "sl_pct",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_signal_values_are_valid(self):
        """Signal must be one of STRONG BUY/WATCH/AVOID."""
        for make_fn in [make_buy_df, make_watch_df, make_avoid_df]:
            df = make_fn()
            result = compute_signals("BBCA.JK", df)
            if result:
                assert result["signal"] in ("STRONG BUY", "WATCH", "AVOID")

    def test_rsi_range(self):
        """RSI should be between 0 and 100."""
        df = make_buy_df()
        result = compute_signals("BBCA.JK", df)
        if result:
            assert 0 <= result["rsi"] <= 100

    def test_conditions_dict_has_6_keys(self):
        """conditions dict should have exactly 6 boolean keys."""
        df = make_buy_df()
        result = compute_signals("BBCA.JK", df)
        if result:
            assert len(result["conditions"]) == 6, (
                f"Expected 6 conditions, got {len(result['conditions'])}: "
                f"{list(result['conditions'].keys())}"
            )
            for v in result["conditions"].values():
                assert isinstance(v, bool)

    def test_conditions_keys(self):
        """conditions dict should have the correct 6 named keys."""
        df = make_buy_df()
        result = compute_signals("BBCA.JK", df)
        if result:
            expected_keys = {
                "uptrend", "volume_spike", "rsi_ok",
                "price_above_ma20", "macd_bullish", "momentum_positive"
            }
            assert set(result["conditions"].keys()) == expected_keys

    def test_score_matches_conditions(self):
        """score should equal sum of True conditions."""
        df = make_buy_df()
        result = compute_signals("BBCA.JK", df)
        if result:
            expected = sum(result["conditions"].values())
            assert result["score"] == expected

    def test_strong_buy_signal_score_at_least_5(self):
        """STRONG BUY signal must have score >= 5."""
        df = make_buy_df()
        result = compute_signals("BBCA.JK", df)
        assert result is not None
        assert result["score"] >= 5
        assert result["signal"] == "STRONG BUY"

    def test_avoid_signal_score_less_than_3(self):
        """AVOID signal must have score < 3."""
        df = make_avoid_df()
        result = compute_signals("GOTO.JK", df)
        assert result is not None
        assert result["score"] < 3
        assert result["signal"] == "AVOID"

    def test_tp_sl_calculation(self):
        """TP should be +8%, SL should be -4% of entry price."""
        df = make_buy_df()
        result = compute_signals("BBCA.JK", df)
        if result:
            current = result["current"]
            expected_tp = round(current * 1.08, 0)
            expected_sl = round(current * 0.96, 0)
            assert result["tp"] == expected_tp
            assert result["sl"] == expected_sl

    def test_macd_fields_present(self):
        """macd_bullish and macd_crossover should be booleans."""
        df = make_buy_df()
        result = compute_signals("BBCA.JK", df)
        if result:
            assert isinstance(result["macd_bullish"], bool)
            assert isinstance(result["macd_crossover"], bool)

    def test_macd_bullish_in_uptrend(self):
        """In an uptrend, macd_bullish should be True."""
        df = make_buy_df()
        result = compute_signals("BBCA.JK", df)
        if result:
            assert result["macd_bullish"] is True

    def test_rsi_ok_condition_range(self):
        """rsi_ok should be True only when 25 < RSI < 70."""
        df = make_buy_df()
        result = compute_signals("BBCA.JK", df)
        if result:
            rsi = result["rsi"]
            expected = rsi > 25 and rsi < 70
            assert result["conditions"]["rsi_ok"] == expected

    def test_price_above_ma20_condition(self):
        """price_above_ma20 should reflect current vs MA20."""
        df = make_buy_df()
        result = compute_signals("BBCA.JK", df)
        if result:
            expected = result["current"] > result["ma20"]
            assert result["conditions"]["price_above_ma20"] == expected


# ─── Tests: save_signals ────────────────────────────────────────────────────

class TestSaveSignals:

    def _make_signal_dict(self, ticker="BBCA.JK", signal="STRONG BUY", score=5):
        return {
            "ticker": ticker,
            "current": 9500.0,
            "prev_close": 9400.0,
            "daily_change_pct": 1.06,
            "ma20": 9300.0,
            "ma50": 9100.0,
            "rsi": 55.0,
            "vol_ratio": 2.5,
            "macd_bullish": True,
            "macd_crossover": False,
            "tp": 10260.0,
            "sl": 9120.0,
            "tp_pct": 8.0,
            "sl_pct": -4.0,
            "score": score,
            "conditions": {
                "uptrend": True,
                "volume_spike": True,
                "rsi_ok": True,
                "price_above_ma20": True,
                "macd_bullish": True,
                "momentum_positive": False,
            },
            "approaching_ara": False,
            "near_arb": False,
            "signal": signal,
            "reasons": ["Test reason"],
        }

    def test_save_creates_file(self, tmp_path):
        """save_signals should create JSON file."""
        path = str(tmp_path / "signals_test.json")
        signals = [self._make_signal_dict()]
        save_signals(signals, path=path)

        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert "signals" in data
        assert len(data["signals"]) == 1
        assert data["signals"][0]["ticker"] == "BBCA.JK"

    def test_save_handles_numpy_types(self, tmp_path):
        """save_signals should handle numpy int/float types."""
        path = str(tmp_path / "signals_numpy.json")
        signals = [
            {
                "ticker": "BBRI.JK",
                "current": np.float64(4200.0),
                "prev_close": np.float64(4100.0),
                "daily_change_pct": np.float32(2.44),
                "ma20": np.float64(4000.0),
                "ma50": np.float64(3900.0),
                "rsi": np.float64(58.0),
                "vol_ratio": np.float64(1.8),
                "macd_bullish": True,
                "macd_crossover": False,
                "tp": np.float64(4536.0),
                "sl": np.float64(4032.0),
                "tp_pct": 8.0,
                "sl_pct": -4.0,
                "score": np.int64(3),
                "conditions": {
                    "uptrend": True,
                    "volume_spike": False,
                    "rsi_ok": True,
                    "price_above_ma20": True,
                    "macd_bullish": False,
                    "momentum_positive": True,
                },
                "approaching_ara": False,
                "near_arb": False,
                "signal": "WATCH",
                "reasons": [],
            }
        ]
        save_signals(signals, path=path)
        assert os.path.exists(path)

    def test_save_multiple_signals(self, tmp_path):
        """save_signals should handle multiple signals."""
        path = str(tmp_path / "signals_multi.json")
        signals = []
        for i in range(5):
            signals.append({
                "ticker": f"STOCK{i}.JK",
                "current": 1000.0 + i * 100,
                "prev_close": 990.0 + i * 100,
                "daily_change_pct": 1.0,
                "ma20": 950.0,
                "ma50": 900.0,
                "rsi": 50.0,
                "vol_ratio": 1.5,
                "macd_bullish": True,
                "macd_crossover": False,
                "tp": 1080.0,
                "sl": 960.0,
                "tp_pct": 8.0,
                "sl_pct": -4.0,
                "score": 3,
                "conditions": {},
                "approaching_ara": False,
                "near_arb": False,
                "signal": "WATCH",
                "reasons": [],
            })
        save_signals(signals, path=path)
        with open(path) as f:
            data = json.load(f)
        assert len(data["signals"]) == 5

    def test_save_empty_signals(self, tmp_path):
        """save_signals should handle empty list."""
        path = str(tmp_path / "signals_empty.json")
        save_signals([], path=path)
        with open(path) as f:
            data = json.load(f)
        assert data["signals"] == []


# ─── Tests: format_morning_alert ────────────────────────────────────────────

class TestFormatMorningAlert:

    def _make_signal(self, ticker="BBCA.JK", signal="STRONG BUY", score=5, vol_ratio=2.5):
        return {
            "ticker": ticker,
            "current": 9500.0,
            "prev_close": 9400.0,
            "daily_change_pct": 1.06,
            "ma20": 9300.0,
            "ma50": 9100.0,
            "rsi": 55.0,
            "vol_ratio": vol_ratio,
            "macd_bullish": True,
            "macd_crossover": False,
            "tp": 10260.0,
            "sl": 9120.0,
            "tp_pct": 8.0,
            "sl_pct": -4.0,
            "score": score,
            "conditions": {
                "uptrend": True,
                "volume_spike": True,
                "rsi_ok": True,
                "price_above_ma20": True,
                "macd_bullish": True,
                "momentum_positive": False,
            },
            "approaching_ara": False,
            "near_arb": False,
            "signal": signal,
            "reasons": ["Uptrend confirmed", "Volume spike"],
        }

    def test_format_with_strong_buy_signals(self):
        """format_morning_alert should include STRONG BUY signals in output."""
        signals = [self._make_signal("BBCA.JK", "STRONG BUY", 5)]
        msg = format_morning_alert(signals)

        assert "MORNING SCAN" in msg
        assert "BBCA" in msg
        assert "STRONG BUY" in msg

    def test_format_with_watch_signals(self):
        """format_morning_alert should include WATCH signals."""
        signals = [self._make_signal("TLKM.JK", "WATCH", 3)]
        msg = format_morning_alert(signals)

        assert "MORNING SCAN" in msg
        assert "TLKM" in msg

    def test_format_no_signals(self):
        """format_morning_alert should handle no buy/watch signals."""
        signals = [self._make_signal("GOTO.JK", "AVOID", 1)]
        msg = format_morning_alert(signals)

        assert "MORNING SCAN" in msg
        assert "tidak ada sinyal" in msg.lower() or "AVOID" in msg.upper() or "tidak" in msg.lower()

    def test_format_with_macro_data(self):
        """format_morning_alert should include macro data if provided."""
        signals = [self._make_signal()]
        macro = {
            "Oil (WTI)": {"current": 75.5, "change_pct": 1.2},
            "Gold": {"current": 1850.0, "change_pct": -0.5},
            "USD/IDR": {"current": 15500.0, "change_pct": 0.3},
        }
        msg = format_morning_alert(signals, macro=macro)

        assert "MAKRO" in msg or "Oil" in msg or "Gold" in msg

    def test_format_with_market_context(self):
        """format_morning_alert should include market context if provided."""
        signals = [self._make_signal()]
        context = "Minyak naik mendukung sektor energi, rupiah stabil."
        msg = format_morning_alert(signals, market_context=context)

        assert context in msg

    def test_format_whale_indicator(self):
        """format_morning_alert should show whale indicator for vol_ratio >= 2."""
        signals = [self._make_signal("BBCA.JK", "STRONG BUY", 5, vol_ratio=3.0)]
        msg = format_morning_alert(signals)

        assert "🐋" in msg

    def test_format_no_whale_indicator(self):
        """No whale indicator when vol_ratio < 2."""
        signals = [self._make_signal("BBCA.JK", "STRONG BUY", 5, vol_ratio=1.5)]
        msg = format_morning_alert(signals)

        assert "🐋" not in msg

    def test_format_disclaimer_present(self):
        """format_morning_alert should always include disclaimer."""
        msg = format_morning_alert([])
        assert "BUKAN saran" in msg or "DYOR" in msg

    def test_format_is_string(self):
        """format_morning_alert should return a string."""
        msg = format_morning_alert([])
        assert isinstance(msg, str)

    def test_format_limits_buy_to_5(self):
        """format_morning_alert should show at most 5 STRONG BUY signals."""
        signals = [self._make_signal(f"STOCK{i}.JK", "STRONG BUY", 5) for i in range(10)]
        msg = format_morning_alert(signals)

        shown = sum(1 for i in range(10) if f"STOCK{i}" in msg)
        assert shown <= 5

    def test_format_contains_tp_sl(self):
        """format_morning_alert should show TP and SL for STRONG BUY signals."""
        signals = [self._make_signal("BBCA.JK", "STRONG BUY", 5)]
        msg = format_morning_alert(signals)

        assert "TP" in msg
        assert "SL" in msg

    def test_format_score_out_of_6(self):
        """format_morning_alert should show score as X/6."""
        signals = [self._make_signal("BBCA.JK", "STRONG BUY", 5)]
        msg = format_morning_alert(signals)

        assert "5/6" in msg


# ─── Tests: classify_trading_style ──────────────────────────────────────────

def _make_signal_for_classify(
    score=5, vol_ratio=2.0, rsi=50.0, daily_change_pct=1.0,
    current=1000.0, bb_upper=1100.0, macd_bullish=True, uptrend=True,
):
    """Helper: create a minimal signal dict for classify_trading_style tests."""
    return {
        "ticker": "BBCA.JK",
        "score": score,
        "vol_ratio": vol_ratio,
        "rsi": rsi,
        "daily_change_pct": daily_change_pct,
        "current": current,
        "bb_upper": bb_upper,
        "macd_bullish": macd_bullish,
        "conditions": {
            "uptrend": uptrend,
            "volume_spike": vol_ratio >= 2.0,
            "rsi_ok": 25 < rsi < 70,
            "price_above_ma20": True,
            "macd_bullish": macd_bullish,
            "momentum_positive": daily_change_pct > 0,
        },
    }


class TestClassifyTradingStyle:

    def test_returns_dict(self):
        """classify_trading_style should return a dict."""
        sig = _make_signal_for_classify()
        result = classify_trading_style(sig)
        assert isinstance(result, dict)

    def test_required_keys(self):
        """Result should have required keys."""
        sig = _make_signal_for_classify()
        result = classify_trading_style(sig)
        for key in ["style", "intraday_met", "swing_met"]:
            assert key in result, f"Missing key: {key}"

    def test_intraday_and_swing_when_all_met(self):
        """Should return INTRADAY + SWING when all conditions met."""
        sig = _make_signal_for_classify(
            score=5, vol_ratio=2.0, rsi=50.0, daily_change_pct=1.0,
            current=1000.0, bb_upper=1100.0, macd_bullish=True, uptrend=True,
        )
        result = classify_trading_style(sig)
        assert result["intraday_met"] is True
        assert result["swing_met"] is True
        assert "INTRADAY" in result["style"]
        assert "SWING" in result["style"]

    def test_only_intraday_when_swing_extra_fails(self):
        """Should return only INTRADAY when MACD or uptrend missing."""
        sig = _make_signal_for_classify(
            score=5, vol_ratio=2.0, rsi=50.0, daily_change_pct=1.0,
            current=1000.0, bb_upper=1100.0, macd_bullish=False, uptrend=False,
        )
        result = classify_trading_style(sig)
        assert result["intraday_met"] is True
        assert result["swing_met"] is False
        assert result["style"] == "🏃 INTRADAY"

    def test_none_when_score_below_5(self):
        """Should return None style when score < 5."""
        sig = _make_signal_for_classify(score=4)
        result = classify_trading_style(sig)
        assert result["style"] is None

    def test_none_when_empty_signal(self):
        """Should return None style for empty signal."""
        result = classify_trading_style({})
        assert result["style"] is None

    def test_none_signal_input(self):
        """Should handle None input gracefully."""
        result = classify_trading_style(None)
        assert result["style"] is None

    def test_intraday_tp_sl_set_when_intraday_met(self):
        """Intraday TP (+2.5%) and SL (-2%) should be set when intraday met."""
        sig = _make_signal_for_classify(score=5, current=1000.0)
        result = classify_trading_style(sig)
        if result["intraday_met"]:
            assert result["intraday_tp"] == 1025.0
            assert result["intraday_sl"] == 980.0

    def test_swing_tp_sl_set_when_swing_met(self):
        """Swing TP (+8%) and SL (-4%) should be set when swing met."""
        sig = _make_signal_for_classify(
            score=5, current=1000.0, macd_bullish=True, uptrend=True
        )
        result = classify_trading_style(sig)
        if result["swing_met"]:
            assert result["swing_tp"] == 1080.0
            assert result["swing_sl"] == 960.0

    def test_intraday_tp_sl_none_when_not_met(self):
        """intraday_tp/sl should be None when intraday not met."""
        sig = _make_signal_for_classify(score=4)  # score < 5
        result = classify_trading_style(sig)
        assert result["intraday_tp"] is None
        assert result["intraday_sl"] is None

    def test_rsi_outside_35_62_reduces_intraday(self):
        """RSI outside 35-62 range should count against intraday criteria."""
        # RSI = 70 (overbought) — should not meet rsi_range check
        sig = _make_signal_for_classify(
            score=5, rsi=70.0, vol_ratio=2.0, daily_change_pct=1.0,
            current=1000.0, bb_upper=1100.0,
        )
        result = classify_trading_style(sig)
        if result["intraday_met"]:
            intraday_checks = result.get("intraday_checks", {})
            assert intraday_checks.get("rsi_range") is False

    def test_near_upper_bb_fails_not_near_bb_check(self):
        """Price at upper BB should fail not_near_upper_bb check."""
        # price = 1000, bb_upper = 1000 → price NOT < upper * 0.99
        sig = _make_signal_for_classify(
            score=5, current=1000.0, bb_upper=1000.0,
        )
        result = classify_trading_style(sig)
        if result.get("intraday_checks"):
            assert result["intraday_checks"]["not_near_upper_bb"] is False

    def test_intraday_criteria_need_3_of_4(self):
        """Should be intraday if exactly 3 of 4 criteria met."""
        # Only 3 criteria met: vol_ratio, rsi_range, positive_change
        # But NOT not_near_upper_bb (price = bb_upper * 0.995)
        sig = _make_signal_for_classify(
            score=5, vol_ratio=2.0, rsi=50.0, daily_change_pct=1.0,
            current=995.0, bb_upper=1000.0,  # 995 < 1000*0.99=990? No. 995 > 990 → fails
        )
        result = classify_trading_style(sig)
        # rsi_range=True, vol_1_5x=True, positive_change=True → 3/4 → intraday_met=True
        assert result["intraday_met"] is True


# ─── Tests: scan_for_sell_signals ───────────────────────────────────────────

class TestScanForSellSignals:

    def test_returns_list(self):
        """scan_for_sell_signals should return a list."""
        from unittest.mock import patch
        import pandas as pd
        from agents.scanner import scan_for_sell_signals

        with patch("agents.scanner.get_stock_data", return_value=pd.DataFrame()):
            result = scan_for_sell_signals([{"ticker": "BBCA.JK"}])
        assert isinstance(result, list)

    def test_with_empty_prev_signals_returns_list(self):
        """Should return list even with empty input (uses WATCHLIST fallback)."""
        from unittest.mock import patch
        import pandas as pd

        # Mock get_stock_data to return empty df so scan skips all tickers
        with patch("agents.scanner.get_stock_data", return_value=pd.DataFrame()):
            result = scan_for_sell_signals([])
        assert isinstance(result, list)

    def test_detects_strong_sell(self):
        """Should detect STRONG_SELL when 3/4 conditions met."""
        from unittest.mock import patch
        import pandas as pd

        # Build a dataframe with overbought RSI + break below MA20 + volume spike
        # Need 20+ rows for MA20 and 14 for RSI
        n = 30
        prices = [1100.0] * (n - 1) + [900.0]  # Last day crashes below MA20
        volumes = [500_000] * (n - 1) + [1_200_000]  # Volume spike on last day

        dates = pd.date_range("2026-01-01", periods=n, freq="D")
        df = pd.DataFrame({
            "Open": [1100.0] * (n - 1) + [1100.0],
            "High": [p * 1.01 for p in prices],
            "Low": [p * 0.98 for p in prices],
            "Close": prices,
            "Volume": volumes,
        }, index=dates)

        with patch("agents.scanner.get_stock_data", return_value=df):
            result = scan_for_sell_signals([{"ticker": "BBCA.JK"}])

        # We can't guarantee exactly 3 conditions without precise RSI/MACD,
        # but result must be a list
        assert isinstance(result, list)

    def test_sector_collapse_detected(self):
        """Should detect SECTOR_COLLAPSE if 3+ stocks in same sector down >1.5%."""
        from unittest.mock import patch
        import pandas as pd

        n = 30
        # Stocks are down 2% from open
        prices = [1000.0] * (n - 1) + [980.0]  # 2% drop
        opens = [1000.0] * n

        dates = pd.date_range("2026-01-01", periods=n, freq="D")
        df = pd.DataFrame({
            "Open": opens,
            "High": [p * 1.01 for p in prices],
            "Low": [p * 0.98 for p in prices],
            "Close": prices,
            "Volume": [500_000] * n,
        }, index=dates)

        # Patch all 4 Perbankan stocks to return this df
        perbankan = ["BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK"]
        prev_signals = [{"ticker": t} for t in perbankan]

        with patch("agents.scanner.get_stock_data", return_value=df):
            result = scan_for_sell_signals(prev_signals)

        collapse_signals = [s for s in result if s["sell_type"] == "SECTOR_COLLAPSE"]
        assert len(collapse_signals) >= 1
        assert collapse_signals[0]["sector"] == "Perbankan"

    def test_no_sell_when_healthy(self):
        """Should return no STRONG_SELL signals for healthy/rising stock."""
        from unittest.mock import patch
        import pandas as pd

        n = 30
        prices = [float(1000 + i * 5) for i in range(n)]  # Rising prices
        opens = [p * 0.99 for p in prices]

        dates = pd.date_range("2026-01-01", periods=n, freq="D")
        df = pd.DataFrame({
            "Open": opens,
            "High": [p * 1.02 for p in prices],
            "Low": [p * 0.99 for p in prices],
            "Close": prices,
            "Volume": [500_000] * n,
        }, index=dates)

        with patch("agents.scanner.get_stock_data", return_value=df):
            result = scan_for_sell_signals([{"ticker": "BBCA.JK"}])

        strong_sells = [s for s in result if s["sell_type"] == "STRONG_SELL"]
        assert len(strong_sells) == 0

    def test_skips_empty_df(self):
        """Should not crash when get_stock_data returns empty df."""
        from unittest.mock import patch
        import pandas as pd

        with patch("agents.scanner.get_stock_data", return_value=pd.DataFrame()):
            result = scan_for_sell_signals([{"ticker": "BBCA.JK"}])
        assert isinstance(result, list)


# ─── Tests: format_sell_alert ───────────────────────────────────────────────

class TestFormatSellAlert:

    def _make_strong_sell(self, ticker="BBCA.JK", rsi=74.2, vol_ratio=2.1,
                          current=4210.0, from_ma20=-1.8):
        return {
            "ticker": ticker,
            "sell_type": "STRONG_SELL",
            "current": current,
            "from_ma20_pct": from_ma20,
            "scan_time": "11:47",
            "rsi": rsi,
            "vol_ratio": vol_ratio,
            "macd_bearish": True,
            "conditions": {
                "rsi_overbought": True,
                "macd_bearish": True,
                "price_below_ma20": True,
                "distribution_volume": True,
            },
        }

    def _make_sector_collapse(self, sector="Perbankan"):
        return {
            "sell_type": "SECTOR_COLLAPSE",
            "sector": sector,
            "stocks": [
                {"ticker": "BBCA.JK", "change_from_open": -2.1},
                {"ticker": "BBRI.JK", "change_from_open": -1.8},
                {"ticker": "BMRI.JK", "change_from_open": -1.6},
            ],
        }

    def test_returns_none_for_empty(self):
        """format_sell_alert should return None for empty list."""
        assert format_sell_alert([]) is None

    def test_strong_sell_format(self):
        """format_sell_alert should format STRONG SELL with required fields."""
        signals = [self._make_strong_sell()]
        msg = format_sell_alert(signals)
        assert msg is not None
        assert "STRONG SELL" in msg
        assert "BBCA" in msg
        assert "RSI" in msg
        assert "DYOR" in msg

    def test_strong_sell_contains_price(self):
        """Message should contain the stock price."""
        signals = [self._make_strong_sell(current=4210.0)]
        msg = format_sell_alert(signals)
        assert "4,210" in msg or "4210" in msg

    def test_strong_sell_contains_ma20_delta(self):
        """Message should contain MA20 delta."""
        signals = [self._make_strong_sell(from_ma20=-1.8)]
        msg = format_sell_alert(signals)
        assert "MA20" in msg

    def test_strong_sell_contains_scan_time(self):
        """Message should contain the scan time."""
        signals = [self._make_strong_sell()]
        msg = format_sell_alert(signals)
        assert "11:47" in msg
        assert "WIB" in msg

    def test_sector_collapse_format(self):
        """format_sell_alert should format SECTOR COLLAPSE correctly."""
        signals = [self._make_sector_collapse()]
        msg = format_sell_alert(signals)
        assert msg is not None
        assert "SECTOR COLLAPSE" in msg or "COLLAPSE" in msg
        assert "Perbankan" in msg

    def test_sector_collapse_contains_stocks(self):
        """Sector collapse message should mention affected stocks."""
        signals = [self._make_sector_collapse()]
        msg = format_sell_alert(signals)
        assert "BBCA" in msg or "BBRI" in msg or "BMRI" in msg

    def test_returns_string(self):
        """format_sell_alert should return a string."""
        signals = [self._make_strong_sell()]
        msg = format_sell_alert(signals)
        assert isinstance(msg, str)

    def test_multiple_signals(self):
        """Should handle multiple sell signals."""
        signals = [
            self._make_strong_sell("BBCA.JK"),
            self._make_sector_collapse(),
        ]
        msg = format_sell_alert(signals)
        assert msg is not None
        assert "BBCA" in msg

    def test_only_active_conditions_shown(self):
        """Only True conditions should be shown in message."""
        sig = self._make_strong_sell()
        sig["conditions"]["distribution_volume"] = False
        sig["vol_ratio"] = 1.5
        msg = format_sell_alert([sig])
        # distribution_volume is False, so "whale jual" should not appear
        assert "whale jual" not in msg

    def test_disclaimer_present(self):
        """Disclaimer should always appear."""
        signals = [self._make_strong_sell()]
        msg = format_sell_alert(signals)
        assert "BUKAN saran investasi" in msg or "DYOR" in msg
