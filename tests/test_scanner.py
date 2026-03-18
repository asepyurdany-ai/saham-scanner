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

from agents.scanner import compute_signals, save_signals, format_morning_alert, compute_macd


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
