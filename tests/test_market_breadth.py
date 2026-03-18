"""
Unit tests for agents/market_breadth.py
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.market_breadth import (
    fetch_ihsg_data,
    calculate_breadth,
    check_market_gate,
    format_breadth_alert,
)


# ─── Helpers ────────────────────────────────────────────────────────────────

def _make_ihsg(current=7200, open_=7200, prev_close=7100, change_from_open=0.0, change_from_prev=1.4):
    return {
        "current": current,
        "open": open_,
        "prev_close": prev_close,
        "change_from_open_pct": change_from_open,
        "change_from_prev_pct": change_from_prev,
    }


def _make_signals(advancing: int, declining: int):
    """Create a list of signal dicts with specified advancing/declining counts."""
    signals = []
    for _ in range(advancing):
        signals.append({"daily_change_pct": 1.0})
    for _ in range(declining):
        signals.append({"daily_change_pct": -1.0})
    return signals


# ─── fetch_ihsg_data ─────────────────────────────────────────────────────────

class TestFetchIhsgData:

    def _make_hist(self, closes, opens=None):
        """Create a mock yfinance history DataFrame."""
        if opens is None:
            opens = [c * 0.99 for c in closes]
        dates = pd.date_range("2026-01-01", periods=len(closes), freq="D")
        return pd.DataFrame({
            "Open": opens,
            "High": [c * 1.01 for c in closes],
            "Low": [c * 0.99 for c in closes],
            "Close": closes,
            "Volume": [1_000_000] * len(closes),
        }, index=dates)

    def test_returns_dict_on_success(self):
        """Should return dict with required keys."""
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = self._make_hist([7100.0, 7200.0], opens=[7050.0, 7150.0])
        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = fetch_ihsg_data()
        assert isinstance(result, dict)
        for key in ["current", "open", "prev_close", "change_from_open_pct", "change_from_prev_pct"]:
            assert key in result

    def test_returns_none_on_empty_data(self):
        """Should return None when yfinance returns empty DataFrame."""
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = fetch_ihsg_data()
        assert result is None

    def test_returns_none_on_exception(self):
        """Should return None on any exception."""
        with patch("yfinance.Ticker", side_effect=Exception("Network error")):
            result = fetch_ihsg_data()
        assert result is None

    def test_change_from_open_calculation(self):
        """change_from_open_pct should be correct."""
        # current=7200, open=7100 → +1.408%
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = self._make_hist(
            [7100.0, 7200.0], opens=[7100.0, 7100.0]
        )
        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = fetch_ihsg_data()
        if result:
            expected = ((7200.0 - 7100.0) / 7100.0) * 100
            assert abs(result["change_from_open_pct"] - expected) < 0.01

    def test_change_from_prev_calculation(self):
        """change_from_prev_pct should be correct."""
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = self._make_hist(
            [7000.0, 7070.0], opens=[7070.0, 7070.0]
        )
        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = fetch_ihsg_data()
        if result:
            expected = ((7070.0 - 7000.0) / 7000.0) * 100
            assert abs(result["change_from_prev_pct"] - expected) < 0.01


# ─── calculate_breadth ───────────────────────────────────────────────────────

class TestCalculateBreadth:

    def test_returns_dict(self):
        signals = _make_signals(10, 5)
        result = calculate_breadth(signals)
        assert isinstance(result, dict)

    def test_required_keys(self):
        signals = _make_signals(10, 5)
        result = calculate_breadth(signals)
        for key in ["advancing", "declining", "breadth_pct", "breadth_signal"]:
            assert key in result

    def test_correct_advancing_count(self):
        signals = _make_signals(14, 8)
        result = calculate_breadth(signals)
        assert result["advancing"] == 14

    def test_correct_declining_count(self):
        signals = _make_signals(14, 8)
        result = calculate_breadth(signals)
        assert result["declining"] == 8

    def test_breadth_pct_calculation(self):
        """breadth_pct = advancing / total * 100."""
        signals = _make_signals(14, 8)  # 14/22 = 63.6%
        result = calculate_breadth(signals)
        expected = (14 / 22) * 100
        assert abs(result["breadth_pct"] - expected) < 0.1

    def test_strong_signal_above_65(self):
        """breadth > 65% → STRONG."""
        signals = _make_signals(15, 5)  # 75%
        result = calculate_breadth(signals)
        assert result["breadth_signal"] == "STRONG"

    def test_weak_signal_below_35(self):
        """breadth < 35% → WEAK."""
        signals = _make_signals(6, 16)  # 27%
        result = calculate_breadth(signals)
        assert result["breadth_signal"] == "WEAK"

    def test_neutral_signal_between_35_65(self):
        """35% ≤ breadth ≤ 65% → NEUTRAL."""
        signals = _make_signals(10, 10)  # 50%
        result = calculate_breadth(signals)
        assert result["breadth_signal"] == "NEUTRAL"

    def test_empty_signals_returns_neutral(self):
        result = calculate_breadth([])
        assert result["advancing"] == 0
        assert result["declining"] == 0
        assert result["breadth_signal"] == "NEUTRAL"

    def test_all_advancing(self):
        signals = _make_signals(10, 0)
        result = calculate_breadth(signals)
        assert result["breadth_pct"] == 100.0
        assert result["breadth_signal"] == "STRONG"

    def test_all_declining(self):
        signals = _make_signals(0, 10)
        result = calculate_breadth(signals)
        assert result["breadth_pct"] == 0.0
        assert result["breadth_signal"] == "WEAK"

    def test_zero_change_counts_as_declining(self):
        """daily_change_pct == 0 should count as declining (not advancing)."""
        signals = [{"daily_change_pct": 0.0}, {"daily_change_pct": 1.0}]
        result = calculate_breadth(signals)
        assert result["advancing"] == 1
        assert result["declining"] == 1

    def test_boundary_exactly_65(self):
        """Exactly 65% → NEUTRAL (not STRONG, which requires > 65)."""
        # 13/20 = 65% exactly → NEUTRAL
        signals = _make_signals(13, 7)
        result = calculate_breadth(signals)
        assert result["breadth_signal"] == "NEUTRAL"

    def test_boundary_exactly_35(self):
        """Exactly 35% → NEUTRAL (not WEAK, which requires < 35)."""
        # 7/20 = 35% exactly → NEUTRAL
        signals = _make_signals(7, 13)
        result = calculate_breadth(signals)
        assert result["breadth_signal"] == "NEUTRAL"


# ─── check_market_gate ───────────────────────────────────────────────────────

class TestCheckMarketGate:

    def test_returns_dict(self):
        result = check_market_gate(_make_ihsg(), {"breadth_pct": 50, "breadth_signal": "NEUTRAL"})
        assert isinstance(result, dict)

    def test_required_keys(self):
        result = check_market_gate(_make_ihsg(), {"breadth_pct": 50, "breadth_signal": "NEUTRAL"})
        assert "gate" in result
        assert "reason" in result

    def test_gate_open_normal_conditions(self):
        """Normal conditions → OPEN."""
        ihsg = _make_ihsg(change_from_open=0.5)
        breadth = {"breadth_pct": 60, "breadth_signal": "NEUTRAL"}
        result = check_market_gate(ihsg, breadth)
        assert result["gate"] == "OPEN"

    def test_gate_closed_ihsg_down_1_5(self):
        """IHSG down > 1.5% from open → CLOSED."""
        ihsg = _make_ihsg(change_from_open=-1.6)
        breadth = {"breadth_pct": 60, "breadth_signal": "NEUTRAL"}
        result = check_market_gate(ihsg, breadth)
        assert result["gate"] == "CLOSED"

    def test_gate_closed_breadth_weak(self):
        """Breadth WEAK → CLOSED."""
        ihsg = _make_ihsg(change_from_open=0.1)
        breadth = {"breadth_pct": 30, "breadth_signal": "WEAK"}
        result = check_market_gate(ihsg, breadth)
        assert result["gate"] == "CLOSED"

    def test_gate_cautious_ihsg_down_0_5_to_1_5(self):
        """IHSG down 0.5-1.5% from open → CAUTIOUS."""
        ihsg = _make_ihsg(change_from_open=-1.0)
        breadth = {"breadth_pct": 55, "breadth_signal": "NEUTRAL"}
        result = check_market_gate(ihsg, breadth)
        assert result["gate"] == "CAUTIOUS"

    def test_gate_cautious_breadth_35_to_50(self):
        """Breadth 35-50% → CAUTIOUS."""
        ihsg = _make_ihsg(change_from_open=0.2)
        breadth = {"breadth_pct": 45, "breadth_signal": "NEUTRAL"}
        result = check_market_gate(ihsg, breadth)
        assert result["gate"] == "CAUTIOUS"

    def test_gate_closed_has_reason(self):
        """CLOSED gate should have a reason."""
        ihsg = _make_ihsg(change_from_open=-2.0)
        breadth = {"breadth_pct": 60, "breadth_signal": "NEUTRAL"}
        result = check_market_gate(ihsg, breadth)
        assert result["gate"] == "CLOSED"
        assert len(result["reason"]) > 0

    def test_both_none_returns_open(self):
        """Both ihsg_data and breadth None → OPEN (default)."""
        result = check_market_gate(None, None)
        assert result["gate"] == "OPEN"

    def test_ihsg_none_uses_breadth_only(self):
        """With ihsg_data=None, should use breadth only."""
        breadth = {"breadth_pct": 20, "breadth_signal": "WEAK"}
        result = check_market_gate(None, breadth)
        assert result["gate"] == "CLOSED"

    def test_breadth_none_uses_ihsg_only(self):
        """With breadth=None, should use ihsg only."""
        ihsg = _make_ihsg(change_from_open=-2.0)
        result = check_market_gate(ihsg, None)
        assert result["gate"] == "CLOSED"

    def test_gate_values_valid(self):
        """gate should be one of OPEN, CAUTIOUS, CLOSED."""
        ihsg = _make_ihsg(change_from_open=-0.3)
        breadth = {"breadth_pct": 60, "breadth_signal": "NEUTRAL"}
        result = check_market_gate(ihsg, breadth)
        assert result["gate"] in ("OPEN", "CAUTIOUS", "CLOSED")


# ─── format_breadth_alert ────────────────────────────────────────────────────

class TestFormatBreadthAlert:

    def _breadth(self, advancing=8, declining=14, breadth_pct=36.0, signal="WEAK"):
        return {
            "advancing": advancing,
            "declining": declining,
            "breadth_pct": breadth_pct,
            "breadth_signal": signal,
        }

    def test_returns_none_when_gate_open(self):
        """No alert needed when gate is OPEN."""
        gate = {"gate": "OPEN", "reason": "OK"}
        result = format_breadth_alert(_make_ihsg(), self._breadth(), gate)
        assert result is None

    def test_returns_string_when_cautious(self):
        gate = {"gate": "CAUTIOUS", "reason": "IHSG turun"}
        result = format_breadth_alert(_make_ihsg(change_from_open=-0.8), self._breadth(), gate)
        assert isinstance(result, str)

    def test_returns_string_when_closed(self):
        gate = {"gate": "CLOSED", "reason": "Breadth lemah"}
        result = format_breadth_alert(_make_ihsg(change_from_open=-1.8), self._breadth(), gate)
        assert isinstance(result, str)

    def test_closed_gate_shows_red_block(self):
        gate = {"gate": "CLOSED", "reason": "Breadth lemah"}
        result = format_breadth_alert(_make_ihsg(), self._breadth(), gate)
        assert "CLOSED" in result

    def test_cautious_gate_shows_warning(self):
        gate = {"gate": "CAUTIOUS", "reason": "IHSG turun"}
        result = format_breadth_alert(_make_ihsg(), self._breadth(), gate)
        assert "CAUTIOUS" in result

    def test_contains_ihsg_value(self):
        gate = {"gate": "CAUTIOUS", "reason": "test"}
        ihsg = _make_ihsg(current=7234.0)
        result = format_breadth_alert(ihsg, self._breadth(), gate)
        assert "7" in result  # IHSG value present

    def test_contains_breadth_info(self):
        gate = {"gate": "CAUTIOUS", "reason": "test"}
        result = format_breadth_alert(_make_ihsg(), self._breadth(advancing=8, declining=14), gate)
        assert "8" in result and "22" in result or "Breadth" in result or "breadth" in result.lower()

    def test_contains_wib_time(self):
        gate = {"gate": "CLOSED", "reason": "test"}
        result = format_breadth_alert(_make_ihsg(), self._breadth(), gate)
        assert "WIB" in result

    def test_contains_advice(self):
        gate = {"gate": "CAUTIOUS", "reason": "test"}
        result = format_breadth_alert(_make_ihsg(), self._breadth(), gate)
        assert "BUY" in result or "tahan" in result.lower() or "lemah" in result.lower()

    def test_alert_format_missing_text(self):
        """format_breadth_alert should contain ALERT keyword."""
        gate = {"gate": "CLOSED", "reason": "test"}
        result = format_breadth_alert(_make_ihsg(), self._breadth(), gate)
        assert "ALERT" in result or "BREADTH" in result or "MARKET" in result
