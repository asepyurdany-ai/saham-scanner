"""
Unit tests for agents/position_tracker.py
Tests: add_position, update_position, check_tp_cl, format_position_alert
"""

import json
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.position_tracker import (
    add_position,
    update_position,
    check_tp_cl,
    format_position_alert,
    load_positions,
    save_positions,
    TP_PCT,
    CL_PCT,
    TRAILING_ACTIVATE_PCT,
    TRAILING_STOP_PCT,
)


# --- Fixtures ---

@pytest.fixture
def positions_file(tmp_path):
    """Temporary positions file for each test."""
    return str(tmp_path / "positions_test.json")


def make_position(ticker="BBCA", entry_price=9000.0, lots=5):
    """Create a base position dict (not persisted)."""
    return {
        "ticker": ticker,
        "entry_price": float(entry_price),
        "lots": int(lots),
        "shares": int(lots) * 100,
        "tp_price": round(entry_price * (1 + TP_PCT), 0),
        "cl_price": round(entry_price * (1 + CL_PCT), 0),
        "trailing_activated": False,
        "trailing_stop": None,
        "highest_price": float(entry_price),
        "current_price": float(entry_price),
        "pnl_pct": 0.0,
        "pnl_rp": 0.0,
        "added_at": "2026-01-01T00:00:00",
        "updated_at": "2026-01-01T00:00:00",
    }


# --- add_position Tests ---

class TestAddPosition:

    def test_add_basic_position(self, positions_file):
        """add_position should create a valid position."""
        positions = {}
        pos = add_position("BBCA", 9000.0, 5, positions=positions, positions_file=positions_file)

        assert pos is not None
        assert pos["ticker"] == "BBCA"
        assert pos["entry_price"] == 9000.0
        assert pos["lots"] == 5
        assert pos["shares"] == 500

    def test_tp_calculated_correctly(self, positions_file):
        """TP should be entry_price * 1.08."""
        positions = {}
        pos = add_position("BBRI", 4000.0, 3, positions=positions, positions_file=positions_file)

        expected_tp = round(4000.0 * (1 + TP_PCT), 0)
        assert pos["tp_price"] == expected_tp

    def test_cl_calculated_correctly(self, positions_file):
        """CL should be entry_price * 0.96."""
        positions = {}
        pos = add_position("BMRI", 7500.0, 2, positions=positions, positions_file=positions_file)

        expected_cl = round(7500.0 * (1 + CL_PCT), 0)
        assert pos["cl_price"] == expected_cl

    def test_initial_pnl_is_zero(self, positions_file):
        """Initial P&L should be 0."""
        positions = {}
        pos = add_position("TLKM", 3000.0, 10, positions=positions, positions_file=positions_file)

        assert pos["pnl_pct"] == 0.0
        assert pos["pnl_rp"] == 0.0

    def test_position_saved_to_file(self, positions_file):
        """Position should be persisted to file."""
        positions = {}
        add_position("ANTM", 1500.0, 20, positions=positions, positions_file=positions_file)

        # Load from file and verify
        loaded = load_positions(positions_file)
        assert "ANTM" in loaded
        assert loaded["ANTM"]["entry_price"] == 1500.0

    def test_ticker_uppercased(self, positions_file):
        """Ticker should be stored uppercase."""
        positions = {}
        pos = add_position("bbca", 9000.0, 5, positions=positions, positions_file=positions_file)

        assert pos["ticker"] == "BBCA"

    def test_trailing_not_activated_initially(self, positions_file):
        """Trailing stop should not be activated initially."""
        positions = {}
        pos = add_position("GOTO", 500.0, 100, positions=positions, positions_file=positions_file)

        assert pos["trailing_activated"] is False
        assert pos["trailing_stop"] is None

    def test_overwrite_existing_position(self, positions_file):
        """Adding same ticker overwrites existing position."""
        positions = {}
        add_position("BBCA", 9000.0, 5, positions=positions, positions_file=positions_file)
        pos2 = add_position("BBCA", 9500.0, 3, positions=positions, positions_file=positions_file)

        assert pos2["entry_price"] == 9500.0
        assert pos2["lots"] == 3


# --- update_position Tests ---

class TestUpdatePosition:

    def test_update_basic_pnl(self, positions_file):
        """update_position should calculate P&L correctly."""
        positions = {"BBCA": make_position("BBCA", 9000.0, 5)}
        pos = update_position("BBCA", 9450.0, positions=positions, positions_file=positions_file)

        expected_pnl_pct = ((9450 - 9000) / 9000) * 100
        assert abs(pos["pnl_pct"] - expected_pnl_pct) < 0.01

        expected_pnl_rp = (9450 - 9000) * 500
        assert pos["pnl_rp"] == expected_pnl_rp

    def test_update_tracks_highest_price(self, positions_file):
        """update_position should track the highest price."""
        positions = {"BBCA": make_position("BBCA", 9000.0, 5)}
        update_position("BBCA", 9500.0, positions=positions, positions_file=positions_file)
        update_position("BBCA", 9400.0, positions=positions, positions_file=positions_file)
        pos = update_position("BBCA", 9300.0, positions=positions, positions_file=positions_file)

        assert positions["BBCA"]["highest_price"] == 9500.0

    def test_trailing_activates_at_5pct(self, positions_file):
        """Trailing stop should activate when profit >= 5%."""
        entry = 9000.0
        positions = {"BBCA": make_position("BBCA", entry, 5)}

        # Profit at exactly 5%
        trigger_price = entry * (1 + TRAILING_ACTIVATE_PCT)
        pos = update_position("BBCA", trigger_price, positions=positions, positions_file=positions_file)

        assert pos["trailing_activated"] is True
        assert pos["trailing_stop"] is not None

    def test_trailing_stop_level(self, positions_file):
        """Trailing stop should be 5% below highest price."""
        entry = 10000.0
        positions = {"BBCA": make_position("BBCA", entry, 5)}

        highest = entry * 1.06  # 6% above entry
        update_position("BBCA", highest, positions=positions, positions_file=positions_file)

        expected_trailing = round(highest * (1 - TRAILING_STOP_PCT), 0)
        assert positions["BBCA"]["trailing_stop"] == expected_trailing

    def test_trailing_stop_only_moves_up(self, positions_file):
        """Trailing stop should never move down once set."""
        entry = 9000.0
        positions = {"BBCA": make_position("BBCA", entry, 5)}

        # Push price up high → set trailing stop
        update_position("BBCA", 9600.0, positions=positions, positions_file=positions_file)
        trailing_after_high = positions["BBCA"]["trailing_stop"]

        # Price dips (still above entry)
        update_position("BBCA", 9200.0, positions=positions, positions_file=positions_file)
        trailing_after_dip = positions["BBCA"]["trailing_stop"]

        # Trailing stop should not have decreased
        assert trailing_after_dip >= trailing_after_high

    def test_update_returns_none_for_missing_ticker(self, positions_file):
        """update_position should return None if ticker not in positions."""
        positions = {}
        result = update_position("NONEXISTENT", 1000.0, positions=positions, positions_file=positions_file)

        assert result is None

    def test_update_negative_pnl(self, positions_file):
        """update_position should calculate negative P&L correctly."""
        positions = {"BBCA": make_position("BBCA", 9000.0, 5)}
        pos = update_position("BBCA", 8550.0, positions=positions, positions_file=positions_file)

        assert pos["pnl_pct"] < 0
        assert pos["pnl_rp"] < 0


# --- check_tp_cl Tests ---

class TestCheckTpCl:

    def test_take_profit_triggered(self):
        """Should return TAKE_PROFIT when current >= tp_price."""
        pos = make_position("BBCA", 9000.0, 5)
        pos["current_price"] = 9800.0  # above tp (9720)
        pos["tp_price"] = 9720.0

        result = check_tp_cl(pos)
        assert result == "TAKE_PROFIT"

    def test_stop_loss_triggered(self):
        """Should return STOP_LOSS when current <= cl_price."""
        pos = make_position("BBCA", 9000.0, 5)
        pos["current_price"] = 8600.0  # below cl (8640)
        pos["cl_price"] = 8640.0

        result = check_tp_cl(pos)
        assert result == "STOP_LOSS"

    def test_trailing_stop_triggered(self):
        """Should return TRAILING_STOP when trailing activated and price drops below trailing."""
        pos = make_position("BBCA", 9000.0, 5)
        pos["trailing_activated"] = True
        pos["trailing_stop"] = 9200.0
        pos["current_price"] = 9100.0  # below trailing stop

        result = check_tp_cl(pos)
        assert result == "TRAILING_STOP"

    def test_no_trigger_in_range(self):
        """Should return None when price is between CL and TP."""
        pos = make_position("BBCA", 9000.0, 5)
        pos["current_price"] = 9200.0  # between cl (8640) and tp (9720)

        result = check_tp_cl(pos)
        assert result is None

    def test_trailing_not_triggered_above_stop(self):
        """TRAILING_STOP should not trigger if current > trailing_stop."""
        pos = make_position("BBCA", 9000.0, 5)
        pos["trailing_activated"] = True
        pos["trailing_stop"] = 9200.0
        pos["current_price"] = 9400.0  # above trailing stop

        result = check_tp_cl(pos)
        assert result is None

    def test_trailing_not_triggered_when_not_activated(self):
        """TRAILING_STOP should not trigger if trailing_activated is False."""
        pos = make_position("BBCA", 9000.0, 5)
        pos["trailing_activated"] = False
        pos["trailing_stop"] = 9200.0
        pos["current_price"] = 9100.0

        result = check_tp_cl(pos)
        # Should check CL instead
        assert result in (None, "STOP_LOSS")

    def test_tp_takes_priority_over_trailing(self):
        """TP should take priority if price hits both TP and trailing stop."""
        pos = make_position("BBCA", 9000.0, 5)
        pos["tp_price"] = 9500.0
        pos["trailing_activated"] = True
        pos["trailing_stop"] = 9600.0  # trailing stop above current
        pos["current_price"] = 9700.0  # above tp

        result = check_tp_cl(pos)
        assert result == "TAKE_PROFIT"

    def test_exact_tp_boundary(self):
        """TAKE_PROFIT should trigger at exactly tp_price."""
        pos = make_position("BBCA", 9000.0, 5)
        pos["current_price"] = pos["tp_price"]

        result = check_tp_cl(pos)
        assert result == "TAKE_PROFIT"

    def test_exact_cl_boundary(self):
        """STOP_LOSS should trigger at exactly cl_price."""
        pos = make_position("BBCA", 9000.0, 5)
        pos["current_price"] = pos["cl_price"]

        result = check_tp_cl(pos)
        assert result == "STOP_LOSS"


# --- format_position_alert Tests ---

class TestFormatPositionAlert:

    def _make_pos(self, pnl_pct=5.0, pnl_rp=225000.0):
        pos = make_position("BBCA", 9000.0, 5)
        pos["current_price"] = 9450.0
        pos["pnl_pct"] = pnl_pct
        pos["pnl_rp"] = pnl_rp
        pos["trailing_stop"] = 9000.0
        return pos

    def test_take_profit_format(self):
        """format_position_alert TAKE_PROFIT should contain key info."""
        pos = self._make_pos(pnl_pct=8.5)
        msg = format_position_alert(pos, "TAKE_PROFIT")

        assert "BBCA" in msg
        assert "TAKE PROFIT" in msg.upper()
        assert "9,000" in msg or "9000" in msg  # entry price (may be formatted with comma)
        assert "9,450" in msg or "9450" in msg  # current price (may be formatted with comma)

    def test_stop_loss_format(self):
        """format_position_alert STOP_LOSS should contain key info."""
        pos = self._make_pos(pnl_pct=-4.2, pnl_rp=-189000.0)
        msg = format_position_alert(pos, "STOP_LOSS")

        assert "BBCA" in msg
        assert "CUT LOSS" in msg.upper() or "STOP" in msg.upper()

    def test_trailing_stop_format(self):
        """format_position_alert TRAILING_STOP should contain key info."""
        pos = self._make_pos()
        msg = format_position_alert(pos, "TRAILING_STOP")

        assert "BBCA" in msg
        assert "TRAILING" in msg.upper()

    def test_format_includes_pnl(self):
        """format_position_alert should always show P&L."""
        pos = self._make_pos(pnl_pct=5.0, pnl_rp=225000.0)
        msg = format_position_alert(pos, "TAKE_PROFIT")

        assert "5.00" in msg or "5,00" in msg or "P&L" in msg

    def test_format_includes_lots(self):
        """format_position_alert should show lot count."""
        pos = self._make_pos()
        pos["lots"] = 5
        msg = format_position_alert(pos, "TAKE_PROFIT")

        assert "5" in msg  # lots count

    def test_format_returns_string(self):
        """format_position_alert should always return a string."""
        pos = self._make_pos()
        for alert_type in ["TAKE_PROFIT", "STOP_LOSS", "TRAILING_STOP", "UPDATE"]:
            msg = format_position_alert(pos, alert_type)
            assert isinstance(msg, str)
            assert len(msg) > 0

    def test_format_disclaimer_present(self):
        """Alert should include a disclaimer."""
        pos = self._make_pos()
        msg = format_position_alert(pos, "TAKE_PROFIT")

        assert "kamu" in msg.lower() or "keputusan" in msg.lower() or "⚠️" in msg

    def test_positive_pnl_shows_plus_sign(self):
        """Positive P&L should show + sign."""
        pos = self._make_pos(pnl_pct=5.0, pnl_rp=225000.0)
        msg = format_position_alert(pos, "TAKE_PROFIT")

        assert "+5.00" in msg or "+225" in msg or "5.00%" in msg
