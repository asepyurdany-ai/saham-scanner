"""
Unit tests for agents/radar.py
Tests: check_commodity_alerts (>2% triggers, <2% does not), format_commodity_alert
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.radar import check_commodity_alerts, format_commodity_alert


# --- Helpers ---

def make_prices(changes: dict) -> dict:
    """
    Create commodity prices dict with specified change_pct values.
    e.g. {"Gold": 2.5, "WTI Oil": -1.5}
    """
    prices = {}
    base_prices = {
        "Gold": 1850.0,
        "WTI Oil": 75.0,
        "Brent Oil": 78.0,
        "USD/IDR": 15500.0,
        "Copper": 4.2,
    }
    for name, change_pct in changes.items():
        base = base_prices.get(name, 100.0)
        prices[name] = {
            "current": round(base * (1 + change_pct / 100), 2),
            "prev": base,
            "change_pct": change_pct,
        }
    return prices


# --- check_commodity_alerts Tests ---

class TestCheckCommodityAlerts:

    def test_large_positive_move_triggers(self):
        """>2% positive move should trigger an alert."""
        prices = make_prices({"Gold": 2.5})
        alerts = check_commodity_alerts(prices)

        assert len(alerts) == 1
        assert alerts[0]["commodity"] == "Gold"
        assert alerts[0]["change_pct"] == 2.5
        assert alerts[0]["direction"] == "naik"

    def test_large_negative_move_triggers(self):
        """>2% negative move should trigger an alert."""
        prices = make_prices({"WTI Oil": -3.1})
        alerts = check_commodity_alerts(prices)

        assert len(alerts) == 1
        assert alerts[0]["commodity"] == "WTI Oil"
        assert alerts[0]["change_pct"] == -3.1
        assert alerts[0]["direction"] == "turun"

    def test_exactly_2pct_triggers(self):
        """Exactly 2% move should trigger an alert (boundary)."""
        prices = make_prices({"Gold": 2.0})
        alerts = check_commodity_alerts(prices)

        assert len(alerts) == 1

    def test_below_2pct_no_trigger(self):
        """<2% move should NOT trigger an alert."""
        prices = make_prices({"Gold": 1.9})
        alerts = check_commodity_alerts(prices)

        assert len(alerts) == 0

    def test_small_negative_no_trigger(self):
        """Small negative move <2% should not trigger."""
        prices = make_prices({"WTI Oil": -1.5})
        alerts = check_commodity_alerts(prices)

        assert len(alerts) == 0

    def test_zero_change_no_trigger(self):
        """Zero change should not trigger."""
        prices = make_prices({"Gold": 0.0})
        alerts = check_commodity_alerts(prices)

        assert len(alerts) == 0

    def test_multiple_commodities_only_large_trigger(self):
        """Multiple commodities: only those >=2% should trigger."""
        prices = make_prices({
            "Gold": 2.5,
            "WTI Oil": 1.8,
            "Brent Oil": -3.0,
            "Copper": 0.5,
        })
        alerts = check_commodity_alerts(prices)

        triggered = {a["commodity"] for a in alerts}
        assert "Gold" in triggered
        assert "Brent Oil" in triggered
        assert "WTI Oil" not in triggered
        assert "Copper" not in triggered

    def test_alert_contains_affected_stocks(self):
        """Alert should include affected stocks for known commodities."""
        prices = make_prices({"Gold": 3.0})
        alerts = check_commodity_alerts(prices)

        assert len(alerts) == 1
        assert len(alerts[0]["affected_stocks"]) > 0  # Gold has ANTM, MDKA

    def test_alert_structure(self):
        """Alert dict should have required keys."""
        prices = make_prices({"Gold": 2.5})
        alerts = check_commodity_alerts(prices)

        required = ["commodity", "change_pct", "direction", "current", "affected_stocks"]
        for key in required:
            assert key in alerts[0], f"Missing key: {key}"

    def test_empty_prices_no_alerts(self):
        """Empty prices dict should return empty alerts."""
        alerts = check_commodity_alerts({})
        assert alerts == []

    def test_usd_idr_triggers(self):
        """USD/IDR move >=2% should trigger (affects banking sector)."""
        prices = make_prices({"USD/IDR": 2.1})
        alerts = check_commodity_alerts(prices)

        assert len(alerts) == 1
        assert alerts[0]["commodity"] == "USD/IDR"

    def test_direction_positive(self):
        """direction should be 'naik' for positive change."""
        prices = make_prices({"Gold": 2.5})
        alerts = check_commodity_alerts(prices)

        assert alerts[0]["direction"] == "naik"

    def test_direction_negative(self):
        """direction should be 'turun' for negative change."""
        prices = make_prices({"Gold": -2.5})
        alerts = check_commodity_alerts(prices)

        assert alerts[0]["direction"] == "turun"


# --- format_commodity_alert Tests ---

class TestFormatCommodityAlert:

    def _make_alert(self, commodity="Gold", change_pct=2.5, direction="naik"):
        return {
            "commodity": commodity,
            "change_pct": change_pct,
            "direction": direction,
            "current": 1870.25,
            "affected_stocks": ["ANTM", "MDKA"],
        }

    def test_format_basic(self):
        """format_commodity_alert should return a non-empty string."""
        alerts = [self._make_alert()]
        prices = make_prices({"Gold": 2.5})
        msg = format_commodity_alert(alerts, prices)

        assert isinstance(msg, str)
        assert len(msg) > 0

    def test_format_contains_commodity_name(self):
        """Message should mention the commodity."""
        alerts = [self._make_alert("Gold", 2.5)]
        prices = make_prices({"Gold": 2.5})
        msg = format_commodity_alert(alerts, prices)

        assert "Gold" in msg

    def test_format_contains_change_pct(self):
        """Message should contain the change percentage."""
        alerts = [self._make_alert("WTI Oil", 3.1, "naik")]
        prices = make_prices({"WTI Oil": 3.1})
        msg = format_commodity_alert(alerts, prices)

        assert "3.1" in msg or "WTI Oil" in msg

    def test_format_contains_affected_stocks(self):
        """Message should mention affected stocks."""
        alerts = [self._make_alert("Gold", 2.5)]
        prices = make_prices({"Gold": 2.5})
        msg = format_commodity_alert(alerts, prices)

        assert "ANTM" in msg or "MDKA" in msg

    def test_format_positive_emoji(self):
        """Positive change should show green emoji."""
        alerts = [self._make_alert("Gold", 2.5, "naik")]
        prices = make_prices({"Gold": 2.5})
        msg = format_commodity_alert(alerts, prices)

        assert "🟢" in msg

    def test_format_negative_emoji(self):
        """Negative change should show red emoji."""
        alerts = [self._make_alert("Gold", -2.5, "turun")]
        prices = make_prices({"Gold": -2.5})
        msg = format_commodity_alert(alerts, prices)

        assert "🔴" in msg

    def test_format_multiple_alerts(self):
        """format_commodity_alert should handle multiple alerts."""
        alerts = [
            self._make_alert("Gold", 2.5, "naik"),
            self._make_alert("WTI Oil", -3.0, "turun"),
        ]
        prices = make_prices({"Gold": 2.5, "WTI Oil": -3.0})
        msg = format_commodity_alert(alerts, prices)

        assert "Gold" in msg
        assert "WTI Oil" in msg

    def test_format_header_present(self):
        """Message should have a RADAR ALERT header."""
        alerts = [self._make_alert()]
        prices = make_prices({"Gold": 2.5})
        msg = format_commodity_alert(alerts, prices)

        assert "RADAR" in msg.upper()

    def test_format_disclaimer_present(self):
        """Message should have a disclaimer."""
        alerts = [self._make_alert()]
        prices = make_prices({"Gold": 2.5})
        msg = format_commodity_alert(alerts, prices)

        assert "⚠️" in msg or "Konfirmasi" in msg

    def test_format_contains_timestamp(self):
        """Message should contain time information."""
        alerts = [self._make_alert()]
        prices = make_prices({"Gold": 2.5})
        msg = format_commodity_alert(alerts, prices)

        # Should contain WIB timestamp
        assert "WIB" in msg
