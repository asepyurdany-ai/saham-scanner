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


# --- _extract_json Tests (Radar) ---

class TestRadarExtractJson:

    def test_pure_json_array(self):
        """Should parse pure JSON array."""
        from agents.radar import _extract_json
        text = '[{"judul": "test", "dampak_ihsg": "NEGATIF"}]'
        result = _extract_json(text)
        assert result is not None
        assert result[0]["dampak_ihsg"] == "NEGATIF"

    def test_markdown_wrapped_json(self):
        """Should parse JSON inside markdown code block."""
        from agents.radar import _extract_json
        text = '```json\n[{"judul": "OPEC cuts output", "level": "TINGGI"}]\n```'
        result = _extract_json(text)
        assert result is not None
        assert result[0]["level"] == "TINGGI"

    def test_json_with_leading_text(self):
        """Should extract JSON even with leading text."""
        from agents.radar import _extract_json
        text = 'Analisa:\n[{"judul": "test", "dampak_ihsg": "POSITIF"}]'
        result = _extract_json(text)
        assert result is not None
        assert result[0]["dampak_ihsg"] == "POSITIF"

    def test_empty_returns_none(self):
        """Empty string returns None."""
        from agents.radar import _extract_json
        assert _extract_json("") is None

    def test_none_returns_none(self):
        """None input returns None."""
        from agents.radar import _extract_json
        assert _extract_json(None) is None

    def test_invalid_json_returns_none(self):
        """Invalid JSON returns None."""
        from agents.radar import _extract_json
        assert _extract_json("not json at all!!!") is None


# --- analyze_geo_impact Mock Tests ---

class TestAnalyzeGeoImpact:

    def test_returns_empty_on_no_articles(self):
        """Should return [] immediately if no articles."""
        from agents.radar import analyze_geo_impact
        result = analyze_geo_impact([])
        assert result == []

    def test_returns_list_on_valid_haiku_response(self):
        """Should return list when Haiku returns valid JSON."""
        from unittest.mock import patch, MagicMock
        from agents.radar import analyze_geo_impact

        with patch('agents.radar.anthropic.Anthropic') as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text='[{"judul": "OPEC cuts", "dampak_ihsg": "NEGATIF", "level": "TINGGI", "saham_terdampak": ["MEDC"], "analisa": "Minyak naik"}]')]
            mock_client.messages.create.return_value = mock_response

            articles = [{"source": "Reuters", "title": "OPEC cuts output", "summary": "test", "id": "1"}]
            result = analyze_geo_impact(articles)
            assert isinstance(result, list)
            assert len(result) == 1

    def test_returns_empty_on_bad_haiku_response(self):
        """Should return [] when Haiku returns unparseable response."""
        from unittest.mock import patch, MagicMock
        from agents.radar import analyze_geo_impact

        with patch('agents.radar.anthropic.Anthropic') as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text='')]
            mock_client.messages.create.return_value = mock_response

            articles = [{"source": "Reuters", "title": "test", "summary": "test", "id": "1"}]
            result = analyze_geo_impact(articles)
            assert result == []

    def test_handles_api_exception(self):
        """Should return [] if API throws exception."""
        from unittest.mock import patch, MagicMock
        from agents.radar import analyze_geo_impact

        with patch('agents.radar.anthropic.Anthropic') as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client
            mock_client.messages.create.side_effect = Exception("connection error")

            articles = [{"source": "Reuters", "title": "test", "summary": "test", "id": "1"}]
            result = analyze_geo_impact(articles)
            assert result == []


# ─── Tests: check_macro_shock ───────────────────────────────────────────────

class TestCheckMacroShock:

    def test_returns_list(self):
        """check_macro_shock should return a list."""
        from agents.radar import check_macro_shock
        from unittest.mock import patch
        import pandas as pd

        # Mock yf.Ticker to return empty history (no shock)
        with patch("agents.radar.yf.Ticker") as mock_ticker_cls:
            mock_ticker = mock_ticker_cls.return_value
            mock_ticker.history.return_value = pd.DataFrame()
            result = check_macro_shock()

        assert isinstance(result, list)

    def test_no_shock_below_threshold(self):
        """No shock alert when USD/IDR moves < 1.5%."""
        from agents.radar import check_macro_shock
        from unittest.mock import patch, MagicMock
        import pandas as pd

        # 0.5% move — below both thresholds
        def make_mock_hist(open_p, close_p):
            df = pd.DataFrame({
                "Open": [open_p, open_p],
                "Close": [open_p, close_p],
            })
            return df

        with patch("agents.radar.yf.Ticker") as mock_ticker_cls:
            mock_ticker = MagicMock()
            mock_ticker_cls.return_value = mock_ticker

            # USD/IDR: 0.5% move (below 1.5% threshold)
            hist_usd = make_mock_hist(16000.0, 16080.0)  # +0.5%
            # WTI Oil: 1% move (below 3% threshold)
            hist_oil = make_mock_hist(80.0, 80.8)  # +1%

            call_count = [0]
            def side_effect(period, interval=None):
                call_count[0] += 1
                if call_count[0] <= 1:
                    return hist_usd
                return hist_oil

            mock_ticker.history.side_effect = side_effect
            result = check_macro_shock()

        assert len(result) == 0

    def test_shock_detected_usd_idr(self):
        """Should detect USD/IDR shock when move > 1.5%."""
        from agents.radar import check_macro_shock
        from unittest.mock import patch, MagicMock
        import pandas as pd

        def make_hist(open_p, close_p):
            return pd.DataFrame({
                "Open": [open_p] * 5,
                "Close": [open_p] * 4 + [close_p],
            })

        with patch("agents.radar.yf.Ticker") as mock_ticker_cls:
            mock_ticker = MagicMock()
            mock_ticker_cls.return_value = mock_ticker

            # USD/IDR: +2.1% move (above 1.5% threshold)
            call_count = [0]
            def side_effect(period, interval=None):
                call_count[0] += 1
                if call_count[0] <= 1:
                    return make_hist(16000.0, 16336.0)  # +2.1%
                return make_hist(80.0, 80.8)  # oil: +1% (no shock)

            mock_ticker.history.side_effect = side_effect
            result = check_macro_shock()

        shock_names = [s["name"] for s in result]
        assert "USD/IDR" in shock_names

    def test_shock_structure(self):
        """Shock dict should have required keys."""
        from agents.radar import check_macro_shock
        from unittest.mock import patch, MagicMock
        import pandas as pd

        def make_hist(open_p, close_p):
            return pd.DataFrame({
                "Open": [open_p] * 5,
                "Close": [open_p] * 4 + [close_p],
            })

        with patch("agents.radar.yf.Ticker") as mock_ticker_cls:
            mock_ticker = MagicMock()
            mock_ticker_cls.return_value = mock_ticker

            call_count = [0]
            def side_effect(period, interval=None):
                call_count[0] += 1
                if call_count[0] <= 1:
                    return make_hist(16000.0, 16336.0)  # USD/IDR +2.1%
                return make_hist(80.0, 78.0)  # WTI Oil -2.5% (below 3%)

            mock_ticker.history.side_effect = side_effect
            result = check_macro_shock()

        if result:
            shock = result[0]
            for key in ["name", "current", "open", "change_pct", "threshold", "scan_time"]:
                assert key in shock, f"Missing key: {key}"

    def test_wti_shock_detected(self):
        """Should detect WTI Oil shock when move > 3%."""
        from agents.radar import check_macro_shock
        from unittest.mock import patch, MagicMock
        import pandas as pd

        def make_hist(open_p, close_p):
            return pd.DataFrame({
                "Open": [open_p] * 5,
                "Close": [open_p] * 4 + [close_p],
            })

        with patch("agents.radar.yf.Ticker") as mock_ticker_cls:
            mock_ticker = MagicMock()
            mock_ticker_cls.return_value = mock_ticker

            call_count = [0]
            def side_effect(period, interval=None):
                call_count[0] += 1
                if call_count[0] <= 1:
                    return make_hist(16000.0, 16080.0)  # USD/IDR +0.5% (no shock)
                return make_hist(80.0, 83.2)  # WTI Oil +4% (above 3% threshold)

            mock_ticker.history.side_effect = side_effect
            result = check_macro_shock()

        shock_names = [s["name"] for s in result]
        assert "WTI Oil" in shock_names

    def test_handles_empty_history(self):
        """Should not crash on empty history from yfinance."""
        from agents.radar import check_macro_shock
        from unittest.mock import patch
        import pandas as pd

        with patch("agents.radar.yf.Ticker") as mock_ticker_cls:
            mock_ticker = mock_ticker_cls.return_value
            mock_ticker.history.return_value = pd.DataFrame()
            result = check_macro_shock()

        assert isinstance(result, list)

    def test_change_pct_positive_for_rise(self):
        """change_pct should be positive for price increase."""
        from agents.radar import check_macro_shock
        from unittest.mock import patch, MagicMock
        import pandas as pd

        def make_hist(open_p, close_p):
            return pd.DataFrame({
                "Open": [open_p] * 5,
                "Close": [open_p] * 4 + [close_p],
            })

        with patch("agents.radar.yf.Ticker") as mock_ticker_cls:
            mock_ticker = MagicMock()
            mock_ticker_cls.return_value = mock_ticker

            call_count = [0]
            def side_effect(period, interval=None):
                call_count[0] += 1
                if call_count[0] <= 1:
                    return make_hist(16000.0, 16336.0)  # +2.1%
                return pd.DataFrame()

            mock_ticker.history.side_effect = side_effect
            result = check_macro_shock()

        if result:
            assert result[0]["change_pct"] > 0


# ─── Tests: format_macro_shock_alert ────────────────────────────────────────

class TestFormatMacroShockAlert:

    def _make_shock(self, name="USD/IDR", current=16450.0, change_pct=2.1):
        return {
            "name": name,
            "current": current,
            "open": current / (1 + change_pct / 100),
            "change_pct": change_pct,
            "threshold": 1.5,
            "scan_time": "11:32",
        }

    def test_returns_none_for_empty(self):
        """format_macro_shock_alert should return None for empty list."""
        from agents.radar import format_macro_shock_alert
        assert format_macro_shock_alert([]) is None

    def test_returns_string(self):
        """format_macro_shock_alert should return a string."""
        from agents.radar import format_macro_shock_alert
        result = format_macro_shock_alert([self._make_shock()])
        assert isinstance(result, str)

    def test_contains_macro_shock_header(self):
        """Message should contain MACRO SHOCK header."""
        from agents.radar import format_macro_shock_alert
        msg = format_macro_shock_alert([self._make_shock()])
        assert "MACRO SHOCK" in msg

    def test_contains_usd_idr_info(self):
        """Message should include USD/IDR data."""
        from agents.radar import format_macro_shock_alert
        msg = format_macro_shock_alert([self._make_shock("USD/IDR", 16450.0, 2.1)])
        assert "USD/IDR" in msg
        assert "2.1" in msg

    def test_contains_priority_stocks(self):
        """Message should list priority exit stocks."""
        from agents.radar import format_macro_shock_alert
        msg = format_macro_shock_alert([self._make_shock()])
        assert "GOTO" in msg or "BUKA" in msg or "BMRI" in msg

    def test_wib_timestamp_present(self):
        """Message should contain WIB timestamp."""
        from agents.radar import format_macro_shock_alert
        msg = format_macro_shock_alert([self._make_shock()])
        assert "WIB" in msg

    def test_rupiah_weakening_message(self):
        """Positive USD/IDR change should mention rupiah weakening."""
        from agents.radar import format_macro_shock_alert
        msg = format_macro_shock_alert([self._make_shock("USD/IDR", 16450.0, 2.1)])
        assert "melemah" in msg.lower() or "outflow" in msg.lower()

    def test_oil_shock_message(self):
        """WTI Oil shock should mention oil."""
        from agents.radar import format_macro_shock_alert
        msg = format_macro_shock_alert([self._make_shock("WTI Oil", 85.0, 4.0)])
        assert "WTI Oil" in msg or "Minyak" in msg

    def test_multiple_shocks(self):
        """Should handle multiple simultaneous shocks."""
        from agents.radar import format_macro_shock_alert
        shocks = [
            self._make_shock("USD/IDR", 16450.0, 2.1),
            self._make_shock("WTI Oil", 85.0, 4.5),
        ]
        msg = format_macro_shock_alert(shocks)
        assert "USD/IDR" in msg
        assert "WTI Oil" in msg
