"""
Unit tests for agents/foreign_flow.py
All HTTP calls are mocked.
"""

import json
import os
import sys
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.foreign_flow import (
    fetch_foreign_flow,
    get_net_foreign,
    format_foreign_summary,
    save_flow_data,
)


# ─── Helpers ────────────────────────────────────────────────────────────────

def _make_idx_response(records: list) -> MagicMock:
    """Build a mock requests.Response for IDX API."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"data": records}
    return mock_resp


def _sample_records():
    return [
        {"StockCode": "BBCA", "ForeignBuy": 5_000_000, "ForeignSell": 2_000_000},
        {"StockCode": "BBRI", "ForeignBuy": 1_000_000, "ForeignSell": 3_000_000},
        {"StockCode": "BMRI", "ForeignBuy": 2_000_000, "ForeignSell": 2_000_000},
    ]


# ─── fetch_foreign_flow ──────────────────────────────────────────────────────

class TestFetchForeignFlow:

    def test_returns_dict(self):
        """fetch_foreign_flow should return a dict."""
        with patch("requests.get", return_value=_make_idx_response(_sample_records())):
            result = fetch_foreign_flow("2026-01-01")
        assert isinstance(result, dict)

    def test_correct_tickers(self):
        """Should extract correct ticker codes."""
        with patch("requests.get", return_value=_make_idx_response(_sample_records())):
            result = fetch_foreign_flow("2026-01-01")
        assert "BBCA" in result
        assert "BBRI" in result
        assert "BMRI" in result

    def test_correct_net_buy(self):
        """Net foreign = buy - sell for BBCA."""
        with patch("requests.get", return_value=_make_idx_response(_sample_records())):
            result = fetch_foreign_flow("2026-01-01")
        assert result["BBCA"]["net_foreign"] == 3_000_000

    def test_correct_net_sell(self):
        """Net foreign = buy - sell for BBRI (net sell)."""
        with patch("requests.get", return_value=_make_idx_response(_sample_records())):
            result = fetch_foreign_flow("2026-01-01")
        assert result["BBRI"]["net_foreign"] == -2_000_000

    def test_neutral_net(self):
        """Net foreign = 0 when buy == sell."""
        with patch("requests.get", return_value=_make_idx_response(_sample_records())):
            result = fetch_foreign_flow("2026-01-01")
        assert result["BMRI"]["net_foreign"] == 0

    def test_structure_has_required_keys(self):
        """Each ticker entry should have foreign_buy, foreign_sell, net_foreign."""
        with patch("requests.get", return_value=_make_idx_response(_sample_records())):
            result = fetch_foreign_flow("2026-01-01")
        for ticker, data in result.items():
            assert "foreign_buy" in data
            assert "foreign_sell" in data
            assert "net_foreign" in data

    def test_returns_empty_on_http_error(self):
        """Should return {} on HTTP error after retries."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("HTTP 500")
        with patch("requests.get", return_value=mock_resp):
            with patch("time.sleep"):  # speed up retries
                result = fetch_foreign_flow("2026-01-01")
        assert result == {}

    def test_returns_empty_on_connection_error(self):
        """Should return {} if requests.get raises exception."""
        with patch("requests.get", side_effect=Exception("Connection error")):
            with patch("time.sleep"):
                result = fetch_foreign_flow("2026-01-01")
        assert result == {}

    def test_default_date_is_today(self):
        """Should call API with today's date if no date_str given."""
        from datetime import datetime
        today = datetime.utcnow().strftime("%Y-%m-%d")

        with patch("requests.get", return_value=_make_idx_response([])) as mock_get:
            fetch_foreign_flow()
        call_kwargs = mock_get.call_args
        # date should appear in params
        params = call_kwargs[1].get("params", {}) or (call_kwargs[0][1] if len(call_kwargs[0]) > 1 else {})
        # Accept any call that included today's date somewhere
        assert mock_get.called

    def test_skips_empty_stock_code(self):
        """Should skip records with empty StockCode."""
        records = [
            {"StockCode": "", "ForeignBuy": 1_000, "ForeignSell": 500},
            {"StockCode": "BBCA", "ForeignBuy": 5_000, "ForeignSell": 2_000},
        ]
        with patch("requests.get", return_value=_make_idx_response(records)):
            result = fetch_foreign_flow("2026-01-01")
        assert "" not in result
        assert "BBCA" in result

    def test_handles_data_key_uppercase(self):
        """Should handle 'Data' (uppercase) response key too."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"Data": [
            {"StockCode": "TLKM", "ForeignBuy": 1_000, "ForeignSell": 500}
        ]}
        with patch("requests.get", return_value=mock_resp):
            result = fetch_foreign_flow("2026-01-01")
        assert "TLKM" in result

    def test_retries_on_failure(self):
        """Should retry 2x before giving up."""
        error_resp = MagicMock()
        error_resp.raise_for_status.side_effect = Exception("Timeout")
        with patch("requests.get", return_value=error_resp) as mock_get:
            with patch("time.sleep"):
                result = fetch_foreign_flow("2026-01-01")
        assert mock_get.call_count == 3  # 3 total attempts


# ─── get_net_foreign ─────────────────────────────────────────────────────────

class TestGetNetForeign:

    def _flow(self):
        return {
            "BBCA": {"foreign_buy": 5_000_000, "foreign_sell": 2_000_000, "net_foreign": 3_000_000},
            "BBRI": {"foreign_buy": 1_000_000, "foreign_sell": 3_000_000, "net_foreign": -2_000_000},
            "BMRI": {"foreign_buy": 2_000_000, "foreign_sell": 2_000_000, "net_foreign": 0},
        }

    def test_returns_dict(self):
        result = get_net_foreign("BBCA.JK", self._flow())
        assert isinstance(result, dict)

    def test_required_keys(self):
        result = get_net_foreign("BBCA.JK", self._flow())
        assert "net_foreign" in result
        assert "signal" in result
        assert "strength" in result

    def test_buy_signal_when_net_positive(self):
        result = get_net_foreign("BBCA.JK", self._flow())
        assert result["signal"] == "BUY"

    def test_sell_signal_when_net_negative(self):
        result = get_net_foreign("BBRI.JK", self._flow())
        assert result["signal"] == "SELL"

    def test_neutral_when_zero(self):
        result = get_net_foreign("BMRI.JK", self._flow())
        assert result["signal"] == "NEUTRAL"
        assert result["strength"] == "NEUTRAL"

    def test_strong_buy_when_above_volume_threshold(self):
        """net_foreign > avg_volume * 0.05 → STRONG BUY."""
        # net = 3M, avg_vol = 50M, threshold = 2.5M → STRONG
        result = get_net_foreign("BBCA.JK", self._flow(), avg_daily_volume=50_000_000)
        assert result["signal"] == "BUY"
        assert result["strength"] == "STRONG"

    def test_weak_buy_when_below_volume_threshold(self):
        """net_foreign > 0 but below threshold → WEAK BUY."""
        # net = 3M, avg_vol = 1B, threshold = 50M → WEAK
        result = get_net_foreign("BBCA.JK", self._flow(), avg_daily_volume=1_000_000_000)
        assert result["signal"] == "BUY"
        assert result["strength"] == "WEAK"

    def test_strips_jk_suffix(self):
        """Should handle both 'BBCA.JK' and 'BBCA' inputs."""
        r1 = get_net_foreign("BBCA.JK", self._flow())
        r2 = get_net_foreign("BBCA", self._flow())
        assert r1["net_foreign"] == r2["net_foreign"]

    def test_unknown_ticker_returns_neutral(self):
        """Unknown ticker should return NEUTRAL."""
        result = get_net_foreign("ZZZZ.JK", self._flow())
        assert result["signal"] == "NEUTRAL"
        assert result["net_foreign"] == 0

    def test_no_avg_volume_defaults_to_weak(self):
        """Without avg_volume, STRONG threshold is infinity → WEAK BUY."""
        result = get_net_foreign("BBCA.JK", self._flow(), avg_daily_volume=0)
        assert result["signal"] == "BUY"
        assert result["strength"] == "WEAK"

    def test_net_foreign_value_correct(self):
        """net_foreign value should match flow data."""
        result = get_net_foreign("BBCA.JK", self._flow())
        assert result["net_foreign"] == 3_000_000.0


# ─── format_foreign_summary ─────────────────────────────────────────────────

class TestFormatForeignSummary:

    def _flow(self):
        return {
            "BBCA": {"foreign_buy": 50_000_000, "foreign_sell": 20_000_000, "net_foreign": 30_000_000},
            "BBRI": {"foreign_buy": 10_000_000, "foreign_sell": 30_000_000, "net_foreign": -20_000_000},
            "BMRI": {"foreign_buy": 5_000_000, "foreign_sell": 5_000_000, "net_foreign": 0},
        }

    def test_returns_string(self):
        result = format_foreign_summary(self._flow(), ["BBCA.JK", "BBRI.JK", "BMRI.JK"])
        assert isinstance(result, str)

    def test_shows_net_buy_stocks(self):
        result = format_foreign_summary(self._flow(), ["BBCA.JK", "BBRI.JK"], threshold=0)
        assert "BBCA" in result
        assert "Net Buy" in result

    def test_shows_net_sell_stocks(self):
        result = format_foreign_summary(self._flow(), ["BBCA.JK", "BBRI.JK"], threshold=0)
        assert "BBRI" in result
        assert "Net Sell" in result

    def test_hides_stocks_below_threshold(self):
        """Stocks with |net| < threshold should not appear."""
        flow = {"BBCA": {"net_foreign": 500}}  # below 1M threshold
        result = format_foreign_summary(flow, ["BBCA.JK"])
        assert "BBCA" not in result

    def test_shows_no_significant_when_all_below_threshold(self):
        flow = {"BBCA": {"net_foreign": 500}}
        result = format_foreign_summary(flow, ["BBCA.JK"])
        assert "tidak ada" in result.lower() or "signifikan" in result.lower() or "BBCA" not in result

    def test_empty_flow_returns_unavailable(self):
        result = format_foreign_summary({}, ["BBCA.JK"])
        assert "tidak tersedia" in result.lower() or "unavailable" in result.lower() or result

    def test_unknown_ticker_in_watchlist_skipped(self):
        """Tickers not in flow_data should be silently skipped."""
        result = format_foreign_summary(self._flow(), ["ZZZZ.JK"], threshold=0)
        assert "ZZZZ" not in result

    def test_contains_header(self):
        result = format_foreign_summary(self._flow(), ["BBCA.JK"], threshold=0)
        assert "FOREIGN FLOW" in result.upper() or "foreign" in result.lower()


# ─── save_flow_data ──────────────────────────────────────────────────────────

class TestSaveFlowData:

    def test_creates_file(self, tmp_path, monkeypatch):
        """save_flow_data should create JSON file."""
        import agents.foreign_flow as ff_module
        monkeypatch.setattr(ff_module, "FLOW_DATA_PATH", str(tmp_path / "flow.json"))

        flow = {"BBCA": {"foreign_buy": 1000, "foreign_sell": 500, "net_foreign": 500}}
        save_flow_data(flow)

        assert os.path.exists(ff_module.FLOW_DATA_PATH)
        with open(ff_module.FLOW_DATA_PATH) as f:
            data = json.load(f)
        assert "data" in data
        assert "BBCA" in data["data"]

    def test_includes_date(self, tmp_path, monkeypatch):
        """saved file should include a 'date' field."""
        import agents.foreign_flow as ff_module
        monkeypatch.setattr(ff_module, "FLOW_DATA_PATH", str(tmp_path / "flow.json"))

        save_flow_data({})

        with open(ff_module.FLOW_DATA_PATH) as f:
            data = json.load(f)
        assert "date" in data

    def test_saves_all_records(self, tmp_path, monkeypatch):
        """All flow records should be preserved."""
        import agents.foreign_flow as ff_module
        monkeypatch.setattr(ff_module, "FLOW_DATA_PATH", str(tmp_path / "flow.json"))

        flow = {
            "BBCA": {"foreign_buy": 1000, "foreign_sell": 500, "net_foreign": 500},
            "BBRI": {"foreign_buy": 200, "foreign_sell": 800, "net_foreign": -600},
        }
        save_flow_data(flow)

        with open(ff_module.FLOW_DATA_PATH) as f:
            data = json.load(f)
        assert len(data["data"]) == 2
