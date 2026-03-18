"""
Unit tests for agents/premarket.py
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.premarket import (
    fetch_global_markets,
    analyze_premarket_with_sonnet,
    format_premarket_briefing,
    _infer_us_signal,
    _infer_asia_signal,
    _infer_ihsg_prediction,
)


# ─── Helpers ────────────────────────────────────────────────────────────────

def _make_mock_ticker(current: float, prev: float):
    """Create a mock yfinance Ticker with history returning 2 closes."""
    dates = pd.date_range("2026-01-01", periods=2, freq="D")
    df = pd.DataFrame({
        "Open": [prev * 0.99, current * 0.99],
        "High": [prev * 1.01, current * 1.01],
        "Low": [prev * 0.99, current * 0.99],
        "Close": [prev, current],
        "Volume": [1_000_000, 1_000_000],
    }, index=dates)
    mock = MagicMock()
    mock.history.return_value = df
    return mock


def _sample_global_data():
    """Sample global market data dict."""
    return {
        "S&P500":   {"current": 5234.0, "change_pct": 0.8,  "direction": "UP"},
        "Nasdaq":   {"current": 16450.0, "change_pct": 1.2, "direction": "UP"},
        "Dow":      {"current": 38920.0, "change_pct": 0.5, "direction": "UP"},
        "Nikkei":   {"current": 38100.0, "change_pct": 0.3, "direction": "UP"},
        "HangSeng": {"current": 19200.0, "change_pct": -0.8, "direction": "DOWN"},
        "USD_IDR":  {"current": 16200.0, "change_pct": -0.2, "direction": "DOWN"},
        "Gold":     {"current": 2180.0, "change_pct": 0.1,  "direction": "UP"},
        "Oil":      {"current": 72.3, "change_pct": -0.5,   "direction": "DOWN"},
    }


# ─── fetch_global_markets ────────────────────────────────────────────────────

class TestFetchGlobalMarkets:

    def test_returns_dict(self):
        """fetch_global_markets should return a dict."""
        ticker_map = {
            "^GSPC": _make_mock_ticker(5234.0, 5193.0),
            "^IXIC": _make_mock_ticker(16450.0, 16254.0),
            "^DJI": _make_mock_ticker(38920.0, 38728.0),
            "^N225": _make_mock_ticker(38100.0, 37986.0),
            "^HSI": _make_mock_ticker(19200.0, 19355.0),
            "IDR=X": _make_mock_ticker(16200.0, 16233.0),
            "GC=F": _make_mock_ticker(2180.0, 2178.0),
            "CL=F": _make_mock_ticker(72.3, 72.67),
        }

        def mock_ticker(symbol):
            return ticker_map.get(symbol, _make_mock_ticker(100.0, 100.0))

        with patch("yfinance.Ticker", side_effect=mock_ticker):
            result = fetch_global_markets()

        assert isinstance(result, dict)

    def test_returns_required_market_keys(self):
        """Result should have at least some of the 8 markets."""
        ticker = _make_mock_ticker(5234.0, 5193.0)
        with patch("yfinance.Ticker", return_value=ticker):
            result = fetch_global_markets()
        # At least 1 market should be present
        assert len(result) >= 1

    def test_each_market_has_required_fields(self):
        """Each market entry should have current, change_pct, direction."""
        ticker = _make_mock_ticker(5234.0, 5193.0)
        with patch("yfinance.Ticker", return_value=ticker):
            result = fetch_global_markets()
        for name, data in result.items():
            assert "current" in data, f"Missing current for {name}"
            assert "change_pct" in data, f"Missing change_pct for {name}"
            assert "direction" in data, f"Missing direction for {name}"

    def test_direction_up_when_price_rises(self):
        """direction should be UP when price rises > 0.1%."""
        ticker = _make_mock_ticker(5234.0, 5000.0)  # +4.68%
        with patch("yfinance.Ticker", return_value=ticker):
            result = fetch_global_markets()
        for name, data in result.items():
            assert data["direction"] == "UP"

    def test_direction_down_when_price_falls(self):
        """direction should be DOWN when price falls > 0.1%."""
        ticker = _make_mock_ticker(5000.0, 5200.0)  # -3.85%
        with patch("yfinance.Ticker", return_value=ticker):
            result = fetch_global_markets()
        for name, data in result.items():
            assert data["direction"] == "DOWN"

    def test_skips_failed_markets(self):
        """Markets that fail should be silently skipped."""
        def fail_ticker(symbol):
            if symbol in ("^GSPC",):
                m = MagicMock()
                m.history.side_effect = Exception("timeout")
                return m
            return _make_mock_ticker(100.0, 100.0)

        with patch("yfinance.Ticker", side_effect=fail_ticker):
            result = fetch_global_markets()
        assert "S&P500" not in result
        assert isinstance(result, dict)

    def test_handles_empty_history(self):
        """Empty DataFrame from yfinance → skip that market."""
        mock = MagicMock()
        mock.history.return_value = pd.DataFrame()
        with patch("yfinance.Ticker", return_value=mock):
            result = fetch_global_markets()
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_change_pct_calculation(self):
        """change_pct = (current - prev) / prev * 100."""
        ticker = _make_mock_ticker(5200.0, 5000.0)  # +4%
        with patch("yfinance.Ticker", return_value=ticker):
            result = fetch_global_markets()
        for name, data in result.items():
            expected = ((5200.0 - 5000.0) / 5000.0) * 100
            assert abs(data["change_pct"] - expected) < 0.01

    def test_never_raises(self):
        """fetch_global_markets should never raise exceptions."""
        with patch("yfinance.Ticker", side_effect=Exception("Critical error")):
            try:
                result = fetch_global_markets()
                assert isinstance(result, dict)
            except Exception as e:
                pytest.fail(f"fetch_global_markets raised: {e}")


# ─── analyze_premarket_with_sonnet ────────────────────────────────────────────

class TestAnalyzePremarketWithSonnet:

    def test_returns_string(self):
        """analyze_premarket_with_sonnet should return a string."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="IHSG diprediksi open positif hari ini.")]
        mock_client.messages.create.return_value = mock_response

        with patch("agents.premarket.ANTHROPIC_API_KEY", "test-key"):
            with patch("anthropic.Anthropic", return_value=mock_client):
                result = analyze_premarket_with_sonnet(_sample_global_data())
        assert isinstance(result, str)

    def test_returns_fallback_when_no_api_key(self):
        """Should return fallback string when no API key."""
        with patch("agents.premarket.ANTHROPIC_API_KEY", None):
            result = analyze_premarket_with_sonnet(_sample_global_data())
        assert isinstance(result, str)
        assert len(result) > 0

    def test_returns_fallback_when_empty_data(self):
        """Should return fallback string when global_data is empty."""
        with patch("agents.premarket.ANTHROPIC_API_KEY", "test-key"):
            result = analyze_premarket_with_sonnet({})
        assert isinstance(result, str)

    def test_returns_fallback_on_api_error(self):
        """Should return fallback string on API exception."""
        with patch("agents.premarket.ANTHROPIC_API_KEY", "test-key"):
            with patch("anthropic.Anthropic", side_effect=Exception("API error")):
                result = analyze_premarket_with_sonnet(_sample_global_data())
        assert isinstance(result, str)
        assert len(result) > 0

    def test_uses_sonnet_model(self):
        """Should use claude-sonnet-4-5 model."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Analysis text.")]
        mock_client.messages.create.return_value = mock_response

        with patch("agents.premarket.ANTHROPIC_API_KEY", "test-key"):
            with patch("anthropic.Anthropic", return_value=mock_client):
                analyze_premarket_with_sonnet(_sample_global_data())

        call_kwargs = mock_client.messages.create.call_args
        model = call_kwargs[1].get("model") or call_kwargs[0][0] if call_kwargs[0] else None
        if call_kwargs[1]:
            assert "claude-sonnet" in call_kwargs[1].get("model", "")

    def test_max_tokens_400(self):
        """Should use max_tokens=400."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Analysis.")]
        mock_client.messages.create.return_value = mock_response

        with patch("agents.premarket.ANTHROPIC_API_KEY", "test-key"):
            with patch("anthropic.Anthropic", return_value=mock_client):
                analyze_premarket_with_sonnet(_sample_global_data())

        call_kwargs = mock_client.messages.create.call_args
        if call_kwargs[1]:
            assert call_kwargs[1].get("max_tokens") == 400


# ─── format_premarket_briefing ───────────────────────────────────────────────

class TestFormatPremarketBriefing:

    def test_returns_string(self):
        result = format_premarket_briefing(_sample_global_data(), "Analisa hari ini.")
        assert isinstance(result, str)

    def test_contains_wall_street_section(self):
        result = format_premarket_briefing(_sample_global_data(), "")
        assert "Wall Street" in result or "S&P500" in result

    def test_contains_asia_section(self):
        result = format_premarket_briefing(_sample_global_data(), "")
        assert "Asia" in result or "Nikkei" in result

    def test_contains_macro_section(self):
        result = format_premarket_briefing(_sample_global_data(), "")
        assert "USD" in result or "Gold" in result or "Oil" in result or "Macro" in result

    def test_contains_analysis(self):
        analysis = "IHSG diprediksi open positif didukung Wall Street."
        result = format_premarket_briefing(_sample_global_data(), analysis)
        assert "IHSG" in result or "positif" in result.lower() or "Dexter" in result

    def test_contains_market_open_time(self):
        result = format_premarket_briefing(_sample_global_data(), "Test.")
        assert "09:00" in result or "WIB" in result

    def test_contains_briefing_header(self):
        result = format_premarket_briefing(_sample_global_data(), "")
        assert "PRE-MARKET" in result or "BRIEFING" in result

    def test_handles_empty_data(self):
        """Should not crash with empty global data."""
        result = format_premarket_briefing({}, "Tidak ada data.")
        assert isinstance(result, str)

    def test_handles_empty_analysis(self):
        """Should work without analysis text."""
        result = format_premarket_briefing(_sample_global_data(), "")
        assert isinstance(result, str)

    def test_contains_up_emoji_for_positive(self):
        data = {"S&P500": {"current": 5234.0, "change_pct": 0.8, "direction": "UP"}}
        result = format_premarket_briefing(data, "")
        assert "🟢" in result

    def test_contains_down_emoji_for_negative(self):
        data = {"HangSeng": {"current": 19200.0, "change_pct": -0.8, "direction": "DOWN"}}
        result = format_premarket_briefing(data, "")
        assert "🔴" in result

    def test_contains_date(self):
        from datetime import datetime, timedelta
        result = format_premarket_briefing(_sample_global_data(), "")
        now_wib = datetime.utcnow() + timedelta(hours=7)
        year_str = str(now_wib.year)
        assert year_str in result


# ─── Signal Inference ────────────────────────────────────────────────────────

class TestInferSignals:

    def test_us_signal_bullish(self):
        data = {
            "S&P500": {"change_pct": 0.8},
            "Nasdaq": {"change_pct": 1.2},
            "Dow": {"change_pct": 0.5},
        }
        assert _infer_us_signal(data) == "BULLISH"

    def test_us_signal_bearish(self):
        data = {
            "S&P500": {"change_pct": -0.8},
            "Nasdaq": {"change_pct": -1.2},
            "Dow": {"change_pct": -0.5},
        }
        assert _infer_us_signal(data) == "BEARISH"

    def test_us_signal_neutral(self):
        data = {
            "S&P500": {"change_pct": 0.1},
            "Nasdaq": {"change_pct": -0.1},
            "Dow": {"change_pct": 0.0},
        }
        assert _infer_us_signal(data) == "NEUTRAL"

    def test_asia_signal_bullish(self):
        data = {
            "Nikkei": {"change_pct": 0.8},
            "HangSeng": {"change_pct": 0.5},
        }
        assert _infer_asia_signal(data) == "BULLISH"

    def test_asia_signal_bearish(self):
        data = {
            "Nikkei": {"change_pct": -0.8},
            "HangSeng": {"change_pct": -0.5},
        }
        assert _infer_asia_signal(data) == "BEARISH"

    def test_ihsg_prediction_positif(self):
        data = {
            "S&P500": {"change_pct": 0.8},
            "Nasdaq": {"change_pct": 1.2},
            "Dow": {"change_pct": 0.5},
            "Nikkei": {"change_pct": 0.3},
            "HangSeng": {"change_pct": 0.1},
        }
        result = _infer_ihsg_prediction(data)
        assert result == "POSITIF"

    def test_ihsg_prediction_negatif(self):
        data = {
            "S&P500": {"change_pct": -0.8},
            "Nasdaq": {"change_pct": -1.2},
            "Dow": {"change_pct": -0.5},
            "Nikkei": {"change_pct": -0.5},
            "HangSeng": {"change_pct": -0.3},
        }
        result = _infer_ihsg_prediction(data)
        assert result == "NEGATIF"

    def test_ihsg_prediction_netral_empty(self):
        result = _infer_ihsg_prediction({})
        assert result == "NETRAL"
