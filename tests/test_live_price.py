"""
Tests for agents/live_price.py

Coverage:
  - is_fresh() logic (inside/outside market hours)
  - is_market_hours() logic
  - Source fallback (mock each source)
  - Graceful failure when all sources fail
  - format_live_prices()
  - get_live_price() integration (all mocked)
  - get_live_prices() batch
  - run_live_price_check() smoke test

All HTTP calls are mocked — no real network requests.
"""

import time
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch, call

import pytest

# Module under test
from agents.live_price import (
    is_fresh,
    is_market_hours,
    get_live_price,
    get_live_prices,
    format_live_prices,
    run_live_price_check,
    _parse_google_price,
    _parse_google_pct,
    _ts_to_wib_str,
    _fetch_yf_v8,
    _fetch_yf_v7,
    _fetch_stockbit,
    WATCHLIST,
)


# ─── Fixtures & helpers ───────────────────────────────────────────────────────

def _make_ts(offset_minutes: int = 0, tz: timezone = timezone.utc) -> int:
    """Return unix timestamp = now + offset_minutes."""
    return int((datetime.now(tz) + timedelta(minutes=offset_minutes)).timestamp())


def _market_open_ts() -> datetime:
    """Datetime that falls during IDX market hours (Mon 05:00 UTC)."""
    # Find next or current Monday
    now = datetime.now(timezone.utc)
    days_until_mon = (7 - now.weekday()) % 7
    mon = now + timedelta(days=days_until_mon)
    return mon.replace(hour=4, minute=0, second=0, microsecond=0)  # 04:00 UTC = 11:00 WIB


def _market_closed_ts() -> datetime:
    """Datetime that falls outside IDX market hours (Saturday 10:00 UTC)."""
    now = datetime.now(timezone.utc)
    days_until_sat = (5 - now.weekday()) % 7
    sat = now + timedelta(days=days_until_sat)
    return sat.replace(hour=10, minute=0, second=0, microsecond=0)


# ─── 1. is_market_hours() ────────────────────────────────────────────────────

class TestIsMarketHours:
    def test_weekday_during_trading_hours(self):
        """Mon 04:00 UTC = 11:00 WIB → should be open."""
        dt = datetime(2026, 3, 23, 4, 0, tzinfo=timezone.utc)  # Monday
        assert is_market_hours(dt) is True

    def test_weekday_just_before_open(self):
        """Mon 01:54 UTC = 08:54 WIB → before open (01:55 UTC)."""
        dt = datetime(2026, 3, 23, 1, 54, tzinfo=timezone.utc)
        assert is_market_hours(dt) is False

    def test_weekday_at_open(self):
        """Mon 01:55 UTC = 08:55 WIB → just opened."""
        dt = datetime(2026, 3, 23, 1, 55, tzinfo=timezone.utc)
        assert is_market_hours(dt) is True

    def test_weekday_at_close(self):
        """Mon 08:00 UTC = 15:00 WIB → closed (close is exclusive)."""
        dt = datetime(2026, 3, 23, 8, 0, tzinfo=timezone.utc)
        assert is_market_hours(dt) is False

    def test_weekday_after_close(self):
        """Mon 10:00 UTC = 17:00 WIB → after close."""
        dt = datetime(2026, 3, 23, 10, 0, tzinfo=timezone.utc)
        assert is_market_hours(dt) is False

    def test_saturday(self):
        """Saturday → always closed."""
        dt = datetime(2026, 3, 21, 4, 0, tzinfo=timezone.utc)  # Saturday
        assert is_market_hours(dt) is False

    def test_sunday(self):
        """Sunday → always closed."""
        dt = datetime(2026, 3, 22, 4, 0, tzinfo=timezone.utc)  # Sunday
        assert is_market_hours(dt) is False

    def test_friday_during_hours(self):
        """Friday during trading hours → open."""
        dt = datetime(2026, 3, 20, 5, 0, tzinfo=timezone.utc)  # Friday
        assert is_market_hours(dt) is True


# ─── 2. is_fresh() ───────────────────────────────────────────────────────────

class TestIsFresh:
    def test_fresh_during_market_hours(self):
        """10-min-old data during market hours → fresh."""
        ts = _make_ts(offset_minutes=-10)
        with patch("agents.live_price._now_utc") as mock_now:
            mock_now.return_value = datetime(2026, 3, 23, 4, 0, tzinfo=timezone.utc)
            # ts should be 10 min ago from mock_now
            ts_val = int((datetime(2026, 3, 23, 3, 50, tzinfo=timezone.utc)).timestamp())
            assert is_fresh(ts_val, max_age_minutes=30) is True

    def test_stale_during_market_hours(self):
        """45-min-old data during market hours → stale."""
        with patch("agents.live_price._now_utc") as mock_now:
            mock_now.return_value = datetime(2026, 3, 23, 4, 0, tzinfo=timezone.utc)
            ts_val = int((datetime(2026, 3, 23, 3, 14, tzinfo=timezone.utc)).timestamp())  # 46 min ago
            assert is_fresh(ts_val, max_age_minutes=30) is False

    def test_old_data_outside_market_hours(self):
        """Day-old data outside market hours → still fresh (last close OK)."""
        with patch("agents.live_price._now_utc") as mock_now:
            mock_now.return_value = datetime(2026, 3, 21, 10, 0, tzinfo=timezone.utc)  # Saturday
            ts_val = int((datetime(2026, 3, 20, 8, 0, tzinfo=timezone.utc)).timestamp())  # yesterday
            assert is_fresh(ts_val, max_age_minutes=30) is True

    def test_none_timestamp_always_false(self):
        """None timestamp → always stale."""
        assert is_fresh(None) is False

    def test_zero_timestamp_always_false(self):
        """Zero timestamp → always stale."""
        assert is_fresh(0) is False

    def test_custom_max_age(self):
        """Respects custom max_age_minutes parameter."""
        with patch("agents.live_price._now_utc") as mock_now:
            mock_now.return_value = datetime(2026, 3, 23, 4, 0, tzinfo=timezone.utc)
            ts_val = int((datetime(2026, 3, 23, 3, 56, tzinfo=timezone.utc)).timestamp())  # 4 min ago
            assert is_fresh(ts_val, max_age_minutes=3) is False  # 3-min threshold → stale
            assert is_fresh(ts_val, max_age_minutes=5) is True   # 5-min threshold → fresh

    def test_boundary_exact_threshold(self):
        """Data exactly at threshold boundary → stale (must be strictly less than)."""
        with patch("agents.live_price._now_utc") as mock_now:
            now_dt = datetime(2026, 3, 23, 4, 0, tzinfo=timezone.utc)
            mock_now.return_value = now_dt
            ts_val = int((now_dt - timedelta(minutes=30)).timestamp())  # exactly 30 min ago
            # 30 * 60 = 1800 seconds; not < 1800 → stale
            assert is_fresh(ts_val, max_age_minutes=30) is False


# ─── 3. Helper functions ──────────────────────────────────────────────────────

class TestHelpers:
    def test_parse_google_price_rp(self):
        assert _parse_google_price("Rp 3,050.00") == pytest.approx(3050.0)

    def test_parse_google_price_no_currency(self):
        assert _parse_google_price("6,775.00") == pytest.approx(6775.0)

    def test_parse_google_price_invalid(self):
        assert _parse_google_price("N/A") is None

    def test_parse_google_pct_negative(self):
        assert _parse_google_pct("-5.73%") == pytest.approx(-5.73)

    def test_parse_google_pct_positive(self):
        assert _parse_google_pct("+2.11%") == pytest.approx(2.11)

    def test_parse_google_pct_invalid(self):
        assert _parse_google_pct("N/A") is None

    def test_ts_to_wib_str(self):
        # 2026-03-17 09:14:46 UTC → +7 = 16:14:46 WIB
        ts = 1773738886
        result = _ts_to_wib_str(ts)
        assert "WIB" in result
        assert "2026" in result

    def test_ts_to_wib_str_none(self):
        assert _ts_to_wib_str(None) == "unknown"


# ─── 4. Individual source mocks ───────────────────────────────────────────────

class TestSourceFetchers:
    def test_fetch_yf_v8_success(self):
        """YF v8 returns price and timestamp."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "chart": {
                "result": [{
                    "meta": {
                        "regularMarketPrice": 3050.0,
                        "regularMarketTime": 1774320000,  # fresh ts
                        "regularMarketChangePercent": -5.73,
                    }
                }]
            }
        }
        with patch("agents.live_price.requests.get", return_value=mock_resp):
            result = _fetch_yf_v8("TLKM")
        assert result is not None
        assert result["price"] == 3050.0
        assert result["source"] == "yf_v8"
        assert result["change_pct"] == pytest.approx(-5.73)

    def test_fetch_yf_v8_http_error(self):
        """YF v8 returns None on HTTP 401."""
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        with patch("agents.live_price.requests.get", return_value=mock_resp):
            result = _fetch_yf_v8("TLKM")
        assert result is None

    def test_fetch_yf_v8_empty_result(self):
        """YF v8 returns None when result list is empty."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"chart": {"result": []}}
        with patch("agents.live_price.requests.get", return_value=mock_resp):
            result = _fetch_yf_v8("TLKM")
        assert result is None

    def test_fetch_yf_v8_network_error(self):
        """YF v8 returns None on connection error."""
        import requests as req_lib
        with patch("agents.live_price.requests.get", side_effect=req_lib.RequestException("timeout")):
            result = _fetch_yf_v8("TLKM")
        assert result is None

    def test_fetch_yf_v7_no_crumb(self):
        """YF v7 returns None when crumb unavailable."""
        with patch("agents.live_price._get_yf_crumb", return_value=None):
            result = _fetch_yf_v7("TLKM")
        assert result is None

    def test_fetch_stockbit_http_error(self):
        """Stockbit returns None on 403."""
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        with patch("agents.live_price.requests.get", return_value=mock_resp):
            result = _fetch_stockbit("TLKM")
        assert result is None

    def test_fetch_stockbit_success(self):
        """Stockbit returns price on success."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": {"last_price": 3050, "change_percent": -5.73, "timestamp": 1774320000}
        }
        with patch("agents.live_price.requests.get", return_value=mock_resp):
            result = _fetch_stockbit("TLKM")
        assert result is not None
        assert result["price"] == 3050.0
        assert result["source"] == "stockbit"


# ─── 5. get_live_price() integration with fallback ───────────────────────────

class TestGetLivePrice:
    def _fresh_ts(self) -> int:
        """Return a timestamp from 5 minutes ago (fresh during market hours)."""
        return int((datetime.now(timezone.utc) - timedelta(minutes=5)).timestamp())

    def _stale_ts(self) -> int:
        """Return a timestamp from March 17 (stale)."""
        return 1773738886  # 2026-03-17 09:14 UTC

    def test_returns_google_finance_when_fresh(self):
        """get_live_price picks Google Finance when it returns fresh data."""
        fresh_ts = self._fresh_ts()
        with patch("agents.live_price._fetch_google_finance") as mock_gf, \
             patch("agents.live_price.is_market_hours", return_value=True):
            mock_gf.return_value = {
                "price": 3050.0,
                "change_pct": -5.73,
                "timestamp_utc": fresh_ts,
                "source": "google_finance",
            }
            result = get_live_price("TLKM")

        assert result["ticker"] == "TLKM"
        assert result["price"] == 3050.0
        assert result["source"] == "google_finance"
        assert result["is_fresh"] is True
        assert result["error"] is None

    def test_falls_back_to_yf_v8_when_google_fails(self):
        """Falls back to YF v8 when Google Finance fails."""
        fresh_ts = self._fresh_ts()
        with patch("agents.live_price._fetch_google_finance", return_value=None), \
             patch("agents.live_price._fetch_yf_v8") as mock_v8, \
             patch("agents.live_price.is_market_hours", return_value=True):
            mock_v8.return_value = {
                "price": 3050.0,
                "change_pct": None,
                "timestamp_utc": fresh_ts,
                "source": "yf_v8",
            }
            result = get_live_price("TLKM")

        assert result["source"] == "yf_v8"
        assert result["is_fresh"] is True

    def test_falls_back_to_yf_v7_when_v8_fails(self):
        """Falls back to YF v7 when Google and v8 fail."""
        fresh_ts = self._fresh_ts()
        with patch("agents.live_price._fetch_google_finance", return_value=None), \
             patch("agents.live_price._fetch_yf_v8", return_value=None), \
             patch("agents.live_price._fetch_yf_v7") as mock_v7, \
             patch("agents.live_price.is_market_hours", return_value=True):
            mock_v7.return_value = {
                "price": 3060.0,
                "change_pct": -5.58,
                "timestamp_utc": fresh_ts,
                "source": "yf_v7",
            }
            result = get_live_price("TLKM")

        assert result["source"] == "yf_v7"
        assert result["price"] == 3060.0

    def test_falls_back_to_stockbit_when_others_fail(self):
        """Falls back to Stockbit as last resort."""
        fresh_ts = self._fresh_ts()
        with patch("agents.live_price._fetch_google_finance", return_value=None), \
             patch("agents.live_price._fetch_yf_v8", return_value=None), \
             patch("agents.live_price._fetch_yf_v7", return_value=None), \
             patch("agents.live_price._fetch_stockbit") as mock_sb, \
             patch("agents.live_price.is_market_hours", return_value=True):
            mock_sb.return_value = {
                "price": 3055.0,
                "change_pct": -5.68,
                "timestamp_utc": fresh_ts,
                "source": "stockbit",
            }
            result = get_live_price("TLKM")

        assert result["source"] == "stockbit"
        assert result["price"] == 3055.0

    def test_returns_stale_when_all_sources_stale(self):
        """Returns best stale result when all sources have old data (Nyepi scenario)."""
        stale_ts = self._stale_ts()
        with patch("agents.live_price._fetch_google_finance") as mock_gf, \
             patch("agents.live_price._fetch_yf_v8", return_value=None), \
             patch("agents.live_price._fetch_yf_v7", return_value=None), \
             patch("agents.live_price._fetch_stockbit", return_value=None), \
             patch("agents.live_price.is_market_hours", return_value=True):
            mock_gf.return_value = {
                "price": 3050.0,
                "change_pct": -5.73,
                "timestamp_utc": stale_ts,
                "source": "google_finance",
            }
            result = get_live_price("TLKM")

        assert result["price"] == 3050.0
        assert result["is_fresh"] is False  # stale
        assert result["source"] == "google_finance"
        assert result["error"] is None  # not an error, just stale

    def test_all_sources_fail_returns_error_dict(self):
        """Returns error dict when ALL sources fail completely."""
        with patch("agents.live_price._fetch_google_finance", return_value=None), \
             patch("agents.live_price._fetch_yf_v8", return_value=None), \
             patch("agents.live_price._fetch_yf_v7", return_value=None), \
             patch("agents.live_price._fetch_stockbit", return_value=None):
            result = get_live_price("TLKM")

        assert result["price"] is None
        assert result["is_fresh"] is False
        assert result["error"] is not None

    def test_ticker_normalization_jk_suffix(self):
        """TLKM.JK and TLKM both normalize to TLKM."""
        fresh_ts = self._fresh_ts()
        with patch("agents.live_price._fetch_google_finance") as mock_gf, \
             patch("agents.live_price.is_market_hours", return_value=True):
            mock_gf.return_value = {
                "price": 3050.0, "change_pct": None,
                "timestamp_utc": fresh_ts, "source": "google_finance",
            }
            result_with_jk = get_live_price("TLKM.JK")
            result_without = get_live_price("TLKM")

        assert result_with_jk["ticker"] == "TLKM"
        assert result_without["ticker"] == "TLKM"

    def test_never_crashes_on_exception(self):
        """get_live_price never raises — even if sources throw exceptions."""
        with patch("agents.live_price._fetch_google_finance", side_effect=RuntimeError("boom")), \
             patch("agents.live_price._fetch_yf_v8", side_effect=ValueError("bad")), \
             patch("agents.live_price._fetch_yf_v7", return_value=None), \
             patch("agents.live_price._fetch_stockbit", return_value=None):
            result = get_live_price("TLKM")

        # Should not raise; should return a valid dict (error dict when all fail)
        assert isinstance(result, dict)
        assert "ticker" in result
        assert "is_fresh" in result
        # All sources raised/returned None → error state
        assert result.get("error") is not None or result.get("price") is None

    def test_stale_outside_market_hours_is_fresh(self):
        """March 17 data outside market hours should be marked is_fresh=True."""
        stale_ts = self._stale_ts()
        with patch("agents.live_price._fetch_google_finance") as mock_gf, \
             patch("agents.live_price.is_market_hours", return_value=False):  # market CLOSED
            mock_gf.return_value = {
                "price": 3050.0, "change_pct": -5.73,
                "timestamp_utc": stale_ts, "source": "google_finance",
            }
            result = get_live_price("TLKM")

        # Outside market hours → any data is "fresh enough"
        assert result["is_fresh"] is True


# ─── 6. get_live_prices() batch ──────────────────────────────────────────────

class TestGetLivePrices:
    def test_batch_returns_all_tickers(self):
        """get_live_prices returns a dict keyed by normalized ticker."""
        fresh_ts = int((datetime.now(timezone.utc) - timedelta(minutes=5)).timestamp())
        mock_result = {
            "price": 3050.0, "change_pct": -5.73,
            "timestamp_utc": fresh_ts, "source": "google_finance",
        }
        with patch("agents.live_price._fetch_google_finance", return_value=mock_result), \
             patch("agents.live_price.is_market_hours", return_value=True), \
             patch("agents.live_price.time.sleep"):  # skip sleep in tests
            result = get_live_prices(["TLKM", "BBCA", "BBRI"])

        assert set(result.keys()) == {"TLKM", "BBCA", "BBRI"}
        for ticker, d in result.items():
            assert d["price"] == 3050.0

    def test_batch_strips_jk_suffix(self):
        """get_live_prices normalizes .JK suffix in batch."""
        fresh_ts = int((datetime.now(timezone.utc) - timedelta(minutes=5)).timestamp())
        mock_result = {
            "price": 3050.0, "change_pct": None,
            "timestamp_utc": fresh_ts, "source": "google_finance",
        }
        with patch("agents.live_price._fetch_google_finance", return_value=mock_result), \
             patch("agents.live_price.is_market_hours", return_value=True), \
             patch("agents.live_price.time.sleep"):
            result = get_live_prices(["TLKM.JK", "BBCA.JK"])

        assert "TLKM" in result
        assert "BBCA" in result
        assert "TLKM.JK" not in result


# ─── 7. format_live_prices() ─────────────────────────────────────────────────

class TestFormatLivePrices:
    def _sample_prices(self) -> dict:
        return {
            "TLKM": {
                "ticker": "TLKM", "price": 3050.0, "change_pct": -5.73,
                "source": "google_finance", "timestamp": "2026-03-24 09:15 WIB",
                "timestamp_utc": 1774320000, "is_fresh": True, "error": None,
            },
            "BBCA": {
                "ticker": "BBCA", "price": 6775.0, "change_pct": None,
                "source": "yf_v8", "timestamp": "2026-03-17 16:14 WIB",
                "timestamp_utc": 1773738886, "is_fresh": False, "error": None,
            },
            "FAIL": {
                "ticker": "FAIL", "price": None, "change_pct": None,
                "source": None, "timestamp": "unknown",
                "timestamp_utc": None, "is_fresh": False, "error": "All sources failed",
            },
        }

    def test_output_is_string(self):
        output = format_live_prices(self._sample_prices())
        assert isinstance(output, str)

    def test_output_contains_tickers(self):
        output = format_live_prices(self._sample_prices())
        assert "TLKM" in output
        assert "BBCA" in output

    def test_output_shows_fresh_indicator(self):
        output = format_live_prices(self._sample_prices())
        assert "✅" in output  # TLKM is fresh
        assert "⚠️" in output  # BBCA is stale

    def test_output_shows_error_ticker(self):
        output = format_live_prices(self._sample_prices())
        assert "FAIL" in output
        assert "ERROR" in output

    def test_empty_prices(self):
        output = format_live_prices({})
        assert isinstance(output, str)


# ─── 8. Smoke test run_live_price_check() ────────────────────────────────────

class TestRunLivePriceCheck:
    def test_smoke_returns_dict(self):
        """run_live_price_check returns dict without crashing (all mocked)."""
        fresh_ts = int((datetime.now(timezone.utc) - timedelta(minutes=5)).timestamp())
        mock_result = {
            "price": 3050.0, "change_pct": -5.73,
            "timestamp_utc": fresh_ts, "source": "google_finance",
        }
        with patch("agents.live_price._fetch_google_finance", return_value=mock_result), \
             patch("agents.live_price.is_market_hours", return_value=True), \
             patch("agents.live_price.time.sleep"):
            result = run_live_price_check(["TLKM", "BBCA"])

        assert isinstance(result, dict)
        assert "TLKM" in result
        assert "BBCA" in result

    def test_uses_default_watchlist_if_none(self):
        """run_live_price_check uses WATCHLIST constant by default."""
        fresh_ts = int((datetime.now(timezone.utc) - timedelta(minutes=5)).timestamp())
        mock_result = {
            "price": 3050.0, "change_pct": None,
            "timestamp_utc": fresh_ts, "source": "google_finance",
        }
        with patch("agents.live_price._fetch_google_finance", return_value=mock_result), \
             patch("agents.live_price.is_market_hours", return_value=True), \
             patch("agents.live_price.time.sleep"):
            result = run_live_price_check()

        # Should have all 22 watchlist tickers
        assert len(result) == len(WATCHLIST)


# ─── 9. WATCHLIST constant ───────────────────────────────────────────────────

class TestWatchlist:
    def test_watchlist_has_22_tickers(self):
        assert len(WATCHLIST) == 22

    def test_watchlist_no_jk_suffix(self):
        """WATCHLIST uses bare tickers without .JK."""
        for t in WATCHLIST:
            assert not t.endswith(".JK"), f"{t} should not have .JK suffix"

    def test_watchlist_unique(self):
        assert len(WATCHLIST) == len(set(WATCHLIST))
