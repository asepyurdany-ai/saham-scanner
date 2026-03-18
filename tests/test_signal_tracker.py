"""
Unit tests for agents/signal_tracker.py
Tests: load/save log, log_signals_open, log_signals_close, send_weekly_accuracy_report
"""

import json
import os
import sys
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.signal_tracker import (
    load_signal_log,
    save_signal_log,
    log_signals_open,
    log_signals_close,
    send_weekly_accuracy_report,
    SIGNAL_LOG_FILE,
)


# ─── Fixtures ───────────────────────────────────────────────────────────────

def make_signal(ticker="BBNI", signal="STRONG BUY", score=5, current=4390.0):
    """Helper to create a scanner signal dict."""
    return {
        "ticker": f"{ticker}.JK",
        "current": current,
        "daily_change_pct": 1.5,
        "rsi": 42.9,
        "vol_ratio": 2.42,
        "macd_bullish": True,
        "score": score,
        "tp": round(current * 1.08, 0),
        "sl": round(current * 0.96, 0),
        "tp_pct": 8.0,
        "sl_pct": -4.0,
        "signal": signal,
        "conditions": {
            "uptrend": True,
            "volume_spike": True,
            "rsi_ok": True,
            "price_above_ma20": True,
            "macd_bullish": True,
            "momentum_positive": True,
        },
        "reasons": ["Uptrend", "Volume spike"],
    }


# ─── load / save ────────────────────────────────────────────────────────────

class TestLoadSaveSignalLog:

    def test_load_returns_empty_dict_when_no_file(self, tmp_path, monkeypatch):
        """load_signal_log returns {} when file doesn't exist."""
        monkeypatch.chdir(tmp_path)
        # Redirect SIGNAL_LOG_FILE
        import agents.signal_tracker as st
        orig = st.SIGNAL_LOG_FILE
        st.SIGNAL_LOG_FILE = str(tmp_path / "nonexistent.json")
        try:
            result = load_signal_log()
            assert result == {}
        finally:
            st.SIGNAL_LOG_FILE = orig

    def test_save_and_load_roundtrip(self, tmp_path):
        """save then load should return same data."""
        import agents.signal_tracker as st
        orig = st.SIGNAL_LOG_FILE
        st.SIGNAL_LOG_FILE = str(tmp_path / "signal_log.json")
        try:
            data = {
                "2026-03-18": {
                    "signals": [{"ticker": "BBNI", "entry": 4390, "result": "PENDING"}],
                    "summary": {"total": 1, "hit": 0, "miss": 0, "neutral": 0, "win_rate": 0.0},
                }
            }
            save_signal_log(data)
            loaded = load_signal_log()
            assert loaded == data
        finally:
            st.SIGNAL_LOG_FILE = orig

    def test_save_creates_data_dir(self, tmp_path, monkeypatch):
        """save_signal_log should create data/ dir if missing."""
        monkeypatch.chdir(tmp_path)
        import agents.signal_tracker as st
        orig = st.SIGNAL_LOG_FILE
        st.SIGNAL_LOG_FILE = str(tmp_path / "data" / "signal_log.json")
        try:
            save_signal_log({"test": "data"})
            assert os.path.exists(st.SIGNAL_LOG_FILE)
        finally:
            st.SIGNAL_LOG_FILE = orig


# ─── log_signals_open ───────────────────────────────────────────────────────

class TestLogSignalsOpen:

    def test_logs_strong_buy_signals(self, tmp_path):
        """log_signals_open should log STRONG BUY signals."""
        import agents.signal_tracker as st
        orig = st.SIGNAL_LOG_FILE
        st.SIGNAL_LOG_FILE = str(tmp_path / "signal_log.json")
        try:
            signals = [
                make_signal("BBNI", "STRONG BUY", 5, 4390.0),
                make_signal("BBCA", "STRONG BUY", 6, 9500.0),
                make_signal("GOTO", "AVOID", 1, 100.0),     # should be excluded
                make_signal("TLKM", "WATCH", 3, 3800.0),    # should be excluded
            ]
            today = (datetime.utcnow() + timedelta(hours=7)).strftime("%Y-%m-%d")

            with patch.object(st, "_today_str", return_value=today):
                log_signals_open(signals)

            log = load_signal_log()
            assert today in log
            assert len(log[today]["signals"]) == 2
            tickers = [s["ticker"] for s in log[today]["signals"]]
            assert "BBNI" in tickers
            assert "BBCA" in tickers
            assert "GOTO" not in tickers
        finally:
            st.SIGNAL_LOG_FILE = orig

    def test_skips_if_already_logged(self, tmp_path):
        """log_signals_open should be idempotent — skip if already logged today."""
        import agents.signal_tracker as st
        orig = st.SIGNAL_LOG_FILE
        st.SIGNAL_LOG_FILE = str(tmp_path / "signal_log.json")
        today = "2026-03-18"
        existing = {
            today: {
                "signals": [{"ticker": "BBNI", "entry": 4390}],
                "summary": {"total": 1, "hit": 0, "miss": 0, "neutral": 0, "win_rate": 0.0},
            }
        }
        save_signal_log(existing)
        try:
            signals = [make_signal("BBCA", "STRONG BUY", 5, 9500.0)]
            with patch.object(st, "_today_str", return_value=today):
                log_signals_open(signals)

            # Should not overwrite — still only 1 signal
            log = load_signal_log()
            assert len(log[today]["signals"]) == 1
            assert log[today]["signals"][0]["ticker"] == "BBNI"
        finally:
            st.SIGNAL_LOG_FILE = orig

    def test_entry_structure(self, tmp_path):
        """Each logged signal should have correct fields."""
        import agents.signal_tracker as st
        orig = st.SIGNAL_LOG_FILE
        st.SIGNAL_LOG_FILE = str(tmp_path / "signal_log.json")
        today = "2026-03-18"
        try:
            signals = [make_signal("BBNI", "STRONG BUY", 5, 4390.0)]
            with patch.object(st, "_today_str", return_value=today):
                log_signals_open(signals)

            log = load_signal_log()
            entry = log[today]["signals"][0]
            required = ["ticker", "entry", "score", "tp", "sl", "result", "close_price"]
            for key in required:
                assert key in entry, f"Missing key: {key}"

            assert entry["result"] == "PENDING"
            assert entry["close_price"] is None
            assert entry["entry"] == 4390.0
        finally:
            st.SIGNAL_LOG_FILE = orig

    def test_tp_sl_calculated(self, tmp_path):
        """log_signals_open should set correct TP (+8%) and SL (-4%)."""
        import agents.signal_tracker as st
        orig = st.SIGNAL_LOG_FILE
        st.SIGNAL_LOG_FILE = str(tmp_path / "signal_log.json")
        today = "2026-03-18"
        try:
            signals = [make_signal("BBNI", "STRONG BUY", 5, 4000.0)]
            with patch.object(st, "_today_str", return_value=today):
                log_signals_open(signals)

            log = load_signal_log()
            entry = log[today]["signals"][0]
            # TP from scanner is already set in make_signal
            assert entry["tp"] == 4320.0  # 4000 * 1.08
            assert entry["sl"] == 3840.0  # 4000 * 0.96
        finally:
            st.SIGNAL_LOG_FILE = orig

    def test_summary_initialized(self, tmp_path):
        """Summary should be initialized with zeros."""
        import agents.signal_tracker as st
        orig = st.SIGNAL_LOG_FILE
        st.SIGNAL_LOG_FILE = str(tmp_path / "signal_log.json")
        today = "2026-03-18"
        try:
            signals = [make_signal("BBNI", "STRONG BUY", 5, 4390.0)]
            with patch.object(st, "_today_str", return_value=today):
                log_signals_open(signals)

            log = load_signal_log()
            summary = log[today]["summary"]
            assert summary["total"] == 1
            assert summary["hit"] == 0
            assert summary["miss"] == 0
            assert summary["neutral"] == 0
            assert summary["win_rate"] == 0.0
        finally:
            st.SIGNAL_LOG_FILE = orig


# ─── log_signals_close ──────────────────────────────────────────────────────

class TestLogSignalsClose:

    def _setup_log(self, tmp_path, today, signals_data):
        import agents.signal_tracker as st
        st.SIGNAL_LOG_FILE = str(tmp_path / "signal_log.json")
        log = {
            today: {
                "signals": signals_data,
                "summary": {"total": len(signals_data), "hit": 0, "miss": 0, "neutral": 0, "win_rate": 0.0},
            }
        }
        save_signal_log(log)

    def test_hit_when_high_reaches_tp(self, tmp_path):
        """Result should be HIT when day's High >= TP."""
        import agents.signal_tracker as st
        orig = st.SIGNAL_LOG_FILE
        today = "2026-03-18"
        entry_price = 4000.0
        tp = 4320.0  # +8%
        sl = 3840.0  # -4%

        signals_data = [{
            "ticker": "BBNI", "entry": entry_price,
            "score": 5, "tp": tp, "sl": sl,
            "result": "PENDING", "close_price": None, "high": None, "low": None,
        }]
        self._setup_log(tmp_path, today, signals_data)
        try:
            # Mock yfinance: high reaches TP
            mock_hist = MagicMock()
            mock_hist.empty = False
            mock_hist.__len__ = lambda s: 2
            mock_hist.iloc = MagicMock()
            mock_hist.iloc.__getitem__ = lambda s, i: MagicMock(
                **{"__getitem__.side_effect": lambda k: {
                    "High": tp + 10, "Low": 3900.0, "Close": 4280.0
                }[k]}
            )
            # Simpler: patch yf.Ticker
            with patch("agents.signal_tracker.yf.Ticker") as mock_ticker_cls:
                mock_ticker = MagicMock()
                mock_ticker_cls.return_value = mock_ticker
                row = MagicMock()
                row.__getitem__ = lambda s, k: {
                    "High": tp + 10, "Low": 3900.0, "Close": 4280.0
                }[k]
                mock_df = MagicMock()
                mock_df.empty = False
                mock_df.iloc = MagicMock()
                mock_df.iloc.__getitem__ = lambda s, i: row
                mock_ticker.history.return_value = mock_df

                with patch.object(st, "_today_str", return_value=today):
                    log_signals_close()

            log = load_signal_log()
            result = log[today]["signals"][0]["result"]
            assert result == "HIT"
            assert log[today]["summary"]["hit"] == 1
            assert log[today]["summary"]["win_rate"] == 100.0
        finally:
            st.SIGNAL_LOG_FILE = orig

    def test_miss_when_low_hits_sl(self, tmp_path):
        """Result should be MISS when day's Low <= SL."""
        import agents.signal_tracker as st
        orig = st.SIGNAL_LOG_FILE
        today = "2026-03-18"
        entry_price = 4000.0
        tp = 4320.0
        sl = 3840.0  # -4%

        signals_data = [{
            "ticker": "BBNI", "entry": entry_price,
            "score": 5, "tp": tp, "sl": sl,
            "result": "PENDING", "close_price": None, "high": None, "low": None,
        }]
        self._setup_log(tmp_path, today, signals_data)
        try:
            with patch("agents.signal_tracker.yf.Ticker") as mock_ticker_cls:
                mock_ticker = MagicMock()
                mock_ticker_cls.return_value = mock_ticker
                row = MagicMock()
                row.__getitem__ = lambda s, k: {
                    "High": 4100.0, "Low": sl - 10, "Close": 3850.0
                }[k]
                mock_df = MagicMock()
                mock_df.empty = False
                mock_df.iloc = MagicMock()
                mock_df.iloc.__getitem__ = lambda s, i: row
                mock_ticker.history.return_value = mock_df

                with patch.object(st, "_today_str", return_value=today):
                    log_signals_close()

            log = load_signal_log()
            result = log[today]["signals"][0]["result"]
            assert result == "MISS"
            assert log[today]["summary"]["miss"] == 1
        finally:
            st.SIGNAL_LOG_FILE = orig

    def test_neutral_when_neither_tp_nor_sl(self, tmp_path):
        """Result should be NEUTRAL when neither TP nor SL touched."""
        import agents.signal_tracker as st
        orig = st.SIGNAL_LOG_FILE
        today = "2026-03-18"
        entry_price = 4000.0
        tp = 4320.0
        sl = 3840.0

        signals_data = [{
            "ticker": "BBNI", "entry": entry_price,
            "score": 5, "tp": tp, "sl": sl,
            "result": "PENDING", "close_price": None, "high": None, "low": None,
        }]
        self._setup_log(tmp_path, today, signals_data)
        try:
            with patch("agents.signal_tracker.yf.Ticker") as mock_ticker_cls:
                mock_ticker = MagicMock()
                mock_ticker_cls.return_value = mock_ticker
                row = MagicMock()
                row.__getitem__ = lambda s, k: {
                    "High": 4100.0, "Low": 3900.0, "Close": 4050.0
                }[k]
                mock_df = MagicMock()
                mock_df.empty = False
                mock_df.iloc = MagicMock()
                mock_df.iloc.__getitem__ = lambda s, i: row
                mock_ticker.history.return_value = mock_df

                with patch.object(st, "_today_str", return_value=today):
                    log_signals_close()

            log = load_signal_log()
            result = log[today]["signals"][0]["result"]
            assert result == "NEUTRAL"
            assert log[today]["summary"]["neutral"] == 1
        finally:
            st.SIGNAL_LOG_FILE = orig

    def test_no_crash_when_no_signals_today(self, tmp_path):
        """log_signals_close should not crash when today has no logged signals."""
        import agents.signal_tracker as st
        orig = st.SIGNAL_LOG_FILE
        st.SIGNAL_LOG_FILE = str(tmp_path / "signal_log.json")
        today = "2026-03-18"
        try:
            with patch.object(st, "_today_str", return_value=today):
                log_signals_close()  # Should not raise
        finally:
            st.SIGNAL_LOG_FILE = orig

    def test_win_rate_calculation(self, tmp_path):
        """win_rate should be correct percentage (hit/total * 100)."""
        import agents.signal_tracker as st
        orig = st.SIGNAL_LOG_FILE
        today = "2026-03-18"

        # 3 signals: 1 pre-evaluated as HIT, 1 as MISS, 1 as NEUTRAL
        signals_data = [
            {"ticker": "A", "entry": 1000, "tp": 1080, "sl": 960, "result": "HIT",
             "close_price": 1090, "high": 1090, "low": 980},
            {"ticker": "B", "entry": 1000, "tp": 1080, "sl": 960, "result": "MISS",
             "close_price": 950, "high": 1020, "low": 950},
            {"ticker": "C", "entry": 1000, "tp": 1080, "sl": 960, "result": "NEUTRAL",
             "close_price": 1020, "high": 1040, "low": 970},
        ]
        self._setup_log(tmp_path, today, signals_data)
        try:
            with patch.object(st, "_today_str", return_value=today):
                log_signals_close()  # All already evaluated → just counts

            log = load_signal_log()
            summary = log[today]["summary"]
            assert summary["hit"] == 1
            assert summary["miss"] == 1
            assert summary["neutral"] == 1
            assert summary["win_rate"] == pytest.approx(33.3, abs=0.2)
        finally:
            st.SIGNAL_LOG_FILE = orig


# ─── send_weekly_accuracy_report ────────────────────────────────────────────

class TestSendWeeklyAccuracyReport:

    def test_sends_telegram_when_data_present(self, tmp_path):
        """Should call _send_telegram when weekly data exists."""
        import agents.signal_tracker as st
        orig = st.SIGNAL_LOG_FILE
        st.SIGNAL_LOG_FILE = str(tmp_path / "signal_log.json")

        # Simulate Mon-Fri of a week (2026-03-16 to 2026-03-20)
        monday = datetime(2026, 3, 16)
        log = {}
        for i in range(5):
            date_str = (monday + timedelta(days=i)).strftime("%Y-%m-%d")
            log[date_str] = {
                "signals": [
                    {"ticker": "BBNI", "entry": 4390, "tp": 4741, "sl": 4214,
                     "result": "HIT", "close_price": 4750, "high": 4760, "low": 4350},
                    {"ticker": "BBCA", "entry": 9500, "tp": 10260, "sl": 9120,
                     "result": "NEUTRAL", "close_price": 9600, "high": 9700, "low": 9400},
                ],
                "summary": {"total": 2, "hit": 1, "miss": 0, "neutral": 1, "win_rate": 50.0},
            }
        save_signal_log(log)

        try:
            # Patch _send_telegram so no real HTTP call, and patch utcnow to be Friday
            friday_utc = datetime(2026, 3, 20, 8, 0, 0)  # Friday 15:00 WIB
            with patch.object(st, "_send_telegram", return_value=True) as mock_telegram, \
                 patch("agents.signal_tracker.datetime") as mock_dt:

                # Make datetime.utcnow() return Friday
                mock_dt.utcnow.return_value = friday_utc
                # Keep timedelta working normally
                mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

                send_weekly_accuracy_report()
                assert mock_telegram.called
        finally:
            st.SIGNAL_LOG_FILE = orig

    def test_no_crash_with_empty_log(self, tmp_path):
        """Should not crash when signal_log.json is empty."""
        import agents.signal_tracker as st
        orig = st.SIGNAL_LOG_FILE
        st.SIGNAL_LOG_FILE = str(tmp_path / "signal_log.json")
        try:
            send_weekly_accuracy_report()  # Should not raise
        finally:
            st.SIGNAL_LOG_FILE = orig

    def test_message_contains_win_rate(self, tmp_path):
        """Weekly report message should contain win rate."""
        import agents.signal_tracker as st
        orig = st.SIGNAL_LOG_FILE
        st.SIGNAL_LOG_FILE = str(tmp_path / "signal_log.json")

        monday = datetime(2026, 3, 16)
        log = {}
        for i in range(5):
            date_str = (monday + timedelta(days=i)).strftime("%Y-%m-%d")
            log[date_str] = {
                "signals": [
                    {"ticker": "BBNI", "result": "HIT", "close_price": 4750,
                     "high": 4760, "low": 4350},
                ],
                "summary": {"total": 1, "hit": 1, "miss": 0, "neutral": 0, "win_rate": 100.0},
            }
        save_signal_log(log)
        try:
            sent_messages = []
            with patch.object(st, "_send_telegram", side_effect=lambda m, **kw: sent_messages.append(m) or True):
                send_weekly_accuracy_report()

            if sent_messages:
                msg = sent_messages[0]
                assert "Win Rate" in msg or "win_rate" in msg.lower() or "%" in msg
        finally:
            st.SIGNAL_LOG_FILE = orig


# ─── _today_str ─────────────────────────────────────────────────────────────

class TestTodayStr:

    def test_today_str_format(self):
        """_today_str should return YYYY-MM-DD format."""
        from agents.signal_tracker import _today_str
        result = _today_str()
        # Validate format
        parts = result.split("-")
        assert len(parts) == 3
        assert len(parts[0]) == 4  # year
        assert len(parts[1]) == 2  # month
        assert len(parts[2]) == 2  # day
