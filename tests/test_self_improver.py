"""
Unit tests for agents/self_improver.py
Tests: log_signal_result, analyze_performance, generate_improvement_report, log_error
"""

import json
import os
import sys
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.self_improver import (
    log_signal_result,
    analyze_performance,
    generate_improvement_report,
    log_error,
    PERFORMANCE_LOG_FILE,
    ERROR_LOG_FILE,
)


# ─── Helpers ────────────────────────────────────────────────────────────────

def _patch_log_file(tmp_path, module, attr, filename):
    """Helper: patch a log file path to use tmp_path."""
    orig = getattr(module, attr)
    new_path = str(tmp_path / filename)
    setattr(module, attr, new_path)
    return orig, new_path


# ─── Tests: log_signal_result ───────────────────────────────────────────────

class TestLogSignalResult:

    def test_creates_log_file(self, tmp_path):
        """log_signal_result should create performance_log.json."""
        import agents.self_improver as si
        orig_path = si.PERFORMANCE_LOG_FILE
        si.PERFORMANCE_LOG_FILE = str(tmp_path / "performance_log.json")
        try:
            log_signal_result("2026-03-18", "BBNI", 4390.0, 4500.0, "HIT_TP")
            assert os.path.exists(si.PERFORMANCE_LOG_FILE)
        finally:
            si.PERFORMANCE_LOG_FILE = orig_path

    def test_appends_entry(self, tmp_path):
        """log_signal_result should append to existing log."""
        import agents.self_improver as si
        orig_path = si.PERFORMANCE_LOG_FILE
        si.PERFORMANCE_LOG_FILE = str(tmp_path / "performance_log.json")
        try:
            log_signal_result("2026-03-18", "BBNI", 4390.0, 4500.0, "HIT_TP")
            log_signal_result("2026-03-18", "BBCA", 9500.0, 9200.0, "HIT_CL")

            with open(si.PERFORMANCE_LOG_FILE) as f:
                log = json.load(f)

            assert len(log) == 2
            assert log[0]["ticker"] == "BBNI"
            assert log[1]["ticker"] == "BBCA"
        finally:
            si.PERFORMANCE_LOG_FILE = orig_path

    def test_entry_structure(self, tmp_path):
        """log_signal_result entry should have all required fields."""
        import agents.self_improver as si
        orig_path = si.PERFORMANCE_LOG_FILE
        si.PERFORMANCE_LOG_FILE = str(tmp_path / "performance_log.json")
        try:
            log_signal_result("2026-03-18", "BBNI", 4390.0, 4500.0, "HIT_TP")
            with open(si.PERFORMANCE_LOG_FILE) as f:
                log = json.load(f)

            entry = log[0]
            for key in ["date", "ticker", "entry", "close", "result", "timestamp"]:
                assert key in entry, f"Missing key: {key}"
        finally:
            si.PERFORMANCE_LOG_FILE = orig_path

    def test_valid_results(self, tmp_path):
        """Should accept all valid result values."""
        import agents.self_improver as si
        orig_path = si.PERFORMANCE_LOG_FILE
        si.PERFORMANCE_LOG_FILE = str(tmp_path / "performance_log.json")
        try:
            for result in ["HIT_TP", "HIT_CL", "NEUTRAL"]:
                log_signal_result("2026-03-18", "BBNI", 4000.0, 4100.0, result)
        finally:
            si.PERFORMANCE_LOG_FILE = orig_path

    def test_invalid_result_raises(self, tmp_path):
        """Should raise ValueError for invalid result."""
        import agents.self_improver as si
        orig_path = si.PERFORMANCE_LOG_FILE
        si.PERFORMANCE_LOG_FILE = str(tmp_path / "performance_log.json")
        try:
            with pytest.raises(ValueError):
                log_signal_result("2026-03-18", "BBNI", 4000.0, 4100.0, "INVALID")
        finally:
            si.PERFORMANCE_LOG_FILE = orig_path

    def test_entry_values_correct(self, tmp_path):
        """Entry values should match inputs."""
        import agents.self_improver as si
        orig_path = si.PERFORMANCE_LOG_FILE
        si.PERFORMANCE_LOG_FILE = str(tmp_path / "performance_log.json")
        try:
            log_signal_result("2026-03-18", "BMRI", 4750.0, 4850.0, "HIT_TP")
            with open(si.PERFORMANCE_LOG_FILE) as f:
                log = json.load(f)
            assert log[0]["date"] == "2026-03-18"
            assert log[0]["ticker"] == "BMRI"
            assert log[0]["entry"] == 4750.0
            assert log[0]["close"] == 4850.0
            assert log[0]["result"] == "HIT_TP"
        finally:
            si.PERFORMANCE_LOG_FILE = orig_path

    def test_timestamp_is_iso_format(self, tmp_path):
        """Timestamp should be ISO format string."""
        import agents.self_improver as si
        orig_path = si.PERFORMANCE_LOG_FILE
        si.PERFORMANCE_LOG_FILE = str(tmp_path / "performance_log.json")
        try:
            log_signal_result("2026-03-18", "BBNI", 4000.0, 4100.0, "NEUTRAL")
            with open(si.PERFORMANCE_LOG_FILE) as f:
                log = json.load(f)
            ts = log[0]["timestamp"]
            # Should parse without error
            datetime.fromisoformat(ts)
        finally:
            si.PERFORMANCE_LOG_FILE = orig_path


# ─── Tests: analyze_performance ─────────────────────────────────────────────

class TestAnalyzePerformance:

    def test_error_when_no_file(self, tmp_path):
        """Should return error status when no log file exists."""
        import agents.self_improver as si
        orig_path = si.PERFORMANCE_LOG_FILE
        si.PERFORMANCE_LOG_FILE = str(tmp_path / "nonexistent.json")
        try:
            result = analyze_performance()
            assert result["status"] == "error"
        finally:
            si.PERFORMANCE_LOG_FILE = orig_path

    def test_insufficient_data_when_few_days(self, tmp_path):
        """Should return insufficient_data when < 10 trading days."""
        import agents.self_improver as si
        orig_path = si.PERFORMANCE_LOG_FILE
        si.PERFORMANCE_LOG_FILE = str(tmp_path / "performance_log.json")
        try:
            # Only 3 days of data
            log = []
            for i in range(3):
                log.append({
                    "date": f"2026-03-{i+1:02d}",
                    "ticker": "BBNI",
                    "entry": 4000.0,
                    "close": 4100.0,
                    "result": "HIT_TP",
                    "timestamp": datetime.utcnow().isoformat(),
                })
            with open(si.PERFORMANCE_LOG_FILE, "w") as f:
                json.dump(log, f)

            result = analyze_performance()
            assert result["status"] == "insufficient_data"
            assert result["days"] == 3
            assert result["needed"] == 10
        finally:
            si.PERFORMANCE_LOG_FILE = orig_path

    def test_ok_status_with_10_days(self, tmp_path):
        """Should return ok status with 10+ days of data."""
        import agents.self_improver as si
        orig_path = si.PERFORMANCE_LOG_FILE
        si.PERFORMANCE_LOG_FILE = str(tmp_path / "performance_log.json")
        try:
            log = []
            for i in range(10):
                log.append({
                    "date": f"2026-03-{i+1:02d}",
                    "ticker": "BBNI",
                    "entry": 4000.0,
                    "close": 4100.0,
                    "result": "HIT_TP",
                    "timestamp": datetime.utcnow().isoformat(),
                })
            with open(si.PERFORMANCE_LOG_FILE, "w") as f:
                json.dump(log, f)

            result = analyze_performance()
            assert result["status"] == "ok"
            assert result["total_signals"] == 10
            assert result["overall_win_rate"] == 100.0
        finally:
            si.PERFORMANCE_LOG_FILE = orig_path

    def test_win_rate_calculation(self, tmp_path):
        """Win rate should be HIT_TP / total * 100."""
        import agents.self_improver as si
        orig_path = si.PERFORMANCE_LOG_FILE
        si.PERFORMANCE_LOG_FILE = str(tmp_path / "performance_log.json")
        try:
            log = []
            results = ["HIT_TP"] * 7 + ["HIT_CL"] * 2 + ["NEUTRAL"] * 1
            for i, r in enumerate(results):
                log.append({
                    "date": f"2026-03-{i+1:02d}",
                    "ticker": "BBNI",
                    "entry": 4000.0,
                    "close": 4100.0,
                    "result": r,
                    "timestamp": datetime.utcnow().isoformat(),
                })
            with open(si.PERFORMANCE_LOG_FILE, "w") as f:
                json.dump(log, f)

            result = analyze_performance()
            assert result["status"] == "ok"
            assert result["hits"] == 7
            assert result["losses"] == 2
            assert result["neutrals"] == 1
            assert result["overall_win_rate"] == pytest.approx(70.0, abs=0.1)
        finally:
            si.PERFORMANCE_LOG_FILE = orig_path

    def test_returns_dict(self):
        """analyze_performance should always return a dict."""
        result = analyze_performance()
        assert isinstance(result, dict)

    def test_insights_structure_when_ok(self, tmp_path):
        """OK result should have required keys."""
        import agents.self_improver as si
        orig_path = si.PERFORMANCE_LOG_FILE
        si.PERFORMANCE_LOG_FILE = str(tmp_path / "performance_log.json")
        try:
            log = []
            for i in range(10):
                log.append({
                    "date": f"2026-03-{i+1:02d}",
                    "ticker": "BBNI",
                    "entry": 4000.0,
                    "close": 4100.0,
                    "result": "HIT_TP",
                    "timestamp": datetime.utcnow().isoformat(),
                })
            with open(si.PERFORMANCE_LOG_FILE, "w") as f:
                json.dump(log, f)

            result = analyze_performance()
            for key in ["status", "total_signals", "trading_days", "hits", "losses",
                        "neutrals", "overall_win_rate"]:
                assert key in result, f"Missing key: {key}"
        finally:
            si.PERFORMANCE_LOG_FILE = orig_path


# ─── Tests: generate_improvement_report ─────────────────────────────────────

class TestGenerateImprovementReport:

    def _populate_log(self, path, n=10, result_pattern=None):
        """Populate a performance log with n entries."""
        if result_pattern is None:
            result_pattern = ["HIT_TP"] * 6 + ["HIT_CL"] * 2 + ["NEUTRAL"] * 2
        log = []
        now = datetime.utcnow()
        for i in range(n):
            r = result_pattern[i % len(result_pattern)]
            log.append({
                "date": f"2026-03-{i+1:02d}",
                "ticker": "BBNI",
                "entry": 4000.0,
                "close": 4100.0,
                "result": r,
                "timestamp": (now - timedelta(days=i)).isoformat(),
            })
        with open(path, "w") as f:
            json.dump(log, f)

    def test_returns_empty_when_no_log(self, tmp_path):
        """Should return empty string when no log exists."""
        import agents.self_improver as si
        orig_path = si.PERFORMANCE_LOG_FILE
        si.PERFORMANCE_LOG_FILE = str(tmp_path / "nonexistent.json")
        try:
            result = generate_improvement_report()
            assert result == ""
        finally:
            si.PERFORMANCE_LOG_FILE = orig_path

    def test_returns_string_with_data(self, tmp_path):
        """Should return non-empty string when data exists."""
        import agents.self_improver as si
        orig_path = si.PERFORMANCE_LOG_FILE
        si.PERFORMANCE_LOG_FILE = str(tmp_path / "performance_log.json")
        try:
            self._populate_log(si.PERFORMANCE_LOG_FILE)
            with patch.object(si, "_send_telegram", return_value=True):
                result = generate_improvement_report()
            assert isinstance(result, str)
            assert len(result) > 0
        finally:
            si.PERFORMANCE_LOG_FILE = orig_path

    def test_calls_send_telegram(self, tmp_path):
        """Should call _send_telegram with the report."""
        import agents.self_improver as si
        orig_path = si.PERFORMANCE_LOG_FILE
        si.PERFORMANCE_LOG_FILE = str(tmp_path / "performance_log.json")
        try:
            self._populate_log(si.PERFORMANCE_LOG_FILE)
            with patch.object(si, "_send_telegram", return_value=True) as mock_tg:
                generate_improvement_report()
            assert mock_tg.called
        finally:
            si.PERFORMANCE_LOG_FILE = orig_path

    def test_message_contains_required_sections(self, tmp_path):
        """Message should have performance stats and insights."""
        import agents.self_improver as si
        orig_path = si.PERFORMANCE_LOG_FILE
        si.PERFORMANCE_LOG_FILE = str(tmp_path / "performance_log.json")
        try:
            self._populate_log(si.PERFORMANCE_LOG_FILE)
            sent_messages = []
            with patch.object(si, "_send_telegram",
                              side_effect=lambda m, **kw: sent_messages.append(m) or True):
                generate_improvement_report()

            if sent_messages:
                msg = sent_messages[0]
                assert "Hit TP" in msg or "HIT_TP" in msg or "%" in msg
                assert "WEEKLY PERFORMANCE" in msg or "Sinyal" in msg
        finally:
            si.PERFORMANCE_LOG_FILE = orig_path

    def test_handles_empty_log(self, tmp_path):
        """Should not crash with empty log file."""
        import agents.self_improver as si
        orig_path = si.PERFORMANCE_LOG_FILE
        si.PERFORMANCE_LOG_FILE = str(tmp_path / "performance_log.json")
        try:
            with open(si.PERFORMANCE_LOG_FILE, "w") as f:
                json.dump([], f)
            result = generate_improvement_report()
            assert result == ""
        finally:
            si.PERFORMANCE_LOG_FILE = orig_path


# ─── Tests: log_error ───────────────────────────────────────────────────────

class TestLogError:

    def test_creates_error_log(self, tmp_path):
        """log_error should create error_log.json."""
        import agents.self_improver as si
        orig_path = si.ERROR_LOG_FILE
        si.ERROR_LOG_FILE = str(tmp_path / "error_log.json")
        try:
            log_error("Scanner", "Connection timeout", "get_stock_data")
            assert os.path.exists(si.ERROR_LOG_FILE)
        finally:
            si.ERROR_LOG_FILE = orig_path

    def test_appends_entry(self, tmp_path):
        """log_error should append entries."""
        import agents.self_improver as si
        orig_path = si.ERROR_LOG_FILE
        si.ERROR_LOG_FILE = str(tmp_path / "error_log.json")
        try:
            log_error("Scanner", "Error A", "ctx1")
            log_error("Radar", "Error B", "ctx2")

            with open(si.ERROR_LOG_FILE) as f:
                log = json.load(f)

            assert len(log) == 2
            assert log[0]["agent"] == "Scanner"
            assert log[1]["agent"] == "Radar"
        finally:
            si.ERROR_LOG_FILE = orig_path

    def test_entry_structure(self, tmp_path):
        """Error entry should have required fields."""
        import agents.self_improver as si
        orig_path = si.ERROR_LOG_FILE
        si.ERROR_LOG_FILE = str(tmp_path / "error_log.json")
        try:
            log_error("Scanner", "Test error", "test context")
            with open(si.ERROR_LOG_FILE) as f:
                log = json.load(f)

            entry = log[0]
            for key in ["timestamp", "agent", "error", "context"]:
                assert key in entry, f"Missing key: {key}"
        finally:
            si.ERROR_LOG_FILE = orig_path

    def test_values_stored_correctly(self, tmp_path):
        """Entry values should match inputs."""
        import agents.self_improver as si
        orig_path = si.ERROR_LOG_FILE
        si.ERROR_LOG_FILE = str(tmp_path / "error_log.json")
        try:
            log_error("Radar", "API timeout", "check_macro_shock")
            with open(si.ERROR_LOG_FILE) as f:
                log = json.load(f)

            assert log[0]["agent"] == "Radar"
            assert log[0]["error"] == "API timeout"
            assert log[0]["context"] == "check_macro_shock"
        finally:
            si.ERROR_LOG_FILE = orig_path

    def test_context_optional(self, tmp_path):
        """log_error should work without context."""
        import agents.self_improver as si
        orig_path = si.ERROR_LOG_FILE
        si.ERROR_LOG_FILE = str(tmp_path / "error_log.json")
        try:
            log_error("Scanner", "Some error")
            with open(si.ERROR_LOG_FILE) as f:
                log = json.load(f)
            assert len(log) == 1
        finally:
            si.ERROR_LOG_FILE = orig_path

    def test_telegram_alert_on_3_repeat_errors(self, tmp_path):
        """Should send Telegram alert when same error occurs 3+ times."""
        import agents.self_improver as si
        orig_path = si.ERROR_LOG_FILE
        si.ERROR_LOG_FILE = str(tmp_path / "error_log.json")
        try:
            with patch.object(si, "_send_telegram", return_value=True) as mock_tg:
                log_error("Scanner", "Repeat error", "ctx")
                log_error("Scanner", "Repeat error", "ctx")
                log_error("Scanner", "Repeat error", "ctx")  # 3rd occurrence

            # Should have sent a Telegram alert
            assert mock_tg.called
        finally:
            si.ERROR_LOG_FILE = orig_path

    def test_no_telegram_on_first_two_errors(self, tmp_path):
        """Should NOT send Telegram on the first 2 occurrences."""
        import agents.self_improver as si
        orig_path = si.ERROR_LOG_FILE
        si.ERROR_LOG_FILE = str(tmp_path / "error_log.json")
        try:
            with patch.object(si, "_send_telegram", return_value=True) as mock_tg:
                log_error("Scanner", "Unique error 1", "ctx")
                log_error("Scanner", "Unique error 2", "ctx")

            assert not mock_tg.called
        finally:
            si.ERROR_LOG_FILE = orig_path

    def test_different_agents_not_aggregated(self, tmp_path):
        """Same error from different agents should not trigger alert."""
        import agents.self_improver as si
        orig_path = si.ERROR_LOG_FILE
        si.ERROR_LOG_FILE = str(tmp_path / "error_log.json")
        try:
            with patch.object(si, "_send_telegram", return_value=True) as mock_tg:
                log_error("Scanner", "Timeout", "ctx")
                log_error("Radar", "Timeout", "ctx")
                log_error("Sentinel", "Timeout", "ctx")

            # Each is a different agent — should not trigger alert
            assert not mock_tg.called
        finally:
            si.ERROR_LOG_FILE = orig_path

    def test_timestamp_iso_format(self, tmp_path):
        """Timestamp should be ISO-parseable."""
        import agents.self_improver as si
        orig_path = si.ERROR_LOG_FILE
        si.ERROR_LOG_FILE = str(tmp_path / "error_log.json")
        try:
            log_error("Agent", "err", "ctx")
            with open(si.ERROR_LOG_FILE) as f:
                log = json.load(f)
            datetime.fromisoformat(log[0]["timestamp"])  # Should not raise
        finally:
            si.ERROR_LOG_FILE = orig_path

    def test_existing_log_preserved(self, tmp_path):
        """Existing log entries should be preserved when appending."""
        import agents.self_improver as si
        orig_path = si.ERROR_LOG_FILE
        si.ERROR_LOG_FILE = str(tmp_path / "error_log.json")
        try:
            # Pre-populate with 2 entries
            existing = [
                {"timestamp": "2026-01-01T00:00:00", "agent": "Old", "error": "old error", "context": ""},
                {"timestamp": "2026-01-02T00:00:00", "agent": "Old2", "error": "old error2", "context": ""},
            ]
            with open(si.ERROR_LOG_FILE, "w") as f:
                json.dump(existing, f)

            log_error("New", "new error", "")
            with open(si.ERROR_LOG_FILE) as f:
                log = json.load(f)

            assert len(log) == 3
            assert log[0]["agent"] == "Old"
            assert log[2]["agent"] == "New"
        finally:
            si.ERROR_LOG_FILE = orig_path
