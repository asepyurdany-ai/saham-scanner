"""
Unit tests for agents/trade_journal.py
Tests: save_trade, get_journal_stats, format_journal_report
"""

import json
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.trade_journal import (
    save_trade,
    get_journal_stats,
    format_journal_report,
    load_journal,
    save_journal,
    _generate_trade_id,
    _format_hold_time,
)


# --- Helpers ---

def make_trade(ticker="BBCA", entry=6070, exit_p=7010, lots=4,
               followed_signal=True, signal_score=5, hold_minutes=215,
               profit_pct=None, profit_rp=None):
    shares = lots * 100
    if profit_rp is None:
        profit_rp = (exit_p - entry) * shares
    if profit_pct is None:
        profit_pct = (exit_p - entry) / entry * 100

    return {
        "ticker": ticker,
        "entry_price": entry,
        "exit_price": exit_p,
        "lots": lots,
        "shares": shares,
        "modal_idr": entry * shares,
        "profit_rp": profit_rp,
        "profit_pct": round(profit_pct, 2),
        "entry_time": "2026-03-18T09:47:00",
        "exit_time": "2026-03-18T13:22:00",
        "hold_minutes": hold_minutes,
        "followed_signal": followed_signal,
        "signal_score": signal_score,
        "result": "WIN" if profit_rp > 0 else ("LOSS" if profit_rp < 0 else "NEUTRAL"),
    }


@pytest.fixture
def journal_file(tmp_path):
    """Temporary journal file for each test."""
    return str(tmp_path / "trade_journal_test.json")


# --- _format_hold_time Tests ---

class TestFormatHoldTime:
    def test_minutes_only(self):
        assert _format_hold_time(45) == "45m"

    def test_hours_only(self):
        assert _format_hold_time(120) == "2j"

    def test_hours_and_minutes(self):
        assert _format_hold_time(215) == "3j 35m"

    def test_zero_minutes(self):
        assert _format_hold_time(0) == "0m"

    def test_one_hour_exact(self):
        assert _format_hold_time(60) == "1j"


# --- _generate_trade_id Tests ---

class TestGenerateTradeId:
    def test_basic_id(self):
        trades = []
        id_ = _generate_trade_id("BBCA", "20260318", trades)
        assert id_ == "BBCA-20260318-001"

    def test_sequential_ids(self):
        trades = [{"id": "BBCA-20260318-001"}]
        id_ = _generate_trade_id("BBCA", "20260318", trades)
        assert id_ == "BBCA-20260318-002"

    def test_different_ticker_starts_at_001(self):
        trades = [{"id": "BBCA-20260318-001"}]
        id_ = _generate_trade_id("BMRI", "20260318", trades)
        assert id_ == "BMRI-20260318-001"


# --- save_trade Tests ---

class TestSaveTrade:

    def test_save_basic_trade(self, journal_file):
        trade = make_trade()
        saved = save_trade(trade, journal_file=journal_file)

        assert saved is not None
        assert "id" in saved
        assert saved["id"].startswith("BBCA-")
        assert saved["ticker"] == "BBCA"

    def test_trade_persisted_to_file(self, journal_file):
        trade = make_trade("BMRI", 4700, 5100, 5)
        save_trade(trade, journal_file=journal_file)

        loaded = load_journal(journal_file)
        assert len(loaded) == 1
        assert loaded[0]["ticker"] == "BMRI"

    def test_multiple_trades_accumulated(self, journal_file):
        save_trade(make_trade("BBCA", 6070, 7010, 4), journal_file=journal_file)
        save_trade(make_trade("BMRI", 4700, 4500, 5), journal_file=journal_file)
        save_trade(make_trade("TLKM", 3000, 3100, 10), journal_file=journal_file)

        loaded = load_journal(journal_file)
        assert len(loaded) == 3

    def test_result_auto_set_win(self, journal_file):
        trade = make_trade()
        trade.pop("result")
        saved = save_trade(trade, journal_file=journal_file)
        assert saved["result"] == "WIN"

    def test_result_auto_set_loss(self, journal_file):
        trade = make_trade("BMRI", 4700, 4500, 5, profit_pct=-4.26, profit_rp=-10000)
        trade.pop("result")
        saved = save_trade(trade, journal_file=journal_file)
        assert saved["result"] == "LOSS"

    def test_existing_id_not_overwritten(self, journal_file):
        trade = make_trade()
        trade["id"] = "BBCA-20260318-CUSTOM"
        saved = save_trade(trade, journal_file=journal_file)
        assert saved["id"] == "BBCA-20260318-CUSTOM"

    def test_sequential_ids_same_ticker_same_day(self, journal_file):
        t1 = save_trade(make_trade("BBCA"), journal_file=journal_file)
        t2 = save_trade(make_trade("BBCA"), journal_file=journal_file)
        assert t1["id"] == "BBCA-20260318-001"
        assert t2["id"] == "BBCA-20260318-002"


# --- get_journal_stats Tests ---

class TestGetJournalStats:

    def test_empty_journal_returns_zeros(self, journal_file):
        stats = get_journal_stats(journal_file)
        assert stats["total_trades"] == 0
        assert stats["wins"] == 0
        assert stats["win_rate_pct"] == 0.0

    def test_counts_wins_losses(self, journal_file):
        save_trade(make_trade("BBCA", 6070, 7010, 4, profit_pct=15.5, profit_rp=376000), journal_file=journal_file)
        save_trade(make_trade("BMRI", 4700, 4500, 5, followed_signal=False, profit_pct=-4.26, profit_rp=-10000), journal_file=journal_file)
        save_trade(make_trade("TLKM", 3000, 3100, 10, profit_pct=3.33, profit_rp=100000), journal_file=journal_file)

        stats = get_journal_stats(journal_file)
        assert stats["total_trades"] == 3
        assert stats["wins"] == 2
        assert stats["losses"] == 1

    def test_win_rate_calculation(self, journal_file):
        save_trade(make_trade("BBCA", 6070, 7010, 4), journal_file=journal_file)  # WIN
        save_trade(make_trade("BMRI", 4700, 4500, 5, followed_signal=False, profit_pct=-4.26, profit_rp=-10000), journal_file=journal_file)  # LOSS
        save_trade(make_trade("TLKM", 3000, 3100, 10), journal_file=journal_file)  # WIN

        stats = get_journal_stats(journal_file)
        assert stats["win_rate_pct"] == pytest.approx(66.7, abs=0.1)

    def test_total_profit_sum(self, journal_file):
        save_trade(make_trade("BBCA", profit_rp=376000), journal_file=journal_file)
        save_trade(make_trade("BMRI", followed_signal=False, profit_pct=-4.26, profit_rp=-100000), journal_file=journal_file)

        stats = get_journal_stats(journal_file)
        assert stats["total_profit_rp"] == 276000

    def test_best_trade(self, journal_file):
        save_trade(make_trade("BBCA", profit_pct=15.5, profit_rp=376000), journal_file=journal_file)
        save_trade(make_trade("BMRI", profit_pct=3.0, profit_rp=50000), journal_file=journal_file)

        stats = get_journal_stats(journal_file)
        assert stats["best_trade"]["ticker"] == "BBCA"
        assert stats["best_trade"]["profit_pct"] == pytest.approx(15.5, abs=0.1)

    def test_worst_trade(self, journal_file):
        save_trade(make_trade("BBCA", profit_pct=15.5, profit_rp=376000), journal_file=journal_file)
        save_trade(make_trade("BMRI", followed_signal=False, profit_pct=-4.26, profit_rp=-50000), journal_file=journal_file)

        stats = get_journal_stats(journal_file)
        assert stats["worst_trade"]["ticker"] == "BMRI"

    def test_signal_win_rate_vs_non_signal(self, journal_file):
        # Signal trades: 2 wins, 1 loss → 66.7%
        save_trade(make_trade("BBCA", followed_signal=True, profit_pct=15.5, profit_rp=376000), journal_file=journal_file)
        save_trade(make_trade("EXCL", followed_signal=True, profit_pct=5.0, profit_rp=100000), journal_file=journal_file)
        save_trade(make_trade("ADRO", followed_signal=True, profit_pct=-3.0, profit_rp=-50000), journal_file=journal_file)
        # Non-signal trades: 1 loss
        save_trade(make_trade("BMRI", followed_signal=False, profit_pct=-4.0, profit_rp=-80000), journal_file=journal_file)

        stats = get_journal_stats(journal_file)
        assert stats["signal_win_rate"] == pytest.approx(66.7, abs=0.1)
        assert stats["non_signal_win_rate"] == 0.0

    def test_avg_hold_minutes(self, journal_file):
        t1 = make_trade(hold_minutes=120)
        t2 = make_trade(hold_minutes=240)
        save_trade(t1, journal_file=journal_file)
        save_trade(t2, journal_file=journal_file)

        stats = get_journal_stats(journal_file)
        assert stats["avg_hold_minutes"] == 180


# --- format_journal_report Tests ---

class TestFormatJournalReport:

    def test_empty_journal_message(self, journal_file):
        msg = format_journal_report(journal_file)
        assert "TRADING JOURNAL" in msg
        assert "Belum ada" in msg or "kosong" in msg.lower() or "trade" in msg.lower()

    def test_report_contains_win_rate(self, journal_file):
        save_trade(make_trade("BBCA", profit_pct=15.5, profit_rp=376000), journal_file=journal_file)
        save_trade(make_trade("BMRI", followed_signal=False, profit_pct=-3.0, profit_rp=-50000), journal_file=journal_file)
        msg = format_journal_report(journal_file)
        assert "Win rate" in msg or "win rate" in msg

    def test_report_contains_ticker_names(self, journal_file):
        save_trade(make_trade("BBCA", profit_pct=15.5, profit_rp=376000), journal_file=journal_file)
        msg = format_journal_report(journal_file)
        assert "BBCA" in msg

    def test_report_contains_profit(self, journal_file):
        save_trade(make_trade("BBCA", profit_pct=15.5, profit_rp=376000), journal_file=journal_file)
        msg = format_journal_report(journal_file)
        assert "Rp" in msg or "profit" in msg.lower()

    def test_report_returns_string(self, journal_file):
        msg = format_journal_report(journal_file)
        assert isinstance(msg, str)
        assert len(msg) > 0

    def test_report_contains_signal_stats(self, journal_file):
        save_trade(make_trade("BBCA", followed_signal=True, profit_pct=15.5, profit_rp=376000), journal_file=journal_file)
        save_trade(make_trade("BMRI", followed_signal=False, profit_pct=-3.0, profit_rp=-50000), journal_file=journal_file)
        msg = format_journal_report(journal_file)
        # Should mention signal win rates
        assert "sinyal" in msg.lower() or "Dexter" in msg

    def test_report_contains_hold_time(self, journal_file):
        save_trade(make_trade(hold_minutes=167), journal_file=journal_file)
        msg = format_journal_report(journal_file)
        assert "hold" in msg.lower() or "Avg" in msg
