"""
Unit tests for agents/market_context.py
Full coverage for shared intelligence system.
"""

import copy
import json
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agents.market_context as mc
from agents.market_context import (
    DEFAULT_CONTEXT,
    load_context,
    save_context,
    update_macro,
    update_sentiment,
    update_geo,
    update_performance,
    update_foreign_flow,
    update_market_breadth,
    update_premarket,
    compute_market_mode,
    get_dynamic_threshold,
    get_context,
    format_context_summary,
)


# ─── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def patch_context_file(tmp_path, monkeypatch):
    """Redirect CONTEXT_FILE to tmp_path for isolation."""
    tmp_file = str(tmp_path / "market_context.json")
    monkeypatch.setattr(mc, "CONTEXT_FILE", tmp_file)
    yield tmp_file


# ─── test_default_context_structure ─────────────────────────────────────────

class TestDefaultContextStructure:

    def test_default_has_required_top_keys(self):
        ctx = load_context()
        for key in ["updated_at", "market_mode", "macro", "sentiment", "geo", "performance"]:
            assert key in ctx, f"Missing top-level key: {key}"

    def test_default_macro_keys(self):
        ctx = load_context()
        macro = ctx["macro"]
        assert "rupiah_change_pct" in macro
        assert "oil_change_pct" in macro
        assert "gold_change_pct" in macro
        assert "macro_shock_active" in macro
        assert "macro_signal" in macro

    def test_default_sentiment_keys(self):
        ctx = load_context()
        sent = ctx["sentiment"]
        assert "news_sentiment" in sent
        assert "confidence" in sent
        assert "affected_sectors" in sent
        assert "affected_tickers" in sent
        assert "summary" in sent

    def test_default_geo_keys(self):
        ctx = load_context()
        geo = ctx["geo"]
        assert "risk_level" in geo
        assert "active_events" in geo
        assert "geo_signal" in geo

    def test_default_performance_keys(self):
        ctx = load_context()
        perf = ctx["performance"]
        assert "recent_win_rate" in perf
        assert "buy_threshold_override" in perf
        assert "best_rsi_zone" in perf
        assert "macd_weight" in perf
        assert "volume_weight" in perf

    def test_default_market_mode_is_normal(self):
        ctx = load_context()
        assert ctx["market_mode"] == "NORMAL"

    def test_default_macro_signal_neutral(self):
        ctx = load_context()
        assert ctx["macro"]["macro_signal"] == "NEUTRAL"

    def test_default_sentiment_netral(self):
        ctx = load_context()
        assert ctx["sentiment"]["news_sentiment"] == "NETRAL"

    def test_default_geo_risk_low(self):
        ctx = load_context()
        assert ctx["geo"]["risk_level"] == "LOW"

    def test_load_returns_deep_copy(self):
        """Modifying returned context should not affect DEFAULT_CONTEXT."""
        ctx = load_context()
        ctx["market_mode"] = "RISK_OFF"
        ctx2 = load_context()
        assert ctx2["market_mode"] == "NORMAL"


# ─── test_update_macro ───────────────────────────────────────────────────────

class TestUpdateMacro:

    def test_updates_rupiah_pct(self):
        update_macro(-1.5, 0.5, 0.3, False)
        ctx = load_context()
        assert ctx["macro"]["rupiah_change_pct"] == -1.5

    def test_updates_oil_pct(self):
        update_macro(0.0, -2.5, 0.0, False)
        ctx = load_context()
        assert ctx["macro"]["oil_change_pct"] == -2.5

    def test_updates_gold_pct(self):
        update_macro(0.0, 0.0, 1.5, False)
        ctx = load_context()
        assert ctx["macro"]["gold_change_pct"] == 1.5

    def test_sets_shock_active(self):
        update_macro(-2.0, 0.0, 0.0, True)
        ctx = load_context()
        assert ctx["macro"]["macro_shock_active"] is True

    def test_macro_signal_bearish_when_oil_low_and_rupiah_weak(self):
        """oil < -2% AND rupiah < -1% → BEARISH"""
        update_macro(-1.5, -3.0, 0.0, False)
        ctx = load_context()
        assert ctx["macro"]["macro_signal"] == "BEARISH"

    def test_macro_signal_neutral_normal(self):
        """Default conditions → NEUTRAL"""
        update_macro(0.1, 0.5, 0.2, False)
        ctx = load_context()
        assert ctx["macro"]["macro_signal"] == "NEUTRAL"

    def test_macro_signal_neutral_when_only_oil_down(self):
        """oil < -2% but rupiah ok → NEUTRAL"""
        update_macro(0.1, -3.0, 0.0, False)
        ctx = load_context()
        assert ctx["macro"]["macro_signal"] == "NEUTRAL"

    def test_persists_to_file(self, patch_context_file):
        update_macro(-1.0, 2.0, 0.5, False)
        assert os.path.exists(patch_context_file)
        with open(patch_context_file) as f:
            data = json.load(f)
        assert data["macro"]["rupiah_change_pct"] == -1.0

    def test_updates_market_mode(self):
        """Macro shock → market_mode should be RISK_OFF"""
        update_macro(-2.0, -4.0, 0.0, True)
        ctx = load_context()
        assert ctx["market_mode"] == "RISK_OFF"


# ─── test_update_sentiment ──────────────────────────────────────────────────

class TestUpdateSentiment:

    def test_updates_news_sentiment(self):
        update_sentiment("POSITIF", "TINGGI", ["Perbankan"], ["BBCA.JK"], "BI tahan suku bunga")
        ctx = load_context()
        assert ctx["sentiment"]["news_sentiment"] == "POSITIF"

    def test_updates_confidence(self):
        update_sentiment("NEGATIF", "SEDANG", [], [], "")
        ctx = load_context()
        assert ctx["sentiment"]["confidence"] == "SEDANG"

    def test_updates_affected_sectors(self):
        update_sentiment("POSITIF", "TINGGI", ["Perbankan", "Coal"], [], "")
        ctx = load_context()
        assert "Perbankan" in ctx["sentiment"]["affected_sectors"]
        assert "Coal" in ctx["sentiment"]["affected_sectors"]

    def test_updates_affected_tickers(self):
        update_sentiment("POSITIF", "TINGGI", [], ["BBCA.JK", "BBRI.JK"], "")
        ctx = load_context()
        assert "BBCA.JK" in ctx["sentiment"]["affected_tickers"]

    def test_updates_summary(self):
        update_sentiment("NETRAL", "SEDANG", [], [], "Pasar cenderung sideways")
        ctx = load_context()
        assert "sideways" in ctx["sentiment"]["summary"]

    def test_risk_on_when_positif_and_bullish_macro(self):
        """POSITIF sentiment + BULLISH macro → RISK_ON"""
        # First set macro to BULLISH
        ctx = load_context()
        ctx["macro"]["macro_signal"] = "BULLISH"
        save_context(ctx)
        update_sentiment("POSITIF", "TINGGI", [], [], "")
        ctx2 = load_context()
        assert ctx2["market_mode"] == "RISK_ON"

    def test_persists_to_file(self, patch_context_file):
        update_sentiment("NEGATIF", "TINGGI", ["Mining"], ["ANTM.JK"], "Nikel turun")
        assert os.path.exists(patch_context_file)
        with open(patch_context_file) as f:
            data = json.load(f)
        assert data["sentiment"]["news_sentiment"] == "NEGATIF"


# ─── test_update_geo ────────────────────────────────────────────────────────

class TestUpdateGeo:

    def test_updates_risk_level(self):
        update_geo("HIGH", ["Konflik Iran-Israel"], "RISK_OFF")
        ctx = load_context()
        assert ctx["geo"]["risk_level"] == "HIGH"

    def test_updates_active_events(self):
        events = ["Fed rate hike", "Oil embargo"]
        update_geo("MEDIUM", events, "CAUTIOUS")
        ctx = load_context()
        assert "Fed rate hike" in ctx["geo"]["active_events"]

    def test_updates_geo_signal(self):
        update_geo("LOW", [], "NEUTRAL")
        ctx = load_context()
        assert ctx["geo"]["geo_signal"] == "NEUTRAL"

    def test_high_geo_triggers_risk_off(self):
        update_geo("HIGH", ["Major conflict"], "RISK_OFF")
        ctx = load_context()
        assert ctx["market_mode"] == "RISK_OFF"

    def test_medium_geo_normal_sentiment_is_normal(self):
        """MEDIUM geo alone (no NEGATIF sentiment) → NORMAL"""
        update_geo("MEDIUM", ["Minor tension"], "CAUTIOUS")
        ctx = load_context()
        assert ctx["market_mode"] == "NORMAL"

    def test_persists_to_file(self, patch_context_file):
        update_geo("MEDIUM", ["Trade war"], "CAUTIOUS")
        assert os.path.exists(patch_context_file)
        with open(patch_context_file) as f:
            data = json.load(f)
        assert data["geo"]["risk_level"] == "MEDIUM"


# ─── test_update_performance ────────────────────────────────────────────────

class TestUpdatePerformance:

    def test_updates_win_rate(self):
        update_performance(0.65, None, [35, 62])
        ctx = load_context()
        assert ctx["performance"]["recent_win_rate"] == 0.65

    def test_updates_threshold_override(self):
        update_performance(0.40, 6, [35, 62])
        ctx = load_context()
        assert ctx["performance"]["buy_threshold_override"] == 6

    def test_updates_rsi_zone(self):
        update_performance(0.55, None, [30, 60])
        ctx = load_context()
        assert ctx["performance"]["best_rsi_zone"] == [30, 60]

    def test_none_win_rate(self):
        update_performance(None, None)
        ctx = load_context()
        assert ctx["performance"]["recent_win_rate"] is None

    def test_none_threshold_override(self):
        update_performance(0.75, None)
        ctx = load_context()
        assert ctx["performance"]["buy_threshold_override"] is None

    def test_rsi_zone_unchanged_if_none(self):
        """rsi_zone=None should not overwrite existing value."""
        update_performance(0.5, None, [40, 65])
        update_performance(0.6, None, rsi_zone=None)
        ctx = load_context()
        assert ctx["performance"]["best_rsi_zone"] == [40, 65]

    def test_persists_to_file(self, patch_context_file):
        update_performance(0.72, None, [35, 62])
        assert os.path.exists(patch_context_file)
        with open(patch_context_file) as f:
            data = json.load(f)
        assert data["performance"]["recent_win_rate"] == 0.72


# ─── test_compute_market_mode ────────────────────────────────────────────────

class TestComputeMarketMode:

    def _make_ctx(self, macro_shock=False, geo_risk="LOW", sentiment="NETRAL",
                  macro_signal="NEUTRAL"):
        ctx = copy.deepcopy(DEFAULT_CONTEXT)
        ctx["macro"]["macro_shock_active"] = macro_shock
        ctx["macro"]["macro_signal"] = macro_signal
        ctx["geo"]["risk_level"] = geo_risk
        ctx["sentiment"]["news_sentiment"] = sentiment
        return ctx

    def test_risk_off_when_macro_shock(self):
        ctx = self._make_ctx(macro_shock=True)
        assert compute_market_mode(ctx) == "RISK_OFF"

    def test_risk_off_when_geo_high(self):
        ctx = self._make_ctx(geo_risk="HIGH")
        assert compute_market_mode(ctx) == "RISK_OFF"

    def test_risk_off_when_macro_shock_and_geo_high(self):
        ctx = self._make_ctx(macro_shock=True, geo_risk="HIGH")
        assert compute_market_mode(ctx) == "RISK_OFF"

    def test_cautious_when_geo_medium_and_neg_sentiment(self):
        ctx = self._make_ctx(geo_risk="MEDIUM", sentiment="NEGATIF")
        assert compute_market_mode(ctx) == "CAUTIOUS"

    def test_risk_on_when_positif_and_bullish(self):
        ctx = self._make_ctx(sentiment="POSITIF", macro_signal="BULLISH")
        assert compute_market_mode(ctx) == "RISK_ON"

    def test_normal_when_default(self):
        ctx = self._make_ctx()
        assert compute_market_mode(ctx) == "NORMAL"

    def test_normal_when_geo_medium_and_positif_sentiment(self):
        """MEDIUM geo but POSITIF sentiment → still NORMAL (no CAUTIOUS)"""
        ctx = self._make_ctx(geo_risk="MEDIUM", sentiment="POSITIF")
        assert compute_market_mode(ctx) == "NORMAL"

    def test_normal_when_positif_but_not_bullish(self):
        """POSITIF sentiment + NEUTRAL macro → NORMAL (not RISK_ON)"""
        ctx = self._make_ctx(sentiment="POSITIF", macro_signal="NEUTRAL")
        assert compute_market_mode(ctx) == "NORMAL"

    def test_macro_shock_overrides_risk_on(self):
        """RISK_OFF takes priority over RISK_ON signals"""
        ctx = self._make_ctx(macro_shock=True, sentiment="POSITIF", macro_signal="BULLISH")
        assert compute_market_mode(ctx) == "RISK_OFF"

    def test_compute_without_ctx_loads_from_file(self):
        """compute_market_mode() without args loads from file."""
        # File doesn't exist → returns NORMAL
        result = compute_market_mode()
        assert result in ("NORMAL", "RISK_OFF", "CAUTIOUS", "RISK_ON")


# ─── test_get_dynamic_threshold_per_mode ────────────────────────────────────

class TestGetDynamicThreshold:

    def _set_mode(self, mode):
        ctx = load_context()
        ctx["market_mode"] = mode
        ctx["performance"]["buy_threshold_override"] = None
        save_context(ctx)

    def test_risk_off_returns_6(self):
        self._set_mode("RISK_OFF")
        assert get_dynamic_threshold() == 6

    def test_cautious_returns_6(self):
        self._set_mode("CAUTIOUS")
        assert get_dynamic_threshold() == 6

    def test_normal_returns_5(self):
        self._set_mode("NORMAL")
        assert get_dynamic_threshold() == 5

    def test_risk_on_returns_4(self):
        self._set_mode("RISK_ON")
        assert get_dynamic_threshold() == 4

    def test_performance_override_takes_priority(self):
        """buy_threshold_override should override market_mode"""
        ctx = load_context()
        ctx["market_mode"] = "RISK_ON"  # would be 4
        ctx["performance"]["buy_threshold_override"] = 6  # override to 6
        save_context(ctx)
        assert get_dynamic_threshold() == 6

    def test_none_override_uses_market_mode(self):
        ctx = load_context()
        ctx["market_mode"] = "RISK_ON"
        ctx["performance"]["buy_threshold_override"] = None
        save_context(ctx)
        assert get_dynamic_threshold() == 4

    def test_default_context_returns_5(self):
        """No file → DEFAULT_CONTEXT → NORMAL → threshold 5"""
        assert get_dynamic_threshold() == 5


# ─── test_context_persistence ───────────────────────────────────────────────

class TestContextPersistence:

    def test_save_and_load_roundtrip(self):
        ctx = load_context()
        ctx["market_mode"] = "CAUTIOUS"
        ctx["macro"]["rupiah_change_pct"] = -2.5
        ctx["sentiment"]["news_sentiment"] = "NEGATIF"
        save_context(ctx)

        loaded = load_context()
        assert loaded["market_mode"] == "CAUTIOUS"
        assert loaded["macro"]["rupiah_change_pct"] == -2.5
        assert loaded["sentiment"]["news_sentiment"] == "NEGATIF"

    def test_save_adds_updated_at(self):
        ctx = load_context()
        save_context(ctx)
        loaded = load_context()
        assert loaded["updated_at"] is not None
        assert "T" in loaded["updated_at"]  # ISO format

    def test_load_missing_file_returns_default(self, patch_context_file):
        """File doesn't exist → returns deep copy of DEFAULT_CONTEXT."""
        assert not os.path.exists(patch_context_file)
        ctx = load_context()
        assert ctx["market_mode"] == "NORMAL"
        assert ctx["macro"]["macro_shock_active"] is False

    def test_load_corrupt_file_returns_default(self, patch_context_file):
        """Corrupt JSON → returns DEFAULT_CONTEXT, no exception."""
        with open(patch_context_file, "w") as f:
            f.write("NOT VALID JSON {{{")
        ctx = load_context()
        assert ctx["market_mode"] == "NORMAL"

    def test_load_empty_file_returns_default(self, patch_context_file):
        """Empty file → returns DEFAULT_CONTEXT."""
        with open(patch_context_file, "w") as f:
            f.write("")
        ctx = load_context()
        assert ctx["market_mode"] == "NORMAL"

    def test_load_partial_ctx_merges_defaults(self, patch_context_file):
        """Partial context (missing keys) → merged with defaults."""
        partial = {"market_mode": "RISK_ON", "macro": {"rupiah_change_pct": -1.0}}
        with open(patch_context_file, "w") as f:
            json.dump(partial, f)
        ctx = load_context()
        assert ctx["market_mode"] == "RISK_ON"
        assert ctx["macro"]["rupiah_change_pct"] == -1.0
        # Missing keys filled from defaults
        assert "oil_change_pct" in ctx["macro"]
        assert "sentiment" in ctx

    def test_multiple_updates_accumulate(self):
        """Multiple updates should persist correctly."""
        update_macro(0.1, 0.2, 0.3, False)
        update_geo("MEDIUM", ["event1"], "CAUTIOUS")
        update_sentiment("NETRAL", "SEDANG", ["Perbankan"], [], "")

        ctx = load_context()
        assert ctx["macro"]["rupiah_change_pct"] == 0.1
        assert ctx["geo"]["risk_level"] == "MEDIUM"
        assert ctx["sentiment"]["news_sentiment"] == "NETRAL"


# ─── test_format_context_summary ────────────────────────────────────────────

class TestFormatContextSummary:

    def test_returns_string(self):
        result = format_context_summary()
        assert isinstance(result, str)

    def test_contains_market_mode(self):
        # Set conditions that produce CAUTIOUS: geo=MEDIUM + sentiment=NEGATIF
        update_geo("MEDIUM", ["Trade tension"], "CAUTIOUS")
        update_sentiment("NEGATIF", "TINGGI", ["Mining"], [], "")
        result = format_context_summary()
        assert "CAUTIOUS" in result

    def test_contains_rupiah_info(self):
        update_macro(-1.5, 0.0, 0.0, False)
        result = format_context_summary()
        assert "Rupiah" in result
        assert "-1.5" in result or "−1.5" in result or "-1.50" in result

    def test_contains_sentiment(self):
        update_sentiment("POSITIF", "TINGGI", [], [], "")
        result = format_context_summary()
        assert "POSITIF" in result

    def test_contains_geo_risk(self):
        update_geo("HIGH", ["Conflict"], "RISK_OFF")
        result = format_context_summary()
        assert "HIGH" in result

    def test_shows_macro_shock_warning(self):
        update_macro(-2.0, -4.0, 0.0, True)
        result = format_context_summary()
        assert "MACRO SHOCK" in result

    def test_risk_off_has_red_emoji(self):
        update_macro(-2.0, -4.0, 0.0, True)
        result = format_context_summary()
        assert "🔴" in result

    def test_risk_on_has_green_emoji(self):
        ctx = load_context()
        ctx["market_mode"] = "RISK_ON"
        ctx["macro"]["macro_signal"] = "BULLISH"
        ctx["sentiment"]["news_sentiment"] = "POSITIF"
        save_context(ctx)
        result = format_context_summary()
        assert "🟢" in result

    def test_no_crash_on_default_context(self):
        """format_context_summary should never raise."""
        try:
            result = format_context_summary()
            assert len(result) > 0
        except Exception as e:
            pytest.fail(f"format_context_summary raised: {e}")


# ─── test_get_context ────────────────────────────────────────────────────────

class TestGetContext:

    def test_returns_dict(self):
        ctx = get_context()
        assert isinstance(ctx, dict)

    def test_includes_fresh_market_mode(self):
        """get_context() should recompute market_mode."""
        # Save with stale mode
        ctx = load_context()
        ctx["market_mode"] = "NORMAL"
        ctx["macro"]["macro_shock_active"] = True  # should be RISK_OFF
        save_context(ctx)

        fresh = get_context()
        assert fresh["market_mode"] == "RISK_OFF"

    def test_no_crash_when_file_missing(self):
        result = get_context()
        assert result["market_mode"] == "NORMAL"


# ─── test_new_default_context_fields ─────────────────────────────────────────

class TestNewDefaultContextFields:

    def test_default_has_foreign_flow_key(self):
        ctx = load_context()
        assert "foreign_flow" in ctx

    def test_default_has_market_gate_key(self):
        ctx = load_context()
        assert "market_gate" in ctx

    def test_default_has_premarket_key(self):
        ctx = load_context()
        assert "premarket" in ctx

    def test_default_market_gate_is_open(self):
        ctx = load_context()
        assert ctx["market_gate"] == "OPEN"

    def test_default_foreign_flow_signal_neutral(self):
        ctx = load_context()
        assert ctx["foreign_flow"]["foreign_flow_signal"] == "NEUTRAL"

    def test_default_premarket_prediction_netral(self):
        ctx = load_context()
        assert ctx["premarket"]["ihsg_open_prediction"] == "NETRAL"

    def test_default_foreign_flow_keys(self):
        ctx = load_context()
        ff = ctx["foreign_flow"]
        assert "foreign_flow_signal" in ff
        assert "top_bought" in ff
        assert "top_sold" in ff

    def test_default_premarket_keys(self):
        ctx = load_context()
        pm = ctx["premarket"]
        assert "us_signal" in pm
        assert "asia_signal" in pm
        assert "ihsg_open_prediction" in pm

    def test_default_market_breadth_keys(self):
        ctx = load_context()
        mb = ctx["market_breadth"]
        assert "breadth_pct" in mb
        assert "ihsg_change_from_open" in mb


# ─── test_update_foreign_flow ────────────────────────────────────────────────

class TestUpdateForeignFlow:

    def test_updates_foreign_flow_signal(self):
        update_foreign_flow("BULLISH", ["BBCA", "BBRI"], ["GOTO"])
        ctx = load_context()
        assert ctx["foreign_flow"]["foreign_flow_signal"] == "BULLISH"

    def test_updates_top_bought(self):
        update_foreign_flow("BULLISH", ["BBCA", "BBRI"], [])
        ctx = load_context()
        assert "BBCA" in ctx["foreign_flow"]["top_bought"]
        assert "BBRI" in ctx["foreign_flow"]["top_bought"]

    def test_updates_top_sold(self):
        update_foreign_flow("BEARISH", [], ["GOTO", "BUKA"])
        ctx = load_context()
        assert "GOTO" in ctx["foreign_flow"]["top_sold"]

    def test_accepts_neutral_signal(self):
        update_foreign_flow("NEUTRAL", [], [])
        ctx = load_context()
        assert ctx["foreign_flow"]["foreign_flow_signal"] == "NEUTRAL"

    def test_persists_to_file(self, patch_context_file):
        update_foreign_flow("BULLISH", ["BBCA"], [])
        assert os.path.exists(patch_context_file)
        with open(patch_context_file) as f:
            data = json.load(f)
        assert data["foreign_flow"]["foreign_flow_signal"] == "BULLISH"

    def test_empty_lists_allowed(self):
        """Empty lists should not raise."""
        update_foreign_flow("NEUTRAL", [], [])
        ctx = load_context()
        assert ctx["foreign_flow"]["top_bought"] == []
        assert ctx["foreign_flow"]["top_sold"] == []


# ─── test_update_market_breadth ──────────────────────────────────────────────

class TestUpdateMarketBreadth:

    def test_updates_market_gate(self):
        update_market_breadth("CAUTIOUS", 45.0, -0.8)
        ctx = load_context()
        assert ctx["market_gate"] == "CAUTIOUS"

    def test_updates_breadth_pct(self):
        update_market_breadth("OPEN", 67.5, 0.3)
        ctx = load_context()
        assert ctx["market_breadth"]["breadth_pct"] == 67.5

    def test_updates_ihsg_change(self):
        update_market_breadth("CLOSED", 28.0, -1.8)
        ctx = load_context()
        assert ctx["market_breadth"]["ihsg_change_from_open"] == -1.8

    def test_gate_closed(self):
        update_market_breadth("CLOSED", 25.0, -2.0)
        ctx = load_context()
        assert ctx["market_gate"] == "CLOSED"

    def test_gate_open(self):
        update_market_breadth("OPEN", 70.0, 0.5)
        ctx = load_context()
        assert ctx["market_gate"] == "OPEN"

    def test_persists_to_file(self, patch_context_file):
        update_market_breadth("CAUTIOUS", 42.0, -0.6)
        assert os.path.exists(patch_context_file)
        with open(patch_context_file) as f:
            data = json.load(f)
        assert data["market_gate"] == "CAUTIOUS"
        assert data["market_breadth"]["breadth_pct"] == 42.0


# ─── test_update_premarket ────────────────────────────────────────────────────

class TestUpdatePremarket:

    def test_updates_us_signal(self):
        update_premarket("BULLISH", "NEUTRAL", "POSITIF")
        ctx = load_context()
        assert ctx["premarket"]["us_signal"] == "BULLISH"

    def test_updates_asia_signal(self):
        update_premarket("NEUTRAL", "BEARISH", "NETRAL")
        ctx = load_context()
        assert ctx["premarket"]["asia_signal"] == "BEARISH"

    def test_updates_prediction(self):
        update_premarket("BULLISH", "BULLISH", "POSITIF")
        ctx = load_context()
        assert ctx["premarket"]["ihsg_open_prediction"] == "POSITIF"

    def test_accepts_negatif_prediction(self):
        update_premarket("BEARISH", "BEARISH", "NEGATIF")
        ctx = load_context()
        assert ctx["premarket"]["ihsg_open_prediction"] == "NEGATIF"

    def test_accepts_netral_prediction(self):
        update_premarket("NEUTRAL", "NEUTRAL", "NETRAL")
        ctx = load_context()
        assert ctx["premarket"]["ihsg_open_prediction"] == "NETRAL"

    def test_persists_to_file(self, patch_context_file):
        update_premarket("BEARISH", "NEUTRAL", "NEGATIF")
        assert os.path.exists(patch_context_file)
        with open(patch_context_file) as f:
            data = json.load(f)
        assert data["premarket"]["us_signal"] == "BEARISH"
        assert data["premarket"]["ihsg_open_prediction"] == "NEGATIF"

    def test_multiple_updates_accumulate(self):
        """Multiple premarket updates should persist the latest."""
        update_premarket("BULLISH", "BULLISH", "POSITIF")
        update_premarket("BEARISH", "BEARISH", "NEGATIF")
        ctx = load_context()
        assert ctx["premarket"]["us_signal"] == "BEARISH"
        assert ctx["premarket"]["ihsg_open_prediction"] == "NEGATIF"
