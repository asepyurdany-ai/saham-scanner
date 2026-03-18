"""
Unit tests for agents/sentinel.py
Tests: _extract_json, get_affected_tickers, format_news_alert, fetch_news, analyze_with_haiku
"""

import json
import os
import sys
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.sentinel import (
    _extract_json,
    get_affected_tickers,
    format_news_alert,
    load_seen,
    save_seen,
)


# --- _extract_json Tests ---

class TestExtractJson:

    def test_pure_json_array(self):
        """Should parse pure JSON array."""
        text = '[{"judul": "test", "sentimen": "POSITIF"}]'
        result = _extract_json(text)
        assert result is not None
        assert isinstance(result, list)
        assert result[0]["sentimen"] == "POSITIF"

    def test_json_with_markdown_code_block(self):
        """Should parse JSON wrapped in ```json ... ``` block."""
        text = '```json\n[{"judul": "test", "sentimen": "NEGATIF"}]\n```'
        result = _extract_json(text)
        assert result is not None
        assert result[0]["sentimen"] == "NEGATIF"

    def test_json_with_plain_code_block(self):
        """Should parse JSON wrapped in ``` ... ``` (no language tag)."""
        text = '```\n[{"judul": "test", "dampak": "TINGGI"}]\n```'
        result = _extract_json(text)
        assert result is not None
        assert result[0]["dampak"] == "TINGGI"

    def test_json_with_leading_text(self):
        """Should extract JSON even with leading explanatory text."""
        text = 'Berikut analisa saya:\n[{"judul": "test", "sentimen": "NETRAL"}]'
        result = _extract_json(text)
        assert result is not None
        assert result[0]["sentimen"] == "NETRAL"

    def test_json_with_trailing_text(self):
        """Should extract JSON even with trailing text."""
        text = '[{"judul": "test", "sentimen": "POSITIF"}]\n\nSemoga membantu.'
        result = _extract_json(text)
        assert result is not None
        assert result[0]["sentimen"] == "POSITIF"

    def test_empty_string_returns_none(self):
        """Empty string should return None."""
        result = _extract_json("")
        assert result is None

    def test_none_input_returns_none(self):
        """None input should return None."""
        result = _extract_json(None)
        assert result is None

    def test_invalid_json_returns_none(self):
        """Gibberish should return None."""
        result = _extract_json("this is not json at all !!!")
        assert result is None

    def test_partial_json_returns_none(self):
        """Truncated JSON should return None."""
        result = _extract_json('[{"judul": "test", "sentimen":')
        assert result is None

    def test_json_object_not_array(self):
        """Should also handle JSON object (not just array)."""
        text = '{"judul": "test", "sentimen": "POSITIF"}'
        result = _extract_json(text)
        assert result is not None
        assert result["sentimen"] == "POSITIF"

    def test_whitespace_around_json(self):
        """Should handle extra whitespace."""
        text = '   \n  [{"judul": "test"}]  \n  '
        result = _extract_json(text)
        assert result is not None

    def test_multiple_fields_preserved(self):
        """All fields should be preserved after extraction."""
        data = [{"judul": "BI Rate", "sentimen": "POSITIF", "dampak": "TINGGI",
                 "saham": ["BBCA", "BBRI"], "ringkasan": "BI tahan rate"}]
        text = json.dumps(data)
        result = _extract_json(text)
        assert result[0]["saham"] == ["BBCA", "BBRI"]
        assert result[0]["ringkasan"] == "BI tahan rate"


# --- get_affected_tickers Tests ---

class TestGetAffectedTickers:

    def test_bank_indonesia_keyword(self):
        """'bank indonesia' maps to banking stocks."""
        tickers = get_affected_tickers("Bank Indonesia Tahan Suku Bunga", "")
        assert "BBCA.JK" in tickers
        assert "BBRI.JK" in tickers

    def test_nikel_keyword(self):
        """'nikel' maps to mining stocks."""
        tickers = get_affected_tickers("Harga Nikel Turun", "dampak ke emiten tambang")
        assert "ANTM.JK" in tickers
        assert "MDKA.JK" in tickers

    def test_coal_keyword_english(self):
        """'coal' maps to coal stocks."""
        tickers = get_affected_tickers("Coal price drops", "")
        assert "ADRO.JK" in tickers
        assert "PTBA.JK" in tickers

    def test_batu_bara_keyword(self):
        """'batu bara' maps to coal stocks."""
        tickers = get_affected_tickers("Harga batu bara turun drastis", "")
        assert "ADRO.JK" in tickers

    def test_oil_keyword(self):
        """'oil' maps to oil stocks."""
        tickers = get_affected_tickers("Oil price surges", "")
        assert "MEDC.JK" in tickers
        assert "AKRA.JK" in tickers

    def test_no_keyword_returns_empty(self):
        """No matching keyword → empty list."""
        tickers = get_affected_tickers("Cuaca cerah di Jakarta hari ini", "")
        assert tickers == []

    def test_case_insensitive(self):
        """Keyword matching should be case-insensitive."""
        tickers = get_affected_tickers("RUPIAH MENGUAT", "")
        assert "BBCA.JK" in tickers

    def test_keyword_in_summary(self):
        """Should also match keywords in summary, not just title."""
        tickers = get_affected_tickers("Berita terbaru", "harga emas naik signifikan")
        assert "ANTM.JK" in tickers

    def test_multiple_keywords_deduped(self):
        """Multiple keyword matches should not duplicate tickers."""
        # 'suku bunga' and 'bi rate' both map to banking — no duplicates
        tickers = get_affected_tickers("BI Rate dan suku bunga naik", "")
        assert tickers.count("BBCA.JK") == 1

    def test_returns_list(self):
        """Return type should always be a list."""
        result = get_affected_tickers("random text", "more random")
        assert isinstance(result, list)


# --- Seen ID Persistence Tests ---

class TestSeenPersistence:

    def test_save_and_load_seen(self, tmp_path):
        """Saved seen IDs should be loadable."""
        import agents.sentinel as s
        original = s.LAST_SEEN_FILE
        s.LAST_SEEN_FILE = str(tmp_path / "seen.json")
        try:
            seen = {"id1", "id2", "id3"}
            save_seen(seen)
            loaded = load_seen()
            assert "id1" in loaded
            assert "id2" in loaded
        finally:
            s.LAST_SEEN_FILE = original

    def test_load_empty_if_no_file(self, tmp_path):
        """Should return empty set if file doesn't exist."""
        import agents.sentinel as s
        original = s.LAST_SEEN_FILE
        s.LAST_SEEN_FILE = str(tmp_path / "nonexistent.json")
        try:
            result = load_seen()
            assert isinstance(result, set)
            assert len(result) == 0
        finally:
            s.LAST_SEEN_FILE = original


# --- analyze_with_haiku Mock Tests ---

class TestAnalyzeWithHaiku:

    @patch('agents.sentinel.anthropic.Anthropic')
    def test_returns_list_on_valid_response(self, mock_anthropic):
        """Should return list when Haiku returns valid JSON."""
        from agents.sentinel import analyze_with_haiku
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='[{"judul": "BI Rate", "sentimen": "POSITIF", "dampak": "TINGGI", "saham": ["BBCA"], "ringkasan": "BI tahan rate"}]')]
        mock_client.messages.create.return_value = mock_response

        articles = [{"source": "Kontan", "title": "BI Rate", "summary": "BI tahan suku bunga"}]
        result = analyze_with_haiku(articles)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["sentimen"] == "POSITIF"

    @patch('agents.sentinel.anthropic.Anthropic')
    def test_returns_empty_on_haiku_empty_response(self, mock_anthropic):
        """Should return [] when Haiku returns empty/unparseable response."""
        from agents.sentinel import analyze_with_haiku
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='')]
        mock_client.messages.create.return_value = mock_response

        articles = [{"source": "Kontan", "title": "Test", "summary": "Test"}]
        result = analyze_with_haiku(articles)
        assert result == []

    @patch('agents.sentinel.anthropic.Anthropic')
    def test_returns_empty_on_markdown_wrapped_json(self, mock_anthropic):
        """Should handle markdown-wrapped JSON from Haiku."""
        from agents.sentinel import analyze_with_haiku
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='```json\n[{"judul": "Test", "sentimen": "NETRAL", "dampak": "RENDAH", "saham": [], "ringkasan": "test"}]\n```')]
        mock_client.messages.create.return_value = mock_response

        articles = [{"source": "Kontan", "title": "Test", "summary": "Test"}]
        result = analyze_with_haiku(articles)
        assert isinstance(result, list)
        assert len(result) == 1

    @patch('agents.sentinel.anthropic.Anthropic')
    def test_returns_empty_list_on_empty_articles(self, mock_anthropic):
        """Should return [] immediately if articles list is empty."""
        from agents.sentinel import analyze_with_haiku
        result = analyze_with_haiku([])
        assert result == []
        mock_anthropic.assert_not_called()

    @patch('agents.sentinel.anthropic.Anthropic')
    def test_handles_api_exception(self, mock_anthropic):
        """Should return [] if Anthropic API throws exception."""
        from agents.sentinel import analyze_with_haiku
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        mock_client.messages.create.side_effect = Exception("API error")

        articles = [{"source": "Kontan", "title": "Test", "summary": "Test"}]
        result = analyze_with_haiku(articles)
        assert result == []
