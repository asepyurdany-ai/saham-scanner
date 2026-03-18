# TDD — Saham Scanner
*Hedge fund discipline: test dulu, build kemudian.*

---

## Philosophy

Setiap agent harus punya unit test sebelum code masuk ke repo. Tidak ada exceptions.
- **Red → Green → Refactor**: tulis test yang fail dulu, baru implement, lalu clean up
- **Tests sebagai dokumentasi**: test names harus jelas mendeskripsikan behavior
- **Isolasi**: setiap test independent, tidak bergantung pada test lain
- **No external calls**: semua API call di-mock dalam tests

---

## Cara Jalankan Tests

```bash
# Semua tests
cd /home/asepyudi/saham-scanner
.venv/bin/python -m pytest tests/ -v

# Satu file
.venv/bin/python -m pytest tests/test_scanner.py -v

# Satu test spesifik
.venv/bin/python -m pytest tests/test_scanner.py::TestSignalScoring::test_strong_buy_signal -v

# Coverage report
.venv/bin/python -m pytest tests/ --cov=agents --cov-report=term-missing
```

**Status saat ini: 81 tests, semua passed ✅**

---

## Test Coverage per Agent

### 1. Scanner (`tests/test_scanner.py`)

**Yang ditest:**

| Test Class | Behavior |
|---|---|
| `TestSignalScoring` | Scoring logic: 4/5 kondisi = BUY, <3 = AVOID |
| `TestTechnicalIndicators` | RSI calculation, MA crossover detection |
| `TestVolumeAnalysis` | Volume spike detection (>2x avg = whale) |
| `TestMacroParsing` | Parse oil/gold/rupiah dari yfinance |
| `TestSendTelegram` | Telegram message formatting + send (mock) |
| `TestSaveSignals` | Persistent signal storage ke JSON |

**Kondisi scoring (hedge fund rules):**
```
✅ Trend naik → MA20 > MA50
✅ Volume spike → Vol hari ini > 2x rata-rata 5 hari
✅ Foreign net buy → proxy via volume + price action
✅ RSI tidak overbought → RSI < 70
✅ Sentimen berita → dari Sentinel agent
```

**Contoh test yang wajib ada kalau tambah fitur baru:**
```python
def test_no_signal_without_volume_confirmation():
    """Tidak ada BUY signal kalau volume normal meski RSI bagus."""
    ...

def test_whale_detection_2x_volume():
    """Volume 2x rata-rata 5 hari = whale activity = score +1."""
    ...

def test_overbought_rsi_blocks_buy():
    """RSI >= 70 → tidak bisa dapat score penuh."""
    ...
```

---

### 2. Position Tracker (`tests/test_position_tracker.py`)

**Yang ditest:**

| Test Class | Behavior |
|---|---|
| `TestAddPosition` | Tambah posisi baru, kalkulasi TP/CL/shares |
| `TestUpdatePosition` | Update harga current, recalculate P&L |
| `TestCheckTpCl` | Trigger TP (+8%), CL (-4%), trailing stop |
| `TestTrailingStop` | Aktivasi di +5%, ikuti harga naik, trigger di -5% dari peak |
| `TestFormatAlert` | Format pesan Telegram yang dikirim ke Asep |
| `TestPersistence` | Load/save positions.json |

**Critical paths yang harus selalu pass:**
```python
def test_tp_hit_at_8_percent():
    """TP alert keluar pas harga +8% dari entry."""
    ...

def test_cl_hit_at_minus_4_percent():
    """CL alert keluar pas harga -4% dari entry."""
    ...

def test_trailing_stop_activates_at_5_percent():
    """Trailing stop mulai aktif saat profit mencapai +5%."""
    ...

def test_trailing_stop_follows_price_up():
    """Trailing stop ikut naik saat harga naik."""
    ...

def test_trailing_stop_triggers_on_pullback():
    """Trailing stop trigger kalau harga turun 5% dari peak."""
    ...
```

---

### 3. Radar (`tests/test_radar.py`)

**Yang ditest:**

| Test Class | Behavior |
|---|---|
| `TestCommodityAlert` | Alert keluar kalau komoditas bergerak >2% |
| `TestSectorImpact` | Mapping komoditas → sektor IHSG yang terdampak |
| `TestGeoNews` | Parse RSS feed geopolitik (mock) |
| `TestAlertThreshold` | Tidak alert kalau movement kecil (<2%) |

---

### 4. Sentinel (belum ada test — TODO)

**Test yang perlu dibuat:**

```python
class TestRSSParsing:
    def test_parse_kontan_feed(): ...
    def test_parse_bisnis_feed(): ...
    def test_handle_empty_feed(): ...
    def test_deduplicate_news(): ...

class TestSentimentAnalysis:
    def test_positive_sentiment_returns_positive(): ...
    def test_negative_news_triggers_alert(): ...
    def test_mock_claude_haiku_response(): ...

class TestKeywordMapping:
    def test_keyword_maps_to_correct_tickers(): ...
    def test_unknown_keyword_returns_empty(): ...
```

---

## Aturan TDD di Project Ini

### Wajib sebelum merge/push:
1. Tulis test untuk feature baru
2. Jalankan full test suite: `pytest tests/ -v`
3. Semua harus pass — tidak ada toleransi untuk broken tests
4. Kalau fix bug: tulis test yang reproduce bug tersebut dulu, baru fix

### Wajib saat refactor:
1. Jangan ubah test kecuali behavior memang berubah secara intentional
2. Run tests sebelum dan sesudah refactor
3. Coverage tidak boleh turun

### Mock policy:
```python
# ✅ BENAR — mock semua external calls
@patch('agents.scanner.yf.Ticker')
@patch('agents.scanner.send_telegram')
def test_morning_scan(mock_telegram, mock_ticker):
    ...

# ❌ SALAH — jangan hit API asli dalam tests
def test_morning_scan():
    result = run_morning_scan()  # ini hit Yahoo Finance + Telegram!
    ...
```

---

## Cara Tambah Test Baru

```bash
# 1. Buat test file baru (kalau agent baru)
touch tests/test_sentinel.py

# 2. Template dasar
"""
Unit tests for agents/sentinel.py
Tests: parse_rss, analyze_sentiment, send_alert
"""
import pytest
from unittest.mock import patch, MagicMock
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.sentinel import parse_rss, analyze_sentiment

class TestRSSParsing:
    def test_parse_valid_feed(self):
        """parse_rss harus return list of articles."""
        ...
```

---

## CI Checklist (sebelum push ke GitHub)

```bash
# Wajib semua green sebelum git push
cd /home/asepyudi/saham-scanner
.venv/bin/python -m pytest tests/ -v --tb=short

# Expected output:
# XX passed, 0 failed
```

Kalau ada yang failed → **fix dulu, baru push**. Tidak ada broken tests di repo.

---

*Last updated: 2026-03-18 by Dexter 🔪*
