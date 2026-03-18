# 📈 Saham Scanner

AI-powered IHSG trading assistant dengan hedge fund discipline. Bukan robot trading — ini scanner cerdas yang bantu kamu ambil keputusan beli/jual berdasarkan data, bukan emosi.

> **Disclaimer:** Bukan saran investasi. Semua sinyal harus dikonfirmasi sendiri sebelum eksekusi.

---

## Overview

Saham Scanner adalah daemon yang jalan 24/7 di server dan otomatis:
- Kasih morning briefing sebelum market buka (08:45 WIB)
- Monitor 22 saham IDX30/LQ45 setiap 10 menit selama market hours
- Alert kalau komoditas (oil/gold/rupiah) bergerak signifikan
- Monitor berita Indonesia yang relevan ke saham kamu
- Track posisi yang kamu beli, alert otomatis TP/CL/trailing stop

Semua alert dikirim langsung ke **Telegram**.

---

## Architecture

```
saham-scanner/
├── main.py                 # Daemon scheduler
├── agents/
│   ├── scanner.py          # Technical analysis + whale detection
│   ├── sentinel.py         # News sentiment monitor
│   ├── radar.py            # Commodity + geopolitik monitor
│   └── position_tracker.py # Portfolio P&L tracker
├── data/
│   ├── positions.json      # Posisi aktif kamu
│   ├── signals.json        # History sinyal
│   ├── sentinel_seen.json  # Berita yang sudah diproses
│   └── radar_*.json        # State radar
├── tests/                  # 81 unit tests
├── .env                    # API keys (jangan di-commit!)
├── TDD.md                  # Test guidelines
└── README.md               # Ini
```

---

## Agents

### 1. 🔍 Scanner
**File:** `agents/scanner.py`

Technical analysis engine dengan hedge fund discipline. Sinyal BUY hanya keluar kalau minimal **4 dari 5 kondisi** terpenuhi:

| Kondisi | Logika |
|---|---|
| ✅ Trend naik | MA20 > MA50 |
| ✅ Volume spike | Volume hari ini > 2x rata-rata 5 hari (whale activity) |
| ✅ RSI sehat | RSI < 70 (tidak overbought) |
| ✅ Harga di atas MA | Current price > MA20 |
| ✅ Momentum positif | Daily change > 0% |

**Signal levels:**
- 🐋 **STRONG BUY** (score 4-5): sinyal kuat, whale masuk, konfirmasi teknikal bagus
- 👀 **WATCH** (score 3): potensi bagus tapi belum konfirmasi penuh
- ❌ **AVOID** (score 0-2): skip dulu

**Macro context** (untuk morning briefing):
- WTI Oil, Gold, USD/IDR — pengaruh ke sektor

**Schedule:**
- `08:45 WIB` — Morning briefing + macro context
- `08:55–15:00 WIB` — Real-time scan setiap 10 menit
- `15:35 WIB` — Closing delta report (open vs close semua saham)

**Watchlist (22 saham IDX30/LQ45):**
```
Perbankan : BBCA, BBRI, BMRI, BBNI
Telco     : TLKM, EXCL
Mining    : ANTM, MDKA, MEDC
Tech      : GOTO, BUKA
Industri  : ASII, AALI
Consumer  : UNVR, ICBP, INDF
Coal      : ADRO, PTBA
Semen     : SMGR, INTP
Oil       : AKRA, ELSA
```

**Contoh output:**
```
🏦 MORNING SCAN — 18 Mar 2026 08:45 WIB

📊 MACRO:
  🛢 WTI Oil: $92.36 (-4.0%) → Watch MEDC, AKRA, ELSA
  🥇 Gold: $2,180 (+0.3%) → Watch ANTM, MDKA
  💵 USD/IDR: 15,820 (+0.5%)

🟢 STRONG BUY:
🐋 BBNI Rp4,390 (+1.62%) | RSI 42.9 | Vol 2.42x | Score 4/5
🐋 EXCL Rp2,960 (+15.62%) | RSI 39.6 | Vol 2.4x | Score 4/5

👀 WATCH LIST:
BBCA Rp6,775 (+0.37%) | Score 3/5
BMRI Rp4,730 (+0.64%) | Score 3/5
```

---

### 2. 📰 Sentinel
**File:** `agents/sentinel.py`

Monitor berita keuangan Indonesia secara real-time. Gunakan Claude Haiku untuk analisa sentimen.

**Sources (RSS):**
- Kontan
- Bisnis.com
- CNBC Indonesia
- Detik Finance

**Cara kerja:**
1. Fetch 5 artikel terbaru per source
2. Filter artikel yang relevan ke watchlist (via keyword map)
3. Kirim ke Claude Haiku untuk analisa sentimen + dampak
4. Alert hanya untuk berita dampak **TINGGI** atau **SEDANG**

**Keyword mapping (contoh):**
```
"suku bunga" / "bi rate"  → BBCA, BBRI, BMRI, BBNI
"nikel"                   → ANTM, MDKA, INCO
"batu bara" / "coal"      → ADRO, PTBA, ITMG
"minyak" / "oil"          → MEDC, AKRA, ELSA
"rupiah"                  → BBCA, BBRI, BMRI
```

**Schedule:** Setiap 30 menit selama market hours (09:30, 10:30, 11:30, 13:00, 14:00)

**Contoh output:**
```
📰 NEWS ALERT — 10:32 WIB

🟢🔥 BI Pertahankan Suku Bunga 5.75%
   BI hold rate → positif untuk perbankan, kredit tidak naik
   📊 Dampak: BBCA, BBRI, BMRI, BBNI

🔴📌 Harga Nikel Turun 3% di LME
   Penurunan nikel tekan margin emiten tambang nikel
   📊 Dampak: ANTM, MDKA
```

---

### 3. 🌍 Radar
**File:** `agents/radar.py`

Monitor komoditas global dan berita geopolitik yang mempengaruhi IHSG.

**Komoditas yang dipantau:**
| Komoditas | Ticker | Sektor terdampak |
|---|---|---|
| Gold | GC=F | ANTM, MDKA |
| WTI Oil | CL=F | MEDC, AKRA, ELSA |
| Brent Oil | BZ=F | MEDC, AKRA, ELSA |
| USD/IDR | IDR=X | BBCA, BBRI, BMRI |
| Copper | HG=F | ANTM, MDKA |

**Alert threshold:** Gerakan ≥ **2%** dalam sehari

**Geopolitical news sources:**
- Reuters Business
- Al Jazeera
- BBC Business

**Keywords geopolitik yang dipantau:**
OPEC, oil, fed, interest rate, China, trade war, sanctions, Indonesia, gold, recession, dll.

**Schedule:**
- `09:00 WIB` — Commodity check saat market open
- `09:30, 10:30, 11:30, 13:00, 14:00 WIB` — Commodity + geo check

**Contoh output:**
```
🌍 RADAR ALERT — 13:00 WIB

⚡ PERGERAKAN KOMODITAS SIGNIFIKAN:
🔴 WTI Oil turun 4.0% → Harga: $92.36
   📊 Watch: MEDC, AKRA, ELSA
🔴 Brent Oil turun 2.2% → Harga: $101.12
   📊 Watch: MEDC, AKRA, ELSA
```

---

### 4. 📊 Position Tracker
**File:** `agents/position_tracker.py`

Track posisi saham yang kamu beli. Auto-alert saat TP/CL/trailing stop hit.

**Cara pakai:**
Kirim command ke Telegram (diproses oleh daemon):
```
/beli BBCA 9250 10
       ↑    ↑    ↑
    ticker price lots
```

**Rules:**
| Event | Threshold | Action |
|---|---|---|
| Take Profit | +8% dari entry | 🟢 Alert "Jual sekarang" |
| Cut Loss | -4% dari entry | 🔴 Alert "Cut Loss sekarang" |
| Trailing Stop aktif | +5% dari entry | 🔒 Stop mulai ikut harga naik |
| Trailing Stop trigger | -5% dari peak | 🔒 Alert "Trailing stop hit" |

**Trailing stop logic:**
```
Entry: Rp 9,000
Naik ke Rp 9,500 (+5.5%) → trailing aktif
Peak Rp 9,800 → trailing stop = Rp 9,800 × 0.95 = Rp 9,310
Harga turun ke Rp 9,300 → TRIGGERED: jual di Rp 9,300
Profit terkunci: +3.3% (bukan -4%)
```

**Monitor interval:** Setiap 10 menit selama market hours

**Contoh alert:**
```
🟢 TAKE PROFIT — BBCA

Entry  : Rp 9,250
Current: Rp 10,000 (+8.1%)
P&L    : +Rp 750,000 (10 lots)

💰 TP tercapai! Pertimbangkan jual sekarang.
```

---

## Daemon (`main.py`)

Self-healing scheduler. Kalau satu agent gagal, otomatis di-reset pada run berikutnya (max 3 consecutive failures sebelum cooldown + reset).

**Full schedule:**
```
08:45 WIB  → Morning Scan (Scanner + macro)
08:55 WIB  → Real-time scan mulai (tiap 10 menit)
09:00 WIB  → Commodity check (Radar)
09:30 WIB  → Sentinel + Radar
10:30 WIB  → Sentinel + Radar
11:30 WIB  → Sentinel + Radar
13:00 WIB  → Sentinel + Radar
14:00 WIB  → Sentinel + Radar
15:00 WIB  → Real-time scan terakhir
15:35 WIB  → Closing delta report
Weekend    → SKIP semua
```

---

## Setup

### 1. Install dependencies
```bash
cd /home/asepyudi/saham-scanner
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### 2. Buat `.env`
```bash
ANTHROPIC_API_KEY=sk-ant-...
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=5922770410
```

### 3. Jalankan daemon
```bash
nohup .venv/bin/python -u main.py > /tmp/saham_scanner.log 2>&1 &
echo "PID: $!"
```

### 4. Cek status
```bash
ps aux | grep main.py
tail -f /tmp/saham_scanner.log
```

### 5. Stop daemon
```bash
kill <PID>
```

---

## Tests

```bash
# Jalankan semua tests
.venv/bin/python -m pytest tests/ -v

# Status saat ini: 81 tests passed ✅
```

Lihat `TDD.md` untuk guidelines lengkap.

---

## Data Files

| File | Isi |
|---|---|
| `data/positions.json` | Posisi aktif + P&L |
| `data/signals.json` | History sinyal scanner |
| `data/sentinel_seen.json` | ID berita yang sudah diproses |
| `data/radar_commodities.json` | Harga komoditas terakhir |
| `data/radar_geo_seen.json` | ID geo news yang sudah diproses |

---

## Roadmap

- [ ] Sentinel unit tests
- [ ] Google Maps API key → mudik helper
- [ ] Foreign net buy/sell data dari IDX langsung (lebih akurat)
- [ ] Webhook Telegram → command `/beli` langsung diproses tanpa restart

---

*Built by Dexter 🔪 — hedge fund discipline, real-time execution*
