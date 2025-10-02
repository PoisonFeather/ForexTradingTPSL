# 📈 Forex Ensemble TPSL Advisor

Un proiect educațional în Python care folosește mai multe **strategii de day trading** (EMA crossover, MACD, Bollinger, ATR breakout etc.) pentru a propune **entry / stop loss / take profit** pe perechi forex.  
Rezultatele strategiilor sunt combinate printr-un **agregator cu scoruri de încredere**, iar verdictul final afișează direcția (LONG/SHORT/NEUTRAL), TP/SL și raportul Risk:Reward.

> ⚠️ **Disclaimer:** Acest cod este pentru scopuri educaționale și testare. Nu reprezintă sfat financiar sau sistem de trading recomandat pentru live markets.

---

## 🚀 Funcționalități
- Fetch de cotații și candles prin **provideri modulari**:
  - `twelve_data` (free intraday FX, limitat)
  - `alpha_vantage` (intraday doar premium)
  - `mock` (generator sintetic pentru teste)
- 8 strategii implementate (`forex_ensemble/strategies/`):
  - EMA crossover + RSI filter
  - Bollinger mean reversion
  - MACD crossover
  - Opening Range Breakout
  - ATR Channel Breakout
  - Keltner Squeeze
  - Trend Pullback to EMA
  - Donchian Breakout
- Agregator cu:
  - normalizare direcție (long/short/neutral)
  - medie ponderată TP/SL/entry
  - consens + confidence level
- CLI cu output tabelar (`rich`) și opțional JSON
- Opțiuni extra:
  - risk sizing simplu (lots) în funcție de capital și % risc
  - dump semnale în CSV/JSON pentru analiză ulterioară

---

## 📦 Instalare

1. Clonează repo-ul:
   ```bash
   git clone https://github.com/<user>/ForexTradingTPSL.git
   cd ForexTradingTPSL
Creează un virtualenv și instalează dependințele:
python -m venv .venv
source .venv/bin/activate
pip install -e .
Setează variabilele de mediu în fișier .env:
TWELVE_DATA_API_KEY=your_api_key_here
ALPHA_VANTAGE_API_KEY=optional_if_you_have_premium
DEFAULT_SYMBOL=EURUSD
DEFAULT_TIMEFRAME=5min
🖥️ Utilizare CLI
Help general:
python -m forex_ensemble.app --help
Exemplu cu provider mock
python -m forex_ensemble.app advise --provider mock --symbol EURUSD --timeframe 5min
Exemplu cu Twelve Data (recomandat)
python -m forex_ensemble.app advise --provider twelve_data --symbol EURUSD --timeframe 5min
Output JSON
python -m forex_ensemble.app advise --provider twelve_data --symbol EURUSD --timeframe 5min --json
Cu risk sizing
python -m forex_ensemble.app advise \
  --provider twelve_data \
  --symbol EURUSD \
  --timeframe 5min \
  --risk-pct 1.0 \
  --equity 10000
📊 Exemple de output
Tabel semnale brute:
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┓
┃ Strategy                   ┃ Dir   ┃ Entry ┃ SL    ┃ TP    ┃ Conf ┃ Notes               ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━━━━━━━━┩
│ EMA Crossover + RSI filter │ long  │ 1.1755│1.1750 │1.1763 │ 0.68 │ ema_f=1.1749,...    │
│ Opening Range Breakout     │ long  │ 1.1755│1.1751 │1.1762 │ 1.00 │ OR=(1.1725,1.1734)  │
│ ...                        │ ...   │ ...   │ ...   │ ...   │ ...  │ ...                 │
└────────────────────────────┴───────┴───────┴───────┴───────┴──────┴─────────────────────┘
Decizia agregată:
Ensemble Decision
Direction: LONG  |  Confidence: 0.91
Entry: 1.17551 | SL: 1.17508 | TP: 1.17632 | R:R≈1.92
📂 Structură proiect
forex_ensemble/
├─ app.py              # CLI
├─ config.py           # settings din .env
├─ types.py            # dataclasses comune
├─ data/               # provideri (mock, alpha_vantage, twelve_data)
├─ strategies/         # 8 strategii implementate
├─ ensemble/           # agregator de semnale
├─ utils/              # indicatori tehnici (EMA, RSI, ATR, etc.)
tests/                 # teste unitare


⚠️ Disclaimer
Acest proiect este destinat exclusiv educației și cercetării. Nu reprezintă sfat financiar. Trading-ul pe piața forex implică risc ridicat de pierdere a capitalului. Folosește date mock sau paper trading pentru teste.