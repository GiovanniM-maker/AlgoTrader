# AlgoTrader Pro

> **Bot di trading algoritmico professionale su crypto** — Strategia ibrida proprietaria a 5 layer, ML ensemble (XGBoost + Random Forest + LSTM), paper trading con €10.000 virtuali, dashboard real-time completa.

---

## Indice

- [Architettura](#architettura)
- [Quick Start](#quick-start)
- [Installazione manuale](#installazione-manuale)
- [Docker](#docker)
- [Configurazione](#configurazione)
- [Workflow completo](#workflow-completo)
- [Dashboard](#dashboard)
- [API Reference](#api-reference)
- [Strategia e Logica](#strategia-e-logica)
- [Risk Management](#risk-management)
- [FAQ](#faq)

---

## Architettura

```
                     ┌─────────────────────────────────┐
                     │        DATI DI MERCATO           │
                     │  Bybit WS + REST  |  CoinGecko  │
                     └──────────────┬──────────────────┘
                                    │ OHLCV (Parquet cache)
                     ┌──────────────▼──────────────────┐
                     │       FEATURE ENGINEERING        │
                     │     50+ features per candela     │
                     └──────┬──────────┬───────────────┘
                            │          │
           ┌────────────────▼──┐  ┌───▼────────────────┐
           │   SIGNAL LAYERS   │  │   ML ENSEMBLE       │
           │ L1: Tecnico  30%  │  │  XGBoost      33%  │
           │ L2: Volume   25%  │  │  RandomForest 33%  │
           │ L3: Sentiment 15% │  │  LSTM (PyTorch)33% │
           └────────────┬──────┘  └───┬────────────────┘
                        │   weight 30%│ weight 30%
                        └─────┬───────┘
                     ┌────────▼────────────────────────┐
                     │      STRATEGY AGGREGATOR         │
                     │   confidence score 0-100         │
                     │   threshold: 65 LONG / 70 SHORT  │
                     └──────────────┬──────────────────┘
                                    │
                     ┌──────────────▼──────────────────┐
                     │         RISK MANAGER             │
                     │  Kelly ½ + ATR stop + EV guard  │
                     └──────────────┬──────────────────┘
                                    │
                     ┌──────────────▼──────────────────┐
                     │        PAPER EXECUTOR            │
                     │   slippage 0.05%, fee 0.1%      │
                     └──────────────┬──────────────────┘
                                    │
          ┌─────────────────────────▼──────────────────────────┐
          │              FASTAPI + WEBSOCKET                    │
          │         Dashboard real-time su localhost:8000       │
          └────────────────────────────────────────────────────┘
```

### Stack tecnologico

| Layer | Tecnologia |
|---|---|
| Backend | Python 3.11, FastAPI, uvicorn |
| Data | pybit (Bybit V5), requests (CoinGecko), pandas 2.x, Parquet |
| Indicatori | pandas-ta (130+ indicatori tecnici) |
| ML | XGBoost + scikit-learn RF + PyTorch LSTM 2-layer |
| Sentiment | CryptoPanic API, Fear&Greed (alternative.me), Google Trends (pytrends) |
| NLP | VADER (NLTK) per scoring news |
| DB | SQLite (dev) → PostgreSQL (prod) |
| Scheduling | APScheduler 3.x |
| Dashboard | HTML5 + Vanilla JS, TradingView Lightweight-Charts |
| Deploy | Docker + Docker Compose |

---

## Quick Start

### Prerequisiti
- Python 3.11+
- pip
- (opzionale) Docker + Docker Compose

### 1. Clone e setup ambiente

```bash
git clone <repo-url>
cd algo-trader-pro

# Crea virtualenv
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# oppure: .venv\Scripts\activate  # Windows

# Installa dipendenze
pip install -r requirements.txt
```

### 2. Configura le variabili d'ambiente

```bash
cp .env.example .env
# Modifica .env con le tue API key
nano .env  # o usa il tuo editor preferito
```

Variabili minime da impostare:

```bash
# Scegli provider (coingecko = no key richiesta, bybit = inserisci key sotto)
DATA_PROVIDER=coingecko

# Bybit (opzionale per phase 2)
BYBIT_API_KEY=la_tua_key
BYBIT_API_SECRET=il_tuo_secret

# CryptoPanic (opzionale ma consigliato per sentiment)
CRYPTOPANIC_API_KEY=la_tua_key
```

### 3. Scarica dati storici

```bash
# 2 anni di dati per BTC e ETH (default)
python scripts/download_historical.py

# Oppure personalizza
python scripts/download_historical.py --symbols BTC ETH SOL --days 365
```

### 4. Addestra i modelli ML

```bash
# Training completo (XGBoost + RF + LSTM)
python scripts/train_models.py

# Senza LSTM (più veloce, ~5 min vs ~30 min)
python scripts/train_models.py --no-lstm
```

### 5. Esegui un backtest

```bash
# Backtest 1 anno BTC con Monte Carlo e report HTML
python scripts/run_backtest.py --days 365 --monte-carlo --report --open-browser
```

### 6. Avvia il bot (paper trading)

```bash
python scripts/start_bot.py
# Dashboard aperta automaticamente su http://localhost:8000
```

---

## Installazione manuale

### requirements.txt — dipendenze principali

```
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
websockets>=12.0
pydantic>=2.5.0
python-dotenv>=1.0.0
pyyaml>=6.0.1

# Data
pandas>=2.1.0
numpy>=1.26.0
pyarrow>=14.0.0
pandas-ta>=0.3.14b
pybit>=5.6.0
requests>=2.31.0
aiohttp>=3.9.0
pytrends>=4.9.2

# ML
scikit-learn>=1.4.0
xgboost>=2.0.3
torch>=2.1.0  # CPU version
joblib>=1.3.2

# NLP/Sentiment
nltk>=3.8.1
vaderSentiment>=3.3.2

# DB
SQLAlchemy>=2.0.25
aiosqlite>=0.19.0
asyncpg>=0.29.0  # per PostgreSQL

# Scheduling
APScheduler>=3.10.4

# Utils
tqdm>=4.66.1
tabulate>=0.9.0
```

---

## Docker

### Avvio rapido con Docker

```bash
# Build e avvio (SQLite, paper trading)
docker compose -f docker/docker-compose.yml up -d

# Verifica stato
docker compose -f docker/docker-compose.yml ps

# Log in tempo reale
docker compose -f docker/docker-compose.yml logs -f app

# Dashboard
open http://localhost:8000
```

### Con PostgreSQL (produzione)

```bash
# Aggiorna .env
POSTGRES_PASSWORD=una_password_sicura
DATABASE_URL=postgresql+asyncpg://algotrader:una_password_sicura@postgres:5432/algotrader

# Avvia con profilo postgres
docker compose -f docker/docker-compose.yml --profile postgres up -d

# Con pgAdmin (UI database)
docker compose -f docker/docker-compose.yml --profile postgres --profile admin up -d
# pgAdmin su http://localhost:5050
```

### Eseguire script in Docker

```bash
# Download dati dentro il container
docker compose -f docker/docker-compose.yml run --rm app \
    python scripts/download_historical.py --symbols BTC ETH

# Training modelli
docker compose -f docker/docker-compose.yml run --rm app \
    python scripts/train_models.py --no-lstm

# Backtest
docker compose -f docker/docker-compose.yml run --rm app \
    python scripts/run_backtest.py --report
```

---

## Configurazione

### Parametri principali (`config/settings.yaml`)

```yaml
paper_trading:
  initial_capital: 10000.0    # Capitale virtuale in €
  max_positions: 3            # Posizioni simultanee massime

strategy:
  confidence_threshold_long: 65   # Min confidence per LONG
  confidence_threshold_short: 70  # Min confidence per SHORT
  layer_weights:
    technical: 0.30
    volume: 0.25
    sentiment: 0.15
    ml_ensemble: 0.30

risk:
  kelly_fraction: 0.5         # ½ Kelly (conservativo)
  min_position_size: 0.01     # 1% min capitale per trade
  max_position_size: 0.15     # 15% max capitale per trade
  max_portfolio_risk: 0.20    # 20% max rischio totale
  atr_sl_multiplier: 2.0      # ATR × 2 = stop loss
  tp_rr_ratio: 2.5            # R:R = 2.5x
  max_hold_hours: 48          # Exit dopo 48h se in perdita
  ev_lookback_trades: 30      # EV calcolato su ultimi 30 trade
```

### Variabili d'ambiente (`.env`)

| Variabile | Descrizione | Default |
|---|---|---|
| `APP_ENV` | Ambiente (paper/live) | `paper` |
| `LOG_LEVEL` | Livello log | `INFO` |
| `DATA_PROVIDER` | Provider dati (coingecko/bybit) | `coingecko` |
| `BYBIT_API_KEY` | API key Bybit | — |
| `BYBIT_API_SECRET` | API secret Bybit | — |
| `BYBIT_TESTNET` | Usa testnet Bybit | `false` |
| `CRYPTOPANIC_API_KEY` | API key CryptoPanic | — |
| `DATABASE_URL` | URL database | `sqlite:///./database/algotrader.db` |
| `SECRET_KEY` | Chiave segreta app | — |
| `TELEGRAM_BOT_TOKEN` | Token bot Telegram | — |
| `TELEGRAM_CHAT_ID` | Chat ID Telegram | — |

---

## Workflow completo

```
1. download_historical.py   ← Scarica 2 anni OHLCV → Parquet cache
         │
         ▼
2. train_models.py          ← Feature engineering → XGBoost + RF + LSTM
         │
         ▼
3. run_backtest.py          ← Backtest 2 anni + Monte Carlo + report HTML
         │
         ▼
4. start_bot.py             ← Paper trading live + dashboard http://localhost:8000
```

---

## Dashboard

Apri il browser su `http://localhost:8000` per accedere alla dashboard completa.

### Sezioni disponibili

| Sezione | Contenuto |
|---|---|
| **Dashboard** | Equity attuale, P&L oggi, Win Rate, Fear&Greed live |
| **Portfolio** | Equity curve, 9 metriche (Sharpe, Sortino, Calmar...), posizioni aperte |
| **Signals** | Segnali attivi con barre confidence per layer, storico segnali |
| **Trades** | Storico trade completo con breakdown segnali, filtri, paginazione |
| **Backtest** | Form configurazione, avvio asincrono, risultati con grafici Chart.js |
| **ML Models** | Accuracy XGBoost/RF/LSTM, feature importance top 20 |

### Aggiornamento real-time

La dashboard si aggiorna tramite WebSocket (`/ws/realtime`) e polling REST:

| Dato | Frequenza |
|---|---|
| Equity curve | 60 secondi |
| Sommario KPI | 10 secondi |
| Market feed | 15 secondi |
| Segnali live | 30 secondi |
| Trade aperti | WebSocket push istantaneo |
| Trade chiusi | WebSocket push istantaneo |

---

## API Reference

Base URL: `http://localhost:8000/api/v1`

### Endpoints principali

```
GET  /health                  — Health check del server
GET  /api/v1/dashboard/summary — KPI: equity, P&L, win rate, status
GET  /api/v1/dashboard/market-feed — Prezzo, volume, Fear&Greed

GET  /api/v1/portfolio/equity-curve — Serie temporale equity + drawdown
GET  /api/v1/portfolio/positions    — Posizioni aperte
GET  /api/v1/portfolio/metrics      — Sharpe, Sortino, Calmar, Max DD...

GET  /api/v1/trades           — Lista trade (paginata, filtri)
GET  /api/v1/trades/{id}      — Singolo trade con breakdown segnali

GET  /api/v1/signals/live     — Segnali attivi correnti
GET  /api/v1/signals/history  — Storico segnali

GET  /api/v1/ml/performance   — Metriche modelli ML

POST /api/v1/bot/start        — Avvia il bot
POST /api/v1/bot/pause        — Mette in pausa
POST /api/v1/bot/stop         — Ferma il bot
GET  /api/v1/bot/status       — Stato corrente

POST /api/v1/backtests/run    — Avvia backtest asincrono
GET  /api/v1/backtests/{id}   — Risultato backtest
GET  /api/v1/backtests/       — Lista backtest

WS   /ws/realtime             — WebSocket real-time feed
```

### WebSocket eventi

```json
// Trade aperto
{"type": "trade_opened", "data": {"symbol": "BTCUSDT", "direction": "LONG", "entry": 42000, "size": 0.05}}

// Trade chiuso
{"type": "trade_closed", "data": {"symbol": "BTCUSDT", "pnl": 124.5, "exit_reason": "take_profit"}}

// Aggiornamento equity
{"type": "equity_update", "data": {"equity": 10450.25, "pnl_day": 125.5}}

// Segnale generato
{"type": "signal", "data": {"symbol": "BTCUSDT", "direction": "LONG", "confidence": 72.3}}

// Tick di mercato
{"type": "market_tick", "data": {"symbol": "BTCUSDT", "price": 42150.00, "volume_24h": 1234567}}
```

---

## Strategia e Logica

### Flusso di decisione (ogni candela chiusa)

```
Per ogni simbolo monitorato (default: BTCUSDT, ETHUSDT):

  1. LAYER TECNICO (peso 30%)
     ├── RSI(7, 14, 21)        → ipercomprato/ipervenduto
     ├── MACD(12, 26, 9)       → momentum e crossover
     ├── Bollinger Bands       → volatilità e mean-reversion
     ├── EMA Crossover(9,21,50,200) → trend direction
     ├── Ichimoku Cloud        → supporto/resistenza multi-timeframe
     └── VWAP deviation        → fair value vs prezzo corrente

  2. LAYER VOLUME (peso 25%)
     ├── Volume Anomaly        → spike whale detection (z-score > 2.5)
     ├── OBV slope             → accumulo/distribuzione
     └── CVD (Cumulative Vol Delta) → pressione buy vs sell

  3. LAYER SENTIMENT (peso 15%)
     ├── Fear & Greed Index    → 0 (terrore) → 100 (avidità)
     ├── CryptoPanic news      → VADER NLP scoring su ultime 24h
     └── Google Trends         → interesse search "bitcoin" relativo

  4. LAYER ML ENSEMBLE (peso 30%)
     ├── XGBoost classifier    → 50+ feature, early stopping
     ├── Random Forest         → class_weight balanced
     └── LSTM PyTorch          → sequenze 60 candele (lookback 60h)
     └── Meta-learner (LR)     → stacking OOF → confidence 0-100

  5. RISK GATE (filtro finale)
     IF confidence ≥ 65 (LONG) o ≥ 70 (SHORT):
       → Kelly ½ → position size [1%, 15%]
       → ATR × 2.0 → stop loss (× 3.0 se volatilità alta)
       → Stop × 2.5 → take profit (R:R = 2.5)
       → Trailing stop ATR × 1.5
       → Time exit se > 48h e profit < 0.5%
       → Auto-pause se EV rolling 30 trade < 0
```

### Filosofia EV (Expected Value)

Il bot è progettato per massimizzare **l'Expected Value**, non il win rate:

```
EV = (win_rate × avg_win) - (loss_rate × avg_loss)

Esempio:
  Win rate: 40%    avg_win: €300
  Loss rate: 60%   avg_loss: €100
  EV = (0.4 × 300) - (0.6 × 100) = 120 - 60 = +60€ per trade ✅
```

Un win rate del 40% con R:R = 3x è più profittevole di un win rate del 60% con R:R = 1x.

---

## Risk Management

### Kelly Criterion (½ Kelly)

```
f* = (b × p - q) / b × 0.5

dove:
  p = win_rate stimato dal ML (aggiornato ogni 50 trade)
  q = 1 - p
  b = avg_win / avg_loss (dal trade history)
  0.5 = fattore di sicurezza (½ Kelly)

Range clamped: [1%, 15%] del capitale
```

### Stop Loss ATR dinamico

```
In condizioni normali (range daily ≤ 5%):
  stop_distance = ATR(14) × 2.0

In alta volatilità (range daily > 5%):
  stop_distance = ATR(14) × 3.0  ← più largo per evitare stop hunt
```

### Limiti di portafoglio

| Parametro | Valore |
|---|---|
| Max posizioni simultane | 3 |
| Max rischio per singolo trade | 15% capitale |
| Max rischio totale portafoglio | 20% capitale |
| Exit temporale automatica | 48h se profit < 0.5% |
| Auto-pause su EV negativo | EV < 0 su ultimi 30 trade |

---

## FAQ

**D: Posso usarlo per trading reale?**
R: `live_executor.py` è uno stub che lancia `NotImplementedError` su tutti i metodi — è fisicamente impossibile eseguire ordini reali senza modifiche esplicite al codice. Paper trading only per default.

**D: Quanto dura il training ML?**
R: Con `--no-lstm`: ~5-10 minuti. Con LSTM completo: ~20-40 minuti (CPU). Con GPU: ~5 minuti.

**D: Posso aggiungere nuovi simboli?**
R: Sì, modifica `config/settings.yaml` → `trading.symbols`. Il sistema scarica automaticamente i dati al prossimo avvio.

**D: Come passo da CoinGecko a Bybit?**
R: Cambia `DATA_PROVIDER=bybit` nel `.env`. Zero modifiche al codice.

**D: Come aggiungo notifiche Telegram?**
R: Imposta `TELEGRAM_BOT_TOKEN` e `TELEGRAM_CHAT_ID` nel `.env`. Il EventBus gestisce la notifica automaticamente.

**D: Come aggiorno i modelli ML periodicamente?**
R: Crea un cron job: `0 2 * * 0 python scripts/train_models.py` (ogni domenica alle 2:00).

---

## Struttura del progetto

```
algo-trader-pro/
├── config/                          # Configurazione
│   ├── settings.yaml                # Master config
│   └── strategies/hybrid_v1.yaml
├── src/
│   ├── core/                        # Motore principale
│   │   ├── engine.py                # Orchestratore event-driven
│   │   ├── event_bus.py             # Pub/sub interno
│   │   ├── clock.py                 # Astrazione tempo
│   │   └── state_manager.py         # Stato globale thread-safe
│   ├── data/
│   │   ├── providers/               # Bybit WS/REST, CoinGecko, sentiment
│   │   ├── pipeline/                # Normalizer, Feature Engineer
│   │   └── cache/                   # OHLCVStore (Parquet)
│   ├── signals/
│   │   ├── layer1_technical/        # RSI, MACD, BB, EMA, Ichimoku, VWAP
│   │   ├── layer2_volume/           # Volume anomaly, OBV, CVD
│   │   ├── layer3_sentiment/        # F&G, CryptoPanic, Google Trends
│   │   └── aggregator.py            # Weighted sum → score 0-100
│   ├── ml/
│   │   ├── ensemble/                # XGBoost, RF, LSTM, Combiner
│   │   └── training/                # Trainer, WalkForward
│   ├── strategy/                    # hybrid_strategy.py
│   ├── risk/                        # Kelly, ATR, Trailing, TimeExit, EV
│   ├── execution/                   # PaperExecutor (LiveExecutor=stub)
│   ├── portfolio/                   # PortfolioManager, Metrics
│   ├── backtesting/                 # BacktestEngine, MonteCarlo, Report
│   ├── api/                         # FastAPI routes + WebSocket
│   └── utils/                       # Config, Logger, DB
├── dashboard/                       # Frontend HTML/JS
├── database/
│   ├── schema.sql                   # Schema SQLite/PostgreSQL
│   └── ohlcv_cache/                 # Dati Parquet (gitignored)
├── scripts/
│   ├── download_historical.py       # CLI download dati
│   ├── train_models.py              # CLI training ML
│   ├── run_backtest.py              # CLI backtest
│   └── start_bot.py                 # CLI avvio bot
├── docker/
│   ├── Dockerfile                   # Multi-stage build
│   └── docker-compose.yml           # App + optional PostgreSQL
├── results/                         # Report backtest HTML (gitignored)
├── logs/                            # Log file (gitignored)
├── .env                             # Secrets (gitignored!)
├── .env.example                     # Template secrets
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Licenza

Uso privato. Non ridistribuire senza autorizzazione.

---

*AlgoTrader Pro v1.0.0 — Paper Trading Mode Only*
