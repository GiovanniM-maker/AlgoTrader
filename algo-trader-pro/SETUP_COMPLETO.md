# AlgoTrader Pro — Setup completo per far funzionare tutto

## Problema: dashboard vuota, niente segnali, niente dati

Il bot parte ma **senza dati storici e senza modelli ML** non può generare segnali significativi. Ecco cosa serve.

---

## 1. Cosa manca (e perché)

| Componente | Stato attuale | Effetto |
|------------|---------------|---------|
| **Cache OHLCV** | Vuota | Engine carica ~200 candele per valutare la strategy. Senza storico → segnali su 1 sola candela (inutili) |
| **Modelli ML** | Non addestrati | Layer 4 (30% peso) non contribuisce → segnali meno accurati |
| **CoinGecko connect** | Fix applicato | Il provider ora si connette prima del polling → prezzi live funzionano |
| **CRYPTOPANIC_API_KEY** | Opzionale | Senza: sentiment news disabilitato (layer 3 parziale) |

---

## 2. Passi per far funzionare tutto

### A. In locale (prima del deploy)

1. **Configura `.env`**:
   ```
   DATA_PROVIDER=coingecko
   CRYPTOPANIC_API_KEY=xxx   # da https://cryptopanic.com/developers/api/
   ```

2. **Download dati storici** (CoinGecko, nessuna API key):
   ```bash
   python scripts/download_historical.py --provider coingecko --symbols BTC ETH SOL BNB --days 365 --timeframe 5
   ```
   (timeframe 5 = 5m, allineato a `config/settings.yaml` timeframes.primary)

3. **Training modelli ML**:
   ```bash
   python scripts/train_models.py --symbols BTC ETH --timeframe 5 --no-lstm
   ```
   (`--no-lstm` = più veloce, LSTM opzionale)

4. **Deploy** (include cache e modelli nel tarball):
   ```bash
   ./scripts/deploy_gcp.sh
   ```

### B. Sulla VM GCP (se hai già deployato senza dati)

SSH sulla VM e esegui:

```bash
cd /tmp/algo-trader-pro

# 1. Download storico (CoinGecko)
sudo docker compose -f docker/docker-compose.yml run --rm app \
  python scripts/download_historical.py --provider coingecko --symbols BTC ETH SOL BNB --days 365 --timeframe 5

# 2. Training modelli (5m = allineato a engine)
sudo docker compose -f docker/docker-compose.yml run --rm app \
  python scripts/train_models.py --symbols BTC ETH --timeframe 5 --no-lstm

# 3. Riavvio bot
sudo docker compose -f docker/docker-compose.yml restart app
```

Oppure usa lo script:
```bash
sudo bash scripts/setup_vm_first_time.sh
```

---

## 3. Verifiche

- **Cache OHLCV**: `ls database/ohlcv_cache/BTCUSDT/` → deve esserci `5/` con `data.parquet`
- **Modelli ML**: `ls src/ml/models/` → deve esserci `COMBINED_5m_*`
- **Log bot**: `docker compose -f docker/docker-compose.yml logs -f app` → niente "Provider is not connected"
- **Segnali live**: `curl http://localhost:8000/api/v1/signals/live` → JSON con confidence_score

---

## 4. Timeframe e coerenza

- **Engine** usa `timeframes.primary` da config (default `5m` → codice `"5"`)
- **Download** usa `--timeframe 60` (1h) per training ML
- **MLPredictor** cerca `COMBINED_1h_*` (timeframe 1h)
- **Strategy** può usare 5m per segnali live; ML usa 1h per predizioni (configurabile)

Se vuoi tutto su 5m: scarica con `--timeframe 5` e addestra con `--timeframe 5`. I modelli saranno `COMBINED_5m_*` e il predictor deve ricevere timeframe `"5m"` (verifica in `MLPredictor` e `HybridStrategy`).

---

## 5. Bybit (opzionale, per dati migliori)

Se hai API Bybit:
- `.env`: `DATA_PROVIDER=bybit`, `BYBIT_API_KEY`, `BYBIT_API_SECRET`
- Download: `--provider bybit` (più veloce, dati più granulari)
- Bybit supporta anche WebSocket per candele live in tempo reale
