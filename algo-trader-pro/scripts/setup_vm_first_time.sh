#!/bin/bash
# =============================================================================
# AlgoTrader Pro - Setup iniziale sulla VM GCP
# =============================================================================
# Esegui DOPO il deploy, la prima volta (o dopo reset).
#
# Cosa fa:
#   1. Download dati storici OHLCV (CoinGecko o Bybit)
#   2. Training modelli ML (XGBoost + RF, senza LSTM per velocità)
#   3. Riavvio del bot
#
# Prerequisiti:
#   - .env configurato con CRYPTOPANIC_API_KEY (opzionale ma consigliato)
#   - Per dati storici: DATA_PROVIDER=coingecko (default) o bybit con API key
#
# Uso sulla VM:
#   cd /tmp/algo-trader-pro
#   sudo bash scripts/setup_vm_first_time.sh
# =============================================================================

set -e

cd "$(dirname "$0")/.."
COMPOSE="sudo docker compose -f docker/docker-compose.yml"

echo "=== AlgoTrader Pro - Setup iniziale ==="
echo ""

# 1. Download dati storici
echo "1. Download dati OHLCV storici..."
$COMPOSE run --rm app python scripts/download_historical.py \
  --provider coingecko \
  --symbols BTC ETH SOL BNB \
  --days 365 \
  --timeframe 5 \
  || {
    echo "   Fallback: provo con Bybit (serve BYBIT_API_KEY in .env)"
    $COMPOSE run --rm app python scripts/download_historical.py \
      --provider bybit \
      --symbols BTC ETH SOL BNB \
      --days 365 \
      --timeframe 5 \
      || true
  }
echo ""

# 2. Training modelli ML (timeframe 5 = 5m, allineato a config timeframes.primary)
echo "2. Training modelli ML (XGBoost + RF, no LSTM, 5m)..."
$COMPOSE run --rm app python scripts/train_models.py \
  --symbols BTC ETH \
  --timeframe 5 \
  --no-lstm \
  || echo "   ⚠️  Training fallito (serve cache OHLCV da step 1)"
echo ""

# 3. Riavvio bot
echo "3. Riavvio container bot..."
$COMPOSE restart app
echo ""

echo "=== Setup completato ==="
echo "Dashboard: http://\$(curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip 2>/dev/null || echo 'IP'):8000"
echo ""
echo "Verifica: docker compose -f docker/docker-compose.yml logs -f app"
