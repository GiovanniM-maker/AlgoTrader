#!/bin/bash
# =============================================================================
# AlgoTrader Pro - Deploy su Google Cloud VM
# =============================================================================
# Prerequisiti:
#   1. gcloud auth login (completato)
#   2. gcloud config set project aesthetic-guild-465312-c4
# =============================================================================

set -e

# Usa gcloud dal progetto se presente, altrimenti da PATH (installare: https://cloud.google.com/sdk/docs/install)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
GCLOUD_BIN="${PROJECT_DIR}/google-cloud-sdk/bin/gcloud"
if [[ ! -x "$GCLOUD_BIN" ]]; then
  GCLOUD_BIN="gcloud"
fi
PROJECT="aesthetic-guild-465312-c4"
ZONE="us-central1-a"
INSTANCE="web-server-20260303-163719"
LOCAL_DIR="$PROJECT_DIR"

export PATH="$(dirname "$GCLOUD_BIN"):$PATH"

echo "=== AlgoTrader Pro - Deploy GCP ==="
echo ""

# Verifica auth (usa GCLOUD_BIN esplicito)
if ! "$GCLOUD_BIN" auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null | grep -q .; then
  echo "ERRORE: gcloud non è autenticato. Esegui: $GCLOUD_BIN auth login"
  exit 1
fi

echo "✓ gcloud autenticato"
"$GCLOUD_BIN" config set project "$PROJECT" 2>/dev/null || true
echo "✓ Progetto: $PROJECT"
echo ""

# 1. Crea regola firewall per porta 8000 (idempotente)
echo "1. Creazione regola firewall (porta 8000)..."
if "$GCLOUD_BIN" compute firewall-rules describe allow-algotrader --project="$PROJECT" 2>/dev/null; then
  echo "   Regola esistente."
else
  "$GCLOUD_BIN" compute firewall-rules create allow-algotrader \
    --project="$PROJECT" \
    --allow=tcp:8000 \
    --direction=INGRESS \
    --priority=1000 \
    --network=default \
    --source-ranges=0.0.0.0/0 \
    --description="AlgoTrader dashboard"
  echo "   Regola creata."
fi
echo ""

# 2. Copia progetto sulla VM (escludi cache e venv)
echo "2. Copia progetto sulla VM..."
TARBALL="/tmp/algotrader-deploy.tar.gz"
PARENT_DIR="$(dirname "$LOCAL_DIR")"
FOLDER_NAME="$(basename "$LOCAL_DIR")"
cd "$PARENT_DIR"
tar -czf "$TARBALL" \
  --exclude='.venv' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.git' \
  --exclude='database/ohlcv_cache/*' \
  --exclude='*.tar.gz' \
  --exclude='google-cloud-sdk' \
  "$FOLDER_NAME"

"$GCLOUD_BIN" compute scp "$TARBALL" "$INSTANCE:/tmp/" --zone="$ZONE" --project="$PROJECT"
rm -f "$TARBALL"
echo "   File copiati."
echo ""

# 3. Esegui setup e avvio sul server
echo "3. Setup e avvio bot sulla VM..."
"$GCLOUD_BIN" compute ssh "$INSTANCE" --zone="$ZONE" --project="$PROJECT" --command='
set -e
cd /tmp
rm -rf algo-trader-pro
tar -xzf algotrader-deploy.tar.gz
rm algotrader-deploy.tar.gz
cd algo-trader-pro

# Installa Docker se non presente
if ! command -v docker &>/dev/null; then
  echo "   Installazione Docker..."
  curl -fsSL https://get.docker.com | sh
  sudo usermod -aG docker $USER
fi

# Crea .env se mancante
if [ ! -f .env ]; then
  cp .env.example .env
  echo "   .env creato da .env.example - modifica CRYPTOPANIC_API_KEY!"
fi

# Libera spazio da build precedenti falliti
echo "   Pulizia cache Docker..."
(docker system prune -af 2>/dev/null || sudo docker system prune -af 2>/dev/null) || true

# Prepara directory e permessi (database deve essere scrivibile dal container)
echo "   Preparazione directory..."
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
mkdir -p database/ohlcv_cache src/ml/models logs results
touch database/algotrader.db 2>/dev/null || true
chmod -R 777 database logs results src/ml/models 2>/dev/null || true

# Avvia con Docker (sudo necessario su VM GCP)
echo "   Avvio container..."
sudo docker compose -f docker/docker-compose.yml up -d --build 2>/dev/null || sudo docker-compose -f docker/docker-compose.yml up -d --build 2>/dev/null || true

echo ""
echo "   ✓ Bot avviato!"
echo "   Dashboard: http://$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip 2>/dev/null || echo "IP_ESTERNO"):8000"
'
echo ""

# Ottieni IP
IP=$("$GCLOUD_BIN" compute instances describe "$INSTANCE" --zone="$ZONE" --project="$PROJECT" --format="get(networkInterfaces[0].accessConfigs[0].natIP)" 2>/dev/null || true)
echo "=== Deploy completato ==="
echo ""
echo "Dashboard: http://${IP:-34.45.210.181}:8000"
echo ""
echo "Log: gcloud compute ssh $INSTANCE --zone=$ZONE --project=$PROJECT --command='cd /tmp/algo-trader-pro && docker compose -f docker/docker-compose.yml logs -f app'"
echo ""
