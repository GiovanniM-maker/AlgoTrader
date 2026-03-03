#!/bin/bash
# =============================================================================
# AlgoTrader Pro - Setup SSL con nginx
# =============================================================================
# Due opzioni:
#   A) Con DOMINIO: Let's Encrypt (certificato gratuito, valido)
#   B) Solo IP: certificato self-signed (browser mostrerà avviso, ma connessione cifrata)
#
# Uso:
#   Con dominio:  sudo ./scripts/setup_ssl_nginx.sh mio-dominio.com
#   Solo IP:      sudo ./scripts/setup_ssl_nginx.sh --self-signed
# =============================================================================

set -e

DOMAIN="${1:-}"
SELF_SIGNED=false
if [ "$1" = "--self-signed" ]; then
  SELF_SIGNED=true
  DOMAIN=""
fi

echo "=== AlgoTrader Pro - Setup SSL ==="
echo ""

# Verifica root
if [ "$(id -u)" -ne 0 ]; then
  echo "Esegui con: sudo $0 $*"
  exit 1
fi

# Installa nginx e certbot (se non presenti)
apt-get update -qq
apt-get install -y nginx certbot python3-certbot-nginx 2>/dev/null || true

# Crea directory per certificati
mkdir -p /etc/nginx/ssl

if [ "$SELF_SIGNED" = true ]; then
  # --- Opzione B: Self-signed ---
  echo "Generazione certificato self-signed..."
  openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout /etc/nginx/ssl/algotrader.key \
    -out /etc/nginx/ssl/algotrader.crt \
    -subj "/CN=AlgoTrader Pro/O=AlgoTrader/C=IT" 2>/dev/null
  echo "  ✓ Certificato creato (valido 365 giorni)"
  echo ""
  echo "  ⚠️  Il browser mostrerà 'Connessione non sicura' — clicca 'Avanzate' → 'Procedi'"
  echo ""
else
  # --- Opzione A: Let's Encrypt (richiede dominio) ---
  if [ -z "$DOMAIN" ]; then
    echo "Per Let's Encrypt serve un dominio. Esempio:"
    echo "  sudo $0 algotrader.tuodominio.com"
    echo ""
    echo "Assicurati che il DNS punti a questo server:"
    echo "  A record: $DOMAIN → $(curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip 2>/dev/null || echo 'IP_ESTERNO')"
    exit 1
  fi
  echo "Configurazione per dominio: $DOMAIN"
  # Prima configura nginx base, poi certbot
fi

# Config nginx
NGINX_CONF="/etc/nginx/sites-available/algotrader"

if [ "$SELF_SIGNED" = true ]; then
  cat > "$NGINX_CONF" << 'NGINX_SELF'
map $http_upgrade $connection_upgrade {
    default upgrade;
    ''      close;
}
upstream algotrader_app { server 127.0.0.1:8000; }
server {
    listen 80;
    server_name _;
    return 301 https://\$host\$request_uri;
}
server {
    listen 443 ssl;
    server_name _;
    ssl_certificate /etc/nginx/ssl/algotrader.crt;
    ssl_certificate_key /etc/nginx/ssl/algotrader.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    location /ws/ {
        proxy_pass http://algotrader_app;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection $connection_upgrade;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
        proxy_send_timeout 86400;
    }
    location / {
        proxy_pass http://algotrader_app;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection $connection_upgrade;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
    }
}
NGINX_SELF
else
  # Solo HTTP (per Let's Encrypt: certbot modificherà la config)
  cat > "$NGINX_CONF" << 'NGINX_EOF'
upstream algotrader_app { server 127.0.0.1:8000; }
server {
    listen 80;
    server_name _;
    location / {
        proxy_pass http://algotrader_app;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
    }
}
NGINX_EOF
fi

ln -sf "$NGINX_CONF" /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default 2>/dev/null || true
nginx -t && systemctl reload nginx
echo "  ✓ nginx configurato"

# Apri porta 443 su GCP (se gcloud disponibile)
if command -v gcloud &>/dev/null; then
  gcloud compute firewall-rules create allow-https --allow=tcp:443 --direction=INGRESS --priority=1000 2>/dev/null || echo "  (regola firewall 443 già esistente o gcloud non configurato)"
fi

echo ""
echo "=== Completato ==="
if [ "$SELF_SIGNED" = true ]; then
  IP=$(curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip 2>/dev/null || echo "IP_ESTERNO")
  echo "Dashboard HTTPS: https://${IP}"
  echo "(Accetta l'avviso del browser per procedere)"
else
  echo "Per Let's Encrypt: certbot --nginx -d $DOMAIN"
fi
echo ""
