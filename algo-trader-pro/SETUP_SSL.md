# AlgoTrader Pro — Setup SSL/HTTPS

Per eliminare l'avviso "Sito non sicuro" del browser, configura HTTPS.

## Opzione A: Certificato self-signed (solo IP, es. 34.45.210.181)

Sulla VM GCP, dopo il deploy:

```bash
# SSH sulla VM
gcloud compute ssh web-server-20260303-163719 --zone=us-central1-a --project=aesthetic-guild-465312-c4

# Esegui lo script (richiede sudo)
cd /tmp/algo-trader-pro
chmod +x scripts/setup_ssl_nginx.sh
sudo ./scripts/setup_ssl_nginx.sh --self-signed
```

Poi apri la **porta 443** nel firewall GCP (se non già aperta):

```bash
gcloud compute firewall-rules create allow-https --allow=tcp:443 --direction=INGRESS
```

Accedi a **https://34.45.210.181** (senza :8000).  
Il browser mostrerà un avviso — clicca "Avanzate" → "Procedi" (la connessione è comunque cifrata).

---

## Opzione B: Let's Encrypt (con dominio)

Se hai un dominio (es. `algotrader.tuodominio.com`) che punta all'IP della VM:

1. Configura il record DNS: `A algotrader.tuodominio.com → IP_DELLA_VM`
2. Sulla VM:
   ```bash
   sudo ./scripts/setup_ssl_nginx.sh algotrader.tuodominio.com
   sudo certbot --nginx -d algotrader.tuodominio.com
   ```
3. Accedi a **https://algotrader.tuodominio.com** — nessun avviso.

---

## WebSocket e HTTPS

Con nginx come reverse proxy, il WebSocket (`/ws/realtime`) funziona correttamente grazie agli header `Upgrade` e `Connection`.  
La dashboard userà automaticamente `wss://` quando la pagina è servita via HTTPS.
