# Deploy AlgoTrader Pro su Google Cloud

## Stato attuale

- ✅ **gcloud CLI installato** in `algo-trader-pro/google-cloud-sdk/`
- ⏳ **Autenticazione** — da completare (vedi sotto)
- 📄 **Script di deploy** pronto: `scripts/deploy_gcp.sh`

---

## Passo 1: Autenticazione (solo una volta)

Apri il **Terminale** (Cursor o macOS) ed esegui:

```bash
export PATH="/Users/giovannimavilla/.claude-worktrees/Richieste difficile | App Trading/nostalgic-lalande/algo-trader-pro/google-cloud-sdk/bin:$PATH"
gcloud auth login
```

1. Si aprirà il browser
2. Accedi con il tuo account Google (quello usato per GCP)
3. Autorizza l’accesso
4. Torna al terminale: l’autenticazione è completata

---

## Passo 2: Esegui il deploy

```bash
cd "/Users/giovannimavilla/.claude-worktrees/Richieste difficile | App Trading/nostalgic-lalande/algo-trader-pro"
chmod +x scripts/deploy_gcp.sh
./scripts/deploy_gcp.sh
```

Lo script:
1. Crea la regola firewall per la porta 8000
2. Copia il progetto sulla VM
3. Installa Docker (se manca)
4. Avvia il bot

---

## Passo 3: Dashboard

Apri nel browser: **http://34.45.210.181:8000**

---

## Aggiungere gcloud al PATH in modo permanente

Aggiungi questa riga al file `~/.zshrc`:

```bash
export PATH="/Users/giovannimavilla/.claude-worktrees/Richieste difficile | App Trading/nostalgic-lalande/algo-trader-pro/google-cloud-sdk/bin:$PATH"
```

Poi esegui: `source ~/.zshrc`

---

## Comandi utili

| Comando | Descrizione |
|---------|-------------|
| `./scripts/deploy_gcp.sh` | Deploy completo |
| Vedere i log | `gcloud compute ssh web-server-20260303-163719 --zone=us-central1-a --command='cd /tmp/algo-trader-pro && docker compose -f docker/docker-compose.yml logs -f app'` |
| Riavviare il bot | `gcloud compute ssh web-server-20260303-163719 --zone=us-central1-a --command='cd /tmp/algo-trader-pro && docker compose -f docker/docker-compose.yml restart'` |

---

## Configurare .env sulla VM

Se serve modificare le API key dopo il deploy:

```bash
gcloud compute ssh web-server-20260303-163719 --zone=us-central1-a
nano /tmp/algo-trader-pro/.env   # modifica CRYPTOPANIC_API_KEY ecc.
cd /tmp/algo-trader-pro && docker compose -f docker/docker-compose.yml restart
exit
```
