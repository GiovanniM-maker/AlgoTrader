# Quando pushare — Guida al push su GitHub e deploy su server

Questa guida spiega come pubblicare le modifiche sia su **GitHub** (backup e versioning) sia sul **server GCP** (VM in produzione).

---

## Prerequisiti

- **Git** installato e configurato
- **SSH key** collegata a GitHub (per `git@github.com:...`)
- **gcloud** autenticato (`gcloud auth login`)
- Esegui i comandi dalla **root del progetto**:
  ```bash
  cd "/Users/giovannimavilla/.claude-worktrees/Richieste difficile | App Trading/nostalgic-lalande/algo-trader-pro"
  ```

---

## 1. Push su GitHub

### Prima volta (collegamento iniziale)

Se il repo non è ancora collegato a GitHub:

```bash
git remote add origin git@github.com:GiovanniM-maker/AlgoTrader.git
git branch -M main
git add .
git commit -m "Descrizione delle modifiche"
git push -u origin main
```

### Dopo la prima volta (modifiche successive)

```bash
git add .
git commit -m "Descrizione delle modifiche"
git push
```

---

## 2. Deploy sul server GCP

Per aggiornare il bot sulla VM (dashboard, engine, ecc.):

```bash
./scripts/deploy_gcp.sh
```

Lo script:
- Copia il progetto sulla VM
- Ricostruisce l’immagine Docker
- Riavvia i container

**Dashboard:** http://34.45.210.181:8000

---

## 3. Workflow completo (GitHub + server)

Quando hai finito le modifiche e vuoi pubblicare tutto:

```bash
# 1. Vai nella cartella del progetto
cd "/Users/giovannimavilla/.claude-worktrees/Richieste difficile | App Trading/nostalgic-lalande/algo-trader-pro"

# 2. Push su GitHub
git add .
git commit -m "Descrizione delle modifiche"
git push

# 3. Deploy sul server GCP
./scripts/deploy_gcp.sh
```

---

## Riepilogo comandi

| Azione | Comando |
|--------|---------|
| Solo GitHub | `git add . && git commit -m "msg" && git push` |
| Solo server | `./scripts/deploy_gcp.sh` |
| Entrambi | Prima `git push`, poi `./scripts/deploy_gcp.sh` |

---

## Troubleshooting

- **"Permission denied (publickey)"** su GitHub → Verifica che la SSH key sia aggiunta al tuo account GitHub
- **"gcloud non autenticato"** → Esegui `gcloud auth login`
- **Deploy lento** → La prima build Docker può richiedere alcuni minuti
