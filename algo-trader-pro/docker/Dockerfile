# =============================================================================
# AlgoTrader Pro — Dockerfile
# Multi-stage build: build → runtime
# Python 3.11 slim (riduce immagine finale da ~1.5GB a ~600MB)
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1: Builder — installa dipendenze e compila moduli C
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# Metadata
LABEL maintainer="AlgoTrader Pro"
LABEL version="1.0.0"
LABEL description="Algorithmic trading bot with ML ensemble and real-time dashboard"

# Dipendenze di sistema per compilazione (rimosse nell'immagine finale)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libhdf5-dev \
    pkg-config \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Crea virtualenv isolato
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Aggiorna pip e installa wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copia solo requirements.txt per sfruttare la cache Docker layer
COPY requirements.txt /tmp/requirements.txt

# PyTorch CPU-first: evita di scaricare ~4GB di CUDA su VM con poco spazio
RUN pip install --no-cache-dir torch>=2.2.0 --index-url https://download.pytorch.org/whl/cpu

# Installa il resto (escludi torch già installato)
RUN grep -v '^torch' /tmp/requirements.txt > /tmp/requirements-docker.txt && \
    pip install --no-cache-dir -r /tmp/requirements-docker.txt


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2: Runtime — immagine finale leggera
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Solo runtime deps (niente build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    libgomp1 \
    curl \
    tini \
    && rm -rf /var/lib/apt/lists/*

# Copia virtualenv dal builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Crea utente non-root (sicurezza)
RUN groupadd -r trader && useradd -r -g trader -d /app -s /sbin/nologin trader

# Directory di lavoro
WORKDIR /app

# Copia codice sorgente
COPY --chown=trader:trader . /app

# Crea directory persistenti (montate come volume)
RUN mkdir -p \
    /app/database/ohlcv_cache \
    /app/src/ml/models \
    /app/logs \
    /app/results \
    && chown -R trader:trader /app

# Variabili di ambiente defaults (override in docker-compose)
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app
ENV APP_ENV=paper
ENV LOG_LEVEL=INFO
ENV HOST=0.0.0.0
ENV PORT=8000

# Porta esposta
EXPOSE 8000

# Health check — verifica che l'API risponda
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Usa tini come init process (gestisce segnali SIGINT/SIGTERM correttamente)
ENTRYPOINT ["/usr/bin/tini", "--"]

# Utente non-root
USER trader

# Comando default: avvia il bot
CMD ["python", "scripts/start_bot.py", "--host", "0.0.0.0", "--no-browser"]
