#!/usr/bin/env python3
"""
AlgoTrader Pro - Main Bot Entry Point
======================================
Avvia il sistema completo: paper trading bot + FastAPI dashboard.

Uso:
    python scripts/start_bot.py [--config config/settings.yaml] [--no-browser]
"""

import argparse
import asyncio
import signal
import sys
import threading
import time
import webbrowser
from pathlib import Path

# Aggiungo root al path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Compatibilità pandas-ta-classic (espone pandas_ta_classic, non pandas_ta)
try:
    import pandas_ta  # noqa: F401
except ImportError:
    import pandas_ta_classic
    import sys
    sys.modules["pandas_ta"] = pandas_ta_classic

import uvicorn

BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║        █████╗ ██╗      ██████╗  ██████╗                         ║
║       ██╔══██╗██║     ██╔════╝ ██╔═══██╗                        ║
║       ███████║██║     ██║  ███╗██║   ██║                        ║
║       ██╔══██║██║     ██║   ██║██║   ██║                        ║
║       ██║  ██║███████╗╚██████╔╝╚██████╔╝                        ║
║       ╚═╝  ╚═╝╚══════╝ ╚═════╝  ╚═════╝                        ║
║                                                                  ║
║   ████████╗██████╗  █████╗ ██████╗ ███████╗██████╗              ║
║      ██╔══╝██╔══██╗██╔══██╗██╔══██╗██╔════╝██╔══██╗             ║
║      ██║   ██████╔╝███████║██║  ██║█████╗  ██████╔╝             ║
║      ██║   ██╔══██╗██╔══██║██║  ██║██╔══╝  ██╔══██╗             ║
║      ██║   ██║  ██║██║  ██║██████╔╝███████╗██║  ██║             ║
║      ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚══════╝╚═╝  ╚═╝            ║
║                                                                  ║
║              P R O  v1.0.0  —  Paper Trading Mode               ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ──────────────────────────────────────────────────────────────────
# Argparse
# ──────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="AlgoTrader Pro — Avvia il bot di paper trading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  python scripts/start_bot.py
  python scripts/start_bot.py --no-browser
  python scripts/start_bot.py --host 0.0.0.0 --port 8080
        """
    )
    parser.add_argument("--config", default="config/settings.yaml",
                        help="Percorso file di configurazione (default: config/settings.yaml)")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Host per FastAPI (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000,
                        help="Porta per FastAPI (default: 8000)")
    parser.add_argument("--no-browser", action="store_true",
                        help="Non aprire il browser automaticamente")
    parser.add_argument("--log-level", default=None,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Override log level dal .env")
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────
# Port cleanup (evita "address already in use")
# ──────────────────────────────────────────────────────────────────

def ensure_port_free(host: str, port: int) -> None:
    """Termina processi che usano la porta, per evitare Errno 48."""
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((host, port))
        sock.close()
        if result == 0:
            # Porta in uso
            try:
                import subprocess
                out = subprocess.run(
                    ["lsof", "-ti", f":{port}"],
                    capture_output=True, text=True, timeout=5,
                )
                if out.returncode == 0 and out.stdout.strip():
                    pids = out.stdout.strip().split()
                    for pid in pids[:3]:
                        subprocess.run(["kill", "-9", pid], capture_output=True, timeout=2)
                    print(f"  ⚠️  Porta {port} era occupata — processi terminati")
                    import time
                    time.sleep(2)
            except Exception:
                pass
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────
# Startup checks
# ──────────────────────────────────────────────────────────────────

def run_startup_checks(cfg) -> bool:
    """Verifica prerequisiti prima di avviare il bot."""
    print("\n📋 Controllo prerequisiti...")
    ok = True

    # 1. Database
    try:
        from src.utils.db import init_db
        init_db()
        print("  ✅ Database inizializzato")
    except Exception as e:
        print(f"  ❌ Errore database: {e}")
        ok = False

    # 2. Directory modelli ML
    models_dir = ROOT / "src" / "ml" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    if any(models_dir.glob("*.pkl")) or any(models_dir.glob("*.pt")) or any(models_dir.glob("*.json")):
        print("  ✅ Modelli ML trovati")
    else:
        print("  ⚠️  Modelli ML non trovati — il bot userà solo i layer tecnico/volume/sentiment")
        print("     Esegui: python scripts/train_models.py per addestrare i modelli")

    # 3. Cache dati storici
    cache_dir = ROOT / "database" / "ohlcv_cache"
    if cache_dir.exists() and any(cache_dir.rglob("*.parquet")):
        count = len(list(cache_dir.rglob("*.parquet")))
        print(f"  ✅ Cache OHLCV trovata ({count} file Parquet)")
    else:
        print("  ⚠️  Cache OHLCV vuota — il provider scaricherà dati in real-time")
        print("     Per pre-scaricare: python scripts/download_historical.py")

    # 4. Provider dati
    data_provider = cfg.get("data_provider", "coingecko")
    print(f"  ℹ️  Provider dati: {data_provider.upper()}")

    # 5. Verifica .env
    import os
    if data_provider == "bybit":
        if not os.getenv("BYBIT_API_KEY") or not os.getenv("BYBIT_API_SECRET"):
            print("  ❌ BYBIT_API_KEY / BYBIT_API_SECRET mancanti nel .env")
            ok = False
        else:
            print("  ✅ Credenziali Bybit trovate")

    cryptopanic_key = os.getenv("CRYPTOPANIC_API_KEY", "")
    if cryptopanic_key:
        print("  ✅ CryptoPanic API key trovata")
    else:
        print("  ⚠️  CryptoPanic API key mancante — sentiment news disabilitato")

    return ok


# ──────────────────────────────────────────────────────────────────
# Component initialization
# ──────────────────────────────────────────────────────────────────

def initialize_components(cfg, cfg_obj=None):
    """Crea e wiring tutti i componenti del sistema."""
    print("\n🔧 Inizializzazione componenti...")

    components = {}
    # cfg_obj è il Pydantic config per SignalAggregator (richiede .strategy.layer_weights)
    if cfg_obj is None and "strategy" in cfg:
        import types
        cfg_obj = types.SimpleNamespace(
            strategy=types.SimpleNamespace(
                layer_weights=types.SimpleNamespace(
                    layer1_technical=cfg.get("strategy", {}).get("layer_weights", {}).get("layer1_technical", 0.30),
                    layer2_volume=cfg.get("strategy", {}).get("layer_weights", {}).get("layer2_volume", 0.25),
                    layer3_sentiment=cfg.get("strategy", {}).get("layer_weights", {}).get("layer3_sentiment", 0.15),
                    layer4_ml=cfg.get("strategy", {}).get("layer_weights", {}).get("layer4_ml", 0.30),
                )
            )
        )

    # ── Data providers ──────────────────────────────────────────
    import os
    data_provider = "bybit" if (os.getenv("BYBIT_API_KEY") and os.getenv("BYBIT_API_SECRET")) else "coingecko"
    cfg["data_provider"] = data_provider

    if data_provider == "bybit":
        from src.data.providers.bybit_rest import BybitRESTProvider
        from src.data.providers.bybit_ws import BybitWebSocketProvider
        components["rest_provider"] = BybitRESTProvider()
        components["ws_provider"] = BybitWebSocketProvider()
        print("  ✅ Bybit REST + WebSocket provider")
    else:
        from src.data.providers.coingecko_rest import CoinGeckoRestProvider
        components["rest_provider"] = CoinGeckoRestProvider()
        components["ws_provider"] = None
        print("  ✅ CoinGecko provider (Phase 1)")

    # ── Sentiment providers ──────────────────────────────────────
    from src.data.providers.fear_greed import FearGreedProvider
    from src.data.providers.cryptopanic import CryptoPanicProvider
    from src.data.providers.google_trends import GoogleTrendsProvider
    components["fear_greed_provider"] = FearGreedProvider()
    components["cryptopanic_provider"] = CryptoPanicProvider()
    components["google_trends_provider"] = GoogleTrendsProvider()
    print("  ✅ Sentiment providers (F&G, CryptoPanic, Google Trends)")

    # ── OHLCV Store ──────────────────────────────────────────────
    from src.data.cache.ohlcv_store import OHLCVStore
    cache_dir = str(ROOT / "database" / "ohlcv_cache")
    components["ohlcv_store"] = OHLCVStore(cache_dir=cache_dir)
    print("  ✅ OHLCV Store (Parquet cache)")

    # ── Feature engineer ─────────────────────────────────────────
    from src.data.pipeline.feature_engineer import FeatureEngineer
    components["feature_engineer"] = FeatureEngineer()
    print("  ✅ Feature Engineer (50+ features)")

    # ── ML Ensemble (opzionale) ──────────────────────────────────
    models_dir = ROOT / "src" / "ml" / "models"
    ml_ensemble = None
    try:
        from src.ml.predictor import MLPredictor
        tf_primary = cfg.get("timeframes", {}).get("primary", "1h")
        predictor = MLPredictor(models_dir=models_dir, timeframe=tf_primary)
        if predictor.load():
            ml_ensemble = predictor
            components["ml_ensemble"] = ml_ensemble
            print("  ✅ ML Ensemble caricato (XGBoost + RF + LSTM)")
        else:
            components["ml_ensemble"] = None
            print("  ⚠️  ML Ensemble non disponibile — esegui: python scripts/train_models.py")
    except Exception as e:
        print(f"  ⚠️  ML Ensemble non disponibile: {e}")
        components["ml_ensemble"] = None

    # ── Signal aggregator ────────────────────────────────────────
    from src.signals.aggregator import SignalAggregator
    components["aggregator"] = SignalAggregator(config=cfg_obj)
    print("  ✅ Signal Aggregator (5-layer)")

    # ── Risk manager ─────────────────────────────────────────────
    from src.risk.risk_manager import RiskManager
    components["risk_manager"] = RiskManager(config=cfg)
    print("  ✅ Risk Manager (Kelly + ATR + EV)")

    # ── Paper executor ───────────────────────────────────────────
    from src.execution.paper_executor import PaperExecutor
    initial_capital = cfg.get("paper_trading", {}).get("initial_capital", 10000.0)
    components["paper_executor"] = PaperExecutor(
        initial_capital=initial_capital,
        fee_rate=cfg.get("paper_trading", {}).get("default_fee_rate", 0.001),
        slippage_std=cfg.get("backtesting", {}).get("slippage_std_pct", 0.0005),
    )
    print("  ✅ Paper Executor (slippage 0.05%, fee 0.1%)")

    # ── Portfolio manager ────────────────────────────────────────
    from src.portfolio.portfolio_manager import PortfolioManager
    db_path = str(ROOT / "database" / "algotrader.db")
    components["portfolio_manager"] = PortfolioManager(
        paper_executor=components["paper_executor"],
        db_path=db_path,
    )
    print(f"  ✅ Portfolio Manager (capitale: €{initial_capital:,.2f})")

    # ── Hybrid strategy ────────────────────────────────────────
    from src.strategy.hybrid_strategy import HybridStrategy
    symbols = list(cfg.get("symbols_mapping", {}).values()) or ["BTCUSDT", "ETHUSDT"]
    cfg["bybit"] = cfg.get("bybit") or {}
    cfg["bybit"]["symbols"] = symbols
    tf_str = cfg.get("timeframes", {}).get("primary", "5m")
    # Mappa stringa → codice Bybit: 5m→5, 15m→15, 1h→60, 4h→240, 1d→D
    bybit_interval = {"5m": "5", "15m": "15", "1h": "60", "4h": "240", "1d": "D"}.get(tf_str, "5")
    cfg["bybit"]["timeframes"] = {"primary": bybit_interval}

    strategy = HybridStrategy(
        config=cfg,
        aggregator=components["aggregator"],
        risk_manager=components["risk_manager"],
        ml_ensemble=components.get("ml_ensemble"),
    )
    components["strategy"] = strategy

    # ── Core engine ──────────────────────────────────────────────
    from src.core.engine import TradingEngine
    components["engine"] = TradingEngine(
        config=cfg,
        paper_executor=components["paper_executor"],
        portfolio_manager=components["portfolio_manager"],
        strategy=components["strategy"],
        bybit_ws=components.get("ws_provider"),
        bybit_rest=components["rest_provider"] if data_provider == "bybit" else None,
        coingecko_rest=components["rest_provider"] if data_provider == "coingecko" else None,
        fear_greed_provider=components["fear_greed_provider"],
        cryptopanic_provider=components["cryptopanic_provider"],
        google_trends_provider=components["google_trends_provider"],
        ohlcv_store=components["ohlcv_store"],
    )
    print("  ✅ Trading Engine inizializzato")

    return components


# ──────────────────────────────────────────────────────────────────
# FastAPI server — stesso event loop dell'engine (WebSocket funziona)
# ──────────────────────────────────────────────────────────────────

def run_server_with_engine(host: str, port: int, engine, paper_executor):
    """Avvia uvicorn nel main thread; l'engine parte dal lifespan (stesso loop = WebSocket OK)."""
    try:
        from src.api import server as api_server
        api_server.set_engine(engine)
        api_server.paper_executor = paper_executor
    except Exception:
        pass

    config = uvicorn.Config(
        "src.api.server:app",
        host=host,
        port=port,
        log_level="warning",
        reload=False,
        workers=1,
    )
    server = uvicorn.Server(config)
    server.run()


def main():
    print(BANNER)

    args = parse_args()

    # ── Carico config ────────────────────────────────────────────
    print("⚙️  Carico configurazione...")
    cfg_obj = None
    try:
        from src.utils.config_loader import load_config
        cfg_path = ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
        cfg_obj = load_config(config_path=cfg_path, force_reload=True)
        cfg = cfg_obj.model_dump() if hasattr(cfg_obj, "model_dump") else {}
    except Exception as e:
        print(f"  ⚠️  Config non caricata ({e}), uso defaults")
        cfg = {}

    # Override log level se specificato
    if args.log_level:
        import os
        os.environ["LOG_LEVEL"] = args.log_level

    # ── Startup checks ───────────────────────────────────────────
    if not run_startup_checks(cfg):
        print("\n❌ Prerequisiti non soddisfatti. Correggi gli errori sopra e riprova.\n")
        sys.exit(1)

    # ── Inizializza componenti ───────────────────────────────────
    try:
        components = initialize_components(cfg, cfg_obj=cfg_obj)
    except Exception as e:
        print(f"\n❌ Errore inizializzazione componenti: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ── Avvia server API (engine parte dal lifespan, stesso loop = WebSocket OK)
    ensure_port_free(args.host, args.port)
    print(f"\n🌐 Avvio server API su http://{args.host}:{args.port}...")
    print("  (Engine e WebSocket nello stesso event loop)")

    # Apri browser prima di avviare (uvicorn blocca)
    if not args.no_browser:
        url = f"http://{args.host}:{args.port}"
        def _open_browser():
            time.sleep(2)
            webbrowser.open(url)
        threading.Thread(target=_open_browser, daemon=True).start()

    try:
        run_server_with_engine(
            args.host, args.port,
            components["engine"],
            components["paper_executor"],
        )
    except KeyboardInterrupt:
        pass

    print("\n👋 AlgoTrader Pro arrestato. Arrivederci!\n")


if __name__ == "__main__":
    main()
