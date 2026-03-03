#!/usr/bin/env python3
"""
AlgoTrader Pro - Training Modelli ML
======================================
Addestra l'ensemble ML (XGBoost + Random Forest + LSTM) sui dati storici.
Usa walk-forward validation per prevenire overfitting.

Uso:
    python scripts/train_models.py
    python scripts/train_models.py --symbols BTC ETH --no-lstm
    python scripts/train_models.py --walk-forward --folds 5
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")  # Silenzia sklearn/torch warnings durante training

# Aggiungo root al path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

BANNER = """
╔══════════════════════════════════════════════════════════╗
║        AlgoTrader Pro — Training Modelli ML             ║
╚══════════════════════════════════════════════════════════╝
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Addestra ensemble ML (XGBoost + Random Forest + LSTM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  # Training completo su BTC e ETH (default)
  python scripts/train_models.py

  # Solo XGBoost e RF (salta LSTM — più veloce)
  python scripts/train_models.py --no-lstm

  # Walk-forward validation con 5 fold
  python scripts/train_models.py --walk-forward --folds 5

  # Training su tutti gli asset nella cache
  python scripts/train_models.py --symbols BTC ETH SOL

  # Force retraining anche se modelli esistono
  python scripts/train_models.py --force
        """
    )
    parser.add_argument(
        "--symbols", nargs="+",
        default=["BTC", "ETH"],
        help="Simboli su cui addestrare (default: BTC ETH)"
    )
    parser.add_argument(
        "--timeframe", type=int, default=60,
        help="Timeframe in minuti (default: 60)"
    )
    parser.add_argument(
        "--no-lstm", action="store_true",
        help="Salta training LSTM (PyTorch) — più veloce"
    )
    parser.add_argument(
        "--walk-forward", action="store_true",
        help="Esegui walk-forward validation dopo il training base"
    )
    parser.add_argument(
        "--folds", type=int, default=3,
        help="Numero di fold per walk-forward (default: 3)"
    )
    parser.add_argument(
        "--train-months", type=int, default=12,
        help="Mesi di training per ogni fold walk-forward (default: 12)"
    )
    parser.add_argument(
        "--test-months", type=int, default=2,
        help="Mesi di test per ogni fold walk-forward (default: 2)"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory output modelli (default: src/ml/models)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Forza retraining anche se modelli esistono"
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────
# Verifica prerequisiti
# ──────────────────────────────────────────────────────────────────

# Mappa timeframe (minuti) → stringa per output modelli (COMBINED_5m_*, COMBINED_1h_*)
TIMEFRAME_MAP = {1: "1m", 5: "5m", 15: "15m", 30: "15m", 60: "1h", 240: "4h", 1440: "1d"}
# Chiave store OHLCV (deve coincidere con download_historical STORE_TIMEFRAME_MAP)
STORE_TF_MAP = {1: "1", 5: "5", 15: "15", 30: "15", 60: "60", 240: "240", 1440: "D"}


def check_prerequisites(symbols: list, timeframe: int) -> tuple:
    """Carica e combina dati da tutti i simboli. Ritorna (df, ok)."""
    import pandas as pd
    from src.data.cache.ohlcv_store import OHLCVStore

    cache_dir = str(ROOT / "database" / "ohlcv_cache")
    store = OHLCVStore(cache_dir=cache_dir)
    store_tf = STORE_TF_MAP.get(timeframe, "60")
    dfs = []

    print("📊 Carico dati dalla cache...")
    for symbol in symbols:
        pair = f"{symbol}USDT"
        try:
            df = store.load(pair, store_tf)
            if df is not None and len(df) > 0:
                df["symbol"] = symbol
                dfs.append(df)
                from datetime import datetime, timezone
                t_min = datetime.fromtimestamp(df["open_time"].min() / 1000, tz=timezone.utc).date()
                t_max = datetime.fromtimestamp(df["open_time"].max() / 1000, tz=timezone.utc).date()
                print(f"  ✅ {pair}: {len(df):,} candele ({t_min} → {t_max})")
            else:
                print(f"  ⚠️  {pair}: nessun dato in cache (esegui download_historical.py)")
        except Exception as e:
            print(f"  ❌ {pair}: errore caricamento — {e}")

    if not dfs:
        return None, False

    combined = pd.concat(dfs).sort_values("open_time").reset_index(drop=True)
    print(f"\n  📈 Dataset combinato: {len(combined):,} righe totali\n")
    return combined, True


# ──────────────────────────────────────────────────────────────────
# Print metriche in tabella
# ──────────────────────────────────────────────────────────────────

def print_metrics_table(metrics: dict, title: str = "METRICHE"):
    print(f"\n{'═' * 55}")
    print(f"  {title}")
    print(f"{'═' * 55}")

    rows = [
        ("Accuracy", metrics.get("accuracy", 0), "%", 100),
        ("Precision (Long)", metrics.get("precision_long", 0), "%", 100),
        ("Recall (Long)", metrics.get("recall_long", 0), "%", 100),
        ("F1-Score", metrics.get("f1", 0), "", 1),
        ("ROC-AUC", metrics.get("roc_auc", 0), "", 1),
        ("Win Rate stimato", metrics.get("win_rate", 0), "%", 100),
        ("Kelly Fraction", metrics.get("kelly_fraction", 0), "%", 100),
    ]

    for name, value, unit, scale in rows:
        if value is None:
            continue
        # Normalizza a 0-1 per la barra
        v_norm = value / scale if scale > 1 else value
        bar_len = 20
        filled = int(bar_len * min(v_norm, 1.0))
        bar = "█" * filled + "░" * (bar_len - filled)

        if unit == "%":
            val_str = f"{value*100:.1f}%"
        else:
            val_str = f"{value:.4f}"

        color = "✅" if v_norm >= 0.6 else ("⚠️ " if v_norm >= 0.5 else "❌")
        print(f"  {color} {name:<22} [{bar}] {val_str}")

    print(f"{'═' * 55}")


# ──────────────────────────────────────────────────────────────────
# Training pipeline principale
# ──────────────────────────────────────────────────────────────────

def run_training(df, args, models_dir: Path) -> dict:
    """Esegue il training completo e ritorna metriche."""
    from src.data.pipeline.feature_engineer import FeatureEngineer
    from src.ml.training.trainer import ModelTrainer

    # ── Feature engineering ──────────────────────────────────────
    print("🔧 Feature Engineering...")
    engineer = FeatureEngineer()
    try:
        features_df = engineer.compute_features(df)
        # Rimuovi colonna symbol (non numerica) per il training
        if "symbol" in features_df.columns:
            features_df = features_df.drop(columns=["symbol"])
        print(f"  ✅ {len(features_df.columns)} feature calcolate su {len(features_df):,} campioni")
    except Exception as e:
        print(f"  ❌ Errore feature engineering: {e}")
        raise

    # ── Training ─────────────────────────────────────────────────
    print("\n🏋️  Avvio training ensemble...")
    trainer = ModelTrainer(config={"use_lstm": not args.no_lstm})

    t_start = time.time()
    # Train su dati combinati (symbol usato per naming)
    report = trainer.train_all(
        features_df,
        symbol="COMBINED",
        timeframe=TIMEFRAME_MAP.get(args.timeframe, "1h"),
    )
    elapsed = time.time() - t_start

    print(f"\n  ⏱️  Training completato in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    # Estrai metriche per print_metrics_table (usa XGBoost test come riferimento)
    xgb_test = report.get("xgboost", {}).get("test_metrics", {})
    metrics = {
        "accuracy": xgb_test.get("accuracy"),
        "precision_long": xgb_test.get("precision"),
        "recall_long": xgb_test.get("recall"),
        "f1": xgb_test.get("f1"),
        "roc_auc": xgb_test.get("roc_auc"),
    }
    return metrics


# ──────────────────────────────────────────────────────────────────
# Walk-forward validation
# ──────────────────────────────────────────────────────────────────

def run_walk_forward(df, args, models_dir: Path) -> list:
    """Esegue walk-forward validation e ritorna metriche per fold."""
    from src.data.pipeline.feature_engineer import FeatureEngineer
    from src.ml.training.walk_forward import WalkForwardValidator

    print("\n🔄 Walk-Forward Validation...")
    engineer = FeatureEngineer()
    features_df = engineer.compute_features(df)

    validator = WalkForwardValidator(
        train_months=args.train_months,
        test_months=args.test_months,
        n_folds=args.folds,
        models_dir=str(models_dir),
        use_lstm=not args.no_lstm,
    )

    fold_results = validator.run(features_df)
    return fold_results


def print_walk_forward_summary(fold_results: list):
    print(f"\n{'═' * 65}")
    print("  WALK-FORWARD VALIDATION — RISULTATI PER FOLD")
    print(f"{'═' * 65}")
    print(f"  {'Fold':<6} {'Periodo':<24} {'Accuracy':>9} {'ROC-AUC':>8} {'F1':>7} {'Win Rate':>9}")
    print("  " + "─" * 60)

    accuracies = []
    aucs = []
    f1s = []

    for fold in fold_results:
        fold_num = fold.get("fold", "?")
        period = fold.get("period", "—")
        acc = fold.get("accuracy", 0)
        auc = fold.get("roc_auc", 0)
        f1 = fold.get("f1", 0)
        wr = fold.get("win_rate", 0)
        ok = "✅" if acc >= 0.55 else "⚠️ "
        print(f"  {ok} {fold_num:<5} {period:<24} {acc*100:>8.1f}% {auc:>8.4f} {f1:>7.4f} {wr*100:>8.1f}%")
        accuracies.append(acc)
        aucs.append(auc)
        f1s.append(f1)

    if accuracies:
        import statistics
        print("  " + "─" * 60)
        print(f"  {'MEDIA':<6} {'':<24} {statistics.mean(accuracies)*100:>8.1f}% "
              f"{statistics.mean(aucs):>8.4f} {statistics.mean(f1s):>7.4f}")
        print(f"  {'STD':<6} {'':<24} {statistics.stdev(accuracies)*100:>8.1f}% "
              f"{statistics.stdev(aucs):>8.4f} {statistics.stdev(f1s):>7.4f}" if len(accuracies) > 1 else "")

    print(f"{'═' * 65}")


# ──────────────────────────────────────────────────────────────────
# Salva report training nel DB
# ──────────────────────────────────────────────────────────────────

def save_training_report(metrics: dict, fold_results: list, symbols: list):
    """Salva i risultati del training nel database."""
    try:
        from src.utils.db import get_session
        from datetime import datetime, timezone
        import json

        with get_session() as session:
            session.execute(
                """
                INSERT INTO ml_model_runs
                    (model_name, version, trained_at, symbols, metrics_json, fold_metrics_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    "ensemble_v1",
                    datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
                    datetime.now(timezone.utc).isoformat(),
                    ",".join(symbols),
                    json.dumps(metrics),
                    json.dumps(fold_results),
                )
            )
    except Exception as e:
        pass  # DB non critico per il training


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main():
    print(BANNER)
    args = parse_args()

    print(f"  Simboli:       {', '.join(args.symbols)}")
    print(f"  Timeframe:     {args.timeframe}m")
    print(f"  LSTM:          {'No' if args.no_lstm else 'Sì'}")
    print(f"  Walk-forward:  {'Sì (' + str(args.folds) + ' fold)' if args.walk_forward else 'No'}")
    print()

    # ── Init ─────────────────────────────────────────────────────
    try:
        from src.utils.db import init_db
        init_db()
    except Exception as e:
        print(f"  ⚠️  DB: {e}")

    models_dir = Path(args.output_dir) if args.output_dir else ROOT / "src" / "ml" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output dir:    {models_dir}\n")

    # Controlla se modelli esistono già
    if not args.force:
        existing = list(models_dir.glob("*.pkl")) + list(models_dir.glob("*.pt")) + list(models_dir.glob("*.json"))
        if existing:
            print(f"⚠️  Trovati {len(existing)} file modello esistenti.")
            print("   Usa --force per forzare il retraining, oppure continuo con walk-forward.\n")
            if not args.walk_forward:
                print("✅ Modelli già presenti. Usa --force per retraining.\n")
                print("→ Prossimo passo: python scripts/run_backtest.py\n")
                return

    # ── Carico dati ──────────────────────────────────────────────
    df, ok = check_prerequisites(args.symbols, args.timeframe)
    if not ok:
        print("\n❌ Dati insufficienti. Esegui prima:")
        print("   python scripts/download_historical.py\n")
        sys.exit(1)

    total_start = time.time()

    # ── Training base ─────────────────────────────────────────────
    print("─" * 55)
    print("FASE 1: Training Ensemble Base")
    print("─" * 55)
    metrics = {}
    try:
        metrics = run_training(df, args, models_dir)
        print_metrics_table(metrics, "METRICHE TRAINING BASE")
    except Exception as e:
        print(f"\n❌ Errore training: {e}")
        import traceback
        traceback.print_exc()
        # Non uscire — prova comunque walk-forward se richiesto

    # ── Walk-forward (opzionale) ──────────────────────────────────
    fold_results = []
    if args.walk_forward:
        print("\n" + "─" * 55)
        print("FASE 2: Walk-Forward Validation")
        print("─" * 55)
        try:
            fold_results = run_walk_forward(df, args, models_dir)
            print_walk_forward_summary(fold_results)
        except Exception as e:
            print(f"\n⚠️  Walk-forward fallito: {e}")

    # ── Salva nel DB ─────────────────────────────────────────────
    if metrics:
        save_training_report(metrics, fold_results, args.symbols)

    # ── Riepilogo finale ─────────────────────────────────────────
    total_elapsed = time.time() - total_start
    print(f"\n{'═' * 55}")
    print(f"  TRAINING COMPLETATO in {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
    print(f"  Modelli salvati in: {models_dir}")
    print(f"{'═' * 55}")
    print(f"\n→ Prossimo passo: python scripts/run_backtest.py\n")


if __name__ == "__main__":
    main()
