#!/usr/bin/env python3
"""
AlgoTrader Pro - Backtest Engine CLI
======================================
Esegue il backtest sulla strategia ML con dati storici.
Produce report HTML con equity curve + Monte Carlo.

Uso:
    python scripts/run_backtest.py
    python scripts/run_backtest.py --symbol BTC --days 365
    python scripts/run_backtest.py --report --open-browser
"""

import argparse
import asyncio
import json
import random
import sys
import time
import types
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Aggiungo root al path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

BANNER = """
╔══════════════════════════════════════════════════════════╗
║        AlgoTrader Pro — Backtest Engine                 ║
╚══════════════════════════════════════════════════════════╝
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Esegui backtest della strategia ML su dati storici",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  python scripts/run_backtest.py
  python scripts/run_backtest.py --symbol ETH --days 180
  python scripts/run_backtest.py --days 730 --monte-carlo --report
        """
    )
    parser.add_argument("--symbol", default="BTC", help="Simbolo (default: BTC)")
    parser.add_argument("--timeframe", type=int, default=60, help="Timeframe minuti (default: 60)")
    parser.add_argument("--days", type=int, default=365, help="Giorni storici (default: 365)")
    parser.add_argument("--capital", type=float, default=10000.0, help="Capitale iniziale € (default: 10000)")
    parser.add_argument("--confidence-long", type=float, default=65.0, help="Soglia LONG (default: 65)")
    parser.add_argument("--confidence-short", type=float, default=70.0, help="Soglia SHORT (default: 70)")
    parser.add_argument("--monte-carlo", action="store_true", help="Esegui Monte Carlo")
    parser.add_argument("--mc-sims", type=int, default=1000, help="Simulazioni MC (default: 1000)")
    parser.add_argument("--report", action="store_true", help="Genera report HTML")
    parser.add_argument("--output-dir", default="results", help="Directory output (default: results/)")
    parser.add_argument("--open-browser", action="store_true", help="Apri report nel browser")
    return parser.parse_args()


# Mappa timeframe (minuti) → stringa
TIMEFRAME_MAP = {1: "1m", 5: "5m", 15: "15m", 30: "15m", 60: "1h", 240: "4h", 1440: "1d"}


def _build_config(args) -> object:
    """Costruisce config object per aggregator/risk/strategy."""
    def _ns(**kw):
        ns = types.SimpleNamespace()
        for k, v in kw.items():
            if isinstance(v, dict):
                setattr(ns, k, _ns(**v))
            else:
                setattr(ns, k, v)
        return ns

    return _ns(
        strategy=_ns(
            layer_weights=_ns(
                layer1_technical=0.30,
                layer2_volume=0.25,
                layer3_sentiment=0.15,
                layer4_ml=0.30,
            ),
            min_confidence_long=args.confidence_long,
            min_confidence_short=args.confidence_short,
            allow_short=False,
            max_concurrent_positions=3,
            max_portfolio_risk_pct=20.0,
        ),
        risk=_ns(
            fee_rate=0.001,
            slippage_std=0.0005,
            kelly_fraction_multiplier=0.5,
            kelly_min_fraction=0.01,
            kelly_max_fraction=0.15,
            atr_stop_multiplier=2.0,
            atr_high_volatility_multiplier=3.0,
            volatility_threshold_pct=5.0,
            take_profit=_ns(risk_reward_ratio=2.5),
        ),
    )


def load_data(symbol: str, timeframe: int, days: int):
    """Carica dati dalla cache Parquet."""
    from src.data.cache.ohlcv_store import OHLCVStore

    pair = f"{symbol}USDT"
    tf_str = TIMEFRAME_MAP.get(timeframe, "1h")
    store = OHLCVStore()

    print(f"📊 Carico dati {pair} ({timeframe}m, {days}d)...")
    df = store.load(pair, tf_str)

    if df is None or len(df) == 0:
        print(f"\n❌ Nessun dato per {pair}. Esegui: python scripts/download_historical.py --symbols {symbol}\n")
        return None

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    cutoff_ms = int(cutoff.timestamp() * 1000)
    df = df[df["open_time"] >= cutoff_ms].reset_index(drop=True)

    if len(df) == 0:
        print(f"\n❌ Nessun dato nel periodo richiesto.\n")
        return None

    t_min = datetime.fromtimestamp(df["open_time"].min() / 1000, tz=timezone.utc).date()
    t_max = datetime.fromtimestamp(df["open_time"].max() / 1000, tz=timezone.utc).date()
    print(f"  ✅ {len(df):,} candele ({t_min} → {t_max})")
    return df


def load_ml_models(args):
    """Carica modelli ML dall'ultima run COMBINED_1h_*."""
    models_dir = ROOT / "src" / "ml" / "models"
    tf_str = TIMEFRAME_MAP.get(args.timeframe, "1h")
    prefix = "COMBINED_" + tf_str + "_"

    candidates = sorted(
        [d for d in models_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)],
        key=lambda d: d.name,
    )
    if not candidates:
        return None

    save_dir = candidates[-1]
    try:
        import joblib
        from src.ml.ensemble.ensemble_combiner import EnsembleCombiner
        from src.ml.ensemble.xgboost_model import XGBoostModel
        from src.ml.ensemble.random_forest_model import RandomForestModel
        from sklearn.preprocessing import StandardScaler

        meta = joblib.load(str(save_dir / "meta.joblib"))
        scaler = joblib.load(str(save_dir / "scaler.joblib"))
        feature_cols = meta["feature_cols"]

        xgb = XGBoostModel()
        xgb.load(str(save_dir / "xgboost.ubj"), feature_names=feature_cols)

        rf = RandomForestModel()
        rf.load(str(save_dir / "random_forest.joblib"))

        combiner = EnsembleCombiner()
        combiner.load(str(save_dir / "ensemble_combiner.joblib"))

        lstm = None
        if (save_dir / "lstm.pt").exists():
            from src.ml.ensemble.lstm_model import LSTMTrainer
            lstm = LSTMTrainer()
            lstm.load(str(save_dir / "lstm.pt"), input_size=len(feature_cols))

        return {
            "scaler": scaler,
            "xgb": xgb,
            "rf": rf,
            "lstm": lstm,
            "combiner": combiner,
            "feature_cols": feature_cols,
            "meta": meta,
        }
    except Exception as e:
        print(f"  ⚠️  Errore caricamento modelli: {e}")
        return None


def run_simple_backtest(features_df, df, models, args):
    """
    Backtest semplificato: usa ML confidence per segnali LONG.
    Simula trade con stop-loss e take-profit.
    """
    import numpy as np

    capital = args.capital
    conf_long = args.confidence_long / 100.0
    conf_short = args.confidence_short / 100.0
    fee_rate = 0.001
    slippage_std = 0.0005
    stop_loss_pct = 0.02
    take_profit_pct = 0.04

    feature_cols = models["feature_cols"]
    scaler = models["scaler"]
    xgb = models["xgb"]
    rf = models["rf"]
    lstm = models["lstm"]
    combiner = models["combiner"]

    # Allinea features_df e df per indice (stesso numero di righe dopo dropna)
    # features_df ha meno righe per dropna - usiamo merge su open_time
    if "open_time" not in features_df.columns and "open_time" in df.columns:
        # features_df potrebbe avere indice diverso - prendiamo le righe in comune
        pass

    # Assumiamo features_df e df allineati per le ultime N righe
    # features_df viene da compute_features(df) che fa dropna - quindi meno righe
    # Dobbiamo iterare su df e per ogni riga trovare la prediction
    # La soluzione più semplice: usare le righe di features_df che hanno open_time
    # e fare join con df su open_time

    # Merge features con OHLCV (suffissi per evitare duplicati)
    ohlcv = df[["open_time", "open", "high", "low", "close", "volume"]].rename(
        columns={"open": "o", "high": "h", "low": "l", "close": "c", "volume": "v"}
    )
    if "open_time" in features_df.columns:
        merged = features_df.merge(ohlcv, on="open_time", how="inner")
    else:
        merged = features_df.copy()
        merged["open_time"] = df["open_time"].iloc[: len(features_df)].values
        merged = merged.merge(ohlcv, on="open_time", how="inner")

    # Usa esattamente le colonne del modello (riempi con 0 se mancanti)
    X = np.zeros((len(merged), len(feature_cols)), dtype=np.float32)
    for j, col in enumerate(feature_cols):
        if col in merged.columns:
            X[:, j] = merged[col].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X_scaled = scaler.transform(X)

    # Predict con ensemble
    xgb_proba = xgb.predict_proba(X_scaled)
    rf_proba = rf.predict_proba(X_scaled)
    if lstm is not None:
        from src.ml.training.trainer import ModelTrainer
        meta_cfg = models["meta"].get("config") or {}
        lookback = meta_cfg.get("lookback", 60) if isinstance(meta_cfg, dict) else 60
        try:
            X_seq, _ = ModelTrainer.prepare_sequences(X_scaled, None, lookback)
            lstm_proba = lstm.predict_proba(X_seq)
            # Allinea lunghezze - lstm ha meno righe
            n_lstm = len(lstm_proba)
            lstm_proba_full = np.full(len(X_scaled), 0.5, dtype=np.float64)
            lstm_proba_full[-n_lstm:] = lstm_proba
        except Exception:
            lstm_proba_full = np.full(len(X_scaled), 0.5, dtype=np.float64)
    else:
        lstm_proba_full = np.full(len(X_scaled), 0.5, dtype=np.float64)

    ml_confidence = combiner.predict(xgb_proba, rf_proba, lstm_proba_full)  # 0-100
    ml_confidence = np.clip(ml_confidence, 0, 100)

    # Backtest loop
    equity = capital
    cash = capital
    peak_equity = capital
    position = None
    equity_curve = []
    trades = []

    for i in range(len(merged)):
        row = merged.iloc[i]
        ts_ms = row["open_time"]
        ts_str = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat()
        close = float(row["c"])
        high = float(row["h"])
        low = float(row["l"])
        conf = float(ml_confidence[i]) / 100.0  # 0-1

        # Check exit
        if position is not None:
            exit_price = None
            exit_reason = None
            if position["direction"] == "LONG":
                if low <= position["stop_loss"]:
                    exit_price = position["stop_loss"]
                    exit_reason = "stop_loss"
                elif high >= position["take_profit"]:
                    exit_price = position["take_profit"]
                    exit_reason = "take_profit"
            if exit_price is not None:
                slip = random.gauss(0, slippage_std)
                exit_price *= (1 - abs(slip))
                pnl = position["quantity"] * (exit_price - position["entry_price"]) - position["entry_fee"] - position["quantity"] * exit_price * fee_rate
                cash += position["size_usd"] + pnl
                equity = cash
                trades.append({
                    "entry_time": position["entry_time"],
                    "exit_time": ts_str,
                    "direction": position["direction"],
                    "entry_price": position["entry_price"],
                    "exit_price": exit_price,
                    "quantity": position["quantity"],
                    "pnl": pnl,
                    "pnl_pct": (pnl / position["size_usd"]) * 100,
                    "exit_reason": exit_reason,
                })
                position = None

        # Mark-to-market
        if position is not None:
            unreal = position["quantity"] * (close - position["entry_price"])
            equity = cash + position["size_usd"] + unreal
        else:
            equity = cash

        if equity > peak_equity:
            peak_equity = equity
        dd = (equity - peak_equity) / peak_equity * 100 if peak_equity > 0 else 0
        equity_curve.append({"timestamp": ts_str, "equity": round(equity, 4), "drawdown_pct": round(dd, 4)})

        # Entry
        if position is None and conf >= conf_long:
            size_usd = min(equity * 0.1, cash * 0.95)
            if size_usd > 10:
                slip = random.gauss(0, slippage_std)
                fill_price = close * (1 + abs(slip))
                qty = size_usd / fill_price
                fee = size_usd * fee_rate
                cash -= size_usd + fee
                position = {
                    "direction": "LONG",
                    "entry_price": fill_price,
                    "quantity": qty,
                    "size_usd": size_usd,
                    "entry_fee": fee,
                    "entry_time": ts_str,
                    "stop_loss": fill_price * (1 - stop_loss_pct),
                    "take_profit": fill_price * (1 + take_profit_pct),
                }

    # Close position at end
    if position is not None:
        last_row = merged.iloc[-1]
        last_close = float(last_row["c"])
        last_ts = datetime.fromtimestamp(last_row["open_time"] / 1000, tz=timezone.utc).isoformat()
        pnl = position["quantity"] * (last_close - position["entry_price"]) - position["entry_fee"] - position["quantity"] * last_close * fee_rate
        cash += position["size_usd"] + pnl
        trades.append({
            "entry_time": position["entry_time"],
            "exit_time": last_ts,
            "direction": position["direction"],
            "entry_price": position["entry_price"],
            "exit_price": last_close,
            "quantity": position["quantity"],
            "pnl": pnl,
            "pnl_pct": (pnl / position["size_usd"]) * 100,
            "exit_reason": "end_of_data",
        })

    # Metrics
    returns = np.diff([e["equity"] for e in equity_curve]) / np.array([e["equity"] for e in equity_curve[:-1]])
    sharpe = float(returns.mean() / returns.std() * (8760 ** 0.5)) if len(returns) > 1 and returns.std() > 0 else 0
    equities = [e["equity"] for e in equity_curve]
    peak = equities[0]
    max_dd = 0
    for eq in equities:
        peak = max(peak, eq)
        dd = (eq - peak) / peak * 100 if peak > 0 else 0
        max_dd = min(max_dd, dd)

    total_return = (equity - capital) / capital * 100
    n_years = max(len(equity_curve) / 8760, 1 / 365)
    cagr = ((equity / capital) ** (1 / n_years) - 1) * 100 if capital > 0 else 0

    metrics = {
        "sharpe_ratio": sharpe,
        "total_return_pct": total_return,
        "max_drawdown": max_dd,
        "cagr": cagr,
        "calmar_ratio": abs(cagr / max_dd) if max_dd != 0 else 0,
    }

    return {
        "equity_curve": equity_curve,
        "trades": trades,
        "metrics": metrics,
        "first_price": float(df.iloc[0]["close"]),
        "last_price": float(df.iloc[-1]["close"]),
        "start_equity": capital,
        "symbol": "N/A",  # ReportGenerator fallback
        "timeframe": "N/A",
    }


def print_backtest_results(result: dict, symbol: str, capital: float):
    """Stampa risultati in formato tabellare."""
    equity_curve = result.get("equity_curve", [])
    trades = result.get("trades", [])
    metrics = result.get("metrics", {})

    final_equity = equity_curve[-1]["equity"] if equity_curve else capital
    total_return = (final_equity / capital - 1) * 100
    n_trades = len(trades)
    wins = [t for t in trades if t.get("pnl", 0) > 0]
    losses = [t for t in trades if t.get("pnl", 0) <= 0]
    win_rate = len(wins) / max(n_trades, 1) * 100
    avg_win = sum(t["pnl"] for t in wins) / max(len(wins), 1)
    avg_loss = abs(sum(t["pnl"] for t in losses)) / max(len(losses), 1)
    profit_factor = sum(t["pnl"] for t in wins) / max(abs(sum(t["pnl"] for t in losses)), 0.01)

    print(f"\n{'═' * 65}")
    print(f"  RISULTATI BACKTEST — {symbol}USDT | {len(equity_curve)} candele")
    print(f"{'═' * 65}")
    print(f"\n  💰 RENDIMENTO")
    icon = "🟢" if total_return > 0 else "🔴"
    print(f"  {icon} Capitale iniziale:   {capital:>12,.2f} €")
    print(f"  {icon} Capitale finale:     {final_equity:>12,.2f} €")
    print(f"  {icon} Rendimento totale:   {total_return:>11.2f}%")

    first = result.get("first_price", 0)
    last = result.get("last_price", 0)
    if first and last:
        bh_return = (last / first - 1) * 100
        alpha = total_return - bh_return
        print(f"     Buy & Hold {symbol}:     {bh_return:>11.2f}%")
        print(f"  {'🟢' if alpha > 0 else '🔴'} Alpha vs B&H:        {alpha:>11.2f}%")

    print(f"\n  📊 STATISTICHE TRADE")
    print(f"     Trade totali:        {n_trades:>12,}")
    print(f"     Trade vincenti:      {len(wins):>12,}")
    print(f"     Trade perdenti:      {len(losses):>12,}")
    print(f"  {'🟢' if win_rate >= 45 else '🔴'} Win Rate:            {win_rate:>11.1f}%")
    print(f"     Media vincita:       {avg_win:>12.2f} €")
    print(f"     Media perdita:       {avg_loss:>12.2f} €")
    rr = avg_win / max(avg_loss, 0.01)
    print(f"  {'🟢' if rr >= 1.5 else '🔴'} Risk/Reward ratio:   {rr:>12.2f}x")
    print(f"  {'🟢' if profit_factor >= 1.3 else '🔴'} Profit Factor:       {profit_factor:>12.2f}")

    print(f"\n  📉 RISK METRICS")
    sharpe = metrics.get("sharpe_ratio", 0)
    max_dd = metrics.get("max_drawdown", 0)
    print(f"  {'🟢' if sharpe >= 1 else '🔴'} Sharpe Ratio:        {sharpe:>12.3f}")
    print(f"  {'🟢' if abs(max_dd) <= 15 else '🔴'} Max Drawdown:        {max_dd:>11.2f}%")
    print(f"{'═' * 65}")


def print_monte_carlo_results(mc_result, capital: float):
    """Stampa risultati Monte Carlo (accetta dict o MonteCarloResult)."""
    if hasattr(mc_result, "to_dict"):
        mc_result = mc_result.to_dict()
    print(f"\n{'═' * 65}")
    print(f"  MONTE CARLO — {mc_result.get('n_simulations', 1000)} Simulazioni")
    print(f"{'═' * 65}")
    pct = mc_result.get("final_equity_percentiles", {})
    if not pct:
        print("  ⚠️  Nessun risultato")
        return
    # Percentili come equity finale
    p5 = pct.get("5", capital)
    p50 = pct.get("50", capital)
    p95 = pct.get("95", capital)
    print(f"\n  DISTRIBUZIONE CAPITALE FINALE:")
    print(f"     5° percentile:   €{p5:>12,.0f}  ({(p5/capital-1)*100:+.1f}%)")
    print(f"    50° percentile:   €{p50:>12,.0f}  ({(p50/capital-1)*100:+.1f}%)")
    print(f"    95° percentile:   €{p95:>12,.0f}  ({(p95/capital-1)*100:+.1f}%)")
    prob_profit = mc_result.get("probability_of_profit", 0)
    prob_ruin = mc_result.get("probability_of_ruin", 0)
    print(f"\n  {'🟢' if prob_profit >= 60 else '🔴'} Probabilità profitto: {prob_profit:>6.1f}%")
    print(f"  {'🟢' if prob_ruin <= 5 else '🔴'} Probabilità rovina:   {prob_ruin:>6.1f}%")
    print(f"{'═' * 65}")


def main():
    print(BANNER)
    args = parse_args()

    print(f"  Simbolo:       {args.symbol}USDT")
    print(f"  Timeframe:     {args.timeframe}m")
    print(f"  Periodo:       {args.days} giorni")
    print(f"  Capitale:      €{args.capital:,.2f}")
    print(f"  Soglia LONG:   {args.confidence_long}%")
    print(f"  Monte Carlo:   {'Sì' if args.monte_carlo else 'No'}")
    print()

    try:
        from src.utils.db import init_db
        init_db()
    except Exception as e:
        print(f"  ⚠️  DB: {e}")

    df = load_data(args.symbol, args.timeframe, args.days)
    if df is None:
        sys.exit(1)

    print("\n🔧 Feature Engineering...")
    from src.data.pipeline.feature_engineer import FeatureEngineer
    engineer = FeatureEngineer()
    features_df = engineer.compute_features(df)
    if "symbol" in features_df.columns:
        features_df = features_df.drop(columns=["symbol"])
    print(f"  ✅ {len(features_df.columns)} feature calcolate")

    models = load_ml_models(args)
    if models is None:
        print("  ⚠️  Modelli ML non trovati — backtest senza ML")
        sys.exit(1)
    print("  ✅ ML Ensemble caricato")

    print(f"\n▶️  Avvio backtest ({len(features_df):,} candele)...")
    t_start = time.time()
    result = run_simple_backtest(features_df, df, models, args)
    print(f"  ✅ Backtest completato in {time.time()-t_start:.1f}s")

    print_backtest_results(result, args.symbol, args.capital)

    mc_result = None
    if args.monte_carlo and result.get("trades"):
        print(f"\n🎲 Monte Carlo ({args.mc_sims} simulazioni)...")
        from src.backtesting.monte_carlo import MonteCarloSimulator
        mc = MonteCarloSimulator(n_simulations=args.mc_sims)
        mc_result = mc.run(result["trades"], args.capital)
        print_monte_carlo_results(mc_result, args.capital)

    if args.report:
        print(f"\n📄 Generazione report HTML...")
        output_dir = ROOT / args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / f"backtest_{args.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        try:
            from src.backtesting.report_generator import ReportGenerator
            # ReportGenerator usa getattr(bt, ...) - serve un oggetto
            bt_obj = types.SimpleNamespace(
                equity_curve=result.get("equity_curve", []),
                trades=result.get("trades", []),
                metrics={**result.get("metrics", {}), "symbol": args.symbol + "USDT", "timeframe": f"{args.timeframe}m"},
                start_equity=args.capital,
            )
            mc_obj = mc_result if mc_result and hasattr(mc_result, "to_dict") else types.SimpleNamespace(
                to_dict=lambda: {},
                final_equity_percentiles={},
                probability_of_profit=0,
                probability_of_ruin=0,
                sample_paths=[],
                n_simulations=0,
                initial_capital=args.capital,
            )
            ReportGenerator().generate_html_report(
                backtest_result=bt_obj,
                monte_carlo_result=mc_obj,
                output_path=str(report_path),
            )
            print(f"  ✅ Report: {report_path}")
            if args.open_browser:
                import webbrowser
                webbrowser.open(f"file://{report_path.absolute()}")
        except Exception as e:
            print(f"  ⚠️  Report: {e}")

    print(f"\n→ Prossimo passo: python scripts/start_bot.py\n")


if __name__ == "__main__":
    main()
