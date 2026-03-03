"""
Training orchestrator for the full ML ensemble pipeline.

Responsibilities
----------------
1. Accept a feature DataFrame and produce a labelled, scaled dataset.
2. Train XGBoost, Random Forest, and LSTM models.
3. Generate out-of-fold (OOF) predictions with TimeSeriesSplit(5).
4. Train the EnsembleCombiner meta-learner on those OOF predictions.
5. Persist every artefact to disk under
   ``src/ml/models/{symbol}_{timeframe}_{timestamp}/``.
6. Log all metrics to the ``ml_model_runs`` database table.
7. Return a rich training-report dictionary.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

# ---- internal imports -------------------------------------------------------
# These are resolved relative to the package root; adjust if your PYTHONPATH
# differs from the repo root.
from src.ml.ensemble.ensemble_combiner import EnsembleCombiner
from src.ml.ensemble.lstm_model import LSTMTrainer
from src.ml.ensemble.random_forest_model import RandomForestModel
from src.ml.ensemble.xgboost_model import XGBoostModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Repository-root detection helper
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent  # src/ml/training/
_REPO_ROOT = _HERE.parent.parent.parent   # algo-trader-pro/


def _models_base_dir() -> Path:
    return _REPO_ROOT / "src" / "ml" / "models"


def _db_path() -> Path:
    return _REPO_ROOT / "data" / "trading.db"


# ===========================================================================
# ModelTrainer
# ===========================================================================


class ModelTrainer:
    """
    Orchestrates end-to-end model training for a given symbol / timeframe.

    Parameters
    ----------
    config : dict, optional
        Overrides for default hyper-parameters.  Supported keys:

        * ``lookback``            (int,   default 60)
        * ``test_size``           (float, default 0.30)
        * ``n_cv_splits``         (int,   default 5)
        * ``lstm_epochs``         (int,   default 50)
        * ``lstm_batch_size``     (int,   default 64)
        * ``lstm_lr``             (float, default 0.001)
        * ``db_logging``          (bool,  default True)
        * ``target_column``       (str,   default 'target')
        * ``forward_returns_col`` (str,   default 'future_return_1')
        * ``bullish_threshold``   (float, default 0.0)
    """

    _DEFAULTS: Dict[str, Any] = {
        "lookback": 60,
        "test_size": 0.30,
        "n_cv_splits": 5,
        "lstm_epochs": 50,
        "lstm_batch_size": 64,
        "lstm_lr": 0.001,
        "use_lstm": True,
        "db_logging": True,
        "target_column": "target",
        "forward_returns_col": "future_return_1",
        "bullish_threshold": 0.0,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = {**self._DEFAULTS, **(config or {})}
        logger.info("ModelTrainer initialised with config: %s", self.config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_all(
        self,
        features_df: pd.DataFrame,
        symbol: str,
        timeframe: str,
    ) -> Dict[str, Any]:
        """
        Run the complete training pipeline.

        Parameters
        ----------
        features_df : pd.DataFrame
            Feature matrix.  Must contain all feature columns plus either a
            pre-computed target column (name given by ``config['target_column']``)
            or a forward-return column from which a target is derived.
        symbol : str
            Ticker symbol, e.g. ``'BTCUSDT'``.
        timeframe : str
            Bar interval, e.g. ``'1h'``.

        Returns
        -------
        dict
            Training report containing per-model metrics, OOF metrics,
            feature importance, save directory, and timestamp.
        """
        run_ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        save_dir = _models_base_dir() / f"{symbol}_{timeframe}_{run_ts}"
        save_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Artefacts will be saved to: %s", save_dir)

        # ----------------------------------------------------------------
        # 1. Create / validate target variable
        # ----------------------------------------------------------------
        df = features_df.copy()
        df = self._create_target(df)
        df = df.dropna()

        # ----------------------------------------------------------------
        # 2. Separate features from target
        # ----------------------------------------------------------------
        target_col = self.config["target_column"]
        fwd_col = self.config["forward_returns_col"]
        # Exclude target and any columns used to derive it (avoid leakage)
        exclude = {target_col, fwd_col, "symbol"}
        feature_cols = [c for c in df.columns if c not in exclude]
        X_full = df[feature_cols].values.astype(np.float32)
        y_full = df[target_col].values.astype(np.int32)

        # ----------------------------------------------------------------
        # 3. Chronological 70/30 split (no shuffle — time series)
        # ----------------------------------------------------------------
        split_idx = int(len(X_full) * (1.0 - self.config["test_size"]))
        X_train_raw, X_test_raw = X_full[:split_idx], X_full[split_idx:]
        y_train, y_test = y_full[:split_idx], y_full[split_idx:]
        logger.info(
            "Split — train: %d samples, test: %d samples",
            len(y_train),
            len(y_test),
        )

        # ----------------------------------------------------------------
        # 4. Scale features: fit on train only
        # ----------------------------------------------------------------
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        scaler_path = str(save_dir / "scaler.joblib")
        joblib.dump(scaler, scaler_path)
        logger.info("Scaler saved to %s", scaler_path)

        # ----------------------------------------------------------------
        # 5. Train XGBoost
        # ----------------------------------------------------------------
        logger.info("--- Training XGBoost ---")
        xgb_model = XGBoostModel()
        xgb_metrics = xgb_model.train(X_train, y_train, feature_names=feature_cols)
        xgb_test_metrics = self._evaluate_on_test(xgb_model, X_test, y_test, "XGBoost")
        xgb_model.save(str(save_dir / "xgboost.ubj"))

        # ----------------------------------------------------------------
        # 6. Train Random Forest
        # ----------------------------------------------------------------
        logger.info("--- Training Random Forest ---")
        rf_model = RandomForestModel()
        rf_metrics = rf_model.train(X_train, y_train, feature_names=feature_cols)
        rf_test_metrics = self._evaluate_on_test(rf_model, X_test, y_test, "RandomForest")
        rf_model.save(str(save_dir / "random_forest.joblib"))

        use_lstm = self.config.get("use_lstm", True)
        lookback = self.config["lookback"]
        lstm_trainer = None
        lstm_metrics = {}
        lstm_test_metrics = {}

        if use_lstm:
            # ----------------------------------------------------------------
            # 7. Prepare LSTM sequences from scaled data
            # ----------------------------------------------------------------
            logger.info("--- Preparing LSTM sequences (lookback=%d) ---", lookback)

            X_full_scaled = scaler.transform(X_full)
            X_seq_full, y_seq_full = self.prepare_sequences(X_full_scaled, y_full, lookback)

            seq_split_idx = int(len(X_seq_full) * (1.0 - self.config["test_size"]))
            X_seq_train = X_seq_full[:seq_split_idx]
            y_seq_train = y_seq_full[:seq_split_idx]
            X_seq_test = X_seq_full[seq_split_idx:]
            y_seq_test = y_seq_full[seq_split_idx:]

            # ----------------------------------------------------------------
            # 8. Train LSTM
            # ----------------------------------------------------------------
            logger.info("--- Training LSTM ---")
            lstm_trainer = LSTMTrainer()
            lstm_metrics = lstm_trainer.train(
                X_seq_train,
                y_seq_train,
                epochs=self.config["lstm_epochs"],
                batch_size=self.config["lstm_batch_size"],
                lr=self.config["lstm_lr"],
            )
            lstm_test_metrics = self._evaluate_lstm_on_test(
                lstm_trainer, X_seq_test, y_seq_test
            )
            lstm_trainer.save(str(save_dir / "lstm.pt"))

        # ----------------------------------------------------------------
        # 9. Generate OOF predictions with TimeSeriesSplit(5)
        # ----------------------------------------------------------------
        logger.info("--- Generating OOF predictions for meta-learner ---")
        xgb_oof, rf_oof, lstm_oof, y_oof = self._generate_oof_predictions(
            X_train, y_train, lookback, scaler, feature_cols, use_lstm=use_lstm
        )

        # ----------------------------------------------------------------
        # 10. Train EnsembleCombiner meta-learner
        # ----------------------------------------------------------------
        logger.info("--- Training EnsembleCombiner ---")
        combiner = EnsembleCombiner()
        combiner.fit_meta_learner(xgb_oof, rf_oof, lstm_oof, y_oof)
        combiner.save(str(save_dir / "ensemble_combiner.joblib"))

        # ----------------------------------------------------------------
        # 11. Persist metadata (feature names, config, run info)
        # ----------------------------------------------------------------
        meta = {
            "symbol": symbol,
            "timeframe": timeframe,
            "run_ts": run_ts,
            "feature_cols": feature_cols,
            "config": self.config,
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
        }
        joblib.dump(meta, str(save_dir / "meta.joblib"))

        # ----------------------------------------------------------------
        # 12. Build training report
        # ----------------------------------------------------------------
        report: Dict[str, Any] = {
            "symbol": symbol,
            "timeframe": timeframe,
            "run_ts": run_ts,
            "save_dir": str(save_dir),
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "xgboost": {
                "val_metrics": xgb_metrics,
                "test_metrics": xgb_test_metrics,
                "feature_importance": xgb_model.get_feature_importance(),
            },
            "random_forest": {
                "val_metrics": rf_metrics,
                "test_metrics": rf_test_metrics,
                "feature_importance": rf_model.get_feature_importance(),
            },
            "lstm": {
                "val_metrics": lstm_metrics,
                "test_metrics": lstm_test_metrics,
            },
            "ensemble": {
                "oof_n_samples": int(len(y_oof)),
                "is_meta_fitted": combiner.is_fitted,
            },
        }

        # ----------------------------------------------------------------
        # 13. Log to database
        # ----------------------------------------------------------------
        if self.config["db_logging"]:
            self._log_to_db(report)

        logger.info(
            "Training complete.  Artefacts at: %s",
            save_dir,
        )
        return report

    # ------------------------------------------------------------------

    def load_models(
        self,
        symbol: str,
        timeframe: str,
    ) -> Dict[str, Any]:
        """
        Load the most recently saved model artefacts for *symbol*/*timeframe*.

        Returns
        -------
        dict with keys ``'xgboost'``, ``'random_forest'``, ``'lstm'``,
        ``'combiner'``, ``'scaler'``, ``'meta'``.
        """
        base = _models_base_dir()
        prefix = f"{symbol}_{timeframe}_"
        candidates = sorted(
            [d for d in base.iterdir() if d.is_dir() and d.name.startswith(prefix)],
            key=lambda d: d.name,
        )
        if not candidates:
            raise FileNotFoundError(
                f"No saved models found for {symbol}/{timeframe} under {base}"
            )
        save_dir = candidates[-1]  # latest by lexicographic sort on timestamp
        logger.info("Loading models from %s", save_dir)

        meta = joblib.load(str(save_dir / "meta.joblib"))
        scaler: StandardScaler = joblib.load(str(save_dir / "scaler.joblib"))
        feature_cols: List[str] = meta["feature_cols"]

        xgb_model = XGBoostModel()
        xgb_model.load(str(save_dir / "xgboost.ubj"), feature_names=feature_cols)

        rf_model = RandomForestModel()
        rf_model.load(str(save_dir / "random_forest.joblib"))

        lstm_trainer = LSTMTrainer()
        n_features = len(feature_cols)
        lstm_trainer.load(str(save_dir / "lstm.pt"), input_size=n_features)

        combiner = EnsembleCombiner()
        combiner.load(str(save_dir / "ensemble_combiner.joblib"))

        return {
            "xgboost": xgb_model,
            "random_forest": rf_model,
            "lstm": lstm_trainer,
            "combiner": combiner,
            "scaler": scaler,
            "meta": meta,
        }

    # ------------------------------------------------------------------

    @staticmethod
    def prepare_sequences(
        X_scaled: np.ndarray,
        y: Optional[np.ndarray] = None,
        lookback: int = 60,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Convert a 2-D scaled feature array into 3-D LSTM sequences.

        Parameters
        ----------
        X_scaled : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,), optional
            If provided, the corresponding labels for each sequence are
            returned (aligned to the *last* bar of each window).
        lookback : int
            Number of bars per sequence.

        Returns
        -------
        X_seq : np.ndarray, shape (n_samples - lookback, lookback, n_features)
        y_seq : np.ndarray or None
        """
        n_samples, n_features = X_scaled.shape
        if n_samples <= lookback:
            raise ValueError(
                f"Not enough samples ({n_samples}) for lookback={lookback}."
            )

        n_seq = n_samples - lookback
        X_seq = np.empty((n_seq, lookback, n_features), dtype=np.float32)
        for i in range(n_seq):
            X_seq[i] = X_scaled[i : i + lookback]

        y_seq: Optional[np.ndarray] = None
        if y is not None:
            y_seq = y[lookback:].astype(np.float32)

        return X_seq, y_seq

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure a binary target column exists in *df*.

        If the column specified by ``config['target_column']`` is already
        present, it is used as-is.  Otherwise it is derived from the forward
        return column: label = 1 when return > bullish_threshold, else 0.
        """
        target_col = self.config["target_column"]
        if target_col in df.columns:
            logger.info("Using pre-existing target column '%s'.", target_col)
            return df

        fwd_col = self.config["forward_returns_col"]
        if fwd_col not in df.columns:
            raise KeyError(
                f"Neither target column '{target_col}' nor forward-return "
                f"column '{fwd_col}' found in features_df.  "
                f"Available columns: {list(df.columns)}"
            )

        threshold = self.config["bullish_threshold"]
        df[target_col] = (df[fwd_col] > threshold).astype(int)
        pct_bullish = df[target_col].mean() * 100
        logger.info(
            "Target created from '%s' > %.4f.  Bullish rate: %.1f %%",
            fwd_col,
            threshold,
            pct_bullish,
        )
        return df

    def _evaluate_on_test(
        self,
        model: XGBoostModel | RandomForestModel,
        X_test: np.ndarray,
        y_test: np.ndarray,
        label: str,
    ) -> Dict[str, float]:
        """Compute test-set metrics for XGBoost or RF models."""
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        y_proba = model.predict_proba(X_test)
        y_pred = (y_proba >= 0.5).astype(int)
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
        }
        logger.info(
            "[%s | test] accuracy=%.4f  precision=%.4f  recall=%.4f  "
            "f1=%.4f  roc_auc=%.4f",
            label,
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
            metrics["roc_auc"],
        )
        return metrics

    def _evaluate_lstm_on_test(
        self,
        trainer: LSTMTrainer,
        X_seq_test: np.ndarray,
        y_seq_test: np.ndarray,
    ) -> Dict[str, float]:
        """Compute test-set metrics for the LSTM trainer."""
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        y_proba = trainer.predict_proba(X_seq_test)
        y_pred = (y_proba >= 0.5).astype(int)
        metrics = {
            "accuracy": float(accuracy_score(y_seq_test, y_pred)),
            "precision": float(precision_score(y_seq_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_seq_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_seq_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_seq_test, y_proba)),
        }
        logger.info(
            "[LSTM | test] accuracy=%.4f  precision=%.4f  recall=%.4f  "
            "f1=%.4f  roc_auc=%.4f",
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
            metrics["roc_auc"],
        )
        return metrics

    def _generate_oof_predictions(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        lookback: int,
        scaler: StandardScaler,
        feature_cols: List[str],
        use_lstm: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Use TimeSeriesSplit(n_splits) to generate OOF predictions from each
        base model.  Returns arrays aligned to the same samples.

        Returns
        -------
        xgb_oof, rf_oof, lstm_oof : np.ndarray  shape (n_oof,)
        y_oof : np.ndarray  shape (n_oof,)
        """
        n_splits = self.config["n_cv_splits"]
        tscv = TimeSeriesSplit(n_splits=n_splits)

        xgb_oof_parts: List[np.ndarray] = []
        rf_oof_parts: List[np.ndarray] = []
        lstm_oof_parts: List[np.ndarray] = []
        y_oof_parts: List[np.ndarray] = []

        for fold_idx, (tr_idx, val_idx) in enumerate(tscv.split(X_train), start=1):
            logger.info(
                "OOF fold %d/%d — train=%d  val=%d",
                fold_idx,
                n_splits,
                len(tr_idx),
                len(val_idx),
            )
            X_tr_fold = X_train[tr_idx]
            y_tr_fold = y_train[tr_idx]
            X_val_fold = X_train[val_idx]
            y_val_fold = y_train[val_idx]

            # XGBoost fold
            xgb_fold = XGBoostModel()
            xgb_fold.train(X_tr_fold, y_tr_fold, feature_names=feature_cols)
            xgb_oof_parts.append(xgb_fold.predict_proba(X_val_fold))

            # Random Forest fold
            rf_fold = RandomForestModel()
            rf_fold.train(X_tr_fold, y_tr_fold, feature_names=feature_cols)
            rf_oof_parts.append(rf_fold.predict_proba(X_val_fold))

            # LSTM fold — requires sequence alignment (or use 0.5 when disabled)
            if not use_lstm:
                n = len(y_val_fold)
                lstm_oof_parts.append(np.full(n, 0.5, dtype=np.float64))
                y_oof_parts.append(y_val_fold)
            elif len(X_tr_fold) > lookback and len(X_val_fold) > 0:
                # Combine tr+val, build sequences, split again
                combined = np.vstack([X_tr_fold, X_val_fold])
                y_combined = np.concatenate([y_tr_fold, y_val_fold])
                X_seq_all, y_seq_all = self.prepare_sequences(
                    combined, y_combined, lookback
                )
                # The first len(y_tr_fold) - lookback sequences belong to train
                seq_train_size = len(y_tr_fold) - lookback
                if seq_train_size > 0 and (len(X_seq_all) - seq_train_size) > 0:
                    X_seq_tr = X_seq_all[:seq_train_size]
                    y_seq_tr = y_seq_all[:seq_train_size]
                    X_seq_val = X_seq_all[seq_train_size:]
                    y_seq_val = y_seq_all[seq_train_size:]

                    lstm_fold = LSTMTrainer()
                    lstm_fold.train(
                        X_seq_tr,
                        y_seq_tr,
                        epochs=self.config["lstm_epochs"],
                        batch_size=self.config["lstm_batch_size"],
                        lr=self.config["lstm_lr"],
                    )
                    lstm_proba = lstm_fold.predict_proba(X_seq_val)
                    # Align XGBoost/RF OOF to LSTM (LSTM has fewer due to lookback)
                    n_lstm = len(lstm_proba)
                    # Trim xgb/rf OOF parts from this fold to match
                    xgb_oof_parts[-1] = xgb_oof_parts[-1][-n_lstm:]
                    rf_oof_parts[-1] = rf_oof_parts[-1][-n_lstm:]
                    y_val_fold = y_val_fold[-n_lstm:]
                    lstm_oof_parts.append(lstm_proba)
                    y_oof_parts.append(y_val_fold)
                else:
                    # Not enough data for LSTM in this fold — fill with 0.5
                    n = len(y_val_fold)
                    lstm_oof_parts.append(np.full(n, 0.5, dtype=np.float64))
                    y_oof_parts.append(y_val_fold)
            else:
                n = len(y_val_fold)
                lstm_oof_parts.append(np.full(n, 0.5, dtype=np.float64))
                y_oof_parts.append(y_val_fold)

        xgb_oof = np.concatenate(xgb_oof_parts)
        rf_oof = np.concatenate(rf_oof_parts)
        lstm_oof = np.concatenate(lstm_oof_parts)
        y_oof = np.concatenate(y_oof_parts)
        return xgb_oof, rf_oof, lstm_oof, y_oof

    def _log_to_db(self, report: Dict[str, Any]) -> None:
        """Insert a row into the ``ml_model_runs`` SQLite table."""
        db_path = _db_path()
        db_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            con = sqlite3.connect(str(db_path))
            cur = con.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS ml_model_runs (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol      TEXT    NOT NULL,
                    timeframe   TEXT    NOT NULL,
                    run_ts      TEXT    NOT NULL,
                    save_dir    TEXT    NOT NULL,
                    n_train     INTEGER,
                    n_test      INTEGER,
                    xgb_roc_auc REAL,
                    rf_roc_auc  REAL,
                    lstm_roc_auc REAL,
                    xgb_f1      REAL,
                    rf_f1       REAL,
                    lstm_f1     REAL,
                    created_at  TEXT    DEFAULT (datetime('now'))
                )
                """
            )
            cur.execute(
                """
                INSERT INTO ml_model_runs
                    (symbol, timeframe, run_ts, save_dir, n_train, n_test,
                     xgb_roc_auc, rf_roc_auc, lstm_roc_auc,
                     xgb_f1, rf_f1, lstm_f1)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report["symbol"],
                    report["timeframe"],
                    report["run_ts"],
                    report["save_dir"],
                    report["n_train"],
                    report["n_test"],
                    report["xgboost"]["test_metrics"].get("roc_auc"),
                    report["random_forest"]["test_metrics"].get("roc_auc"),
                    report["lstm"]["test_metrics"].get("roc_auc"),
                    report["xgboost"]["test_metrics"].get("f1"),
                    report["random_forest"]["test_metrics"].get("f1"),
                    report["lstm"]["test_metrics"].get("f1"),
                ),
            )
            con.commit()
            con.close()
            logger.info("Metrics logged to database: %s", db_path)
        except Exception as exc:
            logger.warning("Failed to log metrics to database: %s", exc)
