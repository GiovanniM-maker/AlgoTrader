"""
src/ml/training/walk_forward.py

Walk-Forward Validator
======================
Performs time-series cross-validation (walk-forward analysis) to give an
unbiased estimate of out-of-sample model performance.

Unlike standard k-fold cross-validation, walk-forward respects temporal
order: the model is always trained on data that precedes the test window.
This prevents look-ahead bias and produces realistic estimates of how the
model would have performed in live deployment.

Window structure (default: train=6 months, test=1 month, step=1 month)
-----------------------------------------------------------------------

  |←──── train (6 mo) ────→|← test (1 mo) →|
                             |←──── train (6 mo) ────→|← test (1 mo) →|
                                              |←──── train (6 mo) ────→|← test (1 mo) →|

Each fold advances by *step_months* so folds overlap in the training window
but test windows never overlap.

Minimum size guards
-------------------
Folds where ``len(train_df) < 500`` or ``len(test_df) < 50`` are skipped
with a warning log.

Usage
-----
    from src.ml.training.walk_forward import WalkForwardValidator

    validator = WalkForwardValidator(train_months=6, test_months=1, step_months=1)
    results = validator.validate(features_df, trainer=my_trainer)
    print(results["mean_roc_auc"])
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Minimum sizes below which a fold is skipped
_MIN_TRAIN_ROWS: int = 500
_MIN_TEST_ROWS: int = 50

# Number of days per calendar month (approximate)
_DAYS_PER_MONTH: int = 30


# ---------------------------------------------------------------------------
# WalkForwardValidator
# ---------------------------------------------------------------------------


class WalkForwardValidator:
    """
    Time-series walk-forward cross-validator for the AlgoTrader ML pipeline.

    Parameters
    ----------
    train_months : int
        Number of months in each training window.  Default: 6.
    test_months : int
        Number of months in each out-of-sample test window.  Default: 1.
    step_months : int
        How many months to advance the window between folds.  Default: 1.
    """

    def __init__(
        self,
        train_months: int = 6,
        test_months: int = 1,
        step_months: int = 1,
    ) -> None:
        if train_months < 1:
            raise ValueError(f"train_months must be >= 1, got {train_months}")
        if test_months < 1:
            raise ValueError(f"test_months must be >= 1, got {test_months}")
        if step_months < 1:
            raise ValueError(f"step_months must be >= 1, got {step_months}")

        self.train_months: int = train_months
        self.test_months: int = test_months
        self.step_months: int = step_months
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(
        self,
        features_df: pd.DataFrame,
        trainer: Any,
        symbol: str = "BTCUSDT",
        timeframe: str = "60",
        target_col: str = "target",
        feature_cols: Optional[List[str]] = None,
    ) -> dict:
        """
        Run walk-forward validation across all generated folds.

        Parameters
        ----------
        features_df : pd.DataFrame
            Full feature matrix.  Must have a ``DatetimeIndex`` or a column
            named ``datetime`` / ``timestamp`` / ``open_time`` that can be
            converted to a DatetimeIndex.
        trainer :
            Instance of ``ModelTrainer`` (or any object exposing
            ``train_all(df, symbol, timeframe) -> dict``).
        symbol : str
            Symbol label passed to the trainer (e.g. ``"BTCUSDT"``).
        timeframe : str
            Timeframe label passed to the trainer (e.g. ``"60"``).
        target_col : str
            Name of the binary target column in ``features_df``.
            Default: ``"target"``.
        feature_cols : list of str, optional
            Subset of columns to use as features.  If ``None`` all columns
            except the target are used.

        Returns
        -------
        dict
            Aggregated cross-validation results:
            ``mean_accuracy``, ``std_accuracy``, ``mean_roc_auc``,
            ``std_roc_auc``, ``mean_precision``, ``std_precision``,
            ``mean_recall``, ``std_recall``, ``mean_f1``, ``std_f1``,
            ``n_folds``, ``n_folds_skipped``, ``fold_details``.
        """
        self.logger.info(
            "Walk-forward validation started: train=%dmo, test=%dmo, step=%dmo, rows=%d",
            self.train_months, self.test_months, self.step_months, len(features_df),
        )

        # Ensure DatetimeIndex
        features_df = self._ensure_datetime_index(features_df)

        # Generate fold windows
        folds = self._get_folds(features_df)
        self.logger.info("Generated %d walk-forward folds.", len(folds))

        if not folds:
            self.logger.warning("No valid folds generated — check data length.")
            return self._empty_results()

        fold_results: List[dict] = []
        skipped: int = 0

        for fold_idx, fold in enumerate(folds):
            train_start = fold["train_start"]
            train_end = fold["train_end"]
            test_start = fold["test_start"]
            test_end = fold["test_end"]

            # ---------------------------------------------------------------
            # Slice train / test windows
            # ---------------------------------------------------------------
            train_df = features_df.loc[train_start:train_end].copy()
            test_df = features_df.loc[test_start:test_end].copy()

            # Minimum-size guards
            if len(train_df) < _MIN_TRAIN_ROWS:
                self.logger.warning(
                    "Fold %d skipped: train_df too small (%d < %d rows). "
                    "Window: train=[%s, %s]",
                    fold_idx,
                    len(train_df),
                    _MIN_TRAIN_ROWS,
                    train_start.date(),
                    train_end.date(),
                )
                skipped += 1
                continue

            if len(test_df) < _MIN_TEST_ROWS:
                self.logger.warning(
                    "Fold %d skipped: test_df too small (%d < %d rows). "
                    "Window: test=[%s, %s]",
                    fold_idx,
                    len(test_df),
                    _MIN_TEST_ROWS,
                    test_start.date(),
                    test_end.date(),
                )
                skipped += 1
                continue

            # Validate target column exists in both splits
            if target_col not in train_df.columns:
                self.logger.warning(
                    "Fold %d skipped: target column '%s' not found.", fold_idx, target_col
                )
                skipped += 1
                continue

            self.logger.info(
                "Fold %d/%d — train=[%s, %s] (%d rows), test=[%s, %s] (%d rows)",
                fold_idx + 1, len(folds),
                train_start.date(), train_end.date(), len(train_df),
                test_start.date(), test_end.date(), len(test_df),
            )

            # ---------------------------------------------------------------
            # Train models on training fold
            # ---------------------------------------------------------------
            try:
                _train_report = trainer.train_all(
                    df=train_df,
                    symbol=symbol,
                    timeframe=timeframe,
                )
            except Exception as exc:
                self.logger.error(
                    "Fold %d — trainer.train_all raised: %s", fold_idx, exc
                )
                skipped += 1
                continue

            # ---------------------------------------------------------------
            # Evaluate on test fold
            # ---------------------------------------------------------------
            try:
                fold_metrics = self._evaluate_on_test(
                    trainer=trainer,
                    test_df=test_df,
                    target_col=target_col,
                    feature_cols=feature_cols,
                    fold_idx=fold_idx,
                )
            except Exception as exc:
                self.logger.error(
                    "Fold %d — evaluation raised: %s", fold_idx, exc
                )
                skipped += 1
                continue

            fold_metrics.update({
                "fold_idx": fold_idx,
                "train_start": str(train_start.date()),
                "train_end": str(train_end.date()),
                "test_start": str(test_start.date()),
                "test_end": str(test_end.date()),
                "n_train": len(train_df),
                "n_test": len(test_df),
            })
            fold_results.append(fold_metrics)

            self.logger.info(
                "Fold %d metrics: accuracy=%.4f roc_auc=%.4f precision=%.4f recall=%.4f f1=%.4f",
                fold_idx,
                fold_metrics.get("accuracy", 0),
                fold_metrics.get("roc_auc", 0),
                fold_metrics.get("precision", 0),
                fold_metrics.get("recall", 0),
                fold_metrics.get("f1", 0),
            )

        # ---------------------------------------------------------------
        # Aggregate results
        # ---------------------------------------------------------------
        results = self.aggregate_results(fold_results)
        results["n_folds_skipped"] = skipped
        results["fold_details"] = fold_results

        self.logger.info(
            "Walk-forward complete: n_folds=%d, n_skipped=%d, "
            "mean_acc=%.4f, mean_roc_auc=%.4f",
            results["n_folds"],
            skipped,
            results.get("mean_accuracy", 0),
            results.get("mean_roc_auc", 0),
        )
        return results

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def aggregate_results(self, fold_results: List[dict]) -> dict:
        """
        Compute mean and standard deviation of key metrics across all folds.

        Parameters
        ----------
        fold_results : list of dict
            One dict per completed fold, each containing at minimum:
            ``accuracy``, ``roc_auc``, ``precision``, ``recall``, ``f1``.

        Returns
        -------
        dict
            Keys: ``mean_{metric}``, ``std_{metric}`` for each metric,
            plus ``n_folds`` and ``fold_details``.
        """
        if not fold_results:
            result = self._empty_results()
            result["fold_details"] = []
            return result

        metrics_to_aggregate = ["accuracy", "roc_auc", "precision", "recall", "f1"]
        aggregated: dict = {"n_folds": len(fold_results)}

        for metric in metrics_to_aggregate:
            values = [
                fold[metric]
                for fold in fold_results
                if metric in fold and fold[metric] is not None
            ]
            if values:
                mean_val = float(np.mean(values))
                std_val = float(np.std(values, ddof=0))
            else:
                mean_val = 0.0
                std_val = 0.0

            aggregated[f"mean_{metric}"] = round(mean_val, 6)
            aggregated[f"std_{metric}"] = round(std_val, 6)

        aggregated["fold_details"] = fold_results
        return aggregated

    # ------------------------------------------------------------------
    # Fold generation
    # ------------------------------------------------------------------

    def _get_folds(self, df: pd.DataFrame) -> List[dict]:
        """
        Generate the list of walk-forward fold windows.

        Each fold is a dict with keys:
        ``train_start``, ``train_end``, ``test_start``, ``test_end``
        (all ``pd.Timestamp`` objects).

        The first test window starts immediately after the first training
        window ends.  Subsequent windows are advanced by ``step_months``.
        """
        if df.empty:
            return []

        train_delta = timedelta(days=self.train_months * _DAYS_PER_MONTH)
        test_delta = timedelta(days=self.test_months * _DAYS_PER_MONTH)
        step_delta = timedelta(days=self.step_months * _DAYS_PER_MONTH)

        data_start: pd.Timestamp = df.index.min()
        data_end: pd.Timestamp = df.index.max()

        folds: List[dict] = []
        train_start: pd.Timestamp = data_start

        while True:
            train_end = train_start + train_delta
            test_start = train_end
            test_end = test_start + test_delta

            # Stop generating folds when the test window would exceed the data
            if test_start >= data_end:
                break
            if test_end > data_end:
                test_end = data_end

            folds.append({
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
            })

            train_start = train_start + step_delta

        return folds

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def _evaluate_on_test(
        self,
        trainer: Any,
        test_df: pd.DataFrame,
        target_col: str,
        feature_cols: Optional[List[str]],
        fold_idx: int,
    ) -> dict:
        """
        Generate predictions on ``test_df`` and compute classification metrics.

        The trainer is expected to have a ``predict(df) -> np.ndarray`` method
        that returns class predictions (0 / 1), and optionally a
        ``predict_proba(df) -> np.ndarray`` method returning probabilities for
        ROC-AUC computation.

        Returns
        -------
        dict
            Keys: ``accuracy``, ``roc_auc``, ``precision``, ``recall``, ``f1``.
            Any metric that cannot be computed defaults to 0.0.
        """
        y_true = test_df[target_col].values.astype(int)

        # Build feature matrix
        if feature_cols is not None:
            available = [c for c in feature_cols if c in test_df.columns]
            X_test = test_df[available]
        else:
            X_test = test_df.drop(
                columns=[c for c in [target_col] if c in test_df.columns],
                errors="ignore",
            )

        # Class predictions
        try:
            y_pred = np.array(trainer.predict(X_test)).astype(int)
        except Exception as exc:
            self.logger.warning(
                "Fold %d — trainer.predict failed: %s.  Returning zero metrics.",
                fold_idx, exc,
            )
            return {
                "accuracy": 0.0, "roc_auc": 0.0,
                "precision": 0.0, "recall": 0.0, "f1": 0.0,
            }

        if len(y_pred) != len(y_true):
            self.logger.warning(
                "Fold %d — prediction length mismatch: expected %d got %d.",
                fold_idx, len(y_true), len(y_pred),
            )
            min_len = min(len(y_pred), len(y_true))
            y_pred = y_pred[:min_len]
            y_true = y_true[:min_len]

        # Core metrics
        accuracy = self._accuracy(y_true, y_pred)
        precision, recall, f1 = self._precision_recall_f1(y_true, y_pred)

        # ROC-AUC (requires predict_proba)
        roc_auc = 0.0
        try:
            y_proba = np.array(trainer.predict_proba(X_test))
            if y_proba.ndim == 2:
                y_proba = y_proba[:, 1]  # positive class probability
            if len(y_proba) == len(y_true):
                roc_auc = self._roc_auc(y_true, y_proba)
        except Exception as exc:
            self.logger.debug(
                "Fold %d — predict_proba not available or failed: %s. "
                "ROC-AUC set to 0.",
                fold_idx, exc,
            )

        return {
            "accuracy":  round(float(accuracy),  6),
            "roc_auc":   round(float(roc_auc),   6),
            "precision": round(float(precision),  6),
            "recall":    round(float(recall),     6),
            "f1":        round(float(f1),         6),
        }

    # ------------------------------------------------------------------
    # Metric implementations (no sklearn dependency at import time)
    # ------------------------------------------------------------------

    @staticmethod
    def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Simple accuracy: fraction of correct predictions."""
        if len(y_true) == 0:
            return 0.0
        return float(np.mean(y_true == y_pred))

    @staticmethod
    def _precision_recall_f1(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        positive_label: int = 1,
    ) -> tuple[float, float, float]:
        """
        Compute binary precision, recall, and F1 score for the positive class.
        Returns (precision, recall, f1).  All default to 0.0 on edge cases.
        """
        tp = int(np.sum((y_true == positive_label) & (y_pred == positive_label)))
        fp = int(np.sum((y_true != positive_label) & (y_pred == positive_label)))
        fn = int(np.sum((y_true == positive_label) & (y_pred != positive_label)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        return precision, recall, f1

    @staticmethod
    def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
        """
        Compute ROC-AUC using the trapezoidal rule (Mann-Whitney U statistic).

        Equivalent to ``sklearn.metrics.roc_auc_score`` for binary labels.
        Returns 0.5 (random) when only one class is present.
        """
        classes = np.unique(y_true)
        if len(classes) < 2:
            return 0.5

        # Wilcoxon-Mann-Whitney estimator: AUC = P(score(pos) > score(neg))
        pos_scores = y_score[y_true == 1]
        neg_scores = y_score[y_true == 0]

        if len(pos_scores) == 0 or len(neg_scores) == 0:
            return 0.5

        # Count concordant pairs (pos_score > neg_score) + 0.5 * ties
        n_pos = len(pos_scores)
        n_neg = len(neg_scores)

        concordant = 0
        tied = 0
        for ps in pos_scores:
            concordant += int(np.sum(ps > neg_scores))
            tied += int(np.sum(ps == neg_scores))

        auc = (concordant + 0.5 * tied) / (n_pos * n_neg)
        return float(auc)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure *df* has a ``DatetimeIndex``.

        Checks for a DatetimeIndex first; if absent, searches for a column
        named ``datetime``, ``timestamp``, or ``open_time`` and converts it.
        Raises ``ValueError`` if no suitable column is found.
        """
        if isinstance(df.index, pd.DatetimeIndex):
            return df

        # Try common timestamp column names
        for col in ("datetime", "timestamp", "open_time", "date"):
            if col in df.columns:
                df = df.copy()
                df[col] = pd.to_datetime(df[col], unit="ms" if col == "open_time" else None)
                df = df.set_index(col).sort_index()
                logger.debug("Set DatetimeIndex from column '%s'.", col)
                return df

        raise ValueError(
            "features_df must have a DatetimeIndex or a column named "
            "'datetime', 'timestamp', or 'open_time'."
        )

    @staticmethod
    def _empty_results() -> dict:
        """Return a zeroed-out results dict for when validation cannot run."""
        return {
            "mean_accuracy":  0.0, "std_accuracy":  0.0,
            "mean_roc_auc":   0.0, "std_roc_auc":   0.0,
            "mean_precision": 0.0, "std_precision": 0.0,
            "mean_recall":    0.0, "std_recall":    0.0,
            "mean_f1":        0.0, "std_f1":        0.0,
            "n_folds": 0,
            "n_folds_skipped": 0,
            "fold_details": [],
        }
