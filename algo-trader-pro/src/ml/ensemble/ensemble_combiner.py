"""
Meta-learner that combines XGBoost, Random Forest, and LSTM predictions into a
single confidence score (0–100).

The combiner supports two inference modes:

1. **Meta-learner mode** (preferred): a LogisticRegression trained on
   out-of-fold (OOF) predictions via :meth:`fit_meta_learner`.
2. **Weighted-average fallback**: simple dot product of individual probabilities
   with configurable weights when the meta-learner has not been fitted.

The final ``confidence_score`` equals ``meta_learner.predict_proba(…)[class_1]
* 100``.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Default per-model weights used in weighted-average fallback.
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "xgboost": 0.4,
    "random_forest": 0.3,
    "lstm": 0.3,
}


class EnsembleCombiner:
    """
    Meta-learner ensemble that combines three base-model probability estimates.

    Parameters
    ----------
    weights : dict, optional
        Per-model weights for the fallback weighted-average mode.
        Keys must include ``'xgboost'``, ``'random_forest'``, and ``'lstm'``.

    Attributes
    ----------
    meta_learner : LogisticRegression or None
        Fitted meta-learner.  None until :meth:`fit_meta_learner` is called.
    meta_scaler : StandardScaler or None
        Scaler for the meta-feature stack.
    weights : dict
    is_fitted : bool
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, weights: Optional[Dict[str, float]] = None) -> None:
        self.weights: Dict[str, float] = weights if weights is not None else dict(_DEFAULT_WEIGHTS)
        self._validate_weights()

        self.meta_learner: Optional[LogisticRegression] = None
        self.meta_scaler: Optional[StandardScaler] = None
        self.is_fitted: bool = False

    # ------------------------------------------------------------------
    # Meta-learner training
    # ------------------------------------------------------------------

    def fit_meta_learner(
        self,
        xgb_oof_proba: np.ndarray,
        rf_oof_proba: np.ndarray,
        lstm_oof_proba: np.ndarray,
        y: np.ndarray,
    ) -> "EnsembleCombiner":
        """
        Train a Logistic Regression meta-learner on out-of-fold predictions.

        The three OOF probability arrays are stacked into a feature matrix
        ``(n_samples, 3)`` and a StandardScaler is applied before fitting.

        Parameters
        ----------
        xgb_oof_proba : np.ndarray, shape (n_samples,)
            XGBoost class-1 OOF probabilities.
        rf_oof_proba : np.ndarray, shape (n_samples,)
            Random Forest class-1 OOF probabilities.
        lstm_oof_proba : np.ndarray, shape (n_samples,)
            LSTM class-1 OOF probabilities.
        y : np.ndarray, shape (n_samples,)
            True binary labels.

        Returns
        -------
        self
        """
        meta_X = self._stack(xgb_oof_proba, rf_oof_proba, lstm_oof_proba)

        self.meta_scaler = StandardScaler()
        meta_X_scaled = self.meta_scaler.fit_transform(meta_X)

        self.meta_learner = LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=42,
        )
        self.meta_learner.fit(meta_X_scaled, y)
        self.is_fitted = True

        logger.info(
            "EnsembleCombiner meta-learner fitted on %d samples.  "
            "Intercept: %.4f  Coefficients: %s",
            len(y),
            float(self.meta_learner.intercept_[0]),
            self.meta_learner.coef_.tolist(),
        )
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        xgb_proba: float | np.ndarray,
        rf_proba: float | np.ndarray,
        lstm_proba: float | np.ndarray,
    ) -> float | np.ndarray:
        """
        Combine base-model probabilities into a confidence score 0–100.

        When the meta-learner has been fitted, it is used.  Otherwise the
        method falls back to the weighted average of the three probabilities.

        Parameters
        ----------
        xgb_proba, rf_proba, lstm_proba : float or np.ndarray
            Class-1 probabilities from each base model.  Either all scalars
            (single-sample inference) or all 1-D arrays of the same length.

        Returns
        -------
        float or np.ndarray
            Confidence score(s) in [0, 100].
        """
        scalar_input = np.isscalar(xgb_proba)

        xgb_arr = np.atleast_1d(np.asarray(xgb_proba, dtype=np.float64))
        rf_arr = np.atleast_1d(np.asarray(rf_proba, dtype=np.float64))
        lstm_arr = np.atleast_1d(np.asarray(lstm_proba, dtype=np.float64))

        if self.is_fitted and self.meta_learner is not None:
            scores = self._predict_meta(xgb_arr, rf_arr, lstm_arr)
        else:
            logger.debug(
                "Meta-learner not fitted; using weighted-average fallback."
            )
            scores = self._predict_weighted(xgb_arr, rf_arr, lstm_arr)

        if scalar_input:
            return float(scores[0])
        return scores

    # ------------------------------------------------------------------
    # Agreement measure
    # ------------------------------------------------------------------

    def get_model_agreement(
        self,
        probas_dict: Dict[str, float | np.ndarray],
    ) -> float | np.ndarray:
        """
        Measure how much the three models agree with each other.

        Agreement is defined as 1 minus the coefficient of variation (std/mean)
        of the three probabilities, clamped to [0, 1].  A value of 1 means
        all models output the same probability; 0 means maximum disagreement.

        Parameters
        ----------
        probas_dict : dict
            ``{'xgboost': ..., 'random_forest': ..., 'lstm': ...}``
            Values may be scalars or 1-D arrays.

        Returns
        -------
        float or np.ndarray
            Agreement score(s) in [0, 1].
        """
        required = {"xgboost", "random_forest", "lstm"}
        missing = required - set(probas_dict.keys())
        if missing:
            raise ValueError(f"probas_dict is missing keys: {missing}")

        xgb_arr = np.atleast_1d(np.asarray(probas_dict["xgboost"], dtype=np.float64))
        rf_arr = np.atleast_1d(np.asarray(probas_dict["random_forest"], dtype=np.float64))
        lstm_arr = np.atleast_1d(np.asarray(probas_dict["lstm"], dtype=np.float64))

        stack = np.stack([xgb_arr, rf_arr, lstm_arr], axis=0)  # (3, n)
        means = stack.mean(axis=0)                              # (n,)
        stds = stack.std(axis=0)                                # (n,)

        # Coefficient of variation, safe against zero mean
        cv = np.where(means > 1e-9, stds / means, stds)
        # Clamp CV to [0, 1] then invert
        agreement = 1.0 - np.clip(cv, 0.0, 1.0)

        scalar = (
            np.isscalar(probas_dict["xgboost"])
            and np.isscalar(probas_dict["random_forest"])
            and np.isscalar(probas_dict["lstm"])
        )
        if scalar:
            return float(agreement[0])
        return agreement

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Serialize the combiner (meta-learner + scaler + weights) to *path*.

        Parameters
        ----------
        path : str
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "meta_learner": self.meta_learner,
            "meta_scaler": self.meta_scaler,
            "weights": self.weights,
            "is_fitted": self.is_fitted,
        }
        joblib.dump(payload, path, compress=3)
        logger.info("EnsembleCombiner saved to %s", path)

    def load(self, path: str) -> None:
        """
        Deserialize a previously saved EnsembleCombiner from *path*.

        Parameters
        ----------
        path : str
        """
        payload = joblib.load(path)
        self.meta_learner = payload["meta_learner"]
        self.meta_scaler = payload.get("meta_scaler")
        self.weights = payload.get("weights", dict(_DEFAULT_WEIGHTS))
        self.is_fitted = payload.get("is_fitted", False)
        logger.info("EnsembleCombiner loaded from %s", path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _predict_meta(
        self,
        xgb_arr: np.ndarray,
        rf_arr: np.ndarray,
        lstm_arr: np.ndarray,
    ) -> np.ndarray:
        """Use the fitted meta-learner to predict confidence scores."""
        meta_X = self._stack(xgb_arr, rf_arr, lstm_arr)
        if self.meta_scaler is not None:
            meta_X = self.meta_scaler.transform(meta_X)
        proba_class1 = self.meta_learner.predict_proba(meta_X)[:, 1]
        return proba_class1 * 100.0

    def _predict_weighted(
        self,
        xgb_arr: np.ndarray,
        rf_arr: np.ndarray,
        lstm_arr: np.ndarray,
    ) -> np.ndarray:
        """Weighted-average fallback."""
        w_xgb = self.weights["xgboost"]
        w_rf = self.weights["random_forest"]
        w_lstm = self.weights["lstm"]
        total_w = w_xgb + w_rf + w_lstm
        weighted = (w_xgb * xgb_arr + w_rf * rf_arr + w_lstm * lstm_arr) / total_w
        return weighted * 100.0

    @staticmethod
    def _stack(
        xgb_arr: np.ndarray,
        rf_arr: np.ndarray,
        lstm_arr: np.ndarray,
    ) -> np.ndarray:
        """Stack three 1-D arrays into a (n, 3) feature matrix."""
        return np.column_stack([xgb_arr, rf_arr, lstm_arr])

    def _validate_weights(self) -> None:
        required = {"xgboost", "random_forest", "lstm"}
        missing = required - set(self.weights.keys())
        if missing:
            raise ValueError(f"weights dict is missing keys: {missing}")
        total = sum(self.weights.values())
        if not (0.99 <= total <= 1.01):
            logger.warning(
                "Weights sum to %.4f (expected 1.0).  "
                "They will be normalised at inference time.",
                total,
            )
