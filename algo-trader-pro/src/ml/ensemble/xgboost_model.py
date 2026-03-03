"""
XGBoost binary classifier for algorithmic trading signal prediction.

Predicts the probability of a bullish outcome (class 1) given a feature matrix.
Supports early stopping via an internal 10% validation split, class-imbalance
correction via scale_pos_weight, and full sklearn-style evaluation metrics stored
on self.metrics after training.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional

import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class XGBoostModel:
    """
    XGBoost binary classifier wrapper tailored for trading signal prediction.

    Attributes
    ----------
    model : xgb.XGBClassifier
        Underlying XGBoost estimator.  None until :meth:`train` or :meth:`load`
        has been called.
    feature_names : list[str]
        Column names seen during training, in order.
    metrics : dict
        Evaluation metrics populated after :meth:`train` completes.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        early_stopping_rounds: int = 50,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_names: list[str] = []
        self.metrics: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: Optional[list[str]] = None,
    ) -> Dict[str, float]:
        """
        Train the XGBoost classifier.

        Internally splits off 10 % of the training data as a validation set
        for early stopping.  Class imbalance is addressed via
        ``scale_pos_weight``.

        Parameters
        ----------
        X_train : np.ndarray, shape (n_samples, n_features)
        y_train : np.ndarray, shape (n_samples,)  — binary labels {0, 1}
        feature_names : list[str], optional
            Names of each column in X_train.  Used by
            :meth:`get_feature_importance`.

        Returns
        -------
        dict
            Training metrics: accuracy, precision, recall, f1, roc_auc.
        """
        if feature_names is not None:
            self.feature_names = list(feature_names)
        else:
            self.feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

        # ----------------------------------------------------------------
        # Compute scale_pos_weight to handle class imbalance
        # ----------------------------------------------------------------
        n_neg = int(np.sum(y_train == 0))
        n_pos = int(np.sum(y_train == 1))
        if n_pos == 0:
            raise ValueError("Training set contains no positive (bullish) samples.")
        scale_pos_weight = n_neg / n_pos
        logger.info(
            "Class distribution — negatives: %d, positives: %d, "
            "scale_pos_weight: %.4f",
            n_neg,
            n_pos,
            scale_pos_weight,
        )

        # ----------------------------------------------------------------
        # Internal 10 % validation split (stratified, no time-leakage risk
        # here because the outer caller is responsible for that boundary)
        # ----------------------------------------------------------------
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.10,
            random_state=self.random_state,
            stratify=y_train,
        )

        # ----------------------------------------------------------------
        # Build and fit the classifier
        # ----------------------------------------------------------------
        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            use_label_encoder=False,
            eval_metric="logloss",
            early_stopping_rounds=self.early_stopping_rounds,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbosity=0,
        )

        self.model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        logger.info(
            "XGBoost training complete.  Best iteration: %d",
            self.model.best_iteration,
        )

        # ----------------------------------------------------------------
        # Evaluate on the held-out validation set
        # ----------------------------------------------------------------
        self.metrics = self._evaluate(X_val, y_val, dataset_label="validation")
        return self.metrics

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return the probability of class 1 (bullish) for each sample.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,)
            Probabilities in [0, 1].
        """
        self._assert_fitted()
        proba_matrix = self.model.predict_proba(X)
        return proba_matrix[:, 1].astype(np.float64)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Return binary predictions using the given probability threshold.

        Parameters
        ----------
        X : np.ndarray
        threshold : float

        Returns
        -------
        np.ndarray of int  {0, 1}
        """
        return (self.predict_proba(X) >= threshold).astype(int)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Save the fitted model to *path* using XGBoost's native format.

        Parameters
        ----------
        path : str
            File path, e.g. ``"models/xgb_BTCUSDT_1h.ubj"``
        """
        self._assert_fitted()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.model.save_model(path)
        logger.info("XGBoostModel saved to %s", path)

    def load(self, path: str, feature_names: Optional[list[str]] = None) -> None:
        """
        Load a previously saved model from *path*.

        Parameters
        ----------
        path : str
        feature_names : list[str], optional
            Must match the feature order used during the original training if
            you intend to call :meth:`get_feature_importance` by name.
        """
        self.model = xgb.XGBClassifier()
        self.model.load_model(path)
        if feature_names is not None:
            self.feature_names = list(feature_names)
        logger.info("XGBoostModel loaded from %s", path)

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Return feature importances sorted in descending order.

        Uses XGBoost's ``gain`` importance type, which measures the average
        gain of each feature when it is used in a split.

        Returns
        -------
        dict
            ``{feature_name: importance_score}`` sorted descending by score.
        """
        self._assert_fitted()
        raw_importance = self.model.get_booster().get_score(importance_type="gain")

        # XGBoost internally names features f0, f1, … unless feature_names
        # were passed at fit time via feature_names param.  Map back to the
        # caller-supplied names when available.
        named_importance: Dict[str, float] = {}
        for key, score in raw_importance.items():
            if key.startswith("f") and key[1:].isdigit():
                idx = int(key[1:])
                if idx < len(self.feature_names):
                    named_importance[self.feature_names[idx]] = float(score)
                else:
                    named_importance[key] = float(score)
            else:
                named_importance[key] = float(score)

        return dict(
            sorted(named_importance.items(), key=lambda kv: kv[1], reverse=True)
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        dataset_label: str = "test",
    ) -> Dict[str, float]:
        """Compute and log classification metrics."""
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)

        metrics = {
            "accuracy": float(accuracy_score(y, y_pred)),
            "precision": float(
                precision_score(y, y_pred, zero_division=0)
            ),
            "recall": float(recall_score(y, y_pred, zero_division=0)),
            "f1": float(f1_score(y, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, y_proba)),
        }

        logger.info(
            "[XGBoost | %s] accuracy=%.4f  precision=%.4f  recall=%.4f  "
            "f1=%.4f  roc_auc=%.4f",
            dataset_label,
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
            metrics["roc_auc"],
        )
        return metrics

    def _assert_fitted(self) -> None:
        if self.model is None:
            raise RuntimeError(
                "XGBoostModel has not been trained yet.  "
                "Call train() or load() first."
            )
