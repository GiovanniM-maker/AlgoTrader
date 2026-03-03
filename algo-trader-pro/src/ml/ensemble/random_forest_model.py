"""
Random Forest binary classifier for algorithmic trading signal prediction.

Provides the same public interface as XGBoostModel so that the ensemble
combiner and training orchestrator can treat all models uniformly.
Uses scikit-learn's RandomForestClassifier with balanced class weights and
parallel tree building.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class RandomForestModel:
    """
    Random Forest binary classifier wrapper for trading signal prediction.

    Attributes
    ----------
    model : RandomForestClassifier
        Underlying scikit-learn estimator.  None until :meth:`train` or
        :meth:`load` has been called.
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
        n_estimators: int = 200,
        max_depth: int = 10,
        min_samples_split: int = 20,
        min_samples_leaf: int = 10,
        max_features: str = "sqrt",
        n_jobs: int = -1,
        random_state: int = 42,
        class_weight: str = "balanced",
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.class_weight = class_weight

        self.model: Optional[RandomForestClassifier] = None
        self.feature_names: List[str] = []
        self.metrics: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Train the Random Forest classifier.

        Internally reserves 10 % of the data as a validation set for
        metric computation after training.

        Parameters
        ----------
        X_train : np.ndarray, shape (n_samples, n_features)
        y_train : np.ndarray, shape (n_samples,)  — binary labels {0, 1}
        feature_names : list[str], optional
            Names for each column in X_train.

        Returns
        -------
        dict
            Validation metrics: accuracy, precision, recall, f1, roc_auc.
        """
        if feature_names is not None:
            self.feature_names = list(feature_names)
        else:
            self.feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

        n_neg = int(np.sum(y_train == 0))
        n_pos = int(np.sum(y_train == 1))
        logger.info(
            "Class distribution — negatives: %d, positives: %d",
            n_neg,
            n_pos,
        )

        # Hold out 10 % for metric evaluation (stratified).
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.10,
            random_state=self.random_state,
            stratify=y_train,
        )

        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            class_weight=self.class_weight,
            oob_score=True,
        )

        logger.info("Fitting RandomForestClassifier …")
        self.model.fit(X_tr, y_tr)
        logger.info(
            "Random Forest training complete.  OOB score: %.4f",
            self.model.oob_score_,
        )

        self.metrics = self._evaluate(X_val, y_val, dataset_label="validation")
        self.metrics["oob_score"] = float(self.model.oob_score_)
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
        Serialize the fitted model and feature names to *path* with joblib.

        Parameters
        ----------
        path : str
            File path, e.g. ``"models/rf_BTCUSDT_1h.joblib"``
        """
        self._assert_fitted()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "model": self.model,
            "feature_names": self.feature_names,
            "metrics": self.metrics,
            "params": self._get_params(),
        }
        joblib.dump(payload, path, compress=3)
        logger.info("RandomForestModel saved to %s", path)

    def load(self, path: str) -> None:
        """
        Deserialize a previously saved model from *path*.

        Parameters
        ----------
        path : str
        """
        payload = joblib.load(path)
        self.model = payload["model"]
        self.feature_names = payload.get("feature_names", [])
        self.metrics = payload.get("metrics", {})
        logger.info("RandomForestModel loaded from %s", path)

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Return feature importances sorted in descending order.

        Uses scikit-learn's MDI (Mean Decrease in Impurity) importances
        stored in ``feature_importances_`` after fitting.

        Returns
        -------
        dict
            ``{feature_name: importance_score}`` sorted descending by score.
        """
        self._assert_fitted()
        importances = self.model.feature_importances_

        if len(self.feature_names) == len(importances):
            names = self.feature_names
        else:
            names = [f"feature_{i}" for i in range(len(importances))]

        importance_dict = {
            name: float(score) for name, score in zip(names, importances)
        }
        return dict(
            sorted(importance_dict.items(), key=lambda kv: kv[1], reverse=True)
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
            "precision": float(precision_score(y, y_pred, zero_division=0)),
            "recall": float(recall_score(y, y_pred, zero_division=0)),
            "f1": float(f1_score(y, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, y_proba)),
        }

        logger.info(
            "[RandomForest | %s] accuracy=%.4f  precision=%.4f  recall=%.4f  "
            "f1=%.4f  roc_auc=%.4f",
            dataset_label,
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
            metrics["roc_auc"],
        )
        return metrics

    def _get_params(self) -> Dict:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "n_jobs": self.n_jobs,
            "random_state": self.random_state,
            "class_weight": self.class_weight,
        }

    def _assert_fitted(self) -> None:
        if self.model is None:
            raise RuntimeError(
                "RandomForestModel has not been trained yet.  "
                "Call train() or load() first."
            )
