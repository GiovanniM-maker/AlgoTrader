"""
ML Predictor — Wrapper per inference in tempo reale.

Carica XGBoost, Random Forest, (LSTM opzionale) e EnsembleCombiner
dalla directory COMBINED_1h_*, applica feature engineering e restituisce
un confidence score 0–100 per la HybridStrategy.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TIMEFRAME_MAP = {1: "1m", 5: "5m", 15: "15m", 30: "15m", 60: "1h", 240: "4h", 1440: "1d"}


class MLPredictor:
    """
    Wrapper per inference ML che accetta OHLCV DataFrame e restituisce
    confidence score 0–100.

    Parameters
    ----------
    models_dir : Path or str
        Directory contenente le sottodirectory COMBINED_1h_*.
    timeframe : str or int
        Timeframe (es. "1h" o 60).
    """

    def __init__(
        self,
        models_dir: Optional[Path] = None,
        timeframe: str = "1h",
    ) -> None:
        self.models_dir = Path(models_dir) if models_dir else Path(__file__).parent / "models"
        tf_str = TIMEFRAME_MAP.get(timeframe, timeframe) if isinstance(timeframe, int) else timeframe
        self.tf_str = tf_str
        self._scaler = None
        self._xgb = None
        self._rf = None
        self._lstm = None
        self._combiner = None
        self._feature_cols = None
        self._meta = None
        self._loaded = False

    def load(self) -> bool:
        """Carica modelli dalla directory COMBINED_* più recente."""
        import joblib

        prefix = f"COMBINED_{self.tf_str}_"
        candidates = sorted(
            [d for d in self.models_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)],
            key=lambda d: d.name,
        )
        if not candidates:
            logger.warning("MLPredictor: nessuna directory %s* trovata in %s", prefix, self.models_dir)
            return False

        save_dir = candidates[-1]
        try:
            self._meta = joblib.load(str(save_dir / "meta.joblib"))
            self._scaler = joblib.load(str(save_dir / "scaler.joblib"))
            self._feature_cols = self._meta["feature_cols"]

            from src.ml.ensemble.ensemble_combiner import EnsembleCombiner
            from src.ml.ensemble.xgboost_model import XGBoostModel
            from src.ml.ensemble.random_forest_model import RandomForestModel

            self._xgb = XGBoostModel()
            self._xgb.load(str(save_dir / "xgboost.ubj"), feature_names=self._feature_cols)

            self._rf = RandomForestModel()
            self._rf.load(str(save_dir / "random_forest.joblib"))

            self._combiner = EnsembleCombiner()
            self._combiner.load(str(save_dir / "ensemble_combiner.joblib"))

            self._lstm = None
            if (save_dir / "lstm.pt").exists():
                from src.ml.ensemble.lstm_model import LSTMTrainer
                self._lstm = LSTMTrainer()
                self._lstm.load(str(save_dir / "lstm.pt"), input_size=len(self._feature_cols))

            self._loaded = True
            logger.info("MLPredictor caricato da %s", save_dir.name)
            return True
        except Exception as exc:
            logger.warning("MLPredictor load failed: %s", exc)
            return False

    def predict(self, df: pd.DataFrame) -> float:
        """
        Predice confidence score 0–100 da OHLCV DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV con colonne open, high, low, close, volume (e opz. open_time).

        Returns
        -------
        float
            Confidence 0–100. Ritorna 50.0 (neutro) se non caricato o errore.
        """
        if not self._loaded:
            return 50.0

        try:
            from src.data.pipeline.feature_engineer import build_features

            features_df = build_features(df, sentiment=None)
            if features_df is None or features_df.empty or len(features_df) < 1:
                return 50.0

            # Ultima riga
            last = features_df.iloc[[-1]]
            X = np.zeros((1, len(self._feature_cols)), dtype=np.float32)
            for j, col in enumerate(self._feature_cols):
                if col in last.columns:
                    X[0, j] = float(last[col].iloc[0])
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            X_scaled = self._scaler.transform(X)

            xgb_proba = self._xgb.predict_proba(X_scaled)
            rf_proba = self._rf.predict_proba(X_scaled)

            if self._lstm is not None:
                meta_cfg = self._meta.get("config") or {}
                lookback = meta_cfg.get("lookback", 60) if isinstance(meta_cfg, dict) else 60
                try:
                    from src.ml.training.trainer import ModelTrainer
                    X_seq, _ = ModelTrainer.prepare_sequences(X_scaled, None, lookback)
                    lstm_proba = self._lstm.predict_proba(X_seq)
                    if hasattr(lstm_proba, "flatten"):
                        lstm_proba = float(lstm_proba.flatten()[-1])
                    else:
                        lstm_proba = float(lstm_proba[-1])
                except Exception:
                    lstm_proba = 0.5
            else:
                lstm_proba = 0.5

            xgb_p = float(np.asarray(xgb_proba).flatten()[-1])
            rf_p = float(np.asarray(rf_proba).flatten()[-1])

            score = float(self._combiner.predict(xgb_p, rf_p, lstm_proba))
            return float(np.clip(score, 0.0, 100.0))
        except Exception as exc:
            logger.warning("MLPredictor predict failed: %s", exc)
            return 50.0
