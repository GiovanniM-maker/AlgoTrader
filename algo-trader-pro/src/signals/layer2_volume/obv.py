"""
Layer-2 Volume Signal: On-Balance Volume (OBV) Trend.

OBV is a running total of signed volume: +volume when price closes higher,
-volume when price closes lower.  A rising OBV trend indicates net buying
pressure; a falling trend indicates net selling pressure.

Algorithm
---------
1. Compute OBV via pandas_ta.
2. Fit a linear regression (least-squares slope) over the last 10 OBV values.
3. Normalise the slope by the current price level so the signal is
   scale-independent across assets:
       normalised_slope = slope / close_price
4. Map to value via tanh squashing (smooth, bounded):
       value = tanh(normalised_slope * scale_factor)
5. Strength = |value|

The scale_factor is empirically tuned so that a normalised slope of 0.5
gives a value of ~0.76 (a strong but not saturated signal).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pandas_ta as ta  # type: ignore[import]

from src.signals.base_signal import BaseSignal, SignalResult

logger = logging.getLogger(__name__)

_MIN_PERIODS: int = 20
_REGRESSION_WINDOW: int = 10
_TANH_SCALE: float = 2.0   # controls steepness of value curve


class OBVSignal(BaseSignal):
    """
    OBV linear-regression-slope signal.
    """

    name: str = "obv"
    layer: int = 2
    weight: float = 1.0

    def __init__(
        self,
        regression_window: int = _REGRESSION_WINDOW,
        weight: float = 1.0,
    ) -> None:
        self.regression_window = regression_window
        self.weight = weight

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def compute(self, data: pd.DataFrame) -> SignalResult:
        """
        Compute the OBV trend signal.

        Parameters
        ----------
        data : pd.DataFrame
            Must contain ``close``, ``volume``.

        Returns
        -------
        SignalResult
            metadata keys: obv_current, obv_slope, normalised_slope,
                           regression_window, close
        """
        if not self.validate_data(data, min_periods=_MIN_PERIODS):
            return self._neutral_result("insufficient data for OBV")

        for col in ("close", "volume"):
            if col not in data.columns:
                return self._neutral_result(f"missing '{col}' column")

        # ---- Compute OBV ----
        obv_series = data.ta.obv()

        if obv_series is None or obv_series.dropna().empty:
            logger.warning("[%s] pandas_ta OBV failed; computing manually.", self.name)
            obv_series = self._manual_obv(data)

        if obv_series is None or len(obv_series.dropna()) < self.regression_window:
            return self._neutral_result("not enough OBV values for regression")

        obv_clean = obv_series.dropna()
        obv_current: float = float(obv_clean.iloc[-1])
        close: float = float(data["close"].iloc[-1])

        if np.isnan(close) or close <= 0.0:
            return self._neutral_result("invalid close price")

        # ---- Linear regression slope over the last N bars ----
        recent_obv = obv_clean.iloc[-self.regression_window:].values.astype(float)
        slope: float = self._linreg_slope(recent_obv)

        # Normalise by close price to make it asset-agnostic
        normalised_slope: float = slope / close

        # ---- Map to signal value ----
        value: float = float(np.tanh(_TANH_SCALE * normalised_slope))
        value = float(np.clip(value, -1.0, 1.0))
        strength: float = abs(value)

        logger.debug(
            "[%s] OBV=%.0f  slope=%.4f  norm_slope=%.6f  "
            "value=%+.4f  strength=%.4f",
            self.name, obv_current, slope, normalised_slope, value, strength,
        )

        return SignalResult(
            name=self.name,
            layer=self.layer,
            value=value,
            strength=strength,
            metadata={
                "obv_current": round(obv_current, 2),
                "obv_slope": round(slope, 4),
                "normalised_slope": round(normalised_slope, 8),
                "regression_window": self.regression_window,
                "close": round(close, 6),
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _manual_obv(data: pd.DataFrame) -> pd.Series:
        """Compute OBV without pandas_ta as a fallback."""
        close = data["close"].astype(float)
        volume = data["volume"].astype(float)
        direction = np.sign(close.diff().fillna(0.0))
        signed_vol = direction * volume
        return signed_vol.cumsum()

    @staticmethod
    def _linreg_slope(y: np.ndarray) -> float:
        """
        Compute the least-squares slope of *y* against an integer index.

        Uses the closed-form formula:
            slope = (N * Σxy - Σx * Σy) / (N * Σx² - (Σx)²)
        """
        n = len(y)
        if n < 2:
            return 0.0
        x = np.arange(n, dtype=float)
        sx = x.sum()
        sy = y.sum()
        sxy = (x * y).sum()
        sxx = (x * x).sum()
        denom = n * sxx - sx * sx
        if denom == 0.0:
            return 0.0
        return float((n * sxy - sx * sy) / denom)
