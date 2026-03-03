"""
Layer-2 Volume Signal: Cumulative Volume Delta (CVD).

Since we only have OHLCV data (no tick data), we estimate the buying and
selling volume fractions from the intra-bar price action.

Estimation formula
------------------
For each bar:
    bar_range      = high - low + 1e-9          (avoid division by zero)
    body_fraction  = (close - open) / bar_range  ∈ [-1, +1]

    buying_volume  = volume * max(body_fraction, 0.0)
    selling_volume = volume * max(-body_fraction, 0.0)
    delta          = buying_volume - selling_volume

This is sometimes called the "naive CVD estimator" or the "body CVD".
It is a reasonable proxy when tick data is unavailable.

CVD = cumulative sum of delta over the available bars.

Signal
------
1. Compute CVD over the full window.
2. Fit a linear regression slope over the last 10 bars of CVD.
3. Normalise by close price (as in OBV) for scale-independence.
4. Map via tanh to obtain value ∈ [-1, +1].
5. Strength = |value|.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.signals.base_signal import BaseSignal, SignalResult

logger = logging.getLogger(__name__)

_MIN_PERIODS: int = 20
_REGRESSION_WINDOW: int = 10
_TANH_SCALE: float = 2.0


class CVDSignal(BaseSignal):
    """
    Estimated Cumulative Volume Delta trend signal.
    """

    name: str = "cvd"
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
        Compute the CVD signal from OHLCV data.

        Parameters
        ----------
        data : pd.DataFrame
            Must contain ``open``, ``high``, ``low``, ``close``, ``volume``.

        Returns
        -------
        SignalResult
            metadata keys: cvd_current, cvd_slope, normalised_slope,
                           regression_window, delta_last, close
        """
        if not self.validate_data(data, min_periods=_MIN_PERIODS):
            return self._neutral_result("insufficient data for CVD")

        for col in ("open", "high", "low", "close", "volume"):
            if col not in data.columns:
                return self._neutral_result(f"missing '{col}' column")

        cvd_series = self._compute_cvd(data)

        if len(cvd_series.dropna()) < self.regression_window:
            return self._neutral_result("not enough CVD values for regression")

        close: float = float(data["close"].iloc[-1])
        cvd_current: float = float(cvd_series.iloc[-1])
        delta_last: float = float(self._compute_delta(data).iloc[-1])

        if np.isnan(close) or close <= 0.0:
            return self._neutral_result("invalid close price")

        if np.isnan(cvd_current):
            return self._neutral_result("NaN in CVD")

        # ---- Linear regression slope ----
        recent_cvd = cvd_series.iloc[-self.regression_window:].values.astype(float)
        slope: float = self._linreg_slope(recent_cvd)
        normalised_slope: float = slope / close

        # ---- Map to signal ----
        value: float = float(np.clip(np.tanh(_TANH_SCALE * normalised_slope), -1.0, 1.0))
        strength: float = abs(value)

        logger.debug(
            "[%s] CVD=%.2f  slope=%.4f  norm_slope=%.8f  "
            "value=%+.4f  strength=%.4f",
            self.name, cvd_current, slope, normalised_slope, value, strength,
        )

        return SignalResult(
            name=self.name,
            layer=self.layer,
            value=value,
            strength=strength,
            metadata={
                "cvd_current": round(cvd_current, 4),
                "cvd_slope": round(slope, 6),
                "normalised_slope": round(normalised_slope, 10),
                "regression_window": self.regression_window,
                "delta_last": round(delta_last, 4),
                "close": round(close, 6),
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_delta(data: pd.DataFrame) -> pd.Series:
        """
        Estimate per-bar volume delta from OHLCV.

        delta = volume * (close - open) / (high - low + 1e-9)

        This gives a signed fraction of volume attributed to buyers vs sellers.
        """
        open_ = data["open"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        close = data["close"].astype(float)
        volume = data["volume"].astype(float)

        bar_range = (high - low).clip(lower=1e-9)
        body_fraction = (close - open_) / bar_range
        delta = volume * body_fraction
        return delta

    def _compute_cvd(self, data: pd.DataFrame) -> pd.Series:
        """Return the cumulative sum of per-bar volume delta."""
        return self._compute_delta(data).cumsum()

    @staticmethod
    def _linreg_slope(y: np.ndarray) -> float:
        """Least-squares slope of y against integer index."""
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
