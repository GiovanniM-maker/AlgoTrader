"""
Layer-1 Technical Signal: VWAP Deviation.

VWAP is the intraday volume-weighted average price; it is used by
institutional traders as a benchmark.  Prices significantly below VWAP
suggest undervaluation (buy pressure likely), prices significantly above
VWAP suggest overvaluation (sell pressure likely).

Signal construction
-------------------
We compute the VWAP deviation in rolling-standard-deviation units:

    z = (close - vwap) / rolling_std(close, window=20)

Mapping:
    z ≤ -1.5  →  value = +1.0   (price well below VWAP = undervalued)
    z ≥ +1.5  →  value = -1.0   (price well above VWAP = overvalued)
    -1.5 < z < +1.5  →  linear interpolation

Strength = min(|z| / 1.5, 1.0)

Notes on VWAP with daily OHLCV data
-------------------------------------
VWAP is strictly an intraday concept.  When computed on daily OHLCV it
becomes a rolling VWAP (equivalent to a volume-weighted moving average).
pandas_ta's `vwap` implementation requires a DatetimeIndex; if that is
not available we fall back to a manual computation.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pandas_ta as ta  # type: ignore[import]

from src.signals.base_signal import BaseSignal, SignalResult

logger = logging.getLogger(__name__)

_MIN_PERIODS: int = 25
_STD_WINDOW: int = 20
_Z_THRESHOLD: float = 1.5   # std-dev units at which signal saturates to ±1.0


class VWAPSignal(BaseSignal):
    """
    VWAP deviation z-score signal.
    """

    name: str = "vwap"
    layer: int = 1
    weight: float = 1.0

    def __init__(self, std_window: int = _STD_WINDOW, weight: float = 1.0) -> None:
        self.std_window = std_window
        self.weight = weight

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def compute(self, data: pd.DataFrame) -> SignalResult:
        """
        Compute the VWAP deviation signal.

        Parameters
        ----------
        data : pd.DataFrame
            Must contain ``high``, ``low``, ``close``, ``volume``.
            A DatetimeIndex is preferred so pandas_ta can compute VWAP
            correctly; otherwise a manual rolling VWAP is used.

        Returns
        -------
        SignalResult
            metadata keys: vwap, close, z_score, std_window
        """
        if not self.validate_data(data, min_periods=_MIN_PERIODS):
            return self._neutral_result("insufficient data for VWAP")

        for col in ("high", "low", "close", "volume"):
            if col not in data.columns:
                return self._neutral_result(f"missing '{col}' column")

        vwap_series = self._compute_vwap(data)

        if vwap_series is None or vwap_series.dropna().empty:
            return self._neutral_result("VWAP computation returned no values")

        vwap: float = float(vwap_series.iloc[-1])
        close: float = float(data["close"].iloc[-1])

        if np.isnan(vwap) or np.isnan(close):
            return self._neutral_result("NaN in VWAP or close")

        # Rolling standard deviation of close
        rolling_std: float = float(data["close"].rolling(self.std_window).std().iloc[-1])
        if np.isnan(rolling_std) or rolling_std <= 0.0:
            rolling_std = float(data["close"].std())
        if rolling_std <= 0.0:
            return self._neutral_result("zero standard deviation")

        z_score: float = (close - vwap) / rolling_std

        value, strength = self._score(z_score)

        logger.debug(
            "[%s] close=%.4f  vwap=%.4f  z=%.4f  "
            "value=%+.4f  strength=%.4f",
            self.name, close, vwap, z_score, value, strength,
        )

        return SignalResult(
            name=self.name,
            layer=self.layer,
            value=value,
            strength=strength,
            metadata={
                "vwap": round(vwap, 6),
                "close": round(close, 6),
                "z_score": round(z_score, 6),
                "rolling_std": round(rolling_std, 6),
                "std_window": self.std_window,
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_vwap(self, data: pd.DataFrame) -> pd.Series | None:
        """
        Attempt to compute VWAP via pandas_ta; fall back to a manual
        rolling VWAP if the index is not a DatetimeIndex.
        """
        try:
            if isinstance(data.index, pd.DatetimeIndex):
                result = data.ta.vwap()
                if result is not None and not result.dropna().empty:
                    return result
        except Exception as exc:
            logger.debug("[%s] pandas_ta VWAP failed (%s); using manual.", self.name, exc)

        # Manual rolling VWAP: typical_price * volume / rolling_volume
        typical = (data["high"] + data["low"] + data["close"]) / 3.0
        vol = data["volume"].replace(0, np.nan)
        tpv = typical * vol
        rolling_tpv = tpv.rolling(self.std_window).sum()
        rolling_vol = vol.rolling(self.std_window).sum()
        return rolling_tpv / rolling_vol

    @staticmethod
    def _score(z: float) -> tuple[float, float]:
        """Map a z-score to (value, strength)."""
        # Bullish zone: price significantly below VWAP
        if z <= -_Z_THRESHOLD:
            value = 1.0
            strength = min(1.0, abs(z) / _Z_THRESHOLD)
            return value, strength

        # Bearish zone: price significantly above VWAP
        if z >= _Z_THRESHOLD:
            value = -1.0
            strength = min(1.0, abs(z) / _Z_THRESHOLD)
            return value, strength

        # Linear interpolation in the neutral zone
        value = float(np.clip(-z / _Z_THRESHOLD, -1.0, 1.0))
        strength = abs(z) / _Z_THRESHOLD
        return value, strength
