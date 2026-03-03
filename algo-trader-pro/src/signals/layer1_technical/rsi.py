"""
Layer-1 Technical Signal: Relative Strength Index (RSI-14).

Strategy
--------
* RSI < 30  → oversold  → bullish  (+1.0)
* RSI > 70  → overbought → bearish (-1.0)
* 30-70     → linear interpolation centred at 50 → 0.0

Strength is proportional to the distance from the neutral 50 level, scaled
so that the extreme thresholds (30 / 70) yield a strength of 1.0.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pandas_ta as ta  # type: ignore[import]

from src.signals.base_signal import BaseSignal, SignalResult

logger = logging.getLogger(__name__)

_MIN_PERIODS: int = 20   # RSI(14) needs at least 15 rows; we add a margin
_RSI_PERIOD: int = 14
_OVERSOLD: float = 30.0
_OVERBOUGHT: float = 70.0
_MIDPOINT: float = 50.0


class RSISignal(BaseSignal):
    """
    RSI-14 mean-reversion signal.

    The signal is purely contrarian: extreme oversold readings are scored as
    strongly bullish and extreme overbought readings as strongly bearish.
    The intermediate zone is linearly interpolated so that the signal is
    continuous across the full 0-100 RSI range.
    """

    name: str = "rsi"
    layer: int = 1
    weight: float = 1.0

    def __init__(self, period: int = _RSI_PERIOD, weight: float = 1.0) -> None:
        self.period = period
        self.weight = weight

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def compute(self, data: pd.DataFrame) -> SignalResult:
        """
        Compute the RSI signal from an OHLCV DataFrame.

        Parameters
        ----------
        data : pd.DataFrame
            Must contain a ``close`` column.  Index ordering is assumed
            ascending (oldest first).

        Returns
        -------
        SignalResult
            value   : float in [-1.0, +1.0]
            strength: float in [ 0.0,  1.0]
            metadata: {rsi_value: float, period: int}
        """
        if not self.validate_data(data, min_periods=_MIN_PERIODS):
            return self._neutral_result("insufficient data for RSI")

        if "close" not in data.columns:
            logger.error("[%s] 'close' column missing from DataFrame.", self.name)
            return self._neutral_result("missing 'close' column")

        rsi_series: pd.Series = data.ta.rsi(length=self.period)

        if rsi_series is None or rsi_series.dropna().empty:
            logger.warning("[%s] pandas_ta returned no RSI values.", self.name)
            return self._neutral_result("RSI computation returned NaN")

        rsi: float = float(rsi_series.iloc[-1])

        if np.isnan(rsi):
            return self._neutral_result("RSI value is NaN")

        value, strength = self._score(rsi)

        logger.debug(
            "[%s] RSI=%.2f  →  value=%+.4f  strength=%.4f",
            self.name, rsi, value, strength,
        )

        return SignalResult(
            name=self.name,
            layer=self.layer,
            value=value,
            strength=strength,
            metadata={"rsi_value": round(rsi, 4), "period": self.period},
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _score(rsi: float) -> tuple[float, float]:
        """
        Map an RSI reading to (value, strength).

        Zones
        -----
        [0,  30)  oversold  → value   +1.0  to  0.0 as RSI rises to 30
                             → strength (30 - rsi) / 30
        [30, 50)  weak bull → value   +0.0  to  0.0 (linear, centred at 50)
                             → strength proportional to |rsi - 50| / 20
        (50, 70]  weak bear → same logic, mirrored
        (70, 100] overbought→ value  -1.0, strength (rsi - 70) / 30
        """
        rsi = float(np.clip(rsi, 0.0, 100.0))

        if rsi <= _OVERSOLD:
            # Oversold zone: strongest bullish signal at RSI=0
            value = 1.0
            strength = (_OVERSOLD - rsi) / _OVERSOLD          # 0→1 as RSI→0
            strength = max(0.0, min(1.0, strength))
        elif rsi >= _OVERBOUGHT:
            # Overbought zone: strongest bearish signal at RSI=100
            value = -1.0
            strength = (rsi - _OVERBOUGHT) / _OVERSOLD        # 0→1 as RSI→100
            strength = max(0.0, min(1.0, strength))
        else:
            # Neutral zone: linear interpolation from +1→0→-1
            # Maps 30→+1, 50→0, 70→-1 using a simple linear formula
            value = (_MIDPOINT - rsi) / (_MIDPOINT - _OVERSOLD)   # +1 at 30, -1 at 70
            value = float(np.clip(value, -1.0, 1.0))
            # Strength = distance from centre, normalised so edges (30/70) = 1.0
            strength = abs(rsi - _MIDPOINT) / (_MIDPOINT - _OVERSOLD)
            strength = float(np.clip(strength, 0.0, 1.0))

        return value, strength
