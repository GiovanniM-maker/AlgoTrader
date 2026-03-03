"""
Layer-1 Technical Signal: Bollinger Bands (20, 2.0).

Strategy
--------
%B = (close - lower) / (upper - lower)

* %B < 0.05  → price touching / below lower band → bullish  (+1.0)
* %B > 0.95  → price touching / above upper band → bearish  (-1.0)
* 0.05–0.95  → linear interpolation centred at 0.5 → 0.0

Bandwidth modifier
------------------
Bandwidth = (upper - lower) / middle  measures how wide the bands are.
A narrow bandwidth (< 0.05, Squeeze state) signals a potential breakout.
When a squeeze is detected the base strength is boosted by a factor so
the signal is weighted higher by the aggregator.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pandas_ta as ta  # type: ignore[import]

from src.signals.base_signal import BaseSignal, SignalResult

logger = logging.getLogger(__name__)

_MIN_PERIODS: int = 25
_BB_LENGTH: int = 20
_BB_STD: float = 2.0
_LOWER_THRESHOLD: float = 0.05
_UPPER_THRESHOLD: float = 0.95
_SQUEEZE_THRESHOLD: float = 0.05   # bandwidth below this → squeeze state
_SQUEEZE_STRENGTH_BONUS: float = 0.20


class BollingerBandsSignal(BaseSignal):
    """
    Bollinger Bands %B mean-reversion signal with bandwidth awareness.
    """

    name: str = "bollinger_bands"
    layer: int = 1
    weight: float = 1.0

    def __init__(
        self,
        length: int = _BB_LENGTH,
        std: float = _BB_STD,
        weight: float = 1.0,
    ) -> None:
        self.length = length
        self.std = std
        self.weight = weight

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def compute(self, data: pd.DataFrame) -> SignalResult:
        """
        Compute the Bollinger Bands signal.

        Parameters
        ----------
        data : pd.DataFrame
            Must contain ``close``.

        Returns
        -------
        SignalResult
            metadata keys: percent_b, bandwidth, upper, middle, lower,
                           is_squeeze
        """
        if not self.validate_data(data, min_periods=_MIN_PERIODS):
            return self._neutral_result("insufficient data for Bollinger Bands")

        if "close" not in data.columns:
            return self._neutral_result("missing 'close' column")

        bb_df = data.ta.bbands(length=self.length, std=self.std)

        if bb_df is None or bb_df.empty:
            return self._neutral_result("BBands computation failed")

        # pandas_ta column names: BBL_20_2.0, BBM_20_2.0, BBU_20_2.0,
        #                         BBB_20_2.0 (bandwidth), BBP_20_2.0 (%B)
        std_tag = f"{self.std:.1f}"
        lower_col = f"BBL_{self.length}_{std_tag}"
        middle_col = f"BBM_{self.length}_{std_tag}"
        upper_col = f"BBU_{self.length}_{std_tag}"
        bw_col = f"BBB_{self.length}_{std_tag}"
        pct_col = f"BBP_{self.length}_{std_tag}"

        # Tolerate minor naming differences by scanning available columns
        def _find_col(prefix: str) -> str | None:
            for c in bb_df.columns:
                if c.startswith(prefix):
                    return c
            return None

        lower_col = _find_col("BBL") or lower_col
        middle_col = _find_col("BBM") or middle_col
        upper_col = _find_col("BBU") or upper_col
        bw_col = _find_col("BBB") or bw_col
        pct_col = _find_col("BBP") or pct_col

        missing = [
            c for c in (lower_col, middle_col, upper_col)
            if c not in bb_df.columns
        ]
        if missing:
            logger.error("[%s] Missing BB columns: %s", self.name, missing)
            return self._neutral_result("BB columns not found")

        last = bb_df.iloc[-1]
        lower: float = float(last[lower_col])
        middle: float = float(last[middle_col])
        upper: float = float(last[upper_col])
        close: float = float(data["close"].iloc[-1])

        if any(np.isnan(x) for x in (lower, middle, upper)):
            return self._neutral_result("NaN in Bollinger Bands values")

        band_width: float = upper - lower
        if band_width <= 0.0:
            return self._neutral_result("degenerate band width (zero spread)")

        # %B — prefer pandas_ta's computed value, fall back to manual
        if pct_col in bb_df.columns and not np.isnan(last.get(pct_col, float("nan"))):
            percent_b: float = float(last[pct_col])
        else:
            percent_b = (close - lower) / band_width

        # Bandwidth as a fraction of the middle band (Keltner / BB bandwidth)
        bandwidth: float = band_width / middle if middle > 0 else 0.0
        is_squeeze: bool = bandwidth < _SQUEEZE_THRESHOLD

        value, base_strength = self._score(percent_b)

        # Boost strength when bands are in a squeeze (pending breakout)
        strength: float = min(1.0, base_strength + (_SQUEEZE_STRENGTH_BONUS if is_squeeze else 0.0))

        logger.debug(
            "[%s] close=%.4f  %%B=%.4f  bandwidth=%.4f  squeeze=%s  "
            "value=%+.4f  strength=%.4f",
            self.name, close, percent_b, bandwidth, is_squeeze, value, strength,
        )

        return SignalResult(
            name=self.name,
            layer=self.layer,
            value=value,
            strength=strength,
            metadata={
                "percent_b": round(percent_b, 6),
                "bandwidth": round(bandwidth, 6),
                "upper": round(upper, 6),
                "middle": round(middle, 6),
                "lower": round(lower, 6),
                "is_squeeze": is_squeeze,
                "close": round(close, 6),
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _score(percent_b: float) -> tuple[float, float]:
        """
        Map %B → (value, strength).

        Zones
        -----
        %B ≤ 0.05  → value = +1.0   strength = (0.05 - %B) / 0.05, min 0.0
        %B ≥ 0.95  → value = -1.0   strength = (%B - 0.95) / 0.05, min 0.0
        0.05–0.95  → value linear from +1→0→-1 centred at 0.5
                     strength = |value|
        """
        if percent_b <= _LOWER_THRESHOLD:
            value = 1.0
            # Strength saturates quickly below the lower band
            strength = min(1.0, (_LOWER_THRESHOLD - percent_b) / _LOWER_THRESHOLD)
            return value, strength

        if percent_b >= _UPPER_THRESHOLD:
            value = -1.0
            strength = min(1.0, (percent_b - _UPPER_THRESHOLD) / (1.0 - _UPPER_THRESHOLD))
            return value, strength

        # Linear interpolation: 0.5 → 0.0, 0.05 → +1.0, 0.95 → -1.0
        value = (0.5 - percent_b) / (0.5 - _LOWER_THRESHOLD)
        value = float(np.clip(value, -1.0, 1.0))
        strength = abs(value)
        return value, strength
