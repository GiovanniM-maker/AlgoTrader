"""
Layer-1 Technical Signal: MACD (12, 26, 9).

Strategy
--------
* Direction of the MACD histogram determines bullish / bearish value.
* Strength is the histogram magnitude normalised by ATR(14) so that the
  scale is meaningful across different assets and timeframes.
* A histogram crossover (sign change on the most recent bar) boosts
  strength by a fixed bonus, capped at 1.0.

Signal value formula
--------------------
    value = sign(histogram)            # +1 or -1
    raw_ratio = abs(histogram) / atr   # how large is the move relative to volatility
    base_strength = tanh(raw_ratio * 3)  # tanh squashes to (0,1), scales smoothly
    strength = min(base_strength + (0.25 if crossover else 0.0), 1.0)

The tanh with scaling factor 3 means:
  - raw_ratio = 0.1  → strength ≈ 0.29
  - raw_ratio = 0.3  → strength ≈ 0.72
  - raw_ratio ≥ 0.5  → strength ≈ 0.91+
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pandas_ta as ta  # type: ignore[import]

from src.signals.base_signal import BaseSignal, SignalResult

logger = logging.getLogger(__name__)

_MIN_PERIODS: int = 60          # MACD(12,26,9) needs 34 + signal; ATR(14) adds more
_FAST: int = 12
_SLOW: int = 26
_SIGNAL: int = 9
_ATR_PERIOD: int = 14
_CROSSOVER_BONUS: float = 0.25  # extra strength when a histogram crossover is detected
_TANH_SCALE: float = 3.0        # controls how quickly strength saturates


class MACDSignal(BaseSignal):
    """
    MACD histogram directional signal with ATR-normalised strength.
    """

    name: str = "macd"
    layer: int = 1
    weight: float = 1.0

    def __init__(
        self,
        fast: int = _FAST,
        slow: int = _SLOW,
        signal: int = _SIGNAL,
        atr_period: int = _ATR_PERIOD,
        weight: float = 1.0,
    ) -> None:
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.atr_period = atr_period
        self.weight = weight

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def compute(self, data: pd.DataFrame) -> SignalResult:
        """
        Compute the MACD signal from an OHLCV DataFrame.

        Parameters
        ----------
        data : pd.DataFrame
            Must contain ``open``, ``high``, ``low``, ``close``, ``volume``.

        Returns
        -------
        SignalResult
            metadata keys: macd, signal_line, histogram, crossover, atr
        """
        if not self.validate_data(data, min_periods=_MIN_PERIODS):
            return self._neutral_result("insufficient data for MACD")

        required = {"high", "low", "close"}
        missing = required - set(data.columns)
        if missing:
            logger.error("[%s] Missing columns: %s", self.name, missing)
            return self._neutral_result(f"missing columns: {missing}")

        # ---- MACD ----
        macd_df = data.ta.macd(fast=self.fast, slow=self.slow, signal=self.signal)
        if macd_df is None or macd_df.empty:
            return self._neutral_result("MACD computation failed")

        # pandas_ta column naming: MACDh_12_26_9, MACD_12_26_9, MACDs_12_26_9
        hist_col = f"MACDh_{self.fast}_{self.slow}_{self.signal}"
        macd_col = f"MACD_{self.fast}_{self.slow}_{self.signal}"
        sig_col = f"MACDs_{self.fast}_{self.slow}_{self.signal}"

        if hist_col not in macd_df.columns:
            # Fallback: pick columns by position if naming changed
            hist_col, macd_col, sig_col = (
                macd_df.columns[1],
                macd_df.columns[0],
                macd_df.columns[2],
            )

        hist_series: pd.Series = macd_df[hist_col].dropna()
        if len(hist_series) < 2:
            return self._neutral_result("not enough MACD histogram values")

        histogram: float = float(hist_series.iloc[-1])
        prev_histogram: float = float(hist_series.iloc[-2])
        macd_val: float = float(macd_df[macd_col].iloc[-1])
        signal_val: float = float(macd_df[sig_col].iloc[-1])

        if any(np.isnan(x) for x in (histogram, macd_val, signal_val)):
            return self._neutral_result("NaN in MACD values")

        # ---- ATR for normalisation ----
        atr_series = data.ta.atr(length=self.atr_period)
        atr: float = float(atr_series.iloc[-1]) if atr_series is not None else 0.0
        if np.isnan(atr) or atr <= 0.0:
            # Fall back to a rough estimate: 1% of close
            atr = float(data["close"].iloc[-1]) * 0.01
            logger.debug("[%s] ATR fallback to 1%% of close = %.6f", self.name, atr)

        # ---- Crossover detection ----
        crossover: bool = (histogram * prev_histogram) < 0.0  # sign change

        # ---- Score ----
        value: float = float(np.sign(histogram)) if histogram != 0.0 else 0.0
        raw_ratio: float = abs(histogram) / atr
        base_strength: float = float(np.tanh(_TANH_SCALE * raw_ratio))
        strength: float = min(1.0, base_strength + (_CROSSOVER_BONUS if crossover else 0.0))

        logger.debug(
            "[%s] hist=%.6f  atr=%.6f  ratio=%.4f  crossover=%s  "
            "value=%+.4f  strength=%.4f",
            self.name, histogram, atr, raw_ratio, crossover, value, strength,
        )

        return SignalResult(
            name=self.name,
            layer=self.layer,
            value=value,
            strength=strength,
            metadata={
                "macd": round(macd_val, 6),
                "signal_line": round(signal_val, 6),
                "histogram": round(histogram, 6),
                "crossover": crossover,
                "atr": round(atr, 6),
                "raw_ratio": round(raw_ratio, 6),
            },
        )
