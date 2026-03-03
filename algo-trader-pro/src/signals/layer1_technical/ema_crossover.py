"""
Layer-1 Technical Signal: Triple EMA Crossover (9 / 21 / 50).

Strategy
--------
Full bullish alignment:  EMA9 > EMA21 > EMA50  AND  close > EMA9  → +1.0
Full bearish alignment:  EMA9 < EMA21 < EMA50  AND  close < EMA9  → -1.0

Partial alignment scores
------------------------
Each of the four conditions contributes a partial score:

    conditions = [
        EMA9  vs EMA21  (+0.25 or -0.25),
        EMA21 vs EMA50  (+0.25 or -0.25),
        close vs EMA9   (+0.25 or -0.25),
        slope of EMA9   (+0.25 or -0.25),   ← directional momentum
    ]

    value = sum(conditions)   ∈ [-1.0, +1.0]

Strength
--------
Strength is the normalised spread between EMA9 and EMA50 relative to ATR(14).
This tells us *how far* the short-term average has moved from the long-term
average, giving a conviction measure that is scale-independent.

Crossover bonus
---------------
If EMA9 crossed above EMA21 within the last 3 bars: +0.20 to strength.
If EMA9 crossed below EMA21 within the last 3 bars: +0.20 to strength
(crossovers are significant regardless of direction).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pandas_ta as ta  # type: ignore[import]

from src.signals.base_signal import BaseSignal, SignalResult

logger = logging.getLogger(__name__)

_MIN_PERIODS: int = 60
_EMA_FAST: int = 9
_EMA_MED: int = 21
_EMA_SLOW: int = 50
_ATR_PERIOD: int = 14
_CROSSOVER_LOOKBACK: int = 3
_CROSSOVER_STRENGTH_BONUS: float = 0.20
_SPREAD_TANH_SCALE: float = 2.0


class EMACrossoverSignal(BaseSignal):
    """
    Triple EMA crossover directional signal.
    """

    name: str = "ema_crossover"
    layer: int = 1
    weight: float = 1.0

    def __init__(
        self,
        fast: int = _EMA_FAST,
        medium: int = _EMA_MED,
        slow: int = _EMA_SLOW,
        atr_period: int = _ATR_PERIOD,
        weight: float = 1.0,
    ) -> None:
        self.fast = fast
        self.medium = medium
        self.slow = slow
        self.atr_period = atr_period
        self.weight = weight

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def compute(self, data: pd.DataFrame) -> SignalResult:
        """
        Compute the EMA crossover signal from an OHLCV DataFrame.

        Parameters
        ----------
        data : pd.DataFrame
            Must contain ``close``, ``high``, ``low``.

        Returns
        -------
        SignalResult
        """
        if not self.validate_data(data, min_periods=_MIN_PERIODS):
            return self._neutral_result("insufficient data for EMA crossover")

        if "close" not in data.columns:
            return self._neutral_result("missing 'close' column")

        # ---- Compute EMAs ----
        ema_fast = data.ta.ema(length=self.fast)
        ema_med = data.ta.ema(length=self.medium)
        ema_slow = data.ta.ema(length=self.slow)

        if any(s is None or s.dropna().empty for s in (ema_fast, ema_med, ema_slow)):
            return self._neutral_result("EMA computation failed")

        e9: float = float(ema_fast.iloc[-1])
        e21: float = float(ema_med.iloc[-1])
        e50: float = float(ema_slow.iloc[-1])
        close: float = float(data["close"].iloc[-1])

        if any(np.isnan(x) for x in (e9, e21, e50, close)):
            return self._neutral_result("NaN in EMA values")

        # EMA9 slope: compare current vs prior bar
        prev_e9: float = float(ema_fast.iloc[-2]) if len(ema_fast.dropna()) >= 2 else e9

        # ---- Partial condition scoring ----
        cond_e9_vs_e21: float = 0.25 if e9 > e21 else -0.25
        cond_e21_vs_e50: float = 0.25 if e21 > e50 else -0.25
        cond_close_vs_e9: float = 0.25 if close > e9 else -0.25
        cond_slope: float = 0.25 if e9 > prev_e9 else -0.25

        value: float = float(
            np.clip(
                cond_e9_vs_e21 + cond_e21_vs_e50 + cond_close_vs_e9 + cond_slope,
                -1.0, 1.0,
            )
        )

        # ---- Strength: spread between fast and slow EMAs, ATR-normalised ----
        atr_series = data.ta.atr(length=self.atr_period)
        atr: float = (
            float(atr_series.iloc[-1])
            if atr_series is not None and not np.isnan(atr_series.iloc[-1])
            else close * 0.01
        )

        spread: float = abs(e9 - e50) / max(atr, 1e-9)
        base_strength: float = float(np.tanh(_SPREAD_TANH_SCALE * spread / 10.0))

        # ---- Crossover bonus ----
        crossover: bool = self._detect_crossover(ema_fast, ema_med, lookback=_CROSSOVER_LOOKBACK)
        strength: float = min(1.0, base_strength + (_CROSSOVER_STRENGTH_BONUS if crossover else 0.0))

        logger.debug(
            "[%s] EMA9=%.4f EMA21=%.4f EMA50=%.4f close=%.4f  "
            "value=%+.4f  strength=%.4f  crossover=%s",
            self.name, e9, e21, e50, close, value, strength, crossover,
        )

        return SignalResult(
            name=self.name,
            layer=self.layer,
            value=value,
            strength=strength,
            metadata={
                "ema_fast": round(e9, 6),
                "ema_medium": round(e21, 6),
                "ema_slow": round(e50, 6),
                "close": round(close, 6),
                "spread_atr_ratio": round(spread, 4),
                "crossover": crossover,
                "conditions": {
                    "ema9_vs_ema21": cond_e9_vs_e21 > 0,
                    "ema21_vs_ema50": cond_e21_vs_e50 > 0,
                    "close_vs_ema9": cond_close_vs_e9 > 0,
                    "ema9_slope_up": cond_slope > 0,
                },
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_crossover(fast: pd.Series, slow: pd.Series, lookback: int) -> bool:
        """
        Return True if the fast EMA crossed the slow EMA within the last
        *lookback* bars.

        A crossover is defined as a sign change in (fast - slow).
        """
        diff = fast - slow
        diff_clean = diff.dropna()
        if len(diff_clean) < lookback + 1:
            return False
        recent = diff_clean.iloc[-(lookback + 1):]
        signs = np.sign(recent.values)
        # Any adjacent pair with opposite signs → crossover
        return bool(np.any(signs[:-1] * signs[1:] < 0))
