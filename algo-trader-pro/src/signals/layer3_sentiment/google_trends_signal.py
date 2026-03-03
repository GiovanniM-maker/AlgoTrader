"""
Layer-3 Sentiment Signal: Google Trends.

Google Trends provides a relative search-interest score (0–100) for a
cryptocurrency keyword.  We interpret this score as a proxy for retail
investor attention.

Hype-cycle model
----------------
Search interest follows a bubble-like arc:

    Rising from low baseline  →  early public awareness      →  mild bullish
    High and still rising     →  peak retail FOMO incoming   →  bullish
    Already at extreme high   →  potential cycle top (hype peak) → mild bearish
    Falling from high         →  interest waning             →  bearish
    Low and stable            →  accumulation phase          →  neutral

Algorithm
---------
1. Compute a 7-period rolling average (≈ 1-week MA for daily data).
2. Acceleration = (current_score - rolling_avg) / max(rolling_avg, 1)
   This tells us how fast interest is growing relative to the recent baseline.
3. Scoring table:

    score > 90  AND acceleration > 0.10  → value = -0.3   (hype peak)
    score > 90                           → value = -0.2   (elevated / late-stage)
    score > 70  AND acceleration > 0.10  → value = +0.5   (rising awareness)
    score > 70                           → value = +0.2   (high but flat)
    score > 50  AND acceleration > 0.15  → value = +0.3   (breakout in interest)
    score > 50                           → value = +0.1   (moderate interest)
    score ≤ 50                           → value =  0.0   (low interest, neutral)

4. Strength = min(|acceleration| + score / 200.0, 1.0)
   This combines how fast things are changing with how elevated interest is.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np

from src.signals.base_signal import BaseSignal, SignalResult

logger = logging.getLogger(__name__)

_MA_WINDOW: int = 7
_HIGH_THRESHOLD: float = 70.0
_EXTREME_THRESHOLD: float = 90.0
_RISING_ACCEL: float = 0.10    # 10% above rolling average counts as "rising"
_FAST_ACCEL: float = 0.15      # 15% above rolling average counts as "fast breakout"


class GoogleTrendsSignal(BaseSignal):
    """
    Google Trends search-interest hype-cycle signal.
    """

    name: str = "google_trends"
    layer: int = 3
    weight: float = 1.0

    def __init__(self, ma_window: int = _MA_WINDOW, weight: float = 1.0) -> None:
        self.ma_window = ma_window
        self.weight = weight

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    def compute_from_series(
        self,
        trend_scores: Sequence[float],
        extra_metadata: Optional[dict] = None,
    ) -> SignalResult:
        """
        Compute the signal from a historical series of Google Trends scores.

        Parameters
        ----------
        trend_scores : sequence of float
            Chronologically ordered Trends scores in [0, 100].
            At least ``ma_window + 1`` values are required for the
            acceleration calculation; fewer values fall back to the
            current score only.
        extra_metadata : dict, optional
            Additional metadata to attach.

        Returns
        -------
        SignalResult
        """
        if not trend_scores:
            return self._neutral_result("no Google Trends data")

        arr = np.array([float(v) for v in trend_scores], dtype=float)
        arr = np.clip(arr, 0.0, 100.0)

        current_score: float = float(arr[-1])

        # Rolling average
        if len(arr) >= self.ma_window:
            rolling_avg: float = float(arr[-self.ma_window:].mean())
        else:
            rolling_avg = float(arr.mean())

        # Acceleration: relative deviation from moving average
        if rolling_avg > 0:
            acceleration: float = (current_score - rolling_avg) / rolling_avg
        else:
            acceleration = 0.0

        value: float = self._score(current_score, acceleration)
        strength: float = min(
            1.0,
            abs(acceleration) + current_score / 200.0,
        )

        logger.debug(
            "[%s] score=%.1f  rolling_avg=%.1f  accel=%.4f  "
            "value=%+.4f  strength=%.4f",
            self.name, current_score, rolling_avg, acceleration, value, strength,
        )

        meta: dict = {
            "trend_score": round(current_score, 2),
            "rolling_avg": round(rolling_avg, 4),
            "acceleration": round(acceleration, 6),
            "ma_window": self.ma_window,
        }
        if extra_metadata:
            meta.update(extra_metadata)

        return SignalResult(
            name=self.name,
            layer=self.layer,
            value=value,
            strength=strength,
            metadata=meta,
        )

    def compute_from_value(
        self,
        trend_score: float,
        rolling_avg: Optional[float] = None,
    ) -> SignalResult:
        """
        Compute the signal from a single current score (and optionally a
        pre-computed rolling average).

        Parameters
        ----------
        trend_score : float
            Current Trends score in [0, 100].
        rolling_avg : float, optional
            Pre-computed rolling average.  If None, acceleration = 0.

        Returns
        -------
        SignalResult
        """
        current_score = float(np.clip(trend_score, 0.0, 100.0))

        if rolling_avg is not None:
            avg = float(rolling_avg)
            acceleration = (current_score - avg) / avg if avg > 0 else 0.0
        else:
            avg = current_score
            acceleration = 0.0

        value = self._score(current_score, acceleration)
        strength = min(1.0, abs(acceleration) + current_score / 200.0)

        return SignalResult(
            name=self.name,
            layer=self.layer,
            value=value,
            strength=strength,
            metadata={
                "trend_score": round(current_score, 2),
                "rolling_avg": round(avg, 4),
                "acceleration": round(acceleration, 6),
            },
        )

    # ------------------------------------------------------------------
    # BaseSignal / aggregator compatibility
    # ------------------------------------------------------------------

    def compute(self, data) -> SignalResult:  # type: ignore[override]
        """
        Aggregator entry-point.

        Accepts a dict with either:
        * ``google_trends_scores`` (list[float]) — historical series, OR
        * ``google_trends_score`` (float) + optional ``google_trends_rolling_avg`` (float)
        """
        import pandas as pd

        if isinstance(data, pd.DataFrame):
            return self._neutral_result("GoogleTrendsSignal expects a sentiment dict")

        if isinstance(data, dict):
            # Prefer historical series
            series = data.get("google_trends_scores")
            if series:
                return self.compute_from_series(series)

            score: Optional[float] = data.get("google_trends_score")
            if score is None:
                return self._neutral_result("google_trends_score key not found in sentiment_data")

            avg: Optional[float] = data.get("google_trends_rolling_avg")
            return self.compute_from_value(float(score), rolling_avg=avg)

        return self._neutral_result("unsupported input type")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _score(score: float, acceleration: float) -> float:
        """Map score + acceleration to a signal value."""
        if score > _EXTREME_THRESHOLD:
            if acceleration > _RISING_ACCEL:
                return -0.3   # Hype peak forming
            return -0.2       # Elevated but not accelerating

        if score > _HIGH_THRESHOLD:
            if acceleration > _RISING_ACCEL:
                return 0.5    # Rising awareness — bullish early signal
            return 0.2        # High but stable

        if score > 50.0:
            if acceleration > _FAST_ACCEL:
                return 0.3    # Fast breakout in interest
            return 0.1        # Moderate interest, slight bullish

        return 0.0            # Low interest — neutral
