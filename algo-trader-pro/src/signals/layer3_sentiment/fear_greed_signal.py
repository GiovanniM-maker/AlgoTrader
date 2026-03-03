"""
Layer-3 Sentiment Signal: Fear & Greed Index.

The Fear & Greed Index runs from 0 (Extreme Fear) to 100 (Extreme Greed).
The strategy is *contrarian* at the extremes and neutral in the middle:

    0–20  (Extreme Fear):  value = +1.0   → buy the fear
    20–40 (Fear):          value = +0.5   → mild bullish lean
    40–60 (Neutral):       value =  0.0   → no bias
    60–80 (Greed):         value = -0.3   → slight caution
    80–100 (Extreme Greed):value = -0.8   → sell the greed

Within each zone the value is linearly interpolated between the zone
boundaries so the output is continuous.

Strength = |value - 0| — that is, the magnitude of the current reading's
contrarian bias.  Extreme readings produce high strength.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from src.signals.base_signal import BaseSignal, SignalResult

logger = logging.getLogger(__name__)

# Zone boundaries: (lower_bound, upper_bound, left_value, right_value)
# Interpolation is linear between left_value and right_value across the zone.
_ZONES: list[tuple[float, float, float, float]] = [
    (0.0, 20.0, 1.0, 1.0),    # Extreme Fear  → flat at +1.0
    (20.0, 40.0, 1.0, 0.5),   # Fear          → +1.0 → +0.5
    (40.0, 60.0, 0.5, -0.3),  # Neutral zone  → +0.5 → -0.3
    (60.0, 80.0, -0.3, -0.8), # Greed         → -0.3 → -0.8
    (80.0, 100.0, -0.8, -0.8),# Extreme Greed → flat at -0.8
]


class FearGreedSignal(BaseSignal):
    """
    Fear & Greed Index contrarian sentiment signal.

    This signal does not take a DataFrame; it consumes a single integer
    value provided by the external data provider (Alternative.me or similar).

    The ``compute`` method accepts a DataFrame for interface compatibility
    but the actual entry-point is ``compute_from_value``.
    """

    name: str = "fear_greed"
    layer: int = 3
    weight: float = 1.0

    def __init__(self, weight: float = 1.0) -> None:
        self.weight = weight

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    def compute_from_value(self, fear_greed_value: float) -> SignalResult:
        """
        Compute the signal from a raw Fear & Greed index value.

        Parameters
        ----------
        fear_greed_value : float
            Raw index value in [0, 100].

        Returns
        -------
        SignalResult
        """
        fg = float(np.clip(fear_greed_value, 0.0, 100.0))

        value: float = self._map_to_value(fg)
        strength: float = min(1.0, abs(value))

        zone_label: str = self._zone_label(fg)

        logger.debug(
            "[%s] fear_greed=%.1f  zone=%s  value=%+.4f  strength=%.4f",
            self.name, fg, zone_label, value, strength,
        )

        return SignalResult(
            name=self.name,
            layer=self.layer,
            value=value,
            strength=strength,
            metadata={
                "fear_greed_index": round(fg, 2),
                "zone": zone_label,
            },
        )

    # ------------------------------------------------------------------
    # BaseSignal compatibility — aggregator passes sentiment_data dict
    # ------------------------------------------------------------------

    def compute(self, data) -> SignalResult:  # type: ignore[override]
        """
        Compatibility shim.

        When called with a dict (from the aggregator's layer-3 path),
        extracts ``fear_greed`` from the dict.  When called with a
        DataFrame (unlikely for this signal), returns neutral.
        """
        import pandas as pd

        if isinstance(data, dict):
            fg_raw = data.get("fear_greed")
            if fg_raw is None:
                return self._neutral_result("fear_greed key not found in sentiment_data")
            fg_value = fg_raw.get("value", fg_raw) if isinstance(fg_raw, dict) else fg_raw
            return self.compute_from_value(float(fg_value))

        if isinstance(data, pd.DataFrame):
            return self._neutral_result("FearGreedSignal expects a sentiment dict, not a DataFrame")

        return self._neutral_result("unsupported input type")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _map_to_value(fg: float) -> float:
        """Linearly interpolate within the matching zone."""
        for lo, hi, v_lo, v_hi in _ZONES:
            if lo <= fg <= hi:
                if hi == lo:
                    return v_lo
                t = (fg - lo) / (hi - lo)
                return float(v_lo + t * (v_hi - v_lo))
        # Should never reach here after clipping
        return 0.0

    @staticmethod
    def _zone_label(fg: float) -> str:
        if fg <= 20:
            return "extreme_fear"
        if fg <= 40:
            return "fear"
        if fg <= 60:
            return "neutral"
        if fg <= 80:
            return "greed"
        return "extreme_greed"
