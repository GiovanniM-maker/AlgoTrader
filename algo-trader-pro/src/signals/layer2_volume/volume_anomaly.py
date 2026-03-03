"""
Layer-2 Volume Signal: Volume Anomaly / Whale Detection.

High-volume candles relative to recent history indicate institutional
activity (whale trades).  The price direction on the anomalous candle
determines whether the signal is bullish or bearish.

Algorithm
---------
1. Compute a 20-period rolling z-score for volume:
       z = (volume - rolling_mean(20)) / rolling_std(20)

2. Price direction for the current bar:
       up   if close > open
       down if close < open
       flat otherwise

3. Signal:
       z > 2.5 AND price up   → value = +min(z / 3.0, 1.0)
       z > 2.5 AND price down → value = -min(z / 3.0, 1.0)
       z ≤ 2.5                → value = 0.0  (no anomaly)

4. Strength = |value| (so strength equals the capped z-ratio)

The z-score threshold of 2.5 corresponds to ~1.2% of normally-distributed
volume observations appearing as anomalies — a pragmatic balance between
sensitivity and false-positive rate.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.signals.base_signal import BaseSignal, SignalResult

logger = logging.getLogger(__name__)

_MIN_PERIODS: int = 25
_ROLLING_WINDOW: int = 20
_Z_THRESHOLD: float = 2.5
_Z_SATURATION: float = 3.0   # z-score at which value saturates to ±1.0


class VolumeAnomalySignal(BaseSignal):
    """
    Volume z-score whale-detection signal.
    """

    name: str = "volume_anomaly"
    layer: int = 2
    weight: float = 1.0

    def __init__(
        self,
        window: int = _ROLLING_WINDOW,
        z_threshold: float = _Z_THRESHOLD,
        weight: float = 1.0,
    ) -> None:
        self.window = window
        self.z_threshold = z_threshold
        self.weight = weight

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def compute(self, data: pd.DataFrame) -> SignalResult:
        """
        Compute the volume anomaly signal.

        Parameters
        ----------
        data : pd.DataFrame
            Must contain ``open``, ``close``, ``volume``.

        Returns
        -------
        SignalResult
            metadata keys: volume_zscore, direction, is_anomaly,
                           volume, rolling_mean_volume, rolling_std_volume
        """
        if not self.validate_data(data, min_periods=_MIN_PERIODS):
            return self._neutral_result("insufficient data for volume anomaly")

        for col in ("open", "close", "volume"):
            if col not in data.columns:
                return self._neutral_result(f"missing '{col}' column")

        volume = data["volume"].astype(float)
        close = data["close"].astype(float)
        open_ = data["open"].astype(float)

        # ---- Rolling z-score ----
        roll_mean = volume.rolling(self.window).mean()
        roll_std = volume.rolling(self.window).std(ddof=1)

        vol_now: float = float(volume.iloc[-1])
        mean_now: float = float(roll_mean.iloc[-1])
        std_now: float = float(roll_std.iloc[-1])

        if np.isnan(mean_now) or np.isnan(std_now) or std_now <= 0.0:
            return self._neutral_result("cannot compute volume z-score")

        z_score: float = (vol_now - mean_now) / std_now

        # ---- Price direction ----
        close_now: float = float(close.iloc[-1])
        open_now: float = float(open_.iloc[-1])

        if close_now > open_now:
            direction: str = "up"
        elif close_now < open_now:
            direction = "down"
        else:
            direction = "flat"

        # ---- Signal ----
        is_anomaly: bool = z_score > self.z_threshold

        if is_anomaly and direction == "up":
            value: float = min(z_score / _Z_SATURATION, 1.0)
        elif is_anomaly and direction == "down":
            value = -min(z_score / _Z_SATURATION, 1.0)
        else:
            value = 0.0

        strength: float = abs(value)

        logger.debug(
            "[%s] vol=%.0f  z=%.4f  dir=%s  anomaly=%s  "
            "value=%+.4f  strength=%.4f",
            self.name, vol_now, z_score, direction, is_anomaly, value, strength,
        )

        return SignalResult(
            name=self.name,
            layer=self.layer,
            value=value,
            strength=strength,
            metadata={
                "volume_zscore": round(z_score, 4),
                "direction": direction,
                "is_anomaly": is_anomaly,
                "volume": round(vol_now, 2),
                "rolling_mean_volume": round(mean_now, 2),
                "rolling_std_volume": round(std_now, 2),
            },
        )
