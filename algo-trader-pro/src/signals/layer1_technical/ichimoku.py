"""
Layer-1 Technical Signal: Ichimoku Kinko Hyo (9, 26, 52).

Three independent binary conditions each contribute a fraction of the
final [-1, +1] score:

    1. Price position vs Cloud (+0.33 / -0.33)
    2. TK cross — Tenkan-sen vs Kijun-sen (+0.33 / -0.33)
    3. Chikou span vs price 26 periods ago (+0.34 / -0.34)

All three bullish → value = +1.0
All three bearish → value = -1.0
Mixed              → proportional intermediate value

Strength = fraction of conditions that agree with the dominant direction.

pandas_ta Ichimoku usage
------------------------
pandas_ta returns two DataFrames: (span_df, no_lookahead_span_df).
The first DataFrame contains:
    ISA_9  : Senkou Span A  (cloud upper / leading span A)
    ISB_26 : Senkou Span B  (cloud lower / leading span B)
    ITS_9  : Tenkan-sen     (conversion line)
    IKS_26 : Kijun-sen      (base line)
    ICS_26 : Chikou span    (lagging span, shifted back)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pandas_ta as ta  # type: ignore[import]

from src.signals.base_signal import BaseSignal, SignalResult

logger = logging.getLogger(__name__)

_MIN_PERIODS: int = 110   # 52 for cloud + 26 chikou lookback + buffer
_TENKAN: int = 9
_KIJUN: int = 26
_SENKOU: int = 52

# Contribution weights (must sum to 1.0 for value to reach ±1.0)
_W_CLOUD: float = 0.33
_W_TK: float = 0.33
_W_CHIKOU: float = 0.34


class IchimokuSignal(BaseSignal):
    """
    Ichimoku Cloud three-condition composite signal.
    """

    name: str = "ichimoku"
    layer: int = 1
    weight: float = 1.0

    def __init__(
        self,
        tenkan: int = _TENKAN,
        kijun: int = _KIJUN,
        senkou: int = _SENKOU,
        weight: float = 1.0,
    ) -> None:
        self.tenkan = tenkan
        self.kijun = kijun
        self.senkou = senkou
        self.weight = weight

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def compute(self, data: pd.DataFrame) -> SignalResult:
        """
        Compute the Ichimoku composite signal.

        Parameters
        ----------
        data : pd.DataFrame
            Must contain ``high``, ``low``, ``close``.

        Returns
        -------
        SignalResult
            metadata keys: cloud_position, tk_cross, chikou_confirmation,
                           tenkan, kijun, span_a, span_b, chikou
        """
        if not self.validate_data(data, min_periods=_MIN_PERIODS):
            return self._neutral_result("insufficient data for Ichimoku")

        for col in ("high", "low", "close"):
            if col not in data.columns:
                return self._neutral_result(f"missing '{col}' column")

        # ---- Compute Ichimoku ----
        try:
            result = data.ta.ichimoku(
                tenkan=self.tenkan,
                kijun=self.kijun,
                senkou=self.senkou,
                lookahead=False,
            )
        except Exception as exc:
            logger.error("[%s] pandas_ta Ichimoku error: %s", self.name, exc)
            return self._neutral_result("Ichimoku computation error")

        # pandas_ta returns a tuple of DataFrames when lookahead=False
        if isinstance(result, tuple):
            ichi_df = result[0]
        else:
            ichi_df = result

        if ichi_df is None or ichi_df.empty:
            return self._neutral_result("Ichimoku DataFrame is empty")

        # ---- Extract columns (resilient to minor pandas_ta version diffs) ----
        def _col(prefix: str) -> str | None:
            for c in ichi_df.columns:
                if c.startswith(prefix):
                    return c
            return None

        tenkan_col = _col("ITS")
        kijun_col = _col("IKS")
        span_a_col = _col("ISA")
        span_b_col = _col("ISB")
        chikou_col = _col("ICS")

        missing = [
            name
            for name, col in {
                "ITS (Tenkan)": tenkan_col,
                "IKS (Kijun)": kijun_col,
                "ISA (Span A)": span_a_col,
                "ISB (Span B)": span_b_col,
            }.items()
            if col is None
        ]
        if missing:
            logger.error("[%s] Missing Ichimoku columns: %s", self.name, missing)
            return self._neutral_result("Ichimoku columns not found")

        # Retrieve the last complete row
        last_idx = ichi_df[tenkan_col].last_valid_index()
        if last_idx is None:
            return self._neutral_result("all Ichimoku values are NaN")

        row = ichi_df.loc[last_idx]
        tenkan_val: float = float(row[tenkan_col])
        kijun_val: float = float(row[kijun_col])
        span_a_val: float = float(row[span_a_col])
        span_b_val: float = float(row[span_b_col])

        close: float = float(data["close"].loc[last_idx] if last_idx in data.index else data["close"].iloc[-1])

        if any(np.isnan(x) for x in (tenkan_val, kijun_val, span_a_val, span_b_val, close)):
            return self._neutral_result("NaN in Ichimoku values")

        # ---- Condition 1: price vs cloud ----
        cloud_top = max(span_a_val, span_b_val)
        cloud_bot = min(span_a_val, span_b_val)

        if close > cloud_top:
            cloud_score = _W_CLOUD      # above cloud → bullish
            cloud_position = "above"
        elif close < cloud_bot:
            cloud_score = -_W_CLOUD     # below cloud → bearish
            cloud_position = "below"
        else:
            cloud_score = 0.0           # inside cloud → neutral
            cloud_position = "inside"

        # ---- Condition 2: TK cross ----
        tk_bullish: bool = tenkan_val > kijun_val
        tk_score: float = _W_TK if tk_bullish else -_W_TK
        tk_cross: bool = tk_bullish

        # ---- Condition 3: Chikou confirmation ----
        chikou_confirmation: bool = False
        chikou_score: float = 0.0

        if chikou_col is not None and chikou_col in ichi_df.columns:
            chikou_series = ichi_df[chikou_col].dropna()
            if not chikou_series.empty:
                chikou_val: float = float(chikou_series.iloc[-1])
                # Chikou span is the close shifted back kijun periods; compare
                # it to the close at that time (price 26 bars ago).
                lookback = self.kijun
                if len(data) > lookback:
                    price_then: float = float(data["close"].iloc[-1 - lookback])
                    if not np.isnan(chikou_val) and not np.isnan(price_then):
                        chikou_confirmation = chikou_val > price_then
                        chikou_score = _W_CHIKOU if chikou_confirmation else -_W_CHIKOU

        value: float = float(np.clip(cloud_score + tk_score + chikou_score, -1.0, 1.0))

        # Strength: fraction of conditions aligned with the dominant direction
        signed_conditions = [cloud_score, tk_score, chikou_score]
        if value >= 0:
            aligned = sum(1 for s in signed_conditions if s > 0)
        else:
            aligned = sum(1 for s in signed_conditions if s < 0)
        strength: float = aligned / 3.0

        logger.debug(
            "[%s] cloud=%s  TK=%s  chikou=%s  "
            "value=%+.4f  strength=%.4f",
            self.name, cloud_position, tk_cross, chikou_confirmation, value, strength,
        )

        return SignalResult(
            name=self.name,
            layer=self.layer,
            value=value,
            strength=strength,
            metadata={
                "cloud_position": cloud_position,
                "tk_cross": tk_cross,
                "chikou_confirmation": chikou_confirmation,
                "tenkan": round(tenkan_val, 6),
                "kijun": round(kijun_val, 6),
                "span_a": round(span_a_val, 6),
                "span_b": round(span_b_val, 6),
                "cloud_top": round(cloud_top, 6),
                "cloud_bot": round(cloud_bot, 6),
                "close": round(close, 6),
            },
        )
