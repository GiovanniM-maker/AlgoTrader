"""
src/signals/aggregator.py

Signal Aggregator — combines Layer 1 (technical), Layer 2 (volume),
Layer 3 (sentiment), and Layer 4 (ML) signals into a single confidence
score on a 0–100 scale.

Architecture
------------
Each layer contributes a *layer score* computed as the weighted mean of
(signal.value * signal.strength) across all signals in that layer.  The
raw score is then a weighted sum of all layer scores, normalised to the
[-1, +1] range.  Finally the raw score is mapped to [0, 100].

    layer_score_i = mean(signal.value * signal.strength  for signal in layer_i)
    raw_score     = sum(layer_score_i * layer_weight_i)   ∈ [-1, +1]
    confidence    = (raw_score + 1) / 2 * 100             ∈ [0, 100]

Direction thresholds
--------------------
    raw_score > +0.10  → LONG
    raw_score < -0.10  → SHORT
    otherwise          → NEUTRAL
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.signals.base_signal import SignalResult

logger = logging.getLogger(__name__)

# Direction thresholds on the raw score
_LONG_THRESHOLD: float = 0.10
_SHORT_THRESHOLD: float = -0.10


@dataclass
class AggregationResult:
    """
    Full output of a single aggregation run.

    Attributes
    ----------
    confidence_score : float
        Final composite score in [0, 100].
    direction : str
        Trading bias: ``'LONG'``, ``'SHORT'``, or ``'NEUTRAL'``.
    raw_score : float
        Weighted sum of layer scores in [-1, +1].
    layer1_score : float
        Weighted layer score for technical signals.
    layer2_score : float
        Weighted layer score for volume signals.
    layer3_score : float
        Weighted layer score for sentiment signals.
    ml_score : float
        Weighted layer score contributed by the ML ensemble.
    breakdown : dict
        Per-signal values: name → {value, strength, weighted_contribution}.
    """

    confidence_score: float
    direction: str
    raw_score: float
    layer1_score: float
    layer2_score: float
    layer3_score: float
    ml_score: float
    breakdown: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "confidence_score": round(self.confidence_score, 4),
            "direction": self.direction,
            "raw_score": round(self.raw_score, 6),
            "layer1_score": round(self.layer1_score, 6),
            "layer2_score": round(self.layer2_score, 6),
            "layer3_score": round(self.layer3_score, 6),
            "ml_score": round(self.ml_score, 6),
            "breakdown": self.breakdown,
        }

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"AggregationResult("
            f"confidence={self.confidence_score:.2f}, "
            f"direction={self.direction}, "
            f"raw={self.raw_score:+.4f})"
        )


class SignalAggregator:
    """
    Combines signals from all four layers into a single AggregationResult.

    Parameters
    ----------
    config : object
        Must expose a ``strategy.layer_weights`` mapping with keys:
        ``layer1_technical``, ``layer2_volume``, ``layer3_sentiment``,
        ``layer4_ml``.  The weights are normalised internally if they
        do not sum to 1.0.

    Example
    -------
    ::

        result = aggregator.aggregate(
            layer1_signals={"rsi": rsi_result, "macd": macd_result},
            layer2_signals={"volume_anomaly": va_result, "obv": obv_result},
            layer3_signals={"fear_greed": fg_result},
            ml_score=0.72,          # raw probability 0–1 from ensemble
        )
        print(result.confidence_score)   # e.g. 71.3
        print(result.direction)          # LONG / SHORT / NEUTRAL
    """

    def __init__(self, config: Any) -> None:
        raw_weights: Dict[str, float] = {
            "layer1_technical": float(
                getattr(config.strategy.layer_weights, "layer1_technical", 0.30)
                if hasattr(config.strategy.layer_weights, "layer1_technical")
                else config.strategy.layer_weights.get("layer1_technical", 0.30)
            ),
            "layer2_volume": float(
                getattr(config.strategy.layer_weights, "layer2_volume", 0.25)
                if hasattr(config.strategy.layer_weights, "layer2_volume")
                else config.strategy.layer_weights.get("layer2_volume", 0.25)
            ),
            "layer3_sentiment": float(
                getattr(config.strategy.layer_weights, "layer3_sentiment", 0.15)
                if hasattr(config.strategy.layer_weights, "layer3_sentiment")
                else config.strategy.layer_weights.get("layer3_sentiment", 0.15)
            ),
            "layer4_ml": float(
                getattr(config.strategy.layer_weights, "layer4_ml", 0.30)
                if hasattr(config.strategy.layer_weights, "layer4_ml")
                else config.strategy.layer_weights.get("layer4_ml", 0.30)
            ),
        }

        # Normalise so weights sum exactly to 1.0
        total = sum(raw_weights.values())
        if total <= 0.0:
            raise ValueError("Layer weights must sum to a positive number.")
        if abs(total - 1.0) > 0.01:
            logger.warning(
                "Layer weights sum to %.4f (expected 1.0) — normalising automatically.",
                total,
            )
        self._weights: Dict[str, float] = {k: v / total for k, v in raw_weights.items()}

        logger.debug(
            "SignalAggregator initialised with normalised weights: %s",
            {k: round(v, 4) for k, v in self._weights.items()},
        )

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------

    def aggregate(
        self,
        layer1_signals: Dict[str, SignalResult],
        layer2_signals: Dict[str, SignalResult],
        layer3_signals: Dict[str, SignalResult],
        ml_score: Optional[float] = None,
    ) -> AggregationResult:
        """
        Aggregate all signal layers into a single AggregationResult.

        Parameters
        ----------
        layer1_signals : dict
            Mapping of signal_name → SignalResult for technical signals.
        layer2_signals : dict
            Mapping of signal_name → SignalResult for volume signals.
        layer3_signals : dict
            Mapping of signal_name → SignalResult for sentiment signals.
        ml_score : float, optional
            Raw ML ensemble output.  Pass either:
            * A value in [-1, +1] (already scaled), or
            * A confidence percentage in [0, 100] (converted internally via
              ``(ml_confidence / 100) * 2 - 1``).
            If None, the ML layer contributes 0.0 and its weight is
            redistributed proportionally among the other layers.

        Returns
        -------
        AggregationResult
        """
        breakdown: Dict[str, Any] = {}

        # ---- Layer 1 ----
        l1_raw = self._compute_layer_score(layer1_signals, breakdown)

        # ---- Layer 2 ----
        l2_raw = self._compute_layer_score(layer2_signals, breakdown)

        # ---- Layer 3 ----
        l3_raw = self._compute_layer_score(layer3_signals, breakdown)

        # ---- Layer 4 (ML) ----
        # Determine the ML value on [-1, +1] scale
        if ml_score is not None:
            # Accept both probability [0,1] and confidence [0,100] inputs
            if 0.0 <= ml_score <= 1.0:
                ml_value = (ml_score * 2.0) - 1.0
            elif 0.0 <= ml_score <= 100.0:
                ml_value = (ml_score / 100.0) * 2.0 - 1.0
            else:
                # Already on [-1, +1]; clamp for safety
                ml_value = max(-1.0, min(1.0, float(ml_score)))
            breakdown["ml_ensemble"] = {
                "value": round(ml_value, 6),
                "strength": abs(ml_value),
                "raw_input": round(ml_score, 6),
            }
        else:
            ml_value = 0.0
            breakdown["ml_ensemble"] = {
                "value": 0.0,
                "strength": 0.0,
                "raw_input": None,
            }

        # ---- Determine effective weights ----
        # If ML is absent we redistribute its weight proportionally
        if ml_score is None:
            w1, w2, w3, w4 = self._effective_weights(ml_absent=True)
        else:
            w1 = self._weights["layer1_technical"]
            w2 = self._weights["layer2_volume"]
            w3 = self._weights["layer3_sentiment"]
            w4 = self._weights["layer4_ml"]

        # ---- Raw score [-1, +1] ----
        raw_score: float = (
            l1_raw * w1
            + l2_raw * w2
            + l3_raw * w3
            + ml_value * w4
        )
        # Clamp to [-1, +1] in case of floating-point drift
        raw_score = max(-1.0, min(1.0, raw_score))

        # ---- Convert to 0–100 confidence ----
        confidence_score: float = (raw_score + 1.0) / 2.0 * 100.0

        # ---- Direction ----
        if raw_score > _LONG_THRESHOLD:
            direction = "LONG"
        elif raw_score < _SHORT_THRESHOLD:
            direction = "SHORT"
        else:
            direction = "NEUTRAL"

        result = AggregationResult(
            confidence_score=round(confidence_score, 4),
            direction=direction,
            raw_score=round(raw_score, 6),
            layer1_score=round(l1_raw, 6),
            layer2_score=round(l2_raw, 6),
            layer3_score=round(l3_raw, 6),
            ml_score=round(ml_value, 6),
            breakdown=breakdown,
        )

        logger.debug(
            "Aggregation complete: confidence=%.2f direction=%s raw=%.4f "
            "[L1=%.4f L2=%.4f L3=%.4f ML=%.4f]",
            confidence_score,
            direction,
            raw_score,
            l1_raw,
            l2_raw,
            l3_raw,
            ml_value,
        )

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_layer_score(
        signals: Dict[str, SignalResult],
        breakdown: Dict[str, Any],
    ) -> float:
        """
        Compute the weighted-mean contribution for a single layer.

        layer_score = mean(signal.value * signal.strength  for each signal)

        Signals with strength == 0 and value == 0 (neutral/fallback) are
        included in the mean denominator so they dilute weak layers.

        Parameters
        ----------
        signals : dict
            signal_name → SignalResult
        breakdown : dict
            Mutated in-place: adds per-signal entry.

        Returns
        -------
        float in [-1, +1]
        """
        if not signals:
            return 0.0

        contributions: list[float] = []
        for name, result in signals.items():
            weighted_val = result.value * result.strength
            contributions.append(weighted_val)
            breakdown[name] = {
                "value": round(result.value, 6),
                "strength": round(result.strength, 6),
                "weighted_contribution": round(weighted_val, 6),
                "metadata": result.metadata,
            }

        if not contributions:
            return 0.0

        layer_score = sum(contributions) / len(contributions)
        return max(-1.0, min(1.0, layer_score))

    def _effective_weights(
        self,
        ml_absent: bool = False,
    ) -> tuple[float, float, float, float]:
        """
        Return (w1, w2, w3, w4) effective weights.

        When ML is absent, redistribute its weight proportionally among the
        remaining three layers.
        """
        w1 = self._weights["layer1_technical"]
        w2 = self._weights["layer2_volume"]
        w3 = self._weights["layer3_sentiment"]
        w4 = self._weights["layer4_ml"]

        if not ml_absent:
            return w1, w2, w3, w4

        # Redistribute w4 proportionally to w1, w2, w3
        non_ml_total = w1 + w2 + w3
        if non_ml_total <= 0.0:
            return 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0

        scale = (non_ml_total + w4) / non_ml_total
        return w1 * scale, w2 * scale, w3 * scale, 0.0
