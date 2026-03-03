"""
Base signal interface for the algorithmic trading bot.

All concrete signal classes must inherit from BaseSignal and implement
the compute() method. SignalResult carries the computed output together
with enough metadata for logging, back-testing and debugging.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SignalResult:
    """
    Immutable output produced by a single signal computation.

    Attributes
    ----------
    name : str
        Human-readable identifier matching BaseSignal.name.
    layer : int
        Signal layer (1 = technical, 2 = volume, 3 = sentiment, 4 = ML).
    value : float
        Directional score in [-1.0, +1.0].
        Negative values are bearish, positive values are bullish.
    strength : float
        Confidence / conviction in [0.0, 1.0].
        Represents how extreme or reliable the current reading is.
    metadata : dict
        Arbitrary key-value pairs for logging and introspection.
        Each signal defines what it stores here.
    """

    name: str
    layer: int
    value: float
    strength: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Clamp to valid ranges to guard against edge-case numeric drift.
        self.value = max(-1.0, min(1.0, float(self.value)))
        self.strength = max(0.0, min(1.0, float(self.strength)))

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def is_bullish(self) -> bool:
        return self.value > 0.0

    @property
    def is_bearish(self) -> bool:
        return self.value < 0.0

    @property
    def is_neutral(self) -> bool:
        return self.value == 0.0

    @property
    def weighted_value(self) -> float:
        """value * strength — useful for quick aggregation."""
        return self.value * self.strength

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "layer": self.layer,
            "value": round(self.value, 6),
            "strength": round(self.strength, 6),
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:  # pragma: no cover
        direction = "BULL" if self.is_bullish else ("BEAR" if self.is_bearish else "NEUT")
        return (
            f"SignalResult(name={self.name!r}, layer={self.layer}, "
            f"value={self.value:+.4f}, strength={self.strength:.4f}, "
            f"direction={direction})"
        )


class BaseSignal(ABC):
    """
    Abstract base class for every signal in the system.

    Subclasses must define class-level attributes ``name`` and ``layer``,
    and implement the ``compute()`` method.

    Parameters
    ----------
    weight : float, optional
        Relative importance of this signal within its layer.
        The aggregator uses this when computing the weighted layer score.
        Default is 1.0.
    """

    name: str = "base"
    layer: int = 0
    weight: float = 1.0

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def compute(self, data: pd.DataFrame) -> SignalResult:
        """
        Compute the signal from OHLCV data (or sentiment data for layer-3
        signals that override this signature via their own interface).

        Parameters
        ----------
        data : pd.DataFrame
            OHLCV DataFrame.  Expected columns: ``open``, ``high``, ``low``,
            ``close``, ``volume``.  Index should be a DatetimeIndex.

        Returns
        -------
        SignalResult
        """
        ...

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def validate_data(self, data: pd.DataFrame, min_periods: int) -> bool:
        """
        Return True when *data* has enough rows to compute the signal.

        Parameters
        ----------
        data : pd.DataFrame
            Input data.
        min_periods : int
            Minimum number of non-NaN rows required.

        Returns
        -------
        bool
        """
        if data is None or data.empty:
            logger.warning("[%s] Empty DataFrame received.", self.name)
            return False

        if len(data) < min_periods:
            logger.warning(
                "[%s] Insufficient data: need %d rows, got %d.",
                self.name,
                min_periods,
                len(data),
            )
            return False

        return True

    def _neutral_result(self, reason: str = "insufficient data") -> SignalResult:
        """Return a zero-valued neutral SignalResult for fallback paths."""
        logger.debug("[%s] Returning neutral result: %s", self.name, reason)
        return SignalResult(
            name=self.name,
            layer=self.layer,
            value=0.0,
            strength=0.0,
            metadata={"reason": reason},
        )

    @staticmethod
    def _safe_normalize(value: float, lo: float, hi: float) -> float:
        """
        Linearly map *value* from [lo, hi] to [-1.0, +1.0].

        Returns 0.0 when the range is degenerate.
        """
        span = hi - lo
        if span == 0.0:
            return 0.0
        return max(-1.0, min(1.0, 2.0 * (value - lo) / span - 1.0))

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}(name={self.name!r}, layer={self.layer}, weight={self.weight})"
