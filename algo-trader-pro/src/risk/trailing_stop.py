"""
src/risk/trailing_stop.py

Trailing stop manager.

Supports two trailing modes:
    ATR-based  – stop tracks price at a distance of atr * multiplier
    Percentage – stop tracks price at a fixed percentage distance

The core rule is: the stop only ever moves in the profitable direction.
A long position's stop can only move up; a short's can only move down.

Usage
-----
    manager = TrailingStopManager(mode="atr", atr_multiplier=2.0)

    new_stop = manager.update(
        position=position.to_dict(),
        current_price=29450.0,
        atr=320.0,
    )
    if new_stop is not None:
        position.stop_loss = new_stop
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_VALID_MODES = frozenset({"atr", "percentage"})


class TrailingStopManager:
    """
    Manages dynamic (trailing) stop updates for open positions.

    Parameters
    ----------
    mode            : str    – ``'atr'`` or ``'percentage'``
    atr_multiplier  : float  – distance multiplier when mode='atr'   (default 2.0)
    trail_pct       : float  – percentage distance when mode='percentage' (default 1.5)
                               e.g. 1.5 means stop stays 1.5 % behind current price
    """

    def __init__(
        self,
        mode: str = "atr",
        atr_multiplier: float = 2.0,
        trail_pct: float = 1.5,
    ) -> None:
        mode = mode.lower()
        if mode not in _VALID_MODES:
            raise ValueError(
                f"mode must be one of {sorted(_VALID_MODES)}, got {mode!r}"
            )
        if atr_multiplier <= 0:
            raise ValueError(f"atr_multiplier must be > 0, got {atr_multiplier}")
        if trail_pct <= 0:
            raise ValueError(f"trail_pct must be > 0, got {trail_pct}")

        self.mode: str = mode
        self.atr_multiplier: float = atr_multiplier
        self.trail_pct: float = trail_pct

        logger.debug(
            "TrailingStopManager: mode=%s atr_mult=%.2f trail_pct=%.3f%%",
            self.mode,
            self.atr_multiplier,
            self.trail_pct,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        position: dict,
        current_price: float,
        atr: float,
    ) -> Optional[float]:
        """
        Compute a new trailing stop for the position and return it only if
        it represents an improvement over the current stop.

        A LONG position's stop improves when it moves higher.
        A SHORT position's stop improves when it moves lower.

        Parameters
        ----------
        position      : dict
            Must contain:
                ``direction``  – ``'LONG'`` or ``'SHORT'``
                ``stop_loss``  – current stop-loss price (float)
        current_price : float
            Latest market price for the symbol.
        atr           : float
            Current ATR value (required even in percentage mode for API
            uniformity; ignored in percentage mode).

        Returns
        -------
        float or None
            The new (improved) stop-loss price, or ``None`` if the
            trailing stop does not improve on the current stop.

        Raises
        ------
        ValueError
            On invalid inputs.
        """
        direction: str = position.get("direction", "").upper()
        if direction not in {"LONG", "SHORT"}:
            raise ValueError(
                f"position['direction'] must be 'LONG' or 'SHORT', got {direction!r}"
            )

        current_stop: float = float(position.get("stop_loss", 0.0))

        if current_price <= 0:
            raise ValueError(f"current_price must be > 0, got {current_price}")
        if atr < 0:
            raise ValueError(f"atr must be >= 0, got {atr}")

        # Compute the candidate trailing stop --------------------------------
        candidate_stop: float = self._compute_candidate(
            current_price=current_price,
            atr=atr,
            direction=direction,
        )

        # Only update if it represents a favourable move ---------------------
        improved: bool = self._is_improvement(
            direction=direction,
            current_stop=current_stop,
            candidate_stop=candidate_stop,
        )

        if improved:
            logger.debug(
                "Trailing stop updated [%s]: %.4f -> %.4f (price=%.4f)",
                direction,
                current_stop,
                candidate_stop,
                current_price,
            )
            return candidate_stop

        logger.debug(
            "Trailing stop NOT updated [%s]: candidate=%.4f <= current=%.4f",
            direction,
            candidate_stop,
            current_stop,
        )
        return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_candidate(
        self, current_price: float, atr: float, direction: str
    ) -> float:
        """Return the trailing stop candidate based on the configured mode."""
        if self.mode == "atr":
            if atr <= 0:
                raise ValueError(
                    "ATR mode requires atr > 0; received atr={atr}"
                )
            offset = atr * self.atr_multiplier
        else:  # percentage mode
            offset = current_price * (self.trail_pct / 100.0)

        if direction == "LONG":
            return current_price - offset
        else:  # SHORT
            return current_price + offset

    @staticmethod
    def _is_improvement(
        direction: str, current_stop: float, candidate_stop: float
    ) -> bool:
        """
        Return True if candidate_stop is strictly better than current_stop
        for the given direction.

        For LONG:  better means higher (closer to, but still below, current price)
        For SHORT: better means lower  (closer to, but still above, current price)
        """
        if direction == "LONG":
            return candidate_stop > current_stop
        else:  # SHORT
            return candidate_stop < current_stop
