"""
src/risk/kelly_criterion.py

Kelly Criterion position-sizing module.

The Kelly formula answers: "What fraction of my bankroll should I risk
on this trade in order to maximise long-run geometric growth?"

    f* = (b·p − q) / b

where:
    p = probability of winning
    q = 1 − p  (probability of losing)
    b = avg_win / avg_loss  (win/loss ratio of payoffs)

In practice, fractional Kelly is used (fraction_multiplier < 1) to reduce
variance and drawdown risk.

Usage
-----
    from src.risk.kelly_criterion import KellyCriterion

    kelly = KellyCriterion(config)
    fraction = kelly.calculate(
        ml_win_probability=0.60,
        avg_win_pct=0.03,
        avg_loss_pct=0.015,
    )
    # fraction is clamped to [min_fraction, max_fraction]
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class KellyCriterion:
    """
    Kelly Criterion position-sizing calculator with empirical trade-history
    updates and safety clamping.

    Parameters
    ----------
    config : object
        Must expose a ``risk`` attribute with the following fields:
          - kelly_fraction_multiplier   (float, e.g. 0.5)
          - kelly_min_fraction          (float, e.g. 0.01)
          - kelly_max_fraction          (float, e.g. 0.15)
          - kelly_rolling_window_trades (int,   e.g. 50)
    """

    def __init__(self, config: Any) -> None:
        risk_cfg = config.risk

        self.fraction_multiplier: float = float(risk_cfg.kelly_fraction_multiplier)
        self.min_fraction: float = float(risk_cfg.kelly_min_fraction)
        self.max_fraction: float = float(risk_cfg.kelly_max_fraction)
        self.rolling_window: int = int(risk_cfg.kelly_rolling_window_trades)

        # Each entry: {"pnl_pct": float, "won": bool}
        self._trade_history: List[Dict[str, Any]] = []

        # Validate parameter sanity
        if not (0 < self.min_fraction < self.max_fraction <= 1.0):
            raise ValueError(
                f"Fraction bounds invalid: min={self.min_fraction}, "
                f"max={self.max_fraction}"
            )
        if not (0 < self.fraction_multiplier <= 1.0):
            raise ValueError(
                f"fraction_multiplier must be in (0, 1], "
                f"got {self.fraction_multiplier}"
            )
        if self.rolling_window < 10:
            raise ValueError(
                f"rolling_window must be >= 10, got {self.rolling_window}"
            )

        logger.debug(
            "KellyCriterion initialised: multiplier=%.2f, range=[%.3f, %.3f], "
            "window=%d",
            self.fraction_multiplier,
            self.min_fraction,
            self.max_fraction,
            self.rolling_window,
        )

    # ------------------------------------------------------------------
    # Core calculation
    # ------------------------------------------------------------------

    def calculate(
        self,
        ml_win_probability: float,
        avg_win_pct: float,
        avg_loss_pct: float,
    ) -> float:
        """
        Compute the fractional Kelly position size.

        Parameters
        ----------
        ml_win_probability : float
            Model-estimated probability of a winning trade (0 < p < 1).
        avg_win_pct : float
            Average percentage gain on winning trades (> 0).
        avg_loss_pct : float
            Average percentage loss on losing trades (> 0, expressed as a
            positive number, e.g. 0.015 for a 1.5 % loss).

        Returns
        -------
        float
            Fraction of portfolio to risk, clamped to [min_fraction, max_fraction].
        """
        # --- Input validation ------------------------------------------------
        if not (0.0 < ml_win_probability < 1.0):
            raise ValueError(
                f"ml_win_probability must be in (0, 1), got {ml_win_probability}"
            )
        if avg_win_pct <= 0:
            raise ValueError(f"avg_win_pct must be > 0, got {avg_win_pct}")
        if avg_loss_pct <= 0:
            raise ValueError(f"avg_loss_pct must be > 0, got {avg_loss_pct}")

        p: float = ml_win_probability
        q: float = 1.0 - p
        b: float = avg_win_pct / avg_loss_pct  # win/loss ratio

        # Kelly formula
        raw_kelly: float = (b * p - q) / b

        if raw_kelly <= 0:
            # Negative or zero Kelly means the bet has negative/zero EV;
            # fall back to the minimum fraction (risk management will
            # likely veto the trade anyway via the EV filter).
            logger.debug(
                "Kelly fraction non-positive (%.4f); clamping to min_fraction.",
                raw_kelly,
            )
            return self.min_fraction

        fractional_kelly: float = raw_kelly * self.fraction_multiplier
        result: float = self._clamp(fractional_kelly)

        logger.debug(
            "Kelly: p=%.3f q=%.3f b=%.3f raw_f*=%.4f frac_f*=%.4f clamped=%.4f",
            p,
            q,
            b,
            raw_kelly,
            fractional_kelly,
            result,
        )
        return result

    # ------------------------------------------------------------------
    # Trade-history management
    # ------------------------------------------------------------------

    def add_trade_result(self, pnl_pct: float) -> None:
        """
        Append a single closed-trade result to the rolling history.

        The history is trimmed to rolling_window length so old trades
        gradually phase out.

        Parameters
        ----------
        pnl_pct : float
            Realised P&L percentage for the trade (can be negative).
        """
        self._trade_history.append(
            {"pnl_pct": float(pnl_pct), "won": pnl_pct > 0.0}
        )
        # Trim to window
        if len(self._trade_history) > self.rolling_window:
            self._trade_history = self._trade_history[-self.rolling_window:]

    def update_from_trade_history(
        self, trades: list
    ) -> Tuple[float, float, float]:
        """
        Bulk-load a list of trade dicts and return empirical statistics.

        Each trade in *trades* must have at least a ``pnl_pct`` key.
        The last ``rolling_window`` entries are kept.

        Parameters
        ----------
        trades : list of dict
            Each dict: {"pnl_pct": float, ...}

        Returns
        -------
        (win_rate, avg_win_pct, avg_loss_pct) : Tuple[float, float, float]
            win_rate    – fraction of trades that were profitable  (0-1)
            avg_win_pct – mean % return on winning trades (positive)
            avg_loss_pct– mean % loss on losing trades   (positive)
        """
        # Replace internal history with the new list (trimmed to window)
        self._trade_history = [
            {"pnl_pct": float(t["pnl_pct"]), "won": float(t["pnl_pct"]) > 0.0}
            for t in trades
        ][-self.rolling_window:]

        return self._compute_stats(self._trade_history)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_current_stats(self) -> dict:
        """
        Return a summary of the current rolling trade statistics and the
        Kelly fraction they imply.

        Returns
        -------
        dict with keys:
            win_rate      : float   – fraction of winning trades
            avg_win       : float   – average winning trade return (pct)
            avg_loss      : float   – average losing trade loss (pct, positive)
            kelly_fraction: float   – clamped fractional Kelly
            ev            : float   – expected value per trade (pct)
            sample_size   : int     – number of trades in rolling window
        """
        history = self._trade_history

        if len(history) < 2:
            return {
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "kelly_fraction": self.min_fraction,
                "ev": 0.0,
                "sample_size": len(history),
            }

        win_rate, avg_win, avg_loss = self._compute_stats(history)

        if avg_win > 0 and avg_loss > 0:
            kelly_fraction = self.calculate(win_rate, avg_win, avg_loss)
            ev = win_rate * avg_win - (1 - win_rate) * avg_loss
        else:
            kelly_fraction = self.min_fraction
            ev = 0.0

        return {
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "kelly_fraction": kelly_fraction,
            "ev": ev,
            "sample_size": len(history),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_stats(
        history: List[Dict[str, Any]]
    ) -> Tuple[float, float, float]:
        """
        Compute (win_rate, avg_win_pct, avg_loss_pct) from a list of
        trade dicts containing 'pnl_pct'.
        """
        if not history:
            return 0.0, 0.0, 0.0

        wins = [t["pnl_pct"] for t in history if t["pnl_pct"] > 0.0]
        losses = [abs(t["pnl_pct"]) for t in history if t["pnl_pct"] <= 0.0]

        win_rate: float = len(wins) / len(history)
        avg_win: float = sum(wins) / len(wins) if wins else 0.0
        avg_loss: float = sum(losses) / len(losses) if losses else 0.0

        return win_rate, avg_win, avg_loss

    def _clamp(self, value: float) -> float:
        """Clamp *value* to [min_fraction, max_fraction]."""
        return max(self.min_fraction, min(self.max_fraction, value))
