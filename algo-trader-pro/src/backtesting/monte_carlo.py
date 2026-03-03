"""
src/backtesting/monte_carlo.py

Monte Carlo Simulator
======================
Estimates the distribution of future equity-curve outcomes by bootstrapping
from the empirical trade-return distribution observed in a backtest.

Algorithm
---------
For each simulation run (default: 1 000):
  1. Draw *n_trades* returns (with replacement) from the historical PnL list.
  2. Compound them into an equity path starting at ``initial_capital``.
  3. Track the peak equity and maximum drawdown along the path.
  4. Store the final equity and the full path (up to ``_MAX_SAMPLE_PATHS`` paths
     are retained for chart rendering).

After all runs compute:
  * Percentile bands (5 / 25 / 50 / 75 / 95) for both final equity and max-DD.
  * Probability of profit  (ending above ``initial_capital``).
  * Probability of ruin    (ending below 50 % of ``initial_capital``).

Minimum trade count
-------------------
At least 5 completed trades are required for a meaningful simulation.
When fewer are available the method returns a ``MonteCarloResult`` with
``n_simulations = 0`` and a log-warning rather than raising.

Usage
-----
    from src.backtesting.monte_carlo import MonteCarloSimulator

    mc = MonteCarloSimulator(n_simulations=2000)
    result = mc.run(trades=backtest_result.trades, initial_capital=10_000.0)
    print(result.to_dict())
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# How many equity paths to store in the result for chart rendering
_MAX_SAMPLE_PATHS: int = 20

# Minimum number of completed trades required to run the simulation
_MIN_TRADES: int = 5


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class MonteCarloResult:
    """
    Full output of a Monte Carlo simulation run.

    Attributes
    ----------
    final_equity_percentiles : dict
        Maps percentile (int) → final-equity value (float).
        Keys: 5, 25, 50, 75, 95.
    max_drawdown_percentiles : dict
        Maps percentile (int) → max-drawdown percentage value (float).
        Keys: 5, 25, 50, 75, 95.
    probability_of_profit : float
        Percentage of simulations that ended above ``initial_capital``.
        Range: 0.0 – 100.0.
    probability_of_ruin : float
        Percentage of simulations whose equity dropped below 50 % of
        ``initial_capital`` at any point.  Range: 0.0 – 100.0.
    median_final_equity : float
        Median final equity across all simulations (same as
        ``final_equity_percentiles[50]``).
    sample_paths : list of list of float
        Up to 20 equity paths (each a list of floats starting at
        ``initial_capital``) for visualisation.
    n_simulations : int
        Number of Monte Carlo runs actually executed.
        Zero when there were not enough historical trades.
    initial_capital : float
        Starting capital used for the simulation.
    """

    final_equity_percentiles: Dict[int, float]
    max_drawdown_percentiles: Dict[int, float]
    probability_of_profit: float
    probability_of_ruin: float
    median_final_equity: float
    sample_paths: List[List[float]]
    n_simulations: int
    initial_capital: float

    # -----------------------------------------------------------------------
    # Serialisation
    # -----------------------------------------------------------------------

    def to_dict(self) -> dict:
        """
        Return a JSON-serialisable representation of all fields.

        Keys map 1-to-1 to the attribute names, with ``sample_paths`` included
        in full (suitable for embedding in a front-end chart payload).
        """
        return {
            "final_equity_percentiles": {
                str(k): round(v, 4)
                for k, v in self.final_equity_percentiles.items()
            },
            "max_drawdown_percentiles": {
                str(k): round(v, 4)
                for k, v in self.max_drawdown_percentiles.items()
            },
            "probability_of_profit": round(self.probability_of_profit, 4),
            "probability_of_ruin": round(self.probability_of_ruin, 4),
            "median_final_equity": round(self.median_final_equity, 4),
            "sample_paths": [
                [round(v, 4) for v in path]
                for path in self.sample_paths
            ],
            "n_simulations": self.n_simulations,
            "initial_capital": round(self.initial_capital, 4),
        }

    # -----------------------------------------------------------------------
    # Convenience properties
    # -----------------------------------------------------------------------

    @property
    def expected_worst_case_equity(self) -> float:
        """5th-percentile final equity (tail-risk proxy)."""
        return self.final_equity_percentiles.get(5, self.initial_capital)

    @property
    def expected_best_case_equity(self) -> float:
        """95th-percentile final equity."""
        return self.final_equity_percentiles.get(95, self.initial_capital)

    @property
    def median_max_drawdown_pct(self) -> float:
        """50th-percentile maximum drawdown (%)."""
        return self.max_drawdown_percentiles.get(50, 0.0)

    def __repr__(self) -> str:
        return (
            f"MonteCarloResult("
            f"n={self.n_simulations}, "
            f"p50_equity={self.median_final_equity:.2f}, "
            f"prob_profit={self.probability_of_profit:.1f}%, "
            f"prob_ruin={self.probability_of_ruin:.1f}%, "
            f"p50_maxdd={self.median_max_drawdown_pct:.1f}%)"
        )


# ---------------------------------------------------------------------------
# Simulator class
# ---------------------------------------------------------------------------


class MonteCarloSimulator:
    """
    Bootstrap-based Monte Carlo simulator for equity-curve analysis.

    Parameters
    ----------
    n_simulations : int
        Number of bootstrap runs.  Default: 1 000.
    confidence_intervals : list of int
        Percentile levels to compute.  Default: [5, 25, 50, 75, 95].
    """

    def __init__(
        self,
        n_simulations: int = 1_000,
        confidence_intervals: Optional[List[int]] = None,
    ) -> None:
        self.n_simulations: int = max(1, n_simulations)
        self.confidence_intervals: List[int] = (
            confidence_intervals
            if confidence_intervals is not None
            else [5, 25, 50, 75, 95]
        )
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        trades: List[dict],
        initial_capital: float = 10_000.0,
    ) -> MonteCarloResult:
        """
        Run the Monte Carlo simulation.

        Parameters
        ----------
        trades : list of dict
            Completed trade records.  Each dict must contain a ``pnl_pct``
            key (net P&L as a percentage of notional, e.g. 2.5 for +2.5 %).
        initial_capital : float
            Starting portfolio value in USDT.  Default: 10 000.

        Returns
        -------
        MonteCarloResult
            Full simulation output including percentile bands, probability
            statistics, and up to 20 sample equity paths.
        """
        self.logger.info(
            "Monte Carlo simulation started: n_sims=%d, trades=%d, capital=%.2f",
            self.n_simulations,
            len(trades),
            initial_capital,
        )

        # ---------------------------------------------------------------
        # Guard: not enough trades
        # ---------------------------------------------------------------
        if len(trades) < _MIN_TRADES:
            self.logger.warning(
                "Insufficient trades for Monte Carlo simulation: "
                "need at least %d, got %d.  Returning empty result.",
                _MIN_TRADES,
                len(trades),
            )
            return self._empty_result(initial_capital)

        # ---------------------------------------------------------------
        # Extract per-trade return fractions
        # ---------------------------------------------------------------
        pnl_returns: List[float] = []
        for t in trades:
            raw = t.get("pnl_pct")
            if raw is None:
                continue
            try:
                pnl_returns.append(float(raw) / 100.0)
            except (TypeError, ValueError):
                continue

        if len(pnl_returns) < _MIN_TRADES:
            self.logger.warning(
                "Too few valid pnl_pct values (%d) in trade list.  "
                "Returning empty result.",
                len(pnl_returns),
            )
            return self._empty_result(initial_capital)

        n_trades: int = len(pnl_returns)

        # ---------------------------------------------------------------
        # Run simulations
        # ---------------------------------------------------------------
        final_equities: List[float] = []
        max_drawdowns_pct: List[float] = []
        sample_paths: List[List[float]] = []

        for i in range(self.n_simulations):
            # Bootstrap: resample n_trades returns WITH replacement
            sampled: List[float] = random.choices(pnl_returns, k=n_trades)

            equity: float = initial_capital
            path: List[float] = [equity]
            peak: float = equity
            max_dd: float = 0.0

            for ret in sampled:
                equity *= (1.0 + ret)
                path.append(equity)

                if equity > peak:
                    peak = equity

                # Drawdown is computed relative to the running peak
                if peak > 0:
                    dd = (peak - equity) / peak
                    if dd > max_dd:
                        max_dd = dd

            final_equities.append(equity)
            max_drawdowns_pct.append(max_dd * 100.0)

            if i < _MAX_SAMPLE_PATHS:
                sample_paths.append(path)

        # ---------------------------------------------------------------
        # Aggregate statistics
        # ---------------------------------------------------------------
        final_equity_percentiles: Dict[int, float] = {
            p: self._percentile(final_equities, p)
            for p in self.confidence_intervals
        }
        max_dd_percentiles: Dict[int, float] = {
            p: self._percentile(max_drawdowns_pct, p)
            for p in self.confidence_intervals
        }

        # Probability of profit: fraction ending above initial_capital
        prob_profit: float = (
            sum(1 for e in final_equities if e > initial_capital)
            / self.n_simulations
            * 100.0
        )

        # Probability of ruin: fraction ending below 50 % of initial_capital
        ruin_threshold: float = initial_capital * 0.5
        prob_ruin: float = (
            sum(1 for e in final_equities if e < ruin_threshold)
            / self.n_simulations
            * 100.0
        )

        median_equity: float = final_equity_percentiles.get(50, initial_capital)

        result = MonteCarloResult(
            final_equity_percentiles=final_equity_percentiles,
            max_drawdown_percentiles=max_dd_percentiles,
            probability_of_profit=round(prob_profit, 4),
            probability_of_ruin=round(prob_ruin, 4),
            median_final_equity=round(median_equity, 4),
            sample_paths=sample_paths,
            n_simulations=self.n_simulations,
            initial_capital=initial_capital,
        )

        self.logger.info(
            "Monte Carlo complete: %s", result
        )
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _percentile(self, data: List[float], pct: int) -> float:
        """
        Compute the *pct*-th percentile of *data* using linear interpolation
        (equivalent to numpy's ``percentile`` with ``method='linear'``).

        Parameters
        ----------
        data : list of float
            Unsorted numeric data.  Must be non-empty.
        pct : int
            Percentile level in [0, 100].

        Returns
        -------
        float
            Interpolated percentile value.
        """
        if not data:
            return 0.0

        sorted_data: List[float] = sorted(data)
        n: int = len(sorted_data)

        if n == 1:
            return sorted_data[0]

        # Map percentile to a fractional index in [0, n-1]
        # Using the "inclusive" / C=1 convention (same as numpy default)
        index: float = (pct / 100.0) * (n - 1)
        lower: int = int(math.floor(index))
        upper: int = int(math.ceil(index))

        if lower == upper:
            return sorted_data[lower]

        # Linear interpolation between adjacent values
        frac: float = index - lower
        return sorted_data[lower] * (1.0 - frac) + sorted_data[upper] * frac

    @staticmethod
    def _empty_result(initial_capital: float) -> MonteCarloResult:
        """
        Return a zero-simulation result when the trade list is too short.
        All equity values default to ``initial_capital`` (no change),
        all probabilities and drawdowns are 0.
        """
        pct_keys = [5, 25, 50, 75, 95]
        equity_pcts = {p: initial_capital for p in pct_keys}
        dd_pcts = {p: 0.0 for p in pct_keys}

        return MonteCarloResult(
            final_equity_percentiles=equity_pcts,
            max_drawdown_percentiles=dd_pcts,
            probability_of_profit=0.0,
            probability_of_ruin=0.0,
            median_final_equity=initial_capital,
            sample_paths=[],
            n_simulations=0,
            initial_capital=initial_capital,
        )
