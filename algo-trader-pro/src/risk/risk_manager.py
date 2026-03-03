"""
src/risk/risk_manager.py

Risk Management Orchestrator.

Responsibilities
----------------
1. Gate on maximum concurrent positions.
2. Detect high-volatility regimes via ATR-percent and adapt stop width.
3. Size positions using a modified Kelly Criterion (half-Kelly by default)
   with a confidence boost for high-conviction signals.
4. Enforce maximum single-position and portfolio-level risk caps.
5. Compute stop-loss and take-profit prices for both LONG and SHORT.

Kelly Criterion
---------------
    b  = avg_win / avg_loss   (win/loss ratio)
    p  = win_rate
    q  = 1 - p
    kelly_raw = (b*p - q) / b * kelly_fraction_multiplier

The result is clamped to [kelly_min, kelly_max] and optionally boosted by
1.2 × for confidence > 80 (capped at kelly_max again).

Position size
-------------
    risk_amount      = equity * kelly_fraction
    position_size    = (risk_amount / stop_distance) * entry_price
    position_size    = min(position_size, equity * 0.35)   # single-trade cap

Portfolio risk check
--------------------
    total_current_risk = sum of risk_amount for all open positions
    if total_current_risk + new_risk_amount > equity * max_portfolio_risk_pct/100:
        reduce position size proportionally; if still tiny → return None
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, List, Any


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

def leverage_from_confidence(confidence: float, config: Optional[dict] = None) -> float:
    """
    Return leverage (1-10x) based on signal confidence.
    Default bands: 55-60→2x, 60-70→5x, 70-80→8x, 80+→10x.
    """
    bands = [
        (55, 60, 2.0),
        (60, 70, 5.0),
        (70, 80, 8.0),
        (80, 101, 10.0),
    ]
    if config:
        try:
            cfg_bands = config.get("strategy", {}).get("leverage_by_confidence", [])
            if cfg_bands:
                bands = [
                    (b["min_conf"], b["max_conf"], float(b["leverage"]))
                    for b in cfg_bands
                ]
        except Exception:
            pass
    for lo, hi, lev in bands:
        if lo <= confidence < hi:
            return lev
    return 1.0


@dataclass
class PositionSizing:
    """
    Fully resolved position sizing parameters returned by RiskManager.

    Attributes
    ----------
    position_size_usd : float
        Dollar value of the position (notional).
    quantity : float
        Number of base-asset units to trade.
    stop_loss : float
        Absolute price of the stop-loss.
    take_profit : float
        Absolute price of the take-profit.
    kelly_fraction : float
        Effective Kelly fraction used for this trade.
    risk_amount : float
        Maximum USDT at risk on this trade (distance to stop × quantity).
    atr_multiplier : float
        ATR multiplier used to compute stop distance.
    stop_distance : float
        Absolute price distance from entry to stop-loss.
    """
    position_size_usd: float
    quantity: float
    stop_loss: float
    take_profit: float
    kelly_fraction: float
    risk_amount: float
    atr_multiplier: float
    stop_distance: float


# ---------------------------------------------------------------------------
# RiskManager
# ---------------------------------------------------------------------------

class RiskManager:
    """
    Position-sizing and risk-guardrail orchestrator.

    Parameters
    ----------
    config : dict
        Full application config dict.  Relevant sub-keys:
        ``risk.*``, ``strategy.max_concurrent_positions``,
        ``strategy.max_portfolio_risk_pct``.
    """

    # Hard cap: never risk more than this fraction of equity on one trade.
    MAX_SINGLE_POSITION_PCT: float = 0.35

    def __init__(self, config: dict) -> None:
        self.logger = logging.getLogger(__name__)

        risk_cfg: dict = config.get("risk", {})
        strategy_cfg: dict = config.get("strategy", {})
        tp_cfg: dict = risk_cfg.get("take_profit", {})

        # ---- Kelly parameters ----
        self.kelly_fraction_multiplier: float = float(
            risk_cfg.get("kelly_fraction_multiplier", 0.5)
        )
        self.kelly_min: float = float(
            risk_cfg.get("kelly_min_fraction", 0.01)
        )
        self.kelly_max: float = float(
            risk_cfg.get("kelly_max_fraction", 0.15)
        )

        # ---- ATR stop multipliers ----
        self.atr_stop_multiplier: float = float(
            risk_cfg.get("atr_stop_multiplier", 2.0)
        )
        self.atr_high_vol_multiplier: float = float(
            risk_cfg.get("atr_high_volatility_multiplier", 3.0)
        )
        self.volatility_threshold_pct: float = float(
            risk_cfg.get("volatility_threshold_pct", 5.0)
        )

        # ---- Portfolio limits ----
        self.max_concurrent_positions: int = int(
            strategy_cfg.get("max_concurrent_positions", 3)
        )
        self.max_portfolio_risk_pct: float = float(
            strategy_cfg.get("max_portfolio_risk_pct", 20.0)
        )

        # ---- Risk / reward ----
        self.risk_reward_ratio: float = float(
            tp_cfg.get("risk_reward_ratio", 2.5)
        )

        # ---- Adaptive Kelly state (updated via update_kelly_params) ----
        self.win_rate: float = 0.50      # initial estimate: 50%
        self.avg_win: float = 0.02       # initial estimate: 2% win
        self.avg_loss: float = 0.01      # initial estimate: 1% loss

        self.logger.info(
            "RiskManager initialised | "
            "kelly_mult=%.2f kelly=[%.3f,%.3f] "
            "atr_mult=[%.1f,%.1f] vol_thresh=%.1f%% "
            "max_pos=%d max_portfolio_risk=%.1f%% rr=%.2f",
            self.kelly_fraction_multiplier,
            self.kelly_min, self.kelly_max,
            self.atr_stop_multiplier, self.atr_high_vol_multiplier,
            self.volatility_threshold_pct,
            self.max_concurrent_positions,
            self.max_portfolio_risk_pct,
            self.risk_reward_ratio,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def size_position(
        self,
        symbol: str,
        direction: str,
        confidence: float,
        entry_price: float,
        current_equity: float,
        atr: float,
        open_positions: list,
    ) -> Optional[PositionSizing]:
        """
        Compute full position sizing or return None if the trade is blocked.

        Parameters
        ----------
        symbol : str
            Trading pair identifier (used for logging only).
        direction : str
            'LONG' or 'SHORT'.
        confidence : float
            Aggregated confidence score [0, 100].
        entry_price : float
            Intended fill price.
        current_equity : float
            Total portfolio value in USDT.
        atr : float
            Average True Range for the symbol at the current timeframe.
        open_positions : list
            List of currently open Trade objects (any object with a
            ``risk_amount`` attribute or 'risk_amount' key).

        Returns
        -------
        PositionSizing or None
        """
        # ----------------------------------------------------------------
        # Guard 1: maximum concurrent positions
        # ----------------------------------------------------------------
        if len(open_positions) >= self.max_concurrent_positions:
            self.logger.info(
                "[%s] Position rejected: max concurrent positions reached (%d/%d).",
                symbol, len(open_positions), self.max_concurrent_positions,
            )
            return None

        # ----------------------------------------------------------------
        # Guard 2: sanity-check inputs
        # ----------------------------------------------------------------
        if entry_price <= 0.0:
            self.logger.warning("[%s] Invalid entry_price=%.6f; rejecting.", symbol, entry_price)
            return None
        if current_equity <= 0.0:
            self.logger.warning("[%s] Invalid equity=%.2f; rejecting.", symbol, current_equity)
            return None
        if atr <= 0.0:
            # Use 1% of entry price as a safe fallback ATR.
            atr = entry_price * 0.01
            self.logger.warning(
                "[%s] ATR is zero/negative; using 1%% fallback = %.6f", symbol, atr
            )

        # ----------------------------------------------------------------
        # Step 1: Volatility regime detection
        # ----------------------------------------------------------------
        daily_atr_pct: float = (atr / entry_price) * 100.0
        high_vol: bool = daily_atr_pct > self.volatility_threshold_pct

        atr_multiplier: float = (
            self.atr_high_vol_multiplier if high_vol else self.atr_stop_multiplier
        )

        self.logger.debug(
            "[%s] ATR=%.6f (%.2f%% of price)  regime=%s  multiplier=%.1f",
            symbol, atr, daily_atr_pct,
            "HIGH_VOL" if high_vol else "NORMAL", atr_multiplier,
        )

        # ----------------------------------------------------------------
        # Step 2: Stop distance
        # ----------------------------------------------------------------
        stop_distance: float = atr * atr_multiplier

        # ----------------------------------------------------------------
        # Step 3: Kelly Criterion
        # ----------------------------------------------------------------
        kelly_fraction = self._compute_kelly(confidence)

        self.logger.debug(
            "[%s] Kelly: win_rate=%.3f avg_win=%.4f avg_loss=%.4f → fraction=%.4f",
            symbol, self.win_rate, self.avg_win, self.avg_loss, kelly_fraction,
        )

        # ----------------------------------------------------------------
        # Step 4: Risk amount and initial position size
        # ----------------------------------------------------------------
        risk_amount: float = current_equity * kelly_fraction

        # position_size = how much notional value to hold such that if the
        # price moves stop_distance against us, we lose exactly risk_amount.
        #
        #   loss_per_unit = stop_distance
        #   units = risk_amount / stop_distance
        #   notional = units * entry_price
        position_size_usd: float = (risk_amount / stop_distance) * entry_price

        # ----------------------------------------------------------------
        # Step 5: Single-position cap (35% of equity)
        # ----------------------------------------------------------------
        max_position: float = current_equity * self.MAX_SINGLE_POSITION_PCT
        if position_size_usd > max_position:
            self.logger.debug(
                "[%s] Position capped: %.2f → %.2f (35%% cap).",
                symbol, position_size_usd, max_position,
            )
            position_size_usd = max_position

        # ----------------------------------------------------------------
        # Step 6: Portfolio-level risk check
        # ----------------------------------------------------------------
        total_existing_risk = self._total_open_risk(open_positions)
        max_total_risk: float = current_equity * (self.max_portfolio_risk_pct / 100.0)

        if total_existing_risk + risk_amount > max_total_risk:
            # Attempt to reduce to the remaining portfolio risk budget.
            remaining_risk_budget: float = max_total_risk - total_existing_risk
            self.logger.info(
                "[%s] Portfolio risk budget: total_existing=%.2f + new=%.2f > max=%.2f. "
                "Reducing position.",
                symbol, total_existing_risk, risk_amount, max_total_risk,
            )

            if remaining_risk_budget <= 0.0:
                self.logger.info(
                    "[%s] No remaining portfolio risk budget; rejecting trade.", symbol
                )
                return None

            # Scale down proportionally.
            scale_factor: float = remaining_risk_budget / risk_amount
            risk_amount = remaining_risk_budget
            position_size_usd *= scale_factor
            kelly_fraction *= scale_factor

        # ----------------------------------------------------------------
        # Step 7: Minimum viable position check
        # ----------------------------------------------------------------
        min_position_usd: float = 10.0  # hard floor: $10 minimum order
        if position_size_usd < min_position_usd:
            self.logger.info(
                "[%s] Position size too small (%.2f < %.2f); rejecting.",
                symbol, position_size_usd, min_position_usd,
            )
            return None

        # ----------------------------------------------------------------
        # Step 8: Quantity
        # ----------------------------------------------------------------
        quantity: float = position_size_usd / entry_price

        # ----------------------------------------------------------------
        # Step 9: Stop-loss and take-profit prices
        # ----------------------------------------------------------------
        if direction == "LONG":
            stop_loss: float = entry_price - stop_distance
            take_profit: float = entry_price + stop_distance * self.risk_reward_ratio
        else:  # SHORT
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - stop_distance * self.risk_reward_ratio

        # Sanity check: stops must be positive.
        if stop_loss <= 0.0:
            self.logger.warning(
                "[%s] Computed stop_loss=%.6f is non-positive; rejecting.", symbol, stop_loss
            )
            return None

        self.logger.info(
            "[%s] Position sized: dir=%s size_usd=%.2f qty=%.6f "
            "entry=%.4f sl=%.4f tp=%.4f kelly=%.4f risk=%.2f",
            symbol, direction, position_size_usd, quantity,
            entry_price, stop_loss, take_profit, kelly_fraction, risk_amount,
        )

        return PositionSizing(
            position_size_usd=round(position_size_usd, 4),
            quantity=round(quantity, 8),
            stop_loss=round(stop_loss, 8),
            take_profit=round(take_profit, 8),
            kelly_fraction=round(kelly_fraction, 6),
            risk_amount=round(risk_amount, 4),
            atr_multiplier=atr_multiplier,
            stop_distance=round(stop_distance, 8),
        )

    def update_kelly_params(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> None:
        """
        Update the adaptive Kelly parameters using exponential smoothing.

        New observations are blended with the current estimates at alpha=0.1
        so the parameters evolve slowly and are not thrown off by a single
        outlier trade.

        Parameters
        ----------
        win_rate : float
            Observed win rate in [0, 1] from the most recent sample window.
        avg_win : float
            Average winning return as a decimal (e.g. 0.03 for 3%).
        avg_loss : float
            Average losing return as a decimal (e.g. 0.015 for 1.5%),
            expressed as a positive number.
        """
        alpha: float = 0.1  # exponential smoothing factor

        old_wr, old_aw, old_al = self.win_rate, self.avg_win, self.avg_loss

        # Clamp inputs to valid ranges before blending.
        win_rate = max(0.0, min(1.0, win_rate))
        avg_win = max(1e-6, avg_win)    # avoid division by zero in Kelly
        avg_loss = max(1e-6, avg_loss)  # expressed as positive

        self.win_rate = (1.0 - alpha) * self.win_rate + alpha * win_rate
        self.avg_win = (1.0 - alpha) * self.avg_win + alpha * avg_win
        self.avg_loss = (1.0 - alpha) * self.avg_loss + alpha * avg_loss

        self.logger.debug(
            "Kelly params updated: "
            "win_rate %.4f→%.4f  avg_win %.4f→%.4f  avg_loss %.4f→%.4f",
            old_wr, self.win_rate,
            old_aw, self.avg_win,
            old_al, self.avg_loss,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_kelly(self, confidence: float) -> float:
        """
        Compute the effective Kelly fraction for this signal.

        Formula
        -------
            b         = avg_win / avg_loss
            kelly_raw = (b*p - q) / b * kelly_fraction_multiplier
            clamped   = max(kelly_min, min(kelly_max, kelly_raw))

        A confidence > 80 boosts the fraction by 20%, re-clamped to kelly_max.

        Parameters
        ----------
        confidence : float
            Aggregated signal confidence [0, 100].

        Returns
        -------
        float
            Effective Kelly fraction in [kelly_min, kelly_max].
        """
        p: float = self.win_rate
        q: float = 1.0 - p

        # Protect against degenerate avg_loss.
        avg_loss = max(self.avg_loss, 1e-9)
        b: float = self.avg_win / avg_loss

        if b <= 0.0:
            return self.kelly_min

        # Raw Kelly formula: f* = (b*p - q) / b
        kelly_raw: float = ((b * p - q) / b) * self.kelly_fraction_multiplier

        # Clamp to configured range.
        kelly_fraction: float = max(self.kelly_min, min(self.kelly_max, kelly_raw))

        # Confidence boost for high-conviction signals.
        if confidence > 80.0:
            kelly_fraction = min(self.kelly_max, kelly_fraction * 1.2)

        return kelly_fraction

    @staticmethod
    def _total_open_risk(open_positions: list) -> float:
        """
        Sum the risk_amount across all open positions.

        Supports both Trade objects (with .risk_amount attribute) and
        plain dicts (with 'risk_amount' key).  Missing / None values are
        treated as 0.
        """
        total: float = 0.0
        for pos in open_positions:
            try:
                if hasattr(pos, "risk_amount"):
                    val = pos.risk_amount
                elif isinstance(pos, dict):
                    val = pos.get("risk_amount", 0.0)
                else:
                    val = 0.0
                total += float(val) if val is not None else 0.0
            except (TypeError, ValueError):
                pass
        return total
