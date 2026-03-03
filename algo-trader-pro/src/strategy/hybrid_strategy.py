"""
src/strategy/hybrid_strategy.py

5-Layer Hybrid Strategy Orchestrator.

Architecture
------------
Layer 1 – Technical  : RSI, MACD, Bollinger Bands, EMA Crossover, Ichimoku, VWAP
Layer 2 – Volume     : Volume Anomaly, OBV, CVD
Layer 3 – Sentiment  : Fear & Greed, CryptoPanic, Google Trends
Layer 4 – ML         : Optional ensemble model (XGBoost + RandomForest + LSTM)
Layer 5 – Risk       : ATR-based stops, Kelly sizing, portfolio guardrails

The strategy evaluates all layers for each (symbol, DataFrame) pair and
returns a single TradeSignal if the confidence threshold is met, or None if
no trade should be placed.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

from src.signals.layer1_technical.rsi import RSISignal
from src.signals.layer1_technical.macd import MACDSignal
from src.signals.layer1_technical.bollinger_bands import BollingerBandsSignal
from src.signals.layer1_technical.ema_crossover import EMACrossoverSignal
from src.signals.layer1_technical.ichimoku import IchimokuSignal
from src.signals.layer1_technical.vwap import VWAPSignal
from src.signals.layer2_volume.volume_anomaly import VolumeAnomalySignal
from src.signals.layer2_volume.obv import OBVSignal
from src.signals.layer2_volume.cvd import CVDSignal
from src.signals.layer3_sentiment.fear_greed_signal import FearGreedSignal
from src.signals.layer3_sentiment.cryptopanic_signal import CryptoPanicSignal
from src.signals.layer3_sentiment.google_trends_signal import GoogleTrendsSignal
from src.signals.aggregator import SignalAggregator


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class TradeSignal:
    """
    Fully resolved trade signal produced by the HybridStrategy.

    All fields are populated before this object is returned; callers must
    never receive a partially-initialised TradeSignal.
    """
    symbol: str
    direction: str                     # 'LONG' or 'SHORT'
    confidence_score: float            # 0 – 100
    position_size_usd: float
    entry_price: float
    stop_loss: float
    take_profit: float
    signal_breakdown: Dict[str, Any]   # per-layer and per-signal details
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def __repr__(self) -> str:
        return (
            f"TradeSignal({self.symbol} {self.direction} "
            f"conf={self.confidence_score:.2f} "
            f"entry={self.entry_price:.4f} "
            f"sl={self.stop_loss:.4f} tp={self.take_profit:.4f} "
            f"size_usd={self.position_size_usd:.2f})"
        )


# ---------------------------------------------------------------------------
# Strategy class
# ---------------------------------------------------------------------------

class HybridStrategy:
    """
    Multi-layer hybrid trading strategy.

    Parameters
    ----------
    config : dict
        Full application config dict (parsed from settings.yaml or equivalent).
    aggregator : SignalAggregator
        Pre-constructed aggregator carrying layer weights.
    risk_manager : RiskManager
        Handles position sizing and risk guardrails.
    ml_ensemble : optional
        Trained ML ensemble object with a ``predict(feature_df)`` method that
        returns a float in [0, 100].  Pass None to disable ML layer.
    """

    # Minimum number of OHLCV rows required before generating any signal.
    # Lowered to 20 for faster testing; raise back to 100 for production.
    MIN_CANDLES: int = 20

    def __init__(
        self,
        config: dict,
        aggregator: SignalAggregator,
        risk_manager: Any,
        ml_ensemble: Optional[Any] = None,
    ) -> None:
        self.logger = logging.getLogger(__name__)

        # ----------------------------------------------------------------
        # Strategy thresholds from config
        # ----------------------------------------------------------------
        strategy_cfg: dict = config.get("strategy", {})
        self.min_confidence_long: float = float(
            strategy_cfg.get("min_confidence_long", 65)
        )
        self.min_confidence_short: float = float(
            strategy_cfg.get("min_confidence_short", 70)
        )
        self.allow_short: bool = bool(strategy_cfg.get("allow_short", False))

        # ----------------------------------------------------------------
        # Core dependencies
        # ----------------------------------------------------------------
        self.config = config
        self.aggregator = aggregator
        self.risk_manager = risk_manager
        self.ml_ensemble = ml_ensemble

        # Last evaluation (for logging); set after each evaluate() call
        self._last_evaluation: Optional[Dict[str, Any]] = None

        # ----------------------------------------------------------------
        # Layer 1 – Technical signals
        # ----------------------------------------------------------------
        sig_cfg: dict = config.get("signals", {})

        rsi_cfg = sig_cfg.get("rsi", {})
        self._rsi = RSISignal(
            period=int(rsi_cfg.get("period", 14)),
        )

        macd_cfg = sig_cfg.get("macd", {})
        self._macd = MACDSignal(
            fast=int(macd_cfg.get("fast_period", 12)),
            slow=int(macd_cfg.get("slow_period", 26)),
            signal=int(macd_cfg.get("signal_period", 9)),
        )

        bb_cfg = sig_cfg.get("bollinger", {})
        self._bb = BollingerBandsSignal(
            length=int(bb_cfg.get("period", 20)),
            std=float(bb_cfg.get("std_dev", 2.0)),
        )

        ema_cfg = sig_cfg.get("ema_crossover", {})
        self._ema = EMACrossoverSignal(
            fast=int(ema_cfg.get("fast_period", 9)),
            medium=int(ema_cfg.get("medium_period", 21)),
            slow=int(ema_cfg.get("slow_period", 50)),
        )

        self._ichimoku = IchimokuSignal(
            tenkan=9,
            kijun=26,
            senkou=52,
        )

        vwap_cfg = sig_cfg.get("vwap", {})
        self._vwap = VWAPSignal(
            std_window=int(vwap_cfg.get("std_window", 20)),
        )

        # ----------------------------------------------------------------
        # Layer 2 – Volume signals
        # ----------------------------------------------------------------
        vanom_cfg = sig_cfg.get("volume_anomaly", {})
        self._volume_anomaly = VolumeAnomalySignal(
            window=int(vanom_cfg.get("lookback_period", 20)),
            z_threshold=float(vanom_cfg.get("zscore_threshold", 2.5)),
        )

        obv_cfg = sig_cfg.get("obv", {})
        self._obv = OBVSignal(
            regression_window=int(obv_cfg.get("regression_window", 10)),
        )

        self._cvd = CVDSignal(
            regression_window=10,
        )

        # ----------------------------------------------------------------
        # Layer 3 – Sentiment signals
        # ----------------------------------------------------------------
        self._fear_greed = FearGreedSignal()
        self._cryptopanic = CryptoPanicSignal()
        self._google_trends = GoogleTrendsSignal()

        self.logger.info(
            "HybridStrategy initialised | "
            "min_conf_long=%.1f min_conf_short=%.1f allow_short=%s ml=%s",
            self.min_confidence_long,
            self.min_confidence_short,
            self.allow_short,
            "enabled" if ml_ensemble is not None else "disabled",
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def evaluate(
        self,
        symbol: str,
        df: pd.DataFrame,
        sentiment_data: dict,
        current_equity: float,
        open_positions: list,
    ) -> Optional[TradeSignal]:
        """
        Full strategy evaluation pipeline.

        Parameters
        ----------
        symbol : str
            Trading pair, e.g. "BTCUSDT".
        df : pd.DataFrame
            OHLCV DataFrame with columns: open, high, low, close, volume.
            Must have at least MIN_CANDLES rows.
        sentiment_data : dict
            Flat dict with keys:
            - ``fear_greed``            : float [0, 100]
            - ``cryptopanic_score``     : float [-1, 1]
            - ``cryptopanic_article_count`` : int (optional)
            - ``google_trends_score``   : float [0, 100]
            - ``google_trends_scores``  : list[float]  (optional, historical)
            - ``google_trends_rolling_avg`` : float (optional)
        current_equity : float
            Current total portfolio value in USDT.
        open_positions : list
            List of currently open Trade objects (used for portfolio limits).

        Returns
        -------
        TradeSignal or None
        """
        # ----------------------------------------------------------------
        # Guard: minimum data requirement
        # ----------------------------------------------------------------
        if df is None or len(df) < self.MIN_CANDLES:
            self.logger.debug(
                "[%s] Skipping: only %d candles available (need %d).",
                symbol,
                0 if df is None else len(df),
                self.MIN_CANDLES,
            )
            return None

        # ----------------------------------------------------------------
        # Layer 1 – Technical signals
        # Compute in a thread executor to avoid blocking the event loop if
        # pandas_ta computations are CPU-bound.
        # ----------------------------------------------------------------
        loop = asyncio.get_event_loop()
        layer1_signals = await loop.run_in_executor(
            None, self._compute_layer1, df
        )

        # ----------------------------------------------------------------
        # Layer 2 – Volume signals
        # ----------------------------------------------------------------
        layer2_signals = await loop.run_in_executor(
            None, self._compute_layer2, df
        )

        # ----------------------------------------------------------------
        # Layer 3 – Sentiment signals
        # These signals accept the sentiment_data dict directly via their
        # compute() method (BaseSignal compatibility shim).
        # ----------------------------------------------------------------
        layer3_signals = self._compute_layer3(sentiment_data)

        # ----------------------------------------------------------------
        # Layer 4 – ML inference (optional)
        # ----------------------------------------------------------------
        ml_value: Optional[float] = await loop.run_in_executor(
            None, self._compute_ml_score, df
        )

        # ----------------------------------------------------------------
        # Aggregation
        # ----------------------------------------------------------------
        agg_result = self.aggregator.aggregate(
            layer1_signals=layer1_signals,
            layer2_signals=layer2_signals,
            layer3_signals=layer3_signals,
            ml_score=ml_value,
        )

        confidence: float = agg_result.confidence_score
        direction: str = agg_result.direction

        self.logger.info(
            "[%s] Aggregation: direction=%s confidence=%.2f "
            "[L1=%.4f L2=%.4f L3=%.4f ML=%s]",
            symbol,
            direction,
            confidence,
            agg_result.layer1_score,
            agg_result.layer2_score,
            agg_result.layer3_score,
            f"{agg_result.ml_score:.4f}" if ml_value is not None else "n/a",
        )

        # ----------------------------------------------------------------
        # Threshold check
        # ----------------------------------------------------------------
        tf_str = self.config.get("timeframes", {}).get("primary", "1h")

        def _set_eval(action: str, tid: Optional[str] = None) -> None:
            raw = {"aggregation": agg_result.to_dict(), "layer1": {n: r.to_dict() for n, r in layer1_signals.items()}, "layer2": {n: r.to_dict() for n, r in layer2_signals.items()}, "layer3": {n: r.to_dict() for n, r in layer3_signals.items()}}
            self._last_evaluation = {
                "timestamp": datetime.utcnow().isoformat(),
                "symbol": symbol,
                "timeframe": tf_str,
                "confidence_score": confidence,
                "direction": direction.lower(),
                "layer1_score": agg_result.layer1_score,
                "layer2_score": agg_result.layer2_score,
                "layer3_score": agg_result.layer3_score,
                "ml_score": agg_result.ml_score,
                "raw_signals": raw,
                "action_taken": action,
                "trade_id": tid,
            }

        if direction == "LONG" and confidence >= self.min_confidence_long:
            pass  # proceed
        elif direction == "SHORT" and self.allow_short and confidence >= self.min_confidence_short:
            pass  # proceed
        else:
            _set_eval("skipped_threshold")
            self.logger.debug(
                "[%s] Signal rejected: direction=%s confidence=%.2f "
                "(need LONG>%.1f or SHORT>%.1f allow_short=%s)",
                symbol, direction, confidence,
                self.min_confidence_long, self.min_confidence_short,
                self.allow_short,
            )
            return None

        # ----------------------------------------------------------------
        # Entry price and ATR
        # ----------------------------------------------------------------
        entry_price: float = float(df.iloc[-1]["close"])
        atr: float = self._compute_atr(df)

        self.logger.debug(
            "[%s] entry_price=%.6f  atr=%.6f", symbol, entry_price, atr
        )

        # ----------------------------------------------------------------
        # Risk management – position sizing
        # ----------------------------------------------------------------
        sizing = self.risk_manager.size_position(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            entry_price=entry_price,
            current_equity=current_equity,
            atr=atr,
            open_positions=open_positions,
        )

        if sizing is None:
            _set_eval("skipped_max_positions")
            self.logger.info(
                "[%s] Risk manager rejected position (limits exceeded).", symbol
            )
            return None

        # ----------------------------------------------------------------
        # Assemble signal breakdown
        # ----------------------------------------------------------------
        breakdown: Dict[str, Any] = {
            "aggregation": agg_result.to_dict(),
            "layer1": {
                name: res.to_dict()
                for name, res in layer1_signals.items()
            },
            "layer2": {
                name: res.to_dict()
                for name, res in layer2_signals.items()
            },
            "layer3": {
                name: res.to_dict()
                for name, res in layer3_signals.items()
            },
            "ml_score_raw": ml_value,
            "sizing": {
                "kelly_fraction": sizing.kelly_fraction,
                "risk_amount": sizing.risk_amount,
                "atr_multiplier": sizing.atr_multiplier,
                "stop_distance": sizing.stop_distance,
            },
        }

        # ----------------------------------------------------------------
        # Build and return TradeSignal
        # ----------------------------------------------------------------
        signal = TradeSignal(
            symbol=symbol,
            direction=direction,
            confidence_score=confidence,
            position_size_usd=sizing.position_size_usd,
            entry_price=entry_price,
            stop_loss=sizing.stop_loss,
            take_profit=sizing.take_profit,
            signal_breakdown=breakdown,
            timestamp=datetime.utcnow(),
        )

        # Will set trade_id when engine records the trade
        _set_eval("trade_opened", tid=None)

        self.logger.info(
            "[%s] TradeSignal generated: %s", symbol, signal
        )
        return signal

    # ------------------------------------------------------------------
    # Private – layer computation helpers
    # ------------------------------------------------------------------

    def _compute_layer1(self, df: pd.DataFrame) -> dict:
        """
        Compute all 6 Layer-1 technical signals.

        Returns a dict mapping signal name → SignalResult.
        Each computation is wrapped in try/except so a single failing
        indicator never blocks the whole evaluation.
        """
        results = {}
        signals = [
            ("rsi", self._rsi),
            ("macd", self._macd),
            ("bollinger_bands", self._bb),
            ("ema_crossover", self._ema),
            ("ichimoku", self._ichimoku),
            ("vwap", self._vwap),
        ]
        for name, signal_obj in signals:
            try:
                results[name] = signal_obj.compute(df)
            except Exception as exc:
                self.logger.warning(
                    "Layer1 signal '%s' raised exception: %s — using neutral.",
                    name, exc, exc_info=True,
                )
                results[name] = signal_obj._neutral_result(f"exception: {exc}")
        return results

    def _compute_layer2(self, df: pd.DataFrame) -> dict:
        """Compute all 3 Layer-2 volume signals."""
        results = {}
        signals = [
            ("volume_anomaly", self._volume_anomaly),
            ("obv", self._obv),
            ("cvd", self._cvd),
        ]
        for name, signal_obj in signals:
            try:
                results[name] = signal_obj.compute(df)
            except Exception as exc:
                self.logger.warning(
                    "Layer2 signal '%s' raised exception: %s — using neutral.",
                    name, exc, exc_info=True,
                )
                results[name] = signal_obj._neutral_result(f"exception: {exc}")
        return results

    def _compute_layer3(self, sentiment_data: dict) -> dict:
        """
        Compute all 3 Layer-3 sentiment signals.

        Each sentiment signal has a ``compute(data)`` shim that accepts a dict.
        """
        results = {}
        signals = [
            ("fear_greed", self._fear_greed),
            ("cryptopanic", self._cryptopanic),
            ("google_trends", self._google_trends),
        ]
        for name, signal_obj in signals:
            try:
                results[name] = signal_obj.compute(sentiment_data)
            except Exception as exc:
                self.logger.warning(
                    "Layer3 signal '%s' raised exception: %s — using neutral.",
                    name, exc, exc_info=True,
                )
                results[name] = signal_obj._neutral_result(f"exception: {exc}")
        return results

    def _compute_ml_score(self, df: pd.DataFrame) -> Optional[float]:
        """
        Run ML ensemble inference.

        Returns a score in [-1, +1] suitable for the aggregator's ml_score
        parameter, or None if ML is disabled or inference fails.

        The ensemble's ``predict`` method should return a float in [0, 100]
        where 100 = strong bullish, 0 = strong bearish, 50 = neutral.
        We convert: ml_value = (ml_score / 50.0) - 1.0

        The feature_df uses the last 60 candles of the OHLCV DataFrame.
        """
        if self.ml_ensemble is None:
            return None

        try:
            # Build feature DataFrame from the most recent 60 candles.
            feature_df = df.tail(60).copy()

            # Validate the feature_df has sufficient data before calling predict.
            if len(feature_df) < 10:
                self.logger.warning(
                    "ML inference: insufficient rows in feature_df (%d < 10).",
                    len(feature_df),
                )
                return None

            # The ensemble.predict method returns a float in [0, 100].
            ml_score: float = float(self.ml_ensemble.predict(feature_df))

            # Clamp to valid range in case the model returns out-of-bounds values.
            ml_score = float(np.clip(ml_score, 0.0, 100.0))

            # Convert [0, 100] → [-1, +1] centred at 50.
            ml_value: float = (ml_score / 50.0) - 1.0
            ml_value = float(np.clip(ml_value, -1.0, 1.0))

            self.logger.debug(
                "ML inference: raw_score=%.4f → scaled_value=%+.4f",
                ml_score, ml_value,
            )
            return ml_value

        except Exception as exc:
            self.logger.warning(
                "ML inference failed: %s — proceeding without ML layer.",
                exc, exc_info=True,
            )
            return None

    # ------------------------------------------------------------------
    # ATR computation
    # ------------------------------------------------------------------

    def _compute_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """
        Compute the Average True Range for the last bar.

        Tries pandas_ta first for accuracy; falls back to a pure-pandas
        implementation if pandas_ta is unavailable or returns NaN.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame with at least ``period + 1`` rows and columns
            ``high``, ``low``, ``close``.
        period : int
            ATR lookback period (default 14).

        Returns
        -------
        float
            The ATR value for the most recent bar.  Returns 1% of the closing
            price as a last-resort fallback if computation is impossible.
        """
        required_cols = {"high", "low", "close"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            self.logger.warning("_compute_atr: missing columns %s; using 1%% fallback.", missing)
            return float(df["close"].iloc[-1]) * 0.01 if "close" in df.columns else 1.0

        if len(df) < period + 1:
            # Not enough data: fall back to simple high-low range.
            hl_range = (df["high"] - df["low"]).mean()
            return float(hl_range) if not np.isnan(hl_range) else float(df["close"].iloc[-1]) * 0.01

        # ---- pandas_ta path ----
        try:
            import pandas_ta as ta  # type: ignore[import]
            atr_series = df.ta.atr(length=period)
            if atr_series is not None:
                last_atr = float(atr_series.iloc[-1])
                if not np.isnan(last_atr) and last_atr > 0.0:
                    return last_atr
        except Exception:
            pass  # fall through to manual computation

        # ---- Manual True Range computation ----
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_series = true_range.rolling(window=period, min_periods=period).mean()
        last_atr = float(atr_series.iloc[-1])

        if np.isnan(last_atr) or last_atr <= 0.0:
            # Ultimate fallback: 1% of current close.
            fallback = float(close.iloc[-1]) * 0.01
            self.logger.warning(
                "_compute_atr: manual ATR is NaN/zero, using 1%% fallback = %.6f",
                fallback,
            )
            return fallback

        return last_atr
