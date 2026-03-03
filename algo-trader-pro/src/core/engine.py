"""
src/core/engine.py

TradingEngine — Main Orchestrator
==================================
Central coordinator that wires together every subsystem of AlgoTrader Pro:

  * Data ingestion  — Bybit WebSocket (live) or REST polling (paper / offline)
  * Sentiment       — Fear & Greed, CryptoPanic, Google Trends (scheduled refresh)
  * Strategy        — HybridStrategy evaluation on every closed candle
  * Execution       — PaperExecutor for simulated fills, stop/TP checks
  * Persistence     — OHLCVStore, PortfolioManager, equity snapshots, event log

Execution Modes
---------------
  WebSocket mode  (bybit_ws provided)
    BybitWebSocketClient fires _on_candle_closed whenever a confirmed candle
    arrives.  This is the path used for live-connected paper trading.

  Polling mode   (bybit_ws is None)
    _polling_loop() hits the Bybit REST endpoint every 60 s and detects
    newly closed candles by comparing the `open_time` of the latest bar.
    Used for pure-offline or testnet paper trading without a WS connection.

Scheduler
---------
APScheduler (AsyncIOScheduler) drives periodic sentiment refreshes:
  * Fear & Greed   — every 60 minutes
  * CryptoPanic    — every 15 minutes
  * Google Trends  — every 24 hours

Thread / async safety
---------------------
All public methods are coroutines; the scheduler runs in the same event loop
via AsyncIOScheduler.  PaperExecutor and PortfolioManager are called only
from coroutines so no extra locking is needed for in-process state.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time as _time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from src.utils.db import init_db, get_db
from sqlalchemy import text

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TradingEngine
# ---------------------------------------------------------------------------


class TradingEngine:
    """
    Main trading-engine orchestrator.

    Parameters
    ----------
    config : dict
        Full application configuration dict (loaded from settings.yaml).
    paper_executor :
        Instance of ``PaperExecutor`` — handles order simulation and stop checks.
    portfolio_manager :
        Instance of ``PortfolioManager`` — persists trades and equity snapshots.
    strategy :
        Instance of ``HybridStrategy`` — returns a ``TradeSignal`` or None.
    bybit_ws : optional
        ``BybitWebSocketClient`` for real-time candle feed.
        When *None* the engine falls back to REST polling.
    bybit_rest : optional
        ``BybitRestClient`` for polling and price queries.
    fear_greed_provider : optional
        Provider with an async ``fetch()`` method returning a sentiment dict.
    cryptopanic_provider : optional
        Provider with an async ``fetch_sentiment()`` method.
    google_trends_provider : optional
        Provider with an async ``fetch()`` method.
    ohlcv_store : optional
        ``OHLCVStore`` for reading cached OHLCV history.
    """

    # ------------------------------------------------------------------
    # Sentinel value used to detect first-seen candle per symbol
    # ------------------------------------------------------------------
    _NO_CANDLE: int = -1

    def __init__(
        self,
        config: dict,
        paper_executor: Any,
        portfolio_manager: Any,
        strategy: Any,
        bybit_ws: Optional[Any] = None,
        bybit_rest: Optional[Any] = None,
        coingecko_rest: Optional[Any] = None,
        fear_greed_provider: Optional[Any] = None,
        cryptopanic_provider: Optional[Any] = None,
        google_trends_provider: Optional[Any] = None,
        ohlcv_store: Optional[Any] = None,
    ) -> None:
        # ---- Core components ------------------------------------------------
        self.config = config
        self.paper_executor = paper_executor
        self.portfolio_manager = portfolio_manager
        self.strategy = strategy

        # ---- Optional data-feed components ----------------------------------
        self.bybit_ws = bybit_ws
        self.bybit_rest = bybit_rest
        self.coingecko_rest = coingecko_rest  # fallback provider for CoinGecko polling

        # ---- Sentiment providers --------------------------------------------
        self.fear_greed_provider = fear_greed_provider
        self.cryptopanic_provider = cryptopanic_provider
        self.google_trends_provider = google_trends_provider

        # ---- OHLCV cache ----------------------------------------------------
        self.ohlcv_store = ohlcv_store

        # ---- Internal state -------------------------------------------------
        self.logger = logging.getLogger(__name__)
        self.scheduler = AsyncIOScheduler()
        self.is_running: bool = False
        self._started_at: Optional[datetime] = None

        # Sentiment cache: stores the latest value from each provider.
        # Structure intentionally mirrors what HybridStrategy expects.
        self.sentiment_cache: Dict[str, Any] = {
            "fear_greed": None,       # dict or None
            "cryptopanic": {},        # {symbol: score} or {}
            "google_trends": {},      # {symbol: score} or {}
        }

        # Latest mid-price per symbol (updated from candle close)
        self.current_prices: Dict[str, float] = {}

        # Last seen candle open_time per symbol (used in polling mode to detect new bars)
        self._last_open_time: Dict[str, int] = {}

        # Last signal evaluation per symbol (for /signals/live API)
        self.last_signals: Dict[str, Dict[str, Any]] = {}

        # ---- Symbol list ----------------------------------------------------
        # Priority: config['bybit']['symbols'] → fallback list
        bybit_cfg = config.get("bybit", {})
        self.symbols: List[str] = bybit_cfg.get(
            "symbols", ["BTCUSDT", "ETHUSDT"]
        )

        # Timeframe: Bybit interval code (string), default 60 = 1 h
        self.timeframe: str = str(
            bybit_cfg.get("timeframes", {}).get("primary", "60")
        )

        self.logger.debug(
            "TradingEngine created",
            extra={
                "symbols": self.symbols,
                "timeframe": self.timeframe,
                "ws_mode": bybit_ws is not None,
            },
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """
        Start the trading engine.

        Steps:
          1. Initialise the database schema (idempotent).
          2. Log a startup event to ``bot_events``.
          3. Fetch initial sentiment values.
          4. Start APScheduler for periodic sentiment refreshes.
          5. Connect to Bybit WebSocket *or* start REST-polling loop.
          6. Mark engine as running.
        """
        self.logger.info("Starting TradingEngine...")

        # Step 1 – DB init (idempotent, safe to call on every boot)
        try:
            init_db()
            self.logger.info("Database schema verified / initialised.")
        except Exception as exc:
            self.logger.error("DB initialisation failed: %s", exc)
            raise

        self.is_running = True
        self._started_at = datetime.now(tz=timezone.utc)

        # Step 2 – Log startup event
        await self._log_bot_event("startup", "TradingEngine starting up", {
            "symbols": self.symbols,
            "timeframe": self.timeframe,
            "mode": "paper",
        })

        # Step 2b – Restore open positions from DB (recovery after restart)
        open_trades = self.portfolio_manager.get_open_trades_from_db()
        if open_trades:
            total_realized = self.portfolio_manager.get_total_realized_pnl()
            self.paper_executor.restore_open_positions(
                open_trades, total_realized_pnl=total_realized
            )

        # Step 3 – Initial sentiment fetch (best-effort; failures are non-fatal)
        self.logger.info("Fetching initial sentiment data...")
        await asyncio.gather(
            self._fetch_fear_greed(),
            self._fetch_cryptopanic(),
            self._fetch_google_trends(),
            return_exceptions=True,
        )

        # Step 4 – Start sentiment scheduler
        self._start_sentiment_scheduler()

        # Step 5 – Connect data feed
        if self.bybit_ws is not None:
            self.logger.info("Connecting Bybit WebSocket...")
            await self._start_websocket_feed()
        else:
            # CoinGecko (o altro REST) richiede connect() prima del polling
            if self.coingecko_rest is not None:
                try:
                    await self.coingecko_rest.connect()
                    self.logger.info("CoinGecko REST provider connected.")
                except Exception as exc:
                    self.logger.warning("CoinGecko connect failed: %s — polling may fail.", exc)
            self.logger.info("Bybit WS not provided — starting REST polling loop.")
            asyncio.get_event_loop().create_task(self._polling_loop())

        self.logger.info(
            "TradingEngine started. Symbols=%s, Timeframe=%s, Mode=paper",
            self.symbols,
            self.timeframe,
        )

    async def stop(self) -> None:
        """
        Gracefully stop the trading engine.

        Shuts down the APScheduler, disconnects the WebSocket (if active),
        and logs a shutdown event.
        """
        self.logger.info("Stopping TradingEngine...")
        self.is_running = False

        # Shut down scheduler without waiting for running jobs to complete
        try:
            if self.scheduler.running:
                self.scheduler.shutdown(wait=False)
                self.logger.info("APScheduler stopped.")
        except Exception as exc:
            self.logger.warning("Error stopping scheduler: %s", exc)

        # Disconnect WebSocket
        if self.bybit_ws is not None:
            try:
                await self.bybit_ws.disconnect()
                self.logger.info("Bybit WebSocket disconnected.")
            except Exception as exc:
                self.logger.warning("Error disconnecting WS: %s", exc)

        # Disconnect CoinGecko REST
        if self.coingecko_rest is not None:
            try:
                await self.coingecko_rest.disconnect()
                self.logger.info("CoinGecko REST disconnected.")
            except Exception as exc:
                self.logger.warning("Error disconnecting CoinGecko: %s", exc)

        # Log shutdown event
        try:
            await self._log_bot_event("shutdown", "TradingEngine stopped", {})
        except Exception:
            pass  # Best-effort; don't raise during shutdown

        self.logger.info("TradingEngine stopped.")

    # ------------------------------------------------------------------
    # Core candle-processing pipeline
    # ------------------------------------------------------------------

    async def _on_candle_closed(self, symbol: str, ohlcv_row: dict) -> None:
        """
        Called once per closed (confirmed) OHLCV candle.

        Parameters
        ----------
        symbol : str
            Trading pair, e.g. ``"BTCUSDT"``.
        ohlcv_row : dict
            Single candle data with keys:
            ``open_time, open, high, low, close, volume, close_time``

        Pipeline:
          1. Persist new candle row to OHLCVStore.
          2. Load the last 200 candles as a DataFrame.
          3. Update ``current_prices[symbol]``.
          4. Check stop-loss / take-profit on open positions.
          5. Update trailing stops.
          6. Evaluate strategy → optionally open a new position.
          7. Save equity snapshot.
        """
        if not self.is_running:
            return

        self.logger.debug("Candle closed: symbol=%s open_time=%s close=%s",
                          symbol, ohlcv_row.get("open_time"), ohlcv_row.get("close"))

        # ------------------------------------------------------------------
        # 1. Persist new candle to the OHLCV store
        # ------------------------------------------------------------------
        if self.ohlcv_store is not None:
            try:
                new_row_df = pd.DataFrame([ohlcv_row])
                self.ohlcv_store.save(new_row_df, symbol=symbol, timeframe=self.timeframe)
            except Exception as exc:
                self.logger.warning("OHLCVStore save failed for %s: %s", symbol, exc)

        # ------------------------------------------------------------------
        # 2. Load recent candle history (last 200 bars) as a DataFrame
        # ------------------------------------------------------------------
        df: Optional[pd.DataFrame] = None
        if self.ohlcv_store is not None:
            try:
                # load() accepts start_ms/end_ms, not limit
                # Compute start_ms = 200 candles back (timeframe is a string like "60" or "1h")
                try:
                    _tf_minutes = int(self.timeframe)
                except (ValueError, TypeError):
                    _tf_minutes = 60  # default 1h
                _start_ms = int((_time.time() - 200 * _tf_minutes * 60) * 1000)
                df = self.ohlcv_store.load(
                    symbol=symbol,
                    timeframe=self.timeframe,
                    start_ms=_start_ms,
                )
            except Exception as exc:
                self.logger.warning("OHLCVStore load failed for %s: %s", symbol, exc)

        # Fallback: build a minimal single-row DataFrame from the new row
        if df is None or df.empty:
            df = pd.DataFrame([ohlcv_row])

        # ------------------------------------------------------------------
        # 3. Update current price
        # ------------------------------------------------------------------
        current_price: float = float(ohlcv_row.get("close", 0.0))
        self.current_prices[symbol] = current_price

        # Build the full price map for multi-symbol stop checks
        current_prices_map: Dict[str, float] = dict(self.current_prices)

        # Usa low/high della candela per stop/TP (cattura il minimo della candela!)
        candle_low = float(ohlcv_row.get("low", current_price))
        candle_high = float(ohlcv_row.get("high", current_price))
        candle_extremes = {symbol: (candle_low, candle_high)}

        # ------------------------------------------------------------------
        # 4. Check stops (stop-loss, take-profit) on all open positions
        # ------------------------------------------------------------------
        try:
            triggered_trades = self.paper_executor.check_stops(
                current_prices_map, candle_extremes=candle_extremes
            )
        except Exception as exc:
            self.logger.error("check_stops error for %s: %s", symbol, exc)
            triggered_trades = []

        # Close out any triggered positions
        # check_stops() returns List[Tuple[trade_id, exit_price, exit_reason]]
        for trade_id, exit_price, exit_reason in triggered_trades:
            try:
                closed_trade = self.paper_executor.execute_sell(
                    trade_id=trade_id,
                    exit_price=exit_price,
                    exit_reason=exit_reason,
                )
                if closed_trade is not None and self.portfolio_manager is not None:
                    self.portfolio_manager.record_trade_close(closed_trade)
                    self.logger.info(
                        "Position closed: symbol=%s id=%s reason=%s pnl=%.2f",
                        closed_trade.symbol,
                        closed_trade.trade_id,
                        closed_trade.exit_reason,
                        closed_trade.net_pnl,
                    )
                    await self._log_bot_event(
                        "trade_closed",
                        f"Trade closed: {closed_trade.symbol} {closed_trade.direction}",
                        {
                            "trade_id": closed_trade.trade_id,
                            "exit_reason": closed_trade.exit_reason,
                            "net_pnl": closed_trade.net_pnl,
                            "pnl_pct": closed_trade.pnl_pct,
                        },
                    )
            except Exception as exc:
                self.logger.error("Error closing triggered trade %s: %s",
                                  trade_id, exc)

        # ------------------------------------------------------------------
        # 5. Update trailing stops for remaining open positions
        # ------------------------------------------------------------------
        try:
            self.paper_executor.update_trailing_stops(current_prices_map)
        except Exception as exc:
            self.logger.warning("update_trailing_stops error: %s", exc)

        # ------------------------------------------------------------------
        # 6. Evaluate strategy
        # ------------------------------------------------------------------
        # Gather inputs for strategy evaluation
        try:
            current_equity: float = (
                self.paper_executor.get_portfolio_value(current_prices_map)
                if hasattr(self.paper_executor, "get_portfolio_value")
                else 10_000.0
            )
        except Exception:
            current_equity = 10_000.0

        try:
            open_positions: list = self.paper_executor.get_open_positions()
        except Exception:
            open_positions = []

        # Only evaluate strategy for the candle's symbol
        signal = None
        if self.strategy is not None:
            try:
                signal = await self.strategy.evaluate(
                    symbol=symbol,
                    df=df,
                    sentiment_data=self.sentiment_cache,
                    current_equity=current_equity,
                    open_positions=open_positions,
                )
                # Log every evaluation to signals_log (for dashboard + Analisi)
                eval_data = getattr(self.strategy, "_last_evaluation", None)
                if eval_data is not None:
                    self.last_signals[symbol] = {
                        "confidence_score": eval_data.get("confidence_score", 0),
                        "direction": eval_data.get("direction", "neutral"),
                        "layer1_score": eval_data.get("layer1_score", 0),
                        "layer2_score": eval_data.get("layer2_score", 0),
                        "layer3_score": eval_data.get("layer3_score", 0),
                        "ml_score": eval_data.get("ml_score", 0),
                        "action_taken": eval_data.get("action_taken", "skipped_threshold"),
                        "timestamp": eval_data.get("timestamp", ""),
                    }
                    if self.portfolio_manager is not None:
                        try:
                            self.portfolio_manager.log_signal(eval_data)
                        except Exception as log_exc:
                            self.logger.warning("log_signal failed: %s", log_exc)
            except Exception as exc:
                self.logger.error("Strategy evaluate error for %s: %s", symbol, exc)

        # ------------------------------------------------------------------
        # 7. Execute signal if returned (evita duplicati: 1 posizione per symbol)
        # ------------------------------------------------------------------
        if signal is not None:
            # Blocca se abbiamo già una posizione aperta sullo stesso symbol
            symbols_with_position = {t.symbol for t in open_positions}
            if signal.symbol in symbols_with_position:
                self.logger.info(
                    "Skip open %s: già presente posizione aperta su questo symbol",
                    signal.symbol,
                )
                signal = None

        if signal is not None:
            self.logger.info(
                "Signal received: symbol=%s direction=%s confidence=%.2f entry=%.4f",
                signal.symbol,
                signal.direction,
                signal.confidence_score,
                signal.entry_price,
            )
            try:
                quantity = (
                    signal.position_size_usd / signal.entry_price
                    if signal.entry_price > 0
                    else 0.0
                )
                from src.risk.risk_manager import leverage_from_confidence
                leverage = leverage_from_confidence(signal.confidence_score, self.config)

                new_trade = self.paper_executor.execute_buy(
                    symbol=signal.symbol,
                    direction=signal.direction,
                    entry_price=signal.entry_price,
                    position_size_usd=signal.position_size_usd,
                    quantity=quantity,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    confidence_score=signal.confidence_score,
                    signal_breakdown=signal.signal_breakdown,
                    kelly_fraction=getattr(signal, "kelly_fraction", 0.0),
                    risk_amount=getattr(signal, "risk_amount", 0.0),
                    leverage=leverage,
                )
                if new_trade is not None and self.portfolio_manager is not None:
                    self.portfolio_manager.record_trade_open(new_trade)
                    self.logger.info(
                        "Trade opened: symbol=%s id=%s qty=%.6f sl=%.4f tp=%.4f",
                        new_trade.symbol,
                        new_trade.trade_id,
                        new_trade.quantity,
                        new_trade.stop_loss,
                        new_trade.take_profit,
                    )
                    await self._log_bot_event(
                        "trade_opened",
                        f"Trade opened: {new_trade.symbol} {new_trade.direction}",
                        {
                            "trade_id": new_trade.trade_id,
                            "direction": new_trade.direction,
                            "entry_price": new_trade.entry_price,
                            "quantity": new_trade.quantity,
                            "stop_loss": new_trade.stop_loss,
                            "take_profit": new_trade.take_profit,
                            "confidence_score": new_trade.confidence_score,
                        },
                    )
            except Exception as exc:
                self.logger.error("execute_buy error for %s: %s", symbol, exc)
                await self._log_bot_event(
                    "error",
                    f"execute_buy failed for {symbol}",
                    {"error": str(exc)},
                )

        # ------------------------------------------------------------------
        # 8. Save equity snapshot every candle
        # ------------------------------------------------------------------
        if self.portfolio_manager is not None:
            try:
                # save_equity_snapshot() expects a snapshot dict from PaperExecutor
                _snapshot = self.paper_executor.get_equity_snapshot(current_prices_map)
                self.portfolio_manager.save_equity_snapshot(_snapshot)
            except Exception as exc:
                self.logger.warning("save_equity_snapshot error: %s", exc)

    # ------------------------------------------------------------------
    # REST polling loop (fallback when WebSocket is unavailable)
    # ------------------------------------------------------------------

    async def _polling_loop(self) -> None:
        """
        Poll Bybit REST every 60 seconds for the latest closed candle.

        For each symbol: fetch the most recent kline bar and compare its
        ``open_time`` against ``_last_open_time``.  If a newer bar has
        appeared, call ``_on_candle_closed``.

        Handles transient errors gracefully: on failure for a symbol the
        loop logs a warning and retries on the next iteration.
        """
        POLL_INTERVAL_S = 60  # seconds between polls

        self.logger.info(
            "REST polling loop started. interval=%ds symbols=%s",
            POLL_INTERVAL_S, self.symbols,
        )

        while self.is_running:
            for symbol in self.symbols:
                if not self.is_running:
                    break
                try:
                    await self._poll_symbol(symbol)
                except Exception as exc:
                    self.logger.warning(
                        "Polling error for %s: %s — will retry next interval.",
                        symbol, exc,
                    )

            # Sleep before next round; break early if stopped
            for _ in range(POLL_INTERVAL_S):
                if not self.is_running:
                    break
                await asyncio.sleep(1)

        self.logger.info("REST polling loop exited.")

    async def _poll_symbol(self, symbol: str) -> None:
        """
        Fetch the latest kline for *symbol* and fire ``_on_candle_closed``
        if a new bar has appeared.

        Priority:
          1. Bybit REST (fetch_klines) — full OHLCV bar
          2. CoinGecko REST (fetch_current_price) — synthetic candle from spot price
        """
        # ------------------------------------------------------------------
        # Path A: Bybit REST provider
        # ------------------------------------------------------------------
        if self.bybit_rest is not None:
            loop = asyncio.get_event_loop()
            try:
                df: pd.DataFrame = await loop.run_in_executor(
                    None,
                    lambda: self.bybit_rest.fetch_klines(
                        symbol=symbol,
                        interval=self.timeframe,
                        limit=2,  # fetch 2 bars: latest closed + forming
                    ),
                )
            except Exception as exc:
                raise RuntimeError(f"REST fetch_klines failed for {symbol}: {exc}") from exc

            if df is None or df.empty:
                self.logger.debug("Empty kline response for %s", symbol)
                return

            # The REST client returns bars newest-first; the second row (index 1)
            # is the most recently *closed* bar (index 0 is still forming).
            if len(df) >= 2:
                closed_bar = df.iloc[1]
            else:
                closed_bar = df.iloc[0]

            new_open_time: int = int(closed_bar["open_time"])
            last_seen: int = self._last_open_time.get(symbol, self._NO_CANDLE)

            if new_open_time > last_seen:
                self._last_open_time[symbol] = new_open_time
                ohlcv_row = {
                    "open_time":  new_open_time,
                    "open":       float(closed_bar["open"]),
                    "high":       float(closed_bar["high"]),
                    "low":        float(closed_bar["low"]),
                    "close":      float(closed_bar["close"]),
                    "volume":     float(closed_bar["volume"]),
                    "close_time": int(closed_bar.get("close_time", new_open_time)),
                }
                await self._on_candle_closed(symbol, ohlcv_row)
            return

        # ------------------------------------------------------------------
        # Path B: CoinGecko fallback — build a synthetic candle from spot price
        # ------------------------------------------------------------------
        if self.coingecko_rest is not None:
            try:
                price: float = await self.coingecko_rest.fetch_current_price(symbol)
            except Exception as exc:
                self.logger.warning("CoinGecko price fetch failed for %s: %s", symbol, exc)
                return

            # Align to the current candle bucket (timeframe: "5","15","60","240" or "D")
            _tf_map = {"5": 5, "15": 15, "60": 60, "240": 240, "D": 1440}
            _tf_minutes = _tf_map.get(str(self.timeframe))
            if _tf_minutes is None:
                try:
                    _tf_minutes = int(self.timeframe)
                except (ValueError, TypeError):
                    _tf_minutes = 60
            now_ms = int(_time.time() * 1000)
            candle_open_time = now_ms - (now_ms % (_tf_minutes * 60 * 1000))

            last_seen = self._last_open_time.get(symbol, self._NO_CANDLE)

            if candle_open_time > last_seen:
                self._last_open_time[symbol] = candle_open_time
                ohlcv_row = {
                    "open_time":  candle_open_time,
                    "open":       price,
                    "high":       price,
                    "low":        price,
                    "close":      price,
                    "volume":     0.0,
                    "close_time": candle_open_time + _tf_minutes * 60 * 1000 - 1,
                }
                self.logger.debug(
                    "CoinGecko synthetic candle: %s close=%.4f open_time=%d",
                    symbol, price, candle_open_time,
                )
                await self._on_candle_closed(symbol, ohlcv_row)
            return

        # ------------------------------------------------------------------
        # No provider available
        # ------------------------------------------------------------------
        self.logger.debug("No REST provider configured — cannot poll %s.", symbol)

    # ------------------------------------------------------------------
    # WebSocket feed setup
    # ------------------------------------------------------------------

    async def _start_websocket_feed(self) -> None:
        """
        Register a candle callback with the Bybit WebSocket client and
        initiate the connection.  The WS client is responsible for
        reconnection logic.
        """

        async def _ws_callback(symbol: str, ohlcv_row: dict) -> None:
            """Adapts the WS client callback signature to our pipeline."""
            try:
                await self._on_candle_closed(symbol, ohlcv_row)
            except Exception as exc:
                self.logger.error("WS callback error for %s: %s", symbol, exc)

        # Bybit interval code ("5", "60", ...) → WS expects human-readable ("5m", "1h", ...)
        _BYBIT_TO_WS: Dict[str, str] = {"5": "5m", "15": "15m", "60": "1h", "240": "4h", "D": "1d"}
        ws_timeframe: str = _BYBIT_TO_WS.get(self.timeframe, "5m")

        ws_subscribed = False
        try:
            self.bybit_ws.subscribe(
                symbols=self.symbols,
                timeframe=ws_timeframe,
                callback=_ws_callback,
            )
            ws_subscribed = True
            self.logger.info("WS subscribed: %s @ %s", self.symbols, ws_timeframe)
        except Exception as exc:
            self.logger.error("WS subscribe failed: %s", exc)
            self.logger.info(
                "WS subscribe failed — falling back to REST polling loop."
            )
            asyncio.get_event_loop().create_task(self._polling_loop())
            return  # Do not attempt to connect a WS that has no subscriptions

        if ws_subscribed:
            try:
                await self.bybit_ws.connect()
                self.logger.info("Bybit WebSocket connected.")
            except Exception as exc:
                self.logger.error("WS connect failed: %s", exc)
                self.logger.info("Falling back to REST polling after WS connect failure.")
                asyncio.get_event_loop().create_task(self._polling_loop())

    # ------------------------------------------------------------------
    # Sentiment scheduler
    # ------------------------------------------------------------------

    def _start_sentiment_scheduler(self) -> None:
        """
        Configure and start the APScheduler with three periodic jobs:

        * ``fear_greed``     — every 60 minutes
        * ``cryptopanic``    — every 15 minutes
        * ``google_trends``  — every 24 hours

        Each job is wrapped so that failures do not crash the scheduler.
        """
        self.scheduler.add_job(
            self._fetch_fear_greed,
            trigger="interval",
            minutes=60,
            id="fear_greed",
            name="Fear & Greed refresh",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )
        self.scheduler.add_job(
            self._fetch_cryptopanic,
            trigger="interval",
            minutes=15,
            id="cryptopanic",
            name="CryptoPanic refresh",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )
        self.scheduler.add_job(
            self._fetch_google_trends,
            trigger="interval",
            hours=24,
            id="google_trends",
            name="Google Trends refresh",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )
        self.scheduler.add_job(
            self._refresh_prices_and_broadcast_equity,
            trigger="interval",
            seconds=10,
            id="price_equity_broadcast",
            name="Prezzi + broadcast equity (mark-to-market ogni 10s)",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )

        try:
            self.scheduler.start()
            self.logger.info(
                "APScheduler started with %d sentiment jobs.",
                len(self.scheduler.get_jobs()),
            )
        except Exception as exc:
            self.logger.error("Failed to start APScheduler: %s", exc)

    # ------------------------------------------------------------------
    # Sentiment fetchers
    # ------------------------------------------------------------------

    async def _fetch_fear_greed(self) -> None:
        """
        Refresh the Fear & Greed index from the provider.
        On any error the cached value is preserved and a warning is logged.
        """
        if self.fear_greed_provider is None:
            return
        try:
            result = await self.fear_greed_provider.fetch()
            self.sentiment_cache["fear_greed"] = result
            self.logger.debug(
                "Fear & Greed updated: value=%s classification=%s",
                result.get("value") if isinstance(result, dict) else result,
                result.get("value_classification") if isinstance(result, dict) else "?",
            )
        except Exception as exc:
            self.logger.warning(
                "Fear & Greed fetch failed (keeping cached value): %s", exc
            )

    async def _fetch_cryptopanic(self) -> None:
        """
        Refresh CryptoPanic sentiment scores from the provider.
        On any error the cached value is preserved and a warning is logged.
        """
        if self.cryptopanic_provider is None:
            return
        try:
            result = await self.cryptopanic_provider.fetch_sentiment(self.symbols)
            self.sentiment_cache["cryptopanic"] = result
            if isinstance(result, dict):
                self.sentiment_cache.update(result)
            self.logger.debug("CryptoPanic sentiment updated: %s", result)
        except Exception as exc:
            self.logger.warning(
                "CryptoPanic fetch failed (keeping cached value): %s", exc
            )

    async def _fetch_google_trends(self) -> None:
        """
        Refresh Google Trends interest scores from the provider.
        On any error the cached value is preserved and a warning is logged.
        """
        if self.google_trends_provider is None:
            return
        try:
            result = await self.google_trends_provider.fetch()
            self.sentiment_cache["google_trends"] = result
            if isinstance(result, dict):
                self.sentiment_cache.update(result)
            self.logger.debug("Google Trends updated: %s", result)
        except Exception as exc:
            self.logger.warning(
                "Google Trends fetch failed (keeping cached value): %s", exc
            )

    async def _refresh_prices_and_broadcast_equity(self) -> None:
        """
        Aggiorna i prezzi correnti, salva equity snapshot in DB e invia via WebSocket.
        Eseguito ogni 30 secondi. Supporta Bybit e CoinGecko.
        """
        if not self.is_running:
            return
        try:
            # Fetch prices from Bybit (sync) or CoinGecko (async)
            if self.bybit_rest is not None:
                loop = asyncio.get_event_loop()
                for symbol in self.symbols:
                    try:
                        price = await loop.run_in_executor(
                            None,
                            lambda s=symbol: self.bybit_rest.fetch_current_price(s),
                        )
                        self.current_prices[symbol] = price
                    except Exception as exc:
                        self.logger.debug("Price fetch failed for %s: %s", symbol, exc)
            elif self.coingecko_rest is not None:
                for symbol in self.symbols:
                    try:
                        price = await self.coingecko_rest.fetch_current_price(symbol)
                        self.current_prices[symbol] = price
                    except Exception as exc:
                        self.logger.warning(
                            "Price fetch failed for %s: %s — prezzi dashboard non aggiornati",
                            symbol, exc,
                        )

            current_prices_map = dict(self.current_prices)

            # Controllo stop-loss/take-profit intra-candela (ogni 10s)
            # Così catturiamo cali rapidi come il -2k BTC che altrimenti
            # sfuggirebbero (check solo a chiusura candela ogni 5 min)
            try:
                triggered = self.paper_executor.check_stops(current_prices_map)
                for trade_id, exit_price, exit_reason in triggered:
                    try:
                        closed = self.paper_executor.execute_sell(
                            trade_id=trade_id,
                            exit_price=exit_price,
                            exit_reason=exit_reason,
                        )
                        if closed and self.portfolio_manager:
                            self.portfolio_manager.record_trade_close(closed)
                            self.logger.info(
                                "Position closed (intra-candle): %s id=%s reason=%s pnl=%.2f",
                                closed.symbol, closed.trade_id[:8], exit_reason, closed.net_pnl,
                            )
                            await self._log_bot_event(
                                "trade_closed",
                                f"Trade closed: {closed.symbol} {closed.direction}",
                                {"trade_id": closed.trade_id, "exit_reason": exit_reason, "net_pnl": closed.net_pnl},
                            )
                    except Exception as exc:
                        self.logger.warning("Error closing triggered trade %s: %s", trade_id[:8], exc)
            except Exception as exc:
                self.logger.debug("check_stops in price refresh failed: %s", exc)

            equity = (
                self.paper_executor.get_portfolio_value(current_prices_map)
                if hasattr(self.paper_executor, "get_portfolio_value")
                else 10_000.0
            )
            cash = getattr(self.paper_executor, "cash", equity)

            # Timestamp UTC con suffisso Z per corretto parsing JS
            now_utc = datetime.now(tz=timezone.utc)
            ts_iso = now_utc.isoformat().replace("+00:00", "Z")

            # Salva equity snapshot in DB ogni 30 sec (Equity Curve dinamica)
            if self.portfolio_manager is not None:
                try:
                    snapshot = self.paper_executor.get_equity_snapshot(current_prices_map)
                    snapshot["timestamp"] = ts_iso
                    self.portfolio_manager.save_equity_snapshot(snapshot)
                except Exception as exc:
                    self.logger.debug("save_equity_snapshot failed: %s", exc)

            # Broadcast via WebSocket
            from src.api.server import get_feed
            feed = get_feed()
            await feed.broadcast_equity_update({
                "equity": round(equity, 4),
                "cash": round(cash, 4),
                "drawdown_pct": 0.0,
                "timestamp": ts_iso,
            })
            # Broadcast bot_status e market_tick per dashboard real-time
            await feed.broadcast("bot_status", {"status": "RUNNING" if self.is_running else "STOPPED"})
            for sym, pr in current_prices_map.items():
                if pr and pr > 0:
                    await feed.broadcast_market_tick({"symbol": sym, "price": pr, "change_24h_pct": 0.0})
        except Exception as exc:
            self.logger.debug("Price/equity broadcast failed: %s", exc)

    # ------------------------------------------------------------------
    # Utility: status dict
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """
        Return a lightweight status dictionary for the API / dashboard.

        Returns
        -------
        dict
            Keys: ``status``, ``started_at``, ``symbols``, ``timeframe``,
            ``mode``, ``current_prices``, ``sentiment_cache``.
        """
        return {
            "status": "RUNNING" if self.is_running else "STOPPED",
            "started_at": (
                self._started_at.isoformat() if self._started_at else None
            ),
            "symbols": self.symbols,
            "timeframe": self.timeframe,
            "mode": "paper",
            "current_prices": dict(self.current_prices),
            "sentiment_cache": {
                "fear_greed": self.sentiment_cache.get("fear_greed"),
                "cryptopanic_symbols": list(
                    self.sentiment_cache.get("cryptopanic", {}).keys()
                ),
                "google_trends_symbols": list(
                    self.sentiment_cache.get("google_trends", {}).keys()
                ),
            },
            "open_positions_count": self._count_open_positions(),
        }

    def _count_open_positions(self) -> int:
        """Return the number of currently open positions (best-effort)."""
        try:
            return len(self.paper_executor.get_open_positions())
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # Internal: bot event logger
    # ------------------------------------------------------------------

    async def _log_bot_event(
        self,
        event_type: str,
        reason: str,
        details: dict,
    ) -> None:
        """
        Persist a row to ``bot_events`` in a fire-and-forget manner.
        Failures are logged at DEBUG level so as not to disturb the main flow.
        """
        import json as _json

        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, self._write_bot_event, event_type, reason, details)
        except Exception as exc:
            self.logger.debug("bot_events write failed: %s", exc)

    @staticmethod
    def _write_bot_event(event_type: str, reason: str, details: dict) -> None:
        """Synchronous DB write, safe to call from a thread executor."""
        import json as _json

        try:
            with get_db() as session:
                session.execute(
                    text(
                        "INSERT INTO bot_events (event_type, reason, details) "
                        "VALUES (:et, :reason, :details)"
                    ),
                    {
                        "et": event_type,
                        "reason": reason,
                        "details": _json.dumps(details),
                    },
                )
        except Exception:
            pass  # Non-fatal; DB may not be ready yet during unit tests
