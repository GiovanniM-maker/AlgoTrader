"""
src/api/routes/dashboard.py
============================
Dashboard summary and market-feed endpoints.

GET /api/v1/dashboard/summary
    Returns the top-level portfolio KPIs shown in the dashboard header
    cards: equity, cash, P&L, win rate, Fear & Greed index, etc.

GET /api/v1/dashboard/market-feed
    Returns real-time (or near-real-time) market tickers for the four
    primary trading pairs: BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT.

Data priority (summary)
-----------------------
1. Live data from the TradingEngine (if running).
2. Database records (equity_snapshots, trades, sentiment_data).
3. Safe defaults / zeros (if DB is also empty – first launch).

Data priority (market-feed)
---------------------------
1. ``engine_instance.current_prices`` (if engine is running and has prices).
2. Bybit public REST API (no authentication required).
3. Zeroed placeholder ticks (if Bybit is unreachable).
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

import src.api.server as _server_module  # late import to avoid circular deps

logger = logging.getLogger(__name__)
router = APIRouter()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DB_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "database", "algotrader.db"
)


def _get_db_path() -> str:
    """Resolve the database path, preferring the DATABASE_URL env var."""
    env_url = os.getenv("DATABASE_URL", "")
    if env_url.startswith("sqlite:///"):
        return env_url.replace("sqlite:///", "", 1)
    if env_url.startswith("sqlite://./"):
        return env_url.replace("sqlite://./", "", 1)
    # Default: relative from project root
    base = os.path.dirname(__file__)
    return os.path.normpath(os.path.join(base, "..", "..", "..", "database", "algotrader.db"))


def _connect() -> sqlite3.Connection:
    """Open the SQLite database in read-only mode (WAL-compatible)."""
    db_path = _get_db_path()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _safe_connect() -> Optional[sqlite3.Connection]:
    """Return a DB connection, or None if the database file does not exist yet."""
    try:
        return _connect()
    except Exception as exc:
        logger.warning("Could not open database: %s", exc)
        return None


def _fear_greed_label(value: Optional[float]) -> str:
    """Map a Fear & Greed [0-100] value to its human-readable label."""
    if value is None:
        return "Unknown"
    if value <= 25:
        return "Extreme Fear"
    if value <= 45:
        return "Fear"
    if value <= 55:
        return "Neutral"
    if value <= 75:
        return "Greed"
    return "Extreme Greed"


# ---------------------------------------------------------------------------
# GET /dashboard/summary
# ---------------------------------------------------------------------------


@router.get(
    "/dashboard/summary",
    summary="Portfolio summary for the dashboard header cards.",
    tags=["dashboard"],
)
async def get_dashboard_summary() -> JSONResponse:
    """
    Return the top-level portfolio KPIs.

    Response fields
    ---------------
    equity            : float  – Current total portfolio value (USDT).
    cash              : float  – Uninvested cash (USDT).
    initial_capital   : float  – Starting capital (USDT).
    pnl_today_pct     : float  – Today's P&L percentage.
    pnl_total_pct     : float  – Total P&L percentage since inception.
    win_rate          : float  – Fraction of closed trades that were profitable.
    open_positions_count : int – Number of open positions.
    bot_status        : str   – 'RUNNING', 'PAUSED', or 'STOPPED'.
    fear_greed_value  : float | null – Latest Fear & Greed [0-100].
    fear_greed_label  : str   – Human-readable F&G classification.
    last_updated      : str   – ISO-8601 UTC timestamp.
    """
    engine = _server_module.engine_instance
    now_utc = datetime.now(tz=timezone.utc)
    initial_capital: float = 10_000.0

    # ---- Defaults ----
    equity: float = initial_capital
    cash: float = initial_capital
    pnl_today_pct: float = 0.0
    pnl_total_pct: float = 0.0
    win_rate: float = 0.0
    total_closed: int = 0
    open_positions_count: int = 0
    bot_status: str = "STOPPED"
    fear_greed_value: Optional[float] = None
    fear_greed_label_str: str = "Unknown"

    # ---- 1. Try live engine data (priorità su DB) ----
    got_live_equity = False
    if engine is not None:
        try:
            status = engine.get_status()  # type: ignore[attr-defined]
            bot_status = status.get("status", "STOPPED")
            open_positions_count = status.get("open_positions_count", 0)

            # Live portfolio value ha priorità su DB
            if hasattr(engine, "paper_executor") and engine.paper_executor is not None:
                pe = engine.paper_executor
                current_prices: Dict[str, float] = dict(getattr(engine, "current_prices", {}))
                try:
                    equity = pe.get_portfolio_value(current_prices)
                    cash = getattr(pe, "cash", equity)
                    got_live_equity = True
                except Exception:
                    pass
        except Exception as exc:
            logger.warning("Could not read engine status: %s", exc)

    # ---- Commissioni e verifica (default) ----
    fees_today: float = 0.0
    fees_total: float = 0.0
    total_pnl_realized: float = 0.0
    min_equity_24h: Optional[float] = None

    # ---- 2. Query database (fallback per equity se engine non disponibile) ----
    conn = _safe_connect()
    if conn is not None:
        try:
            # Equity/cash da DB solo se non abbiamo dati live (escludi bogus)
            if not got_live_equity:
                row = conn.execute(
                    """SELECT equity, cash FROM equity_snapshots
                       WHERE NOT (equity=10000 AND positions_value=0 AND open_trades=0)
                       ORDER BY timestamp DESC LIMIT 1"""
                ).fetchone()
                if row:
                    equity = float(row["equity"])
                    cash = float(row["cash"])

            # Today's P&L: compare today's first snapshot to latest (escludi bogus)
            today_start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
            first_today = conn.execute(
                """SELECT equity FROM equity_snapshots
                   WHERE timestamp >= ? AND NOT (equity=10000 AND positions_value=0 AND open_trades=0)
                   ORDER BY timestamp ASC LIMIT 1""",
                (today_start,),
            ).fetchone()
            if first_today:
                equity_start_of_day = float(first_today["equity"])
                if equity_start_of_day > 0:
                    pnl_today_pct = (equity - equity_start_of_day) / equity_start_of_day * 100.0

            # Total P&L (escludi bogus 10000/0/0)
            first_snapshot = conn.execute(
                """SELECT equity FROM equity_snapshots
                   WHERE NOT (equity=10000 AND positions_value=0 AND open_trades=0)
                   ORDER BY timestamp ASC LIMIT 1"""
            ).fetchone()
            if first_snapshot:
                first_equity = float(first_snapshot["equity"])
                if first_equity > 0:
                    pnl_total_pct = (equity - first_equity) / first_equity * 100.0

            # Win rate and total trades from closed trades
            total_row = conn.execute(
                "SELECT COUNT(*) as cnt FROM trades WHERE status = 'closed'"
            ).fetchone()
            win_row = conn.execute(
                "SELECT COUNT(*) as cnt FROM trades WHERE status = 'closed' AND net_pnl > 0"
            ).fetchone()
            total_closed = int(total_row["cnt"]) if total_row else 0
            total_wins = int(win_row["cnt"]) if win_row else 0
            if total_closed > 0:
                win_rate = total_wins / total_closed

            # Open positions count (override engine value if DB query available)
            open_row = conn.execute(
                "SELECT COUNT(*) as cnt FROM trades WHERE status = 'open'"
            ).fetchone()
            if open_row:
                open_positions_count = int(open_row["cnt"])

            # Fear & Greed from sentiment_data (source = 'fear_greed')
            fg_row = conn.execute(
                "SELECT value, raw_data FROM sentiment_data "
                "WHERE source = 'fear_greed' ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            if fg_row:
                fear_greed_value = float(fg_row["value"])
                # Try to get label from raw_data JSON first
                try:
                    raw = json.loads(fg_row["raw_data"] or "{}")
                    fear_greed_label_str = raw.get(
                        "value_classification", _fear_greed_label(fear_greed_value)
                    )
                except Exception:
                    fear_greed_label_str = _fear_greed_label(fear_greed_value)

            # Commissioni: oggi e totali
            today_start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
            fees_today_row = conn.execute(
                """
                SELECT
                    (SELECT COALESCE(SUM(entry_fee), 0) FROM trades WHERE entry_time >= ?)
                    + (SELECT COALESCE(SUM(COALESCE(exit_fee, 0)), 0) FROM trades WHERE exit_time IS NOT NULL AND exit_time >= ?)
                """,
                (today_start, today_start),
            ).fetchone()
            fees_today = float(fees_today_row[0]) if fees_today_row else 0.0

            fees_total_row = conn.execute(
                "SELECT COALESCE(SUM(COALESCE(entry_fee, 0) + COALESCE(exit_fee, 0)), 0) FROM trades"
            ).fetchone()
            fees_total = float(fees_total_row[0]) if fees_total_row else 0.0

            pnl_row = conn.execute(
                "SELECT COALESCE(SUM(net_pnl), 0) FROM trades WHERE status = 'closed'"
            ).fetchone()
            total_pnl_realized = float(pnl_row[0]) if pnl_row else 0.0

            # Min equity ultime 24h (trough) — mostra il dato "mancante" del drawdown
            cutoff_24h = (now_utc - timedelta(hours=24)).isoformat()
            min_eq = conn.execute(
                """SELECT MIN(equity) FROM equity_snapshots
                   WHERE timestamp >= ? AND NOT (equity=10000 AND positions_value=0 AND open_trades=0)
                   AND open_trades > 0""",
                (cutoff_24h,),
            ).fetchone()
            min_equity_24h = float(min_eq[0]) if min_eq and min_eq[0] is not None else None

        except Exception as exc:
            logger.error("DB query error in dashboard/summary: %s", exc, exc_info=True)
        finally:
            conn.close()

    # Final label computation (if not set from DB raw_data)
    if fear_greed_value is not None and fear_greed_label_str == "Unknown":
        fear_greed_label_str = _fear_greed_label(fear_greed_value)

    # Verifica conti: Capitale + Σ(PnL) = Equity (pnl già netto di fees)
    equity_verifica = initial_capital + total_pnl_realized

    return JSONResponse(
        content={
            "equity": round(equity, 4),
            "cash": round(cash, 4),
            "initial_capital": initial_capital,
            "pnl_today_pct": round(pnl_today_pct, 4),
            "pnl_total_pct": round(pnl_total_pct, 4),
            "win_rate": round(win_rate, 4),
            "total_trades": total_closed,
            "open_positions_count": open_positions_count,
            "bot_status": bot_status,
            "fear_greed_value": fear_greed_value,
            "fear_greed_label": fear_greed_label_str,
            "fees_today": round(fees_today, 4),
            "fees_total": round(fees_total, 4),
            "total_pnl_realized": round(total_pnl_realized, 4),
            "verifica": {
                "formula": "Capitale iniziale + Σ(PnL netto) = Equity (solo chiusi)",
                "initial_capital": initial_capital,
                "total_pnl_realized": round(total_pnl_realized, 4),
                "equity_from_closed": round(equity_verifica, 4),
                "equity_actual": round(equity, 4),
                "fees_total": round(fees_total, 4),
                "min_equity_24h": round(min_equity_24h, 4) if min_equity_24h is not None else None,
            },
            "last_updated": now_utc.isoformat(),
        }
    )


# ---------------------------------------------------------------------------
# Bybit ticker helper
# ---------------------------------------------------------------------------

_BYBIT_TICKER_URL = "https://api.bybit.com/v5/market/tickers"
_TRACKED_SYMBOLS: List[str] = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]

# Timeout for Bybit REST calls (seconds)
_BYBIT_TIMEOUT = 5.0


async def _fetch_bybit_ticker(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Fetch the 24-hour ticker for a single spot symbol from Bybit's public API.

    Returns a parsed ticker dict on success, or None on any error.
    """
    try:
        async with httpx.AsyncClient(timeout=_BYBIT_TIMEOUT) as client:
            resp = await client.get(
                _BYBIT_TICKER_URL,
                params={"category": "spot", "symbol": symbol},
            )
            resp.raise_for_status()
            payload = resp.json()

            # Bybit v5 response structure:
            # { "result": { "list": [ { "symbol": "BTCUSDT", "lastPrice": "...", ... } ] } }
            result = payload.get("result", {})
            tickers = result.get("list", [])
            if not tickers:
                return None

            t = tickers[0]
            last_price = float(t.get("lastPrice", 0))
            prev_price = float(t.get("prevPrice24h", last_price) or last_price)
            change_pct = (
                (last_price - prev_price) / prev_price * 100.0 if prev_price else 0.0
            )

            return {
                "symbol": symbol,
                "price": round(last_price, 8),
                "change_24h_pct": round(change_pct, 4),
                "volume_24h": round(float(t.get("volume24h", 0) or 0), 2),
                "high_24h": round(float(t.get("highPrice24h", 0) or 0), 8),
                "low_24h": round(float(t.get("lowPrice24h", 0) or 0), 8),
            }
    except Exception as exc:
        logger.warning("Bybit ticker fetch failed for %s: %s", symbol, exc)
        return None


def _placeholder_tick(symbol: str) -> Dict[str, Any]:
    """Return a zeroed placeholder tick when real data is unavailable."""
    return {
        "symbol": symbol,
        "price": 0.0,
        "change_24h_pct": 0.0,
        "volume_24h": 0.0,
        "high_24h": 0.0,
        "low_24h": 0.0,
    }


# ---------------------------------------------------------------------------
# GET /dashboard/market-feed
# ---------------------------------------------------------------------------


@router.get(
    "/dashboard/market-feed",
    summary="Real-time market data for tracked trading pairs.",
    tags=["dashboard"],
)
async def get_market_feed() -> JSONResponse:
    """
    Return live market tickers for BTCUSDT, ETHUSDT, SOLUSDT, and BNBUSDT.

    Data priority
    -------------
    1. ``engine_instance.current_prices`` (lowest latency – updated on every candle).
    2. Bybit public REST API (no authentication required).
    3. Placeholder zeros (fallback when Bybit is unreachable).

    Response: list of ``MarketTick`` objects.
    """
    engine = _server_module.engine_instance
    ticks: List[Dict[str, Any]] = []

    # ---- 1. Try engine current_prices ----
    engine_prices: Dict[str, float] = {}
    if engine is not None:
        try:
            engine_prices = dict(getattr(engine, "current_prices", {}))
        except Exception:
            pass

    # ---- 2. Fetch from Bybit for each symbol ----
    import asyncio as _asyncio

    async def _get_tick(symbol: str) -> Dict[str, Any]:
        if symbol in engine_prices and engine_prices[symbol] > 0:
            # Engine has a live price; still return a minimal tick.
            # We still need volume/high/low, so fetch from Bybit unless it fails.
            bybit_tick = await _fetch_bybit_ticker(symbol)
            if bybit_tick:
                # Override price with the engine's more recent value
                bybit_tick["price"] = round(engine_prices[symbol], 8)
                return bybit_tick
            return {
                "symbol": symbol,
                "price": round(engine_prices[symbol], 8),
                "change_24h_pct": 0.0,
                "volume_24h": 0.0,
                "high_24h": engine_prices[symbol],
                "low_24h": engine_prices[symbol],
            }
        bybit_tick = await _fetch_bybit_ticker(symbol)
        return bybit_tick if bybit_tick else _placeholder_tick(symbol)

    # Run all ticker fetches concurrently
    results = await _asyncio.gather(*[_get_tick(sym) for sym in _TRACKED_SYMBOLS])
    ticks = list(results)

    return JSONResponse(content=ticks)


# ---------------------------------------------------------------------------
# GET /dashboard/ohlcv
# ---------------------------------------------------------------------------

# Timeframe label → minutes mapping (for limit calculation)
_TF_MINUTES: Dict[str, int] = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240, "6h": 360, "12h": 720, "1d": 1440,
}

# Dashboard timeframe (1h, 4h, 1d, 15m) → cache key (OHLCVStore uses 5, 15, 60, 240, D)
_TF_TO_STORE: Dict[str, str] = {
    "1h": "60",
    "4h": "240",
    "1d": "D",
    "15m": "15",
    "5m": "5",
}


@router.get(
    "/dashboard/ohlcv",
    summary="OHLCV price candles for a symbol and timeframe from local cache.",
    tags=["dashboard"],
)
async def get_ohlcv_chart(
    symbol: str = Query(..., description="Trading pair, e.g. BTCUSDT"),
    timeframe: str = Query(default="1h", description="Candle timeframe: 1h, 4h, 1d, etc."),
    limit: int = Query(
        default=200,
        ge=10,
        le=2000,
        description="Maximum number of candles to return (most recent).",
    ),
) -> JSONResponse:
    """
    Return OHLCV candles for the requested symbol and timeframe from the
    local Parquet cache.  Used by the per-crypto price chart in the dashboard.

    Response: list of candle objects with fields
        time, open, high, low, close, volume
    where ``time`` is a Unix timestamp in **seconds** (for LightweightCharts).
    """
    engine = _server_module.engine_instance
    ohlcv_store = None
    if engine is not None:
        try:
            ohlcv_store = getattr(engine, "ohlcv_store", None)
        except Exception:
            pass

    candles: List[Dict[str, Any]] = []

    # Map dashboard timeframe (1h, 4h, 1d, 15m) to cache key (60, 240, D, 15)
    store_tf = _TF_TO_STORE.get(timeframe, timeframe)

    if ohlcv_store is not None:
        try:
            import pandas as pd
            df = ohlcv_store.load(
                symbol=symbol.upper(),
                timeframe=store_tf,
                limit=limit,
            )
            if not df.empty:
                for _, row in df.tail(limit).iterrows():
                    ts = int(row.get("open_time", 0))
                    # LightweightCharts expects seconds
                    if ts > 1e12:
                        ts = ts // 1000
                    candles.append({
                        "time": ts,
                        "open": round(float(row.get("open", 0)), 6),
                        "high": round(float(row.get("high", 0)), 6),
                        "low": round(float(row.get("low", 0)), 6),
                        "close": round(float(row.get("close", 0)), 6),
                        "volume": round(float(row.get("volume", 0)), 2),
                    })
        except Exception as exc:
            logger.warning("OHLCV chart error for %s/%s: %s", symbol, timeframe, exc)

    # Fallback: try CoinGecko public API for recent data when cache is empty
    if not candles:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # CoinGecko symbol map
                cg_ids = {
                    "BTCUSDT": "bitcoin", "ETHUSDT": "ethereum",
                    "SOLUSDT": "solana", "BNBUSDT": "binancecoin",
                }
                cg_id = cg_ids.get(symbol.upper(), "")
                if cg_id:
                    tf_days = {"1h": 7, "4h": 30, "1d": 90}.get(timeframe, 7)
                    resp = await client.get(
                        f"https://api.coingecko.com/api/v3/coins/{cg_id}/ohlc",
                        params={"vs_currency": "usd", "days": str(tf_days)},
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        # CoinGecko OHLC: [[timestamp_ms, open, high, low, close], ...]
                        for item in data[-limit:]:
                            ts_s = item[0] // 1000
                            candles.append({
                                "time": ts_s,
                                "open": item[1], "high": item[2],
                                "low": item[3], "close": item[4],
                                "volume": 0.0,
                            })
        except Exception as exc:
            logger.debug("CoinGecko OHLCV fallback failed: %s", exc)

    return JSONResponse(content=candles)
