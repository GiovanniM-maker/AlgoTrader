"""
src/api/routes/portfolio.py
============================
Portfolio-related API endpoints.

GET /api/v1/portfolio/equity-curve
    Returns the time-series of portfolio equity snapshots for chart rendering.

GET /api/v1/portfolio/positions
    Returns all currently open positions enriched with live pricing and
    unrealised P&L.

GET /api/v1/portfolio/metrics
    Computes and returns the full suite of risk-adjusted performance metrics
    (Sharpe, Sortino, Max Drawdown, Calmar, Profit Factor, Win Rate, etc.)
    from the equity curve and closed trade history.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import httpx
import pandas as pd
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from src.portfolio.metrics import compute_all_metrics

import src.api.server as _server_module  # late import – avoids circular dep

logger = logging.getLogger(__name__)
router = APIRouter()

# ---------------------------------------------------------------------------
# DB helper
# ---------------------------------------------------------------------------

_BYBIT_TICKER_URL = "https://api.bybit.com/v5/market/tickers"
_BYBIT_TIMEOUT = 5.0


def _get_db_path() -> str:
    env_url = os.getenv("DATABASE_URL", "")
    if env_url.startswith("sqlite:///"):
        return env_url.replace("sqlite:///", "", 1)
    if env_url.startswith("sqlite://./"):
        return env_url.replace("sqlite://./", "", 1)
    base = os.path.dirname(__file__)
    return os.path.normpath(
        os.path.join(base, "..", "..", "..", "database", "algotrader.db")
    )


def _safe_connect() -> Optional[sqlite3.Connection]:
    try:
        db_path = _get_db_path()
        conn = sqlite3.connect(
            f"file:{db_path}?mode=ro", uri=True, check_same_thread=False
        )
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as exc:
        logger.warning("Cannot open DB: %s", exc)
        return None


async def _fetch_current_price(symbol: str) -> Optional[float]:
    """Fetch the last traded price for *symbol* from Bybit public REST."""
    try:
        async with httpx.AsyncClient(timeout=_BYBIT_TIMEOUT) as client:
            resp = await client.get(
                _BYBIT_TICKER_URL,
                params={"category": "spot", "symbol": symbol},
            )
            resp.raise_for_status()
            data = resp.json()
            tickers = data.get("result", {}).get("list", [])
            if tickers:
                return float(tickers[0].get("lastPrice", 0) or 0)
    except Exception as exc:
        logger.debug("Price fetch failed for %s: %s", symbol, exc)
    return None


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    """Convert a sqlite3.Row to a plain dict."""
    return dict(zip(row.keys(), tuple(row)))


# ---------------------------------------------------------------------------
# GET /portfolio/equity-curve
# ---------------------------------------------------------------------------


@router.get(
    "/portfolio/equity-curve",
    summary="Equity curve time-series for the portfolio chart.",
    tags=["portfolio"],
)
async def get_equity_curve(
    hours: int = Query(
        default=24,
        ge=1,
        le=8760,
        description="Number of hours of history to return (1 – 8 760).",
    ),
) -> JSONResponse:
    """
    Return the portfolio equity curve for the requested time window.

    Each point in the series contains:
    - ``timestamp`` : ISO-8601 UTC string.
    - ``equity``    : Total portfolio value in USDT.
    - ``drawdown_pct`` : Current drawdown from the peak (positive %).
    - ``cash``      : Available uninvested cash in USDT.

    If no snapshots exist yet (first launch), a single synthetic data point
    at the current time with the default starting capital (10 000 USDT) is
    returned so the chart always has something to render.
    """
    cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=hours)
    cutoff_iso = cutoff.isoformat()

    conn = _safe_connect()
    points: List[Dict[str, Any]] = []

    if conn is not None:
        try:
            # Escludi snapshot bogus (10000/0/0) che inquinano la curva
            rows = conn.execute(
                """
                SELECT timestamp, equity, drawdown_pct, cash
                FROM equity_snapshots
                WHERE timestamp >= ?
                  AND NOT (equity = 10000 AND positions_value = 0 AND open_trades = 0)
                ORDER BY timestamp ASC
                """,
                (cutoff_iso,),
            ).fetchall()

            for row in rows:
                points.append(
                    {
                        "timestamp": row["timestamp"],
                        "equity": round(float(row["equity"]), 4),
                        "drawdown_pct": round(float(row["drawdown_pct"]), 4),
                        "cash": round(float(row["cash"]), 4) if row["cash"] is not None else None,
                    }
                )

            # Downsample: snapshot ogni 10s → troppi punti. Prendiamo 1 ogni N per curva leggibile.
            # 24h = 864 punti (10s) → target ~150. 7d → ~200. 30d → ~300.
            max_points = 200 if hours <= 168 else 400
            if len(points) > max_points:
                step = max(1, len(points) // max_points)
                points = [points[i] for i in range(0, len(points), step)]
        except Exception as exc:
            logger.error("equity-curve DB error: %s", exc, exc_info=True)
        finally:
            conn.close()

    if not points:
        # Return 2 points (start + end of window) so LightweightCharts draws a visible line.
        now_utc = datetime.now(tz=timezone.utc)
        start_utc = now_utc - timedelta(hours=hours)
        points = [
            {
                "timestamp": start_utc.isoformat().replace("+00:00", "Z"),
                "equity": 10_000.0,
                "drawdown_pct": 0.0,
                "cash": 10_000.0,
            },
            {
                "timestamp": now_utc.isoformat().replace("+00:00", "Z"),
                "equity": 10_000.0,
                "drawdown_pct": 0.0,
                "cash": 10_000.0,
            },
        ]
    elif len(points) == 1:
        # Single point: duplicate so the chart draws a flat line.
        pts = points[0]
        points = [pts, {**pts, "timestamp": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")}]

    return JSONResponse(content=points)


# ---------------------------------------------------------------------------
# GET /portfolio/positions
# ---------------------------------------------------------------------------


@router.get(
    "/portfolio/positions",
    summary="All currently open trading positions with unrealised P&L.",
    tags=["portfolio"],
)
async def get_open_positions() -> JSONResponse:
    """
    Return all open positions enriched with current market prices.

    For each open position the endpoint computes:
    - ``current_price``        : Latest market price from Bybit REST.
    - ``unrealized_pnl``       : (current_price − entry_price) × quantity × direction_sign.
    - ``unrealized_pnl_pct``   : unrealized_pnl / notional_value × 100.

    If a price cannot be fetched for a symbol, ``current_price`` defaults
    to ``entry_price`` (i.e. zero unrealised P&L).
    """
    import asyncio as _asyncio

    engine = _server_module.engine_instance

    # Build a price cache: prefer engine's live prices, fall back to Bybit REST.
    engine_prices: Dict[str, float] = {}
    if engine is not None:
        try:
            engine_prices = dict(getattr(engine, "current_prices", {}))
        except Exception:
            pass

    conn = _safe_connect()
    positions: List[Dict[str, Any]] = []

    if conn is not None:
        try:
            rows = conn.execute(
                """
                SELECT trade_id, symbol, direction, entry_price, quantity,
                       notional_value, stop_loss_price, take_profit_price,
                       confidence_score, entry_time, signal_breakdown, leverage
                FROM trades
                WHERE status = 'open'
                ORDER BY entry_time DESC
                """
            ).fetchall()

            # Collect distinct symbols we need prices for.
            symbols_needed = {row["symbol"] for row in rows if row["symbol"] not in engine_prices}

            # Fetch missing prices from Bybit concurrently.
            async def _get_price(sym: str) -> tuple:
                p = await _fetch_current_price(sym)
                return sym, p

            if symbols_needed:
                fetched = await _asyncio.gather(*[_get_price(s) for s in symbols_needed])
                for sym, price in fetched:
                    if price is not None:
                        engine_prices[sym] = price

            for row in rows:
                symbol = row["symbol"]
                entry_price = float(row["entry_price"])
                quantity = float(row["quantity"])
                notional_value = float(row["notional_value"])
                direction = row["direction"]
                current_price = engine_prices.get(symbol, entry_price)

                # P&L calculation
                direction_sign = 1.0 if direction == "long" else -1.0
                unrealized_pnl = (current_price - entry_price) * quantity * direction_sign
                unrealized_pnl_pct = (
                    (unrealized_pnl / notional_value * 100.0) if notional_value > 0 else 0.0
                )

                # Parse signal_breakdown JSON if present
                signal_breakdown = None
                try:
                    raw_sb = row["signal_breakdown"]
                    if raw_sb:
                        signal_breakdown = json.loads(raw_sb)
                except Exception:
                    pass

                positions.append(
                    {
                        "trade_id": row["trade_id"],
                        "symbol": symbol,
                        "direction": direction,
                        "entry_price": round(entry_price, 8),
                        "current_price": round(current_price, 8),
                        "quantity": round(quantity, 8),
                        "notional_value": round(notional_value, 4),
                        "unrealized_pnl": round(unrealized_pnl, 4),
                        "unrealized_pnl_pct": round(unrealized_pnl_pct, 4),
                        "stop_loss": round(float(row["stop_loss_price"]), 8),
                        "take_profit": (
                            round(float(row["take_profit_price"]), 8)
                            if row["take_profit_price"] is not None
                            else None
                        ),
                        "confidence_score": round(float(row["confidence_score"]), 2),
                        "entry_time": row["entry_time"],
                        "signal_breakdown": signal_breakdown,
                        "leverage": round(float(row["leverage"]) if "leverage" in row.keys() else 1.0, 1),
                    }
                )
        except Exception as exc:
            logger.error("positions DB error: %s", exc, exc_info=True)
        finally:
            conn.close()

    return JSONResponse(content=positions)


# ---------------------------------------------------------------------------
# GET /portfolio/metrics
# ---------------------------------------------------------------------------


@router.get(
    "/portfolio/metrics",
    summary="Full suite of risk-adjusted performance metrics.",
    tags=["portfolio"],
)
async def get_portfolio_metrics() -> JSONResponse:
    """
    Compute and return the full performance metric suite.

    Metrics returned
    ----------------
    sharpe_ratio, sortino_ratio, max_drawdown_pct, calmar_ratio,
    profit_factor, win_rate, avg_win_pct, avg_loss_pct,
    total_trades, total_return_pct, initial_equity, final_equity.

    Computation source
    ------------------
    - Equity curve: all rows in ``equity_snapshots`` (full history).
    - Trades: all rows in ``trades`` with ``status = 'closed'``.

    If either data source is empty, sensible zeros are returned rather
    than raising an error.
    """
    conn = _safe_connect()
    metrics: Dict[str, Any] = {
        "sharpe_ratio": None,
        "sortino_ratio": None,
        "max_drawdown_pct": 0.0,
        "calmar_ratio": None,
        "profit_factor": None,
        "win_rate": 0.0,
        "avg_win_pct": 0.0,
        "avg_loss_pct": 0.0,
        "total_trades": 0,
        "total_return_pct": 0.0,
        "initial_equity": 10_000.0,
        "final_equity": 10_000.0,
    }

    if conn is None:
        return JSONResponse(content=metrics)

    try:
        # ------ Load equity curve ------
        equity_rows = conn.execute(
            "SELECT equity FROM equity_snapshots ORDER BY timestamp ASC"
        ).fetchall()
        equity_values = [float(r["equity"]) for r in equity_rows]
        equity_series = pd.Series(equity_values, dtype=float)

        # ------ Load closed trades ------
        trade_rows = conn.execute(
            """
            SELECT net_pnl, pnl_pct, status
            FROM trades
            WHERE status = 'closed'
            """
        ).fetchall()
        trades_list: List[Dict[str, Any]] = [_row_to_dict(r) for r in trade_rows]

        if len(equity_series) >= 2 or trades_list:
            # Provide a minimal equity series if only trades are available.
            if len(equity_series) < 2:
                equity_series = pd.Series([10_000.0, 10_000.0], dtype=float)

            metrics = compute_all_metrics(
                equity_series=equity_series,
                trades=trades_list,
                periods_per_year=8_760,
                risk_free_rate=0.02,
            )

    except Exception as exc:
        logger.error("metrics computation error: %s", exc, exc_info=True)
    finally:
        conn.close()

    # Replace Python inf with None for JSON serialisation safety.
    for k, v in metrics.items():
        if v == float("inf") or v == float("-inf"):
            metrics[k] = None

    return JSONResponse(content=metrics)
