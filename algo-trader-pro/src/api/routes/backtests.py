"""
src/api/routes/backtests.py
============================
Backtesting management endpoints.

GET  /api/v1/backtests              – List all completed backtest runs.
POST /api/v1/backtests/run          – Queue and start a new backtest job.
GET  /api/v1/backtests/{run_id}     – Full result for a completed run.
GET  /api/v1/backtests/{run_id}/status – Live progress for a running job.

Architecture
------------
Backtests are CPU/IO-intensive and can take several minutes.  To avoid
blocking the FastAPI event loop, each run is submitted as a FastAPI
``BackgroundTask`` which schedules the coroutine on the same event loop
but allows the HTTP response to be returned immediately.

``_running_backtests`` is an in-memory dict that tracks job progress.
On completion the full result is persisted to the ``backtest_results``
DB table.  If the server restarts mid-run, the in-memory state is lost
but completed runs remain available via DB queries.

Data flow
---------
1. POST /backtests/run  →  generates run_id, adds to _running_backtests,
                           schedules _run_backtest_task in background.
2. _run_backtest_task   →  fetches historical OHLCV from CoinGecko,
                           runs BacktestEngine.run(), optionally runs
                           MonteCarloSimulator, persists to DB, updates
                           _running_backtests[run_id].
3. GET /backtests/{id}  →  returns in-memory result if still running,
                           else queries DB.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()

# ---------------------------------------------------------------------------
# In-memory backtest job tracker
# ---------------------------------------------------------------------------

# Maps run_id (str) -> status dict with keys: status, progress, result, error
_running_backtests: Dict[str, Dict[str, Any]] = {}

# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------


class BacktestRequest(BaseModel):
    """Request body for POST /backtests/run."""

    symbol: str = "bitcoin"
    """CoinGecko coin ID (e.g. 'bitcoin', 'ethereum', 'solana')."""

    timeframe: str = "1h"
    """Candle interval: '1h', '4h', or '1d'."""

    start_date: str = "2023-01-01"
    """Start date in YYYY-MM-DD format."""

    end_date: str = "2024-01-01"
    """End date in YYYY-MM-DD format."""

    initial_capital: float = 10_000.0
    """Starting capital in USDT (must be positive)."""


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


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


def _open_db_rw() -> sqlite3.Connection:
    """Open the database in read-write mode (needed for INSERT)."""
    db_path = _get_db_path()
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _safe_connect_ro() -> Optional[sqlite3.Connection]:
    """Open the database read-only, or return None on failure."""
    try:
        db_path = _get_db_path()
        conn = sqlite3.connect(
            f"file:{db_path}?mode=ro", uri=True, check_same_thread=False
        )
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as exc:
        logger.warning("Cannot open DB (read-only): %s", exc)
        return None


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    d: Dict[str, Any] = {}
    for key in row.keys():
        val = row[key]
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            val = None
        if key in ("monte_carlo_data", "symbols", "config_snapshot") and val is not None:
            try:
                val = json.loads(val)
            except (json.JSONDecodeError, TypeError):
                pass
        d[key] = val
    return d


# ---------------------------------------------------------------------------
# Background task
# ---------------------------------------------------------------------------


async def _run_backtest_task(run_id: str, request: BacktestRequest) -> None:
    """
    Execute a full backtest pipeline and persist the result to the database.

    Steps
    -----
    1. Mark the job as ``running`` in ``_running_backtests``.
    2. Load historical OHLCV data via the CoinGecko data module.
    3. Build engine config from defaults + request parameters.
    4. Run ``BacktestEngine.run()``.
    5. Run ``MonteCarloSimulator.run()`` (optional, on the trade list).
    6. Persist the result to ``backtest_results`` table.
    7. Mark the job as ``complete`` (or ``failed`` on exception).

    Progress updates (0→100 %) are written to ``_running_backtests[run_id]``
    so the frontend polling ``/backtests/{run_id}/status`` can show a
    progress bar.
    """
    _running_backtests[run_id] = {"status": "running", "progress": 0, "result": None, "error": None}
    logger.info("Backtest job %s started: symbol=%s %s→%s", run_id, request.symbol, request.start_date, request.end_date)

    try:
        # ---- Step 1: Import heavy modules inside the task to keep startup fast ----
        import asyncio as _asyncio
        import pandas as pd

        from src.backtesting.backtest_engine import BacktestEngine
        from src.backtesting.monte_carlo import MonteCarloSimulator

        _running_backtests[run_id]["progress"] = 5

        # ---- Step 2: Load OHLCV data ----
        # Try to use the project's data module; fall back to CoinGecko direct fetch.
        ohlcv_df: Optional[pd.DataFrame] = None

        try:
            from src.data.coingecko_client import CoinGeckoClient  # type: ignore[import]
            client = CoinGeckoClient()

            # Map common coin IDs to symbols for the DB / backtest engine
            _SYMBOL_MAP: Dict[str, str] = {
                "bitcoin": "BTCUSDT",
                "ethereum": "ETHUSDT",
                "solana": "SOLUSDT",
                "binancecoin": "BNBUSDT",
            }
            symbol_upper = _SYMBOL_MAP.get(request.symbol.lower(), request.symbol.upper())

            # Attempt to call fetch_ohlcv (or similar) on the client.
            if hasattr(client, "fetch_ohlcv"):
                ohlcv_df = await client.fetch_ohlcv(  # type: ignore[attr-defined]
                    coin_id=request.symbol,
                    timeframe=request.timeframe,
                    start_date=request.start_date,
                    end_date=request.end_date,
                )
            elif hasattr(client, "get_ohlcv"):
                ohlcv_df = client.get_ohlcv(  # type: ignore[attr-defined]
                    coin_id=request.symbol,
                    timeframe=request.timeframe,
                    start_date=request.start_date,
                    end_date=request.end_date,
                )
        except Exception as data_exc:
            logger.warning(
                "CoinGecko client failed (%s); attempting direct HTTP fetch.", data_exc
            )

        # If the client didn't provide data, build a minimal DataFrame via HTTP.
        if ohlcv_df is None or ohlcv_df.empty:
            ohlcv_df = await _fetch_coingecko_ohlcv_direct(request)

        if ohlcv_df is None or ohlcv_df.empty:
            raise ValueError(
                f"No OHLCV data available for symbol='{request.symbol}' "
                f"between {request.start_date} and {request.end_date}."
            )

        _running_backtests[run_id]["progress"] = 25
        logger.info(
            "Backtest %s: loaded %d candles for %s.", run_id, len(ohlcv_df), request.symbol
        )

        # ---- Step 3: Build config ----
        # Use a minimal config that mirrors the project's default config structure
        # so BacktestEngine can consume it.
        config = _build_backtest_config(request)

        _running_backtests[run_id]["progress"] = 30

        # ---- Step 4: Run BacktestEngine ----
        # BacktestEngine expects (config, strategy, risk_manager) but in the
        # backtest context it creates internal instances if strategy/risk_manager
        # are not passed.  We try both signatures.
        try:
            engine = BacktestEngine(
                config=config,
                initial_capital=request.initial_capital,
            )
        except TypeError:
            # Fallback: some versions require positional strategy/risk args.
            engine = BacktestEngine(config=config)  # type: ignore[call-arg]

        result = await engine.run(
            df=ohlcv_df,
            symbol=request.symbol,
            start_date=request.start_date,
            end_date=request.end_date,
        )

        _running_backtests[run_id]["progress"] = 75
        logger.info(
            "Backtest %s engine run complete. return=%.2f%%",
            run_id,
            result.total_return_pct if hasattr(result, "total_return_pct") else 0,
        )

        # ---- Step 5: Monte Carlo simulation ----
        mc_data: Optional[Dict[str, Any]] = None
        try:
            trades_for_mc = result.trades if hasattr(result, "trades") else []
            if len(trades_for_mc) >= 10:
                mc = MonteCarloSimulator(n_simulations=500)  # Reduced for speed
                mc_result = mc.run(
                    trades=trades_for_mc,
                    initial_capital=request.initial_capital,
                )
                mc_data = mc_result.to_dict() if hasattr(mc_result, "to_dict") else None
                _running_backtests[run_id]["progress"] = 90
        except Exception as mc_exc:
            logger.warning("Monte Carlo failed (non-fatal): %s", mc_exc)

        # ---- Step 6: Persist to DB ----
        result_dict = result.to_dict() if hasattr(result, "to_dict") else {}
        _persist_backtest_result(run_id, request, result_dict, mc_data)

        _running_backtests[run_id]["progress"] = 100
        _running_backtests[run_id]["status"] = "complete"
        _running_backtests[run_id]["result"] = result_dict
        logger.info("Backtest job %s completed successfully.", run_id)

    except Exception as exc:
        logger.error("Backtest job %s FAILED: %s", run_id, exc, exc_info=True)
        _running_backtests[run_id]["status"] = "failed"
        _running_backtests[run_id]["error"] = str(exc)
        _running_backtests[run_id]["progress"] = 0


async def _fetch_coingecko_ohlcv_direct(request: BacktestRequest):
    """
    Fetch historical OHLCV from CoinGecko public API and return a DataFrame.

    CoinGecko's free /coins/{id}/market_chart endpoint returns OHLCV as
    lists of [timestamp_ms, open, high, low, close].  We use the 'ohlc'
    endpoint for daily data and parse market_chart for intra-day.
    """
    import httpx
    import pandas as pd

    # CoinGecko 'ohlc' endpoint supports 1, 7, 14, 30, 90, 180, 365 days.
    # We calculate the total days in the requested range.
    try:
        start_dt = datetime.strptime(request.start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(request.end_date, "%Y-%m-%d")
        days = (end_dt - start_dt).days
        if days <= 0:
            raise ValueError("end_date must be after start_date")
        # CoinGecko max for free tier: 365 days
        days = min(days, 365)
    except ValueError as exc:
        raise ValueError(f"Invalid date range: {exc}") from exc

    url = f"https://api.coingecko.com/api/v3/coins/{request.symbol}/ohlc"
    params = {"vs_currency": "usd", "days": str(days)}

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            raw = resp.json()
    except Exception as exc:
        raise ValueError(
            f"CoinGecko OHLCV fetch failed for '{request.symbol}': {exc}"
        ) from exc

    if not raw:
        raise ValueError(f"CoinGecko returned empty data for '{request.symbol}'.")

    # raw is a list of [timestamp_ms, open, high, low, close]
    df = pd.DataFrame(raw, columns=["open_time", "open", "high", "low", "close"])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["volume"] = 0.0  # CoinGecko OHLC endpoint doesn't return volume
    df = df.sort_values("open_time").reset_index(drop=True)

    return df


def _build_backtest_config(request: BacktestRequest) -> Any:
    """
    Build a minimal config object that BacktestEngine can consume.

    Tries to load the real project config first; falls back to a plain dict
    that exposes values via attribute access (using a SimpleNamespace-like
    object).
    """
    import types

    # Try to get the real config
    try:
        from src.utils.config_loader import get_config  # type: ignore[import]
        cfg = get_config()
        # Patch the backtest-specific fields onto a copy
        if hasattr(cfg, "backtesting") and hasattr(cfg.backtesting, "initial_capital"):
            cfg.backtesting.initial_capital = request.initial_capital  # type: ignore[attr-defined]
        return cfg
    except Exception:
        pass

    # Fallback: build a nested SimpleNamespace config
    def _ns(**kw):
        """Recursively convert a dict to a SimpleNamespace."""
        ns = types.SimpleNamespace()
        for k, v in kw.items():
            if isinstance(v, dict):
                setattr(ns, k, _ns(**v))
            else:
                setattr(ns, k, v)
        return ns

    return _ns(
        app=_ns(environment="paper", log_level="INFO"),
        trading=_ns(
            symbols=["BTCUSDT"],
            timeframe=request.timeframe,
            initial_capital=request.initial_capital,
            max_open_positions=3,
            confidence_threshold=60.0,
            allow_shorts=False,
        ),
        risk=_ns(
            max_position_size_pct=5.0,
            stop_loss_atr_multiplier=2.0,
            take_profit_atr_multiplier=4.0,
            max_daily_loss_pct=3.0,
            max_drawdown_pct=15.0,
            kelly_fraction=0.25,
        ),
        backtesting=_ns(
            initial_capital=request.initial_capital,
            slippage_pct=0.05,
            fee_pct=0.1,
        ),
        ml=_ns(enabled=False, model_type="xgboost"),
        signals=_ns(
            layer1_weight=0.4,
            layer2_weight=0.2,
            layer3_weight=0.2,
            ml_weight=0.2,
        ),
    )


def _persist_backtest_result(
    run_id: str,
    request: BacktestRequest,
    result_dict: Dict[str, Any],
    mc_data: Optional[Dict[str, Any]],
) -> None:
    """Insert the backtest result into the ``backtest_results`` table."""
    try:
        conn = _open_db_rw()
        config_snapshot = json.dumps(
            {
                "symbol": request.symbol,
                "timeframe": request.timeframe,
                "start_date": request.start_date,
                "end_date": request.end_date,
                "initial_capital": request.initial_capital,
            }
        )
        symbols_json = json.dumps([request.symbol])
        mc_json = json.dumps(mc_data) if mc_data else None

        conn.execute(
            """
            INSERT OR REPLACE INTO backtest_results (
                run_id, config_snapshot, start_date, end_date,
                initial_capital, final_equity,
                total_return_pct, sharpe_ratio, sortino_ratio,
                max_drawdown_pct, calmar_ratio, profit_factor,
                win_rate, total_trades, winning_trades, losing_trades,
                avg_win_pct, avg_loss_pct, avg_hold_hours,
                monte_carlo_data, symbols, created_at
            ) VALUES (
                ?, ?, ?, ?,
                ?, ?,
                ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?
            )
            """,
            (
                run_id,
                config_snapshot,
                request.start_date,
                request.end_date,
                request.initial_capital,
                float(result_dict.get("final_equity", request.initial_capital)),
                float(result_dict.get("total_return_pct", 0.0)),
                _safe_float(result_dict.get("sharpe_ratio")),
                _safe_float(result_dict.get("sortino_ratio")),
                float(result_dict.get("max_drawdown_pct", 0.0)),
                _safe_float(result_dict.get("calmar_ratio")),
                _safe_float(result_dict.get("profit_factor")),
                _safe_float(result_dict.get("win_rate")),
                int(result_dict.get("total_trades", 0)),
                int(result_dict.get("winning_trades", 0)),
                int(result_dict.get("losing_trades", 0)),
                _safe_float(result_dict.get("avg_win_pct")),
                _safe_float(result_dict.get("avg_loss_pct")),
                _safe_float(result_dict.get("avg_hold_hours")),
                mc_json,
                symbols_json,
                datetime.now(tz=timezone.utc).isoformat(),
            ),
        )
        conn.commit()
        conn.close()
        logger.info("Backtest result %s persisted to DB.", run_id)
    except Exception as exc:
        logger.error("Failed to persist backtest %s: %s", run_id, exc, exc_info=True)


def _safe_float(val: Any) -> Optional[float]:
    """Convert val to float, returning None for inf/nan/None."""
    if val is None:
        return None
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# GET /backtests
# ---------------------------------------------------------------------------


@router.get(
    "/backtests",
    summary="List all completed backtest runs.",
    tags=["backtests"],
)
async def list_backtests() -> JSONResponse:
    """
    Return a summary list of all persisted backtest runs, ordered by
    creation time (newest first).

    Response fields per entry
    -------------------------
    run_id, start_date, end_date, initial_capital, final_equity,
    total_return_pct, sharpe_ratio, max_drawdown_pct, total_trades,
    win_rate, created_at.
    """
    conn = _safe_connect_ro()
    if conn is None:
        return JSONResponse(content=[])

    try:
        rows = conn.execute(
            """
            SELECT
                run_id, start_date, end_date, initial_capital, final_equity,
                total_return_pct, sharpe_ratio, max_drawdown_pct, total_trades,
                win_rate, created_at
            FROM backtest_results
            ORDER BY created_at DESC
            LIMIT 100
            """
        ).fetchall()
        results = [_row_to_dict(r) for r in rows]
    except Exception as exc:
        logger.error("list_backtests DB error: %s", exc, exc_info=True)
        results = []
    finally:
        conn.close()

    return JSONResponse(content=results)


# ---------------------------------------------------------------------------
# POST /backtests/run
# ---------------------------------------------------------------------------


@router.post(
    "/backtests/run",
    summary="Queue a new backtest run (returns immediately).",
    tags=["backtests"],
)
async def run_backtest(
    request: BacktestRequest,
    background_tasks: BackgroundTasks,
) -> JSONResponse:
    """
    Start a new backtest asynchronously.

    The endpoint returns immediately with a ``run_id`` that the client
    can use to poll ``GET /backtests/{run_id}/status`` for progress
    updates, and ``GET /backtests/{run_id}`` for the full result.

    Request body (all optional, defaults shown)
    --------------------------------------------
    symbol         : "bitcoin"      – CoinGecko coin ID.
    timeframe      : "1h"           – Candle interval.
    start_date     : "2023-01-01"   – Backtest start (YYYY-MM-DD).
    end_date       : "2024-01-01"   – Backtest end (YYYY-MM-DD).
    initial_capital: 10000.0        – Starting equity in USDT.
    """
    # Validate date format
    for date_str, field_name in [(request.start_date, "start_date"), (request.end_date, "end_date")]:
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid {field_name} format: '{date_str}'. Expected YYYY-MM-DD.",
            )

    if datetime.strptime(request.end_date, "%Y-%m-%d") <= datetime.strptime(request.start_date, "%Y-%m-%d"):
        raise HTTPException(
            status_code=422,
            detail="end_date must be strictly after start_date.",
        )

    if request.initial_capital <= 0:
        raise HTTPException(
            status_code=422,
            detail="initial_capital must be positive.",
        )

    run_id = str(uuid.uuid4())
    _running_backtests[run_id] = {
        "status": "pending",
        "progress": 0,
        "result": None,
        "error": None,
    }

    # Schedule the backtest as a background task so we can return immediately.
    background_tasks.add_task(_run_backtest_task, run_id, request)

    logger.info(
        "Backtest queued: run_id=%s symbol=%s %s→%s capital=%.0f",
        run_id,
        request.symbol,
        request.start_date,
        request.end_date,
        request.initial_capital,
    )

    return JSONResponse(
        status_code=202,
        content={
            "run_id": run_id,
            "status": "started",
            "message": (
                f"Backtest queued with run_id={run_id}. "
                f"Poll GET /api/v1/backtests/{run_id}/status for progress."
            ),
        },
    )


# ---------------------------------------------------------------------------
# GET /backtests/{run_id}
# ---------------------------------------------------------------------------


@router.get(
    "/backtests/{run_id}",
    summary="Full result for a backtest run (polls in-memory then DB).",
    tags=["backtests"],
)
async def get_backtest(run_id: str) -> JSONResponse:
    """
    Return the full result of a backtest run.

    Lookup order
    ------------
    1. Check ``_running_backtests`` (in-memory) – if the job is still in
       progress, return the current status rather than blocking.
    2. Query the ``backtest_results`` DB table for persisted results.

    Returns 404 if the run_id is not found in either location.
    """
    # ---- 1. In-memory check ----
    if run_id in _running_backtests:
        job = _running_backtests[run_id]
        if job["status"] in ("pending", "running"):
            return JSONResponse(
                content={
                    "run_id": run_id,
                    "status": job["status"],
                    "progress_pct": job.get("progress", 0),
                    "message": "Backtest is still in progress.",
                }
            )
        if job["status"] == "failed":
            return JSONResponse(
                status_code=500,
                content={
                    "run_id": run_id,
                    "status": "failed",
                    "error": job.get("error", "Unknown error"),
                },
            )
        # Completed in-memory result
        if job["status"] == "complete" and job.get("result"):
            return JSONResponse(
                content={
                    "run_id": run_id,
                    "status": "complete",
                    **job["result"],
                }
            )

    # ---- 2. DB lookup ----
    conn = _safe_connect_ro()
    if conn is None:
        raise HTTPException(status_code=503, detail="Database unavailable.")

    try:
        row = conn.execute(
            "SELECT * FROM backtest_results WHERE run_id = ?", (run_id,)
        ).fetchone()

        if row is None:
            raise HTTPException(
                status_code=404,
                detail=f"Backtest run '{run_id}' not found.",
            )

        result = _row_to_dict(row)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("get_backtest DB error run_id=%s: %s", run_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error.")
    finally:
        conn.close()

    return JSONResponse(content=result)


# ---------------------------------------------------------------------------
# GET /backtests/{run_id}/status
# ---------------------------------------------------------------------------


@router.get(
    "/backtests/{run_id}/status",
    summary="Progress status for a running or completed backtest job.",
    tags=["backtests"],
)
async def get_backtest_status(run_id: str) -> JSONResponse:
    """
    Return the current progress of a backtest job.

    Intended for frontend polling (e.g. every 2 seconds) while the job
    is in ``running`` state.

    Response fields
    ---------------
    run_id        : str   – The backtest job ID.
    status        : str   – 'pending', 'running', 'complete', or 'failed'.
    progress_pct  : float – Estimated completion [0-100].
    error         : str | null – Error message if status == 'failed'.
    """
    # ---- In-memory first ----
    if run_id in _running_backtests:
        job = _running_backtests[run_id]
        return JSONResponse(
            content={
                "run_id": run_id,
                "status": job.get("status", "unknown"),
                "progress_pct": job.get("progress", 0),
                "error": job.get("error"),
            }
        )

    # ---- Check DB for completed runs ----
    conn = _safe_connect_ro()
    if conn is None:
        raise HTTPException(status_code=503, detail="Database unavailable.")

    try:
        row = conn.execute(
            "SELECT run_id, created_at FROM backtest_results WHERE run_id = ?",
            (run_id,),
        ).fetchone()
    except Exception as exc:
        logger.error("get_backtest_status DB error: %s", exc)
        row = None
    finally:
        conn.close()

    if row is not None:
        return JSONResponse(
            content={
                "run_id": run_id,
                "status": "complete",
                "progress_pct": 100.0,
                "error": None,
            }
        )

    raise HTTPException(
        status_code=404,
        detail=f"Backtest run '{run_id}' not found.",
    )
