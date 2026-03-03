"""
src/api/routes/trades.py
=========================
Trade history endpoints.

GET /api/v1/trades
    Paginated list of trades with optional symbol and status filters.

GET /api/v1/trades/{trade_id}
    Full detail for a single trade, including the parsed signal_breakdown JSON.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Path, Query
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)
router = APIRouter()

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


def _trade_row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    """
    Convert a trade DB row to a serialisable dict.

    Handles:
    - Null-safety for all optional numeric fields.
    - JSON-parsing of ``signal_breakdown``.
    - Replacement of Python ``float('inf')`` / ``float('nan')`` with None.
    """
    d: Dict[str, Any] = {}
    for key in row.keys():
        val = row[key]
        if key == "signal_breakdown" and val is not None:
            try:
                val = json.loads(val)
            except (json.JSONDecodeError, TypeError):
                val = None
        elif isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            val = None
        d[key] = val
    return d


# ---------------------------------------------------------------------------
# GET /trades
# ---------------------------------------------------------------------------


@router.get(
    "/trades",
    summary="Paginated trade history with optional filters.",
    tags=["trades"],
)
async def list_trades(
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)."),
    limit: int = Query(
        default=50,
        ge=1,
        le=200,
        description="Number of records per page (max 200).",
    ),
    symbol: Optional[str] = Query(
        default=None,
        description="Filter by trading pair, e.g. 'BTCUSDT'.",
    ),
    status: Optional[str] = Query(
        default=None,
        description="Filter by trade status: 'open', 'closed', 'cancelled', or 'error'.",
    ),
) -> JSONResponse:
    """
    Return a paginated list of trade records ordered by entry time (newest first).

    Response schema
    ---------------
    ``trades``  : list of trade dicts.
    ``total``   : total matching records.
    ``page``    : current page.
    ``pages``   : total number of pages.
    """
    conn = _safe_connect()
    if conn is None:
        return JSONResponse(
            content={"trades": [], "total": 0, "page": page, "pages": 0}
        )

    try:
        # ---- Build WHERE clause ----
        conditions: List[str] = []
        params: List[Any] = []

        if symbol is not None:
            conditions.append("symbol = ?")
            params.append(symbol.upper())

        if status is not None:
            allowed_statuses = {"open", "closed", "cancelled", "error"}
            if status.lower() not in allowed_statuses:
                raise HTTPException(
                    status_code=422,
                    detail=f"Invalid status '{status}'. Allowed: {sorted(allowed_statuses)}",
                )
            conditions.append("status = ?")
            params.append(status.lower())

        where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        # ---- Count total matching rows ----
        count_row = conn.execute(
            f"SELECT COUNT(*) AS cnt FROM trades {where_clause}", params
        ).fetchone()
        total: int = int(count_row["cnt"]) if count_row else 0

        # ---- Paginate ----
        pages: int = math.ceil(total / limit) if total > 0 else 0
        offset: int = (page - 1) * limit

        rows = conn.execute(
            f"""
            SELECT
                trade_id, symbol, direction, status,
                entry_time, entry_price, entry_slippage, entry_fee,
                quantity, notional_value,
                kelly_fraction, risk_amount,
                exit_time, exit_price, exit_slippage, exit_fee, exit_reason,
                gross_pnl, net_pnl, pnl_pct, duration_minutes,
                stop_loss_price, take_profit_price, atr_at_entry,
                confidence_score, signal_breakdown, leverage,
                created_at, updated_at
            FROM trades
            {where_clause}
            ORDER BY entry_time DESC
            LIMIT ? OFFSET ?
            """,
            [*params, limit, offset],
        ).fetchall()

        trades = [_trade_row_to_dict(r) for r in rows]

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("trades list DB error: %s", exc, exc_info=True)
        return JSONResponse(
            content={"trades": [], "total": 0, "page": page, "pages": 0}
        )
    finally:
        conn.close()

    return JSONResponse(
        content={
            "trades": trades,
            "total": total,
            "page": page,
            "pages": pages,
        }
    )


# ---------------------------------------------------------------------------
# GET /trades/{trade_id}
# ---------------------------------------------------------------------------


@router.get(
    "/trades/{trade_id}",
    summary="Full details for a single trade including signal breakdown.",
    tags=["trades"],
)
async def get_trade(
    trade_id: str = Path(
        ...,
        description="The UUID trade_id of the trade to retrieve.",
    ),
) -> JSONResponse:
    """
    Return the complete record for a single trade, identified by its UUID.

    The ``signal_breakdown`` field is returned as a parsed JSON object
    (not a raw string) for convenient frontend rendering.

    Raises 404 if no trade with the given ``trade_id`` exists.
    """
    conn = _safe_connect()
    if conn is None:
        raise HTTPException(status_code=503, detail="Database unavailable.")

    try:
        row = conn.execute(
            """
            SELECT
                trade_id, symbol, direction, status,
                entry_time, entry_price, entry_slippage, entry_fee,
                quantity, notional_value,
                kelly_fraction, risk_amount,
                exit_time, exit_price, exit_slippage, exit_fee, exit_reason,
                gross_pnl, net_pnl, pnl_pct, duration_minutes,
                stop_loss_price, take_profit_price, atr_at_entry,
                confidence_score, signal_breakdown,
                created_at, updated_at
            FROM trades
            WHERE trade_id = ?
            """,
            (trade_id,),
        ).fetchone()

        if row is None:
            raise HTTPException(
                status_code=404,
                detail=f"Trade '{trade_id}' not found.",
            )

        trade_dict = _trade_row_to_dict(row)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("get_trade DB error for trade_id=%s: %s", trade_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error fetching trade.")
    finally:
        conn.close()

    return JSONResponse(content=trade_dict)
