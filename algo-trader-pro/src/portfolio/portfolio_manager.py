"""
src/portfolio/portfolio_manager.py

Portfolio Manager.

Bridges the in-memory PaperExecutor with the SQLite database, persisting
all trade lifecycle events, equity snapshots, and signal logs.  Also
provides query methods used by the REST API and dashboard.

Database schema assumed (from database/schema.sql):
- trades
- equity_snapshots
- signals_log
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional


class PortfolioManager:
    """
    Persistence and reporting layer for the paper trading portfolio.

    Parameters
    ----------
    paper_executor : PaperExecutor
        The live executor whose in-memory state is the source of truth
        for open positions and trade history.
    db_path : str
        Path to the SQLite database file.  The directory must already exist
        (created by db.py or the startup script).
    """

    def __init__(
        self,
        paper_executor: Any,
        db_path: str = "./database/algotrader.db",
    ) -> None:
        self.executor = paper_executor
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)

        # Verify the DB file exists or log a warning so callers know to run
        # the schema initialisation script first.
        if not os.path.exists(db_path):
            self.logger.warning(
                "Database file not found at '%s'. "
                "Run schema.sql to initialise the database before recording trades.",
                db_path,
            )

    # ------------------------------------------------------------------
    # Connection helper
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        """
        Return a new SQLite connection with row_factory set to
        sqlite3.Row so query results are accessible as dicts.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    # ------------------------------------------------------------------
    # Trade persistence
    # ------------------------------------------------------------------

    def record_trade_open(self, trade: Any) -> None:
        """
        Insert a newly opened trade into the ``trades`` table.

        Uses INSERT OR IGNORE so duplicate calls (e.g. restart recovery)
        are silently discarded.

        Parameters
        ----------
        trade : Trade
            A PaperExecutor Trade object (or any object/dict with the
            required attributes).
        """
        d = trade.to_dict() if hasattr(trade, "to_dict") else dict(trade)

        sql = """
            INSERT OR IGNORE INTO trades (
                trade_id, symbol, direction, status,
                entry_time, entry_price, entry_slippage, entry_fee,
                quantity, notional_value,
                kelly_fraction, risk_amount,
                stop_loss_price, take_profit_price,
                confidence_score, signal_breakdown, leverage,
                created_at, updated_at
            ) VALUES (
                :trade_id, :symbol, :direction, :status,
                :entry_time, :entry_price, :entry_slippage, :entry_fee,
                :quantity, :notional_value,
                :kelly_fraction, :risk_amount,
                :stop_loss_price, :take_profit_price,
                :confidence_score, :signal_breakdown, :leverage,
                :created_at, :updated_at
            )
        """
        now_iso = datetime.utcnow().isoformat()
        params = {
            "trade_id": d.get("trade_id"),
            "symbol": d.get("symbol"),
            "direction": d.get("direction", "long").lower(),
            "status": "open",
            "entry_time": d.get("entry_time", now_iso),
            "entry_price": d.get("entry_price"),
            "entry_slippage": d.get("entry_slippage", 0.0),
            "entry_fee": d.get("entry_fee", 0.0),
            "quantity": d.get("quantity"),
            "notional_value": d.get("notional_value"),
            "kelly_fraction": d.get("kelly_fraction"),
            "risk_amount": d.get("risk_amount"),
            "stop_loss_price": d.get("stop_loss_price", d.get("stop_loss")),
            "take_profit_price": d.get("take_profit_price", d.get("take_profit")),
            "confidence_score": d.get("confidence_score", 0.0),
            "signal_breakdown": json.dumps(d.get("signal_breakdown", {})),
            "leverage": d.get("leverage", 1.0),
            "created_at": now_iso,
            "updated_at": now_iso,
        }

        try:
            with self._get_conn() as conn:
                conn.execute(sql, params)
            self.logger.debug(
                "record_trade_open: trade_id=%s", d.get("trade_id", "?")[:8]
            )
        except sqlite3.Error as exc:
            self.logger.error("record_trade_open failed: %s", exc, exc_info=True)

    def record_trade_close(self, trade: Any) -> None:
        """
        Update an existing open trade record with exit information.

        Parameters
        ----------
        trade : Trade
            A PaperExecutor Trade object (or dict) with all exit fields
            populated.
        """
        d = trade.to_dict() if hasattr(trade, "to_dict") else dict(trade)

        sql = """
            UPDATE trades SET
                exit_time        = :exit_time,
                exit_price       = :exit_price,
                exit_fee         = :exit_fee,
                exit_slippage    = :exit_slippage,
                exit_reason      = :exit_reason,
                gross_pnl        = :gross_pnl,
                net_pnl          = :net_pnl,
                pnl_pct          = :pnl_pct,
                duration_minutes = :duration_minutes,
                status           = 'closed',
                updated_at       = :updated_at
            WHERE trade_id = :trade_id
        """
        params = {
            "trade_id": d.get("trade_id"),
            "exit_time": d.get("exit_time"),
            "exit_price": d.get("exit_price"),
            "exit_fee": d.get("exit_fee", 0.0),
            "exit_slippage": d.get("exit_slippage", 0.0),
            "exit_reason": (d.get("exit_reason") or "").lower() or None,
            "gross_pnl": d.get("gross_pnl", 0.0),
            "net_pnl": d.get("net_pnl", 0.0),
            "pnl_pct": d.get("pnl_pct", 0.0),
            "duration_minutes": d.get("duration_minutes", 0),
            "updated_at": datetime.utcnow().isoformat(),
        }

        try:
            with self._get_conn() as conn:
                conn.execute(sql, params)
            self.logger.debug(
                "record_trade_close: trade_id=%s pnl=%.4f",
                d.get("trade_id", "?")[:8], d.get("net_pnl", 0.0),
            )
        except sqlite3.Error as exc:
            self.logger.error("record_trade_close failed: %s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # Equity snapshots
    # ------------------------------------------------------------------

    def save_equity_snapshot(self, snapshot: dict) -> None:
        """
        Persist a portfolio equity snapshot.

        Uses INSERT OR REPLACE so repeated snapshots at the same UTC
        second are idempotent (the most recent call wins).

        Parameters
        ----------
        snapshot : dict
            Dict as returned by ``PaperExecutor.get_equity_snapshot()``.
            Expected keys: equity, cash, positions_value, drawdown_pct,
            open_trades.  Timestamp defaults to now if not provided.
        """
        sql = """
            INSERT OR REPLACE INTO equity_snapshots
                (timestamp, equity, cash, positions_value, drawdown_pct, open_trades)
            VALUES
                (:timestamp, :equity, :cash, :positions_value, :drawdown_pct, :open_trades)
        """
        now_utc = datetime.now(tz=timezone.utc)
        ts_default = now_utc.isoformat().replace("+00:00", "Z")
        params = {
            "timestamp": snapshot.get("timestamp", ts_default),
            "equity": snapshot.get("equity", 0.0),
            "cash": snapshot.get("cash", 0.0),
            "positions_value": snapshot.get("positions_value", 0.0),
            "drawdown_pct": snapshot.get("drawdown_pct", 0.0),
            "open_trades": snapshot.get("open_trades", 0),
        }

        # Non salvare snapshot bogus (10000/0/0) se abbiamo trade aperti nel DB
        if (
            params["equity"] == 10_000.0
            and params["positions_value"] == 0.0
            and params["open_trades"] == 0
        ):
            try:
                with self._get_conn() as conn:
                    open_cnt = conn.execute(
                        "SELECT COUNT(*) FROM trades WHERE status = 'open'"
                    ).fetchone()[0]
                    if open_cnt > 0:
                        self.logger.debug(
                            "save_equity_snapshot: skip bogus 10000/0/0 (have %d open trades)",
                            open_cnt,
                        )
                        return
            except Exception:
                pass

        try:
            with self._get_conn() as conn:
                conn.execute(sql, params)
            self.logger.debug(
                "save_equity_snapshot: equity=%.2f drawdown=%.2f%%",
                params["equity"], params["drawdown_pct"],
            )
        except sqlite3.Error as exc:
            self.logger.error("save_equity_snapshot failed: %s", exc, exc_info=True)

    def get_equity_history(self, hours: int = 24) -> List[dict]:
        """
        Retrieve equity snapshots from the last N hours.

        Parameters
        ----------
        hours : int
            Lookback window in hours (default 24).

        Returns
        -------
        list of dicts, ordered by timestamp ascending.
        """
        cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
        sql = """
            SELECT timestamp, equity, cash, positions_value, drawdown_pct, open_trades
            FROM equity_snapshots
            WHERE timestamp >= :cutoff
            ORDER BY timestamp ASC
        """
        try:
            with self._get_conn() as conn:
                rows = conn.execute(sql, {"cutoff": cutoff}).fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error as exc:
            self.logger.error("get_equity_history failed: %s", exc, exc_info=True)
            return []

    # ------------------------------------------------------------------
    # Trade history
    # ------------------------------------------------------------------

    def get_trade_history(
        self,
        symbol: Optional[str] = None,
        limit: int = 50,
        status: Optional[str] = None,
    ) -> List[dict]:
        """
        Retrieve trade records from the database with optional filters.

        Parameters
        ----------
        symbol : str, optional
            Filter to a specific trading pair.
        limit : int
            Maximum number of records to return (default 50).
        status : str, optional
            Filter by trade status: 'open', 'closed', 'cancelled'.

        Returns
        -------
        list of dicts, ordered by entry_time descending.
        """
        conditions: List[str] = []
        params: Dict[str, Any] = {"limit": limit}

        if symbol is not None:
            conditions.append("symbol = :symbol")
            params["symbol"] = symbol

        if status is not None:
            conditions.append("status = :status")
            params["status"] = status.lower()

        where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        sql = f"""
            SELECT *
            FROM trades
            {where_clause}
            ORDER BY entry_time DESC
            LIMIT :limit
        """

        try:
            with self._get_conn() as conn:
                rows = conn.execute(sql, params).fetchall()
            results = []
            for row in rows:
                d = dict(row)
                # Deserialise the JSON signal_breakdown field.
                if d.get("signal_breakdown") and isinstance(d["signal_breakdown"], str):
                    try:
                        d["signal_breakdown"] = json.loads(d["signal_breakdown"])
                    except (json.JSONDecodeError, TypeError):
                        pass
                results.append(d)
            return results
        except sqlite3.Error as exc:
            self.logger.error("get_trade_history failed: %s", exc, exc_info=True)
            return []

    # ------------------------------------------------------------------
    # Open positions
    # ------------------------------------------------------------------

    def get_open_positions(self) -> List[dict]:
        """
        Return the currently open positions from the in-memory executor
        as a list of serialisable dicts.

        Uses ``executor.positions`` (trade_id → Trade) as the authoritative
        source so the API always reflects the latest in-memory state.

        Returns
        -------
        list of dicts, one per open position.
        """
        result: List[dict] = []
        for trade_id, trade in self.executor.positions.items():
            d = trade.to_dict() if hasattr(trade, "to_dict") else {}
            # Ensure trade_id is always present.
            d["trade_id"] = trade_id
            result.append(d)
        return result

    def get_open_trades_from_db(self) -> List[dict]:
        """
        Fetch open trades from the database (for recovery after restart).
        Returns list of dicts with all trade fields.
        """
        return self.get_trade_history(status="open", limit=100)

    def get_total_realized_pnl(self) -> float:
        """
        Return the sum of net_pnl for all closed trades.
        Used by restore_open_positions to compute correct cash after restart.
        """
        try:
            with self._get_conn() as conn:
                row = conn.execute(
                    "SELECT COALESCE(SUM(net_pnl), 0) AS total FROM trades WHERE status = 'closed'"
                ).fetchone()
                return float(row["total"]) if row else 0.0
        except sqlite3.Error as exc:
            self.logger.error("get_total_realized_pnl failed: %s", exc)
            return 0.0

    # ------------------------------------------------------------------
    # Signal log
    # ------------------------------------------------------------------

    def log_signal(self, signal_data: dict) -> None:
        """
        Insert a signal event into the ``signals_log`` table.

        Parameters
        ----------
        signal_data : dict
            Expected keys:
            - timestamp (str ISO-8601, defaults to now)
            - symbol (str)
            - timeframe (str)
            - confidence_score (float)
            - direction (str: long/short/neutral)
            - layer1_score, layer2_score, layer3_score, ml_score (float)
            - raw_signals (dict, will be JSON-serialised)
            - action_taken (str, must match schema CHECK constraint)
            - trade_id (str, optional FK)
        """
        sql = """
            INSERT INTO signals_log (
                timestamp, symbol, timeframe,
                confidence_score, direction,
                layer1_score, layer2_score, layer3_score, ml_score,
                raw_signals, action_taken, trade_id
            ) VALUES (
                :timestamp, :symbol, :timeframe,
                :confidence_score, :direction,
                :layer1_score, :layer2_score, :layer3_score, :ml_score,
                :raw_signals, :action_taken, :trade_id
            )
        """
        raw_signals = signal_data.get("raw_signals", {})
        if isinstance(raw_signals, dict):
            raw_signals = json.dumps(raw_signals)

        params = {
            "timestamp": signal_data.get("timestamp", datetime.utcnow().isoformat()),
            "symbol": signal_data.get("symbol", "UNKNOWN"),
            "timeframe": signal_data.get("timeframe", "1h"),
            "confidence_score": float(signal_data.get("confidence_score", 0.0)),
            "direction": str(signal_data.get("direction", "neutral")).lower(),
            "layer1_score": float(signal_data.get("layer1_score", 0.0)),
            "layer2_score": float(signal_data.get("layer2_score", 0.0)),
            "layer3_score": float(signal_data.get("layer3_score", 0.0)),
            "ml_score": float(signal_data.get("ml_score", 0.0)),
            "raw_signals": raw_signals,
            "action_taken": signal_data.get("action_taken", "skipped_threshold"),
            "trade_id": signal_data.get("trade_id"),
        }

        try:
            with self._get_conn() as conn:
                conn.execute(sql, params)
            self.logger.debug(
                "log_signal: %s %s conf=%.2f action=%s",
                params["symbol"], params["direction"],
                params["confidence_score"], params["action_taken"],
            )
        except sqlite3.Error as exc:
            self.logger.error("log_signal failed: %s", exc, exc_info=True)

    def get_signal_history(
        self,
        symbol: Optional[str] = None,
        limit: int = 100,
    ) -> List[dict]:
        """
        Retrieve signal log entries, ordered by timestamp descending.

        Parameters
        ----------
        symbol : str, optional
            Filter to a specific trading pair.
        limit : int
            Maximum number of records (default 100).

        Returns
        -------
        list of dicts.
        """
        conditions: List[str] = []
        params: Dict[str, Any] = {"limit": limit}

        if symbol is not None:
            conditions.append("symbol = :symbol")
            params["symbol"] = symbol

        where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        sql = f"""
            SELECT *
            FROM signals_log
            {where_clause}
            ORDER BY timestamp DESC
            LIMIT :limit
        """

        try:
            with self._get_conn() as conn:
                rows = conn.execute(sql, params).fetchall()
            results = []
            for row in rows:
                d = dict(row)
                if d.get("raw_signals") and isinstance(d["raw_signals"], str):
                    try:
                        d["raw_signals"] = json.loads(d["raw_signals"])
                    except (json.JSONDecodeError, TypeError):
                        pass
                results.append(d)
            return results
        except sqlite3.Error as exc:
            self.logger.error("get_signal_history failed: %s", exc, exc_info=True)
            return []

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    def get_summary_stats(self) -> dict:
        """
        Compute high-level performance statistics from all closed trades
        stored in the database.

        Returns
        -------
        dict with keys:
            total_trades, win_rate, total_pnl_pct,
            best_trade_pnl, worst_trade_pnl,
            avg_win_pct, avg_loss_pct,
            profit_factor
        """
        sql = """
            SELECT
                COUNT(*)                                            AS total_trades,
                SUM(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END)      AS winning_trades,
                SUM(CASE WHEN net_pnl <= 0 THEN 1 ELSE 0 END)     AS losing_trades,
                SUM(net_pnl)                                        AS total_pnl,
                SUM(pnl_pct)                                        AS total_pnl_pct,
                MAX(net_pnl)                                        AS best_trade_pnl,
                MIN(net_pnl)                                        AS worst_trade_pnl,
                AVG(CASE WHEN net_pnl > 0 THEN pnl_pct END)       AS avg_win_pct,
                AVG(CASE WHEN net_pnl <= 0 THEN pnl_pct END)      AS avg_loss_pct,
                SUM(CASE WHEN net_pnl > 0 THEN net_pnl ELSE 0 END) AS gross_profit,
                ABS(SUM(CASE WHEN net_pnl <= 0 THEN net_pnl ELSE 0 END)) AS gross_loss
            FROM trades
            WHERE status = 'closed'
        """

        default_stats = {
            "total_trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "total_pnl_pct": 0.0,
            "best_trade_pnl": 0.0,
            "worst_trade_pnl": 0.0,
            "avg_win_pct": 0.0,
            "avg_loss_pct": 0.0,
            "profit_factor": 0.0,
        }

        try:
            with self._get_conn() as conn:
                row = conn.execute(sql).fetchone()

            if row is None or row["total_trades"] == 0:
                return default_stats

            total_trades = int(row["total_trades"] or 0)
            winning_trades = int(row["winning_trades"] or 0)

            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

            gross_profit = float(row["gross_profit"] or 0.0)
            gross_loss = float(row["gross_loss"] or 0.0)
            profit_factor = (
                gross_profit / gross_loss if gross_loss > 0.0 else float("inf")
            )

            return {
                "total_trades": total_trades,
                "win_rate": round(win_rate, 4),
                "total_pnl": round(float(row["total_pnl"] or 0.0), 4),
                "total_pnl_pct": round(float(row["total_pnl_pct"] or 0.0), 4),
                "best_trade_pnl": round(float(row["best_trade_pnl"] or 0.0), 4),
                "worst_trade_pnl": round(float(row["worst_trade_pnl"] or 0.0), 4),
                "avg_win_pct": round(float(row["avg_win_pct"] or 0.0), 4),
                "avg_loss_pct": round(float(row["avg_loss_pct"] or 0.0), 4),
                "profit_factor": (
                    round(profit_factor, 4) if profit_factor != float("inf") else None
                ),
            }

        except sqlite3.Error as exc:
            self.logger.error("get_summary_stats failed: %s", exc, exc_info=True)
            return default_stats
