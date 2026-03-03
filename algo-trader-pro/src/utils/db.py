"""
AlgoTrader Pro - Database Engine & Session Management
======================================================
SQLAlchemy 2.0 sync setup for SQLite (Phase 1).
Provides:
  - Engine creation with WAL mode and connection pool
  - Thread-safe session factory via sessionmaker
  - `get_db()` context manager for safe session lifecycle
  - `init_db()` that bootstraps the schema from schema.sql
  - Declarative Base for ORM models
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA_PATH = _PROJECT_ROOT / "database" / "schema.sql"

# ---------------------------------------------------------------------------
# Default database URL
# ---------------------------------------------------------------------------

_DEFAULT_DB_URL = "sqlite:///./database/algotrader.db"


def _resolve_db_url() -> str:
    """
    Resolve the database URL with the following priority:
    1. DATABASE_URL environment variable
    2. Application config (if loaded)
    3. Default SQLite path
    """
    env_url = os.getenv("DATABASE_URL")
    if env_url:
        return env_url

    try:
        from src.utils.config_loader import get_config  # noqa: PLC0415

        return get_config().database.url
    except Exception:
        return _DEFAULT_DB_URL


# ---------------------------------------------------------------------------
# SQLite-specific event hooks
# ---------------------------------------------------------------------------


def _configure_sqlite(dbapi_conn: object, _connection_record: object) -> None:
    """
    Called on every new SQLite connection.
    Enables WAL journal mode and foreign key enforcement.
    """
    cursor = dbapi_conn.cursor()  # type: ignore[attr-defined]
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA cache_size=-64000")   # 64 MB page cache
    cursor.execute("PRAGMA temp_store=MEMORY")
    cursor.close()


# ---------------------------------------------------------------------------
# Engine singleton
# ---------------------------------------------------------------------------

_engine_lock = threading.Lock()
_engine_instance: Optional[Engine] = None


def get_engine(db_url: Optional[str] = None, echo: bool = False) -> Engine:
    """
    Return the singleton SQLAlchemy Engine.

    Creates the engine on first call and caches it for the process lifetime.

    Args:
        db_url: Override the database URL (defaults to config/env resolution).
        echo:   If True, log all SQL statements. Useful for debugging.

    Returns:
        A configured :class:`sqlalchemy.engine.Engine`.
    """
    global _engine_instance

    if _engine_instance is not None:
        return _engine_instance

    with _engine_lock:
        if _engine_instance is not None:
            return _engine_instance

        resolved_url = db_url or _resolve_db_url()

        # Resolve echo setting from config if not explicitly passed
        if not echo:
            try:
                from src.utils.config_loader import get_config  # noqa: PLC0415

                echo = get_config().database.echo
            except Exception:
                echo = False

        logger.info("Creating database engine", extra={"url": resolved_url})

        is_sqlite = resolved_url.startswith("sqlite")

        connect_args: dict = {}
        engine_kwargs: dict = {
            "echo": echo,
        }

        if is_sqlite:
            connect_args["check_same_thread"] = False
            engine_kwargs["connect_args"] = connect_args
            # SQLite doesn't benefit from a pool — use StaticPool or NullPool for file DBs
            # For a file-based SQLite, the default SingletonThreadPool is fine.
            # We explicitly set pool_pre_ping to detect stale connections.
            engine_kwargs["pool_pre_ping"] = True
        else:
            # For future PostgreSQL / other engines
            engine_kwargs["pool_size"] = 5
            engine_kwargs["max_overflow"] = 10
            engine_kwargs["pool_pre_ping"] = True

        engine = create_engine(resolved_url, **engine_kwargs)

        # Register SQLite PRAGMAs for every new connection
        if is_sqlite:
            event.listen(engine, "connect", _configure_sqlite)

        _engine_instance = engine
        logger.info("Database engine created", extra={"url": resolved_url})
        return _engine_instance


# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------

_session_factory: Optional[sessionmaker] = None
_session_lock = threading.Lock()


def get_session_factory(engine: Optional[Engine] = None) -> sessionmaker:
    """
    Return the singleton session factory (sessionmaker).

    Args:
        engine: Override the engine (defaults to get_engine()).

    Returns:
        A configured :class:`sqlalchemy.orm.sessionmaker`.
    """
    global _session_factory

    if _session_factory is not None:
        return _session_factory

    with _session_lock:
        if _session_factory is not None:
            return _session_factory

        resolved_engine = engine or get_engine()
        _session_factory = sessionmaker(
            bind=resolved_engine,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
        )
        logger.debug("Session factory created")
        return _session_factory


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """
    Provide a transactional database session as a context manager.

    Automatically commits on success and rolls back on any exception.
    Always closes the session on exit.

    Usage::

        from src.utils.db import get_db

        with get_db() as session:
            result = session.execute(text("SELECT * FROM trades WHERE status='open'"))
            rows = result.fetchall()

        # For ORM usage:
        with get_db() as session:
            trade = session.get(Trade, trade_id)
            trade.status = "closed"
            # session.commit() called automatically on exit

    Yields:
        :class:`sqlalchemy.orm.Session` — an active database session.

    Raises:
        Any exception from database operations (after rollback).
    """
    factory = get_session_factory()
    session: Session = factory()
    try:
        yield session
        session.commit()
        logger.debug("DB session committed")
    except Exception as exc:
        session.rollback()
        logger.error(
            "DB session rolled back due to exception",
            extra={"error": str(exc), "error_type": type(exc).__name__},
        )
        raise
    finally:
        session.close()
        logger.debug("DB session closed")


# ---------------------------------------------------------------------------
# Schema initialization
# ---------------------------------------------------------------------------


def init_db(engine: Optional[Engine] = None) -> None:
    """
    Initialize the database by executing schema.sql.

    Idempotent — all statements use CREATE TABLE IF NOT EXISTS, so calling
    this multiple times is safe.

    Args:
        engine: Override the engine (defaults to get_engine()).

    Raises:
        FileNotFoundError: If schema.sql cannot be found.
        sqlalchemy.exc.SQLAlchemyError: On any database error.
    """
    resolved_engine = engine or get_engine()

    if not _SCHEMA_PATH.exists():
        raise FileNotFoundError(
            f"Schema file not found: {_SCHEMA_PATH}\n"
            "Ensure database/schema.sql exists in the project root."
        )

    logger.info("Initializing database schema", extra={"schema_path": str(_SCHEMA_PATH)})

    sql_content = _SCHEMA_PATH.read_text(encoding="utf-8")

    # Split on semicolons to execute individual statements
    # (SQLite's execute() does not support multi-statement strings)
    statements = [stmt.strip() for stmt in sql_content.split(";") if stmt.strip()]

    with resolved_engine.begin() as conn:
        for stmt in statements:
            if stmt:
                try:
                    conn.execute(text(stmt))
                except Exception as exc:
                    # Log but continue for non-critical statements (e.g. duplicate index)
                    logger.warning(
                        "Schema statement warning",
                        extra={"stmt_preview": stmt[:80], "error": str(exc)},
                    )

    # ---- Migrations (add columns to existing tables) ----
    _run_migrations(resolved_engine)

    logger.info(
        "Database schema initialized",
        extra={"statements_executed": len(statements)},
    )


def _run_migrations(engine: Engine) -> None:
    """Run schema migrations for existing databases."""
    migrations = [
        "ALTER TABLE trades ADD COLUMN leverage REAL DEFAULT 1.0",
    ]
    with engine.begin() as conn:
        for sql in migrations:
            try:
                conn.execute(text(sql))
                logger.debug("Migration applied: %s", sql[:60])
            except Exception as exc:
                if "duplicate column" in str(exc).lower():
                    pass  # Column already exists
                else:
                    logger.warning("Migration skipped: %s", exc)


def drop_all_tables(engine: Optional[Engine] = None) -> None:
    """
    Drop all application tables. USE WITH EXTREME CAUTION.
    Only available in paper/test environments.

    Args:
        engine: Override the engine.

    Raises:
        RuntimeError: If called in live environment.
    """
    try:
        from src.utils.config_loader import get_config  # noqa: PLC0415

        if get_config().app.environment == "live":
            raise RuntimeError(
                "drop_all_tables() is DISABLED in live environment. "
                "This is a safety guard to prevent data loss."
            )
    except ImportError:
        pass

    resolved_engine = engine or get_engine()
    tables = [
        "bot_events",
        "backtest_results",
        "ml_model_runs",
        "sentiment_data",
        "signals_log",
        "equity_snapshots",
        "trades",
        "ohlcv",
        "schema_version",
    ]

    logger.warning("Dropping all tables", extra={"tables": tables})

    with resolved_engine.begin() as conn:
        conn.execute(text("PRAGMA foreign_keys=OFF"))
        for table in tables:
            conn.execute(text(f"DROP TABLE IF EXISTS {table}"))
        conn.execute(text("PRAGMA foreign_keys=ON"))

    logger.warning("All tables dropped")


def get_db_stats(engine: Optional[Engine] = None) -> dict:
    """
    Return basic database statistics (row counts per table).

    Args:
        engine: Override the engine.

    Returns:
        Dict mapping table_name -> row_count.
    """
    resolved_engine = engine or get_engine()
    tables = [
        "ohlcv",
        "trades",
        "equity_snapshots",
        "signals_log",
        "sentiment_data",
        "ml_model_runs",
        "backtest_results",
        "bot_events",
    ]

    stats: dict = {}
    with resolved_engine.connect() as conn:
        for table in tables:
            try:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                stats[table] = result.scalar() or 0
            except Exception:
                stats[table] = -1  # Table does not exist yet

    return stats


# ---------------------------------------------------------------------------
# Declarative Base for ORM models
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    """
    Base class for all SQLAlchemy ORM models.

    Usage in a model file::

        from src.utils.db import Base
        from sqlalchemy.orm import Mapped, mapped_column
        from sqlalchemy import String, Float, Integer

        class Trade(Base):
            __tablename__ = "trades"
            id: Mapped[int] = mapped_column(Integer, primary_key=True)
            symbol: Mapped[str] = mapped_column(String(20), nullable=False)
            ...
    """

    pass


# ---------------------------------------------------------------------------
# Module initialization — ensure engine is ready when module is imported
# ---------------------------------------------------------------------------

logger.debug("db module loaded", extra={"schema_path": str(_SCHEMA_PATH)})
