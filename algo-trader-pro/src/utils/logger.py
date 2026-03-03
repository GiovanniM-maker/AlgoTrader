"""
AlgoTrader Pro - Structured Logger
====================================
Provides structured logging with:
  - JSON formatter for production environments
  - Human-readable console formatter for development
  - File rotation handler writing to logs/ directory
  - Log level driven by application config
  - Thread-safe, per-name logger registry
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LOGS_DIR = _PROJECT_ROOT / "logs"
_LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_LOG_LEVEL = "INFO"
_LOG_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"
_CONSOLE_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
)
_LOCK = threading.Lock()
_CONFIGURED = False

# ---------------------------------------------------------------------------
# JSON Formatter
# ---------------------------------------------------------------------------


class JSONFormatter(logging.Formatter):
    """
    Formats log records as single-line JSON objects.

    Output example::

        {"ts":"2024-03-01T12:00:00.123Z","level":"INFO","logger":"data.coingecko",
         "msg":"Fetched 100 candles","symbol":"BTCUSDT","timeframe":"1h"}
    """

    RESERVED_ATTRS = frozenset(
        (
            "args",
            "asctime",
            "created",
            "exc_info",
            "exc_text",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "module",
            "msecs",
            "message",
            "msg",
            "name",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "thread",
            "threadName",
        )
    )

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        # Core fields
        log_obj: Dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%S.%f"
            )[:-3]
            + "Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        # Source location (only for WARNING+)
        if record.levelno >= logging.WARNING:
            log_obj["file"] = record.pathname
            log_obj["line"] = record.lineno
            log_obj["func"] = record.funcName

        # Exception info
        if record.exc_info:
            log_obj["exc"] = self.formatException(record.exc_info)
        if record.exc_text:
            log_obj["exc_text"] = record.exc_text
        if record.stack_info:
            log_obj["stack"] = record.stack_info

        # Extra fields added via logger.info("msg", extra={...})
        for key, value in record.__dict__.items():
            if key not in self.RESERVED_ATTRS and not key.startswith("_"):
                try:
                    json.dumps(value)  # ensure serializable
                    log_obj[key] = value
                except (TypeError, ValueError):
                    log_obj[key] = str(value)

        return json.dumps(log_obj, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Console (Human-Readable) Formatter
# ---------------------------------------------------------------------------


class ConsoleFormatter(logging.Formatter):
    """
    Colorized, human-readable formatter for development use.
    Falls back to plain text if the terminal does not support ANSI codes.
    """

    GREY = "\x1b[38;20m"
    GREEN = "\x1b[32;20m"
    YELLOW = "\x1b[33;20m"
    RED = "\x1b[31;20m"
    BOLD_RED = "\x1b[31;1m"
    CYAN = "\x1b[36;20m"
    RESET = "\x1b[0m"

    LEVEL_COLORS = {
        logging.DEBUG: GREY,
        logging.INFO: GREEN,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: BOLD_RED,
    }

    def __init__(self, use_colors: bool = True) -> None:
        super().__init__(fmt=_CONSOLE_FORMAT, datefmt=_LOG_DATE_FORMAT)
        self._use_colors = use_colors and sys.stderr.isatty()

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        if self._use_colors:
            color = self.LEVEL_COLORS.get(record.levelno, self.GREY)
            record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


# ---------------------------------------------------------------------------
# Root logger setup
# ---------------------------------------------------------------------------


def _resolve_log_level() -> int:
    """
    Resolve log level from:
    1. LOG_LEVEL environment variable
    2. Application config (if already loaded)
    3. Default: INFO
    """
    env_level = os.getenv("LOG_LEVEL", "").upper()
    if env_level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        return getattr(logging, env_level)

    # Try to get from config without creating a circular import
    try:
        from src.utils.config_loader import get_config  # noqa: PLC0415

        cfg_level = get_config().app.log_level.upper()
        return getattr(logging, cfg_level, logging.INFO)
    except Exception:
        return logging.INFO


def _resolve_environment() -> str:
    """
    Detect the current environment (paper/live/etc.) for formatter selection.
    """
    env = os.getenv("APP_ENV", "paper").lower()
    try:
        from src.utils.config_loader import get_config  # noqa: PLC0415

        env = get_config().app.environment.lower()
    except Exception:
        pass
    return env


def configure_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    environment: Optional[str] = None,
    force: bool = False,
) -> None:
    """
    Configure the root logger with console and optional file handlers.

    This function is idempotent — calling it multiple times has no effect
    unless ``force=True`` is passed.

    Args:
        log_level:   Override log level (e.g. "DEBUG"). If None, auto-detected.
        log_file:    Log file name (placed in logs/). Defaults to "algotrader.log".
        environment: "paper" or "live" drives formatter selection.
        force:       Re-configure even if already set up.
    """
    global _CONFIGURED

    with _LOCK:
        if _CONFIGURED and not force:
            return

        resolved_level = (
            getattr(logging, log_level.upper(), logging.INFO)
            if log_level
            else _resolve_log_level()
        )
        resolved_env = environment or _resolve_environment()
        use_json = resolved_env == "live"

        root = logging.getLogger()
        root.setLevel(resolved_level)

        # Clear any existing handlers
        root.handlers.clear()

        # --- Console handler ---
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(resolved_level)
        if use_json:
            console_handler.setFormatter(JSONFormatter())
        else:
            console_handler.setFormatter(ConsoleFormatter(use_colors=True))
        root.addHandler(console_handler)

        # --- Rotating file handler ---
        log_filename = log_file or "algotrader.log"
        log_path = _LOGS_DIR / log_filename

        file_handler = logging.handlers.RotatingFileHandler(
            filename=str(log_path),
            maxBytes=50 * 1024 * 1024,  # 50 MB per file
            backupCount=10,
            encoding="utf-8",
        )
        file_handler.setLevel(resolved_level)
        file_handler.setFormatter(JSONFormatter())  # always JSON in files
        root.addHandler(file_handler)

        # --- Error-only file handler (for quick error review) ---
        error_log_path = _LOGS_DIR / "errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            filename=str(error_log_path),
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(JSONFormatter())
        root.addHandler(error_handler)

        # Silence noisy third-party loggers
        for noisy in ("urllib3", "requests", "aiohttp", "asyncio", "httpcore"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

        _CONFIGURED = True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger for the given name, configuring the root logger on first call.

    Usage::

        from src.utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Starting data fetch", extra={"symbol": "BTCUSDT"})
        logger.warning("Rate limit hit", extra={"retry_in": 60})
        logger.error("Unexpected error", exc_info=True)

    Args:
        name: Logger name — typically ``__name__`` from the calling module.

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)


def log_exception(
    logger: logging.Logger,
    message: str,
    exc: Optional[BaseException] = None,
    **extra: Any,
) -> None:
    """
    Log an exception with full traceback, structured as a JSON-compatible record.

    Args:
        logger:  Logger instance.
        message: Human-readable error description.
        exc:     Exception to log (uses sys.exc_info() if not provided).
        **extra: Additional key-value pairs to include in the log record.
    """
    exc_info = (type(exc), exc, exc.__traceback__) if exc else sys.exc_info()
    tb_str = "".join(traceback.format_exception(*exc_info)) if exc_info[0] else ""

    extra_data: Dict[str, Any] = {"traceback": tb_str, **extra}
    logger.error(message, exc_info=exc_info, extra=extra_data)


# ---------------------------------------------------------------------------
# Module-level auto-configure when imported
# ---------------------------------------------------------------------------

configure_logging()

# ---------------------------------------------------------------------------
# Internal module logger
# ---------------------------------------------------------------------------

_logger = get_logger(__name__)
_logger.debug("Logger module initialized", extra={"logs_dir": str(_LOGS_DIR)})
