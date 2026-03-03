"""
src/core/event_bus.py

Asynchronous pub/sub event bus for the algorithmic trading engine.
Supports both sync and async subscribers, thread-safe publishing,
and a singleton pattern for global access.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import threading
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

class EventType(Enum):
    OHLCV_UPDATE = "ohlcv_update"
    SIGNAL_GENERATED = "signal_generated"
    TRADE_SIGNAL = "trade_signal"
    ORDER_FILLED = "order_filled"
    POSITION_CLOSED = "position_closed"
    EQUITY_UPDATE = "equity_update"
    BOT_STATUS_CHANGED = "bot_status_changed"
    ERROR = "error"
    ALERT = "alert"
    SENTIMENT_UPDATE = "sentiment_update"


# ---------------------------------------------------------------------------
# Event
# ---------------------------------------------------------------------------

class Event:
    """
    Immutable envelope that travels through the bus.

    Attributes
    ----------
    type      : EventType  – what happened
    data      : dict       – payload (caller-defined schema per event type)
    source    : str        – logical name of the component that emitted this
    timestamp : datetime   – UTC moment of creation
    """

    __slots__ = ("type", "data", "source", "timestamp")

    def __init__(self, type: EventType, data: dict, source: str = "") -> None:
        if not isinstance(type, EventType):
            raise TypeError(f"type must be an EventType, got {type!r}")
        if not isinstance(data, dict):
            raise TypeError(f"data must be a dict, got {type(data)}")

        self.type: EventType = type
        self.data: dict = data
        self.source: str = source
        self.timestamp: datetime = datetime.utcnow()

    def __repr__(self) -> str:
        return (
            f"Event(type={self.type.value!r}, source={self.source!r}, "
            f"timestamp={self.timestamp.isoformat()})"
        )


# ---------------------------------------------------------------------------
# EventBus
# ---------------------------------------------------------------------------

class EventBus:
    """
    Singleton async pub/sub bus.

    Usage
    -----
    bus = EventBus.get_instance()

    # Subscribe (sync or async callback):
    bus.subscribe(EventType.ORDER_FILLED, my_async_handler)
    bus.subscribe(EventType.ERROR, my_sync_handler)

    # Publish from async context:
    await bus.publish(Event(EventType.ORDER_FILLED, {...}, source="executor"))

    # Publish from sync context (fire-and-forget onto the running loop):
    bus.publish_sync(Event(EventType.ERROR, {...}))
    """

    _instance: Optional["EventBus"] = None
    _instance_lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # Singleton
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        # Keyed by EventType; each value is a list of callables.
        self._subscribers: Dict[EventType, List[Callable]] = {
            et: [] for et in EventType
        }
        self._lock: threading.RLock = threading.RLock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        logger.debug("EventBus initialised.")

    @classmethod
    def get_instance(cls) -> "EventBus":
        """Return the process-wide singleton."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Tear down the singleton (useful in tests)."""
        with cls._instance_lock:
            cls._instance = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create_loop(self) -> asyncio.AbstractEventLoop:
        """
        Return the running event loop if one exists, otherwise create a new
        one and store it for future publish_sync calls.
        """
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            pass

        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def subscribe(self, event_type: EventType, callback: Callable) -> None:
        """
        Register *callback* to be invoked whenever an event of *event_type*
        is published.  The callback may be a plain function or a coroutine
        function; both are handled transparently.

        Parameters
        ----------
        event_type : EventType
        callback   : Callable[[Event], Any]  – sync or async
        """
        if not isinstance(event_type, EventType):
            raise TypeError(f"event_type must be an EventType, got {event_type!r}")
        if not callable(callback):
            raise TypeError(f"callback must be callable, got {callback!r}")

        with self._lock:
            if callback not in self._subscribers[event_type]:
                self._subscribers[event_type].append(callback)
                logger.debug(
                    "Subscribed %s to %s", callback.__qualname__, event_type.value
                )

    def unsubscribe(self, event_type: EventType, callback: Callable) -> None:
        """Remove a previously registered callback."""
        with self._lock:
            try:
                self._subscribers[event_type].remove(callback)
                logger.debug(
                    "Unsubscribed %s from %s",
                    callback.__qualname__,
                    event_type.value,
                )
            except ValueError:
                pass  # callback was never registered – silently ignore

    def subscriber_count(self, event_type: EventType) -> int:
        """Return the number of subscribers for a given event type."""
        with self._lock:
            return len(self._subscribers[event_type])

    async def publish(self, event: Event) -> None:
        """
        Dispatch *event* to all registered subscribers.

        Async subscribers are awaited directly; sync subscribers are called
        in the current thread.  Exceptions raised by individual subscribers
        are caught, logged, and do not prevent other subscribers from running.

        Parameters
        ----------
        event : Event
        """
        if not isinstance(event, Event):
            raise TypeError(f"Expected Event, got {type(event)}")

        with self._lock:
            callbacks = list(self._subscribers[event.type])

        logger.debug(
            "Publishing %s from %r to %d subscriber(s)",
            event.type.value,
            event.source,
            len(callbacks),
        )

        for cb in callbacks:
            try:
                if inspect.iscoroutinefunction(cb):
                    await cb(event)
                else:
                    cb(event)
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "Subscriber %s raised an exception while handling %s: %s",
                    getattr(cb, "__qualname__", repr(cb)),
                    event.type.value,
                    exc,
                )

    def publish_sync(self, event: Event) -> None:
        """
        Publish *event* from a synchronous context.

        Behaviour depends on whether an event loop is already running:

        * **Running loop found** – the coroutine is scheduled as a task via
          ``asyncio.run_coroutine_threadsafe`` (safe to call from any thread).
        * **No running loop** – falls back to ``asyncio.run`` on an internal
          loop so that async subscribers are still awaited.

        Parameters
        ----------
        event : Event
        """
        if not isinstance(event, Event):
            raise TypeError(f"Expected Event, got {type(event)}")

        try:
            loop = asyncio.get_running_loop()
            # We are inside an already-running loop (e.g. called from a sync
            # method that lives inside an async task).  Schedule the coroutine
            # and return immediately; the caller cannot await the result.
            future = asyncio.run_coroutine_threadsafe(self.publish(event), loop)
            # Optional: surface exceptions in debug mode.
            future.add_done_callback(self._future_exception_logger)
        except RuntimeError:
            # No running loop – use the internal loop synchronously.
            loop = self._get_or_create_loop()
            loop.run_until_complete(self.publish(event))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _future_exception_logger(future: "asyncio.Future") -> None:
        """Log unhandled exceptions from fire-and-forget futures."""
        exc = future.exception()
        if exc is not None:
            logger.error("EventBus fire-and-forget task raised: %s", exc)
