from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from datetime import UTC, datetime
from typing import Any

from hear.db import HearDB, db
from hear.models.events import BackendEventType, EventLevel, EventSource

_LEVEL_TO_SEVERITY = {
    logging.DEBUG: "DEBUG",
    logging.INFO: "INFO",
    logging.WARNING: "WARNING",
    logging.ERROR: "ERROR",
    logging.CRITICAL: "CRITICAL",
}

_EVENT_LEVEL_TO_PY_LEVEL: dict[EventLevel, int] = {
    EventLevel.DEBUG: logging.DEBUG,
    EventLevel.INFO: logging.INFO,
    EventLevel.WARN: logging.WARNING,
    EventLevel.ERROR: logging.ERROR,
}


class CloudJsonFormatter(logging.Formatter):
    """Formats logs as single-line JSON compatible with Google Cloud Logging."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "severity": _LEVEL_TO_SEVERITY.get(record.levelno, "INFO"),
            "logger": record.name,
            "message": record.getMessage(),
        }
        event_type = getattr(record, "event_type", None)
        if event_type:
            entry["event_type"] = event_type
        client_id = getattr(record, "client_id", None)
        if client_id:
            entry["client_id"] = client_id
        session_id = getattr(record, "session_id", None)
        if session_id:
            entry["session_id"] = session_id
        payload = getattr(record, "payload", None)
        if isinstance(payload, dict) and payload:
            entry["payload"] = payload
        if record.exc_info:
            entry["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure the 'hear' logger hierarchy to emit structured JSON to stdout."""
    hear_logger = logging.getLogger("hear")
    hear_logger.setLevel(level)
    if not any(isinstance(h.formatter, CloudJsonFormatter) for h in hear_logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(CloudJsonFormatter())
        hear_logger.handlers = [handler]
    hear_logger.propagate = False


_event_logger = logging.getLogger("hear.events")


def log_event(
    event_type: BackendEventType | str,
    *,
    level: EventLevel = EventLevel.INFO,
    session_id: str | None = None,
    client_id: str | None = None,
    persist: bool = True,
    target_db: HearDB | None = None,
    **payload: Any,
) -> None:
    """Emit a structured backend log line and optionally persist it to SQLite events."""
    py_level = _EVENT_LEVEL_TO_PY_LEVEL.get(level, logging.INFO)
    event_name = str(event_type)
    _event_logger.log(
        py_level,
        event_name,
        extra={
            "event_type": event_name,
            "session_id": session_id,
            "client_id": client_id,
            "payload": payload,
        },
    )
    if persist:
        store = target_db or db
        try:
            store.record_event(
                source=EventSource.BACKEND,
                level=level,
                event_type=event_type,
                payload=payload,
                session_id=session_id,
                client_id=client_id,
                ts=time.time(),
            )
        except Exception as exc:  # noqa: BLE001
            _event_logger.warning("Failed to persist event %s to SQLite: %s", event_name, exc)


async def log_event_async(
    event_type: BackendEventType | str,
    *,
    level: EventLevel = EventLevel.INFO,
    session_id: str | None = None,
    client_id: str | None = None,
    persist: bool = True,
    target_db: HearDB | None = None,
    **payload: Any,
) -> None:
    """Async wrapper offloading SQLite event persistence off the event loop."""
    py_level = _EVENT_LEVEL_TO_PY_LEVEL.get(level, logging.INFO)
    event_name = str(event_type)
    _event_logger.log(
        py_level,
        event_name,
        extra={
            "event_type": event_name,
            "session_id": session_id,
            "client_id": client_id,
            "payload": payload,
        },
    )
    if persist:
        store = target_db or db
        try:
            await asyncio.to_thread(
                store.record_event,
                source=EventSource.BACKEND,
                level=level,
                event_type=event_type,
                payload=payload,
                session_id=session_id,
                client_id=client_id,
                ts=time.time(),
            )
        except Exception as exc:  # noqa: BLE001
            _event_logger.warning("Failed to persist event %s to SQLite: %s", event_name, exc)
