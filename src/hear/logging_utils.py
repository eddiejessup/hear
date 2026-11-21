from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from datetime import UTC, datetime
from typing import Any

from hear.db import HearDB
from hear.models.events import BackendEventType, EventLevel, EventSource

_LEVEL_TO_SEVERITY = {
    logging.DEBUG: "DEBUG",
    logging.INFO: "INFO",
    logging.WARNING: "WARNING",
    logging.ERROR: "ERROR",
    logging.CRITICAL: "CRITICAL",
}

_EVENT_LEVEL_TO_PY = {
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


class EventLogger:
    """Emits structured backend logs and optionally persists them to an injected HearDB."""

    def __init__(self, store: HearDB | None = None, logger_name: str = "hear.events") -> None:
        self._store = store
        self._logger = logging.getLogger(logger_name)

    def log(
        self,
        event_type: BackendEventType | str,
        *,
        level: EventLevel = EventLevel.INFO,
        session_id: str | None = None,
        client_id: str | None = None,
        persist: bool = True,
        **payload: Any,
    ) -> None:
        py_level = _EVENT_LEVEL_TO_PY.get(level, logging.INFO)
        self._logger.log(
            py_level,
            str(event_type),
            extra={
                "event_type": str(event_type),
                "session_id": session_id,
                "client_id": client_id,
                "payload": payload,
            },
        )
        if persist and self._store is not None:
            try:
                self._store.record_event(
                    source=EventSource.BACKEND,
                    level=level,
                    event_type=event_type,
                    payload=payload,
                    session_id=session_id,
                    client_id=client_id,
                    ts=time.time(),
                )
            except Exception as exc:  # noqa: BLE001
                self._logger.warning("Failed to persist event %s to SQLite: %s", event_type, exc)

    async def log_async(
        self,
        event_type: BackendEventType | str,
        *,
        level: EventLevel = EventLevel.INFO,
        session_id: str | None = None,
        client_id: str | None = None,
        persist: bool = True,
        **payload: Any,
    ) -> None:
        py_level = _EVENT_LEVEL_TO_PY.get(level, logging.INFO)
        self._logger.log(
            py_level,
            str(event_type),
            extra={
                "event_type": str(event_type),
                "session_id": session_id,
                "client_id": client_id,
                "payload": payload,
            },
        )
        if persist and self._store is not None:
            try:
                await asyncio.to_thread(
                    self._store.record_event,
                    source=EventSource.BACKEND,
                    level=level,
                    event_type=event_type,
                    payload=payload,
                    session_id=session_id,
                    client_id=client_id,
                    ts=time.time(),
                )
            except Exception as exc:  # noqa: BLE001
                self._logger.warning("Failed to persist event %s to SQLite: %s", event_type, exc)
