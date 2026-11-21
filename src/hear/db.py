from __future__ import annotations

import json
import math
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hear.models.config import AppConfig
from hear.models.events import BackendEventType, EventLevel, EventSource
from hear.models.gemini import GeminiContent, GeminiPart, GroundingSource
from hear.models.session import (
    AssistantMessage,
    LensSummary,
    MessageLocation,
    TriggerLocation,
    UIMessage,
    UserChatMessage,
)

SCHEMA_VERSION = 1
MAX_EVENT_PAYLOAD_BYTES = 8192
_MAX_FUTURE_TS_SKEW_SECONDS = 300.0

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    client_id TEXT NOT NULL,
    started_at REAL NOT NULL,
    last_active_at REAL NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1,
    last_trigger_lat REAL,
    last_trigger_lon REAL,
    last_trigger_label TEXT,
    last_trigger_ts REAL,
    last_audio_ended_at REAL,
    used_lens_ids_json TEXT NOT NULL DEFAULT '[]',
    used_axes_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_sessions_client_active
    ON sessions (client_id, is_active, last_active_at DESC);

CREATE TABLE IF NOT EXISTS messages (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    session_id TEXT NOT NULL,
    client_id TEXT NOT NULL,
    role TEXT NOT NULL,
    type TEXT NOT NULL,
    ts REAL NOT NULL,
    lat REAL,
    lon REAL,
    short_label TEXT,
    nearby_features_json TEXT NOT NULL DEFAULT '[]',
    lens_id TEXT,
    lens_name TEXT,
    era TEXT,
    scale TEXT,
    topic_summary TEXT,
    entities_json TEXT NOT NULL DEFAULT '[]',
    reason TEXT,
    model TEXT,
    text TEXT NOT NULL,
    prompt_text TEXT,
    audio_url TEXT,
    sources_json TEXT NOT NULL DEFAULT '[]',
    search_queries_json TEXT NOT NULL DEFAULT '[]',
    llm_latency_ms INTEGER,
    tts_latency_ms INTEGER,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_session_seq
    ON messages (session_id, seq);

CREATE INDEX IF NOT EXISTS idx_messages_client_ts
    ON messages (client_id, ts DESC);

CREATE INDEX IF NOT EXISTS idx_messages_client_geo
    ON messages (client_id, lat, lon)
    WHERE lat IS NOT NULL AND lon IS NOT NULL;

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    text,
    short_label,
    topic_summary,
    content='messages',
    content_rowid='seq'
);

CREATE TRIGGER IF NOT EXISTS messages_fts_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, text, short_label, topic_summary)
    VALUES (new.seq, new.text, COALESCE(new.short_label, ''), COALESCE(new.topic_summary, ''));
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_ad AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, text, short_label, topic_summary)
    VALUES (
        'delete',
        old.seq,
        old.text,
        COALESCE(old.short_label, ''),
        COALESCE(old.topic_summary, '')
    );
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_au AFTER UPDATE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, text, short_label, topic_summary)
    VALUES (
        'delete',
        old.seq,
        old.text,
        COALESCE(old.short_label, ''),
        COALESCE(old.topic_summary, '')
    );
    INSERT INTO messages_fts(rowid, text, short_label, topic_summary)
    VALUES (new.seq, new.text, COALESCE(new.short_label, ''), COALESCE(new.topic_summary, ''));
END;

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    source TEXT NOT NULL,
    level TEXT NOT NULL,
    session_id TEXT,
    client_id TEXT,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_events_ts ON events (ts DESC);
CREATE INDEX IF NOT EXISTS idx_events_client_ts ON events (client_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_events_session_ts ON events (session_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_events_level_ts ON events (level, ts DESC);

CREATE TABLE IF NOT EXISTS location_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    client_id TEXT NOT NULL,
    ts REAL NOT NULL,
    lat REAL NOT NULL,
    lon REAL NOT NULL,
    accuracy REAL,
    heading REAL,
    speed REAL,
    short_label TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_loc_session_ts ON location_samples (session_id, ts);
CREATE INDEX IF NOT EXISTS idx_loc_client_ts ON location_samples (client_id, ts DESC);

CREATE TABLE IF NOT EXISTS config_overrides (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    updated_at REAL NOT NULL
);
"""


@dataclass
class PersistedSessionState:
    session_id: str
    client_id: str
    started_at: float
    last_active_at: float
    last_trigger_location: TriggerLocation | None
    last_trigger_time: float | None
    last_audio_ended_at: float | None
    used_lens_ids: list[str] = field(default_factory=list)
    ui_messages: list[UIMessage] = field(default_factory=list)
    gemini_contents: list[GeminiContent] = field(default_factory=list)


def _safe_json_loads(raw: str | None, fallback: Any) -> Any:
    if not raw:
        return fallback
    try:
        return json.loads(raw)
    except ValueError:
        return fallback


def _normalize_event_ts(raw_ts: Any, now: float) -> float:
    if isinstance(raw_ts, bool) or not isinstance(raw_ts, (int, float)):
        return now
    ts = float(raw_ts)
    if not math.isfinite(ts):
        return now
    # Normalize JS millisecond timestamps (e.g. Date.now()) into Unix seconds
    if ts > 1e11:
        ts /= 1000.0
    if ts <= 0.0 or ts > now + _MAX_FUTURE_TS_SKEW_SECONDS:
        return now
    return ts


def _serialize_event_payload(payload: Any) -> str:
    raw = json.dumps(payload or {}, default=str)
    if len(raw.encode("utf-8")) <= MAX_EVENT_PAYLOAD_BYTES:
        return raw
    preview_limit = max(0, MAX_EVENT_PAYLOAD_BYTES - 128)
    preview = raw.encode("utf-8")[:preview_limit].decode("utf-8", errors="ignore")
    return json.dumps({"_truncated": True, "_preview": preview})


class HearDB:
    """Thread-safe SQLite store (WAL mode) for sessions, messages, events, and config."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None

    def _connect(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.execute("PRAGMA foreign_keys=ON;")
        self._migrate(conn)
        self._conn = conn
        return conn

    def _migrate(self, conn: sqlite3.Connection) -> None:
        row = conn.execute("PRAGMA user_version;").fetchone()
        version = int(row[0]) if row else 0
        if version < SCHEMA_VERSION:
            conn.executescript(_SCHEMA_SQL)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION};")
            conn.commit()

    def init(self) -> None:
        with self._lock:
            self._connect()

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.commit()
                finally:
                    self._conn.close()
                    self._conn = None

    @staticmethod
    def _upsert_session_in_tx(
        conn: sqlite3.Connection,
        *,
        session_id: str,
        client_id: str,
        started_at: float,
        last_active_at: float,
        is_active: bool,
        last_trigger_location: TriggerLocation | None,
        last_trigger_time: float | None,
        last_audio_ended_at: float | None,
        used_lens_ids: list[str],
    ) -> None:
        conn.execute(
            """
            INSERT INTO sessions (
                session_id, client_id, started_at, last_active_at, is_active,
                last_trigger_lat, last_trigger_lon, last_trigger_label, last_trigger_ts,
                last_audio_ended_at, used_lens_ids_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                last_active_at = excluded.last_active_at,
                is_active = excluded.is_active,
                last_trigger_lat = excluded.last_trigger_lat,
                last_trigger_lon = excluded.last_trigger_lon,
                last_trigger_label = excluded.last_trigger_label,
                last_trigger_ts = excluded.last_trigger_ts,
                last_audio_ended_at = excluded.last_audio_ended_at,
                used_lens_ids_json = excluded.used_lens_ids_json
            """,
            (
                session_id,
                client_id,
                started_at,
                last_active_at,
                1 if is_active else 0,
                last_trigger_location.lat if last_trigger_location else None,
                last_trigger_location.lon if last_trigger_location else None,
                last_trigger_location.short_label if last_trigger_location else None,
                last_trigger_time
                if last_trigger_time is not None
                else (last_trigger_location.timestamp if last_trigger_location else None),
                last_audio_ended_at,
                json.dumps(used_lens_ids),
            ),
        )

    def mark_session_inactive(self, session_id: str) -> None:
        if not session_id:
            return
        with self._lock:
            conn = self._connect()
            conn.execute(
                "UPDATE sessions SET is_active = 0, last_active_at = ? WHERE session_id = ?",
                (time.time(), session_id),
            )
            conn.commit()

    def update_last_audio_ended_at(self, session_id: str, ts: float) -> None:
        if not session_id:
            return
        with self._lock:
            conn = self._connect()
            with conn:
                conn.execute(
                    """
                    UPDATE sessions
                    SET last_audio_ended_at = ?, last_active_at = ?
                    WHERE session_id = ?
                    """,
                    (ts, ts, session_id),
                )

    def save_narration_turn(
        self,
        *,
        session_id: str,
        client_id: str,
        started_at: float,
        last_active_at: float,
        last_trigger_location: TriggerLocation | None,
        last_trigger_time: float | None,
        last_audio_ended_at: float | None,
        used_lens_ids: list[str],
        message: AssistantMessage,
        prompt_text: str,
        llm_latency_ms: int | None = None,
        tts_latency_ms: int | None = None,
        era: str | None = None,
        scale: str | None = None,
        topic_summary: str | None = None,
        entities: list[str] | None = None,
    ) -> None:
        with self._lock:
            conn = self._connect()
            with conn:
                self._upsert_session_in_tx(
                    conn,
                    session_id=session_id,
                    client_id=client_id,
                    started_at=started_at,
                    last_active_at=last_active_at,
                    is_active=True,
                    last_trigger_location=last_trigger_location,
                    last_trigger_time=last_trigger_time,
                    last_audio_ended_at=last_audio_ended_at,
                    used_lens_ids=used_lens_ids,
                )
                self._insert_assistant_message_in_tx(
                    conn,
                    session_id=session_id,
                    client_id=client_id,
                    message=message,
                    prompt_text=prompt_text,
                    llm_latency_ms=llm_latency_ms,
                    tts_latency_ms=tts_latency_ms,
                    era=era,
                    scale=scale,
                    topic_summary=topic_summary,
                    entities=entities or [],
                )

    def save_chat_turn(
        self,
        *,
        session_id: str,
        client_id: str,
        started_at: float,
        last_active_at: float,
        last_trigger_location: TriggerLocation | None,
        last_trigger_time: float | None,
        last_audio_ended_at: float | None,
        used_lens_ids: list[str],
        user_message: UserChatMessage,
        assistant_message: AssistantMessage,
        prompt_text: str,
        lat: float | None = None,
        lon: float | None = None,
        llm_latency_ms: int | None = None,
        tts_latency_ms: int | None = None,
    ) -> None:
        with self._lock:
            conn = self._connect()
            with conn:
                self._upsert_session_in_tx(
                    conn,
                    session_id=session_id,
                    client_id=client_id,
                    started_at=started_at,
                    last_active_at=last_active_at,
                    is_active=True,
                    last_trigger_location=last_trigger_location,
                    last_trigger_time=last_trigger_time,
                    last_audio_ended_at=last_audio_ended_at,
                    used_lens_ids=used_lens_ids,
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO messages (
                        id, session_id, client_id, role, type, ts, lat, lon, short_label,
                        text, prompt_text
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        user_message.id,
                        session_id,
                        client_id,
                        user_message.role,
                        user_message.type,
                        user_message.timestamp,
                        lat,
                        lon,
                        user_message.location_label,
                        user_message.text,
                        prompt_text,
                    ),
                )
                self._insert_assistant_message_in_tx(
                    conn,
                    session_id=session_id,
                    client_id=client_id,
                    message=assistant_message,
                    prompt_text=None,
                    llm_latency_ms=llm_latency_ms,
                    tts_latency_ms=tts_latency_ms,
                    era=None,
                    scale=None,
                    topic_summary=None,
                    entities=[],
                )

    @staticmethod
    def _insert_assistant_message_in_tx(
        conn: sqlite3.Connection,
        *,
        session_id: str,
        client_id: str,
        message: AssistantMessage,
        prompt_text: str | None,
        llm_latency_ms: int | None,
        tts_latency_ms: int | None,
        era: str | None,
        scale: str | None,
        topic_summary: str | None,
        entities: list[str],
    ) -> None:
        loc = message.location
        conn.execute(
            """
            INSERT OR REPLACE INTO messages (
                id, session_id, client_id, role, type, ts,
                lat, lon, short_label, nearby_features_json,
                lens_id, lens_name, era, scale, topic_summary, entities_json,
                reason, model, text, prompt_text, audio_url,
                sources_json, search_queries_json, llm_latency_ms, tts_latency_ms
            ) VALUES (
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?
            )
            """,
            (
                message.id,
                session_id,
                client_id,
                message.role,
                message.type,
                message.timestamp,
                loc.lat if loc else None,
                loc.lon if loc else None,
                loc.short_label if loc else None,
                json.dumps(loc.nearby_features if loc else []),
                message.lens.id if message.lens else None,
                message.lens.name if message.lens else None,
                era,
                scale,
                topic_summary,
                json.dumps(entities),
                message.reason,
                message.model,
                message.text,
                prompt_text,
                message.audio_url,
                json.dumps([s.model_dump() for s in message.sources]),
                json.dumps(message.search_queries),
                llm_latency_ms,
                tts_latency_ms,
            ),
        )

    def load_latest_active_session(
        self, client_id: str, *, audio_dir: Path | None = None
    ) -> PersistedSessionState | None:
        with self._lock:
            conn = self._connect()
            row = conn.execute(
                """
                SELECT * FROM sessions
                WHERE client_id = ? AND is_active = 1
                ORDER BY last_active_at DESC
                LIMIT 1
                """,
                (client_id,),
            ).fetchone()
            if row is None:
                return None

            session_id = str(row["session_id"])
            msg_rows = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY seq ASC",
                (session_id,),
            ).fetchall()

            ui_messages: list[UIMessage] = []
            gemini_contents: list[GeminiContent] = []
            for m in msg_rows:
                role = str(m["role"])
                msg_type = str(m["type"])
                text = str(m["text"])
                prompt_text = m["prompt_text"]

                if role == "user":
                    ui_messages.append(
                        UserChatMessage(
                            id=str(m["id"]),
                            timestamp=float(m["ts"]),
                            location_label=m["short_label"],
                            text=text,
                        )
                    )
                    user_prompt = str(prompt_text) if prompt_text else f"User message: {text}"
                    gemini_contents.append(
                        GeminiContent(role="user", parts=[GeminiPart(text=user_prompt)])
                    )
                else:
                    loc: MessageLocation | None = None
                    if m["lat"] is not None and m["lon"] is not None and m["short_label"]:
                        raw_features = _safe_json_loads(m["nearby_features_json"], [])
                        features = (
                            [str(f) for f in raw_features] if isinstance(raw_features, list) else []
                        )
                        loc = MessageLocation(
                            lat=float(m["lat"]),
                            lon=float(m["lon"]),
                            short_label=str(m["short_label"]),
                            nearby_features=features,
                        )
                    lens: LensSummary | None = None
                    if m["lens_id"] and m["lens_name"]:
                        lens = LensSummary(id=str(m["lens_id"]), name=str(m["lens_name"]))

                    raw_sources = _safe_json_loads(m["sources_json"], [])
                    sources: list[GroundingSource] = []
                    if isinstance(raw_sources, list):
                        for item in raw_sources:
                            if isinstance(item, dict) and item.get("uri"):
                                sources.append(
                                    GroundingSource(
                                        title=str(item.get("title") or item["uri"]),
                                        uri=str(item["uri"]),
                                    )
                                )
                    raw_queries = _safe_json_loads(m["search_queries_json"], [])
                    queries = [str(q) for q in raw_queries] if isinstance(raw_queries, list) else []

                    audio_url: str | None = m["audio_url"]
                    if (
                        audio_url
                        and audio_dir is not None
                        and not (audio_dir / Path(audio_url).name).exists()
                    ):
                        audio_url = None

                    ui_messages.append(
                        AssistantMessage(
                            id=str(m["id"]),
                            type="narration" if msg_type == "narration" else "chat_reply",
                            timestamp=float(m["ts"]),
                            location=loc,
                            lens=lens,
                            reason=m["reason"],
                            model=str(m["model"] or ""),
                            text=text,
                            audio_url=audio_url,
                            sources=sources,
                            search_queries=queries,
                        )
                    )
                    if msg_type == "narration":
                        narration_prompt = (
                            str(prompt_text)
                            if prompt_text
                            else f"[LOCATION TRIGGER — {m['short_label'] or 'Unknown'}]"
                        )
                        gemini_contents.append(
                            GeminiContent(role="user", parts=[GeminiPart(text=narration_prompt)])
                        )
                    gemini_contents.append(
                        GeminiContent(role="model", parts=[GeminiPart(text=text)])
                    )

            last_trig_loc: TriggerLocation | None = None
            if (
                row["last_trigger_lat"] is not None
                and row["last_trigger_lon"] is not None
                and row["last_trigger_label"]
            ):
                last_trig_loc = TriggerLocation(
                    lat=float(row["last_trigger_lat"]),
                    lon=float(row["last_trigger_lon"]),
                    short_label=str(row["last_trigger_label"]),
                    timestamp=float(row["last_trigger_ts"] or row["last_active_at"]),
                )

            raw_lenses = _safe_json_loads(row["used_lens_ids_json"], [])
            used_lens_ids = [str(x) for x in raw_lenses] if isinstance(raw_lenses, list) else []

            return PersistedSessionState(
                session_id=session_id,
                client_id=client_id,
                started_at=float(row["started_at"]),
                last_active_at=float(row["last_active_at"]),
                last_trigger_location=last_trig_loc,
                last_trigger_time=(
                    float(row["last_trigger_ts"]) if row["last_trigger_ts"] is not None else None
                ),
                last_audio_ended_at=(
                    float(row["last_audio_ended_at"])
                    if row["last_audio_ended_at"] is not None
                    else None
                ),
                used_lens_ids=used_lens_ids,
                ui_messages=ui_messages,
                gemini_contents=gemini_contents,
            )

    def record_location_sample(
        self,
        *,
        session_id: str,
        client_id: str,
        started_at: float,
        ts: float,
        lat: float,
        lon: float,
        accuracy: float | None = None,
        heading: float | None = None,
        speed: float | None = None,
        short_label: str | None = None,
    ) -> None:
        with self._lock:
            conn = self._connect()
            with conn:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO sessions (
                        session_id, client_id, started_at, last_active_at, is_active
                    ) VALUES (?, ?, ?, ?, 1)
                    """,
                    (session_id, client_id, started_at, ts),
                )
                conn.execute(
                    "UPDATE sessions SET last_active_at = ? WHERE session_id = ?",
                    (ts, session_id),
                )
                conn.execute(
                    """
                    INSERT INTO location_samples (
                        session_id, client_id, ts, lat, lon, accuracy, heading, speed, short_label
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (session_id, client_id, ts, lat, lon, accuracy, heading, speed, short_label),
                )

    def record_event(
        self,
        *,
        source: EventSource,
        level: EventLevel,
        event_type: BackendEventType | str,
        payload: dict[str, Any] | None = None,
        session_id: str | None = None,
        client_id: str | None = None,
        ts: float | None = None,
    ) -> None:
        self.record_events_batch(
            [
                {
                    "ts": ts if ts is not None else time.time(),
                    "source": source,
                    "level": level,
                    "session_id": session_id,
                    "client_id": client_id,
                    "event_type": event_type,
                    "payload": payload or {},
                }
            ]
        )

    def record_events_batch(self, items: list[dict[str, Any]]) -> int:
        if not items:
            return 0
        now = time.time()
        rows: list[tuple[float, str, str, str | None, str | None, str, str]] = []
        for item in items:
            ts = _normalize_event_ts(item.get("ts"), now)
            payload_json = _serialize_event_payload(item.get("payload"))
            rows.append(
                (
                    ts,
                    str(item.get("source") or EventSource.BACKEND),
                    str(item.get("level") or EventLevel.INFO),
                    str(item["session_id"]) if item.get("session_id") else None,
                    str(item["client_id"]) if item.get("client_id") else None,
                    str(item.get("event_type") or "unknown"),
                    payload_json,
                )
            )
        with self._lock:
            conn = self._connect()
            with conn:
                conn.executemany(
                    """
                    INSERT INTO events (
                        ts, source, level, session_id, client_id, event_type, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
        return len(rows)

    def prune_old_events(self, max_age_days: int = 30) -> int:
        cutoff = time.time() - max_age_days * 86400.0
        with self._lock:
            conn = self._connect()
            with conn:
                cur = conn.execute("DELETE FROM events WHERE ts < ?", (cutoff,))
                return cur.rowcount

    def load_config_override(self) -> tuple[AppConfig, float] | None:
        with self._lock:
            conn = self._connect()
            row = conn.execute(
                "SELECT value_json, updated_at FROM config_overrides WHERE key = 'app_config'"
            ).fetchone()
            if row is None:
                return None
            data = _safe_json_loads(row["value_json"], None)
            if not isinstance(data, dict):
                return None
            return AppConfig.model_validate(data), float(row["updated_at"])

    def save_config_override(self, config: AppConfig, updated_at: float | None = None) -> float:
        ts = updated_at if updated_at is not None else time.time()
        with self._lock:
            conn = self._connect()
            with conn:
                conn.execute(
                    """
                    INSERT INTO config_overrides (key, value_json, updated_at)
                    VALUES ('app_config', ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        value_json = excluded.value_json,
                        updated_at = excluded.updated_at
                    """,
                    (config.model_dump_json(), ts),
                )
        return ts

    def search_messages(
        self, query: str, *, client_id: str | None = None, limit: int = 20
    ) -> list[dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            if client_id:
                rows = conn.execute(
                    """
                    SELECT m.id, m.session_id, m.client_id, m.role, m.type, m.ts,
                           m.short_label, m.lens_name, m.text
                    FROM messages_fts f
                    JOIN messages m ON m.seq = f.rowid
                    WHERE messages_fts MATCH ? AND m.client_id = ?
                    ORDER BY m.ts DESC
                    LIMIT ?
                    """,
                    (query, client_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT m.id, m.session_id, m.client_id, m.role, m.type, m.ts,
                           m.short_label, m.lens_name, m.text
                    FROM messages_fts f
                    JOIN messages m ON m.seq = f.rowid
                    WHERE messages_fts MATCH ?
                    ORDER BY m.ts DESC
                    LIMIT ?
                    """,
                    (query, limit),
                ).fetchall()
            return [dict(r) for r in rows]

    def get_recent_events(
        self,
        *,
        client_id: str | None = None,
        session_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if client_id:
            clauses.append("client_id = ?")
            params.append(client_id)
        if session_id:
            clauses.append("session_id = ?")
            params.append(session_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        with self._lock:
            conn = self._connect()
            rows = conn.execute(
                f"SELECT * FROM events {where} ORDER BY ts DESC, id DESC LIMIT ?",
                params,
            ).fetchall()
            result: list[dict[str, Any]] = []
            for r in rows:
                d = dict(r)
                d["payload"] = _safe_json_loads(d.pop("payload_json", "{}"), {})
                result.append(d)
            return result
