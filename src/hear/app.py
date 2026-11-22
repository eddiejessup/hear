from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.requests import HTTPConnection
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from hear.auth import AuthSettings, EnvReader, verify_token
from hear.config import ConfigManager
from hear.db import EventRecord, HearDB
from hear.debug import RuntimeDebugProvider
from hear.gemini import GeminiClient
from hear.geo import GeoService
from hear.logging_utils import (
    EventLogger,
    bind_event_context,
    configure_logging,
    set_event_session_id,
)
from hear.models.api import (
    ChatRequest,
    ChatResponse,
    ClientEventsRequest,
    ClientEventsResponse,
    NarrateRequest,
    NarrateResponse,
    PingRequest,
    PingResponse,
    ResetSessionResponse,
    StateResponse,
    UpdateConfigResponse,
)
from hear.models.config import AppConfig
from hear.models.events import BackendEventType, EventLevel, EventSource
from hear.models.session import CurrentLocation
from hear.paths import PROJECT_ROOT, AppPaths
from hear.session import SessionBusyError, SessionManager, WalkSession, sanitize_client_id

logger = logging.getLogger(__name__)


@dataclass
class AppContainer:
    """Explicit runtime dependencies for a single Hear FastAPI application instance."""

    paths: AppPaths
    env: EnvReader
    auth: AuthSettings
    db: HearDB
    events: EventLogger
    config_manager: ConfigManager
    geo: GeoService
    gemini: GeminiClient
    session_manager: SessionManager
    debug: RuntimeDebugProvider


def build_container(
    *,
    paths: AppPaths | None = None,
    data_dir: Path | str | None = None,
    config_path: Path | str | None = None,
    env: EnvReader | None = None,
    auth_token: str | None = None,
    allow_unauthenticated: bool | None = None,
    gemini_api_key: str | None = None,
    geo: GeoService | None = None,
    gemini: GeminiClient | None = None,
) -> AppContainer:
    if paths is not None:
        resolved_paths = paths
        resolved_env = env if env is not None else EnvReader(dotenv_path=resolved_paths.dotenv_path)
    else:
        resolved_env = env if env is not None else EnvReader(dotenv_path=PROJECT_ROOT / ".env")
        resolved_paths = AppPaths.from_env(
            data_dir=data_dir,
            config_path=config_path,
            env=resolved_env,
        )
    resolved_paths.ensure_dirs()
    auth = AuthSettings(
        resolved_env,
        auth_token=auth_token,
        allow_unauthenticated=allow_unauthenticated,
    )
    store = HearDB(resolved_paths.db_path)
    events = EventLogger(store=store)
    cfg_mgr = ConfigManager(resolved_paths.config_path, store=store)
    geo_svc = geo or GeoService(event_logger=events)
    gemini_client = gemini or GeminiClient(
        audio_dir=resolved_paths.audio_dir,
        api_key=gemini_api_key,
        env=resolved_env,
        event_logger=events,
    )
    sessions = SessionManager(
        store=store,
        audio_dir=resolved_paths.audio_dir,
        gemini=gemini_client,
        event_logger=events,
    )
    debug = RuntimeDebugProvider(resolved_paths, resolved_env)
    return AppContainer(
        paths=resolved_paths,
        env=resolved_env,
        auth=auth,
        db=store,
        events=events,
        config_manager=cfg_mgr,
        geo=geo_svc,
        gemini=gemini_client,
        session_manager=sessions,
        debug=debug,
    )


def _extract_client_id(conn: HTTPConnection) -> str:
    raw_id = conn.headers.get("x-hear-client-id")
    if not raw_id:
        cookie_id = conn.cookies.get("hear_client_id")
        raw_id = unquote(cookie_id) if cookie_id else None
    return sanitize_client_id(raw_id)


@asynccontextmanager
async def _acquire_generation_turn(session: WalkSession) -> AsyncGenerator[None]:
    try:
        async with session.generation_turn():
            yield
    except SessionBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


class _AuthMiddleware:
    """Pure ASGI auth middleware so request ContextVars stay in the same task."""

    def __init__(self, app: ASGIApp, auth: AuthSettings) -> None:
        self.app = app
        self.auth = auth

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        conn = HTTPConnection(scope)
        if not conn.url.path.startswith("/api/"):
            await self.app(scope, receive, send)
            return

        expected = self.auth.expected_token()
        if not expected:
            if self.auth.is_unauthenticated_allowed():
                await self.app(scope, receive, send)
                return
            resp = JSONResponse(
                {
                    "detail": (
                        "Server authentication is not configured. "
                        "Set HEAR_AUTH_TOKEN (or HEAR_ALLOW_UNAUTHENTICATED=1 for local dev)."
                    )
                },
                status_code=503,
            )
            await resp(scope, receive, send)
            return

        provided = conn.headers.get("x-hear-token")
        if not provided:
            cookie_token = conn.cookies.get("hear_token")
            provided = unquote(cookie_token) if cookie_token else ""
        if not verify_token(provided, expected):
            resp = JSONResponse({"detail": "Unauthorized"}, status_code=401)
            await resp(scope, receive, send)
            return

        await self.app(scope, receive, send)


class _RequestLoggingMiddleware:
    """Pure ASGI request logger that shares ContextVar state with downstream handlers."""

    def __init__(self, app: ASGIApp, events: EventLogger) -> None:
        self.app = app
        self.events = events

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        t0 = time.monotonic()
        conn = HTTPConnection(scope)
        path = conn.url.path
        method = str(scope.get("method", ""))
        is_api = path.startswith("/api/")
        client_id = _extract_client_id(conn) if is_api else None
        status_code: int | None = None

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
            await send(message)

        with bind_event_context(client_id=client_id):
            try:
                await self.app(scope, receive, send_wrapper)
            except Exception as exc:
                latency_ms = int((time.monotonic() - t0) * 1000)
                await self.events.log_async(
                    BackendEventType.HTTP_UNHANDLED_ERROR,
                    level=EventLevel.ERROR,
                    method=method,
                    path=path,
                    latency_ms=latency_ms,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
                raise

            if is_api and status_code is not None:
                latency_ms = int((time.monotonic() - t0) * 1000)
                # Persist authenticated API errors to SQLite; keep 401/503 probes and 2xx in stdout
                if status_code >= 400:
                    await self.events.log_async(
                        BackendEventType.HTTP_ERROR,
                        level=EventLevel.ERROR if status_code >= 500 else EventLevel.WARN,
                        persist=status_code not in {401, 503},
                        method=method,
                        path=path,
                        status_code=status_code,
                        latency_ms=latency_ms,
                    )
                elif path != "/api/events":
                    self.events.log(
                        BackendEventType.HTTP_REQUEST,
                        level=EventLevel.INFO,
                        persist=False,
                        method=method,
                        path=path,
                        status_code=status_code,
                        latency_ms=latency_ms,
                    )


def create_app(
    *,
    container: AppContainer | None = None,
    paths: AppPaths | None = None,
    data_dir: Path | str | None = None,
    config_path: Path | str | None = None,
    env: EnvReader | None = None,
    auth_token: str | None = None,
    allow_unauthenticated: bool | None = None,
    gemini_api_key: str | None = None,
    configure_logs: bool = True,
) -> FastAPI:
    if configure_logs:
        configure_logging()

    c = container or build_container(
        paths=paths,
        data_dir=data_dir,
        config_path=config_path,
        env=env,
        auth_token=auth_token,
        allow_unauthenticated=allow_unauthenticated,
        gemini_api_key=gemini_api_key,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
        await asyncio.to_thread(c.db.init)
        await asyncio.to_thread(c.config_manager.get)
        pruned = await asyncio.to_thread(c.db.prune_old_events, 30)
        await c.events.log_async(
            BackendEventType.SERVER_STARTED,
            persist=True,
            pruned_events=pruned,
            db_path=str(c.db.path),
        )
        try:
            yield
        finally:
            try:
                await c.geo.close()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to close GeoService during shutdown: %s", exc)
            try:
                await c.gemini.close()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to close GeminiClient during shutdown: %s", exc)
            try:
                await asyncio.to_thread(c.db.close)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to close HearDB during shutdown: %s", exc)

    app = FastAPI(title="Hear — Historical Walking Companion", lifespan=lifespan)
    app.state.container = c
    app.mount("/static", StaticFiles(directory=c.paths.static_dir), name="static")

    def _get_session(request: Request) -> WalkSession:
        session = c.session_manager.get(_extract_client_id(request))
        set_event_session_id(session.session_id)
        return session

    app.add_middleware(_AuthMiddleware, auth=c.auth)
    app.add_middleware(_RequestLoggingMiddleware, events=c.events)

    @app.get("/", response_model=None)
    async def index() -> Response:
        index_file = c.paths.static_dir / "index.html"
        if not index_file.exists():
            return HTMLResponse("<h1>static/index.html not found</h1>", status_code=404)
        return FileResponse(index_file, media_type="text/html")

    @app.get("/api/state")
    async def get_state(request: Request) -> StateResponse:
        session = _get_session(request)
        cfg = c.config_manager.get()
        return StateResponse(
            session_id=session.session_id,
            has_api_key=c.gemini.has_api_key(),
            current_location=session.current_location,
            last_trigger_location=session.last_trigger_location,
            messages=session.ui_messages,
            used_lens_ids=session.used_lens_ids,
            trigger_status=session.compute_trigger_status(cfg),
            config=cfg,
            debug_info=c.debug.get_info(),
        )

    @app.post("/api/ping")
    async def ping_location(req: PingRequest, request: Request) -> PingResponse:
        session = _get_session(request)
        cfg = c.config_manager.get()

        audio_ended_updated = session.update_audio_state(
            lat=req.lat,
            lon=req.lon,
            is_playing_audio=req.is_playing_audio,
            is_paused_audio=req.is_paused_audio,
            audio_just_ended=req.audio_just_ended,
        )
        if audio_ended_updated:
            await session.persist_last_audio_ended_at()

        geo = await c.geo.reverse_geocode(req.lat, req.lon)
        cur_loc = CurrentLocation(
            lat=req.lat,
            lon=req.lon,
            accuracy=req.accuracy,
            heading=req.heading,
            speed=req.speed,
            short_label=geo.short_label,
            full_address=geo.full_address,
            timestamp=time.time(),
        )
        session.current_location = cur_loc
        await session.record_location_sample_if_needed(cur_loc)

        status = session.compute_trigger_status(cfg)
        if req.audio_just_ended:
            await c.events.log_async(
                BackendEventType.AUDIO_ENDED_PING,
                short_label=cur_loc.short_label,
            )
        if status.should_trigger:
            await c.events.log_async(
                BackendEventType.AUTO_TRIGGER_READY,
                reason=status.trigger_reason,
                distance_moved_m=status.distance_moved_m,
                silence_elapsed_s=status.silence_elapsed_s,
                short_label=cur_loc.short_label,
            )

        return PingResponse(
            location=cur_loc,
            trigger_status=status,
        )

    @app.post("/api/narrate")
    async def trigger_narration(req: NarrateRequest, request: Request) -> NarrateResponse:
        session = _get_session(request)
        async with _acquire_generation_turn(session):
            cfg = c.config_manager.get()
            lat, lon, heading = session.resolve_coords(req.lat, req.lon, req.heading)

            if lat is None or lon is None:
                raise HTTPException(status_code=400, detail="No GPS coordinates available yet.")

            geo = await c.geo.reverse_geocode(lat, lon)
            msg_record = await session.execute_narration_turn(
                config=cfg,
                lat=lat,
                lon=lon,
                heading=heading,
                geo=geo,
                requested_lens_id=req.lens_id,
                custom_focus=req.custom_focus,
                reason=req.reason,
            )
            return NarrateResponse(
                message=msg_record,
                trigger_status=session.compute_trigger_status(cfg),
            )

    @app.post("/api/chat")
    async def steer_chat(req: ChatRequest, request: Request) -> ChatResponse:
        cleaned_message = req.message.strip()
        if not cleaned_message:
            raise HTTPException(status_code=400, detail="Empty message.")

        session = _get_session(request)
        async with _acquire_generation_turn(session):
            cfg = c.config_manager.get()
            lat, lon, _ = session.resolve_coords(req.lat, req.lon)

            short_label: str | None = None
            if lat is not None and lon is not None:
                geo = await c.geo.reverse_geocode(lat, lon)
                short_label = geo.short_label

            user_msg_ui, assistant_msg_ui = await session.execute_chat_turn(
                config=cfg,
                cleaned_message=cleaned_message,
                lat=lat,
                lon=lon,
                short_label=short_label,
            )
            return ChatResponse(
                user_message=user_msg_ui,
                assistant_message=assistant_msg_ui,
                trigger_status=session.compute_trigger_status(cfg),
            )

    @app.post("/api/events")
    async def ingest_client_events(
        req: ClientEventsRequest, request: Request
    ) -> ClientEventsResponse:
        session = _get_session(request)
        if not req.events:
            return ClientEventsResponse(accepted=0)

        items = [
            EventRecord(
                ts=ev.ts,
                source=EventSource.FRONTEND,
                level=ev.level,
                session_id=ev.session_id or session.session_id,
                client_id=session.client_id,
                event_type=ev.event_type,
                payload=ev.payload,
            )
            for ev in req.events
        ]
        accepted = await asyncio.to_thread(c.db.record_events_batch, items)
        return ClientEventsResponse(accepted=accepted)

    @app.get("/api/audio/{filename}")
    async def get_audio(filename: str) -> FileResponse:
        safe_name = Path(filename).name
        file_path = c.paths.audio_dir / safe_name
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Audio file not found")
        return FileResponse(
            file_path,
            media_type="audio/wav",
            headers={"Cache-Control": "private, max-age=86400, immutable"},
        )

    @app.get("/api/config")
    async def get_config() -> AppConfig:
        return c.config_manager.get()

    @app.put("/api/config")
    async def update_config(new_config: AppConfig, request: Request) -> UpdateConfigResponse:
        session = _get_session(request)
        saved = await asyncio.to_thread(c.config_manager.save, new_config)
        await c.events.log_async(
            BackendEventType.CONFIG_UPDATED,
            model=saved.gemini.model,
            auto_trigger_enabled=saved.triggers.auto_trigger_enabled,
            lens_count=len(saved.lenses),
        )
        return UpdateConfigResponse(
            config=saved,
            trigger_status=session.compute_trigger_status(saved),
        )

    @app.post("/api/session/reset")
    async def reset_session(request: Request) -> ResetSessionResponse:
        session = _get_session(request)
        async with session.lock:
            previous_id = await session.reset_async()
            set_event_session_id(session.session_id)
            cfg = c.config_manager.get()
            await c.events.log_async(
                BackendEventType.SESSION_RESET,
                previous_session_id=previous_id,
            )
            return ResetSessionResponse(
                session_id=session.session_id,
                trigger_status=session.compute_trigger_status(cfg),
            )

    return app


def main() -> None:
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    reload_enabled = os.environ.get("HEAR_NO_RELOAD", "").lower() not in {"1", "true"}
    uvicorn.run(
        "hear.app:create_app",
        host=host,
        port=port,
        reload=reload_enabled,
        factory=True,
    )
