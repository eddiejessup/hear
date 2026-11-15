import os
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from hear.auth import get_expected_auth_token, is_unauthenticated_allowed, verify_token
from hear.config import config_manager
from hear.debug import get_debug_info
from hear.gemini import close_gemini_http_client, get_api_key
from hear.geo import close_geo_http_client, reverse_geocode_and_context
from hear.models.api import (
    ChatRequest,
    ChatResponse,
    NarrateRequest,
    NarrateResponse,
    PingRequest,
    PingResponse,
    ResetSessionResponse,
    StateResponse,
    UpdateConfigResponse,
)
from hear.models.config import AppConfig
from hear.models.session import CurrentLocation
from hear.paths import AUDIO_DIR, STATIC_DIR, ensure_dirs
from hear.session import WalkSession, session_manager

ensure_dirs()


def _get_session(request: Request) -> WalkSession:
    raw_id = request.headers.get("x-hear-client-id") or request.cookies.get("hear_client_id")
    return session_manager.get(raw_id)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
    yield
    await close_geo_http_client()
    await close_gemini_http_client()


app = FastAPI(title="Hear — Historical Walking Companion", lifespan=lifespan)


@app.middleware("http")
async def auth_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    if not request.url.path.startswith("/api/"):
        return await call_next(request)

    expected = get_expected_auth_token()
    if not expected:
        if is_unauthenticated_allowed():
            return await call_next(request)
        return JSONResponse(
            {
                "detail": (
                    "Server authentication is not configured. "
                    "Set HEAR_AUTH_TOKEN (or HEAR_ALLOW_UNAUTHENTICATED=1 for local dev)."
                )
            },
            status_code=503,
        )

    provided = request.headers.get("x-hear-token") or request.cookies.get("hear_token") or ""
    if not verify_token(provided, expected):
        return JSONResponse({"detail": "Unauthorized"}, status_code=401)

    return await call_next(request)


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h1>static/index.html not found</h1>", status_code=404)
    return HTMLResponse(index_file.read_text(encoding="utf-8"))


@app.get("/api/state")
async def get_state(request: Request) -> StateResponse:
    session = _get_session(request)
    cfg = config_manager.get()
    has_api_key = True
    try:
        get_api_key()
    except HTTPException:
        has_api_key = False

    return StateResponse(
        session_id=session.session_id,
        has_api_key=has_api_key,
        current_location=session.current_location,
        last_trigger_location=session.last_trigger_location,
        messages=session.ui_messages,
        used_lens_ids=session.used_lens_ids,
        trigger_status=session.compute_trigger_status(cfg),
        config=cfg,
        debug_info=get_debug_info(),
    )


@app.post("/api/ping")
async def ping_location(req: PingRequest, request: Request) -> PingResponse:
    session = _get_session(request)
    cfg = config_manager.get()

    if req.audio_just_ended or (session.is_playing_audio and not req.is_playing_audio):
        session.last_audio_ended_at = time.time()
    session.is_playing_audio = req.is_playing_audio

    prefetch_overpass = session.should_prefetch_overpass(cfg, req.lat, req.lon)
    geo = await reverse_geocode_and_context(req.lat, req.lon, include_overpass=prefetch_overpass)
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

    status = session.compute_trigger_status(cfg)
    return PingResponse(
        location=cur_loc,
        trigger_status=status,
    )


@app.post("/api/narrate")
async def trigger_narration(req: NarrateRequest, request: Request) -> NarrateResponse:
    session = _get_session(request)
    if session.is_generating or session.lock.locked():
        raise HTTPException(
            status_code=409,
            detail="Narration or response generation is already in progress.",
        )

    async with session.lock:
        cfg = config_manager.get()
        lat, lon, heading = session.resolve_coords(req.lat, req.lon, req.heading)

        if lat is None or lon is None:
            raise HTTPException(status_code=400, detail="No GPS coordinates available yet.")

        session.is_generating = True
        try:
            geo = await reverse_geocode_and_context(lat, lon, include_overpass=True)
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
        finally:
            session.is_generating = False


@app.post("/api/chat")
async def steer_chat(req: ChatRequest, request: Request) -> ChatResponse:
    cleaned_message = req.message.strip()
    if not cleaned_message:
        raise HTTPException(status_code=400, detail="Empty message.")

    session = _get_session(request)
    async with session.lock:
        cfg = config_manager.get()
        lat, lon, _ = session.resolve_coords(req.lat, req.lon)

        session.is_generating = True
        try:
            short_label: str | None = None
            if lat is not None and lon is not None:
                geo = await reverse_geocode_and_context(lat, lon, include_overpass=False)
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
        finally:
            session.is_generating = False


@app.get("/api/audio/{filename}")
async def get_audio(filename: str) -> FileResponse:
    safe_name = Path(filename).name
    file_path = AUDIO_DIR / safe_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(
        file_path,
        media_type="audio/wav",
        headers={"Cache-Control": "private, max-age=86400, immutable"},
    )


@app.get("/api/config")
async def get_config() -> AppConfig:
    return config_manager.get()


@app.put("/api/config")
async def update_config(new_config: AppConfig, request: Request) -> UpdateConfigResponse:
    session = _get_session(request)
    saved = config_manager.save(new_config)
    return UpdateConfigResponse(
        config=saved,
        trigger_status=session.compute_trigger_status(saved),
    )


@app.post("/api/session/reset")
async def reset_session(request: Request) -> ResetSessionResponse:
    session = _get_session(request)
    async with session.lock:
        session.reset()
        cfg = config_manager.get()
        return ResetSessionResponse(
            session_id=session.session_id,
            trigger_status=session.compute_trigger_status(cfg),
        )


def main() -> None:
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    reload_enabled = os.environ.get("HEAR_NO_RELOAD", "").lower() not in {"1", "true"}
    uvicorn.run("hear.app:app", host=host, port=port, reload=reload_enabled)
