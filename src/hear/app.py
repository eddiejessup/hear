import time
import uuid
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

from hear.config import config_manager
from hear.gemini import call_gemini_text, call_gemini_tts, get_api_key
from hear.geo import compass_bearing_label, reverse_geocode_and_context
from hear.models import ChatRequest, NarrateRequest, PingRequest
from hear.paths import AUDIO_DIR, STATIC_DIR, ensure_dirs
from hear.session import session

ensure_dirs()

app = FastAPI(title="Hear — Historical Walking Companion")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h1>static/index.html not found</h1>", status_code=404)
    return HTMLResponse(index_file.read_text(encoding="utf-8"))


@app.get("/api/state")
async def get_state() -> dict[str, Any]:
    cfg = config_manager.get()
    has_api_key = True
    try:
        get_api_key()
    except HTTPException:
        has_api_key = False

    return {
        "session_id": session.session_id,
        "has_api_key": has_api_key,
        "current_location": session.current_location,
        "last_trigger_location": session.last_trigger_location,
        "messages": session.ui_messages,
        "used_lens_ids": session.used_lens_ids,
        "trigger_status": session.compute_trigger_status(cfg),
        "config": cfg,
    }


@app.post("/api/ping")
async def ping_location(req: PingRequest) -> dict[str, Any]:
    cfg = config_manager.get()

    if req.audio_just_ended or (session.is_playing_audio and not req.is_playing_audio):
        session.last_audio_ended_at = time.time()
    session.is_playing_audio = req.is_playing_audio

    geo = await reverse_geocode_and_context(req.lat, req.lon, include_overpass=False)
    session.current_location = {
        "lat": req.lat,
        "lon": req.lon,
        "accuracy": req.accuracy,
        "heading": req.heading,
        "speed": req.speed,
        "short_label": geo["short_label"],
        "full_address": geo["full_address"],
        "timestamp": time.time(),
    }

    status = session.compute_trigger_status(cfg)
    return {
        "location": session.current_location,
        "trigger_status": status,
    }


@app.post("/api/narrate")
async def trigger_narration(req: NarrateRequest) -> dict[str, Any]:
    async with session.lock:
        cfg = config_manager.get()
        cur_loc = session.current_location or {}
        lat = req.lat if req.lat is not None else cur_loc.get("lat")
        lon = req.lon if req.lon is not None else cur_loc.get("lon")
        heading = req.heading if req.heading is not None else cur_loc.get("heading")

        if lat is None or lon is None:
            raise HTTPException(status_code=400, detail="No GPS coordinates available yet.")

        lat_f = float(lat)
        lon_f = float(lon)
        heading_f = float(heading) if heading is not None else None

        session.is_generating = True
        try:
            geo = await reverse_geocode_and_context(lat_f, lon_f, include_overpass=True)
            lens = session.sample_lens(cfg, requested_lens_id=req.lens_id)

            compass = compass_bearing_label(heading_f)
            nearby_str = (
                "\nNearby mapped features (within 150m): " + "; ".join(geo["nearby_features"])
                if geo["nearby_features"]
                else ""
            )
            heading_str = f" | Heading: {compass}" if compass else ""
            focus_str = (
                f"\nAdditional user request for this stop: {req.custom_focus}"
                if req.custom_focus
                else ""
            )

            lens_name = lens.get("name", "Historical Lens")
            lens_prompt = lens.get("prompt", "")

            location_prompt = (
                f"[LOCATION TRIGGER — {geo['short_label']}]\n"
                f"Coordinates: {lat_f:.5f}, {lon_f:.5f}{heading_str}\n"
                f"Full address context: {geo['full_address']}"
                f"{nearby_str}\n\n"
                f"Selected Lens ({lens_name}): {lens_prompt}"
                f"{focus_str}\n\n"
                "Narrate an insightful historical segment about this location "
                "through the selected lens."
            )

            gemini_cfg = cfg.get("gemini") or {}
            model_name = str(gemini_cfg.get("model", "gemini-3.8-flash"))
            temperature = float(gemini_cfg.get("temperature", 0.75))
            enable_search = bool(gemini_cfg.get("enable_google_search", True))
            system_prompt = str(cfg.get("system_prompt", ""))

            trial_contents = [
                *session.gemini_contents,
                {"role": "user", "parts": [{"text": location_prompt}]},
            ]

            llm_result = await call_gemini_text(
                contents=trial_contents,
                system_prompt=system_prompt,
                model=model_name,
                temperature=temperature,
                enable_google_search=enable_search,
            )

            reply_text = llm_result["text"]

            audio_url: str | None = None
            if gemini_cfg.get("tts_enabled", True):
                audio_url = await call_gemini_tts(
                    text=reply_text,
                    tts_model=str(gemini_cfg.get("tts_model", "gemini-3.8-flash-tts")),
                    voice_name=str(gemini_cfg.get("tts_voice", "Charon")),
                    style_prompt=str(gemini_cfg.get("tts_style_prompt", "")),
                )

            now = time.time()
            session.gemini_contents = [
                *trial_contents,
                {"role": "model", "parts": [{"text": reply_text}]},
            ]
            session.last_trigger_location = {
                "lat": lat_f,
                "lon": lon_f,
                "short_label": geo["short_label"],
                "timestamp": now,
            }
            session.last_trigger_time = now
            session.last_audio_ended_at = now

            msg_record: dict[str, Any] = {
                "id": uuid.uuid4().hex[:10],
                "role": "assistant",
                "type": "narration",
                "timestamp": now,
                "location": {
                    "lat": lat_f,
                    "lon": lon_f,
                    "short_label": geo["short_label"],
                    "nearby_features": geo["nearby_features"],
                },
                "lens": {"id": lens.get("id"), "name": lens_name},
                "reason": req.reason,
                "model": model_name,
                "text": reply_text,
                "audio_url": audio_url,
                "sources": llm_result["sources"],
                "search_queries": llm_result["search_queries"],
            }
            session.ui_messages.append(msg_record)
            session.persist_transcript()

            return {
                "message": msg_record,
                "trigger_status": session.compute_trigger_status(cfg),
            }
        finally:
            session.is_generating = False


@app.post("/api/chat")
async def steer_chat(req: ChatRequest) -> dict[str, Any]:
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Empty message.")

    async with session.lock:
        cfg = config_manager.get()
        cur_loc = session.current_location or {}
        lat = req.lat if req.lat is not None else cur_loc.get("lat")
        lon = req.lon if req.lon is not None else cur_loc.get("lon")

        session.is_generating = True
        try:
            loc_prefix = ""
            short_label: str | None = None
            if lat is not None and lon is not None:
                lat_f = float(lat)
                lon_f = float(lon)
                geo = await reverse_geocode_and_context(lat_f, lon_f, include_overpass=False)
                short_label = geo["short_label"]
                loc_prefix = f"[User current location: {short_label} ({lat_f:.5f}, {lon_f:.5f})]\n"

            user_msg_ui: dict[str, Any] = {
                "id": uuid.uuid4().hex[:10],
                "role": "user",
                "type": "chat",
                "timestamp": time.time(),
                "location_label": short_label,
                "text": req.message.strip(),
            }
            session.ui_messages.append(user_msg_ui)

            gemini_cfg = cfg.get("gemini") or {}
            model_name = str(gemini_cfg.get("model", "gemini-3.8-flash"))
            temperature = float(gemini_cfg.get("temperature", 0.75))
            enable_search = bool(gemini_cfg.get("enable_google_search", True))
            system_prompt = str(cfg.get("system_prompt", ""))

            user_prompt_text = f"{loc_prefix}User message: {req.message.strip()}"
            trial_contents = [
                *session.gemini_contents,
                {"role": "user", "parts": [{"text": user_prompt_text}]},
            ]

            llm_result = await call_gemini_text(
                contents=trial_contents,
                system_prompt=system_prompt,
                model=model_name,
                temperature=temperature,
                enable_google_search=enable_search,
            )

            reply_text = llm_result["text"]

            audio_url: str | None = None
            if gemini_cfg.get("tts_enabled", True):
                audio_url = await call_gemini_tts(
                    text=reply_text,
                    tts_model=str(gemini_cfg.get("tts_model", "gemini-3.8-flash-tts")),
                    voice_name=str(gemini_cfg.get("tts_voice", "Charon")),
                    style_prompt=str(gemini_cfg.get("tts_style_prompt", "")),
                )

            now = time.time()
            session.gemini_contents = [
                *trial_contents,
                {"role": "model", "parts": [{"text": reply_text}]},
            ]
            session.last_audio_ended_at = now

            assistant_msg_ui: dict[str, Any] = {
                "id": uuid.uuid4().hex[:10],
                "role": "assistant",
                "type": "chat_reply",
                "timestamp": now,
                "location": (
                    {"lat": float(lat), "lon": float(lon), "short_label": short_label}
                    if short_label and lat is not None and lon is not None
                    else None
                ),
                "model": model_name,
                "text": reply_text,
                "audio_url": audio_url,
                "sources": llm_result["sources"],
                "search_queries": llm_result["search_queries"],
            }
            session.ui_messages.append(assistant_msg_ui)
            session.persist_transcript()

            return {
                "user_message": user_msg_ui,
                "assistant_message": assistant_msg_ui,
                "trigger_status": session.compute_trigger_status(cfg),
            }
        finally:
            session.is_generating = False


@app.get("/api/audio/{filename}")
async def get_audio(filename: str) -> FileResponse:
    safe_name = Path(filename).name
    file_path = AUDIO_DIR / safe_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(file_path, media_type="audio/wav")


@app.get("/api/config")
async def get_config() -> dict[str, Any]:
    return config_manager.get()


@app.put("/api/config")
async def update_config(new_config: dict[str, Any]) -> dict[str, Any]:
    saved = config_manager.save(new_config)
    return {
        "config": saved,
        "trigger_status": session.compute_trigger_status(saved),
    }


@app.post("/api/session/reset")
async def reset_session() -> dict[str, Any]:
    async with session.lock:
        session.reset()
        cfg = config_manager.get()
        return {
            "session_id": session.session_id,
            "trigger_status": session.compute_trigger_status(cfg),
        }


def main() -> None:
    uvicorn.run("hear.app:app", host="127.0.0.1", port=8000, reload=True)
