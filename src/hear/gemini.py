import base64
import io
import os
import re
import time
import uuid
import wave
from typing import Any

import httpx
from fastapi import HTTPException

from hear.models import GeminiTextResult, GroundingSource
from hear.paths import AUDIO_DIR, PROJECT_ROOT

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"


def get_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
    if not key:
        env_file = PROJECT_ROOT / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped.startswith("GEMINI_API_KEY=") or stripped.startswith("GOOGLE_API_KEY="):
                    key = stripped.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not key:
        raise HTTPException(
            status_code=400,
            detail=(
                "GEMINI_API_KEY is not set. "
                "Export GEMINI_API_KEY in your shell or add it to a .env file."
            ),
        )
    return key


def pcm_to_wav_bytes(
    pcm_bytes: bytes, sample_rate: int = 24000, channels: int = 1, sampwidth: int = 2
) -> bytes:
    if pcm_bytes.startswith(b"RIFF"):
        return pcm_bytes
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


async def call_gemini_text(
    contents: list[dict[str, Any]],
    system_prompt: str,
    model: str,
    temperature: float = 0.75,
    enable_google_search: bool = True,
) -> GeminiTextResult:
    """Call Gemini generateContent REST API with multi-turn contents and Google Search grounding."""
    api_key = get_api_key()
    url = f"{GEMINI_API_BASE}/models/{model}:generateContent"

    payload: dict[str, Any] = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
        },
    }
    if system_prompt:
        payload["system_instruction"] = {"parts": [{"text": system_prompt}]}
    if enable_google_search:
        payload["tools"] = [{"google_search": {}}]

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            url,
            headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
            json=payload,
        )

    if resp.status_code != 200:
        err_text = resp.text
        try:
            err_json = resp.json()
            err_text = str(err_json.get("error", {}).get("message", err_text))
        except ValueError:
            pass
        raise HTTPException(
            status_code=resp.status_code, detail=f"Gemini API error ({model}): {err_text}"
        )

    data = resp.json()
    candidates = data.get("candidates") or []
    if not candidates:
        raise HTTPException(status_code=500, detail="Gemini returned no candidates.")

    candidate = candidates[0]
    parts = (candidate.get("content") or {}).get("parts") or []
    text_chunks = [
        str(p.get("text", "")) for p in parts if "text" in p and not p.get("thought", False)
    ]
    if not text_chunks:
        text_chunks = [str(p.get("text", "")) for p in parts if "text" in p]
    full_text = "\n".join(t for t in text_chunks if t).strip()

    grounding = candidate.get("groundingMetadata") or {}
    raw_queries = grounding.get("webSearchQueries") or []
    search_queries = [str(q) for q in raw_queries if q]

    sources: list[GroundingSource] = []
    for chunk in grounding.get("groundingChunks") or []:
        web = chunk.get("web") or {}
        uri = web.get("uri")
        if uri and isinstance(uri, str):
            title = str(web.get("title") or uri)
            sources.append({"title": title, "uri": uri})

    return {
        "text": full_text,
        "search_queries": search_queries,
        "sources": sources[:6],
    }


async def call_gemini_tts(
    text: str,
    tts_model: str = "gemini-3.8-flash-tts",
    voice_name: str = "Charon",
    style_prompt: str = "",
) -> str | None:
    """Generate spoken audio using Gemini TTS and save as a WAV file in sessions/audio/."""
    api_key = get_api_key()
    url = f"{GEMINI_API_BASE}/models/{tts_model}:generateContent"

    clean_text = re.sub(r"[*_#`~]+", "", text).strip()
    if style_prompt:
        prompt_text = f"{style_prompt.strip()}\n\nTranscript to read aloud:\n{clean_text}"
    else:
        prompt_text = clean_text

    payload: dict[str, Any] = {
        "contents": [{"parts": [{"text": prompt_text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {
                        "voiceName": voice_name,
                    }
                }
            },
        },
    }

    async with httpx.AsyncClient(timeout=90.0) as client:
        resp = await client.post(
            url,
            headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
            json=payload,
        )

    if resp.status_code != 200:
        print(f"[TTS Warning] {tts_model} returned {resp.status_code}: {resp.text[:300]}")
        return None

    data = resp.json()
    candidates = data.get("candidates") or []
    if not candidates:
        return None

    parts = (candidates[0].get("content") or {}).get("parts") or []
    for part in parts:
        inline = part.get("inlineData") or part.get("inline_data")
        if inline and inline.get("data"):
            raw_bytes = base64.b64decode(inline["data"])
            mime = str(inline.get("mimeType") or inline.get("mime_type") or "audio/pcm;rate=24000")
            rate = 24000
            m = re.search(r"rate=(\d+)", mime)
            if m:
                rate = int(m.group(1))
            wav_bytes = pcm_to_wav_bytes(raw_bytes, sample_rate=rate)
            audio_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
            wav_path = AUDIO_DIR / f"{audio_id}.wav"
            wav_path.write_bytes(wav_bytes)
            return f"/api/audio/{audio_id}.wav"

    return None
