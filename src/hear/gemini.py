from __future__ import annotations

import asyncio
import base64
import io
import logging
import re
import time
import uuid
import wave
from pathlib import Path
from typing import Any

import httpx
from fastapi import HTTPException

from hear.auth import EnvReader
from hear.logging_utils import EventLogger
from hear.models.events import BackendEventType, EventLevel
from hear.models.gemini import GeminiContent, GeminiTextResult, GroundingSource

logger = logging.getLogger(__name__)

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
MAX_AUDIO_FILES = 25
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def prune_old_audio_files(audio_dir: Path, max_files: int = MAX_AUDIO_FILES) -> None:
    """Keep only the newest `max_files` WAV files in `audio_dir` to bound tmpfs/disk usage."""
    try:
        wav_files = sorted(
            audio_dir.glob("*.wav"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for stale in wav_files[max_files:]:
            stale.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("Failed to prune old audio files: %s", exc)


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


def _write_and_prune_wav(audio_dir: Path, wav_path: Path, wav_bytes: bytes) -> None:
    audio_dir.mkdir(parents=True, exist_ok=True)
    wav_path.write_bytes(wav_bytes)
    prune_old_audio_files(audio_dir)


class GeminiClient:
    """Gemini text + TTS client owning its HTTP pool, API key lookup, and audio directory."""

    def __init__(
        self,
        *,
        audio_dir: Path,
        api_key: str | None = None,
        env: EnvReader | None = None,
        event_logger: EventLogger | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.audio_dir = audio_dir
        self._api_key_override = api_key
        self._env = env
        self._events = event_logger if event_logger is not None else EventLogger(store=None)
        self._http_client = http_client
        self._owns_client = http_client is None

    def _get_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=90.0)
            self._owns_client = True
        return self._http_client

    async def close(self) -> None:
        if self._owns_client and self._http_client is not None and not self._http_client.is_closed:
            await self._http_client.aclose()
        if self._owns_client:
            self._http_client = None

    def get_api_key(self) -> str:
        if self._api_key_override is not None:
            key = self._api_key_override.strip()
        elif self._env is not None:
            key = self._env.get("GEMINI_API_KEY") or self._env.get("GOOGLE_API_KEY")
        else:
            key = ""
        if not key:
            raise HTTPException(
                status_code=400,
                detail=(
                    "GEMINI_API_KEY is not set. "
                    "Export GEMINI_API_KEY in your shell or add it to a .env file."
                ),
            )
        return key

    def has_api_key(self) -> bool:
        try:
            self.get_api_key()
            return True
        except HTTPException:
            return False

    async def _post_with_retry(
        self,
        url: str,
        api_key: str,
        payload: dict[str, Any],
        timeout: float,
        max_attempts: int = 3,
    ) -> httpx.Response:
        client = self._get_client()
        headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}

        for attempt in range(max_attempts):
            try:
                resp = await client.post(url, headers=headers, json=payload, timeout=timeout)
                if resp.status_code not in _RETRYABLE_STATUS_CODES or attempt == max_attempts - 1:
                    return resp
                logger.warning(
                    "Gemini returned %d (attempt %d/%d), retrying...",
                    resp.status_code,
                    attempt + 1,
                    max_attempts,
                )
            except httpx.TransportError as exc:
                if attempt == max_attempts - 1:
                    raise
                logger.warning(
                    "Gemini transport error %s (attempt %d/%d), retrying...",
                    exc,
                    attempt + 1,
                    max_attempts,
                )
            await asyncio.sleep(0.8 * (2**attempt))

        raise RuntimeError("Unreachable retry loop exit")

    async def generate_text(
        self,
        contents: list[GeminiContent],
        system_prompt: str,
        model: str,
        temperature: float = 0.75,
        enable_google_search: bool = True,
    ) -> GeminiTextResult:
        """Call Gemini generateContent REST API with multi-turn contents and Google Search."""
        api_key = self.get_api_key()
        url = f"{GEMINI_API_BASE}/models/{model}:generateContent"

        payload: dict[str, Any] = {
            "contents": [c.model_dump() for c in contents],
            "generationConfig": {
                "temperature": temperature,
            },
        }
        if system_prompt:
            payload["system_instruction"] = {"parts": [{"text": system_prompt}]}
        if enable_google_search:
            payload["tools"] = [{"google_search": {}}]

        t0 = time.monotonic()
        try:
            resp = await self._post_with_retry(url, api_key, payload, timeout=60.0)
        except httpx.TransportError as exc:
            await self._events.log_async(
                BackendEventType.GEMINI_TEXT_ERROR,
                level=EventLevel.ERROR,
                model=model,
                error=str(exc),
                error_type=type(exc).__name__,
                latency_ms=int((time.monotonic() - t0) * 1000),
            )
            raise HTTPException(
                status_code=502, detail=f"Gemini network error ({model}): {exc}"
            ) from exc

        if resp.status_code != 200:
            err_text = resp.text
            try:
                err_json = resp.json()
                err_text = str(err_json.get("error", {}).get("message", err_text))
            except ValueError:
                pass
            await self._events.log_async(
                BackendEventType.GEMINI_TEXT_ERROR,
                level=EventLevel.ERROR,
                model=model,
                status_code=resp.status_code,
                error=err_text[:300],
                latency_ms=int((time.monotonic() - t0) * 1000),
            )
            raise HTTPException(
                status_code=resp.status_code, detail=f"Gemini API error ({model}): {err_text}"
            )

        data = resp.json()
        candidates = data.get("candidates") or []
        if not candidates:
            await self._events.log_async(
                BackendEventType.GEMINI_TEXT_ERROR,
                level=EventLevel.ERROR,
                model=model,
                error="no_candidates",
                latency_ms=int((time.monotonic() - t0) * 1000),
            )
            raise HTTPException(status_code=500, detail="Gemini returned no candidates.")

        candidate = candidates[0]
        parts = (candidate.get("content") or {}).get("parts") or []
        text_chunks = [
            str(p.get("text", "")) for p in parts if "text" in p and not p.get("thought", False)
        ]
        if not text_chunks:
            text_chunks = [str(p.get("text", "")) for p in parts if "text" in p]
        full_text = "\n".join(t for t in text_chunks if t).strip()
        if not full_text:
            finish_reason = candidate.get("finishReason") or "UNKNOWN"
            await self._events.log_async(
                BackendEventType.GEMINI_TEXT_ERROR,
                level=EventLevel.ERROR,
                model=model,
                error="empty_text",
                finish_reason=finish_reason,
                latency_ms=int((time.monotonic() - t0) * 1000),
            )
            raise HTTPException(
                status_code=502,
                detail=f"Gemini returned empty text ({model}, finishReason={finish_reason}).",
            )

        grounding = candidate.get("groundingMetadata") or {}
        raw_queries = grounding.get("webSearchQueries") or []
        search_queries = [str(q) for q in raw_queries if q]

        sources: list[GroundingSource] = []
        for chunk in grounding.get("groundingChunks") or []:
            web = chunk.get("web") or {}
            uri = web.get("uri")
            if uri and isinstance(uri, str):
                title = str(web.get("title") or uri)
                sources.append(GroundingSource(title=title, uri=uri))

        return GeminiTextResult(
            text=full_text,
            search_queries=search_queries,
            sources=sources[:6],
        )

    async def generate_tts(
        self,
        text: str,
        tts_model: str = "gemini-3.8-flash-tts",
        voice_name: str = "Charon",
        style_prompt: str = "",
    ) -> str | None:
        """Generate spoken audio using Gemini TTS and save as a WAV file in `audio_dir`."""
        api_key = self.get_api_key()
        url = f"{GEMINI_API_BASE}/models/{tts_model}:generateContent"

        clean_text = re.sub(r"[*_#`~]+", "", text).strip()
        part: dict[str, Any] = {"text": clean_text}
        if style_prompt.strip():
            part["speechMetadata"] = {"style": style_prompt.strip()}

        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [part]}],
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

        t0 = time.monotonic()
        try:
            resp = await self._post_with_retry(url, api_key, payload, timeout=90.0)
        except httpx.TransportError as exc:
            logger.warning("[TTS Warning] %s network error: %s", tts_model, exc)
            await self._events.log_async(
                BackendEventType.GEMINI_TTS_ERROR,
                level=EventLevel.WARN,
                model=tts_model,
                error=str(exc),
                error_type=type(exc).__name__,
                latency_ms=int((time.monotonic() - t0) * 1000),
            )
            return None

        if resp.status_code != 200:
            logger.warning(
                "[TTS Warning] %s returned %d: %s", tts_model, resp.status_code, resp.text[:300]
            )
            await self._events.log_async(
                BackendEventType.GEMINI_TTS_ERROR,
                level=EventLevel.WARN,
                model=tts_model,
                status_code=resp.status_code,
                error=resp.text[:300],
                latency_ms=int((time.monotonic() - t0) * 1000),
            )
            return None

        try:
            data = resp.json()
        except ValueError as exc:
            logger.warning("[TTS Warning] %s returned invalid JSON: %s", tts_model, exc)
            await self._events.log_async(
                BackendEventType.GEMINI_TTS_ERROR,
                level=EventLevel.WARN,
                model=tts_model,
                error=f"invalid_json: {exc}",
                latency_ms=int((time.monotonic() - t0) * 1000),
            )
            return None

        candidates = data.get("candidates") or []
        if not candidates:
            await self._events.log_async(
                BackendEventType.GEMINI_TTS_ERROR,
                level=EventLevel.WARN,
                model=tts_model,
                error="no_candidates",
                latency_ms=int((time.monotonic() - t0) * 1000),
            )
            return None

        parts = (candidates[0].get("content") or {}).get("parts") or []
        for part in parts:
            inline = part.get("inlineData") or part.get("inline_data")
            if inline and inline.get("data"):
                try:
                    raw_bytes = base64.b64decode(inline["data"])
                    mime = str(
                        inline.get("mimeType") or inline.get("mime_type") or "audio/pcm;rate=24000"
                    )
                    rate = 24000
                    m = re.search(r"rate=(\d+)", mime)
                    if m:
                        rate = int(m.group(1))
                    wav_bytes = pcm_to_wav_bytes(raw_bytes, sample_rate=rate)
                    audio_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
                    wav_path = self.audio_dir / f"{audio_id}.wav"
                    await asyncio.to_thread(
                        _write_and_prune_wav, self.audio_dir, wav_path, wav_bytes
                    )
                    return f"/api/audio/{audio_id}.wav"
                except (ValueError, TypeError, OSError, wave.Error) as exc:
                    logger.warning("[TTS Warning] %s audio decode/write failed: %s", tts_model, exc)
                    await self._events.log_async(
                        BackendEventType.GEMINI_TTS_ERROR,
                        level=EventLevel.WARN,
                        model=tts_model,
                        error=f"decode_write_failed: {exc}",
                        latency_ms=int((time.monotonic() - t0) * 1000),
                    )
                    return None

        await self._events.log_async(
            BackendEventType.GEMINI_TTS_ERROR,
            level=EventLevel.WARN,
            model=tts_model,
            error="no_inline_audio",
            latency_ms=int((time.monotonic() - t0) * 1000),
        )
        return None
