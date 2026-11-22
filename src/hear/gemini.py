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
from hear.http_client import ManagedAsyncClient
from hear.logging_utils import EventLogger
from hear.models.events import BackendEventType, EventLevel
from hear.models.gemini import GeminiContent, GeminiTextResult, GeminiUsage, GroundingSource

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


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _parse_usage_metadata(data: dict[str, Any]) -> GeminiUsage | None:
    raw = data.get("usageMetadata")
    if not isinstance(raw, dict):
        return None
    usage = GeminiUsage(
        prompt_tokens=_int_or_none(raw.get("promptTokenCount")),
        output_tokens=_int_or_none(raw.get("candidatesTokenCount")),
        thought_tokens=_int_or_none(raw.get("thoughtsTokenCount")),
        cached_tokens=_int_or_none(raw.get("cachedContentTokenCount")),
        tool_use_prompt_tokens=_int_or_none(raw.get("toolUsePromptTokenCount")),
        total_tokens=_int_or_none(raw.get("totalTokenCount")),
    )
    if not usage.model_dump(exclude_none=True):
        return None
    return usage


def _write_and_prune_wav(audio_dir: Path, wav_path: Path, wav_bytes: bytes) -> None:
    audio_dir.mkdir(parents=True, exist_ok=True)
    wav_path.write_bytes(wav_bytes)
    prune_old_audio_files(audio_dir)


class _GeminiRequestError(Exception):
    def __init__(
        self,
        *,
        http_status: int,
        detail: str,
        log_fields: dict[str, Any],
        warning_msg: str | None = None,
    ) -> None:
        super().__init__(detail)
        self.http_status = http_status
        self.detail = detail
        self.log_fields = log_fields
        self.warning_msg = warning_msg


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
        self._http = ManagedAsyncClient(http_client=http_client, timeout=90.0)

    def _get_client(self) -> httpx.AsyncClient:
        return self._http.get()

    async def close(self) -> None:
        await self._http.close()

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

    async def _log_gemini_error(
        self,
        *,
        event_type: BackendEventType,
        level: EventLevel,
        model: str,
        t0: float,
        **fields: Any,
    ) -> None:
        await self._events.log_async(
            event_type,
            level=level,
            model=model,
            latency_ms=int((time.monotonic() - t0) * 1000),
            **fields,
        )

    async def _post_generate_content(
        self,
        *,
        model: str,
        payload: dict[str, Any],
        timeout: float,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        api_key = self.get_api_key()
        url = f"{GEMINI_API_BASE}/models/{model}:generateContent"

        try:
            resp = await self._post_with_retry(url, api_key, payload, timeout=timeout)
        except httpx.TransportError as exc:
            raise _GeminiRequestError(
                http_status=502,
                detail=f"Gemini network error ({model}): {exc}",
                warning_msg=f"{model} network error: {exc}",
                log_fields={"error": str(exc), "error_type": type(exc).__name__},
            ) from exc

        if resp.status_code != 200:
            err_text = resp.text
            try:
                err_json = resp.json()
                err_text = str(err_json.get("error", {}).get("message", err_text))
            except ValueError:
                pass
            raise _GeminiRequestError(
                http_status=resp.status_code,
                detail=f"Gemini API error ({model}): {err_text}",
                warning_msg=f"{model} returned {resp.status_code}: {err_text[:300]}",
                log_fields={"status_code": resp.status_code, "error": err_text[:300]},
            )

        try:
            data = resp.json()
        except ValueError as exc:
            raise _GeminiRequestError(
                http_status=502,
                detail=f"Gemini returned invalid JSON ({model}): {exc}",
                warning_msg=f"{model} returned invalid JSON: {exc}",
                log_fields={"error": f"invalid_json: {exc}"},
            ) from exc

        candidates = data.get("candidates") or []
        if not candidates:
            raise _GeminiRequestError(
                http_status=500,
                detail="Gemini returned no candidates.",
                log_fields={"error": "no_candidates"},
            )

        return data, candidates

    async def generate_text(
        self,
        contents: list[GeminiContent],
        system_prompt: str,
        model: str,
        temperature: float = 0.75,
        enable_google_search: bool = True,
        thinking_level: str | None = None,
    ) -> GeminiTextResult:
        """Call Gemini generateContent REST API with multi-turn contents and Google Search."""
        generation_config: dict[str, Any] = {}
        # Gemini 4 deprecates custom sampling parameters (temperature is ignored server-side).
        if not model.startswith("gemini-4"):
            generation_config["temperature"] = temperature
        if thinking_level:
            generation_config["thinkingConfig"] = {"thinkingLevel": thinking_level}

        payload: dict[str, Any] = {
            "contents": [c.model_dump() for c in contents],
        }
        if generation_config:
            payload["generationConfig"] = generation_config
        if system_prompt:
            payload["system_instruction"] = {"parts": [{"text": system_prompt}]}
        if enable_google_search:
            payload["tools"] = [{"google_search": {}}]

        t0 = time.monotonic()
        try:
            data, candidates = await self._post_generate_content(
                model=model, payload=payload, timeout=60.0
            )
        except _GeminiRequestError as err:
            await self._log_gemini_error(
                event_type=BackendEventType.GEMINI_TEXT_ERROR,
                level=EventLevel.ERROR,
                model=model,
                t0=t0,
                **err.log_fields,
            )
            raise HTTPException(status_code=err.http_status, detail=err.detail) from err.__cause__

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
            await self._log_gemini_error(
                event_type=BackendEventType.GEMINI_TEXT_ERROR,
                level=EventLevel.ERROR,
                model=model,
                t0=t0,
                error="empty_text",
                finish_reason=finish_reason,
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
            usage=_parse_usage_metadata(data),
        )

    async def generate_tts(
        self,
        text: str,
        tts_model: str = "gemini-3.8-flash-tts",
        voice_name: str = "Charon",
        style_prompt: str = "",
    ) -> str | None:
        """Generate spoken audio using Gemini TTS and save as a WAV file in `audio_dir`."""
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
            _, candidates = await self._post_generate_content(
                model=tts_model, payload=payload, timeout=90.0
            )
        except _GeminiRequestError as err:
            if err.warning_msg:
                logger.warning("[TTS Warning] %s", err.warning_msg)
            await self._log_gemini_error(
                event_type=BackendEventType.GEMINI_TTS_ERROR,
                level=EventLevel.WARN,
                model=tts_model,
                t0=t0,
                **err.log_fields,
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
                    await self._log_gemini_error(
                        event_type=BackendEventType.GEMINI_TTS_ERROR,
                        level=EventLevel.WARN,
                        model=tts_model,
                        t0=t0,
                        error=f"decode_write_failed: {exc}",
                    )
                    return None

        await self._log_gemini_error(
            event_type=BackendEventType.GEMINI_TTS_ERROR,
            level=EventLevel.WARN,
            model=tts_model,
            t0=t0,
            error="no_inline_audio",
        )
        return None
