from __future__ import annotations

import asyncio
import os
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from hear.db import HearDB
from hear.gemini import GeminiClient, pcm_to_wav_bytes, prune_old_audio_files
from hear.logging_utils import EventLogger, bind_event_context, set_event_session_id
from hear.models.config import AppConfig, GeminiConfig, LensConfig
from hear.models.events import BackendEventType
from hear.models.gemini import (
    GeminiContent,
    GeminiPart,
    GeminiTextResult,
    GeminiUsage,
    GroundingSource,
)
from hear.session import WalkSession
from tests.helpers import FakeGeminiClient, make_geo


def test_prune_old_audio_files(tmp_path: Path) -> None:
    for i in range(5):
        f = tmp_path / f"clip_{i}.wav"
        f.write_bytes(pcm_to_wav_bytes(b"\x00\x00" * 100))
        mtime = 1000.0 + i
        os.utime(f, (mtime, mtime))

    prune_old_audio_files(tmp_path, max_files=3)
    remaining = sorted(p.name for p in tmp_path.glob("*.wav"))
    assert remaining == ["clip_2.wav", "clip_3.wav", "clip_4.wav"]


def test_empty_gemini_response_raises_502(tmp_path: Path) -> None:
    fake_resp = httpx.Response(
        200,
        json={"candidates": [{"finishReason": "SAFETY", "content": {"parts": []}}]},
    )
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.post.return_value = fake_resp

    gemini = GeminiClient(audio_dir=tmp_path, api_key="fake-key", http_client=mock_http)
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            gemini.generate_text(
                contents=[GeminiContent(role="user", parts=[GeminiPart(text="Hello")])],
                system_prompt="System",
                model="gemini-3.8-flash",
            )
        )
    assert exc_info.value.status_code == 502
    assert "SAFETY" in str(exc_info.value.detail)


def test_tts_malformed_response_degrades_to_none(tmp_path: Path) -> None:
    bad_json_resp = httpx.Response(200, content=b"not-json")
    bad_b64_resp = httpx.Response(
        200,
        json={
            "candidates": [{"content": {"parts": [{"inlineData": {"data": "!!!not-base64!!!"}}]}}]
        },
    )
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.post.side_effect = [bad_json_resp, bad_b64_resp]

    gemini = GeminiClient(audio_dir=tmp_path, api_key="fake-key", http_client=mock_http)
    assert asyncio.run(gemini.generate_tts("Hello")) is None
    assert asyncio.run(gemini.generate_tts("Hello")) is None


def test_gemini_config_model_id_validation() -> None:
    cfg = GeminiConfig()
    assert cfg.model == "gemini-4-hydrogen"
    assert cfg.thinking_level == "medium"

    with pytest.raises(ValidationError):
        GeminiConfig(model="../bad/model")
    with pytest.raises(ValidationError):
        GeminiConfig.model_validate({"thinking_level": "minimal"})


def test_gemini_generate_text_payload_thinking_and_temperature(tmp_path: Path) -> None:
    ok_resp = httpx.Response(
        200,
        json={"candidates": [{"content": {"parts": [{"text": "Historical note."}]}}]},
    )
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.post.return_value = ok_resp

    gemini = GeminiClient(audio_dir=tmp_path, api_key="fake-key", http_client=mock_http)
    asyncio.run(
        gemini.generate_text(
            contents=[GeminiContent(role="user", parts=[GeminiPart(text="Hello")])],
            system_prompt="System",
            model="gemini-4-hydrogen",
            temperature=0.75,
            thinking_level="medium",
        )
    )
    payload_g4 = mock_http.post.call_args.kwargs["json"]
    assert payload_g4["generationConfig"] == {"thinkingConfig": {"thinkingLevel": "medium"}}
    assert "temperature" not in payload_g4["generationConfig"]

    asyncio.run(
        gemini.generate_text(
            contents=[GeminiContent(role="user", parts=[GeminiPart(text="Hello")])],
            system_prompt="System",
            model="gemini-3.8-flash",
            temperature=0.75,
            thinking_level="low",
        )
    )
    payload_g38 = mock_http.post.call_args.kwargs["json"]
    assert payload_g38["generationConfig"] == {
        "temperature": 0.75,
        "thinkingConfig": {"thinkingLevel": "low"},
    }


def test_gemini_usage_metadata_parsed_and_logged_on_completion_events(
    tmp_path: Path, db_store: HearDB
) -> None:
    resp_with_usage = httpx.Response(
        200,
        json={
            "candidates": [{"content": {"parts": [{"text": "Historical note."}]}}],
            "usageMetadata": {
                "promptTokenCount": 4200,
                "candidatesTokenCount": 310,
                "thoughtsTokenCount": 680,
                "cachedContentTokenCount": 3900,
                "toolUsePromptTokenCount": 950,
                "totalTokenCount": 6140,
            },
        },
    )
    resp_without_usage = httpx.Response(
        200,
        json={"candidates": [{"content": {"parts": [{"text": "No usage."}]}}]},
    )
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.post.side_effect = [resp_with_usage, resp_without_usage]

    gemini = GeminiClient(audio_dir=tmp_path, api_key="fake-key", http_client=mock_http)
    res1 = asyncio.run(
        gemini.generate_text(
            contents=[GeminiContent(role="user", parts=[GeminiPart(text="Hello")])],
            system_prompt="System",
            model="gemini-4-hydrogen",
            thinking_level="medium",
        )
    )
    assert res1.usage == GeminiUsage(
        prompt_tokens=4200,
        output_tokens=310,
        thought_tokens=680,
        cached_tokens=3900,
        tool_use_prompt_tokens=950,
        total_tokens=6140,
    )

    res2 = asyncio.run(
        gemini.generate_text(
            contents=[GeminiContent(role="user", parts=[GeminiPart(text="Hello")])],
            system_prompt="System",
            model="gemini-4-hydrogen",
        )
    )
    assert res2.usage is None

    fake_gemini = FakeGeminiClient(
        [
            GeminiTextResult(
                text="Narration text.",
                search_queries=["q1"],
                sources=[GroundingSource(title="BHO", uri="https://example.com")],
                usage=GeminiUsage(
                    prompt_tokens=1200,
                    output_tokens=250,
                    thought_tokens=500,
                    cached_tokens=0,
                    total_tokens=1950,
                ),
            ),
            GeminiTextResult(
                text="Chat reply.",
                search_queries=["q2", "q3"],
                sources=[],
                usage=GeminiUsage(
                    prompt_tokens=5000,
                    output_tokens=180,
                    thought_tokens=320,
                    cached_tokens=4100,
                    total_tokens=5500,
                ),
            ),
        ]
    )
    sess = WalkSession(client_id="elmo", store=db_store, gemini=fake_gemini)
    sess.reset()
    cfg = AppConfig(
        gemini=GeminiConfig(
            model="gemini-4-hydrogen",
            thinking_level="medium",
            tts_enabled=False,
        ),
        lenses=[LensConfig(id="econ", name="Econ", enabled=True, prompt="Econ")],
    )
    geo = make_geo(
        lat=51.5531,
        lon=-0.1977,
        short_label="Achilles Road, West Hampstead",
        full_address="Achilles Road, London",
        road="Achilles Road",
        neighborhood="West Hampstead",
    )
    asyncio.run(sess.execute_narration_turn(cfg, 51.5531, -0.1977, None, geo))
    asyncio.run(
        sess.execute_chat_turn(
            cfg, "Who financed it?", 51.5531, -0.1977, "Achilles Road, West Hampstead"
        )
    )

    events = {
        e["event_type"]: e["payload"]
        for e in db_store.get_recent_events(client_id="elmo", session_id=sess.session_id)
    }
    nar = events[BackendEventType.NARRATION_COMPLETED]
    assert nar["thinking_level"] == "medium"
    assert nar["prompt_tokens"] == 1200
    assert nar["output_tokens"] == 250
    assert nar["thought_tokens"] == 500
    assert nar["cached_tokens"] == 0
    assert nar["total_tokens"] == 1950
    assert "tool_use_prompt_tokens" not in nar

    chat = events[BackendEventType.CHAT_COMPLETED]
    assert chat["thinking_level"] == "medium"
    assert chat["prompt_tokens"] == 5000
    assert chat["cached_tokens"] == 4100
    assert chat["thought_tokens"] == 320
    assert chat["search_query_count"] == 2


def test_event_logger_contextvar_correlation_and_malformed_text_json(
    tmp_path: Path, db_store: HearDB
) -> None:
    ev_logger = EventLogger(store=db_store)
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.post.return_value = httpx.Response(200, content=b"not-json")
    gemini = GeminiClient(
        audio_dir=tmp_path,
        api_key="fake-key",
        http_client=mock_http,
        event_logger=ev_logger,
    )
    with bind_event_context(client_id="elmo"):
        set_event_session_id("sess_ctx_1")
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(
                gemini.generate_text(
                    contents=[GeminiContent(role="user", parts=[GeminiPart(text="Hi")])],
                    system_prompt="System",
                    model="gemini-3.8-flash",
                )
            )
        assert exc_info.value.status_code == 502

    events = db_store.get_recent_events(client_id="elmo", session_id="sess_ctx_1")
    assert len(events) == 1
    assert events[0]["event_type"] == BackendEventType.GEMINI_TEXT_ERROR
