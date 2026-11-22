from __future__ import annotations

import asyncio
import os
import re
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch
from urllib.parse import quote

import httpx
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

from hear.app import build_container, create_app
from hear.auth import AuthSettings, EnvReader, verify_token
from hear.config import ConfigManager
from hear.db import HearDB
from hear.debug import RuntimeDebugProvider
from hear.gemini import GeminiClient, pcm_to_wav_bytes, prune_old_audio_files
from hear.geo import GeoCache, GeoService, compass_bearing_label, haversine_meters
from hear.logging_utils import EventLogger, bind_event_context, set_event_session_id
from hear.models.api import NarrateRequest, PingRequest
from hear.models.config import AppConfig, GeminiConfig, LensConfig, TriggersConfig
from hear.models.events import BackendEventType, EventLevel, EventSource
from hear.models.gemini import (
    GeminiContent,
    GeminiPart,
    GeminiTextResult,
    GeminiUsage,
    GroundingSource,
)
from hear.models.geo import GeoContext
from hear.models.session import CurrentLocation, TriggerLocation
from hear.paths import PROJECT_ROOT, AppPaths, resolve_data_dir
from hear.session import SessionManager, WalkSession, sanitize_client_id


class _FakeGeminiClient:
    def __init__(self, responses: list[GeminiTextResult], audio_url: str | None = None) -> None:
        self._responses = list(responses)
        self._audio_url = audio_url
        self.text_calls: list[dict[str, object]] = []

    async def generate_text(
        self,
        contents: list[GeminiContent],
        system_prompt: str,
        model: str,
        temperature: float = 0.75,
        enable_google_search: bool = True,
        thinking_level: str | None = None,
    ) -> GeminiTextResult:
        self.text_calls.append(
            {
                "model": model,
                "temperature": temperature,
                "enable_google_search": enable_google_search,
                "thinking_level": thinking_level,
            }
        )
        return self._responses.pop(0)

    async def generate_tts(
        self,
        text: str,
        tts_model: str = "gemini-3.8-flash-tts",
        voice_name: str = "Charon",
        style_prompt: str = "",
    ) -> str | None:
        return self._audio_url


class TestGeoHelpers(unittest.TestCase):
    def test_haversine_meters(self) -> None:
        self.assertAlmostEqual(haversine_meters(51.5, -0.1, 51.5, -0.1), 0.0)
        dist = haversine_meters(51.5142, -0.0885, 51.5152, -0.0885)
        self.assertGreater(dist, 100.0)
        self.assertLess(dist, 125.0)

    def test_compass_bearing(self) -> None:
        self.assertIsNone(compass_bearing_label(None))
        self.assertIsNone(compass_bearing_label(-5.0))
        self.assertIsNone(compass_bearing_label(365.0))
        self.assertEqual(compass_bearing_label(0.0), "N (0°)")
        self.assertEqual(compass_bearing_label(359.8), "N (0°)")
        self.assertEqual(compass_bearing_label(90.0), "E (90°)")
        self.assertEqual(compass_bearing_label(225.0), "SW (225°)")

    def test_geo_cache_anchor_lru_and_ttl(self) -> None:
        cache = GeoCache(max_entries=2, ttl_seconds=60.0)
        base_ctx = GeoContext(
            lat=51.5142,
            lon=-0.0885,
            short_label="Cornhill, City of London",
            full_address="Cornhill, London, UK",
            road="Cornhill",
            neighborhood="City of London",
            city="London",
        )
        cache.set(51.5142, -0.0885, base_ctx)

        # Within 25m -> cache hit
        entry = cache.find_close_entry(51.51425, -0.0885)
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertEqual(entry.result.short_label, "Cornhill, City of London")

        # Updating within 25m preserves original geocoded anchor coordinates
        updated_ctx = base_ctx.model_copy(update={"short_label": "1 Cornhill, City of London"})
        cache.set(51.51425, -0.0885, updated_ctx)
        entry = cache.find_close_entry(51.5142, -0.0885)
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertAlmostEqual(entry.lat, 51.5142)
        self.assertEqual(entry.result.short_label, "1 Cornhill, City of London")

        # Add a second entry far away, then touch the first entry so it becomes MRU
        other_ctx = base_ctx.model_copy(update={"lat": 51.5200, "short_label": "Moorgate"})
        cache.set(51.5200, -0.0885, other_ctx)
        self.assertIsNotNone(cache.find_close_entry(51.5142, -0.0885))

        # Adding a third entry evicts the LRU (51.5200) and keeps the MRU (51.5142)
        third_ctx = base_ctx.model_copy(update={"lat": 51.5300, "short_label": "Angel"})
        cache.set(51.5300, -0.0885, third_ctx)
        self.assertIsNotNone(cache.find_close_entry(51.5142, -0.0885))
        self.assertIsNone(cache.find_close_entry(51.5200, -0.0885))

    def test_reverse_geocode_caches_nominatim_result(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = httpx.Response(
            200,
            json={
                "display_name": "3 ter, Rue des Rosiers, Quartier Saint-Gervais, Paris",
                "address": {
                    "house_number": "3 ter",
                    "road": "Rue des Rosiers",
                    "quarter": "Quartier Saint-Gervais",
                    "city": "Paris",
                },
            },
        )
        geo = GeoService(http_client=mock_client)

        ctx1 = asyncio.run(geo.reverse_geocode(48.8566, 2.3604))
        self.assertEqual(ctx1.short_label, "3 ter Rue des Rosiers, Quartier Saint-Gervais, Paris")
        self.assertEqual(mock_client.get.call_count, 1)

        # Nearby coordinate within 25m reuses cached result without another HTTP call
        ctx2 = asyncio.run(geo.reverse_geocode(48.85665, 2.3604))
        self.assertEqual(ctx2.short_label, ctx1.short_label)
        self.assertEqual(mock_client.get.call_count, 1)

    def test_nominatim_failure_backoff(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.side_effect = httpx.ConnectError("Connection refused")
        geo = GeoService(http_client=mock_client)

        res1 = asyncio.run(geo.fetch_nominatim(51.5142, -0.0885))
        self.assertFalse(res1[5])
        self.assertEqual(mock_client.get.call_count, 1)

        # Immediate second call during 15s backoff window skips HTTP call
        res2 = asyncio.run(geo.fetch_nominatim(51.5142, -0.0885))
        self.assertFalse(res2[5])
        self.assertEqual(mock_client.get.call_count, 1)


class TestWalkSession(unittest.TestCase):
    def setUp(self) -> None:
        self.config = AppConfig(
            triggers=TriggersConfig(
                auto_trigger_enabled=True,
                min_distance_meters=120.0,
                min_cooldown_seconds=90.0,
                max_silence_seconds=300.0,
                min_drift_for_silence_trigger_meters=15.0,
            ),
            lenses=[
                LensConfig(id="econ", name="Economics", enabled=True, prompt="Econ prompt"),
                LensConfig(id="pol", name="Politics", enabled=True, prompt="Pol prompt"),
                LensConfig(id="off", name="Disabled", enabled=False, prompt="Off prompt"),
            ],
        )

    def test_sanitize_client_id(self) -> None:
        self.assertEqual(sanitize_client_id(None), "default")
        self.assertEqual(sanitize_client_id("  ../bad/id!@# "), "badid")
        self.assertEqual(sanitize_client_id("pixel_9_pro-1"), "pixel_9_pro-1")

    def test_sample_lens_without_replacement_and_reset(self) -> None:
        session = WalkSession("test")
        first = session.sample_lens(self.config)
        second = session.sample_lens(self.config)
        self.assertNotEqual(first.id, second.id)
        self.assertNotIn("off", session.used_lens_ids)

        # Pool exhausted -> resets and samples again
        third = session.sample_lens(self.config)
        self.assertIn(third.id, {"econ", "pol"})
        self.assertEqual(len(session.used_lens_ids), 1)

    def test_failed_gemini_turn_does_not_mutate_used_lens_ids(self) -> None:
        class _FlakyGeminiClient:
            def __init__(self) -> None:
                self.should_fail = True

            async def generate_text(
                self,
                contents: list[GeminiContent],
                system_prompt: str,
                model: str,
                temperature: float = 0.75,
                enable_google_search: bool = True,
                thinking_level: str | None = None,
            ) -> GeminiTextResult:
                if self.should_fail:
                    raise HTTPException(status_code=502, detail="Transient Gemini error")
                return GeminiTextResult(text="Recovered narration.", search_queries=[], sources=[])

            async def generate_tts(
                self,
                text: str,
                tts_model: str = "gemini-3.8-flash-tts",
                voice_name: str = "Charon",
                style_prompt: str = "",
            ) -> str | None:
                return None

        geo = GeoContext(
            lat=51.5142,
            lon=-0.0885,
            short_label="Cornhill",
            full_address="Cornhill, London",
            road="Cornhill",
            neighborhood="City",
            city="London",
        )
        cfg = self.config.model_copy(update={"gemini": GeminiConfig(tts_enabled=False)})
        flaky = _FlakyGeminiClient()
        session = WalkSession("test_fail_lens", gemini=flaky)

        # Failed turn on fresh session leaves used_lens_ids untouched
        with self.assertRaises(HTTPException):
            asyncio.run(session.execute_narration_turn(cfg, 51.5142, -0.0885, None, geo))
        self.assertEqual(session.used_lens_ids, [])

        # Failed turn when pool is exhausted does NOT clear used_lens_ids
        session.used_lens_ids = ["econ", "pol"]
        with self.assertRaises(HTTPException):
            asyncio.run(session.execute_narration_turn(cfg, 51.5142, -0.0885, None, geo))
        self.assertEqual(session.used_lens_ids, ["econ", "pol"])

        # Subsequent successful turn resets the exhausted pool and records the chosen lens
        flaky.should_fail = False
        msg = asyncio.run(session.execute_narration_turn(cfg, 51.5142, -0.0885, None, geo))
        assert msg.lens is not None
        self.assertEqual(session.used_lens_ids, [msg.lens.id])

    def test_trigger_heuristics(self) -> None:
        session = WalkSession("test")
        # No location yet -> should not trigger
        status = session.compute_trigger_status(self.config)
        self.assertFalse(status.should_trigger)

        # First GPS lock -> initial_location trigger
        now = time.time()
        session.current_location = CurrentLocation(
            lat=51.5142,
            lon=-0.0885,
            short_label="Start",
            full_address="Start",
            timestamp=now,
        )
        status = session.compute_trigger_status(self.config)
        self.assertTrue(status.should_trigger)
        self.assertEqual(status.trigger_reason, "initial_location")

        # After triggering at Start, cooldown blocks immediate re-trigger
        session.last_trigger_location = TriggerLocation(
            lat=51.5142, lon=-0.0885, short_label="Start", timestamp=now
        )
        session.last_audio_ended_at = now
        # Move ~150m north while still in cooldown
        session.current_location = CurrentLocation(
            lat=51.5156,
            lon=-0.0885,
            short_label="Moved",
            full_address="Moved",
            timestamp=now + 10,
        )
        status_cooldown = session.compute_trigger_status(self.config)
        self.assertFalse(status_cooldown.should_trigger)
        self.assertGreater(status_cooldown.cooldown_remaining_s, 0)

        # Once cooldown expires (100s ago) and moved >= 120m -> distance trigger
        session.last_audio_ended_at = now - 100.0
        session.last_trigger_time = now - 110.0
        status_ready = session.compute_trigger_status(self.config)
        self.assertTrue(status_ready.should_trigger)
        assert status_ready.trigger_reason is not None
        self.assertTrue(status_ready.trigger_reason.startswith("distance"))

    def test_initial_location_respects_grace_window_after_opening_chat(self) -> None:
        session = WalkSession("test_chat_first")
        now = time.time()
        session.current_location = CurrentLocation(
            lat=51.5142,
            lon=-0.0885,
            short_label="Start",
            full_address="Start",
            timestamp=now,
        )
        # Simulate an opening chat turn before any narration (last_trigger_location is None)
        session.last_trigger_time = now
        session.last_audio_ended_at = now
        status = session.compute_trigger_status(self.config)
        self.assertFalse(status.should_trigger)
        # Uses short 15s grace window instead of full 90s min_cooldown_seconds
        self.assertGreater(status.cooldown_remaining_s, 0)
        self.assertLessEqual(status.cooldown_remaining_s, 15.0)

        # After 16s (well before 90s), initial_location is allowed to fire
        session.last_trigger_time = now - 16.0
        session.last_audio_ended_at = now - 16.0
        status_after_grace = session.compute_trigger_status(self.config)
        self.assertTrue(status_after_grace.should_trigger)
        self.assertEqual(status_after_grace.trigger_reason, "initial_location")

    def test_pause_hold_and_time_distance_expiry(self) -> None:
        session = WalkSession("test_pause_hold")
        now = time.time()
        start_lat, start_lon = 51.5142, -0.0885
        moved_lat, moved_lon = 51.5160, -0.0885  # ~200m north

        session.last_trigger_location = TriggerLocation(
            lat=start_lat, lon=start_lon, short_label="Start", timestamp=now - 120.0
        )
        session.last_trigger_time = now - 120.0
        session.last_audio_ended_at = now - 120.0
        session.current_location = CurrentLocation(
            lat=moved_lat,
            lon=moved_lon,
            short_label="Moved",
            full_address="Moved",
            timestamp=now,
        )

        # User pauses mid-clip at start_lat, start_lon
        session.update_audio_state(
            lat=start_lat,
            lon=start_lon,
            is_playing_audio=False,
            is_paused_audio=True,
            now=now - 60.0,
        )
        # Walked 200m (>150m), but only paused for 60s (<180s) -> hold stays active
        status_recent = session.compute_trigger_status(self.config)
        self.assertFalse(status_recent.should_trigger)
        self.assertTrue(status_recent.is_paused_audio)

        # Paused for 200s (>180s), but still standing near pause spot (<150m) -> hold stays active
        session.paused_at = now - 200.0
        session.paused_location = (moved_lat, moved_lon)
        status_nearby = session.compute_trigger_status(self.config)
        self.assertFalse(status_nearby.should_trigger)
        self.assertTrue(status_nearby.is_paused_audio)

        # Paused for 200s (>180s) AND walked ~200m (>150m) from pause spot -> hold expires!
        session.paused_location = (start_lat, start_lon)
        status_expired = session.compute_trigger_status(self.config)
        self.assertFalse(status_expired.is_paused_audio)
        self.assertTrue(status_expired.should_trigger)

    def test_session_manager_eviction(self) -> None:
        mgr = SessionManager(max_sessions=2)
        s1 = mgr.get("c1")
        s1.last_active_at = 100.0
        s2 = mgr.get("c2")
        s2.last_active_at = 200.0
        mgr.get("c3")
        self.assertNotIn("c1", mgr._sessions)
        self.assertIn("c2", mgr._sessions)
        self.assertIn("c3", mgr._sessions)


class TestAudioPruningAndAuth(unittest.TestCase):
    def test_prune_old_audio_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            for i in range(5):
                f = tmp_path / f"clip_{i}.wav"
                f.write_bytes(pcm_to_wav_bytes(b"\x00\x00" * 100))
                mtime = 1000.0 + i
                os.utime(f, (mtime, mtime))

            prune_old_audio_files(tmp_path, max_files=3)
            remaining = sorted(p.name for p in tmp_path.glob("*.wav"))
            self.assertEqual(remaining, ["clip_2.wav", "clip_3.wav", "clip_4.wav"])

    def test_verify_token_and_env_reader(self) -> None:
        self.assertTrue(verify_token("secret123", "secret123"))
        self.assertFalse(verify_token("wrong", "secret123"))
        self.assertFalse(verify_token("", "secret123"))

        with tempfile.TemporaryDirectory() as tmpdir:
            dotenv = Path(tmpdir) / ".env"
            dotenv.write_text('HEAR_AUTH_TOKEN="from_dotenv"\n', encoding="utf-8")
            env = EnvReader(dotenv_path=dotenv, environ={"GEMINI_API_KEY": "from_env"})
            auth = AuthSettings(env)
            self.assertEqual(auth.expected_token(), "from_dotenv")
            self.assertEqual(env.get("GEMINI_API_KEY"), "from_env")

    def test_api_coordinate_validation_and_duplicate_narrate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            test_app = create_app(
                data_dir=tmp_path,
                config_path=tmp_path / "config.yaml",
                env=EnvReader(environ={}),
                auth_token="test-token",
                configure_logs=False,
            )
            with TestClient(test_app) as client:
                # Unauthorized without token
                r_unauth = client.get("/api/state")
                self.assertEqual(r_unauth.status_code, 401)

                headers = {"x-hear-token": "test-token", "x-hear-client-id": "testclient"}
                # Out-of-bounds latitude rejected by Pydantic
                r_bad_coords = client.post(
                    "/api/ping",
                    json={"lat": 95.0, "lon": 0.0},
                    headers=headers,
                )
                self.assertEqual(r_bad_coords.status_code, 422)

                # Duplicate /api/narrate or /api/chat while is_generating is True -> 409 Conflict
                container = test_app.state.container
                sess = container.session_manager.get("testclient")
                sess.is_generating = True
                try:
                    r_conflict = client.post(
                        "/api/narrate",
                        json={"lat": 51.5142, "lon": -0.0885},
                        headers=headers,
                    )
                    self.assertEqual(r_conflict.status_code, 409)

                    r_chat_conflict = client.post(
                        "/api/chat",
                        json={"message": "Tell me more", "lat": 51.5142, "lon": -0.0885},
                        headers=headers,
                    )
                    self.assertEqual(r_chat_conflict.status_code, 409)
                finally:
                    sess.is_generating = False

                # Authenticated 4xx errors persisted to SQLite; unauthenticated 401 probes skipped
                events = container.db.get_recent_events(limit=20)
                error_statuses = [
                    ev["payload"].get("status_code")
                    for ev in events
                    if ev["event_type"] == BackendEventType.HTTP_ERROR
                ]
                self.assertNotIn(401, error_statuses)
                self.assertIn(409, error_statuses)
                self.assertIn(422, error_statuses)

    def test_static_frontend_modules_served(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            test_app = create_app(
                data_dir=tmp_path,
                config_path=tmp_path / "config.yaml",
                env=EnvReader(environ={}),
                auth_token="test-token",
                configure_logs=False,
            )
            with TestClient(test_app) as client:
                index_resp = client.get("/")
                self.assertEqual(index_resp.status_code, 200)
                self.assertIn("/static/app.css", index_resp.text)
                self.assertIn("/static/app.js", index_resp.text)

                for asset in (
                    "/static/app.css",
                    "/static/state.js",
                    "/static/api.js",
                    "/static/telemetry.js",
                    "/static/audio.js",
                    "/static/drawers.js",
                    "/static/app.js",
                ):
                    resp = client.get(asset)
                    self.assertEqual(resp.status_code, 200, f"Failed to serve {asset}")

    def test_static_frontend_dom_ids_and_imports_integrity(self) -> None:
        static_dir = AppPaths.from_env().static_dir
        html = (static_dir / "index.html").read_text(encoding="utf-8")
        html_ids = set(re.findall(r'\bid="([^"]+)"', html))

        exports_by_file: dict[Path, set[str]] = {}
        for js_file in sorted(static_dir.glob("*.js")):
            src = js_file.read_text(encoding="utf-8")
            exported = set(
                re.findall(
                    r"\bexport\s+(?:async\s+)?(?:function|const|let|var|class)\s+([A-Za-z_$][\w$]*)",
                    src,
                )
            )
            for block in re.findall(r"\bexport\s*\{([^}]+)\}", src):
                for item in block.split(","):
                    cleaned = item.strip()
                    if not cleaned:
                        continue
                    parts = re.split(r"\s+as\s+", cleaned)
                    exported.add(parts[-1].strip())
            exports_by_file[js_file.resolve()] = exported

        for js_file in sorted(static_dir.glob("*.js")):
            src = js_file.read_text(encoding="utf-8")
            for rel_import in re.findall(r'from\s+["\'](\.[^"\']+)["\']', src):
                target = (js_file.parent / rel_import).resolve()
                self.assertTrue(
                    target.is_file(),
                    f"Broken relative JS import '{rel_import}' in {js_file.name}",
                )
            for names_block, rel_import in re.findall(
                r'import\s*\{([^}]+)\}\s*from\s*["\'](\.[^"\']+)["\']',
                src,
            ):
                target = (js_file.parent / rel_import).resolve()
                target_exports = exports_by_file.get(target, set())
                for item in names_block.split(","):
                    cleaned = item.strip()
                    if not cleaned:
                        continue
                    imported_name = re.split(r"\s+as\s+", cleaned)[0].strip()
                    self.assertIn(
                        imported_name,
                        target_exports,
                        f"Missing export '{imported_name}' in {target.name} "
                        f"(imported by {js_file.name})",
                    )
            for dom_id in re.findall(r'getElementById\(\s*["\']([^"\']+)["\']\s*\)', src):
                self.assertIn(
                    dom_id,
                    html_ids,
                    f"Missing DOM id='{dom_id}' in index.html (referenced by {js_file.name})",
                )

    def test_empty_gemini_response_raises_502(self) -> None:
        fake_resp = httpx.Response(
            200,
            json={"candidates": [{"finishReason": "SAFETY", "content": {"parts": []}}]},
        )
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.post.return_value = fake_resp

        with tempfile.TemporaryDirectory() as tmpdir:
            gemini = GeminiClient(
                audio_dir=Path(tmpdir),
                api_key="fake-key",
                http_client=mock_http,
            )
            with self.assertRaises(HTTPException) as ctx:
                asyncio.run(
                    gemini.generate_text(
                        contents=[GeminiContent(role="user", parts=[GeminiPart(text="Hello")])],
                        system_prompt="System",
                        model="gemini-3.8-flash",
                    )
                )
            self.assertEqual(ctx.exception.status_code, 502)
            self.assertIn("SAFETY", str(ctx.exception.detail))

    def test_thought_only_gemini_response_raises_502_and_excludes_thoughts(self) -> None:
        thought_only_resp = httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "finishReason": "MAX_TOKENS",
                        "content": {
                            "parts": [
                                {"text": "Internal chain-of-thought reasoning...", "thought": True}
                            ]
                        },
                    }
                ]
            },
        )
        mixed_resp = httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "finishReason": "STOP",
                        "content": {
                            "parts": [
                                {"text": "Secret reasoning...", "thought": True},
                                {"text": "Visible historical narration."},
                            ]
                        },
                    }
                ]
            },
        )
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.post.side_effect = [thought_only_resp, mixed_resp]

        with tempfile.TemporaryDirectory() as tmpdir:
            gemini = GeminiClient(
                audio_dir=Path(tmpdir),
                api_key="fake-key",
                http_client=mock_http,
            )
            with self.assertRaises(HTTPException) as ctx:
                asyncio.run(
                    gemini.generate_text(
                        contents=[GeminiContent(role="user", parts=[GeminiPart(text="Hello")])],
                        system_prompt="System",
                        model="gemini-4-hydrogen",
                        thinking_level="medium",
                    )
                )
            self.assertEqual(ctx.exception.status_code, 502)
            self.assertIn("MAX_TOKENS", str(ctx.exception.detail))

            result = asyncio.run(
                gemini.generate_text(
                    contents=[GeminiContent(role="user", parts=[GeminiPart(text="Hello")])],
                    system_prompt="System",
                    model="gemini-4-hydrogen",
                    thinking_level="medium",
                )
            )
            self.assertEqual(result.text, "Visible historical narration.")

    def test_ping_and_narrate_normalize_negative_gps_sentinels(self) -> None:
        ping = PingRequest(
            lat=51.5142,
            lon=-0.0885,
            accuracy=-1.0,
            heading=-1.0,
            speed=-1.0,
        )
        self.assertIsNone(ping.accuracy)
        self.assertIsNone(ping.heading)
        self.assertIsNone(ping.speed)

        ping_non_finite = PingRequest(
            lat=51.5142,
            lon=-0.0885,
            accuracy=float("nan"),
            heading=365.0,
            speed=float("inf"),
        )
        self.assertIsNone(ping_non_finite.accuracy)
        self.assertIsNone(ping_non_finite.heading)
        self.assertIsNone(ping_non_finite.speed)

        ping_valid = PingRequest(
            lat=51.5142,
            lon=-0.0885,
            accuracy=12.5,
            heading=0.0,
            speed=0.0,
        )
        self.assertEqual(ping_valid.accuracy, 12.5)
        self.assertEqual(ping_valid.heading, 0.0)
        self.assertEqual(ping_valid.speed, 0.0)

        narrate = NarrateRequest(lat=51.5142, lon=-0.0885, heading=-1.0)
        self.assertIsNone(narrate.heading)

        with self.assertRaises(ValidationError):
            PingRequest.model_validate({"lat": 51.5, "lon": -0.1, "heading": "invalid"})

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            container = build_container(
                data_dir=tmp_path,
                config_path=tmp_path / "config.yaml",
                env=EnvReader(environ={}),
                allow_unauthenticated=True,
            )
            geo_ctx = GeoContext(
                lat=51.5142,
                lon=-0.0885,
                short_label="Cornhill",
                full_address="Cornhill, London",
                road="Cornhill",
                neighborhood="City",
                city="London",
            )
            with patch.object(container.geo, "reverse_geocode", AsyncMock(return_value=geo_ctx)):
                test_app = create_app(container=container, configure_logs=False)
                with TestClient(test_app) as client:
                    resp = client.post(
                        "/api/ping",
                        json={
                            "lat": 51.5142,
                            "lon": -0.0885,
                            "accuracy": -1,
                            "heading": -1,
                            "speed": -1,
                        },
                    )
                    self.assertEqual(resp.status_code, 200)
                    loc = resp.json()["location"]
                    self.assertIsNone(loc["accuracy"])
                    self.assertIsNone(loc["heading"])
                    self.assertIsNone(loc["speed"])

    def test_cookie_auth_and_client_id_unquote_url_encoded_values(self) -> None:
        special_token = "abc+def/123=45@secret:token"
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            test_app = create_app(
                data_dir=tmp_path,
                config_path=tmp_path / "config.yaml",
                env=EnvReader(environ={}),
                auth_token=special_token,
                configure_logs=False,
            )
            audio_file = test_app.state.container.paths.audio_dir / "clip.wav"
            audio_file.write_bytes(pcm_to_wav_bytes(b"\x00\x00" * 10))

            with TestClient(test_app) as client:
                client.cookies.set("hear_token", quote(special_token, safe=""))
                client.cookies.set("hear_client_id", quote("elmo/phone", safe=""))

                state_resp = client.get("/api/state")
                self.assertEqual(state_resp.status_code, 200)
                self.assertIn("elmophone", test_app.state.container.session_manager._sessions)

                audio_resp = client.get("/api/audio/clip.wav")
                self.assertEqual(audio_resp.status_code, 200)

        # Raw header containing literal %2B is NOT unquoted
        literal_percent_token = "tok%2Ben"
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            test_app = create_app(
                data_dir=tmp_path,
                config_path=tmp_path / "config.yaml",
                env=EnvReader(environ={}),
                auth_token=literal_percent_token,
                configure_logs=False,
            )
            with TestClient(test_app) as client:
                ok_resp = client.get(
                    "/api/state",
                    headers={"x-hear-token": literal_percent_token},
                )
                self.assertEqual(ok_resp.status_code, 200)

    def test_http_events_preserve_session_id_across_middleware(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            container = build_container(
                data_dir=tmp_path,
                config_path=tmp_path / "config.yaml",
                env=EnvReader(environ={}),
                allow_unauthenticated=True,
            )
            test_app = create_app(container=container, configure_logs=False)
            logged_requests: list[tuple[str, str | None, str | None]] = []
            orig_emit = container.events._emit

            def _capture_emit(
                event_type: BackendEventType | str,
                *,
                level: EventLevel,
                session_id: str | None,
                client_id: str | None,
                payload: dict[str, Any],
            ) -> tuple[str | None, str | None]:
                resolved_sid, resolved_cid = orig_emit(
                    event_type,
                    level=level,
                    session_id=session_id,
                    client_id=client_id,
                    payload=payload,
                )
                if event_type == BackendEventType.HTTP_REQUEST:
                    logged_requests.append(
                        (str(payload.get("path", "")), resolved_sid, resolved_cid)
                    )
                return resolved_sid, resolved_cid

            with (
                patch.object(container.events, "_emit", side_effect=_capture_emit),
                TestClient(test_app, raise_server_exceptions=False) as client,
            ):
                headers = {"x-hear-client-id": "elmo"}
                state_resp = client.get("/api/state", headers=headers)
                self.assertEqual(state_resp.status_code, 200)
                first_sid = state_resp.json()["session_id"]

                # HTTP_REQUEST for /api/state carries session_id & client_id via ContextVar
                self.assertIn(("/api/state", first_sid, "elmo"), logged_requests)

                # HTTP_ERROR for /api/narrate (400: no coords yet) persists with session_id
                err_resp = client.post("/api/narrate", json={}, headers=headers)
                self.assertEqual(err_resp.status_code, 400)

                # HTTP_UNHANDLED_ERROR persists with session_id
                with patch.object(
                    container.geo,
                    "reverse_geocode",
                    AsyncMock(side_effect=RuntimeError("boom")),
                ):
                    crash_resp = client.post(
                        "/api/ping",
                        json={"lat": 51.5142, "lon": -0.0885},
                        headers=headers,
                    )
                    self.assertEqual(crash_resp.status_code, 500)

                # /api/session/reset updates ContextVar session_id for SESSION_RESET & HTTP_REQUEST
                reset_resp = client.post("/api/session/reset", headers=headers)
                self.assertEqual(reset_resp.status_code, 200)
                new_sid = reset_resp.json()["session_id"]
                self.assertNotEqual(new_sid, first_sid)
                self.assertIn(("/api/session/reset", new_sid, "elmo"), logged_requests)

                events = container.db.get_recent_events(client_id="elmo", limit=20)
                http_errors = [
                    ev for ev in events if ev["event_type"] == BackendEventType.HTTP_ERROR
                ]
                self.assertEqual(len(http_errors), 1)
                self.assertEqual(http_errors[0]["session_id"], first_sid)

                unhandled = [
                    ev for ev in events if ev["event_type"] == BackendEventType.HTTP_UNHANDLED_ERROR
                ]
                self.assertEqual(len(unhandled), 1)
                self.assertEqual(unhandled[0]["session_id"], first_sid)

                resets = [ev for ev in events if ev["event_type"] == BackendEventType.SESSION_RESET]
                self.assertEqual(len(resets), 1)
                self.assertEqual(resets[0]["session_id"], new_sid)

    def test_tts_malformed_response_degrades_to_none(self) -> None:
        bad_json_resp = httpx.Response(200, content=b"not-json")
        bad_b64_resp = httpx.Response(
            200,
            json={
                "candidates": [
                    {"content": {"parts": [{"inlineData": {"data": "!!!not-base64!!!"}}]}}
                ]
            },
        )
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.post.side_effect = [bad_json_resp, bad_b64_resp]

        with tempfile.TemporaryDirectory() as tmpdir:
            gemini = GeminiClient(
                audio_dir=Path(tmpdir),
                api_key="fake-key",
                http_client=mock_http,
            )
            self.assertIsNone(asyncio.run(gemini.generate_tts("Hello")))
            self.assertIsNone(asyncio.run(gemini.generate_tts("Hello")))

    def test_config_manager_atomic_save_and_corrupt_yaml_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "config.yaml"
            mgr = ConfigManager(cfg_path)
            initial = AppConfig(system_prompt="Initial prompt")
            mgr.save(initial)
            self.assertEqual(mgr.get().system_prompt, "Initial prompt")

            # Write invalid YAML to disk and bump mtime -> falls back to last valid config
            cfg_path.write_text(":::invalid_yaml: [unclosed", encoding="utf-8")
            os.utime(cfg_path, (time.time() + 10.0, time.time() + 10.0))
            self.assertEqual(mgr.get().system_prompt, "Initial prompt")

    def test_gemini_config_model_id_validation(self) -> None:
        cfg = GeminiConfig()
        self.assertEqual(cfg.model, "gemini-4-hydrogen")
        self.assertEqual(cfg.thinking_level, "medium")

        with self.assertRaises(ValidationError):
            GeminiConfig(model="../bad/model")
        with self.assertRaises(ValidationError):
            GeminiConfig.model_validate({"thinking_level": "minimal"})

    def test_gemini_generate_text_payload_thinking_and_temperature(self) -> None:
        ok_resp = httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"text": "Historical note."}]}}]},
        )
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.post.return_value = ok_resp

        with tempfile.TemporaryDirectory() as tmpdir:
            gemini = GeminiClient(
                audio_dir=Path(tmpdir),
                api_key="fake-key",
                http_client=mock_http,
            )
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
            self.assertEqual(
                payload_g4["generationConfig"],
                {"thinkingConfig": {"thinkingLevel": "medium"}},
            )
            self.assertNotIn("temperature", payload_g4["generationConfig"])

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
            self.assertEqual(
                payload_g38["generationConfig"],
                {"temperature": 0.75, "thinkingConfig": {"thinkingLevel": "low"}},
            )

    def test_walk_session_forwards_thinking_level(self) -> None:
        fake_gemini = _FakeGeminiClient(
            [GeminiTextResult(text="Narration text.", search_queries=[], sources=[])]
        )
        sess = WalkSession(client_id="elmo", gemini=fake_gemini)
        cfg = AppConfig(
            gemini=GeminiConfig(
                model="gemini-4-hydrogen",
                thinking_level="high",
                tts_enabled=False,
            )
        )
        asyncio.run(sess.run_gemini_turn(cfg, "Tell me about this street."))
        self.assertEqual(len(fake_gemini.text_calls), 1)
        self.assertEqual(fake_gemini.text_calls[0]["model"], "gemini-4-hydrogen")
        self.assertEqual(fake_gemini.text_calls[0]["thinking_level"], "high")

    def test_gemini_usage_metadata_parsed_and_logged_on_completion_events(self) -> None:
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

        with tempfile.TemporaryDirectory() as tmpdir:
            gemini = GeminiClient(
                audio_dir=Path(tmpdir),
                api_key="fake-key",
                http_client=mock_http,
            )
            res1 = asyncio.run(
                gemini.generate_text(
                    contents=[GeminiContent(role="user", parts=[GeminiPart(text="Hello")])],
                    system_prompt="System",
                    model="gemini-4-hydrogen",
                    thinking_level="medium",
                )
            )
            self.assertEqual(
                res1.usage,
                GeminiUsage(
                    prompt_tokens=4200,
                    output_tokens=310,
                    thought_tokens=680,
                    cached_tokens=3900,
                    tool_use_prompt_tokens=950,
                    total_tokens=6140,
                ),
            )

            res2 = asyncio.run(
                gemini.generate_text(
                    contents=[GeminiContent(role="user", parts=[GeminiPart(text="Hello")])],
                    system_prompt="System",
                    model="gemini-4-hydrogen",
                )
            )
            self.assertIsNone(res2.usage)

            store = HearDB(Path(tmpdir) / "hear.db")
            store.init()
            fake_gemini = _FakeGeminiClient(
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
            sess = WalkSession(client_id="elmo", store=store, gemini=fake_gemini)
            sess.reset()
            cfg = AppConfig(
                gemini=GeminiConfig(
                    model="gemini-4-hydrogen",
                    thinking_level="medium",
                    tts_enabled=False,
                ),
                lenses=[LensConfig(id="econ", name="Econ", enabled=True, prompt="Econ")],
            )
            geo = GeoContext(
                lat=51.5531,
                lon=-0.1977,
                short_label="Achilles Road, West Hampstead",
                full_address="Achilles Road, London",
                road="Achilles Road",
                neighborhood="West Hampstead",
                city="London",
            )
            asyncio.run(sess.execute_narration_turn(cfg, 51.5531, -0.1977, None, geo))
            asyncio.run(
                sess.execute_chat_turn(
                    cfg, "Who financed it?", 51.5531, -0.1977, "Achilles Road, West Hampstead"
                )
            )

            events = {
                e["event_type"]: e["payload"]
                for e in store.get_recent_events(client_id="elmo", session_id=sess.session_id)
            }
            nar = events[BackendEventType.NARRATION_COMPLETED]
            self.assertEqual(nar["thinking_level"], "medium")
            self.assertEqual(nar["prompt_tokens"], 1200)
            self.assertEqual(nar["output_tokens"], 250)
            self.assertEqual(nar["thought_tokens"], 500)
            self.assertEqual(nar["cached_tokens"], 0)
            self.assertEqual(nar["total_tokens"], 1950)
            self.assertNotIn("tool_use_prompt_tokens", nar)

            chat = events[BackendEventType.CHAT_COMPLETED]
            self.assertEqual(chat["thinking_level"], "medium")
            self.assertEqual(chat["prompt_tokens"], 5000)
            self.assertEqual(chat["cached_tokens"], 4100)
            self.assertEqual(chat["thought_tokens"], 320)
            self.assertEqual(chat["search_query_count"], 2)
            store.close()

    def test_resolve_data_dir_and_app_paths(self) -> None:
        self.assertEqual(resolve_data_dir(""), PROJECT_ROOT / "data")
        with tempfile.TemporaryDirectory() as tmpdir:
            expected = Path(tmpdir).resolve()
            self.assertEqual(resolve_data_dir(tmpdir), expected)
            with patch.dict("os.environ", {"HEAR_DATA_DIR": tmpdir}):
                self.assertEqual(resolve_data_dir(), expected)
            paths = AppPaths.from_env(data_dir=tmpdir)
            self.assertEqual(paths.db_path, expected / "hear.db")
            self.assertEqual(paths.audio_dir, expected / "audio")


class TestSQLitePersistenceAndEvents(unittest.TestCase):
    def test_session_rehydration_client_isolation_and_fts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = HearDB(Path(tmpdir) / "hear.db")
            cfg = AppConfig(
                gemini=GeminiConfig(tts_enabled=False),
                lenses=[
                    LensConfig(id="econ", name="Economics", enabled=True, prompt="Econ lens"),
                ],
            )
            geo = GeoContext(
                lat=51.5142,
                lon=-0.0885,
                short_label="Cornhill, City of London",
                full_address="Cornhill, London, UK",
                road="Cornhill",
                neighborhood="City of London",
                city="London",
            )
            fake_gemini = _FakeGeminiClient(
                [
                    GeminiTextResult(
                        text="Cornhill sat at the heart of London's mercantile credit networks.",
                        search_queries=["Cornhill London credit history"],
                        sources=[
                            GroundingSource(title="British History", uri="https://example.com/bh")
                        ],
                    ),
                    GeminiTextResult(
                        text="The Royal Exchange was founded by Thomas Gresham in 1565.",
                        search_queries=[],
                        sources=[],
                    ),
                ]
            )

            mgr1 = SessionManager(store=store, gemini=fake_gemini)
            elmo = mgr1.get("elmo")
            asyncio.run(
                elmo.execute_narration_turn(
                    config=cfg,
                    lat=51.5142,
                    lon=-0.0885,
                    heading=90.0,
                    geo=geo,
                    reason="manual",
                )
            )
            asyncio.run(
                elmo.execute_chat_turn(
                    config=cfg,
                    cleaned_message="Who founded the Royal Exchange?",
                    lat=51.5142,
                    lon=-0.0885,
                    short_label="Cornhill, City of London",
                )
            )

            original_session_id = elmo.session_id

            # Simulate server restart with a brand-new SessionManager backed by the same DB
            mgr2 = SessionManager(store=store, gemini=fake_gemini)
            restored = mgr2.get("elmo")
            self.assertEqual(restored.session_id, original_session_id)
            self.assertEqual(len(restored.ui_messages), 3)
            self.assertEqual(len(restored.gemini_contents), 4)
            self.assertEqual(restored.used_lens_ids, ["econ"])
            self.assertIsNotNone(restored.last_trigger_location)
            assert restored.last_trigger_location is not None
            self.assertEqual(restored.last_trigger_location.short_label, "Cornhill, City of London")

            # Guest client_id stays completely isolated
            guest = mgr2.get("guest_abc")
            self.assertNotEqual(guest.session_id, original_session_id)
            self.assertEqual(guest.ui_messages, [])

            # FTS5 search finds persisted narrations scoped by client_id
            hits = store.search_messages("Gresham", client_id="elmo")
            self.assertEqual(len(hits), 1)
            self.assertEqual(store.search_messages("Gresham", client_id="guest_abc"), [])

            # Reset marks the previous session inactive so restart starts fresh
            restored.reset()
            self.assertNotEqual(restored.session_id, original_session_id)
            mgr3 = SessionManager(store=store, gemini=fake_gemini)
            fresh = mgr3.get("elmo")
            self.assertEqual(fresh.ui_messages, [])

            store.close()

    def test_location_sample_downsampling(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = HearDB(Path(tmpdir) / "hear.db")
            session = WalkSession("elmo", store=store)
            now = 1_700_000_000.0
            loc1 = CurrentLocation(
                lat=51.5142, lon=-0.0885, short_label="A", full_address="A", timestamp=now
            )
            loc_same = CurrentLocation(
                lat=51.51421, lon=-0.0885, short_label="A", full_address="A", timestamp=now + 6.0
            )
            loc_moved = CurrentLocation(
                lat=51.5145, lon=-0.0885, short_label="B", full_address="B", timestamp=now + 12.0
            )
            loc_later = CurrentLocation(
                lat=51.5145, lon=-0.0885, short_label="B", full_address="B", timestamp=now + 50.0
            )

            self.assertTrue(asyncio.run(session.record_location_sample_if_needed(loc1)))
            self.assertFalse(asyncio.run(session.record_location_sample_if_needed(loc_same)))
            self.assertTrue(asyncio.run(session.record_location_sample_if_needed(loc_moved)))
            self.assertTrue(asyncio.run(session.record_location_sample_if_needed(loc_later)))
            store.close()

    def test_config_override_persists_in_sqlite_across_restarts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            store = HearDB(tmp_path / "hear.db")
            cfg_path = tmp_path / "config.yaml"
            cfg_path.write_text("system_prompt: From YAML\n", encoding="utf-8")

            mgr1 = ConfigManager(cfg_path, store=store)
            self.assertEqual(mgr1.get().system_prompt, "From YAML")
            mgr1.save(AppConfig(system_prompt="Saved via UI"))

            # Simulate fresh container where baked-in config.yaml is older than the DB override
            older_ts = time.time() - 3600.0
            cfg_path.write_text("system_prompt: Baked In Default\n", encoding="utf-8")
            os.utime(cfg_path, (older_ts, older_ts))

            mgr2 = ConfigManager(cfg_path, store=store)
            self.assertEqual(mgr2.get().system_prompt, "Saved via UI")
            store.close()

    def test_client_events_endpoint_and_pruning(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            test_app = create_app(
                data_dir=tmp_path,
                config_path=tmp_path / "config.yaml",
                env=EnvReader(environ={}),
                auth_token="test-token",
                configure_logs=False,
            )
            with TestClient(test_app) as client:
                resp = client.post(
                    "/api/events",
                    headers={"x-hear-token": "test-token", "x-hear-client-id": "elmo"},
                    json={
                        "events": [
                            {
                                "level": EventLevel.WARN,
                                "event_type": "gps_gap",
                                "payload": {"gap_seconds": 42},
                            },
                            {
                                "level": EventLevel.INFO,
                                "event_type": "visibility_change",
                                "payload": {"state": "hidden"},
                            },
                        ]
                    },
                )
                self.assertEqual(resp.status_code, 200)
                self.assertEqual(resp.json()["accepted"], 2)

                store = test_app.state.container.db
                events = store.get_recent_events(client_id="elmo", limit=10)
                self.assertEqual(len(events), 2)
                self.assertEqual(events[0]["source"], EventSource.FRONTEND)

                # Insert an expired event and verify pruning removes only the old row
                store.record_event(
                    source=EventSource.BACKEND,
                    level=EventLevel.INFO,
                    event_type="stale_event",
                    ts=time.time() - 40 * 86400.0,
                )
                pruned = store.prune_old_events(max_age_days=30)
                self.assertEqual(pruned, 1)
                self.assertEqual(len(store.get_recent_events(client_id="elmo", limit=10)), 2)

    def test_rehydration_clears_missing_audio_url_and_keeps_existing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio_dir = root / "audio"
            audio_dir.mkdir()
            (audio_dir / "present.wav").write_bytes(b"RIFF")
            store = HearDB(root / "hear.db")
            store.init()
            cfg = AppConfig(
                gemini=GeminiConfig(tts_enabled=True),
                lenses=[LensConfig(id="l1", name="L1", prompt="P1")],
            )
            geo = GeoContext(
                lat=51.5142,
                lon=-0.0885,
                short_label="Cornhill",
                full_address="Cornhill, London",
                road="Cornhill",
                neighborhood="City",
                city="London",
            )
            fake_present = _FakeGeminiClient(
                [GeminiTextResult(text="One", search_queries=[], sources=[])],
                audio_url="/api/audio/present.wav",
            )
            fake_missing = _FakeGeminiClient(
                [GeminiTextResult(text="Two", search_queries=[], sources=[])],
                audio_url="/api/audio/missing.wav",
            )
            mgr1 = SessionManager(store=store, audio_dir=audio_dir, gemini=fake_present)
            sess = mgr1.get("elmo")
            asyncio.run(sess.execute_narration_turn(cfg, 51.5142, -0.0885, None, geo))
            sess._gemini = fake_missing
            asyncio.run(sess.execute_narration_turn(cfg, 51.5150, -0.0885, None, geo))

            mgr2 = SessionManager(store=store, audio_dir=audio_dir)
            restored = mgr2.get("elmo")
            self.assertEqual(len(restored.ui_messages), 2)
            self.assertEqual(
                getattr(restored.ui_messages[0], "audio_url", None), "/api/audio/present.wav"
            )
            self.assertIsNone(getattr(restored.ui_messages[1], "audio_url", None))
            store.close()

    def test_same_second_multiple_resets_produce_unique_session_ids(self) -> None:
        session = WalkSession(client_id="elmo")
        seen = {session.session_id}
        for _ in range(5):
            session.reset(persist_deactivate=False)
            self.assertNotIn(session.session_id, seen)
            seen.add(session.session_id)

    def test_event_ts_normalization_and_payload_truncation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = HearDB(Path(tmpdir) / "hear.db")
            store.init()
            now = time.time()
            store.record_events_batch(
                [
                    {
                        "ts": (now - 10.0) * 1000.0,  # JS milliseconds
                        "source": EventSource.FRONTEND,
                        "level": EventLevel.INFO,
                        "event_type": "js_ms_event",
                        "client_id": "elmo",
                        "payload": {"ok": True},
                    },
                    {
                        "ts": now + 999999.0,  # Far future timestamp
                        "source": EventSource.FRONTEND,
                        "level": EventLevel.INFO,
                        "event_type": "future_event",
                        "client_id": "elmo",
                        "payload": {"blob": "x" * 20000},
                    },
                ]
            )
            events = {e["event_type"]: e for e in store.get_recent_events(client_id="elmo")}
            self.assertAlmostEqual(events["js_ms_event"]["ts"], now - 10.0, delta=2.0)
            self.assertLessEqual(events["future_event"]["ts"], time.time() + 1.0)
            self.assertTrue(events["future_event"]["payload"].get("_truncated"))
            store.close()

    def test_event_logger_contextvar_correlation_and_malformed_text_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = HearDB(Path(tmpdir) / "hear.db")
            store.init()
            ev_logger = EventLogger(store=store)
            mock_http = AsyncMock(spec=httpx.AsyncClient)
            mock_http.post.return_value = httpx.Response(200, content=b"not-json")
            gemini = GeminiClient(
                audio_dir=Path(tmpdir),
                api_key="fake-key",
                http_client=mock_http,
                event_logger=ev_logger,
            )
            with bind_event_context(client_id="elmo"):
                set_event_session_id("sess_ctx_1")
                with self.assertRaises(HTTPException) as ctx:
                    asyncio.run(
                        gemini.generate_text(
                            contents=[GeminiContent(role="user", parts=[GeminiPart(text="Hi")])],
                            system_prompt="System",
                            model="gemini-3.8-flash",
                        )
                    )
                self.assertEqual(ctx.exception.status_code, 502)

            events = store.get_recent_events(client_id="elmo", session_id="sess_ctx_1")
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["event_type"], BackendEventType.GEMINI_TEXT_ERROR)
            store.close()

    def test_last_audio_ended_at_persists_without_creating_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = HearDB(Path(tmpdir) / "hear.db")
            store.init()
            # Updating a non-existent session should not insert a row
            store.update_last_audio_ended_at("missing_session", 1234.5)
            self.assertIsNone(store.load_latest_active_session("elmo"))

            # Updating an existing session persists last_audio_ended_at across reload
            cfg = AppConfig(
                gemini=GeminiConfig(tts_enabled=True),
                lenses=[LensConfig(id="l1", name="L1", prompt="P1")],
            )
            geo = GeoContext(
                lat=51.5142,
                lon=-0.0885,
                short_label="Cornhill",
                full_address="Cornhill, London",
                road="Cornhill",
                neighborhood="City",
                city="London",
            )
            fake_gemini = _FakeGeminiClient(
                [GeminiTextResult(text="Hi", search_queries=[], sources=[])],
                audio_url="/api/audio/clip.wav",
            )
            sess = WalkSession(client_id="elmo", store=store, gemini=fake_gemini)
            asyncio.run(sess.execute_narration_turn(cfg, 51.5142, -0.0885, None, geo))
            changed = sess.update_audio_state(
                lat=51.5142,
                lon=-0.0885,
                is_playing_audio=False,
                audio_just_ended=True,
                now=5000.0,
            )
            self.assertTrue(changed)
            asyncio.run(sess.persist_last_audio_ended_at())

            reloaded = store.load_latest_active_session("elmo")
            self.assertIsNotNone(reloaded)
            assert reloaded is not None
            self.assertAlmostEqual(reloaded.last_audio_ended_at or 0.0, 5000.0)
            store.close()

    def test_config_manager_readonly_yaml_save_not_clobbered_by_get(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "config.yaml"
            cfg_path.write_text("system_prompt: Disk prompt\n", encoding="utf-8")
            store = HearDB(Path(tmpdir) / "hear.db")
            store.init()
            mgr = ConfigManager(cfg_path, store=store)
            with patch("os.replace", side_effect=OSError("read-only filesystem")):
                mgr.save(AppConfig(system_prompt="SQLite prompt"))
            self.assertEqual(mgr.get().system_prompt, "SQLite prompt")
            store.close()

    def test_app_paths_empty_data_dir_env_reader_and_ensure_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            env_data = root / "from_env_reader"
            missing_static = root / "nonexistent_static"

            paths_empty = AppPaths.from_env(
                data_dir="",
                config_path=str(root / "custom.yaml"),
                project_root=root,
                static_dir=missing_static,
            )
            self.assertEqual(paths_empty.data_dir, root / "data")
            self.assertEqual(paths_empty.config_path, root / "custom.yaml")
            paths_empty.ensure_dirs()
            self.assertTrue(paths_empty.data_dir.is_dir())
            self.assertTrue(paths_empty.audio_dir.is_dir())
            self.assertFalse(missing_static.exists())

            paths_from_env = AppPaths.from_env(
                project_root=root,
                env=EnvReader(environ={"HEAR_DATA_DIR": str(env_data)}),
            )
            self.assertEqual(paths_from_env.data_dir, env_data)

    def test_create_app_isolated_config_and_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir).resolve()
            cfg_path = tmp_path / "isolated_config.yaml"
            with patch.dict("os.environ", {"HEAR_AUTH_TOKEN": "host-secret"}):
                test_app = create_app(
                    data_dir=tmp_path,
                    config_path=cfg_path,
                    env=EnvReader(environ={}),
                    allow_unauthenticated=True,
                    configure_logs=False,
                )
                with TestClient(test_app) as client:
                    state_resp = client.get("/api/state")
                    self.assertEqual(state_resp.status_code, 200)
                    self.assertFalse(state_resp.json()["has_api_key"])

                    put_resp = client.put(
                        "/api/config",
                        json=AppConfig(system_prompt="Isolated prompt").model_dump(),
                    )
                    self.assertEqual(put_resp.status_code, 200)
                    self.assertTrue(cfg_path.exists())
                    self.assertIn("Isolated prompt", cfg_path.read_text(encoding="utf-8"))

    def test_injected_http_client_not_replaced_or_closed(self) -> None:
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.is_closed = True

        geo = GeoService(http_client=mock_http)
        self.assertIs(geo._get_client(), mock_http)
        asyncio.run(geo.close())
        mock_http.aclose.assert_not_awaited()

        with tempfile.TemporaryDirectory() as tmpdir:
            gemini = GeminiClient(audio_dir=Path(tmpdir), api_key="k", http_client=mock_http)
            self.assertIs(gemini._get_client(), mock_http)
            asyncio.run(gemini.close())
            mock_http.aclose.assert_not_awaited()

    def test_lifespan_shutdown_closes_db_even_if_geo_close_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            container = build_container(
                data_dir=tmp_path,
                config_path=tmp_path / "config.yaml",
                env=EnvReader(environ={}),
                allow_unauthenticated=True,
            )
            geo_close = AsyncMock(side_effect=RuntimeError("geo shutdown boom"))
            gemini_close = AsyncMock()
            with (
                patch.object(container.geo, "close", geo_close),
                patch.object(container.gemini, "close", gemini_close),
            ):
                test_app = create_app(container=container, configure_logs=False)
                with TestClient(test_app) as client:
                    self.assertEqual(client.get("/api/config").status_code, 200)
                    self.assertIsNotNone(container.db._conn)
                geo_close.assert_awaited_once()
                gemini_close.assert_awaited_once()
                self.assertIsNone(container.db._conn)

    def test_runtime_debug_provider_resolves_git_lazily(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = AppPaths.from_env(data_dir=tmpdir, project_root=Path(tmpdir))
            env = EnvReader(environ={})
            with patch("hear.debug.subprocess.run") as mock_run:
                mock_run.return_value.returncode = 0
                mock_run.return_value.stdout = "abc1234\n"
                provider = RuntimeDebugProvider(paths, env)
                self.assertEqual(mock_run.call_count, 0)

                info1 = provider.get_info()
                self.assertEqual(info1.git_commit, "abc1234")
                self.assertEqual(mock_run.call_count, 2)

                # Subsequent calls reuse cached git metadata
                info2 = provider.get_info()
                self.assertEqual(info2.git_commit, "abc1234")
                self.assertEqual(mock_run.call_count, 2)


if __name__ == "__main__":
    unittest.main()
