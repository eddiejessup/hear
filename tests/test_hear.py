from __future__ import annotations

import asyncio
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

from hear.app import create_app
from hear.auth import AuthSettings, EnvReader, verify_token
from hear.config import ConfigManager
from hear.db import HearDB
from hear.gemini import GeminiClient, pcm_to_wav_bytes, prune_old_audio_files
from hear.geo import GeoCache, GeoService, compass_bearing_label, haversine_meters
from hear.logging_utils import EventLogger, bind_event_context, set_event_session_id
from hear.models.config import AppConfig, GeminiConfig, LensConfig, TriggersConfig
from hear.models.events import BackendEventType, EventLevel, EventSource
from hear.models.gemini import GeminiContent, GeminiPart, GeminiTextResult, GroundingSource
from hear.models.geo import GeoContext
from hear.models.session import CurrentLocation, TriggerLocation
from hear.paths import PROJECT_ROOT, AppPaths, resolve_data_dir
from hear.session import SessionManager, WalkSession, sanitize_client_id


class _FakeGeminiClient:
    def __init__(self, responses: list[GeminiTextResult], audio_url: str | None = None) -> None:
        self._responses = list(responses)
        self._audio_url = audio_url

    async def generate_text(
        self,
        contents: list[GeminiContent],
        system_prompt: str,
        model: str,
        temperature: float = 0.75,
        enable_google_search: bool = True,
    ) -> GeminiTextResult:
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

    def test_geo_cache_upgrade_and_ttl(self) -> None:
        cache = GeoCache(max_entries=2, ttl_seconds=60.0)
        base_ctx = GeoContext(
            lat=51.5142,
            lon=-0.0885,
            short_label="Cornhill, City of London",
            full_address="Cornhill, London, UK",
            road="Cornhill",
            neighborhood="City of London",
            city="London",
            nearby_features=[],
        )
        cache.set(51.5142, -0.0885, base_ctx, has_overpass=False)

        # Within 25m without overpass -> cache hit with has_overpass=False
        entry = cache.find_close_entry(51.51425, -0.0885)
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertFalse(entry.has_overpass)

        # Upgrade with overpass features preserves original geocoded anchor coordinates
        upgraded_ctx = base_ctx.model_copy(update={"nearby_features": ["Royal Exchange"]})
        cache.set(51.51425, -0.0885, upgraded_ctx, has_overpass=True)
        entry = cache.find_close_entry(51.5142, -0.0885)
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertTrue(entry.has_overpass)
        self.assertAlmostEqual(entry.lat, 51.5142)
        self.assertEqual(entry.result.nearby_features, ["Royal Exchange"])

        # Add a second entry far away, then touch the first entry so it becomes MRU
        other_ctx = base_ctx.model_copy(update={"lat": 51.5200, "short_label": "Moorgate"})
        cache.set(51.5200, -0.0885, other_ctx, has_overpass=True)
        self.assertIsNotNone(cache.find_close_entry(51.5142, -0.0885))

        # Adding a third entry evicts the LRU (51.5200) and keeps the MRU (51.5142)
        third_ctx = base_ctx.model_copy(update={"lat": 51.5300, "short_label": "Angel"})
        cache.set(51.5300, -0.0885, third_ctx, has_overpass=True)
        self.assertIsNotNone(cache.find_close_entry(51.5142, -0.0885))
        self.assertIsNone(cache.find_close_entry(51.5200, -0.0885))

    def test_nominatim_failure_backoff(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.is_closed = False
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

    def test_trigger_heuristics_and_prefetch(self) -> None:
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
        self.assertTrue(session.should_prefetch_overpass(self.config, 51.5142, -0.0885))

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
        # But distance is >75% of 120m, so overpass prefetch should activate
        self.assertTrue(session.should_prefetch_overpass(self.config, 51.5156, -0.0885))

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
            test_app = create_app(
                data_dir=Path(tmpdir),
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
            test_app = create_app(
                data_dir=Path(tmpdir),
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
                    "/static/audio.js",
                    "/static/drawers.js",
                    "/static/app.js",
                ):
                    resp = client.get(asset)
                    self.assertEqual(resp.status_code, 200, f"Failed to serve {asset}")

    def test_empty_gemini_response_raises_502(self) -> None:
        fake_resp = httpx.Response(
            200,
            json={"candidates": [{"finishReason": "SAFETY", "content": {"parts": []}}]},
        )
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.is_closed = False
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
        mock_http.is_closed = False
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
        with self.assertRaises(ValidationError):
            GeminiConfig(model="../bad/model")

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
                nearby_features=["Royal Exchange"],
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
            test_app = create_app(
                data_dir=Path(tmpdir),
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
                nearby_features=[],
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
            mock_http.is_closed = False
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
                nearby_features=[],
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


if __name__ == "__main__":
    unittest.main()
