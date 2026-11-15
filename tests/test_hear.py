import asyncio
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

from hear.app import app
from hear.auth import verify_token
from hear.config import ConfigManager
from hear.gemini import call_gemini_text, pcm_to_wav_bytes, prune_old_audio_files
from hear.geo import (
    GeoCache,
    _fetch_nominatim,
    compass_bearing_label,
    haversine_meters,
)
from hear.models.config import AppConfig, GeminiConfig, LensConfig, TriggersConfig
from hear.models.gemini import GeminiContent, GeminiPart
from hear.models.geo import GeoContext
from hear.models.session import CurrentLocation, TriggerLocation
from hear.session import SessionManager, WalkSession, sanitize_client_id


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

        # Within 25m without requiring overpass -> cache hit
        self.assertIsNotNone(cache.get_if_close(51.51425, -0.0885, require_overpass=False))
        # Requiring overpass when entry doesn't have it -> cache miss
        self.assertIsNone(cache.get_if_close(51.51425, -0.0885, require_overpass=True))

        # Upgrade with overpass features preserves original geocoded anchor coordinates
        upgraded_ctx = base_ctx.model_copy(update={"nearby_features": ["Royal Exchange"]})
        cache.set(51.51425, -0.0885, upgraded_ctx, has_overpass=True)
        entry = cache.find_close_entry(51.5142, -0.0885)
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertAlmostEqual(entry.lat, 51.5142)
        self.assertEqual(entry.result.nearby_features, ["Royal Exchange"])

        # Add a second entry far away, then touch the first entry so it becomes MRU
        other_ctx = base_ctx.model_copy(update={"lat": 51.5200, "short_label": "Moorgate"})
        cache.set(51.5200, -0.0885, other_ctx, has_overpass=True)
        self.assertIsNotNone(cache.get_if_close(51.5142, -0.0885, require_overpass=True))

        # Adding a third entry evicts the LRU (51.5200) and keeps the MRU (51.5142)
        third_ctx = base_ctx.model_copy(update={"lat": 51.5300, "short_label": "Angel"})
        cache.set(51.5300, -0.0885, third_ctx, has_overpass=True)
        self.assertIsNotNone(cache.get_if_close(51.5142, -0.0885, require_overpass=True))
        self.assertIsNone(cache.get_if_close(51.5200, -0.0885, require_overpass=True))

    def test_nominatim_failure_backoff(self) -> None:
        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.ConnectError("Connection refused")
        with patch("hear.geo._nominatim_backoff_until", 0.0):
            res1 = asyncio.run(_fetch_nominatim(mock_client, 51.5142, -0.0885))
            self.assertFalse(res1[5])
            self.assertEqual(mock_client.get.call_count, 1)

            # Immediate second call during 15s backoff window skips HTTP call
            res2 = asyncio.run(_fetch_nominatim(mock_client, 51.5142, -0.0885))
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
                # Set distinct mtimes
                mtime = 1000.0 + i
                import os

                os.utime(f, (mtime, mtime))

            with patch("hear.gemini.AUDIO_DIR", tmp_path):
                prune_old_audio_files(max_files=3)

            remaining = sorted(p.name for p in tmp_path.glob("*.wav"))
            self.assertEqual(remaining, ["clip_2.wav", "clip_3.wav", "clip_4.wav"])

    def test_verify_token(self) -> None:
        self.assertTrue(verify_token("secret123", "secret123"))
        self.assertFalse(verify_token("wrong", "secret123"))
        self.assertFalse(verify_token("", "secret123"))

    def test_api_coordinate_validation_and_duplicate_narrate(self) -> None:
        client = TestClient(app)
        with patch("hear.app.get_expected_auth_token", return_value="test-token"):
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
            from hear.app import session_manager

            sess = session_manager.get("testclient")
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

    def test_empty_gemini_response_raises_502(self) -> None:
        fake_resp = httpx.Response(
            200,
            json={"candidates": [{"finishReason": "SAFETY", "content": {"parts": []}}]},
        )
        with (
            patch("hear.gemini.get_api_key", return_value="fake-key"),
            patch("hear.gemini._post_gemini_with_retry", new=AsyncMock(return_value=fake_resp)),
        ):
            with self.assertRaises(HTTPException) as ctx:
                asyncio.run(
                    call_gemini_text(
                        contents=[GeminiContent(role="user", parts=[GeminiPart(text="Hello")])],
                        system_prompt="System",
                        model="gemini-3.8-flash",
                    )
                )
            self.assertEqual(ctx.exception.status_code, 502)
            self.assertIn("SAFETY", str(ctx.exception.detail))

    def test_config_manager_atomic_save_and_corrupt_yaml_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "config.yaml"
            mgr = ConfigManager(cfg_path)
            initial = AppConfig(system_prompt="Initial prompt")
            mgr.save(initial)
            self.assertEqual(mgr.get().system_prompt, "Initial prompt")

            # Write invalid YAML to disk and bump mtime -> falls back to last valid config
            cfg_path.write_text(":::invalid_yaml: [unclosed", encoding="utf-8")
            import os

            os.utime(cfg_path, (time.time() + 10.0, time.time() + 10.0))
            self.assertEqual(mgr.get().system_prompt, "Initial prompt")

    def test_gemini_config_model_id_validation(self) -> None:
        with self.assertRaises(ValidationError):
            GeminiConfig(model="../bad/model")


if __name__ == "__main__":
    unittest.main()
