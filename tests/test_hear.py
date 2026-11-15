import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from hear.app import app
from hear.auth import verify_token
from hear.gemini import pcm_to_wav_bytes, prune_old_audio_files
from hear.geo import GeoCache, compass_bearing_label, haversine_meters
from hear.models.config import AppConfig, LensConfig, TriggersConfig
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
        self.assertEqual(compass_bearing_label(0.0), "N (0°)")
        self.assertEqual(compass_bearing_label(90.0), "E (90°)")
        self.assertEqual(compass_bearing_label(225.0), "SW (225°)")

    def test_geo_cache_upgrade_and_ttl(self) -> None:
        cache = GeoCache(max_entries=4, ttl_seconds=60.0)
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

        # Upgrade with overpass features
        upgraded_ctx = base_ctx.model_copy(update={"nearby_features": ["Royal Exchange"]})
        cache.set(51.51425, -0.0885, upgraded_ctx, has_overpass=True)
        hit = cache.get_if_close(51.5142, -0.0885, require_overpass=True)
        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertEqual(hit.nearby_features, ["Royal Exchange"])


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

            # Duplicate /api/narrate while session.is_generating is True -> 409 Conflict
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
            finally:
                sess.is_generating = False


if __name__ == "__main__":
    unittest.main()
