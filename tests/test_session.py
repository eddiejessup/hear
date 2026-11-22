from __future__ import annotations

import asyncio
import time

from hear.models.config import AppConfig, GeminiConfig
from hear.models.gemini import GeminiTextResult
from hear.models.session import CurrentLocation, TriggerLocation
from hear.session import SessionManager, WalkSession, sanitize_client_id
from tests.helpers import FakeGeminiClient


def test_sanitize_client_id() -> None:
    assert sanitize_client_id(None) == "default"
    assert sanitize_client_id("  ../bad/id!@# ") == "badid"
    assert sanitize_client_id("pixel_9_pro-1") == "pixel_9_pro-1"


def test_sample_lens_without_replacement_and_reset(walk_config: AppConfig) -> None:
    session = WalkSession("test")
    first = session.sample_lens(walk_config)
    second = session.sample_lens(walk_config)
    assert first.id != second.id
    assert "off" not in session.used_lens_ids

    # Pool exhausted -> resets and samples again
    third = session.sample_lens(walk_config)
    assert third.id in {"econ", "pol"}
    assert len(session.used_lens_ids) == 1


def test_trigger_heuristics(walk_config: AppConfig) -> None:
    session = WalkSession("test")
    # No location yet -> should not trigger
    status = session.compute_trigger_status(walk_config)
    assert not status.should_trigger

    # First GPS lock -> initial_location trigger
    now = time.time()
    session.current_location = CurrentLocation(
        lat=51.5142,
        lon=-0.0885,
        short_label="Start",
        full_address="Start",
        timestamp=now,
    )
    status = session.compute_trigger_status(walk_config)
    assert status.should_trigger
    assert status.trigger_reason == "initial_location"

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
    status_cooldown = session.compute_trigger_status(walk_config)
    assert not status_cooldown.should_trigger
    assert status_cooldown.cooldown_remaining_s > 0

    # Once cooldown expires (100s ago) and moved >= 120m -> distance trigger
    session.last_audio_ended_at = now - 100.0
    session.last_trigger_time = now - 110.0
    status_ready = session.compute_trigger_status(walk_config)
    assert status_ready.should_trigger
    assert status_ready.trigger_reason is not None
    assert status_ready.trigger_reason.startswith("distance")


def test_initial_location_respects_grace_window_after_opening_chat(walk_config: AppConfig) -> None:
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
    status = session.compute_trigger_status(walk_config)
    assert not status.should_trigger
    # Uses short 15s grace window instead of full 90s min_cooldown_seconds
    assert 0 < status.cooldown_remaining_s <= 15.0

    # After 16s (well before 90s), initial_location is allowed to fire
    session.last_trigger_time = now - 16.0
    session.last_audio_ended_at = now - 16.0
    status_after_grace = session.compute_trigger_status(walk_config)
    assert status_after_grace.should_trigger
    assert status_after_grace.trigger_reason == "initial_location"


def test_pause_hold_and_time_distance_expiry(walk_config: AppConfig) -> None:
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
    status_recent = session.compute_trigger_status(walk_config)
    assert not status_recent.should_trigger
    assert status_recent.is_paused_audio

    # Paused for 200s (>180s), but still standing near pause spot (<150m) -> hold stays active
    session.paused_at = now - 200.0
    session.paused_location = (moved_lat, moved_lon)
    status_nearby = session.compute_trigger_status(walk_config)
    assert not status_nearby.should_trigger
    assert status_nearby.is_paused_audio

    # Paused for 200s (>180s) AND walked ~200m (>150m) from pause spot -> hold expires!
    session.paused_location = (start_lat, start_lon)
    status_expired = session.compute_trigger_status(walk_config)
    assert not status_expired.is_paused_audio
    assert status_expired.should_trigger


def test_session_manager_eviction() -> None:
    mgr = SessionManager(max_sessions=2)
    s1 = mgr.get("c1")
    s1.last_active_at = 100.0
    s2 = mgr.get("c2")
    s2.last_active_at = 200.0
    mgr.get("c3")
    assert "c1" not in mgr._sessions
    assert "c2" in mgr._sessions
    assert "c3" in mgr._sessions


def test_walk_session_forwards_thinking_level() -> None:
    fake_gemini = FakeGeminiClient(
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
    assert len(fake_gemini.text_calls) == 1
    assert fake_gemini.text_calls[0]["model"] == "gemini-4-hydrogen"
    assert fake_gemini.text_calls[0]["thinking_level"] == "high"


def test_same_second_multiple_resets_produce_unique_session_ids() -> None:
    session = WalkSession(client_id="elmo")
    seen = {session.session_id}
    for _ in range(5):
        session.reset(persist_deactivate=False)
        assert session.session_id not in seen
        seen.add(session.session_id)
