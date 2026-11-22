from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from hear.db import EventRecord, HearDB
from hear.models.config import AppConfig, GeminiConfig, LensConfig
from hear.models.events import EventLevel, EventSource
from hear.models.gemini import GeminiTextResult, GroundingSource
from hear.models.session import CurrentLocation
from hear.session import SessionManager, WalkSession
from hear.testing import FakeGeminiClient, make_geo


def test_session_rehydration_client_isolation_and_fts(db_store: HearDB) -> None:
    cfg = AppConfig(
        gemini=GeminiConfig(tts_enabled=False),
        lenses=[LensConfig(id="econ", name="Economics", enabled=True, prompt="Econ lens")],
    )
    geo = make_geo()
    fake_gemini = FakeGeminiClient(
        [
            GeminiTextResult(
                text="Cornhill sat at the heart of London's mercantile credit networks.",
                search_queries=["Cornhill London credit history"],
                sources=[GroundingSource(title="British History", uri="https://example.com/bh")],
            ),
            GeminiTextResult(
                text="The Royal Exchange was founded by Thomas Gresham in 1565.",
                search_queries=[],
                sources=[],
            ),
        ]
    )

    mgr1 = SessionManager(store=db_store, gemini=fake_gemini)
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
    mgr2 = SessionManager(store=db_store, gemini=fake_gemini)
    restored = mgr2.get("elmo")
    assert restored.session_id == original_session_id
    assert len(restored.ui_messages) == 3
    assert len(restored.gemini_contents) == 4
    assert restored.used_lens_ids == ["econ"]
    assert restored.last_trigger_location is not None
    assert restored.last_trigger_location.short_label == "Cornhill, City of London"

    # Guest client_id stays completely isolated
    guest = mgr2.get("guest_abc")
    assert guest.session_id != original_session_id
    assert guest.ui_messages == []

    # FTS5 search finds persisted narrations scoped by client_id
    hits = db_store.search_messages("Gresham", client_id="elmo")
    assert len(hits) == 1
    assert db_store.search_messages("Gresham", client_id="guest_abc") == []

    # Reset marks the previous session inactive so restart starts fresh
    restored.reset()
    assert restored.session_id != original_session_id
    mgr3 = SessionManager(store=db_store, gemini=fake_gemini)
    fresh = mgr3.get("elmo")
    assert fresh.ui_messages == []


def test_location_sample_downsampling(db_store: HearDB) -> None:
    session = WalkSession("elmo", store=db_store)
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

    assert asyncio.run(session.record_location_sample_if_needed(loc1))
    assert not asyncio.run(session.record_location_sample_if_needed(loc_same))
    assert asyncio.run(session.record_location_sample_if_needed(loc_moved))
    assert asyncio.run(session.record_location_sample_if_needed(loc_later))


def test_rehydration_clears_missing_audio_url_and_keeps_existing(
    tmp_path: Path, db_store: HearDB
) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "present.wav").write_bytes(b"RIFF")

    cfg = AppConfig(
        gemini=GeminiConfig(tts_enabled=True),
        lenses=[LensConfig(id="l1", name="L1", prompt="P1")],
    )
    geo = make_geo(short_label="Cornhill", full_address="Cornhill, London", neighborhood="City")
    fake_present = FakeGeminiClient(
        [GeminiTextResult(text="One", search_queries=[], sources=[])],
        audio_url="/api/audio/present.wav",
    )
    fake_missing = FakeGeminiClient(
        [GeminiTextResult(text="Two", search_queries=[], sources=[])],
        audio_url="/api/audio/missing.wav",
    )

    mgr1 = SessionManager(store=db_store, audio_dir=audio_dir, gemini=fake_present)
    sess = mgr1.get("elmo")
    asyncio.run(sess.execute_narration_turn(cfg, 51.5142, -0.0885, None, geo))
    sess._gemini = fake_missing
    asyncio.run(sess.execute_narration_turn(cfg, 51.5150, -0.0885, None, geo))

    mgr2 = SessionManager(store=db_store, audio_dir=audio_dir)
    restored = mgr2.get("elmo")
    assert len(restored.ui_messages) == 2
    assert getattr(restored.ui_messages[0], "audio_url", None) == "/api/audio/present.wav"
    assert getattr(restored.ui_messages[1], "audio_url", None) is None


def test_event_ts_normalization_and_payload_truncation(db_store: HearDB) -> None:
    now = time.time()
    db_store.record_events_batch(
        [
            EventRecord(
                ts=(now - 10.0) * 1000.0,  # JS milliseconds
                source=EventSource.FRONTEND,
                level=EventLevel.INFO,
                event_type="js_ms_event",
                client_id="elmo",
                payload={"ok": True},
            ),
            EventRecord(
                ts=now + 999999.0,  # Far future timestamp
                source=EventSource.FRONTEND,
                level=EventLevel.INFO,
                event_type="future_event",
                client_id="elmo",
                payload={"blob": "x" * 20000},
            ),
        ]
    )
    events = {e["event_type"]: e for e in db_store.get_recent_events(client_id="elmo")}
    assert events["js_ms_event"]["ts"] == pytest.approx(now - 10.0, abs=2.0)
    assert events["future_event"]["ts"] <= time.time() + 1.0
    assert events["future_event"]["payload"].get("_truncated") is True


def test_last_audio_ended_at_persists_without_creating_row(db_store: HearDB) -> None:
    # Updating a non-existent session should not insert a row
    db_store.update_last_audio_ended_at("missing_session", 1234.5)
    assert db_store.load_latest_active_session("elmo") is None

    # Updating an existing session persists last_audio_ended_at across reload
    cfg = AppConfig(
        gemini=GeminiConfig(tts_enabled=True),
        lenses=[LensConfig(id="l1", name="L1", prompt="P1")],
    )
    geo = make_geo(short_label="Cornhill", full_address="Cornhill, London", neighborhood="City")
    fake_gemini = FakeGeminiClient(
        [GeminiTextResult(text="Hi", search_queries=[], sources=[])],
        audio_url="/api/audio/clip.wav",
    )
    sess = WalkSession(client_id="elmo", store=db_store, gemini=fake_gemini)
    asyncio.run(sess.execute_narration_turn(cfg, 51.5142, -0.0885, None, geo))
    changed = sess.update_audio_state(
        lat=51.5142,
        lon=-0.0885,
        is_playing_audio=False,
        audio_just_ended=True,
        now=5000.0,
    )
    assert changed
    asyncio.run(sess.persist_last_audio_ended_at())

    reloaded = db_store.load_latest_active_session("elmo")
    assert reloaded is not None
    assert reloaded.last_audio_ended_at == pytest.approx(5000.0)
