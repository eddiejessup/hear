from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch
from urllib.parse import quote

from fastapi.testclient import TestClient

from hear.app import build_container, create_app
from hear.auth import EnvReader
from hear.gemini import pcm_to_wav_bytes
from hear.models.config import AppConfig
from hear.models.events import BackendEventType, EventLevel, EventSource
from hear.paths import AppPaths
from hear.testing import make_geo, make_test_app


def test_api_coordinate_validation_and_duplicate_narrate(tmp_path: Path) -> None:
    test_app = make_test_app(tmp_path)
    with TestClient(test_app) as client:
        # Unauthorized without token
        r_unauth = client.get("/api/state")
        assert r_unauth.status_code == 401

        headers = {"x-hear-token": "test-token", "x-hear-client-id": "testclient"}
        # Out-of-bounds latitude rejected by Pydantic
        r_bad_coords = client.post(
            "/api/ping",
            json={"lat": 95.0, "lon": 0.0},
            headers=headers,
        )
        assert r_bad_coords.status_code == 422

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
            assert r_conflict.status_code == 409

            r_chat_conflict = client.post(
                "/api/chat",
                json={"message": "Tell me more", "lat": 51.5142, "lon": -0.0885},
                headers=headers,
            )
            assert r_chat_conflict.status_code == 409
        finally:
            sess.is_generating = False

        # Authenticated 4xx errors persisted to SQLite; unauthenticated 401 probes skipped
        events = container.db.get_recent_events(limit=20)
        error_statuses = [
            ev["payload"].get("status_code")
            for ev in events
            if ev["event_type"] == BackendEventType.HTTP_ERROR
        ]
        assert 401 not in error_statuses
        assert 409 in error_statuses
        assert 422 in error_statuses


def test_ping_endpoint_accepts_negative_gps_sentinels(tmp_path: Path) -> None:
    container = build_container(
        data_dir=tmp_path,
        config_path=tmp_path / "config.yaml",
        env=EnvReader(environ={}),
        allow_unauthenticated=True,
    )
    with patch.object(container.geo, "reverse_geocode", AsyncMock(return_value=make_geo())):
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
            assert resp.status_code == 200
            loc = resp.json()["location"]
            assert loc["accuracy"] is None
            assert loc["heading"] is None
            assert loc["speed"] is None


def test_cookie_auth_and_client_id_unquote_url_encoded_values(tmp_path: Path) -> None:
    special_token = "abc+def/123=45@secret:token"
    test_app = make_test_app(tmp_path / "special", auth_token=special_token)
    audio_file = test_app.state.container.paths.audio_dir / "clip.wav"
    audio_file.write_bytes(pcm_to_wav_bytes(b"\x00\x00" * 10))

    with TestClient(test_app) as client:
        client.cookies.set("hear_token", quote(special_token, safe=""))
        client.cookies.set("hear_client_id", quote("elmo/phone", safe=""))

        state_resp = client.get("/api/state")
        assert state_resp.status_code == 200
        assert "elmophone" in test_app.state.container.session_manager._sessions

        audio_resp = client.get("/api/audio/clip.wav")
        assert audio_resp.status_code == 200

    # Raw header containing literal %2B is NOT unquoted
    literal_percent_token = "tok%2Ben"
    raw_app = make_test_app(tmp_path / "literal", auth_token=literal_percent_token)
    with TestClient(raw_app) as client:
        ok_resp = client.get(
            "/api/state",
            headers={"x-hear-token": literal_percent_token},
        )
        assert ok_resp.status_code == 200


def test_http_events_preserve_session_id_across_middleware(tmp_path: Path) -> None:
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
            logged_requests.append((str(payload.get("path", "")), resolved_sid, resolved_cid))
        return resolved_sid, resolved_cid

    with (
        patch.object(container.events, "_emit", side_effect=_capture_emit),
        TestClient(test_app, raise_server_exceptions=False) as client,
    ):
        headers = {"x-hear-client-id": "elmo"}
        state_resp = client.get("/api/state", headers=headers)
        assert state_resp.status_code == 200
        first_sid = state_resp.json()["session_id"]

        # HTTP_REQUEST for /api/state carries session_id & client_id via ContextVar
        assert ("/api/state", first_sid, "elmo") in logged_requests

        # HTTP_ERROR for /api/narrate (400: no coords yet) persists with session_id
        err_resp = client.post("/api/narrate", json={}, headers=headers)
        assert err_resp.status_code == 400

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
            assert crash_resp.status_code == 500

        # /api/session/reset updates ContextVar session_id for SESSION_RESET & HTTP_REQUEST
        reset_resp = client.post("/api/session/reset", headers=headers)
        assert reset_resp.status_code == 200
        new_sid = reset_resp.json()["session_id"]
        assert new_sid != first_sid
        assert ("/api/session/reset", new_sid, "elmo") in logged_requests

        events = container.db.get_recent_events(client_id="elmo", limit=20)
        http_errors = [ev for ev in events if ev["event_type"] == BackendEventType.HTTP_ERROR]
        assert len(http_errors) == 1
        assert http_errors[0]["session_id"] == first_sid

        unhandled = [
            ev for ev in events if ev["event_type"] == BackendEventType.HTTP_UNHANDLED_ERROR
        ]
        assert len(unhandled) == 1
        assert unhandled[0]["session_id"] == first_sid

        resets = [ev for ev in events if ev["event_type"] == BackendEventType.SESSION_RESET]
        assert len(resets) == 1
        assert resets[0]["session_id"] == new_sid


def test_client_events_endpoint_and_pruning(tmp_path: Path) -> None:
    test_app = make_test_app(tmp_path)
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
        assert resp.status_code == 200
        assert resp.json()["accepted"] == 2

        store = test_app.state.container.db
        events = store.get_recent_events(client_id="elmo", limit=10)
        assert len(events) == 2
        assert events[0]["source"] == EventSource.FRONTEND

        # Insert an expired event and verify pruning removes only the old row
        store.record_event(
            source=EventSource.BACKEND,
            level=EventLevel.INFO,
            event_type="stale_event",
            ts=time.time() - 40 * 86400.0,
        )
        pruned = store.prune_old_events(max_age_days=30)
        assert pruned == 1
        assert len(store.get_recent_events(client_id="elmo", limit=10)) == 2


def test_create_app_isolated_config_and_env(tmp_path: Path) -> None:
    cfg_path = tmp_path.resolve() / "isolated_config.yaml"
    with patch.dict("os.environ", {"HEAR_AUTH_TOKEN": "host-secret"}):
        test_app = make_test_app(
            tmp_path.resolve(),
            config_path=cfg_path,
            auth_token=None,
            allow_unauthenticated=True,
        )
        with TestClient(test_app) as client:
            state_resp = client.get("/api/state")
            assert state_resp.status_code == 200
            assert not state_resp.json()["has_api_key"]

            put_resp = client.put(
                "/api/config",
                json=AppConfig(system_prompt="Isolated prompt").model_dump(),
            )
            assert put_resp.status_code == 200
            assert cfg_path.exists()
            assert "Isolated prompt" in cfg_path.read_text(encoding="utf-8")


def test_lifespan_shutdown_closes_db_even_if_geo_close_raises(tmp_path: Path) -> None:
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
            assert client.get("/api/config").status_code == 200
            assert container.db._conn is not None
        geo_close.assert_awaited_once()
        gemini_close.assert_awaited_once()
        assert container.db._conn is None


def test_static_frontend_modules_served(tmp_path: Path) -> None:
    test_app = make_test_app(tmp_path)
    with TestClient(test_app) as client:
        index_resp = client.get("/")
        assert index_resp.status_code == 200
        assert "/static/app.css" in index_resp.text
        assert "/static/app.js" in index_resp.text

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
            assert resp.status_code == 200, f"Failed to serve {asset}"


def test_static_frontend_dom_ids_and_imports_integrity() -> None:
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
            assert target.is_file(), f"Broken relative JS import '{rel_import}' in {js_file.name}"
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
                assert imported_name in target_exports, (
                    f"Missing export '{imported_name}' in {target.name} "
                    f"(imported by {js_file.name})"
                )
        for dom_id in re.findall(r'getElementById\(\s*["\']([^"\']+)["\']\s*\)', src):
            assert dom_id in html_ids, (
                f"Missing DOM id='{dom_id}' in index.html (referenced by {js_file.name})"
            )
