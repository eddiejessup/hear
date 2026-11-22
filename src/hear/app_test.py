from __future__ import annotations

import re
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from hear.app import build_container, create_app
from hear.auth import EnvReader
from hear.models.config import AppConfig
from hear.models.events import BackendEventType, EventLevel, EventSource
from hear.paths import AppPaths
from hear.testing import make_test_app


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
