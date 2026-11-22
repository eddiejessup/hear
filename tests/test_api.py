from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
from fastapi.testclient import TestClient

from hear.app import build_container, create_app
from hear.auth import AuthSettings, EnvReader, verify_token
from hear.debug import RuntimeDebugProvider
from hear.gemini import GeminiClient
from hear.geo import GeoService
from hear.models.config import AppConfig
from hear.models.events import BackendEventType, EventLevel, EventSource
from hear.paths import PROJECT_ROOT, AppPaths, resolve_data_dir
from tests.helpers import make_test_app


def test_verify_token_and_env_reader(tmp_path: Path) -> None:
    assert verify_token("secret123", "secret123")
    assert not verify_token("wrong", "secret123")
    assert not verify_token("", "secret123")

    dotenv = tmp_path / ".env"
    dotenv.write_text('HEAR_AUTH_TOKEN="from_dotenv"\n', encoding="utf-8")
    env = EnvReader(dotenv_path=dotenv, environ={"GEMINI_API_KEY": "from_env"})
    auth = AuthSettings(env)
    assert auth.expected_token() == "from_dotenv"
    assert env.get("GEMINI_API_KEY") == "from_env"


def test_resolve_data_dir_and_app_paths(tmp_path: Path) -> None:
    assert resolve_data_dir("") == PROJECT_ROOT / "data"
    expected = tmp_path.resolve()
    assert resolve_data_dir(str(tmp_path)) == expected
    with patch.dict("os.environ", {"HEAR_DATA_DIR": str(tmp_path)}):
        assert resolve_data_dir() == expected
    paths = AppPaths.from_env(data_dir=tmp_path)
    assert paths.db_path == expected / "hear.db"
    assert paths.audio_dir == expected / "audio"


def test_app_paths_empty_data_dir_env_reader_and_ensure_dirs(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    env_data = root / "from_env_reader"
    missing_static = root / "nonexistent_static"

    paths_empty = AppPaths.from_env(
        data_dir="",
        config_path=str(root / "custom.yaml"),
        project_root=root,
        static_dir=missing_static,
    )
    assert paths_empty.data_dir == root / "data"
    assert paths_empty.config_path == root / "custom.yaml"
    paths_empty.ensure_dirs()
    assert paths_empty.data_dir.is_dir()
    assert paths_empty.audio_dir.is_dir()
    assert not missing_static.exists()

    paths_from_env = AppPaths.from_env(
        project_root=root,
        env=EnvReader(environ={"HEAR_DATA_DIR": str(env_data)}),
    )
    assert paths_from_env.data_dir == env_data


def test_runtime_debug_provider_resolves_git_lazily(tmp_path: Path) -> None:
    paths = AppPaths.from_env(data_dir=tmp_path, project_root=tmp_path)
    env = EnvReader(environ={})
    with patch("hear.debug.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "abc1234\n"
        provider = RuntimeDebugProvider(paths, env)
        assert mock_run.call_count == 0

        info1 = provider.get_info()
        assert info1.git_commit == "abc1234"
        assert mock_run.call_count == 2

        # Subsequent calls reuse cached git metadata
        info2 = provider.get_info()
        assert info2.git_commit == "abc1234"
        assert mock_run.call_count == 2


def test_injected_http_client_not_replaced_or_closed(tmp_path: Path) -> None:
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.is_closed = True

    geo = GeoService(http_client=mock_http)
    assert geo._get_client() is mock_http
    asyncio.run(geo.close())
    mock_http.aclose.assert_not_awaited()

    gemini = GeminiClient(audio_dir=tmp_path, api_key="k", http_client=mock_http)
    assert gemini._get_client() is mock_http
    asyncio.run(gemini.close())
    mock_http.aclose.assert_not_awaited()


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
