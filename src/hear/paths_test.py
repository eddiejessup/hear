from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from hear.auth import EnvReader
from hear.paths import PROJECT_ROOT, AppPaths, resolve_data_dir


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
