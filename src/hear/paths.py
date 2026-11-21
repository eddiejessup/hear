from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from hear.auth import EnvReader

PACKAGE_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = PACKAGE_DIR.parent.parent.resolve()


def resolve_data_dir(
    raw: Path | str | None = None,
    *,
    project_root: Path = PROJECT_ROOT,
    env: EnvReader | None = None,
) -> Path:
    resolved_root = project_root.expanduser().resolve()
    if isinstance(raw, Path):
        return raw.expanduser().resolve()
    if raw is None:
        value = (
            env.get("HEAR_DATA_DIR")
            if env is not None
            else os.environ.get("HEAR_DATA_DIR", "").strip()
        )
    else:
        value = raw.strip()
    if value:
        return Path(value).expanduser().resolve()
    return resolved_root / "data"


@dataclass(frozen=True)
class AppPaths:
    project_root: Path
    config_path: Path
    static_dir: Path
    data_dir: Path

    @classmethod
    def from_env(
        cls,
        *,
        data_dir: Path | str | None = None,
        config_path: Path | str | None = None,
        project_root: Path = PROJECT_ROOT,
        static_dir: Path = PACKAGE_DIR / "static",
        env: EnvReader | None = None,
    ) -> AppPaths:
        resolved_root = project_root.expanduser().resolve()
        resolved_static = static_dir.expanduser().resolve()
        resolved_data = resolve_data_dir(data_dir, project_root=resolved_root, env=env)
        if isinstance(config_path, Path):
            resolved_config = config_path.expanduser().resolve()
        elif config_path is not None and config_path.strip():
            resolved_config = Path(config_path.strip()).expanduser().resolve()
        else:
            resolved_config = resolved_root / "config.yaml"
        return cls(
            project_root=resolved_root,
            config_path=resolved_config,
            static_dir=resolved_static,
            data_dir=resolved_data,
        )

    @property
    def db_path(self) -> Path:
        return self.data_dir / "hear.db"

    @property
    def audio_dir(self) -> Path:
        return self.data_dir / "audio"

    @property
    def dotenv_path(self) -> Path:
        return self.project_root / ".env"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
