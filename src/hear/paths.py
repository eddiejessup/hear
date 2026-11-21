from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PACKAGE_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = PACKAGE_DIR.parent.parent.resolve()


def resolve_data_dir(raw: str | None = None, *, project_root: Path = PROJECT_ROOT) -> Path:
    value = os.environ.get("HEAR_DATA_DIR", "").strip() if raw is None else raw.strip()
    if value:
        return Path(value).expanduser().resolve()
    return project_root / "data"


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
        config_path: Path | None = None,
        project_root: Path = PROJECT_ROOT,
        static_dir: Path = PACKAGE_DIR / "static",
    ) -> AppPaths:
        resolved_data = (
            Path(data_dir).expanduser().resolve()
            if data_dir is not None
            else resolve_data_dir(project_root=project_root)
        )
        resolved_config = (
            config_path.expanduser().resolve()
            if config_path is not None
            else project_root / "config.yaml"
        )
        return cls(
            project_root=project_root,
            config_path=resolved_config,
            static_dir=static_dir,
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
        self.static_dir.mkdir(parents=True, exist_ok=True)
