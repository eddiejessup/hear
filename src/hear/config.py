from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path

import yaml
from pydantic import ValidationError

from hear.db import HearDB
from hear.models.config import AppConfig

logger = logging.getLogger(__name__)


class ConfigManager:
    """Loads/hot-reloads config.yaml and optionally persists UI overrides in SQLite."""

    def __init__(self, path: Path, store: HearDB | None = None) -> None:
        self.path = path
        self._store = store
        self._cache: AppConfig | None = None
        self._mtime: float = 0.0
        self._db_loaded: bool = False

    def _load_file_config(self) -> tuple[AppConfig, float]:
        if not self.path.exists():
            return AppConfig(), 0.0
        mtime = self.path.stat().st_mtime
        with open(self.path, encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            cfg = AppConfig.model_validate(loaded) if isinstance(loaded, dict) else AppConfig()
        return cfg, mtime

    def get(self) -> AppConfig:
        file_mtime = self.path.stat().st_mtime if self.path.exists() else 0.0

        # On first read, prefer a persisted SQLite override if it is newer than config.yaml
        # (so UI saves survive Cloud Run restarts, while local config.yaml edits still win).
        if not self._db_loaded and self._store is not None:
            self._db_loaded = True
            try:
                override = self._store.load_config_override()
                if override is not None:
                    override_cfg, updated_at = override
                    if updated_at >= file_mtime:
                        self._cache = override_cfg
                        self._mtime = file_mtime
                        return self._cache
            except (ValidationError, OSError, Exception) as exc:  # noqa: BLE001
                logger.warning("Failed to load config override from SQLite: %s", exc)

        if not self.path.exists():
            if self._cache is None:
                self._cache = AppConfig()
            return self._cache

        try:
            if file_mtime != self._mtime or self._cache is None:
                self._cache, self._mtime = self._load_file_config()
        except (OSError, yaml.YAMLError, ValidationError) as exc:
            logger.warning("Failed to reload config.yaml, keeping previous config: %s", exc)
            if self._cache is None:
                self._cache = AppConfig()
        return self._cache

    def save(self, new_config: AppConfig) -> AppConfig:
        now = time.time()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(
                dir=self.path.parent, prefix=".config_", suffix=".yaml.tmp", text=True
            )
            tmp_path = Path(tmp_name)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    yaml.safe_dump(
                        new_config.model_dump(),
                        f,
                        sort_keys=False,
                        allow_unicode=True,
                        width=100,
                    )
                os.replace(tmp_path, self.path)
            except BaseException:
                tmp_path.unlink(missing_ok=True)
                raise
            self._mtime = self.path.stat().st_mtime
        except OSError as exc:
            if self._store is None:
                raise
            logger.warning("Failed to write config.yaml (continuing with SQLite save): %s", exc)

        if self._store is not None:
            # Ensure DB timestamp is >= file mtime so restart favors the saved config
            self._store.save_config_override(new_config, updated_at=max(now, self._mtime + 0.001))
            self._db_loaded = True

        self._cache = new_config
        return self._cache
