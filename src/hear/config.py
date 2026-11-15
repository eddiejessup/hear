import logging
import os
import tempfile
from pathlib import Path

import yaml
from pydantic import ValidationError

from hear.models.config import AppConfig
from hear.paths import CONFIG_PATH

logger = logging.getLogger(__name__)


class ConfigManager:
    """Loads and hot-reloads config.yaml when modified on disk."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._cache: AppConfig | None = None
        self._mtime: float = 0.0

    def get(self) -> AppConfig:
        if not self.path.exists():
            if self._cache is None:
                self._cache = AppConfig()
            return self._cache
        try:
            mtime = self.path.stat().st_mtime
            if mtime != self._mtime or self._cache is None:
                with open(self.path, encoding="utf-8") as f:
                    loaded = yaml.safe_load(f)
                    self._cache = (
                        AppConfig.model_validate(loaded)
                        if isinstance(loaded, dict)
                        else AppConfig()
                    )
                self._mtime = mtime
        except (OSError, yaml.YAMLError, ValidationError) as exc:
            logger.warning("Failed to reload config.yaml, keeping previous config: %s", exc)
            if self._cache is None:
                self._cache = AppConfig()
        return self._cache

    def save(self, new_config: AppConfig) -> AppConfig:
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
        self._cache = new_config
        return self._cache


config_manager = ConfigManager(CONFIG_PATH)
