from pathlib import Path

import yaml

from hear.models.config import AppConfig
from hear.paths import CONFIG_PATH


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
        mtime = self.path.stat().st_mtime
        if mtime != self._mtime or self._cache is None:
            with open(self.path, encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                self._cache = (
                    AppConfig.model_validate(loaded) if isinstance(loaded, dict) else AppConfig()
                )
            self._mtime = mtime
        return self._cache

    def save(self, new_config: AppConfig) -> AppConfig:
        with open(self.path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                new_config.model_dump(),
                f,
                sort_keys=False,
                allow_unicode=True,
                width=100,
            )
        self._mtime = self.path.stat().st_mtime
        self._cache = new_config
        return self._cache


config_manager = ConfigManager(CONFIG_PATH)
