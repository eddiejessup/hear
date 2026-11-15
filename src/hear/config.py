from pathlib import Path
from typing import Any

import yaml

from hear.paths import CONFIG_PATH


class ConfigManager:
    """Loads and hot-reloads config.yaml when modified on disk."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._cache: dict[str, Any] = {}
        self._mtime: float = 0.0

    def get(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._cache
        mtime = self.path.stat().st_mtime
        if mtime != self._mtime or not self._cache:
            with open(self.path, encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                self._cache = loaded if isinstance(loaded, dict) else {}
            self._mtime = mtime
        return self._cache

    def save(self, new_config: dict[str, Any]) -> dict[str, Any]:
        with open(self.path, "w", encoding="utf-8") as f:
            yaml.safe_dump(new_config, f, sort_keys=False, allow_unicode=True, width=100)
        self._mtime = self.path.stat().st_mtime
        self._cache = new_config
        return self._cache


config_manager = ConfigManager(CONFIG_PATH)
