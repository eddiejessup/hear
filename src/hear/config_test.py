from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import patch

from hear.config import ConfigManager
from hear.db import HearDB
from hear.models.config import AppConfig


def test_config_manager_atomic_save_and_corrupt_yaml_fallback(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    mgr = ConfigManager(cfg_path)
    initial = AppConfig(system_prompt="Initial prompt")
    mgr.save(initial)
    assert mgr.get().system_prompt == "Initial prompt"

    # Write invalid YAML to disk and bump mtime -> falls back to last valid config
    cfg_path.write_text(":::invalid_yaml: [unclosed", encoding="utf-8")
    os.utime(cfg_path, (time.time() + 10.0, time.time() + 10.0))
    assert mgr.get().system_prompt == "Initial prompt"


def test_config_override_persists_in_sqlite_across_restarts(
    tmp_path: Path, db_store: HearDB
) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("system_prompt: From YAML\n", encoding="utf-8")

    mgr1 = ConfigManager(cfg_path, store=db_store)
    assert mgr1.get().system_prompt == "From YAML"
    mgr1.save(AppConfig(system_prompt="Saved via UI"))

    # Simulate fresh container where baked-in config.yaml is older than the DB override
    older_ts = time.time() - 3600.0
    cfg_path.write_text("system_prompt: Baked In Default\n", encoding="utf-8")
    os.utime(cfg_path, (older_ts, older_ts))

    mgr2 = ConfigManager(cfg_path, store=db_store)
    assert mgr2.get().system_prompt == "Saved via UI"


def test_config_manager_readonly_yaml_save_not_clobbered_by_get(
    tmp_path: Path, db_store: HearDB
) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("system_prompt: Disk prompt\n", encoding="utf-8")
    mgr = ConfigManager(cfg_path, store=db_store)
    with patch("os.replace", side_effect=OSError("read-only filesystem")):
        mgr.save(AppConfig(system_prompt="SQLite prompt"))
    assert mgr.get().system_prompt == "SQLite prompt"
