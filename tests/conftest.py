from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest

from hear.db import HearDB
from hear.models.config import AppConfig
from tests.helpers import make_walk_config


@pytest.fixture
def walk_config() -> AppConfig:
    return make_walk_config()


@pytest.fixture
def db_store(tmp_path: Path) -> Generator[HearDB]:
    store = HearDB(tmp_path / "hear.db")
    store.init()
    try:
        yield store
    finally:
        store.close()
