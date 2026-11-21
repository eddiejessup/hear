import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = PACKAGE_DIR.parent.parent.resolve()


def resolve_data_dir(raw: str | None = None) -> Path:
    value = os.environ.get("HEAR_DATA_DIR", "").strip() if raw is None else raw.strip()
    if value:
        return Path(value).expanduser().resolve()
    return PROJECT_ROOT / "data"


CONFIG_PATH = PROJECT_ROOT / "config.yaml"
STATIC_DIR = PACKAGE_DIR / "static"
DATA_DIR = resolve_data_dir()
DB_PATH = DATA_DIR / "hear.db"
SESSIONS_DIR = DATA_DIR / "sessions"
AUDIO_DIR = DATA_DIR / "audio"


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
