from pathlib import Path

PACKAGE_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = PACKAGE_DIR.parent.parent.resolve()

CONFIG_PATH = PROJECT_ROOT / "config.yaml"
STATIC_DIR = PACKAGE_DIR / "static"
SESSIONS_DIR = PROJECT_ROOT / "sessions"
AUDIO_DIR = SESSIONS_DIR / "audio"


def ensure_dirs() -> None:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
