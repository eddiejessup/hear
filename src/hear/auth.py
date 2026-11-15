import os
import secrets
from pathlib import Path

from hear.paths import PROJECT_ROOT


class _DotEnvCache:
    """Reads and caches key-value pairs from .env based on file mtime."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._mtime: float = 0.0
        self._values: dict[str, str] = {}

    def get(self, name: str) -> str:
        val = os.environ.get(name, "").strip()
        if val:
            return val
        if not self.path.exists():
            return ""
        try:
            mtime = self.path.stat().st_mtime
            if mtime != self._mtime:
                parsed: dict[str, str] = {}
                for line in self.path.read_text(encoding="utf-8").splitlines():
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#") or "=" not in stripped:
                        continue
                    key, raw_val = stripped.split("=", 1)
                    parsed[key.strip()] = raw_val.strip().strip('"').strip("'")
                self._values = parsed
                self._mtime = mtime
        except OSError:
            return ""
        return self._values.get(name, "")


_dotenv_cache = _DotEnvCache(PROJECT_ROOT / ".env")


def read_env_var(name: str) -> str:
    return _dotenv_cache.get(name)


def get_expected_auth_token() -> str:
    return read_env_var("HEAR_AUTH_TOKEN")


def is_unauthenticated_allowed() -> bool:
    val = read_env_var("HEAR_ALLOW_UNAUTHENTICATED").lower()
    return val in {"1", "true", "yes"}


def verify_token(provided: str, expected: str) -> bool:
    if not provided or not expected:
        return False
    return secrets.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))
