from __future__ import annotations

import os
import secrets
from collections.abc import Mapping
from pathlib import Path


class EnvReader:
    """Reads env vars from an environment mapping with mtime-cached .env file fallback."""

    def __init__(
        self,
        dotenv_path: Path | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self.dotenv_path = dotenv_path
        self._environ = environ
        self._mtime: float = 0.0
        self._values: dict[str, str] = {}

    def get(self, name: str) -> str:
        source = self._environ if self._environ is not None else os.environ
        val = source.get(name, "").strip()
        if val:
            return val
        if self.dotenv_path is None or not self.dotenv_path.exists():
            return ""
        try:
            mtime = self.dotenv_path.stat().st_mtime
            if mtime != self._mtime:
                parsed: dict[str, str] = {}
                for line in self.dotenv_path.read_text(encoding="utf-8").splitlines():
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


class AuthSettings:
    """Resolves expected auth token and unauthenticated fallback without global state."""

    def __init__(
        self,
        env: EnvReader | None = None,
        *,
        auth_token: str | None = None,
        allow_unauthenticated: bool | None = None,
    ) -> None:
        self._env = env
        self._auth_token_override = auth_token
        self._allow_unauth_override = allow_unauthenticated

    def expected_token(self) -> str:
        if self._auth_token_override is not None:
            return self._auth_token_override.strip()
        return self._env.get("HEAR_AUTH_TOKEN") if self._env is not None else ""

    def is_unauthenticated_allowed(self) -> bool:
        if self._allow_unauth_override is not None:
            return self._allow_unauth_override
        if self._env is None:
            return False
        return self._env.get("HEAR_ALLOW_UNAUTHENTICATED").lower() in {"1", "true", "yes"}


def verify_token(provided: str, expected: str) -> bool:
    if not provided or not expected:
        return False
    return secrets.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))
