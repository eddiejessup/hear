import os
import secrets

from hear.paths import PROJECT_ROOT


def _read_env_var(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if val:
        return val
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith(f"{name}="):
                return stripped.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def get_expected_auth_token() -> str:
    return _read_env_var("HEAR_AUTH_TOKEN")


def is_unauthenticated_allowed() -> bool:
    val = _read_env_var("HEAR_ALLOW_UNAUTHENTICATED").lower()
    return val in {"1", "true", "yes"}


def verify_token(provided: str, expected: str) -> bool:
    if not provided or not expected:
        return False
    return secrets.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))
