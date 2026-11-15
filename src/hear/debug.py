import os
import platform
import subprocess
import time
from datetime import UTC, datetime

from hear.models.session import DebugInfo
from hear.paths import PROJECT_ROOT

SERVER_START_EPOCH = time.time()
SERVER_STARTED_AT_ISO = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _detect_git_value(args: list[str], env_key: str) -> str:
    env_val = os.environ.get(env_key, "").strip()
    if env_val:
        return env_val
    try:
        res = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except OSError, subprocess.SubprocessError:
        pass
    return "unknown"


GIT_COMMIT = _detect_git_value(["rev-parse", "--short", "HEAD"], "HEAR_GIT_COMMIT")
GIT_COMMIT_TIME = _detect_git_value(["show", "-s", "--format=%cI", "HEAD"], "HEAR_GIT_COMMIT_TIME")
BUILD_TIME = os.environ.get("HEAR_BUILD_TIME", "").strip() or "local-dev"
CLOUD_RUN_REVISION = os.environ.get("K_REVISION", "").strip() or "local"
CLOUD_RUN_SERVICE = os.environ.get("K_SERVICE", "").strip() or "local"
PYTHON_VERSION = platform.python_version()


def get_debug_info() -> DebugInfo:
    return DebugInfo(
        git_commit=GIT_COMMIT,
        git_commit_time=GIT_COMMIT_TIME,
        build_time=BUILD_TIME,
        cloud_run_revision=CLOUD_RUN_REVISION,
        cloud_run_service=CLOUD_RUN_SERVICE,
        python_version=PYTHON_VERSION,
        server_started_at=SERVER_STARTED_AT_ISO,
        server_uptime_seconds=int(time.time() - SERVER_START_EPOCH),
    )
