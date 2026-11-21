from __future__ import annotations

import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from hear.auth import EnvReader
from hear.models.session import DebugInfo
from hear.paths import AppPaths


def _detect_git_value(args: list[str], env_key: str, *, env: EnvReader, cwd: Path) -> str:
    env_val = env.get(env_key)
    if env_val:
        return env_val
    try:
        res = subprocess.run(
            ["git", *args],
            cwd=cwd,
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


class RuntimeDebugProvider:
    """Collects build/runtime metadata lazily per app instance (no import-time side effects)."""

    def __init__(self, paths: AppPaths, env: EnvReader) -> None:
        self._paths = paths
        self._env = env
        self._started_epoch = time.time()
        self._started_at_iso = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        self._git_commit: str | None = None
        self._git_commit_time: str | None = None
        self._build_time = env.get("HEAR_BUILD_TIME") or "local-dev"
        self._cloud_run_revision = env.get("K_REVISION") or "local"
        self._cloud_run_service = env.get("K_SERVICE") or "local"
        self._python_version = platform.python_version()
        self._data_dir = str(paths.data_dir)
        self._litestream_enabled = bool(env.get("HEAR_LITESTREAM_BUCKET"))

    def _ensure_git_metadata(self) -> tuple[str, str]:
        if self._git_commit is None:
            self._git_commit = _detect_git_value(
                ["rev-parse", "--short", "HEAD"],
                "HEAR_GIT_COMMIT",
                env=self._env,
                cwd=self._paths.project_root,
            )
        if self._git_commit_time is None:
            self._git_commit_time = _detect_git_value(
                ["show", "-s", "--format=%cI", "HEAD"],
                "HEAR_GIT_COMMIT_TIME",
                env=self._env,
                cwd=self._paths.project_root,
            )
        return self._git_commit, self._git_commit_time

    def get_info(self) -> DebugInfo:
        git_commit, git_commit_time = self._ensure_git_metadata()
        return DebugInfo(
            git_commit=git_commit,
            git_commit_time=git_commit_time,
            build_time=self._build_time,
            cloud_run_revision=self._cloud_run_revision,
            cloud_run_service=self._cloud_run_service,
            python_version=self._python_version,
            server_started_at=self._started_at_iso,
            server_uptime_seconds=int(time.time() - self._started_epoch),
            data_dir=self._data_dir,
            litestream_enabled=self._litestream_enabled,
        )
