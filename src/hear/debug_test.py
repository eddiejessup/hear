from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from hear.auth import EnvReader
from hear.debug import RuntimeDebugProvider
from hear.paths import AppPaths


def test_runtime_debug_provider_resolves_git_lazily(tmp_path: Path) -> None:
    paths = AppPaths.from_env(data_dir=tmp_path, project_root=tmp_path)
    env = EnvReader(environ={})
    with patch("hear.debug.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "abc1234\n"
        provider = RuntimeDebugProvider(paths, env)
        assert mock_run.call_count == 0

        info1 = provider.get_info()
        assert info1.git_commit == "abc1234"
        assert mock_run.call_count == 2

        # Subsequent calls reuse cached git metadata
        info2 = provider.get_info()
        assert info2.git_commit == "abc1234"
        assert mock_run.call_count == 2
