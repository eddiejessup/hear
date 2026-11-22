from __future__ import annotations

from pathlib import Path

from hear.auth import AuthSettings, EnvReader, verify_token


def test_verify_token_and_env_reader(tmp_path: Path) -> None:
    assert verify_token("secret123", "secret123")
    assert not verify_token("wrong", "secret123")
    assert not verify_token("", "secret123")

    dotenv = tmp_path / ".env"
    dotenv.write_text('HEAR_AUTH_TOKEN="from_dotenv"\n', encoding="utf-8")
    env = EnvReader(dotenv_path=dotenv, environ={"GEMINI_API_KEY": "from_env"})
    auth = AuthSettings(env)
    assert auth.expected_token() == "from_dotenv"
    assert env.get("GEMINI_API_KEY") == "from_env"
