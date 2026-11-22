from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from hear.app import create_app
from hear.auth import EnvReader
from hear.models.config import AppConfig, LensConfig, TriggersConfig
from hear.models.gemini import GeminiContent, GeminiTextResult
from hear.models.geo import GeoContext


class FakeGeminiClient:
    def __init__(self, responses: list[GeminiTextResult], audio_url: str | None = None) -> None:
        self._responses = list(responses)
        self._audio_url = audio_url
        self.text_calls: list[dict[str, object]] = []

    async def generate_text(
        self,
        contents: list[GeminiContent],
        system_prompt: str,
        model: str,
        temperature: float = 0.75,
        enable_google_search: bool = True,
        thinking_level: str | None = None,
    ) -> GeminiTextResult:
        self.text_calls.append(
            {
                "model": model,
                "temperature": temperature,
                "enable_google_search": enable_google_search,
                "thinking_level": thinking_level,
            }
        )
        return self._responses.pop(0)

    async def generate_tts(
        self,
        text: str,
        tts_model: str = "gemini-3.8-flash-tts",
        voice_name: str = "Charon",
        style_prompt: str = "",
    ) -> str | None:
        return self._audio_url


def make_geo(
    *,
    lat: float = 51.5142,
    lon: float = -0.0885,
    short_label: str = "Cornhill, City of London",
    full_address: str = "Cornhill, London, UK",
    road: str = "Cornhill",
    neighborhood: str = "City of London",
    city: str = "London",
) -> GeoContext:
    return GeoContext(
        lat=lat,
        lon=lon,
        short_label=short_label,
        full_address=full_address,
        road=road,
        neighborhood=neighborhood,
        city=city,
    )


def make_walk_config() -> AppConfig:
    return AppConfig(
        triggers=TriggersConfig(
            auto_trigger_enabled=True,
            min_distance_meters=120.0,
            min_cooldown_seconds=90.0,
            max_silence_seconds=300.0,
            min_drift_for_silence_trigger_meters=15.0,
        ),
        lenses=[
            LensConfig(id="econ", name="Economics", enabled=True, prompt="Econ prompt"),
            LensConfig(id="pol", name="Politics", enabled=True, prompt="Pol prompt"),
            LensConfig(id="off", name="Disabled", enabled=False, prompt="Off prompt"),
        ],
    )


def make_test_app(
    tmp_path: Path,
    *,
    auth_token: str | None = "test-token",
    allow_unauthenticated: bool | None = None,
    config_path: Path | None = None,
) -> FastAPI:
    return create_app(
        data_dir=tmp_path,
        config_path=config_path or (tmp_path / "config.yaml"),
        env=EnvReader(environ={}),
        auth_token=auth_token,
        allow_unauthenticated=allow_unauthenticated,
        configure_logs=False,
    )
