from pydantic import Field

from hear.models.common import HearModel

_MODEL_ID_PATTERN = r"^[a-zA-Z0-9._-]+$"


class GeminiConfig(HearModel):
    model: str = Field(default="gemini-3.8-flash", pattern=_MODEL_ID_PATTERN)
    temperature: float = Field(default=0.75, ge=0.0, le=2.0)
    enable_google_search: bool = True
    tts_enabled: bool = True
    tts_model: str = Field(default="gemini-3.8-flash-tts", pattern=_MODEL_ID_PATTERN)
    tts_voice: str = Field(default="Charon", min_length=1, max_length=64)
    tts_style_prompt: str = Field(default="", max_length=1000)


class TriggersConfig(HearModel):
    auto_trigger_enabled: bool = True
    min_distance_meters: float = Field(default=120.0, ge=0.0, le=10000.0)
    min_cooldown_seconds: float = Field(default=90.0, ge=0.0, le=86400.0)
    max_silence_seconds: float = Field(default=300.0, ge=0.0, le=86400.0)
    min_drift_for_silence_trigger_meters: float = Field(default=15.0, ge=0.0, le=10000.0)


class LensConfig(HearModel):
    id: str = ""
    name: str = "Historical Lens"
    enabled: bool = True
    prompt: str = ""


class AppConfig(HearModel):
    gemini: GeminiConfig = Field(default_factory=GeminiConfig)
    triggers: TriggersConfig = Field(default_factory=TriggersConfig)
    system_prompt: str = ""
    lenses: list[LensConfig] = Field(default_factory=list)
