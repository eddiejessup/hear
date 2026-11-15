from pydantic import Field

from hear.models.common import HearModel


class GeminiConfig(HearModel):
    model: str = "gemini-3.8-flash"
    temperature: float = 0.75
    enable_google_search: bool = True
    tts_enabled: bool = True
    tts_model: str = "gemini-3.8-flash-tts"
    tts_voice: str = "Charon"
    tts_style_prompt: str = ""


class TriggersConfig(HearModel):
    auto_trigger_enabled: bool = True
    min_distance_meters: float = 120.0
    min_cooldown_seconds: float = 90.0
    max_silence_seconds: float = 300.0
    min_drift_for_silence_trigger_meters: float = 15.0


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
