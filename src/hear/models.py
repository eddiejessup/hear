from typing import TypedDict

from pydantic import BaseModel, Field


class GeoContext(TypedDict):
    lat: float
    lon: float
    short_label: str
    full_address: str
    road: str
    neighborhood: str
    city: str
    nearby_features: list[str]


class GroundingSource(TypedDict):
    title: str
    uri: str


class GeminiTextResult(TypedDict):
    text: str
    search_queries: list[str]
    sources: list[GroundingSource]


class GeminiConfig(BaseModel):
    model: str = "gemini-3.8-flash"
    temperature: float = 0.75
    enable_google_search: bool = True
    tts_enabled: bool = True
    tts_model: str = "gemini-3.8-flash-tts"
    tts_voice: str = "Charon"
    tts_style_prompt: str = ""


class TriggersConfig(BaseModel):
    auto_trigger_enabled: bool = True
    min_distance_meters: float = 120.0
    min_cooldown_seconds: float = 90.0
    max_silence_seconds: float = 300.0
    min_drift_for_silence_trigger_meters: float = 15.0


class LensConfig(BaseModel):
    id: str = ""
    name: str = "Historical Lens"
    enabled: bool = True
    prompt: str = ""


class AppConfig(BaseModel):
    gemini: GeminiConfig = Field(default_factory=GeminiConfig)
    triggers: TriggersConfig = Field(default_factory=TriggersConfig)
    system_prompt: str = ""
    lenses: list[LensConfig] = Field(default_factory=list)


class PingRequest(BaseModel):
    lat: float
    lon: float
    accuracy: float | None = None
    heading: float | None = None
    speed: float | None = None
    is_playing_audio: bool = False
    audio_just_ended: bool = False


class NarrateRequest(BaseModel):
    lat: float | None = None
    lon: float | None = None
    heading: float | None = None
    lens_id: str | None = None
    custom_focus: str | None = None
    reason: str | None = "manual"


class ChatRequest(BaseModel):
    message: str
    lat: float | None = None
    lon: float | None = None
