from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class HearModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        validate_default=True,
        ser_json_inf_nan="constants",
    )


class GeoContext(HearModel):
    lat: float
    lon: float
    short_label: str
    full_address: str
    road: str
    neighborhood: str
    city: str
    nearby_features: list[str]


class GroundingSource(HearModel):
    title: str
    uri: str


class GeminiPart(HearModel):
    text: str


class GeminiContent(HearModel):
    role: Literal["user", "model"]
    parts: list[GeminiPart]


class GeminiTextResult(HearModel):
    text: str
    search_queries: list[str]
    sources: list[GroundingSource]


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


class CurrentLocation(HearModel):
    lat: float
    lon: float
    accuracy: float | None = None
    heading: float | None = None
    speed: float | None = None
    short_label: str
    full_address: str
    timestamp: float


class TriggerLocation(HearModel):
    lat: float
    lon: float
    short_label: str
    timestamp: float


class TriggerStatus(HearModel):
    auto_trigger_enabled: bool
    should_trigger: bool
    trigger_reason: str | None = None
    distance_moved_m: float | None = None
    min_distance_m: float
    silence_elapsed_s: float | None = None
    cooldown_remaining_s: float
    min_cooldown_s: float
    max_silence_s: float
    is_generating: bool
    is_playing_audio: bool


class MessageLocation(HearModel):
    lat: float
    lon: float
    short_label: str
    nearby_features: list[str] = Field(default_factory=list)


class LensSummary(HearModel):
    id: str
    name: str


class UserChatMessage(HearModel):
    id: str
    role: Literal["user"] = "user"
    type: Literal["chat"] = "chat"
    timestamp: float
    location_label: str | None = None
    text: str


class AssistantMessage(HearModel):
    id: str
    role: Literal["assistant"] = "assistant"
    type: Literal["narration", "chat_reply"]
    timestamp: float
    location: MessageLocation | None = None
    lens: LensSummary | None = None
    reason: str | None = None
    model: str
    text: str
    audio_url: str | None = None
    sources: list[GroundingSource] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list)


type UIMessage = UserChatMessage | AssistantMessage


class SessionTranscript(HearModel):
    session_id: str
    started_at: float
    messages: list[UIMessage]


class PingRequest(HearModel):
    lat: float
    lon: float
    accuracy: float | None = None
    heading: float | None = None
    speed: float | None = None
    is_playing_audio: bool = False
    audio_just_ended: bool = False


class NarrateRequest(HearModel):
    lat: float | None = None
    lon: float | None = None
    heading: float | None = None
    lens_id: str | None = None
    custom_focus: str | None = None
    reason: str | None = "manual"


class ChatRequest(HearModel):
    message: str
    lat: float | None = None
    lon: float | None = None


class StateResponse(HearModel):
    session_id: str
    has_api_key: bool
    current_location: CurrentLocation | None
    last_trigger_location: TriggerLocation | None
    messages: list[UIMessage]
    used_lens_ids: list[str]
    trigger_status: TriggerStatus
    config: AppConfig


class PingResponse(HearModel):
    location: CurrentLocation
    trigger_status: TriggerStatus


class NarrateResponse(HearModel):
    message: AssistantMessage
    trigger_status: TriggerStatus


class ChatResponse(HearModel):
    user_message: UserChatMessage
    assistant_message: AssistantMessage
    trigger_status: TriggerStatus


class UpdateConfigResponse(HearModel):
    config: AppConfig
    trigger_status: TriggerStatus


class ResetSessionResponse(HearModel):
    session_id: str
    trigger_status: TriggerStatus
