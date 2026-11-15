from pydantic import Field

from hear.models.common import HearModel
from hear.models.config import AppConfig
from hear.models.session import (
    AssistantMessage,
    CurrentLocation,
    DebugInfo,
    TriggerLocation,
    TriggerStatus,
    UIMessage,
    UserChatMessage,
)


class PingRequest(HearModel):
    lat: float = Field(ge=-90.0, le=90.0)
    lon: float = Field(ge=-180.0, le=180.0)
    accuracy: float | None = Field(default=None, ge=0.0)
    heading: float | None = None
    speed: float | None = None
    is_playing_audio: bool = False
    audio_just_ended: bool = False


class NarrateRequest(HearModel):
    lat: float | None = Field(default=None, ge=-90.0, le=90.0)
    lon: float | None = Field(default=None, ge=-180.0, le=180.0)
    heading: float | None = None
    lens_id: str | None = None
    custom_focus: str | None = None
    reason: str | None = "manual"


class ChatRequest(HearModel):
    message: str
    lat: float | None = Field(default=None, ge=-90.0, le=90.0)
    lon: float | None = Field(default=None, ge=-180.0, le=180.0)


class StateResponse(HearModel):
    session_id: str
    has_api_key: bool
    current_location: CurrentLocation | None
    last_trigger_location: TriggerLocation | None
    messages: list[UIMessage]
    used_lens_ids: list[str]
    trigger_status: TriggerStatus
    config: AppConfig
    debug_info: DebugInfo


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
