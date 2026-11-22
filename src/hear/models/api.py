import math
from typing import Any

from pydantic import Field, field_validator

from hear.models.common import HearModel
from hear.models.config import AppConfig
from hear.models.events import EventLevel
from hear.models.session import (
    AssistantMessage,
    CurrentLocation,
    DebugInfo,
    TriggerLocation,
    TriggerStatus,
    UIMessage,
    UserChatMessage,
)


def _normalize_non_negative(value: Any) -> Any:
    if (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and (not math.isfinite(value) or value < 0.0)
    ):
        return None
    return value


def _normalize_heading(value: Any) -> Any:
    if (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and (not math.isfinite(value) or value < 0.0 or value > 360.0)
    ):
        return None
    return value


class PingRequest(HearModel):
    lat: float = Field(ge=-90.0, le=90.0)
    lon: float = Field(ge=-180.0, le=180.0)
    accuracy: float | None = Field(default=None, ge=0.0)
    heading: float | None = Field(default=None, ge=0.0, le=360.0)
    speed: float | None = Field(default=None, ge=0.0)
    is_playing_audio: bool = False
    is_paused_audio: bool = False
    audio_just_ended: bool = False

    @field_validator("accuracy", "speed", mode="before")
    @classmethod
    def _clean_non_negative(cls, value: Any) -> Any:
        return _normalize_non_negative(value)

    @field_validator("heading", mode="before")
    @classmethod
    def _clean_heading(cls, value: Any) -> Any:
        return _normalize_heading(value)


class NarrateRequest(HearModel):
    lat: float | None = Field(default=None, ge=-90.0, le=90.0)
    lon: float | None = Field(default=None, ge=-180.0, le=180.0)
    heading: float | None = Field(default=None, ge=0.0, le=360.0)
    lens_id: str | None = Field(default=None, max_length=128)
    custom_focus: str | None = Field(default=None, max_length=1000)
    reason: str | None = Field(default="manual", max_length=128)

    @field_validator("heading", mode="before")
    @classmethod
    def _clean_heading(cls, value: Any) -> Any:
        return _normalize_heading(value)


class ChatRequest(HearModel):
    message: str = Field(max_length=4000)
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


class ClientEventInput(HearModel):
    ts: float | None = None
    level: EventLevel = EventLevel.INFO
    session_id: str | None = Field(default=None, max_length=128)
    event_type: str = Field(min_length=1, max_length=64)
    payload: dict[str, Any] = Field(default_factory=dict)


class ClientEventsRequest(HearModel):
    events: list[ClientEventInput] = Field(default_factory=list, max_length=100)


class ClientEventsResponse(HearModel):
    accepted: int
