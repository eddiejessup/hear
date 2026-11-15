from typing import Literal

from pydantic import Field

from hear.models.common import HearModel
from hear.models.gemini import GroundingSource


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
    is_paused_audio: bool = False


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


class DebugInfo(HearModel):
    git_commit: str
    git_commit_time: str
    build_time: str
    cloud_run_revision: str
    cloud_run_service: str
    python_version: str
    server_started_at: str
    server_uptime_seconds: int
