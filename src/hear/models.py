from typing import TypedDict

from pydantic import BaseModel


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


class LensConfig(TypedDict, total=False):
    id: str
    name: str
    enabled: bool
    prompt: str


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
