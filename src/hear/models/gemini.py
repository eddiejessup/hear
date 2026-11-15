from typing import Literal

from hear.models.common import HearModel


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
