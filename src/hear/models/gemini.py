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


class GeminiUsage(HearModel):
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    thought_tokens: int | None = None
    cached_tokens: int | None = None
    tool_use_prompt_tokens: int | None = None
    total_tokens: int | None = None


class GeminiTextResult(HearModel):
    text: str
    search_queries: list[str]
    sources: list[GroundingSource]
    usage: GeminiUsage | None = None
