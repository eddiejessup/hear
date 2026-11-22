from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import httpx

from hear.gemini import GeminiClient
from hear.geo import GeoService


def test_injected_http_client_not_replaced_or_closed(tmp_path: Path) -> None:
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.is_closed = True

    geo = GeoService(http_client=mock_http)
    assert geo._get_client() is mock_http
    asyncio.run(geo.close())
    mock_http.aclose.assert_not_awaited()

    gemini = GeminiClient(audio_dir=tmp_path, api_key="k", http_client=mock_http)
    assert gemini._get_client() is mock_http
    asyncio.run(gemini.close())
    mock_http.aclose.assert_not_awaited()
