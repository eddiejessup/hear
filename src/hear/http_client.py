from __future__ import annotations

from collections.abc import Mapping

import httpx


class ManagedAsyncClient:
    """Lazy httpx.AsyncClient wrapper that only owns and closes clients it creates."""

    def __init__(
        self,
        *,
        http_client: httpx.AsyncClient | None = None,
        timeout: float = 30.0,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self._client = http_client
        self._owns_client = http_client is None
        self._timeout = timeout
        self._headers = dict(headers) if headers is not None else None

    def get(self) -> httpx.AsyncClient:
        if not self._owns_client and self._client is not None:
            return self._client
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self._timeout, headers=self._headers)
            self._owns_client = True
        return self._client

    async def close(self) -> None:
        if self._owns_client and self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        if self._owns_client:
            self._client = None
