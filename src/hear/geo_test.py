from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest

from hear.geo import GeoCache, GeoService, compass_bearing_label, haversine_meters
from hear.testing import make_geo


def test_haversine_meters() -> None:
    assert haversine_meters(51.5, -0.1, 51.5, -0.1) == pytest.approx(0.0)
    dist = haversine_meters(51.5142, -0.0885, 51.5152, -0.0885)
    assert 100.0 < dist < 125.0


def test_compass_bearing() -> None:
    assert compass_bearing_label(None) is None
    assert compass_bearing_label(-5.0) is None
    assert compass_bearing_label(365.0) is None
    assert compass_bearing_label(0.0) == "N (0°)"
    assert compass_bearing_label(359.8) == "N (0°)"
    assert compass_bearing_label(90.0) == "E (90°)"
    assert compass_bearing_label(225.0) == "SW (225°)"


def test_geo_cache_anchor_lru_and_ttl() -> None:
    cache = GeoCache(max_entries=2, ttl_seconds=60.0)
    base_ctx = make_geo()
    cache.set(51.5142, -0.0885, base_ctx)

    # Within 25m -> cache hit
    entry = cache.find_close_entry(51.51425, -0.0885)
    assert entry is not None
    assert entry.result.short_label == "Cornhill, City of London"

    # Updating within 25m preserves original geocoded anchor coordinates
    updated_ctx = base_ctx.model_copy(update={"short_label": "1 Cornhill, City of London"})
    cache.set(51.51425, -0.0885, updated_ctx)
    entry = cache.find_close_entry(51.5142, -0.0885)
    assert entry is not None
    assert entry.lat == pytest.approx(51.5142)
    assert entry.result.short_label == "1 Cornhill, City of London"

    # Add a second entry far away, then touch the first entry so it becomes MRU
    other_ctx = base_ctx.model_copy(update={"lat": 51.5200, "short_label": "Moorgate"})
    cache.set(51.5200, -0.0885, other_ctx)
    assert cache.find_close_entry(51.5142, -0.0885) is not None

    # Adding a third entry evicts the LRU (51.5200) and keeps the MRU (51.5142)
    third_ctx = base_ctx.model_copy(update={"lat": 51.5300, "short_label": "Angel"})
    cache.set(51.5300, -0.0885, third_ctx)
    assert cache.find_close_entry(51.5142, -0.0885) is not None
    assert cache.find_close_entry(51.5200, -0.0885) is None


def test_reverse_geocode_caches_nominatim_result() -> None:
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.get.return_value = httpx.Response(
        200,
        json={
            "display_name": "3 ter, Rue des Rosiers, Quartier Saint-Gervais, Paris",
            "address": {
                "house_number": "3 ter",
                "road": "Rue des Rosiers",
                "quarter": "Quartier Saint-Gervais",
                "city": "Paris",
            },
        },
    )
    geo = GeoService(http_client=mock_client)

    ctx1 = asyncio.run(geo.reverse_geocode(48.8566, 2.3604))
    assert ctx1.short_label == "3 ter Rue des Rosiers, Quartier Saint-Gervais, Paris"
    assert mock_client.get.call_count == 1

    # Nearby coordinate within 25m reuses cached result without another HTTP call
    ctx2 = asyncio.run(geo.reverse_geocode(48.85665, 2.3604))
    assert ctx2.short_label == ctx1.short_label
    assert mock_client.get.call_count == 1


def test_nominatim_failure_backoff() -> None:
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.get.side_effect = httpx.ConnectError("Connection refused")
    geo = GeoService(http_client=mock_client)

    res1 = asyncio.run(geo.fetch_nominatim(51.5142, -0.0885))
    assert not res1[5]
    assert mock_client.get.call_count == 1

    # Immediate second call during 15s backoff window skips HTTP call
    res2 = asyncio.run(geo.fetch_nominatim(51.5142, -0.0885))
    assert not res2[5]
    assert mock_client.get.call_count == 1
