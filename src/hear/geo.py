from __future__ import annotations

import math
import time
from dataclasses import dataclass

import httpx

from hear.http_client import ManagedAsyncClient
from hear.logging_utils import EventLogger
from hear.models.events import BackendEventType, EventLevel
from hear.models.geo import GeoContext

_USER_AGENT = "HearWalkingCompanion/0.1 (personal-history-agent)"
_NOMINATIM_FAILURE_BACKOFF_SECONDS = 15.0


def haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance in meters between two (lat, lon) points."""
    r = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * r * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def compass_bearing_label(heading: float | None) -> str | None:
    if heading is None or math.isnan(heading) or heading < 0 or heading > 360:
        return None
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = round(heading / 45.0) % 8
    deg = int(round(heading)) % 360
    return f"{dirs[idx]} ({deg}°)"


@dataclass
class _CacheEntry:
    lat: float
    lon: float
    result: GeoContext
    timestamp: float


class GeoCache:
    def __init__(self, max_entries: int = 16, ttl_seconds: float = 300.0) -> None:
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._entries: list[_CacheEntry] = []

    def _prune(self, now: float) -> None:
        self._entries = [e for e in self._entries if (now - e.timestamp) < self.ttl_seconds]

    def find_close_entry(
        self,
        lat: float,
        lon: float,
        max_dist_m: float = 25.0,
        *,
        now: float | None = None,
    ) -> _CacheEntry | None:
        ts = now if now is not None else time.time()
        self._prune(ts)
        for idx in range(len(self._entries) - 1, -1, -1):
            entry = self._entries[idx]
            if haversine_meters(lat, lon, entry.lat, entry.lon) <= max_dist_m:
                if idx != len(self._entries) - 1:
                    self._entries.pop(idx)
                    self._entries.append(entry)
                return entry
        return None

    def set(
        self,
        lat: float,
        lon: float,
        result: GeoContext,
        max_dist_m: float = 25.0,
    ) -> GeoContext:
        now = time.time()
        existing = self.find_close_entry(lat, lon, max_dist_m=max_dist_m, now=now)
        if existing is not None:
            # Keep existing.lat and existing.lon anchored at the originally geocoded point
            existing.result = result
            existing.timestamp = now
            return result

        self._entries.append(
            _CacheEntry(
                lat=lat,
                lon=lon,
                result=result,
                timestamp=now,
            )
        )
        if len(self._entries) > self.max_entries:
            self._entries.pop(0)
        return result


class GeoService:
    """Stateful reverse-geocoding service with cache, backoff, and HTTP pool."""

    def __init__(
        self,
        *,
        cache: GeoCache | None = None,
        event_logger: EventLogger | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.cache = cache if cache is not None else GeoCache()
        self._events = event_logger if event_logger is not None else EventLogger(store=None)
        self._http = ManagedAsyncClient(
            http_client=http_client,
            timeout=6.0,
            headers={"User-Agent": _USER_AGENT},
        )
        self._nominatim_backoff_until: float = 0.0

    def _get_client(self) -> httpx.AsyncClient:
        return self._http.get()

    async def close(self) -> None:
        await self._http.close()

    async def fetch_nominatim(self, lat: float, lon: float) -> tuple[str, str, str, str, str, bool]:
        short_label = f"{lat:.5f}, {lon:.5f}"
        full_address = short_label
        road = ""
        neighborhood = ""
        city = ""
        ok = False

        if time.time() < self._nominatim_backoff_until:
            return short_label, full_address, road, neighborhood, city, False

        client = self._get_client()
        t0 = time.monotonic()
        try:
            resp = await client.get(
                "https://nominatim.openstreetmap.org/reverse",
                params={
                    "format": "jsonv2",
                    "lat": f"{lat:.6f}",
                    "lon": f"{lon:.6f}",
                    "zoom": "18",
                    "addressdetails": "1",
                },
                timeout=6.0,
            )
            latency_ms = int((time.monotonic() - t0) * 1000)
            if resp.status_code == 200:
                data = resp.json()
                ok = True
                full_address = str(data.get("display_name") or full_address)
                addr = data.get("address") or {}
                road = str(
                    addr.get("road")
                    or addr.get("pedestrian")
                    or addr.get("footway")
                    or addr.get("path")
                    or addr.get("square")
                    or data.get("name")
                    or ""
                )
                house_number = str(addr.get("house_number") or "")
                neighborhood = str(
                    addr.get("neighbourhood")
                    or addr.get("quarter")
                    or addr.get("suburb")
                    or addr.get("borough")
                    or addr.get("city_district")
                    or ""
                )
                city = str(
                    addr.get("city")
                    or addr.get("town")
                    or addr.get("village")
                    or addr.get("municipality")
                    or addr.get("county")
                    or ""
                )
                parts: list[str] = []
                if road:
                    parts.append(f"{house_number} {road}".strip())
                if neighborhood and neighborhood != road:
                    parts.append(neighborhood)
                if city and city != neighborhood:
                    parts.append(city)
                if parts:
                    short_label = ", ".join(parts)
                await self._events.log_async(
                    BackendEventType.GEO_NOMINATIM_OK,
                    level=EventLevel.DEBUG,
                    persist=False,
                    latency_ms=latency_ms,
                    short_label=short_label,
                )
            else:
                await self._events.log_async(
                    BackendEventType.GEO_NOMINATIM_ERROR,
                    level=EventLevel.WARN,
                    status_code=resp.status_code,
                    latency_ms=latency_ms,
                    lat=round(lat, 5),
                    lon=round(lon, 5),
                )
        except (httpx.HTTPError, ValueError) as exc:
            latency_ms = int((time.monotonic() - t0) * 1000)
            await self._events.log_async(
                BackendEventType.GEO_NOMINATIM_ERROR,
                level=EventLevel.WARN,
                error=str(exc),
                error_type=type(exc).__name__,
                latency_ms=latency_ms,
                lat=round(lat, 5),
                lon=round(lon, 5),
            )

        if not ok:
            self._nominatim_backoff_until = time.time() + _NOMINATIM_FAILURE_BACKOFF_SECONDS

        return short_label, full_address, road, neighborhood, city, ok

    async def reverse_geocode(self, lat: float, lon: float) -> GeoContext:
        """Resolve (lat, lon) into street and neighborhood names via Nominatim."""
        entry = self.cache.find_close_entry(lat, lon)
        if entry is not None:
            return entry.result

        short_label, full_address, road, neighborhood, city, nom_ok = await self.fetch_nominatim(
            lat, lon
        )
        result = GeoContext(
            lat=lat,
            lon=lon,
            short_label=short_label,
            full_address=full_address,
            road=road,
            neighborhood=neighborhood,
            city=city,
        )
        if nom_ok:
            return self.cache.set(lat, lon, result)
        return result
