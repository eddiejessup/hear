import asyncio
import math
import time
from dataclasses import dataclass

import httpx

from hear.logging_utils import log_event_async
from hear.models.events import BackendEventType, EventLevel
from hear.models.geo import GeoContext

_USER_AGENT = "HearWalkingCompanion/0.1 (personal-history-agent)"
_NOMINATIM_FAILURE_BACKOFF_SECONDS = 15.0
_http_client: httpx.AsyncClient | None = None
_nominatim_backoff_until: float = 0.0


def get_geo_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            timeout=6.0,
            headers={"User-Agent": _USER_AGENT},
        )
    return _http_client


async def close_geo_http_client() -> None:
    global _http_client
    if _http_client is not None and not _http_client.is_closed:
        await _http_client.aclose()
    _http_client = None


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
    has_overpass: bool
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
        has_overpass: bool,
        max_dist_m: float = 25.0,
    ) -> GeoContext:
        now = time.time()
        existing = self.find_close_entry(lat, lon, max_dist_m=max_dist_m, now=now)
        if existing is not None:
            if existing.has_overpass and not has_overpass:
                result = result.model_copy(
                    update={"nearby_features": existing.result.nearby_features}
                )
                has_overpass = True
            # Keep existing.lat and existing.lon anchored at the originally geocoded point
            existing.result = result
            existing.has_overpass = has_overpass
            existing.timestamp = now
            return result

        self._entries.append(
            _CacheEntry(
                lat=lat,
                lon=lon,
                result=result,
                has_overpass=has_overpass,
                timestamp=now,
            )
        )
        if len(self._entries) > self.max_entries:
            self._entries.pop(0)
        return result


geo_cache = GeoCache()


async def _fetch_nominatim(
    client: httpx.AsyncClient, lat: float, lon: float
) -> tuple[str, str, str, str, str, bool]:
    global _nominatim_backoff_until
    short_label = f"{lat:.5f}, {lon:.5f}"
    full_address = short_label
    road = ""
    neighborhood = ""
    city = ""
    ok = False

    if time.time() < _nominatim_backoff_until:
        return short_label, full_address, road, neighborhood, city, False

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
            await log_event_async(
                BackendEventType.GEO_NOMINATIM_OK,
                level=EventLevel.DEBUG,
                persist=False,
                latency_ms=latency_ms,
                short_label=short_label,
            )
        else:
            await log_event_async(
                BackendEventType.GEO_NOMINATIM_ERROR,
                level=EventLevel.WARN,
                status_code=resp.status_code,
                latency_ms=latency_ms,
                lat=round(lat, 5),
                lon=round(lon, 5),
            )
    except (httpx.HTTPError, ValueError) as exc:
        latency_ms = int((time.monotonic() - t0) * 1000)
        await log_event_async(
            BackendEventType.GEO_NOMINATIM_ERROR,
            level=EventLevel.WARN,
            error=str(exc),
            error_type=type(exc).__name__,
            latency_ms=latency_ms,
            lat=round(lat, 5),
            lon=round(lon, 5),
        )

    if not ok:
        _nominatim_backoff_until = time.time() + _NOMINATIM_FAILURE_BACKOFF_SECONDS

    return short_label, full_address, road, neighborhood, city, ok


async def _fetch_overpass(
    client: httpx.AsyncClient, lat: float, lon: float
) -> tuple[list[str], bool]:
    overpass_query = f"""
    [out:json][timeout:4];
    (
      node(around:150,{lat:.6f},{lon:.6f})["historic"];
      way(around:150,{lat:.6f},{lon:.6f})["historic"];
      node(around:150,{lat:.6f},{lon:.6f})["tourism"~"attraction|museum|artwork|viewpoint"];
      way(around:150,{lat:.6f},{lon:.6f})["tourism"~"attraction|museum"];
      node(around:150,{lat:.6f},{lon:.6f})["amenity"~"place_of_worship|theatre|library|townhall|marketplace|pub"];
      way(around:150,{lat:.6f},{lon:.6f})["amenity"~"place_of_worship|theatre|library|townhall|marketplace"];
      way(around:150,{lat:.6f},{lon:.6f})["building"]["name"];
      node(around:150,{lat:.6f},{lon:.6f})["memorial"];
    );
    out tags center 15;
    """
    nearby_features: list[str] = []
    ok = False
    t0 = time.monotonic()
    try:
        ov_resp = await client.post(
            "https://overpass-api.de/api/interpreter",
            data={"data": overpass_query},
            timeout=4.5,
        )
        latency_ms = int((time.monotonic() - t0) * 1000)
        if ov_resp.status_code == 200:
            elements = ov_resp.json().get("elements", [])
            ok = True
            seen: set[str] = set()
            for el in elements:
                tags = el.get("tags") or {}
                name = tags.get("name") or tags.get("historic:name") or tags.get("inscription")
                if not name or not isinstance(name, str) or name in seen:
                    continue
                seen.add(name)
                kind = str(
                    tags.get("historic")
                    or tags.get("memorial")
                    or tags.get("tourism")
                    or tags.get("amenity")
                    or tags.get("building")
                    or "feature"
                )
                extra: list[str] = []
                if tags.get("start_date"):
                    extra.append(f"built {tags['start_date']}")
                if tags.get("architect"):
                    extra.append(f"architect: {tags['architect']}")
                if tags.get("wikipedia"):
                    extra.append(f"wiki: {tags['wikipedia']}")
                suffix = f" ({kind}{', ' + ', '.join(extra) if extra else ''})"
                nearby_features.append(f"{name}{suffix}")
            await log_event_async(
                BackendEventType.GEO_OVERPASS_OK,
                level=EventLevel.DEBUG,
                persist=False,
                latency_ms=latency_ms,
                feature_count=len(nearby_features),
            )
        else:
            await log_event_async(
                BackendEventType.GEO_OVERPASS_ERROR,
                level=EventLevel.WARN,
                status_code=ov_resp.status_code,
                latency_ms=latency_ms,
                lat=round(lat, 5),
                lon=round(lon, 5),
            )
    except (httpx.HTTPError, ValueError) as exc:
        latency_ms = int((time.monotonic() - t0) * 1000)
        await log_event_async(
            BackendEventType.GEO_OVERPASS_ERROR,
            level=EventLevel.WARN,
            error=str(exc),
            error_type=type(exc).__name__,
            latency_ms=latency_ms,
            lat=round(lat, 5),
            lon=round(lon, 5),
        )
    return nearby_features[:10], ok


async def reverse_geocode_and_context(
    lat: float, lon: float, include_overpass: bool = True
) -> GeoContext:
    """Resolve (lat, lon) into street/neighborhood names and nearby OSM features."""
    entry = geo_cache.find_close_entry(lat, lon)
    if entry is not None and (not include_overpass or entry.has_overpass):
        return entry.result

    client = get_geo_http_client()

    if entry is not None:
        nearby_features, ov_ok = await _fetch_overpass(client, lat, lon)
        upgraded = entry.result.model_copy(update={"nearby_features": nearby_features})
        if ov_ok:
            return geo_cache.set(lat, lon, upgraded, has_overpass=True)
        return upgraded

    if include_overpass:
        (
            (short_label, full_address, road, neighborhood, city, nom_ok),
            (nearby_features, ov_ok),
        ) = await asyncio.gather(
            _fetch_nominatim(client, lat, lon),
            _fetch_overpass(client, lat, lon),
        )
    else:
        short_label, full_address, road, neighborhood, city, nom_ok = await _fetch_nominatim(
            client, lat, lon
        )
        nearby_features = []
        ov_ok = False

    result = GeoContext(
        lat=lat,
        lon=lon,
        short_label=short_label,
        full_address=full_address,
        road=road,
        neighborhood=neighborhood,
        city=city,
        nearby_features=nearby_features,
    )
    if nom_ok:
        return geo_cache.set(lat, lon, result, has_overpass=include_overpass and ov_ok)
    return result
