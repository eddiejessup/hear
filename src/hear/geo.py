import math
import time

import httpx

from hear.models.geo import GeoContext


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
    if heading is None or math.isnan(heading) or heading < 0:
        return None
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = round(heading / 45.0) % 8
    return f"{dirs[idx]} ({int(round(heading))}°)"


class GeoCache:
    def __init__(self) -> None:
        self.last_lat: float | None = None
        self.last_lon: float | None = None
        self.last_result: GeoContext | None = None
        self.last_has_overpass: bool = False
        self.last_time: float = 0.0

    def get_if_close(
        self, lat: float, lon: float, require_overpass: bool, max_dist_m: float = 25.0
    ) -> GeoContext | None:
        if (
            self.last_result is not None
            and self.last_lat is not None
            and self.last_lon is not None
            and (time.time() - self.last_time) < 300
            and (not require_overpass or self.last_has_overpass)
            and haversine_meters(lat, lon, self.last_lat, self.last_lon) <= max_dist_m
        ):
            return self.last_result
        return None

    def set(self, lat: float, lon: float, result: GeoContext, has_overpass: bool) -> None:
        self.last_lat = lat
        self.last_lon = lon
        self.last_result = result
        self.last_has_overpass = has_overpass
        self.last_time = time.time()


geo_cache = GeoCache()


async def reverse_geocode_and_context(
    lat: float, lon: float, include_overpass: bool = True
) -> GeoContext:
    """Resolve (lat, lon) into street/neighborhood names and nearby OSM features."""
    cached = geo_cache.get_if_close(lat, lon, require_overpass=include_overpass)
    if cached is not None:
        return cached

    headers = {"User-Agent": "HearWalkingCompanion/0.1 (personal-history-agent)"}
    short_label = f"{lat:.5f}, {lon:.5f}"
    full_address = short_label
    neighborhood = ""
    city = ""
    road = ""
    nearby_features: list[str] = []

    async with httpx.AsyncClient(timeout=6.0, headers=headers) as client:
        # 1. Nominatim reverse geocode
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
            )
            if resp.status_code == 200:
                data = resp.json()
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
        except httpx.HTTPError:
            pass

        # 2. Optional Overpass lookup for nearby named historic/landmark features within 150m
        if include_overpass:
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
            try:
                ov_resp = await client.post(
                    "https://overpass-api.de/api/interpreter",
                    data={"data": overpass_query},
                    timeout=4.5,
                )
                if ov_resp.status_code == 200:
                    elements = ov_resp.json().get("elements", [])
                    seen: set[str] = set()
                    for el in elements:
                        tags = el.get("tags") or {}
                        name = (
                            tags.get("name") or tags.get("historic:name") or tags.get("inscription")
                        )
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
            except httpx.HTTPError:
                pass

    result = GeoContext(
        lat=lat,
        lon=lon,
        short_label=short_label,
        full_address=full_address,
        road=road,
        neighborhood=neighborhood,
        city=city,
        nearby_features=nearby_features[:10],
    )
    geo_cache.set(lat, lon, result, has_overpass=include_overpass)
    return result
