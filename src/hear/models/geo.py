from hear.models.common import HearModel


class GeoContext(HearModel):
    lat: float
    lon: float
    short_label: str
    full_address: str
    road: str
    neighborhood: str
    city: str
    nearby_features: list[str]
