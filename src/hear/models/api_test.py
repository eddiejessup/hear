from __future__ import annotations

import pytest
from pydantic import ValidationError

from hear.models.api import NarrateRequest, PingRequest


def test_ping_and_narrate_normalize_negative_and_non_finite_gps_sentinels() -> None:
    ping = PingRequest(
        lat=51.5142,
        lon=-0.0885,
        accuracy=-1.0,
        heading=-1.0,
        speed=-1.0,
    )
    assert ping.accuracy is None
    assert ping.heading is None
    assert ping.speed is None

    ping_non_finite = PingRequest(
        lat=51.5142,
        lon=-0.0885,
        accuracy=float("nan"),
        heading=365.0,
        speed=float("inf"),
    )
    assert ping_non_finite.accuracy is None
    assert ping_non_finite.heading is None
    assert ping_non_finite.speed is None

    ping_valid = PingRequest(
        lat=51.5142,
        lon=-0.0885,
        accuracy=12.5,
        heading=0.0,
        speed=0.0,
    )
    assert ping_valid.accuracy == 12.5
    assert ping_valid.heading == 0.0
    assert ping_valid.speed == 0.0

    narrate = NarrateRequest(lat=51.5142, lon=-0.0885, heading=-1.0)
    assert narrate.heading is None

    with pytest.raises(ValidationError):
        PingRequest.model_validate({"lat": 51.5, "lon": -0.1, "heading": "invalid"})
