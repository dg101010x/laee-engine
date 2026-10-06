"""Distances, travel estimators (haversine circuity and OSRM) and spatial cells (H3 / geohash)."""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Mapping

import numpy as np
import pytest

from laee.geo.distance import (
    CIRCUITY_FACTORS,
    EARTH_MEAN_RADIUS_KM,
    WGS84_EQUATORIAL_RADIUS_KM,
    HaversineCircuity,
    OsrmError,
    OsrmTravel,
    TravelEstimator,
    calibrate_circuity,
    haversine_km,
    parse_osrm_table,
)
from laee.geo.index import (
    cell_distance,
    cell_for,
    geohash_cell_area_km2,
    geohash_decode,
    geohash_encode,
    geohash_precision_for_resolution,
    index_backend,
    neighbors,
)

NAIROBI_CBD = (-1.2864, 36.8172)
WESTLANDS = (-1.2676, 36.8108)


# --- haversine -------------------------------------------------------------------------------


def test_one_degree_of_longitude_on_the_equator() -> None:
    # The reference 111.32 km is the WGS84 equatorial value; the spherical model with the IUGG
    # mean radius gives 111.195 km, about 0.11% short, which is within haversine's ~0.5% error.
    assert float(haversine_km(0.0, 0.0, 0.0, 1.0)) == pytest.approx(111.32, rel=2e-3)
    equatorial = haversine_km(0.0, 0.0, 0.0, 1.0, radius_km=WGS84_EQUATORIAL_RADIUS_KM)
    assert float(equatorial) == pytest.approx(111.3195, abs=1e-3)


def test_nairobi_cbd_to_westlands() -> None:
    # dlat 0.0188 deg ~ 2.090 km north, dlon 0.0064 deg ~ 0.711 km west -> ~2.208 km.
    d = float(haversine_km(*NAIROBI_CBD, *WESTLANDS))
    assert d == pytest.approx(2.208, abs=0.01)
    assert float(haversine_km(*WESTLANDS, *NAIROBI_CBD)) == pytest.approx(d, rel=1e-12)


def test_haversine_is_vectorised_and_broadcasts() -> None:
    lats = np.array([[0.0, 0.0], [1.0, -1.0]])
    lons = np.array([[0.0, 1.0], [0.0, 0.0]])
    d = haversine_km(0.0, 0.0, lats, lons)
    assert d.shape == (2, 2)
    assert d[0, 0] == 0.0
    assert d[1, 0] == pytest.approx(d[1, 1])
    assert d[0, 1] == pytest.approx(2 * math.pi * EARTH_MEAN_RADIUS_KM / 360)


def test_haversine_antipodes_and_dateline() -> None:
    half_circumference = math.pi * EARTH_MEAN_RADIUS_KM
    assert float(haversine_km(0.0, 0.0, 0.0, 180.0)) == pytest.approx(half_circumference)
    assert float(haversine_km(0.0, 179.5, 0.0, -179.5)) == pytest.approx(111.195, abs=1e-3)


@pytest.mark.parametrize(
    ("lat", "lon"), [(91.0, 0.0), (0.0, 180.5), (float("nan"), 0.0), (0.0, float("inf"))]
)
def test_haversine_rejects_invalid_coordinates(lat: float, lon: float) -> None:
    with pytest.raises(ValueError):
        haversine_km(lat, lon, 0.0, 0.0)


# --- circuity fallback -----------------------------------------------------------------------


def test_circuity_defaults_lie_in_the_documented_band_and_rise_as_density_falls() -> None:
    factors = [CIRCUITY_FACTORS[c] for c in ("urban", "suburban", "rural")]
    assert all(1.2 <= f <= 1.4 for f in factors)
    assert factors == sorted(factors)
    assert HaversineCircuity().factor == CIRCUITY_FACTORS["urban"]
    assert HaversineCircuity.for_density("rural").factor == CIRCUITY_FACTORS["rural"]


def test_circuity_scales_great_circle_distance() -> None:
    est = HaversineCircuity(1.3)
    dest = np.array([WESTLANDS, NAIROBI_CBD])
    out = est.travel_km(NAIROBI_CBD, dest)
    assert out.shape == (2,)
    assert out[0] == pytest.approx(1.3 * float(haversine_km(*NAIROBI_CBD, *WESTLANDS)))
    assert out[1] == 0.0
    assert est.travel_km(NAIROBI_CBD, np.empty((0, 2))).shape == (0,)
    assert isinstance(est, TravelEstimator)


def test_circuity_validation() -> None:
    with pytest.raises(ValueError):
        HaversineCircuity(0.9)
    with pytest.raises(ValueError):
        HaversineCircuity.for_density("metropolitan")
    with pytest.raises(ValueError):
        HaversineCircuity().travel_km(NAIROBI_CBD, np.array([1.0, 2.0]))


def test_calibrate_circuity_recovers_the_reference_ratio() -> None:
    class Reference:
        def travel_km(self, origin: tuple[float, float], destinations: np.ndarray) -> np.ndarray:
            straight = haversine_km(origin[0], origin[1], destinations[:, 0], destinations[:, 1])
            out = 1.45 * straight + 0.05
            out[-1] = np.inf  # one unroutable destination must be ignored
            return out

    dest = np.array([(-1.30, 36.80), (-1.25, 36.85), (-1.27, 36.78), (-1.32, 36.83), (0.0, 37.0)])
    factor = calibrate_circuity(NAIROBI_CBD, dest, Reference())
    assert factor == pytest.approx(1.45, abs=0.03)
    assert HaversineCircuity(factor).factor == factor


# --- OSRM ------------------------------------------------------------------------------------

OSRM_OK = {
    "code": "Ok",
    "distances": [[1520.4, 3302.0, None]],
    "durations": [[240.0, 413.4, None]],
    "sources": [{"location": [36.8172, -1.2864], "name": "Kenyatta Avenue", "distance": 3.1}],
    "destinations": [
        {"location": [36.8108, -1.2676], "name": "Waiyaki Way", "distance": 5.0},
        {"location": [36.8300, -1.3000], "name": "Mombasa Road", "distance": 8.2},
        {"location": [36.9000, -1.2000], "name": "", "distance": 120.0},
    ],
}


class RecordingFetch:
    def __init__(self, *responses: object) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, dict[str, str], float]] = []

    def __call__(self, url: str, headers: Mapping[str, str], timeout_s: float, /) -> object:
        self.calls.append((url, dict(headers), timeout_s))
        return self.responses.pop(0)


def test_osrm_table_url() -> None:
    osrm = OsrmTravel("https://router.example.org/", RecordingFetch(), profile="driving")
    url = osrm.table_url(NAIROBI_CBD, np.array([WESTLANDS, (-1.3, 36.83)]))
    assert url == (
        "https://router.example.org/table/v1/driving/"
        "36.817200,-1.286400;36.810800,-1.267600;36.830000,-1.300000"
        "?sources=0&destinations=1;2&annotations=distance,duration"
    )


def test_osrm_parses_distances_durations_and_unroutable() -> None:
    fetch = RecordingFetch(json.dumps(OSRM_OK).encode())
    osrm = OsrmTravel("http://localhost:5000", fetch, user_agent="laee-tests/1.0", timeout_s=7.5)
    dest = np.array([WESTLANDS, (-1.30, 36.83), (-1.20, 36.90)])
    table = osrm.table(NAIROBI_CBD, dest)
    np.testing.assert_allclose(table.distances_km[:2], [1.5204, 3.302])
    np.testing.assert_allclose(table.durations_min[:2], [4.0, 6.89])
    assert np.isinf(table.distances_km[2]) and np.isinf(table.durations_min[2])
    [(url, headers, timeout)] = fetch.calls
    assert url.startswith("http://localhost:5000/table/v1/driving/36.817200,-1.286400;")
    assert headers["User-Agent"] == "laee-tests/1.0"
    assert timeout == 7.5
    assert isinstance(osrm, TravelEstimator)


def test_osrm_travel_km_and_minutes_accept_a_decoded_mapping() -> None:
    dest = np.array([WESTLANDS, (-1.30, 36.83), (-1.20, 36.90)])
    osrm = OsrmTravel("http://osrm", RecordingFetch(OSRM_OK, OSRM_OK))
    np.testing.assert_allclose(osrm.travel_km(NAIROBI_CBD, dest)[:2], [1.5204, 3.302])
    np.testing.assert_allclose(osrm.travel_minutes(NAIROBI_CBD, dest)[:2], [4.0, 6.89])


def test_osrm_error_codes_raise() -> None:
    error = {"code": "InvalidQuery", "message": "Query string malformed close to position 28"}
    osrm = OsrmTravel("http://osrm", RecordingFetch(json.dumps(error)))
    with pytest.raises(OsrmError, match="InvalidQuery") as info:
        osrm.travel_km(NAIROBI_CBD, np.array([WESTLANDS]))
    assert info.value.code == "InvalidQuery"
    assert "malformed" in info.value.message


def test_osrm_rejects_responses_without_distance_annotation() -> None:
    no_distances = {"code": "Ok", "durations": [[60.0]]}
    with pytest.raises(OsrmError, match="distances"):
        parse_osrm_table(no_distances, 1)
    with pytest.raises(OsrmError, match="expected 2"):
        parse_osrm_table({"code": "Ok", "distances": [[1.0]], "durations": [[1.0]]}, 2)


def test_osrm_splits_large_tables_into_sequential_requests() -> None:
    def response(n: int, offset: float) -> dict[str, object]:
        return {
            "code": "Ok",
            "distances": [[1000.0 * (offset + i) for i in range(n)]],
            "durations": [[60.0 * (offset + i) for i in range(n)]],
        }

    fetch = RecordingFetch(response(2, 1), response(2, 3), response(1, 5))
    osrm = OsrmTravel("http://osrm", fetch, max_destinations=2)
    dest = np.array([(-1.28 + 0.01 * i, 36.8) for i in range(5)])
    np.testing.assert_allclose(osrm.travel_km(NAIROBI_CBD, dest), [1, 2, 3, 4, 5])
    assert [url.split("destinations=")[1].split("&")[0] for url, _, _ in fetch.calls] == [
        "1;2",
        "1;2",
        "1",
    ]


def test_osrm_skips_the_request_for_no_destinations_and_validates_config() -> None:
    fetch = RecordingFetch()
    assert OsrmTravel("http://osrm", fetch).travel_km(NAIROBI_CBD, np.empty((0, 2))).size == 0
    assert fetch.calls == []
    with pytest.raises(ValueError):
        OsrmTravel("osrm.example.org", fetch)
    with pytest.raises(ValueError):
        OsrmTravel("http://osrm", fetch, profile="driving/../x")


# --- spatial cells: H3 and geohash -----------------------------------------------------------


@pytest.fixture
def no_h3(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make `import h3` fail, as on an install without the optional geo extra."""
    monkeypatch.setitem(sys.modules, "h3", None)


def test_h3_path_uses_the_v4_api() -> None:
    h3 = pytest.importorskip("h3")
    assert index_backend() == "h3"
    cell = cell_for(*NAIROBI_CBD, 9)
    assert cell == h3.latlng_to_cell(*NAIROBI_CBD, 9)
    ring = neighbors(cell, 1)
    assert len(ring) == 7 and cell in ring
    assert ring == set(h3.grid_disk(cell, 1))
    assert {cell_distance(cell, other) for other in ring - {cell}} == {1}
    far = cell_for(*WESTLANDS, 9)
    assert cell_distance(cell, far) == h3.grid_distance(cell, far) > 1


def test_h3_cell_distance_rejects_mixed_resolutions_and_grids() -> None:
    pytest.importorskip("h3")
    with pytest.raises(ValueError):
        cell_distance(cell_for(*NAIROBI_CBD, 9), cell_for(*NAIROBI_CBD, 8))
    with pytest.raises(ValueError):
        cell_distance(cell_for(*NAIROBI_CBD, 9), geohash_encode(*NAIROBI_CBD, 7))


@pytest.mark.usefixtures("no_h3")
def test_geohash_fallback_when_h3_is_missing() -> None:
    assert index_backend() == "geohash"
    cell = cell_for(*NAIROBI_CBD, 9)
    assert len(cell) == geohash_precision_for_resolution(9) == 7
    assert cell == geohash_encode(*NAIROBI_CBD, 7)
    ring = neighbors(cell, 1)
    assert len(ring) == 9 and cell in ring
    assert {cell_distance(cell, other) for other in ring - {cell}} == {1}
    assert len(neighbors(cell, 2)) == 25
    far = cell_for(*WESTLANDS, 9)
    # A precision-7 cell is ~153 m square; Westlands is ~2.09 km north and ~0.71 km west.
    assert 12 <= cell_distance(cell, far) <= 15


@pytest.mark.usefixtures("no_h3")
def test_h3_cells_need_h3() -> None:
    with pytest.raises(ImportError, match="h3"):
        neighbors("897a6e42ddbffff", 1)


def test_geohash_reference_vectors() -> None:
    assert geohash_encode(57.64911, 10.40744, 11) == "u4pruydqqvj"
    lat, lon = geohash_decode("ezs42")
    assert lat == pytest.approx(42.605, abs=1e-3)
    assert lon == pytest.approx(-5.603, abs=1e-3)
    centre = geohash_decode(geohash_encode(*NAIROBI_CBD, 9))
    assert float(haversine_km(*centre, *NAIROBI_CBD)) < 0.005


def test_geohash_neighbours_wrap_the_antimeridian_and_stop_at_the_poles() -> None:
    east = geohash_encode(0.0, 179.99, 5)
    west = geohash_encode(0.0, -179.99, 5)
    assert west in neighbors(east, 1)
    assert cell_distance(east, west) == 1
    polar = geohash_encode(89.99, 0.0, 5)
    assert len(neighbors(polar, 1)) == 6
    with pytest.raises(ValueError):
        cell_distance(east, geohash_encode(0.0, 0.0, 6))
    with pytest.raises(ValueError):
        neighbors("ab!", 1)


def test_geohash_precision_tracks_h3_cell_area() -> None:
    precisions = [geohash_precision_for_resolution(r) for r in range(16)]
    assert precisions == sorted(precisions)
    h3 = pytest.importorskip("h3")
    for res, precision in enumerate(precisions):
        h3_area = h3.average_hexagon_area(res, unit="km^2")
        ratio = geohash_cell_area_km2(precision) / h3_area
        # Nearest in log area: never off by more than sqrt(32), half a geohash step.
        assert 1 / math.sqrt(32) <= ratio <= math.sqrt(32)
        for other in (precision - 1, precision + 1):
            if 1 <= other <= 12:
                assert abs(math.log(geohash_cell_area_km2(other) / h3_area)) >= abs(
                    math.log(ratio)
                )


def test_cell_argument_validation() -> None:
    with pytest.raises(ValueError):
        cell_for(0.0, 0.0, 16)
    with pytest.raises(ValueError):
        cell_for(95.0, 0.0, 5)
    with pytest.raises(ValueError):
        neighbors(geohash_encode(0.0, 0.0, 5), -1)
