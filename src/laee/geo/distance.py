"""Great-circle and travel distances (spec §7).

Straight-line distance understates how far apart two places are for a customer: highways, rivers,
one-way systems and sparse road networks all lengthen the trip. `OsrmTravel` asks an OSRM router
for road distances and durations. `HaversineCircuity` is the offline fallback: great-circle
distance scaled by a network circuity factor.

Both network adapters in this package (OSRM here, Overpass in `laee.geo.discovery`) take an
injectable `Fetch` callable, so the engine never opens a socket on its own.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

EARTH_MEAN_RADIUS_KM = 6371.0088
"""IUGG mean Earth radius R1, the usual radius for the spherical (haversine) model."""

WGS84_EQUATORIAL_RADIUS_KM = 6378.137

DensityClass = Literal["urban", "suburban", "rural"]
DENSITY_CLASSES: tuple[DensityClass, ...] = ("urban", "suburban", "rural")
_DENSITY_LOOKUP: dict[str, DensityClass] = {name: name for name in DENSITY_CLASSES}

# Network/great-circle distance ratios ("circuity"). Empirical anchors: Boscoe, Henry & Zdeb (2012,
# The Professional Geographer 64(2):188-196) report US driving distances at roughly 1.4x the
# straight line, and Giacomin & Levinson (2015, Environment and Planning B 42(6):1040-1053) show
# circuity falls as network connectivity rises and as trips get longer. We know of no published
# calibration for Nairobi, Lagos, Accra or Johannesburg, so these values are assumptions inside the
# usual 1.2-1.4 band. They are ordered by connectivity: dense cores offer the most route choice,
# while suburban estates (gated compounds, cul-de-sacs, few through-roads) and sparse rural
# networks force longer detours. Urban is not set lower because the trips there are short (often
# under 2 km), and on short trips one-way systems and access detours make up a large share of the
# distance. The three values are deliberately close: the evidence does not support large
# differences. Measure the factor for a specific market with `calibrate_circuity` when a router is
# available.
CIRCUITY_FACTORS: dict[DensityClass, float] = {"urban": 1.30, "suburban": 1.35, "rural": 1.40}

DEFAULT_USER_AGENT = "laee-geo/0.1 (Local Adaptive Elasticity Engine; travel estimator)"


def validate_density_class(density_class: str) -> DensityClass:
    """Return `density_class` typed as a `DensityClass`, or raise ValueError."""
    try:
        return _DENSITY_LOOKUP[density_class]
    except KeyError:
        raise ValueError(
            f"unknown density class {density_class!r}; expected one of {DENSITY_CLASSES}"
        ) from None


def _coords(lat: ArrayLike, lon: ArrayLike) -> tuple[FloatArray, FloatArray]:
    la = np.asarray(lat, dtype=np.float64)
    lo = np.asarray(lon, dtype=np.float64)
    if not (np.all(np.isfinite(la)) and np.all(np.isfinite(lo))):
        raise ValueError("coordinates must be finite numbers")
    if np.any(np.abs(la) > 90.0) or np.any(np.abs(lo) > 180.0):
        raise ValueError("latitude must lie in [-90, 90] and longitude in [-180, 180]")
    return la, lo


def check_point(lat: float, lon: float) -> tuple[float, float]:
    """Validate one (lat, lon) pair and return it as floats."""
    la, lo = _coords(lat, lon)
    if la.ndim != 0 or lo.ndim != 0:
        raise ValueError("expected a single latitude/longitude pair")
    return float(la), float(lo)


def haversine_km(
    lat1: ArrayLike,
    lon1: ArrayLike,
    lat2: ArrayLike,
    lon2: ArrayLike,
    *,
    radius_km: float = EARTH_MEAN_RADIUS_KM,
) -> FloatArray:
    """Great-circle distance in km between points given in decimal degrees.

    Inputs broadcast like numpy ufuncs; scalar inputs give a 0-d array (use ``float()``).
    The spherical model is within about 0.5% of the WGS84 ellipsoid everywhere.
    """
    la1, lo1 = _coords(lat1, lon1)
    la2, lo2 = _coords(lat2, lon2)
    phi1 = np.radians(la1)
    phi2 = np.radians(la2)
    h = (
        np.sin((phi2 - phi1) / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(np.radians(lo2 - lo1) / 2.0) ** 2
    )
    # Rounding can push h a hair above 1 for antipodal points, where arcsin would return NaN.
    return np.asarray(2.0 * radius_km * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0))), dtype=np.float64)


def as_destinations(destinations: ArrayLike) -> FloatArray:
    """Validate destinations as an (n, 2) float array of (lat, lon) rows."""
    arr = np.asarray(destinations, dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("destinations must be an (n, 2) array of (lat, lon) rows")
    _coords(arr[:, 0], arr[:, 1])
    return arr


@runtime_checkable
class TravelEstimator(Protocol):
    """Anything that turns an origin and destinations into travel distances in km."""

    def travel_km(self, origin: tuple[float, float], destinations: np.ndarray) -> np.ndarray:
        """Travel distance (km) from `origin` to each (lat, lon) row; inf when unreachable."""
        ...


@dataclass(frozen=True)
class HaversineCircuity:
    """Offline travel estimate: great-circle distance times a circuity factor (>= 1)."""

    factor: float = CIRCUITY_FACTORS["urban"]

    def __post_init__(self) -> None:
        if not (math.isfinite(self.factor) and self.factor >= 1.0):
            raise ValueError("circuity factor must be a finite number >= 1")

    @classmethod
    def for_density(cls, density_class: str) -> HaversineCircuity:
        """Estimator using the default circuity factor for a density class."""
        return cls(CIRCUITY_FACTORS[validate_density_class(density_class)])

    def travel_km(self, origin: tuple[float, float], destinations: np.ndarray) -> np.ndarray:
        lat, lon = check_point(*origin)
        dest = as_destinations(destinations)
        return self.factor * haversine_km(lat, lon, dest[:, 0], dest[:, 1])


def calibrate_circuity(
    origin: tuple[float, float],
    destinations: ArrayLike,
    reference: TravelEstimator,
    *,
    min_distance_km: float = 0.5,
) -> float:
    """Median network/great-circle ratio measured with `reference` (e.g. an `OsrmTravel`).

    Feed the result to `HaversineCircuity` so later offline estimates match the local network.
    """
    lat, lon = check_point(*origin)
    dest = as_destinations(destinations)
    straight = haversine_km(lat, lon, dest[:, 0], dest[:, 1])
    network = np.asarray(reference.travel_km((lat, lon), dest), dtype=np.float64)
    if network.shape != straight.shape:
        raise ValueError("reference estimator returned the wrong number of distances")
    # Routers snap both ends to the road network, which adds a roughly fixed offset; on very short
    # pairs that offset dominates the ratio, hence the minimum distance. The median resists the
    # occasional pair that detours around a river or a highway.
    usable = (straight >= min_distance_km) & np.isfinite(network)
    if not np.any(usable):
        raise ValueError(f"no routable destinations at least {min_distance_km} km from the origin")
    return max(1.0, float(np.median(network[usable] / straight[usable])))


class Fetch(Protocol):
    """Injectable HTTP GET shared by the OSRM and Overpass adapters.

    Called as ``fetch(url, headers, timeout_s)``. It returns the decoded JSON object, or the raw
    body as ``str``/``bytes``. Retries, caching and proxies are the callable's job; the adapters
    only build requests and parse responses.
    """

    def __call__(self, url: str, headers: Mapping[str, str], timeout_s: float, /) -> object: ...


def decode_json(body: object) -> Mapping[str, Any]:
    """Accept a decoded JSON object or a JSON body (str/bytes) and return the object."""
    if isinstance(body, Mapping):
        return body
    if isinstance(body, (bytes, bytearray)):
        body = body.decode("utf-8")
    if isinstance(body, str):
        doc = json.loads(body)
        if isinstance(doc, Mapping):
            return doc
        raise ValueError("expected a JSON object at the top level of the response")
    raise TypeError(f"fetch returned {type(body).__name__}; expected a mapping, str or bytes")


class OsrmError(RuntimeError):
    """OSRM answered with a code other than "Ok" (e.g. InvalidQuery, NoSegment, TooBig)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"OSRM table request failed with code {code!r}: {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True)
class TravelTable:
    """One origin's road distances (km) and durations (minutes); inf marks no route."""

    distances_km: FloatArray
    durations_min: FloatArray


def _first_row(doc: Mapping[str, Any], key: str, n: int) -> FloatArray:
    rows = doc.get(key)
    if not (isinstance(rows, list) and len(rows) == 1 and isinstance(rows[0], list)):
        raise OsrmError("MalformedResponse", f"expected {key!r} to hold exactly one row")
    if len(rows[0]) != n:
        raise OsrmError("MalformedResponse", f"expected {n} {key}, got {len(rows[0])}")
    # OSRM returns null for a destination it cannot reach (e.g. an island with no ferry).
    return np.array([np.inf if v is None else float(v) for v in rows[0]], dtype=np.float64)


def parse_osrm_table(payload: object, n_destinations: int) -> TravelTable:
    """Parse an OSRM Table response for one source (metres -> km, seconds -> minutes)."""
    doc = decode_json(payload)
    code = doc.get("code")
    if code != "Ok":
        raise OsrmError(str(code), str(doc.get("message", "no message given")))
    distances_m = _first_row(doc, "distances", n_destinations)
    durations_s = _first_row(doc, "durations", n_destinations)
    return TravelTable(distances_km=distances_m / 1000.0, durations_min=durations_s / 60.0)


_PROFILE = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True)
class OsrmTravel:
    """Road distances from an OSRM server's Table service (one source, many destinations).

    Requests are split so no single call exceeds `max_destinations` (osrm-routed rejects tables
    above its ``--max-table-size``, 100 coordinates by default) and are issued sequentially.
    """

    base_url: str
    fetch: Fetch
    profile: str = "driving"
    user_agent: str = DEFAULT_USER_AGENT
    timeout_s: float = 10.0
    max_destinations: int = 99

    def __post_init__(self) -> None:
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        if not _PROFILE.match(self.profile):
            raise ValueError(f"invalid OSRM profile {self.profile!r}")
        if not self.user_agent.strip():
            raise ValueError("user_agent must be non-empty")
        if not (math.isfinite(self.timeout_s) and self.timeout_s > 0):
            raise ValueError("timeout_s must be positive")
        if self.max_destinations < 1:
            raise ValueError("max_destinations must be at least 1")

    def table_url(self, origin: tuple[float, float], destinations: ArrayLike) -> str:
        """Table API URL with the origin as the only source (OSRM wants lon,lat order)."""
        lat, lon = check_point(*origin)
        dest = as_destinations(destinations)
        if len(dest) == 0:
            raise ValueError("at least one destination is required")
        coords = ";".join(f"{x:.6f},{y:.6f}" for y, x in [(lat, lon), *dest.tolist()])
        targets = ";".join(str(i) for i in range(1, len(dest) + 1))
        return (
            f"{self.base_url.rstrip('/')}/table/v1/{self.profile}/{coords}"
            f"?sources=0&destinations={targets}&annotations=distance,duration"
        )

    def table(self, origin: tuple[float, float], destinations: ArrayLike) -> TravelTable:
        dest = as_destinations(destinations)
        if len(dest) == 0:
            return TravelTable(np.empty(0), np.empty(0))
        headers = {"User-Agent": self.user_agent, "Accept": "application/json"}
        parts: list[TravelTable] = []
        for start in range(0, len(dest), self.max_destinations):
            chunk = dest[start : start + self.max_destinations]
            body = self.fetch(self.table_url(origin, chunk), headers, self.timeout_s)
            parts.append(parse_osrm_table(body, len(chunk)))
        return TravelTable(
            distances_km=np.concatenate([p.distances_km for p in parts]),
            durations_min=np.concatenate([p.durations_min for p in parts]),
        )

    def travel_km(self, origin: tuple[float, float], destinations: np.ndarray) -> np.ndarray:
        return self.table(origin, destinations).distances_km

    def travel_minutes(self, origin: tuple[float, float], destinations: np.ndarray) -> np.ndarray:
        return self.table(origin, destinations).durations_min
