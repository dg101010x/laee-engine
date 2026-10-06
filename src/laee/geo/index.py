"""Hierarchical spatial cells for bucketing businesses, competitors and events (spec §71).

Uses Uber H3 (v4 API) when the optional ``h3`` package is importable. Otherwise it falls back to
a pure-Python geohash grid whose precision is matched to the requested H3 resolution by cell area.
H3 cell ids are always 15 hexadecimal characters and geohash cells here are at most 12
characters, so `neighbors` and `cell_distance` can tell which grid a cell belongs to and never mix
the two.
"""

from __future__ import annotations

import importlib
import math
import operator
from types import ModuleType
from typing import Literal

from .distance import EARTH_MEAN_RADIUS_KM, check_point

IndexBackend = Literal["h3", "geohash"]

H3_MAX_RESOLUTION = 15
GEOHASH_MAX_PRECISION = 12

_GEOHASH_ALPHABET = "0123456789bcdefghjkmnpqrstuvwxyz"
_GEOHASH_VALUE = {ch: i for i, ch in enumerate(_GEOHASH_ALPHABET)}
_HEX = frozenset("0123456789abcdef")
_KM_PER_DEGREE = 2.0 * math.pi * EARTH_MEAN_RADIUS_KM / 360.0

# Mean H3 hexagon area (km^2) at resolutions 0-15, from the H3 cell statistics table.
_H3_MEAN_AREA_KM2 = (
    4.357449e6, 6.097884e5, 8.680178e4, 1.239343e4, 1.770348e3, 2.529039e2, 3.612906e1,
    5.161293, 7.373276e-1, 1.053325e-1, 1.504750e-2, 2.149643e-3, 3.070919e-4, 4.387027e-5,
    6.267181e-6, 8.953115e-7,
)


def _load_h3() -> ModuleType | None:
    try:
        return importlib.import_module("h3")
    except ImportError:
        return None


def index_backend() -> IndexBackend:
    """Which grid `cell_for` will use in this environment."""
    return "h3" if _load_h3() is not None else "geohash"


def _check_resolution(resolution: int) -> int:
    res = operator.index(resolution)
    if not 0 <= res <= H3_MAX_RESOLUTION:
        raise ValueError(f"resolution must be between 0 and {H3_MAX_RESOLUTION}, got {res}")
    return res


def _bit_split(precision: int) -> tuple[int, int]:
    """(longitude bits, latitude bits) of a geohash; bits interleave starting with longitude."""
    total = 5 * precision
    return (total + 1) // 2, total // 2


def geohash_cell_area_km2(precision: int) -> float:
    """Area of a geohash cell of `precision` characters at the equator."""
    lon_bits, lat_bits = _bit_split(precision)
    return (360.0 / (1 << lon_bits) * _KM_PER_DEGREE) * (180.0 / (1 << lat_bits) * _KM_PER_DEGREE)


def geohash_precision_for_resolution(resolution: int) -> int:
    """Geohash precision whose cell area is closest, in log terms, to H3's at `resolution`.

    Geohash areas shrink 32-fold per character and H3 areas 7-fold per resolution, so the match
    is within a factor of sqrt(32) ~ 5.7 in area (~2.4 in cell width).
    """
    target = _H3_MEAN_AREA_KM2[_check_resolution(resolution)]
    return min(
        range(1, GEOHASH_MAX_PRECISION + 1),
        key=lambda p: abs(math.log(geohash_cell_area_km2(p) / target)),
    )


def _bisect(value: float, lo: float, hi: float, bits: int) -> int:
    index = 0
    for _ in range(bits):
        mid = (lo + hi) / 2.0
        if value >= mid:
            index = (index << 1) | 1
            lo = mid
        else:
            index <<= 1
            hi = mid
    return index


def _to_geohash(lat_index: int, lon_index: int, precision: int) -> str:
    lon_bits, lat_bits = _bit_split(precision)
    code = 0
    for i in range(5 * precision):
        if i % 2 == 0:
            lon_bits -= 1
            bit = (lon_index >> lon_bits) & 1
        else:
            lat_bits -= 1
            bit = (lat_index >> lat_bits) & 1
        code = (code << 1) | bit
    return "".join(
        _GEOHASH_ALPHABET[(code >> (5 * (precision - 1 - i))) & 31] for i in range(precision)
    )


def _from_geohash(cell: str) -> tuple[int, int, int]:
    """(latitude index, longitude index, precision) of a geohash cell."""
    if not isinstance(cell, str) or not 1 <= len(cell) <= GEOHASH_MAX_PRECISION:
        raise ValueError(f"not a geohash cell: {cell!r}")
    code = 0
    for ch in cell:
        if ch not in _GEOHASH_VALUE:
            raise ValueError(f"not a geohash cell: {cell!r} (invalid character {ch!r})")
        code = (code << 5) | _GEOHASH_VALUE[ch]
    precision = len(cell)
    lat_index = lon_index = 0
    for i in range(5 * precision):
        bit = (code >> (5 * precision - 1 - i)) & 1
        if i % 2 == 0:
            lon_index = (lon_index << 1) | bit
        else:
            lat_index = (lat_index << 1) | bit
    return lat_index, lon_index, precision


def geohash_encode(lat: float, lon: float, precision: int) -> str:
    """Standard base-32 geohash of a point."""
    lat, lon = check_point(lat, lon)
    if not 1 <= precision <= GEOHASH_MAX_PRECISION:
        raise ValueError(f"geohash precision must be between 1 and {GEOHASH_MAX_PRECISION}")
    lon_bits, lat_bits = _bit_split(precision)
    return _to_geohash(
        _bisect(lat, -90.0, 90.0, lat_bits), _bisect(lon, -180.0, 180.0, lon_bits), precision
    )


def geohash_decode(cell: str) -> tuple[float, float]:
    """Centre (lat, lon) of a geohash cell."""
    lat_index, lon_index, precision = _from_geohash(cell)
    lon_bits, lat_bits = _bit_split(precision)
    lat = -90.0 + (lat_index + 0.5) * 180.0 / (1 << lat_bits)
    lon = -180.0 + (lon_index + 0.5) * 360.0 / (1 << lon_bits)
    return lat, lon


def geohash_neighbors(cell: str, k: int) -> set[str]:
    """Cells within `k` steps (8-connected, so Chebyshev distance) of `cell`, itself included.

    Longitude wraps across the antimeridian; rows beyond the poles do not exist.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    lat_index, lon_index, precision = _from_geohash(cell)
    lon_bits, lat_bits = _bit_split(precision)
    n_lat, n_lon = 1 << lat_bits, 1 << lon_bits
    return {
        _to_geohash(lat_index + dy, (lon_index + dx) % n_lon, precision)
        for dy in range(-k, k + 1)
        if 0 <= lat_index + dy < n_lat
        for dx in range(-k, k + 1)
    }


def geohash_grid_distance(a: str, b: str) -> int:
    """Grid steps between two geohash cells of equal precision (consistent with neighbours)."""
    lat_a, lon_a, prec_a = _from_geohash(a)
    lat_b, lon_b, prec_b = _from_geohash(b)
    if prec_a != prec_b:
        raise ValueError(f"cells have different precisions ({prec_a} and {prec_b})")
    n_lon = 1 << _bit_split(prec_a)[0]
    d_lon = abs(lon_a - lon_b)
    return max(abs(lat_a - lat_b), min(d_lon, n_lon - d_lon))


def is_h3_cell(cell: str) -> bool:
    """True when `cell` has the shape of an H3 cell id (15 lowercase hex characters)."""
    return isinstance(cell, str) and len(cell) == 15 and set(cell) <= _HEX


def _require_h3(cell: str) -> ModuleType:
    h3 = _load_h3()
    if h3 is None:
        raise ImportError(
            f"{cell!r} is an H3 cell but the optional h3 package is not installed "
            "(pip install 'laee[geo]')"
        )
    if not h3.is_valid_cell(cell):
        raise ValueError(f"invalid H3 cell: {cell!r}")
    return h3


def cell_for(lat: float, lon: float, resolution: int) -> str:
    """Cell containing the point: H3 at `resolution`, else the area-matched geohash."""
    lat, lon = check_point(lat, lon)
    res = _check_resolution(resolution)
    h3 = _load_h3()
    if h3 is not None:
        return str(h3.latlng_to_cell(lat, lon, res))
    return geohash_encode(lat, lon, geohash_precision_for_resolution(res))


def neighbors(cell: str, k: int) -> set[str]:
    """All cells within grid distance `k` of `cell`, including `cell` (H3 grid_disk semantics)."""
    k = operator.index(k)
    if k < 0:
        raise ValueError("k must be non-negative")
    if is_h3_cell(cell):
        return {str(c) for c in _require_h3(cell).grid_disk(cell, k)}
    return geohash_neighbors(cell, k)


def cell_distance(a: str, b: str) -> int:
    """Grid distance between two cells of the same grid and resolution."""
    if is_h3_cell(a) != is_h3_cell(b):
        raise ValueError("cannot measure distance between an H3 cell and a geohash cell")
    if not is_h3_cell(a):
        return geohash_grid_distance(a, b)
    h3 = _require_h3(a)
    _require_h3(b)
    if h3.get_resolution(a) != h3.get_resolution(b):
        raise ValueError("cells have different H3 resolutions")
    try:
        return int(h3.grid_distance(a, b))
    except h3.H3BaseException as exc:
        # H3 cannot measure across some pentagon distortions or over very long ranges.
        raise ValueError(f"H3 could not compute the grid distance from {a} to {b}: {exc}") from exc
