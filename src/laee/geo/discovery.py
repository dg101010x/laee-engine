"""Competitor discovery from OpenStreetMap through the Overpass API (spec §5, §84 step 9).

Overpass usage policy. The public instances are shared, volunteer-run infrastructure:
* One request at a time. Each client IP gets a small number of query slots; parallel or
  back-to-back bursts are answered with HTTP 429 until a slot frees up (see /api/status).
  `discover_competitors` makes exactly one synchronous request, so do not call it concurrently.
  Back off and retry on 429 and 504.
* A descriptive User-Agent naming the application and a contact, so operators can reach you
  instead of blocking you. `user_agent` is required, and generic library defaults are rejected.
* Timeouts. The query declares ``[timeout:25]``, which keeps it small enough to be scheduled
  promptly. The client timeout must exceed it, or the client gives up while the server still
  holds the slot.
* Caching. Shop locations change slowly. Persist results (e.g. per H3 cell, spec §71) and refresh
  them on a schedule, typically weeks; never query per page view or per model run. For heavy use
  (the overpass-api.de operators suggest well under 10,000 queries and 1 GB a day) run a private
  instance and pass its URL as `endpoint`.
"""

from __future__ import annotations

import math
import urllib.parse
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .competition import Competitor
from .distance import Fetch, check_point, decode_json, haversine_km
from .market import INDUSTRY_IDS, validate_industry_id

OVERPASS_ENDPOINT = "https://overpass-api.de/api/interpreter"
OVERPASS_TIMEOUT_S = 25
DEFAULT_CLIENT_TIMEOUT_S = 40.0
MAX_RADIUS_M = 50_000.0
DEDUPE_RADIUS_M = 50.0

# OSM tags per industry id, following the OSM wiki's conventions:
# grocery      supermarket and convenience are the main food-retail tags. shop=general is the
#              wiki's general store (the duka or provision shop). shop=kiosk is the small booth,
#              the roadside kiosk selling staples in Nairobi and Lagos. greengrocer covers produce
#              sellers. shop=grocery is not queried: the wiki points mappers to
#              convenience/supermarket instead.
# restaurant   fast_food is included: chicken, chips and nyama choma spots compete with sit-down
#              restaurants for the same meals.
# apparel      shop=clothes is canonical. Older or looser tags (boutique, fashion) are still common
#              in African data.
# electronics  phone shops (mobile_phone) dominate electronics retail in these cities.
# pharmacy     amenity=pharmacy is canonical. The wiki's shop=chemist is a drugstore, but "chemist"
#              is the everyday East and West African word for a pharmacy and local mappers use it
#              that way.
# hair_salon   shop=hairdresser (including hairdresser=barber), plus shop=beauty, because beauty
#              salons there usually also do hair (braids, weaves).
# auto_repair  The wiki tags a repair garage shop=car_repair even though it is a service.
#              shop=tyres covers tyre fitters, who also do puncture repair and wheel alignment.
#              shop=car_parts is retail, not repair. shop=motorcycle_repair serves a different
#              customer base (boda-boda and okada mechanics) and would need its own industry id.
# professional_service  the office=* tags for professional firms.
# other_service, other  no coherent tag set exists; add these competitors manually.
# ecommerce    no geographic competitors.
OSM_TAGS: dict[str, list[tuple[str, str]]] = {
    "grocery": [
        ("shop", "supermarket"),
        ("shop", "convenience"),
        ("shop", "greengrocer"),
        ("shop", "general"),
        ("shop", "kiosk"),
    ],
    "cafe": [("amenity", "cafe")],
    "restaurant": [("amenity", "restaurant"), ("amenity", "fast_food")],
    "apparel": [("shop", "clothes"), ("shop", "shoes"), ("shop", "boutique"), ("shop", "fashion")],
    "electronics": [("shop", "electronics"), ("shop", "mobile_phone"), ("shop", "computer")],
    "pharmacy": [("amenity", "pharmacy"), ("shop", "chemist")],
    "hair_salon": [("shop", "hairdresser"), ("shop", "beauty")],
    "auto_repair": [("shop", "car_repair"), ("shop", "tyres")],
    "professional_service": [
        ("office", "accountant"),
        ("office", "lawyer"),
        ("office", "estate_agent"),
        ("office", "insurance"),
        ("office", "financial"),
        ("office", "tax_advisor"),
    ],
    "other_service": [],
    "wholesale": [("shop", "wholesale")],
    "ecommerce": [],
    "other": [],
}

_TAG_INDUSTRY: dict[tuple[str, str], str] = {
    tag: industry for industry in INDUSTRY_IDS for tag in OSM_TAGS[industry]
}
_TEXT_TAGS = ("cuisine", "brand", "description")
_GENERIC_AGENTS = ("python-requests", "python-urllib", "python-httpx", "aiohttp", "curl/", "wget/")


class OverpassError(RuntimeError):
    """Overpass returned an error remark or a malformed document."""


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _check_radius(radius_m: float) -> float:
    radius = float(radius_m)
    if not (math.isfinite(radius) and 0 < radius <= MAX_RADIUS_M):
        raise ValueError(
            f"radius_m must be in (0, {MAX_RADIUS_M:g}]; split larger areas into several queries"
        )
    return radius


def overpass_query(industry_id: str, lat: float, lon: float, radius_m: float) -> str:
    """Overpass QL selecting the industry's nodes and ways within `radius_m` of the point."""
    tags = OSM_TAGS[validate_industry_id(industry_id)]
    if not tags:
        raise ValueError(f"no OpenStreetMap tags are defined for industry {industry_id!r}")
    lat, lon = check_point(lat, lon)
    around = f"(around:{_check_radius(radius_m):g},{lat:.6f},{lon:.6f})"
    statements = [
        f'  {kind}["{_escape(key)}"="{_escape(value)}"]{around};'
        for key, value in tags
        for kind in ("node", "way")
    ]
    header = f"[out:json][timeout:{OVERPASS_TIMEOUT_S}];"
    return "\n".join([header, "(", *statements, ");", "out center;"])


@dataclass(frozen=True)
class _Found:
    competitor: Competitor
    n_tags: int


def _element(element: object, industry_id: str | None) -> _Found | None:
    if not isinstance(element, Mapping):
        return None
    kind, osm_id = element.get("type"), element.get("id")
    if kind not in ("node", "way", "relation") or not isinstance(osm_id, int):
        return None
    # Nodes carry lat/lon; ways and relations carry a "center" when queried with `out center`.
    point = element if "lat" in element else element.get("center")
    if not isinstance(point, Mapping):
        return None
    lat, lon = point.get("lat"), point.get("lon")
    if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
        return None
    if not (abs(lat) <= 90 and abs(lon) <= 180):
        return None
    raw_tags = element.get("tags")
    tags: Mapping[str, Any] = raw_tags if isinstance(raw_tags, Mapping) else {}
    matched = [
        (key, value)
        for key, value in tags.items()
        if isinstance(value, str)
        and (key, value) in _TAG_INDUSTRY
        and (industry_id is None or _TAG_INDUSTRY[(key, value)] == industry_id)
    ]
    if not matched:
        return None
    category = industry_id or min(
        (_TAG_INDUSTRY[tag] for tag in matched), key=INDUSTRY_IDS.index
    )
    name = tags.get("name")
    attributes = [f"{key}={value}" for key, value in matched]
    attributes += [f"{key}={tags[key]}" for key in _TEXT_TAGS if isinstance(tags.get(key), str)]
    competitor = Competitor(
        id=f"osm:{kind}/{osm_id}",
        name=name.strip() if isinstance(name, str) and name.strip() else None,
        lat=float(lat),
        lon=float(lon),
        category=category,
        text_attributes=tuple(attributes),
    )
    return _Found(competitor, len(tags))


def _name_key(name: str) -> str:
    return " ".join(name.casefold().split())


def _dedupe(found: list[_Found]) -> list[Competitor]:
    # A shop is often mapped twice, as a point and as its building outline. Keep the element with
    # the most tags (ties broken by id) and drop same-name elements within DEDUPE_RADIUS_M of it.
    # Chain branches further apart survive. Unnamed elements cannot be matched; all are kept.
    kept: list[Competitor] = []
    by_name: dict[str, list[Competitor]] = {}
    for item in sorted(found, key=lambda f: (-f.n_tags, f.competitor.id)):
        c = item.competitor
        if c.name is not None:
            same_name = by_name.setdefault(_name_key(c.name), [])
            if any(
                float(haversine_km(c.lat, c.lon, o.lat, o.lon)) * 1000.0 <= DEDUPE_RADIUS_M
                for o in same_name
            ):
                continue
            same_name.append(c)
        kept.append(c)
    return kept


def parse_overpass(
    payload: object, origin: tuple[float, float], *, industry_id: str | None = None
) -> list[Competitor]:
    """Competitors from an Overpass JSON response, nearest to `origin` first.

    Nodes use their coordinates and ways their ``center``. Elements without coordinates or
    without a known tag are skipped. With `industry_id`, only that industry's tags count and every
    competitor gets it as its category. Otherwise the category comes from the tags.
    """
    o_lat, o_lon = check_point(*origin)
    if industry_id is not None:
        validate_industry_id(industry_id)
    doc = decode_json(payload)
    remark = doc.get("remark")
    if isinstance(remark, str) and "error" in remark.casefold():
        # Overpass reports timeouts and memory exhaustion as HTTP 200 with a remark and a
        # truncated element list; returning that list would silently understate competition.
        raise OverpassError(f"Overpass query failed: {remark}")
    elements = doc.get("elements")
    if not isinstance(elements, list):
        raise OverpassError("Overpass response has no 'elements' list")
    found = [f for f in (_element(e, industry_id) for e in elements) if f is not None]
    competitors = _dedupe(found)
    if not competitors:
        return []
    distance = haversine_km(
        o_lat, o_lon, np.array([c.lat for c in competitors]), np.array([c.lon for c in competitors])
    )
    order = sorted(range(len(competitors)), key=lambda i: (float(distance[i]), competitors[i].id))
    return [competitors[i] for i in order]


def _check_user_agent(user_agent: str) -> str:
    agent = user_agent.strip()
    if len(agent) < 8 or agent.casefold().startswith(_GENERIC_AGENTS):
        raise ValueError(
            "user_agent must identify the application and a contact, e.g. "
            "'laee-geo/0.1 (+https://example.org; ops@example.org)'"
        )
    return agent


def discover_competitors(
    industry_id: str,
    lat: float,
    lon: float,
    radius_m: float,
    fetch: Fetch,
    *,
    user_agent: str,
    endpoint: str = OVERPASS_ENDPOINT,
    timeout_s: float = DEFAULT_CLIENT_TIMEOUT_S,
) -> list[Competitor]:
    """Query Overpass once for the industry's competitors around a point.

    Industries without OSM tags (ecommerce, other, other_service) return an empty list without a
    request. See the module docstring for the usage policy this follows.
    """
    industry = validate_industry_id(industry_id)
    agent = _check_user_agent(user_agent)
    if not (math.isfinite(timeout_s) and timeout_s > OVERPASS_TIMEOUT_S):
        raise ValueError(f"timeout_s must exceed the server-side timeout of {OVERPASS_TIMEOUT_S}s")
    if not endpoint.startswith(("http://", "https://")):
        raise ValueError("endpoint must start with http:// or https://")
    if not OSM_TAGS[industry]:
        return []
    query = overpass_query(industry, lat, lon, radius_m)
    url = f"{endpoint}?{urllib.parse.urlencode({'data': query})}"
    body = fetch(url, {"User-Agent": agent, "Accept": "application/json"}, timeout_s)
    return parse_overpass(body, (lat, lon), industry_id=industry)
