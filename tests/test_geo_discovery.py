"""OpenStreetMap competitor discovery through Overpass, tested against fixture payloads."""

from __future__ import annotations

import json
import urllib.parse
from collections.abc import Mapping

import pytest

from laee.geo.discovery import (
    OSM_TAGS,
    OVERPASS_ENDPOINT,
    OverpassError,
    discover_competitors,
    overpass_query,
    parse_overpass,
)
from laee.geo.distance import haversine_km
from laee.geo.market import INDUSTRY_IDS

ORIGIN = (-1.2864, 36.8172)  # Nairobi CBD
AGENT = "laee-tests/1.0 (+https://example.org/laee; geo-team@example.org)"

OVERPASS_FIXTURE = {
    "version": 0.6,
    "generator": "Overpass API 0.7.62.1 084b4234",
    "osm3s": {"timestamp_osm_base": "2026-10-01T09:00:00Z", "copyright": "ODbL"},
    "elements": [
        {
            "type": "node",
            "id": 1001,
            "lat": -1.2860,
            "lon": 36.8180,
            "tags": {
                "amenity": "cafe",
                "name": "Java House",
                "cuisine": "coffee_shop",
                "brand": "Java House",
            },
        },
        {
            # The same cafe mapped again as its building outline, ~40 m away.
            "type": "way",
            "id": 2002,
            "center": {"lat": -1.2858, "lon": 36.8183},
            "nodes": [9001, 9002, 9003, 9004, 9001],
            "tags": {"amenity": "cafe", "name": "JAVA  house", "building": "yes"},
        },
        {
            # Another branch of the chain ~1.9 km away: a different competitor.
            "type": "node",
            "id": 1003,
            "lat": -1.2700,
            "lon": 36.8110,
            "tags": {"amenity": "cafe", "name": "Java House"},
        },
        {"type": "node", "id": 1004, "lat": -1.2866, "lon": 36.8170, "tags": {"amenity": "cafe"}},
        {
            "type": "way",
            "id": 2005,
            "center": {"lat": -1.2830, "lon": 36.8150},
            "nodes": [9101, 9102, 9103, 9101],
            "tags": {"amenity": "cafe", "name": "Artcaffe", "description": "espresso, brunch"},
        },
        {"type": "way", "id": 2006, "tags": {"amenity": "cafe", "name": "No Centre"}},
        {"type": "node", "id": 1007, "lat": -1.2865, "lon": 36.8171, "tags": {"amenity": "bank"}},
        {
            "type": "node",
            "id": 1008,
            "lat": -1.2880,
            "lon": 36.8200,
            "tags": {"amenity": "restaurant", "name": "  Mama Oliech  ", "cuisine": "african"},
        },
    ],
}


# Cafes in the fixture after dedupe, nearest to ORIGIN first (~31 m, ~99 m, ~450 m, ~1.9 km).
CAFES_NEAREST_FIRST = ["osm:node/1004", "osm:node/1001", "osm:way/2005", "osm:node/1003"]


class RecordingFetch:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[tuple[str, dict[str, str], float]] = []

    def __call__(self, url: str, headers: Mapping[str, str], timeout_s: float, /) -> object:
        self.calls.append((url, dict(headers), timeout_s))
        return self.response


# --- tag table -------------------------------------------------------------------------------


def test_osm_tags_cover_every_industry_id() -> None:
    assert set(OSM_TAGS) == set(INDUSTRY_IDS)
    for industry in ("ecommerce", "other", "other_service"):
        assert OSM_TAGS[industry] == []


def test_osm_tags_follow_the_agreed_mapping() -> None:
    def values(industry: str, key: str) -> set[str]:
        return {v for k, v in OSM_TAGS[industry] if k == key}

    assert values("grocery", "shop") == {
        "supermarket",
        "convenience",
        "greengrocer",
        "general",
        "kiosk",
    }
    assert OSM_TAGS["cafe"] == [("amenity", "cafe")]
    assert values("restaurant", "amenity") == {"restaurant", "fast_food"}
    assert values("apparel", "shop") == {"clothes", "shoes", "boutique", "fashion"}
    assert values("electronics", "shop") == {"electronics", "mobile_phone", "computer"}
    assert set(OSM_TAGS["pharmacy"]) == {("amenity", "pharmacy"), ("shop", "chemist")}
    assert values("hair_salon", "shop") == {"hairdresser", "beauty"}
    assert values("auto_repair", "shop") == {"car_repair", "tyres"}
    assert values("professional_service", "office") == {
        "accountant",
        "lawyer",
        "estate_agent",
        "insurance",
        "financial",
        "tax_advisor",
    }
    assert OSM_TAGS["wholesale"] == [("shop", "wholesale")]


def test_each_tag_belongs_to_one_industry() -> None:
    tags = [tag for industry in INDUSTRY_IDS for tag in OSM_TAGS[industry]]
    assert len(tags) == len(set(tags))


# --- query -----------------------------------------------------------------------------------


def test_overpass_query_text() -> None:
    assert overpass_query("cafe", *ORIGIN, 1500) == (
        "[out:json][timeout:25];\n"
        "(\n"
        '  node["amenity"="cafe"](around:1500,-1.286400,36.817200);\n'
        '  way["amenity"="cafe"](around:1500,-1.286400,36.817200);\n'
        ");\n"
        "out center;"
    )


def test_overpass_query_has_a_node_and_way_statement_per_tag() -> None:
    query = overpass_query("pharmacy", 5.6037, -0.1870, 2500.5)  # Accra
    for tag in ('["amenity"="pharmacy"]', '["shop"="chemist"]'):
        for kind in ("node", "way"):
            assert f"{kind}{tag}(around:2500.5,5.603700,-0.187000);" in query
    grocery = overpass_query("grocery", 6.5244, 3.3792, 3000)  # Lagos
    assert grocery.count("(around:3000,6.524400,3.379200);") == 2 * len(OSM_TAGS["grocery"])


def test_overpass_query_rejects_untagged_industries_and_bad_input() -> None:
    for industry in ("ecommerce", "other", "other_service"):
        with pytest.raises(ValueError, match="no OpenStreetMap tags"):
            overpass_query(industry, *ORIGIN, 1000)
    with pytest.raises(ValueError):
        overpass_query("coffee", *ORIGIN, 1000)
    with pytest.raises(ValueError):
        overpass_query("cafe", *ORIGIN, 0)
    with pytest.raises(ValueError):
        overpass_query("cafe", *ORIGIN, 75_000)
    with pytest.raises(ValueError):
        overpass_query("cafe", 95.0, 0.0, 1000)


# --- parsing ---------------------------------------------------------------------------------


def test_parse_overpass_nodes_ways_and_dedupe() -> None:
    found = parse_overpass(OVERPASS_FIXTURE, ORIGIN, industry_id="cafe")
    assert [c.id for c in found] == CAFES_NEAREST_FIRST
    by_id = {c.id: c for c in found}
    # The building outline duplicates node 1001 (same name, ~40 m): the richer node is kept.
    assert "osm:way/2002" not in by_id
    assert float(haversine_km(-1.2860, 36.8180, -1.2858, 36.8183)) * 1000 < 50
    # The branch 1.9 km away is a separate competitor despite the shared name.
    assert by_id["osm:node/1003"].name == "Java House"
    artcaffe = by_id["osm:way/2005"]
    assert (artcaffe.lat, artcaffe.lon) == (-1.2830, 36.8150)
    assert artcaffe.text_attributes == ("amenity=cafe", "description=espresso, brunch")
    assert by_id["osm:node/1004"].name is None
    java = by_id["osm:node/1001"]
    assert java.category == "cafe"
    assert java.text_attributes == ("amenity=cafe", "cuisine=coffee_shop", "brand=Java House")
    assert java.reference_price is None and java.size_weight == 1.0


def test_parse_overpass_without_industry_infers_categories() -> None:
    found = parse_overpass(json.dumps(OVERPASS_FIXTURE), ORIGIN)
    categories = {c.id: c.category for c in found}
    assert categories["osm:node/1008"] == "restaurant"
    assert categories["osm:node/1001"] == "cafe"
    assert "osm:node/1007" not in categories  # a bank is not a tagged competitor
    assert next(c for c in found if c.id == "osm:node/1008").name == "Mama Oliech"
    distances = [float(haversine_km(*ORIGIN, c.lat, c.lon)) for c in found]
    assert distances == sorted(distances)


def test_parse_overpass_raises_on_runtime_error_remarks() -> None:
    timed_out = {
        "version": 0.6,
        "elements": [],
        "remark": 'runtime error: Query timed out in "query" at line 3 after 25 seconds.',
    }
    with pytest.raises(OverpassError, match="timed out"):
        parse_overpass(timed_out, ORIGIN)
    with pytest.raises(OverpassError, match="elements"):
        parse_overpass({"version": 0.6}, ORIGIN)
    assert parse_overpass({"elements": []}, ORIGIN) == []


# --- discovery -------------------------------------------------------------------------------


def test_discover_competitors_sends_one_polite_request() -> None:
    fetch = RecordingFetch(json.dumps(OVERPASS_FIXTURE).encode())
    found = discover_competitors("cafe", *ORIGIN, 1500, fetch, user_agent=AGENT)
    [(url, headers, timeout)] = fetch.calls
    endpoint, query_string = url.split("?", 1)
    assert endpoint == OVERPASS_ENDPOINT
    assert urllib.parse.parse_qs(query_string)["data"] == [overpass_query("cafe", *ORIGIN, 1500)]
    assert headers["User-Agent"] == AGENT
    assert timeout > 25
    assert [c.id for c in found] == CAFES_NEAREST_FIRST
    assert all(c.category == "cafe" for c in found)


def test_discover_competitors_custom_endpoint() -> None:
    fetch = RecordingFetch({"elements": []})
    found = discover_competitors(
        "auto_repair", -26.2041, 28.0473, 5000, fetch,  # Johannesburg
        user_agent=AGENT, endpoint="https://overpass.internal/api/interpreter", timeout_s=60.0,
    )
    assert found == []
    [(url, _, timeout)] = fetch.calls
    assert url.startswith("https://overpass.internal/api/interpreter?data=")
    assert timeout == 60.0


def test_discover_competitors_skips_untagged_industries_without_a_request() -> None:
    fetch = RecordingFetch(OVERPASS_FIXTURE)
    for industry in ("ecommerce", "other", "other_service"):
        assert discover_competitors(industry, *ORIGIN, 1000, fetch, user_agent=AGENT) == []
    assert fetch.calls == []


@pytest.mark.parametrize("agent", ["", "   ", "python-requests/2.32.3", "curl/8.5.0", "laee"])
def test_discover_competitors_requires_a_descriptive_user_agent(agent: str) -> None:
    fetch = RecordingFetch(OVERPASS_FIXTURE)
    with pytest.raises(ValueError, match="user_agent"):
        discover_competitors("cafe", *ORIGIN, 1000, fetch, user_agent=agent)
    assert fetch.calls == []


def test_discover_competitors_client_timeout_must_exceed_the_server_timeout() -> None:
    with pytest.raises(ValueError, match="timeout"):
        discover_competitors(
            "cafe", *ORIGIN, 1000, RecordingFetch(OVERPASS_FIXTURE), user_agent=AGENT, timeout_s=20
        )
