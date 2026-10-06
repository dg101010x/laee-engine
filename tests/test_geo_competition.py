"""Competitor influence, competitive pressure, price indices and entry/exit series."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from laee.geo.catchment import catchment_overlap
from laee.geo.competition import (
    INFLUENCE_COLUMNS,
    BusinessLocation,
    Competitor,
    DateLike,
    competition_change_series,
    competitive_price_index,
    competitive_pressure,
    effective_competitor_count,
    hashing_embedding,
    is_active,
    price_band_similarity,
    product_similarity,
)
from laee.geo.distance import EARTH_MEAN_RADIUS_KM, CIRCUITY_FACTORS, haversine_km

ORIGIN = (-1.2864, 36.8172)  # Nairobi CBD
KM_PER_DEG_LAT = math.pi * EARTH_MEAN_RADIUS_KM / 180.0
CAFE_LAMBDA = math.log(2) / 0.75  # median of the cafe prior (0.25, 0.5, 1, 2)


def north(km: float) -> tuple[float, float]:
    """Point exactly `km` great-circle km due north of ORIGIN."""
    return ORIGIN[0] + km / KM_PER_DEG_LAT, ORIGIN[1]


def cafe(cid: str, km: float, price: float | None = None, **kwargs: Any) -> Competitor:
    lat, lon = north(km)
    return Competitor(cid, f"Cafe {cid}", lat, lon, "cafe", price, **kwargs)


def business(price: float | None = 100.0, **kwargs: Any) -> BusinessLocation:
    return BusinessLocation(ORIGIN[0], ORIGIN[1], "cafe", price, **kwargs)


# --- similarity ------------------------------------------------------------------------------


def test_hashing_embedding_is_unit_length_and_normalises_text() -> None:
    e = hashing_embedding("Espresso bar, pastries & Wi-Fi")
    assert e.shape == (256,)
    assert np.linalg.norm(e) == pytest.approx(1.0)
    assert hashing_embedding("espresso coffee bar") @ hashing_embedding(
        "COFFEE / espresso bar"
    ) == pytest.approx(1.0)
    assert hashing_embedding("Café") @ hashing_embedding("cafe") == pytest.approx(1.0)
    assert not hashing_embedding("").any()
    assert not hashing_embedding(" -- !! ").any()
    assert hashing_embedding("tilapia", dim=64).shape == (64,)


def test_hashing_embedding_ranks_related_text_higher() -> None:
    base = hashing_embedding("fresh espresso coffee and pastries")
    related = hashing_embedding("espresso, filter coffee, croissants and pastries")
    unrelated = hashing_embedding("tyre fitting, wheel alignment, puncture repair")
    assert base @ related > 0.4
    assert base @ related > base @ unrelated + 0.3


def test_hashing_embedding_is_identical_across_processes() -> None:
    # Python salts str hashes per process; the embedding must not depend on that.
    text = "Mama Oliech: fish, ugali & sukuma wiki"
    code = (
        "import json; from laee.geo.competition import hashing_embedding; "
        f"print(json.dumps(hashing_embedding({text!r}).tolist()))"
    )
    src = str(Path(__file__).resolve().parents[1] / "src")
    outputs = []
    for seed in ("0", "4242"):
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": src}
        result = subprocess.run(
            [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
        )
        outputs.append(json.loads(result.stdout))
    assert outputs[0] == outputs[1] == hashing_embedding(text).tolist()


def test_product_similarity_blends_text_and_category() -> None:
    b = business(text_attributes=("espresso", "pastries", "wifi"))
    same_text = cafe("a", 0.1, text_attributes=("espresso", "pastries", "wifi"))
    restaurant_same_text = Competitor(
        "r", None, *north(0.1), "restaurant", text_attributes=("espresso", "pastries", "wifi")
    )
    unrelated = cafe("u", 0.1, text_attributes=("tyre fitting", "wheel alignment"))
    no_text = cafe("n", 0.1)
    restaurant_no_text = Competitor("r2", None, *north(0.1), "restaurant")
    assert product_similarity(b, same_text) == pytest.approx(1.0)
    assert product_similarity(b, restaurant_same_text) == pytest.approx(0.5)
    assert 0.5 <= product_similarity(b, unrelated) < 0.65
    assert product_similarity(b, no_text) == 1.0
    assert product_similarity(b, restaurant_no_text) == 0.0
    assert product_similarity(b, restaurant_same_text, category_weight=0.2) == pytest.approx(0.8)


def test_price_band_similarity() -> None:
    r = price_band_similarity(100.0, [100.0, 120.0, 200.0, None, np.nan])
    np.testing.assert_allclose(
        r, [1.0, math.exp(-math.log(1.2) / 0.25), math.exp(-math.log(2) / 0.25), 1.0, 1.0]
    )
    # Symmetric in the price ratio, and a wider tau is more tolerant.
    assert float(price_band_similarity(120.0, 100.0)) == pytest.approx(r[1])
    assert float(price_band_similarity(100.0, 200.0, tau=1.0)) == pytest.approx(0.5)
    np.testing.assert_array_equal(price_band_similarity(None, [50.0, 500.0]), [1.0, 1.0])
    with pytest.raises(ValueError):
        price_band_similarity(100.0, [0.0])
    with pytest.raises(ValueError):
        price_band_similarity(100.0, [10.0], tau=0.0)


def test_effective_competitor_count() -> None:
    for n in (1, 2, 4, 17):
        assert effective_competitor_count(np.full(n, 0.37)) == pytest.approx(n)
    assert effective_competitor_count([0.9, 0.05, 0.05]) == pytest.approx(1 / 0.815)
    assert effective_competitor_count([]) == 0.0
    assert effective_competitor_count([0.0, 0.0]) == 0.0


def test_competitive_price_index_direction_and_weighting() -> None:
    cheaper = competitive_price_index(100.0, [120.0, 120.0], [1.0, 3.0])
    assert cheaper is not None and cheaper.cpi == pytest.approx(1.2)
    pricier = competitive_price_index(100.0, [80.0], [1.0])
    assert pricier is not None
    assert pricier.cpi == pytest.approx(0.8) and pricier.relative_price == pytest.approx(0.25)
    mixed = competitive_price_index(100.0, [90.0, 150.0, None], [3.0, 1.0, 5.0])
    assert mixed is not None
    assert mixed.weighted_competitor_price == pytest.approx((3 * 90 + 150) / 4)
    assert mixed.n_priced == 2
    assert competitive_price_index(None, [90.0], [1.0]) is None
    assert competitive_price_index(100.0, [None], [1.0]) is None
    assert competitive_price_index(100.0, [90.0], [0.0]) is None


# --- competitive pressure --------------------------------------------------------------------


def test_influence_is_the_product_of_the_spec_components() -> None:
    near, far = cafe("near", 0.2, 110.0, size_weight=2.0), cafe("far", 3.0, 150.0)
    summary = competitive_pressure(business(100.0), [far, near], as_of=date(2026, 6, 1))
    table = summary.influence.set_index("competitor_id")
    circuity = CIRCUITY_FACTORS["urban"]
    expected = {
        "near": 2.0 * math.exp(-CAFE_LAMBDA * 0.2 * circuity) * math.exp(-math.log(1.1) / 0.25),
        "far": 1.0 * math.exp(-CAFE_LAMBDA * 3.0 * circuity) * math.exp(-math.log(1.5) / 0.25),
    }
    for cid, value in expected.items():
        assert table.loc[cid, "influence"] == pytest.approx(value)
        assert table.loc[cid, "similarity"] == 1.0
        assert table.loc[cid, "customer_overlap"] == 1.0
    assert summary.total_pressure == pytest.approx(sum(expected.values()))
    assert list(summary.influence["competitor_id"]) == ["near", "far"]
    assert tuple(summary.influence.columns) == INFLUENCE_COLUMNS
    assert summary.influence["share"].sum() == pytest.approx(1.0)
    weighted_price = (expected["near"] * 110 + expected["far"] * 150) / sum(expected.values())
    assert summary.weighted_competitor_price == pytest.approx(weighted_price)
    assert summary.competitive_price_index == pytest.approx(weighted_price / 100.0)
    assert summary.effective_radius_km == pytest.approx(0.75)


def test_identical_competitors_count_as_n_effective_competitors() -> None:
    for n in (1, 3, 4, 9):
        rivals = [cafe(f"c{i}", 0.6, 120.0, text_attributes=("espresso",)) for i in range(n)]
        summary = competitive_pressure(business(text_attributes=("espresso",)), rivals)
        assert summary.n_active == n
        assert summary.effective_competitors == pytest.approx(n)
        single = summary.influence["influence"].iloc[0]
        assert summary.total_pressure == pytest.approx(n * single)


def test_competitors_are_not_treated_equally() -> None:
    rivals = [
        cafe("close", 0.3, 100.0),
        cafe("distant", 2.5, 100.0),
        cafe("premium", 0.3, 300.0),
        Competitor("salon", None, *north(0.3), "hair_salon", 100.0),
    ]
    table = competitive_pressure(business(100.0), rivals).influence.set_index("competitor_id")
    assert table.loc["close", "influence"] > table.loc["distant", "influence"]
    assert table.loc["close", "influence"] > table.loc["premium", "influence"]
    assert table.loc["salon", "influence"] == 0.0


@pytest.mark.parametrize(
    ("business_price", "rival_price", "direction"),
    [(80.0, 100.0, 1), (100.0, 100.0, 0), (125.0, 100.0, -1)],
)
def test_cpi_above_one_when_the_business_is_cheaper(
    business_price: float, rival_price: float, direction: int
) -> None:
    rivals = [cafe("a", 0.4, rival_price), cafe("b", 1.1, rival_price), cafe("c", 0.9)]
    summary = competitive_pressure(business(business_price), rivals)
    cpi, relative = summary.competitive_price_index, summary.relative_price
    assert cpi is not None and relative is not None
    assert summary.n_priced == 2
    assert cpi == pytest.approx(rival_price / business_price)
    assert np.sign(round(cpi - 1.0, 12)) == direction
    assert np.sign(round(relative, 12)) == -direction
    assert relative == pytest.approx((business_price - rival_price) / rival_price)


def test_unknown_prices() -> None:
    summary = competitive_pressure(business(None), [cafe("a", 0.4, 120.0)])
    assert summary.competitive_price_index is None and summary.relative_price is None
    assert summary.influence.loc[0, "price_similarity"] == 1.0
    assert summary.n_priced == 0
    assert summary.total_pressure > 0


def test_as_of_respects_opening_and_closing_dates() -> None:
    rivals = [
        cafe("always", 0.5),
        cafe("opens_march", 0.5, opened_at=date(2026, 3, 1)),
        cafe("closes_feb", 0.5, closed_at=date(2026, 2, 1)),
        cafe("2025_to_june", 0.5, opened_at="2025-01-01", closed_at=datetime(2026, 6, 1, 9)),
    ]

    def active(as_of: DateLike | None) -> set[str]:
        summary = competitive_pressure(business(), rivals, as_of=as_of)
        return set(summary.influence["competitor_id"])

    assert active(date(2024, 12, 31)) == {"always", "closes_feb"}
    assert active(date(2026, 1, 15)) == {"always", "closes_feb", "2025_to_june"}
    assert active(pd.Timestamp("2026-02-01")) == {"always", "2025_to_june"}
    assert active("2026-03-01") == {"always", "opens_march", "2025_to_june"}
    assert active(date(2026, 7, 1)) == {"always", "opens_march"}
    # No date: the list is a current snapshot, so anything with a closing date is gone.
    assert active(None) == {"always", "opens_march"}
    assert is_active(rivals[1], date(2026, 3, 1)) and not is_active(rivals[1], date(2026, 2, 28))


def test_ecommerce_competition_ignores_distance() -> None:
    shop = BusinessLocation(*ORIGIN, "ecommerce", 50.0)
    rivals = [
        Competitor("near", None, *north(1.0), "ecommerce", 50.0),
        Competitor("lagos", None, 6.5244, 3.3792, "ecommerce", 50.0),
    ]
    summary = competitive_pressure(shop, rivals)
    assert summary.distance_lambda == 0.0 and summary.effective_radius_km is None
    np.testing.assert_allclose(summary.influence["influence"], [1.0, 1.0])
    assert summary.effective_competitors == pytest.approx(2.0)


def test_market_radius_and_custom_travel_estimator() -> None:
    east = Competitor("east", None, ORIGIN[0], ORIGIN[1] + 0.009, "cafe")
    west = Competitor("west", None, ORIGIN[0], ORIGIN[1] - 0.009, "cafe")
    outside = cafe("outside", 4.0)

    class RiverToTheEast:
        """Customers must detour via a bridge to reach anything east of the business."""

        def travel_km(self, origin: tuple[float, float], destinations: np.ndarray) -> np.ndarray:
            d = haversine_km(origin[0], origin[1], destinations[:, 0], destinations[:, 1])
            return np.where(destinations[:, 1] > origin[1], 4.0 * d, 1.2 * d)

    summary = competitive_pressure(
        business(), [east, west, outside], travel=RiverToTheEast(), market_radius_km=2.0
    )
    table = summary.influence.set_index("competitor_id")
    assert summary.n_active == 2 and "outside" not in table.index
    assert table.loc["east", "distance_km"] == pytest.approx(table.loc["west", "distance_km"])
    detour = 4.0 / 1.2
    assert table.loc["east", "travel_km"] == pytest.approx(detour * table.loc["west", "travel_km"])
    assert table.loc["east", "influence"] < table.loc["west", "influence"]


def test_catchment_lambda_adds_customer_overlap() -> None:
    rivals = [cafe("a", 0.3), cafe("b", 1.5)]
    plain = competitive_pressure(business(), rivals).influence.set_index("competitor_id")
    with_overlap = competitive_pressure(business(), rivals, catchment_lambda=1.1)
    table = with_overlap.influence.set_index("competitor_id")
    expected = catchment_overlap(plain["travel_km"].to_numpy(), 1.1)
    np.testing.assert_allclose(table.loc[plain.index, "customer_overlap"], expected)
    np.testing.assert_allclose(
        table.loc[plain.index, "influence"], plain["influence"].to_numpy() * expected
    )
    assert with_overlap.catchment_lambda == 1.1


def test_no_competitors_and_input_validation() -> None:
    empty = competitive_pressure(business(), [])
    assert empty.total_pressure == 0.0 and empty.effective_competitors == 0.0
    assert empty.n_active == 0 and empty.competitive_price_index is None
    assert tuple(empty.influence.columns) == INFLUENCE_COLUMNS
    unnamed = competitive_pressure(business(), [Competitor("x", None, *north(0.2), "cafe")])
    pd.testing.assert_series_equal(unnamed.influence.dtypes, empty.influence.dtypes)
    assert pd.isna(unnamed.influence.loc[0, "name"])  # missing, not the string "None"
    with pytest.raises(ValueError, match="duplicate"):
        competitive_pressure(business(), [cafe("a", 0.2), cafe("a", 0.4)])
    with pytest.raises(ValueError):
        competitive_pressure(business(), [cafe("a", 0.2)], price_tau=0.0)
    with pytest.raises(ValueError):
        Competitor("x", None, *ORIGIN, "coffee")
    with pytest.raises(ValueError, match="closes before it opens"):
        cafe("x", 0.1, opened_at=date(2026, 5, 1), closed_at=date(2026, 4, 1))
    with pytest.raises(ValueError):
        BusinessLocation(*ORIGIN, "cafe", density_class="dense")  # type: ignore[arg-type]
    assert cafe("t", 0.1, text_attributes="espresso").text_attributes == ("espresso",)


# --- entry / exit series (spec §29) ----------------------------------------------------------


def test_competition_change_series_tracks_entries_and_exits() -> None:
    incumbent = cafe("incumbent", 0.5, 100.0)
    entrant = cafe("entrant", 0.3, 100.0, opened_at=date(2026, 3, 1))
    leaver = cafe("leaver", 1.0, 100.0, closed_at=date(2026, 2, 1))
    rivals = [incumbent, entrant, leaver]
    days = [date(2026, 1, 15), date(2026, 2, 15), date(2026, 3, 15)]
    series = competition_change_series(business(100.0), rivals, days)

    trading = {"incumbent": days[0], "entrant": days[2], "leaver": days[0]}
    influence = {
        c.id: competitive_pressure(business(100.0), [c], as_of=trading[c.id]).total_pressure
        for c in rivals
    }
    assert all(value > 0 for value in influence.values())
    assert list(series.columns) == ["competition", "delta_competition", "n_active"]
    assert series.index.name == "date"
    assert list(series.index) == list(pd.to_datetime(days))
    np.testing.assert_allclose(
        series["competition"],
        [
            influence["incumbent"] + influence["leaver"],
            influence["incumbent"],
            influence["incumbent"] + influence["entrant"],
        ],
    )
    assert math.isnan(series["delta_competition"].iloc[0])
    assert series["delta_competition"].iloc[1] == pytest.approx(-influence["leaver"])
    assert series["delta_competition"].iloc[2] == pytest.approx(influence["entrant"])
    assert list(series["n_active"]) == [2, 1, 2]
    for day, level in zip(days, series["competition"]):
        assert competitive_pressure(business(100.0), rivals, as_of=day).total_pressure == (
            pytest.approx(level)
        )


def test_competition_change_series_validates_dates() -> None:
    rivals = [cafe("a", 0.5)]
    with pytest.raises(ValueError, match="increasing"):
        competition_change_series(business(), rivals, [date(2026, 2, 1), date(2026, 1, 1)])
    with pytest.raises(ValueError):
        competition_change_series(business(), rivals, [])
