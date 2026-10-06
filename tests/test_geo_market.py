"""Market radius priors, distance decay, radius selection, catchment and demographics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from laee.geo.catchment import (
    catchment_overlap,
    distance_band_shares,
    fit_distance_decay,
    weighted_demographics,
)
from laee.geo.market import (
    INDUSTRY_IDS,
    PROVISIONAL,
    RADIUS_PRIORS_KM,
    decay_lambda,
    density_adjusted_radii,
    distance_weights,
    industry_decay_lambda,
    industry_median_radius_km,
    is_geographic,
    select_market_radius,
)

APP_INDUSTRY_IDS = {
    "grocery",
    "cafe",
    "restaurant",
    "apparel",
    "electronics",
    "pharmacy",
    "hair_salon",
    "auto_repair",
    "professional_service",
    "other_service",
    "wholesale",
    "ecommerce",
    "other",
}


# --- radius priors ---------------------------------------------------------------------------


def test_priors_cover_exactly_the_app_industry_ids() -> None:
    assert set(RADIUS_PRIORS_KM) == APP_INDUSTRY_IDS == set(INDUSTRY_IDS)


def test_spec_section_6_priors_are_verbatim() -> None:
    assert RADIUS_PRIORS_KM["cafe"] == (0.25, 0.5, 1.0, 2.0)  # spec "coffee"
    assert RADIUS_PRIORS_KM["restaurant"] == (0.5, 1.0, 2.0, 3.0)
    assert RADIUS_PRIORS_KM["grocery"] == (1.0, 3.0, 5.0, 10.0)
    assert RADIUS_PRIORS_KM["auto_repair"] == (3.0, 5.0, 10.0, 20.0)
    assert RADIUS_PRIORS_KM["hair_salon"] == (1.0, 2.0, 5.0)
    assert RADIUS_PRIORS_KM["professional_service"] == (3.0, 5.0, 10.0)


def test_provisional_marks_every_prior_not_taken_from_the_spec() -> None:
    from_spec = {
        "cafe",
        "restaurant",
        "grocery",
        "auto_repair",
        "hair_salon",
        "professional_service",
    }
    assert PROVISIONAL == APP_INDUSTRY_IDS - from_spec - {"ecommerce"}


def test_priors_are_positive_and_increasing_except_ecommerce() -> None:
    assert RADIUS_PRIORS_KM["ecommerce"] == ()
    assert not is_geographic("ecommerce")
    for industry, radii in RADIUS_PRIORS_KM.items():
        if industry == "ecommerce":
            continue
        assert is_geographic(industry)
        assert radii and all(r > 0 for r in radii)
        assert list(radii) == sorted(set(radii))


def test_density_adjusted_radii_widen_in_lower_density_areas() -> None:
    urban = density_adjusted_radii("cafe", "urban")
    suburban = density_adjusted_radii("cafe", "suburban")
    rural = density_adjusted_radii("cafe", "rural")
    assert urban == RADIUS_PRIORS_KM["cafe"]
    assert all(u < s < r for u, s, r in zip(urban, suburban, rural))
    assert density_adjusted_radii("ecommerce", "rural") == ()
    with pytest.raises(ValueError):
        density_adjusted_radii("coffee", "urban")
    with pytest.raises(ValueError):
        density_adjusted_radii("cafe", "dense")


def test_median_radius_and_default_lambda() -> None:
    assert industry_median_radius_km("cafe") == 0.75
    assert industry_median_radius_km("hair_salon", "rural") == 6.0
    assert industry_median_radius_km("ecommerce") is None
    assert industry_decay_lambda("cafe") == pytest.approx(math.log(2) / 0.75)
    assert industry_decay_lambda("ecommerce") == 0.0


# --- distance decay --------------------------------------------------------------------------


def test_decay_lambda_halves_weight_at_the_median_radius() -> None:
    for radius in (0.25, 0.75, 3.0, 12.5):
        lam = decay_lambda(radius)
        assert float(distance_weights(radius, lam)) == pytest.approx(0.5)
        assert float(distance_weights(2 * radius, lam)) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        decay_lambda(0.0)
    with pytest.raises(ValueError):
        decay_lambda(float("inf"))


def test_distance_weights() -> None:
    w = distance_weights(np.array([0.0, 1.0, 2.0, np.inf]), 0.5)
    np.testing.assert_allclose(w, [1.0, math.exp(-0.5), math.exp(-1.0), 0.0])
    np.testing.assert_array_equal(distance_weights(np.array([0.0, np.inf]), 0.0), [1.0, 1.0])
    with pytest.raises(ValueError):
        distance_weights([-0.1], 1.0)
    with pytest.raises(ValueError):
        distance_weights([np.nan], 1.0)
    with pytest.raises(ValueError):
        distance_weights([1.0], -1.0)


# --- market radius selection (spec §13) ------------------------------------------------------


def test_select_market_radius_is_the_argmax() -> None:
    scores = {0.5: 0.61, 1.0: 0.74, 2.0: 0.70, 3.0: 0.52}
    best, evaluated = select_market_radius((3.0, 0.5, 2.0, 1.0), scores.__getitem__)
    assert best == 1.0
    assert evaluated == scores
    assert list(evaluated) == [0.5, 1.0, 2.0, 3.0]


def test_select_market_radius_ties_go_to_the_smaller_radius() -> None:
    best, _ = select_market_radius([5.0, 2.0, 1.0], lambda r: 0.8 if r >= 2.0 else 0.3)
    assert best == 2.0
    best, _ = select_market_radius([1.0, 2.0], lambda r: 0.5)
    assert best == 1.0


def test_select_market_radius_tolerance_and_nan() -> None:
    scores = {0.5: 0.700, 1.0: 0.709, 2.0: 0.718, 3.0: float("nan")}
    assert select_market_radius(scores, scores.__getitem__)[0] == 2.0
    assert select_market_radius(scores, scores.__getitem__, tie_tolerance=0.01)[0] == 1.0
    with pytest.raises(ValueError, match="NaN"):
        select_market_radius([1.0, 2.0], lambda r: float("nan"))


def test_select_market_radius_handles_non_geographic_industries() -> None:
    with pytest.raises(ValueError, match="ecommerce"):
        select_market_radius(density_adjusted_radii("ecommerce", "urban"), lambda r: 1.0)
    with pytest.raises(ValueError):
        select_market_radius([0.0, 1.0], lambda r: 1.0)


# --- customer geography (spec §11) -----------------------------------------------------------


def test_distance_band_shares_default_bands() -> None:
    d = [0.1, 0.49, 0.5, 0.99, 1.0, 1.99, 2.0, 4.99, 5.0, 12.0, np.nan]
    shares = distance_band_shares(d)
    assert list(shares.index) == ["0-0.5km", "0.5-1km", "1-2km", "2-5km", "5+km"]
    np.testing.assert_allclose(shares.to_numpy(), [0.2, 0.2, 0.2, 0.2, 0.2])


def test_distance_band_shares_with_weights_and_custom_bands() -> None:
    shares = distance_band_shares([0.2, 0.7, 3.0], bands=(1.0, 2.0), weights=[1.0, 3.0, 4.0])
    assert list(shares.index) == ["0-1km", "1-2km", "2+km"]
    np.testing.assert_allclose(shares.to_numpy(), [0.5, 0.0, 0.5])
    assert shares.sum() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        distance_band_shares([0.5], bands=(2.0, 1.0))
    with pytest.raises(ValueError):
        distance_band_shares([-1.0])
    with pytest.raises(ValueError):
        distance_band_shares([np.nan])


def test_fit_distance_decay_recovers_a_known_lambda(rng: np.random.Generator) -> None:
    for true_lambda in (0.4, 1.3, 3.0):
        distances = rng.exponential(scale=1.0 / true_lambda, size=20_000)
        assert fit_distance_decay(distances) == pytest.approx(true_lambda, rel=0.05)


def test_fit_distance_decay_with_weights_matches_repeated_observations() -> None:
    weighted = fit_distance_decay([0.5, 2.0], weights=[3.0, 1.0])
    assert weighted == pytest.approx(fit_distance_decay([0.5, 0.5, 0.5, 2.0]))
    assert weighted == pytest.approx(4.0 / 3.5)
    with pytest.raises(ValueError):
        fit_distance_decay([0.0, 0.0])


def test_catchment_overlap_is_one_at_zero_distance_and_decreasing() -> None:
    d = np.array([0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 40.0, np.inf])
    w = catchment_overlap(d, 1.2)
    assert w[0] == 1.0
    assert np.all(np.diff(w) < 0)
    assert w[-1] == 0.0 and w[-2] < 1e-9
    np.testing.assert_array_equal(catchment_overlap(d, 0.0), np.ones_like(d))


def test_catchment_overlap_matches_monte_carlo(rng: np.random.Generator) -> None:
    # Twice the share of i's customers (exponential distance, uniform bearing) that live closer
    # to a competitor d km away than to i.
    lam, n = 1.5, 400_000
    r = rng.exponential(1.0 / lam, n)
    x = r * np.cos(rng.uniform(0.0, 2.0 * np.pi, n))
    for d in (0.2, 0.8, 2.5):
        simulated = 2.0 * np.mean(x > d / 2.0)
        assert float(catchment_overlap(d, lam)) == pytest.approx(simulated, abs=0.006)


# --- demographics (spec §12) -----------------------------------------------------------------


def test_weighted_demographics_normalises_weights() -> None:
    values = pd.DataFrame(
        {"median_income": [100.0, 200.0, 400.0], "renter_share": [0.2, 0.4, 0.6]},
        index=["cell_a", "cell_b", "cell_c"],
    )
    out = weighted_demographics(values, [2.0, 1.0, 1.0])
    assert out["median_income"] == pytest.approx(0.5 * 100 + 0.25 * 200 + 0.25 * 400)
    assert out["renter_share"] == pytest.approx(0.5 * 0.2 + 0.25 * 0.4 + 0.25 * 0.6)
    pd.testing.assert_series_equal(out, weighted_demographics(values, [0.5, 0.25, 0.25]))


def test_weighted_demographics_renormalises_over_available_cells() -> None:
    values = pd.DataFrame(
        {
            "median_income": [100.0, np.nan, 400.0],
            "unemployment": [np.nan, np.nan, np.nan],
            "household_size": [np.nan, 5.0, 3.0],
        },
        index=["a", "b", "c"],
    )
    weights = pd.Series({"c": 1.0, "a": 3.0, "b": 0.0})  # aligned by index, not position
    out = weighted_demographics(values, weights)
    assert out["median_income"] == pytest.approx((3 * 100 + 1 * 400) / 4)
    assert math.isnan(out["unemployment"])
    assert out["household_size"] == pytest.approx(3.0)  # cell b has zero weight


def test_weighted_demographics_validation() -> None:
    values = pd.DataFrame({"x": [1.0, 2.0]}, index=["a", "b"])
    with pytest.raises(ValueError):
        weighted_demographics(values, [1.0])
    with pytest.raises(ValueError):
        weighted_demographics(values, [0.0, 0.0])
    with pytest.raises(ValueError):
        weighted_demographics(values, [1.0, -1.0])
    with pytest.raises(ValueError):
        weighted_demographics(values, pd.Series({"a": 1.0}))
