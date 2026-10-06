"""The synthetic generator respects its stated data-generating process."""

import numpy as np
import pandas as pd
import pytest

from laee import schema
from laee.synthetic import COST_SHIFTER, COVARIATE_COLUMNS, DEMAND_PROXY, simulate_panel

SALES_COLUMNS = (
    schema.BUSINESS_ID, schema.PRODUCT_ID, schema.LOCATION_ID, schema.DATE, schema.PRICE,
    schema.UNITS, schema.REVENUE, schema.UNIT_COST, schema.PROMO_FLAG, schema.DISCOUNT_PCT,
    schema.REGULAR_PRICE, schema.INVENTORY, schema.STOCKOUT_FLAG, schema.CENSORED_FLAG,
)


def _within(values: pd.Series, groups: pd.Series) -> pd.Series:
    return values - values.groupby(groups).transform("mean")


def test_panel_has_canonical_columns_and_shape(rng):
    panel, truth = simulate_panel(rng, n_products=7, n_days=30)
    assert set(SALES_COLUMNS) | {DEMAND_PROXY, COST_SHIFTER} <= set(panel.columns)
    assert len(panel) == 7 * 30
    assert panel[schema.PRODUCT_ID].nunique() == 7
    for flag in (schema.PROMO_FLAG, schema.STOCKOUT_FLAG, schema.CENSORED_FLAG):
        assert panel[flag].dtype == bool
    assert (panel[schema.PRICE] > 0).all() and (panel[schema.UNITS] > 0).all()
    assert np.allclose(panel[schema.REVENUE], panel[schema.PRICE] * panel[schema.UNITS])
    dates = pd.to_datetime(panel[schema.DATE]).sort_values().unique()
    assert len(dates) == 30 and (np.diff(dates) == np.timedelta64(1, "D")).all()
    assert str(dates[0])[:10] == "2025-01-01"
    assert list(truth["product_covariates"].columns) == list(COVARIATE_COLUMNS)
    assert len(truth["latent_demand"]) == len(truth["expected_demand"]) == len(panel)


def test_same_seed_reproduces_panel_and_other_seed_differs():
    first, _ = simulate_panel(np.random.default_rng(7), n_products=5, n_days=40)
    again, _ = simulate_panel(np.random.default_rng(7), n_products=5, n_days=40)
    other, _ = simulate_panel(np.random.default_rng(8), n_products=5, n_days=40)
    pd.testing.assert_frame_equal(first, again)
    assert not np.allclose(first[schema.UNITS], other[schema.UNITS])


def test_noiseless_panel_satisfies_structural_demand_equation(rng):
    panel, truth = simulate_panel(rng, n_products=6, n_days=90, noise_sd=0.0)
    alpha = panel[schema.PRODUCT_ID].map(truth["product_fe"])
    beta = panel[schema.PRODUCT_ID].map(truth["elasticity_by_product"])
    time_effect = truth["time_effect"].reindex(pd.to_datetime(panel[schema.DATE])).to_numpy()
    predicted = (
        alpha
        + beta * np.log(panel[schema.PRICE])
        + time_effect
        + truth["promo_lift"] * panel[schema.PROMO_FLAG]
        + truth["proxy_coefficient"] * panel[DEMAND_PROXY]
    )
    np.testing.assert_allclose(np.log(panel[schema.UNITS]), predicted, atol=1e-10)
    np.testing.assert_allclose(panel[schema.UNITS], truth["expected_demand"], rtol=1e-12)
    assert truth["elasticity"] == -1.5 and not truth["heterogeneous"]


def test_heterogeneous_elasticities_follow_the_callable_and_are_recoverable(rng):
    def rule(covariates: pd.DataFrame) -> np.ndarray:
        return -1.2 - 0.4 * covariates["quality"].to_numpy()

    panel, truth = simulate_panel(rng, n_products=8, n_days=200, noise_sd=0.0, elasticity=rule)
    expected = rule(truth["product_covariates"])
    assert truth["heterogeneous"]
    np.testing.assert_allclose(list(truth["elasticity_by_product"].values()), expected)
    assert truth["elasticity"] == pytest.approx(expected.mean())
    time_effect = truth["time_effect"].reindex(pd.to_datetime(panel[schema.DATE])).to_numpy()
    adjusted = (
        np.log(panel[schema.UNITS]) - time_effect - panel[DEMAND_PROXY]
        - truth["promo_lift"] * panel[schema.PROMO_FLAG]
    )
    for product, rows in panel.groupby(schema.PRODUCT_ID).groups.items():
        slope = np.polyfit(np.log(panel.loc[rows, schema.PRICE]), adjusted[rows], 1)[0]
        assert slope == pytest.approx(truth["elasticity_by_product"][product], abs=1e-8)


def _within_correlations(confounding: float, seed: int) -> tuple[float, float]:
    panel, _ = simulate_panel(np.random.default_rng(seed), n_products=10, n_days=365,
                              confounding=confounding)
    products = panel[schema.PRODUCT_ID]
    log_price = _within(np.log(panel[schema.REGULAR_PRICE]), products)
    corr_proxy = np.corrcoef(log_price, _within(panel[DEMAND_PROXY], products))[0, 1]
    corr_cost = np.corrcoef(log_price, _within(panel[COST_SHIFTER], products))[0, 1]
    return corr_proxy, corr_cost


def test_confounding_moves_price_with_the_demand_proxy_and_cost_shifter_is_relevant():
    # Sample correlations between persistent AR(1) series are noisy (spurious-regression
    # problem), so the population sign is checked on the average over seeds.
    confounded = np.array([_within_correlations(0.8, seed) for seed in range(20)])
    exogenous = np.array([_within_correlations(0.0, seed) for seed in range(20)])
    assert confounded[:, 0].mean() > 0.25
    assert abs(exogenous[:, 0].mean()) < 0.08
    assert confounded[:, 1].mean() > 0.3 and exogenous[:, 1].mean() > 0.5


def test_prices_are_sticky_with_temporary_promotions(rng):
    panel, truth = simulate_panel(rng, n_products=20, n_days=365, promo_rate=0.08, promo_depth=0.25)
    changes = panel.groupby(schema.PRODUCT_ID)[schema.REGULAR_PRICE].agg(
        lambda prices: int((prices.diff() != 0).sum() - 1)
    )
    assert 2 <= changes.mean() <= 25
    assert truth["realized_promo_share"] == pytest.approx(0.08, abs=0.02)
    promo = panel[schema.PROMO_FLAG]
    ratio = panel[schema.PRICE] / panel[schema.REGULAR_PRICE]
    np.testing.assert_allclose(ratio[promo], 0.75, atol=0.01)
    np.testing.assert_allclose(panel.loc[promo, schema.DISCOUNT_PCT], 0.25, atol=0.01)
    assert (ratio[~promo] == 1.0).all() and (panel.loc[~promo, schema.DISCOUNT_PCT] == 0).all()
    cents = np.round(panel[schema.REGULAR_PRICE] * 100).astype(int) % 10
    assert (cents == 9).all()


def test_stockouts_censor_latent_demand():
    panel, truth = simulate_panel(np.random.default_rng(3), stockout_rate=0.3)
    units, demand = panel[schema.UNITS].to_numpy(), truth["latent_demand"]
    censored, stockout = panel[schema.CENSORED_FLAG].to_numpy(), panel[schema.STOCKOUT_FLAG].to_numpy()
    inventory = panel[schema.INVENTORY].to_numpy()
    assert (units <= demand + 1e-12).all()
    assert (units[censored] < demand[censored]).all()
    np.testing.assert_allclose(units[stockout], inventory[stockout])
    assert not (censored & ~stockout).any()
    unconstrained = np.isnan(inventory)
    np.testing.assert_allclose(units[unconstrained], demand[unconstrained])
    assert 0.2 < truth["realized_censored_share"] < 0.3
    assert truth["realized_censored_share"] == pytest.approx(censored.mean())


def test_parameters_reuse_the_same_random_draws():
    clean, clean_truth = simulate_panel(np.random.default_rng(11), stockout_rate=0.0)
    stocked, stocked_truth = simulate_panel(np.random.default_rng(11), stockout_rate=0.25)
    np.testing.assert_array_equal(clean_truth["latent_demand"], stocked_truth["latent_demand"])
    assert not clean[schema.STOCKOUT_FLAG].any() and clean[schema.INVENTORY].isna().all()
    assert stocked[schema.STOCKOUT_FLAG].any()


def test_sparsity_drops_rows_at_random(rng):
    panel, truth = simulate_panel(rng, n_products=20, n_days=365, sparsity=0.3)
    assert len(panel) / (20 * 365) == pytest.approx(0.7, abs=0.02)
    assert len(truth["latent_demand"]) == len(panel)
    per_product = panel.groupby(schema.PRODUCT_ID).size()
    assert per_product.min() > 200


def test_poisson_noise_gives_zero_heavy_counts_with_the_right_mean(rng):
    panel, truth = simulate_panel(rng, n_products=30, noise="poisson", noise_sd=0.0, base_demand=1.0)
    units = panel[schema.UNITS].to_numpy()
    assert np.array_equal(units, np.round(units))
    assert (units == 0).mean() > 0.25
    assert units.mean() == pytest.approx(truth["expected_demand"].mean(), rel=0.05)


def test_lognormal_noise_is_mean_one_with_requested_sd(rng):
    panel, truth = simulate_panel(rng, n_products=30, noise_sd=0.3)
    log_ratio = np.log(panel[schema.UNITS].to_numpy() / truth["expected_demand"])
    assert log_ratio.std() == pytest.approx(0.3, abs=0.01)
    assert log_ratio.mean() == pytest.approx(-0.045, abs=0.01)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"noise": "gaussian"},
        {"promo_rate": 1.0},
        {"stockout_rate": -0.1},
        {"sparsity": 1.0},
        {"noise_sd": -1.0},
        {"elasticity": lambda covariates: [-1.0, -2.0]},
        {"elasticity": float("nan")},
    ],
)
def test_invalid_arguments_are_rejected(rng, kwargs):
    with pytest.raises(ValueError):
        simulate_panel(rng, n_products=5, n_days=10, **kwargs)
