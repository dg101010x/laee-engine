import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from laee.simulation import (
    ElasticityBelief,
    cvar,
    cvar_shortfall,
    downside_probability,
    price_grid,
    risk_profile,
    simulate_prices,
    standard_deviation,
    stepped_prices,
    variance,
)
from laee.types import ElasticityEstimate, Interval


def _estimate(
    value: float, std_error: float, intervals: tuple[Interval, ...] = ()
) -> ElasticityEstimate:
    return ElasticityEstimate(
        value=value,
        std_error=std_error,
        intervals=list(intervals),
        method="test",
        identification_tier=1,
        n_obs=200,
        n_products=1,
    )


def test_default_grid_spans_twenty_percent_in_half_percent_steps():
    grid = price_grid(10.0)
    assert grid.size == 81
    assert grid[0] == 8.0 and grid[-1] == 12.0
    np.testing.assert_allclose(np.diff(grid), 0.05)
    assert grid[40] == 10.0


@pytest.mark.parametrize("price", [9.99, 1 / 3, 1234.567, 0.07, 250.0])
def test_grid_contains_current_price_exactly_once(price):
    for grid in (price_grid(price), price_grid(price, (0.01, 0.05), 0.01)):
        assert np.count_nonzero(grid == price) == 1
        assert np.all(np.diff(grid) > 0)


def test_stepped_prices_keep_round_prices_round():
    assert stepped_prices(10.0, [30, -20, 0], 0.005).tolist() == [11.5, 9.0, 10.0]
    assert stepped_prices(9.99, [1], 0.15)[0] == 11.4885


def test_custom_grid_gets_current_price_inserted(rng):
    sim = simulate_prices(
        10.0, 50.0, ElasticityBelief.normal(-1.5, 0.2), grid=[12.0, 9.0, 11.0], n_draws=100, rng=rng
    )
    assert sim.prices.tolist() == [9.0, 10.0, 11.0, 12.0]
    assert sim.current_index == 1


def test_point_mass_belief_reproduces_constant_elasticity_formulas(rng):
    elasticity, p0, q0, cost, fixed = -1.8, 10.0, 200.0, 6.0, 150.0
    sim = simulate_prices(
        p0,
        q0,
        ElasticityBelief.normal(elasticity, 0.0),
        unit_cost=cost,
        fixed_cost=fixed,
        n_draws=50,
        rng=rng,
    )
    quantity = q0 * (sim.prices / p0) ** elasticity
    profit = (sim.prices - cost) * quantity - fixed
    profit0 = (p0 - cost) * q0 - fixed
    np.testing.assert_allclose(sim.quantity.mean, quantity, rtol=1e-12)
    np.testing.assert_allclose(sim.revenue.mean, sim.prices * quantity, rtol=1e-12)
    np.testing.assert_allclose(sim.profit.mean, profit, rtol=1e-12)
    np.testing.assert_allclose(sim.quantity.change_pct, 100 * (quantity - q0) / q0, atol=1e-9)
    np.testing.assert_allclose(
        sim.revenue.change_pct, 100 * (sim.prices * quantity / (p0 * q0) - 1), atol=1e-9
    )
    np.testing.assert_allclose(sim.profit.change_pct, 100 * (profit - profit0) / profit0, atol=1e-9)
    np.testing.assert_allclose(sim.profit.p05, sim.profit.p95, rtol=1e-12)
    # Profit rises with price below the Lerner price 6 * 1.8 / 0.8 = 13.5.
    np.testing.assert_array_equal(sim.prob_improves, (sim.prices > p0).astype(float))


def test_expected_quantity_matches_lognormal_moment(rng):
    # For a normal belief, E[(P/P0)**eps] = exp(mu*x + sigma**2 * x**2 / 2) with x = ln(P/P0).
    mu, sigma = -1.5, 0.4
    sim = simulate_prices(
        10.0,
        100.0,
        ElasticityBelief.normal(mu, sigma),
        grid=[8.0, 9.0, 11.0, 12.0],
        n_draws=200_000,
        rng=rng,
    )
    x = np.log(sim.prices / 10.0)
    np.testing.assert_allclose(
        sim.quantity.mean, 100.0 * np.exp(mu * x + 0.5 * sigma**2 * x**2), rtol=2e-3
    )


def test_current_price_row_is_certain(rng):
    sim = simulate_prices(10.0, 100.0, ElasticityBelief.normal(-1.2, 0.5), unit_cost=7.0, rng=rng)
    i = sim.current_index
    assert sim.prices[i] == 10.0
    assert np.all(sim.quantity_draws[i] == 100.0)
    assert sim.profit.mean[i] == sim.profit.baseline == 300.0
    assert (
        sim.quantity.change_pct[i] == sim.revenue.change_pct[i] == sim.profit.change_pct[i] == 0.0
    )
    assert sim.prob_improves[i] == 0.0
    assert (
        sim.risk.std[i] == 0.0 and sim.risk.shortfall[i] == 0.0 and sim.risk.downside_prob[i] == 0.0
    )


def test_prob_profit_improves_matches_analytic_threshold(rng):
    # Above the current price, profit improves iff eps > ln((P0 - c) / (P - c)) / ln(P / P0).
    mu, sd, p0, cost, price = -2.0, 0.5, 10.0, 6.0, 11.0
    sim = simulate_prices(
        p0,
        100.0,
        ElasticityBelief.normal(mu, sd),
        unit_cost=cost,
        grid=[price],
        n_draws=100_000,
        rng=rng,
    )
    threshold = math.log((p0 - cost) / (price - cost)) / math.log(price / p0)
    expected = 1.0 - stats.norm.cdf(threshold, mu, sd)
    assert sim.prob_improves[sim.prices == price].item() == pytest.approx(expected, abs=0.005)


def test_unknown_cost_simulates_revenue_only(rng):
    sim = simulate_prices(10.0, 100.0, ElasticityBelief.normal(-0.6, 0.1), n_draws=500, rng=rng)
    assert sim.profit is None and sim.profit_draws is None and sim.unit_cost is None
    assert sim.improvement_metric == "revenue"
    frame = sim.to_frame()
    assert not [column for column in frame.columns if "profit" in column]
    assert {"prob_revenue_improves", "revenue_change_cvar", "revenue_shortfall"} <= set(
        frame.columns
    )
    assert (frame.loc[frame.price > 10.0, "prob_revenue_improves"] > 0.99).all()


def test_to_frame_is_tidy(rng):
    sim = simulate_prices(
        250.0, 30.0, ElasticityBelief.normal(-1.4, 0.3), unit_cost=150.0, n_draws=1000, rng=rng
    )
    frame = sim.to_frame()
    assert len(frame) == sim.prices.size
    assert frame["is_current"].sum() == 1
    assert frame.loc[frame["is_current"], "price"].item() == 250.0
    assert frame.loc[frame["is_current"], "price_change_pct"].item() == 0.0
    for name in ("quantity", "revenue", "profit"):
        assert {
            f"expected_{name}",
            f"{name}_p05",
            f"{name}_p50",
            f"{name}_p95",
            f"{name}_change_pct",
        } <= set(frame)
        assert (frame[f"{name}_p05"] <= frame[f"{name}_p50"]).all()
        assert (frame[f"{name}_p50"] <= frame[f"{name}_p95"]).all()
    assert {
        "prob_profit_improves",
        "prob_profit_declines",
        "profit_change_sd",
        "profit_shortfall",
    } <= set(frame)


def test_simulation_is_reproducible_from_seed():
    belief = ElasticityBelief.student_t(-1.5, 0.3, 5)
    first = simulate_prices(
        10.0, 100.0, belief, unit_cost=5.0, n_draws=500, rng=np.random.default_rng(3)
    )
    second = simulate_prices(
        10.0, 100.0, belief, unit_cost=5.0, n_draws=500, rng=np.random.default_rng(3)
    )
    pd.testing.assert_frame_equal(first.to_frame(), second.to_frame())


def test_negative_current_profit_uses_absolute_denominator(rng):
    sim = simulate_prices(
        10.0,
        100.0,
        ElasticityBelief.normal(-0.5, 0.0),
        unit_cost=8.0,
        fixed_cost=500.0,
        n_draws=10,
        rng=rng,
    )
    assert sim.profit.baseline == -300.0
    assert np.all(sim.profit.change_pct[sim.prices > 10.0] > 0.0)
    assert np.all(sim.profit.change_pct[sim.prices < 10.0] < 0.0)


def test_simulation_risk_matches_profit_change_draws(rng):
    sim = simulate_prices(
        10.0, 100.0, ElasticityBelief.normal(-2.0, 0.5), unit_cost=6.0, n_draws=2000, rng=rng
    )
    changes = sim.profit_draws - sim.profit.baseline
    assert sim.risk.alpha == 0.10
    np.testing.assert_allclose(sim.risk.variance, changes.var(axis=1))
    np.testing.assert_allclose(sim.risk.std, changes.std(axis=1))
    np.testing.assert_allclose(sim.risk.cvar, cvar(changes, 0.10))
    np.testing.assert_allclose(sim.risk.downside_prob, (changes < 0).mean(axis=1))
    i = sim.current_index
    assert np.all(np.diff(sim.risk.std[i:]) > 0) and np.all(np.diff(sim.risk.std[: i + 1]) < 0)


def test_simulate_accepts_elasticity_estimate(rng):
    sim = simulate_prices(10.0, 100.0, _estimate(-1.5, 0.2), unit_cost=5.0, n_draws=4000, rng=rng)
    assert sim.elasticity_draws.mean() == pytest.approx(-1.5, abs=0.02)
    assert sim.elasticity_draws.std() == pytest.approx(0.2, rel=0.05)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"current_price": 0.0},
        {"base_quantity": -1.0},
        {"unit_cost": -1.0},
        {"fixed_cost": -5.0},
        {"n_draws": 0},
        {"grid": [-1.0, 10.0]},
        {"rel_range": (-1.5, 0.2)},
        {"step": 0.0},
    ],
)
def test_simulate_rejects_invalid_inputs(kwargs, rng):
    args = {"current_price": 10.0, "base_quantity": 100.0} | kwargs
    with pytest.raises(ValueError):
        simulate_prices(
            args.pop("current_price"),
            args.pop("base_quantity"),
            ElasticityBelief.normal(-1.5, 0.2),
            rng=rng,
            **args,
        )


def test_normal_belief(rng):
    belief = ElasticityBelief.normal(-1.5, 0.3)
    draws = belief.draw(rng, 100_000)
    assert draws.mean() == pytest.approx(-1.5, abs=0.01)
    assert draws.std() == pytest.approx(0.3, rel=0.01)
    z = stats.norm.ppf(0.975)
    assert belief.interval(0.95) == pytest.approx((-1.5 - z * 0.3, -1.5 + z * 0.3))
    assert belief.prob_below(-1.0) == pytest.approx(stats.norm.cdf(-1.0, -1.5, 0.3))
    assert belief.mean == -1.5 and belief.sd == 0.3


def test_point_mass_belief():
    belief = ElasticityBelief.normal(-2.0, 0.0)
    assert belief.interval(0.95) == (-2.0, -2.0)
    assert belief.prob_below(-1.0) == 1.0 and belief.prob_below(-3.0) == 0.0
    assert np.all(belief.draw(np.random.default_rng(0), 10) == -2.0)


def test_student_t_belief(rng):
    belief = ElasticityBelief.student_t(-1.5, 0.3, 5)
    assert belief.sd == pytest.approx(0.3 * math.sqrt(5 / 3))
    half = stats.t.ppf(0.975, 5) * 0.3
    assert belief.interval(0.95) == pytest.approx((-1.5 - half, -1.5 + half))
    assert belief.prob_below(-1.0) == pytest.approx(stats.t.cdf((-1.0 + 1.5) / 0.3, 5))
    assert belief.draw(rng, 200_000).std() == pytest.approx(belief.sd, rel=0.03)


def test_empirical_belief(rng):
    samples = rng.normal(-2.0, 0.4, 5000)
    belief = ElasticityBelief.from_samples(samples)
    assert belief.mean == pytest.approx(samples.mean())
    assert belief.sd == pytest.approx(samples.std(ddof=1))
    assert belief.interval(0.9) == pytest.approx(tuple(np.quantile(samples, [0.05, 0.95])))
    assert belief.prob_below(-2.0) == pytest.approx(np.mean(samples < -2.0))
    assert set(belief.draw(rng, 1000)) <= set(samples)
    samples[0] = 99.0
    assert belief.samples[0] != 99.0


def test_from_estimate_uses_std_error_without_understating_intervals():
    z = stats.norm.ppf(0.975)
    consistent = ElasticityBelief.from_estimate(
        _estimate(-1.63, 0.15, (Interval(-1.92, -1.34, 0.95),))
    )
    assert consistent.family == "normal" and consistent.mean == -1.63 and consistent.sd == 0.15
    wide = ElasticityBelief.from_estimate(_estimate(-1.63, 0.05, (Interval(-2.2, -1.0, 0.95),)))
    assert wide.sd == pytest.approx(1.2 / (2 * z))
    interval_only = ElasticityBelief.from_estimate(
        _estimate(-1.2, float("nan"), (Interval(-1.5, -0.9, 0.8),))
    )
    assert interval_only.sd == pytest.approx(0.6 / (2 * stats.norm.ppf(0.9)))
    heavy = ElasticityBelief.from_estimate(_estimate(-1.2, 0.2), df=8)
    assert heavy.family == "student_t" and heavy.scale == 0.2 and heavy.df == 8


def test_from_estimate_rejects_estimates_without_uncertainty():
    with pytest.raises(ValueError, match="Rule 4"):
        ElasticityBelief.from_estimate(_estimate(-1.2, float("nan")))
    with pytest.raises(ValueError, match="Rule 4"):
        ElasticityBelief.from_estimate(_estimate(-1.2, 0.0, (Interval(-1.2, -1.2, 0.95),)))


@pytest.mark.parametrize(
    "build",
    [
        lambda: ElasticityBelief.normal(-1.0, -0.1),
        lambda: ElasticityBelief.normal(float("nan"), 0.1),
        lambda: ElasticityBelief.student_t(-1.0, 0.2, 2),
        lambda: ElasticityBelief.from_samples([-1.0]),
        lambda: ElasticityBelief.from_samples([-1.0, float("nan")]),
        lambda: ElasticityBelief("lognormal", -1.0, 0.1),
    ],
)
def test_belief_validation(build):
    with pytest.raises(ValueError):
        build()


@pytest.mark.parametrize("alpha", [0.05, 0.10, 0.25])
def test_cvar_matches_normal_closed_form(alpha):
    mu, sigma = 2.0, 3.0
    draws = np.random.default_rng(11).normal(mu, sigma, 1_000_000)
    expected = mu - sigma * stats.norm.pdf(stats.norm.ppf(alpha)) / alpha
    assert cvar(draws, alpha) == pytest.approx(expected, abs=0.02)
    assert cvar_shortfall(draws, alpha) == pytest.approx(max(0.0, -expected), abs=0.02)


def test_cvar_is_exact_for_fractional_tails():
    x = np.random.default_rng(0).permutation(np.arange(1.0, 11.0))
    assert cvar(x, 0.25) == pytest.approx((1 + 2 + 0.5 * 3) / 2.5)
    assert cvar(x, 0.10) == 1.0
    assert cvar(x, 0.05) == 1.0
    assert cvar(x, 1.0) == pytest.approx(5.5)


def test_risk_measures_are_vectorised_over_rows():
    rng = np.random.default_rng(5)
    matrix = rng.normal(0.5, 1.0, size=(4, 1000)) * np.array([[1.0], [2.0], [3.0], [4.0]])
    np.testing.assert_allclose(variance(matrix), matrix.var(axis=1))
    np.testing.assert_allclose(standard_deviation(matrix), matrix.std(axis=1))
    np.testing.assert_allclose(cvar(matrix, 0.1), np.sort(matrix, axis=1)[:, :100].mean(axis=1))
    np.testing.assert_allclose(downside_probability(matrix), (matrix < 0).mean(axis=1))
    profile = risk_profile(matrix, 0.1)
    np.testing.assert_allclose(profile.cvar, cvar(matrix, 0.1))
    np.testing.assert_allclose(profile.shortfall, np.maximum(0.0, -profile.cvar))
    assert profile.alpha == 0.1


def test_shortfall_is_zero_when_worst_outcomes_are_gains():
    assert cvar_shortfall(np.array([1.0, 2.0, 3.0]), 0.5) == 0.0
    assert cvar_shortfall(np.array([-4.0, 2.0, 3.0, 5.0]), 0.25) == 4.0
    assert downside_probability(np.array([-1.0, 0.0, 1.0, 2.0])) == 0.25


@pytest.mark.parametrize("alpha", [0.0, -0.1, 1.5])
def test_cvar_rejects_bad_alpha(alpha):
    with pytest.raises(ValueError):
        cvar(np.ones(10), alpha)
