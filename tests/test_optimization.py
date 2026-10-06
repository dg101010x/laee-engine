import math

import numpy as np
import pytest
from scipy import stats

from laee.optimization import (
    PriceConstraints,
    candidate_prices,
    experiment_suggestion,
    optimize_price,
    price_bounds,
    probe_prices,
)
from laee.simulation import ElasticityBelief
from laee.types import ElasticityEstimate, Interval

WIDE = PriceConstraints(min_margin=0.0, max_rel_change=0.6)
STEP = 0.005


def _optimize(
    belief,
    unit_cost,
    constraints=WIDE,
    *,
    current_price=10.0,
    base_quantity=100.0,
    seed=0,
    **kwargs,
):
    return optimize_price(
        current_price,
        base_quantity,
        belief,
        unit_cost,
        constraints,
        confidence=0.85,
        rng=np.random.default_rng(seed),
        **kwargs,
    )


@pytest.mark.parametrize(
    ("elasticity", "unit_cost"), [(-3.0, 6.0), (-2.0, 4.0), (-4.0, 7.5), (-2.5, 5.0), (-1.6, 3.1)]
)
def test_lerner_rule_for_constant_elasticity(elasticity, unit_cost):
    rec = _optimize(ElasticityBelief.normal(elasticity, 0.0), unit_cost)
    lerner = unit_cost * elasticity / (1.0 + elasticity)
    assert rec.status == "ok"
    assert abs(rec.recommended_price - lerner) <= STEP * 10.0 + 1e-12
    assert rec.binding_constraints == ()


@pytest.mark.parametrize("elasticity", [-0.3, -0.6, -0.9])
def test_inelastic_demand_hits_max_rel_change(elasticity):
    rec = _optimize(ElasticityBelief.normal(elasticity, 0.02), 4.0, PriceConstraints())
    assert rec.status == "ok"
    assert rec.recommended_price == 11.5
    assert rec.binding_constraints == ("max_rel_change",)
    assert rec.expected_quantity_change_pct < 0.0
    assert rec.expected_revenue_change_pct > 0.0 and rec.expected_profit_change_pct > 0.0


def test_min_margin_binds_for_very_elastic_demand():
    # The Lerner price 9 * 15 / 14 = 9.64 sits below the 10% markup floor of 9.90.
    rec = _optimize(ElasticityBelief.normal(-15.0, 0.0), 9.0, PriceConstraints())
    assert rec.recommended_price == 9.9
    assert rec.binding_constraints == ("min_margin",)


def test_business_max_binds():
    rec = _optimize(ElasticityBelief.normal(-0.5, 0.0), 4.0, PriceConstraints(business_max=11.0))
    assert rec.recommended_price == 11.0
    assert rec.binding_constraints == ("business_max",)


def test_business_max_below_current_price_binds():
    rec = _optimize(ElasticityBelief.normal(-1.5, 0.0), 5.0, PriceConstraints(business_max=9.0))
    assert rec.recommended_price == 9.0
    assert rec.binding_constraints == ("business_max",)
    assert rec.expected_profit_change_pct < 0.0


def test_conflicting_limits_give_no_feasible_price():
    rec = _optimize(ElasticityBelief.normal(-2.0, 0.2), 11.0, PriceConstraints())
    assert rec.status == "no_feasible_price"
    assert rec.binding_constraints == ("min_margin", "max_rel_change")
    assert rec.recommended_price == rec.current_price == 10.0
    assert rec.diagnostics["bounds"] == (12.1, 11.5)


def test_recommendation_fields_are_consistent():
    belief = ElasticityBelief.normal(-1.8, 0.3)
    rec = _optimize(belief, 5.0, PriceConstraints(), risk_aversion=0.2)
    assert rec.status == "ok"
    assert rec.elasticity == -1.8
    assert rec.elasticity_interval_95 == pytest.approx(belief.interval(0.95))
    assert rec.confidence == 0.85
    frame = rec.diagnostics["evaluated"]
    row = frame.loc[frame.price == rec.recommended_price].iloc[0]
    assert row.feasible
    assert rec.expected_quantity_change_pct == pytest.approx(row.quantity_change_pct)
    assert rec.expected_revenue_change_pct == pytest.approx(row.revenue_change_pct)
    assert rec.expected_profit_change_pct == pytest.approx(row.profit_change_pct)
    assert rec.probability_profit_improves == pytest.approx(row.prob_profit_improves)
    assert rec.risk == pytest.approx(row.profit_shortfall / rec.diagnostics["current_profit"])
    assert row.objective == pytest.approx(frame.loc[frame.feasible, "objective"].max())


def test_optimize_accepts_elasticity_estimate_and_is_reproducible():
    estimate = ElasticityEstimate(
        value=-3.0,
        std_error=0.1,
        intervals=[Interval(-3.2, -2.8, 0.95)],
        method="fe",
        identification_tier=2,
        n_obs=500,
        n_products=1,
    )
    first, second = _optimize(estimate, 6.0, seed=4), _optimize(estimate, 6.0, seed=4)
    assert first.status == "ok" and first.elasticity == -3.0
    assert abs(first.recommended_price - 9.0) <= 0.1
    assert first == second


@pytest.mark.parametrize("measure", ["cvar", "std"])
def test_higher_risk_aversion_moves_towards_current_price(measure):
    belief = ElasticityBelief.normal(-3.0, 1.0)
    recs = [
        _optimize(
            belief,
            6.0,
            PriceConstraints(max_rel_change=0.3),
            risk_aversion=lam,
            risk_measure=measure,
        )
        for lam in (0.0, 0.1, 0.25, 0.5, 2.0)
    ]
    distances = [abs(rec.recommended_price - 10.0) for rec in recs]
    risks = [rec.risk for rec in recs]
    assert distances == sorted(distances, reverse=True)
    assert risks == sorted(risks, reverse=True)
    assert distances[0] >= 1.0
    assert recs[-1].recommended_price == 10.0 and recs[-1].risk == 0.0


def test_risk_aversion_barely_matters_when_uncertainty_is_low():
    belief = ElasticityBelief.normal(-3.0, 0.05)
    rec = _optimize(belief, 6.0, PriceConstraints(max_rel_change=0.3), risk_aversion=2.0)
    assert abs(rec.recommended_price - 9.0) <= STEP * 10.0


def test_price_bounds_follow_spec_66():
    bounds = price_bounds(10.0, 6.0, PriceConstraints(business_min=9.0, business_max=11.0))
    assert (bounds.lower, bounds.upper) == (9.0, 11.0)
    assert bounds.lower_limits == {"max_rel_change": 8.5, "min_margin": 6.6, "business_min": 9.0}
    assert bounds.upper_limits == {"max_rel_change": 11.5, "business_max": 11.0}
    assert bounds.violated_by(8.0) == ("business_min", "max_rel_change")
    assert bounds.violated_by(12.0) == ("max_rel_change", "business_max")
    assert bounds.violated_by(10.0) == ()
    assert "min_margin" not in price_bounds(10.0, None, PriceConstraints()).lower_limits


def _check_snapped(current, cost, constraints, is_allowed):
    bounds = price_bounds(current, cost, constraints)
    prices = candidate_prices(current, bounds, constraints, step=STEP)
    if not bounds.feasible:
        assert prices.size == 0
        return
    assert np.all((prices >= bounds.lower) & (prices <= bounds.upper))
    assert all(is_allowed(p) for p in prices if p != current)
    for probe in probe_prices(current, bounds, constraints, step=STEP):
        assert not bounds.contains(probe) and is_allowed(probe)


def test_retail_endings_never_leave_the_feasible_range():
    rng = np.random.default_rng(42)
    for _ in range(400):
        current = round(float(rng.uniform(0.6, 80.0)), 2)
        cost = round(current * float(rng.uniform(0.1, 0.98)), 2)
        business_max = (
            round(current * float(rng.uniform(0.9, 1.3)), 2) if rng.random() < 0.4 else None
        )
        constraints = PriceConstraints(
            min_margin=float(rng.uniform(0.0, 0.3)),
            max_rel_change=float(rng.uniform(0.01, 0.3)),
            business_max=business_max,
            price_endings="retail_99",
        )
        _check_snapped(current, cost, constraints, lambda p: round(p * 100) % 100 in (49, 99))


@pytest.mark.parametrize("round_to", [5, 10])
def test_whole_unit_rounding_never_leaves_the_feasible_range(round_to):
    rng = np.random.default_rng(round_to)
    for _ in range(400):
        current = float(rng.integers(20, 5000))
        cost = float(rng.integers(1, int(current)))
        business_min = (
            float(rng.integers(int(current * 0.8), int(current) + 1))
            if rng.random() < 0.4
            else None
        )
        constraints = PriceConstraints(
            min_margin=float(rng.uniform(0.0, 0.3)),
            max_rel_change=float(rng.uniform(0.01, 0.3)),
            business_min=business_min,
            round_to=round_to,
        )
        _check_snapped(current, cost, constraints, lambda p: p % round_to == 0)


def test_recommendation_uses_retail_endings():
    rec = _optimize(
        ElasticityBelief.normal(-2.2, 0.0), 6.0, PriceConstraints(price_endings="retail_99")
    )
    # The Lerner price is 11.00; the nearest allowed .49/.99 price inside [8.50, 11.50] is 10.99.
    assert rec.recommended_price == 10.99
    assert rec.binding_constraints == ()


def test_recommendation_uses_whole_currency_units():
    rec = _optimize(
        ElasticityBelief.normal(-2.0, 0.0),
        120.0,
        PriceConstraints(round_to=10),
        current_price=230.0,
    )
    assert rec.recommended_price == 240.0
    assert rec.binding_constraints == ()


def test_current_price_off_the_lattice_can_be_kept():
    # +/-2% of KES 103 holds no multiple of 10, so the only candidate is the current price.
    constraints = PriceConstraints(round_to=10, max_rel_change=0.02)
    rec = _optimize(ElasticityBelief.normal(-0.5, 0.0), 50.0, constraints, current_price=103.0)
    assert rec.status == "ok" and rec.recommended_price == 103.0
    assert rec.binding_constraints == ("max_rel_change",)


def test_empty_lattice_gives_no_feasible_price():
    # The margin floor 10.56 is above the current price and [10.56, 10.70] holds no .49/.99 price.
    constraints = PriceConstraints(max_rel_change=0.07, price_endings="retail_99")
    rec = _optimize(ElasticityBelief.normal(-2.0, 0.1), 9.6, constraints)
    assert rec.status == "no_feasible_price"
    assert rec.binding_constraints == ("price_endings",)


def test_inventory_constraint_drops_prices_that_oversell():
    belief = ElasticityBelief.normal(-3.0, 0.0)
    free = _optimize(belief, 6.0, PriceConstraints(min_margin=0.0, max_rel_change=0.3))
    rec = _optimize(
        belief, 6.0, PriceConstraints(min_margin=0.0, max_rel_change=0.3, max_units=120.0)
    )
    assert free.recommended_price == 9.0
    assert rec.status == "ok"
    assert rec.binding_constraints == ("inventory",)
    assert 100.0 * (1.0 + rec.expected_quantity_change_pct / 100.0) <= 120.0
    cheapest_in_stock = 10.0 * 1.2 ** (-1.0 / 3.0)
    assert 0.0 <= rec.recommended_price - cheapest_in_stock <= STEP * 10.0
    frame = rec.diagnostics["evaluated"]
    assert (frame.loc[frame.feasible, "expected_quantity"] <= 120.0).all()
    assert "inventory" in set(frame.loc[~frame.feasible, "excluded_by"])


def test_inventory_shortfall_everywhere_is_infeasible():
    rec = _optimize(ElasticityBelief.normal(-2.0, 0.0), 5.0, PriceConstraints(max_units=50.0))
    assert rec.status == "no_feasible_price"
    assert rec.binding_constraints == ("inventory",)
    assert rec.recommended_price == 10.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_margin": -0.1},
        {"max_rel_change": 0.0},
        {"max_rel_change": 1.0},
        {"business_min": 12.0, "business_max": 10.0},
        {"price_endings": "retail_95"},
        {"price_endings": "retail_99", "round_to": 5},
        {"round_to": 0},
        {"max_units": -1.0},
    ],
)
def test_constraint_validation(kwargs):
    with pytest.raises(ValueError):
        PriceConstraints(**kwargs)


def test_lattice_neighbours_and_snapping():
    lattice = PriceConstraints(price_endings="retail_99").lattice
    assert lattice.next_above(9.99) == 10.49 and lattice.next_above(10.0) == 10.49
    assert lattice.next_below(9.99) == 9.49 and lattice.next_below(0.49) is None
    np.testing.assert_array_equal(lattice.snap([9.70, 9.80, 12.40], 9.0, 11.0), [9.49, 9.99, 10.99])
    kes = PriceConstraints(round_to=5).lattice
    np.testing.assert_array_equal(
        kes.snap([212.0, 288.0, 251.0], 212.5, 287.5), [215.0, 285.0, 250.0]
    )


def test_needs_cost_keeps_price_and_simulates_revenue_only():
    rec = _optimize(ElasticityBelief.normal(-0.7, 0.2), None, PriceConstraints())
    assert rec.status == "needs_cost"
    assert rec.recommended_price == rec.current_price == 10.0
    assert (
        rec.expected_profit_change_pct == 0.0 and rec.risk == 0.0 and rec.binding_constraints == ()
    )
    assert rec.confidence == 0.85
    assert "Rule 9" in rec.diagnostics["reason"]
    simulation = rec.diagnostics["simulation"]
    assert simulation.profit is None and simulation.improvement_metric == "revenue"
    frame = simulation.to_frame()
    assert "expected_revenue" in frame and "expected_profit" not in frame
    assert (frame.price.min(), frame.price.max()) == (8.5, 11.5)
    # Revenue would rise with a higher price, but revenue alone is never optimised.
    assert frame.loc[frame.price == 11.5, "prob_revenue_improves"].item() > 0.9


def test_uncertain_belief_keeps_current_price_when_gain_is_not_robust():
    rec = _optimize(ElasticityBelief.normal(-1.0, 0.5), 1.0, PriceConstraints())
    assert rec.status == "uncertain"
    assert rec.diagnostics["prob_elastic"] == pytest.approx(0.5)
    assert rec.diagnostics["unrestricted_price"] == 11.5
    assert rec.recommended_price == 10.0
    assert rec.binding_constraints == ()
    frame = rec.diagnostics["evaluated"]
    assert frame.loc[frame.price == 11.5, "prob_profit_improves"].item() < 0.8


def test_uncertain_belief_moves_for_a_robust_gain():
    rec = _optimize(ElasticityBelief.normal(-1.0, 0.3), 5.0, PriceConstraints())
    assert rec.status == "uncertain"
    assert rec.recommended_price == 11.5
    assert rec.probability_profit_improves >= 0.8
    assert rec.binding_constraints == ("max_rel_change",)


@pytest.mark.parametrize(
    ("prob_elastic", "status"),
    [(0.15, "ok"), (0.25, "uncertain"), (0.5, "uncertain"), (0.75, "uncertain"), (0.85, "ok")],
)
def test_uncertain_band(prob_elastic, status):
    sd = 0.3
    mean = -1.0 - sd * stats.norm.ppf(prob_elastic)
    rec = _optimize(ElasticityBelief.normal(mean, sd), 2.0, PriceConstraints())
    assert rec.diagnostics["prob_elastic"] == pytest.approx(prob_elastic)
    assert rec.status == status


def test_confidence_is_required_and_passed_through():
    belief = ElasticityBelief.normal(-2.0, 0.2)
    with pytest.raises(TypeError):
        optimize_price(10.0, 100.0, belief, 5.0, PriceConstraints(), rng=np.random.default_rng(0))
    with pytest.raises(TypeError, match="Rule 12"):
        optimize_price(
            10.0,
            100.0,
            belief,
            5.0,
            PriceConstraints(),
            confidence=None,
            rng=np.random.default_rng(0),
        )
    with pytest.raises(ValueError):
        optimize_price(
            10.0,
            100.0,
            belief,
            5.0,
            PriceConstraints(),
            confidence=1.5,
            rng=np.random.default_rng(0),
        )
    assert _optimize(belief, 5.0, PriceConstraints()).confidence == 0.85


@pytest.mark.parametrize(
    "kwargs",
    [
        {"risk_measure": "variance"},
        {"risk_aversion": -1.0},
        {"fixed_cost": -1.0},
        {"cvar_alpha": 0.0},
        {"step": 0.0},
    ],
)
def test_optimizer_rejects_invalid_options(kwargs):
    with pytest.raises(ValueError):
        _optimize(ElasticityBelief.normal(-2.0, 0.2), 5.0, PriceConstraints(), **kwargs)


def test_experiment_suggestion_narrows_more_with_longer_tests():
    belief = ElasticityBelief.normal(-1.6, 0.5)
    suggestion = experiment_suggestion(
        belief, 10.0, PriceConstraints(), daily_units=200, test_days=(14, 28, 56)
    )
    assert suggestion.recommended and suggestion.reason == "interval_wide"
    assert suggestion.test_price == 10.5 and suggestion.test_change_pct == pytest.approx(5.0)
    widths = [plan.interval_width for plan in suggestion.plans]
    reductions = [plan.width_reduction_pct for plan in suggestion.plans]
    assert suggestion.interval_width > widths[0] > widths[1] > widths[2]
    assert 0.0 < reductions[0] < reductions[1] < reductions[2]
    assert suggestion.days_to_threshold > 56
    assert "+5.0% price test" in suggestion.message


def test_experiment_information_matches_simulated_regression():
    sigma, days, share = 0.3, 28, 0.5
    belief = ElasticityBelief.normal(-1.5, 0.5)
    suggestion = experiment_suggestion(
        belief,
        10.0,
        PriceConstraints(),
        daily_units=100,
        test_days=(days,),
        noise_sd=sigma,
        test_share=share,
    )
    information = suggestion.plans[0].posterior_sd ** -2 - belief.sd**-2
    rng = np.random.default_rng(9)
    reps = 20_000
    on_test = rng.permuted(
        np.tile(np.r_[np.ones(days // 2), np.zeros(days // 2)], (reps, 1)), axis=1
    )
    log_price = on_test * math.log(suggestion.test_price / 10.0)
    log_units = -1.5 * log_price + rng.normal(0.0, sigma, size=(reps, days))
    centred = log_price - log_price.mean(axis=1, keepdims=True)
    slopes = (centred * log_units).sum(axis=1) / (centred**2).sum(axis=1)
    assert slopes.var() == pytest.approx(1.0 / information, rel=0.05)


def test_no_experiment_when_interval_is_narrow():
    suggestion = experiment_suggestion(
        ElasticityBelief.normal(-1.6, 0.1), 10.0, PriceConstraints(), daily_units=50
    )
    assert not suggestion.recommended and suggestion.reason == "interval_narrow"
    assert suggestion.test_price is None and suggestion.plans == ()


def test_experiment_test_price_respects_constraints():
    # An increase is ruled out by business_max, so the test lowers the price by about 5%,
    # snapped to a whole multiple of KES 10.
    constraints = PriceConstraints(round_to=10, business_max=250.0)
    suggestion = experiment_suggestion(
        ElasticityBelief.normal(-1.5, 0.5), 250.0, constraints, daily_units=30, unit_cost=150.0
    )
    assert suggestion.recommended and suggestion.test_price == 240.0


def test_experiment_without_a_feasible_test_price():
    constraints = PriceConstraints(round_to=10, max_rel_change=0.02)
    suggestion = experiment_suggestion(
        ElasticityBelief.normal(-1.5, 0.5), 250.0, constraints, daily_units=30
    )
    assert not suggestion.recommended and suggestion.reason == "no_feasible_test_price"
