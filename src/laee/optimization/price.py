"""Risk-aware price optimisation and experiment suggestions (spec §37-38, §66-67, §78, §89).

:func:`optimize_price` maximises ``E[profit(P)] - risk_aversion * risk(P)`` over the
feasible candidate prices (spec §89) and returns a :class:`Recommendation` shaped
like spec §67. :func:`experiment_suggestion` turns a wide elasticity interval into a
proposed controlled price test (spec §78).
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from laee.optimization.constraints import (
    PriceBounds,
    PriceConstraints,
    candidate_prices,
    order_constraints,
    price_bounds,
    probe_prices,
    within_inventory,
)
from laee.simulation.price import (
    DEFAULT_N_DRAWS,
    DEFAULT_STEP,
    ElasticityBelief,
    SimulationResult,
    as_belief,
    simulate_prices,
    stepped_prices,
)
from laee.simulation.risk import DEFAULT_ALPHA
from laee.types import ElasticityEstimate

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
RiskMeasure = Literal["cvar", "std"]
RecommendationStatus = Literal["ok", "needs_cost", "no_feasible_price", "uncertain"]
ExperimentReason = Literal["interval_wide", "interval_narrow", "no_feasible_test_price"]

UNCERTAIN_PROB_ELASTIC: tuple[float, float] = (0.2, 0.8)
"""The belief is "uncertain" when P(elasticity < -1) falls in this closed band."""
UNCERTAIN_MIN_PROB_IMPROVES = 0.8
"""Under an uncertain belief a price change needs at least this probability of raising profit."""
DEFAULT_WIDTH_THRESHOLD = 0.6
DEFAULT_DAY_NOISE_SD = 0.2
_TIE_TOL = 1e-12

__all__ = [
    "DEFAULT_DAY_NOISE_SD",
    "DEFAULT_WIDTH_THRESHOLD",
    "UNCERTAIN_MIN_PROB_IMPROVES",
    "UNCERTAIN_PROB_ELASTIC",
    "ExperimentPlan",
    "ExperimentSuggestion",
    "Recommendation",
    "experiment_suggestion",
    "optimize_price",
]


@dataclass(frozen=True)
class Recommendation:
    """A price recommendation (spec §67).

    Percent changes compare the recommended price with the current price (``7.8``
    means +7.8%). ``risk`` is the risk measure at the recommended price as a share of
    current profit (``0.21`` means 21%): for ``"cvar"`` the average profit lost in the
    worst ``cvar_alpha`` share of elasticity scenarios, for ``"std"`` the standard
    deviation of the profit change. ``elasticity`` is the belief mean, reported with
    its 95% interval (spec §91 Rule 4). ``confidence`` is the caller's confidence score,
    passed through unchanged (Rule 12).

    ``needs_cost`` and ``no_feasible_price`` keep the current price, so their changes,
    ``probability_profit_improves`` and ``risk`` are all zero.
    """

    current_price: float
    recommended_price: float
    elasticity: float
    elasticity_interval_95: tuple[float, float]
    expected_quantity_change_pct: float
    expected_revenue_change_pct: float
    expected_profit_change_pct: float
    probability_profit_improves: float
    risk: float
    binding_constraints: tuple[str, ...]
    status: RecommendationStatus
    confidence: float
    diagnostics: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_positive(value: float) -> bool:
    return math.isfinite(value) and value > 0.0


def _is_non_negative(value: float) -> bool:
    return math.isfinite(value) and value >= 0.0


def _check_inputs(
    current_price: float,
    base_quantity: float,
    unit_cost: float | None,
    constraints: PriceConstraints,
    risk_aversion: float,
    risk_measure: str,
    fixed_cost: float,
    confidence: float | None,
    n_draws: int,
    step: float,
    cvar_alpha: float,
) -> None:
    if confidence is None:
        raise TypeError(
            "confidence is required: every recommendation carries a confidence score (spec Rule 12)"
        )
    if not isinstance(constraints, PriceConstraints):
        raise TypeError(f"constraints must be PriceConstraints, got {type(constraints).__name__}")
    _require(_is_positive(current_price), f"current_price must be positive, got {current_price!r}")
    _require(_is_positive(base_quantity), f"base_quantity must be positive, got {base_quantity!r}")
    _require(
        unit_cost is None or _is_non_negative(unit_cost),
        f"unit_cost must be non-negative, got {unit_cost!r}",
    )
    _require(
        _is_non_negative(risk_aversion),
        f"risk_aversion must be non-negative, got {risk_aversion!r}",
    )
    _require(
        risk_measure in ("cvar", "std"),
        f"risk_measure must be 'cvar' or 'std', got {risk_measure!r}",
    )
    _require(_is_non_negative(fixed_cost), f"fixed_cost must be non-negative, got {fixed_cost!r}")
    _require(
        math.isfinite(confidence) and 0.0 <= confidence <= 1.0,
        f"confidence must lie in [0, 1], got {confidence!r}",
    )
    _require(operator.index(n_draws) >= 1, f"n_draws must be at least 1, got {n_draws!r}")
    _require(_is_positive(step), f"step must be positive, got {step!r}")
    _require(0.0 < cvar_alpha <= 1.0, f"cvar_alpha must lie in (0, 1], got {cvar_alpha!r}")


def _tolerance(value: float) -> float:
    return _TIE_TOL * max(1.0, abs(value))


def _argmax_nearest(
    objective: FloatArray, allowed: BoolArray, prices: FloatArray, current_price: float
) -> int:
    """Best allowed index; near-ties go to the price closest to the current price."""
    index = np.flatnonzero(allowed)
    values = objective[index]
    best = float(values.max())
    near = index[values >= best - _tolerance(best)]
    return int(near[np.argmin(np.abs(prices[near] - current_price))])


def _nearest(allowed: BoolArray, prices: FloatArray, current_price: float) -> int:
    index = np.flatnonzero(allowed)
    return int(index[np.argmin(np.abs(prices[index] - current_price))])


def _exclusions(price: float, in_stock: bool, bounds: PriceBounds) -> tuple[str, ...]:
    names = bounds.violated_by(price)
    return names if in_stock else order_constraints((*names, "inventory"))


def _hold(
    current_price: float,
    belief: ElasticityBelief,
    confidence: float,
    status: RecommendationStatus,
    binding: tuple[str, ...],
    diagnostics: dict[str, Any],
) -> Recommendation:
    return Recommendation(
        current_price=current_price,
        recommended_price=current_price,
        elasticity=belief.mean,
        elasticity_interval_95=belief.interval(0.95),
        expected_quantity_change_pct=0.0,
        expected_revenue_change_pct=0.0,
        expected_profit_change_pct=0.0,
        probability_profit_improves=0.0,
        risk=0.0,
        binding_constraints=binding,
        status=status,
        confidence=confidence,
        diagnostics=diagnostics,
    )


def optimize_price(
    current_price: float,
    base_quantity: float,
    belief: ElasticityBelief | ElasticityEstimate,
    unit_cost: float | None,
    constraints: PriceConstraints,
    *,
    risk_aversion: float = 0.0,
    risk_measure: RiskMeasure = "cvar",
    fixed_cost: float = 0.0,
    confidence: float,
    rng: np.random.Generator,
    n_draws: int = DEFAULT_N_DRAWS,
    step: float = DEFAULT_STEP,
    cvar_alpha: float = DEFAULT_ALPHA,
) -> Recommendation:
    """Recommend the price maximising ``E[profit] - risk_aversion * risk`` (spec §38, §89).

    Candidates are the ``step`` grid (relative to the current price) inside the bounds of
    spec §66, both bounds themselves, and the current price when it is within the bounds.
    With ``price_endings`` or ``round_to`` set, new prices are snapped to allowed price
    points inside the bounds. Prices whose expected quantity exceeds ``max_units`` are
    dropped. Every candidate is simulated under the same elasticity draws.

    Risk is measured on the profit change relative to the current price, in currency:
    ``"cvar"`` uses the CVaR shortfall (average loss in the worst ``cvar_alpha`` share of
    scenarios, zero if those are still gains); ``"std"`` uses its standard deviation.
    Both are zero at the current price, so a larger ``risk_aversion`` pulls the
    recommendation towards it.

    ``binding_constraints`` lists the constraints that exclude a price the decision rule
    would otherwise have chosen: the next price one step or lattice point beyond either
    bound, a candidate dropped for inventory, or the current price itself.
    ``diagnostics["evaluated"]`` tabulates every simulated price (those excluded prices
    included) with its risk, objective, feasibility and the constraints excluding it;
    ``diagnostics["unrestricted_price"]`` is the choice before the uncertain-status rule.

    Status rules:

    * ``needs_cost``: ``unit_cost`` is None. Revenue is never optimised on its own
      (spec Rule 9); the current price is kept and ``diagnostics["simulation"]`` holds a
      revenue-only simulation of the candidate prices.
    * ``no_feasible_price``: the bounds cross, no allowed price point fits inside them, or
      every candidate breaks the inventory limit. The current price is kept.
    * ``uncertain``: ``0.2 <= P(elasticity < -1) <= 0.8``, so the belief cannot say
      whether demand is elastic. The current price is kept unless some candidate
      improves the risk-adjusted objective and raises profit with probability at least
      0.8; the best such candidate is then recommended. If the current price is itself
      infeasible, the nearest feasible price stands in for it.
    * ``ok``: otherwise.

    ``confidence`` is required (keyword-only, no default) because spec Rule 12 requires a
    confidence score on every recommendation; it must lie in ``[0, 1]`` and is passed
    through unchanged. Leaving it out, or passing None, raises ``TypeError``.
    """
    model = as_belief(belief)
    _check_inputs(
        current_price,
        base_quantity,
        unit_cost,
        constraints,
        risk_aversion,
        risk_measure,
        fixed_cost,
        confidence,
        n_draws,
        step,
        cvar_alpha,
    )

    def simulate(cost: float | None, grid: FloatArray | None) -> SimulationResult:
        return simulate_prices(
            current_price,
            base_quantity,
            model,
            unit_cost=cost,
            fixed_cost=fixed_cost,
            grid=grid,
            rel_range=(-constraints.max_rel_change, constraints.max_rel_change),
            step=step,
            n_draws=n_draws,
            rng=rng,
            cvar_alpha=cvar_alpha,
        )

    if unit_cost is None:
        bounds = price_bounds(current_price, None, constraints)
        prices = candidate_prices(current_price, bounds, constraints, step=step)
        simulation = simulate(None, prices if prices.size else None)
        reason = "unit_cost is required: revenue is never optimised without profit (spec Rule 9)"
        return _hold(
            current_price,
            model,
            confidence,
            "needs_cost",
            (),
            {"reason": reason, "simulation": simulation},
        )

    bounds = price_bounds(current_price, unit_cost, constraints)
    candidates = candidate_prices(current_price, bounds, constraints, step=step)
    if candidates.size == 0:
        binding: tuple[str, ...]
        if bounds.feasible:
            binding = (constraints.snapping,) if constraints.snapping else ()
            reason = "no allowed price point lies within the bounds"
        else:
            binding, reason = bounds.conflicting(), "the lower price limit exceeds the upper limit"
        diagnostics = {"reason": reason, "bounds": (bounds.lower, bounds.upper)}
        return _hold(current_price, model, confidence, "no_feasible_price", binding, diagnostics)

    probes = probe_prices(current_price, bounds, constraints, step=step)
    simulation = simulate(unit_cost, np.concatenate([candidates, probes]))
    return _recommend(
        simulation, candidates, bounds, constraints, model, risk_aversion, risk_measure, confidence
    )


def _recommend(
    sim: SimulationResult,
    candidates: FloatArray,
    bounds: PriceBounds,
    constraints: PriceConstraints,
    belief: ElasticityBelief,
    risk_aversion: float,
    risk_measure: RiskMeasure,
    confidence: float,
) -> Recommendation:
    assert sim.profit is not None
    prices, p0 = sim.prices, sim.current_price
    risk_values = sim.risk.shortfall if risk_measure == "cvar" else sim.risk.std
    objective = sim.profit.mean - risk_aversion * risk_values
    in_stock = within_inventory(sim.quantity.mean, constraints)
    feasible = np.isin(prices, candidates) & in_stock
    if not feasible.any():
        reason = "every candidate price sells more than max_units"
        return _hold(
            p0,
            belief,
            confidence,
            "no_feasible_price",
            ("inventory",),
            {"reason": reason, "simulation": sim},
        )

    prob_elastic = belief.prob_below(-1.0)
    uncertain = UNCERTAIN_PROB_ELASTIC[0] <= prob_elastic <= UNCERTAIN_PROB_ELASTIC[1]
    unrestricted = _argmax_nearest(objective, feasible, prices, p0)
    eligible = np.ones(prices.size, dtype=bool)
    if uncertain:
        fallback = (
            sim.current_index if feasible[sim.current_index] else _nearest(feasible, prices, p0)
        )
        to_beat = objective[fallback] + _tolerance(objective[fallback])
        eligible = (sim.prob_improves >= UNCERTAIN_MIN_PROB_IMPROVES) & (objective > to_beat)
        eligible[fallback] = True
    chosen = _argmax_nearest(objective, feasible & eligible, prices, p0)

    exclusions = [
        _exclusions(float(p), bool(ok), bounds) for p, ok in zip(prices, in_stock, strict=True)
    ]
    beats_chosen = objective > objective[chosen] + _tolerance(objective[chosen])
    blocked = np.flatnonzero(~feasible & eligible & beats_chosen)
    binding = order_constraints(name for i in blocked for name in exclusions[i])

    frame = sim.to_frame()
    frame["risk"] = risk_values
    frame["objective"] = objective
    frame["feasible"] = feasible
    frame["eligible"] = eligible
    frame["excluded_by"] = [",".join(names) for names in exclusions]
    profit0 = sim.profit.baseline
    diagnostics: dict[str, Any] = {
        "prob_elastic": prob_elastic,
        "uncertain_band": UNCERTAIN_PROB_ELASTIC,
        "uncertain_min_prob_improves": UNCERTAIN_MIN_PROB_IMPROVES,
        "risk_measure": risk_measure,
        "risk_aversion": risk_aversion,
        "cvar_alpha": sim.risk.alpha,
        "bounds": (bounds.lower, bounds.upper),
        "current_profit": profit0,
        "expected_profit": float(sim.profit.mean[chosen]),
        "objective": float(objective[chosen]),
        "risk_value": float(risk_values[chosen]),
        "unrestricted_price": float(prices[unrestricted]),
        "evaluated": frame,
        "simulation": sim,
    }
    return Recommendation(
        current_price=p0,
        recommended_price=float(prices[chosen]),
        elasticity=belief.mean,
        elasticity_interval_95=belief.interval(0.95),
        expected_quantity_change_pct=float(sim.quantity.change_pct[chosen]),
        expected_revenue_change_pct=float(sim.revenue.change_pct[chosen]),
        expected_profit_change_pct=float(sim.profit.change_pct[chosen]),
        probability_profit_improves=float(sim.prob_improves[chosen]),
        risk=float(risk_values[chosen]) / abs(profit0) if profit0 != 0.0 else math.nan,
        binding_constraints=binding,
        status="uncertain" if uncertain else "ok",
        confidence=float(confidence),
        diagnostics=diagnostics,
    )


@dataclass(frozen=True)
class ExperimentPlan:
    """Expected precision after running the price test for ``test_days`` days."""

    test_days: int
    posterior_sd: float
    interval_width: float
    width_reduction_pct: float


@dataclass(frozen=True)
class ExperimentSuggestion:
    """Whether to run a controlled price test, and what it would buy (spec §77-78)."""

    recommended: bool
    reason: ExperimentReason
    current_price: float
    test_price: float | None
    test_change_pct: float | None
    interval_95: tuple[float, float]
    interval_width: float
    width_threshold: float
    noise_sd: float
    plans: tuple[ExperimentPlan, ...]
    days_to_threshold: int | None
    message: str


def _test_price(
    current_price: float, bounds: PriceBounds, constraints: PriceConstraints, change: float
) -> float | None:
    """Feasible price near ``current * (1 + change)``, strictly on the side ``change`` points to."""
    if not bounds.feasible:
        return None
    target = float(stepped_prices(current_price, [1], change)[0])
    target = min(max(target, bounds.lower), bounds.upper)
    lattice = constraints.lattice
    if lattice is not None:
        k_lo, k_hi = lattice.index_range(bounds.lower, bounds.upper)
        if k_lo > k_hi:
            return None
        target = float(lattice.snap([target], bounds.lower, bounds.upper)[0])
        if (target - current_price) * change <= 0.0:
            beyond = (
                lattice.next_above(current_price)
                if change > 0.0
                else lattice.next_below(current_price)
            )
            if beyond is None:
                return None
            target = beyond
    if not bounds.contains(target) or (target - current_price) * change <= 0.0:
        return None
    return target


def _plan(days: int, prior_sd: float, info_per_day: float, width: float) -> ExperimentPlan:
    posterior_sd = 1.0 / math.sqrt(prior_sd**-2 + days * info_per_day)
    ratio = posterior_sd / prior_sd
    return ExperimentPlan(days, posterior_sd, width * ratio, 100.0 * (1.0 - ratio))


def experiment_suggestion(
    belief: ElasticityBelief | ElasticityEstimate,
    current_price: float,
    constraints: PriceConstraints,
    *,
    daily_units: float,
    test_days: tuple[int, ...] = (14, 28),
    unit_cost: float | None = None,
    test_change: float = 0.05,
    width_threshold: float = DEFAULT_WIDTH_THRESHOLD,
    noise_sd: float | None = None,
    test_share: float = 0.5,
) -> ExperimentSuggestion:
    """Propose a controlled price test when the 95% elasticity interval is wider than
    ``width_threshold`` (spec §78), and estimate how much each test length narrows it.

    The test price is ``current * (1 + test_change)`` (default +5%), clipped to the
    constraint bounds and snapped to allowed price points; if no increase is feasible the
    same decrease is tried. Pass ``unit_cost`` so that a decrease respects the margin
    floor.

    Precision gain uses a Fisher-information approximation. Assumptions:

    * Daily demand follows ``ln Q_t = a + elasticity * ln P_t + u_t`` with independent
      normal noise of known standard deviation ``sigma`` and constant elasticity during
      the test. Unless ``noise_sd`` is given, ``sigma**2 = 1 / daily_units + 0.2**2``:
      Poisson counting noise (delta method, ``Var(ln N) ~ 1 / E[N]``) plus 20% day-level
      demand noise. ``daily_units`` is taken to hold at both prices.
    * Switchback design: a random share ``w = test_share`` of the ``T`` test days use the
      test price and the rest the current price, so randomisation removes confounding.
      With the intercept re-estimated, the information about the elasticity is
      ``T * w * (1 - w) * ln(P_test / P_current)**2 / sigma**2``.
    * The current belief acts as an independent normal prior with variance ``sd**2``, so
      precisions add; the interval narrows in proportion to the posterior standard
      deviation.
    """
    model = as_belief(belief)
    _require(_is_positive(current_price), f"current_price must be positive, got {current_price!r}")
    _require(_is_positive(daily_units), f"daily_units must be positive, got {daily_units!r}")
    days = tuple(operator.index(d) for d in test_days)
    _require(
        len(days) > 0 and min(days) >= 1,
        f"test_days must hold positive day counts, got {test_days!r}",
    )
    _require(0.0 < test_change < 1.0, f"test_change must lie in (0, 1), got {test_change!r}")
    _require(
        _is_positive(width_threshold), f"width_threshold must be positive, got {width_threshold!r}"
    )
    _require(
        noise_sd is None or _is_positive(noise_sd), f"noise_sd must be positive, got {noise_sd!r}"
    )
    _require(0.0 < test_share < 1.0, f"test_share must lie in (0, 1), got {test_share!r}")

    lower, upper = model.interval(0.95)
    width = upper - lower
    sigma = (
        noise_sd if noise_sd is not None else math.sqrt(1.0 / daily_units + DEFAULT_DAY_NOISE_SD**2)
    )

    def suggestion(
        reason: ExperimentReason,
        message: str,
        test_price: float | None = None,
        plans: tuple[ExperimentPlan, ...] = (),
        days_needed: int | None = None,
    ) -> ExperimentSuggestion:
        change = (
            None if test_price is None else 100.0 * (test_price - current_price) / current_price
        )
        return ExperimentSuggestion(
            recommended=test_price is not None,
            reason=reason,
            current_price=current_price,
            test_price=test_price,
            test_change_pct=change,
            interval_95=(lower, upper),
            interval_width=width,
            width_threshold=width_threshold,
            noise_sd=sigma,
            plans=plans,
            days_to_threshold=days_needed,
            message=message,
        )

    if width <= width_threshold or model.sd == 0.0:
        return suggestion(
            "interval_narrow",
            f"The 95% elasticity interval is {width:.2f} wide, within the {width_threshold:.2f} "
            "threshold, so no experiment is needed.",
        )
    bounds = price_bounds(current_price, unit_cost, constraints)
    test_price = _test_price(current_price, bounds, constraints, test_change)
    if test_price is None:
        test_price = _test_price(current_price, bounds, constraints, -test_change)
    if test_price is None:
        return suggestion(
            "no_feasible_test_price",
            "The elasticity is uncertain but the price constraints leave no test price; "
            "relax max_rel_change or the business price limits to run an experiment.",
        )

    log_step = math.log(test_price / current_price)
    info_per_day = test_share * (1.0 - test_share) * log_step**2 / sigma**2
    plans = tuple(_plan(d, model.sd, info_per_day, width) for d in days)
    target_sd = model.sd * width_threshold / width
    days_needed = max(1, math.ceil((target_sd**-2 - model.sd**-2) / info_per_day))
    longest = max(plans, key=lambda plan: plan.test_days)
    change_pct = 100.0 * (test_price - current_price) / current_price
    message = (
        f"The 95% elasticity interval [{lower:.2f}, {upper:.2f}] is wide. A controlled "
        f"{change_pct:+.1f}% price test ({current_price:g} -> {test_price:g}) on a random "
        f"{test_share:.0%} of days for {longest.test_days} days would narrow it to about "
        f"{longest.interval_width:.2f} ({longest.width_reduction_pct:.0f}% narrower); about "
        f"{days_needed} days would bring it under {width_threshold:.2f}."
    )
    return suggestion("interval_wide", message, test_price, plans, days_needed)
