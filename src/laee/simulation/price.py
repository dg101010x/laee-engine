"""Counterfactual price simulator (spec §34-36, Phase 8).

An uncertain own-price elasticity is represented by an :class:`ElasticityBelief`.
:func:`simulate_prices` draws elasticities from it once and evaluates every candidate
price under the same draws, so differences between prices are not Monte Carlo noise.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from laee.simulation.risk import DEFAULT_ALPHA, RiskProfile, risk_profile
from laee.types import ElasticityEstimate, Interval

FloatArray = NDArray[np.float64]
BeliefFamily = Literal["normal", "student_t", "empirical"]
ImprovementMetric = Literal["profit", "revenue"]

DEFAULT_REL_RANGE: tuple[float, float] = (-0.20, 0.20)
DEFAULT_STEP = 0.005
DEFAULT_N_DRAWS = 4000

__all__ = [
    "DEFAULT_N_DRAWS",
    "DEFAULT_REL_RANGE",
    "DEFAULT_STEP",
    "ElasticityBelief",
    "OutcomeSummary",
    "SimulationResult",
    "as_belief",
    "price_grid",
    "simulate_prices",
    "stepped_prices",
]


def _finite(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return number


def _positive(name: str, value: float) -> float:
    number = _finite(name, value)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return number


def _non_negative(name: str, value: float) -> float:
    number = _finite(name, value)
    if number < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")
    return number


def _count(name: str, value: int) -> int:
    number = operator.index(value)
    if number < 1:
        raise ValueError(f"{name} must be at least 1, got {value!r}")
    return number


@dataclass(frozen=True, eq=False)
class ElasticityBelief:
    """Uncertainty about a constant own-price elasticity.

    Build it with :meth:`normal`, :meth:`student_t`, :meth:`from_estimate` or
    :meth:`from_samples`. ``loc`` is the mean. ``scale`` is the standard deviation
    (normal), the scale parameter (Student t) or the sample standard deviation
    (empirical). A zero scale is a point mass, which is useful for what-if analysis.
    """

    family: BeliefFamily
    loc: float
    scale: float
    df: float | None = None
    samples: FloatArray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.family not in ("normal", "student_t", "empirical"):
            raise ValueError(f"unknown belief family {self.family!r}")
        _finite("loc", self.loc)
        _non_negative("scale", self.scale)
        if self.family == "student_t" and (self.df is None or not self.df > 2.0):
            raise ValueError("a Student t belief needs df > 2 so that its variance is finite")
        if self.family == "empirical" and self.samples is None:
            raise ValueError("an empirical belief needs samples")

    @classmethod
    def normal(cls, mean: float, sd: float) -> ElasticityBelief:
        """Normal belief with the given mean and standard deviation."""
        return cls("normal", _finite("mean", mean), _non_negative("sd", sd))

    @classmethod
    def student_t(cls, mean: float, scale: float, df: float) -> ElasticityBelief:
        """Location-scale Student t belief; heavier tails than the normal for small samples."""
        return cls(
            "student_t", _finite("mean", mean), _non_negative("scale", scale), _finite("df", df)
        )

    @classmethod
    def from_samples(cls, samples: ArrayLike) -> ElasticityBelief:
        """Empirical belief from posterior or bootstrap draws; :meth:`draw` resamples them."""
        values = np.array(samples, dtype=np.float64)
        if values.ndim != 1 or values.size < 2:
            raise ValueError("samples must be a 1-D array with at least two draws")
        if not np.all(np.isfinite(values)):
            raise ValueError("samples must be finite")
        values.setflags(write=False)
        return cls("empirical", float(values.mean()), float(values.std(ddof=1)), None, values)

    @classmethod
    def from_estimate(
        cls, estimate: ElasticityEstimate, *, df: float | None = None
    ) -> ElasticityBelief:
        """Belief centred on ``estimate.value``: normal, or Student t when ``df`` is given.

        The spread is the larger of ``estimate.std_error`` and the spread implied by each
        reported interval, so an interval that carries extra uncertainty (for example
        model disagreement, spec §39-40) is never understated. An estimate without any
        usable uncertainty is rejected (spec §91 Rule 4).
        """
        value = _finite("estimate.value", estimate.value)
        spreads = [_spread_from_interval(iv, df) for iv in estimate.intervals]
        if math.isfinite(estimate.std_error) and estimate.std_error > 0.0:
            spreads.append(float(estimate.std_error))
        usable = [s for s in spreads if s is not None and s > 0.0]
        if not usable:
            raise ValueError(
                "estimate carries no uncertainty (std_error or intervals); see spec Rule 4"
            )
        if df is None:
            return cls.normal(value, max(usable))
        return cls.student_t(value, max(usable), df)

    @property
    def mean(self) -> float:
        return self.loc

    @property
    def sd(self) -> float:
        if self.family == "student_t":
            assert self.df is not None
            return self.scale * math.sqrt(self.df / (self.df - 2.0))
        return self.scale

    def draw(self, rng: np.random.Generator, n: int) -> FloatArray:
        """Draw ``n`` elasticities."""
        n = _count("n", n)
        if self.family == "normal":
            return rng.normal(self.loc, self.scale, size=n)
        if self.family == "student_t":
            assert self.df is not None
            return self.loc + self.scale * rng.standard_t(self.df, size=n)
        assert self.samples is not None
        return rng.choice(self.samples, size=n, replace=True)

    def interval(self, level: float = 0.95) -> tuple[float, float]:
        """Central credible interval at ``level``."""
        if not 0.0 < level < 1.0:
            raise ValueError(f"level must lie in (0, 1), got {level!r}")
        if self.family == "empirical":
            assert self.samples is not None
            lo, hi = np.quantile(self.samples, [(1.0 - level) / 2.0, (1.0 + level) / 2.0])
            return float(lo), float(hi)
        half = self.scale * self._standard_ppf((1.0 + level) / 2.0)
        return self.loc - half, self.loc + half

    def prob_below(self, threshold: float) -> float:
        """Probability that the elasticity is below ``threshold``."""
        if self.family == "empirical":
            assert self.samples is not None
            return float(np.mean(self.samples < threshold))
        if self.scale == 0.0:
            return float(self.loc < threshold)
        z = (threshold - self.loc) / self.scale
        if self.family == "student_t":
            return float(stats.t.cdf(z, self.df))
        return float(stats.norm.cdf(z))

    def _standard_ppf(self, q: float) -> float:
        if self.family == "student_t":
            return float(stats.t.ppf(q, self.df))
        return float(stats.norm.ppf(q))


def _spread_from_interval(interval: Interval, df: float | None) -> float | None:
    if not (0.0 < interval.level < 1.0) or not interval.upper > interval.lower:
        return None
    q = (1.0 + interval.level) / 2.0
    z = float(stats.norm.ppf(q)) if df is None else float(stats.t.ppf(q, df))
    return (interval.upper - interval.lower) / (2.0 * z)


def as_belief(belief: ElasticityBelief | ElasticityEstimate) -> ElasticityBelief:
    """Accept either a belief or a shared :class:`~laee.types.ElasticityEstimate`."""
    if isinstance(belief, ElasticityBelief):
        return belief
    if isinstance(belief, ElasticityEstimate):
        return ElasticityBelief.from_estimate(belief)
    raise TypeError(f"expected ElasticityBelief or ElasticityEstimate, got {type(belief).__name__}")


def _decimal(value: float) -> Decimal:
    return Decimal(repr(float(value)))


def stepped_prices(base_price: float, k: ArrayLike, step: float) -> FloatArray:
    """``base_price * (1 + k * step)`` for integer ``k``, computed in decimal arithmetic.

    Decimal arithmetic keeps round prices round: ``10.00 * (1 + 30 * 0.005)`` is exactly
    ``11.5`` rather than ``11.499999999999998``.
    """
    base = _decimal(_non_negative("base_price", base_price))
    size = _decimal(_finite("step", step))
    ks = np.atleast_1d(np.asarray(k))
    if ks.size and not np.issubdtype(ks.dtype, np.integer):
        raise TypeError("k must hold integers")
    return np.array([float(base * (1 + int(i) * size)) for i in ks.ravel()], dtype=np.float64)


def price_grid(
    current_price: float,
    rel_range: tuple[float, float] = DEFAULT_REL_RANGE,
    step: float = DEFAULT_STEP,
) -> FloatArray:
    """Sorted candidate prices ``current * (1 + k * step)`` covering ``rel_range`` (spec §37).

    ``step`` is relative to the current price, and the grid always contains the
    current price exactly, even when ``rel_range`` excludes zero.
    """
    p0 = _positive("current_price", current_price)
    lo, hi = _finite("rel_range[0]", rel_range[0]), _finite("rel_range[1]", rel_range[1])
    step = _positive("step", step)
    if not -1.0 < lo <= hi:
        raise ValueError(f"rel_range must satisfy -1 < low <= high, got {rel_range!r}")
    ks = np.arange(math.ceil(lo / step - 1e-9), math.floor(hi / step + 1e-9) + 1)
    grid = stepped_prices(p0, ks, step)
    return np.unique(np.append(grid[grid > 0.0], p0))


def _custom_grid(grid: ArrayLike, current_price: float) -> FloatArray:
    prices = np.asarray(grid, dtype=np.float64)
    if prices.ndim != 1:
        raise ValueError("grid must be one-dimensional")
    if not np.all(np.isfinite(prices) & (prices > 0.0)):
        raise ValueError("grid prices must be finite and positive")
    return np.unique(np.append(prices, current_price))


@dataclass(frozen=True, eq=False)
class OutcomeSummary:
    """Per-price summary of one simulated outcome (quantity, revenue or profit).

    ``baseline`` is the outcome at the current price, which is certain because
    ``(P0 / P0) ** elasticity == 1``. ``change_pct`` is ``100 * (mean - baseline) /
    |baseline|`` and is NaN when the baseline is zero.
    """

    baseline: float
    mean: FloatArray = field(repr=False)
    p05: FloatArray = field(repr=False)
    p50: FloatArray = field(repr=False)
    p95: FloatArray = field(repr=False)
    change_pct: FloatArray = field(repr=False)


def _summarise(draws: FloatArray, baseline: float) -> OutcomeSummary:
    mean = draws.mean(axis=1)
    p05, p50, p95 = np.percentile(draws, [5.0, 50.0, 95.0], axis=1)
    if baseline == 0.0:
        change = np.full_like(mean, np.nan)
    else:
        change = 100.0 * (mean - baseline) / abs(baseline)
    return OutcomeSummary(baseline, mean, p05, p50, p95, change)


@dataclass(frozen=True, eq=False)
class SimulationResult:
    """Simulated outcomes for every candidate price.

    Draw matrices have shape ``(n_prices, n_draws)`` and share the same elasticity
    draws across prices. ``improvement_metric`` is ``"profit"`` when the unit cost is
    known and ``"revenue"`` otherwise; ``prob_improves`` and ``risk`` refer to the
    change in that metric relative to the current price.
    """

    prices: FloatArray
    current_index: int
    current_price: float
    base_quantity: float
    unit_cost: float | None
    fixed_cost: float
    elasticity_draws: FloatArray = field(repr=False)
    quantity_draws: FloatArray = field(repr=False)
    revenue_draws: FloatArray = field(repr=False)
    profit_draws: FloatArray | None = field(repr=False)
    quantity: OutcomeSummary = field(repr=False)
    revenue: OutcomeSummary = field(repr=False)
    profit: OutcomeSummary | None = field(repr=False)
    improvement_metric: ImprovementMetric
    prob_improves: FloatArray = field(repr=False)
    risk: RiskProfile = field(repr=False)

    def to_frame(self) -> pd.DataFrame:
        """One row per candidate price.

        Columns: ``price``, ``price_change_pct`` and ``is_current``; for each of quantity,
        revenue and (only when the cost is known) profit, ``expected_<x>``, ``<x>_p05``,
        ``<x>_p50``, ``<x>_p95`` and ``<x>_change_pct``; and for the improvement metric
        ``<m>``, ``prob_<m>_improves``, ``prob_<m>_declines``, ``<m>_change_sd``,
        ``<m>_change_cvar`` and ``<m>_shortfall``.
        """
        columns: dict[str, ArrayLike] = {
            "price": self.prices,
            "price_change_pct": 100.0 * (self.prices - self.current_price) / self.current_price,
            "is_current": np.arange(self.prices.size) == self.current_index,
        }
        outcomes = {"quantity": self.quantity, "revenue": self.revenue}
        if self.profit is not None:
            outcomes["profit"] = self.profit
        for name, summary in outcomes.items():
            columns[f"expected_{name}"] = summary.mean
            columns[f"{name}_p05"] = summary.p05
            columns[f"{name}_p50"] = summary.p50
            columns[f"{name}_p95"] = summary.p95
            columns[f"{name}_change_pct"] = summary.change_pct
        metric = self.improvement_metric
        columns[f"prob_{metric}_improves"] = self.prob_improves
        columns[f"prob_{metric}_declines"] = self.risk.downside_prob
        columns[f"{metric}_change_sd"] = self.risk.std
        columns[f"{metric}_change_cvar"] = self.risk.cvar
        columns[f"{metric}_shortfall"] = self.risk.shortfall
        return pd.DataFrame(columns)


def simulate_prices(
    current_price: float,
    base_quantity: float,
    belief: ElasticityBelief | ElasticityEstimate,
    *,
    unit_cost: float | None = None,
    fixed_cost: float = 0.0,
    grid: ArrayLike | None = None,
    rel_range: tuple[float, float] = DEFAULT_REL_RANGE,
    step: float = DEFAULT_STEP,
    n_draws: int = DEFAULT_N_DRAWS,
    rng: np.random.Generator,
    cvar_alpha: float = DEFAULT_ALPHA,
) -> SimulationResult:
    """Simulate quantity, revenue and (with a unit cost) gross profit at candidate prices.

    ``base_quantity`` is the quantity sold per period at ``current_price``. Candidate
    prices come from ``grid`` when given, otherwise from :func:`price_grid` with
    ``rel_range`` and ``step``; the current price is always included. ``fixed_cost``
    is the per-period fixed cost ``F`` of spec §36 and only matters with a unit cost.
    Risk measures use ``cvar_alpha`` as the CVaR tail share.
    """
    model = as_belief(belief)
    p0 = _positive("current_price", current_price)
    q0 = _positive("base_quantity", base_quantity)
    fixed = _non_negative("fixed_cost", fixed_cost)
    n = _count("n_draws", n_draws)
    prices = price_grid(p0, rel_range, step) if grid is None else _custom_grid(grid, p0)
    current_index = int(np.searchsorted(prices, p0))

    elasticities = model.draw(rng, n)
    # Spec §34: Q1 = Q0 * (P1 / P0) ** elasticity for every (price, draw) pair, written as
    # exp(elasticity * ln(P1 / P0)) so the current-price row is exactly Q0 for every draw.
    quantity = q0 * np.exp(np.outer(np.log(prices / p0), elasticities))
    revenue = prices[:, None] * quantity
    revenue0 = p0 * q0

    cost: float | None = None
    profit: FloatArray | None = None
    profit_summary: OutcomeSummary | None = None
    if unit_cost is None:
        metric: ImprovementMetric = "revenue"
        metric_draws, metric0 = revenue, revenue0
    else:
        cost = _non_negative("unit_cost", unit_cost)
        # Spec §36: Profit = (P - C) * Q - F.
        profit = (prices - cost)[:, None] * quantity - fixed
        profit0 = (p0 - cost) * q0 - fixed
        profit_summary = _summarise(profit, profit0)
        metric = "profit"
        metric_draws, metric0 = profit, profit0

    return SimulationResult(
        prices=prices,
        current_index=current_index,
        current_price=p0,
        base_quantity=q0,
        unit_cost=cost,
        fixed_cost=fixed,
        elasticity_draws=elasticities,
        quantity_draws=quantity,
        revenue_draws=revenue,
        profit_draws=profit,
        quantity=_summarise(quantity, q0),
        revenue=_summarise(revenue, revenue0),
        profit=profit_summary,
        improvement_metric=metric,
        prob_improves=np.mean(metric_draws > metric0, axis=1),
        risk=risk_profile(metric_draws - metric0, cvar_alpha),
    )
