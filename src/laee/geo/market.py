"""Industry micro-market radii, distance decay and market-radius selection (spec §5-7, §13).

Markets are defined around each business by distance, never by ZIP or postcode (spec Rule 7).
The radii below are priors (spec §6: "Do not hard-code these as truth"). `select_market_radius`
lets observed predictive accuracy choose among them.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable

import numpy as np
from numpy.typing import ArrayLike

from .distance import DensityClass, FloatArray, validate_density_class

INDUSTRY_IDS: tuple[str, ...] = (
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
)

# Candidate market radii in km. Entries without a "provisional" note are spec §6 verbatim (its
# "coffee" prior is stored under "cafe"). The provisional entries are our assumptions, chosen by
# analogy with the spec's examples:
#   apparel      comparison goods; shoppers compare across shops and travel to market districts
#                (Gikomba, Balogun, Makola), so the reach is wider than for convenience goods.
#   electronics  specialist goods with city-wide clusters (Luthuli Avenue, Computer Village).
#   pharmacy     convenience necessity bought close to home; dense chemist networks.
#   wholesale    serves resellers across a metro region.
#   other_service, other  no specific evidence, so they take the spec §5.1 Layer A generic grid.
# ecommerce has no geographic market: the empty tuple means "no radius".
RADIUS_PRIORS_KM: dict[str, tuple[float, ...]] = {
    "grocery": (1.0, 3.0, 5.0, 10.0),
    "cafe": (0.25, 0.5, 1.0, 2.0),
    "restaurant": (0.5, 1.0, 2.0, 3.0),
    "apparel": (1.0, 2.0, 5.0, 10.0),
    "electronics": (2.0, 5.0, 10.0, 20.0),
    "pharmacy": (0.5, 1.0, 2.0, 3.0),
    "hair_salon": (1.0, 2.0, 5.0),
    "auto_repair": (3.0, 5.0, 10.0, 20.0),
    "professional_service": (3.0, 5.0, 10.0),
    "other_service": (0.25, 0.5, 1.0, 2.0, 3.0, 5.0),
    "wholesale": (5.0, 10.0, 20.0, 50.0),
    "ecommerce": (),
    "other": (0.25, 0.5, 1.0, 2.0, 3.0, 5.0),
}

PROVISIONAL: frozenset[str] = frozenset(
    {"apparel", "electronics", "pharmacy", "other_service", "wholesale", "other"}
)

# The spec's radii are read as urban values. If outlets form a Poisson field of density rho, the
# expected distance to the nearest one is 1 / (2 sqrt(rho)), so the market scale grows like
# rho^(-1/2). Assuming suburban outlet density is about half the urban level and rural about a
# ninth gives the multipliers sqrt(2) ~ 1.5 and sqrt(9) = 3. These density ratios are assumptions.
DENSITY_RADIUS_MULTIPLIERS: dict[DensityClass, float] = {
    "urban": 1.0,
    "suburban": 1.5,
    "rural": 3.0,
}


def validate_industry_id(industry_id: str) -> str:
    """Return `industry_id` if it is one of `INDUSTRY_IDS`, else raise ValueError."""
    if industry_id not in RADIUS_PRIORS_KM:
        raise ValueError(f"unknown industry id {industry_id!r}; expected one of {INDUSTRY_IDS}")
    return industry_id


def is_geographic(industry_id: str) -> bool:
    """False for industries without a physical market (ecommerce)."""
    return bool(RADIUS_PRIORS_KM[validate_industry_id(industry_id)])


def density_adjusted_radii(industry_id: str, density_class: str) -> tuple[float, ...]:
    """Prior radii (km) widened for lower-density areas; empty for non-geographic industries."""
    multiplier = DENSITY_RADIUS_MULTIPLIERS[validate_density_class(density_class)]
    return tuple(r * multiplier for r in RADIUS_PRIORS_KM[validate_industry_id(industry_id)])


def industry_median_radius_km(industry_id: str, density_class: str = "urban") -> float | None:
    """Median of the density-adjusted priors, or None when the industry has no radius."""
    radii = density_adjusted_radii(industry_id, density_class)
    return float(np.median(radii)) if radii else None


def decay_lambda(median_radius_km: float) -> float:
    """Decay rate ln 2 / r, so that exp(-lambda d) is exactly 0.5 at d = r (spec §7)."""
    if not (math.isfinite(median_radius_km) and median_radius_km > 0):
        raise ValueError("median radius must be a positive finite number of km")
    return math.log(2.0) / median_radius_km


def industry_decay_lambda(industry_id: str, density_class: str = "urban") -> float:
    """Default distance-decay rate per km; 0.0 (no decay) for non-geographic industries."""
    median = industry_median_radius_km(industry_id, density_class)
    return 0.0 if median is None else decay_lambda(median)


def distance_weights(d_km: ArrayLike, lam: float) -> FloatArray:
    """Distance-decay weights exp(-lambda d) (spec §7); infinite distance gives weight 0."""
    d = np.asarray(d_km, dtype=np.float64)
    if np.any(np.isnan(d)) or np.any(d < 0):
        raise ValueError("distances must be non-negative (inf is allowed for unreachable)")
    if not (math.isfinite(lam) and lam >= 0):
        raise ValueError("lambda must be a non-negative finite number")
    if lam == 0.0:
        # Avoid exp(-0 * inf) = NaN: no decay means unit weight at any distance.
        return np.ones_like(d)
    return np.asarray(np.exp(-lam * d), dtype=np.float64)


def select_market_radius(
    candidate_radii: Iterable[float],
    score_fn: Callable[[float], float],
    *,
    tie_tolerance: float = 0.0,
) -> tuple[float, dict[float, float]]:
    """MarketRadius* = argmax_R PredictiveAccuracy(R) (spec §13).

    `score_fn(radius_km)` returns the predictive accuracy of a model whose market uses that
    radius; higher is better and NaN marks a radius that could not be evaluated. The winner is the
    smallest radius scoring within `tie_tolerance` of the best score. With the default of 0 only
    exact ties go to the smaller radius; pass e.g. one standard error to prefer the smallest
    market that is not meaningfully worse. Returns the winner and every radius's score, in
    ascending order of radius.
    """
    radii = sorted({float(r) for r in candidate_radii})
    if not radii:
        raise ValueError(
            "no candidate radii: industries without a geographic market (ecommerce) have none"
        )
    if any(not (math.isfinite(r) and r > 0) for r in radii):
        raise ValueError("candidate radii must be positive finite numbers of km")
    if not (math.isfinite(tie_tolerance) and tie_tolerance >= 0):
        raise ValueError("tie_tolerance must be a non-negative finite number")
    scores = {r: float(score_fn(r)) for r in radii}
    valid = {r: s for r, s in scores.items() if not math.isnan(s)}
    if not valid:
        raise ValueError("score_fn returned NaN for every candidate radius")
    best = max(valid.values())
    return min(r for r, s in valid.items() if s >= best - tie_tolerance), scores
