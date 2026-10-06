"""Dispersion and downside measures over simulated outcomes (spec §38 and §89).

Every function reduces the last axis, which holds Monte Carlo draws, so a
``(n_prices, n_draws)`` matrix gives one value per candidate price and a 1-D array
gives a 0-d array.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

DEFAULT_ALPHA = 0.10

__all__ = [
    "DEFAULT_ALPHA",
    "RiskProfile",
    "cvar",
    "cvar_shortfall",
    "downside_probability",
    "risk_profile",
    "standard_deviation",
    "variance",
]


def _as_draws(outcomes: ArrayLike) -> FloatArray:
    draws = np.asarray(outcomes, dtype=np.float64)
    if draws.ndim == 0 or draws.shape[-1] == 0:
        raise ValueError("outcomes need at least one draw along the last axis")
    return draws


def _values(result: ArrayLike) -> FloatArray:
    return np.asarray(result, dtype=np.float64)


def variance(outcomes: ArrayLike) -> FloatArray:
    """Variance of each row of draws (the variance of the simulated distribution)."""
    return _values(np.var(_as_draws(outcomes), axis=-1))


def standard_deviation(outcomes: ArrayLike) -> FloatArray:
    """Standard deviation of each row of draws."""
    return _values(np.std(_as_draws(outcomes), axis=-1))


def cvar(outcomes: ArrayLike, alpha: float = DEFAULT_ALPHA) -> FloatArray:
    """Lower-tail conditional value at risk: the mean of the worst ``alpha`` share of draws.

    Outcomes are "higher is better" (for example a change in profit), so the worst
    draws are the smallest ones. The result is exact for the empirical distribution
    of the draws, including when ``alpha * n_draws`` is not a whole number.
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must lie in (0, 1], got {alpha!r}")
    draws = _as_draws(outcomes)
    n = draws.shape[-1]
    tail = alpha * n
    if round(tail) >= 1 and abs(tail - round(tail)) < 1e-9:
        tail = float(round(tail))
    k = int(np.floor(tail))
    if k >= n:
        return _values(draws.mean(axis=-1))
    # Acerbi-Tasche estimator: the k worst draws enter with full weight and the
    # (k+1)-th worst with weight alpha*n - k, i.e. (1/alpha) * integral_0^alpha q(u) du
    # for the empirical quantile function q.
    part = np.partition(draws, k, axis=-1)
    total = part[..., :k].sum(axis=-1) + (tail - k) * part[..., k]
    return _values(total / tail)


def cvar_shortfall(outcomes: ArrayLike, alpha: float = DEFAULT_ALPHA) -> FloatArray:
    """Expected loss in the worst ``alpha`` share of draws, ``max(0, -CVaR_alpha)``.

    Applied to changes relative to the status quo, this is how much is lost on
    average in a bad case; it is zero when even the worst draws are gains.
    """
    return _shortfall(cvar(outcomes, alpha))


def _shortfall(tail_mean: FloatArray) -> FloatArray:
    return np.where(tail_mean < 0.0, -tail_mean, 0.0)


def downside_probability(outcomes: ArrayLike, threshold: float = 0.0) -> FloatArray:
    """Share of draws strictly below ``threshold``."""
    return _values(np.mean(_as_draws(outcomes) < threshold, axis=-1))


@dataclass(frozen=True, eq=False)
class RiskProfile:
    """Risk of the change in an outcome relative to the current price, one value per price."""

    alpha: float
    variance: FloatArray = field(repr=False)
    std: FloatArray = field(repr=False)
    cvar: FloatArray = field(repr=False)
    shortfall: FloatArray = field(repr=False)
    downside_prob: FloatArray = field(repr=False)


def risk_profile(changes: ArrayLike, alpha: float = DEFAULT_ALPHA) -> RiskProfile:
    """Compute every risk measure for a ``(n_prices, n_draws)`` matrix of outcome changes."""
    draws = _as_draws(changes)
    tail_mean = cvar(draws, alpha)
    return RiskProfile(
        alpha=alpha,
        variance=variance(draws),
        std=standard_deviation(draws),
        cvar=tail_mean,
        shortfall=_shortfall(tail_mean),
        downside_prob=downside_probability(draws),
    )
