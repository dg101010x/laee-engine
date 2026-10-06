"""Synthetic SMB sales panels with a known causal own-price elasticity.

Every LAEE estimator is validated on panels drawn from :func:`simulate_panel`, whose
data-generating process (DGP) is fully known. For product ``i`` on day ``t`` (natural logs)::

    ln λ_it = α_i + β_i·ln P_it + w(t) + s(t) + ω_week(t) + ζ_t + κ·promo_it + a_it
    D_it    = λ_it·exp(e_it)                    noise="lognormal", e ~ N(-σ²/2, σ²)
    D_it    ~ Poisson(λ_it·exp(e_it))           noise="poisson" (Poisson-lognormal mixture)
    Q_it    = min(D_it, inventory_it)            stockouts censor demand (spec §63-64)

* ``α_i`` product fixed effect; ``w``/``s`` weekly and annual seasonality (one harmonic each,
  so the default Fourier nuisance features span them); ``ω``/``ζ`` common weekly and daily
  shocks. Common shocks move demand only, never price.
* The product-day demand shock is ``a_it + e_it``. Its persistent AR(1) part ``a_it`` is
  anticipated by the firm, which passes ``confounding·0.5·a`` into the regular price whenever it
  reprices; the analyst observes the same signal as the ``demand_proxy`` column. Regressions of
  ln Q on ln P that omit the proxy are therefore biased upwards (spec §15), while conditioning
  on it restores unconfoundedness exactly; the unanticipated part ``e_it`` is the demand noise.
* Cross-sectional confounding: markups rise by ``confounding`` log points per log point of
  product demand level, so pooled regressions without product fixed effects are biased upwards.
* ``cost_shifter`` is a product-specific AR(1) log unit-cost shock passed one-for-one into the
  regular price at repricing dates. It moves price but not demand: a valid instrument.
* Regular prices are sticky (Calvo repricing, mean spell 20-45 days, charm-rounded to ``x.x9``).
  Temporary promotions cut the price by ``promo_depth`` and lift demand by ``promo_lift`` log
  points (display/feature effect, spec §62), so promotions must be controlled for.
* With ``stockout_rate > 0`` a product-day is supply-constrained with that probability: stock is
  a uniform 10-80% of the product's typical daily demand and does not respond to price, so
  censored rows flatten the price response unless they are masked out.
* ``sparsity`` drops product-day rows completely at random (unbalanced panel with gaps).

All randomness comes from the ``rng`` argument and every random block is drawn
unconditionally in a fixed order, so changing e.g. ``stockout_rate`` with the same seed keeps
the latent demand draws identical (common random numbers).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize, signal

from laee import schema

DEMAND_PROXY = "demand_proxy"
COST_SHIFTER = "cost_shifter"
NOISE_MODELS = ("lognormal", "poisson")

ElasticitySpec = float | Callable[[pd.DataFrame], Any]

COVARIATE_COLUMNS = (
    schema.PRODUCT_ID,
    "log_base_price",
    "log_base_cost",
    "demand_level",
    "quality",
    "category",
)

_DEMAND_LEVEL_SD = 0.4
_BASE_UNIT_COST = 4.0
_COST_LEVEL_SD = 0.15
_BASE_MARKUP = 2.0
_MARKUP_NOISE_SD = 0.08
_MARKUP_DEMAND_RESPONSE = 1.0
_SHOCK_PASS_THROUGH = 0.5
_SHOCK_SD = 0.25
_SHOCK_RHO = 0.98
_COST_RHO = 0.98
_REPRICE_HAZARD = (1 / 45, 1 / 20)
_PROMO_LENGTH = (3, 7)
_WEEKLY_AMPLITUDE = 0.15
_ANNUAL_AMPLITUDE = 0.20
_WEEK_SHOCK_SD = 0.08
_DAY_SHOCK_SD = 0.04
_STOCK_FRACTION = (0.1, 0.8)
_PRICE_STEP = 0.10
_N_CATEGORIES = 4
# 1970-01-01 (day 0) was a Thursday: day 2.5 sits between Saturday and Sunday, day 196 is mid-July.
_WEEKLY_PEAK_DAY = 2.5
_ANNUAL_PEAK_DAY = 196.0
_YEAR_DAYS = 365.25


def simulate_panel(
    rng: np.random.Generator,
    *,
    n_products: int = 20,
    n_days: int = 365,
    elasticity: ElasticitySpec = -1.5,
    confounding: float = 0.8,
    cost_shock_sd: float = 0.10,
    promo_rate: float = 0.05,
    promo_depth: float = 0.2,
    stockout_rate: float = 0.0,
    sparsity: float = 0.0,
    noise: str = "lognormal",
    noise_sd: float = 0.3,
    start: str = "2025-01-01",
    base_demand: float = 20.0,
    promo_lift: float = 0.15,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Draw a daily product panel with known elasticity; see the module docstring for the DGP.

    Parameters
    ----------
    rng:
        Source of all randomness.
    elasticity:
        A constant own-price elasticity, or a callable mapping the product covariate frame
        (columns ``COVARIATE_COLUMNS``, one row per product) to one elasticity per product.
    confounding:
        Scales both the price response to the anticipated demand shock and the markup response
        to product demand levels; 0 makes price exogenous given product and promotion.
    cost_shock_sd:
        Stationary standard deviation of the log unit-cost shock (the instrument).
    promo_rate, promo_depth:
        Expected share of promotion days and fractional price cut while on promotion.
    stockout_rate:
        Probability that a product-day is supply-constrained (realised censoring is slightly lower).
    sparsity:
        Probability that a product-day row is missing.
    noise, noise_sd:
        ``"lognormal"`` (continuous units) or ``"poisson"`` (counts); ``noise_sd`` is the sd of the
        mean-one log-normal demand multiplier in both cases (0 gives exact or pure-Poisson demand).
    base_demand:
        Typical expected daily units of a product at the average base price (sets the zero share
        of Poisson panels).
    promo_lift:
        Log demand lift of promotions on top of their price cut.

    Returns
    -------
    panel, truth
        ``panel`` holds the ``laee.schema`` sales columns plus ``demand_proxy`` and
        ``cost_shifter``; ``DISCOUNT_PCT`` is a fraction, ``INVENTORY`` is NaN on days whose
        stock does not bind and ``AVAILABLE_AT`` is the following midnight (a day's sales are
        known once it closes, spec §75). ``truth`` holds ``elasticity`` (the constant, or the mean of the
        per-product values), ``elasticity_by_product``, ``heterogeneous``, ``product_covariates``,
        ``product_fe``, ``time_effect`` (w + s + ω + ζ by date), ``promo_lift``,
        ``proxy_coefficient``, ``shock_pass_through``, ``markup_demand_response``,
        ``expected_demand`` and ``latent_demand`` (row-aligned with ``panel``), realised promotion,
        stockout and censoring shares, and ``params``.
    """
    _validate(n_products, n_days, cost_shock_sd, promo_rate, promo_depth, stockout_rate,
              sparsity, noise, noise_sd, base_demand)
    dates = pd.date_range(pd.Timestamp(start).normalize(), periods=n_days, freq="D")
    day = dates.to_numpy().astype("datetime64[D]").astype(np.int64)
    width = max(3, len(str(n_products - 1)))
    product_ids = np.array([f"P{i:0{width}d}" for i in range(n_products)])

    demand_z = rng.standard_normal(n_products)
    quality = rng.standard_normal(n_products)
    category = rng.integers(0, _N_CATEGORIES, size=n_products)
    log_cost0 = np.log(_BASE_UNIT_COST) + _COST_LEVEL_SD * rng.standard_normal(n_products)
    log_markup = (
        np.log(_BASE_MARKUP)
        + confounding * _MARKUP_DEMAND_RESPONSE * _DEMAND_LEVEL_SD * demand_z
        + _MARKUP_NOISE_SD * rng.standard_normal(n_products)
    )
    log_price0 = log_cost0 + log_markup
    covariates = pd.DataFrame(
        {
            schema.PRODUCT_ID: product_ids,
            "log_base_price": log_price0,
            "log_base_cost": log_cost0,
            "demand_level": demand_z,
            "quality": quality,
            "category": category,
        }
    )
    beta, heterogeneous = _resolve_elasticity(elasticity, covariates)
    alpha = np.log(base_demand) + _DEMAND_LEVEL_SD * demand_z - beta * log_price0.mean()

    shock = _ar1(rng, n_products, n_days, _SHOCK_RHO, _SHOCK_SD)
    cost = _ar1(rng, n_products, n_days, _COST_RHO, cost_shock_sd)
    regular_price = _sticky_regular_price(
        rng, log_price0, cost, confounding * _SHOCK_PASS_THROUGH * shock
    )
    promo = _promotion_mask(rng, n_products, n_days, promo_rate)
    price = np.where(promo, np.round(regular_price * (1.0 - promo_depth), 2), regular_price)
    price = np.maximum(price, 0.01)
    time_effect = _time_effect(rng, day)

    log_lambda = (
        alpha[:, None]
        + beta[:, None] * np.log(price)
        + time_effect[None, :]
        + promo_lift * promo
        + shock
    )
    expected = np.exp(log_lambda)
    multiplier = np.exp(noise_sd * rng.standard_normal((n_products, n_days)) - 0.5 * noise_sd**2)
    constrained = rng.random((n_products, n_days)) < stockout_rate
    stock_share = rng.uniform(*_STOCK_FRACTION, size=(n_products, n_days))
    keep = rng.random(n_products * n_days) >= sparsity
    if noise == "poisson":
        demand = rng.poisson(expected * multiplier).astype(float)
    else:
        demand = expected * multiplier

    typical = np.exp(alpha + beta * log_price0)[:, None]
    stock = stock_share * typical
    if noise == "poisson":
        stock = np.floor(stock)
    units = np.where(constrained, np.minimum(demand, stock), demand)
    stockout = constrained & (demand >= stock)
    censored = constrained & (demand > stock)
    inventory = np.where(constrained, stock, np.nan)
    unit_cost = np.round(np.exp(log_cost0[:, None] + cost), 2)
    discount = np.where(promo, np.round(1.0 - price / regular_price, 4), 0.0)

    panel = pd.DataFrame(
        {
            schema.BUSINESS_ID: "B0",
            schema.PRODUCT_ID: np.repeat(product_ids, n_days),
            schema.LOCATION_ID: "L0",
            schema.DATE: np.tile(dates.to_numpy(), n_products),
            schema.PRICE: price.ravel(),
            schema.UNITS: units.ravel(),
            schema.REVENUE: (price * units).ravel(),
            schema.UNIT_COST: unit_cost.ravel(),
            schema.PROMO_FLAG: promo.ravel(),
            schema.DISCOUNT_PCT: discount.ravel(),
            schema.REGULAR_PRICE: regular_price.ravel(),
            schema.INVENTORY: inventory.ravel(),
            schema.STOCKOUT_FLAG: stockout.ravel(),
            schema.CENSORED_FLAG: censored.ravel(),
            schema.AVAILABLE_AT: np.tile((dates + pd.Timedelta(days=1)).to_numpy(), n_products),
            DEMAND_PROXY: shock.ravel(),
            COST_SHIFTER: cost.ravel(),
        }
    )
    panel = panel.loc[keep].reset_index(drop=True)

    params = {
        "n_products": n_products,
        "n_days": n_days,
        "confounding": confounding,
        "cost_shock_sd": cost_shock_sd,
        "promo_rate": promo_rate,
        "promo_depth": promo_depth,
        "stockout_rate": stockout_rate,
        "sparsity": sparsity,
        "noise": noise,
        "noise_sd": noise_sd,
        "start": str(dates[0].date()),
        "base_demand": base_demand,
        "promo_lift": promo_lift,
        "demand_level_sd": _DEMAND_LEVEL_SD,
        "cost_level_sd": _COST_LEVEL_SD,
        "markup_noise_sd": _MARKUP_NOISE_SD,
        "shock_sd": _SHOCK_SD,
        "shock_rho": _SHOCK_RHO,
        "cost_rho": _COST_RHO,
        "reprice_hazard": _REPRICE_HAZARD,
        "promo_length": _PROMO_LENGTH,
        "weekly_amplitude": _WEEKLY_AMPLITUDE,
        "annual_amplitude": _ANNUAL_AMPLITUDE,
        "week_shock_sd": _WEEK_SHOCK_SD,
        "day_shock_sd": _DAY_SHOCK_SD,
        "stock_fraction": _STOCK_FRACTION,
    }
    flat_promo = promo.ravel()[keep]
    truth: dict[str, Any] = {
        "elasticity": float(beta.mean()),
        "elasticity_by_product": dict(zip(product_ids.tolist(), beta.tolist(), strict=True)),
        "heterogeneous": heterogeneous,
        "product_covariates": covariates,
        "product_fe": dict(zip(product_ids.tolist(), alpha.tolist(), strict=True)),
        "time_effect": pd.Series(time_effect, index=dates, name="time_effect"),
        "promo_lift": float(promo_lift),
        "proxy_coefficient": 1.0,
        "shock_pass_through": confounding * _SHOCK_PASS_THROUGH,
        "markup_demand_response": confounding * _MARKUP_DEMAND_RESPONSE,
        "expected_demand": expected.ravel()[keep],
        "latent_demand": demand.ravel()[keep],
        "realized_promo_share": _share(flat_promo),
        "realized_stockout_share": _share(stockout.ravel()[keep]),
        "realized_censored_share": _share(censored.ravel()[keep]),
        "params": params,
    }
    return panel, truth


def _validate(
    n_products: int,
    n_days: int,
    cost_shock_sd: float,
    promo_rate: float,
    promo_depth: float,
    stockout_rate: float,
    sparsity: float,
    noise: str,
    noise_sd: float,
    base_demand: float,
) -> None:
    if n_products < 1 or n_days < 1:
        raise ValueError("n_products and n_days must be positive")
    if noise not in NOISE_MODELS:
        raise ValueError(f"noise must be one of {NOISE_MODELS}, got {noise!r}")
    if cost_shock_sd < 0 or noise_sd < 0:
        raise ValueError("standard deviations must be non-negative")
    if not 0 <= promo_rate < 1 or not 0 <= promo_depth < 1:
        raise ValueError("promo_rate and promo_depth must lie in [0, 1)")
    if not 0 <= stockout_rate <= 1 or not 0 <= sparsity < 1:
        raise ValueError("stockout_rate must lie in [0, 1] and sparsity in [0, 1)")
    if base_demand <= 0:
        raise ValueError("base_demand must be positive")


def _resolve_elasticity(
    elasticity: ElasticitySpec, covariates: pd.DataFrame
) -> tuple[np.ndarray, bool]:
    n = len(covariates)
    if callable(elasticity):
        values = np.asarray(elasticity(covariates.copy()), dtype=float).reshape(-1)
        if values.shape != (n,):
            raise ValueError(f"elasticity callable must return {n} values, got {values.shape}")
        heterogeneous = True
    else:
        values = np.full(n, float(elasticity))
        heterogeneous = False
    if not np.all(np.isfinite(values)):
        raise ValueError("elasticities must be finite")
    return values, heterogeneous


def _ar1(rng: np.random.Generator, n_rows: int, n_cols: int, rho: float, sd: float) -> np.ndarray:
    """Stationary AR(1) rows x_t = rho·x_{t-1} + u_t with marginal standard deviation ``sd``."""
    innovations = rng.standard_normal((n_rows, n_cols)) * (sd * np.sqrt(1.0 - rho**2))
    innovations[:, 0] /= np.sqrt(1.0 - rho**2)
    return signal.lfilter([1.0], [1.0, -rho], innovations, axis=1)


def _charm_round(price: np.ndarray) -> np.ndarray:
    """Round to the nearest 10 cents minus one cent (9.99, 10.09, ...): spec §65 price points."""
    return np.round(np.maximum(np.round(price / _PRICE_STEP) * _PRICE_STEP - 0.01, 0.09), 2)


def _sticky_regular_price(
    rng: np.random.Generator,
    log_price0: np.ndarray,
    cost: np.ndarray,
    anticipation: np.ndarray,
) -> np.ndarray:
    """Calvo repricing: on repricing days the shelf price jumps to the rounded target, else stays."""
    n_products, n_days = cost.shape
    hazard = rng.uniform(*_REPRICE_HAZARD, size=n_products)
    reprice = rng.random((n_products, n_days)) < hazard[:, None]
    reprice[:, 0] = True
    target = _charm_round(np.exp(log_price0[:, None] + cost + anticipation))
    last = np.maximum.accumulate(np.where(reprice, np.arange(n_days)[None, :], 0), axis=1)
    return np.take_along_axis(target, last, axis=1)


def _promotion_mask(
    rng: np.random.Generator, n_products: int, n_days: int, rate: float
) -> np.ndarray:
    lo, hi = _PROMO_LENGTH
    start_draws = rng.random((n_products, n_days))
    lengths = rng.integers(lo, hi + 1, size=(n_products, n_days))
    if rate <= 0:
        return np.zeros((n_products, n_days), dtype=bool)
    starts = start_draws < _promo_start_probability(rate)
    rows, cols = np.nonzero(starts)
    edges = np.zeros((n_products, n_days + hi + 1))
    np.add.at(edges, (rows, cols), 1.0)
    np.add.at(edges, (rows, cols + lengths[rows, cols]), -1.0)
    return np.cumsum(edges, axis=1)[:, :n_days] > 0.5


def _promo_start_probability(rate: float) -> float:
    """Daily start probability p whose stationary coverage 1 - Π_k (1 - p·P(L > k)) equals rate."""
    lo, hi = _PROMO_LENGTH
    lengths = np.arange(lo, hi + 1)
    survival = np.array([(lengths > k).mean() for k in range(hi)])

    def gap(p: float) -> float:
        return 1.0 - np.prod(1.0 - p * survival) - rate

    return float(optimize.brentq(gap, 0.0, 1.0, xtol=1e-14))


def _time_effect(rng: np.random.Generator, day: np.ndarray) -> np.ndarray:
    weekly = _WEEKLY_AMPLITUDE * np.cos(2 * np.pi * (day - _WEEKLY_PEAK_DAY) / 7.0)
    annual = _ANNUAL_AMPLITUDE * np.cos(2 * np.pi * (day - _ANNUAL_PEAK_DAY) / _YEAR_DAYS)
    week = (day + 3) // 7
    week_codes = week - week.min()
    week_shock = _WEEK_SHOCK_SD * rng.standard_normal(int(week_codes.max()) + 1)
    day_shock = _DAY_SHOCK_SD * rng.standard_normal(len(day))
    return weekly + annual + week_shock[week_codes] + day_shock


def _share(flags: np.ndarray) -> float:
    return float(flags.mean()) if flags.size else float("nan")
