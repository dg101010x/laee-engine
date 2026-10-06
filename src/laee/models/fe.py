"""Fixed-effects own-price elasticity baselines (spec §14, §39, §76; Phase 3).

``estimate_twfe`` fits ``ln Q_it = α_i + γ_t + β ln P_it + X_it δ + ε_it`` by OLS after absorbing
both sets of fixed effects with the method of alternating projections (Guimarães & Portugal
2010; Gaure 2013), so unbalanced panels with gaps are handled exactly. ``method="ppml"`` fits
``E[Q_it] = exp(α_i + γ_t + β ln P_it + X_it δ)`` by Poisson pseudo-maximum likelihood (Santos
Silva & Tenreyro 2006), keeping zero-sales days, via IRLS with the fixed effects absorbed on
weighted data (Correia, Guimarães & Zylkin 2020). Both report CR1 standard errors clustered by
product by default, singletons are dropped as in reghdfe (Correia 2015), and fixed effects nested
within the clusters do not count towards K in the CR1 factor. ``estimate_loglog_ols`` is the
naive pooled log-log regression, kept only as a tier-7 benchmark (spec §15: it is confounded).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from laee import schema
from laee.models.inference import (
    CovarianceResult,
    EstimationSample,
    MaskLike,
    control_matrix,
    day_numbers,
    emit_warnings,
    factorize,
    ols_vcov,
    prepare_sample,
    sandwich_vcov,
    t_intervals,
)
from laee.types import ElasticityEstimate

TIER_FIXED_EFFECTS = 6
TIER_CORRELATION = 7
TIME_FE_CHOICES = ("day", "week", "month")
METHODS = ("ols", "ppml")
MAP_TOL = 1e-10
MAP_MAX_ITER = 10_000
IRLS_TOL = 1e-10
IRLS_MAX_ITER = 200
COLLINEARITY_RTOL = 1e-8
_ETA_BOUND = 100.0


# --------------------------------------------------------------------------- fixed-effect algebra


@dataclass(frozen=True)
class DemeanResult:
    values: np.ndarray
    iterations: int
    converged: bool


def time_period_codes(dates: pd.Series, time_fe: str) -> np.ndarray:
    """Dense codes of calendar days, Monday-start weeks or months."""
    days = day_numbers(dates)
    if time_fe == "day":
        key = days
    elif time_fe == "week":
        key = (days + 3) // 7  # 1970-01-01 was a Thursday, so this bins Monday-to-Sunday weeks
    elif time_fe == "month":
        key = days.astype("datetime64[D]").astype("datetime64[M]").astype(np.int64)
    else:
        raise ValueError(f"time_fe must be one of {TIME_FE_CHOICES} or None, got {time_fe!r}")
    return factorize(key)[0]


def demean(
    values: np.ndarray,
    groups: Sequence[np.ndarray],
    *,
    weights: np.ndarray | None = None,
    tol: float = MAP_TOL,
    max_iter: int = MAP_MAX_ITER,
) -> DemeanResult:
    """Residualise columns on several sets of fixed effects by alternating (weighted) projections.

    Each sweep subtracts the (weighted) group means of every fixed effect in turn; iteration
    stops when the largest mean removed in a sweep, relative to max(1, max|column|), is below
    ``tol``. One set of fixed effects is exact after a single sweep. The limit is the residual of
    a (weighted) least-squares projection on all the dummies (von Neumann-Halperin theorem).
    """
    data = np.array(values, dtype=float, copy=True)
    vector = data.ndim == 1
    if vector:
        data = data[:, None]
    n_obs = data.shape[0]
    w = np.ones(n_obs) if weights is None else np.asarray(weights, dtype=float)
    projections = []
    for codes in groups:
        n_levels = int(codes.max()) + 1
        indicator = sparse.csr_matrix((w, (codes, np.arange(n_obs))), shape=(n_levels, n_obs))
        totals = np.asarray(indicator.sum(axis=1)).ravel()
        projections.append((codes, indicator, np.where(totals > 0, totals, 1.0)))
    scale = np.maximum(1.0, np.abs(data).max(axis=0)) if n_obs else np.ones(data.shape[1])
    iterations, converged = 0, False
    for iterations in range(1, max_iter + 1):
        change = 0.0
        for codes, indicator, totals in projections:
            means = (indicator @ data) / totals[:, None]
            data -= means[codes]
            change = max(change, float(np.max(np.abs(means) / scale, initial=0.0)))
        if len(projections) <= 1 or change < tol:
            converged = True
            break
    return DemeanResult(data[:, 0] if vector else data, iterations, converged)


def prune_groups(
    groups: Sequence[np.ndarray], outcome: np.ndarray | None = None
) -> tuple[np.ndarray, int, int]:
    """Iteratively drop singleton fixed-effect groups and, given ``outcome``, all-zero groups.

    A singleton's fixed effect fits it perfectly, so it carries no information on β but would
    inflate N and G in the small-sample factors (Correia 2015). For Poisson models a group whose
    outcomes are all zero has a fixed effect at -inf (separation; Correia, Guimarães & Zylkin
    2021). Returns the keep mask, the number of singleton rows and the number of separated rows.
    """
    n_obs = len(groups[0]) if groups else 0
    keep = np.ones(n_obs, dtype=bool)
    n_singletons = n_separated = 0
    changed = True
    while changed:
        changed = False
        for codes in groups:
            n_levels = int(codes.max()) + 1
            counts = np.bincount(codes, weights=keep, minlength=n_levels)
            single = keep & (counts[codes] == 1)
            if single.any():
                keep &= ~single
                n_singletons += int(single.sum())
                changed = True
            if outcome is not None:
                totals = np.bincount(codes, weights=np.where(keep, outcome, 0.0), minlength=n_levels)
                separated = keep & (totals[codes] <= 0)
                if separated.any():
                    keep &= ~separated
                    n_separated += int(separated.sum())
                    changed = True
    return keep, n_singletons, n_separated


def drop_singletons(groups: Sequence[np.ndarray]) -> np.ndarray:
    """Keep mask after iteratively removing rows that are alone in some fixed-effect group."""
    return prune_groups(groups)[0]


def absorbed_dof(
    groups: Sequence[np.ndarray], cluster_codes: np.ndarray | None = None
) -> tuple[int, int]:
    """Degrees of freedom absorbed by the fixed effects: (for CR1, for HC1).

    The dummies of two fixed effects span L1 + L2 - C dimensions, C being the number of
    connected components of the bipartite group graph (Abowd, Creecy & Kramarz 2002). For CR1,
    fixed effects nested within the clusters are excluded (as in reghdfe and xtreg, fe).
    """
    levels = [int(codes.max()) + 1 for codes in groups]
    total = sum(levels)
    if len(groups) == 2:
        total -= _n_components(groups[0], groups[1])
    elif len(groups) > 2:
        raise ValueError("at most two sets of fixed effects are supported")
    nested = sum(
        n for codes, n in zip(groups, levels, strict=True)
        if cluster_codes is not None and _is_nested(codes, cluster_codes)
    )
    return max(0, total - nested), total


def _n_components(first: np.ndarray, second: np.ndarray) -> int:
    n_first, n_second = int(first.max()) + 1, int(second.max()) + 1
    size = n_first + n_second
    graph = sparse.coo_matrix((np.ones(len(first)), (first, n_first + second)), shape=(size, size))
    return int(connected_components(graph, directed=False)[0])


def _is_nested(codes: np.ndarray, cluster_codes: np.ndarray) -> bool:
    n_clusters = int(cluster_codes.max()) + 1
    pairs = np.unique(codes.astype(np.int64) * n_clusters + cluster_codes)
    return bool(np.all(np.bincount(pairs // n_clusters) <= 1))


def _select_regressors(
    demeaned: np.ndarray, raw: np.ndarray, names: Sequence[str]
) -> tuple[list[int], list[str]]:
    """Indices of regressors that keep variation after absorption and are not collinear.

    Gram-Schmidt with re-orthogonalisation in column order; ln P (column 0) must survive.
    """
    kept: list[int] = []
    basis: list[np.ndarray] = []
    for j in range(demeaned.shape[1]):
        column = demeaned[:, j]
        reference = np.linalg.norm(raw[:, j] - raw[:, j].mean())
        residual = column.copy()
        for _ in range(2):
            for q in basis:
                residual -= (q @ residual) * q
        norm = np.linalg.norm(residual)
        threshold = COLLINEARITY_RTOL * max(reference, float(np.linalg.norm(column)))
        if reference > 0 and norm > threshold:
            kept.append(j)
            basis.append(residual / norm)
        elif j == 0:
            raise ValueError("ln price has no variation left after absorbing the fixed effects")
    return kept, [names[j] for j in range(len(names)) if j not in kept]


# --------------------------------------------------------------------------- estimators


@dataclass(frozen=True)
class _Design:
    outcome: np.ndarray
    regressors: np.ndarray
    names: list[str]
    groups: list[np.ndarray]
    product_codes: np.ndarray
    cluster_codes: np.ndarray | None
    n_singletons: int
    n_separated: int


def _build_design(
    sample: EstimationSample,
    controls: Sequence[str],
    outcome: np.ndarray,
    time_fe: str | None,
    cluster: str | None,
    *,
    pooled: bool = False,
    poisson: bool = False,
) -> _Design:
    frame = sample.frame
    regressors = np.column_stack([sample.log_price, control_matrix(frame, controls)])
    product_codes = factorize(frame[schema.PRODUCT_ID])[0]
    if pooled:
        groups = [np.zeros(len(frame), dtype=np.int64)]
    else:
        groups = [product_codes]
        if time_fe is not None:
            groups.append(time_period_codes(frame[schema.DATE], time_fe))
    cluster_codes = factorize(frame[cluster])[0] if cluster is not None else None
    keep, n_singletons, n_separated = (
        (np.ones(len(frame), dtype=bool), 0, 0)
        if pooled else prune_groups(groups, outcome if poisson else None)
    )
    if not keep.any():
        raise ValueError("no observations left after dropping singletons/separated groups")
    return _Design(
        outcome=outcome[keep],
        regressors=regressors[keep],
        names=["log_price", *controls],
        groups=[factorize(codes[keep])[0] for codes in groups],
        product_codes=factorize(product_codes[keep])[0],
        cluster_codes=factorize(cluster_codes[keep])[0] if cluster_codes is not None else None,
        n_singletons=n_singletons,
        n_separated=n_separated,
    )


def estimate_twfe(
    panel: pd.DataFrame,
    *,
    controls: Sequence[str] = (),
    time_fe: str | None = "week",
    cluster: str | None = schema.PRODUCT_ID,
    mask: MaskLike = None,
    method: str = "ols",
) -> ElasticityEstimate:
    """Two-way fixed-effects own-price elasticity (identification tier 6, spec §76).

    Parameters
    ----------
    panel:
        Sales panel with ``PRODUCT_ID``, ``DATE``, ``PRICE``, ``UNITS`` and the controls.
    controls:
        Numeric or boolean columns entering linearly (e.g. ``PROMO_FLAG``, a demand proxy).
    time_fe:
        ``"day"``, ``"week"`` (Monday-start), ``"month"`` or None (product effects only).
    cluster:
        Column defining clusters for CR1 standard errors (None gives HC1). Fewer than 10
        clusters fall back to HC1 with a warning.
    mask:
        Boolean row filter (True keeps), e.g. ``~panel[CENSORED_FLAG]``.
    method:
        ``"ols"`` on ln Q (rows with units <= 0 dropped and counted) or ``"ppml"`` on Q
        (zeros kept).

    Diagnostics include ``n_obs``, the sample counts of ``prepare_sample`` (``n_masked_out``,
    ``n_dropped_invalid``, ``n_dropped_nonpositive``, ``n_stockout_rows_included``),
    ``n_singletons_dropped``, ``n_clusters``, ``se_type`` (CR1/HC1), ``df``, ``within_r2`` (OLS:
    R² of the demeaned regression; PPML: weighted R² of the final IRLS working regression),
    ``sd_demeaned_log_price`` (identifying variation, μ-weighted for PPML),
    ``log_price_variance_share``, ``n_products_with_price_variation``, ``absorbed_dof``,
    ``control_coefficients``, ``dropped_controls`` and ``warnings``; PPML adds
    ``irls_iterations``, ``irls_converged``, ``deviance``, ``pseudo_r2`` (squared correlation of
    Q and μ), ``n_zero_units`` and ``n_separated_dropped``.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if time_fe is not None and time_fe not in TIME_FE_CHOICES:
        raise ValueError(f"time_fe must be one of {TIME_FE_CHOICES} or None, got {time_fe!r}")
    controls = tuple(controls)
    sample = prepare_sample(
        panel, columns=controls, mask=mask, cluster=cluster,
        units_rule="positive" if method == "ols" else "nonnegative",
    )
    if method == "ols":
        design = _build_design(sample, controls, np.log(sample.units), time_fe, cluster)
        return _fit_ols(sample, design, method="twfe_ols", tier=TIER_FIXED_EFFECTS, time_fe=time_fe)
    design = _build_design(sample, controls, sample.units, time_fe, cluster, poisson=True)
    return _fit_ppml(sample, design, time_fe=time_fe)


def estimate_loglog_ols(
    panel: pd.DataFrame,
    *,
    controls: Sequence[str] = (),
    cluster: str | None = schema.PRODUCT_ID,
    mask: MaskLike = None,
) -> ElasticityEstimate:
    """Naive pooled OLS of ln Q on ln P (and controls) with an intercept only.

    Ignores product and time heterogeneity and price endogeneity (spec §15), so it is reported
    as identification tier 7 (pure predictive correlation) and serves as a benchmark.
    """
    controls = tuple(controls)
    sample = prepare_sample(panel, columns=controls, mask=mask, cluster=cluster)
    design = _build_design(sample, controls, np.log(sample.units), None, cluster, pooled=True)
    return _fit_ols(sample, design, method="loglog_ols", tier=TIER_CORRELATION, time_fe=None)


def _fit_ols(
    sample: EstimationSample, design: _Design, *, method: str, tier: int, time_fe: str | None
) -> ElasticityEstimate:
    stacked = demean(np.column_stack([design.outcome, design.regressors]), design.groups)
    y_dm, x_dm = stacked.values[:, 0], stacked.values[:, 1:]
    kept, dropped = _select_regressors(x_dm, design.regressors, design.names)
    x_dm = x_dm[:, kept]
    coef = np.linalg.lstsq(x_dm, y_dm, rcond=None)[0]
    resid = y_dm - x_dm @ coef
    k_cluster, k_hc = absorbed_dof(design.groups, design.cluster_codes)
    cov = ols_vcov(x_dm, resid, cluster_codes=design.cluster_codes,
                   n_absorbed_cluster=k_cluster, n_absorbed_hc=k_hc)
    total = float(y_dm @ y_dm)
    within_r2 = 1.0 - float(resid @ resid) / total if total > 0 else float("nan")
    return _assemble(
        sample, design, coef, cov, kept, dropped, method=method, tier=tier, time_fe=time_fe,
        demeaned_price=x_dm[:, 0], weights=None, within_r2=within_r2, absorbed=k_hc,
        converged=stacked.converged, extra={"map_iterations": stacked.iterations},
    )


def _fit_ppml(sample: EstimationSample, design: _Design, *, time_fe: str | None) -> ElasticityEstimate:
    x_raw_dm = demean(design.regressors, design.groups).values
    kept, dropped = _select_regressors(x_raw_dm, design.regressors, design.names)
    x = design.regressors[:, kept]
    y = design.outcome
    fit = _ppml_irls(y, x, design.groups)
    final = demean(x, design.groups, weights=fit.mu)
    x_dm = final.values
    k_cluster, k_hc = absorbed_dof(design.groups, design.cluster_codes)
    cov = sandwich_vcov(
        (x_dm * fit.mu[:, None]).T @ x_dm,
        x_dm * (y - fit.mu)[:, None],
        cluster_codes=design.cluster_codes,
        k_cluster=x.shape[1] + k_cluster,
        k_hc=x.shape[1] + k_hc,
    )
    messages = []
    if not fit.converged:
        messages.append(f"PPML IRLS did not converge in {fit.iterations} iterations")
    if np.any(fit.mu < 1e-12):
        messages.append("some fitted means are ~0: possible separation; estimates may be unstable")
    pseudo_r2 = float(np.corrcoef(y, fit.mu)[0, 1] ** 2) if np.std(y) > 0 else float("nan")
    return _assemble(
        sample, design, fit.coef, cov, kept, dropped, method="twfe_ppml", tier=TIER_FIXED_EFFECTS,
        time_fe=time_fe, demeaned_price=x_dm[:, 0], weights=fit.mu, within_r2=fit.working_r2,
        absorbed=k_hc, converged=final.converged and fit.map_converged, extra_messages=messages,
        extra={
            "irls_iterations": fit.iterations,
            "irls_converged": fit.converged,
            "deviance": fit.deviance,
            "pseudo_r2": pseudo_r2,
            "n_zero_units": int(np.sum(y == 0)),
            "n_separated_dropped": design.n_separated,
        },
    )


@dataclass(frozen=True)
class _PoissonFit:
    coef: np.ndarray
    mu: np.ndarray
    deviance: float
    working_r2: float
    iterations: int
    converged: bool
    map_converged: bool


def _ppml_irls(y: np.ndarray, x: np.ndarray, groups: Sequence[np.ndarray]) -> _PoissonFit:
    """Poisson PML by IRLS (Newton-Raphson for the canonical log link) with absorbed FE.

    Each step regresses the working response z = η + (y - μ)/μ on the regressors with weights μ;
    by weighted Frisch-Waugh-Lovell the fixed effects are absorbed by weighted demeaning, and the
    new linear predictor is z minus the weighted within residual.
    """
    mu = 0.5 * (y + y.mean())
    eta = np.log(mu)
    coef = np.zeros(x.shape[1])
    deviance_old = np.inf
    working_r2 = float("nan")
    converged = map_converged = False
    iteration = 0
    for iteration in range(1, IRLS_MAX_ITER + 1):
        z = eta + (y - mu) / mu
        stacked = demean(np.column_stack([z, x]), groups, weights=mu)
        map_converged = stacked.converged
        z_dm, x_dm = stacked.values[:, 0], stacked.values[:, 1:]
        weighted = x_dm * mu[:, None]
        coef_new = np.linalg.solve(weighted.T @ x_dm, weighted.T @ z_dm)
        resid = z_dm - x_dm @ coef_new
        total = float(np.sum(mu * z_dm**2))
        working_r2 = 1.0 - float(np.sum(mu * resid**2)) / total if total > 0 else float("nan")
        eta = np.clip(z - resid, -_ETA_BOUND, _ETA_BOUND)
        mu = np.exp(eta)
        deviance = _poisson_deviance(y, mu)
        step = float(np.max(np.abs(coef_new - coef), initial=0.0))
        coef = coef_new
        relative = abs(deviance - deviance_old) / max(min(deviance, deviance_old), 0.1)
        if relative < IRLS_TOL and step < 1e-8:
            converged = True
            break
        deviance_old = deviance
    return _PoissonFit(coef, mu, deviance, working_r2, iteration, converged, map_converged)


def _poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    positive = y > 0
    return float(2.0 * (np.sum(y[positive] * np.log(y[positive] / mu[positive])) - np.sum(y - mu)))


def _assemble(
    sample: EstimationSample,
    design: _Design,
    coef: np.ndarray,
    cov: CovarianceResult,
    kept: list[int],
    dropped: list[str],
    *,
    method: str,
    tier: int,
    time_fe: str | None,
    demeaned_price: np.ndarray,
    weights: np.ndarray | None,
    within_r2: float,
    absorbed: int,
    converged: bool,
    extra: dict[str, object],
    extra_messages: Sequence[str] = (),
) -> ElasticityEstimate:
    std_errors = np.sqrt(np.clip(np.diag(cov.vcov), 0.0, None))
    value, std_error = float(coef[0]), float(std_errors[0])
    messages = [*sample.warnings, *cov.warnings, *extra_messages]
    if dropped:
        messages.append(f"dropped controls collinear with the fixed effects or each other: {dropped}")
    if not converged:
        messages.append("alternating projections hit the iteration cap; fixed effects not fully absorbed")
    price = design.regressors[:, 0]
    with_variation = _n_groups_with_variation(price, design.product_codes)
    if with_variation < 2:
        messages.append(f"only {with_variation} product(s) have price variation; β is identified "
                        "from very few price changes")
    w = np.ones(len(price)) if weights is None else weights
    sd_demeaned = float(np.sqrt(np.average(demeaned_price**2, weights=w)))
    raw_var = float(np.var(price))
    names = [design.names[j] for j in kept]
    diagnostics: dict[str, object] = {
        "method": method,
        "time_fe": time_fe,
        "n_obs": len(price),
        **sample.counts,
        "n_singletons_dropped": design.n_singletons,
        "n_clusters": cov.n_clusters,
        "se_type": cov.kind,
        "df": cov.df,
        "within_r2": within_r2,
        "sd_demeaned_log_price": sd_demeaned,
        "log_price_variance_share": sd_demeaned**2 / raw_var if raw_var > 0 else float("nan"),
        "n_products_with_price_variation": with_variation,
        "n_time_periods": int(design.groups[1].max()) + 1 if len(design.groups) > 1 else None,
        "absorbed_dof": absorbed,
        "control_coefficients": {
            name: {"coef": float(c), "std_error": float(s)}
            for name, c, s in zip(names[1:], coef[1:], std_errors[1:], strict=True)
        },
        "dropped_controls": dropped,
        **extra,
        "warnings": messages,
    }
    emit_warnings(messages, stacklevel=5)
    return ElasticityEstimate(
        value=value,
        std_error=std_error,
        intervals=t_intervals(value, std_error, cov.df),
        method=method,
        identification_tier=tier,
        n_obs=len(price),
        n_products=int(design.product_codes.max()) + 1,
        diagnostics=diagnostics,
    )


def _n_groups_with_variation(values: np.ndarray, codes: np.ndarray) -> int:
    series = pd.Series(values)
    spread = series.groupby(codes).max() - series.groupby(codes).min()
    return int((spread > 1e-12).sum())
