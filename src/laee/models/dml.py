"""Partially linear Double Machine Learning for own-price elasticity (spec §16-17; Phase 4).

Model (Chernozhukov, Chetverikov, Demirer, Duflo, Hansen, Newey & Robins 2018, "Double/debiased
machine learning for treatment and structural parameters", Econometrics Journal):

    Y = θ·T + m0(X) + ε,   E[ε | T, X] = 0,   with Y = ln Q and T = ln P.

The partialling-out score ψ = (Ỹ - θT̃)·T̃ on cross-fitted residuals Ỹ = Y - m̂(X) and
T̃ = T - ĝ(X), where m̂ ≈ E[Y|X] and ĝ ≈ E[T|X], is Neyman-orthogonal, so first-order errors in
the nuisance fits do not bias θ̂ = ΣT̃Ỹ / ΣT̃². Cross-fitting uses contiguous calendar-time blocks
with an embargo around each held-out block (sales and prices are serially correlated, spec §46),
or whole products held out (``fold_scheme="grouped"``). A held-out calendar block cannot learn
its own seasonal level or common daily shocks from other blocks, and that misfit loads on θ̂
whenever prices share time patterns, so by default Y, T and X are demeaned within calendar day
(day fixed effects, as in two-way FE). Every day lies entirely inside one fold's training,
held-out or embargo set, so this never mixes held-out and training rows. In grouped mode product
fixed effects cannot be learned for held-out products, so Y, T and X are instead demeaned within
product (the within-group DML of Clarke & Polselli 2023, "Double machine learning for static
panel models with fixed effects"); both transformations are exact when the nuisance is additive
in the fixed effects.
Repeated cross-fitting is combined by the median with the median-adjusted variance
(Chernozhukov et al. 2018, sec. 3.4).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from laee import schema
from laee.models.fe import absorbed_dof, demean, prune_groups
from laee.models.inference import (
    MIN_CLUSTERS,
    EstimationSample,
    MaskLike,
    day_numbers,
    emit_warnings,
    factorize,
    prepare_sample,
    sandwich_vcov,
    t_intervals,
)
from laee.models.learners import (
    FoldSplit,
    LearnerSpec,
    benchmark_learners,
    build_nuisance_features,
    cross_fit_predict,
    learner_factory,
    learner_label,
)
from laee.types import ElasticityEstimate

TIER_DML = 5
FOLD_SCHEMES = ("time_blocked", "grouped")
TIME_FE_CHOICES = ("day", None)
WEAK_VARIATION_SHARE = 0.05
FOLD_JITTER = 0.25


# --------------------------------------------------------------------------- folds


def time_blocked_folds(
    days: np.ndarray,
    n_folds: int,
    embargo_days: int = 0,
    rng: np.random.Generator | None = None,
) -> list[FoldSplit]:
    """Contiguous calendar-time folds with an embargo, as (train_idx, test_idx) pairs.

    Block boundaries sit at the row-weighted quantiles k/n_folds of the calendar days; with
    ``rng`` each interior boundary is jittered by up to ±25% of a block so repeated
    cross-fitting draws different splits. A row is a training row for block k only if its day
    is more than ``embargo_days`` before the block's first day or after its last day.
    """
    days = np.asarray(days, dtype=np.int64)
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if embargo_days < 0:
        raise ValueError("embargo_days must be non-negative")
    unique_days, counts = np.unique(days, return_counts=True)
    if len(unique_days) < n_folds:
        raise ValueError(f"need at least {n_folds} distinct days for {n_folds} time blocks")
    targets = np.arange(1, n_folds) / n_folds
    if rng is not None:
        targets = targets + rng.uniform(-FOLD_JITTER, FOLD_JITTER, size=n_folds - 1) / n_folds
    cumulative = np.cumsum(counts) / counts.sum()
    first = np.clip(np.searchsorted(cumulative, targets, side="left") + 1, 1, len(unique_days) - 1)
    if np.any(np.diff(first) <= 0):
        chunks = np.array_split(np.arange(len(unique_days)), n_folds)
        first = np.cumsum([len(chunk) for chunk in chunks])[:-1]
    block = np.searchsorted(unique_days[first], days, side="right")
    folds = []
    for k in range(n_folds):
        test = np.flatnonzero(block == k)
        lo, hi = days[test].min(), days[test].max()
        train = np.flatnonzero((days < lo - embargo_days) | (days > hi + embargo_days))
        folds.append((train, test))
    return folds


def grouped_folds(
    group_codes: np.ndarray, n_folds: int, rng: np.random.Generator | None = None
) -> list[FoldSplit]:
    """Folds that hold out whole groups (products); groups are dealt round-robin after shuffling."""
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    codes, n_groups = factorize(np.asarray(group_codes))
    if n_groups < n_folds:
        raise ValueError(f"need at least {n_folds} groups for {n_folds} grouped folds")
    order = rng.permutation(n_groups) if rng is not None else np.arange(n_groups)
    fold_of_group = np.empty(n_groups, dtype=np.int64)
    fold_of_group[order] = np.arange(n_groups) % n_folds
    fold = fold_of_group[codes]
    return [(np.flatnonzero(fold != k), np.flatnonzero(fold == k)) for k in range(n_folds)]


# --------------------------------------------------------------------------- score and aggregation


@dataclass(frozen=True)
class PartiallyLinearFit:
    theta: float
    std_error: float
    df: float
    variance_type: str
    n_clusters: int | None
    warnings: tuple[str, ...] = ()


def partialling_out(
    y_resid: np.ndarray,
    t_resid: np.ndarray,
    cluster_codes: np.ndarray | None = None,
    *,
    n_absorbed_cluster: int = 0,
    n_absorbed_hc: int = 0,
    min_clusters: int = MIN_CLUSTERS,
) -> PartiallyLinearFit:
    """θ̂ = ΣT̃Ỹ / ΣT̃² with its influence-function variance.

    ψ_i = (Ỹ_i - θ̂T̃_i)T̃_i / mean(T̃²) and Var(θ̂) = c·Σ_g(Σ_{i∈g} ψ_i)² / N² with the CR1 factor
    c = G/(G-1)·(N-1)/(N-1-K_fe), where K_fe counts fixed effects absorbed in-sample by a
    within transformation and not nested in the clusters (``n_absorbed_cluster``), as in the
    TWFE baseline. Without clusters (or with fewer than ``min_clusters``) each row is its own
    cluster: HC1 with c = N/(N-1-K_fe), K_fe = ``n_absorbed_hc``.
    """
    y_resid = np.asarray(y_resid, dtype=float)
    t_resid = np.asarray(t_resid, dtype=float)
    sxx = float(t_resid @ t_resid)
    if not sxx > 0:
        raise ValueError("residualised log price has no variation")
    theta = float(t_resid @ y_resid) / sxx
    cov = sandwich_vcov(
        np.array([[sxx]]),
        ((y_resid - theta * t_resid) * t_resid)[:, None],
        cluster_codes=cluster_codes,
        k_cluster=1 + n_absorbed_cluster,
        k_hc=1 + n_absorbed_hc,
        min_clusters=min_clusters,
    )
    return PartiallyLinearFit(theta, float(np.sqrt(cov.vcov[0, 0])), cov.df, cov.kind,
                              cov.n_clusters, cov.warnings)


def median_aggregate(thetas: Sequence[float], std_errors: Sequence[float]) -> tuple[float, float]:
    """Median over cross-fitting repeats: θ̃ = med θ_s, σ̃² = med(σ_s² + (θ_s - θ̃)²)."""
    thetas_arr = np.asarray(thetas, dtype=float)
    ses = np.asarray(std_errors, dtype=float)
    theta = float(np.median(thetas_arr))
    return theta, float(np.sqrt(np.median(ses**2 + (thetas_arr - theta) ** 2)))


# --------------------------------------------------------------------------- estimator


@dataclass(frozen=True)
class _Repeat:
    theta: float
    std_error: float
    r2_y: float
    r2_t: float
    residual_share: float
    fold_sizes: list[int]
    fold_ranges: list[tuple[str, str]]
    fit: PartiallyLinearFit


@dataclass(frozen=True)
class _DmlData:
    y: np.ndarray
    t: np.ndarray
    features: np.ndarray
    days: np.ndarray
    product_codes: np.ndarray
    cluster_codes: np.ndarray | None
    raw_price_var: float
    time_fe: str | None
    n_singletons: int
    n_absorbed_cluster: int
    n_absorbed_hc: int
    warnings: tuple[str, ...]


def resolve_time_fe(time_fe: str | None, fold_scheme: str, n_products: int) -> str | None:
    """``"auto"`` means day effects under time-blocked folds with 2+ products, else none."""
    if time_fe == "auto":
        return "day" if fold_scheme == "time_blocked" and n_products >= 2 else None
    if time_fe not in TIME_FE_CHOICES:
        raise ValueError(f"time_fe must be 'auto', 'day' or None, got {time_fe!r}")
    if time_fe == "day" and fold_scheme == "grouped":
        raise ValueError("time_fe='day' would mix held-out and training products in the day means; "
                         "use time_fe=None with fold_scheme='grouped'")
    return time_fe


def _prepare_dml_data(
    sample: EstimationSample, controls: Sequence[str], fold_scheme: str, time_fe: str | None,
    cluster: str | None,
) -> _DmlData:
    """Features and outcomes after the leak-free fixed-effect transformation of the fold scheme.

    time_blocked + day effects: Y, T and X are demeaned within calendar day. Every day lies
    entirely inside one fold's training, held-out or embargo set, so no held-out row enters a
    training row's day mean. grouped: Y, T and X are demeaned within product, and each product
    lies entirely inside one fold. Rows alone in their day/product carry no information and are
    dropped.
    """
    frame = sample.frame
    product_codes = factorize(frame[schema.PRODUCT_ID])[0]
    time_fe = resolve_time_fe(time_fe, fold_scheme, int(product_codes.max()) + 1)
    days = day_numbers(frame[schema.DATE])
    if fold_scheme == "grouped":
        absorbed = product_codes
    elif time_fe == "day":
        absorbed = factorize(days)[0]
    else:
        absorbed = None
    keep, n_singletons = np.ones(len(frame), dtype=bool), 0
    if absorbed is not None:
        keep, n_singletons, _ = prune_groups([absorbed])
        if not keep.any():
            raise ValueError("every row is alone in its day/product; nothing left to estimate from")
    frame = frame.iloc[np.flatnonzero(keep)]
    seasonal = time_fe is None
    features = build_nuisance_features(
        frame, controls, product_fe=fold_scheme == "time_blocked",
        fourier_annual=3 if seasonal else 0, fourier_weekly=2 if seasonal else 0,
    ).to_numpy()
    if not np.all(np.isfinite(features)):
        raise ValueError("nuisance features contain non-finite values")
    messages: tuple[str, ...] = ()
    span = int(days[keep].max() - days[keep].min()) + 1
    if seasonal and fold_scheme == "time_blocked" and span < 2 * 365:
        messages = (
            f"time_fe=None with time-blocked folds over {span} days: each held-out block's seasonal "
            "level is extrapolated from other blocks (annual Fourier terms), which is unreliable "
            "with less than two years of data; prefer time_fe='day' (needs 2+ products).",
        )
    y, t = np.log(sample.units[keep]), sample.log_price[keep]
    cluster_codes = factorize(frame[cluster])[0] if cluster is not None else None
    n_absorbed_cluster = n_absorbed_hc = 0
    if absorbed is not None:
        groups = [factorize(absorbed[keep])[0]]
        stacked = demean(np.column_stack([y, t, features]), groups).values
        y, t, features = stacked[:, 0], stacked[:, 1], stacked[:, 2:]
        n_absorbed_cluster, n_absorbed_hc = absorbed_dof(groups, cluster_codes)
    return _DmlData(
        y=y,
        t=t,
        features=features,
        days=days[keep],
        product_codes=factorize(product_codes[keep])[0],
        cluster_codes=cluster_codes,
        raw_price_var=float(np.var(sample.log_price[keep])),
        time_fe=time_fe,
        n_singletons=n_singletons,
        n_absorbed_cluster=n_absorbed_cluster,
        n_absorbed_hc=n_absorbed_hc,
        warnings=messages,
    )


def estimate_dml(
    panel: pd.DataFrame,
    controls: Sequence[str] = (),
    *,
    learner_y: LearnerSpec = "ridge",
    learner_t: LearnerSpec = "ridge",
    n_folds: int = 5,
    n_repeats: int = 3,
    fold_scheme: str = "time_blocked",
    embargo_days: int = 7,
    cluster: str | None = schema.PRODUCT_ID,
    mask: MaskLike = None,
    rng: np.random.Generator | None = None,
    time_fe: str | None = "auto",
) -> ElasticityEstimate:
    """Cross-fitted partially linear DML elasticity of ln Q on ln P (identification tier 5).

    Parameters
    ----------
    controls:
        Columns entering the nuisance features X next to product dummies (time_blocked) and,
        without day effects, annual/weekly Fourier seasonality (see ``build_nuisance_features``);
        rows with missing controls are dropped. Do not pass instruments such as a cost shifter:
        partialling them out of ln P removes exogenous variation.
    learner_y, learner_t:
        Learner name (``"ridge"``, ``"hgb"``, ``"rf"``), class, instance (cloned per fit) or
        zero-argument factory for E[ln Q | X] and E[ln P | X].
    n_folds, n_repeats:
        Cross-fitting folds per repeat and number of independent repeats (median-combined).
    fold_scheme:
        ``"time_blocked"`` (contiguous calendar blocks, boundaries jittered per repeat) or
        ``"grouped"`` (whole products held out, with within-product demeaning).
    embargo_days:
        Training rows within this many days of a held-out block are excluded (time_blocked only).
    cluster:
        Column for cluster-robust variance of the score (None: observation level). Fewer than 10
        clusters fall back to observation level with a warning; intervals use t with G-1 df.
    mask:
        Boolean row filter (True keeps), e.g. to drop stockout/censored rows.
    rng:
        Randomness for fold jitter/assignment; defaults to ``np.random.default_rng(0)``.
    time_fe:
        ``"day"`` absorbs day fixed effects by demeaning Y, T and X within calendar day
        (time_blocked only: a held-out block cannot learn its own seasonal level or common
        shocks from other blocks, and the misfit would load on θ̂). None relies on Fourier
        seasonality. ``"auto"`` picks ``"day"`` for time_blocked folds with 2+ products.

    Diagnostics include ``n_obs``, the sample counts of ``prepare_sample``, ``n_clusters``,
    ``se_type``, ``df``, ``time_fe``, ``within_transform``, ``n_singletons_dropped``,
    ``absorbed_dof``, ``repeat_estimates``/``repeat_std_errors``, out-of-fold ``nuisance_r2_y``
    and ``nuisance_r2_t`` (medians over repeats, measured on the transformed Y and T; per-repeat
    lists in ``repeat_r2_*``), ``residual_price_variance_share`` = Var(T̃)/Var(ln P),
    ``fold_sizes``, ``fold_date_ranges``, learner labels and ``warnings``.
    """
    if fold_scheme not in FOLD_SCHEMES:
        raise ValueError(f"fold_scheme must be one of {FOLD_SCHEMES}, got {fold_scheme!r}")
    if n_folds < 2 or n_repeats < 1:
        raise ValueError("need n_folds >= 2 and n_repeats >= 1")
    rng = np.random.default_rng(0) if rng is None else rng
    controls = tuple(controls)
    make_y, make_t = learner_factory(learner_y), learner_factory(learner_t)
    sample = prepare_sample(panel, columns=controls, mask=mask, cluster=cluster)
    data = _prepare_dml_data(sample, controls, fold_scheme, time_fe, cluster)
    y, t, features, days = data.y, data.t, data.features, data.days

    repeats = []
    for _ in range(n_repeats):
        if fold_scheme == "grouped":
            folds = grouped_folds(data.product_codes, n_folds, rng)
        else:
            folds = time_blocked_folds(days, n_folds, embargo_days, rng)
        y_resid = y - cross_fit_predict(features, y, folds, make_y)
        t_resid = t - cross_fit_predict(features, t, folds, make_t)
        fit = partialling_out(y_resid, t_resid, data.cluster_codes,
                              n_absorbed_cluster=data.n_absorbed_cluster,
                              n_absorbed_hc=data.n_absorbed_hc)
        repeats.append(
            _Repeat(
                theta=fit.theta,
                std_error=fit.std_error,
                r2_y=_r2(y, y_resid),
                r2_t=_r2(t, t_resid),
                residual_share=float(np.var(t_resid)) / data.raw_price_var if data.raw_price_var > 0 else 0.0,
                fold_sizes=[len(test) for _, test in folds],
                fold_ranges=[_date_range(days[test]) for _, test in folds]
                if fold_scheme == "time_blocked" else [],
                fit=fit,
            )
        )

    theta, std_error = median_aggregate([r.theta for r in repeats], [r.std_error for r in repeats])
    first = repeats[0].fit
    share = float(np.median([r.residual_share for r in repeats]))
    messages = [*sample.warnings, *data.warnings, *first.warnings]
    if share < WEAK_VARIATION_SHARE:
        messages.append(
            f"weak identifying variation: residual price variance is {share:.1%} of Var(ln P) "
            f"(< {WEAK_VARIATION_SHARE:.0%}); the controls explain almost all price movement"
        )
    diagnostics: dict[str, object] = {
        "method": "dml_plr",
        "n_obs": len(y),
        **sample.counts,
        "n_clusters": first.n_clusters,
        "se_type": first.variance_type,
        "df": first.df,
        "fold_scheme": fold_scheme,
        "time_fe": data.time_fe,
        "within_transform": "product" if fold_scheme == "grouped" else data.time_fe,
        "n_singletons_dropped": data.n_singletons,
        "absorbed_dof": data.n_absorbed_hc,
        "n_folds": n_folds,
        "n_repeats": n_repeats,
        "embargo_days": embargo_days if fold_scheme == "time_blocked" else None,
        "learner_y": learner_label(learner_y),
        "learner_t": learner_label(learner_t),
        "n_features": features.shape[1],
        "repeat_estimates": [r.theta for r in repeats],
        "repeat_std_errors": [r.std_error for r in repeats],
        "nuisance_r2_y": float(np.median([r.r2_y for r in repeats])),
        "nuisance_r2_t": float(np.median([r.r2_t for r in repeats])),
        "repeat_r2_y": [r.r2_y for r in repeats],
        "repeat_r2_t": [r.r2_t for r in repeats],
        "residual_price_variance_share": share,
        "fold_sizes": [r.fold_sizes for r in repeats],
        "fold_date_ranges": [r.fold_ranges for r in repeats],
        "warnings": messages,
    }
    emit_warnings(messages)
    return ElasticityEstimate(
        value=theta,
        std_error=std_error,
        intervals=t_intervals(theta, std_error, first.df),
        method="dml_plr",
        identification_tier=TIER_DML,
        n_obs=len(y),
        n_products=int(data.product_codes.max()) + 1,
        diagnostics=diagnostics,
    )


def benchmark_nuisance(
    panel: pd.DataFrame,
    controls: Sequence[str] = (),
    *,
    names: Sequence[str] = ("ridge",),
    n_folds: int = 5,
    fold_scheme: str = "time_blocked",
    embargo_days: int = 7,
    time_fe: str | None = "auto",
    mask: MaskLike = None,
    rng: np.random.Generator | None = None,
) -> dict[str, dict[str, float]]:
    """Out-of-fold RMSE of candidate learners for both DML nuisances (spec §17).

    Uses exactly the transformed outcomes, features and fold construction of ``estimate_dml``
    (one repeat; the same ``rng`` state reproduces its first repeat's folds), so the learner
    choice is benchmarked on the problem the estimator actually solves. Returns
    ``{"y": {name: rmse}, "t": {name: rmse}}``.
    """
    if fold_scheme not in FOLD_SCHEMES:
        raise ValueError(f"fold_scheme must be one of {FOLD_SCHEMES}, got {fold_scheme!r}")
    rng = np.random.default_rng(0) if rng is None else rng
    controls = tuple(controls)
    sample = prepare_sample(panel, columns=controls, mask=mask)
    data = _prepare_dml_data(sample, controls, fold_scheme, time_fe, None)
    if fold_scheme == "grouped":
        folds = grouped_folds(data.product_codes, n_folds, rng)
    else:
        folds = time_blocked_folds(data.days, n_folds, embargo_days, rng)
    return {
        "y": benchmark_learners(data.features, data.y, folds, names),
        "t": benchmark_learners(data.features, data.t, folds, names),
    }


def _r2(target: np.ndarray, resid: np.ndarray) -> float:
    total = float(np.sum((target - target.mean()) ** 2))
    return 1.0 - float(resid @ resid) / total if total > 0 else float("nan")


def _date_range(days: np.ndarray) -> tuple[str, str]:
    first, last = np.array([days.min(), days.max()]).astype("datetime64[D]")
    return str(first), str(last)


# --------------------------------------------------------------------------- comparison


def compare_estimates(a: ElasticityEstimate, b: ElasticityEstimate) -> dict[str, object]:
    """Contrast two elasticity estimates (e.g. DML against TWFE, spec Phase 4).

    Returns the difference a - b, whether the 95% intervals overlap, and a Hausman-style
    z = (θ_a - θ_b) / sqrt(se_a² + se_b²) with its two-sided normal p-value. The z statistic
    ignores the covariance between the estimates: both are fitted on the same data and are
    usually positively correlated, so the true Var(θ_a - θ_b) is smaller than se_a² + se_b² and
    |z| is conservative (too small).
    """
    difference = a.value - b.value
    se_difference = float(np.hypot(a.std_error, b.std_error))
    if se_difference > 0:
        z = difference / se_difference
    else:
        z = 0.0 if difference == 0 else float(np.copysign(np.inf, difference))
    ia, ib = a.interval(0.95), b.interval(0.95)
    return {
        "method_a": a.method,
        "method_b": b.method,
        "difference": float(difference),
        "std_error_difference": se_difference,
        "hausman_z": float(z),
        "p_value": float(2.0 * stats.norm.sf(abs(z))),
        "intervals_overlap_95": bool(ia.lower <= ib.upper and ib.lower <= ia.upper),
        "note": "z treats the estimates as independent and ignores their covariance; with "
                "positively correlated estimates from the same data it is conservative.",
    }
