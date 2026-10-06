"""Shared inference and sample-preparation helpers for LAEE elasticity estimators.

Covers robust sandwich variances (cluster-robust CR1 with the Stata/reghdfe small-sample factor
G/(G-1)·(N-1)/(N-K), HC1 fallback), t intervals with G-1 degrees of freedom (Cameron & Miller
2015, "A Practitioner's Guide to Cluster-Robust Inference"), the pairs cluster bootstrap
(Cameron, Gelbach & Miller 2008), and the row filtering every estimator shares (masks, invalid
prices, non-positive units, stockout flags; spec §63-64).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from laee import schema
from laee.types import ElasticityEstimate, Interval

DEFAULT_LEVELS: tuple[float, ...] = (0.8, 0.95)
MIN_CLUSTERS = 10
UNIT_RULES = ("positive", "nonnegative")

MaskLike = np.ndarray | pd.Series | Sequence[bool] | None


class EstimationWarning(UserWarning):
    """An estimate was produced under conditions that weaken it (few clusters, censoring, ...)."""


def emit_warnings(messages: Sequence[str], stacklevel: int = 3) -> None:
    for message in messages:
        warnings.warn(message, EstimationWarning, stacklevel=stacklevel)


# --------------------------------------------------------------------------- sample preparation


@dataclass(frozen=True)
class EstimationSample:
    """Rows of a panel that survive filtering, with the shared log-price and units arrays."""

    frame: pd.DataFrame
    positions: np.ndarray
    log_price: np.ndarray
    units: np.ndarray
    counts: dict[str, int]
    warnings: tuple[str, ...] = field(default_factory=tuple)


def resolve_mask(panel: pd.DataFrame, mask: MaskLike) -> np.ndarray:
    """Boolean row-inclusion array aligned with ``panel`` (True keeps the row)."""
    if mask is None:
        return np.ones(len(panel), dtype=bool)
    if isinstance(mask, pd.Series):
        if not mask.index.equals(panel.index):
            raise ValueError("mask index must equal the panel index")
        if mask.isna().any():
            raise ValueError("mask contains missing values")
        values = mask.to_numpy(dtype=bool) if pd.api.types.is_bool_dtype(mask) else mask.to_numpy()
    else:
        values = np.asarray(mask)
    if values.shape != (len(panel),):
        raise ValueError(f"mask must have shape ({len(panel)},), got {values.shape}")
    if values.dtype == np.bool_:
        return values.copy()
    if values.dtype.kind not in "biuf" or not np.isin(values, (0, 1)).all():
        raise ValueError("mask must be boolean")
    return values.astype(bool)


def day_numbers(dates: pd.Series | pd.Index | np.ndarray) -> np.ndarray:
    """Integer calendar days since 1970-01-01 (time of day and time zone are dropped)."""
    parsed = pd.to_datetime(dates if isinstance(dates, pd.Series) else pd.Series(dates))
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        parsed = parsed.dt.tz_localize(None)
    if parsed.isna().any():
        raise ValueError("dates contain missing values")
    return parsed.to_numpy().astype("datetime64[D]").astype(np.int64)


def factorize(values: pd.Series | np.ndarray) -> tuple[np.ndarray, int]:
    """Dense integer codes 0..G-1 for a grouping variable; missing identifiers are rejected."""
    codes, uniques = pd.factorize(values, sort=True)
    if (codes < 0).any():
        raise ValueError("grouping identifiers contain missing values")
    return codes.astype(np.int64), len(uniques)


def control_matrix(frame: pd.DataFrame, controls: Sequence[str]) -> np.ndarray:
    """Numeric (or boolean) control columns as a float matrix of shape (n, len(controls))."""
    if not controls:
        return np.empty((len(frame), 0))
    columns = []
    for name in controls:
        column = frame[name]
        if not (pd.api.types.is_bool_dtype(column) or pd.api.types.is_numeric_dtype(column)):
            raise TypeError(f"control {name!r} must be numeric or boolean, got {column.dtype}")
        columns.append(column.to_numpy(dtype=float, na_value=np.nan))
    return np.column_stack(columns)


def prepare_sample(
    panel: pd.DataFrame,
    *,
    columns: Sequence[str] = (),
    mask: MaskLike = None,
    units_rule: str = "positive",
    cluster: str | None = None,
) -> EstimationSample:
    """Filter a sales panel down to rows usable by an own-price elasticity estimator.

    Rows are dropped when excluded by ``mask``, when price is missing or non-positive, when units,
    dates, product ids, the cluster id or any of ``columns`` are missing, and, under
    ``units_rule="positive"`` (log outcomes), when units are zero or negative (``"nonnegative"``
    keeps zeros for Poisson models). Warns when stockout/censored rows remain in the sample.
    """
    if units_rule not in UNIT_RULES:
        raise ValueError(f"units_rule must be one of {UNIT_RULES}")
    required = [schema.PRODUCT_ID, schema.DATE, schema.PRICE, schema.UNITS, *columns]
    if cluster is not None:
        required.append(cluster)
    missing = [name for name in dict.fromkeys(required) if name not in panel.columns]
    if missing:
        raise KeyError(f"panel is missing columns {missing}")

    include = resolve_mask(panel, mask)
    price = pd.to_numeric(panel[schema.PRICE], errors="coerce").to_numpy(dtype=float)
    units = pd.to_numeric(panel[schema.UNITS], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(price) & (price > 0) & np.isfinite(units)
    valid &= panel[schema.PRODUCT_ID].notna().to_numpy()
    valid &= pd.to_datetime(panel[schema.DATE], errors="coerce").notna().to_numpy()
    for name in dict.fromkeys([*columns, *([cluster] if cluster else [])]):
        valid &= panel[name].notna().to_numpy()
    with np.errstate(invalid="ignore"):
        nonpositive = units <= 0 if units_rule == "positive" else units < 0
    keep = include & valid & ~nonpositive
    counts = {
        "n_input": len(panel),
        "n_masked_out": int((~include).sum()),
        "n_dropped_invalid": int((include & ~valid).sum()),
        "n_dropped_nonpositive": int((include & valid & nonpositive).sum()),
    }
    messages: list[str] = []
    flagged = np.zeros(len(panel), dtype=bool)
    for flag in (schema.STOCKOUT_FLAG, schema.CENSORED_FLAG):
        if flag in panel.columns:
            flagged |= panel[flag].fillna(False).to_numpy(dtype=bool)
    counts["n_stockout_rows_included"] = int((keep & flagged).sum())
    if counts["n_stockout_rows_included"]:
        messages.append(
            f"{counts['n_stockout_rows_included']} rows flagged {schema.STOCKOUT_FLAG}/"
            f"{schema.CENSORED_FLAG} are in the estimation sample; their units understate demand "
            f"(spec §63-64). Pass mask=~(panel['{schema.STOCKOUT_FLAG}'] | "
            f"panel['{schema.CENSORED_FLAG}']) to exclude them."
        )
    n_candidates = int((include & valid).sum())
    if units_rule == "positive" and n_candidates and counts["n_dropped_nonpositive"] > 0.05 * n_candidates:
        messages.append(
            f"{counts['n_dropped_nonpositive']} of {n_candidates} rows have zero or negative units and "
            "were dropped; a log-outcome model on the remainder is selected on the outcome. "
            "Consider method='ppml'."
        )
    positions = np.flatnonzero(keep)
    if positions.size == 0:
        raise ValueError("no usable rows after filtering")
    return EstimationSample(
        frame=panel.iloc[positions],
        positions=positions,
        log_price=np.log(price[positions]),
        units=units[positions],
        counts=counts,
        warnings=tuple(messages),
    )


# --------------------------------------------------------------------------- robust variances


@dataclass(frozen=True)
class CovarianceResult:
    """A sandwich covariance matrix and how it was computed."""

    vcov: np.ndarray
    kind: str
    df: float
    n_clusters: int | None
    warnings: tuple[str, ...] = ()


def cr1_factor(n_obs: int, n_params: int, n_clusters: int) -> float:
    """Small-sample factor G/(G-1)·(N-1)/(N-K) of the CR1 cluster-robust variance."""
    if n_clusters < 2 or n_obs <= n_params:
        raise ValueError("CR1 needs at least 2 clusters and more observations than parameters")
    return n_clusters / (n_clusters - 1) * (n_obs - 1) / (n_obs - n_params)


def hc1_factor(n_obs: int, n_params: int) -> float:
    """Small-sample factor N/(N-K) of the HC1 heteroskedasticity-robust variance."""
    if n_obs <= n_params:
        raise ValueError("HC1 needs more observations than parameters")
    return n_obs / (n_obs - n_params)


def cluster_sums(scores: np.ndarray, cluster_codes: np.ndarray, n_clusters: int) -> np.ndarray:
    """Per-cluster sums of score rows, shape (G, k)."""
    scores = np.asarray(scores, dtype=float).reshape(len(cluster_codes), -1)
    return np.column_stack([
        np.bincount(cluster_codes, weights=scores[:, j], minlength=n_clusters)
        for j in range(scores.shape[1])
    ])


def sandwich_vcov(
    bread: np.ndarray,
    scores: np.ndarray,
    *,
    cluster_codes: np.ndarray | None = None,
    k_cluster: int,
    k_hc: int,
    min_clusters: int = MIN_CLUSTERS,
) -> CovarianceResult:
    """Robust covariance A⁻¹ B A⁻¹ of an M-estimator with Hessian ``bread`` and per-row ``scores``.

    With ``cluster_codes`` and at least ``min_clusters`` clusters the meat sums scores within
    clusters (CR1, ``df = G - 1``); otherwise it is the HC1 outer product (``df = N - k_hc``) and,
    when clusters were requested, a warning explains the fallback. ``k_cluster``/``k_hc`` are the
    parameter counts used in the small-sample factors (fixed effects nested in the clusters are
    not counted for CR1).
    """
    if min_clusters < 2:
        raise ValueError("min_clusters must be at least 2")
    scores = np.asarray(scores, dtype=float)
    scores = scores.reshape(scores.shape[0], -1)
    n_obs = scores.shape[0]
    bread_inv = np.linalg.inv(np.asarray(bread, dtype=float).reshape(scores.shape[1], scores.shape[1]))
    messages: tuple[str, ...] = ()
    n_clusters: int | None = None
    if cluster_codes is not None:
        codes, n_clusters = factorize(np.asarray(cluster_codes))
        if n_clusters >= min_clusters:
            sums = cluster_sums(scores, codes, n_clusters)
            vcov = cr1_factor(n_obs, k_cluster, n_clusters) * bread_inv @ (sums.T @ sums) @ bread_inv
            return CovarianceResult(_symmetrize(vcov), "CR1", float(n_clusters - 1), n_clusters)
        messages = (
            f"only {n_clusters} clusters (< {min_clusters}): cluster-robust inference is unreliable, "
            "falling back to HC1, which ignores within-cluster (serial) correlation; consider "
            "cluster_bootstrap_ci or more clusters.",
        )
    vcov = hc1_factor(n_obs, k_hc) * bread_inv @ (scores.T @ scores) @ bread_inv
    return CovarianceResult(_symmetrize(vcov), "HC1", float(n_obs - k_hc), n_clusters, messages)


def ols_vcov(
    X: np.ndarray,
    resid: np.ndarray,
    *,
    cluster_codes: np.ndarray | None = None,
    n_absorbed_cluster: int = 0,
    n_absorbed_hc: int = 0,
    min_clusters: int = MIN_CLUSTERS,
) -> CovarianceResult:
    """CR1/HC1 covariance of OLS coefficients on (possibly FE-demeaned) regressors ``X``.

    ``n_absorbed_*`` add the degrees of freedom of absorbed fixed effects to K.
    """
    X = np.asarray(X, dtype=float).reshape(len(resid), -1)
    n_params = X.shape[1]
    return sandwich_vcov(
        X.T @ X,
        X * np.asarray(resid, dtype=float)[:, None],
        cluster_codes=cluster_codes,
        k_cluster=n_params + n_absorbed_cluster,
        k_hc=n_params + n_absorbed_hc,
        min_clusters=min_clusters,
    )


def t_intervals(
    value: float, std_error: float, df: float | None, levels: Sequence[float] = DEFAULT_LEVELS
) -> list[Interval]:
    """Symmetric t intervals (normal when ``df`` is None or infinite) at each level."""
    intervals = []
    for level in levels:
        if not 0 < level < 1:
            raise ValueError(f"interval level must lie in (0, 1), got {level}")
        tail = 0.5 + level / 2
        if df is None or not np.isfinite(df):
            quantile = float(stats.norm.ppf(tail))
        else:
            quantile = float(stats.t.ppf(tail, df))
        intervals.append(Interval(lower=value - quantile * std_error, upper=value + quantile * std_error,
                                  level=float(level)))
    return intervals


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


# --------------------------------------------------------------------------- cluster bootstrap


@dataclass(frozen=True)
class BootstrapResult:
    """Percentile intervals and draws from a pairs cluster bootstrap."""

    intervals: list[Interval]
    draws: np.ndarray
    std_error: float
    n_failed: int

    def interval(self, level: float) -> Interval:
        for iv in self.intervals:
            if abs(iv.level - level) < 1e-9:
                return iv
        raise KeyError(f"no {level:.0%} interval")


BOOTSTRAP_ERRORS = (ValueError, ZeroDivisionError, FloatingPointError, np.linalg.LinAlgError)


def cluster_bootstrap_ci(
    estimator_fn: Callable[[pd.DataFrame], ElasticityEstimate | float],
    panel: pd.DataFrame,
    cluster_col: str,
    n_boot: int,
    rng: np.random.Generator,
    levels: Sequence[float] = DEFAULT_LEVELS,
) -> BootstrapResult:
    """Percentile intervals from resampling whole clusters with replacement.

    Each draw takes G clusters with replacement and relabels the k-th drawn copy ``"<id>#k"``
    in ``cluster_col`` (and in ``PRODUCT_ID`` when it differs from the cluster column), so a
    cluster drawn twice enters fixed effects, cluster-robust variances and grouped folds as two
    distinct units. Draws whose estimator raises a numerical/data error (e.g. a resample without
    price variation) are skipped and counted in ``n_failed``; more than half failing raises.
    Estimator warnings are silenced inside the loop.
    """
    if n_boot < 2:
        raise ValueError("n_boot must be at least 2")
    codes, n_clusters = factorize(panel[cluster_col])
    if n_clusters < 2:
        raise ValueError("cluster bootstrap needs at least 2 clusters")
    order = np.argsort(codes, kind="stable")
    bounds = np.concatenate([[0], np.cumsum(np.bincount(codes, minlength=n_clusters))])
    members = [order[bounds[g]:bounds[g + 1]] for g in range(n_clusters)]
    relabel = [cluster_col] + ([schema.PRODUCT_ID] if schema.PRODUCT_ID in panel.columns
                               and cluster_col != schema.PRODUCT_ID else [])
    draws: list[float] = []
    n_failed = 0
    for _ in range(n_boot):
        chosen = rng.integers(0, n_clusters, size=n_clusters)
        rows = np.concatenate([members[g] for g in chosen])
        copy_tag = np.repeat(np.arange(n_clusters), [len(members[g]) for g in chosen])
        sample = panel.iloc[rows].reset_index(drop=True)
        suffix = "#" + pd.Series(copy_tag).astype(str)
        sample = sample.assign(**{col: sample[col].astype(str) + suffix for col in relabel})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", EstimationWarning)
            try:
                result = estimator_fn(sample)
            except BOOTSTRAP_ERRORS:
                n_failed += 1
                continue
        value = result.value if isinstance(result, ElasticityEstimate) else float(result)
        if np.isfinite(value):
            draws.append(float(value))
        else:
            n_failed += 1
    if n_failed > n_boot / 2 or len(draws) < 2:
        raise RuntimeError(f"{n_failed} of {n_boot} bootstrap draws failed")
    values = np.asarray(draws)
    intervals = []
    for level in levels:
        if not 0 < level < 1:
            raise ValueError(f"interval level must lie in (0, 1), got {level}")
        lower, upper = np.quantile(values, [(1 - level) / 2, (1 + level) / 2])
        intervals.append(Interval(lower=float(lower), upper=float(upper), level=float(level)))
    return BootstrapResult(intervals, values, float(values.std(ddof=1)), n_failed)
