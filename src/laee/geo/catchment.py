"""Customer catchment and catchment-weighted demographics (spec §11, §12).

The catchment model used throughout: a business's customers lie in uniformly random directions
from it, at distances that are exponentially distributed with rate lambda. `fit_distance_decay`
estimates that rate from observed customer distances. `catchment_overlap` turns it into the
customer-overlap term W_ij of the competitor influence score (spec §8).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import special

from .distance import FloatArray

DEFAULT_BANDS_KM: tuple[float, ...] = (0.5, 1.0, 2.0, 5.0)

# Beyond z = lambda d / 2 = 50 the overlap is below 1e-20, i.e. zero in double precision.
_MAX_OVERLAP_Z = 50.0


def _distances_and_weights(
    distances_km: ArrayLike, weights: ArrayLike | None
) -> tuple[FloatArray, FloatArray]:
    d = np.asarray(distances_km, dtype=np.float64).ravel()
    w = np.ones_like(d) if weights is None else np.asarray(weights, dtype=np.float64).ravel()
    if w.shape != d.shape:
        raise ValueError("weights must have one entry per distance")
    if np.any(~np.isfinite(w)) or np.any(w < 0):
        raise ValueError("weights must be finite and non-negative")
    keep = ~np.isnan(d)
    d, w = d[keep], w[keep]
    if np.any(d < 0) or np.any(np.isinf(d)):
        raise ValueError("distances must be finite and non-negative")
    if d.size == 0 or w.sum() <= 0:
        raise ValueError("no customer distances with positive weight")
    return d, w


def _band_label(lo: float, hi: float | None) -> str:
    return f"{lo:g}+km" if hi is None else f"{lo:g}-{hi:g}km"


def distance_band_shares(
    distances_km: ArrayLike,
    bands: Sequence[float] = DEFAULT_BANDS_KM,
    *,
    weights: ArrayLike | None = None,
) -> pd.Series:
    """Share of customers (or of `weights`, e.g. visits) in each distance band (spec §11).

    Bands are left-closed: the defaults give 0-0.5, 0.5-1, 1-2, 2-5 and 5+ km. NaN distances
    (customers without a location) are dropped.
    """
    edges = np.asarray(bands, dtype=np.float64)
    if edges.ndim != 1 or edges.size == 0 or np.any(edges <= 0) or np.any(np.diff(edges) <= 0):
        raise ValueError("bands must be positive and strictly increasing")
    d, w = _distances_and_weights(distances_km, weights)
    band = np.searchsorted(edges, d, side="right")
    shares = np.bincount(band, weights=w, minlength=edges.size + 1) / w.sum()
    lows = [0.0, *edges.tolist()]
    highs: list[float | None] = [*edges.tolist(), None]
    labels = [_band_label(lo, hi) for lo, hi in zip(lows, highs)]
    return pd.Series(shares, index=pd.Index(labels, name="distance_band"), name="share")


def fit_distance_decay(distances_km: ArrayLike, *, weights: ArrayLike | None = None) -> float:
    """Maximum-likelihood rate of an exponential customer-distance distribution.

    For d ~ Exponential(lambda) the MLE is 1 / mean(d), or sum(w) / sum(w d) with weights.
    """
    d, w = _distances_and_weights(distances_km, weights)
    weighted_total = float(np.dot(w, d))
    if weighted_total <= 0:
        raise ValueError("distances must have a positive mean to estimate a decay rate")
    return float(w.sum()) / weighted_total


def catchment_overlap(distance_km: ArrayLike, catchment_lambda: float) -> FloatArray:
    """Overlap coefficient of two identical catchments whose centres are `distance_km` apart.

    The overlap coefficient is the integral of min(f_i, f_j) over the plane, with f the customer
    density described in the module docstring. By symmetry it equals twice the share of i's
    customers who live closer to j than to i. Integrating over the exponential distance and the
    uniform bearing gives the closed form
        W(d) = (2/pi) Ki_1(z) = 1 - (2/pi) * integral_0^z K_0(t) dt,   with z = lambda d / 2,
    where K_0 is the modified Bessel function and Ki_1 the Bickley function. W is 1 at d = 0 and
    falls monotonically to 0. Infinite distance gives 0, and lambda = 0 (no decay) gives 1.
    """
    d = np.asarray(distance_km, dtype=np.float64)
    if np.any(np.isnan(d)) or np.any(d < 0):
        raise ValueError("distances must be non-negative (inf is allowed for unreachable)")
    if not (math.isfinite(catchment_lambda) and catchment_lambda >= 0):
        raise ValueError("catchment lambda must be a non-negative finite number")
    if catchment_lambda == 0.0:
        return np.ones_like(d)
    z = np.minimum(catchment_lambda * d / 2.0, _MAX_OVERLAP_Z)
    _, integral_k0 = special.iti0k0(z)
    overlap = 1.0 - (2.0 / math.pi) * np.asarray(integral_k0, dtype=np.float64)
    return np.asarray(np.clip(np.where(z >= _MAX_OVERLAP_Z, 0.0, overlap), 0.0, 1.0))


def weighted_demographics(values: pd.DataFrame, weights: ArrayLike | pd.Series) -> pd.Series:
    """Catchment-weighted demographics X_market = sum_k w_k X_k (spec §12).

    `values` has one row per area k (H3 cell, census tract) and one column per variable.
    `weights` are customer probabilities per area: a Series aligned on the index, or an
    array-like in row order. Weights are normalised. A variable missing (NaN) in some areas is
    averaged over the areas where it is present, with the weights renormalised; a variable that
    is missing everywhere, or present only where the weight is zero, gives NaN.
    """
    if isinstance(weights, pd.Series):
        if not weights.index.is_unique:
            raise ValueError("weights index must be unique")
        aligned = weights.reindex(values.index)
        if aligned.isna().any():
            raise ValueError("weights are missing for some rows of values")
        w = aligned.to_numpy(dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64).ravel()
        if w.shape != (len(values),):
            raise ValueError("weights must have one entry per row of values")
    if np.any(~np.isfinite(w)) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weights must be finite, non-negative and not all zero")
    x = values.to_numpy(dtype=np.float64)
    present = ~np.isnan(x)
    weight_present = (w[:, None] * present).sum(axis=0)
    weighted_sum = (w[:, None] * np.where(present, x, 0.0)).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(weight_present > 0, weighted_sum / weight_present, np.nan)
    return pd.Series(result, index=values.columns, name="catchment_weighted")
