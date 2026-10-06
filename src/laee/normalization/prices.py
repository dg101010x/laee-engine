"""Product-id and price normalization (spec §65; spec §84 step 3)."""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from laee.features.temporal import (
    IMPUTED_ZERO,
    PRICE_OUTLIER,
    PRICE_ROBUST_Z,
    bool_column,
    ensure_unique_days,
    float_column,
    group_positions,
    map_unique,
    normalize_dates,
    require_columns,
    series_keys,
)
from laee.schema import DATE, PRICE, PROMO_FLAG, REGULAR_PRICE

_DASHES = "‐‑‒–—―⁃−﹘﹣－"
_SINGLE_QUOTES = "‘’‚‛′ʼ`´"
_DOUBLE_QUOTES = "“”„‟″«»"
_ID_TRANSLATION = str.maketrans(
    {
        **dict.fromkeys(_DASHES, "-"),
        **dict.fromkeys(_SINGLE_QUOTES, "'"),
        **dict.fromkeys(_DOUBLE_QUOTES, '"'),
    }
)

_MAD_TO_SIGMA = 1.4826
_MEAN_AD_TO_SIGMA = 1.2533  # sqrt(pi / 2)
_MIN_SCORED_PRICES = 3

# Candidate price-grid spacings, ascending (spec §65: prices move in increments).
_PRICE_TICKS = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 25.0, 50.0, 100.0, 200.0, 250.0, 500.0, 1000.0)
_ON_GRID_SHARE = 0.9
_MAX_TICK_SHARE_OF_PRICE = 0.1

CHANGE_COLUMNS = ("old_price", "new_price", "rel_change")


@dataclass(frozen=True)
class PricePoints:
    """Observed price grid of one product (spec §65: use actual prices, not false precision).

    ``points`` are the sorted distinct prices on the detected grid, observed ``counts``
    times each. ``tick`` is the grid spacing: the largest round increment, at most 10% of
    the median price, on which at least 90% of the prices lie. ``min_step`` is the
    smallest gap between adjacent points, or ``tick`` when a single price was observed.
    It is the smallest price move worth treating as a real change.
    """

    points: np.ndarray
    counts: np.ndarray
    tick: float
    min_step: float


def normalize_product_ids(series: pd.Series) -> pd.Series:
    """Canonical product identifiers, so spelling variants of one product share an id.

    Applies Unicode NFKC, maps dash and quote variants to ``-``, ``'`` and ``"``, collapses
    whitespace runs, trims and casefolds. Missing or blank values become NaN. Integral
    floats (1001.0, which numeric SKU columns with blanks produce) become ``"1001"``.
    """
    return map_unique(series, _normalize_id)


def flag_price_outliers(panel: pd.DataFrame, z: float = 5.0, *, min_ratio: float = 3.0) -> pd.DataFrame:
    """Flag gross price errors with a robust z-score on log price within each series.

    ``z = (log p - median) / (1.4826 * MAD)`` per series (product, plus business and
    location when present). Prices are discrete (spec §65), so more than half the days
    often share one price and the MAD is 0; the scale then falls back to
    ``1.2533 * mean absolute deviation``. A row is flagged when ``|z| > z`` *and* its price
    is more than ``min_ratio`` times above or below the median. The second condition
    keeps genuine promotions and regular price changes, rarely beyond 3x, in the sample.
    It still catches typos, pack-versus-unit prices and separator misreads, which are
    10x or more. Non-positive prices are always flagged. Series with fewer than three
    usable prices are not scored. Imputed zero-sales days are neither scored nor flagged.

    Adds ``price_outlier`` (bool) and ``price_robust_z``. Both use the whole history, so
    they are retrospective screening labels, not live features.
    """
    if not z > 0:
        raise ValueError("z must be positive")
    if not min_ratio >= 1:
        raise ValueError("min_ratio must be at least 1")
    keys = series_keys(panel)
    require_columns(panel, [PRICE])
    out = panel.copy()
    price = float_column(out, PRICE)
    imputed = bool_column(out, IMPUTED_ZERO)
    usable = np.isfinite(price) & (price > 0) & ~imputed
    log_price = np.full(len(out), np.nan)
    log_price[usable] = np.log(price[usable])
    score = np.full(len(out), np.nan)
    deviation = np.full(len(out), np.nan)
    groups = out[keys].reset_index(drop=True).groupby(keys, sort=False, dropna=False).indices
    for positions in groups.values():
        values = log_price[positions]
        valid = np.isfinite(values)
        if valid.sum() < _MIN_SCORED_PRICES:
            continue
        dev = values - np.median(values[valid])
        scale = _MAD_TO_SIGMA * np.median(np.abs(dev[valid]))
        if scale == 0:
            scale = _MEAN_AD_TO_SIGMA * np.mean(np.abs(dev[valid]))
        score[positions] = np.where(valid, dev / scale if scale > 0 else 0.0, np.nan)
        deviation[positions] = dev
    with np.errstate(invalid="ignore"):
        flagged = (np.abs(score) > z) & (np.abs(deviation) > math.log(min_ratio))
    flagged |= np.isfinite(price) & (price <= 0) & ~imputed
    out[PRICE_ROBUST_Z] = score
    out[PRICE_OUTLIER] = flagged
    return out


def price_points(prices: Iterable[float] | pd.Series | np.ndarray) -> PricePoints:
    """Distinct observed prices on their grid, and the minimum meaningful price step.

    Pass transaction prices, or panel prices from single-price days. Daily averages of
    mixed prices fall between grid points; when they are under 10% of the values they
    are left out of ``points``.
    """
    values = pd.to_numeric(pd.Series(prices), errors="coerce").to_numpy(dtype="float64")
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        raise ValueError("price_points needs at least one positive price")
    cap = _MAX_TICK_SHARE_OF_PRICE * float(np.median(values))
    tick = _PRICE_TICKS[0]
    for candidate in _PRICE_TICKS[1:]:
        if candidate > cap:
            break
        if _on_grid(values, candidate).mean() >= _ON_GRID_SHARE:
            tick = candidate
    on_grid = _on_grid(values, tick)
    if on_grid.mean() < _ON_GRID_SHARE:
        on_grid[:] = True  # no grid at all: keep every price, rounded to the cent
    snapped = np.round(np.round(values[on_grid] / tick) * tick, 6)
    points, counts = np.unique(snapped, return_counts=True)
    min_step = float(np.min(np.diff(points))) if points.size > 1 else float(tick)
    return PricePoints(points=points, counts=counts, tick=float(tick), min_step=round(min_step, 6))


def detect_price_changes(panel: pd.DataFrame, min_rel_change: float = 0.01) -> pd.DataFrame:
    """Regular-price change events: series keys, ``date``, old_price, new_price, rel_change.

    The reference is ``regular_price`` when present (run ``detect_promotions`` first),
    otherwise ``price`` on days not flagged ``promo_flag``. An event is recorded when the
    reference moves at least ``min_rel_change`` away from the current level, so
    sub-threshold noise is ignored but slow drift is caught once it adds up (spec §65).
    Days flagged ``price_outlier`` and imputed zero-sales days are skipped.

    ``date`` is the first day the new price was charged. A trailing ``regular_price``
    confirms a change some days after it happens, so with that reference each event is
    dated back over the run of non-promotional days already at the new price. The
    table describes history and is not a live feature.
    """
    if not min_rel_change > 0:
        raise ValueError("min_rel_change must be positive")
    keys = series_keys(panel)
    require_columns(panel, [DATE, PRICE])
    frame = panel.reset_index(drop=True)
    frame[DATE] = normalize_dates(frame[DATE])
    ensure_unique_days(frame, keys)
    price = float_column(frame, PRICE)
    use_regular = REGULAR_PRICE in frame.columns
    reference = float_column(frame, REGULAR_PRICE) if use_regular else price
    skip = bool_column(frame, IMPUTED_ZERO) | bool_column(frame, PRICE_OUTLIER)
    promo = bool_column(frame, PROMO_FLAG)
    reference_ok = np.isfinite(reference) & (reference > 0) & ~skip
    if not use_regular:
        reference_ok &= ~promo
    walk_ok = np.isfinite(price) & (price > 0) & ~skip & ~promo
    dates = frame[DATE].to_numpy()

    records: list[tuple[object, ...]] = []
    for positions in group_positions(frame, keys):
        level = math.nan
        floor = -1  # position (within the series) the backdating walk may not pass
        for k, row in enumerate(positions):
            if not reference_ok[row]:
                continue
            if math.isnan(level):
                level, floor = float(reference[row]), k
                continue
            new = float(reference[row])
            rel = new / level - 1.0
            if abs(rel) < min_rel_change:
                continue
            start = k
            # Only regular_price lags the change; a raw price reference sees it on the day.
            for j in range(k - 1, floor if use_regular else k - 1, -1):
                earlier = positions[j]
                if not walk_ok[earlier]:
                    continue
                if abs(price[earlier] / new - 1.0) >= min_rel_change:
                    break
                start = j
            first = positions[start]
            records.append((*(frame[key].iat[first] for key in keys), dates[first], level, new, rel))
            level, floor = new, k
    events = pd.DataFrame(records, columns=[*keys, DATE, *CHANGE_COLUMNS])
    events[DATE] = pd.to_datetime(events[DATE]).astype("datetime64[ns]")
    for col in CHANGE_COLUMNS:
        events[col] = events[col].astype("float64")
    return events.sort_values([*keys, DATE], kind="stable").reset_index(drop=True)


def _normalize_id(value: object) -> object:
    if value is None or value is pd.NA:
        return np.nan
    if isinstance(value, (float, np.floating)):
        if math.isnan(value):
            return np.nan
        if float(value).is_integer():
            value = int(value)
    text = unicodedata.normalize("NFKC", str(value)).translate(_ID_TRANSLATION)
    text = " ".join(text.split()).casefold()
    return text if text else np.nan


def _on_grid(values: np.ndarray, tick: float) -> np.ndarray:
    quotient = values / tick
    return np.abs(quotient - np.round(quotient)) <= 1e-6
