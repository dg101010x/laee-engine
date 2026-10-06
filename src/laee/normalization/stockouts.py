"""Stockout and censoring detection (spec §63-64; spec §84 step 5).

Zero sales on a day with nothing in stock are not zero demand (spec §63). A day that
sold everything available is right-censored, because true demand was at least the
observed sales (spec §64). Neither kind of day may enter elasticity estimation as if it
measured demand. :func:`estimation_mask` returns the rows that are safe to use.

Inventory semantics
-------------------
With ``inventory_timing="start_of_day"`` (default), the inventory value for date d is
the stock available for sale that day. Then ``stockout_flag`` is ``stock <= 0`` and
``censored_flag`` is ``units >= stock``. With ``"end_of_day"`` the value is the closing
count, so available stock is closing stock plus units sold and ``censored_flag`` reduces
to closing stock <= 0. A count applies only to its own date; a weekly stock-take says
nothing about the other six days.

Without inventory
-----------------
For days without a stock count, ``suspected_stockout`` marks runs of consecutive
zero-sales days whose probability under a Poisson model with the series' trailing mean
daily rate is below ``zero_run_prob``, i.e. ``exp(-rate * run_length) < zero_run_prob``.
The rate is the units sold over the ``rate_window_days`` calendar days before the run,
divided by those days. Days without a row count as zero-sales days and known stockout
days are excluded. At least ``min_history_days`` of such exposure must precede the run.
Zero-sales days must be present as rows (see
:func:`laee.ingestion.fill_zero_sales_days`). The flag needs the whole run, including
days after a given row, so it is a retrospective estimation screen, not a live feature.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd

from laee.features.temporal import (
    DAYS_OUT_OF_STOCK,
    INVENTORY_AVAILABLE,
    PRICE_OUTLIER,
    SERIES_KEYS,
    SUSPECTED_STOCKOUT,
    bool_column,
    ensure_unique_days,
    float_column,
    group_positions,
    normalize_dates,
    require_columns,
    series_keys,
)
from laee.schema import CENSORED_FLAG, DATE, INVENTORY, PRICE, PRODUCT_ID, STOCKOUT_FLAG, UNITS

INVENTORY_TIMINGS = ("start_of_day", "end_of_day")


def detect_stockouts(
    panel: pd.DataFrame,
    inventory: pd.DataFrame | None = None,
    zero_run_prob: float = 0.01,
    *,
    inventory_timing: str = "start_of_day",
    rate_window_days: int = 28,
    min_history_days: int = 7,
) -> pd.DataFrame:
    """Add stockout and censoring flags (see module doc).

    ``inventory`` (e.g. from :func:`laee.ingestion.ingest_inventory`) is joined on
    product_id, date and whichever of business_id/location_id both frames have. Its counts
    take precedence over an ``inventory`` column already in the panel. Several counts for
    one product-day are reduced to the highest.

    Adds ``inventory``, ``inventory_available``, ``stockout_flag``, ``censored_flag``,
    ``days_out_of_stock`` (consecutive stockout days up to and including the row) and
    ``suspected_stockout``.
    """
    if not 0 < zero_run_prob < 1:
        raise ValueError("zero_run_prob must lie strictly between 0 and 1")
    if inventory_timing not in INVENTORY_TIMINGS:
        raise ValueError(f"inventory_timing must be one of {', '.join(INVENTORY_TIMINGS)}")
    if rate_window_days < 1 or min_history_days < 1:
        raise ValueError("rate_window_days and min_history_days must be at least 1")
    keys = series_keys(panel)
    require_columns(panel, [DATE, UNITS])
    out = panel.copy()
    out[DATE] = normalize_dates(out[DATE])
    ensure_unique_days(out, keys)
    frame = out.reset_index(drop=True)

    stock = float_column(frame, INVENTORY)
    if inventory is not None:
        counted = _merge_inventory(frame, inventory)
        if len(frame) and not np.isfinite(counted).any():
            warnings.warn(
                "inventory matched no panel row; check product ids, dates and "
                "business_id/location_id",
                UserWarning,
                stacklevel=2,
            )
        stock = np.where(np.isfinite(counted), counted, stock)
    units = float_column(frame, UNITS)
    available = stock if inventory_timing == "start_of_day" else stock + units
    known = np.isfinite(stock)
    stockout = known & (stock <= 0)
    with np.errstate(invalid="ignore"):
        censored = np.isfinite(available) & np.isfinite(units) & (units >= available)

    days = frame[DATE].to_numpy(dtype="datetime64[D]").astype(np.int64)
    days_out = np.zeros(len(frame), dtype=np.int64)
    suspected = np.zeros(len(frame), dtype=bool)
    for positions in group_positions(frame, keys):
        series_days = days[positions]
        days_out[positions] = _run_lengths(series_days, stockout[positions])
        suspected[positions] = _improbable_zero_runs(
            series_days,
            units[positions],
            known[positions],
            stockout[positions],
            zero_run_prob,
            rate_window_days,
            min_history_days,
        )

    out[INVENTORY] = stock
    out[INVENTORY_AVAILABLE] = available
    out[STOCKOUT_FLAG] = stockout
    out[CENSORED_FLAG] = censored
    out[DAYS_OUT_OF_STOCK] = days_out
    out[SUSPECTED_STOCKOUT] = suspected
    return out


def estimation_mask(panel: pd.DataFrame) -> pd.Series:
    """True for rows safe to use in elasticity estimation (spec §63-64).

    Excludes stockouts, censored days, suspected stockouts and price outliers (each only
    when its column exists), plus rows lacking a positive finite price or finite,
    non-negative units. Promotional days stay in, because they carry price variation;
    control for them with promo_flag and discount_pct (spec §62) rather than drop them.
    """
    mask = np.ones(len(panel), dtype=bool)
    for col in (STOCKOUT_FLAG, CENSORED_FLAG, SUSPECTED_STOCKOUT, PRICE_OUTLIER):
        mask &= ~bool_column(panel, col)
    if PRICE in panel.columns:
        price = float_column(panel, PRICE)
        mask &= np.isfinite(price) & (price > 0)
    if UNITS in panel.columns:
        units = float_column(panel, UNITS)
        mask &= np.isfinite(units) & (units >= 0)
    return pd.Series(mask, index=panel.index, name="estimation_mask")


def _merge_inventory(frame: pd.DataFrame, inventory: pd.DataFrame) -> np.ndarray:
    require_columns(inventory, [PRODUCT_ID, DATE, INVENTORY])
    join = [key for key in SERIES_KEYS if key in frame.columns and key in inventory.columns]
    counts = inventory[[*join, DATE, INVENTORY]].copy()
    counts[DATE] = normalize_dates(counts[DATE])
    counts[INVENTORY] = pd.to_numeric(counts[INVENTORY], errors="coerce")
    counts = counts.groupby([*join, DATE], as_index=False, dropna=False)[INVENTORY].max()
    merged = frame[[*join, DATE]].merge(counts, on=[*join, DATE], how="left", validate="many_to_one")
    return merged[INVENTORY].to_numpy(dtype="float64")


def _run_lengths(days: np.ndarray, flags: np.ndarray) -> np.ndarray:
    """Consecutive flagged calendar days ending at each row (0 where not flagged)."""
    lengths = np.zeros(len(days), dtype=np.int64)
    for k in range(len(days)):
        if not flags[k]:
            continue
        continues = k > 0 and flags[k - 1] and days[k] - days[k - 1] == 1
        lengths[k] = lengths[k - 1] + 1 if continues else 1
    return lengths


def _improbable_zero_runs(
    days: np.ndarray,
    units: np.ndarray,
    stock_known: np.ndarray,
    stockout: np.ndarray,
    zero_run_prob: float,
    window: int,
    min_history: int,
) -> np.ndarray:
    flags = np.zeros(len(days), dtype=bool)
    if len(days) == 0:
        return flags
    sold = np.where(stockout, 0.0, np.nan_to_num(units, nan=0.0))
    cum_sold = np.concatenate([[0.0], np.cumsum(sold)])
    cum_stockout = np.concatenate([[0], np.cumsum(stockout)])
    zero = (np.nan_to_num(units, nan=1.0) == 0) & ~stock_known
    k = 0
    while k < len(days):
        if not zero[k]:
            k += 1
            continue
        end = k + 1
        while end < len(days) and zero[end] and days[end] - days[end - 1] == 1:
            end += 1
        start = int(days[k])
        first = int(np.searchsorted(days, start - window, side="left"))
        # Exposure: calendar days in [start - window, start) since the series began,
        # minus days already known to be out of stock (their zero sales say nothing).
        exposure = start - max(start - window, int(days[0])) - int(cum_stockout[k] - cum_stockout[first])
        if exposure >= min_history:
            rate = (cum_sold[k] - cum_sold[first]) / exposure
            run_length = int(days[end - 1]) - start + 1
            if math.exp(-rate * run_length) < zero_run_prob:
                flags[k:end] = True
        k = end
    return flags
