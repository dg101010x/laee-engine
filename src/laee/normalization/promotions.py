"""Promotion detection and regular-price estimation (spec §62, §74; spec §84 step 4).

Each series (product, plus business and location when present) is scanned forward in time.
Every decision for date d uses only rows dated before d, plus that day's own price for the
flags. One causal pass yields the regular price, the drop episodes and both promotion flags.

Regular price
-------------
``regular_price`` on date d is the median of the three most recent *regular* prices dated
in the trailing window ``[d - window_days, d)``. If the window holds none (slow movers),
it is the most recent regular price before the window; with no history yet it is NaN.
Regular prices are those of days with sales that are not imputed, not marked
promotional by the export (explicit flag or discount), and not inside an open drop
episode. This is a trailing calendar-time window with a robust statistic, chosen over a
trailing 75th percentile because the percentile lags every repricing by one to three
weeks. During that lag, promotions are measured against a stale price, which in
inflationary markets with frequent increases hides many of them. The median of three
adopts an increase after two observations, ignores a single erroneous price, and
excludes promotional prices by construction rather than by outvoting them. The result is
always a price that was actually charged, never an average; with two prices in the
window the newer one is used (spec §65).

Drop episodes
-------------
An episode starts on a day priced at least ``min_depth`` below ``regular_price``. Its
baseline is that regular price. It closes in one of three ways:

* *recovered*: a later day is priced at least ``(1 - revert_tolerance) * baseline``, i.e.
  back at, or above, the prior regular price;
* *settled*: the price holds at a shallower level for :data:`SETTLE_DAYS` consecutive
  observed days, i.e. no longer ``min_depth`` below the baseline but still more than
  ``revert_tolerance`` below it. The drop ended in a new, lower regular price, which
  joins the history. A single in-between day, such as the mixed daily average on the
  day a promotion ends, does not close the episode;
* *expired*: it has lasted more than ``max_duration_days`` without recovering. It was a
  permanent price change, so its prices join the regular-price history.

Only a recovered episode is a promotion.

Promotion flags
---------------
* ``promo_flag`` (retrospective) is set by an explicit flag, an explicit discount > 0, or
  a day inside a recovered episode. Whether a drop recovers depends on later days, so
  this flag looks up to ``max_duration_days`` ahead.
* ``promo_flag_realtime`` (past only) is set by an explicit flag, an explicit discount
  > 0, or a day on which an episode was open and had begun fewer than
  ``max_duration_days`` days earlier. It cannot know whether the drop will recover, so it
  also flags the start of a permanent price cut.

Which flag to use
-----------------
Elasticity estimation on history must use ``promo_flag``, which labels past days
correctly, as a control alongside ``regular_price`` and ``discount_pct``. Live prediction,
and any feature fed to a model at prediction time, must use ``promo_flag_realtime``:
``promo_flag`` would leak up to ``max_duration_days`` of future prices (spec §74).
``regular_price`` uses only rows dated before the day. ``discount_pct`` adds that day's
own price, which the business sets. Both therefore serve either purpose.

``discount_pct`` is the larger of the export's explicit discount and the depth of the
day's price below ``regular_price``, as a fraction (0.2 means 20% off). The export's own
promotion and discount columns are kept as ``promo_flag_explicit`` and
``discount_pct_explicit``, so running the detector again gives the same result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from laee.features.temporal import (
    DISCOUNT_PCT_EXPLICIT,
    IMPUTED_ZERO,
    PROMO_FLAG_EXPLICIT,
    PROMO_FLAG_REALTIME,
    bool_column,
    ensure_unique_days,
    float_column,
    group_positions,
    normalize_dates,
    require_columns,
    series_keys,
)
from laee.schema import DATE, DISCOUNT_PCT, PRICE, PROMO_FLAG, REGULAR_PRICE, UNITS

RECENT_REGULAR_PRICES = 3
SETTLE_DAYS = 2
_OPEN_END = np.iinfo(np.int64).max


@dataclass
class _Episode:
    start: int
    baseline: float
    days: list[int] = field(default_factory=list)  # non-explicit observed days and prices
    prices: list[float] = field(default_factory=list)
    shallow: int = 0  # consecutive observed days at a shallower, not recovered, level
    shallow_from: int = 0  # index into days/prices where that shallow run starts

    def add(self, day: int, price: float, explicit: bool, *, shallow: bool) -> None:
        if shallow:
            if self.shallow == 0:
                self.shallow_from = len(self.days)
            self.shallow += 1
        else:
            self.shallow = 0
        if not explicit:
            self.days.append(day)
            self.prices.append(price)


def detect_promotions(
    panel: pd.DataFrame,
    window_days: int = 28,
    min_depth: float = 0.10,
    max_duration_days: int = 21,
    revert_tolerance: float = 0.02,
) -> pd.DataFrame:
    """Add regular_price, discount_pct, promo_flag and promo_flag_realtime (see module doc).

    Also adds ``promo_flag_explicit`` and ``discount_pct_explicit``, which hold the export's
    own promotion information. Imputed zero-sales days and days without units are not
    observations; they get the flags of the episode covering their date. ``date`` is
    returned as ``datetime64[ns]``.
    """
    if window_days < 1 or max_duration_days < 1:
        raise ValueError("window_days and max_duration_days must be at least 1")
    if not 0 <= revert_tolerance < min_depth < 1:
        raise ValueError("need 0 <= revert_tolerance < min_depth < 1")
    keys = series_keys(panel)
    require_columns(panel, [DATE, PRICE])
    out = panel.copy()
    out[DATE] = normalize_dates(out[DATE])
    ensure_unique_days(out, keys)
    frame = out.reset_index(drop=True)

    price = float_column(frame, PRICE)
    explicit_flag = bool_column(frame, PROMO_FLAG_EXPLICIT if PROMO_FLAG_EXPLICIT in frame else PROMO_FLAG)
    explicit_discount = float_column(
        frame, DISCOUNT_PCT_EXPLICIT if DISCOUNT_PCT_EXPLICIT in frame else DISCOUNT_PCT
    )
    explicit = explicit_flag | (np.nan_to_num(explicit_discount) > 0)
    observed = np.isfinite(price) & (price > 0) & ~bool_column(frame, IMPUTED_ZERO)
    if UNITS in frame.columns:
        observed &= float_column(frame, UNITS) > 0

    regular = np.full(len(frame), np.nan)
    retrospective = np.zeros(len(frame), dtype=bool)
    realtime = np.zeros(len(frame), dtype=bool)
    days = frame[DATE].to_numpy(dtype="datetime64[D]").astype(np.int64)
    for positions in group_positions(frame, keys):
        series_days = days[positions]
        regular[positions], episodes = _scan_series(
            series_days,
            price[positions],
            observed[positions],
            explicit[positions],
            window_days,
            min_depth,
            max_duration_days,
            revert_tolerance,
        )
        for start, end, recovered in episodes:
            inside = positions[(series_days >= start) & (series_days < end)]
            realtime[inside[days[inside] - start < max_duration_days]] = True
            if recovered:
                retrospective[inside] = True

    with np.errstate(divide="ignore", invalid="ignore"):
        depth = np.where(np.isfinite(price) & (price > 0), np.clip(1.0 - price / regular, 0.0, None), np.nan)
    out[REGULAR_PRICE] = regular
    out[DISCOUNT_PCT] = np.fmax(explicit_discount, depth)
    out[PROMO_FLAG] = explicit | retrospective
    out[PROMO_FLAG_REALTIME] = explicit | realtime
    out[PROMO_FLAG_EXPLICIT] = explicit_flag
    out[DISCOUNT_PCT_EXPLICIT] = explicit_discount
    return out


def _scan_series(
    days: np.ndarray,
    price: np.ndarray,
    observed: np.ndarray,
    explicit: np.ndarray,
    window: int,
    min_depth: float,
    max_duration: int,
    tolerance: float,
) -> tuple[np.ndarray, list[tuple[int, int, bool]]]:
    """One forward pass over a date-sorted series: regular prices and drop episodes.

    Returns the regular price of every row and (start day, end day exclusive, recovered)
    per episode, where the end is the day the episode's close became known. Row k's
    regular price is computed before row k's price is looked at.
    """
    regular = np.full(len(days), np.nan)
    episodes: list[tuple[int, int, bool]] = []
    history_days: list[int] = []
    history_prices: list[float] = []
    active: _Episode | None = None
    # Python scalars: this loop runs once per row, and numpy scalar access dominates otherwise.
    day_list, price_list = days.tolist(), price.tolist()
    observed_list, explicit_list = observed.tolist(), explicit.tolist()
    for k, day in enumerate(day_list):
        if active is not None and day - active.start > max_duration:
            # Still down after max_duration days: a permanent change, so the drop's prices
            # were regular after all. The decision depends on time only.
            episodes.append((active.start, day, False))
            history_days.extend(active.days)
            history_prices.extend(active.prices)
            active = None
        level = _recent_regular(history_days, history_prices, day, window)
        regular[k] = level
        if not observed_list[k]:
            continue
        current, is_explicit = price_list[k], explicit_list[k]
        if active is not None:
            if current >= active.baseline * (1.0 - tolerance):
                episodes.append((active.start, day, True))  # back at the prior regular price
                active = None
            else:
                shallow = current > active.baseline * (1.0 - min_depth)
                active.add(day, current, is_explicit, shallow=shallow)
                if active.shallow < SETTLE_DAYS:
                    continue
                # Settled at a shallower level: that run is the new regular price.
                episodes.append((active.start, day, False))
                history_days.extend(active.days[active.shallow_from:])
                history_prices.extend(active.prices[active.shallow_from:])
                active = None
                continue
        if not math.isnan(level) and current <= level * (1.0 - min_depth):
            active = _Episode(start=day, baseline=level)
            active.add(day, current, is_explicit, shallow=False)
        elif not is_explicit:
            history_days.append(day)
            history_prices.append(current)
    if active is not None:
        episodes.append((active.start, _OPEN_END, False))
    return regular, episodes


def _recent_regular(history_days: list[int], history_prices: list[float], day: int, window: int) -> float:
    """Median of the last three regular prices inside the window (see module doc)."""
    tail = slice(-RECENT_REGULAR_PRICES, None)
    recent = [p for d, p in zip(history_days[tail], history_prices[tail]) if d >= day - window]
    if not recent:
        return history_prices[-1] if history_prices else math.nan
    if len(recent) % 2:
        return sorted(recent)[len(recent) // 2]
    return recent[-1]  # even count: the newest, rather than an unobserved average (spec §65)
