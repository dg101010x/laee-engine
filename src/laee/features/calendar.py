"""Calendar features (spec §59): weekday, month position, payday window, seasonality, holidays.

Every feature is a deterministic function of the date, so it is known in advance and
safe for live prediction (see ``FEATURE_REGISTRY`` in :mod:`laee.features.temporal`).

The built-in holiday table holds only fixed-date public holidays (same month and day
every year) for KE, NG, GH, ZA, US and GB. It does not hold moveable feasts
(Easter, Eid, Diwali, Thanksgiving), substitute days off when a holiday falls on a
weekend, or one-off declared holidays. Callers supply those through ``holidays``.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from laee.features.temporal import normalize_dates, require_columns
from laee.schema import DATE

DAY_OF_WEEK = "day_of_week"
IS_WEEKEND = "is_weekend"
WEEK_OF_YEAR = "week_of_year"
MONTH = "month"
DAYS_TO_MONTH_END = "days_to_month_end"
DAYS_SINCE_MONTH_START = "days_since_month_start"
IS_PAYDAY_WINDOW = "is_payday_window"
IS_HOLIDAY = "is_holiday"
HOLIDAY_NAME = "holiday_name"

DEFAULT_PAYDAY_DAYS = (25, 26, 27, 28, 29, 30, 31, 1, 2)
_ANNUAL_PERIOD_DAYS = 365.25
_EPOCH = pd.Timestamp("1970-01-01")

# (month, day, name, first_year, last_year); None leaves the range open.
_FIXED_HOLIDAYS: dict[str, tuple[tuple[int, int, str, int | None, int | None], ...]] = {
    "KE": (
        (1, 1, "New Year's Day", None, None),
        (5, 1, "Labour Day", None, None),
        (6, 1, "Madaraka Day", None, None),
        # Reinstated in 2018 and observed since under several names (Moi Day, Huduma Day).
        (10, 10, "Mazingira Day", 2018, None),
        (10, 20, "Mashujaa Day", None, None),
        (12, 12, "Jamhuri Day", None, None),
        (12, 25, "Christmas Day", None, None),
        (12, 26, "Boxing Day", None, None),
    ),
    "NG": (
        (1, 1, "New Year's Day", None, None),
        (5, 1, "Workers' Day", None, None),
        (5, 29, "Democracy Day", 2000, 2018),
        (6, 12, "Democracy Day", 2019, None),
        (10, 1, "Independence Day", None, None),
        (12, 25, "Christmas Day", None, None),
        (12, 26, "Boxing Day", None, None),
    ),
    "GH": (
        (1, 1, "New Year's Day", None, None),
        (3, 6, "Independence Day", None, None),
        (5, 1, "May Day", None, None),
        (9, 21, "Founder's Day", 2009, 2018),
        (9, 21, "Kwame Nkrumah Memorial Day", 2019, None),
        (12, 25, "Christmas Day", None, None),
        (12, 26, "Boxing Day", None, None),
    ),
    "ZA": (
        (1, 1, "New Year's Day", None, None),
        (3, 21, "Human Rights Day", None, None),
        (4, 27, "Freedom Day", None, None),
        (5, 1, "Workers' Day", None, None),
        (6, 16, "Youth Day", None, None),
        (8, 9, "National Women's Day", None, None),
        (9, 24, "Heritage Day", None, None),
        (12, 16, "Day of Reconciliation", None, None),
        (12, 25, "Christmas Day", None, None),
        (12, 26, "Day of Goodwill", None, None),
    ),
    "US": (
        (1, 1, "New Year's Day", None, None),
        (6, 19, "Juneteenth National Independence Day", 2021, None),
        (7, 4, "Independence Day", None, None),
        (11, 11, "Veterans Day", None, None),
        (12, 25, "Christmas Day", None, None),
    ),
    "GB": (
        (1, 1, "New Year's Day", None, None),
        (12, 25, "Christmas Day", None, None),
        (12, 26, "Boxing Day", None, None),
    ),
}

SUPPORTED_COUNTRIES = tuple(sorted(_FIXED_HOLIDAYS))


def fourier_name(period: str, kind: str, order: int) -> str:
    """Column name of a Fourier term, e.g. ``fourier_annual_sin_1``."""
    return f"fourier_{period}_{kind}_{order}"


def builtin_holidays(country_code: str, years: Iterable[int]) -> dict[pd.Timestamp, str]:
    """Fixed-date public holidays of ``country_code`` falling in ``years``."""
    code = country_code.strip().upper()
    if code not in _FIXED_HOLIDAYS:
        raise ValueError(
            f"no built-in holidays for {country_code!r}; supported: {', '.join(SUPPORTED_COUNTRIES)}"
        )
    table: dict[pd.Timestamp, str] = {}
    for year in sorted(set(int(y) for y in years)):
        for month, day, name, first, last in _FIXED_HOLIDAYS[code]:
            if (first is None or year >= first) and (last is None or year <= last):
                table[pd.Timestamp(year, month, day)] = name
    return table


def add_calendar_features(
    panel: pd.DataFrame,
    *,
    country_code: str | None = None,
    holidays: Mapping[object, object] | None = None,
    payday_days: Sequence[int] = DEFAULT_PAYDAY_DAYS,
    fourier_annual: int = 3,
    fourier_weekly: int = 2,
) -> pd.DataFrame:
    """Add calendar columns derived from ``date``.

    Adds ``day_of_week`` (0 = Monday), ``is_weekend`` (Saturday/Sunday), ``week_of_year``
    (ISO), ``month``, ``days_to_month_end`` (0 on the last day), ``days_since_month_start``
    (0 on the 1st), ``is_payday_window`` (day of month in ``payday_days``), Fourier terms
    ``fourier_annual_{sin,cos}_k`` (period 365.25 days, k = 1..fourier_annual) and
    ``fourier_weekly_{sin,cos}_k`` (period 7 days, k = 1..fourier_weekly), plus
    ``is_holiday`` and ``holiday_name``.

    ``holidays`` adds dates to the built-in table of ``country_code``, as ``{date: name}``,
    ``{name: date}`` or ``{name: [dates]}``. Names falling on the same date are joined
    with " / ".
    """
    require_columns(panel, [DATE])
    paydays = _paydays(payday_days)
    if fourier_annual < 0 or fourier_weekly < 0:
        raise ValueError("Fourier orders must be non-negative")
    if fourier_weekly > 3:
        raise ValueError("fourier_weekly above 3 aliases with lower orders on a 7-day period")
    out = panel.copy()
    dates = normalize_dates(out[DATE])
    if dates.isna().any():
        raise ValueError("date contains missing values")
    out[DATE] = dates
    weekday = dates.dt.dayofweek.astype("int64")
    day = dates.dt.day.astype("int64")
    out[DAY_OF_WEEK] = weekday
    out[IS_WEEKEND] = weekday >= 5
    out[WEEK_OF_YEAR] = dates.dt.isocalendar().week.astype("int64").to_numpy()
    out[MONTH] = dates.dt.month.astype("int64")
    out[DAYS_TO_MONTH_END] = dates.dt.days_in_month.astype("int64") - day
    out[DAYS_SINCE_MONTH_START] = day - 1
    out[IS_PAYDAY_WINDOW] = day.isin(paydays)

    # Annual phase runs on days since 1970 so it is continuous across year boundaries.
    elapsed = (dates - _EPOCH).dt.days.to_numpy(dtype="float64")
    for order in range(1, fourier_annual + 1):
        angle = 2.0 * np.pi * order * elapsed / _ANNUAL_PERIOD_DAYS
        out[fourier_name("annual", "sin", order)] = np.sin(angle)
        out[fourier_name("annual", "cos", order)] = np.cos(angle)
    for order in range(1, fourier_weekly + 1):
        angle = 2.0 * np.pi * order * weekday.to_numpy(dtype="float64") / 7.0
        out[fourier_name("weekly", "sin", order)] = np.sin(angle)
        out[fourier_name("weekly", "cos", order)] = np.cos(angle)

    table = _holiday_table(country_code, holidays, dates.dt.year.unique())
    names = dates.map(table)
    out[HOLIDAY_NAME] = names.to_numpy(dtype=object)
    out[IS_HOLIDAY] = names.notna().to_numpy()
    return out


def _paydays(payday_days: Sequence[int]) -> list[int]:
    days = [int(d) for d in payday_days]
    if any(not 1 <= d <= 31 for d in days):
        raise ValueError("payday_days must be days of the month between 1 and 31")
    return days


def _holiday_table(
    country_code: str | None,
    holidays: Mapping[object, object] | None,
    years: Iterable[int],
) -> dict[pd.Timestamp, str]:
    names: dict[pd.Timestamp, list[str]] = {}
    if country_code is not None:
        for date, name in builtin_holidays(country_code, years).items():
            names.setdefault(date, []).append(name)
    for date, name in _supplied_holidays(holidays or {}):
        if name not in names.setdefault(date, []):
            names[date].append(name)
    return {date: " / ".join(labels) for date, labels in names.items()}


def _supplied_holidays(holidays: Mapping[object, object]) -> list[tuple[pd.Timestamp, str]]:
    pairs: list[tuple[pd.Timestamp, str]] = []
    for key, value in holidays.items():
        if isinstance(value, str):
            try:
                pairs.append((_holiday_date(key), value))
            except ValueError:  # {"Eid al-Fitr": "2024-04-10"}: a name is never a date
                pairs.append((_holiday_date(value), str(key)))
        elif isinstance(value, (pd.Timestamp, dt.date, np.datetime64)):
            pairs.append((_holiday_date(value), str(key)))
        elif isinstance(value, Iterable):
            pairs.extend((_holiday_date(item), str(key)) for item in value)
        else:
            raise TypeError(
                f"holidays entry {key!r}: {value!r}; use {{date: name}} or {{name: [dates]}}"
            )
    return pairs


def _holiday_date(value: object) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(value)  # type: ignore[arg-type]
    except (ValueError, TypeError) as error:
        raise ValueError(f"holiday date {value!r} is not a date") from error
    if stamp is pd.NaT:
        raise ValueError(f"holiday date {value!r} is missing")
    if stamp.tzinfo is not None:
        stamp = stamp.tz_localize(None)
    return stamp.normalize()
