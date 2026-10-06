import numpy as np
import pandas as pd
import pytest

from laee.features import (
    FEATURE_REGISTRY,
    SUPPORTED_COUNTRIES,
    add_calendar_features,
    add_lag_features,
    add_rolling_features,
    as_of,
    builtin_holidays,
    describe_feature,
    fourier_name,
    lag_name,
    rolling_name,
    trailing_stat,
)
from laee.features.temporal import (
    PROMO_FLAG_REALTIME,
    ROLLING_STATS,
    bool_column,
    ensure_unique_days,
    float_column,
    group_positions,
    map_unique,
    normalize_dates,
    require_columns,
    series_keys,
)
from laee.ingestion import fill_zero_sales_days, ingest_sales
from laee.normalization import detect_promotions, detect_stockouts, flag_price_outliers
from laee.schema import (
    AVAILABLE_AT,
    DATE,
    DISCOUNT_PCT,
    INVENTORY,
    LOCATION_ID,
    PANEL_KEYS,
    PRICE,
    PRODUCT_ID,
    PROMO_FLAG,
    REGULAR_PRICE,
    REVENUE,
    UNITS,
)

START = pd.Timestamp("2024-01-01")


def gappy_panel(rng: np.random.Generator, n_days: int = 120, products=("a", "b", "c")) -> pd.DataFrame:
    """Random product-day panel with ~20% of days missing and occasional price drops."""
    frames = []
    for product in products:
        dates = pd.date_range(START, periods=n_days, freq="D")
        keep = rng.random(n_days) > 0.2
        base = rng.choice([80.0, 100.0, 120.0])
        price = np.where(rng.random(n_days) < 0.15, base * 0.75, base)
        frames.append(
            pd.DataFrame(
                {
                    PRODUCT_ID: product,
                    DATE: dates[keep],
                    PRICE: price[keep],
                    UNITS: rng.poisson(4.0, n_days)[keep].astype(float),
                    PROMO_FLAG: (rng.random(n_days) < 0.05)[keep],
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


# --- lags and rolling windows ----------------------------------------------------------


def test_lag_features_use_calendar_days_not_row_positions():
    panel = pd.DataFrame(
        {
            PRODUCT_ID: ["a"] * 5 + ["b"] * 2,
            DATE: ["2024-03-01", "2024-03-02", "2024-03-04", "2024-03-08", "2024-03-09", "2024-03-01", "2024-03-02"],
            UNITS: [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0],
            PRICE: [9.0, 9.5, 9.0, 8.0, 9.0, 1.0, 1.0],
            PROMO_FLAG: [False, True, False, False, False, False, False],
        }
    )
    out = add_lag_features(panel)
    np.testing.assert_array_equal(out[lag_name(UNITS, 1)], [np.nan, 1, np.nan, np.nan, 4, np.nan, 10])
    np.testing.assert_array_equal(out[lag_name(UNITS, 7)], [np.nan, np.nan, np.nan, 1, 2, np.nan, np.nan])
    np.testing.assert_array_equal(out[lag_name(PRICE, 1)], [np.nan, 9.0, np.nan, np.nan, 8.0, np.nan, 1.0])
    # A row shift would have read 3 for 2024-03-08 and leaked product a into product b.
    assert out[DATE].dtype == "datetime64[ns]"
    flags = add_lag_features(panel, cols=PROMO_FLAG, lags_days=1)[lag_name(PROMO_FLAG, 1)]
    np.testing.assert_array_equal(flags, [np.nan, 0.0, np.nan, np.nan, 0.0, np.nan, 0.0])

    by_location = pd.concat([panel.assign(**{LOCATION_ID: "x"}), panel.assign(**{LOCATION_ID: "y", UNITS: 0.0})])
    lagged = add_lag_features(by_location, cols=UNITS, lags_days=1, by=(PRODUCT_ID, LOCATION_ID))
    assert lagged[lag_name(UNITS, 1)].iloc[1] == 1.0 and lagged[lag_name(UNITS, 1)].iloc[8] == 0.0
    with pytest.raises(ValueError, match="repeat"):
        add_lag_features(by_location, cols=UNITS)  # (product, date) is not unique across locations
    with pytest.raises(ValueError, match="positive integers"):
        add_lag_features(panel, lags_days=(0,))
    with pytest.raises(ValueError, match="key column"):
        add_lag_features(panel, cols=DATE)


def _brute_force(panel: pd.DataFrame, col: str, window: int, stat: str, min_periods: int) -> np.ndarray:
    dates = normalize_dates(panel[DATE])
    result = []
    for i in range(len(panel)):
        same = (panel[PRODUCT_ID] == panel[PRODUCT_ID].iloc[i]).to_numpy()
        inside = same & (dates >= dates.iloc[i] - pd.Timedelta(days=window)).to_numpy() & (dates < dates.iloc[i]).to_numpy()
        values = panel[col].to_numpy(dtype=float)[inside]
        values = values[~np.isnan(values)]
        if stat == "count":
            result.append(float(values.size))
        elif values.size < max(min_periods, 1) or (stat in ("std", "var") and values.size < 2):
            result.append(np.nan)
        else:
            func = {"std": lambda v: np.std(v, ddof=1), "var": lambda v: np.var(v, ddof=1)}.get(stat, getattr(np, stat))
            result.append(float(func(values)))
    return np.array(result)


def test_rolling_features_match_brute_force_strictly_past_windows(rng):
    panel = gappy_panel(rng, n_days=50)
    panel.loc[rng.random(len(panel)) < 0.1, UNITS] = np.nan
    out = add_rolling_features(panel, UNITS, windows_days=(3, 7), stats=ROLLING_STATS, min_periods=2)
    for window in (3, 7):
        for stat in ROLLING_STATS:
            expected = _brute_force(panel, UNITS, window, stat, min_periods=2)
            np.testing.assert_allclose(out[rolling_name(UNITS, stat, window)], expected, err_msg=f"{stat} {window}")
    first_rows = out.groupby(PRODUCT_ID).head(1)
    assert first_rows[rolling_name(UNITS, "mean", 7)].isna().all()  # nothing strictly before the first day
    assert (first_rows[rolling_name(UNITS, "count", 7)] == 0).all()
    with pytest.raises(ValueError, match="unknown stat"):
        add_rolling_features(panel, UNITS, stats=("mode",))


def test_trailing_stat_quantile_and_count():
    panel = pd.DataFrame(
        {PRODUCT_ID: "a", DATE: pd.date_range(START, periods=5, freq="D"), PRICE: [100.0, 80.0, 100.0, 100.0, 90.0]}
    )
    q75 = trailing_stat(panel, PRICE, 3, stat="quantile", q=0.75)
    np.testing.assert_array_equal(q75, [np.nan, 100.0, 100.0, 100.0, 100.0])
    low = trailing_stat(panel, PRICE, 3, stat="quantile", q=0.0)
    np.testing.assert_array_equal(low, [np.nan, 100.0, 80.0, 80.0, 80.0])
    assert trailing_stat(panel, PRICE, 3, stat="count").tolist() == [0.0, 1.0, 2.0, 3.0, 3.0]
    with pytest.raises(ValueError, match="needs q"):
        trailing_stat(panel, PRICE, 3, stat="quantile")
    with pytest.raises(ValueError, match="unknown stat"):
        trailing_stat(panel, PRICE, 3, stat="mode")


# --- leakage -----------------------------------------------------------------------------

PAST_ONLY = [
    lag_name(UNITS, 1), lag_name(UNITS, 7), lag_name(PRICE, 1), lag_name(PRICE, 7),
    rolling_name(UNITS, "mean", 7), rolling_name(UNITS, "sum", 28), rolling_name(UNITS, "std", 28),
    rolling_name(UNITS, "max", 7), rolling_name(UNITS, "count", 28), rolling_name(PRICE, "median", 28),
    REGULAR_PRICE,
]
SAME_DAY = [DISCOUNT_PCT, PROMO_FLAG_REALTIME]  # also use the day's own price


def _features(panel: pd.DataFrame) -> pd.DataFrame:
    out = add_lag_features(panel, cols=(UNITS, PRICE), lags_days=(1, 7))
    out = add_rolling_features(out, UNITS, windows_days=(7, 28), stats=("mean", "sum", "std", "max", "count"))
    out = add_rolling_features(out, PRICE, windows_days=28, stats="median")
    return detect_promotions(out).set_index([PRODUCT_ID, DATE]).sort_index()


def test_changing_future_rows_never_changes_past_features(rng):
    base = gappy_panel(rng)
    reference = _features(base)
    for offset in (15, 44, 77, 110):
        cutoff = START + pd.Timedelta(days=offset)
        future = (base[DATE] >= cutoff).to_numpy()
        changed = base.copy()
        changed.loc[future, PRICE] = changed.loc[future, PRICE] * rng.uniform(0.5, 1.5, future.sum())
        changed.loc[future, UNITS] = rng.poisson(9.0, future.sum()).astype(float)
        changed.loc[future, PROMO_FLAG] = rng.random(future.sum()) < 0.5
        changed = changed[~(future & (rng.random(len(changed)) < 0.3))]  # delete future rows
        missing = pd.MultiIndex.from_product([["a", "b", "c"], pd.date_range(cutoff, periods=40, freq="D")])
        missing = missing.difference(pd.MultiIndex.from_frame(changed[[PRODUCT_ID, DATE]]))[:25]
        inserted = pd.DataFrame(
            {PRODUCT_ID: missing.get_level_values(0), DATE: missing.get_level_values(1), PRICE: 55.0, UNITS: 30.0, PROMO_FLAG: True}
        )
        result = _features(pd.concat([changed, inserted], ignore_index=True))

        shared = reference.index.intersection(result.index)
        dates = shared.get_level_values(DATE)
        on_or_before, before = shared[dates <= cutoff], shared[dates < cutoff]
        assert len(before) > 0
        pd.testing.assert_frame_equal(reference.loc[on_or_before, PAST_ONLY], result.loc[on_or_before, PAST_ONLY])
        pd.testing.assert_frame_equal(reference.loc[before, SAME_DAY], result.loc[before, SAME_DAY])


def test_retrospective_promo_flag_is_the_only_label_that_looks_ahead():
    prices = np.r_[np.full(40, 100.0), np.full(10, 80.0), np.full(30, 100.0)]
    panel = pd.DataFrame({PRODUCT_ID: "a", DATE: pd.date_range(START, periods=80, freq="D"), PRICE: prices, UNITS: 2.0})
    never_reverts = panel.assign(**{PRICE: np.r_[np.full(40, 100.0), np.full(40, 80.0)]})
    before, after = detect_promotions(panel), detect_promotions(never_reverts)
    past = slice(40, 45)  # days before a cutoff at day 45; only later prices differ
    assert before[PROMO_FLAG].iloc[past].all() and not after[PROMO_FLAG].iloc[past].any()
    assert before[PROMO_FLAG_REALTIME].iloc[past].all() and after[PROMO_FLAG_REALTIME].iloc[past].all()
    assert not describe_feature(PROMO_FLAG).realtime_safe and describe_feature(PROMO_FLAG_REALTIME).realtime_safe


# --- availability and registry --------------------------------------------------------


def test_as_of_keeps_rows_available_by_prediction_time():
    panel = ingest_sales(
        "date,product,price,qty\n2024-03-01,a,10,1\n2024-03-02,a,10,2\n2024-03-03,a,11,1\n"
    ).panel
    assert panel[AVAILABLE_AT].tolist() == [pd.Timestamp("2024-03-02"), pd.Timestamp("2024-03-03"), pd.Timestamp("2024-03-04")]
    known = as_of(panel, "2024-03-03")
    assert known[DATE].tolist() == [pd.Timestamp("2024-03-01"), pd.Timestamp("2024-03-02")]
    assert len(as_of(panel, pd.Timestamp("2024-03-02 23:59"))) == 1
    unknown = panel.assign(**{AVAILABLE_AT: [pd.Timestamp("2024-03-02"), pd.NaT, pd.NaT]})
    assert len(as_of(unknown, "2030-01-01")) == 1
    with pytest.raises(KeyError, match="available_at"):
        as_of(panel.drop(columns=AVAILABLE_AT), "2024-03-03")
    with pytest.raises(ValueError, match="timezone"):
        as_of(panel, pd.Timestamp("2024-03-03", tz="UTC"))


def test_feature_registry_describes_every_generated_column():
    csv = "date,product,price,qty\n" + "".join(
        f"2024-03-{d:02d},{p},{10 if d % 9 else 8},{1 + d % 3}\n" for p in ("a", "b") for d in range(1, 29) if d % 5
    )
    panel = flag_price_outliers(ingest_sales(csv).panel)
    panel = fill_zero_sales_days(detect_promotions(panel))
    panel = detect_stockouts(panel)
    panel = add_lag_features(panel, cols=(UNITS, PRICE, PROMO_FLAG))
    panel = add_rolling_features(panel, UNITS, stats=ROLLING_STATS)
    panel = add_calendar_features(panel, country_code="KE")
    raw = {*PANEL_KEYS, PRICE, UNITS, REVENUE, INVENTORY, AVAILABLE_AT}
    generated = [col for col in panel.columns if col not in raw]
    assert len(generated) > 50
    for col in generated:
        spec = describe_feature(col)
        assert spec.source and spec.availability
    assert describe_feature(lag_name(UNITS, 7)).realtime_safe
    lagged_label = describe_feature(lag_name(PROMO_FLAG, 1))
    assert not lagged_label.realtime_safe and lagged_label.retrospective
    assert describe_feature(lag_name(PROMO_FLAG_REALTIME, 1)).realtime_safe
    assert describe_feature(rolling_name(UNITS, "mean", 7)).realtime_safe
    assert describe_feature("holiday_name").realtime_safe and describe_feature(fourier_name("weekly", "sin", 2)).realtime_safe
    assert not describe_feature("censored_flag").realtime_safe and not describe_feature("censored_flag").retrospective
    assert all(key == spec.pattern for key, spec in FEATURE_REGISTRY.items())
    with pytest.raises(KeyError):
        describe_feature("tomorrow_units")


# --- calendar ------------------------------------------------------------------------------


def test_calendar_features_values():
    dates = ["2024-02-29", "2024-03-01", "2024-03-02", "2024-03-03", "2024-03-25", "2024-12-30", "2025-01-01"]
    out = add_calendar_features(pd.DataFrame({DATE: dates}))
    assert out["day_of_week"].tolist() == [3, 4, 5, 6, 0, 0, 2]
    assert out["is_weekend"].tolist() == [False, False, True, True, False, False, False]
    assert out["week_of_year"].tolist() == [9, 9, 9, 9, 13, 1, 1]  # ISO: 2024-12-30 is in 2025-W01
    assert out["month"].tolist() == [2, 3, 3, 3, 3, 12, 1]
    assert out["days_to_month_end"].tolist() == [0, 30, 29, 28, 6, 1, 30]
    assert out["days_since_month_start"].tolist() == [28, 0, 1, 2, 24, 29, 0]
    assert out["is_payday_window"].tolist() == [True, True, True, False, True, True, True]
    elapsed = (pd.to_datetime(dates) - pd.Timestamp("1970-01-01")).days.to_numpy()
    np.testing.assert_allclose(out[fourier_name("annual", "sin", 2)], np.sin(4 * np.pi * elapsed / 365.25))
    for period, order in (("annual", 3), ("weekly", 2)):
        sin, cos = out[fourier_name(period, "sin", order)], out[fourier_name(period, "cos", order)]
        np.testing.assert_allclose(sin**2 + cos**2, 1.0)
    mondays = out[out["day_of_week"] == 0]
    assert (mondays[fourier_name("weekly", "sin", 1)].abs() < 1e-12).all()
    assert (mondays[fourier_name("weekly", "cos", 1)] == 1.0).all()
    assert not out["is_holiday"].any() and out["holiday_name"].isna().all()
    assert fourier_name("annual", "sin", 4) not in out.columns
    custom = add_calendar_features(pd.DataFrame({DATE: dates}), payday_days=(1,), fourier_annual=0, fourier_weekly=0)
    assert custom["is_payday_window"].tolist() == [False, True, False, False, False, False, True]
    assert not any(col.startswith("fourier_") for col in custom.columns)


def test_calendar_holidays_built_in_and_supplied_moveable_dates():
    dates = ["2024-04-10", "2024-06-01", "2024-10-10", "2024-12-12", "2024-12-25", "2024-12-26", "2024-12-27", "2024-03-29"]
    supplied = {
        "2024-04-10": "Eid al-Fitr",
        "Good Friday": pd.Timestamp("2024-03-29"),
        "Diwali": ["2024-11-01", "2025-10-21"],
        "2024-12-25": "Christmas Day",  # already built in: not repeated
    }
    out = add_calendar_features(pd.DataFrame({DATE: dates}), country_code="ke", holidays=supplied)
    assert [name if isinstance(name, str) else None for name in out["holiday_name"]] == [
        "Eid al-Fitr", "Madaraka Day", "Mazingira Day", "Jamhuri Day", "Christmas Day", "Boxing Day", None, "Good Friday",
    ]
    assert out["is_holiday"].tolist() == [True, True, True, True, True, True, False, True]
    assert pd.isna(out["holiday_name"].iloc[6])
    merged = add_calendar_features(pd.DataFrame({DATE: ["2024-12-25"]}), country_code="GB", holidays={"2024-12-25": "Store anniversary"})
    assert merged["holiday_name"].tolist() == ["Christmas Day / Store anniversary"]
    only_supplied = add_calendar_features(pd.DataFrame({DATE: ["2024-11-01", "2024-12-25"]}), holidays={"Diwali": ["2024-11-01"]})
    assert only_supplied["is_holiday"].tolist() == [True, False]
    name_first = add_calendar_features(pd.DataFrame({DATE: ["2024-06-16"]}), holidays={"Eid al-Adha": "2024-06-16"})
    assert name_first["holiday_name"].tolist() == ["Eid al-Adha"]
    with pytest.raises(ValueError, match="not a date"):
        add_calendar_features(pd.DataFrame({DATE: dates}), holidays={"Eid": "someday"})

    assert pd.Timestamp("2018-05-29") in builtin_holidays("NG", [2018, 2019])
    assert pd.Timestamp("2018-06-12") not in builtin_holidays("NG", [2018])
    assert builtin_holidays("NG", [2019])[pd.Timestamp("2019-06-12")] == "Democracy Day"
    assert pd.Timestamp("2020-06-19") not in builtin_holidays("US", [2020])
    assert pd.Timestamp("2021-06-19") in builtin_holidays("US", [2021])
    assert builtin_holidays("ZA", [2024])[pd.Timestamp("2024-06-16")] == "Youth Day"
    assert builtin_holidays("GH", [2024])[pd.Timestamp("2024-03-06")] == "Independence Day"
    assert set(SUPPORTED_COUNTRIES) == {"KE", "NG", "GH", "ZA", "US", "GB"}
    with pytest.raises(ValueError, match="supported"):
        add_calendar_features(pd.DataFrame({DATE: dates}), country_code="FR")
    with pytest.raises(TypeError):
        add_calendar_features(pd.DataFrame({DATE: dates}), holidays={"Eid": 5})
    with pytest.raises(ValueError):
        add_calendar_features(pd.DataFrame({DATE: dates}), fourier_weekly=4)
    with pytest.raises(ValueError):
        add_calendar_features(pd.DataFrame({DATE: dates}), payday_days=(0,))


# --- shared helpers -------------------------------------------------------------------------


def test_temporal_helpers():
    stamps = normalize_dates(pd.Series(pd.to_datetime(["2024-03-01 23:30"]).tz_localize("Africa/Lagos")))
    assert stamps.tolist() == [pd.Timestamp("2024-03-01")] and stamps.dtype == "datetime64[ns]"
    assert normalize_dates(["2024-03-02"]).tolist() == [pd.Timestamp("2024-03-02")]
    frame = pd.DataFrame({PRODUCT_ID: ["a", "b", "a"], LOCATION_ID: "x", DATE: pd.to_datetime(["2024-03-03", "2024-03-01", "2024-03-01"])})
    assert series_keys(frame) == [PRODUCT_ID, LOCATION_ID]
    assert [list(p) for p in group_positions(frame, [PRODUCT_ID])] == [[2, 0], [1]]
    with pytest.raises(KeyError, match="product_id"):
        series_keys(frame.drop(columns=PRODUCT_ID))
    with pytest.raises(KeyError, match="price"):
        require_columns(frame, [PRICE])
    with pytest.raises(ValueError, match="repeat"):
        ensure_unique_days(pd.concat([frame, frame]), [PRODUCT_ID])
    flags = pd.DataFrame({"f": ["False", "true", None, 1, 0.0, np.nan]})
    assert bool_column(flags, "f").tolist() == [False, True, False, True, False, False]
    assert not bool_column(flags, "absent").any()
    values = float_column(pd.DataFrame({"v": ["1.5", "x", 2]}), "v")
    np.testing.assert_array_equal(values, [1.5, np.nan, 2.0])
    values[0] = 0.0  # writable, not a read-only copy-on-write view
    assert np.isnan(float_column(frame, "absent")).all()
    calls = []
    mapped = map_unique(pd.Series(["x", None, "y", "x"], name="s"), lambda v: calls.append(v) or v.upper(), missing="?")
    assert mapped.tolist() == ["X", "?", "Y", "X"] and mapped.name == "s" and calls == ["x", "y"]
