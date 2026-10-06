import numpy as np
import pandas as pd
import pytest

from laee.features.temporal import (
    DAYS_OUT_OF_STOCK,
    DISCOUNT_PCT_EXPLICIT,
    IMPUTED_ZERO,
    INVENTORY_AVAILABLE,
    PRICE_OUTLIER,
    PRICE_ROBUST_Z,
    PROMO_FLAG_EXPLICIT,
    PROMO_FLAG_REALTIME,
    SUSPECTED_STOCKOUT,
)
from laee.ingestion import fill_zero_sales_days, ingest_inventory, ingest_sales, to_daily_panel
from laee.normalization import (
    detect_price_changes,
    detect_promotions,
    detect_stockouts,
    estimation_mask,
    flag_price_outliers,
    normalize_product_ids,
    price_points,
)
from laee.schema import (
    CENSORED_FLAG,
    DATE,
    DISCOUNT_PCT,
    INVENTORY,
    PRICE,
    PRODUCT_ID,
    PROMO_FLAG,
    REGULAR_PRICE,
    STOCKOUT_FLAG,
    UNITS,
)

START = pd.Timestamp("2024-01-01")


def series(prices, units=3.0, product="a", start=START) -> pd.DataFrame:
    prices = np.asarray(prices, dtype=float)
    return pd.DataFrame(
        {
            PRODUCT_ID: product,
            DATE: pd.date_range(start, periods=len(prices), freq="D"),
            PRICE: prices,
            UNITS: np.broadcast_to(np.asarray(units, dtype=float), prices.shape).copy(),
        }
    )


def day(offset: int) -> pd.Timestamp:
    return START + pd.Timedelta(days=offset)


def flagged(frame: pd.DataFrame, col: str) -> list[int]:
    return np.flatnonzero(frame[col].to_numpy()).tolist()


# --- product ids and prices ------------------------------------------------------------


def test_normalize_product_ids_unifies_spelling_variants():
    raw = pd.Series(
        [
            "  Coca-Cola 500ml ", "COCA–COLA  500ML", "coca—cola 500ml",
            "Mama’s “Best” Unga", "mama's \"best\" unga", 1001.0, 7,
            None, np.nan, "   ", "ＳＫＵ－１",
        ],
        dtype=object,
    )
    out = normalize_product_ids(raw).tolist()
    assert out[:3] == ["coca-cola 500ml"] * 3
    assert out[3:5] == ['mama\'s "best" unga'] * 2
    assert out[5:7] == ["1001", "7"]
    assert all(pd.isna(v) for v in out[7:10])
    assert out[10] == "sku-1"


def test_flag_price_outliers_catches_gross_errors_but_keeps_promotions():
    prices = np.full(100, 100.0)
    prices[10:23] = 70.0  # a 30%-off promotion
    prices[50] = 1000.0  # extra zero typed
    prices[80] = 10.0  # decimal misread
    out = flag_price_outliers(series(prices))
    assert flagged(out, PRICE_OUTLIER) == [50, 80]
    assert out[PRICE_ROBUST_Z].iloc[50] > 5 and out[PRICE_ROBUST_Z].iloc[80] < -5
    assert out[PRICE_ROBUST_Z].iloc[15] == pytest.approx(-3.08, abs=0.01)  # MAD = 0: mean-AD scale


def test_flag_price_outliers_spares_regime_changes_small_series_and_imputed_rows():
    regime = series(np.r_[np.full(90, 100.0), np.full(10, 250.0)])
    out = flag_price_outliers(regime)
    assert out[PRICE_ROBUST_Z].iloc[-1] > 5 and not out[PRICE_OUTLIER].any()  # 2.5x < min_ratio
    assert flag_price_outliers(regime, min_ratio=2.0)[PRICE_OUTLIER].sum() == 10

    levels = pd.concat([series([100.0] * 10, product="a"), series([300.0] * 10, product="b")])
    assert not flag_price_outliers(levels)[PRICE_OUTLIER].any()  # scored within each product

    tiny = flag_price_outliers(series([100.0, 1000.0]))
    assert tiny[PRICE_ROBUST_Z].isna().all() and not tiny[PRICE_OUTLIER].any()

    odd = series([100.0, 100.0, 0.0, 100.0, 100.0])
    odd[IMPUTED_ZERO] = [False, False, False, False, True]
    odd.loc[4, PRICE] = 5000.0
    out = flag_price_outliers(odd)
    assert out[PRICE_OUTLIER].tolist() == [False, False, True, False, False]
    assert np.isnan(out[PRICE_ROBUST_Z].iloc[4])
    with pytest.raises(ValueError):
        flag_price_outliers(odd, z=0)


def test_price_points_returns_grid_and_minimum_step():
    usd = price_points([9.99, 10.49, 10.99, 10.49, np.nan, -1.0])
    np.testing.assert_allclose(usd.points, [9.99, 10.49, 10.99])
    assert usd.counts.tolist() == [1, 2, 1] and usd.tick == 0.01 and usd.min_step == 0.5

    kes = price_points(pd.Series([50, 55, 60, 70, 55]))
    assert kes.points.tolist() == [50, 55, 60, 70] and kes.tick == 5.0 and kes.min_step == 5.0

    averaged = price_points([100.0] * 10 + [90.0] * 9 + [96.67])  # one mixed-price daily average
    assert averaged.points.tolist() == [90.0, 100.0] and averaged.counts.tolist() == [9, 10]
    assert averaged.tick == 5.0 and averaged.min_step == 10.0  # tick capped at 10% of the median

    assert price_points(np.array([10.0])).min_step == 1.0
    no_grid = price_points([33.333333, 34.5678])
    np.testing.assert_allclose(no_grid.points, [33.33, 34.57])
    with pytest.raises(ValueError):
        price_points([np.nan, 0.0])


def _price_change_panel() -> pd.DataFrame:
    panel = series(np.r_[np.full(30, 100.0), np.full(30, 110.0), np.full(40, 95.0)])
    panel[PROMO_FLAG] = False
    panel.loc[45, [PRICE, PROMO_FLAG]] = [80.0, True]
    return panel


def test_detect_price_changes_from_raw_prices():
    events = detect_price_changes(_price_change_panel())
    assert events.columns.tolist() == [PRODUCT_ID, DATE, "old_price", "new_price", "rel_change"]
    assert events[DATE].tolist() == [day(30), day(60)]
    assert events["old_price"].tolist() == [100.0, 110.0]
    assert events["new_price"].tolist() == [110.0, 95.0]
    np.testing.assert_allclose(events["rel_change"], [0.1, 95 / 110 - 1])
    assert detect_price_changes(_price_change_panel(), min_rel_change=0.2).empty


def test_detect_price_changes_from_regular_price_dates_back_to_first_new_price():
    promos = detect_promotions(_price_change_panel())
    assert promos[REGULAR_PRICE].iloc[31] == 100.0 and promos[REGULAR_PRICE].iloc[32] == 110.0
    assert promos[REGULAR_PRICE].iloc[81] == 110.0 and promos[REGULAR_PRICE].iloc[82] == 95.0
    events = detect_price_changes(promos)
    assert events[DATE].tolist() == [day(30), day(60)]  # confirmed on days 32 and 82
    assert events["new_price"].tolist() == [110.0, 95.0]


# --- promotions ------------------------------------------------------------------------


def _promo_path(rng: np.random.Generator) -> pd.DataFrame:
    prices = np.full(170, 100.0)
    prices[60:74] = 80.0  # two-week promotion that reverts
    prices[120:] = 85.0  # permanent cut
    return series(prices, units=1 + rng.poisson(3.0, size=170))


def test_two_week_promotion_retrospective_versus_realtime(rng):
    out = detect_promotions(_promo_path(rng))
    promo_days = list(range(60, 74))
    cut_days = list(range(120, 141))  # first 21 days of the permanent cut
    assert flagged(out, PROMO_FLAG) == promo_days
    assert flagged(out, PROMO_FLAG_REALTIME) == promo_days + cut_days
    differ = np.flatnonzero(out[PROMO_FLAG].to_numpy() != out[PROMO_FLAG_REALTIME].to_numpy())
    assert differ.tolist() == cut_days  # realtime cannot know the cut never reverts

    regular = out[REGULAR_PRICE].to_numpy()
    assert np.isnan(regular[0])
    assert (regular[1:142] == 100.0).all()  # the promotion does not move the regular price
    assert (regular[142:] == 85.0).all()  # the cut is adopted after 22 days
    discount = out[DISCOUNT_PCT].to_numpy()
    np.testing.assert_allclose(discount[60:74], 0.2)
    np.testing.assert_allclose(discount[120:142], 0.15)
    assert (discount[1:60] == 0).all() and (discount[142:] == 0).all()
    assert not out[PROMO_FLAG_EXPLICIT].any() and out[DISCOUNT_PCT_EXPLICIT].isna().all()


def test_regular_price_follows_increases_so_later_promotions_are_found():
    # 100 -> 130 on day 40, then 20% off 130 for a week: below the old price too.
    prices = np.r_[np.full(40, 100.0), np.full(10, 130.0), np.full(7, 104.0), np.full(20, 130.0)]
    out = detect_promotions(series(prices))
    assert out[REGULAR_PRICE].iloc[41] == 100.0 and out[REGULAR_PRICE].iloc[42] == 130.0
    assert flagged(out, PROMO_FLAG) == list(range(50, 57))
    np.testing.assert_allclose(out[DISCOUNT_PCT].iloc[50:57], 0.2)


def test_promotion_episode_closing_rules():
    base = np.full(30, 100.0)
    # A mixed daily average (90: half the sales still at the promo price) as the promotion ends.
    mixed = detect_promotions(series(np.r_[base, np.full(5, 80.0), [93.0], np.full(10, 100.0)]))
    assert flagged(mixed, PROMO_FLAG) == list(range(30, 36))
    # Ending above the old price (promotion, then an increase) still reveals a temporary drop.
    higher = detect_promotions(series(np.r_[base, np.full(5, 80.0), np.full(10, 108.0)]))
    assert flagged(higher, PROMO_FLAG) == list(range(30, 35))
    # Settling at a shallower level for two days: a permanent cut to 95, not a promotion.
    settled = detect_promotions(series(np.r_[base, np.full(5, 80.0), np.full(10, 95.0)]))
    assert not settled[PROMO_FLAG].any()
    assert flagged(settled, PROMO_FLAG_REALTIME) == list(range(30, 36))  # the day-35 price was ambiguous
    assert settled[REGULAR_PRICE].iloc[36] == 100.0 and settled[REGULAR_PRICE].iloc[37] == 95.0


def test_drop_longer_than_max_duration_is_not_a_promotion():
    prices = np.r_[np.full(60, 100.0), np.full(30, 80.0), np.full(30, 100.0)]
    out = detect_promotions(series(prices))
    assert not out[PROMO_FLAG].any()
    assert flagged(out, PROMO_FLAG_REALTIME) == list(range(60, 81))


def test_explicit_promotion_information_and_idempotence():
    panel = series(np.full(40, 100.0))
    panel[PROMO_FLAG] = False
    panel[DISCOUNT_PCT] = 0.0
    panel.loc[1:5, [PRICE, PROMO_FLAG]] = [60.0, True]  # flagged promotion days
    panel.loc[10, PROMO_FLAG] = True  # e.g. a bundle offer at the normal price
    panel.loc[20, DISCOUNT_PCT] = 0.05  # coupon recorded, shelf price unchanged
    out = detect_promotions(panel)
    assert flagged(out, PROMO_FLAG) == [1, 2, 3, 4, 5, 10, 20]
    assert flagged(out, PROMO_FLAG_REALTIME) == [1, 2, 3, 4, 5, 10, 20]
    # Explicitly promotional prices never enter the regular-price window.
    assert out[REGULAR_PRICE].iloc[2:7].tolist() == [100.0] * 5
    assert out[DISCOUNT_PCT].iloc[20] == 0.05
    assert out[DISCOUNT_PCT].iloc[3] == pytest.approx(0.4)
    assert flagged(out, PROMO_FLAG_EXPLICIT) == [1, 2, 3, 4, 5, 10]
    pd.testing.assert_frame_equal(detect_promotions(out), out)
    with pytest.raises(ValueError):
        detect_promotions(panel, min_depth=0.02, revert_tolerance=0.05)


def test_regular_price_uses_past_only_and_falls_back_for_slow_movers():
    slow = pd.DataFrame(
        {
            PRODUCT_ID: "a",
            DATE: [day(0), day(40), day(80), day(120)],
            PRICE: [100.0, 100.0, 90.0, 90.0],
            UNITS: 1.0,
        }
    )
    regular = detect_promotions(slow)[REGULAR_PRICE].tolist()
    assert np.isnan(regular[0]) and regular[1:] == [100.0, 100.0, 90.0]
    cheaper_today = slow.assign(**{PRICE: [100.0, 100.0, 50.0, 90.0]})
    assert detect_promotions(cheaper_today)[REGULAR_PRICE].iloc[2] == 100.0


# --- stockouts -------------------------------------------------------------------------


def test_stockouts_and_censoring_from_start_of_day_inventory():
    panel = series(np.full(10, 10.0), units=[5, 5, 5, 5, 0, 0, 3, 5, 2, 1])
    counts = pd.DataFrame(
        {
            PRODUCT_ID: "a",
            DATE: pd.date_range(START, periods=10, freq="D"),
            INVENTORY: [20, 15, 10, 5, 0, 0, 10, 7, 2, 4],
            "business_id": "default",
        }
    )
    counts = pd.concat([counts, counts.iloc[[6]].assign(**{INVENTORY: 3})])  # a lower recount
    out = detect_stockouts(panel, counts)
    assert flagged(out, STOCKOUT_FLAG) == [4, 5]
    assert flagged(out, CENSORED_FLAG) == [3, 4, 5, 8]  # sold out, or nothing to sell
    assert out[DAYS_OUT_OF_STOCK].tolist() == [0, 0, 0, 0, 1, 2, 0, 0, 0, 0]
    assert out[INVENTORY].tolist() == [20, 15, 10, 5, 0, 0, 10, 7, 2, 4]
    assert out[INVENTORY_AVAILABLE].tolist() == out[INVENTORY].tolist()
    assert not out[SUSPECTED_STOCKOUT].any()


def test_stockouts_from_end_of_day_counts_and_panel_inventory():
    panel = series(np.full(10, 10.0), units=[5, 5, 5, 5, 0, 0, 3, 5, 2, 1])
    panel[INVENTORY] = [15, 10, 5, 0, 0, 0, 7, 2, 0, 3]
    out = detect_stockouts(panel, inventory_timing="end_of_day")
    assert out[INVENTORY_AVAILABLE].tolist() == [20, 15, 10, 5, 0, 0, 10, 7, 2, 4]
    assert flagged(out, CENSORED_FLAG) == [3, 4, 5, 8]
    assert flagged(out, STOCKOUT_FLAG) == [3, 4, 5, 8]
    with pytest.warns(UserWarning, match="matched no panel row"):
        detect_stockouts(panel, pd.DataFrame({PRODUCT_ID: ["zzz"], DATE: [START], INVENTORY: [1.0]}))
    with pytest.raises(ValueError):
        detect_stockouts(panel, inventory_timing="noon")


def _zero_run_panel() -> pd.DataFrame:
    cycle = np.array([2.0, 3.0, 4.0, 3.0, 3.0])
    fast_units = np.r_[np.resize(cycle, 40), np.zeros(4), np.resize(cycle, 10), [0.0], np.resize(cycle, 5)]
    slow_units = np.where(np.arange(60) % 5 == 0, 1.0, 0.0)
    new_units = np.array([5.0, 5.0, 5.0, 0.0, 0.0, 0.0, 5.0])
    rows = []
    for product, units in (("fast", fast_units), ("slow", slow_units), ("new", new_units)):
        frame = series(np.full(len(units), 10.0), units=units, product=product)
        rows.append(frame[frame[UNITS] > 0])  # an export only lists days with sales
    return fill_zero_sales_days(to_daily_panel(pd.concat(rows, ignore_index=True)))


def test_poisson_zero_runs_flag_suspected_stockouts():
    out = detect_stockouts(_zero_run_panel())
    fast = out[out[PRODUCT_ID] == "fast"].reset_index(drop=True)
    assert flagged(fast, SUSPECTED_STOCKOUT) == [40, 41, 42, 43]  # exp(-3 * 4) is tiny
    assert not fast.loc[54, SUSPECTED_STOCKOUT]  # one quiet day at ~2.6/day is plausible
    assert not out.loc[out[PRODUCT_ID] == "slow", SUSPECTED_STOCKOUT].any()  # 0.2/day sells rarely
    assert not out.loc[out[PRODUCT_ID] == "new", SUSPECTED_STOCKOUT].any()  # under 7 days of history
    assert not out[STOCKOUT_FLAG].any() and not out[CENSORED_FLAG].any()
    strict = detect_stockouts(_zero_run_panel(), zero_run_prob=0.1)
    assert strict.loc[(strict[PRODUCT_ID] == "fast") & (strict[DATE] == day(54)), SUSPECTED_STOCKOUT].item()


def test_known_stock_rules_out_suspected_stockouts():
    panel = _zero_run_panel()
    in_stock = pd.DataFrame({PRODUCT_ID: "fast", DATE: [day(d) for d in range(40, 44)], INVENTORY: 8.0})
    out = detect_stockouts(panel, in_stock)
    assert not out[SUSPECTED_STOCKOUT].any() and not out[STOCKOUT_FLAG].any()


def test_estimation_mask_excludes_contaminated_rows_but_keeps_promotions():
    panel = pd.DataFrame(
        {
            PRICE: [10.0, 10, 10, 10, 10, 0, np.nan, 10, 10],
            UNITS: [1.0, 1, 1, 1, 1, 1, 1, np.nan, 0],
            STOCKOUT_FLAG: [False, True, False, False, False, False, False, False, False],
            CENSORED_FLAG: [False, False, True, False, False, False, False, False, False],
            SUSPECTED_STOCKOUT: [False, False, False, True, False, False, False, False, False],
            PRICE_OUTLIER: [False, False, False, False, True, False, False, False, False],
            PROMO_FLAG: [True, False, False, False, False, False, False, False, False],
        }
    )
    mask = estimation_mask(panel)
    assert mask.tolist() == [True, False, False, False, False, False, False, False, True]
    assert estimation_mask(panel[[PRICE, UNITS]]).tolist() == [True] * 5 + [False, False, False, True]


# --- full chain on a messy export ------------------------------------------------------


def test_normalization_pipeline_on_messy_export(rng):
    sale_days = [d for d in range(90) if not 60 <= d <= 64 and d != 75]  # stockout and a quiet day
    lines = ["Date,Item,Price,Qty"]
    for d in sale_days:
        price = "KSh 1,200.00" if d == 50 else ("100" if 30 <= d <= 39 else "120")
        lines.append(f"{day(d):%d/%m/%Y},Unga 2kg,\"{price}\",{1 + rng.poisson(1.5)}")
    sales = ingest_sales("\n".join(lines) + "\n")
    stock_lines = ["Item,Date,Stock"] + [
        f"UNGA 2KG,{day(d):%d/%m/%Y},{0 if 60 <= d <= 64 else 40}" for d in range(55, 70)
    ]
    counts, _ = ingest_inventory("\n".join(stock_lines))

    panel = flag_price_outliers(sales.panel)
    panel = detect_promotions(panel)
    panel = fill_zero_sales_days(panel)
    panel = detect_stockouts(panel, counts)
    mask = estimation_mask(panel)

    assert len(panel) == 90 and panel[IMPUTED_ZERO].sum() == 6
    assert flagged(panel, PRICE_OUTLIER) == [50]
    assert flagged(panel, PROMO_FLAG) == list(range(30, 40))
    assert flagged(panel, STOCKOUT_FLAG) == list(range(60, 65))
    assert not panel[SUSPECTED_STOCKOUT].any()
    assert sorted(set(range(90)) - set(np.flatnonzero(mask))) == [50, 60, 61, 62, 63, 64]
