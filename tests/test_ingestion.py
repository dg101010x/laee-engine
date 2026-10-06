import datetime as dt

import numpy as np
import pandas as pd
import pytest

from laee.features.temporal import IMPUTED_ZERO, N_PRICES, N_TRANSACTIONS, PRICE_MAX, PRICE_MIN
from laee.ingestion import (
    PRODUCT_LABEL,
    REORDER_POINT,
    SALES_DROP_REASONS,
    SALES_SYNONYMS,
    SOURCE_ROW,
    detect_columns,
    fill_zero_sales_days,
    ingest_inventory,
    ingest_sales,
    normalize_header,
    parse_bool,
    parse_dates,
    parse_numeric,
    read_table,
    to_daily_panel,
)
from laee.normalization import detect_promotions
from laee.schema import (
    AVAILABLE_AT,
    BUSINESS_ID,
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
    UNIT_COST,
    UNITS,
)

MESSY_CSV = """Daily sales export
Shop: Mama Njeri Kiosk, Nairobi

Sale Date,Item Name,Unit Price (KSh),Qty,Line Total,Promo
01/03/2024,Unga 2kg,"KSh 1,234.50",2,"2,469.00",
01/03/2024,Unga 2kg,"KSh 1,234.50",2,"2,469.00",
02/03/2024,unga  2KG ,,1,"KSh 1,200.00",yes
02/03/2024,Unga 2kg,"1,234.50",1,,
13/03/2024,Sugar 1kg,KES 180,3,,
14/03/2024,Sugar 1kg,190,-1,-190,
15/03/2024,,190,1,190,
xx/03/2024,Sugar 1kg,190,1,190,
16/03/2024,Sugar 1kg,0,1,0,
17/03/2024,Sugar 1kg,195,0,0,
18/03/2024,Sugar 1kg,195,,,
19/03/2024,Sugar 1kg,,2,,
20/03/2024,Sugar 1kg,KSh 200/=,2,400,
"""


def days(*values: str) -> list[pd.Timestamp]:
    return [pd.Timestamp(v) for v in values]


# --- parse_numeric -----------------------------------------------------------------


def test_parse_numeric_strips_currencies_and_reads_negatives():
    raw = [
        "KSh 1,234.50", "KES 999", "(1,000.00)", "USD 12.5", "$3", "R 1 234.50", "₦2,500",
        "GH₵ 50", "£7.25", "€ 8", "ZAR 10", "NGN 1,000,000", "GHS 3.5", "-KSh 40",
        "1,200/=", "500/-", "US$12", "120-", "", "n/a", "abc",
    ]
    parsed = parse_numeric(pd.Series(raw))
    expected = [
        1234.5, 999, -1000, 12.5, 3, 1234.5, 2500, 50, 7.25, 8, 10, 1_000_000, 3.5, -40,
        1200, 500, 12, -120, np.nan, np.nan, np.nan,
    ]
    np.testing.assert_allclose(parsed.values.to_numpy(), expected)
    assert parsed.n_invalid == 1
    assert parsed.decimal == "."
    assert parsed.ambiguous_example is None


def test_parse_numeric_detects_decimal_comma_per_column():
    european = parse_numeric(pd.Series(["1.234,50", "€ 12,5", "R 1 234,50", "999", "1.000.000", "(2.500,00)"]))
    np.testing.assert_allclose(european.values, [1234.5, 12.5, 1234.5, 999, 1_000_000, -2500])
    assert european.decimal == ","
    american = parse_numeric(pd.Series(["1,234.50", "12.5", "1,000,000"]))
    np.testing.assert_allclose(american.values, [1234.5, 12.5, 1_000_000])
    assert american.decimal == "."
    # A lone "1,234" follows the column's settled convention.
    np.testing.assert_allclose(parse_numeric(pd.Series(["1,234", "12,5"])).values, [1.234, 12.5])
    np.testing.assert_allclose(parse_numeric(pd.Series(["1,234", "12.5"])).values, [1234, 12.5])


def test_parse_numeric_settles_lone_thousands_groups_by_magnitude_or_flags_them():
    np.testing.assert_allclose(parse_numeric(pd.Series(["1.500", "750", "2.000"])).values, [1500, 750, 2000])
    np.testing.assert_allclose(parse_numeric(pd.Series(["1,250", "1,500", "2"])).values, [1.25, 1.5, 2.0])
    tie = parse_numeric(pd.Series(["1.250", "2.500"]))
    np.testing.assert_allclose(tie.values, [1.25, 2.5])
    assert tie.ambiguous_example == "1.250"


def test_parse_numeric_percent_units_and_typed_input():
    percent = parse_numeric(pd.Series(["10%", "5 %"]))
    assert percent.percent and percent.values.tolist() == [10.0, 5.0]
    with_units = parse_numeric(pd.Series(["3 pcs", "2kg", "1.5 L"]), strip_units=True)
    assert with_units.values.tolist() == [3.0, 2.0, 1.5]
    assert parse_numeric(pd.Series(["3 pcs"])).n_invalid == 1
    typed = parse_numeric(pd.Series([1, 2.5]))
    assert typed.values.tolist() == [1.0, 2.5]
    mixed = parse_numeric(pd.Series([1.5, "2,5", None], dtype=object))
    np.testing.assert_allclose(mixed.values, [1.5, 2.5, np.nan])
    assert parse_numeric(pd.Series([None, ""], dtype=object)).values.isna().all()
    assert parse_numeric(pd.Series(["1.2E+03"])).values.tolist() == [1200.0]


# --- parse_dates -------------------------------------------------------------------


def test_parse_dates_infers_order_from_values_above_twelve():
    day_first = parse_dates(pd.Series(["03/04/2024", "25/04/2024"]))
    assert day_first.values.tolist() == days("2024-04-03", "2024-04-25")
    assert day_first.dayfirst is True and day_first.warnings == ()
    month_first = parse_dates(pd.Series(["04/03/2024", "04/25/2024"]))
    assert month_first.values.tolist() == days("2024-04-03", "2024-04-25")
    assert month_first.dayfirst is False and month_first.warnings == ()


def test_parse_dates_ambiguous_column_warns_and_honours_dayfirst():
    raw = pd.Series(["03/04/2024", "05/06/2024", "07/07/2024"])
    default = parse_dates(raw)
    assert default.values.tolist() == days("2024-04-03", "2024-06-05", "2024-07-07")
    assert len(default.warnings) == 1 and "assumed day-first" in default.warnings[0]
    assert "2 value(s)" in default.warnings[0]  # 07/07 reads the same either way
    month_first = parse_dates(raw, dayfirst=False)
    assert month_first.values.tolist() == days("2024-03-04", "2024-05-06", "2024-07-07")
    assert "dayfirst=False" in month_first.warnings[0]
    assert parse_dates(raw, dayfirst=True).values.tolist() == default.values.tolist()


def test_parse_dates_evidence_beats_contradicting_dayfirst_and_mixed_orders_warn():
    contradicted = parse_dates(pd.Series(["25/04/2024", "03/04/2024"]), dayfirst=False)
    assert contradicted.values.tolist() == days("2024-04-25", "2024-04-03")
    assert "ignored" in contradicted.warnings[0]
    mixed = parse_dates(pd.Series(["25/04/2024", "04/26/2024", "03/04/24"]))
    assert mixed.values.tolist() == days("2024-04-25", "2024-04-26", "2024-04-03")
    assert "mixes day-first" in mixed.warnings[0]


def test_parse_dates_mixed_formats_and_rejects():
    raw = pd.Series(
        [
            "2024-03-01", "2024-03-02T23:30:00+03:00", "3 Mar 2024", "March 4, 2024 2:30 PM",
            "05/03/2024", "20240306", "45358", pd.Timestamp("2024-03-08 17:45"),
            dt.date(2024, 3, 9), "13/03/24", "2024 03 14", "Fri, 15 March 2024",
            "", "bad", "5", "01/01/1900", "31/02/2024", None, "Mar-24", "4 March",
        ],
        dtype=object,
    )
    parsed = parse_dates(raw)
    expected = [
        *days("2024-03-01", "2024-03-02", "2024-03-03", "2024-03-04", "2024-03-05", "2024-03-06",
              "2024-03-07", "2024-03-08", "2024-03-09", "2024-03-13", "2024-03-14", "2024-03-15"),
        *[pd.NaT] * 8,  # incomplete dates are rejected, never completed with today's year
    ]
    assert parsed.values.dtype == "datetime64[ns]"
    assert parsed.values.tolist() == expected
    assert parsed.warnings == ()
    typed = parse_dates(pd.Series(pd.to_datetime(["2024-03-01 10:00"]).tz_localize("Africa/Nairobi")))
    assert typed.values.tolist() == days("2024-03-01")


def test_parse_bool():
    raw = pd.Series(["yes", "", "0", "Easter promo", None, 1, 0.0, "regular", "N", "TRUE", "x", "0,00"], dtype=object)
    assert parse_bool(raw).tolist() == [True, False, False, True, False, True, False, False, False, True, True, False]


# --- headers and reading -------------------------------------------------------------


def test_detect_columns_from_synonyms():
    columns = ["Item Name", "Sale Date", "Unit Price (KSh)", "QTY", "Line_Total", "COGS", "On Sale", "Discount %", "Stock On Hand"]
    assert detect_columns(columns, SALES_SYNONYMS) == {
        PRODUCT_ID: "Item Name",
        DATE: "Sale Date",
        PRICE: "Unit Price (KSh)",
        UNITS: "QTY",
        REVENUE: "Line_Total",
        UNIT_COST: "COGS",
        PROMO_FLAG: "On Sale",
        DISCOUNT_PCT: "Discount %",
        INVENTORY: "Stock On Hand",
    }
    assert normalize_header("Unit Price (KSh)") == normalize_header("unit_price") == "unitprice"


def test_column_map_overrides_detection_in_either_orientation():
    columns = ["Product", "SKU", "Tarehe", "Bei", "Idadi"]
    mapping = detect_columns(
        columns, SALES_SYNONYMS, {"product_id": "SKU", "date": "Tarehe", "Bei": "price", "qty": "Idadi"}
    )
    assert mapping[PRODUCT_ID] == "SKU" and mapping[DATE] == "Tarehe"
    assert mapping[PRICE] == "Bei" and mapping[UNITS] == "Idadi"
    with pytest.raises(ValueError, match="column_map entry"):
        detect_columns(columns, SALES_SYNONYMS, {"colour": "Bei"})
    with pytest.raises(ValueError, match="more than one field"):
        detect_columns(columns, SALES_SYNONYMS, {"price": "Bei", "revenue": "Bei"})


def test_read_table_handles_encodings_delimiters_title_lines_and_bad_lines():
    utf16 = "Date\tSKU\tSelling Price\tUnits Sold\n2024-03-01\tA-1\t10.5\t3\n".encode("utf-16")
    frame, malformed = read_table(utf16, header_hints=["date", "sku", "units_sold"])
    assert list(frame.columns) == ["Date", "SKU", "Selling Price", "Units Sold"]
    assert frame.iloc[0].tolist() == ["2024-03-01", "A-1", "10.5", "3"] and malformed == 0

    ragged = (
        "date,product,price,units\r\n"
        "2024-03-01,Bread, white,50,2\r\n"  # unquoted comma: cannot be aligned
        "2024-03-02,Milk,60,1\r\n"
        "\r\n"
        "2024-03-03,Eggs,20,1,,\r\n"  # trailing delimiters are harmless
        "2024-03-04,Salt\r\n"  # short rows are padded
    )
    frame, malformed = read_table(ragged)
    assert malformed == 1
    assert frame.values.tolist() == [
        ["2024-03-02", "Milk", "60", "1"],
        ["2024-03-03", "Eggs", "20", "1"],
        ["2024-03-04", "Salt", "", ""],
    ]
    result = ingest_sales(ragged)
    assert result.report.dropped["malformed_row"] == 1 and result.report.rows_read == 4
    assert any("malformed" in w for w in result.report.warnings)

    duplicated_header = read_table("Price,Price,,Date\n1,2,3,2024-03-01\n")[0]
    assert list(duplicated_header.columns) == ["Price", "Price.1", "Unnamed: 2", "Date"]


def test_ingest_sales_messy_csv_end_to_end():
    result = ingest_sales(MESSY_CSV)
    report = result.report
    assert report.rows_read == 13 and report.rows_used == 5
    assert report.dropped == {
        "missing_product": 1,
        "unparseable_date": 1,
        "missing_units": 1,
        "negative_units_returns": 1,
        "zero_units": 1,
        "missing_price": 1,
        "non_positive_price": 1,
        "duplicate_row": 1,
        "malformed_row": 0,
    }
    assert set(report.dropped) == set(SALES_DROP_REASONS)
    assert report.rows_read == report.rows_used + sum(report.dropped.values())
    assert report.column_mapping == {
        PRODUCT_ID: "Item Name",
        DATE: "Sale Date",
        PRICE: "Unit Price (KSh)",
        UNITS: "Qty",
        REVENUE: "Line Total",
        PROMO_FLAG: "Promo",
    }
    assert (report.date_min, report.date_max) == (pd.Timestamp("2024-03-01"), pd.Timestamp("2024-03-20"))
    assert report.n_products == 2
    assert len(report.warnings) == 1 and "duplicate" in report.warnings[0]

    tx = result.transactions
    assert tx[PRODUCT_ID].tolist() == ["sugar 1kg", "sugar 1kg", "unga 2kg", "unga 2kg", "unga 2kg"]
    assert tx[SOURCE_ROW].tolist() == [4, 12, 0, 2, 3]
    assert tx[PRICE].tolist() == [180.0, 200.0, 1234.5, 1200.0, 1234.5]  # 1200 derived from revenue
    assert tx[REVENUE].tolist() == [540.0, 400.0, 2469.0, 1200.0, 1234.5]  # 540 derived from price
    assert tx[PRODUCT_LABEL].tolist()[3] == "unga 2KG"
    assert tx[PROMO_FLAG].tolist() == [False, False, False, True, False]
    assert set(tx[BUSINESS_ID]) == set(tx[LOCATION_ID]) == {"default"}

    panel = result.panel
    assert list(panel.columns[:4]) == list(PANEL_KEYS)
    unga = panel[panel[PRODUCT_ID] == "unga 2kg"].reset_index(drop=True)
    assert unga[DATE].tolist() == days("2024-03-01", "2024-03-02")
    assert unga[UNITS].tolist() == [2.0, 2.0]
    assert unga[REVENUE].tolist() == [2469.0, 2434.5]
    assert unga[PRICE].tolist() == [1234.5, 1217.25]
    assert unga[N_TRANSACTIONS].tolist() == [1, 2] and unga[N_PRICES].tolist() == [1, 2]
    assert unga[PROMO_FLAG].tolist() == [False, True]
    assert len(panel) == 4  # no zero-sales days are invented

    kept = ingest_sales(MESSY_CSV, drop_duplicates=False)
    assert kept.report.dropped["duplicate_row"] == 0 and kept.report.rows_used == 6
    assert kept.panel.loc[0, UNITS] == 3.0 and kept.panel.loc[2, UNITS] == 4.0  # sugar 03-13, unga 03-01
    assert not any("duplicate" in w for w in kept.report.warnings)


def test_ingest_sales_semicolon_european_bytes_with_title_line():
    text = (
        "Rapport des ventes\r\n"
        "Date;Product;Price;Qty;Total\r\n"
        "01.03.2024;Café Touba;1.250,00 €;2;2.500,00\r\n"
        "02.03.2024;Café Touba;1.100,50;1;\r\n"
        "02.03.2024;Pain;250;4;1.000\r\n"
    )
    result = ingest_sales(text.encode("cp1252"))
    panel = result.panel.set_index([PRODUCT_ID, DATE])
    assert panel.loc[("café touba", pd.Timestamp("2024-03-01")), PRICE] == 1250.0
    assert panel.loc[("café touba", pd.Timestamp("2024-03-02")), REVENUE] == 1100.5
    assert panel.loc[("pain", pd.Timestamp("2024-03-02")), REVENUE] == 1000.0
    assert result.report.rows_used == 3
    assert any("assumed day-first" in w for w in result.report.warnings)


def test_ingest_sales_dataframe_source_with_typed_columns_and_location_column():
    df = pd.DataFrame(
        {
            "SKU": [1001.0, 1001.0, np.nan, 1002.0],
            "Order Date": pd.to_datetime(["2024-05-01 10:00", "2024-05-01 15:30", "2024-05-02 09:00", "2024-05-02 12:00"]),
            "Selling Price": [10.0, 12.0, 11.0, 5.0],
            "Quantity": [1, 2, 1, 4],
            "Branch": ["Westlands", "Westlands", "CBD", " cbd "],
        }
    )
    result = ingest_sales(df, {"location_id": "Branch"}, business_id="shop-7")
    assert result.report.dropped["missing_product"] == 1
    assert result.report.column_mapping[LOCATION_ID] == "Branch"
    panel = result.panel
    assert panel[[BUSINESS_ID, PRODUCT_ID, LOCATION_ID]].values.tolist() == [
        ["shop-7", "1001", "westlands"],
        ["shop-7", "1002", "cbd"],
    ]
    assert panel[PRICE].round(4).tolist() == [11.3333, 5.0]


def test_ingest_sales_reads_file_paths(tmp_path):
    path = tmp_path / "sales.csv"
    path.write_text("date,product,price,units\n2024-03-01,bread,50,2\n", encoding="utf-8")
    assert ingest_sales(path).report.rows_used == 1
    assert ingest_sales(str(path)).report.rows_used == 1


def test_ingest_sales_discount_cost_and_inventory_columns():
    csv = (
        "date,product,price,qty,discount,cost,stock\n"
        "2024-03-01,a,100,1,10%,60,12 pcs\n"
        "2024-03-02,a,90,2,0,,10\n"
        "2024-03-03,a,100,1,,70,\n"
    )
    tx = ingest_sales(csv).transactions
    np.testing.assert_allclose(tx[DISCOUNT_PCT], [0.1, 0.0, np.nan])
    np.testing.assert_allclose(tx[UNIT_COST], [60, np.nan, 70])
    np.testing.assert_allclose(tx[INVENTORY], [12, 10, np.nan])

    fractions = ingest_sales("date,product,price,qty,discount_pct\n2024-03-01,a,100,1,0.25\n").transactions
    assert fractions[DISCOUNT_PCT].tolist() == [0.25]
    amounts = ingest_sales(
        "date,product,price,qty,total,discount\n2024-03-01,a,850,1,850,150\n2024-03-02,a,1000,1,1000,0\n"
    )
    np.testing.assert_allclose(amounts.transactions[DISCOUNT_PCT], [0.15, 0.0])
    assert any("amounts, not percentages" in w for w in amounts.report.warnings)


def test_ingest_sales_warns_on_flat_prices_and_inconsistent_revenue():
    csv = (
        "date,product,price,qty,total\n"
        "2024-03-01,flat,50,1,50\n"
        "2024-03-02,flat,50,2,100\n"
        "2024-03-01,varied,10,1,10\n"
        "2024-03-02,varied,12,1,15\n"
    )
    warnings = ingest_sales(csv).report.warnings
    assert any("fewer than 2 distinct prices" in w and "flat" in w and "varied" not in w for w in warnings)
    assert any("1 row(s) have revenue differing" in w for w in warnings)


def test_ingest_sales_requires_core_columns():
    with pytest.raises(ValueError, match="units"):
        ingest_sales("date,product,price\n2024-03-01,a,1\n")
    with pytest.raises(ValueError, match="price or a revenue"):
        ingest_sales("date,product,qty\n2024-03-01,a,1\n")


# --- aggregation and zero-day filling ------------------------------------------------


def test_to_daily_panel_aggregation_math():
    tx = pd.DataFrame(
        {
            PRODUCT_ID: ["a", "a", "a", "a", "b"],
            DATE: ["2024-03-01"] * 4 + ["2024-03-05"],
            PRICE: [100.0, 90.0, 100.0, 100.0, 5.0],
            UNITS: [2.0, 1.0, 1.0, 1.0, 3.0],
            UNIT_COST: [60.0, np.nan, 70.0, 64.0, 2.0],
            PROMO_FLAG: [False, True, False, False, False],
            DISCOUNT_PCT: [0.0, 0.1, np.nan, 0.0, np.nan],
            INVENTORY: [10.0, 9.0, np.nan, 7.0, 1.0],
        }
    )
    panel = to_daily_panel(tx)
    a = panel.iloc[0]
    assert (a[BUSINESS_ID], a[LOCATION_ID]) == ("default", "default")
    assert a[UNITS] == 5 and a[REVENUE] == 490 and a[PRICE] == pytest.approx(98.0)
    assert (a[N_TRANSACTIONS], a[N_PRICES], a[PRICE_MIN], a[PRICE_MAX]) == (4, 2, 90.0, 100.0)
    assert a[UNIT_COST] == pytest.approx((60 * 2 + 70 + 64) / 4)  # units-weighted, missing cost skipped
    assert a[PROMO_FLAG] and a[DISCOUNT_PCT] == 0.1 and a[INVENTORY] == 10
    assert a[AVAILABLE_AT] == pd.Timestamp("2024-03-02")
    assert panel[DATE].tolist() == days("2024-03-01", "2024-03-05")
    assert panel.iloc[1][PRICE] == 5.0 and pd.isna(panel.iloc[1][DISCOUNT_PCT])


def _gappy_panel() -> pd.DataFrame:
    rows = [
        ("a", "2024-03-01", 100.0, 2.0, False),
        ("a", "2024-03-04", 80.0, 5.0, True),
        ("a", "2024-03-06", 100.0, 1.0, False),
        ("a", "2024-03-08", 100.0, 0.0, False),  # an exported zero row, not a sale
        ("b", "2024-03-03", 50.0, 1.0, False),
    ]
    tx = pd.DataFrame(rows, columns=[PRODUCT_ID, DATE, PRICE, UNITS, PROMO_FLAG])
    return to_daily_panel(tx)


def test_fill_zero_sales_days_stays_inside_sales_range_and_uses_regular_price():
    filled = fill_zero_sales_days(_gappy_panel())
    a = filled[filled[PRODUCT_ID] == "a"].set_index(DATE)
    assert a.index.tolist() == days(
        "2024-03-01", "2024-03-02", "2024-03-03", "2024-03-04", "2024-03-05", "2024-03-06", "2024-03-08"
    )  # 03-07 lies after the last sale (03-06), so it is not added
    imputed = a[a[IMPUTED_ZERO]]
    assert imputed.index.tolist() == days("2024-03-02", "2024-03-03", "2024-03-05")
    assert imputed[PRICE].tolist() == [100.0, 100.0, 100.0]  # 03-05 skips the 03-04 promo price
    assert (imputed[UNITS] == 0).all() and (imputed[REVENUE] == 0).all()
    assert (imputed[N_TRANSACTIONS] == 0).all() and filled[N_TRANSACTIONS].dtype == "int64"
    assert not imputed[PROMO_FLAG].any() and filled[PROMO_FLAG].dtype == bool
    assert imputed[AVAILABLE_AT].tolist() == days("2024-03-03", "2024-03-04", "2024-03-06")
    assert imputed[PRICE_MIN].isna().all()
    assert filled[filled[PRODUCT_ID] == "b"].shape[0] == 1
    assert filled[IMPUTED_ZERO].sum() == 3
    pd.testing.assert_frame_equal(fill_zero_sales_days(filled), filled)  # idempotent

    without_promo = fill_zero_sales_days(_gappy_panel().drop(columns=PROMO_FLAG))
    assert without_promo.set_index([PRODUCT_ID, DATE]).loc[("a", pd.Timestamp("2024-03-05")), PRICE] == 80.0


def test_fill_zero_sales_days_after_promotion_detection():
    panel = detect_promotions(_gappy_panel())
    filled = fill_zero_sales_days(panel)
    added = filled[filled[IMPUTED_ZERO]].set_index(DATE)
    assert added[REGULAR_PRICE].tolist() == [100.0, 100.0, 100.0]
    assert added[DISCOUNT_PCT].tolist() == [0.0, 0.0, 0.0]


# --- inventory -----------------------------------------------------------------------

INVENTORY_CSV = """Item,Count Date,Stock On Hand,Reorder Level
Unga 2kg,01/03/2024,12 pcs,5
Unga 2kg,01/03/2024,10,5
unga 2KG,02/03/2024,0,5
Sugar 1kg,13/03/2024,-2,3
Sugar 1kg,13/03/2024,-2,3
,14/03/2024,4,3
Sugar 1kg,32/03/2024,4,3
Sugar 1kg,15/03/2024,n/a,3
"""


def test_ingest_inventory_shares_parsing_and_combines_counts():
    counts, report = ingest_inventory(INVENTORY_CSV, location_id="cbd")
    assert counts[[PRODUCT_ID, DATE, INVENTORY, REORDER_POINT]].values.tolist() == [
        ["sugar 1kg", pd.Timestamp("2024-03-13"), -2.0, 3.0],
        ["unga 2kg", pd.Timestamp("2024-03-01"), 12.0, 5.0],
        ["unga 2kg", pd.Timestamp("2024-03-02"), 0.0, 5.0],
    ]
    assert set(counts[LOCATION_ID]) == {"cbd"}
    assert counts[AVAILABLE_AT].tolist() == days("2024-03-14", "2024-03-02", "2024-03-03")
    assert report.rows_read == 8 and report.rows_used == 4
    assert report.dropped == {
        "missing_product": 1,
        "unparseable_date": 1,
        "missing_inventory": 1,
        "duplicate_row": 1,
        "malformed_row": 0,
    }
    assert report.column_mapping == {
        PRODUCT_ID: "Item",
        DATE: "Count Date",
        INVENTORY: "Stock On Hand",
        REORDER_POINT: "Reorder Level",
    }
    assert report.n_products == 2
    assert any("negative stock" in w for w in report.warnings)
    assert any("kept the highest" in w for w in report.warnings)
