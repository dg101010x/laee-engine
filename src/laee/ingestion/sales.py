"""Sales ingestion: messy exports to clean transactions and a daily product panel.

Covers spec §84 steps 1-2 (ingest, validate) and builds the §4 panel keyed by
(business_id, product_id, location_id, date).

:func:`ingest_sales` takes a DataFrame, CSV text, CSV bytes or a file path. Text may use
any common encoding and delimiter, with title lines above the header. It maps headers
to canonical columns by synonym and reads numbers written with currency symbols or codes,
either thousands style (1,234.50 or 1.234,50), spaces or parentheses. It reads dates in
mixed formats, derives a missing price or revenue, and drops, and counts, the rows it
cannot use. :class:`IngestReport` is the audit trail.

Ingestion never invents zero-sales days. A day without rows may be a day without data
(shop closed, export gap) as easily as a day without demand, so
:func:`fill_zero_sales_days` adds such days only when asked, flagged ``imputed_zero``.

:mod:`laee.ingestion.inventory` shares the parsing helpers (:func:`read_table`,
:func:`detect_columns`, :func:`parse_numeric`, :func:`parse_dates`, :func:`parse_bool`).
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import math
import os
import re
import warnings
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype

from laee.features.temporal import (
    IMPUTED_ZERO,
    N_PRICES,
    N_TRANSACTIONS,
    PRICE_MAX,
    PRICE_MIN,
    bool_column,
    ensure_unique_days,
    float_column,
    group_positions,
    map_unique,
    normalize_dates,
    require_columns,
    series_keys,
)
from laee.normalization.prices import normalize_product_ids
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

PRODUCT_LABEL = "product_label"
SOURCE_ROW = "source_row"

Source = pd.DataFrame | str | bytes | os.PathLike

# Header synonyms per canonical field, in priority order (first match wins).
SALES_SYNONYMS: dict[str, tuple[str, ...]] = {
    PRODUCT_ID: (
        "product_id", "product", "sku", "item", "item_name", "product_name", "description",
        "item_id", "product_code", "item_code", "item_description", "product_description",
    ),
    DATE: (
        "date", "day", "order_date", "sale_date", "timestamp", "datetime",
        "sales_date", "transaction_date", "txn_date", "invoice_date", "receipt_date", "date_time",
    ),
    PRICE: ("price", "unit_price", "selling_price", "price_per_unit", "sale_price", "unit_selling_price"),
    UNITS: ("units", "qty", "quantity", "units_sold", "qty_sold", "quantity_sold", "pcs"),
    REVENUE: (
        "revenue", "sales", "amount", "total", "line_total",
        "net_sales", "total_sales", "sales_amount", "net_amount", "total_amount", "line_amount", "gross_sales",
    ),
    UNIT_COST: ("unit_cost", "cost", "cogs", "cost_price", "buying_price", "purchase_price"),
    PROMO_FLAG: ("promo_flag", "promo", "promotion", "on_sale", "is_promo", "on_promotion"),
    DISCOUNT_PCT: ("discount_pct", "discount", "discount_percent", "discount_rate", "discount_percentage"),
    INVENTORY: (
        "inventory", "stock", "on_hand", "stock_on_hand",
        "stock_level", "qty_on_hand", "quantity_on_hand", "closing_stock",
    ),
    BUSINESS_ID: ("business_id",),
    LOCATION_ID: ("location_id",),
}

SALES_DROP_REASONS = (
    "missing_product",
    "unparseable_date",
    "missing_units",
    "negative_units_returns",
    "zero_units",
    "missing_price",
    "non_positive_price",
    "duplicate_row",
    "malformed_row",
)

_MISSING_TOKENS = frozenset({"", "na", "n/a", "nan", "null", "none", "nil", "-", "--", "—", "#n/a", "#value!"})

_CURRENCY_CODES = (
    "kshs", "ksh", "kes", "ngn", "ghs", "gh₵", "zar", "usd", "us", "eur", "gbp",
    "ugx", "tzs", "rwf", "xof", "xaf", "cfa", "egp", "shs", "sh", "r", "n",
)
_CURRENCY_CODE_RE = re.compile(
    r"(?<![a-z])(?:"
    + "|".join(re.escape(code) for code in sorted(_CURRENCY_CODES, key=len, reverse=True))
    + r")\.?(?![a-z])",
    re.IGNORECASE,
)
_CURRENCY_SYMBOL_RE = re.compile("[$€£₦₵¢¥₹]")
_SHILLING_SUFFIX_RE = re.compile(r"\s*/\s*[=-]\s*$")  # East African "500/=" and "500/-"
_UNIT_SUFFIX_RE = re.compile(r"[a-z]+\.?$", re.IGNORECASE)  # "3pcs", "2kg"
_NUMBER_RE = re.compile(r"(?=.*\d)[\d.,]+")
_SCIENTIFIC_RE = re.compile(r"\d+(?:\.\d+)?e[-+]?\d+", re.IGNORECASE)
_SPREAD_MARGIN = 0.1  # decades of log10 spread needed to prefer one separator reading

# Year-first dates never reach dateutil, which applies dayfirst to them ("2024 03 01" -> Jan 3).
_ISO_DATE_RE = re.compile(r"(\d{4})[-/. ](\d{1,2})[-/. ](\d{1,2})(?:$|[T\s,])")
_NUMERIC_DATE_RE = re.compile(r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{4}|\d{2})(?:$|[T\s,])")
_COMPACT_DATE_RE = re.compile(r"\d{8}")
_EXCEL_SERIAL_RE = re.compile(r"\d{5}(?:\.\d+)?")
_MONTH_NAME_RE = re.compile(r"(?<![a-z])(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", re.IGNORECASE)
_YEAR_RE = re.compile(r"(?<!\d)\d{4}(?!\d)")
_EXCEL_EPOCH = pd.Timestamp("1899-12-30")
_YEAR_RANGE = (1970, 2100)

_DELIMITERS = (",", ";", "\t", "|")
_SNIFF_LINES = 50
_LINE_BREAK_RE = re.compile(r"\r\n|\r|\n")
_TRUE_TOKENS = frozenset({"1", "true", "t", "yes", "y", "x", "on", "promo", "promotion", "sale", "on sale", "on_sale"})
_FALSE_TOKENS = _MISSING_TOKENS | {"0", "false", "f", "no", "n", "off", "regular", "normal", "standard", "full price"}


@dataclass
class IngestReport:
    """Audit trail of one ingestion run.

    ``rows_read`` counts the data rows in the source, malformed lines included.
    ``rows_used`` counts those that passed validation. ``dropped`` maps every drop reason
    to its count, so ``rows_read == rows_used + sum(dropped.values())``.
    ``column_mapping`` maps canonical fields to source columns.
    """

    rows_read: int
    rows_used: int
    dropped: dict[str, int]
    column_mapping: dict[str, str]
    date_min: pd.Timestamp | None
    date_max: pd.Timestamp | None
    n_products: int
    warnings: list[str] = field(default_factory=list)


@dataclass
class IngestResult:
    """Clean transactions, their daily panel and the ingestion report."""

    transactions: pd.DataFrame
    panel: pd.DataFrame
    report: IngestReport


@dataclass(frozen=True)
class NumericParse:
    """Result of :func:`parse_numeric`.

    ``decimal`` is the column's decimal separator, used for values whose separators do
    not settle it themselves. ``ambiguous_example`` is a value that could not be read
    either way with confidence; it is None when the column was unambiguous.
    """

    values: pd.Series
    decimal: str
    n_invalid: int
    percent: bool
    ambiguous_example: str | None


@dataclass(frozen=True)
class DateParse:
    """Result of :func:`parse_dates`: days (NaT when unparseable), order used, warnings."""

    values: pd.Series
    dayfirst: bool | None
    warnings: tuple[str, ...]


def ingest_sales(
    source: Source,
    column_map: Mapping[str, str] | None = None,
    *,
    business_id: str = "default",
    location_id: str = "default",
    dayfirst: bool | None = None,
    drop_duplicates: bool = True,
) -> IngestResult:
    """Ingest a sales export into transactions, a daily panel and an audit report.

    Headers are matched to fields through :data:`SALES_SYNONYMS`, ignoring case, spaces,
    punctuation and bracketed suffixes such as "(KSh)". ``column_map`` overrides detection.
    It maps fields to source columns, e.g. ``{"units": "Qty Sold"}``; the reverse
    orientation is also accepted. Required fields are product, date and units, plus a price
    or a revenue column. A missing price is derived as revenue / units and a missing
    revenue as price x units.

    Rows are dropped for the first failing reason in :data:`SALES_DROP_REASONS` order.
    Negative units, or negative revenue, are returns. Exact duplicate rows are dropped.
    An export with dates but no time or receipt id cannot tell a double-exported line from
    a genuine repeat sale, so a warning is recorded; ``drop_duplicates=False`` keeps them.
    Product ids are normalised with
    :func:`laee.normalization.normalize_product_ids`; the raw label is kept as
    ``product_label``. Business and location ids come from ``business_id`` /
    ``location_id`` unless ``column_map`` points them at columns.

    Ambiguous day/month dates are settled by any value above 12; failing that,
    ``dayfirst`` decides (day-first by default, the KE/NG/GH/ZA/GB convention) and a
    warning is recorded.
    """
    raw, malformed = read_table(source, header_hints=_header_hints(SALES_SYNONYMS, column_map))
    mapping = detect_columns(raw.columns, SALES_SYNONYMS, column_map)
    _require_fields(mapping, raw.columns, (PRODUCT_ID, DATE, UNITS))
    if PRICE not in mapping and REVENUE not in mapping:
        raise ValueError(f"need a price or a revenue column; found columns {list(raw.columns)}")
    notes: list[str] = []

    labels = _labels(raw[mapping[PRODUCT_ID]])
    product = normalize_product_ids(labels)
    dates = parse_dates(raw[mapping[DATE]], dayfirst=dayfirst, name=mapping[DATE])
    notes.extend(dates.warnings)
    units = _numeric_field(raw, mapping, UNITS, notes, strip_units=True).values
    empty = pd.Series(np.nan, index=raw.index)
    price_given = _numeric_field(raw, mapping, PRICE, notes).values if PRICE in mapping else empty
    revenue_given = _numeric_field(raw, mapping, REVENUE, notes).values if REVENUE in mapping else empty
    derive = price_given.isna() & revenue_given.notna() & units.gt(0)
    price = price_given.mask(derive, revenue_given / units)
    revenue = revenue_given.fillna(price * units)

    reason = pd.Series(None, index=raw.index, dtype=object)
    for name, failed in (
        ("missing_product", product.isna()),
        ("unparseable_date", dates.values.isna()),
        ("missing_units", units.isna()),
        ("negative_units_returns", units.lt(0) | revenue.lt(0)),
        ("zero_units", units.eq(0)),
        ("missing_price", price.isna()),
        ("non_positive_price", price.le(0)),
    ):
        reason = reason.mask(reason.isna() & failed, name)
    if drop_duplicates:
        reason = reason.mask(reason.isna() & _exact_duplicates(raw), "duplicate_row")
    keep = reason.isna()

    columns: dict[str, object] = {
        BUSINESS_ID: _row_ids(raw, mapping, BUSINESS_ID, business_id),
        PRODUCT_ID: product,
        LOCATION_ID: _row_ids(raw, mapping, LOCATION_ID, location_id),
        DATE: dates.values,
        PRICE: price,
        UNITS: units,
        REVENUE: revenue,
    }
    if UNIT_COST in mapping:
        cost = _numeric_field(raw, mapping, UNIT_COST, notes).values
        if cost.lt(0).any():
            notes.append(f"column '{mapping[UNIT_COST]}': {int(cost.lt(0).sum())} negative cost(s) ignored")
            cost = cost.mask(cost.lt(0))
        columns[UNIT_COST] = cost
        both = keep & cost.notna()
        if both.any() and (cost[both] > price[both]).mean() > 0.5:
            notes.append(
                f"column '{mapping[UNIT_COST]}': cost exceeds price on most rows; check that it is "
                "a per-unit cost and not a line total"
            )
    if PROMO_FLAG in mapping:
        columns[PROMO_FLAG] = parse_bool(raw[mapping[PROMO_FLAG]])
    if DISCOUNT_PCT in mapping:
        parsed = _numeric_field(raw, mapping, DISCOUNT_PCT, notes)
        columns[DISCOUNT_PCT] = _discount_fraction(parsed, revenue, mapping[DISCOUNT_PCT], notes)
    if INVENTORY in mapping:
        columns[INVENTORY] = _numeric_field(raw, mapping, INVENTORY, notes, strip_units=True).values
    columns[PRODUCT_LABEL] = labels
    columns[SOURCE_ROW] = np.arange(len(raw))
    table = pd.DataFrame(columns, index=raw.index)
    transactions = (
        table.loc[keep.to_numpy()]
        .sort_values([*PANEL_KEYS, SOURCE_ROW], kind="stable")
        .reset_index(drop=True)
    )

    if PRICE in mapping and REVENUE in mapping:
        expected = price_given * units
        mismatch = keep & price_given.notna() & revenue_given.notna()
        mismatch &= (revenue_given - expected).abs() > 0.01 * expected.abs() + 0.01
        if mismatch.any():
            notes.append(
                f"{int(mismatch.sum())} row(s) have revenue differing from price x units by more "
                "than 1% (line discounts, or a receipt total?); the panel price is revenue / units"
            )
    n_duplicates = int((reason == "duplicate_row").sum())
    if n_duplicates and set(raw.columns) <= set(mapping.values()):
        notes.append(
            f"{n_duplicates} exact duplicate row(s) dropped; the export has no receipt or "
            "transaction id column, so some may be genuine repeat sales (drop_duplicates=False "
            "keeps them)"
        )
    if malformed:
        notes.append(f"{malformed} malformed line(s) with more fields than the header were skipped")
    if transactions.empty:
        notes.append("no usable rows")
    else:
        flat = _low_variation_note(transactions)
        if flat:
            notes.append(flat)

    dropped = {name: int((reason == name).sum()) for name in SALES_DROP_REASONS}
    dropped["malformed_row"] = malformed
    date_min, date_max = _date_bounds(transactions)
    report = IngestReport(
        rows_read=len(raw) + malformed,
        rows_used=len(transactions),
        dropped=dropped,
        column_mapping=dict(mapping),
        date_min=date_min,
        date_max=date_max,
        n_products=int(transactions[PRODUCT_ID].nunique()),
        warnings=notes,
    )
    return IngestResult(transactions=transactions, panel=to_daily_panel(transactions), report=report)


def to_daily_panel(transactions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transactions to one row per (business, product, location, date).

    ``units`` and ``revenue`` are sums and ``price`` is revenue / units, the
    units-weighted average price. ``n_transactions`` counts rows and ``n_prices`` counts
    distinct transaction prices; ``price_min`` and ``price_max`` bound them. ``unit_cost``
    is the units-weighted mean over rows that have one. ``promo_flag`` is any,
    ``discount_pct`` the max and ``inventory`` the max (the most stock seen on hand that
    day). ``available_at`` is the end of the day, when the day's sales are known (spec §75).
    Missing business/location ids become "default". No zero-sales days are added.
    """
    require_columns(transactions, [PRODUCT_ID, DATE, PRICE, UNITS])
    tx = transactions.copy()
    for key in (BUSINESS_ID, LOCATION_ID):
        if key not in tx.columns:
            tx[key] = "default"
    tx[DATE] = normalize_dates(tx[DATE])
    tx[PRICE] = float_column(tx, PRICE)
    tx[UNITS] = float_column(tx, UNITS)
    tx[REVENUE] = float_column(tx, REVENUE) if REVENUE in tx.columns else tx[PRICE] * tx[UNITS]
    tx["_price_key"] = tx[PRICE].round(6)
    named: dict[str, tuple[str, str]] = {
        UNITS: (UNITS, "sum"),
        REVENUE: (REVENUE, "sum"),
        N_TRANSACTIONS: (UNITS, "size"),
        N_PRICES: ("_price_key", "nunique"),
        PRICE_MIN: (PRICE, "min"),
        PRICE_MAX: (PRICE, "max"),
    }
    optional: list[str] = []
    if UNIT_COST in tx.columns:
        cost = float_column(tx, UNIT_COST)
        has_cost = np.isfinite(cost)
        tx["_cost_value"] = np.where(has_cost, cost * tx[UNITS].to_numpy(), 0.0)
        tx["_cost_units"] = np.where(has_cost, tx[UNITS].to_numpy(), 0.0)
        named["_cost_value"] = ("_cost_value", "sum")
        named["_cost_units"] = ("_cost_units", "sum")
        optional.append(UNIT_COST)
    if PROMO_FLAG in tx.columns:
        tx[PROMO_FLAG] = bool_column(tx, PROMO_FLAG)
        named[PROMO_FLAG] = (PROMO_FLAG, "any")
        optional.append(PROMO_FLAG)
    for col in (DISCOUNT_PCT, INVENTORY):
        if col in tx.columns:
            tx[col] = float_column(tx, col)
            named[col] = (col, "max")
            optional.append(col)
    panel = tx.groupby(list(PANEL_KEYS), sort=True, dropna=False).agg(**named).reset_index()
    panel[PRICE] = (panel[REVENUE] / panel[UNITS]).where(panel[UNITS] > 0)
    if UNIT_COST in optional:
        cost_units = panel.pop("_cost_units")
        panel[UNIT_COST] = (panel.pop("_cost_value") / cost_units).where(cost_units > 0)
    panel[AVAILABLE_AT] = panel[DATE] + pd.Timedelta(days=1)
    order = [*PANEL_KEYS, PRICE, UNITS, REVENUE, N_TRANSACTIONS, N_PRICES, PRICE_MIN, PRICE_MAX]
    return panel[[*order, *optional, AVAILABLE_AT]]


def fill_zero_sales_days(panel: pd.DataFrame) -> pd.DataFrame:
    """Add explicit zero-sales rows for missing days inside each series' sales range.

    Only days between a series' first and last day with positive units are added; the
    range is never extended. Added rows have ``imputed_zero`` True (False elsewhere) and
    units, revenue, n_transactions and n_prices of 0. Their price is the last observed
    regular price: the price on the latest earlier day not flagged ``promo_flag`` when
    that column exists, else ``regular_price`` carried forward, else the last observed
    price. ``regular_price`` and ``unit_cost`` are carried forward; ``regular_price``
    falls back to that fill price before the series' first estimate. ``discount_pct`` is
    recomputed against ``regular_price`` (0 without it). Boolean flags are False,
    ``available_at`` is the end of the day and any other column is missing.
    """
    keys = series_keys(panel)
    require_columns(panel, [DATE, PRICE, UNITS])
    out = panel.copy()
    out[DATE] = normalize_dates(out[DATE])
    ensure_unique_days(out, keys)
    out[IMPUTED_ZERO] = bool_column(out, IMPUTED_ZERO)
    flag_columns = [col for col in out.columns if is_bool_dtype(out[col].dtype)]
    missing = _missing_sale_days(out, keys)
    if missing.empty:
        return out.sort_values([*keys, DATE], kind="stable").reset_index(drop=True)
    missing["_new"] = True
    combined = pd.concat([out, missing], ignore_index=True, sort=False)
    combined = combined.sort_values([*keys, DATE], kind="stable").reset_index(drop=True)
    new = bool_column(combined, "_new")
    combined = combined.drop(columns="_new")
    groups = group_positions(combined, keys)

    price = float_column(combined, PRICE)
    observed = ~bool_column(combined, IMPUTED_ZERO) & ~new & np.isfinite(price) & (price > 0)
    candidates = []
    if PROMO_FLAG in combined.columns:
        candidates.append(np.where(observed & ~bool_column(combined, PROMO_FLAG), price, np.nan))
    if REGULAR_PRICE in combined.columns:
        candidates.append(float_column(combined, REGULAR_PRICE))
    candidates.append(np.where(observed, price, np.nan))
    fill = np.full(len(combined), np.nan)
    for candidate in candidates:
        fill = np.where(np.isnan(fill), _carry_forward(candidate, groups), fill)

    combined.loc[new, PRICE] = fill[new]
    combined.loc[new, UNITS] = 0.0
    for col in (REVENUE, N_TRANSACTIONS, N_PRICES):
        if col in combined.columns:
            combined.loc[new, col] = 0
    for col in (REGULAR_PRICE, UNIT_COST):
        if col in combined.columns:
            carried = _carry_forward(float_column(combined, col), groups)
            if col == REGULAR_PRICE:  # no estimate yet (start of series): the fill price is one
                carried = np.where(np.isnan(carried), fill, carried)
            combined[col] = np.where(new, carried, float_column(combined, col))
    if DISCOUNT_PCT in combined.columns:
        if REGULAR_PRICE in combined.columns:
            with np.errstate(divide="ignore", invalid="ignore"):
                depth = np.clip(1.0 - fill / float_column(combined, REGULAR_PRICE), 0.0, 1.0)
        else:
            depth = np.zeros(len(combined))
        combined.loc[new, DISCOUNT_PCT] = depth[new]
    if AVAILABLE_AT in combined.columns:
        combined.loc[new, AVAILABLE_AT] = combined.loc[new, DATE] + pd.Timedelta(days=1)
    for col in flag_columns:
        combined[col] = bool_column(combined, col)
    combined[IMPUTED_ZERO] = combined[IMPUTED_ZERO].to_numpy() | new
    for col in (N_TRANSACTIONS, N_PRICES):
        if col in out.columns and pd.api.types.is_integer_dtype(out[col].dtype):
            combined[col] = combined[col].astype("int64")
    return combined[list(out.columns)]


def read_table(source: Source, *, header_hints: Iterable[str] = ()) -> tuple[pd.DataFrame, int]:
    """Load ``source`` as a raw table and count the malformed lines skipped.

    DataFrames are copied with string column names. Text, bytes or a file path are read
    as CSV with every cell kept as a string. The encoding (UTF-8/16 with or without BOM,
    else cp1252) and the delimiter (comma, semicolon, tab or pipe) are detected. Title
    lines above the header are skipped: the header is the first line naming at least
    two of ``header_hints``. Blank lines are ignored and short rows are padded. A row with
    more non-empty fields than the header (e.g. an unquoted "Bread, white") cannot be
    aligned, so it is skipped and counted.
    """
    if isinstance(source, pd.DataFrame):
        frame = source.copy()
        frame.columns = [str(col).strip() for col in frame.columns]
        return frame.reset_index(drop=True), 0
    text = _source_text(source)
    lines = _LINE_BREAK_RE.split(text)[:_SNIFF_LINES]
    delimiter = _sniff_delimiter(lines)
    header_line = _find_header_line(lines, delimiter, {normalize_header(h) for h in header_hints})
    offset = 0
    for _ in range(header_line):
        line_break = _LINE_BREAK_RE.search(text, offset)
        offset = len(text) if line_break is None else line_break.end()
    reader = csv.reader(io.StringIO(text[offset:]), delimiter=delimiter, skipinitialspace=True)
    header = [field.strip() for field in next(reader, [])]
    width = len(header)
    rows: list[list[str]] = []
    malformed = 0
    for fields in reader:
        count = len(fields)
        if count == width:
            rows.append(fields)
        elif count > width:
            if any(field.strip() for field in fields[width:]):
                malformed += 1
            else:
                rows.append(fields[:width])  # trailing delimiters only
        elif count > 1 or (count == 1 and fields[0].strip()):  # blank lines are skipped
            rows.append(fields + [""] * (width - count))
    return pd.DataFrame(rows, columns=_unique_names(header), dtype=str), malformed


def normalize_header(name: object) -> str:
    """Header key for synonym matching: bracketed parts removed, casefolded, alphanumerics only.

    "Unit Price (KSh)", "unit_price" and "UnitPrice" all become "unitprice".
    """
    text = re.sub(r"\(.*?\)|\[.*?\]", " ", str(name))
    return re.sub(r"[^0-9a-z]+", "", text.casefold())


def detect_columns(
    columns: Iterable[object],
    synonyms: Mapping[str, Sequence[str]],
    column_map: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Map canonical fields to source columns: ``column_map`` first, then synonyms.

    A ``column_map`` entry may read field -> column or column -> field. When both readings
    are possible, field -> column wins. Field names may be canonical names or any synonym.
    Each source column feeds at most one field.
    """
    names = [str(col) for col in columns]
    lookup = _field_lookup(synonyms)
    mapping: dict[str, str] = {}
    for key, value in (column_map or {}).items():
        target, column = _resolve_map_entry(str(key), str(value), names, lookup)
        if column in mapping.values():
            raise ValueError(f"column_map maps column {column!r} to more than one field")
        mapping[target] = column
    taken = set(mapping.values())
    by_header: dict[str, list[str]] = {}
    for name in names:
        by_header.setdefault(normalize_header(name), []).append(name)
    for target, candidates in synonyms.items():
        if target in mapping:
            continue
        for candidate in candidates:
            free = [name for name in by_header.get(normalize_header(candidate), []) if name not in taken]
            if free:
                mapping[target] = free[0]
                taken.add(free[0])
                break
    return mapping


def parse_numeric(values: pd.Series | Sequence[object], *, strip_units: bool = False) -> NumericParse:
    """Parse messy numeric text to floats (NaN when missing or unreadable).

    Handles currency symbols and codes (KSh, KES, NGN, ₦, GHS, ₵, R, ZAR, $, €, £, USD and
    more), East African "/=" suffixes, spaces and apostrophes as thousands separators,
    parentheses or a trailing minus for negatives, percent signs and scientific notation.
    With ``strip_units`` a trailing unit word ("3 pcs", "2kg") is dropped.

    Comma versus dot is resolved per value where the value settles it: both present (the
    last one is decimal), a repeated separator (thousands), or a group not of three digits
    (decimal). The remaining values, such as "1,234" or "1.250", follow the majority of
    the settled values in the column. If no value settles it, the reading that keeps the
    column's magnitudes most consistent is used; if that is a tie, "." is the decimal and
    ``ambiguous_example`` is set.
    """
    series = values if isinstance(values, pd.Series) else pd.Series(list(values))
    if is_numeric_dtype(series.dtype) and not is_bool_dtype(series.dtype):
        return NumericParse(series.astype("float64"), ".", 0, False, None)
    codes, uniques = pd.factorize(series, use_na_sentinel=True)
    uniques = list(uniques)
    if not uniques:
        return NumericParse(pd.Series(np.nan, index=series.index, dtype="float64"), ".", 0, False, None)
    parsed = np.full(len(uniques), np.nan)
    invalid = np.zeros(len(uniques), dtype=bool)
    pending: list[tuple[int, str, bool]] = []
    percent = False
    for i, raw in enumerate(uniques):
        if isinstance(raw, (bool, np.bool_)):
            invalid[i] = True
            continue
        if isinstance(raw, (int, float, np.integer, np.floating)):
            parsed[i] = float(raw)
            continue
        text = str(raw).strip()
        if text.casefold() in _MISSING_TOKENS:
            continue
        token = _numeric_token(text, strip_units)
        if token is None:
            invalid[i] = True
            continue
        cleaned, negative, has_percent = token
        percent = percent or has_percent
        pending.append((i, cleaned, negative))
    frequency = np.bincount(codes[codes >= 0], minlength=len(uniques))
    decimal, ambiguous_example = _choose_decimal(pending, frequency, parsed, uniques)
    for i, cleaned, negative in pending:
        value = _token_value(cleaned, decimal)
        if value is None:
            invalid[i] = True
        else:
            parsed[i] = -value if negative else value
    result = np.where(codes >= 0, parsed[codes], np.nan)
    return NumericParse(
        values=pd.Series(result, index=series.index, dtype="float64"),
        decimal=decimal,
        n_invalid=int(frequency[invalid].sum()),
        percent=percent,
        ambiguous_example=ambiguous_example,
    )


def parse_dates(
    values: pd.Series | Sequence[object],
    *,
    dayfirst: bool | None = None,
    name: str = DATE,
) -> DateParse:
    """Parse mixed-format dates to days (``datetime64[ns]`` at midnight, NaT if unreadable).

    Accepts ISO dates (2024-03-01, with or without a time or offset; the wall-clock date
    is kept), d/m/y or m/d/y with "/", "-" or "." separators and 2- or 4-digit years,
    YYYYMMDD, Excel serial day numbers, dates with month names ("1 Mar 2024") and
    datetime objects. Years outside 1970-2100 are rejected as typos.

    The day/month order of numeric dates comes from the data. A value with a first part
    above 12 must be day-first; one with a second part above 12 must be month-first.
    Such values always use the order valid for them. The others follow the order
    evidenced by the column. Without evidence, ``dayfirst`` decides, defaulting to
    day-first, and a warning is returned.
    """
    series = values if isinstance(values, pd.Series) else pd.Series(list(values))
    if is_datetime64_any_dtype(series.dtype):
        stamps = series
        if isinstance(stamps.dtype, pd.DatetimeTZDtype):
            stamps = stamps.dt.tz_localize(None)
        return DateParse(_within_years(stamps.dt.normalize().astype("datetime64[ns]")), None, ())
    codes, uniques = pd.factorize(series, use_na_sentinel=True)
    uniques = list(uniques)
    if not uniques:
        return DateParse(pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]"), None, ())
    stamps_by_unique = np.full(len(uniques), np.datetime64("NaT"), dtype="datetime64[ns]")
    parts: dict[int, tuple[int, int, int]] = {}
    pairs: dict[int, tuple[int, int, int]] = {}
    texts: dict[int, str] = {}
    for i, raw in enumerate(uniques):
        if isinstance(raw, (dt.date, np.datetime64)):
            stamps_by_unique[i] = _midnight(pd.Timestamp(raw))
            continue
        if isinstance(raw, (int, float, np.integer, np.floating)) and not isinstance(raw, (bool, np.bool_)):
            stamps_by_unique[i] = _numeric_date(float(raw))
            continue
        text = str(raw).strip()
        if text.casefold() in _MISSING_TOKENS:
            continue
        if match := _ISO_DATE_RE.match(text):
            parts[i] = (int(match[1]), int(match[2]), int(match[3]))
        elif match := _NUMERIC_DATE_RE.match(text):
            pairs[i] = (int(match[1]), int(match[2]), _full_year(match[3]))
        elif _COMPACT_DATE_RE.fullmatch(text) or _EXCEL_SERIAL_RE.fullmatch(text):
            stamps_by_unique[i] = _numeric_date(float(text))
        else:
            texts[i] = text
    frequency = np.bincount(codes[codes >= 0], minlength=len(uniques))
    order, notes = _resolve_day_order(pairs, uniques, frequency, dayfirst, name)
    for i, (first, second, year) in pairs.items():
        day_first = first > 12 or (second <= 12 and order)
        parts[i] = (year, second, first) if day_first else (year, first, second)
    if parts:
        valid = {i: p for i, p in parts.items() if _YEAR_RANGE[0] <= p[0] <= _YEAR_RANGE[1]}
        if valid:
            index = np.fromiter(valid, dtype=np.int64, count=len(valid))
            components = pd.DataFrame(list(valid.values()), columns=["year", "month", "day"])
            built = pd.to_datetime(components, errors="coerce")
            stamps_by_unique[index] = built.to_numpy(dtype="datetime64[ns]")
    for i, text in texts.items():
        stamps_by_unique[i] = _text_date(text, order)
    result = pd.Series(
        np.where(codes >= 0, stamps_by_unique[codes], np.datetime64("NaT")),
        index=series.index,
        dtype="datetime64[ns]",
    )
    return DateParse(_within_years(result), order if pairs else None, tuple(notes))


def parse_bool(values: pd.Series | Sequence[object]) -> pd.Series:
    """Parse promotion-style flags. Missing, 0, "no", "false" or "regular" give False.

    Other non-empty text, such as a promotion name, gives True, as does any non-zero number.
    """
    series = values if isinstance(values, pd.Series) else pd.Series(list(values))
    return map_unique(series, _flag, missing=False).astype(bool)


def _source_text(source: object) -> str:
    if isinstance(source, bytes):
        return _decode(source)
    if isinstance(source, os.PathLike) or (
        isinstance(source, str) and "\n" not in source and "\r" not in source and os.path.isfile(source)
    ):
        with open(source, "rb") as handle:
            return _decode(handle.read())
    if isinstance(source, str):
        return source.lstrip("﻿")
    raise TypeError(f"unsupported source type {type(source).__name__}; pass a DataFrame, CSV text or bytes")


def _decode(data: bytes) -> str:
    if data.startswith(b"\xef\xbb\xbf"):
        return data[3:].decode("utf-8", errors="replace")
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")


def _split_fields(line: str, delimiter: str) -> list[str]:
    return next(csv.reader([line], delimiter=delimiter), [])


def _sniff_delimiter(lines: list[str]) -> str:
    nonblank = [line for line in lines if line.strip()]
    if not nonblank:
        raise ValueError("source contains no data")
    best, best_score = ",", (0.0, 0)
    for delimiter in _DELIMITERS:
        counts = [len(_split_fields(line, delimiter)) for line in nonblank]
        modal = max(Counter(counts).items(), key=lambda item: (item[1], item[0]))[0]
        if modal < 2:
            continue
        # Prefer the delimiter that splits most lines into the same number of fields.
        score = (sum(count == modal for count in counts) / len(counts), modal)
        if score > best_score:
            best, best_score = delimiter, score
    return best


def _find_header_line(lines: list[str], delimiter: str, hints: set[str]) -> int:
    first_multi_field = None
    for number, line in enumerate(lines):
        if not line.strip():
            continue
        fields = _split_fields(line, delimiter)
        if len({normalize_header(f) for f in fields} & hints) >= 2:
            return number
        if first_multi_field is None and len(fields) > 1:
            first_multi_field = number
    return first_multi_field or 0


def _unique_names(header: list[str]) -> list[str]:
    """Header names made unique ("Price", "Price.1") with blanks named "Unnamed: i"."""
    seen: Counter[str] = Counter()
    names = []
    for position, name in enumerate(header):
        name = name or f"Unnamed: {position}"
        names.append(f"{name}.{seen[name]}" if seen[name] else name)
        seen[name] += 1
    return names


def _field_lookup(synonyms: Mapping[str, Sequence[str]]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for target, names in synonyms.items():
        for name in (target, *names):
            lookup.setdefault(normalize_header(name), target)
    return lookup


def _find_column(name: str, columns: list[str]) -> str | None:
    if name in columns:
        return name
    matches = [col for col in columns if normalize_header(col) == normalize_header(name)]
    return matches[0] if len(matches) == 1 else None


def _resolve_map_entry(key: str, value: str, columns: list[str], lookup: dict[str, str]) -> tuple[str, str]:
    for target_name, column_name in ((key, value), (value, key)):
        target, column = lookup.get(normalize_header(target_name)), _find_column(column_name, columns)
        if target is not None and column is not None:
            return target, column
    raise ValueError(
        f"column_map entry {key!r}: {value!r} must pair a field "
        f"({', '.join(dict.fromkeys(lookup.values()))}) with a source column ({', '.join(columns)})"
    )


def _require_fields(mapping: Mapping[str, str], columns: Iterable[object], required: Iterable[str]) -> None:
    missing = [name for name in required if name not in mapping]
    if missing:
        raise ValueError(
            f"could not find column(s) for {', '.join(missing)} among {[str(c) for c in columns]}; "
            f"pass column_map, e.g. {{'{missing[0]}': '<column name>'}}"
        )


def _header_hints(synonyms: Mapping[str, Sequence[str]], column_map: Mapping[str, str] | None) -> set[str]:
    hints = {name for target, names in synonyms.items() for name in (target, *names)}
    for key, value in (column_map or {}).items():
        hints.update((str(key), str(value)))
    return hints


def _numeric_field(
    raw: pd.DataFrame,
    mapping: Mapping[str, str],
    target: str,
    notes: list[str],
    *,
    strip_units: bool = False,
) -> NumericParse:
    column = mapping[target]
    parsed = parse_numeric(raw[column], strip_units=strip_units)
    if parsed.n_invalid:
        notes.append(f"column '{column}': {parsed.n_invalid} value(s) could not be read as numbers")
    if parsed.ambiguous_example is not None:
        notes.append(
            f"column '{column}': separators are ambiguous (e.g. '{parsed.ambiguous_example}' could "
            "be a decimal or a thousands group); read '.' as the decimal point"
        )
    return parsed


def _numeric_token(text: str, strip_units: bool) -> tuple[str, bool, bool] | None:
    """Strip currency, spacing and sign markup; return (digits/separators, negative, had %)."""
    value = _SHILLING_SUFFIX_RE.sub("", text.replace("−", "-")).strip()
    percent = "%" in value
    value = _CURRENCY_CODE_RE.sub("", value.replace("%", ""))
    value = _CURRENCY_SYMBOL_RE.sub("", value)
    value = re.sub(r"[\s'’]", "", value)
    negative = False
    if value.startswith("(") and value.endswith(")"):
        negative, value = True, value[1:-1]
    if value[:1] in ("-", "+"):
        negative, value = negative != (value[0] == "-"), value[1:]
    if value.endswith("-"):
        negative, value = not negative, value[:-1]
    if strip_units:
        value = _UNIT_SUFFIX_RE.sub("", value)
    if _SCIENTIFIC_RE.fullmatch(value) or _NUMBER_RE.fullmatch(value):
        return value, negative, percent
    return None


def _separator_kind(token: str) -> tuple[str, str | None]:
    """How a token's separators read: ("own"|"strong", decimal), ("ambiguous", sep) or ("none", None)."""
    has_dot, has_comma = "." in token, "," in token
    if has_dot and has_comma:
        return "own", token[max(token.rfind("."), token.rfind(","))]
    if not (has_dot or has_comma):
        return "none", None
    separator = "." if has_dot else ","
    if token.count(separator) > 1:
        return "strong", "," if separator == "." else "."  # repeated: thousands separator
    head, tail = token.split(separator)
    if len(tail) == 3 and 1 <= len(head) <= 3 and not head.startswith("0"):
        return "ambiguous", separator  # "1,234": a thousands group or three decimals
    return "strong", separator


def _token_value(token: str, decimal: str) -> float | None:
    if _SCIENTIFIC_RE.fullmatch(token):
        return float(token)
    kind, separator = _separator_kind(token)
    if kind in ("own", "strong") and separator is not None:
        decimal = separator
    thousands = "," if decimal == "." else "."
    try:
        return float(token.replace(thousands, "").replace(decimal, "."))
    except ValueError:
        return None


def _choose_decimal(
    pending: list[tuple[int, str, bool]],
    frequency: np.ndarray,
    fixed: np.ndarray,
    uniques: list[object],
) -> tuple[str, str | None]:
    votes = {".": 0, ",": 0}
    ambiguous: list[int] = []
    for i, token, _ in pending:
        kind, separator = _separator_kind(token)
        if kind in ("own", "strong") and separator is not None:
            votes[separator] += int(frequency[i])
        elif kind == "ambiguous":
            ambiguous.append(i)
    if votes["."] or votes[","]:
        return ("." if votes["."] >= votes[","] else ","), None
    if not ambiguous:
        return ".", None
    known = fixed[np.isfinite(fixed)]
    spread = {}
    for decimal in (".", ","):
        readings = [_token_value(token, decimal) for _, token, _ in pending]
        spread[decimal] = _log_spread(np.array([*known, *(r for r in readings if r is not None)]))
    if spread[","] < spread["."] - _SPREAD_MARGIN:
        return ",", None
    if spread["."] < spread[","] - _SPREAD_MARGIN:
        return ".", None
    return ".", str(uniques[ambiguous[0]]).strip()


def _log_spread(values: np.ndarray) -> float:
    magnitudes = np.abs(values[np.isfinite(values)])
    magnitudes = magnitudes[magnitudes > 0]
    return float(np.std(np.log10(magnitudes))) if magnitudes.size > 1 else 0.0


def _resolve_day_order(
    pairs: Mapping[int, tuple[int, int, int]],
    uniques: list[object],
    frequency: np.ndarray,
    dayfirst: bool | None,
    name: str,
) -> tuple[bool, list[str]]:
    day_only = [i for i, (a, b, _) in pairs.items() if a > 12 >= b]
    month_only = [i for i, (a, b, _) in pairs.items() if b > 12 >= a]
    either = [i for i, (a, b, _) in pairs.items() if a <= 12 and b <= 12 and a != b]

    def count(ids: list[int]) -> int:
        return int(sum(frequency[i] for i in ids))

    def example(ids: list[int]) -> str:
        return str(uniques[ids[0]]).strip()

    notes: list[str] = []
    if day_only and month_only:
        order = dayfirst if dayfirst is not None else count(day_only) >= count(month_only)
        rest = f"; {count(either)} ambiguous value(s) read {_order_name(order)}" if either else ""
        notes.append(
            f"date column '{name}' mixes day-first (e.g. '{example(day_only)}') and month-first "
            f"(e.g. '{example(month_only)}') dates; each was read in the only valid order{rest}"
        )
    elif day_only or month_only:
        order = bool(day_only)
        if dayfirst is not None and dayfirst != order:
            notes.append(
                f"date column '{name}': dayfirst={dayfirst} ignored because values such as "
                f"'{example(day_only or month_only)}' are only valid {_order_name(order)}"
            )
    elif either:
        order = True if dayfirst is None else dayfirst
        how = (
            "assumed day-first (the KE/NG/GH/ZA/GB convention); pass dayfirst=False for month-first exports"
            if dayfirst is None
            else f"read {_order_name(order)} as requested by dayfirst={dayfirst}"
        )
        notes.append(
            f"date column '{name}': {count(either)} value(s) such as '{example(either)}' are valid "
            f"day-first and month-first and no value has a part above 12; {how}"
        )
    else:
        order = True if dayfirst is None else dayfirst
    return order, notes


def _order_name(dayfirst: bool) -> str:
    return "day-first" if dayfirst else "month-first"


def _full_year(text: str) -> int:
    year = int(text)
    if len(text) == 2:
        year += 2000 if year < 69 else 1900  # the strptime %y pivot
    return year


def _numeric_date(value: float) -> np.datetime64:
    if value.is_integer() and 19000101 <= value <= 21001231:
        digits = str(int(value))
        try:
            return _midnight(pd.Timestamp(int(digits[:4]), int(digits[4:6]), int(digits[6:])))
        except ValueError:
            return np.datetime64("NaT")
    if 20000 <= value <= 80000:  # Excel serial day number (1900 date system)
        return _midnight(_EXCEL_EPOCH + pd.Timedelta(days=math.floor(value)))
    return np.datetime64("NaT")


def _text_date(text: str, dayfirst: bool) -> np.datetime64:
    # dateutil turns a bare "5" into the 5th of the current month; demand a month name or a year.
    if not (_MONTH_NAME_RE.search(text) or _YEAR_RE.search(text)):
        return np.datetime64("NaT")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stamp = pd.to_datetime(text, dayfirst=dayfirst, errors="coerce")
    except (ValueError, OverflowError, TypeError):
        return np.datetime64("NaT")
    return np.datetime64("NaT") if pd.isna(stamp) else _midnight(stamp)


def _midnight(stamp: pd.Timestamp) -> np.datetime64:
    if stamp.tzinfo is not None:
        stamp = stamp.tz_localize(None)
    return stamp.normalize().to_datetime64()


def _within_years(stamps: pd.Series) -> pd.Series:
    years = stamps.dt.year
    return stamps.where((years >= _YEAR_RANGE[0]) & (years <= _YEAR_RANGE[1]))


def _flag(value: object) -> bool:
    if value is None or value is pd.NA:
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(np.isfinite(value) and value != 0)
    text = " ".join(str(value).split()).casefold()
    if text in _FALSE_TOKENS:
        return False
    if text in _TRUE_TOKENS:
        return True
    try:
        return float(text.replace(",", ".")) != 0
    except ValueError:
        return True  # any other text, e.g. a promotion name


def _labels(values: pd.Series) -> pd.Series:
    """Raw labels as whitespace-trimmed strings; missing markers become NaN."""

    def clean(value: object) -> object:
        if isinstance(value, (float, np.floating)) and float(value).is_integer():
            value = int(value)
        text = " ".join(str(value).split())
        return np.nan if text.casefold() in _MISSING_TOKENS else text

    return map_unique(values, clean)


def _row_ids(raw: pd.DataFrame, mapping: Mapping[str, str], target: str, default: str) -> pd.Series:
    if target not in mapping:
        return pd.Series([default] * len(raw), index=raw.index)
    return normalize_product_ids(_labels(raw[mapping[target]])).fillna(default)


def _exact_duplicates(raw: pd.DataFrame) -> pd.Series:
    try:
        return raw.duplicated(keep="first")
    except TypeError:  # unhashable cells in a DataFrame source
        return raw.astype(str).duplicated(keep="first")


def _discount_fraction(parsed: NumericParse, revenue: pd.Series, column: str, notes: list[str]) -> pd.Series:
    """Discount as a fraction of the pre-discount price (0.1 = 10% off)."""
    values = parsed.values
    finite = values[np.isfinite(values)]
    if finite.empty:
        return values
    if finite.max() > 100:
        notes.append(
            f"discount column '{column}' has values above 100, so they are amounts, not "
            "percentages; converted to amount / (amount + revenue), assuming revenue is net"
        )
        gross = values + revenue
        fraction = (values / gross).where(gross > 0)
    elif parsed.percent or finite.max() > 1:
        fraction = values / 100.0
    else:
        fraction = values
    return fraction.clip(lower=0.0, upper=1.0)


def _low_variation_note(transactions: pd.DataFrame) -> str | None:
    keys = series_keys(transactions)
    distinct = (
        transactions.assign(_price_key=transactions[PRICE].round(6))
        .groupby(keys, dropna=False)["_price_key"]
        .nunique()
        .reset_index()
    )
    flat = distinct.loc[distinct["_price_key"] < 2, PRODUCT_ID].astype(str).tolist()
    if not flat:
        return None
    listed = ", ".join(flat[:10]) + (", ..." if len(flat) > 10 else "")
    return (
        f"{len(flat)} of {len(distinct)} product(s) have fewer than 2 distinct prices, so their "
        f"own sales carry no identifying price variation: {listed}"
    )


def _date_bounds(transactions: pd.DataFrame) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if transactions.empty:
        return None, None
    return pd.Timestamp(transactions[DATE].min()), pd.Timestamp(transactions[DATE].max())


def _missing_sale_days(panel: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    sold = panel.loc[(float_column(panel, UNITS) > 0) & ~bool_column(panel, IMPUTED_ZERO), [*keys, DATE]]
    if sold.empty:
        return pd.DataFrame(columns=[*keys, DATE])
    bounds = sold.groupby(keys, sort=False, dropna=False)[DATE].agg(["min", "max"]).reset_index()
    pieces = []
    for record in bounds.to_dict("records"):
        piece = pd.DataFrame({DATE: pd.date_range(record["min"], record["max"], freq="D")})
        for key in keys:
            piece[key] = record[key]
        pieces.append(piece)
    grid = pd.concat(pieces, ignore_index=True)[[*keys, DATE]]
    grid[DATE] = grid[DATE].astype("datetime64[ns]")
    present = panel[[*keys, DATE]].assign(_present=True)
    merged = grid.merge(present, on=[*keys, DATE], how="left")
    return merged.loc[merged["_present"].isna(), [*keys, DATE]].reset_index(drop=True)


def _carry_forward(values: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    carried = values.copy()
    for positions in groups:
        carried[positions] = pd.Series(values[positions]).ffill().to_numpy()
    return carried
