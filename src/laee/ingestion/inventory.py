"""Inventory ingestion: stock-on-hand exports to a clean product-day stock table (spec §63).

Parsing (encodings, delimiters, header synonyms, currency/thousands handling, mixed
date formats) is shared with :mod:`laee.ingestion.sales`.

``inventory`` is the stock on hand recorded for the date.
:func:`laee.normalization.detect_stockouts` reads it as the stock available at the start
of that day by default; pass ``inventory_timing="end_of_day"`` there for closing counts.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from laee.ingestion.sales import (
    SALES_SYNONYMS,
    IngestReport,
    Source,
    _date_bounds,
    _exact_duplicates,
    _header_hints,
    _labels,
    _numeric_field,
    _require_fields,
    _row_ids,
    detect_columns,
    parse_dates,
    read_table,
)
from laee.normalization.prices import normalize_product_ids
from laee.schema import AVAILABLE_AT, BUSINESS_ID, DATE, INVENTORY, LOCATION_ID, PANEL_KEYS, PRODUCT_ID

REORDER_POINT = "reorder_point"

INVENTORY_SYNONYMS: dict[str, tuple[str, ...]] = {
    PRODUCT_ID: SALES_SYNONYMS[PRODUCT_ID],
    DATE: SALES_SYNONYMS[DATE] + ("count_date", "stock_date", "snapshot_date", "as_of", "as_of_date"),
    INVENTORY: SALES_SYNONYMS[INVENTORY] + ("closing_balance", "balance", "quantity", "qty"),
    REORDER_POINT: ("reorder_point", "reorder_level", "min_stock", "minimum_stock"),
    BUSINESS_ID: ("business_id",),
    LOCATION_ID: ("location_id",),
}

INVENTORY_DROP_REASONS = ("missing_product", "unparseable_date", "missing_inventory", "duplicate_row", "malformed_row")


def ingest_inventory(
    source: Source,
    column_map: Mapping[str, str] | None = None,
    *,
    business_id: str = "default",
    location_id: str = "default",
    dayfirst: bool | None = None,
) -> tuple[pd.DataFrame, IngestReport]:
    """Ingest a stock-on-hand export into one row per (business, product, location, date).

    Returns the panel keys, ``inventory`` (stock on hand), ``reorder_point`` when the
    export has one, and ``available_at``. A count is assumed known by the end of its
    day, which is conservative for leakage. Negative stock is kept, because it reads as
    out of stock, and is noted. Several rows for one product-day keep the highest count
    and are noted. ``rows_used`` counts the input rows that passed validation, before
    that combination.
    """
    raw, malformed = read_table(source, header_hints=_header_hints(INVENTORY_SYNONYMS, column_map))
    mapping = detect_columns(raw.columns, INVENTORY_SYNONYMS, column_map)
    _require_fields(mapping, raw.columns, (PRODUCT_ID, DATE, INVENTORY))
    notes: list[str] = []

    product = normalize_product_ids(_labels(raw[mapping[PRODUCT_ID]]))
    dates = parse_dates(raw[mapping[DATE]], dayfirst=dayfirst, name=mapping[DATE])
    notes.extend(dates.warnings)
    stock = _numeric_field(raw, mapping, INVENTORY, notes, strip_units=True).values

    reason = pd.Series(None, index=raw.index, dtype=object)
    for name, failed in (
        ("missing_product", product.isna()),
        ("unparseable_date", dates.values.isna()),
        ("missing_inventory", stock.isna()),
    ):
        reason = reason.mask(reason.isna() & failed, name)
    reason = reason.mask(reason.isna() & _exact_duplicates(raw), "duplicate_row")
    keep = reason.isna()

    table = pd.DataFrame(
        {
            BUSINESS_ID: _row_ids(raw, mapping, BUSINESS_ID, business_id),
            PRODUCT_ID: product,
            LOCATION_ID: _row_ids(raw, mapping, LOCATION_ID, location_id),
            DATE: dates.values,
            INVENTORY: stock,
        },
        index=raw.index,
    )
    if REORDER_POINT in mapping:
        table[REORDER_POINT] = _numeric_field(raw, mapping, REORDER_POINT, notes).values
    rows = table.loc[keep.to_numpy()]

    negative = int((rows[INVENTORY] < 0).sum())
    if negative:
        notes.append(f"{negative} negative stock count(s) kept; they are treated as out of stock")
    value_columns = [col for col in (INVENTORY, REORDER_POINT) if col in rows.columns]
    counts = rows.groupby(list(PANEL_KEYS), sort=True, dropna=False)[value_columns].max().reset_index()
    repeated = len(rows) - len(counts)
    if repeated:
        notes.append(f"{repeated} extra count(s) for an already-counted product-day; kept the highest")
    counts[AVAILABLE_AT] = counts[DATE] + pd.Timedelta(days=1)

    dropped = {name: int((reason == name).sum()) for name in INVENTORY_DROP_REASONS}
    dropped["malformed_row"] = malformed
    if malformed:
        notes.append(f"{malformed} malformed line(s) with more fields than the header were skipped")
    date_min, date_max = _date_bounds(rows)
    report = IngestReport(
        rows_read=len(raw) + malformed,
        rows_used=int(np.count_nonzero(keep.to_numpy())),
        dropped=dropped,
        column_mapping=dict(mapping),
        date_min=date_min,
        date_max=date_max,
        n_products=int(counts[PRODUCT_ID].nunique()),
        warnings=notes,
    )
    return counts, report
