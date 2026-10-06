"""Panel normalization: prices, promotions and stockouts (spec §62-65; §84 steps 3-5).

Recommended order on a panel from :func:`laee.ingestion.ingest_sales`::

    panel = flag_price_outliers(result.panel)
    panel = detect_promotions(panel)          # on actual observations
    panel = fill_zero_sales_days(panel)       # laee.ingestion; uses the regular prices
    panel = detect_stockouts(panel, inventory)
    sample = panel[estimation_mask(panel)]

Every function returns a new frame and treats ``imputed_zero`` rows as non-observations.
"""

from laee.normalization.prices import (
    PricePoints,
    detect_price_changes,
    flag_price_outliers,
    normalize_product_ids,
    price_points,
)
from laee.normalization.promotions import detect_promotions
from laee.normalization.stockouts import detect_stockouts, estimation_mask

__all__ = [
    "PricePoints",
    "detect_price_changes",
    "detect_promotions",
    "detect_stockouts",
    "estimation_mask",
    "flag_price_outliers",
    "normalize_product_ids",
    "price_points",
]
