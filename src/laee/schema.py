"""Canonical column names shared by every LAEE module. Import these instead of string literals."""

BUSINESS_ID = "business_id"
PRODUCT_ID = "product_id"
LOCATION_ID = "location_id"
DATE = "date"

PRICE = "price"
UNITS = "units"
REVENUE = "revenue"
UNIT_COST = "unit_cost"

PROMO_FLAG = "promo_flag"
DISCOUNT_PCT = "discount_pct"
REGULAR_PRICE = "regular_price"
INVENTORY = "inventory"
STOCKOUT_FLAG = "stockout_flag"
CENSORED_FLAG = "censored_flag"

AVAILABLE_AT = "available_at"

PANEL_KEYS = (BUSINESS_ID, PRODUCT_ID, LOCATION_ID, DATE)
REQUIRED_SALES_COLUMNS = (PRODUCT_ID, DATE, PRICE, UNITS)
