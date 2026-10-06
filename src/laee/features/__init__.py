"""Leakage-safe feature construction (spec §59, §74-75)."""

from laee.features.calendar import (
    SUPPORTED_COUNTRIES,
    add_calendar_features,
    builtin_holidays,
    fourier_name,
)
from laee.features.temporal import (
    FEATURE_REGISTRY,
    FeatureSpec,
    add_lag_features,
    add_rolling_features,
    as_of,
    describe_feature,
    lag_name,
    rolling_name,
    trailing_stat,
)

__all__ = [
    "FEATURE_REGISTRY",
    "SUPPORTED_COUNTRIES",
    "FeatureSpec",
    "add_calendar_features",
    "add_lag_features",
    "add_rolling_features",
    "as_of",
    "builtin_holidays",
    "describe_feature",
    "fourier_name",
    "lag_name",
    "rolling_name",
    "trailing_stat",
]
