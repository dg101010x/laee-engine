"""Result types shared across modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class Interval:
    lower: float
    upper: float
    level: float


@dataclass
class ElasticityEstimate:
    value: float
    std_error: float
    intervals: list[Interval]
    method: str
    identification_tier: int
    n_obs: int
    n_products: int
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def interval(self, level: float) -> Interval:
        for iv in self.intervals:
            if abs(iv.level - level) < 1e-9:
                return iv
        raise KeyError(f"no {level:.0%} interval")


@dataclass(frozen=True)
class ModelVersionRecord:
    """Spec section 73: everything needed to reproduce a prediction."""

    model_version: str
    training_window: tuple[str, str]
    feature_version: str
    data_timestamp: datetime
    prediction_timestamp: datetime
    llm_version: str | None = None
    embedding_version: str | None = None
