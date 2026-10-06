"""Competitor influence, competitive pressure and price indices (spec §8-10, §29, §61).

Never treat all competitors equally (spec Rule 6). Each competitor j of business i gets an
influence
    C_ij = S_j * D_ij * M_ij * R_ij * W_ij
from its size S, distance decay D, product similarity M, price-band similarity R and customer
overlap W. Competition_i is the sum of C_ij over competitors active at the evaluation date.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from laee.schema import DATE

from .catchment import catchment_overlap
from .distance import (
    DensityClass,
    FloatArray,
    HaversineCircuity,
    TravelEstimator,
    check_point,
    haversine_km,
    validate_density_class,
)
from .market import distance_weights, industry_decay_lambda, validate_industry_id

DEFAULT_EMBEDDING_DIM = 256
DEFAULT_PRICE_TAU = 0.25
DEFAULT_CATEGORY_WEIGHT = 0.5
NGRAM_RANGE = (3, 5)

DateLike = date | datetime | str | np.datetime64
PriceValues = ArrayLike | Sequence[float | None]
"""Prices with unknown entries given as None or NaN."""

INFLUENCE_COLUMNS = (
    "competitor_id",
    "name",
    "category",
    "distance_km",
    "travel_km",
    "size_weight",
    "distance_decay",
    "similarity",
    "price_similarity",
    "customer_overlap",
    "reference_price",
    "influence",
    "share",
)


def _as_date(value: object) -> date:
    if isinstance(value, datetime):  # includes pandas.Timestamp
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, (str, np.datetime64)):
        return pd.Timestamp(value).date()
    raise TypeError(f"expected a date, datetime or ISO date string, got {type(value).__name__}")


def _as_text_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, Iterable):
        raise TypeError("text_attributes must be a string or an iterable of strings")
    items = tuple(value)
    if not all(isinstance(item, str) for item in items):
        raise TypeError("text_attributes must contain only strings")
    return items


def _check_price(price: float | None, what: str) -> float | None:
    if price is None:
        return None
    value = float(price)
    if not (math.isfinite(value) and value > 0):
        raise ValueError(f"{what} must be a positive finite number or None")
    return value


@dataclass(frozen=True)
class Competitor:
    """A competing business. `category` is one of `laee.geo.market.INDUSTRY_IDS`.

    `text_attributes` hold free text such as products, services, price tier, cuisine, reviews or a
    description (spec §9). `size_weight` is S_j, the competitor's relative size or market
    relevance. Dates follow spec §29: a competitor is active from `opened_at` (inclusive) until
    `closed_at` (exclusive).
    """

    id: str
    name: str | None
    lat: float
    lon: float
    category: str
    reference_price: float | None = None
    size_weight: float = 1.0
    text_attributes: tuple[str, ...] = ()
    opened_at: date | None = None
    closed_at: date | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("competitor id must be a non-empty string")
        lat, lon = check_point(self.lat, self.lon)
        validate_industry_id(self.category)
        size = float(self.size_weight)
        if not (math.isfinite(size) and size >= 0):
            raise ValueError("size_weight must be a non-negative finite number")
        opened = None if self.opened_at is None else _as_date(self.opened_at)
        closed = None if self.closed_at is None else _as_date(self.closed_at)
        if opened is not None and closed is not None and closed < opened:
            raise ValueError(f"competitor {self.id!r} closes before it opens")
        object.__setattr__(self, "lat", lat)
        object.__setattr__(self, "lon", lon)
        object.__setattr__(self, "size_weight", size)
        object.__setattr__(self, "reference_price", _check_price(self.reference_price, "price"))
        object.__setattr__(self, "text_attributes", _as_text_tuple(self.text_attributes))
        object.__setattr__(self, "opened_at", opened)
        object.__setattr__(self, "closed_at", closed)


@dataclass(frozen=True)
class BusinessLocation:
    """The business whose market is being measured.

    `density_class` selects the default radius prior and circuity factor.
    """

    lat: float
    lon: float
    industry_id: str
    reference_price: float | None = None
    text_attributes: tuple[str, ...] = ()
    density_class: DensityClass = "urban"

    def __post_init__(self) -> None:
        lat, lon = check_point(self.lat, self.lon)
        validate_industry_id(self.industry_id)
        object.__setattr__(self, "lat", lat)
        object.__setattr__(self, "lon", lon)
        object.__setattr__(self, "density_class", validate_density_class(self.density_class))
        object.__setattr__(self, "reference_price", _check_price(self.reference_price, "price"))
        object.__setattr__(self, "text_attributes", _as_text_tuple(self.text_attributes))


def is_active(competitor: Competitor, as_of: DateLike | None) -> bool:
    """Whether the competitor trades on `as_of` (opened_at <= as_of < closed_at).

    With `as_of=None` the list is read as a current snapshot: every competitor without a
    `closed_at` counts. Pass an explicit date for any historical computation (spec §74).
    """
    if as_of is None:
        return competitor.closed_at is None
    day = _as_date(as_of)
    opened, closed = competitor.opened_at, competitor.closed_at
    return (opened is None or opened <= day) and (closed is None or day < closed)


_SEPARATORS = re.compile(r"[\W_]+")


def _words(text: str) -> list[str]:
    folded = unicodedata.normalize("NFKD", text.casefold())
    plain = "".join(ch for ch in folded if not unicodedata.combining(ch))
    return [w for w in _SEPARATORS.split(plain) if w]


def _char_ngrams(words: Iterable[str]) -> Iterator[str]:
    n_min, n_max = NGRAM_RANGE
    for word in words:
        padded = f" {word} "
        for n in range(n_min, min(n_max, len(padded)) + 1):
            for start in range(len(padded) - n + 1):
                yield padded[start : start + n]


@lru_cache(maxsize=1 << 16)
def _bucket(ngram: str, dim: int) -> tuple[int, float]:
    # blake2b instead of hash(): Python salts str hashes per process (PYTHONHASHSEED).
    value = int.from_bytes(hashlib.blake2b(ngram.encode("utf-8"), digest_size=8).digest(), "big")
    return value % dim, 1.0 if value >> 63 else -1.0


def hashing_embedding(text: str, dim: int = DEFAULT_EMBEDDING_DIM) -> FloatArray:
    """Unit-length character n-gram hashing vector of `text`, identical across runs.

    Text is case-folded and accent-stripped, then split on non-word characters; each word
    contributes its space-padded character 3- to 5-grams. Every n-gram adds +1 or -1 (a sign hash)
    to one bucket, which keeps collisions unbiased in dot products. Empty text gives zeros.
    """
    if dim < 1:
        raise ValueError("dim must be positive")
    vec = np.zeros(dim, dtype=np.float64)
    for gram in _char_ngrams(_words(text)):
        index, sign = _bucket(gram, dim)
        vec[index] += sign
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


def _embed(attributes: Sequence[str], dim: int) -> FloatArray | None:
    vec = hashing_embedding(" ".join(attributes), dim)
    return vec if vec.any() else None


def _similarity(
    e_i: FloatArray | None, cat_i: str, e_j: FloatArray | None, cat_j: str, category_weight: float
) -> float:
    match = 1.0 if cat_i == cat_j else 0.0
    if e_i is None or e_j is None:
        # Without text on both sides only the category signal exists; use it alone rather than
        # count the missing text as dissimilar.
        return match
    text = max(0.0, float(e_i @ e_j))
    return (1.0 - category_weight) * text + category_weight * match


def _check_category_weight(category_weight: float) -> float:
    if not 0.0 <= category_weight <= 1.0:
        raise ValueError("category_weight must lie in [0, 1]")
    return category_weight


def product_similarity(
    business: BusinessLocation,
    competitor: Competitor,
    *,
    category_weight: float = DEFAULT_CATEGORY_WEIGHT,
    dim: int = DEFAULT_EMBEDDING_DIM,
) -> float:
    """M_ij in [0, 1] (spec §9): text cosine blended with a category match.

    M = (1 - w) * max(0, cos(E_i, E_j)) + w * [category_i == category_j]. When either side has no
    text attributes, M is the category term alone.
    """
    return _similarity(
        _embed(business.text_attributes, dim),
        business.industry_id,
        _embed(competitor.text_attributes, dim),
        competitor.category,
        _check_category_weight(category_weight),
    )


def price_band_similarity(
    business_price: float | None, competitor_prices: PriceValues, tau: float = DEFAULT_PRICE_TAU
) -> FloatArray:
    """R_ij = exp(-|ln(P_i / P_j)| / tau); 1.0 (no penalty) where either price is unknown.

    Unknown competitor prices are None or NaN. With tau = 0.25 a competitor 20% dearer scores
    0.48 and one at twice the price 0.06.
    """
    if not (math.isfinite(tau) and tau > 0):
        raise ValueError("tau must be a positive finite number")
    p_j = np.array(competitor_prices, dtype=np.float64)
    known = ~np.isnan(p_j)
    if np.any(p_j[known] <= 0) or np.any(np.isinf(p_j[known])):
        raise ValueError("competitor prices must be positive and finite")
    out = np.ones_like(p_j)
    p_i = _check_price(business_price, "business price")
    if p_i is None:
        return out
    out[known] = np.exp(-np.abs(np.log(p_i / p_j[known])) / tau)
    return out


def effective_competitor_count(influence: ArrayLike) -> float:
    """Inverse Herfindahl index of normalised influences: n for n equal competitors (spec §9)."""
    c = np.asarray(influence, dtype=np.float64).ravel()
    if np.any(~np.isfinite(c)) or np.any(c < 0):
        raise ValueError("influence values must be finite and non-negative")
    total = float(c.sum())
    if total <= 0:
        return 0.0
    shares = c / total
    return float(1.0 / np.dot(shares, shares))


@dataclass(frozen=True)
class PriceIndex:
    """Competitor-relative price position (spec §10, §61)."""

    cpi: float
    weighted_competitor_price: float
    relative_price: float
    n_priced: int


def competitive_price_index(
    business_price: float | None, competitor_prices: PriceValues, weights: ArrayLike
) -> PriceIndex | None:
    """CPI_i = sum_j w_ij P_j / P_i, with the weights normalised over competitors with a price.

    CPI > 1 means the business is cheaper than its weighted competitors. relative_price is
    (P_i - P_w) / P_w, where P_w is the weighted competitor price. Returns None when the business
    price is unknown or no competitor with positive weight has a price.
    """
    p = np.array(competitor_prices, dtype=np.float64).ravel()
    w = np.asarray(weights, dtype=np.float64).ravel()
    if p.shape != w.shape:
        raise ValueError("competitor_prices and weights must have the same length")
    if np.any(~np.isfinite(w)) or np.any(w < 0):
        raise ValueError("weights must be finite and non-negative")
    known = ~np.isnan(p)
    if np.any(p[known] <= 0) or np.any(np.isinf(p[known])):
        raise ValueError("competitor prices must be positive and finite")
    p_i = _check_price(business_price, "business price")
    priced = known & (w > 0)
    if p_i is None or not np.any(priced):
        return None
    p_w = float(np.dot(w[priced], p[priced]) / w[priced].sum())
    return PriceIndex(
        cpi=p_w / p_i,
        weighted_competitor_price=p_w,
        relative_price=(p_i - p_w) / p_w,
        n_priced=int(priced.sum()),
    )


@dataclass(frozen=True, eq=False)
class CompetitionSummary:
    """Competitive pressure on one business at one date.

    `influence` has one row per active competitor inside the market (columns
    `INFLUENCE_COLUMNS`), sorted by influence, largest first.
    """

    as_of: date | None
    total_pressure: float
    effective_competitors: float
    n_active: int
    competitive_price_index: float | None
    weighted_competitor_price: float | None
    relative_price: float | None
    n_priced: int
    distance_lambda: float
    catchment_lambda: float | None
    influence: pd.DataFrame

    @property
    def effective_radius_km(self) -> float | None:
        """Travel distance at which the distance-decay weight halves (None without decay)."""
        return math.log(2.0) / self.distance_lambda if self.distance_lambda > 0 else None


@dataclass(frozen=True)
class _Params:
    travel: TravelEstimator
    distance_lambda: float
    catchment_lambda: float | None
    price_tau: float
    category_weight: float
    embedding_dim: int
    market_radius_km: float | None


def _params(
    business: BusinessLocation,
    travel: TravelEstimator | None,
    distance_lambda: float | None,
    catchment_lambda: float | None,
    price_tau: float,
    category_weight: float,
    embedding_dim: int,
    market_radius_km: float | None,
) -> _Params:
    lam = (
        industry_decay_lambda(business.industry_id, business.density_class)
        if distance_lambda is None
        else float(distance_lambda)
    )
    if not (math.isfinite(lam) and lam >= 0):
        raise ValueError("distance_lambda must be a non-negative finite number")
    if catchment_lambda is not None and not (
        math.isfinite(catchment_lambda) and catchment_lambda >= 0
    ):
        raise ValueError("catchment_lambda must be a non-negative finite number")
    if not (math.isfinite(price_tau) and price_tau > 0):
        raise ValueError("price_tau must be a positive finite number")
    if embedding_dim < 1:
        raise ValueError("embedding_dim must be positive")
    if market_radius_km is not None and not market_radius_km > 0:
        raise ValueError("market_radius_km must be positive")
    return _Params(
        travel=HaversineCircuity.for_density(business.density_class) if travel is None else travel,
        distance_lambda=lam,
        catchment_lambda=catchment_lambda,
        price_tau=price_tau,
        category_weight=_check_category_weight(category_weight),
        embedding_dim=embedding_dim,
        market_radius_km=market_radius_km,
    )


def _unique(competitors: Iterable[Competitor]) -> list[Competitor]:
    items = list(competitors)
    seen: set[str] = set()
    for c in items:
        if c.id in seen:
            raise ValueError(f"duplicate competitor id {c.id!r}")
        seen.add(c.id)
    return items


def _empty_table() -> pd.DataFrame:
    # dtype=str matches what pandas infers for the text columns of a non-empty table.
    text = {"competitor_id", "name", "category"}
    return pd.DataFrame(
        {col: pd.Series(dtype=str if col in text else np.float64) for col in INFLUENCE_COLUMNS}
    )


def _influence_table(
    business: BusinessLocation, competitors: Sequence[Competitor], p: _Params
) -> pd.DataFrame:
    if not competitors:
        return _empty_table()
    lat = np.array([c.lat for c in competitors])
    lon = np.array([c.lon for c in competitors])
    straight = haversine_km(business.lat, business.lon, lat, lon)
    if p.market_radius_km is not None:
        # The market radius is a physical radius (spec §5.1 Layer A), like discovery's around:.
        inside = straight <= p.market_radius_km
        competitors = [c for c, keep in zip(competitors, inside) if keep]
        lat, lon, straight = lat[inside], lon[inside], straight[inside]
        if not competitors:
            return _empty_table()
    travel = np.asarray(
        p.travel.travel_km((business.lat, business.lon), np.column_stack([lat, lon])),
        dtype=np.float64,
    )
    if travel.shape != straight.shape or np.any(np.isnan(travel)) or np.any(travel < 0):
        raise ValueError("travel estimator must return one non-negative distance per competitor")
    decay = distance_weights(travel, p.distance_lambda)
    e_i = _embed(business.text_attributes, p.embedding_dim)
    similarity = np.array(
        [
            _similarity(
                e_i,
                business.industry_id,
                _embed(c.text_attributes, p.embedding_dim),
                c.category,
                p.category_weight,
            )
            for c in competitors
        ]
    )
    prices = np.array(
        [np.nan if c.reference_price is None else c.reference_price for c in competitors]
    )
    price_sim = price_band_similarity(business.reference_price, prices, p.price_tau)
    overlap = (
        np.ones_like(travel)
        if p.catchment_lambda is None
        else catchment_overlap(travel, p.catchment_lambda)
    )
    size = np.array([c.size_weight for c in competitors])
    influence = size * decay * similarity * price_sim * overlap
    total = float(influence.sum())
    table = pd.DataFrame(
        {
            "competitor_id": [c.id for c in competitors],
            "name": pd.Series([c.name for c in competitors], dtype=str),
            "category": [c.category for c in competitors],
            "distance_km": straight,
            "travel_km": travel,
            "size_weight": size,
            "distance_decay": decay,
            "similarity": similarity,
            "price_similarity": price_sim,
            "customer_overlap": overlap,
            "reference_price": prices,
            "influence": influence,
            "share": influence / total if total > 0 else np.zeros_like(influence),
        }
    )
    return table.sort_values(
        ["influence", "competitor_id"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)


def competitive_pressure(
    business: BusinessLocation,
    competitors: Iterable[Competitor],
    *,
    as_of: DateLike | None = None,
    travel: TravelEstimator | None = None,
    distance_lambda: float | None = None,
    catchment_lambda: float | None = None,
    price_tau: float = DEFAULT_PRICE_TAU,
    category_weight: float = DEFAULT_CATEGORY_WEIGHT,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    market_radius_km: float | None = None,
) -> CompetitionSummary:
    """Competition_i = sum_j C_ij over competitors active at `as_of` (spec §8-10, §29, §61).

    Defaults: `travel` is the circuity-scaled haversine for the business's density class.
    `distance_lambda` is ln 2 over the industry's median prior radius, or 0 (no decay) for
    ecommerce. With `catchment_lambda=None` the customer overlap W is 1 (unknown); pass
    `fit_distance_decay(customer_distances)` to estimate it. `market_radius_km` drops competitors
    beyond that straight-line radius. `as_of` follows `is_active`.
    """
    p = _params(
        business,
        travel,
        distance_lambda,
        catchment_lambda,
        price_tau,
        category_weight,
        embedding_dim,
        market_radius_km,
    )
    day = None if as_of is None else _as_date(as_of)
    active = [c for c in _unique(competitors) if is_active(c, day)]
    table = _influence_table(business, active, p)
    influence = table["influence"].to_numpy(dtype=np.float64)
    price = competitive_price_index(
        business.reference_price, table["reference_price"].to_numpy(dtype=np.float64), influence
    )
    return CompetitionSummary(
        as_of=day,
        total_pressure=float(influence.sum()),
        effective_competitors=effective_competitor_count(influence),
        n_active=len(table),
        competitive_price_index=None if price is None else price.cpi,
        weighted_competitor_price=None if price is None else price.weighted_competitor_price,
        relative_price=None if price is None else price.relative_price,
        n_priced=0 if price is None else price.n_priced,
        distance_lambda=p.distance_lambda,
        catchment_lambda=p.catchment_lambda,
        influence=table,
    )


def competition_change_series(
    business: BusinessLocation,
    competitors: Iterable[Competitor],
    dates: Iterable[DateLike],
    *,
    travel: TravelEstimator | None = None,
    distance_lambda: float | None = None,
    catchment_lambda: float | None = None,
    price_tau: float = DEFAULT_PRICE_TAU,
    category_weight: float = DEFAULT_CATEGORY_WEIGHT,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    market_radius_km: float | None = None,
) -> pd.DataFrame:
    """Competition_t and Delta Competition_t = Competition_t - Competition_{t-1} (spec §29).

    One row per date (strictly increasing), indexed by date. Columns: `competition`,
    `delta_competition` (NaN on the first date, which has no predecessor) and `n_active`.
    Entries and exits of competitors are what move the series. Each competitor's influence is
    computed once, so a router is queried once per competitor rather than once per date.
    """
    days = [_as_date(d) for d in dates]
    if not days:
        raise ValueError("at least one date is required")
    if any(later <= earlier for earlier, later in zip(days, days[1:])):
        raise ValueError("dates must be strictly increasing")
    p = _params(
        business,
        travel,
        distance_lambda,
        catchment_lambda,
        price_tau,
        category_weight,
        embedding_dim,
        market_radius_km,
    )
    items = [c for c in _unique(competitors) if any(is_active(c, day) for day in days)]
    table = _influence_table(business, items, p)
    by_id = dict(zip(table["competitor_id"], table["influence"].to_numpy(dtype=np.float64)))
    in_market = [(c, by_id[c.id]) for c in items if c.id in by_id]
    levels: list[float] = []
    counts: list[int] = []
    for day in days:
        active = [value for c, value in in_market if is_active(c, day)]
        levels.append(float(np.sum(active)) if active else 0.0)
        counts.append(len(active))
    frame = pd.DataFrame(
        {"competition": levels, "n_active": counts},
        index=pd.DatetimeIndex(pd.to_datetime(days), name=DATE),
    )
    frame.insert(1, "delta_competition", frame["competition"].diff())
    return frame
