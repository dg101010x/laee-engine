"""Business constraints on recommended prices (spec §65 price discretisation, §66 constraints).

Prices that a business may charge are described in two layers:

* :class:`PriceBounds` turns :class:`PriceConstraints` into a feasible interval
  ``[lower, upper]`` and remembers which constraint sets each limit.
* :class:`PriceLattice` lists the allowed price points inside that interval when the
  business wants psychological endings (``price_endings="retail_99"`` gives .49/.99 for
  decimal currencies) or whole-unit rounding (``round_to=5`` or ``10`` for currencies
  quoted without cents, such as KES or NGN).

Snapping always picks lattice points inside ``[lower, upper]``, so it can never make a
price infeasible.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from laee.simulation.price import price_grid, stepped_prices

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

PRICE_ENDINGS: dict[str, tuple[str, str]] = {"retail_99": ("0.49", "0.5")}
"""Named price endings as ``(offset, spacing)`` of the lattice ``offset + k * spacing``."""

CONSTRAINT_ORDER: tuple[str, ...] = (
    "min_margin",
    "business_min",
    "max_rel_change",
    "business_max",
    "inventory",
    "price_endings",
    "round_to",
)

_INDEX_TOL = 1e-9

__all__ = [
    "CONSTRAINT_ORDER",
    "PRICE_ENDINGS",
    "PriceBounds",
    "PriceConstraints",
    "PriceLattice",
    "candidate_prices",
    "order_constraints",
    "price_bounds",
    "probe_prices",
    "within_inventory",
]


def order_constraints(names: Iterable[str]) -> tuple[str, ...]:
    """Deduplicate constraint names and sort them in :data:`CONSTRAINT_ORDER`."""
    unique = set(names)
    return tuple(name for name in CONSTRAINT_ORDER if name in unique)


def _decimals(*texts: str) -> int:
    exponents = [Decimal(text).as_tuple().exponent for text in texts]
    return max([0, *(-e for e in exponents if isinstance(e, int))])


@dataclass(frozen=True)
class PriceLattice:
    """Allowed price points ``offset + k * spacing`` for integer ``k``, rounded to ``decimals``."""

    offset: float
    spacing: float
    decimals: int

    def points(self, k: ArrayLike) -> FloatArray:
        steps = np.asarray(k, dtype=np.float64)
        return np.round(self.offset + steps * self.spacing, self.decimals)

    def _point(self, k: int) -> float:
        return float(self.points(k))

    def index_range(self, lower: float, upper: float) -> tuple[int, int]:
        """Indices of the first and last lattice points inside ``[lower, upper]``.

        The range is empty when the first index exceeds the last.
        """
        k_lo = math.ceil((lower - self.offset) / self.spacing - _INDEX_TOL)
        k_hi = math.floor((upper - self.offset) / self.spacing + _INDEX_TOL)
        if self._point(k_lo) < lower:
            k_lo += 1
        if self._point(k_hi) > upper:
            k_hi -= 1
        return k_lo, k_hi

    def snap(self, prices: ArrayLike, lower: float, upper: float) -> FloatArray:
        """Map each price to the nearest lattice point inside ``[lower, upper]``."""
        k_lo, k_hi = self.index_range(lower, upper)
        if k_lo > k_hi:
            raise ValueError(f"no allowed price point lies in [{lower}, {upper}]")
        values = np.asarray(prices, dtype=np.float64)
        k = np.clip(np.rint((values - self.offset) / self.spacing), k_lo, k_hi)
        return self.points(k)

    def next_above(self, price: float) -> float:
        """Smallest lattice point strictly above ``price``."""
        k = math.floor((price - self.offset) / self.spacing)
        while self._point(k) > price:
            k -= 1
        while self._point(k) <= price:
            k += 1
        return self._point(k)

    def next_below(self, price: float) -> float | None:
        """Largest positive lattice point strictly below ``price``, if any."""
        k = math.ceil((price - self.offset) / self.spacing)
        while self._point(k) < price:
            k += 1
        while self._point(k) >= price:
            k -= 1
        point = self._point(k)
        return point if point > 0.0 else None


def _optional_positive(name: str, value: float | None) -> None:
    if value is not None and not (math.isfinite(value) and value > 0.0):
        raise ValueError(f"{name} must be positive and finite, got {value!r}")


@dataclass(frozen=True)
class PriceConstraints:
    """Constraints every recommended price must satisfy (spec §66).

    ``min_margin`` is a markup on unit cost: ``price >= cost * (1 + min_margin)``.
    ``max_rel_change`` caps the move in either direction, so prices stay within
    ``current * (1 +/- max_rel_change)``. ``business_min`` and ``business_max`` are
    absolute limits set by the business (legal, competitor or brand limits).
    ``price_endings`` (``"retail_99"``) and ``round_to`` (for example 5 or 10) restrict
    new prices to allowed price points and cannot be combined. Keeping the current price
    is always allowed when it lies within the bounds, even if it is not on the lattice.
    ``max_units`` drops prices whose expected quantity per period exceeds the stock
    available.
    """

    min_margin: float = 0.10
    max_rel_change: float = 0.15
    business_min: float | None = None
    business_max: float | None = None
    price_endings: str | None = None
    round_to: float | None = None
    max_units: float | None = None

    def __post_init__(self) -> None:
        if not (math.isfinite(self.min_margin) and self.min_margin >= 0.0):
            raise ValueError(f"min_margin must be non-negative, got {self.min_margin!r}")
        if not (math.isfinite(self.max_rel_change) and 0.0 < self.max_rel_change < 1.0):
            raise ValueError(f"max_rel_change must lie in (0, 1), got {self.max_rel_change!r}")
        for name in ("business_min", "business_max", "round_to", "max_units"):
            _optional_positive(name, getattr(self, name))
        if (
            self.business_min is not None
            and self.business_max is not None
            and self.business_min > self.business_max
        ):
            raise ValueError("business_min exceeds business_max")
        if self.price_endings is not None and self.price_endings not in PRICE_ENDINGS:
            raise ValueError(
                f"unknown price_endings {self.price_endings!r}; use one of {sorted(PRICE_ENDINGS)}"
            )
        if self.price_endings is not None and self.round_to is not None:
            raise ValueError("choose either price_endings or round_to, not both")

    @property
    def lattice(self) -> PriceLattice | None:
        """Allowed price points for new prices, or None when any price is allowed."""
        if self.price_endings is not None:
            offset, spacing = PRICE_ENDINGS[self.price_endings]
            return PriceLattice(float(offset), float(spacing), _decimals(offset, spacing))
        if self.round_to is not None:
            return PriceLattice(0.0, float(self.round_to), _decimals(repr(self.round_to)))
        return None

    @property
    def snapping(self) -> str | None:
        """Name of the active snapping constraint, if any."""
        if self.price_endings is not None:
            return "price_endings"
        if self.round_to is not None:
            return "round_to"
        return None


@dataclass(frozen=True)
class PriceBounds:
    """Feasible price interval and the limit each constraint imposes on it."""

    lower: float
    upper: float
    lower_limits: dict[str, float]
    upper_limits: dict[str, float]

    @property
    def feasible(self) -> bool:
        return self.lower <= self.upper

    def contains(self, price: float) -> bool:
        return self.lower <= price <= self.upper

    def violated_by(self, price: float) -> tuple[str, ...]:
        """Constraints whose limits exclude ``price``."""
        names = [name for name, limit in self.lower_limits.items() if price < limit]
        names += [name for name, limit in self.upper_limits.items() if price > limit]
        return order_constraints(names)

    def conflicting(self) -> tuple[str, ...]:
        """Constraints whose limits cross each other; empty when the bounds are feasible."""
        if self.feasible:
            return ()
        names = [name for name, limit in self.lower_limits.items() if limit > self.upper]
        names += [name for name, limit in self.upper_limits.items() if limit < self.lower]
        return order_constraints(names)


def price_bounds(
    current_price: float, unit_cost: float | None, constraints: PriceConstraints
) -> PriceBounds:
    """Feasible price interval of spec §66.

    ``lower = max(cost * (1 + min_margin), business_min, current * (1 - max_rel_change))``
    and ``upper = min(current * (1 + max_rel_change), business_max)``. Without a unit cost
    the margin limit is left out.
    """
    if not (math.isfinite(current_price) and current_price > 0.0):
        raise ValueError(f"current_price must be positive, got {current_price!r}")
    down, up = stepped_prices(current_price, [-1, 1], constraints.max_rel_change)
    lower = {"max_rel_change": float(down)}
    upper = {"max_rel_change": float(up)}
    if unit_cost is not None:
        lower["min_margin"] = float(stepped_prices(unit_cost, [1], constraints.min_margin)[0])
    if constraints.business_min is not None:
        lower["business_min"] = float(constraints.business_min)
    if constraints.business_max is not None:
        upper["business_max"] = float(constraints.business_max)
    return PriceBounds(max(lower.values()), min(upper.values()), lower, upper)


def candidate_prices(
    current_price: float,
    bounds: PriceBounds,
    constraints: PriceConstraints,
    *,
    step: float,
) -> FloatArray:
    """Sorted feasible candidate prices: the relative ``step`` grid inside the bounds plus
    both bounds, snapped to the lattice when one is configured, plus the current price
    when it lies within the bounds. Empty when nothing is feasible.
    """
    if not bounds.feasible:
        return np.empty(0)
    rel_range = (bounds.lower / current_price - 1.0, bounds.upper / current_price - 1.0)
    grid = price_grid(current_price, rel_range, step)
    inside = grid[(grid >= bounds.lower) & (grid <= bounds.upper)]
    raw = np.concatenate([inside, [bounds.lower, bounds.upper]])
    lattice = constraints.lattice
    if lattice is not None:
        k_lo, k_hi = lattice.index_range(bounds.lower, bounds.upper)
        raw = np.empty(0) if k_lo > k_hi else lattice.snap(raw, bounds.lower, bounds.upper)
    if bounds.contains(current_price):
        raw = np.append(raw, current_price)
    return np.unique(raw)


def probe_prices(
    current_price: float,
    bounds: PriceBounds,
    constraints: PriceConstraints,
    *,
    step: float,
) -> FloatArray:
    """The next allowed price just beyond each bound (one grid step or lattice point).

    The optimiser evaluates these to tell whether a bound is binding.
    """
    lattice = constraints.lattice
    if lattice is not None:
        above: float | None = lattice.next_above(bounds.upper)
        below = lattice.next_below(bounds.lower)
    else:
        k_up = math.floor((bounds.upper / current_price - 1.0) / step + _INDEX_TOL) + 1
        k_down = math.ceil((bounds.lower / current_price - 1.0) / step - _INDEX_TOL) - 1
        above, below = (float(p) for p in stepped_prices(current_price, [k_up, k_down], step))
        while above <= bounds.upper:
            k_up += 1
            above = float(stepped_prices(current_price, [k_up], step)[0])
        while below >= bounds.lower:
            k_down -= 1
            below = float(stepped_prices(current_price, [k_down], step)[0])
    probes = [p for p in (below, above) if p is not None and p > 0.0]
    return np.array(probes, dtype=np.float64)


def within_inventory(expected_quantity: ArrayLike, constraints: PriceConstraints) -> BoolArray:
    """True where the expected quantity fits within ``max_units`` (always true without it)."""
    quantity = np.asarray(expected_quantity, dtype=np.float64)
    if constraints.max_units is None:
        return np.ones(quantity.shape, dtype=bool)
    return quantity <= constraints.max_units
