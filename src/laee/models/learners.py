"""Nuisance learners and features for Double Machine Learning (spec §16-17).

The core path is numpy-only: :class:`RidgeLearner` solves standardised ridge regression in
closed form. scikit-learn learners (histogram gradient boosting, random forests) are optional
and imported lazily by :func:`make_learner`. Spec §17 says to benchmark learners rather than
assume one is best: :func:`benchmark_learners` reports out-of-fold RMSE on caller-supplied folds,
and ``laee.models.dml.benchmark_nuisance`` runs it on exactly the transformed data and folds of
a DML fit.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from laee import schema
from laee.models.inference import day_numbers

FoldSplit = tuple[np.ndarray, np.ndarray]
YEAR_DAYS = 365.25
MAX_WEEKLY_HARMONICS = 3

_SKLEARN_DEFAULTS: dict[str, dict[str, Any]] = {
    "hgb": {
        "max_iter": 200,
        "learning_rate": 0.05,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "early_stopping": False,
        "random_state": 0,
    },
    "rf": {
        "n_estimators": 200,
        "min_samples_leaf": 5,
        "max_features": 0.5,
        "random_state": 0,
    },
}
_ALIASES = {
    "ridge": "ridge",
    "hgb": "hgb",
    "hist_gradient_boosting": "hgb",
    "histgradientboosting": "hgb",
    "rf": "rf",
    "random_forest": "rf",
    "randomforest": "rf",
}


@runtime_checkable
class Learner(Protocol):
    """Minimal regression learner: ``fit`` returns the fitted learner, ``predict`` a 1-d array."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> Learner: ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...


LearnerSpec = str | Learner | Callable[[], Learner]


class LearnerUnavailableError(ImportError):
    """The requested learner needs an optional dependency that is not installed."""


@dataclass(eq=False)
class RidgeLearner:
    """Ridge regression on standardised features, solved in closed form with numpy.

    Minimises ||y - ȳ - Zb||² + alpha·||b||² where Z are the columns centred and scaled to unit
    standard deviation on the training data (constant columns get coefficient 0); the intercept
    is not penalised. ``alpha = 0`` gives minimum-norm least squares.
    """

    alpha: float = 1.0
    mean_: np.ndarray = field(init=False, repr=False)
    scale_: np.ndarray = field(init=False, repr=False)
    coef_: np.ndarray = field(init=False, repr=False)
    intercept_: float = field(init=False, repr=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> RidgeLearner:
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")
        X = np.asarray(X, dtype=float)
        X = X.reshape(X.shape[0], -1)
        y = np.asarray(y, dtype=float).ravel()
        if X.shape[0] != y.shape[0] or X.shape[0] == 0:
            raise ValueError("X and y must have the same, non-zero number of rows")
        self.mean_ = X.mean(axis=0)
        sd = X.std(axis=0)
        self.scale_ = np.where(sd > 0, sd, 1.0)
        z = (X - self.mean_) / self.scale_
        y_mean = float(y.mean())
        if self.alpha > 0:
            gram = z.T @ z + self.alpha * np.eye(z.shape[1])
            coef = np.linalg.solve(gram, z.T @ (y - y_mean))
        else:
            coef = np.linalg.lstsq(z, y - y_mean, rcond=None)[0]
        self.coef_ = coef / self.scale_
        self.intercept_ = y_mean - float(self.mean_ @ self.coef_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return X.reshape(X.shape[0], -1) @ self.coef_ + self.intercept_


def available_learners() -> tuple[str, ...]:
    """Canonical learner names usable in this environment."""
    try:
        import sklearn.ensemble  # noqa: F401
    except ImportError:
        return ("ridge",)
    return ("ridge", "hgb", "rf")


def make_learner(name: str, **params: Any) -> Learner:
    """Instantiate a learner by name: ``ridge``; with scikit-learn, ``hgb`` and ``rf``.

    ``params`` override the defaults (e.g. ``make_learner("hgb", max_iter=100)``).
    """
    key = _ALIASES.get(name.lower().strip())
    if key is None:
        raise ValueError(f"unknown learner {name!r}; choose from {sorted(set(_ALIASES))}")
    if key == "ridge":
        return RidgeLearner(**params)
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    except ImportError as exc:
        raise LearnerUnavailableError(
            f"learner {name!r} needs scikit-learn, which is not installed; install the optional "
            "extra (pip install 'laee[ml]') or use learner 'ridge'"
        ) from exc
    estimator = HistGradientBoostingRegressor if key == "hgb" else RandomForestRegressor
    return estimator(**{**_SKLEARN_DEFAULTS[key], **params})


def clone_learner(learner: Learner) -> Learner:
    """Unfitted copy of a learner (scikit-learn ``clone`` when available, else a deep copy)."""
    if hasattr(learner, "get_params"):
        try:
            from sklearn.base import clone
        except ImportError:
            return copy.deepcopy(learner)
        return clone(learner)
    return copy.deepcopy(learner)


def learner_factory(spec: LearnerSpec) -> Callable[[], Learner]:
    """Zero-argument constructor giving a fresh learner for every cross-fitting fit.

    ``spec`` may be a learner name, a learner class, a learner instance (cloned per fit) or a
    callable returning a new learner.
    """
    if isinstance(spec, str):
        name = spec
        make_learner(name)
        return lambda: make_learner(name)
    if isinstance(spec, type):
        return spec
    if hasattr(spec, "fit") and hasattr(spec, "predict"):
        template = spec
        return lambda: clone_learner(template)
    if callable(spec):
        return spec
    raise TypeError(f"cannot build learners from {spec!r}")


def learner_label(spec: LearnerSpec) -> str:
    if isinstance(spec, str):
        return _ALIASES.get(spec.lower().strip(), spec)
    if isinstance(spec, type):
        return spec.__name__
    if hasattr(spec, "fit"):
        return type(spec).__name__
    return getattr(spec, "__name__", type(spec).__name__)


def build_nuisance_features(
    panel: pd.DataFrame,
    controls: Sequence[str] = (),
    *,
    product_fe: bool = True,
    fourier_annual: int = 3,
    fourier_weekly: int = 2,
) -> pd.DataFrame:
    """Feature matrix X for the nuisance regressions E[ln Q | X] and E[ln P | X].

    Columns, in order: one-hot product dummies ``product[<id>]`` (all levels), annual Fourier
    terms ``annual_sin{k}``/``annual_cos{k}`` (period 365.25 days), weekly Fourier terms
    ``weekly_sin{k}``/``weekly_cos{k}`` (period 7), then the controls: numeric and boolean
    columns as floats under their own names, other columns one-hot encoded as ``name[level]``.
    Fourier terms use absolute day numbers, so phases agree across panels and year ends.
    Missing control values stay NaN. Dense one-hot dummies cost O(rows × products) memory.
    """
    if fourier_annual < 0 or not 0 <= fourier_weekly <= MAX_WEEKLY_HARMONICS:
        raise ValueError(f"fourier_annual must be >= 0 and fourier_weekly in [0, {MAX_WEEKLY_HARMONICS}]")
    n_obs = len(panel)
    blocks: list[np.ndarray] = []
    names: list[str] = []
    if product_fe:
        codes, uniques = pd.factorize(panel[schema.PRODUCT_ID], sort=True)
        if (codes < 0).any():
            raise ValueError("product identifiers contain missing values")
        blocks.append(_one_hot(codes, len(uniques)))
        names.extend(f"product[{level}]" for level in uniques)
    if fourier_annual or fourier_weekly:
        day = day_numbers(panel[schema.DATE]).astype(float)
        harmonics = (("annual", YEAR_DAYS, fourier_annual), ("weekly", 7.0, fourier_weekly))
        for label, period, n_harmonics in harmonics:
            for k in range(1, n_harmonics + 1):
                angle = 2.0 * np.pi * k * day / period
                blocks.append(np.column_stack([np.sin(angle), np.cos(angle)]))
                names.extend([f"{label}_sin{k}", f"{label}_cos{k}"])
    for name in controls:
        column = panel[name]
        if pd.api.types.is_bool_dtype(column) or pd.api.types.is_numeric_dtype(column):
            blocks.append(column.to_numpy(dtype=float, na_value=np.nan)[:, None])
            names.append(name)
        else:
            codes, uniques = pd.factorize(column, sort=True)
            dummies = _one_hot(np.where(codes < 0, 0, codes), len(uniques))
            dummies[codes < 0] = np.nan
            blocks.append(dummies)
            names.extend(f"{name}[{level}]" for level in uniques)
    data = np.column_stack(blocks) if blocks else np.empty((n_obs, 0))
    return pd.DataFrame(data, columns=names, index=panel.index)


def _one_hot(codes: np.ndarray, n_levels: int) -> np.ndarray:
    dummies = np.zeros((len(codes), n_levels))
    dummies[np.arange(len(codes)), codes] = 1.0
    return dummies


def cross_fit_predict(
    X: np.ndarray, y: np.ndarray, folds: Sequence[FoldSplit], factory: Callable[[], Learner]
) -> np.ndarray:
    """Out-of-fold predictions: for each (train, test) split fit a fresh learner on train rows."""
    predictions = np.full(len(y), np.nan)
    for train, test in folds:
        if len(train) == 0:
            raise ValueError("a cross-fitting fold has no training rows")
        model = factory().fit(X[train], y[train])
        predictions[test] = np.asarray(model.predict(X[test]), dtype=float).ravel()
    if np.isnan(predictions).any():
        raise ValueError("folds must assign every row to exactly one held-out set")
    return predictions


def benchmark_learners(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    folds: Sequence[FoldSplit],
    names: Sequence[str] = ("ridge",),
) -> dict[str, float]:
    """Out-of-fold RMSE of each named learner, pooled over all held-out rows of ``folds``."""
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float).ravel()
    scores: dict[str, float] = {}
    for name in names:
        squared, count = 0.0, 0
        for train, test in folds:
            model = make_learner(name).fit(X_arr[train], y_arr[train])
            error = y_arr[test] - np.asarray(model.predict(X_arr[test]), dtype=float).ravel()
            squared += float(error @ error)
            count += len(test)
        if count == 0:
            raise ValueError("folds contain no held-out rows")
        scores[name] = float(np.sqrt(squared / count))
    return scores


__all__ = [
    "FoldSplit",
    "Learner",
    "LearnerSpec",
    "LearnerUnavailableError",
    "RidgeLearner",
    "available_learners",
    "benchmark_learners",
    "build_nuisance_features",
    "clone_learner",
    "cross_fit_predict",
    "learner_factory",
    "learner_label",
    "make_learner",
]
