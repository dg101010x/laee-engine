"""Double Machine Learning, its cross-fitting and learners, proven on known answers."""

import sys
from contextlib import nullcontext

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from laee import schema
from laee.models.dml import (
    WEAK_VARIATION_SHARE,
    benchmark_nuisance,
    compare_estimates,
    estimate_dml,
    grouped_folds,
    median_aggregate,
    partialling_out,
    time_blocked_folds,
)
from laee.models.fe import estimate_loglog_ols, estimate_twfe
from laee.models.inference import EstimationWarning, factorize, t_intervals
from laee.models.learners import (
    LearnerUnavailableError,
    RidgeLearner,
    available_learners,
    benchmark_learners,
    build_nuisance_features,
    learner_factory,
    make_learner,
)
from laee.synthetic import DEMAND_PROXY, simulate_panel
from laee.types import ElasticityEstimate

CONTROLS = (DEMAND_PROXY, schema.PROMO_FLAG)
TOLERANCE = 0.15


# --------------------------------------------------------------------------- known answers


def test_dml_recovers_truth_where_naive_regression_fails():
    panel, truth = simulate_panel(np.random.default_rng(2026), confounding=0.8)
    beta = truth["elasticity"]
    assert abs(estimate_loglog_ols(panel).value - beta) > 0.3
    est = estimate_dml(panel, CONTROLS, rng=np.random.default_rng(1))
    assert abs(est.value - beta) < TOLERANCE
    assert est.identification_tier == 5 and est.method == "dml_plr"
    assert [iv.level for iv in est.intervals] == [0.8, 0.95]
    diagnostics = est.diagnostics
    assert diagnostics["time_fe"] == "day" and diagnostics["fold_scheme"] == "time_blocked"
    assert len(diagnostics["repeat_estimates"]) == 3 and diagnostics["n_obs"] == len(panel)
    assert 0 < diagnostics["nuisance_r2_y"] < 1 and 0 < diagnostics["nuisance_r2_t"] < 1
    assert diagnostics["residual_price_variance_share"] > WEAK_VARIATION_SHARE
    assert diagnostics["se_type"] == "CR1" and diagnostics["df"] == 19
    assert diagnostics["absorbed_dof"] == 365
    assert diagnostics["warnings"] == []
    proxy_omitted = estimate_dml(panel, [schema.PROMO_FLAG])
    assert proxy_omitted.value - beta > 0.2
    grouped = estimate_dml(panel, CONTROLS, fold_scheme="grouped")
    assert abs(grouped.value - beta) < TOLERANCE
    assert grouped.diagnostics["within_transform"] == "product"
    comparison = compare_estimates(est, estimate_twfe(panel, controls=CONTROLS, time_fe="day"))
    assert comparison["intervals_overlap_95"] and abs(comparison["hausman_z"]) < 2


def test_dml_95_percent_intervals_cover_the_truth():
    hits = []
    for seed in range(50):
        panel, truth = simulate_panel(np.random.default_rng(seed), n_days=150)
        interval = estimate_dml(panel, CONTROLS, rng=np.random.default_rng(seed)).interval(0.95)
        hits.append(interval.lower <= truth["elasticity"] <= interval.upper)
    assert np.mean(hits) >= 0.85


def test_dml_mask_excluding_censored_rows_reduces_bias():
    panel, truth = simulate_panel(np.random.default_rng(5), stockout_rate=0.3)
    beta = truth["elasticity"]
    keep = ~(panel[schema.STOCKOUT_FLAG] | panel[schema.CENSORED_FLAG])
    with pytest.warns(EstimationWarning, match=schema.STOCKOUT_FLAG):
        included = estimate_dml(panel, CONTROLS)
    excluded = estimate_dml(panel, CONTROLS, mask=keep)
    assert included.value - beta > 0.2
    assert abs(excluded.value - beta) < min(TOLERANCE, abs(included.value - beta) / 2)
    assert excluded.diagnostics["n_masked_out"] == int((~keep).sum())


def test_weak_identifying_variation_is_flagged(rng):
    panel, _ = simulate_panel(rng, n_products=12, n_days=120)
    jitter = np.random.default_rng(0).normal(scale=0.002, size=len(panel))
    panel = panel.assign(price_signal=np.log(panel[schema.PRICE]) + jitter)
    with pytest.warns(EstimationWarning, match="weak identifying variation"):
        est = estimate_dml(panel, ["price_signal"])
    assert est.diagnostics["residual_price_variance_share"] < WEAK_VARIATION_SHARE


# --------------------------------------------------------------------------- cross-fitting


class RecordingLearner:
    """Predicts the training mean and records the (product, day) rows it trained and predicted on."""

    def __init__(self, log: list, decode):
        self.log, self.decode = log, decode

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RecordingLearner":
        self.entry = {"train": self.decode(X)}
        self.log.append(self.entry)
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.entry["test"] = self.decode(X)
        return np.full(len(X), self.mean_)


def _keyed_panel(n_products: int, n_days: int) -> pd.DataFrame:
    """Balanced panel with controls a=(i+1)(t+1), b=i+1, c=t+1 identifying product i, day t.

    Within-day demeaning maps (a, b) to ((t+1)·b̃, b̃) and within-product demeaning maps (a, c)
    to ((i+1)·c̃, c̃), so the row stays decodable from the transformed features the learners see
    (b̃ and c̃ are never 0 because n_products and n_days are even).
    """
    panel, _ = simulate_panel(np.random.default_rng(42), n_products=n_products, n_days=n_days)
    product = factorize(panel[schema.PRODUCT_ID])[0].astype(float)
    start = pd.Timestamp(panel[schema.DATE].min())
    day = ((pd.to_datetime(panel[schema.DATE]) - start).dt.days).to_numpy(dtype=float)
    return panel.assign(key_a=(product + 1) * (day + 1), key_b=product + 1, key_c=day + 1)


def _decoder(mode: str, n_products: int, n_days: int):
    def decode(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a, b, c = X[:, -3], X[:, -2], X[:, -1]
        if mode == "day":
            product, day = b + (n_products + 1) / 2 - 1, a / b - 1
        elif mode == "product":
            product, day = a / c - 1, c + (n_days + 1) / 2 - 1
        else:
            product, day = b - 1, c - 1
        for values in (product, day):
            np.testing.assert_allclose(values, np.rint(values), atol=1e-6)
        return np.rint(product).astype(int), np.rint(day).astype(int)

    return decode


@pytest.mark.parametrize(
    ("fold_scheme", "time_fe", "mode"),
    [("time_blocked", "auto", "day"), ("time_blocked", None, "raw"), ("grouped", "auto", "product")],
)
def test_nuisance_models_never_train_on_held_out_or_embargoed_rows(fold_scheme, time_fe, mode):
    n_products, n_days, n_folds, embargo, n_repeats = 10, 120, 4, 10, 2
    panel = _keyed_panel(n_products, n_days)
    log: list = []

    def factory():
        return RecordingLearner(log, _decoder(mode, n_products, n_days))

    seasonal_only = time_fe is None
    expectation = pytest.warns(EstimationWarning, match="time_fe=None") if seasonal_only else nullcontext()
    with expectation:
        estimate_dml(panel, ["key_a", "key_b", "key_c"], learner_y=factory, learner_t=factory,
                     n_folds=n_folds, n_repeats=n_repeats, fold_scheme=fold_scheme,
                     embargo_days=embargo, time_fe=time_fe, rng=np.random.default_rng(0))
    assert len(log) == 2 * n_folds * n_repeats
    all_rows = {(i, t) for i in range(n_products) for t in range(n_days)}
    for start in range(0, len(log), n_folds):
        held_out = [row for entry in log[start:start + n_folds] for row in zip(*entry["test"])]
        assert len(held_out) == len(all_rows) and set(held_out) == all_rows
    for entry in log:
        (train_products, train_days), (test_products, test_days) = entry["train"], entry["test"]
        assert not set(zip(train_products, train_days)) & set(zip(test_products, test_days))
        if fold_scheme == "grouped":
            assert not set(train_products) & set(test_products)
        else:
            first, last = test_days.min(), test_days.max()
            assert np.array_equal(np.unique(test_days), np.arange(first, last + 1))
            assert np.all((train_days < first - embargo) | (train_days > last + embargo))
            outside = [t for t in range(n_days) if t < first - embargo or t > last + embargo]
            assert set(train_days) == set(outside)


def test_time_blocked_folds_are_contiguous_partitions_with_embargo():
    days = np.repeat(np.arange(100), 3)
    folds = time_blocked_folds(days, 5, embargo_days=4)
    assert [len(test) for _, test in folds] == [60] * 5
    assert sorted(np.concatenate([test for _, test in folds]).tolist()) == list(range(300))
    for train, test in folds:
        block = np.unique(days[test])
        assert np.array_equal(block, np.arange(block.min(), block.max() + 1))
        others = np.setdiff1d(np.arange(300), test)
        far = others[(days[others] < block.min() - 4) | (days[others] > block.max() + 4)]
        assert np.array_equal(train, far)
    jittered = time_blocked_folds(days, 5, 4, np.random.default_rng(0))
    assert [len(test) for _, test in jittered] != [60] * 5
    with pytest.raises(ValueError):
        time_blocked_folds(np.arange(3), 5)


def test_grouped_folds_hold_out_whole_products(rng):
    codes = np.repeat(np.arange(13), 7)
    folds = grouped_folds(codes, 5, rng)
    held_out = [np.unique(codes[test]) for _, test in folds]
    assert sorted(np.concatenate(held_out).tolist()) == list(range(13))
    for (train, test), products in zip(folds, held_out):
        assert not set(codes[train]) & set(products)
        assert len(train) + len(test) == len(codes)
    with pytest.raises(ValueError):
        grouped_folds(np.arange(3), 5)


# --------------------------------------------------------------------------- score and aggregation


def test_partialling_out_score_and_cluster_robust_variance(rng):
    n = 300
    clusters = np.repeat(np.arange(12), 25)
    t_resid = rng.normal(size=n)
    y_resid = -1.3 * t_resid + rng.normal(size=n)
    theta = (t_resid @ y_resid) / (t_resid @ t_resid)
    psi = (y_resid - theta * t_resid) * t_resid / np.mean(t_resid**2)
    sums = np.array([psi[clusters == g].sum() for g in range(12)])
    clustered = partialling_out(y_resid, t_resid, clusters)
    assert clustered.theta == pytest.approx(theta, rel=1e-12)
    assert clustered.std_error == pytest.approx(np.sqrt(12 / 11 * (sums @ sums) / n**2), rel=1e-10)
    assert clustered.variance_type == "CR1" and clustered.df == 11
    pointwise = partialling_out(y_resid, t_resid)
    assert pointwise.std_error == pytest.approx(np.sqrt(n / (n - 1) * (psi @ psi) / n**2), rel=1e-10)
    assert pointwise.variance_type == "HC1"
    absorbed = partialling_out(y_resid, t_resid, clusters, n_absorbed_cluster=5)
    factor = 12 / 11 * (n - 1) / (n - 1 - 5)
    assert absorbed.std_error == pytest.approx(np.sqrt(factor * (sums @ sums) / n**2), rel=1e-10)
    with pytest.raises(ValueError):
        partialling_out(y_resid, np.zeros(n))


def test_repeated_cross_fitting_is_median_aggregated(rng):
    panel, _ = simulate_panel(rng, n_products=12, n_days=120)
    est = estimate_dml(panel, CONTROLS, n_repeats=4, rng=np.random.default_rng(3))
    thetas = np.array(est.diagnostics["repeat_estimates"])
    ses = np.array(est.diagnostics["repeat_std_errors"])
    assert len(np.unique(np.round(thetas, 12))) > 1
    median = np.median(thetas)
    assert est.value == pytest.approx(median, rel=1e-12)
    assert est.std_error == pytest.approx(np.sqrt(np.median(ses**2 + (thetas - median) ** 2)), rel=1e-12)
    assert est.interval(0.95).upper == pytest.approx(median + stats.t.ppf(0.975, 11) * est.std_error)
    # median of σ² + (θ - θ̃)² over repeats = median(1.01, 1.01, 0.01)
    assert median_aggregate([1.0, 3.0, 2.0], [0.1, 0.1, 0.1]) == pytest.approx((2.0, np.sqrt(1.01)))


def test_dml_input_validation_and_fallbacks():
    panel, _ = simulate_panel(np.random.default_rng(12), n_products=6, n_days=120)
    with pytest.warns(EstimationWarning, match="HC1"):
        few = estimate_dml(panel, CONTROLS)
    assert few.diagnostics["se_type"] == "HC1"
    with pytest.raises(ValueError):
        estimate_dml(panel, CONTROLS, fold_scheme="grouped", time_fe="day")
    with pytest.raises(ValueError):
        estimate_dml(panel, CONTROLS, fold_scheme="random")
    with pytest.raises(ValueError):
        estimate_dml(panel, CONTROLS, n_folds=8, fold_scheme="grouped")
    with pytest.raises(ValueError):
        estimate_dml(panel, CONTROLS, time_fe="week")
    single, _ = simulate_panel(np.random.default_rng(13), n_products=1, n_days=365)
    with pytest.warns(EstimationWarning, match="HC1"), pytest.warns(EstimationWarning, match="time_fe=None"):
        est = estimate_dml(single, CONTROLS)
    assert est.diagnostics["time_fe"] is None and np.isfinite(est.value)


# --------------------------------------------------------------------------- learners


def test_ridge_learner_matches_closed_form(rng):
    X = rng.normal(size=(200, 4)) * [1.0, 10.0, 0.1, 1.0] + [0.0, 5.0, 0.0, 0.0]
    X[:, 3] = 2.0
    y = X @ [1.0, -0.2, 3.0, 0.0] + 0.5 + rng.normal(scale=0.1, size=200)
    model = RidgeLearner(alpha=2.0).fit(X, y)
    mean, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    z = (X - mean) / sd
    coef = np.linalg.solve(z.T @ z + 2.0 * np.eye(4), z.T @ (y - y.mean()))
    np.testing.assert_allclose(model.predict(X), z @ coef + y.mean(), rtol=1e-12)
    assert model.coef_[3] == 0.0
    ols = RidgeLearner(alpha=0.0).fit(X[:, :3], y)
    design = np.column_stack([np.ones(200), X[:, :3]])
    np.testing.assert_allclose(ols.predict(X[:, :3]), design @ np.linalg.lstsq(design, y, rcond=None)[0])


def test_nuisance_feature_layout(rng):
    panel, _ = simulate_panel(rng, n_products=3, n_days=21)
    panel["segment"] = np.where(panel[schema.PRODUCT_ID] == "P000", "a", "b")
    X = build_nuisance_features(panel, [DEMAND_PROXY, schema.PROMO_FLAG, "segment"])
    fourier = [f"{period}_{f}{k}" for period, n in (("annual", 3), ("weekly", 2))
               for k in range(1, n + 1) for f in ("sin", "cos")]
    assert list(X.columns) == [
        "product[P000]", "product[P001]", "product[P002]", *fourier,
        DEMAND_PROXY, schema.PROMO_FLAG, "segment[a]", "segment[b]",
    ]
    assert X.index.equals(panel.index)
    assert (X.filter(like="product[").sum(axis=1) == 1).all()
    weekly = X.loc[panel[schema.PRODUCT_ID] == "P000", ["weekly_sin1", "weekly_cos2"]].to_numpy()
    np.testing.assert_allclose(weekly[7:], weekly[:-7], atol=1e-9)
    np.testing.assert_allclose(X["annual_sin2"] ** 2 + X["annual_cos2"] ** 2, 1.0)
    minimal = build_nuisance_features(panel, product_fe=False, fourier_annual=0, fourier_weekly=1)
    assert list(minimal.columns) == ["weekly_sin1", "weekly_cos1"]
    with pytest.raises(ValueError):
        build_nuisance_features(panel, fourier_weekly=4)


def test_make_learner_reports_missing_scikit_learn(monkeypatch):
    assert isinstance(make_learner("ridge", alpha=3.0), RidgeLearner)
    with pytest.raises(ValueError, match="unknown learner"):
        make_learner("xgboost")
    monkeypatch.setitem(sys.modules, "sklearn", None)
    monkeypatch.setitem(sys.modules, "sklearn.ensemble", None)
    with pytest.raises(LearnerUnavailableError, match="scikit-learn"):
        make_learner("hgb")
    assert available_learners() == ("ridge",)
    with pytest.raises(LearnerUnavailableError):
        learner_factory("random_forest")


def test_learner_specs_give_fresh_learners():
    template = RidgeLearner(alpha=5.0)
    from_instance = learner_factory(template)
    first, second = from_instance(), from_instance()
    assert first is not template and first is not second and first.alpha == 5.0
    assert isinstance(learner_factory(RidgeLearner)(), RidgeLearner)
    assert learner_factory(lambda: RidgeLearner(alpha=0.5))().alpha == 0.5
    with pytest.raises(TypeError):
        learner_factory(42)


def test_benchmark_learners_reports_out_of_fold_rmse(rng):
    X = rng.normal(size=(400, 3))
    y = X @ [1.0, -2.0, 0.5] + rng.normal(scale=0.3, size=400)
    folds = time_blocked_folds(np.arange(400), 4)
    errors = np.concatenate([
        y[test] - RidgeLearner().fit(X[train], y[train]).predict(X[test]) for train, test in folds
    ])
    scores = benchmark_learners(X, y, folds, ["ridge"])
    assert scores["ridge"] == pytest.approx(np.sqrt(np.mean(errors**2)), rel=1e-12)
    assert scores["ridge"] == pytest.approx(0.3, abs=0.05)


def test_benchmark_nuisance_scores_both_nuisances_on_the_dml_problem(rng):
    panel, _ = simulate_panel(rng, n_products=12, n_days=90)
    scores = benchmark_nuisance(panel, CONTROLS, names=["ridge"], n_folds=3)
    assert set(scores) == {"y", "t"}
    transformed = panel.assign(lp=np.log(panel[schema.PRICE]))
    day_demeaned = transformed["lp"] - transformed.groupby(schema.DATE)["lp"].transform("mean")
    assert 0 < scores["t"]["ridge"] < day_demeaned.std()
    assert scores["y"]["ridge"] > 0


def test_scikit_learn_learners_plug_into_dml():
    pytest.importorskip("sklearn")
    assert set(available_learners()) == {"ridge", "hgb", "rf"}
    hgb = make_learner("hgb", max_iter=60)
    assert hgb.get_params()["max_iter"] == 60 and hgb.get_params()["random_state"] == 0
    rf = make_learner("random_forest", n_estimators=30)
    panel, truth = simulate_panel(np.random.default_rng(9), n_products=10, n_days=120)
    est = estimate_dml(panel, CONTROLS, learner_y=hgb, learner_t=rf, n_folds=2, n_repeats=1)
    assert np.isfinite(est.value) and abs(est.value - truth["elasticity"]) < 0.6
    assert est.diagnostics["learner_y"] == "HistGradientBoostingRegressor"
    assert est.diagnostics["learner_t"] == "RandomForestRegressor"


# --------------------------------------------------------------------------- comparison


def _estimate(value: float, std_error: float, method: str) -> ElasticityEstimate:
    return ElasticityEstimate(value=value, std_error=std_error,
                              intervals=t_intervals(value, std_error, df=None), method=method,
                              identification_tier=5, n_obs=100, n_products=10)


def test_compare_estimates_reports_difference_overlap_and_z():
    a, b = _estimate(-1.5, 0.1, "dml_plr"), _estimate(-1.2, 0.1, "twfe_ols")
    result = compare_estimates(a, b)
    z = -0.3 / np.sqrt(0.02)
    assert result["difference"] == pytest.approx(-0.3)
    assert result["hausman_z"] == pytest.approx(z)
    assert result["p_value"] == pytest.approx(2 * stats.norm.sf(abs(z)))
    assert result["intervals_overlap_95"] is True
    assert "covariance" in result["note"]
    assert compare_estimates(a, _estimate(-0.5, 0.1, "naive"))["intervals_overlap_95"] is False
