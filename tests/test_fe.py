"""Fixed-effects baselines and shared inference helpers, proven on known answers."""

from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from laee import schema
from laee.models.fe import absorbed_dof, demean, estimate_loglog_ols, estimate_twfe
from laee.models.inference import (
    EstimationWarning,
    cluster_bootstrap_ci,
    factorize,
    ols_vcov,
    t_intervals,
)
from laee.synthetic import DEMAND_PROXY, simulate_panel

CONTROLS = (DEMAND_PROXY, schema.PROMO_FLAG)
TOLERANCE = 0.15


def _one_hot(codes: np.ndarray) -> np.ndarray:
    codes = factorize(codes)[0]
    return np.eye(codes.max() + 1)[codes]


def _week_codes(panel: pd.DataFrame) -> np.ndarray:
    days = pd.to_datetime(panel[schema.DATE]).to_numpy().astype("datetime64[D]").astype(np.int64)
    return factorize((days + 3) // 7)[0]


# --------------------------------------------------------------------------- inference helpers


def test_cr1_and_hc1_match_hand_computed_example():
    x = [Fraction(v) for v in (1, 2, 3, 4, 5, 6, 7, 8)]
    y = [Fraction(v) for v in (2, 1, 4, 3, 6, 5, 9, 7)]
    groups = [0, 0, 1, 1, 2, 2, 3, 3]
    n, n_groups, k = len(x), 4, 2
    sx, sxx = sum(x), sum(v * v for v in x)
    sy, sxy = sum(y), sum(a * b for a, b in zip(x, y))
    det = n * sxx - sx * sx
    slope = (n * sxy - sx * sy) / det
    intercept = (sy - slope * sx) / n
    resid = [b - intercept - slope * a for a, b in zip(x, y)]
    inv = [[sxx / det, -sx / det], [-sx / det, Fraction(n) / det]]  # (X'X)^-1

    def sandwich_slope_variance(meat, factor):
        row = inv[1]
        return factor * sum(row[i] * meat[i][j] * row[j] for i in range(2) for j in range(2))

    sums = {g: [Fraction(0), Fraction(0)] for g in range(n_groups)}
    for a, u, g in zip(x, resid, groups):
        sums[g][0] += u
        sums[g][1] += a * u
    cluster_meat = [[sum(s[i] * s[j] for s in sums.values()) for j in range(2)] for i in range(2)]
    cr1 = sandwich_slope_variance(
        cluster_meat, Fraction(n_groups, n_groups - 1) * Fraction(n - 1, n - k)
    )
    hc_meat = [[sum(u * u * (1, a)[i] * (1, a)[j] for a, u in zip(x, resid)) for j in range(2)]
               for i in range(2)]
    hc1 = sandwich_slope_variance(hc_meat, Fraction(n, n - k))

    X = np.column_stack([np.ones(n), np.array(x, dtype=float)])
    u = np.array(resid, dtype=float)
    clustered = ols_vcov(X, u, cluster_codes=np.array(groups), min_clusters=2)
    assert clustered.kind == "CR1" and clustered.df == n_groups - 1
    assert clustered.vcov[1, 1] == pytest.approx(float(cr1), rel=1e-12)
    fallback = ols_vcov(X, u, cluster_codes=np.array(groups))
    assert fallback.kind == "HC1" and fallback.df == n - k and fallback.warnings
    assert fallback.vcov[1, 1] == pytest.approx(float(hc1), rel=1e-12)


def test_t_intervals_use_requested_degrees_of_freedom():
    narrow, wide = t_intervals(-1.5, 0.1, df=19)
    assert (narrow.level, wide.level) == (0.8, 0.95)
    assert wide.upper == pytest.approx(-1.5 + 2.093024 * 0.1, abs=1e-6)
    assert narrow.lower == pytest.approx(-1.5 - 1.327728 * 0.1, abs=1e-6)
    normal = t_intervals(0.0, 1.0, df=None, levels=(0.95,))[0]
    assert normal.upper == pytest.approx(1.959964, abs=1e-6)


def test_alternating_projections_equal_least_squares_on_dummies(rng):
    n = 600
    first = rng.integers(0, 15, n)
    second = rng.integers(0, 40, n)
    values = rng.normal(size=(n, 2)) + 0.3 * first[:, None] - 0.1 * second[:, None]
    groups = [factorize(first)[0], factorize(second)[0]]
    dummies = np.column_stack([_one_hot(first), _one_hot(second)])
    result = demean(values, groups)
    expected = values - dummies @ np.linalg.lstsq(dummies, values, rcond=None)[0]
    assert result.converged
    np.testing.assert_allclose(result.values, expected, atol=1e-8)
    weights = rng.uniform(0.2, 3.0, n)
    root = np.sqrt(weights)[:, None]
    weighted = demean(values, groups, weights=weights)
    coef = np.linalg.lstsq(dummies * root, values * root, rcond=None)[0]
    np.testing.assert_allclose(weighted.values, values - dummies @ coef, atol=1e-8)


def test_absorbed_dof_counts_connected_components_and_nesting():
    products = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    weeks = np.array([0, 1, 0, 1, 2, 3, 2, 3])
    rank = np.linalg.matrix_rank(np.column_stack([_one_hot(products), _one_hot(weeks)]))
    k_cluster, k_hc = absorbed_dof([products, weeks], products)
    assert k_hc == rank == 6
    assert k_cluster == rank - 4
    crossed = np.array([0, 1, 1, 0, 0, 1, 1, 0])
    assert absorbed_dof([products, weeks], crossed) == (6, 6)


# --------------------------------------------------------------------------- OLS TWFE


def test_twfe_equals_dummy_variable_regression_with_cr1_on_unbalanced_panel(rng):
    panel, _ = simulate_panel(rng, n_products=12, n_days=60, sparsity=0.3)
    est = estimate_twfe(panel, controls=CONTROLS, time_fe="week")
    y = np.log(panel[schema.UNITS].to_numpy())
    x = np.column_stack([np.log(panel[schema.PRICE]), panel[DEMAND_PROXY],
                         panel[schema.PROMO_FLAG].astype(float)])
    products = factorize(panel[schema.PRODUCT_ID])[0]
    dummies = np.column_stack([_one_hot(products), _one_hot(_week_codes(panel))])
    design = np.column_stack([x, dummies])
    coef = np.linalg.lstsq(design, y, rcond=None)[0]
    resid = y - design @ coef
    x_tilde = x - dummies @ np.linalg.lstsq(dummies, x, rcond=None)[0]
    n_obs, n_clusters = len(y), 12
    k = x.shape[1] + np.linalg.matrix_rank(dummies) - n_clusters  # product FE nested in clusters
    bread = np.linalg.inv(x_tilde.T @ x_tilde)
    sums = np.vstack([(x_tilde[products == g] * resid[products == g, None]).sum(axis=0)
                      for g in range(n_clusters)])
    factor = n_clusters / (n_clusters - 1) * (n_obs - 1) / (n_obs - k)
    vcov = factor * bread @ sums.T @ sums @ bread
    assert est.diagnostics["n_singletons_dropped"] == 0
    assert est.value == pytest.approx(coef[0], abs=1e-8)
    assert est.std_error == pytest.approx(np.sqrt(vcov[0, 0]), rel=1e-6)
    proxy = est.diagnostics["control_coefficients"][DEMAND_PROXY]
    assert proxy["coef"] == pytest.approx(coef[1], abs=1e-8)
    assert proxy["std_error"] == pytest.approx(np.sqrt(vcov[1, 1]), rel=1e-6)
    resid_dm = demean(resid, [products, _week_codes(panel)]).values
    assert est.diagnostics["within_r2"] == pytest.approx(
        1 - resid_dm @ resid_dm / (demean(y, [products, _week_codes(panel)]).values ** 2).sum(), rel=1e-8
    )


def test_twfe_recovers_truth_where_naive_regression_fails():
    panel, truth = simulate_panel(np.random.default_rng(2026), confounding=0.8)
    beta = truth["elasticity"]
    naive = estimate_loglog_ols(panel)
    assert abs(naive.value - beta) > 0.3
    assert naive.identification_tier == 7 and naive.method == "loglog_ols"
    without_proxy = estimate_twfe(panel, controls=[schema.PROMO_FLAG], time_fe="day")
    assert without_proxy.value - beta > 0.2
    for time_fe in ("day", "week"):
        est = estimate_twfe(panel, controls=CONTROLS, time_fe=time_fe)
        assert abs(est.value - beta) < TOLERANCE
        assert est.identification_tier == 6 and est.method == "twfe_ols"
        assert [iv.level for iv in est.intervals] == [0.8, 0.95]
        wide, narrow = est.interval(0.95), est.interval(0.8)
        assert wide.lower < narrow.lower < est.value < narrow.upper < wide.upper
        diagnostics = est.diagnostics
        assert diagnostics["n_clusters"] == 20 and diagnostics["se_type"] == "CR1"
        assert diagnostics["df"] == 19 and diagnostics["warnings"] == []
        assert diagnostics["n_obs"] == est.n_obs == len(panel)
        assert diagnostics["n_dropped_nonpositive"] == 0
        assert diagnostics["n_products_with_price_variation"] == 20
        assert 0 < diagnostics["within_r2"] < 1 and diagnostics["sd_demeaned_log_price"] > 0.05
        assert diagnostics["control_coefficients"][DEMAND_PROXY]["coef"] == pytest.approx(1.0, abs=0.1)


def test_twfe_95_percent_intervals_cover_the_truth():
    hits = []
    for seed in range(50):
        panel, truth = simulate_panel(np.random.default_rng(seed), n_days=150)
        interval = estimate_twfe(panel, controls=CONTROLS, time_fe="day").interval(0.95)
        hits.append(interval.lower <= truth["elasticity"] <= interval.upper)
    assert np.mean(hits) >= 0.85


def test_few_clusters_fall_back_to_hc1_with_a_warning():
    panel, _ = simulate_panel(np.random.default_rng(4), n_products=6, n_days=200)
    with pytest.warns(EstimationWarning, match="HC1"):
        est = estimate_twfe(panel, controls=CONTROLS, time_fe="day")
    diagnostics = est.diagnostics
    assert diagnostics["se_type"] == "HC1" and diagnostics["n_clusters"] == 6
    assert diagnostics["df"] == est.n_obs - 3 - diagnostics["absorbed_dof"]
    assert any("HC1" in message for message in diagnostics["warnings"])


def test_rows_with_zero_units_are_dropped_and_counted(rng):
    panel, _ = simulate_panel(rng, n_products=12, n_days=120, noise="poisson", base_demand=3.0)
    zeros = int((panel[schema.UNITS] == 0).sum())
    assert zeros > 0
    with pytest.warns(EstimationWarning, match="zero or negative units"):
        est = estimate_twfe(panel, controls=CONTROLS, time_fe="week")
    assert est.diagnostics["n_dropped_nonpositive"] == zeros
    assert est.n_obs == len(panel) - zeros - est.diagnostics["n_singletons_dropped"]


def test_excluding_censored_stockout_rows_reduces_bias():
    panel, truth = simulate_panel(np.random.default_rng(5), stockout_rate=0.3)
    beta = truth["elasticity"]
    with pytest.warns(EstimationWarning, match=schema.STOCKOUT_FLAG):
        included = estimate_twfe(panel, controls=CONTROLS, time_fe="day")
    keep = ~(panel[schema.STOCKOUT_FLAG] | panel[schema.CENSORED_FLAG])
    excluded = estimate_twfe(panel, controls=CONTROLS, time_fe="day", mask=keep)
    assert included.value - beta > 0.2
    assert abs(excluded.value - beta) < min(TOLERANCE, abs(included.value - beta) / 2)
    assert included.diagnostics["n_stockout_rows_included"] == int((~keep).sum())
    assert excluded.diagnostics["n_masked_out"] == int((~keep).sum())
    assert excluded.diagnostics["n_stockout_rows_included"] == 0


def test_controls_absorbed_by_fixed_effects_are_dropped(rng):
    panel, _ = simulate_panel(rng, n_products=12, n_days=90)
    panel = panel.assign(category=factorize(panel[schema.PRODUCT_ID])[0] % 3)
    with pytest.warns(EstimationWarning, match="collinear"):
        est = estimate_twfe(panel, controls=["category", DEMAND_PROXY], time_fe="week")
    assert est.diagnostics["dropped_controls"] == ["category"]
    assert list(est.diagnostics["control_coefficients"]) == [DEMAND_PROXY]


def test_price_without_within_variation_is_not_identified(rng):
    panel, _ = simulate_panel(rng, n_products=12, n_days=60)
    flat = panel.assign(**{schema.PRICE: panel.groupby(schema.PRODUCT_ID)[schema.PRICE].transform("first")})
    with pytest.raises(ValueError, match="no variation"):
        estimate_twfe(flat)


def test_singleton_fixed_effect_groups_are_dropped(rng):
    panel, _ = simulate_panel(rng, n_products=12, n_days=60)
    last_day = panel[schema.DATE].max()
    lone = panel[(panel[schema.DATE] != last_day) | (panel[schema.PRODUCT_ID] == "P000")]
    est = estimate_twfe(lone, controls=CONTROLS, time_fe="day")
    assert est.diagnostics["n_singletons_dropped"] == 1
    assert est.n_obs == len(lone) - 1


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"method": "gmm"}, ValueError),
        ({"time_fe": "quarter"}, ValueError),
        ({"controls": ["missing_column"]}, KeyError),
        ({"mask": np.ones(3, dtype=bool)}, ValueError),
    ],
)
def test_invalid_arguments_are_rejected(rng, kwargs, error):
    panel, _ = simulate_panel(rng, n_products=12, n_days=30)
    with pytest.raises(error):
        estimate_twfe(panel, **kwargs)


# --------------------------------------------------------------------------- PPML


def _poisson_newton(y: np.ndarray, design: np.ndarray) -> np.ndarray:
    coef = np.linalg.lstsq(design, np.log(y + 0.5), rcond=None)[0]
    for _ in range(100):
        mu = np.exp(design @ coef)
        step = np.linalg.solve((design * mu[:, None]).T @ design, design.T @ (y - mu))
        coef += step
        if np.max(np.abs(step)) < 1e-13:
            break
    return coef


def test_ppml_equals_brute_force_poisson_glm_with_dummies(rng):
    panel, _ = simulate_panel(rng, n_products=12, n_days=35, noise="poisson", noise_sd=0.0, base_demand=2.0)
    est = estimate_twfe(panel, controls=CONTROLS, time_fe="week", method="ppml")
    y = panel[schema.UNITS].to_numpy()
    x = np.column_stack([np.log(panel[schema.PRICE]), panel[DEMAND_PROXY],
                         panel[schema.PROMO_FLAG].astype(float)])
    products = factorize(panel[schema.PRODUCT_ID])[0]
    weeks = _one_hot(_week_codes(panel))
    design = np.column_stack([x, _one_hot(products), weeks[:, 1:]])
    coef = _poisson_newton(y, design)
    mu = np.exp(design @ coef)
    hessian_inv = np.linalg.inv((design * mu[:, None]).T @ design)
    scores = design * (y - mu)[:, None]
    sums = np.vstack([scores[products == g].sum(axis=0) for g in range(12)])
    n_obs, k = len(y), x.shape[1] + weeks.shape[1] - 1
    vcov = 12 / 11 * (n_obs - 1) / (n_obs - k) * hessian_inv @ sums.T @ sums @ hessian_inv
    assert est.diagnostics["irls_converged"] and est.method == "twfe_ppml"
    assert est.value == pytest.approx(coef[0], abs=1e-7)
    assert est.std_error == pytest.approx(np.sqrt(vcov[0, 0]), rel=1e-6)
    assert est.diagnostics["control_coefficients"][DEMAND_PROXY]["coef"] == pytest.approx(coef[1], abs=1e-7)


def test_ppml_keeps_zero_sales_and_recovers_truth_where_log_ols_fails():
    panel, truth = simulate_panel(np.random.default_rng(6), n_products=50, noise="poisson",
                                  noise_sd=0.0, base_demand=1.0)
    beta = truth["elasticity"]
    assert (panel[schema.UNITS] == 0).mean() > 0.25
    ppml = estimate_twfe(panel, controls=CONTROLS, time_fe="day", method="ppml")
    with pytest.warns(EstimationWarning, match="zero or negative units"):
        log_ols = estimate_twfe(panel, controls=CONTROLS, time_fe="day")
    error = ppml.value - beta
    assert abs(error) < 0.2 and abs(error) < 3 * ppml.std_error
    assert log_ols.value - beta > 0.4
    diagnostics = ppml.diagnostics
    assert diagnostics["n_dropped_nonpositive"] == 0
    assert diagnostics["n_zero_units"] > 0.25 * ppml.n_obs
    assert ppml.identification_tier == 6 and diagnostics["se_type"] == "CR1"


def test_ppml_drops_products_that_never_sell(rng):
    panel, _ = simulate_panel(rng, n_products=12, n_days=60, noise="poisson", base_demand=2.0)
    dead = panel[schema.PRODUCT_ID] == "P003"
    panel.loc[dead, schema.UNITS] = 0.0
    est = estimate_twfe(panel, controls=CONTROLS, time_fe="week", method="ppml")
    assert est.diagnostics["n_separated_dropped"] == int(dead.sum())
    assert est.n_products == 11 and np.isfinite(est.value)


# --------------------------------------------------------------------------- cluster bootstrap


def test_cluster_bootstrap_resamples_and_relabels_whole_clusters(rng):
    panel, truth = simulate_panel(np.random.default_rng(8), n_products=15, n_days=120)
    analytic = estimate_twfe(panel, controls=CONTROLS, time_fe="day")
    product_counts = []

    def estimator(sample: pd.DataFrame):
        product_counts.append(sample[schema.PRODUCT_ID].nunique())
        return estimate_twfe(sample, controls=CONTROLS, time_fe="day")

    boot = cluster_bootstrap_ci(estimator, panel, schema.PRODUCT_ID, n_boot=60, rng=rng)
    assert boot.n_failed == 0 and len(boot.draws) == 60
    assert product_counts == [15] * 60
    assert [iv.level for iv in boot.intervals] == [0.8, 0.95]
    wide, narrow = boot.interval(0.95), boot.interval(0.8)
    assert wide.lower < narrow.lower < narrow.upper < wide.upper
    assert boot.interval(0.95).lower < truth["elasticity"] < boot.interval(0.95).upper
    assert 0.5 < boot.std_error / analytic.std_error < 2.0


def test_cluster_bootstrap_keeps_products_of_duplicated_clusters_distinct(rng):
    panel, _ = simulate_panel(rng, n_products=10, n_days=20)
    panel["store"] = factorize(panel[schema.PRODUCT_ID])[0] % 4

    def distinct_product_days(sample: pd.DataFrame) -> float:
        assert sample.groupby(["store", schema.PRODUCT_ID]).ngroups == sample[schema.PRODUCT_ID].nunique()
        return float(sample[schema.PRODUCT_ID].nunique() * 20 - len(sample))

    boot = cluster_bootstrap_ci(distinct_product_days, panel, "store", n_boot=20, rng=rng)
    np.testing.assert_array_equal(boot.draws, 0.0)
