"""LAEE elasticity models: fixed-effects baselines (tier 6) and Double Machine Learning (tier 5)."""

from laee.models.dml import (
    benchmark_nuisance,
    compare_estimates,
    estimate_dml,
    grouped_folds,
    median_aggregate,
    partialling_out,
    time_blocked_folds,
)
from laee.models.fe import demean, estimate_loglog_ols, estimate_twfe
from laee.models.inference import (
    BootstrapResult,
    EstimationWarning,
    cluster_bootstrap_ci,
    ols_vcov,
    sandwich_vcov,
    t_intervals,
)
from laee.models.learners import (
    Learner,
    LearnerUnavailableError,
    RidgeLearner,
    available_learners,
    benchmark_learners,
    build_nuisance_features,
    make_learner,
)

__all__ = [
    "BootstrapResult",
    "EstimationWarning",
    "Learner",
    "LearnerUnavailableError",
    "RidgeLearner",
    "available_learners",
    "benchmark_learners",
    "benchmark_nuisance",
    "build_nuisance_features",
    "cluster_bootstrap_ci",
    "compare_estimates",
    "demean",
    "estimate_dml",
    "estimate_loglog_ols",
    "estimate_twfe",
    "grouped_folds",
    "make_learner",
    "median_aggregate",
    "ols_vcov",
    "partialling_out",
    "sandwich_vcov",
    "t_intervals",
    "time_blocked_folds",
]
