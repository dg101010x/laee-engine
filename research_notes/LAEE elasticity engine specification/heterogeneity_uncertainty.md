# Heterogeneous and time-varying elasticity, estimator ensembles, uncertainty quantification and confidence scoring (LAEE v2 research notes)

Scope: replacements for v1 §18–21, §39–40, §43–45, §50–51, for a single small business (about 50–5,000 transactions per product, few price changes, many products, daily data with gaps). Implementation target: Python with numpy/scipy/pandas, optional scikit-learn/LightGBM/EconML.

**Notation used throughout.** Product $j$, day $t$, row $i=(j,t)$. $Y_i=\log Q_i$ (see the zero-sales caveat in §0), $T_i=\log P_i$, $X_i$ = confounders/controls, $Z_i\subseteq X_i$ = heterogeneity features. Nuisances $m(x)=E[Y\mid X=x]$ and $e(x)=E[T\mid X=x]$ are fitted with cross-fitting. Residuals: $\tilde Y_i=Y_i-\hat m^{(-k(i))}(X_i)$ and $\tilde T_i=T_i-\hat e^{(-k(i))}(X_i)$. $\tau(z)$ is the elasticity function. "Independent price change" ($K_{pc}$) is defined in §5.

**How these notes were verified.**
- WebFetch was blocked for arxiv.org, projecteuclid.org, proceedings.neurips.cc, proceedings.mlr.press, grf-labs.github.io, econml.azurewebsites.net, cran.r-project.org and pmc.ncbi.nlm.nih.gov.
- The shared WebSearch budget ran out before the last verification batch.
- **Paper-level facts** come from WebSearch result summaries.
- **(source code)** marks facts read directly from package source downloaded from PyPI: econml 0.17.0, skgrf 0.3.0 (which vendors the GRF C++ core), mapie 1.5.0, river 0.26.1, statsmodels 0.15.0, ruptures 1.1.10 and arch 8.0.0.
- **[lit]** marks standard results I could not check against the full text in this session. Where one of these has a DOI or URL, it comes from prior knowledge and was not fetched.
- **[own computation]** marks results from scratch numpy/scipy simulations I ran this session. They can be reproduced from the formulas given.
- **[heuristic]** marks proposed thresholds that still need simulation tuning.

---

## 0. What is wrong, vague or untestable in v1 §18–21, §39–40, §43–45, §50–51

### Takeaway
v1 names reasonable method families but never specifies an estimand, loss, data requirement or acceptance test. The elasticity function, the regime probabilities, the window and ensemble weights, the interval method and all three scores are left as unspecified $f(\cdot)$ or $w_k$. Several items are also substantively wrong:
- the regime mixture averages elasticities instead of demand curves;
- conformal prediction is proposed for a target that is never observed;
- a pure gradient-boosting demand model is treated as an elasticity estimator;
- correlated ensemble members are combined as if independent.

Finally, "Confidence: 81% / 87%" is not the probability of any defined event, so it can never be calibrated or falsified.

### Cited Findings
- **§14 demand model.** $\ln Q_{it}=\alpha_i+\gamma_t+\beta\ln P_{it}+f(X_{it})+\epsilon_{it}$, where $\gamma_t$ is a time fixed effect (lines 549–561). — [v1 spec §14](/home/user/laee-engine/docs/spec/v1-original.md)
- **§18 heterogeneity.**
  - $\epsilon_i=f(business, product, location, competition, customer, time, shock)$. The subscript has no $t$, although "time" and "shock" are arguments (lines 753–764).
  - The example output includes "Confidence: 81%" and labels such as "Local competition: High" with no thresholds (lines 731–749).
  - — [v1 spec §18](/home/user/laee-engine/docs/spec/v1-original.md)
- **§19 causal forest.**
  - Model: $\ln Q_{it}=\tau(Z_{it})\ln P_{it}+f(X_{it})+u_{it}$ (lines 774–782).
  - Listed heterogeneity variables: income, competition, market density, price tier, product category, customer concentration, business size, local demand volatility, geographic isolation (lines 794–804).
  - Not specified: orthogonalization, leaf size, honesty, clustering, variance method.
  - — [v1 spec §19](/home/user/laee-engine/docs/spec/v1-original.md)
- **§20 time-varying elasticity.**
  - Lists 30/60/90/180/365-day rolling estimates (lines 826–834), but the combination formula uses only four windows and drops the 60-day one: $\epsilon_t=w_1\epsilon_{30}+w_2\epsilon_{90}+w_3\epsilon_{180}+w_4\epsilon_{365}$ (lines 838–847).
  - The weights are "learned using validation" (line 849).
  - — [v1 spec §20](/home/user/laee-engine/docs/spec/v1-original.md)
- **§21 regimes.**
  - Ten regimes: NORMAL, HIGH_DEMAND, LOW_DEMAND, PROMOTION, COMPETITIVE_SHOCK, ECONOMIC_SHOCK, LOCAL_EVENT, SUPPLY_SHOCK, SEASONAL_PEAK, SEASONAL_LOW (lines 857–868).
  - "Estimate $P(Regime=r\mid X_t)$" and then $E[\epsilon_t]=\sum_r P(r\mid X_t)\epsilon_r$ (lines 872–884). No estimation procedure is given.
  - — [v1 spec §21](/home/user/laee-engine/docs/spec/v1-original.md)
- **§39 intervals.** "Use: bootstrap, conformal prediction, Bayesian posterior, causal inference standard errors, depending on the model" (lines 1421–1428). — [v1 spec §39](/home/user/laee-engine/docs/spec/v1-original.md)
- **§40 ensemble.**
  - Model A two-way FE; B DML; C causal forest; D gradient-boosting demand model; E Bayesian hierarchical.
  - $\hat\epsilon=\sum_k w_k\hat\epsilon_k$ with $\sum_k w_k=1$, and "Learn weights from out-of-sample historical performance" (lines 1436–1467).
  - — [v1 spec §40](/home/user/laee-engine/docs/spec/v1-original.md)
- **§43–45 scores.**
  - DQS $=f(sampleSize, priceVariation, dataRecency, missingness, stockouts, promotionContamination, identificationStrength)$, with labels 0.91 excellent, 0.72 good, 0.51 weak, 0.27 insufficient (lines 1553–1573).
  - IDS $=f(priceVariation, exogeneity, instrumentStrength, controlQuality, temporalVariation, competitiveVariation)$ (lines 1581–1591).
  - Confidence $=f(DQS, IDS, modelAgreement, sampleSize, intervalWidth, outOfSamplePerformance)$, with the example "Confidence: 87%" (lines 1601–1625).
  - — [v1 spec §43–45](/home/user/laee-engine/docs/spec/v1-original.md)
- **§48–49 metrics and backtest.** §48 lists elasticity metrics "MAE, RMSE, coverage of confidence intervals, sign accuracy" (lines 1696–1703). §49 proposes counterfactual validation on historical price changes, Error $=$ Predicted $\Delta Q$ − Actual $\Delta Q$ (lines 1729–1749). — [v1 spec §48–49](/home/user/laee-engine/docs/spec/v1-original.md)
- **§50–51 drift.**
  - §50 says to monitor elasticity, competitor, customer, geographic, economic and news drift "using PSI and distribution-distance metrics", and to retrain when "prediction error > threshold OR feature drift > threshold OR elasticity regime changes" (lines 1755–1782).
  - §51 says to recalculate market definition, competition, catchment, demographics and event exposure "periodically" (lines 1798–1808).
  - — [v1 spec §50–51](/home/user/laee-engine/docs/spec/v1-original.md)
- **§63–64 and §76.** §63–64 require censoring or latent-demand modeling for stockouts (lines 2143–2181). §76 defines identification Tiers 1–7 (randomized experiments through pure predictive correlations), and "the system must report which tier supports the elasticity estimate" (lines 2460–2485). — [v1 spec §63–64, §76](/home/user/laee-engine/docs/spec/v1-original.md)

### Inferences

**Defect register (problem type: W = wrong, V = vague, U = untestable).**

| # | v1 item | Type | Problem | v2 fix (see) |
|---|---|---|---|---|
| 1 | §21 $E[\epsilon_t]=\sum_r P(r\mid X_t)\epsilon_r$ | **W** | This is the slope of $E[\log Q]$, not of $E[Q]$. Profit and revenue depend on $E[Q]$. If $Q(p)=\sum_r\pi_rQ_r(p)$, then $\partial\ln E[Q]/\partial\ln p=\sum_r w_r\epsilon_r$ with demand-share weights $w_r=\pi_rQ_r(p)/\sum_s\pi_sQ_s(p)$. **[own computation]** With $\pi=(0.7,0.3)$, $Q_r(1)=(100,300)$ and $\epsilon=(-2,-0.8)$: v1 gives −1.640, the correct value is −1.325 (a numerical derivative of $\ln E[Q]$ also gives −1.325). | Mix demand curves in the simulator, never elasticities (§2 card T3). |
| 2 | §21 regimes | V/W | No model, likelihood or labels for $P(r\mid X_t)$. Most of the ten regimes are *observed* event categories (promotion, local event, season), not latent states. Ten regime-specific $\epsilon_r$ each need price variation *inside* the regime, which an SMB with few price changes cannot supply. | Observed categories become covariates in $\tau(Z)$. At most a 2–3-state latent level regime (§2 T3). |
| 3 | §21 | W (leakage) | No distinction between filtered $P(S_t\mid\mathcal F_t)$ and smoothed $P(S_t\mid\mathcal F_T)$ probabilities. Smoothed probabilities use future data, which violates §74–75 if used for decisions. | Use filtered or one-step-predicted probabilities only (§2 T3). |
| 4 | §20 windows | W/V | Five windows are listed but four combined. A 30-day window often contains no price change, so $\sum\tilde T^2\approx0$ and the estimate is undefined. "Validation" has no target, because $\epsilon_t$ is never observed. There are no weight constraints, and overlapping windows are highly correlated. | §2 T1 (state space) and T2 (admissible windows with an explicit backtest loss). |
| 5 | §19 forest | V | No orthogonalization (the forest must be fed $\tilde Y,\tilde T$, as GRF and EconML do). The listed $Z$ variables (income, density, business size) are constant within one business, so heterogeneity along them cannot be learned from that business's data; it needs multi-tenant pooling (§80) or the prior (§42). No leaf, honesty, cluster or variance specification. | §1 cards. |
| 6 | §14/§19 with $\gamma_t$ | W (identification) | When the whole store reprices on one day (for example a menu update), a day fixed effect absorbs it, so that event contributes **zero** identifying variation. The elasticity is then identified only from *relative* price changes across products, which mixes own-price and cross-price effects (§31). Without day fixed effects, store-wide changes are confounded with common demand shocks. | IDS information shares (§5); control products in event studies (§3 E2). |
| 7 | §14 $\ln Q$ | W | Undefined on zero-sales days. Dropping zeros selects on the outcome, and $\log(Q+1)$ biases the elasticity. | Weekly aggregation for slow movers, or a Poisson pseudo-likelihood partially linear model. |
| 8 | §18/§45 "Confidence 81/87%" | **U** | The probability refers to no defined event. | Calibrated event probabilities (§5). |
| 9 | §39 conformal | **W** | Conformal prediction needs observed targets, and the elasticity is never observed, so conformal cannot produce an elasticity interval. It applies only to observables (daily demand, $\Delta\log Q$ after a price change). There is also no rule for combining member intervals and no coverage acceptance criterion, even though §48 lists coverage as a metric. | §4 taxonomy. |
| 10 | §40 Model D | W | A gradient-boosting demand model has no causal elasticity. Its price partial dependence is confounded, and trees are piecewise constant in price, so the derivative is zero almost everywhere. | Use it as a nuisance model and demand simulator only. If its implied elasticity is reported, use finite differences and tag it Tier 7. |
| 11 | §40 Model E | W | The hierarchical Bayes model is a prior and shrinkage *layer*. Averaging it with unshrunk members double-counts the data. | Use it as the combiner (§3 E5). |
| 12 | §40 $\sum w_k\hat\epsilon_k$ | V/W | "Out-of-sample performance" is undefined for an unobserved target. The members share data, so treating them as independent understates variance. **[own computation]** With member sd 0.30/0.35/0.40 and pairwise correlation 0.9, naive inverse-variance weighting claims sd 0.198, but the true sd of that combination is 0.329. | §3 E1–E5. |
| 13 | §43–45 | **U** | $f(\cdot)$ is unspecified. identificationStrength in DQS duplicates IDS. sampleSize enters Confidence directly, through DQS and through intervalWidth (triple counting). "Exogeneity" cannot be estimated from data, and "instrumentStrength" is undefined without an instrument (the usual SMB case). The label cut-offs are arbitrary. | §5 (diagnostic vector, tier + robustness value, calibrated event probabilities). |
| 14 | §50 | V | No bins, no sample-size-dependent thresholds and no multiplicity control across many products. "Elasticity drift" is not directly observable, and §50 does not say what gets retrained. **[own computation]** With 10 bins and no drift, the PSI > 0.25 rule fires 77% of the time at n=m=50 and 23% at n=m=100. | §6. |
| 15 | §51 | V | "Periodically" gives no cadence or trigger. There is no requirement to version recalculated market features, so back-filling history leaks information (§75). | §6.7. |

**Minimum content every v2 elasticity output should declare:**
1. the estimand: which $\tau$, at which $Z$, as of which date;
2. the estimator and its version;
3. the identification tier plus information shares by source of price variation (§5);
4. the interval type (confidence for the parameter, or prediction for an outcome), the method, and its last coverage test;
5. "confidence" stated as the calibrated probability of a named event.

### Gaps
- I read only §14, §16–21, §38–51, §62–65, §69, §76 and §88 of v1. §41–42 (hierarchical prior), §77–78 (experimentation and active learning) and §80 (multi-tenant pooling) may partly address issues 5, 6 and 11; that is another researcher's scope and was not checked in depth.

---

## 1. Heterogeneous elasticity with a continuous treatment (price)

### Takeaway
For a continuous price, GRF's causal forest, EconML's CausalForestDML, the R-learner and LinearDML all solve the same residual-on-residual (partially linear, "Robinson") moment, either locally or globally. In GRF's source code the forest's local elasticity is literally a forest-weighted OLS slope of $\tilde Y$ on $\tilde T$.

At SMB data sizes the default should be **pooled-across-products linear-interaction DML** with at most 5 pre-registered interactions, cluster-robust or wild-cluster-bootstrap inference, and **empirical-Bayes shrinkage** to product level.

Causal forests are a gated add-on. Use them only with thousands of pooled rows and dozens of independent price changes, and only after they beat both a constant and a linear model on held-out R-loss and pass a calibration test. The DR-learner and orthogonal random forests are poor fits for a continuous, rarely changing price.

### Cited Findings

#### Shared basis: orthogonal partially linear moment and R-loss
- EconML describes CausalForestDML as fitting "a forest that solves the local moment equation problem: E[(Y − E[Y|X,W] − <θ(x), T − E[T|X,W]> − β(x)) (T;1) | X=x] = 0 where E[Y|X,W] and E[T|X,W] are fitted in a first stage in a cross-fitting manner" (source code, econml/dml/causal_forest.py). — [EconML 0.17.0](https://pypi.org/project/econml/0.17.0/)
- R-learner: Nie & Wager, "Quasi-oracle estimation of heterogeneous treatment effects", *Biometrika* 108(2):299–319 (2021).
  - It is a two-step method in which either step can use any loss-minimization method (penalized regression, boosting, neural networks).
  - For penalized kernel regression it has a quasi-oracle property: it attains oracle error bounds even when the nuisance estimates are not very accurate.
  - — [Stanford GSB](https://www.gsb.stanford.edu/faculty-research/publications/quasi-oracle-estimation-heterogeneous-treatment-effects); [NSF PAR copy](https://par.nsf.gov/servlets/purl/10311702)
- EconML's R-loss scorer (RScorer, source code; default cv=2):
  - Residuals: Yres = Y − E[Y|X,W] and Tres = T − E[T|X,W], cross-fitted.
  - loss(cate) = E_n[(Yres − ⟨cate(X), Tres⟩)²]; base_loss = min_θ E_n[(Yres − θ·Tres)²]; score = 1 − loss/base_loss.
  - "A negative score, means that the cate model performs even worse than a constant effect model and hints at overfitting."
  - It cites Nie & Wager and Schuler et al. (2018) for causal model selection.
  - — [EconML 0.17.0, econml/score/rscorer.py](https://pypi.org/project/econml/0.17.0/)
- DML background: cross-fitted Neyman-orthogonal scores, root-n inference, and median aggregation over repeated sample splits. **[lit]** Chernozhukov et al. (2018), *Econometrics Journal* 21(1):C1–C68. — [DOI](https://doi.org/10.1111/ectj.12097)

#### GRF and causal forests
- GRF fits any quantity defined by local moment equations. It solves a forest-weighted local estimating equation, and splits are chosen from a gradient-based linear approximation: a "labeling step" computes pseudo-outcomes, followed by an ordinary CART regression step. — [Athey, Tibshirani & Wager 2019, *Annals of Statistics* 47(2), arXiv:1610.01271](https://arxiv.org/pdf/1610.01271) (search summary; the [Project Euclid version](https://projecteuclid.org/journals/annals-of-statistics/volume-47/issue-2/Generalized-random-forests/10.1214/18-AOS1709.full) was blocked)
- **Exact pseudo-outcome in GRF's C++ core** (source code, `InstrumentalRelabelingStrategy::relabel`). The causal forest is run as an instrumental forest with the instrument equal to the treatment (skgrf passes `z_hat=w_hat`). In a parent node $P$ with weighted means $\bar W_P$ and $\bar Y_P$:
  $$\tau_P=\frac{\sum_{i\in P}(W_i-\bar W_P)(Y_i-\bar Y_P)}{\sum_{i\in P}(W_i-\bar W_P)^2},\qquad \rho_i=(W_i-\bar W_P)\big[(Y_i-\bar Y_P)-\tau_P(W_i-\bar W_P)\big].$$
  CART on $\rho_i$ then chooses the split. If the denominator is ≈ 0 (no treatment variation in the node), relabeling returns early and no split is made. Here $Y,W$ are the already-centered $\tilde Y,\tilde T$. — [skgrf 0.3.0 (vendors grf core)](https://pypi.org/project/skgrf/0.3.0/)
- **Forest prediction** (source code, `InstrumentalPredictionStrategy::predict`): $\hat\tau(x)=\dfrac{\overline{WY}\cdot\bar 1-\bar Y\,\bar W}{\overline{WW}\cdot\bar 1-\bar W\,\bar W}$. The averages are forest-weighted ($\alpha_i(x)$), so this is a weighted local OLS slope of $\tilde Y$ on $\tilde W$ with intercept. — [skgrf 0.3.0](https://pypi.org/project/skgrf/0.3.0/)
- **Variance by the bootstrap of little bags** (source code, `compute_variance`):
  - Trees are grown in groups of `ci_group_size` on the same half-sample.
  - For each tree, compute an influence-type term $\rho$ from leaf moments $\psi_1,\psi_2$.
  - var_between = mean over groups of (group-mean $\rho$)²; var_total = mean of $\rho^2$; group_noise = (var_total − var_between)/(ci_group_size − 1).
  - Debiased variance = var_between − group_noise, computed through an "objective Bayes" debiaser so it cannot go negative.
  - — [skgrf 0.3.0](https://pypi.org/project/skgrf/0.3.0/)
- **GRF causal-forest defaults** (source code, skgrf `GRFForestCausalRegressor` signature and docstrings):
  - n_estimators (num.trees) = 2000, sample_fraction = 0.5, mtry default "sqrt(p) + 20", min_node_size = 5.
  - honesty = True, honesty_fraction = 0.5, honesty_prune_leaves = True.
  - alpha ("the maximum imbalance of a split") = 0.05, imbalance_penalty = 0, stabilize_splits = True, equalize_cluster_weights = False.
  - ci_group_size = 2 ("At least 2 is required to provide confidence intervals"); "If ci_group_size > 1, the max allowed [sample] fraction is 0.5."
  - — [skgrf 0.3.0](https://pypi.org/project/skgrf/0.3.0/). The current R package is grf 2.6.1 (May 2026), whose defaults were not checked. — [CRAN grf manual](https://cran.r-project.org/web/packages/grf/grf.pdf) (listed in search results; fetch blocked)
- GRF's tunable parameters are sample.fraction, mtry, min.node.size, honesty.fraction, honesty.prune.leaves, alpha and imbalance.penalty. For small or marginally powered data, honesty.prune.leaves = FALSE may help but needs more trees, and tuning does not adjust the number of trees. GRF also provides `multi_arm_causal_forest` for multiple discrete treatments. — [grf causal_forest on rdrr.io](https://rdrr.io/cran/grf/man/causal_forest.html); [multi_arm_causal_forest](https://rdrr.io/cran/grf/man/multi_arm_causal_forest.html) (search summaries)
- Honesty trade-off: honesty halves the data available both for choosing splits and for estimating leaves. In small samples, honest forests had slightly higher CATE RMSE than adaptive forests but substantially better coverage of nominal 95% intervals. — [Hou 2025, "Honesty in Causal Forests: When It Helps and When It Hurts", arXiv:2506.13107](https://arxiv.org/pdf/2506.13107); [metricgate GRF honesty explainer](https://metricgate.com/blogs/grf-honest-splitting-causal-trees/) (one search summary drew on both; I could not tell which of the two makes the RMSE and coverage claim)
- Wager & Athey (2018), *JASA* 113(523):1228–1242: honesty (each unit's outcome is used either for splitting or for leaf estimation, never both) plus subsampling give asymptotically unbiased, Gaussian forest predictions. — [search summary listing](https://cameron.econ.ucdavis.edu/sfu2022/ML_2022_part5_More_Causal.pdf); **[lit]** [DOI](https://doi.org/10.1080/01621459.2017.1319839). For an application with clustered data see [Athey & Wager 2019, "Estimating treatment effects with causal forests"](https://par.nsf.gov/servlets/purl/10311714) (title only).
- For continuous treatments, GRF's doubly robust average effects use weights that require $\mathrm{Var}[W\mid X=x]$, estimated with a separate forest. — [search summary on GRF continuous treatment](https://arxiv.org/pdf/1610.01271) (snippet; not verified against the grf docs)
- A 2024 study compares random-forest CATE methods for continuous treatments. — [PMC11577706](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11577706/) (title only; fetch blocked)

#### EconML estimators (source code, econml 0.17.0)
- **CausalForestDML defaults:**
  - model_y = 'auto', model_t = 'auto' ("the best-fitting of a set of linear and forest models"), cv = 2, mc_iters = None.
  - n_estimators = 100, criterion = "mse" (alternative "het"), max_depth = None, min_samples_split = 10, min_samples_leaf = 5.
  - min_weight_fraction_leaf = 0, min_var_fraction_leaf = None, max_features = "auto" (equal to n_features), max_samples = 0.45, min_balancedness_tol = 0.45.
  - honest = True, inference = True, fit_intercept = True, subforest_size = 4 ("`n_estimators` must be divisible by `subforest_size`").
  - Inference uses "a bootstrap-of-little-bags approach … with an objective Bayesian debiasing correction".
  - min_var_fraction_leaf "avoids performing splits where … the variance of the treatment is small and hence the local parameter is not well identified".
  - Doubly robust ATE "is not available for continuous treatments".
  - `tune()` trains 100-tree forests on the grid {min_weight_fraction_leaf: [0.0001, .01], max_depth: [3, 5, None], min_var_fraction_leaf: [0.001, .01]} and picks by out-of-sample R-score on a ~70/30 split.
  - With `groups`, "All rows corresponding to the same group will be kept together during splitting."
  - — [EconML 0.17.0](https://pypi.org/project/econml/0.17.0/)
- **LinearDML defaults:** model_y = 'auto', model_t = 'auto', featurizer = None, fit_cate_intercept = True, cv = 2, mc_iters = None. — [EconML 0.17.0](https://pypi.org/project/econml/0.17.0/)
- **DRLearner defaults:** model_propensity = 'auto', model_regression = 'auto', model_final = StatsModelsLinearRegression(), min_propensity = 1e-6, cv = 2. It requires a propensity model, so it is built for discrete treatments. — [EconML 0.17.0](https://pypi.org/project/econml/0.17.0/)
- **DMLOrthoForest defaults:** n_trees = 500, min_leaf_size = 10, max_depth = 10, subsample_ratio = 0.7, bootstrap = False, lambda_reg = 0.01, model_Y = WeightedLassoCVWrapper(cv=3), global_residualization = False, global_res_cv = 2. — [EconML 0.17.0](https://pypi.org/project/econml/0.17.0/)

#### DR-learner and orthogonal random forests (papers)
- Kennedy (2023), "Towards optimal doubly robust estimation of heterogeneous causal effects", *Electronic Journal of Statistics* 17(2):3008–3049. It is a two-stage doubly robust CATE estimator with sample splitting, comes with a generic error bound, and states sufficient conditions for oracle efficiency. — [Project Euclid](https://projecteuclid.org/journals/electronic-journal-of-statistics/volume-17/issue-2/Towards-optimal-doubly-robust-estimation-of-heterogeneous-causal-effects/10.1214/23-EJS2157.full); [Oxford APTS meta-learner notes](https://www.stats.ox.ac.uk/~evans/APTS/meta-learners.html)
- Oprescu, Syrgkanis & Wu (2019), "Orthogonal Random Forest for Causal Inference", ICML, PMLR 97:4932–4941. It handles discrete or continuous treatments. When the nuisances have a locally sparse form, local ℓ1-penalized regression reaches the needed rates, giving oracle-rate error. — [PMLR](https://proceedings.mlr.press/v97/oprescu19a.html)

#### Validating heterogeneity
- Chernozhukov, Demirer, Duflo & Fernández-Val, Fisher–Schultz Lecture, *Econometrica* 93(4):1121–1164 (2025). It proposes BLP (best linear predictor of effects on an ML proxy), GATES (sorted group average effects) and CLAN, in randomized experiments. — [Wiley](https://onlinelibrary.wiley.com/doi/full/10.3982/ECTA19303); [Econometric Society](https://www.econometricsociety.org/publications/econometrica/2025/07/01/FisherSchultz-Lecture-Generic-Machine-Learning-Inference-on-Heterogeneous-Treatment-Effects-in-Randomized-Experiments-With-an-Application-to-Immunization-in-India)
- EconML's DRTester implements three checks on doubly robust outcomes (discrete treatments; source code):
  - BLP: "If the CATE model captures true heterogeneity, then the OLS estimate on the CATE predictions should be positive and significantly different than 0."
  - Calibration: $\mathrm{Cal}_G=\sum_k\pi(k)\lvert E[s(Z)\mid k]-E[Y^{DR}\mid k]\rvert$, $\mathrm{Cal}_O=\sum_k\pi(k)\lvert E[s(Z)\mid k]-E[Y^{DR}]\rvert$, $R^2_C=1-\mathrm{Cal}_G/\mathrm{Cal}_O$.
  - QINI/AUTOC.
  - — [EconML 0.17.0, econml/validate/drtester.py](https://pypi.org/project/econml/0.17.0/)

### Inferences

#### 1.1 What is being estimated when price is continuous
- **Global partially linear estimate.** $\hat\theta=\sum\tilde T\tilde Y/\sum\tilde T^2$ converges to $\theta^*=E[\mathrm{Cov}(Y,T\mid X)]/E[\mathrm{Var}(T\mid X)]$. Under heterogeneity this equals $E[w(X)\tau(X)]$ with $w=\mathrm{Var}(T\mid X)/E[\mathrm{Var}(T\mid X)]$: a **price-variation-weighted** average elasticity. A pooled "business elasticity" therefore over-weights products and periods with more price variation. To get a product-average or revenue-weighted elasticity, average the conditional estimates explicitly.
- **Forest estimate.** $\hat\tau(x)$ is the same object localized by forest weights (source code above). If the demand curve is nonlinear, it is a local *arc* elasticity over the observed log-price range. Do not extrapolate beyond the observed range ± about 10% without the prior **[heuristic]**.
- **Exact information decomposition.** For any partition of rows into sources $k$ (randomized test, regular price change, promotion, store-wide change), $\hat\theta=\sum_k I_k\hat\theta_k$ with $I_k=\sum_{i\in k}\tilde T_i^2/\sum_i\tilde T_i^2$ and $\hat\theta_k=\sum_{i\in k}\tilde T_i\tilde Y_i/\sum_{i\in k}\tilde T_i^2$. This is pure algebra and it feeds the identification score (§5).

#### 1.2 Feasibility gates and expected precision
- **Precision formula.** $se(\hat\epsilon)\approx\sigma_u/(\sqrt{n_{eff}}\cdot sd(\tilde T))$. **[own computation]** A Monte Carlo check with n=200, two price levels (sd 0.05) and $\sigma_u=0.6$ gave an empirical sd of 0.845 against 0.849 from the formula.
- **Implication.** A typical SMB product with 200 usable days, 5% residual price variation and noisy log demand ($\sigma_u=0.6$) has $se\approx0.85$, so the 95% interval is about ±1.7: essentially uninformative without pooling or the prior.

| $\sigma_u$ | n | sd($\tilde T$)=0.02 | 0.05 | 0.10 |
|---|---|---|---|---|
| 0.3 | 50 | 2.12 | 0.85 | 0.42 |
| 0.3 | 200 | 1.06 | 0.42 | 0.21 |
| 0.3 | 1000 | 0.47 | 0.19 | 0.09 |
| 0.6 | 50 | 4.24 | 1.70 | 0.85 |
| 0.6 | 200 | 2.12 | 0.85 | 0.42 |
| 0.6 | 1000 | 0.95 | 0.38 | 0.19 |

**Method gates [heuristic; tune by the semi-synthetic simulation in §3 E3]:**

| Method | Minimum data to run | Otherwise |
|---|---|---|
| Prior only (§41–42) | — | Used when $K_{pc}=0$ or sd($\tilde T$) < 0.01 |
| Pooled partially linear DML (all products of the business) | n ≥ 100 valid rows, $K_{pc}$ ≥ 3, residual share RS = Var($\tilde T$)/Var($T$) ≥ 0.1 | Prior only |
| Per-product slope plus EB shrinkage (1.7) | $K_{pc,j}$ ≥ 2 for product $j$ | Product gets the pooled/prior estimate |
| LinearDML with p interactions | ≥ 10 independent price changes per interaction parameter, and ≥ 3 per level of each binary feature | Drop that interaction |
| CausalForestDML / grf | pooled n ≥ 2,000 rows, ≥ 50 independent price changes across ≥ 20 products, plus the validation gate in 1.5 | LinearDML |
| Per-product forest | Never at SMB scale | — |

#### 1.3 Method card: linear-interaction DML (recommended default)
- **Model.** $Y=(\theta_0+\theta^\top\phi(Z))\,T+g(X)+u$. Center $\phi$ so that $\theta_0$ is the elasticity at average conditions.

**Algorithm**
1. **Folds.** Use K=5 folds from GroupKFold on blocks (product × 28-day block). EconML's default cv=2 (source code) leaves only half the data for each nuisance fit, which hurts small samples. Grouped blocks also stop serially correlated neighbors from leaking across folds.
2. **Nuisance models.**
   - Default: ridge regression with product dummies, day-of-week, holiday, weather, competitor price index and demand lags of at least 1 day (never same-day).
   - Use LightGBM (num_leaves=15, min_data_in_leaf=20, learning_rate=0.05, early stopping) only when n ≥ 2,000.
   - Do not let $\hat e$ use features that mechanically set price. Handle promotional price as a separate treatment column, or exclude promotion days (v1 §62).
3. **Residuals.** Compute $\tilde Y$ and $\tilde T$, then check RS ≥ 0.1.
4. **Final stage.** $\hat\vartheta=\big(\sum_i\tilde T_i^2\bar\phi_i\bar\phi_i^\top\big)^{-1}\sum_i\tilde T_i\tilde Y_i\bar\phi_i$ with $\bar\phi_i=(1,\phi(Z_i))$.
5. **Inference.**
   - Cluster-robust: $\hat V=c\,J^{-1}\big(\sum_g s_gs_g^\top\big)J^{-1}$, with $J=\sum_i\tilde T_i^2\bar\phi_i\bar\phi_i^\top$ and $s_g=\sum_{i\in g}\tilde T_i(\tilde Y_i-\tilde T_i\bar\phi_i^\top\hat\vartheta)\bar\phi_i$.
   - Use the CR1 factor $c=\frac{G}{G-1}\frac{n-1}{n-p-1}$ and $t_{G-1}$ critical values (80%: 1.2816 asymptotically; 95%: 1.96).
   - If G < ~30 clusters, use the wild cluster bootstrap (§4 U2).
   - Repeat cross-fitting S=5 times and take the median **[lit, Chernozhukov et al. 2018]**: $\hat\theta=\mathrm{med}_s\hat\theta_s$, $\hat V=\mathrm{med}_s[\hat V_s+(\hat\theta_s-\hat\theta)^2]$.
6. **Ridge on interactions.** When p > 3, apply ridge to $\theta$ (not $\theta_0$), with $\lambda$ chosen by held-out R-loss.

**Defaults.** At most 5 pre-registered features: price-tier z-score, weekend, peak season, competitor price index z-score (time-varying) and category group.

**Failure modes**
- $\hat e$ absorbs the price variation (for example, product dummies plus a flexible trend reproduce a single step change), so $\tilde T\approx0$ and the SE explodes. Watch RS.
- Interaction collinearity when all price changes occur at one level of $\phi$.
- Promotions dominate identification. Check the promotion information share $I_{promo}$.
- Store-wide repricing (defect #6).
- Zeros in Q (defect #7).

**Tests that catch a wrong implementation**
1. **Algebra.** Given the residual arrays, $\hat\vartheta$ equals numpy `lstsq` on $[\tilde T,\tilde T\phi]$. It must also equal EconML `LinearDML` when both receive identical explicit splits (pass `cv` as an iterable of (train, test) index pairs) and identical nuisance models.
2. **Simulation (R=500 replications).**
   - Design: confounder-driven prices, 3–6 price steps per product, $\tau=-1.5+0.5Z$, AR(1) noise.
   - Require |bias| < 0.05 for each coefficient, and 95% interval coverage within [0.93, 0.97] (the binomial band for R=500).
3. **Null.** With a constant $\tau$, the 5%-level interaction test rejects in 3–7% of replications.
4. **Invariances.** Multiplying all prices by 1.1 leaves $\hat\vartheta$ unchanged. Duplicating every row leaves $\hat\vartheta$ unchanged and must **not** shrink the clustered SE (this catches non-clustered SE bugs).
5. **Walk-forward leakage.** Changing data after date t must not change residuals or estimates for decisions dated ≤ t.

#### 1.4 Method card: R-learner with a flexible final stage
- **Loss.** $\hat L(\tau)=\frac1n\sum_i(\tilde Y_i-\tau(Z_i)\tilde T_i)^2+\Lambda(\tau)$. For an arbitrary ML regressor, use the equivalent weighted problem: target $\tilde Y_i/\tilde T_i$, weight $\tilde T_i^2$.
- **Implementation.**
  - Drop rows with $|\tilde T_i|<10^{-3}$: their weight is ≈0 but the pseudo-outcome blows up numerically.
  - Regularize heavily: LightGBM with min_data_in_leaf ≥ 50 and num_leaves ≤ 7, or kernel ridge.
- **SMB role.** Use it mainly as the **model-selection loss** (RScorer) rather than as the reported estimator.
- **Failure mode.** With few price changes, most rows have $\tilde T\approx0$, so the effective sample is the rows around price changes.
- **Tests.**
  - For the constant-$\tau$ model class, the minimizer equals the closed-form partially linear $\hat\theta$.
  - In simulation, the true $\tau$'s R-score is ≥ the constant model's.
  - Under a constant $\tau$, a flexible model's held-out score is ≈ 0 or negative.

#### 1.5 Method card: causal forest (grf or EconML CausalForestDML), gated

**Recommended SMB settings [heuristic]:**
```python
CausalForestDML(model_y=<ridge or small LGBM>, model_t=<same>,
                cv=<5 GroupKFold splits on product x 28-day blocks>,
                n_estimators=2000,          # divisible by subforest_size=4
                criterion="mse", min_samples_leaf=20, min_var_fraction_leaf=0.05,
                max_depth=None,             # let tune() choose from {3, 5, None}
                max_samples=0.45, min_balancedness_tol=0.45,
                honest=True, inference=True, random_state=0)
# call .tune(Y, T, X=Z, W=X_ctrl, groups=product_id), then
#      .fit(Y, T, X=Z, W=X_ctrl, groups=product_id)
```
In R, the equivalent is `causal_forest(X, Y, W, Y.hat, W.hat, clusters = product_id, num.trees = 4000, min.node.size = 20, tune.parameters = "all")`.

**Why these differ from the defaults**
- EconML's default 100 trees with subforest_size = 4 give only 25 little bags for the variance, so the intervals are noisy. GRF's default of 2000 trees with ci_group_size = 2 gives 1,000 groups.
- min_samples_leaf = 5 allows leaves containing only 2–3 price levels, which makes local slopes unstable.
- min_var_fraction_leaf directly encodes "a leaf must contain price variation" (source-code docstring).
- Keep honesty on: the small-sample evidence (Hou 2025) favors it for interval coverage.

**Validation gate (all three must pass before heterogeneity is shown to users)**
1. **R-score.** On held-out data, the forest's R-score beats both LinearDML and the constant model by more than 2 paired cluster-bootstrap standard errors.
2. **Calibration regression.** This is a continuous-treatment analogue of BLP and GRF's `test_calibration`; GRF's exact definition was not verified. On held-out rows, fit $\tilde Y_i=\beta_1\bar\tau\,\tilde T_i+\beta_2(\hat\tau(Z_i)-\bar\tau)\tilde T_i+e_i$ with cluster-robust errors. Require $\hat\beta_2>0$ (one-sided p < 0.05) with a CI that includes 1, meaning the heterogeneity is both real and correctly scaled.
3. **GATES-style check.** Split products into tertiles of cross-fitted $\hat\tau$, re-estimate the partially linear model within each tertile, and require monotone ordering with top and bottom 80% intervals that do not overlap.

**Failure modes**
- Leaves without price variation (the core then makes no split).
- Spurious heterogeneity along features that proxy *when* prices changed.
- Under-coverage where heterogeneity is sharp.
- Extrapolation outside the observed price range.

**Tests**
1. **Weight equivalence.** Recompute $\hat\tau(x)$ as a weighted OLS of $\tilde Y$ on $\tilde T$ using the forest weights (grf `get_forest_weights`, assumed from the grf docs and not verified this session) and compare with `predict()` to within 1e-8.
2. **Constant-$\tau$ simulation.** About 5% of 95% intervals exclude the truth, and the calibration $\beta_2$ is significant in about 5% of replications.
3. **Step heterogeneity.** With $\tau=-2$ if $Z>0$ else $-1$, 3,000 pooled rows and 60 price changes, the GATES ordering is correct in ≥ 80% of replications.
4. **Degenerate subgroup.** A subgroup with constant price must produce finite predictions, not NaN or inf.

#### 1.6 Method card: DR-learner (discretized price) and orthogonal random forest
- **DR-learner construction.**
  - Define $A_{it}=1$ if price ≥ the product's reference price + 3% within ±28 days of a change **[heuristic]**.
  - Pseudo-outcome **[lit, Kennedy 2023]**: $\varphi=\hat\mu_1(X)-\hat\mu_0(X)+\frac{A-\hat\pi(X)}{\hat\pi(X)(1-\hat\pi(X))}(Y-\hat\mu_A(X))$. Regress $\varphi$ on $Z$.
  - Convert to an arc elasticity by dividing by $E[T\mid A=1,Z]-E[T\mid A=0,Z]$.
  - Clip $\hat\pi$ to [0.05, 0.95]. EconML's default min_propensity = 1e-6 is far too permissive.
- **Why it struggles at SMB scale.** Long runs at a single price push $\hat\pi$ toward 0 or 1, which inflates variance.
- **Its role here.** A robustness check, and access to DRTester's BLP, calibration and QINI.
- **Truly continuous doubly robust alternatives.** Generalized-propensity dose-response methods **[lit: Kennedy, Ma, McHugh & Small 2017, JRSS-B, [DOI](https://doi.org/10.1111/rssb.12212)]** and Riesz-representer average derivatives both need the conditional density of log-price given X. That density cannot be estimated with 2–5 price levels per product.
- **Orthogonal random forest.** It refits nuisance models locally for every target point. That is computationally heavy and high-variance below roughly 5,000 rows **[inference]**. Prefer CausalForestDML, which residualizes globally.

#### 1.7 Product-level elasticities by empirical-Bayes shrinkage (bridge to §41)
- **Per-product slopes.** For each product, compute $\hat\theta_j$ and $se_j$ with the 1.3 machinery (product-specific slopes on shared nuisances).
- **Random-effects model.** $\hat\theta_j\sim N(\epsilon_j,se_j^2)$ and $\epsilon_j\sim N(\mu+\gamma^\top w_j,\tau^2)$, where $w_j$ are product attributes. Estimate $(\mu,\gamma,\tau^2)$ by REML or DerSimonian–Laird **[lit, [DOI](https://doi.org/10.1016/0197-2456(86)90046-2)]**.
- **Posterior.** $\epsilon_j\mid\cdot\sim N\big(B_j(\mu+\gamma^\top w_j)+(1-B_j)\hat\theta_j,\;(1-B_j)se_j^2\big)$ with $B_j=se_j^2/(se_j^2+\tau^2)$, plus a term for the uncertainty in $\hat\mu,\hat\gamma,\hat\tau^2$.
- **Products with no price change** get $N(\mu+\gamma^\top w_j,\tau^2+\mathrm{Var}(\hat\mu+\hat\gamma^\top w_j))$.
- **Prior share.** $B_j$ is reported as the "prior share" diagnostic (§5).
- **Test.** Simulate 100 products with $\epsilon_j\sim N(-1.5,0.4^2)$ and heterogeneous $se_j$. The EB posterior's MSE must beat both the raw $\hat\theta_j$ and the pooled $\hat\mu$, and 80% posterior intervals must cover ≈ 80% of the $\epsilon_j$.

### Gaps
- I could not read the current grf 2.6.1 documentation (blocked). skgrf 0.3.0 vendors an older grf core, so current R defaults (num.trees, mtry formula, and whether `lm_forest` or `test_calibration` changed) are unverified.
- I found no benchmark of causal forests at n ≈ 500–5,000 with a continuous treatment taking only a few distinct values. The PMC comparison (PMC11577706) could not be read.
- GRF's exact `test_calibration` and `get_forest_weights` behavior was not verified, because the search budget ran out.
- The ≥10-events-per-parameter, ≥50-price-change and RS ≥ 0.1 gates are heuristics.

---

## 2. Time-varying elasticity

### Takeaway
Treat time variation as a **shrinkage problem**, not a window-picking problem.

- **Default.** A state-space model in which the elasticity is a random walk, run on the residualized pair $(\tilde Y_t,\tilde T_t)$ with a Kalman filter. Its drift variance $q$ is estimated by maximum likelihood, so it collapses to the static estimate when the data show no drift. It also learns nothing on days without price variation, which is exactly the right behavior.
- **Rolling windows** are diagnostics. If combined, use only windows that contain price variation, with equal (Pesaran–Pick) or backtest-learned simplex weights.
- **Latent regimes and change points.** Markov switching and change-point detection are identifiable for *demand-level* shifts. Elasticity shifts are identifiable only with several price changes on each side.
- **Observable "regimes"** (promotion, event, season, competitor entry) belong in $\tau(Z)$ as covariates.
- **Regime mixing** must mix demand curves, not elasticities.

### Cited Findings
- **Forecast combination across windows (AveW).** Pesaran & Pick, "Forecast Combination Across Estimation Windows", *JBES* 29(2):307–318 (2011). Averaging forecasts over estimation windows improves accuracy without estimating break dates or sizes when breaks are uncertain. — search summary; related work by the same authors: [Pesaran, Pick & Pranovich, "Optimal forecasts in the presence of structural breaks"](https://ideas.repec.org/p/cam/camdae/1163.html) and [PDF](https://files.econ.cam.ac.uk/people-files/mhp1/fp13/PPP-Revision-1.pdf)
- **Dynamic linear models in marketing.** A multivariate dynamic linear model estimated with the Kalman filter lets response parameters, including regular price elasticity, vary over time as a function of marketing strategy. It allows "greater statistical efficiency with parameter evolution and explanation in one step". — [Ataman, Mela & van Heerde (MSI/working-paper PDF; published in JMR 2010 per prior knowledge, [lit])](https://people.duke.edu/~mela/bio/papers/Ataman_Mela_Van_Heerde_2006.pdf); [ResearchGate](https://www.researchgate.net/publication/232906146_The_Long-Term_Effect_of_Marketing_Strategy_on_Brand_Sales)
- **Bayesian time-varying-coefficient marketing-mix model.** — [arXiv:2106.03322](https://arxiv.org/pdf/2106.03322) (title only)
- **Coefficient-stability tests in statsmodels** (source code, `RecursiveLS`):
  - CUSUM: $W_t=\frac1{\hat\sigma}\sum_{j=k+1}^t w_j$ over recursive residuals $w_j$.
  - CUSUM of squares: $s_t=\sum_{j\le t}w_j^2/\sum_{j\le T}w_j^2$.
  - `plot_cusum(alpha=0.05)` draws significance bounds; "Evidence of parameter instability may be found if the CUSUM statistic moves out of the significance bounds."
  - Reference: Brown, Durbin & Evans (1975), *JRSS-B* 37(2):149–192.
  - — [statsmodels 0.15.0](https://pypi.org/project/statsmodels/0.15.0/)
- **Markov switching.** Hamilton (1989), *Econometrica* 57:357–384. The forward filter gives filtered and smoothed regime probabilities, and Kim (1994) gives the smoothing recursion ("Kim smoother"). — [Hamilton, "Regime-Switching Models" (UCSD)](https://econweb.ucsd.edu/~jhamilto/palgrav1.pdf); [Kuan lecture notes](https://homepage.ntu.edu.tw/~ckuan/pdf/Lec-Markov_note.pdf); [Kim, Dynamic linear models with Markov switching](https://www.researchgate.net/profile/Chang-Jin-Kim/publication/223036114_Dynamic_Linear_Model_with_Markov_Switching/links/5acc76524585151e80ab98be/Dynamic-Linear-Model-with-Markov-Switching.pdf)
- **statsmodels Markov switching API** (source code):
  - `MarkovRegression(endog, k_regimes, trend="c", exog=None, order=0, exog_tvtp=None, switching_trend=True, switching_exog=True, switching_variance=False)` with model $y_t=a_{S_t}+x_t'\beta_{S_t}+\varepsilon_t$.
  - `exog_tvtp` gives time-varying transition probabilities.
  - `fit(... method="bfgs", maxiter=100, em_iter=5, search_reps=0, search_iter=5, search_scale=1.0)` estimates by maximum likelihood through the Hamilton filter.
  - Outputs include `filtered_marginal_probabilities` "Pr[S_t=s_t | Y_t]", `smoothed_marginal_probabilities` "Pr[S_t=s_t | Y_T]" and `expected_durations`.
  - The docstring warns: "This model is new and API stability is not guaranteed."
  - — [statsmodels 0.15.0](https://pypi.org/project/statsmodels/0.15.0/); [statsmodels Markov example](https://www.statsmodels.org/devel//examples/notebooks/generated/markov_autoregression.html)
- **PELT.** Killick, Fearnhead & Eckley (2012), *JASA* 107(500):1590–1598. It minimizes penalized segment costs (penalized likelihood, MDL), and its cost is linear in n under mild conditions. — [arXiv:1101.1438](https://arxiv.org/pdf/1101.1438); [Lancaster](https://research.lancaster-university.uk/en/publications/optimal-detection-of-changepoints-with-a-linear-computational-cos/); [R changepoint manual](https://cran.r-project.org/web/packages/changepoint/changepoint.pdf)
- **ruptures 1.1.10** (source code):
  - `Pelt(model="l2", custom_cost=None, min_size=2, jump=5, params=None)`; `predict(pen)` takes a user-supplied penalty > 0.
  - `CostLinear` (model "linear") is the "Least-square estimate for linear changes". The first column is the observed variable and the remaining columns are covariates; min_size = 2.
  - — [ruptures 1.1.10](https://pypi.org/project/ruptures/1.1.10/)
- **BOCPD.** Adams & MacKay (2007, arXiv:0710.3742) recursively maintain the posterior over run length (time since the last change point). A constant hazard $\lambda$ implies a geometric gap between change points with mean $1/\lambda$. The method uses conjugate exponential-family models (Normal-Gamma for unknown mean and variance). — [Semantic Scholar](https://www.semanticscholar.org/paper/Bayesian-Online-Changepoint-Detection-Adams-MacKay/9cacd41e4fbe518436877a6f1b24982099216e46); [Gundersen derivation](https://gregorygundersen.com/blog/2019/08/13/bocd/); robust variant: [OpenReview](https://openreview.net/pdf?id=Dg20tpmMVz)

### Inferences

#### T0. What the data can support
Elasticity variation is observable only at price changes. With $K_{pc}$ price changes there are at most $K_{pc}$ local slope "observations".
- Never compute a window estimate with fewer than 2 independent price changes, or with sd($\tilde T$) < 0.01.
- Declare the elasticity "time-varying" only if $\hat q>0$ is significant. Because $q=0$ is on the boundary, the likelihood-ratio test uses the critical value of the mixture $\tfrac12\chi^2_0+\tfrac12\chi^2_1$ (2.71 at 5%) **[lit]**.

#### T1. Method card: Kalman time-varying elasticity (recommended)
**Model.** Observation $\tilde Y_t=a_t+\beta_t\tilde T_t+v_t$, $v_t\sim N(0,\sigma^2)$. State $\alpha_t=(a_t,\beta_t)'=\alpha_{t-1}+w_t$, $w_t\sim N(0,\mathrm{diag}(q_a,q_\beta)\sigma^2)$. Here $a_t$ is a local level for residual demand. Set $q_a=0$ if the nuisance model already handles levels. A mean-reverting variant uses $\beta_t-\mu=\phi(\beta_{t-1}-\mu)+w_t$. With $x_t=(1,\tilde T_t)'$:
```
predict:  α_{t|t-1} = α_{t-1|t-1};   P_{t|t-1} = P_{t-1|t-1} + Q
innov:    e_t = Ỹ_t − x_t'α_{t|t-1};  F_t = x_t' P_{t|t-1} x_t + σ²
update:   K_t = P_{t|t-1} x_t / F_t;  α_{t|t} = α_{t|t-1} + K_t e_t;  P_{t|t} = P_{t|t-1} − K_t x_t' P_{t|t-1}
loglik:   −½ Σ_t [ log(2π F_t) + e_t² / F_t ]
missing day: skip the update step (prediction only), so uncertainty grows by Q
```
- **Learning only at price changes.** When $\tilde T_t\approx0$, the β-component of the gain is ≈ 0, so nothing is learned about $\beta$.
- **Initialization.** Take $\beta_0$ and its prior variance from the hierarchical prior or pooled estimate (for example $N(\mu_{prior},0.5^2)$). That is the formal link to v1 §41–42.
- **Hyperparameters.** Estimate $\log q_\beta$ and $\sigma^2$ by maximum likelihood. A scalar numpy filter (about 40 lines) plus `scipy.optimize.minimize` is simplest.
- **Small-data fallback.** With fewer than 2 price-change episodes at least 30 days apart, fix $q_\beta$ from the prior **[heuristic]**. For example, an elasticity drift sd of 0.05 per 30 days gives $q_\beta\sigma^2=0.05^2/30$ per day.
- **Discounting alternative** (West & Harrison style **[lit; [DOI](https://doi.org/10.1007/b98971)]**). Use $P_{t|t-1}=P_{t-1|t-1}/\delta$. $\delta=0.995$ per day gives an effective memory of about $1/(1-\delta)=200$ days.
- **Smoothing.** Use the RTS smoother only for historical analysis, never for decisions (§74–75).
- **Interval.** $\beta_{t|t}\pm z\sqrt{P_{t|t,\beta\beta}}$ is conditional on $\hat q$. Bootstrap or profile over $q$ to include its uncertainty.

**Failure modes**
- $\hat q$ at the boundary (common, and fine: it means no detectable drift).
- $q$ too large, so $\beta_t$ jumps at every price change.
- Promotion-driven price changes inflate apparent drift (exclude them or model them separately).
- Unmodelled level shifts load onto $\beta_t$ (add the $a_t$ state).

**Tests**
1. **Exact identity.** With $q=0$ and a diffuse prior, the final $\beta_{T|T}$ equals the OLS slope of $\tilde Y$ on $(1,\tilde T)$ and matches statsmodels `RecursiveLS`.
2. **Gaps.** Inserting NaN days changes only $P$, never the point estimates.
3. **Planted break.** With a step in $\beta$ at day 200 and price changes every 30 days, the 95% filtered band covers the truth in ≥ 90% of days once two post-break price changes have occurred.
4. **No-drift simulation.** $\hat q=0$ in a majority of replications, and the LR-test rejection rate is ≈ 5%.

#### T2. Method card: rolling and discounted windows (diagnostic and combination)
- **Window estimate.** $\hat\epsilon_{w,t}=\sum_{s\in(t-w,t]}\tilde T_s\tilde Y_s/\sum\tilde T_s^2$.
- **Exponentially weighted version.** $\hat\epsilon_{\lambda,t}=\sum\lambda^{t-s}\tilde T_s\tilde Y_s/\sum\lambda^{t-s}\tilde T_s^2$ with half-life $h=\ln2/\ln(1/\lambda)$. Choose $h\in\{30,90,180,365,\infty\}$ days by backtest loss (§3 E2).
- **Admissibility.** A window enters only if it has ≥ 2 independent price changes and sd($\tilde T$) ≥ 0.01. Otherwise its weight is 0 and the rest are renormalized. This also fixes v1's missing 60-day term.
- **Combination.** Use simplex weights learned on the backtest loss and shrunk toward equal weights (Pesaran–Pick AveW as the no-information default): $w=\lambda w^*+(1-\lambda)\mathbf 1/K$ with $\lambda=n_{events}/(n_{events}+20)$ **[heuristic]**.
- **Do not inverse-variance-weight nested windows.**
  - Under a constant elasticity and homoskedastic errors, the nested OLS estimates satisfy $\mathrm{Cov}(\hat\epsilon_S,\hat\epsilon_L)=\mathrm{Var}(\hat\epsilon_L)$ (a Hausman-type property) **[lit/inference]**.
  - The efficient combination then puts all weight on the longest window, which is wrong exactly when there *is* drift.
  - The same property gives a cheap drift test: $H=(\hat\epsilon_S-\hat\epsilon_L)^2/(\mathrm{Var}\hat\epsilon_S-\mathrm{Var}\hat\epsilon_L)\sim\chi^2_1$ under no drift. Under heteroskedastic or clustered errors, bootstrap $H$.
- **Tests.**
  - A window with no price change returns "inadmissible", never NaN or inf.
  - Weights are ≥ 0 and sum to 1.
  - Under a constant $\beta$, the combined bias is ≈ 0 and $H$ rejects in ≈ 5% of replications.

#### T3. Method card: regimes and the correct way to compute "E[ε_t]"
1. **Reclassify v1's ten regimes.**
   - PROMOTION, LOCAL_EVENT, SEASONAL_PEAK/LOW, SUPPLY_SHOCK (stockouts) and COMPETITIVE_SHOCK (when competitor entry is observed) are **observed**. Put them in $Z$ (so they can modulate $\tau$) and in $X$ (so they shift the level).
   - At most a K=2–3 latent regime remains (for example NORMAL / HIGH_DEMAND / LOW_DEMAND), optionally with transition probabilities driven by the news/economic shock index through `exog_tvtp`.
2. **Latent Markov-switching regression.** `MarkovRegression(endog=Ỹ, k_regimes=2, trend="c", exog=T̃, switching_exog=True, switching_variance=False, exog_tvtp=[1, shock_{t-1}])`, then `fit(search_reps=20)` (the default is 0; use random restarts against local maxima).
   - **Hamilton filter [lit]:** $\xi_{t|t-1}=\mathbf P_t'\xi_{t-1|t-1}$, $\xi_{t|t}\propto\xi_{t|t-1}\odot\eta_t$ with $\eta_{t,r}=N(y_t;\,a_r+\beta_r\tilde T_t,\sigma^2)$, and log-likelihood $\sum_t\log\mathbf 1'(\xi_{t|t-1}\odot\eta_t)$.
   - **Time-varying transitions:** $p_{rs,t}=\exp(z_t'\gamma_{rs})/\sum_{s'}\exp(z_t'\gamma_{rs'})$.
3. **Alternative without Markov dynamics: mixture of experts via EM.**
   - E-step: $\gamma_{tr}\propto\pi_r(X_t)\,N(y_t;a_r+\epsilon_rT_t+x_t'b_r,\sigma_r^2)$.
   - M-step: weighted least squares per regime, then a multinomial logit of $\gamma_{tr}$ on $X_t$ for the gating $\pi_r(X_t)$.
   - Use multiple restarts.
4. **Real-time probabilities** must be filtered, $P(S_t\mid\mathcal F_t)$, or predicted, $\pi_{r,t+1}=\mathbf P'\xi_{t|t}$. Never use smoothed probabilities for decisions.
5. **Correct mixing.** Expected demand is $E[Q_t(p)]=\sum_r\pi_{rt}Q_r(p)$, and its elasticity is $\sum_r\frac{\pi_{rt}Q_r(p)}{\sum_s\pi_{st}Q_s(p)}\epsilon_r$ (defect #1). The price simulator should *sample* regimes, or mix demand, and propagate the uncertainty.

**Identification requirements [heuristic]**
- Each regime needs ≥ 2 independent price changes inside it, and an expected duration ≥ 2× the typical spacing between price changes.
- Diagnostic: regime-specific information $I_r=\sum_t P(S_t=r\mid\cdot)\tilde T_t^2$. If $I_r$ is small, fix $\beta_r=\beta$ (switch only the intercept; switching_exog can be a per-regressor boolean list).

**Failure modes**
- Label switching (impose the ordering $a_1<a_2$).
- A degenerate regime that absorbs outliers (worse with switching_variance=True).
- Local maxima.
- "Regimes" that simply track promotions.

**Tests**
1. Simulate two regimes with known $\mathbf P$ and $\beta_r$ and recover them within their CIs.
2. Probabilities lie in [0, 1] and sum to 1.
3. **Leakage.** Deleting data after t leaves the day-t probability unchanged.
4. A hand-written Hamilton filter reproduces the statsmodels log-likelihood at fixed parameters to within 1e-8.
5. **Mixing.** Check against a numerical derivative of $\ln E[Q]$ (the defect #1 example).

#### T4. Method card: change-point detection (PELT and BOCPD)
**PELT [lit, Killick et al.]**
- Objective: $\min_{m,\tau_{1:m}}\sum_{j=0}^m\mathcal C(y_{(\tau_j+1):\tau_{j+1}})+\beta m$.
- Pruning: drop candidate $s$ when $F(s)+\mathcal C(y_{s+1:t})+K>F(t)$, with $K=0$ for RSS-type costs.
- **Cost choice.**
  - Level shifts: use `model="l2"` on $\tilde Y$.
  - Elasticity changes: use `model="linear"` (CostLinear) on columns $[\tilde Y, 1, \tilde T]$, where segment cost = RSS.
- **Penalty.** Because ruptures costs are RSS, not −2·log-likelihood, a BIC-type penalty must be scaled by the noise variance: $\beta=\hat\sigma^2\,(p+1)\log n$, where p is the number of coefficients per segment **[inference: with known $\sigma^2$, −2·log-likelihood = RSS/$\sigma^2$ + constant]**.
- **Parameters.** For slope changes set min_size ≥ 28 days so a segment can contain price variation. With n ≤ 2,000 use jump=1; the default jump=5 only considers every 5th index as a break.

**BOCPD [lit, Adams & MacKay]**
- Growth: $P(r_t=r_{t-1}+1,x_{1:t})=P(r_{t-1},x_{1:t-1})\,\pi_t^{(r)}(1-H)$.
- Change: $P(r_t=0,x_{1:t})=\sum_{r_{t-1}}P(r_{t-1},x_{1:t-1})\,\pi_t^{(r)}H$.
- Normalize each step. $\pi_t^{(r)}$ is the predictive density under the run-length-specific posterior.
- **Regression version** (Normal-inverse-gamma prior on $(a,\beta,\sigma^2)$ **[lit]**):
  - $\Lambda_n=\Lambda_0+X'X$, $\mu_n=\Lambda_n^{-1}(\Lambda_0\mu_0+X'y)$, $a_n=a_0+n/2$, $b_n=b_0+\tfrac12(y'y+\mu_0'\Lambda_0\mu_0-\mu_n'\Lambda_n\mu_n)$.
  - Predictive: Student-t with $2a_n$ degrees of freedom, location $x'\mu_n$ and scale² $(b_n/a_n)(1+x'\Lambda_n^{-1}x)$.
  - Use rank-one updates and cap the run length at $R_{max}=730$.
- **Defaults [heuristic].**
  - Hazard $H=1/365$ for elasticity regimes and $H=1/90$ for demand-level regimes.
  - Alarm when $P(r_t\le7\mid x_{1:t})>0.5$, with the threshold calibrated by simulation to at most 1 false alarm per product-year.

**SMB reality.** Level-shift detection is feasible. Slope-change detection needs ≥ 2 price changes on each side, so most products will never support it. Use the Hausman window test (T2) or the backtest standardized errors (§6.6) instead.

**Tests**
1. On iid no-change data, PELT with the scaled BIC penalty finds 0 changes in ≥ 95% of simulations.
2. A planted 1σ level shift is detected within ±7 days in ≥ 80% of simulations.
3. BOCPD run-length posteriors sum to 1 at every t, and with $H\to0$ the posterior equals the batch (single-segment) posterior.
4. A slope change with no price variation on one side must **not** be detected (catches a mis-specified cost).

### Gaps
- I found no primary study of PELT or BOCPD for detecting *slope* (elasticity) changes when price changes are sparse. All thresholds here are heuristics to be set by simulation.
- I did not verify West & Harrison or Durbin & Koopman equations, or the Inoue–Jin–Rossi (2017) rolling-window selection method, against the full texts (blocked or out of search budget). The Kalman recursions given are standard.
- The Pesaran & Pick (2011) bibliographic details come from a search summary; the paper itself was not opened.

---

## 3. Combining elasticity estimators when the true elasticity is never observed

### Takeaway
Weight members on criteria that carry information about the **causal slope**, not the demand level:
1. held-out **R-loss**;
2. **counterfactual backtests** on held-out historical price changes (predict Δlog Q for the realized Δlog p using only pre-change data);
3. **semi-synthetic simulation**, which plants a known elasticity into the business's own covariate and price paths.

Combine with simplex-constrained stacking shrunk toward equal weights (the forecast-combination puzzle). Get the ensemble's uncertainty by **jointly bootstrapping all members**, because they are strongly correlated. Treat member disagreement as a diagnostic, not as a variance.

Pure demand-prediction stacking and Bayesian model averaging can select a model whose elasticity is badly biased.

### Cited Findings
- **Why ordinary cross-validation fails.** Cross-validation cannot select CATE models because counterfactual outcomes are unobserved. Mahajan, Mitliagkas, Neal & Syrgkanis (ICLR 2024) evaluated 34 surrogate model-selection metrics across 78 datasets and 415 CATE estimators, including R-score and doubly-robust-based scores. — [arXiv:2211.01939](https://arxiv.org/abs/2211.01939)
- **Factual prediction criteria are limited.** Curth & van der Schaar (ICML 2023) group criteria into factual (prediction), plug-in surrogate and pseudo-outcome surrogate. They show the limits of factual prediction criteria that judge nuisance-model generalization, and find pseudo-outcome variants less prone to "congeniality bias" than plug-in ones. — [PMLR v202](https://proceedings.mlr.press/v202/curth23b/curth23b.pdf); [arXiv:2302.02923](https://arxiv.org/pdf/2302.02923)
- **Counterfactual cross-validation.** Saito & Yasui (ICML 2020, PMLR 119:8398–8407) aim to preserve the *rank order* of candidate CATE predictors' performance so that selection is stable. — [PMLR](https://proceedings.mlr.press/v119/saito20a.html); [code](https://github.com/usaito/counterfactual-cv)
- **EconML ensembling** (source code):
  - `RScorer.best_model` = argmax R-score.
  - `RScorer.ensemble(cate_models, eta=1000.0)` gives weights = softmax(eta × R-score) over models with finite scores and returns an `EnsembleCateEstimator` (a weighted average of member predictions).
  - — [EconML 0.17.0](https://pypi.org/project/econml/0.17.0/)
- **Stacking vs Bayesian model averaging.** Yao, Vehtari, Simpson & Gelman (2018), *Bayesian Analysis* 13(3):917–1007. Bayesian model averaging is flawed in the M-open setting (the true process is not among the candidates). Stacking of predictive distributions generalizes point-estimate stacking to any proper scoring rule and uses PSIS-LOO for the leave-one-out predictive densities. — [Project Euclid](https://projecteuclid.org/journals/bayesian-analysis/volume-13/issue-3/Using-Stacking-to-Average-Bayesian-Predictive-Distributions-with-Discussion/10.1214/17-BA1091.full); [Columbia PDF](https://sites.stat.columbia.edu/gelman/research/published/stacking_paper_discussion_rejoinder.pdf)
- **Forecast combination puzzle.** Equal weights often beat estimated "optimal" weights. Smith & Wallis (2009) attribute this to estimation error in the weights: when the optimal weights are near equal, the simple average can dominate. — [Tinbergen DP 14-127](https://papers.tinbergen.nl/14127.pdf); [Wang, Hyndman et al., 50-year review, arXiv:2205.04216](https://arxiv.org/pdf/2205.04216); [ResearchGate](https://www.researchgate.net/publication/265186991_The_forecast_combination_puzzle_A_simple_theoretical_explanation)
- **Stacked regressions** (non-negative least squares on cross-validated predictions): **[lit]** Breiman (1996), *Machine Learning* 24:49–64. — [DOI](https://doi.org/10.1007/BF00117832)

### Inferences

#### E0. Ensemble members for v2 (replacing v1 A–E)

| Member | Estimator | Tier (§76) | Notes |
|---|---|---|---|
| M1 | Within-product two-way FE OLS (product, day-of-week, week fixed effects) | 6 | Baseline |
| M2 | Pooled partially linear DML (§1.3 with φ = 1) | 5 | Default |
| M3 | LinearDML with interactions | 5 | Gated |
| M4 | CausalForestDML | 5 | Gated (§1.5) |
| M5 | Event study / local DiD around price changes, with untreated products of the business as controls; pre-trend test on leads | 3 (if pre-trends pass) | Closest to a natural experiment |
| Combiner | Bayesian hierarchical layer (E5) | — | Not a peer member |
| (not a member) | GBM demand model | 7 | Nuisance and simulator only |

#### E1. Criterion 1: held-out R-loss
- **Score.** $R(\hat\tau)=\sum_{i\in H}(\tilde Y_i-\hat\tau(Z_i)\tilde T_i)^2$, with residuals from nuisance models fitted *without* the held-out block $H$ (as RScorer does), and score $=1-R(\hat\tau)/R(\bar\theta)$.
- **Noise.** Score differences are small relative to noise when $\tilde T$ is small. Compute a paired cluster-bootstrap SE of each difference and prefer a model only if it wins by more than 2 SE.
- **EconML's softmax weights.** With the default η = 1000, a score gap of 0.005 already gives an e^5 ≈ 150:1 weight ratio, effectively winner-take-all. If softmax is used, set $\eta\approx1/SE(\Delta score)$.

#### E2. Criterion 2: counterfactual backtest on held-out price changes (v1 §49 made precise)
**Event definition [heuristic].** A product $j$ at date $t_e$ with $\Delta_e=\log p_{after}-\log p_{before}$, $|\Delta_e|\ge0.02$, no promotion and no stockout in $[t_e-w,t_e+w)$, $w=14$ days.

**Procedure**
1. Fit each member on data before $t_e-1$ (walk-forward).
2. **Predicted change.** $\widehat{\Delta\log Q}_{k,e}=\hat\tau_k(Z_e)\Delta_e+[\hat g(X_{after})-\hat g(X_{before})]$. The second term is the change explained by controls (season, weather, competitor index).
3. **Observed change.** $\Delta\log Q_e=\overline{\log Q}_{[t_e,t_e+w)}-\overline{\log Q}_{[t_e-w,t_e)}$, with noise variance $v_e\approx\hat\sigma^2(1/n_{before}+1/n_{after})$.
4. **Loss.** $L_k=\sum_e(\Delta\log Q_e-\widehat{\Delta\log Q}_{k,e})^2/v_e$.
5. **Event-level elasticity.** $\epsilon_e=(\Delta\log Q_e-\Delta\hat g_e)/\Delta_e$ has noise variance $\approx v_e/\Delta_e^2$, so small price changes carry almost no information.

**Caveats**
- The event elasticity is unbiased only if the price change was not timed to demand shocks. Run a "lead" placebo, $\Delta\log Q$ over $[t_e-2w,t_e-w)$ vs $[t_e-w,t_e)$, and drop events that fail.
- With 3–20 events per product, pool across products (and across tenants in §80).

#### E3. Criterion 3: semi-synthetic ("plasmode") simulation, which gives truth for weights and calibration
1. **Data.** Keep the business's real $X$, real price path $T$ and the real gap/stockout pattern.
2. **Simulated outcome.** $Y^{sim}=\hat g(X)+\epsilon^{true}(Z)\,T+u^*$.
   - $\epsilon^{true}$ is drawn from the prior (for example $N(-1.5,0.7^2)$), optionally heterogeneous.
   - $u^*$ = moving-block-resampled real residuals plus $\kappa\,(T-\hat e(X))$, with $\kappa\in\{0,\pm0.5\}$, to inject pricing endogeneity.
3. **Replications.** R ≥ 200 per business. They are cheap for the linear members; use fewer for forests.
4. **Outputs.** Each member's $|\hat\epsilon-\epsilon^{true}|$, which gives weights by minimum MSE, the calibration data for §5 and the coverage checks for §4.
5. **Caveat.** "Congeniality": rankings favor methods that resemble the simulator's $\hat g$ (Curth & van der Schaar). Rotate the simulator's nuisance class (ridge, LightGBM, spline).

#### E4. Weight estimation and combined uncertainty
- **Stacking.** $w^*=\arg\min_{w\in\Delta_K}\sum_e(y_e-\sum_kw_k\hat y_{k,e})^2/v_e$, solved by SLSQP or by NNLS followed by normalization. Then shrink: $w=\lambda w^*+(1-\lambda)\mathbf 1/K$ with $\lambda=n_e/(n_e+20)$ **[heuristic]**.
  - For time series, use **leave-future-out**, not leave-one-out, predictive densities when stacking Bayesian demand models **[lit, not verified]**.
- **Inverse-variance (GLS) weights.** $w=\Sigma^{-1}\mathbf 1/(\mathbf 1'\Sigma^{-1}\mathbf 1)$ and $\mathrm{Var}=1/(\mathbf 1'\Sigma^{-1}\mathbf 1)$. $\Sigma$ must come from a **joint** bootstrap: resample clusters or blocks and refit every member on each replicate.
  - **[own computation]** With member sd 0.30/0.35/0.40 and ρ = 0.9, GLS gives weights (1.44, 0.15, −0.59), which are unstable and negative, with sd 0.283.
  - Naive independent weights (0.44, 0.32, 0.25) claim sd 0.198, but the true sd of that combination is 0.329.
  - Rule: constrain weights to the simplex, and report the SE from the bootstrap distribution of the *combined* estimator.
- **Bayesian model averaging.** $P(M_k\mid D)\propto p(D\mid M_k)P(M_k)$ needs comparable likelihoods for the same $Y$, which DML and forest estimators do not have, and in M-open settings it concentrates on one model (Yao et al.). Do not use it for the elasticity combiner. It is acceptable, via stacking, for choosing among Bayesian demand-model variants.

#### E5. Recommended combiner: a hierarchical "bias-aware" model
- **Model.** Treat each member's output $(\hat\epsilon_k,se_k)$ as $\hat\epsilon_k=\epsilon+b_k+e_k$, with $e$ jointly normal with covariance $\Sigma$ from E4 and a member bias $b_k\sim N(0,s_k^2)$.
- **Bias scale by tier [heuristic; calibrate with E3].** $s_k$ = 0 for Tier 1, 0.1 for Tier 3, 0.2 for Tier 5, 0.3 for Tier 6 and 0.6 for Tier 7.
- **Prior.** $\epsilon\sim N(\mu_{prior},s_0^2)$ from §41–42.
- **Posterior.**
  - $\mathrm{Var}(\epsilon\mid\cdot)=\big(s_0^{-2}+\mathbf 1'(\Sigma+S)^{-1}\mathbf 1\big)^{-1}$ with $S=\mathrm{diag}(s_k^2)$.
  - $E(\epsilon\mid\cdot)=\mathrm{Var}(\epsilon\mid\cdot)\big(s_0^{-2}\mu_{prior}+\mathbf 1'(\Sigma+S)^{-1}\hat{\boldsymbol\epsilon}\big)$.
  - Use the simplex-projected version if negative implied weights appear.
- **Effect.** Non-sampling (identification) error enters the reported interval openly, instead of being hidden in a "confidence" label.

#### Pitfalls the spec should name
- **Good demand prediction does not mean a good elasticity.** Prediction loss is dominated by level and seasonality, so a model that ignores price can win on MAPE. A confounded model also predicts well in-distribution, because price was set high exactly when demand was expected to be high.
- **Endogenous held-out events reward confounded members.** Use the lead placebo filter (E2).
- **Reusing events.** Using the same events to choose weights and to report accuracy is optimistic. Use nested walk-forward.
- **Agreement is not accuracy.** Members share the same data and the same confounding, so all can agree and all be wrong.

#### Tests
1. Members with planted biases (+0.3, 0, −0.3) and known variances: the combiner's MSE is ≤ the best single member's and coverage is ≈ nominal (E3 harness).
2. **Duplication invariance.** Adding an exact copy of a member leaves the stacked estimate unchanged. Naive inverse-variance weighting fails this, because it double counts.
3. Weights lie on the simplex, and the combined estimate falls within [min, max] of the members.
4. **No leakage.** Weights used at date t depend only on events before t.

### Gaps
- I did not verify Mahajan et al.'s specific ranking of metrics (only their scope), Breiman (1996), or the leave-future-out stacking literature.
- I found no published study that stacks elasticity estimators on retail price-change backtests. E2–E5 are derived designs.
- The tier bias scales $s_k$ have no empirical source. They must be calibrated by E3, or later by experiments (§77).

---

## 4. Uncertainty quantification: calibrated 80% and 95% intervals

### Takeaway
Keep two kinds of interval apart.
- **(A) Intervals for the elasticity parameter.** Obtain these from orthogonal-score sandwich SEs, multiplier, wild-cluster, pairs-cluster or block bootstraps, or Bayesian posteriors. They **cannot** be conformalized.
- **(B) Prediction intervals for observables** (daily demand, Δlog Q after a price change, simulated revenue). These can be conformalized: split conformal across exchangeable units, and ACI/AgACI, EnbPI or weighted conformal for time series.

Coverage must be *tested*:
- for (A), with the semi-synthetic harness (truth known);
- for (B), with walk-forward empirical coverage plus binomial, Kupiec and Christoffersen tests.

Sample-size limits matter. About 100 intervals are needed to tell 80% coverage from 72%. With the common step size γ = 0.005, ACI's guarantee is vacuous over one year of daily data.

### Cited Findings
- **Wild cluster bootstrap.** Cameron, Gelbach & Miller (2008), *Review of Economics and Statistics* 90(3):414–427. Standard cluster-robust tests over-reject with few (5–30) clusters. They propose a wild cluster bootstrap-t (with asymptotic refinement) using Rademacher weights. — [IDEAS/RePEc](https://ideas.repec.org/a/tpr/restat/v90y2008i3p414-427.html); [Stata WCB overview](https://www.stata.com/features/overview/wild-cluster-bootstrap-inference/); [World Bank blog on few clusters](https://blogs.worldbank.org/en/impactevaluations/beware-of-studies-with-a-small-number-of-clusters)
- **Very few clusters.** Webb (2023), "Reworking wild bootstrap-based inference for clustered errors", *Canadian Journal of Economics* (six-point weights). — [Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1111/caje.12661); [QED WP 1315](http://qed.econ.queensu.ca/working_papers/papers/qed_wp_1315.pdf). Fast implementation: [fwildclusterboot](https://s3alfisc.github.io/blog/post/1000x-faster-wild-cluster-bootstrap-inference-in-r-with-fwildclusterboot/)
- **Automatic block length** (arch `optimal_block_length`, source code; Politis & White 2004 with the Patton–Politis–White 2009 correction):
  - $b^{OPT}=\left(\frac{2g^2}{d}n\right)^{1/3}$, with $g=\sum_{k=-m}^m h(k/m)|k|\hat\gamma_k$, $h(x)=\min(1,2(1-|x|))$, $d=c(\hat\sigma^2)^2$, $\hat\sigma^2=\sum_{k=-m}^m h(k/m)\hat\gamma_k$.
  - $c=2$ for the stationary bootstrap and 4/3 for the circular bootstrap.
  - $m$ is the first lag where $k_n=\max(5,\log_{10}n)$ consecutive autocorrelations lie inside $\pm2\sqrt{\log_{10}(n)/n}$, with $m\le\lceil\sqrt n+k_n\rceil$.
  - — [arch 8.0.0](https://pypi.org/project/arch/8.0.0/)
- **MAPIE time-series conformal** (source code):
  - `TimeSeriesRegressor(method="enbpi" | "aci")`. "EnbPI only corresponds to TimeSeriesRegressor if the cv argument is of type BlockBootstrap", and `update()` "will replace the oldest [conformity scores] with the newest".
  - ACI update as implemented: $\alpha_{t+1}=\mathrm{clip}(\alpha_t+\gamma(\alpha-\mathbb 1\{y_t\notin\hat C_t\}),0,1)$. "If the quantile is equal to zero, the method will produce an infinite set size."
  - `BlockBootstrap(n_resamplings=30, length=None, n_blocks=None, overlapping=False)`.
  - MAPIE also ships exchangeability tests (permutation and online martingale tests) and risk monitoring.
  - — [MAPIE 1.5.0](https://pypi.org/project/mapie/1.5.0/)
- **ACI.** Gibbs & Candès (NeurIPS 2021): $\alpha_{t+1}=\alpha_t+\gamma(\alpha-\mathbb 1\{Y_t\notin\hat C_t(\alpha_t)\})$. Long-run coverage converges to $1-\alpha$ even under arbitrary distribution shift. Performance is highly sensitive to γ; the theory suggests γ proportional to the variation of $\alpha^*_t$ but gives no estimation procedure. — [arXiv:2106.00170](https://arxiv.org/pdf/2106.00170); [NeurIPS PDF](https://papers.neurips.cc/paper/2021/file/0d441de75945e5acbc865406fc9a2559-Paper.pdf); [critique, arXiv:2511.04275](https://arxiv.org/pdf/2511.04275)
  - Follow-up for tuning γ: Gibbs & Candès, "Conformal Inference for Online Prediction with Arbitrary Distribution Shifts", *JMLR* 25 (2024). — [JMLR](http://www.jmlr.org/papers/v25/22-1218.html) (title only)
  - Deterministic bound **[lit, not verified verbatim]**: $\big|\frac1T\sum_{t=1}^T\mathrm{err}_t-\alpha\big|\le\frac{\max(\alpha_1,1-\alpha_1)+\gamma}{\gamma T}$.
- **AgACI.** Zaffran et al., "Adaptive Conformal Predictions for Time Series" (ICML 2022), aggregates ACI over a grid of γ values. — [PMLR v162](https://proceedings.mlr.press/v162/zaffran22a/zaffran22a.pdf)
- **EnbPI.** Xu & Xie (ICML 2021) train B bootstrap models and form leave-one-out ensemble aggregates for the residuals (no separate calibration split). Coverage is approximate and asymptotic, under stationary, strongly mixing errors and a consistent base predictor. — [PMLR v139](https://proceedings.mlr.press/v139/xu21h.html); [official code](https://github.com/hamrel-cxu/EnbPI); [2026 benchmark of time-series conformal methods](https://arxiv.org/html/2601.18509v2) (title only)
- **Conformal beyond exchangeability.** Barber, Candès, Ramdas & Tibshirani (2023), *Annals of Statistics* 51(2):816–845, cover non-exchangeable data and non-symmetric algorithms, with electricity and election experiments. — [Project Euclid](https://projecteuclid.org/journals/annals-of-statistics/volume-51/issue-2/Conformal-prediction-beyond-exchangeability/10.1214/23-AOS2276.full); [arXiv:2202.13415](https://arxiv.org/abs/2202.13415)
  - Weighted-quantile bound **[lit, not verified verbatim]**: coverage ≥ $1-\alpha-\sum_i\tilde w_i\,d_{TV}(Z,Z^{i})$ with $\tilde w_i=w_i/(1+\sum_jw_j)$; example weights $w_i=0.99^{\,n+1-i}$.
- **Counterfactual conformal.** Lei & Candès (2021), *JRSS-B* 83:911–938. Conformal intervals for counterfactuals and individual treatment effects have finite-sample average coverage in randomized experiments and approximate, doubly robust coverage under strong ignorability. They build on weighted split conformal. — [Wiley](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/rssb.12445); [systematic review, arXiv:2509.21660](https://arxiv.org/html/2509.21660v1)
- **Coverage backtests.** Kupiec (1995) proportion-of-failures likelihood-ratio test. Christoffersen (1998) independence and conditional coverage tests, with $LR_{CC}=LR_{POF}+LR_{IND}\sim\chi^2_2$. — [Federal Reserve FEDS 2005-21 review](https://www.federalreserve.gov/pubs/feds/2005/200521/200521pap.pdf); [MathWorks VaR backtesting overview](https://www.mathworks.com/help/risk/overview-of-var-backtesting.html)
- **Simulation-based calibration (SBC) for Bayesian computation.** Talts et al. (2018/2020) check rank uniformity. Modrák et al. (*Bayesian Analysis*, 2023/2025) add the joint log-likelihood as a test quantity. — [Stan SBC guide](https://mc-stan.org/docs/stan-users-guide/simulation-based-calibration.html); [Modrák et al. PDF](https://paulbuerkner.com/publications/pdf/2023__Modrak_et_al__Bayesian_Analysis.pdf)
- **Further [lit] references.**
  - Conformalized quantile regression: Romano, Patterson & Candès (2019). — [arXiv:1905.03222](https://arxiv.org/abs/1905.03222)
  - Conformal tutorial: Angelopoulos & Bates. — [arXiv:2107.07511](https://arxiv.org/abs/2107.07511)
  - Moving-block bootstrap: Künsch (1989). — [DOI](https://doi.org/10.1214/aos/1176347265)
  - Stationary bootstrap: Politis & Romano (1994). — [DOI](https://doi.org/10.1080/01621459.1994.10476870)
  - Interval score and proper scoring rules: Gneiting & Raftery (2007). — [DOI](https://doi.org/10.1198/016214506000001437)

### Inferences

#### U0. Taxonomy for the spec

| Object | Method | Valid when | How tested |
|---|---|---|---|
| Elasticity θ (pooled or interaction) | Cluster-robust sandwich SE (U1) | G ≳ 30 clusters, orthogonal score | Semi-synthetic coverage (U7) |
| θ with few clusters | Wild cluster bootstrap-t (U2) | G ≥ 6 (Webb weights if G ≤ 12) | Same |
| θ including nuisance and model-selection variability; ensemble Σ | Pairs-cluster bootstrap with full refit (U3) | G ≳ 30 | Same |
| θ for a single product series | Moving-block or stationary bootstrap on the score (U4) | ≥ 2 independent price changes; < 5% degenerate resamples | Same |
| Product-level θ_j, prior-informed | Bayesian / EB posterior (U5, §1.7, §3 E5) | Prior correct on average | SBC plus semi-synthetic |
| τ(x) from a forest | Bootstrap of little bags | Honest forest, small bias | Semi-synthetic (§1.5 tests) |
| β_t (time-varying) | Kalman filtered variance, plus a bootstrap over q | Model adequate | Planted-break simulation (§2 T1) |
| Daily demand, Δlog Q, revenue | Split conformal / CQR / ACI / EnbPI / weighted conformal (U6) | See U6 | Walk-forward coverage (U7) |
| Simulated demand at an *unobserved* price | Parameter draws ⊕ conformal residuals (U6) | Model-based only | Price-change backtest (§3 E2) |

#### U1. Analytic orthogonal-score interval (pooled partially linear model)
- **Estimate and score.** $\hat\theta=\sum\tilde T\tilde Y/\sum\tilde T^2$ and $\psi_i=(\tilde Y_i-\hat\theta\tilde T_i)\tilde T_i$, with $J=\frac1n\sum\tilde T_i^2$.
- **Cluster-robust variance.** $\hat V=\frac{G}{G-1}\cdot\frac{\frac1n\sum_g(\sum_{i\in g}\psi_i)^2}{nJ^2}$.
- **Interval.** $\hat\theta\pm t_{G-1,1-\alpha/2}\sqrt{\hat V}$.
- **Clusters.** Use product × 28-day blocks within one product's series, or products when pooled.
- **Cross-fitting randomness.** Use the median over 5 repeated splits (§1.3).
- **Test.** For iid homoskedastic data this interval matches OLS's HC0 interval on the residuals. The coverage gates are in U7.

#### U2. Wild cluster bootstrap (few clusters)
1. For b = 1…B (B = 999 by default, 9,999 for publication-grade p-values), draw $v_g$ per cluster: Rademacher ±1, or Webb's six-point $\{\pm\sqrt{1.5},\pm1,\pm\sqrt{0.5}\}$ each with probability 1/6 when G ≤ 12. Rademacher has only $2^G$ distinct draws, which is 64 when G = 6.
2. Set $\tilde Y^*_i=\hat\theta\tilde T_i+v_{g(i)}\hat u_i$ with $\hat u_i=\tilde Y_i-\hat\theta\tilde T_i$.
3. Compute $\hat\theta^*_b$ and the bootstrap t, $t^*_b=(\hat\theta^*_b-\hat\theta)/se^*_b$.
4. Interval: $[\hat\theta-t^*_{(1-\alpha/2)}se,\;\hat\theta-t^*_{(\alpha/2)}se]$.

Notes:
- Imposing the null (restricted residuals, WCR) is preferred for tests (see the CGM, Webb and fwildclusterboot sources). Invert the test for intervals when G < 10.
- Holding the nuisance residuals fixed is the multiplier-bootstrap shortcut. It is justified asymptotically by Neyman orthogonality **[lit]**, but it ignores nuisance re-estimation. Use U3 when G allows.
- With G < 6, report only Bayesian/EB intervals.
- **Test.** With G = 8 heterogeneous clusters, CR1 under-covers while WCB coverage stays in the binomial band.

#### U3. Pairs-cluster bootstrap with full refit
- **Procedure.** Resample products with replacement and refit everything: nuisances, members and weights. B = 200–500 (refits are expensive).
- **Outputs.** The ensemble's joint Σ (§3 E4) and the variability of the weights.
- **When.** G ≳ 30, in line with CGM's finding that problems arise with 5–30 clusters.

#### U4. Moving-block or stationary bootstrap (one product's series)
- **Block length.** Apply `arch.bootstrap.optimal_block_length` to the score series $\psi_t$ (not raw Y), and use the stationary bootstrap (`StationaryBootstrap`, mean block length = b_sb), B = 999.
- **Gaps.** Resample on the calendar index with NaNs kept inside blocks.
- **Degenerate resamples.** Report the share of resamples where $\sum\tilde T^{*2}\approx0$. These occur when price variation sits in a few *short* episodes, such as a one-week promotion. If more than 5% of resamples are degenerate, the interval is untrustworthy, so fall back to U5. Persistent price steps are fine: **[own computation]** 0 of 5,000 resamples were degenerate with 3 persistent steps and b = 14.
- **Test.** With AR(1) residuals (ρ = 0.5), an iid bootstrap under-covers and the block bootstrap covers ≈ nominal.

#### U5. Bayesian posterior intervals
- **Conjugate normal update.** $\epsilon\mid\hat\theta\sim N\big(\frac{s_0^{-2}\mu_0+se^{-2}\hat\theta}{s_0^{-2}+se^{-2}},\frac1{s_0^{-2}+se^{-2}}\big)$, with equal-tailed 80% and 95% intervals.
- **Hierarchical models.** Use §1.7 and §3 E5.
- **What coverage means.** Frequentist coverage holds on average over the prior if the prior is right, which is why the prior must itself be validated (§41).
- **SBC test.** Draw ε from the prior, simulate data, and compute the rank of ε within the posterior draws (L = 100 draws, 1,000 replications). The rank histogram must pass a χ² uniformity test (p > 0.01), and 80% intervals must fall in their binomial band.

#### U6. Conformal intervals for observable outcomes
**Split conformal** (for exchangeable units such as held-out products or businesses):
- Score $s_i=|y_i-\hat\mu(x_i)|$, or a normalized $|y_i-\hat\mu|/\hat\sigma(x_i)$.
- $\hat q$ = the $\lceil(n_{cal}+1)(1-\alpha)\rceil$-th smallest score; interval $\hat\mu(x)\pm\hat q$.
- **Calibration-set size [lit].** A finite interval needs $n_{cal}\ge\lceil1/\alpha\rceil-1$ (19 for 95%). Aim for n_cal ≳ 100 for stable widths.
- **CQR [lit].** Score $\max(\hat q_{lo}(x)-y,\;y-\hat q_{hi}(x))$; interval $[\hat q_{lo}-\hat q,\;\hat q_{hi}+\hat q]$, using LightGBM quantile objectives at α/2 and 1 − α/2.

**Daily time series** are *not* exchangeable (seasonality, drift). Wrap the base forecaster or quantile model with ACI.
- **[own computation]** With γ = 0.005, the deterministic ACI bound is 0.52 at T = 365 for α = 0.05 (vacuous) and 0.26 at T = 730. With γ = 0.05 it is 0.055 at T = 365 and 0.027 at T = 730. For α = 0.20 and γ = 0.05 it is 0.047 at T = 365.
- **Recommendation.** For SMB daily data, use AgACI-style aggregation over γ ∈ {0.005, 0.01, 0.02, 0.05, 0.1}, or a fixed γ in the range 0.02–0.05.
- **Infinite intervals.** When $\alpha_t\le0$ the interval is infinite. Display it as "too uncertain to bound", but still count it in coverage.

**EnbPI** (MAPIE `BlockBootstrap(n_resamplings=30, length = block length from U4)`) can serve as the demand-forecast interval when retraining B models is affordable. Its guarantee is asymptotic only.

**Weighted conformal** (Barber et al.) with $w_i=0.99^{\text{age}_i}$ is a simple non-exchangeable fallback.

**Simulated outcomes at a new price $p'$:**
1. Draw $\epsilon^{(s)}$ from the posterior (U5 / E5).
2. Draw a residual $u^{(s)}$ from the conformal residual distribution.
3. $Q^{(s)}(p')=\exp\{\widehat{\log Q}(p_0)+\epsilon^{(s)}\log(p'/p_0)+u^{(s)}\}$.

A price that was never observed is a covariate shift in T. Lei–Candès-style weighting would need the density ratio of price given X, which cannot be estimated with a few price levels. So label these intervals "model-based", and validate them on the event backtest (§3 E2): form 80% and 95% intervals for $\Delta\log Q_e$ and test their coverage.

#### U7. Coverage tests and acceptance criteria
**Binomial acceptance bands [own computation]** (two-sided 95% band for empirical coverage $\hat c$):

| Nominal | N = 30 | N = 50 | N = 100 | N = 200 | N = 500 |
|---|---|---|---|---|---|
| 0.80 | [0.633, 0.933] | [0.680, 0.900] | [0.720, 0.880] | [0.745, 0.855] | [0.764, 0.834] |
| 0.95 | [0.867, 1.000] | [0.880, 1.000] | [0.900, 0.990] | [0.920, 0.980] | [0.930, 0.968] |

**Kupiec proportion-of-failures test.** With $x$ misses out of $N$ and $p=1-$nominal: $LR_{POF}=-2\ln[(1-p)^{N-x}p^x]+2\ln[(1-\hat p)^{N-x}\hat p^x]\sim\chi^2_1$, where $\hat p=x/N$.

**Christoffersen independence test.** From transition counts $n_{ab}$ of the miss indicators: $\pi_{01}=n_{01}/(n_{00}+n_{01})$, $\pi_{11}=n_{11}/(n_{10}+n_{11})$, $\pi=(n_{01}+n_{11})/N$.
- $LR_{IND}=-2\ln[(1-\pi)^{n_{00}+n_{10}}\pi^{n_{01}+n_{11}}]+2\ln[(1-\pi_{01})^{n_{00}}\pi_{01}^{n_{01}}(1-\pi_{11})^{n_{10}}\pi_{11}^{n_{11}}]\sim\chi^2_1$.
- $LR_{CC}=LR_{POF}+LR_{IND}\sim\chi^2_2$.
- This matters because misses clustered in time (one regime) are as bad as too many misses.

**Interval score [lit, Gneiting & Raftery]** for comparing methods (lower is better): $S_\alpha(l,u;y)=(u-l)+\frac2\alpha(l-y)\mathbb 1\{y<l\}+\frac2\alpha(y-u)\mathbb 1\{y>u\}$.

**Parameter intervals (type A) can only be tested in simulation.**
- Use ≥ 500 replications per scenario.
- Scenario grid: 2–3 vs 10+ price changes; gaps; promotions; stockouts; pricing endogeneity κ.

**Proposed v2 acceptance criteria [heuristic]:**
- Across the scenario grid, 80% intervals have coverage in [0.75, 0.85] and 95% intervals in [0.92, 0.98].
- On the real walk-forward, outcome intervals pass Kupiec and Christoffersen at the 1% level.

**Tests that catch wrong implementations**
1. **Split conformal**, over 1,000 replications on exchangeable data: coverage must be ≥ 1 − α and ≤ 1 − α + 1/(n_cal + 1) **[lit]**. Using $\lceil n(1-\alpha)\rceil$ instead of $\lceil(n+1)(1-\alpha)\rceil$ shows up as slight under-coverage.
2. **ACI.** After a mid-series doubling of the noise variance, rolling 90-day coverage returns to the band. A flipped update sign makes coverage collapse.
3. **Bootstrap.** Under homoskedastic iid clusters, the wild cluster bootstrap and the analytic interval agree to within ~10% in width.

### Gaps
- I could not verify the exact ACI bound or the EnbPI algorithm text (fetch blocked). The MAPIE implementation was checked instead.
- I found no evidence on the finite-sample coverage of causal-forest intervals at n ≈ 1,000 with a continuous, few-level treatment.
- Romano et al. (CQR), Angelopoulos & Bates, and "conformal PID control" (Angelopoulos, Candès & Tibshirani, NeurIPS 2023, a newer alternative to ACI) were not verified in-session because the search budget was exhausted.

---

## 5. Principled data-quality, identification and confidence scores

### Takeaway
Replace v1's three unspecified $f(\cdot)$ scores with three layers:
1. **An auditable diagnostic vector** of raw quantities with units.
2. **An identification report**: the §76 tier, information shares by source of price variation, and an omitted-variable-bias **robustness value** (implemented in EconML, source code verified).
3. **Confidence as calibrated probabilities of named events**: P(sign correct), P(|ε − ε̂| ≤ 0.5), P(ε < −1) and expected absolute error. These come from the posterior and interval machinery and are recalibrated with an isotonic map fitted on semi-synthetic replicates and price-change backtests.

"DQS" becomes a *predicted standard error* computed from the data design before any model is fitted. Every score is accepted only if its reliability diagram stays within binomial bands and it ranks realized errors correctly.

### Cited Findings
- **EconML DML sensitivity analysis** (source code, `econml/validate/sensitivity_analysis.py` and `econml/dml/dml.py`; tagged `[Chernozhukov2022]`):
  - `dml_sensitivity_values` computes $\theta=E[\tilde Y\tilde T]/E[\tilde T^2]$, $\sigma^2=E[(\tilde Y-\theta\tilde T)^2]$ and $\nu^2=1/E[\tilde T^2]$, with a delta-method covariance.
  - `sensitivity_interval` uses $C=|\rho|\sqrt{c_y}\sqrt{c_t/(1-c_t)}/2$ and bounds $\theta\pm C(\sigma^2\sqrt{\nu^2/\sigma^2}+\nu^2\sqrt{\sigma^2/\nu^2})=\theta\pm|\rho|\sqrt{c_y}\sqrt{c_t/(1-c_t)}\,\sigma\nu$. The 'ci' variant adds normal quantiles. Defaults: c_y = c_t = 0.05, ρ = 1, α = 0.05.
  - `robustness_value(null_hypothesis=0, alpha=0.05)` is "the level of confounding (between 0 and 1) in *both* the treatment and outcome that would result in enough omitted variable bias such that we can no longer reject the null hypothesis". It is found by binary search and returns 0 if the interval already contains the null.
  - — [EconML 0.17.0](https://pypi.org/project/econml/0.17.0/); [EconML references page](https://www.pywhy.org/EconML/spec/references.html)
  - The formula has the structure of Chernozhukov, Cinelli, Newey, Sharma & Syrgkanis, "Long story short: omitted variable bias in causal machine learning" (NBER WP 30302, 2022) **[lit; I made the attribution by matching the formula; the paper was not fetched]**. — [NBER](https://www.nber.org/papers/w30302)
- **v1's tier hierarchy** (randomized experiments = Tier 1 … pure predictive correlations = Tier 7) is required in the output. — [v1 spec §76](/home/user/laee-engine/docs/spec/v1-original.md)
- **Calibration diagnostics for CATE models** in DRTester ($\mathrm{Cal}_G$, $\mathrm{Cal}_O$, $R^2_C$). — [EconML 0.17.0](https://pypi.org/project/econml/0.17.0/)
- **Coverage tests** (binomial, Kupiec, Christoffersen) as cited in §4. — [Federal Reserve FEDS 2005-21](https://www.federalreserve.gov/pubs/feds/2005/200521/200521pap.pdf)
- **Probability-calibration tools [lit, not verified in-session].**
  - Isotonic-regression calibration of scores: Zadrozny & Elkan (2002). — [DOI](https://doi.org/10.1145/775047.775151)
  - Proper scoring rules (Brier, interval score): Gneiting & Raftery (2007). — [DOI](https://doi.org/10.1198/016214506000001437)

### Inferences

#### 5.1 Diagnostic vector D (computed and stored per product × estimate, shown in the UI)
1. **Valid rows** $n$: product-days that are not stocked out and not missing.
2. **Effective rows.**
   - $n_{eff}=n(1-\hat\rho_1)/(1+\hat\rho_1)$, using the lag-1 autocorrelation of $\tilde Y$ (AR(1) approximation).
   - For weighted estimators, Kish's $(\sum w)^2/\sum w^2$.
   - **[own computation]** n = 365 gives $n_{eff}$ = 243, 122 and 41 for ρ = 0.2, 0.5 and 0.8.
3. **Independent price changes $K_{pc}$ [heuristic definition].** $|\Delta\log p|\ge0.02$; the new price is held ≥ 7 days; changes are ≥ 14 days apart; not a promotion; no stockout within ±7 days.
4. **Price variation.**
   - sd($\tilde T$) and CV(p) = sd(p)/mean(p).
   - **Residual share** RS = Var($\tilde T$)/Var($T$): the share of price variation not predictable from the controls. RS < 0.1 means weak identification and poor overlap.
5. **Missingness.** Share of calendar days missing, and the longest gap.
6. **Stockout and censoring share** (v1 §63–64).
7. **Promotion contamination.**
   - Promotion share of price variance: $\sum_{promo}(T-\bar T)^2/\sum(T-\bar T)^2$.
   - Promotion information share $I_{promo}$ (§1.1).
8. **Recency.** Days since the last independent price change, days since the last observation, and the share of rows in the last 180 days.
9. **Precision.** se($\hat\epsilon$), 80% and 95% interval widths, and the **prior share** $B=se^2/(se^2+s_0^2)$. B > 0.8 means "mostly industry prior".
10. **Model agreement.**
    - Between-member dispersion, plus a meta-analytic heterogeneity statistic $Q=\sum_k(\hat\epsilon_k-\bar\epsilon)^2/se_k^2$ and $I^2=\max(0,1-(K-1)/Q)$.
    - Because the members are correlated, compare Q with its **joint-bootstrap** null distribution, not with $\chi^2_{K-1}$.
11. **Sensitivity.** Robustness values RV₀ (null ε = 0, the sign) and RV₋₁ (null ε = −1, the elastic/inelastic boundary), from EconML.
12. **Backtest.** Number of held-out price-change events, the mean and sd of the standardized errors $z_e$, and empirical 80% and 95% coverage.

#### 5.2 IDS becomes an identification report: tier + information shares + robustness value
- **Information shares.** $I_k=\sum_{i\in k}\tilde T_i^2/\sum\tilde T_i^2$ for $k\in$ {randomized test, regular change, promotion, store-wide change, competitor-driven}. Because $\hat\theta=\sum_kI_k\hat\theta_k$ exactly (§1.1), the report can say, for example: "62% of the information comes from 4 regular price changes and 38% from promotions."
- **Tier rules [heuristic].**
  - Tier 1 if randomized price tests (§77) supply > 50% of the information.
  - Tier 3 if the event study (§3 E2/M5) passes its lead test (leads jointly insignificant at 10%, and |lead| < 0.5|ε̂|).
  - Tier 5 for DML with RS ≥ 0.1 and $K_{pc}$ ≥ 3.
  - Tier 6 for fixed effects only.
  - Tier 7 for prior-only or GBM-implied estimates.
- **Endogeneity placebo ("lead test").** Regress $\tilde Y_t$ on future price residuals $\tilde T_{t+h}$ for h = 7, 14 and 28. A significant coefficient means prices are set in anticipation of demand, which flags endogeneity and caps the tier at 6.
- **Robustness values.** They express "exogeneity" in measurable units: how strongly an omitted confounder would have to relate (as partial R²) to both price and demand to overturn the sign (RV₀) or the elastic/inelastic call (RV₋₁).

#### 5.3 Confidence as calibrated event probabilities
Given the reporting distribution $N(\hat\epsilon,s^2)$ (posterior from §3 E5; s already includes the tier bias variance):
- $p_{sign}=\Phi(|\hat\epsilon|/s)$
- $p_{close}=\Phi(0.5/s)-\Phi(-0.5/s)$, the probability of being within ±0.5
- $p_{elastic}=\Phi((-1-\hat\epsilon)/s)$, i.e. P(ε < −1)
- $EAE=s\sqrt{2/\pi}$
- $p_{decision}$ = P(profit at the recommended price > profit at the current price), by posterior simulation through §34–37

**Worked values [own computation]:**

| $\hat\epsilon$ | s | $p_{sign}$ | $p_{close}$ | EAE | P(ε < −1) |
|---|---|---|---|---|---|
| −1.5 | 0.3 | 1.000 | 0.904 | 0.239 | 0.952 |
| −1.5 | 0.7 | 0.984 | 0.525 | 0.559 | 0.762 |
| −0.6 | 0.5 | 0.885 | 0.683 | 0.399 | 0.212 |

**Headline "Confidence".** Define it as $p_{close}$ or $p_{decision}$, never as an unlabeled percentage. With a negative prior, $p_{sign}$ is dominated by the prior, so always show it next to the prior share B.

**Recalibration**
1. **Data.** Collect pairs (predicted probability $\pi$, outcome $o\in\{0,1\}$):
   - from semi-synthetic replicates (§3 E3), with $o=\mathbb 1\{|\epsilon^{true}-\hat\epsilon|\le0.5\}$ and so on;
   - from price-change backtests, with o = 1 if the observed $\Delta\log Q_e$ falls in its predicted 80% band.
2. **Fit.** `sklearn.isotonic.IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")`, and report $g(\pi)$.
3. **When D adds information beyond s.** Fit a monotone-constrained model (logistic regression on [logit π, log $K_{pc}$, RS, stockout share, promotion share, B], or LightGBM with `monotone_constraints`), then apply isotonic calibration on top.
4. **Expected absolute error.** Fit a quantile or linear model of $|\epsilon^{true}-\hat\epsilon|$ on (s, D) in the simulations.

#### 5.4 DQS becomes a pre-model predicted standard error
- **Definition.** $DQS_{se}=\hat\sigma_u/(\sqrt{n_{eff}}\cdot sd(\tilde T))$, computed from the design alone. Its units are elasticity standard errors and it is falsifiable: compare it with the realized se and |error| in simulation (the §1.2 Monte Carlo gave 0.845 against a predicted 0.849).
- **Labels [heuristic], tied to decisions:**
  - "insufficient" if $DQS_{se}$ > 1.0: the 95% interval is wider than ±2, so elastic and inelastic cannot be told apart;
  - "weak" for 0.5–1.0;
  - "good" for 0.25–0.5;
  - "excellent" below 0.25.
- **No double counting.** Missingness, stockouts and promotions enter only through $n_{eff}$, sd($\tilde T$) and the information shares.

#### 5.5 Validation protocol for any score
1. **Reliability.** On held-out (business × product × time) data, group predictions into 10 probability bins. Each bin's observed frequency must fall inside its binomial 95% band, and $ECE=\sum_b\frac{n_b}{N}|\bar o_b-\bar\pi_b|\le0.05$. Report the Brier score.
2. **Ranking.** The Spearman correlation between confidence and realized |error| (in simulation) or $|z_e|$ (in backtests) must be negative and significant (target ρ ≤ −0.3).
3. **Monotonicity unit tests.**
   - Adding a genuine independent price change must not lower confidence.
   - Converting days to stockout must not raise it.
   - Duplicating rows must not change $n_{eff}$-based outputs or the clustered SE.
4. **Stability.** Bootstrap the score. If the label flips in more than 20% of resamples, report it as unstable.
5. **Decision utility.** Realized pricing regret (§48) should fall monotonically across confidence deciles.

### Gaps
- I found no primary source on calibrating "data-quality → elasticity error" mappings. The design here is derived from general calibration and conformal practice.
- The isotonic, Brier and ECE references come from prior knowledge, not in-session verification.
- The thresholds (2% price change, 7/14-day rules, ±0.5 tolerance, RS < 0.1, DQS label cut-offs) are heuristics for the E3 simulation harness to tune.
- I did not fetch the "Long story short" paper itself; the robustness value's partial-R² interpretation follows the EconML docstring.

---

## 6. Drift monitoring and retraining triggers

### Takeaway
Monitor four things:
1. **Standardized forecast errors**, with a two-sided CUSUM (k = 0.5, h = 5; in-control ARL ≈ 465 days, 1σ shift detected in ≈ 10 days) or with Page–Hinkley rescaled to z-units.
2. **Rolling interval coverage** against binomial bands, plus the trajectory of ACI's $\alpha_t$.
3. **Elasticity stability** through Kalman innovations on price-change days, backtest $z_e$ and the Hausman window test.
4. **Feature drift** with PSI or KS at **sample-size-dependent critical values**. The fixed 0.1/0.25 PSI rules fire falsely 23–77% of the time at n = 50–100 per window.

On retraining: update the elasticity state on every price change, refit nuisance models weekly, and refit the heterogeneity models only on *confirmed* alarms, with false-alarm control across products.

### Cited Findings
- **PSI rules of thumb.** < 0.10 little shift, 0.10–0.25 moderate, > 0.25 significant. Yurdakul & Naranjo show that a properly rescaled PSI is asymptotically χ², with critical values that depend on both sample sizes and the number of bins; simulations support the approximation down to n, m ≈ 100. They argue for asymptotic critical values instead of rules of thumb. — [Journal of Risk Model Validation](https://www.risk.net/journal-of-risk-model-validation/7725371/statistical-properties-of-the-population-stability-index); [ResearchGate](https://www.researchgate.net/publication/347728512_Statistical_Properties_of_the_Population_Stability_Index); [review of population-stability tests, arXiv:2303.01227](https://arxiv.org/pdf/2303.01227); [PSI in medical data, PMC11844046](https://pmc.ncbi.nlm.nih.gov/articles/PMC11844046/)
- **Tabular CUSUM design.** k = 0.5 with h = 4 or 5 gives good ARL for a 1σ shift. h = 4 detects faster but has a shorter in-control ARL; Montgomery recommends k = σ/2 and h = 5σ. — [SigmaXL tabular CUSUM](https://www.sigmaxl.com/TabularCUSUM.html); [SigmaXL CUSUM ARL](https://www.sigmaxl.com/CUSUMARL.html); [Auburn CUSUM notes](https://www.eng.auburn.edu/~maghssa/INSY7330/Cusum-Control-Charts-M2013-Maghsoodloo.pdf); [SPC for Excel](https://www.spcforexcel.com/Downloads/pdf/Keeping-the-Process-on-Target-CUSUM-Charts.pdf); [SAS/QC CUSUM](https://support.sas.com/documentation/onlinedoc/qc/142/cusum.pdf)
- **river 0.26.1 drift detectors** (source code):
  - `PageHinkley(min_instances=30, delta=0.005, threshold=50.0, alpha=1-0.0001, mode="both")`. "This detector implements the CUSUM control chart … also supports the two-sided Page-Hinkley test"; alpha is "the forgetting factor, used to weight the observed value and the mean". It cites Page (1954), *Biometrika* 41:100–115.
  - `ADWIN(delta=0.002, clock=32, max_buckets=5, min_window_length=5, grace_period=10)`.
  - `KSWIN(alpha=0.005, window_size=100, stat_size=30)`.
  - `DDM(warm_start=30, warning_threshold=2.0, drift_threshold=3.0)`.
  - — [river 0.26.1](https://pypi.org/project/river/0.26.1/)
- **Recursive-residual coefficient-stability tests** in statsmodels (CUSUM, CUSUM of squares). — [statsmodels 0.15.0](https://pypi.org/project/statsmodels/0.15.0/)
- **MAPIE exchangeability-testing module** (source code): `FixedDatasetExchangeabilityTest`, `OnlineExchangeabilityTest`, `OnlineMartingaleTest`, permutation tests and `RiskMonitoring`. It includes a gamma-exponential mixture bound adapted from `confseq` (Howard et al. 2021, "Time-uniform, nonparametric, nonasymptotic confidence sequences", *Annals of Statistics* 49(2)). I did not check which monitors use that bound. — [MAPIE 1.5.0](https://pypi.org/project/mapie/1.5.0/)

### Inferences

#### 6.1 PSI with valid critical values
- **Formula.** $PSI=\sum_b(a_b-e_b)\ln(a_b/e_b)$, where $a_b$ is the current share and $e_b$ the reference share in bin b. It is symmetric in a and e.
- **Simulated behavior with no drift [own computation; B = 10 equal-probability bins, 20,000 simulations]:**

| n = m | P(PSI > 0.10) | P(PSI > 0.25) | Simulated q95 | $\chi^2_{9,0.95}(1/n+1/m)$ |
|---|---|---|---|---|
| 50 | 0.986 | 0.774 | 1.480 | 0.677 |
| 100 | 0.853 | 0.226 | 0.366 | 0.338 |
| 200 | 0.377 | 0.004 | 0.176 | 0.169 |
| 500 | 0.003 | 0.000 | 0.068 | 0.068 |
| 2000 | 0.000 | 0.000 | 0.017 | 0.017 |

- **Rule.**
  - For n, m ≥ 100, use $PSI_{crit}=\chi^2_{B-1,1-\alpha}(1/n+1/m)$ with α = 0.01.
  - For smaller windows (the n = 50 case breaks down because of empty bins), use B = 5 quantile bins from the reference, add-½ smoothing and permutation critical values.
  - Use natural categories for discrete variables (price levels, day types).
  - Choose the reference window (the training window of the deployed model) and the test window (for example the last 28 days) explicitly.

#### 6.2 KS test
- **Statistic.** $D=\sup|F_n-G_m|$, with 5% critical value ≈ $1.358\sqrt{(n+m)/(nm)}$. **[own computation]** That is 0.351 at n = m = 30, 0.192 at 100 and 0.101 at 365.
- **Ties.** With discrete values (prices, counts) KS is conservative. Use permutation p-values (`scipy.stats.permutation_test`), or PSI/χ² on categories.

#### 6.3 CUSUM on standardized forecast errors (primary demand monitor)
- **Statistics.** $z_t=(y_t-\hat y_t)/\hat\sigma_t$, with $\hat\sigma_t$ from the predictive distribution. $S^+_t=\max(0,S^+_{t-1}+z_t-k)$ and $S^-_t=\max(0,S^-_{t-1}-z_t-k)$. Alarm if $S^\pm>h$, then reset.
- **ARLs [own computation; Siegmund approximation $ARL\approx\frac{e^{-2\Delta b}+2\Delta b-1}{2\Delta^2}$ with $b=h+1.166$, checked by simulation]:**

| h (k = 0.5) | Two-sided in-control ARL | P(≥ 1 false alarm in 365 days) | One-sided ARL for a 1σ shift |
|---|---|---|---|
| 4 | 169 (simulated 165) | 0.89 | 8.3 |
| 5 | 469 (simulated 463) | 0.54 | 10.3 (simulated mean 10.4, median 9) |
| 6 | 1,286 | 0.25 | 12.3 |
| 7 | 3,510 | 0.10 | 14.3 |

- **Multiplicity.** 500 products monitored daily at h = 5 would give about 500 × 365 / 469 ≈ 390 false alarms per year. Therefore:
  - run the primary monitor at **business level**, on $\bar z_t\sqrt J$ across J products (z-scores of independent products; correlated products need a variance estimate);
  - use h = 7 for product-level charts;
  - require persistence, or apply Benjamini–Hochberg across products.
- **Miscalibration also alarms.** A miscalibrated $\hat\sigma_t$ triggers CUSUM. That is useful, because it is itself a reason to recalibrate.

#### 6.4 Page–Hinkley with river
- **Statistic [lit].** $m_T=\sum_{t\le T}(x_t-\bar x_t-\delta)$, $M_T=\min_{t\le T}m_t$; alarm if $m_T-M_T>\lambda$ (mirrored for downward changes).
- **Scaling.** river's default λ = 50 is in raw input units, so on z-scores it almost never fires. Rescale: feed z-scores, set δ ≈ half the shift to detect (for example 0.25), and choose λ by simulation to hit the target in-control ARL (as in 6.3).
- **Forgetting factor.** alpha < 1 discounts old data; the default $1-10^{-4}$ is nearly no forgetting.

#### 6.5 Interval-coverage monitor
- **Rolling band.** For 80% intervals over the last 60 days, alarm if coverage falls outside the band [0.700, 0.900]; for 90 days the band is [0.711, 0.878] **[own computation]**.
- **Monthly test.** Christoffersen conditional coverage at p < 0.01.
- **ACI.** Keep intervals calibrated between retrains, and treat a sustained climb of $\alpha_t$ away from α as drift.

#### 6.6 "Elasticity drift"
Elasticity is not observable, so monitor its footprints:
1. **Standardized Kalman innovations on price-change days**, $e_t/\sqrt{F_t}$, with a CUSUM over price-change events.
2. **Backtest standardized errors** $z_e=(\Delta\log Q_e-\widehat{\Delta\log Q}_e)/\sqrt{v_e+\Delta_e^2s^2}$ for each new event. Two consecutive |z_e| > 2, or a one-sample t-test of the last 5 events at p < 0.05, signal drift **[heuristic]**.
3. **Hausman test** comparing the recent window with the full history (§2 T2).
4. **BOCPD** P(change in the last 14 days) > 0.8 on residual levels.

#### 6.7 Retraining policy (proposal for the spec)

| Cadence or trigger | Action |
|---|---|
| Daily | Kalman update of $\beta_t$; ACI update of $\alpha_t$; CUSUM statistics; append new rows |
| Each qualifying price change | Re-estimate the pooled partially linear model and the posterior; append the backtest event; update $z_e$ monitors |
| Weekly | Refit nuisance models $\hat m,\hat e$ walk-forward; PSI/KS against the reference window |
| Monthly, or on a confirmed alarm | Refit LinearDML/forest, ensemble weights, confidence calibrator and the prior link |
| Confirmed alarm (any one) | Business-level CUSUM (h = 5) or product-level CUSUM (h = 7); coverage test rejects (p < 0.01); PSI above its χ² critical value (α = 0.01) on ≥ 2 key features for 2 consecutive weeks; elasticity-drift rule in 6.6; competitor entry or exit (an observed event, which is a covariate change, not latent drift) |
| Market definition (§51) | Recompute monthly. Version every derived feature with a `valid_from` date and never back-fill history with recalculated geography (§75 leakage) |

Every refit records the model version, data window and reference distribution (§73).

#### 6.8 Tests
1. **No shift.** The empirical yearly false-alarm rate of CUSUM(h = 5) is ≈ 0.54 per product-year (binomial band).
2. **1σ level shift at day 300.** Median detection delay ≈ 9 days and mean ≈ 10.
3. **PSI.** Symmetric in (a, e); 0 for identical inputs; finite with empty bins (smoothing); the false-alarm rate at the critical value is ≈ α under the null.
4. **Page–Hinkley.** Reproduce river's documented example (a stream that switches from {0, 1} to {4, …, 7} at index 1000 is detected at index 1006).
5. **Triggers.** Unit-test the trigger logic so that a single-day spike does not cause a full refit when persistence is required.

### Gaps
- I did not fetch the Yurdakul (2018) paper. The χ²(1/n + 1/m) scaling is my reconstruction, consistent with the search summary and confirmed by my simulation for n ≥ 100, but it is not quoted from the paper.
- The ARL values come from the Siegmund approximation and my simulation, cross-checked against search-summary statements for h = 4 and 5. The specific Montgomery table values were not read.
- I found no SMB-specific or retail-specific study of drift-trigger policies. The cadence table is a design proposal.
- The search budget ran out before I could check newer online-conformal monitors (conformal PID control, MAPIE's risk-monitoring API details) in depth.
