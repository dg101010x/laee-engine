# Causal identification of own-price elasticity (∂ln Q/∂ln P) from sparse single-SMB sales panels

Conventions used in these notes:
- `[src]` = verified this session by reading the package source code (shallow clones of the public GitHub repos: DoubleML main @ 6884b5f, 2026-09-23; EconML 0.17.0 main; pyfixest main, 2026-10-05; linearmodels main, 2026-09-21).
- `†` = well-established reference cited from memory. I did not re-open it this session: the shared web-search budget ran out partway through, and WebFetch was blocked for arxiv.org, docs.doubleml.org and university repositories. Check the DOI before quoting exact numbers.
- "Own simulation" = a scratch Monte Carlo I ran this session (numpy/scikit-learn). The DGPs are described inline so they can be rebuilt as regression tests. These results are inferences, not citations.
- Notation: j = product, t = day, D = ln P (posted regular price), Y = ln Q or Q (counts), X = controls, Δ = change in ln P, β or ε = own-price elasticity.

---

## 0. Audit of v1 spec §14–17, §29–31, §40, §44, §76–77, §86: what is wrong, outdated or underspecified

### Takeaway
v1 picks reasonable estimator families. As specified, though, it would give confidently wrong elasticities on SMB data. It treats TWFE as a safe baseline. It uses random K-fold DML with time-local and post-treatment features. It specifies no handling of zero sales, no variance estimator, no diagnostics and no minimum data. It ranks estimators where it should rank research designs, and it learns ensemble weights from predictive accuracy, which says nothing about causal validity.

### Cited Findings
- TWFE regressions estimate weighted sums of group×period treatment effects, and the weights can be negative. The coefficient can therefore be negative even when every effect is positive. The problem is more likely with non-binary treatments such as price. — [de Chaisemartin & D'Haultfœuille 2020, AER 110(9)](https://www.aeaweb.org/articles?id=10.1257%2Faer.20181169); [NBER w25904](https://www.nber.org/papers/w25904)
- With a continuous treatment, TWFE puts negative weights on the effects of lower-dose units because it uses them as controls. It also mixes level and slope parameters that parallel trends alone cannot separate. — [Callaway, Goodman-Bacon & Sant'Anna, NBER w32117](https://www.nber.org/system/files/working_papers/w32117/w32117.pdf); [arXiv 2107.02637 (rev. Dec 2025)](https://arxiv.org/abs/2107.02637)
- ATEs measured on log(1+Y) or arcsinh(Y) are not invariant to the units of Y when the treatment moves the extensive margin. — [Chen & Roth 2024, QJE 139(2):891–936](https://academic.oup.com/qje/article-abstract/139/2/891/7473710)
- For weakly dependent time-series and panel data, the recommended cross-fitting splits the sample into many (K≥10) time-adjacent folds. For each fold, both the fold and its immediate neighbours are left out of nuisance training ("neighbors-left-out"). — [Semenova, Goldman, Chernozhukov & Taddy 2023, Quantitative Economics 14:471–510](https://www.econometricsociety.org/publications/quantitative-economics/2023/05/01/Inference-on-Heterogeneous-Treatment-Effects-in-High-Dimensional-Dynamic-Panels-under-Weak-Dependence)
- DoubleML's built-in resampling is random. Without clusters it uses sklearn `RepeatedKFold` (or `RepeatedStratifiedKFold` for binary treatments) over observations. With `cluster_cols` it uses `KFold(shuffle=True)` over cluster identifiers. It never builds contiguous time blocks or gaps. Custom (train, test) partitions can be injected with `set_sample_splitting(all_smpls, all_smpls_cluster=None)`. — [src: doubleml/utils/resampling.py, double_ml_sampling_mixins.py](https://github.com/DoubleML/doubleml-for-py)
- EconML 0.17's final-stage OLS (`StatsModelsLinearRegression`) only supports `cov_type` in {nonrobust, HC0, HC1}. The `groups=` argument changes fold splitting (it upgrades to GroupKFold) but does **not** cluster the standard errors. — [src: econml/sklearn_extensions/linear_model.py; econml/_ortho_learner.py](https://github.com/py-why/EconML)
- A conventional 2SLS t-test at 1.96 has true 5% size only when the first-stage F exceeds 104.7. At F=10 the correct critical value is 3.43. Applying the correction to 57 AER papers made about half of their "significant" results insignificant. — [Lee, McCrary, Moreira & Porter 2022, AER 112(10)](https://www.nber.org/system/files/working_papers/w29124/revisions/w29124.rev0.pdf); [Princeton IRS note](https://irs.princeton.edu/news/2022/lee-aer-Oct-2022)
- A meta-analysis of 1,851 price elasticities from 81 studies finds a mean of −2.62 (SD 2.21), more elastic than the −1.76 mean in Tellis (1988). — [Bijmolt, van Heerde & Pieters 2005, JMR 42(2):141–156](https://research.rug.nl/en/publications/new-empirical-generalizations-on-the-determinants-of-price-elasti/)

### Inferences
Audit by section. Severity: **C** = will produce wrong elasticities or wrong confidence intervals; **H** = important gap; **M** = should fix.

**§14 Demand model.**
- (C) `ln Q` is undefined when a product sells zero units. Daily per-product SMB sales are dominated by zeros and small counts, and the spec says nothing about this. The default for daily data should be PPML (Poisson pseudo-ML) with fixed effects, keeping log-OLS for aggregated series with fewer than ~5% zeros. Own simulation (true β=−1.5, 400 days, mean 0.5 units/day, 60% zeros) gave these estimates: log(Q+1) −0.45; log(Q+1) with Q rescaled ×100 −2.36; dropping zeros −0.29; PPML −1.52 (details in §1).
- (H) The estimand is never defined. With heterogeneous elasticities, β is a weighted average (possibly with negative weights), not "the" elasticity. The spec should name its targets explicitly: (a) product-level ε_j, meaning the elasticity of E[Q] at the current price; (b) a portfolio summary weighted by revenue.
- (H) The fixed-effect structure is ambiguous for a single business. γ_t becomes a store-day effect shared by all products, so β is identified only from *relative* price changes across products on the same day. Menu-wide repricing, which is common in restaurants and kiosks, is absorbed almost entirely. The spec must state the γ granularity (e.g. category×week) and report the identifying variation left after the FE.
- (H) The price variable is undefined. A realized unit value (revenue/quantity) builds in division bias and reverse causation through quantity discounts (Deaton 1988†, [JSTOR](https://www.jstor.org/stable/1809142)). The treatment should be the posted regular price, with promotions handled as a separate treatment.
- (M) The model is static. Elasticities to temporary promotions include stockpiling and differ from regular-price elasticities (Hendel & Nevo 2006†, [doi](https://doi.org/10.1111/j.1468-0262.2006.00721.x)).
- (M) There is no variance specification (cluster, HAC or Driscoll–Kraay).

**§15.** (M) The endogeneity story is right but incomplete. Missing sources:
- stockouts that censor demand after price cuts;
- unit-value price bias;
- promotions bundled with displays;
- price-change *timing* chosen in response to past demand.

The section should also say that cost-driven price variation is the *good* variation and must not be controlled away. The direction of the bias is ambiguous: markups during demand peaks push β toward zero or positive, while clearance markdowns on slow stock can push either way.

**§16 DML.**
- (C) Plain random K-fold cross-fitting is used even though daily sales are serially correlated and the features are time-local. Own simulation (730 days, AR(1) errors with ρ=0.7, four exogenous step price changes, true β=−1.5):
  - DML with a time index among the features, random folds: mean −0.75, SD 1.97.
  - DML with lagged and rolling demand features, random folds: mean −0.73, 95% coverage 0.20.
- (C) No variance estimator is given. In the same simulation, iid standard errors covered the truth only 62% of the time, against 90% for Newey–West, even for OLS.
- (H) There is no overlap or residual-price-variation diagnostic. Σ(D − m̂(X))² can be close to zero (sticky prices plus time features), or *inflated* by nuisance extrapolation: own simulation, residual-to-raw price variance ratio 2.4–2.8 with blocked folds.
- (H) Repeated cross-fitting and its aggregation rule are unspecified.
- (M) The spec writes m(X)=E[Y|X] and g(X)=E[T|X]. DoubleML calls these ℓ (E[Y|X]) and m (E[D|X]), and uses g for E[Y−Dθ|X]. Mixing the two conventions invites bugs. [src: doubleml/plm/plr.py](https://github.com/DoubleML/doubleml-for-py)
- (M) It never says what DML estimates under heterogeneity: PLR returns a Var(D|X)-weighted average of θ(X).

**§17 Nuisance features.** (C) The list contains several bad controls:
- Same-day `promotion` is part of the treatment.
- Same-day `inventory` and `stockout` are post-treatment.
- `traffic` is a mediator.
- Same-day `competitor prices` may be responses to our price.
- `rolling demand` that includes day t leaks the outcome.
- `lagged demand` without lagged price. Own simulation: OLS with ln Q_{t−1} as a control returned −0.49 against a true −1.5, close to the analytical β(1−ρ) = −0.45, with 5% coverage.
- `time fixed effects` as tree features cannot be learned out-of-fold under blocked splits, and they leak under random splits.

Supplier cost and price variables are also wrongly treated: they are instruments, not confounders. Controlling for instrument-like variables removes good variation and amplifies any remaining bias (Pearl 2010†, [UCLA R-356](https://ftp.cs.ucla.edu/pub/stat_ser/r356.pdf); Wooldridge 2016†, [doi](https://doi.org/10.1016/j.rie.2016.01.001)).
- (M) Demand shifters that matter in emerging markets are missing: payday/month-end, Ramadan and Eid, school-fee periods, harvest cycles, power outages (load-shedding), and public-holiday shifts.
- (M) "Benchmark them" gives no criterion. The criterion must be out-of-fold nuisance loss under time-aware CV (DoubleML exposes `nuisance_loss` and `evaluate_learners` [src]). Never select learners on θ̂ or its SE. Prefer stacking (see §2).

**§29–30 Competitor entry/exit DiD.**
- (H) Wrong estimand. Competitor entry shifts demand directly, so it violates the exclusion restriction and is **not** an instrument for own price. It identifies effects of competition, not own-price elasticity.
- (H) The canonical TWFE `Treated×Post` estimator is biased with staggered entries and heterogeneous effects.
- (H) A single business has no control market inside its own data, so the design needs multi-tenant donors.
- (M) Entry timing is endogenous (entrants pick growing markets), and announcement or construction can cause anticipation effects.
- (M) There are no event-study leads or lags, no clustering, and nothing for inference with few treated units.

**§31 Cross-price.** (H) Competitor prices respond to the same local demand shocks, so cross-price elasticities from OLS are biased upward unless instrumented or randomized. "Sufficient variation" is undefined. Within-store substitution also breaks SUTVA in within-store DiD designs that use other products as controls.

**§40 Ensemble.** (C) Learning weights from out-of-sample *predictive* performance does not measure causal validity: a demand model can forecast well and still have a biased price coefficient. Model D (a gradient-boosting demand model's implied elasticity) is purely correlational (tier 7). The ensemble also mixes different estimands (TWFE's weighted average, DML's variance-weighted average, a forest ATE) and ignores the covariance between estimators fitted on the same data, so the ensemble CI is wrong. A replacement is in §6.

**§44 IDS.** (H) No formula, no thresholds. Some inputs ("exogeneity", "controlQuality") are not observable, and "competitiveVariation" has nothing to do with own-price identification. Replace with gate-based diagnostics (§6).

**§76 Hierarchy.** (H) It ranks estimators, not designs. DML and FE rest on the same identifying assumption (selection on observables, with FE), so putting DML a tier above FE is a category error. Whether IV beats DiD depends on diagnostics. The list omits the most common SMB design (event study / RD-in-time around discrete regular price changes), synthetic control and switchbacks. There are no gates and no "unidentified → prior only" tier.

**§77 Experimentation.** (H) The 10%/90% customer-level split has three problems:
- It is infeasible with posted shelf prices.
- It carries legal and ethical risk (personalised-pricing disclosure rules, fairness backlash).
- It is statistically inefficient: Var ∝ 1/n₁ + 1/n₂, so 10/90 needs 2.78× the sample of 50/50 (computed this session).

The section also has no power analysis, no carryover or washout handling and no stopping rules.

**§86 Recommended V1.** (H) TWFE baseline + DML(features) + CausalForest, ensembled on out-of-sample fit, should be replaced by a data-condition router (§6):
- PPML event-study/imputation around regular price changes by default;
- PLR-DML with time-aware folds as a robustness check;
- IV when cost data exist;
- hierarchical shrinkage with a tier-dependent bias variance.

Causal forests need far more within-business price variation than SMBs have (§1–2). Defer them.

**Adjacent sections (outside the assigned list but affecting identification).**
- §20: 30-day rolling windows often contain no price change at all, so the elasticity is unidentified within the window.
- §39: conformal prediction produces prediction intervals for outcomes, not confidence intervals for a causal parameter.
- §62–64: promotions and stockouts must enter the *treatment and sample definition*, not the control set.

### Gaps
- I could not open docs.doubleml.org or arxiv.org (blocked) and verified DoubleML behaviour from source instead. Release notes for when the aggregation rule changed were not available.
- I found no empirical literature specific to SMBs in African or other emerging markets on elasticity identification. The web-search budget ran out before I could search for it.

---

## 1. Two-way fixed effects log-log demand: when is the TWFE elasticity biased, what are the remedies, and how should zero sales be handled?

### Takeaway
TWFE log-log OLS is only a safe baseline when elasticities are homogeneous, price changes are not timed on product-specific demand trends, and zeros are negligible. With heterogeneous elasticities, staggered or repeated continuous price changes and many zeros (the SMB norm), it returns a non-convex weighted average on a scale that is undefined at zero.

Recommended approach:
- Default to PPML with product fixed effects.
- Use heterogeneity-robust event-study or imputation estimators (Borusyak–Jaravel–Spiess, Callaway–Sant'Anna, Sun–Abraham, de Chaisemartin–D'Haultfœuille) around discrete regular-price changes.
- Use the de Chaisemartin et al. "switchers vs stayers" estimators for repeated continuous changes.
- Keep TWFE-OLS as a diagnostic, and always report its weights.

### Cited Findings
**TWFE with heterogeneous effects**
- TWFE = weighted sum of group×period ATEs, with weights that may be negative. Software: `twowayfeweights` (Stata), `TwoWayFEWeights` (R; CRAN July 2025). — [dCDH 2020 AER](https://www.aeaweb.org/articles?id=10.1257%2Faer.20181169); [TwoWayFEWeights](https://cran.r-project.org/web/packages/TwoWayFEWeights/TwoWayFEWeights.pdf)
- Goodman-Bacon decomposition (binary staggered adoption): the TWFE DiD equals a variance-weighted average of all 2×2 DiDs. These include "forbidden" comparisons that use already-treated units as controls, which are biased when effects change over time. Units treated in the middle of the panel get the most weight. — Goodman-Bacon 2021†, J. Econometrics 225(2):254–277, [doi](https://doi.org/10.1016/j.jeconom.2021.03.014)

**Remedies for staggered binary events**
- Callaway & Sant'Anna build cohort×time ATT(g,t) against never-treated or not-yet-treated controls, with doubly robust covariate adjustment, then aggregate to event time. — CS 2021†, J. Econometrics 225(2):200–230, [doi](https://doi.org/10.1016/j.jeconom.2020.12.001)
- Sun & Abraham show TWFE event-study leads and lags are contaminated by effects from other periods. They propose an interaction-weighted estimator (saturated cohort×relative-time dummies, weighted by cohort shares). — SA 2021†, J. Econometrics 225(2):175–199, [doi](https://doi.org/10.1016/j.jeconom.2020.09.006)
- BJS show conventional regression event-study estimators are not unbiased for the relevant estimands without strong homogeneity. Their imputation estimator fits unit and time FE on untreated observations only, imputes Y(0) for treated observations and averages the differences. Stata: `did_imputation`. — [Borusyak, Jaravel & Spiess 2024, REStud 91(6):3253–3285](https://academic.oup.com/restud/article/91/6/3253/7601390)
- A Python BJS imputation estimator exists in the `diff-diff` package. — [diff-diff docs v2.9.1](https://diff-diff.readthedocs.io/en/v2.9.1/api/imputation.html)
- pyfixest (2026-10) provides `event_study(estimator="twfe"|"did2s"|"saturated")`, `did2s` (Gardner's two-stage estimator; its point estimates coincide with BJS imputation†), `lpdid` (local-projections DiD), `fepois` (PPML with high-dimensional FE and a `separation_check` option), `wildboottest`, `ritest` (randomization inference) and `ccv`. It has **no** Goodman-Bacon decomposition (the `decompose` method is a Gelbach decomposition). — [src: pyfixest/did/estimation.py; estimation/models/feols_.py; estimation/api/fepois.py](https://github.com/py-econometrics/pyfixest)
- DoubleML (main branch, Sept 2026, pre-0.12) has `DoubleMLDIDMulti` (multi-period/staggered DiD with ML nuisances) with `control_group` in {never_treated, not_yet_treated} and `anticipation_periods`. — [src: doubleml/did/did_multi.py](https://github.com/DoubleML/doubleml-for-py)

**Remedies for continuous or repeated treatment (prices)**
- With a continuous dose, parallel trends identifies ATT-type level effects. Comparing across doses (slopes, i.e. elasticities) needs a stronger "no selection on gains" type assumption. TWFE estimands have several interpretations, each with important limitations. — [CGBS NBER w32117](https://www.nber.org/system/files/working_papers/w32117/w32117.pdf); [Callaway five-minute summary](https://bcallaway11.github.io/posts/five-minute-did-continuous-treatment)
- Where treatment is continuous in every period (taxes, tariffs, *prices*), units are split into **switchers** (treatment changed between t−1 and t) and **stayers**. The switchers' slopes are identified nonparametrically by DiD comparisons with stayers that share the same baseline treatment. Two weighted averages of slopes get doubly robust, √n, semiparametrically efficient estimators, with an IV extension, illustrated on gasoline taxes. Stata/R: `did_multiplegt_stat`. — [dCDH, Pasquier & Vazquez-Bare, SSRN 4011782](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4011782); [arXiv 2201.06898](https://arxiv.org/pdf/2201.06898); [did_multiplegt_stat](https://github.com/Credible-Answers/did_multiplegt_stat)
- Companion estimators exist for continuous treatments with **no stayers**, the case where every unit's price changes. — [dCDH et al., AEA P&P 2024](https://www.aeaweb.org/articles?id=10.1257%2Fpandp.20241049)
- Non-binary, non-absorbing treatments with dynamic effects: "DiD estimators of intertemporal treatment effects" (`did_multiplegt_dyn`). — [dCDH, HAL wp hal-03873903](https://ideas.repec.org/p/hal/wpaper/hal-03873903.html)
- Survey of the modern DiD toolkit: Roth, Sant'Anna, Bilinski & Poe 2023†, J. Econometrics 235(2):2218–2244. — [arXiv 2201.01194](https://arxiv.org/pdf/2201.01194)

**Zero sales**
- log(1+Y) and arcsinh(Y) ATEs depend on the units of Y whenever the treatment affects the extensive margin. Alternatives: (i) report a normalized ATE in levels, θ/E[Y(0)], estimable by Poisson regression; (ii) report extensive and intensive margins separately; (iii) value the extensive margin explicitly. — [Chen & Roth 2024 QJE](https://academic.oup.com/qje/article-abstract/139/2/891/7473710); [arXiv 2212.06080](https://arxiv.org/pdf/2212.06080)
- Log-linearized OLS is inconsistent for the elasticity of E[Y|X] under heteroskedasticity. PPML is consistent if the conditional mean is correctly specified and handles zeros naturally. — Santos Silva & Tenreyro 2006†, REStat 88(4):641–658, [doi](https://doi.org/10.1162/rest.88.4.641)
- Poisson with unit fixed effects has no incidental-parameter problem (conditional-likelihood equivalence). — Hausman, Hall & Griliches 1984†, [doi](https://doi.org/10.2307/1911191). PPML with high-dimensional FE must detect and drop separated observations (FE cells where Q is always 0). — Correia, Guimarães & Zylkin 2020†, [doi](https://doi.org/10.1177/1536867X20909691)
- Nonlinear (Poisson) DiD with staggered timing via an "extended TWFE" with cohort×time interactions. — Wooldridge 2023†, Econometrics Journal 26(3):C31–C66, [doi](https://doi.org/10.1093/ectj/utad016)

### Inferences

#### Estimator card 1A — TWFE log-log OLS (diagnostic only, never a "safe baseline")
- **Formula** (balanced panel, by FWL): `β̂ = Σ_jt D̈_jt Y_jt / Σ_jt D̈_jt²`, where `D̈ = D − D̄_j· − D̄_·t + D̄··` is two-way demeaned ln P. For unbalanced panels, use iterative alternating projections (pyfixest).
- **Weight decomposition.** If the noise-free outcome is `Y = α_j + γ_t + β_jt·D_jt`, then `β̂ = Σ w_jt β_jt` with `w_jt = D̈_jt·D_jt / Σ D̈²` and Σw = 1. Report Σ_{w<0} w, the share of negative weights, and product-level weights `w_j = Σ_t w_jt`.
  - *Own check:* 12 products × 40 days with sticky prices and β_j ~ U(−3, −0.5). The identity held exactly (β̂ = Σwβ = −1.758). 25% of cell weights were negative (summing to −0.068), and the TWFE value differed from the simple mean of β_j (−1.929).
- **Variance:** cluster by product (CR1) when there are ≥ 30–40 products; Driscoll–Kraay by day for common shocks; wild cluster bootstrap with few products (pyfixest `wildboottest`).
- **When acceptable:** ≥ 2 regular price changes per product; zero share < 5% (aggregate to weekly if needed); price changes not menu-wide; negative-weight share < 10%.
- **Failure modes:**
  - heterogeneous β with staggered changes (wrong magnitude, even wrong sign);
  - menu-wide repricing absorbed by γ_t;
  - within-store substitution contaminating "control" products;
  - zeros;
  - unit-value price.
- **Test that catches a wrong implementation:** generate the noise-free heterogeneous panel above and assert `β̂_TWFE == Σ w_jt β_j` to 1e-10 and `Σw == 1`. Then add a staggered design in which effects grow with time since the change, and assert that the robust estimators (1C) recover the true average while TWFE does not.

#### Estimator card 1B — PPML with product FE (default static model for daily counts)
- **Model:** `E[Q_jt | ·] = exp(α_j + γ_{c(j),w(t)} + β·ln P_jt + X_jt'δ)`. Here γ is category×week (or week only), and X holds recurring calendar terms (day-of-week, holidays, paydays), weather and *lagged* competitor prices. β is the elasticity of expected quantity.
- **Score/estimator:** solve `Σ (Q − μ̂) x = 0` by IRLS or the pyfixest `fepois` HDFE routine.
- **Variance:** sandwich `V = H⁻¹ (Σ_g s_g s_gᵀ) H⁻¹`, with `H = Σ μ̂ x xᵀ` and `s_g = Σ_{i∈g} (Q_i − μ̂_i) x_i`. Cluster by product, and use Driscoll–Kraay by day if common shocks matter.
  - pyfixest NW/DK requires an explicit `lag` (there is no default), and it treats consecutive time ids as one step apart even across date gaps [src]. Reindex onto a complete daily calendar, or use a date-distance-aware kernel, before applying DK to gappy SMB data.
- **Zero taxonomy (must be built in ETL):**
  1. Structural zeros (store closed, product not listed or not yet launched): drop.
  2. Censored zeros or low sales (stockout, using *opening* inventory or stockout flags): drop the day, or use a censored model. Flags measured at end of day are post-treatment.
  3. True zeros: keep them in PPML.
- **Own simulation** (Poisson demand, β = −1.5, 400 days, four price levels ±20%):

| mean units/day at reference price | share of zero days | log(1+Q) OLS | log(1+100·Q) OLS | log Q, zeros dropped | PPML |
|---|---|---|---|---|---|
| 0.5 | 0.60 | −0.45 | −2.36 | −0.29 | −1.52 |
| 2 | 0.14 | −1.05 | −2.94 | −1.02 | −1.50 |
| 10 | 0.00 | −1.43 | −1.60 | −1.59 | −1.50 |

  Even without zeros (μ=10), log-OLS is off (−1.59) because of retransformation and heteroskedasticity, which is the Santos Silva–Tenreyro point.
- **Minimum data:** ≥ 2 distinct price levels per product with ≥ 14 days at each; ≥ 30 non-zero days per product.
- **Failure modes:** separation (all-zero FE cells; drop them); misspecified mean (check with a RESET-type test on μ̂ powers); overdispersion (fine for consistency, but use sandwich SEs); incidental-parameter downward SE bias when many FE are estimated from few observations (use small-sample cluster corrections or the wild score bootstrap).
- **Tests:** (i) simulate Poisson data with known β and assert the PPML estimate is within 2 SE of truth over 200 draws, with coverage of 0.93–0.97; (ii) scale invariance: multiplying Q by 100 must leave β̂ unchanged (only α shifts) — the log(1+Q) estimator fails this test by construction; (iii) a lint rule that log1p or arcsinh outcomes are never used for elasticities.

#### Estimator card 1C — heterogeneity-robust event study around discrete regular-price changes
- **Event definition (ETL):** a change in the modal daily *regular* price with |Δ ln P| ≥ 0.03 that persists ≥ 21 days. It must not coincide (±7 days) with a promotion, stockout, supplier or pack-size change, or a new product listing. Changes within 14 days of each other are merged into one event.
- **Binary staggered version** (first change per product as an absorbing treatment): estimate ATT by event time with BJS imputation (`did2s` / `diff-diff`), CS (`DoubleMLDIDMulti` with an ML nuisance, or a CS port) or SA (pyfixest `saturated`). Use outcomes in logs for weekly aggregates with few zeros, or Wooldridge's Poisson ETWFE for counts.
- **Elasticity (Wald-DiD ratio):** `ε̂ = Σ_j ω_j·ATT_j / Σ_j ω_j·Δ_j`, the average effect on log demand divided by the average log price change. Use ω_j ∝ number of post-period observations. For per-event slopes, average `τ̂_j/Δ_j` with weights ∝ Δ_j²; this weighting is efficient under homogeneity and is the analogue of dCDH's |Δ|-weighted slope average.
- **Repeated continuous changes:** use the dCDH–Pasquier–Vazquez-Bare switcher/stayer slopes, `slope = E[ΔY | switcher] − E[ΔY | stayer, same baseline D]`, normalized by E[ΔD]. Products differ in absolute price level, so define treatment as log price relative to the product's own reference price; "same baseline" then means similar relative position (my inference, not in the paper).
- **Variance:** BJS conservative clustered variance by product; CS multiplier bootstrap; with fewer than 10 events, use randomization inference over placebo dates (see §4).
- **Minimum data:** ≥ 3 qualifying events (across products and/or time); ≥ 28 days pre and post; a comparison set of not-yet- or never-changed products that are **not substitutes** (different category) or, better, the same SKU at other tenants.
- **Failure modes:**
  - *Menu-wide repricing.* Time FE absorb the common change, leaving only dose differences across products. That identifies slopes only under CGBS's stronger assumption (no selection on gains: owners raised prices most where demand was least elastic). I would expect this assumption to fail often because owners price on gut feel about elasticity (inference).
  - *Substitution to control products.* Controls rise when the treated product's price rises, which overstates |ε|.
  - Anticipation and stockpiling.
  - Concurrent unobserved changes.
- **Test:** simulate staggered price increases where effects grow with event time and differ by cohort. Assert the robust estimator recovers the true weighted ATT within Monte Carlo error, and assert that TWFE's bias has the sign predicted by its weights (a regression test for both the robust estimator and the weight calculator).

#### Practical rules
- Never interpret γ_t-absorbed variation as identified. Report `share_identifying = Var(D̈)/Var(D)`: the fraction of log-price variance that survives the FE.
- For single-business panels, use category×week FE rather than day FE unless there are ≥ 20 products per category. Day FE with few products eat most of the variation and amplify noise.

### Gaps
- I could not verify the current maintenance status of Python ports of CS (`csdid`, `differences`) or of `did_multiplegt_dyn` / `did_multiplegt_stat`. No Python Goodman-Bacon decomposition was confirmed; it is easy to implement from the 2×2 formulas.
- The dCDH robustness ratio (the minimal SD of effects compatible with a sign reversal, `|β̂|/σ(w)`) is from memory of dCDH 2020. Verify before implementing.
- Titles seen in search results but not read: "Interpreting Event-Studies from Recent Difference-in-Differences Methods" ([arXiv 2401.12309](https://arxiv.org/pdf/2401.12309)) and "(Empirical) Bayes Approaches to Parallel Trends" ([arXiv 2404.11839](https://arxiv.org/html/2404.11839v1)).

---

## 2. Double/debiased ML for price elasticity: model choice, orthogonal scores, cross-fitting with serial correlation, aggregation, variance, small samples, learners, and controls

### Takeaway
DML is selection-on-observables with flexible controls. It cannot create identifying variation. For SMB daily data it is credible only when all of the following hold:
1. partialling-out PLR on posted log price;
2. recurring (not time-local) features;
3. no post-treatment controls;
4. time-aware or grouped cross-fitting, repeated over splits;
5. HAC, Driscoll–Kraay or cluster variance, and randomization inference when there are few price changes;
6. a hard gate on residual price variation.

With only a handful of price regimes, flexible learners overfit spurious price–season coincidences and DML standard errors under-cover badly. Use regularized linear nuisances or plain PPML/OLS with a pre-specified seasonal basis instead.

### Cited Findings
- DML framework: Neyman-orthogonal scores plus cross-fitting give √n inference when nuisance estimators converge faster than n^{−1/4}. — Chernozhukov et al. 2018†, Econometrics Journal 21(1):C1–C68, [doi](https://doi.org/10.1111/ectj.12097)
- **DoubleML internals [src]:**
  - `DoubleMLPLR(..., n_folds=5, n_rep=1, score="partialling out")`. The `dml_procedure` option is gone, so pooled-score "DML2" is the only procedure.
  - The variance is computed as `J = mean(psi_deriv)`, `sigma² = mean(psi²)/J²` (`_var_est`).
  - Repeated cross-fitting is aggregated by `θ = median_s θ_s`, `ub = median_s(θ_s + 1.96·se_s)`, `se = (ub − θ)/1.96`. P-values and CI bounds are medians over repetitions.

  — [src: doubleml/utils/_estimation.py `_aggregate_coefs_and_ses`, double_ml_framework.py](https://github.com/DoubleML/doubleml-for-py)
- The original DML paper's aggregation rule is `θ̃ = median_s θ̂_s` and `σ̃² = median_s[σ̂_s² + (θ̂_s − θ̃)²]`, which explicitly adds split-to-split dispersion. — Chernozhukov et al. 2018†, [doi](https://doi.org/10.1111/ectj.12097)
- **DoubleML clustering [src]:**
  - `DoubleMLClusterResampling` draws `KFold(shuffle=True)` folds over unique cluster ids for each cluster variable and takes the Cartesian product of folds across cluster variables (two-way → K² cells).
  - `DoubleMLClusterData` is deprecated (removal in 0.12.0); use `DoubleMLData(..., cluster_cols=[...])`.

  — [src: utils/resampling.py; data/__init__.py](https://github.com/DoubleML/doubleml-for-py)
- Multiway cluster-robust DML: a multiway cross-fitting algorithm with a multiway cluster-robust standard-error formula. In a demand application with market-share data, two-way cluster-robust SEs were larger than non-robust ones. — [Chiang, Kato, Ma & Sasaki, JBES (2022), arXiv 1909.03489](https://arxiv.org/abs/1909.03489); [DoubleML example](https://docs.doubleml.org/stable/examples/py_double_ml_multiway_cluster.html)
- **Panel DML (Clarke & Polselli):** procedures for static panels with fixed effects. They recommend **first-differencing plus an ensemble learning strategy**. R package `xtdml`. — [Clarke & Polselli 2026, Econometrics Journal 29(1):69–86](https://ideas.repec.org/a/oup/emjrnl/v29y2026i1p69-86..html); [xtdml](https://github.com/POLSEAN/XTDML)
- DoubleML implements Clarke–Polselli as `DoubleMLPLPR(obj, ml_l, ml_m, approach=...)` with approaches `cre_general`, `cre_normal` (Mundlak unit means of X), `fd_exact` (default) and `wg_approx` (within-group). `DoubleMLPanelData(static_panel=True)` sets `cluster_cols = id_col`, so it clusters by unit and splits folds by unit. — [src: doubleml/plm/plpr.py; data/panel_data.py](https://github.com/DoubleML/doubleml-for-py)
- **Neighbors-left-out cross-fitting:** K ≥ 10 time-adjacent folds; for each, hold out the fold and its neighbours. Theory via Strassen coupling under weak dependence. — [Semenova et al. 2023 QE](https://www.econometricsociety.org/publications/quantitative-economics/2023/05/01/Inference-on-Heterogeneous-Treatment-Effects-in-High-Dimensional-Dynamic-Panels-under-Weak-Dependence); [arXiv 1712.09988](https://arxiv.org/abs/1712.09988)
- **Stacking:** DML combined with stacking is more robust to partially unknown functional forms than a single pre-selected learner. *Short-stacking* reuses the cross-fitted predictions to choose stacking weights (much cheaper). *Pooled stacking* forces common weights across folds. — [Ahrens, Hansen, Schaffer & Wiemann 2025, JAE 40(3):249–269](https://ideas.repec.org/a/wly/japmet/v40y2025i3p249-269.html); [ddml short-stacking note](https://thomaswiemann.com/ddml/articles/stacking.html)
- **EconML [src]:**
  - `fit(..., groups=)` keeps groups together; with integer `cv` it becomes GroupKFold.
  - `cv` also accepts an iterable of (train, test) splits, so blocked splits can be injected.
  - `residuals_` (needs `cache_values=True`) returns (Y_res, T_res, X, W), with the docstring warning that samples are **not guaranteed to be in input order**.
  - Final-stage covariance supports only nonrobust/HC0/HC1.

  — [src: econml/_ortho_learner.py; dml/_rlearner.py; sklearn_extensions/linear_model.py](https://github.com/py-why/EconML)
- **DoubleML extras [src]:** `psi`, `psi_deriv`, `predictions`, `nuisance_loss` and `evaluate_learners(metric=rmse)` are all exposed, so HAC/DK variances can be computed from `psi`. `DoubleMLAPOS` handles multi-valued discrete treatments (average potential outcomes per price level). Its legacy `trimming_threshold` default is 1e-2 (deprecated; propensity processing now goes through a config object). `DoubleMLLPLR` (partially logistic) exists. — [src](https://github.com/DoubleML/doubleml-for-py)
- **Cross-validation with dependent data:** standard K-fold CV is valid for purely autoregressive models whose errors are uncorrelated (Bergmeir, Hyndman & Koo 2018†, [doi](https://doi.org/10.1016/j.csda.2017.11.003)). Otherwise, hv-block CV drops a gap of h observations around each validation block (Racine 2000†, [doi](https://doi.org/10.1016/S0304-4076(00)00030-0)).
- **HAR inference:** Newey–West kernel HAC (Newey & West 1987†, [doi](https://doi.org/10.2307/1913610)); Driscoll–Kraay for panels with cross-sectional dependence (1998†, [doi](https://doi.org/10.1162/003465398557825)). Lazarus, Lewis, Stock & Watson 2018† (JBES 36(4), [doi](https://doi.org/10.1080/07350015.2018.1506926)) recommend either NW with bandwidth S ≈ 1.3·T^{1/2} and fixed-b critical values, or equal-weighted cosine (EWC) with ν ≈ 0.4·T^{2/3} and t_ν critical values. These are recalled from memory, not re-verified.
- **Continuous-treatment average derivatives:** debiasing via automatically learned Riesz representers. — Chernozhukov, Newey & Singh 2022†, Econometrica 90(3), [doi](https://doi.org/10.3982/ECTA18515)
- **Recent (2026) titles seen in search results, contents not reviewed:** "Cross-Fitting-Free Debiased ML with Multiway Dependence" ([RePEc/arXiv 2602.11333](https://ideas.repec.org/p/arx/papers/2602.11333.html)); "Semiparametric Inference for Dynamic Causal Effects from Observational Time Series" ([arXiv 2609.27531](https://arxiv.org/html/2609.27531)); "Double Machine Learning meets Panel Data – Promises, Pitfalls, and Potential Solutions" ([arXiv 2409.01266](https://arxiv.org/html/2409.01266v1), 2024); "What, Why, and How: An Empiricist's Guide to DML" ([ISR](https://pubsonline.informs.org/doi/10.1287/isre.2024.0888)).

### Inferences

#### Estimator card 2A — PLR, partialling-out (primary DML estimator for continuous log price)
```
Model:   Y = θ·D + g0(X) + ζ,  E[ζ|D,X]=0;   D = m0(X) + V,  E[V|X]=0;   ℓ0(X)=E[Y|X]
Score:   ψ(W;θ,ℓ,m) = (Y − ℓ(X) − θ(D − m(X))) · (D − m(X))          (Neyman-orthogonal in ℓ, m)
Cross-fit (DML2): Ṽ_i = D_i − m̂_{−k(i)}(X_i),  Ũ_i = Y_i − ℓ̂_{−k(i)}(X_i)
θ̂ = Σ_i Ṽ_i Ũ_i / Σ_i Ṽ_i²
Ĵ = −(1/n) Σ Ṽ_i² ;  ψ̂_i = (Ũ_i − θ̂Ṽ_i)·Ṽ_i
HC:        Var(θ̂) = (1/n) · [(1/n)Σψ̂_i²] / Ĵ²
Cluster:   Var(θ̂) = [G/(G−1)] · Σ_g (Σ_{i∈g} ψ̂_i)² / (n Ĵ)²
Driscoll–Kraay (panel; h_t = Σ_j ψ̂_jt over products on day t):
           Ω̂ = Σ_t h_t² + 2 Σ_{L=1..S} (1 − L/(S+1)) Σ_t h_t h_{t−L};   Var(θ̂) = Ω̂ / (n Ĵ)²
NW (single product series): same with h_t = ψ̂_t.
EWC alternative: Λ_q = √(2/T) Σ_t h_t cos(πq(t−½)/T), q=1..ν;  LRV̂ = (1/ν) Σ_q Λ_q²;  Var(θ̂) = T·LRV̂ / (n Ĵ)²;  use t_ν critical values (ν ≈ 0.4·T^{2/3}†).
Repeated splits s=1..S: θ̃ = median θ̂_s;  σ̃² = median[σ̂_s² + (θ̂_s − θ̃)²]   (2018 rule; more conservative than DoubleML's median-CI-bound rule)
```
- **Estimand under heterogeneity.** If Y = θ(X)·D + g(X) + ζ, then `θ_PLR = E[Var(D|X)·θ(X)] / E[Var(D|X)]`. This is a conditional-variance-weighted average that over-weights products and periods with more price variation, not a revenue-weighted elasticity. It follows directly from the partialling-out formula. To get a revenue-weighted or product-level ε, fit per-product or interaction models (CATE on product features) and reweight.
- **Interactive model (IRM)** is for *binary* treatments, e.g. "post-change vs pre-change" or "high vs low arm". ATE score: `ψ = g(1,X) − g(0,X) + D(Y−g(1,X))/m(X) − (1−D)(Y−g(0,X))/(1−m(X)) − θ`, with elasticity = θ/Δ. Event designs ordered in time violate overlap, because "post" is a deterministic function of the date. Use IRM only for randomized arms or switchbacks.
- **Few discrete price points** (SMBs often cycle among two or three prices): `DoubleMLAPOS` gives E[Y(p_k)] for each level, and arc elasticities between adjacent levels are `(ln E[Q(p_b)] − ln E[Q(p_a)]) / (ln p_b − ln p_a)`. Run it on Y in levels so zeros are kept.
- **Count outcomes in DML.** The textbook PLR on ln Q inherits the zero problem. Options:
  - (a) PLR on log weekly sales when weekly zeros are under 5%.
  - (b) A multiplicative orthogonal moment. Under `E[Q|D,X] = exp(θD)·h0(X)`, `E[(Q·e^{−θD} − h_θ(X))·(D − m(X))] = 0` with `h_θ(X) = E[Q e^{−θD}|X]`. The moment is Neyman-orthogonal in (h, m) because E[D − m|X] = 0. It needs an outer solve over θ, refitting h_θ each time. *This is my own derivation, not from a verified source; validate it by simulation before use.*
  - (c) Use PPML-FE (1B) as the primary estimator and DML as a robustness check.

#### Cross-fitting under serial correlation (defaults)

**Single product series.** Use contiguous blocks:
- K = 5 blocks (K = 3 if T < 300), each block ≥ 60 days.
- Remove a gap g on both sides of the test block from training: `g = max(14, L_feat + 7, ⌈ln 0.05 / ln ρ̂⌉)`. Here L_feat is the longest feature look-back and ρ̂ is the AR(1) coefficient of residualized log sales. For ρ̂ = 0.7, the autocorrelation term is 9 days, so the 14-day floor binds.
- Alternative: neighbors-left-out with K = 10, dropping blocks k±1.

**Multi-product, single business.** Folds must be **date blocks shared by all products**, because same-day store shocks are common across products. Product-only folds, which is DoubleMLPLPR's default via `cluster_cols = id`, leave common daily shocks straddling train and test.
- DoubleML-native option: `DoubleMLData(..., cluster_cols=["block14"])`, where block14 = 14-day period id. This gives cluster-based folds plus block-clustered SEs.
- Two-way option: `cluster_cols = ["product_id", "block14"]` (Chiang et al., K² cells). Needs ≥ 20 products and ≥ 30 blocks.

**Repetitions.** Use S = 10 with random block offsets (rotate the boundaries by U(0, block length)); deterministic blocks would produce identical repetitions. Aggregate with the 2018 rule and log both rules.

**Own simulation on the leakage and small-sample questions.** DGP: 730 days; log demand = 2 − 1.5·ln P + 0.3·sin(annual) + day-of-week effect + AR(1) error (ρ = 0.7, σ_e = 0.3); four exogenous step price changes of 5–15%; learners are HistGradientBoosting (150 iterations, 15 leaves) or RidgeCV.

| estimator (features; folds) | mean θ̂ | SD | 95% coverage, iid SE | 95% coverage, NW-HAC (S = 35) |
|---|---|---|---|---|
| OLS (Fourier + day-of-week) | −1.57 | 0.59 | 0.62 | 0.90 |
| OLS + ln Q_{t−1} (bad control) | −0.49 | 0.21 | – | 0.05 |
| DML-GBM (time index + season; random) | −0.75 | 1.97 | 0.83 | 0.77 |
| DML-GBM (time index + season; blocked, gap 14) | −1.35 | 1.41 | 0.37 | 0.63 |
| DML-GBM (lag/rolling demand + season; random) | −0.73 | 0.50 | 0.22 | 0.20 |
| DML-GBM (season only; random), 40 reps | −1.37 | 0.61 | 0.50 | 0.68 |
| DML-GBM (season only; blocked), 40 reps | −1.38 | 0.64 | 0.42 | 0.65 |
| DML-Ridge (Fourier + day-of-week; random), 200 reps | −1.54 | 0.67 | 0.55 | 0.84 |
| DML-Ridge (Fourier + day-of-week; blocked), 200 reps | −1.58 | 0.76 | 0.38 | 0.64 |

Rows without a rep count used 60 reps. What the table shows:
1. A raw time index, or lagged and rolling demand, destroys DML.
2. Even with clean recurring features, a flexible learner trained on about four price regimes learns spurious price–season associations. Its out-of-fold residual price variance was 1.03× (random) and 2.77× (blocked) the raw price variance, and its SEs under-cover.
3. Blocked folds alone did not fix this. When the price process is a step function, held-out blocks contain price regimes never seen in training, so m̂ extrapolates.
4. With few changes, even OLS with HAC under-covers: in a separate 1,000-rep run, NW coverage was 0.876 and EWC(ν = 32) coverage 0.907. Randomization inference over placebo price paths reached 0.965 coverage (200 reps) (see §4).

**Practical rules from these results:**
- Gate DML on a minimum number of price regimes (≥ 5 qualifying changes per product, or pooled across products).
- Use regularized linear nuisances on recurring bases when there are fewer than ~20 regimes.
- Treat residual-variance ratio > 1 as a failure flag.
- Make randomization inference the primary inference whenever there are fewer than ~10 events.

#### Controls: what must and must not go in X
**Include (pre-treatment confounders of both price and demand):**
- day-of-week, Fourier annual terms (K = 2–3), public holidays, paydays/month-end, religious calendars (Ramadan/Eid, Easter, Diwali), school calendar;
- weather (exogenous);
- local events known in advance;
- *lagged* competitor prices (t−1 or earlier);
- product attributes, category, and product-specific unit effects (CRE means, first-differencing or within-transformation, not one-hot FE in trees);
- days since product launch;
- an indicator for left-digit/price-ending changes (§4).

**Exclude:**
- same-day promotion, display or coupon flags (define treatment = regular price and analyze promo days separately, or drop them);
- same-day or end-of-day inventory and stockout flags (post-treatment; opening inventory is fine);
- foot traffic and transactions count (mediators);
- same-day total store sales (contains the outcome);
- same-day competitor prices (possible reactions);
- rolling demand windows that include day t;
- lagged demand **without** lagged price — if dynamics are needed, include both ln Q_{t−k} and ln P_{t−k}, as in an autoregressive distributed-lag model, and report the long-run elasticity `(b0 + Σb_k) / (1 − Σc_k)`;
- raw time index or date fixed effects as tree features;
- supplier cost, FX or tax variables, which are instruments (§3), not controls.

**Learner defaults:**
- Short-stacking with non-negative least-squares weights over:
  - (i) RidgeCV/LassoCV on the recurring basis plus CRE means;
  - (ii) LightGBM with num_leaves = 15, max_depth = 4, learning_rate = 0.05, n_estimators ≤ 500 with early stopping on an inner time-blocked split, min_child_samples = 20, subsample = 0.8 (bagging_freq = 1), colsample_bytree = 0.8, reg_lambda = 1;
  - (iii) RandomForest with 500 trees, min_samples_leaf = 10, max_features = 0.33.
- Below ~1,000 rows or ~20 price regimes, keep only (i).
- Choose and tune learners only on out-of-fold RMSE of ℓ and m under the same time-aware splits.

**Mandatory diagnostics:**
- `RVS = Σ Ṽ² / Σ (D − D̄)²`, the cross-fitted residual price variance share. Gate 0.05 ≤ RVS ≤ 1.0; warn below 0.10; treat > 1 as a nuisance extrapolation failure.
- sd(Ṽ) ≥ 0.02 (2% residual price variation).
- Out-of-fold R² of m̂ (> 0.95 means an overlap failure).
- Ratio of HAC to iid SE (> 1.5 means serial dependence matters).
- Placebo-date randomization inference.
- DoubleML `sensitivity_analysis` (§7).

**Minimum data:** ≥ 5 qualifying price changes per product (or ≥ 20 pooled across ≥ 10 products), ≥ 180 days per product, RVS ≥ 0.05.

**Tests that catch a wrong implementation:**
1. Rebuild the DGP above and assert a pipeline with blocked or grouped folds and a ridge nuisance gives |bias| < 0.1, and that randomization-inference coverage is in [0.93, 0.97] over 200 reps.
2. A feature-lint test: the pipeline must refuse ln Q_{t−1} without ln P_{t−1}, any same-day inventory or promo column, and any date index in tree learners. The simulated −0.49 vs −1.5 bias is the regression fixture.
3. With identical splits, the in-house θ̂ and ψ̂ must match DoubleML's `coef` and `psi` to 1e-8.
4. The DK SE computed from DoubleML `psi` must match a direct DK computation from residuals. For EconML, re-align `residuals_` through an index carried in W, because their order is not guaranteed.
5. The RVS gate fires when D is a deterministic function of X.

### Gaps
- I could not confirm from the docs when DoubleML replaced the 2018 aggregation rule with the median-CI-bound rule, or the reasons given.
- I did not verify the exact LLSW constants and fixed-b critical-value tables. Implement them from the paper.
- I found no published DML application to single-SMB daily price data. Small-sample DML guidance for fewer than ~1,000 observations rests on Clarke–Polselli, Ahrens et al. and my own simulations only.
- The multiplicative DML moment for counts is my derivation and needs validation.

---

## 3. Instruments realistically available to SMBs, and weak-IV diagnostics and inference

### Takeaway
The most credible SMB instruments are product-specific cost shocks, with date FE soaking up common demand effects:
- supplier invoice and wholesale cost changes;
- commodity, FX or fuel shocks interacted with each product's exposure;
- product-specific tax and levy changes.

Hausman instruments (the same SKU's price elsewhere) need multi-tenant data and fail under common demand shocks. Lagged prices are rarely valid. Few price changes make first stages weak, so always report the effective F and the Anderson–Rubin set (closed form for one instrument). Never report only 2SLS t-intervals.

### Cited Findings
- **tF procedure:** the critical value is a smooth function of the first-stage F. A true 5% t-test needs F > 104.7; keeping F = 10 as the threshold means replacing 1.96 with 3.43. This is for the just-identified, single-instrument case. — [Lee, McCrary, Moreira & Porter 2022 AER](https://www.nber.org/system/files/working_papers/w29124/revisions/w29124.rev0.pdf); [Stata help `tf`](https://www.princeton.edu/~davidlee/wp/tf.sthlp)
- **Keane & Neal (2024):** use the Anderson–Rubin test instead of the t-test **even with strong instruments**. 2SLS standard errors are artificially small in samples where the 2SLS estimate is close to OLS, and the resulting power asymmetry inflates false positives when the OLS bias is positive. Their illustration covers AER IV papers 2011–2023. — [Annual Review of Economics 16:185–212](https://www.annualreviews.org/content/journals/10.1146/annurev-economics-092123-111021)
- **Effective F:** a scaled version of the non-robust first-stage F that is robust to heteroskedasticity, autocorrelation and clustering. Critical values range from about 11 to 23.1 depending on the reduced-form covariance; the conservative value 23.1 corresponds to a 10% worst-case Nagar bias at 5% size. The ~37.4 value for a 5% bias tolerance is recalled from memory. — [Montiel Olea & Pflueger 2013 JBES](https://joseluismontielolea.com/Montiel-OleaJBES.pdf); [Stata implementation](https://journals.sagepub.com/doi/pdf/10.1177/1536867X1501500113)
- pyfixest `Feiv.IV_Diag()` / `eff_F()` computes the Montiel Olea–Pflueger effective F from first-stage coefficients and their (robust or clustered) covariance. — [src: pyfixest/estimation/models/feiv_.py](https://github.com/py-econometrics/pyfixest)
- **linearmodels IV results [src]:** `first_stage`, `durbin()`, `wu_hausman()`, `wooldridge_score()`, `sargan`, `basmann`, `wooldridge_overid`, `basmann_f`, `c_stat`. **`anderson_rubin()` is the LIML overidentification test (n·ln κ̂ ~ χ²_q), not the weak-IV-robust AR test for β.** `IVLIML` supports `fuller` and `kappa`; kernel HAC (Bartlett, QS) and cluster covariances are available. — [src: linearmodels/iv/results.py, model.py, covariance.py](https://github.com/bashtage/linearmodels)
- **Weak-IV survey:** recommends reporting the effective F and using weak-IV-robust (AR-type) inference. — Andrews, Stock & Sun 2019†, Annual Review of Economics 11:727–753, [doi](https://doi.org/10.1146/annurev-economics-080218-025643)
- **Hausman instruments:** prices of the same product in other markets proxy common cost shocks and are invalid under common (e.g. national) demand shocks. — Hausman 1996† ([NBER chapter](https://www.nber.org/books-and-chapters/economics-new-goods/valuation-new-goods-under-perfect-and-imperfect-competition)); Nevo 2001† ([doi](https://doi.org/10.1111/1468-0262.00194))
- **Uniform pricing:** most US food, drug and mass-merchandise chains charge nearly uniform prices across stores despite very different demographics and competition. Demand estimates show substantial within-chain variation in elasticities; the median chain gives up about $16m a year in profit, and uniform pricing cuts profit by roughly 7–9% relative to flexible pricing. — [DellaVigna & Gentzkow 2019, QJE 134(4):2011–2084](https://academic.oup.com/qje/article-abstract/134/4/2011/5523148); [NBER digest](https://www.nber.org/digest/mar18/effects-uniform-pricing-us-retail-chains). I recall that their demand identification instruments a store's price with the chain's prices elsewhere (a leave-out Hausman-type IV), but I did not verify this.
- **Left-digit bias:** in US scanner data covering 3,500 products at 25 chains, consumers respond to a 1¢ increase from a 99-ending as if it were more than a 20¢ increase. — [Strulov-Shlain 2023, REStud 90(5):2612–2645](https://academic.oup.com/restud/article-abstract/90/5/2612/6931812)
- **Shift-share designs:** identification can come from exogenous shocks (Borusyak, Hull & Jaravel 2022†, [doi](https://doi.org/10.1093/restud/rdab030)) or exogenous shares (Goldsmith-Pinkham, Sorkin & Swift 2020†, [doi](https://doi.org/10.1257/aer.20181047)).
- **Count-data IV:** a multiplicative-moment GMM estimator. — Mullahy 1997†, REStat 79(4):586–593
- **Recent titles, contents not reviewed:** "An Empirical Comparison of Weak-IV-Robust Procedures in Just-Identified Models" ([arXiv 2506.18001](https://arxiv.org/pdf/2506.18001)); "The Robust F-Statistic as a Test for Weak Instruments" ([arXiv 2309.01637](https://arxiv.org/pdf/2309.01637)); a REStud 2025 manuscript, "A Robust Test for Weak Instruments for 2SLS with Multiple…" ([restud.com](https://www.restud.com/wp-content/uploads/2025/12/MS32919manuscript.pdf)), authors not verified.

### Inferences

#### Estimator card 3A — just-identified IV with Anderson–Rubin set (primary IV procedure)
```
Partial out W (product FE, date or category×week FE, calendar, weather) from y = ln Q (weekly) or use CF-PPML for counts, d = ln P, z:
First stage   d = π_d z + W'δ_d + v ;   reduced form   y = π_y z + W'δ_y + e
2SLS          β̂ = π̂_y / π̂_d = Σ z̃y / Σ z̃d
Var(β̂)        = Ω̂(z̃·û) / (Σ z̃ d)²,   û = y − β̂d − W'δ̂;   Ω̂ by HC / cluster(product) / DK(day)
F_eff (k=1)   = π̂_d² / V̂(π̂_d)  [robust/HAC/cluster variance] — equals Olea–Pflueger effective F with one instrument
AR(β0)        = (π̂_y − β0 π̂_d)² / (V̂_yy − 2β0 V̂_yd + β0² V̂_dd)  ~ χ²₁    (V̂ = joint robust covariance of (π̂_y, π̂_d))
AR set        = {β0 : Aβ0² + Bβ0 + C ≤ 0},  c = χ²₁(1−α)
                A = π̂_d² − c V̂_dd ;  B = −2(π̂_y π̂_d − c V̂_yd) ;  C = π̂_y² − c V̂_yy
                A > 0 (⇔ F_eff > c): bounded [(−B ∓ √(B²−4AC))/(2A)]
                A < 0, B²−4AC > 0: union of two rays;   A < 0, B²−4AC < 0: whole real line (unidentified)
```
- **Own check:** n = 300, first-stage robust F = 8.4. The closed-form AR set was (−1.789, −0.544) and grid inversion gave exactly the same interval; the Wald interval was (−1.383, −0.522), narrower and over-confident.
- **Decision rules:**
  - F_eff < 10: report the AR set only; the estimate is tier 6 or worse unless the AR set excludes the decision threshold.
  - 10 ≤ F_eff < 23.1: AR is the primary interval.
  - F_eff ≥ 23.1: report the Wald interval with tF-adjusted critical values plus AR.
  - The 1.96 critical value is valid only if F ≥ 104.7.
- **Overidentified (several cost shifters):** report 2SLS, LIML and Fuller(1) (`IVLIML(fuller=1)`), the Hansen J or overidentification test, the k-degree-of-freedom AR test (or CLR†), and the effective F. A simpler path is to collapse the shifters into one predicted-cost index z.
- **Counts:** control-function PPML. Stage 1: OLS of ln P on z and W, keep residual v̂. Stage 2: PPML of Q on ln P, v̂ and W. Bootstrap both stages together (block bootstrap by product or by time block). The t-test on v̂ is a robust Durbin–Wu–Hausman test (§7).
- **Minimum data:** ≥ 3 distinct shifter episodes per product, or ≥ 20 product×episode cells pooled; F_eff ≥ 10 for a point estimate to be shown.
- **Tests:**
  1. The just-identified 2SLS must equal π̂_y/π̂_d to machine precision.
  2. With π_d = 0, the AR set must be unbounded in ≥ 95% of simulations and cover the truth at the nominal rate, while the Wald interval under-covers.
  3. With k = 1, F_eff must equal the squared robust first-stage t.
  4. A code-review assertion that linearmodels' `anderson_rubin()` is never used as the weak-IV interval.

#### Instrument catalogue for SMBs (validity conditions and failure modes; my synthesis)
1. **Supplier invoice or wholesale cost (product-specific)**, entered as ln C_{j,t−k} with k = 0–14 days.
   - *Relevance:* pass-through, often lagged and in rounded steps.
   - *Exclusion fails when:* the supplier raises prices because of a nationwide demand surge for that product (use date FE plus category×week FE); the cost change coincides with a quality or pack-size change, including shrinkflation (flag it in ETL); or the SMB itself bargains costs down when its own demand is high, which is rare.
   - *Data:* purchase records or bills — the most valuable optional data to collect at onboarding.
2. **Commodity, FX or fuel shocks interacted with exposure (shift-share).** `z_jt = s_j × Δln(FX_t)` or `s_j × ln(commodity_t)`, where s_j is the product's import or input share, always with date FE.
   - *Caveat:* in emerging markets, FX and fuel shocks also cut real incomes and shift demand, more strongly for imported or "luxury" goods. Exposure is then correlated with income-elasticity, which violates exclusion. Add s_j × macro-demand-proxy controls (e.g. sales of low-exposure staples), or use only products with similar income elasticity.
3. **Tax and levy changes on some products** (VAT-rate or base changes, excise or sugar levies). These are legislated and product-specific, so they are exogenous to one SMB's local demand. Use them as IV or as a taxed-vs-untaxed DiD.
   - *Risks:* partial or anticipated pass-through; tax salience effects on demand; announcement effects.
4. **Hausman / leave-market-out price.** `z_jt = mean ln P` of the same GTIN/SKU at other tenants more than R km away (e.g. R ≥ 50 km), excluding own price, with date FE.
   - Relevant when tenants share suppliers.
   - Invalid under national or regional demand shocks (ads, holidays, weather fronts). Test with an overidentification test against a cost IV when both exist.
   - Platform-only: it needs multi-tenant data.
5. **Lagged prices:** valid only if demand shocks are serially uncorrelated beyond the lag. That is implausible given AR(1)-type sales errors (my simulation used ρ = 0.7). Treat them as the lowest tier, and only within dynamic-panel GMM with AR(2) tests.
6. **Rounding or price-point discreteness** (my inference; no published validation found). When costs drift continuously but prices jump between round points (e.g. KES 50 or 100, NGN 50 or 100 multiples, or 9-endings), the date of a jump is triggered by cumulative cost crossing a threshold. A rule-based simulated instrument such as `z = ceil(c_t × markup_j / unit) × unit` can isolate the cost-driven part. In high-inflation markets, real prices also erode smoothly between nominal resets because of aggregate inflation, giving "sawtooth" real-price variation that is plausibly unrelated to product-specific demand. The timing of resets may still be state-dependent, so test with leads.
7. **Manufacturer recommended-retail-price list changes:** exogenous to local demand, but national campaigns may coincide with them.

**Competitor entry or exit, weather and events are demand shifters, never instruments for own price.**

### Gaps
- I did not retrieve the tF critical-value function table; implement it from Lee et al.'s supplementary material.
- I did not re-verify the full Montiel Olea–Pflueger critical-value table (only 23.1 was confirmed by search).
- No SMB- or Africa-specific IV elasticity studies were found within the budget.
- DellaVigna–Gentzkow's exact IV construction and elasticity magnitudes were not verified.

---

## 4. Quasi-experiments: event studies around discrete regular-price changes, DiD around competitor entry/exit, RD around price endings, synthetic control for a single business

### Takeaway
The workhorse SMB design is an event study, equivalently a regression discontinuity in time (RDiT), around discrete regular-price changes:
- PPML on daily counts;
- seasonality and holidays modelled;
- a "donut" around the change date;
- event-specific trends;
- inference by randomization over placebo dates.

Upgrade it to DiD when clean, non-substitute comparison products or the same SKU at other tenants exist. The other quasi-experiments answer different questions:
- Competitor-entry DiD estimates competition effects, not own-price elasticity.
- RD at price endings estimates left-digit jumps, not slopes, and the firm chooses its running variable.
- Synthetic control for one business requires a multi-tenant donor pool.

### Cited Findings
- **Few policy changers:** when identification comes from a small number of changing groups, point estimators are not consistent, but the asymptotic distribution can be estimated under any point null using residuals from the many non-changing groups. — [Conley & Taber 2011, REStat 93(1):113–125](https://ideas.repec.org/a/tpr/restat/v93y2011i1p113-125.html)
- **Pre-trend tests** often have low power, and conditioning the analysis on passing one can worsen bias and under-coverage. — Roth 2022, AER: Insights 4(3):305–322 ([record](https://ouci.dntb.gov.ua/en/works/73oo8yX7/))
- **Robust inference under bounded parallel-trend violations**, with post-treatment violations assumed not "too different" from pre-trends (relative-magnitude or smoothness restrictions): Rambachan & Roth 2023, REStud 90(5):2555–2591 ([HonestDiD]; described in the [DiD survey](https://arxiv.org/pdf/2201.01194)).
- **Equivalence-based pre-trend tests** reverse the null to "trends differ by more than a margin". — [JBES 2024](https://www.tandfonline.com/doi/full/10.1080/07350015.2024.2308121) (authors not verified)
- **Left-digit effects** are large (above). Firms price as if the bias were much smaller than it is, more consistent with rules of thumb than optimization. — [Strulov-Shlain 2023](https://academic.oup.com/restud/article-abstract/90/5/2612/6931812). A related REStud paper exploits Lyft price thresholds: ["Left-Digit Bias at Lyft", REStud 90(6):3186](https://academic.oup.com/restud/article-abstract/90/6/3186/7045819) (authors not verified).
- DoubleML includes an RDD module (`doubleml/rdd/rdd.py`, flexible covariate-adjusted RD). — [src](https://github.com/DoubleML/doubleml-for-py)
- **Interrupted time series:** segmented regression with explicit seasonality and autocorrelation handling. — Bernal, Cummins & Gasparrini 2017†, IJE 46(1):348–355, [doi](https://doi.org/10.1093/ije/dyw098)
- **RD in time:** pitfalls include time-varying confounders, serial correlation, bandwidth choice and short-run vs long-run effects. — Hausman & Rapson 2018†, Annual Review of Resource Economics 10:533–552, [doi](https://doi.org/10.1146/annurev-resource-121517-033306)
- **RD with a discrete running variable:** honest CIs (Kolesár & Rothe 2018†, [doi](https://doi.org/10.1257/aer.20160945)); robust bias-corrected inference (Calonico, Cattaneo & Titiunik 2014†, [doi](https://doi.org/10.3982/ECTA11757)); manipulation (density) test (McCrary 2008†, [doi](https://doi.org/10.1016/j.jeconom.2007.05.005)).
- **Synthetic control and extensions:** ADH 2010† ([doi](https://doi.org/10.1198/jasa.2009.ap08746)); Abadie 2021† feasibility and data requirements ([doi](https://doi.org/10.1257/jel.20191450)); synthetic DiD (Arkhangelsky et al. 2021†, [doi](https://doi.org/10.1257/aer.20190159)); conformal inference (Chernozhukov, Wüthrich & Zhu 2021†, [doi](https://doi.org/10.1080/01621459.2021.1920957)).
- **Stacked DiD with clean controls** per event. — Cengiz et al. 2019†, [doi](https://doi.org/10.1093/qje/qjz014)
- **Price dynamics:** regular prices are sticky; temporary sales are frequent and revert (Nakamura & Steinsson 2008†, [doi](https://doi.org/10.1162/qjec.2008.123.4.1415)). Temporary price cuts trigger stockpiling, so short-run promo elasticities overstate long-run ones (Hendel & Nevo 2006†).

### Inferences

#### Estimator card 4A — RDiT/event-study PPML around regular price changes (default for single-business data)
```
Events e = (product j, date t_e, Δ_e = ln P_post − ln P_pre), |Δ_e| ≥ 0.03, new price persists ≥ 21 days, no promo/stockout/supplier/pack-size flag within ±7 days
Window  t ∈ [t_e − W, t_e + W], W = 28 days (56 if mean sales < 2/day); donut: drop t_e−2 … t_e+2
PPML    E[Q_jt] = exp( α_e + β·ln P_jt + DOW_t + holiday/payday_t + λ_e·(t − t_e) )      (one trend slope per event, same pre/post)
Event-time form: replace β·ln P with Σ_k δ_k·1{t − t_e ∈ bin_k}; plot δ_k/Δ_e; leads test anticipation/pre-trends
Pooled elasticity β̂ over events (each event contributes ∝ Δ_e² × information)
Inference (primary when events < 10): randomization over placebo dates —
   draw R=999 pseudo-dates from windows with no price change for the same product(s), assign each the actual Δ_e,
   recompute β̂ → p = (1 + #{|β̂_placebo| ≥ |β̂|}) / (R + 1); invert over β0 (test on Y·e^{−β0 ln P}) for a CI
Secondary: NW HAC within event (lag 7–14), cluster by event if ≥ 20 events
```
- **Expected precision, computed before estimating** (this formula is the "minimum data" gate). Over a ±W-day window, `SE(β̂) ≈ √( 2·LRV / (W·Σ_e Δ_e²) )`, with `LRV ≈ (φ/μ)·DEFF` the long-run variance of daily log sales (Poisson: 1/μ; φ = overdispersion; DEFF from serial correlation).
  - Worked example: μ = 4 units/day, Δ = 0.10, W = 28, DEFF = 3 gives **SE ≈ 2.3 for one event**, and ≈ 0.73 for ten such events.
  - Gate: if the projected SE exceeds 1.0, do not show an own-data elasticity. Show the prior and recommend an experiment (§5).
- **Own simulation supporting randomization inference.** With four step changes and AR(1) errors (ρ = 0.7), 95% coverage of the parametric estimator was 0.876 with NW (S = 1.3√T, z critical values; 1,000 reps), 0.907 with EWC (ν = 0.4T^{2/3}, t critical values), 0.863 with EWC (ν = 8), and **0.965 with randomization inference** over placebo price paths (200 reps). Asymptotic HAR methods under-cover when the regressor is a persistent step function, so randomization inference should be primary.
- **Failure modes and countermeasures:**

| failure | countermeasure |
|---|---|
| concurrent supplier, quality or pack-size change | ETL flags; drop the event |
| anticipation or stockpiling around announced increases | donut; lead coefficients |
| price changed in response to a demand spike (regression to the mean) | leads; placebo test using pre-period demand shocks |
| holidays or paydays straddling t_e | explicit calendar terms |
| stockouts after a price cut (censoring) | drop censored days |
| habit or reference-price dynamics (short-run ≠ long-run) | report δ_k profile; long-run = average of bins ≥ 4 weeks |

- **Tests:** a DGP with seasonality + trend + AR(1) and **no** price effect must give β̂ ≈ 0, with randomization-inference rejection rate ≈ α; leave-one-event-out stability (no single event moves β̂ by more than 1 SE); the placebo-outcome check (§7).

#### Estimator card 4B — DiD with comparison products or other tenants (tier upgrade)
- Use the event structure from 4A with comparison units that did not change price in the window. Estimate with BJS/CS/SA (staggered) or dCDH (repeated or continuous). Elasticity = ATT/Δ (see 1C).
- **Controls must be non-substitutes.** Spillover test: estimate the "effect" on same-category unchanged products. A significant positive effect indicates substitution; exclude those products and report a cross-price estimate instead.
- **Inference:** cluster by product; with fewer than ~5 treated units, use Conley–Taber-style inference or randomization inference; report the HonestDiD breakdown M̄* (the largest relative pre-trend violation under which the CI still excludes the decision threshold).

#### 4C — DiD around competitor entry/exit (platform-level analysis, a different estimand)
- **Estimand:** the effect of competitor entry on own demand, price and margin. It is not own-price elasticity.
- **Valid uses:** (i) a triple difference testing whether β changes after entry, using own price changes before and after (regime heterogeneity, which links to v1 §21); (ii) features for the hierarchical prior; (iii) a demand shifter for supply-side (pricing-response) analysis.
- **Design:** event at announcement or construction start; treatment intensity = distance band (continuous dose, so CGBS caveats apply); CS/SA/BJS with never- or not-yet-treated businesses (multi-tenant) as controls; exclude controls within the entrant's catchment (spatial spillovers); with few treated businesses, Conley–Taber or randomization inference; HonestDiD for pre-trends.
- **A single business has no internal control**, so use synthetic control or synthetic DiD with other tenants as donors.

#### 4D — RD around price endings and thresholds
- **This does not identify the elasticity slope.** The firm chooses its own price, the running variable, so sorting (bunching at 9-endings) violates RD assumptions by construction. A density test will reject.
- **Use instead:**
  - (i) add `crosses_left_digit` and `ending_9` features to the demand model so that a change like 9.99 → 10.49 does not load a left-digit jump onto β;
  - (ii) compare within-product changes that cross a left-digit boundary with changes of similar size that do not (the cross-threshold vs non-crossing comparison used in the left-digit literature, from memory);
  - (iii) RD *in time* (4A) is the valid RD for prices.
- With discrete running variables, use Kolesár–Rothe honest intervals.

#### 4E — Synthetic control for a single business or product
```
Weights w ≥ 0, Σw = 1 minimizing pre-period Σ_t (Y_1t − Σ_j w_j Y_jt)² on weekly log sales (or levels, then log-ratio)
ATT_t = Y_1t − Σ_j w_j Y_jt  (post);  ε̂ = mean_post(ATT_t on log scale) / Δ
Inference: placebo-in-space p = rank(post/pre RMSPE ratio of treated) / (J+1); or conformal (CWZ); SDID for unit+time weights
```
- **Requirements:** donors J ≥ 10–20 (same SKU at other tenants, or similar products in non-substitute categories); weekly aggregation; ≥ 20 pre-period weeks; pre-period RMSPE ≤ 0.5 × SD of the outcome.
- **Failure modes:** supplier-driven price changes also hit the donors (other tenants with the same supplier change price too), so drop donors with changes within ±4 weeks; interpolation bias; too few donors.
- **Test:** placebo runs on untreated donors must give randomization-inference rejection rates close to α.

### Gaps
- I found no empirical validation of event-study elasticities against experiments for SMB retail. This is a natural internal study once experiments run (the "flywheel").
- I did not verify the authors of "Left-Digit Bias at Lyft" or of the JBES equivalence-test paper.
- I did not re-verify the HonestDiD formulas (relative-magnitude bounds); implement them from Rambachan & Roth.

---

## 5. Randomized and switchback price experiments for SMBs: designs, power and sample size, carryover, legal and ethical limits

### Takeaway
For SMBs the workable gold standard is **time-based (switchback) or product-level randomization of posted prices**, not customer-level 10/90 splits. Size every experiment with a count-data power calculation.

Per-product volumes are low. A ±10% single-product test at 4 units/day gives SE ≈ 1.0 on the elasticity after 120 days. Designs therefore need:
- larger price contrasts;
- pooling across similar products;
- block randomization with washout periods for carryover;
- targeting of the *decision threshold* β* = −1/margin rather than β = 0;
- a legal and ethical review.

### Cited Findings
- **Switchback design (Bojinov, Simchi-Levi & Zhao 2023):** units are exposed sequentially to random treatments over periods. Under assumptions about the order of the carryover effect, the optimal design solves a minimax discrete optimization. Inference uses exact randomization p-values or a finite-population CLT, giving conservative tests and CIs. — [Management Science 69(7)](https://pubsonline.informs.org/doi/10.1287/mnsc.2022.4583); [HBS page](https://www.hbs.edu/faculty/Pages/item.aspx?num=62141)
- **Newer switchback work, titles only (not reviewed):** surrogate-variable switchbacks ([Management Science](https://pubsonline.informs.org/doi/10.1287/mnsc.2023.03818)); "Spectral Design of Random-Duration Switchbacks" ([arXiv 2609.13698](https://arxiv.org/pdf/2609.13698), 2026); multivariate covariate adjustment for switchbacks, "CUPED on Steroids" ([arXiv 2608.24038](https://arxiv.org/pdf/2608.24038), 2026); switchbacks under geometric mixing (Hu & Wager 2022†, [arXiv 2209.00197](https://arxiv.org/abs/2209.00197)).
- **Design-based DiD for staggered adoption** (randomized rollout timing). — Athey & Imbens 2022†, J. Econometrics 226(1):62–79, [doi](https://doi.org/10.1016/j.jeconom.2020.10.012)
- **Inference after adaptive (bandit) experiments** needs special estimators. — Hadad et al. 2021†, PNAS, [doi](https://doi.org/10.1073/pnas.2014602118)
- **Field price experiments:** a large-scale randomized experiment over many price levels at an online firm (ZipRecruiter) was used to estimate demand and evaluate personalized pricing (Dubé & Misra 2023†, JPE 131(1), [doi](https://doi.org/10.1086/720793); specific findings not re-verified). Randomized cost shocks among Kenyan agricultural traders were used to study pass-through and competition (Bergquist & Dinerstein 2020†, AER 110(12), [doi](https://doi.org/10.1257/aer.20171397)).
- **Fairness norms constrain profit-seeking price changes.** Customers judge prices as unfair when firms exploit demand shifts. — Kahneman, Knetsch & Thaler 1986†, AER 76(4) ([JSTOR](https://www.jstor.org/stable/1806070))
- **EU disclosure:** traders must tell consumers when a price was personalised by automated decision-making (amendment to Consumer Rights Directive Art. 6(1)(ea)). — Directive (EU) 2019/2161†, [EUR-Lex](https://eur-lex.europa.eu/eli/dir/2019/2161/oj)

### Inferences

#### Power and sample-size formulas (derived here; the variance formula was validated by simulation)
```
Two arms P_L < P_H, Δ = ln(P_H/P_L); daily mean μ_L, μ_H = μ_L·e^{βΔ}; overdispersion φ (Var Q = φμ)
Estimator: β̂ = [ln Q̄_H − ln Q̄_L]/Δ   (≡ PPML with an arm dummy)
Var(β̂) ≈ (φ·DEFF/Δ²) · (1/(n_H μ_H) + 1/(n_L μ_L))
Days for target SE s (equal split):             n = 2φ·DEFF·(1/μ_H + 1/μ_L) / (Δ² s²)
Days to reject β = β0 with power 1−κ, level α:  n = (z_{1−α/2} + z_{1−κ})² · 2φ·DEFF·(1/μ_H + 1/μ_L) / (Δ² (β − β0)²)
Continuous/multi-level design:                  Var(β̂) ≈ φ·DEFF / (n·μ̄·Var(ln P))  → put arms at the extremes of the allowed range (D-optimal for a slope)
Pooling J products with a common β:            Var ÷ J  (or replace n·μ by Σ_j n_j μ_j)
Unequal allocation penalty:                     Var ∝ 1/p + 1/(1−p);  10/90 needs 2.78× the days of 50/50
Block (switchback) randomization, block length b days, AR(1) daily correlation ρ:
   DEFF = 1 + 2·Σ_{k=1}^{b−1}(1 − k/b)·ρ^k   (ρ=0.5: b=3→1.83, b=7→2.43;  ρ=0.7: b=7→3.63)
Washout: discard c days after each switch → n_eff = n − c·(#switches)
Decision threshold (constant elasticity, marginal cost c, current margin m = (P−c)/P):
   Lerner optimum (P*−c)/P* = −1/β  ⇒  raise price iff β > −1/m;  revenue-only rule: β > −1
```
- **Simulation check** (μ_L = 4/day, +10% price, β = −1.5, n = 120 days, Poisson, 4,000 reps):
  - SE from the formula 0.994 vs simulated SD 1.015;
  - power against β = 0: formula 0.33, simulation 0.34;
  - power against β = −1: formula 0.07, simulation 0.08.
- **Days required in that setting:**
  - 474 for SE ≤ 0.5;
  - 1,897 for SE ≤ 0.25;
  - **3,722 to distinguish β = −1.5 from −1 with 80% power.**
  - A +20% contrast gives SE ≈ 0.54 at 120 days (computed).
  - Pooling 10 similar products with a common β brings that to about 0.17.

**Conclusion:** single-product SMB experiments must use larger price contrasts and pooling. A hierarchical prior turns the experiment into an update rather than a stand-alone estimate.

#### Designs, in order of feasibility for an SMB
1. **Product switchback.** Randomize the price arm for each block of b = 7 days, so blocks balance day-of-week, with 1–2 washout days after each switch. Stratify blocks by month. Use ≥ 8 blocks per arm. Analyze with PPML (arm dummy + calendar) and Fisher randomization inference over block assignments.
2. **Cross-product randomized crossover.** Pair products by volume and margin *across* categories, to avoid within-pair substitution. Each week, randomize which product in each pair gets the higher price. Comparing within a pair and week removes store-day shocks.
3. **Randomized staggered rollout** of a planned permanent price change across products (stepped wedge). Gives design-based DiD (Athey–Imbens) and naturally extends to the 1C event estimators.
4. **Multi-branch SMBs:** cluster-randomize at the branch×week level.
5. **Customer-level discount codes** sent by SMS or WhatsApp. Discounts are generally accepted, and this is the only legitimate form of customer-level variation in most settings. The estimand is the elasticity among coupon recipients and redeemers, and it requires customer identifiers.
6. **Bandits** (Thompson sampling over a price grid) only after an initial randomized phase, with adaptive-inference corrections.

#### Carryover and interference
- **Carryover test:** regress the outcome on the current arm, the previous block's arm and day-in-block. A non-zero previous-arm coefficient, or a strong day-in-block gradient in the first days of a block, means the washout must be longer.
- **Interference:** cross-product substitution means the arm contrast of one product understates the category-level effect. Also record total category sales, and randomize at category level when the decision is a category-wide price change.

#### Legal and ethical guardrails (inferences; jurisdiction-specific law not verified this session)
- **Time-based variation** (everyone sees the same posted price at a given time) is ordinary retail behaviour and the lowest-risk design.
- **Customer-level differential pricing** carries fairness-backlash risk and triggers disclosure duties where pricing is personalised by algorithm (EU Omnibus†). There are newer US state rules (e.g. New York's 2025 algorithmic-pricing disclosure law) and data-protection rules on automated decisions (e.g. South Africa POPIA, Kenya Data Protection Act 2019, Nigeria Data Protection Act 2023). All of these are **unverified in this session; obtain local legal review**.
- **Competition-law price-discrimination rules** mostly target dominant firms or B2B sales (e.g. US Robinson–Patman Act†). SMBs are rarely dominant.
- **A platform risk worth flagging:** pooling competitor price data and recommending prices to *competing* tenants from a shared engine can raise algorithmic hub-and-spoke coordination concerns. This is my inference, prompted by recent US litigation over shared pricing software (not verified this session). Keep tenants' recommendations independent and exclude rivals' non-public prices.
- **Guardrails:**
  - cap test deviations (e.g. ±15% of the current price);
  - exclude essentials, staple foods and medicines, and any product under price-control or anti-gouging rules (several emerging markets have essential-goods price-control laws; not verified);
  - require owner opt-in;
  - charge exactly the displayed price and never use deceptive "was/now" reference prices;
  - set stop rules (e.g. halt if category revenue falls more than X% or complaints arise);
  - log every randomization for audit.

#### Tests that catch a wrong implementation
1. Reproduce the simulation above: the power calculator's SE must match Monte Carlo within 5%.
2. Fisher randomization-inference p-values under a null DGP must be uniform (KS test).
3. With an injected carryover, the carryover detector must flag it.
4. The allocation penalty function must return 2.778 for 10/90.

### Gaps
- No peer-reviewed SMB field price experiments from Africa were found within the search budget.
- Typical carryover lengths for SMB retail and food service are unknown, so they should be measured internally.
- I did not verify the exact Bojinov et al. optimal block structure (the order-m carryover design) from the paper.
- Jurisdiction-specific laws (Kenya, Nigeria, South Africa, Ghana, Egypt, EU, US states) need legal review.

---

## 6. Scoring identification strength: replacing the v1 IDS and 7-tier hierarchy with observable diagnostic gates

### Takeaway
Replace the estimator-ranked hierarchy and the free-form IDS with a **design-based tier** that an estimate earns by passing explicit, observable gates:
- number of independent price changes;
- residual price variation;
- instrument strength;
- pre-trend, placebo and randomization-inference results;
- sensitivity robustness values;
- precision relative to the decision threshold.

The tier decides (a) what the product may show the user and (b) how much the estimate may move the hierarchical prior. Ensemble weights then come from tier and precision, never from predictive fit.

### Cited Findings
- **Instrument-strength thresholds:** F > 104.7 for nominal t-tests (LMMP); conservative effective-F critical value 23.1 (Montiel Olea–Pflueger); use AR even with strong instruments (Keane–Neal). — [LMMP](https://www.nber.org/system/files/working_papers/w29124/revisions/w29124.rev0.pdf); [MOP](https://joseluismontielolea.com/Montiel-OleaJBES.pdf); [Keane & Neal](https://www.annualreviews.org/content/journals/10.1146/annurev-economics-092123-111021)
- **Pre-trend tests** have low power and pre-testing distorts inference (Roth 2022, [record](https://ouci.dntb.gov.ua/en/works/73oo8yX7/)), so sensitivity/breakdown analysis à la Rambachan–Roth is preferable ([survey](https://arxiv.org/pdf/2201.01194)).
- **Few changes:** inference with few changers uses non-changers' residuals. — [Conley & Taber 2011](https://ideas.repec.org/a/tpr/restat/v93y2011i1p113-125.html)
- **Robustness value (RV):** summarizes the confounder strength (share of residual variance explained in both treatment and outcome) needed to overturn a conclusion. — [Cinelli & Hazlett 2020, JRSS-B 82(1):39–67](https://academic.oup.com/jrsssb/article-abstract/82/1/39/7056023). DoubleML reports `rv` and `rva` for DML models. — [src](https://github.com/DoubleML/doubleml-for-py)
- **Plausibility range:** meta-analytic mean −2.62, SD 2.21, from 1,851 elasticities (Bijmolt et al. 2005, [link](https://research.rug.nl/en/publications/new-empirical-generalizations-on-the-determinants-of-price-elasti/)). Elasticities vary substantially within chains (DellaVigna & Gentzkow 2019, [link](https://academic.oup.com/qje/article-abstract/134/4/2011/5523148)).
- **Bias-adjusted evidence synthesis:** observational evidence is down-weighted by adding elicited bias variance. — Turner et al. 2009†, JRSS-A 172(1):21–47, [doi](https://doi.org/10.1111/j.1467-985X.2008.00547.x)

### Inferences

#### Observable inputs, computed per product×estimate
- `n_ev`: qualifying independent regular-price-change events (definition in 1C). Also `n_eff = (Σ|Δ_e|)² / ΣΔ_e²`, a Kish-style effective count. Menu-wide simultaneous changes count as **one** time event, plus dose observations.
- `RVS` and `sd_resid`: cross-fitted residual price variance share and residual SD (2A).
- `F_eff` and AR-set boundedness when an IV is used (3A).
- `p_pre`: joint pre-trend test p-value. `M̄*`: HonestDiD breakdown value.
- `p_RI`: placebo-date randomization-inference p-value (4A).
- `RV_{q*,α}`: robustness value against the **decision threshold** β* (§7).
- `z_dec = |β̂ − β*| / SE`: decision precision. Use β* = −1/m with margin m, or −1 for revenue decisions.
- `z_agree`: agreement between two estimators of the *same* design and estimand, e.g. PPML-RDiT vs PLR-DML: `|β̂_A − β̂_B| / √(SE_A² + SE_B² − 2·Cov)`. Get the covariance from a joint block bootstrap.
- Plausibility: β̂ ∈ [−8, 0.3], and the prior-predictive p-value under the hierarchical prior.

#### Proposed tiers (an estimate gets the highest tier whose gates all pass; my proposal)

| tier | design | gates |
|---|---|---|
| 1 | Randomized price experiment (switchback, product-level, rollout) | randomization log verified; ≥ 8 blocks per arm; carryover test p > 0.1 or washout applied; randomization-inference p reported; z_dec ≥ 1 or SE ≤ 0.5 |
| 2 | Exogenous-driver natural experiment (documented supplier, tax or RRP change) with comparison units, **or** strong IV | driver documented for ≥ 1 event; pre-trend leads within HonestDiD bounds (M̄* ≥ 1); p_RI < 0.1; or F_eff ≥ 23.1 with bounded AR set |
| 3 | Multi-event DiD or event study with clean comparison units, heterogeneity-robust estimator | n_ev ≥ 3; spillover test passed; M̄* ≥ 0.5; p_RI < 0.1; robust estimator (BJS/CS/SA/dCDH), not TWFE |
| 4 | Single-series RDiT/ITS around regular changes, **or** moderate IV | n_ev ≥ 2; donut + calendar; p_RI < 0.2; leads ≈ 0; or 10 ≤ F_eff < 23.1 with bounded AR set |
| 5 | Selection on observables (PLR-DML or PPML-FE) with adequate variation | n_ev ≥ 4 or (RVS ≥ 0.10 and sd_resid ≥ 0.03); time-aware folds; HAC/DK or randomization inference; RV_{q*,0.05} ≥ 0.05 |
| 6 | Weak observational | identifying variation exists (n_ev ≥ 1 or RVS ≥ 0.02) but tier-5 gates fail |
| 7 | No identifying variation | n_ev = 0 and RVS < 0.02: **show the prior only**, labelled "not estimated from your data"; recommend an experiment |

**Continuous IDS ∈ [0, 1]:** `IDS = base_tier × q`, with base = {1.00, 0.85, 0.75, 0.60, 0.45, 0.25, 0} and q the geometric mean of the applicable sub-scores:

| sub-score | formula |
|---|---|
| events | 1 − e^{−n_eff/3} |
| variation | min(1, RVS/0.2) |
| pre-trends | min(1, M̄*) |
| placebo | 1 − min(1, p_RI/0.2) |
| sensitivity | min(1, RV/0.1) |
| agreement | exp(−max(0, z_agree − 1)) |
| precision | min(1, z_dec/2) |

The weights and constants are placeholders to be calibrated. Calibration route: wherever both an observational and an experimental estimate exist for the same product, regress |β̂_obs − β̂_exp| on the diagnostics, so the score predicts bias.

#### How tiers feed the rest of the system (replacing v1 §40 and §86)
- **Selection rule:** report the estimate from the highest tier available. Pool (inverse-variance) only estimates from the **same tier and estimand** that pass an agreement test, i.e. Cochran's Q p > 0.1. Never weight by out-of-sample demand-forecast accuracy, and never include the GBM demand model's implied elasticity.
- **Hierarchical update with a tier-dependent bias variance:** use `1/(SE² + τ²_tier)` as the likelihood precision. Placeholder τ_tier = {0, 0.15, 0.25, 0.4, 0.5, 0.8, ∞} for tiers 1–7, to be calibrated from experiments versus observational estimates. A tier-6 estimate with SE = 0.3 then moves the prior about as much as an SE ≈ 0.85 estimate, so weakly identified data cannot override strong priors.
- **Data-condition router (defaults):**

| data condition | primary estimator | robustness / secondary | max tier |
|---|---|---|---|
| randomized test run | PPML arm contrast + Fisher randomization inference | event-time profile | 1 |
| documented cost, tax or supplier-driven change(s) | event study with comparison units; IV (AR) if costs are logged | RDiT single series | 2 |
| ≥ 3 regular changes across products, non-substitute controls | BJS/did2s, CS, SA, dCDH; Poisson ETWFE for counts | TWFE-PPML (diagnostic, with weights) | 3 |
| 1–2 changes, single product | RDiT-PPML + placebo randomization inference | – | 4 if gates pass, else 6 |
| many small or continuous changes, rich covariates | PLR-DML (time-aware folds, ridge or stacked nuisance) + DK/randomization inference | PPML-FE | 5 |
| cost shifter with F_eff ≥ 10 | IV with AR set / CF-PPML | DWH vs OLS | 2–4 |
| only temporary promotions | separate *promotional* elasticity with lead/lag dynamics (stockpiling); never relabel as regular elasticity | – | flag |
| no price variation | hierarchical prior only; recommend an experiment | – | 7 |

- **Minimum display rule:** show an own-data elasticity only if tier ≤ 5 **and** projected SE ≤ 1.0 **and** IDS ≥ 0.3. Otherwise show the prior-based estimate with an explanation.

### Gaps
- I found no published scoring system for identification strength of elasticity estimates. The tiers, gates and IDS formula above are a proposal requiring calibration.
- Turner et al.'s bias-modelling approach is the closest literature analogue, but I did not re-verify it this session.

---

## 7. Endogeneity tests (Durbin–Wu–Hausman) and sensitivity analysis for unobserved confounding (Cinelli–Hazlett bounds, E-values)

### Takeaway
Run the regression-based (control-function) Durbin–Wu–Hausman test with HAC or cluster SEs only when a credible instrument exists, and never treat a non-rejection as licence to go back to OLS. For every observational elasticity, report Cinelli–Hazlett robustness values (OLS/FE/PPML-linearized) or Chernozhukov et al. DML bounds (via DoubleML). Compute them **against the decision-relevant null** (−1/margin or −1), not 0, and benchmark them against observed covariates. E-values fit continuous log-price treatments poorly; use them only for binary event contrasts.

### Cited Findings
- **Hausman specification test** compares an efficient estimator under H0 (OLS) with one that is consistent under H1 (IV). — Hausman 1978†, Econometrica 46(6), [doi](https://doi.org/10.2307/1913827)
- **linearmodels** implements `durbin()`, `wu_hausman()` and the heteroskedasticity-robust `wooldridge_score()` endogeneity tests, plus `c_stat()`. — [src: linearmodels/iv/results.py](https://github.com/bashtage/linearmodels)
- **Power asymmetry of 2SLS t-tests near OLS** also undermines Hausman-type pretests. — [Keane & Neal 2024](https://www.annualreviews.org/content/journals/10.1146/annurev-economics-092123-111021). A Hausman pretest followed by choosing OLS or IV can have severe size distortions. — Guggenberger 2010†, Econometric Theory 26(2):369–382
- **Cinelli & Hazlett (2020):** omitted-variable bias is expressed through partial R² values. The robustness value is the minimum strength of association (equal partial R² with treatment and outcome) a confounder needs to explain away 100q% of the estimate, or to make it insignificant at level α. Software: `sensemakr` (R/Stata; a Python port exists). — [JRSS-B](https://academic.oup.com/jrsssb/article-abstract/82/1/39/7056023); [sensemakr CRAN](https://cran.r-project.org/web/packages/sensemakr/sensemakr.pdf)
- **OVB for causal ML (Chernozhukov, Cinelli, Newey, Sharma & Syrgkanis):** the bias is bounded by plausibility judgements on the maximal explanatory power of omitted variables, covering ATEs, average causal derivatives and policy effects, with DML-based inference. — [NBER w30302](https://www.nber.org/papers/w30302); [arXiv 2112.13398](https://arxiv.org/abs/2112.13398)
- **DoubleML implementation [src]:**
  - Call: `sensitivity_analysis(cf_y=0.03, cf_d=0.03, rho=1.0, level=0.95, null_hypothesis=0.0)`.
  - Bounds: `θ± = θ̂ ± |ρ|·√(cf_y·cf_d/(1−cf_d))·S`, with `S = √(σ²·ν²)`.
  - For PLR: `σ² = mean[(Y − ℓ̂ − θ̂(D − m̂))²]`, `ν² = 1/mean[(D − m̂)²]`, Riesz representer `(D − m̂)·ν²`.
  - `rv` / `rva` are found by `minimize_scalar` over cf_y = cf_d at which the bound or CI reaches the null.

  — [src: double_ml_framework.py; utils/_sensitivity.py; plm/plr.py](https://github.com/DoubleML/doubleml-for-py)
- **E-value:** for a risk ratio RR (inverted if below 1), `E = RR + √(RR·(RR−1))`, the minimum strength of association, on the RR scale, that an unmeasured confounder must have with both treatment and outcome to explain away the estimate. For continuous outcomes, use the approximation RR ≈ exp(0.91·d) with d a standardized mean difference. — VanderWeele & Ding 2017†, Annals of Internal Medicine 167(4):268–274, [doi](https://doi.org/10.7326/M16-2607)
- **Recent titles, contents not reviewed:** "Sensitivity Analysis for Causal ML: A Use Case at Booking.com" ([arXiv 2510.09109](https://arxiv.org/pdf/2510.09109)); "Sensitivity Analysis for Treatment Effects in DiD Models using Riesz Representation" ([arXiv 2510.09064](https://arxiv.org/pdf/2510.09064)).

### Inferences

#### Card 7A — robust Durbin–Wu–Hausman (control-function form)
```
Stage 1: d = π z + W'δ + v  → v̂   (same W, FE as the structural equation)
Stage 2: y = β d + W'γ + ρ v̂ + e    (OLS for ln y; PPML for counts)
Test:    t = ρ̂ / SE_HAC/cluster(ρ̂);  reject exogeneity if |t| > t_crit   (k endogenous → Wald on ρ vector)
Classical form (homoskedastic only): H = (β̂_IV − β̂_OLS)² / (V̂_IV − V̂_OLS) ~ χ²₁
```
- **Interpretation rules:**
  - Non-rejection with a weak instrument (F_eff < 10) is uninformative.
  - Never use the test outcome to switch to OLS (pretest bias). Report both, with tier per §6.
  - Rejection plus a credible instrument means the IV estimate (with AR set) supersedes OLS and DML.
- **Without an instrument, use falsification diagnostics instead:**
  - *Timing / reverse-causality test:* do demand surprises at t−k (residuals from the calendar model) predict regular-price changes at t? A significant effect means prices respond to demand, so require designs that difference out pre-change demand (leads, donut) and lower the tier of selection-on-observables estimates.
  - *Placebo outcome:* the "effect" of product A's price change on unrelated categories' sales should be ≈ 0.
  - *Placebo dates* (4A).
- **Test:** under an exogenous DGP the rejection rate should be ≈ α (with HAC); under an endogenous DGP with a strong instrument, power should approach 1.

#### Card 7B — Cinelli–Hazlett for OLS / TWFE-OLS / linearized estimates
```
Inputs: t = β̂/SE (use HAC/cluster SE), df = residual degrees of freedom (with HAC: use effective df, e.g. EWC ν — my approximation)
Partial R² of treatment with outcome:  R²_{Y~D|X} = t² / (t² + df)
f = t/√df ;  f_q = q·|f|
RV_q = ½·(√(f_q⁴ + 4f_q²) − f_q²)
RV_{q,α}: f* = f_q − |t_{α/2, df−1}|/√(df−1);  RV_{q,α} = 0 if f* ≤ 0 else ½(√(f*⁴ + 4f*²) − f*²)   (sensemakr adds an edge-case branch when f_q is large — port from source)
Bias bound for hypothesized (R²_{Y~Z|D,X}, R²_{D~Z|X}):
   |bias| ≤ SE(β̂)·√df·√( R²_{Y~Z|D,X}·R²_{D~Z|X} / (1 − R²_{D~Z|X}) )
Decision-threshold version: to move β̂ to β*, use q = (β̂ − β*)/β̂ (same sign required)
```
- **Own check (the test fixture):** my implementation reproduces Cinelli–Hazlett's published Darfur example from t = 4.18, df = 783: R²_{Y~D|X} = 0.0218, RV_{q=1} = 0.1386, RV_{q=1, α=0.05} = 0.0761 (published: 2.2%, 13.9%, 7.6%).
- **Elasticity framing:** with β̂ = −1.8, SE = 0.4 and margin 40%, β* = −2.5, so the question is whether confounding could make the true β more negative than −2.5 and reverse a "raise price" decision. Compute RV for q = (β̂ − β*)/β̂ in the needed direction.
- **Benchmarking:** express the confounder's strength as k× the explanatory power of an observed covariate group (e.g. "a confounder as strong as the entire seasonal block"), using the sensemakr benchmark formulas (port from source; not reproduced here).

#### Card 7C — DML sensitivity (DoubleML)
- Call `dml.sensitivity_analysis(cf_y, cf_d, rho=1.0, null_hypothesis=β*)` and report bounds, `rv` and `rva`.
- Defaults to show: cf_y = cf_d = 0.03, plus a benchmark calibrated by refitting with one observed covariate group dropped. The gains in R² from that group set the scale for cf_y and cf_d.
- **Caveat:** DoubleML computes these from iid-type score variances. Under serial dependence, the `rva` (CI-based) value is too optimistic, so recompute the CI part with DK/HAC (my inference).

#### Card 7D — E-values (limited use)
- Valid only for binary contrasts, e.g. post vs pre a +10% change, or arm H vs L. Convert the elasticity to a risk ratio on expected quantity: `RR = exp(β̂·Δ)`; invert it if below 1.
- Example: β̂ = −1.5, Δ = ln 1.1 gives RR = 0.867 → 1/RR = 1.154 → **E = 1.154 + √(1.154 × 0.154) ≈ 1.58** (computed). A confounder associated with both the price change and demand by a risk ratio of 1.58 could explain the effect away.
- For continuous log price, the partial-R² approaches (7B/7C) are the natural parameterization. Do not show E-values for those.

#### Tests that catch a wrong implementation
1. The Darfur reproduction above.
2. DoubleML `sensitivity_analysis` with cf_y = cf_d = 0 must reproduce θ̂ and the CI exactly.
3. With a simulated omitted confounder of known partial R², the true bias must fall inside the bounds at the matching cf values.
4. DWH size and power simulations (7A).

### Gaps
- How partial-R² sensitivity should treat HAC or effective degrees of freedom under serial correlation is not settled in the sources I checked; the effective-df substitution is my approximation.
- I did not port or verify the sensemakr benchmark-bound formulas or the edge-case branch of RV_{q,α}.
- E-value conversions for continuous treatments were recalled from memory, not re-verified.
