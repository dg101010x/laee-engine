# Gate every elasticity on real price variation

LAEE should stop treating elasticity as something an estimator ensemble can always produce. It should treat it as something a business's data either identifies or does not. The v2 algorithm therefore needs six changes. First, **Poisson pseudo-likelihood (PPML) and event-study designs around discrete regular-price changes** become the default estimators, with randomization inference over placebo dates. Second, a **closed-form hierarchical Bayesian layer** ships in V1 and pools industry, country, market, business and product evidence with honest uncertainty. Third, **a design-based identification tier** decides what the product may show and how far each estimate may move the prior. Fourth, censored-demand, regular-price and calendar handling come *before* estimation. Fifth, markets are **estimated spatial-choice kernels** built from open African data, not radii. Sixth, the optimizer becomes **risk-constrained** (loss-averse utility, a 20% cap on the probability of loss, a CVaR budget) and can answer CHANGE, HOLD or TEST. Most of this matters because SMB data are thin: one product selling 4 units a day, with a single 10% price change, gives an elasticity standard error of roughly 2. For most SMB products the honest output is therefore a prior-based range plus a recommended experiment, not "−1.43 ± 0.3". The v1 specification names reasonable method families, but as written it would produce confidently wrong numbers. Its log demand is undefined on zero-sales days. Its DML uses random folds and post-treatment controls. Its ensemble weights reward forecast accuracy rather than causal validity. Its regime mixture averages elasticities rather than demand curves. Its confidence scores refer to no defined event. And its LLM is asked for exactly the predictive judgments that §2 forbids. The evidence on African shop-level elasticities is close to nonexistent, so every default below is a starting value that must be calibrated on platform data and experiments.

## v1 picks plausible tools but would ship confidently wrong numbers

The most damaging v1 defects are not missing features. They are prescriptions that actively bias estimates or make uncertainty meaningless, and each has a concrete, testable replacement. The table collects the critical ones; later sections give the details.

| v1 § | What v1 says | Why it is wrong | v2 replacement |
|---|---|---|---|
| §14, §16 | Model ln Q with OLS/DML | Undefined at zero sales. log(1+Q) gives unit-dependent effects ([Chen & Roth 2024](https://academic.oup.com/qje/article-abstract/139/2/891/7473710)). Log-OLS is inconsistent under heteroskedasticity ([Santos Silva & Tenreyro](https://personal.lse.ac.uk/tenreyro/jensen08k.pdf)) | PPML / NB2 on counts with product fixed effects; log-OLS only for weekly aggregates with ≤5% zero weeks |
| §14 | TWFE as baseline | TWFE elasticities are weighted sums of effects with possibly negative weights ([de Chaisemartin & D'Haultfœuille 2020](https://www.aeaweb.org/articles?id=10.1257%2Faer.20181169)). With continuous doses, TWFE mixes level and slope effects ([Callaway, Goodman-Bacon & Sant'Anna](https://www.nber.org/system/files/working_papers/w32117/w32117.pdf)). Day fixed effects absorb menu-wide repricing | Diagnostic only, always with a weight report; category×week fixed effects; heterogeneity-robust event studies |
| §16 | Random K-fold DML cross-fitting | Random folds leak information across serially correlated days. The theory calls for blocked, neighbors-left-out folds ([Semenova et al. 2023](https://www.econometricsociety.org/publications/quantitative-economics/2023/05/01/Inference-on-Heterogeneous-Treatment-Effects-in-High-Dimensional-Dynamic-Panels-under-Weak-Dependence)) | Time-blocked folds with gaps; HAC, Driscoll–Kraay or randomization inference; residual-price-variation gate |
| §17 | Nuisance features include same-day promotion, inventory, stockout, traffic, rolling demand, lagged demand, time FE, cost | These are post-treatment variables, mediators, outcome leakage, or instruments wrongly used as controls | Feature lint that refuses them; costs become instruments |
| §20 | 30–365-day rolling windows combined with "validated" weights | Most 30-day windows contain no price change, so the slope is undefined. Five windows are listed but only four combined. The target is never observed | Kalman random-walk elasticity on residualized data; only admissible windows count |
| §21 | E[ε] = Σ P(r)·ε_r over 10 regimes | Averages slopes of E[log Q] rather than E[Q]. Most of the 10 regimes are observed categories, not latent states. Smoothed probabilities leak future data | Observed categories become covariates; at most 2–3 latent regimes; filtered probabilities only; mix **demand curves** in the simulator |
| §29–30 | Competitor-entry DiD for elasticity | Entry shifts demand directly, so it identifies competition effects, not own-price elasticity. Staggered TWFE is biased. A single business has no controls | Platform-level ring event study ([Butts](https://www.kylebutts.com/papers/did-rings/)) feeding priors |
| §39 | Conformal intervals for elasticity | Conformal prediction needs observed targets, and the elasticity is never observed | Parameter intervals from sandwich, wild-cluster or block bootstrap, or the posterior. Conformal is used only for observable outcomes |
| §40, §86 | Ensemble weights from out-of-sample predictive accuracy; a gradient-boosting demand model as a member | Forecast accuracy says nothing about causal slope. Members are highly correlated. The hierarchical model double-counts the data | Tier-based selection; a bias-aware hierarchical combiner; joint bootstrap; GBM used only as nuisance/simulator |
| §41–42, §85 | One-line prior, deferred to V2 | No sampling variance, no τ estimator, no prior variance. Shrinkage is deferred even though it is the cheapest fix for small data | Two-stage Gaussian tree with integrated τ, shipped in V1 |
| §43–45 | DQS/IDS/Confidence = unspecified f(·); "Confidence 87%" | Untestable, with triple counting of sample size | A predicted standard error, a design tier with information shares, and calibrated probabilities of named events |
| §48 | MAPE/sMAPE; regret against an unknowable P* | MAPE is undefined at zero sales, and minimizing expected regret is the same as maximizing expected profit, which cancels §38's risk aversion | RMSSE, Poisson deviance and bias; field regret against a randomized hold-out |
| §34, §37–38 | Point-ε constant-elasticity simulation over ±20%; λ·Var penalty | Ignores uncertainty. Biased toward recommending moves. λ·Var changes the decision when the currency unit changes | Posterior-predictive simulation within the observed price support; loss-averse utility plus chance and CVaR constraints |
| §66 | `cost*1.10` as margin floor | That is a 9.09% margin, not 10%. No handling of an empty feasible set; rounding after optimization can break constraints | Margin on net price; a discrete feasible set built before optimizing; an INFEASIBLE status |
| §77–78 | 10/90 customer split; "a 5% test would substantially improve the estimate" | 10/90 needs 2.78× the sample of 50/50. Customer-level prices are often infeasible or unlawful. A 5% test barely moves the posterior at SMB volumes | Time-based switchbacks with larger price gaps, chosen by expected value of sample information |
| §2, §23–28 | Llama outputs severity, direction, radius, coordinates and confidence; one NetShock index | Contradicts §2 itself. LLM self-reported confidence clusters at 80–100% ([Xiong et al.](https://arxiv.org/abs/2306.13063)). One index mixes opposite-signed channels | Grounded extraction only; per-type exposure features; kernels estimated from outcomes |

Several cross-cutting omissions run through v1. It never defines the **estimand**: which elasticity, at which price, over which horizon, and whether regular or promotional. It never says whether price is the posted price or revenue ÷ units. It has no minimum-data rules, no test that would catch a wrong implementation, and no African-specific inputs: payday cycles, Hijri holidays, load-shedding, informal competitors, cash price points, or languages Llama does not support. Symbols collide; for example, W_ij means four different things across §7, §8, §10 and §32.

## Identification comes from price-change events, not estimator choice

### Define the estimand and the price before estimating anything

The spec should name a **canonical estimand**: the shop-level own-price elasticity of expected quantity, ∂ln E[Q]/∂ln P, at the current **regular** posted price, as a 4–12-week response. Promotional (temporary-cut) elasticities and short-run (0–2 week) responses are reported separately and never pooled with it. This matters because static estimates built on temporary cuts overstate own-price elasticities by about 30% ([Hendel & Nevo 2006](https://www.econometricsociety.org/publications/econometrica/2006/11/01/measuring-implications-sales-and-consumer-inventory-behavior)). In one grocery field experiment, the mean randomized elasticity was −0.34, against roughly −2.0 from observational variation ([Bray et al., Unjournal summary](https://unjournal.pubpub.org/pub/evalsumbraybray)). The treatment variable must be the **posted price** from the item master or menu, or the daily modal transaction price. It must never be revenue ÷ units on the same rows that supply Q. Recording noise then enters both sides of the regression and pulls the slope toward −1 (division bias, [NBER w26066](https://www.nber.org/system/files/working_papers/w26066/w26066.pdf)). In the team's simulation at SMB-scale price variation, just 5% unit-recording noise moved a true −1.5 to about −1.28.

Under heterogeneity, every pooled estimator is a weighted average, and the weights should be stated. Partialling-out DML converges to a **Var(price | X)-weighted** average of local elasticities, so it over-weights products and periods with more price variation. That is not a revenue-weighted business elasticity. Product-level or revenue-weighted summaries must be averaged explicitly from conditional estimates.

### Make PPML event studies around regular price changes the workhorse

For daily counts the default static model is **PPML with product fixed effects and category×week time effects**: E[Q_jt] = exp(α_j + γ_c(j),w(t) + β·ln P_jt + X'δ). Here X holds recurring calendar terms, weather and *lagged* competitor prices. PPML is consistent when the mean is correctly specified, handles zeros, and has no incidental-parameter problem with unit fixed effects. pyfixest's `fepois` provides it with a separation check ([pyfixest](https://github.com/py-econometrics/pyfixest)). The team's simulations show the stakes. At 0.5 units/day (60% zero days) and a true elasticity of −1.5:

| Estimator | Estimate |
|---|---|
| OLS on log(1+Q) | −0.45 |
| Same, with Q rescaled ×100 | −2.36 |
| OLS on log Q, zeros dropped | −0.29 |
| PPML | −1.52 |

ETL must also sort zeros into three kinds. Structural zeros (store closed, item not listed) are dropped. Censored zeros (stockouts) enter a censored likelihood. True zeros are kept.

The default single-business design is a **regression discontinuity in time / event study** around discrete regular-price changes. The event and model specification is:

- **Qualifying event:** |Δln P| ≥ 0.03, new price held ≥ 21 days, and no promotion, stockout, supplier or pack-size change within ±7 days.
- **Window:** ±28 days (±56 if mean sales are below 2/day), with a ±2-day donut around the change.
- **PPML terms:** day-of-week, holiday and payday terms, plus one trend slope per event.
- **Inference:** primarily **randomization over placebo dates** (999 pseudo-dates from windows with no price change), inverted for a confidence interval.

Randomization inference is primary because asymptotic HAC methods under-cover when the regressor is a persistent step function. In the team's simulation (four step changes, AR(1) errors with ρ = 0.7), Newey–West intervals covered the truth 87.6% of the time, equal-weighted-cosine intervals 90.7%, and placebo-path randomization inference 96.5%. When few products change price, Conley–Taber inference built from non-changers' residuals is the analogue ([Conley & Taber 2011](https://ideas.repec.org/a/tpr/restat/v93y2011i1p113-125.html)).

When non-substitute comparison products, or the same SKU at other tenants, are available, the design upgrades to a heterogeneity-robust DiD. Options include Borusyak–Jaravel–Spiess imputation ([REStud 2024](https://academic.oup.com/restud/article/91/6/3253/7601390)), Callaway–Sant'Anna or Sun–Abraham for staggered first changes, and the de Chaisemartin–Pasquier–Vazquez-Bare switcher/stayer slopes for repeated continuous price changes ([SSRN 4011782](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4011782)). The elasticity is the Wald-DiD ratio ΣωATT/ΣωΔ. Controls must be tested for substitution spillovers, because a rise in the control product's sales after the treated product's price rise inflates |ε|. Pre-trend tests have low power, and conditioning on passing them distorts inference ([Roth 2022](https://ouci.dntb.gov.ua/en/works/73oo8yX7/)). v2 should therefore report a HonestDiD-style breakdown value rather than a pass/fail pre-test.

Before estimating, the system should project precision from the design: SE ≈ √(2·LRV/(W·ΣΔ²)). One event at 4 units/day with a 10% change and a 28-day window gives **SE ≈ 2.3**. Ten such events give about 0.73. **If the projected SE exceeds 1.0, no own-data elasticity is shown**, only the prior and an experiment proposal.

### Keep DML, but only with strict hygiene and as a robustness layer

DML is selection on observables with flexible controls. It cannot create identifying variation. It is credible for SMB data only under all of the following conditions:

- **Model:** partialling-out PLR on posted log price, with recurring rather than time-local features.
- **Folds:** time-blocked cross-fitting folds shared by all products. Use K = 5 blocks of at least 60 days, with a gap g = max(14, longest feature look-back + 7, AR-implied decay) removed on each side. Repeat over S = 10 random block offsets and aggregate with the original rule σ̃² = median[σ̂_s² + (θ̂_s − θ̃)²].
- **Variance:** HAC, Driscoll–Kraay or block-cluster; randomization inference when there are fewer than about 10 events.
- **Gates:** residual price variance share RVS = ΣṼ²/Σ(D − D̄)² between 0.05 and 1.0. An RVS above 1 signals nuisance extrapolation. Also require sd(Ṽ) ≥ 0.02 and out-of-fold R² of the price model ≤ 0.95.

Library behavior matters here. DoubleML's built-in resampling is random KFold, or KFold over shuffled cluster ids, and never builds contiguous blocks, so blocked splits must be injected with `set_sample_splitting`. Its repeated-split aggregation takes the median of CI bounds rather than the 2018 rule ([DoubleML source](https://github.com/DoubleML/doubleml-for-py)). EconML's final-stage covariance supports only nonrobust/HC0/HC1, and its `groups=` argument changes the folds but does **not** cluster the standard errors ([EconML source](https://github.com/py-why/EconML)). For panels, Clarke & Polselli recommend first-differencing with ensemble learners ([Econometrics Journal 2026](https://ideas.repec.org/a/oup/emjrnl/v29y2026i1p69-86..html)). Learners should be short-stacked with non-negative weights ([Ahrens et al. 2025](https://ideas.repec.org/a/wly/japmet/v40y2025i3p249-269.html)) and selected only on out-of-fold nuisance loss, never on θ̂. Below about 1,000 rows or about 20 price regimes, only ridge on a recurring basis should remain.

The team's simulations explain why. With a raw time index or lagged and rolling demand among the features, DML averaged about −0.75 against a true −1.5. Even clean seasonal features with gradient boosting learned spurious price–season coincidences from four price regimes, and 95% coverage fell to 42–68%. Adding ln Q_{t−1} as a control without lagged price returned −0.49, close to the analytic β(1−ρ). The spec therefore needs a **feature-lint rule** that refuses all of the following in elasticity nuisance models:

- same-day promotion, display or coupon flags;
- same-day or end-of-day inventory and stockout flags;
- foot traffic and transaction counts;
- same-day competitor prices;
- rolling windows that include day t;
- lagged demand without lagged price;
- date indices as tree features;
- supplier cost, FX or tax variables.

If dynamics are needed, use an ARDL with both lagged quantity and lagged price, and report the long-run elasticity.

### Use costs and taxes as instruments, with weak-IV-robust inference

The credible SMB instruments are product-specific cost shifters, with date fixed effects soaking up common demand:

- supplier invoice changes, which makes purchase records the most valuable optional data to collect at onboarding;
- commodity, FX or fuel shocks interacted with each product's import share, a shift-share design (but in emerging markets FX shocks also cut real incomes, so exposure can violate exclusion);
- product-specific VAT, excise or sugar-levy changes.

Hausman instruments (the same SKU's price at tenants more than 50 km away) need multi-tenant data and fail under common demand shocks. Competitor entry, weather and events are demand shifters, **never** instruments. Inference must not rely on 2SLS t-tests. A nominal 5% t-test needs a first-stage F above 104.7 ([Lee, McCrary, Moreira & Porter 2022](https://www.nber.org/system/files/working_papers/w29124/revisions/w29124.rev0.pdf)), and Anderson–Rubin is recommended even with strong instruments ([Keane & Neal 2024](https://www.annualreviews.org/content/journals/10.1146/annurev-economics-092123-111021)). v2 should compute the effective F ([Montiel Olea & Pflueger](https://joseluismontielolea.com/Montiel-OleaJBES.pdf); conservative critical value 23.1) and the closed-form Anderson–Rubin set, then act on F_eff:

| F_eff | What to report |
|---|---|
| Below 10 | The AR set only |
| 10 to 23.1 | AR as the primary interval |
| 23.1 or above | Wald interval with tF-adjusted critical values, plus AR |

One trap must be guarded in code review: linearmodels' `anderson_rubin()` is the LIML overidentification test, not the weak-IV-robust AR test ([linearmodels source](https://github.com/bashtage/linearmodels)). For counts, use control-function PPML with a joint block bootstrap. The t-statistic on the first-stage residual doubles as a robust Durbin–Wu–Hausman test, but a non-rejection must never be used to switch back to OLS.

### Report sensitivity against the decision threshold, not against zero

Every observational estimate should carry an omitted-variable robustness value: Cinelli–Hazlett partial-R² bounds for OLS-type estimates ([JRSS-B 2020](https://academic.oup.com/jrsssb/article-abstract/82/1/39/7056023)), and the Chernozhukov et al. DML bounds via DoubleML's `sensitivity_analysis` ([NBER w30302](https://www.nber.org/papers/w30302)). The bounds should be computed against the **decision-relevant null**: β* = −1/margin for profit decisions, or −1 for revenue. The question is whether confounding could flip "raise price" to "cut price", not whether the sign could flip. Because DoubleML computes these from iid-type score variances, the CI-based value should be recomputed with HAC. E-values fit only binary contrasts, such as a single +10% change.

### Replace the 7-tier estimator ranking with design tiers that earn their place

v1 §76 ranks *estimators*. DML sits above fixed effects even though both rest on the same selection-on-observables assumption, and the most common SMB design (an event study around a regular price change) is missing. v2 should rank **designs**, each gated by observable diagnostics:

| Tier | Design | Gates (an estimate gets the highest tier whose gates all pass) |
|---|---|---|
| 1 | Randomized price test (switchback, product-level, staggered rollout) | Randomization log verified; ≥ 8 blocks per arm; carryover handled; randomization-inference p reported |
| 2 | Documented exogenous driver (supplier, tax, RRP) with comparisons, or strong IV | Driver documented; HonestDiD M̄* ≥ 1; p_RI < 0.1; or F_eff ≥ 23.1 with a bounded AR set |
| 3 | Multi-event robust DiD/event study with clean controls | ≥ 3 events; spillover test passed; M̄* ≥ 0.5; robust estimator, not TWFE |
| 4 | Single-series RDiT, or moderate IV | ≥ 2 events; leads ≈ 0; p_RI < 0.2; or 10 ≤ F_eff < 23.1 |
| 5 | Selection on observables (PLR-DML, PPML-FE) | ≥ 4 events or RVS ≥ 0.10; time-aware folds; robustness value ≥ 0.05 against β* |
| 6 | Weak observational | Some variation, but tier-5 gates fail |
| 7 | No variation | **Prior only**, labeled "not estimated from your data" |

The tier does two jobs. It sets **display rules**: own-data elasticity only at tier ≤ 5 with projected SE ≤ 1. It also sets a **bias variance** added to the likelihood when the estimate updates the hierarchical prior, with placeholder τ_tier = 0, 0.15, 0.25, 0.4, 0.5, 0.8 and ∞ for tiers 1–7. With that bias term, a tier-6 estimate with SE 0.3 moves the prior only as much as an SE ≈ 0.85 estimate would.

A **data-condition router** replaces v1 §86's fixed ensemble:

| Data condition | Estimator |
|---|---|
| Randomized test | PPML arm contrast plus Fisher randomization inference |
| Documented cost or tax changes | Event study plus AR-IV |
| ≥ 3 regular changes with clean controls | BJS, Callaway–Sant'Anna, Sun–Abraham or dCDH |
| 1–2 changes | RDiT-PPML plus placebo randomization inference |
| Many small changes with rich covariates | Gated PLR-DML |
| Only promotions | A separate promotional elasticity, never relabeled as regular |
| No variation | Prior plus experiment |

There is no published scoring system for identification strength. The tiers and the continuous IDS (base tier score × the geometric mean of sub-scores for events, variation, pre-trends, placebo, sensitivity, agreement and precision) are a proposal. They should be calibrated by regressing |β̂_obs − β̂_exp| on the diagnostics wherever both observational and experimental estimates exist.

### Experiments should be time-based switchbacks with real power calculations

Customer-level 10/90 posted-price splits are operationally infeasible in shops, carry fairness and disclosure risk (the EU requires disclosure of algorithmically personalized prices, [Directive 2019/2161](https://eur-lex.europa.eu/eli/dir/2019/2161/oj)), and need 2.78× the sample of a 50/50 design. The workable gold standard is a **switchback**: the same posted price for everyone at a given time, randomized across time blocks, analyzed with randomization p-values ([Bojinov, Simchi-Levi & Zhao 2023](https://pubsonline.informs.org/doi/10.1287/mnsc.2022.4583)). Defaults:

- 7-day blocks, which balance day-of-week;
- 1–2 washout days after each switch;
- ≥ 8 blocks per arm;
- Poisson GLM analysis with block and weekday effects;
- a carryover test on the previous block's arm.

Power is sobering. In the team's verified formula, Var(β̂) ≈ (φ·DEFF/Δ²)(1/(n_H μ_H) + 1/(n_L μ_L)), a +10% test at 4 units/day for 120 days gives SE ≈ 1.0, and separating −1.5 from −1.0 with 80% power would take about 3,700 days. The remedies are larger gaps (+15–20%), pooling across similar products, and treating the test as an update to a hierarchical prior. A cross-product randomized crossover is an alternative: pair products across categories and randomize which one in each pair gets the higher price each week. Coupons sent by SMS or WhatsApp are the only defensible form of customer-level variation. No peer-reviewed SMB field price experiments from Africa were found, and jurisdiction-specific law (Kenya, Nigeria, South Africa, Ghana) needs counsel before experimentation is enabled.

## Sparse data makes hierarchical shrinkage the core, not a V2 add-on

### A two-stage Gaussian tree replaces the one-line prior

v1 §41's ε_i ~ N(μ, σ²) conflates two different variances: the sampling noise s_i² of a business's own estimate and the true between-business spread τ². Without s_i, "update with the business's own data" cannot be computed. v2 should use a **two-stage design**. Inside the tenant boundary, the router produces exactly one (b, s) per business-product per estimand, plus tier, n_eff and number of price spells. At the population level, an exact Gaussian tree pools: industry → country → region → market → business → product. Each level has its own variance τ_ℓ, the root is a literature hyperprior, and the likelihood carries the tier bias term (b ~ N(θ, s² + δ_tier²)).

The upward and downward message passes are closed-form, O(N) numpy. In the team's tests they matched dense GLS to below 1e-15 and the multivariate-normal log-likelihood exactly. A node with no data automatically receives the parent posterior plus τ², which is the cold-start predictive prior. Computing the joint posterior in one shot, or caching priors **leave-one-tenant-out**, avoids the double counting that v1 §79's flywheel would otherwise create when a business's own data are in the population snapshot used as its prior.

τ must be estimated **once per level per industry**, never per small cell:

- **Estimator by number of groups:** REML for meta-regression and for ≥ 10 groups; Paule–Mandel for 3–9 groups; DerSimonian–Laird only as a starting value. This follows [Veroniki et al. 2016](https://onlinelibrary.wiley.com/doi/full/10.1002/jrsm.1164), who favor PM and REML.
- **Regularize away from zero:** penalized ML-II, so τ is not degenerately estimated at zero ([Chung et al. 2013](https://link.springer.com/article/10.1007/s11336-013-9328-2)).
- **Priors on τ:** HalfNormal, with scales σ_R = σ_M = 0.4, σ_B = 0.7 and σ_P = 0.8 elasticity units to start.

Plug-in empirical Bayes understates uncertainty. Morris introduced a (k−3)/(k−1) correction for exactly this reason ([Morris 1983](https://pmc.ncbi.nlm.nih.gov/articles/PMC2872278/)), and Kass–Steffey give a first-order variance correction ([JASA 1989](https://doi.org/10.1080/01621459.1989.10478825)). In the team's coverage simulation:

- Plug-in REML 95% intervals covered only 83.5–87.8% with 8–30 businesses at τ = 0.5. REML hit τ̂ = 0 in 27.8% of 8-business cells.
- **Grid integration over τ** (600–800 points with quadrature weights, BDA3-style, [Gelman et al.](http://www.stat.columbia.edu/~gelman/book/)) restored 93–96% coverage while still cutting interval width by 23–46% relative to no pooling.

That grid is the production default for business-level intervals. Omitting the Δτ quadrature weights on a non-uniform grid is a real bug, which dropped coverage to about 87%, and it should be a unit test.

### Seed the prior from literature, then let platform data dominate

Published magnitudes bracket what a single shop faces:

- **Brand or firm level:** meta-analytic means of −1.76 ([Tellis 1988](https://www.jstor.org/stable/3172944)) and −2.62 across 1,851 estimates ([Bijmolt, van Heerde & Pieters 2005](https://journals.sagepub.com/doi/10.1509/jmkr.42.2.141.62296)). Bijmolt et al. find that correcting for endogeneity **increases** magnitude, so priors trained on naive observational estimates inherit attenuation bias.
- **Category level:** much smaller. US food categories range from 0.27 to 0.81 in absolute value ([Andreyeva et al. 2010](https://scholars.duke.edu/display/pub963616)).

Store-level elasticities vary a lot. Eleven trading-area variables explain 67% of the variation across 83 supermarkets, with demographics mattering more than competition ([Hoch et al. 1995](https://www.researchgate.net/publication/237130017_Determinants_of_Store-Level_Price_Elasticity)). Elasticities fall in magnitude with own-store income, yet chains price nearly uniformly and forgo about $16m a year at the median ([DellaVigna & Gentzkow](https://www.nber.org/system/files/working_papers/w23996/revisions/w23996.rev0.pdf)).

African evidence exists only at household or category level:

- **Low-income countries:** cereals −0.61 vs −0.43 in high-income countries ([Green et al. 2013](https://pubmed.ncbi.nlm.nih.gov/23775799/)).
- **Nigeria:** household own-price elasticities of −0.63 to −1.79, with local rice about −1.69 ([J. Applied Econ.](https://www.tandfonline.com/doi/full/10.1080/15140326.2021.1980351)).
- **South Africa:** carbonated soft drinks −1.18 and sugar −2.42 ([Stacey et al. 2017](https://www.sciencedirect.com/science/article/pii/S0091743517301871)).
- **Kenya:** negative own-price elasticities for every food group ([Korir et al. 2020](https://ideas.repec.org/a/eee/ecmode/v92y2020icp99-108.html)).
- **Mobile services:** inelastic, around −0.2 short-run and −0.8 long-run ([UCT](https://open.uct.ac.za/server/api/core/bitstreams/e9df6445-af7f-40a1-b82c-0e227850a952/content)).

**No peer-reviewed shop-level elasticity for a kiosk, duka, spaza or open-market trader was found.** Category numbers are a magnitude *lower bound* for a shop's own-price elasticity, because neighbors sell the same sachet. Credit "book" accounts, proximity and daily small-pack purchasing push the other way, with unknown net effect.

The defaults should therefore be wide engineering placeholders, overwritten once platform data exist:

| Category | Prior mean | Prior SD |
|---|---|---|
| Packaged FMCG | −1.8 | 0.8 |
| LMIC informal staples | −2.0 | 1.0 |
| Café/restaurant item | −1.4 | 0.8 |
| Pharmacy/OTC | −1.2 | 0.8 |
| Salons and auto repair | −1.3 | 0.9 |
| Fuel station | −3.0 | 1.5 |

Each also gets a predictive τ ≈ 0.7. With these values, P(ε > 0) stays at about 2–15%. Airtime and data resellers sell at operator-set face value and should be excluded from price optimization.

v1 §42's `f(...)` becomes a concrete **meta-analytic-predictive meta-regression**:

- **Intercepts and category effects:** literature-anchored industry intercepts, plus a category effect δ_cat ~ N(0, 0.5²).
- **Weak, sign-informed slopes:**
  - log income, toward zero at higher income, about +0.10·|m| per log unit. This is derived from the Green et al. cross-country ratio, so a 2–3× within-city income gap moves |ε| by only about 7–11% at the prior.
  - travel-time-weighted competition, more elastic, −0.10·|m| per SD;
  - density, only after orthogonalizing it against competition;
  - price tier.
- **Identification-tier shift:** prediction is made at the best tier, which corrects Bijmolt's attenuation.
- **New-business prior:** N(x*ᵀβ̂, τ² + x*ᵀV_βx*), plus a τ_shift² inflation outside the training hull.
- **Data rule:** at least 10 businesses per covariate.
- **Release gate:** leave-one-business-out z-scores with mean in [−0.1, 0.1], SD in [0.9, 1.1], and 80/95% coverage within ±2 points, checked by country, income tercile and tier. A prior SD below 0.5 for a business with no data is almost certainly overconfident.

### The posterior should move with information, not transaction counts

"50 transactions" (v1 §41) is the wrong unit. With zero price variation, 10,000 transactions carry zero information about elasticity. Information scales with n·Var(ln p) and the number of independent price spells. In the team's calculations at daily noise σ_e = 0.5, own data reach about 64% weight after a year at ±5% price variation, but in about 3 months at ±10%. **Doubling price variation is worth quadrupling the observations.** The product should show "Based X% on your own data", where X = 100·(1 − B) and B is the shrinkage factor. With fewer than 4 price spells, s² is floored at 3s² and labeled "early signal – not used for recommendations". When sd(ln p) < 0.02, the output reads "not measurable yet".

Prior-only output must be presented as a benchmark, for example: "−1.8, likely −3.5 to −0.1, from 46 similar cafés in Nairobi; not yet measured for your business". It should be capped at "low" confidence and allowed at most an exploratory step of 5% or less. When the data conflict with the prior (p < 0.01 on the prior-predictive z-score), the prior switches to a robust MAP mixture, 0.8·N(m, v) + 0.2·N(m_parent, 2²) ([Schmidli et al. 2014](https://onlinelibrary.wiley.com/doi/abs/10.1111/biom.12242)). Positive shop-level estimates are treated as identification failures unless a Tier 1–2 replication says otherwise. They are pooled on the raw scale, never truncated before pooling. Genuine Giffen behavior is confined to subsistence staples among the very poor ([Jensen & Miller 2008](https://doi.org/10.1257/aer.98.4.1553)), and a shop with substitutes next door is not that setting.

### Multi-tenant learning needs only summaries, plus disclosure control

Only per-business stage-1 summaries should cross the tenant boundary: (b, s, n_eff, number of price spells, tier, estimand id, taxonomy keys, and location keys at H3 resolution 7 or coarser). For Gaussian hierarchies these summaries are near-sufficient. Two-stage pooling is asymptotically as efficient as pooling raw data for common effects ([Lin & Zeng 2010](https://doi.org/10.1093/biomet/asq006)), so v1's "federated-style" iteration is unnecessary.

Released aggregates need disclosure control, because in a two-business cell one business's leave-one-out prior *is* its competitor's elasticity. Defaults:

- a k_min of 10 businesses from at least 5 tenants before a cell is displayed;
- a 50% cap on any one business's precision weight;
- monthly batching, with values rounded to 0.1;
- Laplace differential privacy for external benchmarks ([Dwork et al. 2006](https://doi.org/10.1007/11681878_14)).

Below about 20 businesses per cell, ε = 1 noise of SD ≥ 0.5 swamps a τ of about 0.7. The legal basis for cross-tenant learning under Kenya's DPA 2019, Nigeria's NDPA 2023 and South Africa's POPIA was not researched and needs counsel.

## Heterogeneity, time variation and confidence must be gated and calibrated

### Default to linear-interaction DML; let causal forests earn their way in

For continuous price, GRF's causal forest, EconML's CausalForestDML, the R-learner and LinearDML all solve the same residual-on-residual moment. In GRF's C++ core, the forest's local elasticity is literally a forest-weighted OLS slope of Ỹ on T̃ ([skgrf, which vendors the GRF core](https://pypi.org/project/skgrf/0.3.0/)). A forest therefore learns nothing where leaves lack price variation. v1's listed heterogeneity variables (income, density, business size) are constant within one business, so heterogeneity along them can only be learned across tenants.

A typical product with 200 usable days, 5% residual price variation and σ_u = 0.6 has an elasticity SE of about 0.85 from its own data. The default should therefore be **pooled-across-products LinearDML with at most five pre-registered interactions**:

- price-tier z-score;
- weekend;
- peak season;
- competitor-price-index z-score;
- category group.

Folds are GroupKFold on product × 28-day blocks, with K = 5 rather than EconML's default cv = 2. Inference uses CR1 cluster errors, a wild cluster bootstrap when there are fewer than about 30 clusters, and the median over 5 repeated splits. Product-level elasticities then come from **empirical-Bayes shrinkage** of per-product slopes toward a random-effects meta-regression on product attributes.

Causal forests are gated: at least 2,000 pooled rows and at least 50 independent price changes across at least 20 products. Settings differ from the defaults:

| Setting | v2 value | EconML default |
|---|---|---|
| n_estimators | 2,000 | 100 (only 25 little bags for variance) |
| min_samples_leaf | 20 | 5 |
| min_var_fraction_leaf | 0.05 | — |
| honest | True | True |

Keep honesty on, because it buys interval coverage in small samples ([Hou 2025](https://arxiv.org/pdf/2506.13107)). A forest's heterogeneity reaches users only if it passes three gates:

1. It beats both LinearDML and a constant model on held-out R-loss by more than 2 paired-bootstrap SEs ([Nie & Wager](https://www.gsb.stanford.edu/faculty-research/publications/quasi-oracle-estimation-heterogeneous-treatment-effects); EconML's RScorer, where a negative score "hints at overfitting", [EconML 0.17](https://pypi.org/project/econml/0.17.0/)).
2. A continuous-treatment calibration regression shows a positive heterogeneity coefficient with a CI covering 1.
3. GATES-style tertiles are correctly ordered ([Chernozhukov, Demirer, Duflo & Fernández-Val 2025](https://onlinelibrary.wiley.com/doi/full/10.3982/ECTA19303)).

DR-learners and orthogonal random forests suit discrete treatments or large samples, not a price with 2–5 levels.

### Time variation is a shrinkage problem, not a window-picking problem

The elasticity should follow a **random walk inside a Kalman filter** on the residualized pair (Ỹ_t, T̃_t), with the drift variance q estimated by maximum likelihood. Four properties make this the right default:

- **Information arrives only at price changes.** When T̃ ≈ 0 the gain for β is ≈ 0, so nothing is learned on days without price variation.
- **It collapses to the static estimate** when q̂ = 0. Test that boundary with the ½χ²₀ + ½χ²₁ mixture (critical value 2.71).
- **It initializes from the hierarchical prior**, the formal link to §41.
- **It only filters for decisions.** The RTS smoother is reserved for historical analysis.

Kalman-filtered time-varying price elasticities have marketing precedent ([Ataman, Mela & van Heerde](https://people.duke.edu/~mela/bio/papers/Ataman_Mela_Van_Heerde_2006.pdf)). Rolling windows survive as diagnostics, admitted only with at least 2 independent price changes. If they are combined, use simplex weights shrunk toward equal weights, because equal weights often win. That is the forecast-combination puzzle ([Smith & Wallis](https://papers.tinbergen.nl/14127.pdf)); Pesaran and Pick average across windows for the same reason ([related work](https://ideas.repec.org/p/cam/camdae/1163.html)). Nested windows must never be inverse-variance weighted. Their covariance structure gives a cheap Hausman drift test instead.

v1's ten regimes split two ways:

- **Observed categories:** promotion, local event, seasonal peak/low, supply shock and competitive shock become covariates in τ(Z) and in the level.
- **Latent states:** at most a 2–3-state Markov-switching level regime (statsmodels `MarkovRegression` with `exog_tvtp` driven by the lagged shock index, [statsmodels](https://pypi.org/project/statsmodels/0.15.0/)), using filtered probabilities only.

A regime may switch the slope only if it contains at least 2 price changes. The simulator must mix **demand curves**: E[Q(p)] = Σπ_r Q_r(p). The implied elasticity weights each regime by its demand share. In the team's worked example, v1's formula gives −1.64 where the correct value is −1.33. Change-point detection (PELT with a σ²-scaled BIC penalty and min_size ≥ 28 days; BOCPD with hazard 1/365) is feasible for level shifts. It is rarely feasible for slope shifts, which need at least 2 price changes on each side.

### Combine estimators with a bias-aware hierarchical model, never by forecast skill

Ordinary cross-validation cannot select causal models, because counterfactuals are unobserved ([Mahajan et al., ICLR 2024](https://arxiv.org/abs/2211.01939)). Factual-prediction criteria are limited and prone to "congeniality bias" ([Curth & van der Schaar 2023](https://proceedings.mlr.press/v202/curth23b/curth23b.pdf)). Bayesian model averaging degenerates when the true process is not among the candidates ([Yao et al. 2018](https://projecteuclid.org/journals/bayesian-analysis/volume-13/issue-3/Using-Stacking-to-Average-Bayesian-Predictive-Distributions-with-Discussion/10.1214/17-BA1091.full)). v2 should weight members on criteria that carry information about the slope:

1. Held-out R-loss.
2. **Counterfactual backtests on held-out regular-price changes.** Predict Δlog Q for the realized Δlog p using only pre-change data, and filter out events that fail a lead placebo.
3. **Semi-synthetic "plasmode" simulation.** Plant a known elasticity into the business's own covariate, price and gap paths, with injected endogeneity, and rotate the simulator's nuisance class to avoid congeniality bias.

The recommended combiner treats each member as ε̂_k = ε + b_k + e_k. The errors e_k have a covariance Σ from a **joint** bootstrap, and the member biases b_k scale with tier (0 / 0.1 / 0.2 / 0.3 / 0.6). The combiner is closed-form against the prior. Identification error then enters the reported interval openly. The joint bootstrap is not optional. In the team's example with three members correlated at 0.9, naive independent weighting claims an SD of 0.198 when the true SD of that combination is 0.329.

### Intervals: parameters and observables need different machinery

| Object | Method | How coverage is tested |
|---|---|---|
| Pooled elasticity θ | Cluster-robust sandwich SE when there are ≳ 30 clusters | Semi-synthetic simulation |
| θ with few clusters | Wild cluster bootstrap-t ([Cameron, Gelbach & Miller 2008](https://ideas.repec.org/a/tpr/restat/v90y2008i3p414-427.html)); Webb six-point weights when G ≤ 12 ([Webb 2023](https://onlinelibrary.wiley.com/doi/abs/10.1111/caje.12661)); Bayesian/EB only when G < 6 | Semi-synthetic simulation |
| θ for one product series | Stationary bootstrap on the score, with the block length from `arch.optimal_block_length` ([arch](https://pypi.org/project/arch/8.0.0/)); fall back to the posterior if more than 5% of resamples have no price variation | Semi-synthetic simulation |
| Product posterior | Hierarchical posterior | Simulation-based calibration plus leave-one-business-out |
| Daily demand, Δlog Q, revenue | Split conformal or CQR across exchangeable units; ACI/AgACI for time series ([Gibbs & Candès](https://arxiv.org/pdf/2106.00170)); weighted conformal with 0.99^age weights ([Barber et al.](https://arxiv.org/abs/2202.13415)) | Walk-forward coverage with Kupiec and Christoffersen tests ([Fed FEDS 2005-21](https://www.federalreserve.gov/pubs/feds/2005/200521/200521pap.pdf)) |
| Demand at an unobserved price | Posterior draws ⊕ conformal residuals, labeled "model-based" | Price-change backtest |

Three numbers anchor the acceptance criteria:

- **ACI step size:** the common γ = 0.005 makes ACI's guarantee vacuous over a year of daily data (bound 0.52 at T = 365). Use γ in 0.02–0.05, or aggregate over a γ grid.
- **Calibration sample size:** about 100 intervals are needed to distinguish 80% coverage from 72%.
- **Acceptance bands:** across a simulation grid (few vs many price changes, gaps, promotions, stockouts, endogeneity), 80% intervals must cover 0.75–0.85 and 95% intervals 0.92–0.98.

### Confidence becomes a calibrated probability of a named event

"Confidence: 87%" should disappear. In its place, the system reports probabilities computed from the reporting distribution, which already includes the tier bias:

- P(sign correct);
- P(|ε − ε̂| ≤ 0.5);
- P(ε < −1);
- expected absolute error;
- at the recommendation level, **P(profit at the recommended price > profit today)**.

These are recalibrated with isotonic regression fitted on semi-synthetic replicates and price-change backtests. They are accepted only if reliability bins fall within binomial bands, ECE ≤ 0.05, and confidence ranks realized error (Spearman ≤ −0.3).

The three v1 scores are replaced:

- **DQS becomes a pre-model predicted standard error**, σ̂_u/(√n_eff·sd(T̃)). It is falsifiable, and labels map to decisions: insufficient > 1.0, weak 0.5–1.0, good 0.25–0.5, excellent < 0.25.
- **IDS becomes an identification report.** It gives the tier, plus information shares by source of price variation (the pooled slope decomposes *exactly* as Σ_k I_k θ̂_k over regular changes, promotions, store-wide changes and tests), plus robustness values against 0 and −1.
- **A lead test caps the tier.** If demand residuals today are predicted by price residuals 7, 14 or 28 days later, prices are being set in anticipation of demand, and the tier is capped at 6.

### Drift monitors need sample-size-aware thresholds and multiplicity control

v1's PSI rules of thumb (0.1/0.25) fire constantly at SMB window sizes. Under no drift with 100 observations per window, PSI exceeds 0.1 in about 85% of windows. A properly rescaled PSI is asymptotically χ² with critical values that depend on sample size ([Yurdakul & Naranjo](https://www.risk.net/journal-of-risk-model-validation/7725371/statistical-properties-of-the-population-stability-index)). Use PSI_crit = χ²_{B−1,0.99}·(1/n + 1/m), with permutation tests below n = 100. The primary demand monitor should be a two-sided CUSUM on standardized forecast errors (k = 0.5, h = 5; in-control ARL ≈ 465 days; a 1σ shift detected in about 10 days, [SigmaXL ARL](https://www.sigmaxl.com/CUSUMARL.html)). Daily charts on 500 products would produce about 390 false alarms a year, so run it at business level, use h = 7 for product charts, and apply Benjamini–Hochberg. "Elasticity drift" is unobservable, so monitor its footprints instead:

- Kalman innovations on price-change days;
- backtest z-scores (two consecutive |z| > 2);
- the window Hausman test.

On a trigger, retraining follows this cadence:

| Cadence | Action |
|---|---|
| Daily | Kalman and ACI updates |
| Each qualifying price change | Re-estimate the posterior |
| Weekly | Refit nuisance models |
| Monthly, or on a confirmed alarm | Refit heterogeneity models, ensemble weights and calibrators |

A challenger is redeployed only if it beats the champion on the last 12 origins and passes the simulator regret gate.

## Demand data lie: stockouts, promotions, calendars and weather

### Model stockouts in the likelihood, never as regressors or by dropping days

Stockouts truncate demand exactly where price cuts push it up, so ignoring them flattens the curve. Naive sales-based demand rates are biased even with few stockouts ([Anupindi, Dada & Gupta 1998](https://pubsonline.informs.org/doi/10.1287/mksc.17.4.406)). The team's simulations show how much:

| Scenario | Naive estimate | Dropping sold-out days | Censored Poisson | True value |
|---|---|---|---|---|
| Weekly replenishment, 13% zero-stock days | −0.38 to −0.44 | — | — | −1.5 |
| Daily production cap, 41% of days sold out | −1.00 | −0.62 | −1.51 | −1.5 |

Two lessons follow. **Dropping sold-out days is worse than ignoring them**, and the right metric is the lost-demand share, not the share of stockout days. The bias can also reverse: when an FX-driven supplier shortage causes both a price rise and stockouts, ignoring censoring makes demand look *too* elastic. v2 should report corr(ln P, censoring) with every estimate.

Once censoring is flagged, the default is a right-censored Poisson or NB regression of the Terza type ([Terza 1985](https://www.sciencedirect.com/science/article/abs/pii/0165176585900539)). It must be implemented in-house, because statsmodels has no general right-censored count model. The EM version imputes E[D | D ≥ s] and runs PPML on the completed series, with multiple imputation and Rubin's rules when integrating with DML. Kaplan–Meier per price regime is a nonparametric cross-check ([Huh et al. 2011](https://pubsonline.informs.org/doi/abs/10.1287/opre.1100.0906)). Most SMBs lack inventory data, so detection runs down a ladder:

1. **Merchant signals.** A "sold out" button, restock dates or end-of-day counts. Restock dates alone recovered most of the bias in simulation, which makes them the highest-value onboarding field.
2. **Restock-cycle rule.** Zero-sales days after the last positive-sales day in a high-sales cycle are treated as stockouts.
3. **Capacity heaping.** Repeated identical daily maxima indicate a production cap.
4. **Last-sale timestamps with exposure offsets.** For Poisson demand, the stock-out time captures all the information in sales timing ([Jain, Rudi & Wang 2015](https://ideas.repec.org/a/inm/oropre/v63y2015i1p134-150.html)).
5. **Zero-run tests.** These are nearly useless below 0.5 units/day: at α = 0.01 a run needs about 10 days to flag.

HMM detectors validated on big-box data (63% detection, 15.5% false alerts, [Montoya & Gonzalez 2019](https://pubsonline.informs.org/doi/10.1287/msom.2018.0732)) belong in v3.

### Separate regular and promotional price with a spell-based filter

Temporary price cuts mix brand switching, borrowing from other weeks and category expansion, roughly a third each ([van Heerde, Leeflang & Wittink 2004](https://pubsonline.informs.org/doi/abs/10.1287/mksc.1040.0061)). Post-promotion dips run 4–25% of the bump ([van Heerde et al. 2000](https://research.tilburguniversity.edu/en/publications/the-estimation-of-pre-and-postpromotion-dips-with-store-level-sca/)), and reference prices persist about a year while weekly prices change every two weeks ([Eichenbaum, Jaimovich & Rebelo 2011](https://www.aeaweb.org/articles?id=10.1257%2Faer.101.1.234)).

v2 needs a **spell-based regular-price filter**:

1. Split the price series into spells of identical price.
2. A spell is long if it lasts at least 7 days.
3. A long spell is a "V-sale" if its price is at least max(3%, one price-grid step) below the regular price and the price returns within 28 days.
4. The regular price is the price of the most recent long, non-V spell.
5. In real time, a new low price counts as a temporary cut until it has persisted 28 days.

A centered running mode fails here: it flags the day before a permanent change.

The demand model then estimates β_R on ln R_t and β_D on ln(p/R) separately, with post-dip terms for storable goods. Clearance markdowns are excluded from β_D. β_R is reported as "not identified" when only promotions vary. In the team's simulation, pooling a true regular −0.8 with promotional −3.0 returned −2.90. That would have set price at about 1.53× cost where the true demand is inelastic, so the recommendation would have pointed the wrong way. Calendar endogeneity matters too: retail prices fall during seasonal demand peaks ([Chevalier, Kashyap & Rossi 2003](https://www.aeaweb.org/articles?id=10.1257%2F000282803321455142)). At least 50% of price variance must therefore survive partialling out calendar effects. Otherwise the estimate is flagged "price moves with calendar".

### Model reference effects and cash price points at the decision stage

Losses loom larger than gains in reference-price response, with a meta-analytic loss-aversion ratio of 1.49–1.73 ([Neumann & Böckenholt 2014](https://ideas.repec.org/a/eee/jouret/v90y2014i2p182-197.html)). Estimating η_gain and η_loss needs at least 2 price changes in each direction, which SMBs rarely have. So v2 should carry them as a prior (λ ~ logN(ln 1.5, 0.5²)), stage increases, and avoid reading the first 2–4 weeks after a change as the long-run elasticity.

Price endings matter. A 1-cent rise from a 99-ending is perceived like a 15–25 cent rise ([Strulov-Shlain 2023](https://academic.oup.com/restud/article-abstract/90/5/2612/6931812)), and 9-endings raised demand in three field experiments ([Anderson & Simester 2003](https://econpapers.repec.org/RePEc:kap:qmktec:v:1:y:2003:i:1:p:93-110)). This is US evidence. In cash economies, firms set "convenient" prices instead ([Knotek 2011](https://ideas.repec.org/a/tpr/restat/v93y2011i3p1076-1086.html)). v2 should **learn each shop's price grid**: the largest g in {1, 2, 5, 10, 20, 50, 100, …} that divides at least 80% of historical posted prices. Recommendations snap to that grid, and endings, left-digit crossings and denomination crossings enter the model as features when they vary. Haggled informal prices are customer-specific and endogenous, so they must never be used as identifying variation.

### Calendars must be country-specific and lunar-aware

Calendar effects drive both demand and the owner's pricing, so they belong in both nuisance models. Spending jumps at check receipt ([Stephens 2003](https://www.aeaweb.org/articles?id=10.1257%2F000282803321455386)), and food-stamp recipients' intake falls 10–15% over the month ([Shapiro 2005](https://econwpa.ub.uni-muenchen.de/econ-wp/mic/papers/0304/0304005.pdf)). African pay cycles are documented mainly by industry sources: the 25th is South Africa's common payday ([Arcadia Finance](https://www.arcadiafinance.co.za/personal-finance/income/salary-dates-in-south-africa/)), plus early-month social-grant days, and Nigerian salaries arrive between the 24th and month-end. v2 should ship country payday presets, with a learned day-of-month Fourier fallback for markets without a known norm (such as Kenya).

Islamic holidays move about 11 days per Gregorian year. The `holidays` package covers Kenya, Nigeria, South Africa, Ghana and others, labels unconfirmed Eid dates "(estimated)", and notes that Nigerian dates usually fall a day after Umm al-Qura ([holidays](https://pypi.org/project/holidays/)). Future lunar dates should therefore be spread over ±1 day. Ramadan needs an indicator, a day-of-Ramadan term and a last-10-days term. Seasonality depends on history length:

| History | Annual seasonality |
|---|---|
| Under 13 months | None |
| 13–24 months | Shrunk month dummies |
| 24 months or more | Fourier terms |

v1's one-hot week-of-year and its Price×Weekend/Holiday/Event interactions should be banned unless a within-cell identification test passes: at least 2 price levels with at least 30 units each and SE ≤ 0.5.

### Weather is a nuisance control with strict leakage rules

Weather effects are nonlinear and climate-relative, and adaptation lowers sensitivity, except to extreme heat ([Roth Tran](https://pubsonline.informs.org/doi/abs/10.1287/mnsc.2023.4799)). Use:

- **Temperature:** ERA5-Land through Open-Meteo.
- **Rainfall history:** CHIRPS at 0.05°, daily from 1981 ([Funk et al. 2015](https://www.nature.com/articles/sdata201566)). CHIRPS beat ERA5 against Ethiopian gauges ([MAP 2024](https://www.iris.sssup.it/retrieve/d3bd82b5-a3be-4b7c-8056-2324d61e8687/s00703-024-01008-0.pdf)), but validations conflict elsewhere, so the source should be configurable per region.
- **Features:** apparent-temperature maximum, opening-hours rain, a climatology anomaly and a 7-day rain total.
- **Inclusion gate:** at least 180 days of history, at least 20 rain days and an improvement in rolling Poisson deviance.

Realized reanalysis is fine for retrospective elasticity estimation. Forecasting and price simulation, however, must be trained on **archived lead-h forecasts**, such as Open-Meteo's Previous Runs API, mostly archived from January 2024 ([Open-Meteo](https://open-meteo.com/en/docs/previous-runs-api)), to avoid train/serve skew. ERA5 and CHIRPS carry `available_at` lags of about 5 days and 2 days to 3 weeks respectively, and days must be computed in local time (UTC+3 in East Africa).

### Cross-price effects need an identification test, and forecasts need the right metrics

About 10% of 1,060 published cross-price elasticities have the wrong sign ([Sethuraman et al., via NBER w12756](https://www.nber.org/system/files/working_papers/w12756/w12756.pdf)), and shop-wide inflation repricing makes them unidentified. v2 should first split v1 §31 into within-shop cannibalization and competitor price effects. It should then estimate a single shrunken "substitute-group price index" coefficient only when the partial price correlation is below 0.8 and the substitute has at least 2 independent price changes. Substitutes' stockouts serve as natural experiments for diversion ratios. Even when cross effects are skipped, group promotions must be controlled in the own-price model.

For baseline forecasting, use a model ladder:

- seasonal naive;
- ETS for smooth series;
- SBA or TSB for intermittent series ([Syntetos & Boylan 2005](https://eprints.lancs.ac.uk/id/eprint/73097/); [Teunter et al. 2011](https://ideas.repec.org/a/eee/ejores/v214y2011i3p606-615.html));
- one global LightGBM with a Poisson or Tweedie loss and monotone price constraints.

Train on unconstrained demand and select on RMSSE or MASE plus bias, as in M5 ([Makridakis et al. 2022](https://www.sciencedirect.com/science/article/pii/S0169207021001874)). The forecaster's own price response is not an elasticity.

## Markets are estimated choice kernels, not radii

### Replace radii and the five-factor index with a spatial-logit market

v1's competitor index C_ij = S·D·M·R·W counts distance twice. Customer overlap between two exponential catchments is itself a slowly decaying function of distance, halving at about 1.2× the median customer distance. The index also has no scale for S, no diminishing returns, a "cheaper" CPI that is unnormalized (ten identical rivals at parity give CPI = 10·W̄), and a radius chosen by argmax of predictive accuracy, which always "finds" a radius even when none matters.

IO evidence says competition is very local, type-specific and diminishing. A Walmart supercenter within one mile cut incumbent revenue about 16%, decaying quickly with distance, with no causal price effect ([Arcidiacono et al. 2020](https://www.aeaweb.org/articles?id=10.1257%2Fapp.20180047)). Rival effects depend on product type ([Seim 2006](https://ideas.repec.org/a/rje/randje/v37y20063p619-640.html)). Spatial choice models with overlapping choice sets avoid defining markets ex ante ([Ellickson, Grieco & Khvastunov 2020](https://onlinelibrary.wiley.com/doi/abs/10.1111/1756-2171.12310)).

v2 should define the market as a **Huff/spatial-logit kernel with an outside option** ([Huff 1963](https://www.semanticscholar.org/paper/A-Probabilistic-Analysis-of-Shopping-Center-Trade-Huff/d35f5666d9455a4cb8a6c75898a95d8109858784)):

- **Utility:** u_cj = γ·log A_j − β·t_cj, where t_cj is generalized travel time in minutes from H3 resolution-9 origin cells, and A_j is attractiveness.
- **Estimation:**
  - With multi-store flows: Poisson with origin fixed effects, which equals the conditional logit.
  - With own-store customers only: a **population-offset Poisson**. Without the offset, β absorbs population geography (team simulation: 0.174 vs a true 0.12).
- **Shrinkage:** below about 100 geocoded customers, shrink hard toward the hierarchical prior.
- **Reporting:** any "radius" is a quantile of the fitted customer-distance distribution. For an exponential kernel, R50 = 1.678/β and R80 = 1.784·R50.

Competitive pressure then follows from a spatial nested logit, which maps directly into own- and cross-elasticities. Free cross-elasticity estimates are shrunk toward that structure. Similarity enters inside the logit, not as a multiplied raw factor.

Empirical reach is short for frequent purchases. A 1% increase in distance cuts spending about 0.4%, more steeply for frequently bought goods ([Agarwal, Jensen & Monte 2017](https://www.nber.org/system/files/working_papers/w23616/revisions/w23616.rev1.pdf)), and transit time drives restaurant choice ([Davis, Dingel, Monras & Morales 2019](https://www.journals.uchicago.edu/doi/abs/10.1086/701680)). In Nairobi, walking accounts for about 41% of trips and matatus about 29% ([Salon & Gulyani 2019](https://www.researchgate.net/publication/334439119_Commuting_in_Urban_Kenya_Unpacking_Travel_Demand_in_Large_and_Small_Kenyan_Cities)). v1's "grocery 1–10 km" therefore badly mismatches dukas and spaza shops.

The team proposes log-normal decay priors expressed as median customer distance:

| Category | Median customer distance | sd(log β) |
|---|---|---|
| Kiosk/duka/spaza | 0.3 km | 0.5 |
| Café | 0.7 km | 0.6 |
| Local eatery | 0.6 km | 0.6 |
| Salon/kinyozi | 0.8 km | 0.6 |
| Pharmacy | 0.7 km | 0.5 |
| Formal supermarket | 2.5 km | 0.6 |
| Garage | 3.0 km | 0.8 |
| Professional services | 5.0 km | 0.8 |

Car-dominant suburbs multiply the median by 2.5. **These are unvalidated.** No African trade-area measurements by business type were found, and once a category × city cell has at least 20 businesses with customer geography, the empirical-Bayes posterior should replace them.

Kernel form (exponential by default, then Gaussian, then truncated power with λ > 2) is chosen among at most three pre-registered candidates by spatial-block cross-validation. Kernel parameters are fitted on folds disjoint from the elasticity folds and then frozen, so outcome information never leaks into DML features. Two descriptive outputs replace v1's tuned radius:

- **Reduced-form band index.** Estimated distance-band × similarity-class coefficients on log(1+N_rivals), with bands [0, 0.25), [0.25, 0.5), [0.5, 1), [1, 2) and [2, 5) km.
- **Effective competitive reach.** The outermost band with a Holm-significant coefficient.

### Build the geography from open African data, with licensing guards

**Routing.** Self-host OSRM (BSD-2, [Mapbox](https://blog.mapbox.com/more-open-project-osrm-switches-to-bsd-fde88d5ef462?gi=4841f987e406)) or Valhalla (MIT; isochrones and multimodal, [GitHub](https://github.com/valhalla/valhalla)) on monthly Geofabrik country extracts, with walking and driving profiles. Calibrate free-flow car times with a congestion multiplier estimated from about 1,000 traffic-aware OD pairs per city per quarter. Euclidean distance × circuity ([Boeing 2022](https://onlinelibrary.wiley.com/doi/abs/10.1111/gean.12281)) is acceptable only for prefiltering and barrier-free walking cores. Euclidean and network measures diverge in suburbs and for transit ([Apparicio et al. 2008](https://pmc.ncbi.nlm.nih.gov/articles/PMC2265683/)), and barriers such as Thika Road or Lagos's lagoon bridges matter.

**Competitor discovery.** Combine three sources:

- OSM (ODbL);
- Overture Places (85M+ places, permissive licence, no OSM data, an existence `confidence` field, [Overture](https://docs.overturemaps.org/guides/places/));
- Foursquare OS Places (100M+ places, Apache-2.0, an incomplete `date_closed`, [Foursquare](https://foursquare.com/resources/blog/products/foursquare-open-source-places-a-new-foundational-dataset-for-the-geospatial-community/)).

Deduplicate with Fellegi–Sunter weights (Splink), keep chain branches 300 m apart separate, and give each competitor an existence probability π_k calibrated on field audits. Google Places can be used only transiently: place_id may be stored, but coordinates only for up to 30 days and names and ratings not at all ([Google terms](https://cloud.google.com/maps-platform/terms/maps-service-terms)). Rating histories and a persistent competitor table cannot be built from it. Informal vendors are missing from every source, and the gap is plausibly larger in poorer areas. OSM *building* completeness in sub-Saharan Africa is only about 30% ([Herfort et al. 2023](https://www.nature.com/articles/s41467-023-39698-6)). Field audits of 20–40 H3 cells per city, stratified by density and wealth, should therefore estimate per-source capture rates, so that rival counts can be scaled by 1/π̂. Embedding similarity becomes one input to a calibrated competitor classifier trained on about 300 labeled pairs per city × category (target AUC ≥ 0.85). Raw cosine is never a substitution probability; text-based peer sets overlap only 62% with expert peers even at their best ([Hoberg & Phillips 2016](https://econpapers.repec.org/RePEc:ucp:jpolec:doi:10.1086/688176)).

**Relative price.** Replace §10's CPI with rp_j = log P_j − Σŵ_jk log P_k, with normalized weights, computed on size-normalized matched items, and emitted only when price coverage is at least 0.5 with at least 3 priced rivals. Otherwise it is NA.

**Demographics.** v1 §12's census variables mostly do not exist below city level. Nigeria has had no census since 2006 and relies on GRID3 bottom-up estimates ([HDX](https://data.humdata.org/dataset/bottom-up-gridded-population-estimates-for-nigeria-version-2-0)). Use layered rasters:

| Layer | Source | Notes |
|---|---|---|
| Population | WorldPop R2025A, 100 m, CC BY, still alpha ([release](https://data.worldpop.org/repo/prj/Global_2015_2030/R2025A/doc/Global2_Release_Statement_R2025A_v1.pdf)), and GHSL | Flag ward-level disagreement between the two |
| Built form and growth | Google Open Buildings v3 and its 2016–2023 temporal series ([Open Buildings](https://sites.research.google/gr/open-buildings/)) | Density and income proxies |
| Activity | VIIRS night lights, 500 m ([EOG](https://eogdata.mines.edu/products/vnl/)) | Saturates in CBDs |
| Wealth | Relative Wealth Index, 2.4 km | **CC BY-NC**, so commercial use needs a licence ([HDX](https://data.humdata.org/dataset/relative-wealth-index)); otherwise an in-house building × radiance proxy calibrated to DHS at admin-1 |
| Calibration | DHS | Region level only, because cluster coordinates are displaced up to 2 km urban and 5–10 km rural ([DHS SAR7](https://dhsprogram.com/pubs/pdf/SAR7/SAR7.pdf)) |

Aggregate rasters mass-conservingly to H3 resolution 9, then to the catchment with normalized kernel weights. Use the law of total variance for "income variance". Never use WorldPop's 2026–2030 projections as observed values in backtests.

**Storage.** Supabase lacks h3-pg ([issue #245](https://github.com/supabase/postgres/issues/245)), so H3 is computed in Python and stored as BIGINT beside PostGIS geography, with GiST indexes and exact `ST_DWithin` filters. Every geo feature carries snapshot, routing-engine and kernel versions, and nothing is back-filled into history.

### Competitor entry is a platform-level ring event study

Entry and exit should be detected from monthly open-data snapshots (first-seen/last-seen, Foursquare `date_closed`, Overture confidence drops, OSM lifecycle tags), with date intervals rather than dates. Estimate with Callaway–Sant'Anna or BJS on distance bins, with not-yet-treated and far-ring controls, a ±1-month donut, and outer-ring null tests ([Butts](https://www.kylebutts.com/papers/did-rings/)). Pool by category × city tier, planning for 30–50 treated businesses per cell. Single-business entry effects are not identified. The pooled effects feed priors, and a triple difference (price × post × near) tests whether entry makes demand more elastic.

## Recommendations should be risk-constrained decisions that can say HOLD or TEST

### Simulate the posterior predictive inside the observed price support

The simulator should push **posterior draws** (S = 4,000) through the demand model, using common random numbers across candidate prices. The pieces are:

- **Baseline:** Q0 is the model's baseline forecast at the current price, from the same model that produces Q1.
- **Demand:** ln Q = lnQ̂0 + ε·r + ½γr², a log-quadratic, with optional negative-binomial noise.
- **Sales and profit:** sales are min(D, stock), and profit is (κP − c)·Q. Here κ = 1/(1+VAT) − payment fee, a net-price factor that v1 omits entirely.

Constant elasticity is the tangent in log-log space. When |ε| rises with price, it **overpredicts quantity for both increases and cuts**. In the team's logit example, CE predicted +17.4% profit at +20% where the truth was +2.6%, and v1's grid would have chosen the +20% boundary while the true optimum was +6.5%. Even at ±10%, CE quantity bias (about 3%) is the same order as typical profit gains. The "recommend" range should therefore be the intersection of P0·[0.90, 1.10] and [0.95·historical min, 1.05·historical max]. Wider moves happen only in "experiment" mode. A single-price history yields only {HOLD, TEST}. Linear refits near the optimum suffice for learning ([Besbes & Zeevi](https://pubsonline.informs.org/doi/10.1287/opre.1080.0640)), which supports staying local.

### Optimize loss-averse profit subject to a loss probability and CVaR budget

The most decision-relevant scalar is the local Lerner rule: **raise price iff ε > −1/m0**, where m0 is the current margin net of VAT and fees. Report its posterior probability (for example, "88% chance a small increase raises profit"). The closed-form optimum P* = (c/κ)·ε/(1+ε) is hypersensitive: the markup factor is 11 at ε = −1.1 and 2 at ε = −2. It should serve only as a test oracle.

The objectives v1 offers disagree sharply. On one posterior (ε ~ N(−1.8, 0.6²), P0 = 10, c = 6), the team found:

| Criterion | Chosen price |
|---|---|
| Expected profit | 12.0 |
| Mean − 1·SD | 10.5 |
| Minimax regret | 11.5 |
| Robust maximin | 10.0 |
| 20% loss-probability cap | 11.0 |

v1's λ·Var is not scale-invariant: the same λ picks 12.0 with prices in units and 10.0 in cents.

The recommended default:

- **Objective:** maximize expected loss-averse utility relative to the status quo, E[ΔΠ] − (λ_LA − 1)·E[(−ΔΠ)⁺], with λ_LA = 2.25 from lab estimates ([Tversky & Kahneman 1992](https://doi.org/10.1007/BF00122574)). It should be elicited per owner, since no SMB calibration exists.
- **Chance constraint:** P(ΔΠ_28d < 0) ≤ 0.20, tightened to 0.10 for tiers 6–7.
- **Downside budget:** CVaR₀.₉₀ of loss ≤ 5% of baseline 28-day profit ([Rockafellar & Uryasev 2000](https://doi.org/10.21314/JOR.2000.038)).
- **Materiality:** an expected gain of at least 1% of baseline, or HOLD.

Loss aversion alone does not bound the probability of loss; the chance constraint does. When P(ε < −1/m0) already exceeds δ, no increase of any size is allowed. The posterior is unsure of *direction*, so the action becomes TEST. Inelastic estimates (|ε| ≤ 1) make CE profit unbounded. They must be flagged as boundary solutions, capped at +5–10%, shrunk toward the prior, and followed by a measurement window.

### Build the feasible set before optimizing, with fairness and legal rules

Constraints are applied as a discrete feasible set, which always contains P0 exactly and is never formed by rounding a continuous optimum:

1. **Price ladder:** the shop's learned price ladder, not US-style x.99 endings in cash markets.
2. **Margin floor:** P ≥ c/(κ(1 − 0.10)). v1's `cost*1.10` is a 9.09% margin.
3. **Step limit:** ±10% per change, ±5% for staples and tiers 6 and above.
4. **Frequency limit:** at least 14 days between changes and at most 4 changes per quarter.
5. **Competitor penalty:** soft, and ignored if the competitor data are more than 30 days old.
6. **Fairness rule:** 82% of respondents called raising snow-shovel prices after a storm unfair, while cost-driven increases are accepted ([Kahneman, Knetsch & Thaler 1986](https://eml.berkeley.edu/~saez/course131/Kahneman-FairnessConstraintProfit-1986.pdf)). So the engine blocks increases when the shock engine shows a demand surge and costs rose less than 2%, and permits cost pass-through.
7. **Legal default:** one posted price per product, store and moment, and respect for regulated maxima.
8. **Infeasibility:** an empty set returns INFEASIBLE with the binding constraints, never a silent relaxation.

Rue La La's tool-recommended increases did not reduce sell-through for medium and high price points ([Harvard D3](https://d3.harvard.edu/analytics-online-retailer-demand-forecasting-price-optimization/)), but no African evidence on fairness perceptions or owner acceptance was found. A pricing platform that pools competitor data and recommends prices to competing tenants also raises hub-and-spoke coordination questions. Recommendations should stay tenant-independent and exclude rivals' non-public prices.

### Learn while pricing, but choose tests by value of information

Greedy certainty-equivalent pricing stops learning ([Keskin & Zeevi 2014](https://pubsonline.informs.org/doi/10.1287/opre.2014.1294)). In the team's simulation it ended far from the optimum in 29% of runs, against about 10% for Thompson sampling or constrained least squares. v2 should run **safe batched Thompson sampling** ([Ferreira, Simchi-Levi & Wang 2018](https://pubsonline.informs.org/doi/10.1287/opre.2018.1755)):

- weekly epochs, with the switching limits above;
- a hierarchical prior as the Thompson-sampling prior;
- a conservative filter that keeps cumulative profit within 5% of the status quo with 90% probability ([Wu et al. 2016](http://proceedings.mlr.press/v48/wu16.html));
- a dispersion guard that proposes a TEST when posterior SD stays above 0.25 without price variation.

Few changes suffice in theory: regret is O(log^(m)T) with m changes ([Cheung, Simchi-Levi & Wang 2017](https://www.semanticscholar.org/paper/Technical-Note-Dynamic-Pricing-and-Demand-Learning-Cheung-Simchi-Levi/7b0c9bcfb697c1b4f0d2bfaa0e528a6510c65cef)). Tests are selected by preposterior **expected value of sample information**, using the production decision rule. The team's calculations reverse v1 §78. At 100 units/week, a 2-week-per-arm ±5% test barely moves the posterior SD (0.60 → 0.58), while a 12-vs-10 test with 2–4 weeks per arm is worth about 2.8% of baseline profit when the risk constraint binds. Information pays most exactly when a constraint is blocking a probably-profitable move. Larger gaps beat smaller ones, and 2–4 weeks per arm beats 8. Concurrent tests should cover at most 20% of revenue.

### Evaluate three objects separately with as-of rolling origins

**Forecasts.** Use mean-consistent metrics: RMSSE, Poisson deviance, and |bias| ≤ 5%. MAE and WAPE prefer an all-zero forecast on intermittent items, and MAPE/sMAPE are undefined or 200% at zero sales ([Hyndman & Koehler 2006](https://doi.org/10.1016/j.ijforecast.2006.03.001)). For probabilistic forecasts, use the interval score and pinball loss ([Gneiting & Raftery 2007](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf)), coverage, and the randomized PIT.

**Elasticities.** These are scored only where truth exists:

- synthetic confounded data;
- experiments, where z = (ε̂_model − ε̂_exp)/√(se²+se²) should be about N(0, 1);
- as-of counterfactual event studies on historical price changes, with placebo dates and stratification by the owner's reason code.

"Sign accuracy" is nearly vacuous because almost every elasticity is negative. It is replaced by **decision-direction accuracy** relative to the break-even elasticity.

**Recommendations.** Score them with:

- oracle regret in a simulator suite as a release gate (beat both status quo and CE plug-in on mean and p90 regret);
- **field uplift against a randomized hold-out** (at least 10% of eligible products, or switchback blocks);
- uplift-interval calibration (80% coverage of 0.70–0.90, overclaim ratio 0.8–1.25).

**Backtest protocol.** Weekly origins, at least 12 of them; an embargo equal to the label window; nested tuning; and population priors refit per origin from other tenants' data available at that origin.

**Bitemporal data.** v1 §74–75 should be extended to full bitemporal facts (valid time plus `available_at`, append-only revisions):

- POS day-close cutoffs and late-sync handling;
- competitor-price `available_at` set to collection time;
- market definitions versioned by discovery date;
- LLM features frozen at extraction time.

**Lineage.** Every prediction records data snapshot, code commit, model hash, population-prior version, LLM model, prompt and schema hashes, RNG seed spawn keys, and the risk configuration, so that a replay reproduces the price and the risk numbers exactly.

### Replace the output contract

The final output should carry:

- an action (CHANGE, HOLD or TEST) and a 28-day horizon;
- expected changes in quantity, revenue and profit, each with 80% ranges;
- **P(worse off than today)**, a 1-in-10 bad case and CVaR;
- the elasticity mean, its 80% interval and its tier;
- the break-even elasticity;
- binding constraints, a boundary flag and lineage.

Communication should use natural frequencies in local currency per week. For example: "Expected +X; bad case about 1 week in 10, −Y; about 2 in 10 chance this leaves you worse off". It should also show volume and revenue falling while profit rises, so owners are not alarmed.

## The LLM should extract facts; the demand model should learn shocks

### Restrict Llama to grounded slots and move judgments into estimated parameters

v1's architecture principle is right, and its implementation violates it. The §2 example asks Llama for a negative "probability" (−0.63), severity, scope, duration and self-reported confidence. LLM self-reported confidence is overconfident, clustering at 80–100% in multiples of 5 ([Xiong et al.](https://arxiv.org/abs/2306.13063)), so it cannot be multiplied into effect sizes.

v2's extraction schema (`shock_extract.v2`) should allow only grounded fields:

- an `events[]` array of at most 5 events, plus an `article_has_relevant_event` flag;
- `event_type` from a closed codebook enum;
- status (announced, threatened, ongoing, ended, cancelled, postponed, rumoured) and modality;
- place-name *strings* with roles;
- ISO dates plus the original date text;
- stated quantities, each with a verbatim evidence span;
- organizations (never private individuals);
- verbatim evidence quotes.

Coordinates, radii, severity, direction, magnitude and confidence are removed. The pipeline, not the LLM, stamps `first_seen_at`, which becomes `available_at`. Every event type has exactly one owning engine:

| Event type | Owning engine |
|---|---|
| Scheduled holidays, fixtures, elections | Calendar engine (§59) |
| Floods and other natural hazards | Weather engine (§58) |
| Business openings and closures | Competitor engine (§29) |
| Fuel-price and load-shedding schedules | Official feeds first |
| Unscheduled events, and changes to scheduled ones | News |

Individual crimes are excluded. Removing event definitions from a prompt cost 11.95 F1 points in a ChatGPT event-extraction study ([Gao et al. 2023](https://arxiv.org/abs/2303.03836)), so the codebook prompt must carry a one-line definition with include and exclude notes for every type. It should also carry few-shot examples, including a negative example with no event.

### Pin providers that pass a strict-schema canary, and translate unsupported languages

The Hugging Face router exposes an OpenAI-compatible endpoint at `router.huggingface.co/v1` with `:fastest`, `:cheapest` and `:preferred` routing ([HF docs](https://github.com/huggingface/hub-docs/blob/main/docs/inference-providers/index.md)). HF does not guarantee strict `json_schema` support across providers and recommends pinning a provider ([HF structured outputs](https://github.com/huggingface/hub-docs/blob/main/docs/inference-providers/guides/structured-output.md)). Provider support differs:

- **Groq** limits strict mode to GPT-OSS models ([Groq](https://console.groq.com/docs/structured-outputs)).
- **Cerebras** requires `additionalProperties: false` on every object ([Cerebras](https://inference-docs.cerebras.ai/capabilities/structured-outputs)).
- **Self-hosted vLLM** removed the old `guided_*` fields in v0.12 in favor of `structured_outputs` ([vLLM](https://github.com/vllm-project/vllm/blob/main/docs/features/structured_outputs.md)).
- **llama.cpp** grammars ignore constraints on `number` types ([llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md)), so numeric ranges are enforced after decoding.

v2 should run a **daily canary**. Each live provider for Llama-3.3-70B-Instruct gets 5 articles with the strict schema, and only providers that pass on validity and slot-F1 are routed to. Extraction runs at temperature 0, without chain-of-thought, with a fixed seed and a budget cap. At verified prices of $0.10–0.22 per million input tokens and $0.32–0.50 per million output tokens ([Artificial Analysis](https://artificialanalysis.ai/models/llama-3-3-instruct-70b/providers)), a roughly 3,600-token call costs about $0.0004–0.0009 per article. Cost is not the constraint; coverage and accuracy are.

Language is the bigger problem. Llama 3.3 supports eight languages, none of them African, and Meta discourages use in others without fine-tuning; its knowledge cutoff is December 2023 ([model card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_3/MODEL_CARD.md)). On IrokoBench, high-resource languages beat African languages by about 45%, and translate-test narrows the gap for English-centric models ([IrokoBench](https://aclanthology.org/2025.naacl-long.139/)). Swahili, Hausa, Yoruba, Igbo, Zulu and Amharic text should therefore be machine-translated before extraction. NLLB-200 weights are **non-commercial** ([NLLB](https://github.com/facebookresearch/fairseq/tree/nllb)), so production needs another MT model or a commercial API. Relevance triage should eventually move to a fine-tuned encoder. ConfliBERT beat Llama 3.1 on conflict classification (weighted F1 0.87 vs 0.77) and is hundreds of times faster ([Political Analysis 2025](https://www.cambridge.org/core/journals/political-analysis/article/extractive-versus-generative-language-models-for-political-conflict-text-classification/B00077A671DF5656043E4BC45B3864A1)).

### Validate in layers, calibrate with a supervised model, and never discard silently

Validation runs in stages, each emitting reason codes:

1. **Syntax:** JSON parse and Draft 2020-12 schema validation.
2. **Semantics:** status–date rules and value ranges.
3. **Grounding:** every quote, place and number must appear in the source, with fuzzy partial-ratio ≥ 90.
4. **Geography:** the resolved place lies inside the target country and city buffer.
5. **Self-consistency:** k = 3 extra samples for high-impact types.
6. **Calibration:** a logistic calibrator produces p_e from agreement, grounding score, source precision prior, modality, attribution, independent-outlet count and translation flag. Isotonic calibration is used only above about 1,000 labels.
7. **Event-level deduplication.**

The retry budget is one repair call with the validator errors attached, then a fallback provider, then **quarantine**. Nothing is discarded silently, since silent discards make data missing-not-at-random. Accuracy is measured on an in-house, double-annotated gold set of at least 400 articles per launch country, with Cohen's κ ≥ 0.7 on event type. Targets are a hallucination rate below 2%, a fabricated-event rate below 1%, and ECE ≤ 0.05. No public gold set covers load-shedding, fuel prices, cash shortages or minibus strikes, so building one is mandatory work.

### Treat news coverage as a measured, missable input

No single source gives dense, timely, precise local coverage of African cities. GDELT correlates only 0.20–0.26 spatially with hand-coded data and is biased toward capitals ([Hammond & Weidmann 2014](https://journals.sagepub.com/doi/full/10.1177/2053168014539924)), so it serves only for URL discovery: free, 15-minute latency, a 3-month window ([GDELT](https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/)). Event Registry is used for paid backfill to 2014 ([newsapi.ai](https://newsapi.ai/plans)). ACLED requires a corporate licence and is used only to validate unrest ([ACLED](https://acleddata.com/acled-api-documentation)). Official feeds come first for schedulable shocks. Each month, a coverage audit per city and event family enables news features only if recall is at least 0.5 and median latency at most 24 hours. Otherwise the system emits `news_missing=1` rather than a zero, because zero would mean "calm". A `news.coverage_index` control absorbs changes in news volume over time.

Deduplication runs at two levels. Documents are matched by SimHash-64 (Hamming ≤ 3), MinHash-LSH (threshold 0.7 for rewrites, 0.9 for syndication; [datasketch](https://github.com/ekzhu/datasketch/blob/master/docs/lsh.rst)) and LaBSE cross-lingual matching. Events are clustered with versioned updates, so "strike extended to Friday" revises one event rather than adding a second. Novelty decides only whether to skip an LLM call; it never scales a shock.

Place names are resolved by a deterministic chain: a curated city gazetteer with local aliases, then GeoNames dumps, then self-hosted Nominatim. The public Nominatim service caps use at 1 request per second and forbids periodic bulk use ([OSMF policy](https://operations.osmfoundation.org/policies/nominatim/)). Mordecai 3's mean error of 23.8 km or more ([GitHub](https://github.com/ahalterman/mordecai3)) is far too coarse for intra-city work. Every footprint carries a precision class and an uncertainty radius, and city-level mentions get uniform kernels rather than false point decay.

Embedding models must be chosen with language coverage in mind. Common sentence-transformers multilingual models do not list Swahili, Hausa or Yoruba ([sentence-transformers](https://github.com/UKPLab/sentence-transformers/blob/master/docs/sentence_transformer/pretrained_models.md)). v2 should embed the LLM's English summary with BGE-M3 or multilingual-e5 ([BGE-M3](https://github.com/FlagOpen/FlagEmbedding/blob/master/research/BGE_M3/README.md)), z-normalize cosine against a null distribution for each model, and select models on an in-house African-language pair benchmark.

### Emit per-type, per-phase exposures and estimate their kernels from outcomes

Shocks differ in sign, geometry and time profile:

- **Chile blackout:** transactions fell about 35% on the day, and nearly half was recouped within four days ([UChicago EPIC](https://climate.uchicago.edu/news/how-do-blackouts-impact-the-local-economy/)).
- **Indian electricity shortages:** cut plant revenue 5–10% ([Allcott et al. 2016](https://www.aeaweb.org/articles?id=10.1257/aer.20140389)).
- **South African load-shedding:** cost retailers an estimated R13.72bn in six months of 2015 ([University of Pretoria](https://repository.up.ac.za/items/0ac2b4e5-603a-4131-afcc-e2dba4d169aa)).
- **Fuel prices:** fuel-price spikes shift spending from eating out to groceries ([Gicheva, Hastings & Villas-Boas 2010](https://www.aeaweb.org/articles?id=10.1257/aer.100.2.480)).
- **Sports:** NFL game days raise card spending about 77% in stadium ZIP codes ([Bank of America Institute](https://institute.bankofamerica.com/content/dam/economic-insights/sports-impact-on-local-economies.pdf)).
- **Mass layoffs:** produce persistent regional spillovers ([Gathmann et al. 2020](https://discovery.ucl.ac.uk/id/eprint/10055494/)).
- **South Africa's July 2021 unrest:** damaged about 3,000 looted stores ([ACCORD](https://www.accord.org.za/conflict-trends/the-july-2021-protests-and-socio-political-unrest-in-south-africa/)).
- **Transport strikes:** permanently changed some commuters' routes ([Larcom, Rauch & Willems 2017](https://ideas.repec.org/a/oup/qjecon/v132y2017i4p2019-2055..html)).

A single NetShock index lets these opposite-signed channels cancel. v2 should therefore emit separate exposures per event type and phase (lead, active, post, rebound):

E_i,t^{k,φ} = Σ p_e · q_e · r_e,i · S_k(distance to footprint) · T_k^φ(t)

The components are:

- **p_e:** the calibrated probability that the event is real and correctly extracted.
- **q_e:** intensity computed in code from stated quantities, such as outage hours overlapping trading hours, or log attendance. Never article counts.
- **r_e,i:** relevance. An expert industry-rules matrix acts as a hard gate with a sign prior, and embeddings may modulate it only within allowed pairs, capped at 0.3 weight after at least 500 labels.
- **S_k:** a truncated spatial kernel (exponential, Gaussian or uniform over polygons), with exact convolution for location uncertainty.
- **T_k^φ:** temporal phases, including a separate **rebound** coefficient for postponed purchases.

Level shifts such as fuel price and tax enter as log-level steps, not decaying shocks.

Kernel parameters are estimated in three stages:

1. A pooled ring × lag event study with staggered-robust estimators, clustered by event and city-week.
2. A parametric fit of distance scale D and half-life H by weighted least squares.
3. Hierarchical pooling of D, H and β by type, city and industry, shrunk toward sign priors (PyMC-Marketing provides geometric and Weibull carry-over kernels, [pymc-marketing](https://github.com/pymc-labs/pymc-marketing/blob/main/pymc_marketing/mmm/transformers.py)).

The minimum data to estimate a type's kernel are 20 event clusters, 3 populated distance rings and 30 pre-period days; otherwise the result is labeled "prior-dominated". Sub-daily half-lives (v1's "concert: hours") cannot be identified from daily sales and become same-day indicators. Placebo shifts of ±28 days and random relocation within the city must produce estimates centered on zero.

Safety rests on separation and lineage:

- **Data isolation:** only article text, the target area and the publication date reach the provider, never business or customer data.
- **PII:** private-person names are forbidden by the schema and scrubbed by Presidio.
- **Prompt injection:** the LLM has no tools, untrusted text is spotlighted, and the Prompt Guard 2 score is a quarantine feature, not an automatic drop (97.5% recall at 1% FPR, but evaluated on no African language, [model card](https://github.com/meta-llama/PurpleLlama/blob/main/Llama-Prompt-Guard-2/86M/MODEL_CARD.md)).
- **Corroboration:** high-impact types (riot, curfew, cash shock) need an official source or at least 2 independent outlets before entering features.
- **Lineage and look-ahead:** every LLM call is logged with model, provider, prompt and schema hashes. Features are recomputed from stored outputs and never by re-calling the LLM on history. Backtests are split at the extractor's knowledge cutoff to detect pretraining look-ahead ([Glasserman & Lin 2023](https://arxiv.org/abs/2309.17322)).

Provider data-retention terms, and cross-border transfer under POPIA, NDPA and Kenya's DPA, still need legal review.

## Conclusion

The research changes what LAEE's central product claim should be. v1 promises a personalized elasticity for every product. The evidence shows that for most SMB products, own data cannot pin elasticity down: SEs near 1–2 are typical, and African shop-level benchmarks do not exist. LAEE's real defensible assets are therefore not estimators. They are the **calibrated prior** (a meta-analytic tree whose honesty is proven by leave-one-business-out coverage), the **identification bookkeeping** (tiers, information shares and decision-threshold robustness values that say exactly how much of a number came from the business's own price changes), and the **experiment engine** that turns a binding risk constraint into a value-of-information-ranked switchback test. That reframes the flywheel. Its fuel is not "more data" but more *independent price variation*, and every randomized test is worth far more than months of passive transactions, because it both identifies the business's elasticity at tier 1 and recalibrates the tier-bias variances that discount everyone else's observational estimates.

Three uncertainties should shape the roadmap rather than be papered over. First, nearly every numeric default here is engineering judgment: priors, tier bias variances, kernel half-lives, decay medians for dukas and salons, λ_LA = 2.25, and the 20% loss cap. The architecture is built so these can be calibrated, through plasmode simulation, the first pilot's experiments, and leave-one-tenant-out checks. Second, the African-specific inputs are the least verified: payday norms, cash price grids, informal competitor capture rates, African-language extraction accuracy, and the law on experimentation and cross-tenant learning. A pilot in two or three cities with field audits, merchant-reported stockouts and restock dates, and a gold news set is a prerequisite, not a refinement. Third, the order of v1's MVP should invert. A closed-form shrinkage layer, PPML event studies, stockout and promotion handling, and the risk-constrained HOLD/TEST optimizer belong in V1. Causal forests, latent regimes, cross-price networks and the news engine should wait until data volume and gates say they can be trusted.
