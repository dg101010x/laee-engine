# Hierarchical Bayesian shrinkage and empirical priors for SMB price elasticity (with African and emerging-market emphasis)

Provenance (read first). WebFetch was egress-blocked for every academic host tried (gtellis.net, web.stanford.edu, files.core.ac.uk, academia.edu, nber.org, pmc.ncbi.nlm.nih.gov, errorstatistics.com, arxiv.org). The shared WebSearch budget for the turn ran out after about ten searches, so later queries (category elasticities for apparel, OTC, services, e-commerce; promotional vs regular price; Africa fuel and transport) could not be run. Tags used below:
- **[V]**: confirmed in this session from search-result text at the URL given.
- **[L]**: established literature cited from the researcher's own knowledge. I could not re-verify these this session; the DOIs are believed correct, but check them before quoting any number.
- **[C]**: computed in this session with numpy 2.4.6 and scipy 1.17.1 (the scratch scripts are not saved in the repo). All formulas and golden values needed to reproduce them are given inline.

Conventions: ε = ∂lnQ/∂lnP (negative for normal demand), the same as spec §14. "Business" means a tenant's store or outlet. "Stage-1 estimate" means the per-business (or per-business-product) elasticity estimate b and its standard error s, produced inside the tenant boundary by FE, DML or experiments (§25–28, §76).

## 1. What is wrong or underspecified in v1 §41–42, §80 and §85–86?

### Takeaway
v1 names the right idea but gives no likelihood, no estimator, no variance components, no numbers and no uncertainty propagation.
- §41's single equation ε_i ~ N(μ_industry,market, σ²) conflates between-business heterogeneity with sampling error. It describes one level of the five-level diagram it draws.
- §42's `PriorElasticity = f(...)` has no functional form, no target estimand and no prior variance, so it is not yet a prior.
- §80 does not define what crosses the tenant boundary or how released aggregates are protected.
- §85–86 defer shrinkage to V2, yet shrinkage is the cheapest and most important component for small-data SMBs.

### Cited Findings
- [V] Morris (1983) multiplies the estimated empirical-Bayes shrinkage coefficient by (m − 3)/(m − 1). He proposed this modification so that the estimator "resemble[s] the shrinkage behaviour of the hierarchical Bayes estimator", which shows that naive plug-in shrinkage needs a small-sample correction. Sources: [PMC2872278](https://pmc.ncbi.nlm.nih.gov/articles/PMC2872278/); [Morris 1983, JASA 78(381):47–55](https://errorstatistics.com/wp-content/uploads/2015/11/morris.pdf).
- [L] Empirical-Bayes posterior variances computed with estimated hyperparameters plugged in ignore hyperparameter uncertainty. Kass and Steffey derive a first-order correction. Source: [Kass & Steffey 1989, JASA 84:717–726](https://doi.org/10.1080/01621459.1989.10478825).
- [V] The DerSimonian–Laird τ² estimator, "currently widely used by default", "has been long challenged". Simulations favour Paule–Mandel (dichotomous and continuous data) and REML (continuous data). Recommended interval methods are profile likelihood, Q-profile and generalised Q-profile. Source: [Veroniki et al. 2016, Res Synth Methods](https://onlinelibrary.wiley.com/doi/full/10.1002/jrsm.1164).
- [V] Variance-parameter estimates in multilevel models can be degenerate (exactly zero). A log-gamma(2, λ→0) penalty keeps the estimate about one standard error from zero when the MLE is zero. Source: [Chung, Rabe-Hesketh, Dorie, Gelman & Liu 2013, Psychometrika 78(4):685–709](https://link.springer.com/article/10.1007/s11336-013-9328-2).
- [V] In a meta-analysis of 1,851 elasticities, accounting for price endogeneity "significantly increases the magnitude" of price elasticities, while accounting for heterogeneity has negligible effects. Priors trained on naive observational estimates will therefore be biased toward zero. Source: [Bijmolt, van Heerde & Pieters 2005, JMR 42(2):141–156](https://journals.sagepub.com/doi/10.1509/jmkr.42.2.141.62296).
- [V] Store-specific elasticities for 83 supermarkets across 18 categories vary substantially. Eleven demographic and competitive variables explain on average 67% of the variation, and demographics matter much more than competition. Source: [Hoch, Kim, Montgomery & Rossi 1995, JMR 32(1):17–29](https://www.researchgate.net/publication/237130017_Determinants_of_Store-Level_Price_Elasticity).
- [V] There is "substantial within-chain variation in price elasticities". Elasticities vary inversely with income, and a store's elasticity is predicted by its own-store income, not chain-level income. Source: [DellaVigna & Gentzkow, "Uniform Pricing in US Retail Chains"](https://www.nber.org/system/files/working_papers/w23996/revisions/w23996.rev0.pdf) (also at [sdellavi.com](https://sdellavi.com/pdf/uniform-pricing.pdf); QJE 2019, [L] doi 10.1093/qje/qjz019).
- [V] A robust meta-analytic-predictive (rMAP) prior mixes an informative historical component with a vague component. The vague weight w controls how fast historical information is discounted under prior-data conflict. Source: [Schmidli et al. 2014, Biometrics 70(4):1023–1032](https://onlinelibrary.wiley.com/doi/abs/10.1111/biom.12242). Follow-up work notes that w "should be based on the degree of confidence in the relevance of the historical data" but that there is no clear guidance on quantifying it: [modified rMAP, medRxiv 2023](https://www.medrxiv.org/content/10.1101/2023.01.28.23285146v1.full).

### Inferences
Defects and the fix proposed for each:

**§41 (Bayesian hierarchical shrinkage)**
1. *σ² is ambiguous.* Two variances are needed. The sampling variance s_i² of the business's own estimate says how noisy its data are. The between-business variance τ² says how different businesses truly are. The weight on own data is τ²/(τ² + s_i²), and v1 has no s_i at all, so "update with the business's own data" cannot be computed. **Fix:** a two-stage model, b_i | θ_i ~ N(θ_i, s_i²) and θ_i ~ N(μ_cell, τ²) (§2).
2. *There is no estimator for μ or σ²*, and no rule for cells with one to three businesses. In those cells ML/REML τ² often equals 0, which silently produces complete pooling and overconfidence. In this session's simulation REML gave τ̂² = 0 in 27.8% of 8-business cells with true τ = 0.5 [C]. **Fix:** estimate τ once per level per industry (pooled across cells), not per cell; use a half-normal prior on τ; integrate over τ (§2.5).
3. *One level in the equation, five in the diagram.* v1 does not say how levels nest or cross (industry and region are crossed), gives no variance component per level, and has no backoff for empty cells. **Fix:** an exact tree algorithm with one τ per level (§2.6).
4. *"50 transactions" is the wrong measure of information.* Information about ε scales with n × Var(ln p) and with the number of independent price changes. With zero price variation, 10,000 transactions carry zero information about elasticity. **Fix:** an information-based data weight (§6).
5. *No uncertainty propagation.* Plug-in EB 95% intervals covered only 83.5–87.8% at k ≤ 30 and τ = 0.5 in the simulation [C]. This breaks the spirit of Rule 4 ("never report an elasticity without uncertainty"), because the reported uncertainty is wrong. **Fix:** grid integration over τ (93–96% coverage [C]).
6. *Sign is not handled.* A normal prior puts mass on ε > 0, and raw stage-1 estimates are often positive. **Fix:** keep normal pooling on the raw scale, and add decision-layer rules plus conflict checks (§6).
7. *The estimand is not harmonised.* v1 would pool promotional with regular-price elasticities, item-level with category-level, short-run with long-run, and Tier-1 with Tier-6 estimates (§76). **Fix:** an "estimand contract" field that must match before pooling (§2.1), plus a bias allowance by tier (§3).
8. *Products within a business are correlated.* They share demand shocks and controls. Treating product estimates as independent overstates business-level precision. **Fix:** block-GLS messages (§2.7).
9. *Double counting through the flywheel (§79).* If the monthly population snapshot that includes business i's data is then used as business i's prior and updated again with its own data, b_i is counted twice. This matters most in small cells or for dominant businesses. **Fix:** either compute the joint posterior in one shot (the tree algorithm, which has no double counting by construction), or compute cached priors leave-one-tenant-out. Given τ, the two are mathematically identical [derivation: sequential Gaussian conjugacy].
10. *No time dimension.* Hyperparameters drift (§20–21 regimes). **Fix:** versioned monthly snapshots on a rolling window, and optionally a random-walk business effect (§6).
11. *Normality of business effects is unchecked.* Heavy tails, such as niche price-insensitive shops alongside hyper-competitive ones, cause normal priors to over-shrink outliers. **Fix:** prior-data conflict checks plus an rMAP mixture prior (§6).

**§42 ("Local Elasticity Prior")**
1. `f(...)` has no functional form, no estimand, no training data, no loss function and **no variance**. A point value is not a prior. **Fix:** a meta-regression meta-analytic-predictive (MAP) prior, ε_new ~ N(x*ᵀβ̂, τ² + x*ᵀV_βx*) (§3).
2. Inputs overlap: industry vs businessType, location vs income/density/competition. Collinear covariates make the coefficients meaningless and the prior unstable. **Fix:** a single taxonomy key (industry → sub-industry), continuous covariates standardised, density *or* competition (orthogonalised), and ridge priors.
3. There is no mechanism to combine literature magnitudes with platform data. **Fix:** literature sets the root (industry) hyperprior, N(m_lit, s_lit²), and platform data update it through the same Gaussian machinery (§3).
4. Training on naive observational estimates bakes the attenuation bias that Bijmolt et al. document into the prior. **Fix:** an identification-tier covariate or bias allowance, predicting at the best tier.
5. There is no honesty requirement. **Fix:** leave-one-business-out (LOBO) calibration of prior predictive intervals as a release gate (§3 and §6).
6. `Update(...)` is undefined. **Fix:** the conjugate normal update, with an rMAP mixture under conflict.
7. There is no extrapolation guard for a new country or industry. **Fix:** inflate the variance with τ_shift², or back off to the parent level.

**§80 (multi-tenant learning)**
1. v1 does not say what crosses the boundary (raw rows, features, estimates or sufficient statistics). **Fix:** only per-business stage-1 summaries (b, s, n_eff, number of price changes, tier, tags) leave the tenant partition (§5).
2. θ_industry,region,market is estimable only with enough businesses per cell. There is no minimum k and no backoff.
3. There is no protection for released aggregates. In a cell of two businesses, B's leave-one-out prior *is* A's elasticity, which is a competitor's trade secret. **Fix:** k_min, a dominance rule, batching, and optional differential privacy (§5).
4. "Federated-style" is undefined. For Gaussian hierarchical models, one-shot aggregation of summaries is exact, so iterative FedAvg is unnecessary (§5).
5. Differencing attacks across model versions are not addressed (§5).

**§85–86 (MVP, recommended initial model)**
1. Shrinkage is deferred to V2, yet V1's target users (§41's "50 transactions") are exactly those whose FE/DML/GBM estimates will be noisy and often wrong-signed. A closed-form EB layer is a few dozen lines of numpy and should ship in V1 as the final post-estimation step, with the literature prior used for cold start.
2. §86's ensemble weights "from historical out-of-sample performance" do not exist for a new SMB. Its base learners use the same data, so their errors are highly correlated. Averaging them neither adds information nor permits an independence-based SE. Feeding FE, DML and CF outputs into the hierarchy as separate observations double counts the data. **Fix:** exactly one stage-1 (b, s) per business-product per estimand: either the best-identified estimator, or a covariance-aware GLS combination with a covariance matrix bootstrapped within the tenant.
3. DML cross-fitting and causal forests are unstable at 50–500 observations. Partial pooling should come *before* heterogeneity models on the roadmap.
4. §86's code returns `elasticity = ensemble(...)` with no uncertainty, which violates Rule 4.
5. Phase 6 lists industry → regional → market → business but omits the product level and the hyperparameter estimation step.

Proposed replacement statement for §41 (for the spec writer):
```
Stage 1 (inside tenant):  b_ij, s_ij  = estimate and SE of θ_ij (canonical estimand, §2.1), plus tier, n_eff, G (price changes)
Stage 2 (population):     b_ij | θ_ij ~ N(θ_ij, s_ij² + δ_tier²)
                          θ_ij = θ_i + u_ij,   u_ij ~ N(0, τ_P²)          (product within business)
                          θ_i  = θ_m(i) + v_i, v_i ~ N(0, τ_B²)            (business within market)
                          θ_m  = θ_r(m) + w_m, w_m ~ N(0, τ_M²)            (market within region×industry)
                          θ_r  = x_rᵀβ_ind + z_r, z_r ~ N(0, τ_R²)          (region×industry around meta-regression)
                          β_ind ~ N(β_lit, Σ_lit)                          (literature hyperprior)
                          τ_ℓ ~ HalfNormal(σ_ℓ)                             (σ_R=σ_M=0.4, σ_B=0.7, σ_P=0.8 to start)
Posterior: exact Gaussian tree passes given τ; integrate τ_B (grid) or use penalised ML-II for all τ with Kass–Steffey correction.
```

### Gaps
- v1 does not say whether "50 transactions" means rows, days or orders, so the available information cannot be quantified from the spec alone.
- The legal basis and consent for cross-tenant learning (data-processing agreements, opt-in) are outside statistical scope and not covered here.
- Spec sections read: §14, §39–45, §69, §76–80, §85–88, Phase 6, §91. Other sections may constrain this design and were not checked.

## 2. Hierarchical models: partial pooling, empirical Bayes, τ² with few groups, nested levels, and full vs closed-form Bayes

### Takeaway
Use a two-stage design.
1. Each tenant produces (b, s) for a fixed estimand.
2. Exact Gaussian partial pooling runs on a tree (industry → region → market → business → product). Variance components are estimated once per level (REML, Paule–Mandel or penalised ML-II), and τ-uncertainty is integrated by a one-dimensional grid where it matters.

All of this is closed-form numpy/scipy and was validated in-session against dense linear algebra and published reference values. PyMC/NumPyro should be reserved for offline validation and for crossed or non-conjugate variants. Plug-in EB intervals undercover with few groups; grid Bayes restores near-nominal coverage while still cutting interval width by 23–46% versus no pooling [C].

### Cited Findings
- [L] Normal–normal hierarchical model with the marginal posterior of τ computed on a grid after integrating out μ and θ analytically (the "eight schools" example, BDA3 chapter 5). Source: [Gelman et al., Bayesian Data Analysis 3rd ed.](http://www.stat.columbia.edu/~gelman/book/).
- [L] James–Stein: with k ≥ 3 normal means of equal known variance, shrinking toward a fixed point dominates the MLE in total squared error (k ≥ 4 when shrinking toward the estimated grand mean). Sources: [James & Stein 1961](https://projecteuclid.org/euclid.bsmsp/1200512173); [Efron & Morris 1973, JASA](https://doi.org/10.1080/01621459.1973.10481350); [Efron & Morris 1975, JASA 70:311–319](https://doi.org/10.1080/01621459.1975.10479864). The baseball example uses 18 players at 45 at-bats.
- [V] Morris (1983), "Parametric Empirical Bayes Inference: Theory and Applications", JASA 78(381):47–55, introduced the (m − 3)/(m − 1) shrinkage modification. Sources: [PMC2872278](https://pmc.ncbi.nlm.nih.gov/articles/PMC2872278/); [PDF](https://errorstatistics.com/wp-content/uploads/2015/11/morris.pdf).
- [L] DerSimonian–Laird method-of-moments τ². Source: [DerSimonian & Laird 1986, Control Clin Trials 7:177–188](https://doi.org/10.1016/0197-2456(86)90046-2).
- [L] Paule–Mandel τ² solves generalised Q = k − 1. Source: [Paule & Mandel 1982, J Res NBS 87:377–385](https://doi.org/10.6028/jres.087.022).
- [V] Veroniki et al. (2016) compared 16 τ² estimators and 7 interval methods. They recommend Paule–Mandel (all data types) and REML (continuous) over DerSimonian–Laird, and profile-likelihood or Q-profile intervals. Source: [Veroniki et al. 2016](https://onlinelibrary.wiley.com/doi/full/10.1002/jrsm.1164). Related simulation studies on τ² estimators: [arXiv 1904.01948](https://arxiv.org/pdf/1904.01948), [arXiv 1902.07154](https://arxiv.org/pdf/1902.07154).
- [L] Simulation comparison of heterogeneity estimators: DerSimonian–Laird is negatively biased when heterogeneity is large; REML and PM are preferred. Sources: [Langan et al. 2019, Res Synth Methods 10:83–98](https://doi.org/10.1002/jrsm.1316); [Viechtbauer 2005, JEBS 30:261–293](https://doi.org/10.3102/10769986030003261).
- [L] Q-profile confidence interval for τ². Source: [Viechtbauer 2007, Stat Med 26:37–52](https://doi.org/10.1002/sim.2514). The reference implementation is metafor ([Viechtbauer 2010, JSS 36(3)](https://doi.org/10.18637/jss.v036.i03)).
- [L] Hartung–Knapp–Sidik–Jonkman (HKSJ) intervals for μ outperform DerSimonian–Laird Wald intervals with few studies. Source: [IntHout, Ioannidis & Borm 2014, BMC Med Res Methodol 14:25](https://doi.org/10.1186/1471-2288-14-25).
- [L] The prediction interval for a *new* unit is μ̂ ± t_{k−2}·√(τ̂² + SE(μ̂)²). Sources: [Higgins, Thompson & Spiegelhalter 2009, JRSS-A 172:137–159](https://doi.org/10.1111/j.1467-985X.2008.00552.x); [Riley, Higgins & Deeks 2011, BMJ 342:d549](https://doi.org/10.1136/bmj.d549).
- [L] I² = (Q − df)/Q. Source: [Higgins & Thompson 2002, Stat Med 21:1539–1558](https://doi.org/10.1002/sim.1186).
- [L] Priors on τ: avoid inverse-gamma(ε, ε); use uniform-on-τ when groups are not too few, and half-t or half-Cauchy when groups are few. Source: [Gelman 2006, Bayesian Analysis 1:515–534](https://doi.org/10.1214/06-BA117A).
- [L] Half-normal τ priors for meta-analyses with few studies. Sources: [Friede, Röver, Wandel & Neuenschwander 2017, Res Synth Methods 8:79–91](https://doi.org/10.1002/jrsm.1217). Exact semi-analytic posterior via the bayesmeta package: [Röver 2020, JSS 93(6)](https://doi.org/10.18637/jss.v093.i06). Empirical informative priors for τ: [Turner et al. 2015, Stat Med 34:984–998](https://doi.org/10.1002/sim.6381).
- [V] Penalised (nondegenerate) ML for variance parameters. Source: [Chung et al. 2013](https://link.springer.com/article/10.1007/s11336-013-9328-2).
- [V] Robust empirical-Bayes confidence intervals (venue confirmed; content [L]). Parametric EBCIs can undercover when the normal prior is wrong; robust EBCIs replace z with a critical value that depends on the shrinkage factor and moments of the effects. Source: [Armstrong, Kolesár & Plagborg-Møller 2022, Econometrica](https://www.econometricsociety.org/publications/econometrica/2022/11/01/Robust-Empirical-Bayes-Confidence-Intervals/file/ecta200502.pdf).
- [V] Applied EB guides (titles confirmed; content [L]):
  - [Walters, "Empirical Bayes Methods in Labor Economics", NBER w33091](https://www.nber.org/system/files/working_papers/w33091/w33091.pdf). It covers shrinkage when effects correlate with precision.
  - [Kou & Yang, "Optimal shrinkage estimation in heteroscedastic hierarchical linear models"](https://arxiv.org/pdf/1503.06262).
  - [L] [Xie, Kou & Brown 2012, JASA (SURE shrinkage, heteroscedastic)](https://doi.org/10.1080/01621459.2012.728154).
- [L] Hierarchical Bayes for marketing elasticities (unit-level regressions with a common prior; bayesm's `rhierLinearModel`). Sources: [Rossi, Allenby & McCulloch 2005, Bayesian Statistics and Marketing](https://doi.org/10.1002/0470863692); [bayesm on CRAN](https://cran.r-project.org/package=bayesm); [Allenby & Rossi 1999, J Econometrics](https://doi.org/10.1016/S0304-4076(98)00055-4).
- [L] Store-level hierarchical-Bayes elasticities across 83 Dominick's stores for micro-marketing pricing. Source: [Montgomery 1997, Marketing Science 16(4):315–337](https://doi.org/10.1287/mksc.16.4.315).
- [L] Sign-constrained (truncated) hierarchical prior for account-level price elasticities. Source: [Boatwright, McCulloch & Rossi 1999, JASA 94:1063–1073](https://doi.org/10.1080/01621459.1999.10473859).
- [V] Stan's guide to hierarchical priors exists: [Stan User's Guide §1.10](https://mc-stan.org/docs/2_19/stan-users-guide/hierarchical-priors-section.html). [L] Non-centred parameterisation avoids the funnel when data are weak: [Betancourt & Girolami 2015](https://arxiv.org/abs/1312.0906). Documentation: [PyMC](https://www.pymc.io/), [NumPyro](https://num.pyro.ai/en/stable/).
- [L] Gaussian belief propagation on a tree gives exact marginals; BLUP / mixed-model equations cover crossed effects. Sources: [Weiss & Freeman 2001, Neural Computation](https://doi.org/10.1162/089976601750541769); [Robinson 1991, Stat Sci 6:15–32](https://doi.org/10.1214/ss/1177011926).

### Inferences

#### 2.1 Estimand contract (what may be pooled together)
Every stage-1 record must carry: estimand id; level (product-at-shop or category-at-shop); price type (regular only, with promotions excluded per §62); horizon (canonical: 4–12-week response to a regular-price change); identification tier (§76); n_obs, n_eff and G (number of distinct regular-price spells of at least 7 days); sd(ln p); period; and the taxonomy and location keys.
- Records are pooled only if the estimand id matches.
- Records whose (b, s) come from different estimators of the same data are never pooled together; each business-product-estimand pair contributes exactly one record.

#### 2.2 One-level normal–normal model (closed form; validated [C])
```
b_i | θ_i ~ N(θ_i, s_i²)        θ_i | μ_i, τ ~ N(μ_i, τ²),  μ_i = cell mean or x_iᵀβ
marginal:   b_i ~ N(μ_i, s_i² + τ²)
B_i = s_i²/(s_i² + τ²)                               # shrinkage factor = weight on the prior
E[θ_i|b] = (1−B_i)·b_i + B_i·μ_i                    # given μ, τ
Var[θ_i|b] = (1−B_i)·s_i² = 1/(1/s_i² + 1/τ²)
GLS mean:   w_i = 1/(s_i²+τ²),  μ̂ = Σw_i b_i/Σw_i,  Var(μ̂) = 1/Σw_i
μ integrated (flat prior), τ fixed:
            E = (1−B_i)·b_i + B_i·μ̂ ;   Var = (1−B_i)·s_i² + B_i²·Var(μ̂)
new unit (cold start):  θ_new ~ N(μ̂, τ² + Var(μ̂))      # predictive, not N(μ̂, Var(μ̂)), not N(μ̂, σ²)
```
**Leave-one-out identity.** Updating the LOO predictive prior N(μ̂_(−i), τ² + Var(μ̂_(−i))) with b_i gives exactly the joint posterior above. Using the *full-data* μ̂ as a prior and then updating with b_i double counts b_i.

#### 2.3 Estimating τ² (formulas validated on BCG reference data [C])
```
DL:   w=1/s²; μ_FE=Σw b/Σw; Q=Σw(b−μ_FE)²;  τ²=max(0,(Q−(k−1))/(Σw − Σw²/Σw))
PM:   solve Q_gen(τ²)=Σ(b−μ̂(τ²))²/(s²+τ²) = k−1 (Q_gen decreasing; if Q_gen(0) ≤ k−1 → τ²=0); scipy.optimize.brentq
REML: ℓ_R(τ²) = −½Σlog(s²+τ²) − ½log Σw − ½Σw(b−μ̂)²
      fixed point: τ² ← max(0, Σw²((b−μ̂)² − s²)/Σw² + 1/Σw)   (or bounded minimize_scalar on −ℓ_R; both agree)
ML:   ℓ(τ²) = −½Σlog(s²+τ²) − ½Σw(b−μ̂)²      (biased low; do not use for small k)
Meta-regression REML: ℓ_R = −½Σlog(s²+τ²) − ½log|XᵀWX| − ½(b−Xβ̂)ᵀW(b−Xβ̂),  β̂=(XᵀWX)⁻¹XᵀWb
HKSJ for μ: Var_HK = Σw(b−μ̂)²/((k−1)Σw);  CI μ̂ ± t_{k−1,.975}√Var_HK
Prediction interval (new business): μ̂ ± t_{k−2,.975}·√(τ̂² + Var(μ̂))
Q-profile CI: {τ²: χ²_{k−1,.025} ≤ Q_gen(τ²) ≤ χ²_{k−1,.975}};   I² = max(0,(Q−(k−1))/Q)
```
**Defaults.**
- REML for meta-regression and for any level with at least 10 groups.
- PM for 3–9 groups.
- DL only as a fast screen or a starting value.
- Never estimate τ per cell when cells have fewer than about 10 businesses. Estimate one τ per level per industry, pooling cells, so that hundreds of businesses inform τ_B.

**Failure modes.**
- τ̂² = 0 boundary: 27.8% of 8-group replications at τ = 0.5 [C]. This leads to complete pooling and overconfidence.
- DL is negatively biased when heterogeneity is large.
- The REML fixed point can be slow; use bounded maximisation as a fallback.
- s_i are estimated, not known. Clustered or serially correlated errors make s_i too small and the data weight too large.
- If s_i correlates with θ_i (precision-dependence), standard EB is biased. Add ln s_i as a meta-regression covariate and test its slope (see Walters).

#### 2.4 James–Stein / Efron–Morris / Morris (closed form)
```
JS (equal s², shrink to grand mean, k≥4): θ̂_i = b̄ + (1 − (k−3)s²/S)₊ (b_i − b̄),  S = Σ(b_i − b̄)²
Morris (unequal s_i, r regression coefs; r=1 → (k−3)/(k−1)):  B̂_i = ((k−r−2)/(k−r)) · s_i²/(s_i² + τ̂²)   [general-r form is L]
```
Use James–Stein only as a sanity check (unequal s_i are the norm). Use Morris's factor when k is small and the grid is not used.

#### 2.5 Integrating over τ (production default for intervals) and the evidence for it
Exact posterior for the one-level model (flat prior on μ, any proper prior on τ), BDA3 §5.4:
```
log p(τ|b) = log p(τ) + ½ log Var(μ̂(τ)) − ½Σ log(s_i²+τ²) − ½Σ (b_i−μ̂(τ))²/(s_i²+τ²)
π_g ∝ p(τ_g|b)·Δτ_g        # Δτ_g quadrature weights are mandatory on non-uniform grids
E[θ_i|b]  = Σ_g π_g M_ig,   M_ig = (1−B_ig) b_i + B_ig μ̂_g
Var[θ_i|b]= Σ_g π_g (V_ig + M_ig²) − E²,   V_ig = (1−B_ig) s_i² + B_ig² Var(μ̂_g)
intervals: exact mixture quantiles (brentq on Σ_g π_g Φ((x−M_ig)/√V_ig) = q) or the normal approximation (≈1 pp lower coverage)
```
**Defaults.**
- Grid: 600–800 points on [0, 5σ_τ + 3·max s_i], including τ = 0. A finer segment near 0 is fine if Δτ weights are applied.
- Prior on τ: HalfNormal(σ_τ) in elasticity units, with σ_τ = 0.7 at the business level.
- A uniform prior on τ yields a proper posterior only if k ≥ 3, because p(b|τ) ∝ τ^{1−k} as τ → ∞ [derivation].
- Never use p(τ) ∝ 1/τ: the posterior is improper at τ → 0.

**Cheaper alternative (Kass–Steffey first-order correction, valid only if τ̂² > 0 and k ≥ ~10):**
```
Var_KS(θ_i) ≈ (1−B_i)s_i² + B_i² Var(μ̂) + [B_i(1−B_i)(b_i−μ̂)]² · Var(ln τ̂²)
∂E[θ_i]/∂ln τ² = B_i(1−B_i)(b_i−μ̂)     # verified vs finite difference [C]
Var(ln τ̂²) ≈ 1/(τ̂⁴ · I(τ̂²)),  I ≈ ½ tr(P²), P = W − W11ᵀW/(1ᵀW1)   (ML approx: ½Σw_i²)
```
**Coverage evidence [C]** (nominal 95%; business elasticities θ_i ~ N(−2, τ²); s_i ~ U(0.3, 1.2); 1,500–3,000 replications):

| k (businesses) | true τ | known τ (oracle) | plug-in REML EB | grid Bayes, HalfNormal(1) prior, exact quantiles | no pooling |
|---|---|---|---|---|---|
| 4 | 0.5 | — | 0.865 | 0.960 | 0.949 |
| 8 | 0.5 | 0.949 | 0.835–0.837 | 0.953 | 0.954 |
| 8 | 1.0 | — | 0.875 | 0.930 | 0.952 |
| 30 | 0.5 | 0.950 | 0.876–0.878 | 0.933 | 0.950 |
| 30 | 1.0 | 0.950 | 0.937 | 0.943 | — |
| 100 | 0.5 | 0.950 | 0.931 | — | — |

Mean interval width (normal approximation): no pooling 2.94–2.95; grid 2.15 (k=4), 1.87 (k=8, τ=0.5), 2.25 (k=8, τ=1), 1.60 (k=30). That is 23–46% narrower than no pooling, at 92–96% coverage.

Bug found while validating: a non-uniform τ grid *without* Δτ weights dropped coverage to 0.866–0.886. This should be a unit test.

#### 2.6 Nested levels: exact O(N) tree algorithm (validated [C])
Model: θ_root ~ N(m_root, v_root) (literature or meta-regression), θ_v | θ_parent ~ N(θ_parent, τ_{ℓ(v)}²), and observed nodes b_v | θ_v ~ N(θ_v, s_v²).

```
UPWARD (deepest level first). For node v collect factors F_v = {(b_v, s_v²) if observed} ∪ {(a_c, A_c + τ_c²) : child c has data}
   P_v = Σ_f 1/A_f ;  a_v = (Σ_f a_f/A_f)/P_v ;  A_v = 1/P_v           # likelihood of θ_v from all data below v
   log Z_v = −½Σ_f log(2πA_f) + ½log(2πA_v) + a_v²/(2A_v) − Σ_f a_f²/(2A_f)
   log p(all b | τ) = Σ_v log Z_v + log N(a_root ; m_root, A_root + v_root)    # exact marginal likelihood for ML-II
DOWNWARD (root first)
   root: prec = 1/v_root + 1/A_root ; M = (m_root/v_root + a_root/A_root)/prec ; V = 1/prec
   child c of v, with message (a_c, Ã_c = A_c+τ_c²):
      cavity (v without c): Λ = 1/V_v − 1/Ã_c ;  m = (M_v/V_v − a_c/Ã_c)/Λ
      prior of θ_c: N(m, 1/Λ + τ_c²);  posterior prec = 1/(1/Λ+τ_c²) + 1/A_c ; mean = (m/(1/Λ+τ_c²) + a_c/A_c)/prec
   child with no data: posterior = N(M_v, V_v + τ_c²)   ← exactly the cold-start prior predictive of a new business
```
- **Vectorisation.** Process one level at a time with `np.bincount(parent_idx, weights=1/Ã)` and `np.bincount(parent_idx, weights=a/Ã)`. The cost is O(N) per hyperparameter evaluation.
- **Hyperparameters.** Maximise `log p(b|τ) + Σ_ℓ log HalfNormal(τ_ℓ; σ_ℓ)` over ln τ_ℓ with L-BFGS-B (penalised ML-II, in the spirit of Chung et al.), using one τ per level per industry. Then apply the Kass–Steffey correction, or grid τ_B (the most influential) holding the others at their MAP values.
- **Validation [C].** On a 12-node, 4-level tree the tree posterior means and variances matched dense GLS (`m + PHᵀ(HPHᵀ+R)⁻¹(b−Hm)`) to below 1e-15. On a 14-node tree the tree log-likelihood matched `scipy.stats.multivariate_normal.logpdf` exactly (−12.812375363501369).
- **Leaves rule.** Use product-level estimates as leaves and leave the business node latent, *or* use a business-level pooled estimate. Never both, because that double counts the data.
- **Backoff.** Statistically the tree needs no backoff: sparse nodes inherit the parent's posterior plus τ². Backoff is a *display and privacy* rule (§5).

#### 2.7 Crossed effects, product categories, correlated products
- **Industry × region are crossed.** For V1, define tree nodes as combinations: industry → (country within industry, optional level) → (region within industry-country) → (market within industry-region) → business → product. Region-wide effects across industries enter as meta-regression covariates (income, density, country dummies).
- **V2 (genuinely crossed random effects).** Solve the Henderson mixed-model equations `[XᵀR⁻¹X, XᵀR⁻¹Z; ZᵀR⁻¹X, ZᵀR⁻¹Z+G⁻¹][β;u] = [XᵀR⁻¹b; ZᵀR⁻¹b]` with `scipy.sparse.linalg.splu`. Posterior variances need selected inversion, or offline NUTS.
- **Product heterogeneity.** Put the product category (e.g., soft drinks vs maize flour) as a fixed covariate δ_cat in the meta-regression, learned platform-wide, and keep u_ij as residual product heterogeneity within the business.
- **Correlated product estimates within a business.** Pass the stage-1 joint covariance S_i (from a stacked within-tenant regression):
  ```
  b_i | θ_i ~ N(θ_i·1, S_i + τ_P² I)  →  message to business node: P_i = 1ᵀ(S_i+τ_P²I)⁻¹1,  a_i = 1ᵀ(S_i+τ_P²I)⁻¹b_i / P_i
  ```

#### 2.8 Full Bayes vs closed form
- **Online service (per request):** tree passes with cached hyperparameters, plus a grid over τ_B. This is milliseconds of numpy.
- **Offline (monthly):** penalised ML-II over all τ_ℓ and β on summary statistics, plus (optionally) NUTS in NumPyro or PyMC to validate the closed form and to fit variants. Use a non-centred parameterisation:
  ```python
  # NumPyro sketch (not executed in-session; numpyro not installed)
  def model(b, s, cell, n_cells):
      mu = numpyro.sample("mu", dist.Normal(-1.8, 0.8))
      tau = numpyro.sample("tau", dist.HalfNormal(0.7))
      with numpyro.plate("cells", n_cells):
          z = numpyro.sample("z", dist.Normal(0.0, 1.0))
      theta = mu + tau * z                     # non-centred
      numpyro.sample("b", dist.Normal(theta[cell], s), obs=b)
  ```
- **Variants that need MCMC:**
  - Student-t business effects (heavy tails).
  - Truncated-normal sign constraints (Boatwright et al. 1999).
  - Joint modelling of uncertainty in s_i.
  - Crossed effects with posterior variances at scale.
  - The mixture "price-insensitive niche vs competitive" class model.

#### 2.9 Method cards
| Method | Use for | Defaults | Minimum data | Failure modes | Test that catches a wrong implementation |
|---|---|---|---|---|---|
| Normal–normal update | every business/product posterior | τ from snapshot | s_i finite, > 0 | s_i too small (ignored clustering); estimand mismatch | limits: s→0 gives b_i; s→∞ gives the prior; τ→0 gives the pooled mean; posterior var ≤ min(s², τ²+Var μ̂); oracle-τ coverage 0.950 ± 0.005 in simulation |
| DerSimonian–Laird | screening, starting value | — | k ≥ 2 (≥ 10 for stability) | negative bias; boundary at 0 | BCG: Q = 152.2330, τ² = 0.3088, μ̂ = −0.7141 (SE 0.1787) |
| Paule–Mandel | τ for 3–9 groups | brentq, tol 1e-12 | k ≥ 3 | bracket errors; boundary at 0 | BCG: τ² = 0.3181, μ̂ = −0.7150 (SE 0.1809) |
| REML | meta-regression; ≥ 10 groups | fixed point or bounded maximisation | k ≥ 10, ≥ 10 businesses per covariate | boundary at 0; slow fixed point | BCG: τ² = 0.3132, μ̂ = −0.7145 (SE 0.1798); Q-profile 95% CI for τ² [0.1197, 1.1115]; ML τ² = 0.2800 |
| HKSJ / PI | cell-mean CI; new-business PI | t_{k−1}, t_{k−2} | k ≥ 3 | anti-conservative when k ≤ 3 | BCG: HKSJ SE 0.1808; PI(t) [−2.0084, 0.5793]; PI(z) [−1.8667, 0.4376] |
| James–Stein | sanity check | positive part | k ≥ 4 | unequal s | baseball, hits in 45 at-bats = [18, 17, 16, 15, 14, 14, 13, 12, 11, 11, 10, 10, 10, 10, 10, 9, 8, 7]: c = 0.2117 (raw p, σ² = p̄(1−p̄)/45, p̄ = 0.2654; matches Efron & Morris's published 0.212); arcsine version x = √45·arcsin(2p−1) gives c = 0.2090 |
| Grid over τ | production intervals | HalfNormal(0.7); 600–800 points | k ≥ 1 | missing Δτ; upper bound too low; normal approximation; τ-prior scale in the wrong units (HalfNormal(0.7) on eight-schools-scale data forces complete pooling, all means ≈ 7.7 [C]) | eight schools, uniform prior on τ, flat on μ, grid [0.0001, 40] with 4,000 points: E[τ\|y] = 6.52; θ means [11.38, 7.89, 6.14, 7.65, 5.13, 6.14, 10.66, 8.45], SDs [8.31, 6.27, 7.75, 6.54, 6.35, 6.70, 6.78, 7.87]; Monte Carlo agrees to ±0.03; coverage ≥ 0.93 (k=8, τ=0.5) |
| Kass–Steffey | cheap correction | — | τ̂² > 0, k ≥ 10 | undefined at 0 | derivative vs finite difference (0.36519849 for b = −0.4, s² = 0.9, μ = −2, τ² = 0.49) |
| Tree passes | 5-level nesting | per-level τ | any | cavity not removed (double counting); order bugs | equals dense GLS to 1e-10; log-likelihood equals MVN logpdf; permutation invariance; adding a constant c to all b and m_root shifts every posterior mean by c |

Further golden values [C]:
- Eight schools (y = [28, 8, −3, 7, −1, 1, 18, 12], σ = [15, 10, 16, 11, 9, 11, 10, 18]): Q = 4.707 (df 7, p = 0.696); DL = PM = REML = ML τ² = 0; pooled mean 7.686 (SE 4.072).
- BCG data (13 trials, log risk ratio; counts tpos, tneg, cpos, cneg):
  ```
  4/119/11/128, 6/300/29/274, 3/228/11/209, 62/13536/248/12619, 33/5036/47/5761, 180/1361/372/1079, 8/2537/10/619,
  505/87886/499/87892, 29/7470/45/7232, 17/1699/65/1600, 186/50448/141/27197, 5/2493/3/2338, 27/16886/29/17825
  ```
  with log RR = ln[(tpos/(tpos+tneg))/(cpos/(cpos+cneg))] and v = 1/tpos − 1/(tpos+tneg) + 1/cpos − 1/(cpos+cneg). These computed values match the metafor documentation values as I recall them (τ²_REML = 0.3132, Q = 152.2330, CI [0.1197, 1.1115]). Re-run metafor once to certify.

#### 2.10 Reference code (validated logic)
```python
import numpy as np
from scipy import optimize

def pooled(b, s2, tau2):
    w = 1.0 / (s2 + tau2); mu = np.sum(w * b) / np.sum(w)
    return mu, 1.0 / np.sum(w), w

def tau2_dl(b, s2):
    w = 1.0 / s2; mu = np.sum(w * b) / np.sum(w); Q = np.sum(w * (b - mu) ** 2)
    return max(0.0, (Q - (len(b) - 1)) / (np.sum(w) - np.sum(w ** 2) / np.sum(w)))

def tau2_pm(b, s2):
    k = len(b)
    f = lambda t2: np.sum((b - pooled(b, s2, t2)[0]) ** 2 / (s2 + t2)) - (k - 1)
    if f(0.0) <= 0: return 0.0
    hi = max(1e-8, np.var(b, ddof=1))
    while f(hi) > 0: hi *= 2
    return optimize.brentq(f, 0.0, hi, xtol=1e-14)

def tau2_reml(b, s2, tol=1e-12, maxit=10_000):
    t2 = tau2_dl(b, s2)
    for _ in range(maxit):
        mu, _, w = pooled(b, s2, t2)
        new = max(0.0, np.sum(w**2 * ((b - mu)**2 - s2)) / np.sum(w**2) + 1.0 / np.sum(w))
        if abs(new - t2) < tol: return new
        t2 = new
    return t2

def grid_posterior(b, s2, sigma_tau=0.7, n=800, upper=None):
    """sigma_tau=None -> uniform prior on tau (needs k >= 3); else HalfNormal(sigma_tau)."""
    if upper is None:
        upper = 5 * (sigma_tau or 1.0) + 3 * np.sqrt(s2.max())
    grid = np.linspace(0.0, upper, n); dt = np.gradient(grid)
    logp = np.empty(n); M = np.empty((n, len(b))); V = np.empty_like(M)
    for g, t in enumerate(grid):
        mu, vmu, w = pooled(b, s2, t * t)
        log_prior = 0.0 if sigma_tau is None else -0.5 * (t / sigma_tau) ** 2
        logp[g] = (log_prior + 0.5 * np.log(vmu)
                   - 0.5 * np.sum(np.log(s2 + t * t)) - 0.5 * np.sum(w * (b - mu) ** 2))
        B = s2 / (s2 + t * t); M[g] = (1 - B) * b + B * mu; V[g] = (1 - B) * s2 + B**2 * vmu
    pi = np.exp(logp - logp.max()) * dt; pi /= pi.sum()
    mean = pi @ M
    return mean, pi @ (V + M**2) - mean**2, pi, grid
```

### Gaps
- No published estimate was found of the *between-business* SD of elasticities (τ_B) for SMBs, in any region. The default HalfNormal scales (0.4 / 0.4 / 0.7 / 0.8) are engineering judgements to be replaced by REML / ML-II on the platform's seed data.
- Morris's (1983) full EB posterior-variance formula and the general (k − r − 2)/(k − r) form were not verified in-session; only the (m − 3)/(m − 1) factor was.
- Not verified in-session (recalled): Blattberg & George (1991, JASA) shrinkage of brand–store elasticities; Montgomery & Rossi (1999, JMR 36(4):413–423) theory-based priors. Both are directly relevant precedents; confirm citations before quoting.
- The robust-EBCI critical-value formula of Armstrong et al. (2022) was not accessed (Econometrica PDF blocked).

## 3. Published elasticity magnitudes to seed priors, and how to build an honest prior

### Takeaway
- Brand- or firm-level ("selective demand") meta-analytic means are −1.76 (Tellis, 367 estimates) and −2.62 (Bijmolt et al., 1,851 estimates; endogeneity-corrected estimates are larger in magnitude).
- Category or market-level food elasticities are much smaller (|ε| = 0.27–0.81) and about 40% larger in low-income than high-income countries for cereals (−0.61 vs −0.43).
- A single shop faces an elasticity *between* category and brand level. It sits near the brand or store level for undifferentiated goods in dense markets, and nearer the category level where location, credit or relationships lock customers in.
- Build the prior as a meta-analytic-predictive meta-regression fitted on platform stage-1 estimates:
  - literature-anchored industry intercepts;
  - weak, sign-informed slopes for price tier, income, competition and density;
  - a bias allowance by identification tier;
  - predictive variance τ² + x*ᵀVx*, released only if leave-one-business-out calibration passes.

### Cited Findings
- [V] Tellis 1988: 367 price elasticities from about 220 brands or markets; uncorrected arithmetic mean −1.76 "with a very narrow distribution around the mean". Sources: [JSTOR 3172944](https://www.jstor.org/stable/3172944); [MSI working paper](https://www.msi.org/working-papers/the-price-elasticity-of-selective-demand-a-metaanalysis-of-sales-response-models/).
- [V] Bijmolt, van Heerde & Pieters 2005: 1,851 elasticities from 81 studies.
  - Average −2.62.
  - Sales elasticities have increased in magnitude over four decades, while market-share and choice elasticities stayed stable.
  - Endogeneity correction significantly increases magnitude; heterogeneity correction is negligible.

  Sources: [SAGE](https://journals.sagepub.com/doi/10.1509/jmkr.42.2.141.62296); [ResearchGate](https://www.researchgate.net/publication/232906184_New_Empirical_Generalizations_on_the_Determinants_of_Price_Elasticity).
- [V, existence only] A later summary of meta-analytic price, cross-price and advertising elasticity insights; content not accessed. Source: [Elgar Encyclopedia of Pricing, ch. 30](https://www.elgaronline.com/display/book/9781035307319/ch30.xml).
- [V, existence only] A newer meta-analysis of U.S. food demand elasticities testing scanner-data effects. Source: [Applied Economic Perspectives & Policy](https://onlinelibrary.wiley.com/doi/abs/10.1002/aepp.13414).
- [V] Andreyeva, Long & Brownell 2010 (AJPH 100(2):216–222): U.S. food and non-alcoholic beverage elasticities range 0.27–0.81 in absolute value. Food away from home, soft drinks, juice and meats are most responsive (0.7–0.8). Sources: [Duke Scholars](https://scholars.duke.edu/display/pub963616); [PubMed 20019319](https://www.unboundmedicine.com/medline/citation/20019319/The_impact_of_food_prices_on_consumption:_a_systematic_review_of_research_on_the_price_elasticity_of_demand_for_food_).
- [L] Andreyeva et al. 2010, per-category mean |ε| (from 160 studies; the endpoints and the 0.7–0.8 group are confirmed above): food away from home 0.81, soft drinks 0.79, juice 0.76, beef 0.75, pork 0.72, fruit 0.70, poultry 0.68, dairy 0.65, cereals 0.60, vegetables 0.58, fish 0.50, fats/oils 0.48, cheese 0.44, sweets/sugar 0.34, eggs 0.27. Same [Duke Scholars](https://scholars.duke.edu/display/pub963616) record.
- [V] Green et al. 2013 (BMJ): 136 studies, 3,495 own-price food elasticities, 162 countries. A 1% rise in cereal prices reduces consumption by 0.61% (95% CI 0.56–0.66) in low-income countries vs 0.43% (0.36–0.51) in high-income countries. Effects are larger in poorer countries and poorer households. Source: [PubMed 23775799](https://pubmed.ncbi.nlm.nih.gov/23775799/). [V, existence] Companion global meta-analysis including cross-price effects: [Cornelsen et al. 2015, Health Econ](https://onlinelibrary.wiley.com/doi/10.1002/hec.3107).
- [V] Hoch et al. 1995: 67% of store-level elasticity variation is explained by 11 trading-area variables; demographics matter more than competition. Source: [ResearchGate](https://www.researchgate.net/publication/237130017_Determinants_of_Store-Level_Price_Elasticity).
- [V] DellaVigna & Gentzkow: elasticities fall in magnitude with income and are predicted by own-store income. Near-uniform chain pricing means the median chain forgoes about $16m a year versus optimal prices; real firms do not price at their Lerner optimum. Source: [NBER](https://www.nber.org/system/files/working_papers/w23996/revisions/w23996.rev0.pdf).
- [V] Mobile telephony in developing countries: short-run elasticity −0.19, long-run −0.82 (short-run vs long-run evidence). The search summary drew on several sources without mapping numbers to papers. Sources: [UCT thesis](https://open.uct.ac.za/server/api/core/bitstreams/e9df6445-af7f-40a1-b82c-0e227850a952/content); [Info. Econ. & Policy 2021](https://www.sciencedirect.com/science/article/abs/pii/S0167624521000275).
- [L] Lerner condition for a profit-maximising single-product firm: (P − MC)/P = −1/ε, so firm-level |ε| > 1 at an interior optimum. Source: [Lerner 1934, Rev Econ Stud](https://doi.org/10.2307/2967480).
- [L] Online price-comparison settings: demand for the lowest-priced items is extremely elastic, and firms respond with obfuscation. Source: [Ellison & Ellison 2009, Econometrica 77:427–452](https://doi.org/10.3982/ECTA5708). Contrast, online grocery: within-household price sensitivity is *lower* online than offline ([Chu, Chintagunta & Cebollada 2008, Marketing Science](https://doi.org/10.1287/mksc.1070.0288)). Online demand is more elastic for air travel ([Granados, Gupta & Kauffman 2012, ISR](https://doi.org/10.1287/isre.1100.0312)). [V, existence] A 2025 preprint on online shopping and offline price sensitivity: [arXiv 2506.15103](https://arxiv.org/pdf/2506.15103).
- [L] Promotions vs permanent prices:
  - Price promotions have mostly transient effects on category incidence, brand choice and quantity, so post-promotion dips are common ([Pauwels, Hanssens & Siddarth 2002, JMR](https://doi.org/10.1509/jmkr.39.4.421.19114)).
  - Long-run promotion exposure raises consumers' price sensitivity ([Mela, Gupta & Lehmann 1997, JMR](https://doi.org/10.1177/002224379703400205)).
  - Only about a third of a promotional bump comes from brand switching ([van Heerde, Gupta & Wittink 2003, JMR](https://doi.org/10.1509/jmkr.40.4.481.19385)).

  [V, existence] How price elasticities vary over the business cycle: [van Heerde et al., UNC repository](https://cdr.lib.unc.edu/downloads/3n204669g).
- [L] Gasoline, market level, after publication-bias correction: short-run about −0.09 and long-run about −0.31 ([Havranek, Irsova & Janda 2012, Energy Econ](https://doi.org/10.1016/j.eneco.2011.09.003)). Earlier international meta-analysis means: short-run about −0.26, long-run about −0.58 ([Espey 1998, Energy Econ](https://doi.org/10.1016/S0140-9883(97)00013-3)).
- [L] Medical care demand under randomised cost-sharing: elasticity about −0.2 ([Manning et al. 1987, AER 77:251–277](https://www.jstor.org/stable/1804094); JSTOR id from memory).
- [L] Bias modelling: add a variance term (or shift) for lower-quality evidence when synthesising. Source: [Turner, Spiegelhalter, Smith & Thompson 2009, JRSS-A 172:21–47](https://doi.org/10.1111/j.1467-985X.2008.00547.x).
- [L] Meta-regression is generally not advisable with fewer than about 10 studies per modelled covariate. Source: [Cochrane Handbook ch. 10](https://training.cochrane.org/handbook/current/chapter-10).
- [V, existence only] U.S. household appliance price elasticity report: [LBNL / OSTI 929429](https://www.osti.gov/servlets/purl/929429).

### Inferences

#### 3.1 Which elasticity does a single shop face?
- Category or market elasticities (Andreyeva, Green, fuel, telecom) measure how total consumption responds when *all* sellers' prices move. They are a magnitude **lower bound** for a shop's own-price elasticity.
- A shop raising price alone also loses customers to nearby shops. Brand-level meta-analytic means (−1.76 / −2.62) are the closest published analogue for an item at a shop.
- For a multi-item shop, the profit-relevant response also nets out within-store substitution (cannibalisation). The item's own elasticity therefore *overstates* the profit loss from a price rise when substitutes sit in the same store, which argues for later category-level pricing (§31–32).
- The Lerner inverse rule gives a weak consistency anchor. Using contribution margin m = (P − MC)/P [C]:

  | m | 0.15 | 0.20 | 0.25 | 0.30 | 0.40 | 0.50 | 0.60 | 0.70 |
  |---|---|---|---|---|---|---|---|---|
  | implied ε | −6.67 | −5.00 | −4.00 | −3.33 | −2.50 | −2.00 | −1.67 | −1.43 |

  Use it only as a prior predictive check, because SMBs often use cost-plus pricing or inertia (DellaVigna & Gentzkow show even chains deviate). Firm-level prior means should still sit at or below about −1.2.

#### 3.2 Default industry anchors (engineering placeholders, not literature values)
The estimand is the shop-level own-price elasticity of a product or service line at regular prices, over a 4–12-week response.
- m_lit is the industry-mean prior; s_lit is the uncertainty of that mean.
- The predictive SD for a new business adds τ for business and product heterogeneity (0.7 used here) [C].
- These defaults must be overwritten by platform-fitted values after seeding.

| Category | Literature anchor (level) | m_lit | s_lit | Predictive SD | 90% predictive interval | P(ε>0) | P(ε>−1) | Confidence |
|---|---|---|---|---|---|---|---|---|
| Grocery/convenience packaged FMCG item | brand −1.76 / −2.62; category 0.27–0.81 | −1.8 | 0.8 | 1.06 | [−3.5, −0.1] | 0.045 | 0.226 | medium |
| Staples in LMIC informal/convenience retail | household −0.63 to −1.79 (Nigeria), sugar −2.42 (South Africa), cereals −0.61 (low-income countries) | −2.0 | 1.0 | 1.22 | [−4.0, 0.0] | 0.051 | 0.206 | low–medium |
| Café/restaurant menu item | food away from home ≈ −0.8 (category) | −1.4 | 0.8 | 1.06 | [−3.1, 0.3] | 0.094 | 0.353 | low–medium |
| Apparel | brand-level means only | −2.0 | 1.0 | 1.22 | [−4.0, 0.0] | 0.051 | 0.206 | low |
| Consumer electronics/phones | brand −2.62 (durables likely higher; unverified) | −2.5 | 1.0 | 1.22 | [−4.5, −0.5] | 0.020 | 0.110 | low |
| Pharmacy/OTC | medical care ≈ −0.2 (market) | −1.2 | 0.8 | 1.06 | [−2.9, 0.5] | 0.129 | 0.425 | low |
| Hair/beauty/personal services | none found | −1.3 | 0.9 | 1.14 | [−3.2, 0.6] | 0.127 | 0.396 | low |
| Auto repair | none found | −1.3 | 0.9 | 1.14 | [−3.2, 0.6] | 0.127 | 0.396 | low |
| Professional services | none found | −1.2 | 0.9 | 1.14 | [−3.1, 0.7] | 0.146 | 0.430 | low |
| Wholesale/B2B | none found | −2.5 | 1.2 | 1.39 | [−4.8, −0.2] | 0.036 | 0.140 | low |
| E-commerce | very high on comparison sites vs lower for online grocery | −2.5 | 1.2 | 1.39 | [−4.8, −0.2] | 0.036 | 0.140 | low |
| Fuel station (station-level) | market short-run −0.09 to −0.26; station level far more elastic (homogeneous product, posted prices) | −3.0 | 1.5 | 1.66 | [−5.7, −0.3] | 0.035 | 0.113 | low |

**Airtime and data resale agents** usually sell at operator-set face value; do not run price optimisation there. Mobile category elasticities are −0.19 to −0.82 (Section 4).

**Weight of the literature.** s_lit = 0.8–1.0 makes the literature worth about 0.85–1.9 "business-equivalents" (per-business total variance 0.85–1.2) [C]. Platform data dominate as soon as a few businesses exist, which is intended, because the literature comes from other countries, levels and decades.

#### 3.3 Meta-regression MAP prior (the concrete `f(...)`)
```
θ_i = β_ind[i] + δ_cat[i] + β_tierᵀ·tier_i + β_inc·ln(inc_i/inc_ref) + β_comp·comp_z + (β_dens·dens_z ⟂ comp) + γ_tier(i) + u_i
b_i | θ_i ~ N(θ_i, s_i² + δ_tier(i)²),   u_i ~ tree random effects (§2.6)
Fit: W = diag(1/(s_i² + δ² + τ²));  β̂ = (XᵀWX + Σ₀⁻¹)⁻¹(XᵀWb + Σ₀⁻¹β₀);  V_β = (XᵀWX + Σ₀⁻¹)⁻¹;  τ² by REML on ℓ_R
Prior for a new business with covariates x*:  N(x*ᵀβ̂ evaluated at tier = best,  τ_total² + x*ᵀV_βx* [+ τ_shift² if x* outside training hull])
```
Starting priors on slopes (Σ₀ diagonal, all weak, all learned from data):
- **Income.** The Green et al. low- vs high-income cereal ratio (0.61/0.43, ln = 0.3497) over a cross-country income ratio of 20–60× gives d ln|ε|/d ln(income) ≈ −0.085 to −0.117 [C].
  - On the raw scale use β_inc ~ N(+0.10·|m_lit|, (0.10·|m_lit|)²) per ln-unit of income, positive meaning toward zero at higher income (DellaVigna & Gentzkow give the sign).
  - Within a city, a 2–3× income difference therefore moves |ε| by only about 7–11% at the prior (0.10 × ln 2 to 0.10 × ln 3). Larger effects must come from data.
- **Competition.** Use comp_z = standardised ln(1 + competitors weighted by travel time). β_comp ~ N(−0.10·|m_lit|, (0.10·|m_lit|)²) per SD, meaning more competition is more elastic. It is kept smaller than income because Hoch et al. find competition less influential than demographics.
- **Density.** Include only after orthogonalising against competition (otherwise drop it): N(−0.05·|m_lit|, (0.10·|m_lit|)²).
- **Price tier.** Premium N(+0.2·|m_lit|, (0.15·|m_lit|)²); discount N(−0.15·|m_lit|, (0.15·|m_lit|)²) relative to mid-tier. No verified source was found for own-price tier effects (see Gaps).
- **Identification-tier bias allowance.** δ_tier = 0 for Tiers 1–3, 0.2 for Tier 4–5, 0.4 for Tier 6 (FE), 0.6 for Tier 7. Also include a tier shift γ_tier and predict at the best tier, which corrects the attenuation Bijmolt et al. document.
- **Product category.** δ_cat ~ N(0, 0.5²), with literature ordering (e.g., soft drinks more elastic than eggs) used only as a sanity check.
- **Minimum data.** At least 10 businesses per covariate (Cochrane rule of thumb). Below that, drop covariates and keep only industry × region intercepts.

#### 3.4 Price tier, competition, income, density: what is and is not supported
- **Supported (direction):** lower income means more elastic (DellaVigna & Gentzkow; Green et al.); demographics outweigh competition (Hoch et al.); larger magnitude when endogeneity is handled (Bijmolt et al.).
- **Not supported by verified evidence in-session:** the magnitudes of tier, density and competition effects. Treat them as learnable with weak priors, and report their posterior SDs.

#### 3.5 Short-run vs long-run and which estimand to fix
- Promotional (temporary) elasticities mix stockpiling and switching, and their long-run effect is small (Pauwels et al.). Regular-price changes are the SMB decision.
- Long-run responses are larger than short-run for telecom (−0.82 vs −0.19) and fuel (−0.31 vs −0.09 corrected).
- **Canonical estimand:** the 4–12-week response to a regular-price change. Report, but do not pool, short-run (0–2 weeks) and promotional estimates.

#### 3.6 Honesty gates for the prior (release criteria)
For each held-out business j (leave-one-tenant-out) compute z_j = (b_j − m_j)/√(v_j + s_j²).
- Require mean(z) ∈ [−0.1, 0.1], SD(z) ∈ [0.9, 1.1] and 80%/95% coverage within ±2 pp for n ≥ 200 businesses, per industry where n allows.
  - SD(z) > 1.1 means the prior is overconfident: inflate τ.
  - SD(z) < 0.9 means the prior is too wide.
- Also check calibration by country, income tercile and tier.

### Gaps
- No verified shop-level or SMB elasticity estimates were found for hair/beauty salons, auto repair, professional services, wholesale/B2B, apparel or consumer electronics. The search budget ran out before these queries. Those rows in the anchor table are judgement placeholders.
- Bijmolt et al.'s durables vs non-durables and life-cycle findings, their median, and Tellis's category breakdown were not verified in-session.
- Hoch et al.'s per-variable coefficients and DellaVigna & Gentzkow's numeric elasticity–income slope and elasticity distribution were not accessible (PDF fetches blocked). Only directions and the 67% and $16m figures are verified.
- Restaurant-specific (full-service vs limited-service) elasticities (Okrent & Alston, USDA ERS) were not verified.
- No meta-analysis later than 2005 for brand-level price elasticities was confirmed with numbers. The Elgar chapter and the AEPP meta-analysis exist, but their content was not accessed.

## 4. Price elasticity evidence in African and other low- and middle-income markets

### Takeaway
Household and category elasticities for staples in sub-Saharan Africa are larger in magnitude than in rich countries:
- cereals −0.61 vs −0.43 across income groups;
- Nigerian household own-price elasticities −0.63 to −1.79 (local rice about −1.7);
- South African carbonated soft drinks −1.18 and sugar −2.42;
- Kenyan elasticities negative for all food groups, with several staples above unity.

Mobile telephony is price-inelastic (about −0.2 to −0.3 short-run, about −0.8 long-run). No peer-reviewed *shop-level* elasticities for kiosks, dukas or open markets were found, so the platform's own data must supply that level, with LMIC literature only shaping category-level hyperpriors and the income slope.

### Cited Findings
- [V] Green et al. 2013: larger price responses in low-income countries and poorer households; cereals −0.61 in low-income vs −0.43 in high-income countries. Source: [PubMed 23775799](https://pubmed.ncbi.nlm.nih.gov/23775799/).
- [V] Kenya: a QUAIDS model on nationally representative household survey data gives significant negative own-price elasticities for all food groups. The authors conclude that income-oriented policies matter more than price-oriented ones. Source: [Korir, Rizov & Ruto 2020, Economic Modelling 92:99–108](https://ideas.repec.org/a/eee/ecmode/v92y2020icp99-108.html) (open access: [EconStor](https://www.econstor.eu/bitstream/10419/224501/1/EM_Korir_Rizov_Ruto.pdf)).
- [V] Kenya, zero-adjusted QUAIDS: staple cereals (rice, maize, wheat, beans) are about 27% of household food spending; price elasticities for rice, wheat and beans exceed unity in absolute value; maize and beans are necessities. The search summary blended Kenyan sources, so confirm which paper each number belongs to. Sources: [J. Dev. & Agric. Econ. 17(4) 2025](https://academicjournals.org/journal/JDAE/article-abstract/6FC5D1173999); also [Food consumption in Kenya: demand-system elasticities](https://www.researchgate.net/publication/360920560_Food_Consumption_in_Kenya_Demand_System_Elasticities_and_Implications_Towards_Healthy_Eating).
- [V] Nigeria: three waves of the General Household Survey (LSMS-ISA, 2010–2016) with QUAIDS. Own-price elasticities range −0.63 to −1.79; local rice is −1.64 (compensated) and −1.69 (uncompensated). Another LSMS analysis classifies imported rice as a luxury and local rice as a necessity. Source: [Farm Households' Demand Response to Escalating Food Prices in Nigeria, J. Applied Econ.](https://www.tandfonline.com/doi/full/10.1080/15140326.2021.1980351). Related: [food price changes and welfare in Nigeria](https://www.tandfonline.com/doi/full/10.1080/15140326.2020.1743103); [rice demand QUAIDS](https://archive.aessweb.com/index.php/5005/article/download/2038/3253).
- [V] South Africa: censored QUAIDS combining expenditure-survey data with sub-national prices. Own-price elasticities: carbonated soft drinks −1.18, concentrates −1.17, milk −1.1, sugar −2.42; fruit juice and tea/coffee not significant. Source: [Stacey, Tugendhaft & Hofman 2017, Preventive Medicine](https://www.sciencedirect.com/science/article/pii/S0091743517301871). [V, existence] An observational evaluation of beverage purchases after the Health Promotion Levy: [Lancet Planetary Health 2021](https://www.sciencedirect.com/science/article/pii/S2542519620303041). Newer South African food/drink demand-system elasticities: [Dare et al. 2025](https://journals.sagepub.com/doi/10.1177/22799036251350956). Cigarette elasticity via the Deaton method: [PMC8685941](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8685941/). [UNU-WIDER, Income and price elasticities in South Africa](https://www.wider.unu.edu/publication/income-and-price-elasticities-demand-south-africa).
- [V, existence] Nigerian sugar-sweetened beverage elasticities (AIDS model): [academia.edu](https://www.academia.edu/105683907/Effect_of_price_and_income_on_the_demand_for_sugar_sweetened_beverages_in_Nigeria_an_analysis_of_household_consumption_data_using_an_almost_ideal_demand_system_AIDS_). Kenyan maize supply and demand responsiveness (ARDL): [Cogent Food & Agriculture 2021](https://www.tandfonline.com/doi/full/10.1080/23311932.2021.1957318).
- [V, attribution unclear] Mobile services: developing countries short-run −0.19 and long-run −0.82; an average of −0.27 across eight African countries (Botswana, Ghana, Kenya, Mozambique, Nigeria, South Africa, Tanzania, Zambia); telephone demand price- and income-inelastic in both the short and long run in African countries. The search summary drew on these sources without mapping each number to one: [UCT](https://open.uct.ac.za/server/api/core/bitstreams/e9df6445-af7f-40a1-b82c-0e227850a952/content); [IEP 2021 firm-level](https://www.sciencedirect.com/science/article/abs/pii/S0167624521000275); [IEP 2021 SSA regulation and prices](https://www.sciencedirect.com/science/article/abs/pii/S0167624521000512); [IDRC telecom elasticities](https://idl-bnc-idrc.dspacedirect.org/bitstreams/0158d327-10e1-4c87-bbfc-87e8770d33cb/download).
- [L] Giffen behaviour is real but confined to subsistence staples among very poor households. Randomised staple-price subsidies show it strongly for rice in Hunan and weakly for wheat in Gansu (China). Source: [Jensen & Miller 2008, AER 98(4):1553–1577](https://doi.org/10.1257/aer.98.4.1553).
- [L] Data sources for LMIC demand systems: [World Bank LSMS / LSMS-ISA](https://www.worldbank.org/en/programs/lsms); [USDA ERS International Food Consumption Patterns](https://www.ers.usda.gov/data-products/international-food-consumption-patterns/); [USDA ERS Commodity and Food Elasticities](https://www.ers.usda.gov/data-products/commodity-and-food-elasticities/).

### Inferences
- **Level mismatch.** Every African number above is a household-level, market-wide elasticity (the response to category price changes). A kiosk's own-price elasticity for an identical branded sachet should be *larger in magnitude*, because neighbouring kiosks sell the same item. Three frictions push the other way: informal credit ("book" accounts), proximity in dense settlements, and small-pack or daily purchasing. Their net effect is unknown, so the staples default is −2.0 with s_lit = 1.0 (wide).
- **Income interaction via Slutsky.** ε_uncomp = ε_comp − w·η (w = budget share, η = expenditure elasticity).
  - Example: maize flour with a 0.20 budget share and η = 0.7 contributes −0.14 of income effect to the household-level elasticity.
  - For a single shop's item the relevant budget share is tiny, so the income-effect channel is negligible at the shop level. The larger LMIC magnitudes at category level partly reflect budget share.
  - Implement income as the weak ln-income slope (§3.3), not as a large multiplicative LMIC factor at shop level.
- **LMIC category hyperpriors.** Where the platform estimates category-level elasticities (e.g., for market-level demand models or §57 shocks), scale high-income-country literature magnitudes by about 1.4 for low-income catchments (the Green et al. cereals ratio). Prior SD should be at least 0.3.
- **Inflation and nominal prices.** In high-inflation economies, frequent nominal changes are not real price changes. Stage-1 regressions must use prices deflated by a local CPI or relative to competitor prices (§10 CPI), or the elasticity is confounded with inflation.
- **Telecom.** Airtime is resold at face value, so agents are not price setters. Data-bundle pricing is operator-level. Exclude both from SMB price optimisation, or treat them as fixed-price pass-through.
- **Positive estimates on staples in very poor catchments** are the one place where a genuinely positive market-level slope is conceivable (Jensen & Miller). A *shop-level* positive slope remains implausible because substitutes exist next door. Treat it as an identification failure unless an experiment says otherwise (§6).
- **Country tags.** Use country as a top split under industry (industry → country → region → market). Elasticity levels, currency denominations and informality differ, and country-level τ is estimable once 3 or more countries have data.

### Gaps
- No peer-reviewed shop-level elasticity estimates for informal retail (kiosks, dukas, spazas, open markets) in Kenya, Nigeria, Ghana or South Africa were found. No FMCG retail-audit (NielsenIQ/Kantar) elasticities were found (proprietary and not searchable).
- Ghana-specific demand-system elasticities were not retrieved (search budget exhausted).
- Fuel and public-transport (matatu, boda-boda, danfo) fare elasticities for African cities were not retrieved. Only global gasoline meta-analyses ([L]) are cited.
- Income elasticities for Africa (e.g., a meta-analysis of food income elasticities across Africa, recalled as Colen et al. 2018, Food Policy) and USDA ERS cross-country own-price values (Muhammad et al. 2011) were not verified.
- The effects of price points, coin and note denominations, and pack-size or price-point thresholds on SMB demand in African markets are plausible but unsourced.

## 5. Learning population-level parameters across tenants without pooling raw data (§80)

### Takeaway
Only per-business summary records cross the tenant boundary: (b, s, n_eff, G, tier, estimand id, taxonomy and location keys). For Gaussian hierarchies these are near-sufficient. Inverse-variance two-stage pooling is asymptotically as efficient as pooling raw data for common effects, and exactly equal under homoskedastic within-business OLS. One-shot aggregation therefore replaces federated learning; there is no FedAvg loop.

Protect released aggregates with:
- leave-one-tenant-out priors;
- k_min ≥ 10 businesses per displayed cell;
- a precision-dominance cap;
- monthly batching against differencing;
- formal differential privacy (Laplace, ε ≈ 1) for externally displayed benchmarks with n ≥ 50.

Below about 20 businesses per cell, differential-privacy noise (SD ≥ 0.5) swamps the signal (τ ≈ 0.7).

### Cited Findings
- [L] Meta-analysis of summary statistics is asymptotically as efficient as individual-level analysis for estimating common effects (under regularity conditions). Source: [Lin & Zeng 2010, Biometrika 97:321–332](https://doi.org/10.1093/biomet/asq006).
- [L] One-stage (raw-data) and two-stage (summary) individual-participant-data meta-analyses usually agree. They can differ when within-study samples are small or estimation assumptions differ, which is exactly the SMB regime. Source: [Burke, Ensor & Riley 2017, Stat Med 36:855–875](https://doi.org/10.1002/sim.7141).
- [L] FedAvg exists for iterative training of non-linear models on decentralised data. Source: [McMahan et al. 2017, AISTATS](https://proceedings.mlr.press/v54/mcmahan17a.html). Secure aggregation hides individual updates from the server: [Bonawitz et al. 2017, ACM CCS](https://doi.org/10.1145/3133956.3133982).
- [L] The Laplace mechanism releases f(D) + Lap(Δ₁/ε) and is ε-DP. Source: [Dwork, McSherry, Nissim & Smith 2006, TCC](https://doi.org/10.1007/11681878_14). The Gaussian mechanism with σ = Δ₂√(2 ln(1.25/δ))/ε (ε < 1) is (ε, δ)-DP; composition and post-processing theorems apply. Source: [Dwork & Roth 2014](https://doi.org/10.1561/0400000042). Library: [OpenDP](https://opendp.org/).
- [L] Statistical disclosure control for magnitude tables uses minimum-frequency, (n,k)-dominance and p% rules. Source: [Hundepool et al. 2012, Statistical Disclosure Control, Wiley](https://doi.org/10.1002/9781118348239).
- [V, title] Precision-dependent effects and other EB pitfalls in applied settings: [Walters, NBER w33091](https://www.nber.org/system/files/working_papers/w33091/w33091.pdf).

### Inferences

#### 5.1 Architecture (replaces §80 text)
1. **Seed job, per tenant, inside the tenant partition.** For every business-product, compute the canonical stage-1 estimate (b, s) with clustered or HAC SE, plus n_obs, n_eff, G, sd(ln p), tier, period, taxonomy keys (industry, sub-industry, product category, price tier) and coarse location keys (country, region, market id, H3 resolution ≤ 7). Write *only* these rows to a `population_summaries` table. This is how the 1.2M raw rows seed priors without leaving tenant isolation.
2. **Population fit (monthly, versioned).** Run REML or penalised ML-II for τ_ℓ and β on the summaries (§2–3). Store the hyperparameter snapshot H_v plus per-cell aggregates (Σw, Σwb, Σw², n, tenant ids) so that leave-one-tenant-out values cost O(1).
3. **Serving.** Tenant i's prior is computed leave-one-tenant-out from H_v, excluding all of that tenant's businesses. The posterior update uses only tenant i's own stage-1 data. Record the H_v id with each estimate (Rule 11).
4. **Flywheel (§79).** Experiment results (Tier 1) enter `population_summaries` with δ_tier = 0 and so carry the most weight.

#### 5.2 Exactness of summaries
- With business-specific intercepts and controls partialled out within the tenant, the pooled common-slope OLS satisfies β̂ = Σ_i Sxy_i / Σ_i Sxx_i = Σ_i Sxx_i·b_i / Σ_i Sxx_i. That is identical to inverse-variance pooling of b_i when the residual variance is common [derivation].
- Sharing (b, s) therefore loses nothing for that model. Sharing (Sxx, Sxy, Syy, n) is equivalent but exposes slightly more (the price and quantity variances).

#### 5.3 Leakage risks and controls
- **Small cells.** With 2 businesses, A's leave-one-out prior equals B's estimate. Rules:
  - Display a cell-level prior only if n_LOO ≥ k_min = 10 businesses from at least 5 distinct tenants; otherwise display the parent level.
  - The statistical posterior may still use the cell; only display and export are gated.
- **Dominance.** If one business holds more than 50% of a cell's total precision weight Σw, cap its weight at 50% for released aggregates, or suppress the cell.
- **Differencing across versions.** Batch updates monthly, release only rounded values (0.1) and ranges, and never release cell aggregates whose membership changed by a single business without noise.
- **Formal DP (external benchmarks or partner exports).** Use the equal-weight clipped mean, clip each business's b to [L, U] = [−6, 1], treat n as public, under replace-one adjacency: Δ = (U − L)/n. Laplace SD = √2·(U − L)/(n·ε) [C]:

  | n businesses | ε = 0.5 | ε = 1 | ε = 2 |
  |---|---|---|---|
  | 10 | 1.98 | 0.99 | 0.50 |
  | 20 | 0.99 | 0.50 | 0.25 |
  | 50 | 0.40 | 0.20 | 0.10 |
  | 100 | 0.20 | 0.10 | 0.05 |
  | 500 | 0.04 | 0.02 | 0.01 |

  - Precision-weighted means are data-dependent. Use a noisy numerator Σw̄(b̄ − m_lit) with w̄ = min(w, w_cap) (Δ = w_cap·max|b̄ − m_lit|) divided by a noisy denominator Σw̄ (Δ = w_cap), splitting ε between the two (post-processing keeps DP).
  - For scalar releases the Gaussian mechanism is worse than Laplace: its multiplier √(2 ln(1.25/δ)) = 5.30 at δ = 1e-6, versus √2 for Laplace [C].
  - Track per-cell ε across releases (sequential composition).
- **Unit of privacy = tenant.** A tenant with several businesses in a cell contributes one averaged value, so each tenant contributes at most one bounded quantity per cell.
- **What is not needed.** Secure aggregation is unnecessary when the platform itself computes the summaries (the operator already processes tenant data). It becomes relevant only if summaries are computed on tenant-controlled infrastructure or by partners.

#### 5.4 Failure modes and tests
- **Two-stage normality.** Small-n businesses (G ≤ 3 price spells) give non-normal b with unreliable s.
  - Exclude them from *hyperparameter* fitting; they keep their own floored posteriors (§6.2).
  - For G ≥ 4, inflate s² by ν/(ν − 2) with ν = G − 1.
- **Precision-dependence.** Add ln s_i as a meta-regression covariate; a significant slope means "small businesses differ", not merely noise.
- **Tests.**
  1. *Isolation:* recompute tenant A's prior after deleting tenant A's summaries; it must be bit-identical to the served prior.
  2. *Exactness:* simulated OLS businesses — summary pooling equals raw-data pooled OLS to 1e-10.
  3. *DP sensitivity:* brute-force replace-one neighbours over random datasets; the maximum |Δ output| must be ≤ Δ.
  4. *Suppression:* any cell with n_LOO < 10 must never appear in export payloads.

### Gaps
- No source was found on cross-tenant elasticity learning in commercial SaaS specifically. The architecture is assembled from the meta-analysis and differential-privacy literatures.
- Legal and regulatory constraints (Kenya Data Protection Act 2019, Nigeria NDPA 2023, South Africa POPIA) on using tenant data to build shared priors were not researched; this needs legal review. The law names are recalled, not verified in-session.
- Formal DP for τ² (variance) releases was not worked out in detail. It needs a clipped noisy sum of squares, and its utility at small n is poor.

## 6. Cold start, posterior speed, shrinkage diagnostics, prior widths, and near-zero or positive estimates

### Takeaway
- Before data, show the meta-analytic-predictive prior honestly: a range, labelled as a benchmark from N similar businesses, with no personalised claim. P(ε > 0) is about 2–15% at the defaults.
- The posterior should move in proportion to *information*, (n × Var ln p) and the number of independent price changes, not transaction count. At daily noise 0.4–0.5, ±5% price variation needs about 6–12 months to give own data 50–65% weight, while ±10% variation gets there in about 3 months [C].
- Diagnose with shrinkage factors, pooling factors, prior-data conflict p-values, a robust mixture prior, prior predictive and Lerner checks, and simulation-based calibration.
- Treat positive or near-zero shop-level estimates as identification failures unless they are experimentally replicated.

### Cited Findings
- [L] Prior-data conflict check: compare the observed sufficient statistic with its prior predictive distribution. Source: [Evans & Moshonov 2006, Bayesian Analysis 1:893–914](https://doi.org/10.1214/06-BA129).
- [V] Robust MAP mixture prior with a vague component; the weight governs discounting under conflict. Source: [Schmidli et al. 2014](https://onlinelibrary.wiley.com/doi/abs/10.1111/biom.12242). [V, existence] Newer adaptive-borrowing variants: [SAM self-adapting mixture prior, Biometrics 2023](https://academic.oup.com/biometrics/article/79/4/2857/7587575); [modified rMAP, medRxiv 2023](https://www.medrxiv.org/content/10.1101/2023.01.28.23285146v1.full); [EB rMAP, arXiv 2109.10237](https://arxiv.org/pdf/2109.10237); [WAIC-gated mixture priors, arXiv 2510.05085](https://arxiv.org/pdf/2510.05085); [Bayesian dynamic borrowing, arXiv 2609.13976](https://arxiv.org/pdf/2609.13976).
- [L] Pooling factor per level: λ = 1 − Var_k(E[θ_k|y]) / E[Var_k(θ_k)]. λ ≈ 0 means no pooling; λ ≈ 1 means complete pooling. Source: [Gelman & Pardoe 2006, Technometrics 48:241–251](https://doi.org/10.1198/004017005000000517).
- [L] Prior predictive checks and visual workflow: [Gabry et al. 2019, JRSS-A](https://doi.org/10.1111/rssa.12378); [Gelman et al. 2020, Bayesian Workflow](https://arxiv.org/abs/2011.01808). Simulation-based calibration for validating inference code: [Talts et al. 2018](https://arxiv.org/abs/1804.06788).
- [V] Endogeneity correction makes elasticities larger in magnitude, so naive estimates are attenuated toward zero or positive. Source: [Bijmolt et al. 2005](https://journals.sagepub.com/doi/10.1509/jmkr.42.2.141.62296).
- [L] Genuine Giffen behaviour is limited to subsistence staples among the extremely poor. Source: [Jensen & Miller 2008](https://doi.org/10.1257/aer.98.4.1553).
- [L] Truncated or sign-constrained hierarchical priors for elasticities. Source: [Boatwright, McCulloch & Rossi 1999](https://doi.org/10.1080/01621459.1999.10473859).
- [V] Variance estimates at the zero boundary need regularisation. Source: [Chung et al. 2013](https://link.springer.com/article/10.1007/s11336-013-9328-2).

### Inferences

#### 6.1 Presenting prior-only estimates honestly
- **Content.** "Estimated price sensitivity −1.8 (likely range −3.5 to −0.1). Benchmark from 46 similar cafés in Nairobi; not yet measured for your business. A ±10% price test over 8 weeks would cut this range by about 44%."
- **Tagging.** Label the evidence "Tier 0 – benchmark" in the §76 hierarchy.
- **Guardrails.**
  - Never present a prior-only number as "your elasticity".
  - Never output a price recommendation larger than an exploratory step (≤ 5%) from prior-only evidence (link to §66 and §77).
  - Confidence (§45) for prior-only output must be capped; for example, data weight = 0 gives at most "low".
- **Data weight.** Show "Based X% on your own data", where X = 100·(1 − B_i), or in the multi-level case the posterior-precision share contributed by own data.

#### 6.2 How fast the posterior should move [C]
SE of own-data elasticity, assuming i.i.d. daily noise and exogenous variation: s ≈ σ_e/(√n·sd(ln p)). Own-data weight w = τ²/(τ² + s²) with τ = 0.7:

| σ_e | sd(ln p) | n = 30 | n = 90 | n = 180 | n = 365 |
|---|---|---|---|---|---|
| 0.3 | 0.03 | s 1.83, w 0.13 | 1.05, 0.31 | 0.75, 0.47 | 0.52, 0.64 |
| 0.3 | 0.05 | 1.10, 0.29 | 0.63, 0.55 | 0.45, 0.71 | 0.31, 0.83 |
| 0.3 | 0.10 | 0.55, 0.62 | 0.32, 0.83 | 0.22, 0.91 | 0.16, 0.95 |
| 0.5 | 0.03 | 3.04, 0.05 | 1.76, 0.14 | 1.24, 0.24 | 0.87, 0.39 |
| 0.5 | 0.05 | 1.83, 0.13 | 1.05, 0.31 | 0.75, 0.47 | 0.52, 0.64 |
| 0.5 | 0.10 | 0.91, 0.37 | 0.53, 0.64 | 0.37, 0.78 | 0.26, 0.88 |

- Doubling price variation is worth quadrupling the observations.
- **Halving the prior SD** requires s = SD_prior/√3 = 0.577·SD_prior. For a prior SD of 0.8 and σ_e = 0.4 that is 300 days at ±5% or 75 days at ±10%. For a prior SD of 1.0 it is 192 or 48 days [C].
- **Experiment value for §78** (prior SD 0.8, σ_e = 0.4, alternating prices) [C]:
  - ±5%: 4, 8 and 12 weeks give posterior SD 0.707, 0.641 and 0.590 (12%, 20% and 26% reduction).
  - ±10%: 4, 8 and 12 weeks give 0.549, 0.444 and 0.383 (31%, 44% and 52% reduction).
- **Independent price changes.** With serial correlation the effective n is about the number of price spells G, not days. Use clustered-by-spell SEs.
  - If G ≥ 4, inflate s² by ν/(ν − 2), ν = G − 1 (the t-to-normal variance match).
  - If G ≤ 3, exclude the record from hyperparameter fitting. For the business's own posterior, use the floor s² × 3 (the ν = 3 value) and label it "early signal – not used for price recommendations".
  - If sd(ln p) < 0.02, the data are uninformative: report "not measurable yet".
- **Time drift (optional).** θ_i,t = θ_i,t−1 + η_t with η ~ N(0, q²), q ≈ 0.05 per month: a Kalman update keeps the posterior responsive after regime changes (§21).

#### 6.3 Shrinkage diagnostics (log per estimate and per level)
- **Per estimate:** B_i, own-data weight, posterior/prior SD ratio, and the level that dominates the prior.
- **Per level:** the pooling factor λ_ℓ (Gelman & Pardoe), τ̂_ℓ with its posterior SD, the fraction of cells at τ̂ = 0, and I² within cells.
- **Alerts:**
  - λ_ℓ > 0.95 everywhere: the level is uninformative; consider dropping it.
  - τ̂ posterior piling up at the HalfNormal prior's tail: the model is misspecified.
- **Prior-data conflict.** p_i = 2Φ(−|b_i − m_i|/√(v_i + s_i²)) (Evans & Moshonov form for the normal case).
  - Flag if p_i < 0.01. Then switch to the rMAP prior: π(ε) = 0.8·N(m, v) + 0.2·N(m_parent, 2.0²).
  - The posterior is closed form:
    ```
    for component k: w_k' ∝ w_k·N(b; m_k, v_k + s²),  v_k' = 1/(1/v_k + 1/s²),  m_k' = v_k'(m_k/v_k + b/s²)
    ```
  - Test: as |b − m| grows, the posterior weight must shift monotonically to the vague component, and the posterior mean must approach b.

#### 6.4 Prior predictive checks (run at each snapshot)
1. Simulate ε from the prior for each business; the implied % change in Q for +10% price is 1.1^ε − 1. Check the share of businesses for which a price rise *increases* revenue (P(ε > −1)), against how often such businesses exist in the data.
2. **Lerner check.** For businesses that report margins, compare the prior median −1/m against the prior. A gap above about 1.5 elasticity units flags either a mis-centred prior or systematic under-pricing.
3. **Prior mass above zero.** Require P(ε > 0) ≤ 0.05 for retail and FMCG priors and ≤ 0.15 for services (the defaults in §3.2 satisfy this).
4. **Simulation-based calibration.** Draw hyperparameters from their priors, then business θ, then (b, s), then run the full pipeline. Rank statistics must be uniform, and 90% interval coverage must fall within the binomial tolerance.

#### 6.5 Near-zero or positive estimates (Giffen-like noise vs real)
- **Causes, in order of prior likelihood for SMB data:**
  1. Price endogeneity (demand-driven price setting), the attenuation Bijmolt et al. document.
  2. Omitted promotions, advertising or seasonality.
  3. Stockouts or censoring (§63–64).
  4. Unit-price measurement error, such as revenue/quantity across mixed pack sizes, which attenuates toward 0.
  5. Quality or assortment changes at the same time as price changes.
  6. Genuine price-quality or prestige effects (rare).
  7. Giffen behaviour, which is market-level and subsistence staples only (Jensen & Miller), not shop level.
- **Rules:**
  1. Never truncate b_i before pooling; truncation biases μ̂ and τ̂.
  2. Pool on the raw scale; the hierarchy pulls noisy positives toward the negative mean automatically. In the in-session tree example, a raw +0.4 (SE 1.2) became −1.24 (SD 0.65) [C].
  3. At the decision layer, report P(ε > 0) and P(ε > −1). If P(ε > 0) > 0.10, mark the estimate "identification suspect", recommend no price change, and propose an experiment.
  4. Mark a positive estimate as *real* only if it comes from Tier 1–2 evidence, replicates, and passes the conflict check against a vague component; then let the rMAP vague component carry it.
  5. Exclude flagged positives from hyperparameter fitting, or give them δ_tier = 0.6.
  6. If |ε| < 1 at the posterior, the optimiser will push price to its bounds. Cap per-cycle changes (§66) and require experimental confirmation.
- **Sign constraint options (decision layer only):** a truncated-normal posterior at ε ≤ 0, or a lognormal on −ε for display. Keep the pooling model Gaussian to preserve closed form and unbiased hyperparameters.

#### 6.6 Sensible prior widths
- New business in a known industry with platform data: predictive SD about 0.8–1.0. This is τ_M, τ_B, τ_P combined plus parameter uncertainty, once fitted.
- New industry or country with literature only: predictive SD about 1.1–1.7 (§3.2).
- A prior SD below 0.5 for a business with no own data is almost certainly overconfident unless LOBO calibration (§3.6) proves otherwise.
- A prior SD above 2 makes the prior useless; widen only via the rMAP vague component, not the main component.

### Gaps
- No empirical study was found on how often published or SMB elasticity estimates are positive, or what share of those are real versus artefacts.
- The q (drift) default and the rMAP weight w = 0.2 are judgement calls. The follow-up rMAP literature notes there is no clear guidance for quantifying w. Tune both by walk-forward backtesting (§47).
- The noise levels in §6.2 (σ_e = 0.3–0.5 for daily log quantity) are illustrative. Compute the platform's actual residual SDs by industry from the seed data before fixing gating thresholds.
