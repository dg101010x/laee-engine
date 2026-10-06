# Demand-data pitfalls and demand modelling for small-business price elasticity (LAEE v2 input)

Scope and evidence tags. These notes feed the expanded LAEE algorithm specification. WebFetch was blocked by the egress policy for every research domain tried (SSRN, Springer, arXiv, INFORMS PDFs at Columbia, MSI, gwern, open-meteo.com, otexts.com, uchile.cl), and the shared WebSearch budget ran out near the end. Claims therefore rest on (a) search-result summaries of primary sources, (b) library source code inspected locally (statsforecast 2.1.1, holidays 0.106, hijridate 2.6.0, statsmodels 0.15.0, all downloaded from PyPI), and (c) my own Monte Carlo simulations (numpy/scipy/statsmodels; scripts in the session scratchpad `.../scratchpad/sims/`, data-generating processes (DGPs) restated below so they can be rebuilt as tests). Tags: **[snippet]** = taken from a search-result summary, full text not opened; **[recalled]** = established literature I recall but could not re-open this session (verify before quoting); **[pkg]** = verified by reading installed package source; **[own sim]** = my simulation. Default parameters marked "default" are engineering judgement unless a source is attached.

Common simulation DGP (used repeatedly below, "base DGP"): one product, 365 days, Poisson daily demand $D_t\sim\text{Pois}(\mu_t)$, $\ln\mu_t=\ln\mu_0+\beta\ln(P_t/100)+\text{dow}_t$, true $\beta=-1.5$, day-of-week effects (Mon..Sun) $=[0,-0.1,-0.1,0,0.15,0.3,0.1]$, piecewise-constant price $100\to110\to95$ with 2 permanent changes at random days in $[60,305)$ ("few price changes"), 150–300 replications.

---

## Q0. What is wrong or underspecified in the v1 spec (§31–32, §58–65, and the sections they depend on: §4, §14, §16–17, §20, §43, §48)?

### Takeaway
v1 names the right hazards but specifies no estimator, threshold, detection rule or data contract for any of them, and its core estimand (OLS or DML on $\ln Q$, §14/§16) is undefined on zero-sales days and badly biased toward zero for slow movers. Several v1 prescriptions would actively damage estimates: dropping stockout days, using stockout, inventory or rolling demand as nuisance controls, Price×Holiday/Weekend interactions, and 30-day rolling elasticities on data with 1–3 price changes.

### Cited Findings
- §14 and §16 define $Y=\ln Q$ and estimate $\beta$ by OLS or DML residual-on-residual ([v1 §14 L545–581, §16 L607–657](/home/user/laee-engine/docs/spec/v1-original.md)). The same spec targets data where many days have $Q=0$, so $\ln 0$ is undefined. Using $\log(1+Q)$ instead gives effects that depend on the units of $Q$ and "can be any magnitude" when price moves the extensive margin ([Chen & Roth 2024, QJE 139(2):891–936](https://bibbase.org/network/publication/chen-roth-logswithzerossomeproblemsandsolutions-2024)) [snippet]. OLS on a log-linearised constant-elasticity model is inconsistent under heteroskedasticity, while Poisson pseudo-ML (PPML) is consistent and handles zeros ([Santos Silva & Tenreyro 2006, REStat](https://personal.lse.ac.uk/tenreyro/jensen08k.pdf)).
- §63 says "Either censor those observations or model latent demand" and §64 says "use a censored-demand model", but neither names a model, a likelihood or a way to detect stockouts when inventory is missing ([v1 §63 L2143–2167, §64 L2171–2181](/home/user/laee-engine/docs/spec/v1-original.md)). The literature shows that naive sales-based demand rates are biased "even for items that have very few occurrences of stock-outs" ([Anupindi, Dada & Gupta 1998, Marketing Science 17(4)](https://pubsonline.informs.org/doi/10.1287/mksc.17.4.406)).
- §17 puts `inventory`, `stockout`, `lagged demand` and `rolling demand` among the DML nuisance features ([v1 §17 L677–703](/home/user/laee-engine/docs/spec/v1-original.md)). Stockouts are caused by demand exceeding stock, and price drives demand, so a stockout is a post-treatment variable (see Inferences).
- §62 lists promotion flags but gives no rule for separating regular-price from promotional response ([v1 §62 L2123–2139](/home/user/laee-engine/docs/spec/v1-original.md)). Static demand estimates built on temporary price cuts overstate own-price elasticities by about 30%, understate cross-price elasticities by up to 5×, and overstate substitution to the outside option by over 200% ([Hendel & Nevo 2006, Econometrica](https://www.econometricsociety.org/publications/econometrica/2006/11/01/measuring-implications-sales-and-consumer-inventory-behavior)). In a grocery field experiment the average experimental elasticity was −0.34, against about −2.0 from observational price variation ([Bray, Sanders & Stamatopoulos, Unjournal evaluation summary](https://unjournal.pubpub.org/pub/evalsumbraybray); [SSRN 4899765](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4899765)) [snippet].
- §61 models only competitor-relative ("external") reference prices ([v1 §61 L2098–2119](/home/user/laee-engine/docs/spec/v1-original.md)). Gain-loss asymmetry (losses weigh more than gains) is a published empirical generalisation ([Kalyanaram & Winer 1995, Marketing Science 14(3)](https://doi.org/10.1287/mksc.14.3.g161)). A meta-analysis puts the loss-aversion ratio at λ = 1.49 (base) to 1.73 (enhanced model) across 109 estimates ([Neumann & Böckenholt 2014, J. Retailing 90(2)](https://ideas.repec.org/a/eee/jouret/v90y2014i2p182-197.html)).
- §65 says to "maintain price-change thresholds" without defining them, and ignores price endings ([v1 §65 L2185–2199](/home/user/laee-engine/docs/spec/v1-original.md)). A $9 ending raised demand in all three of Anderson & Simester's field experiments ([QME 1(1):93–110](https://econpapers.repec.org/RePEc:kap:qmktec:v:1:y:2003:i:1:p:93-110)). Consumers react to a 1-cent rise from a 99-ending price as if it were a 15–25 cent rise ([Strulov-Shlain 2023, RESTUD 90(5):2612–2645](https://academic.oup.com/restud/article-abstract/90/5/2612/6931812)) [snippet].
- §58 lists temperature, precipitation, humidity, snow, heat alerts and air quality, and says only "Create nonlinear features … Do NOT assume a linear relationship" ([v1 §58 L2008–2027](/home/user/laee-engine/docs/spec/v1-original.md)). It names no data source, forecast-vs-realised handling, climatology baseline or lag structure. Retail weather sensitivity varies with local climate and adaptation ([Roth Tran, Management Science](https://pubsonline.informs.org/doi/abs/10.1287/mnsc.2023.4799)) [snippet]. Reanalysis rainfall has documented biases over parts of Africa ([ERA5 vs CHIRPS, Ethiopia](https://www.iris.sssup.it/retrieve/d3bd82b5-a3be-4b7c-8056-2324d61e8687/s00703-024-01008-0.pdf)) [snippet].
- §59 lists `payday` as a single feature with no definition, has no lunar-calendar holidays (Ramadan, Eid), and asks for Price×Weekend/Holiday/LocalEvent interactions ([v1 §59 L2031–2059](/home/user/laee-engine/docs/spec/v1-original.md)). Retail prices tend to fall during seasonal demand peaks because retail margins fall then ([Chevalier, Kashyap & Rossi 2003, AER](https://www.aeaweb.org/articles?id=10.1257%2F000282803321455142)). So calendar-cell price variation is endogenous.
- §31 defines $\epsilon_{ij}$ with $j$ = "competitor/substitute" and says "Estimate this only when sufficient variation exists" without defining "sufficient" ([v1 §31 L1165–1188](/home/user/laee-engine/docs/spec/v1-original.md)). In a meta-analysis about 10% of 1,060 published cross-price elasticities had the wrong (negative) sign ([NBER w12756 citing Sethuraman, Srinivasan & Kim 1999](https://www.nber.org/system/files/working_papers/w12756/w12756.pdf)) [snippet]. That points to noisy estimation.
- §32 sets edge weight = Similarity × GeographicOverlap × PriceBandSimilarity × ObservedSubstitution and never defines how ObservedSubstitution is estimated ([v1 §32 L1192–1220](/home/user/laee-engine/docs/spec/v1-original.md)).
- §4's panel has `price`, `units_sold`, `revenue` but never says whether price is the posted price or a unit value, revenue ÷ units ([v1 §4 L162–230](/home/user/laee-engine/docs/spec/v1-original.md)). Measurement error in the denominator of a ratio regressor causes "division bias", because the error enters both sides ([NBER w26066, discussing Borjas 1980](https://www.nber.org/system/files/working_papers/w26066/w26066.pdf)) [snippet].
- §48 uses MAPE and SMAPE ([v1 §48 L1683–1703](/home/user/laee-engine/docs/spec/v1-original.md)). Both are undefined or unstable on zero-sales days. The M5 hierarchical retail competition (42,840 Walmart series) scored with scaled errors instead ([Makridakis et al. 2022, IJF](https://www.sciencedirect.com/science/article/pii/S0169207021001874)) [snippet; metric name recalled].
- §43's Data Quality Score includes `stockouts` and `promotionContamination` but gives no formula ([v1 §43 L1547–1573](/home/user/laee-engine/docs/spec/v1-original.md)).

### Inferences
- **[own sim] Magnitudes behind the critique** (base DGP; numbers are means over replications, truth = −1.5 unless stated):

| v1 practice | Result | Section |
|---|---|---|
| OLS on $\log(1+Q)$ at 0.3 units/day (73% zero days) | −0.23 (PPML −1.36) | Q6 |
| OLS on $\log(1+Q)$ at 1 unit/day (35% zero days) | −0.72 (PPML −1.52) | Q6 |
| Dropping sold-out days (daily cap, 41% of days censored) | −0.62 (ignoring censoring −1.00; censored Poisson −1.51) | Q1 |
| Ignoring stockouts, weekly replenishment, 13% zero-stock days | −0.38 to −0.44 (oracle censored model −1.42 to −1.53) | Q1 |
| Price = revenue/units with 5% unit-recording noise, SMB-scale price variation | −1.28 (posted price −1.59 in the same DGP; plim formula in Q3) | Q3 |
| Pooling regular and promotional price variation (true regular β = −0.8, promo β = −3.0) | −2.90 | Q2 |
| Adding a trailing 28-day rolling-demand control (§17) at 10 units/day | −1.69 ± 0.51, against −1.50 ± 0.37 without it (more biased and about 40% noisier) | below |

- **Rolling or lagged demand as a DML control (§17) is a bad control when prices are persistent.** With 1–3 permanent price changes, trailing demand is a function of earlier prices, which makes it post-treatment for the price change. In the base DGP (DML-style partialling and PPML, with no confounding to remove) it added 7–17% bias in magnitude and raised the sampling SD by 34–63% at 3/10/30 units per day [own sim]. v2 rule: keep lagged and rolling demand in forecasting models only. In the elasticity nuisance models, use only pre-change baselines, or handle persistence with time effects or a local-level term.
- **Stockout and inventory as covariates (§17) are post-treatment**: price → demand → probability of stockout. They must enter through the likelihood (censoring or exposure, see Q1), never as regressors.
- **Proposed v2 replacement map:**

| v1 | v2 rule (details in the referenced section) |
|---|---|
| §4 `price` | Store `posted_price` from the item master or menu (or the daily modal transaction price) plus `price_source`. Never compute price as revenue/units on the same rows that give Q (Q3). |
| §14/§16 $\ln Q$ | Estimate β with PPML or NB2 (or the censored-count likelihood when censoring is flagged). DML must use a multiplicative or Poisson-type model, not $\ln Q$, unless zero periods are ≤5% after aggregation (Q6). |
| §17 | Remove `inventory`, `stockout`, and lagged/rolling demand from the elasticity nuisance features (above). |
| §20 rolling 30/60/90-day β | Allowed only for windows whose Fisher information meets SE ≤ 0.5 (Q6 formula). Otherwise use a dynamic or shrinkage prior. |
| §31/§32 | Split into within-shop cross-price (Q7) and competitor price effects. Add an identification test before estimating. Edge weight becomes prior × posterior evidence (Q7). |
| §43 DQS | Concrete components: regular-price Fisher information, estimated lost-demand share, unresolved suspected censoring, share of price variance from promotions, price source, zero share (Q1, Q2, Q6). |
| §48 | Replace MAPE/SMAPE with MASE or RMSSE, Poisson deviance and mean error (bias) (Q8). |
| §58 | Weather spec (Q5). |
| §59 | Calendar spec, incl. payday presets, Hijri holidays and a within-cell identification test for interactions (Q4). |
| §61 | Add an internal (memory-based) reference price with gain/loss asymmetry and short-run vs long-run elasticity (Q3). |
| §62 | Regular-price filter, separate β_R and β_D, post-promotion dip terms, endogeneity checks (Q2). |
| §63/§64 | Detection ladder plus censored likelihood, EM or Kaplan–Meier (Q1). |
| §65 | Learned price grid, ending classes, left-digit perceived price (Q3). |

- **Suggested DQS components** (judgement; combine by geometric mean, each in [0,1]):
  - $q_\text{info}=\min(1, I_\text{reg}/16)$, where $I_\text{reg}=\sum_t\hat\mu_t\tilde x_t^2$ uses regular-price variation only. 16 corresponds to SE 0.25.
  - $q_\text{cens}=1-\min(1,\hat L/0.3)$, where $\hat L$ is the estimated lost-demand share. Multiply by 0.8 if censoring is suspected but unresolved.
  - $q_\text{promo}=1-\text{Var}(\ln P/R)/\text{Var}(\ln P)$ when β_R is the target.
  - $q_\text{src}\in\{1,0.7,0.3\}$ for posted, modal-transaction and unit-value price respectively.
  - $q_\text{zero}$: penalise when the daily zero share is above 0.8 and units are below 400.
- **New data-contract fields to request at onboarding** (cheap for SMBs):
  - transaction timestamps (not just daily totals);
  - restock or delivery dates;
  - a "sold out" button or end-of-day stock flag;
  - promotion calendar or markdown reason codes;
  - package size, to catch hidden price changes;
  - country and pay-calendar preset.

### Gaps
- v1 never says whether the input is transaction-level or daily aggregates. Several remedies (stock-out timing, modal price, within-day promotions) need transaction timestamps.
- I did not find a published Neyman-orthogonal DML score for a partially linear Poisson or log-link model, which is what a DML version of PPML needs. The DML researcher's notes should cover it.
- The rolling-demand bad-control result comes from my DGP with i.i.d. demand shocks. With persistent demand shocks that also drive price changes, lagged demand could reduce confounding, and the trade-off was not simulated.

---

## Q1. Censored demand under stockouts: how big is the bias, how to detect stockouts without inventory data, and how to correct (Tobit/censored regression, EM unconstraining, Kaplan–Meier)?

### Takeaway
Stockouts truncate the demand response exactly where price cuts push demand up, so ignoring them biases elasticities toward zero. In my simulations the attenuation ranged from 2% to 75% depending on how strongly censoring tracks price. Dropping stockout days is worse than ignoring them. When the censoring point is known, a right-censored Poisson/NB likelihood (or the equivalent EM) or an exposure offset from the time of the last sale removes the bias. Detecting censoring from sales alone (zero runs, timestamps) recovers only part of it. The highest-value fix for SMBs is to collect restock dates or a "sold out" signal.

### Cited Findings
- **Bias direction and size:**
  - Naive demand rates from observed sales are biased even for items with very few stockouts ([Anupindi, Dada & Gupta 1998](https://pubsonline.informs.org/doi/10.1287/mksc.17.4.406)).
  - Ignoring changing product availability biases demand estimates. A correction using availability observed every four hours across 54 vending machines predicted significantly larger stockout effects on profit ([Conlon & Mortimer 2013, AEJ: Micro 5(4)](https://www.aeaweb.org/articles?id=10.1257%2Fmic.5.4.1)).
  - The authors' response in the Bray et al. Unjournal exchange states that "ignoring censored demand from stockouts biases elasticity estimates towards zero, both in the experimental and in the observational setting". An evaluator had noted that the paper did not discuss stockouts ([authors' response](https://unjournal.pubpub.org/pub/authorsresponsebraybray/release/1); [evaluation 1](https://unjournal.pubpub.org/pub/eval1braybray)) [snippet].
  - In airline unconstraining simulations, once 60–80% of observations are constrained, EM and projection-detruncation estimates of the unconstrained mean exceed naive methods by 20–80% ([Weatherford & Pölt 2002, J. Revenue & Pricing Mgmt](https://link.springer.com/article/10.1057/palgrave.rpm.5170027)) [snippet].
  - On a stockout-annotated fresh-retail dataset (hourly availability), latent-demand recovery cut systematic demand underestimation from 7.37% to near zero and improved accuracy by 2.73% ([FreshRetailNet-50K, arXiv 2505.16319](https://arxiv.org/abs/2505.16319)) [snippet]. The authors note that existing benchmarks treat every zero sale as absence of demand.
  - The worldwide average retail out-of-stock rate was reported at 8.3% ([Gruen, Corsten & Bharadwaj 2002 GMA report, as cited in "Forty years of out-of-stock research"](https://www.academia.edu/26855575/Forty_years_of_Out_of_Stock_research_and_shelves_are_still_empty)) [snippet].
- **Methods:**
  - Expectation maximisation (EM) and projection detruncation (PD) were the best of six unconstraining methods compared by Weatherford & Pölt. Only EM has a rigorous statistical basis and proven convergence; PD is a heuristic ([Skurla Babić evaluation](https://www.emc-review.com/sites/default/files/2019-2/Ruzica%20Skurla%20Babic.pdf)) [snippet].
  - Terza (1985) gave the Poisson analogue of the Tobit (a censored Poisson regression), with gradient, Hessian and Newton–Raphson, illustrated on shopping behaviour ([Economics Letters 18(4):361–365](https://www.sciencedirect.com/science/article/abs/pii/0165176585900539)). Stata ships a right-censored Poisson as `rcpoisson` ([Raciborski 2011](https://journals.sagepub.com/doi/10.1177/1536867X1101100106)).
  - A Kaplan–Meier estimator can be used inside data-driven inventory control with censored demand, and is asymptotically consistent for discrete demand ([Huh, Levi, Rusmevichientong & Orlin 2011, OR 59(4):929–941](https://pubsonline.informs.org/doi/abs/10.1287/opre.1100.0906)).
  - For Poisson or normal demand with unknown mean, all the information in the timing of sales is captured by one number, the stock-out time. Using timing observations removed 76.1% of the profit loss incurred when only stock-out events were observed ([Jain, Rudi & Wang 2015, OR 63(1):134–150](https://ideas.repec.org/a/inm/oropre/v63y2015i1p134-150.html)) [snippet].
  - Vulcano, van Ryzin & Ratliff combine a multinomial-logit choice model with nonhomogeneous Poisson arrivals and treat observed sales as an incomplete observation of "primary demand". Their EM uses closed-form steps, converges to a stationary point, and needs the seller's aggregate market share for identification ([OR 60(2):313–334, 2012](https://pubsonline.informs.org/doi/10.1287/opre.1110.1012)) [snippet].
  - A Tobit (censored-regression) model is the standard way to unconstrain sales data when estimating price elasticities ([SAS Global Forum 448-2013](https://support.sas.com/resources/papers/proceedings13/448-2013.pdf)) [snippet]. "Tobit exponential smoothing" extends ETS to censored sales ([IJPR 2025](https://www.tandfonline.com/doi/full/10.1080/00207543.2025.2600521); [arXiv 2409.05412](https://arxiv.org/pdf/2409.05412)) [snippet].
- **Detection without inventory:**
  - A hidden Markov model on point-of-sale data with three latent states, one of them out-of-stock, was validated against shelf audits for 14 products × 10 stores: 63.48% detection power with 15.52% false alerts ([Montoya & Gonzalez 2019, M&SOM 21(4):932–948](https://pubsonline.informs.org/doi/10.1287/msom.2018.0732)).
  - Runs of consecutive zero sales are treated as probabilistic anomalies that strongly indicate shelf out-of-stock ([EJOR, "Fixing shelf out-of-stock with signals in point-of-sale data"](https://www.sciencedirect.com/science/article/abs/pii/S0377221717309888)) [snippet].
  - Practitioner rule: zero sales for 24–48 h on a high-velocity SKU is a high-confidence out-of-stock signal ([Field Agent](https://www.fieldagentcanada.com/blog/how-to-track-out-of-stock-issues-in-retail-before-they-hurt-sales); [Fieldpie](https://www.fieldpie.com/blog/out-of-stock-detection/)) [snippet; industry blogs].
- **Library facts:**
  - statsmodels 0.15.0 has no Tobit and no general right-censored count model. Its `_RCensored*` classes are private and censor at zero to build the hurdle zero part. `HurdleCountModel` (added in 0.14) raises `NotImplementedError` for `offset` or `exposure` ([statsmodels on PyPI](https://pypi.org/project/statsmodels/)) [pkg].
  - `GLM` accepts `offset` and `exposure` ([statsmodels](https://pypi.org/project/statsmodels/)) [pkg].
- **Textbook baseline:** Talluri & van Ryzin cover censored-normal EM unconstraining ([The Theory and Practice of Revenue Management, Springer 2004](https://doi.org/10.1007/b139000)) [recalled; not opened].

### Inferences
**Why the bias happens, and its sign.** Lower price raises demand, which makes stockouts more likely, which caps observed sales at low prices, which flattens the estimated curve. So when censoring is concentrated at low prices, the bias is toward zero. The sign flips when shortages coincide with price rises. In inflationary or FX-shock periods common in Nigeria and Kenya, a supplier shortage can trigger both a price rise and stockouts; ignoring censoring then makes demand look too elastic. v2 should therefore report $\widehat{\text{corr}}(\ln P_t, \text{censor}_t)$ alongside every estimate.

**[own sim] Magnitudes.**

Model A, daily production cap (bakery or restaurant style), μ0 = 4/day, sales = min(D, cap):

| Cap | Censored days | Naive PPML | Drop sold-out days | Censored Poisson |
|---|---|---|---|---|
| 8 | 7.8% | −1.47 | −1.22 | −1.57 |
| 6 | 26.0% | −1.17 | −0.83 | −1.48 |
| 5 | 41.4% | −1.00 | −0.62 | −1.51 |

Model B, weekly replenishment to order-up-to level S with stock carried over, μ0 = 3/day. Two runs with different seeds; means, with SD in brackets for the second run:

| S | Zero-stock days | Naive (run 1) | Naive (run 2) | Oracle censored, inventory known (run 1) | Oracle censored (run 2) | Zero-run detection, α=0.01 single pass (run 1) | Iterative detection, α=0.05 (run 2) | Restock-date-aware (run 2) |
|---|---|---|---|---|---|---|---|---|
| 25 | 1.9% | −1.17 | −1.02 (0.56) | −1.55 | −1.35 (0.73) | −1.20 | −1.13 (0.69) | −1.25 (0.68) |
| 21 | 6.4% | −0.71 | −0.72 (0.39) | −1.45 | −1.49 (0.63) | −0.93 | −0.94 (0.54) | −1.37 (0.59) |
| 18 | 13.1% | −0.44 | −0.38 (0.29) | −1.53 | −1.42 (0.79) | −0.57 | −0.87 (0.67) | −1.34 (0.75) |

  - Zero-run flag precision was 0.40–0.78 and recall 0.18–0.33.
  - The share of zero-stock days understates censoring, because the sell-out day itself is partially censored. A 2% share of zero-stock days still attenuated β by 20–30%. The relevant metric is the lost-demand share, not the share of stockout days.
  - The naive estimate has a smaller SD, so it is biased and falsely precise.

Model C, intraday Poisson arrivals over 12 opening hours with a fixed hourly profile, μ0 = 6/day, daily cap:

| Cap | Sold-out days | Naive | Exposure offset, oracle sell-out flag | Timestamp-only detection |
|---|---|---|---|---|
| 9 | 15% | −1.28 | −1.45 | −1.28 |
| 7 | 40% | −1.02 | −1.49 | −1.18 |
| 6 | 56% | −0.78 | −1.46 | −1.07 |

**M1. Right-censored Poisson or NB regression (Terza-type), the default estimator once censoring is flagged.**
- Model: $\mu_t=\exp(\alpha+\beta\ln P_t+x_t'\gamma+\ln e_t)$, with exposure $e_t$ = 1 unless M4(d) applies. Let $c_t=1$ if observed sales $s_t$ is a lower bound on demand.
- Log-likelihood: $\ell=\sum_{c_t=0}[s_t\ln\mu_t-\mu_t-\ln s_t!]+\sum_{c_t=1}\ln\Pr(D_t\ge s_t)$, with $\Pr(D\ge s)$ = `scipy.stats.poisson.sf(s-1, mu)`. Use `logsf` for numerical stability.
- NB2 version: `nbinom.logpmf(s, n=r, p=r/(r+mu))` and `nbinom.logsf(s-1, n=r, p=r/(r+mu))`.
- Defaults:
  - initialise from PPML on all data;
  - optimise with BFGS (gtol 1e-6);
  - switch to NB2 when the Pearson dispersion $\hat\phi>1.5$;
  - SE from a sandwich using numerical scores, clustered by week.
- Must be implemented in-house, since statsmodels has no general version [pkg].
- Minimum data: at least 30 uncensored days per price level. Warn if more than 50% of days at any price level are censored, and refuse to estimate (flag "capacity-bound: elasticity not identified at this price") above 80%, because μ is then pinned only by the Poisson/NB tail shape.
- Failure modes:
  - The censoring likelihood is not distribution-robust the way PPML is. Overdispersion with a Poisson censored model biases β, so test dispersion.
  - Misidentified censoring points.
  - Stockouts caused by supply rather than demand are fine (censoring stays exogenous given μ). Stock levels the owner sets in response to expected demand are also fine, because the censoring point is observed.

**M2. EM unconstraining (equivalent to M1 when the censoring point is known; useful as a modular step).**
- Poisson E-step for each censored day: $\hat s_t=E[D_t\mid D_t\ge s_t]=\mu_t\,\Pr(D\ge s_t-1)/\Pr(D\ge s_t)$ = `mu*poisson.sf(s-2,mu)/poisson.sf(s-1,mu)`.
- M-step: PPML on the completed series (fractional counts are fine). Iterate until $\max|\Delta\theta|<10^{-9}$, at most 500 iterations.
- Normal (Tobit) version for log weekly demand, with $z=(c-m)/\sigma$ and $\lambda=\phi(z)/(1-\Phi(z))$:
  - $E[Y\mid Y\ge c]=m+\sigma\lambda$;
  - $E[Y^2\mid Y\ge c]=m^2+\sigma^2+\sigma(c+m)\lambda$;
  - M-step: OLS on $E[Y]$, then $\sigma^2=\overline{E[Y^2]-2E[Y]m+m^2}$.
- [own sim] With 36% censoring, Poisson EM matched the direct censored MLE to 4 decimals in 12 iterations. With 35% censoring, Tobit EM matched the direct Tobit MLE (β and σ) in 31 iterations.
- Do not use projection detruncation (it is a heuristic).
- Use multiple imputation for DML integration: draw 10–20 completed datasets from $D\mid D\ge s_t,\hat\mu_t$ (truncated Poisson/NB), run the elasticity estimator on each, and combine with Rubin's rules.

**M3. Kaplan–Meier per price regime (nonparametric cross-check; natural for SMBs with 2–4 price levels).**
- Within regime r (stratify by weekend vs weekday, or divide by a weekday index estimated on uncensored days):
  - risk set $R_k=\#\{c_t=0,s_t\ge k\}+\#\{c_t=1,s_t>k\}$;
  - events $E_k=\#\{c_t=0,s_t=k\}$;
  - survival $\hat S(k)=\prod_{j\le k}(1-E_j/R_j)=\hat P(D>k)$;
  - restricted mean $\hat\mu_r=\sum_{k=0}^{K}\hat S(k)$.
- The convention matters: a sold-out observation means $D\ge s$, so it leaves the risk set after $k=s-1$. [own sim] With Poisson(5) demand and random stock of 3–9 units: the correct convention gave 4.97, the naive mean 4.24, and the wrong convention (censored observation still at risk at $k=s$) gave 5.58.
- Elasticity: $\hat\beta=\Delta\ln\hat\mu/\Delta\ln P$ between regimes.
- Minimum data: at least 30 days per regime, censored share below 50%, and an uncensored largest observation; otherwise the mean is only a lower bound.
- Use: flag misspecification if $|\hat\beta_{KM}-\hat\beta_{M1}|>2\,\text{SE}$.

**M4. Detecting censoring without inventory data (ladder; take the highest rung the data allow).**
- (a) Merchant signals. A "sold out" button, restock or delivery logs, or end-of-day stock counts give exact $c_t$. Restock dates alone already recover most of the bias (table above).
- (b) Restock-date-aware rule. Within each replenishment cycle whose total sales are at or above the cycle-total 80th percentile:
  - treat zero-sales days after the last positive-sales day as stockouts and drop them;
  - mark that last positive day as right-censored.
- (c) Capacity heaping (daily-production caps). Let $m=\max_t s_t$, computed per weekday group if the cap differs at weekends. If $\#\{s_t=m\}\ge 3$ and the observed frequency is at least 3× the model-implied $\sum_t\Pr(D_t\ge m\mid\hat\mu_t)$ (binomial p < 0.01), treat days with $s_t=m$ as censored at $m$, then apply M1.
- (d) Stock-out timing (transaction timestamps; builds on Jain, Rudi & Wang):
  - Estimate the intraday cumulative demand profile $\Lambda_g(\tau)$ (hourly bins, groups g = weekday/weekend) from days believed uncensored; need at least 8 per group, otherwise pool.
  - For each day with last-sale time $\tau_t$, compute $p_t=\exp(-\hat\mu_t(1-\Lambda(\tau_t)))$.
  - Flag a sell-out if $p_t<0.05$, then fit PPML with offset $\ln e_t$, $e_t=\Lambda(\tau_t)$. A known sell-out flag gives the unbiased "oracle" column of the table above. Timestamp-only flags recover only part, because late-day sell-outs are indistinguishable from a natural lull (but also cause little bias).
- (e) Zero-run test on daily data:
  - Under no stockout, $\Pr(\text{zero run } t..u)=\prod_{\tau=t}^{u}e^{-\hat\mu_\tau}$ (Poisson) or $\prod(r/(r+\hat\mu_\tau))^{r}$ (NB). Flag the run if this is below α.
  - Defaults: α = 0.01 single pass, or α = 0.05 with 3 refits that exclude flagged days.
  - On a flag, drop the run days (demand unobserved) and right-censor the last positive day before the run.
  - Minimum flaggable run length $k^*=\lceil-\ln\alpha/\mu\rceil$ (Poisson) or $\lceil\ln(1/\alpha)/\ln(1+\mu)\rceil$ (NB, r = 1):

| μ (units/day) | k* Poisson, α=0.01 | k* NB(r=1), α=0.01 | k* Poisson, α=0.05 | k* NB(r=1), α=0.05 |
|---|---|---|---|---|
| 0.25 | 19 | 21 | 12 | 14 |
| 0.5 | 10 | 12 | 6 | 8 |
| 1 | 5 | 7 | 3 | 5 |
| 2 | 3 | 5 | 2 | 3 |
| 5 | 1 | 3 | 1 | 2 |

  Zero-run detection is therefore close to useless for slow movers (below 0.5 units/day), consistent with the low recall in the simulation.
- (f) HMM (Montoya & Gonzalez): defer to v3. Its 63% detection and 15.5% false-alarm rates come from a big-box retailer, not from SMB data.

**M5. Substitution under stockouts (Vulcano, van Ryzin & Ratliff EM), reconstructed from the published model description; verify the exact algorithm in the paper.**
- Setup: in period t, customers arrive Poisson($\lambda_t$) and choose $j\in S_t$ with probability $v_j/(v_0+V(S_t))$, where $V(S)=\sum_{k\in S}v_k$. Let $m_t$ be the number of purchases.
- E-step: unobserved no-purchase arrivals $\hat a_t=m_t\,v_0/V(S_t)$ (Poisson thinning).
- M-step:
  - $N_t=m_t+\hat a_t$;
  - $v_j\leftarrow\sum_t z_{jt}\big/\sum_{t:j\in S_t}N_t/(v_0+V(S_t))$;
  - rescale $v_0$ so that $V(\text{full})/(v_0+V(\text{full}))=s$, the known market share.
- Primary demand for j: $\sum_t N_t\,v_j/(v_0+V(\text{full}))$.
- SMB caveat: the market share s is unknown. Elicit it from the merchant ("of customers who want something in this category, what share buy here?") or skip M5 and use the simpler substitution flags in Q7. M5 is mainly needed when a substitute's stockout inflates another product's sales and so biases that product's own-price β.

**Integration rules for v2:**
- Run detection before every estimation.
- Never drop sold-out days; censor them instead.
- Never use stockout flags as regressors.
- Report the lost-demand share $\hat L=\sum_t(\hat E[D_t\mid\cdot]-s_t)/\sum_t\hat E[D_t]$.
- The forecasting target is unconstrained demand.
- Rules by $\hat L$: below 2% with no price correlation, a naive PPML is acceptable but flagged; 2–50% calls for M1/M2; above 50% at a price level, report that level as not identified and recommend a price-increase test.

**Tests that catch wrong implementations:**
- T1.1 Daily-cap DGP (μ0 = 4, cap 6, 150 reps): the censored-Poisson mean β must lie in [−1.62, −1.38]. Naive must be above −1.30, and drop-sold-out above naive. A pipeline that drops sold-out days fails.
- T1.2 EM equals direct MLE within 1e-4 on the same data (Poisson and Tobit).
- T1.3 With no censored rows, the censored likelihood equals PPML exactly.
- T1.4 Kaplan–Meier convention: Poisson(5) demand, stock ~ U{3..9}, n = 5,000; the restricted mean must lie in [4.85, 5.10]. The wrong convention gives about 5.58.
- T1.5 Zero-run thresholds: at μ = 1/day and α = 0.01, a 4-day run is not flagged ($e^{-4}=0.018$) and a 5-day run is ($e^{-5}=0.0067$).
- T1.6 Exposure offset: in the intraday DGP (μ0 = 6, cap 7) with known sell-out flags, mean β must lie in [−1.62, −1.38].
- T1.7 Sign reversal: build a DGP where supply shortages force both a 20% price rise and stockouts on random weeks. The naive β must be more negative than the truth, and the censored model must recover the truth.

### Gaps
- Could not open Vulcano et al. or Talluri & van Ryzin to confirm the exact EM update equations; the M5 formulas are a reconstruction.
- No published evidence was found on stockout frequency or patterns specifically for African informal retail (dukas, kiosks) or SMB restaurants.
- Montoya & Gonzalez's HMM details (emission model, priors) were not accessible.
- The bias sizes above are mine, from stylised DGPs. No paper found reports elasticity bias versus censoring rate for SMB-scale data.

---

## Q2. Promotions: separating regular-price and promotional elasticities, post-promotion dips, stockpiling, deal depth, endogeneity, regular-price imputation

### Takeaway
Most observed price variation in retail is temporary, and temporary-cut responses mix brand switching, stockpiling and category expansion. Pooling them with permanent price changes yields "elasticities" several times too large for regular-price decisions. v2 must impute a regular price, estimate β_R (regular) and β_D (promotional depth) separately with post-promotion dip terms, and refuse to report β_R when regular-price variation is absent.

### Cited Findings
- **Decomposition and dynamics:**
  - The promotion bump decomposes into cross-brand switching, cross-period effects (demand borrowed from other weeks) and category expansion, each about one third on average across four store-level datasets ([van Heerde, Leeflang & Wittink 2004, Marketing Science 23(3):317–334](https://pubsonline.informs.org/doi/abs/10.1287/mksc.1040.0061)).
  - Post-promotion dips were estimated at 4–25% of the current sales effect, using Almon, unrestricted dynamic and exponential-decay lead/lag structures on tuna and toilet-tissue store data ([van Heerde, Leeflang & Wittink 2000, JMR 37(3):383–395](https://research.tilburguniversity.edu/en/publications/the-estimation-of-pre-and-postpromotion-dips-with-store-level-sca/)).
  - Determinants of pre- and post-promotion dips are analysed in [Macé & Neslin 2004, JMR](https://journals.sagepub.com/doi/abs/10.1509/jmkr.41.3.339.35992) and stockpiling decomposition in [Ailawadi et al. 2007, JMR](https://journals.sagepub.com/doi/abs/10.1509/jmkr.44.3.450) [titles only].
  - Static estimates overstate own-price elasticity by about 30% and understate cross-price elasticities by up to 5× ([Hendel & Nevo 2006](https://www.econometricsociety.org/publications/econometrica/2006/11/01/measuring-implications-sales-and-consumer-inventory-behavior)).
- **Experimental vs observational:** the field experiment produced 389,890 random in-store prices over 35 weeks for 409 products at 82 test stores, compared against 34 control stores. Mean experimental elasticity was −0.34 versus about −2.0 observationally ([Unjournal summary](https://unjournal.pubpub.org/pub/evalsumbraybray)) [snippet].
- **Meta-analysis:** the mean of 1,851 price elasticities from 81 studies is −2.62. Accounting for price endogeneity strongly increases elasticity magnitude; accounting for heterogeneity does not ([Bijmolt, van Heerde & Pieters 2005, JMR](https://research.rug.nl/en/publications/new-empirical-generalizations-on-the-determinants-of-price-elasti/)).
- **Regular-price filters:**
  - Reference prices, defined as the most common price within a quarter, last on average about a year, while weekly prices change about every two weeks ([Eichenbaum, Jaimovich & Rebelo 2011, AER 101(1):234–62](https://www.aeaweb.org/articles?id=10.1257%2Faer.101.1.234)).
  - Nakamura & Steinsson's sales filter flags temporary V-shaped (not necessarily symmetric) cuts. Their median monthly frequency of non-sale price changes is about half the frequency including sales (9–12% vs 19–20%) ([NBER conference version](https://users.nber.org/~confer/2006/mef06/steinsson.pdf)) [snippet].
  - One weekly-data implementation requires the price to return to the regular level within 8 weeks (variant A) or 4 weeks (variant B) ([Ray 2023, Economica](https://onlinelibrary.wiley.com/doi/full/10.1111/ecca.12492)) [snippet].
  - Kehoe & Midrigan use a running-mode regular price ([JME 2015](https://www.sciencedirect.com/science/article/pii/S0304393214001809)) [recalled; window not verified].
- **Endogeneity:**
  - Prices fall on average during a product's seasonal demand peaks, mainly through lower retail margins, consistent with loss-leader models ([Chevalier, Kashyap & Rossi 2003](https://www.aeaweb.org/articles?id=10.1257%2F000282803321455142)).
  - Follow-up: "Why do retail prices fall during seasonal demand peaks?" ([Butters et al. 2025, RAND](https://onlinelibrary.wiley.com/doi/full/10.1111/1756-2171.12490)) [title].
- **Deal depth:** consumers "discount the discounts". Purchase intentions do not change until a discount threshold is passed (lower for name brands than store brands), and there is a saturation point beyond which deeper discounts add little, giving an S-shaped response (Gupta & Cooper 1992, JCR, as summarised in [IJRM 2015 replication](https://www.sciencedirect.com/science/article/abs/pii/S0167811615001032)) [snippet].
- **Long-run effects:** see [Pauwels, Hanssens & Siddarth 2002, JMR](https://journals.sagepub.com/doi/10.1509/jmkr.39.4.421.19114) and flexible-consumption competition ([Bell, Iyer & Padmanabhan 2002](https://journals.sagepub.com/doi/10.1509/jmkr.39.3.292.19103)) [titles; content recalled: permanent effects of promotions are typically near zero].
- **Hidden price changes:** package-size changes ("shrinkflation") alter the effective unit price ([Marketing Science 45(1) 2026](https://ideas.repec.org/a/inm/ormksc/v45y2026i1p142-158.html)) [title].

### Inferences
**Regular-price imputation, REG-1 (daily data; defaults). This is spell-based, because a centred running mode flips about one day before a permanent change and falsely flags that day as a promotion; I hit that bug while building test T2.1.**
1. Daily posted price $p_t$: the item-master price, else the modal transaction unit price of the day (ties go to the highest price). Normalise to per-unit (per kg, per ml) when package size changes.
2. Split the series into spells, i.e. maximal runs of an identical $p_t$.
3. A spell is **long** if it lasts at least $L_\text{min}=7$ days.
4. A long spell is a **V-sale** if both hold:
   - its price is at or below $(1-\delta)R_\text{prev}$, where δ = max(3%, one grid step) and $R_\text{prev}$ is the current regular price;
   - the price returns to at least $(1-\delta/2)R_\text{prev}$ within $L_\text{max}=28$ days of the spell's start.
5. The regular price $R_t$ is the price of the most recent long, non-V spell starting at or before t. Before the first such spell, use that spell's price.
6. Temporary price reduction: $\text{TPR}_t=1\{p_t\le(1-\delta)R_t\}$.
7. Real time, with no look-ahead (v1 §75): a new lower price counts as a TPR until it has persisted $L_\text{max}$ days, then it is reclassified as a regular change. Retrospective estimation uses the full series.
8. Depth $d_t=1-p_t/R_t$. Separately code clearance markdowns (end-of-day or near-expiry), BOGO and multibuy, and bundles. Exclude clearance from β_D because it coincides with low demand, which biases β_D toward zero.

**Demand model (PPML, NB or censored likelihood from Q1):**

$$\ln\mu_t=\alpha+\beta_R\ln R_t+\beta_D\ln(p_t/R_t)+\sum_{b}\gamma_b\,\text{POST}^{(b)}_t+\gamma_\text{pre}\text{PRE}_t+\text{calendar}+\text{weather}+\dots$$

- $\text{POST}^{(b)}_t=\ln(1/(1-d^{\text{last}}))\cdot 1\{\text{days since the last episode ended}\in b\}$, with bins b ∈ {1–3, 4–7, 8–14} days for storable goods and no post terms for perishables or restaurant meals.
- Include PRE only if promotions are announced in advance.
- Deal-depth nonlinearity (thresholds and saturation): use a hinge $\beta_{D1}\min(d,0.10)+\beta_{D2}\max(d-0.10,0)$ only with at least 2 distinct depths and at least 6 episodes. Treat cuts below 3% as noise (no promo).
- Use β_R for permanent price recommendations and β_D for promotion planning. The net promotion effect is the bump minus the summed dips.

**[own sim] Promotion DGP:** weekly, 104 weeks, 50 units/week at R = 100, one permanent rise to 108 at week 52, 15% promo weeks at 10/20/30% depth, β_R = −0.8, β_D = −3.0, next-week dip of 35% of the bump.
- Pooled single-log-price PPML: −2.90.
- Separate model: β_R = −0.80, β_D = −3.12.
- Pooled plus a lagged-depth term: −2.89, so lag terms do not rescue pooling.
- Consequence for pricing: under constant elasticity the optimal markup is $P^*=c\,\beta/(1+\beta)$. With the pooled −2.9 the engine would set $P^*\approx1.53c$. With the true regular −0.8 demand is inelastic, there is no interior optimum, and constraints decide. The two recommendations point in opposite directions.

**Endogeneity controls:**
- Promotion timing correlates with calendar peaks (loss leaders) or troughs (happy hours, clearance). Include those calendar drivers in both nuisance models.
- Require that at least 50% of the raw variance of $\ln(p/R)$ remains after partialling out calendar effects.
- Event-study check: mean residual demand over days −7..−1 before unannounced promotions should be about 0 (|t| < 2); otherwise flag "promotion timed on demand shocks".

**Minimum data and fallbacks:**
- β_R needs at least 2 regular price levels, each held ≥ 28 days, and Fisher information meeting SE ≤ 0.5 (Q6 formula).
- β_D needs at least 3 promotion episodes.
- With promotions only and no regular changes, report β_R as "not identified" (prior only) and never back it out of β_D.

**Tests:**
- T2.1 REG-1 on `[100]*10+[80]*2+[100]*10+[110]*30+[90]*1+[110]*5`: regular = 100 for indices 0–21 and 110 from index 22; TPR flags only at indices 10–11 and 52; the move to 110 counts as permanent. A centred ±14-day running mode fails this test because it flags index 21.
- T2.1b REG-1 V-shape: `[100]*20+[85]*14+[100]*20` gives regular 100 throughout and TPR on indices 20–33 (a 14-day promotion that returns). `[100]*20+[85]*40` gives a regular change to 85 at index 20 retrospectively; in real-time mode the first 28 days are TPR and then reclassified.
- T2.2 On the promotion DGP, separate β_R must lie in [−1.0, −0.6] and the pooled estimate must fail that check.
- T2.3 Post-dip recovery: with a 35% dip injected in the next week, the estimated first-bin POST effect must be negative and its 95% CI must cover the injected effect.
- T2.4 A package-size change from 500 g to 450 g at a constant shelf price must register as an 11.1% per-unit price rise.

### Gaps
- No SMB- or Africa-specific promotion-response estimates were found.
- The Bray et al. full text, which may decompose why observational elasticities are larger, could not be opened.
- Exact Kehoe–Midrigan and Nakamura–Steinsson window parameters were not verified, so the 14/28-day defaults above are judgement for daily data.
- No source found for a defensible numeric prior on the ratio β_D/β_R.

---

## Q3. Reference prices and behavioural responses: internal and external reference prices, loss aversion, 9-endings and left-digit bias, rounding in cash economies

### Takeaway
Reference effects make the short-run response to a price change differ from the long-run elasticity, with price increases hitting harder than decreases. With 1–3 price changes an SMB cannot estimate these effects, so v2 should model them with a prior (sensitivity band plus transition cost), not estimate them. Price endings and cash rounding matter mostly at the recommendation stage: snap recommended prices to the shop's learned price grid and to left-digit thresholds. Never derive price as revenue ÷ units, because noise in recorded units drags the elasticity toward −1.

### Cited Findings
- **Reference prices:**
  - Gain-loss asymmetry (losses hurt more than equal-size gains please) is an empirical generalisation of reference-price research ([Kalyanaram & Winer 1995](https://doi.org/10.1287/mksc.14.3.g161)).
  - The integrative review covering formation, retrieval and effects of reference prices is [Mazumdar, Raj & Sinha 2005, J. Marketing 69:84–102](https://www.semanticscholar.org/paper/Reference-Price-Research:-Review-and-Propositions-Mazumdar-Raj/57324b259fdd9e60b068684ce56f801422987dc2).
  - Briesch et al. compared reference-price model forms on scanner panels (peanut butter, detergent, coffee, tissue) with latent-class heterogeneity ([JCR 24(2):202–214, 1997](https://www.kellogg.northwestern.edu/faculty/research/researchdetail?guid=a4b1e334-9bf4-45e5-beb3-d1545270b35d)). [recalled: a brand-specific, memory-based (past-price) reference fitted best.]
  - Loss aversion in product choice: λ = 1.49 (base) to 1.73 (enhanced model) from 109 estimates in 33 studies. Analysis method explains about 57% of the variation in λ ([Neumann & Böckenholt 2014](https://ideas.repec.org/a/eee/jouret/v90y2014i2p182-197.html)).
- **Price endings:**
  - $9 endings raised demand in all three field experiments, more strongly for new items, which suggests they work best when customers have little information ([Anderson & Simester 2003](https://econpapers.repec.org/RePEc:kap:qmktec:v:1:y:2003:i:1:p:93-110)).
  - In scanner data on 3,500 products from 25 US chains, a 1-cent rise from a 99-ending price is perceived like a 15–25 cent rise. The paper also quantifies firms' pricing response to left-digit bias ([Strulov-Shlain 2023](https://academic.oup.com/restud/article-abstract/90/5/2612/6931812); [replication package](https://zenodo.org/records/7075310)) [snippet].
  - Counterpoints exist: "If You Think 9-Ending Prices Are Low, Think Again" ([Bar-Ilan WP 2019-06](https://ideas.repec.org/p/biu/wpaper/2019-06.html)) and "Zero-Ending Prices, Cognitive Convenience, and Price Rigidity" ([EconStor](https://ideas.repec.org/a/zbw/espost/265251.html)) [titles only].
- **Cash and convenient prices:**
  - Firms set "convenient" prices that speed up transactions for items that are usually paid in cash, bought alone or with few other items, and sold in high-traffic settings ([Knotek 2011, REStat 93(3):1076–1086](https://ideas.repec.org/a/tpr/restat/v93y2011i3p1076-1086.html)).
  - Price points and menu costs both contribute to rigidity ([JME 2024](https://ideas.repec.org/a/eee/moneco/v145y2024ics0304393224000126.html)) [title].
- **Nigeria:**
  - Kobo coins of ½–25k were withdrawn effective 28 Feb 2007; coins still in circulation are 50 kobo, ₦1 and ₦2 ([Wikipedia: Nigerian naira](https://en.wikipedia.org/wiki/Nigerian_naira)) [snippet].
  - Those coins are "not much in use because of their very low value" ([Leftover Currency](https://www.leftovercurrency.com/exchange/nigerian-naira/nigerian-naira-coins/)) [snippet; dealer site].
- **Division bias:** when the measured outcome is also the denominator of the measured price, measurement error enters both sides and biases the slope even if the error is classical ([NBER w26066 on Borjas 1980, J. Human Resources 15(3):409–423](https://www.nber.org/system/files/working_papers/w26066/w26066.pdf)) [snippet].

### Inferences
**Internal reference price (memory-based):**
- $r_t=\theta r_{t-1}+(1-\theta)p_{t-1}$, with $r_0=p_0$ and $\theta=0.5^{1/h}$ for a half-life of h days. Grid h ∈ {7, 14, 30, 60}; choose by profile likelihood.
- Gain and loss terms: $g_t=\max(\ln r_t-\ln p_t,0)$ and $l_t=\max(\ln p_t-\ln r_t,0)$.
- Model: $\ln\mu_t=\dots+\beta\ln p_t+\eta_g g_t-\eta_l l_t$, with loss aversion $\eta_l>\eta_g\ge0$.
- Implications:
  - Impact elasticity of a permanent increase is $\beta-\eta_l$; of a decrease, $\beta-\eta_g$. Long-run elasticity is β.
  - Transition cost of a permanent increase of Δ log points is roughly $\eta_l\Delta\cdot h/\ln 2$ "full-demand days" (geometric decay).
  - Do not read the first 2–4 weeks after a change as the long-run β. Compare weeks 1–2 with weeks 5–8.
- Identification needs at least 2 changes in each direction, each followed by at least 2h days. SMBs rarely have this, so the default is not to estimate.
  - Report a sensitivity band using the loss-aversion ratio prior λ = η_l/η_g ≈ 1.5 (Neumann & Böckenholt), with a log-normal prior N(ln 1.5, 0.5²).
  - Cap per-step increases (v1 §66). The asymmetry justifies staged increases.
- External reference (v1 §61) needs competitor price time series. Cross-sectional competitor gaps are absorbed by the business fixed effect and cannot identify a within-business effect; keep them for heterogeneity only.

**Price endings and left-digit (perceived-price model, Lacetera–Pope–Sydnor form [recalled]):**
- $\tilde p=\lfloor p\rfloor_k+(1-\theta)(p-\lfloor p\rfloor_k)$, where $\lfloor p\rfloor_k$ truncates p to its leading-digit unit $10^{k}$.
- Strulov-Shlain's "15–25 cents" implies $0.01+0.99\theta\in[0.15,0.25]$, so θ ≈ 0.14–0.24 (US grocery, cents). That prior is unvalidated for integer-currency markets.
- Example (KES): 99 → 100 is a perceived jump of $1+9\theta\approx2.8$ KES at θ = 0.2.
- Features:
  - ending class ∈ {9/99, 5, 0, other};
  - left-digit crossing $1\{\text{leading digit or digit count changes}\}$.
- Rarely identified per item, so use θ in the optimiser: prefer the candidate just below a left-digit threshold when profits are within the indifference band.

**Cash rounding and price grid (learned per shop):**
- For g ∈ {1, 2, 5, 10, 20, 50, 100, 500, 1000} (currency units), compute the share of historical posted prices divisible by g. Choose the largest g with share ≥ 0.8.
- Snap recommendations to that grid. v1 §65's "price-change threshold" becomes one grid step.
- Note the minimum relative step g/p (a KES 50 item on a KES 10 grid moves in 20% steps). Large discrete steps help identification but make optimisation coarse, so evaluate only grid points.
- Never use negotiated or haggled transaction prices (informal retail) as identifying variation, because they are customer-specific and endogenous. Use the posted or list price.

**Division bias.**
- With $\ln\hat P=\ln P-u$ and $\ln Q^{rec}=\ln Q+u$: $\text{plim}\,\hat\beta=(\beta\sigma_x^2-\sigma_u^2)/(\sigma_x^2+\sigma_u^2)$, a weighted average of β and −1.
- [own sim] At 20 units/day, recorded units = true units × $e^{u}$, revenue correct, price derived as revenue/units:

| sd(u) | 0 | 0.05 | 0.10 | 0.20 |
|---|---|---|---|---|
| Mean $\hat\beta$ (truth −1.5) | −1.59 | −1.28 | −1.12 | −1.03 |

- The plim formula predicts −1.29 at sd(u) = 0.05. Because SMB price variation is small (σ_x ≈ 0.06), even 5% unit-recording noise destroys the estimate.

**Tests:**
- T3.1 Perceived-price function at θ = 0.2: $\tilde p(4.99)=4.792$, $\tilde p(5.00)=5.00$; $\tilde p(99)=97.2$, $\tilde p(100)=100$.
- T3.2 Grid learner: on prices {50, 60, 70, 80, 95, 100, 120, 150}, g = 10 has 7/8 = 0.875 ≥ 0.8 → g = 10; g = 50 has 3/8 → rejected.
- T3.3 Division-bias guard: given only revenue and units, the pipeline must refuse to compute the elasticity from the derived price unless an independent posted price exists, or must instrument it with the lagged modal price. In the simulation the estimate must return within ±0.2 of the truth.
- T3.4 Reference-price recursion: h = 7 gives θ = 0.9057. After a step from 100 to 110, $r_t$ reaches 105 after about 7 days.

### Gaps
- No verified sources on Kenyan shilling price-ending conventions, M-Pesa amount granularity (believed whole shillings, unverified), or Nigerian retail rounding to ₦50/₦100.
- No African or emerging-market estimate of left-digit or 9-ending effects was found.
- No evidence was found on SMB-scale reference-price effect sizes, so the η values remain unpinned.
- The Lacetera–Pope–Sydnor functional form is recalled, not re-verified.

---

## Q4. Calendar effects: weekly cycles, payday and end-of-month, holidays (Ramadan/Eid, Christmas, school terms), encoding, and price × calendar interactions

### Takeaway
Calendar effects are large and partly drive the shop's own pricing (peaks get price cuts), so they must enter both nuisance models. Payday is country-specific: the 25th is common in South Africa, month-end in Nigeria, plus early-month grant days. Islamic holidays move about 11 days per Gregorian year and need a Hijri calendar with ±1 day local uncertainty. With 1–2 years of SMB data, most v1 calendar dummies and every Price×calendar interaction are unidentified and must be gated by an explicit within-cell variation test.

### Cited Findings
- **Pay-cycle consumption:**
  - Social Security recipients' spending (both amount and probability) rises immediately after check receipt ([Stephens 2003, AER 93(1):406–422](https://www.aeaweb.org/articles?id=10.1257%2F000282803321455386)).
  - Paycheck timing study (UK): [Stephens, NBER w9356](https://ideas.repec.org/p/nbr/nberwo/9356.html) [title].
  - Food-stamp recipients' caloric intake falls 10–15% over the 30 days after receipt; the implied daily discount factor is 0.996 ([Shapiro 2005, J. Public Econ. 89:303–325](https://econwpa.ub.uni-muenchen.de/econ-wp/mic/papers/0304/0304005.pdf)).
  - Emerging-market evidence: daily normalised credit-card spending rises 4.5 percentage points on the first day after payday ([China Economic Review 2024](https://www.sciencedirect.com/science/article/abs/pii/S1043951X24001950)).
- **Africa (mostly industry or press sources):**
  - The 25th is the most common South African payday ([Arcadia Finance](https://www.arcadiafinance.co.za/personal-finance/income/salary-dates-in-south-africa/)) [snippet].
  - SASSA social grants are normally paid at the beginning of each month and move point-of-sale and ATM volumes ([Moneyweb](https://www.moneyweb.co.za/news/economy/social-grants-cause-consumer-spending-uptick/)) [snippet].
  - 76% of South Africans regularly run out of money before payday, 57% by mid-month ([News24/Fin24](https://www.news24.com/Fin24/most-south-africans-out-of-money-before-payday-study-20210414)) [snippet]. The pre-payday squeeze is called "Mampara week" ([Old Mutual](https://www.oldmutual.co.za/news/the-payday-cycle-keeping-south-african-pockets-under-pressure/)) [snippet].
  - Nigeria: salaried workers are mostly paid between the 24th and month-end, and traders' incomes rise in "salary week" ([Cerebre Plus](https://www.cerebreplus.com/blog/salary-week-marketing-nigeria)) [snippet; marketing blog, low reliability].
  - Payroll norms reported as: Namibia mostly the 25th, Angola the 25th/26th, Zimbabwe month-end ([EOR HQ payroll calendar](https://eorhq.com/guides/payroll-calendar-global/)) [snippet; attribution uncertain].
  - Large-scale Kenyan cash-transfer general-equilibrium work exists ([NBER w26600](https://www.nber.org/system/files/working_papers/w26600/w26600.pdf)) but is not about intra-month timing.
- **Ramadan and Eid:**
  - A panel study reports purchase increases among Muslims of 40.6% during Ramadan and 76.3% for Eid ([Int. Rev. Economics 2024](https://link.springer.com/article/10.1007/s12232-024-00445-y)) [snippet; attribution of the figures to this paper not confirmed].
  - Early Ramadan brings grocery stock-ups; late Ramadan shifts toward clothing ahead of Eid ([Zawya](https://www.zawya.com/en/press-release/research-and-studies/ramadan-and-eid-al-fitr-impact-in-retail-sales-and-consumer-spending-b66oqls1)) [snippet; industry].
  - Spending shifts to late-night hours ([Visa](https://usa.visa.com/partner-with-us/visa-consulting-analytics/economic-insights/bustling-ramadan-night-time-economy.html)) [snippet; industry].
- **Tooling** ([holidays on PyPI](https://pypi.org/project/holidays/); [hijridate](https://pypi.org/project/hijridate/)) [pkg]:
  - `holidays` 0.106 has country modules for Kenya, Nigeria, South Africa, Ghana, Egypt, Tanzania, Uganda, Morocco, Ethiopia, Senegal and Rwanda.
  - Islamic holidays carry an "(estimated)" label when not confirmed. Kenya's Eid dates are confirmed for 2014–2024; Nigeria's Eid al-Fitr for 1979–2026.
  - The Nigeria module notes that "the dates of the Islamic calendar usually fall a day later than the corresponding dates in the Umm al-Qura calendar".
  - An internal `ramadan_beginning_dates(year)` exists.
  - `hijridate` converts using the Umm al-Qura calendar.
  - The South Africa module has no Islamic holidays (they are not public holidays there).
- **Price/calendar endogeneity:** retail prices fall during a product's seasonal demand peaks ([Chevalier, Kashyap & Rossi 2003](https://www.aeaweb.org/articles?id=10.1257%2F000282803321455142)).

### Inferences
**Encoding (defaults by history length H):**
- **Day of week:** always 6 dummies; partial pooling across products if a product has fewer than 8 weeks.
- **Pay cycle, option A (preset):**
  - Compute `days_since_payday` from a country preset: ZA private payday the 25th plus SASSA early-month grant days; NG the 24th–month-end window; others by merchant choice.
  - Bins {0–2, 3–7, 8–14, 15–21, 22+}.
  - Weekend paydays roll to the previous business day [recalled practice; make configurable].
- **Pay cycle, option B (learned; default when the country preset is unknown, e.g. Kenya):**
  - Day-of-month phase $\varphi=(\text{dom}-1)/\text{days\_in\_month}$, terms $\sin(2\pi k\varphi),\cos(2\pi k\varphi)$ for k = 1, 2, plus a month-turn dummy (last 3 and first 3 days).
  - Ridge or random-effect shrinkage. Only when H ≥ 6 months.
- **Annual seasonality:** none if H < 13 months (use a local level or trend instead); shrunk month dummies for 13–24 months; Fourier with K = 2–3 annual harmonics for H ≥ 24 months (dynamic-harmonic-regression style) [recalled].
- **Never** one-hot `week_of_year` (52 parameters with ≤ 2 observations each).
- **Holiday windows (retail defaults; restaurants use ±1 day):**
  - public holidays from `holidays`, window −3..+1;
  - Christmas ramp Dec 15–24, then Dec 25–26 and Dec 27–31 as separate terms;
  - Easter Good Friday through Easter Monday;
  - Ramadan indicator plus a day-of-Ramadan linear term (1–30) plus a last-10-days dummy;
  - Eid al-Fitr −3..+2;
  - Eid al-Adha (Tabaski in West Africa) −3..+2;
  - school terms and local festivals from merchant or country config (see Gaps).
- **Lunar uncertainty:** for future dates, use estimated dates and spread the effect over ±1 day with a triangular kernel (weights 0.25/0.5/0.25). Store the calendar version used at prediction time (v1 §75 `available_at`).
- **Price × calendar interactions:** allowed only if, within the cell, there are at least 2 distinct price levels with at least 30 units each and the cell's Fisher-information SE ≤ 0.5. Otherwise omit; never estimate many interactions. The preferred single-parameter alternative is $\beta_t=\beta_0+\beta_1\cdot\widehat{\text{baseline-demand index}}_t$.
- **Identification check:** after partialling calendar effects out of $\ln P$, at least 50% of its raw variance must remain, otherwise flag "price moves with calendar; weak identification".

**Tests:**
- T4.1 `holidays.country_holidays('NG', years=2026)` contains Eid al-Fitr (confirmed through 2026), and the Kenyan 2026 Eid is labelled "(estimated)".
- T4.2 The payday preset for ZA puts 2026-10-25 (a Sunday) on Friday 2026-10-23 when rollback is enabled.
- T4.3 Interaction gate: a synthetic product with one price change in July must not receive a Price×December term.
- T4.4 Simulate a 15% payday-window uplift and confirm recovery within ±5 percentage points with 12 months of 5 units/day.

### Gaps
- No peer-reviewed intra-month consumption-cycle study for Kenya, Nigeria or South African retail was found; African evidence above is from industry or press.
- No verified Kenyan payday norm.
- School-term calendars (Kenya 3 terms, Nigeria 3 terms, SA 4 terms) were not researched and need per-country, per-year configuration.
- Ramadan magnitudes come from a snippet whose attribution is uncertain; no restaurant-specific Ramadan or Eid study was found.

---

## Q5. Weather: nonlinear effects on retail and restaurant demand, African-coverage data sources, and leakage-safe use

### Takeaway
Weather mainly helps as a nuisance control that sharpens elasticity precision. Its effects are nonlinear, seasonal and climate-relative, and only partly shifted across days. For Africa, use ERA5 or ERA5-Land through Open-Meteo for temperature, and CHIRPS for historical rainfall where it validates better. Train any forecasting or price-simulation model on archived forecasts at the same lead time it will use in production, never on realised reanalysis.

### Cited Findings
- **Effects:**
  - Unusual weather explains a moderate share of short-term variation in US retail sales; retail is more sensitive to temperature than sunshine, with rainfall less relevant ([Starr-McCluer 2000](https://www.researchgate.net/publication/228288612_The_Effects_of_Weather_on_Retail_Sales)) [snippet].
  - Temperature, precipitation and sunshine all affect daily brick-and-mortar sales, and most squared terms are significant, so effects are nonlinear ([J. Retailing & Consumer Services study](https://www.researchgate.net/publication/335173325_The_impact_of_daily_weather_on_retail_sales_An_empirical_study_in_brick-and-mortar_stores)) [snippet].
  - French fashion retail (Bertrand et al. 2015) shows seasonally different, nonlinear weather effects (same source) [snippet].
- **Persistence and adaptation (Roth Tran):** weather induces little intertemporal substitution; only a small share of immediate responses is offset over time. Sensitivity falls with historical exposure for precipitation, snow and cold, but not for extreme heat ([Management Science](https://pubsonline.informs.org/doi/abs/10.1287/mnsc.2023.4799); [FRBSF WP 2022-02](https://www.frbsf.org/wp-content/uploads/wp2022-02.pdf)) [snippet].
- **Climate-relative measurement:** weather impacts should be measured against local climate ([MDPI 2023](https://www.mdpi.com/2673-7116/3/3/30)) [title].
- **Restaurants and convenience retail:**
  - Temperature, rain and wind affect restaurant sales and menu-item mix ([restaurant study](https://www.researchgate.net/publication/305926091_The_effect_of_weather_factors_on_restaurant_sales)) [snippet].
  - One study reports +5.2% sales on relatively good-weather days and −2.6% on worse days ([Applied Spatial Analysis 2021](https://link.springer.com/article/10.1007/s12061-021-09397-0)) [snippet; attribution uncertain].
  - Further studies: Chinese convenience-store chain ([JRCS 2021](https://www.sciencedirect.com/science/article/abs/pii/S0969698921001491)) and online food delivery in three Chinese cities ([Theor. Appl. Climatol. 2023](https://link.springer.com/article/10.1007/s00704-023-04542-w)) [titles].
- **Data sources:**
  - Open-Meteo's Historical Weather API serves ERA5 (0.25°, from 1940), ERA5-Land (0.1°, from 1950) and ECMWF IFS (9 km, from 2017), updated daily with about a 5-day delay ([Open-Meteo docs](https://open-meteo.com/en/docs/historical-weather-api)) [snippet].
  - Open-Meteo's Previous Runs API serves archived forecasts at fixed lead offsets of 1–7 days (e.g. `temperature_2m_previous_day1` = the value forecast 24 h before). Most models are archived from January 2024, GFS 2 m temperature from March 2021. It is intended for ML without look-ahead bias ([Previous Runs API](https://open-meteo.com/en/docs/previous-runs-api); [Open-Meteo blog](https://openmeteo.substack.com/p/weather-forecasts-from-previous-model-runs)) [snippet].
  - A Historical Forecast API also archives high-resolution forecasts ([docs](https://open-meteo.com/en/docs/historical-forecast-api)) [title].
  - CHIRPS: quasi-global 50°S–50°N, 0.05°, daily, pentadal and monthly from 1981, blending satellite and station data. Preliminary product latency is about 2 days, final about 3 weeks ([Funk et al. 2015, Scientific Data 2:150066](https://www.nature.com/articles/sdata201566)). CHIRPS v3 superseded v2 in June 2025 and covers 60°N–60°S ([UCSB CHC](https://www.chc.ucsb.edu/data)) [snippet].
- **African validation:**
  - Ethiopia: CHIRPS generally outperforms ERA5, especially at altitude. CHIRPS daily bias ratios are 0.75–1.2; both underestimate rain in north-eastern and eastern lowlands ([MAP 2024](https://www.iris.sssup.it/retrieve/d3bd82b5-a3be-4b7c-8056-2324d61e8687/s00703-024-01008-0.pdf)) [snippet].
  - Ghana: ERA5 had the lowest correlations with gauges and overestimated rain. In an extreme convective event in eastern South Africa, ERA5 matched stations while CHIRPS and CMORPH underestimated ([arXiv 2501.14829](https://arxiv.org/html/2501.14829v3)) [snippet].
  - ERA5 reduced ERA-Interim's tropical-African wet bias ([verification study](https://www.researchgate.net/publication/363304224_Verification_of_ERA5_and_ERA-Interim_precipitation_over_Africa_at_intra-annual_and_interannual_timescales)) [snippet].
  - ERA5 underestimates where stations are sparse, e.g. the Congo Basin ([global ERA5 evaluation](https://www.sciencedirect.com/science/article/pii/S1470160X24009385)) [snippet].
  - Continental multi-product assessment: [J. Hydrol. Regional Studies 2023](https://www.sciencedirect.com/science/article/pii/S221458182300201X); East Africa: [Atmosphere 2020](https://doi.org/10.3390/atmos11090996) [titles].

### Inferences
**Features (local-day aggregation in the shop's timezone; East Africa is UTC+3, so evening storms shift days if UTC is used):**
- $T^{max}$ in °C;
- apparent-temperature maximum (heat stress in humid tropics);
- daily precipitation in mm;
- precipitation during opening hours (from hourly data; for restaurants, the lunch 11–14 h and dinner 18–21 h sums);
- rain day $1\{\ge1\text{ mm}\}$;
- heavy rain $1\{\ge\text{local wet-day 90th percentile}\}$, falling back to 20 mm;
- temperature anomaly $z=(T^{max}-\text{clim}_{doy})/\text{sd}_{doy}$ against a 1991–2020 ERA5 day-of-year climatology (±15-day window);
- lag-1 rain;
- 7-day rain total (market access and farm incomes);
- rainy-season indicator from the CHIRPS climatology.

**Functional form:**
- Natural cubic spline on z (knots −1, 0, 1), or on $T^{max}$ with knots at the local 25/50/75 percentiles.
- Rain: rain day + $\log(1+\text{mm})$ + heavy-rain dummy.
- No weather × price terms unless the business actually varies prices with weather.
- Weather does enter the stockout model in Q1 (hot day → beverage sell-out → censoring).

**Role and leakage:**
- Retrospective elasticity estimation may use realised weather (ERA5, CHIRPS final): contemporaneous weather is exogenous to the shop's price and only absorbs variance.
- For real-time forecasts and price simulation at decision time T for date T+h, use the lead-h forecast (Previous Runs `_previous_day{h}`), and train the forecasting model on archived lead-h forecasts. Training on realised weather and serving forecasts creates train/serve skew.
- Availability rules (`available_at`): ERA5 at valid time + about 5 days; CHIRPS preliminary + about 2 days, final + about 3 weeks. Neither may be a same-day feature.

**Source choice and gating:**
- Temperature: ERA5-Land through Open-Meteo.
- Rainfall history: CHIRPS (0.05°), with ERA5 as fallback. A pluggable per-region choice beats a global default, given the conflicting validations.
- Include weather only if:
  - H ≥ 180 days, with at least 20 rain days and a $T^{max}$ range of at least 8 °C;
  - it improves rolling-origin Poisson deviance by at least 1%;
  - spline coefficients are sign-stable across folds.

**Failure modes:**
- Grid-cell mismatch for coastal or urban microclimates (0.25° ≈ 28 km).
- UTC day boundaries.
- Many correlated weather variables overfit.
- Weather proxies for season if annual seasonality is not modelled first.
- Reanalysis convective-rain errors.

**Tests:**
- T5.1 Leakage: for every training row of the forecasting model with decision time T and target T+h, assert `weather.available_at ≤ T` and lead = h.
- T5.2 Timezone: an hourly rain spike at 22:00 local (19:00 UTC) in Nairobi must land in the local day's total.
- T5.3 Gate: with 120 days of data the weather block must be omitted.
- T5.4 Spline recovery: simulate an inverted-U temperature effect (peak at local p75) on top of the base DGP; the fitted spline's argmax must fall within one knot interval, and β must be unchanged within ±0.1 with or without weather.

### Gaps
- No peer-reviewed weather-demand study for African SMB retail, kiosks or restaurants was found.
- Open-Meteo commercial licensing terms and the Historical Forecast API start date were not verified (fetch blocked).
- IMERG and TAMSAT were not evaluated.
- The "10%" intertemporal figure in Roth Tran's summary was ambiguous in the snippet and is omitted.

---

## Q6. Intermittent and sparse demand: Croston, SBA, TSB, count models, aggregation level, and elasticity estimation with many zeros

### Takeaway
For elasticity, estimate β with PPML (or NB2 or hurdle) on counts, never with log-OLS or log(1+Q). For forecasting slow movers, use SBA or TSB, or temporal aggregation (ADIDA/IMAPA). The binding constraint for SMBs is information, not method: $\text{SE}(\hat\beta)\approx\sqrt{\phi}/\sqrt{\sum_t\mu_t\tilde x_t^2}$. A product needs about 1,600 units across a 10% price change, or about 400 across a 20% change, to reach SE ≈ 0.5. Most SMB products therefore need hierarchical priors and experiments.

### Cited Findings
- **Methods:**
  - Croston's method (1972) forecasts nonzero demand sizes and inter-demand intervals separately ([JORS 23(3):289–303, as cited in statsforecast](https://link.springer.com/article/10.1057/jors.1972.50)) [pkg].
  - Croston's method is biased. The Syntetos–Boylan approximation (SBA) corrects this, is approximately unbiased, and is the Croston variant with substantial empirical support (3,000 automotive series; compared against SMA(13), SES and Croston) ([Syntetos & Boylan 2005, IJF 21(2):303–314](https://eprints.lancs.ac.uk/id/eprint/73097/)) [snippet].
  - TSB updates a demand probability every period instead of the interval, is unbiased at every point in time, and suits items at risk of obsolescence. Croston does not update during zero runs ([Teunter, Syntetos & Babai 2011, EJOR 214(3):606–615](https://ideas.repec.org/a/eee/ejores/v214y2011i3p606-615.html)) [snippet].
  - Categorisation cut-offs ADI = 1.32 and CV² = 0.49 split series into smooth, intermittent, erratic and lumpy (Syntetos, Boylan & Croston 2005), as described in [arXiv 2103.13812](https://arxiv.org/pdf/2103.13812) [snippet]. The boundary is discussed in [Kostenko & Hyndman 2006](https://robjhyndman.com/papers/idcat.pdf) [title; boundary formula recalled].
  - The need for classification is questioned in [OpenForecast 2024](https://openforecast.org/2024/07/16/intermittent-demand-classifications-is-that-what-you-need/) [title].
  - ADIDA (temporal aggregation, 5,000 RAF SKUs) forecasts more accurately at higher aggregation levels ([Nikolopoulos et al. 2011, JORS 62(3):544–554](https://eprints.lancs.ac.uk/id/eprint/58330/)) [snippet].
- **statsforecast 2.1.1 defaults** ([statsforecast on PyPI](https://pypi.org/project/statsforecast/)) [pkg]:
  - `CrostonClassic`: α = 0.1 for both sizes and intervals.
  - `CrostonOptimized`: α chosen in [0.1, 0.3] by golden-section search.
  - `CrostonSBA`: 0.95 × Croston with α = 0.1, i.e. 1 − α/2.
  - `TSB`: requires `alpha_d` and `alpha_p` (no defaults).
  - `ADIDA`: aggregation level = round(mean inter-demand interval), SES on the aggregated sums, equal split back.
  - `IMAPA`: averages SES forecasts over aggregation levels 1..round(mean interval).
  - Intervals are computed as `diff(nonzero_idx+1, prepend=0)`, so the first interval counts from the series start.
- **Zeros and logs:**
  - PPML handles zeros and is consistent under heteroskedasticity. In the gravity application, log-linear OLS gave an elasticity about twice the PPML value ([Santos Silva & Tenreyro 2006](https://personal.lse.ac.uk/tenreyro/jensen08k.pdf)).
  - log(1+Y) effects are unit-dependent and arbitrary when the extensive margin moves ([Chen & Roth 2024](https://bibbase.org/network/publication/chen-roth-logswithzerossomeproblemsandsolutions-2024)) [snippet].
  - Hurdle and two-part count models were introduced by [Mullahy 1986, J. Econometrics](https://www.sciencedirect.com/science/article/pii/0304407686900023) [title]. An iterated-OLS alternative is [Bellégo, Benatia & Pape, arXiv 2203.11820](https://arxiv.org/pdf/2203.11820) [title].
  - statsmodels `HurdleCountModel` defaults to `dist='poisson'`, `zerodist='poisson'` and has no offset or exposure support [pkg].
- **Benchmarks (2026):** decision-aware intermittent-demand benchmark ([arXiv 2609.13840](https://arxiv.org/pdf/2609.13840)); probabilistic intermittent benchmarks ([fable.intermittent, arXiv 2609.28607](https://arxiv.org/pdf/2609.28607)); hierarchical Bayesian TSB ([arXiv 2511.12749](https://arxiv.org/pdf/2511.12749)) [titles].

### Inferences
**Classification (for forecaster choice only, not for elasticity):**
- ADI = mean inter-demand interval (statsforecast convention), and $CV^2=\text{Var}/\text{mean}^2$ of the nonzero sizes (population variance).
- On y = [0,0,3,0,0,0,2,0,1,0]: intervals (3, 4, 2), so ADI = 3.0 (n/n_nonzero would give 3.33; pin the convention); CV² = 0.167, so "intermittent".
- [recalled] The Kostenko–Hyndman boundary "SBA better if CV² > 2 − 3·ADI/2" passes through (1, 0.5) and (1.33, 0), which reproduces both box cut-offs.

**Recursions (initialise $\hat z_0$ = first nonzero size, $\hat p_0$ = first interval):**
- Croston, updated only when $y_t>0$: $\hat z\leftarrow\hat z+\alpha(y_t-\hat z)$ and $\hat p\leftarrow\hat p+\alpha(q_t-\hat p)$, where $q_t$ is the number of periods since the last demand. Forecast $\hat z/\hat p$.
- SBA: $(1-\alpha/2)\hat z/\hat p$.
- TSB:
  - every period, $\hat d\leftarrow\hat d+\alpha_p(1\{y_t>0\}-\hat d)$ with $\hat d_0=1\{y_0>0\}$;
  - only when $y_t>0$, $\hat z\leftarrow\hat z+\alpha_d(y_t-\hat z)$;
  - forecast $\hat d\hat z$.
- Defaults: α = 0.1 (Croston/SBA); α_d = 0.1; α_p = 0.1, tuned over {0.05, 0.1, 0.2, 0.3} by rolling-origin MSE. Never tune on MAE, which rewards all-zero forecasts.

**Elasticity with zeros:**
- PPML: solve $\sum_t(Q_t-\mu_t)w_t=0$ with $\mu_t=\exp(w_t'\theta)$ (statsmodels GLM Poisson). SEs clustered by week, or HAC with lag 7.
- Use NB2 for efficiency when $\hat\phi>1.5$.
- Hurdle, to explain extensive vs intensive margins:
  - $\Pr(Q>0)=1-\exp(-e^{z'\delta})$;
  - $E[Q\mid Q>0]=\lambda/(1-e^{-\lambda})$;
  - total elasticity = $\partial\ln\Pr(Q>0)/\partial\ln p+\partial\ln E[Q\mid Q>0]/\partial\ln p$.
  - Build in-house if exposure offsets from Q1 are needed.
- [own sim] Base DGP, mean $\hat\beta$ (truth −1.5):

| μ0 (units/day) | Zero days | OLS log(1+Q) | OLS log Q on Q>0 | PPML (daily) | Weekly log-OLS |
|---|---|---|---|---|---|
| 0.3 | 73% | −0.23 | −0.14 | −1.36 | −0.96 |
| 1 | 35% | −0.72 | −0.58 | −1.52 | −1.64 |
| 3 | 5% | −1.17 | −1.28 | −1.48 | −1.51 |
| 10 | 0% | −1.44 | −1.61 | −1.51 | −1.50 |

**Aggregation rule:**
- PPML is valid at daily level whatever the zero share, so aggregation is optional for estimation.
- Aggregate to weeks for forecasting, or for estimation when the daily zero share exceeds 0.5 and the median price spell is at least 14 days.
- Weekly price is the units-weighted posted price; split or drop weeks that contain a price change.
- Weekly log-OLS is acceptable only with ≤ 5% zero weeks and at least 10 units per week; at about 2 units/week it gave −0.96.

**Minimum data (Fisher information), [own sim] validated:**
- $\text{SE}(\hat\beta)\approx\sqrt{\hat\phi}/\sqrt{\sum_t\hat\mu_t\tilde x_t^2}$, where $\tilde x$ is $\ln p$ residualised on the controls (μ-weighted). Monte Carlo SD matched the formula within about 10% across 18 scenarios. Examples (MC SD vs formula):

| Units/yr | Price step | MC SD | Formula SE |
|---|---|---|---|
| 1,095 | 10% | 0.70 | 0.70 |
| 1,095 | 20% | 0.38 | 0.38 |
| 365 | 20% | 0.67 | 0.66 |
| 110 | 20% | 1.26 | 1.20 |

- Closed form for one permanent change of Δ log points, with share p of units at the new price: SE ≈ $1/(\Delta\sqrt{N p(1-p)})$. Units required at p = 0.5, φ = 1 (multiply by φ):

| Target SE | Δ = 0.05 | Δ = 0.10 | Δ = 0.20 |
|---|---|---|---|
| 0.25 | 25,600 | 6,400 | 1,600 |
| 0.5 | 6,400 | 1,600 | 400 |

- This should drive v1 §20 (rolling windows), §43–45 (DQS and confidence) and the decision to borrow strength or recommend an experiment.

**Tests:**
- T6.1 Reference values on y = [0,0,3,0,0,0,2,0,1,0] with α = 0.1:
  - Croston: $\hat z$ = 2.71, $\hat p$ = 2.99, forecast 0.9064;
  - SBA: 0.8610;
  - TSB (α_d = α_p = 0.1): $\hat d$ = 0.2107, forecast 0.5711.
- T6.2 Unit invariance: rescaling Q by 100 leaves PPML β unchanged and changes the log(1+Q) β.
- T6.3 μ0 = 0.3 DGP: PPML mean β in [−1.7, −1.2]; log(1+Q) must fail.
- T6.4 Reported SE within 15% of Monte Carlo SD in the base DGP at 1,095 units/yr and a 10% step.

### Gaps
- Could not verify the Kostenko–Hyndman boundary formula or the M5 metric definitions directly.
- No source compared hurdle vs PPML elasticities on retail SMB data.
- The orthogonal DML score for count outcomes is open (see Q0).

---

## Q7. Cross-price elasticities and cannibalisation within one shop: estimation with little data, and when to skip

### Takeaway
Within-shop cross-price elasticities are small, noisy and often wrong-signed even in large datasets, and SMB price changes are frequently synchronised (shop-wide inflation updates), which makes them unidentified. v2 should run an explicit identification test. Where it passes, estimate one shrunken "substitute-group price index" coefficient per product, and use substitutes' stockouts as natural experiments. Otherwise skip, but still control for group promotions when estimating own-price β.

### Cited Findings
- **Meta-analytic magnitudes:**
  - About 10% of 1,060 published cross-price elasticities were negative (Sethuraman, Srinivasan & Kim 1999, Marketing Science 18(1):23–41) ([NBER w12756](https://www.nber.org/system/files/working_papers/w12756/w12756.pdf)) [snippet].
  - The meta-analytic mean cross-price elasticity is 0.52, and cross-price elasticity scales with market share ([Rev. Ind. Org. 2017](https://link.springer.com/article/10.1007/s11151-017-9578-8)) [snippet].
  - Static models understate cross-price elasticities by up to 5× ([Hendel & Nevo 2006](https://www.econometricsociety.org/publications/econometrica/2006/11/01/measuring-implications-sales-and-consumer-inventory-behavior)).
- **Shrinkage and structure:**
  - Montgomery & Rossi use differential shrinkage that holds elasticities close to additive-utility restrictions, with store variation through expenditure elasticities; this improved on Bayesian and non-Bayesian benchmarks (1999 Mitchell Prize) ([JMR 36(4):413–423](https://journals.sagepub.com/doi/10.1177/002224379903600402)).
  - Smith, Rossi & Allenby use a log-linear demand system in which weak separability imposes equality restrictions on cross-price elasticities. The partition into groups is estimated, and it can conflict with category labels ([Marketing Science 38(4):690–710, 2019](https://pubsonline.informs.org/doi/abs/10.1287/mksc.2019.1159)).
- **Availability as identification:**
  - Stockout-based substitution: [Anupindi et al. 1998](https://pubsonline.informs.org/doi/10.1287/mksc.17.4.406); [Vulcano et al. 2012](https://pubsonline.informs.org/doi/10.1287/opre.1110.1012).
  - Conlon & Mortimer used availability observed every four hours as variation identifying substitution ([AEJ: Micro 2013](https://www.aeaweb.org/articles?id=10.1257%2Fmic.5.4.1)).
  - Multi-store substitution in high-density traditional retail ([EJOR 266(1) 2018](https://ideas.repec.org/a/eee/ejores/v266y2018i1p99-111.html)) and restricted-logit demand transfer at scale ([arXiv 2608.12680](https://arxiv.org/pdf/2608.12680)) [titles].
- **Promotion cannibalisation:** about one third of a promotion bump is cross-brand switching ([van Heerde et al. 2004](https://pubsonline.informs.org/doi/abs/10.1287/mksc.1040.0061)).

### Inferences
**Split v1 §31 into two concepts:**
- (A) Within-shop cross-price and cannibalisation, estimable from own data.
- (B) Competitor price effects, which need competitor price time series. Competitor information is usually cross-sectional for SMBs and is then absorbed by fixed effects.

**Identification test (skip rule):**
- $\text{SE}(\beta_{ij})\approx\sqrt{\phi}/(\sqrt{N_i}\,\text{sd}(\ln p_j)\sqrt{1-\rho_{ij}^2})$, where $\rho_{ij}$ is the partial correlation of $\ln p_i,\ln p_j$ given controls.
- Estimate only if:
  - $N_i\text{Var}(\ln p_j)(1-\rho^2)\ge 1/\text{SE}_\text{target}^2$ (≥ 4 for SE 0.5);
  - $|\rho_{ij}|<0.8$;
  - j has at least 2 price changes not within ±7 days of i's changes.
- Shop-wide repricing (common under NGN/KES inflation) gives ρ ≈ 1, so skip.

**Low-parameter specification:**
- $\ln\mu_{it}=\dots+\beta_{ii}\ln p_{it}+\beta_{iG}\ln\bar P_{G(i),-i,t}$, where $\ln\bar P_{G,-i}=\sum_{j\in G,j\ne i}w_j\ln p_{jt}$ with fixed revenue shares $w_j$ from the training window.
- Groups come from merchant tags, category, or LLM text similarity (v1 §32 Similarity).
- Prior: $\beta_{iG}\sim N(0,0.5^2)$, or $N(0.3,0.3^2)$ for merchant-declared substitutes (judgement; the meta-analytic mean 0.52 is for brand-level data and is not a direct prior).
- Partial pooling across products in the group.

**Stockout natural experiment:** diversion $D_{j\to i}=(\bar Q_i^{\,j\text{ out}}-\hat Q_i^{\,j\text{ in}})/\hat Q_j^{\,\text{in}}$ on days when j is out and i is in stock. Needs at least 5 such days. Use it for the prior sign and magnitude, and for v1 §32 ObservedSubstitution.

**v1 §32 edge weight:**
- Replace the 4-factor product with prior inclusion probability $\pi_{ij}=\sigma(a+b\cdot\text{sim}_{ij}+c\cdot\text{sameCategory}_{ij})$ times posterior evidence (diversion and β_iG posterior).
- In the v1 product, a zero in any factor (e.g. missing geographic overlap) silently deletes the edge.

**Structural systems, for reference only:**
- AIDS: $w_i=\alpha_i+\sum_j\gamma_{ij}\ln p_j+\beta_i\ln(X/P)$, with homogeneity $\sum_j\gamma_{ij}=0$ and symmetry $\gamma_{ij}=\gamma_{ji}$. Marshallian elasticity $e_{ij}=-\delta_{ij}+(\gamma_{ij}-\beta_i w_j)/w_i$ (Stone-index approximation) [recalled; Deaton & Muellbauer 1980].
- Nested logit: $\ln(s_j/s_0)=x_j\beta-\alpha p_j+\sigma\ln s_{j|g}$ [recalled; Berry 1994]. It needs market size, i.e. the outside good, which a single shop does not observe.
- Both are infeasible beyond 3–5 items with SMB data, so skip them in V2.

**Even when skipping cross effects,** include $\ln\bar P_{G,-i}$ or group promotion dummies as controls in i's own-price model. Otherwise coincident promotions on substitutes bias β_ii toward zero.

**Tests:**
- T7.1 Two products with perfectly synchronised price changes: the gate must refuse cross-price estimation.
- T7.2 Simulated substitutes with $\beta_{iG}=0.4$ and independent price changes (5,000 units): the posterior mean must lie in [0.2, 0.6] and shrink toward the prior mean when units are 500.
- T7.3 Inject a 30% diversion during j's stockouts; the diversion estimator must recover it within ±10 percentage points with 10 stockout days.

### Gaps
- No SMB- or emerging-market cross-price estimates were found.
- AIDS and nested-logit formulas are recalled, not verified this session.
- No study was found validating LLM text similarity as a substitute-group prior.

---

## Q8. Demand-forecasting baselines for SMBs and hierarchical reconciliation

### Takeaway
Use a per-product baseline ladder: seasonal naive (always), ETS for smooth series, SBA/TSB/ADIDA/IMAPA for intermittent ones, and one global LightGBM (Poisson/Tweedie) model across the shop's products. Select by rolling-origin MASE or RMSSE plus bias. Train on unconstrained demand (Q1). Reconcile product→category→shop forecasts, by bottom-up or structurally scaled WLS for short histories and MinT-shrink only with enough residual history. Do not use the ML model's price response as the elasticity.

### Cited Findings
- M5 required forecasts for 42,840 hierarchical Walmart series (30,490 at the bottom level, summed upward). It ran on Kaggle from 3 March to 30 June 2020 ([Makridakis, Spiliotis & Assimakopoulos 2022, IJF 38(4):1346–1364](https://www.sciencedirect.com/science/article/pii/S0169207021001874)) [snippet]. Representativeness of M5 data is discussed in [arXiv 2103.02941](https://arxiv.org/pdf/2103.02941) [title].
- MinT reconciliation minimises the trace of the reconciled forecast-error covariance ([Wickramasuriya, Athanasopoulos & Hyndman 2019, JASA 114(526):804–819](https://www.researchgate.net/publication/323762400_Optimal_Forecast_Reconciliation_for_Hierarchical_and_Grouped_Time_Series_Through_Trace_Minimization)) [snippet].
- Further reconciliation sources:
  - textbook treatment ([FPP2 §10.7](https://otexts.com/fpp2/reconciliation.html)) [title];
  - review ([IJF 2023](https://www.sciencedirect.com/science/article/pii/S0169207023001097)) [title];
  - non-negative reconciliation ([Monash WP 15/19](https://www.monash.edu/business/ebs/research/publications/ebs/wp15-2019.pdf)) [title];
  - reconciliation for very short hierarchical series ([arXiv 2409.18550](https://arxiv.org/pdf/2409.18550)) [title];
  - reference implementations ([hierts](https://github.com/elephaint/hierts)) [title].
- Temporal aggregation (ADIDA) improves intermittent forecasts ([Nikolopoulos et al. 2011](https://eprints.lancs.ac.uk/id/eprint/58330/)).
- Training on censored sales perpetuates underestimation; latent-demand recovery removes most of the bias ([FreshRetailNet-50K](https://arxiv.org/abs/2505.16319)) [snippet]. Tobit ETS is a censoring-aware ETS ([IJPR 2025](https://www.tandfonline.com/doi/full/10.1080/00207543.2025.2600521)) [snippet].

### Inferences
**Ladder (defaults; statsforecast and statsmodels names):**
- `SeasonalNaive(season_length=7)` on daily data, and the mean of the last 4 same weekdays.
- Smooth series: `ETSModel(error='add', trend=None, seasonal='add', seasonal_periods=7)` on daily data, or ETS(A,N,N) on weekly totals.
- Intermittent or lumpy: `CrostonSBA`, `TSB(alpha_d=0.1, alpha_p=0.1)`, `ADIDA`, `IMAPA`.
- Global LightGBM across the shop's products (and across shops when tenancy allows). Judgement defaults (not sourced):
  - `objective='tweedie'` (variance power 1.2) or `'poisson'`;
  - `learning_rate=0.05`, `num_leaves=15`, `min_data_in_leaf=30`;
  - `feature_fraction=0.8`, `bagging_fraction=0.8`, `lambda_l2=1`;
  - early stopping on a time split;
  - `monotone_constraints` of −1 on log price and promotion depth, so demand cannot rise with price [recalled LightGBM feature].
- Features: calendar and payday (Q4), lead-matched weather (Q5), price/regular/depth/reference gap/ending class (Q2, Q3), and lags and rolling means that start at or after the forecast horizon.

**Selection:**
- Rolling-origin over the last 8–12 weeks.
- $\text{MASE}=\text{mean}|e_t|/\text{mean}|y_t-y_{t-7}|$ (in-sample scaling).
- $\text{RMSSE}=\sqrt{\text{mean}(e_t^2)/\text{mean}((y_t-y_{t-1})^2)}$ [recalled M5 definition].
- Mean error (bias), and Poisson deviance.
- Fall back to seasonal naive if nothing beats it by at least 5% MASE.

**Role relative to elasticity:**
- The forecaster supplies the baseline at the current price. The price simulator applies the causal β: $\hat Q(p')=\hat Q(p_0)(p'/p_0)^{\hat\beta_R}$, using the promotional β_D for temporary prices.
- In DML, the nuisance $E[Y\mid X]$ must exclude price and price-derived features. The forecaster's own price response is not a causal elasticity (consistent with v1 §2).

**Hierarchy:**
- Summing matrix S (product → category → shop; optionally daily → weekly). Reconciled forecasts $\tilde y=SG\hat y$.
- Choice of G:
  - bottom-up: $G=[0\mid I]$;
  - MinT: $G=(S'W^{-1}S)^{-1}S'W^{-1}$, with W the base-forecast error covariance (OLS: W = I; WLS-structural: $W=\text{diag}(S\mathbf 1)$; WLS-variance: diagonal residual variances; MinT-shrink: shrunk sample covariance) [recalled from FPP/JASA].
- SMB defaults:
  - WLS-structural when fewer than about 4 × (number of series) residual days exist;
  - MinT-shrink otherwise;
  - bottom-up when bottom series are smooth.
- Clip negatives to zero and re-aggregate, or use non-negative reconciliation.

**Tests:**
- T8.1 Coherence: reconciled product forecasts sum to the category and shop forecasts within 1e-9.
- T8.2 Leakage: no lag feature uses data after T for horizon h ≥ 1.
- T8.3 The LightGBM monotone constraint holds: raising price by 10% never raises predicted demand.
- T8.4 The intermittent ladder is run on a 90%-zero series; the selector must not choose an all-zero forecast when bias is checked.

### Gaps
- M5 winning-method details (LightGBM prevalence, Tweedie objective, the SNAP-day calendar features) are recalled, not verified; fetch was blocked.
- Exact MinT-shrink estimator details were not re-verified.
- No SMB-specific forecasting benchmark was found. The 2026 intermittent benchmarks (arXiv 2609.13840, 2609.28607) were seen as titles only.
