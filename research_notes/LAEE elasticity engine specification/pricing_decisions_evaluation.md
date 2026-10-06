# Price simulation, risk-aware price optimisation, price experimentation and out-of-sample evaluation for LAEE

Scope: these notes cover how to turn an uncertain elasticity into a safe, profitable price recommendation for small businesses (sparse data, low tolerance for losses, often cash-priced African markets). They also cover how to prove, out of sample, that the system's predictions and recommendations are good. They feed the expanded LAEE algorithm specification, the successor to `/home/user/laee-engine/docs/spec/v1-original.md`.

**Evidence tags**
- **[S]**: confirmed in this session (2026-10-06) from web-search results.
- **[U]**: established literature cited from memory, with DOI or URL. **Not re-verified in this session**, for two reasons. The shared web-search budget ran out after one batch of ten searches. Every full-text domain tried was blocked by the egress proxy: pubsonline.informs.org, dspace.mit.edu, hbs.edu, eml.berkeley.edu, web.mit.edu, business.columbia.edu, arxiv.org, proceedings.mlr.press, docs.feast.dev, otexts.com, numpy.org and sites.stat.washington.edu. Check the bibliographic details and any quoted figure before a [U] item goes into the spec.
- **[C]**: computed in this session in the project venv (numpy 2.4.6, scipy 1.17.1, pandas 3.0.6). The scripts are `verify_part1.py`, `verify_part2.py` and `verify_part3.py` in the session scratchpad (`/tmp/claude-0/-home-user-price-elasticity-predictor/e5589faf-26e1-5db5-b2fb-87bd4bdbab60/scratchpad/`). The results are deterministic test vectors or illustrative simulations. They are **not** evidence about real businesses.

**Notation**

| Symbol | Meaning |
|---|---|
| P | price |
| P0 | current price |
| c | unit variable cost |
| ε | own-price elasticity (negative) |
| r | ln(P/P0) |
| τ | VAT rate included in the displayed price |
| f | percentage payment fee (card or mobile money) |
| κ | 1/(1+τ) − f, the net share of the displayed price |
| m0 | (κP0 − c)/(κP0), the current percentage margin |
| H | decision horizon |
| Π(P) | profit over H |
| ΔΠ(P) | Π(P) − Π(P0) |
| S | number of posterior draws |

---

## 0. What v1 gets wrong or leaves underspecified (§34–38, §46–51, §66–67, §73–79, §89)

### Takeaway
v1 has the right pipeline: simulate, compute profit, adjust for risk, apply constraints, backtest. The decision-critical pieces, however, are inconsistent, unobservable or unspecified:
- It simulates with a point constant elasticity over ±20%, which is biased toward recommending changes.
- Its variance penalty is not scale-invariant.
- Its regret uses an unknowable optimum, and minimising expected regret silently cancels the risk aversion in §38.
- It scores intermittent counts with MAPE and sMAPE.
- It relies on one fixed train/validation/test split.
- It uses PSI without sample-size-aware thresholds.
- It proposes a 10/90 customer-split price test that is inefficient and usually infeasible for posted-price shops.
- It claims that "a 5% experiment would substantially improve the estimate", which is false at small-business volumes.

### Cited Findings
- §34 simulates `Q1 = Q0·(P1/P0)^ε` with a single point ε and computes expected quantity, revenue and gross profit "for every candidate price". §33 already outputs `elasticity_lower` and `elasticity_upper`, which §34 never uses — [v1 spec §33–34](/home/user/laee-engine/docs/spec/v1-original.md)
- §36 defines `Profit = (P − C)Q − F` with `C = C(P, X)` — [v1 spec §36](/home/user/laee-engine/docs/spec/v1-original.md)
- §37 searches `np.arange(P0*0.80, P0*1.20, P0*0.005)` and picks `argmax(expected_profit)` — [v1 spec §37](/home/user/laee-engine/docs/spec/v1-original.md)
- §38 uses `Objective = E[Profit] − λ·Var(Profit)` "or ... CVaR". No value is given for λ or for the CVaR level — [v1 spec §38](/home/user/laee-engine/docs/spec/v1-original.md)
- §46 uses one split: train Jan 2023–Dec 2024, validate Jan–Jun 2025, test Jul–Dec 2025. §47 describes an expanding monthly walk-forward — [v1 spec §46–47](/home/user/laee-engine/docs/spec/v1-original.md)
- §48 lists:
  - demand metrics: MAE, RMSE, MAPE, SMAPE;
  - elasticity metrics: MAE, RMSE, interval coverage, sign accuracy;
  - `Regret = Profit(P*) − Profit(P_recommended)`, with "The model should minimize expected regret".

  — [v1 spec §48](/home/user/laee-engine/docs/spec/v1-original.md)
- §49 compares predicted and actual %ΔQ after historical price changes. §50 uses PSI and "distribution-distance metrics" and retrains when "prediction error > threshold OR feature drift > threshold" without values. §51 says "recalculate ... periodically" — [v1 spec §49–51](/home/user/laee-engine/docs/spec/v1-original.md)
- §66 sets `price_min = max(cost*1.10, business_minimum_price)` and `price_max = min(current_price*1.15, business_maximum_price)` — [v1 spec §66](/home/user/laee-engine/docs/spec/v1-original.md)
- §67 returns `"confidence": 0.86` and `"risk": 0.21` without defining either — [v1 spec §67](/home/user/laee-engine/docs/spec/v1-original.md)
- §73 lists version fields. §74–75 require `available_at ≤ prediction_time` — [v1 spec §73–75](/home/user/laee-engine/docs/spec/v1-original.md)
- §77 proposes "10% of customers: $10.99; 90%: $9.99". §78 says "A controlled 5% price experiment would substantially improve the estimate." §79 says "SMB changes price → Observed result → Causal elasticity evidence" — [v1 spec §77–79](/home/user/laee-engine/docs/spec/v1-original.md)
- §89 defines `P* = argmax_P E[(P−C)Q(P|X)] − λ·Risk(P)` — [v1 spec §89](/home/user/laee-engine/docs/spec/v1-original.md)

### Inferences

#### Defect list (the fix for each is detailed in the referenced note section)

| v1 § | Defect | Fix (note section) |
|---|---|---|
| 34 | Uses a point ε, so neither parameter uncertainty nor demand noise is propagated. This contradicts Rule 4 and §39. | Posterior-predictive simulation with common random numbers (§1) |
| 34 | Constant-elasticity (CE) extrapolation to ±20% **overpredicts quantity in both directions** whenever elasticity magnitude rises with price. In the logit example [C], CE predicts +17.4% profit at +20% where the truth is +2.6%. The simulator is therefore biased toward recommending changes. | Curvature-capable forms, plus restricting candidates to the support of historical prices (§1) |
| 34 | `Q0` is undefined: last period, an average, or a forecast? | `Q0` := the model's baseline forecast at P0 for the same horizon and covariates, from the same model as Q1 (§1) |
| 34 | Ignores stockouts and capacity (Rule 5), cross-effects on the business's own products (§31–32) and reference-price dynamics. | `Q_sold = min(D, stock)`, plus a substitute check (§1, §2) |
| 35 | `%ΔRevenue` must be measured against counterfactual revenue at P0 over the same horizon, not against last period. | §1 |
| 36 | Cost depends on volume (discounts, capacity), not price: use `C(Q,X)`. VAT and payment fees are missing. F does not move the argmax. | κ formula; avoidable step costs only (§2) |
| 37 | The ±20% grid conflicts with §66 (+15%) and §65 (discrete prices). `np.arange` excludes 1.20·P0 and can miss P0 through float error. It also produces prices that cannot be paid in cash (e.g. 10.35). A boundary argmax is never flagged. When \|ε\| ≤ 1 the objective is unbounded, so the result is always the boundary. | A discrete feasible set that always contains P0 exactly; boundary and unbounded flags (§2, §5) |
| 38 | λ·Var has units of 1/currency, so the decision depends on the currency unit. [C]: λ=0.001 picks 12.0 with prices in units but 10.0 with the same prices in cents. Variance also penalises upside, the distribution behind Var/CVaR is unspecified, and there is no CVaR level α and no status-quo reference. | Loss-averse utility vs status quo, chance constraint and CVaR budget (§3) |
| 46 | The single split needs three years of data, which most small businesses do not have. One origin gives a high-variance estimate. Tuning leaks into the test, and there is no embargo. | Rolling origin (§6) |
| 47 | No horizon, refit cadence, minimum training length, baselines, significance test or **decision** evaluation. | §6 |
| 48 | MAPE is undefined at zero sales. sMAPE is 200% whenever y=0 and ŷ>0. MAE and WAPE reward the zero forecast on intermittent items [C], yet profit needs the mean. Elasticity MAE, RMSE and coverage need a ground truth that exists only in simulation or experiments. Regret needs the unobservable P*. Also `E[Regret(P)] = E[max_P' Π(P')] − E[Π(P)]`, so **minimising expected regret ≡ maximising expected profit**, which reverts §38's risk aversion. | Mean-consistent metrics, proper scores and observable regret proxies (§6) |
| 49 | A naive pre/post %ΔQ: no adjustment for seasonality or confounders, no windows, no uncertainty on the "actual", endogenous owner-chosen changes, and the asymmetry of % changes. | As-of event study with log effects and z-scores (§6) |
| 50 | No PSI thresholds. With 100 observations per window, PSI > 0.1 in 85% of windows **without any drift** [C]. No multiple-testing control. Retraining is conflated with redeploying. | Sample-size-aware PSI, CUSUM, FDR, champion/challenger gate (§7) |
| 51 | "Periodically" has no cadence. Market-definition changes are not versioned by `available_at`, so they get applied retroactively in backtests. | Bitemporal versioning (§7) |
| 66 | `cost*1.10` is a 10% **markup** (a 9.09% margin). There is no step limit downward and no frequency limit. An empty interval (min > max) is not handled. Rounding after optimisation can break constraints. | Feasible-set pipeline (§5) |
| 67 | The output has no horizon, no intervals for ΔQ/ΔR/ΔΠ, no P(loss), no action type (change, hold or test), no binding constraints, no identification tier (§76) and no lineage. "risk" and "confidence" are undefined. | Output contract (§3) |
| 73 | Missing: data snapshot hash, code commit, seeds, environment lock, prompt version, risk/constraint config and the population-prior version. | Lineage tuple (§7) |
| 74–75 | Missing:<ul><li>bitemporal and revision semantics for backfills and data vintages;</li><li>training-time leakage rules (scalers, encoders, tuning);</li><li>cross-tenant prior leakage;</li><li>look-ahead from LLM pretraining;</li><li>a label-window embargo;</li><li>timezone and day-close rules;</li><li>late-arriving POS syncs.</li></ul> | §7 |
| 77 | A 10/90 split inflates estimator variance **2.78×** versus 50/50 [C]. A customer-level price split in a posted-price shop is often operationally infeasible, perceived as unfair, and possibly unlawful. There is no sample size, stopping rule or guardrail. | Switchback (time-randomised) designs and VOI-driven test selection (§4, §5) |
| 78 | False at SMB volumes [C]: +5% for 2 weeks per arm at 100 units/week moves posterior SD(ε) from 0.60 to 0.58. An SE of 0.3 at +5% needs ≈18,700 units even with pure Poisson noise. | EVSI, larger gaps and pooled priors (§4) |
| 79 | Owner-chosen price changes are not automatically causal. A recommender that converges to one price stops learning (incomplete learning). Recommendations also change the data the model later learns from (performative feedback). | Dispersion guard, logged reason codes, randomisation (§4, §6) |
| 89 | Missing: constraints, horizon, status-quo reference, discrete feasible set and the value of information. λ is unspecified. | §2–§4 |

#### v2 replacement defaults at a glance

| Item | Default | Section |
|---|---|---|
| Decision horizon H | 28 days | §3 |
| Posterior draws S | 4,000 (20,000 if CVaR at α=0.95) | §1 |
| Candidate range ("recommend" mode) | Intersection of P0·[0.90, 1.10] and [0.95·min historical price, 1.05·max historical price] | §1, §5 |
| Price set | The business's price ladder (cash-convenient points) ∪ {P0} | §5 |
| Objective | E[U_LA(ΔΠ_H)], with loss aversion λ_LA = 2.25 | §3 |
| Chance constraint | P(ΔΠ_H < 0) ≤ δ = 0.20 (0.10 if identification tier ≥ 6) | §3 |
| Downside budget | CVaR_0.90 of loss ≤ 5% of baseline profit over H | §3 |
| Step and frequency | \|ΔP/P0\| ≤ 10% per change; ≥ 14 days between changes; ≤ 4 changes/quarter | §4, §5 |
| Margin floor | (κP − c)/(κP) ≥ 10% | §5 |
| Experiments | Switchback, 2–4 weeks per arm, +15–20% gap on low-revenue-share items, chosen by EVSI | §4 |
| Backtest | Weekly rolling origins, ≥ 12 origins, embargo = label window, as-of everything | §6 |
| Drift | Two-sided CUSUM on standardised errors (k=0.5, h=5); PSI vs a χ² threshold; BH-FDR across features | §7 |

### Gaps
- Sections outside the assignment interact with these issues and were only skimmed: §39–45 (intervals, ensemble, confidence) and §60–65 (segments, anchoring, promotion and stockout contamination, censoring, discretisation).

---

## 1. Demand functional forms for simulation

### Takeaway
Use the constant-elasticity (log-log) form only as a **local** approximation inside the empirical price support (default ±10% of P0), and always with posterior draws rather than a point ε. Constant elasticity is the tangent line in log-log space. If elasticity magnitude rises with price, which is the usual case and is known as "Marshall's second law", that tangent lies above the true curve. CE therefore **overpredicts quantity for both price increases and price cuts**, by roughly ½·|dε/d ln P|·r² in log terms, and the simulator is biased toward recommending moves. For wider moves, or when estimates are near or above ε = −1, use a curvature-capable form (log-quadratic, logit/WTP, or monotone piecewise) and inflate uncertainty outside the support.

### Cited Findings
- Standard price-response forms used in revenue management (linear, constant-elasticity, logit) and the reservation-price/WTP interpretation of logit demand appear in Phillips' textbook — [Phillips 2005, *Pricing and Revenue Optimization*, Stanford UP](https://books.google.com/books?vid=ISBN9780804746984) [U]
- Besbes & Zeevi, "On the (surprising) sufficiency of linear models for dynamic pricing with demand learning", *Management Science* — [listing](https://dl.acm.org/doi/10.1287/mnsc.2014.2031) [S for title and DOI]. Its content is [U]: a linear demand model refit locally can learn the optimal price with near-optimal regret even when the true demand is nonlinear, because only local behaviour near the optimum matters.
- den Boer's survey reviews parametric and nonparametric demand models used in pricing-and-learning — [den Boer 2015, *Surveys in Operations Research and Management Science* 20(1):1–18](https://doi.org/10.1016/j.sorms.2015.03.001) [S]
- Duan's smearing estimator for retransforming log-linear predictions: `E[Q|x] = exp(x'β̂)·(1/n)Σ exp(ê_i)` — [Duan 1983, JASA 78(383):605–610](https://doi.org/10.1080/01621459.1983.10478017) [U]
- Poisson pseudo-maximum-likelihood fits log-linear mean models with zero outcomes and avoids retransformation bias under heteroskedasticity — [Santos Silva & Tenreyro 2006, REStat 88(4):641–658](https://doi.org/10.1162/rest.88.4.641) [U]
- Besbes & Zeevi (2009) frame the cost of not knowing the demand function as regret against a clairvoyant who knows it, in both parametric and nonparametric settings — [Operations Research 57(6):1407–1420](https://pubsonline.informs.org/doi/10.1287/opre.1080.0640) [S]

### Inferences

#### Functional forms (with κ: replace c by c/κ in every closed form)

| Form | Q(P) | ε(P) | P* (constant c) | Use when | Failure mode |
|---|---|---|---|---|---|
| Constant elasticity | A·P^ε | ε (constant) | c·ε/(1+ε), only if ε < −1 | Small moves (≤ ±10%) inside the support | No choke price. Unbounded profit if ε ≥ −1. Tangent overprediction. Explodes near −1. |
| Linear | a − bP | −bP/(a − bP) | (a/b + c)/2 | Wide observed price range; a choke price exists | Q < 0 beyond a/b (clip at 0) |
| Exponential (semi-log) | A·e^{−bP} | −bP | c + 1/b | \|ε\| rises roughly linearly in P | Thin tail |
| Logit / WTP | M/(1 + e^{(P−μ)/s}) | −(P/s)·σ((P−μ)/s) | Solves P* − c = s/σ((P*−μ)/s) | Bounded market; reservation-price heterogeneity | Needs market size M; flat in the tails |
| Log-quadratic | ln Q = a + ε0·r + ½γ·r² | ε0 + γ·r | Numerical; Lerner holds with ε(P*) | Testing for curvature (γ < 0 means Marshall's second law) | Needs at least 3 distinct price levels |
| Monotone piecewise (isotonic in log P, covariate-adjusted) | Step or linear between observed price levels | Segment slopes | Evaluate only at ladder points | Discrete cash price ladders with few levels | No information between or beyond the observed levels |

Here σ(z) = 1/(1 + e^{−z}). General rule for every form: at the optimum, `(κP* − c)/(κP*) = −1/ε(P*)`. The Lerner condition uses the elasticity **at the optimum**, not at the current price, unless elasticity is constant.

**Closed-form test vectors [C]** (c = 6, κ = 1):

| Form | Parameters | P* |
|---|---|---|
| CE | ε = −1.1 | 66 (11× cost) |
| CE | ε = −1.2 | 36 |
| CE | ε = −1.5 | 18 |
| CE | ε = −2 | 12 |
| CE | ε = −3 | 9 |
| CE | ε = −4 | 8 |
| Linear | Q = 100 − 5P | 13 |
| Exponential | b = 0.25 | 10 |
| Logit | M=100, μ=12, s=2 | 11.1143 (profit 311.43; FOC: P* − c = 5.1143 = s/σ) |

All values match a dense grid argmax.

#### When CE extrapolation fails, with numbers [C]
Logistic-WTP truth: M=100, μ=12, s=2, P0=10, c=6. At P0: ε = −1.3447, Q0 = 73.106, current profit 292.42.

| P1 | CE quantity | True quantity | CE / true | CE ΔΠ% | True ΔΠ% |
|---|---|---|---|---|---|
| 8 (−20%) | 98.689 | 88.080 | 1.120 | −32.50 | −39.76 |
| 9 (−10%) | 84.233 | 81.757 | 1.030 | −13.58 | −16.12 |
| 11 (+10%) | 64.312 | 62.246 | 1.033 | +9.96 | +6.43 |
| 12 (+20%) | 57.211 | 50.000 | 1.144 | **+17.39** | **+2.59** |

- The CE-implied optimum is 23.41. A v1 grid would return the +20% boundary (12), but the true optimum is 11.11 (+6.50%).
- Second-order log-quantity bias ≈ −½·(dε/d ln P)·r², with dε/d ln P = −6.26 here. The approximation and the exact bias are:

  | Move | Approximation | Exact |
  |---|---|---|
  | +5% | 0.0075 | 0.0080 |
  | +10% | 0.0284 | 0.0327 |
  | +20% | 0.104 | 0.135 |

- **At a ±10% guardrail the CE quantity bias is about 3%, the same order as typical profit gains**, so curvature must be estimated or moves kept small.
- Linear Q = 100 − 5P at P0 = 10 (ε = −1) [C]: CE predicts 62.5 vs a true 60 at P = 8, and 41.67 vs 40 at P = 12. CE overpredicts both ways here too.

Other failure triggers:
- moving beyond the historical price support;
- crossing the competitor's price (the demand curve is kinked around relative price parity);
- crossing a cash denomination threshold (e.g. a price that now needs an extra coin or note);
- ε ≥ −1;
- intermittent zero sales (log undefined);
- capacity or stock limits;
- long-run reference-price adaptation (short-run ε understates the long-run response).

#### Local vs arc elasticity
- Point elasticity is ε(P0) = d ln Q/d ln P at P0.
- Midpoint arc elasticity: `ε_arc = [(Q1−Q0)/((Q1+Q0)/2)] / [(P1−P0)/((P1+P0)/2)]`.
- Log-arc elasticity: `ε_log-arc = ln(Q1/Q0)/ln(P1/P0)`. Use it in counterfactual validation because it is consistent with the log model and symmetric.
- [C] Linear demand, 10 → 12: point ε is −1.0 at 10 and −1.5 at 12; midpoint arc −1.2222; log-arc −1.2239.
- DML and fixed-effects estimates are weighted averages of local slopes over the **historical** price variation, so they behave like arc elasticities over the observed support. Using them far outside that support is extrapolation.

#### Simulating with parameter uncertainty (posterior predictive)

```text
inputs: posterior draws θ_s = (lnQ̂0_s, ε_s, γ_s, c_s) for s = 1..S  (S = 4000 default)
        candidate set F (always contains P0 exactly); horizon H; dispersion k (NB)
for each s (vectorised over all candidates, with the SAME θ_s and the SAME noise
uniforms for every P: common random numbers):
    r          = ln(P / P0)
    mu_s(P)    = exp(lnQ̂0_s + ε_s·r + 0.5·γ_s·r²) · smear      # mean demand per period
    D_s(P)     ~ NegBin(mean = mu_s(P)·H, size = k)             # optional realised-noise layer
    Qsold_s(P) = min(D_s(P), stock_H)                           # Rule 5: stockouts
    Π_s(P)     = (κ·P − c_s)·Qsold_s(P) − ΔF(P)                 # ΔF = avoidable step costs only
ΔΠ_s(P) = Π_s(P) − Π_s(P0)
```

Closed forms for tests:
- If ε ~ N(μ, σ²) and the baseline is known, `E[(P/P0)^ε] = exp(μ·r + ½σ²·r²)` (lognormal mean). [C] The Monte Carlo value matches to 5 decimals: for μ = −1.6, σ = 0.35 at P/P0 = 0.8, plug-in 1.42908, formula 1.43344, MC 1.43343.
- The Jensen shift in the mean is small, at most +0.3% at ±20%. The **spread** is what matters: `SD(ln Q1 | ε uncertainty) = |r|·σ` exactly, 0.0781 at −20% [C].
- Delta method in full: `Var(ln Q1) ≈ Var(lnQ̂0) + r²·Var(ε̂) + 2r·Cov(lnQ̂0, ε̂) + σ²_noise/n`. Regress on ln(P/P0), not ln P, so the intercept is the log demand at the current price and its covariance with ε̂ is minimal there.
- Horizon matters [C]. For a 4-week horizon at P = 11 in the §3 toy, parameter uncertainty is 52% of the variance of realised profit and Poisson noise is 48%. Parameter variance grows ∝ H² and noise ∝ H.
  - Compute risk on **horizon totals**.
  - Report "model risk", which testing can reduce, separately from "sales noise", which it cannot.

#### Support restriction and count data
- **Recommend mode:** candidates lie in the intersection of P0·[0.90, 1.10] and [0.95·P_min,hist, 1.05·P_max,hist]. Prices outside that range are allowed only in **experiment mode**.
- **Single-price history:** if the product has only ever had one price, its elasticity comes purely from the hierarchical prior. The action space is then {HOLD, TEST}, not CHANGE.
- **Uncertainty outside the support (heuristic):** widen the ε uncertainty with distance from the observed range, `σ_ε(P) = σ_ε·(1 + dist(r, support)/0.05)`.
- **Daily product sales are small counts with zeros, so never regress log(Q):**
  - Fit a Poisson or negative-binomial GLM with log link, `ln E[Q] = α + ε·ln(P/P0) + x'β`.
  - The coefficient on log price is the elasticity, and no retransformation is needed.
  - If OLS on ln Q is used, apply Duan smearing consistently to **both** Q0 and Q1.

#### Tests that catch wrong implementations
```python
def test_ce_closed_form():           # grid argmax == c*e/(1+e) within the grid step, e in {-1.5,-2,-3,-4}
def test_ce_unbounded_flag():        # e >= -1 -> status 'UNBOUNDED/BOUNDARY', never a finite "optimum"
def test_tangent_overprediction():   # logit(M=100,mu=12,s=2), P0=10: CE/true quantity at 12 == 1.1442 (±1e-3)
def test_lognormal_mean():           # MC mean of (P/P0)**eps within 0.1% of exp(mu*r + .5*sd**2*r**2)
def test_crn_reduces_variance():     # Var(ΔΠ) estimate with common random numbers < with independent draws
def test_support_guard():            # recommend-mode candidate outside the historical support is rejected
def test_monotone_draws():           # every posterior draw gives non-increasing mean demand in P
def test_status_quo_exact():         # ΔΠ_s(P0) == 0 for every draw; P0 is in F exactly (no float drift)
```

### Gaps
- I did not verify in-session the exact regret result of Besbes & Zeevi's "sufficiency of linear models" paper (beyond its title and DOI) or the precise statement of Phillips' formulas; the full texts are blocked.
- I found no empirical estimates of demand curvature (γ) or choke prices for small African retailers. A pilot with ≥ 3 price levels per product is needed.
- "Marshall's second law" (|ε| rising with price) is a standard assumption but is not universal. The log-quadratic γ must be estimated, not assumed.

---

## 2. Profit optimisation

### Takeaway
Optimise risk-adjusted profit over a **discrete, constraint-feasible price set that always contains P0**, and use the closed-form Lerner/inverse-elasticity rule only as a test oracle and an explanation device. The most decision-relevant scalar is the local rule: **raise the price iff |ε| < 1/m0**, where m0 is the current percentage margin net of VAT and fees. Report the posterior probability of that event, for example "88% chance a small increase raises profit".

### Cited Findings
- Lerner index: at the profit-maximising price, `(P − MC)/P = −1/ε` — [Lerner 1934, *Review of Economic Studies* 1(3):157–175](https://doi.org/10.2307/2967480) [U]
- Meta-analytic price elasticities mostly lie below −1:
  - mean −2.62 across 1,851 elasticities — [Bijmolt, van Heerde & Pieters 2005, JMR 42(2):141–156](https://doi.org/10.1509/jmkr.42.2.141.62296) [U];
  - mean −1.76 — [Tellis 1988, JMR 25(4):331–341](https://doi.org/10.1177/002224378802500401) [U].
- Rue La La (Ferreira, Lee & Simchi-Levi, *M&SOM* 18(1):69–88):
  - machine learning forecast demand for new products, and a multi-product price-optimisation algorithm with reference-price effects became a decision-support tool in daily use;
  - in a field experiment, sell-through **did not decrease** due to tool-recommended price increases for medium and high price-point products.

  — [Harvard D3 summary](https://d3.harvard.edu/analytics-online-retailer-demand-forecasting-price-optimization/), [HBS listing](https://www.hbs.edu/faculty/Pages/item.aspx?num=49523) [S]. The often-quoted revenue lift of about 9.7% (90% CI roughly 2.3%–17.8%) is [U] and was not verified.
- Thompson sampling can be combined with a linear program to respect inventory constraints across multiple products, with theoretical guarantees and good numerical performance — [Ferreira, Simchi-Levi & Wang 2018, *Operations Research* 66(6):1586–1602](https://pubsonline.informs.org/doi/10.1287/opre.2018.1755) [S]

### Inferences

#### Single product
- **Profit:** `Π(P) = (κP − c)·Q(P) − F`, with `κ = 1/(1+τ) − f`. τ is the VAT or turnover-tax rate when displayed prices include tax, and f the percentage card or mobile-money fee.
- **CE optimum:** `P* = (c/κ)·ε/(1+ε)`. Test vector [C]: c = 6, ε = −2, τ = 0.16 (illustrative), f = 0.01 gives κ = 0.85207 and **P* = 14.0834**, against a naive 12.0.
- **Local rule (any demand form):** `sign(dΠ/dP at P0) = sign(1 − m0·|ε|)`. Raise iff `ε > −1/m0`; cut iff `ε < −1/m0`.
  - Example: P0 = 10, c = 6 gives m0 = 0.40 and a threshold of −2.5.
  - With a posterior N(−1.8, 0.6²), P(a small increase raises profit) = **0.879** [C].
- **Break-even elasticity for a finite move (CE):** `ε_BE(P) = ln[(κP0 − c)/(κP − c)] / ln(P/P0)`.
  - For P > P0 the move is profitable iff ε > ε_BE(P). For P < P0 it is profitable iff ε < ε_BE(P).
  - With a normal posterior, `P(ΔΠ < 0) = Φ((ε_BE − μ)/σ)` for increases. This gives the chance constraint in closed form.
  - Example (P0 = 10, c = 6): ε_BE = −2.414, −2.341, −2.279 and −2.224 at P = 10.5, 11, 11.5 and 12. With μ = −1.8 and σ = 0.6, P(loss) = 0.153, 0.184, 0.213 and 0.240. These match Monte Carlo [C].
- **Hypersensitivity:** the markup factor ε/(1+ε) is 11, 6, 3, 2 and 1.5 for ε = −1.1, −1.2, −1.5, −2 and −3 [C]. With realistic posterior SDs of 0.3–0.6, the plug-in P* is meaningless. Always optimise the risk-adjusted objective on a grid.

#### When the estimate is inelastic (|ε| ≤ 1)
- Under CE, profit rises with P everywhere, so any optimiser returns the upper bound.
- Possible causes:
  - true short-run inelasticity (staples, habit, weak competition);
  - attenuation or endogeneity bias;
  - a short measurement window that misses churn and reference-price adaptation;
  - aggregation.
- Policy:
  - **Never jump.** Treat the signal as "direction = increase".
  - Cap the step at +5% to +10%.
  - Require a follow-up measurement window of at least 28 days.
  - Run the fairness check (§5).
  - Shrink toward the hierarchical prior; meta-analytic means are around −1.8 to −2.6 [U].
  - Flag the result `boundary_solution = true`.

#### Fixed costs, cost structure and cost uncertainty
- F never moves the argmax. Include only **avoidable** fixed costs that change with the decision, for example an extra staff shift when volume exceeds capacity: a step function F(Q).
- If unit cost depends on volume (supplier discounts), use c = c(Q) and evaluate on the grid.
- Sample c_s from a cost distribution when supplier prices are volatile, as in inflationary markets.
- Report contribution ΔΠ separately from the net-profit level.

#### Multi-product interactions
- **First-order condition:** `Q_i + Σ_j (κP_j − c_j)·∂Q_j/∂P_i = 0`. In margin form:

  ```text
  m_i = −(1 + Σ_{j≠i} m_j · ε_ji · R_j/R_i) / ε_ii
  ```

  Here m_j = (κP_j − c_j)/(κP_j), R_j = revenue, and ε_ji = ∂ln Q_j/∂ln P_i. Substitutes (ε_ji > 0) push prices **above** single-product Lerner prices; complements push them below.
- **Test vector [C]:**
  - Inputs: A = (1000, 800), c = (6, 4), ε11 = −4, ε12 = 0.3, ε21 = 0.2, ε22 = −3.5.
  - Separate Lerner prices are (8.0, 5.6); the joint optimum is **(9.7196, 5.6282)**.
  - The margins (0.3827, 0.2893) satisfy the formula exactly.
  - Joint profit is 5.5531 against 5.4871 when priced separately.
- **Failure mode [C]:** a CE system with own- and cross-elasticities is **globally unbounded**. Raising P1 toward infinity sends unbounded demand to product 2: profit is 5.49 at P1 = 8, 7.40 at 80, 18.59 at 8,000 and 73.99 at 8×10⁶. Always optimise inside a box (±10%), or use a demand system with adding-up such as multinomial logit with an outside option.
- **Default for small businesses:** cross-elasticities are rarely identified, so optimise products independently. Where the substitute network (§31–32) has an edge, also evaluate joint profit for the top two substitutes at the candidate price and reject candidates that lower joint profit.

#### Grid search vs closed form
- **Grid:** enumerate the business's price ladder ∩ guardrails, add P0 exactly, then dedupe and sort. With K ≤ 50 candidates and S = 4,000 draws (200k evaluations), a vectorised numpy pass takes milliseconds. Closed forms are used only as oracles.
- **Never round a continuous optimum.** Test vector [C]: c = 6.2, ε = −2 gives P* = 12.4. With feasible set {10, 15}, nearest rounding picks 10 (profit 0.0380), but the discrete argmax is **15** (profit 0.0391).

#### Tests
```python
def test_lerner_oracle():           # grid == c*e/(1+e); with VAT/fee: == (c/kappa)*e/(1+e) -> 14.0834
def test_local_rule_sign():         # sign(dΠ/dP at P0) == sign(1 - m0*|eps|) for random (eps, m0)
def test_break_even_closed_form():  # P(loss) closed form == MC within 0.005 (P=11 -> 0.1835)
def test_multiproduct_foc():        # joint optimum margins satisfy the m_i formula to 1e-6
def test_box_required():            # unbounded CE cross system -> optimiser refuses without bounds
def test_discrete_not_rounded():    # {10,15}, c=6.2, eps=-2 -> 15, not 10
def test_status_quo_in_set():       # P0 always a candidate; ΔΠ(P0) == 0
```

### Gaps
- The meta-analytic elasticity means, the Lerner DOI and the Rue La La revenue figure were not verified in-session.
- I found no elasticity evidence specific to African small businesses (search budget exhausted).

---

## 3. Risk-aware objectives (and communicating the downside)

### Takeaway
On one posterior, standard criteria recommend very different prices. In the toy below, expected profit picks 12.0, a 1-SD penalty picks 10.5, a 20% loss-probability cap picks 11.0, minimax regret picks 11.5 and robust maximin picks 10.0 (no change).

Recommended default for small businesses:
- **Objective:** maximise expected loss-averse utility relative to the status quo, with λ_LA = 2.25.
- **Constraints:** P(ΔΠ < 0) ≤ 0.20 and CVaR_0.90 of the loss ≤ 5% of baseline profit over the horizon, plus step guardrails.
- **Communication:** give the expected gain, a 1-in-10 bad case and the chance of ending up worse off than today, in local currency per week.

Loss aversion alone does **not** bound the probability of loss; the chance constraint is what does.

### Cited Findings
- **CVaR:** `F_α(x, ζ) = ζ + (1−α)⁻¹·E[(L(x,Y) − ζ)⁺]`.
  - Minimising over ζ gives CVaR_α, and the minimiser is VaR_α.
  - F is convex in (x, ζ) when the loss is convex in x, and the sampled version is a linear program.

  — [Rockafellar & Uryasev 2000, *Journal of Risk* 2(3):21–41](https://doi.org/10.21314/JOR.2000.038) [U]
- Robust optimisation maximises the worst case over an uncertainty set; box sets are conservative and ellipsoidal sets less so — [Ben-Tal & Nemirovski 1998, *Mathematics of OR* 23(4):769–805](https://doi.org/10.1287/moor.23.4.769) [U]; [Ben-Tal, El Ghaoui & Nemirovski 2009, *Robust Optimization*, Princeton UP](https://press.princeton.edu/books/hardcover/9780691143682/robust-optimization) [U]
- Cumulative prospect theory estimates loss aversion λ ≈ 2.25 and curvature α = β ≈ 0.88 from lab choices — [Tversky & Kahneman 1992, *J. Risk & Uncertainty* 5(4):297–323](https://doi.org/10.1007/BF00122574) [U]
- Minimax-regret decision criterion — [Savage 1951, JASA 46(253):55–67](https://doi.org/10.1080/01621459.1951.10500768) [U]
- "Conservative bandits" explore new actions to maximise revenue while keeping cumulative revenue above a fixed fraction of a baseline policy, uniformly over time. An action is safe at round t if cumulative reward up to t stays above that fraction of the baseline — [Wu, Shariff, Lattimore & Szepesvári 2016, ICML, PMLR 48:1254–1262](http://proceedings.mlr.press/v48/wu16.html) [S]
- High-confidence ("safe") policy improvement adopts a new policy only when a lower confidence bound on its improvement over the current policy is positive — [Thomas, Theocharous & Ghavamzadeh 2015, ICML, PMLR 37](http://proceedings.mlr.press/v37/thomas15.html) [U]
- Risk communication: natural frequencies ("2 in 10") are understood better than single-event probabilities — [Gigerenzer & Hoffrage 1995, *Psychological Review* 102(4):684–704](https://doi.org/10.1037/0033-295X.102.4.684) [U]; [Spiegelhalter 2017, *Annual Review of Statistics and Its Application* 4:31–60](https://doi.org/10.1146/annurev-statistics-010814-020148) [U]

### Inferences

#### Formulas
All are computed on horizon totals `ΔΠ_s(P)`, s = 1..S, with common random numbers.
- **Mean-SD:** `J = E[Π] − κ_sd·SD[Π]`, with κ_sd dimensionless. **Do not use λ·Var.** Rescaling money by a factor a multiplies E by a and Var by a², so the argmax changes. [C]: λ = 0.001 picks 12.0 with prices in units and 10.0 when the same prices are in cents.
- **Probability of loss:** `PL(P) = mean(ΔΠ_s < 0)`. Closed form under CE and a normal posterior: `Φ((ε_BE(P) − μ)/σ)` for increases (§2).
- **CVaR** with loss L = −ΔΠ: `CVaR_α = VaR_α + mean((L − VaR_α)⁺)/(1 − α)`, which equals the mean of the worst ⌈(1−α)S⌉ losses. [C]: at P = 12 in the toy, the Rockafellar–Uryasev minimisation and the tail mean both give **42.8367**, with ζ* = VaR = 24.1262.
- **Loss-averse expected utility:** `U(ΔΠ) = ΔΠ if ΔΠ ≥ 0, else λ_LA·ΔΠ`. Equivalently, `E[U] = E[ΔΠ] − (λ_LA − 1)·E[(−ΔΠ)⁺]`: expected profit minus a lower-partial-moment penalty measured against today.
- **Robust maximin:** `max_P min_{ε∈U} Π(P; ε)`.
  - Under CE, ∂Π/∂ε = Π·r has the sign of r. The worst case is therefore the lower interval endpoint for increases and the upper endpoint for decreases, so two evaluations suffice.
  - This is equivalent to acting only if the **whole credible interval** lies on the profitable side of ε_BE(P). With a 90% box this is often too conservative.
- **Minimax regret:** `min_P max_{ε∈U} [max_{P'} Π(P'; ε) − Π(P; ε)]`, evaluated on a grid of P × ε.
- **Chance-constrained:** `max E[Π]` subject to `PL ≤ δ`.
- **Floor on PL:** as the step goes to zero, PL tends to P(ε < −1/m0), which is **0.12** in the toy. If that floor exceeds δ, no increase is allowed at any step size. The posterior is then unsure of the *direction*, and the right action is TEST (§4), not a smaller change.

#### Worked comparison [C]
Toy: P0 = 10, c = 6, Q0 = 100/week, ε ~ N(−1.8, 0.6²), parameter uncertainty only, 400k draws, baseline profit 400/week. The 90% interval for ε is [−2.787, −0.813].

| P | E[ΔΠ] | SD(Π) | P(loss) | CVaR90 (loss) | E[U_LA], λ=2.25 | Robust (90% box) | Max regret | Plug-in Π |
|---|---|---|---|---|---|---|---|---|
| 9.5 | −15.97 | 11.80 | 0.911 | 36.27 | −36.58 | 364.91 | 152.43 | 383.85 |
| 10.0 | 0 | 0 | 0 | 0 | 0 | **400.00** | 117.33 | 400.00 |
| 10.5 | 12.35 | 12.05 | 0.152 | 8.40 | 11.21 | 392.79 | 84.84 | 412.17 |
| 11.0 | 21.87 | 24.10 | 0.183 | 18.85 | 19.11 | 383.36 | 54.62 | 421.18 |
| 11.5 | 29.18 | 35.99 | 0.212 | 30.52 | 24.35 | 372.57 | **31.22** | 427.67 |
| 12.0 | **34.75** | 47.61 | 0.240 | 42.84 | **27.45** | 360.98 | 42.81 | **432.14** |

Choices on this posterior:

| Criterion | Chosen P |
|---|---|
| Plug-in ε | 12.0 |
| E[Π] | 12.0 |
| E − 0.5·SD | 11.5 |
| E − 1·SD | 10.5 |
| E[ΔΠ] − CVaR90 | 10.5 |
| E[Π] subject to PL ≤ 0.20 | **11.0** |
| E[Π] subject to CVaR90 ≤ 10% of Π0 | 11.5 |
| Loss-averse EU (λ = 2.25) alone | 12.0, with P(loss) 24% |
| Robust maximin | 10.0 (no change) |
| Minimax regret | 11.5 |
| **Default (LA-EU + PL ≤ 0.20 + CVaR90 ≤ 5% of Π0)** | **11.0** |

What the default means at P = 11:
- expected profit **+5.47%**, quantity −15.6% and revenue **−7.2%** (revenue falls while profit rises; Rule 9);
- P(worse off than today) 18%;
- CVaR90 4.7% of weekly profit.

#### Recommended defaults
- **Objective:** E[U_LA(ΔΠ_H)] with λ_LA = 2.25, configurable between 1.5 and 3. λ_LA = 1 recovers risk-neutral expected profit.
- **Constraints:**
  - PL ≤ δ, with δ = 0.20 by default, 0.25 for identification tiers 1–2 (experiments and natural experiments) and 0.10 for tiers 6–7 (fixed effects and pure correlation; v1 §76);
  - CVaR_0.90(L) ≤ 5% of baseline profit over H;
  - step and frequency guardrails (§5).
- **Materiality:** require E[ΔΠ_H] ≥ max(1% of baseline horizon profit, a small configurable cash minimum). Otherwise HOLD.
- **Action labels:**
  - **CHANGE:** a feasible, material improvement exists.
  - **HOLD:** no feasible improving price, or the gain is immaterial.
  - **TEST:** the PL floor exceeds δ, but the value of information exceeds the test's cost (§4).
- **Recommendation-level definitions:**
  - "confidence" := P(ΔΠ(P_rec) > 0). This makes v1 Rule 12 interpretable.
  - "risk" := CVaR90 as a percentage of baseline profit.

#### Output contract (replaces v1 §67)
```json
{
  "action": "CHANGE", "current_price": 10.00, "recommended_price": 11.00, "currency": "KES",
  "horizon_days": 28,
  "expected_change_pct": {"quantity": -15.6, "revenue": -7.2, "gross_profit": 5.5},
  "interval_80_pct": {"gross_profit": [-2.2, 13.3]},
  "prob_worse_than_today": 0.18,
  "bad_case_1_in_10_profit_pct": -2.2,
  "cvar90_profit_loss_pct": 4.7,
  "elasticity": {"mean": -1.8, "ci80": [-2.57, -1.03], "identification_tier": 5},
  "break_even_elasticity": -2.34,
  "prob_increase_direction_correct": 0.88,
  "binding_constraints": ["max_step_up"], "boundary_solution": false,
  "risk_config": {"lambda_la": 2.25, "delta": 0.20, "cvar_alpha": 0.90, "cvar_budget_pct": 5},
  "lineage": {"prediction_id": "...", "data_snapshot": "sha256:...", "model": "...", "seed": "..."}
}
```
The values come from the §3 toy at P = 11. ΔΠ is monotone in ε at a fixed price, so the 80% profit range [−2.2%, +13.3%] maps directly from the ε 80% interval [−2.569, −1.031]. In production, the interval and bad-case fields must be computed from the same posterior draws as every other field.

#### Communication
For each recommendation show three lines, in local currency per week and as percentages:
- "Expected: +X per week (80% range a to b)";
- "Bad case, about 1 week in 10: −Y versus today";
- "Chance this leaves you worse off than today: about 2 in 10".

Use frequency phrasing [U]. Show quantity and revenue falling alongside profit rising, so owners are not alarmed by lower volume. Never show a point elasticity without its range (Rule 4).

#### Tests
```python
def test_cvar_ru_equals_tail():       # RU minimisation == tail mean (42.8367 at P=12 in the toy)
def test_cvar_monotone_in_alpha():    # CVaR_0.95 >= CVaR_0.90 >= E[L]
def test_robust_endpoints():          # CE: min over an eps grid == min over the two interval endpoints
def test_scale_invariance():          # money x100 -> same decision for SD-, PL-, CVaR-%-, LA-based rules
def test_mean_variance_not_invariant():  # documents why λ·Var is banned
def test_status_quo_zero_risk():      # ΔΠ(P0)=0 -> PL=0, CVaR=0, U=0
def test_regret_equivalence():        # argmin E[regret] == argmax E[Π] (guards against reintroducing v1 §48)
def test_default_toy_choice():        # toy posterior -> default picks 11.0; E[U_LA] alone picks 12.0
```

### Gaps
- No calibration of λ_LA for small-business owners (African or otherwise) was found. The 2.25 value comes from laboratory gambles [U] and should be elicited, for example with a 3-question lottery in onboarding, or tuned by tier.
- I could not verify risk-communication studies with small-business owners or in low-literacy settings.

---

## 4. Learning while pricing (bandits, Thompson sampling, value of information, test design, incomplete learning)

### Takeaway
Greedy "estimate, then price at the estimated optimum" can stop learning. In an illustrative simulation [C] it ended far from the optimum in **29%** of runs, against 10–11% for Thompson sampling and a constrained-variance policy.

Use Thompson sampling or forced price dispersion, under limits on how often prices switch and with constraints that protect the baseline. But a single small business learns very little per test. A +10% test at 20 units/day for 14 days per arm gives SE(ε) ≈ 0.92. Tests must therefore:
- be chosen by value of information: larger gaps, 2–4 weeks per arm, pooled hierarchical priors;
- be run as time-randomised switchback designs, not customer splits;
- be proposed mainly when a risk constraint is blocking a probably-profitable move, which is exactly when information is valuable [C].

### Cited Findings
- **den Boer (2015):** survey of dynamic pricing and learning — [*Surveys in OR & MS* 20(1):1–18](https://doi.org/10.1016/j.sorms.2015.03.001); [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2334429) [S]
- **Besbes & Zeevi (2009):** a single product with an initial inventory and an unknown demand function, in parametric and nonparametric versions.
  - Their policies learn the demand function "on the fly".
  - They derive regret lower bounds that hold for any admissible policy, and algorithms whose regret is close to them.
  - The size of the regret is interpreted as the economic value of prior knowledge of the demand function.

  — [*Operations Research* 57(6):1407–1420](https://pubsonline.informs.org/doi/10.1287/opre.1080.0640) [S]
- **Keskin & Zeevi (2014):** linear demand with unknown parameters.
  - The smallest achievable revenue loss over T periods is of order **√T** in "the former case" and **log T** in "the latter case".
  - Greedy iterated least squares (ILS) suffers from **incomplete learning**, while certain constrained variants achieve the minimum asymptotic loss rate.

  — [*Operations Research* 62(5):1142–1167](https://pubsonline.informs.org/doi/10.1287/opre.2014.1294) [S]. From memory [U], the two cases are: former = nothing is known about the parameters; latter = the expected demand at an incumbent price is known. The constrained-ILS rule perturbs the greedy price to `p̄ ± κ·t^(−1/4)` when it falls within that distance of the average past price [U].
- **Thompson sampling for pricing:** see [Ferreira, Simchi-Levi & Wang 2018](https://pubsonline.informs.org/doi/10.1287/opre.2018.1755) [S], summarised in §2.
- **Cheung, Simchi-Levi & Wang (2017):**
  - Setting: the demand function is unknown but belongs to a known finite set, and **at most m price changes** are allowed over T periods.
  - Result: a policy achieves regret **O(log^(m) T)** (the logarithm iterated m times), which is optimal up to a constant.

  — [*Operations Research* 65(6):1722–1731](https://www.semanticscholar.org/paper/Technical-Note-Dynamic-Pricing-and-Demand-Learning-Cheung-Simchi-Levi/7b0c9bcfb697c1b4f0d2bfaa0e528a6510c65cef) [S]
- **Related work, titles and DOIs only [S]:**
  - "Dynamic Pricing with Unknown Nonparametric Demand and Limited Price Changes", *Operations Research* — [DOI](https://pubsonline.informs.org/doi/10.1287/opre.2020.0445) (authors not confirmed);
  - "Dynamic Pricing with Demand Learning and Reference Effects", *Management Science* — [DOI](https://dl.acm.org/doi/10.1287/mnsc.2021.4234) (authors not confirmed; recalled as den Boer & Keskin [U]);
  - Qiang & Bayati, "Dynamic pricing with demand covariates" — [PDF](https://web.stanford.edu/~bayati/papers/dpdc.pdf). Its claim that greedy pricing can be near-optimal when covariates vary enough is [U].
- **Switchback experiments (Bojinov, Simchi-Levi & Zhao 2023):**
  - The firm repeatedly assigns a random treatment to the same unit over time.
  - They derive optimal designs for a given order of carryover by solving a minimax discrete optimisation.
  - Inference uses exact randomisation-based p-values or a finite-population central limit theorem.

  — [*Management Science* 69(7):3759–3777](https://pubsonline.informs.org/doi/10.1287/mnsc.2022.4583) [S]
- **Conservative (baseline-protecting) exploration** — [Wu et al. 2016](http://proceedings.mlr.press/v48/wu16.html) [S]
- **Further learning-and-earning literature [U]:**
  - controlled-variance pricing and the incomplete learning of certainty-equivalent pricing — [den Boer & Zwart 2014, *Management Science* 60(3):770–783](https://doi.org/10.1287/mnsc.2013.1788);
  - a myopic Bayesian policy can get stuck at an "uninformative" price — [Harrison, Keskin & Zeevi 2012, *Management Science* 58(3):570–586](https://doi.org/10.1287/mnsc.1110.1426);
  - meta-learning a Thompson-sampling prior across products — [Bastani, Simchi-Levi & Zhu 2022, *Management Science* 68(3):1865–1881](https://doi.org/10.1287/mnsc.2021.4071);
  - Thompson sampling tutorial — [Russo et al. 2018, *Foundations and Trends in ML* 11(1)](https://doi.org/10.1561/2200000070);
  - knowledge-gradient value of information — [Frazier, Powell & Dayanik 2008, *SIAM J. Control Optim.* 47(5):2410–2439](https://doi.org/10.1137/070693424);
  - always-valid inference for continuously monitored A/B tests — [Johari et al. 2022, *Operations Research* 70(3):1806–1821](https://doi.org/10.1287/opre.2021.2135);
  - bandit pricing that exploits economic structure — [Misra, Schwartz & Abernethy 2019, *Marketing Science* 38(2):226–252](https://doi.org/10.1287/mksc.2018.1129).

### Inferences

#### Incomplete learning, demonstrated [C]
Setup:
- demand D = 100 − 8P + N(0, 10²), c = 4, so P* = 8.25 and optimal profit is 144.5 per period;
- a deliberately wrong normal prior (mean (80, −4), SD (50, 5));
- a history of only two nearby prices (7.0 and 7.4), as for a business that has barely varied its price;
- T = 1,000 periods, 300 runs, price bounds [5, 12].

| Policy | Mean regret (% of optimal profit) | P(\|P_T − P*\| > 0.5) | p90 cumulative regret | Price variance, first 100 periods |
|---|---|---|---|---|
| Greedy (posterior mean, certainty-equivalent) | 1.20% | **0.287** | 4,642 | 0.083 |
| Constrained ILS (κ = 1) | 0.94% | 0.113 | 2,786 | 0.241 |
| Thompson sampling | 1.14% | 0.100 | 2,748 | 0.481 |

Both remedies cut the stuck rate by about two-thirds and the p90 cumulative regret by about 40%. The numbers are illustrative only.

#### Recommended policy: "safe batched Thompson sampling with a dispersion guard"
```text
per product, per decision epoch (weekly):
  if days_since_last_change < 14 or changes_this_quarter >= 4: keep price
  θ~ ~ posterior(θ)                              # Poisson-GLM Laplace/conjugate approx, hierarchical prior
  P_ts = argmax_{P ∈ F} Π(P; θ~)                  # F = ladder ∩ guardrails ∪ {P0}  (§5)
  # conservative filter (full posterior, horizon H, common random numbers):
  if PL(P_ts) > δ_c or CVaR90(P_ts) > budget or
     P(cum profit since start < (1−α_c)·cum baseline) > δ_c:      # Wu et al.-style baseline protection
       P_ts = best feasible P passing the filter (P0 always passes)
  # dispersion guard (anti incomplete learning):
  if posterior_sd(ε) > σ_target and Σ_{s≤t}(P_s − P̄)²/P̄² < κ_d·√t: propose TEST (VOI-ranked)
  post P_ts; log price, reason code, randomisation flag; update posterior daily
```
Defaults:

| Parameter | Value |
|---|---|
| α_c | 0.05 (cumulative profit never more than 5% below the status quo) |
| δ_c | 0.10 |
| σ_target | 0.25 |
| κ_d | 0.05 (relative-price dispersion) |
| Epoch | Weekly |

Use the hierarchical (multi-tenant) posterior as the Thompson-sampling prior, as in Bastani et al.'s meta-learned prior [U].

Keskin & Zeevi's log T case [S, case definitions U] has a practical reading. A small business has abundant data at its current price, so demand at P0 is nearly known and mainly the slope must be learned. Two-point designs around P0 are therefore efficient.

#### Value of information for choosing test changes
- **Expected value of perfect information:** `EVPI = E_θ[max_P Π(P;θ)] − max_P E_θ[Π(P;θ)]`.
- **Expected value of sample information:** `EVSI(design) = E_{data}[ max_P E[Π(P)|data] ] − max_P E[Π(P)]`, evaluated by preposterior simulation.
- **Conjugate update** for an ε-estimate from a two-arm test:

  ```text
  se²  = (1/N_A + 1/N_B)·φ / r²
  s1²  = 1/(1/s0² + 1/se²)
  m1   = s1²·(m0/s0² + ε̂/se²)
  ```

  Here N = units sold per arm and φ = overdispersion. The decision rule applied after the test must be the production rule, including risk constraints. Run a test iff `EVSI × post-test horizon − expected test-phase cost > 0`. Frazier et al.'s knowledge gradient [U] is the one-step version.
- **Results [C]:** prior N(−1.8, 0.6²), P0 = 10, c = 6, decision used for 26 weeks after the test. Net values are given as a percentage of baseline profit over the whole test-plus-26-week window. Posterior SDs are rounded medians.

  | Volume | Production rule | Decision without test | EVPI (% of weekly profit) | Test 11 vs 10 (2 / 4 / 8 wk per arm) | Test 12 vs 10 (2 / 4 / 8 wk per arm) |
  |---|---|---|---|---|---|
  | 100 units/wk | Risk-neutral | 12.0 | 1.74% | −0.68 / −1.15 / −1.91% | −0.08 / −0.29 / −0.93% |
  | 100 units/wk | Chance-constrained (δ = 0.2) | 11.0 | 4.95% | **+1.57 / +1.62 / +1.16%** | **+2.81 / +2.86 / +2.34%** |
  | 700 units/wk | Risk-neutral | 12.0 | 1.74% | +0.05 / −0.47 / −1.45% | +0.63 / +0.13 / −0.70% |
  | 700 units/wk | Chance-constrained | 11.0 | 4.95% | **+3.16 / +2.80 / +1.88%** | **+3.90 / +3.46 / +2.64%** |

  | Volume | Test design | Posterior SD after test (from 0.60) |
  |---|---|---|
  | 100 units/wk | 11 vs 10, 2 / 4 / 8 wk per arm | 0.53 / 0.48 / 0.41 |
  | 100 units/wk | 12 vs 10, 2 / 4 / 8 wk per arm | 0.42 / 0.35 / 0.27 |
  | 100 units/wk | 10.5 vs 10, 2 wk per arm (hand calculation) | 0.577 |
  | 700 units/wk | 12 vs 10, 2 wk per arm | 0.21 |

  The probability that the test phase itself earns less than holding the price was 0.18 for the 11 vs 10 test and 0.23 for 12 vs 10.
- **Lessons:**
  1. Information is worth most when a risk constraint is binding.
  2. Larger gaps beat smaller ones.
  3. Two to four weeks per arm beat eight.
  4. For a risk-neutral optimiser that already jumps to the boundary, testing looks worthless, which is misleading.

#### Sample size and duration
- **Estimator:** `ε̂ = [ln(Q_B/n_B) − ln(Q_A/n_A)] / ln(P_B/P_A)`, with `SE ≈ √(φ·(1/N_A + 1/N_B)) / |ln(P_B/P_A)|`. For negative-binomial daily sales, φ = 1 + μ_day/k.
- **Verified [C]:** 20 units/day, 14 days per arm, +10% gap, ε = −1.5. The formula gives 0.920 and simulation 0.921. With NB(k = 5), φ = 5, the SE becomes about 2.0.
- **Units needed in total** (balanced arms, Poisson, best case), `N = 4/(SE²·r²)` [C]:

  | Price gap | SE 0.5 | SE 0.3 | SE 0.2 |
  |---|---|---|---|
  | +5% | 6,721 | 18,670 | 42,008 |
  | +10% | 1,761 | 4,893 | 11,008 |
  | +15% | 819 | 2,275 | 5,119 |
  | +20% | 481 | 1,337 | 3,008 |

- **Unbalanced allocation** inflates variance by `(1/w + 1/(1−w))/4` relative to 50/50 [C]: 10/90 gives 2.78×, 20/80 gives 1.56× and 30/70 gives 1.19×. Use 50/50 in time and control exposure through product selection and duration instead.
- **Design and analysis:**
  - Use a switchback design [S] with day-level blocks when carryover is negligible (cafés, prepared food). Use ABBA week blocks when stockpiling or reference effects are likely (staples).
  - Balance arms within each weekday, and run at least 2 full weekly cycles per arm.
  - Drop the first day after each switch if carryover is suspected. Update shelf and menu prices before switching.
  - Analyse with a Poisson GLM including block and weekday effects, and report randomisation p-values [S].
  - Stop only at a fixed horizon or at pre-registered checkpoints with a Bayesian decision rule or always-valid inference [U]. No unplanned peeking.
- **Exposure cap:** concurrent tests on products together covering ≤ 20% of revenue, with P(test-phase loss) ≤ 0.30 per test.

#### Constraints on change size and frequency (defaults)
- |ΔP/P0| ≤ 10% per change, or 5% for staples and high-visibility items.
- At least 14 days between changes to the same product, and at most 4 changes per quarter.
- No increases on items flagged by the news/shock engine as under a demand surge unless the increase is cost-justified (§5).
- Theory supports learning with few changes: Cheung et al. [S] give log^(m) T regret with m changes.

#### Tests
```python
def test_se_formula_vs_sim():          # 20/day,14d,+10%,eps=-1.5: formula 0.920 vs Poisson sim within 2%
def test_allocation_factor():          # w=0.1 -> 2.778
def test_conjugate_update():           # posterior mean/var == closed form; s1 < s0 always
def test_evsi_free_info_nonneg():      # zero-cost test -> EVSI >= 0 for the Bayes-optimal rule (MC tolerance)
def test_dispersion_guard_triggers():  # constant-price history with posterior_sd > σ_target -> TEST proposed
def test_conservative_filter():        # simulated cumulative profit >= (1-α_c)·baseline with prob >= 1-δ_c
def test_switchback_balance():         # each arm gets equal counts per weekday; no two prices at once in a store
def test_greedy_can_stick():           # regression test: greedy stuck-rate > TS stuck-rate on the §4 setup
```

### Gaps
- Not verified in-session:
  - the exact rates in Besbes & Zeevi (2009). From memory, the upper bounds are about n^(−1/4) (nonparametric) and n^(−1/3) (parametric), up to log factors, against a lower bound of order n^(−1/2), in relative-regret terms; verify before citing;
  - the definitions of Keskin & Zeevi's two cases;
  - the theorem statements of den Boer & Zwart, Harrison et al. and Bastani et al.
- No evidence was found on whether African small-business owners will accept price experimentation, or on customer reactions to frequent price switching in informal markets.
- The VOI and simulation results use simple CE/Poisson or linear-Gaussian worlds. They need re-running on the hierarchical model once it exists.

---

## 5. Business and behavioural constraints (price points, margin, competitors, fairness, law)

### Takeaway
Build constraints into a **discrete feasible set before optimising**; never round a continuous optimum afterwards. The set combines:
- the business's cash-convenient price ladder;
- a margin-on-price floor;
- step and frequency limits;
- soft, staleness-checked competitor-relative penalties;
- a fairness rule: cost-justified increases are acceptable, but exploiting a demand surge is not (82% judged a post-snowstorm shovel price rise unfair);
- a legal and ethical default of one posted price per product, store and moment, with no individualised prices and tests run in time rather than across customers.

### Cited Findings
- **Kahneman, Knetsch & Thaler (1986):** when a hardware store raised snow-shovel prices from $15 to $20 the morning after a snowstorm, about **82%** of respondents called it unfair or very unfair. Raising prices when profits are threatened (for example by cost increases) is acceptable, and so is keeping prices when costs fall; exploiting shifts in demand by raising prices is judged unfair — [*American Economic Review* 76(4):728–741](https://eml.berkeley.edu/~saez/course131/Kahneman-FairnessConstraintProfit-1986.pdf); [EconPapers](https://econpapers.repec.org/RePEc:aea:aecrev:v:76:y:1986:i:4:p:728-41) [S]. The "dual entitlement" framing (reference transaction and reference profit), and the finding that removing a discount is judged more acceptable than an equivalent price increase, come from the same paper [U].
- Rue La La's tool-recommended increases on medium and high price-point products did not reduce sell-through — [Harvard D3](https://d3.harvard.edu/analytics-online-retailer-demand-forecasting-price-optimization/) [S]
- **Psychological price points [U]:**
  - $9 price endings raised demand in field experiments — [Anderson & Simester 2003, *Quantitative Marketing and Economics* 1(1):93–110](https://doi.org/10.1023/A:1023581927405);
  - left-digit effects in price cognition — [Thomas & Morwitz 2005, *J. Consumer Research* 32(1):54–64](https://doi.org/10.1086/429600);
  - "convenient" round prices, which reduce the coins and change needed, are stickier — [Knotek 2011, REStat 93(3):1076–1086](https://doi.org/10.1162/REST_a_00104).
- Reference-price research finds consumers react more to increases (perceived as losses) than to equivalent decreases — [Kalyanaram & Winer 1995, *Marketing Science* 14(3):G161–G169](https://doi.org/10.1287/mksc.14.3.G161) [U]
- **Legal anchors outside Africa [U]:**
  - US Robinson–Patman Act, price discrimination in sales of goods between business purchasers — [15 U.S.C. §13](https://www.law.cornell.edu/uscode/text/15/13);
  - EU Omnibus Directive (EU) 2019/2161, which requires disclosure when a price is personalised by automated decision-making — [EUR-Lex](https://eur-lex.europa.eu/eli/dir/2019/2161/oj).

### Inferences

#### Feasible-set pipeline (order matters; output always includes P0 and a status)
1. **Price ladder L.** Use the business's declared price points. Otherwise learn them from its own history: the mode of trailing digits and the step between used prices.
   - In cash-heavy markets, default to "convenient" multiples of the smallest commonly circulating denomination.
   - Do **not** default to x.99 endings, which are US evidence [U], unless the business already uses them or takes card or mobile-money payments at exact amounts.
2. **Margin floor (margin on net price, not markup).** `P ≥ c/(κ·(1 − m_min))`, default m_min = 0.10.
   - Test vector: c = 6, κ = 1 gives a floor of **6.667**. v1's `cost*1.10` gives 6.6, which is a 9.09% margin.
3. **Step guardrail.** `P0·(1 − d_down) ≤ P ≤ P0·(1 + d_up)`, defaults d_up = d_down = 0.10. Use 0.05 for staples and for identification tiers ≥ 6.
4. **Frequency.** At least 14 days since the last change; at most 4 changes per quarter.
5. **Competitor-relative (soft).** `J ← J − w·max(0, P/CPI_asof − (1 + δ_cp))·Π0_H`, with δ_cp = 0.10 and w = 1.
   - Ignore the competitor price index if its newest underlying price is more than 30 days old.
   - A hard cap applies only if the owner opts in.
6. **Fairness.**
   - If the shock or event index shows a demand surge (shock_index ≥ +0.5 SD) and the cost index rose < 2%, block increases or require explicit owner override with a warning.
   - Allow cost pass-through increases up to Δc·(1 + current markup).
   - Prefer framing an increase as removing a discount where a discount exists [U].
7. **Legal and ethical.**
   - One price per product, store and moment by default; no individualised or person-targeted prices.
   - Tests run by time (switchback) or by store or channel, only where the law allows.
   - Respect statutory or regulated maximum prices.
   - The displayed price must equal the charged price, so labels change before the switch.
8. **Inventory.** Evaluate with `Q_sold = min(D, stock_H)`. Raising prices when stock is short can be profit-maximising but is fairness-sensitive, so apply rule 6.
9. **Infeasibility.** If the set is empty (e.g. the margin floor exceeds the step cap), return `status = INFEASIBLE` with the binding constraints. Never silently relax a constraint.

#### Psychological price points: how to model them
- If at least 2 distinct endings have each been used for ≥ 28 days, add ending dummies, or a "crossed denomination threshold" dummy, to the demand model. Otherwise treat endings only as feasibility constraints.
- Crossing a denomination boundary can produce a discontinuous drop in demand that constant elasticity cannot represent.

#### Tests
```python
def test_margin_floor_is_margin():        # c=6, m_min=0.10 -> P_min == 6.6667 (not 6.6)
def test_feasible_contains_p0():
def test_infeasible_reported():           # floor > step cap -> INFEASIBLE + binding list
def test_no_post_rounding_violation():    # every recommended price ∈ ladder and satisfies every constraint
def test_fairness_block():                # surge + flat cost -> no increase recommended
def test_cost_passthrough_allowed():      # cost +10% -> increase up to Δc·(1+markup) allowed
def test_competitor_staleness():          # CPI older than 30 days -> penalty ignored
def test_single_posted_price():           # experiment designer never assigns 2 prices to one product/store/time
```

### Gaps
- **African cash conventions not verified:** denominations in circulation, rounding norms at the till, and how mobile-money exact payments change them, in Kenya, Nigeria, Ghana and South Africa. This needs central-bank sources and merchant interviews.
- **African law not verified:**
  - South Africa: the Competition Act 89 of 1998 (s.9, price discrimination by dominant firms) and the Consumer Protection Act 68 of 2008 (s.23, displayed-price rules, including whether the lower of two displayed prices binds);
  - Kenya: the Competition Act 2010 and any essential-goods price-control law;
  - Nigeria: the Federal Competition and Consumer Protection Act 2018.

  Counsel review per country is required before experimentation is enabled.
- No evidence was found on fairness perceptions of price increases in African retail.
- The numeric thresholds in rules 5–6 (0.5 SD surge, 2% cost change, 30-day staleness) are engineering defaults without empirical calibration.

---

## 6. Evaluation and backtesting (forecasts, elasticities, recommendations)

### Takeaway
Evaluate three objects separately, always with rolling-origin, as-of backtests:
1. **Demand forecasts.**
   - Primary metrics must be optimised by the mean, because profit is linear in mean demand: scaled squared error (RMSSE) or Poisson deviance.
   - Secondary: MASE and WAPE.
   - Probabilistic: pinball loss, interval score, coverage and the randomised PIT for count data.
2. **Elasticities, only where ground truth exists:**
   - synthetic data with a known ε;
   - randomised tests;
   - counterfactual event studies on historical price changes, scored with uncertainty-aware z-scores and with "decision-direction accuracy" relative to the break-even elasticity rather than raw sign accuracy.
3. **Recommendations,** with observable regret proxies:
   - realised uplift against a randomised status-quo holdout;
   - calibration of the predicted uplift intervals;
   - oracle regret, measured only in a simulator suite.

### Cited Findings
- Interval estimation (Gneiting & Raftery 2007, JASA 102:359–378): the interval score is proposed as a utility function that rewards narrow intervals and penalises misses; it traces to Winkler (1972) and is their eq. (43). Proper and strictly proper scoring rules are defined in the same paper — [PDF](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf) [S]. A reference implementation exists in the R package scoringutils — [interval_score](https://epiforecasts.io/scoringutils/reference/interval_score.html) [S]
- **Forecast-evaluation literature [U]:**
  - Rolling-origin out-of-sample evaluation: multiple origins and horizons, fixed vs rolling origins, recalibrating at each origin — [Tashman 2000, IJF 16(4):437–450](https://doi.org/10.1016/S0169-2070(00)00065-0).
  - MASE scales errors by the in-sample MAE of the naive (or seasonal-naive) forecast. MAPE is infinite or undefined at zero actuals, and sMAPE is asymmetric and unstable — [Hyndman & Koehler 2006, IJF 22(4):679–688](https://doi.org/10.1016/j.ijforecast.2006.03.001).
  - Rolling-origin time-series cross-validation in practice — [Hyndman & Athanasopoulos, *Forecasting: Principles and Practice* (3rd ed.)](https://otexts.com/fpp3/tscv.html).
  - Squared error is consistent for the mean and absolute error for the median, so a metric fixes which point forecast is "best" — [Gneiting 2011, JASA 106(494):746–762](https://doi.org/10.1198/jasa.2011.r10138).
- **Count-forecast evaluation [U]:**
  - randomised PIT for count data — [Czado, Gneiting & Held 2009, *Biometrics* 65(4):1254–1261](https://doi.org/10.1111/j.1541-0420.2009.01191.x);
  - retail count forecasts — [Kolassa 2016, IJF 32(3):788–803](https://doi.org/10.1016/j.ijforecast.2015.12.004);
  - the M5 competition (Walmart, intermittent series, RMSSE/WRMSSE) — [Makridakis, Spiliotis & Assimakopoulos 2022, IJF 38(4):1346–1364](https://doi.org/10.1016/j.ijforecast.2021.11.013).
- **Comparing forecasters [U]:**
  - test of equal predictive accuracy — [Diebold & Mariano 1995, JBES 13(3):253–263](https://doi.org/10.1080/07350015.1995.10524599);
  - small-sample correction — [Harvey, Leybourne & Newbold 1997, IJF 13(2):281–291](https://doi.org/10.1016/S0169-2070(96)00719-4).
- **Counterfactual and off-policy evaluation [U]:**
  - synthetic control for constructing counterfactual baselines — [Abadie, Diamond & Hainmueller 2010, JASA 105(490):493–505](https://doi.org/10.1198/jasa.2009.ap08746);
  - doubly robust off-policy evaluation, which needs logged randomised actions — [Dudík, Langford & Li 2011](https://arxiv.org/abs/1103.4601).
- Regret in the pricing literature is measured against a clairvoyant who knows the demand function — [Besbes & Zeevi 2009](https://pubsonline.informs.org/doi/10.1287/opre.1080.0640) [S]. It is computable only where the truth is known, i.e. in simulation.
- Switchback designs give randomisation-based inference for policy comparisons over time — [Bojinov et al. 2023](https://pubsonline.informs.org/doi/10.1287/mnsc.2022.4583) [S]

### Inferences

#### Rolling-origin protocol (replaces v1 §46–47)
- **Origins:** weekly, starting at the first date with ≥ 8 weeks of training history (≥ 52 weeks if annual seasonality is modelled) and ending at T − H.
- **Horizons:** 1, 7 and 28 days, plus the decision-horizon total.
- **Windows:**
  - expanding window by default; a 365-day sliding variant as a robustness check;
  - refit cadence equal to production cadence;
  - nested tuning: hyperparameters chosen only on data before each origin.
- **Embargo:** a gap equal to the label window between the last training label and the origin. For example, 28 days for 28-day sales labels.
- **As-of everything:** all features use as-of joins (§7), and the hierarchical/population prior is refit from **other tenants' data up to the origin only**.
- **Minimum before claiming performance:** ≥ 12 origins and ≥ 26 weeks of out-of-sample predictions per product family.
- **Reporting:**
  - per series and pooled, weighted by revenue;
  - skill against naive, seasonal-naive (weekday) and 4-week moving-average baselines;
  - Diebold–Mariano test with the Harvey–Leybourne–Newbold correction on loss differentials [U];
  - for pricing, compare against the "status-quo" and "CE plug-in" policies.

#### Metrics
- **Primary point metrics (mean-consistent):**
  - `RMSSE = sqrt( mean_test (y−ŷ)² / mean_train (y_t − y_{t−1})² )`;
  - Poisson deviance `2·Σ[y·ln(y/ŷ) − (y − ŷ)]`;
  - bias = Σ(ŷ − y)/Σy. Require |bias| ≤ 5%, because profit is linear in mean demand.
- **Secondary point metrics:**
  - `MASE = MAE_test / mean_train|y_t − y_{t−m}|`, with m = 7 for daily data;
  - `WAPE = Σ|y − ŷ| / Σy`.
- **Drop MAPE and sMAPE.** [C]: on Poisson(0.6) sales:

  | Forecast | MAE | WAPE | MSE | Poisson deviance | MAPE | sMAPE |
  |---|---|---|---|---|---|---|
  | Zero (the median) | **0.594** | **1.000** | 0.945 | ∞ (32.1 with ŷ floored at 1e-12) | undefined | 200% |
  | True mean | 0.657 | 1.105 | **0.592** | **1.05** | ∞ | 140% |

  MAE and WAPE prefer the median forecast of 0, which would make every price look unprofitable. MSE and deviance prefer the mean.
- **Probabilistic metrics:**
  - pinball loss `QS_τ(q, y) = max(τ(y − q), (τ − 1)(y − q))`;
  - interval score `IS_α(l, u; y) = (u − l) + (2/α)(l − y)·1{y < l} + (2/α)(y − u)·1{y > u}`;
  - [C] verified identity: `IS_α = (2/α)·[QS_{α/2}(l, y) + QS_{1−α/2}(u, y)]`;
  - [C] the expected interval score is minimised at the true quantiles (1.280 vs 1.2816 for N(0,1), α = 0.2);
  - empirical coverage at 50/80/95%. With 200 intervals, the 95% sampling band around 80% is ±5.5 percentage points [C];
  - randomised-PIT histograms for counts [U].
- **Elasticity metrics:**
  - On synthetic data with confounded prices and a known ε: bias, RMSE and 80/95% interval coverage.
  - On experiments: `z = (ε̂_model − ε̂_exp)/√(se_model² + se_exp²)` should be approximately N(0,1).
  - "Sign accuracy" is nearly vacuous because almost every ε is negative. Use **decision-direction accuracy** instead: `1{sign(ε̂ − ε_BE) = sign(ε_ref − ε_BE)}`, scored only on decisive references where |ε_ref − ε_BE|/se_ref ≥ 2.

#### Counterfactual validation on historical price changes (replaces v1 §49)
1. **Event definition:** a price change of at least 3% that persists ≥ 14 days, with no concurrent promotion or stockout flag (v1 §62–63) and a logged reason code (cost, competitor, demand, other).
2. **Model:** train as of the day before the change.
3. **Predicted effect:** `Δ̂ = ln Q̂_post(P_post) − ln Q̂_post(P_pre)`, where Q̂_post(P_pre) is the counterfactual baseline with no change.
4. **Realised effect:** `Δ = ln Q_post − ln Q̂_post(P_pre)`. Use a difference-in-differences or synthetic-control baseline from similar products or stores where available [U].
5. **Uncertainty:** `Var(Δ) ≈ φ(1/N_post + 1/N_pre) + Var(baseline)`.
6. **Scoring:** z-score coverage, MAE of (Δ̂ − Δ), decision-direction accuracy, and a log-arc elasticity per event.
7. **Placebo test:** run the same pipeline on random dates with no price change. Estimated "effects" must centre on 0 with nominal coverage, otherwise the baseline model is biased.
8. **Stratify by reason code.** Owner-initiated changes are endogenous, e.g. demand-driven increases; treat them as tier 3–6 evidence, not tier 1.

#### Observable definitions of pricing regret (replace v1 §48)
1. **Oracle regret in a simulator suite (release gate):** `R = Π(P*_true) − Π(P_rec)` over synthetic markets covering:
   - CE, linear, logit and log-quadratic demand;
   - confounded price histories;
   - intermittent counts;
   - shocks, stockouts and single-price histories.

   The pipeline must beat both the status-quo and CE plug-in policies on mean **and** p90 regret.
2. **Field regret against the status quo:** randomise "follow recommendation" against "hold" by switchback blocks or random product holdouts (≥ 10% of eligible products). Then `Û = mean Π_rec − mean Π_hold`, with a confidence interval from randomisation inference [S]. Realised regret against the status quo is −Û.
3. **Uplift calibration:** the share of baseline-adjusted realised ΔΠ inside the predicted 80% interval (target 0.70–0.90), and an overclaim ratio `Σ predicted ΔΠ / Σ realised ΔΠ` (target 0.8–1.25).
4. **Ex-post posterior regret** (model-based, computable): `E_post[max_P Π(P) − Π(P_rec)]` under the posterior updated with the outcomes. It is a diagnostic only, not ground truth.
5. **Off-policy value** of alternative rules is available **only** for periods with logged randomised prices, using IPS or doubly robust estimators [U].

#### Minimum backtest windows (defaults)

| Purpose | Minimum |
|---|---|
| Own-data elasticity training | ≥ 8 weeks of daily data and ≥ 2 distinct price levels held ≥ 14 days each (≥ 52 weeks if annual seasonality is modelled) |
| Forecast evaluation | ≥ 12 weekly origins |
| Interval calibration | ≥ 100 interval evaluations per product family (binomial SE 4 points at 80%) [C] |
| Decision calibration | ≥ 30 pooled pricing events before reporting uplift calibration |

These are engineering defaults, not values from the literature.

#### Tests
```python
def test_fold_no_future():          # every fold: max(train.available_at) < origin; label windows end < origin - embargo
def test_prior_refit_per_origin():  # population prior for origin t uses only other tenants' rows with available_at <= t
def test_mase_vector():             # train [3,0,2,5,1,0,4,2], test [2,3,0,1], yhat=2 -> MASE 0.3684 (scale 19/7)
def test_interval_score_vectors():  # IS(8,12;10,.2)=4; IS(8,12;13,.2)=14; IS(8,12;7,.2)=14
def test_pinball_vectors():         # QS_.9(10;13)=2.7; QS_.9(10;7)=0.3
def test_is_pinball_identity():
def test_metric_prefers_mean():     # Poisson(0.6): deviance & MSE rank mean forecast above zero forecast
def test_synthetic_recovery():      # 500 synthetic confounded datasets: 95% CI coverage in [0.92, 0.98]
def test_placebo_events():          # placebo 'effects' mean ~0, 80% coverage in [0.72, 0.88]
def test_regret_gate():             # simulator suite: pipeline mean & p90 regret < status quo and CE plug-in
```

### Gaps
- The content of Tashman (2000) and Hyndman & Koehler (2006), and the M5 metric details, were not re-verified (blocked).
- I found no published accuracy benchmarks for forecasting small-business products in African markets.
- The minimum windows and targets above are engineering defaults that should be revisited after the first pilot.

---

## 7. Leakage prevention, drift monitoring and reproducibility

### Takeaway
- **Leakage:** make every fact **bitemporal**, recording when it was true (valid time) and when the system knew it (available_at). Use backward as-of joins. Freeze LLM-extracted features at extraction time, because a newer LLM leaks the future through pretraining. Refit population priors per backtest origin.
- **Drift:** replace fixed PSI thresholds. Under no drift, PSI ≈ (1/n + 1/m)·χ²_{B−1}, so with 100 observations per window PSI exceeds 0.1 in 85% of windows. Monitor errors with two-sided CUSUM (k = 0.5, h = 5, in-control ARL ≈ 465 periods) and control false discoveries across features and products.
- **Reproducibility:** store a full lineage tuple per prediction, so that any recommendation can be replayed exactly.

### Cited Findings
- **Point-in-time joins (Feast):** `get_historical_features` joins each entity row (key + event_timestamp) to the latest feature row whose timestamp is ≤ the entity timestamp, within the feature view's TTL — [Feast docs](https://docs.feast.dev/getting-started/concepts/point-in-time-joins) [U; fetch blocked]
- **Look-ahead bias in LLM features:** an LLM can carry knowledge of later outcomes into historical backtests; anonymising entities is proposed as a check — [Glasserman & Lin 2023](https://arxiv.org/abs/2309.17322) [U; fetch blocked]
- **General leakage:**
  - purged and embargoed cross-validation for overlapping label windows — [López de Prado 2018, *Advances in Financial Machine Learning*, Wiley](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086) [U];
  - leakage taxonomy and detection — [Kaufman, Rosset, Perlich & Stitelman 2012, ACM TKDD 6(4)](https://doi.org/10.1145/2382577.2382579) [U];
  - leakage as a cause of irreproducible machine-learning results — [Kapoor & Narayanan 2023, *Patterns* 4(9)](https://doi.org/10.1016/j.patter.2023.100804) [U].
- **Drift detection:**
  - CUSUM — [Page 1954, *Biometrika* 41(1/2):100–115](https://doi.org/10.1093/biomet/41.1-2.100) [U];
  - concept-drift survey — [Gama et al. 2014, ACM Computing Surveys 46(4)](https://doi.org/10.1145/2523813) [U];
  - false-discovery-rate control — [Benjamini & Hochberg 1995, JRSS-B 57(1):289–300](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x) [U];
  - the 0.1/0.25 PSI rules of thumb come from credit-scoring practice — [Siddiqi 2006, *Credit Risk Scorecards*, Wiley](https://www.wiley.com/en-us/Credit+Risk+Scorecards%3A+Developing+and+Implementing+Intelligent+Credit+Scoring-p-9780471754510) [U].
- **Feedback loops and ML technical debt:**
  - predictions that change the data they are later evaluated on ("performative prediction") — [Perdomo et al. 2020, ICML, PMLR 119](http://proceedings.mlr.press/v119/perdomo20a.html) [U];
  - hidden technical debt in ML systems — [Sculley et al. 2015, NeurIPS](https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems) [U].

### Inferences

#### Bitemporal data contract (extends v1 §74–75)
- **Fields on every fact:**
  - entity keys;
  - `effective_from` and `effective_to`: valid time, when it was true in the world;
  - `observed_at`: when it was measured;
  - `available_at`: knowledge time, when it landed in our store;
  - `source`, `revision_id`.
- **Never update in place.** Append revision rows, using system-versioned or range-typed tables.
- **As-of read:** the latest revision with `available_at ≤ t` whose valid interval covers the target time.
- **Sales:**
  - `available_at` = the POS sync time. Offline-first POS devices sync late.
  - A day's aggregate is available only after a "day close" cutoff, for example local d+1 06:00.
  - Store UTC and compute days in the business's local timezone.
- **Competitor prices:** `available_at` = collection time, never the claimed price date. Backfills get the backfill time.
- **News and LLM features:**
  - `available_at = max(published, ingested, extracted)`.
  - Store LLM outputs as data, with `llm_model_id`, `prompt_hash` and `extraction_run_id`.
  - In backtests, never re-extract history with a newer LLM whose training cutoff is after the backtest period (look-ahead [U]). Test with anonymised entity names.
- **Demographics and macro data:** use the vintage (release) available at t, not the latest revised series.
- **Market definitions and competitor entries (§51):** versioned with `available_at`. A competitor discovered in 2026 that opened in 2024 appears in backtests only from its discovery date. Any "oracle" analysis that uses the open date must be labelled as such.

#### Training-time leakage rules
- Fit scalers, imputers, encoders and feature selection inside each fold.
- DML cross-fitting folds must be time-blocked in backtests.
- Refit the population/hierarchical prior per origin from data available at the origin.
- Apply an embargo equal to the label window.
- No features computed with future information, for example stockout flags derived from later inventory counts or promotions planned after t.

#### Leakage tests (as-of join verified in pandas 3.0.6 [C])
```python
# pd.merge_asof(events, feats, left_on="prediction_time", right_on="available_at",
#               by="product_id", direction="backward", allow_exact_matches=False,
#               tolerance=pd.Timedelta(days=30))       # both frames sorted on their keys
def test_future_canary():   # inject feature with available_at = t + 1 day and value 99.0 -> never appears at t
def test_asof_bound():      # assert (joined.available_at <= joined.prediction_time).all() (NaT allowed)
def test_strict_same_day(): # same-timestamp feature excluded when allow_exact_matches=False
                            # [C]: inclusive join picked 10.5 at 2026-09-10; strict picked 10.0
def test_time_shift():      # shifting features toward the future must NOT improve backtest skill
def test_label_shift():     # training on labels shifted by -h collapses skill to baseline
```

#### Drift monitoring
- **PSI.**
  - Formula: `PSI = Σ (a_i − e_i)·ln(a_i/e_i) = KL(a‖e) + KL(e‖a)`, verified [C].
  - Bins are reference-window deciles (B = 10), or B = 5 when n < 200; floor proportions at 1e-4.
  - Test vector [C]: a = (.2, .3, .5), e = (.25, .25, .5) gives **0.020273**.
  - Behaviour under **no drift** [C], with 10 decile bins:

    | n = m per window | Mean PSI | (2/n)(B−1) | q95 PSI | (2/n)χ²_{.95,9} | P(PSI > 0.1) | P(PSI > 0.25) |
    |---|---|---|---|---|---|---|
    | 50 | 0.494 | 0.360 | 1.143 | 0.677 | 0.98 | 0.76 |
    | 100 | 0.193 | 0.180 | 0.369 | 0.338 | **0.85** | **0.22** |
    | 200 | 0.092 | 0.090 | 0.171 | 0.169 | 0.37 | 0.004 |
    | 500 | 0.036 | 0.036 | 0.068 | 0.068 | 0.004 | 0 |
    | 1000 | 0.018 | 0.018 | 0.034 | 0.034 | 0 | 0 |

  - **Default thresholds:**
    - WARN when `PSI > max(0.10, (1/n + 1/m)·χ²_{0.99, B−1})`;
    - ALARM when `PSI > max(0.25, (1/n + 1/m)·χ²_{0.999, B−1})`;
    - for B = 10 the statistical part is 21.666 or 27.877 times (1/n + 1/m) [C];
    - for n < 100 use a permutation test.
    - Apply Benjamini–Hochberg control across features and products [U].
- **KS test.** It has low power at small-business sample sizes. Power to detect a 0.5 SD shift at α = 0.05 [C]:

  | n = m | 20 | 30 | 50 | 100 | 300 |
  |---|---|---|---|---|---|
  | Power | 0.21 | 0.31 | 0.53 | 0.83 | 0.999 |

  At large n, KS flags trivial shifts. Report the D statistic as an effect size and alarm only if D > 0.15 **and** q < 0.05.
- **CUSUM on standardised forecast errors.**
  - Inputs: `z_t = (y_t − ŷ_t)/σ̂_t`, or normal scores of the randomised PIT for counts.
  - Updates: `S⁺_t = max(0, S⁺_{t−1} + z_t − k)` and `S⁻_t = max(0, S⁻_{t−1} − z_t − k)`. Alarm when either exceeds h, then reset.
  - Run lengths, simulated [C] vs Siegmund approximation:

    | h | In-control ARL (sim / approx) | ARL for a 1 SD shift (sim / approx) |
    |---|---|---|
    | 4 | 167.8 / 169.0 | 8.4 / 8.3 |
    | 5 (default) | 465.4 / 469.1 | 10.4 / 10.3 |

  - Siegmund approximation: one-sided `ARL ≈ (e^{−2Δb} + 2Δb − 1)/(2Δ²)` with Δ = shift − k and b = h + 1.166. Two-sided: `1/ARL = 1/ARL⁺ + 1/ARL⁻`.
  - Implication: with daily checks per product, an in-control ARL of 465 means about 0.8 false alarms per product per year, or about 160 a year across 200 products. Run CUSUM on **weekly** or product-family aggregates, or raise h, and route alarms through FDR control.
- **Other monitors:**
  - **Calibration drift:** rolling 80% coverage outside [0.70, 0.90] over ≥ 50 evaluations.
  - **Elasticity drift:** `|μ_t − μ_{t−w}|/√(σ_t² + σ_{t−w}²) > 2` on two consecutive updates, or the regime model's change flag (v1 §21).
  - **Identification drift:** price variance over the last 90 days ≈ 0 while posterior SD(ε) > 0.25 means elasticity is no longer being learned; trigger TEST (§4).
- **Retraining vs redeploying.**
  - Retrain on: a schedule (weekly refit, monthly full retrain), a CUSUM alarm, a PSI alarm on key drivers (own price, competitor index, shock index), calibration drift or a regime change.
  - **Redeploy only if** the challenger beats the champion on the last 12 origins (Diebold–Mariano/HLN p < 0.10, or skill ≥ +2%) **and** passes the simulator regret gate (§6).
  - Automatically roll back if a post-deploy CUSUM alarm fires within 14 days.

#### Reproducibility (extends v1 §73)
**Lineage tuple per prediction:**

| Group | Fields |
|---|---|
| Identity and timing | `prediction_id`, `business_id`, `product_id`, `as_of` (decision time), `created_at` |
| Data | `data_snapshot_id` (content hash of input partitions, or a lake snapshot ID), `feature_view_versions` (name@hash), `training_window` |
| Code and environment | `code_commit`, `env_lock_hash` |
| Model | `model_artifact_hash`, `hyperparameters`, `population_prior_version` |
| LLM features | `llm_model_id`, `prompt_hash`, `extraction_run_id` |
| Randomness | `rng_root_entropy` + `spawn_key`, `n_draws` |
| Decision config | `risk_config_hash` (λ_LA, δ, α, budget), `constraint_config_version` |
| Output | `output_hash` |

**Seeds:**
- Use one root `np.random.SeedSequence(entropy)` per run and spawn a child per (business, product, component). [C]: `SeedSequence(20261006).spawn(3)` reproduced identical streams, first draws 606916, 377881, 343116 under numpy 2.4.6; `Generator.spawn` exists.
- Never use the global `np.random.seed`.
- Golden values are tied to the pinned numpy version.

**Determinism and audit:**
- Replay test: from the stored tuple, recompute the recommendation and assert an identical price and risk numbers to within 1e-9.
- Run golden tests with `OMP_NUM_THREADS=1` to avoid BLAS reduction-order noise.
- Store posterior draws, or their seeds, so the downside numbers shown to an owner can be audited later.

#### Tests
```python
def test_psi_vector():          # 0.020273
def test_psi_null_threshold():  # n=m=100, no drift: WARN rate <= 2% with the χ² threshold (vs 85% at fixed 0.1)
def test_cusum_arl():           # k=.5,h=5 two-sided in-control ARL within [430, 500] by simulation
def test_lineage_complete():    # every prediction row has all lineage fields non-null
def test_replay_identical():    # replay from lineage reproduces recommendation exactly
def test_seed_spawn_stable():   # SeedSequence(…).spawn(3) streams equal stored golden values
def test_llm_features_frozen(): # backtest feature matrix uses stored extraction rows, never live LLM calls
```

### Gaps
- Not verified in-session:
  - the Feast documentation wording;
  - Glasserman & Lin's quantitative findings;
  - the original sources and statistical justification of the 0.1/0.25 PSI rules. The χ² approximation above is my derivation from the symmetric-KL form, checked by simulation; the literature on PSI's sampling distribution could not be accessed.
- No guidance was found on POS sync latency or day-close conventions for offline-first small businesses in African markets. Measure them in the pilot.
- CUSUM tuning assumes roughly independent standardised errors. Autocorrelated daily errors shorten the in-control ARL, so re-simulate the ARL on residuals from the real model before fixing h.
