# 05 - Priors and hierarchical pooling

Status: specification only. Nothing in this chapter is implemented yet. It replaces v1 §41 (Bayesian hierarchical shrinkage), §42 (proprietary "Local Elasticity Prior"), the statistical part of §80 (multi-tenant learning), and the shrinkage parts of §85-86 and Phase 6.

Evidence tags used in this chapter (inherited from the research notes in `research_notes/LAEE elasticity engine specification/priors_shrinkage.md`, cited below as "the priors notes"):

- **[V]** confirmed by the researcher from search-result text at the URL given.
- **[L]** established literature cited from the researcher's knowledge; not re-verified. Check the source before quoting a number to users.
- **[C]** computed with numpy/scipy. Values marked **[C, re-run]** were recomputed while writing this chapter with the repo's `.venv` (numpy and scipy as pinned in `pyproject.toml`) and matched the priors notes.
- **[E]** engineering default chosen by this spec. It has no external source and must be calibrated by the release gates in §6.

Sign convention: ε = ∂ln Q/∂ln P, negative for normal demand, as in v1 §14 and the existing estimators in `src/laee/models/`.

---

## 1. Purpose, inputs, outputs

### 1.1 Purpose

Most LAEE tenants have 50-5,000 transactions per product and only a handful of regular-price changes. Their own-data elasticity estimates from chapter 03 (FE, DML) and chapter 04 (heterogeneity, ensemble) are therefore noisy, often wrong-signed, and sometimes undefined. This chapter specifies the layer that:

1. supplies an honest prior for every business-product before it has data (cold start);
2. combines that prior with the business's own evidence by exact, closed-form Gaussian partial pooling over the hierarchy industry -> country -> region -> local market -> business -> product;
3. learns the population parameters of that hierarchy from the platform's reference data (about 1.2 million sales rows) and from tenants, without any raw row leaving its tenant partition;
4. detects when the prior and the data disagree, handles near-zero and positive estimates, and labels every number by how much of it comes from the business's own data.

The layer is applied exactly once per business-product-estimand, as the last statistical step before decisions (chapter 07). It is cheap: every online computation is closed-form numpy, and no MCMC runs in the serving path.

### 1.2 Inputs

| Input | Producer | Content |
|---|---|---|
| Stage-1 summary per business-product-estimand | chapters 03 and 04, inside the tenant partition | (b, s) = data-only estimate and standard error of the canonical estimand; method; identification tier (v1 §76); n_obs; n_eff; K_pc (independent price changes, definition owned by chapter 04); sd of residual log price; estimation window; optional within-business covariance of product estimates |
| Taxonomy keys | chapter 01 | `industry_id` (one of the 13 ids in `laee.geo.market.INDUSTRY_IDS`), optional sub-industry, product category, price tier |
| Location keys and market covariates | chapter 02 | country, region, primary market id, density class (`urban`/`suburban`/`rural`), competition index, competitive price index (v1 §10) |
| Income covariate | chapters 01/09 (data sources) | catchment income or a proxy, plus the country median used as reference |
| Literature root priors | this chapter, §4.5 | per-industry (m_lit, s_lit) |
| Prior snapshot H_v | this chapter, §4.6 (monthly offline job) | hyperparameters τ_ℓ, meta-regression β and V_β, τ grid weights, leave-one-tenant-out cell aggregates |

Hard requirement on the stage-1 input: **(b, s) must be data-only.** If chapter 04's ensemble combiner (its "bias-aware" hierarchical combiner) is used, it must be run with an infinite prior variance (no prior term) and must state whether its s already includes identification-bias variance. Otherwise the prior is applied twice. See §4.1.

### 1.3 Outputs

For every business-product-estimand the layer returns an `ElasticityEstimate` (`src/laee/types.py`) with:

- `value` = posterior mean, `std_error` = posterior SD, `intervals` = posterior 80/90/95% intervals;
- `method` = `"hier_posterior"` or `"prior_only"`;
- `identification_tier` = the stage-1 tier (pooling does not change identification), or 0 for prior-only (proposed; §4.10 and §7);
- `diagnostics["prior"]` = a fixed set of keys (§5.3): prior mean and SD, own-data share, evidence label, conflict p-value, P(ε > 0), P(ε > -1), dominating prior level, snapshot id.

The offline job additionally writes a versioned prior snapshot and per-level diagnostics (pooling factors, τ̂ and their uncertainty, calibration statistics).

---

## 2. Changes from v1, with evidence

| # | v1 text | Problem | v2 change | Evidence |
|---|---|---|---|---|
| 1 | §41: ε_i ~ N(μ_industry,market, σ²) | One σ² conflates sampling noise of the business's estimate with true between-business heterogeneity. The weight on own data cannot be computed without both. | Two-stage model: b_i given θ_i ~ N(θ_i, s_i²); θ_i ~ N(parent, τ²). Own-data weight = τ²/(τ² + s_i²). | Standard normal-normal model, BDA3 ch. 5 [L]; Efron & Morris 1973/1975 [L] |
| 2 | §41 diagram has five levels, the equation one | No variance per level, no nesting rule, no backoff. | Exact Gaussian tree with one τ per level per industry, O(N) upward/downward passes (§4.3). | Gaussian belief propagation on trees is exact (Weiss & Freeman 2001 [L]); validated against dense GLS to < 1e-15 [C] |
| 3 | §41 gives no estimator for σ² | With few businesses per cell, ML/REML τ̂² is often exactly 0, which silently becomes complete pooling and overconfidence. | τ estimated once per level per industry across cells; PM for 3-9 groups, REML for ≥ 10; half-normal hyperprior; integrate over τ on a grid for intervals. | REML τ̂² = 0 in 28% of 8-group replications with true τ = 0.5 [C, re-run]; plug-in EB 95% intervals cover 83%, grid Bayes 95% [C, re-run]; Veroniki et al. 2016 [V]; Chung et al. 2013 [V] |
| 4 | §41 "50 transactions" | Transaction count is not information. Information about ε scales with n·Var(ln p) and the number of independent price changes. | Own-data weight is driven by s (which already encodes price variation) and by K_pc floors (§4.1, §4.7). | Derivation s ≈ σ_e/(√n·sd(ln p)) [C] |
| 5 | §42 PriorElasticity = f(...) | No functional form, no target estimand, no variance, collinear inputs. A point value is not a prior. | Meta-analytic-predictive (MAP) meta-regression prior N(x*ᵀβ̂, τ_total² + x*ᵀV_βx*), literature-anchored industry intercepts, weak sign-informed slopes, one covariate per concept (§4.5). | Hoch et al. 1995 [V]; DellaVigna & Gentzkow 2019 [V]; Green et al. 2013 [V]; Schmidli et al. 2014 [V] |
| 6 | §42 Update(...) | Undefined. | Conjugate normal update; robust mixture (rMAP) update under prior-data conflict (§4.8). | Schmidli et al. 2014 [V]; Evans & Moshonov 2006 [L] |
| 7 | none | Naive observational estimates are attenuated; a prior trained on them is biased toward zero. | Identification-tier bias variance and tier shift; predict at the best tier (§4.5). | Bijmolt, van Heerde & Pieters 2005: endogeneity correction "significantly increases the magnitude" [V] |
| 8 | §80 "learn θ_industry,region,market" | Does not say what crosses the tenant boundary or how released aggregates are protected. | Only stage-1 summaries cross; leave-one-tenant-out priors; k_min, dominance cap, batching; optional Laplace DP for external releases (§4.6). | Lin & Zeng 2010 [L]; Dwork et al. 2006 [L]; Hundepool et al. 2012 [L] |
| 9 | §80 "federated-style aggregation" | Undefined; iterative federated training is unnecessary for Gaussian models. | One-shot aggregation of summaries (exact for the Gaussian hierarchy). | Lin & Zeng 2010 [L]; derivation in priors notes §5.2 |
| 10 | §79 flywheel | A population snapshot that includes business i, used as business i's prior and updated with its data again, counts b_i twice. | Joint tree posterior, equivalently leave-one-tenant-out priors (§4.2, §4.6). | Sequential Gaussian conjugacy [derivation] |
| 11 | §85-86 defer hierarchy to V2 | The V1 target users are exactly those whose estimates are noisy. Shrinkage is the cheapest high-value component. | Ship the one-level closed-form layer and the literature prior table with the first release; tree and meta-regression follow. | priors notes §1 |
| 12 | §86 ensemble of FE, DML, CF | Same-data estimators have correlated errors; feeding several into the hierarchy double counts. | Exactly one stage-1 record per business-product-estimand (§4.1). | priors notes §1; chapter 04 |
| 13 | none | Normal priors put mass on ε > 0; raw estimates are often positive. | Pool on the raw scale; sign handled at the decision layer with P(ε > 0) gates (§4.9). | Boatwright, McCulloch & Rossi 1999 [L]; Jensen & Miller 2008 [L] |
| 14 | none | Prior-only numbers can be shown as if measured. | Evidence labels (prior_only, prior_dominated, blended, data_dominated) and guardrails (§4.10). | priors notes §6.1 |

---

## 3. Notation

| Symbol | Meaning |
|---|---|
| i, j | business i; product j within business i |
| ℓ | hierarchy level: I (industry root), C (country), R (region), M (market), B (business), P (product) |
| θ_v | true canonical elasticity at tree node v |
| b_v, s_v | stage-1 data-only estimate at an observed node and its standard error |
| s̃_v² | effective sampling variance after inflation and bias allowance (§4.1) |
| τ_ℓ | between-node SD at level ℓ (heterogeneity of children around their parent) |
| μ, μ̂ | mean of a group; its GLS estimate |
| B_v | shrinkage factor = weight on the prior = s̃_v²/(s̃_v² + τ²) in the one-level case |
| w_own | own-data share = 1 - V_post/V_prior (equals 1 - B in the one-level case) |
| k | number of groups (businesses, markets, ...) informing a τ estimate |
| K_pc | independent regular-price changes (definition owned by chapter 04: \|Δ ln p\| ≥ 0.02, held ≥ 7 days, ≥ 14 days apart, not a promotion, no stockout within ±7 days) |
| m_lit, s_lit | literature root prior mean and SD of an industry's mean elasticity |
| x_i, β, V_β | meta-regression covariates, coefficients, coefficient covariance |
| δ_tier | identification-bias SD added to stage-1 variance by tier |
| H_v | prior snapshot version v |
| Φ, φ | standard normal CDF, PDF |

All variances are in squared elasticity units. "Industry" always means one of the 13 app ids.

---

## 4. Methods

### 4.1 Stage-1 input contract and admissibility

**When to use.** Always, before any pooling.

**Estimand contract.** Records are pooled only if all of these match:

- level: product-at-shop (default) or category-at-shop;
- price type: regular prices; promotional periods excluded per v1 §62 and chapter 01;
- horizon: the canonical 4-12-week response to a regular-price change. Short-run (0-2 weeks) and promotional elasticities may be reported but are never pooled with the canonical estimand (Pauwels, Hanssens & Siddarth 2002 [L]: promotion effects are mostly transient);
- price deflation: prices deflated by a local CPI or expressed relative to competitor prices where inflation is material (chapter 01). In high-inflation economies nominal changes are not real price changes.

Each record carries `estimand_id`; mismatched ids never share a tree.

**One record per business-product-estimand.** FE, DML and causal-forest outputs computed on the same data are not independent observations. The stage-1 record is either the best-identified single estimator or chapter 04's combined estimate with a bootstrapped joint covariance. It must be data-only (no prior term).

**Effective variance.**

```
s̃² = s² · infl(K_pc) + δ_tier²·[bias_variance_included == False]

infl(K_pc) = ν/(ν - 2),  ν = K_pc - 1     if K_pc ≥ 4    (t-to-normal variance match)
           = 3                             if K_pc ∈ {2, 3} (the ν = 3 value; "early signal")
record is inadmissible                     if K_pc ≤ 1 or sd(residual ln p) < 0.02, or s not finite or s ≤ 0
```

δ_tier defaults (priors notes §3.3, [E]): 0 for Tiers 1-3, 0.2 for Tiers 4-5, 0.4 for Tier 6 (FE), 0.6 for Tier 7. Chapter 04's combiner uses a different bias scale (0, 0.1, 0.2, 0.3, 0.6 for Tiers 1, 3, 5, 6, 7). Only one of the two may be applied to a record; the flag `bias_variance_included` decides. Unifying the two tables is an open question (§7).

Standard errors must be clustered or HAC (chapter 03). Unclustered SEs on serially correlated sales make s too small and the own-data weight too large.

**Who enters hyperparameter fitting.** Only records with K_pc ≥ 4 and not flagged "identification suspect" (§4.9). Records with K_pc ∈ {2, 3} receive their own posterior but do not inform τ or β, because their b is far from normal and their s unreliable (priors notes §5.4).

**Outputs.** `StageOneSummary` (§5.2). **Failure modes.** Estimand mismatch (pooling promotional with regular elasticities); SE understated by ignored clustering; two estimators of the same data both entered. **Cost.** O(1) per record.

### 4.2 One-level normal-normal partial pooling (the core update)

**When to use.** Every posterior. The tree in §4.3 is this update applied recursively; the cold-start and LMIC defaults in §4.5 are its prior side.

**Model and closed form** (BDA3 ch. 5 [L]; validated [C]):

```
b_i | θ_i ~ N(θ_i, s̃_i²)
θ_i | μ_i, τ ~ N(μ_i, τ²)          μ_i = parent mean or x_iᵀβ
marginal:  b_i ~ N(μ_i, s̃_i² + τ²)

B_i = s̃_i² / (s̃_i² + τ²)
E[θ_i | b]   = (1 - B_i)·b_i + B_i·μ_i
Var[θ_i | b] = (1 - B_i)·s̃_i² = 1 / (1/s̃_i² + 1/τ²)

GLS group mean:  w_i = 1/(s̃_i² + τ²),  μ̂ = Σ w_i b_i / Σ w_i,  Var(μ̂) = 1/Σ w_i
μ integrated (flat prior), τ fixed:
  E[θ_i | b]   = (1 - B_i)·b_i + B_i·μ̂
  Var[θ_i | b] = (1 - B_i)·s̃_i² + B_i²·Var(μ̂)

new member (cold start):  θ_new ~ N(μ̂, τ² + Var(μ̂))
```

The cold-start prior is the predictive distribution N(μ̂, τ² + Var(μ̂)), not N(μ̂, Var(μ̂)) and not v1's N(μ, σ²) with an unspecified σ².

**Leave-one-out identity.** Updating the leave-one-out predictive prior N(μ̂_(-i), τ² + Var(μ̂_(-i))) with b_i gives exactly the joint posterior above. Using the full-data μ̂ (which already contains b_i) as a prior and then updating with b_i double counts b_i. This is why §4.6 serves leave-one-tenant-out priors.

**Pseudocode.**

```
def shrink(b, s2_eff, mu_prior, v_prior):
    # v_prior = τ² + Var(μ̂) for a leave-one-out group prior, or a full prior variance
    B = s2_eff / (s2_eff + v_prior)
    mean = (1 - B) * b + B * mu_prior
    var  = (1 - B) * s2_eff            # = 1/(1/s2_eff + 1/v_prior)
    return mean, var, 1 - B            # w_own = 1 - B
```

**Minimum data.** One admissible record. With no admissible record the output is the prior itself (§4.7, §4.10).

**Outputs.** Posterior mean, variance, B, w_own.

**Failure modes.** s̃ too small (unclustered SE): over-weights own data. τ too small (boundary τ̂ = 0): complete pooling and overconfidence; mitigated by §4.4. Precision dependence: if s_i correlates with θ_i (for example small businesses are systematically less elastic), shrinking toward a common μ is biased; add ln s_i as a meta-regression covariate and test its slope (Walters, NBER w33091 [V, title]).

**Invariants to test.** s̃ -> 0 gives b_i; s̃ -> ∞ gives the prior; τ -> 0 gives μ̂; posterior variance ≤ min(s̃², τ² + Var(μ̂)).

**Cost.** O(1) per record.

### 4.3 The hierarchy: exact Gaussian tree passes

**When to use.** Whenever more than one level has data, which is the normal state after the first prior snapshot.

**Tree definition.** Industry and region are crossed in reality (a region contains many industries). V2 uses nested combination nodes, with region-wide effects that cut across industries carried by meta-regression covariates (§4.5):

```
industry (root; literature prior N(m_lit, s_lit²) or meta-regression mean)
  -> country within industry           τ_C
    -> region within industry-country  τ_R
      -> market within industry-region τ_M   (market = chapter 02 primary market id)
        -> business                    τ_B
          -> product                   τ_P
```

v1 §41 omitted the country level. It is added because elasticity levels, currencies and informality differ by country (priors notes §4), and a country τ becomes estimable once 3 or more countries have data. While fewer than 3 countries have data, τ_C is fixed at the median of its hyperprior, HalfNormal(0.4), which is 0.4 × 0.674 = 0.27 [E].

A business belongs to exactly one market node: chapter 02's primary market. For ecommerce (no geographic market, `RADIUS_PRIORS_KM["ecommerce"] == ()`), the market level is skipped and businesses hang directly from the region (or country) node.

**Leaves.** Use product-level estimates as leaves with the business node latent, or a business-level pooled estimate as the leaf. Never both for the same business: that double counts the same transactions.

**Model.** θ_root ~ N(m_root, v_root); θ_v | θ_parent ~ N(θ_parent, τ_ℓ(v)²); observed node b_v | θ_v ~ N(θ_v, s̃_v²).

**Algorithm** (priors notes §2.6, validated [C]: matches dense GLS `m + PHᵀ(HPHᵀ + R)⁻¹(b - Hm)` to below 1e-15 on a 12-node, 4-level tree; log-likelihood matches `scipy.stats.multivariate_normal.logpdf` on a 14-node tree, -12.812375363501369):

```
UPWARD (deepest level first). For node v collect factors
  F_v = {(b_v, s̃_v²) if v observed} ∪ {(a_c, A_c + τ_c²) : child c with data below it}
  P_v = Σ_f 1/A_f ;   a_v = (Σ_f a_f/A_f) / P_v ;   A_v = 1/P_v
  log Z_v = -½ Σ_f log(2π A_f) + ½ log(2π A_v) + a_v²/(2A_v) - Σ_f a_f²/(2A_f)
  log p(all b | τ) = Σ_v log Z_v + log N(a_root ; m_root, A_root + v_root)

DOWNWARD (root first)
  root:  prec = 1/v_root + 1/A_root ;  M = (m_root/v_root + a_root/A_root)/prec ;  V = 1/prec
  child c of v with upward message (a_c, Ã_c = A_c + τ_c²):
     cavity of v without c:  Λ = 1/V_v - 1/Ã_c ;  m = (M_v/V_v - a_c/Ã_c)/Λ
     prior of θ_c:           N(m, 1/Λ + τ_c²)
     posterior:              prec = 1/(1/Λ + τ_c²) + 1/A_c ;  mean = (m/(1/Λ + τ_c²) + a_c/A_c)/prec
  child with no data:        posterior = N(M_v, V_v + τ_c²)
```

The last line is the cold-start prior of a new business (or new product) in an existing parent. The cavity step is what prevents double counting; omitting it is the most likely implementation bug.

**Vectorisation.** Process one level at a time. Upward: `P = np.bincount(parent_idx, weights=1/Ã, minlength=n_parents)` and `S = np.bincount(parent_idx, weights=a/Ã, minlength=n_parents)`. Downward: gather the parent's (M, V) by `parent_idx`. Cost O(N) per hyperparameter evaluation. Measured upward step for 200,000 leaves into 20,000 parents: about 3.6 ms on this container [C, re-run; indicative only].

**Correlated products within a business.** Products share demand shocks and controls. If chapter 03/04 supplies a joint covariance S_i for the business's product estimates (from a stacked within-tenant regression), the message from the products to the business node is

```
P_i = 1ᵀ (S_i + τ_P² I)⁻¹ 1,     a_i = 1ᵀ (S_i + τ_P² I)⁻¹ b_i / P_i,     A_i = 1/P_i
```

and product posteriors are obtained by the matching block Gaussian conditional. Without S_i, products are treated as independent and the business-level precision is overstated; record `products_assumed_independent = True` in diagnostics.

**Meta-regression offset.** The tree operates on residualised estimates b̃ = b - x_iᵀβ̂ (§4.5). Posterior means get x_iᵀβ̂ added back; posterior variances get x_iᵀV_βx_i added. This ignores the covariance between β̂ and the random effects, which is second order when β is estimated from many businesses; the exact alternative is the mixed-model equations below.

**Crossed effects (later version).** If region-wide effects across industries must be random rather than fixed covariates, solve Henderson's mixed-model equations `[XᵀR⁻¹X, XᵀR⁻¹Z; ZᵀR⁻¹X, ZᵀR⁻¹Z + G⁻¹][β; u] = [XᵀR⁻¹b; ZᵀR⁻¹b]` with `scipy.sparse.linalg.splu` (Robinson 1991 [L]). Posterior variances then need selected inversion or offline MCMC. Not in the first release.

**Backoff.** The tree needs no statistical backoff: a node with no data inherits its parent's posterior plus τ². Backoff exists only as a display and privacy rule (§4.6).

**Minimum data.** Any. Levels with no data simply pass the parent through.

**Failure modes.** Cavity not removed; levels processed in the wrong order; leaves at two levels for the same business; market id changing between snapshots (chapter 02 must version market ids).

**Cost.** Online: one downward path from root to the tenant's products, O(depth + products) given cached upward messages, microseconds to milliseconds. Offline: O(N) per likelihood evaluation.

### 4.4 Between-group variance τ² with few groups

#### 4.4.1 Which estimator, when

| Situation | Estimator | Interval for τ² | Notes |
|---|---|---|---|
| Level-wide τ_ℓ in the tree (default) | Penalised ML-II: maximise log p(b \| τ) + Σ_ℓ log HalfNormal(τ_ℓ; σ_ℓ) over ln τ_ℓ, L-BFGS-B | Profile likelihood or posterior grid | One τ per level per industry, pooled over all cells of that level |
| One level, k ≥ 10 groups, or meta-regression | REML | Q-profile | Veroniki et al. 2016 recommend REML for continuous data [V] |
| One level, 3 ≤ k ≤ 9 | Paule-Mandel | Q-profile | Recommended for all data types by Veroniki et al. 2016 [V] |
| Fast screen, starting value | DerSimonian-Laird | none | Negatively biased when heterogeneity is large (Langan et al. 2019 [L]); "long challenged" (Veroniki et al. [V]) |
| k ≤ 2 | do not estimate | - | Borrow the level-wide τ from the tree fit or use the hyperprior |
| Small-sample point shrinkage without grid | Morris factor | - | multiply B̂ by (k - 3)/(k - 1) (Morris 1983 [V]) |
| Sanity check only | James-Stein | - | equal variances; k ≥ 4 when shrinking to the grand mean [L] |

**Rule: never estimate τ per cell when cells have fewer than about 10 businesses.** Estimate one τ per level per industry with cell-specific means. For a single level this is meta-regression REML with X = one-hot cell indicators, so hundreds of businesses inform τ_B even if each market has three.

#### 4.4.2 Formulas (validated on the BCG reference data [C, re-run])

```
DL:    w = 1/s²;  μ_FE = Σ w b / Σ w;  Q = Σ w (b - μ_FE)²
       τ² = max(0, (Q - (k - 1)) / (Σ w - Σ w² / Σ w))

PM:    solve Q_gen(τ²) = Σ (b - μ̂(τ²))² / (s² + τ²) = k - 1
       Q_gen is decreasing in τ²; if Q_gen(0) ≤ k - 1 then τ² = 0; else scipy.optimize.brentq on [0, hi],
       hi doubled from var(b) until Q_gen(hi) < k - 1; xtol 1e-14

REML:  ℓ_R(τ²) = -½ Σ log(s² + τ²) - ½ log Σ w - ½ Σ w (b - μ̂)²,   w = 1/(s² + τ²)
       fixed point τ² <- max(0, Σ w² ((b - μ̂)² - s²) / Σ w² + 1/Σ w), start at DL, tol 1e-12;
       fallback: scipy.optimize.minimize_scalar(bounded) on -ℓ_R over [0, 10·var(b) + max s²]

ML:    ℓ(τ²) = -½ Σ log(s² + τ²) - ½ Σ w (b - μ̂)²       (biased low for small k; diagnostics only)

Meta-regression REML:
       ℓ_R = -½ Σ log(s² + τ²) - ½ log|XᵀWX| - ½ (b - Xβ̂)ᵀ W (b - Xβ̂),   β̂ = (XᵀWX)⁻¹ XᵀWb

HKSJ CI for μ:  Var_HK = Σ w (b - μ̂)² / ((k - 1) Σ w);  μ̂ ± t_{k-1, .975} √Var_HK
Prediction interval for a new group:  μ̂ ± t_{k-2, .975} · √(τ̂² + Var(μ̂))      (Higgins et al. 2009 [L])
Q-profile CI:   {τ² : χ²_{k-1, .025} ≤ Q_gen(τ²) ≤ χ²_{k-1, .975}}             (Viechtbauer 2007 [L])
I² = max(0, (Q - (k - 1))/Q)                                                (Higgins & Thompson 2002 [L])
```

BCG golden values (13 trials, log risk ratio), recomputed while writing this chapter: DL τ² = 0.3088, Q = 152.2330, μ̂ = -0.7141 (SE 0.1787); PM τ² = 0.3181, μ̂ = -0.7150 (SE 0.1809); REML τ² = 0.3132, μ̂ = -0.7145 (SE 0.1798); HKSJ SE 0.1808; t prediction interval [-2.0084, 0.5793]. The priors notes report the REML Q-profile CI [0.1197, 1.1115] and ML τ² = 0.2800 [C]. These match metafor's documented values as recalled by the researcher; certify once against metafor (§7).

#### 4.4.3 Floors and the zero boundary

With few groups the ML/REML/PM estimate sits at τ̂² = 0 with high probability even when heterogeneity is real: REML gave exactly 0 in 28% of 8-group replications with true τ = 0.5 (θ_i ~ N(-2, 0.5²), s_i ~ U(0.3, 1.2), 1,000 replications) [C, re-run]. A zero τ means complete pooling, and the business's interval collapses to the pooled-mean interval.

Three defences, in order of preference:

1. **Integrate over τ** for every interval that is shown to users (§4.4.4). This is the production default.
2. **Penalise** the variance components in ML-II with a half-normal hyperprior per level (σ_C = 0.4 [E], σ_R = 0.4, σ_M = 0.4, σ_B = 0.7, σ_P = 0.8 [E, priors notes]). Chung et al. 2013 show that a weak penalty (they use log-gamma(2, λ -> 0)) keeps the mode about one standard error from zero when the MLE is at zero [V]. A half-normal does not have that exact property; it is used here because the grid posterior in §4.4.4 is calibrated under it.
3. **Hard floor for plug-in paths only.** Where a plug-in τ is unavoidable (for example a cached single τ used in a batch job), use τ_plug = max(τ̂, τ_floor) with τ_floor,ℓ = the 10th percentile of the level's posterior for τ from the latest snapshot [E]. Never use p(τ) ∝ 1/τ: its posterior is improper at τ -> 0. Never use inverse-gamma(ε, ε) priors on τ² (Gelman 2006 [L]).

Hyperprior scale must be in elasticity units. A HalfNormal(0.7) prior applied to data on another scale (for example the eight-schools data) forces complete pooling (all eight means about 7.7 [C]); unit tests must not reuse the production σ_ℓ on non-elasticity fixtures.

#### 4.4.4 Integrating over τ (production intervals)

For the most influential level (τ_B) the snapshot stores a grid posterior. For a one-level group with a flat prior on μ (BDA3 §5.4 [L]):

```
log p(τ | b) = log p(τ) + ½ log Var(μ̂(τ)) - ½ Σ log(s_i² + τ²) - ½ Σ (b_i - μ̂(τ))²/(s_i² + τ²)
π_g ∝ p(τ_g | b) · Δτ_g                          # quadrature weights are mandatory on non-uniform grids
M_ig = (1 - B_ig) b_i + B_ig μ̂_g;   V_ig = (1 - B_ig) s_i² + B_ig² Var(μ̂_g)
E[θ_i | b]   = Σ_g π_g M_ig
Var[θ_i | b] = Σ_g π_g (V_ig + M_ig²) - E²
interval endpoints: solve Σ_g π_g Φ((x - M_ig)/√V_ig) = q with brentq (exact mixture quantiles)
```

Defaults: 700 grid points (600-800 acceptable) on [0, 5σ_τ + 3·max s_i], including τ = 0; HalfNormal(σ_B = 0.7) prior. A uniform prior on τ gives a proper posterior only for k ≥ 3, because p(b | τ) ∝ τ^(1-k) as τ -> ∞.

In the tree, grid τ_B and hold the other τ_ℓ at their penalised ML-II values. Store π_g (snapshot-level) and, per request, evaluate the business's posterior at each grid point by re-running the downward pass with τ_B = τ_g; with cached upward messages this is O(G·depth).

**Coverage evidence** (nominal 95%; θ_i ~ N(-2, τ²); s_i ~ U(0.3, 1.2); priors notes [C], two cells re-run):

| k | true τ | oracle τ | plug-in REML EB | grid, HalfNormal(1), exact quantiles | no pooling |
|---|---|---|---|---|---|
| 4 | 0.5 | - | 0.865 | 0.960 | 0.949 |
| 8 | 0.5 | 0.949 | 0.835-0.837 (re-run 0.833) | 0.953 (re-run 0.948, normal approximation) | 0.954 |
| 8 | 1.0 | - | 0.875 | 0.930 | 0.952 |
| 30 | 0.5 | 0.950 | 0.876-0.878 | 0.933 | 0.950 |
| 30 | 1.0 | 0.950 | 0.937 | 0.943 | - |
| 100 | 0.5 | 0.950 | 0.931 | - | - |

Grid intervals are 23-46% narrower than no-pooling intervals (mean width 1.87 vs 2.94 at k = 8, τ = 0.5) while keeping 92-96% coverage. A non-uniform grid without Δτ weights dropped coverage to 0.866-0.886 in the priors notes; this is a required unit test.

**Cheaper alternative (Kass & Steffey 1989 [L]),** valid only if τ̂² > 0 and k ≥ about 10:

```
Var_KS(θ_i) ≈ (1 - B_i) s_i² + B_i² Var(μ̂) + [B_i (1 - B_i)(b_i - μ̂)]² · Var(ln τ̂²)
∂E[θ_i]/∂ln τ² = B_i (1 - B_i)(b_i - μ̂)
Var(ln τ̂²) ≈ 1/(τ̂⁴ · I(τ̂²)),  I ≈ ½ tr(P²),  P = W - W 1 1ᵀ W/(1ᵀ W 1)
```

Golden value: derivative 0.36519849 at b = -0.4, s² = 0.9, μ = -2, τ² = 0.49 [C].

**Failure modes.** Grid upper bound too low (posterior mass truncated: assert π at the last grid point < 1e-6); missing Δτ; τ-prior in the wrong units; normal approximation of mixture quantiles (about 1 percentage point lower coverage).

**Cost.** Offline grid: 700 × O(N) per level. Online: 700 × O(depth) per request, well under 10 ms in numpy [E, estimate].

### 4.5 Initial prior table and the meta-regression prior

#### 4.5.1 Which elasticity a single shop faces

- Category or market-level elasticities (food demand systems, fuel, telecom) measure the response when all sellers' prices move. They are a lower bound on the magnitude of one shop's own-price elasticity, because a shop that raises its price alone also loses customers to neighbours.
- Brand-level meta-analytic means are the closest published analogue for an item at a shop: -1.76 (Tellis 1988, 367 estimates, "very narrow distribution around the mean" [V]) and -2.62 (Bijmolt et al. 2005, 1,851 estimates from 81 studies [V]).
- Lerner consistency anchor (Lerner 1934 [L]): at a profit-maximising interior optimum (P - MC)/P = -1/ε, so |ε| > 1. Implied ε for contribution margin m: 0.15 -> -6.67, 0.20 -> -5.00, 0.30 -> -3.33, 0.40 -> -2.50, 0.50 -> -2.00, 0.60 -> -1.67, 0.70 -> -1.43 [C]. Used only as a prior predictive check (§4.8), because SMBs often price by cost-plus or inertia, and even chains forgo large profits by not pricing at the optimum (DellaVigna & Gentzkow: the median chain forgoes about $16m a year [V]).

#### 4.5.2 Root prior per industry id

Estimand: shop-level own-price elasticity of a product or service line at regular prices over a 4-12-week horizon. m_lit is the prior mean of the industry mean elasticity; s_lit is the uncertainty of that mean. The predictive SD for a new business before any platform fit adds τ_seed = 0.7 [E] for business and product heterogeneity: SD_pred = √(s_lit² + 0.7²). P columns use N(m_lit, SD_pred²). All derived columns were recomputed for this chapter [C, re-run].

**These are engineering placeholders anchored on literature, not literature values. They are overwritten by the first platform fit (§4.6) and retained only as the root hyperprior.**

| industry_id | m_lit | s_lit | SD_pred | 90% predictive interval | P(ε > 0) | P(ε > -1) | Anchor and evidence | Confidence |
|---|---|---|---|---|---|---|---|---|
| grocery | -1.8 | 0.8 | 1.06 | [-3.55, -0.05] | 0.045 | 0.226 | brand-level -1.76 (Tellis [V]) / -2.62 (Bijmolt [V]); U.S. food category \|ε\| 0.27-0.81 (Andreyeva et al. 2010 [V]) | medium |
| grocery, LMIC staples override (see below) | -2.0 | 1.0 | 1.22 | [-4.01, 0.01] | 0.051 | 0.206 | Nigeria household -0.63 to -1.79, local rice about -1.7 [V]; South Africa sugar -2.42, soft drinks -1.18 [V]; cereals -0.61 low-income vs -0.43 high-income countries (Green et al. 2013 [V]) | low-medium |
| cafe | -1.4 | 0.8 | 1.06 | [-3.15, 0.35] | 0.094 | 0.353 | food away from home about -0.8, category level (Andreyeva et al. [L] per-category value) | low-medium |
| restaurant | -1.4 | 0.8 | 1.06 | [-3.15, 0.35] | 0.094 | 0.353 | same anchor as cafe; no verified evidence separates them | low-medium |
| apparel | -2.0 | 1.0 | 1.22 | [-4.01, 0.01] | 0.051 | 0.206 | brand-level means only; no apparel-specific source verified | low |
| electronics | -2.5 | 1.0 | 1.22 | [-4.51, -0.49] | 0.020 | 0.110 | brand-level -2.62 [V]; "durables likely more elastic" is unverified | low |
| pharmacy | -1.2 | 0.8 | 1.06 | [-2.95, 0.55] | 0.129 | 0.425 | medical care about -0.2 at market level (Manning et al. 1987 [L]) | low |
| hair_salon | -1.3 | 0.9 | 1.14 | [-3.18, 0.58] | 0.127 | 0.396 | none found | low |
| auto_repair | -1.3 | 0.9 | 1.14 | [-3.18, 0.58] | 0.127 | 0.396 | none found | low |
| professional_service | -1.2 | 0.9 | 1.14 | [-3.08, 0.68] | 0.146 | 0.430 | none found | low |
| other_service | -1.3 | 1.0 | 1.22 | [-3.31, 0.71] | 0.143 | 0.403 | none found; services mean with s_lit widened to 1.0 [E, this chapter] | low |
| wholesale | -2.5 | 1.2 | 1.39 | [-4.79, -0.21] | 0.036 | 0.140 | none found; homogeneous goods sold to resellers argue for high magnitude [E] | low |
| ecommerce | -2.5 | 1.2 | 1.39 | [-4.79, -0.21] | 0.036 | 0.140 | very elastic on price-comparison sites (Ellison & Ellison 2009 [L]); less elastic for online grocery (Chu et al. 2008 [L]) | low |
| other | -1.8 | 1.2 | 1.39 | [-4.09, 0.49] | 0.098 | 0.282 | Tellis brand-level mean, widest s_lit [E, this chapter] | low |

Notes on the table:

- **Rows added by this chapter.** The priors notes give no row for `other_service` or `other`; their values are this chapter's engineering choices. Cafe and restaurant share one row in the notes.
- **LMIC staples override.** For `grocery` businesses in low- and lower-middle-income countries, the country node's prior is N(-2.0, 1.0²) for products in staple categories (cereals, flour, rice, sugar, cooking oil, beans; the category list is owned by chapter 01). The override is applied at the country node, not as an income multiplier at the shop level (see §4.5.4), so it is not double counted with the income slope.
- **Weight of the literature.** With s_lit = 0.8-1.2 the literature is worth about 1.3-1.8 "business-equivalents" (ratio of per-business predictive variance to s_lit²) [C, re-run]. Platform data dominate as soon as a few businesses in the cell have data. This is intended: the literature comes from other countries, aggregation levels and decades.
- **Excluded lines.** Airtime and mobile data resale are sold at operator-set face value; do not run price optimisation for them (priors notes §4). Mobile category elasticities are about -0.2 to -0.3 short-run and about -0.8 long-run in developing countries [V, attribution of individual numbers unclear]. Fuel retail has no app industry id; if added, the notes suggest N(-3.0, 1.5²) at station level [E].
- **Category-level priors (not shop-level).** Where another chapter needs a category or market-level elasticity (for example chapter 06 shocks), scale high-income literature magnitudes by about 1.4 for low-income catchments (the Green et al. cereals ratio 0.61/0.43) with a prior SD of at least 0.3. Do not apply this factor to shop-level priors.

#### 4.5.3 The meta-regression prior (the concrete f(...) of v1 §42)

**When to use.** From the first platform snapshot onward, for every root-to-business mean. Before the first snapshot, the root prior is the table above with zero slopes.

**Model.**

```
θ_i = β_ind[i] + δ_cat[i] + β_tierᵀ·tier_i + β_inc·ln(inc_i / inc_ref,country(i))
      + β_comp·comp_z,i + β_dens·dens_⊥,i + β_s·ln s_i + γ_idtier(i) + u_i
b_i | θ_i ~ N(θ_i, s̃_i²)           u_i = tree random effects (country, region, market, business, product)

Fit (penalised GLS given τ):
  W = diag(1/(s̃_i² + τ_total,i²))
  β̂  = (XᵀWX + Σ₀⁻¹)⁻¹ (XᵀWb + Σ₀⁻¹β₀)
  V_β = (XᵀWX + Σ₀⁻¹)⁻¹
  τ by REML on ℓ_R (or jointly by penalised ML-II with the tree)

Prior for a new business with covariates x* (evaluated at the best identification tier):
  N( x*ᵀβ̂ ,  τ_C² + τ_R² + τ_M² + τ_B² (+ τ_P² for a product) + x*ᵀV_βx* [+ τ_shift²] )
```

τ_total,i² is the sum of the level variances between the industry root and the record's level. τ_shift² is added when x* lies outside the convex hull of the training covariates or the business is in a country with no platform data; default τ_shift = 0.3 [E]. In the tree implementation the random-effect part is not a sum but the downward pass, which uses the actual posteriors of the business's country, region and market.

**Covariates and their starting priors Σ₀** (all diagonal, weak, learned from data; priors notes §3.3). Slopes are stated relative to |m_lit| of the industry, and positive means "toward zero":

| Covariate | Definition | Prior mean | Prior SD | Basis |
|---|---|---|---|---|
| β_ind | industry intercept | m_lit (table above) | s_lit | §4.5.2 |
| δ_cat | product category within industry | 0 | 0.5 | [E]; literature ordering (for example soft drinks more elastic than eggs, Andreyeva [L]) used only as a sanity check |
| β_inc | ln(catchment income / country median income) | +0.10·\|m_lit\| per ln-unit | 0.10·\|m_lit\| | direction: elasticity falls in magnitude with income (DellaVigna & Gentzkow [V]; Green et al. [V]); magnitude derived from the Green et al. cereals ratio over a 20-60x cross-country income ratio, d ln\|ε\|/d ln income ≈ -0.085 to -0.117 [C] |
| β_comp | comp_z = standardised ln(1 + travel-time-weighted competitor count) from chapter 02 | -0.10·\|m_lit\| per SD | 0.10·\|m_lit\| | direction: more competition, more elastic; kept smaller than income because demographics outweigh competition (Hoch et al. 1995 [V]) |
| β_dens | density residualised on comp_z (dens_⊥) | -0.05·\|m_lit\| per SD | 0.10·\|m_lit\| | [E]; included only after orthogonalising against competition, otherwise dropped |
| β_tier: premium | competitive price index (v1 §10) < 0.9, that is, priced above weighted competitors | +0.20·\|m_lit\| | 0.15·\|m_lit\| | [E]; no verified source for tier effects |
| β_tier: discount | competitive price index > 1.1 | -0.15·\|m_lit\| | 0.15·\|m_lit\| | [E]; same |
| β_s | ln s_i (precision dependence) | 0 | 0.2 | [E]; tests for small-business differences (Walters [V, title]) |
| γ_idtier | identification tier dummies {1-3, 4-5, 6, 7}, reference = best tier with ≥ 10 businesses | 0 | 0.5 | corrects the attenuation of naive estimates (Bijmolt et al. [V]); prediction is made at the reference tier |

Thresholds 0.9/1.1 for price tier are [E]. The density classes `urban`/`suburban`/`rural` from `laee.geo.distance` can be used as dummies instead of a continuous density if chapter 02 provides no continuous measure.

**Worked implication.** At the prior, a 2-3x income difference within a city moves |ε| by only about 7-11% (0.10 × ln 2 to 0.10 × ln 3). Larger effects must come from data. Hoch et al. found that 11 demographic and competitive variables explain on average 67% of store-level elasticity variation across 83 stores [V], so the slopes are expected to be learnable once enough businesses exist.

**Minimum data.** At least 10 businesses per estimated covariate (Cochrane Handbook ch. 10 rule of thumb [L]) and at least 10 per tier dummy level. Below that, drop the covariate and keep industry (and country) intercepts only. Standardise continuous covariates on the training set and store the scaling in the snapshot.

**Outputs.** β̂, V_β, covariate scalers, list of active covariates, per-industry τ_ℓ.

**Failure modes.** Collinear inputs (industry vs business type; location vs income, density and competition): v1 §42 listed all of them; v2 keeps one taxonomy key and one variable per concept. Attenuated training estimates: handled by γ_idtier. Extrapolation to a new country or industry: τ_shift and parent backoff. Income proxies of poor quality in informal settlements: chapter 09 owns the source; if the proxy is missing, set the covariate to 0 (country median) and add its prior variance β_inc-SD² × E[(ln inc ratio)²] to the predictive variance [E].

**Cost.** One penalised GLS per snapshot: O(N·p² + p³) with p < 50.

#### 4.5.4 African and emerging-market evidence: how it enters

- Every African number available is a household-level, market-wide elasticity (QUAIDS or AIDS demand systems on survey data): Kenya, all food groups significantly negative (Korir, Rizov & Ruto 2020 [V]), rice, wheat and beans above unity in magnitude [V, source attribution to be confirmed]; Nigeria -0.63 to -1.79 (LSMS-ISA 2010-2016) [V]; South Africa soft drinks -1.18, concentrates -1.17, milk -1.1, sugar -2.42 (Stacey et al. 2017) [V].
- No peer-reviewed shop-level elasticity for kiosks, dukas, spazas or open markets was found (priors notes §4, gap). The platform's own data must supply that level.
- A kiosk's own-price elasticity for an identical branded item should be larger in magnitude than the household category elasticity because the next kiosk sells the same item. Informal credit ("book" accounts), proximity in dense settlements and small-pack daily purchasing push the other way. The net effect is unknown, so the LMIC staples override is wide: N(-2.0, 1.0²).
- Income effects (Slutsky ε_uncomp = ε_comp - w·η) depend on the budget share w. For one shop's item w is tiny, so the larger LMIC category magnitudes do not transfer as a multiplier. Income enters only through the weak within-country slope β_inc; cross-country level differences enter through the country node.
- Genuine Giffen behaviour exists but only for subsistence staples among very poor households at market level (Jensen & Miller 2008 [L]). A positive shop-level slope remains implausible (§4.9).

### 4.6 Learning population parameters without pooling raw data

#### 4.6.1 What crosses the tenant boundary

Only `StageOneSummary` rows (§5.2): b, s, s̃, n_obs, n_eff, K_pc, sd of residual ln p, identification tier, estimand id, estimation window, taxonomy keys, coarse location keys (country, region, market id, H3 resolution ≤ 7), and the covariates of §4.5.3. No transaction rows, no customer data, no prices or quantities. These summaries are written by a job that runs inside each tenant partition to a `population_summaries` store (§5.4).

For the Gaussian hierarchy these summaries lose nothing that matters: with business-specific intercepts and controls partialled out within the tenant, the pooled common-slope OLS is β̂ = Σ_i Sxx_i·b_i / Σ_i Sxx_i, which equals inverse-variance pooling of b_i when the residual variance is common [derivation, priors notes §5.2]. More generally, summary-statistic meta-analysis is asymptotically as efficient as individual-level analysis for common effects (Lin & Zeng 2010 [L]). One- and two-stage analyses can differ when within-unit samples are small (Burke, Ensor & Riley 2017 [L]), which is the SMB regime; this is why K_pc ≤ 3 records are excluded from hyperparameter fitting. Iterative federated averaging (McMahan et al. 2017 [L]) is not needed.

#### 4.6.2 The reference set

The platform's reference sales data (about 1.2 million rows across businesses and markets) seeds the first snapshot. It is processed exactly like tenant data: the stage-1 job runs per reference business in its own partition and writes summaries tagged `source = "reference"`. Two additions:

- a source indicator in the meta-regression (reference vs tenant), with prior N(0, 0.3²) [E], so that systematic differences between the reference population and the tenant population are estimated rather than absorbed into the industry intercepts;
- the reference set's licence and provenance are owned by chapter 09. If any reference rows come from a tenant, that tenant's isolation rules apply to them.

#### 4.6.3 Snapshot job (monthly, versioned)

```
def build_snapshot(summaries, lit_table, window_months=24):        # window [E]
    S = admissible(summaries, window)            # §4.1; K_pc ≥ 4, not suspect, estimand canonical
    for industry in INDUSTRY_IDS:
        S_ind = S[industry]
        # 1. starting values: DL/PM per level from one-level fits; β from penalised GLS with τ at prior medians
        # 2. penalised ML-II over (ln τ_C, ln τ_R, ln τ_M, ln τ_B, ln τ_P) and β via tree log-likelihood (§4.3),
        #    alternating: β-step (penalised GLS given τ), τ-step (L-BFGS-B given β); stop when Δ log-post < 1e-6
        # 3. grid posterior π_g over τ_B holding other τ at their modes (§4.4.4)
        # 4. upward messages for every node; per-cell aggregates per tenant: Σw, Σw·b̃, n, tenant ids
        # 5. release gates (§6.3): LOTO calibration, prior predictive checks; if failed, inflate τ_B (§4.8) and re-gate
    return Snapshot(version, created_at, window, per-industry hyperparameters, aggregates, gate results)
```

Snapshots are immutable and every served estimate records its snapshot id (v1 Rule 11). The rolling 24-month window [E] lets hyperparameters drift with regimes (chapter 04 owns within-business time variation).

#### 4.6.4 Leave-one-tenant-out (LOTO) serving

Tenant T's prior must not contain T's own data (double counting, §4.2; and privacy, §4.6.5). Serving rule:

1. **Cell aggregates.** For every node on T's path, subtract T's contribution from the stored upward message. Because messages are sums of precisions and precision-weighted means, this is O(1) per node: P_v^(-T) = P_v - Σ_{c ∈ T} 1/Ã_c, S_v^(-T) = S_v - Σ_{c ∈ T} a_c/Ã_c.
2. **Global hyperparameters (τ_ℓ, β).** Reuse the snapshot values if T's businesses hold less than 1% of the level's total precision weight [E]; otherwise refit LOTO for that industry (cached per tenant per snapshot).
3. **Own data.** T's businesses are then attached as children of the LOTO node posteriors, and the downward pass produces their posteriors. All of T's businesses are T's own data and may be used jointly.

By the leave-one-out identity this equals the joint posterior when hyperparameters are held fixed.

#### 4.6.5 Disclosure control

The posterior computation may use any cell. What is *displayed* or *exported* is gated:

- **k_min.** A cell-level prior or benchmark is shown only if, after LOTO, it rests on ≥ 10 businesses from ≥ 5 distinct tenants [E, priors notes]. Otherwise show the parent level ("benchmark from similar businesses in Kenya" rather than "in your market"). Reason: in a two-business cell, B's LOTO prior is A's elasticity, a competitor's trade secret.
- **Dominance.** If one tenant holds more than 50% of a displayed cell's precision weight Σw, cap its weight at 50% for the released aggregate or suppress the cell (an (n, k)-dominance rule, Hundepool et al. 2012 [L]).
- **Differencing.** Released aggregates change only at monthly snapshots, are rounded to 0.1 and shown as ranges; a cell whose membership changed by exactly one business since the last release is not re-released without noise.
- **Unit of privacy = tenant.** A tenant with several businesses in a cell contributes one averaged value to any released statistic.

#### 4.6.6 Optional differential privacy (external benchmarks and partner exports only)

Internal serving does not use DP noise; the platform already processes tenant data and the k_min/dominance rules apply. For any release outside the platform (public benchmark pages, partner exports):

```
equal-weight clipped mean over n tenants in a cell:
  clip each tenant's b to [L, U] = [-6, 1];  n treated as public; replace-one adjacency
  sensitivity Δ = (U - L)/n
  release mean + Laplace(Δ/ε_dp)          # SD = √2·(U - L)/(n·ε_dp)
```

Laplace mechanism (Dwork et al. 2006 [L]); post-processing and composition (Dwork & Roth 2014 [L]); library OpenDP [L]. Noise SD [C]:

| n tenants | ε_dp = 0.5 | ε_dp = 1 | ε_dp = 2 |
|---|---|---|---|
| 10 | 1.98 | 0.99 | 0.50 |
| 20 | 0.99 | 0.50 | 0.25 |
| 50 | 0.40 | 0.20 | 0.10 |
| 100 | 0.20 | 0.10 | 0.05 |
| 500 | 0.04 | 0.02 | 0.01 |

Defaults: ε_dp = 1 per cell per release, external release only for n ≥ 50 [E, priors notes]. Below about 20 tenants the noise (SD ≥ 0.5) is comparable to the between-business SD (about 0.7) and the release is useless. Precision-weighted DP means use a noisy numerator Σ w̄(b̄ - m_lit) with w̄ = min(w, w_cap) (sensitivity w_cap·max|b̄ - m_lit|) over a noisy denominator Σ w̄ (sensitivity w_cap), splitting ε_dp between them. For scalar releases Laplace beats the Gaussian mechanism (multiplier √2 vs √(2 ln(1.25/δ)) = 5.30 at δ = 1e-6 [C]). Track cumulative ε_dp per cell (sequential composition). DP for τ² releases is not specified (§7). Legal basis for cross-tenant learning (Kenya Data Protection Act 2019, Nigeria NDPA 2023, South Africa POPIA; names recalled, not verified) is chapter 09's scope.

**Failure modes.** Tenant leakage through small cells; leakage through cached LOTO values computed with the wrong tenant id; differencing across versions. **Cost.** Negligible.

### 4.7 Cold start and posterior speed

#### 4.7.1 Cold start

A business-product with no admissible stage-1 record gets the prior predictive from the tree downward pass (or, before the first snapshot, the §4.5.2 table): N(M_parent, V_parent + τ_ℓ²). Its output is labelled `prior_only` (§4.10).

Order of fallbacks for the parent node: product's business node (new product in a known business) -> market -> region -> country -> industry root. The tree does this automatically; the display rule of §4.6.5 decides which level is named to the user.

#### 4.7.2 How fast the posterior should move

The own-data weight is w = τ²/(τ² + s²), and for i.i.d. daily noise with exogenous price variation s ≈ σ_e/(√n·sd(ln p)). With τ = 0.7 [C, priors notes]:

| σ_e | sd(ln p) | n = 30 | n = 90 | n = 180 | n = 365 |
|---|---|---|---|---|---|
| 0.3 | 0.03 | s 1.83, w 0.13 | 1.05, 0.31 | 0.75, 0.47 | 0.52, 0.64 |
| 0.3 | 0.05 | 1.10, 0.29 | 0.63, 0.55 | 0.45, 0.71 | 0.31, 0.83 |
| 0.3 | 0.10 | 0.55, 0.62 | 0.32, 0.83 | 0.22, 0.91 | 0.16, 0.95 |
| 0.5 | 0.03 | 3.04, 0.05 | 1.76, 0.14 | 1.24, 0.24 | 0.87, 0.39 |
| 0.5 | 0.05 | 1.83, 0.13 | 1.05, 0.31 | 0.75, 0.47 | 0.52, 0.64 |
| 0.5 | 0.10 | 0.91, 0.37 | 0.53, 0.64 | 0.37, 0.78 | 0.26, 0.88 |

(n = days of usable data.) Consequences:

- Doubling price variation is worth quadrupling the observations.
- Halving the prior SD needs s = SD_prior/√3. For prior SD 0.8 and σ_e = 0.4: 300 days at ±5% variation or 75 days at ±10%; for prior SD 1.0: 192 or 48 days [C].
- Value of an experiment (chapter 07 and v1 §77-78), prior SD 0.8, σ_e = 0.4, alternating prices [C]: ±5% for 4/8/12 weeks gives posterior SD 0.707/0.641/0.590; ±10% gives 0.549/0.444/0.383.

The σ_e values are illustrative. The snapshot job must compute actual residual SDs of daily log quantity per industry from the reference set and publish this table with platform values (§7).

#### 4.7.3 Independent price changes

With serially correlated sales, the effective sample size for ε is about the number of independent price spells, not days. Rules (applied in §4.1):

| K_pc | Own posterior | Enters hyperparameter fit | Label floor |
|---|---|---|---|
| 0, or sd(residual ln p) < 0.02 | none; prior only | no | `prior_only`, message "not measurable yet" |
| 1 | none; prior only | no | `prior_only` |
| 2-3 | yes, with s² × 3 | no | at most `prior_dominated`; text "early signal - not used for price recommendations" |
| ≥ 4 | yes, with s² × ν/(ν - 2), ν = K_pc - 1 | yes | by own-data share |

The K_pc = 1 exclusion is [E]: a single change identifies a before/after contrast that is confounded with any concurrent shock.

#### 4.7.4 Time drift (optional)

To keep the posterior responsive after regime changes, the business effect may follow a random walk θ_i,t = θ_i,t-1 + η_t, η ~ N(0, q²), q = 0.05 per month [E], implemented as a Kalman predict step that adds q² × months-since-last-update to the business node's variance before the update. Chapter 04 owns time-varying elasticity; if chapter 04 already supplies a time-varying stage-1 estimate, do not also apply drift here.

**Cost.** O(1).

### 4.8 Prior-data conflict and prior predictive checks

#### 4.8.1 Per-estimate conflict check

```
z_i = (b_i - m_i) / √(v_i + s̃_i²)          m_i, v_i = LOTO prior mean and variance
p_i = 2 Φ(-|z_i|)                           (Evans & Moshonov 2006 [L], normal case)
```

If p_i < 0.01, switch that estimate to the robust MAP mixture prior (Schmidli et al. 2014 [V]):

```
π(ε) = (1 - w_v)·N(m_i, v_i) + w_v·N(m_parent, 2.0²),   w_v = 0.2 [E]
posterior, closed form, for each component k:
  w_k' ∝ w_k · N(b_i ; m_k, v_k + s̃_i²)
  v_k' = 1/(1/v_k + 1/s̃_i²);   m_k' = v_k'·(m_k/v_k + b_i/s̃_i²)
mean = Σ w_k' m_k';  var = Σ w_k' (v_k' + m_k'²) - mean²;  intervals: exact mixture quantiles (brentq)
```

m_parent is the posterior mean of the node one level above the prior's dominating level. As |b - m| grows, the posterior weight moves monotonically to the vague component and the posterior mean approaches b; this is a required test. The mixture weight has no settled quantification method (modified rMAP, medRxiv 2023 [V]); w_v = 0.2 and the 0.01 threshold are tuned by walk-forward backtests (chapter 08).

Conflict does not remove the record from the user's posterior, but a record with p_i < 0.001 [E] is excluded from the next hyperparameter fit until reviewed (it is either a mis-coded estimand or a genuinely unusual business).

#### 4.8.2 Per-level diagnostics (logged at each snapshot)

- Shrinkage factor B and own-data share per estimate; posterior/prior SD ratio; the dominating prior level (the level whose message contributes the largest precision to the cavity prior).
- Pooling factor per level (Gelman & Pardoe 2006 [L]): λ_ℓ = 1 - Var_k(E[θ_k | y]) / E_k[Var(θ_k | y)]. λ ≈ 0 means no pooling, λ ≈ 1 complete pooling. Alert if λ_ℓ > 0.95 in every cell of a level: the level is uninformative and may be dropped.
- τ̂_ℓ with its posterior SD or Q-profile interval; fraction of cells with one-level τ̂ = 0; I² within cells.
- Alert if the grid posterior for τ_B piles up in the upper tail of its HalfNormal prior (posterior median > 2σ_B): the normal model is misspecified (heavy tails, a mixture of niche and competitive shops).

#### 4.8.3 Prior predictive checks (release gates for each snapshot)

Run per industry on the snapshot's LOTO priors (Gabry et al. 2019 [L]; Gelman et al. 2020 [L]):

1. **Calibration (the main gate).** For each held-out business j, z_j = (b_j - m_j)/√(v_j + s̃_j²). Require, for industries with ≥ 200 businesses, mean(z) ∈ [-0.1, 0.1], SD(z) ∈ [0.9, 1.1], and 80%/95% coverage within ±2 percentage points of nominal. SD(z) > 1.1 means the prior is overconfident: inflate τ_B by the factor that brings SD(z) to 1.0 and re-gate. SD(z) < 0.9 means too wide. Also report by country, income tercile and identification tier. Industries with fewer than 200 businesses report the statistics without gating and keep the label confidence "low".
2. **Revenue-direction share.** Simulate ε from each business's prior; the share with P(ε > -1) (price rise increases revenue) is compared with the share observed among Tier 1-3 estimates; a gap above 0.15 [E] is flagged.
3. **Lerner check.** For businesses that report margins, compare -1/m with the prior median. A gap above 1.5 elasticity units [priors notes] flags a mis-centred prior or systematic under-pricing. Informational, not a gate.
4. **Prior mass above zero.** Platform-fitted priors must satisfy P(ε > 0) ≤ 0.06 for goods retail (grocery, apparel, electronics, wholesale, ecommerce) and ≤ 0.15 for food service, pharmacy, services and other [E]. The priors notes proposed 0.05 for retail; the literature seed rows for apparel and LMIC staples are at 0.051, so this chapter sets the goods-retail cap at 0.06 and places pharmacy (seed 0.129) with services. A failing industry keeps its previous snapshot.
5. **Simulation-based calibration** (Talts et al. 2018 [L]) of the whole pipeline on synthetic hierarchies (§6.1).

**Cost.** O(N) per snapshot.

### 4.9 Near-zero and wrong-sign estimates

**Causes, most to least likely in SMB data** (priors notes §6.5): price endogeneity (prices set in anticipation of demand; the attenuation Bijmolt et al. document [V]); omitted promotions, advertising or seasonality; stockouts or censoring (chapter 01); unit-price measurement error across pack sizes (attenuates toward 0); simultaneous quality or assortment changes; genuine prestige effects (rare); Giffen behaviour (market-level subsistence staples only, Jensen & Miller 2008 [L]).

**Rules.**

1. Never truncate or sign-flip b_i before pooling. Truncation biases μ̂ upward in magnitude and τ̂ downward.
2. Pool on the raw scale. The hierarchy pulls a noisy positive toward the negative parent automatically: in the priors notes' tree example a raw +0.4 (SE 1.2) became -1.24 (posterior SD 0.65) [C].
3. Report P(ε > 0) and P(ε > -1) from the posterior (exact mixture CDF when §4.4.4 or §4.8 applies). If P(ε > 0) > 0.10, mark the estimate `identification_suspect`: chapter 07 recommends no price change and proposes an experiment.
4. A positive estimate is treated as real only if it comes from Tier 1-2 evidence, replicates in a second independent window or experiment, and the rMAP vague component carries it (p_i < 0.01 with posterior weight on the vague component > 0.5). Then the flag is lifted.
5. Records with stage-1 b > 0 and Tier ≥ 4 are excluded from hyperparameter fitting, or, if the industry has fewer than 50 admissible records, kept with δ_tier = 0.6 [priors notes].
6. If the posterior mean satisfies -1 < E[ε] < 0, a profit optimiser pushes price to its bound. Chapter 07 must cap per-cycle changes (v1 §66) and require experimental confirmation; this chapter only supplies P(ε > -1).
7. **Sign constraints only at the decision layer.** If chapter 07 needs a sign-respecting distribution, use the posterior truncated to ε ≤ 0 (renormalised) or a lognormal on -ε fitted by moments for display. The pooling model stays Gaussian so that it stays closed form and its hyperparameters stay unbiased. Truncated hierarchical priors (Boatwright, McCulloch & Rossi 1999 [L]) are an offline MCMC variant only.

**Near-zero but negative** estimates with tight SEs (for example -0.3 ± 0.1 at Tier 5-6) are not flagged automatically; the conflict check (§4.8) catches them if they are far from the prior. Chapter 03's endogeneity lead test should cap their tier.

### 4.10 Labelling prior-only and prior-dominated numbers

**Own-data share.** w_own = 1 - V_post/V_prior, where V_prior is the LOTO prior variance of the node and V_post the posterior variance after adding the tenant's own data. It equals 1 - B in the one-level case and is well defined in the tree and for the mixture (use the mixture variances).

**Labels** (thresholds [E]; the 0.2 boundary matches chapter 04's "prior share B > 0.8 means mostly industry prior"):

| Label | Condition | Shown to user as | Decision guardrail (chapter 07) |
|---|---|---|---|
| `prior_only` | no admissible own record | "Benchmark from N similar businesses in <displayed level>; not yet measured for your business" | no recommendation beyond an exploratory step of at most 5%; confidence capped at "low" |
| `prior_dominated` | w_own < 0.2, or K_pc ∈ {2, 3} | "Mostly based on similar businesses (X% your data)" | same as prior_only |
| `blended` | 0.2 ≤ w_own < 0.8 | "Based X% on your own data" | normal rules |
| `data_dominated` | w_own ≥ 0.8 | "Based mainly on your own data" | normal rules |

Overlay flags, any label: `identification_suspect` (§4.9), `prior_data_conflict` (§4.8), `extrapolated` (τ_shift applied), `early_signal` (K_pc ∈ {2, 3}).

**Example text** (priors notes §6.1): "Estimated price sensitivity -1.8 (likely range -3.5 to -0.1). Benchmark from 46 similar cafes in Nairobi; not yet measured for your business. A ±10% price test over 8 weeks would cut this range by about 44%." The range is the 90% posterior interval; the N is the LOTO business count of the displayed level; the experiment line is computed with §4.7.2 using the platform's σ_e.

**Never** present a prior-only number as "your elasticity", and never let a prior-only or prior-dominated estimate raise the v1 §45 confidence above "low". Chapter 04 owns the confidence definitions; this chapter supplies w_own and the label.

**Identification tier for prior-only output.** Proposed `identification_tier = 0` ("benchmark: no own-data identification"). The research notes conflict here: chapter 04's notes put prior-only under Tier 7, the priors notes propose a separate Tier 0. Tier 7 means "pure predictive correlation" in the tenant's own data, which a prior-only number does not have, so this chapter proposes 0 pending chapter 00's contract (§7). Posteriors keep the stage-1 tier: pooling improves precision, not identification.

### 4.11 Offline full-Bayes validation (not in the serving path)

Monthly, on a 5% sample of summaries per industry [E], fit the same hierarchy by NUTS in NumPyro or PyMC with a non-centred parameterisation (Betancourt & Girolami 2015 [L]) and compare posterior means and SDs with the closed form (§6.2). The same offline harness fits variants that need MCMC: Student-t business effects (heavy tails), truncated sign-constrained priors, joint uncertainty in s_i, crossed effects, and a two-class "niche vs competitive" mixture. A variant is promoted only if it beats the closed form on LOTO calibration and on chapter 08's walk-forward metrics.

```python
# NumPyro sketch from the priors notes (not executed: numpyro is not installed)
def model(b, s, cell, n_cells):
    mu = numpyro.sample("mu", dist.Normal(-1.8, 0.8))
    tau = numpyro.sample("tau", dist.HalfNormal(0.7))
    with numpyro.plate("cells", n_cells):
        z = numpyro.sample("z", dist.Normal(0.0, 1.0))
    theta = mu + tau * z                       # non-centred
    numpyro.sample("b", dist.Normal(theta[cell], s), obs=b)
```

### 4.12 Cost summary

| Operation | Where | Complexity | Indicative time |
|---|---|---|---|
| One-level update (§4.2) | online | O(1) | microseconds |
| Downward path with τ_B grid (§4.3-4.4) | online, per business | O(G·(depth + products)), G = 700 | < 10 ms [E, estimate] |
| LOTO subtraction (§4.6.4) | online, per tenant | O(depth × tenant businesses) | microseconds |
| Tree log-likelihood (§4.3) | offline | O(N) | about 3.6 ms per level per 200k leaves [C, re-run, indicative] |
| Penalised ML-II (§4.4) | offline, per industry | 100-300 likelihood evaluations [E] | seconds |
| τ grid posterior | offline | G × O(N) | seconds |
| LOTO calibration gate | offline | O(N) with aggregates | seconds |
| NUTS validation (§4.11) | offline, sampled | MCMC | minutes to hours |

---

## 5. Interfaces

### 5.1 Module layout (new package)

```
src/laee/priors/__init__.py          public API re-exports
src/laee/priors/contracts.py         StageOneSummary, PriorSpec, PriorSnapshot, PosteriorDiagnostics, labels
src/laee/priors/admissibility.py     effective variance, K_pc rules, tier bias (§4.1)
src/laee/priors/normal_normal.py     one-level update, GLS mean, grid-over-τ posterior, Kass-Steffey (§4.2, §4.4.4)
src/laee/priors/variance.py          tau2_dl, tau2_pm, tau2_reml, tau2_ml, tau2_reml_metareg, q_profile_ci,
                                     hksj_interval, prediction_interval, morris_factor, james_stein (§4.4)
src/laee/priors/tree.py              HierarchyTree, upward, downward, log_marginal_likelihood,
                                     fit_hyperparameters (penalised ML-II) (§4.3)
src/laee/priors/literature.py        INDUSTRY_ROOT_PRIORS, LMIC_STAPLES_OVERRIDE, SLOPE_PRIORS (§4.5.2-4.5.3)
src/laee/priors/meta_regression.py   fit_meta_regression, predict_prior (§4.5.3)
src/laee/priors/population.py        build_snapshot, loto_prior, summary store access (§4.6)
src/laee/priors/disclosure.py        k_min/dominance checks, laplace_release (§4.6.5-4.6.6)
src/laee/priors/conflict.py          conflict_pvalue, rmap_update, pooling_factor, prior_predictive_checks (§4.8)
src/laee/priors/posterior.py         apply_prior (main entry), prior_only_estimate, labels (§4.9-4.10)
```

The package imports `laee.types` (ElasticityEstimate, Interval), `laee.geo.market.INDUSTRY_IDS` and `laee.schema`; it does not import `laee.models`. Estimators in `laee.models` do not import `laee.priors`: the prior is applied after estimation, never inside it.

### 5.2 Data contracts (`contracts.py`)

```python
@dataclass(frozen=True)
class StageOneSummary:
    tenant_id: str
    business_id: str
    product_id: str | None              # None for a business-level record
    estimand_id: str                    # e.g. "own_price/product_at_shop/regular/4-12w"
    b: float                            # data-only estimate (ElasticityEstimate.value)
    s: float                            # data-only SE (ElasticityEstimate.std_error)
    bias_variance_included: bool        # True if s already includes tier-bias variance (chapter 04 combiner)
    method: str                         # ElasticityEstimate.method, e.g. "dml_plr", "twfe_ols"
    identification_tier: int            # 1..7
    n_obs: int
    n_eff: float | None
    k_pc: int                           # independent price changes (chapter 04 definition)
    sd_resid_log_price: float
    window: tuple[str, str]             # ISO dates
    industry_id: str                    # validated by laee.geo.market.validate_industry_id
    sub_industry: str | None
    product_category: str | None
    country: str                        # ISO 3166-1 alpha-2
    region_id: str
    market_id: str | None               # None for ecommerce
    h3_r7: str | None
    covariates: dict[str, float]        # ln_income_ratio, comp_z, dens_perp, cpi_relative, ...
    source: str                         # "tenant" | "reference" | "experiment"
    product_cov: tuple[tuple[float, ...], ...] | None = None   # optional joint covariance S_i

@dataclass(frozen=True)
class PriorSpec:
    mean: float
    var: float
    level: str                          # "product" | "business" | "market" | "region" | "country" | "industry"
    n_businesses_loto: int
    n_tenants_loto: int
    displayable: bool                   # k_min and dominance rules passed at this level
    snapshot_id: str
    mixture: tuple[tuple[float, float, float], ...] | None = None   # (weight, mean, var) if rMAP

@dataclass(frozen=True)
class PriorSnapshot:
    snapshot_id: str
    created_at: datetime
    window: tuple[str, str]
    estimand_id: str
    per_industry: dict[str, "IndustryHyperparameters"]   # τ_ℓ, β̂, V_β, scalers, τ_B grid and weights
    gate_results: dict[str, dict[str, float]]
    code_version: str
```

Conversion from an estimator output: `StageOneSummary.from_estimate(est: ElasticityEstimate, *, keys, k_pc, ...)` reads `value`, `std_error`, `method`, `identification_tier`, `n_obs`, and `diagnostics["sd_demeaned_log_price"]` (FE) or the residual price SD reported by DML. The DML estimator currently reports `residual_price_variance_share` but not the residual SD; chapter 03 must add `sd_resid_log_price` to both estimators' diagnostics for this contract (§7).

### 5.3 Main entry points (`posterior.py`)

```python
def apply_prior(
    stage1: ElasticityEstimate | None,
    summary: StageOneSummary | None,
    keys: "PriorKeys",                  # industry, country, region, market, business, product, covariates
    snapshot: PriorSnapshot,
    *,
    levels: tuple[float, ...] = (0.80, 0.90, 0.95),
) -> ElasticityEstimate: ...

def prior_only_estimate(keys: "PriorKeys", snapshot: PriorSnapshot, *, levels=(0.80, 0.90, 0.95)) -> ElasticityEstimate: ...
```

Returned `ElasticityEstimate`:

| Field | Value |
|---|---|
| `value` | posterior mean (prior mean if prior-only) |
| `std_error` | posterior SD |
| `intervals` | `Interval(lower, upper, level)` per level; exact mixture quantiles when a τ grid or rMAP mixture applies, otherwise normal quantiles |
| `method` | `"hier_posterior"` or `"prior_only"` |
| `identification_tier` | stage-1 tier; 0 for prior-only (proposed) |
| `n_obs`, `n_products` | from stage-1; 0 and 1 for prior-only |
| `diagnostics["prior"]` | see below |
| `diagnostics["stage1"]` | the stage-1 value, std_error, method and diagnostics, unchanged |

`diagnostics["prior"]` keys (all required):

```
snapshot_id, estimand_id, prior_mean, prior_sd, prior_level, prior_displayable,
n_businesses_loto, n_tenants_loto, s_effective, k_pc, shrinkage_B, own_data_share,
label, flags, conflict_z, conflict_p, rmap_applied, rmap_vague_weight_post,
p_positive, p_inelastic (P(ε > -1)), tau_used (dict level -> value or "grid"),
meta_regression_mean, meta_regression_var, extrapolated, products_assumed_independent
```

### 5.4 Storage

- `population_summaries`: one row per `StageOneSummary`, partitioned by industry and snapshot month. Written only by the per-tenant stage-1 job. Contains no transaction-level fields.
- `prior_snapshots`: immutable snapshot objects keyed by `snapshot_id`; referenced from `ModelVersionRecord.model_version` of every served estimate (v1 Rule 11; chapter 08 owns versioning).
- `prior_cell_aggregates`: per snapshot, per node, per tenant: Σ 1/Ã, Σ a/Ã, n_businesses; enables O(1) LOTO.

Access control for these stores is chapter 09's scope; the rule here is that serving code for tenant T reads only aggregates and hyperparameters, never other tenants' summary rows.

### 5.5 Cross-chapter contracts

| Chapter | This chapter needs | This chapter provides |
|---|---|---|
| 00 | tier numbering incl. prior-only; estimand ids | labels, diagnostics keys |
| 01 | estimand fields, promo/stockout exclusion, staple category list, CPI deflation | - |
| 02 | primary market id (versioned), comp_z, density, competitive price index | - |
| 03 | data-only (b, s), clustered SE, `sd_resid_log_price`, lead-test tier caps | priors for any estimator that needs one |
| 04 | K_pc definition; data-only combiner output with `bias_variance_included`; time-varying stage-1 | prior mean/variance for its combiner and Kalman initialisation if needed (without re-applying here) |
| 06 | - | category-level LMIC scaling rule (§4.5.2) |
| 07 | - | posterior, P(ε > 0), P(ε > -1), label, guardrail flags, experiment value table |
| 08 | walk-forward backtests for w_v, thresholds, q | release-gate statistics |
| 09 | legal basis, income sources, reference-set provenance | disclosure rules for released aggregates |

---

## 6. Validation

### 6.1 Known-answer simulations

**S1. One-level hierarchy.** θ_i ~ N(-2, τ²), s_i ~ U(0.3, 1.2), b_i ~ N(θ_i, s_i²); k ∈ {4, 8, 30, 100}, τ ∈ {0.5, 1.0}; ≥ 1,500 replications per cell. Criteria:
- oracle-τ 95% coverage within 0.950 ± 0.006 (binomial tolerance at 1,500 × k draws);
- grid-over-τ 95% coverage in [0.925, 0.970] for every cell, and ≥ 0.94 for k = 8, τ = 0.5 with exact quantiles;
- grid mean interval width at least 20% below no-pooling width for k ≥ 8;
- posterior MSE of θ below the no-pooling MSE in every cell.

**S2. Five-level tree.** Simulate industry -> 3 countries -> 4 regions -> 5 markets -> 4-10 businesses -> 1-8 products with known τ_ℓ = (0.3, 0.3, 0.3, 0.6, 0.5) [E] and s from the §4.7.2 table. Criteria: penalised ML-II recovers τ_B within ±25% (median over 200 replications); posterior 90% coverage per level in [0.87, 0.93]; tree means and variances equal dense GLS to 1e-10 on every replication with ≤ 500 nodes.

**S3. Cold start.** Delete all data for 10% of businesses; their prior predictive 80% and 95% intervals cover the true θ at nominal ±2 pp.

**S4. Precision dependence.** Make θ_i depend on ln s_i (slope 0.3). Without β_s the conflict rate rises; with β_s the slope is recovered within ±0.1 and calibration passes.

**S5. Attenuated training estimates.** Generate Tier 6 records with bias +0.4 toward zero and Tier 1 records unbiased. With γ_idtier the predicted prior mean at the best tier is within ±0.1 of the truth; without it the bias appears in the cold-start prior.

**S6. Heavy tails.** Draw θ_i from a t₃ with the same SD. Report coverage (expected to drop); the τ_B tail alert (§4.8.2) and the rMAP switch must fire on ≥ 50% of the outlying businesses (|θ - μ| > 3τ).

**S7. Simulation-based calibration** (Talts et al. 2018 [L]). Draw hyperparameters from their hyperpriors, then θ, then (b, s); run the full pipeline; ranks of true θ among 100 posterior draws, 1,000 replications: χ² uniformity p > 0.01; 80% intervals within their binomial band.

### 6.2 Unit tests with golden values

| Test | Expected |
|---|---|
| DL on BCG | τ² = 0.3088, Q = 152.2330, μ̂ = -0.7141, SE 0.1787 (tol 1e-4) |
| PM on BCG | τ² = 0.3181, μ̂ = -0.7150, SE 0.1809 |
| REML on BCG | τ² = 0.3132, μ̂ = -0.7145, SE 0.1798; Q-profile CI [0.1197, 1.1115]; ML τ² = 0.2800 |
| HKSJ and PI on BCG | HKSJ SE 0.1808; PI(t) [-2.0084, 0.5793]; PI(z) [-1.8667, 0.4376] |
| Eight schools | Q = 4.707 (df 7, p = 0.696); DL = PM = REML = ML τ² = 0; pooled mean 7.686 (SE 4.072) |
| Eight schools grid (uniform on τ, flat μ, grid [0.0001, 40], 4,000 points) | E[τ \| y] = 6.52; θ means [11.38, 7.89, 6.14, 7.65, 5.13, 6.14, 10.66, 8.45]; SDs [8.31, 6.27, 7.75, 6.54, 6.35, 6.70, 6.78, 7.87] (tol 0.03) |
| James-Stein, Efron-Morris baseball (18 players, 45 at-bats; hits [18, 17, 16, 15, 14, 14, 13, 12, 11, 11, 10, 10, 10, 10, 10, 9, 8, 7]) | shrinkage c = 0.2117 on raw proportions (published 0.212); arcsine version 0.2090 |
| Kass-Steffey derivative | 0.36519849 at b = -0.4, s² = 0.9, μ = -2, τ² = 0.49 |
| Tree vs dense GLS | max abs difference < 1e-10 in means and variances |
| Tree log-likelihood vs `multivariate_normal.logpdf` | equal to 1e-10 (priors notes fixture: -12.812375363501369) |
| Δτ weights | non-uniform grid with Δτ equals a fine uniform grid to 1e-3 in posterior mean; removing Δτ must change the answer (regression guard) |

BCG input: 13 trials (tpos/tneg/cpos/cneg) as listed in the priors notes §2.9; log RR = ln[(tpos/(tpos + tneg))/(cpos/(cpos + cneg))], v = 1/tpos - 1/(tpos + tneg) + 1/cpos - 1/(cpos + cneg).

### 6.3 Invariance and property tests

- Limits: s -> 0 returns b; s -> ∞ returns the prior; τ -> 0 returns the pooled mean.
- Adding a constant c to all b and to m_root shifts every posterior mean by c and leaves variances unchanged.
- Permuting businesses, products or tenants leaves every posterior unchanged.
- Duplicating a stage-1 record must be rejected by admissibility (one record per business-product-estimand), not silently double the precision.
- **No double counting:** serving tenant T with the joint tree equals serving with its LOTO prior plus own update, to 1e-10, with hyperparameters fixed.
- **Isolation:** deleting all of tenant T's summaries and rebuilding leaves T's served prior identical (bit-identical for cell aggregates; for global hyperparameters identical when the §4.6.4 1% rule triggers a LOTO refit, otherwise within 1e-3 in mean and variance).
- **Exactness of summaries:** simulated OLS businesses with common residual variance; summary pooling equals raw-data pooled OLS to 1e-10.
- **DP sensitivity:** brute-force replace-one neighbours over random datasets; max |Δ output| ≤ Δ.
- **Suppression:** no cell with n_LOO < 10 businesses or < 5 tenants appears in any export payload.
- **rMAP monotonicity:** as |b - m| increases, the vague-component posterior weight increases monotonically and the posterior mean approaches b.
- **Sign handling:** a raw positive estimate with large s pools to a negative posterior mean; truncation is never applied before pooling (assert raw b preserved in summaries).
- **Labels:** w_own computed from variances equals 1 - B in the one-level case; label boundaries at 0.2 and 0.8 are inclusive as specified.

### 6.4 Release criteria for a snapshot

A snapshot is promoted only if, for each industry with ≥ 200 admissible businesses: the §4.8.3 calibration gate passes (mean z in [-0.1, 0.1], SD z in [0.9, 1.1], 80%/95% coverage ±2 pp); the P(ε > 0) caps hold; the NUTS comparison (§4.11) agrees within 0.05 in posterior mean and 10% in posterior SD for ≥ 95% of sampled businesses [E]; and chapter 08's walk-forward backtest of posterior predictions on held-out price-change events is not worse than the previous snapshot by more than its tolerance. Industries below 200 businesses inherit the previous snapshot's gate status and stay labelled low confidence.

---

## 7. Open questions

1. **Tier for prior-only output.** This chapter proposes `identification_tier = 0`; chapter 04's notes use Tier 7. Chapter 00 must decide and update `ElasticityEstimate` documentation.
2. **One bias-variance table.** δ_tier here (0/0.2/0.4/0.6) and chapter 04's combiner scale (0/0.1/0.2/0.3/0.6) differ and both are uncalibrated. They should be unified and calibrated by experiments (Tier 1 vs observational estimates on the same products).
3. **τ hyperprior scales and τ_seed = 0.7.** No published between-business SD of SMB elasticities was found in any region. Replace by the first REML / ML-II fit on the reference set and report the fitted values here.
4. **Platform σ_e.** The §4.7.2 table uses illustrative daily noise 0.3-0.5. Compute per-industry residual SDs from the reference set before fixing the K_pc and label thresholds.
5. **Industry rows without evidence.** hair_salon, auto_repair, professional_service, other_service, wholesale, apparel, electronics and other have no verified shop-level evidence. Targeted literature search (category elasticities for apparel, OTC, services, e-commerce; Ghana demand systems; African fuel and transport fares) was not completed by the research.
6. **Certify golden values against metafor** (BCG REML CI, ML τ²) and verify Morris's general (k - r - 2)/(k - r) factor, Blattberg & George (1991) and Montgomery & Rossi (1999), which the priors notes recalled but did not verify.
7. **rMAP weight and conflict threshold.** w_v = 0.2 and p < 0.01 have no settled method; tune by walk-forward backtests (chapter 08).
8. **Robust EB intervals.** Armstrong, Kolesár & Plagborg-Møller (2022) robust EBCIs could replace the grid when normality of business effects fails; the critical-value formula was not accessed.
9. **DP for variance components.** A clipped noisy sum of squares for τ² releases is not worked out; utility at small n is expected to be poor.
10. **Crossed effects.** Whether region-wide random effects across industries add accuracy beyond covariates needs an offline comparison (Henderson equations or MCMC).
11. **Diagnostics from chapter 03.** DML reports `residual_price_variance_share` but not `sd_resid_log_price`; FE reports `sd_demeaned_log_price`. A single key is needed.
12. **Legal basis** for using tenant summaries in shared priors (data processing agreements, opt-in, African data-protection law) is chapter 09's and must be settled before tenant summaries enter a snapshot. Until then, snapshots use the reference set only.

---

## 8. References

URLs as recorded in the research notes. Tags as defined at the top.

Hierarchical models, empirical Bayes, τ² estimation
- [L] Gelman et al., Bayesian Data Analysis, 3rd ed. http://www.stat.columbia.edu/~gelman/book/
- [L] James & Stein 1961. https://projecteuclid.org/euclid.bsmsp/1200512173
- [L] Efron & Morris 1973, JASA. https://doi.org/10.1080/01621459.1973.10481350
- [L] Efron & Morris 1975, JASA 70:311-319. https://doi.org/10.1080/01621459.1975.10479864
- [V] Morris 1983, JASA 78(381):47-55. https://pmc.ncbi.nlm.nih.gov/articles/PMC2872278/ ; https://errorstatistics.com/wp-content/uploads/2015/11/morris.pdf
- [L] Kass & Steffey 1989, JASA 84:717-726. https://doi.org/10.1080/01621459.1989.10478825
- [L] DerSimonian & Laird 1986, Control Clin Trials 7:177-188. https://doi.org/10.1016/0197-2456(86)90046-2
- [L] Paule & Mandel 1982, J Res NBS 87:377-385. https://doi.org/10.6028/jres.087.022
- [V] Veroniki et al. 2016, Res Synth Methods. https://onlinelibrary.wiley.com/doi/full/10.1002/jrsm.1164
- [V, existence] τ² estimator simulations: https://arxiv.org/pdf/1904.01948 ; https://arxiv.org/pdf/1902.07154
- [L] Langan et al. 2019, Res Synth Methods 10:83-98. https://doi.org/10.1002/jrsm.1316
- [L] Viechtbauer 2005, JEBS 30:261-293. https://doi.org/10.3102/10769986030003261
- [L] Viechtbauer 2007, Stat Med 26:37-52. https://doi.org/10.1002/sim.2514
- [L] Viechtbauer 2010, metafor, JSS 36(3). https://doi.org/10.18637/jss.v036.i03
- [L] IntHout, Ioannidis & Borm 2014, BMC Med Res Methodol 14:25. https://doi.org/10.1186/1471-2288-14-25
- [L] Higgins, Thompson & Spiegelhalter 2009, JRSS-A 172:137-159. https://doi.org/10.1111/j.1467-985X.2008.00552.x
- [L] Riley, Higgins & Deeks 2011, BMJ 342:d549. https://doi.org/10.1136/bmj.d549
- [L] Higgins & Thompson 2002, Stat Med 21:1539-1558. https://doi.org/10.1002/sim.1186
- [L] Gelman 2006, Bayesian Analysis 1:515-534. https://doi.org/10.1214/06-BA117A
- [L] Friede, Röver, Wandel & Neuenschwander 2017, Res Synth Methods 8:79-91. https://doi.org/10.1002/jrsm.1217
- [L] Röver 2020, bayesmeta, JSS 93(6). https://doi.org/10.18637/jss.v093.i06
- [L] Turner et al. 2015, Stat Med 34:984-998. https://doi.org/10.1002/sim.6381
- [V] Chung, Rabe-Hesketh, Dorie, Gelman & Liu 2013, Psychometrika 78(4):685-709. https://link.springer.com/article/10.1007/s11336-013-9328-2
- [V, venue] Armstrong, Kolesár & Plagborg-Møller 2022, Econometrica. https://www.econometricsociety.org/publications/econometrica/2022/11/01/Robust-Empirical-Bayes-Confidence-Intervals/file/ecta200502.pdf
- [V, title] Walters, Empirical Bayes Methods in Labor Economics, NBER w33091. https://www.nber.org/system/files/working_papers/w33091/w33091.pdf
- [V, title] Kou & Yang, optimal shrinkage in heteroscedastic hierarchical linear models. https://arxiv.org/pdf/1503.06262
- [L] Xie, Kou & Brown 2012, JASA. https://doi.org/10.1080/01621459.2012.728154
- [L] Weiss & Freeman 2001, Neural Computation. https://doi.org/10.1162/089976601750541769
- [L] Robinson 1991, Stat Sci 6:15-32. https://doi.org/10.1214/ss/1177011926
- [V] Stan User's Guide, hierarchical priors. https://mc-stan.org/docs/2_19/stan-users-guide/hierarchical-priors-section.html
- [L] Betancourt & Girolami 2015. https://arxiv.org/abs/1312.0906 ; PyMC https://www.pymc.io/ ; NumPyro https://num.pyro.ai/en/stable/
- [L] Gelman & Pardoe 2006, Technometrics 48:241-251. https://doi.org/10.1198/004017005000000517
- [L] Turner, Spiegelhalter, Smith & Thompson 2009, JRSS-A 172:21-47. https://doi.org/10.1111/j.1467-985X.2008.00547.x
- [L] Cochrane Handbook ch. 10. https://training.cochrane.org/handbook/current/chapter-10

Marketing and elasticity hierarchies
- [L] Rossi, Allenby & McCulloch 2005, Bayesian Statistics and Marketing. https://doi.org/10.1002/0470863692 ; bayesm https://cran.r-project.org/package=bayesm
- [L] Allenby & Rossi 1999, J Econometrics. https://doi.org/10.1016/S0304-4076(98)00055-4
- [L] Montgomery 1997, Marketing Science 16(4):315-337. https://doi.org/10.1287/mksc.16.4.315
- [L] Boatwright, McCulloch & Rossi 1999, JASA 94:1063-1073. https://doi.org/10.1080/01621459.1999.10473859

Priors: robust mixtures, conflict, workflow
- [V] Schmidli et al. 2014, Biometrics 70(4):1023-1032. https://onlinelibrary.wiley.com/doi/abs/10.1111/biom.12242
- [V] Modified rMAP, medRxiv 2023. https://www.medrxiv.org/content/10.1101/2023.01.28.23285146v1.full
- [V, existence] SAM prior, Biometrics 2023. https://academic.oup.com/biometrics/article/79/4/2857/7587575 ; EB rMAP https://arxiv.org/pdf/2109.10237
- [L] Evans & Moshonov 2006, Bayesian Analysis 1:893-914. https://doi.org/10.1214/06-BA129
- [L] Gabry et al. 2019, JRSS-A. https://doi.org/10.1111/rssa.12378
- [L] Gelman et al. 2020, Bayesian Workflow. https://arxiv.org/abs/2011.01808
- [L] Talts et al. 2018, simulation-based calibration. https://arxiv.org/abs/1804.06788

Elasticity magnitudes
- [V] Tellis 1988, JMR. https://www.jstor.org/stable/3172944 ; https://www.msi.org/working-papers/the-price-elasticity-of-selective-demand-a-metaanalysis-of-sales-response-models/
- [V] Bijmolt, van Heerde & Pieters 2005, JMR 42(2):141-156. https://journals.sagepub.com/doi/10.1509/jmkr.42.2.141.62296
- [V] Hoch, Kim, Montgomery & Rossi 1995, JMR 32(1):17-29. https://www.researchgate.net/publication/237130017_Determinants_of_Store-Level_Price_Elasticity
- [V] DellaVigna & Gentzkow, Uniform Pricing in US Retail Chains (QJE 2019). https://www.nber.org/system/files/working_papers/w23996/revisions/w23996.rev0.pdf
- [V] Andreyeva, Long & Brownell 2010, AJPH 100(2):216-222. https://scholars.duke.edu/display/pub963616
- [V] Green et al. 2013, BMJ. https://pubmed.ncbi.nlm.nih.gov/23775799/
- [V, existence] Cornelsen et al. 2015, Health Econ. https://onlinelibrary.wiley.com/doi/10.1002/hec.3107
- [L] Lerner 1934, Rev Econ Stud. https://doi.org/10.2307/2967480
- [L] Ellison & Ellison 2009, Econometrica 77:427-452. https://doi.org/10.3982/ECTA5708
- [L] Chu, Chintagunta & Cebollada 2008, Marketing Science. https://doi.org/10.1287/mksc.1070.0288
- [L] Pauwels, Hanssens & Siddarth 2002, JMR. https://doi.org/10.1509/jmkr.39.4.421.19114
- [L] Mela, Gupta & Lehmann 1997, JMR. https://doi.org/10.1177/002224379703400205
- [L] Manning et al. 1987, AER 77:251-277. https://www.jstor.org/stable/1804094
- [L] Havranek, Irsova & Janda 2012, Energy Econ. https://doi.org/10.1016/j.eneco.2011.09.003
- [L] Espey 1998, Energy Econ. https://doi.org/10.1016/S0140-9883(97)00013-3

African and emerging markets
- [V] Korir, Rizov & Ruto 2020, Economic Modelling 92:99-108. https://ideas.repec.org/a/eee/ecmode/v92y2020icp99-108.html ; https://www.econstor.eu/bitstream/10419/224501/1/EM_Korir_Rizov_Ruto.pdf
- [V] Kenya zero-adjusted QUAIDS, J. Dev. & Agric. Econ. 17(4) 2025. https://academicjournals.org/journal/JDAE/article-abstract/6FC5D1173999
- [V] Nigeria farm households' demand response, J. Applied Econ. https://www.tandfonline.com/doi/full/10.1080/15140326.2021.1980351
- [V] Stacey, Tugendhaft & Hofman 2017, Preventive Medicine. https://www.sciencedirect.com/science/article/pii/S0091743517301871
- [V, attribution unclear] Mobile telephony elasticities: https://open.uct.ac.za/server/api/core/bitstreams/e9df6445-af7f-40a1-b82c-0e227850a952/content ; https://www.sciencedirect.com/science/article/abs/pii/S0167624521000275
- [L] Jensen & Miller 2008, AER 98(4):1553-1577. https://doi.org/10.1257/aer.98.4.1553
- [L] World Bank LSMS. https://www.worldbank.org/en/programs/lsms ; USDA ERS elasticities https://www.ers.usda.gov/data-products/commodity-and-food-elasticities/

Multi-tenant learning and privacy
- [L] Lin & Zeng 2010, Biometrika 97:321-332. https://doi.org/10.1093/biomet/asq006
- [L] Burke, Ensor & Riley 2017, Stat Med 36:855-875. https://doi.org/10.1002/sim.7141
- [L] McMahan et al. 2017, AISTATS. https://proceedings.mlr.press/v54/mcmahan17a.html
- [L] Bonawitz et al. 2017, ACM CCS. https://doi.org/10.1145/3133956.3133982
- [L] Dwork, McSherry, Nissim & Smith 2006, TCC. https://doi.org/10.1007/11681878_14
- [L] Dwork & Roth 2014. https://doi.org/10.1561/0400000042 ; OpenDP https://opendp.org/
- [L] Hundepool et al. 2012, Statistical Disclosure Control. https://doi.org/10.1002/9781118348239

---

Summary:
1. Replaces v1 §41-42/§80 with a two-stage model: data-only stage-1 (b, s) per business-product, exact closed-form Gaussian pooling on an industry -> country -> region -> market -> business -> product tree.
2. τ per level is fitted by penalised ML-II/REML/PM across cells, and intervals integrate over τ on a grid (plug-in EB covers about 83%, grid about 95% at k = 8).
3. Gives seed priors for all 13 industry ids, an LMIC staples override and a meta-regression for tier, income, competition and density; all are weak and labelled where they are placeholders.
4. Only summaries cross tenant boundaries; priors are served leave-one-tenant-out with k_min, dominance and optional Laplace DP, and there is no double counting.
5. Adds conflict checks (rMAP), sign rules, K_pc-based cold start, own-data-share labels, golden-value tests and snapshot release gates.
