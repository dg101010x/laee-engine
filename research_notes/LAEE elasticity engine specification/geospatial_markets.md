# Geospatial micro-market definition and competition measurement for small local businesses (African-city focus)

> **Provenance tags on every bullet.**
> - **[S]**: confirmed in this session only through search-engine result summaries. Every attempt to open primary pages failed: the egress proxy blocked docs.overturemaps.org, developers.google.com, openrouteservice.org, data.humdata.org, foursquare.com, kylebutts.com, arxiv.org, pmc.ncbi.nlm.nih.gov and web.mit.edu.
> - **[L]**: established literature or documentation cited from memory and *not* re-verified this session. Check the numbers before quoting them.
> - **[C]**: computed locally this session (Python 3.11, numpy/scipy, h3 4.5.0), so these appear under Inferences.
> - **[I]**: my inference or design recommendation.
>
> The shared web-search budget ran out partway through. Planned checks on Busso–Galiani, Bergquist–Dinerstein, US trip-distance surveys, mobility-data decay estimates, Lagos/Accra/Johannesburg mode shares, Overpass limits, Google Routes pricing, GraphHopper, DellaVigna–Gentzkow and African census granularity therefore did not run. Those items are tagged [L] or listed under Gaps. "Current" means October 2026.
>
> **Proposed v2 notation (used throughout):**
> - j = focal business; k = rival; c = origin cell (H3 res 9 by default)
> - t_cj = generalised travel time in minutes; d = distance in km
> - K_θ(t) = decay kernel (default e^{−βt}); Pop_c = residential population; D_c = demand mass; A_j = attractiveness or size
> - m_jk ∈ [0,1] = calibrated substitutability; π_k = existence probability
> - s_cj = choice probability; ω_cj = origin weights of j's customers; O_jk = catchment overlap; ŵ_jk = normalised index weights
>
> Each symbol has one meaning. v1 used W_ij for four different things.

## Q1. What in v1 §5–13, §29–32, §57, §61 and §71 is wrong, unsupported or underspecified?

### Takeaway
v1 lists sensible intentions but no algorithm. Its competitor index C_ij = S·D·M·R·W and its price index are mathematically flawed: distance is double-counted, there is no demand-model basis, and the CPI is unnormalised. Its radius priors and its "argmax_R predictive accuracy" rule are unsupported and prone to overfitting. Its demographic and entry/DiD sections assume data and identification that do not hold in African cities: Nigeria has had no census since 2006, the RWI licence forbids commercial use, staggered-entry TWFE is biased, and ring spillovers bias near-vs-far comparisons.

### Cited Findings
- When a Walmart Supercenter opens within one mile, incumbent supermarkets lose about 16% of revenue. The effect "decays quickly with distance", and there is no causal effect on incumbent prices — [Arcidiacono, Ellickson, Mela & Singleton (2020), AEJ: Applied](https://www.aeaweb.org/articles?id=10.1257%2Fapp.20180047); [working-paper PDF](https://public.econ.duke.edu/~psarcidi/walmart.pdf) [S]
- Large-format grocery entry has "strongly localized effects" on nearby incumbents' sales, employment and exit. Outlets of larger chains are hit hardest — [Ellickson & Grieco (2013), J. Urban Economics 75:1–14](https://econpapers.repec.org/RePEc:eee:juecon:v:75:y:2013:i:c:p:1-14) [S]
- A spatially aggregated discrete-choice model with overlapping consumer choice sets "avoids the need to define markets ex ante". It finds substantial competition across formats (supercenters, club stores, traditional grocers) — [Ellickson, Grieco & Khvastunov (2020), RAND J. Econ. 51(1):189–232](https://onlinelibrary.wiley.com/doi/abs/10.1111/1756-2171.12310) [S]
- In 83 supermarkets × 18 categories, 11 demographic and competitive variables explained on average 67% of the variation in store-level price elasticity. Demographics were "much more influential" than competition variables — [Hoch, Kim, Montgomery & Rossi (1995), JMR 32(1):17–29](https://www.researchgate.net/publication/237130017_Determinants_of_Store-Level_Price_Elasticity) [S]
- Ring DiD (inner ring vs. a slightly farther outer ring) needs explicit identifying assumptions. Because effects decay with distance, Butts proposes a nonparametric estimator of the effect-vs-distance curve (building on Cattaneo et al. 2019) that does not require knowing how far effects reach — [Butts (2023), "JUE Insight: Difference-in-differences with geocoded microdata", J. Urban Econ. 133:103493](https://www.kylebutts.com/papers/did-rings/); [code](https://github.com/kylebutts/Difference-in-Differences-Ring-Method) [S]
- With staggered adoption and heterogeneous effects, TWFE DiD is a weighted average of 2×2 comparisons. Some use already-treated units as controls, and weights can be negative — [Goodman-Bacon (2021), J. Econometrics 225(2)](https://doi.org/10.1016/j.jeconom.2021.03.014); [de Chaisemartin & D'Haultfœuille (2020), AER 110(9)](https://doi.org/10.1257/aer.20181169) [L]
- Meta's Relative Wealth Index (2.4 km resolution; 93 LMICs on HDX) is licensed **CC BY-NC 4.0** — [HDX: Relative Wealth Index](https://data.humdata.org/dataset/relative-wealth-index); [Chi et al. (2022), PNAS 119(3):e2113658119](https://www.pnas.org/content/pnas/119/3/e2113658119.full.pdf) [S]
- Nigeria's last census was in 2006. GRID3/WorldPop bottom-up gridded estimates fill the gap — [HDX: GRID3 Nigeria v2.0](https://data.humdata.org/dataset/bottom-up-gridded-population-estimates-for-nigeria-version-2-0); [Univ. of Southampton](https://www.southampton.ac.uk/news/2020/10/worldpop-census-calculation-nigeria.page) [S]
- Google's terms let you store Places place IDs indefinitely and cache lat/lng for up to 30 consecutive calendar days. Other Places content (names, ratings, photos, etc.) has no caching exception — [Google Maps Platform Service Specific Terms](https://cloud.google.com/maps-platform/terms/maps-service-terms); [summary](https://openplacesapi.com/blog/can-you-store-places-api-results) [S]
- Supabase ships PostGIS and pgRouting. h3-pg is an open request (supabase/postgres issue #245) and is not on the documented extension list — [issue #245](https://github.com/supabase/postgres/issues/245); [Supabase extensions](https://supabase.com/docs/guides/database/extensions); [pgRouting docs](https://github.com/supabase/supabase/blob/master/apps/docs/content/guides/database/extensions/pgrouting.mdx) [S]
- US card data show limited consumer mobility:
  - In the median sector, spending in an average out-of-home location is only 34% of home-location spending.
  - A 1% increase in distance lowers spending by about 0.4%.
  - Spending falls faster with distance in sectors bought more often.

  Sources: [Agarwal, Jensen & Monte (2017), NBER w23616](https://www.nber.org/system/files/working_papers/w23616/revisions/w23616.rev1.pdf); [VoxEU summary](https://voxeu.org/article/geography-consumption) [S]
- In Nairobi, walking accounts for about 41% of trips and matatus about 29%. 98% of households report matatu or bus service within a 20-minute walk (JICA 2014 and World Bank surveys) — [Salon & Gulyani (2019), Sustainability](https://www.researchgate.net/publication/334439119_Commuting_in_Urban_Kenya_Unpacking_Travel_Demand_in_Large_and_Small_Kenyan_Cities); [Nairobi vs. Kisumu comparison](https://pmc.ncbi.nlm.nih.gov/articles/PMC10345788/) [S]
- Rivals' competitive effects depend on product type, and firms differentiate (by quality tier or location) to soften competition — [Mazzeo (2002), RAND 33(2):221–242](https://econpapers.repec.org/RePEc:rje:randje:v:33:y:2002:i:summer:p:221-242); [Seim (2006), RAND 37(3):619–640](https://ideas.repec.org/a/rje/randje/v37y20063p619-640.html) [S]
- Most of the change in competitive conduct comes with the second and third entrants; later entrants matter little — [Bresnahan & Reiss (1991), JPE 99(5)](https://doi.org/10.1086/261786) [L]
- Text-cosine competitor sets work only once they are validated: the best TNIC specification overlaps 62% with Capital IQ peers — [Hoberg & Phillips (2016), JPE 124(5)](https://econpapers.repec.org/RePEc:ucp:jpolec:doi:10.1086/688176) [S]

### Inferences
#### Overlap math behind the double-counting claim [C]
Take two exponential catchment kernels e^{−βr} over uniform population and let d be the distance between the two businesses. The normalised customer overlap O(d)/O(0) is:

| βd | Overlap O(d)/O(0) | Kernel e^{−βd} |
|---|---|---|
| 0.5 | 0.94 | 0.61 |
| 1 | 0.81 | 0.37 |
| 2 | 0.51 | 0.14 |
| 3 | 0.28 | 0.05 |
| 6 | 0.03 | 0.002 |

Overlap is a deterministic, slowly decaying function of distance. It halves at βd ≈ 2.03, i.e. about 1.21 × the median customer distance. Multiplying "distance decay" by "customer overlap" therefore penalises distance twice.

#### Cross-cutting defects [I]
1. **Symbol collision (wrong).** W_ij is used for four different things: the decay kernel e^{−λd} (§7), "estimated customer overlap" (§8), price-index weights (§10) and substitute-network edges (§32). §8 also has a separate D_ij for distance decay. Use the notation above.
2. **Units (underspecified).** §5/§6 use km, while §9 ("1.2-mile market") and §11 use miles. λ has no unit. Store metres internally, express travel cost in minutes, and store the unit of every parameter.
3. **Two inconsistent competition definitions (wrong).** §8 defines Competition_i = Σ C_ij; §9 defines CompetitivePressure_i = Σ M·D·PriceSim·QualitySim. §9's display math is also malformed: it opens with `\[` and closes with a code fence. Keep one definition, derived from a demand model (Q5).
4. **Nothing is estimated (unsupported).** No v1 geospatial quantity has an estimator, standard error or validation target, so scales are arbitrary and cannot be mapped to elasticities.
5. **Reproducibility (underspecified).** Geo features change with every POI snapshot, OSM rebuild and embedding model. §73 needs these fields on every geo feature: `poi_snapshot_date`, `osm_snapshot`, `routing_engine/version`, `kernel_params_version`, `similarity_model_version`.

#### Section by section [I]
- **§5 Layer A radii (underspecified).** 250 m–5 km works as a global set of fixed feature bands, which lets models pool across businesses. It does not work as a market definition. Replace it with an estimated continuous kernel (Q2), plus shared distance or time bands for reduced-form rival counts (Q5).
- **§6 RADIUS_PRIORS (unsupported).**
  - No source; discrete radii with no uncertainty; implicitly straight-line and car-based.
  - "grocery 1–10 km" lumps chain supermarkets with dukas, kiosks and spaza shops, whose catchments are walking-scale.
  - R_i = f(industry, density, mobility, ticket) is never specified.
  - **Fix:** log-normal priors on decay β per (category × density tier × dominant mode), stated as a median customer distance or time and converted with R50 = 1.678/β (Q2). Update them hierarchically (Q7). The purchase-frequency effect in Agarwal et al. supports a ticket/frequency modifier.
- **§7 travel-time kernel (underspecified).** No routing engine, mode, congestion treatment or estimation of λ (see Q3, Q2, Q7).
- **§8 C_ij = S·D·M·R·W (wrong).**
  - (a) D and W are both functions of distance (overlap math above), so distance is double-counted; triple-counted if §7's W is also applied.
  - (b) S_j has no scale, so big-box stores dominate.
  - (c) A linear sum over rivals has no diminishing returns, contradicting Bresnahan–Reiss.
  - (d) It is not normalised relative to the focal firm's own attractiveness.
  - (e) A product of five noisy terms compounds error, and a single zero term silently deletes a rival.
  - (f) It assumes rivals only hurt. It ignores agglomeration and comparison-shopping clusters such as auto-parts streets, salon rows and market halls.
  - (g) Separability is assumed, never tested.
  - **Fix:** Pressure_j derived from a spatial nested logit, plus estimated distance-band × similarity-class coefficients (Q5).
- **§9 embedding cosine (unsupported).** Cosine can be negative and is not a substitution probability. Review text mixes sentiment and quality with product type, and results drift when the model changes. **Fix:** use cosine as one input to a calibrated classifier P(competitor | features) trained on labelled pairs (Q5).
- **§10 CPI = Σ_j W_ij P_j / P_i (wrong).**
  - The weights are not normalised. Ten identical rivals all priced at P_i give CPI = 10·W̄, which the spec would read as "cheaper than competitors" even at price parity.
  - The direction (CPI > 1 = cheaper) is the reverse of convention.
  - Arithmetic means of raw prices mix basket composition.
  - **Fix:** rp_j = log P_j − Σ ŵ_jk log P_k with Σ ŵ = 1, computed on matched items, with coverage gating and NA when coverage is low (Q5).
- **§11 customer geography (underspecified; the example misleads).**
  - The band shares are in miles and confound decay with annulus area. The 0.5–1 mi band has 3× the area of the 0–0.5 mi band, so its 31% share against 28% still means per-capita visit probability *falls* with distance.
  - Model per-capita rates with a population offset (Q2).
  - Loyalty and delivery customers are selected samples.
  - Delivery zones are platform-imposed censoring, not behaviour.
  - Privacy aggregation is unspecified.
- **§12 demographics (unsupported for target cities).**
  - Median income, home ownership, renter share, education and unemployment are mostly unavailable below city level in Nigeria (no census since 2006) and are patchy elsewhere.
  - The best open wealth surface (RWI) is non-commercial.
  - "w_k = customer probability" needs normalisation and an extensive-vs-intensive distinction.
  - "Income variance" needs the law of total variance (Q6).
- **§13 MarketRadius* = argmax_R PredictiveAccuracy(R) (wrong as stated).**
  - (a) Picking the best of several R on the same data shows a positive "gain" even when no radius matters (winner's curse), and R̂ is unstable.
  - (b) The four listed targets (overlap, competitor effects, price response, sales) will pick different R, and no rule combines them.
  - (c) The best-predicting R can proxy for unobserved demand, such as a CBD's daytime population, rather than competition.
  - (d) Tuning R per business on tiny samples overfits.
  - (e) Tuning R on outcomes later used in DML (§16) leaks outcome information into features.
  - **Fix:** hierarchical likelihood estimation of the decay with confidence intervals; ≤ 3 pre-registered kernels compared under spatial-block CV nested inside cross-fitting; radii reported only as derived quantiles (Q7).
- **§29 entry/exit (unsupported data).** opened_at/closed_at, price_change, rating_change and category_change are unobserved for most rivals. Google rating histories cannot be stored under its terms. Open-data event dates are lagged "first-seen" dates. ΔCompetition on an arbitrary-scale index has no interpretation.
- **§30 DiD (wrong as specified).**
  - TWFE with staggered entries is biased.
  - "Near vs. not near" needs Butts's ring assumptions.
  - Entry is endogenous to demand trends.
  - "Similar businesses" has no matching rule.
  - Controls near treated units are contaminated.
  - There is no power analysis, and one business's single event is not identified (Q8).
- **§31 cross-price (underspecified).** Rival price series rarely exist for African SMBs. "Sufficient variation" has no threshold. Rival prices are endogenous because they react to the same shocks. **Fix:** restrict ε_jk through the spatial logit, so a few parameters generate all cross-elasticities (Q5), and estimate freely only with instruments.
- **§32 substitute network (circular).** The edge weight multiplies in "ObservedSubstitution", which is the quantity being estimated. A product-level graph across rivals needs menu/item matching that informal SMBs lack. **Fix:** split it into (i) a business-level competition graph, with edges = O_jk × m_jk from estimated parameters, and (ii) product-level substitution *within* the focal business from its own transactions.
- **§57 local economic shock (unsupported inputs).** Neighbourhood unemployment, spending and housing series mostly do not exist. Replace them with proxies that carry declared lag and resolution (Q6): VIIRS radiance (500 m), building growth from Open Buildings 2.5D Temporal (annual, 2016–2023), POI births/deaths from monthly open snapshots, and national fuel, FX and CPI shocks.
- **§61 price anchoring (duplicate).** PriceGap repeats §10; P_market is undefined; percentiles need rival price distributions that rarely exist. Use the single rp_j with coverage gating.
- **§71 storage (partly infeasible on Supabase).**
  - Without h3-pg, compute H3 in Python (h3 v4) and store BIGINT columns.
  - "Competitors → neighbouring H3 cells" needs k derived from R and resolution, followed by an exact distance filter.
  - Building OD matrices from roads in PostGIS/pgRouting is far slower than an external OSRM or Valhalla.
  - "Census geographies" rarely exist publicly at small-area level.
  - Snapshot versioning is missing (Q7).

### Gaps
- The v1 radius values cannot be validated: no African study measuring trade-area size by business type was found in this session.

## Q2. Trade-area and market definition: Huff/Reilly, distance-decay forms and calibration, catchments, isochrones vs radii, empirical trade-area sizes

### Takeaway
Define the market as an estimated spatial choice kernel inside a Huff/spatial-logit model (u = γ·log A − β·t, plus an outside option), fitted by maximum likelihood. Multi-store flows can use a Poisson GLM with origin fixed effects, which is equivalent to the conditional logit. With only the focal store's customers, use a population-offset Poisson model. Report any "radius" as a quantile of the fitted customer-distance distribution. The empirical evidence on reach is mostly from the US and is very local for frequent purchases (grocery ≈ 1 mile, gasoline ≈ 1.5 mi, cinemas ≈ 5 mi). In walking-dominated Nairobi (41% of trips on foot), convenience trade areas should be walking-scale.

### Cited Findings
- Probabilistic trade areas originate with [Huff (1963), Land Economics 39(1):81–90](https://www.semanticscholar.org/paper/A-Probabilistic-Analysis-of-Shopping-Center-Trade-Huff/d35f5666d9455a4cb8a6c75898a95d8109858784) [S]; follow-up [Huff (1964), J. Marketing 28(3):34–38](https://doi.org/10.1177/002224296402800307) [L]
- Huff utility is U_ij = A_j^γ · d_ij^−λ. Huff estimated λ by matching expected and observed shares with γ held at 1. When shares are observed, the MCI log-centering transformation of Nakanishi & Cooper (1974) is "probably the best way" to estimate it. Maximum likelihood on shares, customer flows or total market areas is the alternative — [Wieland, R Journal: MCI](https://digitalcommons.unl.edu/cgi/viewcontent.cgi?article=1592&context=r-journal); [MCI manual](http://cran.r-universe.dev/MCI/doc/manual.html); [Nakanishi & Cooper (1974), JMR 11(3)](https://journals.sagepub.com/doi/abs/10.1177/002224377401100309) [S]
- Frequent-purchase activities such as grocery have large distance exponents; furniture has small ones — [Wikipedia: Huff model](https://en.wikipedia.org/wiki/Huff_model) [S]
- Open implementations exist: Python package [`huff`](https://pypi.org/project/huff/1.9.10/), described in [arXiv 2602.17640](https://arxiv.org/html/2602.17640v1); [Esri Business Analyst Huff](https://pro.arcgis.com/en/pro-app/3.5/tool-reference/business-analyst/understanding-huff-model.htm) [S]
- Huff models have been calibrated with location big data — [Liang et al. (2020), Transactions in GIS 24:681–703](https://onlinelibrary.wiley.com/doi/10.1111/tgis.12624) — and with taxi trajectories — [ACM SIGSPATIAL workshop 2017](https://dl.acm.org/doi/10.1145/3151547.3151553) [S]
- Choice-based (store-intercept) samples need special estimators — [O'Kelly (1999), Environment & Planning A](https://journals.sagepub.com/doi/10.1068/a310613) [S]
- A 2026 example parameterises an extended Huff model empirically for hospital catchments — [IJHG 2026](https://link.springer.com/article/10.1186/s12942-026-00478-2) [S]
- Reilly's law: B_a/B_b = (P_a/P_b)(D_b/D_a)². Converse breakpoint: D_a = D_ab / (1 + √(P_b/P_a)) — [Reilly's law](https://en.wikipedia.org/wiki/Reilly%27s_law_of_retail_gravitation); [Converse (1949), J. Marketing 14(3)](https://doi.org/10.1177/002224294901400303) [L]
- Gravity models can be fitted by Poisson regression — [Flowerdew & Aitkin (1982), J. Regional Science 22(2)](https://doi.org/10.1111/j.1467-9787.1982.tb00744.x) [L]. The conditional-logit likelihood equals the Poisson likelihood with group fixed effects — [Guimarães, Figueiredo & Woodward (2003), REStat 85(1)](https://doi.org/10.1162/003465303762687811) [L]
- Distance elasticity of spending is about −0.4 in the median sector, and decline is steeper for frequently bought goods — [Agarwal, Jensen & Monte (2017)](https://www.nber.org/system/files/working_papers/w23616/revisions/w23616.rev1.pdf) [S]
- In NYC (Yelp data), transit time has a first-order effect on restaurant choice, and consumption is about half as segregated as residences — [Davis, Dingel, Monras & Morales (2019), JPE 127(4):1684–1738](https://www.journals.uchicago.edu/doi/abs/10.1086/701680); [replication](https://github.com/jdingel/DavisDingelMonrasMorales) [S]
- Movie theatres: reducing travel by one mile is worth about $1.18 of admission at first, and marginal travel cost falls with distance (nonlinear). Comparing 5/10/15-mile rings, theatres within 5 miles dominate choice — [Davis (2006), RAND 37(4):964–982](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1756-2171.2006.tb00066.x) [S]
- Grocery: −16% incumbent revenue within 1 mile of supercenter entry, decaying quickly — [Arcidiacono et al. (2020)](https://www.aeaweb.org/articles?id=10.1257%2Fapp.20180047) [S]
- Gasoline: local competition is typically defined within about 1.5 miles (≈ 2 km), and more stations lower both price levels and dispersion — [Barron, Taylor & Umbeck (2004), as summarised in Energy Sources Part B (2023)](https://www.tandfonline.com/doi/full/10.1080/15567249.2023.2189330); [Eckert (2013) survey](https://onlinelibrary.wiley.com/doi/10.1111/j.1467-6419.2011.00698.x) [S]
- The E2SFCA splits catchments into 0–10 / 10–20 / 20–30-minute zones with weights 1.00 / 0.68 / 0.22 ("slow decay"). An erratum says the Gaussian-consistent fast set should be 1.00 / 0.42 / 0.03 — [Luo & Qi (2009), Health & Place 15(4)](https://www.sciencedirect.com/science/article/abs/pii/S1353829209000574); [erratum](https://www.sciencedirect.com/science/article/pii/S1353829210001577) [S]
- Customer spotting (mapping in-store customers' home locations) and the primary/secondary trade-area conventions — [Applebaum (1966), JMR 3(2)](https://doi.org/10.1177/002224376600300202) [L]
- Nairobi mode shares: walking about 41%, matatu about 29% — [Salon & Gulyani (2019)](https://www.researchgate.net/publication/334439119_Commuting_in_Urban_Kenya_Unpacking_Travel_Demand_in_Large_and_Small_Kenyan_Cities) [S]

### Inferences
#### A. Core model (replaces §5–§7, §11, §13) [I]
For category g, origin cells c (H3 res 9), businesses j ∈ J_c with t_cj ≤ t_max, and an outside option 0:
```
u_cj = γ·log A_j − β·t_cj + ξ_j               exponential (default)
     | γ·log A_j − λ·log(t_cj + t0)          power (needs λ > 2 in 2-D or a hard truncation)
     | γ·log A_j − t_cj² / (2σ²)             Gaussian
s_cj = exp(u_cj) / ( exp(u_c0) + Σ_{k∈J_c} exp(u_ck) )
Q_j  = Σ_c D_c · s_cj ,   D_c = Pop_c · φ_g   (φ_g = per-capita category demand)
```
- Huff (1963) is the special case with no outside option and power decay. Keep the outside option so that entry can expand total category demand, not only redistribute it.
- **Defaults:**
  - t in minutes of generalised travel time (Q3)
  - t0 = 1 min (power form)
  - t_max = P99 of the prior kernel (≈ 6.64/β for exponential decay with uniform population)
  - A_j = floor-area or footprint proxy, else 1
  - prior γ ~ N(1, 0.5²)
  - u_c0 = 0, with φ_g absorbing penetration

#### B. Estimating decay by data situation [I] [C]
1. **Multi-store flows T_cj** (platform data): `log E[T_cj] = θ_c + γ log A_j − β t_cj` with origin fixed effects θ_c.
   - [C] On a simulated Huff (60 origins, 6 stores, true γ = 1.0, β = 1.3/km), conditional-logit MLE gave (0.97017, 1.29643) and Poisson with origin FE gave (0.97025, 1.29656); SE(β) ≈ 0.02.
   - **Unit test:** the two estimators agree to within 1e-3 relative.
2. **Own-store customers only.** This is the usual SMB case: delivery addresses, WhatsApp location pins, loyalty sign-ups with estate names, intercept surveys.
   - Exact model: `N_c ~ Poisson(Pop_c · e^{θ0} · s_cj(β, γ, u0))`.
   - Cheap approximation: `log E[N_c] = log Pop_c + θ0 − β t_cj − ψ·log(1 + Σ_k π_k m_jk e^{−β t_ck})`; ψ ≈ 1 is consistent with logit.
   - Dropping the rival term makes β absorb competition. Dropping the population offset makes β absorb population geography.
   - [C] With the population offset, about 270 geocoded customers spread over 600 res-9 cells (t uniform 0–30 min, β = 0.12/min) gave mean β̂ = 0.1221, sd 0.0094 (≈ 8% relative), and 93% coverage of the 95% profile-likelihood CI over 300 simulations.
   - [C] When residential density itself declines away from the store and the offset is omitted, β̂ = 0.174 against a true 0.12; the offset model gave 0.115. This is a ready-made regression test.

   Scipy sketch (dependencies: numpy and scipy only):
   ```python
   def fit_exp_decay(n_c, pop_c, t_c):
       off = np.log(np.clip(pop_c, 1e-12, None))
       def nll(x):                                   # Poisson NLL up to a constant
           eta = off + x[0] - np.exp(x[1]) * t_c     # beta = exp(x[1]) > 0
           return np.sum(np.exp(eta) - n_c * eta)
       x0 = np.array([np.log(n_c.sum() / pop_c.sum()), np.log(0.1)])
       res = scipy.optimize.minimize(nll, x0, method="BFGS")
       return np.exp(res.x[1]), res                  # CI: profile over beta, re-optimising theta0
   ```
3. **No customer geography.** Use the hierarchical prior (§F) and update it with distance-band entry responses (Q8).
- Below about 100 geocoded customers, shrink hard toward the prior. Relative SD scales like √(270/N) × 8%, so roughly 13% at N = 100. Confirm with a parametric bootstrap per category.

#### C. Kernel forms and derived radii [I] [C]
- [C] **Exponential**, with uniform population: customer distance ~ Gamma(2, β).

  | Statistic | Value |
  |---|---|
  | P50 | 1.6783/β |
  | P70 | 2.4392/β |
  | P80 | 2.9943/β |
  | P90 | 3.8897/β |
  | P95 | 4.7439/β |
  | Mean | 2/β |
  | Kernel half-life | ln2/β |

- [C] **Gaussian** e^{−d²/2σ²}: distance ~ Rayleigh(σ); P50 = 1.1774σ, P80 = 1.7941σ, P90 = 2.1460σ, P95 = 2.4477σ.
- [C] **Power** (d + d0)^−λ in 2-D with uniform population: total mass diverges unless λ > 2. With d0 = 0.1 km and cut-offs of 10 / 100 / 1000 km:
  - λ = 1.5 gives mass 5.2 / 18.8 / 62.0
  - λ = 2 grows logarithmically: 3.6 / 5.9 / 8.2
  - λ = 2.5 converges: 3.59 / 4.02 / 4.15

  So enforce λ > 2 or truncate. Huff-style d^−λ with small λ assigns absurd shares to remote areas.
- Default to exponential: finite at 0, integrable, one interpretable parameter. Gaussian suits walking catchments with a sharp edge. Choose by held-out log-likelihood under spatial-block CV (Q7).
- **Operational radius** (for UI and reports only): R_p = weighted p-quantile of t_cj with weights D_c·s_cj. Report R50, R80 ("primary trade area") and R95. For an exponential kernel with uniform population, R80/R50 = 1.784, which is a quick sanity check.
- **Competitive reach ≠ customer reach.** Overlap halves at d ≈ 2.03/β ≈ 1.21·R50 (Q1 [C]), so rivals out to roughly the own-customer P70–P80 still matter.

#### D. Reilly/Converse (sanity checks only) [I]
- Generalised breakpoint: d_A = D_AB / (1 + (A_B/A_A)^{1/λ}) for power decay; d_A = D_AB/2 + ln(A_A/A_B)/(2β) for exponential.
- Tests: equal attractiveness gives the midpoint; A_A > A_B moves the breakpoint toward B; λ = 2 with A = population reproduces Converse.

#### E. Catchments from customer or delivery data [I]
- **Pipeline:** geocode → H3 res 9 → counts N_c → join Pop_c (Q6) → fit B.2 → primary catchment = smallest set of cells holding 80% of fitted customers.
  - Addresses in Nairobi and Lagos are often landmark- or estate-based, so prefer GPS pins captured at order time.
  - [Plus Codes](https://github.com/google/open-location-code) are open [L].
- **Privacy:** store only H3 res 8/9 counts; suppress cells with fewer than 5 customers in shared outputs.
- **Delivery zones:** the platform's radius truncates demand. Treat cells outside it as structural zeros (censoring), or estimate β from walk-in data. Never read truncation as steep decay.
- **KDE** of customer points confounds customer density with population density. Use it for display only.

#### F. Default priors for dense African cores (engineering defaults, not evidence) [I]
Expressed as median Euclidean customer distance. β0 = 1.678/median assumes exponential decay with locally uniform population. Prior: log β ~ N(log β0, sd²). R80 = 1.784 × median.

| Category | Median km | β0 /km | sd(log β) | R80 km | v1 prior (km) |
|---|---|---|---|---|---|
| Kiosk/duka/spaza/convenience | 0.3 | 5.6 | 0.5 | 0.54 | grocery 1–10 |
| Café/coffee | 0.7 | 2.4 | 0.6 | 1.25 | 0.25–2 |
| Local eatery (kibanda, buka, chop bar, fast food) | 0.6 | 2.8 | 0.6 | 1.07 | — |
| Sit-down restaurant | 1.5 | 1.1 | 0.7 | 2.7 | 0.5–3 |
| Salon/barber (kinyozi) | 0.8 | 2.1 | 0.6 | 1.43 | 1–5 |
| Pharmacy/chemist | 0.7 | 2.4 | 0.5 | 1.25 | — |
| Supermarket (formal chain) | 2.5 | 0.67 | 0.6 | 4.5 | 1–10 |
| Auto repair/garage (jua kali) | 3.0 | 0.56 | 0.8 | 5.4 | 3–20 |
| Professional services | 5.0 | 0.34 | 0.8 | 8.9 | 3–10 |

- **Modifiers:**
  - Car-dominant low-density suburbs (e.g., northern Johannesburg; Karen/Runda in Nairobi): median × 2.5.
  - CBDs and market districts: demand mass must include daytime population (Q6).
  - Higher ticket or lower purchase frequency: smaller β (the direction found by Agarwal et al.).
  - Specialist clusters: allow positive same-category co-location effects.
- **Replacement rule:** once a (category × city) cell has ≥ 20 businesses with customer geography, the empirical-Bayes posterior replaces these defaults (Q7).

#### G. Isochrones vs radii [I]
- Feed the kernel with travel times (Q3) and keep isochrone polygons for display and storage.
- Network time matters where there are barriers: limited-crossing highways (Nairobi's Thika Road and Mombasa Road), Lagos's lagoon bridges, rail lines, rivers, gated estates. It also matters where modes differ and where congestion is high.
- In barrier-free walking cores, Euclidean distance × circuity is adequate for ranking.

#### H. Tests [I]
- s_cj sums to 1 including the outside option.
- Monotone in A_j and in t_cj.
- Unit invariance: switching km→m rescales β by 1/1000 and leaves s unchanged.
- Parameter recovery: 95% CI coverage of 93–97% over ≥ 200 simulations.
- Monte Carlo check that R50 = 1.678/β (±1%).
- An untruncated power kernel with λ ≤ 2 raises an error.
- Censoring test: simulated delivery-zone truncation does not bias β once censored cells are excluded.
- Offset test (B.2).

### Gaps
- Not retrieved (fetch blocked): Huff's original fitted λ values (clothing vs. furniture), Seim's band cut-offs and estimates, DDMM's travel-time coefficients, Liang et al.'s decay estimates.
- No African trade-area measurements by business type were found (no customer-distance distributions for dukas, salons, chemists or garages in Nairobi, Lagos, Accra or Johannesburg). The §F priors are unvalidated.
- Not checked after the search budget ran out: mode shares for Lagos, Accra and Johannesburg; okada/boda-boda regulation (e.g., Lagos restrictions); delivery-platform radius norms in African cities.

## Q3. Travel-time computation

### Takeaway
Self-host OSRM (BSD-2; fastest OD tables) or Valhalla (MIT; matrix, isochrones, multimodal, tile-based memory) on monthly Geofabrik country extracts. Run walking and driving profiles, plus GTFS-based transit where it exists. Calibrate free-flow driving times with a small paid sample of traffic-aware times. Use straight-line distance × a city-specific circuity factor only for prefiltering and for barrier-free dense cores. The public openrouteservice API is suitable only for prototyping.

### Cited Findings
- OSRM is licensed under the simplified 2-clause BSD licence (previously AGPL) — [Mapbox blog](https://blog.mapbox.com/more-open-project-osrm-switches-to-bsd-fde88d5ef462?gi=4841f987e406); [Wikipedia](https://en.wikipedia.org/wiki/Open_Source_Routing_Machine) [S]
- osrm-routed's `--max-table-size` defaults to 100 (locations per /table request; capped to prevent out-of-memory) and can be raised when self-hosting — [OSRM course](https://medium.com/@imadsaddik/7-osrm-course-changing-global-parameters-8a4daa759672); [R osrm docs](https://cran.r-project.org/web/packages/osrm/osrm.pdf) [S]
- Practitioner rule of thumb (exact original source unclear): OSRM preprocessing needs about 10× the map-file size in RAM, and serving about 3–5× — [practitioner write-up](https://www.ecotner.com/blog/gcp-osrm-and-python); [Medium](https://americantiredistributors.medium.com/route-optimization-getting-google-cloud-and-osrm-set-up-4f863ade5d98) [S]
- Valhalla is MIT-licensed. It provides time+distance matrices, isochrones, map matching, TSP and elevation, on a tiled graph loaded on demand (one metro region up to the whole planet) — [GitHub](https://github.com/valhalla/valhalla); [Interline](https://www.interline.io/valhalla/) [S]. Planet-scale serving in 16–32 GB RAM is claimed for localised query patterns — [Big Iron 2026 guide](https://www.bigiron.cc/guides/openrouteservice-vs-osrm-vs-valhalla-self-hosted-routing) [S]
- openrouteservice public API limits; sources conflict:
  - Free tier ≈ 2,500 requests/day and 40,000/month, 40 concurrent; about 500 isochrone requests/day.
  - Matrix: ≤ 3,500 origin×destination elements per request (e.g., 50×50), or 25 with dynamic arguments.
  - Isochrones: ≤ 5 locations, ≤ 10 intervals, ≤ 120 km distance range; time ranges ≤ 20 h foot, 5 h cycling, 1 h driving.

  Sources: [HeiGIT plans](https://account.heigit.org/info/plans); [ORS restrictions](https://openrouteservice.org/restrictions/); [apispine summary](https://apispine.com/openrouteserviceorg/pricing) [S]
- Inter-city road circuity is about 1.3 globally: USA 1.22–1.25, Western Europe 1.25, Japan 1.09, Brazil 1.33. The study sampled 26 countries and did *not* cover urban street routes — [Ballou, Rahardja & Sakai (2002), Transportation Research A](https://www.semanticscholar.org/paper/Selected-country-circuity-factors-for-road-travel-Ballou-Rahardja/f3f5e4d3bb2c4d21100dca1f9b2cefe5517adecf) [S]
- Street-network circuity (street length ÷ straight-line distance between adjacent nodes) has been computed for 8,914 urban areas in 178 countries from OSM/OSMnx, with open data repositories — [Boeing (2022), Geographical Analysis](https://onlinelibrary.wiley.com/doi/abs/10.1111/gean.12281); [preprint](https://arxiv.org/pdf/2009.09106) [S]; 2026 update — [Boeing (2026), EPB](https://journals.sagepub.com/doi/10.1177/23998083261446991) [S]. Related: [Giacomin & Levinson (2015)](https://journals.sagepub.com/doi/10.1068/b130131p) and [Boeing (2019), walk vs. drive circuity](https://www.ocf.berkeley.edu/~gboeing/publications/2019-boeing-morphology-circuity-streets.pdf) [S]
- In Montréal, Euclidean and Manhattan distances correlate > 0.95 with network distance and time overall. The correlation is weaker in suburbs and with distance from the CBD, and much weaker for public-transit time — [Apparicio et al. (2008), IJHG](https://pmc.ncbi.nlm.nih.gov/articles/PMC2265683/); [Apparicio et al. (2017)](https://link.springer.com/article/10.1186/s12942-017-0105-9) [S]
- Nairobi's matatu network was mapped to GTFS by the Digital Matatus project — [Williams et al. (2015), J. Transport Geography 49](https://doi.org/10.1016/j.jtrangeo.2015.10.005) [L]
- Supabase supports pgRouting — [Supabase docs](https://github.com/supabase/supabase/blob/master/apps/docs/content/guides/database/extensions/pgrouting.mdx) [S]

### Inferences
#### Engines and deployment [I]
- **OSRM:** cell→POI tables (car and foot; CH or MLD). One container per country extract from [Geofabrik](https://download.geofabrik.de/africa.html) [L] (Kenya, Nigeria, Ghana, South Africa), rebuilt monthly. Record `osm_snapshot` and the profile hash.
- **Valhalla:** isochrones, time-dependent costing (when historical speeds can be supplied), per-request costing and multimodal routing.
- **ORS public API:** prototyping only.
- **GraphHopper:** the open-source core is Apache-2.0, but its Matrix API is believed to be commercial-only (not verified).
- **pgRouting on Supabase:** ad-hoc paths only; too slow for OD matrices.

#### OD design [I]
- Origins: H3 res-9 centroids within the focal business's prior P99 (≈ 100–400 cells).
- Destinations: the focal business plus candidate rivals.
- Prefilter by Euclidean distance ≤ 1.5 × R_max.
- Chunk requests to the server's table limit (e.g., 50×50 on a default OSRM).
- Store results in `travel_time(origin_h3_r9, poi_id, mode, time_band, minutes, metres, snap_m, engine, osm_snapshot)`.

#### Failure modes [I]
- **Snapping:** a snap distance over 150 m is common in informal settlements where OSM lacks footpaths. Fall back to Euclidean × c_mode and flag the pair.
- **Barriers or broken networks:** a d_net/d_E ratio above 2.5 flags either. Inspect before trusting.
- **Access tags:** `access=private` and `barrier=gate` can wrongly block real pedestrian access.
- **Congestion:** OSRM car profiles are free-flow, which badly understates peak times in Nairobi and Lagos (magnitude not verified). Estimate a multiplier κ(city, hour-band, distance-band) = median(t_traffic / t_osrm) from ~1,000 stratified OD pairs per city per quarter using a traffic-aware commercial API (Google Routes list prices not verified in-session). This is a minor cost.
- **Walking speed:** the usual default is about 5 km/h (≈ 12 min per network km). Confirm each engine's profile default.

#### Mode mixing in dense African cores [I]
- Logsum generalised time across walking, matatu/danfo/trotro and car or moto-taxi:

  `t_eff = −(1/μ)·log Σ_m exp(−μ·(t_m + c_m))`

  - Transit times include access, wait and in-vehicle time, computed from GTFS via [r5py](https://r5py.readthedocs.io/) [L].
  - μ = 0.2/min is a placeholder.
- Simpler default: walking for d ≤ 2 km; min(walk, transit) beyond.
- Motorbike taxis (boda-boda, okada) have no standard profile. Use the car network with a lower congestion multiplier, calibrate it, and respect local bans.

#### Circuity and when straight-line is acceptable [I]
- Estimate circuity c = median(d_net/d_E) per city, mode and distance band from about 2,000 random pairs 0.3–5 km apart. Short trips usually show higher circuity; verify per city.
- Euclidean × c is acceptable for:
  - prefiltering and ranking;
  - barrier-free dense grids;
  - features averaged over many rivals, where errors cancel;
  - areas where the OSM walk network is visibly incomplete, since a broken network can be worse than straight lines.
- It is not acceptable for transit-dependent customers (per Apparicio), barrier-rich cities, or peak-hour car access to CBDs.

#### Tests [I] [C]
- d_net ≥ 0.98·d_E for ≥ 99% of pairs, allowing for snapping.
- Spearman correlation between d_net and d_E > 0.9 per city.
- Determinism: the same snapshot gives identical tables.
- Golden file of 500 fixed OD pairs per city; report drift percentiles on each OSM rebuild.
- [C] Haversine sanity check: 1° of latitude = 111.195 km. Haversine and PostGIS geography distance agree within 0.5%.
- Foot-profile symmetry within 5%.

### Gaps
- Not checked (search budget exhausted): Google Routes / Distance Matrix 2026 per-element pricing and free caps.
- ORS quotas conflict between sources.
- No measured congestion multipliers for African cities were found.
- OSM footpath completeness in informal settlements is unquantified.
- Not verified: whether the Digital Matatus GTFS is still current, whether GTFS exists for Lagos, Accra and Johannesburg minibus systems, and GraphHopper's open-source matrix status.

## Q4. Competitor discovery and entity resolution

### Takeaway
No single source is complete in African cities. Combine three layers, with strict entity resolution and a per-competitor existence probability:
- **OSM** (ODbL; uneven points of interest)
- **Overture Places** (> 85M places; CDLA-Permissive-2.0/Apache-2.0; no OSM data; mostly low confidence scores in Kenya and Nigeria)
- **Foursquare OS Places** (100M+ places; Apache-2.0; monthly updates; date_closed present but incomplete)

Use Google Places only transiently and within its terms (store place_id only; lat/lng ≤ 30 days). Measure completeness with field audits: informal vendors are mostly missing from every source, and the miss rate correlates with income.

### Cited Findings
- Overture places (search-reported, as of September 2026):
  - Licensed CDLA Permissive 2.0 and Apache 2.0, with some CC0. Contains no OSM data and carries no ODbL share-alike obligation.
  - More than 85M features from AllThePlaces, BrightQuery, DAC, Foursquare, Krick, Microsoft, PinMeTo, RenderSEO and Meta.
  - `confidence` ∈ [0,1] estimates whether the place exists. A `taxonomy` property holds the primary category, hierarchy path and alternates.

  Sources: [Overture Places guide](https://docs.overturemaps.org/guides/places/); [Place schema](https://docs.overturemaps.org/schema/reference/places/place/); [AWS Open Data registry](https://registry.opendata.aws/overture/) [S]
- About 40% of Overture records overall derive from OSM; 0% of Places do — [OSM wiki: Overture Maps](https://wiki.openstreetmap.org/wiki/Overture_Maps) [S]
- One evaluation reports lower confidence inside an upstream "geocode-fallback" cluster (mean 0.729 vs. 0.810 elsewhere); attribution uncertain — surfaced alongside [dtcc-core issue #116](https://github.com/dtcc-platform/dtcc-core/issues/116) [S]
- Reported share of Overture places with confidence > 0.9 is 0.22% in Kenya and 0.11% in Nigeria, with most places below 0.5; attribution uncertain — surfaced with [Echo Analytics' analysis](https://www.echo-analytics.com/blog/analyzing-overture-maps-foundations-places-data) [S, unverified]
- Foursquare OS Places has 100M+ POIs and 22 core attributes, updates monthly, and is licensed Apache 2.0 (commercial use allowed). It includes `date_closed`, but closed POIs *without* a date_closed value appear more common — [Foursquare announcement](https://foursquare.com/resources/blog/products/foursquare-open-source-places-a-new-foundational-dataset-for-the-geospatial-community/); [Simon Willison](https://simonwillison.net/2024/Nov/20/foursquare-open-source-places/); [OSM community forum](https://community.openstreetmap.org/t/foursquare-releases-100m-poi-dataset-under-apache-2-0/121883); [Hugging Face](https://huggingface.co/datasets/foursquare/fsq-os-places) [S]
- Google Places API (New) prices per 1,000 calls, from secondary aggregators (Google's own page was blocked):

  | SKU | Price per 1,000 | Free / month |
  |---|---|---|
  | Nearby Search Pro, 0–100K | $32.00 | 5,000 |
  | Nearby Search Pro, 100–500K | $25.60 | — |
  | Nearby Search Pro, 500K–1M | $19.20 | — |
  | Nearby Search Enterprise | — | 1,000 |
  | Place Details Essentials | $5 | — |
  | Place Details Pro | $17 | — |
  | Place Details Enterprise | $20 | — |
  | Place Details Enterprise + Atmosphere | $25 | — |
  | IDs-only SKUs | free | — |

  Nearby Search has no Essentials tier. Sources: [bizcollect](https://bizcollect.dev/blog/google-places-api-pricing); [Woosmap](https://www.woosmap.com/blog/google-maps-api-pricing-breakdown); [Google pricing list](https://developers.google.com/maps/billing-and-pricing/pricing) [S]. Very low deep-volume tiers reported by aggregators (e.g., $2.40/1,000 at 5M+) look implausible; check them.
- Google caching rules: place_id may be stored indefinitely and lat/lng cached for ≤ 30 days; there is no caching exception for names, ratings or photos — [Google Maps Platform Service Specific Terms](https://cloud.google.com/maps-platform/terms/maps-service-terms) [S]
- OSM *building* completeness (not POIs) is about 30% in sub-Saharan Africa. Globally, 1,848 urban centres exceed 80% and 9,163 are below 20%. Completeness is low in Egyptian and Ethiopian cities and higher in Tanzania, Uganda and West Africa. Humanitarian mapping has reduced the inequality but not removed it — [Herfort et al. (2023), Nature Communications](https://www.nature.com/articles/s41467-023-39698-6) [S]
- Among 25 cities in a global healthy-cities indicator study, Maiduguri (Nigeria) was an outlier for how poorly OSM represented daily-living amenities — [arXiv 2205.05240](https://arxiv.org/pdf/2205.05240) [S]
- Firm-centric competitor sets defined by a text-cosine threshold (TNIC) overlap 62% with Capital IQ peers — [Hoberg & Phillips (2016)](https://econpapers.repec.org/RePEc:ucp:jpolec:doi:10.1086/688176) [S]
- OSM data is licensed ODbL 1.0: share-alike applies to derivative databases that are publicly used, and attribution to produced works — [OSM copyright](https://www.openstreetmap.org/copyright) [L]. The public Overpass instance asks users to stay under roughly 10,000 queries/day and 1 GB/day — [Overpass API wiki](https://wiki.openstreetmap.org/wiki/Overpass_API) [L, not verified]
- Probabilistic record linkage: [Fellegi & Sunter (1969), JASA](https://doi.org/10.1080/01621459.1969.10501049) [L]; Python implementation: [Splink](https://github.com/moj-analytical-services/splink) [L]

### Inferences
#### Category crosswalk (LAEE taxonomy; OSM tag pages are [L]) [I]
| LAEE category | OSM tags | Vernacular cues for text classification |
|---|---|---|
| Café | `amenity=cafe`, `shop=coffee` | — |
| Restaurant | `amenity=restaurant` | East-African "hotel" usually means eatery |
| Local eatery / fast food | `amenity=fast_food`, `amenity=food_court` | kibanda, buka/bukka, mama put, chop bar |
| Supermarket | `shop=supermarket` | — |
| Kiosk / convenience | `shop=convenience`, `shop=general`, `shop=kiosk`, `shop=greengrocer`, `amenity=marketplace` | duka, spaza, mama mboga |
| Salon / barber | `shop=hairdresser` (+ `hairdresser=barber`), `shop=beauty` | kinyozi |
| Pharmacy | `amenity=pharmacy` (+ `dispensing=*`), `healthcare=pharmacy`; `shop=chemist` = drugstore without a pharmacist | "chemist" commonly means pharmacy |
| Auto repair | `shop=car_repair`, `shop=tyres`, `shop=car_parts` | fundi, jua kali garage |
| Professional services | `office=lawyer\|accountant\|tax_advisor\|estate_agent\|insurance\|consulting` | — |
| Mobile-money agent (activity proxy) | `amenity=money_transfer` | East-African use of `amenity=mobile_money_agent` (verify) |

- Map the Overture, Foursquare and Google category keys explicitly. Pin the release and schema version, because Overture's category fields changed across releases.
- The LLM (v1 §2) only classifies free-text names into this taxonomy, and is validated against labels first.

Prototype Overpass query (production should use Geofabrik extracts with `osmium tags-filter`):
```
[out:json][timeout:120];
( nwr(around:3000,-1.2864,36.8172)["amenity"~"^(cafe|restaurant|fast_food|food_court)$"];
  nwr(around:3000,-1.2864,36.8172)["shop"~"^(coffee|tea|bakery)$"]; );
out center tags;
```

#### Ingest → dedup → existence pipeline [I]
1. **Monthly snapshots.** Overture GeoParquet filtered by bbox (DuckDB), Foursquare parquet, OSM pbf. Keep `snapshot_date` and native IDs.
2. **Clean.**
   - Flag or drop points that share exact coordinates with 5 or more other POIs, or that sit on admin centroids. These are geocode fallbacks and create fake CBD clusters.
   - Drop records with Overture confidence below 0.3 or with a Foursquare date_closed.
3. **Block.** Compare records in the same H3 res-9 cell or a k = 1 neighbour (≤ ~0.7 km).
4. **Score pairs:**
   - Name similarity after normalisation: lowercase; strip accents, punctuation, legal suffixes (Ltd, Limited, Enterprises) and generic words (shop, hotel, salon); then Jaro–Winkler or token-set ratio.
   - Distance: ≤ 50 m strong evidence, ≤ 150 m moderate.
   - Taxonomy compatibility; phone, website or brand match.
   - Combine with Fellegi–Sunter weights estimated by EM in Splink.
5. **Cluster.** Connected components, with one constraint: no two records from the *same* source with different native IDs. This keeps chain branches separate (e.g., two Java House or Chicken Inn outlets 300 m apart).
6. **Entity table** with lineage and `p_exists`, from a logistic model calibrated on field audits. Inputs: number of sources, Overture confidence, Foursquare date_refreshed age, OSM last-edit age, and Google business_status (fetched transiently).
7. **Features** use expected rivals: π_k × kernel × m_jk.

#### Completeness and the informal sector [I]
- **Field audits.** Audit 20–40 H3 res-9 cells per city, stratified by density and wealth proxy. Enumerate every target-category business, including kiosks and street vendors. Fit per-source capture rates π_s(cell features).
- **Capture–recapture.** Lincoln–Petersen N̂ = n1·n2/m assumes independent sources. Overture ingests Foursquare, so the two are not independent; use the audits as ground truth instead.
- **Bias.** The informal undercount is larger in poorer areas (non-classical measurement error), which can distort elasticity-by-income results. Scale expected rival counts by 1/π̂(cell), or control for π̂.

#### Google use [I]
- Use Google only to verify specific candidate POIs and read business_status transiently, storing place_id. Never build the persistent competitor table from Google content.
- Illustrative cost of a Nairobi sweep:
  - Nairobi is about 700 km². Covering it with 500 m-radius circles takes about 1,100 tiles, at ~0.65 km² of effective coverage each.
  - At 8 categories that is about 8,600 Nearby Search Pro calls.
  - (8,600 − 5,000 free) × $0.032 ≈ $115 per sweep at list price.
  - Per-call result caps (not verified) may force smaller tiles in dense areas.
  - Because results cannot be cached, sweeps recur, which is a cost and terms-of-service risk.

#### Licensing hygiene [I]
- Keep OSM-derived tables separate from proprietary data: a combined database offered to customers risks ODbL share-alike. Get legal review.
- Keep attribution files for Overture and Foursquare.
- Business registries list legal entities, not outlet coordinates, so they help only to verify formal firms: Kenya BRS (eCitizen), Nigeria CAC, Ghana ORC, South Africa CIPC [L]. Access and costs are not verified.

#### Tests [I]
- Synthetic dedup benchmark with injected duplicates (jittered names and coordinates): precision ≥ 0.95, recall ≥ 0.90.
- Chain branches 300 m apart remain two entities.
- A salon and a pharmacy at the same coordinates remain distinct.
- Idempotent entity IDs for the same snapshot.
- At least 98% of IDs persist month to month when sources have not changed.
- The geocode-fallback detector catches injected stacked points.
- Taxonomy mapping covers ≥ 95% of records per city.

### Gaps
- No quantitative POI completeness was found for Nairobi, Lagos, Accra or Johannesburg by category; building completeness is not a substitute for POI completeness.
- Not verified (docs blocked): Overture's operating-status field and its taxonomy keys.
- Not verified: Google Nearby Search (New) per-call result caps and deep volume tiers.
- Municipal business-permit registers (e.g., Nairobi County) were not researched.

## Q5. Measuring competitive pressure and substitutability

### Takeaway
IO evidence shows competitive effects are very local and type-specific, shrink after the first few rivals, and often move quantities more than prices. Demographics explain more of store-level elasticity than competition does. Replace v1's indices with:
- (i) a pressure measure derived from a spatial nested logit, which maps directly into own- and cross-price elasticities;
- (ii) estimated rival counts by distance band × similarity class;
- (iii) a calibrated similarity score, with embeddings as one input;
- (iv) a log relative price with normalised weights and coverage gating.

### Cited Findings
- Seim (2006) models entry with endogenous product-type choice. Rivals' differentiated products have non-uniform competitive effects, distant competitors affect payoffs much less, and the returns to differentiation are significant — [Seim (2006), RAND 37(3)](https://ideas.repec.org/a/rje/randje/v37y20063p619-640.html) [S]
- Firms have a strong incentive to differentiate (motels choose quality tiers), and competitor effects differ by product type — [Mazzeo (2002), RAND 33(2)](https://econpapers.repec.org/RePEc:rje:randje:v:33:y:2002:i:summer:p:221-242) [S]
- Geography and ownership jointly determine fast-food prices; mergers between McDonald's franchisees can significantly raise prices — [Thomadsen (2005), RAND 36(4):908–929](https://econpapers.repec.org/RePEc:rje:randje:v:36:y:2005:4:p:908-929) [S]
- Davis (2006) estimates travel costs and the extent of local market power for movie theatres, with 5-mile rings dominant — [Davis (2006)](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1756-2171.2006.tb00066.x) [S]
- A semiparametric model of spatial price competition compares global and local competition; US wholesale gasoline competition is mostly local — [Pinkse, Slade & Brett (2002), Econometrica 70(3):1111–1153](https://doi.org/10.1111/1468-0262.00321) [S content; L DOI]
- More stations nearby lowers both price levels and price dispersion — [Barron, Taylor & Umbeck (2004), via 2023 summary](https://www.tandfonline.com/doi/full/10.1080/15567249.2023.2189330) [S]. Related: [Lewis (2008)](https://mslewis.people.clemson.edu/Research/Lewis_Price_Dispersion.pdf); [Chandra & Tappata (2011)](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1756-2171.2011.00150.x); [Davis (2025), J. Industrial Economics](https://onlinelibrary.wiley.com/doi/10.1111/joie.12418) [S, findings not read]
- Supercenter entry: −16% incumbent revenue within 1 mile, with no causal price effect — [Arcidiacono et al. (2020)](https://www.aeaweb.org/articles?id=10.1257%2Fapp.20180047) [S]
- Ellickson, Grieco & Khvastunov estimate spatial discrete choice with overlapping choice sets and no ex-ante markets. The method works even without price data and finds cross-format substitution — [Ellickson, Grieco & Khvastunov (2020)](https://onlinelibrary.wiley.com/doi/abs/10.1111/1756-2171.12310) [S]
- Demographics plus competition explain 67% of store-level elasticity variation, with demographics dominant — [Hoch et al. (1995)](https://www.researchgate.net/publication/237130017_Determinants_of_Store-Level_Price_Elasticity) [S]
- Firm-centric text-similarity competitor networks — [Hoberg & Phillips (2016)](https://econpapers.repec.org/RePEc:ucp:jpolec:doi:10.1086/688176) [S]
- Logit and nested-logit inversion and elasticities — [Berry (1994), RAND 25(2)](https://doi.org/10.2307/2555829) [L]. Diminishing effects of additional entrants — [Bresnahan & Reiss (1991)](https://doi.org/10.1086/261786) [L]
- Developing-country evidence (from memory; details unverified):
  - Randomised entry of retailers in the Dominican Republic lowered prices and raised quality — [Busso & Galiani, NBER w20054](https://www.nber.org/papers/w20054) [L]
  - An RCT subsidising trader entry in Kenyan maize markets found evidence of substantial market power among local traders — [Bergquist & Dinerstein (2020), AER 110(12)](https://doi.org/10.1257/aer.20171397) [L]
  - Mobile-phone rollout reduced grain price dispersion across Niger's markets — [Aker (2010), AEJ: Applied 2(3)](https://doi.org/10.1257/app.2.3.46) [L]
- Chains often price uniformly across stores despite different local elasticities, so local competition may not show up in chain-outlet prices — [DellaVigna & Gentzkow (2019), QJE 134(4)](https://doi.org/10.1093/qje/qjz019) [L]
- Two-step floating catchment (supply-to-demand ratio) method — [Luo & Wang (2003), EPB](https://doi.org/10.1068/b29120) [L]. E2SFCA zone weights 1.00 / 0.68 / 0.22 — [Luo & Qi (2009)](https://www.sciencedirect.com/science/article/abs/pii/S1353829209000574) [S]

### Inferences
#### A. Demand-consistent pressure (primary definition; replaces §8/§9) [I]
Spatial nested logit, with nests g = LAEE subcategory and nesting parameter σ ∈ [0,1):
```
u_cj = γ log A_j − β t_cj − α_p p_j + ξ_j + ζ_cg + (1−σ) ε_cj
ω_cj = D_c s_cj / Σ_c' D_c' s_c'j
Pressure_j = 1 − Σ_c ω_cj · s_{cj|g}                  (expected within-nest share lost to rivals)
ε_jj = −α_p p_j Σ_c ω_cj [ 1/(1−σ) − σ/(1−σ)·s_{cj|g} − s_cj ]
ε_jk = +α_p p_k Σ_c ω_cj [ σ/(1−σ)·s_{ck|g} + s_ck ]      (same nest; other nests: α_p p_k Σ_c ω_cj s_ck)
```
- With σ = 0: ε_jj = −α_p p_j (1 − s̄_j). More nearby substitutes lower own share, which makes demand more elastic. This is the testable link v1 lacks.
- Without rival prices, α_p comes from the focal firm's own price variation (DML, v1 §16). Pressure_j and the logit-implied ε_jk structure then act as features and priors, and free cross-elasticity estimates are shrunk toward that structure (fixes §31/§32).
- Similarity enters *inside* the logit, via nests or a log m_jk utility shifter, never as a multiplied raw factor.

#### B. Reduced-form band index (for DML nuisance features and entry studies) [I]
```
N_j[b,s] = Σ_k π_k · 1[t_jk ∈ b] · 1[class(j,k) = s]
bands (half-open, km or walk-minutes): [0,0.25) [0.25,0.5) [0.5,1) [1,2) [2,5)
classes s: same leaf & price tier ±1 | same leaf, other tier | same parent | adjacent category
CI_j = Σ_b Σ_s θ_bs · log(1 + N_j[b,s])
```
- θ_bs are *estimated*, and log(1+N) gives diminishing returns.
- Impose a monotone decline of |θ_b| with distance (isotonic) only after testing.
- Leave the innermost same-category band sign-free, so agglomeration can show up.

#### C. Substitutability m_jk [I]
- **Features:**
  - Taxonomy distance: 1 = same leaf, 0.5 = same parent, 0.2 = adjacent, else 0.
  - Price-tier compatibility: exp(−|Δtier|).
  - Tag Jaccard (cuisine or service).
  - Text cosine from a pinned multilingual embedding of name + category + menu/services, excluding review sentiment.
  - Chain vs. independent; opening-hours overlap.
- **Model:** logistic regression P(competitor | x) with isotonic calibration.
- **Labels:** about 300 human-labelled pairs per city × category, or co-visitation and customer overlap where platform data exist.
- **Targets:** AUC ≥ 0.85; report the Brier score. Pick the threshold by F1 (Hoberg–Phillips style).
- **Versioning:** a model change means full recomputation plus a feature A/B check.

#### D. Relative price (replaces §10 CPI and §61 PriceGap) [I]
```
w_jk = π_k · K_θ(t_jk) · m_jk ;   ŵ_jk = w_jk / Σ_k w_jk
rp_j = log P_j − Σ_k ŵ_jk log P_k        (positive = more expensive than the weighted market)
coverage_j = Σ_{k priced} w_jk / Σ_k w_jk ; emit rp_j only if coverage ≥ 0.5 and ≥ 3 priced rivals, else NA
```
- Match items on size-normalised units: KSh per 500 ml soda, per standard haircut, per ¼ chicken.
- With only ordinal price tiers: rp_tier = tier_j − Σ ŵ tier_k, flagged low-confidence.
- **Tests:**
  - Replicating every rival leaves rp unchanged; v1's formula fails this.
  - Scaling all prices by 1.1 leaves rp unchanged.
  - A rival at t → ∞ has no effect.
  - The sign convention holds.

#### E. Saturation (2SFCA) as a complementary feature [I]
```
R_k = S_k / Σ_c Pop_c K(t_ck)
A_c = Σ_k R_k K(t_ck)
Sat_j = Σ_c ω_cj A_c
```
Use the Luo–Qi zone weights (1.00 / 0.68 / 0.22 at 0–10 / 10–20 / 20–30 min) as the non-parametric default kernel.

#### F. Priors from the evidence [I]
- Expect demographic and catchment features to explain more elasticity heterogeneity than competition (Hoch).
- Expect entry to move quantities more than prices (Arcidiacono). LAEE should *test* per category whether a new rival justifies a price change, not assume it.

#### G. Tests [I]
- Logit elasticities match finite-difference elasticities of simulated demand within 1e-4.
- Pressure rises monotonically as a rival moves closer, and a rival at infinity changes nothing.
- Duplicating a rival raises logit pressure, so a regression test must assert that dedup runs *before* features are built.
- Band boundaries are half-open and consistent.
- With no true competitor effect, the θ_bs are jointly insignificant at the nominal rate.

### Gaps
- No African estimates were found of how local competitor density changes SMB own-price elasticity. The Busso–Galiani and Bergquist–Dinerstein details could not be re-checked after the search budget ran out.
- Thomadsen's and Davis's numerical elasticities and markups were not retrieved.
- No labelled competitor-similarity datasets for African cities exist that I could find.
- Online availability of menus and item prices for African SMBs (WhatsApp catalogues, delivery apps) was not researched.

## Q6. Fine-grained demographics with African coverage, and aggregation to a catchment

### Takeaway
Build demographics from layered open rasters:
- **Population:** WorldPop R2025A (100 m, CC BY 4.0, annual 2015–2030, alpha) or GHSL R2023A (100 m, five-yearly 1975–2030). For Nigeria, GRID3 bottom-up v2.x.
- **Built form and growth:** Google Open Buildings v3 (CC BY 4.0 or ODbL) and its 2.5D Temporal series (2016–2023).
- **Activity:** VIIRS (500 m, mostly CC BY 4.0).
- **Wealth:** the Relative Wealth Index only under a commercial licence, since it is CC BY-NC.
- **DHS:** region-level calibration only, because cluster coordinates are displaced 0–2 km urban, 0–5 km rural and up to 10 km for 1% of rural clusters.

Aggregate rasters to H3 res 9 with mass-conserving assignment, then to the catchment with normalised kernel weights: sums for extensive variables, population-weighted means for intensive ones.

### Cited Findings
- WorldPop Global2 R2025A: 100 m (3 arc-second) annual population counts for 242 countries, 2015–2030, using constrained top-down disaggregation; CC BY 4.0. Released September 2025 as an alpha that "may change" — [Release statement](https://data.worldpop.org/repo/prj/Global_2015_2030/R2025A/doc/Global2_Release_Statement_R2025A_v1.pdf); [WorldPop hub](https://hub.worldpop.org/geodata/listing?id=135) [S]. A mirror lists age/sex rasters at 1 km — [mirror](https://github.com/yuiseki/worldpop-jp-2026-01) [S]
- GHSL GHS-POP R2023A: 100 m and 1 km in Mollweide (EPSG:54009), five-year epochs 1975–2030. Built from GPWv4.11 census inputs and disaggregated by built-up volume. Open and free with source acknowledgement under EC reuse rules — [JRC catalogue](https://data.jrc.ec.europa.eu/dataset/2ff68a52-5b5b-4a22-8f40-c41da8332cfe); [GHSL datasets](https://human-settlement.emergency.copernicus.eu/datasets.php) [S]. EC reuse policy generally means CC BY 4.0 [L]
- Meta/CIESIN HRSL: about 30 m population estimated from census plus building detection; CC BY 4.0; no longer updated as of 2024; Africa tiles on AWS and HDX — [AWS registry](https://registry.opendata.aws/dataforgood-fb-hrsl/); [HDX Africa](https://data.humdata.org/dataset/highresolutionpopulationdensitymaps); [Digital Earth Africa](https://docs.digitalearthafrica.org/en/latest/sandbox/notebooks/Datasets/High_Resolution_Population_Density.html) [S]
- Relative Wealth Index: 2.4 km micro-estimates built from satellite, mobile-network, topographic and Facebook connectivity data. The paper covers 135 LMICs and HDX hosts 93. Licensed CC BY-NC 4.0 — [Chi, Fang, Chatterjee & Blumenstock (2022), PNAS 119(3)](https://www.pnas.org/content/pnas/119/3/e2113658119.full.pdf); [HDX](https://data.humdata.org/dataset/relative-wealth-index) [S]. Values are relative within each country and not comparable across countries [L]
- Google Open Buildings v3 has 1.8B building detections over 58M km² (Africa, South/Southeast Asia, Latin America & Caribbean), from inference run in May 2023. Users choose CC BY 4.0 *or* ODbL. The 2.5D Temporal product gives annual building presence, fractional counts and heights for 2016–2023 at 4 m effective resolution, derived from Sentinel-2 — [Open Buildings](https://sites.research.google/gr/open-buildings/); [Temporal](https://sites.research.google/gr/open-buildings/temporal/) [S]
- VIIRS nighttime lights from the EOG / Payne Institute: annual VNL at 500 m (15 arc-sec); many products CC BY 4.0 with attribution — [EOG VNL](https://eogdata.mines.edu/products/vnl/) [S]. Night lights serve as an activity proxy where official statistics are weak — [Henderson, Storeygard & Weil (2012), AER 102(2)](https://doi.org/10.1257/aer.102.2.994) [L]
- DHS displaces cluster coordinates in a random direction: urban 0–2 km, rural 0–5 km, and a random 1% of rural clusters up to 10 km. Displacement stays within the country, survey region and admin-2 — [DHS SAR7](https://dhsprogram.com/pubs/pdf/SAR7/SAR7.pdf); [DHS GPS data](https://dhsprogram.com/Methodology/GPS-Data.cfm) [S]
- GRID3 Nigeria v2.0/v2.1: about 100 m bottom-up estimates from roughly 1,100 surveyed locations plus geospatial covariates, with age-sex groups and uncertainty. Produced by WorldPop with Nigeria's National Population Commission — [HDX](https://data.humdata.org/dataset/bottom-up-gridded-population-estimates-for-nigeria-version-2-0); [GRID3 v2.1](https://data.grid3.org/datasets/grid3-nga-population-v2-1); [README](https://data.worldpop.org/repo/wopr/NGA/population/v2.0/NGA_population_v2_0_README.pdf) [S]
- H3-native population layers exist, e.g., Kontur's "Population Density for 400m H3 Hexagons" for South Africa — [HDX Kontur ZAF](https://data.humdata.org/dataset/kontur-population-south-africa) [S]
- OSM building footprints are only about 30% complete in sub-Saharan Africa, so they are a poor density proxy — [Herfort et al. (2023)](https://www.nature.com/articles/s41467-023-39698-6) [S]

### Inferences
#### Feature menu (each tagged with its reference year) [I]
| Feature | Source | Notes |
|---|---|---|
| Residential population | WorldPop R2025A and GHSL | Keep both; flag ward-level disagreement > 30% |
| Growth | WorldPop 2020→2025; Open Buildings Temporal fractional-count growth 2016→2023 | — |
| Building count, area, mean height | Open Buildings v3 / 2.5D | Density and income proxies |
| Radiance per capita | VIIRS | Saturates and blooms in CBDs |
| Wealth | RWI (licensed only) or an in-house proxy | Proxy: building height × footprint × radiance per capita, calibrated to DHS wealth quintiles at admin-1 |
| Commercial intensity | Overture POI density | — |
| Daytime population | GHSL R2023A non-residential built-up split (L, not verified) plus POI/office density | Needed for CBD and market businesses; residential rasters miss commuters |

#### Aggregation [I] [C]
1. **Raster → H3 res 9.** Assign each raster cell to the H3 cell containing its centroid and *sum* counts (mass-conserving).
   - [C] A 3-arc-sec cell is about 92.7 m × 92.7 m ≈ 0.0086 km² at the equator. An average res-9 cell is 0.1053 km², so ≈ 12 WorldPop cells per res-9 cell. A res-10 cell (0.0150 km²) holds only ≈ 1.75, so use res 10 only with HRSL (≈ 16 cells per res-10 cell).
   - For intensive rasters (RWI 2.4 km, VIIRS 500 m), give each res-9 cell the value of its containing tile, and use population-weighted means when aggregating up.
2. **Polygon → H3.** Dasymetric allocation x_h = X_p · Pop_{h∩p} / Pop_p, using gridded population as the ancillary layer. Fall back to area weighting only if no population layer is available.
3. **H3 → catchment:**
   ```
   v_c = Pop_c · K_θ(t_cj) ;  ω_c = v_c / Σ v_c       (use fitted ω_cj from Q2 when available)
   M_j = Σ_c v_c                                      (effective population: extensive)
   X̄_j = Σ_c ω_c X_c                                  (intensive means)
   Var_j = Σ_c ω_c [ σ_c² + (X_c − X̄_j)² ]            (law of total variance; σ_c² = 0 if unknown)
   ```
4. **CRS.** WorldPop and HRSL use WGS84 arc-second grids, whose cell area shrinks with cos(lat); this matters at Johannesburg (≈ 26°S). GHSL is equal-area Mollweide. Compute cell areas explicitly.
5. **Temporal hygiene.** WorldPop's 2026–2030 figures are projections. Never use them as observed values in backtests (v1 §74).

#### §57 proxies (resolution / lag) [I]
| Proxy | Resolution | Lag / cadence |
|---|---|---|
| VIIRS monthly | 500 m | ~1–2 months |
| Open Buildings Temporal | 4 m | Annual; ends 2023 |
| POI births/deaths from Overture and Foursquare snapshots | Point | Monthly |
| National CPI, fuel and FX | National | Monthly |

Drop any sub-city input that is unavailable in a country.

#### Tests [I]
- Mass conservation: H3 sums equal raster sums inside the study polygon within 0.5%.
- Weighting a constant field returns the constant.
- Weights sum to 1.
- Intensive means do not change when population is duplicated across cells with identical values.
- A latitude-area test for WGS84 rasters.
- Dasymetric allocation preserves polygon totals.
- Licence guard: production builds refuse NC-licensed layers unless a licence flag is set.

### Gaps
- Not verified (search budget exhausted): small-area census availability and licences for Kenya 2019 (sub-location/ward), South Africa 2022 (small-area layer) and Ghana 2021.
- Not verified: DHS commercial-use terms and the GRID3 licence.
- No accuracy assessment of WorldPop R2025A in informal settlements (e.g., Kibera, Makoko) was found.
- No validated open daytime-population layer for African cities was found.

## Q7. H3/PostGIS implementation and principled learning of the market radius/decay

### Takeaway
Use H3 res 9 (≈ 0.35 km between cell centres) for 250 m–2 km markets, res 8 (≈ 0.92 km) for 2–10 km and res 7 (≈ 2.44 km) for 10–25 km. Compute H3 in Python (h3 v4) because Supabase lacks h3-pg, and store it as BIGINT columns beside PostGIS geography with GiST indexes and exact `ST_DWithin` filters. Learn decay by hierarchical, profile-likelihood estimation; choose among ≤ 3 pre-registered kernels by spatial-block CV nested inside DML cross-fitting; report radii only as derived quantiles, never as argmax-tuned hyperparameters.

### Cited Findings
- The current H3 docs give these average hexagon statistics:

  | Res | Area (km²) | Edge (km) |
  |---|---|---|
  | 7 | 5.1612932 | 1.220629759 |
  | 8 | 0.7373276 | 0.461354684 |
  | 9 | 0.1053325 | 0.174375668 |

  Sources: [H3 cell statistics](https://h3geo.org/docs/core-library/restable/); [v3 table](https://h3geo.org/docs/3.x/core-library/restable/) [S]
- Supabase supports PostGIS and pgRouting; h3-pg is unlisted and requested in an open issue — [issue #245](https://github.com/supabase/postgres/issues/245); [extensions](https://supabase.com/docs/guides/database/extensions) [S]
- Methods that avoid choosing a single radius:
  - Nonparametric distance-bin treatment-effect curves — [Butts (2023)](https://www.kylebutts.com/papers/did-rings/) [S]
  - Semiparametric distance-based competition — [Pinkse, Slade & Brett (2002)](https://doi.org/10.1111/1468-0262.00321) [L]
  - Market models with no ex-ante market definition — [Ellickson, Grieco & Khvastunov (2020)](https://onlinelibrary.wiley.com/doi/abs/10.1111/1756-2171.12310) [S]
- Blocked (spatial) cross-validation for spatially structured data — [Roberts et al. (2017), Ecography 40(8)](https://doi.org/10.1111/ecog.02881) [L]
- PostGIS details [L]:
  - `ST_DWithin(geography, geography, metres)` uses the index and is spheroidal by default — [ST_DWithin](https://postgis.net/docs/ST_DWithin.html)
  - `<->` gives index-assisted KNN ordering — [KNN](https://postgis.net/docs/geometry_distance_knn.html)
  - `ST_HexagonGrid` (PostGIS ≥ 3.1) builds planar hex grids, which are not H3 — [ST_HexagonGrid](https://postgis.net/docs/ST_HexagonGrid.html)

### Inferences
#### H3 numbers verified locally [C]
- **Edge lengths disagree.** h3-py 4.5.0's `average_hexagon_edge_length()` returns older values (res 7: 1.406476; res 8: 0.531414; res 9: 0.200786 km) that differ from the docs table above. Areas agree, so derive cell spacing from area.
- **Centre-to-centre spacing** √(2A/√3):

  | Res | Spacing (km) |
  |---|---|
  | 5 | 17.09 |
  | 6 | 6.459 |
  | 7 | 2.441 |
  | 8 | 0.923 |
  | 9 | 0.349 |
  | 10 | 0.132 |
  | 11 | 0.050 |

- **Disk sizes.** `grid_disk(k)` returns 3k(k+1)+1 cells (19, 37, 61, 91, 127, 397 for k = 2, 3, 4, 5, 6, 11).
- **Ring distances.** At res 9 in central Nairobi, ring centroids lie 0.360–0.369 km away for k = 1, 0.955–1.106 km for k = 3, and 1.577–1.843 km for k = 5.
- **API names.** Verified h3 v4 function names: `latlng_to_cell`, `grid_disk`, `grid_ring`, `cell_to_latlng`, `average_hexagon_area`.

#### Resolution rule [I]
- Choose the coarsest resolution whose spacing is ≤ R50/3, giving ≥ ~30 cells inside the median radius.
- Compute demographics at res 9 (Q6). Use the kernel's resolution for routing origins.
- Prefilter with k = ceil(R/spacing) + 1 (the +1 covers cell extent), then apply an exact distance filter.

#### Schema sketch (Supabase/PostGIS) [I]
```sql
create table poi (
  poi_id uuid primary key, name text, category_laee text, subcategory text, price_tier smallint,
  geog geography(Point,4326) not null,
  h3_r7 bigint not null, h3_r8 bigint not null, h3_r9 bigint not null,  -- computed in Python (h3 v4)
  p_exists real, first_seen date, last_seen date, sources jsonb, snapshot_date date not null);
create index poi_geog_gix on poi using gist (geog);
create index on poi (h3_r9); create index on poi (h3_r8); create index on poi (category_laee, snapshot_date);

create table h3_demog (h3_r9 bigint, ref_year int, pop_worldpop real, pop_ghsl real, bldg_n int,
  bldg_area_m2 real, bldg_h_mean real, viirs real, rwi real, src_versions jsonb, primary key (h3_r9, ref_year));

create table travel_time (origin_h3_r9 bigint, poi_id uuid, mode text, time_band text, minutes real,
  metres real, snap_m real, engine text, osm_snapshot date,
  primary key (origin_h3_r9, poi_id, mode, time_band, osm_snapshot));

create table kernel_params (category_laee text, city text, mode text, form text, beta real, beta_lo real,
  beta_hi real, n_businesses int, fitted_at timestamptz, version text,
  primary key (category_laee, city, mode, version));
```
- **Exact candidate query:** `where st_dwithin(geog, $origin::geography, $r_m) and category_laee = any($cats) and snapshot_date = $snap`.
- **Nearest-N:** `order by geog <-> $origin::geography limit $n`.
- **Planar operations** use local UTM: Nairobi EPSG:32737 (37S), Lagos EPSG:32631 (31N), Accra EPSG:32630 (30N), Johannesburg EPSG:32735 (35S).

#### Learning decay without overfitting (replaces §13) [I]
1. **Hierarchy.** log β_j = μ_cat + u_city + u_cat×density + v_j, with Gaussian random effects.
   - Fit by pooled Poisson (Q2.B) across platform businesses that have customer geography.
   - Shrink each business's β̂_j toward its group (empirical Bayes) in proportion to its precision.
   - Businesses with no customer data inherit the posterior mean.
2. **Uncertainty.** Use the profile-likelihood CI {β : 2[ℓ(β̂) − ℓ_p(β)] ≤ 3.84}. Propagate it by drawing β, or at least report sensitivity at β_lo and β_hi.
3. **Kernel form.**
   - Pre-register ≤ 3 candidates: exponential (default), Gaussian, truncated power.
   - Use spatial-block CV with ≥ 5 folds and blocks ≥ 2 × R80: H3 res-6 blocks (≈ 36 km²) for most SMB categories, res 5 for car-oriented ones.
   - Score on held-out Poisson log-likelihood.
   - If the winner beats the exponential by < 2 SE, keep the exponential.
4. **Sales-only case.** Estimate the band coefficients θ_b (Q5.B) inside the demand/DML model with cross-fitting. "Effective competitive reach" is the outermost band with a significant θ_b after Holm correction. This is a descriptive output, not a tuned hyperparameter.
5. **DML hygiene.** Fit kernel parameters on folds disjoint from the elasticity fold (or on earlier periods), then freeze them. Never tune them on the outcome being explained.
6. **Stability report.** Bootstrap the full selection procedure and report how often each form is selected and the distribution of R80.

#### Tests [I]
- **Prefilter completeness:** for 1,000 random origins, H3 ring + exact filter returns exactly the brute-force haversine set. This catches an off-by-one k.
- **Distance agreement:** PostGIS geography vs. Python haversine within 0.5%.
- **CI coverage:** profile CIs cover the truth 93–97% of the time over 500 simulations; Q2 [C] gave 93% over 300.
- **Spurious-radius test:** simulate data with no competitor effect.
  - v1's argmax-R rule always "finds" a radius with positive in-sample gain.
  - The replacement's band coefficients are jointly null at ≈ 5%.
- **Leakage canary:** features built from kernels fit on the evaluation fold must fail the canary, i.e., they inflate CV scores on a pure-noise outcome.

### Gaps
- The cause of the H3 docs-vs-library edge-length discrepancy was not established; area-based spacing was verified and should be used.
- Supabase's extension list may have changed since the issue was filed; recheck.
- No Supabase/PostGIS performance benchmarks were gathered.

## Q8. Competitor entry and exit as natural experiments: data and DiD/event-study design

### Takeaway
Treat openings and closings as staggered treatments whose effects vary with distance and whose dates are imprecise. Detect events from monthly open-data snapshots: first-seen/last-seen, Foursquare `date_closed`, Overture confidence drops, OSM lifecycle tags. Estimate with heterogeneity-robust event-study estimators (Callaway–Sant'Anna, Sun–Abraham, Borusyak–Jaravel–Spiess) on distance bins, with not-yet-treated and far-ring controls. Test pre-trends and outer-ring nulls (Butts), pool across many businesses, and benchmark magnitudes against the US grocery result: −16% revenue within 1 mile and no price response.

### Cited Findings
- Arcidiacono et al. combine 11 years of weekly scanner data with exact supercenter opening dates. They find −16% incumbent revenue within 1 mile, decaying quickly with distance, and no causal price effect despite large cross-sectional price gaps — [Arcidiacono et al. (2020)](https://www.aeaweb.org/articles?id=10.1257%2Fapp.20180047); [PDF](https://public.econ.duke.edu/~psarcidi/walmart.pdf) [S]
- Entry effects on incumbents' sales, employment and exit are localised (1994–2006 panel, store-level census data) — [Ellickson & Grieco (2013)](https://econpapers.repec.org/RePEc:eee:juecon:v:75:y:2013:i:c:p:1-14) [S]
- Ring-DiD assumptions and a nonparametric effect-vs-distance estimator, plus a December 2024 extension to spatial spillovers — [Butts (2023)](https://www.kylebutts.com/papers/did-rings/); [Butts (2024)](https://www.kylebutts.com/files/Spillover.pdf); [arXiv 2105.03737](https://arxiv.org/pdf/2105.03737) [S]
- Heterogeneity-robust staggered DiD estimators [L]:
  - [Callaway & Sant'Anna (2021), J. Econometrics 225(2)](https://doi.org/10.1016/j.jeconom.2020.12.001)
  - [Sun & Abraham (2021)](https://doi.org/10.1016/j.jeconom.2020.09.006)
  - [Borusyak, Jaravel & Spiess (2024), REStud 91(6)](https://doi.org/10.1093/restud/rdae007)
  - [de Chaisemartin & D'Haultfœuille (2020)](https://doi.org/10.1257/aer.20181169)
  - [Goodman-Bacon (2021)](https://doi.org/10.1016/j.jeconom.2021.03.014), which decomposes TWFE bias
- Ring-design precedent around point shocks — [Linden & Rockoff (2008), AER 98(3)](https://doi.org/10.1257/aer.98.3.1103) [L]. Entry locations are strategic, i.e. endogenous — [Seim (2006)](https://ideas.repec.org/a/rje/randje/v37y20063p619-640.html) [S]; [Jia (2008), Econometrica 76(6)](https://doi.org/10.3982/ECTA6649) [L]
- Event-date sources and limits:
  - Foursquare `date_closed` exists but is often missing for closed POIs — [Simon Willison](https://simonwillison.net/2024/Nov/20/foursquare-open-source-places/) [S]
  - Overture `confidence` encodes existence — [Overture guide](https://docs.overturemaps.org/guides/places/) [S]
  - Google content other than place_id cannot be stored, so no Google-based history can be built — [Google terms](https://cloud.google.com/maps-platform/terms/maps-service-terms) [S]
- Recent entry evidence in gasoline markets — [Davis (2025), J. Industrial Economics](https://onlinelibrary.wiley.com/doi/10.1111/joie.12418) [S, findings not read]

### Inferences
#### Event table [I]
```
competitor_event(poi_id, event_type ∈ {open, close, relocate, rebrand, format_change},
                 date_lo, date_hi, detected_in_snapshot, sources, p_event)
```
- **Open:** absent in snapshot t−1 and present in t with p_exists ≥ 0.7, not explained by dedup or an ID change. Date interval (t−1, t].
- **Close:** absent from 2 consecutive snapshots, *or* a Foursquare `date_closed`, *or* Overture confidence < 0.3, *or* OSM `disused:*` / `was:*` lifecycle tags.
- **Corroboration:**
  - A transient Google business_status check, storing only place_id plus our own derived flag (legal check required).
  - LAEE merchants reporting new neighbours (cheap and accurate in-network).
  - Periodic field audits.

#### Estimation design [I]
- **Outcomes:** the focal business's weekly or monthly log revenue, log transactions and units, plus its own price index (price channel).
- **Treatment:** the first opening of a similar rival (m_jk ≥ m*) in bin b ∈ {[0,.25), [.25,.5), [.5,1), [1,2)} km (or walk-minutes). Cohort g = month of entry.
- **Estimator:** Callaway–Sant'Anna ATT(g, e) per bin, with not-yet-treated units and units whose nearest entrant is ≥ 3 km as controls. Alternative: BJS imputation with α_j + γ_{city×t}.
- **Event window:** −12…+12 months, excluding e ∈ {−1, 0} (a "donut" for date uncertainty).
- **Diagnostics:**
  - Joint pre-trend test.
  - Outer-bin null (Butts's unaffected-outer-ring assumption).
  - Placebo entry dates.
  - Sensitivity to whether events are dated at date_lo or date_hi.
- **Inference:** SEs clustered by H3 res-6 block, or Conley SEs.
- **Pooling:** pool by category × city tier. Single-business estimates are not identified. Feed pooled effects as priors into business-level models (v1 §41).
- **Heterogeneity:** chain vs. independent entrant, price-tier gap, the incumbent's pre-entry Pressure_j, wealth proxy, mode share.
- **Elasticity channel:** a triple difference p × Post × Near tests whether entry makes own demand more elastic. Feasible only with own price variation.
- **Exits:** exits are often caused by local decline. Prefer exits driven by chain-level decisions, where corporate failure or restructuring closes outlets regardless of local demand. The Kenyan retail-chain collapses of 2017–2020 are candidates (unverified).

#### Data and power [I]
- Planning heuristic: ≥ 30–50 treated focal businesses per city × category, each with ≥ 6 months of pre-period. Confirm by simulation using each category's residual SD.
- POI snapshots at most monthly apart; outcomes weekly.

#### Tests [I]
- **Staggered-effects simulation** with dynamic, heterogeneous effects: TWFE is biased, possibly with the wrong sign, while CS and BJS recover the truth within 2 SE.
- **Spillover simulation:** with a true effect out to 1.5 km, an inner (< 1 km) vs. outer (1–2 km) ring estimator is biased toward 0, and the binned estimator recovers the curve.
- **Date-noise simulation:** shifting event dates by U(0, 3) months attenuates estimates; the donut plus binned event time limits the bias.
- **Dedup regression:** a source ID change must not create a fake open/close pair.

### Gaps
- Not measured: how long new businesses take to appear in Overture, Foursquare or OSM in African cities.
- Not verified: Overture's operating-status fields.
- No African SMB entry-effect estimates were found.
- Real power for SMB-level outcomes is unknown until LAEE data accumulate.
