# Local Adaptive Elasticity Engine (LAEE)
## Proprietary SMB Price Elasticity & Local Market Intelligence Engine

**Status:** Core algorithm specification  
**Version:** 1.0  
**Primary objective:** Predict how demand, revenue, and profit will change when an SMB changes price.

---

# 1. Executive Objective

Build an AI-powered pricing intelligence engine for small and medium-sized businesses.

The system must answer:

> "If this business changes the price of this product/service by X%, what is the expected change in quantity demanded, revenue, and profit given this business's exact local market, competitors, customers, and current external conditions?"

The system must NOT output a generic industry elasticity.

It must estimate a conditional local elasticity:

\[
\epsilon_{i,t} =
\frac{\partial \ln Q_{i,t}}
{\partial \ln P_{i,t}}
\]

where elasticity is conditional on:

- business
- product
- location
- local competitive environment
- customer demographics
- historical demand
- historical prices
- promotions
- seasonality
- calendar effects
- weather
- local economic conditions
- competitor pricing
- competitor entry/exit
- local events
- local news
- macroeconomic conditions
- supply/cost shocks
- demand shocks
- recent abnormal events

The system should ultimately predict:

1. Estimated own-price elasticity
2. Expected quantity change
3. Expected revenue change
4. Expected gross-profit change
5. Recommended price
6. Expected confidence interval
7. Main factors affecting the recommendation
8. Local market conditions
9. Potential risks

---

# 2. Critical Architectural Principle

## Llama is NOT the elasticity estimator.

Llama should NOT directly predict:

> "Elasticity = -1.43"

from a prompt.

That would be unreliable and difficult to validate.

Instead:

### Structured numerical models estimate elasticity.

### Llama interprets unstructured information.

Use Llama to transform text into structured variables.

Example:

```json
{
  "event_type": "employment_shock",
  "direction": "negative",
  "affected_sector": "technology",
  "geographic_scope": 0.87,
  "severity": 0.78,
  "duration_estimate_days": 180,
  "consumer_spending_probability": -0.63,
  "confidence": 0.91
}
```

Those numerical variables enter the actual prediction model.

This makes the system:

- testable
- reproducible
- auditable
- backtestable
- substantially less dependent on LLM hallucinations

---

# 3. Overall Architecture

```text
                    ┌──────────────────────┐
                    │  BUSINESS INPUT      │
                    │ price / sales / SKU  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  GEO MARKET ENGINE   │
                    │ micro-market mapping │
                    └──────────┬───────────┘
                               │
                               ▼
             ┌─────────────────────────────────────┐
             │      LOCAL ENVIRONMENT ENGINE       │
             │ competitors / demographics / traffic│
             └──────────────────┬──────────────────┘
                                │
                                ▼
             ┌─────────────────────────────────────┐
             │       EXTERNAL SHOCK ENGINE         │
             │ news / events / weather / economy   │
             │ Llama → structured shock variables  │
             └──────────────────┬──────────────────┘
                                │
                                ▼
             ┌─────────────────────────────────────┐
             │       DEMAND / CAUSAL ENGINE         │
             │ FE + DML + causal forest + ML       │
             └──────────────────┬──────────────────┘
                                │
                                ▼
             ┌─────────────────────────────────────┐
             │      PRICE SIMULATION ENGINE         │
             │ price → quantity → revenue → profit │
             └──────────────────┬──────────────────┘
                                │
                                ▼
                    ┌──────────────────────┐
                    │ SMB INSIGHT LAYER    │
                    │ recommendation       │
                    │ confidence            │
                    │ explanation           │
                    └──────────────────────┘
```

---

# 4. DATA MODEL

Every observation should ultimately exist at approximately:

```text
business_id
product_id
location_id
date
```

The core panel should look like:

```text
business_id
product_id
location_id
date

price
units_sold
revenue
cost
margin

promotion_flag
discount_percent
inventory
stockout_flag

competitor_count
competitor_weighted_price
competitor_distance
competitor_density

population
median_income
income_distribution
population_growth
age_distribution
education_distribution

traffic_index
foot_traffic_index

weather
temperature
precipitation

holiday
day_of_week
month
season

local_event_score
economic_shock_score
news_shock_score
consumer_confidence
unemployment
employment_growth

competitor_entry
competitor_exit

lagged_price
lagged_quantity
rolling_quantity
rolling_price
```

---

# 5. GEOLOCATION ENGINE

## 5.1 Do NOT define a market using ZIP code alone.

ZIP codes are administrative boundaries.

Consumer markets are behavioral.

Instead construct a **Dynamic Micro-Market**.

For each business, create a market around the business using several geographic definitions.

### Layer A — Physical Radius

Generate:

```text
250m
500m
1km
2km
3km
5km
```

depending on industry.

For a restaurant:

```text
250m → 1.5km
```

may matter more.

For a furniture store:

```text
5km → 25km
```

may matter more.

---

# 6. INDUSTRY-SPECIFIC MARKET RADIUS

Define:

\[
R_i = f(\text{industry},\text{urban density},\text{mobility},\text{average ticket})
\]

Start with a configurable prior.

Example:

```python
RADIUS_PRIORS = {
    "coffee": [0.25, 0.5, 1.0, 2.0],
    "restaurant": [0.5, 1.0, 2.0, 3.0],
    "grocery": [1.0, 3.0, 5.0, 10.0],
    "auto_repair": [3.0, 5.0, 10.0, 20.0],
    "hair_salon": [1.0, 2.0, 5.0],
    "professional_service": [3.0, 5.0, 10.0]
}
```

Do not hard-code these as truth.

They are priors.

The model should learn the effective radius from observed behavior.

---

# 7. TRAVEL-TIME MARKET

Straight-line distance is insufficient.

Two businesses 1 mile apart may be effectively farther apart if:

- separated by highways
- separated by rivers
- separated by mountains
- separated by poor road networks
- separated by traffic
- separated by geographic barriers

Therefore calculate:

\[
d_{ij}^{travel}
\]

rather than relying exclusively on Euclidean distance.

Create:

\[
W_{ij} = e^{-\lambda d_{ij}}
\]

where:

- \(W_{ij}\) = competitive influence
- \(d_{ij}\) = travel distance/time
- \(\lambda\) = decay parameter

---

# 8. COMPETITOR INFLUENCE SCORE

Do NOT count every competitor equally.

For competitor \(j\):

\[
C_{ij}
=
S_j
\cdot
D_{ij}
\cdot
M_{ij}
\cdot
R_{ij}
\cdot
W_{ij}
\]

where:

### \(S_j\)
Size / estimated market relevance

### \(D_{ij}\)
Distance decay

### \(M_{ij}\)
Product similarity

### \(R_{ij}\)
Price-band similarity

### \(W_{ij}\)
Estimated customer overlap

Then:

\[
Competition_i = \sum_j C_{ij}
\]

---

# 9. COMPETITOR SIMILARITY

Create an embedding for:

```text
business category
products
services
price tier
reviews
business description
```

Calculate:

\[
M_{ij} = cosine(E_i,E_j)
\]

Then combine:

\[
CompetitivePressure_i
=
\sum_j
M_{ij}
D_{ij}
PriceSimilarity_{ij}
QualitySimilarity_{ij}
```

This is much better than:

> "There are 17 restaurants nearby."

Instead:

> "There are 4 highly substitutable restaurants within an effective 1.2-mile market."

---

# 10. COMPETITIVE PRICE INDEX

Calculate:

\[
CPI_i =
\frac{
\sum_j W_{ij}P_j
}{
P_i
}
\]

Interpretation:

```text
CPI > 1
business is cheaper than weighted competitors

CPI < 1
business is more expensive than weighted competitors

CPI ≈ 1
business is near competitive parity
```

Also calculate:

\[
RelativePrice_i =
\frac{P_i - P_{competitor,weighted}}
{P_{competitor,weighted}}
\]

---

# 11. CUSTOMER GEOGRAPHY

If customer addresses, delivery zones, loyalty data, or anonymized transaction geography are available, estimate:

\[
Pr(customer\ visits\ business \mid location)
\]

Construct a demand surface.

Example:

```text
0-0.5 mile      28%
0.5-1 mile      31%
1-2 miles       24%
2-5 miles       13%
5+ miles          4%
```

Use this to create a **customer catchment area**.

The catchment area should eventually replace the arbitrary radius.

---

# 12. DEMOGRAPHIC FEATURES

For the actual catchment area, aggregate:

- median income
- income variance
- household size
- population density
- age
- education
- employment
- unemployment
- home ownership
- renter share
- population growth
- business density

Use weighted averages:

\[
X_{market}
=
\sum_k w_k X_k
\]

where \(w_k\) represents estimated customer probability.

---

# 13. MARKET DEFINITION LEARNING

The market should NOT remain fixed.

Estimate:

\[
MarketRadius^* =
argmax_R
PredictiveAccuracy(R)
\]

Try multiple market definitions and evaluate their ability to explain:

- customer overlap
- competitor effects
- price response
- sales changes

A good local-market definition should reduce irrelevant geographic spillovers.

---

# 14. DEMAND MODEL

The fundamental demand model is:

\[
\ln Q_{it}
=
\alpha_i
+
\gamma_t
+
\beta \ln P_{it}
+
f(X_{it})
+
\epsilon_{it}
\]

where:

- \(Q\) = quantity demanded
- \(P\) = price
- \(\alpha_i\) = business/product fixed effect
- \(\gamma_t\) = time fixed effect
- \(X\) = controls

The basic elasticity is:

\[
\epsilon = \beta
\]

because:

\[
\frac{\partial \ln Q}{\partial \ln P}=\beta
\]

---

# 15. WHY SIMPLE REGRESSION IS NOT ENOUGH

Price is endogenous.

Example:

```text
Business expects huge demand
        ↓
Business raises price
        ↓
Demand remains high
```

A naive regression might incorrectly conclude:

> "Higher prices cause high demand."

This is why the system needs causal identification.

---

# 16. DOUBLE MACHINE LEARNING

Implement Double Machine Learning.

Define:

\[
Y = \ln Q
\]

\[
T = \ln P
\]

\[
X = confounding\ variables
\]

Estimate:

\[
m(X)=E[Y|X]
\]

and:

\[
g(X)=E[T|X]
\]

Then:

\[
\tilde Y = Y-\hat m(X)
\]

\[
\tilde T = T-\hat g(X)
\]

Finally:

\[
\hat{\epsilon}
=
\frac{
\sum \tilde T\tilde Y
}{
\sum \tilde T^2
}
\]

Use cross-fitting.

Example:

```text
Fold 1:
train nuisance models on folds 2-5
predict fold 1

Fold 2:
train on folds 1,3,4,5
predict fold 2

...
```

---

# 17. NUISANCE MODEL FEATURES

Include:

```text
lagged demand
rolling demand
seasonality
day of week
month
holiday
weather
promotion
inventory
stockout
competitor count
competitor prices
local income
local population
local unemployment
traffic
events
news shocks
business fixed effects
product fixed effects
time fixed effects
```

Use models such as:

```text
LightGBM
XGBoost
CatBoost
Random Forest
HistGradientBoosting
```

Benchmark them.

Do not assume one is universally best.

---

# 18. HETEROGENEOUS ELASTICITY

The user does not need:

```text
Elasticity = -1.4
```

They need:

```text
Your estimated elasticity:
-1.72

95% CI:
[-2.01, -1.39]

Local competition:
High

Customer income sensitivity:
High

Recent demand shock:
Moderate negative

Confidence:
81%
```

Define:

\[
\epsilon_i =
f(
business,
product,
location,
competition,
customer,
time,
shock
)
\]

---

# 19. CAUSAL FOREST / HETEROGENEOUS TREATMENT EFFECT MODEL

Use causal forests or another heterogeneous-treatment-effect method.

Model:

\[
\ln Q_{it}
=
\tau(Z_{it})\ln P_{it}
+
f(X_{it})
+
u_{it}
\]

where:

\[
\tau(Z_{it})
\]

is the elasticity for that particular market condition.

Possible heterogeneity variables:

```text
income
competition
market density
price tier
product category
customer concentration
business size
local demand volatility
geographic isolation
```

---

# 20. TIME-VARYING ELASTICITY

Elasticity is not constant.

Estimate:

\[
\epsilon_{i,t}
\]

rather than:

\[
\epsilon_i
\]

Use:

```text
30-day
60-day
90-day
180-day
365-day
```

rolling estimates.

Then combine:

\[
\epsilon_t =
w_1\epsilon_{30}
+
w_2\epsilon_{90}
+
w_3\epsilon_{180}
+
w_4\epsilon_{365}
\]

with weights learned using validation.

---

# 21. ELASTICITY REGIME DETECTION

Create regimes:

```text
NORMAL
HIGH_DEMAND
LOW_DEMAND
PROMOTION
COMPETITIVE_SHOCK
ECONOMIC_SHOCK
LOCAL_EVENT
SUPPLY_SHOCK
SEASONAL_PEAK
SEASONAL_LOW
```

The system should estimate:

\[
P(Regime=r|X_t)
\]

Then:

\[
E[\epsilon_t]
=
\sum_r
P(r|X_t)
\epsilon_r
\]

This prevents a temporary abnormal period from permanently contaminating the baseline elasticity.

---

# 22. NEWS / SHOCK ENGINE

This is where Llama becomes extremely valuable.

Continuously ingest:

- local news
- economic news
- business openings
- business closures
- layoffs
- strikes
- construction
- road closures
- natural disasters
- major events
- festivals
- school calendars
- sports events
- regulatory changes
- tax changes
- crime events
- major employer changes
- transportation changes

---

# 23. Llama NEWS EXTRACTION

For every article:

```json
{
  "event_id": "...",
  "event_type": "...",
  "geography": "...",
  "latitude": null,
  "longitude": null,
  "affected_radius_km": null,
  "start_date": "...",
  "end_date": "...",
  "severity": 0.0,
  "demand_direction": -1,
  "demand_magnitude": 0.0,
  "supply_direction": 0,
  "supply_magnitude": 0.0,
  "industries_affected": [],
  "population_affected": 0.0,
  "confidence": 0.0
}
```

Llama must output structured JSON.

Never allow free-form Llama output directly into the numerical model.

---

# 24. SHOCK SPATIAL DECAY

For an event at location \(e\):

\[
Shock_{ie}
=
Severity_e
\times
IndustryMatch_{ie}
\times
SpatialDecay_{ie}
\times
TemporalDecay_{ie}
\times
Confidence_e
\]

Spatial decay:

\[
SpatialDecay_{ie}
=
e^{-\lambda d_{ie}}
\]

Temporal decay:

\[
TemporalDecay_{ie}
=
e^{-\delta \Delta t}
\]

Total shock:

\[
Shock_i
=
\sum_e Shock_{ie}
\]

---

# 25. SHOCK DIRECTION

Represent shocks continuously:

```text
-1 = strongly negative
 0 = neutral
+1 = strongly positive
```

Then:

\[
NetShock =
\sum_e
ShockMagnitude_e
\times
ShockDirection_e
\]

Do not simply classify events as positive/negative.

---

# 26. NEWS NOVELTY

The same event should not repeatedly affect the model.

Calculate:

\[
Novelty_t =
1 -
Similarity(CurrentEvent,HistoricalEvents)
\]

High novelty:

```text
new major event
```

Low novelty:

```text
duplicate article
```

Deduplicate news using embeddings + event IDs + temporal clustering.

---

# 27. SHOCK HALF-LIFE

Every shock should have a learned half-life.

\[
Shock_t =
Shock_0
\left(\frac12\right)^{t/H}
\]

where:

- \(H\) = half-life

Different shocks have different \(H\).

Example:

```text
road closure:
H ≈ days/weeks

concert:
H ≈ hours/days

layoffs:
H ≈ months

tax change:
H ≈ months/years
```

Do not hard-code these permanently.

Learn them from historical outcomes whenever possible.

---

# 28. LOCAL EVENT INDEX

Construct:

\[
LEI_{it}
=
\sum_e
S_e
I_e
D_e
T_e
C_e
\]

where:

- \(S_e\) = severity
- \(I_e\) = industry relevance
- \(D_e\) = geographic decay
- \(T_e\) = temporal decay
- \(C_e\) = confidence

This becomes one of the model's external features.

---

# 29. COMPETITOR ENTRY / EXIT

Competitor changes should be treated as quasi-experimental events.

For every competitor:

```text
opened_at
closed_at
price_change
rating_change
category_change
```

Create:

\[
\Delta Competition_t
=
Competition_t-Competition_{t-1}
\]

This can be used in difference-in-differences/event-study models.

---

# 30. DIFFERENCE-IN-DIFFERENCES

When a meaningful competitor enters:

```text
treated market:
businesses near new competitor

control market:
similar businesses without competitor entry
```

Estimate:

\[
Y_{it}
=
\alpha_i
+
\gamma_t
+
\beta(Treated_i\times Post_t)
+
\epsilon_{it}
\]

This estimates the effect of competition on demand/pricing.

---

# 31. CROSS-PRICE ELASTICITY

Do not only model:

\[
\frac{\partial Q_i}{\partial P_i}
\]

Also model:

\[
\epsilon_{ij}
=
\frac{\partial Q_i}{\partial P_j}
\frac{P_j}{Q_i}
\]

where \(j\) is a competitor/substitute.

This answers:

> "If my competitor raises prices 5%, how much additional demand should I expect?"

Estimate this only when sufficient variation exists.

---

# 32. SUBSTITUTE NETWORK

Construct a graph:

```text
Product A
   ↕
Product B
   ↕
Competitor Product C
   ↕
Competitor Product D
```

Edge weight:

\[
W_{ij}
=
Similarity_{ij}
\times
GeographicOverlap_{ij}
\times
PriceBandSimilarity_{ij}
\times
ObservedSubstitution_{ij}
\]

This produces a local substitute network.

---

# 33. DEMAND MODEL OUTPUT

For every product/business/time:

```json
{
  "elasticity": -1.63,
  "elasticity_lower": -1.92,
  "elasticity_upper": -1.37,
  "confidence": 0.84,
  "competitive_pressure": 0.77,
  "local_demand_index": 1.13,
  "shock_index": -0.28,
  "price_position": 0.94
}
```

---

# 34. PRICE SIMULATION ENGINE

Once elasticity is estimated:

\[
\frac{Q_1}{Q_0}
=
\left(
\frac{P_1}{P_0}
\right)^\epsilon
\]

Therefore:

\[
Q_1
=
Q_0
\left(
\frac{P_1}{P_0}
\right)^\epsilon
\]

For every candidate price, calculate:

```text
expected quantity
expected revenue
expected gross profit
```

---

# 35. REVENUE

\[
Revenue=P\times Q
\]

Therefore:

\[
Revenue_1
=
P_1Q_1
\]

Compare:

\[
\Delta Revenue
=
Revenue_1-Revenue_0
\]

and:

\[
\%\Delta Revenue
=
\frac{Revenue_1-Revenue_0}{Revenue_0}
\]

---

# 36. PROFIT MODEL

Do not optimize revenue alone.

Use:

\[
Profit
=
(P-C)Q
-F
\]

where:

- \(P\) = price
- \(C\) = variable unit cost
- \(Q\) = quantity
- \(F\) = relevant fixed costs

If variable cost changes by price regime:

\[
C=C(P,X)
\]

allow the model to account for that.

---

# 37. OPTIMAL PRICE

Find:

\[
P^*
=
argmax_P
Profit(P)
\]

Run a numerical search.

Example:

```python
candidate_prices = np.arange(
    current_price * 0.80,
    current_price * 1.20,
    current_price * 0.005
)
```

For each price:

```text
predict quantity
predict revenue
predict cost
predict profit
```

Choose:

```text
argmax(expected_profit)
```

---

# 38. RISK-AWARE OPTIMIZATION

Do not optimize expected profit only.

Define:

\[
Objective(P)
=
E[Profit(P)]
-
\lambda Risk(P)
\]

where:

\[
Risk(P)=Var(Profit(P))
\]

or use a downside metric such as CVaR.

For an SMB, a recommendation with slightly lower expected profit but substantially lower downside risk may be superior.

---

# 39. PRICE ELASTICITY INTERVAL

Never return a point estimate alone.

Generate:

```text
elasticity:
-1.63

80% interval:
[-1.79, -1.47]

95% interval:
[-1.94, -1.32]
```

Use:

- bootstrap
- conformal prediction
- Bayesian posterior
- causal inference standard errors

depending on the model.

---

# 40. ENSEMBLE ARCHITECTURE

Do not rely on one estimator.

Create:

### Model A
Two-way fixed effects

### Model B
Double Machine Learning

### Model C
Causal Forest

### Model D
Gradient boosting demand model

### Model E
Bayesian hierarchical elasticity model

Then combine:

\[
\hat\epsilon
=
\sum_k w_k\hat\epsilon_k
\]

where:

\[
\sum_k w_k=1
\]

Learn weights from out-of-sample historical performance.

---

# 41. BAYESIAN HIERARCHICAL SHRINKAGE

This is extremely important for SMBs.

A small business may only have:

```text
50 transactions
```

You cannot estimate highly reliable elasticity from that alone.

Use hierarchical priors:

\[
\epsilon_i
\sim
N(\mu_{industry,market},
\sigma^2)
\]

Then update with the business's own data.

Conceptually:

```text
Industry evidence
      ↓
Regional evidence
      ↓
Local-market evidence
      ↓
Business evidence
      ↓
Product evidence
```

The less data a business has, the more the model should shrink toward reliable higher-level estimates.

The more data it obtains, the more individual behavior dominates.

---

# 42. PROPRIETARY "LOCAL ELASTICITY PRIOR"

Create a proprietary prior:

\[
PriorElasticity
=
f(
industry,
priceTier,
income,
competition,
density,
businessType,
location
)
\]

Then:

\[
PosteriorElasticity
=
Update(
PriorElasticity,
ObservedBusinessData
)
\]

This becomes a major proprietary component.

---

# 43. DATA QUALITY SCORE

Every elasticity estimate must have a quality score.

Create:

\[
DQS =
f(
sampleSize,
priceVariation,
dataRecency,
missingness,
stockouts,
promotionContamination,
identificationStrength
)
\]

Example:

```text
DQS = 0.91 → excellent
DQS = 0.72 → good
DQS = 0.51 → weak
DQS = 0.27 → insufficient
```

---

# 44. IDENTIFICATION SCORE

Separately calculate:

\[
IDS =
f(
priceVariation,
exogeneity,
instrumentStrength,
controlQuality,
temporalVariation,
competitiveVariation
)
\]

This prevents the product from pretending an observational estimate is causal when it isn't.

---

# 45. CONFIDENCE SCORE

Final confidence should combine:

\[
Confidence
=
f(
DQS,
IDS,
modelAgreement,
sampleSize,
intervalWidth,
outOfSamplePerformance
)
\]

Example:

```text
Confidence: 87%

Why:
- strong historical price variation
- 18 months of data
- high model agreement
- low missingness
- stable competitive environment
```

---

# 46. MODEL VALIDATION

Use time-based validation.

NEVER randomly shuffle the entire dataset.

Instead:

```text
TRAIN:
Jan 2023 → Dec 2024

VALIDATION:
Jan 2025 → Jun 2025

TEST:
Jul 2025 → Dec 2025
```

The model must predict the future using the past.

---

# 47. WALK-FORWARD BACKTESTING

Run:

```text
Train → predict next period

Expand training window

Train → predict next period

Repeat
```

Example:

```text
Train Jan-Jun
Predict Jul

Train Jan-Jul
Predict Aug

Train Jan-Aug
Predict Sep
```

This approximates actual deployment.

---

# 48. EVALUATION METRICS

Measure:

### Demand prediction

```text
MAE
RMSE
MAPE
SMAPE
```

### Elasticity prediction

```text
MAE
RMSE
coverage of confidence intervals
sign accuracy
```

### Business value

Most importantly:

```text
revenue uplift
profit uplift
pricing regret
```

Define pricing regret:

\[
Regret
=
Profit(P^*)-Profit(P_{recommended})
\]

The model should minimize expected regret.

---

# 49. COUNTERFACTUAL VALIDATION

When historical price changes exist:

```text
Actual price:
$10 → $11

Model prediction:
quantity decreases 14.2%

Actual:
quantity decreases 13.1%
```

Measure:

\[
Error=
Predicted\Delta Q-Actual\Delta Q
\]

Track this continuously.

---

# 50. MODEL DRIFT

Monitor:

```text
elasticity drift
competitor drift
customer drift
geographic drift
economic drift
news distribution drift
```

Use:

\[
PSI
\]

and distribution-distance metrics.

Retrain when:

```text
prediction error > threshold
OR
feature drift > threshold
OR
elasticity regime changes
```

---

# 51. GEOGRAPHIC MODEL DRIFT

Markets change.

A competitor opens.

A major employer closes.

A highway changes.

A neighborhood grows.

Therefore recalculate:

```text
market definition
competition
customer catchment
demographics
event exposure
```

periodically.

Do not treat location as static.

---

# 52. FEATURE STORE

Create a centralized feature store.

Suggested namespaces:

```text
geo.*
competition.*
customer.*
product.*
pricing.*
demand.*
weather.*
economic.*
news.*
events.*
business.*
model.*
```

Example:

```text
competition.weighted_density
competition.weighted_price
competition.similarity
competition.entry_rate

geo.population_density
geo.median_income
geo.income_variance

news.local_shock_index
news.consumer_confidence_shock

demand.units_7d
demand.units_30d
demand.volatility_90d
```

---

# 53. LLM FEATURE EXTRACTION SERVICE

Create a separate service:

```text
/news-processing
```

Pipeline:

```text
news
 ↓
deduplication
 ↓
geocoding
 ↓
Llama classification
 ↓
structured JSON
 ↓
validation
 ↓
embedding
 ↓
shock score
 ↓
feature store
```

Never allow an unvalidated LLM response into production modeling.

---

# 54. LLM VALIDATION

Every Llama output must pass:

```text
JSON schema validation
range validation
geographic validation
date validation
confidence threshold
duplicate detection
```

Example:

```python
assert -1 <= demand_direction <= 1
assert 0 <= severity <= 1
assert 0 <= confidence <= 1
```

If validation fails:

```text
retry
```

If repeated failure:

```text
discard
```

---

# 55. LLM PROMPTING PRINCIPLE

Do not ask:

> "Will this news affect this business?"

Instead ask Llama to extract objective information.

Bad:

```text
Predict whether this restaurant will lose customers.
```

Good:

```text
Extract:
1. event
2. location
3. affected industries
4. likely direction of demand
5. estimated severity
6. duration
7. geographic radius
8. evidence
9. confidence
```

The numerical model should make the final inference.

---

# 56. NEWS EMBEDDINGS

Store embeddings for:

```text
event
business
industry
product
competitor
```

Then calculate semantic relevance:

\[
R_{event,business}
=
cosine(E_{event},E_{business})
\]

Combine with explicit industry rules.

Never use embedding similarity alone.

---

# 57. LOCAL ECONOMIC SHOCK ENGINE

Create:

\[
EconomicShock_{i,t}
=
f(
unemployment,
employment,
income,
businessOpenings,
businessClosures,
consumerSpending,
housing,
localEvents
)
\]

Use local data whenever available.

---

# 58. WEATHER ENGINE

For weather-sensitive businesses:

```text
temperature
precipitation
humidity
snow
heat alerts
air quality
```

Create nonlinear features:

\[
WeatherEffect=f(Temperature,Precipitation,...)
\]

Do NOT assume a linear relationship.

---

# 59. CALENDAR ENGINE

Features:

```text
day_of_week
week_of_year
month
holiday
school_break
payday
tax_deadline
sports_event
local_festival
```

Create interactions:

\[
Price \times Weekend
\]

\[
Price \times Holiday
\]

\[
Price \times LocalEvent
\]

---

# 60. CUSTOMER PRICE SENSITIVITY SEGMENTS

If transaction-level customer information exists, estimate segments.

Example:

```text
Segment A:
high income / low price sensitivity

Segment B:
middle income / moderate price sensitivity

Segment C:
high promotion sensitivity
```

Then:

\[
Q=\sum_s Q_s
\]

and:

\[
\epsilon_{total}
=
\sum_s Share_s\epsilon_s
\]

subject to the appropriate weighting.

---

# 61. PRICE ANCHORING

Measure competitor-relative pricing.

Features:

```text
price percentile
relative competitor price
price gap
discount depth
promotion frequency
```

Example:

\[
PriceGap=
\frac{P_i-P_{market}}{P_{market}}
\]

This can explain why a 5% price increase may have different effects depending on whether the business is already 15% cheaper or 15% more expensive than competitors.

---

# 62. PROMOTION CONTAMINATION

Promotions create endogeneity.

Create explicit variables:

```text
promotion
coupon
bundle
BOGO
advertising
display
seasonal_sale
```

Do not treat promotional price as ordinary price without controls.

---

# 63. STOCKOUT CONTAMINATION

Observed quantity is not necessarily demand.

If inventory = 0:

```text
units sold = 0
```

does NOT mean:

```text
demand = 0
```

Create:

```text
stockout_flag
inventory_available
days_out_of_stock
```

Either censor those observations or model latent demand.

---

# 64. CENSORING

If:

\[
Q_{observed}<Q_{demand}
\]

because of inventory constraints, use a censored-demand model.

Never blindly train the elasticity model on stockout observations.

---

# 65. PRICE DISCRETIZATION

Prices often change in increments.

Example:

```text
$9.99
$10.49
$10.99
```

Do not overstate precision.

Use actual observed prices and maintain price-change thresholds.

---

# 66. PRICE RECOMMENDATION CONSTRAINTS

The optimizer must respect:

```text
minimum margin
maximum price change
legal constraints
business-defined constraints
psychological price points
competitor constraints
inventory constraints
```

Example:

```python
price_min = max(
    cost * 1.10,
    business_minimum_price
)

price_max = min(
    current_price * 1.15,
    business_maximum_price
)
```

---

# 67. RECOMMENDATION GENERATION

The final system should return:

```json
{
  "current_price": 10.00,
  "recommended_price": 10.79,
  "elasticity": -1.42,
  "expected_quantity_change": -10.5,
  "expected_revenue_change": 3.1,
  "expected_profit_change": 7.8,
  "confidence": 0.86,
  "risk": 0.21
}
```

---

# 68. EXPLANATION ENGINE

Llama can turn structured model output into human-readable insights.

Example input:

```json
{
  "recommended_price": 10.79,
  "elasticity": -1.42,
  "competition": "high",
  "relative_price": -0.08,
  "local_demand": 1.11,
  "news_shock": -0.14
}
```

Output:

> Your model estimates that demand is moderately price-sensitive.
>
> The recommended increase is limited because nearby competitors are priced relatively close to you and local demand has recently weakened.
>
> The model estimates that moving from $10.00 to $10.79 could increase revenue while maintaining a manageable expected decline in volume.

Llama explains the model. It does not invent the numbers.

---

# 69. PROPRIETARY CORE: LOCAL ADAPTIVE ELASTICITY SCORE

Create a proprietary final representation:

\[
LAE_{i,t}
=
F(
BaseElasticity,
Competition,
RelativePrice,
CustomerSensitivity,
DemandRegime,
EconomicShock,
NewsShock,
Seasonality,
DataQuality
)
\]

Internally retain the full mathematical model.

The product-facing representation can be:

```text
LAE = -1.42
confidence = 86%
market sensitivity = high
```

---

# 70. MODEL STACK

Recommended production stack:

```text
Python
│
├── pandas / polars
├── numpy
├── scipy
├── scikit-learn
├── lightgbm / xgboost
├── econml
├── statsmodels
├── shap
├── geopandas
├── shapely
├── h3
├── networkx
│
├── PostgreSQL/PostGIS
│
├── vector database
│
└── HuggingFace Transformers
       │
       └── Llama
```

---

# 71. GEOSPATIAL STORAGE

Use PostGIS.

Store:

```text
business locations
competitor locations
customer regions
events
market polygons
roads
census geographies
```

Use H3 indexing where useful.

Example:

```text
business → H3 cell
competitors → neighboring H3 cells
events → H3 cells
demographics → H3/census mapping
```

This makes repeated spatial calculations much faster.

---

# 72. DATABASE SCHEMA

Minimum tables:

```text
businesses
products
transactions
prices
inventory
competitors
competitor_prices
locations
market_areas
demographics
weather
events
news_articles
news_events
economic_indicators
features
elasticity_estimates
price_predictions
model_versions
backtests
```

---

# 73. MODEL VERSIONING

Every prediction must record:

```text
model_version
training_window
feature_version
llm_version
embedding_version
data_timestamp
prediction_timestamp
```

This allows exact reproduction.

---

# 74. DATA LEAKAGE PREVENTION

This is critical.

When predicting:

```text
2026-10-01
```

the model must only receive information that was available by:

```text
2026-10-01
```

Do not accidentally use future news, future competitor prices, future sales, or future demographic revisions during training.

---

# 75. TEMPORAL FEATURE RULE

Every feature should have:

```text
effective_at
observed_at
available_at
```

The model should only use:

\[
available\_at \le prediction\_time
\]

---

# 76. CAUSAL IDENTIFICATION HIERARCHY

Rank estimates:

### Tier 1
Randomized pricing experiments

### Tier 2
Strong natural experiments

### Tier 3
Difference-in-differences

### Tier 4
Instrumental variables

### Tier 5
Double Machine Learning

### Tier 6
Fixed effects

### Tier 7
Pure predictive correlations

The system must report which tier supports the elasticity estimate.

---

# 77. EXPERIMENTATION ENGINE

Eventually allow SMBs to run controlled pricing experiments.

Example:

```text
10% of customers:
$10.99

90%:
$9.99
```

Subject to legal/ethical/business constraints.

Use experiments to directly identify elasticity.

This becomes the highest-quality proprietary data source.

---

# 78. ACTIVE LEARNING

The system should identify where it is uncertain.

Example:

```text
Elasticity:
-1.2 to -2.1
```

Rather than merely reporting uncertainty, recommend:

> "A controlled 5% price experiment would substantially improve the estimate."

This turns uncertainty into an experimentation opportunity.

---

# 79. PROPRIETARY DATA FLYWHEEL

The product should improve as customers use it.

```text
SMB connects sales
        ↓
Model learns demand
        ↓
Model identifies uncertainty
        ↓
Platform recommends experiment
        ↓
SMB changes price
        ↓
Observed result
        ↓
Causal elasticity evidence
        ↓
Improved local prior
        ↓
Better future predictions
```

This creates proprietary training data.

---

# 80. MULTI-TENANT LEARNING

Do NOT mix raw customer data between businesses.

Instead learn population-level parameters:

\[
\theta_{industry,region,market}
\]

while keeping business-specific data isolated.

Use hierarchical modeling/federated-style aggregation where appropriate.

---

# 81. PRIVACY

Never expose:

```text
Customer identities
Individual addresses
PII
Private transaction records
```

to the LLM unless explicitly necessary and legally appropriate.

Prefer aggregated geographic signals.

---

# 82. SECURITY

Separate:

```text
raw customer data
model features
LLM inputs
LLM outputs
model predictions
```

Do not send the entire transaction database to Llama.

Only send the minimal structured context needed for an inference.

---

# 83. API ARCHITECTURE

Suggested services:

```text
POST /business/onboard

POST /data/transactions

POST /market/build

POST /competitors/refresh

POST /news/process

POST /features/build

POST /elasticity/estimate

POST /price/simulate

POST /price/optimize

GET /insights/{business_id}

GET /model/{business_id}/{product_id}
```

---

# 84. PIPELINE

The complete production pipeline:

```text
1. Ingest business data
2. Validate transaction data
3. Normalize prices
4. Detect promotions
5. Detect stockouts
6. Geocode business
7. Build H3 location
8. Build candidate market
9. Identify competitors
10. Calculate competitor similarity
11. Calculate competitive pressure
12. Build customer catchment
13. Add demographics
14. Add weather
15. Add economic indicators
16. Ingest local news
17. Deduplicate news
18. Geocode events
19. Send event text through Llama
20. Validate Llama JSON
21. Calculate shock scores
22. Construct feature store
23. Build demand panel
24. Remove leakage
25. Estimate baseline FE elasticity
26. Estimate DML elasticity
27. Estimate heterogeneous elasticity
28. Estimate Bayesian posterior
29. Ensemble models
30. Generate uncertainty interval
31. Calculate data-quality score
32. Calculate identification score
33. Simulate price changes
34. Optimize price
35. Calculate expected profit
36. Calculate downside risk
37. Generate recommendation
38. Send structured result to Llama
39. Generate human explanation
40. Store model version
41. Monitor prediction
42. Compare prediction with actual result
43. Retrain/update model
```

---

# 85. INITIAL MVP

Do NOT attempt everything simultaneously.

Build V1 with:

```text
transaction history
price
quantity
business location
competitor locations
competitor prices
basic demographics
seasonality
weather
```

Model:

```text
Two-way FE
+
Double Machine Learning
+
Gradient Boosting
```

Then add:

```text
V2:
causal forest
Bayesian hierarchy
```

Then:

```text
V3:
news/shock engine
```

Then:

```text
V4:
experimentation engine
```

Then:

```text
V5:
cross-price elasticity
dynamic market learning
```

---

# 86. RECOMMENDED INITIAL MODEL

For V1:

```python
baseline_model = TwoWayFixedEffects()

dml_model = DoubleMachineLearning(
    outcome="log_quantity",
    treatment="log_price",
    controls=features
)

heterogeneity_model = CausalForest(
    treatment="log_price",
    outcome="log_quantity",
    heterogeneity_features=market_features
)
```

Then:

```python
elasticity = ensemble(
    baseline_model,
    dml_model,
    heterogeneity_model
)
```

Do not simply average them.

Determine ensemble weights using historical out-of-sample performance.

---

# 87. PROPRIETARY COMPONENTS TO PRIORITIZE

The most defensible intellectual property is NOT:

```text
Llama
LightGBM
XGBoost
PostGIS
```

Those are tools.

Your defensible system should be the combination of:

### 1. Dynamic Micro-Market Definition
Learning the actual economic market surrounding a business.

### 2. Competitive Influence Graph
Quantifying which competitors actually matter.

### 3. Local Shock Engine
Converting geographically relevant events into demand shocks.

### 4. Hierarchical Elasticity Prior
Borrowing statistical strength from similar businesses without blindly pooling them.

### 5. Local Adaptive Elasticity
Making elasticity conditional on the business's current environment.

### 6. Elasticity Regime Detection
Recognizing that elasticity changes during unusual periods.

### 7. Counterfactual Price Simulator
Turning elasticity into actual business outcomes.

### 8. Active Learning / Experimentation
Using uncertainty to determine which price experiments generate the most information.

### 9. Proprietary Outcome Dataset
Every validated prediction → actual outcome becomes training/evaluation data.

---

# 88. MOST IMPORTANT MODEL EQUATION

The conceptual final model should be:

\[
\boxed{
\epsilon_{i,t}
=
F(
B_i,
P_i,
G_i,
C_{i,t},
D_i,
S_t,
N_{i,t},
W_t,
E_t,
R_t
)
}
\]

where:

```text
B = business characteristics
P = product characteristics
G = geography
C = competition
D = customer/demographics
S = seasonality
N = news/local shocks
W = weather
E = economic conditions
R = current demand regime
```

The actual implementation should use a combination of causal estimation, hierarchical modeling, and machine learning rather than trying to force everything into one giant neural network.

---

# 89. MOST IMPORTANT BUSINESS EQUATION

Ultimately the customer doesn't care about elasticity by itself.

They care about:

\[
\boxed{
P^*
=
argmax_P
E[(P-C)Q(P|X)]
-
\lambda Risk(P)
}
\]

The product's job is therefore:

```text
Estimate demand
       ↓
Estimate elasticity
       ↓
Understand local environment
       ↓
Simulate prices
       ↓
Estimate revenue
       ↓
Estimate profit
       ↓
Estimate downside
       ↓
Recommend price
```

---

# 90. CLAUDE CODE IMPLEMENTATION ORDER

Implement in this exact order.

## Phase 1 — Data Foundation

Create:

```text
src/
  ingestion/
  normalization/
  geospatial/
  features/
  models/
  simulation/
  optimization/
  llm/
  evaluation/
  api/
```

Build:

```text
transaction ingestion
price normalization
product normalization
business normalization
```

---

## Phase 2 — Geospatial Engine

Implement:

```text
H3 indexing
distance calculation
travel distance
competitor discovery
market radius
competitor similarity
competitive pressure
```

Write unit tests for every component.

---

## Phase 3 — Baseline Elasticity

Implement:

```text
log-log regression
two-way fixed effects
```

Produce:

```text
elasticity
standard error
confidence interval
```

---

## Phase 4 — DML

Implement:

```text
nuisance model
cross-fitting
residualization
elasticity estimation
confidence interval
```

Compare DML against FE.

---

## Phase 5 — Heterogeneous Elasticity

Implement:

```text
causal forest
market heterogeneity
business heterogeneity
```

Return observation-specific elasticity.

---

## Phase 6 — Hierarchical Model

Implement:

```text
industry prior
regional prior
market prior
business posterior
```

This should improve small-data SMB estimates.

---

## Phase 7 — News Engine

Implement:

```text
news ingestion
deduplication
geocoding
Llama extraction
JSON validation
embedding
shock scoring
temporal decay
spatial decay
```

---

## Phase 8 — Price Simulator

Implement:

```text
price grid
quantity forecast
revenue forecast
profit forecast
risk forecast
```

---

## Phase 9 — Optimization

Implement:

```text
expected-profit optimization
risk-adjusted optimization
business constraints
```

---

## Phase 10 — Backtesting

Create automated tests:

```text
historical training period
future test period
predicted price
actual outcome
regret
```

Run continuously.

---

# 91. NON-NEGOTIABLE ENGINEERING RULES

Claude Code MUST follow these rules.

### Rule 1
Never use Llama as the primary numerical elasticity estimator.

### Rule 2
Never use future information during training.

### Rule 3
Never treat correlation as causal elasticity.

### Rule 4
Never report an elasticity without uncertainty.

### Rule 5
Never ignore stockouts.

### Rule 6
Never treat all competitors equally.

### Rule 7
Never define local markets solely by ZIP code.

### Rule 8
Never allow unvalidated LLM output into the numerical model.

### Rule 9
Never optimize revenue without considering profit.

### Rule 10
Never deploy a model without walk-forward backtesting.

### Rule 11
Every prediction must be reproducible from a model/data version.

### Rule 12
Every recommendation must contain a confidence score.

---

# 92. FINAL PRODUCT OUTPUT

The SMB dashboard should eventually display something like:

```text
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PRICE INTELLIGENCE

Product:
12oz Signature Latte

Current price:
$5.99

Recommended price:
$6.49

Estimated elasticity:
-1.34

Confidence:
88%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EXPECTED EFFECT

Quantity:
-8.1%

Revenue:
+5.4%

Gross profit:
+9.7%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHY?

Local competition:
HIGH

Your current price:
6% below weighted competitors

Local demand:
+8% above baseline

Customer price sensitivity:
MODERATE

Recent local shock:
-2.1%

Seasonality:
POSITIVE

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RISK

95% elasticity range:
[-1.61, -1.12]

Expected downside:
-2.3% revenue

Model identification:
STRONG

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RECOMMENDATION

Increase price gradually to $6.49.

The model estimates that the additional margin should
outweigh the expected volume loss under current market
conditions.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

# 93. THE ACTUAL MOAT

The product should ultimately become a continuously updating economic model of each SMB's local market.

The moat is:

```text
Business data
       +
Geographic structure
       +
Competitive structure
       +
Historical price experiments
       +
Local events
       +
News
       +
Economic conditions
       +
Observed outcomes
       ↓
Local Adaptive Elasticity Engine
       ↓
Better predictions
       ↓
More customer experiments
       ↓
More proprietary outcome data
       ↓
Better model
```

The goal is not to build:

> "an AI that predicts elasticity."

The goal is to build:

> **a continuously learning causal model of how a specific business's local market responds to price.**

That distinction should drive the entire architecture.
