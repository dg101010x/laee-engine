# 09 — Data sources, privacy, security and compliance

Status: draft for LAEE v2. Replaces v1 §70 (model stack, data-source parts only), §71 (geospatial storage, source parts only), §72 (database schema, source tables only), §80 (multi-tenant learning), §81 (privacy) and §82 (security). Written 2026-10-06.

This chapter says which external data the engine may use and under what terms, what may cross a tenant boundary, what may be sent to a language model, how the engine service protects its keys and data, and which legal and fairness limits apply to the price advice it gives. It does not define estimation methods. Methods live in the chapters cross-referenced below.

| Topic | Owning chapter |
|---|---|
| Interfaces, glossary, versioning contract | `00-overview-and-contracts.md` |
| Bitemporal facts (`effective_from`, `observed_at`, `available_at`), ingestion of tenant sales | `01-data-foundation.md` |
| Trade areas, competitor discovery and entity resolution, travel time, demographic aggregation to H3 | `02-geospatial-market.md` |
| Weather and calendar controls inside the demand model, identification | `03-identification-and-core-estimators.md` |
| Drift and uncertainty | `04-heterogeneity-time-ensemble-uncertainty.md` |
| Cross-tenant pooling mathematics | `05-priors-and-hierarchical-pooling.md` |
| News ingestion, LLM extraction, validation, geocoding, embeddings | `06-shock-engine.md` |
| Feasible price set, fairness rule, experiments | `07-pricing-decisions.md` |
| Monitoring, lineage, backtests, operations | `08-evaluation-scoring-operations.md` |

---

## 9.0 Conventions

### 9.0.1 Evidence tags

Every factual claim about an external source (licence, price, limit, coverage) carries one tag and, in the catalogue, a "last checked" date. The research behind this chapter was done on 2026-10-06 under an egress proxy that blocked most primary documentation pages and a shared search budget that ran out. Many terms could therefore be seen only in search-result summaries.

| Tag | Meaning | Last-checked value used in tables |
|---|---|---|
| **[V]** | Read on the primary page or its source repository during research | 2026-10-06 |
| **[S]** | Seen only in a search-result summary; the primary page was not opened | 2026-10-06 (snippet) |
| **[L]** | Recalled from literature or documentation; not re-verified | never |
| **[C]** | Computed by the research agents | 2026-10-06 |
| **[D]** | Design decision of this spec, not a finding | n/a |

Rule [D]: any item tagged [S] or [L] in the licence, cost or rate-limit columns MUST be re-verified on the provider's own page, and the date recorded in the source registry (§9.2), before production use. This is a launch gate (§9.10).

### 9.0.2 Normative words

MUST, MUST NOT, SHOULD and MAY have their usual meanings. Defaults marked [D] are starting values to be tuned, not evidence.

### 9.0.3 Not legal advice

Statements about law in this chapter are engineering guardrails written by non-lawyers. The research notes did not verify any African statute's text. Counsel review per country is a launch gate (§9.10).

---

## 9.1 Principles

1. **Tenant data never leaves its tenant**, except as the summary records defined in §9.5.3. Raw rows, features and per-product series stay inside the tenant partition. (Fixes v1 §80, which did not say what crosses the boundary.)
2. **No tenant or customer data goes to an LLM provider by default.** The news extractor sees only public article text. Any other LLM use (explanations, name classification) has an explicit field allowlist (§9.6).
3. **Every external source is registered** with its licence class, terms reference, last verification date, cadence and caching rule before any code reads it (§9.2). Unregistered sources fail the build.
4. **Licence-incompatible data is refused by code**, not by policy text. Non-commercial (NC) layers and transient-only content cannot enter persistent production tables (§9.3.8).
5. **Missing is missing.** When a source is down, stale, or fails a coverage audit, the feature is NaN with a missing indicator, never zero. Zero means "calm" or "no competitors" and biases estimates (from the shock-engine notes; applies to every source).
6. **Point-in-time honesty.** Every external fact carries `available_at` = when it landed in our store, per the bitemporal contract in `01-data-foundation.md`. Projections (for example WorldPop 2026–2030) are never stored as observations.
7. **Least privilege everywhere.** No engine job holds a credential that can reach more data than that job needs (§9.7).
8. **Price advice is decision support for one posted price**, constrained by fairness and law, never individualised (§9.8).

---

## 9.2 Source registry contract

Every external source is a row in `source_registry`. The registry generalises the news source registry proposed in the shock-engine notes to all external data.

```text
source_registry(
  source_id            text primary key,          -- e.g. 'osm_geofabrik', 'overture_places', 'open_meteo_hist'
  family               text,                      -- places | routing | geocoding | population | built | wealth | weather | news | events | model | macro | calendar
  provider             text,
  access_method        text,                      -- bulk_download | api | self_hosted | package
  endpoint             text,
  licence_id           text,                      -- SPDX-style where one exists, else provider name + version
  licence_class        text,                      -- see table below
  attribution_text     text,                      -- exact string shown in UI/exports
  terms_ref            text,                      -- URL of the terms page actually read
  terms_verified_at    date,                      -- null = never verified -> blocks production
  verified_by          text,
  cost_model           text,                      -- free | per_call | subscription | licence_fee
  monthly_budget_usd   numeric,                   -- hard cap enforced by the client
  rate_limit           jsonb,                     -- {per_second, per_day, concurrency, notes}
  refresh_cadence      text,                      -- monthly | daily | per_release | on_demand
  cache_rule           text,                      -- persistent | ttl:<days> | id_only | transient
  retention_days       int,
  countries            text[],
  active               boolean,
  stats                jsonb                      -- last_success_at, failures_7d, staleness_days, coverage metrics
)
```

Licence classes [D]:

| `licence_class` | Meaning | Allowed persistent use | Examples (verify each, §9.3) |
|---|---|---|---|
| `open_permissive` | Commercial use allowed; attribution at most | Persistent tables, derived features, exports | Overture Places, Foursquare OS Places, OSRM/Valhalla code |
| `open_attribution` | Commercial use allowed with attribution | Persistent; attribution shown wherever derived values are displayed or exported | WorldPop, HRSL, Open Buildings (CC BY option), VIIRS, GeoNames |
| `open_sharealike` | Attribution plus share-alike on public derivative databases | Persistent, but in a separate schema; never merged into a database offered to third parties without legal review | OpenStreetMap (ODbL) |
| `noncommercial` | NC licence | Research and evaluation only; refused by the production licence guard unless a commercial licence is recorded | Relative Wealth Index (CC BY-NC), NLLB-200 weights (CC-BY-NC) |
| `proprietary_transient` | Paid or ToS-limited; content may be cached only briefly or not at all | IDs and short-TTL caches only | Google Places |
| `commercial_licence_required` | Must sign a licence before any use by a company | Only after the licence is recorded | ACLED |
| `paid_api` | Paid subscription, terms per contract | Per contract | Event Registry, Open-Meteo commercial plan |
| `unknown` | Not yet verified | Nothing in production | Default for every new source |

Each feature row in the feature store references the `source_id` and source version (snapshot date or release id) of every input, extending the lineage rules in `08-evaluation-scoring-operations.md`.

---

## 9.3 Catalogue of external sources

Each family has two tables: (a) what the source provides, African coverage and quality, resolution, licence and cost; (b) access, rate limits, refresh cadence, caching rule and failure handling. Methods for using the data are in the owning chapter.

### 9.3.1 Places and map data (competitors, roads, place names)

Owning chapter: `02-geospatial-market.md` (competitor discovery, entity resolution, existence probability).

**(a) Content, coverage, licence, cost**

| Source | Provides | African coverage and quality | Resolution | Licence | Cost | Last checked |
|---|---|---|---|---|---|---|
| OpenStreetMap (OSM) | POIs, road and footpath network, buildings, admin areas, place names | Uneven. Building completeness about 30% in sub-Saharan Africa; low in Egyptian and Ethiopian cities, higher in Tanzania, Uganda and West Africa [S] (Herfort et al. 2023). Maiduguri was an outlier for poor amenity coverage among 25 cities [S]. POI completeness for Nairobi, Lagos, Accra and Johannesburg by category is unmeasured. Footpaths in informal settlements often missing | Vector features | ODbL 1.0: attribution on produced works; share-alike on publicly used derivative databases [L] | Free | never ([L]) |
| Overture Maps Places | More than 85M places from AllThePlaces, BrightQuery, DAC, Foursquare, Krick, Microsoft, PinMeTo, RenderSEO and Meta; `confidence` in [0,1]; `taxonomy` | Reported share with confidence > 0.9 is 0.22% in Kenya and 0.11% in Nigeria, most below 0.5 [S, attribution uncertain]. Geocode-fallback clusters (stacked points) reported [S] | Points | CDLA-Permissive-2.0 and Apache-2.0, some CC0. Places contains no OSM data, so no ODbL share-alike [S] | Free (AWS Open Data) | 2026-10-06 (snippet) |
| Foursquare OS Places | 100M+ POIs, 22 core attributes, `date_closed` | Not quantified for Africa. Closed POIs often lack `date_closed` [S] | Points | Apache 2.0, commercial use allowed [S] | Free | 2026-10-06 (snippet) |
| Google Places API (New) | Verification of specific candidate POIs; `business_status`; place_id | Not measured; assumed broader than open sources in formal sectors, still weak on informal vendors [D] | Points | Google Maps Platform terms: place_id may be stored indefinitely; lat/lng cached at most 30 days; no caching exception for names, ratings or photos [S] | Per 1,000 calls (aggregator figures): Nearby Search Pro $32.00 (0–100K), $25.60 (100–500K), $19.20 (500K–1M), 5,000 free per month; Nearby Search Enterprise 1,000 free per month; Place Details Essentials $5, Pro $17, Enterprise $20, Enterprise + Atmosphere $25; IDs-only SKUs free [S]. Deep-volume tiers such as $2.40 at 5M+ look implausible and are unverified | 2026-10-06 (snippet) |
| Business registries: Kenya BRS (eCitizen), Nigeria CAC, Ghana ORC, South Africa CIPC | Legal entities; no outlet coordinates | Formal firms only | Entity | Not verified | Not verified | never ([L]) |

**(b) Access, limits, cadence, caching, failure**

| Source | Access method | Rate limits and usage policy | Refresh | Caching and storage rule [D unless tagged] | Failure handling [D] |
|---|---|---|---|---|---|
| OSM | Production: Geofabrik country extracts (`.osm.pbf`) for Kenya, Nigeria, Ghana, South Africa [L], filtered with `osmium tags-filter`. Prototyping only: public Overpass API | Public Overpass asks users to stay under about 10,000 queries/day and 1 GB/day [L, not verified]. Production MUST NOT depend on public Overpass | Monthly extract; record `osm_snapshot` | Full local copy allowed. OSM-derived tables live in a separate schema (`osm_*`) and are never merged into a database delivered to tenants or partners without legal review of ODbL share-alike [I from notes]. Attribution "© OpenStreetMap contributors" shown on any map or export using OSM-derived data | Download failure: keep the previous snapshot; set `staleness_days`; alert at 45 days. A diff that removes more than 10% of a city's POIs in one month is quarantined for review |
| Overture Places | GeoParquet from the AWS Open Data registry, filtered by bounding box (DuckDB) | Bulk download; no API limit relevant | Per Overture release (monthly in practice [L]); pin release and schema version, because category fields changed across releases [S] | Persistent. Keep the attribution file | Drop records with confidence < 0.3; flag points stacked with ≥ 5 others or on admin centroids (`02-geospatial-market.md`). Release schema change: pipeline fails closed until the crosswalk is updated |
| Foursquare OS Places | Parquet (also mirrored on Hugging Face) [S] | Bulk download | Monthly [S] | Persistent. Keep attribution file | Drop records with `date_closed`. Treat as not independent of Overture, which ingests Foursquare [S] |
| Google Places | API key restricted to Places API and to the engine's egress IPs | Per-call billing; per-call result caps for Nearby Search not verified | On demand only | Store place_id only, indefinitely. Lat/lng TTL ≤ 30 days. Names, ratings and photos MUST NOT be stored [S]. Never build the persistent competitor table from Google content. Persistent table holds place_id plus our own derived `p_exists` | Monthly budget cap enforced in client (MVP default $200/month per city [D]). On cap or error: fall back to open sources and mark `google_checked = false`. Illustrative Nairobi sweep: about 8,600 Nearby Search Pro calls, about $115 per sweep at list price [C, from notes]; sweeps are not part of the MVP |
| Business registries | Manual or per-request lookups | Unverified | On demand | Store registry id only | Not in MVP |

### 9.3.2 Routing and geocoding

Owning chapters: `02-geospatial-market.md` (travel time) and `06-shock-engine.md` (toponym resolution).

**(a) Content, coverage, licence, cost**

| Source | Provides | African coverage and quality | Licence | Cost | Last checked |
|---|---|---|---|---|---|
| OSRM (self-hosted) | Fast car and foot OD tables (CH or MLD) | Only as good as OSM roads; free-flow car times understate peak congestion in Nairobi and Lagos (size unmeasured); snaps over 150 m common in informal settlements [D from notes] | BSD-2-Clause (was AGPL) [S] | Infrastructure only. Rule of thumb: preprocessing needs about 10× the extract size in RAM, serving about 3–5× [S] | 2026-10-06 (snippet) |
| Valhalla (self-hosted) | Matrices, isochrones, multimodal, map matching, time-dependent costing | Same OSM dependence | MIT [S] | Infrastructure only; tiled graph loaded on demand [S] | 2026-10-06 (snippet) |
| openrouteservice public API | Matrix and isochrones | Same OSM dependence | Service terms (HeiGIT) | Free tier; limits conflict between sources: about 2,500 requests/day and 40,000/month, 40 concurrent, about 500 isochrones/day; matrix ≤ 3,500 elements per request [S] | 2026-10-06 (snippet) |
| GraphHopper | Routing | OSM | Open-source core Apache-2.0; Matrix API believed commercial-only [L, not verified] | Not verified | never |
| pgRouting on Supabase | Ad-hoc shortest paths in Postgres | OSM | Supported extension on Supabase [S] | Included | 2026-10-06 (snippet) |
| Google Routes / Distance Matrix | Traffic-aware times for congestion calibration | Presumed good in large cities; not measured | Google Maps Platform terms | 2026 per-element prices not verified | never |
| GTFS feeds (e.g. Digital Matatus, Nairobi) | Scheduled minibus routes for transit times | Nairobi mapped in the Digital Matatus project [L]; currency of that feed and existence of GTFS for Lagos, Accra and Johannesburg minibus systems not verified | Per feed, not verified | Free where available | never |
| Nominatim | Geocoding from OSM | OSM quality | Data ODbL; public service under the OSMF usage policy [S] | Public service free but heavily restricted (below); self-hosting costs infrastructure only | 2026-10-06 (snippet) |
| GeoNames | Gazetteer (local dumps or web service) | Completeness for markets, bus parks and informal settlements unknown | Creative Commons attribution licence [S] | Free | 2026-10-06 (snippet) |

**(b) Access, limits, cadence, caching, failure**

| Source | Rate limits and usage policy | Refresh | Caching rule [D] | Failure handling [D] |
|---|---|---|---|---|
| OSRM | `--max-table-size` defaults to 100 locations per `/table` request and can be raised when self-hosting [S] | Rebuild monthly on the new Geofabrik extract; record profile hash | `travel_time` table keyed by origin H3 cell, POI, mode, time band and `osm_snapshot` (schema in `02-geospatial-market.md`) | Snap > 150 m or network/straight-line ratio > 2.5: fall back to Euclidean × city circuity factor and flag the pair. Golden file of 500 OD pairs per city; report drift on each rebuild |
| Valhalla | Self-hosted, no external limit | Monthly with OSRM | As OSRM | As OSRM |
| openrouteservice | As above; conflicting quotas | n/a | n/a | Prototyping only. MUST NOT be a production dependency |
| Google Routes | Not verified | Quarterly calibration sample of about 1,000 stratified OD pairs per city [D from notes] | Store only the derived congestion multiplier κ(city, hour band, distance band), not raw Google responses, subject to terms review | If unavailable: keep last κ; if none, use free-flow times and widen the travel-time uncertainty band |
| Nominatim public | Absolute maximum 1 request per second; identifying User-Agent or Referer required; bulk geocoding discouraged; periodic scripted requests treated as bulk; per a snippet, long-running scripts limited to 4 requests per minute; results must be cached [S] | n/a | Cache every result | Production MUST use a self-hosted Nominatim or cached results only. Public instance allowed in development behind a client that enforces 1 req/s, a custom User-Agent and a cache |
| GeoNames | Web service: 10,000 credits/day and 1,000/hour per application, mandatory `username` [S] | Local dump refreshed quarterly [D] | Use local dumps, not the web service, in production | Missing toponym: route to the human-in-the-loop alias queue (`06-shock-engine.md`) |

### 9.3.3 Population, built environment, wealth and surveys

Owning chapter: `02-geospatial-market.md` (raster-to-H3 aggregation, catchment weighting).

**(a) Content, coverage, licence, cost**

| Source | Provides | African coverage and quality | Resolution and time | Licence | Cost | Last checked |
|---|---|---|---|---|---|---|
| WorldPop Global2 R2025A | Annual population counts (constrained top-down disaggregation); age/sex rasters at 1 km on a mirror | 242 countries. Released September 2025 as an alpha that "may change". No accuracy assessment in informal settlements (e.g. Kibera, Makoko) found | 100 m (3 arc-sec), 2015–2030; 2026–2030 are projections | CC BY 4.0 [S] | Free | 2026-10-06 (snippet) |
| GHSL GHS-POP R2023A (JRC) | Population disaggregated by built-up volume from GPWv4.11 census inputs | Global | 100 m and 1 km, Mollweide (EPSG:54009), five-year epochs 1975–2030 | Free with source acknowledgement under EC reuse rules [S]; generally means CC BY 4.0 [L] | Free | 2026-10-06 (snippet) |
| HRSL (Meta/CIESIN) | Population from census plus building detection | Africa tiles on AWS and HDX; no longer updated as of 2024 | About 30 m | CC BY 4.0 [S] | Free | 2026-10-06 (snippet) |
| GRID3 Nigeria v2.0/v2.1 | Bottom-up population with age-sex groups and uncertainty, from about 1,100 surveyed locations | Nigeria only; matters because Nigeria has had no census since 2006 [S] | About 100 m | Not verified | Free (HDX) | 2026-10-06 (snippet); licence never |
| Kontur population | H3-native population (400 m hexagons), e.g. South Africa | Per country on HDX | H3 ~400 m | Not verified | Free | 2026-10-06 (snippet); licence never |
| Relative Wealth Index (Meta / Chi et al. 2022) | Relative wealth micro-estimates from satellite, mobile-network, topographic and Facebook connectivity data | 135 LMICs in the paper, 93 on HDX; values relative within a country and not comparable across countries [L] | 2.4 km | **CC BY-NC 4.0** [S] — non-commercial | Free for non-commercial use; commercial licence terms not researched | 2026-10-06 (snippet) |
| DHS (The DHS Program) | Survey microdata, wealth quintiles, cluster GPS | Most African countries; cluster coordinates displaced 0–2 km urban, 0–5 km rural, up to 10 km for 1% of rural clusters, within admin-2 [S] | Admin-1 calibration only | Registration and project approval required [L]; commercial-use terms not verified | Free on approval [L] | 2026-10-06 (snippet) for displacement; terms never |
| National censuses: Kenya 2019, South Africa 2022, Ghana 2021; Nigeria 2006 | Small-area counts and attributes where published | Small-area availability and licences not verified; median income, renter share, unemployment mostly unavailable below city level in Nigeria and patchy elsewhere [S] | Ward / small area where available | Not verified | Not verified | never |
| Google Open Buildings v3 | 1.8B building detections (inference May 2023) | Africa, South/Southeast Asia, Latin America and Caribbean; 58M km² | Footprints | User chooses **CC BY 4.0 or ODbL** [S] | Free | 2026-10-06 (snippet) |
| Open Buildings 2.5D Temporal | Annual building presence, fractional counts and heights from Sentinel-2 | Same footprint as v3 | 4 m effective, annual 2016–2023 | As Open Buildings [S] | Free | 2026-10-06 (snippet) |
| VIIRS night lights (EOG / Payne Institute) | Annual and monthly radiance; activity proxy where statistics are weak | Global; saturates and blooms in CBDs [D from notes] | 500 m (15 arc-sec); monthly lag about 1–2 months [D from notes] | Many products CC BY 4.0 with attribution [S]; check per product | Free (registration) [L] | 2026-10-06 (snippet) |

**(b) Access, limits, cadence, caching, failure**

| Source | Access | Refresh | Caching rule [D] | Failure handling [D] |
|---|---|---|---|---|
| WorldPop | Bulk raster download from the WorldPop hub | Per release; pin release id (R2025A) | Persistent; aggregated to `h3_demog` with `src_versions` | If a release changes values by more than 30% in a ward, keep both versions and flag. Never use 2026–2030 values as observed in backtests |
| GHSL | Bulk download (JRC) | Per release | Persistent | Keep alongside WorldPop; flag ward-level disagreement > 30% [D from notes] |
| HRSL | AWS / HDX tiles | Frozen (no updates) | Persistent | Use only for res-10 work; label vintage |
| GRID3 | HDX | Per release | Persistent once licence verified | Licence unknown → `licence_class = unknown` → not in production |
| Kontur | HDX | Per release | Persistent once licence verified | As GRID3 |
| RWI | HDX | Frozen | `noncommercial`: research schema only; the production licence guard refuses it unless a commercial licence is recorded | Use in-house proxy (building height × footprint × radiance per capita, calibrated to DHS wealth quintiles at admin-1) [D from notes] |
| DHS | Registered download | Per survey | Admin-1 aggregates only; microdata MUST NOT be redistributed [L]; stored in a restricted schema | Not in MVP until terms verified |
| Census | National statistics offices | Per release | Per licence | Not in MVP until availability and licence verified |
| Open Buildings | Bulk download (choose the CC BY 4.0 option to avoid share-alike) [D] | Frozen v3; 2.5D ends 2023 | Persistent | Label vintage; use only for growth up to 2023 |
| VIIRS | EOG download | Annual and monthly | Persistent | Monthly product missing: carry forward with `staleness_days`; never same-month feature |

### 9.3.4 Weather

Owning chapter: `03-identification-and-core-estimators.md` (weather as nuisance control and gating). The leakage rule comes from the demand-data notes: retrospective elasticity estimation may use realised weather; forecasting and price simulation at decision time T for date T+h must use the lead-h forecast and be trained on archived lead-h forecasts.

**(a) Content, coverage, licence, cost**

| Source | Provides | African coverage and quality | Resolution and latency | Licence | Cost | Last checked |
|---|---|---|---|---|---|---|
| ERA5 (ECMWF / Copernicus) | Hourly reanalysis: temperature, precipitation, humidity, wind | Global. Reduced ERA-Interim's tropical-African wet bias [S]; underestimates rain where stations are sparse (Congo Basin) [S]; in Ghana had the lowest correlation with gauges and overestimated rain [S]; matched stations in one extreme convective event in eastern South Africa [S] | 0.25° (about 28 km), from 1940; about 5-day delay via Open-Meteo [S] | Copernicus licence, commercial use with attribution [L, not verified] | Free via Copernicus Climate Data Store [L] | 2026-10-06 (snippet) for coverage; licence never |
| ERA5-Land | Land reanalysis, better for 2 m temperature | Global land | 0.1°, from 1950 [S] | As ERA5 [L] | Free [L] | 2026-10-06 (snippet) |
| Open-Meteo Historical Weather API | ERA5, ERA5-Land and ECMWF IFS (9 km, from 2017) through one API, updated daily with about 5-day delay [S] | Inherits ERA5 strengths and weaknesses | As above | Commercial licensing terms **not verified** in research. Recalled: free API for non-commercial use, paid plan for commercial use, data under CC BY 4.0 [L] | Not verified | 2026-10-06 (snippet); terms never |
| Open-Meteo Previous Runs API | Archived forecasts at fixed lead offsets 1–7 days (e.g. `temperature_2m_previous_day1`); most models archived from January 2024, GFS 2 m temperature from March 2021; intended for ML without look-ahead [S] | Global | Model grid | As Open-Meteo | As Open-Meteo | 2026-10-06 (snippet) |
| Open-Meteo Historical Forecast API | Archived high-resolution forecasts [S, title only] | Global | Start date not verified | As Open-Meteo | As Open-Meteo | 2026-10-06 (title) |
| CHIRPS (UCSB Climate Hazards Center) | Rainfall blending satellite and stations | Quasi-global. v2: 50°S–50°N from 1981; v3 superseded v2 in June 2025 and covers 60°N–60°S [S]. Outperforms ERA5 in Ethiopia, especially at altitude (daily bias ratios 0.75–1.2) [S]; underestimated an extreme convective event in eastern South Africa [S] | 0.05°, daily/pentadal/monthly; preliminary about 2 days, final about 3 weeks [S] | Not verified | Free [L] | 2026-10-06 (snippet); licence never |

**(b) Access, limits, cadence, caching, failure**

| Source | Access | Rate limits | Refresh | Caching rule [D] | Failure handling [D] |
|---|---|---|---|---|---|
| ERA5 / ERA5-Land | Via Open-Meteo (MVP) or Copernicus CDS (later) | CDS queue limits not researched | Daily pull; `available_at` = valid time + about 5 days [D from notes] | Persistent per H3 res-7 cell (grid is coarser than res 7) and local day in the shop's timezone | Late data: feature is NaN for those days; never same-day. Coastal and urban microclimate mismatch at 0.25° documented as a known limitation |
| Open-Meteo (all APIs) | HTTPS API | Not verified (recalled: about 10,000 calls/day on the free non-commercial tier [L]) | Daily | Persistent (subject to verified licence). Batch requests per city grid, not per shop | Retry with exponential backoff; after 24 h of failure, alert; forecasts missing → simulation uses climatology and widens intervals |
| Open-Meteo Previous Runs | HTTPS API | As above | Daily | Persistent; store `lead_days` and model id | Before 2024 most leads unavailable → forecasting models trained only on periods with archived leads |
| CHIRPS | Bulk download from CHC | n/a | Preliminary daily, final monthly | Persist both preliminary and final as separate revisions (bitemporal); `available_at` = prelim + about 2 days, final + about 3 weeks | Final revision replaces preliminary only for dates after its `available_at` |

Source choice per region is pluggable [D from notes]: temperature from ERA5-Land; rainfall history from CHIRPS with ERA5 fallback, because African validations conflict.

### 9.3.5 News, events, official notices and calendars

Owning chapter: `06-shock-engine.md` (ingestion, deduplication, extraction, coverage audit).

**(a) Content, coverage, licence, cost**

| Source | Provides | African coverage and quality | Latency and window | Licence and terms | Cost | Last checked |
|---|---|---|---|---|---|---|
| Tier A: official and structured notices — load-shedding schedules, fuel-price announcements (e.g. Kenya EPRA monthly review, Nigeria NBS price watch), utility planned-interruption notices (e.g. Kenya Power, Ghana ECG), EskomSePush API, gazetted holidays, sports fixtures, election dates | Dates and areas of scheduled shocks | Most precise and timely where they exist | Per publisher | **Endpoints, cadence and licence for every named feed not verified** | Mostly free; EskomSePush API terms not verified | never |
| Tier B: local outlets' RSS or sitemaps; BBC, VOA, DW and RFI language services (Swahili, Hausa, Yoruba, French) | Unscheduled local events | Sparse and uneven; some South African and Nigerian titles paywalled; much local news is on radio and social media, which were not assessed | Poll every 10–30 min [D] | Per publisher; store `terms_ref`; respect robots.txt [D]. Outlet inventory not compiled or verified | Free to poll | never |
| Google News RSS | Discovery of articles by locale | Unknown | Unknown | No official API; links are Google redirects; terms, rate limits and item caps not verified [L] | Free | never |
| GDELT DOC 2.0 | URL discovery; machine-translated search across 65 languages (98.4% of non-English volume) [S] | Capital-city bias; correlation with hand-coded data 0.20–0.26 spatial, 0.33–0.64 temporal [S]; its event codes and geocodes are not ground truth | Processed within 15 min; rolling 3-month window [S] | Terms not verified. Recalled: free for any use with citation [L] | Free | 2026-10-06 (snippet); terms never |
| Event Registry / NewsAPI.ai | Archive search back to 2014; 150,000+ sources in 60+ languages (claimed) [S] | Not measured | Archive and recent | Commercial subscription | Free tier 2,000 tokens (one snippet says up to 500/day); 5K plan $90/month, 10K about $150/month, 25K about $300/month; overage $0.015/token; article search 1 token recent (30 days), 5 tokens per archive year; event search 5 recent, 20 per archive year [S] | 2026-10-06 (snippet) |
| ACLED | Political violence and protest events with locations | Africa focus; strong for unrest validation | Update cadence weekly [L, not verified] | **Commercial entities need a corporate licence before accessing or using the data** [S]. Whether the licence allows use as model training labels is not verified. API moved from keys to login-based auth (cookies or OAuth); legacy keys ran until 15 September 2025 [S] | Licence fee not researched | 2026-10-06 (snippet) |
| `holidays` Python package (0.106) | Public holidays incl. Kenya, Nigeria, South Africa, Ghana; Islamic holidays labelled "(estimated)" when unconfirmed | Kenya Eid confirmed 2014–2024; Nigeria Eid al-Fitr 1979–2026; South Africa module has no Islamic holidays (not public holidays there) [V, package source] | n/a | Package licence not recorded in notes (recalled MIT [L]) | Free | 2026-10-06 (package source) |
| `hijridate` package | Umm al-Qura Hijri conversion [V] | Local moon-sighting may differ by ±1 day | n/a | Not recorded (verify) | Free | 2026-10-06 |

**(b) Access, limits, cadence, caching, failure** [D unless tagged]

| Source | Access and limits | Refresh | Caching and retention | Failure handling |
|---|---|---|---|---|
| Tier A notices | Feed or page polling; prefer APIs; human-curated where no feed | Per publisher; daily minimum | Persistent structured records (dates, areas) | Route to calendar or structured engine, not the LLM. Missing feed → event family disabled for that city (`news_missing = 1`) |
| Tier B RSS | HTTP polling with conditional GET; honour robots.txt and terms | 10–30 min | Store canonical URL, title, metadata, extracted fields and evidence quotes of at most 300 characters. Full text only transiently for processing unless the licence allows retention [D from notes]. Transient text deleted within 7 days [D] | Source failing for 24 h → `stats.last_success_at` stale → coverage index falls; never fill zeros |
| Google News RSS | Discovery only, pending legal check | — | URLs only | Not in MVP |
| GDELT DOC 2.0 | HTTPS API; parameter names and record caps not verified | 15–60 min | URLs and GDELT metadata only; GDELT event codes not used as truth | API failure → Tier B continues; coverage index reflects the gap |
| Event Registry | API key; token budget | Backfill jobs only | Per contract | Daily token cap in client |
| ACLED | OAuth after corporate licence | Weekly [L] | Per licence; validation datasets only | Not in MVP |
| `holidays` / `hijridate` | Pinned package versions | Each release | Generated tables persisted with package version | Estimated dates carry ±1 day uncertainty; CI test that Nigeria 2026 contains Eid al-Fitr [V, from notes] |

Coverage gate [D from notes]: news features for a (city, event family) are enabled only when monthly recall against reference events is at least 0.5 and median latency is at most 24 h. Otherwise `news_missing = 1` and the exposure is NaN.

### 9.3.6 Language models, embeddings and safety models

Owning chapter: `06-shock-engine.md` (prompts, schema, validation, calibration). This section covers terms, cost and access only.

**(a) Content, coverage, licence, cost**

| Source | Provides | African-language support | Licence | Cost | Last checked |
|---|---|---|---|---|---|
| Hugging Face Inference Providers (router) | OpenAI-compatible chat completions at `https://router.huggingface.co/v1` (chat only); routing policies `:fastest` (default), `:cheapest`, `:preferred`; provider pinning by suffix or path. Chat providers listed: Baseten, Cerebras, Cohere, DeepInfra, Featherless AI, Fireworks, Groq, HF Inference, Novita, Nscale, OVHcloud AI, Public AI, Scaleway, Together, Z.ai. Embedding providers: HF Inference, Scaleway, Together [V] | n/a | HF terms; provider data-retention and training-use terms **not documented on the pages read and not reviewed** [V] | HF charges provider rates with no added fee; monthly credits $0.10 free users, $2.00 PRO ($9/month [S]), $2.00 per Team/Enterprise seat; then pay-as-you-go; with a custom provider key the provider bills directly [V] | 2026-10-06 |
| Llama 3.3 70B Instruct | Event extraction | Supports EN, DE, FR, IT, PT, HI, ES, TH; Meta "strongly discourage[s]" other languages without fine-tuning and controls; knowledge cutoff December 2023; 128k context [V] | Llama 3.3 Community License [V]; acceptable-use policy and attribution terms not verified | Provider prices per M tokens (in/out): DeepInfra Turbo FP8 $0.10/$0.32; Novita $0.14/$0.40; Parasail FP8 $0.22/$0.50; Scaleway about $1.05 blended [S]. About $0.44–$4 per 1,000 articles [C from notes] | 2026-10-06 |
| Llama 3.1 8B Instruct | Relevance triage | Same 8 languages [S] | Llama 3.1 Community License [L] | Provider-dependent | 2026-10-06 (snippet) |
| Llama 4 Scout / Maverick | Fallback for Arabic- and French-heavy markets | Adds Arabic, Indonesian, Tagalog, Vietnamese; knowledge cutoff August 2024 [S] | Llama 4 Community License [L] | Provider-dependent | 2026-10-06 (snippet) |
| LaBSE | Cross-lingual duplicate detection | 109 languages; good for translation pairs, weaker for non-translation similarity [V]; African-language list not verified | Not verified (recalled Apache-2.0 [L]) | Self-host on CPU | 2026-10-06 |
| BGE-M3 | Relevance and clustering embeddings | "More than 100 working languages", 8,192-token inputs, 1,024-d [V] | Not verified (recalled MIT [L]) | Self-host | 2026-10-06 |
| multilingual-e5 (small/base/large) | Relevance and clustering embeddings | XLM-R backbone; per recall includes Swahili, Hausa, Somali, Amharic but not Yoruba, Igbo or Zulu [L] | Not verified (recalled MIT [L]) | Self-host | 2026-10-06 |
| paraphrase-multilingual-* | — | Do **not** list sw, ha, yo, ig, zu or am [V] | — | — | Excluded |
| jina-embeddings-v3 | — | Unverified | Believed CC-BY-NC [L] | — | Excluded until verified |
| Llama Prompt Guard 2 (86M, 22M) | Prompt-injection and jailbreak classifier | 86M evaluated on EN, FR, DE, HI, IT, PT, ES, TH; 22M has no multilingual pretraining; 512-token context; recall 97.5% / 88.7% at 1% FPR [V] | Not recorded (verify) | Self-host; 92.4 ms / 19.3 ms per 512 tokens [V] | 2026-10-06 |
| Microsoft Presidio | PII detection and anonymisation | English-centric recognisers by default [L] | Not verified (recalled MIT [L]) | Self-host | never |
| NLLB-200 | Machine translation | Covers many African languages [L] | **CC-BY-NC 4.0, non-commercial** [V] | — | 2026-10-06 — excluded from production |
| MADLAD-400 | Machine translation candidate | Not verified | Not verified (believed Apache-2.0 [L]) | Self-host | never |
| vLLM (self-hosted fallback) | Inference server with `structured_outputs`; `guided_*` fields removed in v0.12.0 [V] | n/a | Apache-2.0 [L] | GPU infrastructure | 2026-10-06 |

**(b) Access, limits, cadence, caching, failure** [D unless tagged]

| Item | Rule |
|---|---|
| Authentication | Fine-grained HF token with only the "Make calls to Inference Providers" permission [V for the permission name]; one token per environment; startup test asserts it cannot write (§9.7) |
| Rate limits | HF documents no router inference limits; limits are per provider and unpublished [V]. Hub API limits (5-minute windows, PRO 2,500 requests) apply to Hub endpoints such as the provider-mapping API, not inference [S] |
| Provider choice | Never use `:fastest` or `:cheapest` in production, because they can route to a provider whose terms have not been reviewed. Pin to providers that (1) appear `live` in `inferenceProviderMapping`, (2) pass the strict-schema canary in `06-shock-engine.md`, and (3) have data-retention terms reviewed and recorded in `source_registry` |
| Cadence | Provider canary daily; gold-set scoring weekly |
| Caching | Store every LLM output with full lineage and never re-call the LLM for history (reproducibility rule from notes). Embeddings stored float16 with `embedding_model_id`, revision hash and `embedding_version` |
| Budget | Daily spend cap per environment; token-bucket rate limiting; skip LLM on near-duplicates |
| Failure | Retryable statuses 408, 429, 500, 502, 503, 504 with exponential backoff and jitter; non-retryable 400 marks the provider failed for that schema version; after 5 attempts switch provider; if no provider passes, extraction pauses and news features become `news_missing = 1` rather than degrading silently |
| Licence guard | Model ids are checked against `source_registry`; `noncommercial` or `unknown` models cannot be loaded in production |

### 9.3.7 Macro series and tenant-entered data

| Source | Provides | Notes |
|---|---|---|
| National CPI, fuel prices and FX (KNBS, Stats SA, Ghana Statistical Service, Nigeria NBS; central banks) | Deflators, national shocks | Endpoints and licences not verified [L]. Use the vintage available at t, never the latest revised series [D from notes]. Deflating by local CPI is required in high-inflation economies (`05-priors-and-hierarchical-pooling.md`) |
| Competitor prices entered by the tenant or collected by field staff | Relative-price features | Tenant-owned data (§9.5). `available_at` = collection time, never the claimed price date. Never shared with other tenants (§9.8.4) |

### 9.3.8 Cross-cutting rules for all sources [D]

1. **Licence guard.** Every production build checks that each `source_id` in the feature lineage has `licence_class` not in {`noncommercial`, `unknown`} and `terms_verified_at` not null. Violations fail the build (extends the "licence guard" test in the geospatial notes).
2. **Attribution.** A generated `ATTRIBUTION` page lists the `attribution_text` of every source used by any feature visible to a tenant. Exports of maps or tables include the relevant lines.
3. **Snapshots.** Bulk sources are stored by snapshot or release id, never overwritten. Feature rows reference the snapshot.
4. **Staleness.** Each source has a staleness budget (monthly sources 45 days, daily sources 3 days, news 24 h). Exceeding it sets a stale flag that flows into the confidence and data-quality diagnostics in `04-heterogeneity-time-ensemble-uncertainty.md`.
5. **Fallback order** is written per family in the tables above. A fallback never substitutes a lower-licence source (for example RWI for the in-house wealth proxy).
6. **Egress.** The engine reaches only hosts listed in `source_registry.endpoint` (§9.7.5).

---

## 9.4 Recommended stack: MVP and later

The MVP targets a small launch set of cities in Kenya, Nigeria, Ghana and South Africa (the launch set is an open question; §9.11). The guiding choices: open, commercially usable sources first; self-hosted where public services forbid production load; nothing whose licence is unverified.

| Layer | MVP default | Add later (trigger) | Why |
|---|---|---|---|
| Competitor places | OSM (Geofabrik extracts) + Overture Places + Foursquare OS Places, deduplicated with existence probability | Google Places verification of specific candidates (when a tenant disputes a competitor list; budget-capped). Field audits of 20–40 H3 cells per city (before using income-by-competition results) | Open, commercial licences; Google content cannot be stored |
| Travel time | Self-hosted OSRM, car and foot, monthly rebuild. Euclidean × city circuity for prefiltering | Valhalla for isochrones and multimodal; GTFS transit via r5py where feeds exist; Google Routes congestion multiplier (quarterly sample) | OSRM fastest for OD tables; congestion calibration is cheap but terms need review |
| Geocoding | Curated city gazetteer + GeoNames dump + self-hosted Nominatim | Mordecai-style neural ranking if gazetteer recall is poor | Public Nominatim forbids bulk use |
| Population | WorldPop R2025A + GHSL R2023A, both kept | HRSL for res-10 work; GRID3 for Nigeria and Kontur once licences verified | CC BY; two sources expose disagreement |
| Built form and activity | Open Buildings v3 (CC BY 4.0 option) + 2.5D Temporal; VIIRS annual | VIIRS monthly as an activity proxy | Open; useful income and density proxies |
| Wealth | In-house proxy calibrated to DHS admin-1 quintiles (only after DHS terms are verified; until then, no wealth feature) | RWI under a commercial licence, if one is obtained | RWI is CC BY-NC |
| Weather | ERA5-Land temperature and CHIRPS rainfall; Open-Meteo for access (commercial plan if its terms require one); Previous Runs for forecast features | Copernicus CDS direct access; IMERG or TAMSAT if evaluated | Matches the notes' per-region recommendation |
| Calendars | `holidays` + `hijridate`, plus manually curated country paydays and grant days | Official school-term feeds | Authoritative dates beat LLM extraction |
| News and shocks | **Shadow mode only**: Tier A official notices curated by hand; Tier B RSS for a short list of outlets per city; GDELT DOC for discovery; Llama 3.3 70B via HF router pinned to reviewed providers. Features stay `news_missing = 1` until the coverage gate passes | Event Registry backfill; ACLED under licence for validation; self-hosted vLLM; fine-tuned encoder classifier after about 2k labels; MT for unsupported languages with a commercially licensed model | Coverage is sparse and the extractor's accuracy on African local news is unmeasured |
| Embeddings | Self-hosted LaBSE (dedup) and multilingual-e5-base or BGE-M3 (relevance), CPU batch | Re-select on the African-language pair benchmark in `06-shock-engine.md` | Avoids sending text to more providers; licences to verify |
| Safety | Prompt Guard 2 (86M) + Presidio | Fine-tuned Prompt Guard for target languages | Only available off-the-shelf tools |
| Explanations | Deterministic templates (§9.6.3) | LLM explanations, opt-in, allowlisted fields, reviewed provider | Avoids sending tenant numbers to third parties in the MVP |

Indicative MVP running cost of external data, excluding hosting [C from notes, list prices, unverified]: OSM, Overture, Foursquare, WorldPop, GHSL, Open Buildings, VIIRS and CHIRPS are free; Llama extraction about $0.44–$4 per 1,000 articles per city per day depending on provider; Google Places capped by budget; Open-Meteo commercial plan price not verified.

---

## 9.5 Tenant isolation and cross-tenant learning

### 9.5.1 Data classes

| Class | Examples | Where it lives | Who may read it | May cross the tenant boundary? |
|---|---|---|---|---|
| T0: customer personal data | Payer names and phone numbers in mobile-money exports, loyalty ids, delivery addresses | Removed or transformed at ingestion (§9.5.4); never stored raw beyond the ingestion buffer | Ingestion job only | No |
| T1: tenant raw business data | Transactions, prices, costs, inventory, product catalogue, tenant-entered competitor prices | Tenant partition (`tenant_id` on every row, RLS) | That tenant's users; that tenant's engine jobs | No |
| T2: tenant-derived features and models | Feature rows, fitted models, elasticity estimates, recommendations, experiment logs | Tenant partition | As T1 | No |
| T3: per-business summary records | (b, s, n_eff, number of price changes G, tier, estimand id, taxonomy keys, coarse location keys) | `population_summaries` (restricted schema) | Population-fit job only | Yes, as defined in §9.5.3 |
| T4: population aggregates | Hyperparameter snapshots, per-cell sums for leave-one-tenant-out priors | `population_params` | Serving jobs (to compute a tenant's prior); never tenants directly | Only through gated releases (§9.5.3) |
| P: public external data | Everything in §9.3 | Shared schemas per family | All engine jobs | Shared by design (not tenant data) |

### 9.5.2 Enforcement on Supabase [D]

- Every T0–T2 table has `tenant_id uuid not null` and row-level security policies that compare it with the tenant claim of the caller. Policies exist for select, insert, update and delete separately.
- Tenant-facing reads from the Vercel app use the user's JWT and the anon (public) key, so RLS always applies.
- Engine jobs that work for one tenant connect as a dedicated database role without `BYPASSRLS` and set the tenant context for the session (for example a signed claim or a session setting read by the policies). They never use the Supabase service-role key, which bypasses RLS [L].
- Only the population-fit job reads across tenants, and its role has select rights on `population_summaries` alone, not on T1 or T2 tables.
- Shared public tables (P) are read-only for every engine role except the ingestion role of that family.
- Storage buckets for uploaded sales files are per tenant with path prefixes bound to `tenant_id` policies; uploaded files are deleted after successful ingestion plus 30 days [D].

### 9.5.3 What may be aggregated for priors, and how

This replaces v1 §80. The statistics are in `05-priors-and-hierarchical-pooling.md`; the privacy rules are here. All points from the priors notes (§5) unless marked [D].

1. **Only T3 summary records cross the boundary.** A per-tenant seed job, inside the tenant partition, computes for each business-product the canonical stage-1 estimate (b, s) with clustered or HAC SE, plus n_obs, n_eff, G, sd(ln p), tier, period, taxonomy keys (industry, sub-industry, product category, price tier) and coarse location keys (country, region, market id, H3 resolution 7 or coarser). It writes only these rows.
2. **Why summaries suffice.** For Gaussian hierarchies, inverse-variance two-stage pooling is asymptotically as efficient as pooling raw data for common effects, and exactly equal under homoskedastic within-business OLS (Lin & Zeng 2010 [L]; derivation in the notes). There is no federated-learning loop and no need for secure aggregation, because the platform already processes tenant data; secure aggregation becomes relevant only if summaries are computed on tenant-controlled infrastructure or by partners.
3. **Exactly one stage-1 record per business-product per estimand.** Feeding several estimators' outputs as separate rows double-counts data and inflates what is learned about a tenant.
4. **Leave-one-tenant-out priors.** A tenant's prior is computed from population aggregates that exclude every business of that tenant. This avoids double counting and stops a tenant's own data from reappearing as "the market".
5. **Minimum group size for display and export.** A cell-level prior or benchmark is displayed or exported only if the leave-one-out count is at least k_min = 10 businesses from at least 5 distinct tenants; otherwise the parent level is shown. The statistical posterior may still use the small cell internally; only display and export are gated. Reason: in a cell of two businesses, B's leave-one-out prior is A's elasticity, a competitor's trade secret.
6. **Dominance rule.** If one business holds more than 50% of a cell's precision weight, cap its weight at 50% for released aggregates or suppress the cell.
7. **Differencing.** Population snapshots are batched monthly; released values are rounded (0.1) and given as ranges; a cell whose membership changed by a single business is never re-released without noise.
8. **Unit of privacy = tenant.** A tenant with several businesses in a cell contributes one averaged, bounded value.
9. **Optional differential privacy** for external benchmarks and partner exports: equal-weight clipped mean, b clipped to [−6, 1], n public, replace-one adjacency, Laplace noise with SD √2·(U − L)/(n·ε) [C]:

   | n businesses | ε = 0.5 | ε = 1 | ε = 2 |
   |---|---|---|---|
   | 10 | 1.98 | 0.99 | 0.50 |
   | 20 | 0.99 | 0.50 | 0.25 |
   | 50 | 0.40 | 0.20 | 0.10 |
   | 100 | 0.20 | 0.10 | 0.05 |

   Default: DP (ε ≈ 1) only for cells with n ≥ 50; below about 20 businesses the noise (SD ≥ 0.5) swamps the between-business spread (τ ≈ 0.7). Per-cell ε is tracked across releases (sequential composition). The Laplace mechanism is preferred to the Gaussian for scalar releases [C].
10. **Precision-weighted DP releases** use a noisy clipped numerator and noisy denominator with ε split between them, as specified in the notes.
11. **Consent [D].** Contributing T3 records to population priors requires a contractual basis in the tenant agreement (opt-out or opt-in to be decided by counsel; §9.11). A tenant that opts out still receives priors only if counsel confirms that is acceptable; otherwise it receives literature-only priors.
12. **Offboarding [D].** When a tenant leaves or withdraws consent, its T3 rows are deleted and its contribution is removed at the next monthly snapshot; the snapshot id at which removal took effect is logged.

### 9.5.4 Customer personal data at ingestion [D]

The engine needs no customer identity to estimate own-price elasticity. Rules:

- Payer names, phone numbers, emails, national ids and card numbers are dropped at ingestion. This matters because mobile-money statement exports commonly carry payer names and phone numbers.
- If repeat-customer features are later required, the identifier is replaced by a keyed hash (HMAC) with a per-tenant secret held outside the database; the raw value is never stored.
- Customer delivery addresses, when used to estimate trade-area decay (`02-geospatial-market.md`), are geocoded inside the ingestion job, converted to H3 resolution 9 cells, and stored only as counts per cell with cells under 5 customers merged into their resolution-8 parent. Raw addresses are discarded.
- Free-text fields (order notes, product descriptions entered by staff) are scanned with Presidio before storage; detected PERSON, PHONE and EMAIL entities are masked.
- Tests in §9.9 check that no T0 field survives ingestion.

---

## 9.6 PII and LLM input rules

The engine has three possible LLM paths. Each has its own allowlist. Anything not on the allowlist MUST NOT be sent.

### 9.6.1 Path A — news extraction (public text)

From the shock-engine notes (§Q7), adopted as normative:

| May be sent | Must not be sent |
|---|---|
| Article text (title, body, after boilerplate stripping), target city or area, publication date, source id, codebook prompt, few-shot examples, JSON schema from the versioned repository | Any tenant id, business name, business location, product, price, sales figure, customer data, or anything from T0–T4. The request builder can reach only {article, target area, publication date}, enforced by dependency injection and an import-isolation test |

Further rules:
- The schema allows only organisations and offices, never private persons. Presidio scans every output string and drops PERSON entities not on an allowlist of public officials.
- Evidence quotes are capped at 300 characters and scrubbed with Presidio before storage.
- Crime events are excluded, except riots, looting, curfews and large security incidents, because they carry third-party PII and can stigmatise neighbourhoods.
- Raw article text is kept only transiently (§9.3.5).
- The LLM outputs place-name strings, never coordinates; geocoding is deterministic.

### 9.6.2 Path B — classification of public POI names

Competitor names from open sources may be sent to an LLM to map them into the LAEE taxonomy (`02-geospatial-market.md`). Allowed input: POI name, source category tags, city. Not allowed: which tenant the POI competes with, tenant data, or any Google Places content (which may not be stored or reused outside Google's terms [S]).

### 9.6.3 Path C — explanations (v1 §68) [D]

v1 §68 sends structured model output (recommended price, elasticity, competition level, news shock) to Llama for a narrative. That output is confidential tenant business data, not personal data, but sending it to a third-party provider is still a disclosure.

- **MVP:** explanations are generated from deterministic templates filled with the model's numbers. No LLM call.
- **Later, opt-in per tenant:** an LLM may rephrase a template. Allowed input: an allowlisted JSON object of rounded model outputs and categorical labels (e.g. `elasticity_band`, `price_change_pct`, `interval_80`, `competition_level`, `top_drivers[]`). Not allowed: business name, address, coordinates, product names (replace with "Product A"), currency amounts beyond the recommended and current price, raw sales, customer data. Provider must be pinned and reviewed, or self-hosted.
- The LLM explains numbers it is given; every number in its output must match an input number (string check), otherwise the template text is shown instead. "Llama explains the model. It does not invent the numbers" (v1 §68) becomes a test.

### 9.6.4 Prompt injection from external text

Article text and POI names are untrusted input. Indirect prompt injection through retrieved text can take over LLM-integrated applications (Greshake et al. 2023 [L]); OWASP's 2025 LLM Top 10 lists prompt injection first [L]. Controls (from the notes):

| Threat | Control | Residual risk |
|---|---|---|
| Injected instructions in article text | The LLM has no tools and no actions; output is data only. Strict JSON schema with enums, so injection can only change values. Spotlighting: article wrapped in delimiters with a "data, not instructions" rule (Hines et al. 2024 [L]). Every field must be grounded in quoted evidence. Prompt Guard 2 score stored as a quarantine feature (flag, not auto-drop, because articles about AI can trigger it) | Adaptive semantic attacks; Prompt Guard 2 not evaluated in Swahili, Hausa or Yoruba |
| Fake news or poisoning to move prices | Corroboration count after syndication dedup feeds the calibrated event probability. High-impact types (riot, curfew, cash shock) require an official source or at least 2 independent outlets before entering features. Per-source caps on exposure contribution | Coordinated multi-site fakes |
| Schema or grammar as attack surface (structured-output "control-plane" attacks [S, title only]) | Schemas come only from the versioned repository, never from data | — |
| Unbounded consumption | Token-bucket rate limits, daily budget cap, `max_tokens` caps, skip LLM on near-duplicates | — |
| Injection via POI names (Path B) | Names truncated to 120 characters [D]; output restricted to a closed taxonomy enum | Mislabelled category, caught by validation against labels |

Because the pricing recommendation is produced by numerical code from validated features, a successful injection can at most move one event's extracted slots, which then has to pass validation, corroboration and coverage gates before it affects any feature.

### 9.6.5 Logging rules for LLM calls

- One `extraction_log` row per call, with the fields listed in the shock-engine notes: model id, provider, provider model id, mapping status, prompt template id and hash, codebook, few-shot set, schema version, MT model if any, decoding parameters, attempt number, response hash, response JSON (encrypted at rest), token usage, cost estimate, latency, HTTP status, validator version, outcome and reason codes, Prompt Guard score, PII-scrub version.
- Request bodies are not logged in plain text; the article is referenced by `article_id` and `content_sha256`.
- Logs containing evidence quotes inherit the quote retention rule.
- Application logs (Vercel, engine stdout) MUST NOT contain request or response bodies of LLM calls, tenant data rows or secrets. A log scrubber with a deny-list of field names runs on the engine's log handler [D].
- Backtests use stored outputs and report metrics separately before and after each extractor's knowledge cutoff (December 2023 for Llama 3.3) to detect pretraining look-ahead (`08-evaluation-scoring-operations.md`).

### 9.6.6 Cross-border transfer of text to providers

The HF router sends requests to providers that are mostly in the US or EU. The research did not review provider retention terms, and did not verify how POPIA (South Africa), the Nigeria Data Protection Act 2023 and the Kenya Data Protection Act 2019 treat public news text containing personal data [notes: canonical, not re-fetched]. Ghana's Data Protection Act, 2012 (Act 843) [L, not in research notes] is in the same position. Until counsel reviews these:
- only Path A (public text, PII-scrubbed outputs) and Path B (public POI names) are enabled;
- Path C stays on templates;
- a self-hosted vLLM fallback is the route for any market where counsel advises against third-party transfer.

---

## 9.7 Security of the engine service

The deployment is a Vercel-hosted web app, a Supabase project (Postgres with PostGIS, Auth, Storage) and a Python engine service that runs estimation, ingestion and external calls (`00-overview-and-contracts.md`). The research notes cover HF token scope and LLM threats; the rest of this section is design [D] based on standard practice, with Supabase behaviour marked [L] where recalled.

### 9.7.1 Secrets inventory

| Secret | Holder | Scope | Never in | Rotation [D] |
|---|---|---|---|---|
| Supabase anon (public) key | Browser and Vercel | Public by design; protected by RLS | — | On compromise |
| Supabase service-role key (bypasses RLS [L]) | Migration pipeline and a single admin job | Schema migrations; break-glass only | Browser bundle, Vercel client code, `NEXT_PUBLIC_*` variables, engine tenant jobs, logs | Quarterly and after any staff change |
| Engine database roles (per job class: `ingest_<family>`, `tenant_job`, `population_fit`, `read_public`) | Engine service | Grants per §9.5.2 | Browser, Vercel | Quarterly |
| HF fine-grained token | Engine news worker | "Make calls to Inference Providers" only [V] | Any other worker | Quarterly; separate per environment |
| Google Maps Platform key | Engine places worker | Restricted to the Places API (and Routes if enabled), to engine egress IPs, with a quota cap | Browser | Quarterly |
| ACLED OAuth credentials, Event Registry key, Open-Meteo key | Respective workers | Per provider | Others | Per provider policy, at least yearly |
| Per-tenant HMAC secret for customer-id hashing (§9.5.4) | Key management service, read by ingestion only | One tenant | Database | Never rotated silently (rotation breaks joins; requires a re-key job) |

Secrets live in the hosting platform's secret store (Vercel environment variables marked sensitive for server-only code; the engine host's secret manager). They are never committed; CI runs secret scanning.

### 9.7.2 Least privilege [D]

- Engine jobs are split by job class, each with its own database role and its own outbound credentials. A news worker cannot read tenant tables; a tenant estimation job cannot call LLM providers; the population-fit job reads only `population_summaries`.
- Tenant-scoped jobs receive the `tenant_id` from a signed job message and set it as the session tenant context; RLS enforces it. A job that needs data for two tenants is a bug.
- The Vercel server functions call the engine through an authenticated internal API (signed service token with short expiry), never with the service-role key.
- Engine API endpoints (v1 §83) authorise each request by the caller's tenant membership and role (owner, staff, viewer); only owners may approve price changes or enable experiments.

### 9.7.3 Audit logging [D]

An append-only `audit_log` table (insert-only grant; no update or delete for any application role):

```text
audit_log(audit_id, at_utc, actor_type ∈ {user, engine_job, admin}, actor_id, tenant_id (nullable for platform jobs),
          action, object_type, object_id, request_id, ip_hash, outcome, details jsonb)
```

Logged actions: login and role changes; data uploads and deletions; any service-role use (with justification text); population snapshot builds and releases (cell counts, ε spent); exports; price recommendations shown; owner overrides of fairness blocks (§9.8.2); experiment start, allocation and stop; source-registry changes (licence class, terms verification); LLM provider pin changes. Audit rows are retained for at least 3 years [D; legal retention periods to be set by counsel].

### 9.7.4 Data protection at rest and in transit

- TLS for all connections; database encryption at rest as provided by the platform [L].
- Column-level encryption (application-side) for `extraction_log.response_json` and any stored evidence quotes.
- Backups follow the same access rules as the primary; restore drills include an RLS check.
- Supabase project region: chosen per counsel advice on data residency for the launch countries (§9.11).

### 9.7.5 Network and supply chain

- Outbound egress from engine workers is restricted to the endpoints in `source_registry` plus package mirrors at build time.
- Python dependencies pinned by hash; container images rebuilt monthly; model weights pinned by revision hash and verified on load.
- Self-hosted services (OSRM, Valhalla, Nominatim, vLLM, embedding servers) are not exposed to the internet; they listen on a private network reachable only by engine workers.

### 9.7.6 Abuse and cost controls

- Per-tenant rate limits on estimate and optimise endpoints.
- Daily spend caps for every paid source, enforced in the client before the request.
- Alerts when any cap reaches 80%.

---

## 9.8 Legal and fairness constraints on pricing advice

The feasible-set mechanics are in `07-pricing-decisions.md`. This section states the constraints and their basis. None of the African statutes below was verified in research; each row says what it gates until counsel confirms.

### 9.8.1 Default posture [D from notes]

1. **One posted price** per product, store and moment. No individualised or person-targeted prices. LAEE does not use customer-level data to set prices.
2. **Displayed price equals charged price.** Labels change before the switch; the engine's schedule includes the label change.
3. **No deceptive reference prices** ("was/now" claims that were not real prior prices).
4. **Respect regulated maximum prices** and price-control lists; products under such rules are excluded from optimisation and experiments.
5. **Experiments vary prices by time (switchback) or by store or channel**, never by customer; require owner opt-in; cap deviations (for example ±15%); exclude essentials, staple foods and medicines; stop rules on revenue loss or complaints; every randomisation logged (§9.7.3).
6. **Advice, not automation.** The owner approves each change; the engine never pushes prices to a POS without that approval.

### 9.8.2 Perceived fairness

- About 82% of respondents judged a hardware store raising snow-shovel prices from $15 to $20 after a snowstorm unfair or very unfair; raising prices to pass through cost increases was judged acceptable, exploiting demand shifts was not (Kahneman, Knetsch & Thaler 1986 [S]).
- Consumers react more to increases than to equivalent decreases (Kalyanaram & Winer 1995 [L]); removing a discount is judged more acceptable than an equivalent increase [L].
- No evidence on fairness perceptions in African retail was found.

Rules (defaults are uncalibrated engineering values [D from notes]):
- If the shock index shows a demand surge (≥ +0.5 SD) and the cost index rose less than 2%, increases are blocked or require an explicit owner override with a warning; overrides are audit-logged.
- Cost pass-through increases are allowed up to Δc × (1 + current markup).
- Where a discount exists, an increase is framed as removing the discount.
- Shocks that are emergencies (floods, unrest, outages) never produce an increase recommendation for essentials, regardless of override [D].

### 9.8.3 Price discrimination and consumer protection

| Jurisdiction | Instruments named in the notes (all unverified) | Relevance | What it gates in LAEE [D] |
|---|---|---|---|
| South Africa | Competition Act 89 of 1998, s.9 (price discrimination by dominant firms); Consumer Protection Act 68 of 2008, s.23 (displayed-price rules, including whether the lower of two displayed prices binds); POPIA (automated decisions, personal data) | SMB tenants are rarely dominant; displayed-price rules apply to every shop | Displayed = charged; label-change step; experiments only by time or store after counsel review |
| Kenya | Competition Act 2010; any essential-goods price-control law; Kenya Data Protection Act 2019; EPRA regulates fuel prices (notes mention its monthly price review) | Price controls on essentials; automated-decision rules | Essentials list per country excluded from experiments; fuel prices treated as exogenous shocks, never optimised |
| Nigeria | Federal Competition and Consumer Protection Act 2018; Nigeria Data Protection Act 2023 | Consumer protection and data protection | Same defaults; counsel review before experiments |
| Ghana | Not researched in the notes. Data Protection Act, 2012 (Act 843) [L]; competition and consumer-protection regime not researched | Unknown | Experiments disabled in Ghana until counsel review; observational estimates and advice allowed |
| Reference points outside Africa | US Robinson–Patman Act (B2B price discrimination) [L]; EU Directive 2019/2161 (disclosure of prices personalised by automated decision-making) [L]; a 2025 New York algorithmic-pricing disclosure law [L, unverified] | Show the direction of regulation: disclosure of algorithmic and personalised pricing | Supports the no-personalisation default |

### 9.8.4 Algorithmic coordination risk

Pooling competitor price data and recommending prices to competing tenants from one engine can raise hub-and-spoke coordination concerns; this is an inference from the notes, prompted by US litigation over shared pricing software that was not verified. Rules [D]:

- A tenant's recommendation depends only on its own data, public external data, and population priors built from elasticities (T3/T4), never on another tenant's prices, costs, margins or planned changes.
- Tenant-entered competitor prices are T1 data of the tenant that entered them and are never shown to, or used for, any other tenant.
- Population summaries contain elasticities, not prices; price levels are not pooled across tenants.
- The engine never recommends "match competitor X's price" using non-public information, and never coordinates the timing of changes across tenants.
- Experiments are not synchronised across tenants in the same market.

### 9.8.5 Data-protection rules on automated decisions

POPIA, Kenya's DPA 2019 and Nigeria's NDPA 2023 contain rules on automated decisions about individuals [notes, unverified]. Because LAEE makes no decision about an individual customer (one posted price for everyone), these rules should not bite on the price itself; they do apply to any personal data processed at ingestion (§9.5.4). Counsel to confirm.

---

## 9.9 Tests

Acceptance tests that catch a wrong implementation. Owners: platform (P), engine (E), shock engine (S).

| # | Test | Pass condition |
|---|---|---|
| 9.1 | Licence guard (E) | A production build that includes a feature lineage containing a `noncommercial` or `unknown` source, or a source with null `terms_verified_at`, fails |
| 9.2 | Attribution (P) | Every source in any tenant-visible feature lineage appears on the generated attribution page |
| 9.3 | Google caching (E) | No Google Places name, rating or photo appears in any table; Google lat/lng older than 30 days is absent; place_id may persist |
| 9.4 | OSM separation (E) | No export or partner dataset contains rows from `osm_*` schemas without a recorded legal approval |
| 9.5 | Nominatim client (S) | In development: at most 1 request per second, custom User-Agent, cache hit on repeated queries. In production: zero requests to the public host |
| 9.6 | RLS isolation (P) | With tenant A's JWT, every select, insert, update and delete on every T0–T2 table for tenant B rows returns zero rows or fails; automated over the full table list |
| 9.7 | Role scope (P) | The `tenant_job` role cannot read `population_summaries` of other tenants; `population_fit` cannot read T1/T2; no engine role except the admin has `BYPASSRLS` |
| 9.8 | Service-role leak (P) | The built Vercel client bundle and all `NEXT_PUBLIC_*` variables contain no service-role key (pattern scan) |
| 9.9 | Prior isolation (E) | Recomputing tenant A's prior after deleting A's summaries gives a bit-identical served prior (notes test) |
| 9.10 | Suppression (E) | Any cell with leave-one-out n < 10 or fewer than 5 tenants never appears in display or export payloads |
| 9.11 | DP sensitivity (E) | Brute-force replace-one neighbours: maximum change in output ≤ Δ |
| 9.12 | T0 removal (P) | After ingesting a synthetic mobile-money export with names and phone numbers, no stored column matches a phone or name pattern; Presidio finds no PERSON entity in stored free text |
| 9.13 | LLM isolation (S) | The extraction module cannot import tenant repositories; payload regex shows no tenant ids or customer fields |
| 9.14 | PII in outputs (S) | Articles naming private individuals yield zero PERSON entities outside the allowlist across all output strings (tolerance 0) |
| 9.15 | Injection corpus (S) | 200 articles with injected instructions in English, French and Swahili: at least 98% of field-level outputs match the clean extraction; events whose evidence contains injected text are flagged |
| 9.16 | Token scope (S) | At startup the HF token fails an attempted write |
| 9.17 | Explanation numbers (E) | Any number in an LLM explanation not present in its input causes fallback to the template |
| 9.18 | Single posted price (E) | The experiment designer never assigns two prices to one product, store and time |
| 9.19 | Fairness block (E) | Surge with flat cost yields no increase unless an owner override is logged |
| 9.20 | Coordination (E) | Mutating another tenant's prices, costs or competitor entries leaves a tenant's recommendation bit-identical |
| 9.21 | Missing-not-zero (E) | With a source disabled, its features are NaN with missing indicators set, never 0 |
| 9.22 | Audit immutability (P) | Update and delete on `audit_log` fail for every application role |
| 9.23 | Budget caps (E) | A simulated burst stops at the configured daily cap for each paid source |

---

## 9.10 Launch gates

Before a source, market or feature is enabled in production:

1. Every source used has `terms_verified_at` within the last 12 months and a non-blocked `licence_class`; all [S] and [L] licence, price and limit entries in §9.3 have been checked on the provider's page.
2. Counsel review per launch country covering: tenant agreement and data-processing terms (controller/processor roles), consent basis for T3 contributions, cross-border transfer to LLM providers and to the Supabase region, retention periods, and price-experiment legality (§9.8). Experiments stay disabled per country until done.
3. LLM providers pinned only after their retention and training-use terms are reviewed.
4. Tests 9.1–9.23 pass in CI.
5. Coverage gates pass for any enabled news family and city.

---

## 9.11 Open questions and risks

| # | Item | Type | Impact | Proposed handling |
|---|---|---|---|---|
| 1 | Launch city and language set not fixed | Question | Source choice, language routing and legal review all depend on it | Decide in `00-overview-and-contracts.md` |
| 2 | Open-Meteo commercial terms and price; ERA5 and CHIRPS licence texts | Unverified | Weather may need a paid plan or direct CDS access | Verify before launch; CDS direct access as fallback |
| 3 | Google Places and Routes 2026 pricing, result caps and caching terms only from aggregators | Unverified | Cost and ToS risk | Verify on Google's pages; keep Google optional |
| 4 | ODbL share-alike scope for databases that combine OSM-derived features with tenant data | Legal risk | Could force release of derived databases | Separate schemas; counsel review; prefer Overture for shared POI layers |
| 5 | RWI is non-commercial; GRID3, Kontur, DHS and census licences unverified | Licence risk | Wealth and Nigeria population features may be unavailable | In-house proxy; verify licences |
| 6 | POI completeness for informal businesses is unmeasured and correlates with income | Data risk | Biased competition and income-heterogeneity results | Field audits (`02-geospatial-market.md`) |
| 7 | HF router providers' data retention, logging and training use unreviewed | Privacy and legal risk | Article text could be retained by providers | Pin reviewed providers; self-host fallback |
| 8 | Llama 3.x does not support Swahili, Hausa, Yoruba, Igbo, Zulu or Amharic; NLLB is NC; MADLAD licence unverified | Quality and licence risk | Extraction weak in local languages | Translate with a commercially licensed model; per-language metrics |
| 9 | Prompt Guard 2 unevaluated in African languages | Security risk | Missed injections | Rely on structural controls (no tools, enums, grounding) |
| 10 | Local news coverage is sparse; radio and social media not assessed | Data risk | News features may never pass coverage gates in some cities | Ship news in shadow mode; treat as optional |
| 11 | ACLED commercial licence terms, cadence and training-label permission | Unverified | Validation data may be unavailable | Defer to post-MVP |
| 12 | Official feeds (EskomSePush, EPRA, NBS, Kenya Power, ECG) unverified | Unverified | Tier A may require manual curation | Manual curation in MVP |
| 13 | Legal basis for cross-tenant priors (opt-in vs opt-out) under POPIA, NDPA, Kenya DPA, Ghana DPA | Legal question | Priors may be limited to consenting tenants | Counsel; literature-only priors as fallback |
| 14 | Kenya, Nigeria, Ghana and South Africa price-control, consumer-protection and competition rules not verified | Legal risk | Experiments or some recommendations may be unlawful | Experiments off until counsel review; essentials excluded |
| 15 | Hub-and-spoke coordination risk from a shared pricing engine | Legal risk | Competition-law exposure | §9.8.4 rules and test 9.20 |
| 16 | Supabase region and data residency for African tenants | Question | Cross-border transfer obligations | Counsel; document region choice |
| 17 | Supabase lacks h3-pg (open request) | Engineering | H3 computed in Python | As in `02-geospatial-market.md` |
| 18 | DP for variance (τ²) releases not worked out; poor utility at small n | Method gap | External benchmarks limited to means | Release only means and ranges |
| 19 | Fairness thresholds (0.5 SD surge, 2% cost) uncalibrated; no African fairness evidence | Evidence gap | Over- or under-blocking | Monitor overrides and complaints; revisit |
| 20 | WorldPop R2025A is an alpha and may change | Data risk | Silent feature shifts | Pin release; compare on each update |

---

## 9.12 References

URLs are as given in the research notes. Tags show how each was checked during research on 2026-10-06.

**Places, maps, routing, geocoding**
- OpenStreetMap copyright and ODbL: https://www.openstreetmap.org/copyright [L]
- Overpass API wiki: https://wiki.openstreetmap.org/wiki/Overpass_API [L]
- OSM wiki on Overture: https://wiki.openstreetmap.org/wiki/Overture_Maps [S]
- Geofabrik Africa extracts: https://download.geofabrik.de/africa.html [L]
- Overture Places guide: https://docs.overturemaps.org/guides/places/ ; schema: https://docs.overturemaps.org/schema/reference/places/place/ ; AWS registry: https://registry.opendata.aws/overture/ [S]
- Echo Analytics on Overture places: https://www.echo-analytics.com/blog/analyzing-overture-maps-foundations-places-data [S, unverified]
- Foursquare OS Places announcement: https://foursquare.com/resources/blog/products/foursquare-open-source-places-a-new-foundational-dataset-for-the-geospatial-community/ ; Hugging Face: https://huggingface.co/datasets/foursquare/fsq-os-places [S]
- Google Maps Platform pricing: https://developers.google.com/maps/billing-and-pricing/pricing ; aggregators: https://bizcollect.dev/blog/google-places-api-pricing , https://www.woosmap.com/blog/google-maps-api-pricing-breakdown [S]
- Google Maps Platform Service Specific Terms: https://cloud.google.com/maps-platform/terms/maps-service-terms [S]
- Herfort et al. (2023), Nature Communications: https://www.nature.com/articles/s41467-023-39698-6 [S]
- OSRM licence: https://blog.mapbox.com/more-open-project-osrm-switches-to-bsd-fde88d5ef462 [S]
- Valhalla: https://github.com/valhalla/valhalla [S]
- openrouteservice plans and restrictions: https://account.heigit.org/info/plans , https://openrouteservice.org/restrictions/ [S]
- Supabase extensions: https://supabase.com/docs/guides/database/extensions ; h3-pg request: https://github.com/supabase/postgres/issues/245 [S]
- Williams et al. (2015), Digital Matatus: https://doi.org/10.1016/j.jtrangeo.2015.10.005 [L]
- Nominatim usage policy: https://operations.osmfoundation.org/policies/nominatim/ [S]
- GeoNames export and web services: https://www.geonames.org/export/ [S]
- Mordecai 3: https://github.com/ahalterman/mordecai3 [V]

**Population, built environment, wealth, surveys**
- WorldPop R2025A release statement: https://data.worldpop.org/repo/prj/Global_2015_2030/R2025A/doc/Global2_Release_Statement_R2025A_v1.pdf ; hub: https://hub.worldpop.org/geodata/listing?id=135 [S]
- GHSL: https://data.jrc.ec.europa.eu/dataset/2ff68a52-5b5b-4a22-8f40-c41da8332cfe , https://human-settlement.emergency.copernicus.eu/datasets.php [S]
- HRSL: https://registry.opendata.aws/dataforgood-fb-hrsl/ , https://data.humdata.org/dataset/highresolutionpopulationdensitymaps [S]
- GRID3 Nigeria: https://data.humdata.org/dataset/bottom-up-gridded-population-estimates-for-nigeria-version-2-0 , https://data.grid3.org/datasets/grid3-nga-population-v2-1 [S]
- Kontur South Africa: https://data.humdata.org/dataset/kontur-population-south-africa [S]
- Relative Wealth Index: https://data.humdata.org/dataset/relative-wealth-index ; Chi et al. (2022), PNAS: https://www.pnas.org/content/pnas/119/3/e2113658119.full.pdf [S]
- DHS GPS displacement: https://dhsprogram.com/pubs/pdf/SAR7/SAR7.pdf , https://dhsprogram.com/Methodology/GPS-Data.cfm [S]
- Google Open Buildings: https://sites.research.google/gr/open-buildings/ ; Temporal: https://sites.research.google/gr/open-buildings/temporal/ [S]
- VIIRS nighttime lights (EOG): https://eogdata.mines.edu/products/vnl/ [S]; Henderson, Storeygard & Weil (2012): https://doi.org/10.1257/aer.102.2.994 [L]

**Weather**
- Open-Meteo Historical Weather API: https://open-meteo.com/en/docs/historical-weather-api [S]
- Open-Meteo Previous Runs API: https://open-meteo.com/en/docs/previous-runs-api ; blog: https://openmeteo.substack.com/p/weather-forecasts-from-previous-model-runs [S]
- Open-Meteo Historical Forecast API: https://open-meteo.com/en/docs/historical-forecast-api [title]
- CHIRPS: Funk et al. (2015): https://www.nature.com/articles/sdata201566 ; UCSB CHC: https://www.chc.ucsb.edu/data [S]
- ERA5/CHIRPS in Ethiopia: https://www.iris.sssup.it/retrieve/d3bd82b5-a3be-4b7c-8056-2324d61e8687/s00703-024-01008-0.pdf [S]
- Ghana and South Africa comparison: https://arxiv.org/html/2501.14829v3 [S]
- ERA5 vs ERA-Interim over Africa: https://www.researchgate.net/publication/363304224_Verification_of_ERA5_and_ERA-Interim_precipitation_over_Africa_at_intra-annual_and_interannual_timescales [S]
- Global ERA5 evaluation: https://www.sciencedirect.com/science/article/pii/S1470160X24009385 [S]

**News, events, calendars**
- GDELT DOC 2.0: https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/ ; GDELT 2.0: https://blog.gdeltproject.org/gdelt-2-0-our-global-world-in-realtime/ [S]
- Hammond & Weidmann (2014): https://journals.sagepub.com/doi/full/10.1177/2053168014539924 [S]
- Wang et al. (2016), Science: https://doi.org/10.1126/science.aaf6758 [L]
- ACLED API docs, FAQ and EULA: https://acleddata.com/acled-api-documentation , https://acleddata.com/myacled-faqs , https://acleddata.com/eula [S]
- Event Registry / NewsAPI.ai plans: https://newsapi.ai/plans [S]
- `holidays` package: https://pypi.org/project/holidays/ ; `hijridate`: https://pypi.org/project/hijridate/ [V, package source]

**Language models, embeddings, safety**
- HF Inference Providers index: https://github.com/huggingface/hub-docs/blob/main/docs/inference-providers/index.md [V]
- HF pricing: https://github.com/huggingface/hub-docs/blob/main/docs/inference-providers/pricing.md [V]
- HF structured outputs: https://github.com/huggingface/hub-docs/blob/main/docs/inference-providers/guides/structured-output.md [V]
- HF function calling: https://github.com/huggingface/hub-docs/blob/main/docs/inference-providers/guides/function-calling.md [V]
- HF Hub API (provider mapping): https://github.com/huggingface/hub-docs/blob/main/docs/inference-providers/hub-api.md [V]
- HF Hub rate limits: https://huggingface.co/docs/hub/rate-limits [S]
- Llama 3.3 model card: https://github.com/meta-llama/llama-models/blob/main/models/llama3_3/MODEL_CARD.md [V]
- Llama 4 model card: https://llama.com/docs/model-cards-and-prompt-formats/llama4 [S]
- Provider prices (Artificial Analysis): https://artificialanalysis.ai/models/llama-3-3-instruct-70b/providers [S]
- IrokoBench (NAACL 2025): https://aclanthology.org/2025.naacl-long.139/ [S]
- vLLM structured outputs: https://github.com/vllm-project/vllm/blob/main/docs/features/structured_outputs.md [V]
- NLLB README (licence): https://github.com/facebookresearch/fairseq/tree/nllb [V]
- sentence-transformers pretrained models: https://github.com/UKPLab/sentence-transformers/blob/master/docs/sentence_transformer/pretrained_models.md [V]
- BGE-M3: https://github.com/FlagOpen/FlagEmbedding/blob/master/research/BGE_M3/README.md [V]
- multilingual-e5: https://github.com/microsoft/unilm/blob/master/e5/README.md [V]
- Llama Prompt Guard 2 model card: https://github.com/meta-llama/PurpleLlama/blob/main/Llama-Prompt-Guard-2/86M/MODEL_CARD.md [V]
- Greshake et al. (2023), indirect prompt injection: https://arxiv.org/abs/2302.12173 [L]
- Hines et al. (2024), spotlighting: https://arxiv.org/abs/2403.14720 [L]
- OWASP Top 10 for LLM Applications: https://genai.owasp.org/llm-top-10/ [L]
- Structured-output control-plane attacks: https://arxiv.org/abs/2503.24191 [title]
- Microsoft Presidio: https://github.com/microsoft/presidio [L]
- Glasserman & Lin (2023), look-ahead bias: https://arxiv.org/abs/2309.17322 [L]

**Cross-tenant learning and privacy**
- Lin & Zeng (2010), Biometrika: https://doi.org/10.1093/biomet/asq006 [L]
- Burke, Ensor & Riley (2017), Stat Med: https://doi.org/10.1002/sim.7141 [L]
- McMahan et al. (2017), FedAvg: https://proceedings.mlr.press/v54/mcmahan17a.html [L]
- Bonawitz et al. (2017), secure aggregation: https://doi.org/10.1145/3133956.3133982 [L]
- Dwork, McSherry, Nissim & Smith (2006): https://doi.org/10.1007/11681878_14 [L]
- Dwork & Roth (2014): https://doi.org/10.1561/0400000042 [L]
- OpenDP: https://opendp.org/ [L]
- Hundepool et al. (2012), Statistical Disclosure Control: https://doi.org/10.1002/9781118348239 [L]

**Fairness and law**
- Kahneman, Knetsch & Thaler (1986), AER: https://eml.berkeley.edu/~saez/course131/Kahneman-FairnessConstraintProfit-1986.pdf ; https://econpapers.repec.org/RePEc:aea:aecrev:v:76:y:1986:i:4:p:728-41 [S]
- Kalyanaram & Winer (1995): https://doi.org/10.1287/mksc.14.3.G161 [L]
- US Robinson–Patman Act, 15 U.S.C. §13: https://www.law.cornell.edu/uscode/text/15/13 [L]
- EU Directive 2019/2161 (Omnibus): https://eur-lex.europa.eu/eli/dir/2019/2161/oj [L]
- Bojinov, Simchi-Levi & Zhao (2023), switchback experiments: https://pubsonline.informs.org/doi/10.1287/mnsc.2022.4583 [S]
- South Africa Competition Act 89 of 1998, Consumer Protection Act 68 of 2008, POPIA; Kenya Competition Act 2010 and Data Protection Act 2019; Nigeria FCCPA 2018 and NDPA 2023; Ghana Data Protection Act 2012 (Act 843): named from memory in the notes or here, no URLs in the notes, **not verified** [L]

---

## Summary

1. Catalogues every external source (places, routing, geocoding, population, wealth, weather, news, events, LLMs, embeddings, safety models) with coverage, licence, cost, limits, cadence, caching and failure rules, and flags every unverified term with its last-checked date.
2. Recommends an open, commercially licensed MVP stack (OSM/Overture/Foursquare, self-hosted OSRM and Nominatim, WorldPop/GHSL/Open Buildings/VIIRS, ERA5-Land + CHIRPS, news in shadow mode) and excludes NC sources such as RWI and NLLB by code.
3. Defines tenant isolation: only per-business summary records cross the boundary, with leave-one-tenant-out priors, k_min = 10 businesses from 5 tenants, a dominance cap, monthly batching and optional Laplace DP for n ≥ 50.
4. Restricts LLM inputs to public text by allowlist, strips PII, treats article text as untrusted, and specifies secrets, least-privilege database roles, no service-role key in tenant jobs, and an append-only audit log.
5. Constrains price advice to one posted price, fairness-checked and never personalised or coordinated across tenants; Kenyan, Nigerian, Ghanaian and South African law is named but unverified, so experiments stay off until counsel review.
