# Phase 5 — From anomaly maps to an archaeological review catalogue

Phase 5 groups repeated spatial anomalies into reviewable footprints, measures the underlying satellite signals and explains each investigation-priority score. The published Buto run uses the frozen Phase 4 gated validation mosaics. Candidate locations, individual reports and expert comments are restricted research material.

## Open and test the completed run

From PowerShell in `D:\sarchaeology-ai`:

```powershell
$py = ".\.venv\Scripts\python.exe"
$cli = "scripts/phase5_application/candidates.py"
& $py -m pytest tests/test_phase5_candidates.py -q
& $py $cli verify
& $py -m streamlit run app/streamlit_app.py --server.address 127.0.0.1
```

Choose **Open the archaeological candidate review**. The default run is `outputs/phase5/buto_2024`. Set `SARCHAEOLOGY_REVIEW_RUN` to open another completed run. The interface renders local images and projected footprints without requesting map tiles from an external service.

For a static review, open `index.html` in the run directory or extract `restricted_research.zip` and open its index. Load `gis/catalogue.geojson`, a quarterly anomaly raster and its matching evidence raster into QGIS. Rank 1 has the highest investigation priority. Individual reports link to the full machine-readable evidence profile.

### Reproduce without overwriting prior results

The Phase 4 maps, prepared data, Phase 1 sensor products, and the frozen Phase 3 spectral checkpoint must be present. The Phase 5 build consumes the existing frozen spectral residual export. If this export is missing, generate it once using the same Phase 3 checkpoint configuration:

```powershell
& $py scripts/phase3_optical/export_optical_phase38.py --split val --year 2024 --spectral-residuals --output outputs/phase5_inputs/spectral
& $py $cli build --run outputs/phase5/reproduction01
& $py $cli verify --run outputs/phase5/reproduction01
& $py -m pytest -q
```

A build writes to a temporary sibling, checks its products and publishes a fresh destination. A second build at an existing destination fails to protect review history. Inputs and outputs are local. Existing encoders and fusion models run read-only. Phase 5 neither fits on nor scores the audited test observations. Independent-validation geography checks can read the original footprint metadata.

## Results and evidence limits

| Stage | Exercised at Buto | Evidence boundary |
|---|---|---|
| 5.1–5.2 | Four 2024 score mosaics, 109 cross-quarter candidates and geometry descriptors | 82 footprints were below four native score-cell areas. Their morphology contribution is zero. |
| 5.3 | Direct VV/VH backscatter, signed B04/B08/B11/B12 reconstruction residuals, elevation-derived relief, slope and ruggedness | The evidence channels have shared environmental causes. No combination establishes origin. |
| 5.4 | 29 dated SAR scenes, four fused quarters and 16 optical quarters in 2022–2025 | Year-to-year comparisons measure optical contrast. Measured rainfall comparisons require an additional precipitation series. |
| 5.5 | Published settlement point, historical itinerary cross-reference, canal-source record, quarterly vegetation and water proxies | Buto's gazetteer point is contextual. The historical-map reference is schematic. A locally verified ancient canal alignment, surveyed modern infrastructure layers and an authoritative land-use raster were unavailable in this run. |
| 5.6–5.7 | Per-quarter edge screening, minimum size filter, nuisance flags, reason codes and numerical priority terms | Screening flags express alternative explanations. Their frequency cannot determine the archaeological false-positive rate. |
| 5.8 | A pending independent-validation report and curation register were generated. Survey matching, leakage rejection, baseline comparisons, ranking metrics and block resampling are covered by synthetic tests. | No independently surveyed site/control set or second scored landscape was supplied. Archaeological precision, recall and transfer remain unmeasured. |
| 5.9–5.10 | Local review interface, versioned feedback database, individual illustrated reports, GIS layers and two sharing packages | The initial expert feedback table is empty. Real archaeologist agreement awaits independent assessments. |

## Extraction, morphology and spatial support

The Phase 4 training-calibrated ensemble threshold is read from provenance (approximately 0.91345 for Buto). Candidates use the original 40 m scoring cells displayed on the Phase 1 10 m grid. Each quarter independently erodes its observed mask by 40 m, including the outer raster boundary. This limits truncation artifacts at the cost of losing legitimate edge signals. Across the four maps, 744 of the original 11,312 above-threshold display pixel-quarters fell inside this guard.

A logical union consolidates eligible exceedances across quarters. A small dilation bounded by observed support groups adjacent exceedances. Polygonization uses the original above-threshold pixels; the joining pixels add no polygon area. A group can therefore have multiple polygon parts. The default minimum actual area is 1,600 m². One group failed that filter. Stable IDs derive from the occupied mask and its affine. `screening.json` records losses and warning counts. Quarterly screen rasters encode eligible background (0), the edge guard (1), a rejected small group (2), and retained exceedance (3). Nodata identifies unobserved pixels.

Measurements use projected metric geometry. The descriptors include area, perimeter, circular compactness, minimum rotated rectangle fill, elongation, long-axis azimuth, fitted ellipse overlap, and central symmetry. Canny boundaries and seeded probabilistic Hough line segments provide linear-pattern evidence. Local vegetation and relief edges offer separate orientation comparisons for resolved elongated candidates. Orthogonal sensor cells, field boundaries and other modern structures can produce regular shapes. Geometric resolution gating uses an area threshold, and even an accepted narrow feature can be unresolved across its width.

## Sensor evidence and time

The evidence reader checks source grids against the validation mosaic window. It verifies the frozen optical export's split, patch membership, spatial transform and upstream hashes. Overlapping spectral residual patches are averaged on their native footprint. Each signed value is observed minus reconstructed surface reflectance. Terrain comes from the Phase 4 physical-unit raster.

Each candidate is compared with a local background extending 120 m from its boundary. The immediately adjacent display pixel and other retained candidates are excluded. A contrast requires at least half the footprint to be valid and at least 16 valid background pixels. Insufficient measurements are recorded as null. The profiles retain dated VV and VH median contrasts, quarterly sensor support, gate weights, cross-reconstruction and disagreement percentiles, optical residuals, terrain contrast, and clear-observation counts. Reports include actual optical acquisition dates where supplied in the composite metadata.

A quarter is usable for recurrence when at least half the footprint passes the edge and observation requirements. A quarter counts as recurrent when at least 25% of its eligible footprint exceeds the anomaly threshold. Persistence is recurring quarters divided by usable quarters. SAR stability is the larger fraction of available acquisitions showing VV contrast of at least +1 dB or at most −1 dB, requiring at least three acquisitions. Optical histories retain annual mean contrasts and the sign-consistent recurrence of NDVI contrast across matching quarters in different years.

The relative moisture comparison contrasts high- and low-NDMI quarters if their median NDMI differs by at least 0.05. NDMI measures canopy and surface moisture responses. To compare observed precipitation, add `rainfall: path/to/quarters.csv` to the config. It accepts unique `period,precipitation_mm,source` rows with nonnegative measured quarterly totals. The profile then reports exceedance differences between the wettest and driest observed periods. The Buto run has no measured precipitation input.

## Historical context, modern land use and alternative explanations

The registry is `configs/phase5/context.yaml`. ToposText supplies the published Bouto representative location and a Pleiades cross-reference. ToposText permits its linked open data to be used with attribution. The Tabula Peutingeriana place entry is linked as historical-map context. Schiestl's discussion documents the uncertainty of the western Butic Canal route. No fabricated waterway geometry is drawn or used to rank sites.

- ToposText, Bouto and the distribution terms. [Place record](https://topostext.org/place/312307UBou) · [Data downloads](https://topostext.org/TT-downloads)
- Katholische Universität Eichstätt-Ingolstadt, Buto in the Tabula Peutingeriana. [Historical map entry](https://www1.ku.de/ggf/ag/tabula_peutingeriana/trefferanzeige_en.php?id=2792)
- R. Schiestl (2021), *A new look at the Butic Canal, Egypt*. [Article](https://doi.org/10.5194/egqsj-70-29-2021)

For each additional vector or georeferenced raster, enter its `path`, `kind`, `source`, `epoch`, `attribution`, permitted use and `uncertainty_m`. Use `null` for unknown positional uncertainty. Use sources whose permissions cover the intended local and restricted-package use. Distances to representative points describe distance to the mapped geometry; actual site extents may differ. Registry sources and derived copies accompany the restricted GIS package. Documents without reliable georeferencing retain bibliographic provenance only.

Vector kinds used in screening include `road`, `field_boundary`, `irrigation`, `building`, `infrastructure`, `excavation`, `drainage`, `sediment` and `erosion`. Additional rasters can include a licensed georeferenced historical map or categorical land-use imagery. Land-use entries can map categorical pixel codes through a `classes` dictionary. The importer preserves categorical values using nearest-neighbor resampling. Modern mapped overlap and dominant land-use classes can add a penalty. Known ancient settlement proximity has zero ranking weight. Quarterly local surface-class rasters distinguish dense vegetation, water-like spectra and other observed surfaces. These proxies are available regardless of external land-use access.

## Ranking and uncertainty

All terms are visible in `reports/<candidate>/evidence.json`. The default weights are 35% anomaly magnitude, 25% persistence, 20% diversity of supporting sensor families, 10% resolved morphology and 10% acquisition stability. SAR strength uses absolute local backscatter differences scaled to 2 dB. Optical strength uses excess absolute reconstruction residual relative to the background, scaled to 0.035 reflectance. DEM strength uses absolute local-relief contrast scaled to 0.5 m. Each contribution is bounded to 0–1. Diversity counts the sensor families reaching 0.5 and divides by three. Missing inputs remain explicit and lower the coverage term.

Quality scales the weighted sum using sensor availability, average source quality and usable quarters. Capped screening penalties are subtracted, and the result is clipped to [0, 100]. Penalty points and reason codes are recorded separately. A review flag does not delete the footprint. This is a fixed research prioritization policy with hand-set scales; empirical archaeological probabilities require representative labels and independent calibration. The run stores the policy, its digest, model hashes, source digests and observation dates.

## Independent validation protocol

Curate the reference set before inspecting the independent landscape's ranking. A qualified curator should document survey coverage and dates, coordinate accuracy, reasons for a positive determination or a surveyed negative, comparable footprint sizes, survey method, and modern/environmental confounders. Pair known sites with independently surveyed non-site controls in the same landscape and spatial block. Define blocks at a defensible spatial-dependence scale. Keep contextual gazetteer points and review feedback outside this set.

`validation/reference_register_template.csv` supplies the tabular fields. The evaluator takes georeferenced polygon survey units with `unit_id, label, landscape, block_id, matched_set, curator, source, survey_method, survey_date, role`. Set `label` to 1 for a confirmed site and 0 for a verified control. Set `role` to `independent_evaluation`. Matched sets must contain both classes and units must have disjoint interiors.

Supply one JSON study manifest per independently scored landscape. Relative filenames resolve beside the manifest:

```json
{
  "role": "independent_transfer",
  "landscape": "survey_area_02",
  "model_sha256": "<frozen fusion checkpoint digest>",
  "ranking_policy_sha256": "<policy digest>",
  "catalogue": "catalogue.geojson",
  "coverage": "surveyable_coverage.geojson",
  "references": "independent_survey_units.geojson",
  "evidence_rasters": ["quarter_1_evidence.tif", "quarter_2_evidence.tif"]
}
```

```powershell
& $py $cli validate --study D:/surveys/area02/study.json --k 10 --out outputs/phase5/independent01
```

The protocol rejects reference units intersecting the buffered original train, validation, test or excluded footprints. It checks the frozen model/policy declarations and digests the supplied files. Reference units outside surveyed raster support are excluded. A unit takes the maximum priority of a candidate intersecting its interior, with zero for a missed unit. SAR-only, optical-only and DEM-only controls use fixed local contrasts on the full eligible map. The classical comparator averages these three contrasts. These are sensor-specific, untrained baselines. They can score a surveyed unit even where fusion extracted no candidate.

Precision@K and recall@K are calculated on the same eligible, matched units for all methods. Ranking ties receive their expected fractional hit count. The report includes AP, ROC AUC, positive-priority site recall, control flag rate and per-landscape results. Four or more spatial blocks enable percentile intervals from resampling whole blocks. Small block counts, survey ascertainment and the site/control sampling ratio limit extrapolation. A pending report with zero submitted units carries no performance estimate.

## Review and sharing

The review form supports **plausible**, **rejected** and **needs investigation**, with reviewer identifier and notes. SQLite records a new timestamped revision for each decision, bound to the candidate and catalogue hash. Optimistic revision checks protect concurrent edits. Pairwise agreement uses the latest shared assessments and reports counts, disagreement IDs, raw agreement and Cohen's kappa where defined.

```powershell
& $py $cli feedback --out outputs/phase5/reviewer_export01
```

This exports private review history, current decisions and agreement. A reviewer assessment does not enter the independent reference set or trigger training.

For an encrypted copy, the custodian can supply a separately held Fernet key file. Both commands require a fresh output filename. Share the key separately with authorized collaborators. The plaintext research workspace remains local.

```powershell
& $py $cli seal --key-file D:/private/group.key --out outputs/phase5/authorized-research.fernet
& $py $cli unseal --key-file D:/private/group.key --encrypted outputs/phase5/authorized-research.fernet --out D:/private/received-research.zip
```

The two commands use the existing Python environment's authenticated symmetric encryption implementation. [Fernet key and token format](https://cryptography.io/en/latest/fernet/)

| Product | Intended use |
|---|---|
| `restricted_research.zip` | Offline HTML index and illustrated reports, exact footprints, georeferenced imagery, screening, context, policy and provenance. Handle locally or through an access-controlled channel. The ZIP itself is unencrypted. |
| `public_summary.zip` | Coordinate-free aggregate count, quarters and validation status. Constructed using an allowlist. No candidate rows, imagery, report links, original paths, IDs or review notes. |
| `feedback/expert.sqlite` | Private local working database, excluded from both ZIPs. |
| `validation/independent_validation.json` | Reference-set status and supported metrics. |
