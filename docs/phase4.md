# Phase 4 — SAR, optical and terrain fusion at Buto

Phase 4 turns a Phase 1 geographic patch and calendar quarter into a local feature map, then scores departures from a training reference. Its normal entry point is `scripts/phase4_fusion/pipeline.py`. The Buto configuration pins the existing SAR run and the Phase 3.8 Swin/geographic export family. All commands below run in PowerShell from the repository root.

## Test the installation and inspect the existing run

```powershell
Set-Location D:\sarchaeology-ai
.\.venv\Scripts\python.exe -m pytest tests/test_phase4_fusion.py -q
.\.venv\Scripts\python.exe scripts/phase4_fusion/pipeline.py verify --heldout outputs/phase4/buto_heldout
.\.venv\Scripts\python.exe -m pytest -q
```

The first command after changing directory runs a small CPU study that includes training, scoring, masking, spatial leakage checks, polygon export and optional binary labels. It creates temporary fixtures. The verifier reads the completed real products, checks model and detector lineage, grid placement, valid-pixel counts and seasonal observation counts, and rasterizes the GeoJSON back onto each quarterly grid. The last command also runs the earlier phases’ regression tests.

Open `outputs/phase4/buto_maps/2024Q1_anomalies.tif` in QGIS. Display band 1 with a continuous 0–1 color ramp. Add `candidates.geojson` above it and filter the vector layer with `"period" = '2024Q1'`. Add the coverage and persistence rasters to inspect gaps or candidates that recur in several quarters. The polygons use WGS84 longitude/latitude. The rasters use the original projected grid, clipped to the validation windows.

## Reproduce the experiment

### Verified Buto run

| Phase | Ran on real data | Boundary of the evidence |
|---|---|---|
| 4.1 | 240 training and 20 validation locations across four quarters, 1,040 development patch-quarters | A post-lock alignment step prepares 136 test patch-quarters for inference. The 123 excluded footprints remain metadata-only |
| 4.2 | Nine terrain bands with 1,072,983 distinct training pixels contributing to normalization | Surface variation in one low-relief AOI |
| 4.3–4.4 | Native SAR/optical features and terrain tensors passed through all six model variants | Entirely absent, corrupt and deliberately misregistered inputs also have synthetic regression tests |
| 4.5 | All six models completed eight epochs on the RTX 5070 Ti, with validation curves and checkpoints | One initialization per variant. The 32-channel gated representation had effective rank 3.13 on validation |
| 4.6 | Gated export used 2,048 reference cells and 4,096 calibration cells in separate training strips | The 0.91345 ensemble threshold reflects training geography |
| 4.7 | Matched-cell prediction MSE was 0.0893 for concatenation, 0.0938 for gated fusion, 0.1018 for attention, 0.1225 for DEM, 0.2031 for optical and 0.3627 for SAR | Independent label ranking and verified-control false-positive rates were exercised on fixtures. No field labels were supplied for Buto |
| 4.8 | Four quarterly mosaics, 80 pooled validation vectors, a persistence raster and 327 quarter-specific polygons passed read-back checks | Q1/Q2/Q3/Q4 produced 101/72/78/76 polygons. These are threshold exceedances. A location may recur across quarters |

Concatenation had the lowest joint reconstruction MSE among the six default configurations. A seventh eight-epoch run kept the gated architecture, seed, temporal loss and data fixed and disabled the contrastive alignment loss. Its validation MSE was 0.0858 and its effective rank was 4.53. The baseline gated run scored 0.0938 with rank 3.13. When only the hidden SAR target was scored, the optical-only baseline achieved 0.0659 MSE versus 0.0692 for concatenation. Fusion performance depends on which input is available and which task is scored. The vegetation proxy flagged 0.55–1.48% of supported pixels across quarters. No supported pixels met the water-proxy criteria. Archaeological accuracy remains unmeasured.

The completed comparison, validation curves and buffered validation-region errors are in `outputs/phase4/buto_final_evaluation`. The gated baseline produced the distributed maps. The package is `outputs/phase4/buto_candidates.zip`. To create another archive after verification, run `& .\.venv\Scripts\python.exe scripts/phase4_fusion/pipeline.py verify --package outputs/phase4/buto_maps_review.zip` with an unused filename.

### Run commands

The local Python 3.11 environment contains the packages in `pyproject.toml` and the CUDA 12.8 PyTorch build. From a new checkout, `uv sync --locked` installs the locked dependencies. Supply the Phase 1 data, frozen checkpoints and exports listed in `configs/phase4/buto_2024.yaml` separately. Git ignores these large products. A prepared cache can be reused by multiple runs.

```powershell
$py = ".\.venv\Scripts\python.exe"
$cli = "scripts/phase4_fusion/pipeline.py"
$root = "outputs/phase4/reproduction01"
& $py $cli prepare --prepared "$root/data"
& $py $cli train --prepared "$root/data" --runs "$root/runs"
& $py $cli train --prepared "$root/data" --variants gated --alignment-weight 0 --runs "$root/no_alignment"
& $py $cli export --prepared "$root/data" --model "$root/runs/gated" --maps "$root/maps"
& $py $cli evaluate --prepared "$root/data" --runs "$root/runs" --maps "$root/maps" --comparison "$root/no_alignment/gated" --report "$root/report"
& $py $cli align-heldout --prepared "$root/data" --model "$root/runs/gated" --destination "$root/heldout"
& $py $cli verify --prepared "$root/data" --maps "$root/maps" --heldout "$root/heldout"
```

For a one-epoch check on the **existing** real cache, run `& $py $cli train --variants gated --epochs 1 --runs outputs/phase4/smoke01`. Each write command requires a fresh destination. Choose a new run name for a repeat. To repeat the maps with concatenation or attention, point `export --model` to its completed run directory and give the maps a fresh directory. The default export uses the prespecified gated model. `--config` accepts a YAML file for preparation, training and detector settings. Both the scoring code and the dataset restrict development to train and validation.

## 4.1 — Alignment and split contract

`fusion/prepare.py` reads the Phase 1 split Parquets and dataset metadata. It checks patch bounds against the master affine, CRS, window size, extent, unique patch identity and inter-split separation. Distances are edge-to-edge in source pixels. It checks all matched SAR source grids. For each Phase 3.8 export it checks the export split/year, reference grid, row/column association, uniqueness, encoder lineage, and the identity, affine and dimensions of each required raster. Temporal matching uses the acquisition year and quarter. The 2024 archive contains 7/7/8/7 acquisition dates in Q1–Q4.

`manifest.parquet` holds one row per Phase 1 patch and quarter. It includes source window, source archive, SAR indices and selected acquisition indices, original split, readiness, cache path and per-modality coverage. Join indices to `sar_sources` in `alignment_qa.json` to recover dates and original raster paths. Test and excluded entries are geographic reservations during development. Their cache paths in the development manifest are empty. Training, evaluation, detector calibration and map export use the train and validation caches.

After the experiment is fixed, `align-heldout` verifies the frozen development run and upstream hashes, then aligns only the 34 original test locations using the stored train statistics and frozen features. It produces no predictions or performance metrics. The inference directory's `unified_manifest.parquet` links every split via `cache_group` and `cache`. Resolve development paths relative to the original prepared directory and held-out paths relative to the inference directory. The dataset API for this sidecar is `MultiModalArchaeologyDataset("outputs/phase4/buto_heldout", "test", inference_only=True)`. The normal training and evaluation entry points reject this cache.

## 4.2 — Terrain

The existing Phase 1 buffered 30 m Copernicus DSM is the input. Its projected metre grid, spacing, source tag and vertical datum are checked against `dem_processed.parquet`. The real source records EGM2008. The terrain module calculates elevation, slope in degrees, downhill aspect as east/north components, Laplacian curvature, RMS height difference from the eight immediate neighbors, 3×3 topographic position, elevation minus the 11×11 neighborhood mean, and mean hillshade from four azimuths at 45° altitude. Aspect components avoid an interpolation seam at north. Both components are zero on a flat surface.

A common 11×11 validity stencil excludes derivatives that touch nodata or an unobserved DEM edge. Derivatives are calculated at 30 m before bilinear reprojection to the 10 m SAR grid and application of the Phase 1 terrain mask. `terrain.tif` retains physical units. The feature tensors use per-channel mean and standard deviation over the **union** of valid training footprints, with clipping at eight standard deviations. Overlapping windows count each physical training pixel once. `terrain_statistics.json` stores the statistics, valid count and source item IDs. Upsampling does not improve the DEM’s effective resolution. Elevation and its derivatives describe the surface. Buried structures require independent evidence.

## 4.3 — Dataset

`sarchaeology.datasets.multimodal_dataset.MultiModalArchaeologyDataset(prepared, split)` returns one patch-quarter. The train and val constructors verify manifest lineage and membership. Use a standard PyTorch DataLoader, or pass `QuarterPairSampler(dataset.rows, batch_size, seed)` as its batch sampler and set `sampler.epoch` before each training epoch.

| Tensor | Shape per sample | Meaning |
|---|---:|---|
| sar_db, sar_count | 2×128×128, 128×128 | Valid-date mean VV/VH in dB and count of accepted observations per pixel |
| sar_features | 128×16×16 | Phase 2 CAE encoder map averaged across eligible acquisitions with local validity weights |
| stage0, stage1 | 96×32×32, 192×16×16 | Frozen unmasked optical Swin features at native 40 and 80 m strides |
| optical_embedding | 256 | Geographic projection from the Phase 3.8 masked-view export |
| dem | 9×128×128 | Training-standardized aligned terrain |
| optical_valid, dem_valid | 128×128 each | Observed input support after native stage masks and terrain coverage |

The sample also carries patch ID, period, quarter, source row/column and total matched acquisition dates. For SAR, individual dates must satisfy the original 0.85 valid-fraction criterion; the frozen encoder receives the exact Phase 2 train-only normalization. A pixel with zero observations has zero count. Missing optical rows produce zero-valued features and a false optical mask. An indexed row with a corrupt or absent required artifact fails preparation. The fit path compacts tensors in host memory and transfers minibatches to the device. The default Windows loader uses zero workers.

## 4.4 — Fusion

The encoder checkpoint and optical exports stay frozen. A two-layer pointwise terrain encoder operates on masked 40 m cell means. Each sensor projects to 32 shared channels. The optical branch sums a Stage 0 projection, a Stage 1 projection and a broadcast patch embedding. Coarser maps contribute within their native cells by nearest-neighbor expansion. Model output is a 32×32 grid per patch. Patch context and the encoder receptive fields make the effective evidence footprint larger than one output cell.

The gated model learns three spatial logits from features and quality. Its softmax excludes missing modalities. Concatenation uses zeroed blocks, explicit presence indicators and a pointwise projection. Attention uses a learned four-head query over the three local sensor tokens. The three single-sensor models use the same prediction targets and output width. An all-missing cell has zero support in every variant.

## 4.5 — Learning and reproducibility

The fixed targets are cell-mean VV/VH transformed as `(dB + 15)/5`, the first eight frozen Stage 0 channels and all nine normalized terrain features. Predicting fixed targets keeps the target representation independent of fusion-head updates. During fusion training, each patch-quarter hides SAR, optical, terrain or nothing with equal probability. The masked sensor’s prediction error receives four times the nominal weight where another input supports prediction. Each sensor contributes one mean channel loss. Single-sensor baselines observe their own sensor and predict all supported targets.

A cross-sensor contrastive loss uses co-located positive pairs. Negative pairs require at least 128 source pixels between patch edges. The temperature is 0.15. Adjacent quarters of the same location can contribute a cosine-consistency term. The default loss weights are 0.05 for alignment and 0.03 for temporal consistency. Set the corresponding YAML weight to zero to ablate that term. Evaluation remains deterministic over the same validation rows and sensor-removal scenarios.

Each architecture gets eight epochs, AdamW, batch size 16, an initial learning rate of 0.001, seed 42, and gradient clipping at 2. The minimum nominal validation MSE selects a checkpoint within each architecture. `history.csv` contains training loss components, validation MSE, effective covariance rank and mean feature standard deviation per epoch. `run.json`, `validation.parquet` and `best.pt` link model settings, environment, validation errors, checkpoint digest and preparation hash. These local records are readable without an MLflow server. Hardware and library changes can affect floating-point reproducibility.

## 4.6 — Detection and support

The western part of the training strip supplies reference cells. A disjoint eastern training strip calibrates the detectors, separated by the configured buffer. Both strips were eligible for upstream **representation** learning. Sampling caps the contribution of each patch-quarter. Within-strip patch footprints can overlap. The detector estimates standardized RMS feature distance, mean five-neighbor distance, Isolation Forest score, mean leave-one-modality-out prediction error, and pairwise cosine disagreement. The reference sample fits the distance scale, kNN and forest. Empirical distributions in the calibration strip convert all five outputs to comparable percentiles. The exported ensemble is their equal-weight mean. Its threshold is the training calibration 99th percentile. Detector samples, sorted calibration scores, threshold, indices and seed are stored in `detector.npz`.

At least half the 10 m pixels in a 40 m cell must support a modality, and a scored cell needs at least two modalities. SAR gate quality also reflects the fraction of matched dates contributing observations. A displayed pixel must support every modality its cell used. Overlapping supported windows are averaged before thresholding. No extrapolation fills gaps. Per-quarter coverage distinguishes unobserved ground, covered but unscored pixels and scored pixels. Persistence divides the number of flagged quarters by the number of supported quarters and requires at least two observed quarters for a fraction.

## 4.7 — Evaluation and controls

`evaluate` reloads the trained variants for comparison on common validation cells with all three targets available before dropout. It measures each target separately and repeats inference with each sensor removed. Removing the sole available sensor yields an empty metric. The report includes fixed-target errors, rank and generalization gap versus a seeded sample of training locations. `robustness.csv` breaks errors down by quarter and 256-source-row strips. `detector_comparison.csv` compares flagged area and overlap with the ensemble for each scoring method. Flags in dense vegetation (NDVI > 0.65) and water-like spectra (NDWI > 0.30 and NDVI < 0.10) quantify sensitivity to observable nuisance conditions.

For independently sourced validation labels, pass `evaluate --labels controls.geojson` with a fresh report directory. Use a WGS84 GeoJSON FeatureCollection with Polygon or MultiPolygon features. Required properties are `id`, `period` (such as `2024Q2`), `split` (`val`), `source` and binary `label` (0 for a verified control and 1 for a known positive). Each polygon-quarter receives its maximum supported score. Unscored polygons are counted separately. Both label classes enable average precision and ROC AUC. Negative controls enable polygon-level false-positive rate, and positives enable sensitivity at the training-calibrated threshold. Larger polygons have more opportunities to contain a high score. Use comparably sized controls and report their provenance.

The 20 original validation patches form one overlap-connected region. Latitude strips and quarters describe variation within it. They cannot establish independent geographic replication. The main comparison measures feature prediction; an improvement on that task does not establish archaeological detection skill.

## 4.8 — GIS products

| File in the maps directory | Interpretation |
|---|---|
| YYYYQn_anomalies.tif | Ensemble, five percentile scores, three fusion weights and three sensor-quality bands |
| YYYYQn_coverage.tif | Number of covering windows, number of scoring windows and maximum supported modality count |
| candidates.geojson | Connected threshold exceedances by quarter, with area, mean/peak score, average weights, quality and checkpoint SHA-256 |
| persistence.tif | Flagged fraction, flagged quarter count and observed quarter count |
| validation_embeddings.parquet | Mean fused vector over supported cells for each supported validation patch-quarter |
| provenance.json | Settings, upstream hashes, detector lineage, calibration sample sizes, coverage and proxy-control results |

GeoTIFFs are tiled and losslessly compressed with explicit per-band nodata. Candidate geometry follows connected components on the 10 m display grid so offset edge windows land at their true coordinates. Spatial scoring uses 40 m cells. For gated and attention models the weights are learned mixing coefficients. Concatenation reports the available-modality shares. These quantities describe how the fusion calculation combined inputs; they do not identify the physical cause of a candidate.
