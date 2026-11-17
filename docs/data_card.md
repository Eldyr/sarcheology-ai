# SARchaeology AI — Dataset Card

## 1. Dataset identity

- Dataset: `buto_demo_2024`
- Dataset type: Unlabeled multimodal Earth observation patches
- Study area: Buto / Tell el-Fara'in, Egypt
- Study year: 2024
- Coordinate system: EPSG:32636
- Analysis resolution: 10 metres
- Patch size: 128 pixels
- Patch stride: 64 pixels
- Quality status: **PASS_WITH_WARNINGS**

## 2. Purpose

This dataset supports exploratory, unsupervised
landscape anomaly detection for archaeological
prospection research.

It does not contain verified labels indicating
buried structures, archaeological remains,
or confirmed archaeological discoveries.

## 3. Data sources

| Source | Product | Role |
|---|---|---|
| Copernicus Sentinel-1 | RTC SAR | Radar backscatter |
| Copernicus Sentinel-2 | Level-2A | Multispectral reflectance |
| Copernicus DEM | GLO-30 | Terrain context |

### Temporal coverage

- SAR acquisitions recorded: 29
- SAR dates: 2024-01-10 to 2024-12-23
- Optical quarterly composites: 2024Q1, 2024Q2, 2024Q3, 2024Q4

Source acquisition identifiers and local input
paths are recorded in the patch dataset metadata.

## 4. Dataset structure

| Tensor | Shape |
|---|---|
| Engineered features | [19, H, W] |
| Modality validity masks | [4, H, W] |
| Observation quality | [3, H, W] |
| SAR time series | [N, 2, H, W] |
| SAR validity | [N, H, W] |
| Optical quarters | [Q, 6, H, W] |
| Optical validity | [Q, H, W] |

N and Q correspond to the available acquisition
and quarterly composite counts.

### Engineered features

- SAR: 7 temporal descriptors.
- Optical: 7 annual spectral descriptors.
- Terrain: 5 elevation-derived descriptors.

Data remain in their original scientific
units. Model normalization has not yet
been fitted.

## 5. Geographic splitting

Method: geographic_strips_with_buffer

Split axis: x

| Split | Patches |
|---|---:|
| Train | 240 |
| Validation | 20 |
| Test | 34 |
| Excluded | 123 |
| **Total** | **417** |

Geographic separation between complete
patch footprints:

| Split comparison | Separation |
|---|---:|
| train_to_val | 1280.0 m |
| val_to_test | 1280.0 m |
| train_to_test | 3840.0 m |

All time observations from one geographic patch
are assigned to the same split.

Excluded patches are not used during model
fitting or evaluation.

## 6. Quality validation

- QA status: **PASS_WITH_WARNINGS**
- Archives inspected: 417
- Archives passing validation: 417
- Mean joint-valid patch coverage: 98.75%
- Partial scan: False

### Errors

- None

### Warnings

- Validation set is relatively small
- A substantial fraction of patches was excluded

## 7. Processing methodology

1. Discover Sentinel-1, Sentinel-2 and DEM assets.
2. Retrieve and prepare the satellite measurements.
3. Create temporally aggregated SAR and optical features.
4. Calculate elevation-derived terrain descriptors.
5. Align modalities to a common projected grid.
6. Extract overlapping, unlabeled geographic patches.
7. Construct geographically separated data splits.
8. Validate archives, georeferencing and provenance.

The dataset is designed for unsupervised and
self-supervised modeling.

## 8. Known limitations

- The dataset has no verified archaeological
  ground-truth labels.
- Optical imagery is affected by clouds,
  atmospheric correction and seasonal vegetation.
- SAR observations may contain speckle and
  viewing-geometry-related artifacts.
- The Copernicus DEM is a digital surface model
  with approximately 30 m native resolution;
  alignment to 10 m does not add detail.
- Agricultural operations, irrigation, modern
  construction and natural terrain variation
  can resemble archaeological anomalies.
- Overlapping patches and neighboring terrain
  may remain statistically correlated despite
  geographic split exclusion buffers.
- Pixel-level integrity checks do not independently
  establish absolute geolocation accuracy.

## 9. Recommended model use

Use training patches to fit model parameters,
anomaly detectors and normalization statistics.

Use validation patches for model selection.

Use test patches only after model selection
and threshold decisions are finalized.

Keep SAR and optical reconstruction models
separate from the 19-channel engineered-feature
fusion dataset.

The 128-pixel patches may require a documented
input adapter for pretrained encoders expecting
other spatial dimensions.

## 10. Interpretation and verification

Anomaly scores should be treated as research
candidates rather than confirmed archaeology.

Claims about buried structures require independent
evidence such as archaeological field survey,
historical records, ground-penetrating radar,
electrical resistivity tomography, or excavation
conducted under appropriate permissions.
