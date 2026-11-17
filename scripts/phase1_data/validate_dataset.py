
import argparse
import json
import sys

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sarchaeology.evaluation.dataset_quality import (
    audit_dataset,
)


def load_yaml(path):
    with (ROOT / path).open(
        "r", encoding="utf-8"
    ) as f:
        return yaml.safe_load(f)


def render_data_card(report, metadata, split_report):
    """
    Produce a human-readable Markdown data card
    from the actual completed QA results.
    """

    counts = report["split_counts"]

    sar_sources = metadata.get("sar_sources", [])
    optical_sources = metadata.get("optical_sources", [])

    sar_dates = sorted({
        str(s["date"])
        for s in sar_sources
    })

    optical_quarters = sorted({
        str(s["quarter"])
        for s in optical_sources
    })

    errors = report["errors"]
    warnings = report["warnings"]

    errors_markdown = (
        "\n".join(f"- {e}" for e in errors)
        if errors
        else "- None"
    )

    warnings_markdown = (
        "\n".join(f"- {w}" for w in warnings)
        if warnings
        else "- None"
    )

    separation = report["spatial_separation_m"]

    separation_rows = "\n".join(
        f"| {name} | {distance:.1f} m |"
        for name, distance in separation.items()
    )

    valid_mean = report[
        "joint_valid_fraction"
    ]["mean"]

    valid_mean_text = (
        f"{valid_mean:.2%}"
        if valid_mean is not None
        else "Unavailable"
    )

    return f"""# SARchaeology AI — Dataset Card

## 1. Dataset identity

- Dataset: `{report['dataset']}`
- Dataset type: Unlabeled multimodal Earth observation patches
- Study area: Buto / Tell el-Fara'in, Egypt
- Study year: 2024
- Coordinate system: {report['crs']}
- Analysis resolution: 10 metres
- Patch size: {report['patch_size']} pixels
- Patch stride: {report['stride']} pixels
- Quality status: **{report['status']}**

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

- SAR acquisitions recorded: {len(sar_sources)}
- SAR dates: {sar_dates[0] if sar_dates else 'N/A'} to {sar_dates[-1] if sar_dates else 'N/A'}
- Optical quarterly composites: {', '.join(optical_quarters) if optical_quarters else 'N/A'}

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

Method: {split_report['method']}

Split axis: {split_report['axis']}

| Split | Patches |
|---|---:|
| Train | {counts['train']} |
| Validation | {counts['val']} |
| Test | {counts['test']} |
| Excluded | {counts['excluded']} |
| **Total** | **{report['total_patches']}** |

Geographic separation between complete
patch footprints:

| Split comparison | Separation |
|---|---:|
{separation_rows}

All time observations from one geographic patch
are assigned to the same split.

Excluded patches are not used during model
fitting or evaluation.

## 6. Quality validation

- QA status: **{report['status']}**
- Archives inspected: {report['patches_inspected']}
- Archives passing validation: {report['patches_passed']}
- Mean joint-valid patch coverage: {valid_mean_text}
- Partial scan: {report['partial_scan']}

### Errors

{errors_markdown}

### Warnings

{warnings_markdown}

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
"""


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--year",
        type=int,
        default=2024,
    )

    parser.add_argument(
        "--max-patches",
        type=int,
        default=0,
        help="0 audits all patch archives",
    )

    args = parser.parse_args()

    data_cfg = load_yaml("configs/data.yaml")
    quality_cfg = load_yaml("configs/quality.yaml")

    aoi_id = data_cfg["study_area"]["id"]
    dataset_id = f"{aoi_id}_{args.year}"

    print("\nSARchaeology AI — Step 1.10")
    print("Dataset:", dataset_id)
    print("Maximum patches:", args.max_patches or "ALL")

    split_dir = (
        ROOT
        / "data/metadata/manifests/splits"
        / dataset_id
    )

    split_report_path = (
        split_dir / "split_report.json"
    )

    if not split_report_path.exists():
        raise FileNotFoundError(
            "Step 1.9 split report not found. "
            "Run create_spatial_splits.py first."
        )

    metadata_path = (
        ROOT
        / "data/patches/unassigned"
        / dataset_id
        / "dataset_metadata.json"
    )

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    split_report = json.loads(
        split_report_path.read_text(
            encoding="utf-8"
        )
    )

    report = audit_dataset(
        root=ROOT,
        dataset_id=dataset_id,
        config=quality_cfg,
        max_patches=args.max_patches,
    )

    output_dir = (
        ROOT
        / quality_cfg["quality"]["output_directory"]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        output_dir / f"{dataset_id}_qa_report.json"
    )

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nQuality assessment")
    print("Status:", report["status"])

    print(
        "Archives validated:",
        report["patches_passed"],
        "/",
        report["patches_inspected"],
    )

    print("\nSplit counts:")
    for name, count in report["split_counts"].items():
        print(f"  {name}: {count}")

    print("\nWarnings:")
    for warning in report["warnings"]:
        print("  WARNING:", warning)

    print("\nErrors:")
    for error in report["errors"][:20]:
        print("  ERROR:", error)

    if len(report["errors"]) > 20:
        print(
            f"  ...and {len(report['errors']) - 20}"
            " additional errors"
        )

    print("\nReport:", report_path)

    # Never overwrite a completed data card using
    # a sample-only QA run.
    if args.max_patches == 0:
        card_path = (
            ROOT
            / quality_cfg["quality"]["data_card_path"]
        )

        card_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        card = render_data_card(
            report,
            metadata,
            split_report,
        )

        card_path.write_text(
            card,
            encoding="utf-8",
        )

        print("Dataset card:", card_path)

    if report["status"] == "FAIL":
        raise SystemExit(1)

    if report["status"] == "PARTIAL":
        print(
            "\nPartial validation only. "
            "The complete dataset is not yet certified."
        )

    else:
        print(
            "\nPhase 1 dataset validation passed."
        )


if __name__ == "__main__":
    main()
