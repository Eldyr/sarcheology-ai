
"""Phase 3.1: Sentinel-2 reflectance and quarter audit.

Non-destructive: never rescales or rewrites source data.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sarchaeology.data.optical_dataset import BANDS


def audit_split(split, max_patches, seed):
    manifest = (
        ROOT / "data/metadata/manifests/splits"
        / "buto_demo_2024" / f"{split}.parquet"
    )
    frame = pd.read_parquet(manifest)
    selected = frame.head(max_patches)

    rng = np.random.default_rng(seed)
    samples = [[] for _ in BANDS]
    coverage = {}
    quarter_counts = {}
    warnings = []

    for row in selected.itertuples(index=False):
        path = ROOT / row.path

        with np.load(path, allow_pickle=False) as archive:
            optical = archive["optical"]
            valid = archive["optical_valid"]

        if optical.ndim != 4 or optical.shape[1:] != (
            6, 128, 128
        ):
            raise ValueError(
                f"Invalid optical layout in {path}: "
                f"{optical.shape}"
            )

        if valid.shape != (
            optical.shape[0], 128, 128
        ):
            raise ValueError(
                f"Invalid optical validity shape: {path}"
            )

        for q in range(len(optical)):
            key = str(q)

            quarter_counts[key] = (
                quarter_counts.get(key, 0) + 1
            )

            all_finite = np.isfinite(
                optical[q]
            ).all(axis=0)

            usable = (valid[q] > 0) & all_finite
            fraction = float(usable.mean())

            coverage.setdefault(key, []).append(fraction)

            indices = np.flatnonzero(usable)
            if not len(indices):
                continue

            indices = rng.choice(
                indices,
                size=min(512, len(indices)),
                replace=False,
            )

            flat = optical[q].reshape(6, -1)

            for band in range(6):
                samples[band].append(
                    flat[band, indices].astype(np.float32)
                )

    band_stats = {}

    for band, arrays in zip(BANDS, samples):
        if not arrays:
            band_stats[band] = {
                "status": "NO_VALID_SAMPLES"
            }
            continue

        x = np.concatenate(arrays)

        result = {
            "p01": float(np.percentile(x, 1)),
            "median": float(np.median(x)),
            "p99": float(np.percentile(x, 99)),
            "fraction_negative": float(
                np.mean(x < 0)
            ),
            "fraction_above_1": float(
                np.mean(x > 1)
            ),
            "fraction_above_2": float(
                np.mean(x > 2)
            ),
        }

        if result["p99"] > 100:
            warnings.append(
                f"{band}: large values suggest "
                "potential unscaled reflectance"
            )
        elif result["p99"] > 2:
            warnings.append(
                f"{band}: verify reflectance scale "
                "and residual atmospheric artifacts"
            )

        band_stats[band] = result

    return {
        "split": split,
        "manifest_patches": len(frame),
        "inspected_patches": len(selected),
        "quarter_counts": quarter_counts,
        "mean_valid_fraction_by_quarter": {
            k: float(np.mean(v))
            for k, v in coverage.items()
        },
        "band_statistics": band_stats,
        "warnings": warnings,
    }


def audit_quarter_metadata():
    """
    Discover possible quarter metadata.
    Filename discovery is NOT proof of temporal mapping.
    """
    composite_dir = (
        ROOT / "data/processed/optical/composites"
    )

    composite_files = (
        sorted(composite_dir.rglob("*.tif"))
        if composite_dir.exists()
        else []
    )

    metadata_dir = ROOT / "data/metadata"
    metadata_files = []

    if metadata_dir.exists():
        for pattern in (
            "*optical*.parquet",
            "*optical*.json",
            "*composite*.json",
            "*composite*.parquet",
        ):
            metadata_files.extend(
                metadata_dir.rglob(pattern)
            )

    return {
        "composite_files": [
            str(p.relative_to(ROOT))
            for p in composite_files
        ],
        "candidate_metadata_files": sorted({
            str(p.relative_to(ROOT))
            for p in metadata_files
        }),
        "quarter_index_to_date_mapping_verified": False,
        "note": (
            "A filename inventory alone cannot prove "
            "the temporal ordering saved in each NPZ. "
            "Verify the source manifest or tiling code."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-patches",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    if args.max_patches < 1:
        parser.error("--max-patches must be positive")

    report = {
        "phase": "3.1",
        "dataset": "buto_demo_2024",
        "band_order": list(BANDS),
        "splits": [
            audit_split(split, args.max_patches, args.seed)
            for split in ("train", "val")
        ],
        "temporal_metadata": audit_quarter_metadata(),
    }

    output = (
        ROOT / "outputs/quality"
        / "phase3_optical_input_audit.json"
    )
    output.parent.mkdir(
        parents=True, exist_ok=True
    )

    output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))
    print("\nSaved:", output)


if __name__ == "__main__":
    main()
