
"""Build multi-year inventory and train-only statistics."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sarchaeology.data.optical_multiyear import (
    BANDS,
    MultiYearOpticalDataset,
)


def fit_statistics(dataset, seed=42):
    rng = np.random.default_rng(seed)

    values = [[] for _ in BANDS]

    for index in range(len(dataset)):
        sample = dataset[index]

        image = sample["image"].numpy()
        valid = sample["valid_mask"].numpy()[0]

        pixels = np.flatnonzero(valid)

        if not len(pixels):
            continue

        chosen = rng.choice(
            pixels,
            size=min(256, len(pixels)),
            replace=False,
        )

        flattened = image.reshape(6, -1)

        for band in range(6):
            values[band].append(
                flattened[band, chosen]
            )

    medians = []
    scales = []
    percentiles = {}

    for band, chunks in zip(BANDS, values):
        if not chunks:
            raise RuntimeError(
                f"No valid samples for {band}"
            )

        x = np.concatenate(chunks)

        p01, p25, p50, p75, p99 = (
            np.percentile(
                x,
                [1, 25, 50, 75, 99],
            )
        )

        medians.append(float(p50))
        scales.append(
            float(max(p75 - p25, 1e-6))
        )
        percentiles[band] = {
            "p01": float(p01),
            "p99": float(p99),
        }

    return {
        "dataset_id": "buto_optical_multiyear_v1",
        "fit_split": "train",
        "bands": list(BANDS),
        "method": "sampled_median_iqr",
        "sampled_pixels_per_observation": 256,
        "seed": seed,
        "median": medians,
        "scale": scales,
        "percentiles": percentiles,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2022, 2023, 2024, 2025],
    )

    parser.add_argument(
        "--min-valid-fraction",
        type=float,
        default=0.50,
    )

    args = parser.parse_args()

    train = MultiYearOpticalDataset(
        ROOT,
        years=args.years,
        split="train",
        min_valid_fraction=args.min_valid_fraction,
    )

    val = MultiYearOpticalDataset(
        ROOT,
        years=args.years,
        split="val",
        min_valid_fraction=args.min_valid_fraction,
    )

    print("Available training periods:", train.periods)
    print("Training observations:", len(train))
    print("Validation observations:", len(val))

    stats = fit_statistics(train)

    stats.update({
        "training_years": sorted(set(args.years)),
        "available_periods": train.periods,
        "min_valid_fraction": args.min_valid_fraction,
        "training_observations": len(train),
        "validation_observations": len(val),
    })

    stats_path = (
        ROOT / "outputs/statistics"
        / "optical_buto_multiyear_v1.json"
    )
    stats_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    if stats_path.exists():
        raise FileExistsError(
            f"Refusing to replace: {stats_path}"
        )

    stats_path.write_text(
        json.dumps(stats, indent=2),
        encoding="utf-8",
    )

    manifest_dir = (
        ROOT / "data/metadata/manifests"
        / "optical_multiyear_v1"
    )
    manifest_dir.mkdir(
        parents=True, exist_ok=True
    )

    for split, dataset in (
        ("train", train),
        ("val", val),
    ):
        path = manifest_dir / f"{split}.parquet"

        if path.exists():
            raise FileExistsError(
                f"Refusing to replace: {path}"
            )

        pd.DataFrame(dataset.samples).to_parquet(
            path, index=False
        )

    # A quick normalized batch verification.
    normalized_train = MultiYearOpticalDataset(
        ROOT,
        years=args.years,
        split="train",
        min_valid_fraction=args.min_valid_fraction,
        statistics=stats,
    )

    loader = DataLoader(
        normalized_train,
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(loader))

    assert batch["image"].shape[1:] == (
        6, 128, 128
    )
    assert torch.isfinite(
        batch["image"]
    ).all()

    print(
        "Normalized batch:",
        tuple(batch["image"].shape),
    )

    print("Saved normalization:", stats_path)
    print("Saved manifests:", manifest_dir)
    print("\nMulti-year dataset preparation passed.")


if __name__ == "__main__":
    main()
