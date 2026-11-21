
"""SARchaeology AI — Phase 3.1 optical dataset.

Reads Phase 1 NPZ archives without loading SAR arrays.
One sample = one valid quarterly optical composite.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class OpticalDataset(Dataset):
    def __init__(
        self,
        root,
        split="train",
        statistics=None,
        min_valid_fraction=0.50,
        clip=5.0,
    ):
        if split not in ("train", "val", "test"):
            raise ValueError("Invalid split")

        self.root = Path(root)
        self.split = split
        self.statistics = statistics
        self.clip = clip
        self.min_valid_fraction = min_valid_fraction

        manifest = (
            self.root
            / "data/metadata/manifests/splits"
            / "buto_demo_2024"
            / f"{split}.parquet"
        )

        self.manifest_path = manifest
        self.manifest_hash = sha256(manifest)

        if statistics is not None:
            if tuple(statistics["bands"]) != BANDS:
                raise ValueError("Optical band order mismatch")

            if statistics["fit_split"] != "train":
                raise ValueError("Statistics must be train-fitted")

        frame = pd.read_parquet(manifest)
        self.samples = []

        for record in frame.itertuples(index=False):
            patch_path = self.root / record.path

            with np.load(
                patch_path, allow_pickle=False
            ) as archive:
                optical = archive["optical"]
                validity = archive["optical_valid"]

                if (
                    optical.ndim != 4
                    or optical.shape[1:] != (6, 128, 128)
                ):
                    raise ValueError(
                        f"Invalid optical shape: {patch_path}"
                    )

                if validity.shape != (
                    optical.shape[0], 128, 128
                ):
                    raise ValueError(
                        f"Invalid optical mask: {patch_path}"
                    )

                for quarter in range(optical.shape[0]):
                    finite = np.isfinite(
                        optical[quarter]
                    ).all(axis=0)

                    valid = (
                        (validity[quarter] > 0)
                        & finite
                    )

                    fraction = float(valid.mean())

                    if fraction >= min_valid_fraction:
                        self.samples.append({
                            "patch_id": str(record.patch_id),
                            "path": str(record.path),
                            "quarter_index": quarter,
                            "valid_fraction": fraction,
                            "row": int(record.row),
                            "col": int(record.col),
                        })

        if not self.samples:
            raise RuntimeError(
                f"No usable optical samples in {split}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = self.samples[index]
        path = self.root / item["path"]
        q = item["quarter_index"]

        with np.load(path, allow_pickle=False) as data:
            image = np.asarray(
                data["optical"][q], dtype=np.float32
            ).copy()

            source_valid = np.asarray(
                data["optical_valid"][q]
            ) > 0

        valid = (
            source_valid
            & np.isfinite(image).all(axis=0)
        )

        if self.statistics is not None:
            center = np.asarray(
                self.statistics["median"],
                dtype=np.float32
            )[:, None, None]

            scale = np.asarray(
                self.statistics["scale"],
                dtype=np.float32
            )[:, None, None]

            image = (image - center) / scale

            image = np.clip(
                image, -self.clip, self.clip
            )

        # Preserve zeros only as missing-data placeholders.
        # The mask remains authoritative.
        image = np.where(
            valid[None],
            image,
            0.0,
        ).astype(np.float32)

        return {
            "image": torch.from_numpy(image),
            "valid_mask": torch.from_numpy(
                valid[None].copy()
            ),
            "patch_id": item["patch_id"],
            "quarter_index": q,
            "row": item["row"],
            "col": item["col"],
            "valid_fraction": item["valid_fraction"],
        }


def fit_optical_statistics(
    dataset,
    *,
    pixels_per_sample=256,
    seed=42,
):
    """Fit approximate robust statistics on TRAIN data.

    Samples valid pixels per observation to bound RAM.
    """
    if dataset.split != "train":
        raise ValueError(
            "Normalization must use training split only"
        )

    if dataset.statistics is not None:
        raise ValueError(
            "Fit on raw reflectance, not normalized input"
        )

    rng = np.random.default_rng(seed)
    band_samples = [[] for _ in BANDS]

    for i in range(len(dataset)):
        sample = dataset[i]
        image = sample["image"].numpy()
        valid = sample["valid_mask"].numpy()[0]

        indices = np.flatnonzero(valid)

        if not len(indices):
            continue

        chosen = rng.choice(
            indices,
            size=min(len(indices), pixels_per_sample),
            replace=False,
        )

        flattened = image.reshape(6, -1)

        for band_index in range(6):
            band_samples[band_index].append(
                flattened[band_index, chosen]
            )

    median, scale, p01, p99 = [], [], [], []

    for samples in band_samples:
        if not samples:
            raise RuntimeError(
                "Insufficient valid optical pixels"
            )

        values = np.concatenate(samples).astype(
            np.float64
        )

        q01, q25, q50, q75, q99 = np.percentile(
            values, [1, 25, 50, 75, 99]
        )

        median.append(float(q50))
        scale.append(
            float(max(q75 - q25, 1e-6))
        )
        p01.append(float(q01))
        p99.append(float(q99))

    return {
        "dataset_id": "buto_demo_2024",
        "fit_split": "train",
        "train_manifest_sha256": dataset.manifest_hash,
        "bands": list(BANDS),
        "method": "sampled_median_iqr",
        "pixels_per_sample": pixels_per_sample,
        "seed": seed,
        "median": median,
        "scale": scale,
        "p01": p01,
        "p99": p99,
        "min_valid_fraction": dataset.min_valid_fraction,
    }


def create_optical_dataloaders(
    root,
    statistics,
    *,
    batch_size=16,
    num_workers=2,
    min_valid_fraction=None,
    include_test=False,
):
    if min_valid_fraction is None:
        min_valid_fraction = statistics[
            "min_valid_fraction"
        ]

    datasets = {}
    loaders = {}

    splits = (
        ("train", "val", "test")
        if include_test
        else ("train", "val")
    )

    for split in splits:
        dataset = OpticalDataset(
            root,
            split=split,
            statistics=statistics,
            min_valid_fraction=min_valid_fraction,
        )

        if (
            split == "train"
            and dataset.manifest_hash
            != statistics["train_manifest_sha256"]
        ):
            raise RuntimeError(
                "Training manifest fingerprint mismatch"
            )

        options = {
            "batch_size": batch_size,
            "shuffle": split == "train",
            "num_workers": num_workers,
            "pin_memory": torch.cuda.is_available(),
            "drop_last": False,
        }

        if num_workers > 0:
            options["persistent_workers"] = True
            options["prefetch_factor"] = 2

        datasets[split] = dataset
        loaders[split] = DataLoader(
            dataset, **options
        )

    return datasets, loaders
