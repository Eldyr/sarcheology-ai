
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


CHANNEL_NAMES = ("VV", "VH")


def file_sha256(path):
    """Compute fingerprint of a source manifest."""
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def fit_sar_statistics(
    project_root,
    dataset_id,
    *,
    pixels_per_acquisition=128,
    min_valid_fraction=0.85,
    min_scale=0.01,
    seed=42,
    clip_min=-5.0,
    clip_max=5.0,
):
    """
    Fit per-channel robust statistics using ONLY
    the train.parquet geographic split.

    Each eligible SAR acquisition contributes
    at most pixels_per_acquisition paired pixels.

    No validation/test data are read.
    """
    root = Path(project_root).resolve()

    manifest_path = (
        root
        / "data/metadata/manifests/splits"
        / dataset_id
        / "train.parquet"
    )

    metadata_path = (
        root
        / "data/patches/unassigned"
        / dataset_id
        / "dataset_metadata.json"
    )

    if pixels_per_acquisition < 1:
        raise ValueError(
            "pixels_per_acquisition must be >= 1"
        )

    if not 0 <= min_valid_fraction <= 1:
        raise ValueError(
            "min_valid_fraction must be between 0 and 1"
        )

    if min_scale <= 0:
        raise ValueError("min_scale must be positive")

    if clip_min >= clip_max:
        raise ValueError("Invalid clipping bounds")

    manifest = pd.read_parquet(manifest_path)

    metadata = json.loads(
        metadata_path.read_text(
            encoding="utf-8"
        )
    )

    if manifest.empty:
        raise ValueError("Training manifest is empty")

    if not (
        manifest["split"] == "train"
    ).all():
        raise ValueError(
            "Normalization input contains "
            "non-training patches"
        )

    if not manifest["patch_id"].is_unique:
        raise ValueError(
            "Duplicate training patch IDs"
        )

    n_acquisitions = len(
        metadata["sar_sources"]
    )

    patch_size = int(
        metadata["patch_size"]
    )

    # Sorting makes results independent of
    # the order of rows in the manifest.
    manifest = manifest.sort_values(
        "patch_id"
    )

    rng = np.random.default_rng(seed)

    samples = [[], []]

    eligible_observations = 0
    total_observations = 0

    for record in manifest.itertuples(
        index=False
    ):
        path = Path(record.path)

        if not path.is_absolute():
            path = root / path

        with np.load(
            path,
            allow_pickle=False,
        ) as archive:
            sar = archive["sar"].astype(
                np.float32
            )
            validity = archive[
                "sar_valid"
            ].astype(bool)

        if sar.shape != (
            n_acquisitions,
            2,
            patch_size,
            patch_size,
        ):
            raise ValueError(
                f"Unexpected SAR shape: {path}"
            )

        if validity.shape != (
            n_acquisitions,
            patch_size,
            patch_size,
        ):
            raise ValueError(
                f"Unexpected SAR mask shape: {path}"
            )

        for index in range(n_acquisitions):
            total_observations += 1

            valid = validity[index]

            if valid.mean() < min_valid_fraction:
                continue

            image = sar[index]

            # Never sample a missing measurement.
            if not np.isfinite(
                image[:, valid]
            ).all():
                raise ValueError(
                    f"Non-finite valid pixels: {path}"
                )

            eligible_observations += 1

            candidates = np.flatnonzero(
                valid.reshape(-1)
            )

            count = min(
                pixels_per_acquisition,
                len(candidates),
            )

            chosen = rng.choice(
                candidates,
                size=count,
                replace=False,
            )

            flat = image.reshape(
                2, -1
            )

            for channel in range(2):
                samples[channel].append(
                    flat[channel, chosen].copy()
                )

    if eligible_observations == 0:
        raise ValueError(
            "No eligible training SAR observations"
        )

    medians = []
    q25s = []
    q75s = []
    iqrs = []
    scales = []
    sample_counts = []

    for channel in range(2):
        values = np.concatenate(
            samples[channel]
        ).astype(np.float64)

        q25, median, q75 = np.percentile(
            values,
            [25, 50, 75],
        )

        iqr = float(q75 - q25)

        # Normal-equivalent robust scale;
        # minimum protects near-constant data.
        scale = max(
            iqr / 1.349,
            min_scale,
        )

        q25s.append(float(q25))
        medians.append(float(median))
        q75s.append(float(q75))
        iqrs.append(iqr)
        scales.append(float(scale))
        sample_counts.append(
            int(len(values))
        )

    return {
        "schema_version": 1,
        "method": "robust_iqr",
        "dataset_id": dataset_id,
        "fit_split": "train",
        "source_manifest": (
            manifest_path.relative_to(root).as_posix()
        ),
        "source_manifest_sha256": file_sha256(
            manifest_path
        ),
        "channel_names": list(CHANNEL_NAMES),
        "median_db": medians,
        "q25_db": q25s,
        "q75_db": q75s,
        "iqr_db": iqrs,
        "scale_db": scales,
        "sample_counts": sample_counts,
        "eligible_observations": (
            eligible_observations
        ),
        "total_observations": total_observations,
        "geographic_train_patches": (
            len(manifest)
        ),
        "pixels_per_acquisition": (
            pixels_per_acquisition
        ),
        "min_valid_fraction": (
            min_valid_fraction
        ),
        "min_scale": min_scale,
        "seed": seed,
        "clip_min": clip_min,
        "clip_max": clip_max,
    }


class SARNormalizer:
    """
    Apply fixed, train-derived robust normalization.

    Supports image [2,H,W] and mask [1,H,W].
    Invalid pixels remain exactly zero.
    """

    def __init__(self, statistics):
        if statistics.get("method") != "robust_iqr":
            raise ValueError(
                "Unsupported SAR normalization method"
            )

        if statistics.get("fit_split") != "train":
            raise ValueError(
                "Normalization must originate "
                "from the training split"
            )

        if statistics.get("channel_names") != [
            "VV", "VH"
        ]:
            raise ValueError(
                "Unexpected normalization band order"
            )

        self.statistics = statistics

        self.median = torch.tensor(
            statistics["median_db"],
            dtype=torch.float32,
        ).view(2, 1, 1)

        self.scale = torch.tensor(
            statistics["scale_db"],
            dtype=torch.float32,
        ).view(2, 1, 1)

        if not torch.isfinite(
            self.median
        ).all():
            raise ValueError(
                "Non-finite normalization median"
            )

        if not (
            torch.isfinite(self.scale).all()
            and (self.scale > 0).all()
        ):
            raise ValueError(
                "Invalid normalization scale"
            )

        self.clip_min = float(
            statistics["clip_min"]
        )

        self.clip_max = float(
            statistics["clip_max"]
        )

    @classmethod
    def from_json(cls, path):
        data = json.loads(
            Path(path).read_text(
                encoding="utf-8"
            )
        )
        return cls(data)

    def __call__(self, image, valid_mask):
        """
        image: [2,H,W] float tensor, dB
        valid_mask: [1,H,W] float/bool tensor
        """
        if (
            image.ndim != 3
            or image.shape[0] != 2
            or valid_mask.shape != (
                1,
                image.shape[1],
                image.shape[2],
            )
        ):
            raise ValueError(
                "Invalid SAR image/mask dimensions"
            )

        valid = valid_mask > 0

        if not torch.isfinite(
            image[:, valid[0]]
        ).all():
            raise ValueError(
                "Non-finite data inside valid mask"
            )

        normalized = (
            image - self.median.to(image.device)
        ) / self.scale.to(image.device)

        normalized = normalized.clamp(
            self.clip_min,
            self.clip_max,
        )

        # The mask, rather than a data value,
        # determines whether a pixel is valid.
        normalized = torch.where(
            valid.expand_as(normalized),
            normalized,
            torch.zeros_like(normalized),
        )

        return normalized
