
"""SARchaeology AI — Phase 3.3.2 Multi-Year Optical Dataset.

Loads six-band Sentinel-2 quarterly composites using the
original geographic patch manifests from Phase 1.

Features:
- Multi-year optical training data (2022–2025)
- Original geographic train/validation split
- Six Sentinel-2 bands
- Valid-pixel and source-observation-count masks
- Train-only robust normalization
- Year/quarter metadata
- On-demand GeoTIFF window reading
- Compatibility with PyTorch DataLoader

Expected source files:
    data/processed/optical/composites/
        YYYYQn_median.tif
        YYYYQn_valid_count.tif

Expected training sample:
    image:       [6, 128, 128] float32
    valid_mask:  [1, 128, 128] bool

The original Phase 1 files are never modified.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch

from rasterio.windows import Window
from torch.utils.data import Dataset


BANDS = (
    "B02",
    "B03",
    "B04",
    "B08",
    "B11",
    "B12",
)

PATCH_SIZE = 128


class MultiYearOpticalDataset(Dataset):
    """Multi-year Sentinel-2 optical patch dataset."""

    def __init__(
        self,
        root,
        *,
        years=(2022, 2023, 2024, 2025),
        split="train",
        min_valid_fraction=0.50,
        min_source_observations=1,
        statistics=None,
        clip=5.0,
    ):
        super().__init__()

        self.root = Path(root).resolve()
        self.years = tuple(sorted(set(years)))
        self.split = split

        self.min_valid_fraction = float(
            min_valid_fraction
        )
        self.min_source_observations = int(
            min_source_observations
        )

        self.statistics = statistics
        self.clip = float(clip)

        if split not in ("train", "val", "test"):
            raise ValueError(
                "Split must be 'train', 'val', or 'test'"
            )

        if not self.years:
            raise ValueError(
                "At least one training year is required"
            )

        if not 0 <= self.min_valid_fraction <= 1:
            raise ValueError(
                "min_valid_fraction must be between 0 and 1"
            )

        if self.min_source_observations < 1:
            raise ValueError(
                "min_source_observations must be >= 1"
            )

        if self.clip <= 0:
            raise ValueError(
                "clip must be positive"
            )

        if statistics is not None:
            self._validate_statistics(statistics)

        self.manifest_path = (
            self.root
            / "data"
            / "metadata"
            / "manifests"
            / "splits"
            / "buto_demo_2024"
            / f"{split}.parquet"
        )

        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Missing geographic split manifest: "
                f"{self.manifest_path}"
            )

        frame = pd.read_parquet(
            self.manifest_path
        )

        required_columns = {
            "patch_id",
            "row",
            "col",
        }

        missing = (
            required_columns - set(frame.columns)
        )

        if missing:
            raise ValueError(
                f"Missing geographic manifest columns: "
                f"{sorted(missing)}"
            )

        self.composite_dir = (
            self.root
            / "data"
            / "processed"
            / "optical"
            / "composites"
        )

        reference_path = (
            self.composite_dir
            / "2024Q1_median.tif"
        )

        if not reference_path.exists():
            raise FileNotFoundError(
                f"Reference raster not found: "
                f"{reference_path}"
            )

        with rasterio.open(reference_path) as reference:
            self.reference_grid = (
                reference.crs,
                reference.transform,
                reference.width,
                reference.height,
            )

            if reference.count != len(BANDS):
                raise ValueError(
                    "Reference optical raster must "
                    "contain six spectral bands"
                )

        # Preserve the 2024 validation and independent test protocols.
        if split in ("val", "test"):
            selected_years = (2024,)
        else:
            selected_years = self.years

        self.samples = []
        self.periods = []

        for year in selected_years:
            for quarter in range(1, 5):
                label = f"{year}Q{quarter}"

                median_path = (
                    self.composite_dir
                    / f"{label}_median.tif"
                )

                count_path = (
                    self.composite_dir
                    / f"{label}_valid_count.tif"
                )

                median_exists = median_path.exists()
                count_exists = count_path.exists()

                if median_exists != count_exists:
                    raise RuntimeError(
                        f"{label}: incomplete composite pair"
                    )

                if not median_exists:
                    if split in ("val", "test"):
                        raise FileNotFoundError(
                            f"Missing {split} quarter: {label}"
                        )

                    print(
                        f"Skipping unavailable period: "
                        f"{label}"
                    )
                    continue

                self._register_period(
                    label=label,
                    year=year,
                    quarter=quarter,
                    median_path=median_path,
                    count_path=count_path,
                    frame=frame,
                )

        if not self.samples:
            raise RuntimeError(
                f"No usable optical samples in "
                f"split={split}"
            )

        self.samples.sort(
            key=lambda item: (
                item["year"],
                item["quarter"],
                item["patch_id"],
            )
        )

    def _validate_statistics(self, statistics):
        if tuple(statistics["bands"]) != BANDS:
            raise ValueError(
                "Optical normalization band order mismatch"
            )

        if statistics["fit_split"] != "train":
            raise ValueError(
                "Statistics must be fitted on training data"
            )

        median = np.asarray(
            statistics["median"],
            dtype=np.float32,
        )

        scale = np.asarray(
            statistics["scale"],
            dtype=np.float32,
        )

        if median.shape != (6,):
            raise ValueError(
                "Expected six normalization medians"
            )

        if scale.shape != (6,):
            raise ValueError(
                "Expected six normalization scales"
            )

        if not np.isfinite(median).all():
            raise ValueError(
                "Non-finite normalization median"
            )

        if not np.isfinite(scale).all():
            raise ValueError(
                "Non-finite normalization scale"
            )

        if np.any(scale <= 0):
            raise ValueError(
                "Normalization scales must be positive"
            )

    def _validate_raster(self, dataset, expected_bands):
        current_grid = (
            dataset.crs,
            dataset.transform,
            dataset.width,
            dataset.height,
        )

        if current_grid != self.reference_grid:
            raise RuntimeError(
                f"Geographic grid mismatch: "
                f"{dataset.name}"
            )

        if dataset.count != expected_bands:
            raise RuntimeError(
                f"Expected {expected_bands} bands in "
                f"{dataset.name}, got {dataset.count}"
            )

    def _valid_pixels(self, image, source_count):
        """Find pixels suitable for optical training.

        image:
            MaskedArray [6, 128, 128]

        source_count:
            MaskedArray [128, 128]
        """
        reflectance_valid = (
            ~np.ma.getmaskarray(image)
        ).all(axis=0)

        finite = np.isfinite(
            image.data
        ).all(axis=0)

        count_valid = (
            ~np.ma.getmaskarray(source_count)
        )

        enough_observations = (
            source_count.data
            >= self.min_source_observations
        )

        return (
            reflectance_valid
            & finite
            & count_valid
            & enough_observations
        )

    def _register_period(
        self,
        *,
        label,
        year,
        quarter,
        median_path,
        count_path,
        frame,
    ):
        """Validate a quarter and index its usable patches."""

        accepted = 0

        with (
            rasterio.open(median_path) as reflectance,
            rasterio.open(count_path) as counts,
        ):
            self._validate_raster(
                reflectance,
                expected_bands=6,
            )

            self._validate_raster(
                counts,
                expected_bands=1,
            )

            for record in frame.itertuples(index=False):
                row = int(record.row)
                col = int(record.col)

                if (
                    row < 0
                    or col < 0
                    or row + PATCH_SIZE > reflectance.height
                    or col + PATCH_SIZE > reflectance.width
                ):
                    raise RuntimeError(
                        f"Invalid geographic patch window: "
                        f"{record.patch_id}"
                    )

                window = Window(
                    col,
                    row,
                    PATCH_SIZE,
                    PATCH_SIZE,
                )

                image = reflectance.read(
                    window=window,
                    masked=True,
                )

                source_count = counts.read(
                    1,
                    window=window,
                    masked=True,
                )

                valid = self._valid_pixels(
                    image,
                    source_count,
                )

                fraction = float(
                    valid.mean()
                )

                if fraction < self.min_valid_fraction:
                    continue

                self.samples.append({
                    "patch_id": str(record.patch_id),
                    "row": row,
                    "col": col,
                    "year": int(year),
                    "quarter": int(quarter),
                    "period": label,
                    "median_path": str(median_path),
                    "count_path": str(count_path),
                    "valid_fraction": fraction,
                })

                accepted += 1

        self.periods.append(label)

        print(
            f"{self.split} | {label}: "
            f"{accepted} usable optical observations"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        window = Window(
            sample["col"],
            sample["row"],
            PATCH_SIZE,
            PATCH_SIZE,
        )

        with (
            rasterio.open(sample["median_path"]) as reflectance,
            rasterio.open(sample["count_path"]) as counts,
        ):
            image = reflectance.read(
                window=window,
                masked=True,
            )

            source_count = counts.read(
                1,
                window=window,
                masked=True,
            )

        valid = self._valid_pixels(
            image,
            source_count,
        )

        image = np.asarray(
            image.data,
            dtype=np.float32,
        ).copy()

        if self.statistics is not None:
            median = np.asarray(
                self.statistics["median"],
                dtype=np.float32,
            )[:, None, None]

            scale = np.asarray(
                self.statistics["scale"],
                dtype=np.float32,
            )[:, None, None]

            image = (
                image - median
            ) / scale

            image = np.clip(
                image,
                -self.clip,
                self.clip,
            )

        # Invalid pixels are only placeholders.
        # The separate valid_mask is authoritative.
        image = np.where(
            valid[None, :, :],
            image,
            0.0,
        ).astype(np.float32)

        valid_mask = valid[
            None, :, :
        ].copy()

        if not np.isfinite(image).all():
            raise RuntimeError(
                f"Non-finite optical tensor: "
                f"{sample['patch_id']} "
                f"{sample['period']}"
            )

        return {
            "image": torch.from_numpy(
                np.ascontiguousarray(image)
            ),
            "valid_mask": torch.from_numpy(
                np.ascontiguousarray(valid_mask)
            ),
            "patch_id": sample["patch_id"],
            "year": sample["year"],
            "quarter": sample["quarter"],
            "period": sample["period"],
            "row": sample["row"],
            "col": sample["col"],
            "valid_fraction": sample[
                "valid_fraction"
            ],
        }
