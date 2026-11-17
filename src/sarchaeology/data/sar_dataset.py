"""Placeholder module for SARchaeology AI v1.0."""

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset


class SARDataset(Dataset):
    """
    Sentinel-1 VV/VH dataset.

    One item = one acquisition from one
    geographically assigned patch.

    Image: [2, H, W]
    Mask:  [1, H, W]

    No normalization is applied here.
    """

    def __init__(
        self,
        project_root,
        dataset_id="buto_demo_2024",
        split="train",
        min_valid_fraction=0.85,
        cache_size=2,
    ):
        self.root = Path(project_root).resolve()
        self.dataset_id = dataset_id
        self.split = split
        self.min_valid_fraction = float(
            min_valid_fraction
        )
        self.cache_size = int(cache_size)

        if split not in ("train", "val", "test"):
            raise ValueError(
                "Split must be train, val or test"
            )

        if not 0 <= self.min_valid_fraction <= 1:
            raise ValueError(
                "min_valid_fraction must be in [0, 1]"
            )

        if self.cache_size < 0:
            raise ValueError(
                "cache_size cannot be negative"
            )

        manifest_path = (
            self.root
            / "data/metadata/manifests/splits"
            / dataset_id
            / f"{split}.parquet"
        )

        metadata_path = (
            self.root
            / "data/patches/unassigned"
            / dataset_id
            / "dataset_metadata.json"
        )

        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)

        if not metadata_path.is_file():
            raise FileNotFoundError(metadata_path)

        self.manifest = pd.read_parquet(
            manifest_path
        )

        self.metadata = json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )

        if self.manifest.empty:
            raise ValueError(
                f"No patches found in {split}"
            )

        if not (
            self.manifest["split"] == split
        ).all():
            raise ValueError(
                "Manifest contains unexpected split labels"
            )

        if not self.manifest["patch_id"].is_unique:
            raise ValueError(
                "Duplicate geographic patch IDs"
            )

        self.patch_size = int(
            self.metadata["patch_size"]
        )

        self.acquisitions = (
            self.metadata["sar_sources"]
        )

        self.n_acquisitions = len(
            self.acquisitions
        )

        if self.n_acquisitions == 0:
            raise ValueError(
                "Dataset contains no SAR acquisitions"
            )

        # Each entry stores:
        # (path, patch_id, acquisition_index,
        #  acquisition_valid_fraction)
        self.index = []

        # Per-instance cache, not a global cache.
        self._cache = OrderedDict()

        self._build_index()

        if not self.index:
            raise ValueError(
                f"No valid SAR observations for {split}"
            )

    def _resolve_path(self, value):
        path = Path(value)

        if not path.is_absolute():
            path = self.root / path

        if not path.is_file():
            raise FileNotFoundError(path)

        return path

    def _build_index(self):
        """
        Read validity masks once to select
        eligible acquisitions.

        Do not load all SAR imagery into RAM.
        """

        expected_shape = (
            self.n_acquisitions,
            self.patch_size,
            self.patch_size,
        )

        for record in self.manifest.itertuples(
            index=False
        ):
            path = self._resolve_path(
                record.path
            )

            with np.load(
                path,
                allow_pickle=False,
            ) as archive:
                valid = archive["sar_valid"]

            if valid.shape != expected_shape:
                raise ValueError(
                    f"Invalid sar_valid shape "
                    f"in {path.name}: {valid.shape}"
                )

            fractions = valid.mean(
                axis=(1, 2)
            )

            for acquisition_index, fraction in enumerate(
                fractions
            ):
                if fraction >= self.min_valid_fraction:
                    self.index.append(
                        (
                            path,
                            record.patch_id,
                            acquisition_index,
                            float(fraction),
                        )
                    )

    def _read_sar(self, path):
        """
        Load one SAR time series.

        Cache a small number of decompressed
        archives to reduce repeated NPZ reads.
        """

        if (
            self.cache_size > 0
            and path in self._cache
        ):
            self._cache.move_to_end(path)
            return self._cache[path]

        with np.load(
            path,
            allow_pickle=False,
        ) as archive:
            sar = archive["sar"].astype(
                np.float32,
                copy=True,
            )

            sar_valid = archive[
                "sar_valid"
            ].astype(
                bool,
                copy=True,
            )

        expected_sar_shape = (
            self.n_acquisitions,
            2,
            self.patch_size,
            self.patch_size,
        )

        expected_mask_shape = (
            self.n_acquisitions,
            self.patch_size,
            self.patch_size,
        )

        if sar.shape != expected_sar_shape:
            raise ValueError(
                f"Unexpected SAR shape: {sar.shape}"
            )

        if sar_valid.shape != expected_mask_shape:
            raise ValueError(
                f"Unexpected validity shape: "
                f"{sar_valid.shape}"
            )

        if self.cache_size > 0:
            self._cache[path] = (
                sar,
                sar_valid,
            )

            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)

        return sar, sar_valid

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        path, patch_id, acquisition_index, fraction = (
            self.index[index]
        )

        sar, sar_valid = self._read_sar(path)

        image = sar[
            acquisition_index
        ].copy()

        valid_mask = sar_valid[
            acquisition_index
        ].copy()

        # Defensive check against stale masks.
        actual_valid = np.isfinite(
            image
        ).all(axis=0)

        if not np.array_equal(
            actual_valid,
            valid_mask,
        ):
            raise ValueError(
                f"SAR mask mismatch: {patch_id}, "
                f"acquisition {acquisition_index}"
            )

        if not np.isfinite(
            image[:, valid_mask]
        ).all():
            raise ValueError(
                "Invalid radar measurements "
                "inside the validity mask"
            )

        # Replace invalid locations with finite
        # placeholders to support batching.
        #
        # The later loss must ignore these
        # locations using valid_mask.
        image[:, ~valid_mask] = 0.0

        image = torch.from_numpy(
            image.astype(
                np.float32,
                copy=False,
            )
        )

        mask = torch.from_numpy(
            valid_mask[None].astype(
                np.float32,
                copy=False,
            )
        )

        acquisition = self.acquisitions[
            acquisition_index
        ]

        return {
            "image": image,
            "valid_mask": mask,
            "patch_id": patch_id,
            "acquisition_index": acquisition_index,
            "acquisition_date": str(
                acquisition["date"]
            ),
            "valid_fraction": fraction,
        }

    def summary(self):
        unique_patches = {
            sample[1]
            for sample in self.index
        }

        return {
            "split": self.split,
            "geographic_patches": len(
                self.manifest
            ),
            "patches_with_valid_sar": len(
                unique_patches
            ),
            "available_acquisitions": (
                self.n_acquisitions
            ),
            "usable_observations": len(
                self.index
            ),
            "minimum_valid_fraction": (
                self.min_valid_fraction
            ),
        }
