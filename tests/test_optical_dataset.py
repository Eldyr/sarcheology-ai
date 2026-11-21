
import json

import numpy as np
import pandas as pd
import pytest
import torch

from sarchaeology.data.optical_dataset import (
    BANDS,
    OpticalDataset,
    fit_optical_statistics,
    create_optical_dataloaders,
)


@pytest.fixture
def optical_root(tmp_path):
    directory = (
        tmp_path
        / "data/metadata/manifests/splits"
        / "buto_demo_2024"
    )
    directory.mkdir(parents=True)

    patch_dir = tmp_path / "data/patches"
    patch_dir.mkdir(parents=True)

    rng = np.random.default_rng(42)

    for split in ("train", "val", "test"):
        rows = []
        for i in range(2):
            patch_id = f"{split}_{i}"
            relative = f"data/patches/{patch_id}.npz"

            optical = rng.uniform(
                0.05, 0.5,
                size=(4, 6, 128, 128),
            ).astype(np.float32)

            valid = np.ones(
                (4, 128, 128), dtype=np.uint8
            )

            # One quarter is below the 50% threshold.
            valid[3] = 0
            optical[0, :, 0, 0] = np.nan

            np.savez_compressed(
                tmp_path / relative,
                optical=optical,
                optical_valid=valid,
            )

            rows.append({
                "patch_id": patch_id,
                "path": relative,
                "row": i * 128,
                "col": 0,
            })

        pd.DataFrame(rows).to_parquet(
            directory / f"{split}.parquet"
        )

    return tmp_path


def test_band_contract():
    assert BANDS == (
        "B02", "B03", "B04",
        "B08", "B11", "B12",
    )


def test_dataset_shapes_and_missing_pixels(optical_root):
    dataset = OpticalDataset(
        optical_root, split="train"
    )

    assert len(dataset) == 6

    sample = dataset[0]

    assert sample["image"].shape == (6, 128, 128)
    assert sample["valid_mask"].shape == (1, 128, 128)
    assert not sample["valid_mask"][0, 0, 0]
    assert torch.isfinite(sample["image"]).all()


def test_training_only_normalization(optical_root):
    train = OpticalDataset(
        optical_root, split="train"
    )
    stats = fit_optical_statistics(train)

    assert stats["fit_split"] == "train"
    assert len(stats["median"]) == 6
    assert all(value > 0 for value in stats["scale"])

    val = OpticalDataset(
        optical_root, split="val"
    )
    with pytest.raises(ValueError):
        fit_optical_statistics(val)


def test_normalized_dataloaders(optical_root):
    raw = OpticalDataset(
        optical_root, split="train"
    )
    stats = fit_optical_statistics(raw)

    datasets, loaders = create_optical_dataloaders(
        optical_root,
        stats,
        batch_size=2,
        num_workers=0,
    )

    for split in ("train", "val"):
        batch = next(iter(loaders[split]))

        assert batch["image"].shape == (
            2, 6, 128, 128
        )
        assert batch["valid_mask"].shape == (
            2, 1, 128, 128
        )
        assert torch.isfinite(batch["image"]).all()


def test_train_manifest_fingerprint(optical_root):
    raw = OpticalDataset(
        optical_root, split="train"
    )
    stats = fit_optical_statistics(raw)
    stats["train_manifest_sha256"] = "incorrect"

    with pytest.raises(RuntimeError):
        create_optical_dataloaders(
            optical_root,
            stats,
            num_workers=0,
        )
