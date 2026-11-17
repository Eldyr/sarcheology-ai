
import json

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from sarchaeology.data.sar_dataset import SARDataset

from sarchaeology.data.dataloaders import (
    create_sar_dataloaders,
    verify_disjoint_datasets,
)


def create_example_dataset(tmp_path):
    dataset_id = "test_2024"

    split_dir = (
        tmp_path
        / "data/metadata/manifests/splits"
        / dataset_id
    )

    patch_dir = (
        tmp_path
        / "data/patches/unassigned"
        / dataset_id
    )

    split_dir.mkdir(parents=True)
    patch_dir.mkdir(parents=True)

    metadata = {
        "patch_size": 4,
        "sar_sources": [
            {"date": "2024-01-10", "id": "s1"},
            {"date": "2024-01-22", "id": "s2"},
            {"date": "2024-02-03", "id": "s3"},
        ],
    }

    (patch_dir / "dataset_metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    for split, patch_id in [
        ("train", "train_patch"),
        ("val", "val_patch"),
    ]:
        sar = np.full(
            (3, 2, 4, 4),
            -12.0,
            dtype=np.float32,
        )

        valid = np.ones(
            (3, 4, 4),
            dtype=np.uint8,
        )

        # Acquisition 2 has insufficient
        # coverage and should be filtered.
        valid[1] = 0
        sar[1] = np.nan

        # Acquisition 3 is 75% valid.
        valid[2, 0] = 0
        sar[2, :, 0] = np.nan

        relative_path = (
            Path("data/patches/unassigned")
            / dataset_id
            / f"{patch_id}.npz"
        )

        full_path = tmp_path / relative_path

        np.savez_compressed(
            full_path,
            sar=sar,
            sar_valid=valid,
        )

        manifest = pd.DataFrame([{
            "patch_id": patch_id,
            "path": relative_path.as_posix(),
            "split": split,
        }])

        manifest.to_parquet(
            split_dir / f"{split}.parquet",
            index=False,
        )

    return dataset_id


def test_dataset_filters_invalid_acquisitions(tmp_path):
    dataset_id = create_example_dataset(tmp_path)

    dataset = SARDataset(
        project_root=tmp_path,
        dataset_id=dataset_id,
        split="train",
        min_valid_fraction=0.85,
    )

    assert len(dataset) == 1
    assert dataset.summary()[
        "geographic_patches"
    ] == 1


def test_dataset_shapes_and_metadata(tmp_path):
    dataset_id = create_example_dataset(tmp_path)

    dataset = SARDataset(
        project_root=tmp_path,
        dataset_id=dataset_id,
        split="train",
    )

    sample = dataset[0]

    assert sample["image"].shape == (
        2, 4, 4
    )

    assert sample["valid_mask"].shape == (
        1, 4, 4
    )

    assert sample["image"].dtype == torch.float32
    assert sample["valid_mask"].dtype == torch.float32

    assert sample["patch_id"] == "train_patch"
    assert sample["acquisition_index"] == 0
    assert sample["acquisition_date"] == "2024-01-10"

    assert torch.isfinite(
        sample["image"]
    ).all()


def test_invalid_pixels_are_zero_filled(tmp_path):
    dataset_id = create_example_dataset(tmp_path)

    dataset = SARDataset(
        project_root=tmp_path,
        dataset_id=dataset_id,
        split="train",
        min_valid_fraction=0.70,
    )

    assert len(dataset) == 2

    sample = dataset[1]

    image = sample["image"]
    mask = sample["valid_mask"]

    assert sample["acquisition_index"] == 2

    assert torch.all(mask[:, 0, :] == 0)
    assert torch.all(image[:, 0, :] == 0)

    assert torch.all(mask[:, 1:, :] == 1)


def test_geographic_splits_are_disjoint(tmp_path):
    dataset_id = create_example_dataset(tmp_path)

    train = SARDataset(
        tmp_path,
        dataset_id=dataset_id,
        split="train",
    )

    val = SARDataset(
        tmp_path,
        dataset_id=dataset_id,
        split="val",
    )

    verify_disjoint_datasets(train, val)

    assert set(
        train.manifest.patch_id
    ).isdisjoint(
        set(val.manifest.patch_id)
    )


def test_dataloaders(tmp_path):
    dataset_id = create_example_dataset(tmp_path)

    config = {
        "dataset": {
            "dataset_id": dataset_id,
            "min_valid_fraction": 0.70,
            "cache_size": 1,
        },
        "dataloader": {
            "batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "shuffle_train": True,
            "drop_last_train": False,
        },
    }

    datasets, loaders = create_sar_dataloaders(
        tmp_path,
        config,
    )

    batch = next(iter(loaders["train"]))

    assert len(datasets["train"]) == 2

    assert batch["image"].shape == (
        2, 2, 4, 4
    )

    assert batch["valid_mask"].shape == (
        2, 1, 4, 4
    )


def test_reject_invalid_split(tmp_path):
    dataset_id = create_example_dataset(tmp_path)

    with pytest.raises(ValueError):
        SARDataset(
            tmp_path,
            dataset_id=dataset_id,
            split="excluded",
        )
