
import json

import numpy as np
import pandas as pd
import pytest
import torch

from sarchaeology.preprocessing.sar_normalization import (
    SARNormalizer,
    fit_sar_statistics,
)


def example_statistics():
    return {
        "method": "robust_iqr",
        "fit_split": "train",
        "channel_names": ["VV", "VH"],
        "median_db": [-12.0, -20.0],
        "scale_db": [2.0, 4.0],
        "clip_min": -5.0,
        "clip_max": 5.0,
    }


def test_normalization_values():
    normalizer = SARNormalizer(
        example_statistics()
    )

    image = torch.tensor([
        [[-12.0, -10.0]],
        [[-20.0, -16.0]],
    ])

    mask = torch.ones(
        (1, 1, 2)
    )

    result = normalizer(image, mask)

    expected = torch.tensor([
        [[0.0, 1.0]],
        [[0.0, 1.0]],
    ])

    assert torch.allclose(
        result, expected
    )


def test_invalid_pixels_remain_zero():
    normalizer = SARNormalizer(
        example_statistics()
    )

    image = torch.tensor([
        [[-10.0, 0.0]],
        [[-16.0, 0.0]],
    ])

    mask = torch.tensor([
        [[1.0, 0.0]]
    ])

    result = normalizer(image, mask)

    assert torch.all(result[:, 0, 1] == 0)
    assert torch.all(result[:, 0, 0] == 1)


def test_extreme_values_are_clipped():
    normalizer = SARNormalizer(
        example_statistics()
    )

    image = torch.tensor([
        [[100.0]],
        [[100.0]],
    ])

    mask = torch.ones(1, 1, 1)

    result = normalizer(image, mask)

    assert torch.all(result == 5.0)


def test_non_training_statistics_rejected():
    statistics = example_statistics()
    statistics["fit_split"] = "val"

    with pytest.raises(
        ValueError,
        match="training split",
    ):
        SARNormalizer(statistics)


def test_statistics_fit_uses_train_only(tmp_path):
    dataset_id = "synthetic_2024"

    manifest_dir = (
        tmp_path
        / "data/metadata/manifests/splits"
        / dataset_id
    )

    patch_dir = (
        tmp_path
        / "data/patches/unassigned"
        / dataset_id
    )

    manifest_dir.mkdir(
        parents=True
    )

    patch_dir.mkdir(
        parents=True
    )

    metadata = {
        "patch_size": 4,
        "sar_sources": [
            {"id": "s1"},
            {"id": "s2"},
        ],
    }

    (
        patch_dir / "dataset_metadata.json"
    ).write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    train_path = patch_dir / "train.npz"

    # Training VV = -10 dB
    # Training VH = -20 dB
    sar = np.empty(
        (2, 2, 4, 4),
        dtype=np.float32,
    )

    sar[:, 0] = -10.0
    sar[:, 1] = -20.0

    valid = np.ones(
        (2, 4, 4),
        dtype=np.uint8,
    )

    np.savez_compressed(
        train_path,
        sar=sar,
        sar_valid=valid,
    )

    pd.DataFrame([{
        "patch_id": "train_patch",
        "path": str(
            train_path.relative_to(tmp_path)
        ),
        "split": "train",
    }]).to_parquet(
        manifest_dir / "train.parquet",
        index=False,
    )

    result = fit_sar_statistics(
        tmp_path,
        dataset_id,
        pixels_per_acquisition=4,
        seed=42,
    )

    assert result["median_db"] == [
        -10.0, -20.0
    ]

    assert result["fit_split"] == "train"
    assert result["eligible_observations"] == 2
    assert result["sample_counts"] == [8, 8]

    # With a constant synthetic signal,
    # scale must use the configured floor.
    assert result["scale_db"] == [
        0.01, 0.01
    ]


def test_statistics_are_reproducible(tmp_path):
    # Reuse a synthetic train manifest
    # containing variable VV/VH values.
    dataset_id = "repro_2024"

    manifest_dir = (
        tmp_path
        / "data/metadata/manifests/splits"
        / dataset_id
    )

    patch_dir = (
        tmp_path
        / "data/patches/unassigned"
        / dataset_id
    )

    manifest_dir.mkdir(parents=True)
    patch_dir.mkdir(parents=True)

    (
        patch_dir / "dataset_metadata.json"
    ).write_text(
        json.dumps({
            "patch_size": 4,
            "sar_sources": [{"id": "s1"}],
        }),
        encoding="utf-8",
    )

    sar = np.zeros(
        (1, 2, 4, 4),
        dtype=np.float32,
    )

    
    sar[0, 0] = (
        np.arange(16, dtype=np.float32)
        .reshape(4, 4)
        - 20
    )

    sar[0, 1] = (
        np.arange(16, dtype=np.float32)
        .reshape(4, 4)
        - 30
    )


    path = patch_dir / "one.npz"

    np.savez_compressed(
        path,
        sar=sar,
        sar_valid=np.ones(
            (1, 4, 4),
            dtype=np.uint8,
        ),
    )

    pd.DataFrame([{
        "patch_id": "one",
        "path": str(
            path.relative_to(tmp_path)
        ),
        "split": "train",
    }]).to_parquet(
        manifest_dir / "train.parquet",
        index=False,
    )

    first = fit_sar_statistics(
        tmp_path, dataset_id,
        pixels_per_acquisition=8,
        seed=123,
    )

    second = fit_sar_statistics(
        tmp_path, dataset_id,
        pixels_per_acquisition=8,
        seed=123,
    )

    assert first == second
