
import json

import pytest
import torch
import yaml

from sarchaeology.training.experiment import (
    create_experiment,
    load_training_config,
    resolve_amp,
    resolve_device,
    set_global_seed,
)


def test_seed_reproducibility():
    set_global_seed(42)
    first = torch.rand(5)

    set_global_seed(42)
    second = torch.rand(5)

    assert torch.equal(first, second)


def test_cpu_device_supported():
    device = resolve_device("cpu")
    assert device == torch.device("cpu")


def test_invalid_device_rejected():
    with pytest.raises(ValueError):
        resolve_device("unknown")


def test_amp_disabled_on_cpu():
    amp = resolve_amp(
        torch.device("cpu"),
        enabled=True,
        dtype="float16",
    )

    assert amp["enabled"] is False


def test_missing_configuration_rejected(tmp_path):
    path = tmp_path / "sar.yaml"

    path.write_text(
        yaml.safe_dump({
            "dataset": {},
            "training": {},
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_training_config(path)


def test_experiment_files_created(tmp_path):
    norm_path = tmp_path / "normalization.json"

    norm_path.write_text(
        json.dumps({
            "method": "robust_iqr",
            "source_manifest_sha256": "testhash",
        }),
        encoding="utf-8",
    )

    config = {
        "dataset": {
            "dataset_id": "test_dataset"
        },
        "training": {
            "seed": 42
        },
        "experiment": {
            "project": "test_project",
            "model_name": "sar_cae",
            "output_directory": "outputs/experiments/sar",
            "checkpoint_directory": "checkpoints/sar",
        },
    }

    metadata = create_experiment(
        project_root=tmp_path,
        config=config,
        normalization_path=norm_path,
        train_count=10,
        validation_count=4,
    )

    from pathlib import Path

    run_dir = Path(
        metadata["run_directory"]
    )

    ckpt_dir = Path(
        metadata["checkpoint_directory"]
    )

    assert run_dir.exists()
    assert ckpt_dir.exists()

    assert (
        run_dir / "config.yaml"
    ).exists()

    assert (
        run_dir / "metadata.json"
    ).exists()

    assert (
        run_dir / "sar_normalization.json"
    ).exists()

    saved = json.loads(
        (
            run_dir / "metadata.json"
        ).read_text(encoding="utf-8")
    )

    assert saved["status"] == "prepared"
    assert saved["train_observations"] == 10
    assert saved["validation_observations"] == 4
