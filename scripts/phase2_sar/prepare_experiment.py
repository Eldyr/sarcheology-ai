
"""Phase 2.3: prepare and validate a SAR experiment."""

import argparse
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sarchaeology.data.dataloaders import (
    create_sar_dataloaders,
)
from sarchaeology.preprocessing.sar_normalization import (
    SARNormalizer,
)
from sarchaeology.training.experiment import (
    create_experiment,
    load_training_config,
    resolve_amp,
    resolve_device,
    set_global_seed,
    sha256_file,
)


def verify_normalization(root, config):
    dataset_id = config["dataset"]["dataset_id"]

    path = (
        root
        / config["normalization"]["output_path"]
    ).resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"Missing normalization file: {path}\n"
            "First run fit_normalization.py"
        )

    normalizer = SARNormalizer.from_json(path)
    stats = normalizer.statistics

    manifest_path = (
        root
        / "data/metadata/manifests/splits"
        / dataset_id
        / "train.parquet"
    )

    if stats["dataset_id"] != dataset_id:
        raise ValueError(
            "Normalization dataset ID mismatch"
        )

    if stats["fit_split"] != "train":
        raise ValueError(
            "Normalization was not fitted on train"
        )

    if (
        sha256_file(manifest_path)
        != stats["source_manifest_sha256"]
    ):
        raise RuntimeError(
            "Training manifest has changed. "
            "Refit normalization statistics."
        )

    if (
        float(stats["min_valid_fraction"])
        != float(
            config["dataset"]["min_valid_fraction"]
        )
    ):
        raise ValueError(
            "SAR validity thresholds do not match"
        )

    return normalizer, path


def verify_batch(batch, device, amp):
    images = batch["image"]
    masks = batch["valid_mask"]

    if images.ndim != 4 or images.shape[1] != 2:
        raise ValueError(
            f"Invalid image batch shape: {images.shape}"
        )

    if masks.shape != (
        images.shape[0],
        1,
        images.shape[2],
        images.shape[3],
    ):
        raise ValueError(
            "Image and mask dimensions disagree"
        )

    if not torch.isfinite(images).all():
        raise ValueError(
            "Batch contains non-finite SAR values"
        )

    valid = masks > 0

    if not valid.any():
        raise ValueError(
            "Batch contains no valid SAR pixels"
        )

    if not torch.all(
        images[
            ~valid.expand_as(images)
        ] == 0
    ):
        raise ValueError(
            "Invalid SAR pixels are not zero"
        )

    images = images.to(
        device,
        non_blocking=True,
    )

    masks = masks.to(
        device,
        non_blocking=True,
    )

    # Exercise autocasting without a neural network.
    # Real reconstruction-loss arithmetic will be
    # implemented in Step 2.5.
    with torch.no_grad():
        with torch.autocast(
            device_type=device.type,
            dtype=amp["dtype"],
            enabled=amp["enabled"],
        ):
            result = images * masks

    torch.cuda.synchronize() if device.type == "cuda" else None

    if not torch.isfinite(result).all():
        raise ValueError(
            "GPU smoke test produced non-finite values"
        )

    return {
        "image_shape": list(images.shape),
        "mask_shape": list(masks.shape),
        "device": str(images.device),
        "valid_fraction": float(
            masks.mean().item()
        ),
        "minimum": float(images.min().item()),
        "maximum": float(images.max().item()),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/training/sar.yaml",
    )

    args = parser.parse_args()

    config_path = ROOT / args.config
    config = load_training_config(config_path)

    training_cfg = config["training"]

    set_global_seed(
        training_cfg["seed"],
        deterministic=training_cfg["deterministic"],
    )

    device = resolve_device(
        training_cfg["device"]
    )

    amp = resolve_amp(
        device=device,
        enabled=training_cfg["mixed_precision"],
        dtype=training_cfg["amp_dtype"],
    )

    normalizer, stats_path = verify_normalization(
        ROOT,
        config,
    )

    datasets, loaders = create_sar_dataloaders(
        project_root=ROOT,
        config=config,
        normalizer=normalizer,
    )

    train_count = len(datasets["train"])
    validation_count = len(datasets["val"])

    if train_count == 0 or validation_count == 0:
        raise ValueError(
            "Training or validation dataset is empty"
        )

    train_batch = next(iter(loaders["train"]))
    val_batch = next(iter(loaders["val"]))

    train_result = verify_batch(
        train_batch, device, amp
    )
    val_result = verify_batch(
        val_batch, device, amp
    )

    # Only record a prepared experiment after
    # all preceding checks have succeeded.
    metadata = create_experiment(
        project_root=ROOT,
        config=config,
        normalization_path=stats_path,
        train_count=train_count,
        validation_count=validation_count,
    )

    print("\nSARchaeology AI — Phase 2.3")
    print("Model:", config["experiment"]["model_name"])
    print("PyTorch:", torch.__version__)
    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print(
        "AMP:",
        amp["enabled"],
        amp["dtype_name"],
    )

    print(
        "\nTRAIN:",
        json.dumps(train_result, indent=2),
    )

    print(
        "\nVALIDATION:",
        json.dumps(val_result, indent=2),
    )

    print(
        "\nTraining observations:",
        train_count,
    )

    print(
        "Validation observations:",
        validation_count,
    )

    print("\nExperiment ID:", metadata["run_id"])

    print(
        "Experiment directory:",
        metadata["run_directory"],
    )

    print(
        "Checkpoint directory:",
        metadata["checkpoint_directory"],
    )

    print(
        "\nPhase 2.3 experiment preparation passed."
    )


if __name__ == "__main__":
    main()
