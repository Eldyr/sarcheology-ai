
"""Phase 2.6/2.7 — Train the SAR CAE."""

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
from sarchaeology.models.sar_cae import (
    build_sar_cae,
    count_parameters,
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
from sarchaeology.training.losses import build_sar_loss
from sarchaeology.training.trainer import train_model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/training/sar.yaml",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    config = load_training_config(
        ROOT / args.config
    )

    set_global_seed(
        config["training"]["seed"],
        deterministic=config["training"][
            "deterministic"
        ],
    )

    device = resolve_device(
        config["training"]["device"]
    )

    amp = resolve_amp(
        device,
        enabled=config["training"][
            "mixed_precision"
        ],
        dtype=config["training"]["amp_dtype"],
    )

    stats_path = (
        ROOT
        / config["normalization"]["output_path"]
    ).resolve()

    normalizer = SARNormalizer.from_json(
        stats_path
    )

    train_manifest = (
        ROOT
        / "data/metadata/manifests/splits"
        / config["dataset"]["dataset_id"]
        / "train.parquet"
    )

    if (
        sha256_file(train_manifest)
        != normalizer.statistics[
            "source_manifest_sha256"
        ]
    ):
        raise RuntimeError(
            "Training manifest fingerprint mismatch"
        )

    if (
        normalizer.statistics["dataset_id"]
        != config["dataset"]["dataset_id"]
    ):
        raise RuntimeError(
            "Normalization dataset mismatch"
        )

    if (
        float(normalizer.statistics["min_valid_fraction"])
        != float(config["dataset"]["min_valid_fraction"])
    ):
        raise RuntimeError(
            "Validity threshold mismatch"
        )

    datasets, loaders = create_sar_dataloaders(
        project_root=ROOT,
        config=config,
        normalizer=normalizer,
    )

    model = build_sar_cae(config)

    criterion = build_sar_loss(config)

    normalization_hash = sha256_file(
        stats_path
    )

    if args.resume is None:
        # Creates unique checkpoint and output paths.
        metadata = create_experiment(
            project_root=ROOT,
            config=config,
            normalization_path=stats_path,
            train_count=len(datasets["train"]),
            validation_count=len(datasets["val"]),
        )
    else:
        # Resume from the same original experiment.
        checkpoint = torch.load(
            args.resume,
            map_location="cpu",
            weights_only=True,
        )

        run_id = checkpoint["run_id"]

        model_name = config["experiment"][
            "model_name"
        ]

        checkpoint_base = (
            ROOT
            / config["experiment"][
                "checkpoint_directory"
            ]
            / model_name
            / run_id
        ).resolve()

        run_dir = (
            ROOT
            / config["experiment"][
                "output_directory"
            ]
            / model_name
            / run_id
        ).resolve()

        if not (run_dir / "metadata.json").exists():
            raise FileNotFoundError(
                "Original experiment metadata not found"
            )

        metadata = {
            "run_id": run_id,
            "run_directory": str(run_dir),
            "checkpoint_directory": str(
                checkpoint_base
            ),
        }

        # Prevent accidentally loading a checkpoint
        # from another experiment.
        if (
            args.resume.resolve()
            != (checkpoint_base / "last.pt").resolve()
        ):
            raise ValueError(
                "Resume must use this run's last.pt"
            )

    print("\nSARchaeology AI — SAR CAE trainer")
    print("PyTorch:", torch.__version__)
    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print(
        "Train observations:",
        len(datasets["train"]),
    )
    print(
        "Validation observations:",
        len(datasets["val"]),
    )
    print(
        "Model parameters:",
        f"{count_parameters(model):,}",
    )
    print("AMP:", amp["enabled"])
    print("Loss:", criterion.loss_type)
    print("Experiment:", metadata["run_id"])

    result = train_model(
        model=model,
        criterion=criterion,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        device=device,
        amp=amp,
        training_cfg=config["training"],
        checkpoint_dir=metadata[
            "checkpoint_directory"
        ],
        run_dir=metadata["run_directory"],
        run_id=metadata["run_id"],
        normalization_sha256=normalization_hash,
        epochs=args.epochs,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        resume=args.resume,
    )

    print("\nTRAINING FINISHED")
    print(
        json.dumps(
            result,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
