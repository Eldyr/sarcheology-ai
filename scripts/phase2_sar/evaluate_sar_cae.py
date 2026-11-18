
"""
SARchaeology AI — Phase 2.8
Evaluation of the trained Sentinel-1 SAR CAE.

Features:
    - Load best.pt checkpoint
    - Use original train-fitted normalization
    - Evaluate complete validation or test split
    - Compute masked MAE, RMSE, Huber
    - Report separate VV and VH performance
    - Compare against a constant-median baseline
    - Export per-observation metrics
    - Export per-patch metrics
    - Export per-acquisition metrics
    - Save SAR reconstruction visualizations

Example:
    uv run python scripts/phase2_sar/evaluate_sar_cae.py \
        --run-id YOUR_RUN_ID --split val --examples 6
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "src"),
)

from sarchaeology.data.dataloaders import (
    create_sar_dataloaders,
)

from sarchaeology.evaluation.sar_metrics import (
    huber_sums,
    masked_image_metrics,
    summarize,
)

from sarchaeology.models.sar_cae import (
    build_sar_cae,
)

from sarchaeology.preprocessing.sar_normalization import (
    SARNormalizer,
)

from sarchaeology.training.experiment import (
    resolve_amp,
    resolve_device,
    sha256_file,
)


CHANNEL_NAMES = ("VV", "VH")


def save_reconstruction_example(
    target,
    prediction,
    valid_mask,
    patch_id,
    acquisition_date,
    output_path,
):
    """
    Save a 2x3 reconstruction comparison figure.

    Rows:
        VV
        VH

    Columns:
        Original
        Reconstructed
        Absolute reconstruction error

    Values are in normalized units, not dB.
    """

    target = (
        target.detach()
        .float()
        .cpu()
        .numpy()
    )

    prediction = (
        prediction.detach()
        .float()
        .cpu()
        .numpy()
    )

    valid_mask = (
        valid_mask.detach()
        .cpu()
        .numpy()
    )

    valid = valid_mask[0] > 0

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(12, 7),
        constrained_layout=True,
    )

    for channel_index, channel_name in enumerate(
        CHANNEL_NAMES
    ):
        original = np.where(
            valid,
            target[channel_index],
            np.nan,
        )

        reconstructed = np.where(
            valid,
            prediction[channel_index],
            np.nan,
        )

        error = np.abs(
            reconstructed - original
        )

        panels = (
            (original, "Original"),
            (reconstructed, "Reconstructed"),
            (error, "Absolute error"),
        )

        for column, (array, label) in enumerate(panels):
            axis = axes[
                channel_index,
                column,
            ]

            if column == 2:
                finite_errors = error[
                    np.isfinite(error)
                ]

                vmax = (
                    max(
                        float(
                            np.percentile(
                                finite_errors,
                                99,
                            )
                        ),
                        0.01,
                    )
                    if finite_errors.size
                    else 0.01
                )

                vmin = 0.0
                cmap = "magma"

            else:
                vmin = -3.0
                vmax = 3.0
                cmap = "gray"

            image = axis.imshow(
                array,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )

            axis.set_title(
                f"{channel_name} — {label}"
            )

            axis.axis("off")

            fig.colorbar(
                image,
                ax=axis,
                fraction=0.046,
                pad=0.04,
            )

    fig.suptitle(
        f"Patch: {patch_id} | "
        f"Acquisition: {acquisition_date}\n"
        "Sentinel-1 reconstruction "
        "(normalized backscatter)",
        fontsize=12,
    )

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


def locate_artifacts(run_id):
    """
    Locate the original training run and checkpoint.
    """

    run_dir = (
        PROJECT_ROOT
        / "outputs"
        / "experiments"
        / "sar"
        / "sar_cae"
        / run_id
    )

    checkpoint_path = (
        PROJECT_ROOT
        / "checkpoints"
        / "sar"
        / "sar_cae"
        / run_id
        / "best.pt"
    )

    if not run_dir.is_dir():
        raise FileNotFoundError(
            f"Experiment directory not found: {run_dir}"
        )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Best checkpoint not found: {checkpoint_path}"
        )

    config_path = (
        run_dir / "config.yaml"
    )

    statistics_path = (
        run_dir / "sar_normalization.json"
    )

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Missing experiment config: {config_path}"
        )

    if not statistics_path.is_file():
        raise FileNotFoundError(
            f"Missing normalization artifact: {statistics_path}"
        )

    return (
        run_dir,
        checkpoint_path,
        config_path,
        statistics_path,
    )


def verify_normalization(
    config,
    normalizer,
    statistics_path,
    checkpoint,
):
    """
    Verify that evaluation uses the exact train-fitted
    normalization artifact used during training.
    """

    dataset_id = (
        config["dataset"]["dataset_id"]
    )

    statistics = normalizer.statistics

    if statistics["dataset_id"] != dataset_id:
        raise RuntimeError(
            "Normalization dataset ID mismatch"
        )

    if statistics["fit_split"] != "train":
        raise RuntimeError(
            "Normalization must be fitted on train"
        )

    expected_hash = checkpoint[
        "normalization_sha256"
    ]

    actual_hash = sha256_file(
        statistics_path
    )

    if actual_hash != expected_hash:
        raise RuntimeError(
            "Normalization checksum mismatch. "
            "Evaluation must use the same statistics "
            "as training."
        )

    train_manifest = (
        PROJECT_ROOT
        / "data"
        / "metadata"
        / "manifests"
        / "splits"
        / dataset_id
        / "train.parquet"
    )

    if not train_manifest.is_file():
        raise FileNotFoundError(
            train_manifest
        )

    manifest_hash = sha256_file(
        train_manifest
    )

    if (
        manifest_hash
        != statistics["source_manifest_sha256"]
    ):
        raise RuntimeError(
            "Training manifest changed after "
            "normalization was fitted"
        )

    if (
        float(statistics["min_valid_fraction"])
        != float(
            config["dataset"]["min_valid_fraction"]
        )
    ):
        raise RuntimeError(
            "Dataset validity threshold mismatch"
        )

    return actual_hash


def create_evaluation_loader(
    config,
    normalizer,
    split,
):
    """
    Build deterministic evaluation DataLoaders.

    The existing factory also creates the train loader,
    but only the requested evaluation split is iterated.
    """

    # Do not shuffle validation or test data.
    config = dict(config)

    config["dataloader"] = dict(
        config["dataloader"]
    )

    config["dataloader"]["shuffle_train"] = False

    datasets, loaders = create_sar_dataloaders(
        project_root=PROJECT_ROOT,
        config=config,
        include_test=(split == "test"),
        normalizer=normalizer,
    )

    if split not in loaders:
        raise RuntimeError(
            f"Requested split not available: {split}"
        )

    return datasets[split], loaders[split]


def evaluate(
    model,
    loader,
    device,
    amp,
    huber_delta,
    examples,
    output_dir,
):
    """
    Evaluate every observation in the selected split.

    Returns:
        A DataFrame of additive per-observation metrics.
    """

    model.eval()

    rows = []
    saved_examples = 0

    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(
                device,
                non_blocking=True,
            )

            masks = batch["valid_mask"].to(
                device,
                non_blocking=True,
            )

            with torch.autocast(
                device_type=device.type,
                dtype=amp["dtype"],
                enabled=amp["enabled"],
            ):
                outputs = model(
                    images,
                    valid_mask=masks,
                )

            prediction = (
                outputs["reconstruction"]
                .float()
            )

            target = images.float()

            counts, absolute, squared = (
                masked_image_metrics(
                    prediction,
                    target,
                    masks,
                )
            )

            huber = huber_sums(
                prediction,
                target,
                masks,
                delta=huber_delta,
            )

            # After robust normalization, zero is
            # the train-channel median. This is a
            # simple baseline rather than a spatial model.
            baseline_prediction = (
                torch.zeros_like(target)
            )

            (
                _,
                baseline_absolute,
                baseline_squared,
            ) = masked_image_metrics(
                baseline_prediction,
                target,
                masks,
            )

            batch_size = target.shape[0]

            for sample_index in range(batch_size):
                vv_count = int(
                    counts[sample_index, 0].item()
                )

                vh_count = int(
                    counts[sample_index, 1].item()
                )

                patch_id = str(
                    batch["patch_id"][sample_index]
                )

                acquisition_date = str(
                    batch["acquisition_date"][
                        sample_index
                    ]
                )

                row = {
                    "patch_id": patch_id,

                    "acquisition_date": (
                        acquisition_date
                    ),

                    "acquisition_index": int(
                        batch["acquisition_index"][
                            sample_index
                        ]
                    ),

                    "valid_channel_pixels": (
                        vv_count + vh_count
                    ),

                    "vv_count": vv_count,
                    "vh_count": vh_count,

                    "vv_abs_sum": float(
                        absolute[sample_index, 0].item()
                    ),

                    "vh_abs_sum": float(
                        absolute[sample_index, 1].item()
                    ),

                    "vv_sq_sum": float(
                        squared[sample_index, 0].item()
                    ),

                    "vh_sq_sum": float(
                        squared[sample_index, 1].item()
                    ),

                    "abs_sum": float(
                        absolute[sample_index].sum().item()
                    ),

                    "sq_sum": float(
                        squared[sample_index].sum().item()
                    ),

                    "huber_sum": float(
                        huber[sample_index].item()
                    ),

                    "baseline_abs_sum": float(
                        baseline_absolute[
                            sample_index
                        ].sum().item()
                    ),

                    "baseline_sq_sum": float(
                        baseline_squared[
                            sample_index
                        ].sum().item()
                    ),
                }

                rows.append(row)

                if saved_examples < examples:
                    output_path = (
                        output_dir
                        / "examples"
                        / f"example_{saved_examples + 1:02d}.png"
                    )

                    save_reconstruction_example(
                        target=target[sample_index],
                        prediction=prediction[sample_index],
                        valid_mask=masks[sample_index],
                        patch_id=patch_id,
                        acquisition_date=acquisition_date,
                        output_path=output_path,
                    )

                    saved_examples += 1

    if not rows:
        raise RuntimeError(
            "Evaluation produced zero observations"
        )

    return pd.DataFrame(rows)


def export_results(
    frame,
    output_dir,
    run_id,
    split,
    checkpoint,
    config,
    normalization_sha256,
):
    """
    Save summary, observation, geographic and
    acquisition-level evaluation results.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame = frame.copy()

    frame["mae_normalized"] = (
        frame["abs_sum"]
        / frame["valid_channel_pixels"]
    )

    frame["rmse_normalized"] = np.sqrt(
        frame["sq_sum"]
        / frame["valid_channel_pixels"]
    )

    frame["huber_normalized"] = (
        frame["huber_sum"]
        / frame["valid_channel_pixels"]
    )

    frame["vv_mae_normalized"] = (
        frame["vv_abs_sum"]
        / frame["vv_count"]
    )

    frame["vh_mae_normalized"] = (
        frame["vh_abs_sum"]
        / frame["vh_count"]
    )

    # Full individual observation results.
    frame.to_csv(
        output_dir / "per_observation.csv",
        index=False,
    )

    # Aggregate by geographic patch or acquisition.
    for grouping_column, filename in (
        ("patch_id", "per_patch.csv"),
        ("acquisition_date", "per_acquisition.csv"),
    ):
        group_rows = []

        for value, group in frame.groupby(
            grouping_column,
            sort=True,
        ):
            group_rows.append({
                grouping_column: value,
                **summarize(group),
            })

        pd.DataFrame(
            group_rows
        ).to_csv(
            output_dir / filename,
            index=False,
        )

    summary = summarize(frame)

    summary.update({
        "run_id": run_id,
        "split": split,
        "dataset_id": config["dataset"]["dataset_id"],
        "checkpoint_epoch": int(
            checkpoint["epoch"]
        ),
        "loss_type": config["loss"]["type"],
        "huber_delta": float(
            config["loss"]["huber_delta"]
        ),
        "normalization_sha256": (
            normalization_sha256
        ),
        "comparison_baseline": (
            "train_channel_median"
        ),
    })

    summary_path = (
        output_dir / "summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained SAR CAE"
    )

    parser.add_argument(
        "--run-id",
        required=True,
        help="ID of the completed SAR CAE experiment",
    )

    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="val",
    )

    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="Explicitly authorize held-out test evaluation",
    )

    parser.add_argument(
        "--examples",
        type=int,
        default=6,
    )

    args = parser.parse_args()

    if (
        args.split == "test"
        and not args.allow_test
    ):
        parser.error(
            "Test evaluation requires --allow-test"
        )

    if args.examples < 0:
        parser.error(
            "--examples must be non-negative"
        )

    (
        run_dir,
        checkpoint_path,
        config_path,
        statistics_path,
    ) = locate_artifacts(args.run_id)

    config = yaml.safe_load(
        config_path.read_text(
            encoding="utf-8"
        )
    )

    normalizer = SARNormalizer.from_json(
        statistics_path
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if checkpoint["run_id"] != args.run_id:
        raise RuntimeError(
            "Checkpoint run ID does not match"
        )

    normalization_hash = verify_normalization(
        config,
        normalizer,
        statistics_path,
        checkpoint,
    )

    # This implementation reproduces our baseline
    # experiment, which used equal VV/VH weighting
    # and masked Huber reconstruction loss.
    loss_cfg = config["loss"]

    if loss_cfg["type"] != "huber":
        raise ValueError(
            "This evaluation expects the Huber "
            "loss used for baseline CAE training"
        )

    if [
        float(x)
        for x in loss_cfg.get(
            "channel_weights",
            [1.0, 1.0],
        )
    ] != [1.0, 1.0]:
        raise ValueError(
            "Expected equal VV/VH channel weights"
        )

    device = resolve_device(
        config["training"]["device"]
    )

    amp = resolve_amp(
        device=device,
        enabled=config["training"][
            "mixed_precision"
        ],
        dtype=config["training"]["amp_dtype"],
    )

    dataset, loader = create_evaluation_loader(
        config,
        normalizer,
        args.split,
    )

    model = build_sar_cae(
        config
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state"],
        strict=True,
    )

    model.eval()

    output_dir = (
        PROJECT_ROOT
        / "outputs"
        / "evaluations"
        / "sar_cae"
        / args.run_id
        / args.split
    )

    print("\nSARchaeology AI — Phase 2.8")
    print("Model:", model.__class__.__name__)
    print("Run ID:", args.run_id)
    print("Split:", args.split)
    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Device:", device)
    print("AMP enabled:", amp["enabled"])
    print("Observations:", len(dataset))

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    frame = evaluate(
        model=model,
        loader=loader,
        device=device,
        amp=amp,
        huber_delta=float(
            config["loss"]["huber_delta"]
        ),
        examples=args.examples,
        output_dir=output_dir,
    )

    summary = export_results(
        frame=frame,
        output_dir=output_dir,
        run_id=args.run_id,
        split=args.split,
        checkpoint=checkpoint,
        config=config,
        normalization_sha256=normalization_hash,
    )

    print("\nOVERALL RECONSTRUCTION METRICS")
    print(
        "MAE:",
        f"{summary['mae_normalized']:.6f}",
    )
    print(
        "RMSE:",
        f"{summary['rmse_normalized']:.6f}",
    )
    print(
        "Huber:",
        f"{summary['huber_normalized']:.6f}",
    )

    print("\nPER-CHANNEL METRICS")
    print(
        "VV MAE:",
        f"{summary['vv_mae_normalized']:.6f}",
    )
    print(
        "VH MAE:",
        f"{summary['vh_mae_normalized']:.6f}",
    )
    print(
        "VV RMSE:",
        f"{summary['vv_rmse_normalized']:.6f}",
    )
    print(
        "VH RMSE:",
        f"{summary['vh_rmse_normalized']:.6f}",
    )

    print("\nCONSTANT-MEDIAN BASELINE")
    print(
        "Baseline MAE:",
        f"{summary['baseline_mae_normalized']:.6f}",
    )
    print(
        "Baseline RMSE:",
        f"{summary['baseline_rmse_normalized']:.6f}",
    )

    if summary["mae_improvement_pct"] is not None:
        print(
            "MAE improvement:",
            f"{summary['mae_improvement_pct']:.2f}%",
        )

    if summary["rmse_improvement_pct"] is not None:
        print(
            "RMSE improvement:",
            f"{summary['rmse_improvement_pct']:.2f}%",
        )

    print("\nResults saved to:")
    print(output_dir)

    print(
        "\nPhase 2.8 SAR CAE evaluation completed."
    )


if __name__ == "__main__":
    main()
