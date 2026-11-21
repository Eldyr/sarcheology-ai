"""SARchaeology AI - Phase 3.3.3 multi-year optical ViT-Tiny MAE trainer.

Smoke test:
  uv run python scripts/phase3_optical/train_optical_mae_multiyear.py --smoke

Full run:
  uv run python scripts/phase3_optical/train_optical_mae_multiyear.py

Resume:
  uv run python scripts/phase3_optical/train_optical_mae_multiyear.py \
      --resume checkpoints/optical_mae/RUN_ID/last.pt
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
import uuid

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Sampler


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sarchaeology.data.optical_multiyear import (
    BANDS,
    MultiYearOpticalDataset,
)
from sarchaeology.models.optical_mae import OpticalViTMAE


DEFAULT_CONFIG = (
    ROOT / "configs/training/optical_mae.yaml"
)


def digest(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EpochSampler(Sampler):
    """Deterministic training order for any epoch."""

    def __init__(self, dataset, seed):
        self.dataset = dataset
        self.seed = seed
        self.epoch = 1

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(
            self.seed + self.epoch
        )
        return iter(
            torch.randperm(
                len(self.dataset),
                generator=generator,
            ).tolist()
        )

    def __len__(self):
        return len(self.dataset)


def build_model(config):
    return OpticalViTMAE(**config["model"])


def create_loaders(config, stats, years, valid_fraction, min_observations):
    """Create geographically separated multi-year loaders."""
    if tuple(stats["bands"]) != tuple(BANDS):
        raise RuntimeError("Optical band order mismatch")
    if stats["fit_split"] != "train":
        raise RuntimeError("Normalization must be fitted on training data")

    datasets = {
        split: MultiYearOpticalDataset(
            ROOT,
            years=years,
            split=split,
            statistics=stats,
            min_valid_fraction=valid_fraction,
            min_source_observations=min_observations,
        )
        for split in ("train", "val")
    }

    # Explicitly check that each geographic patch belongs to only one split.
    train_ids = {r["patch_id"] for r in datasets["train"].samples}
    val_ids = {r["patch_id"] for r in datasets["val"].samples}
    if train_ids & val_ids:
        raise RuntimeError("Training/validation geographic patch overlap")
    if {r["year"] for r in datasets["val"].samples} != {2024}:
        raise RuntimeError("Validation data must remain 2024-only")

    data_config = config["data"]
    batch_size = int(data_config["batch_size"])
    workers = int(data_config["num_workers"])
    loader_options = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": bool(data_config["pin_memory"]),
        "drop_last": False,
    }
    if workers > 0:
        loader_options["persistent_workers"] = True
        loader_options["prefetch_factor"] = int(
            data_config["prefetch_factor"]
        )
    sampler = EpochSampler(
        datasets["train"], seed=int(config["training"]["seed"])
    )
    loaders = {
        "train": DataLoader(datasets["train"], sampler=sampler, **loader_options),
        "val": DataLoader(datasets["val"], shuffle=False, **loader_options),
    }
    return datasets, loaders, sampler


def masked_statistics(model, image, valid, output, delta):
    """Compute Huber/MAE/MSE sums on valid masked pixels.

    Returns device tensors to avoid per-batch CPU sync.
    """
    prediction = output[
        "predicted_patches"
    ].float()

    target = torch.where(
        valid.expand_as(image).bool(),
        image,
        torch.zeros_like(image),
    ).float()

    target_patches = model.patchify(target)

    valid_patches = model.patchify(
        valid.expand_as(image).float()
    ) > 0.5

    scored = (
        output["patch_mask"].unsqueeze(-1)
        & valid_patches
    )

    diff = torch.where(
        scored,
        prediction - target_patches,
        torch.zeros_like(prediction),
    )

    absolute = diff.abs()

    huber = torch.where(
        absolute <= delta,
        0.5 * diff.square(),
        delta * (absolute - 0.5 * delta),
    )

    count = scored.sum()
    loss = huber.sum() / count.clamp_min(1)

    # Patch elements are ordered:
    # patch_y, patch_x, channel.
    b, n, _ = prediction.shape
    p = model.patch_size
    c = model.in_channels

    channel_abs = absolute.reshape(
        b, n, p * p, c
    ).sum(dim=(0, 1, 2))

    channel_count = scored.reshape(
        b, n, p * p, c
    ).sum(dim=(0, 1, 2))

    return loss, {
        "huber": huber.sum().detach(),
        "absolute": absolute.sum().detach(),
        "square": diff.square().sum().detach(),
        "count": count.detach(),
        "channel_absolute": channel_abs.detach(),
        "channel_count": channel_count.detach(),
    }


def epoch_run(
    model,
    loader,
    optimizer,
    scaler,
    device,
    config,
    *,
    epoch,
    training,
    smoke=False,
):
    model.train(training)

    settings = config["training"]
    seed = int(settings["seed"])

    amp_enabled = device.type == "cuda"

    totals = {
        "huber": torch.zeros((), device=device),
        "absolute": torch.zeros((), device=device),
        "square": torch.zeros((), device=device),
        "count": torch.zeros((), device=device),
        "channel_absolute": torch.zeros(6, device=device),
        "channel_count": torch.zeros(6, device=device),
    }

    batch_count = 0

    # Every validation batch gets the same masks each
    # epoch. Training masks change with the epoch.
    for batch_index, batch in enumerate(loader):
        if smoke and batch_index >= 1:
            break

        image = batch["image"].to(
            device, non_blocking=True
        )

        valid = batch["valid_mask"].to(
            device, non_blocking=True
        )

        mask_generator = torch.Generator(
            device="cpu"
        )

        mask_generator.manual_seed(
            seed
            + (epoch * 100000 if training else 202600000)
            + batch_index
        )

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=amp_enabled,
            ):
                output = model(
                    image,
                    valid,
                    generator=mask_generator,
                )

            loss, statistics = masked_statistics(
                model,
                image,
                valid,
                output,
                float(settings["huber_delta"]),
            )

            if training:
                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(
                        settings["gradient_clip_norm"]
                    ),
                )

                scaler.step(optimizer)
                scaler.update()

        for key in totals:
            totals[key] += statistics[key]

        batch_count += 1

        if (
            training
            and batch_count
            % int(settings["log_every_batches"]) == 0
        ):
            print(
                f"  epoch={epoch} "
                f"batch={batch_count}",
                flush=True,
            )

    if batch_count == 0:
        raise RuntimeError("No batches processed")

    # One synchronization at the end of the epoch.
    count = totals["count"].item()

    if count <= 0:
        raise RuntimeError("No valid masked pixels")

    channel_abs = (
        totals["channel_absolute"].cpu().numpy()
    )
    channel_count = (
        totals["channel_count"].cpu().numpy()
    )

    return {
        "masked_huber": (
            totals["huber"].item() / count
        ),
        "masked_mae": (
            totals["absolute"].item() / count
        ),
        "masked_rmse": (
            totals["square"].item() / count
        ) ** 0.5,
        "per_band_mae": {
            band: float(
                channel_abs[i] / max(channel_count[i], 1)
            )
            for i, band in enumerate(BANDS)
        },
        "masked_channel_pixels": int(count),
        "batches": batch_count,
    }


def atomic_save(path, state):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(
        path.name + "." + uuid.uuid4().hex + ".tmp"
    )

    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_json(path, value):
    path.write_text(
        json.dumps(value, indent=2),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--years", type=int, nargs="+", default=[2022, 2023, 2024, 2025]
    )
    parser.add_argument(
        "--mask-ratio", type=float, choices=[0.60, 0.75], default=0.75
    )
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--min-valid-fraction", type=float, default=None)
    parser.add_argument("--min-source-observations", type=int, default=None)

    args = parser.parse_args()

    if args.smoke and args.resume:
        parser.error(
            "Smoke and resume cannot be combined"
        )

    config_path = args.config.resolve()

    config = yaml.safe_load(
        config_path.read_text(encoding="utf-8")
    )

    if config["training"]["precision"] != "bf16":
        raise ValueError("This trainer expects BF16")

    if args.patience < 1:
        parser.error("--patience must be >= 1")
    if args.mask_ratio not in (0.60, 0.75):
        parser.error("Unsupported mask ratio")
    years = tuple(sorted(set(args.years)))
    if not years:
        parser.error("At least one year is required")
    if 2024 not in years:
        parser.error("Include 2024 so the 2024 benchmark is retained")
    config["model"]["mask_ratio"] = args.mask_ratio
    if args.batch_size is not None:
        if args.batch_size < 1:
            parser.error("--batch-size must be >= 1")
        config["data"]["batch_size"] = args.batch_size
    # Never write into the original 2024 training directories.
    config["experiment"]["output_dir"] = "outputs/training/optical_mae_multiyear"
    config["experiment"]["checkpoint_dir"] = "checkpoints/optical_mae_multiyear"
    config["data"]["statistics_path"] = (
        "outputs/statistics/optical_buto_multiyear_v1.json"
    )
    seed = int(config["training"]["seed"])
    seed_all(seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    statistics_path = (
        ROOT / config["data"]["statistics_path"]
    )

    stats = json.loads(
        statistics_path.read_text(encoding="utf-8")
    )

    valid_fraction = float(
        args.min_valid_fraction
        if args.min_valid_fraction is not None
        else stats.get("min_valid_fraction", config["data"]["min_valid_fraction"])
    )
    min_observations = int(
        args.min_source_observations
        if args.min_source_observations is not None
        else stats.get("min_source_observations", 1)
    )
    if "min_valid_fraction" in stats and abs(
        float(stats["min_valid_fraction"]) - valid_fraction
    ) > 1e-9:
        raise ValueError("Normalization and dataset validity thresholds differ")
    if "min_source_observations" in stats and int(
        stats["min_source_observations"]
    ) != min_observations:
        raise ValueError("Normalization and source-observation thresholds differ")

    # Effective configuration fingerprints include CLI overrides and datasets.
    config["multiyear"] = {
        "years": list(years),
        "patience": args.patience,
        "min_valid_fraction": valid_fraction,
        "min_source_observations": min_observations,
    }
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode("utf-8")
    ).hexdigest()
    stats_hash = digest(statistics_path)

    datasets, loaders, sampler = create_loaders(
        config, stats, years, valid_fraction, min_observations
    )

    manifest_root = ROOT / "data/metadata/manifests/splits/buto_demo_2024"
    manifest_hash = digest(manifest_root / "train.parquet")
    val_manifest_hash = digest(manifest_root / "val.parquet")
    test_manifest_hash = digest(manifest_root / "test.parquet")
    test_ids = set(
        __import__("pandas").read_parquet(manifest_root / "test.parquet")
        ["patch_id"].astype(str)
    )
    train_ids = {r["patch_id"] for r in datasets["train"].samples}
    val_ids = {r["patch_id"] for r in datasets["val"].samples}
    if train_ids & test_ids or val_ids & test_ids:
        raise RuntimeError("Test geographic patches overlap with train or val")
    if len(train_ids) != 240 or len(val_ids) != 20:
        raise RuntimeError("Unexpected geographic footprint counts")

    print("\nSARchaeology AI - Phase 3.3.3 multi-year ViT-Tiny")
    print("Device:", device)
    print("Training samples:", len(datasets["train"]))
    print("Validation samples:", len(datasets["val"]))

    model = build_model(config).to(device)

    print(
        "Parameters:",
        f"{sum(p.numel() for p in model.parameters()):,}",
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(
            config["training"]["weight_decay"]
        ),
    )

    epochs = int(config["training"]["epochs"])

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=float(
                config["training"]["min_learning_rate"]
            ),
        )
    )

    # BF16 does not require gradient scaling.
    scaler = torch.amp.GradScaler(
        "cuda", enabled=False
    )

    start_epoch = 1
    best_loss = float("inf")
    run_id = None
    no_improve = 0

    if args.resume:
        checkpoint = torch.load(
            args.resume.resolve(),
            map_location="cpu",
            weights_only=True,
        )

        for key, expected in (
            ("config_sha256", config_hash),
            ("normalization_sha256", stats_hash),
            ("train_manifest_sha256", manifest_hash),
            ("val_manifest_sha256", val_manifest_hash),
            ("test_manifest_sha256", test_manifest_hash),
        ):
            if checkpoint[key] != expected:
                raise RuntimeError(
                    f"Resume fingerprint mismatch: {key}"
                )

        model.load_state_dict(
            checkpoint["model_state"]
        )
        optimizer.load_state_dict(
            checkpoint["optimizer_state"]
        )
        scheduler.load_state_dict(
            checkpoint["scheduler_state"]
        )

        run_id = checkpoint["run_id"]
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["best_val_loss"]
        no_improve = int(checkpoint.get("no_improve", 0))

    else:
        stamp = datetime.now(
            timezone.utc
        ).strftime("%Y%m%dT%H%M%SZ")

        run_id = (
            f"optical_mae_multiyear_m{int(args.mask_ratio * 100)}_{stamp}_"
            f"{uuid.uuid4().hex[:8]}"
        )

    output_dir = (
        ROOT
        / config["experiment"]["output_dir"]
        / run_id
    )

    checkpoint_dir = (
        ROOT
        / config["experiment"]["checkpoint_dir"]
        / run_id
    )

    if not args.smoke:
        output_dir.mkdir(
            parents=True, exist_ok=True
        )
        checkpoint_dir.mkdir(
            parents=True, exist_ok=True
        )

        if not args.resume:
            (output_dir / "config.yaml").write_text(
                yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
            )

            save_json(
                output_dir / "metadata.json",
                {
                    "run_id": run_id,
                    "config_sha256": config_hash,
                    "normalization_sha256": stats_hash,
                    "train_manifest_sha256": manifest_hash,
                    "val_manifest_sha256": val_manifest_hash,
                    "test_manifest_sha256": test_manifest_hash,
                    "years": list(years),
                    "mask_ratio": args.mask_ratio,
                    "bands": list(BANDS),
                    "device": str(device),
                },
            )

    for epoch in range(
        start_epoch,
        2 if args.smoke else epochs + 1,
    ):
        start_time = time.perf_counter()

        seed_all(seed + epoch)
        sampler.set_epoch(epoch)

        train_metrics = epoch_run(
            model,
            loaders["train"],
            optimizer,
            scaler,
            device,
            config,
            epoch=epoch,
            training=True,
            smoke=args.smoke,
        )

        val_metrics = epoch_run(
            model,
            loaders["val"],
            None,
            scaler,
            device,
            config,
            epoch=epoch,
            training=False,
            smoke=args.smoke,
        )

        current_lr = optimizer.param_groups[0]["lr"]

        improved = (
            val_metrics["masked_huber"] < best_loss
        )

        if improved:
            best_loss = val_metrics["masked_huber"]
            no_improve = 0
        else:
            no_improve += 1

        if not args.smoke:
            scheduler.step()

        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "best_val_masked_huber": best_loss,
            "epochs_without_improvement": no_improve,
            "learning_rate": current_lr,
            "epoch_seconds": (
                time.perf_counter() - start_time
            ),
        }

        print(
            f"Epoch {epoch}/{epochs} "
            f"train={train_metrics['masked_huber']:.6f} "
            f"val={val_metrics['masked_huber']:.6f} "
            f"best={best_loss:.6f} "
            f"time={record['epoch_seconds']:.1f}s",
            flush=True,
        )

        if args.smoke:
            print(
                "\nSMOKE TEST PASSED. "
                "No checkpoints saved."
            )
            return

        with (
            output_dir / "history.jsonl"
        ).open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        state = {
            "epoch": epoch,
            "run_id": run_id,
            "best_val_loss": best_loss,
            "no_improve": no_improve,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "config_sha256": config_hash,
            "normalization_sha256": stats_hash,
            "train_manifest_sha256": manifest_hash,
            "val_manifest_sha256": val_manifest_hash,
            "test_manifest_sha256": test_manifest_hash,
            "metrics": record,
        }

        atomic_save(
            checkpoint_dir / "last.pt",
            state,
        )

        if improved:
            atomic_save(
                checkpoint_dir / "best.pt",
                state,
            )

        if no_improve >= args.patience:
            print(f"Early stopping: {no_improve} epochs without improvement")
            break

    print("\nPhase 3.3.3 multi-year training complete.")
    print("Run ID:", run_id)
    print("Best validation Huber:", best_loss)
    print("Checkpoints:", checkpoint_dir)
    print("Metrics:", output_dir)


if __name__ == "__main__":
    main()
