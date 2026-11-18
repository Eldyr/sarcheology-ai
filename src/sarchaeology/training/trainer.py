
"""Phase 2.6 — Reusable SAR CAE training engine."""

import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn


def make_optimizer(model, cfg):
    if cfg.get("optimizer", "adamw") != "adamw":
        raise ValueError("Only AdamW is supported")

    lr = float(cfg["learning_rate"])
    wd = float(cfg["weight_decay"])

    if lr <= 0 or wd < 0:
        raise ValueError("Invalid optimizer settings")

    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=wd,
    )


def make_scheduler(optimizer, cfg, epochs):
    if cfg.get("scheduler", "cosine") != "cosine":
        raise ValueError("Only cosine scheduler is supported")

    if epochs < 1:
        raise ValueError("epochs must be >= 1")

    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=float(
            cfg.get("min_learning_rate", 1e-6)
        ),
    )


def make_scaler(device, amp):
    # Float16 benefits from gradient scaling.
    # BF16 typically does not require it.
    enabled = (
        device.type == "cuda"
        and amp["enabled"]
        and amp["dtype"] == torch.float16
    )
    return torch.amp.GradScaler(
        "cuda", enabled=enabled
    )


def run_epoch(
    model,
    loader,
    criterion,
    device,
    amp,
    *,
    optimizer=None,
    scaler=None,
    gradient_clip_norm=1.0,
    max_batches=None,
    log_every=50,
):
    """
    Run one train or validation epoch.

    Returns a valid-pixel-weighted mean loss.
    Using per-batch arithmetic means directly
    would bias results when coverage varies.
    """
    is_training = optimizer is not None

    if is_training and scaler is None:
        raise ValueError(
            "Training requires a gradient scaler"
        )

    model.train(is_training)

    total_weighted_loss = 0.0
    total_valid_pixels = 0
    total_samples = 0
    processed_batches = 0
    skipped_updates = 0

    start_time = time.perf_counter()

    for batch_index, batch in enumerate(loader):
        if (
            max_batches is not None
            and batch_index >= max_batches
        ):
            break

        images = batch["image"].to(
            device, non_blocking=True
        )
        masks = batch["valid_mask"].to(
            device, non_blocking=True
        )

        if images.ndim != 4 or images.shape[1] != 2:
            raise ValueError("Expected [B,2,H,W] images")

        if masks.shape != (
            images.shape[0], 1,
            images.shape[2], images.shape[3]
        ):
            raise ValueError("Invalid mask dimensions")

        valid_pixels = int(
            masks.sum().item()
        )

        if valid_pixels == 0:
            raise ValueError(
                "Batch contains no valid SAR pixels"
            )

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        grad_context = (
            nullcontext()
            if is_training
            else torch.no_grad()
        )

        with grad_context:
            with torch.autocast(
                device_type=device.type,
                dtype=amp["dtype"],
                enabled=amp["enabled"],
            ):
                outputs = model(
                    images, valid_mask=masks
                )

            # Perform reconstruction-loss arithmetic
            # outside autocast, in float32.
            loss = criterion(
                outputs["reconstruction"],
                images,
                masks,
            )

        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Non-finite reconstruction loss"
            )

        if is_training:
            old_scale = scaler.get_scale()

            scaler.scale(loss).backward()

            # Required before clipping with AMP.
            scaler.unscale_(optimizer)

            grad_norm = nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=gradient_clip_norm,
                error_if_nonfinite=False,
            )

            if not torch.isfinite(grad_norm):
                # Skip the update if gradients overflow.
                # GradScaler will adjust its scale.
                if not scaler.is_enabled():
                    raise FloatingPointError(
                        "Non-finite gradients"
                    )

            scaler.step(optimizer)
            scaler.update()

            if (
                scaler.is_enabled()
                and scaler.get_scale() < old_scale
            ):
                skipped_updates += 1

        total_weighted_loss += (
            float(loss.detach().item())
            * valid_pixels
        )
        total_valid_pixels += valid_pixels
        total_samples += images.shape[0]
        processed_batches += 1

        if (
            is_training
            and log_every > 0
            and processed_batches % log_every == 0
        ):
            running_loss = (
                total_weighted_loss
                / total_valid_pixels
            )
            print(
                f"  Batch {processed_batches}: "
                f"loss={running_loss:.6f}"
            )

    if total_valid_pixels == 0:
        raise RuntimeError(
            "No batches processed"
        )

    return {
        "loss": (
            total_weighted_loss
            / total_valid_pixels
        ),
        "batches": processed_batches,
        "samples": total_samples,
        "valid_pixels": total_valid_pixels,
        "skipped_updates": skipped_updates,
        "seconds": (
            time.perf_counter() - start_time
        ),
    }


def save_checkpoint(path, state):
    """
    Save to a temporary file, then replace.
    Avoid partially written checkpoints.
    """
    path = Path(path)
    path.parent.mkdir(
        parents=True, exist_ok=True
    )

    temporary = path.with_name(
        path.name + ".tmp"
    )

    torch.save(state, temporary)
    os.replace(temporary, path)


def load_checkpoint(path, device):
    # Only load checkpoint files you trust.
    return torch.load(
        path,
        map_location=device,
        weights_only=True,
    )


def write_history(path, record):
    path = Path(path)
    path.parent.mkdir(
        parents=True, exist_ok=True
    )

    with path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(record) + "\n"
        )


def train_model(
    *,
    model,
    criterion,
    train_loader,
    val_loader,
    device,
    amp,
    training_cfg,
    checkpoint_dir,
    run_dir,
    run_id,
    normalization_sha256,
    epochs=None,
    max_train_batches=None,
    max_val_batches=None,
    resume=None,
):
    """
    Train and validate the SAR autoencoder.

    Saves:
        last.pt
        best.pt
        history.jsonl
    """
    model = model.to(device)
    criterion = criterion.to(device)

    total_epochs = int(
        epochs or training_cfg["epochs"]
    )

    if total_epochs < 1:
        raise ValueError("epochs must be >= 1")

    optimizer = make_optimizer(
        model, training_cfg
    )
    scheduler = make_scheduler(
        optimizer, training_cfg, total_epochs
    )
    scaler = make_scaler(device, amp)

    checkpoint_dir = Path(checkpoint_dir)
    run_dir = Path(run_dir)

    checkpoint_dir.mkdir(
        parents=True, exist_ok=True
    )
    run_dir.mkdir(
        parents=True, exist_ok=True
    )

    start_epoch = 1
    best_val_loss = math.inf

    if resume is not None:
        state = load_checkpoint(
            resume, device
        )

        if (
            state["normalization_sha256"]
            != normalization_sha256
        ):
            raise RuntimeError(
                "Cannot resume: normalization changed"
            )

        if state["total_epochs"] != total_epochs:
            raise RuntimeError(
                "Cannot resume with a different "
                "total epoch count"
            )

        model.load_state_dict(
            state["model_state"]
        )
        optimizer.load_state_dict(
            state["optimizer_state"]
        )
        scheduler.load_state_dict(
            state["scheduler_state"]
        )
        scaler.load_state_dict(
            state["scaler_state"]
        )

        best_val_loss = float(
            state["best_val_loss"]
        )
        start_epoch = int(
            state["epoch"]
        ) + 1

    history_path = run_dir / "history.jsonl"

    if resume is None and history_path.exists():
        raise FileExistsError(
            f"History already exists: {history_path}"
        )

    if start_epoch > total_epochs:
        raise ValueError(
            "Checkpoint has already completed training"
        )

    print("\nStarting SAR CAE training")
    print("Device:", device)
    print("Epochs:", total_epochs)
    print("Starting epoch:", start_epoch)

    for epoch in range(
        start_epoch, total_epochs + 1
    ):
        # Deterministic per-epoch shuffle with
        # the existing num_workers=0 DataLoader.
        torch.manual_seed(
            int(training_cfg["seed"]) + epoch
        )

        print(
            f"\nEpoch {epoch}/{total_epochs}"
        )

        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            amp,
            optimizer=optimizer,
            scaler=scaler,
            gradient_clip_norm=float(
                training_cfg["gradient_clip_norm"]
            ),
            max_batches=max_train_batches,
            log_every=int(
                training_cfg.get(
                    "log_every_batches", 50
                )
            ),
        )

        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            amp,
            max_batches=max_val_batches,
        )

        # Epoch-level learning-rate scheduling.
        # Record the learning rate used during this epoch.
        lr = optimizer.param_groups[0]["lr"]

        # Prepare the learning rate for the next epoch.
        scheduler.step()
        
        val_loss = float(
            val_metrics["loss"]
        )

        improved = val_loss < best_val_loss

        if improved:
            best_val_loss = val_loss

        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "learning_rate": lr,
            "train_seconds": train_metrics["seconds"],
            "val_seconds": val_metrics["seconds"],
            "train_batches": train_metrics["batches"],
            "val_batches": val_metrics["batches"],
            "skipped_updates": train_metrics[
                "skipped_updates"
            ],
            "partial_run": (
                max_train_batches is not None
                or max_val_batches is not None
            ),
        }

        state = {
            "epoch": epoch,
            "total_epochs": total_epochs,
            "run_id": run_id,
            "normalization_sha256": (
                normalization_sha256
            ),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_val_loss": best_val_loss,
            "training_config": dict(training_cfg),
        }

        save_checkpoint(
            checkpoint_dir / "last.pt",
            state,
        )

        if improved:
            save_checkpoint(
                checkpoint_dir / "best.pt",
                state,
            )

        write_history(
            history_path, record
        )

        print(
            f"  train={record['train_loss']:.6f} "
            f"val={record['val_loss']:.6f} "
            f"best={best_val_loss:.6f} "
            f"lr={lr:.8f}"
        )

    return {
        "run_id": run_id,
        "best_val_loss": best_val_loss,
        "checkpoint_dir": str(checkpoint_dir),
        "history_path": str(history_path),
    }
