"""SARchaeology AI Phase 2.15 — optimized Swin SAR masked reconstruction trainer.

Uses the same train/val datasets and train-fitted normalizer as the CAE.
The test split is never loaded. Windows-compatible entry point.

Examples:
  uv run python scripts/phase2_sar/train_sar_swin_mae.py --smoke
  uv run python scripts/phase2_sar/train_sar_swin_mae.py --benchmark --benchmark-batches 20
  uv run python scripts/phase2_sar/train_sar_swin_mae.py
  uv run python scripts/phase2_sar/train_sar_swin_mae.py --resume checkpoints/sar/sar_swin_mae/RUN_ID/last.pt
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import math
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
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sarchaeology.data.dataloaders import create_sar_dataloaders
from sarchaeology.models.sar_swin_mae import build_sar_swin_mae
from sarchaeology.preprocessing.sar_normalization import SARNormalizer

DEFAULT_CONFIG = ROOT / "configs/training/sar_swin_mae.yaml"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_seed(*parts) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def seed_everything(seed: int, strict: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = strict
    torch.backends.cudnn.benchmark = not strict
    torch.use_deterministic_algorithms(strict)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")


def make_mask(batch: dict, *, mask_ratio: float, block_size: int,
              seed: int, epoch: int, validation: bool, device: torch.device) -> torch.Tensor:
    """Stable per-observation mask; validation independent of epoch/batch order."""
    image = batch["image"]
    b, channels, h, w = image.shape
    if channels != 2 or h % block_size or w % block_size:
        raise ValueError(f"Unsupported SAR shape {image.shape} / block size {block_size}")
    if not (0 < mask_ratio < 1):
        raise ValueError("mask_ratio must be between 0 and 1")
    gh, gw = h // block_size, w // block_size
    n = gh * gw
    masked = round(n * mask_ratio)
    if masked == 0 or masked == n:
        raise ValueError("mask ratio rounds to an empty/full mask")
    result = torch.zeros((b, n), dtype=torch.bool)
    for i in range(b):
        s = stable_seed(seed, batch["patch_id"][i], batch["acquisition_date"][i],
                        "validation" if validation else epoch)
        generator = torch.Generator(device="cpu").manual_seed(s)
        order = torch.randperm(n, generator=generator)
        result[i, order[:masked]] = True
    return result.reshape(b, 1, gh, gw).repeat_interleave(
        block_size, 2).repeat_interleave(block_size, 3).to(device, non_blocking=True)


def masked_huber_stats(prediction: torch.Tensor, target: torch.Tensor,
                       valid_mask: torch.Tensor, synthetic_mask: torch.Tensor,
                       *, delta: float):
    """Differentiable masked Huber and detached GPU aggregate tensors; no .item() sync."""
    region = ((valid_mask > 0) & synthetic_mask).expand_as(prediction)
    # `where` eliminates invalid NaN targets from computations and gradients.
    diff = torch.where(region, prediction.float() - target.float(), 0.0)
    abs_diff = diff.abs()
    huber = torch.where(abs_diff <= delta, 0.5 * diff.square(),
                        delta * (abs_diff - 0.5 * delta))
    count = region.sum()
    loss = huber.sum() / count.clamp_min(1)
    stats = {
        "huber_sum": huber.sum().detach(),
        "abs_sum": abs_diff.sum().detach(),
        "sq_sum": diff.square().sum().detach(),
        "count": count.detach(),
    }
    return loss, stats


def _safe_worker_init(worker_id: int) -> None:
    # DataLoader gives each worker a distinct torch initial seed.
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def make_loaders(source_config: dict, normalizer: SARNormalizer, settings: dict):
    """Recreate loaders using the original datasets and collate functions.

    The existing factory is the single authority for geographic splits and
    normalization. This wrapper exposes pinned-memory/prefetch options even if
    that factory ignores them. No test dataset is requested.
    """
    config = copy.deepcopy(source_config)
    config.setdefault("dataloader", {})["batch_size"] = int(settings["batch_size"])
    config["dataloader"]["num_workers"] = 0  # Factory creates only lightweight reference loaders.
    datasets, originals = create_sar_dataloaders(
        project_root=ROOT, config=config, include_test=False, normalizer=normalizer)
    num_workers = int(settings["num_workers"])
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    kwargs = {
        "batch_size": int(settings["batch_size"]),
        "num_workers": num_workers,
        "pin_memory": bool(settings.get("pin_memory", True)) and torch.cuda.is_available(),
        "worker_init_fn": _safe_worker_init if num_workers else None,
    }
    if num_workers:
        kwargs["persistent_workers"] = bool(settings.get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(settings.get("prefetch_factor", 2))
    loaders = {}
    for split in ("train", "val"):
        reference = originals[split]
        loaders[split] = DataLoader(
            datasets[split],
            shuffle=(split == "train"),
            drop_last=False,
            collate_fn=reference.collate_fn,
            generator=torch.Generator().manual_seed(int(settings["seed"]) + (0 if split == "train" else 100000)),
            **kwargs,
        )
    return datasets, loaders


def epoch_metrics(sums: dict) -> dict:
    # Exactly one host transfer per metric at end of epoch.
    vals = {k: v.item() for k, v in sums.items()}
    n = vals["count"]
    if n <= 0:
        raise RuntimeError("No valid synthetically masked channel-pixels")
    return {
        "masked_huber": vals["huber_sum"] / n,
        "masked_mae": vals["abs_sum"] / n,
        "masked_rmse": math.sqrt(vals["sq_sum"] / n),
        "masked_channel_pixels": int(n),
    }


def run_epoch(*, model, loader, device, optimizer, scaler, epoch: int,
              settings: dict, mask_seed: int, training: bool,
              max_batches: int | None = None, log_every: int = 0):
    model.train(training)
    totals = {name: torch.zeros((), dtype=torch.float64, device=device)
              for name in ("huber_sum", "abs_sum", "sq_sum", "count")}
    num_batches = 0
    num_observations = 0
    amp_enabled = bool(settings["mixed_precision"]) and device.type == "cuda"
    amp_dtype = torch.float16 if settings["amp_dtype"] == "float16" else torch.bfloat16
    clip = float(settings["gradient_clip_norm"])
    started = time.perf_counter()
    for batch in loader:
        if max_batches is not None and num_batches >= max_batches:
            break
        image = batch["image"].to(device, non_blocking=True)
        valid = batch["valid_mask"].to(device, non_blocking=True)
        synthetic_mask = make_mask(
            batch, mask_ratio=float(settings["mask_ratio"]),
            block_size=int(settings["mask_block_size"]),
            seed=mask_seed, epoch=epoch, validation=not training, device=device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                result = model(image, valid_mask=valid, synthetic_mask=synthetic_mask)
            loss, components = masked_huber_stats(
                result["reconstruction"], image, valid, synthetic_mask,
                delta=float(settings["huber_delta"]))
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip,
                                                error_if_nonfinite=False)
                scaler.step(optimizer)
                scaler.update()
        for k, value in components.items():
            totals[k].add_(value.to(dtype=torch.float64))
        num_batches += 1
        num_observations += int(image.shape[0])
        if training and log_every and num_batches % log_every == 0:
            # One sync only for periodic progress logging.
            print(f"  epoch={epoch} batches={num_batches} "
                  f"masked_huber={(totals['huber_sum']/totals['count'].clamp_min(1)).item():.6f}",
                  flush=True)
    if num_batches == 0:
        raise RuntimeError("DataLoader produced no observations")
    metrics = epoch_metrics(totals)
    if not all(math.isfinite(metrics[k]) for k in ("masked_huber", "masked_mae", "masked_rmse")):
        raise RuntimeError(f"Non-finite epoch metrics: {metrics}")
    metrics["batches"] = num_batches
    metrics["observations"] = num_observations
    metrics["seconds"] = time.perf_counter() - started
    metrics["observations_per_second"] = num_observations / max(metrics["seconds"], 1e-12)
    return metrics


def save_checkpoint(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        torch.save(data, temp)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def locate_source_run(run_id: str) -> Path:
    paths = [
        ROOT / "outputs/experiments/sar/sar_cae" / run_id,
        ROOT / "outputs/experiments/sar_cae" / run_id,
    ]
    for path in paths:
        if (path / "config.yaml").exists() and (path / "sar_normalization.json").exists():
            return path
    raise FileNotFoundError("Cannot locate CAE config.yaml and sar_normalization.json. "
                            f"Checked: {', '.join(map(str, paths))}. "
                            "Pass --source-run-dir explicitly.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-run-dir", type=Path, default=None,
                        help="CAE run directory containing config.yaml and sar_normalization.json")
    parser.add_argument("--smoke", action="store_true", help="One real train + validation batch, no writes")
    parser.add_argument("--benchmark", action="store_true", help="Time limited train + val batches, no writes")
    parser.add_argument("--benchmark-batches", type=int, default=20)
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()
    if sum(bool(x) for x in (args.smoke, args.benchmark, args.resume is not None)) > 1:
        parser.error("--smoke, --benchmark, and --resume are mutually exclusive")
    cfg_path = args.config.resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    tc, mc, ec = cfg["training"], cfg["model"], cfg["experiment"]
    loss_cfg, val_cfg = cfg["loss"], cfg["evaluation"]
    if mc["architecture"] != "sar_swin_mae" or mc.get("pretrained"):
        raise ValueError("Expected from-scratch sar_swin_mae")
    if loss_cfg["type"] != "huber" or loss_cfg["masked_region_weight"] != 1.0 or loss_cfg["visible_region_weight"] != 0.0:
        raise ValueError("Trainer supports masked-only Huber")
    if not val_cfg["deterministic_masks"]:
        raise ValueError("Validation masks must be deterministic")
    epochs = int(tc["epochs"])
    if epochs < 1 or int(tc["batch_size"]) < 1:
        raise ValueError("Epochs and batch size must be positive")
    seed_everything(int(tc["seed"]), strict=bool(tc.get("deterministic", False)))
    device = torch.device(tc["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    source_dir = (args.source_run_dir.resolve() if args.source_run_dir
                  else locate_source_run(ec["source_run_id"]))
    source_cfg_path = source_dir / "config.yaml"
    norm_path = source_dir / "sar_normalization.json"
    source_cfg = yaml.safe_load(source_cfg_path.read_text(encoding="utf-8"))
    normalizer = SARNormalizer.from_json(norm_path)
    if normalizer.statistics["fit_split"] != "train":
        raise RuntimeError("The SAR normalizer must be fitted using training data only")
    dataset_id = source_cfg["dataset"]["dataset_id"]
    if normalizer.statistics["dataset_id"] != dataset_id:
        raise RuntimeError("Dataset and normalization IDs differ")
    train_manifest = ROOT / "data/metadata/manifests/splits" / dataset_id / "train.parquet"
    val_manifest = train_manifest.with_name("val.parquet")
    fingerprints = {
        "config_sha256": sha256(cfg_path),
        "source_config_sha256": sha256(source_cfg_path),
        "normalization_sha256": sha256(norm_path),
        "train_manifest_sha256": sha256(train_manifest),
        "val_manifest_sha256": sha256(val_manifest),
    }
    # Source manifests must match the trained normalizer's recorded provenance.
    if normalizer.statistics["source_manifest_sha256"] != fingerprints["train_manifest_sha256"]:
        raise RuntimeError("Training manifest differs from normalizer's source manifest")
    datasets, loaders = make_loaders(source_cfg, normalizer, tc)
    print("\nSARchaeology AI — Phase 2.15 optimized trainer")
    print(f"Device: {device}; train: {len(datasets['train'])}; val: {len(datasets['val'])}")
    print(f"Batch size: {tc['batch_size']}; workers: {tc['num_workers']}; "
          f"pinned memory: {tc.get('pin_memory', True)}", flush=True)
    model = build_sar_swin_mae({"model": mc}).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(tc["learning_rate"]),
                                  weight_decay=float(tc["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=float(tc["min_learning_rate"]))
    use_fp16_scaler = bool(tc["mixed_precision"]) and device.type == "cuda" and tc["amp_dtype"] == "float16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16_scaler)
    settings = {**tc, "mask_ratio": float(mc["mask_ratio"]),
                "mask_block_size": int(mc["mask_block_size"]),
                "huber_delta": float(loss_cfg["huber_delta"])}
    start_epoch, best_val = 1, float("inf")
    if args.resume:
        state = torch.load(args.resume.resolve(), map_location="cpu", weights_only=True)
        for k, expected in fingerprints.items():
            if state.get(k) != expected:
                raise RuntimeError(f"Cannot resume: {k} changed. Start a fresh run instead.")
        run_id = state["run_id"]
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        scaler.load_state_dict(state["scaler_state"])
        start_epoch = int(state["epoch"]) + 1
        best_val = float(state["best_val_loss"])
        print(f"Resuming {run_id} at epoch {start_epoch}", flush=True)
    else:
        run_id = "sar_swin_mae_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    out_dir = ROOT / ec["output_directory"] / run_id
    ckpt_dir = ROOT / ec["checkpoint_directory"] / run_id
    transient = args.smoke or args.benchmark
    if not transient:
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        if not args.resume:
            (out_dir / "config.yaml").write_text(cfg_path.read_text(encoding="utf-8"), encoding="utf-8")
            metadata = {"run_id": run_id, "source_run_id": ec["source_run_id"],
                        "torch_version": torch.__version__, "device": str(device), **fingerprints}
            (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if start_epoch > epochs:
        print("All requested epochs already completed.")
        return
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    if args.smoke or args.benchmark:
        print("TRANSIENT MODE: no files or checkpoints will be written.", flush=True)
    try:
        for epoch in (range(1, 2) if transient else range(start_epoch, epochs + 1)):
            seed_everything(int(tc["seed"]) + epoch,
                            strict=bool(tc.get("deterministic", False)))
            # Critical for exact batch order when resuming at an epoch boundary.
            loaders["train"].generator.manual_seed(stable_seed(tc["seed"], "train_loader", epoch))
            loaders["val"].generator.manual_seed(stable_seed(tc["seed"], "val_loader", epoch))
            limit = 1 if args.smoke else args.benchmark_batches if args.benchmark else None
            t0 = time.perf_counter()
            tr = run_epoch(model=model, loader=loaders["train"], device=device,
                           optimizer=optimizer, scaler=scaler, training=True, epoch=epoch,
                           settings=settings, mask_seed=int(tc["seed"]), max_batches=limit,
                           log_every=int(tc.get("log_every_batches", 0)) if not transient else 0)
            va = run_epoch(model=model, loader=loaders["val"], device=device,
                           optimizer=None, scaler=scaler, training=False, epoch=epoch,
                           settings=settings, mask_seed=int(val_cfg["mask_seed"]),
                           max_batches=1 if args.smoke else limit)
            val_loss = va["masked_huber"]
            improved = val_loss < best_val
            if improved:
                best_val = val_loss
            lr = float(optimizer.param_groups[0]["lr"])
            if not transient:
                scheduler.step()
            elapsed = time.perf_counter() - t0
            row = {"epoch": epoch, "train_masked_huber": tr["masked_huber"],
                   "train_masked_mae": tr["masked_mae"], "train_masked_rmse": tr["masked_rmse"],
                   "val_masked_huber": va["masked_huber"], "val_masked_mae": va["masked_mae"],
                   "val_masked_rmse": va["masked_rmse"], "best_val_loss": best_val,
                   "lr": lr, "epoch_seconds": elapsed,
                   "train_batches": tr["batches"], "val_batches": va["batches"],
                   "train_seconds": tr["seconds"], "val_seconds": va["seconds"]}
            print(f"Epoch {epoch}/{epochs} train={tr['masked_huber']:.6f} "
                  f"val={val_loss:.6f} elapsed={elapsed:.1f}s "
                  f"train_batches={tr['batches']} val_batches={va['batches']} "
                  f"lr={lr:.2e}", flush=True)
            if device.type == "cuda":
                print(f"  GPU peak allocated: {torch.cuda.max_memory_allocated()/1024**3:.2f} GiB", flush=True)
            if transient:
                print("SMOKE PASSED" if args.smoke else "BENCHMARK PASSED")
                break
            with (out_dir / "history.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            state = {"epoch": epoch, "run_id": run_id, "best_val_loss": best_val,
                     "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                     "scheduler_state": scheduler.state_dict(), "scaler_state": scaler.state_dict(),
                     "last_metrics": row, **fingerprints}
            save_checkpoint(ckpt_dir / "last.pt", state)
            if improved:
                save_checkpoint(ckpt_dir / "best.pt", state)
    finally:
        # Shuts down persistent Windows worker processes when training ends.
        del loaders
    if not transient:
        print(f"Run ID: {run_id}\nBest validation Huber: {best_val:.6f}\nCheckpoints: {ckpt_dir}")


if __name__ == "__main__":
    main()
