"""SARchaeology AI — Phase 2.16: evaluate best Swin-Tiny checkpoint.

Replays the Phase 2.15 deterministic validation masks (SHA256 seed per
patch/acquisition), checks checkpoint fingerprints, and saves per-observation
and per-patch metrics and example figures. No test split is accessed.

Run from project root:
 uv run python scripts/phase2_sar/evaluate_sar_swin_mae.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from sarchaeology.data.dataloaders import create_sar_dataloaders
from sarchaeology.models.sar_swin_mae import build_sar_swin_mae
from sarchaeology.preprocessing.sar_normalization import SARNormalizer

DEFAULT_RUN = "sar_swin_mae_20261120T165639Z_1cd7f887"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_seed(*parts) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def fixed_mask(patch_ids, dates, height, width, ratio, block_size, seed, device):
    """Exact Phase 2.15 validation mask: randperm on CPU per observation."""
    if height % block_size or width % block_size:
        raise ValueError("Mask block size does not divide image size")
    gh, gw = height // block_size, width // block_size
    count = round(gh * gw * ratio)
    if count < 1:
        raise ValueError("Zero masked blocks")
    result = torch.zeros((len(patch_ids), gh * gw), dtype=torch.bool)
    for i, (patch_id, date) in enumerate(zip(patch_ids, dates)):
        generator = torch.Generator(device="cpu").manual_seed(
            stable_seed(seed, patch_id, date, "validation")
        )
        order = torch.randperm(gh * gw, generator=generator)
        result[i, order[:count]] = True
    return result.reshape(len(patch_ids), 1, gh, gw).repeat_interleave(
        block_size, 2
    ).repeat_interleave(block_size, 3).to(device, non_blocking=True)


def reductions(pred, target, mask, delta):
    """Per-sample per-polarization additive error sums, float32."""
    if pred.shape != target.shape or pred.shape[1] != 2:
        raise ValueError(f"Unexpected image dimensions: {pred.shape} / {target.shape}")
    region = mask.expand_as(pred)
    residual = torch.where(region, pred.float() - target.float(), 0.0)
    abs_error = residual.abs()
    huber = torch.where(abs_error <= delta, 0.5 * residual.square(),
                        delta * (abs_error - 0.5 * delta))
    n = region.sum(dim=(2, 3)).to(torch.float64)
    return {
        "count": n,
        "abs_sum": abs_error.sum(dim=(2, 3)).to(torch.float64),
        "sq_sum": residual.square().sum(dim=(2, 3)).to(torch.float64),
        "huber_sum": huber.sum(dim=(2, 3)).to(torch.float64),
    }


def metrics_from_sums(d):
    n = float(d["count"])
    if n <= 0:
        return {"count": 0, "mae": None, "rmse": None, "huber": None}
    return {
        "count": int(n),
        "mae": float(d["abs_sum"] / n),
        "rmse": float(math.sqrt(d["sq_sum"] / n)),
        "huber": float(d["huber_sum"] / n),
    }


def accumulate(bucket, data):
    for key in ("count", "abs_sum", "sq_sum", "huber_sum"):
        bucket[key] += float(data[key])


def empty():
    return {"count": 0.0, "abs_sum": 0.0, "sq_sum": 0.0, "huber_sum": 0.0}


def save_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path.name}")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_example(path, image, reconstruction, valid, masked, title):
    image = np.asarray(image)
    reconstruction = np.asarray(reconstruction)
    valid = np.asarray(valid, dtype=bool)
    masked = np.asarray(masked, dtype=bool)
    fig, axes = plt.subplots(2, 4, figsize=(13, 7), constrained_layout=True)
    for c, pol in enumerate(("VV", "VH")):
        raw = np.where(valid, image[c], np.nan)
        rec = np.where(valid, reconstruction[c], np.nan)
        partially_hidden = np.where(valid & ~masked, image[c], np.nan)
        error = np.where(valid & masked, abs(reconstruction[c] - image[c]), np.nan)
        values = raw[np.isfinite(raw)]
        low, high = (np.percentile(values, [2, 98]) if len(values) else (-2, 2))
        if high <= low:
            high = low + 1.0
        for j, array in enumerate((raw, partially_hidden, rec, error)):
            opts = dict(cmap="magma" if j == 3 else "gray")
            if j < 3:
                opts.update(vmin=low, vmax=high)
            axes[c, j].imshow(array, **opts)
            axes[c, j].set_title(f"{pol} — {('target', 'masked input', 'prediction', 'masked |error|')[j]}")
            axes[c, j].axis("off")
    fig.suptitle(title)
    fig.savefig(path, dpi=145)
    plt.close(fig)


def locate_source_run(run_id):
    for base in ("outputs/experiments/sar/sar_cae", "outputs/experiments/sar_cae"):
        path = ROOT / base / run_id
        if (path / "config.yaml").is_file() and (path / "sar_normalization.json").is_file():
            return path
    raise FileNotFoundError("Original CAE config and normalizer not found")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--source-run-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Debug only; output marked partial")
    parser.add_argument("--examples", type=int, default=6)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.batch_size < 1 or args.examples < 0:
        parser.error("Invalid batch size/examples")
    torch.manual_seed(42)

    run_dir = ROOT / "outputs/experiments/sar/sar_swin_mae" / args.run_id
    config_path = ROOT / "configs" / "training" / "sar_swin_mae.yaml"  # Must use frozen RUN config, not current YAML.
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing recorded training config: {config_path}")
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    tc, mc, ev = cfg["training"], cfg["model"], cfg["evaluation"]
    if not ev["deterministic_masks"] or mc["architecture"] != "sar_swin_mae":
        raise ValueError("Unexpected Swin training/evaluation configuration")
    source_run = cfg["experiment"]["source_run_id"]
    source_dir = args.source_run_dir.resolve() if args.source_run_dir else locate_source_run(source_run)
    source_cfg_path = source_dir / "config.yaml"
    normalizer_path = source_dir / "sar_normalization.json"
    source_cfg = yaml.safe_load(source_cfg_path.read_text(encoding="utf-8"))
    normalizer = SARNormalizer.from_json(normalizer_path)
    dataset_id = source_cfg["dataset"]["dataset_id"]
    train_manifest = ROOT / "data/metadata/manifests/splits" / dataset_id / "train.parquet"
    val_manifest = train_manifest.with_name("val.parquet")
    expected_fingerprints = {
        "config_sha256": sha256(config_path),
        "source_config_sha256": sha256(source_cfg_path),
        "normalization_sha256": sha256(normalizer_path),
        "train_manifest_sha256": sha256(train_manifest),
        "val_manifest_sha256": sha256(val_manifest),
    }
    checkpoint_path = (args.checkpoint.resolve() if args.checkpoint
                       else ROOT / "checkpoints/sar/sar_swin_mae" / args.run_id / "best.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("run_id") != args.run_id:
        raise RuntimeError("Checkpoint run ID does not match requested run")
    for name, actual in expected_fingerprints.items():
        if checkpoint.get(name) != actual:
            raise RuntimeError(f"Checkpoint fingerprint mismatch: {name}")
    if normalizer.statistics["dataset_id"] != dataset_id or normalizer.statistics["fit_split"] != "train":
        raise RuntimeError("Normalizer is not train-fitted for this dataset")
    if normalizer.statistics["source_manifest_sha256"] != expected_fingerprints["train_manifest_sha256"]:
        raise RuntimeError("Normalizer train-manifest provenance mismatch")

    config_for_loaders = json.loads(json.dumps(source_cfg))
    config_for_loaders["dataloader"]["batch_size"] = args.batch_size
    config_for_loaders["dataloader"]["num_workers"] = 0  # Safe Windows evaluation.
    config_for_loaders["dataloader"]["shuffle_train"] = False
    datasets, loaders = create_sar_dataloaders(
        project_root=ROOT, config=config_for_loaders,
        include_test=False, normalizer=normalizer,
    )
    loader = loaders["val"]
    model = build_sar_swin_mae({"model": mc})
    model.load_state_dict(checkpoint["model_state"], strict=True)
    device = torch.device(args.device)
    model = model.to(device).eval()
    output_dir = ROOT / "outputs/evaluations/sar_swin_mae" / args.run_id / "val"
    examples_dir = output_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    masked_totals = defaultdict(empty)
    full_totals = defaultdict(empty)
    patch_totals = defaultdict(empty)
    patch_obs = defaultdict(int)
    observations = []
    channel_rows = []
    seen_keys = set()
    example_count = 0
    total_batches = 0
    use_amp = bool(tc.get("mixed_precision")) and device.type == "cuda"
    amp_type = torch.float16 if tc.get("amp_dtype") == "float16" else torch.bfloat16
    delta = float(cfg["loss"]["huber_delta"])
    print(f"Evaluating Swin: {args.run_id} / best checkpoint epoch {checkpoint['epoch']}", flush=True)
    print(f"Validation observations (dataset): {len(datasets['val'])}", flush=True)

    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
            x = batch["image"].to(device, non_blocking=True)
            v = batch["valid_mask"].to(device, non_blocking=True).bool()
            b, _, h, w = x.shape
            names = [str(z) for z in batch["patch_id"]]
            dates = [str(z) for z in batch["acquisition_date"]]
            masked = fixed_mask(
                names, dates, h, w,
                float(mc["mask_ratio"]), int(mc["mask_block_size"]),
                int(ev["mask_seed"]), device,
            )
            with torch.autocast(device_type=device.type, dtype=amp_type, enabled=use_amp):
                prediction = model(x, valid_mask=v, synthetic_mask=masked)["reconstruction"]
            # Both reported sets use the SAME masked-input prediction.
            # Full-valid metrics are a diagnostic (NOT unmasked-input CAE equivalent).
            masked_stats = reductions(prediction, x, v & masked, delta)
            full_stats = reductions(prediction, x, v, delta)
            # Move small additive aggregate arrays once per batch.
            m = {k: val.cpu().numpy() for k, val in masked_stats.items()}
            f = {k: val.cpu().numpy() for k, val in full_stats.items()}
            for i in range(b):
                key = (names[i], dates[i])
                if key in seen_keys:
                    raise RuntimeError(f"Duplicate validation observation: {key}")
                seen_keys.add(key)
                observation_m = empty()
                observation_f = empty()
                for channel, pol in enumerate(("VV", "VH")):
                    item_m = {k: float(m[k][i, channel]) for k in m}
                    item_f = {k: float(f[k][i, channel]) for k in f}
                    accumulate(masked_totals[pol], item_m)
                    accumulate(full_totals[pol], item_f)
                    accumulate(observation_m, item_m)
                    accumulate(observation_f, item_f)
                    channel_rows.append({
                        "patch_id": names[i], "acquisition_date": dates[i], "polarization": pol,
                        **{f"masked_{k}": z for k, z in metrics_from_sums(item_m).items()},
                        **{f"full_valid_{k}": z for k, z in metrics_from_sums(item_f).items()},
                    })
                accumulate(masked_totals["overall"], observation_m)
                accumulate(full_totals["overall"], observation_f)
                accumulate(patch_totals[names[i]], observation_m)
                patch_obs[names[i]] += 1
                observations.append({
                    "patch_id": names[i], "acquisition_date": dates[i],
                    **{f"masked_{k}": z for k, z in metrics_from_sums(observation_m).items()},
                    **{f"full_valid_{k}": z for k, z in metrics_from_sums(observation_f).items()},
                })
                if example_count < args.examples:
                    save_example(
                        examples_dir / f"example_{example_count+1:02d}.png",
                        x[i].float().cpu().numpy(), prediction[i].float().cpu().numpy(),
                        v[i, 0].cpu().numpy(), masked[i, 0].cpu().numpy(),
                        f"{names[i]} | {dates[i]} | epoch {checkpoint['epoch']}",
                    )
                    example_count += 1
            total_batches += 1
            if total_batches % 10 == 0:
                print(f"  processed {len(observations)} validation observations", flush=True)

    if not observations:
        raise RuntimeError("No validation observations evaluated")
    if args.max_batches is None and len(observations) != len(datasets["val"]):
        raise RuntimeError("Incomplete validation coverage; refusing to report full evaluation")
    per_patch = [
        {"patch_id": patch_id, "observations": patch_obs[patch_id],
         "masked_count": int(stats["count"]),
         "masked_mae": metrics_from_sums(stats)["mae"],
         "masked_rmse": metrics_from_sums(stats)["rmse"],
         "masked_huber": metrics_from_sums(stats)["huber"]}
        for patch_id, stats in patch_totals.items()
    ]
    per_patch.sort(key=lambda row: row["masked_rmse"], reverse=True)
    save_csv(output_dir / "per_observation.csv", observations)
    save_csv(output_dir / "per_polarization.csv", channel_rows)
    save_csv(output_dir / "per_patch.csv", per_patch)
    report = {
        "phase": "2.16", "run_id": args.run_id,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)) if checkpoint_path.is_relative_to(ROOT) else str(checkpoint_path),
        "best_epoch": int(checkpoint["epoch"]),
        "training_recorded_best_val_masked_huber": float(checkpoint["best_val_loss"]),
        "checkpoint_fingerprints_verified": True,
        "validation_observations": len(observations),
        "validation_patches": len(per_patch),
        "partial": args.max_batches is not None,
        "evaluation_protocol": "fixed validation masks, masked input; masked-only metric vs full-valid metric on the same masked-input output",
        "masked_metrics": {pol: metrics_from_sums(masked_totals[pol]) for pol in ("overall", "VV", "VH")},
        "full_valid_metrics_from_masked_input": {pol: metrics_from_sums(full_totals[pol]) for pol in ("overall", "VV", "VH")},
        "comparison_caveat": "CAE unmasked reconstruction metrics are NOT directly comparable to Swin masked-input metrics. No archaeological labels used.",
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print("\nTop 10 patches by masked RMSE:")
    for row in per_patch[:10]:
        print(f"  {row['patch_id']}: RMSE={row['masked_rmse']:.6f} n={row['observations']}")
    print("\nSaved evaluation:", output_dir)


if __name__ == "__main__":
    main()
