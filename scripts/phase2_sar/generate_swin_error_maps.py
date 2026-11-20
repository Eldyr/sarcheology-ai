"""SARchaeology AI Phase 2.16 — georeferenced Swin validation anomaly maps.

Runs a frozen Swin-Tiny checkpoint under a deterministic ensemble of complementary
block masks. Only masked and originally valid pixels contribute to error maps.
First pass exactly matches Phase 2.15 validation masking. No test data is read.

Usage:
  uv run python scripts/phase2_sar/generate_swin_error_maps.py --max-batches 1
  uv run python scripts/phase2_sar/generate_swin_error_maps.py
  uv run python scripts/phase2_sar/generate_swin_error_maps.py --per-date
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window, bounds as window_bounds
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_sar_swin_mae import (  # local Phase 2.16 evaluator, already validated
    DEFAULT_RUN, locate_source_run, sha256, stable_seed,
)
from sarchaeology.data.dataloaders import create_sar_dataloaders
from sarchaeology.models.sar_swin_mae import build_sar_swin_mae
from sarchaeology.preprocessing.sar_normalization import SARNormalizer

REFERENCE_DEFAULT = "data/processed/aligned/buto_demo_2024_multimodal_10m.tif"


def select_checkpoint_config(checkpoint: dict, run_dir: Path) -> Path:
    """Choose the YAML whose exact bytes match checkpoint config provenance.

    This particular run has an archived YAML mismatch; don't bypass fingerprint checks.
    """
    candidates = [
        run_dir / "config.yaml",
        ROOT / "configs/training/sar_swin_mae.yaml",
        run_dir / "config_checkpoint_verified.yaml",
    ]
    expected = checkpoint["config_sha256"]
    for path in candidates:
        if path.is_file() and sha256(path) == expected:
            return path
    raise RuntimeError("No Swin config matches the saved checkpoint SHA-256")


def complementary_masks(patch_ids, dates, height, width, *, ratio, block_size,
                        seed, passes, device):
    """Yield deterministic masks. Pass 0 = exact training validation mask.

    Over successive passes, choose contiguous, wraparound segments of the SAME
    per-observation randomized block order, with starts distributed across
    the order. This guarantees every block appears in at least one pass if
    passes*round(ratio*nblocks) >= nblocks.
    """
    if height % block_size or width % block_size:
        raise ValueError("Mask block size must divide raster patch size")
    gh, gw = height // block_size, width // block_size
    nblocks = gh * gw
    selected = round(nblocks * ratio)
    if selected < 1 or selected >= nblocks:
        raise ValueError("Expected nonempty partial mask")
    if passes < 2 or passes * selected < nblocks:
        raise ValueError(f"Need more passes for complete coverage: {passes}*{selected} < {nblocks}")
    orders = []
    for patch, date in zip(patch_ids, dates):
        gen = torch.Generator(device="cpu").manual_seed(
            stable_seed(seed, patch, date, "validation")
        )
        orders.append(torch.randperm(nblocks, generator=gen))
    block_indices = torch.arange(selected)
    for pass_index in range(passes):
        offset = (pass_index * nblocks) // passes
        block_mask = torch.zeros((len(orders), nblocks), dtype=torch.bool)
        for i, order in enumerate(orders):
            chosen = order[(block_indices + offset) % nblocks]
            block_mask[i, chosen] = True
        mask = block_mask.reshape(len(orders), 1, gh, gw)
        yield mask.repeat_interleave(block_size, 2).repeat_interleave(
            block_size, 3).to(device, non_blocking=True)


def patch_position(manifest_entry, transform, raster_height, raster_width, *, tolerance=0.05):
    """Verify that patch pixel offsets are consistent with geospatial bounds."""
    row, col = int(manifest_entry["row"]), int(manifest_entry["col"])
    h, w = int(manifest_entry["height"]), int(manifest_entry["width"])
    if row < 0 or col < 0 or row+h > raster_height or col+w > raster_width:
        raise ValueError(f"Out of grid patch at row={row}, col={col} size={h}x{w}")
    if ("xmin" in manifest_entry and "xmax" in manifest_entry and
            "ymin" in manifest_entry and "ymax" in manifest_entry):
        expected = [float(manifest_entry[key]) for key in ("xmin", "ymin", "xmax", "ymax")]
        actual = window_bounds(Window(col, row, w, h), transform)
        if not np.allclose(expected, actual, rtol=0, atol=tolerance):
            raise ValueError(f"Raster/manifest transform mismatch for row={row}, col={col}: "
                             f"{expected} != {actual}")
    return row, col, h, w


class MapAccumulation:
    """Aggregate valid masked errors without treating unscored pixels as zero."""
    def __init__(self, height, width):
        shape = (height, width)
        self.abs_sum = np.zeros(shape, dtype=np.float64)
        self.sq_sum = np.zeros(shape, dtype=np.float64)
        self.count = np.zeros(shape, dtype=np.uint32)

    def add(self, row, col, error, selected):
        """error is float32 [H,W]; selected is bool [H,W]."""
        if error.shape != selected.shape:
            raise ValueError("Error/mask shape mismatch")
        h, w = selected.shape
        a = self.abs_sum[row:row+h, col:col+w]
        s = self.sq_sum[row:row+h, col:col+w]
        n = self.count[row:row+h, col:col+w]
        if a.shape != selected.shape:
            raise ValueError("Patch is outside reference raster")
        if not np.isfinite(error[selected]).all():
            raise ValueError("Non-finite valid masked reconstruction errors")
        absolute = np.abs(error[selected]).astype(np.float64)
        a[selected] += absolute
        s[selected] += absolute * absolute
        n[selected] += 1

    def maps(self):
        covered = self.count > 0
        mean = np.full(self.count.shape, np.nan, dtype=np.float32)
        rmse = np.full(self.count.shape, np.nan, dtype=np.float32)
        mean[covered] = (self.abs_sum[covered] / self.count[covered]).astype(np.float32)
        rmse[covered] = np.sqrt(self.sq_sum[covered] / self.count[covered]).astype(np.float32)
        return mean, rmse


def write_map(path: Path, data, profile, *, nodata):
    target_profile = profile.copy()
    target_profile.update(
        driver="GTiff", count=1, dtype=str(data.dtype), nodata=nodata,
        compress="deflate", predictor=3 if data.dtype.kind == "f" else 2,
        tiled=True, blockxsize=256, blockysize=256,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **target_profile) as dst:
        dst.write(data, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--reference-raster", type=Path, default=Path(REFERENCE_DEFAULT))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--passes", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=None, help="Partial smoke map")
    parser.add_argument("--per-date", action="store_true", help="Also save 29 date-specific maps")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.batch_size < 1 or args.passes < 2 or (args.max_batches is not None and args.max_batches < 1):
        parser.error("Invalid batch size, passes or max-batches")

    run_dir = ROOT / "outputs/experiments/sar/sar_swin_mae" / args.run_id
    checkpoint_path = (args.checkpoint.resolve() if args.checkpoint else
                       ROOT / "checkpoints/sar/sar_swin_mae" / args.run_id / "best.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["run_id"] != args.run_id:
        raise RuntimeError("Checkpoint run ID mismatch")
    config_path = select_checkpoint_config(checkpoint, run_dir)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    tc, mc, ev = config["training"], config["model"], config["evaluation"]
    if not ev["deterministic_masks"]:
        raise RuntimeError("Expected deterministic validation masks")
    source_dir = locate_source_run(config["experiment"]["source_run_id"])
    source_cfg_path, norm_path = source_dir / "config.yaml", source_dir / "sar_normalization.json"
    source_cfg = yaml.safe_load(source_cfg_path.read_text(encoding="utf-8"))
    norm = SARNormalizer.from_json(norm_path)
    dataset_id = source_cfg["dataset"]["dataset_id"]
    split_dir = ROOT / "data/metadata/manifests/splits" / dataset_id
    train_path, val_path = split_dir / "train.parquet", split_dir / "val.parquet"
    checks = {
        "config_sha256": sha256(config_path),
        "source_config_sha256": sha256(source_cfg_path),
        "normalization_sha256": sha256(norm_path),
        "train_manifest_sha256": sha256(train_path),
        "val_manifest_sha256": sha256(val_path),
    }
    for name, value in checks.items():
        if checkpoint.get(name) != value:
            raise RuntimeError(f"Checkpoint fingerprint mismatch: {name}")
    if norm.statistics["dataset_id"] != dataset_id or norm.statistics["fit_split"] != "train":
        raise RuntimeError("Normalizer does not correspond to train split")
    if norm.statistics["source_manifest_sha256"] != checks["train_manifest_sha256"]:
        raise RuntimeError("Normalizer train-manifest provenance mismatch")

    import pandas as pd
    val_manifest = pd.read_parquet(val_path)
    val_lookup = val_manifest.set_index("patch_id").to_dict("index")
    reference_path = args.reference_raster if args.reference_raster.is_absolute() else ROOT / args.reference_raster
    with rasterio.open(reference_path) as ref:
        profile = ref.profile.copy()
        transform = ref.transform
        raster_height, raster_width = ref.height, ref.width
        if str(ref.crs) != str(val_manifest.iloc[0]["crs"]):
            raise RuntimeError(f"Reference CRS {ref.crs} != manifest CRS {val_manifest.iloc[0]['crs']}")
        if not np.isclose(abs(transform.a), 10) or not np.isclose(abs(transform.e), 10):
            raise RuntimeError(f"Expected 10-meter reference grid, got {transform}")
    for patch, entry in val_lookup.items():
        patch_position(entry, transform, raster_height, raster_width)

    loader_cfg = json.loads(json.dumps(source_cfg))
    loader_cfg["dataloader"]["batch_size"] = args.batch_size
    loader_cfg["dataloader"]["num_workers"] = 0
    datasets, loaders = create_sar_dataloaders(
        project_root=ROOT, config=loader_cfg, include_test=False, normalizer=norm
    )
    loader = loaders["val"]
    model = build_sar_swin_mae({"model": mc})
    model.load_state_dict(checkpoint["model_state"], strict=True)
    device = torch.device(args.device)
    model.to(device).eval()
    amp_on = bool(tc.get("mixed_precision")) and device.type == "cuda"
    amp_dtype = torch.float16 if tc.get("amp_dtype") == "float16" else torch.bfloat16
    mean_layers = {"VV": MapAccumulation(raster_height, raster_width),
                   "VH": MapAccumulation(raster_height, raster_width)}
    date_layers = {} if args.per_date else None
    processed = set()
    total_predictions = 0
    full = args.max_batches is None
    output_dir = ROOT / "outputs/maps/sar_swin_mae" / args.run_id / ("val" if full else "val_smoke")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Swin Phase 2.16 maps | epoch {checkpoint['epoch']} | {len(datasets['val'])} observations | "
          f"passes={args.passes} | config={config_path}", flush=True)
    with torch.inference_mode():
        for batch_no, batch in enumerate(loader):
            if args.max_batches is not None and batch_no >= args.max_batches:
                break
            images = batch["image"].to(device, non_blocking=True)
            valid = batch["valid_mask"].to(device, non_blocking=True).bool()
            b, ch, h, w = images.shape
            names = [str(s) for s in batch["patch_id"]]
            dates = [str(s) for s in batch["acquisition_date"]]
            if ch != 2:
                raise ValueError("Expected VV and VH images")
            for name, date in zip(names, dates):
                if (name, date) in processed:
                    raise RuntimeError(f"Duplicate observation: {name} {date}")
                processed.add((name, date))
            # Ensures date metadata covers only known validation patches.
            positions = [patch_position(val_lookup[name], transform, raster_height, raster_width)
                         for name in names]
            for synthetic in complementary_masks(
                names, dates, h, w, ratio=float(mc["mask_ratio"]),
                block_size=int(mc["mask_block_size"]), seed=int(ev["mask_seed"]),
                passes=args.passes, device=device
            ):
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_on):
                    result = model(images, valid_mask=valid, synthetic_mask=synthetic)
                reconstruction = result["reconstruction"].float()
                masked_valid = (valid & synthetic)[:, 0].cpu().numpy().astype(bool)
                residual = (reconstruction - images.float()).cpu().numpy()
                for i, (name, date) in enumerate(zip(names, dates)):
                    row, col, hh, ww = positions[i]
                    if (hh, ww) != (h, w):
                        raise ValueError(f"Patch size mismatch: {name}")
                    for channel, pol in enumerate(("VV", "VH")):
                        mean_layers[pol].add(row, col, residual[i, channel], masked_valid[i])
                        if date_layers is not None:
                            date_layers.setdefault(date, {
                                "VV": MapAccumulation(raster_height, raster_width),
                                "VH": MapAccumulation(raster_height, raster_width),
                            })[pol].add(row, col, residual[i, channel], masked_valid[i])
                total_predictions += b
            if (batch_no + 1) % 5 == 0:
                print(f"  processed {len(processed)} unique observations", flush=True)

    if not processed:
        raise RuntimeError("No validation observations processed")
    if full and len(processed) != len(datasets["val"]):
        raise RuntimeError("Incomplete validation map; refusing to label as full")

    def write_group(layers, out):
        for pol, accumulator in layers.items():
            mean, rmse = accumulator.maps()
            write_map(out / f"{pol.lower()}_mean_abs_error.tif", mean, profile, nodata=np.nan)
            write_map(out / f"{pol.lower()}_rmse.tif", rmse, profile, nodata=np.nan)
            write_map(out / f"{pol.lower()}_prediction_count.tif", accumulator.count, profile, nodata=0)
        # Two-channel aggregate: every SAR-valid spatial pixel has two pols.
        combined_count = layers["VV"].count.astype(np.uint64) + layers["VH"].count.astype(np.uint64)
        total_abs = layers["VV"].abs_sum + layers["VH"].abs_sum
        total_sq = layers["VV"].sq_sum + layers["VH"].sq_sum
        coverage = combined_count > 0
        mean = np.full(combined_count.shape, np.nan, dtype=np.float32)
        rmse = mean.copy()
        mean[coverage] = (total_abs[coverage] / combined_count[coverage]).astype(np.float32)
        rmse[coverage] = np.sqrt(total_sq[coverage] / combined_count[coverage]).astype(np.float32)
        # Across 4 complementary passes and 2 pols count can exceed 2*29 per pixel.
        if combined_count.max() > np.iinfo(np.uint32).max:
            raise OverflowError("Prediction count exceeds uint32")
        write_map(out / "mean_abs_error.tif", mean, profile, nodata=np.nan)
        write_map(out / "rmse.tif", rmse, profile, nodata=np.nan)
        write_map(out / "prediction_count.tif", combined_count.astype(np.uint32), profile, nodata=0)
        write_map(out / "coverage.tif", coverage.astype(np.uint8), profile, nodata=0)
        return mean, rmse, combined_count, coverage

    mean, rmse, count, coverage = write_group(mean_layers, output_dir)
    if date_layers is not None:
        for date, accumulator in sorted(date_layers.items()):
            write_group(accumulator, output_dir / "by_date" / date.replace(":", "-"))

    valid_coverage = int(coverage.sum())
    report = {
        "phase": "2.16-geotiff", "run_id": args.run_id,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_fingerprints_verified": True,
        "config_path": str(config_path.relative_to(ROOT)),
        "evaluation_mode": "complementary deterministic masked-input ensemble",
        "first_pass_reproduces_training_validation_masks": True,
        "mask_passes_per_observation": args.passes,
        "fully_covered_within_each_observation_valid_mask": True,
        "image_size": [raster_height, raster_width], "crs": str(profile["crs"]),
        "pixel_resolution_m": float(abs(transform.a)),
        "validation_observations": len(processed),
        "expected_validation_observations": len(datasets["val"]),
        "partial": not full, "per_date": bool(args.per_date),
        "covered_pixels": valid_coverage,
        "covered_area_km2": valid_coverage * abs(transform.a * transform.e) / 1e6,
        "average_error_on_covered_pixels": float(np.mean(mean[coverage])),
        "average_rmse_on_covered_pixels": float(np.mean(rmse[coverage])),
        "max_prediction_count": int(count.max()),
        "output_dir": str(output_dir),
        "interpretation": "These are masked-input reconstruction error maps, not archaeological detection probabilities. "
                         "Repeated observations/passes/overlapping patches contribute to each pixel's mean.",
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print("Saved GeoTIFFs:", output_dir, flush=True)


if __name__ == "__main__":
    main()
