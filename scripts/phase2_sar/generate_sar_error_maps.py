
"""
SARchaeology AI — Phase 2.9
Georeferenced SAR reconstruction-error maps.

Outputs:
  mean_abs_error.tif
  std_abs_error.tif
  valid_date_count.tif
  optional date-specific error GeoTIFFs

The score is mean absolute reconstruction error
in normalized VV/VH units, not backscatter dB.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
import torch
import yaml
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sarchaeology.data.sar_dataset import SARDataset
from sarchaeology.models.sar_cae import build_sar_cae
from sarchaeology.preprocessing.sar_normalization import (
    SARNormalizer,
)
from sarchaeology.training.experiment import (
    resolve_amp,
    resolve_device,
    sha256_file,
)


RUN_ID = "sar_cae_20261118T192046Z_2dbe6bd2"
NODATA = -9999.0

PATCH_PATTERN = re.compile(
    r"^r(\d+)_c(\d+)$"
)


def patch_origin(patch_id):
    """
    Extract the row/column position of a patch.

    Example:
        r00704_c00192 -> (704, 192)

    IMPORTANT: This assumes the Phase 1 IDs encode
    top-left pixel offsets in the reference grid.
    """
    match = PATCH_PATTERN.fullmatch(
        str(patch_id)
    )

    if match is None:
        raise ValueError(
            f"Cannot parse patch ID: {patch_id}"
        )

    return (
        int(match.group(1)),
        int(match.group(2)),
    )


def verify_reference(reference_path):
    """
    Read grid metadata and verify the expected
    Buto reference grid properties.
    """
    with rasterio.open(reference_path) as src:
        if src.crs is None:
            raise ValueError(
                "Reference raster has no CRS"
            )

        if src.crs.to_epsg() != 32636:
            raise ValueError(
                "Expected EPSG:32636 reference CRS"
            )

        transform = src.transform

        if not (
            np.isclose(abs(transform.a), 10.0)
            and np.isclose(abs(transform.e), 10.0)
            and np.isclose(transform.b, 0.0)
            and np.isclose(transform.d, 0.0)
        ):
            raise ValueError(
                "Expected north-up 10 m reference grid"
            )

        if transform.a <= 0 or transform.e >= 0:
            raise ValueError(
                "Expected north-up grid orientation"
            )

        profile = src.profile.copy()

        return {
            "width": src.width,
            "height": src.height,
            "transform": transform,
            "crs": src.crs,
            "profile": profile,
        }


def write_geotiff(
    path,
    array,
    grid,
    *,
    nodata,
    tags=None,
):
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    profile = {
        "driver": "GTiff",
        "height": grid["height"],
        "width": grid["width"],
        "count": 1,
        "dtype": array.dtype,
        "crs": grid["crs"],
        "transform": grid["transform"],
        "nodata": nodata,
        "compress": "deflate",
        "tiled": True,
    }

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array, 1)

        if tags:
            dst.update_tags(
                **{
                    key: str(value)
                    for key, value in tags.items()
                }
            )


def make_dataset(config, normalizer, split):
    data_cfg = config["dataset"]

    return SARDataset(
        project_root=ROOT,
        dataset_id=data_cfg["dataset_id"],
        split=split,
        min_valid_fraction=data_cfg[
            "min_valid_fraction"
        ],
        cache_size=data_cfg.get("cache_size", 2),
        normalizer=normalizer,
    )


def group_by_acquisition(dataset):
    """
    Group Dataset indices by acquisition date.

    This scans each sample's metadata using the
    existing Dataset API. It avoids dependence
    on private Dataset index structures.
    """
    grouped = defaultdict(list)

    for index in range(len(dataset)):
        sample = dataset[index]

        date = str(sample["acquisition_date"])
        grouped[date].append(index)

    return dict(sorted(grouped.items()))


def predict_date(
    *,
    model,
    dataset,
    sample_indices,
    grid,
    device,
    amp,
    batch_size,
):
    """
    Produce one date-level error map by averaging
    overlapping patch predictions.

    Returns:
        error_map: float32 [H,W], NaN outside coverage
        coverage: uint16 [H,W], patch contributions
    """
    height = grid["height"]
    width = grid["width"]

    error_sum = np.zeros(
        (height, width),
        dtype=np.float64,
    )

    coverage = np.zeros(
        (height, width),
        dtype=np.uint16,
    )

    loader = DataLoader(
        Subset(dataset, sample_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model.eval()

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
                output = model(
                    images,
                    valid_mask=masks,
                )

            prediction = (
                output["reconstruction"]
                .float()
            )

            target = images.float()
            valid = masks > 0

            # Both VV and VH share the paired
            # SAR validity mask.
            error = (
                prediction - target
            ).abs().mean(
                dim=1
            )

            error = error.cpu().numpy()
            validity = (
                valid[:, 0]
                .cpu()
                .numpy()
            )

            for i, patch_id in enumerate(
                batch["patch_id"]
            ):
                row, col = patch_origin(
                    patch_id
                )

                patch_h, patch_w = error[i].shape

                if (
                    row < 0
                    or col < 0
                    or row + patch_h > height
                    or col + patch_w > width
                ):
                    raise ValueError(
                        f"Patch {patch_id} falls outside "
                        "the reference raster. Check "
                        "the patch origin and grid."
                    )

                window = np.s_[
                    row:row + patch_h,
                    col:col + patch_w,
                ]

                mask = validity[i]

                patch_error = error[i]

                if not np.isfinite(
                    patch_error[mask]
                ).all():
                    raise ValueError(
                        f"Non-finite valid errors in "
                        f"{patch_id}"
                    )

                error_sum[window] += np.where(
                    mask,
                    patch_error,
                    0.0,
                )

                # Valid overlapping patches contribute
                # to the average once each.
                coverage[window] += (
                    mask.astype(np.uint16)
                )

    result = np.full(
        (height, width),
        np.nan,
        dtype=np.float32,
    )

    valid_coverage = coverage > 0

    result[valid_coverage] = (
        error_sum[valid_coverage]
        / coverage[valid_coverage]
    ).astype(np.float32)

    return result, coverage


def aggregate_date(
    error_map,
    global_sum,
    global_sq_sum,
    date_count,
):
    """Update temporal statistics with one date map."""
    valid = np.isfinite(error_map)

    values = np.where(
        valid,
        error_map,
        0.0,
    ).astype(np.float64)

    global_sum += values
    global_sq_sum += values * values
    date_count += valid.astype(np.uint16)


def finalize_maps(
    global_sum,
    global_sq_sum,
    date_count,
):
    """
    Mean and population standard deviation
    across available acquisition dates.
    """
    valid = date_count > 0

    mean = np.full(
        date_count.shape,
        NODATA,
        dtype=np.float32,
    )

    std = np.full(
        date_count.shape,
        NODATA,
        dtype=np.float32,
    )

    counts = date_count[valid].astype(
        np.float64
    )

    means = global_sum[valid] / counts

    variances = (
        global_sq_sum[valid] / counts
        - means * means
    )

    variances = np.maximum(
        variances,
        0.0,
    )

    mean[valid] = means.astype(
        np.float32
    )

    std[valid] = np.sqrt(
        variances
    ).astype(np.float32)

    return mean, std


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-id",
        default=RUN_ID,
    )

    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="val",
    )

    parser.add_argument(
        "--allow-test",
        action="store_true",
    )

    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="Phase 1 aligned 10 m reference GeoTIFF",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--save-dates",
        action="store_true",
    )

    args = parser.parse_args()

    if (
        args.split == "test"
        and not args.allow_test
    ):
        parser.error(
            "Test split requires --allow-test"
        )

    if args.batch_size < 1:
        parser.error(
            "--batch-size must be positive"
        )

    run_dir = (
        ROOT
        / "outputs/experiments/sar/sar_cae"
        / args.run_id
    )

    checkpoint_path = (
        ROOT
        / "checkpoints/sar/sar_cae"
        / args.run_id
        / "best.pt"
    )

    config = yaml.safe_load(
        (run_dir / "config.yaml").read_text(
            encoding="utf-8"
        )
    )

    stats_path = (
        run_dir / "sar_normalization.json"
    )

    normalizer = SARNormalizer.from_json(
        stats_path
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if checkpoint["run_id"] != args.run_id:
        raise RuntimeError(
            "Checkpoint run ID mismatch"
        )

    if (
        checkpoint["normalization_sha256"]
        != sha256_file(stats_path)
    ):
        raise RuntimeError(
            "Normalization fingerprint mismatch"
        )

    dataset_id = config["dataset"]["dataset_id"]
    stats = normalizer.statistics

    if (
        stats["dataset_id"] != dataset_id
        or stats["fit_split"] != "train"
    ):
        raise RuntimeError(
            "Invalid train-fitted normalization"
        )

    train_manifest = (
        ROOT
        / "data/metadata/manifests/splits"
        / dataset_id
        / "train.parquet"
    )

    if (
        sha256_file(train_manifest)
        != stats["source_manifest_sha256"]
    ):
        raise RuntimeError(
            "Training manifest fingerprint mismatch"
        )

    if (
        float(stats["min_valid_fraction"])
        != float(config["dataset"][
            "min_valid_fraction"
        ])
    ):
        raise RuntimeError(
            "SAR validity thresholds disagree"
        )

    grid = verify_reference(
        args.reference
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

    dataset = make_dataset(
        config,
        normalizer,
        args.split,
    )

    grouped = group_by_acquisition(
        dataset
    )

    model = build_sar_cae(
        config
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state"],
        strict=True,
    )

    model.eval()

    height = grid["height"]
    width = grid["width"]

    temporal_sum = np.zeros(
        (height, width),
        dtype=np.float64,
    )

    temporal_sq_sum = np.zeros(
        (height, width),
        dtype=np.float64,
    )

    date_count = np.zeros(
        (height, width),
        dtype=np.uint16,
    )

    output_dir = (
        ROOT
        / "outputs/maps/sar_cae"
        / args.run_id
        / args.split
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nSARchaeology AI — Phase 2.9")
    print("Split:", args.split)
    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Device:", device)
    print("Observations:", len(dataset))
    print("Acquisition dates:", len(grouped))
    print("Grid:", width, "x", height)
    print("CRS:", grid["crs"])

    for date, sample_indices in grouped.items():
        print(
            f"Processing {date}: "
            f"{len(sample_indices)} observations"
        )

        error_map, coverage = predict_date(
            model=model,
            dataset=dataset,
            sample_indices=sample_indices,
            grid=grid,
            device=device,
            amp=amp,
            batch_size=args.batch_size,
        )

        aggregate_date(
            error_map,
            temporal_sum,
            temporal_sq_sum,
            date_count,
        )

        if args.save_dates:
            date_array = np.where(
                np.isfinite(error_map),
                error_map,
                NODATA,
            ).astype(np.float32)

            write_geotiff(
                output_dir
                / "per_date"
                / f"sar_error_{date}.tif",
                date_array,
                grid,
                nodata=NODATA,
                tags={
                    "acquisition_date": date,
                    "score": "normalized_absolute_error",
                    "split": args.split,
                },
            )

    mean, std = finalize_maps(
        temporal_sum,
        temporal_sq_sum,
        date_count,
    )

    common_tags = {
        "model": "SARCAE",
        "run_id": args.run_id,
        "checkpoint_epoch": checkpoint["epoch"],
        "split": args.split,
        "score_units": "normalized_backscatter",
        "temporal_aggregation": "equal_date_mean",
    }

    write_geotiff(
        output_dir / "mean_abs_error.tif",
        mean,
        grid,
        nodata=NODATA,
        tags=common_tags,
    )

    write_geotiff(
        output_dir / "std_abs_error.tif",
        std,
        grid,
        nodata=NODATA,
        tags=common_tags,
    )

    write_geotiff(
        output_dir / "valid_date_count.tif",
        date_count,
        grid,
        nodata=0,
        tags=common_tags,
    )

    covered = date_count > 0

    summary = {
        "run_id": args.run_id,
        "split": args.split,
        "checkpoint_epoch": int(
            checkpoint["epoch"]
        ),
        "dataset_id": dataset_id,
        "crs": str(grid["crs"]),
        "pixel_size_m": 10,
        "reference_raster": str(
            args.reference.resolve()
        ),
        "acquisition_dates": len(grouped),
        "eligible_observations": len(dataset),
        "covered_pixels": int(
            covered.sum()
        ),
        "covered_area_km2": float(
            covered.sum() * 100 / 1_000_000
        ),
        "max_valid_date_count": int(
            date_count.max()
        ),
        "mean_error_over_covered_pixels": (
            float(mean[covered].mean())
            if covered.any()
            else None
        ),
    }

    (output_dir / "summary.json").write_text(
        json.dumps(
            summary, indent=2
        ),
        encoding="utf-8",
    )

    print("\nCOVERAGE")
    print(
        "Covered pixels:",
        summary["covered_pixels"]
    )
    print(
        "Covered area (km²):",
        round(
            summary["covered_area_km2"], 4
        )
    )
    print(
        "Maximum contributing dates:",
        summary["max_valid_date_count"]
    )

    print("\nOutputs:", output_dir)
    print("\nPhase 2.9 map generation completed.")


if __name__ == "__main__":
    main()
