"""Placeholder module for SARchaeology AI v1.0."""

import json
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from pyproj import Transformer
from rasterio.windows import Window

from sarchaeology.preprocessing.alignment import (
    FEATURE_NAMES_ALL,
    MASK_NAMES,
    QUALITY_NAMES,
)


def axis_starts(length, size, stride):
    """
    Generate starting positions for fixed-size patches.

    Include one final edge window if the regular
    stride does not land exactly on the boundary.
    """
    if size <= 0 or stride <= 0:
        raise ValueError("Patch size and stride must be positive")

    if stride > size:
        raise ValueError(
            "Stride must not exceed patch size; "
            "otherwise geographic gaps are possible."
        )

    if length < size:
        return []

    starts = list(range(0, length - size + 1, stride))

    last = length - size

    if starts[-1] != last:
        starts.append(last)

    return starts


def iter_patch_windows(width, height, size, stride):
    """
    Yield (row, col, window) for each spatial patch.
    """
    rows = axis_starts(height, size, stride)
    cols = axis_starts(width, size, stride)

    for row in rows:
        for col in cols:
            yield (
                row,
                col,
                Window(col, row, size, size),
            )


def check_grid(source, reference, expected_count, names=None):
    """
    Reject rasters that do not share the exact
    CRS, transform and pixel dimensions.
    """
    if source.count != expected_count:
        raise ValueError(
            f"{source.name}: expected {expected_count} "
            f"bands; found {source.count}"
        )

    if (
        source.crs != reference.crs
        or source.transform != reference.transform
        or source.width != reference.width
        or source.height != reference.height
    ):
        raise ValueError(
            f"Grid mismatch: {source.name}. "
            "Complete Step 1.7 before extracting patches."
        )

    if names is not None:
        if tuple(source.descriptions) != tuple(names):
            raise ValueError(
                f"Band order mismatch in {source.name}: "
                f"{source.descriptions}"
            )


def read_float_patch(source, window):
    """
    Return float32 data with NaN for nodata.
    """
    data = source.read(
        window=window,
        masked=True,
    )

    return np.asarray(
        data.filled(np.nan),
        dtype=np.float32,
    )


def portable_path(path, root):
    """
    Prefer a relative path for reproducibility.
    """
    path = Path(path).resolve()
    root = Path(root).resolve()

    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def write_npz(path, arrays):
    """
    Write an archive through a temporary file,
    then replace the destination.

    Avoid leaving a partially written patch
    if the process is interrupted.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(".npz.partial")

    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                **arrays,
            )

        temporary.replace(path)

    finally:
        if temporary.exists():
            temporary.unlink()


def build_patch_dataset(
    *,
    features_path,
    masks_path,
    quality_path,
    sar_sources,
    optical_sources,
    output_dir,
    manifest_path,
    project_root,
    patch_size=128,
    stride=64,
    min_valid_fraction=0.85,
    max_patches=0,
    overwrite=False,
):
    """
    Generate unlabeled multimodal patches.

    sar_sources:
        [
          {"id": "...", "date": "YYYY-MM-DD",
           "path": Path(...)},
          ...
        ]

    optical_sources:
        [
          {"quarter": "2024Q1", "path": Path(...)},
          ...
        ]

    Each NPZ contains:
        features, masks, quality,
        sar, sar_valid,
        optical, optical_valid
    """

    if not 0 <= min_valid_fraction <= 1:
        raise ValueError(
            "min_valid_fraction must be between 0 and 1"
        )

    if max_patches < 0:
        raise ValueError("max_patches cannot be negative")

    if not sar_sources:
        raise ValueError("No SAR acquisitions supplied")

    if not optical_sources:
        raise ValueError("No optical composites supplied")

    output_dir = Path(output_dir)
    manifest_path = Path(manifest_path)
    project_root = Path(project_root)

    patches_dir = output_dir / "patches"
    patches_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    candidates_checked = 0
    rejected = 0
    cached = 0

    with ExitStack() as stack:
        reference = stack.enter_context(
            rasterio.open(features_path)
        )

        masks_ds = stack.enter_context(
            rasterio.open(masks_path)
        )

        quality_ds = stack.enter_context(
            rasterio.open(quality_path)
        )

        check_grid(
            reference,
            reference,
            19,
            FEATURE_NAMES_ALL,
        )

        check_grid(
            masks_ds,
            reference,
            4,
            MASK_NAMES,
        )

        check_grid(
            quality_ds,
            reference,
            3,
            QUALITY_NAMES,
        )

        sar_datasets = [
            stack.enter_context(
                rasterio.open(source["path"])
            )
            for source in sar_sources
        ]

        optical_datasets = [
            stack.enter_context(
                rasterio.open(source["path"])
            )
            for source in optical_sources
        ]

        for dataset in sar_datasets:
            check_grid(dataset, reference, 2, ("VV", "VH"))

        for dataset in optical_datasets:
            check_grid(
                dataset,
                reference,
                6,
                ("B02", "B03", "B04",
                 "B08", "B11", "B12"),
            )

        if (
            reference.width < patch_size
            or reference.height < patch_size
        ):
            raise ValueError(
                "AOI raster smaller than the patch size"
            )

        converter = Transformer.from_crs(
            reference.crs,
            "EPSG:4326",
            always_xy=True,
        )

        print("Reference grid:", reference.width,
              "x", reference.height)
        print("SAR acquisitions:", len(sar_datasets))
        print("Optical quarters:", len(optical_datasets))
        print("Patch size:", patch_size)
        print("Stride:", stride)
        print("Minimum joint validity:",
              f"{min_valid_fraction:.0%}")

        for row, col, window in iter_patch_windows(
            reference.width,
            reference.height,
            patch_size,
            stride,
        ):
            candidates_checked += 1

            masks = masks_ds.read(window=window)

            # Band 4 = intersection of all modalities.
            joint = masks[3] == 1

            valid_fraction = float(joint.mean())

            if valid_fraction < min_valid_fraction:
                rejected += 1
                continue

            patch_id = f"r{row:05d}_c{col:05d}"

            patch_path = (
                patches_dir / f"{patch_id}.npz"
            )

            if patch_path.exists() and not overwrite:
                with np.load(
                    patch_path,
                    allow_pickle=False,
                ) as existing:
                    required = {
                        "features",
                        "masks",
                        "quality",
                        "sar",
                        "sar_valid",
                        "optical",
                        "optical_valid",
                    }

                    if not required.issubset(
                        existing.files
                    ):
                        raise RuntimeError(
                            f"Incomplete cached patch: "
                            f"{patch_path}"
                        )

                    if (
                        existing["features"].shape
                        != (19, patch_size, patch_size)
                        or existing["sar"].shape[0]
                        != len(sar_datasets)
                        or existing["optical"].shape[0]
                        != len(optical_datasets)
                    ):
                        raise RuntimeError(
                            f"Stale cached patch: "
                            f"{patch_path}. Use --overwrite."
                        )

                cached += 1

            else:
                features = read_float_patch(
                    reference, window
                )

                quality = quality_ds.read(
                    window=window
                ).astype(np.uint16)

                # Joint-valid pixels must contain
                # actual values in all 19 channels.
                if not np.all(
                    np.isfinite(features[:, joint])
                ):
                    raise ValueError(
                        f"Invalid feature data inside "
                        f"joint mask: {patch_id}"
                    )

                # Original SAR acquisitions:
                # [time, 2, H, W]
                sar = np.stack(
                    [
                        read_float_patch(ds, window)
                        for ds in sar_datasets
                    ],
                    axis=0,
                )

                sar_valid = np.all(
                    np.isfinite(sar),
                    axis=1,
                )

                # Optical quarterly reflectance:
                # [quarter, 6, H, W]
                optical = np.stack(
                    [
                        read_float_patch(ds, window)
                        for ds in optical_datasets
                    ],
                    axis=0,
                )

                optical_valid = np.all(
                    np.isfinite(optical),
                    axis=1,
                )

                arrays = {
                    "features": features,
                    "masks": masks.astype(np.uint8),
                    "quality": quality,
                    "sar": sar,
                    "sar_valid": sar_valid.astype(np.uint8),
                    "optical": optical,
                    "optical_valid": optical_valid.astype(np.uint8),
                }

                write_npz(
                    patch_path,
                    arrays,
                )

            xmin, ymin, xmax, ymax = (
                reference.window_bounds(window)
            )

            center_x = (xmin + xmax) / 2
            center_y = (ymin + ymax) / 2

            longitude, latitude = converter.transform(
                center_x,
                center_y,
            )

            rows.append({
                "patch_id": patch_id,
                "path": portable_path(
                    patch_path, project_root
                ),
                "row": row,
                "col": col,
                "height": patch_size,
                "width": patch_size,
                "joint_valid_fraction": valid_fraction,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "center_x": center_x,
                "center_y": center_y,
                "longitude": longitude,
                "latitude": latitude,
                "crs": str(reference.crs),
                "split": "unassigned",
            })

            if (
                max_patches > 0
                and len(rows) >= max_patches
            ):
                break

        metadata = {
            "schema_version": "1.0",
            "dataset_type": "unlabeled_multimodal_patches",
            "crs": str(reference.crs),
            "reference_transform": list(
                reference.transform
            )[:6],
            "reference_width": reference.width,
            "reference_height": reference.height,
            "patch_size": patch_size,
            "stride": stride,
            "minimum_joint_valid_fraction":
                min_valid_fraction,
            "feature_names": list(FEATURE_NAMES_ALL),
            "mask_names": list(MASK_NAMES),
            "quality_names": list(QUALITY_NAMES),
            "sar_sources": [
                {
                    "id": source["id"],
                    "date": source["date"],
                    "path": portable_path(
                        source["path"], project_root
                    ),
                }
                for source in sar_sources
            ],
            "optical_sources": [
                {
                    "quarter": source["quarter"],
                    "path": portable_path(
                        source["path"], project_root
                    ),
                }
                for source in optical_sources
            ],
            "array_layout": {
                "features": "[19,H,W]",
                "masks": "[4,H,W]",
                "quality": "[3,H,W]",
                "sar": "[time,2,H,W]",
                "sar_valid": "[time,H,W]",
                "optical": "[quarter,6,H,W]",
                "optical_valid": "[quarter,H,W]",
            },
            "missing_float_values": "NaN",
            "candidates_checked": candidates_checked,
            "rejected_candidates": rejected,
            "accepted_patches": len(rows),
            "preview_limited": max_patches > 0,
        }

    if not rows:
        raise RuntimeError(
            "No patches passed the joint-valid threshold. "
            "Inspect the Step 1.7 coverage report."
        )

    manifest_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    pd.DataFrame(rows).to_parquet(
        manifest_path,
        index=False,
    )

    metadata_path = (
        output_dir / "dataset_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nPatch generation finished")
    print("Candidates inspected:", candidates_checked)
    print("Rejected by coverage:", rejected)
    print("Accepted patches:", len(rows))
    print("Already cached:", cached)
    print("Manifest:", manifest_path)
    print("Metadata:", metadata_path)

    if max_patches > 0:
        print(
            "Preview run: use --max-patches 0 "
            "for the complete dataset."
        )

    return manifest_path, metadata_path
