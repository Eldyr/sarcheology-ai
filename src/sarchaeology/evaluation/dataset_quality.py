
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from affine import Affine

from sarchaeology.evaluation.spatial_splits import (
    verify_separation,
)

from sarchaeology.preprocessing.alignment import (
    FEATURE_NAMES_ALL,
    MASK_NAMES,
    QUALITY_NAMES,
)


SPLITS = ("train", "val", "test", "excluded")

REQUIRED_ARRAYS = (
    "features",
    "masks",
    "quality",
    "sar",
    "sar_valid",
    "optical",
    "optical_valid",
)


def inspect_patch_archive(
    path,
    record,
    metadata,
    minimum_valid_fraction=0.85,
):
    """
    Validate one NPZ archive and its geographic
    entry in the patch manifest.

    Raises ValueError for failed integrity checks.
    """
    path = Path(path)
    size = int(metadata["patch_size"])
    n_sar = len(metadata["sar_sources"])
    n_optical = len(metadata["optical_sources"])

    expected_shapes = {
        "features": (19, size, size),
        "masks": (4, size, size),
        "quality": (3, size, size),
        "sar": (n_sar, 2, size, size),
        "sar_valid": (n_sar, size, size),
        "optical": (n_optical, 6, size, size),
        "optical_valid": (n_optical, size, size),
    }

    with np.load(path, allow_pickle=False) as archive:
        missing = set(REQUIRED_ARRAYS) - set(archive.files)

        if missing:
            raise ValueError(
                f"Missing arrays: {sorted(missing)}"
            )

        arrays = {
            name: archive[name]
            for name in REQUIRED_ARRAYS
        }

    for name, expected in expected_shapes.items():
        if arrays[name].shape != expected:
            raise ValueError(
                f"{name} shape {arrays[name].shape}; "
                f"expected {expected}"
            )

    features = arrays["features"]
    masks = arrays["masks"]
    quality = arrays["quality"]
    sar = arrays["sar"]
    sar_valid = arrays["sar_valid"]
    optical = arrays["optical"]
    optical_valid = arrays["optical_valid"]

    for name in ("features", "sar", "optical"):
        if not np.issubdtype(
            arrays[name].dtype,
            np.floating,
        ):
            raise ValueError(f"{name} must be floating-point")

        if np.isinf(arrays[name]).any():
            raise ValueError(f"{name} contains infinity")

    if masks.dtype != np.uint8:
        raise ValueError("masks must be uint8")

    if quality.dtype != np.uint16:
        raise ValueError("quality must be uint16")

    for name in ("masks", "sar_valid", "optical_valid"):
        if not np.isin(arrays[name], [0, 1]).all():
            raise ValueError(
                f"{name} contains values other than 0/1"
            )

    # Verify joint validity is exactly the intersection.
    expected_joint = np.all(
        masks[:3] == 1,
        axis=0,
    )

    if not np.array_equal(
        masks[3] == 1,
        expected_joint,
    ):
        raise ValueError("Joint validity mask is inconsistent")

    fraction = float(expected_joint.mean())

    if fraction + 1e-6 < minimum_valid_fraction:
        raise ValueError(
            f"Patch coverage {fraction:.4f} is too low"
        )

    if not np.isclose(
        fraction,
        float(record["joint_valid_fraction"]),
        atol=1e-6,
        rtol=0,
    ):
        raise ValueError(
            "Manifest and archive validity disagree"
        )

    # All 19 engineered features must exist
    # inside the joint-valid region.
    joint_features = features[:, expected_joint]

    if (
        not np.isfinite(joint_features).all()
        or np.any(joint_features == -9999.0)
    ):
        raise ValueError(
            "Invalid engineered features inside joint mask"
        )

    # The feature validity must also respect
    # the separate modality masks.
    channel_ranges = (
        (0, 7),
        (7, 14),
        (14, 19),
    )

    for index, (start, end) in enumerate(channel_ranges):
        modality_valid = masks[index] == 1
        values = features[start:end]

        if not np.isfinite(
            values[:, modality_valid]
        ).all():
            raise ValueError(
                f"Missing features inside modality mask {index}"
            )

        if np.isfinite(
            values[:, ~modality_valid]
        ).any():
            raise ValueError(
                f"Features not masked outside modality {index}"
            )

    # Recompute individual-acquisition masks.
    calculated_sar_valid = np.all(
        np.isfinite(sar),
        axis=1,
    )

    calculated_optical_valid = np.all(
        np.isfinite(optical),
        axis=1,
    )

    if not np.array_equal(
        sar_valid.astype(bool),
        calculated_sar_valid,
    ):
        raise ValueError("SAR acquisition masks are inconsistent")

    if not np.array_equal(
        optical_valid.astype(bool),
        calculated_optical_valid,
    ):
        raise ValueError("Optical quarter masks are inconsistent")

    # Check temporal observation-count consistency.
    calculated_sar_count = calculated_sar_valid.sum(
        axis=0
    )

    if not np.array_equal(
        quality[0],
        calculated_sar_count,
    ):
        raise ValueError(
            "SAR observation counts do not match the time series"
        )

    if np.any(
        quality[1] > calculated_optical_valid.sum(axis=0)
    ):
        raise ValueError(
            "Optical valid-quarter count exceeds available observations"
        )

    if not np.array_equal(
        quality[2] > 0,
        masks[2] == 1,
    ):
        raise ValueError(
            "Terrain availability and terrain mask disagree"
        )

    # Verify geographic coordinates from the
    # original reference raster transform.
    transform = Affine(
        *metadata["reference_transform"]
    )

    row = int(record["row"])
    col = int(record["col"])

    x0, y0 = transform * (col, row)
    x1, y1 = transform * (
        col + size,
        row + size,
    )

    expected_bounds = {
        "xmin": min(x0, x1),
        "xmax": max(x0, x1),
        "ymin": min(y0, y1),
        "ymax": max(y0, y1),
    }

    for name, expected in expected_bounds.items():
        if not np.isclose(
            float(record[name]),
            expected,
            atol=0.05,
            rtol=0,
        ):
            raise ValueError(
                f"Geographic footprint mismatch: {name}"
            )

    if str(record["crs"]) != metadata["crs"]:
        raise ValueError("Patch CRS does not match dataset")

    return {
        "joint_valid_fraction": fraction,
        "archive_bytes": path.stat().st_size,
    }


def validate_aligned_rasters(root, dataset_id, metadata):
    """
    Verify the three Step 1.7 GeoTIFFs still
    match the patch dataset reference grid.
    """
    root = Path(root)
    directory = root / "data/processed/aligned"

    files = {
        "features": (
            directory / f"{dataset_id}_multimodal_10m.tif",
            FEATURE_NAMES_ALL,
        ),
        "masks": (
            directory / f"{dataset_id}_masks_10m.tif",
            MASK_NAMES,
        ),
        "quality": (
            directory / f"{dataset_id}_quality_10m.tif",
            QUALITY_NAMES,
        ),
    }

    for name, (path, expected_names) in files.items():
        with rasterio.open(path) as source:
            if source.count != len(expected_names):
                raise ValueError(
                    f"{name}: incorrect band count"
                )

            if tuple(source.descriptions) != tuple(expected_names):
                raise ValueError(
                    f"{name}: incorrect band order"
                )

            if str(source.crs) != metadata["crs"]:
                raise ValueError(f"{name}: incorrect CRS")

            if (
                source.width != metadata["reference_width"]
                or source.height != metadata["reference_height"]
            ):
                raise ValueError(
                    f"{name}: incorrect raster dimensions"
                )

            if not np.allclose(
                tuple(source.transform)[:6],
                metadata["reference_transform"],
                atol=1e-7,
                rtol=0,
            ):
                raise ValueError(
                    f"{name}: incorrect geographic transform"
                )

    return True


def audit_dataset(
    root,
    dataset_id,
    config,
    max_patches=0,
):
    """
    Perform the Phase 1.10 inventory, spatial
    separation and per-patch quality checks.

    max_patches=0 means inspect every archive.
    """
    root = Path(root)
    q = config["quality"]

    if max_patches < 0:
        raise ValueError("max_patches must be nonnegative")

    manifest_dir = root / "data/metadata/manifests"

    original_path = (
        manifest_dir / f"{dataset_id}_patches.parquet"
    )

    split_dir = (
        manifest_dir / "splits" / dataset_id
    )

    metadata_path = (
        root / "data/patches/unassigned"
        / dataset_id / "dataset_metadata.json"
    )

    original = pd.read_parquet(original_path)
    assigned = pd.read_parquet(
        split_dir / "all_assigned.parquet"
    )

    split_report = json.loads(
        (split_dir / "split_report.json").read_text(
            encoding="utf-8"
        )
    )

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    errors = []
    warnings = []

    if (
        q["require_complete_patch_dataset"]
        and metadata.get("preview_limited", True)
    ):
        errors.append("Dataset is a limited patch preview")

    if metadata.get("accepted_patches") != len(original):
        errors.append(
            "Patch manifest and dataset metadata counts differ"
        )

    if not original["patch_id"].is_unique:
        errors.append("Original patch IDs are not unique")

    if not assigned["patch_id"].is_unique:
        errors.append("Assigned patch IDs are not unique")

    if (
        len(original) != len(assigned)
        or set(original["patch_id"])
        != set(assigned["patch_id"])
    ):
        errors.append(
            "Assigned and original patch inventories differ"
        )

    if "path" in assigned.columns and (
        assigned["path"].duplicated().any()
    ):
        errors.append("Duplicate patch archive paths")

    if not set(assigned["split"]).issubset(
        set(SPLITS)
    ):
        errors.append("Unexpected split labels")

    # The split manifest must not silently change
    # the original geographic patch inventory.
    shared_columns = (
        "path", "row", "col",
        "xmin", "ymin", "xmax", "ymax",
    )

    if (
        original["patch_id"].is_unique
        and assigned["patch_id"].is_unique
        and set(original.patch_id) == set(assigned.patch_id)
    ):
        a = original.set_index("patch_id")
        b = assigned.set_index("patch_id")

        for column in shared_columns:
            if not a[column].sort_index().equals(
                b[column].sort_index()
            ):
                errors.append(
                    f"Split manifest altered original {column}"
                )

    counts = {
        name: int((assigned["split"] == name).sum())
        for name in SPLITS
    }

    for name, minimum in q[
        "minimum_split_patches"
    ].items():
        if counts[name] < minimum:
            errors.append(
                f"{name} has only {counts[name]} patches; "
                f"minimum is {minimum}"
            )

    if counts["val"] < q[
        "warn_below_validation_patches"
    ]:
        warnings.append(
            "Validation set is relatively small"
        )

    excluded_fraction = (
        counts["excluded"] / max(len(assigned), 1)
    )

    if excluded_fraction > q[
        "warn_excluded_fraction_above"
    ]:
        warnings.append(
            "A substantial fraction of patches was excluded"
        )

    if counts != split_report.get("counts"):
        errors.append(
            "Split report and assigned manifest counts disagree"
        )

    fingerprint = hashlib.sha256(
        original_path.read_bytes()
    ).hexdigest()

    if fingerprint != split_report.get("input_sha256"):
        errors.append(
            "Split provenance fingerprint does not match"
        )

    # Verify every split-specific manifest.
    for name in SPLITS:
        subset = pd.read_parquet(
            split_dir / f"{name}.parquet"
        )

        expected_ids = set(
            assigned.loc[
                assigned["split"] == name,
                "patch_id",
            ]
        )

        if (
            len(subset) != counts[name]
            or set(subset["patch_id"]) != expected_ids
            or not (subset["split"] == name).all()
        ):
            errors.append(
                f"{name}.parquet is inconsistent"
            )

    distances = {}

    try:
        distances = verify_separation(
            assigned,
            split_report["axis"],
            q["minimum_separation_m"],
        )
    except (ValueError, AssertionError, KeyError) as exc:
        errors.append(
            f"Spatial separation failed: {exc}"
        )

    # Validate band schema and reference geometry.
    if metadata.get("feature_names") != list(FEATURE_NAMES_ALL):
        errors.append("Unexpected 19-channel feature schema")

    if metadata.get("mask_names") != list(MASK_NAMES):
        errors.append("Unexpected mask schema")

    if metadata.get("quality_names") != list(QUALITY_NAMES):
        errors.append("Unexpected quality schema")

    try:
        validate_aligned_rasters(
            root,
            dataset_id,
            metadata,
        )
    except (OSError, ValueError, KeyError) as exc:
        errors.append(
            f"Aligned raster validation failed: {exc}"
        )

    source_counts = {
        "sar_acquisitions": len(
            metadata.get("sar_sources", [])
        ),
        "optical_quarters": len(
            metadata.get("optical_sources", [])
        ),
    }

    if source_counts["sar_acquisitions"] < 3:
        errors.append("Fewer than three SAR acquisitions")

    if source_counts["optical_quarters"] < 2:
        errors.append("Fewer than two optical quarters")

    # Verify the physical archives exist, including
    # excluded patches, before validating their data.
    for record in assigned.itertuples(index=False):
        path = Path(record.path)

        if not path.is_absolute():
            path = root / path

        if not path.is_file():
            errors.append(
                f"Missing patch file: {record.patch_id}"
            )

    records = assigned.sort_values("patch_id")

    if max_patches > 0:
        records = records.head(max_patches)

    checked = 0
    passed = 0
    total_bytes = 0
    fractions = []

    for _, record in records.iterrows():
        path = Path(record["path"])

        if not path.is_absolute():
            path = root / path

        if not path.is_file():
            continue

        checked += 1

        try:
            result = inspect_patch_archive(
                path,
                record,
                metadata,
                minimum_valid_fraction=q[
                    "minimum_joint_valid_fraction"
                ],
            )

            passed += 1

            fractions.append(
                result["joint_valid_fraction"]
            )

            total_bytes += result["archive_bytes"]

        except (OSError, ValueError, KeyError, EOFError) as exc:
            errors.append(
                f"{record['patch_id']}: {exc}"
            )

    partial = checked < len(assigned)

    if errors:
        status = "FAIL"
    elif partial:
        status = "PARTIAL"
    elif warnings:
        status = "PASS_WITH_WARNINGS"
    else:
        status = "PASS"

    return {
        "dataset": dataset_id,
        "status": status,
        "total_patches": len(assigned),
        "split_counts": counts,
        "excluded_fraction": excluded_fraction,
        "spatial_separation_m": distances,
        "patches_inspected": checked,
        "patches_passed": passed,
        "partial_scan": partial,
        "inspected_archive_bytes": total_bytes,
        "joint_valid_fraction": {
            "min": float(min(fractions)) if fractions else None,
            "mean": float(np.mean(fractions)) if fractions else None,
            "max": float(max(fractions)) if fractions else None,
        },
        "source_counts": source_counts,
        "source_collection_ids": [
            "sentinel-1-rtc",
            "sentinel-2-l2a",
            "cop-dem-glo-30",
        ],
        "patch_size": metadata.get("patch_size"),
        "stride": metadata.get("stride"),
        "crs": metadata.get("crs"),
        "errors": errors,
        "warnings": warnings,
    }
