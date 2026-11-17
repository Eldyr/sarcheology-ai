"""Placeholder module for SARchaeology AI v1.0."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


SPLITS = ("train", "val", "test")


def validate_manifest(
    manifest,
    expected_crs="EPSG:32636",
    patch_span_m=1280.0,
):
    """
    Check geographic metadata before assigning
    any patch to a dataset split.
    """
    required = {
        "patch_id", "path", "row", "col",
        "xmin", "ymin", "xmax", "ymax",
        "center_x", "center_y",
        "crs", "split",
    }

    missing = required - set(manifest.columns)

    if missing:
        raise ValueError(
            f"Missing manifest columns: {sorted(missing)}"
        )

    if manifest.empty:
        raise ValueError("Patch manifest is empty")

    if manifest["patch_id"].duplicated().any():
        raise ValueError("Duplicate patch IDs detected")

    if manifest["path"].duplicated().any():
        raise ValueError("Duplicate patch paths detected")

    if not (manifest["crs"] == expected_crs).all():
        raise ValueError(
            f"Expected projected CRS {expected_crs}"
        )

    if not (
        manifest["split"] == "unassigned"
    ).all():
        raise ValueError(
            "Input must be the unassigned patch manifest"
        )

    columns = [
        "xmin", "ymin", "xmax", "ymax",
        "center_x", "center_y",
    ]

    coords = manifest[columns].to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(coords).all():
        raise ValueError(
            "Non-finite geographic coordinates"
        )

    if not (
        (manifest["xmin"] < manifest["xmax"]).all()
        and (manifest["ymin"] < manifest["ymax"]).all()
    ):
        raise ValueError("Invalid patch bounding boxes")

    x_span = manifest["xmax"] - manifest["xmin"]
    y_span = manifest["ymax"] - manifest["ymin"]

    if not (
        np.allclose(x_span, patch_span_m, atol=0.01)
        and np.allclose(y_span, patch_span_m, atol=0.01)
    ):
        raise ValueError(
            "Patch footprints do not match the "
            f"expected {patch_span_m} m size"
        )

    if not (
        np.allclose(
            manifest["center_x"],
            (manifest["xmin"] + manifest["xmax"]) / 2,
            atol=0.01,
        )
        and np.allclose(
            manifest["center_y"],
            (manifest["ymin"] + manifest["ymax"]) / 2,
            atol=0.01,
        )
    ):
        raise ValueError(
            "Patch center coordinates are inconsistent"
        )


def verify_separation(
    manifest,
    axis,
    gap_m,
):
    """
    Validate geographic separation using the
    complete patch footprints.

    Since the splits occupy ordered strips on
    the selected axis, checking their extreme
    boundaries proves separation for all pairs.
    """
    lower = "xmin" if axis == "x" else "ymin"
    upper = "xmax" if axis == "x" else "ymax"

    distances = {}

    for a, b in (
        ("train", "val"),
        ("val", "test"),
        ("train", "test"),
    ):
        first = manifest[
            manifest["split"] == a
        ]

        second = manifest[
            manifest["split"] == b
        ]

        if first.empty or second.empty:
            raise ValueError(
                f"Empty split found: {a} or {b}"
            )

        distance = (
            float(second[lower].min())
            - float(first[upper].max())
        )

        distances[f"{a}_to_{b}"] = distance

        if distance < gap_m - 0.01:
            raise AssertionError(
                f"Geographic leakage between "
                f"{a} and {b}: {distance:.2f} m"
            )

    return distances


def assign_spatial_splits(
    manifest,
    *,
    axis="x",
    train_fraction=0.65,
    val_fraction=0.20,
    test_fraction=0.15,
    gap_m=640.0,
    expected_crs="EPSG:32636",
    patch_span_m=1280.0,
):
    """
    Split patches into three geographic strips.

    Patches crossing a split boundary or its
    exclusion buffer are marked 'excluded'.

    No imagery is copied or modified.
    """

    if axis not in ("x", "y"):
        raise ValueError("axis must be 'x' or 'y'")

    fractions = [
        train_fraction,
        val_fraction,
        test_fraction,
    ]

    if (
        any(f <= 0 or f >= 1 for f in fractions)
        or not np.isclose(sum(fractions), 1.0)
    ):
        raise ValueError(
            "Split fractions must be positive "
            "and sum to one"
        )

    if not np.isfinite(gap_m) or gap_m < 0:
        raise ValueError(
            "gap_m must be a finite nonnegative value"
        )

    result = manifest.copy()

    validate_manifest(
        result,
        expected_crs=expected_crs,
        patch_span_m=patch_span_m,
    )

    lower = "xmin" if axis == "x" else "ymin"
    upper = "xmax" if axis == "x" else "ymax"

    area_start = float(result[lower].min())
    area_end = float(result[upper].max())
    area_span = area_end - area_start

    # Boundaries divide the geographic extent,
    # not the number of available patches.
    boundary_1 = (
        area_start + area_span * train_fraction
    )

    boundary_2 = (
        area_start
        + area_span * (
            train_fraction + val_fraction
        )
    )

    half_gap = gap_m / 2.0

    # Entire footprint must belong to a region.
    train_condition = (
        result[upper] <= boundary_1 - half_gap
    )

    val_condition = (
        (result[lower] >= boundary_1 + half_gap)
        & (result[upper] <= boundary_2 - half_gap)
    )

    test_condition = (
        result[lower] >= boundary_2 + half_gap
    )

    # Any patch not fully contained in a valid
    # strip is excluded from model training.
    result["split"] = "excluded"

    result.loc[train_condition, "split"] = "train"
    result.loc[val_condition, "split"] = "val"
    result.loc[test_condition, "split"] = "test"

    result["exclusion_reason"] = None

    result.loc[
        result["split"] == "excluded",
        "exclusion_reason",
    ] = "geographic_boundary_buffer"

    counts = {
        name: int((result["split"] == name).sum())
        for name in (*SPLITS, "excluded")
    }

    if any(counts[name] == 0 for name in SPLITS):
        raise ValueError(
            "At least one split has zero patches. "
            f"Counts: {counts}. Try axis='y' or "
            "review the AOI extent and buffer. "
            "Do not use a random split as a fallback."
        )

    distances = verify_separation(
        result,
        axis,
        gap_m,
    )

    report = {
        "method": "geographic_strips_with_buffer",
        "axis": axis,
        "crs": expected_crs,
        "patch_span_m": patch_span_m,
        "gap_m": gap_m,
        "requested_geographic_fractions": {
            "train": train_fraction,
            "val": val_fraction,
            "test": test_fraction,
        },
        "geographic_extent_m": {
            "start": area_start,
            "end": area_end,
        },
        "split_boundaries_m": {
            "train_val": boundary_1,
            "val_test": boundary_2,
        },
        "counts": counts,
        "fraction_of_all_patches": {
            key: value / len(result)
            for key, value in counts.items()
        },
        "actual_separation_m": distances,
        "mean_joint_valid_fraction": {},
    }

    if "joint_valid_fraction" in result.columns:
        for name in SPLITS:
            subset = result[
                result["split"] == name
            ]

            report[
                "mean_joint_valid_fraction"
            ][name] = float(
                subset["joint_valid_fraction"].mean()
            )

    return result, report


def save_split_manifests(
    assigned,
    report,
    output_dir,
):
    """
    Write split manifests without duplicating
    any NPZ patch archives.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    paths["all"] = (
        output_dir / "all_assigned.parquet"
    )

    assigned.to_parquet(
        paths["all"],
        index=False,
    )

    for name in (*SPLITS, "excluded"):
        path = output_dir / f"{name}.parquet"

        subset = assigned[
            assigned["split"] == name
        ].copy()

        subset.to_parquet(
            path,
            index=False,
        )

        paths[name] = path

    report_path = (
        output_dir / "split_report.json"
    )

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    paths["report"] = report_path

    return paths
