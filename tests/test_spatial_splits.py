
import numpy as np
import pandas as pd
import pytest

from sarchaeology.evaluation.spatial_splits import (
    assign_spatial_splits,
    validate_manifest,
    verify_separation,
    save_split_manifests,
)


def make_test_manifest(nx=30, ny=4):
    """
    Synthetic overlapping patches.

    Footprint: 1280 x 1280 metres
    Stride: 640 metres
    CRS: EPSG:32636
    """
    rows = []

    for r in range(ny):
        for c in range(nx):
            xmin = 500000 + c * 640
            ymin = 3400000 + r * 640

            xmax = xmin + 1280
            ymax = ymin + 1280

            patch_id = f"r{r:05d}_c{c:05d}"

            rows.append({
                "patch_id": patch_id,
                "path": f"data/patches/{patch_id}.npz",
                "row": r * 64,
                "col": c * 64,
                "height": 128,
                "width": 128,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "center_x": (xmin + xmax) / 2,
                "center_y": (ymin + ymax) / 2,
                "joint_valid_fraction": 0.95,
                "crs": "EPSG:32636",
                "split": "unassigned",
            })

    return pd.DataFrame(rows)


def test_manifest_validation():
    manifest = make_test_manifest()

    validate_manifest(
        manifest,
        expected_crs="EPSG:32636",
        patch_span_m=1280,
    )


def test_geographic_split_generation():
    manifest = make_test_manifest()

    assigned, report = assign_spatial_splits(
        manifest,
        axis="x",
        gap_m=640,
    )

    counts = assigned["split"].value_counts()

    assert counts["train"] > 0
    assert counts["val"] > 0
    assert counts["test"] > 0
    assert counts["excluded"] > 0

    assert len(assigned) == len(manifest)

    # The original manifest must remain untouched.
    assert (
        manifest["split"] == "unassigned"
    ).all()

    assert report["gap_m"] == 640


def test_no_geographic_overlap():
    assigned, report = assign_spatial_splits(
        make_test_manifest(),
        gap_m=640,
    )

    train = assigned[assigned.split == "train"]
    val = assigned[assigned.split == "val"]
    test = assigned[assigned.split == "test"]

    # Full geographic footprints, not centers.
    assert train.xmax.max() + 640 <= (
        val.xmin.min() + 0.01
    )

    assert val.xmax.max() + 640 <= (
        test.xmin.min() + 0.01
    )

    distances = verify_separation(
        assigned,
        "x",
        640,
    )

    assert distances["train_to_val"] >= 640
    assert distances["val_to_test"] >= 640


def test_y_axis_spatial_split():
    # Make a taller geographic region.
    manifest = make_test_manifest(
        nx=4,
        ny=30,
    )

    assigned, report = assign_spatial_splits(
        manifest,
        axis="y",
        gap_m=640,
    )

    assert all(
        report["counts"][name] > 0
        for name in ("train", "val", "test")
    )


def test_reproducible_assignments():
    manifest = make_test_manifest()

    first, _ = assign_spatial_splits(
        manifest,
        gap_m=640,
    )

    # Input row order must not affect geographic
    # split assignments.
    shuffled = manifest.sample(
        frac=1,
        random_state=42,
    )

    second, _ = assign_spatial_splits(
        shuffled,
        gap_m=640,
    )

    a = first.set_index("patch_id")["split"]
    b = second.set_index("patch_id")["split"]

    pd.testing.assert_series_equal(
        a.sort_index(),
        b.sort_index(),
    )


def test_duplicate_patch_ids_rejected():
    manifest = make_test_manifest()

    manifest.loc[
        manifest.index[1],
        "patch_id",
    ] = manifest.iloc[0]["patch_id"]

    with pytest.raises(
        ValueError,
        match="Duplicate patch IDs",
    ):
        validate_manifest(manifest)


def test_too_small_region_rejected():
    manifest = make_test_manifest().iloc[:10]

    with pytest.raises(
        ValueError,
        match="zero patches",
    ):
        assign_spatial_splits(
            manifest,
            gap_m=640,
        )


def test_save_split_manifests(tmp_path):
    assigned, report = assign_spatial_splits(
        make_test_manifest(),
        gap_m=640,
    )

    paths = save_split_manifests(
        assigned,
        report,
        tmp_path,
    )

    for key in (
        "all",
        "train",
        "val",
        "test",
        "excluded",
        "report",
    ):
        assert paths[key].exists()

    combined = pd.read_parquet(paths["all"])

    assert len(combined) == len(assigned)

    total = 0

    for name in (
        "train",
        "val",
        "test",
        "excluded",
    ):
        subset = pd.read_parquet(paths[name])

        assert (subset["split"] == name).all()
        total += len(subset)

    assert total == len(assigned)
