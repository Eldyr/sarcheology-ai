
import numpy as np
import pandas as pd
import rasterio

from rasterio.transform import from_origin

from sarchaeology.preprocessing.alignment import (
    FEATURE_NAMES_ALL,
    MASK_NAMES,
    QUALITY_NAMES,
)

from sarchaeology.preprocessing.tiling import (
    axis_starts,
    iter_patch_windows,
    build_patch_dataset,
)


def write_raster(path, data, transform, names=None):
    data = np.asarray(data)

    if data.ndim == 2:
        data = data[None]

    is_float = np.issubdtype(
        data.dtype,
        np.floating,
    )

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=data.shape[2],
        height=data.shape[1],
        count=data.shape[0],
        dtype=data.dtype,
        crs="EPSG:32636",
        transform=transform,
        nodata=-9999.0 if is_float else 0,
    ) as dst:
        dst.write(data)

        if names is not None:
            for i, name in enumerate(names, 1):
                dst.set_band_description(i, name)


def test_patch_window_positions():
    assert axis_starts(8, 4, 4) == [0, 4]
    assert axis_starts(7, 4, 2) == [0, 2, 3]
    assert axis_starts(3, 4, 2) == []

    windows = list(
        iter_patch_windows(8, 8, 4, 4)
    )

    assert len(windows) == 4
    assert windows[0][0:2] == (0, 0)
    assert windows[-1][0:2] == (4, 4)


def create_test_sources(tmp_path):
    transform = from_origin(
        1000, 2000, 10, 10
    )

    features = np.full(
        (19, 8, 8),
        0.5,
        dtype=np.float32,
    )

    masks = np.ones(
        (4, 8, 8),
        dtype=np.uint8,
    )

    # One 4x4 region is invalid for SAR,
    # and therefore also jointly invalid.
    masks[0, :4, :4] = 0
    masks[3] = np.all(
        masks[:3] == 1,
        axis=0,
    ).astype(np.uint8)

    features[:7, :4, :4] = -9999.0

    quality = np.stack(
        [
            np.full((8, 8), 29),
            np.full((8, 8), 4),
            np.ones((8, 8)),
        ],
        axis=0,
    ).astype(np.uint16)

    paths = {
        "features": tmp_path / "features.tif",
        "masks": tmp_path / "masks.tif",
        "quality": tmp_path / "quality.tif",
        "sar1": tmp_path / "sar1.tif",
        "sar2": tmp_path / "sar2.tif",
        "optical1": tmp_path / "optical1.tif",
        "optical2": tmp_path / "optical2.tif",
    }

    write_raster(
        paths["features"],
        features,
        transform,
        FEATURE_NAMES_ALL,
    )

    write_raster(
        paths["masks"],
        masks,
        transform,
        MASK_NAMES,
    )

    write_raster(
        paths["quality"],
        quality,
        transform,
        QUALITY_NAMES,
    )

    for key, value in [
        ("sar1", -12.0),
        ("sar2", -14.0),
    ]:
        write_raster(
            paths[key],
            np.full(
                (2, 8, 8),
                value,
                dtype=np.float32,
            ),
            transform,
            ("VV", "VH"),
        )

    for key, value in [
        ("optical1", 0.2),
        ("optical2", 0.4),
    ]:
        write_raster(
            paths[key],
            np.full(
                (6, 8, 8),
                value,
                dtype=np.float32,
            ),
            transform,
            (
                "B02", "B03", "B04",
                "B08", "B11", "B12",
            ),
        )

    return paths


def test_multimodal_patch_generation(tmp_path):
    paths = create_test_sources(tmp_path)

    sar_sources = [
        {
            "id": "sar1",
            "date": "2024-01-10",
            "path": paths["sar1"],
        },
        {
            "id": "sar2",
            "date": "2024-01-22",
            "path": paths["sar2"],
        },
    ]

    optical_sources = [
        {
            "quarter": "2024Q1",
            "path": paths["optical1"],
        },
        {
            "quarter": "2024Q2",
            "path": paths["optical2"],
        },
    ]

    manifest_path, metadata_path = (
        build_patch_dataset(
            features_path=paths["features"],
            masks_path=paths["masks"],
            quality_path=paths["quality"],
            sar_sources=sar_sources,
            optical_sources=optical_sources,
            output_dir=tmp_path / "dataset",
            manifest_path=tmp_path / "patches.parquet",
            project_root=tmp_path,
            patch_size=4,
            stride=4,
            min_valid_fraction=0.85,
        )
    )

    manifest = pd.read_parquet(manifest_path)

    # Four candidate windows; one has no valid pixels.
    assert len(manifest) == 3
    assert "r00000_c00000" not in set(
        manifest["patch_id"]
    )

    assert metadata_path.exists()
    assert (manifest["split"] == "unassigned").all()
    assert (
        manifest["joint_valid_fraction"] >= 0.85
    ).all()

    first_path = tmp_path / manifest.iloc[0]["path"]

    with np.load(
        first_path,
        allow_pickle=False,
    ) as patch:
        assert patch["features"].shape == (
            19, 4, 4
        )
        assert patch["masks"].shape == (
            4, 4, 4
        )
        assert patch["quality"].shape == (
            3, 4, 4
        )
        assert patch["sar"].shape == (
            2, 2, 4, 4
        )
        assert patch["optical"].shape == (
            2, 6, 4, 4
        )

        assert patch["sar_valid"].shape == (
            2, 4, 4
        )
        assert patch["optical_valid"].shape == (
            2, 4, 4
        )

        assert np.isclose(
            patch["sar"][0, 0, 0, 0],
            -12.0,
        )

        assert np.isclose(
            patch["optical"][1, 0, 0, 0],
            0.4,
        )


def test_patch_geographic_bounds(tmp_path):
    paths = create_test_sources(tmp_path)

    sar_sources = [{
        "id": "sar1",
        "date": "2024-01-10",
        "path": paths["sar1"],
    }]

    optical_sources = [{
        "quarter": "2024Q1",
        "path": paths["optical1"],
    }]

    manifest_path, _ = build_patch_dataset(
        features_path=paths["features"],
        masks_path=paths["masks"],
        quality_path=paths["quality"],
        sar_sources=sar_sources,
        optical_sources=optical_sources,
        output_dir=tmp_path / "geo_dataset",
        manifest_path=tmp_path / "geo_patches.parquet",
        project_root=tmp_path,
        patch_size=4,
        stride=4,
        min_valid_fraction=0.85,
    )

    manifest = pd.read_parquet(manifest_path)

    patch = manifest.loc[
        manifest["patch_id"] == "r00000_c00004"
    ].iloc[0]

    assert patch["xmin"] == 1040
    assert patch["xmax"] == 1080
    assert patch["ymin"] == 1960
    assert patch["ymax"] == 2000
    assert patch["crs"] == "EPSG:32636"
