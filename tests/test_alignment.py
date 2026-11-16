
import numpy as np
import rasterio

from rasterio.transform import from_origin

from sarchaeology.preprocessing.sar import GridSpec

from sarchaeology.preprocessing.alignment import (
    build_multimodal_dataset,
    FEATURE_NAMES_ALL,
)

from sarchaeology.features.sar_features import (
    SAR_FEATURE_NAMES,
)
from sarchaeology.features.optical_features import (
    OPTICAL_ANNUAL_NAMES,
)
from sarchaeology.features.terrain_features import (
    FEATURE_NAMES,
)


def write_test_raster(
    path,
    data,
    transform,
    names=None,
):
    data = np.asarray(data)

    if data.ndim == 2:
        data = data[None, :, :]

    nodata = (
        -9999.0
        if np.issubdtype(data.dtype, np.floating)
        else 0
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
        nodata=nodata,
    ) as dst:
        dst.write(data)

        if names is not None:
            for i, name in enumerate(names, 1):
                dst.set_band_description(i, name)


def make_test_inputs(
    tmp_path,
    grid,
    shift_optical=False,
):
    h = grid.height
    w = grid.width

    sar = np.full(
        (7, h, w),
        -12.0,
        dtype=np.float32,
    )

    optical = np.full(
        (7, h, w),
        0.2,
        dtype=np.float32,
    )

    terrain = np.full(
        (5, h, w),
        5.0,
        dtype=np.float32,
    )

    sar_counts = np.full(
        (h, w), 29, dtype=np.uint16
    )

    optical_counts = np.full(
        (h, w), 4, dtype=np.uint16
    )

    terrain_mask = np.ones(
        (h, w), dtype=np.uint8
    )

    # Deliberately invalidate distinct pixels.
    sar_counts[0, 0] = 2
    optical_counts[0, 1] = 1
    terrain_mask[0, 2] = 0

    sar[:, -1, -1] = -9999.0

    original = grid.transform

    optical_transform = (
        from_origin(
            original.c + 10,
            original.f,
            10,
            10,
        )
        if shift_optical
        else original
    )

    paths = {
        "sar_features": tmp_path / "sar.tif",
        "sar_quality": tmp_path / "sar_count.tif",
        "optical_features": tmp_path / "optical.tif",
        "optical_quality": tmp_path / "optical_count.tif",
        "terrain_features": tmp_path / "terrain.tif",
        "terrain_quality": tmp_path / "terrain_mask.tif",
    }

    write_test_raster(
        paths["sar_features"],
        sar,
        original,
        SAR_FEATURE_NAMES,
    )

    write_test_raster(
        paths["sar_quality"],
        sar_counts,
        original,
    )

    write_test_raster(
        paths["optical_features"],
        optical,
        optical_transform,
        OPTICAL_ANNUAL_NAMES,
    )

    write_test_raster(
        paths["optical_quality"],
        optical_counts,
        optical_transform,
    )

    write_test_raster(
        paths["terrain_features"],
        terrain,
        original,
        FEATURE_NAMES,
    )

    write_test_raster(
        paths["terrain_quality"],
        terrain_mask,
        original,
    )

    return paths


def test_multimodal_stack_and_masks(tmp_path):
    aoi_mask = np.ones(
        (3, 4), dtype=bool
    )

    aoi_mask[0, 3] = False

    grid = GridSpec(
        crs="EPSG:32636",
        transform=from_origin(
            0, 30, 10, 10
        ),
        width=4,
        height=3,
        aoi_mask=aoi_mask,
    )

    paths = make_test_inputs(
        tmp_path, grid
    )

    outputs, report = build_multimodal_dataset(
        grid=grid,
        paths=paths,
        output_dir=tmp_path / "aligned",
        stem="test_2024",
    )

    with rasterio.open(
        outputs["features"]
    ) as src:
        values = src.read()

        assert values.shape == (19, 3, 4)
        assert src.count == 19
        assert src.crs.to_epsg() == 32636
        assert src.res == (10.0, 10.0)
        assert src.descriptions == FEATURE_NAMES_ALL

        # This location has valid data everywhere.
        assert np.isclose(
            values[0, 1, 0], -12.0
        )
        assert np.isclose(
            values[7, 1, 0], 0.2
        )
        assert np.isclose(
            values[14, 1, 0], 5.0
        )

        # Optical invalidity does not erase SAR.
        assert values[7, 0, 1] == -9999.0
        assert values[0, 0, 1] == -12.0

    with rasterio.open(
        outputs["masks"]
    ) as src:
        masks = src.read()

        # SAR threshold failed.
        assert masks[0, 0, 0] == 0

        # Optical threshold failed.
        assert masks[1, 0, 1] == 0

        # Terrain unavailable.
        assert masks[2, 0, 2] == 0

        # Pixel outside the AOI.
        assert np.all(masks[:, 0, 3] == 0)

        # Joint mask equals intersection.
        expected = np.all(
            masks[:3] == 1,
            axis=0,
        )

        assert np.array_equal(
            masks[3] == 1,
            expected,
        )

    with rasterio.open(
        outputs["quality"]
    ) as src:
        quality = src.read()

        assert quality.shape == (3, 3, 4)
        assert quality[0, 1, 0] == 29
        assert quality[1, 1, 0] == 4
        assert quality[2, 1, 0] == 1

    assert report["valid_pixels"]["joint"] > 0

    for details in report[
        "source_rasters"
    ].values():
        assert details["reprojected"] is False


def test_misaligned_optical_is_reprojected(tmp_path):
    grid = GridSpec(
        crs="EPSG:32636",
        transform=from_origin(
            0, 30, 10, 10
        ),
        width=4,
        height=3,
        aoi_mask=np.ones(
            (3, 4), dtype=bool
        ),
    )

    paths = make_test_inputs(
        tmp_path,
        grid,
        shift_optical=True,
    )

    outputs, report = build_multimodal_dataset(
        grid=grid,
        paths=paths,
        output_dir=tmp_path / "shifted",
        stem="shift_test",
    )

    assert report["source_rasters"][
        "optical_features"
    ]["reprojected"] is True

    assert report["source_rasters"][
        "optical_quality"
    ]["reprojected"] is True

    with rasterio.open(
        outputs["features"]
    ) as src:
        assert src.transform == grid.transform
        assert src.count == 19

    with rasterio.open(
        outputs["masks"]
    ) as src:
        masks = src.read()

        # Shifted optical image does not cover
        # the leftmost target-grid column.
        assert np.all(
            masks[1, :, 0] == 0
        )
