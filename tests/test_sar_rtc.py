
import numpy as np
import rasterio

from rasterio.transform import from_origin

from sarchaeology.preprocessing.sar import (
    GridSpec,
    create_grid,
    linear_to_db,
    mosaic_rtc,
    write_sar_products,
)

from shapely.geometry import box


def test_linear_to_db():
    linear = np.array(
        [
            [[0.01, 0.1]],
            [[0.001, 1.0]],
        ],
        dtype=np.float32,
    )

    valid = np.array(
        [[True, False]]
    )

    result = linear_to_db(linear, valid)

    assert np.isclose(result[0, 0, 0], -20.0)
    assert np.isclose(result[1, 0, 0], -30.0)
    assert result[0, 0, 1] == -9999.0
    assert result[1, 0, 1] == -9999.0


def test_grid_creation():
    aoi = box(
        30.74, 31.19,
        30.75, 31.20,
    )

    grid = create_grid(
        aoi,
        target_crs="EPSG:32636",
        resolution=10,
    )

    assert grid.width > 0
    assert grid.height > 0
    assert grid.aoi_mask.any()

    assert grid.transform.a == 10
    assert grid.transform.e == -10


def test_adjacent_scene_mosaic():
    grid = GridSpec(
        crs="EPSG:32636",
        transform=from_origin(0, 10, 10, 10),
        width=2,
        height=1,
        aoi_mask=np.ones((1, 2), dtype=bool),
    )

    class Scene:
        def __init__(self, scene_id):
            self.id = scene_id

    scenes = [Scene("A"), Scene("B")]

    arrays = {
        ("A", "vv"): [[0.1, np.nan]],
        ("A", "vh"): [[0.05, np.nan]],
        ("B", "vv"): [[0.2, 0.3]],
        ("B", "vh"): [[0.08, 0.09]],
    }

    def fake_reader(item, polarization, grid):
        return np.asarray(
            arrays[(item.id, polarization)],
            dtype=np.float32,
        )

    mosaic, valid, overlaps = mosaic_rtc(
        scenes,
        grid,
        fake_reader,
    )

    assert valid.all()
    assert overlaps == 1

    # First valid scene keeps the overlapping pixel.
    assert np.isclose(mosaic[0, 0, 0], 0.1)
    assert np.isclose(mosaic[1, 0, 0], 0.05)

    # Second scene fills the missing pixel.
    assert np.isclose(mosaic[0, 0, 1], 0.3)
    assert np.isclose(mosaic[1, 0, 1], 0.09)


def test_output_geotiffs(tmp_path):
    grid = GridSpec(
        crs="EPSG:32636",
        transform=from_origin(0, 20, 10, 10),
        width=2,
        height=2,
        aoi_mask=np.ones((2, 2), dtype=bool),
    )

    linear = np.full(
        (2, 2, 2), 0.01, dtype=np.float32
    )

    valid = np.array([
        [True, True],
        [False, True],
    ])

    paths = write_sar_products(
        linear=linear,
        valid=valid,
        grid=grid,
        output_dir=tmp_path,
        stem="test",
        source_ids=["test_scene"],
    )

    with rasterio.open(paths["db"]) as ds:
        assert ds.count == 2
        assert ds.crs.to_epsg() == 32636
        assert ds.res == (10.0, 10.0)
        assert np.isclose(ds.read(1)[0, 0], -20.0)
        assert ds.read(1)[1, 0] == -9999.0

    with rasterio.open(paths["mask"]) as ds:
        assert np.array_equal(
            ds.read(1),
            valid.astype(np.uint8),
        )
