import numpy as np
import rasterio
from rasterio.transform import from_origin
from sarchaeology.features.terrain_features import (
    calculate_terrain_features,
)
from sarchaeology.ingestion.dem import mosaic_dem
from sarchaeology.preprocessing.sar import GridSpec
from sarchaeology.preprocessing.terrain import (
    align_terrain_features,
    save_terrain_products,
)


def test_flat_terrain():
    dem = np.full((64, 64), 100.0, dtype=np.float32)
    features = calculate_terrain_features(
        dem,
        resolution_m=30,
    )
    assert features.shape == (5, 64, 64)
    assert np.allclose(features[0], 100)
    assert np.allclose(features[1], 0)
    assert np.allclose(features[2], 0)
    assert np.allclose(features[3], 0)
    assert np.allclose(features[4], 0)


def test_uniform_slope():
    # Elevation increases by 0.1 m per horizontal m.
    x = np.arange(64) * 30.0
    dem = np.tile(
        100.0 + 0.1 * x,
        (64, 1),
    ).astype(np.float32)
    features = calculate_terrain_features(
        dem,
        resolution_m=30,
    )
    expected_degrees = np.degrees(np.arctan(0.1))
    # Ignore boundaries influenced by smoothing.
    interior = features[1, 10:-10, 10:-10]
    assert np.allclose(
        interior,
        expected_degrees,
        atol=0.05,
    )


def test_dem_mosaic_preserves_zero():
    grid = GridSpec(
        crs="EPSG:32636",
        transform=from_origin(0, 30, 30, 30),
        width=2,
        height=1,
        aoi_mask=np.ones((1, 2), dtype=bool),
    )

    class Tile:
        def __init__(self, tile_id):
            self.id = tile_id

    tiles = [Tile("A"), Tile("B")]
    arrays = {
        "A": [[0.0, np.nan]],
        "B": [[50.0, 25.0]],
    }

    def fake_reader(item, grid):
        return np.asarray(
            arrays[item.id],
            dtype=np.float32,
        )

    mosaic = mosaic_dem(
        tiles,
        grid,
        reader=fake_reader,
    )
    assert mosaic[0, 0] == 0.0
    assert mosaic[0, 1] == 25.0


def test_alignment_and_export(tmp_path):
    native_grid = GridSpec(
        crs="EPSG:32636",
        transform=from_origin(0, 120, 30, 30),
        width=4,
        height=4,
        aoi_mask=np.ones((4, 4), dtype=bool),
    )
    target_grid = GridSpec(
        crs="EPSG:32636",
        transform=from_origin(0, 120, 10, 10),
        width=12,
        height=12,
        aoi_mask=np.ones((12, 12), dtype=bool),
    )
    dem = np.full((4, 4), 100.0, dtype=np.float32)
    native_features = calculate_terrain_features(
        dem,
        resolution_m=30,
    )
    aligned = align_terrain_features(
        native_features,
        native_grid,
        target_grid,
    )
    assert aligned.shape == (5, 12, 12)
    assert np.allclose(aligned[0], 100)

    paths = save_terrain_products(
        native_dem=dem,
        native_grid=native_grid,
        aligned_features=aligned,
        target_grid=target_grid,
        output_dir=tmp_path,
        name="test",
    )

    with rasterio.open(paths["features"]) as ds:
        assert ds.count == 5
        assert ds.crs.to_epsg() == 32636
        assert ds.res == (10.0, 10.0)
        assert ds.read().shape == (5, 12, 12)

    with rasterio.open(paths["mask"]) as ds:
        assert np.all(ds.read(1) == 1)