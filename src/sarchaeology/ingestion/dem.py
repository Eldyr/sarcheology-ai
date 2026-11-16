"""Placeholder module for SARchaeology AI v1.0."""

import numpy as np
import planetary_computer as pc
import rasterio

from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT


def read_dem_tile(item, grid):
    """
    Read one Copernicus DEM tile and reproject
    it to the requested grid.

    Returns a 2D float32 elevation array.
    Invalid pixels are NaN.
    """

    if "data" not in item.assets:
        raise ValueError(
            f"DEM asset 'data' missing: {item.id}"
        )

    url = pc.sign(item.assets["data"].href)

    with rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        GDAL_HTTP_MAX_RETRY="3",
        GDAL_HTTP_RETRY_DELAY="2",
    ):
        with rasterio.open(url) as source:
            with WarpedVRT(
                source,
                crs=grid.crs,
                transform=grid.transform,
                width=grid.width,
                height=grid.height,
                src_nodata=source.nodata,
                nodata=-9999.0,
                resampling=Resampling.bilinear,
            ) as vrt:

                result = vrt.read(
                    1,
                    masked=True,
                )

                elevation = np.asarray(
                    result.filled(np.nan),
                    dtype=np.float32,
                )

    elevation[~np.isfinite(elevation)] = np.nan

    return elevation


def mosaic_dem(items, grid, reader=read_dem_tile):
    """
    Combine multiple DEM tiles.

    Fill only missing pixels, keeping the
    first valid elevation in overlapping areas.
    """

    mosaic = np.full(
        (grid.height, grid.width),
        np.nan,
        dtype=np.float32,
    )

    for item in sorted(items, key=lambda x: x.id):
        print(f"Reading DEM tile: {item.id}")

        tile = reader(item, grid)

        if tile.shape != mosaic.shape:
            raise ValueError(
                f"Unexpected DEM shape: {tile.shape}"
            )

        fill = (
            ~np.isfinite(mosaic)
            & np.isfinite(tile)
        )

        mosaic[fill] = tile[fill]

    return mosaic
