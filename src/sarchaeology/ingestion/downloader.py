"""Placeholder module for SARchaeology AI v1.0."""

import numpy as np
import planetary_computer as pc
import rasterio

from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT


def read_rtc_band(item, polarization, grid):
    """
    Read one Sentinel-1 RTC polarization from
    Planetary Computer into our target grid.

    Parameters
    ----------
    item:
        A PySTAC Sentinel-1 RTC item.

    polarization:
        Either "vv" or "vh".

    grid:
        GridSpec defining CRS, transform and dimensions.

    Returns
    -------
    np.ndarray:
        2D float32 linear gamma0 power.
        Invalid pixels are represented as NaN.
    """

    polarization = polarization.lower()

    if polarization not in ("vv", "vh"):
        raise ValueError(
            f"Unsupported polarization: {polarization}"
        )

    if polarization not in item.assets:
        raise ValueError(
            f"{polarization} is missing from {item.id}"
        )

    # Sign the asset URL immediately before reading.
    # Do not save signed URLs in permanent manifests.
    asset_url = pc.sign(
        item.assets[polarization].href
    )

    env_settings = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "GDAL_HTTP_MAX_RETRY": "3",
        "GDAL_HTTP_RETRY_DELAY": "2",
    }

    with rasterio.Env(**env_settings):
        with rasterio.open(asset_url) as source:

            source_nodata = source.nodata
            if source_nodata is None:
                source_nodata = -32768.0

            # WarpedVRT performs a virtual reprojection.
            # It reads the source remotely on demand,
            # without exporting the complete scene.
            with WarpedVRT(
                source,
                crs=grid.crs,
                transform=grid.transform,
                width=grid.width,
                height=grid.height,
                resampling=Resampling.bilinear,
                src_nodata=source_nodata,
                nodata=-9999.0,
            ) as vrt:

                image = vrt.read(
                    1,
                    masked=True,
                )

                values = np.asarray(
                    image.filled(np.nan),
                    dtype=np.float32,
                )

    # Gamma0 power must be positive for log10.
    invalid = (
        ~np.isfinite(values)
        | (values <= 0)
    )

    values[invalid] = np.nan

    return values
