"""Placeholder module for SARchaeology AI v1.0."""

from pathlib import Path

import numpy as np
import rasterio

from pyproj import Transformer
from shapely import transform as shapely_transform

from rasterio.enums import Resampling
from rasterio.warp import reproject

from sarchaeology.preprocessing.sar import create_grid
from sarchaeology.features.terrain_features import FEATURE_NAMES


def make_buffered_grid(
    aoi,
    target_crs,
    resolution_m=30,
    buffer_m=900,
):
    """
    Buffer the AOI in projected metres, then
    generate a projected grid covering the buffer.
    """
    forward = Transformer.from_crs(
        "EPSG:4326",
        target_crs,
        always_xy=True,
    )

    backward = Transformer.from_crs(
        target_crs,
        "EPSG:4326",
        always_xy=True,
    )

    projected = shapely_transform(
        aoi,
        forward.transform,
        interleaved=False,
    )

    buffered = projected.buffer(buffer_m)

    buffered_wgs84 = shapely_transform(
        buffered,
        backward.transform,
        interleaved=False,
    )

    return create_grid(
        buffered_wgs84,
        target_crs=target_crs,
        resolution=resolution_m,
    )


def align_terrain_features(
    features,
    source_grid,
    target_grid,
):
    """
    Resample 30 m terrain descriptors onto
    the same 10 m grid used by other sensors.

    This does not create new spatial detail.
    """
    if features.shape[0] != len(FEATURE_NAMES):
        raise ValueError("Expected five terrain channels")

    output = np.full(
        (
            len(FEATURE_NAMES),
            target_grid.height,
            target_grid.width,
        ),
        np.nan,
        dtype=np.float32,
    )

    for band in range(len(FEATURE_NAMES)):
        reproject(
            source=features[band],
            destination=output[band],
            src_transform=source_grid.transform,
            src_crs=source_grid.crs,
            src_nodata=np.nan,
            dst_transform=target_grid.transform,
            dst_crs=target_grid.crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

    # Exclude the padded region from the final model data.
    output[:, ~target_grid.aoi_mask] = np.nan

    return output


def save_terrain_products(
    native_dem,
    native_grid,
    aligned_features,
    target_grid,
    output_dir,
    name="buto_demo",
    nodata=-9999.0,
):
    """
    Save:
      - Native working-grid DEM
      - Five aligned terrain feature bands
      - Binary validity mask
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dem_path = output_dir / f"{name}_dem_30m.tif"
    features_path = (
        output_dir / f"{name}_terrain_10m.tif"
    )
    mask_path = (
        output_dir / f"{name}_terrain_mask.tif"
    )

    def profile(grid, count, dtype):
        return {
            "driver": "GTiff",
            "height": grid.height,
            "width": grid.width,
            "count": count,
            "dtype": dtype,
            "crs": grid.crs,
            "transform": grid.transform,
            "compress": "deflate",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
        }

    # 1. Original elevation on the 30 m working grid.
    native_profile = profile(
        native_grid, 1, "float32"
    )
    native_profile["nodata"] = nodata
    native_profile["predictor"] = 3

    with rasterio.open(
        dem_path, "w", **native_profile
    ) as dst:
        dst.write(
            np.where(
                np.isfinite(native_dem),
                native_dem,
                nodata,
            ).astype(np.float32),
            1,
        )
        dst.set_band_description(1, "elevation_m")
        dst.update_tags(
            vertical_datum="EGM2008",
            source="cop-dem-glo-30",
        )

    # 2. Five features on our common analysis grid.
    valid = (
        target_grid.aoi_mask
        & np.all(np.isfinite(aligned_features), axis=0)
    )

    feature_profile = profile(
        target_grid, 5, "float32"
    )
    feature_profile["nodata"] = nodata
    feature_profile["predictor"] = 3

    saved_features = np.where(
        valid[None, :, :],
        aligned_features,
        nodata,
    ).astype(np.float32)

    with rasterio.open(
        features_path, "w", **feature_profile
    ) as dst:
        dst.write(saved_features)

        for index, name_of_band in enumerate(
            FEATURE_NAMES, start=1
        ):
            dst.set_band_description(
                index, name_of_band
            )

        dst.update_tags(
            source="cop-dem-glo-30",
            vertical_datum="EGM2008",
            original_spacing="approximately 30m",
            analysis_spacing="10m",
            resampling="bilinear",
            features=",".join(FEATURE_NAMES),
        )

    # 3. Shared five-band validity mask.
    mask_profile = profile(
        target_grid, 1, "uint8"
    )
    mask_profile["nodata"] = 0

    with rasterio.open(
        mask_path, "w", **mask_profile
    ) as dst:
        dst.write(valid.astype(np.uint8), 1)

    return {
        "dem": dem_path,
        "features": features_path,
        "mask": mask_path,
    }
