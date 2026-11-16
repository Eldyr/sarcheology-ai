"""Placeholder module for SARchaeology AI v1.0."""

from dataclasses import dataclass
from pathlib import Path
import json
import math

import numpy as np
import rasterio

from pyproj import Transformer
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from shapely.geometry import mapping
from shapely.ops import transform as transform_geometry


@dataclass
class GridSpec:
    crs: str
    transform: object
    width: int
    height: int
    aoi_mask: np.ndarray


def create_grid(
    geometry,
    target_crs="EPSG:32636",
    resolution=10,
):
    """
    Project a WGS84 AOI into UTM and construct
    a north-up grid snapped to the pixel size.
    """

    transformer = Transformer.from_crs(
        "EPSG:4326",
        target_crs,
        always_xy=True,
    )

    projected = transform_geometry(
        transformer.transform,
        geometry,
    )

    left, bottom, right, top = projected.bounds

    left = math.floor(left / resolution) * resolution
    bottom = math.floor(bottom / resolution) * resolution
    right = math.ceil(right / resolution) * resolution
    top = math.ceil(top / resolution) * resolution

    width = round((right - left) / resolution)
    height = round((top - bottom) / resolution)

    affine = from_origin(
        left,
        top,
        resolution,
        resolution,
    )

    aoi_mask = geometry_mask(
        [mapping(projected)],
        out_shape=(height, width),
        transform=affine,
        invert=True,
    )

    return GridSpec(
        crs=target_crs,
        transform=affine,
        width=width,
        height=height,
        aoi_mask=aoi_mask,
    )


def linear_to_db(linear, valid_mask, nodata=-9999.0):
    """
    Convert Sentinel-1 linear gamma0 backscatter
    into decibels.

    linear: [2, height, width]
    valid_mask: [height, width]
    """

    result = np.full(
        linear.shape,
        nodata,
        dtype=np.float32,
    )

    result[:, valid_mask] = (
        10.0 * np.log10(linear[:, valid_mask])
    )

    return result


def mosaic_rtc(items, grid, reader):
    """
    Merge adjacent Sentinel-1 RTC scenes
    belonging to the same acquisition pass.

    The first valid pixel is retained where
    scenes overlap.
    """

    mosaic = np.full(
        (2, grid.height, grid.width),
        np.nan,
        dtype=np.float32,
    )

    valid = np.zeros(
        (grid.height, grid.width),
        dtype=bool,
    )

    overlap_count = 0

    for item in items:
        print(f"  Reading {item.id}")

        vv = reader(item, "vv", grid)
        vh = reader(item, "vh", grid)

        scene_valid = (
            grid.aoi_mask
            & np.isfinite(vv)
            & np.isfinite(vh)
            & (vv > 0)
            & (vh > 0)
        )

        overlap_count += int(
            np.count_nonzero(scene_valid & valid)
        )

        # Only fill pixels that are currently missing.
        fill = scene_valid & ~valid

        mosaic[0, fill] = vv[fill]
        mosaic[1, fill] = vh[fill]

        valid |= fill

    return mosaic, valid, overlap_count


def write_sar_products(
    linear,
    valid,
    grid,
    output_dir,
    stem,
    source_ids,
    nodata=-9999.0,
):
    """
    Save linear backscatter, dB backscatter,
    and a shared VV/VH validity mask.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    linear_output = np.full(
        linear.shape,
        nodata,
        dtype=np.float32,
    )

    linear_output[:, valid] = linear[:, valid]

    db_output = linear_to_db(
        linear,
        valid,
        nodata,
    )

    profile = {
        "driver": "GTiff",
        "height": grid.height,
        "width": grid.width,
        "count": 2,
        "dtype": "float32",
        "crs": grid.crs,
        "transform": grid.transform,
        "nodata": nodata,
        "compress": "deflate",
        "predictor": 3,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }

    paths = {
        "linear": output_dir / f"{stem}_linear.tif",
        "db": output_dir / f"{stem}_db.tif",
        "mask": output_dir / f"{stem}_mask.tif",
    }

    for kind, array in [
        ("linear", linear_output),
        ("db", db_output),
    ]:
        with rasterio.open(paths[kind], "w", **profile) as dst:
            dst.write(array)
            dst.set_band_description(1, "VV")
            dst.set_band_description(2, "VH")

            dst.update_tags(
                measurement="gamma0",
                units=(
                    "linear_power"
                    if kind == "linear"
                    else "dB"
                ),
                source_collection="sentinel-1-rtc",
                source_item_ids=json.dumps(source_ids),
                resampling="bilinear_in_linear_domain",
            )

    # Binary validity raster: 1 = observed; 0 = invalid.
    mask_profile = profile.copy()
    mask_profile.update(
        count=1,
        dtype="uint8",
        nodata=0,
        predictor=1,
    )

    with rasterio.open(paths["mask"], "w", **mask_profile) as dst:
        dst.write(valid.astype(np.uint8), 1)
        dst.set_band_description(1, "VV_VH_valid")

    return {key: str(path) for key, path in paths.items()}
