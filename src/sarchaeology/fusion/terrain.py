"""Terrain derivatives at the DEM working resolution, then mask-aware reprojection."""
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.ndimage import minimum_filter, uniform_filter

from .io import write_raster

BANDS = ("elevation_m", "slope_degrees", "aspect_east", "aspect_north", "laplacian_per_m",
         "ruggedness_rms_m", "tpi_3x3_m", "local_relief_11x11_m", "multidirectional_hillshade")


def derivatives(elevation, spacing):
    valid = np.isfinite(elevation)
    z = np.where(valid, elevation, 0).astype("float64")
    south, east = np.gradient(z, spacing)
    steepness = np.hypot(east, south)
    denom = np.maximum(steepness, 1e-12)
    aspect_e, aspect_n = -east / denom, south / denom
    slope = np.arctan(steepness)
    curvature = np.gradient(east, spacing, axis=1) + np.gradient(south, spacing, axis=0)
    mean = uniform_filter(z, 3)
    ruggedness = np.sqrt(np.maximum(0, (uniform_filter(z*z, 3) - 2*z*mean + z*z)*9/8))
    lights = []
    for angle in np.deg2rad([225, 270, 315, 360]):
        cosine = np.sin(slope)*(aspect_e*np.sin(angle) + aspect_n*np.cos(angle))
        lights.append(np.maximum(0, (np.cos(slope) + cosine)/np.sqrt(2)))
    features = np.stack([z, np.rad2deg(slope), aspect_e, aspect_n, curvature, ruggedness,
                         z-mean, z-uniform_filter(z, 11), np.mean(lights, axis=0)])
    features[:, ~minimum_filter(valid, size=11, mode="constant", cval=False)] = np.nan
    return features.astype("float32")


def prepare_terrain(cfg, transform, crs, shape, train, out):
    shape = tuple(shape)
    metadata = pd.read_parquet(cfg["dem_metadata"]).iloc[0]
    with rasterio.open(cfg["dem"]) as src:
        t = src.transform
        if (not src.crs.is_projected or src.crs.linear_units != "metre" or t.a <= 0
                or t.e != -t.a or t.b or t.d or not np.isclose(t.a, metadata.native_spacing_m)):
            raise ValueError("Supply a north-up square-pixel, metric working-grid DEM")
        tags = src.tags()
        unit = (src.units[0] or "").strip().lower()
        metric_units = {"m", "meter", "meters", "metre", "metres"}
        if (unit and unit not in metric_units) or (not unit and src.descriptions[0] != "elevation_m"):
            raise ValueError("Expected metre elevation units or the Phase 1 elevation_m band")
        if (not tags.get("vertical_datum") or tags["vertical_datum"] != metadata.vertical_datum
                or tags.get("source") != metadata.source_collection):
            raise ValueError("DEM source or vertical datum disagrees with Phase 1 metadata")
        native = derivatives(src.read(1, masked=True).filled(np.nan), t.a)
        aligned = np.full((len(BANDS), *shape), np.nan, dtype="float32")
        reproject(native, aligned, src_transform=t, src_crs=src.crs, src_nodata=np.nan,
                  dst_transform=transform, dst_crs=crs, dst_nodata=np.nan, resampling=Resampling.bilinear)
    with rasterio.open(cfg["dem_mask"]) as mask:
        if mask.crs != crs or mask.transform != transform or mask.shape != shape:
            raise ValueError("DEM coverage mask grid mismatch")
        aligned[:, mask.read(1, masked=True).filled(0) == 0] = np.nan
    valid = np.isfinite(aligned).all(0)
    aligned[:, ~valid] = np.nan
    train_area = np.zeros(shape, bool)
    for p in train.itertuples():
        train_area[p.row:p.row+p.height, p.col:p.col+p.width] = True
    sample = aligned[:, valid & train_area]
    if not sample.shape[1]:
        raise ValueError("No valid terrain in training footprints")
    mean = sample.mean(1, dtype=np.float64).astype("float32")
    scale = sample.std(1, dtype=np.float64).clip(1e-6).astype("float32")
    write_raster(out/"terrain.tif", aligned, crs, transform, BANDS, **tags)
    stats = dict(fit_split="train", unique_pixels=sample.shape[1], bands=list(BANDS),
                 mean=mean.tolist(), scale=scale.tolist(), vertical_datum=tags["vertical_datum"],
                 source=tags["source"], source_item_ids=metadata.source_item_ids,
                 native_spacing_m=float(t.a), valid_pixels=int(valid.sum()))
    normalized = ((aligned - mean[:, None, None])/scale[:, None, None]).clip(-8, 8)
    return np.nan_to_num(normalized), valid, stats
