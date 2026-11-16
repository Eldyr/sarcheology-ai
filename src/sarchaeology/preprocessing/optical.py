"""Placeholder module for SARchaeology AI v1.0."""

import warnings
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import rasterio

from scipy.ndimage import binary_dilation


BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")


def convert_to_reflectance(raw, quantification, offsets):
    """
    Convert Sentinel-2 digital numbers to
    bottom-of-atmosphere surface reflectance.

    DN = 0 means no data.
    """
    raw = np.asarray(raw, dtype=np.float32)

    if raw.ndim != 3 or raw.shape[0] != len(BANDS):
        raise ValueError("Expected [6, height, width]")

    if quantification <= 0:
        raise ValueError("Invalid quantification value")

    offset_array = np.array(
        [offsets[band] for band in BANDS],
        dtype=np.float32,
    )[:, None, None]

    reflectance = (
        raw + offset_array
    ) / quantification

    # Do not treat zero DN as physical reflectance.
    reflectance[raw <= 0] = np.nan

    return reflectance.astype(np.float32)


def make_valid_mask(
    raw,
    scl,
    aoi_mask,
    invalid_classes,
    cloud_buffer_pixels=1,
):
    """
    Keep only valid, cloud-free pixels with
    nonzero measurements in every band.
    """
    invalid_scl = np.isin(
        scl,
        invalid_classes,
    )

    if cloud_buffer_pixels > 0:
        invalid_scl = binary_dilation(
            invalid_scl,
            iterations=cloud_buffer_pixels,
        )

    band_valid = np.all(
        np.isfinite(raw) & (raw > 0),
        axis=0,
    )

    return (
        aoi_mask
        & ~invalid_scl
        & band_valid
    )


def prepare_optical_scene(
    raw,
    scl,
    grid,
    quantification,
    offsets,
    config,
):
    """Convert, mask and prepare one acquisition."""
    valid = make_valid_mask(
        raw=raw,
        scl=scl,
        aoi_mask=grid.aoi_mask,
        invalid_classes=config["invalid_scl_classes"],
        cloud_buffer_pixels=config["cloud_buffer_pixels"],
    )

    reflectance = convert_to_reflectance(
        raw,
        quantification,
        offsets,
    )

    reflectance[:, ~valid] = np.nan

    coverage = (
        np.count_nonzero(valid)
        / max(np.count_nonzero(grid.aoi_mask), 1)
    )

    return reflectance, valid, coverage


def raster_profile(grid, count, dtype, nodata):
    return {
        "driver": "GTiff",
        "height": grid.height,
        "width": grid.width,
        "count": count,
        "dtype": dtype,
        "crs": grid.crs,
        "transform": grid.transform,
        "nodata": nodata,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }


def save_optical_scene(
    reflectance,
    valid,
    scl,
    grid,
    directory,
    stem,
    item_id,
    baseline,
    quantification,
    offsets,
    nodata=-9999.0,
):
    """
    Save reflectance, valid mask and SCL rasters
    with matching projection and dimensions.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    reflectance_path = (
        directory / f"{stem}_reflectance.tif"
    )
    mask_path = directory / f"{stem}_mask.tif"
    scl_path = directory / f"{stem}_scl.tif"

    profile = raster_profile(
        grid, 6, "float32", nodata
    )
    profile["predictor"] = 3

    output = np.where(
        valid[None, :, :],
        reflectance,
        nodata,
    ).astype(np.float32)

    with rasterio.open(
        reflectance_path, "w", **profile
    ) as dst:
        dst.write(output)
        dst.write_mask(
            valid.astype(np.uint8) * 255
        )

        for index, band in enumerate(BANDS, 1):
            dst.set_band_description(index, band)

        dst.update_tags(
            source_item=item_id,
            processing_baseline=str(baseline),
            units="surface_reflectance",
            quantification=str(quantification),
            offsets=str(offsets),
        )

    mask_profile = raster_profile(
        grid, 1, "uint8", 0
    )

    with rasterio.open(
        mask_path, "w", **mask_profile
    ) as dst:
        dst.write(valid.astype(np.uint8), 1)

    scl_profile = raster_profile(
        grid, 1, "uint8", 0
    )

    with rasterio.open(
        scl_path, "w", **scl_profile
    ) as dst:
        dst.write(scl.astype(np.uint8), 1)

    return {
        "reflectance": reflectance_path,
        "mask": mask_path,
        "scl": scl_path,
    }


def save_rgb_preview(
    reflectance_path,
    output_path,
):
    """
    Export RGB for human inspection only.
    Display contrast is not used in training.
    """
    import matplotlib.pyplot as plt

    with rasterio.open(reflectance_path) as ds:
        # B04 red, B03 green, B02 blue
        rgb = ds.read(
            [3, 2, 1],
            masked=True,
        )

        rgb = rgb.filled(np.nan)

    image = np.moveaxis(rgb, 0, -1)

    # Fixed display stretching and gamma correction.
    image = np.clip(image / 0.35, 0, 1)
    image = np.power(image, 1 / 2.2)

    image[~np.isfinite(image)] = 0

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    plt.imsave(output_path, image)


def make_quarterly_composite(
    scene_paths,
    output_dir,
    name,
    minimum_observations=2,
    nodata=-9999.0,
):
    """
    Median composite from aligned optical scenes.

    Process raster blocks instead of loading
    every scene into system RAM simultaneously.
    """
    scene_paths = [Path(p) for p in scene_paths]

    if len(scene_paths) < minimum_observations:
        raise ValueError(
            "Not enough scenes for this composite"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    composite_path = (
        output_dir / f"{name}_median.tif"
    )
    count_path = (
        output_dir / f"{name}_valid_count.tif"
    )

    with ExitStack() as stack:
        sources = [
            stack.enter_context(rasterio.open(p))
            for p in scene_paths
        ]

        reference = sources[0]

        for source in sources[1:]:
            if (
                source.crs != reference.crs
                or source.transform != reference.transform
                or source.width != reference.width
                or source.height != reference.height
                or source.count != 6
            ):
                raise ValueError(
                    "Input scenes do not share a grid"
                )

        profile = reference.profile.copy()
        profile.update(
            dtype="float32",
            nodata=nodata,
            count=6,
            predictor=3,
        )

        count_profile = reference.profile.copy()
        count_profile.update(
            dtype="uint16",
            count=1,
            nodata=0,
            predictor=1,
        )

        with rasterio.open(
            composite_path, "w", **profile
        ) as composite_dst:
            with rasterio.open(
                count_path, "w", **count_profile
            ) as count_dst:

                for _, window in reference.block_windows(1):
                    arrays = [
                        source.read(
                            window=window,
                            masked=True,
                        ).filled(np.nan)
                        for source in sources
                    ]

                    stack_data = np.stack(
                        arrays, axis=0
                    )

                    # Every band must be valid.
                    per_scene_valid = np.all(
                        np.isfinite(stack_data),
                        axis=1,
                    )

                    counts = np.sum(
                        per_scene_valid,
                        axis=0,
                    ).astype(np.uint16)

                    # Ignore incomplete observations
                    # consistently across all bands.
                    stack_data = np.where(
                        per_scene_valid[:, None],
                        stack_data,
                        np.nan,
                    )

                    with warnings.catch_warnings():
                        warnings.simplefilter(
                            "ignore", RuntimeWarning
                        )

                        median = np.nanmedian(
                            stack_data,
                            axis=0,
                        )

                    insufficient = (
                        counts < minimum_observations
                    )

                    median[:, insufficient] = nodata

                    composite_dst.write(
                        median.astype(np.float32),
                        window=window,
                    )

                    count_dst.write(
                        counts,
                        1,
                        window=window,
                    )

        # Document source acquisitions.
        with rasterio.open(
            composite_path, "r+"
        ) as dst:
            dst.update_tags(
                composite="median",
                source_count=len(scene_paths),
                minimum_observations=minimum_observations,
                source_files=";".join(
                    p.name for p in scene_paths
                ),
            )

            for index, band in enumerate(BANDS, 1):
                dst.set_band_description(index, band)

    return composite_path, count_path
