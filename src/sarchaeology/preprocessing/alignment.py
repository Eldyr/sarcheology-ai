"""Placeholder module for SARchaeology AI v1.0."""

import json

from contextlib import ExitStack
from pathlib import Path

import numpy as np
import rasterio

from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window

from sarchaeology.features.sar_features import (
    SAR_FEATURE_NAMES,
)
from sarchaeology.features.optical_features import (
    OPTICAL_ANNUAL_NAMES,
)
from sarchaeology.features.terrain_features import (
    FEATURE_NAMES,
)


MODALITIES = ("sar", "optical", "terrain")

FEATURE_NAMES_ALL = (
    tuple("sar_" + n for n in SAR_FEATURE_NAMES)
    + tuple("optical_" + n for n in OPTICAL_ANNUAL_NAMES)
    + tuple("terrain_" + n for n in FEATURE_NAMES)
)

MASK_NAMES = (
    "sar_valid",
    "optical_valid",
    "terrain_valid",
    "joint_valid",
)

QUALITY_NAMES = (
    "sar_acquisition_count",
    "optical_valid_quarter_count",
    "terrain_available",
)


def same_grid(dataset, grid):
    """
    Check CRS, dimensions and affine transform.

    Having the same EPSG code alone is not enough.
    """
    return (
        dataset.crs == CRS.from_user_input(grid.crs)
        and dataset.transform == grid.transform
        and dataset.width == grid.width
        and dataset.height == grid.height
    )


def open_aligned(
    stack,
    path,
    grid,
    expected_bands,
    resampling,
    output_nodata,
    expected_names=None,
):
    """
    Open a raster on the reference grid.

    No interpolation occurs if the source
    already matches the reference grid.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    src = stack.enter_context(rasterio.open(path))

    if src.crs is None:
        raise ValueError(f"Missing CRS: {path}")

    if src.count != expected_bands:
        raise ValueError(
            f"{path.name}: expected {expected_bands} "
            f"bands, found {src.count}"
        )

    if expected_names is not None:
        if tuple(src.descriptions) != tuple(expected_names):
            raise ValueError(
                f"Unexpected band order in {path.name}. "
                f"Found: {src.descriptions}"
            )

    needs_warp = not same_grid(src, grid)

    details = {
        "source_file": str(path),
        "source_crs": str(src.crs),
        "source_width": src.width,
        "source_height": src.height,
        "source_resolution": list(src.res),
        "reprojected": needs_warp,
    }

    if not needs_warp:
        return src, details

    # Continuous values: bilinear interpolation.
    # Counts and masks: nearest-neighbor.
    if (
        resampling == Resampling.bilinear
        and src.nodata is None
    ):
        raise ValueError(
            f"{path.name}: no nodata metadata. "
            "Cannot safely interpolate."
        )

    aligned = stack.enter_context(
        WarpedVRT(
            src,
            crs=grid.crs,
            transform=grid.transform,
            width=grid.width,
            height=grid.height,
            src_nodata=(
                src.nodata
                if src.nodata is not None
                else 0
            ),
            nodata=output_nodata,
            resampling=resampling,
            init_dest_nodata=True,
        )
    )

    return aligned, details


def read_features(source, window):
    """
    Convert a masked Rasterio block to float32
    with NaN for invalid measurements.
    """
    return source.read(
        window=window,
        masked=True,
    ).astype(np.float32).filled(np.nan)


def read_quality(source, window):
    """Read an observation-count or mask raster."""
    values = source.read(
        1,
        window=window,
        masked=True,
    ).filled(0)

    return values.astype(np.uint16)


def make_profile(
    grid,
    bands,
    dtype,
    nodata,
):
    return {
        "driver": "GTiff",
        "width": grid.width,
        "height": grid.height,
        "count": bands,
        "dtype": dtype,
        "crs": grid.crs,
        "transform": grid.transform,
        "nodata": nodata,
        "compress": "deflate",
        "predictor": 3 if dtype == "float32" else 1,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "BIGTIFF": "IF_SAFER",
    }


def set_band_names(dataset, names):
    for i, name in enumerate(names, 1):
        dataset.set_band_description(i, name)


def build_multimodal_dataset(
    grid,
    paths,
    output_dir,
    stem,
    min_sar_observations=3,
    min_optical_quarters=2,
    nodata=-9999.0,
    require_joint_coverage=True,
    overwrite=False,
):
    """
    Build a 19-band, geographically aligned
    multimodal feature dataset.

    paths keys:
        sar_features, sar_quality,
        optical_features, optical_quality,
        terrain_features, terrain_quality

    Output:
        19-band features
        4-band validity masks
        3-band observation quality
        JSON alignment report
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs = {
        "features": (
            output_dir / f"{stem}_multimodal_10m.tif"
        ),
        "masks": (
            output_dir / f"{stem}_masks_10m.tif"
        ),
        "quality": (
            output_dir / f"{stem}_quality_10m.tif"
        ),
        "report": (
            output_dir / f"{stem}_alignment_report.json"
        ),
    }

    if not overwrite and any(
        p.exists() for p in outputs.values()
    ):
        raise FileExistsError(
            "Alignment output already exists. "
            "Use --overwrite to rebuild."
        )

    required = (
        "sar_features",
        "sar_quality",
        "optical_features",
        "optical_quality",
        "terrain_features",
        "terrain_quality",
    )

    for key in required:
        if key not in paths:
            raise ValueError(f"Missing input: {key}")

    band_schemas = {
        "sar": SAR_FEATURE_NAMES,
        "optical": OPTICAL_ANNUAL_NAMES,
        "terrain": FEATURE_NAMES,
    }

    report = {
        "reference_grid": {
            "crs": str(grid.crs),
            "width": grid.width,
            "height": grid.height,
            "transform": list(grid.transform)[:6],
        },
        "source_rasters": {},
        "bands": list(FEATURE_NAMES_ALL),
        "mask_bands": list(MASK_NAMES),
        "quality_bands": list(QUALITY_NAMES),
        "thresholds": {
            "sar_observations": min_sar_observations,
            "optical_quarters": min_optical_quarters,
        },
    }

    coverage = {
        "sar": 0,
        "optical": 0,
        "terrain": 0,
        "joint": 0,
    }

    with ExitStack() as stack:
        readers = {}
        quality_readers = {}

        for modality in MODALITIES:
            names = band_schemas[modality]

            source, details = open_aligned(
                stack=stack,
                path=paths[f"{modality}_features"],
                grid=grid,
                expected_bands=len(names),
                expected_names=names,
                resampling=Resampling.bilinear,
                output_nodata=nodata,
            )

            readers[modality] = source

            report["source_rasters"][
                f"{modality}_features"
            ] = details

            source, details = open_aligned(
                stack=stack,
                path=paths[f"{modality}_quality"],
                grid=grid,
                expected_bands=1,
                resampling=Resampling.nearest,
                output_nodata=0,
            )

            quality_readers[modality] = source

            report["source_rasters"][
                f"{modality}_quality"
            ] = details

        features_dst = stack.enter_context(
            rasterio.open(
                outputs["features"],
                "w",
                **make_profile(
                    grid,
                    len(FEATURE_NAMES_ALL),
                    "float32",
                    nodata,
                ),
            )
        )

        masks_dst = stack.enter_context(
            rasterio.open(
                outputs["masks"],
                "w",
                **make_profile(
                    grid, 4, "uint8", 0
                ),
            )
        )

        quality_dst = stack.enter_context(
            rasterio.open(
                outputs["quality"],
                "w",
                **make_profile(
                    grid, 3, "uint16", 0
                ),
            )
        )

        set_band_names(
            features_dst,
            FEATURE_NAMES_ALL,
        )

        set_band_names(
            masks_dst,
            MASK_NAMES,
        )

        set_band_names(
            quality_dst,
            QUALITY_NAMES,
        )

        features_dst.update_tags(
            description="Unnormalized multimodal features",
            source="SARchaeology AI",
        )

        # Process blocks to limit memory usage.
        block_size = 256

        for row in range(0, grid.height, block_size):
            for col in range(0, grid.width, block_size):
                h = min(block_size, grid.height - row)
                w = min(block_size, grid.width - col)

                window = Window(col, row, w, h)

                aoi = grid.aoi_mask[
                    row:row + h,
                    col:col + w,
                ]

                arrays = {
                    name: read_features(
                        readers[name], window
                    )
                    for name in MODALITIES
                }

                counts = {
                    name: read_quality(
                        quality_readers[name],
                        window,
                    )
                    for name in MODALITIES
                }

                # All feature channels of each
                # modality must be finite.
                sar_valid = (
                    aoi
                    & np.all(
                        np.isfinite(arrays["sar"]),
                        axis=0,
                    )
                    & (
                        counts["sar"]
                        >= min_sar_observations
                    )
                )

                optical_valid = (
                    aoi
                    & np.all(
                        np.isfinite(arrays["optical"]),
                        axis=0,
                    )
                    & (
                        counts["optical"]
                        >= min_optical_quarters
                    )
                )

                terrain_valid = (
                    aoi
                    & np.all(
                        np.isfinite(arrays["terrain"]),
                        axis=0,
                    )
                    & (counts["terrain"] > 0)
                )

                joint_valid = (
                    sar_valid
                    & optical_valid
                    & terrain_valid
                )

                validity = {
                    "sar": sar_valid,
                    "optical": optical_valid,
                    "terrain": terrain_valid,
                    "joint": joint_valid,
                }

                for name in coverage:
                    coverage[name] += int(
                        np.count_nonzero(validity[name])
                    )

                # Keep each modality independently valid.
                # Joint validity is saved separately.
                for name in MODALITIES:
                    arrays[name][
                        :, ~validity[name]
                    ] = nodata

                combined = np.concatenate(
                    [
                        arrays["sar"],
                        arrays["optical"],
                        arrays["terrain"],
                    ],
                    axis=0,
                ).astype(np.float32)

                mask_stack = np.stack(
                    [
                        sar_valid,
                        optical_valid,
                        terrain_valid,
                        joint_valid,
                    ],
                    axis=0,
                ).astype(np.uint8)

                quality_stack = np.stack(
                    [
                        counts["sar"],
                        counts["optical"],
                        terrain_valid.astype(np.uint16),
                    ],
                    axis=0,
                ).astype(np.uint16)

                # Exclude area outside AOI.
                quality_stack[:, ~aoi] = 0

                features_dst.write(
                    combined,
                    window=window,
                )

                masks_dst.write(
                    mask_stack,
                    window=window,
                )

                quality_dst.write(
                    quality_stack,
                    window=window,
                )

    aoi_pixels = int(
        np.count_nonzero(grid.aoi_mask)
    )

    report["aoi_pixels"] = aoi_pixels
    report["valid_pixels"] = coverage

    report["valid_fractions"] = {
        name: (
            count / aoi_pixels
            if aoi_pixels > 0 else 0.0
        )
        for name, count in coverage.items()
    }

    if (
        require_joint_coverage
        and coverage["joint"] == 0
    ):
        raise RuntimeError(
            "No jointly valid pixels found. "
            "Inspect the alignment and mask inputs."
        )

    outputs["report"].write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    return outputs, report
