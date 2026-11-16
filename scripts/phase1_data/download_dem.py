"""Placeholder module for SARchaeology AI v1.0."""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pystac
import rasterio
import yaml
from shapely.geometry import shape

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sarchaeology.features.terrain_features import (
    FEATURE_NAMES,
    calculate_terrain_features,
)
from sarchaeology.ingestion.dem import mosaic_dem
from sarchaeology.preprocessing.sar import create_grid
from sarchaeology.preprocessing.terrain import (
    align_terrain_features,
    make_buffered_grid,
    save_terrain_products,
)


def load_yaml(filename):
    with open(ROOT / filename, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_aoi(filename):
    with open(ROOT / filename, "r", encoding="utf-8") as f:
        geojson = json.load(f)

    return shape(geojson["features"][0]["geometry"])


def load_dem_items():
    """Load saved STAC item JSON documents.

    These were created during Phase 1.2.
    """
    manifest_path = ROOT / "data/metadata/manifests/dem_scenes.parquet"

    if not manifest_path.exists():
        raise FileNotFoundError(
            "DEM manifest missing. Run scene discovery first."
        )

    manifest = pd.read_parquet(manifest_path)
    items = []

    for filename in manifest["item_path"]:
        path = Path(filename)

        if not path.is_absolute():
            path = ROOT / path

        metadata = json.loads(path.read_text(encoding="utf-8"))

        item = pystac.Item.from_dict(metadata)

        if "data" in item.assets:
            items.append(item)

    if not items:
        raise RuntimeError("No usable Copernicus DEM assets found")

    return items


def make_preview(features_path, mask_path, output_path):
    """Generate visual QA images from the saved elevation, slope and TPI

    channels.
    """
    with rasterio.open(features_path) as ds:
        elevation = ds.read(1).astype(np.float32)
        slope = ds.read(2).astype(np.float32)
        tpi = ds.read(5).astype(np.float32)

    with rasterio.open(mask_path) as ds:
        valid = ds.read(1) == 1

    arrays = [
        elevation,
        slope,
        tpi,
    ]

    titles = [
        "Elevation (m)",
        "Slope (degrees)",
        "TPI (m)",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, data, title in zip(axes, arrays, titles):
        image = data.copy()
        image[~valid] = np.nan

        finite = image[np.isfinite(image)]

        if finite.size == 0:
            raise RuntimeError("No valid terrain pixels")

        vmin, vmax = np.percentile(finite, [2, 98])

        if np.isclose(vmin, vmax):
            vmin = float(finite.min()) - 0.01
            vmax = float(finite.max()) + 0.01

        ax.imshow(
            image,
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(title)
        ax.axis("off")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild previously generated terrain data",
    )

    args = parser.parse_args()

    data_cfg = load_yaml("configs/data.yaml")
    dem_cfg = load_yaml("configs/preprocessing.yaml")["dem"]

    aoi = load_aoi(data_cfg["study_area"]["geojson"])

    target_crs = data_cfg["study_area"]["target_crs"]

    # 10 m analysis grid: exactly the same function
    # and parameters as SAR and optical processing.
    analysis_grid = create_grid(
        aoi,
        target_crs=target_crs,
        resolution=data_cfg["study_area"]["resolution_m"],
    )

    # 30 m grid with surrounding elevation context.
    native_grid = make_buffered_grid(
        aoi,
        target_crs=target_crs,
        resolution_m=dem_cfg["processing_resolution_m"],
        buffer_m=dem_cfg["context_buffer_m"],
    )

    output_dir = ROOT / dem_cfg["output_directory"]
    preview_dir = ROOT / dem_cfg["preview_directory"]

    name = data_cfg["study_area"]["id"]

    paths_expected = [
        output_dir / f"{name}_dem_30m.tif",
        output_dir / f"{name}_terrain_10m.tif",
        output_dir / f"{name}_terrain_mask.tif",
    ]

    if all(p.exists() for p in paths_expected) and not args.overwrite:
        print("Existing terrain outputs found.")
        print("Use --overwrite to rebuild.")
        return

    items = load_dem_items()

    print("DEM tiles:", len(items))
    print(
        "Native DEM grid:",
        native_grid.width,
        "x",
        native_grid.height,
    )
    print(
        "Analysis grid:",
        analysis_grid.width,
        "x",
        analysis_grid.height,
    )

    # Read and mosaic real elevation measurements.
    dem = mosaic_dem(items, native_grid)

    coverage = np.isfinite(dem).mean()
    print(f"Buffered DEM coverage: {coverage:.2%}")

    if not np.all(np.isfinite(dem)):
        raise RuntimeError(
            "Missing DEM values on the buffered grid. "
            "Check tile coverage before deriving features."
        )

    print(
        "Elevation range:",
        float(np.min(dem)),
        float(np.max(dem)),
        "m",
    )

    print("Calculating terrain features...")

    native_features = calculate_terrain_features(
        dem,
        resolution_m=dem_cfg["processing_resolution_m"],
        smoothing_sigma=dem_cfg["smoothing_sigma_pixels"],
        tpi_sigma=dem_cfg["tpi_sigma_pixels"],
    )

    print("Aligning features to 10 m grid...")

    aligned_features = align_terrain_features(
        native_features,
        native_grid,
        analysis_grid,
    )

    paths = save_terrain_products(
        native_dem=dem,
        native_grid=native_grid,
        aligned_features=aligned_features,
        target_grid=analysis_grid,
        output_dir=output_dir,
        name=name,
        nodata=dem_cfg["output_nodata"],
    )

    preview_path = preview_dir / f"{name}_terrain.png"

    make_preview(
        paths["features"],
        paths["mask"],
        preview_path,
    )

    manifest_dir = ROOT / "data/metadata/manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.DataFrame([{
        "aoi_id": name,
        "source_collection": "cop-dem-glo-30",
        "source_item_ids": json.dumps([item.id for item in items]),
        "vertical_datum": "EGM2008",
        "native_spacing_m": dem_cfg["processing_resolution_m"],
        "analysis_spacing_m": data_cfg["study_area"]["resolution_m"],
        "features": ",".join(FEATURE_NAMES),
        "features_path": str(paths["features"].relative_to(ROOT)),
        "mask_path": str(paths["mask"].relative_to(ROOT)),
    }])

    manifest.to_parquet(
        manifest_dir / "dem_processed.parquet",
        index=False,
    )

    print("\nDEM processing complete.")
    print("DEM:", paths["dem"])
    print("Features:", paths["features"])
    print("Mask:", paths["mask"])
    print("Preview:", preview_path)


if __name__ == "__main__":
    main()