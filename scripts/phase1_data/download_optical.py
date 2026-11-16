"""Placeholder module for SARchaeology AI v1.0."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pystac
import rasterio
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from shapely.geometry import shape

from sarchaeology.ingestion.sentinel2 import (
    read_optical_scene,
)

from sarchaeology.preprocessing.sar import create_grid

from sarchaeology.preprocessing.optical import (
    prepare_optical_scene,
    save_optical_scene,
    save_rgb_preview,
    make_quarterly_composite,
)


def load_yaml(filename):
    with open(
        ROOT / filename, "r", encoding="utf-8"
    ) as f:
        return yaml.safe_load(f)


def load_aoi(filename):
    with open(
        ROOT / filename, "r", encoding="utf-8"
    ) as f:
        collection = json.load(f)

    return shape(
        collection["features"][0]["geometry"]
    )


def load_items():
    """Load STAC scene metadata from Step 1.2."""
    path = (
        ROOT
        / "data/metadata/manifests/optical_scenes.parquet"
    )

    manifest = pd.read_parquet(path)
    items = []

    for filename in manifest["item_path"]:
        item_path = Path(filename)

        if not item_path.is_absolute():
            item_path = ROOT / item_path

        metadata = json.loads(
            item_path.read_text(encoding="utf-8")
        )

        items.append(
            pystac.Item.from_dict(metadata)
        )

    return sorted(
        items,
        key=lambda item: item.datetime,
    )


def quarter_name(item):
    quarter = (item.datetime.month - 1) // 3 + 1

    return f"{item.datetime.year}Q{quarter}"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max-scenes",
        type=int,
        default=1,
        help="Number of scenes to process; 0 means all",
    )

    parser.add_argument(
        "--month",
        type=str,
        default=None,
        help="Optional YYYY-MM filter",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    data_cfg = load_yaml("configs/data.yaml")
    optical_cfg = load_yaml(
        "configs/preprocessing.yaml"
    )["optical"]

    aoi = load_aoi(
        data_cfg["study_area"]["geojson"]
    )

    grid = create_grid(
        aoi,
        target_crs=data_cfg[
            "study_area"
        ]["target_crs"],
        resolution=data_cfg[
            "study_area"
        ]["resolution_m"],
    )

    print("Target grid:", grid.width, grid.height)
    print("Target CRS:", grid.crs)

    items = load_items()

    if args.month:
        items = [
            item for item in items
            if item.datetime.strftime("%Y-%m")
            == args.month
        ]

    if args.max_scenes > 0:
        items = items[:args.max_scenes]

    if not items:
        raise RuntimeError("No optical scenes selected")

    output_root = (
        ROOT / optical_cfg["output_directory"]
    )

    scene_dir = output_root / "scenes"
    composite_dir = output_root / "composites"

    preview_dir = (
        ROOT / optical_cfg["preview_directory"]
    )

    scene_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for index, item in enumerate(items, 1):
        quarter = quarter_name(item)
        stem = f"{quarter}_{item.id}"

        print(
            f"\n[{index}/{len(items)}] {item.id}"
        )

        reflectance_path = (
            scene_dir / f"{stem}_reflectance.tif"
        )

        mask_path = scene_dir / f"{stem}_mask.tif"

        scl_path = scene_dir / f"{stem}_scl.tif"

        existing = all(
            path.exists()
            for path in (
                reflectance_path,
                mask_path,
                scl_path,
            )
        )

        if existing and not args.overwrite:
            print("  Using cached outputs")

            with rasterio.open(mask_path) as ds:
                mask = ds.read(1) == 1

            coverage = (
                mask.sum()
                / max(grid.aoi_mask.sum(), 1)
            )

        else:
            print("  Reading six spectral bands + SCL")

            raw, scl, quant, offsets = (
                read_optical_scene(item, grid)
            )

            reflectance, valid, coverage = (
                prepare_optical_scene(
                    raw,
                    scl,
                    grid,
                    quant,
                    offsets,
                    optical_cfg,
                )
            )

            print(f"  Valid AOI: {coverage:.2%}")

            if coverage < optical_cfg[
                "minimum_scene_valid_fraction"
            ]:
                print(
                    "  Skipping: insufficient valid pixels"
                )
                continue

            save_optical_scene(
                reflectance=reflectance,
                valid=valid,
                scl=scl,
                grid=grid,
                directory=scene_dir,
                stem=stem,
                item_id=item.id,
                baseline=item.properties.get(
                    "s2:processing_baseline"
                ),
                quantification=quant,
                offsets=offsets,
                nodata=optical_cfg["output_nodata"],
            )

        preview_path = (
            preview_dir / f"{stem}.png"
        )

        if not preview_path.exists() or args.overwrite:
            save_rgb_preview(
                reflectance_path,
                preview_path,
            )

        rows.append({
            "id": item.id,
            "datetime": item.datetime.isoformat(),
            "quarter": quarter,
            "cloud_cover": item.properties.get(
                "eo:cloud_cover"
            ),
            "processing_baseline": item.properties.get(
                "s2:processing_baseline"
            ),
            "valid_fraction": float(coverage),
            "reflectance_path": str(
                reflectance_path.relative_to(ROOT)
            ),
            "mask_path": str(
                mask_path.relative_to(ROOT)
            ),
            "scl_path": str(
                scl_path.relative_to(ROOT)
            ),
        })

    manifest_path = (
        ROOT
        / "data/metadata/manifests/"
        / "optical_processed.parquet"
    )

    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if rows:
        current = pd.DataFrame(rows)

        if manifest_path.exists():
            previous = pd.read_parquet(manifest_path)

            current = pd.concat(
                [previous, current],
                ignore_index=True,
            )

        current = current.drop_duplicates(
            subset=["id"],
            keep="last",
        )

        current.to_parquet(
            manifest_path,
            index=False,
        )

    # Build or refresh composites from all
    # previously processed scenes.
    quarter_paths = {}

    for raster_path in sorted(
        scene_dir.glob("*_reflectance.tif")
    ):
        quarter = raster_path.name[:6]

        quarter_paths.setdefault(
            quarter, []
        ).append(raster_path)

    min_obs = optical_cfg[
        "minimum_composite_observations"
    ]

    for quarter, paths in sorted(
        quarter_paths.items()
    ):
        if len(paths) < min_obs:
            print(
                f"{quarter}: composite deferred "
                f"({len(paths)} scene)"
            )
            continue

        print(
            f"\nBuilding {quarter} composite "
            f"from {len(paths)} scenes"
        )

        median_path, count_path = (
            make_quarterly_composite(
                paths,
                composite_dir,
                quarter,
                minimum_observations=min_obs,
                nodata=optical_cfg["output_nodata"],
            )
        )

        preview_path = (
            preview_dir / f"{quarter}_median.png"
        )

        save_rgb_preview(
            median_path,
            preview_path,
        )

        print("Composite:", median_path)
        print("Observation count:", count_path)

    print("\nOptical processing finished.")
    print("Manifest:", manifest_path)


if __name__ == "__main__":
    main()
