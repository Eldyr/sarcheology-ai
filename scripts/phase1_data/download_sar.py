"""Placeholder module for SARchaeology AI v1.0."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pystac
import rasterio
import yaml
import matplotlib.pyplot as plt

from shapely.geometry import shape


# Allow running the script from the repository.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sarchaeology.ingestion.downloader import read_rtc_band
from sarchaeology.preprocessing.sar import (
    create_grid,
    mosaic_rtc,
    write_sar_products,
)


def load_yaml(path):
    with open(ROOT / path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_aoi(path):
    with open(ROOT / path, "r", encoding="utf-8") as f:
        data = json.load(f)

    geometry = shape(
        data["features"][0]["geometry"]
    )

    if not geometry.is_valid:
        raise ValueError("Invalid study-area polygon")

    return geometry


def load_sar_items():
    """
    Reconstruct STAC Items from the files
    saved during Step 1.2.
    """

    manifest_path = (
        ROOT / "data/metadata/manifests/sar_scenes.parquet"
    )

    df = pd.read_parquet(manifest_path)

    items = []

    for item_path in df["item_path"]:
        path = Path(item_path)

        if not path.is_absolute():
            path = ROOT / path

        with open(path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        items.append(
            pystac.Item.from_dict(metadata)
        )

    return items


def group_acquisitions(items):
    """
    Group adjacent scene slices by:
      date, platform, absolute orbit,
      orbit direction, relative orbit.

    This prevents unrelated satellite passes
    from being mosaiced together.
    """

    groups = defaultdict(list)

    for item in items:
        props = item.properties

        date = item.datetime.date().isoformat()
        platform = item.id.split("_")[0]

        absolute_orbit = props.get(
            "sat:absolute_orbit"
        )

        # Sentinel-1 product IDs encode absolute orbit
        # in the seventh underscore-separated field.
        if absolute_orbit is None:
            absolute_orbit = item.id.split("_")[6]

        orbit_state = props.get("sat:orbit_state")
        relative_orbit = props.get(
            "sat:relative_orbit"
        )

        key = (
            date,
            platform,
            int(absolute_orbit),
            orbit_state,
            int(relative_orbit),
        )

        groups[key].append(item)

    return dict(sorted(groups.items()))


def make_preview(db_path, mask_path, output_path):
    """
    Generate a grayscale preview for VV and VH.
    """

    with rasterio.open(db_path) as ds:
        db = ds.read().astype(np.float32)

    with rasterio.open(mask_path) as ds:
        valid = ds.read(1) == 1

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 5)
    )

    for index, polarization in enumerate(["VV", "VH"]):
        image = db[index].copy()
        image[~valid] = np.nan

        finite = image[np.isfinite(image)]

        if finite.size == 0:
            raise RuntimeError(
                f"No valid pixels for {polarization}"
            )

        vmin, vmax = np.percentile(
            finite, [2, 98]
        )

        axes[index].imshow(
            image,
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
        )

        axes[index].set_title(
            f"{polarization} backscatter (dB)"
        )
        axes[index].axis("off")

    fig.tight_layout()

    output_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    fig.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Process a specific date, e.g. 2024-01-10",
    )

    parser.add_argument(
        "--max-dates",
        type=int,
        default=1,
        help="Number of acquisitions; 0 means all",
    )

    args = parser.parse_args()

    data_config = load_yaml("configs/data.yaml")
    processing = load_yaml(
        "configs/preprocessing.yaml"
    )["sar"]

    geometry = load_aoi(
        data_config["study_area"]["geojson"]
    )

    grid = create_grid(
        geometry=geometry,
        target_crs=data_config[
            "study_area"
        ]["target_crs"],
        resolution=data_config[
            "study_area"
        ]["resolution_m"],
    )

    print("Target CRS:", grid.crs)
    print("Grid:", grid.width, "x", grid.height)
    print("Pixel spacing: 10 m")

    items = load_sar_items()
    groups = group_acquisitions(items)

    if args.date:
        groups = {
            key: scenes
            for key, scenes in groups.items()
            if key[0] == args.date
        }

        if not groups:
            raise ValueError(
                f"No acquisitions found for {args.date}"
            )

    if args.max_dates > 0:
        groups = dict(
            list(groups.items())[:args.max_dates]
        )

    if not groups:
        raise RuntimeError("No SAR acquisitions selected")

    output_dir = (
        ROOT / processing["output_directory"]
    )

    preview_dir = (
        ROOT / processing["preview_directory"]
    )

    rows = []

    for key, scenes in groups.items():
        date, platform, absolute_orbit, state, rel = key

        stem = (
            f"{date.replace('-', '')}_"
            f"{platform}_"
            f"{state}_r{rel}_a{absolute_orbit}"
        )

        print("\nProcessing:", stem)
        print("Number of slices:", len(scenes))

        linear, valid, overlaps = mosaic_rtc(
            scenes,
            grid,
            read_rtc_band,
        )

        # Fraction of the actual AOI with both
        # polarizations available.
        inside = np.count_nonzero(grid.aoi_mask)
        observed = np.count_nonzero(valid)

        coverage = observed / max(inside, 1)

        print(f"AOI coverage: {coverage:.2%}")
        print("Overlapping source pixels:", overlaps)

        if observed == 0:
            raise RuntimeError(
                f"No valid SAR observations for {date}"
            )

        paths = write_sar_products(
            linear=linear,
            valid=valid,
            grid=grid,
            output_dir=output_dir,
            stem=stem,
            source_ids=[s.id for s in scenes],
            nodata=processing["output_nodata"],
        )

        preview_path = preview_dir / f"{stem}.png"

        make_preview(
            paths["db"],
            paths["mask"],
            preview_path,
        )

        rows.append({
            "stem": stem,
            "date": date,
            "platform": platform,
            "orbit_state": state,
            "relative_orbit": rel,
            "absolute_orbit": absolute_orbit,
            "scene_count": len(scenes),
            "source_ids": json.dumps(
                [s.id for s in scenes]
            ),
            "aoi_coverage": coverage,
            "overlap_pixels": overlaps,
            "linear_path": str(Path(paths["linear"]).relative_to(ROOT)),
            "db_path": str(Path(paths["db"]).relative_to(ROOT)),
            "mask_path": str(Path(paths["mask"]).relative_to(ROOT)),
        })

        print("Saved:", paths["linear"])
        print("Saved:", paths["db"])
        print("Saved:", paths["mask"])
        print("Preview:", preview_path)

    manifest_path = (
        ROOT
        / "data/metadata/manifests/sar_rtc_processed.parquet"
    )

    manifest_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    new_rows = pd.DataFrame(rows)

    if manifest_path.exists():
        previous = pd.read_parquet(manifest_path)

        new_rows = pd.concat(
            [previous, new_rows],
            ignore_index=True,
        )

        new_rows = new_rows.drop_duplicates(
            subset=["stem"],
            keep="last",
        )

    new_rows.to_parquet(
        manifest_path,
        index=False,
    )

    print("\nProcessing complete")
    print("Updated manifest:", manifest_path)


if __name__ == "__main__":
    main()
