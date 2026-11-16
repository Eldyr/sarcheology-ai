
import argparse
import json
import sys

from pathlib import Path

import yaml
from shapely.geometry import shape


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sarchaeology.preprocessing.sar import create_grid
from sarchaeology.preprocessing.alignment import (
    build_multimodal_dataset,
)


def load_yaml(path):
    with open(
        ROOT / path,
        "r",
        encoding="utf-8",
    ) as f:
        return yaml.safe_load(f)


def load_aoi(path):
    with open(
        ROOT / path,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    if data["type"] == "FeatureCollection":
        geometry = data["features"][0]["geometry"]
    elif data["type"] == "Feature":
        geometry = data["geometry"]
    else:
        geometry = data

    return shape(geometry)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--year",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    data_cfg = load_yaml("configs/data.yaml")
    preprocessing_cfg = load_yaml(
        "configs/preprocessing.yaml"
    )

    config = preprocessing_cfg["alignment"]

    year = (
        args.year
        if args.year is not None
        else config["year"]
    )

    study = data_cfg["study_area"]
    aoi_id = study["id"]

    geometry = load_aoi(
        study["geojson"]
    )

    grid = create_grid(
        geometry,
        target_crs=study["target_crs"],
        resolution=study["resolution_m"],
    )

    print("SARchaeology AI — Step 1.7")
    print("Study area:", aoi_id)
    print("Year:", year)
    print("CRS:", grid.crs)
    print(
        "Grid dimensions:",
        grid.width,
        "x",
        grid.height,
    )

    paths = {
        "sar_features": ROOT / (
            "data/processed/features/sar/"
            "sar_temporal_10m.tif"
        ),
        "sar_quality": ROOT / (
            "data/processed/features/sar/"
            "sar_valid_count_10m.tif"
        ),
        "optical_features": ROOT / (
            f"data/processed/features/optical/"
            f"{year}_optical_annual_10m.tif"
        ),
        "optical_quality": ROOT / (
            f"data/processed/features/optical/"
            f"{year}_optical_valid_quarters_10m.tif"
        ),
        "terrain_features": ROOT / (
            f"data/processed/terrain/"
            f"{aoi_id}_terrain_10m.tif"
        ),
        "terrain_quality": ROOT / (
            f"data/processed/terrain/"
            f"{aoi_id}_terrain_mask.tif"
        ),
    }

    missing = [
        str(path)
        for path in paths.values()
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing required files:\n"
            + "\n".join(missing)
        )

    print("\nAll six input files found.")

    output_dir = (
        ROOT / config["output_directory"]
    )

    stem = f"{aoi_id}_{year}"

    outputs, report = build_multimodal_dataset(
        grid=grid,
        paths=paths,
        output_dir=output_dir,
        stem=stem,
        min_sar_observations=config[
            "quality"
        ]["min_sar_acquisitions"],
        min_optical_quarters=config[
            "quality"
        ]["min_optical_quarters"],
        nodata=config["output_nodata"],
        require_joint_coverage=config[
            "require_joint_coverage"
        ],
        overwrite=args.overwrite,
    )

    print("\nCoverage results:")

    for name, fraction in report[
        "valid_fractions"
    ].items():
        print(
            f"  {name:8s}: {fraction:.2%}"
        )

    print("\nReprojection checks:")

    for name, details in report[
        "source_rasters"
    ].items():
        status = (
            "reprojected"
            if details["reprojected"]
            else "already aligned"
        )

        print(f"  {name}: {status}")

    print("\nOutput files:")

    for name, path in outputs.items():
        print(f"  {name}: {path}")

    print("\nStep 1.7 complete.")


if __name__ == "__main__":
    main()
