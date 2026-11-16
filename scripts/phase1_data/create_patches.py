"""Placeholder module for SARchaeology AI v1.0."""

import argparse
import sys
import re

from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sarchaeology.preprocessing.tiling import (
    build_patch_dataset,
)


def load_yaml(path):
    with open(
        ROOT / path,
        "r",
        encoding="utf-8",
    ) as f:
        return yaml.safe_load(f)


def resolve_path(value):
    path = Path(value)

    if not path.is_absolute():
        path = ROOT / path

    if not path.exists():
        raise FileNotFoundError(path)

    return path


def load_sar_sources(year):
    """
    Load all processed SAR acquisitions for the year.

    Do not re-download imagery.
    """
    manifest_path = resolve_path(
        "data/metadata/manifests/"
        "sar_rtc_processed.parquet"
    )

    manifest = pd.read_parquet(manifest_path)

    required = {"stem", "date", "db_path"}

    if not required.issubset(manifest.columns):
        raise ValueError(
            "SAR manifest missing required columns"
        )

    manifest = manifest.drop_duplicates(
        subset=["stem"],
        keep="last",
    ).copy()

    manifest["date"] = pd.to_datetime(
        manifest["date"]
    ).dt.strftime("%Y-%m-%d")

    manifest = manifest[
        manifest["date"].str.startswith(
            str(year)
        )
    ]

    manifest = manifest.sort_values(
        ["date", "stem"]
    )

    sources = []

    for _, row in manifest.iterrows():
        sources.append({
            "id": str(row["stem"]),
            "date": row["date"],
            "path": resolve_path(
                row["db_path"]
            ),
        })

    if not sources:
        raise RuntimeError(
            f"No processed SAR scenes found for {year}"
        )

    return sources


def load_optical_sources(year):
    """
    Load the six-band quarterly composites.

    Each quarter remains an individual temporal
    observation in the patch archive.
    """
    directory = resolve_path(
        "data/processed/optical/composites"
    )

    sources = []

    for path in sorted(
        directory.glob(f"{year}Q*_median.tif")
    ):
        quarter = path.stem.split("_")[0]

        if not re.fullmatch(
            rf"{year}Q[1-4]",
            quarter,
        ):
            continue

        sources.append({
            "quarter": quarter,
            "path": path,
        })

    if not sources:
        raise RuntimeError(
            f"No optical quarterly composites for {year}. "
            "Complete Step 1.4 first."
        )

    return sources


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--year",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max-patches",
        type=int,
        default=5,
        help=(
            "Maximum accepted patches to write. "
            "Use 0 to generate the full dataset. "
            "Defaults to a five-patch preview."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    data_cfg = load_yaml("configs/data.yaml")
    prep_cfg = load_yaml(
        "configs/preprocessing.yaml"
    )

    extraction_cfg = prep_cfg["patch_extraction"]
    patches_cfg = data_cfg["patches"]

    year = (
        args.year
        if args.year is not None
        else extraction_cfg["year"]
    )

    if (
        not extraction_cfg["include_sar_time_series"]
        or not extraction_cfg["include_optical_quarters"]
    ):
        raise ValueError(
            "This patch schema requires both raw "
            "SAR acquisitions and optical composites."
        )

    if extraction_cfg["compression"] != "npz_compressed":
        raise ValueError("Unsupported compression format")

    aoi_id = data_cfg["study_area"]["id"]

    dataset_name = f"{aoi_id}_{year}"

    aligned_dir = (
        ROOT / "data/processed/aligned"
    )

    features_path = resolve_path(
        aligned_dir
        / f"{dataset_name}_multimodal_10m.tif"
    )

    masks_path = resolve_path(
        aligned_dir
        / f"{dataset_name}_masks_10m.tif"
    )

    quality_path = resolve_path(
        aligned_dir
        / f"{dataset_name}_quality_10m.tif"
    )

    sar_sources = load_sar_sources(year)
    optical_sources = load_optical_sources(year)

    output_dir = (
        ROOT
        / extraction_cfg["output_directory"]
        / dataset_name
    )

    manifest_path = (
        ROOT
        / "data/metadata/manifests"
        / f"{dataset_name}_patches.parquet"
    )

    print("\nSARchaeology AI — Step 1.8")
    print("Study area:", aoi_id)
    print("Year:", year)
    print("SAR acquisitions:", len(sar_sources))
    print("Optical quarters:", len(optical_sources))
    print("Output directory:", output_dir)

    build_patch_dataset(
        features_path=features_path,
        masks_path=masks_path,
        quality_path=quality_path,
        sar_sources=sar_sources,
        optical_sources=optical_sources,
        output_dir=output_dir,
        manifest_path=manifest_path,
        project_root=ROOT,
        patch_size=int(
            patches_cfg["size"]
        ),
        stride=int(
            patches_cfg["stride"]
        ),
        min_valid_fraction=float(
            patches_cfg["minimum_valid_fraction"]
        ),
        max_patches=args.max_patches,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
