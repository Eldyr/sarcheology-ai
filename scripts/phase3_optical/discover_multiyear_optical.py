
"""Phase 3.3.2 — discover extra Sentinel-2 years.

Reuses the existing Phase 1 STAC discovery functions.
Does not modify original SAR/DEM/2024 manifests.

Example:
    uv run python scripts/phase3_optical/discover_multiyear_optical.py \
        --years 2022 2023 2025
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sarchaeology.ingestion.catalog import (
    load_config,
    load_aoi,
    connect_catalog,
    search_collection,
    select_optical_items,
    save_items,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[2022, 2023, 2025],
    )

    parser.add_argument(
        "--max-items-per-year",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    years = sorted(set(args.years))

    if 2024 in years:
        parser.error(
            "2024 is protected by this discovery workflow"
        )

    if any(year < 2017 or year > 2100 for year in years):
        parser.error("Unexpected Sentinel-2 year")

    config = load_config(
        str(ROOT / "configs/data.yaml")
    )

    aoi = load_aoi(
        str(ROOT / config["study_area"]["geojson"])
    )

    source = config["satellite_sources"]
    optical_cfg = source["sentinel2"]

    catalog = connect_catalog(
        source["stac_url"]
    )

    collection = optical_cfg["collection"]

    max_items = (
        args.max_items_per_year
        if args.max_items_per_year is not None
        else int(optical_cfg["max_items"])
    )

    if max_items < 1:
        parser.error(
            "max-items-per-year must be positive"
        )

    selected = {}
    counts = {}

    for year in years:
        date_range = (
            f"{year}-01-01/{year}-12-31"
        )

        print(
            f"\nDiscovering Sentinel-2 {year}: "
            f"{date_range}"
        )

        found = search_collection(
            catalog,
            collection,
            aoi.bounds,
            date_range,
            max_items=max_items,
        )

        # Stop rather than silently accepting a
        # potentially truncated yearly search.
        if len(found) >= max_items:
            raise RuntimeError(
                f"{year}: search returned {len(found)} "
                f"items, reaching the configured limit "
                f"{max_items}. Increase "
                "--max-items-per-year and rerun."
            )

        filtered = select_optical_items(
            found,
            max_cloud=optical_cfg["max_cloud_cover"],
        )

        yearly = [
            item for item in filtered
            if item.datetime is not None
            and item.datetime.year == year
        ]

        print("  STAC results:", len(found))
        print("  Selected scenes:", len(yearly))

        counts[str(year)] = {
            "discovered": len(found),
            "selected": len(yearly),
        }

        for item in yearly:
            selected[item.id] = item

    items = sorted(
        selected.values(),
        key=lambda item: item.datetime,
    )

    if not items:
        raise RuntimeError(
            "No eligible Sentinel-2 scenes discovered"
        )

    quarter_counts = Counter(
        f"{item.datetime.year}Q"
        f"{(item.datetime.month - 1) // 3 + 1}"
        for item in items
    )

    print("\nSelected scenes per quarter:")

    for year in years:
        for q in range(1, 5):
            key = f"{year}Q{q}"
            print(
                f"  {key}: {quarter_counts.get(key, 0)}"
            )

    missing = [
        f"{year}Q{q}"
        for year in years
        for q in range(1, 5)
        if quarter_counts.get(f"{year}Q{q}", 0) == 0
    ]

    if missing:
        print(
            "\nWARNING: No selected scenes for:",
            ", ".join(missing),
        )

    # Dedicated metadata directory: do not touch
    # Phase 1 STAC JSON files.
    metadata_root = (
        ROOT
        / "data/metadata/stac/phase3_multiyear"
    )

    manifest_path = (
        ROOT
        / "data/metadata/manifests"
        / "optical_multiyear_scenes.parquet"
    )

    report_path = (
        ROOT
        / "outputs/quality"
        / "phase3_multiyear_discovery.json"
    )

    # Protect previous discovery runs.
    for output in (manifest_path, report_path):
        if output.exists():
            raise FileExistsError(
                f"Refusing to overwrite {output}"
            )

    dataframe = save_items(
        items,
        "optical",
        metadata_root,
    )

    manifest_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    dataframe.to_parquet(
        manifest_path,
        index=False,
    )

    report = {
        "phase": "3.3.2",
        "years": years,
        "collection": collection,
        "max_cloud_cover": optical_cfg[
            "max_cloud_cover"
        ],
        "counts": counts,
        "quarter_counts": dict(quarter_counts),
        "missing_quarters": missing,
        "total_selected": len(items),
        "manifest": str(
            manifest_path.relative_to(ROOT)
        ),
    }

    report_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(
        "\nTotal selected:", len(items)
    )
    print("Manifest:", manifest_path)
    print("Discovery report:", report_path)
    print("\nDiscovery completed.")


if __name__ == "__main__":
    main()
