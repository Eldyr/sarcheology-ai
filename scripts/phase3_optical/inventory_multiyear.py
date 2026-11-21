
"""Inventory existing quarterly Sentinel-2 composites.

No files are modified or downloaded.
"""
import argparse
import json
from pathlib import Path

import rasterio


ROOT = Path(__file__).resolve().parents[2]

COMPOSITES = (
    ROOT / "data/processed/optical/composites"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--years", type=int, nargs="+",
        default=[2022, 2023, 2024, 2025],
    )
    args = parser.parse_args()

    records = []
    reference = None

    for year in args.years:
        for quarter in range(1, 5):
            label = f"{year}Q{quarter}"

            median = COMPOSITES / f"{label}_median.tif"
            count = COMPOSITES / f"{label}_valid_count.tif"

            record = {
                "period": label,
                "median_exists": median.exists(),
                "count_exists": count.exists(),
                "ready": False,
            }

            if median.exists() and count.exists():
                with rasterio.open(median) as ds:
                    current = (
                        ds.crs.to_string() if ds.crs else None,
                        ds.transform,
                        ds.width,
                        ds.height,
                    )

                    record.update({
                        "bands": ds.count,
                        "crs": current[0],
                        "width": ds.width,
                        "height": ds.height,
                    })

                    if label == "2024Q1":
                        reference = current

                with rasterio.open(count) as ds:
                    record["count_bands"] = ds.count
                    record["count_grid_matches"] = (
                        ds.crs.to_string()
                        if ds.crs else None,
                        ds.transform,
                        ds.width,
                        ds.height,
                    ) == current

                record["ready"] = (
                    record["bands"] == 6
                    and record["count_bands"] == 1
                    and record["count_grid_matches"]
                )

            records.append(record)

    # Check against the 2024 reference grid.
    reference_file = COMPOSITES / "2024Q1_median.tif"

    if reference_file.exists():
        with rasterio.open(reference_file) as ds:
            reference = (
                ds.crs.to_string() if ds.crs else None,
                ds.transform,
                ds.width,
                ds.height,
            )

    for record in records:
        if not record["ready"]:
            continue

        with rasterio.open(
            COMPOSITES / f"{record['period']}_median.tif"
        ) as ds:
            current = (
                ds.crs.to_string() if ds.crs else None,
                ds.transform,
                ds.width,
                ds.height,
            )

        record["aligned_to_2024"] = (
            current == reference
        )
        record["ready"] = (
            record["ready"]
            and record["aligned_to_2024"]
        )

    output = (
        ROOT / "outputs/quality"
        / "phase3_multiyear_inventory.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(records, indent=2, default=str),
        encoding="utf-8",
    )

    for record in records:
        status = (
            "READY" if record["ready"] else "MISSING/INVALID"
        )
        print(f"{record['period']}: {status}")

    print(
        "\nReady quarters:",
        sum(record["ready"] for record in records),
        "/", len(records),
    )
    print("Saved:", output)


if __name__ == "__main__":
    main()
