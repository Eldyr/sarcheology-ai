
import argparse
import hashlib
import json
import sys

from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sarchaeology.evaluation.spatial_splits import (
    assign_spatial_splits,
    save_split_manifests,
)


def load_yaml(path):
    with open(ROOT / path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--year",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--axis",
        choices=["x", "y"],
        default=None,
    )

    parser.add_argument(
        "--gap-m",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
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

    config = preprocessing_cfg["spatial_split"]

    year = (
        args.year
        if args.year is not None
        else config["year"]
    )

    axis = args.axis or config["axis"]

    gap_m = (
        args.gap_m
        if args.gap_m is not None
        else config["gap_m"]
    )

    study = data_cfg["study_area"]
    aoi_id = study["id"]

    dataset_name = f"{aoi_id}_{year}"

    manifest_path = (
        ROOT
        / "data/metadata/manifests"
        / f"{dataset_name}_patches.parquet"
    )

    metadata_path = (
        ROOT
        / "data/patches/unassigned"
        / dataset_name
        / "dataset_metadata.json"
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Patch manifest missing: {manifest_path}"
        )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Patch metadata missing: {metadata_path}"
        )

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    manifest = pd.read_parquet(manifest_path)

    if config["require_complete_dataset"]:
        if metadata.get("preview_limited", True):
            raise RuntimeError(
                "The patch dataset is a preview. "
                "Run create_patches.py --max-patches 0 first."
            )

        expected = metadata.get(
            "accepted_patches"
        )

        if expected != len(manifest):
            raise RuntimeError(
                "Patch manifest and metadata disagree "
                f"({len(manifest)} vs {expected})."
            )

    patch_span_m = (
        float(data_cfg["patches"]["size"])
        * float(study["resolution_m"])
    )

    print("\nSARchaeology AI — Step 1.9")
    print("Study area:", aoi_id)
    print("Year:", year)
    print("Input patches:", len(manifest))
    print("Split axis:", axis)
    print("Exclusion buffer:", gap_m, "m")
    print("Patch ground size:", patch_span_m, "m")

    assigned, report = assign_spatial_splits(
        manifest,
        axis=axis,
        train_fraction=config["train_fraction"],
        val_fraction=config["val_fraction"],
        test_fraction=config["test_fraction"],
        gap_m=gap_m,
        expected_crs=study["target_crs"],
        patch_span_m=patch_span_m,
    )

    # Confirm that files referenced by the
    # three model splits are present locally.
    selected = assigned[
        assigned["split"].isin(
            ["train", "val", "test"]
        )
    ]

    missing_files = []

    for filename in selected["path"]:
        path = Path(filename)

        if not path.is_absolute():
            path = ROOT / path

        if not path.is_file():
            missing_files.append(str(path))

    if missing_files:
        raise FileNotFoundError(
            "Missing patch archives. Examples:\n"
            + "\n".join(missing_files[:5])
        )

    report["dataset"] = dataset_name

    report["input_manifest"] = str(
        manifest_path.relative_to(ROOT)
    )

    report["input_sha256"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()

    print("\nSplit results:")

    for name, count in report["counts"].items():
        fraction = report[
            "fraction_of_all_patches"
        ][name]

        print(
            f"  {name:8s}: "
            f"{count:5d} patches ({fraction:.1%})"
        )

    print("\nGeographic separation:")

    for name, distance in report[
        "actual_separation_m"
    ].items():
        print(f"  {name}: {distance:.1f} m")

    if args.dry_run:
        print(
            "\nDry run completed. No manifests saved."
        )
        return

    output_dir = (
        ROOT
        / config["output_directory"]
        / dataset_name
    )

    report_file = output_dir / "split_report.json"

    if report_file.exists() and not args.overwrite:
        raise FileExistsError(
            "Split files already exist. "
            "Use --overwrite to regenerate them."
        )

    paths = save_split_manifests(
        assigned,
        report,
        output_dir,
    )

    print("\nGenerated manifests:")

    for name, path in paths.items():
        print(f"  {name}: {path}")

    print("\nStep 1.9 complete.")


if __name__ == "__main__":
    main()
