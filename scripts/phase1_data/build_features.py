
import argparse
import json
import re
import sys

from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sarchaeology.features.sar_features import (
    SAR_FEATURE_NAMES,
    calculate_sar_temporal_features,
)

from sarchaeology.features.optical_features import (
    OPTICAL_INDEX_NAMES,
    OPTICAL_ANNUAL_NAMES,
    calculate_optical_indices,
    summarize_optical_quarters,
)


def load_config():
    with open(
        ROOT / "configs/preprocessing.yaml",
        "r",
        encoding="utf-8",
    ) as f:
        return yaml.safe_load(f)["features"]


def resolve_path(path):
    path = Path(path)

    if not path.is_absolute():
        path = ROOT / path

    if not path.exists():
        raise FileNotFoundError(path)

    return path


def check_grid(source, reference, bands):
    if source.count != bands:
        raise ValueError(
            f"{source.name}: expected {bands} bands, "
            f"found {source.count}"
        )

    if (
        source.crs != reference.crs
        or source.transform != reference.transform
        or source.width != reference.width
        or source.height != reference.height
    ):
        raise ValueError(
            f"Raster grid does not match: {source.name}"
        )


def make_profile(reference, count, dtype, nodata):
    profile = reference.profile.copy()

    profile.update(
        driver="GTiff",
        count=count,
        dtype=dtype,
        nodata=nodata,
        compress="deflate",
        predictor=3 if dtype == "float32" else 1,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        BIGTIFF="IF_SAFER",
    )

    return profile


def save_band_names(dst, names):
    for index, name in enumerate(names, 1):
        dst.set_band_description(index, name)


def build_sar_features(config, output_root, overwrite):
    sar_cfg = config["sar"]
    nodata = float(config["output_nodata"])

    manifest_path = resolve_path(
        "data/metadata/manifests/"
        "sar_rtc_processed.parquet"
    )

    manifest = pd.read_parquet(manifest_path)

    if "db_path" not in manifest.columns:
        raise ValueError(
            "SAR manifest has no db_path column"
        )

    if "stem" in manifest.columns:
        manifest = manifest.drop_duplicates(
            subset=["stem"],
            keep="last",
        )

    paths = [
        resolve_path(value)
        for value in manifest["db_path"].dropna()
    ]

    minimum = int(
        sar_cfg["min_valid_acquisitions"]
    )

    if len(paths) < minimum:
        raise RuntimeError(
            f"Need at least {minimum} SAR acquisitions, "
            f"but found {len(paths)}. "
            "Process more SAR dates first."
        )

    directory = output_root / "sar"
    directory.mkdir(parents=True, exist_ok=True)

    features_path = (
        directory / "sar_temporal_10m.tif"
    )

    counts_path = (
        directory / "sar_valid_count_10m.tif"
    )

    if (
        features_path.exists()
        and counts_path.exists()
        and not overwrite
    ):
        print("SAR features cached; use --overwrite to rebuild.")
    else:
        print(f"\nSAR: processing {len(paths)} acquisitions")

        with ExitStack() as stack:
            sources = [
                stack.enter_context(rasterio.open(p))
                for p in paths
            ]

            reference = sources[0]

            for source in sources:
                check_grid(source, reference, 2)

            features_dst = stack.enter_context(
                rasterio.open(
                    features_path,
                    "w",
                    **make_profile(
                        reference,
                        len(SAR_FEATURE_NAMES),
                        "float32",
                        nodata,
                    ),
                )
            )

            counts_dst = stack.enter_context(
                rasterio.open(
                    counts_path,
                    "w",
                    **make_profile(
                        reference, 1, "uint16", 0
                    ),
                )
            )

            save_band_names(
                features_dst,
                SAR_FEATURE_NAMES,
            )

            features_dst.update_tags(
                source="sentinel-1-rtc",
                source_acquisitions=len(paths),
                minimum_valid_acquisitions=minimum,
                units="dB_and_dB_statistics",
            )

            for _, window in reference.block_windows(1):
                # [time, 2, block_height, block_width]
                stack_data = np.stack(
                    [
                        src.read(
                            window=window,
                            masked=True,
                        ).filled(np.nan)
                        for src in sources
                    ],
                    axis=0,
                ).astype(np.float32)

                features, counts = (
                    calculate_sar_temporal_features(
                        stack_data,
                        min_valid_acquisitions=minimum,
                    )
                )

                features_dst.write(
                    np.where(
                        np.isfinite(features),
                        features,
                        nodata,
                    ).astype(np.float32),
                    window=window,
                )

                counts_dst.write(
                    counts,
                    1,
                    window=window,
                )

        print("SAR features:", features_path)
        print("SAR valid counts:", counts_path)

    return {
        "source_files": [
            str(p.relative_to(ROOT))
            for p in paths
        ],
        "features": str(features_path.relative_to(ROOT)),
        "valid_count": str(counts_path.relative_to(ROOT)),
        "band_names": list(SAR_FEATURE_NAMES),
    }


def build_quarter_indices(
    composite_path,
    count_path,
    output_path,
    minimum_observations,
    nodata,
):
    print("Calculating indices:", composite_path.name)

    with rasterio.open(composite_path) as source:
        with rasterio.open(count_path) as counts:
            check_grid(counts, source, 1)

            if source.count != 6:
                raise ValueError(
                    "Expected six Sentinel-2 bands: "
                    f"{composite_path}"
                )

            with rasterio.open(
                output_path,
                "w",
                **make_profile(
                    source,
                    len(OPTICAL_INDEX_NAMES),
                    "float32",
                    nodata,
                ),
            ) as dst:
                save_band_names(
                    dst,
                    OPTICAL_INDEX_NAMES,
                )

                dst.update_tags(
                    source_composite=composite_path.name,
                    source_observation_counts=count_path.name,
                    min_scene_observations=minimum_observations,
                )

                for _, window in source.block_windows(1):
                    reflectance = source.read(
                        window=window,
                        masked=True,
                    ).filled(np.nan).astype(np.float32)

                    n = counts.read(
                        1, window=window
                    )

                    valid = (
                        n >= minimum_observations
                    )

                    indices = calculate_optical_indices(
                        reflectance,
                        valid_mask=valid,
                    )

                    dst.write(
                        np.where(
                            np.isfinite(indices),
                            indices,
                            nodata,
                        ).astype(np.float32),
                        window=window,
                    )


def build_annual_optical(
    index_paths,
    output_path,
    count_path,
    minimum_quarters,
    nodata,
):
    print(
        f"Creating annual optical summary "
        f"from {len(index_paths)} quarters"
    )

    with ExitStack() as stack:
        sources = [
            stack.enter_context(rasterio.open(path))
            for path in index_paths
        ]

        reference = sources[0]

        for source in sources:
            check_grid(
                source,
                reference,
                len(OPTICAL_INDEX_NAMES),
            )

        dst = stack.enter_context(
            rasterio.open(
                output_path,
                "w",
                **make_profile(
                    reference,
                    len(OPTICAL_ANNUAL_NAMES),
                    "float32",
                    nodata,
                ),
            )
        )

        count_dst = stack.enter_context(
            rasterio.open(
                count_path,
                "w",
                **make_profile(
                    reference, 1, "uint16", 0
                ),
            )
        )

        save_band_names(
            dst,
            OPTICAL_ANNUAL_NAMES,
        )

        dst.update_tags(
            source_quarters=",".join(
                p.stem.split("_")[0]
                for p in index_paths
            ),
            minimum_valid_quarters=minimum_quarters,
        )

        for _, window in reference.block_windows(1):
            stack_data = np.stack(
                [
                    src.read(
                        window=window,
                        masked=True,
                    ).filled(np.nan)
                    for src in sources
                ],
                axis=0,
            ).astype(np.float32)

            features, counts = (
                summarize_optical_quarters(
                    stack_data,
                    min_valid_quarters=minimum_quarters,
                )
            )

            dst.write(
                np.where(
                    np.isfinite(features),
                    features,
                    nodata,
                ).astype(np.float32),
                window=window,
            )

            count_dst.write(
                counts,
                1,
                window=window,
            )


def build_optical_features(
    config,
    output_root,
    overwrite,
):
    optical_cfg = config["optical"]
    nodata = float(config["output_nodata"])

    directory = ROOT / (
        "data/processed/optical/composites"
    )

    composites = sorted(
        directory.glob("*_median.tif")
    )

    if not composites:
        raise RuntimeError(
            "No optical composites found. "
            "Complete Step 1.4 first."
        )

    destination = output_root / "optical"
    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    min_scenes = int(
        optical_cfg[
            "min_scene_observations_per_quarter"
        ]
    )

    min_quarters = int(
        optical_cfg["min_valid_quarters"]
    )

    by_year = {}
    records = []

    for composite in composites:
        quarter = composite.stem.split("_")[0]

        if not re.fullmatch(r"\d{4}Q[1-4]", quarter):
            raise ValueError(
                f"Unexpected composite name: {composite}"
            )

        observation_count = (
            composite.parent
            / f"{quarter}_valid_count.tif"
        )

        observation_count = resolve_path(
            observation_count
        )

        index_path = (
            destination
            / f"{quarter}_indices_10m.tif"
        )

        if not index_path.exists() or overwrite:
            build_quarter_indices(
                composite_path=composite,
                count_path=observation_count,
                output_path=index_path,
                minimum_observations=min_scenes,
                nodata=nodata,
            )
        else:
            print(f"Using cached indices: {quarter}")

        year = quarter[:4]
        by_year.setdefault(year, []).append(
            index_path
        )

        records.append({
            "quarter": quarter,
            "source_composite": str(
                composite.relative_to(ROOT)
            ),
            "index_file": str(
                index_path.relative_to(ROOT)
            ),
        })

    annual_outputs = []

    for year, index_paths in sorted(
        by_year.items()
    ):
        if len(index_paths) < min_quarters:
            print(
                f"{year}: annual summary skipped; "
                f"only {len(index_paths)} quarters available."
            )
            continue

        annual_path = (
            destination
            / f"{year}_optical_annual_10m.tif"
        )

        counts_path = (
            destination
            / f"{year}_optical_valid_quarters_10m.tif"
        )

        if (
            overwrite
            or not annual_path.exists()
            or not counts_path.exists()
        ):
            build_annual_optical(
                index_paths=index_paths,
                output_path=annual_path,
                count_path=counts_path,
                minimum_quarters=min_quarters,
                nodata=nodata,
            )

        annual_outputs.append({
            "year": year,
            "features": str(
                annual_path.relative_to(ROOT)
            ),
            "valid_quarters": str(
                counts_path.relative_to(ROOT)
            ),
        })

    return {
        "quarterly": records,
        "annual": annual_outputs,
        "index_band_names": list(
            OPTICAL_INDEX_NAMES
        ),
        "annual_band_names": list(
            OPTICAL_ANNUAL_NAMES
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=("sar", "optical", "all"),
        default="all",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    config = load_config()

    output_root = (
        ROOT / config["output_directory"]
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    provenance = {
        "processing_step": "1.6",
        "configuration": config,
    }

    if args.mode in ("sar", "all"):
        provenance["sar"] = build_sar_features(
            config, output_root, args.overwrite
        )

    if args.mode in ("optical", "all"):
        provenance["optical"] = build_optical_features(
            config, output_root, args.overwrite
        )

    manifest_dir = (
        ROOT / "data/metadata/manifests"
    )

    manifest_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path = (
        manifest_dir / "feature_products.json"
    )

    manifest_path.write_text(
        json.dumps(
            provenance,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nStep 1.6 complete.")
    print("Feature manifest:", manifest_path)


if __name__ == "__main__":
    main()
