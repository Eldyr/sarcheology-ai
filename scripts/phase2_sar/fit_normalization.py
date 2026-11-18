
import argparse
import json
import sys

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sarchaeology.preprocessing.sar_normalization import (
    fit_sar_statistics,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    config_path = (
        ROOT / "configs/training/sar.yaml"
    )

    config = yaml.safe_load(
        config_path.read_text(
            encoding="utf-8"
        )
    )

    dataset_cfg = config["dataset"]
    norm_cfg = config["normalization"]

    output_path = (
        ROOT / norm_cfg["output_path"]
    )

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Statistics already exist: {output_path}. "
            "Use --overwrite to refit."
        )

    if not (
        norm_cfg["min_valid_fraction"]
        == dataset_cfg["min_valid_fraction"]
    ):
        raise ValueError(
            "Normalization and Dataset validity "
            "thresholds must agree"
        )

    print("\nSARchaeology AI — Phase 2.2")
    print(
        "Fitting normalization on TRAIN ONLY"
    )

    statistics = fit_sar_statistics(
        project_root=ROOT,
        dataset_id=dataset_cfg["dataset_id"],
        pixels_per_acquisition=norm_cfg[
            "pixels_per_acquisition"
        ],
        min_valid_fraction=norm_cfg[
            "min_valid_fraction"
        ],
        min_scale=norm_cfg["min_scale"],
        seed=norm_cfg["seed"],
        clip_min=norm_cfg["clip_min"],
        clip_max=norm_cfg["clip_max"],
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            statistics,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nTraining observations:",
          statistics["eligible_observations"])

    for i, name in enumerate(
        statistics["channel_names"]
    ):
        print(
            f"{name}: "
            f"median={statistics['median_db'][i]:.4f} dB, "
            f"IQR={statistics['iqr_db'][i]:.4f} dB, "
            f"scale={statistics['scale_db'][i]:.4f} dB"
        )

    print(
        "\nSample counts:",
        statistics["sample_counts"],
    )

    print("Saved:", output_path)
    print("\nPhase 2.2 statistics fitted.")


if __name__ == "__main__":
    main()
