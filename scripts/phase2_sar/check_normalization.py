
import sys
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sarchaeology.data.dataloaders import (
    create_sar_dataloaders,
)

from sarchaeology.preprocessing.sar_normalization import (
    SARNormalizer,
    file_sha256,
)


def main():
    config = yaml.safe_load(
        (
            ROOT / "configs/training/sar.yaml"
        ).read_text(encoding="utf-8")
    )

    stats_path = (
        ROOT
        / config["normalization"]["output_path"]
    )

    normalizer = SARNormalizer.from_json(
        stats_path
    )

    statistics = normalizer.statistics

    manifest_path = (
        ROOT
        / "data/metadata/manifests/splits"
        / config["dataset"]["dataset_id"]
        / "train.parquet"
    )

    if file_sha256(manifest_path) != statistics[
        "source_manifest_sha256"
    ]:
        raise RuntimeError(
            "Training manifest has changed since "
            "normalization was fitted. "
            "Refit the statistics."
        )

    if statistics["dataset_id"] != config[
        "dataset"
    ]["dataset_id"]:
        raise RuntimeError(
            "Normalization dataset ID mismatch"
        )

    datasets, loaders = create_sar_dataloaders(
        project_root=ROOT,
        config=config,
        normalizer=normalizer,
    )

    for split in ("train", "val"):
        batch = next(iter(loaders[split]))

        images = batch["image"]
        masks = batch["valid_mask"]

        assert images.shape[1:] == (
            2, 128, 128
        )

        assert masks.shape[1:] == (
            1, 128, 128
        )

        assert torch.isfinite(images).all()

        assert torch.all(
            images >= normalizer.clip_min
        )

        assert torch.all(
            images <= normalizer.clip_max
        )

        assert torch.all(
            images[
                ~(masks > 0).expand_as(images)
            ] == 0
        )

        print(f"\n{split.upper()} NORMALIZED BATCH")
        print("Shape:", tuple(images.shape))
        print("Minimum:", float(images.min()))
        print("Maximum:", float(images.max()))
        print("Mean:", float(images.mean()))
        print(
            "Valid fraction:",
            float(masks.mean()),
        )

    if torch.cuda.is_available():
        batch = next(iter(loaders["train"]))

        images = batch["image"].to(
            "cuda",
            non_blocking=True,
        )

        masks = batch["valid_mask"].to(
            "cuda",
            non_blocking=True,
        )

        print("\nGPU:", torch.cuda.get_device_name(0))
        print("Normalized image device:", images.device)
        print("Mask device:", masks.device)

    print(
        "\nPhase 2.2 normalization check passed."
    )


if __name__ == "__main__":
    main()
