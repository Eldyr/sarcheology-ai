
import json
import sys

from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sarchaeology.data.dataloaders import (
    create_sar_dataloaders,
)


def main():
    config_path = (
        ROOT / "configs/training/sar.yaml"
    )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        config = yaml.safe_load(f)

    torch.manual_seed(
        config["training"]["seed"]
    )

    datasets, loaders = create_sar_dataloaders(
        project_root=ROOT,
        config=config,
    )

    print("\nSARchaeology AI — Phase 2.1")
    print("PyTorch:", torch.__version__)

    print(
        "CUDA available:",
        torch.cuda.is_available(),
    )

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    for name, dataset in datasets.items():
        print(
            f"\n{name.upper()} DATASET"
        )

        print(
            json.dumps(
                dataset.summary(),
                indent=2,
            )
        )

    # Verify one train batch.
    batch = next(
        iter(loaders["train"])
    )

    images = batch["image"]
    masks = batch["valid_mask"]

    print("\nFIRST TRAIN BATCH")
    print("Images:", tuple(images.shape))
    print("Masks:", tuple(masks.shape))
    print("Image dtype:", images.dtype)
    print("Mask dtype:", masks.dtype)

    print(
        "Finite images:",
        bool(torch.isfinite(images).all()),
    )

    print(
        "Valid pixel fraction:",
        round(
            float(masks.mean()),
            4,
        ),
    )

    print(
        "Patch IDs:",
        batch["patch_id"][:3],
    )

    print(
        "Acquisition dates:",
        batch["acquisition_date"][:3],
    )

    assert images.ndim == 4
    assert images.shape[1:] == (
        2, 128, 128
    )

    assert masks.shape == (
        images.shape[0],
        1,
        128,
        128,
    )

    assert torch.isfinite(images).all()
    assert torch.isfinite(masks).all()
    assert masks.sum() > 0

    # Test a single batch transfer.
    if torch.cuda.is_available():
        device = torch.device("cuda")

        gpu_images = images.to(
            device,
            non_blocking=True,
        )

        gpu_masks = masks.to(
            device,
            non_blocking=True,
        )

        print("\nGPU TRANSFER")
        print(
            "Images device:",
            gpu_images.device,
        )
        print(
            "Masks device:",
            gpu_masks.device,
        )

        del gpu_images
        del gpu_masks

    else:
        print(
            "\nCUDA unavailable; CPU smoke test only."
        )

    print(
        "\nPhase 2.1 DataLoader smoke test passed."
    )


if __name__ == "__main__":
    main()
