
"""Phase 2.4 — SAR CAE architecture smoke test."""

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sarchaeology.data.dataloaders import (
    create_sar_dataloaders,
)
from sarchaeology.models.sar_cae import (
    build_sar_cae,
    count_parameters,
)
from sarchaeology.preprocessing.sar_normalization import (
    SARNormalizer,
)
from sarchaeology.training.experiment import (
    load_training_config,
    resolve_amp,
    resolve_device,
    set_global_seed,
    sha256_file,
)


def main():
    config = load_training_config(
        ROOT / "configs/training/sar.yaml"
    )

    set_global_seed(
        config["training"]["seed"],
        deterministic=config["training"][
            "deterministic"
        ],
    )

    device = resolve_device(
        config["training"]["device"]
    )

    amp = resolve_amp(
        device,
        enabled=config["training"][
            "mixed_precision"
        ],
        dtype=config["training"]["amp_dtype"],
    )

    # Use the original train-fitted statistics.
    stats_path = (
        ROOT
        / config["normalization"]["output_path"]
    )

    normalizer = SARNormalizer.from_json(
        stats_path
    )

    train_manifest = (
        ROOT
        / "data/metadata/manifests/splits"
        / config["dataset"]["dataset_id"]
        / "train.parquet"
    )

    if (
        sha256_file(train_manifest)
        != normalizer.statistics[
            "source_manifest_sha256"
        ]
    ):
        raise RuntimeError(
            "Normalization/train manifest mismatch"
        )

    datasets, loaders = create_sar_dataloaders(
        project_root=ROOT,
        config=config,
        normalizer=normalizer,
    )

    batch = next(iter(loaders["train"]))

    images = batch["image"].to(
        device,
        non_blocking=True,
    )

    masks = batch["valid_mask"].to(
        device,
        non_blocking=True,
    )

    model = build_sar_cae(config).to(device)
    model.eval()

    with torch.inference_mode():
        with torch.autocast(
            device_type=device.type,
            dtype=amp["dtype"],
            enabled=amp["enabled"],
        ):
            outputs = model(
                images,
                valid_mask=masks,
            )

    reconstruction = outputs["reconstruction"]
    latent = outputs["latent_map"]
    embedding = outputs["embedding"]

    batch_size = images.shape[0]

    assert reconstruction.shape == images.shape

    assert latent.shape == (
        batch_size,
        config["model"]["latent_channels"],
        images.shape[-2] // 8,
        images.shape[-1] // 8,
    )

    assert embedding.shape == (
        batch_size,
        config["model"]["embedding_dim"],
    )

    for name, tensor in outputs.items():
        assert torch.isfinite(tensor).all(), (
            f"{name} contains non-finite values"
        )

    print("\nSARchaeology AI — Phase 2.4")
    print("Architecture:", model.__class__.__name__)
    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print("AMP enabled:", amp["enabled"])
    print(
        "Trainable parameters:",
        f"{count_parameters(model):,}",
    )

    print("\nREAL SAR BATCH")
    print("Input:", tuple(images.shape))
    print("Mask:", tuple(masks.shape))

    print("\nMODEL OUTPUTS")
    print(
        "Reconstruction:",
        tuple(reconstruction.shape),
    )
    print("Latent map:", tuple(latent.shape))
    print("Embedding:", tuple(embedding.shape))

    print(
        "\nInput range:",
        float(images.min()),
        float(images.max()),
    )

    print(
        "Reconstruction range:",
        float(reconstruction.min()),
        float(reconstruction.max()),
    )

    if device.type == "cuda":
        print(
            "\nPeak CUDA allocated:",
            round(
                torch.cuda.max_memory_allocated()
                / (1024 ** 3),
                3,
            ),
            "GiB",
        )

    print(
        "\nPhase 2.4 SAR CAE architecture "
        "smoke test passed."
    )


if __name__ == "__main__":
    main()
