
"""Phase 2.5 — Real-data SAR reconstruction loss check."""

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

# Corrected package path: sarchaeology.data
from sarchaeology.data.dataloaders import (
    create_sar_dataloaders,
)
from sarchaeology.models.sar_cae import (
    build_sar_cae,
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
from sarchaeology.training.losses import (
    build_sar_loss,
    reconstruction_error_map,
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

    stats_path = (
        ROOT /
        config["normalization"]["output_path"]
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
            "Training manifest fingerprint mismatch"
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

    model = build_sar_cae(config).to(
        device
    )

    criterion = build_sar_loss(config).to(
        device
    )

    # This is a single forward/backward test,
    # NOT a training epoch.
    model.train()
    model.zero_grad(set_to_none=True)

    with torch.autocast(
        device_type=device.type,
        dtype=amp["dtype"],
        enabled=amp["enabled"],
    ):
        outputs = model(
            images,
            valid_mask=masks,
        )

    # float32 loss is calculated outside autocast.
    reconstruction = outputs["reconstruction"]

    loss = criterion(
        reconstruction,
        images,
        masks,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() >= 0

    # Backpropagation verifies that the
    # reconstruction objective reaches the
    # model's parameters.
    loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]

    assert gradients
    assert all(
        torch.isfinite(gradient).all()
        for gradient in gradients
    )

    assert any(
        torch.count_nonzero(gradient).item() > 0
        for gradient in gradients
    )

    # Anomaly-oriented residual map for inspection.
    with torch.no_grad():
        error_map = reconstruction_error_map(
            reconstruction.detach(),
            images,
            masks,
            method="l1",
        )

    valid_errors = error_map[
        masks.bool()
    ]

    print("\nSARchaeology AI — Phase 2.5")
    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print("AMP enabled:", amp["enabled"])
    print("Loss function:", criterion.loss_type)

    print("\nREAL SAR BATCH")
    print("Input:", tuple(images.shape))
    print(
        "Reconstruction:",
        tuple(reconstruction.shape),
    )
    print("Mask:", tuple(masks.shape))

    print("\nRECONSTRUCTION LOSS")
    print("Scalar loss:", loss.item())
    print("Finite gradients:", True)

    print("\nERROR MAP")
    print("Shape:", tuple(error_map.shape))
    print(
        "Mean valid L1 residual:",
        valid_errors.mean().item(),
    )
    print(
        "Maximum valid L1 residual:",
        valid_errors.max().item(),
    )

    print(
        "\nPhase 2.5 reconstruction loss "
        "check passed."
    )


if __name__ == "__main__":
    main()
