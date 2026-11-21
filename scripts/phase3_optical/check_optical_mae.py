
"""Phase 3.2 optical ViT-Tiny GPU smoke test."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sarchaeology.models.optical_mae import (
    OpticalViTMAE,
)


def main():
    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = OpticalViTMAE().to(device)
    model.train()

    batch_size = 2

    images = torch.randn(
        batch_size, 6, 128, 128,
        device=device,
    )

    valid = torch.ones(
        batch_size, 1, 128, 128,
        dtype=torch.bool,
        device=device,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
    )

    optimizer.zero_grad(set_to_none=True)

    use_amp = device.type == "cuda"

    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=use_amp,
    ):
        output = model(
            images,
            valid,
        )

    # Calculate loss in FP32.
    loss = model.masked_loss(
        images, valid, output
    )

    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite loss")

    loss.backward()

    gradients = [
        p.grad for p in model.parameters()
        if p.grad is not None
    ]

    if not gradients or not all(
        torch.isfinite(g).all() for g in gradients
    ):
        raise RuntimeError("Invalid gradients")

    optimizer.step()

    params = sum(
        p.numel() for p in model.parameters()
    )

    print("\nSARchaeology AI — Phase 3.2")
    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )
        print(
            "Peak VRAM (GB):",
            round(
                torch.cuda.max_memory_allocated()
                / 1024**3, 3
            ),
        )

    print("Parameters:", f"{params:,}")
    print("Input:", tuple(images.shape))
    print(
        "Reconstruction:",
        tuple(output["reconstruction"].shape),
    )
    print(
        "Encoder embedding:",
        tuple(output["embedding"].shape),
    )
    print(
        "Masked token fraction:",
        output["patch_mask"].float().mean().item(),
    )
    print("Masked Huber:", loss.item())
    print("Finite gradients: PASS")
    print("Optimizer step: PASS")


if __name__ == "__main__":
    main()
