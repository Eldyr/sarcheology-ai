
"""
Phase 2.14 — Swin-Tiny GPU forward/backward smoke test.

This checks architecture, gradients, reconstruction
shape, and masked Huber loss without loading data.
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sarchaeology.models.sar_swin_mae import SARSwinMAE


def main():
    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = SARSwinMAE(
        input_size=128,
        in_channels=2,
        mask_ratio=0.4,
        mask_block_size=16,
    ).to(device)

    model.train()

    images = torch.randn(
        2, 2, 128, 128, device=device
    )
    valid_mask = torch.ones(
        2, 1, 128, 128,
        dtype=torch.bool,
        device=device,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.05,
    )

    optimizer.zero_grad(set_to_none=True)

    amp_enabled = device.type == "cuda"

    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16 if amp_enabled else torch.bfloat16,
        enabled=amp_enabled,
    ):
        output = model(
            images,
            valid_mask=valid_mask,
        )

    # Compute loss in float32 for stability.
    pred = output["reconstruction"].float()
    target = images.float()

    region = output["masked_pixels"].expand_as(pred)

    per_pixel_loss = F.huber_loss(
        pred,
        target,
        reduction="none",
        delta=1.0,
    )

    valid_count = region.sum().clamp_min(1)

    loss = (
        (per_pixel_loss * region).sum()
        / valid_count
    )

    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite masked Huber loss")

    loss.backward()

    gradients = [
        p.grad
        for p in model.parameters()
        if p.requires_grad and p.grad is not None
    ]

    if not gradients:
        raise RuntimeError("No gradients produced")

    if not all(torch.isfinite(g).all() for g in gradients):
        raise RuntimeError("Non-finite gradients")

    optimizer.step()

    parameter_count = sum(
        p.numel() for p in model.parameters()
    )

    print("\nSARchaeology AI — Phase 2.14")
    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        print(
            "Peak allocated memory (GB):",
            round(
                torch.cuda.max_memory_allocated()
                / 1024**3,
                3,
            ),
        )

    print("Parameters:", f"{parameter_count:,}")
    print("Input:", tuple(images.shape))
    print(
        "Reconstruction:",
        tuple(pred.shape),
    )
    print(
        "Latent:",
        tuple(output["latent"].shape),
    )
    print(
        "Embedding:",
        tuple(output["embedding"].shape),
    )
    print(
        "Actual masked fraction:",
        round(
            output["masked_pixels"]
            .float().mean().item(),
            4,
        ),
    )
    print("Masked Huber loss:", round(loss.item(), 6))
    print("Finite gradients: PASS")
    print("Optimizer step: PASS")
    print("\nPhase 2.14 smoke test passed.")


if __name__ == "__main__":
    main()
