
"""SARchaeology AI — Phase 3.6: SpectralMAE-style baseline.

Masks whole Sentinel-2 spectral channels, then reconstructs them.
Not an exact reproduction of the published hyperspectral SpectralMAE.

Run:
    uv run python scripts/phase3_optical/train_spectral_mae.py --smoke
    uv run python scripts/phase3_optical/train_spectral_mae.py
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sarchaeology.data.optical_multiyear import (
    MultiYearOpticalDataset,
)

BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")
YEARS = (2022, 2023, 2024, 2025)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SpectralMAE(nn.Module):
    """Per-band patches + spectral attention + spatial reconstruction."""

    def __init__(
        self,
        bands=6,
        image_size=128,
        patch_size=8,
        dim=96,
        depth=3,
        heads=4,
    ):
        super().__init__()
        assert dim % heads == 0
        assert image_size % patch_size == 0

        self.bands = bands
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid = image_size // patch_size

        # Each spectral band is embedded independently.
        self.patch_embed = nn.Conv2d(
            1, dim, kernel_size=patch_size, stride=patch_size
        )

        self.band_embed = nn.Parameter(
            torch.randn(1, 1, bands, dim) * 0.02
        )
        self.mask_token = nn.Parameter(
            torch.zeros(1, 1, 1, dim)
        )

        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.spectral_encoder = nn.TransformerEncoder(
            layer, num_layers=depth,
            enable_nested_tensor=False,
        )

        self.spatial = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GELU(),
        )
        self.decoder = nn.Conv2d(
            dim, bands * patch_size * patch_size, 1
        )

    def forward(self, image, band_mask):
        b, c, h, w = image.shape
        assert (c, h, w) == (
            self.bands, self.image_size, self.image_size
        )
        assert band_mask.shape == (b, c)

        p = self.patch_size
        g = self.grid

        # Band-independent extraction prevents masked-band leakage.
        x = image.reshape(b * c, 1, h, w)
        x = self.patch_embed(x)
        x = x.reshape(b, c, -1, g * g)
        x = x.permute(0, 3, 1, 2)

        # Tokens: [batch, spatial_patch, spectral_band, dim].
        mask = band_mask[:, None, :, None]
        x = torch.where(mask, self.mask_token, x)
        x = x + self.band_embed

        x = x.reshape(b * g * g, c, -1)
        x = self.spectral_encoder(x)
        x = x.reshape(b, g * g, c, -1)

        # Visible and reconstructed band features contribute.
        x = x.mean(dim=2)
        embedding = x.mean(dim=1)

        x = x.transpose(1, 2).reshape(b, -1, g, g)
        x = x + self.spatial(x)

        # Convert spatial patch predictions to full resolution.
        y = self.decoder(x)
        y = y.reshape(b, c, p, p, g, g)
        y = y.permute(0, 1, 4, 2, 5, 3)
        reconstruction = y.reshape(b, c, h, w)

        return {
            "reconstruction": reconstruction,
            "embedding": embedding,
            "band_mask": band_mask,
        }


def make_band_mask(batch_size, generator, masked_bands=2):
    if not 1 <= masked_bands < len(BANDS):
        raise ValueError("Invalid number of masked bands")
    scores = torch.rand(
        batch_size, len(BANDS), generator=generator
    )
    order = scores.argsort(dim=1)
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(1, order[:, :masked_bands], True)
    return mask


def masked_loss(model, image, valid, mask):
    predicted = model(image, mask)["reconstruction"].float()
    scored = (
        mask[:, :, None, None]
        & valid.expand_as(image).bool()
    )
    difference = torch.where(
        scored, predicted - image.float(),
        torch.zeros_like(predicted),
    )
    count = scored.sum().clamp_min(1)
    loss = F.huber_loss(
        difference,
        torch.zeros_like(difference),
        delta=1.0,
        reduction="none",
    ).sum() / count

    return loss, (
        difference.abs().sum().detach(),
        difference.square().sum().detach(),
        count.detach(),
    )


def run_epoch(
    model, loader, optimizer, device, epoch, seed,
    masked_bands, smoke=False,
):
    training = optimizer is not None
    model.train(training)
    sums = torch.zeros(3, device=device)

    # Reproducible validation masks across epochs.
    mask_generator = torch.Generator(device="cpu")
    mask_generator.manual_seed(
        seed + epoch if training else seed + 1000000
    )

    for index, batch in enumerate(loader):
        if smoke and index > 0:
            break

        image = batch["image"].to(device)
        valid = batch["valid_mask"].to(device).bool()

        # Invalid pixels cannot provide useful spectral inputs.
        image = torch.where(
            valid.expand_as(image),
            image,
            torch.zeros_like(image),
        )
        mask = make_band_mask(
            image.shape[0], mask_generator, masked_bands
        ).to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                loss, metrics = masked_loss(
                    model, image, valid, mask
                )
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0
                )
                optimizer.step()

        sums += torch.stack(metrics)

    abs_sum, square_sum, count = sums.tolist()
    if count == 0:
        raise RuntimeError("No valid masked pixels")

    return {
        "masked_mae_normalized": abs_sum / count,
        "masked_rmse_normalized": (
            square_sum / count
        ) ** 0.5,
        "masked_huber_normalized": None,
        "scored_channel_pixels": int(count),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--masked-bands", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.batch_size < 1 or args.epochs < 1:
        parser.error("Batch size and epochs must be positive")

    seed_everything(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    stats_path = (
        ROOT / "outputs/statistics/optical_buto_multiyear_v1.json"
    )
    stats = json.loads(stats_path.read_text(encoding="utf-8"))

    if tuple(stats["bands"]) != BANDS:
        raise RuntimeError("Sentinel-2 band-order mismatch")
    if stats["fit_split"] != "train":
        raise RuntimeError("Statistics must be train-only")

    options = dict(
        years=YEARS,
        statistics=stats,
        min_valid_fraction=float(
            stats.get("min_valid_fraction", 0.5)
        ),
        min_source_observations=int(
            stats.get("min_source_observations", 1)
        ),
    )

    train = MultiYearOpticalDataset(
        ROOT, split="train", **options
    )
    val = MultiYearOpticalDataset(
        ROOT, split="val", **options
    )

    train_ids = {r["patch_id"] for r in train.samples}
    val_ids = {r["patch_id"] for r in val.samples}

    if train_ids & val_ids:
        raise RuntimeError("Geographic split leakage")
    if {r["year"] for r in val.samples} != {2024}:
        raise RuntimeError("Unexpected validation years")

    train_loader = DataLoader(
        train, batch_size=args.batch_size,
        shuffle=True, num_workers=2,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val, batch_size=args.batch_size,
        shuffle=False, num_workers=2,
        pin_memory=device.type == "cuda",
    )

    model = SpectralMAE().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.05,
    )

    stamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    run_id = f"spectral_mae_{stamp}"
    checkpoint_dir = (
        ROOT / "checkpoints/optical_spectral_mae" / run_id
    )
    output_dir = (
        ROOT / "outputs/training/optical_spectral_mae" / run_id
    )

    if not args.smoke:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

    best = float("inf")
    no_improvement = 0

    print("Phase 3.6 — SpectralMAE")
    print("Device:", device)
    print("Train:", len(train), "Validation:", len(val))
    print("Parameters:", sum(
        p.numel() for p in model.parameters()
    ))

    for epoch in range(1, 2 if args.smoke else args.epochs + 1):
        tr = run_epoch(
            model, train_loader, optimizer, device,
            epoch, args.seed, args.masked_bands,
            smoke=args.smoke,
        )
        va = run_epoch(
            model, val_loader, None, device,
            epoch, args.seed, args.masked_bands,
            smoke=args.smoke,
        )

        score = va["masked_mae_normalized"]
        improved = score < best

        if improved:
            best = score
            no_improvement = 0
        else:
            no_improvement += 1

        print(
            f"Epoch {epoch}: "
            f"train MAE={tr['masked_mae_normalized']:.6f}, "
            f"val MAE={score:.6f}, "
            f"best={best:.6f}",
            flush=True,
        )

        if args.smoke:
            print("SMOKE TEST COMPLETED")
            return

        record = {
            "epoch": epoch,
            "train": tr,
            "val": va,
            "best_val_mae": best,
        }
        with (output_dir / "history.jsonl").open(
            "a", encoding="utf-8"
        ) as f:
            f.write(json.dumps(record) + "\n")

        if improved:
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "run_id": run_id,
                    "model_config": {
                        "bands": 6,
                        "image_size": 128,
                        "patch_size": 8,
                        "dim": 96,
                        "depth": 3,
                        "heads": 4,
                    },
                    "masked_bands": args.masked_bands,
                    "best_val_mae": best,
                    "statistics_path": str(stats_path),
                },
                checkpoint_dir / "best.pt",
            )

        if no_improvement >= args.patience:
            print("Early stopping")
            break

    print("Run:", run_id)
    print("Best validation normalized MAE:", best)
    print("Checkpoint:", checkpoint_dir / "best.pt")


if __name__ == "__main__":
    main()
