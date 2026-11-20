
"""
SARchaeology AI — Phase 2.14
Swin-Tiny SAR masked reconstruction autoencoder.

Input:
    image:      [B, 2, 128, 128]
    valid_mask: [B, 1, 128, 128]

Output:
    reconstruction: [B, 2, 128, 128]
    masked_pixels:  [B, 1, 128, 128]
    embedding:      [B, 768]

Uses a Swin-Tiny backbone initialized from scratch.
This is masked-image reconstruction, not a
visible-token-only MAE.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
import timm


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class SARDecoder(nn.Module):
    """Decoder from 4x4 latent features to 128x128."""

    def __init__(self):
        super().__init__()

        self.blocks = nn.ModuleList([
            ConvBlock(768, 384),
            ConvBlock(384, 192),
            ConvBlock(192, 96),
            ConvBlock(96, 64),
            ConvBlock(64, 32),
        ])

        self.output = nn.Conv2d(
            32, 2, kernel_size=3, padding=1
        )

    def forward(self, x):
        for block in self.blocks:
            x = F.interpolate(
                x,
                scale_factor=2,
                mode="bilinear",
                align_corners=False,
            )
            x = block(x)

        return self.output(x)


class SARSwinMAE(nn.Module):
    """
    Swin-Tiny-based masked SAR autoencoder.

    mask_ratio:
        Fraction of spatial masking blocks.

    mask_block_size:
        Size of one square masking block in pixels.

    With mask_ratio=0, the model operates without
    synthetic masking, useful for reconstruction checks.

    With mask_ratio>0, synthetic masked regions
    are reconstructed from the visible context.
    """

    def __init__(
        self,
        input_size=128,
        in_channels=2,
        mask_ratio=0.4,
        mask_block_size=16,
    ):
        super().__init__()

        if input_size != 128:
            raise ValueError(
                "This decoder currently requires "
                "128x128 input images"
            )

        if in_channels != 2:
            raise ValueError("Expected VV/VH channels")

        if not 0 <= mask_ratio < 1:
            raise ValueError(
                "mask_ratio must be in [0, 1)"
            )

        if (
            mask_block_size <= 0
            or input_size % mask_block_size != 0
        ):
            raise ValueError(
                "mask block size must divide input size"
            )

        self.input_size = input_size
        self.mask_ratio = float(mask_ratio)
        self.mask_block_size = mask_block_size

        # Swin-Tiny:
        # depths = (2,2,6,2)
        # heads = (3,6,12,24)
        #
        # Use the untrained architecture with two
        # SAR input channels and 128x128 resolution.
        self.encoder = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=False,
            in_chans=2,
            img_size=input_size,
            features_only=True,
            out_indices=(3,),
        )

        channels = self.encoder.feature_info.channels()

        if channels[-1] != 768:
            raise RuntimeError(
                f"Unexpected Swin channels: {channels}"
            )

        self.decoder = SARDecoder()

    def make_block_mask(self, image):
        """
        Create per-image random 16x16 block masks.

        False = visible
        True = synthetically masked
        """
        batch, _, height, width = image.shape

        if height != self.input_size or width != self.input_size:
            raise ValueError(
                "Unexpected SAR image resolution"
            )

        cells = self.input_size // self.mask_block_size
        total = cells * cells
        number_masked = round(
            total * self.mask_ratio
        )

        scores = torch.rand(
            batch,
            total,
            device=image.device,
        )

        order = scores.argsort(dim=1)

        flat_mask = torch.zeros(
            batch,
            total,
            dtype=torch.bool,
            device=image.device,
        )

        if number_masked:
            flat_mask.scatter_(
                1,
                order[:, :number_masked],
                True,
            )

        mask = flat_mask.reshape(
            batch, 1, cells, cells
        )

        mask = mask.repeat_interleave(
            self.mask_block_size, dim=2
        ).repeat_interleave(
            self.mask_block_size, dim=3
        )

        return mask

    def forward(
        self,
        image,
        valid_mask=None,
        *,
        apply_mask=True,
        synthetic_mask=None,
    ):
        if image.ndim != 4 or image.shape[1:] != (
            2, self.input_size, self.input_size
        ):
            raise ValueError(
                "Expected [B,2,128,128] input"
            )

        if valid_mask is None:
            valid = torch.ones_like(
                image[:, :1], dtype=torch.bool
            )
        else:
            if valid_mask.shape != (
                image.shape[0], 1,
                self.input_size, self.input_size
            ):
                raise ValueError(
                    "Invalid validity-mask shape"
                )
            valid = valid_mask.bool()

        if not valid.flatten(1).any(dim=1).all():
            raise ValueError(
                "Every image needs valid pixels"
            )

        expanded_valid = valid.expand_as(image)

        if not torch.isfinite(
            image[expanded_valid]
        ).all():
            raise ValueError(
                "Non-finite valid SAR values"
            )

        # Remove invalid values, including NaNs.
        clean_image = torch.where(
            expanded_valid,
            image,
            torch.zeros_like(image),
        )

        if synthetic_mask is not None:
            if synthetic_mask.shape != valid.shape:
                raise ValueError(
                    "Invalid synthetic mask shape"
                )
            mask = synthetic_mask.bool()

        elif apply_mask:
            mask = self.make_block_mask(image)

        else:
            mask = torch.zeros_like(valid)

        # Only valid pixels can be considered masked.
        masked_pixels = mask & valid

        # No shortcut from original SAR to decoder.
        input_visible = valid & ~mask

        encoder_input = torch.where(
            input_visible.expand_as(clean_image),
            clean_image,
            torch.zeros_like(clean_image),
        )

        features = self.encoder(encoder_input)
        latent = features[-1]

        # timm Swin feature maps may be channels-last.
        if (
            latent.ndim == 4
            and latent.shape[-1] == 768
            and latent.shape[1] != 768
        ):
            latent = latent.permute(
                0, 3, 1, 2
            ).contiguous()

        if latent.shape[1:] != (768, 4, 4):
            raise RuntimeError(
                "Unexpected Swin feature shape: "
                f"{tuple(latent.shape)}"
            )

        reconstruction = self.decoder(latent)

        if reconstruction.shape != image.shape:
            raise RuntimeError(
                "Reconstruction shape mismatch"
            )

        embedding = latent.mean(dim=(2, 3))

        return {
            "reconstruction": reconstruction,
            "masked_pixels": masked_pixels,
            "embedding": embedding,
            "latent": latent,
        }


def build_sar_swin_mae(config=None):
    """Build Swin-Tiny from a model configuration."""
    config = config or {}

    if "model" in config:
        config = config["model"]

    return SARSwinMAE(
        input_size=int(config.get("input_size", 128)),
        in_channels=int(config.get("in_channels", 2)),
        mask_ratio=float(config.get("mask_ratio", 0.4)),
        mask_block_size=int(
            config.get("mask_block_size", 16)
        ),
    )
