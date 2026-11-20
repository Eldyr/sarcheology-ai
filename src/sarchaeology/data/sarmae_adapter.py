
"""
SARchaeology AI — Phase 2.11
Sentinel-1 SAR to pretrained SARMAE adapter.

Supports:
  - [B, 2, H, W] VV/VH tensors
  - [B, 1, H, W] paired validity masks
  - Independent percentile stretch per polarization
  - Grayscale -> RGB replication
  - 224x224 bilinear image resize
  - Nearest-neighbor validity-mask resize
  - ImageNet normalization
  - Invalid-pixel masking after normalization

Modes:
  raw_db: preferred; input is original finite dB
  normalized_proxy: existing CAE-normalized inputs

Output:
  pixel_values: [2B, 3, 224, 224]
  pixel_mask: [2B, 1, 224, 224]
  polarization_index: [2B], 0=VV, 1=VH

Ordering: VV batch followed by VH batch.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SARMAEAdapterConfig:
    image_size: int = 224
    lower_percentile: float = 1.0
    upper_percentile: float = 99.0
    input_mode: str = "raw_db"
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)


class SARMAEInputAdapter:
    def __init__(self, config=None):
        self.config = config or SARMAEAdapterConfig()

        if self.config.input_mode not in (
            "raw_db", "normalized_proxy"
        ):
            raise ValueError("Unsupported input mode")

        if not (
            0 <= self.config.lower_percentile
            < self.config.upper_percentile <= 100
        ):
            raise ValueError("Invalid percentiles")

        if self.config.image_size <= 0:
            raise ValueError("Invalid image size")

        if len(self.config.mean) != 3:
            raise ValueError("Expected three mean values")

        if len(self.config.std) != 3:
            raise ValueError("Expected three std values")

        if any(s <= 0 for s in self.config.std):
            raise ValueError("All std values must be positive")

    def __call__(self, images, valid_mask):
        if images.ndim != 4 or images.shape[1] != 2:
            raise ValueError(
                "Expected VV/VH [B,2,H,W] input"
            )

        b, _, h, w = images.shape

        if valid_mask.shape != (b, 1, h, w):
            raise ValueError(
                "Expected mask [B,1,H,W]"
            )

        if images.device != valid_mask.device:
            raise ValueError(
                "Images and masks must share device"
            )

        x = images.float()
        mask = valid_mask.bool()

        if not torch.isfinite(x[
            mask.expand_as(x)
        ]).all():
            raise ValueError(
                "Non-finite valid SAR input pixels"
            )

        if not mask.flatten(1).any(dim=1).all():
            raise ValueError(
                "Each observation needs valid pixels"
            )

        # Polarization-major order:
        # images 0..B-1 are VV; B..2B-1 are VH.
        gray = torch.cat(
            [x[:, 0:1], x[:, 1:2]],
            dim=0,
        )
        gray_mask = torch.cat(
            [mask, mask],
            dim=0,
        )

        stretched = torch.zeros_like(gray)

        # Quantiles must use valid pixels only.
        # Computation on CPU avoids unnecessary
        # synchronization for large GPU pipelines.
        for index in range(2 * b):
            values = gray[index, 0]
            valid = gray_mask[index, 0]

            selected = values[valid]

            low = torch.quantile(
                selected,
                self.config.lower_percentile / 100.0,
            )
            high = torch.quantile(
                selected,
                self.config.upper_percentile / 100.0,
            )

            if float(high - low) <= 1e-8:
                scaled = torch.zeros_like(values)
            else:
                scaled = (
                    (values - low) / (high - low)
                ).clamp(0.0, 1.0)

            # Explicit 8-bit quantization before
            # rescaling, matching grayscale PNG style.
            scaled = (
                torch.round(scaled * 255.0) / 255.0
            )

            stretched[index, 0] = torch.where(
                valid,
                scaled,
                0.0,
            )

        size = (
            self.config.image_size,
            self.config.image_size,
        )

        resized = F.interpolate(
            stretched,
            size=size,
            mode="bilinear",
            align_corners=False,
        )

        resized_mask = F.interpolate(
            gray_mask.float(),
            size=size,
            mode="nearest",
        ).bool()

        # Pixels close to invalid source regions may
        # be influenced by bilinear interpolation.
        # Mask-aware interpolation prevents this.
        numerator = F.interpolate(
            stretched * gray_mask.float(),
            size=size,
            mode="bilinear",
            align_corners=False,
        )

        denominator = F.interpolate(
            gray_mask.float(),
            size=size,
            mode="bilinear",
            align_corners=False,
        )

        resized_mask = denominator > 0.999

        resized = numerator / denominator.clamp_min(
            1e-6
        )

        resized = torch.where(
            resized_mask,
            resized,
            torch.zeros_like(resized),
        )

        rgb = resized.repeat(1, 3, 1, 1)

        mean = torch.tensor(
            self.config.mean,
            dtype=rgb.dtype,
            device=rgb.device,
        ).view(1, 3, 1, 1)

        std = torch.tensor(
            self.config.std,
            dtype=rgb.dtype,
            device=rgb.device,
        ).view(1, 3, 1, 1)

        pixel_values = (rgb - mean) / std

        # Use neutral zero in normalized space at
        # invalid locations, not ImageNet-normalized
        # black (-mean/std).
        pixel_values = torch.where(
            resized_mask.expand_as(pixel_values),
            pixel_values,
            torch.zeros_like(pixel_values),
        )

        polarization_index = torch.cat([
            torch.zeros(
                b, dtype=torch.long, device=x.device
            ),
            torch.ones(
                b, dtype=torch.long, device=x.device
            ),
        ])

        return {
            "pixel_values": pixel_values,
            "pixel_mask": resized_mask,
            "polarization_index": polarization_index,
            "source_batch_size": b,
            "input_mode": self.config.input_mode,
        }
