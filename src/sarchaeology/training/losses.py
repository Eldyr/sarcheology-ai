
"""Phase 2.5 — Mask-aware SAR reconstruction losses."""

import torch
from torch import nn
from torch.nn import functional as F


class MaskedSARReconstructionLoss(nn.Module):
    """
    Masked reconstruction loss for Sentinel-1 VV/VH.

    Inputs:
        prediction: [B, 2, H, W]
        target:     [B, 2, H, W]
        valid_mask: [B, 1, H, W]

    Supports:
        mse, l1, huber

    Reduction:
        mean: one scalar across all valid channel-pixels
        none: per-sample loss [B]

    Channel weights are applied per polarization.

    Loss arithmetic is performed in float32,
    including when model outputs use mixed precision.
    """

    def __init__(
        self,
        loss_type="huber",
        huber_delta=1.0,
        reduction="mean",
        channel_weights=(1.0, 1.0),
    ):
        super().__init__()

        if loss_type not in ("mse", "l1", "huber"):
            raise ValueError(
                "loss_type must be mse, l1, or huber"
            )

        if huber_delta <= 0:
            raise ValueError(
                "huber_delta must be positive"
            )

        if reduction not in ("mean", "none"):
            raise ValueError(
                "reduction must be mean or none"
            )

        if len(channel_weights) != 2:
            raise ValueError(
                "Expected two channel weights: VV, VH"
            )

        weights = torch.as_tensor(
            channel_weights,
            dtype=torch.float32,
        )

        if (
            not torch.isfinite(weights).all()
            or (weights < 0).any()
            or weights.sum() <= 0
        ):
            raise ValueError(
                "Channel weights must be finite, "
                "nonnegative, and not both zero"
            )

        self.loss_type = loss_type
        self.huber_delta = float(huber_delta)
        self.reduction = reduction

        self.register_buffer(
            "channel_weights",
            weights.view(1, 2, 1, 1),
        )

    def _validate(self, prediction, target, valid_mask):
        if prediction.ndim != 4:
            raise ValueError(
                "Expected prediction [B,2,H,W]"
            )

        if prediction.shape[1] != 2:
            raise ValueError(
                "Expected two SAR channels: VV and VH"
            )

        if prediction.shape != target.shape:
            raise ValueError(
                "Prediction and target shapes differ"
            )

        expected = (
            prediction.shape[0],
            1,
            prediction.shape[2],
            prediction.shape[3],
        )

        if tuple(valid_mask.shape) != expected:
            raise ValueError(
                f"Expected valid_mask shape {expected}"
            )

        if valid_mask.device != prediction.device:
            raise ValueError(
                "Mask and prediction must "
                "be on the same device"
            )

        if target.device != prediction.device:
            raise ValueError(
                "Target and prediction must "
                "be on the same device"
            )

        if valid_mask.dtype != torch.bool:
            if not torch.isfinite(valid_mask).all():
                raise ValueError(
                    "Mask contains non-finite values"
                )

            if not torch.all(
                (valid_mask == 0) |
                (valid_mask == 1)
            ):
                raise ValueError(
                    "Mask must contain only 0 or 1"
                )

    def _pixel_errors(self, prediction, target):
        if self.loss_type == "mse":
            return (prediction - target).square()

        if self.loss_type == "l1":
            return (prediction - target).abs()

        return F.huber_loss(
            prediction,
            target,
            delta=self.huber_delta,
            reduction="none",
        )

    def forward(self, prediction, target, valid_mask):
        self._validate(
            prediction, target, valid_mask
        )

        valid = valid_mask.bool()
        if not valid.flatten(1).any(dim=1).all():
            raise ValueError(
                "Every sample must contain valid SAR pixels"
            )
        expanded = valid.expand_as(prediction)

        if not valid.any():
            raise ValueError(
                "No valid SAR pixels in batch"
            )

        # Ignore invalid locations BEFORE computing
        # differences, so NaN nodata values cannot
        # contaminate valid loss calculations.
        prediction = torch.where(
            expanded,
            prediction.float(),
            torch.zeros_like(prediction, dtype=torch.float32),
        )

        target = torch.where(
            expanded,
            target.float(),
            torch.zeros_like(target, dtype=torch.float32),
        )

        if (
            not torch.isfinite(prediction).all()
            or not torch.isfinite(target).all()
        ):
            raise ValueError(
                "Non-finite SAR values at valid pixels"
            )

        errors = self._pixel_errors(
            prediction, target
        )

        weights = self.channel_weights.to(
            device=prediction.device
        )

        weighted = errors * weights * expanded

        # Denominator accounts for both:
        # - number of valid pixels
        # - total polarization weight
        weighted_denominator = (
            valid.sum(dim=(1, 2, 3)).float()
            * weights.sum()
        )

        per_sample = (
            weighted.sum(dim=(1, 2, 3))
            / weighted_denominator.clamp_min(1.0e-12)
        )

        if self.reduction == "none":
            # Reject a sample with no valid pixels:
            # its loss is not mathematically defined.
            if (weighted_denominator == 0).any():
                raise ValueError(
                    "A sample has no valid SAR pixels"
                )

            return per_sample

        # Global mean weights each valid pixel
        # equally, rather than each image equally.
        denominator = weighted_denominator.sum()

        return weighted.sum() / denominator


def build_sar_loss(config):
    """Instantiate reconstruction loss from YAML."""
    cfg = config["loss"]

    return MaskedSARReconstructionLoss(
        loss_type=cfg["type"],
        huber_delta=cfg.get("huber_delta", 1.0),
        reduction=cfg.get("reduction", "mean"),
        channel_weights=cfg.get(
            "channel_weights",
            [1.0, 1.0],
        ),
    )


@torch.no_grad()
def reconstruction_error_map(
    prediction,
    target,
    valid_mask,
    method="l1",
):
    """
    Compute a per-pixel reconstruction-error map.

    Returns:
        [B,1,H,W]

    The map averages VV and VH errors.
    Invalid pixels are NaN, not zero.

    Intended for later anomaly visualization,
    not as a training loss.
    """
    if prediction.ndim != 4 or prediction.shape[1] != 2:
        raise ValueError(
            "Expected SAR prediction [B,2,H,W]"
        )

    if prediction.shape != target.shape:
        raise ValueError(
            "Prediction and target shapes differ"
        )

    if valid_mask.shape != (
        prediction.shape[0],
        1,
        prediction.shape[2],
        prediction.shape[3],
    ):
        raise ValueError(
            "Invalid mask shape"
        )

    if method not in ("l1", "mse"):
        raise ValueError(
            "Error map method must be l1 or mse"
        )

    valid = valid_mask.bool()

    if not valid.any():
        raise ValueError("No valid pixels")

    expanded = valid.expand_as(prediction)

    pred = torch.where(
        expanded,
        prediction.float(),
        torch.zeros_like(prediction, dtype=torch.float32),
    )

    truth = torch.where(
        expanded,
        target.float(),
        torch.zeros_like(target, dtype=torch.float32),
    )

    if (
        not torch.isfinite(pred).all()
        or not torch.isfinite(truth).all()
    ):
        raise ValueError(
            "Non-finite values at valid pixels"
        )

    difference = pred - truth

    if method == "l1":
        error = difference.abs()
    else:
        error = difference.square()

    error = error.mean(dim=1, keepdim=True)

    return error.masked_fill(
        ~valid,
        float("nan"),
    )
