
"""
SARchaeology AI — Phase 2.8
Reusable SAR reconstruction evaluation metrics.

All metrics are computed using only valid VV/VH pixels.

Inputs:
    prediction: [B, 2, H, W]
    target:     [B, 2, H, W]
    valid_mask: [B, 1, H, W]

The metric functions return additive sums so we can
aggregate results across batches, observations, patches,
and acquisition dates without incorrectly averaging RMSEs.
"""

import numpy as np
import pandas as pd
import torch


def _validate_inputs(prediction, target, valid_mask):
    """Validate SAR reconstruction and mask shapes."""

    if prediction.ndim != 4 or prediction.shape[1] != 2:
        raise ValueError(
            "Expected prediction shape [B, 2, H, W]"
        )

    if prediction.shape != target.shape:
        raise ValueError(
            "Prediction and target shapes differ"
        )

    expected_mask_shape = (
        prediction.shape[0],
        1,
        prediction.shape[2],
        prediction.shape[3],
    )

    if tuple(valid_mask.shape) != expected_mask_shape:
        raise ValueError(
            f"Expected mask shape {expected_mask_shape}"
        )

    if (
        prediction.device != target.device
        or prediction.device != valid_mask.device
    ):
        raise ValueError(
            "Prediction, target, and mask must "
            "be on the same device"
        )

    if valid_mask.dtype != torch.bool:
        if not torch.isfinite(valid_mask).all():
            raise ValueError(
                "Mask contains non-finite values"
            )

        if not torch.all(
            (valid_mask == 0) | (valid_mask == 1)
        ):
            raise ValueError(
                "Mask must contain only 0 and 1"
            )

    valid = valid_mask.bool()

    if not valid.flatten(1).any(dim=1).all():
        raise ValueError(
            "Every observation must contain valid SAR pixels"
        )

    return valid


def masked_image_metrics(
    prediction,
    target,
    valid_mask,
):
    """
    Return per-observation, per-channel additive metrics.

    Returns:
        counts:   [B, 2] valid pixel counts
        absolute: [B, 2] absolute-error sums
        squared:  [B, 2] squared-error sums

    Invalid pixels do not contribute, even if their
    original values contain NaNs.
    """

    valid = _validate_inputs(
        prediction, target, valid_mask
    )

    expanded = valid.expand_as(prediction)

    prediction = prediction.float()
    target = target.float()

    if not torch.isfinite(prediction[expanded]).all():
        raise ValueError(
            "Non-finite valid predictions"
        )

    if not torch.isfinite(target[expanded]).all():
        raise ValueError(
            "Non-finite valid targets"
        )

    # Mask before subtraction: this prevents NaN
    # values at invalid pixels from contaminating metrics.
    safe_prediction = torch.where(
        expanded,
        prediction,
        torch.zeros_like(prediction),
    )

    safe_target = torch.where(
        expanded,
        target,
        torch.zeros_like(target),
    )

    difference = safe_prediction - safe_target

    absolute = difference.abs()
    squared = difference.square()

    counts = expanded.sum(
        dim=(2, 3)
    ).double()

    absolute_sums = absolute.sum(
        dim=(2, 3)
    ).double()

    squared_sums = squared.sum(
        dim=(2, 3)
    ).double()

    return counts, absolute_sums, squared_sums


def huber_sums(
    prediction,
    target,
    valid_mask,
    delta=1.0,
):
    """
    Return masked Huber-error sums per observation.

    Returns:
        Tensor [B]

    Uses the standard Huber definition:
        0.5 * error^2                 if |error| <= delta
        delta * (|error| - 0.5*delta) otherwise
    """

    if delta <= 0:
        raise ValueError(
            "Huber delta must be positive"
        )

    valid = _validate_inputs(
        prediction, target, valid_mask
    )

    expanded = valid.expand_as(prediction)

    prediction = prediction.float()
    target = target.float()

    if not torch.isfinite(prediction[expanded]).all():
        raise ValueError(
            "Non-finite valid predictions"
        )

    if not torch.isfinite(target[expanded]).all():
        raise ValueError(
            "Non-finite valid targets"
        )

    safe_prediction = torch.where(
        expanded,
        prediction,
        torch.zeros_like(prediction),
    )

    safe_target = torch.where(
        expanded,
        target,
        torch.zeros_like(target),
    )

    difference = (
        safe_prediction - safe_target
    ).abs()

    errors = torch.where(
        difference <= delta,
        0.5 * difference.square(),
        delta * (
            difference - 0.5 * delta
        ),
    )

    return errors.sum(
        dim=(1, 2, 3)
    ).double()


def summarize(frame: pd.DataFrame):
    """
    Aggregate evaluation rows using additive error sums.

    Returns a dictionary containing:
        Overall normalized MAE, RMSE, Huber
        VV/VH MAE and RMSE
        Constant-median baseline MAE and RMSE

    RMSE is calculated after combining squared-error
    sums, never by averaging individual RMSE values.
    """

    if frame.empty:
        raise ValueError(
            "Cannot summarize an empty DataFrame"
        )

    required_columns = (
        "valid_channel_pixels",
        "vv_count",
        "vh_count",
        "abs_sum",
        "sq_sum",
        "huber_sum",
        "baseline_abs_sum",
        "baseline_sq_sum",
        "vv_abs_sum",
        "vh_abs_sum",
        "vv_sq_sum",
        "vh_sq_sum",
    )

    missing = [
        col for col in required_columns
        if col not in frame.columns
    ]

    if missing:
        raise ValueError(
            f"Missing evaluation columns: {missing}"
        )

    total_pixels = float(
        frame["valid_channel_pixels"].sum()
    )

    vv_pixels = float(
        frame["vv_count"].sum()
    )

    vh_pixels = float(
        frame["vh_count"].sum()
    )

    if min(total_pixels, vv_pixels, vh_pixels) <= 0:
        raise ValueError(
            "No valid SAR pixels available for aggregation"
        )

    mae = (
        frame["abs_sum"].sum()
        / total_pixels
    )

    rmse = np.sqrt(
        frame["sq_sum"].sum()
        / total_pixels
    )

    huber = (
        frame["huber_sum"].sum()
        / total_pixels
    )

    baseline_mae = (
        frame["baseline_abs_sum"].sum()
        / total_pixels
    )

    baseline_rmse = np.sqrt(
        frame["baseline_sq_sum"].sum()
        / total_pixels
    )

    vv_mae = (
        frame["vv_abs_sum"].sum()
        / vv_pixels
    )

    vh_mae = (
        frame["vh_abs_sum"].sum()
        / vh_pixels
    )

    vv_rmse = np.sqrt(
        frame["vv_sq_sum"].sum()
        / vv_pixels
    )

    vh_rmse = np.sqrt(
        frame["vh_sq_sum"].sum()
        / vh_pixels
    )

    mae_improvement = (
        100.0 * (1.0 - mae / baseline_mae)
        if baseline_mae > 0
        else None
    )

    rmse_improvement = (
        100.0 * (1.0 - rmse / baseline_rmse)
        if baseline_rmse > 0
        else None
    )

    return {
        "observations": int(len(frame)),
        "valid_channel_pixels": int(total_pixels),

        "mae_normalized": float(mae),
        "rmse_normalized": float(rmse),
        "huber_normalized": float(huber),

        "vv_mae_normalized": float(vv_mae),
        "vh_mae_normalized": float(vh_mae),

        "vv_rmse_normalized": float(vv_rmse),
        "vh_rmse_normalized": float(vh_rmse),

        "baseline_mae_normalized": float(
            baseline_mae
        ),
        "baseline_rmse_normalized": float(
            baseline_rmse
        ),

        "mae_improvement_pct": (
            float(mae_improvement)
            if mae_improvement is not None
            else None
        ),
        "rmse_improvement_pct": (
            float(rmse_improvement)
            if rmse_improvement is not None
            else None
        ),
    }


@torch.no_grad()
def per_pixel_error_map(
    prediction,
    target,
    valid_mask,
    method="l1",
):
    """
    Generate a one-channel reconstruction-error map.

    Returns:
        [B, 1, H, W]

    Invalid locations are NaN.

    This output is useful for Phase 2.9 geographic
    anomaly-map generation.
    """

    valid = _validate_inputs(
        prediction, target, valid_mask
    )

    if method not in ("l1", "mse"):
        raise ValueError(
            "method must be 'l1' or 'mse'"
        )

    expanded = valid.expand_as(prediction)

    prediction = prediction.float()
    target = target.float()

    if not torch.isfinite(prediction[expanded]).all():
        raise ValueError(
            "Non-finite valid predictions"
        )

    if not torch.isfinite(target[expanded]).all():
        raise ValueError(
            "Non-finite valid targets"
        )

    safe_prediction = torch.where(
        expanded,
        prediction,
        torch.zeros_like(prediction),
    )

    safe_target = torch.where(
        expanded,
        target,
        torch.zeros_like(target),
    )

    difference = (
        safe_prediction - safe_target
    )

    if method == "l1":
        errors = difference.abs()
    else:
        errors = difference.square()

    errors = errors.mean(
        dim=1,
        keepdim=True,
    )

    return errors.masked_fill(
        ~valid,
        float("nan"),
    )
