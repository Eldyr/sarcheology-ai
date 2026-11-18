
import numpy as np
import pytest
import pandas as pd
import torch


from sarchaeology.evaluation.sar_metrics import (
    huber_sums,
    masked_image_metrics,
    summarize,
)



def test_masked_metrics_ignore_invalid_pixels():
    target = torch.zeros(1, 2, 1, 2)

    prediction = torch.tensor([
        [
            [[2.0, 999.0]],
            [[4.0, 999.0]],
        ]
    ])

    mask = torch.tensor([
        [[[1.0, 0.0]]]
    ])

    counts, absolute, squared = masked_image_metrics(
        prediction, target, mask
    )

    assert counts.sum().item() == 2
    assert absolute.sum().item() == pytest.approx(6.0)
    assert squared.sum().item() == pytest.approx(20.0)

    assert (
        absolute.sum() / counts.sum()
    ).item() == pytest.approx(3.0)

    assert torch.sqrt(
        squared.sum() / counts.sum()
    ).item() == pytest.approx(np.sqrt(10))


def test_huber_matches_expected_values():
    target = torch.zeros(1, 2, 1, 1)

    prediction = torch.tensor([
        [
            [[0.5]],
            [[2.0]],
        ]
    ])

    mask = torch.ones(1, 1, 1, 1)

    sums = huber_sums(
        prediction, target, mask, delta=1.0
    )

    # Huber(0.5) = 0.125
    # Huber(2.0) = 1.5
    assert sums.item() == pytest.approx(1.625)


def test_error_aggregation_uses_pixel_counts():
    frame = pd.DataFrame([
        {
            "valid_channel_pixels": 2,
            "vv_count": 1,
            "vh_count": 1,
            "abs_sum": 6.0,
            "sq_sum": 20.0,
            "huber_sum": 5.0,
            "baseline_abs_sum": 10.0,
            "baseline_sq_sum": 50.0,
            "vv_abs_sum": 2.0,
            "vh_abs_sum": 4.0,
            "vv_sq_sum": 4.0,
            "vh_sq_sum": 16.0,
        },
        {
            "valid_channel_pixels": 4,
            "vv_count": 2,
            "vh_count": 2,
            "abs_sum": 4.0,
            "sq_sum": 4.0,
            "huber_sum": 2.0,
            "baseline_abs_sum": 8.0,
            "baseline_sq_sum": 16.0,
            "vv_abs_sum": 2.0,
            "vh_abs_sum": 2.0,
            "vv_sq_sum": 2.0,
            "vh_sq_sum": 2.0,
        },
    ])

    metrics = summarize(frame)

    assert metrics["observations"] == 2
    assert metrics["valid_channel_pixels"] == 6

    assert metrics["mae_normalized"] == pytest.approx(
        10.0 / 6.0
    )

    assert metrics["rmse_normalized"] == pytest.approx(
        np.sqrt(24.0 / 6.0)
    )

    assert metrics["huber_normalized"] == pytest.approx(
        7.0 / 6.0
    )


def test_invalid_predictions_are_rejected():
    prediction = torch.full(
        (1, 2, 2, 2), float("nan")
    )
    target = torch.zeros_like(prediction)
    mask = torch.ones(1, 1, 2, 2)

    with pytest.raises(ValueError):
        masked_image_metrics(
            prediction, target, mask
        )
