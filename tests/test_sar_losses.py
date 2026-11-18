
import pytest
import torch

from sarchaeology.training.losses import (
    MaskedSARReconstructionLoss,
    build_sar_loss,
    reconstruction_error_map,
)


def test_masked_mse_exact():
    # One image, two channels, two spatial pixels.
    target = torch.zeros(1, 2, 1, 2)

    prediction = torch.tensor([
        [
            [[2.0, 100.0]],
            [[4.0, 100.0]],
        ]
    ])

    mask = torch.tensor([
        [[[1.0, 0.0]]]
    ])

    loss = MaskedSARReconstructionLoss(
        loss_type="mse"
    )(prediction, target, mask)

    # Valid squared errors: 4 and 16.
    # Result = (4 + 16) / 2 = 10.
    assert loss.item() == pytest.approx(10.0)


def test_masked_l1_exact():
    target = torch.zeros(1, 2, 1, 2)

    prediction = torch.tensor([
        [
            [[2.0, 100.0]],
            [[4.0, 100.0]],
        ]
    ])

    mask = torch.tensor([
        [[[1.0, 0.0]]]
    ])

    loss = MaskedSARReconstructionLoss(
        loss_type="l1"
    )(prediction, target, mask)

    assert loss.item() == pytest.approx(3.0)


def test_masked_huber_exact():
    target = torch.zeros(1, 2, 1, 1)

    prediction = torch.tensor([
        [
            [[0.5]],
            [[2.0]],
        ]
    ])

    mask = torch.ones(1, 1, 1, 1)

    loss = MaskedSARReconstructionLoss(
        loss_type="huber",
        huber_delta=1.0,
    )(prediction, target, mask)

    # Huber(0.5) = 0.125
    # Huber(2.0) = 1.5
    # Mean = 0.8125
    assert loss.item() == pytest.approx(0.8125)


def test_invalid_nan_pixels_ignored():
    prediction = torch.tensor([
        [
            [[1.0, float("nan")]],
            [[3.0, float("nan")]],
        ]
    ])

    target = torch.zeros_like(prediction)

    mask = torch.tensor([
        [[[1.0, 0.0]]]
    ])

    loss = MaskedSARReconstructionLoss(
        loss_type="mse"
    )(prediction, target, mask)

    assert loss.item() == pytest.approx(5.0)


def test_valid_nan_pixels_rejected():
    prediction = torch.tensor([
        [
            [[float("nan")]],
            [[1.0]],
        ]
    ])

    target = torch.zeros_like(prediction)
    mask = torch.ones(1, 1, 1, 1)

    with pytest.raises(ValueError):
        MaskedSARReconstructionLoss()(
            prediction, target, mask
        )


def test_channel_weights():
    prediction = torch.tensor([
        [
            [[1.0]],
            [[3.0]],
        ]
    ])

    target = torch.zeros_like(prediction)
    mask = torch.ones(1, 1, 1, 1)

    loss = MaskedSARReconstructionLoss(
        loss_type="mse",
        channel_weights=(3.0, 1.0),
    )(prediction, target, mask)

    # (3*1 + 1*9) / (3+1) = 3.
    assert loss.item() == pytest.approx(3.0)


def test_no_valid_pixels_rejected():
    x = torch.zeros(1, 2, 2, 2)
    mask = torch.zeros(1, 1, 2, 2)

    with pytest.raises(ValueError):
        MaskedSARReconstructionLoss()(
            x, x, mask
        )


def test_backward_gradients():
    prediction = torch.randn(
        2, 2, 8, 8,
        requires_grad=True,
    )

    target = torch.zeros_like(prediction)
    mask = torch.ones(2, 1, 8, 8)
    mask[:, :, :4, :] = 0

    loss = MaskedSARReconstructionLoss(
        loss_type="huber"
    )(prediction, target, mask)

    loss.backward()

    assert prediction.grad is not None
    assert torch.isfinite(
        prediction.grad
    ).all()

    # Masked positions receive no gradient.
    assert torch.all(
        prediction.grad[:, :, :4, :] == 0
    )

    # Valid positions should generally
    # receive non-zero gradients.
    assert torch.any(
        prediction.grad[:, :, 4:, :] != 0
    )


def test_per_sample_reduction():
    prediction = torch.tensor([
        [[[1.0]], [[1.0]]],
        [[[3.0]], [[3.0]]],
    ])

    target = torch.zeros_like(prediction)
    mask = torch.ones(2, 1, 1, 1)

    loss = MaskedSARReconstructionLoss(
        loss_type="mse",
        reduction="none",
    )(prediction, target, mask)

    assert loss.shape == (2,)
    assert torch.allclose(
        loss,
        torch.tensor([1.0, 9.0]),
    )


def test_reconstruction_error_map():
    prediction = torch.tensor([
        [
            [[1.0, 9.0]],
            [[3.0, 9.0]],
        ]
    ])

    target = torch.zeros_like(prediction)

    mask = torch.tensor([
        [[[1.0, 0.0]]]
    ])

    result = reconstruction_error_map(
        prediction,
        target,
        mask,
        method="l1",
    )

    assert result.shape == (1, 1, 1, 2)
    assert result[0, 0, 0, 0] == 2.0
    assert torch.isnan(result[0, 0, 0, 1])


def test_build_from_config():
    config = {
        "loss": {
            "type": "huber",
            "huber_delta": 1.0,
            "reduction": "mean",
            "channel_weights": [1.0, 1.0],
        }
    }

    criterion = build_sar_loss(config)

    assert isinstance(
        criterion,
        MaskedSARReconstructionLoss,
    )
    assert criterion.loss_type == "huber"
