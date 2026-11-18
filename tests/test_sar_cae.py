
import pytest
import torch

from sarchaeology.models.sar_cae import (
    SARCAE,
    build_sar_cae,
    count_parameters,
)


@pytest.fixture
def small_model():
    # Smaller configuration keeps CPU tests fast.
    return SARCAE(
        in_channels=2,
        base_channels=8,
        latent_channels=16,
        embedding_dim=32,
        group_norm_groups=8,
    )


def test_forward_shapes(small_model):
    images = torch.randn(2, 2, 32, 32)

    outputs = small_model(images)

    assert outputs["reconstruction"].shape == (
        2, 2, 32, 32
    )

    assert outputs["latent_map"].shape == (
        2, 16, 4, 4
    )

    assert outputs["embedding"].shape == (
        2, 32
    )


def test_masked_input(small_model):
    small_model.eval()

    images = torch.randn(1, 2, 32, 32)

    mask = torch.ones(1, 1, 32, 32)
    mask[:, :, :8, :] = 0

    modified = images.clone()
    modified[:, :, :8, :] = 999.0

    with torch.no_grad():
        first = small_model(
            images, valid_mask=mask
        )["reconstruction"]

        second = small_model(
            modified, valid_mask=mask
        )["reconstruction"]

    assert torch.allclose(
        first, second, atol=1e-6
    )


def test_backward_pass(small_model):
    images = torch.randn(2, 2, 32, 32)

    outputs = small_model(images)

    # Simple differentiability check.
    # The actual masked reconstruction objective
    # is implemented in Phase 2.5.
    loss = outputs["reconstruction"].square().mean()
    loss.backward()

    gradients = [
        p.grad
        for p in small_model.parameters()
        if p.grad is not None
    ]

    assert gradients
    assert all(
        torch.isfinite(g).all()
        for g in gradients
    )

    assert any(
        torch.count_nonzero(g).item() > 0
        for g in gradients
    )


def test_invalid_shapes_rejected(small_model):
    with pytest.raises(ValueError):
        small_model(
            torch.randn(1, 3, 32, 32)
        )

    with pytest.raises(ValueError):
        small_model(
            torch.randn(1, 2, 30, 32)
        )

    with pytest.raises(ValueError):
        small_model(
            torch.randn(1, 2, 32, 32),
            valid_mask=torch.ones(
                1, 2, 32, 32
            ),
        )


def test_model_is_trainable(small_model):
    assert count_parameters(small_model) > 0

    assert all(
        p.requires_grad
        for p in small_model.parameters()
    )


def test_config_builder():
    config = {
        "model": {
            "architecture": "sar_cae",
            "in_channels": 2,
            "base_channels": 32,
            "latent_channels": 128,
            "embedding_dim": 128,
            "group_norm_groups": 8,
        }
    }

    model = build_sar_cae(config)

    assert isinstance(model, SARCAE)
    assert model.embedding_dim == 128
