
import pytest
import torch

from sarchaeology.models.optical_mae import (
    OpticalViTMAE,
)


@pytest.fixture(scope="module")
def model():
    return OpticalViTMAE()


def test_patchify_unpatchify(model):
    x = torch.randn(2, 6, 128, 128)

    patches = model.patchify(x)
    restored = model.unpatchify(patches)

    assert patches.shape == (2, 256, 384)
    assert torch.equal(restored, x)


def test_mask_ratio(model):
    tokens = torch.randn(2, 256, 192)
    generator = torch.Generator().manual_seed(42)

    visible, mask, keep, restore = (
        model.random_masking(
            tokens, generator=generator
        )
    )

    assert visible.shape == (2, 64, 192)
    assert mask.shape == (2, 256)
    assert mask.sum(dim=1).tolist() == [192, 192]
    assert keep.shape == (2, 64)
    assert restore.shape == (2, 256)


def test_forward_shapes(model):
    model.eval()

    x = torch.randn(1, 6, 128, 128)
    valid = torch.ones(1, 1, 128, 128)

    with torch.no_grad():
        output = model(
            x,
            valid,
            generator=torch.Generator().manual_seed(11),
        )

    assert output["reconstruction"].shape == (
        1, 6, 128, 128
    )
    assert output["predicted_patches"].shape == (
        1, 256, 384
    )
    assert output["embedding"].shape == (
        1, 192
    )
    assert torch.isfinite(
        output["reconstruction"]
    ).all()


def test_deterministic_mask(model):
    x = torch.randn(1, 6, 128, 128)

    model.eval()

    with torch.no_grad():
        a = model(
            x,
            generator=torch.Generator().manual_seed(2026),
        )
        b = model(
            x,
            generator=torch.Generator().manual_seed(2026),
        )

    assert torch.equal(
        a["patch_mask"], b["patch_mask"]
    )

    assert torch.allclose(
        a["reconstruction"],
        b["reconstruction"],
        atol=1e-6,
    )


def test_invalid_pixels_are_ignored(model):
    model.eval()

    x = torch.randn(1, 6, 128, 128)
    altered = x.clone()

    valid = torch.ones(
        1, 1, 128, 128, dtype=torch.bool
    )

    valid[:, :, :16, :16] = False
    altered[:, :, :16, :16] = float("nan")

    with torch.no_grad():
        a = model(
            x,
            valid,
            generator=torch.Generator().manual_seed(5),
        )
        b = model(
            altered,
            valid,
            generator=torch.Generator().manual_seed(5),
        )

    assert torch.allclose(
        a["reconstruction"],
        b["reconstruction"],
        atol=1e-5,
    )


def test_backward_gradients():
    model = OpticalViTMAE(
        encoder_depth=2,
        decoder_depth=1,
    )

    x = torch.randn(2, 6, 128, 128)
    valid = torch.ones(
        2, 1, 128, 128, dtype=torch.bool
    )

    output = model(x, valid)
    loss = model.masked_loss(
        x, valid, output
    )

    assert torch.isfinite(loss)

    loss.backward()

    assert model.patch_embed.weight.grad is not None
    assert torch.isfinite(
        model.patch_embed.weight.grad
    ).all()


def test_invalid_configuration():
    with pytest.raises(ValueError):
        OpticalViTMAE(patch_size=7)

    with pytest.raises(ValueError):
        OpticalViTMAE(mask_ratio=1.0)
