
import pytest
import torch

from sarchaeology.models.sar_swin_mae import (
    SARSwinMAE,
    build_sar_swin_mae,
)


@pytest.fixture(scope="module")
def model():
    return SARSwinMAE(
        input_size=128,
        mask_ratio=0.4,
        mask_block_size=16,
    )


def test_mask_ratio(model):
    x = torch.zeros(2, 2, 128, 128)
    mask = model.make_block_mask(x)

    assert mask.shape == (2, 1, 128, 128)

    ratios = mask.float().mean(dim=(1, 2, 3))

    # 64 total blocks: 26 masked = 40.625%.
    assert torch.allclose(
        ratios,
        torch.full_like(ratios, 26 / 64),
    )


def test_forward_shapes(model):
    model.eval()

    x = torch.randn(1, 2, 128, 128)
    valid = torch.ones(1, 1, 128, 128)

    with torch.no_grad():
        output = model(x, valid_mask=valid)

    assert output["reconstruction"].shape == x.shape
    assert output["embedding"].shape == (1, 768)
    assert output["latent"].shape == (1, 768, 4, 4)
    assert output["masked_pixels"].shape == valid.shape

    assert torch.isfinite(
        output["reconstruction"]
    ).all()


def test_zero_mask_ratio():
    model = build_sar_swin_mae({
        "input_size": 128,
        "in_channels": 2,
        "mask_ratio": 0.0,
    })
    x = torch.zeros(1, 2, 128, 128)

    mask = model.make_block_mask(x)
    assert not mask.any()


def test_invalid_pixels_never_become_targets(model):
    x = torch.randn(1, 2, 128, 128)
    valid = torch.ones(1, 1, 128, 128)
    valid[:, :, :32, :32] = 0

    model.eval()

    with torch.no_grad():
        output = model(x, valid_mask=valid)

    assert not output["masked_pixels"][
        :, :, :32, :32
    ].any()


def test_invalid_input_values_do_not_affect_output(model):
    x = torch.randn(1, 2, 128, 128)
    valid = torch.ones(1, 1, 128, 128)
    valid[:, :, :16, :16] = 0

    altered = x.clone()
    altered[:, :, :16, :16] = float("nan")

    fixed_mask = torch.zeros(
        1, 1, 128, 128, dtype=torch.bool
    )
    fixed_mask[:, :, 32:48, 32:48] = True

    model.eval()

    with torch.no_grad():
        a = model(
            x,
            valid_mask=valid,
            synthetic_mask=fixed_mask,
        )["reconstruction"]

        b = model(
            altered,
            valid_mask=valid,
            synthetic_mask=fixed_mask,
        )["reconstruction"]

    assert torch.allclose(a, b, atol=1e-5)


def test_configuration_validation():
    with pytest.raises(ValueError):
        SARSwinMAE(input_size=120)

    with pytest.raises(ValueError):
        SARSwinMAE(mask_ratio=1.0)

    with pytest.raises(ValueError):
        SARSwinMAE(mask_block_size=15)
