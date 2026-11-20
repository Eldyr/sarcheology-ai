
import pytest
import torch

from sarchaeology.data.sarmae_adapter import (
    SARMAEAdapterConfig,
    SARMAEInputAdapter,
)


def make_batch(batch_size=2):
    generator = torch.Generator().manual_seed(42)

    x = torch.randn(
        batch_size, 2, 128, 128,
        generator=generator,
    )

    mask = torch.ones(
        batch_size, 1, 128, 128,
    )

    return x, mask


def test_output_shapes():
    images, mask = make_batch()

    adapter = SARMAEInputAdapter(
        SARMAEAdapterConfig(
            input_mode="normalized_proxy"
        )
    )

    output = adapter(images, mask)

    assert output["pixel_values"].shape == (
        4, 3, 224, 224
    )
    assert output["pixel_mask"].shape == (
        4, 1, 224, 224
    )
    assert output["polarization_index"].tolist() == [
        0, 0, 1, 1
    ]


def test_finite_output():
    images, mask = make_batch()

    adapter = SARMAEInputAdapter()
    output = adapter(images, mask)

    assert torch.isfinite(
        output["pixel_values"]
    ).all()


def test_grayscale_replication_before_normalization():
    images, mask = make_batch(1)

    adapter = SARMAEInputAdapter()
    out = adapter(images, mask)["pixel_values"]

    mean = torch.tensor(
        [0.485, 0.456, 0.406]
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225]
    ).view(1, 3, 1, 1)

    restored = out * std + mean

    assert torch.allclose(
        restored[:, 0],
        restored[:, 1],
        atol=1e-5,
    )
    assert torch.allclose(
        restored[:, 1],
        restored[:, 2],
        atol=1e-5,
    )


def test_invalid_pixels_do_not_affect_quantiles():
    images, mask = make_batch(1)

    mask[:, :, :16, :16] = 0

    changed = images.clone()
    changed[:, :, :16, :16] = 1e6

    adapter = SARMAEInputAdapter()

    first = adapter(images, mask)
    second = adapter(changed, mask)

    assert torch.allclose(
        first["pixel_values"],
        second["pixel_values"],
        atol=1e-6,
    )


def test_invalid_outputs_are_zero():
    images, mask = make_batch(1)
    mask[:, :, :32, :32] = 0

    out = SARMAEInputAdapter()(images, mask)

    invalid = ~out["pixel_mask"].expand_as(
        out["pixel_values"]
    )

    assert torch.all(
        out["pixel_values"][invalid] == 0
    )


def test_channel_validation():
    images = torch.zeros(
        1, 3, 128, 128
    )
    mask = torch.ones(
        1, 1, 128, 128
    )

    with pytest.raises(ValueError):
        SARMAEInputAdapter()(
            images, mask
        )


def test_deterministic_preprocessing():
    images, mask = make_batch(1)

    adapter = SARMAEInputAdapter()

    output1 = adapter(images, mask)
    output2 = adapter(images, mask)

    assert torch.equal(
        output1["pixel_values"],
        output2["pixel_values"],
    )


def test_constant_image_is_finite():
    images = torch.full(
        (1, 2, 128, 128),
        -15.0,
    )
    mask = torch.ones(
        (1, 1, 128, 128)
    )

    output = SARMAEInputAdapter()(
        images, mask
    )

    assert torch.isfinite(
        output["pixel_values"]
    ).all()
