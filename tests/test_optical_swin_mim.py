import torch
from sarchaeology.models.optical_swin_mim import OpticalSwinMIM


def test_shapes_patch_order_mask_reproducibility():
    model = OpticalSwinMIM(embed_dim=24, heads=(3, 3, 6, 12), depths=(1, 1, 1, 1), decoder_dim=32)
    data = torch.randn(2, 6, 128, 128)
    mask = torch.ones(2, 1, 128, 128, dtype=torch.bool)
    assert torch.equal(model.unpatchify(model.patchify(data)), data)
    with torch.no_grad():
        a = model(data, mask, generator=torch.Generator().manual_seed(123))
        b = model(data, mask, generator=torch.Generator().manual_seed(123))
    assert a['predicted_patches'].shape == (2, 256, 384)
    assert a['patch_mask'].shape == (2, 256)
    assert (a['patch_mask'].sum(1) == 192).all()  # 75% of 256
    assert torch.equal(a['patch_mask'], b['patch_mask'])
    assert torch.isfinite(a['reconstruction']).all()


def test_masked_training_gradient():
    model = OpticalSwinMIM(embed_dim=24, heads=(3, 3, 6, 12), depths=(1, 1, 1, 1), decoder_dim=32)
    data = torch.rand(1, 6, 128, 128)
    valid = torch.ones(1, 1, 128, 128, dtype=torch.bool)
    valid[:, :, :8, :] = False
    output = model(data, valid, generator=torch.Generator().manual_seed(42))
    pred = output['predicted_patches']
    scored = output['patch_mask'].unsqueeze(-1) & (model.patchify(valid.expand_as(data).float()) > 0.5)
    diff = pred - model.patchify(data)
    loss = ((diff.square() * scored).sum() / scored.sum().clamp_min(1))
    loss.backward()
    assert model.patch_embed.weight.grad is not None
    assert torch.isfinite(model.patch_embed.weight.grad).all()
