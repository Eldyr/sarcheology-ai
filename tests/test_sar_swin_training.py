"""Phase 2.15 trainer invariants. Run with: uv run pytest tests/test_sar_swin_training.py -v"""
from pathlib import Path
import runpy

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
api = runpy.run_path(str(ROOT / "scripts/phase2_sar/train_sar_swin_mae.py"))
make_mask = api["make_mask"]
masked_huber_stats = api["masked_huber_stats"]


def sample():
    return {"image": torch.randn(2, 2, 128, 128),
            "patch_id": ["r00064_c00960", "r00128_c00960"],
            "acquisition_date": ["2024-01-10", "2024-01-22"]}


def mask(batch, epoch, validation):
    return make_mask(batch, mask_ratio=.4, block_size=16, seed=2026,
                     epoch=epoch, validation=validation, device=torch.device("cpu"))


def test_validation_mask_independent_of_epoch():
    batch = sample()
    assert torch.equal(mask(batch, 1, True), mask(batch, 50, True))


def test_training_mask_changes_between_epochs():
    batch = sample()
    assert not torch.equal(mask(batch, 1, False), mask(batch, 2, False))


def test_validation_mask_independent_of_batch_order():
    a = sample()
    b = {"image": a["image"].flip(0),
         "patch_id": list(reversed(a["patch_id"])),
         "acquisition_date": list(reversed(a["acquisition_date"]))}
    assert torch.equal(mask(a, 1, True), mask(b, 999, True).flip(0))


def test_masked_fraction():
    m = mask(sample(), 1, True)
    assert m.shape == (2, 1, 128, 128)
    assert m.float().mean().item() == pytest.approx(26/64)


def test_huber_ignores_visible_pixels():
    target = torch.zeros(1, 2, 2, 2)
    pred = torch.tensor([[[[2., 999.], [999., 999.]],
                          [[4., 999.], [999., 999.]]]])
    region = torch.tensor([[[[True, False], [False, False]]]])
    loss, stats = masked_huber_stats(pred, target, torch.ones(1,1,2,2), region, delta=1.)
    assert loss.item() == pytest.approx(2.5)
    assert stats["count"].item() == 2


def test_huber_ignores_invalid_nan_targets():
    target = torch.zeros(1, 2, 2, 2)
    target[:, :, 0, 0] = float("nan")
    pred = torch.ones_like(target)
    valid = torch.ones(1, 1, 2, 2)
    valid[:, :, 0, 0] = 0
    loss, stats = masked_huber_stats(pred, target, valid,
                                    torch.ones(1,1,2,2,dtype=torch.bool), delta=1.)
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(0.5)
    assert stats["count"].item() == 6
