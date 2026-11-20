"""Fast tests for Phase 2.16 evaluation helpers; no model weights required."""
import importlib.util
from pathlib import Path
import pytest
import torch

PATH = Path(__file__).resolve().parents[1] / "scripts/phase2_sar/evaluate_sar_swin_mae.py"
spec = importlib.util.spec_from_file_location("swin_eval", PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_validation_mask_constant_across_calls():
    kwargs = (['r00064_c00960', 'r00128_c00960'], ['2024-01-10', '2024-01-22'])
    a = mod.fixed_mask(*kwargs, 128, 128, 0.4, 16, 2026, 'cpu')
    b = mod.fixed_mask(*kwargs, 128, 128, 0.4, 16, 2026, 'cpu')
    assert torch.equal(a, b)
    assert a.shape == (2, 1, 128, 128)
    assert a.float().mean().item() == pytest.approx(26/64)


def test_mask_matches_trainer_algorithm():
    patch, date = 'r00064_c00960', '2024-01-10'
    g = torch.Generator(device='cpu').manual_seed(
        mod.stable_seed(2026, patch, date, 'validation')
    )
    order = torch.randperm(64, generator=g)
    expected = torch.zeros((64,), dtype=torch.bool)
    expected[order[:26]] = True
    mask = mod.fixed_mask([patch], [date], 128, 128, 0.4, 16, 2026, 'cpu')
    # One representative pixel from each 16x16 block.
    assert torch.equal(mask[0, 0, ::16, ::16].flatten(), expected)


def test_masked_reduction_excludes_visible_and_invalid():
    prediction = torch.tensor([[[[2., 100.], [100., 100.]], [[4., 100.], [100., 100.]]]])
    truth = torch.zeros_like(prediction)
    mask = torch.tensor([[[[True, False], [False, False]]]])
    totals = mod.reductions(prediction, truth, mask, 1.0)
    assert totals['count'].sum().item() == 2
    assert totals['huber_sum'].sum().item() == pytest.approx(5.0)
    assert totals['abs_sum'].sum().item() == pytest.approx(6.0)
    assert totals['sq_sum'].sum().item() == pytest.approx(20.0)


def test_nan_outside_valid_region_excluded():
    pred = torch.zeros(1, 2, 2, 2)
    truth = torch.full_like(pred, float('nan'))
    mask = torch.zeros(1, 1, 2, 2, dtype=torch.bool)
    mask[:, :, 0, 0] = True
    truth[:, :, 0, 0] = 1.
    sums = mod.reductions(pred, truth, mask, 1.0)
    assert torch.isfinite(sums['huber_sum']).all()
    assert sums['huber_sum'].sum().item() == pytest.approx(1.0)
