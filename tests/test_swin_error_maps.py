"""Phase 2.16 Swin map correctness tests (run from repository root)."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pytest
import torch
from rasterio.transform import from_origin

from scripts.phase2_sar.generate_swin_error_maps import (
    MapAccumulation,
    complementary_masks,
    patch_position,
)
from scripts.phase2_sar.evaluate_sar_swin_mae import fixed_mask


def test_first_mask_reproduces_training_validation():
    kwargs = dict(ratio=0.4, block_size=16, seed=2026, device=torch.device("cpu"))
    masks = list(complementary_masks(
        ["r00064_c00960"], ["2024-01-10"], 128, 128, passes=4, **kwargs
    ))
    expected = fixed_mask(["r00064_c00960"], ["2024-01-10"], 128, 128,
                          0.4, 16, 2026, torch.device("cpu"))
    assert torch.equal(masks[0], expected)
    assert len(masks) == 4
    assert all(int(mask.sum()) == 26 * 16 * 16 for mask in masks)


def test_complementary_masks_cover_every_pixel_and_are_repeatable():
    kwargs = dict(ratio=0.4, block_size=16, seed=2026, passes=4,
                  device=torch.device("cpu"))
    first = list(complementary_masks(["x", "y"], ["day-a", "day-b"], 128, 128, **kwargs))
    second = list(complementary_masks(["x", "y"], ["day-a", "day-b"], 128, 128, **kwargs))
    assert all(torch.equal(a, b) for a, b in zip(first, second))
    assert torch.stack(first).any(dim=0).all().item()


def test_reject_insufficient_masks():
    with pytest.raises(ValueError, match="Need more passes"):
        list(complementary_masks(["x"], ["day"], 128, 128, ratio=0.4,
                                 block_size=16, seed=1, passes=2,
                                 device=torch.device("cpu")))


def test_map_aggregate_counts_only_scored_pixels():
    acc = MapAccumulation(6, 6)
    errors = np.array([[2., 4.], [10., 20.]], dtype=np.float32)
    selected = np.array([[True, False], [False, True]])
    acc.add(1, 2, errors, selected)
    acc.add(1, 2, np.full((2, 2), 6., dtype=np.float32), selected)
    mean, rmse = acc.maps()
    assert np.isnan(mean[0, 0])
    assert acc.count[1, 2] == 2
    assert acc.count[1, 3] == 0
    assert mean[1, 2] == pytest.approx(4.0)
    assert rmse[1, 2] == pytest.approx(np.sqrt(20))
    assert mean[2, 3] == pytest.approx(13.0)


def test_patch_pixel_offsets_match_geographic_bounds():
    transform = from_origin(278810., 3460740., 10., 10.)
    entry = {
        "row": 64, "col": 960, "height": 128, "width": 128,
        "xmin": 288410., "ymin": 3458820.,
        "xmax": 289690., "ymax": 3460100.,
    }
    assert patch_position(entry, transform, 1358, 1362) == (64, 960, 128, 128)
    entry["xmin"] += 10
    with pytest.raises(ValueError, match="transform mismatch"):
        patch_position(entry, transform, 1358, 1362)
