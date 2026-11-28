"""Unit tests for Phase 3.6.2; no satellite dataset required."""
from __future__ import annotations
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from sarchaeology.models.hybrid_optical import SpatialSpectralGatedAdapter


def sample(n=4):
    teacher=F.normalize(torch.randn(n,256),dim=1)
    swin=torch.randn(n,384,8,8)
    spectral=torch.randn(n,96,8,8)
    return teacher,swin,spectral


def test_shapes_normalization_and_gate():
    net=SpatialSpectralGatedAdapter()
    out=net(*sample())
    assert out['embedding'].shape==(4,256)
    assert torch.allclose(out['embedding'].norm(dim=-1),torch.ones(4),atol=1e-5)
    assert 0 <= out['gate_mean'].item() <= 1


def test_gradient_only_through_adapter():
    net=SpatialSpectralGatedAdapter()
    teacher,swin,spectral=sample()
    output=net(teacher,swin,spectral)
    loss=(output['embedding']**2).sum()+output['gate_mean']
    loss.backward()
    assert any(p.grad is not None for p in net.parameters())
    assert teacher.grad is None and swin.grad is None and spectral.grad is None


def test_residual_zero_is_exact_teacher():
    net=SpatialSpectralGatedAdapter()
    net.residual_scale.data.zero_()
    teacher,swin,spectral=sample()
    out=net(teacher,swin,spectral)['embedding']
    assert torch.allclose(out,teacher,atol=1e-6)


def test_shape_rejection():
    net=SpatialSpectralGatedAdapter()
    teacher,swin,spectral=sample()
    with pytest.raises(ValueError):
        net(teacher,swin[:,:100],spectral)
    with pytest.raises(ValueError):
        net(teacher,swin,spectral[:,:,:4,:4])
