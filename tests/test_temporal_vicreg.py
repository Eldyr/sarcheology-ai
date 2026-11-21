import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from sarchaeology.models.temporal_swin.losses import vicreg

def test_vicreg_finite_backward():
    a=torch.randn(8,16,requires_grad=True)
    b=torch.randn(8,16,requires_grad=True)
    inv,var,cov=vicreg(a,b)
    (inv+var+cov).backward()
    assert all(torch.isfinite(v).item() for v in (inv,var,cov))
    assert a.grad is not None and b.grad is not None

def test_vicreg_rejects_singleton():
    try: vicreg(torch.ones(1,4),torch.ones(1,4))
    except ValueError: pass
    else: raise AssertionError('Expected ValueError')
