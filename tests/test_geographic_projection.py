import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from sarchaeology.models.geographic_projection import (GeographicProjector,
            contrastive_loss,variance_covariance_loss,relational_loss)


def test_shape_and_gradients():
    head=GeographicProjector()
    x=torch.randn(8,768)
    z=head(x)
    assert z.shape==(8,256)
    assert torch.allclose(z.norm(dim=1),torch.ones(8),atol=1e-5)
    loss=contrastive_loss(z,torch.tensor([0,1,2,3,0,1,2,3]))
    loss.backward()
    assert head.skip.weight.grad is not None
    assert torch.isfinite(head.skip.weight.grad).all()


def test_positive_pairs_and_repeated_ids():
    z=torch.tensor([[1.,0.],[0.,1.],[1.,0.],[0.,1.]])
    ids=torch.tensor([0,1,0,1])
    good=contrastive_loss(z,ids)
    wrong=contrastive_loss(z,torch.tensor([0,0,1,1]))
    assert good < wrong
    with pytest.raises(ValueError):
        contrastive_loss(z,torch.tensor([0,1,2,3]))


def test_variance_and_preservation():
    x=torch.randn(12,256)
    var,cov=variance_covariance_loss(x)
    assert var>=0 and cov>=0
    assert relational_loss(x,x).item()<1e-5


def test_geographic_negatives_and_pairs():
    import ast
    import math
    from collections import defaultdict
    import numpy as np
    source=(Path(__file__).resolve().parents[1]/'scripts/phase3_optical/train_geographic_projection.py').read_text()
    tree=ast.parse(source)
    names={'grouped','draw_pair','make_pairs','distant_batches','location_coordinates'}
    code=compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),'<functions>','exec')
    context={'defaultdict':defaultdict,'np':np,'torch':torch,'math':math,'re':__import__('re')}
    exec(code,context)
    samples=[]
    for pid,col in [('r00000_c00000',0),('r00000_c00128',128),('r00000_c00256',256),('r00000_c00384',384)]:
        for year,quarter in [(2022,1),(2023,1),(2024,2)]:
            samples.append({'patch_id':pid,'year':year,'quarter':quarter})
    groups=context['grouped'](samples)
    coords=context['location_coordinates'](groups,samples)
    batches=list(context['distant_batches'](list(groups),coords,128,2))
    assert all(len(x)>=2 for x in batches)
    idx,labels=context['make_pairs'](groups,samples,['r00000_c00000','r00000_c00128'],np.random.default_rng(5),.75)
    assert labels.tolist()==[0,1,0,1]
    assert idx[0]!=idx[2]
    assert samples[idx[0]]['patch_id']==samples[idx[2]]['patch_id']
