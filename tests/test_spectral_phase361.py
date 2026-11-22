"""Fast checks for Phase 3.6.1 metrics; run with uv run python -m pytest -q."""
import sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/phase3_optical'))
from evaluate_spectral_mae import ridge_fit, retrieval, spectral_mask, effective_rank


def test_two_band_mask():
    m=spectral_mask(('B11','B12'),4,'cpu')
    assert m.shape==(4,6) and m.sum().item()==8
    assert torch.equal(m[:,4:],torch.ones((4,2),dtype=torch.bool))


def test_train_only_ridge_recovers_linear_relationship():
    rng=np.random.default_rng(33)
    x=rng.normal(size=(200,4))
    y=x@np.array([[1.2,-0.5],[0.3,0.1],[-0.4,0.8],[0.2,0.2]]) + np.array([0.04,0.12])
    fitted=ridge_fit(x,y,1e-6)
    np.testing.assert_allclose(np.c_[x,np.ones(len(x))]@fitted,y,atol=1e-6)


def test_retrieval_matches_geography_across_quarters():
    data=[];vec=[]
    for q in range(1,5):
        for i in range(20):
            data.append({'quarter':q,'patch_id':f'patch{i}'})
            row=np.zeros(20,dtype='float32');row[i]=1
            vec.append(row)
    found=retrieval(torch.tensor(np.stack(vec)),data)
    assert found['top1_accuracy']==1.0
    assert found['top5_accuracy']==1.0
    assert found['queries']==240
    assert effective_rank(np.asarray(vec))['effective_rank']>15
