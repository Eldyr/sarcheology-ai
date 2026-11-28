"""Phase 3.7 tests; run from SARchaeology AI repository root."""
import importlib.util
from pathlib import Path

import pytest
import torch

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/phase3_optical/evaluate_optical_phase37.py'
spec=importlib.util.spec_from_file_location('evaluate_optical_phase37',SCRIPT)
audit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def fake_samples(n=6):
    return [{'patch_id':f'loc-{i:02d}', 'quarter':q,'year':2024}
            for q in (1,2,3,4) for i in range(n)]


def test_retrieval_correct_and_chance():
    s=fake_samples(6)
    # Each observation encodes the same location independently of quarter.
    z=torch.eye(6).repeat(4,1)
    metric,outcomes=audit.retrieval(z,s)
    assert metric['queries']==6*12
    assert metric['top1_accuracy']==1.
    assert metric['top5_accuracy']==1.
    assert metric['chance_top1']==1/6
    assert metric['chance_top5']==5/6
    assert len(outcomes)==72


def test_retrieval_fails_with_missing_quarter_location():
    s=fake_samples(6)
    s=[x for x in s if not (x['patch_id']=='loc-01' and x['quarter']==4)]
    with pytest.raises(RuntimeError,match='Quarter retrieval'):
        audit.retrieval(torch.randn(len(s),12),s)


def test_geographic_test_split_no_leakage():
    class DS:
        samples=fake_samples(6)
    data=audit.validate_test(DS(),set(f'loc-{i:02d}' for i in range(6)),
                             {'training'}, {'validation'})
    assert data['test_observations']==24
    with pytest.raises(RuntimeError,match='leakage'):
        audit.validate_test(DS(),{'loc-01'}, {'loc-01'},set())
    with pytest.raises(RuntimeError,match='Incomplete'):
        class Missing:
            samples=DS.samples[:-1]
        audit.validate_test(Missing(),set(f'loc-{i:02d}' for i in range(6)),
                            {'training'},{'validation'})


def test_bootstrap_zero_difference():
    samples=fake_samples(6)
    z=torch.eye(6).repeat(4,1)
    _,a=audit.retrieval(z,samples)
    res=audit.paired_bootstrap(a,a,repeats=100)
    assert res['difference']==0
    assert res['95pct_ci']==[0.,0.]
    assert res['bootstrap_clusters']==6


def test_full_swin_pass_has_no_mask():
    """A cheap stand-in with the same encoder layers and tensor shapes."""
    from torch import nn
    class Stage(nn.Module):
        def forward(self,x):return x+1
    class Merge(nn.Module):
        def forward(self,x):return x[:,::2,::2,:]
    class Tiny:
        patch_embed=nn.Conv2d(6,4,4,4)
        stages=[Stage(),Stage(),Stage(),Stage()]
        merges=[Merge(),Merge(),Merge()]
        encoder_norm=nn.Identity()
    result=audit.swin_full(Tiny(),torch.zeros(2,6,128,128),capture_stage2=True)
    assert result['embedding'].shape==(2,4)
    assert result['stage2'].shape==(2,4,8,8)
