import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from sarchaeology.data.optical_temporal_pairs import TemporalPairs

class Dummy:
    def __init__(self):
        self.samples=[dict(patch_id=f'p{p}',quarter=q,year=y) for p in range(3) for y in (2022,2023) for q in (1,2,3,4)]
    def __len__(self): return len(self.samples)
    def __getitem__(self,index):
        a=self.samples[index]
        return dict(image=torch.zeros(6,128,128),valid_mask=torch.ones(1,128,128,dtype=torch.bool),**a)

def test_temporal_pair_identity_and_determinism():
    ds=TemporalPairs(Dummy(),same_quarter_probability=.75,seed=42)
    ds.set_epoch(2)
    pairs=[ds.partner_index(i) for i in range(len(ds))]
    assert pairs==[ds.partner_index(i) for i in range(len(ds))]
    for i,j in enumerate(pairs):
        assert i!=j and ds.base.samples[i]['patch_id']==ds.base.samples[j]['patch_id']
