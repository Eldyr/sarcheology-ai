"""Held-out 2024 cross-quarter retrieval: pretrained vs Phase 3.5.

Each query quarter is matched against 20 held-out geographic patches in each
other quarter (4 x 3 = 12 directional quarter-pair tasks).
Uses fixed masks and identical input observations for both encoder models.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from sarchaeology.data.optical_multiyear import MultiYearOpticalDataset
from sarchaeology.models.temporal_swin import TemporalSwin


def embeddings(model, data, batch_size, device):
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0)
    outputs = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            image = batch['image'].to(device)
            valid = batch['valid_mask'].to(device)
            # Sample-dependent mask would require per-row generator calls; here
            # the same deterministic batch mask is shared before/after finetune.
            generator = torch.Generator(device='cpu')
            generator.manual_seed(20261121 + batch_index)
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                out = model.backbone(image,valid,generator=generator)
            outputs.append(out['embedding'].float().cpu())
    return F.normalize(torch.cat(outputs),dim=1)


def evaluate(emb, samples):
    identities = [str(x['patch_id']) for x in samples]
    quarters = [int(x['quarter']) for x in samples]
    records = []
    for q1 in range(1,5):
        queries = [i for i,q in enumerate(quarters) if q==q1]
        for q2 in range(1,5):
            if q1==q2: continue
            gallery = [j for j,q in enumerate(quarters) if q==q2]
            similarity = emb[queries] @ emb[gallery].T
            best = similarity.argmax(dim=1).tolist()
            matches = [identities[gallery[k]]==identities[i] for i,k in zip(queries,best)]
            if len(set(identities[j] for j in gallery)) != len(gallery):
                raise RuntimeError('Gallery contains duplicate patch IDs')
            records.append({'query_quarter':q1,'gallery_quarter':q2,
                'top1_correct':sum(matches),'query_count':len(matches),
                'top1_accuracy':sum(matches)/len(matches)})
    total_correct=sum(r['top1_correct'] for r in records)
    total_count=sum(r['query_count'] for r in records)
    return {'top1_accuracy':total_correct/total_count,'correct':total_correct,
            'queries':total_count,'chance_top1':1/20,'by_quarter_pair':records}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--config',type=Path,default=ROOT/'configs/training/optical_temporal_swin.yaml')
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/evaluation/temporal_swin_retrieval.json')
    args=p.parse_args()
    config=yaml.safe_load(args.config.read_text())
    stats=json.loads((ROOT/config['experiment']['statistics_path']).read_text())
    d=config['data']
    dataset=MultiYearOpticalDataset(ROOT,years=d['years'],split='val',statistics=stats,
        min_valid_fraction=d['min_valid_fraction'],min_source_observations=d['min_source_observations'])
    assert len(dataset)==80
    model_cfg=dict(config['model'])
    projection_dim=model_cfg.pop('projection_dim')
    hidden_dim=model_cfg.pop('projector_hidden_dim')
    model=TemporalSwin(model_cfg,projection_dim,hidden_dim)
    pretrained_path=ROOT/config['experiment']['pretrained_checkpoint']
    original=torch.load(pretrained_path,map_location='cpu',weights_only=True)
    model.backbone.load_state_dict(original['model_state'],strict=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=model.to(device).eval()
    before=evaluate(embeddings(model,dataset,args.batch_size,device),dataset.samples)
    temporal=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    model.load_state_dict(temporal['model_state'],strict=True)
    model.eval()
    after=evaluate(embeddings(model,dataset,args.batch_size,device),dataset.samples)
    result={'baseline':before,'temporal':after,'gain_percentage_points':100*(after['top1_accuracy']-before['top1_accuracy']),
            'interpretation':'Cross-quarter retrieval on 20 held-out locations; NOT archaeological detection validation.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('Baseline Top-1:',round(before['top1_accuracy']*100,2),'%')
    print('Temporal Top-1:',round(after['top1_accuracy']*100,2),'%')
    print('Chance Top-1: 5.0%')
    print('Saved:',args.output)

if __name__=='__main__': main()
