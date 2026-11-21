"""SARchaeology AI Phase 3.5.1: temporal Swin representation diagnostics.

Place at scripts/phase3_optical/diagnose_temporal_swin.py.
Uses the existing Phase 3.4/3.5 classes and the same 2024-only geographic holdout.
No training; does not modify checkpoints.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from sarchaeology.data.optical_multiyear import MultiYearOpticalDataset
from sarchaeology.models.temporal_swin import TemporalSwin


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_checkpoint(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except Exception as exc:
        raise RuntimeError(f'Unable to load checkpoint {path}: {exc}') from exc


def unit(x):
    return F.normalize(x.float(), dim=-1, eps=1e-12)


@torch.no_grad()
def extract_sample_stable(model, data, device, mask_seed, masks, amp):
    """One RNG stream per observation per mask, independent of DataLoader batch size.

    The existing OpticalSwinMIM.forward accepts one generator for a whole batch;
    batch=1 is necessary for stable per-observation masks without modifying it.
    """
    out_backbone, out_projector = [], []
    loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=0)
    model.eval()
    for index, batch in enumerate(loader):
        image = batch['image'].to(device, non_blocking=True)
        valid = batch['valid_mask'].to(device, non_blocking=True)
        b_views, p_views = [], []
        for mask_index in range(masks):
            generator = torch.Generator(device='cpu')
            generator.manual_seed(mask_seed + index * 1009 + mask_index * 1_000_003)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                representation = model(image, valid, generator=generator)
            b_views.append(unit(representation['embedding'].float().cpu()[0]))
            p_views.append(unit(representation['projection'].float().cpu()[0]))
        # Average independently normalized mask-view embeddings, then normalize.
        out_backbone.append(unit(torch.stack(b_views).mean(0)))
        out_projector.append(unit(torch.stack(p_views).mean(0)))
        if (index + 1) % 20 == 0:
            print(f'  processed {index+1}/{len(data)} observations', flush=True)
    return torch.stack(out_backbone), torch.stack(out_projector)


@torch.no_grad()
def extract_legacy(model, data, device, batch_size, amp):
    """Reproduce the original evaluator's batch-index-based mask assignments."""
    backbones = []
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    for index, batch in enumerate(loader):
        generator = torch.Generator(device='cpu')
        generator.manual_seed(20261121 + index)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            out = model.backbone(batch['image'].to(device), batch['valid_mask'].to(device), generator=generator)
        backbones.append(out['embedding'].float().cpu())
    return unit(torch.cat(backbones, dim=0))


def retrieval(vectors, samples):
    ids = [str(row['patch_id']) for row in samples]
    quarters = [int(row['quarter']) for row in samples]
    rows, outcomes = [], []
    top1 = top5 = 0
    for src in range(1, 5):
        qidx = [i for i, q in enumerate(quarters) if q == src]
        for dst in range(1, 5):
            if src == dst:
                continue
            gidx = [i for i, q in enumerate(quarters) if q == dst]
            if len({ids[j] for j in gidx}) != len(gidx):
                raise ValueError(f'Gallery for Q{dst} has duplicated patch IDs')
            if {ids[i] for i in qidx} != {ids[j] for j in gidx}:
                raise ValueError('Query/gallery IDs do not match')
            similarity = vectors[qidx] @ vectors[gidx].T
            k = min(5, len(gidx))
            nearest = similarity.topk(k, dim=1).indices
            right1 = right5 = 0
            for qi, row in enumerate(nearest.tolist()):
                correct = ids[qidx[qi]]
                found1 = ids[gidx[row[0]]] == correct
                found5 = any(ids[gidx[j]] == correct for j in row)
                right1 += int(found1)
                right5 += int(found5)
                outcomes.append({'patch_id':correct, 'source_quarter':src,
                                 'gallery_quarter':dst, 'top1':int(found1), 'top5':int(found5)})
            top1 += right1
            top5 += right5
            rows.append({'query_quarter':src,'gallery_quarter':dst,
                         'top1_correct':right1,'top5_correct':right5,
                         'queries':len(qidx), 'top1':right1/len(qidx),
                         'top5':right5/len(qidx)})
    count = len(outcomes)
    return {'top1_accuracy':top1/count,'top5_accuracy':top5/count,
            'top1_correct':top1,'top5_correct':top5,'queries':count,
            'chance_top1':1/len({str(s['patch_id']) for s in samples}),
            'chance_top5':min(5,len({str(s['patch_id']) for s in samples}))/len({str(s['patch_id']) for s in samples}),
            'quarter_pairs':rows}, outcomes


def representation_stats(vectors, samples):
    x = unit(vectors.double())
    centered = x - x.mean(dim=0)
    std = centered.std(dim=0, unbiased=True)
    # Gram eigenvalues are the nonzero eigenvalues of channel covariance.
    gram = centered @ centered.T / max(len(x)-1, 1)
    eigen = torch.linalg.eigvalsh(gram).clamp_min(0)
    total = float(eigen.sum().item())
    probs = eigen / max(total, 1e-30)
    probs = probs[probs > 1e-12]
    erank = float(torch.exp(-(probs * torch.log(probs)).sum()).item()) if len(probs) else 0.0
    participation = total**2 / max(float((eigen**2).sum().item()),1e-30)
    ids = [str(s['patch_id']) for s in samples]
    quarters = [int(s['quarter']) for s in samples]
    same_location, same_season_other_location, other = [], [], []
    cos = x @ x.T
    for i in range(len(x)):
        for j in range(i+1,len(x)):
            value=float(cos[i,j].item())
            if ids[i] == ids[j]:
                same_location.append(value)
            elif quarters[i] == quarters[j]:
                same_season_other_location.append(value)
            else:
                other.append(value)
    mean = lambda arr: sum(arr)/len(arr) if arr else None
    return {
        'dimension':x.shape[1], 'observations':len(x),
        'mean_coordinate_std':float(std.mean().item()),
        'median_coordinate_std':float(std.median().item()),
        'near_zero_std_dimensions_lt_1e-3':int((std<1e-3).sum().item()),
        'effective_rank':erank, 'participation_ratio':participation,
        'max_centered_rank':min(len(x)-1,x.shape[1]),
        'same_location_different_quarter_mean_cosine':mean(same_location),
        'different_location_same_quarter_mean_cosine':mean(same_season_other_location),
        'different_location_different_quarter_mean_cosine':mean(other),
        'same_location_minus_same_quarter_other_location':mean(same_location)-mean(same_season_other_location),
    }


def paired_bootstrap(outcome_a, outcome_b, samples, seed=42, rounds=2000):
    """Bootstrap by geographic patch (rather than 240 correlated query decisions)."""
    assert len(outcome_a)==len(outcome_b)
    ids=sorted({str(s['patch_id']) for s in samples})
    a,b={},{}
    for name, rows, dest in [('a',outcome_a,a),('b',outcome_b,b)]:
        group=defaultdict(list)
        for row in rows: group[row['patch_id']].append(int(row['top1']))
        assert set(group)==set(ids), (name,'missing geographical IDs')
        for key in ids: dest[key]=sum(group[key])/len(group[key])
    import random
    rng=random.Random(seed)
    differences=[]
    for _ in range(rounds):
        choice=[ids[rng.randrange(len(ids))] for _ in ids]
        differences.append(sum((b[k]-a[k]) for k in choice)/len(choice))
    differences.sort()
    return {'metric':'temporal_minus_pretrained_top1',
            'difference':sum((b[k]-a[k]) for k in ids)/len(ids),
            'bootstrap_unit':'geographic_patch_id', 'replicates':rounds,
            'percentile_95_interval':[differences[int(.025*rounds)],differences[min(rounds-1,int(.975*rounds))]]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--config',type=Path,default=ROOT/'configs/training/optical_temporal_swin.yaml')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/evaluation/temporal_swin_diagnostics.json')
    p.add_argument('--mask-seeds',type=int,default=3,help='Independent deterministic masks per observation (default: 3)')
    p.add_argument('--seed',type=int,default=20261121)
    p.add_argument('--legacy-batch-size',type=int,default=16,help='Original evaluator batch size, used for an optional reproduction')
    p.add_argument('--skip-legacy',action='store_true',help='Skip reproduction of earlier 47.92/32.08%% benchmark')
    p.add_argument('--cpu',action='store_true')
    args=p.parse_args()
    if args.mask_seeds < 1 or args.legacy_batch_size < 1:
        p.error('mask-seeds and legacy-batch-size must be positive')
    config=yaml.safe_load(resolve_path(args.config).read_text(encoding='utf-8'))
    stats=json.loads(resolve_path(config['experiment']['statistics_path']).read_text(encoding='utf-8'))
    d=config['data']
    if stats.get('fit_split') != 'train':
        raise RuntimeError('Expected train-fitted normalization statistics')
    data=MultiYearOpticalDataset(ROOT,years=d['years'],split='val',statistics=stats,
                                 min_valid_fraction=d['min_valid_fraction'],
                                 min_source_observations=d['min_source_observations'])
    if len(data)!=80 or len({str(s['patch_id']) for s in data.samples})!=20 or {int(s['year']) for s in data.samples}!={2024}:
        raise RuntimeError('Expected 80 validation images, 20 held-out locations, year=2024')
    model_cfg=dict(config['model'])
    projection_dim=model_cfg.pop('projection_dim')
    hidden_dim=model_cfg.pop('projector_hidden_dim')
    model=TemporalSwin(model_cfg, projection_dim, hidden_dim)
    source=resolve_path(config['experiment']['pretrained_checkpoint'])
    pretrained=load_checkpoint(source)
    fine_tuned=load_checkpoint(resolve_path(args.checkpoint))
    model.backbone.load_state_dict(pretrained['model_state'],strict=True)
    device=torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    model.to(device).eval()
    amp=device.type=='cuda'
    print('Device:',device,'Validation observations:',len(data),'Mask seeds:',args.mask_seeds,flush=True)
    print('Extracting pretrained backbone representations...',flush=True)
    pretrained_embedding,_=extract_sample_stable(model,data,device,args.seed,args.mask_seeds,amp)
    legacy={}
    if not args.skip_legacy:
        legacy['original_pretrained'],_ = retrieval(extract_legacy(model,data,device,args.legacy_batch_size,amp),data.samples)
    model.load_state_dict(fine_tuned['model_state'],strict=True)
    model.eval()
    print('Extracting fine-tuned backbone and trained projector...',flush=True)
    temporal_embedding,projection_embedding=extract_sample_stable(model,data,device,args.seed,args.mask_seeds,amp)
    if not args.skip_legacy:
        legacy['original_fine_tuned'],_=retrieval(extract_legacy(model,data,device,args.legacy_batch_size,amp),data.samples)
    vectors={'pretrained_backbone':pretrained_embedding,
             'fine_tuned_backbone':temporal_embedding,
             'fine_tuned_projector':projection_embedding}
    result={'method':{
        'validation_observations':len(data),'heldout_locations':20,'validation_year':2024,
        'mask_seed_base':args.seed,'masks_per_observation':args.mask_seeds,
        'masking':'per observation / repeat deterministic, batch-independent',
        'projection_head':'trained head; BatchNorm in eval mode',
        'embedding_aggregation':'L2-normalize each mask embedding, average, then L2-normalize',
        'notes':['This evaluation remains at 75% input masking, not unmasked inference.',
                 'Validation contains one year only; same-quarter cross-year retrieval cannot be evaluated here.',
                 'Effective rank is bounded by 79 due to only 80 examples.',
                 'Projector cosine scores and backbone cosine scores have different dimensionality; compare ranking not raw scale.',
                 'No archaeological-site labels are evaluated.']},
            'checkpoints':{'pretrained':str(source),'fine_tuned':str(resolve_path(args.checkpoint))},
            'representations':{},'legacy_reproduction':legacy}
    outcomes={}
    for name,v in vectors.items():
        ranking,record=retrieval(v,data.samples)
        outcomes[name]=record
        result['representations'][name]={'retrieval':ranking,'geometry':representation_stats(v,data.samples)}
        print(f'{name:23s} Top-1={ranking["top1_accuracy"]*100:6.2f}% '
              f'Top-5={ranking["top5_accuracy"]*100:6.2f}% '
              f'effective_rank={result["representations"][name]["geometry"]["effective_rank"]:.1f}',flush=True)
    result['paired_bootstrap']=paired_bootstrap(outcomes['pretrained_backbone'],outcomes['fine_tuned_backbone'],data.samples)
    target=resolve_path(args.output)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    if legacy:
        a=legacy['original_pretrained']['top1_accuracy']*100
        b=legacy['original_fine_tuned']['top1_accuracy']*100
        print(f'Legacy reproduction (batch size {args.legacy_batch_size}): {a:.2f}% -> {b:.2f}%')
    print('Saved:',target)


if __name__=='__main__':
    main()
