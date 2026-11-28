"""Phase 3.7: independent, geographically held-out optical encoder audit.

This script does not train or select any model. It uses 2024 TEST geographies
only, with train-fitted normalization and immutable checkpoints.

Full-input Swin is evaluated by running its existing encoder layers directly;
this is an inference mode not seen during MIM training. Masked mode reproduces
Phase 3.5.2's stable three-view feature protocol on the test split.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts/phase3_optical'))
from sarchaeology.data.optical_multiyear import MultiYearOpticalDataset
from sarchaeology.models.optical_swin_mim import OpticalSwinMIM
from sarchaeology.models.geographic_projection import GeographicProjector
from sarchaeology.models.hybrid_optical import SpatialSpectralGatedAdapter
from train_spectral_mae import SpectralMAE
from train_hybrid_optical import StageCapture

BANDS = ('B02', 'B03', 'B04', 'B08', 'B11', 'B12')


def resolve(path):
    p=Path(path).expanduser()
    return (p if p.is_absolute() else ROOT/p).resolve()


def digest(path):
    with open(path,'rb') as f:
        h=hashlib.sha256()
        for part in iter(lambda:f.read(1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def checkpoint(path):
    return torch.load(resolve(path),map_location='cpu',weights_only=True)


def frozen(m):
    m.eval().requires_grad_(False)
    return m


def ids_by_split(root, path):
    manifest=pd.read_parquet(root / path)
    if 'patch_id' not in manifest:
        raise RuntimeError(f'Missing patch_id in {path}')
    return set(manifest['patch_id'].astype(str))


def validate_test(dataset, expected_test_ids, train_ids, val_ids):
    if train_ids & val_ids or train_ids & expected_test_ids or val_ids & expected_test_ids:
        raise RuntimeError('Manifest geographic identity leakage')
    ids={str(s['patch_id']) for s in dataset.samples}
    if not ids:raise RuntimeError('No usable 2024 test observations')
    if not ids <= expected_test_ids:
        raise RuntimeError('Test dataset includes identities absent from the test manifest')
    if ids & (train_ids | val_ids):
        raise RuntimeError('Test data contains train/validation identities')
    if {int(s['year']) for s in dataset.samples}!={2024}:
        raise RuntimeError('Expected 2024 test observations only')
    groups=defaultdict(set)
    for s in dataset.samples:
        groups[str(s['patch_id'])].add(int(s['quarter']))
    incomplete={k: sorted(v) for k,v in groups.items() if v!={1,2,3,4}}
    if incomplete:
        raise RuntimeError(f'Incomplete test quarterly coverage for {len(incomplete)} test locations: {list(incomplete.items())[:3]}')
    if len(dataset.samples) != 4*len(ids):
        raise RuntimeError('Duplicate test location/year/quarter observations')
    return {'test_locations':len(ids),'test_observations':len(dataset.samples),
            'manifest_test_ids':len(expected_test_ids)}


def retrieval(embeddings, samples):
    """Four-quarter, directed retrieval; gallery contains all usable test locations.

    Correct outcomes are retained BY patch ID for a paired, clustered bootstrap.
    """
    z=F.normalize(embeddings.float().cpu(), dim=1)
    patch=[str(s['patch_id']) for s in samples]
    q=[int(s['quarter']) for s in samples]
    result=[]; per=[]
    ids=set(patch)
    for source_q in range(1,5):
        query=[i for i,x in enumerate(q) if x==source_q]
        for gallery_q in range(1,5):
            if source_q==gallery_q:continue
            gallery=[i for i,x in enumerate(q) if x==gallery_q]
            if {patch[i] for i in query}!={patch[i] for i in gallery} or len(gallery)!=len(ids):
                raise RuntimeError('Quarter retrieval has incompatible galleries')
            sims=z[query]@z[gallery].T
            neighbors=sims.topk(min(5,len(gallery)),dim=1).indices.tolist()
            hits=[]
            for i, row in enumerate(neighbors):
                correct_id=patch[query[i]]
                hit1=int(patch[gallery[row[0]]]==correct_id)
                hit5=int(any(patch[gallery[k]]==correct_id for k in row))
                result.append({'patch_id':correct_id,'source_quarter':source_q,
                               'gallery_quarter':gallery_q,'top1':hit1,'top5':hit5})
                hits.append((hit1,hit5))
            per.append({'source_quarter':source_q,'gallery_quarter':gallery_q,
                        'top1':float(np.mean([x[0] for x in hits])),
                        'top5':float(np.mean([x[1] for x in hits]))})
    return {'top1_accuracy':float(np.mean([x['top1'] for x in result])),
            'top5_accuracy':float(np.mean([x['top5'] for x in result])),
            'correct_top1':sum(x['top1'] for x in result),
            'correct_top5':sum(x['top5'] for x in result),
            'queries':len(result),'gallery_size':len(ids),
            'chance_top1':1/len(ids),'chance_top5':min(5,len(ids))/len(ids),
            'quarter_pairs':per},result


def paired_bootstrap(control, candidate, *, repeats=2000, seed=42):
    a={(x['patch_id'],x['source_quarter'],x['gallery_quarter']):x for x in control}
    b={(x['patch_id'],x['source_quarter'],x['gallery_quarter']):x for x in candidate}
    if set(a)!=set(b):raise RuntimeError('Retrieval outcomes cannot be paired')
    groups=defaultdict(list)
    for key,x in a.items():groups[key[0]].append(b[key]['top1']-x['top1'])
    means=np.array([np.mean(groups[k]) for k in sorted(groups)], dtype=float)
    rng=np.random.default_rng(seed)
    sampled=rng.integers(0,len(means),(repeats,len(means)))
    dist=means[sampled].mean(axis=1)
    return {'difference':float(means.mean()),'95pct_ci':np.quantile(dist,[.025,.975]).tolist(),
            'bootstrap_clusters':len(means),'replicates':repeats,'cluster':'geographic_patch_id'}


def rank_stats(z):
    x=F.normalize(z.double().cpu(),dim=-1)
    x=x-x.mean(0)
    g=x@x.T/max(1,len(x)-1)
    e=torch.linalg.eigvalsh(g).clamp_min(0)
    s=float(e.sum())
    if s<1e-15:return {'effective_rank':0.,'participation_ratio':0.}
    p=e/e.sum();p=p[p>1e-12]
    return {'effective_rank':float(torch.exp(-(p*p.log()).sum())),
            'participation_ratio':float(e.sum().square()/e.square().sum().clamp_min(1e-20)),
            'max_rank':min(len(x)-1,x.shape[1])}


@torch.inference_mode()
def swin_full(model, x, *, capture_stage2=False):
    """No mask, exact pretrained encoder modules; no decoder and no retraining."""
    h=model.patch_embed(x).permute(0,2,3,1)
    stage2=None
    for index, stage in enumerate(model.stages):
        h=stage(h)
        if index==2 and capture_stage2:
            stage2=h.permute(0,3,1,2).contiguous()
        if index<3:h=model.merges[index](h)
    pooled=model.encoder_norm(h).mean(dim=(1,2))
    return {'embedding':pooled, 'stage2':stage2}


@torch.inference_mode()
def extract_test(ds, swin, geo, spectral, device, *, n_masks, batch_size,
                 include_adapters, spectral_pairs, full_input):
    """One observation per pass to make masked RNG independent of batch size."""
    loader=DataLoader(ds,batch_size=1,shuffle=False,num_workers=0)
    output={'swin_masked':[],'teacher_masked':[]}
    if full_input:output.update(swin_full=[], teacher_full=[])
    if include_adapters:output.update(swin_maps=[], spectral_maps=[])
    sw_hook=StageCapture(swin.stages[2]) if include_adapters else None
    sp_hook=StageCapture(spectral.spatial) if include_adapters else None
    try:
        for i, batch in enumerate(loader):
            x=batch['image'].to(device)
            valid=batch['valid_mask'].to(device).bool()
            x=torch.where(valid.expand_as(x),x,torch.zeros_like(x))
            masked=[]; local=[]
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                for k in range(n_masks):
                    rng=torch.Generator(device='cpu')
                    rng.manual_seed(20261121+i*1009+k*1_000_003)
                    r=swin(x,valid,generator=rng)
                    masked.append(F.normalize(r['embedding'].float(),dim=1))
                    if include_adapters:
                        m=sw_hook.value
                        if m.shape[1:]==(8,8,384):m=m.permute(0,3,1,2)
                        if tuple(m.shape[1:])!=(384,8,8):
                            raise RuntimeError(f'Unexpected Swin local map {m.shape}')
                        local.append(m.float())
                avg=F.normalize(torch.stack(masked).mean(0),dim=1)
                output['swin_masked'].append(avg.cpu())
                output['teacher_masked'].append(geo(avg.float()).cpu())
                if full_input:
                    full=swin_full(swin,x)
                    f=F.normalize(full['embedding'].float(),dim=1)
                    output['swin_full'].append(f.cpu())
                    output['teacher_full'].append(geo(f).cpu())
                if include_adapters:
                    output['swin_maps'].append(torch.stack(local).mean(0).cpu())
                    spec=[]
                    for pair in spectral_pairs:
                        mask=torch.zeros(1,6,dtype=torch.bool,device=device)
                        for band in pair:mask[:,BANDS.index(band)]=True
                        spectral(x,mask)
                        sm=sp_hook.value
                        if tuple(sm.shape[1:])!=(96,16,16):
                            raise RuntimeError(f'Unexpected spectral map {sm.shape}')
                        spec.append(F.adaptive_avg_pool2d(sm.float(),(8,8)))
                    output['spectral_maps'].append(torch.stack(spec).mean(0).cpu())
            if (i+1)%20==0:print(f'  processed {i+1}/{len(ds)} test observations',flush=True)
    finally:
        if sw_hook:sw_hook.close()
        if sp_hook:sp_hook.close()
    return {k:torch.cat(v,0) for k,v in output.items()}


@torch.inference_mode()
def apply_adapter(model, teacher, swin_maps, spectral_maps, device, *, use_spectral, batch_size=16):
    out=[]
    for i in range(0,len(teacher),batch_size):
        sl=slice(i,i+batch_size)
        z=model(teacher[sl].to(device),swin_maps[sl].to(device),
                spectral_maps[sl].to(device),use_spectral=use_spectral)['embedding']
        out.append(z.cpu())
    return torch.cat(out)


def load_model_stack(cfg, device, *, with_adapters):
    geo_path=resolve(cfg['checkpoints']['geographic'])
    geo_cp=checkpoint(geo_path)
    geo_cfg=geo_cp['config']
    exp=geo_cfg['experiment']
    swin_path=resolve(exp['pretrained_checkpoint'])
    swin_arch=yaml.safe_load(resolve(exp['pretrained_config']).read_text(encoding='utf-8'))
    swin=frozen(OpticalSwinMIM(**swin_arch['model']))
    swin.load_state_dict(checkpoint(swin_path)['model_state'],strict=True)
    geo=frozen(GeographicProjector(**geo_cfg['model']))
    geo.load_state_dict(geo_cp['head_state'],strict=True)
    spec=None; spec_cp=None; candidates={}
    fingerprints={'geographic_checkpoint':digest(geo_path),
                  'swin_checkpoint':digest(swin_path),
                  'statistics_file':digest(resolve(exp['statistics_path']))}
    if with_adapters:
        spec_path=resolve(cfg['checkpoints']['spectral'])
        spec_cp=checkpoint(spec_path)
        spec=frozen(SpectralMAE(**spec_cp['model_config']))
        spec.load_state_dict(spec_cp['model_state'],strict=True)
        fingerprints['spectral_checkpoint']=digest(spec_path)
        for label,key,expect_spectral in (
            ('hybrid','hybrid',True),('swin_only_adapter','swin_only_adapter',False)):
            cp_path=resolve(cfg['checkpoints'][key]);cp=checkpoint(cp_path)
            if bool(cp.get('use_spectral')) != expect_spectral:
                raise RuntimeError(f'Incorrect use_spectral flag for {label}')
            if cp.get('provenance',{}).get('geo_sha256') != fingerprints['geographic_checkpoint']:
                raise RuntimeError(f'{label} geographic checkpoint fingerprint mismatch')
            if cp.get('provenance',{}).get('swin_sha256') != fingerprints['swin_checkpoint']:
                raise RuntimeError(f'{label} Swin checkpoint fingerprint mismatch')
            if cp.get('provenance',{}).get('spectral_sha256') != fingerprints['spectral_checkpoint']:
                raise RuntimeError(f'{label} spectral checkpoint fingerprint mismatch')
            adapter=frozen(SpatialSpectralGatedAdapter(**cp['model_config']))
            adapter.load_state_dict(cp['model_state'],strict=True)
            candidates[label]=adapter.to(device)
            fingerprints[f'{label}_checkpoint']=digest(cp_path)
    return swin.to(device),geo.to(device),spec.to(device) if spec is not None else None,candidates,geo_cfg,fingerprints


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/evaluation/optical_phase37.yaml')
    parser.add_argument('--cpu',action='store_true')
    parser.add_argument('--teacher-only',action='store_true',help='Skip both adapters when their checkpoints are unavailable')
    parser.add_argument('--skip-full',action='store_true',help='Skip unmasked encoder inference ablation')
    parser.add_argument('--batch-size',type=int,default=16,help='Adapter projection batch size; extraction is per-observation')
    args=parser.parse_args()
    if args.batch_size<1:parser.error('batch size must be >=1')
    cfg=yaml.safe_load(resolve(args.config).read_text(encoding='utf-8'))
    device=torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    swin,geo,spectral,adapters,geo_cfg,fingerprints=load_model_stack(cfg,device,with_adapters=not args.teacher_only)
    exp=geo_cfg['experiment']; d=geo_cfg['data']
    norm=json.loads(resolve(exp['statistics_path']).read_text('utf-8'))
    if tuple(norm['bands'])!=BANDS or norm.get('fit_split')!='train':
        raise RuntimeError('Normalization must be train-fitted with exact Sentinel-2 band order')
    test=MultiYearOpticalDataset(ROOT,years=[2024],split='test',statistics=norm,
        min_valid_fraction=d['min_valid_fraction'],min_source_observations=d['min_source_observations'])
    prefix=Path('data/metadata/manifests/splits/buto_demo_2024')
    split_ids={name:ids_by_split(ROOT,prefix/f'{name}.parquet') for name in ('train','val','test')}
    coverage=validate_test(test,split_ids['test'],split_ids['train'],split_ids['val'])
    mask_count=int(cfg['protocol']['masked_views']);
    if mask_count!=3:raise RuntimeError('To compare with Phase 3.5.2, use exactly 3 masked views')
    pairs=cfg['protocol']['spectral_pairs']
    if pairs!=[['B11','B12'],['B04','B08'],['B02','B03']]:
        raise RuntimeError('Spectral pairs must match 3.6.2 training')
    print('Independent optical test | ',coverage,'| device',device,flush=True)
    feats=extract_test(test,swin,geo,spectral,device,n_masks=mask_count,
                batch_size=args.batch_size,include_adapters=not args.teacher_only,
                spectral_pairs=pairs,full_input=not args.skip_full)
    vectors={
        'swin_masked':feats['swin_masked'],
        'geographic_teacher_masked':feats['teacher_masked']
    }
    if not args.skip_full:
        vectors['swin_unmasked']=feats['swin_full']
        vectors['geographic_teacher_unmasked']=feats['teacher_full']
    for key,adapter in adapters.items():
        vectors[key]=apply_adapter(adapter,feats['teacher_masked'],feats['swin_maps'],
                                  feats['spectral_maps'],device,
                                  use_spectral=key=='hybrid',batch_size=args.batch_size)
    metrics={};outcomes={}
    for name,vector in vectors.items():
        metrics[name],outcomes[name]=retrieval(vector,test.samples)
        metrics[name]['rank']=rank_stats(vector)
        print(f'  {name:30s} Top-1 {metrics[name]["top1_accuracy"]:.2%} '
              f'Top-5 {metrics[name]["top5_accuracy"]:.2%}',flush=True)
    primary='geographic_teacher_masked'
    comparisons={k:paired_bootstrap(outcomes[primary],v) for k,v in outcomes.items() if k!=primary}
    # Per-observation local-map spatial variance is only a diagnostic, not archaeology accuracy.
    local={}
    if not args.teacher_only:
        for key in ('swin_maps','spectral_maps'):
            v=feats[key].float()
            local[key]={'mean_spatial_std':float(v.flatten(2).std(-1).mean()),
                        'fraction_nearly_spatial_constant':float((v.flatten(2).std(-1)<1e-5).float().mean()),
                        'shape':list(v.shape)}
    report={'phase':'3.7','run_utc':datetime.now(timezone.utc).isoformat(),
            'dataset':'buto_demo_2024', 'evaluation_split':'test',
            'test_coverage':coverage,'feature_mode':'three deterministic 75% spatial masks',
            'full_image_mode':'unmasked pretrained encoder (OOD relative to masked pretraining)',
            'validation_used_in_test_training':False,
            'warnings':['All models were previously chosen using validation. Test is for a one-time audit, not tuning.',
                        'The 12 quarter directions share geographic identities; bootstrap is clustered by patch ID.',
                        'Retrieval is a proxy task and is not evidence of archaeological detection.',
                        'Swin spatial features are 8x8 and require a finer-resolution sensitivity study before anomaly mapping.'],
            'checkpoints_sha256':fingerprints,'metrics':metrics,
            'paired_bootstrap_vs_geographic_teacher':comparisons,'spatial_diagnostics':local}
    destination=resolve(cfg['output']['report_path']);destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite a previous test report: {destination}')
    destination.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('TEST REPORT',destination,flush=True)


if __name__=='__main__':main()
