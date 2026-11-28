"""Phase 3.6.2 — frozen Swin + frozen SpectralMAE spatial-map gated adapter.

Run from the SARchaeology AI repository root:
 uv run python scripts/phase3_optical/train_hybrid_optical.py --smoke
 uv run python scripts/phase3_optical/train_hybrid_optical.py

Requires Phase 3.5.2 frozen Swin train/val caches and their original checkpoints.
Trains ONLY the adapter; validation is used for model selection, not final claims.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts/phase3_optical'))

from sarchaeology.data.optical_multiyear import MultiYearOpticalDataset
from sarchaeology.models.optical_swin_mim import OpticalSwinMIM
from sarchaeology.models.geographic_projection import GeographicProjector, contrastive_loss, relational_loss
from sarchaeology.models.hybrid_optical import SpatialSpectralGatedAdapter
from train_spectral_mae import SpectralMAE
from train_geographic_projection import (
    validate_splits, grouped, location_coordinates, distant_batches,
    make_pairs, retrieval, rank_stats, bootstrap,
)


def resolve(p):
    path=Path(p).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return torch.load(resolve(path), map_location='cpu', weights_only=True)


def frozen(model):
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def datasets_from_geo(geo_checkpoint):
    cfg=geo_checkpoint['config']
    d=cfg['data']; exp=cfg['experiment']
    path=resolve(exp['statistics_path'])
    stats=json.loads(path.read_text(encoding='utf-8'))
    if tuple(stats['bands'])!=('B02','B03','B04','B08','B11','B12') or stats.get('fit_split')!='train':
        raise RuntimeError('Wrong Sentinel-2 statistics or band order')
    opts=dict(years=d['years'], statistics=stats, min_valid_fraction=d['min_valid_fraction'],
              min_source_observations=d['min_source_observations'])
    ds={split:MultiYearOpticalDataset(ROOT, split=split, **opts) for split in ('train','val')}
    validate_splits(ds['train'],ds['val'])
    return ds, cfg, stats, path


def verify_geo_cache(geo_cp, geo_cfg, split, dataset, stats_path):
    fp=geo_cp['fingerprints']; exp=geo_cfg['experiment']
    expected={
        'pretrained_sha256':sha(resolve(exp['pretrained_checkpoint'])),
        'normalization_sha256':sha(stats_path),
        'val_manifest_sha256':sha(resolve('data/metadata/manifests/splits/buto_demo_2024/val.parquet')),
        'train_manifest_sha256':sha(resolve('data/metadata/manifests/splits/buto_demo_2024/train.parquet')),
    }
    for key, value in expected.items():
        if fp.get(key)!=value:
            raise RuntimeError(f'Geographic head fingerprint mismatch: {key}')
    cache=resolve(exp['cache_dir']) / f'{split}_features.pt'
    if not cache.exists():
        raise FileNotFoundError(f'Missing Phase 3.5.2 cached features: {cache}. Run Phase 3.5.2 first.')
    payload=load(cache)
    meta=payload['metadata']
    for key in ('pretrained_sha256','normalization_sha256','val_manifest_sha256','train_manifest_sha256'):
        if meta.get(key)!=fp.get(key):
            raise RuntimeError(f'Cached Swin features have mismatched {key}')
    expected_masks=geo_cfg['data']['validation_masks' if split=='val' else 'train_masks']
    if meta.get('split')!=split or meta.get('sample_count')!=len(dataset) or meta.get('masks')!=expected_masks:
        raise RuntimeError(f'Unexpected {split} Swin feature-cache metadata')
    z=payload['embeddings'].float()
    if z.shape!=(len(dataset),768):
        raise RuntimeError('Unexpected Swin feature shape')
    head=frozen(GeographicProjector(**geo_cfg['model']))
    head.load_state_dict(geo_cp['head_state'],strict=True)
    with torch.inference_mode():
        teacher=head(z).cpu()
    return teacher


class StageCapture:
    """Collect a frozen module output without editing its model implementation."""
    def __init__(self, module):
        self.value=None
        self.hook=module.register_forward_hook(self._capture)

    def _capture(self, module, inputs, result):
        self.value=result.detach()

    def close(self):
        self.hook.remove()


@torch.inference_mode()
def spatial_features(swin, spectral, data, cfg, split, device, smoke=False):
    """Cache 8x8 feature maps per observation in dataset order.

    Swin is evaluated with frozen deterministic random masking, SpectralMAE with
    the SAME three fixed two-band masks as Phase 3.6.1. No training gradients.
    """
    fmap_cfg=cfg['features']
    swin_masks=int(fmap_cfg['train_swin_masks'] if split=='train' else fmap_cfg['val_swin_masks'])
    if swin_masks<1: raise ValueError('At least one Swin mask required')
    batch_size=1 if split=='val' else int(fmap_cfg['feature_batch_size'])
    loader=DataLoader(data,batch_size=batch_size,shuffle=False,
                      num_workers=0 if split=='val' or smoke else int(fmap_cfg['num_workers']),
                      pin_memory=device.type=='cuda')
    swin_capture=StageCapture(swin.stages[2])  # [B,8,8,384] channels-last
    spec_capture=StageCapture(spectral.spatial) # [B,96,16,16]
    swin_chunks=[]; spec_chunks=[]
    try:
        for batch_i, batch in enumerate(loader):
            x=batch['image'].to(device,non_blocking=True)
            valid=batch['valid_mask'].to(device,non_blocking=True).bool()
            x=torch.where(valid.expand_as(x),x,torch.zeros_like(x))
            a=[];b=[]
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                for k in range(swin_masks):
                    gen=torch.Generator(device='cpu')
                    # Per-observation deterministic seed on validation: batch_size=1.
                    gen.manual_seed(20261121 + batch_i*1009+k*1_000_003)
                    swin(x,valid,generator=gen)
                    map_s=swin_capture.value
                    if map_s.shape[1:]==(8,8,384):
                        map_s=map_s.permute(0,3,1,2)
                    if map_s.shape[1:]!=(384,8,8):
                        raise RuntimeError(f'Unexpected stage-2 Swin map {map_s.shape}')
                    a.append(map_s.float())
                for pair in fmap_cfg['spectral_mask_pairs']:
                    mask=torch.zeros(len(x),6,dtype=torch.bool,device=device)
                    for band in pair:
                        mask[:, ('B02','B03','B04','B08','B11','B12').index(band)]=True
                    spectral(x,mask)
                    map_t=spec_capture.value
                    if map_t.shape[1:]!=(96,16,16):
                        raise RuntimeError(f'Unexpected spectral spatial map {map_t.shape}')
                    b.append(F.adaptive_avg_pool2d(map_t.float(),(8,8)))
            swin_chunks.append(torch.stack(a).mean(0).to('cpu',dtype=torch.float16))
            spec_chunks.append(torch.stack(b).mean(0).to('cpu',dtype=torch.float16))
            if (batch_i+1)%100==0:
                print(f'  {split}: extracted {(batch_i+1)*batch_size} samples',flush=True)
            if smoke:break
    finally:
        swin_capture.close();spec_capture.close()
    return {'swin_maps':torch.cat(swin_chunks),'spectral_maps':torch.cat(spec_chunks)}


def feature_cache(swin,spectral,ds,cfg,split,device,fp,rebuild):
    path=resolve(cfg['experiment']['cache_dir'])/f'{split}_maps.pt'
    metadata={**fp,'split':split,'count':len(ds),
              'feature_protocol':cfg['features']}
    if path.exists() and not rebuild:
        obj=load(path)
        if obj.get('metadata')==metadata:
            a=obj['swin_maps']; b=obj['spectral_maps']
            if a.shape==(len(ds),384,8,8) and b.shape==(len(ds),96,8,8):
                print(f'Using {split} cached local maps: {path}',flush=True)
                return {'swin_maps':a,'spectral_maps':b}
    print(f'Extracting {split} spatial-spectral maps...',flush=True)
    features=spatial_features(swin,spectral,ds,cfg,split,device)
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'metadata':metadata,**features},path)
    return features


def train_step(model,teacher,maps,indices,ids,cfg,*,use_spectral=True):
    a=maps['swin_maps'][indices]
    b=maps['spectral_maps'][indices]
    old=teacher[indices]
    result=model(old,a,b,use_spectral=use_spectral)
    z=result['embedding']
    loss_con=contrastive_loss(z,ids,temperature=cfg['training']['temperature'])
    # Penalize changes to the existing successful geographic embedding.
    loss_preserve=(1.-(z*F.normalize(old,dim=1)).sum(1)).mean()
    loss_rel=relational_loss(z,old,temperature=0.2)
    gate=result['gate_mean']
    params=cfg['training']
    loss=(params['contrastive_weight']*loss_con
          + params['preservation_weight']*loss_preserve
          + params['teacher_similarity_weight']*loss_rel
          + params['gate_weight']*gate)
    return loss, {'contrastive':float(loss_con.detach()),
                  'preservation':float(loss_preserve.detach()),
                  'relational':float(loss_rel.detach()),
                  'gate':float(gate.detach()),
                  'residual_scale':float(model.residual_scale.detach())}


@torch.inference_mode()
def project(model,teacher,maps,device,batch_size=32,*,use_spectral=True):
    model.eval(); outputs=[]
    for start in range(0,len(teacher),batch_size):
        sl=slice(start,start+batch_size)
        result=model(teacher[sl].to(device),
                     maps['swin_maps'][sl].to(device),
                     maps['spectral_maps'][sl].to(device),use_spectral=use_spectral)
        outputs.append(result['embedding'].cpu())
    return torch.cat(outputs)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/training/optical_hybrid_swin_spectral.yaml')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--rebuild-cache',action='store_true')
    p.add_argument('--cpu',action='store_true')
    p.add_argument('--no-spectral',action='store_true',help='Train a Swin-only adapter control with identical settings')
    p.add_argument('--epochs',type=int)
    args=p.parse_args()
    cfg=yaml.safe_load(resolve(args.config).read_text(encoding='utf-8'))
    if args.epochs is not None:cfg['training']['epochs']=args.epochs
    tr=cfg['training'];exp=cfg['experiment']
    if int(tr['epochs'])<1 or int(tr['locations_per_batch'])<2:
        p.error('Invalid epochs or location batch size')
    seed=int(tr['seed']); random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    device=torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    if device.type=='cuda':torch.set_float32_matmul_precision('high')
    geo_path=resolve(exp['geographic_checkpoint'])
    geo_cp=load(geo_path)
    ds,geo_cfg,stats,stats_path=datasets_from_geo(geo_cp)
    spectral_path=resolve(exp['spectral_checkpoint'])
    spec_cp=load(spectral_path)
    if spec_cp.get('masked_bands')!=2:raise RuntimeError('Expected two masked spectral bands')
    if tuple(stats['bands'])!=('B02','B03','B04','B08','B11','B12'):
        raise RuntimeError('Band mismatch')
    if args.smoke:
        # Exercise actual both frozen models + differentiable fusion on two IDs.
        ids=[str(s['patch_id']) for s in ds['train'].samples]
        selected=[]; seen=[]
        for pid in dict.fromkeys(ids):
            matched=[i for i,x in enumerate(ids) if x==pid][:2]
            if len(matched)==2:
                selected.extend(matched);seen.append(pid)
            if len(seen)==2:break
        if len(selected)!=4:raise RuntimeError('Need two locations with two observations each')
    pretrained_path=resolve(geo_cfg['experiment']['pretrained_checkpoint'])
    arch=yaml.safe_load(resolve(geo_cfg['experiment']['pretrained_config']).read_text('utf-8'))
    swin=frozen(OpticalSwinMIM(**arch['model']))
    swin.load_state_dict(load(pretrained_path)['model_state'],strict=True)
    spectral=frozen(SpectralMAE(**spec_cp['model_config']))
    spectral.load_state_dict(spec_cp['model_state'],strict=True)
    swin.to(device);spectral.to(device)
    model=SpatialSpectralGatedAdapter(**cfg['model']).to(device)
    if args.smoke:
        mini=Subset(ds['train'],selected)
        raw=spatial_features(swin,spectral,mini,cfg,'train',device,smoke=True)
        # smoke extraction returns one batch; 4 examples if configured batch>=4
        if len(raw['swin_maps'])!=4:
            raise RuntimeError('Smoke requires feature_batch_size >= 4')
        teachers=torch.randn(4,256,device=device)
        maps={k:v.to(device).float() for k,v in raw.items()}
        labels=torch.tensor([0,0,1,1],device=device)
        loss,terms=train_step(model,teachers,maps,torch.arange(4,device=device),labels,cfg,use_spectral=not args.no_spectral)
        loss.backward()
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite smoke loss')
        assert all(x.grad is None for x in swin.parameters())
        assert all(x.grad is None for x in spectral.parameters())
        print('SMOKE PASS: frozen encoders and trainable adapter, loss=',float(loss.detach()),flush=True)
        return
    # Check the original geographic teacher's cached embeddings and provenance.
    teacher={split:verify_geo_cache(geo_cp,geo_cfg,split,ds[split],stats_path)
             for split in ('train','val')}
    expected_fp={'swin_sha256':sha(pretrained_path),'spectral_sha256':sha(spectral_path),
                 'geo_sha256':sha(geo_path),'normalization_sha256':sha(stats_path),
                 'train_manifest_sha256':sha(resolve('data/metadata/manifests/splits/buto_demo_2024/train.parquet')),
                 'val_manifest_sha256':sha(resolve('data/metadata/manifests/splits/buto_demo_2024/val.parquet'))}
    maps={split:feature_cache(swin,spectral,ds[split],cfg,split,device,expected_fp,args.rebuild_cache)
          for split in ('train','val')}
    del swin,spectral
    if device.type=='cuda':torch.cuda.empty_cache()
    baseline,base_outcomes=retrieval(teacher['val'],ds['val'].samples)
    print('Frozen geographic teacher: Top-1',round(baseline['top1_accuracy']*100,2),
          'Top-5',round(baseline['top5_accuracy']*100,2),flush=True)
    if abs(baseline['top1_accuracy']-0.7541666666666667)>.025:
        raise RuntimeError('Teacher retrieval baseline differs from validated Phase 3.5.2; investigate cache/protocol')
    # Caches can live on CPU; transfer only the geographic pair batch to GPU.
    groups=grouped(ds['train'].samples)
    coords=location_coordinates(groups,ds['train'].samples)
    rng=np.random.default_rng(seed)
    optimizer=torch.optim.AdamW(model.parameters(),lr=float(tr['learning_rate']),weight_decay=float(tr['weight_decay']))
    run_id='hybrid_optical_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    outdir=resolve(exp['output_dir'])/run_id;ckdir=resolve(exp['checkpoint_dir'])/run_id
    outdir.mkdir(parents=True,exist_ok=True);ckdir.mkdir(parents=True,exist_ok=True)
    (outdir/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
    (outdir/'provenance.json').write_text(json.dumps(expected_fp,indent=2),encoding='utf-8')
    best=-1.;best_epoch=0;stale=0
    for epoch in range(1,int(tr['epochs'])+1):
        model.train(); losses=[]
        locations=np.array(sorted(groups));rng.shuffle(locations)
        for batch_ids in distant_batches(locations,coords,float(tr['min_negative_distance_pixels']),int(tr['locations_per_batch'])):
            positions,labels=make_pairs(groups,ds['train'].samples,batch_ids,rng,float(tr['same_quarter_probability']))
            idx = torch.tensor(positions, dtype=torch.long)

            inputs = {
                k: v[idx].to(device).float()
                for k, v in maps['train'].items()
            }
            target = teacher['train'][idx].to(device)

            # train_step receives an already-selected batch.
            # Therefore, use local batch indices rather than dataset indices.
            local_idx = torch.arange(len(idx), device=device)

            loss, terms = train_step(
                model,
                target,
                inputs,
                local_idx,
                labels.to(device),
                cfg,
                use_spectral=not args.no_spectral,
            )
            if not torch.isfinite(loss):raise RuntimeError('Non-finite hybrid loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),float(tr['grad_clip']))
            optimizer.step();losses.append(float(loss.detach()))
        if not losses:raise RuntimeError('No geographic batches produced; check coordinate distances')
        z=project(model,teacher['val'],maps['val'],device,use_spectral=not args.no_spectral)
        score,_=retrieval(z,ds['val'].samples)
        val=score['top1_accuracy']
        improved=val>best+1e-9
        if improved:best=val;best_epoch=epoch;stale=0
        else:stale+=1
        record={'epoch':epoch,'train_loss':float(np.mean(losses)),
                'val':score,'val_effective_rank':rank_stats(z),'best_top1':best,
                'residual_scale':float(model.residual_scale.detach())}
        with (outdir/'history.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
        if improved:
            torch.save({'model_state':model.state_dict(),'model_config':cfg['model'],
                        'run_id':run_id,'epoch':epoch,'val':score,'config':cfg,
                        'use_spectral':not args.no_spectral,
                        'provenance':expected_fp,'geo_checkpoint':str(geo_path),
                        'spectral_checkpoint':str(spectral_path)},ckdir/'best.pt')
        print(f'Epoch {epoch:3d} loss={np.mean(losses):.4f} top1={val:.2%} '
              f'top5={score["top5_accuracy"]:.2%} best={best:.2%}',flush=True)
        if stale>=int(tr['patience']):print('Early stopping');break
    saved=load(ckdir/'best.pt');model.load_state_dict(saved['model_state']);model.eval()
    z=project(model,teacher['val'],maps['val'],device,use_spectral=not args.no_spectral)
    final,hybrid_outcomes=retrieval(z,ds['val'].samples)
    report={'run_id':run_id,'best_epoch':best_epoch,'baseline':baseline,'hybrid':final,
            'baseline_rank':rank_stats(teacher['val']), 'hybrid_rank':rank_stats(z),
            'bootstrap':bootstrap(base_outcomes,hybrid_outcomes),
            'validation_selected':True,
            'caution':'20 validation locations and 240 dependent directed queries; held-out test untouched',
            'frozen_pretrained_encoders':True,
            'use_spectral':not args.no_spectral,
            'provenance':expected_fp}
    (outdir/'evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('BEST EPOCH',best_epoch,'TEACHER',baseline['top1_accuracy'],
          'HYBRID',final['top1_accuracy'],'RESULT',outdir/'evaluation.json',flush=True)


if __name__=='__main__':main()
