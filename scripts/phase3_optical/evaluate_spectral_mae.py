"""Phase 3.6.1 — SpectralMAE band-masking evaluation and Swin fusion.

Requires the Phase 3.6 trainer's SpectralMAE class and the Phase 3.5.2
GeographicProjector class. Does NOT change checkpoints or train on validation.

Usage:
 uv run python scripts/phase3_optical/evaluate_spectral_mae.py \
   --spectral-checkpoint checkpoints/optical_spectral_mae/spectral_mae_20261122T125708Z/best.pt \
   --geo-checkpoint checkpoints/optical_geographic_projection/geo_projection_20261122T121336Z_a3fd36a6/best.pt
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts/phase3_optical'))
from sarchaeology.data.optical_multiyear import MultiYearOpticalDataset, BANDS
from sarchaeology.models.geographic_projection import GeographicProjector
from train_spectral_mae import SpectralMAE

BANDS=tuple(BANDS)
COMBINATIONS = (("B11", "B12"), ("B04", "B08"), ("B02", "B03"))

def resolve(p):
    p=Path(p).expanduser()
    return (p if p.is_absolute() else ROOT/p).resolve()

def sha256(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def spectral_mask(names, batch, device):
    mask=torch.zeros(batch, len(BANDS), dtype=torch.bool, device=device)
    for name in names: mask[:, BANDS.index(name)] = True
    return mask

def retrieval(vectors, samples):
    """240 directed queries across the 12 ordered pairs of quarters."""
    z=F.normalize(torch.as_tensor(vectors, dtype=torch.float32),dim=1)
    ids=[str(s['patch_id']) for s in samples]
    qs=[int(s['quarter']) for s in samples]
    records=[]; total1=total5=queries=0
    for q1 in (1,2,3,4):
        ii=[i for i,q in enumerate(qs) if q==q1]
        for q2 in (1,2,3,4):
            if q1==q2: continue
            jj=[j for j,q in enumerate(qs) if q==q2]
            if len(ii)!=20 or len(jj)!=20 or len(set(ids[j] for j in jj))!=20:
                raise RuntimeError('Expected 20 unique locations in every quarterly gallery')
            similarities=z[ii]@z[jj].T
            rankings=similarities.argsort(dim=1,descending=True)
            c1=c5=0
            for i, ranked in zip(ii, rankings):
                ranked_ids=[ids[jj[int(v)]] for v in ranked[:5]]
                c1+=ids[i]==ranked_ids[0]
                c5+=ids[i] in ranked_ids
            n=len(ii); queries+=n;total1+=c1;total5+=c5
            records.append({'query_quarter':q1,'gallery_quarter':q2,'top1':c1/n,'top5':c5/n})
    return {'top1_accuracy':total1/queries, 'top5_accuracy':total5/queries,
            'correct_top1':total1,'correct_top5':total5,'queries':queries,
            'quarter_pairs':records,'chance_top1':1/20,'chance_top5':5/20}

def effective_rank(vectors):
    v=np.asarray(vectors,dtype=np.float64)
    v=v-v.mean(0,keepdims=True)
    spectrum=np.linalg.svd(v,full_matrices=False,compute_uv=False)**2
    spectrum=spectrum[spectrum>1e-18]
    if not len(spectrum): return {'effective_rank':0.,'participation_ratio':0.}
    p=spectrum/spectrum.sum()
    return {'effective_rank':float(np.exp(-(p*np.log(p+1e-40)).sum())),
            'participation_ratio':float(spectrum.sum()**2/(spectrum*spectrum).sum())}

def per_band_accumulator():
    return {band:{'absolute_sum':0.,'square_sum':0.,'count':0} for band in BANDS}

def finalize_acc(acc):
    output={}
    for band,a in acc.items():
        n=a['count']
        output[band]={'mae':a['absolute_sum']/n if n else None,
                      'rmse':(a['square_sum']/n)**0.5 if n else None,
                      'scored_pixels':int(n)}
    vals=[acc[b] for b in BANDS]
    n=sum(v['count'] for v in vals)
    return {'per_band':output,'overall':{'mae':sum(v['absolute_sum'] for v in vals)/n if n else None,
                                      'rmse':(sum(v['square_sum'] for v in vals)/n)**0.5 if n else None,
                                      'scored_pixels':int(n)}}

def update_acc(acc, diff, valid, names):
    # Physical reflectance units. Inputs [B,6,H,W], valid [B,1,H,W].
    for band in names:
        i=BANDS.index(band)
        selected=diff[:,i][valid[:,0]]
        if selected.numel()==0: continue
        a=acc[band]
        a['count']+=int(selected.numel())
        a['absolute_sum']+=float(selected.abs().sum())
        a['square_sum']+=float(selected.square().sum())

def ridge_fit(x,y,alpha):
    """Ridge with unpenalized bias, using training spectra only."""
    x=np.asarray(x,np.float64);y=np.asarray(y,np.float64)
    if x.shape[0]!=y.shape[0]: raise ValueError('Incompatible ridge shapes')
    design=np.concatenate([x,np.ones((len(x),1))],axis=1)
    reg=np.eye(design.shape[1]);reg[-1,-1]=0
    return np.linalg.solve(design.T@design+alpha*reg,design.T@y)

def ridge_samples(dataset, names, samples_limit, per_observation, seed):
    """Train-only pixel sampler. Each row predicts exactly the target masked bands."""
    rng=np.random.default_rng(seed)
    selected=np.linspace(0,len(dataset)-1,min(samples_limit,len(dataset)),dtype=int)
    xs=[];ys=[]
    inputs=[i for i,b in enumerate(BANDS) if b not in names]
    outputs=[BANDS.index(b) for b in names]
    for idx in selected:
        row=dataset[int(idx)]
        im=np.asarray(row['image'],np.float32).reshape(len(BANDS),-1)
        valid=np.asarray(row['valid_mask'],bool).reshape(-1)
        positions=np.flatnonzero(valid)
        if not len(positions): continue
        chosen=rng.choice(positions,size=min(len(positions),per_observation),replace=False)
        xs.append(im[inputs][:,chosen].T)
        ys.append(im[outputs][:,chosen].T)
    if not xs:raise RuntimeError('No valid training pixels for ridge')
    return np.concatenate(xs,axis=0),np.concatenate(ys,axis=0),inputs,outputs

def evaluate_spectral(model,val,raw,stats,device,batch_size,workers,combos,preview_dir=None):
    loader=DataLoader(val,batch_size=batch_size,shuffle=False,num_workers=workers)
    med=torch.tensor(stats['median'],dtype=torch.float32,device=device).view(1,6,1,1)
    scale=torch.tensor(stats['scale'],dtype=torch.float32,device=device).view(1,6,1,1)
    acc={'+'.join(c):per_band_accumulator() for c in combos}
    # One set of spectral embeddings for each fixed band pair; combined for retrieval.
    vectors={'+'.join(c):[] for c in combos}
    model.eval()
    with torch.inference_mode():
        for bi,batch in enumerate(loader):
            im=batch['image'].to(device,non_blocking=True)
            valid=batch['valid_mask'].to(device,non_blocking=True).bool()
            # Explicitly zero missing input pixels; the model also replaces masked band tokens.
            safe=torch.where(valid.expand_as(im),im,torch.zeros_like(im))
            true=safe.float()*scale+med
            for combo in combos:
                key='+'.join(combo)
                mask=spectral_mask(combo,len(im),device)
                with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                    out=model(safe,mask)
                reconstructed=out['reconstruction'].float()*scale+med
                update_acc(acc[key],reconstructed-true,valid,combo)
                vectors[key].append(F.normalize(out['embedding'].float().cpu(),dim=1))
    embeddings={key:F.normalize(torch.cat(chunks),dim=1) for key,chunks in vectors.items()}
    aggregated=F.normalize(torch.stack(list(embeddings.values())).mean(0),dim=1)
    metrics={key:finalize_acc(a) for key,a in acc.items()}
    return metrics,embeddings,aggregated

def ridge_evaluate(val, stats, combo, coefs, inputs, outputs, batch_size):
    acc=per_band_accumulator()
    loader=DataLoader(val,batch_size=batch_size,shuffle=False,num_workers=0)
    med=np.asarray(stats['median'],np.float32).reshape(1,6,1,1)
    scale=np.asarray(stats['scale'],np.float32).reshape(1,6,1,1)
    for batch in loader:
        im=batch['image'].cpu().numpy()
        valid=batch['valid_mask'].cpu().numpy().astype(bool)
        b,c,h,w=im.shape
        x=im[:,inputs].transpose(0,2,3,1).reshape(-1,len(inputs)).astype(np.float64)
        y=np.c_[x,np.ones(len(x))]@coefs
        y=y.reshape(b,h,w,len(outputs)).transpose(0,3,1,2)
        target=(im*scale+med)[:,outputs]
        pred=y*scale[:,outputs]+med[:,outputs]
        diff=np.zeros((b,6,h,w),dtype=np.float32)
        diff[:,outputs]=pred-target
        update_acc(acc,torch.from_numpy(diff),torch.from_numpy(valid),combo)
    return finalize_acc(acc)

def cached_geo_embeddings(geo_cp,geo_config,stats_path,val,device):
    """Load only matching validated frozen-Swin 3-mask feature cache.

    Note: The cached feature order must correspond to current validation order.
    The Phase 3.5.2 cache creation uses the MultiYearOpticalDataset's sample order.
    """
    cp=torch.load(geo_cp,map_location='cpu',weights_only=True)
    cfg=cp['config']
    fp=cp.get('fingerprints',{})
    if sha256(stats_path)!=fp.get('normalization_sha256'):
        raise RuntimeError('Geographic projection normalization fingerprint mismatch')
    src=resolve(cfg['experiment']['pretrained_checkpoint'])
    if sha256(src)!=fp.get('pretrained_sha256'):
        raise RuntimeError('Geographic projection Swin checkpoint fingerprint mismatch')
    if sha256(resolve('data/metadata/manifests/splits/buto_demo_2024/val.parquet')) != fp.get('val_manifest_sha256'):
        raise RuntimeError('Geographic projection geographic validation split fingerprint mismatch')
    cache=resolve(cfg['experiment']['cache_dir'])/'val_features.pt'
    if not cache.exists():
        raise FileNotFoundError(f'Validated frozen Swin cache not found: {cache}\n'
            'Run Phase 3.5.2 feature extraction before the fusion comparison.')
    payload=torch.load(cache,map_location='cpu',weights_only=True)
    metadata=payload.get('metadata',{})
    if metadata.get('split')!='val' or metadata.get('masks')!=3:
        raise RuntimeError('Unexpected frozen Swin cache metadata: inspect cache metadata')
    for key in ('pretrained_sha256','normalization_sha256','val_manifest_sha256'):
        if metadata.get(key)!=fp.get(key):
            raise RuntimeError(f'Frozen Swin cache fingerprint mismatch: {key}')
    feats=payload['embeddings'].float()
    if feats.shape!=(len(val),768):
        raise RuntimeError(f'Unexpected Swin cache features shape {feats.shape}')
    head=GeographicProjector(**cfg['model'])
    head.load_state_dict(cp['head_state'],strict=True)
    head.eval()
    with torch.inference_mode(): z=head(feats).cpu()
    return F.normalize(z,dim=1),F.normalize(feats,dim=1)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spectral-checkpoint',required=True)
    p.add_argument('--geo-checkpoint',default=None)
    p.add_argument('--stats',default='outputs/statistics/optical_buto_multiyear_v1.json')
    p.add_argument('--output-dir',default='outputs/evaluation/optical_spectral_mae')
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--num-workers',type=int,default=0)
    p.add_argument('--ridge-observations',type=int,default=320)
    p.add_argument('--ridge-pixels-per-observation',type=int,default=128)
    p.add_argument('--ridge-alpha',type=float,default=5.0)
    p.add_argument('--cpu',action='store_true')
    p.add_argument('--no-ridge',action='store_true')
    args=p.parse_args()
    if args.batch_size<1: p.error('Positive batch size required')
    device=torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    stats_path=resolve(args.stats)
    stats=json.loads(stats_path.read_text('utf8'))
    if tuple(stats['bands'])!=BANDS or stats['fit_split']!='train':
        raise RuntimeError('Wrong training statistics or spectral band order')
    cp_path=resolve(args.spectral_checkpoint)
    checkpoint=torch.load(cp_path,map_location='cpu',weights_only=True)
    model=SpectralMAE(**checkpoint['model_config']).to(device)
    model.load_state_dict(checkpoint['model_state'],strict=True)
    if checkpoint.get('masked_bands')!=2:
        raise RuntimeError('Expected checkpoint trained with exactly two masked bands')
    options={'years':(2022,2023,2024,2025),'statistics':stats,
             'min_valid_fraction':float(stats.get('min_valid_fraction',0.5)),
             'min_source_observations':int(stats.get('min_source_observations',1))}
    val=MultiYearOpticalDataset(ROOT,split='val',**options)
    if len(val)!=80 or {int(x['year']) for x in val.samples}!={2024}:
        raise RuntimeError('Expected 80 validation observations in 2024')
    if len({x['patch_id'] for x in val.samples})!=20:
        raise RuntimeError('Expected exactly 20 validation geographic locations')
    out_dir=resolve(args.output_dir);out_dir.mkdir(parents=True,exist_ok=True)
    print(f'Phase 3.6.1 | spectral checkpoint epoch={checkpoint["epoch"]} | device={device}',flush=True)
    print('Evaluating 3 fixed missing-band pairs on 80 held-out observations...',flush=True)
    spectral_metrics,embeddings,aggregate=evaluate_spectral(model,val,None,stats,device,args.batch_size,args.num_workers,COMBINATIONS)
    report={'method':{'validation_observations':80,'validation_locations':20,'year':2024,
                      'spectral_mask':'two whole spectral channels, fixed masks',
                      'reflectance_units':'surface reflectance 0..1 units (inverse train normalization)',
                      'note':'Model selection and fusion comparisons on validation are not independent test claims'},
            'spectral_checkpoint':str(cp_path),'spectral_checkpoint_epoch':int(checkpoint['epoch']),
            'missing_band_reconstruction':spectral_metrics,
            'spectral_retrieval':{'combined_three_mask_embeddings':retrieval(aggregate,val.samples),
                                  'embedding_effective_rank':effective_rank(aggregate.numpy()),
                                  'per_mask':{key:retrieval(v,val.samples) for key,v in embeddings.items()}}}
    if not args.no_ridge:
        train=MultiYearOpticalDataset(ROOT,split='train',**options)
        if set(x['patch_id'] for x in train.samples)&set(x['patch_id'] for x in val.samples):
            raise RuntimeError('Train/val geographic leakage')
        print('Fitting train-only ridge regressors for masked-band baselines...',flush=True)
        report['ridge_train_only']={}
        for combo in COMBINATIONS:
            X,Y,inputs,outputs=ridge_samples(train,combo,args.ridge_observations,args.ridge_pixels_per_observation,42)
            coef=ridge_fit(X,Y,args.ridge_alpha)
            report['ridge_train_only']['+'.join(combo)]={
                'fitted_training_pixels':len(X),'alpha':args.ridge_alpha,
                'metrics':ridge_evaluate(val,stats,combo,coef,inputs,outputs,args.batch_size)}
    if args.geo_checkpoint:
        print('Loading Phase 3.5.2 cached Swin features and projection head...',flush=True)
        geo,backbone=cached_geo_embeddings(resolve(args.geo_checkpoint),None,stats_path,val,device)
        report['geographic_retrieval']={'swin_backbone':retrieval(backbone,val.samples),
                                        'swin_projector':retrieval(geo,val.samples)}
        # A priori fusion coefficients; choose final weights on an independent development set.
        report['fixed_weight_fusion']={}
        for weight in (0.,0.25,0.5,1.):
            fused=F.normalize(torch.cat([geo,aggregate*weight],dim=1),dim=1)
            report['fixed_weight_fusion'][str(weight)]={'spectral_weight':weight,
              'retrieval':retrieval(fused,val.samples), 'effective_rank':effective_rank(fused.numpy())}
    path=out_dir/'spectral_mae_evaluation.json'
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('SpectralMAE masked reflectance MAE:',{key:round(x['overall']['mae'],6) for key,x in spectral_metrics.items()},flush=True)
    print('Spectral-only Top-1:',f"{report['spectral_retrieval']['combined_three_mask_embeddings']['top1_accuracy']:.2%}",flush=True)
    if 'geographic_retrieval' in report:
        print('Swin projector Top-1:',f"{report['geographic_retrieval']['swin_projector']['top1_accuracy']:.2%}",flush=True)
        print('Fusion Top-1:',{k:round(v['retrieval']['top1_accuracy'],4) for k,v in report['fixed_weight_fusion'].items()},flush=True)
    print('Saved:',path,flush=True)

if __name__=='__main__':main()
