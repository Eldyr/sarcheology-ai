"""Phase 3.5.2: frozen Phase 3.4 Swin + trainable geography-aware head.

Run from repository root:
  uv run python scripts/phase3_optical/train_geographic_projection.py --smoke
  uv run python scripts/phase3_optical/train_geographic_projection.py

The Swin is strictly frozen; embeddings are extracted/cacheable once. Training
uses 2 observations for each distinct training footprint and avoids training
on any validation/test geographic identity. The retrieval benchmark uses 2024
only and 3 deterministic masks per observation, matching Phase 3.5.1.
"""
from __future__ import annotations
import argparse, hashlib, json, os, random, sys, uuid, re, math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from sarchaeology.data.optical_multiyear import MultiYearOpticalDataset
from sarchaeology.models.optical_swin_mim import OpticalSwinMIM
from sarchaeology.models.geographic_projection import (GeographicProjector, contrastive_loss,
                                                      variance_covariance_loss, relational_loss)


def filepath(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def digest(path):
    return hashlib.sha256(filepath(path).read_bytes()).hexdigest()


def load_checkpoint(path):
    return torch.load(filepath(path), map_location='cpu', weights_only=True)


def fixed_seed(seed):
    random.seed(seed); np.random.seed(seed % (2**32)); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def atomic_torch(path, value):
    path = filepath(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        torch.save(value, temp); os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def validate_splits(train, val):
    train_ids = {str(s['patch_id']) for s in train.samples}
    val_ids = {str(s['patch_id']) for s in val.samples}
    test = pd.read_parquet(ROOT / 'data/metadata/manifests/splits/buto_demo_2024/test.parquet')
    test_ids = set(test['patch_id'].astype(str))
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise RuntimeError('Geographic split leakage')
    if len(train_ids)!=240 or len(val_ids)!=20 or len(train)!=3840 or len(val)!=80:
        raise RuntimeError('Unexpected footprint/sample counts')
    if {int(s['year']) for s in val.samples}!={2024}:
        raise RuntimeError('Validation must be 2024 only')


@torch.inference_mode()
def extract_embeddings(model, data, device, *, n_masks, seed, batch_size, workers, stable):
    """Returns embeddings in dataset sample order, averaged over mask views.

    Validation extraction is batch-independent (batch=1 per sample), exactly as
    in Phase 3.5.1. Train extraction uses deterministic batch-index RNG for speed.
    """
    loader = DataLoader(data, batch_size=1 if stable else batch_size, shuffle=False,
                        num_workers=0 if stable else workers, pin_memory=device.type=='cuda')
    model.eval()
    features = []
    for batch_i, batch in enumerate(loader):
        x = batch['image'].to(device, non_blocking=True)
        valid = batch['valid_mask'].to(device, non_blocking=True)
        views = []
        for mask_i in range(n_masks):
            generator = torch.Generator(device='cpu')
            if stable:
                generator.manual_seed(seed + batch_i*1009 + mask_i*1_000_003)
            else:
                generator.manual_seed(seed + batch_i*1009 + mask_i*1_000_003)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type=='cuda'):
                out = model(x, valid, generator=generator)
            views.append(F.normalize(out['embedding'].float().cpu(), dim=1))
        features.append(F.normalize(torch.stack(views).mean(0), dim=1))
        if stable and (batch_i+1)%20==0:
            print(f'  extracted {batch_i+1}/{len(data)} validation observations', flush=True)
    return torch.cat(features, dim=0)


def get_cached_features(model, data, device, config, split, path, metadata, smoke):
    n_masks = int(config['data']['validation_masks' if split=='val' else 'train_masks'])
    expected = {**metadata, 'split':split, 'sample_count':len(data), 'masks':n_masks,
                'stable':split=='val', 'feature_batch_size':int(config['data']['feature_batch_size'])}
    if path.exists() and not smoke:
        state = load_checkpoint(path)
        if state.get('metadata')==expected:
            features=state['embeddings']
            if features.shape==(len(data), 768):
                print(f'Using cached {split} embeddings: {path}',flush=True)
                return features
    print(f'Extracting {split} frozen embeddings...',flush=True)
    features = extract_embeddings(model,data,device,n_masks=n_masks,seed=20261121,
          batch_size=int(config['data']['feature_batch_size']),
          workers=int(config['data']['num_workers']),stable=split=='val')
    if not smoke: atomic_torch(path, {'metadata':expected,'embeddings':features})
    return features


def grouped(samples):
    groups=defaultdict(list)
    for i,s in enumerate(samples): groups[str(s['patch_id'])].append(i)
    if any(len(v)<2 for v in groups.values()): raise RuntimeError('Requires 2+ temporal observations per geographic ID')
    return groups


def location_coordinates(groups, samples):
    """Require known patch grid positions; do not guess separation distances."""
    result={}
    for pid,indices in groups.items():
        row=samples[indices[0]]
        if 'row' in row and 'col' in row:
            result[pid]=(float(row['row']),float(row['col']))
        else:
            match=re.search(r'r(\d+)_c(\d+)',pid)
            if match is None:
                raise RuntimeError('Cannot infer patch positions: include row/col in sample metadata')
            result[pid]=(float(match.group(1)),float(match.group(2)))
    return result


def distant_batches(locations, coords, minimum_distance, max_size):
    """Greedily choose distinct, spatially separated negative identities.

    Unselected IDs are reconsidered on subsequent batches, while the random
    permutation changes every epoch. Last groups smaller than 2 are skipped.
    """
    remaining=list(locations)
    while remaining:
        chosen=[]
        leftovers=[]
        for pid in remaining:
            rc=coords[pid]
            far_enough=all(math.dist(rc,coords[other])>=minimum_distance for other in chosen)
            if len(chosen)<max_size and far_enough: chosen.append(pid)
            else: leftovers.append(pid)
        if len(chosen)<2:
            # A singleton remainder cannot form a contrastive batch; next epoch reshuffles.
            break
        yield chosen
        remaining=leftovers


def draw_pair(indices, samples, rng, same_quarter_probability):
    i=int(rng.choice(indices))
    q=int(samples[i]['quarter'])
    y=int(samples[i]['year'])
    same=[j for j in indices if j!=i and int(samples[j]['quarter'])==q and int(samples[j]['year'])!=y]
    other=[j for j in indices if j!=i and int(samples[j]['quarter'])!=q]
    choose_same=rng.random()<same_quarter_probability
    pool=(same if choose_same else other) or (other if choose_same else same)
    if not pool: raise RuntimeError('No suitable temporal partner')
    return i,int(rng.choice(pool))


def make_pairs(groups, samples, ids, rng, probability):
    left,right=[],[]
    for pid in ids:
        a,b=draw_pair(groups[pid],samples,rng,probability)
        left.append(a);right.append(b)
    return left+right, torch.arange(len(ids)).repeat(2)


def retrieval(vectors, samples):
    x=F.normalize(vectors.float(),dim=1)
    ids=[str(s['patch_id']) for s in samples]
    quarters=[int(s['quarter']) for s in samples]
    right1=right5=n=0
    per=[]
    outcomes=[]
    for a in range(1,5):
        qi=[i for i,q in enumerate(quarters) if q==a]
        for b in range(1,5):
            if a==b: continue
            gi=[i for i,q in enumerate(quarters) if q==b]
            if len(qi)!=20 or len(gi)!=20 or {ids[i] for i in qi}!={ids[i] for i in gi}:
                raise RuntimeError('Expected 20 matching footprints per quarter')
            top=(x[qi] @ x[gi].T).topk(5,dim=1).indices
            a1=a5=0
            for index,row in enumerate(top.tolist()):
                pid=ids[qi[index]]
                is1=ids[gi[row[0]]]==pid
                is5=any(ids[gi[j]]==pid for j in row)
                a1+=int(is1);a5+=int(is5)
                outcomes.append({'patch_id':pid,'top1':int(is1)})
            right1+=a1;right5+=a5;n+=len(qi)
            per.append({'query_quarter':a,'gallery_quarter':b,'top1':a1/20,'top5':a5/20})
    return {'top1_accuracy':right1/n,'top5_accuracy':right5/n,
            'correct_top1':right1,'correct_top5':right5,'queries':n,
            'quarter_pairs':per},outcomes


def rank_stats(vectors):
    x=F.normalize(vectors.double(),dim=1)
    xc=x-x.mean(0)
    gram=xc @ xc.T/max(len(x)-1,1)
    eig=torch.linalg.eigvalsh(gram).clamp_min(0)
    total=eig.sum().item()
    p=eig/max(total,1e-30)
    p=p[p>1e-12]
    erank=torch.exp(-(p*p.log()).sum()).item() if len(p) else 0
    return {'effective_rank':erank,'participation_ratio':total**2/max(eig.square().sum().item(),1e-30),
            'mean_coordinate_std':xc.std(0).mean().item(), 'max_rank':min(len(x)-1,x.shape[1])}


def bootstrap(a,b,seed=42,replicates=2000):
    groups=defaultdict(list)
    for x,y in zip(a,b):
        assert x['patch_id']==y['patch_id']
        groups[x['patch_id']].append(y['top1']-x['top1'])
    ids=sorted(groups)
    means=np.array([np.mean(groups[x]) for x in ids])
    rng=np.random.default_rng(seed)
    sampled=rng.integers(0,len(means),size=(replicates,len(means)))
    diffs=means[sampled].mean(axis=1)
    return {'difference':float(means.mean()),'95pct_ci':np.quantile(diffs,[.025,.975]).tolist(),
            'unit':'geographic_patch_id','replicates':replicates}


def step(head, features, indices, labels, cfg):
    teacher=features[indices]
    z=head(teacher)
    tr=cfg['training']
    con=contrastive_loss(z,labels,temperature=float(tr['temperature']))
    dist=relational_loss(z,teacher,temperature=float(tr['teacher_temperature']))
    var,cov=variance_covariance_loss(z,target_std=float(tr['target_std']))
    total=(float(tr['contrastive_weight'])*con + float(tr['relational_weight'])*dist +
           float(tr['variance_weight'])*var + float(tr['covariance_weight'])*cov)
    return total, {'contrastive':con.detach().item(),'relational':dist.detach().item(),
                   'variance':var.detach().item(),'covariance':cov.detach().item()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/training/optical_geographic_projection.yaml')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--epochs',type=int)
    p.add_argument('--locations-per-batch',type=int)
    p.add_argument('--cpu',action='store_true')
    p.add_argument('--rebuild-cache',action='store_true')
    args=p.parse_args()
    config=yaml.safe_load(filepath(args.config).read_text('utf-8'))
    if args.epochs is not None: config['training']['epochs']=args.epochs
    if args.locations_per_batch is not None: config['training']['locations_per_batch']=args.locations_per_batch
    tr=config['training'];d=config['data'];exp=config['experiment']
    if int(tr['locations_per_batch'])<2 or int(tr['epochs'])<1: p.error('Invalid batch size or epochs')
    fixed_seed(int(tr['seed']))
    device=torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    if device.type=='cuda': torch.set_float32_matmul_precision('high')
    stat_path=filepath(exp['statistics_path'])
    stats=json.loads(stat_path.read_text('utf-8'))
    if stats.get('fit_split')!='train': raise RuntimeError('Expected train-fitted statistics')
    if 'min_valid_fraction' in stats and abs(float(stats['min_valid_fraction'])-float(d['min_valid_fraction']))>1e-9:
        raise RuntimeError('Normalization validity threshold mismatch')
    if 'min_source_observations' in stats and int(stats['min_source_observations'])!=int(d['min_source_observations']):
        raise RuntimeError('Normalization source-observations mismatch')
    datasets={split:MultiYearOpticalDataset(ROOT,years=d['years'],split=split,statistics=stats,
              min_valid_fraction=d['min_valid_fraction'],min_source_observations=d['min_source_observations'])
              for split in ('train','val')}
    validate_splits(datasets['train'],datasets['val'])
    source=filepath(exp['pretrained_checkpoint'])
    archived_config=yaml.safe_load(filepath(exp['pretrained_config']).read_text('utf-8'))
    backbone=OpticalSwinMIM(**archived_config['model'])
    checkpoint=load_checkpoint(source)
    backbone.load_state_dict(checkpoint['model_state'],strict=True)
    for param in backbone.parameters(): param.requires_grad_(False)
    backbone.to(device).eval()
    sample_meta={'pretrained_sha256':digest(source),'normalization_sha256':digest(stat_path),
                 'pretrained_config_sha256':digest(filepath(exp['pretrained_config'])),
                 'train_manifest_sha256':digest('data/metadata/manifests/splits/buto_demo_2024/train.parquet'),
                 'val_manifest_sha256':digest('data/metadata/manifests/splits/buto_demo_2024/val.parquet')}
    if args.smoke:
        # Exercise the actual frozen encoder + head, but only four observations;
        # avoid the full 3,840-sample feature-cache pass during a smoke test.
        selected=[]
        seen=set()
        for i, row in enumerate(datasets['train'].samples):
            pid=str(row['patch_id'])
            if pid not in seen and len(seen)>=2: continue
            seen.add(pid);selected.append(i)
            if len(selected)==4: break
        if len(selected)!=4 or len(seen)!=2:
            raise RuntimeError('Smoke requires two geographic IDs')
        small=Subset(datasets['train'],selected)
        feats=extract_embeddings(backbone,small,device,n_masks=1,seed=20261121,
                                 batch_size=4,workers=0,stable=False).to(device).clone()
        head=GeographicProjector(**config['model']).to(device)
        with torch.enable_grad():
            loss,terms=step(head,feats,torch.arange(4,device=device),
                            torch.tensor([0,1,0,1],device=device),config)
            loss.backward()
        assert torch.isfinite(loss)
        assert all(param.grad is None for param in backbone.parameters())
        print(f'SMOKE PASSED: loss={loss.item():.5f}; frozen backbone gradients=None; no checkpoints saved')
        return
    cache=filepath(exp['cache_dir']);cache.mkdir(parents=True,exist_ok=True)
    if args.rebuild_cache:
        for split in ['train','val']:
            (cache/f'{split}_features.pt').unlink(missing_ok=True)
    train_features=get_cached_features(backbone,datasets['train'],device,config,'train',
                             cache/'train_features.pt',sample_meta,args.smoke)
    val_features=get_cached_features(backbone,datasets['val'],device,config,'val',
                             cache/'val_features.pt',sample_meta,args.smoke)
    del backbone
    if device.type=='cuda': torch.cuda.empty_cache()
    train_features=F.normalize(train_features.float().to(device).clone(),dim=1)
    val_features=F.normalize(val_features.float().to(device).clone(),dim=1)
    baseline,base_outcomes=retrieval(val_features.cpu(),datasets['val'].samples)
    print(f'Pretrained Swin validation Top-1={baseline["top1_accuracy"]*100:.2f}% Top-5={baseline["top5_accuracy"]*100:.2f}%',flush=True)
    if abs(baseline['top1_accuracy']-146/240)>0.02:
        print('WARNING: baseline differs from 60.83% reference; check preprocessing and masking protocol',flush=True)
    head=GeographicProjector(**config['model']).to(device)
    optimizer=torch.optim.AdamW(head.parameters(),lr=float(tr['learning_rate']),weight_decay=float(tr['weight_decay']))
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=int(tr['epochs']))
    train_groups=grouped(datasets['train'].samples)
    coordinates=location_coordinates(train_groups,datasets['train'].samples)
    rng=np.random.default_rng(int(tr['seed']))
    runid='geo_projection_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    out=filepath(exp['output_dir'])/runid
    ckdir=filepath(exp['checkpoint_dir'])/runid
    best=-1.0; stale=0
    if not args.smoke:
        out.mkdir(parents=True,exist_ok=True);ckdir.mkdir(parents=True,exist_ok=True)
        (out/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
        (out/'metadata.json').write_text(json.dumps({'run_id':runid,**sample_meta},indent=2),encoding='utf-8')
    n_locations=int(tr['locations_per_batch'])
    for epoch in range(1,2 if args.smoke else int(tr['epochs'])+1):
        head.train()
        locations=np.array(sorted(train_groups))
        rng.shuffle(locations)
        epoch_loss=[]
        for chosen in distant_batches(locations, coordinates, float(d['min_negative_distance_pixels']), n_locations):
            if len(chosen)<2: continue
            indices,identities=make_pairs(train_groups,datasets['train'].samples,chosen,rng,float(d['same_quarter_probability']))
            indices=torch.tensor(indices,device=device,dtype=torch.long)
            identities=identities.to(device)
            loss,terms=step(head,train_features,indices,identities,config)
            if not torch.isfinite(loss): raise RuntimeError('Non-finite contrastive loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(),float(tr['grad_clip_norm']))
            optimizer.step()
            epoch_loss.append(loss.item())
            if args.smoke: break
        scheduler.step()
        head.eval()
        with torch.no_grad():
            val_projected=head(val_features).cpu()
        score,outcomes=retrieval(val_projected,datasets['val'].samples)
        quality=score['top1_accuracy']
        improved=quality>best+1e-9
        if improved: best=quality;stale=0
        else: stale+=1
        print(f'Epoch {epoch}: train={np.mean(epoch_loss):.4f} val_top1={quality*100:.2f}% '
              f'val_top5={score["top5_accuracy"]*100:.2f}% best_top1={best*100:.2f}%',flush=True)
        if args.smoke:
            print('SMOKE PASSED; no checkpoints or history saved')
            return
        record={'epoch':epoch,'train_loss':float(np.mean(epoch_loss)),
                'val':score,'effective_rank':rank_stats(val_projected),'best_top1':best}
        with (out/'history.jsonl').open('a',encoding='utf-8') as stream: stream.write(json.dumps(record)+'\n')
        state={'epoch':epoch,'run_id':runid,'head_state':head.state_dict(),
               'optimizer_state':optimizer.state_dict(), 'config':config,'fingerprints':sample_meta,
               'baseline':baseline,'val':score,'best_top1':best}
        atomic_torch(ckdir/'last.pt',state)
        if improved: atomic_torch(ckdir/'best.pt',state)
        if stale>=int(tr['patience']):
            print('Early stopping triggered');break
    head.load_state_dict(load_checkpoint(ckdir/'best.pt')['head_state'])
    head.eval()
    with torch.no_grad(): z=head(val_features).cpu()
    final,final_outcomes=retrieval(z,datasets['val'].samples)
    result={'run_id':runid,'method':{'train_locations':len(train_groups),'train_samples':len(datasets['train']),
            'val_locations':20,'val_samples':80,'masking':'same 3-mask stable evaluation as Phase 3.5.1',
            'frozen_backbone':True,'note':'Checkpoint selected using 2024 validation retrieval; this is model selection, not independent final-test performance.'},
            'baseline':baseline,'projection':final,'baseline_rank':rank_stats(val_features.cpu()),
            'projection_rank':rank_stats(z),'paired_bootstrap':bootstrap(base_outcomes,final_outcomes)}
    (out/'evaluation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('Finished:',runid,'checkpoints:',ckdir,'evaluation:',out/'evaluation.json',flush=True)


if __name__=='__main__': main()
