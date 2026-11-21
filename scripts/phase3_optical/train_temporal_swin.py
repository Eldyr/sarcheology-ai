"""Phase 3.5 temporal Swin: reconstruction + VICReg, with checkpoint and resume.

Usage:
 uv run python scripts/phase3_optical/train_temporal_swin.py --smoke
 uv run python scripts/phase3_optical/train_temporal_swin.py --batch-size 16
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from sarchaeology.data.optical_multiyear import MultiYearOpticalDataset
from sarchaeology.data.optical_temporal_pairs import TemporalPairs
from sarchaeology.models.temporal_swin import TemporalSwin, vicreg, masked_huber

DEFAULT = ROOT / 'configs/training/optical_temporal_swin.yaml'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_atomic(path, object_):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        torch.save(object_, temp)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def make_generator(seed):
    gen = torch.Generator(device='cpu')
    gen.manual_seed(int(seed))
    return gen


def forward_pair(model, batch, device, epoch, batch_index, settings, *, train):
    a = batch['image_a'].to(device, non_blocking=True)
    b = batch['image_b'].to(device, non_blocking=True)
    va = batch['valid_a'].to(device, non_blocking=True)
    vb = batch['valid_b'].to(device, non_blocking=True)
    seed = int(settings['seed']) + (epoch * 100000 if train else 50000000) + batch_index * 2
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        oa = model(a, va, generator=make_generator(seed))
        ob = model(b, vb, generator=make_generator(seed + 1))
    rec = (masked_huber(model.backbone, a, va, oa, settings['huber_delta']) +
           masked_huber(model.backbone, b, vb, ob, settings['huber_delta'])) * 0.5
    inv, var, cov = vicreg(oa['projection'], ob['projection'])
    return rec, inv, var, cov


def run_epoch(model, loader, optimizer, device, epoch, config, *, train, smoke=False):
    model.train(train)
    totals = torch.zeros(5, device=device)
    n = 0
    loss_cfg = config['loss']
    weights = [float(loss_cfg[k]) for k in ('reconstruction_weight','invariance_weight','variance_weight','covariance_weight')]
    for idx, batch in enumerate(loader):
        if smoke and idx >= 1:
            break
        with torch.set_grad_enabled(train):
            rec, inv, var, cov = forward_pair(model, batch, device, epoch, idx, config['training'], train=train)
            total = weights[0]*rec + weights[1]*inv + weights[2]*var + weights[3]*cov
            if not torch.isfinite(total):
                raise RuntimeError('Non-finite temporal training loss')
            if train:
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config['training']['gradient_clip_norm']))
                optimizer.step()
        totals += torch.stack([total.detach(), rec.detach(), inv.detach(), var.detach(), cov.detach()])
        n += 1
    if n == 0:
        raise RuntimeError('No batches')
    values = (totals / n).cpu().tolist()
    return dict(zip(('total','reconstruction','invariance','variance','covariance'), map(float, values)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, default=DEFAULT)
    p.add_argument('--batch-size', type=int)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--resume', type=Path)
    args = p.parse_args()
    if args.smoke and args.resume:
        p.error('--smoke and --resume cannot be combined')
    config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    if args.batch_size is not None:
        if args.batch_size < 2:
            p.error('batch size must be at least 2 for VICReg')
        config['data']['batch_size'] = args.batch_size
    if config['training']['precision'] != 'bf16':
        raise ValueError('Expected BF16 configuration')
    seed = int(config['training']['seed'])
    seed_everything(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.set_float32_matmul_precision('high')
    stats_path = ROOT / config['experiment']['statistics_path']
    stats = json.loads(stats_path.read_text(encoding='utf-8'))
    if stats.get('fit_split') != 'train':
        raise ValueError('Expected train-only normalization')
    d = config['data']
    if 'min_valid_fraction' in stats and float(stats['min_valid_fraction']) != float(d['min_valid_fraction']):
        raise ValueError('Normalization validity threshold differs')
    if 'min_source_observations' in stats and int(stats['min_source_observations']) != int(d['min_source_observations']):
        raise ValueError('Normalization observation threshold differs')
    base = {}
    for split in ('train', 'val'):
        base[split] = MultiYearOpticalDataset(ROOT, years=d['years'], split=split, statistics=stats,
            min_valid_fraction=d['min_valid_fraction'], min_source_observations=d['min_source_observations'])
    train_ids = {str(s['patch_id']) for s in base['train'].samples}
    val_ids = {str(s['patch_id']) for s in base['val'].samples}
    test_file = ROOT / 'data/metadata/manifests/splits/buto_demo_2024/test.parquet'
    test_ids = set(pd.read_parquet(test_file)['patch_id'].astype(str))
    if train_ids & (val_ids | test_ids) or val_ids & test_ids:
        raise RuntimeError('Geographic split overlap')
    if len(train_ids) != 240 or len(val_ids) != 20 or len(base['train']) != 3840 or len(base['val']) != 80:
        raise RuntimeError('Unexpected training/validation footprint or observation counts')
    datasets = {split: TemporalPairs(base[split], d['same_quarter_probability'], seed + (0 if split=='train' else 9000))
                for split in ('train', 'val')}
    workers = int(d['num_workers'])
    options = dict(batch_size=int(d['batch_size']), num_workers=workers,
                   pin_memory=bool(d['pin_memory']), drop_last=True, persistent_workers=False)
    loaders = {'train': DataLoader(datasets['train'], shuffle=True, **options),
               'val': DataLoader(datasets['val'], shuffle=False, **options)}
    model_cfg = dict(config['model'])
    projection_dim = model_cfg.pop('projection_dim')
    hidden_dim = model_cfg.pop('projector_hidden_dim')
    model = TemporalSwin(model_cfg, projection_dim, hidden_dim)
    pretrained_path = ROOT / config['experiment']['pretrained_checkpoint']
    pretrained = torch.load(pretrained_path, map_location='cpu', weights_only=True)
    model.backbone.load_state_dict(pretrained['model_state'], strict=True)
    print('Pretrained Phase 3.4 checkpoint:', pretrained_path)
    print('Parameters:', sum(x.numel() for x in model.parameters()))
    model = model.to(device)
    backbone_params = list(model.backbone.parameters())
    projector_params = list(model.projector.parameters())
    tr = config['training']
    optimizer = torch.optim.AdamW([
        {'params':backbone_params,'lr':float(tr['backbone_learning_rate'])},
        {'params':projector_params,'lr':float(tr['projector_learning_rate'])}], weight_decay=float(tr['weight_decay']))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(tr['epochs']), eta_min=1e-7)
    cfg_bytes = yaml.safe_dump(config, sort_keys=False).encode()
    cfg_sha = hashlib.sha256(cfg_bytes).hexdigest()
    fingerprints = {
        'config_sha256':cfg_sha, 'normalization_sha256':sha(stats_path), 'pretrained_sha256':sha(pretrained_path),
        'train_manifest_sha256':sha(ROOT / 'data/metadata/manifests/splits/buto_demo_2024/train.parquet'),
        'val_manifest_sha256':sha(ROOT / 'data/metadata/manifests/splits/buto_demo_2024/val.parquet'),
        'test_manifest_sha256':sha(test_file),
    }
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run_id = f'temporal_swin_{stamp}_{uuid.uuid4().hex[:8]}'
    start_epoch, best, stale = 1, float('inf'), 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=True)
        for key, value in fingerprints.items():
            if ckpt[key] != value:
                raise ValueError(f'Resume fingerprint mismatch: {key}')
        model.load_state_dict(ckpt['model_state'], strict=True)
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        run_id, start_epoch = ckpt['run_id'], int(ckpt['epoch']) + 1
        best, stale = float(ckpt['best_val_loss']), int(ckpt['stale'])
    out = ROOT / config['experiment']['output_dir'] / run_id
    checkpoints = ROOT / config['experiment']['checkpoint_dir'] / run_id
    if not args.smoke:
        out.mkdir(parents=True,exist_ok=True)
        checkpoints.mkdir(parents=True,exist_ok=True)
        if not args.resume:
            (out/'config.yaml').write_bytes(cfg_bytes)
            (out/'metadata.json').write_text(json.dumps({'run_id':run_id, **fingerprints},indent=2))
    print(f'Phase 3.5: {len(base["train"])} train observations, {len(base["val"])} val, device={device}')
    for epoch in range(start_epoch, 2 if args.smoke else int(tr['epochs'])+1):
        begin = time.perf_counter()
        datasets['train'].set_epoch(epoch)
        datasets['val'].set_epoch(0)
        seed_everything(seed + epoch)
        train_metrics = run_epoch(model,loaders['train'],optimizer,device,epoch,config,train=True,smoke=args.smoke)
        val_metrics = run_epoch(model,loaders['val'],None,device,epoch,config,train=False,smoke=args.smoke)
        scheduler.step()
        current = val_metrics['total']
        improved = current < best - 1e-6
        if improved:
            best, stale = current, 0
        else:
            stale += 1
        record = {'epoch':epoch,'train':train_metrics,'val':val_metrics,'best_val_loss':best,'stale':stale,'seconds':time.perf_counter()-begin}
        print(f"Epoch {epoch}: train={train_metrics['total']:.5f}, val={current:.5f}, rec={val_metrics['reconstruction']:.5f}, inv={val_metrics['invariance']:.5f}, best={best:.5f}, time={record['seconds']:.1f}s", flush=True)
        if args.smoke:
            print('SMOKE TEST PASSED; no files saved')
            return
        with (out/'history.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(record)+'\n')
        state = dict(epoch=epoch,run_id=run_id,best_val_loss=best,stale=stale,
                     model_state=model.state_dict(),optimizer_state=optimizer.state_dict(),
                     scheduler_state=scheduler.state_dict(), **fingerprints)
        save_atomic(checkpoints/'last.pt',state)
        if improved:
            save_atomic(checkpoints/'best.pt',state)
        if stale>=int(tr['patience']):
            print('Early stopping')
            break
    print('Done:',run_id,'best total validation loss:',best,'checkpoints:',checkpoints)


if __name__=='__main__':
    main()
