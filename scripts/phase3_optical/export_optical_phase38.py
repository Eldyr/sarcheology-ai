"""Phase 3.8 — georeferenced, patch-local optical feature exports.

Primary: full-input native-resolution Swin feature maps; independently reproduce
three-mask frozen geographic projection for the 256D per-patch vector.
Optional: SpectralMAE signed reconstruction residual maps for predeclared masks.
No training, fitting, or model selection. Does not build a misleading seamless AOI.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F
import yaml
from affine import Affine
from rasterio.transform import array_bounds

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts/phase3_optical'))
from sarchaeology.data.optical_multiyear import MultiYearOpticalDataset
from sarchaeology.models.optical_swin_mim import OpticalSwinMIM
from sarchaeology.models.geographic_projection import GeographicProjector

BANDS = ('B02', 'B03', 'B04', 'B08', 'B11', 'B12')
STAGE_CHANNELS = (96, 192, 384, 768)
STAGE_STRIDES = (4, 8, 16, 32)
SPECTRAL_PAIRS = (('B11', 'B12'), ('B04', 'B08'), ('B02', 'B03'))
NODATA = -9999.0


def resolve(value):
    p = Path(value).expanduser()
    return (p if p.is_absolute() else ROOT / p).resolve()


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def safe_id(value):
    text = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(value)).strip('._')
    if not text or text in ('.', '..'):
        raise ValueError('Invalid patch ID for filesystem')
    return text


def geographic_sets(root, manifest_dir):
    out = {}
    for name in ('train', 'val', 'test'):
        manifest = root / manifest_dir / f'{name}.parquet'
        if not manifest.exists():
            raise FileNotFoundError(f'Missing {name} manifest {manifest}')
        df = pd.read_parquet(manifest, columns=['patch_id'])
        out[name] = set(df.patch_id.astype(str))
    if out['train'] & out['val'] or out['train'] & out['test'] or out['val'] & out['test']:
        raise RuntimeError('Geographic split manifests overlap')
    return out


def validate_dataset(dataset, split_ids, split, year):
    seen = set()
    sample_ids = set()
    for sample in dataset.samples:
        pid = str(sample['patch_id'])
        if int(sample['year']) != year or int(sample['quarter']) not in (1, 2, 3, 4):
            raise RuntimeError('Dataset includes unexpected year/quarter')
        if pid not in split_ids[split]:
            raise RuntimeError('Dataset sample missing from selected split manifest')
        key = (pid, int(sample['year']), int(sample['quarter']))
        if key in seen:
            raise RuntimeError(f'Duplicate sample {key}')
        seen.add(key)
        sample_ids.add(pid)
    if not seen:
        raise RuntimeError('No usable data to export')
    return len(sample_ids)


def cell_valid_fraction(valid, stride):
    """Native-grid valid fraction, preserving localized missing data."""
    x = np.asarray(valid, dtype=bool)
    if x.shape != (128, 128) or 128 % stride:
        raise ValueError('Expected 128x128 validity and a divisor stride')
    n = 128 // stride
    return x.reshape(n, stride, n, stride).mean(axis=(1, 3)).astype(np.float32)


def patch_transform(reference_transform, row, col, stride=1):
    """Each stage cell covers stride x stride SOURCE pixels, not 10m pixels."""
    return reference_transform * Affine.translation(col, row) * Affine.scale(stride, stride)


def native_stage_tensor(feature, stage_index):
    """Channels-last pretrained Swin stage -> C,H,W numpy float32."""
    expected = (128 // STAGE_STRIDES[stage_index],) * 2 + (STAGE_CHANNELS[stage_index],)
    if tuple(feature.shape) != expected:
        raise RuntimeError(f'Unexpected stage {stage_index} shape {tuple(feature.shape)} != {expected}')
    return feature.detach().float().permute(2, 0, 1).cpu().numpy()


@torch.inference_mode()
def encode_full(swin, x):
    """Use pretrained encoder weights with NO mask tokens (an OOD inference mode)."""
    h = swin.patch_embed(x).permute(0, 2, 3, 1)
    stages = []
    for idx, stage in enumerate(swin.stages):
        h = stage(h)
        stages.append(h.detach())
        if idx < 3:
            h = swin.merges[idx](h)
    embedding = swin.encoder_norm(h).mean(dim=(1, 2))
    return embedding.float(), stages


@torch.inference_mode()
def encode_three_mask(swin, x, valid, dataset_index, n_masks=3):
    """Exactly Phase 3.7's per-observation 3-mask global embedding protocol."""
    pieces = []
    for mask_idx in range(n_masks):
        generator = torch.Generator(device='cpu')
        generator.manual_seed(20261121 + dataset_index * 1009 + mask_idx * 1_000_003)
        result = swin(x, valid, generator=generator)
        pieces.append(F.normalize(result['embedding'].float(), dim=1))
    return F.normalize(torch.stack(pieces).mean(0), dim=1)


def write_geotiff(path, data, transform, crs, valid_pixels, *, tags, nodata=NODATA):
    """Write a georeferenced C,H,W raster, masked where source data are insufficient."""
    values = np.asarray(data, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError('Data must be C,H,W')
    if values.shape[1:] != np.shape(valid_pixels):
        raise ValueError('Mask/grid shape mismatch')
    valid = np.asarray(valid_pixels, dtype=bool)
    if not np.isfinite(values[:, valid]).all():
        raise RuntimeError(f'Nonfinite valid pixels in {path}')
    values = values.copy()
    values[:, ~valid] = nodata
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, 'w', driver='GTiff', height=values.shape[1],
                       width=values.shape[2], count=values.shape[0],
                       dtype='float32', crs=crs, transform=transform, nodata=nodata,
                       compress='deflate', predictor=3, BIGTIFF='IF_SAFER') as dst:
        dst.write(values)
        dst.write_mask((valid * 255).astype(np.uint8))
        dst.update_tags(**{key: str(value) for key, value in tags.items()})


def load_stack(config, device, with_spectral):
    geo_path = resolve(config['checkpoints']['geographic'])
    geo_cp = torch.load(geo_path, map_location='cpu', weights_only=True)
    geo_cfg = geo_cp['config']
    pre_path = resolve(geo_cfg['experiment']['pretrained_checkpoint'])
    pre_cfg_path = resolve(geo_cfg['experiment']['pretrained_config'])
    statistics_path = resolve(geo_cfg['experiment']['statistics_path'])
    pre_arch = yaml.safe_load(pre_cfg_path.read_text(encoding='utf-8'))
    swin = OpticalSwinMIM(**pre_arch['model'])
    swin.load_state_dict(torch.load(pre_path, map_location='cpu', weights_only=True)['model_state'], strict=True)
    geo = GeographicProjector(**geo_cfg['model'])
    geo.load_state_dict(geo_cp['head_state'], strict=True)
    swin.eval().requires_grad_(False).to(device)
    geo.eval().requires_grad_(False).to(device)
    checksums = {'geographic': file_sha256(geo_path), 'swin': file_sha256(pre_path),
                 'swin_config': file_sha256(pre_cfg_path), 'statistics': file_sha256(statistics_path)}
    fp = geo_cp.get('fingerprints', {})
    if fp.get('pretrained_sha256') != checksums['swin'] or fp.get('normalization_sha256') != checksums['statistics']:
        raise RuntimeError('Geographic checkpoint is incompatible with Swin or normalization')
    stats = json.loads(statistics_path.read_text(encoding='utf-8'))
    if tuple(stats['bands']) != BANDS or stats.get('fit_split') != 'train':
        raise RuntimeError('Expected six train-normalized Sentinel-2 bands')
    spectral = None
    if with_spectral:
        from train_spectral_mae import SpectralMAE
        spec_path = resolve(config['checkpoints']['spectral'])
        spec_cp = torch.load(spec_path, map_location='cpu', weights_only=True)
        spectral = SpectralMAE(**spec_cp['model_config'])
        spectral.load_state_dict(spec_cp['model_state'], strict=True)
        spectral.eval().requires_grad_(False).to(device)
        checksums['spectral'] = file_sha256(spec_path)
    return swin, geo, spectral, geo_cfg, stats, checksums


@torch.inference_mode()
def spectral_residuals(model, x, statistics, device):
    """True minus predicted reflectance. Signed residuals; ONLY masked bands scored."""
    scale = np.asarray(statistics['scale'], dtype=np.float32)
    output = []
    for pair in SPECTRAL_PAIRS:
        mask = torch.zeros((1, 6), dtype=torch.bool, device=device)
        indices = [BANDS.index(band) for band in pair]
        mask[:, indices] = True
        prediction = model(x, mask)['reconstruction'].float()
        diff = (x.float() - prediction)[:, indices].squeeze(0).cpu().numpy()
        output.append((pair, diff * scale[indices, None, None]))
    return output


def read_existing_provenance(output):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Nonempty output directory: {output}. Choose --output for a new export.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/export/optical_phase38.yaml')
    p.add_argument('--split', choices=('train', 'val', 'test'), default=None)
    p.add_argument('--year', type=int, default=None)
    p.add_argument('--output', default=None)
    p.add_argument('--max-samples', type=int, default=None, help='Only for smoke/subset exports')
    p.add_argument('--smoke', action='store_true', help='Export two samples to a separate smoke directory')
    p.add_argument('--spectral-residuals', action='store_true', help='Export optional spectral residuals')
    p.add_argument('--cpu', action='store_true')
    args = p.parse_args()
    cfg = yaml.safe_load(resolve(args.config).read_text(encoding='utf-8'))
    split = args.split or cfg['export']['split']
    year = args.year or int(cfg['export']['year'])
    if split not in ('train', 'val', 'test'):
        p.error('Unknown split')
    if split in ('val', 'test') and year != 2024:
        p.error('The existing validation and test dataset protocols use 2024 only')
    if int(cfg['export']['masked_views']) != 3:
        p.error('The masked geographic projection requires exactly 3 views for Phase 3.7 compatibility')
    if not 0.0 <= float(cfg['export']['min_valid_cell_fraction']) <= 1.0:
        p.error('min_valid_cell_fraction must be in [0,1]')
    if args.max_samples is not None and args.max_samples < 1:
        p.error('max-samples must be >=1')
    nmax = 2 if args.smoke else args.max_samples
    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    if device.type == 'cuda':
        torch.set_float32_matmul_precision('high')
    swin, geo, spectral, geo_cfg, stats, hashes = load_stack(cfg, device, args.spectral_residuals)
    d = geo_cfg['data']
    data = MultiYearOpticalDataset(ROOT, years=[year], split=split, statistics=stats,
        min_valid_fraction=d['min_valid_fraction'],
        min_source_observations=d['min_source_observations'])
    manifest_dir = Path(cfg['export']['manifest_dir'])
    ids = geographic_sets(ROOT, manifest_dir)
    n_locations = validate_dataset(data, ids, split, year)
    raw_out = args.output or cfg['export']['output_dir']
    out = resolve(raw_out) / f'{split}_{year}'
    if args.smoke:
        out = resolve(raw_out) / f'_smoke_{split}_{year}'
    read_existing_provenance(out)
    ref_crs, ref_transform, ref_width, ref_height = data.reference_grid
    if ref_crs is None:
        raise RuntimeError('Cannot export GeoTIFF without a source CRS')
    if tuple(ref_transform) == tuple(Affine.identity()):
        raise RuntimeError('Reference raster has an identity geotransform')
    if not (ref_transform.a != 0 and ref_transform.e != 0):
        raise RuntimeError('Unexpected source pixel scale')
    record_list=[]
    stage_info=[]
    for stage, (channels, stride) in enumerate(zip(STAGE_CHANNELS, STAGE_STRIDES)):
        stage_info.append({'stage':stage, 'channels':channels, 'stride_source_pixels':stride,
                           'height':128//stride, 'width':128//stride,
                           'georeferencing':'affine transform from 128x128 source window; no interpolation'})
    chosen_samples = data.samples[:nmax] if nmax else data.samples
    print(f'Export {len(chosen_samples)} samples from {split}/{year} | {n_locations} locations | {device}', flush=True)
    out.mkdir(parents=True, exist_ok=False)
    with torch.inference_mode():
        for i, sample in enumerate(chosen_samples):
            # IMPORTANT: index i uses the UNMODIFIED dataset order for deterministic three-mask protocol.
            datum = data[i]
            x=datum['image'].unsqueeze(0).to(device)
            valid=datum['valid_mask'].unsqueeze(0).to(device).bool()
            x=torch.where(valid.expand_as(x),x,torch.zeros_like(x))
            pid=str(sample['patch_id'])
            folder=out / f'{sample["period"]}' / f'patch_{safe_id(pid)}'
            if folder.exists():
                raise RuntimeError(f'Output patch folder collision: {folder}')
            base_transform=patch_transform(ref_transform,int(sample['row']),int(sample['col']),1)
            valid_array=datum['valid_mask'][0].numpy().astype(bool)
            tags={'phase':'3.8','patch_id':pid,'period':sample['period'],
                  'year':sample['year'],'quarter':sample['quarter'], 'split':split,
                  'native_input_bands':','.join(BANDS),'inference':'unmasked spatial feature maps',
                  'swin_sha256':hashes['swin']}
            write_geotiff(folder/'valid_pixels_10m.tif',valid_array[None].astype(np.float32),
                          base_transform,ref_crs,np.ones_like(valid_array,dtype=bool),tags=tags, nodata=NODATA)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,enabled=device.type=='cuda'):
                full, maps=encode_full(swin,x)
                masked=encode_three_mask(swin,x,valid,i,n_masks=int(cfg['export']['masked_views']))
                full_proj=geo(F.normalize(full.float(),dim=1)).cpu().numpy()[0]
                masked_proj=geo(masked.float()).cpu().numpy()[0]
                if spectral is not None:
                    spectral_maps=spectral_residuals(spectral,x,stats,device)
                else:
                    spectral_maps=[]
            for stage_idx, layer in enumerate(maps):
                values=native_stage_tensor(layer[0],stage_idx)
                stride=STAGE_STRIDES[stage_idx]
                valid_fraction=cell_valid_fraction(valid_array,stride)
                cell_mask=valid_fraction>=float(cfg['export']['min_valid_cell_fraction'])
                transform=patch_transform(ref_transform,int(sample['row']),int(sample['col']),stride)
                stage_tags={**tags, 'stage':stage_idx,'source_stride_pixels':stride,
                            'feature_channels':values.shape[0], 'min_valid_cell_fraction':cfg['export']['min_valid_cell_fraction']}
                write_geotiff(folder/f'swin_stage{stage_idx}_unmasked.tif',values,transform,ref_crs,
                              cell_mask,tags=stage_tags)
                write_geotiff(folder/f'swin_stage{stage_idx}_valid_fraction.tif',valid_fraction[None],
                              transform,ref_crs,np.ones_like(cell_mask,dtype=bool),tags=stage_tags)
            for pair, residual in spectral_maps:
                name='_'.join(pair)
                write_geotiff(folder/f'spectral_residual_{name}_signed_reflectance.tif',residual,
                              base_transform,ref_crs,valid_array,tags={**tags,'bands':','.join(pair),
                              'residual_sign':'actual-minus-predicted','units':'surface_reflectance'})
            bounds=array_bounds(128,128,base_transform)
            row={'patch_id':pid,'split':split,'year':int(sample['year']),
                 'quarter':int(sample['quarter']),'period':sample['period'],
                 'row':int(sample['row']),'col':int(sample['col']),
                 'valid_fraction':float(sample['valid_fraction']),
                 'crs':str(ref_crs),'left':bounds[0],'bottom':bounds[1],
                 'right':bounds[2],'top':bounds[3],
                 'folder':str(folder.relative_to(out)),
                 'masked_views':int(cfg['export']['masked_views'])}
            row.update({f'geographic_masked_{j:03d}':float(v) for j,v in enumerate(masked_proj)})
            row.update({f'geographic_full_{j:03d}':float(v) for j,v in enumerate(full_proj)})
            record_list.append(row)
            if (i+1)%10==0 or (i+1)==len(chosen_samples):
                print(f'  {i+1}/{len(chosen_samples)} observations exported',flush=True)
    df=pd.DataFrame(record_list)
    index_path=out/'patch_embeddings.csv'
    df.to_csv(index_path,index=False,float_format='%.9g')
    report={'phase':'3.8','created_utc':datetime.now(timezone.utc).isoformat(),
            'split':split,'year':year,'subset_export':nmax is not None,
            'sample_count':len(record_list),'usable_geographic_locations':len(set(df.patch_id)),
            'full_dataset_geographic_locations':n_locations,
            'source_bands':BANDS,'source_crs':str(ref_crs),
            'reference_transform':list(ref_transform)[:6],
            'source_raster_size':[ref_height,ref_width],
            'features':stage_info, 'masked_views':int(cfg['export']['masked_views']),
            'geographic_embedding_dimensions':256,
            'spectral_residuals_enabled':bool(spectral is not None),
            'spectral_residual_pairs':SPECTRAL_PAIRS if spectral is not None else [],
            'min_valid_cell_fraction':float(cfg['export']['min_valid_cell_fraction']),
            'provenance_sha256':hashes,
            'notes':['Per-patch exports only; no gaps or overlapping patches are concealed by a fabricated AOI mosaic.',
                     'Unmasked Swin local features use the pretrained encoder out-of-distribution relative to 75%-masked pretraining.',
                     'Geographic masked embeddings replicate Phase 3.7 three-mask protocol in original dataset order.',
                     '8x8 stage-2 grid has effective stride 16 source pixels; do not mislabel it as 10m resolution.',
                     'Source imagery is normalized with train-only statistics, with invalid pixels zero-filled.',
                     'Representations are research features, NOT validated archaeological detection probabilities.',
                     'Independent geographic test was already audited in Phase 3.7; do not tune parameters to this export.']}
    (out/'export_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('FINISHED',out,flush=True)

if __name__=='__main__':
    main()
