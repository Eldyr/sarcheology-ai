"""Validation mosaics, support diagnostics, seasonal persistence and candidate polygons."""
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from affine import Affine
from rasterio.features import shapes
from rasterio.warp import transform_geom
from rasterio.windows import Window
from scipy.ndimage import find_objects, label
from torch.utils.data import DataLoader

from sarchaeology.datasets.multimodal_dataset import MultiModalArchaeologyDataset
from .anomaly import METHODS, describe, fit_detector, restore
from .io import device_for, digest, read_json, save_json, write_raster
from .model import MODALITIES
from .training import to_device

SCORE_BANDS = ("ensemble_percentile", *METHODS)
WEIGHTS = tuple(f"fusion_weight_{m}" for m in MODALITIES)
QUALITY = tuple(f"quality_{m}" for m in MODALITIES)


def expand(x):
    return x.repeat(4,axis=-2).repeat(4,axis=-1)


def polygons(scores,valid,threshold,transform,crs,period,checkpoint):
    active = valid & (scores[0]>=threshold)
    ids,n = label(active)
    objects = find_objects(ids)
    features = []
    for geom,value in shapes(ids.astype("int32"),mask=active,transform=transform):
        idx = int(value)
        ys,xs = objects[idx-1]
        pixels = ids[ys,xs] == idx
        vals = scores[:,ys,xs][:,pixels]
        attrs = dict(period=period,component=idx,pixels=int(pixels.sum()),
                     area_m2=float(abs(transform.determinant)*pixels.sum()),
                     peak=float(vals[0].max()),mean=float(vals[0].mean()),checkpoint_sha256=checkpoint)
        for band,values in zip(WEIGHTS+QUALITY,vals[len(SCORE_BANDS):]):
            attrs[band] = float(values.mean())
        features.append(dict(type="Feature",geometry=transform_geom(crs,"EPSG:4326",geom),properties=attrs))
    return features


def calculate_period(model,detector,ds,indices,cfg,device,origin,shape):
    n_bands = len(SCORE_BANDS+WEIGHTS+QUALITY)
    sums = np.zeros((n_bands,*shape),np.float32)
    counts = np.zeros(shape,np.int16)
    footprint = np.zeros_like(counts)
    max_modalities = np.zeros_like(counts)
    embeddings = []
    for raw in DataLoader(torch.utils.data.Subset(ds,indices),batch_size=cfg["batch_size"]):
        batch = to_device(raw,device)
        x,extra,valid,weights,quality = describe(model,batch,cfg["min_modalities"])
        fine = np.stack([raw["sar_count"].numpy()>0,raw["optical_valid"].numpy(),raw["dem_valid"].numpy()],1)
        scores = np.zeros((*valid.shape,len(SCORE_BANDS)),np.float32)
        if valid.any():
            scores[valid] = detector.scores(x[valid],extra[valid])
        layers = np.concatenate([scores.transpose(0,3,1,2),weights,quality],1)
        for i in range(len(valid)):
            if valid[i].any():
                embeddings.append(dict(patch_id=raw["patch_id"][i],period=raw["period"][i],
                                       valid_cells=int(valid[i].sum()),embedding=x[i][valid[i]].mean(0).tolist()))
            row,col = int(raw["row"][i])-origin[0],int(raw["col"][i])-origin[1]
            window = np.s_[row:row+128,col:col+128]
            active = fine[i].reshape(3,32,4,32,4).mean((2,4))>=0.5
            available = expand(active) & fine[i]
            support = (expand(valid[i]) & (~expand(active) | fine[i]).all(0))
            sums[:,window[0],window[1]] += expand(layers[i])*support
            counts[window] += support
            footprint[window] += 1
            max_modalities[window] = np.maximum(max_modalities[window],available.sum(0))
    result = sums/counts.clip(1)
    result[:,counts==0] = np.nan
    coverage = np.stack([footprint,counts,max_modalities]).astype("float32")
    coverage[:,footprint==0] = np.nan
    return result,coverage,embeddings


def control_proxy(path,shape,transform,crs,origin,score,threshold):
    """Observed flag rates on vegetation and water-like spectra without field labels."""
    with rasterio.open(path) as src:
        if src.crs != crs or src.transform*Affine.translation(origin[1],origin[0]) != transform:
            raise ValueError("Control composite grid mismatch")
        bands = list(src.descriptions)
        idx = [bands.index(b)+1 for b in ("B03","B04","B08")]
        green,red,nir = src.read(idx,window=Window(origin[1],origin[0],shape[1],shape[0]),
                                 masked=True).astype("float32").filled(np.nan)
    valid = np.isfinite(score)
    ndvi = (nir-red)/np.maximum(nir+red,1e-6)
    ndwi = (green-nir)/np.maximum(green+nir,1e-6)
    groups = {"dense_vegetation":ndvi>0.65,"water_like":(ndwi>0.3)&(ndvi<0.1)}
    result = {}
    for k,m in groups.items():
        n = int((m & valid).sum())
        count = int((m & valid & (score>=threshold)).sum())
        result[k] = dict(pixels=n,flagged=count,flag_rate=count/n if n else None)
    return result


def export(prepared,model_dir,out,cfg):
    prepared,out = Path(prepared),Path(out)
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    device = device_for(cfg["device"])
    model,run = restore(model_dir,prepared,device)
    train,val = [MultiModalArchaeologyDataset(prepared,s) for s in ("train","val")]
    detector,cal = fit_detector(model,train,cfg,device,out)
    qa = val.qa
    origin = (int(val.rows.row.min()),int(val.rows.col.min()))
    shape = (int(val.rows.row.max())+128-origin[0],int(val.rows.col.max())+128-origin[1])
    transform = Affine(*qa["transform"])*Affine.translation(origin[1],origin[0])
    crs = qa["crs"]
    hsh = run["checkpoint_sha256"]
    tags = dict(checkpoint_sha256=hsh,split="val",score_stride_source_pixels="4",
                threshold=str(detector.threshold),fusion_variant=model.variant)
    meta = read_json(qa["config"]["metadata"])
    composites = {x["quarter"]:x["path"] for x in meta["optical_sources"]}
    features,period_scores,controls,counts = [],[],{},{}
    embeddings = []
    for period,group in val.rows.groupby("period"):
        values,coverage,patch_vectors = calculate_period(model,detector,val,group.index,cfg,device,origin,shape)
        embeddings.extend(patch_vectors)
        valid = np.isfinite(values[0])
        period_scores.append(values[0])
        features.extend(polygons(values,valid,detector.threshold,transform,crs,period,hsh))
        write_raster(out/f"{period}_anomalies.tif",values,crs,transform,SCORE_BANDS+WEIGHTS+QUALITY,period=period,**tags)
        write_raster(out/f"{period}_coverage.tif",coverage,crs,transform,
                     ("footprints","scored_windows","max_supported_modalities"),period=period,**tags)
        if period in composites:
            controls[period] = control_proxy(composites[period],shape,transform,crs,origin,values[0],detector.threshold)
        counts[period] = dict(valid_pixels=int(valid.sum()),flagged_pixels=int((values[0]>=detector.threshold).sum()))
    scores = np.stack(period_scores)
    observed = np.isfinite(scores).sum(0)
    hits = (scores>=detector.threshold).sum(0)
    persistence = np.stack([np.where(observed>=2,hits/observed.clip(1),np.nan),
                            np.where(observed>0,hits,np.nan),np.where(observed>0,observed,np.nan)])
    write_raster(out/"persistence.tif",persistence,crs,transform,("flagged_fraction","flagged_quarters","observed_quarters"),**tags)
    save_json(out/"candidates.geojson",dict(type="FeatureCollection",features=features))
    pd.DataFrame(embeddings).to_parquet(out/"validation_embeddings.parquet",index=False)
    save_json(out/"provenance.json",dict(split="val",config=cfg,model=run,alignment_sha256=run["qa_hash"],
              detector_sha256=digest(out/"detector.npz"),upstream=qa["provenance"],calibration=cal,
              periods=counts,control_proxies=controls,polygons=len(features),crs=crs,
              display_pixel_m=abs(transform.a),score_cell_m=abs(transform.a)*4,
              polygon_crs="OGC:CRS84",threshold_units="mean of five training-calibrated percentile ranks"))
