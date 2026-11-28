"""Join Phase 1 footprints, Phase 2 SAR and Phase 3.8 optical exports."""
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.transform import array_bounds
import torch
import yaml
from torch.nn import functional as F

from sarchaeology.models.sar_cae import build_sar_cae
from sarchaeology.preprocessing.sar_normalization import SARNormalizer
from .io import device_for, digest, read_json, save_json
from .terrain import prepare_terrain


def load_splits(cfg, meta, transform):
    splits = {}
    for name in ("train", "val", "test", "excluded"):
        frame = pd.read_parquet(Path(cfg["split_dir"])/f"{name}.parquet")
        if not (frame.split == name).all() or not frame.patch_id.is_unique:
            raise ValueError(f"Unexpected or duplicate split labels in {name}")
        for p in frame.itertuples():
            if (p.height != 128 or p.width != 128 or p.row < 0 or p.col < 0
                    or p.row+128 > meta["reference_height"] or p.col+128 > meta["reference_width"]
                    or p.crs != meta["crs"]):
                raise ValueError(f"Invalid footprint {p.patch_id}")
            bounds = array_bounds(128, 128, transform*Affine.translation(p.col, p.row))
            if not np.allclose(bounds, [p.xmin, p.ymin, p.xmax, p.ymax], rtol=0, atol=1e-5):
                raise ValueError(f"Bounds mismatch for {p.patch_id}")
        splits[name] = frame
    if pd.concat(splits.values()).patch_id.duplicated().any():
        raise ValueError("Patch ID assigned to multiple splits")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        delta = np.abs(splits[a][["row","col"]].to_numpy()[:,None] -
                       splits[b][["row","col"]].to_numpy()[None]) - 128
        if ((delta < 0).all(axis=2).any() or
                (np.linalg.norm(delta.clip(0), axis=2) < cfg["buffer_pixels"]).any()):
            raise ValueError(f"Geographic overlap or buffer violation: {a}/{b}")
    return splits


def optical_index(cfg, split, frame, meta):
    path = Path(cfg["optical_exports"])/f"{split}_{cfg['year']}"
    report = read_json(path/"export_manifest.json")
    if (report["split"] != split or report["year"] != cfg["year"] or report["subset_export"]
            or report["source_crs"] != meta["crs"]
            or report["reference_transform"] != meta["reference_transform"]
            or report["source_raster_size"] != [meta["reference_height"], meta["reference_width"]]):
        raise ValueError(f"Incompatible optical export {path}")
    df = pd.read_csv(path/"patch_embeddings.csv")
    if (len(df) != report["sample_count"] or not (df.split == split).all()
            or not (df.year == cfg["year"]).all() or not df.quarter.isin([1,2,3,4]).all()
            or df.duplicated(["patch_id","period"]).any() or not df.patch_id.isin(frame.patch_id).all()):
        raise ValueError(f"Unexpected optical split membership or quarter in {path}")
    coords = frame.set_index("patch_id")
    columns = [f"geographic_masked_{i:03d}" for i in range(256)]
    result = {}
    for i, p in enumerate(df.itertuples()):
        period = f"{p.year}Q{p.quarter}"
        folder = path / p.folder.replace("\\", "/")
        if (p.period != period or (p.row, p.col) != tuple(coords.loc[p.patch_id, ["row","col"]])
                or folder.resolve() != (path/period/f"patch_{p.patch_id}").resolve()):
            raise ValueError(f"Optical window or period mismatch: {p.patch_id}")
        result[p.patch_id, period] = folder, df.loc[i, columns].to_numpy(dtype="float32")
    return result, report["provenance_sha256"]


def optical_arrays(entry, transform, crs, pid, period, split, swin_hash):
    maps = []
    support = np.zeros((128,128), bool)
    embedding = np.zeros(256, np.float32)
    if entry is not None:
        folder, embedding = entry
        if not np.isfinite(embedding).all():
            raise ValueError("Nonfinite optical embedding")
    for stride, channels, name in [(1,1,"valid_pixels_10m"), (4,96,"swin_stage0_unmasked"),
                                    (8,192,"swin_stage1_unmasked")]:
        size = 128//stride
        if entry is None:
            if stride > 1:
                maps.append(np.zeros((channels,size,size), np.float16))
            continue
        with rasterio.open(folder/f"{name}.tif") as src:
            tags = src.tags()
            if (src.shape != (size,size) or src.count != channels or src.crs != crs
                    or src.transform != transform*Affine.scale(stride)
                    or any(tags.get(k) != v for k,v in dict(patch_id=pid, period=period,
                           split=split, swin_sha256=swin_hash).items())):
                raise ValueError(f"Optical raster identity/grid mismatch: {src.name}")
            arr = src.read()
            valid = (src.dataset_mask() != 0) & np.isfinite(arr).all(0)
            if stride == 1:
                support = valid & (arr[0] > 0)
            else:
                support &= valid.repeat(stride,0).repeat(stride,1)
                maps.append(np.where(valid,arr,0).astype("float16"))
    return dict(stage0=maps[0], stage1=maps[1], optical_embedding=embedding, optical_valid=support)


@torch.no_grad()
def quarter_sar(sar, masks, indices, encoder, normalizer, device, min_fraction):
    keep = [i for i in indices if masks[i].any() and masks[i].mean() >= min_fraction]
    counts = np.zeros((128,128), np.uint8)
    sums = np.zeros((2,128,128), np.float32)
    total = torch.zeros((128,16,16), device=device)
    weights = torch.zeros((1,16,16), device=device)
    for start in range(0, len(keep), 4):
        ix = keep[start:start+4]
        masks_b = masks[ix]
        counts += masks_b.sum(0).astype("uint8")
        sums += np.where(masks_b[:,None], sar[ix], 0).sum(0)
        mask = torch.from_numpy(masks_b[:,None]).to(device)
        x = torch.from_numpy(sar[ix]).to(device)
        x = torch.stack([normalizer(v,m) for v,m in zip(x,mask)])
        latent, _ = encoder.encode(x, mask)
        w = F.avg_pool2d(mask.float(), 8)
        total += (latent*w).sum(0)
        weights += w.sum(0)
    return dict(sar_db=sums/counts.clip(1), sar_count=counts,
                sar_features=(total/weights.clamp_min(1e-8)).cpu().numpy().astype("float16")), keep


def unified_manifest(development, heldout):
    keys = ["patch_id","period","row","col"]
    reserved = development[development.split=="test"]
    def identities(frame):
        return set(frame[keys].itertuples(index=False,name=None))
    if (identities(reserved) != identities(heldout) or len(reserved) != len(heldout)
            or heldout.duplicated(keys).any() or not (heldout.split=="test").all()):
        raise ValueError("Held-out observations differ from reserved development footprints")
    dev = development[development.split!="test"].copy()
    dev["cache_group"] = np.where(dev.status=="ready","development","excluded")
    return pd.concat([dev,heldout.assign(cache_group="heldout")],ignore_index=True)


def prepare(cfg, out, *, frozen_data=None, model_dir=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    heldout = frozen_data is not None
    if heldout:
        frozen_data = Path(frozen_data)
        original = read_json(frozen_data/"alignment_qa.json")
        run = read_json(Path(model_dir)/"run.json")
        if (run["qa_hash"] != digest(frozen_data/"alignment_qa.json")
                or run["checkpoint_sha256"] != digest(Path(model_dir)/"best.pt")
                or run["fit_split"] != "train" or run["selection_split"] != "val"):
            raise ValueError("Held-out alignment requires a completed development run")
        cfg = original["config"]
    meta = read_json(cfg["metadata"])
    transform = Affine(*meta["reference_transform"])
    shape = (meta["reference_height"], meta["reference_width"])
    crs = meta["crs"]
    splits = load_splits(cfg, meta, transform)
    dates = pd.to_datetime([s["date"] for s in meta["sar_sources"]])
    indices = {q: np.flatnonzero((dates.year == cfg["year"]) & (dates.quarter == q)).tolist()
               for q in range(1,5)}
    for q in indices.values():
        for i in q:
            with rasterio.open(meta["sar_sources"][i]["path"]) as src:
                if src.crs != crs or src.transform != transform or src.shape != shape or src.count != 2:
                    raise ValueError(f"SAR source grid mismatch: {src.name}")
    if heldout:
        stats = read_json(frozen_data/"terrain_statistics.json")
        with rasterio.open(frozen_data/"terrain.tif") as src:
            if (src.crs != crs or src.transform != transform or src.shape != shape
                    or list(src.descriptions) != stats["bands"] or stats["fit_split"] != "train"):
                raise ValueError("Frozen terrain geometry or normalization mismatch")
            terrain = src.read(masked=True).filled(np.nan)
        dem_valid = np.isfinite(terrain).all(0)
        mean = np.asarray(stats["mean"],dtype="float32")[:,None,None]
        scale = np.asarray(stats["scale"],dtype="float32")[:,None,None]
        dem = np.nan_to_num(((terrain-mean)/scale).clip(-8,8))
    else:
        dem, dem_valid, stats = prepare_terrain(cfg, transform, crs, shape, splits["train"], out)
    save_json(out/"terrain_statistics.json", stats)
    device = device_for(cfg["device"])
    checkpoint = torch.load(cfg["sar_checkpoint"], map_location="cpu", weights_only=False)
    sar_hash = digest(cfg["sar_statistics"])
    split_hashes = {s:digest(Path(cfg["split_dir"])/f"{s}.parquet") for s in splits}
    if (sar_hash != checkpoint["normalization_sha256"] or
            read_json(cfg["sar_statistics"])["source_manifest_sha256"] != split_hashes["train"]):
        raise ValueError("SAR checkpoint / train statistics provenance mismatch")
    normalizer = SARNormalizer.from_json(cfg["sar_statistics"])
    encoder = build_sar_cae(yaml.safe_load(Path(cfg["sar_config"]).read_text()))
    encoder.load_state_dict(checkpoint["model_state"])
    encoder.requires_grad_(False).eval().to(device)
    records = []
    provenance = dict(sar_checkpoint=digest(cfg["sar_checkpoint"]), sar_statistics=sar_hash,
                      split_manifests=split_hashes, dem=digest(cfg["dem"]), metadata=digest(cfg["metadata"]))
    if heldout:
        if any(original["provenance"][k] != v for k,v in provenance.items()):
            raise ValueError("Upstream lineage changed since development")
        provenance["optical"] = original["provenance"]["optical"]
    for split, frame in splits.items():
        if heldout and split != "test":
            continue
        active = heldout or split in ("train","val")
        if active:
            optical, hashes = optical_index(cfg, split, frame, meta)
            if provenance.setdefault("optical", hashes) != hashes:
                raise ValueError("Train and validation optical weights differ")
            (out/split).mkdir()
        for j,p in enumerate(frame.itertuples()):
            window = np.s_[p.row:p.row+128,p.col:p.col+128]
            if active:
                if Path(p.path).stem != p.patch_id:
                    raise ValueError("SAR archive identity mismatch")
                with np.load(p.path, allow_pickle=False) as source:
                    sar, masks = source["sar"], source["sar_valid"].astype(bool)
                if (sar.shape != (len(dates),2,128,128) or masks.shape != (len(dates),128,128)
                        or not np.array_equal(np.isfinite(sar).all(1), masks)):
                    raise ValueError(f"SAR archive mask or dimensions disagree: {p.path}")
            for q, ix in indices.items():
                period = f"{cfg['year']}Q{q}"
                rec = dict(split=split, patch_id=p.patch_id, row=p.row, col=p.col, period=period,
                           sar_indices=ix, sar_archive=p.path, cache="", status="reserved")
                if active:
                    arrays, keep = quarter_sar(sar,masks,ix,encoder,normalizer,device,cfg["sar_min_valid_fraction"])
                    arr = optical_arrays(optical.get((p.patch_id,period)), transform*Affine.translation(p.col,p.row),
                                         crs,p.patch_id,period,split,hashes["swin"])
                    filename = f"{split}/{p.patch_id}_{period}.npz"
                    np.savez(out/filename, **arrays, **arr, dem=dem[:,window[0],window[1]].astype("float16"),
                             dem_valid=dem_valid[window])
                    rec.update(cache=filename, status="ready", sar_used=keep, sar_observations=len(keep),
                               optical_available=(p.patch_id,period) in optical,
                               sar_coverage=float((arrays["sar_count"]>0).mean()),
                               optical_coverage=float(arr["optical_valid"].mean()),
                               dem_coverage=float(dem_valid[window].mean()))
                records.append(rec)
            if active and (j+1)%40 == 0:
                print(f"{split}: prepared {j+1}/{len(frame)} locations", flush=True)
    frame = pd.DataFrame(records)
    frame.to_parquet(out/"manifest.parquet",index=False)
    qa = dict(config=cfg, crs=crs, transform=list(transform)[:6], shape=shape, year=cfg["year"],
              split_patches={s:len(f) for s,f in splits.items()}, sar_sources=meta["sar_sources"],
              matched_acquisitions={q:len(i) for q,i in indices.items()}, provenance=provenance,
              manifest_sha256=digest(out/"manifest.parquet"), cached_rows=int((frame.status=="ready").sum()),
              mean_valid_fraction=frame[frame.status=="ready"].groupby("split")[
                  ["sar_coverage","optical_coverage","dem_coverage"]].mean().to_dict("index"),
              test_policy="Phase 1 footprint metadata only. No test observations loaded.")
    if heldout:
        if original["manifest_sha256"] != digest(frozen_data/"manifest.parquet"):
            raise ValueError("Development manifest hash mismatch")
        combined = unified_manifest(pd.read_parquet(frozen_data/"manifest.parquet"),frame)
        combined.to_parquet(out/"unified_manifest.parquet",index=False)
        qa.update(inference_only=True,frozen_model_sha256=run["checkpoint_sha256"],development_qa_sha256=run["qa_hash"],
                  unified_manifest_sha256=digest(out/"unified_manifest.parquet"),
                  terrain_statistics_sha256=digest(frozen_data/"terrain_statistics.json"),
                  test_policy="Alignment only, after model lock. No fitting, scores or evaluation.")
    save_json(out/"alignment_qa.json", qa)
