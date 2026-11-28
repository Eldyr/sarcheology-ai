"""Controlled validation comparisons and optional independent, labelled validation polygons."""
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.warp import transform_geom
from sklearn.metrics import average_precision_score, roc_auc_score

from .io import digest, read_json, save_json
from .model import MODALITIES, VARIANTS


def label_metrics(labels,export_dir,threshold):
    features = read_json(labels)["features"]
    observations = []
    for feat in features:
        prop = feat["properties"]
        if (prop["split"] != "val" or prop["label"] not in (0,1) or not prop["source"]
                or feat["geometry"]["type"] not in ("Polygon","MultiPolygon")):
            raise ValueError("Expected sourced binary labels for validation polygons")
        period = prop["period"]
        if period not in read_json(export_dir/"provenance.json")["periods"]:
            raise ValueError("Unrecognized label period")
        with rasterio.open(export_dir/f"{period}_anomalies.tif") as src:
            geom = transform_geom("EPSG:4326",src.crs,feat["geometry"])
            inside = geometry_mask([geom],src.shape,src.transform,invert=True)
            values = src.read(1,masked=True)[inside].compressed()
            observations.append(dict(id=prop["id"],period=period,label=prop["label"],source=prop["source"],
                                     pixels=len(values),score=float(values.max()) if len(values) else None))
    frame = pd.DataFrame(observations)
    if frame.empty or frame.duplicated(["id","period"]).any():
        raise ValueError("Empty or duplicate label/period records")
    valid = frame.dropna(subset=["score"])
    result = dict(polygon_quarters=len(frame),scored=len(valid),score_statistic="maximum supported score")
    if valid.label.nunique()==2:
        result.update(average_precision=float(average_precision_score(valid.label,valid.score)),
                      roc_auc=float(roc_auc_score(valid.label,valid.score)))
    for y,key in [(0,"false_positive_rate"),(1,"sensitivity")]:
        s = valid.loc[valid.label==y,"score"]
        result[key] = float((s>=threshold).mean()) if len(s) else None
    return result,frame


def summarize(frame):
    result = {}
    for k in MODALITIES:
        n = int(frame[f"n_{k}"].sum())
        result[f"n_{k}"] = n
        result[f"mse_{k}"] = float(frame[f"sse_{k}"].sum()/n) if n else None
    vals = [result[f"mse_{k}"] for k in MODALITIES]
    result["mean_mse"] = float(np.mean(vals)) if all(v is not None for v in vals) else None
    return result


def detector_comparison(maps,provenance):
    """Compare high-scoring validation area for each train-calibrated detector."""
    rows = []
    for period in provenance["periods"]:
        with rasterio.open(maps/f"{period}_anomalies.tif") as src:
            score = src.read(1,masked=True)
            valid = ~np.ma.getmaskarray(score)
            ensemble = score.data[valid] >= provenance["calibration"]["threshold"]
            for band,name in enumerate(src.descriptions[1:6],start=2):
                values = src.read(band)[valid]
                flags = values >= provenance["config"]["threshold_quantile"]
                union = int(np.count_nonzero(flags | ensemble))
                rows.append(dict(period=period,method=name,valid_display_pixels=len(values),
                                 flagged_display_pixels=int(flags.sum()),mean_percentile=float(values.mean()) if len(values) else None,
                                 jaccard_with_ensemble=float((flags & ensemble).sum()/union) if union else None))
    return rows


def validation_regions(frame,buffer):
    """Whole validation footprints in two disjoint strips with a central guard band."""
    boundary = (frame.row.min()+frame.row.max()+128)/2
    north = frame.row+128 <= boundary-buffer/2
    south = frame.row >= boundary+buffer/2
    return {"north":frame[north],"south":frame[south]}


def evaluate(runs,maps,out,prepared,labels=None,comparisons=()):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader, Subset
    import torch
    from sarchaeology.datasets.multimodal_dataset import MultiModalArchaeologyDataset
    from .anomaly import restore
    from .io import device_for
    from .training import compact_dataset, validate
    out.mkdir(parents=True,exist_ok=False)
    provenance = read_json(maps/"provenance.json")
    summary,hidden_targets,diagnostics,rows,regions = {},{},{},[],[]
    torch.set_num_threads(2)
    validation = compact_dataset(MultiModalArchaeologyDataset(prepared,"val"))
    train_data = MultiModalArchaeologyDataset(prepared,"train")
    ids = train_data.rows.patch_id.unique()
    rng = np.random.default_rng(provenance["config"]["seed"])
    selected = rng.choice(ids,min(20,len(ids)),replace=False)
    train_sample = compact_dataset(Subset(train_data,np.flatnonzero(train_data.rows.patch_id.isin(selected))))
    fig,axes = plt.subplots(1,2,figsize=(11,4.2),layout="constrained")
    folders = [(k,runs/k) for k in VARIANTS]
    folders.extend((f"{p.parent.name}/{p.name}",p) for p in comparisons)
    for variant,folder in folders:
        if not folder.exists():
            if folder in comparisons:
                raise FileNotFoundError(folder)
            continue
        run = read_json(folder/"run.json")
        if run["qa_hash"] != provenance["alignment_sha256"] or run["checkpoint_sha256"] != digest(folder/"best.pt"):
            raise ValueError("Comparison inputs or checkpoint hash differ")
        device = device_for(run["config"]["device"])
        model,_ = restore(folder,prepared,device)
        loader = DataLoader(validation,batch_size=run["config"]["batch_size"])
        validation_mse,rank,frame = validate(model,loader,device,common_support=True)
        sample_loader = DataLoader(train_sample,batch_size=run["config"]["batch_size"])
        train_mse,_,_ = validate(model,sample_loader,device,common_support=True)
        diagnostics[variant] = dict(train_sample_mse=train_mse,val_minus_train_mse=validation_mse-train_mse,**rank)
        hidden_targets[variant] = {k:summarize(frame[frame.hidden==k])[f"mse_{k}"] for k in MODALITIES}
        curve = pd.read_csv(folder/"history.csv")
        for axis,metric in zip(axes,("val_mse","effective_rank")):
            axis.plot(curve.epoch,curve[metric],label=variant,marker=".")
        frame["strip"] = (frame.row//256).astype(str)
        for by in (["hidden"],["hidden","period"],["hidden","strip"]):
            for key,g in frame.groupby(by):
                rows.append(dict(variant=variant,**dict(zip(by,key if isinstance(key,tuple) else (key,))),
                                 **summarize(g)))
        summary[variant] = summarize(frame[frame.hidden=="none"])
        buffer = provenance["config"]["buffer_pixels"]
        for region,samples in validation_regions(frame[frame.hidden=="none"],buffer).items():
            if not samples.empty:
                regions.append(dict(variant=variant,region=region,buffer_pixels=buffer,
                                    locations=int(samples.patch_id.nunique()),first_row=int(samples.row.min()),
                                    end_row_exclusive=int(samples.row.max()+128),**summarize(samples)))
    if not summary:
        raise ValueError("No trained variants found")
    for ax,title in zip(axes,("Validation target MSE","Effective representation rank")):
        ax.set(xlabel="Epoch",title=title)
        ax.legend(fontsize=8)
    fig.savefig(out/"validation_curves.png",dpi=140)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(out/"robustness.csv",index=False)
    pd.DataFrame(regions).to_csv(out/"buffered_regions.csv",index=False)
    pd.DataFrame(detector_comparison(maps,provenance)).to_csv(out/"detector_comparison.csv",index=False)
    result = dict(nominal=summary,hidden_target_mse=hidden_targets,representation_diagnostics=diagnostics,
                  train_diagnostic_patch_ids=sorted(selected.tolist()),control_proxies=provenance["control_proxies"],
                  selection="Minimum validation mean target MSE within each run",
                  geographic_uncertainty="One Buto validation strip. Overlapping patch-quarter cells are dependent.",
                  labels="No sourced archaeological/control polygons supplied")
    if labels:
        result["labels"],records = label_metrics(labels,maps,provenance["calibration"]["threshold"])
        result["label_file_sha256"] = digest(labels)
        records.to_csv(out/"label_rankings.csv",index=False)
    save_json(out/"evaluation.json",result)
    lines = ["# Buto validation results","","Each model predicts the same fixed SAR, optical and terrain targets. "
             "All comparisons use the same cells with valid targets in all three sensors before simulated dropout. "
             "Lower scores indicate better feature reconstruction. Each modality contributes equally to the mean.",
             "","| Model | SAR MSE | Optical MSE | Terrain MSE | Mean MSE |",
             "|---|---:|---:|---:|---:|"]
    for name,s in summary.items():
        lines.append("| "+name+" | "+" | ".join(f"{s[f'mse_{k}']:.4f}" if s[f'mse_{k}'] is not None
                     else "unavailable" for k in MODALITIES)+f" | {s['mean_mse']:.4f} |")
    lines += ["","## Predicting a missing sensor","","Each column hides that sensor and scores only its target.",
              "","| Model | Hidden SAR | Hidden optical | Hidden DEM |","|---|---:|---:|---:|"]
    for name,s in hidden_targets.items():
        lines.append("| "+name+" | "+" | ".join(f"{s[k]:.4f}" if s[k] is not None else "No input"
                                               for k in MODALITIES)+" |")
    lines += ["","## Representation diagnostics","","The train sample uses up to 20 seeded training locations across "
              "all available quarters. The gap is validation minus training MSE on the same target task.",
              "","| Model | Validation gap | Effective rank | Mean feature SD |","|---|---:|---:|---:|"]
    for name,s in diagnostics.items():
        lines.append(f"| {name} | {s['val_minus_train_mse']:.4f} | {s['effective_rank']:.2f} | {s['feature_std']:.3f} |")
    lines += ["","\u0060buffered_regions.csv\u0060 compares whole northern and southern validation footprints with a "
              "central exclusion band. These give two separated within-AOI estimates.",
              "", "Quarter and missing-input errors, valid-cell counts and 256-source-row latitude strips are in "
              "`robustness.csv`. Zero support gives an empty metric. Curves show epoch-wise performance and rank.",
              "", "The original validation patches form one overlap-connected region. The subregions and quarters "
              "describe variation within Buto. Independent regional confidence intervals cannot be estimated here.",
              "", "The seasonal maps use a threshold calibrated on training geography. Dense vegetation and "
              "water-like spectra are nuisance proxies. Their flag rates do not establish archaeological false-positive rates.",
              "", "Field-confirmed polygon metrics, when available, are in evaluation.json. Candidate polygons require field interpretation."]
    (out/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def verify(prepared,maps):
    from rasterio.features import rasterize
    qa = read_json(prepared/"alignment_qa.json")
    prod = read_json(maps/"provenance.json")
    if (prod["alignment_sha256"] != digest(prepared/"alignment_qa.json") or
            qa["manifest_sha256"] != digest(prepared/"manifest.parquet") or
            prod["detector_sha256"] != digest(maps/"detector.npz")):
        raise ValueError("Provenance hash mismatch")
    from affine import Affine
    data = pd.read_parquet(prepared/"manifest.parquet")
    val = data[data.split=="val"]
    transform = Affine(*qa["transform"])*Affine.translation(val.col.min(),val.row.min())
    expected_shape = (val.row.max()+128-val.row.min(),val.col.max()+128-val.col.min())
    seen = []
    for period,stats in prod["periods"].items():
        with rasterio.open(maps/f"{period}_anomalies.tif") as src, rasterio.open(maps/f"{period}_coverage.tif") as cov:
            if (src.crs != qa["crs"] or src.transform != transform or src.shape != expected_shape
                    or cov.transform != transform or cov.shape != expected_shape or cov.crs != src.crs
                    or src.tags()["checkpoint_sha256"] != prod["model"]["checkpoint_sha256"]):
                raise ValueError("Export georeferencing or provenance mismatch")
            scores = src.read(1,masked=True)
            valid = ~np.ma.getmaskarray(scores)
            if (valid.sum() != stats["valid_pixels"] or not np.array_equal(valid,cov.read(2,masked=True).filled(0)>0)
                    or not np.isfinite(scores.compressed()).all() or (scores.compressed()<0).any()
                    or (scores.compressed()>1.00001).any()):
                raise ValueError("Invalid scores or coverage mask")
            seen.append(valid)
    with rasterio.open(maps/"persistence.tif") as src:
        if not np.array_equal(src.read(3,masked=True).filled(0),np.stack(seen).sum(0)):
            raise ValueError("Quarter counts disagree with maps")
    features = read_json(maps/"candidates.geojson")["features"]
    if len(features) != prod["polygons"] or any(f["properties"]["checkpoint_sha256"] !=
            prod["model"]["checkpoint_sha256"] for f in features):
        raise ValueError("Candidate polygon provenance mismatch")
    for period in prod["periods"]:
        geometries = [transform_geom("EPSG:4326",qa["crs"],f["geometry"]) for f in features
                      if f["properties"]["period"] == period]
        burned = (rasterize(geometries,out_shape=expected_shape,transform=transform,dtype="uint8")
                  if geometries else np.zeros(expected_shape,bool))
        with rasterio.open(maps/f"{period}_anomalies.tif") as src:
            flagged = (src.read(1,masked=True) >= prod["calibration"]["threshold"]).filled(False)
        if not np.array_equal(flagged,burned):
            raise ValueError(f"{period} candidate polygons disagree with thresholded scores")
    print(f"Verified {len(seen)} quarterly maps, persistence and {len(features)} polygons.")


def verify_heldout(prepared,heldout):
    from sarchaeology.datasets.multimodal_dataset import MultiModalArchaeologyDataset
    from .prepare import unified_manifest
    ds = MultiModalArchaeologyDataset(heldout,"test",inference_only=True)
    index = heldout/"unified_manifest.parquet"
    if (ds.qa["development_qa_sha256"] != digest(prepared/"alignment_qa.json")
            or ds.qa["terrain_statistics_sha256"] != digest(prepared/"terrain_statistics.json")
            or ds.qa["unified_manifest_sha256"] != digest(index)):
        raise ValueError("Held-out lineage mismatch")
    expected = unified_manifest(pd.read_parquet(prepared/"manifest.parquet"),pd.read_parquet(heldout/"manifest.parquet"))
    keys = ["patch_id","period","split","row","col","cache_group","cache"]
    pd.testing.assert_frame_equal(expected[keys],pd.read_parquet(index)[keys])
    sizes = dict(sar_db=(2,128,128),sar_count=(128,128),sar_features=(128,16,16),
                 stage0=(96,32,32),stage1=(192,16,16),optical_embedding=(256,),dem=(9,128,128),
                 optical_valid=(128,128),dem_valid=(128,128))
    for path in ds.paths:
        with np.load(path,allow_pickle=False) as data:
            for key,shape in sizes.items():
                values = data[key]
                if values.shape != shape or not np.isfinite(values).all():
                    raise ValueError(f"Invalid held-out tensor {path.name}: {key}")
    print(f"Verified {len(ds)} held-out patch-quarters and original split membership.")
