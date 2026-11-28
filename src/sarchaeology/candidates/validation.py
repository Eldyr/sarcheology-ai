"""Evaluate a frozen ranking on independently surveyed, matched site/control units."""
from pathlib import Path
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import box
from sklearn.metrics import average_precision_score,roc_auc_score

from sarchaeology.fusion.io import digest,read_json,save_json

BASELINES = ("sar_only","optical_only","dem_only","classical_mean")


def ranking_metrics(labels,scores,k):
    y,s = np.asarray(labels),np.asarray(scores)
    if set(np.unique(y))!={0,1}:
        return None
    k = min(k,len(s))
    cutoff = np.sort(s)[-k]
    higher,tied = s>cutoff,s==cutoff
    # Expected hits under a uniform ordering of tied scores.
    hits = y[higher].sum()+(k-higher.sum())*y[tied].mean()
    return dict(k=int(k),precision_at_k=float(hits/k),recall_at_k=float(hits/y.sum()),
                average_precision=float(average_precision_score(y,s)),roc_auc=float(roc_auc_score(y,s)))


def curated_units(path):
    frame = gpd.read_file(path)
    required = {"unit_id","label","landscape","block_id","matched_set","curator","source","survey_method",
                "survey_date","role","geometry"}
    if required-set(frame.columns) or frame.empty or frame.crs is None:
        raise ValueError("Reference units need the documented provenance, survey and geography fields")
    if (frame[list(required-{"geometry"})].isna().any().any() or frame.unit_id.duplicated().any()
            or not (frame.role=="independent_evaluation").all() or not frame.label.isin([0,1]).all()
            or not frame.geom_type.isin(["Polygon","MultiPolygon"]).all() or not frame.is_valid.all()
            or frame.is_empty.any()):
        raise ValueError("Invalid independent reference units or review labels")
    for col in required-{"geometry","label"}:
        if frame[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"Empty reference field {col}")
    if not (frame.groupby(["landscape","matched_set"]).label.nunique()==2).all():
        raise ValueError("Each surveyed matched set needs both known sites and verified controls")
    if (frame.groupby(["landscape","matched_set"]).block_id.nunique()>1).any():
        raise ValueError("Keep a matched site/control set within its spatial resampling block")
    indexed = frame.reset_index(drop=True)
    for i,geom in enumerate(indexed.geometry):
        for j in indexed.sindex.query(geom,predicate="intersects"):
            if j>i and geom.intersection(indexed.geometry.iloc[j]).area>0:
                raise ValueError("Independent survey units must have disjoint interiors")
    pd.to_datetime(frame.survey_date,errors="raise")
    return frame


def score_units(refs,catalogue,rasters,protected,coverage):
    refs = refs.to_crs(protected.crs)
    cats = catalogue.to_crs(protected.crs)
    forbidden = protected.geometry.union_all()
    coverage = coverage.to_crs(protected.crs).geometry.union_all()
    rows = []
    for row in refs.itertuples():
        geom = row.geometry
        if forbidden.intersects(geom):
            raise ValueError("Independent reference intersects original development/test geography")
        if not coverage.covers(geom):
            continue
        overlapping = cats[cats.intersects(geom)]
        overlapping = overlapping[overlapping.geometry.intersection(geom).area>0]
        values = dict(priority=float(overlapping.priority.max()) if len(overlapping) else 0.)
        for name in BASELINES:
            maxima = []
            for raster in rasters:
                with rasterio.open(raster) as src:
                    aligned = gpd.GeoSeries([geom],crs=refs.crs).to_crs(src.crs).iloc[0]
                    inside = geometry_mask([aligned.__geo_interface__],src.shape,src.transform,invert=True)
                    band = src.read(src.descriptions.index(name)+1,masked=True)[inside]
                    if len(band) and band.count()/len(band)>=0.8:
                        maxima.append(float(band.max()))
            values[name] = max(maxima) if maxima else None
        rows.append(dict(unit_id=row.unit_id,landscape=row.landscape,block_id=row.block_id,label=int(row.label),
                         matched_set=row.matched_set,area_m2=geom.area,**values))
    return rows


def summarize_validation(frame,k,seed,draws=500):
    complete = frame.dropna(subset=["priority",*BASELINES]).copy()
    # Keep matching intact after coverage screening.
    complete = complete.groupby(["landscape","matched_set"],group_keys=False).filter(lambda g:g.label.nunique()==2)
    if complete.empty:
        return dict(status="insufficient_common_coverage",units=0)
    methods = ("priority",*BASELINES)
    scores = {name:ranking_metrics(complete.label,complete[name],k) for name in methods}
    landscape_results = {key:{name:ranking_metrics(g.label,g[name],k) for name in methods}
                         for key,g in complete.groupby("landscape")}
    blocks = [g for _,g in complete.groupby(["landscape","block_id"])]
    intervals = {}
    if len(blocks)>=4:
        rng = np.random.default_rng(seed)
        samples = {name:[] for name in methods}
        for _ in range(draws):
            g = pd.concat([blocks[i] for i in rng.integers(len(blocks),size=len(blocks))])
            for name in methods:
                result = ranking_metrics(g.label,g[name],k)
                if result:
                    samples[name].append([result[m] for m in ("precision_at_k","recall_at_k","average_precision","roc_auc")])
        for name,a in samples.items():
            if a:
                intervals[name] = dict(zip(("precision_at_k","recall_at_k","average_precision","roc_auc"),
                                          np.quantile(a,[.025,.975],axis=0).T.tolist()))
    return dict(status="evaluated",units=len(complete),sites=int(complete.label.sum()),blocks=len(blocks),
                positive_priority_site_recall=float((complete.loc[complete.label==1,"priority"]>0).mean()),
                positive_priority_control_flag_rate=float((complete.loc[complete.label==0,"priority"]>0).mean()),
                metrics=scores,by_landscape=landscape_results,spatial_block_95pct=intervals,
                scope="Surveyed matched units with common coverage. Recall is conditional on this reference set.")


def evaluate_independent(studies,prepared,policy,model_sha,out,k=10):
    if k<1:
        raise ValueError("K must be positive")
    out.mkdir(parents=True,exist_ok=False)
    if not studies:
        result = dict(status="awaiting_independent_surveys",units=0,metrics=None,
                      requirements=["Surveyed site and matched non-site polygons outside the original Buto footprints",
                                    "Independent landscape catalogue, coverage and baseline rasters under the frozen policy"],
                      review_labels_used=False)
    else:
        qa = read_json(prepared/"alignment_qa.json")
        split_dir = Path(qa["config"]["split_dir"])
        frames = [pd.read_parquet(split_dir/f"{s}.parquet") for s in ("train","val","test","excluded")]
        src = pd.concat(frames)
        guard = qa["config"]["buffer_pixels"]*abs(qa["transform"][0])
        protected = gpd.GeoDataFrame(geometry=[box(r.xmin,r.ymin,r.xmax,r.ymax).buffer(guard) for r in src.itertuples()],
                                     crs=qa["crs"])
        rows,counts,inputs,locations = [],0,[],set()
        for path in map(Path,studies):
            spec = read_json(path)
            if spec["ranking_policy_sha256"]!=policy or spec["model_sha256"]!=model_sha:
                raise ValueError("Independent study uses a different scoring policy or model")
            if spec["role"]!="independent_transfer" or spec["landscape"] in locations:
                raise ValueError("Expect a distinct independent landscape per study")
            locations.add(spec["landscape"])
            resolve = lambda name:path.parent/name
            refs = curated_units(resolve(spec["references"]))
            if set(refs.landscape)!={spec["landscape"]}:
                raise ValueError("Reference and study landscapes disagree")
            cat = gpd.read_file(resolve(spec["catalogue"]))
            coverage = gpd.read_file(resolve(spec["coverage"]))
            if (cat.crs is None or coverage.crs is None or not cat.geometry.is_valid.all()
                    or cat.candidate_id.duplicated().any() or not cat.priority.between(0,100).all()):
                raise ValueError("Invalid independent candidate catalogue or coverage")
            rasters = [resolve(p) for p in spec["evidence_rasters"]]
            if not rasters:
                raise ValueError("Independent baselines require evidence rasters")
            paths = [path,resolve(spec["references"]),resolve(spec["coverage"]),resolve(spec["catalogue"]),*rasters]
            inputs.append({p.name:digest(p) for p in paths})
            rows.extend(score_units(refs,cat,rasters,protected,coverage))
            counts += len(refs)
        frame = pd.DataFrame(rows,columns=["unit_id","landscape","block_id","label","matched_set","area_m2","priority",*BASELINES])
        frame.to_csv(out/"survey_unit_scores.csv",index=False)
        result = summarize_validation(frame,k,42) if len(frame) else dict(status="insufficient_common_coverage",units=0)
        result.update(submitted_units=counts,source_hashes=inputs,review_labels_used=False)
    save_json(out/"independent_validation.json",result)
    pd.DataFrame(columns=["unit_id","label","landscape","block_id","matched_set","curator","source",
                          "survey_method","survey_date","role","geometry_wkt"]).to_csv(out/"reference_register_template.csv",index=False)
    (out/"report.md").write_text("# Independent archaeological validation\n\n"+
        ("No independent surveyed landscapes with matched site and non-site controls were provided. "
         "Precision, recall, ranking skill and geographic transfer remain unmeasured. Reviewer decisions are stored separately.\n"
         if not studies else "Results on surveyed units and block-level intervals are recorded in independent_validation.json.\n"),
         encoding="utf-8")
    return result
