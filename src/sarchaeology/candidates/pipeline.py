"""Build a Phase 5 candidate catalogue from frozen validation products."""
from collections import Counter
from pathlib import Path
import re
from datetime import date
import shutil

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes,rasterize
from rasterio.warp import transform_geom
import yaml

from sarchaeology.fusion.evaluation import verify as verify_phase4
from sarchaeology.fusion.io import digest,read_json,save_json,write_raster
from .evidence import EvidenceGrid,profile
from .extraction import extract
from .interpretation import Context,interpret
from .products import candidate_report,package
from .review import initialize
from .validation import evaluate_independent


def build(config,out):
    out.mkdir(parents=True,exist_ok=False)
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    if any(cfg[k]<0 for k in ("edge_guard_m","group_gap_m","minimum_area_m2")) or cfg["background_m"]<=0:
        raise ValueError("Invalid metric extraction settings")
    verify_phase4(Path(cfg["prepared"]),Path(cfg["maps"]))
    grid = EvidenceGrid(cfg)
    gis = out/"gis"
    gis.mkdir()
    reports = out/"reports"
    reports.mkdir()
    threshold = grid.prov["calibration"]["threshold"]
    candidates,labels,hits,safe,audit = extract(grid.scores,threshold,grid.transform,grid.crs,cfg,grid.prov["score_cell_m"])
    context = Context(cfg["context"],grid,out/"context")
    grid.write_layers(gis/"evidence",safe)
    source_dates = {}
    for p in grid.optical:
        path = Path(cfg["composites"])/f"{p}_median.tif"
        with rasterio.open(path) as src:
            files = src.tags().get("source_files","").split(";")
            tokens = [re.search(r"(?:^|_)(\d{8})T\d{6}(?:_|\.)",f) for f in files if f]
            if any(t is None for t in tokens):
                raise ValueError(f"Unrecognized optical acquisition timestamp in {p}")
            source_dates[p] = sorted({pd.to_datetime(t.group(1),format="%Y%m%d").date().isoformat() for t in tokens})
    support = safe.any(0)
    footprints = [{"type":"Feature","properties":{"landscape":cfg["landscape"]},"geometry":transform_geom(grid.crs,"EPSG:4326",g)}
                  for g,_ in shapes(support.astype("uint8"),mask=support,transform=grid.transform)]
    save_json(gis/"surveyable_coverage.geojson",dict(type="FeatureCollection",features=footprints))
    for i,p in enumerate(grid.periods):
        shutil.copyfile(grid.maps/f"{p}_anomalies.tif",gis/f"{p}_anomalies.tif")
        valid = np.isfinite(grid.scores[i])
        status = np.where(safe[i],0.,1.)
        status[hits[i] & (labels==0)] = 2
        status[hits[i] & (labels>0)] = 3
        status[~valid] = np.nan
        write_raster(gis/f"{p}_screen.tif",status[None],grid.crs,grid.transform,["screen_code"],
                     legend="0 eligible background; 1 edge guard; 2 undersized group; 3 retained exceedance")
        if p in grid.indices:
            ndvi,ndwi = [grid.indices[p][k] for k in ("ndvi","ndwi")]
            cover = np.where(ndvi>.65,1,np.where((ndwi>.3)&(ndvi<.1),2,3)).astype("float32")
            cover[~safe[i]|~np.isfinite(ndvi)|~np.isfinite(ndwi)] = np.nan
            write_raster(gis/f"{p}_surface_context.tif",cover[None],grid.crs,grid.transform,
                         ["spectral_surface_proxy"],legend="1 dense vegetation; 2 water-like; 3 other observed")
    shutil.copyfile(grid.maps/"persistence.tif",gis/"phase4_persistence.tif")
    model_hash = grid.prov["model"]["checkpoint_sha256"]
    policy = {k:cfg[k] for k in ("edge_guard_m","group_gap_m","minimum_area_m2","background_m","priority_weights")}
    policy["implementation"] = {name:digest(Path(__file__).with_name(name+".py")) for name in ("extraction","interpretation","evidence")}
    save_json(out/"policy.json",policy)
    policy_hash = digest(out/"policy.json")
    records,rows,geometry = {},[],{}
    for c in candidates:
        data = profile(c,grid,labels,hits,safe,cfg)
        contacts = context.describe(c)
        ranking = interpret(data,c.descriptors,contacts,cfg["priority_weights"])
        records[c.id] = dict(candidate_id=c.id,profile=data,morphology=c.descriptors,ranking=ranking,context=contacts,
                             provenance=dict(fusion_sha256=model_hash,spectral_sha256=getattr(grid,"spectral_hash",None),
                                             policy_sha256=policy_hash,optical_observation_dates=source_dates))
        geometry[c.id] = transform_geom(grid.crs,"EPSG:4326",c.geometry.__geo_interface__)
        rows.append(dict(candidate_id=c.id,priority=ranking["priority"],quality=ranking["quality"],
                         persistence=ranking["persistent_fraction"],sar_dates=ranking["sar_dates"],
                         screening=ranking["review_status"],reasons="|".join(ranking["flags"]),
                         **c.descriptors,**{f"{k}_strength":v for k,v in ranking["evidence"].items()}))
    rows.sort(key=lambda r:(-r["priority"],r["candidate_id"]))
    catalogue = []
    for i,row in enumerate(rows,1):
        row["rank"] = i
        cid = row["candidate_id"]
        records[cid]["rank"] = i
        catalogue.append(dict(type="Feature",geometry=geometry[cid],properties=row))
    save_json(gis/"catalogue.geojson",dict(type="FeatureCollection",features=catalogue))
    pd.DataFrame(rows,columns=(list(rows[0]) if rows else ["rank","candidate_id","priority"])).to_csv(out/"ranked.csv",index=False)
    for c in candidates:
        candidate_report(reports,grid,c,records[c.id])
    initialize(out/"feedback"/"expert.sqlite")
    audit["flags"] = dict(Counter(f for r in records.values() for f in r["ranking"]["flags"]))
    save_json(out/"screening.json",audit)
    validation = evaluate_independent([],Path(cfg["prepared"]),policy_hash,model_hash,out/"validation")
    links = "".join(f'<tr><td>{row["rank"]}</td><td><a href="reports/{row["candidate_id"]}/report.html">'
                    f'{row["candidate_id"]}</a></td><td>{row["priority"]:.1f}</td><td>{row["area_m2"]:,.0f}</td>'
                    f'<td>{row["screening"]}</td></tr>' for row in rows)
    (out/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>Candidate catalogue</title>'
        '<style>body{font:16px system-ui;max-width:950px;margin:2em auto}td,th{padding:.4em 1em;text-align:left}</style>'
        '<h1>Archaeological candidate review catalogue</h1><p>Restricted research copy. Each link opens imagery, '
        'measurements and the ranking rationale for one cross-quarter footprint.</p><table><tr><th>Rank</th>'
        '<th>Candidate</th><th>Priority / 100</th><th>Area (m²)</th><th>Screening</th></tr>'+links+'</table>',encoding="utf-8")
    save_json(out/"summary.json",dict(candidate_count=len(rows),quarters=grid.periods,validation_status=validation["status"],
                                      landscape=cfg["landscape"],reviewed_at_creation=0))
    save_json(out/"provenance.json",dict(config=cfg,policy_sha256=policy_hash,model_sha256=model_hash,
              spectral_sha256=getattr(grid,"spectral_hash",None),catalogue_sha256=digest(gis/"catalogue.geojson"),
              phase4_provenance_sha256=digest(grid.maps/"provenance.json"),source_sha256=grid.sources,
              historical_registry_sha256=digest(cfg["context"]),optical_dates=source_dates,
              sar_dates=sorted(grid.sar),threshold=threshold,split="val",public_release="aggregate only"))
    package(out,out/"public_summary.zip")
    package(out,out/"restricted_research.zip",restricted=True)
    print(f"Built {len(rows)} cross-quarter candidates from {len(grid.periods)} validation maps.",flush=True)


def verify_products(run):
    import json
    from zipfile import ZipFile
    prov = read_json(run/"provenance.json")
    if (digest(run/"gis/catalogue.geojson")!=prov["catalogue_sha256"] or
            digest(run/"policy.json")!=prov["policy_sha256"]):
        raise ValueError("Catalogue or policy digest mismatch")
    cat = read_json(run/"gis/catalogue.geojson")["features"]
    rows = pd.read_csv(run/"ranked.csv")
    if (len(rows)!=len(cat) or rows.candidate_id.duplicated().any() or list(rows["rank"])!=list(range(1,len(rows)+1))
            or not rows.priority.is_monotonic_decreasing):
        raise ValueError("Invalid ranked catalogue")
    for i,f in enumerate(cat):
        cid = f["properties"]["candidate_id"]
        data = read_json(run/"reports"/cid/"evidence.json")
        rank = data["ranking"]
        computed = np.clip(sum(rank["contributions"].values())-rank["penalty_points"],0,100)
        if (cid!=rows.candidate_id.iloc[i] or not np.isclose(computed,rows.priority.iloc[i])
                or data["rank"]!=i+1 or not (run/"reports"/cid/"report.html").is_file()):
            raise ValueError("Candidate identity, ranking or report mismatch")
        for period,dates in data["provenance"]["optical_observation_dates"].items():
            if any(str(pd.Period(date.fromisoformat(d),freq="Q"))!=period for d in dates):
                raise ValueError("Optical observation dates differ from the composite quarter")
        for p in data["profile"]["quarterly"]:
            if not (run/"reports"/cid/f"{p}.png").is_file():
                raise ValueError("Missing candidate imagery")
    kept = None
    for period in read_json(run/"summary.json")["quarters"]:
        with rasterio.open(run/"gis"/f"{period}_screen.tif") as src:
            current = src.read(1,masked=True).filled(-1)==3
            if kept is None:
                kept = np.zeros_like(current)
                geoms = [(transform_geom("EPSG:4326",src.crs,f["geometry"]),i) for i,f in enumerate(cat,1)]
                burned = rasterize(geoms,out_shape=src.shape,transform=src.transform,dtype="int32") if geoms else np.zeros(src.shape,"int32")
                pixel_area = abs(src.transform.determinant)
            kept |= current
    if not np.array_equal(kept,burned>0):
        raise ValueError("Catalogue differs from the union of retained quarterly pixels")
    if len(rows) and not np.allclose(np.bincount(burned.ravel(),minlength=len(rows)+1)[1:]*pixel_area,rows.area_m2):
        raise ValueError("Polygon areas differ from retained anomaly pixels")
    with ZipFile(run/"public_summary.zip") as z:
        if z.namelist()!=["summary.json"] or set(json.loads(z.read("summary.json")))!={
            "candidate_count","quarters","status","validation","sharing"}:
            raise ValueError("Public release differs from coordinate-free schema")
    with ZipFile(run/"restricted_research.zip") as z:
        if z.testzip() or any("feedback" in p for p in z.namelist()):
            raise ValueError("Restricted package integrity failure")
    print(f"Verified {len(rows)} candidate records, ranking rationales, reports and both releases.")
