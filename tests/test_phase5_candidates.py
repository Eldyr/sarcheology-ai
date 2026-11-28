"""Phase 5 regression tests use synthetic locations and independent survey fixtures."""
from pathlib import Path
from zipfile import ZipFile
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from affine import Affine
from rasterio.crs import CRS
from shapely.geometry import box

from sarchaeology.candidates.evidence import contrast,local_contrast
from sarchaeology.candidates.extraction import extract,morphology
from sarchaeology.candidates.interpretation import interpret
from sarchaeology.candidates.products import package,transform_private_package
from sarchaeology.candidates.review import feedback,save_feedback
from sarchaeology.candidates.validation import (BASELINES,curated_units,evaluate_independent,
                                                 ranking_metrics,score_units,summarize_validation)
from sarchaeology.fusion.io import save_json,write_raster

CRS_M = CRS.from_epsg(32636)
T = Affine(10,0,300000,0,-10,3500000)
EXTRACT = dict(edge_guard_m=10,group_gap_m=20,minimum_area_m2=300)
WEIGHTS = dict(magnitude=.35,persistence=.25,diversity=.2,morphology=.1,acquisition_stability=.1)


def test_extraction_preserves_gaps_and_quarter_identity():
    a = np.zeros((2,32,36),dtype="f")
    a[:,0,2:5] = 1
    a[:,10:13,10:13] = 1
    a[1,10:13,14:17] = 1
    a[:,25,25] = 1
    a[:,:,20] = np.nan
    a[:,10:13,21] = 1
    c,ids,hits,safe,audit = extract(a,.9,T,CRS_M,EXTRACT,40)
    assert len(c)==1 and c[0].geometry.area==1800
    assert not ids[:,13].any() and not hits[:,0].any()
    assert not safe[:,:,20:22].any() and audit["below_minimum_groups"]==1
    assert c[0].id==extract(a,.9,T,CRS_M,EXTRACT,40)[0][0].id
    assert not c[0].descriptors["morphology_resolved"]
    assert c[0].descriptors["morphology_score"]==0
    assert len(extract(a,2,T,CRS_M,EXTRACT,40)[0])==0


def test_metric_shape_descriptors():
    m = np.ones((6,24),bool)
    d = morphology(m,box(0,0,240,60),40)
    assert d["elongation"]==4 and d["rectangularity"]==1 and d["rotational_symmetry"]==1
    assert d["hough_lines"]>0 and d["perimeter_m"]==600 and 0<d["ellipse_iou"]<1
    with pytest.raises(ValueError,match="metre"):
        extract(np.zeros((1,30,30)),.8,T,CRS.from_epsg(4326),EXTRACT,40)


def data():
    quarters = {}
    for q in range(1,5):
        quarters[f"2024Q{q}"] = dict(hit_fraction=.8 if q<=2 else 0,ensemble_percentile=.95,
                  quality_sar=.9,quality_optical=1,quality_dem=1,clear_observations=5,
                  **{b+"_error_contrast":.04 for b in ("B04","B08","B11","B12")})
    return dict(quarterly=quarters,acquisitions={f"2024-01-{d:02d}":{"vv_delta_db":2.,"vh_delta_db":1.}
                for d in range(1,7)},terrain={"relief_delta_m":.7,"valid_fraction":1},
                optical_history={p:dict(ndvi=.12,ndmi=.05,ndmi_level=float(p[-1])*.1) for p in quarters},
                precipitation={"2024Q1":{"mm":60},"2024Q4":{"mm":2}})


def test_ranking_provenance_and_missingness():
    d = data()
    desc = {"morphology_score":.8,"morphology_resolved":True,"elongation":1.5,"hough_lines":2}
    original = interpret(d,desc,[],WEIGHTS)
    context = [{"kind":"known_settlement","intersects":True}]
    assert interpret(d,desc,context,WEIGHTS)["priority"]==original["priority"]
    modern = interpret(d,desc,[{"kind":"road","intersects":True}],WEIGHTS)
    assert modern["priority"]<original["priority"] and "mapped_road" in modern["flags"]
    assert original["persistent_fraction"]==.5 and original["acquisition_stability"]==1
    assert original["measured_rainfall_comparison"]["wettest"]=="2024Q1"
    d["acquisitions"] = {}
    for r in d["quarterly"].values():
        for key in [k for k in r if k.endswith("_error_contrast")]:
            r[key] = None
    missing = interpret(d,desc,[],WEIGHTS)
    assert missing["evidence"]["sar"] is None and missing["evidence"]["optical"] is None
    assert missing["quality"]<original["quality"] and missing["priority"]<original["priority"]


def test_background_data_sufficiency():
    a = np.ones((12,12),dtype="f")
    m = np.zeros_like(a,bool)
    m[4:7,4:7] = True
    a[m] = 3
    assert contrast(a,m,~m)==2
    a[~m] = np.nan
    assert contrast(a,m,~m) is None
    z = local_contrast(np.full((30,30),np.nan),1)
    assert np.isnan(z).all()


def test_review_audit_conflicts_and_agreement(tmp_path):
    db = tmp_path/"reviews.sqlite"
    for reviewer,dec in [("A","plausible"),("B","rejected")]:
        save_feedback(db,"run","item",reviewer,dec,"rationale",0,{"item"})
    hist,latest,agreement = feedback(db,"run")
    assert len(hist)==2 and agreement[0]["disagreements"]==["item"]
    with pytest.raises(ValueError,match="changed"):
        save_feedback(db,"run","item","A","plausible","stale",0,{"item"})
    save_feedback(db,"run","item","A","rejected","updated",1,{"item"})
    hist,latest,agreement = feedback(db,"run")
    assert len(hist)==3 and len(latest)==2 and agreement[0]["agreement"]==1
    assert feedback(db,"other")[0].empty
    with pytest.raises(ValueError):
        save_feedback(db,"run","unknown","A","plausible","",0,{"item"})


def test_public_release_is_an_allowlist(tmp_path):
    save_json(tmp_path/"summary.json",dict(candidate_count=2,quarters=["2024Q1"],validation_status="pending",
                                           secret_coordinates=[30.1,31.2],private_note="private-data"))
    (tmp_path/"ranked.csv").write_text("private-data")
    path = tmp_path/"public.zip"
    package(tmp_path,path)
    with ZipFile(path) as z:
        assert z.namelist()==["summary.json"] and "private-data" not in z.read("summary.json").decode()
        assert "secret_coordinates" not in json.loads(z.read("summary.json"))


def test_restricted_release_requires_correct_key(tmp_path):
    from cryptography.fernet import Fernet,InvalidToken
    src,key,cipher,restored = [tmp_path/n for n in ("private.zip","secret.key","sealed","restored.zip")]
    src.write_bytes(b"private geospatial product")
    key.write_bytes(Fernet.generate_key())
    transform_private_package(src,key,cipher)
    assert src.read_bytes() not in cipher.read_bytes()
    transform_private_package(cipher,key,restored,decrypt=True)
    assert src.read_bytes()==restored.read_bytes()
    key.write_bytes(Fernet.generate_key())
    with pytest.raises(InvalidToken):
        transform_private_package(cipher,key,tmp_path/"wrong.zip",decrypt=True)
    assert not (tmp_path/"wrong.zip").exists()


def test_independent_matching_coverage_and_metrics(tmp_path):
    score_tif = tmp_path/"baselines.tif"
    a = np.zeros((4,64,96),dtype="f")
    a[:,:,:40] = 2
    write_raster(score_tif,a,CRS_M,T,BASELINES)
    geoms = [box(300010,3499500,300180,3499700),box(300520,3499500,300690,3499700)]
    refs = gpd.GeoDataFrame(dict(unit_id=["a","b"],label=[1,0],matched_set="m",block_id="n",
                landscape="external",curator="surveyor",source="field register",survey_method="survey",
                survey_date="2018-01-01",role="independent_evaluation"),geometry=geoms,crs=CRS_M)
    file = tmp_path/"units.geojson"
    refs.to_crs(4326).to_file(file)
    assert len(curated_units(file))==2
    cat = gpd.GeoDataFrame(dict(priority=[84]),geometry=[geoms[0]],crs=CRS_M)
    guard = gpd.GeoDataFrame(geometry=[box(0,0,50,50)],crs=CRS_M)
    cov = gpd.GeoDataFrame(geometry=[box(300000,3499360,300960,3500000)],crs=CRS_M)
    rows = score_units(refs,cat,[score_tif],guard,cov)
    result = summarize_validation(pd.DataFrame(rows),1,42)
    assert result["metrics"]["priority"]["precision_at_k"]==1
    assert result["metrics"]["optical_only"]["average_precision"]==1
    assert result["spatial_block_95pct"]=={}
    with pytest.raises(ValueError,match="original"):
        score_units(refs,cat,[score_tif],cov,cov)
    refs["role"] = "expert_review"
    refs.to_crs(4326).to_file(tmp_path/"reviews.geojson")
    with pytest.raises(ValueError,match="review"):
        curated_units(tmp_path/"reviews.geojson")
    assert ranking_metrics([0,1],[0,0],1)["precision_at_k"]==.5
    assert evaluate_independent([],tmp_path,"p","m",tmp_path/"report")["metrics"] is None


def test_spatial_resampling_and_missing_controls():
    rows = []
    for i in range(5):
        for label in (0,1):
            rows.append(dict(label=label,landscape="another",block_id=i,matched_set=i,priority=80*label,
                             **{k:float(label) for k in BASELINES}))
    frame = pd.DataFrame(rows)
    r = summarize_validation(frame,3,4,draws=20)
    assert r["blocks"]==5 and r["spatial_block_95pct"]["priority"]["average_precision"]==[1.,1.]
    frame.loc[frame.label==0,"dem_only"] = np.nan
    assert summarize_validation(frame,3,4)["status"]=="insufficient_common_coverage"
