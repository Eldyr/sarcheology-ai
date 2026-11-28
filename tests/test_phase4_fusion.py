"""Phase 4 contracts: synthetic geography, missing sensors, training, scoring and GIS round trip."""
import numpy as np
import pandas as pd
import pytest
import rasterio
import torch
from affine import Affine
from torch.utils.data import DataLoader

from sarchaeology.datasets.multimodal_dataset import MultiModalArchaeologyDataset, QuarterPairSampler
from sarchaeology.fusion.anomaly import partition
from sarchaeology.fusion.evaluation import evaluate, label_metrics, validation_regions, verify
from sarchaeology.fusion.export import export
from sarchaeology.fusion.io import digest, read_json, save_json, write_raster
from sarchaeology.fusion.model import LocalFusion, VARIANTS
from sarchaeology.fusion.prepare import load_splits, optical_arrays, quarter_sar, unified_manifest
from sarchaeology.fusion.terrain import derivatives, prepare_terrain
from sarchaeology.fusion.training import geo_negatives, train_all

GRID = Affine(10,0,280000,0,-10,3460000)


def test_native_derivatives_and_holes():
    z = np.indices((40,40))[1].astype(float)*30
    z[9,9] = np.nan
    d = derivatives(z,30)
    assert np.isnan(d[:,9:15,9:15]).all()
    assert np.allclose(d[[1,2,3,4,6,7],24,24],[45,-1,0,0,0,0],atol=1e-5)
    assert d[5,24,24]>0
    assert 0<=d[-1,24,24]<=1


def test_terrain_statistics_ignore_outside_train(tmp_path):
    crs = "EPSG:32636"
    z = np.indices((128,128))[1].astype("float32")
    cfg = dict(dem=tmp_path/"dem.tif",dem_mask=tmp_path/"mask.tif",dem_metadata=tmp_path/"dem.parquet")
    pd.DataFrame([dict(source_collection="fixture",source_item_ids='["tile"]',vertical_datum="EGM2008",
                       native_spacing_m=10)]).to_parquet(cfg["dem_metadata"])
    write_raster(cfg["dem_mask"],np.ones((1,128,128)),crs,GRID,["mask"])
    train = pd.DataFrame([dict(row=24,col=20,height=32,width=32)]*2)
    def fit(folder):
        folder.mkdir()
        write_raster(cfg["dem"],z[None],crs,GRID,["elevation_m"],source="fixture",vertical_datum="EGM2008")
        return prepare_terrain(cfg,GRID,crs,z.shape,train,folder)[2]
    first = fit(tmp_path/"a")
    z[:,90:] += 1200
    second = fit(tmp_path/"b")
    assert first["unique_pixels"]==1024
    assert first["mean"]==second["mean"] and first["scale"]==second["scale"]


def test_sar_quarter_uses_only_eligible_dates():
    class Encoder:
        def encode(self,x,mask):
            pooled = torch.nn.functional.avg_pool2d(x[:,:1],8)
            return pooled.expand(-1,128,-1,-1),None
    sar = np.broadcast_to(np.arange(4,dtype="float32")[:,None,None,None],(4,2,128,128)).copy()
    mask = np.ones((4,128,128),bool)
    mask[2,16:] = False
    norm = lambda image,valid: torch.where(valid,image,0)
    arrays,selected = quarter_sar(sar,mask,[0,1,2],Encoder(),norm,torch.device("cpu"),0.85)
    assert selected==[0,1] and (arrays["sar_count"]==2).all()
    assert np.allclose(arrays["sar_features"],0.5) and np.allclose(arrays["sar_db"],0.5)
    absent,_ = quarter_sar(sar,mask,[],Encoder(),norm,torch.device("cpu"),0.85)
    assert not absent["sar_count"].any() and not absent["sar_features"].any()


def test_split_guard_checks_footprints(tmp_path):
    for split,col in zip(("train","val","test","excluded"),(0,256,512,750)):
        x,y = GRID*(col,0)
        pd.DataFrame([dict(patch_id=split,row=0,col=col,width=128,height=128,crs="EPSG:32636",split=split,
                           xmin=x,xmax=x+1280,ymin=y-1280,ymax=y)]).to_parquet(tmp_path/f"{split}.parquet")
    cfg = dict(split_dir=tmp_path,buffer_pixels=128)
    meta = dict(crs="EPSG:32636",reference_height=128,reference_width=900)
    assert len(load_splits(cfg,meta,GRID))==4
    cfg["buffer_pixels"] = 129
    with pytest.raises(ValueError,match="buffer violation"):
        load_splits(cfg,meta,GRID)
    cfg["buffer_pixels"] = 0
    frame = pd.read_parquet(tmp_path/"val.parquet")
    frame["col"],frame["xmin"],frame["xmax"] = 10,280100,281380
    frame.to_parquet(tmp_path/"val.parquet")
    with pytest.raises(ValueError,match="overlap"):
        load_splits(cfg,meta,GRID)


def test_optical_grid_and_missing(tmp_path):
    missing = optical_arrays(None,GRID,"EPSG:32636","p","2024Q1","val","abc")
    assert missing["stage1"].shape==(192,16,16)
    assert not missing["optical_valid"].any()
    tags = dict(patch_id="p",period="2024Q1",split="val",swin_sha256="abc")
    write_raster(tmp_path/"valid_pixels_10m.tif",np.ones((1,128,128)),"EPSG:32636",
                 GRID*Affine.translation(1,0),["valid"],**tags)
    with pytest.raises(ValueError,match="grid mismatch"):
        optical_arrays((tmp_path,np.zeros(256)),GRID,"EPSG:32636","p","2024Q1","val","abc")


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    path = tmp_path_factory.mktemp("fusion")
    rng = np.random.default_rng(9)
    cfg = dict(device="cpu",epochs=1,width=8,batch_size=4,learning_rate=0.001,seed=5,buffer_pixels=128,
               min_modalities=2,reference_cells=40,calibration_cells=30,threshold_quantile=0.95)
    rows = []
    for split,locations in [("train",[0,256,512,768,1024]),("val",[1280,1346]),("test",[1800])]:
        for col in locations:
            pid = f"p{col}"
            for quarter in (1,2):
                period = f"2024Q{quarter}"
                filename = f"{split}/{pid}_{period}.npz"
                if split != "test":
                    (path/split).mkdir(exist_ok=True)
                    mask = np.ones((128,128),bool)
                    mask[:8] = False
                    np.savez(path/filename,sar_db=rng.normal(-14,2,(2,128,128)).astype("f"),
                             sar_count=mask.astype("uint8"),optical_valid=mask,dem_valid=mask,
                             dem=rng.normal(size=(9,128,128)).astype("f2"),
                             stage0=rng.normal(size=(96,32,32)).astype("f2"),
                             stage1=rng.normal(size=(192,16,16)).astype("f2"),
                             sar_features=rng.normal(size=(128,16,16)).astype("f2"),
                             optical_embedding=rng.normal(size=256).astype("f"))
                rows.append(dict(split=split,patch_id=pid,row=0,col=col,period=period,
                                 cache=filename if split!="test" else "",sar_indices=[0],
                                 status="ready" if split!="test" else "reserved"))
    pd.DataFrame(rows).to_parquet(path/"manifest.parquet")
    meta = path/"metadata.json"
    save_json(meta,dict(optical_sources=[]))
    save_json(path/"alignment_qa.json",dict(config=dict(metadata=str(meta)),transform=list(GRID)[:6],
              crs="EPSG:32636",manifest_sha256=digest(path/"manifest.parquet"),provenance={}))
    return path,cfg


def test_dataset_and_sampling(study):
    path,_ = study
    with pytest.raises(ValueError,match="train or val"):
        MultiModalArchaeologyDataset(path,"test")
    ds = MultiModalArchaeologyDataset(path,"train")
    assert torch.equal(ds[0]["stage0"],ds[0]["stage0"])
    assert ds[0]["sar_db"].shape==(2,128,128)
    sampler = QuarterPairSampler(ds.rows,4,10)
    first = list(sampler)
    assert first==list(sampler)
    assert sorted(sum(first,[]))==list(range(len(ds)))
    for batch in first:
        assert ds.rows.iloc[batch[0]].patch_id==ds.rows.iloc[batch[1]].patch_id
    sampler.epoch=1
    assert first!=list(sampler)
    ref,cal = partition(ds.rows,128)
    assert ds.rows.iloc[ref].col.max()+256<=ds.rows.iloc[cal].col.min()
    assert geo_negatives(torch.tensor([0,0,0]),torch.tensor([0,64,256]))[0].tolist()==[False,False,True]
    regions = validation_regions(pd.DataFrame({"row":range(0,1024,64)}),128)
    assert regions["north"].row.max()+128+128 <= regions["south"].row.min()


def test_heldout_cache_is_inference_only(study,tmp_path):
    path,_ = study
    source = pd.read_parquet(path/"manifest.parquet")
    test = source[source.split=="test"].copy()
    test["status"] = "ready"
    test["cache"] = [f"test/{p.patch_id}_{p.period}.npz" for p in test.itertuples()]
    both = unified_manifest(source,test)
    assert len(both)==len(source) and set(both.cache_group)=={"development","heldout"}
    with pytest.raises(ValueError,match="reserved"):
        unified_manifest(source,test.assign(col=test.col+1))
    test.to_parquet(tmp_path/"manifest.parquet")
    save_json(tmp_path/"alignment_qa.json",dict(inference_only=True,manifest_sha256=digest(tmp_path/"manifest.parquet")))
    with pytest.raises(ValueError,match="cannot be interchanged"):
        MultiModalArchaeologyDataset(tmp_path,"train")
    with pytest.raises(ValueError,match="cannot be interchanged"):
        MultiModalArchaeologyDataset(path,"test",inference_only=True)
    assert len(MultiModalArchaeologyDataset(tmp_path,"test",inference_only=True))==len(test)


@pytest.mark.parametrize("variant",VARIANTS)
def test_fusion_masks_and_gradients(study,variant):
    ds = MultiModalArchaeologyDataset(study[0],"val")
    b = next(iter(DataLoader(ds,batch_size=2)))
    torch.set_num_threads(2)
    model = LocalFusion(variant,width=8)
    b["optical_valid"].zero_()
    out = model(b)
    assert out["weights"][:,1].count_nonzero()==0
    altered = dict(b,stage0=b["stage0"]+10,optical_embedding=b["optical_embedding"]-100)
    assert torch.equal(out["fused"],model(altered)["fused"])
    assert out["fused"].shape==(2,8,32,32)
    sum(p.mean() for p in out["predictions"]).backward()
    out = model(b,torch.ones_like(out["mask"]))
    assert not out["support"].any() and not out["fused"].any()
    assert torch.isfinite(out["weights"]).all()


def test_training_scoring_export_and_labels(study):
    path,cfg = study
    runs,maps,report = [path/k for k in ("runs","maps","report")]
    train_all(path,runs,cfg,["gated"])
    export(path,runs/"gated",maps,cfg)
    verify(path,maps)
    with rasterio.open(maps/"2024Q1_anomalies.tif") as src:
        from rasterio.warp import transform_geom
        from shapely.geometry import box,mapping
        geom = transform_geom(src.crs,"EPSG:4326",mapping(box(*src.bounds)))
        assert src.read(1,masked=True).mask[:8].all()
        assert src.transform==GRID*Affine.translation(1280,0)
    labels = path/"labels.geojson"
    save_json(labels,dict(type="FeatureCollection",features=[
        dict(type="Feature",geometry=geom,properties=dict(id=i,split="val",label=i,source="fixture",
             period=f"2024Q{i+1}")) for i in (0,1)]))
    evaluate(runs,maps,report,path,labels)
    result = read_json(report/"evaluation.json")
    assert result["labels"]["scored"]==2 and "roc_auc" in result["labels"]
    assert (report/"validation_curves.png").is_file()
    invalid = read_json(labels)
    invalid["features"][0]["properties"]["split"] = "test"
    save_json(labels,invalid)
    with pytest.raises(ValueError,match="validation polygons"):
        label_metrics(labels,maps,0.9)
