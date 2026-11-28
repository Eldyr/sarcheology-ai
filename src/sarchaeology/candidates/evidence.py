"""Read native measurements only inside the development mosaic and retain their dates."""
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from scipy import ndimage as ndi
from skimage.feature import canny
from skimage.transform import probabilistic_hough_line

from sarchaeology.fusion.io import digest, read_json, write_raster
from .extraction import disk


def median(x):
    finite = x[np.isfinite(x)]
    return float(np.median(finite)) if len(finite) else None


def avg(values):
    known = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(known)) if known else None


def stack_median(values):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",message="All-NaN slice encountered")
        return np.nanmedian(values,axis=0)


class EvidenceGrid:
    def __init__(self,cfg):
        self.cfg = cfg
        self.maps = Path(cfg["maps"])
        self.sources = {}
        self.prov = read_json(self.maps/"provenance.json")
        self.qa = read_json(Path(cfg["prepared"])/"alignment_qa.json")
        if self.prov["split"]!="val" or self.prov["alignment_sha256"] != digest(Path(cfg["prepared"])/"alignment_qa.json"):
            raise ValueError("Expected a verified Phase 4 validation mosaic")
        self.periods = sorted(self.prov["periods"])
        with rasterio.open(self.maps/f"{self.periods[0]}_anomalies.tif") as src:
            self.shape,self.transform,self.crs = src.shape,src.transform,src.crs
        self.phase4 = {}
        for p in self.periods:
            path = self.maps/f"{p}_anomalies.tif"
            with rasterio.open(path) as src:
                if (src.tags().get("split")!="val" or src.tags().get("period")!=p or
                        src.tags().get("checkpoint_sha256")!=self.prov["model"]["checkpoint_sha256"]):
                    raise ValueError("Phase 4 map identity mismatch")
                names = src.descriptions
            self.phase4[p] = dict(zip(names,self.read(path)))
        self.scores = np.stack([self.phase4[p]["ensemble_percentile"] for p in self.periods])
        self.domain = np.isfinite(self.scores).any(0)
        self.terrain = self.read(Path(cfg["prepared"])/"terrain.tif",["local_relief_11x11_m","slope_degrees","ruggedness_rms_m"])
        info = read_json(self.qa["config"]["metadata"])
        self.sar = {src["date"]:self.read(Path(src["path"]),["VV","VH"]) for src in info["sar_sources"]
                    if str(pd.Period(src["date"],freq="Q")) in self.periods}
        self.optical,self.observations = {},{}
        for path in sorted(Path(cfg["composites"]).glob("*Q?_median.tif")):
            period = path.name.split("_")[0]
            if not re.fullmatch(r"\d{4}Q[1-4]",period):
                continue
            self.optical[period] = self.read(path,["B02","B03","B04","B08","B11","B12"])
            count = path.with_name(f"{period}_valid_count.tif")
            if count.is_file():
                self.observations[period] = self.read(count)[0]
        self.residuals = self.load_residuals(cfg.get("spectral"))
        self.rain = {}
        if cfg.get("rainfall"):
            path = Path(cfg["rainfall"])
            rainfall = pd.read_csv(path)
            if (rainfall.period.duplicated().any() or rainfall.source.isna().any()
                    or not np.isfinite(rainfall.precipitation_mm).all() or (rainfall.precipitation_mm<0).any()):
                raise ValueError("Rainfall needs unique periods, source and nonnegative measured totals")
            self.rain = {r.period:dict(mm=float(r.precipitation_mm),source=str(r.source)) for r in rainfall.itertuples()}
            self.sources[str(path)] = digest(path)
        self.indices = {}
        for p,a in self.optical.items():
            def ratio(x,y):
                return (x-y)/np.where(np.abs(x+y)>1e-6,x+y,np.nan)
            self.indices[p] = dict(ndvi=ratio(a[3],a[2]),ndmi=ratio(a[3],a[4]),ndwi=ratio(a[1],a[3]))

    def read(self,path,bands=None):
        self.sources[str(path)] = digest(path)
        with rasterio.open(path) as src:
            x,y = (~src.transform)*(self.transform.c,self.transform.f)
            col,row = round(x),round(y)
            w = Window(col,row,self.shape[1],self.shape[0])
            if (src.crs!=self.crs or not src.window_transform(w).almost_equals(self.transform)
                    or min(col,row)<0 or col+w.width>src.width or row+w.height>src.height):
                raise ValueError(f"Evidence grid mismatch: {path}")
            idx = [src.descriptions.index(b)+1 for b in bands] if bands else None
            a = src.read(idx,window=w,masked=True).astype("float32").filled(np.nan)
            if hasattr(self,"domain"):
                a[:,~self.domain] = np.nan
            return a

    def load_residuals(self,path):
        if not path:
            return {}
        root = Path(path)
        manifest = read_json(root/"export_manifest.json")
        if (manifest["split"]!="val" or manifest["subset_export"] or not manifest["spectral_residuals_enabled"]
                or any(manifest["provenance_sha256"].get(k)!=v for k,v in self.prov["upstream"]["optical"].items())):
            raise ValueError("Spectral residual export lineage differs from validation features")
        self.spectral_hash = manifest["provenance_sha256"]["spectral"]
        df = pd.read_csv(root/"patch_embeddings.csv")
        expected = pd.read_parquet(Path(self.cfg["prepared"])/"manifest.parquet")
        expected = expected[(expected.split=="val") & expected.optical_available.eq(True)]
        keys = ["patch_id","period","row","col"]
        if df.duplicated(keys).any() or set(df[keys].itertuples(index=False,name=None))!=set(expected[keys].itertuples(index=False,name=None)):
            raise ValueError("Spectral export window membership mismatch")
        result = {}
        for p,rows in df.groupby("period"):
            sums = np.zeros((4,*self.shape),np.float32)
            counts = np.zeros_like(sums)
            for item in rows.itertuples():
                folder = root/p/f"patch_{item.patch_id}"
                for i,pair in enumerate(("B04_B08","B11_B12")):
                    path = folder/f"spectral_residual_{pair}_signed_reflectance.tif"
                    self.sources[str(path)] = digest(path)
                    with rasterio.open(path) as src:
                        x,y = (~self.transform)*(src.transform.c,src.transform.f)
                        x,y = round(x),round(y)
                        if (min(x,y)<0 or y+128>self.shape[0] or x+128>self.shape[1] or src.shape!=(128,128)
                                or src.crs!=self.crs or src.transform!=self.transform*type(self.transform).translation(x,y)
                                or src.tags().get("period")!=p or src.tags().get("patch_id")!=item.patch_id
                                or src.tags().get("split")!="val" or src.tags().get("bands")!=pair.replace("_",",")):
                            raise ValueError(f"Spectral residual grid or identity mismatch: {path}")
                        a = src.read(masked=True).filled(np.nan)
                        ix = np.s_[2*i:2*i+2,y:y+128,x:x+128]
                        sums[ix] += np.nan_to_num(a)
                        counts[ix] += np.isfinite(a)
            result[p] = np.where(counts>0,sums/counts.clip(1),np.nan)
            result[p][:,~np.isfinite(self.phase4[p]["ensemble_percentile"])] = np.nan
        return result

    def write_layers(self,out,safe):
        """Fixed, sensor-specific local-contrast baselines on the entire eligible domain."""
        out.mkdir()
        for p,inside in zip(self.periods,safe):
            dates = [a for date,a in self.sar.items() if str(pd.Period(date,freq="Q"))==p]
            empty = np.full(self.shape,np.nan,np.float32)
            vv = stack_median(np.stack(dates)[:,0]) if dates else empty
            opt = self.indices.get(p,{})
            ndvi,ndmi = opt.get("ndvi",empty),opt.get("ndmi",empty)
            res = self.residuals.get(p,np.full((4,*self.shape),np.nan,np.float32))
            sar = local_contrast(vv,2.)
            optical = local_contrast(ndvi,0.12)
            dem = local_contrast(self.terrain[0],0.5)
            with np.errstate(invalid="ignore"):
                classical = (sar+optical+dem)/3
            bands = ("vv_db","ndvi","ndmi","relief_m","B04_residual","B08_residual","B11_residual","B12_residual",
                     "sar_only","optical_only","dem_only","classical_mean")
            a = np.stack([vv,ndvi,ndmi,self.terrain[0],*res,sar,optical,dem,classical])
            a[:,~inside] = np.nan
            write_raster(out/f"{p}_evidence.tif",a,self.crs,self.transform,bands,period=p,split="val")


def local_contrast(a,scale):
    valid = np.isfinite(a)
    count = ndi.uniform_filter(valid.astype("float32"),size=25,mode="constant")
    mean = ndi.uniform_filter(np.where(valid,a,0),size=25,mode="constant")/np.maximum(count,1e-6)
    return np.where(valid & (count>=0.3),np.abs(a-mean)/scale,np.nan)


def neighbourhood(candidate,labels,domain,radius):
    y,x = candidate.window
    w = (slice(max(0,y.start-radius),min(domain.shape[0],y.stop+radius)),
         slice(max(0,x.start-radius),min(domain.shape[1],x.stop+radius)))
    local = np.zeros(domain[w].shape,bool)
    local[y.start-w[0].start:y.stop-w[0].start,x.start-w[1].start:x.stop-w[1].start] = candidate.pixels
    distance = ndi.distance_transform_edt(~local)
    ring = (distance>1)&(distance<=radius)&(labels[w]==0)&domain[w]
    return w,local,ring


def contrast(a,support,ring):
    v,b = a[support],a[ring]
    v,b = v[np.isfinite(v)],b[np.isfinite(b)]
    if len(v)<max(1,support.sum()*0.5) or len(b)<16:
        return None
    return float(np.median(v)-np.median(b))


def edge_alignment(values,azimuth,scale):
    """Compare the footprint axis to independently observed local image line segments."""
    valid = np.isfinite(values)
    center = median(values)
    if center is None or valid.sum()<36:
        return dict(segments=0,aligned_length_fraction=None)
    normalized = np.clip(np.nan_to_num((values-center)/scale,nan=0.),-1,1)
    edges = canny(normalized,sigma=1.3,low_threshold=.10,high_threshold=.25,mask=valid)
    lines = probabilistic_hough_line(edges,threshold=6,line_length=6,line_gap=1,rng=42)
    total,aligned = 0.,0.
    for (x0,y0),(x1,y1) in lines:
        angle = np.degrees(np.arctan2(x1-x0,y0-y1))%180
        difference = abs((angle-azimuth+90)%180-90)
        length = np.hypot(x1-x0,y1-y0)
        total += length
        if difference<=15:
            aligned += length
    return dict(segments=len(lines),aligned_length_fraction=float(aligned/total) if total else None)


def profile(c,grid,labels,hits,safe,cfg):
    radius = max(2,round(cfg["background_m"]/grid.transform.a))
    w,m,ring = neighbourhood(c,labels,grid.domain,radius)
    periods,dates,history = {},{},{}
    for date,a in grid.sar.items():
        vv,vh = [contrast(b[w],m,ring) for b in a]
        dates[date] = dict(vv_delta_db=vv,vh_delta_db=vh)
    for p,indices in grid.indices.items():
        history[p] = {name:contrast(band[w],m,ring) for name,band in indices.items()}
        history[p]["ndmi_level"] = median(indices["ndmi"][w][m])
    for i,p in enumerate(grid.periods):
        available = m & safe[i][w]
        denom = int(available.sum())
        supported = denom>=0.5*m.sum()
        rec = dict(eligible_fraction=denom/int(m.sum()),hit_fraction=float((hits[i][w]&m).sum()/denom) if supported else None)
        for name,a in grid.phase4[p].items():
            rec[name] = median(a[w][available])
        optical = grid.residuals.get(p)
        for k,b in enumerate(("B04","B08","B11","B12")):
            rec[b+"_signed_residual"] = median(optical[k][w][available]) if optical is not None else None
            rec[b+"_error_contrast"] = contrast(np.abs(optical[k][w]),m,ring) if optical is not None else None
        rec["clear_observations"] = median(grid.observations[p][w][available]) if p in grid.observations else None
        periods[p] = rec
    terrain = {k:contrast(a[w],m,ring) for k,a in zip(("relief_delta_m","slope_delta_deg","ruggedness_delta_m"),grid.terrain)}
    terrain["valid_fraction"] = float(np.isfinite(grid.terrain[0][w][m]).mean())
    parallels = {}
    if c.descriptors["morphology_resolved"] and c.descriptors["elongation"]>=1.5:
        axis = c.descriptors["orientation_deg_from_north"]
        for p in grid.periods:
            if p in grid.indices:
                parallels[p+"_vegetation_edges"] = edge_alignment(grid.indices[p]["ndvi"][w],axis,.25)
        parallels["surface_relief_edges"] = edge_alignment(grid.terrain[0][w],axis,.5)
    return dict(acquisitions=dates,quarterly=periods,optical_history=history,terrain=terrain,
                landscape_edge_comparison=parallels,
                precipitation=grid.rain,
                background_pixels=int(ring.sum()),footprint_pixels=int(m.sum()))
