"""Context joins, nuisance flags and a decomposable investigation-priority score."""
from pathlib import Path
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
import yaml

from sarchaeology.fusion.io import digest,save_json,write_raster
from .evidence import avg

MODERN = {"road","field_boundary","irrigation","building","infrastructure","excavation"}
NATURAL = {"drainage","sediment","erosion"}


class Context:
    def __init__(self,registry,grid,out):
        self.vectors,self.rasters = [],[]
        config = yaml.safe_load(Path(registry).read_text(encoding="utf-8"))
        out.mkdir()
        for i,src in enumerate(config.get("vectors",[])+config.get("rasters",[])):
            for key in ("kind","attribution","source","permission","epoch","uncertainty_m"):
                if key not in src or (key!="uncertainty_m" and not src[key]):
                    raise ValueError(f"Missing context metadata: {key}")
            if src["uncertainty_m"] is not None and src["uncertainty_m"]<0:
                raise ValueError("Invalid source uncertainty")
            src["sha256"] = digest(src["path"])
            if src in config.get("vectors",[]):
                frame = gpd.read_file(src["path"])
                if frame.crs is None or frame.empty or not frame.geometry.is_valid.all():
                    raise ValueError("Context needs valid georeferenced features")
                frame = frame.to_crs(grid.crs)
                frame.to_crs(4326).to_file(out/f"vector_{i}.geojson",driver="GeoJSON")
                self.vectors.append((src,frame))
            else:
                with rasterio.open(src["path"]) as dataset, WarpedVRT(dataset,crs=grid.crs,
                        transform=grid.transform,width=grid.shape[1],height=grid.shape[0],
                        dtype="float32",nodata=-9999,resampling=Resampling.nearest) as vrt:
                    a = vrt.read(masked=True).filled(np.nan)
                a[:,~grid.domain] = np.nan
                write_raster(out/f"raster_{i}.tif",a,grid.crs,grid.transform,
                             [f"band_{k+1}" for k in range(len(a))],source=src["source"])
                self.rasters.append((src,a[0]))
        save_json(out/"sources.json",config)

    def describe(self,c):
        contacts = []
        for src,frame in self.vectors:
            distances = frame.geometry.distance(c.geometry)
            feature = frame.loc[distances.idxmin()]
            contacts.append(dict(kind=src["kind"],name=str(feature.get("name",src["kind"])),
                                 distance_m=float(distances.min()),uncertainty_m=src["uncertainty_m"],
                                 intersects=bool(feature.geometry.intersects(c.geometry)),source=src["source"]))
        for src,a in self.rasters:
            if src["kind"]=="land_use":
                values = a[c.window][c.pixels]
                valid = values[np.isfinite(values)]
                for code,kind in src.get("classes",{}).items():
                    contacts.append(dict(kind=kind,coverage_fraction=len(valid)/len(values),
                                         fraction=float(np.mean(valid==int(code))) if len(valid) else None,
                                         uncertainty_m=src["uncertainty_m"],source=src["source"]))
        return contacts


def capped(value,scale):
    return None if value is None else min(1.,abs(value)/scale)


def interpret(profile,desc,context,weights):
    quarters = profile["quarterly"]
    hits = [r["hit_fraction"] for r in quarters.values() if r["hit_fraction"] is not None]
    persistence = sum(h>=0.25 for h in hits)/len(hits) if hits else None
    strength = [r["ensemble_percentile"] for r in quarters.values() if r["hit_fraction"] is not None and r["hit_fraction"]>0]
    sar = [a["vv_delta_db"] for a in profile["acquisitions"].values() if a["vv_delta_db"] is not None]
    vh = [a["vh_delta_db"] for a in profile["acquisitions"].values() if a["vh_delta_db"] is not None]
    sar_strength = avg([capped(avg([abs(x) for x in vals]),2) for vals in (sar,vh)])
    # Fraction of all available acquisitions showing a consistent signed VV contrast >= 1 dB.
    stability = max(sum(v>=1 for v in sar),sum(v<=-1 for v in sar))/len(sar) if len(sar)>=3 else None
    err = [max(0,r[b+"_error_contrast"]) for r in quarters.values() for b in ("B04","B08","B11","B12")
           if r[b+"_error_contrast"] is not None]
    optical_strength = capped(avg(err),0.035)
    relief = profile["terrain"]["relief_delta_m"]
    dem_strength = capped(relief,0.5)
    evidence = dict(sar=sar_strength,optical=optical_strength,dem=dem_strength)
    known = [x for x in evidence.values() if x is not None]
    diversity = sum(x>=0.5 for x in known)/3
    qa = [avg([r["quality_"+k] for k in evidence]) for r in quarters.values()]
    quality = (avg(qa) or 0)*len(known)/3*(len(hits)/max(1,len(quarters)))
    seasonal = [r["ndvi"] for r in profile["optical_history"].values() if r["ndvi"] is not None]
    seasonal_range = max(seasonal)-min(seasonal) if len(seasonal)>=2 else None
    annual = {}
    for year in sorted({p[:4] for p in profile["optical_history"]}):
        vals = [r["ndvi"] for p,r in profile["optical_history"].items() if p.startswith(year) and r["ndvi"] is not None]
        annual[year] = dict(available_quarters=len(vals),mean_ndvi_contrast=avg(vals))
    same_season = {}
    for q in range(1,5):
        vals = [r["ndvi"] for p,r in profile["optical_history"].items() if p.endswith(f"Q{q}") and r["ndvi"] is not None]
        same_season[f"Q{q}"] = dict(years=len(vals),signed_recurrence=max(sum(v>=0.1 for v in vals),
                                         sum(v<=-0.1 for v in vals))/len(vals) if len(vals)>=2 else None)
    # Compare relative canopy-moisture conditions using only observed quarters.
    moist = sorted((profile["optical_history"].get(p,{}).get("ndmi_level"),p) for p in quarters
                   if profile["optical_history"].get(p,{}).get("ndmi_level") is not None and quarters[p]["hit_fraction"] is not None)
    wet_dry = None
    if len(moist)>=2 and moist[-1][0]-moist[0][0]>=0.05:
        dry,wet = moist[0][1],moist[-1][1]
        wet_dry = dict(lower_ndmi_period=dry,higher_ndmi_period=wet,
                       hit_fraction_change=quarters[wet]["hit_fraction"]-quarters[dry]["hit_fraction"])
    reasons = {}
    def flag(key,penalty):
        reasons[key] = penalty
    for item in context:
        if item.get("intersects") or (item.get("fraction") or 0)>=0.5:
            if item["kind"] in MODERN:
                flag("mapped_"+item["kind"],0.20)
            if item["kind"] in NATURAL:
                flag("mapped_"+item["kind"],0.12)
    if seasonal_range is not None and seasonal_range>=0.25:
        flag("strong_vegetation_contrast_variation",0.08)
    if stability is not None and stability<0.20 and (sar_strength or 0)>0.5:
        flag("episodic_backscatter_or_artifact",0.06)
    if profile["terrain"].get("valid_fraction",1)<0.8:
        flag("incomplete_dem_support",0.04)
    moisture_contrasts = [r["ndmi"] for r in profile["optical_history"].values() if r["ndmi"] is not None]
    if moisture_contrasts and max(moisture_contrasts)-min(moisture_contrasts)>0.35:
        flag("variable_canopy_moisture",0.05)
    if not sar:
        flag("missing_backscatter_contrast",0)
    if not err:
        flag("missing_optical_residual_contrast",0)
    if desc["elongation"]>=4 and desc["hough_lines"] and seasonal_range is not None and seasonal_range>=0.15:
        flag("possible_field_or_irrigation_boundary",0.08)
    for kind,result in profile.get("landscape_edge_comparison",{}).items():
        if result["segments"]>=3 and (result["aligned_length_fraction"] or 0)>=.75:
            flag("parallel_vegetation_edges" if kind.endswith("vegetation_edges") else "parallel_relief_edges",0)
    clear = avg([r["clear_observations"] for r in quarters.values()])
    if clear is not None and clear<3:
        flag("few_clear_optical_observations",0.08)
    if quality<0.75:
        flag("limited_sensor_support",0)
    if not desc["morphology_resolved"]:
        flag("shape_below_four_score_cells",0)
    if len(hits)>=2 and persistence is not None and sum(h>=0.25 for h in hits)==1:
        flag("single_quarter_signal",0)
    terms = dict(magnitude=avg(strength) or 0,persistence=persistence or 0,diversity=diversity,
                 morphology=desc["morphology_score"],acquisition_stability=stability or 0)
    if set(weights)!=set(terms) or min(weights.values())<0 or not np.isclose(sum(weights.values()),1):
        raise ValueError("Priority weights must be nonnegative and sum to one")
    contributions = {k:100*weights[k]*v*quality for k,v in terms.items()}
    penalty = 100*min(0.45,sum(reasons.values()))
    score = float(np.clip(sum(contributions.values())-penalty,0,100))
    rainfall = profile.get("precipitation",{})
    ordered = sorted((r["mm"],p) for p,r in rainfall.items() if p in quarters and quarters[p]["hit_fraction"] is not None)
    precipitation_comparison = None
    if len(ordered)>=2 and ordered[-1][0]>ordered[0][0]:
        dry,wet = ordered[0][1],ordered[-1][1]
        precipitation_comparison = dict(driest=dry,wettest=wet,
                          hit_fraction_change=quarters[wet]["hit_fraction"]-quarters[dry]["hit_fraction"])
    return dict(priority=score,contributions=contributions,penalty_points=penalty,flags=reasons,
                evidence=evidence,quality=quality,available_quarters=len(hits),persistent_fraction=persistence,
                acquisition_stability=stability,sar_dates=len(sar),annual_optical=annual,
                same_quarter_optical=same_season,measured_rainfall_comparison=precipitation_comparison,
                relative_moisture=wet_dry,ndvi_contrast_range=seasonal_range,
                review_status="flagged" if penalty else "retain")
