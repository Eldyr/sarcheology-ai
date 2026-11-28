"""Offline candidate imagery, local GIS layers and coordinate-free public release."""
from html import escape
import json
from pathlib import Path
import shutil
from zipfile import ZipFile,ZIP_DEFLATED

import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import binary_erosion

from sarchaeology.fusion.io import read_json,save_json


def scale(a,lo,hi):
    return np.nan_to_num(np.clip((a-lo)/max(hi-lo,1e-6),0,1),nan=0.0)


def panel_image(grid,c,period):
    y,x = c.window
    pad = 12
    w = (slice(max(0,y.start-pad),min(grid.shape[0],y.stop+pad)),
         slice(max(0,x.start-pad),min(grid.shape[1],x.stop+pad)))
    footprint = np.zeros(grid.domain[w].shape,bool)
    footprint[y.start-w[0].start:y.stop-w[0].start,x.start-w[1].start:x.stop-w[1].start] = c.pixels
    boundary = footprint & ~binary_erosion(footprint)
    a = grid.optical.get(period)
    blank = np.full(grid.shape,np.nan,np.float32)
    rgb = (np.stack([scale(a[k][w],0,0.3) for k in [2,1,0]],-1) if a is not None else
           np.zeros((*boundary.shape,3)))
    optical_ok = np.isfinite(a[2][w]) if a is not None else np.zeros_like(boundary)
    swir = grid.residuals.get(period)
    swir = swir[2] if swir is not None else blank
    score = grid.phase4[period]["ensemble_percentile"]
    quarters = [v[0][w] for date,v in grid.sar.items() if (int(date[5:7])-1)//3+1==int(period[-1]) and date[:4]==period[:4]]
    from .evidence import stack_median
    vv = stack_median(np.stack(quarters)) if quarters else blank[w]
    rasters = [("Optical RGB",rgb,optical_ok),("Anomaly percentile",scale(score[w],0.5,1),np.isfinite(score[w])),
               ("VV ( -20 to -3 dB )",scale(vv,-20,-3),np.isfinite(vv)),
               ("B11 residual ( +/-0.10 )",scale(swir[w],-.10,.10),np.isfinite(swir[w])),
               ("Local relief ( +/-1.5 m )",scale(grid.terrain[0][w],-1.5,1.5),np.isfinite(grid.terrain[0][w]))]
    canvas = Image.new("RGB",(5*210,240),"white")
    draw = ImageDraw.Draw(canvas)
    for i,(caption,vals,valid) in enumerate(rasters):
        image = np.uint8(vals*255)
        if image.ndim==2:
            image = np.repeat(image[...,None],3,axis=2)
        image[~valid] = [64,61,78]
        image[boundary] = [0,235,255]
        tile = Image.fromarray(image)
        factor = min(200/tile.width,200/tile.height)
        tile = tile.resize((max(1,round(tile.width*factor)),max(1,round(tile.height*factor))),Image.Resampling.NEAREST)
        canvas.paste(tile,(210*i+(210-tile.width)//2,24+(200-tile.height)//2))
        draw.text((210*i+6,5),caption,fill="black")
    draw.text((6,222),f"{period}    Cyan: candidate footprint    Grey-purple: missing    Source pixel {grid.transform.a:g} m",fill="black")
    return canvas


def candidate_report(out,grid,c,record):
    folder = out/c.id
    folder.mkdir()
    periods = record["profile"]["quarterly"]
    strongest = max(periods,key=lambda p:periods[p]["hit_fraction"] or 0)
    for p in periods:
        panel_image(grid,c,p).save(folder/f"{p}.png")
    save_json(folder/"evidence.json",record)
    def table(rows):
        return "<table>"+"".join("<tr><th>"+escape(str(k))+"</th><td>"+escape(str(v))+"</td></tr>" for k,v in rows)+"</table>"
    def num(value):
        return "unavailable" if value is None else f"{value:.3f}"
    ranking = record["ranking"]
    modality_table = table([(k,num(v)) for k,v in ranking["evidence"].items()])
    annual_table = table([(y,f"{v['available_quarters']} quarters; mean NDVI contrast {num(v['mean_ndvi_contrast'])}")
                          for y,v in ranking["annual_optical"].items()])
    context_table = table([(c["kind"],f"{c['name']}, {c['distance_m']:.0f} m to mapped geometry. "
                            f"Recorded positional uncertainty (m): {num(c['uncertainty_m'])}. Source: {c['source']}")
                           if "distance_m" in c else (c["kind"],f"Mapped fraction {num(c['fraction'])}")
                           for c in record["context"]])
    quarter_table = table([(p,f"eligible {r['eligible_fraction']:.0%}, exceedance fraction {num(r['hit_fraction'])}, "
                              f"B11 signed residual {num(r['B11_signed_residual'])}") for p,r in periods.items()])
    dates = record["profile"]["acquisitions"]
    content = (f"<h1>{c.id}</h1><p>Restricted research copy · rank {record['rank']} · investigation priority "
               f"{ranking['priority']:.1f}/100</p><p>Feature evidence is conditional on spatial support. "
               "A candidate can have a modern, natural or archaeological explanation.</p>"
               +table([("Area (m²)",round(c.geometry.area)),("Perimeter (m)",round(c.geometry.length)),
                       ("Compactness",num(c.descriptors["compactness"])),("Orientation (degrees from north)",num(c.descriptors["orientation_deg_from_north"])),
                       ("Effective score cell (m)",grid.prov["score_cell_m"]),("Quality",num(ranking["quality"]))])
               +"<h2>Satellite observations</h2>"+quarter_table+
               "".join(f'<figure><img width="100%" src="{p}.png"><figcaption>{p}</figcaption></figure>' for p in [strongest,*[q for q in periods if q!=strongest]])+
               "<h2>Sensor evidence</h2><p>Strength from local contrast, on a 0–1 scale. Missing values indicate insufficient evidence.</p>"
               +modality_table+"<h2>Optical history</h2>"+annual_table+"<h2>Historical and landscape context</h2>"+context_table+
               "<h2>Ranking rationale</h2>"+table([(k,num(v)) for k,v in ranking["contributions"].items()]+[("Penalty points",num(ranking["penalty_points"]))])+
               "<p>"+escape(", ".join(ranking["flags"]) or "No configured flags triggered.")+"</p>"+
               "<h2>Data and uncertainty</h2><p>SAR acquisition dates: "+escape(", ".join(dates))+
               ".</p><p>Frozen fusion checkpoint SHA-256: "+escape(grid.prov["model"]["checkpoint_sha256"])+
               '.</p><p><a href="evidence.json">Full measurements, context, missing evidence and provenance</a></p>')
    (folder/"report.html").write_text('<!doctype html><meta charset="utf-8"><title>'+c.id+
        '</title><style>body{font:16px system-ui;max-width:1100px;margin:2em auto}th,td{padding:.35em;text-align:left;border-bottom:1px solid #ddd}figure{margin:1em 0}</style>'+content,encoding="utf-8")


def package(run,path,restricted=False):
    """Public release is constructed from an allowlist. It contains no row-level records."""
    summary = read_json(run/"summary.json")
    with ZipFile(path,"x",ZIP_DEFLATED) as z:
        if restricted:
            for folder in ("gis","reports","context"):
                for f in sorted((run/folder).rglob("*")):
                    if f.is_file():
                        z.write(f,f.relative_to(run).as_posix())
            for name in ("index.html","summary.json","provenance.json","policy.json","ranked.csv","screening.json"):
                z.write(run/name,name)
            for f in sorted((run/"validation").glob("*")):
                if f.is_file():
                    z.write(f,"validation/"+f.name)
        else:
            report = dict(candidate_count=summary["candidate_count"],quarters=summary["quarters"],
                          status="Unvalidated research candidates",validation=summary["validation_status"],
                          sharing="Aggregate release; exact locations available through the research custodian.")
            z.writestr("summary.json",json.dumps(report,indent=2))


def transform_private_package(source,key_file,destination,*,decrypt=False):
    """Protect a restricted archive with a custodian-supplied Fernet key."""
    from cryptography.fernet import Fernet
    cipher = Fernet(key_file.read_bytes().strip())
    data = source.read_bytes()
    result = cipher.decrypt(data) if decrypt else cipher.encrypt(data)
    with destination.open("xb") as target:
        target.write(result)
