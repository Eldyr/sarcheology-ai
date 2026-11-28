"""Metric candidate footprints and shape diagnostics on the observed anomaly support."""
from dataclasses import dataclass
import hashlib
import math

import numpy as np
from rasterio import features
from scipy import ndimage as ndi
from shapely.geometry import shape
from shapely.ops import unary_union
from skimage.feature import canny
from skimage.measure import regionprops
from skimage.transform import probabilistic_hough_line


@dataclass
class Candidate:
    id: str
    geometry: object
    window: tuple
    pixels: np.ndarray
    descriptors: dict


def disk(radius):
    yy,xx = np.ogrid[-radius:radius+1,-radius:radius+1]
    return xx*xx + yy*yy <= radius*radius


def interior(valid,radius):
    return ndi.binary_erosion(valid,structure=disk(radius),border_value=0) if radius else valid.copy()


def morphology(mask,geometry,cell_m):
    area,perimeter = geometry.area,geometry.length
    rect = geometry.minimum_rotated_rectangle
    corners = np.array(rect.exterior.coords)[:4]
    edges = np.roll(corners,-1,axis=0)-corners
    lengths = np.linalg.norm(edges,axis=1)
    major = edges[lengths.argmax()]
    angle = math.degrees(math.atan2(major[0],major[1])) % 180
    elongation = lengths.max()/max(lengths.min(),1e-9)
    compact = min(1.,4*math.pi*area/(perimeter**2))
    rectangular = area/rect.area
    prop = regionprops(mask.astype("uint8"))[0]
    yy,xx = np.indices(mask.shape)
    y,x = yy-prop.centroid[0],xx-prop.centroid[1]
    c,s = math.cos(prop.orientation),math.sin(prop.orientation)
    ellipse = ((y*c+x*s)/max(prop.axis_major_length/2,0.5))**2 + ((x*c-y*s)/max(prop.axis_minor_length/2,0.5))**2 <= 1
    # Pad the fitted ellipse canvas to avoid artificially high agreement for tight rectangles.
    ey,ex = np.where(mask)
    extent = int(math.ceil(prop.axis_major_length/2))+2
    fy,fx = np.mgrid[-extent:extent+1,-extent:extent+1]
    dy,dx = fy-(prop.centroid[0]%1),fx-(prop.centroid[1]%1)
    full_ellipse = ((dy*c+dx*s)/max(prop.axis_major_length/2,0.5))**2 + ((dx*c-dy*s)/max(prop.axis_minor_length/2,0.5))**2<=1
    intersection = int((ellipse & mask).sum())
    ellipse_iou = intersection/max(1,int(mask.sum()+full_ellipse.sum()-intersection))
    ry,rx = np.rint(2*prop.centroid[0]-ey).astype(int),np.rint(2*prop.centroid[1]-ex).astype(int)
    inside = (ry>=0)&(ry<mask.shape[0])&(rx>=0)&(rx<mask.shape[1])
    symmetry = float(mask[ry[inside],rx[inside]].sum()/len(ey))
    boundary = canny(np.pad(mask.astype(float),2),sigma=1)
    lines = probabilistic_hough_line(boundary,threshold=5,line_length=6,line_gap=1,rng=42)
    # One native score cell can itself look like a perfect rectangle.
    resolved = area >= 4*cell_m**2
    linear = min(1.,max(0.,(elongation-1)/4)) if lines else 0.
    geometric = max(linear,rectangular*symmetry,ellipse_iou*symmetry) if resolved else 0.
    return dict(area_m2=area,perimeter_m=perimeter,compactness=compact,rectangularity=rectangular,
                ellipse_iou=min(1.,ellipse_iou),circularity=compact,elongation=float(elongation),
                orientation_deg_from_north=float(angle),rotational_symmetry=symmetry,hough_lines=len(lines),
                linear_score=linear,morphology_resolved=resolved,morphology_score=float(geometric))


def extract(scores,threshold,transform,crs,cfg,cell_m):
    if not crs.is_projected or crs.linear_units!="metre" or transform.a<=0 or transform.e!=-transform.a or transform.b or transform.d:
        raise ValueError("Candidate measurements require a north-up square metre grid")
    resolution = transform.a
    edge = int(math.ceil(cfg["edge_guard_m"]/resolution))
    valid = np.isfinite(scores)
    safe = np.stack([interior(m,edge) for m in valid])
    raw = valid & (scores>=threshold)
    hits = raw & safe
    union = hits.any(0)
    # Geodesic dilation joins nearby regions without stepping across unobserved pixels.
    grouped = union.copy()
    for _ in range(int(cfg["group_gap_m"]//(2*resolution))):
        grouped = ndi.binary_dilation(grouped) & safe.any(0)
    labels,_ = ndi.label(grouped)
    kept = np.zeros_like(labels)
    candidates = []
    removed = 0
    for idx,w in enumerate(ndi.find_objects(labels),1):
        if w is None:
            continue
        pixels = (labels[w]==idx) & union[w]
        if pixels.sum()*resolution**2 < cfg["minimum_area_m2"]:
            removed += 1
            continue
        t = transform*type(transform).translation(w[1].start,w[0].start)
        geom = unary_union([shape(g) for g,_ in features.shapes(pixels.astype("uint8"),mask=pixels,transform=t)])
        identity = str((str(crs),tuple(t)[:6],pixels.shape)).encode()+np.packbits(pixels).tobytes()
        cid = "C-"+hashlib.sha256(identity).hexdigest()[:12]
        candidates.append(Candidate(cid,geom,w,pixels,morphology(pixels,geom,cell_m)))
        kept[w][pixels] = len(candidates)
    audit = dict(raw_pixel_quarters=int(raw.sum()),edge_pixel_quarters_removed=int((raw & ~safe).sum()),
                 below_minimum_groups=removed,groups_before_area_filter=int(labels.max()),retained=len(candidates),
                 threshold=float(threshold),cell_m=float(cell_m))
    return candidates,kept,hits,safe,audit
