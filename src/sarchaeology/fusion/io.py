"""Shared immutable-run metadata and raster conventions."""
import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio
import torch

NODATA = -9999.0


def digest(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def device_for(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def write_raster(path, data, crs, transform, bands, **tags):
    with rasterio.open(path, "w", driver="GTiff", width=data.shape[2], height=data.shape[1],
                       count=len(bands), dtype="float32", transform=transform, crs=crs,
                       nodata=NODATA, compress="deflate", tiled=True) as dst:
        dst.write(np.where(np.isfinite(data), data, NODATA).astype("float32"))
        dst.descriptions = tuple(bands)
        dst.update_tags(**tags)
