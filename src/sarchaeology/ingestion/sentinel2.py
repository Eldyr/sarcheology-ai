"""Placeholder module for SARchaeology AI v1.0."""

import xml.etree.ElementTree as ET

import numpy as np
import planetary_computer as pc
import rasterio
import requests

from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT


BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")

# Sentinel-2 band IDs in product metadata XML.
BAND_IDS = {
    "B02": "1",
    "B03": "2",
    "B04": "3",
    "B08": "7",
    "B11": "11",
    "B12": "12",
}


def parse_radiometry(xml_bytes, baseline):
    """
    Extract BOA reflectance scaling from
    Sentinel-2 product metadata.

    Return:
        quantification_value, band_offsets
    """
    root = ET.fromstring(xml_bytes)

    quantification = None
    offsets_by_id = {}

    for element in root.iter():
        tag = element.tag.split("}")[-1]

        if tag == "BOA_QUANTIFICATION_VALUE":
            quantification = float(element.text)

        if tag == "BOA_ADD_OFFSET":
            band_id = element.attrib.get("band_id")

            if band_id is not None:
                offsets_by_id[band_id] = float(
                    element.text
                )

    if quantification is None or quantification <= 0:
        raise ValueError(
            "BOA quantification value missing or invalid"
        )

    offsets = {}

    for band in BANDS:
        band_id = BAND_IDS[band]

        if band_id in offsets_by_id:
            offsets[band] = offsets_by_id[band_id]

        elif float(baseline) < 4.0:
            offsets[band] = 0.0

        else:
            raise ValueError(
                f"Missing BOA offset for {band}, "
                f"processing baseline {baseline}"
            )

    return quantification, offsets


def fetch_radiometry(item):
    """
    Fetch product XML instead of guessing
    scaling from the acquisition year.
    """
    if "product-metadata" not in item.assets:
        raise ValueError(
            f"Missing product metadata: {item.id}"
        )

    url = pc.sign(
        item.assets["product-metadata"].href
    )

    response = requests.get(url, timeout=60)
    response.raise_for_status()

    baseline = item.properties.get(
        "s2:processing_baseline"
    )

    if baseline is None:
        raise ValueError(
            f"Missing processing baseline: {item.id}"
        )

    return parse_radiometry(
        response.content,
        baseline,
    )


def read_asset(item, band, grid):
    """
    Read one Sentinel-2 asset and place it
    onto the common UTM analysis grid.

    Bilinear resampling for reflectance.
    Nearest-neighbor resampling for SCL.
    """
    if band not in item.assets:
        raise KeyError(
            f"{band} not available in {item.id}"
        )

    url = pc.sign(item.assets[band].href)

    method = (
        Resampling.nearest
        if band == "SCL"
        else Resampling.bilinear
    )

    with rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        GDAL_HTTP_MAX_RETRY="3",
        GDAL_HTTP_RETRY_DELAY="2",
    ):
        with rasterio.open(url) as src:
            with WarpedVRT(
                src,
                crs=grid.crs,
                transform=grid.transform,
                width=grid.width,
                height=grid.height,
                resampling=method,
                src_nodata=0,
                nodata=0,
            ) as vrt:

                data = vrt.read(
                    1,
                    masked=True,
                    out_dtype="float32",
                )

                return np.asarray(
                    data.filled(0),
                    dtype=np.float32,
                )


def read_optical_scene(item, grid):
    """
    Return raw six-band digital numbers,
    SCL, quantification and offsets.
    """
    quantification, offsets = fetch_radiometry(
        item
    )

    raw_bands = [
        read_asset(item, band, grid)
        for band in BANDS
    ]

    raw = np.stack(
        raw_bands,
        axis=0,
    )

    scl = read_asset(
        item,
        "SCL",
        grid,
    ).astype(np.uint8)

    return raw, scl, quantification, offsets
