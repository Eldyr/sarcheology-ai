
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

from pystac_client import Client
from shapely.geometry import shape


def load_config(path="configs/data.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_aoi(path):
    """Read the first polygon from our GeoJSON."""

    with open(path, "r", encoding="utf-8") as f:
        geojson = json.load(f)

    geometry = shape(
        geojson["features"][0]["geometry"]
    )

    if geometry.is_empty or not geometry.is_valid:
        raise ValueError("Invalid AOI polygon")

    return geometry


def connect_catalog(url):
    """Connect to the public STAC catalog."""

    return Client.open(url)


def search_collection(
    catalog,
    collection,
    bbox,
    date_range=None,
    max_items=300,
):
    """Return STAC items intersecting the AOI."""

    arguments = {
        "collections": [collection],
        "bbox": list(bbox),
        "max_items": max_items,
    }

    if date_range is not None:
        arguments["datetime"] = date_range

    results = catalog.search(**arguments)

    return list(results.items())


def select_sar_items(items):
    """
    Keep Sentinel-1 IW scenes with VV and VH.

    For the initial experiment, select the most
    populated orbit direction / relative-orbit group.
    """

    groups = defaultdict(list)

    for item in items:
        props = item.properties

        mode = props.get("sar:instrument_mode")
        polarizations = set(
            props.get("sar:polarizations", [])
        )

        if mode != "IW":
            continue

        if not {"VV", "VH"}.issubset(polarizations):
            continue

        if not {"vv", "vh"}.issubset(item.assets):
            continue

        orbit_state = props.get("sat:orbit_state")
        relative_orbit = props.get(
            "sat:relative_orbit"
        )

        if orbit_state is None or relative_orbit is None:
            continue

        group_key = (orbit_state, relative_orbit)
        groups[group_key].append(item)

    if not groups:
        raise RuntimeError(
            "No matching Sentinel-1 IW VV/VH scenes "
            "were found. Try a larger AOI or date range."
        )

    best_group = max(groups, key=lambda k: len(groups[k]))

    print("Selected SAR orbit group:", best_group)

    return sorted(
        groups[best_group],
        key=lambda item: item.datetime,
    )


def select_optical_items(items, max_cloud=40):
    """Keep scenes with usable optical assets."""

    required = {
        "B02", "B03", "B04",
        "B08", "B11", "B12", "SCL",
    }

    selected = []

    for item in items:
        cloud = item.properties.get("eo:cloud_cover")

        if cloud is None or cloud > max_cloud:
            continue

        if not required.issubset(item.assets):
            continue

        selected.append(item)

    return sorted(
        selected,
        key=lambda item: item.datetime,
    )


def save_items(items, collection, output_root):
    """
    Save original STAC item metadata and a Parquet
    manifest. We intentionally do not save signed
    URLs because their access tokens expire.
    """

    item_dir = Path(output_root) / collection
    item_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for item in items:
        item_path = item_dir / f"{item.id}.json"

        item_path.write_text(
            json.dumps(item.to_dict(), indent=2),
            encoding="utf-8",
        )

        props = item.properties

        rows.append({
            "id": item.id,
            "collection": collection,
            "datetime": (
                item.datetime.isoformat()
                if item.datetime else None
            ),
            "platform": props.get("platform"),
            "orbit_state": props.get("sat:orbit_state"),
            "relative_orbit": props.get(
                "sat:relative_orbit"
            ),
            "polarizations": ",".join(
                props.get("sar:polarizations", [])
            ),
            "cloud_cover": props.get("eo:cloud_cover"),
            "processing_baseline": props.get(
                "s2:processing_baseline"
            ),
            "item_path": str(item_path),
        })

    return pd.DataFrame(rows)


def run_discovery():
    config = load_config()

    aoi = load_aoi(
        config["study_area"]["geojson"]
    )

    bbox = aoi.bounds

    print("Study area bounds:", bbox)
    print("AOI valid:", aoi.is_valid)

    url = config["satellite_sources"]["stac_url"]
    catalog = connect_catalog(url)

    start = config["time_range"]["start"]
    end = config["time_range"]["end"]
    date_range = f"{start}/{end}"

    source = config["satellite_sources"]

    # 1. Sentinel-1 RTC
    sar_config = source["sentinel1"]

    sar_all = search_collection(
        catalog,
        sar_config["collection"],
        bbox,
        date_range,
        sar_config["max_items"],
    )

    sar_selected = select_sar_items(sar_all)

    # 2. Sentinel-2 L2A
    optical_config = source["sentinel2"]

    optical_all = search_collection(
        catalog,
        optical_config["collection"],
        bbox,
        date_range,
        optical_config["max_items"],
    )

    optical_selected = select_optical_items(
        optical_all,
        optical_config["max_cloud_cover"],
    )

    # 3. Copernicus DEM
    dem_config = source["dem"]

    dem_selected = [
        item for item in search_collection(
            catalog,
            dem_config["collection"],
            bbox,
            max_items=dem_config["max_items"],
        )
        if "data" in item.assets
    ]

    print("\nDiscovery results")
    print("-----------------")
    print("SAR scenes:", len(sar_selected))
    print("Optical scenes:", len(optical_selected))
    print("DEM tiles:", len(dem_selected))

    scene_root = config["paths"]["scene_metadata"]

    manifest_root = Path(
        config["paths"]["manifests"]
    )
    manifest_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    datasets = {
        "sar": sar_selected,
        "optical": optical_selected,
        "dem": dem_selected,
    }

    for name, items in datasets.items():
        if not items:
            print(f"WARNING: No {name} scenes found")
            continue

        df = save_items(
            items,
            name,
            scene_root,
        )

        output = manifest_root / f"{name}_scenes.parquet"
        df.to_parquet(output, index=False)

        print(f"Saved: {output}")


if __name__ == "__main__":
    run_discovery()
