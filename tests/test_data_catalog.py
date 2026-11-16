
from pathlib import Path

import yaml
from shapely.geometry import shape
import json


def test_project_config():
    path = Path("configs/data.yaml")

    assert path.exists(), "data.yaml is missing"

    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assert config["study_area"]["target_crs"] == "EPSG:32636"
    assert config["study_area"]["resolution_m"] == 10

    assert config["patches"]["size"] == 128
    assert config["patches"]["stride"] == 64


def test_aoi_geometry():
    path = Path("configs/aois/buto_demo.geojson")

    assert path.exists(), "AOI GeoJSON is missing"

    with open(path, encoding="utf-8") as f:
        geojson = json.load(f)

    geometry = shape(
        geojson["features"][0]["geometry"]
    )

    assert geometry.is_valid
    assert not geometry.is_empty
    assert geometry.area > 0

    west, south, east, north = geometry.bounds

    assert west < east
    assert south < north


def test_manifest_files():
    manifest_dir = Path("data/metadata/manifests")

    expected = [
        "sar_scenes.parquet",
        "optical_scenes.parquet",
        "dem_scenes.parquet",
    ]

    for filename in expected:
        assert (manifest_dir / filename).exists(), (
            f"Missing manifest: {filename}"
        )
