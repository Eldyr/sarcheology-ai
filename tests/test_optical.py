
import numpy as np
import rasterio

from rasterio.transform import from_origin

from sarchaeology.ingestion.sentinel2 import (
    BANDS,
    BAND_IDS,
    parse_radiometry,
)

from sarchaeology.preprocessing.sar import GridSpec

from sarchaeology.preprocessing.optical import (
    convert_to_reflectance,
    make_valid_mask,
    save_optical_scene,
    make_quarterly_composite,
)


def test_radiometric_metadata():
    offsets_xml = "".join(
        f'<BOA_ADD_OFFSET band_id="{BAND_IDS[b]}">'
        f'-1000</BOA_ADD_OFFSET>'
        for b in BANDS
    )

    xml = (
        "<root>"
        "<BOA_QUANTIFICATION_VALUE>10000"
        "</BOA_QUANTIFICATION_VALUE>"
        f"<BOA_ADD_OFFSET_VALUES_LIST>{offsets_xml}"
        "</BOA_ADD_OFFSET_VALUES_LIST>"
        "</root>"
    )

    quant, offsets = parse_radiometry(
        xml.encode("utf-8"),
        "05.10",
    )

    assert quant == 10000
    assert all(value == -1000 for value in offsets.values())


def test_reflectance_conversion():
    raw = np.full(
        (6, 2, 2),
        3500,
        dtype=np.float32,
    )

    raw[0, 0, 0] = 0

    offsets = {
        band: -1000
        for band in BANDS
    }

    result = convert_to_reflectance(
        raw,
        quantification=10000,
        offsets=offsets,
    )

    assert result.shape == (6, 2, 2)
    assert np.isclose(result[1, 0, 0], 0.25)
    assert np.isnan(result[0, 0, 0])


def test_cloud_mask():
    raw = np.full(
        (6, 2, 2),
        3000,
        dtype=np.float32,
    )

    scl = np.array([
        [4, 9],
        [5, 3],
    ])

    aoi_mask = np.ones(
        (2, 2), dtype=bool
    )

    valid = make_valid_mask(
        raw,
        scl,
        aoi_mask,
        invalid_classes=[0, 1, 3, 8, 9, 10, 11],
        cloud_buffer_pixels=0,
    )

    expected = np.array([
        [True, False],
        [True, False],
    ])

    assert np.array_equal(valid, expected)


def test_quarterly_composite(tmp_path):
    grid = GridSpec(
        crs="EPSG:32636",
        transform=from_origin(0, 20, 10, 10),
        width=2,
        height=2,
        aoi_mask=np.ones((2, 2), dtype=bool),
    )

    valid = np.array([
        [True, True],
        [True, False],
    ])

    scl = np.full(
        (2, 2), 4, dtype=np.uint8
    )

    offsets = {
        band: -1000
        for band in BANDS
    }

    paths = []

    for index, value in enumerate((0.2, 0.4)):
        reflectance = np.full(
            (6, 2, 2),
            value,
            dtype=np.float32,
        )

        output = save_optical_scene(
            reflectance=reflectance,
            valid=valid,
            scl=scl,
            grid=grid,
            directory=tmp_path / "scenes",
            stem=f"scene_{index}",
            item_id=f"test_{index}",
            baseline="05.10",
            quantification=10000,
            offsets=offsets,
        )

        paths.append(output["reflectance"])

    median_path, count_path = (
        make_quarterly_composite(
            paths,
            tmp_path / "composites",
            "2024Q1",
            minimum_observations=2,
        )
    )

    with rasterio.open(median_path) as ds:
        result = ds.read()

        assert result.shape == (6, 2, 2)
        assert np.isclose(result[0, 0, 0], 0.3)
        assert result[0, 1, 1] == -9999.0
        assert ds.crs.to_epsg() == 32636

    with rasterio.open(count_path) as ds:
        counts = ds.read(1)

        assert counts[0, 0] == 2
        assert counts[1, 1] == 0
