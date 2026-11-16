
import numpy as np

from sarchaeology.features.sar_features import (
    calculate_sar_temporal_features,
)

from sarchaeology.features.optical_features import (
    calculate_optical_indices,
    summarize_optical_quarters,
)


def test_sar_temporal_statistics():
    stack = np.zeros(
        (3, 2, 2, 2),
        dtype=np.float32,
    )

    for i, (vv, vh) in enumerate(
        [(-10, -20), (-12, -22), (-14, -24)]
    ):
        stack[i, 0] = vv
        stack[i, 1] = vh

    features, count = (
        calculate_sar_temporal_features(
            stack,
            min_valid_acquisitions=3,
        )
    )

    assert features.shape == (7, 2, 2)
    assert np.all(count == 3)

    assert np.allclose(features[0], -12)
    assert np.allclose(features[1], -22)

    assert np.allclose(features[4], 2)
    assert np.allclose(features[5], 2)

    assert np.allclose(features[6], 10)


def test_sar_requires_paired_observations():
    stack = np.full(
        (3, 2, 1, 2),
        -10.0,
        dtype=np.float32,
    )

    stack[:, 1] = -20.0

    # One acquisition lacks VV at pixel 1.
    stack[0, 0, 0, 1] = np.nan

    features, counts = (
        calculate_sar_temporal_features(
            stack,
            min_valid_acquisitions=3,
        )
    )

    assert counts[0, 0] == 3
    assert counts[0, 1] == 2

    assert np.isfinite(features[:, 0, 0]).all()
    assert np.isnan(features[:, 0, 1]).all()


def test_optical_index_formulas():
    reflectance = np.zeros(
        (6, 2, 2),
        dtype=np.float32,
    )

    reflectance[0] = 0.1  # Blue
    reflectance[1] = 0.2  # Green
    reflectance[2] = 0.2  # Red
    reflectance[3] = 0.6  # NIR
    reflectance[4] = 0.3  # SWIR1
    reflectance[5] = 0.2  # SWIR2

    indices = calculate_optical_indices(
        reflectance
    )

    assert indices.shape == (5, 2, 2)

    assert np.allclose(indices[0], 0.5)
    assert np.allclose(indices[1], 1 / 3)
    assert np.allclose(indices[2], -1 / 6)
    assert np.allclose(indices[3], -0.2)
    assert np.allclose(indices[4], 0.5)


def test_optical_invalid_pixels():
    reflectance = np.full(
        (6, 2, 2),
        0.2,
        dtype=np.float32,
    )

    reflectance[:, 0, 0] = -9999

    valid_mask = np.array([
        [True, False],
        [True, True],
    ])

    indices = calculate_optical_indices(
        reflectance,
        valid_mask=valid_mask,
    )

    assert np.isnan(indices[:, 0, 0]).all()
    assert np.isnan(indices[:, 0, 1]).all()
    assert np.isfinite(indices[:, 1, 0]).all()


def test_annual_optical_statistics():
    stack = np.full(
        (3, 5, 2, 2),
        0.2,
        dtype=np.float32,
    )

    stack[0, 0] = 0.1
    stack[1, 0] = 0.3
    stack[2, 0] = 0.5

    # Only one valid quarter for this pixel.
    stack[1:, :, 1, 1] = np.nan

    features, counts = (
        summarize_optical_quarters(
            stack,
            min_valid_quarters=2,
        )
    )

    assert features.shape == (7, 2, 2)

    assert counts[0, 0] == 3
    assert counts[1, 1] == 1

    assert np.isclose(
        features[0, 0, 0], 0.3
    )

    assert np.isclose(
        features[1, 0, 0], 0.4
    )

    assert np.isnan(
        features[:, 1, 1]
    ).all()
