"""Placeholder module for SARchaeology AI v1.0."""

import warnings

import numpy as np


OPTICAL_INDEX_NAMES = (
    "ndvi",
    "ndmi",
    "bsi",
    "mndwi",
    "nbr",
)


OPTICAL_ANNUAL_NAMES = (
    "ndvi_median",
    "ndvi_range",
    "ndvi_std",
    "ndmi_median",
    "bsi_median",
    "mndwi_median",
    "nbr_median",
)


def safe_ratio(numerator, denominator, valid, eps=1e-6):
    """
    Divide only where the denominator is
    sufficiently far from zero.
    """

    numerator = np.asarray(
        numerator, dtype=np.float32
    )

    denominator = np.asarray(
        denominator, dtype=np.float32
    )

    result = np.full(
        numerator.shape,
        np.nan,
        dtype=np.float32,
    )

    safe = (
        valid
        & np.isfinite(numerator)
        & np.isfinite(denominator)
        & (np.abs(denominator) > eps)
    )

    np.divide(
        numerator,
        denominator,
        out=result,
        where=safe,
    )

    return result


def calculate_optical_indices(
    reflectance,
    valid_mask=None,
    nodata=-9999.0,
):
    """
    Calculate five Sentinel-2 spectral indices.

    Input shape:
        [6, height, width]

    Band order:
        B02, B03, B04, B08, B11, B12

    Returns:
        [5, height, width]
    """

    x = np.asarray(
        reflectance,
        dtype=np.float32,
    )

    if x.ndim != 3 or x.shape[0] != 6:
        raise ValueError(
            "Expected six Sentinel-2 reflectance bands"
        )

    valid = np.all(
        np.isfinite(x) & (x != nodata),
        axis=0,
    )

    if valid_mask is not None:
        valid_mask = np.asarray(
            valid_mask,
            dtype=bool,
        )

        if valid_mask.shape != valid.shape:
            raise ValueError(
                "Optical validity mask shape mismatch"
            )

        valid &= valid_mask

    blue, green, red, nir, swir1, swir2 = x

    # Normalized Difference Vegetation Index
    ndvi = safe_ratio(
        nir - red,
        nir + red,
        valid,
    )

    # Normalized Difference Moisture Index
    ndmi = safe_ratio(
        nir - swir1,
        nir + swir1,
        valid,
    )

    # Bare Soil Index
    bsi = safe_ratio(
        (swir1 + red) - (nir + blue),
        (swir1 + red) + (nir + blue),
        valid,
    )

    # Modified Normalized Difference Water Index
    mndwi = safe_ratio(
        green - swir1,
        green + swir1,
        valid,
    )

    # Normalized Burn Ratio
    nbr = safe_ratio(
        nir - swir2,
        nir + swir2,
        valid,
    )

    return np.stack(
        [ndvi, ndmi, bsi, mndwi, nbr],
        axis=0,
    ).astype(np.float32)


def summarize_optical_quarters(
    stack,
    min_valid_quarters=2,
):
    """
    Summarize quarterly spectral indices.

    Input:
        [quarters, 5, height, width]

    Output:
        features: [7, height, width]
        count: [height, width]
    """

    stack = np.asarray(
        stack,
        dtype=np.float32,
    )

    if stack.ndim != 4 or stack.shape[1] != 5:
        raise ValueError(
            "Expected [quarters, 5, height, width]"
        )

    if not 1 <= min_valid_quarters <= stack.shape[0]:
        raise ValueError(
            "Invalid minimum quarterly count"
        )

    # A quarter counts as valid only when all
    # five calculated indices are available.
    valid = np.all(
        np.isfinite(stack),
        axis=1,
    )

    counts = np.sum(
        valid,
        axis=0,
    ).astype(np.uint16)

    clean = np.where(
        valid[:, None, :, :],
        stack,
        np.nan,
    )

    ndvi = clean[:, 0]
    ndmi = clean[:, 1]
    bsi = clean[:, 2]
    mndwi = clean[:, 3]
    nbr = clean[:, 4]

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            RuntimeWarning,
        )

        ndvi_median = np.nanmedian(
            ndvi, axis=0
        )

        ndvi_range = (
            np.nanmax(ndvi, axis=0)
            - np.nanmin(ndvi, axis=0)
        )

        ndvi_std = np.nanstd(
            ndvi, axis=0
        )

        ndmi_median = np.nanmedian(
            ndmi, axis=0
        )

        bsi_median = np.nanmedian(
            bsi, axis=0
        )

        mndwi_median = np.nanmedian(
            mndwi, axis=0
        )

        nbr_median = np.nanmedian(
            nbr, axis=0
        )

    features = np.stack(
        [
            ndvi_median,
            ndvi_range,
            ndvi_std,
            ndmi_median,
            bsi_median,
            mndwi_median,
            nbr_median,
        ],
        axis=0,
    ).astype(np.float32)

    features[:, counts < min_valid_quarters] = np.nan

    return features, counts
