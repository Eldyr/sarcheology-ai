"""Placeholder module for SARchaeology AI v1.0."""

import warnings

import numpy as np


SAR_FEATURE_NAMES = (
    "vv_median_db",
    "vh_median_db",
    "vv_std_db",
    "vh_std_db",
    "vv_iqr_db",
    "vh_iqr_db",
    "vv_minus_vh_median_db",
)


def calculate_sar_temporal_features(
    stack,
    min_valid_acquisitions=3,
):
    """
    Calculate temporal features from aligned
    Sentinel-1 VV/VH decibel acquisitions.

    Parameters
    ----------
    stack : np.ndarray
        Shape [time, 2, height, width].
        Band 0 = VV, band 1 = VH.
        Invalid data must be NaN.

    min_valid_acquisitions : int
        Required number of valid paired VV/VH
        observations per pixel.

    Returns
    -------
    features : np.ndarray
        Shape [7, height, width].

    count : np.ndarray
        Shape [height, width], uint16.
    """

    stack = np.asarray(stack, dtype=np.float32)

    if stack.ndim != 4 or stack.shape[1] != 2:
        raise ValueError(
            "Expected [time, 2, height, width]"
        )

    if not 1 <= min_valid_acquisitions <= stack.shape[0]:
        raise ValueError(
            "Invalid minimum acquisition count"
        )

    # A temporal observation is accepted only
    # when VV and VH are both available.
    paired_valid = np.all(
        np.isfinite(stack),
        axis=1,
    )

    count = np.sum(
        paired_valid,
        axis=0,
    ).astype(np.uint16)

    clean = np.where(
        paired_valid[:, None, :, :],
        stack,
        np.nan,
    )

    vv = clean[:, 0]
    vh = clean[:, 1]

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore", RuntimeWarning
        )

        vv_median = np.nanmedian(vv, axis=0)
        vh_median = np.nanmedian(vh, axis=0)

        vv_std = np.nanstd(vv, axis=0)
        vh_std = np.nanstd(vh, axis=0)

        vv_iqr = (
            np.nanpercentile(vv, 75, axis=0)
            - np.nanpercentile(vv, 25, axis=0)
        )

        vh_iqr = (
            np.nanpercentile(vh, 75, axis=0)
            - np.nanpercentile(vh, 25, axis=0)
        )

        # Calculate on matching acquisitions,
        # not as the difference of two medians.
        polarization_difference = np.nanmedian(
            vv - vh,
            axis=0,
        )

    features = np.stack(
        [
            vv_median,
            vh_median,
            vv_std,
            vh_std,
            vv_iqr,
            vh_iqr,
            polarization_difference,
        ],
        axis=0,
    ).astype(np.float32)

    insufficient = (
        count < min_valid_acquisitions
    )

    features[:, insufficient] = np.nan

    return features, count
