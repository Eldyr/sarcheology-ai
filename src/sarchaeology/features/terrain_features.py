"""Placeholder module for SARchaeology AI v1.0."""

import numpy as np

from scipy.ndimage import gaussian_filter


FEATURE_NAMES = (
    "elevation_m",
    "slope_degrees",
    "profile_curvature_proxy",
    "plan_curvature_proxy",
    "tpi_m",
)


def calculate_terrain_features(
    dem,
    resolution_m=30.0,
    smoothing_sigma=1.0,
    tpi_sigma=10.0,
):
    """
    Calculate terrain features on a projected
    elevation grid with pixel sizes in metres.

    Input:
        dem: [height, width]

    Output:
        features: [5, height, width]

    Curvature channels are stable directional
    second-derivative proxies rather than exact
    GIS-specific curvature conventions.
    """

    z = np.asarray(dem, dtype=np.float64)

    if z.ndim != 2:
        raise ValueError("DEM must be two-dimensional")

    if min(z.shape) < 3:
        raise ValueError("DEM must be at least 3x3")

    if not np.all(np.isfinite(z)):
        raise ValueError(
            "DEM contains missing pixels; resolve "
            "coverage before terrain derivatives."
        )

    if resolution_m <= 0:
        raise ValueError("Invalid pixel resolution")

    # Suppress small-scale elevation noise.
    smoothed = gaussian_filter(
        z,
        sigma=smoothing_sigma,
        mode="reflect",
    )

    # Raster rows increase toward the south.
    # Geographic y increases toward the north.
    dz_drow, dz_dx = np.gradient(
        smoothed,
        resolution_m,
        resolution_m,
    )

    p = dz_dx
    q = -dz_drow

    gradient_squared = p**2 + q**2

    # Slope in degrees.
    slope = np.degrees(
        np.arctan(np.sqrt(gradient_squared))
    )

    # Second derivatives in projected metres.
    r = np.gradient(
        p, resolution_m, axis=1
    )

    s = -np.gradient(
        p, resolution_m, axis=0
    )

    t = -np.gradient(
        q, resolution_m, axis=0
    )

    # Directional curvature proxies.
    denominator = np.maximum(
        gradient_squared,
        1e-8,
    )

    profile_curvature = (
        p**2 * r
        + 2 * p * q * s
        + q**2 * t
    ) / denominator

    plan_curvature = (
        q**2 * r
        - 2 * p * q * s
        + p**2 * t
    ) / denominator

    # On almost-flat areas, direction is undefined.
    nearly_flat = gradient_squared <= 1e-8

    profile_curvature[nearly_flat] = 0.0
    plan_curvature[nearly_flat] = 0.0

    # Topographic Position Index:
    # elevation minus broader neighborhood mean.
    local_mean = gaussian_filter(
        smoothed,
        sigma=tpi_sigma,
        mode="reflect",
    )

    tpi = smoothed - local_mean

    features = np.stack(
        [
            z,
            slope,
            profile_curvature,
            plan_curvature,
            tpi,
        ],
        axis=0,
    )

    return features.astype(np.float32)
