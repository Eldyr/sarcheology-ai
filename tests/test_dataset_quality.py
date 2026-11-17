
import numpy as np
import pytest

from sarchaeology.evaluation.dataset_quality import (
    inspect_patch_archive,
)


def make_example(tmp_path):
    """
    Build one synthetic, fully valid patch.
    """
    size = 2

    arrays = {
        "features": np.ones(
            (19, size, size),
            dtype=np.float32,
        ),
        "masks": np.ones(
            (4, size, size),
            dtype=np.uint8,
        ),
        "quality": np.zeros(
            (3, size, size),
            dtype=np.uint16,
        ),
        "sar": np.full(
            (3, 2, size, size),
            -12.0,
            dtype=np.float32,
        ),
        "sar_valid": np.ones(
            (3, size, size),
            dtype=np.uint8,
        ),
        "optical": np.full(
            (2, 6, size, size),
            0.2,
            dtype=np.float32,
        ),
        "optical_valid": np.ones(
            (2, size, size),
            dtype=np.uint8,
        ),
    }

    arrays["quality"][0] = 3
    arrays["quality"][1] = 2
    arrays["quality"][2] = 1

    metadata = {
        "patch_size": 2,
        "crs": "EPSG:32636",
        "reference_transform": [
            10, 0, 0,
            0, -10, 20,
        ],
        "sar_sources": [
            {"id": f"sar_{i}"}
            for i in range(3)
        ],
        "optical_sources": [
            {"quarter": f"2024Q{i}"}
            for i in (1, 2)
        ],
    }

    record = {
        "patch_id": "r00000_c00000",
        "row": 0,
        "col": 0,
        "xmin": 0,
        "ymin": 0,
        "xmax": 20,
        "ymax": 20,
        "crs": "EPSG:32636",
        "joint_valid_fraction": 1.0,
    }

    path = tmp_path / "example.npz"

    return path, arrays, metadata, record


def save_example(path, arrays):
    np.savez_compressed(path, **arrays)


def test_valid_patch_passes(tmp_path):
    path, arrays, metadata, record = (
        make_example(tmp_path)
    )

    save_example(path, arrays)

    result = inspect_patch_archive(
        path,
        record,
        metadata,
    )

    assert result["joint_valid_fraction"] == 1.0
    assert result["archive_bytes"] > 0


def test_invalid_joint_mask_rejected(tmp_path):
    path, arrays, metadata, record = (
        make_example(tmp_path)
    )

    # Incorrect joint mask:
    # all modalities valid, but joint says invalid.
    arrays["masks"][3, 0, 0] = 0

    save_example(path, arrays)

    with pytest.raises(
        ValueError,
        match="Joint validity mask",
    ):
        inspect_patch_archive(
            path,
            record,
            metadata,
        )


def test_missing_feature_rejected(tmp_path):
    path, arrays, metadata, record = (
        make_example(tmp_path)
    )

    arrays["features"][0, 0, 0] = np.nan

    save_example(path, arrays)

    with pytest.raises(
        ValueError,
        match="Invalid engineered features",
    ):
        inspect_patch_archive(
            path,
            record,
            metadata,
        )


def test_sar_count_mismatch_rejected(tmp_path):
    path, arrays, metadata, record = (
        make_example(tmp_path)
    )

    arrays["quality"][0, 0, 0] = 2

    save_example(path, arrays)

    with pytest.raises(
        ValueError,
        match="SAR observation counts",
    ):
        inspect_patch_archive(
            path,
            record,
            metadata,
        )


def test_geographic_mismatch_rejected(tmp_path):
    path, arrays, metadata, record = (
        make_example(tmp_path)
    )

    record["xmax"] = 25.0

    save_example(path, arrays)

    with pytest.raises(
        ValueError,
        match="Geographic footprint mismatch",
    ):
        inspect_patch_archive(
            path,
            record,
            metadata,
        )


def test_low_coverage_rejected(tmp_path):
    path, arrays, metadata, record = (
        make_example(tmp_path)
    )

    # Make one of four pixels invalid across
    # the SAR modality and the joint mask.
    arrays["masks"][0, 0, 0] = 0
    arrays["masks"][3, 0, 0] = 0

    arrays["features"][:7, 0, 0] = np.nan
    arrays["sar"][:, :, 0, 0] = np.nan
    arrays["sar_valid"][:, 0, 0] = 0
    arrays["quality"][0, 0, 0] = 0

    record["joint_valid_fraction"] = 0.75

    save_example(path, arrays)

    with pytest.raises(
        ValueError,
        match="coverage",
    ):
        inspect_patch_archive(
            path,
            record,
            metadata,
            minimum_valid_fraction=0.85,
        )
