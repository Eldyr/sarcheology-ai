
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_preprocessing_configuration():
    config_path = ROOT / "configs" / "preprocessing.yaml"

    assert config_path.exists()

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    required_sections = [
        "sar",
        "optical",
        "dem",
        "features",
        "alignment",
        "patch_extraction",
        "spatial_split",
    ]

    for section in required_sections:
        assert section in config, (
            f"Missing configuration section: {section}"
        )


def test_alignment_configuration():
    config_path = ROOT / "configs" / "preprocessing.yaml"

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    alignment = config["alignment"]

    assert alignment["year"] == 2024
    assert alignment["output_nodata"] == -9999.0
    assert alignment["require_joint_coverage"] is True


def test_spatial_split_configuration():
    config_path = ROOT / "configs" / "preprocessing.yaml"

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    split = config["spatial_split"]

    total = (
        split["train_fraction"]
        + split["val_fraction"]
        + split["test_fraction"]
    )

    assert abs(total - 1.0) < 1e-8
    assert split["gap_m"] >= 0
    assert split["axis"] in ("x", "y")
