
"""Phase 2.11: Validate SARMAE adapter on real SAR data."""

import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sarchaeology.data.dataloaders import (
    create_sar_dataloaders,
)
from sarchaeology.data.sarmae_adapter import (
    SARMAEAdapterConfig,
    SARMAEInputAdapter,
)
from sarchaeology.preprocessing.sar_normalization import (
    SARNormalizer,
)


RUN_ID = "sar_cae_20261118T192046Z_2dbe6bd2"


def main():
    run_dir = (
        ROOT / "outputs/experiments/sar/sar_cae"
        / RUN_ID
    )

    config = yaml.safe_load(
        (run_dir / "config.yaml").read_text(
            encoding="utf-8"
        )
    )

    normalizer = SARNormalizer.from_json(
        run_dir / "sar_normalization.json"
    )

    _, loaders = create_sar_dataloaders(
        project_root=ROOT,
        config=config,
        include_test=False,
        normalizer=normalizer,
    )

    batch = next(iter(loaders["val"]))

    assert batch["image"].shape[1:] == (
        2, 128, 128
    )

    # The existing DataLoader emits normalized SAR.
    # This is a pipeline smoke test, not the final
    # scientifically preferred raw-dB pathway.
    adapter = SARMAEInputAdapter(
        SARMAEAdapterConfig(
            input_mode="normalized_proxy"
        )
    )

    adapted = adapter(
        batch["image"],
        batch["valid_mask"],
    )

    pixel_values = adapted["pixel_values"]
    pixel_mask = adapted["pixel_mask"]

    assert pixel_values.shape == (
        2 * batch["image"].shape[0],
        3, 224, 224,
    )
    assert pixel_mask.shape[0] == (
        pixel_values.shape[0]
    )
    assert torch.isfinite(pixel_values).all()

    report = {
        "phase": "2.11",
        "run_id": RUN_ID,
        "dataset_id": config["dataset"]["dataset_id"],
        "input_mode": "normalized_proxy",
        "original_shape": list(
            batch["image"].shape
        ),
        "adapted_shape": list(
            pixel_values.shape
        ),
        "valid_token_input_fraction": float(
            pixel_mask.float().mean()
        ),
        "pixel_values_min": float(
            pixel_values.min()
        ),
        "pixel_values_max": float(
            pixel_values.max()
        ),
        "polarization_order": "VV_then_VH",
        "pretrained_encoder_loaded": False,
        "status": "PASS",
    }

    output_dir = (
        ROOT / "outputs/inspections/sarmae"
    )
    output_dir.mkdir(
        parents=True, exist_ok=True
    )

    (output_dir / "adapter_smoke.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nSARchaeology AI — Phase 2.11")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
