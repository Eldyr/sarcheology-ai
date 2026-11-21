
"""Phase 3.1 — inspect and normalize Sentinel-2 data."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sarchaeology.data.optical_dataset import (
    BANDS,
    OpticalDataset,
    fit_optical_statistics,
    create_optical_dataloaders,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size", type=int, default=16
    )
    parser.add_argument(
        "--num-workers", type=int, default=2
    )
    parser.add_argument(
        "--min-valid-fraction",
        type=float, default=0.50,
    )
    args = parser.parse_args()

    raw = OpticalDataset(
        ROOT,
        split="train",
        min_valid_fraction=args.min_valid_fraction,
    )

    print("Phase 3.1 — Sentinel-2 optical dataset")
    print("Band order:", BANDS)
    print("Usable training observations:", len(raw))

    statistics = fit_optical_statistics(raw)

    output = (
        ROOT
        / "outputs/statistics"
        / "optical_buto_demo_2024.json"
    )

    output.parent.mkdir(
        parents=True, exist_ok=True
    )

    # Do not silently overwrite statistics from a
    # previous experiment with different parameters.
    if output.exists():
        previous = json.loads(
            output.read_text(encoding="utf-8")
        )
        if previous != statistics:
            raise RuntimeError(
                f"Existing statistics differ: {output}"
            )
    else:
        output.write_text(
            json.dumps(statistics, indent=2),
            encoding="utf-8",
        )

    for i, band in enumerate(BANDS):
        print(
            f"{band}: median={statistics['median'][i]:.6f} "
            f"IQR={statistics['scale'][i]:.6f} "
            f"p01={statistics['p01'][i]:.6f} "
            f"p99={statistics['p99'][i]:.6f}"
        )

    datasets, loaders = create_optical_dataloaders(
        ROOT,
        statistics,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        min_valid_fraction=args.min_valid_fraction,
    )

    print("\nTraining samples:", len(datasets["train"]))
    print("Validation samples:", len(datasets["val"]))

    # Verify a batch from BOTH splits.
    for split in ("train", "val"):
        batch = next(iter(loaders[split]))

        x = batch["image"]
        valid = batch["valid_mask"]

        assert x.ndim == 4
        assert x.shape[1:] == (6, 128, 128)
        assert valid.shape == (
            len(x), 1, 128, 128
        )
        assert torch.isfinite(x).all()
        assert valid.any()

        print(
            f"{split}: image={tuple(x.shape)}, "
            f"mask={tuple(valid.shape)}, "
            f"coverage={valid.float().mean().item():.3f}"
        )

    summary = {
        "phase": "3.1",
        "dataset_id": "buto_demo_2024",
        "bands": list(BANDS),
        "training_samples": len(datasets["train"]),
        "validation_samples": len(datasets["val"]),
        "training_patches": len({
            s["patch_id"] for s in raw.samples
        }),
        "input_shape": ["B", 6, 128, 128],
        "normalization": str(output.relative_to(ROOT)),
        "train_manifest_sha256": raw.manifest_hash,
        "status": "DATASET_CHECKS_PASSED",
    }

    report = (
        ROOT / "outputs/quality"
        / "phase3_optical_dataset.json"
    )
    report.parent.mkdir(
        parents=True, exist_ok=True
    )
    report.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n" + json.dumps(summary, indent=2))
    print("\nPhase 3.1 checks passed.")


if __name__ == "__main__":
    main()
