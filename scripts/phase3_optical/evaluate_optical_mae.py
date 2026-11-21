
"""Phase 3.3.1 — Evaluate Optical ViT-Tiny MAE.

Uses exactly the same deterministic validation masks
as the Phase 3.3 trainer.

Outputs:
- metrics.json
- validation_embeddings.npz
- embedding_pca.png
- reconstruction_*.png
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sarchaeology.data.optical_dataset import (
    BANDS,
    OpticalDataset,
)
from sarchaeology.models.optical_mae import (
    OpticalViTMAE,
)


DEFAULT_RUN = (
    "optical_mae_20261121T123946Z_4dff37c7"
)


def sha256(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def load_experiment(run_id, device):
    checkpoint_path = (
        ROOT / "checkpoints/optical_mae"
        / run_id / "best.pt"
    )

    config_path = (
        ROOT / "configs/training/optical_mae.yaml"
    )

    config = yaml.safe_load(
        config_path.read_text(encoding="utf-8")
    )

    stats_path = (
        ROOT / config["data"]["statistics_path"]
    )

    stats = json.loads(
        stats_path.read_text(encoding="utf-8")
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    for key, path in (
        ("config_sha256", config_path),
        ("normalization_sha256", stats_path),
    ):
        if checkpoint[key] != sha256(path):
            raise RuntimeError(
                f"Checkpoint fingerprint mismatch: {key}. "
                "Restore the original experiment "
                "configuration or normalization."
            )

    if tuple(stats["bands"]) != BANDS:
        raise RuntimeError("Optical band mismatch")

    dataset = OpticalDataset(
        ROOT,
        split="val",
        statistics=stats,
        min_valid_fraction=float(
            config["data"]["min_valid_fraction"]
        ),
    )

    if (
        dataset.manifest_hash
        != checkpoint["val_manifest_sha256"]
    ):
        raise RuntimeError(
            "Validation manifest fingerprint mismatch"
        )

    # Preserve training batch boundaries because
    # the validation masking seed uses batch_index.
    loader = DataLoader(
        dataset,
        batch_size=int(
            config["data"]["batch_size"]
        ),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = OpticalViTMAE(
        **config["model"]
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state"],
        strict=True,
    )

    model.eval()

    return (
        model,
        loader,
        config,
        stats,
        checkpoint,
    )


def make_mask(patch_mask, patch_size):
    """[B,N] patch masks -> [B,H,W] pixel masks."""
    b, n = patch_mask.shape
    g = int(n ** 0.5)

    if g * g != n:
        raise ValueError("Patch grid is not square")

    return (
        patch_mask.reshape(b, g, g)
        .repeat_interleave(patch_size, dim=1)
        .repeat_interleave(patch_size, dim=2)
    )


def empty_totals(device):
    return {
        key: torch.zeros(6, device=device, dtype=torch.float64)
        for key in ("abs", "sq", "huber", "count")
    }


def accumulate(totals, prediction, target, selected):
    """Accumulate per-band errors on selected pixels."""
    # [B,6,H,W]
    selected = selected.expand_as(target)
    error = torch.where(
        selected,
        prediction.float() - target.float(),
        torch.zeros_like(target, dtype=torch.float32),
    )

    absolute = error.abs()
    delta = 1.0

    huber = torch.where(
        absolute <= delta,
        0.5 * error.square(),
        delta * (absolute - 0.5 * delta),
    )

    axes = (0, 2, 3)

    totals["abs"] += absolute.sum(
        dim=axes, dtype=torch.float64
    )
    totals["sq"] += error.square().sum(
        dim=axes, dtype=torch.float64
    )
    totals["huber"] += huber.sum(
        dim=axes, dtype=torch.float64
    )
    totals["count"] += selected.sum(
        dim=axes
    ).to(torch.float64)


def summarize(totals):
    values = {
        k: v.detach().cpu().numpy()
        for k, v in totals.items()
    }

    count = values["count"]

    if np.any(count <= 0):
        raise RuntimeError(
            "One or more bands have no scored pixels"
        )

    def metric(sum_key, power=1.0):
        return (
            values[sum_key].sum() / count.sum()
        ) ** power

    return {
        "huber": float(metric("huber")),
        "mae": float(metric("abs")),
        "rmse": float(metric("sq", 0.5)),
        "scored_channel_pixels": int(
            count.sum()
        ),
        "per_band": {
            band: {
                "huber": float(
                    values["huber"][i] / count[i]
                ),
                "mae": float(
                    values["abs"][i] / count[i]
                ),
                "rmse": float(
                    np.sqrt(
                        values["sq"][i] / count[i]
                    )
                ),
                "count": int(count[i]),
            }
            for i, band in enumerate(BANDS)
        },
    }


def denormalize(images, stats):
    median = np.asarray(
        stats["median"], dtype=np.float32
    )[None, :, None, None]

    scale = np.asarray(
        stats["scale"], dtype=np.float32
    )[None, :, None, None]

    return images * scale + median


def prepare_color(image, indices, low, high, valid):
    """Same display stretch for every panel."""
    result = image[list(indices)].transpose(1, 2, 0)
    result = (result - low) / np.maximum(
        high - low, 1e-6
    )
    result = np.clip(result, 0, 1)
    result[~valid] = 0
    return result


def save_reconstruction(
    output_dir,
    index,
    source,
    predicted,
    visible_mask,
    valid_mask,
    stats,
    patch_id,
    quarter,
):
    original = denormalize(
        source[None], stats
    )[0]

    reconstruction = denormalize(
        predicted[None], stats
    )[0]

    completed = np.where(
        visible_mask[None],
        original,
        reconstruction,
    )

    masked_input = np.where(
        visible_mask[None],
        original,
        0,
    )

    selections = [
        ("RGB", (2, 1, 0)),
        ("False color", (3, 2, 1)),
    ]

    fig, axes = plt.subplots(
        2, 5, figsize=(19, 8)
    )

    for r, (label, channels) in enumerate(selections):
        reference = original[list(channels)]

        sample = reference[:, valid_mask]

        if sample.size == 0:
            plt.close(fig)
            return

        low = np.percentile(
            sample, 2, axis=1
        )[None, None, :]

        high = np.percentile(
            sample, 98, axis=1
        )[None, None, :]

        panels = [
            original,
            masked_input,
            reconstruction,
            completed,
        ]

        panel_titles = [
            "Original",
            "Masked input",
            "MAE prediction",
            "Completed image",
        ]

        for c, (data, title) in enumerate(
            zip(panels, panel_titles)
        ):
            rgb = prepare_color(
                data,
                channels,
                low,
                high,
                valid_mask,
            )

            if c == 1:
                rgb[~visible_mask] = 0

            axes[r, c].imshow(rgb)
            axes[r, c].set_title(
                f"{label}: {title}"
            )

        difference = np.mean(
            np.abs(
                predicted[list(channels)]
                - source[list(channels)]
            ),
            axis=0,
        )

        masked_error = np.where(
            (~visible_mask) & valid_mask,
            difference,
            np.nan,
        )

        im = axes[r, 4].imshow(
            masked_error,
            cmap="magma",
        )
        axes[r, 4].set_title(
            f"{label}: masked error (normalized)"
        )

        fig.colorbar(
            im, ax=axes[r, 4], fraction=0.046
        )

        for ax in axes[r]:
            ax.axis("off")

    fig.suptitle(
        f"Patch {patch_id} | quarter index {quarter} "
        f"| sample {index}",
        fontsize=12,
    )

    fig.tight_layout()

    fig.savefig(
        output_dir / f"reconstruction_{index:03d}.png",
        dpi=140,
    )
    plt.close(fig)


def save_embedding_diagnostics(
    output_dir,
    embeddings,
    patch_ids,
    quarters,
):
    np.savez_compressed(
        output_dir / "validation_embeddings.npz",
        embeddings=embeddings,
        patch_ids=np.asarray(
            patch_ids, dtype=str
        ),
        quarter_indices=np.asarray(
            quarters, dtype=np.int64
        ),
    )

    variance = embeddings.var(axis=0)
    centered = embeddings - embeddings.mean(axis=0)

    # 80 observations, 192 encoder features.
    # SVD gives PCA scores without requiring sklearn.
    u, singular, vt = np.linalg.svd(
        centered, full_matrices=False
    )

    coordinates = (
        u[:, :2] * singular[None, :2]
    )

    explained = singular ** 2
    explained = explained / max(
        explained.sum(), 1e-12
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    points = ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=quarters,
        cmap="viridis",
        alpha=0.8,
        s=28,
    )

    ax.set_xlabel(
        f"PC1 ({explained[0] * 100:.1f}%)"
    )
    ax.set_ylabel(
        f"PC2 ({explained[1] * 100:.1f}%)"
    )
    ax.set_title(
        "Validation encoder embeddings"
    )

    fig.colorbar(
        points,
        ax=ax,
        label="Quarter index",
    )

    fig.tight_layout()
    fig.savefig(
        output_dir / "embedding_pca.png",
        dpi=160,
    )
    plt.close(fig)

    return {
        "shape": list(embeddings.shape),
        "mean_feature_variance": float(
            variance.mean()
        ),
        "near_zero_variance_features": int(
            (variance < 1e-8).sum()
        ),
        "pca_explained_variance_pc1": float(
            explained[0]
        ),
        "pca_explained_variance_pc2": float(
            explained[1]
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-id",
        default=DEFAULT_RUN,
    )

    parser.add_argument(
        "--visualizations",
        type=int,
        default=6,
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    (
        model,
        loader,
        config,
        stats,
        checkpoint,
    ) = load_experiment(
        args.run_id,
        device,
    )

    output_dir = (
        ROOT / "outputs/evaluations/optical_mae"
        / args.run_id / "val"
    )

    output_dir.mkdir(
        parents=True, exist_ok=True
    )

    model_totals = empty_totals(device)
    baseline_totals = empty_totals(device)

    embedding_chunks = []
    patch_ids = []
    quarter_indices = []

    observed = 0
    seed = int(
        config["training"]["seed"]
    )

    print(
        "Evaluating checkpoint epoch:",
        checkpoint["epoch"],
    )

    print(
        "Validation observations:",
        len(loader.dataset),
    )

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            images = batch["image"].to(
                device, non_blocking=True
            )

            valid = batch["valid_mask"].to(
                device, non_blocking=True
            ).bool()

            generator = torch.Generator(
                device="cpu"
            )

            # Match Phase 3.3 validation seed EXACTLY.
            generator.manual_seed(
                seed + 202600000 + batch_index
            )

            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                result = model(
                    images,
                    valid,
                    generator=generator,
                )

            pixel_mask = make_mask(
                result["patch_mask"],
                model.patch_size,
            )

            selected = (
                pixel_mask.unsqueeze(1) & valid
            )

            prediction = (
                result["reconstruction"].float()
            )

            baseline = torch.zeros_like(images)

            accumulate(
                model_totals,
                prediction,
                images,
                selected,
            )

            accumulate(
                baseline_totals,
                baseline,
                images,
                selected,
            )

            embedding_chunks.append(
                result["embedding"]
                .float().cpu().numpy()
            )

            patch_ids.extend(
                batch["patch_id"]
            )

            quarters = (
                batch["quarter_index"]
                .cpu().numpy().tolist()
            )

            quarter_indices.extend(quarters)

            remaining = (
                args.visualizations - observed
            )

            if remaining > 0:
                n = min(
                    remaining, len(images)
                )

                images_np = (
                    images[:n].cpu().numpy()
                )
                pred_np = (
                    prediction[:n].cpu().numpy()
                )
                valid_np = (
                    valid[:n, 0].cpu().numpy()
                )
                masked_np = (
                    pixel_mask[:n].cpu().numpy()
                )

                for j in range(n):
                    save_reconstruction(
                        output_dir,
                        observed + j,
                        images_np[j],
                        pred_np[j],
                        ~masked_np[j],
                        valid_np[j],
                        stats,
                        batch["patch_id"][j],
                        quarters[j],
                    )

                observed += n

    model_metrics = summarize(model_totals)
    baseline_metrics = summarize(baseline_totals)

    embeddings = np.concatenate(
        embedding_chunks, axis=0
    )

    diagnostic = save_embedding_diagnostics(
        output_dir,
        embeddings,
        patch_ids,
        quarter_indices,
    )

    baseline_huber = baseline_metrics["huber"]
    model_huber = model_metrics["huber"]

    improvement = (
        1 - model_huber / baseline_huber
    ) * 100

    report = {
        "phase": "3.3.1",
        "run_id": args.run_id,
        "checkpoint_epoch": checkpoint["epoch"],
        "validation_observations": len(loader.dataset),
        "mask_ratio": model.mask_ratio,
        "same_masks_for_both_models": True,
        "model": model_metrics,
        "constant_median_baseline": baseline_metrics,
        "relative_huber_reduction_percent": improvement,
        "embeddings": diagnostic,
        "caveats": [
            "Validation geography is spatially restricted.",
            "Overlapping patches are not independent samples.",
            "Reconstruction error is not archaeology accuracy.",
            "Quarter dates require verified NPZ ordering.",
        ],
    }

    metrics_path = output_dir / "metrics.json"

    metrics_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n--- Validation results ---")
    print(
        "Model masked Huber:",
        round(model_huber, 6),
    )
    print(
        "Baseline masked Huber:",
        round(baseline_huber, 6),
    )
    print(
        "Relative reduction:",
        round(improvement, 2), "%",
    )
    print(
        "Model masked MAE:",
        round(model_metrics["mae"], 6),
    )
    print(
        "Model masked RMSE:",
        round(model_metrics["rmse"], 6),
    )
    print(
        "Embedding variance:",
        diagnostic["mean_feature_variance"],
    )
    print("\nPer-band masked MAE:")

    for band, measures in model_metrics[
        "per_band"
    ].items():
        print(
            f"  {band}: {measures['mae']:.6f}"
        )

    print("\nSaved results:", output_dir)


if __name__ == "__main__":
    main()
