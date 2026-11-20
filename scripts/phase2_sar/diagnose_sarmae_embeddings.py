
"""
SARchaeology AI — Phase 2.13.1
Frozen SARMAE embedding diagnostics.

Produces:
  diagnostics.json
  feature_variance.csv
  pca_coordinates.csv
  pca_by_polarization.png
  pca_by_split.png
  nearest_neighbor_distances.png
  geographic_novelty.png
  per_patch_diagnostics.csv

No retraining. No test-set access.
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ID = "sar_cae_20261118T192046Z_2dbe6bd2"
PATCH_PATTERN = re.compile(r"r(\d+)_c(\d+)")


def load_split(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as data:
        required = (
            "embedding",
            "patch_id",
            "acquisition_date",
            "polarization",
            "valid_tokens",
            "source_revision",
            "input_mode",
        )
        missing = [k for k in required if k not in data.files]
        if missing:
            raise ValueError(f"Missing keys: {missing}")

        result = {
            key: np.array(data[key], copy=True)
            for key in required
        }

    x = result["embedding"].astype(np.float32)

    if x.ndim != 2 or x.shape[1] != 768:
        raise ValueError(f"Unexpected embeddings: {x.shape}")

    if not np.isfinite(x).all():
        raise ValueError("Embeddings contain NaN/Inf")

    n = len(x)
    for key in (
        "patch_id", "acquisition_date",
        "polarization", "valid_tokens",
    ):
        if len(result[key]) != n:
            raise ValueError(f"Metadata mismatch: {key}")

    return result


def normalize_rows(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if (norms <= 1e-12).any():
        raise ValueError("Zero-length embeddings")
    return x / norms


def stratified_sample(data, limit, seed):
    """Sample VV and VH without changing their relative mix."""
    rng = np.random.default_rng(seed)
    indices = []

    for pol in ("VV", "VH"):
        options = np.flatnonzero(
            data["polarization"] == pol
        )
        if not len(options):
            raise ValueError(f"No {pol} embeddings")

        number = min(len(options), max(1, limit // 2))
        indices.extend(
            rng.choice(options, size=number, replace=False)
        )

    return np.asarray(sorted(indices), dtype=np.int64)


def feature_diagnostics(train, val, output_dir):
    """
    Calculate raw feature variance and normalized
    directional variance, rather than relying
    on cosine distances alone.
    """
    rows = []

    for split, data in (("train", train), ("val", val)):
        for pol in ("VV", "VH"):
            x = data["embedding"][
                data["polarization"] == pol
            ].astype(np.float64)

            if len(x) < 2:
                raise ValueError(f"Insufficient {split} {pol}")

            normalized = normalize_rows(x)
            variance = x.var(axis=0, ddof=1)
            direction_variance = normalized.var(
                axis=0, ddof=1
            )

            for dim in range(x.shape[1]):
                rows.append({
                    "split": split,
                    "polarization": pol,
                    "feature": dim,
                    "raw_variance": float(variance[dim]),
                    "direction_variance": float(
                        direction_variance[dim]
                    ),
                })

    frame = pd.DataFrame(rows)
    frame.to_csv(
        output_dir / "feature_variance.csv",
        index=False,
    )

    summary = {}
    for (split, pol), group in frame.groupby(
        ["split", "polarization"]
    ):
        summary[f"{split}_{pol}"] = {
            "median_raw_variance": float(
                group["raw_variance"].median()
            ),
            "mean_raw_variance": float(
                group["raw_variance"].mean()
            ),
            "near_zero_variance_dimensions": int(
                (group["raw_variance"] < 1e-10).sum()
            ),
            "mean_direction_variance": float(
                group["direction_variance"].mean()
            ),
        }

    return summary


def pca_diagnostics(train, val, output_dir, seed):
    """
    Fit PCA only on sampled TRAIN embeddings.
    Transform validation without refitting PCA.
    """
    train_idx = stratified_sample(
        train, limit=3000, seed=seed
    )

    train_x = normalize_rows(
        train["embedding"][train_idx]
    )
    val_x = normalize_rows(val["embedding"])

    pca = PCA(
        n_components=10,
        svd_solver="randomized",
        random_state=seed,
    )
    train_projected = pca.fit_transform(train_x)
    val_projected = pca.transform(val_x)

    rows = []

    for split, data, indices, projected in (
        ("train", train, train_idx, train_projected),
        (
            "val", val,
            np.arange(len(val["embedding"])),
            val_projected,
        ),
    ):
        for j, index in enumerate(indices):
            rows.append({
                "split": split,
                "polarization": str(
                    data["polarization"][index]
                ),
                "patch_id": str(data["patch_id"][index]),
                "acquisition_date": str(
                    data["acquisition_date"][index]
                ),
                "pc1": float(projected[j, 0]),
                "pc2": float(projected[j, 1]),
            })

    coordinates = pd.DataFrame(rows)
    coordinates.to_csv(
        output_dir / "pca_coordinates.csv",
        index=False,
    )

    for group_column, filename in (
        ("polarization", "pca_by_polarization.png"),
        ("split", "pca_by_split.png"),
    ):
        fig, ax = plt.subplots(figsize=(9, 7))

        for name, group in coordinates.groupby(
            group_column
        ):
            ax.scatter(
                group["pc1"],
                group["pc2"],
                s=8,
                alpha=0.45,
                label=name,
            )

        ax.set_xlabel(
            f"PC1 ({pca.explained_variance_ratio_[0]:.1%})"
        )
        ax.set_ylabel(
            f"PC2 ({pca.explained_variance_ratio_[1]:.1%})"
        )
        ax.set_title(
            f"SARMAE embeddings by {group_column}"
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            output_dir / filename,
            dpi=160,
        )
        plt.close(fig)

    ratios = pca.explained_variance_ratio_

    return {
        "pc1_explained_variance": float(ratios[0]),
        "pc2_explained_variance": float(ratios[1]),
        "first_10_explained_variance": float(
            ratios.sum()
        ),
    }


def nearest_neighbor_diagnostics(
    train, val, output_dir, seed
):
    """
    Compare:
      1. Train leave-self-out nearest-neighbor distances
         using sampled training observations.
      2. Validation-to-training nearest-neighbor distances.

    Train and validation are queried separately
    for VV and VH.
    """
    rng = np.random.default_rng(seed)
    output_rows = []
    summary = {}

    for pol in ("VV", "VH"):
        train_x = normalize_rows(
            train["embedding"][
                train["polarization"] == pol
            ]
        )
        val_x = normalize_rows(
            val["embedding"][
                val["polarization"] == pol
            ]
        )

        if len(train_x) < 3:
            raise ValueError(
                f"Not enough train {pol} embeddings"
            )

        nn = NearestNeighbors(
            n_neighbors=2,
            metric="cosine",
            algorithm="brute",
            n_jobs=-1,
        )
        nn.fit(train_x)

        sample_indices = rng.choice(
            len(train_x),
            size=min(1500, len(train_x)),
            replace=False,
        )

        train_distances, train_indices = nn.kneighbors(
            train_x[sample_indices]
        )

        # Explicitly exclude each observation itself.
        train_novelty = []
        for i, original_idx in enumerate(sample_indices):
            candidates = [
                float(d)
                for d, neighbor_idx in zip(
                    train_distances[i],
                    train_indices[i],
                )
                if neighbor_idx != original_idx
            ]
            if not candidates:
                raise RuntimeError(
                    "No non-self training neighbor found"
                )
            train_novelty.append(min(candidates))

        val_distances, _ = nn.kneighbors(
            val_x, n_neighbors=1
        )

        train_novelty = np.asarray(train_novelty)
        val_novelty = val_distances[:, 0]

        for label, distances in (
            ("train_nonself", train_novelty),
            ("validation", val_novelty),
        ):
            summary[f"{pol}_{label}"] = {
                "n": int(len(distances)),
                "mean": float(distances.mean()),
                "median": float(np.median(distances)),
                "p95": float(
                    np.quantile(distances, 0.95)
                ),
            }

            for d in distances:
                output_rows.append({
                    "polarization": pol,
                    "group": label,
                    "distance": float(d),
                })

    frame = pd.DataFrame(output_rows)

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 4), constrained_layout=True
    )

    for ax, pol in zip(axes, ("VV", "VH")):
        subset = frame[frame["polarization"] == pol]

        for name, group in subset.groupby("group"):
            ax.hist(
                group["distance"],
                bins=45,
                density=True,
                alpha=0.5,
                label=name,
            )

        ax.set_title(pol)
        ax.set_xlabel("Cosine nearest-neighbor distance")
        ax.set_ylabel("Density")
        ax.legend()

    fig.savefig(
        output_dir / "nearest_neighbor_distances.png",
        dpi=160,
    )
    plt.close(fig)

    return summary


def geographic_diagnostics(run_id, output_dir):
    path = (
        ROOT / "outputs/evaluations/sarmae"
        / run_id / "val/per_patch.csv"
    )
    if not path.exists():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path)

    def parse_patch(value):
        match = PATCH_PATTERN.fullmatch(str(value))
        if not match:
            raise ValueError(f"Unrecognized patch: {value}")
        return int(match.group(1)), int(match.group(2))

    coords = frame["patch_id"].map(parse_patch)
    frame["row"] = coords.map(lambda x: x[0])
    frame["col"] = coords.map(lambda x: x[1])

    frame = frame.sort_values(
        "mean_novelty", ascending=False
    )

    frame.to_csv(
        output_dir / "per_patch_diagnostics.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(8, 7))
    scatter = ax.scatter(
        frame["col"],
        frame["row"],
        c=frame["mean_novelty"],
        s=100,
        cmap="magma",
        edgecolors="black",
        linewidths=0.4,
    )

    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Patch column (reference-grid pixels)")
    ax.set_ylabel("Patch row (reference-grid pixels)")
    ax.set_title("SARMAE validation patch novelty")
    fig.colorbar(
        scatter, ax=ax,
        label="Mean cosine novelty",
    )

    fig.tight_layout()
    fig.savefig(
        output_dir / "geographic_novelty.png",
        dpi=160,
    )
    plt.close(fig)

    columns = (
        frame.groupby("col")
        .agg(
            patch_count=("patch_id", "size"),
            mean_novelty=("mean_novelty", "mean"),
        )
        .reset_index()
    )

    return {
        "validation_patch_count": int(len(frame)),
        "unique_validation_columns": (
            sorted(frame["col"].unique().tolist())
        ),
        "highest_novelty_patch": str(
            frame.iloc[0]["patch_id"]
        ),
        "top_10_columns": (
            frame.head(10)["col"].tolist()
        ),
        "column_summary": columns.to_dict(
            orient="records"
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-id", default=DEFAULT_RUN_ID
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    embedding_dir = (
        ROOT / "outputs/embeddings/sarmae"
        / args.run_id
    )
    output_dir = (
        ROOT / "outputs/diagnostics/sarmae"
        / args.run_id
    )
    output_dir.mkdir(
        parents=True, exist_ok=True
    )

    train = load_split(embedding_dir / "train.npz")
    val = load_split(embedding_dir / "val.npz")

    for key in ("source_revision", "input_mode"):
        if str(train[key]) != str(val[key]):
            raise RuntimeError(
                f"Train/validation mismatch: {key}"
            )

    if set(train["patch_id"]) & set(val["patch_id"]):
        raise RuntimeError(
            "Identical patch IDs found in both splits"
        )

    report = {
        "run_id": args.run_id,
        "checkpoint_revision": str(
            train["source_revision"]
        ),
        "input_mode": str(train["input_mode"]),
        "train_embeddings": int(
            len(train["embedding"])
        ),
        "validation_embeddings": int(
            len(val["embedding"])
        ),
        "feature_statistics": feature_diagnostics(
            train, val, output_dir
        ),
        "pca": pca_diagnostics(
            train, val, output_dir, args.seed
        ),
        "nearest_neighbors": nearest_neighbor_diagnostics(
            train, val, output_dir, args.seed
        ),
        "geographic_distribution": geographic_diagnostics(
            args.run_id, output_dir
        ),
    }

    path = output_dir / "diagnostics.json"
    path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nSARchaeology AI — Phase 2.13.1")
    print("Training embeddings:",
          report["train_embeddings"])
    print("Validation embeddings:",
          report["validation_embeddings"])
    print("PCA:", report["pca"])
    print(
        "Validation patch columns:",
        report["geographic_distribution"][
            "unique_validation_columns"
        ],
    )
    print("Reports:", output_dir)
    print("Diagnostics completed.")


if __name__ == "__main__":
    main()
