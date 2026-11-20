"""SARchaeology AI — Phase 2.13: frozen SARMAE feature novelty evaluation.

Requires completed (not smoke) train.npz and val.npz from Phase 2.12.
Fits a cosine nearest-neighbor reference on training embeddings only.
This is embedding novelty, NOT archaeological detection accuracy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ID = "sar_cae_20261118T192046Z_2dbe6bd2"


def load_embeddings(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run full Phase 2.12 extraction first")
    keys = (
        "embedding", "patch_id", "acquisition_date", "polarization",
        "valid_tokens", "source_revision", "input_mode",
    )
    with np.load(path, allow_pickle=False) as contents:
        missing = set(keys) - set(contents.files)
        if missing:
            raise ValueError(f"Missing fields in {path}: {sorted(missing)}")
        result = {key: contents[key].copy() for key in keys}
    features = result["embedding"]
    if features.ndim != 2 or features.shape[1] != 768 or len(features) == 0:
        raise ValueError(f"Unexpected embedding shape: {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError("Non-finite embedding values")
    for key in ("patch_id", "acquisition_date", "polarization", "valid_tokens"):
        if len(result[key]) != len(features):
            raise ValueError(f"Inconsistent {key} count")
    norms = np.linalg.norm(features.astype(np.float64), axis=1)
    if (norms < 1e-12).any():
        raise ValueError("Found near-zero embedding")
    result["embedding"] = (features / norms[:, None]).astype(np.float32)
    return result


def score_split(train: dict, val: dict) -> pd.DataFrame:
    rows = []
    for polarization in ("VV", "VH"):
        ref_ids = np.flatnonzero(train["polarization"] == polarization)
        query_ids = np.flatnonzero(val["polarization"] == polarization)
        if len(ref_ids) < 2 or len(query_ids) == 0:
            raise ValueError(f"Not enough {polarization} embeddings")
        nn = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute", n_jobs=-1)
        nn.fit(train["embedding"][ref_ids])
        distances, indices = nn.kneighbors(val["embedding"][query_ids])
        for j, val_idx in enumerate(query_ids):
            train_idx = ref_ids[int(indices[j, 0])]
            rows.append({
                "patch_id": str(val["patch_id"][val_idx]),
                "acquisition_date": str(val["acquisition_date"][val_idx]),
                "polarization": polarization,
                "novelty_cosine": float(max(0.0, distances[j, 0])),
                "valid_tokens": int(val["valid_tokens"][val_idx]),
                "nearest_train_patch_id": str(train["patch_id"][train_idx]),
                "nearest_train_acquisition_date": str(train["acquisition_date"][train_idx]),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    args = parser.parse_args()

    embedding_dir = ROOT / "outputs/embeddings/sarmae" / args.run_id
    output_dir = ROOT / "outputs/evaluations/sarmae" / args.run_id / "val"
    train = load_embeddings(embedding_dir / "train.npz")
    val = load_embeddings(embedding_dir / "val.npz")

    for key in ("source_revision", "input_mode"):
        if str(train[key].item()) != str(val[key].item()):
            raise RuntimeError(f"Train/validation {key} mismatch")
    input_mode = str(val["input_mode"].item())
    if input_mode != "normalized_proxy":
        raise RuntimeError(f"Unexpected input mode: {input_mode}")
    train_patches = set(map(str, train["patch_id"]))
    val_patches = set(map(str, val["patch_id"]))
    if train_patches & val_patches:
        raise RuntimeError("Train and validation share patch IDs")
    # This catches duplicated metadata within either polarization.
    for split_name, dataset in (("train", train), ("val", val)):
        identifiers = list(zip(
            dataset["patch_id"].tolist(),
            dataset["acquisition_date"].tolist(),
            dataset["polarization"].tolist(),
        ))
        if len(set(identifiers)) != len(identifiers):
            raise RuntimeError(f"Duplicate {split_name} (patch,date,polarization) rows")

    scores = score_split(train, val)
    observation_keys = ["patch_id", "acquisition_date"]
    observation = scores.pivot(
        index=observation_keys, columns="polarization", values="novelty_cosine"
    )
    if observation.isna().any().any():
        raise RuntimeError("An observation is missing VV or VH")
    observation = observation.reset_index().rename(
        columns={"VV": "vv_novelty", "VH": "vh_novelty"}
    )
    observation["mean_novelty"] = (
        observation["vv_novelty"] + observation["vh_novelty"]
    ) / 2.0
    patch = observation.groupby("patch_id", as_index=False).agg(
        observations=("mean_novelty", "size"),
        mean_novelty=("mean_novelty", "mean"),
        std_novelty=("mean_novelty", "std"),
        max_novelty=("mean_novelty", "max"),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output_dir / "per_polarization.csv", index=False)
    observation.to_csv(output_dir / "per_observation.csv", index=False)
    patch.to_csv(output_dir / "per_patch.csv", index=False)

    cae_path = (
        ROOT / "outputs/evaluations/sar_cae" / args.run_id / "val/per_observation.csv"
    )
    comparison = {
        "cae_available": cae_path.is_file(),
        "matching_observations": 0,
        "spearman_cae_rmse_vs_sarmae_novelty": None,
    }
    if cae_path.is_file():
        cae = pd.read_csv(cae_path, dtype={"patch_id": str, "acquisition_date": str})
        merged = observation.merge(
            cae[observation_keys + ["mae_normalized", "rmse_normalized", "huber_normalized"]],
            on=observation_keys,
            how="inner",
            validate="one_to_one",
        )
        if len(merged) != len(observation):
            raise RuntimeError("Incomplete CAE/SARMAE observation join")
        merged.to_csv(output_dir / "comparison_with_cae.csv", index=False)
        comparison["matching_observations"] = len(merged)
        if merged["mean_novelty"].nunique() > 1 and merged["rmse_normalized"].nunique() > 1:
            correlation = spearmanr(merged["mean_novelty"], merged["rmse_normalized"]).statistic
            if np.isfinite(correlation):
                comparison["spearman_cae_rmse_vs_sarmae_novelty"] = float(correlation)

    summary = {
        "phase": "2.13",
        "run_id": args.run_id,
        "model": "frozen pretrained SARMAE ViT-Base",
        "split": "val",
        "checkpoint_revision": str(val["source_revision"].item()),
        "input_mode": input_mode,
        "training_embeddings": int(len(train["embedding"])),
        "validation_embeddings": int(len(val["embedding"])),
        "validation_observations": int(len(observation)),
        "validation_patches": len(val_patches),
        "mean_novelty_cosine": float(observation["mean_novelty"].mean()),
        "std_novelty_cosine": float(observation["mean_novelty"].std()),
        "min_novelty_cosine": float(observation["mean_novelty"].min()),
        "max_novelty_cosine": float(observation["mean_novelty"].max()),
        "comparison": comparison,
        "interpretation": (
            "Cosine nearest-neighbor novelty against training features; "
            "not archaeological detection accuracy."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\nSARchaeology AI — Phase 2.13")
    print(json.dumps(summary, indent=2))
    print("\nTop ten highest-novelty validation patches:")
    print(patch.nlargest(10, "mean_novelty").to_string(index=False))
    print("\nSaved:", output_dir)


if __name__ == "__main__":
    main()
