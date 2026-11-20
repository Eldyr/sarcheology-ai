"""Phase 2.16 exploratory CAE/Swin rank comparison (NOT same reconstruction task).

Run after evaluate_sar_swin_mae.py. Scores have distinct objectives, so
compare rank association only; do not compare metric magnitudes or claim accuracy.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SWIN = "sar_swin_mae_20261120T165639Z_1cd7f887"
DEFAULT_CAE = "sar_cae_20261118T192046Z_2dbe6bd2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--swin-run", default=DEFAULT_SWIN)
    ap.add_argument("--cae-run", default=DEFAULT_CAE)
    args = ap.parse_args()
    out = ROOT / "outputs/evaluations/sar_swin_mae" / args.swin_run / "val"
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    if summary["partial"]:
        raise RuntimeError("Run full Swin evaluation first (not --max-batches)")
    swin = pd.read_csv(out / "per_observation.csv")
    cae_path = ROOT / "outputs/evaluations/sar_cae" / args.cae_run / "val/per_observation.csv"
    cae = pd.read_csv(cae_path)
    if "rmse_normalized" not in cae.columns:
        raise ValueError(f"Expected 'rmse_normalized' in {cae_path}: {cae.columns.tolist()}")
    keys = ["patch_id", "acquisition_date"]
    if any(k not in swin.columns or k not in cae.columns for k in keys):
        raise ValueError("Cannot align datasets by patch_id and acquisition_date")
    if swin.duplicated(keys).any() or cae.duplicated(keys).any():
        raise ValueError("Duplicate observation keys")
    matched = swin[keys + ["masked_rmse"]].merge(
        cae[keys + ["rmse_normalized"]], on=keys, how="inner", validate="one_to_one"
    )
    if len(matched) != len(swin):
        raise ValueError(f"Only {len(matched)} of {len(swin)} Swin observations match CAE")
    patch = matched.groupby("patch_id", as_index=False).agg(
        swin_masked_rmse=("masked_rmse", "mean"),
        cae_full_rmse=("rmse_normalized", "mean"),
        observations=("masked_rmse", "size"),
    )
    obs_corr = spearmanr(matched["masked_rmse"], matched["rmse_normalized"])
    patch_corr = spearmanr(patch["swin_masked_rmse"], patch["cae_full_rmse"])
    matched.to_csv(out / "swin_cae_observation_comparison.csv", index=False)
    patch.to_csv(out / "swin_cae_patch_comparison.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(patch["cae_full_rmse"], patch["swin_masked_rmse"])
    ax.set_xlabel("CAE full-input reconstruction RMSE")
    ax.set_ylabel("Swin masked-input, masked-region RMSE")
    ax.set_title("Different tasks: rank-association only")
    fig.tight_layout()
    fig.savefig(out / "swin_cae_patch_association.png", dpi=170)
    plt.close(fig)
    report = {
        "matching_observations": len(matched), "matching_patches": len(patch),
        "observation_spearman": float(obs_corr.statistic),
        "patch_spearman": float(patch_corr.statistic),
        "interpretation": "Rank association only. Different inputs and target regions; no accuracy comparison. Spatially dependent patches; nominal p-values omitted.",
    }
    (out / "comparison_with_cae.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("Saved:", out)


if __name__ == "__main__":
    main()
