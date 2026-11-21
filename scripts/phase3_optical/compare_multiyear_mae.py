"""Phase 3.3.3: compare 75% and 60% optical ViT-MAE checkpoints.

Run from project root, e.g.:
  uv run python scripts/phase3_optical/compare_multiyear_mae.py `
    --a-checkpoint checkpoints/optical_mae_multiyear/<M75_RUN>/best.pt `
    --b-checkpoint checkpoints/optical_mae_multiyear/<M60_RUN>/best.pt `
    --a-config outputs/training/optical_mae_multiyear/<M75_RUN>/config.yaml `
    --b-config outputs/training/optical_mae_multiyear/<M60_RUN>/config.yaml

Design notes:
- Never updates model weights, training outputs, or existing checkpoints.
- Both models receive the same normalized 2024 validation images, same sample
  order, and the same random masking ranks. The visible-token counts differ.
- 'own_mask' scores each model on its own masked patches, and is NOT directly
  comparable as a causal effect of mask ratio.
- 'shared_mask' scores on positions masked by BOTH models. This aligns scored
  pixels, but DOES NOT equalize the amount of visible information.
- Reflectance metrics use raw TIFF reference pixels (not clipped normalized
  targets). Predicted reflectance is inverse-transformed from normalized output.
- Embedding PCA is exploratory, not archaeological validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sarchaeology.data.optical_multiyear import BANDS, MultiYearOpticalDataset
from sarchaeology.models.optical_mae import OpticalViTMAE


def path_in_root(raw):
    path = Path(raw).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a-checkpoint", type=str, required=True, help="75%% best.pt")
    p.add_argument("--b-checkpoint", type=str, required=True, help="60%% best.pt")
    p.add_argument("--a-config", type=str, required=True, help="Matching 75%% config.yaml")
    p.add_argument("--b-config", type=str, required=True, help="Matching 60%% config.yaml")
    p.add_argument("--output-dir", default="outputs/evaluation/optical_mae_multiyear_comparison")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0, help="0 avoids Windows multiprocessing overhead")
    p.add_argument("--seed", type=int, default=20261121)
    p.add_argument("--preview-count", type=int, default=8)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def load_experiment(label, checkpoint_name, config_name, device):
    checkpoint_path = path_in_root(checkpoint_name)
    config_path = path_in_root(config_name)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)


    saved_hash = checkpoint.get("config_sha256")
    current_hash = sha256(config_path)
    if saved_hash != current_hash:
        print(
            f"WARNING [{label}]: Configuration file SHA256 differs "
            "from the checkpoint's recorded fingerprint."
        )
        print(f"  Checkpoint SHA256: {saved_hash}")
        print(f"  Current SHA256:    {current_hash}")
        print(
            "  Continuing with semantic model configuration "
            "validation."
        )

    if not isinstance(config.get("model"), dict):
        raise RuntimeError(
            f"{label}: Invalid model configuration."
        )

    expected_model = {
        "image_size": 128,
        "patch_size": 8,
        "in_channels": 6,
        "encoder_dim": 192,
        "encoder_depth": 8,
        "encoder_heads": 3,
        "decoder_dim": 128,
        "decoder_depth": 4,
        "decoder_heads": 4,
    }

    for key, expected in expected_model.items():
        actual = config["model"].get(key)
        if actual != expected:
            raise RuntimeError(
                f"{label}: {key} mismatch. "
                f"Expected {expected}, got {actual}"
            )
    model_cfg = config["model"]
    ratio = float(model_cfg["mask_ratio"])
    expected = 0.75 if label == "A" else 0.60
    if not math.isclose(ratio, expected, abs_tol=1e-7):
        raise ValueError(f"{label}: expected ratio {expected}, found {ratio}")

    stats_path = path_in_root(config["data"]["statistics_path"])
    if checkpoint.get("normalization_sha256") != sha256(stats_path):
        raise RuntimeError(f"{label}: original normalization statistics SHA256 mismatch")
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    if tuple(stats["bands"]) != tuple(BANDS) or stats["fit_split"] != "train":
        raise RuntimeError(f"{label}: invalid statistics band order or fitting split")
    if checkpoint.get("run_id", "").find(f"_m{int(ratio*100)}_") < 0:
        raise RuntimeError(f"{label}: run ID is inconsistent with mask ratio")

    model = OpticalViTMAE(**model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return {"label":label, "ratio":ratio, "model":model, "config":config,
            "stats":stats, "stats_hash":sha256(stats_path), "checkpoint":checkpoint,
            "checkpoint_path":checkpoint_path, "config_path":config_path}


def initial_totals():
    return {"own_mask":new_totals(), "shared_mask":new_totals()}


def new_totals():
    return {"n":0, "abs_norm":0., "sq_norm":0., "huber_norm":0.,
            "abs_ref":0., "sq_ref":0., "channel_abs_ref":np.zeros(6, dtype=np.float64),
            "channel_n":np.zeros(6, dtype=np.int64)}


def add_metrics(totals, model, pred, norm_image, raw_image, valid, scored_patches, median, scale, delta):
    # patchify uses per-pixel, then channel order: (..., spatial, channel)
    target_norm = model.patchify(norm_image.float())
    target_ref = model.patchify(raw_image.float())
    valid_patches = model.patchify(valid.expand_as(norm_image).float()) > 0.5
    mask = scored_patches.unsqueeze(-1) & valid_patches
    n = int(mask.sum().item())
    if n == 0:
        return

    predicted = pred.float()
    abs_norm = (predicted - target_norm).abs()
    diff_norm = (predicted - target_norm)
    huber = torch.where(abs_norm <= delta, .5 * diff_norm.square(), delta * (abs_norm - .5 * delta))
    # Inverse transform is applied in image domain to guarantee channel order.
    pred_img = model.unpatchify(predicted)
    predicted_ref = model.patchify(pred_img * scale + median)
    diff_ref = predicted_ref - target_ref
    abs_ref = diff_ref.abs()

    totals["n"] += n
    totals["abs_norm"] += float(abs_norm.masked_select(mask).sum().item())
    totals["sq_norm"] += float(diff_norm.square().masked_select(mask).sum().item())
    totals["huber_norm"] += float(huber.masked_select(mask).sum().item())
    totals["abs_ref"] += float(abs_ref.masked_select(mask).sum().item())
    totals["sq_ref"] += float(diff_ref.square().masked_select(mask).sum().item())

    b, patches, _ = predicted.shape
    p = model.patch_size
    for c in range(6):
        channel_mask = mask.reshape(b, patches, p * p, 6)[..., c]
        channel_error = abs_ref.reshape(b, patches, p * p, 6)[..., c]
        totals["channel_abs_ref"][c] += float(channel_error.masked_select(channel_mask).sum().item())
        totals["channel_n"][c] += int(channel_mask.sum().item())


def summarize(totals):
    n = totals["n"]
    if n <= 0:
        raise RuntimeError("No pixels were scored")
    return {"scored_channel_pixels":n,
            "normalized_masked_huber":totals["huber_norm"] / n,
            "normalized_mae":totals["abs_norm"] / n,
            "normalized_rmse":math.sqrt(totals["sq_norm"] / n),
            "reflectance_mae":totals["abs_ref"] / n,
            "reflectance_rmse":math.sqrt(totals["sq_ref"] / n),
            "per_band_reflectance_mae":{
                str(BANDS[i]):float(totals["channel_abs_ref"][i] / max(totals["channel_n"][i],1))
                for i in range(6)}}


def color_rgb(image, valid, bounds):
    # Bands are B02,B03,B04,B08,B11,B12: true-color -> R,G,B = 2,1,0.
    rgb = image[[2,1,0]].transpose(1,2,0)
    lo,hi = bounds
    rgb = np.clip((rgb - lo)/(hi-lo + 1e-7), 0, 1)
    rgb[~valid] = 0
    return rgb


def save_previews(previews, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for i, (raw, valid, mask, a, b, title) in enumerate(previews):
        reference = raw[[2,1,0]].transpose(1,2,0)
        good = valid & np.isfinite(reference).all(axis=2)
        if not good.any():
            continue
        lo, hi = np.percentile(reference[good], [2,98])
        if hi <= lo:
            hi = lo + 0.05
        # Predictions are shown ONLY at identically scored, common masked positions.
        # Outside those positions retain observed raw values to make context comparable.
        a_show = raw.copy()
        b_show = raw.copy()
        a_show[:,mask] = a[:,mask]
        b_show[:,mask] = b[:,mask]
        fig, axes = plt.subplots(1,3,figsize=(12,4))
        for ax, image, label in zip(axes,[raw,a_show,b_show],
                                    ["Original Sentinel-2", "75% MAE (shared pixels)", "60% MAE (shared pixels)"]):
            ax.imshow(color_rgb(image, good, (lo,hi)))
            ax.set_title(label,fontsize=10)
            ax.axis("off")
        fig.suptitle(title + f" | common masked spatial fraction: {mask.mean():.1%}")
        fig.tight_layout()
        fig.savefig(outdir / f"comparison_{i:02d}.png",dpi=160)
        plt.close(fig)


def save_embedding_pca(embeddings, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for label, (emb, quarters) in embeddings.items():
        x = np.concatenate(emb,axis=0)
        if x.shape[0] < 3:
            continue
        x = x - x.mean(axis=0,keepdims=True)
        u,s,_ = np.linalg.svd(x,full_matrices=False)
        pc = u[:,:2]*s[:2]
        var = s**2 / max(np.sum(s**2),1e-12)
        quarter_arr = np.array(quarters)
        fig,ax = plt.subplots(figsize=(7,5))
        for q in (1,2,3,4):
            sel = quarter_arr == q
            ax.scatter(pc[sel,0],pc[sel,1],label=f"2024Q{q}",s=20,alpha=.8)
        ax.set_xlabel(f"PC1 ({var[0]:.1%} variance)")
        ax.set_ylabel(f"PC2 ({var[1]:.1%} variance)")
        ax.set_title(f"{label}: encoder embedding PCA (exploratory)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / f"embeddings_{label}.png",dpi=170)
        plt.close(fig)


def main():
    args = parse_args()
    if args.batch_size <= 0 or args.preview_count < 0:
        raise ValueError("batch size must be positive and preview count nonnegative")
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    exps = {
        "A":load_experiment("A",args.a_checkpoint,args.a_config,device),
        "B":load_experiment("B",args.b_checkpoint,args.b_config,device)
    }
    if exps["A"]["stats_hash"] != exps["B"]["stats_hash"]:
        raise RuntimeError("Experiments used different normalization statistics: direct comparison rejected")
    for key in ("train_manifest_sha256","val_manifest_sha256","test_manifest_sha256"):
        if exps["A"]["checkpoint"].get(key) != exps["B"]["checkpoint"].get(key):
            raise RuntimeError(f"Mismatch in {key}")

    cfg = exps["A"]["config"]
    min_fraction = float(cfg["multiyear"]["min_valid_fraction"])
    min_obs = int(cfg["multiyear"]["min_source_observations"])
    years = tuple(int(y) for y in cfg["multiyear"]["years"])
    for exp in exps.values():
        x = exp["config"]["multiyear"]
        if tuple(x["years"]) != years or float(x["min_valid_fraction"]) != min_fraction or int(x["min_source_observations"]) != min_obs:
            raise RuntimeError("Datasets or validity thresholds differ")

    normalized = MultiYearOpticalDataset(ROOT,years=years,split="val",
                                        min_valid_fraction=min_fraction,min_source_observations=min_obs,
                                        statistics=exps["A"]["stats"])
    raw_dataset = MultiYearOpticalDataset(ROOT,years=years,split="val",
                                         min_valid_fraction=min_fraction,min_source_observations=min_obs,
                                         statistics=None)
    assert [(x["patch_id"],x["period"]) for x in normalized.samples] == [
        (x["patch_id"],x["period"]) for x in raw_dataset.samples]
    if len(normalized) != 80:
        raise RuntimeError(f"Expected 80 validation observations; got {len(normalized)}")
    loader = DataLoader(normalized, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=(device.type=="cuda"))
    output_dir = path_in_root(args.output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    med = torch.as_tensor(exps["A"]["stats"]["median"],dtype=torch.float32,device=device)[None,:,None,None]
    scale = torch.as_tensor(exps["A"]["stats"]["scale"],dtype=torch.float32,device=device)[None,:,None,None]
    delta = float(cfg["training"]["huber_delta"])
    totals = {k:initial_totals() for k in exps}
    embeddings = {k:([],[]) for k in exps}
    previews = []
    processed = 0

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            b = batch["image"].shape[0]
            raw_samples = [raw_dataset[i] for i in range(processed, processed+b)]
            raw_image = torch.stack([x["image"] for x in raw_samples]).to(device)
            image = batch["image"].to(device,non_blocking=True)
            valid = batch["valid_mask"].to(device,non_blocking=True)
            if not torch.equal(valid,torch.stack([x["valid_mask"] for x in raw_samples]).to(device)):
                raise RuntimeError("Raw and normalized validity masks disagree")
            # Both models get identical permutation ranks in each corresponding batch.
            masks_seed = args.seed + batch_index
            predictions = {}
            for label,exp in exps.items():
                gen = torch.Generator(device="cpu")
                gen.manual_seed(masks_seed)
                with torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=(device.type=="cuda")):
                    out = exp["model"](image,valid,generator=gen)
                predictions[label] = out
                emb, quarter = embeddings[label]
                emb.append(out["embedding"].float().cpu().numpy())
                quarter.extend([int(q) for q in batch["quarter"].tolist()])

            common_mask = predictions["A"]["patch_mask"] & predictions["B"]["patch_mask"]
            for label,exp in exps.items():
                mdl = exp["model"]
                for scope, scored in (("own_mask",predictions[label]["patch_mask"]),
                                      ("shared_mask",common_mask)):
                    add_metrics(totals[label][scope],mdl,predictions[label]["predicted_patches"],
                                image,raw_image,valid,scored,med,scale,delta)

            if not args.no_plots and len(previews) < args.preview_count:
                common_img_mask = exps["A"]["model"].unpatchify(
                    common_mask.unsqueeze(-1).expand(-1,-1,exps["A"]["model"].patch_dim).float()
                )[:,0] > 0.5
                reconstructed = {}
                for label,exp in exps.items():
                    r = predictions[label]["reconstruction"].float()*scale+med
                    reconstructed[label] = r.cpu().numpy()
                for idx in range(b):
                    if len(previews) >= args.preview_count:
                        break
                    previews.append((raw_image[idx].cpu().numpy(),valid[idx,0].cpu().numpy(),
                                     common_img_mask[idx].cpu().numpy(),
                                     reconstructed["A"][idx],reconstructed["B"][idx],
                                     f"{batch['patch_id'][idx]} | 2024Q{batch['quarter'][idx].item()}"))
            processed += b
            print(f"Compared {processed}/{len(normalized)} validation observations",flush=True)

    result = {"validation_observations":processed,"device":str(device),"seed":args.seed,
              "note":"Shared-mask errors use identical scored pixels but not equal visible-token counts. Different ratio own-mask errors are not direct fair rankings.",
              "reflectance_note":"Reflectance targets use RAW 2024 TIFF values; predictions are inverse-normalized using train-fitted statistics.",
              "experiments":{}}
    for label,exp in exps.items():
        result["experiments"][label] = {
            "run_id":exp["checkpoint"]["run_id"],"checkpoint_epoch":exp["checkpoint"]["epoch"],
            "mask_ratio":exp["ratio"],"checkpoint_path":str(exp["checkpoint_path"]),
            "config_path":str(exp["config_path"]),
            "own_mask":summarize(totals[label]["own_mask"]),
            "shared_mask":summarize(totals[label]["shared_mask"])}
    (output_dir / "comparison_metrics.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    if not args.no_plots:
        save_previews(previews,output_dir)
        save_embedding_pca(embeddings,output_dir)
    for label in ("A","B"):
        e = result["experiments"][label]
        print(f"{label} mask={e['mask_ratio']:.0%} epoch={e['checkpoint_epoch']}: "
              f"shared reflectance MAE={e['shared_mask']['reflectance_mae']:.6f} "
              f"RMSE={e['shared_mask']['reflectance_rmse']:.6f}")
    print("Saved comparison:",output_dir)
    print("Read comparison_metrics.json and inspect comparison_*.png / embeddings_*.png")


if __name__ == "__main__":
    main()
