"""SARchaeology AI Phase 3.4.1: ViT-MAE vs Swin-MIM evaluation.

Evaluate the exact same 2024 validation samples with equal 75% masks.
Reports raw-reflectance reconstruction errors, masked-region boundary/gradient
errors, per-quarter and per-band MAE, deterministic mask-leakage diagnostics,
RGB reconstructions, and exploratory embedding PCA.

No checkpoints or training data are modified.
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
from sarchaeology.models.optical_swin_mim import OpticalSwinMIM


def resolve_path(raw):
    p = Path(raw).expanduser()
    return (p if p.is_absolute() else ROOT / p).resolve()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vit-checkpoint", required=True)
    p.add_argument("--vit-config", required=True)
    p.add_argument("--swin-checkpoint", required=True)
    p.add_argument("--swin-config", required=True)
    p.add_argument("--output-dir", default="outputs/evaluation/optical_swin_vs_vit")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=20261121)
    p.add_argument("--preview-count", type=int, default=8)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--skip-leakage-test", action="store_true")
    return p.parse_args()


def read_run(name, kind, checkpoint_str, config_str, device):
    cp_path, cfg_path = resolve_path(checkpoint_str), resolve_path(config_str)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cp = torch.load(cp_path, map_location="cpu", weights_only=True)
    ratio = float(cfg["model"]["mask_ratio"])
    if abs(ratio - 0.75) > 1e-7:
        raise RuntimeError(f"{name}: expected 75% masking, got {ratio}")
    if f"_m75_" not in cp["run_id"]:
        raise RuntimeError(f"{name}: run ID is inconsistent with 75% masking")
    if cp.get("config_sha256") != digest(cfg_path):
        print(f"WARNING [{name}]: byte-level config hash differs from checkpoint; "
              "using validated config fields and strict state_dict loading")
    stats_path = resolve_path(cfg["data"]["statistics_path"])
    if digest(stats_path) != cp.get("normalization_sha256"):
        raise RuntimeError(f"{name}: normalization SHA256 mismatch")
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    if tuple(stats["bands"]) != tuple(BANDS) or stats["fit_split"] != "train":
        raise RuntimeError(f"{name}: invalid normalization band order or fit split")
    klass = OpticalViTMAE if kind == "vit" else OpticalSwinMIM
    model = klass(**cfg["model"]).to(device)
    model.load_state_dict(cp["model_state"], strict=True)
    model.eval()
    return {"name": name, "model": model, "kind": kind, "cp": cp,
            "cfg": cfg, "stats": stats, "stats_hash": digest(stats_path),
            "cp_path": str(cp_path), "cfg_path": str(cfg_path)}


def verify_pair(a, b):
    if a["stats_hash"] != b["stats_hash"]:
        raise RuntimeError("Model normalization statistics are different")
    for key in ("train_manifest_sha256", "val_manifest_sha256", "test_manifest_sha256"):
        if a["cp"].get(key) != b["cp"].get(key):
            raise RuntimeError(f"Dataset fingerprint mismatch: {key}")
    for field in ("years", "min_valid_fraction", "min_source_observations"):
        if a["cfg"]["multiyear"][field] != b["cfg"]["multiyear"][field]:
            raise RuntimeError(f"Dataset option mismatch: {field}")
    for field in ("huber_delta",):
        if a["cfg"]["training"][field] != b["cfg"]["training"][field]:
            raise RuntimeError(f"Training scoring option mismatch: {field}")


def create_acc():
    return {"count":0, "abs":0., "sq":0., "huber":0.,
            "band_abs":np.zeros(6, dtype=np.float64),
            "band_sq":np.zeros(6, dtype=np.float64),
            "band_n":np.zeros(6, dtype=np.int64),
            "edge_abs":0., "edge_sq":0., "edge_n":0}


def update_acc(acc, predicted, raw, valid, patch_mask, delta):
    # Work in actual reflectance units. Only valid masked pixels are scored.
    b, c, h, w = raw.shape
    g = int(math.isqrt(patch_mask.shape[1]))
    p = h // g
    assert g * g == patch_mask.shape[1] and p * g == h == w
    spatial = patch_mask.reshape(b, 1, g, g).repeat_interleave(p, 2).repeat_interleave(p, 3)
    spatial = spatial & valid.bool()
    selected = spatial.expand_as(raw)
    error = (predicted.float() - raw.float())
    count = int(selected.sum().item())
    if count == 0:
        return
    difference = error.abs()
    hubs = torch.where(difference <= delta, .5 * error.square(), delta * (difference - .5 * delta))
    acc["count"] += count
    acc["abs"] += difference.masked_select(selected).double().sum().item()
    acc["sq"] += error.square().masked_select(selected).double().sum().item()
    acc["huber"] += hubs.masked_select(selected).double().sum().item()
    for ch in range(c):
        m = spatial[:,0]
        v = error[:,ch].masked_select(m)
        acc["band_abs"][ch] += v.abs().double().sum().item()
        acc["band_sq"][ch] += v.square().double().sum().item()
        acc["band_n"][ch] += v.numel()
    # Gradient errors where BOTH adjacent pixels are valid and masked.
    # This assesses local contrast, not the geometric alignment of exact edges.
    horizontal = spatial[:,:,:,1:] & spatial[:,:,:,:-1]
    vertical = spatial[:,:,1:,:] & spatial[:,:,:-1,:]
    for axis, pairs in ((3, horizontal), (2, vertical)):
        grad_error = torch.diff(predicted.float(), dim=axis) - torch.diff(raw.float(), dim=axis)
        scored = pairs.expand_as(grad_error)
        v = grad_error.masked_select(scored)
        acc["edge_abs"] += v.abs().double().sum().item()
        acc["edge_sq"] += v.square().double().sum().item()
        acc["edge_n"] += v.numel()


def summarize(a):
    n, en = a["count"], a["edge_n"]
    if n == 0 or en == 0:
        raise RuntimeError("No valid pixels or masked-pixel pairs to score")
    return {"scored_channel_pixels":n, "reflectance_mae":a["abs"]/n,
            "reflectance_rmse":math.sqrt(a["sq"]/n),
            "reflectance_huber_delta_1":a["huber"]/n,
            "per_band_mae":{BANDS[i]:a["band_abs"][i]/max(1,a["band_n"][i]) for i in range(6)},
            "per_band_rmse":{BANDS[i]:math.sqrt(a["band_sq"][i]/max(1,a["band_n"][i])) for i in range(6)},
            "masked_gradient_mae":a["edge_abs"]/en,
            "masked_gradient_rmse":math.sqrt(a["edge_sq"]/en),
            "scored_gradient_channel_pairs":en}


def leakage_test(run, img, valid, seed, device):
    """Change only hidden input pixels while reusing deterministic mask ranks.

    Outputs at masked pixels must be invariant to hidden input values.
    The test is sensitive to architectures that allow pre-mask convolutions
    across masked/visible boundaries.
    """
    model = run["model"]
    def generate(x):
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                              enabled=(device.type == "cuda")):
            return model(x,valid,generator=g)
    ref = generate(img)
    patch_mask = ref["patch_mask"]
    # Mask spatial positions without relying on model-specific patchify.
    b, _, h, w = img.shape
    side = int(math.isqrt(patch_mask.shape[1]))
    pixel_mask = patch_mask.reshape(b,1,side,side).repeat_interleave(h//side,2).repeat_interleave(w//side,3)
    perturbed = torch.where(pixel_mask, img + 0.3, img)
    changed = generate(perturbed)
    if not torch.equal(patch_mask, changed["patch_mask"]):
        raise RuntimeError(f"{run['name']}: masking changed during leakage test")
    mask_patches = patch_mask.unsqueeze(-1).expand_as(ref["predicted_patches"])
    diff = (ref["predicted_patches"].float()-changed["predicted_patches"].float()).abs()
    maximum = float(diff.masked_select(mask_patches).max().item())
    return {"max_masked_prediction_change_normalized":maximum,
            "pass":bool(maximum < 1e-4), "tolerance":1e-4,
            "interpretation":"Single perturbation check; not a proof against all forms of leakage"}


def rgb(image, valid, lo, hi):
    x = image[[2,1,0]].transpose(1,2,0)
    x = np.clip((x-lo)/(hi-lo+1e-8),0,1)
    x[~valid] = 0
    return x


def plot_previews(previews, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for idx,(raw, valid, mask, predictions, name) in enumerate(previews):
        true_rgb = raw[[2,1,0]].transpose(1,2,0)
        good = valid & np.isfinite(true_rgb).all(axis=-1)
        if not good.any():
            continue
        lo,hi = np.percentile(true_rgb[good],[2,98])
        if hi <= lo: hi = lo + .05
        panels = [(raw,"Original Sentinel-2")]
        for label in ("vit","swin"):
            composite = raw.copy()
            composite[:,mask] = predictions[label][:,mask]
            panels.append((composite,f"{label.upper()} masked reconstruction"))
        fig,axes = plt.subplots(1,4,figsize=(16,4.3))
        for ax,(data,caption) in zip(axes[:3],panels):
            ax.imshow(rgb(data,good,lo,hi));ax.set_title(caption,fontsize=10);ax.axis("off")
        axes[3].imshow(mask,cmap="gray",vmin=0,vmax=1)
        axes[3].set_title("Masked pixels (white)",fontsize=10);axes[3].axis("off")
        fig.suptitle(name)
        fig.tight_layout()
        fig.savefig(output / f"comparison_{idx:02d}.png",dpi=160)
        plt.close(fig)


def plot_pca(store, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for name,observations in store.items():
        x=np.stack([r["embedding"] for r in observations]).astype(np.float64)
        x -= x.mean(axis=0,keepdims=True)
        u,s,_=np.linalg.svd(x,full_matrices=False)
        variance=s*s/max(np.sum(s*s),1e-12)
        xy=u[:,:2]*s[:2]
        fig,ax=plt.subplots(figsize=(7,5))
        for q in (1,2,3,4):
            which=np.array([r["quarter"] == q for r in observations])
            ax.scatter(xy[which,0],xy[which,1],label=f"2024Q{q}",alpha=.85,s=24)
        ax.set_xlabel(f"PC1 ({variance[0]:.1%})");ax.set_ylabel(f"PC2 ({variance[1]:.1%})")
        ax.set_title(f"{name.upper()} encoder embedding PCA")
        ax.legend();fig.tight_layout()
        fig.savefig(output / f"embeddings_{name}.png",dpi=170)
        plt.close(fig)


def embedding_diagnostics(store):
    """Mean pairwise cos similarity, within patch across quarters vs
    same quarter across different patches, excluding same observation."""
    result={}
    for key,obs in store.items():
        x=np.stack([r["embedding"] for r in obs]).astype(np.float64)
        x=x/(np.linalg.norm(x,axis=1,keepdims=True)+1e-12)
        sim=x@x.T
        ids=np.array([r["patch_id"] for r in obs])
        quarters=np.array([r["quarter"] for r in obs])
        pairs=np.triu(np.ones(sim.shape,dtype=bool),k=1)
        same_patch=pairs & (ids[:,None]==ids[None,:]) & (quarters[:,None]!=quarters[None,:])
        same_quarter=pairs & (ids[:,None]!=ids[None,:]) & (quarters[:,None]==quarters[None,:])
        result[key]={"same_patch_different_quarter_cosine":float(sim[same_patch].mean()),
                     "different_patch_same_quarter_cosine":float(sim[same_quarter].mean()),
                     "difference":float(sim[same_patch].mean()-sim[same_quarter].mean()),
                     "note":"Exploratory; mean cosine can reflect backbone representation scale/anisotropy"}
    return result


def main():
    args=parse_args()
    if args.batch_size < 1 or args.num_workers < 0 or args.preview_count < 0:
        raise ValueError("Invalid batch-size, num-workers or preview-count")
    device=torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    if device.type == "cuda":torch.set_float32_matmul_precision("high")
    runs={
        "vit":read_run("ViT-Tiny","vit",args.vit_checkpoint,args.vit_config,device),
        "swin":read_run("Swin-Tiny","swin",args.swin_checkpoint,args.swin_config,device),
    }
    verify_pair(runs["vit"],runs["swin"])
    cfg=runs["vit"]["cfg"]
    multi=cfg["multiyear"]
    options=dict(years=tuple(multi["years"]),split="val",
                 min_valid_fraction=float(multi["min_valid_fraction"]),
                 min_source_observations=int(multi["min_source_observations"]))
    dataset=MultiYearOpticalDataset(ROOT,statistics=runs["vit"]["stats"],**options)
    raw_dataset=MultiYearOpticalDataset(ROOT,statistics=None,**options)
    if [(x["patch_id"],x["period"]) for x in dataset.samples] != [(x["patch_id"],x["period"]) for x in raw_dataset.samples]:
        raise RuntimeError("Raw and normalized datasets differ in sample order")
    if len(dataset)!=80:raise RuntimeError(f"Expected 80 validation observations, got {len(dataset)}")
    loader=DataLoader(dataset,batch_size=args.batch_size,shuffle=False,
                      num_workers=args.num_workers,pin_memory=(device.type=="cuda"))
    med=torch.as_tensor(runs["vit"]["stats"]["median"],device=device,dtype=torch.float32)[None,:,None,None]
    scale=torch.as_tensor(runs["vit"]["stats"]["scale"],device=device,dtype=torch.float32)[None,:,None,None]
    norm_delta=float(cfg["training"]["huber_delta"])
    out=resolve_path(args.output_dir)
    out.mkdir(parents=True,exist_ok=True)
    metrics={key:create_acc() for key in runs}
    quarter_metrics={key:{q:create_acc() for q in (1,2,3,4)} for key in runs}
    embeddings={key:[] for key in runs}
    previews=[]
    leakage={}
    processed=0
    with torch.inference_mode():
        for batch_id,batch in enumerate(loader):
            b=batch["image"].shape[0]
            image=batch["image"].to(device,non_blocking=True)
            valid=batch["valid_mask"].to(device,non_blocking=True)
            raws=[raw_dataset[i] for i in range(processed,processed+b)]
            if not torch.equal(valid,torch.stack([r["valid_mask"] for r in raws]).to(device)):
                raise RuntimeError("Raw and normalized validity masks do not match")
            raw=torch.stack([r["image"] for r in raws]).to(device)
            predictions={}
            for name,run in runs.items():
                g=torch.Generator(device="cpu")
                g.manual_seed(args.seed+batch_id)
                with torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=device.type=="cuda"):
                    predictions[name]=run["model"](image,valid,generator=g)
                for j in range(b):
                    embeddings[name].append({"embedding":predictions[name]["embedding"][j].float().cpu().numpy(),
                                             "patch_id":str(batch["patch_id"][j]),
                                             "quarter":int(batch["quarter"][j].item())})
            patch_mask=predictions["vit"]["patch_mask"]
            if not torch.equal(patch_mask,predictions["swin"]["patch_mask"]):
                raise RuntimeError("ViT and Swin were not given IDENTICAL masks")
            spatial=patch_mask.reshape(b,1,16,16).repeat_interleave(8,2).repeat_interleave(8,3)
            for name,run in runs.items():
                pred_ref=predictions[name]["reconstruction"].float()*scale+med
                update_acc(metrics[name],pred_ref,raw,valid,patch_mask,norm_delta)
                for q in (1,2,3,4):
                    indexes=(batch["quarter"]==q).nonzero(as_tuple=True)[0]
                    if len(indexes):
                        update_acc(quarter_metrics[name][q],pred_ref[indexes],raw[indexes],valid[indexes],patch_mask[indexes],norm_delta)
                if not args.no_plots and len(previews)<args.preview_count:
                    pass
            if batch_id==0 and not args.skip_leakage_test:
                for name,run in runs.items():
                    # Use only first sample to minimize the extra full forward pass.
                    leakage[name]=leakage_test(run,image[:1],valid[:1],args.seed,device)
                    print(f"{name} leakage check: {leakage[name]}")
                    if not leakage[name]["pass"]:
                        raise RuntimeError(f"{name}: possible leakage from hidden input pixels; STOP evaluation")
            for j in range(b):
                if args.no_plots or len(previews)>=args.preview_count:break
                previews.append((raw[j].cpu().numpy(),valid[j,0].cpu().numpy(),spatial[j,0].cpu().numpy(),
                                 {name:(predictions[name]["reconstruction"][j].float()*scale[0]+med[0]).cpu().numpy() for name in runs},
                                 f"{batch['patch_id'][j]} | 2024Q{int(batch['quarter'][j].item())}"))
            processed+=b
            print(f"Evaluated {processed}/80 observations",flush=True)
    result={"validation_observations":processed,"device":str(device),"seed":args.seed,
            "mask_ratio":.75,"masking_protocol":"same RNG seed, same patch mask; verified exact patch_mask equality",
            "masking_caveat":"ViT encodes visible tokens only; Swin processes a complete grid after mask-token replacement. Equal masks do not equalize computational architecture or training budget.",
            "evaluation_note":"Errors are computed on masked+valid pixels using raw TIFF reflectance as target, with predictions inverse-normalized from train-only statistics.",
            "gradient_note":"Masked-gradient error scores adjacent valid masked pixel pairs; it is a local-contrast error, not a direct archaeological line-detection metric.",
            "models":{}, "embedding_diagnostics":embedding_diagnostics(embeddings),"leakage_tests":leakage}
    for name,run in runs.items():
        result["models"][name]={"run_id":run["cp"]["run_id"],"checkpoint_epoch":run["cp"]["epoch"],
                                "checkpoint":run["cp_path"],"config":run["cfg_path"],
                                "overall":summarize(metrics[name]),
                                "by_quarter":{str(q):summarize(quarter_metrics[name][q]) for q in (1,2,3,4)}}
    (out/"comparison_metrics.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    if not args.no_plots:
        plot_previews(previews,out)
        plot_pca(embeddings,out)
    for name in runs:
        m=result["models"][name]["overall"]
        print(f"{name.upper()}: reflectance MAE={m['reflectance_mae']:.6f}, "
              f"RMSE={m['reflectance_rmse']:.6f}, gradient MAE={m['masked_gradient_mae']:.6f}")
    print("Saved:",out)

if __name__=="__main__":
    main()
