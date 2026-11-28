"""Train development-only fusion heads on cached frozen sensor features."""
import platform
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from sarchaeology.datasets.multimodal_dataset import MultiModalArchaeologyDataset, QuarterPairSampler
from .io import device_for, digest, save_json
from .model import LocalFusion, MODALITIES, inputs


def compact_dataset(dataset):
    """Keep native learned maps in half precision; pool raw target tensors once."""
    compact = []
    for b in DataLoader(dataset,batch_size=8):
        mask,quality,targets = inputs(b)
        for i in range(len(mask)):
            item = {k:b[k][i].clone() for k in ("sar_features","stage0","stage1","optical_embedding",
                                                     "row","col","quarter","patch_id","period")
                    if isinstance(b[k],torch.Tensor)}
            item.update(patch_id=b["patch_id"][i],period=b["period"][i],mask=mask[i].clone(),
                        quality=quality[i].clone(),targets=tuple(t[i].clone() for t in targets))
            compact.append(item)
    return compact


def to_device(batch,device):
    return {k:(v.to(device) if torch.is_tensor(v) else tuple(x.to(device) for x in v)
               if k=="targets" else v) for k,v in batch.items()}


def pooled(x,mask):
    return (x*mask).sum((-1,-2))/mask.sum((-1,-2)).clamp_min(1)


def geo_negatives(row,col,buffer=128):
    edges = torch.stack([(row[:,None]-row).abs(),(col[:,None]-col).abs()]).float()-128
    return edges.clamp_min(0).square().sum(0) >= buffer**2


def alignment_loss(tokens,mask,batch):
    negative = geo_negatives(batch["row"],batch["col"])
    losses = []
    for i,j in ((0,1),(0,2),(1,2)):
        common = (mask[:,i:i+1] & mask[:,j:j+1])
        usable = common.flatten(1).any(1)
        a,b = [F.normalize(pooled(tokens[:,k].float(),common),dim=1) for k in (i,j)]
        logits = (a@b.T)/0.15
        candidates = negative & usable[:,None] & usable[None,:]
        selected = candidates.any(1) & usable
        if selected.any():
            eye = torch.eye(len(a),device=a.device,dtype=torch.bool)
            logits = logits.masked_fill(~(candidates | eye),-1e4)
            losses.append(F.cross_entropy(logits[selected],torch.where(selected)[0]))
    return torch.stack(losses).mean() if losses else tokens.sum()*0


def temporal_loss(embeddings,batch,supported):
    same = (batch["row"][:,None]==batch["row"]) & (batch["col"][:,None]==batch["col"])
    adjacent = (batch["quarter"][:,None]-batch["quarter"]) == 1
    pairs = same & adjacent & supported[:,None] & supported[None,:]
    if pairs.any():
        i,j = torch.where(pairs)
        return (1-F.cosine_similarity(embeddings[i],embeddings[j])).mean()
    return embeddings.sum()*0


def errors(out):
    return torch.cat([(p.float()-t).square().mean(1,keepdim=True)
                      for p,t in zip(out["predictions"],out["targets"])],1)


def rank_metrics(emb):
    vals = torch.linalg.svdvals(emb.double()-emb.double().mean(0))
    energy = vals.square()
    if energy.sum() < 1e-12:
        return dict(effective_rank=0., feature_std=0.)
    prob = energy/energy.sum()
    return dict(effective_rank=float((-(prob*prob.clamp_min(1e-15).log()).sum()).exp()),
                feature_std=float(emb.std(0,unbiased=False).mean()))


@torch.no_grad()
def validate(model,loader,device,common_support=False):
    model.eval()
    details,emb = [],[]
    for raw in loader:
        batch = to_device(raw,device)
        for hidden in (None,0,1,2):
            drop = torch.zeros_like(batch["mask"])
            if hidden is not None:
                drop[:,hidden] = True
            out = model(batch,drop)
            usable = out["mask"] & out["support"]
            if common_support:
                usable = usable & out["mask"].all(1,keepdim=True)
            counts = usable.sum((-1,-2))
            sse = (errors(out)*usable).sum((-1,-2))
            if hidden is None:
                pooled_features = pooled(out["fused"],out["support"])
                emb.append(pooled_features[out["support"].flatten(1).any(1)].cpu())
            for i,(s,n) in enumerate(zip(sse.tolist(),counts.tolist())):
                details.append(dict(patch_id=raw["patch_id"][i],period=raw["period"][i],
                                    row=int(raw["row"][i]),col=int(raw["col"][i]),
                                    hidden="none" if hidden is None else MODALITIES[hidden],
                                    **{f"sse_{k}":v for k,v in zip(MODALITIES,s)},
                                    **{f"n_{k}":v for k,v in zip(MODALITIES,n)}))
    frame = pd.DataFrame(details)
    nominal = frame[frame.hidden=="none"]
    counts = [nominal[f"n_{k}"].sum() for k in MODALITIES]
    if not any(counts):
        raise ValueError("Validation has no supported cells for the requested comparison")
    metric = np.mean([nominal[f"sse_{k}"].sum()/n for k,n in zip(MODALITIES,counts) if n])
    return float(metric),rank_metrics(torch.cat(emb)),frame


def train_variant(variant,cfg,train,valid,rows,out,qa_hash):
    out.mkdir(exist_ok=False)
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = device_for(cfg["device"])
    model = LocalFusion(variant,cfg["width"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"])
    sampler = QuarterPairSampler(rows,cfg["batch_size"],seed)
    loader = DataLoader(train,batch_sampler=sampler)
    val_loader = DataLoader(valid,batch_size=cfg["batch_size"])
    history, best = [],float("inf")
    for epoch in range(cfg["epochs"]):
        model.train()
        sampler.epoch = epoch
        losses = []
        for raw in loader:
            batch = to_device(raw,device)
            mask = batch["mask"]
            drops = torch.randint(0,4,(len(mask),1,1,1),device=device)
            hidden = drops==torch.arange(3,device=device)[None,:,None,None]
            if variant in MODALITIES:
                hidden = torch.zeros_like(hidden)
            out_b = model(batch,hidden)
            eligible = mask & out_b["support"]
            weights = eligible.float()*(1+3*hidden.float())
            if not eligible.any():
                continue
            prediction = (errors(out_b)*weights).sum()/weights.sum()
            embedding = pooled(out_b["fused"],out_b["support"])
            temporal = temporal_loss(embedding,batch,out_b["support"].flatten(1).any(1))
            align = alignment_loss(out_b["tokens"],mask,batch) if variant not in MODALITIES else prediction*0
            loss = prediction + cfg.get("alignment_weight",0.05)*align + cfg.get("temporal_weight",0.03)*temporal
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
            optimizer.step()
            losses.append([float(v.detach()) for v in (loss,prediction,align,temporal)])
        if not losses:
            raise ValueError("No supported training cells")
        metric,rank,details = validate(model,val_loader,device)
        means = np.mean(losses,axis=0)
        history.append(dict(epoch=epoch+1,train_loss=means[0],train_prediction=means[1],
                            train_alignment=means[2],train_temporal=means[3],val_mse=metric,**rank))
        pd.DataFrame(history).to_csv(out/"history.csv",index=False)
        if metric < best:
            best = metric
            torch.save(dict(state=model.state_dict(),variant=variant,width=cfg["width"],epoch=epoch+1,
                            qa_hash=qa_hash,metric=metric),out/"best.pt")
            details.to_parquet(out/"validation.parquet",index=False)
        print(f"{variant} epoch {epoch+1}: train {means[0]:.4f} val {metric:.4f}",flush=True)
    save_json(out/"run.json",dict(config=cfg,variant=variant,fit_split="train",selection_split="val",
              checkpoint_sha256=digest(out/"best.pt"),qa_hash=qa_hash,torch=str(torch.__version__),
              python=platform.python_version(),device=str(device),
              hardware=torch.cuda.get_device_name(device) if device.type=="cuda" else platform.processor(),
              peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None))


def train_all(prepared,out,cfg,variants):
    if cfg["epochs"] < 1:
        raise ValueError("epochs must be positive")
    torch.set_num_threads(2)
    ds = MultiModalArchaeologyDataset(prepared,"train")
    val = MultiModalArchaeologyDataset(prepared,"val")
    out.mkdir(parents=True,exist_ok=False)
    print("Loading train and validation frozen features",flush=True)
    train,valid = compact_dataset(ds),compact_dataset(val)
    for variant in variants:
        train_variant(variant,cfg,train,valid,ds.rows,out/variant,digest(Path(prepared)/"alignment_qa.json"))
