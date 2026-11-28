"""Train-reference anomaly scores calibrated on spatially disjoint training footprints."""
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from .io import digest, read_json
from .model import LocalFusion, inputs
from .training import to_device

METHODS = ("standardized_distance","knn_distance","isolation","cross_reconstruction","disagreement")


def restore(directory,prepared,device):
    directory = Path(directory)
    run = read_json(directory/"run.json")
    if run["checkpoint_sha256"] != digest(directory/"best.pt") or run["qa_hash"] != digest(Path(prepared)/"alignment_qa.json"):
        raise ValueError("Checkpoint or prepared dataset hash mismatch")
    cp = torch.load(directory/"best.pt",map_location=device,weights_only=True)
    model = LocalFusion(cp["variant"],cp["width"]).to(device)
    model.load_state_dict(cp["state"])
    return model.eval(),run


@torch.no_grad()
def describe(model,batch,minimum):
    if model.variant not in ("gated","concat","attention") or minimum not in (2,3):
        raise ValueError("Multimodal anomaly scoring requires fusion and two or three modalities")
    mask,quality,targets = inputs(batch)
    tokens = model.project(batch,targets[2])
    fused,weights,_ = model.fuse(tokens,mask,quality)
    reconstruction = torch.zeros_like(quality[:,0])
    disagreement = torch.zeros_like(reconstruction)
    pairs = torch.zeros_like(reconstruction)
    for i in range(3):
        seen = mask.clone()
        seen[:,i] = False
        latent,_,_ = model.fuse(tokens,seen,quality)
        pred = model.decoder(latent).split((2,8,9),1)[i]
        reconstruction += (pred-targets[i]).square().mean(1)*mask[:,i]
        for j in range(i):
            common = mask[:,i] & mask[:,j]
            disagreement += (1-F.cosine_similarity(tokens[:,i],tokens[:,j],dim=1))*common
            pairs += common
    reconstruction /= mask.sum(1).clamp_min(1)
    disagreement /= pairs.clamp_min(1)
    return (fused.permute(0,2,3,1).cpu().numpy(), torch.stack([reconstruction,disagreement],-1).cpu().numpy(),
            (mask.sum(1)>=minimum).cpu().numpy(),weights.cpu().numpy(),quality.cpu().numpy())


def partition(rows,buffer):
    """Reserve the eastern training strip for detector calibration, leaving a spatial gap."""
    cut = np.quantile(rows.col,0.70)
    reference = np.flatnonzero((rows.col+128 <= cut).to_numpy())
    calibration = np.flatnonzero((rows.col >= cut+buffer).to_numpy())
    if min(len(reference),len(calibration)) == 0:
        raise ValueError("Training geography too narrow for disjoint detector calibration strips")
    return reference,calibration


def sample_cells(model,dataset,indices,budget,cfg,device,rng):
    # Bounded pool with the same number of draws per patch-quarter, in stable manifest order.
    candidates,extras = [],[]
    per_patch = max(1,int(np.ceil(2*budget/len(indices))))
    for raw in DataLoader(Subset(dataset,indices),batch_size=cfg["batch_size"]):
        x,e,valid,_,_ = describe(model,to_device(raw,device),cfg["min_modalities"])
        for grid,extra,mask in zip(x,e,valid):
            ix = np.flatnonzero(mask)
            take = rng.choice(ix,min(len(ix),per_patch),replace=False)
            candidates.append(grid.reshape(-1,grid.shape[-1])[take])
            extras.append(extra.reshape(-1,2)[take])
    x,e = np.concatenate(candidates),np.concatenate(extras)
    if len(x) < 6:
        raise ValueError("Insufficient valid train-reference cells")
    take = rng.choice(len(x),min(budget,len(x)),replace=False)
    return x[take], e[take]


class AnomalyDetector:
    def __init__(self, reference,seed):
        self.mean = reference.mean(0)
        self.scale = reference.std(0).clip(1e-5)
        z = (reference-self.mean)/self.scale
        self.neighbors = NearestNeighbors(n_neighbors=min(5,len(z)),algorithm="brute").fit(z)
        self.forest = IsolationForest(n_estimators=64,random_state=seed,n_jobs=1).fit(z)

    def raw(self, x, extra):
        z = (x-self.mean)/self.scale
        with threadpool_limits(limits=2):
            knn = self.neighbors.kneighbors(z)[0].mean(1)
        return np.column_stack([np.sqrt(np.mean(z*z,axis=1)),knn,-self.forest.score_samples(z),extra])

    def calibrate(self,x,extra,quantile):
        if not 0 < quantile < 1:
            raise ValueError("Calibration quantile must be between zero and one")
        raw = self.raw(x,extra)
        self.sorted_scores = np.sort(raw,axis=0)
        combined = self.percentiles(raw).mean(1)
        if not np.isfinite(combined).all() or np.ptp(combined) == 0:
            raise ValueError("Calibration cells yield no usable variation in anomaly score")
        self.threshold = float(np.quantile(combined,quantile))

    def percentiles(self,raw):
        return np.column_stack([np.searchsorted(self.sorted_scores[:,i],raw[:,i],side="right")
                                /len(self.sorted_scores) for i in range(raw.shape[1])])

    def scores(self,x,extra):
        pct = self.percentiles(self.raw(x,extra))
        return np.column_stack([pct.mean(1),pct])


def fit_detector(model,ds,cfg,device,out):
    if set(ds.rows.split) != {"train"}:
        raise ValueError("Detector fitting requires training footprints")
    ref,cal = partition(ds.rows,cfg["buffer_pixels"])
    rng = np.random.default_rng(cfg["seed"])
    x,_ = sample_cells(model,ds,ref,cfg["reference_cells"],cfg,device,rng)
    cx,ce = sample_cells(model,ds,cal,cfg["calibration_cells"],cfg,device,rng)
    detector = AnomalyDetector(x,cfg["seed"])
    detector.calibrate(cx,ce,cfg["threshold_quantile"])
    np.savez(out/"detector.npz",reference=x,sorted_scores=detector.sorted_scores,threshold=detector.threshold,
             seed=cfg["seed"],reference_indices=ref,calibration_indices=cal)
    return detector,dict(reference_patch_quarters=len(ref),calibration_patch_quarters=len(cal),
                        reference_cells=len(x),calibration_cells=len(cx),threshold=detector.threshold)
