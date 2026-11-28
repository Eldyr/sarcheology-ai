"""Phase 4 patch/quarter tensors and location-paired batch sampling."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from sarchaeology.fusion.io import digest, read_json


class MultiModalArchaeologyDataset(Dataset):
    def __init__(self, prepared, split, *, inference_only=False):
        if inference_only and split != "test":
            raise ValueError("Inference-only cache requires the original test split")
        if not inference_only and split not in {"train", "val"}:
            raise ValueError("Phase 4 development accepts train or val only")
        self.root = Path(prepared)
        self.qa = read_json(self.root/"alignment_qa.json")
        if bool(self.qa.get("inference_only")) != inference_only:
            raise ValueError("Development and inference-only caches cannot be interchanged")
        manifest = self.root/"manifest.parquet"
        if digest(manifest) != self.qa["manifest_sha256"]:
            raise ValueError("Prepared manifest hash mismatch")
        rows = pd.read_parquet(manifest)
        if rows.groupby("patch_id").split.nunique().max() != 1:
            raise ValueError("Patch belongs to multiple splits")
        self.rows = rows.loc[rows.split == split].sort_values(["patch_id","period"]).reset_index(drop=True)
        if (self.rows.empty or self.rows.duplicated(["patch_id","period"]).any()
                or not (self.rows.status == "ready").all()):
            raise ValueError(f"Incomplete or duplicate prepared rows for {split}")
        self.paths = []
        for p in self.rows.itertuples():
            path = self.root/p.cache
            if path.resolve() != (self.root/split/f"{p.patch_id}_{p.period}.npz").resolve() or path.parent != self.root/split:
                raise ValueError("Cache split or identity mismatch")
            self.paths.append(path)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        p = self.rows.iloc[index]
        with np.load(self.paths[index], allow_pickle=False) as data:
            sample = {key:torch.from_numpy(data[key]) for key in data.files}
        sample.update(patch_id=p.patch_id, period=p.period, row=int(p.row), col=int(p.col),
                      quarter=int(p.period[-1]), available_dates=len(p.sar_indices))
        return sample


class QuarterPairSampler(Sampler):
    """Shuffle locations and pair consecutive quarters within a location."""
    def __init__(self, rows, batch_size, seed):
        if batch_size < 4 or batch_size % 2:
            raise ValueError("Use an even batch size >= 4")
        self.groups = [g.sort_values("period").index.to_numpy() for _,g in rows.groupby("patch_id")]
        self.batch_size, self.seed, self.epoch = batch_size, seed, 0

    def __len__(self):
        return (sum(map(len,self.groups))+self.batch_size-1)//self.batch_size

    def __iter__(self):
        rng = np.random.default_rng(self.seed+self.epoch)
        pairs = []
        for indices in self.groups:
            pairs.extend(indices[start:start+2] for start in range(0,len(indices),2))
        order = np.concatenate([pairs[i] for i in rng.permutation(len(pairs))])
        for start in range(0,len(order),self.batch_size):
            yield order[start:start+self.batch_size].tolist()
