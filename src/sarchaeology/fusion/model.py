"""Local fusion on the finest native exported optical grid (40 m at Buto)."""
import torch
from torch import nn
from torch.nn import functional as F

MODALITIES = ("sar","optical","dem")
VARIANTS = ("gated","concat","attention",*MODALITIES)
TARGET_WIDTHS = (2,8,9)


def pool(x):
    return F.avg_pool2d(x.float(),4)


def lift(x):
    return F.interpolate(x.float(),scale_factor=2,mode="nearest")


def inputs(batch):
    if "targets" in batch:
        return batch["mask"],batch["quality"],batch["targets"]
    counts = batch["sar_count"].float()[:,None]
    sar = counts > 0
    optical, dem = batch["optical_valid"][:,None], batch["dem_valid"][:,None]
    coverage = torch.cat([pool(m) for m in (sar,optical,dem)],1)
    quality = coverage.clone()
    quality[:,:1] = pool(counts)/batch["available_dates"].clamp_min(1)[:,None,None,None]
    mask = coverage >= 0.5
    terrain = pool(batch["dem"].float()*dem)/coverage[:,2:3].clamp_min(1e-8)
    db = pool((batch["sar_db"].float()+15)*sar)/coverage[:,:1].clamp_min(1e-8)/5
    stage0 = batch["stage0"].float()
    # Fixed targets, including when their modality is hidden from fusion.
    targets = (db, stage0[:,:8], terrain)
    return mask, quality, targets


class LocalFusion(nn.Module):
    def __init__(self, variant="gated", width=32):
        super().__init__()
        if variant not in VARIANTS or width % 4:
            raise ValueError("Unknown variant or width not divisible by four")
        self.variant = variant
        self.width = width
        self.sar = nn.Conv2d(128,width,1)
        self.stage0 = nn.Conv2d(96,width,1)
        self.stage1 = nn.Conv2d(192,width,1)
        self.embedding = nn.Linear(256,width)
        self.dem = nn.Sequential(nn.Conv2d(9,width,1),nn.SiLU(),nn.Conv2d(width,width,1))
        if variant == "gated":
            self.gates = nn.ModuleList(nn.Conv2d(width+1,1,1) for _ in MODALITIES)
        if variant == "concat":
            self.concat = nn.Conv2d(3*width+3,width,1)
        if variant == "attention":
            self.query = nn.Parameter(torch.zeros(4,width//4))
            self.keys = nn.Linear(width,width)
            self.values = nn.Linear(width,width)
        self.decoder = nn.Sequential(nn.Conv2d(width,width,1),nn.SiLU(),nn.Conv2d(width,sum(TARGET_WIDTHS),1))

    def project(self,batch,terrain):
        opt = (self.stage0(batch["stage0"].float()) + lift(self.stage1(batch["stage1"].float()))
               + self.embedding(batch["optical_embedding"].float())[:,:,None,None])
        return torch.stack([lift(self.sar(batch["sar_features"].float())),opt,self.dem(terrain)],1)

    def fuse(self, tokens, valid, quality):
        if self.variant in MODALITIES:
            active = torch.arange(3,device=tokens.device) == MODALITIES.index(self.variant)
            valid = valid & active[None,:,None,None]
        tokens = tokens*valid[:,:,None]
        denom = valid.sum(1,keepdim=True).clamp_min(1)
        weights = valid.float()/denom
        if self.variant == "gated":
            logits = torch.cat([gate(torch.cat([tokens[:,i], quality[:,i:i+1]],1))
                                for i,gate in enumerate(self.gates)],1)
            weights = logits.float().masked_fill(~valid,-1e4).softmax(1)*valid
        if self.variant == "attention":
            b,m,c,h,w = tokens.shape
            seq = tokens.permute(0,3,4,1,2)
            keys = self.keys(seq).reshape(b,h,w,m,4,c//4)
            values = self.values(seq).reshape_as(keys)
            logits = (keys*self.query).sum(-1)/(c//4)**0.5
            allowed = valid.permute(0,2,3,1)[...,None]
            attention = logits.float().masked_fill(~allowed,-1e4).softmax(3)*allowed
            fused = (attention[...,None]*values).sum(3).reshape(b,h,w,c).permute(0,3,1,2)
            weights = attention.mean(-1).permute(0,3,1,2)
        elif self.variant == "concat":
            fused = self.concat(torch.cat([tokens.flatten(1,2),valid.float()],1))
        else:
            fused = (weights[:,:,None]*tokens).sum(1)
        supported = valid.any(1,keepdim=True)
        return fused*supported, weights, supported

    def forward(self, batch, dropped=None):
        mask,quality,targets = inputs(batch)
        tokens = self.project(batch, targets[2])
        observed = mask if dropped is None else mask & ~dropped
        fused,weights,support = self.fuse(tokens,observed,quality)
        return dict(fused=fused,weights=weights,support=support,mask=mask,quality=quality,
                    targets=targets,tokens=tokens,predictions=self.decoder(fused).split(TARGET_WIDTHS,1))
