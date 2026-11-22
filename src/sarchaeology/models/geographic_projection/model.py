"""Frozen-Swin geographic temporal projection head and training losses."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F


class GeographicProjector(nn.Module):
    """Two-layer trainable residual projection from a frozen 768D encoder."""
    def __init__(self, input_dim=768, hidden_dim=512, output_dim=256):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                     nn.LayerNorm(hidden_dim), nn.GELU(),
                                     nn.Linear(hidden_dim, output_dim))
        self.skip = nn.Linear(input_dim, output_dim, bias=False)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, inputs):
        return F.normalize(self.skip(inputs.float()) + self.scale * self.network(inputs.float()), dim=-1)


def contrastive_loss(z, ids, temperature=0.12):
    """Multi-positive supervised InfoNCE. Positive IDs identical; self excluded.

    All other IDs act as negatives. Construct batches with unique geographic IDs,
    so the 2-view batch contains exactly one matched positive per anchor.
    """
    if z.ndim != 2 or len(ids) != len(z):
        raise ValueError('Incorrect embedding/identity batch')
    ids = ids.to(z.device)
    z = F.normalize(z.float(), dim=1)
    logits = (z @ z.T) / temperature
    eye = torch.eye(len(z), device=z.device, dtype=torch.bool)
    positives = (ids[:, None] == ids[None, :]) & ~eye
    if not positives.any(dim=1).all():
        raise ValueError('Every anchor must have a second same-location view')
    logits = logits.masked_fill(eye, -torch.inf)
    logprob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    return -logprob.masked_fill(~positives, 0).sum(dim=1).div(positives.sum(1)).mean()


def variance_covariance_loss(z, target_std=0.05):
    """Variance and off-diagonal covariance regularization on L2-unit vectors."""
    z = z.float()
    centered = z - z.mean(0)
    std = (centered.var(dim=0, unbiased=len(z)>1) + 1e-4).sqrt()
    variance = F.relu(target_std - std).mean()
    cov = centered.T @ centered / max(len(z)-1, 1)
    cov.fill_diagonal_(0)
    covariance = cov.square().sum() / z.shape[1]
    return variance, covariance


def relational_loss(student, teacher, temperature=0.2):
    """KL preservation of pretrained pairwise similarity distributions.

    The teacher is frozen and remains untouched. Soft distributions preserve
    neighborhood ordering but can conflict with temporal contrastive alignment.
    """
    s = F.normalize(student.float(), dim=1)
    t = F.normalize(teacher.detach().float(), dim=1)
    eye = torch.eye(len(s), dtype=torch.bool, device=s.device)
    st = (s @ s.T / temperature).masked_fill(eye, -1e4)
    te = (t @ t.T / temperature).masked_fill(eye, -1e4)
    return F.kl_div(F.log_softmax(st, dim=1), F.softmax(te, dim=1), reduction='batchmean')
