"""Phase 3.5: pretrained OpticalSwinMIM + trainable VICReg projection head.

The exact Phase 3.4 backbone is retained, including masked reconstruction.
"""
import torch
from torch import nn
from sarchaeology.models.optical_swin_mim import OpticalSwinMIM


class TemporalSwin(nn.Module):
    def __init__(self, swin_config, projection_dim=256, hidden_dim=512):
        super().__init__()
        self.backbone = OpticalSwinMIM(**swin_config)
        dim = int(swin_config.get('embed_dim', 96)) * 8
        self.projector = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, image, valid, *, generator=None):
        output = self.backbone(image, valid, generator=generator)
        output['projection'] = self.projector(output['embedding'])
        return output
