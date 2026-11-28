"""A small spatial-spectral residual adapter on a frozen geographic teacher."""
import torch
from torch import nn
import torch.nn.functional as F


class SpatialSpectralGatedAdapter(nn.Module):
    """Fuse aligned 8x8 Swin/SpectralMAE maps; retain teacher as the anchor.

    Inputs:
      teacher: [B,256] frozen Phase 3.5.2 geographic projection
      swin_map: [B,384,8,8] stage-2 Swin features
      spectral_map: [B,96,8,8] pooled SpectralMAE spatial features
    """
    def __init__(self, swin_channels=384, spectral_channels=96,
                 hidden_dim=128, embedding_dim=256, residual_scale=0.05):
        super().__init__()
        self.swin_norm = nn.GroupNorm(12, swin_channels)
        self.spectral_norm = nn.GroupNorm(12, spectral_channels)
        self.swin_adapter = nn.Conv2d(swin_channels, hidden_dim, 1)
        self.spectral_adapter = nn.Conv2d(spectral_channels, hidden_dim, 1)
        self.gate = nn.Sequential(nn.Conv2d(hidden_dim * 2, hidden_dim, 1), nn.GELU(),
                                  nn.Conv2d(hidden_dim, 1, 1), nn.Sigmoid())
        self.mixer = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                                   nn.GELU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                   nn.Linear(hidden_dim, embedding_dim))
        self.delta_norm = nn.LayerNorm(embedding_dim)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))

    def forward(self, teacher, swin_map, spectral_map, *, use_spectral=True):
        if teacher.ndim != 2 or teacher.shape[1] != 256:
            raise ValueError('Teacher must have shape [B,256]')
        if swin_map.shape[0] != teacher.shape[0] or swin_map.shape[1:] != (384,8,8):
            raise ValueError('Swin stage-2 map must have shape [B,384,8,8]')
        if spectral_map.shape[0] != teacher.shape[0] or spectral_map.shape[1:] != (96,8,8):
            raise ValueError('Spectral map must have shape [B,96,8,8]')
        a = self.swin_adapter(self.swin_norm(swin_map.float()))
        b = (self.spectral_adapter(self.spectral_norm(spectral_map.float()))
             if use_spectral else torch.zeros_like(a))
        g = self.gate(torch.cat((a,b),dim=1))
        merged = a + g * b
        delta = torch.tanh(self.delta_norm(self.mixer(merged)))
        z = F.normalize(F.normalize(teacher.float(), dim=1) + self.residual_scale * delta, dim=1)
        return {'embedding':z, 'gate_mean':g.mean(), 'delta':delta}
