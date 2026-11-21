"""SARchaeology AI Phase 3.4: six-band Swin-Tiny masked-image reconstruction.

This is masked image modeling (MIM): the Swin encoder processes a complete
feature grid containing learned mask tokens, NOT visible-token-only MAE.
Only valid masked pixels contribute to the reconstruction loss in the trainer.
Compatible with the Phase 3.3 multi-year trainer's patchify/output API.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class WindowAttention(nn.Module):
    def __init__(self, dim: int, heads: int, window_size: int = 8):
        super().__init__()
        assert dim % heads == 0
        self.heads, self.size = heads, window_size
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.scale = (dim // heads) ** -0.5
        self.relative_bias = nn.Parameter(torch.zeros(heads, (2 * window_size - 1) ** 2))
        coords = torch.stack(torch.meshgrid(torch.arange(window_size), torch.arange(window_size), indexing="ij"))
        coords = coords.flatten(1)
        relative = coords[:, :, None] - coords[:, None, :]
        relative = relative.permute(1, 2, 0)
        idx = (relative[..., 0] + window_size - 1) * (2 * window_size - 1) + (relative[..., 1] + window_size - 1)
        self.register_buffer("relative_index", idx, persistent=False)
        nn.init.trunc_normal_(self.relative_bias, std=0.02)

    def forward(self, x: torch.Tensor, allowed: torch.Tensor | None):
        # x: [batch, n_windows, window_tokens, dim]
        b, nw, t, c = x.shape
        qkv = self.qkv(x).reshape(b, nw, t, 3, self.heads, c // self.heads)
        q, k, v = qkv.permute(3, 0, 1, 4, 2, 5).unbind(0)
        bias = self.relative_bias[:, self.relative_index.reshape(-1)].reshape(self.heads, t, t)
        additive = bias[None, None].to(q.dtype)
        if allowed is not None:
            # Different shifted regions within a window must not attend.
            additive = additive.masked_fill(~allowed[None, :, None, :, :], torch.finfo(q.dtype).min)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=additive, dropout_p=0.0)
        y = y.transpose(2, 3).reshape(b, nw, t, c)
        return self.proj(y)


class SwinBlock(nn.Module):
    def __init__(self, dim: int, heads: int, resolution: int, window_size: int = 8, shift: bool = False):
        super().__init__()
        self.resolution = resolution
        self.window = min(window_size, resolution)
        self.shift = self.window // 2 if shift and resolution > self.window else 0
        assert resolution % self.window == 0
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, heads, self.window)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        if self.shift:
            # Label each pixel by its ORIGINAL non-wrapping region, then roll.
            r = resolution
            labels = torch.zeros((r, r), dtype=torch.long)
            boundaries = (slice(0, -self.window), slice(-self.window, -self.shift), slice(-self.shift, None))
            label = 0
            for ys in boundaries:
                for xs in boundaries:
                    labels[ys, xs] = label
                    label += 1
            labels = labels.reshape(r // self.window, self.window, r // self.window, self.window)
            labels = labels.permute(0, 2, 1, 3).reshape(-1, self.window * self.window)
            self.register_buffer("allowed", labels[:, :, None] == labels[:, None, :], persistent=False)
        else:
            self.allowed = None

    def forward(self, x: torch.Tensor):
        # channels-last [B, H, W, C]
        b, h, w, c = x.shape
        assert h == w == self.resolution
        shortcut = x
        x = self.norm1(x)
        if self.shift:
            x = torch.roll(x, (-self.shift, -self.shift), (1, 2))
        win = self.window
        nw = h // win
        x = x.reshape(b, nw, win, nw, win, c).permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(b, nw * nw, win * win, c)
        x = self.attn(x, self.allowed)
        x = x.reshape(b, nw, nw, win, win, c).permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(b, h, w, c)
        if self.shift:
            x = torch.roll(x, (self.shift, self.shift), (1, 2))
        x = shortcut + x
        return x + self.mlp(self.norm2(x))


class PatchMerge(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x):
        b, h, w, c = x.shape
        assert h % 2 == 0 and w % 2 == 0
        x = torch.cat((x[:, 0::2, 0::2], x[:, 1::2, 0::2],
                       x[:, 0::2, 1::2], x[:, 1::2, 1::2]), dim=-1)
        return self.reduction(self.norm(x))


class OpticalSwinMIM(nn.Module):
    """Swin-Tiny-style hierarchical encoder with multiscale reconstruction head."""
    def __init__(self, image_size=128, patch_size=8, in_channels=6,
                 mask_ratio=0.75, embed_dim=96, depths=(2, 2, 6, 2),
                 heads=(3, 6, 12, 24), window_size=8, decoder_dim=128):
        super().__init__()
        if image_size != 128 or patch_size != 8 or in_channels != 6:
            raise ValueError("Phase 3.4 expects six-band 128x128 images with 8x8 scoring patches")
        if not 0 < mask_ratio < 1:
            raise ValueError("mask_ratio must lie strictly between zero and one")
        if len(depths) != 4 or len(heads) != 4:
            raise ValueError("Expected four Swin stages")
        self.image_size, self.patch_size = image_size, patch_size
        self.in_channels, self.mask_ratio = in_channels, mask_ratio
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = patch_size * patch_size * in_channels
        self.num_visible = int(self.num_patches * (1 - mask_ratio))
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=4, stride=4)
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim, 1, 1))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.stages = nn.ModuleList()
        self.merges = nn.ModuleList()
        dims = [embed_dim * 2 ** i for i in range(4)]
        for i in range(4):
            resolution = 32 // (2 ** i)
            self.stages.append(nn.Sequential(*[
                SwinBlock(dims[i], heads[i], resolution, window_size, shift=j % 2 == 1)
                for j in range(depths[i])
            ]))
            if i < 3:
                self.merges.append(PatchMerge(dims[i]))
        self.encoder_norm = nn.LayerNorm(dims[-1])
        self.lateral = nn.ModuleList([nn.Conv2d(d, decoder_dim, 1) for d in dims])
        self.refine = nn.Sequential(
            nn.Conv2d(decoder_dim, decoder_dim, 3, padding=1), nn.GELU(),
            nn.Conv2d(decoder_dim, in_channels, 1),
        )

    def patchify(self, images):
        b, c, h, w = images.shape
        p = self.patch_size
        if (c, h, w) != (self.in_channels, self.image_size, self.image_size):
            raise ValueError("Wrong image shape")
        return images.reshape(b, c, h // p, p, w // p, p).permute(0, 2, 4, 3, 5, 1).reshape(b, self.num_patches, self.patch_dim)

    def unpatchify(self, patches):
        b, n, d = patches.shape
        p = self.patch_size
        g = self.image_size // p
        if (n, d) != (self.num_patches, self.patch_dim):
            raise ValueError("Wrong patch shape")
        return patches.reshape(b, g, g, p, p, self.in_channels).permute(0, 5, 1, 3, 2, 4).reshape(b, self.in_channels, self.image_size, self.image_size)

    def forward(self, images, valid_mask=None, *, generator=None):
        b, c, h, w = images.shape
        if (c, h, w) != (self.in_channels, self.image_size, self.image_size):
            raise ValueError("Input must have shape [B,6,128,128]")
        if valid_mask is None:
            valid_mask = torch.ones(b, 1, h, w, dtype=torch.bool, device=images.device)
        if valid_mask.shape != (b, 1, h, w):
            raise ValueError("Wrong validity mask shape")
        valid_mask = valid_mask.bool()
        if not torch.isfinite(images[valid_mask.expand_as(images)]).all():
            raise ValueError("Nonfinite valid input")
        images = torch.where(valid_mask.expand_as(images), images, torch.zeros_like(images))
        # Same CPU RNG mask mechanism as ViT; per-sample mask at 8x8 scale.
        noise = torch.rand(b, self.num_patches, generator=generator, device="cpu").to(images.device)
        order = noise.argsort(dim=1)
        ids_restore = order.argsort(dim=1)
        mask = torch.ones(b, self.num_patches, dtype=torch.bool, device=images.device)
        mask[:, :self.num_visible] = False
        mask = mask.gather(1, ids_restore)
        grid = self.image_size // self.patch_size
        masked_grid = mask.reshape(b, 1, grid, grid)
        x = self.patch_embed(images)
        token_mask = F.interpolate(masked_grid.float(), size=x.shape[-2:], mode="nearest").bool()
        x = torch.where(token_mask, self.mask_token.to(x.dtype), x)
        x = x.permute(0, 2, 3, 1)
        features = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            features.append(x.permute(0, 3, 1, 2))
            if i < 3:
                x = self.merges[i](x)
        embedding = self.encoder_norm(x).mean(dim=(1, 2))
        # Decode via multiscale feature fusion; masked locations cannot see input truth.
        target_size = features[0].shape[-2:]
        decoder = sum(F.interpolate(lateral(feature), size=target_size, mode="bilinear", align_corners=False)
                      for lateral, feature in zip(self.lateral, features))
        prediction = self.refine(decoder)
        reconstruction = F.interpolate(prediction, size=(h, w), mode="bilinear", align_corners=False)
        return {"predicted_patches": self.patchify(reconstruction),
                "patch_mask": mask, "reconstruction": reconstruction,
                "embedding": embedding, "valid_mask": valid_mask}
