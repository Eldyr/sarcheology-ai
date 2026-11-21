
"""Phase 3.2 — Six-band ViT-Tiny masked autoencoder.

The encoder receives visible tokens only.
The decoder reconstructs the full patch sequence.
Loss is computed on valid pixels in masked patches.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class OpticalViTMAE(nn.Module):
    def __init__(
        self,
        image_size=128,
        patch_size=8,
        in_channels=6,
        mask_ratio=0.75,
        encoder_dim=192,
        encoder_depth=8,
        encoder_heads=3,
        decoder_dim=128,
        decoder_depth=4,
        decoder_heads=4,
    ):
        super().__init__()

        if image_size % patch_size != 0:
            raise ValueError(
                "Patch size must divide image size"
            )

        if not 0 < mask_ratio < 1:
            raise ValueError(
                "Mask ratio must be between 0 and 1"
            )

        if encoder_dim % encoder_heads != 0:
            raise ValueError(
                "Encoder dimension/head mismatch"
            )

        if decoder_dim % decoder_heads != 0:
            raise ValueError(
                "Decoder dimension/head mismatch"
            )

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.mask_ratio = mask_ratio

        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.patch_dim = (
            in_channels * patch_size * patch_size
        )

        self.num_visible = int(
            self.num_patches * (1 - mask_ratio)
        )

        if self.num_visible < 1:
            raise ValueError("No visible tokens")

        self.patch_embed = nn.Conv2d(
            in_channels,
            encoder_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.encoder_pos = nn.Parameter(
            torch.zeros(
                1, self.num_patches, encoder_dim
            )
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=encoder_dim,
            nhead=encoder_heads,
            dim_feedforward=encoder_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=encoder_depth,
            enable_nested_tensor=False,
        )

        self.encoder_norm = nn.LayerNorm(
            encoder_dim
        )

        self.decoder_embed = nn.Linear(
            encoder_dim, decoder_dim
        )

        self.mask_token = nn.Parameter(
            torch.zeros(1, 1, decoder_dim)
        )

        self.decoder_pos = nn.Parameter(
            torch.zeros(
                1, self.num_patches, decoder_dim
            )
        )

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=decoder_heads,
            dim_feedforward=decoder_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.decoder = nn.TransformerEncoder(
            decoder_layer,
            num_layers=decoder_depth,
            enable_nested_tensor=False,
        )

        self.decoder_norm = nn.LayerNorm(
            decoder_dim
        )

        self.decoder_prediction = nn.Linear(
            decoder_dim,
            self.patch_dim,
        )

        nn.init.trunc_normal_(
            self.encoder_pos, std=0.02
        )
        nn.init.trunc_normal_(
            self.decoder_pos, std=0.02
        )
        nn.init.normal_(
            self.mask_token, std=0.02
        )

    def patchify(self, images):
        """[B,C,H,W] -> [B,N,P*P*C]."""
        b, c, h, w = images.shape
        p = self.patch_size

        if (
            c != self.in_channels
            or h != self.image_size
            or w != self.image_size
        ):
            raise ValueError(
                "Unexpected image dimensions"
            )

        x = images.reshape(
            b, c, h // p, p, w // p, p
        )

        x = x.permute(
            0, 2, 4, 3, 5, 1
        )

        return x.reshape(
            b, self.num_patches, self.patch_dim
        )

    def unpatchify(self, patches):
        """[B,N,P*P*C] -> [B,C,H,W]."""
        b, n, d = patches.shape

        if (
            n != self.num_patches
            or d != self.patch_dim
        ):
            raise ValueError(
                "Unexpected patch dimensions"
            )

        g = self.grid_size
        p = self.patch_size
        c = self.in_channels

        x = patches.reshape(
            b, g, g, p, p, c
        )

        x = x.permute(
            0, 5, 1, 3, 2, 4
        )

        return x.reshape(
            b, c, self.image_size, self.image_size
        )

    def random_masking(
        self,
        tokens,
        *,
        generator=None,
    ):
        """
        Returns visible tokens and restoration order.

        Uses CPU random numbers for reproducibility
        with a supplied torch.Generator.
        """
        b, n, dim = tokens.shape

        noise = torch.rand(
            b, n, generator=generator,
            device="cpu",
        ).to(tokens.device)

        ids_shuffle = torch.argsort(
            noise, dim=1
        )
        ids_restore = torch.argsort(
            ids_shuffle, dim=1
        )

        ids_keep = ids_shuffle[
            :, :self.num_visible
        ]

        visible = torch.gather(
            tokens,
            1,
            ids_keep.unsqueeze(-1).expand(
                -1, -1, dim
            ),
        )

        mask = torch.ones(
            (b, n),
            device=tokens.device,
            dtype=torch.bool,
        )
        mask[:, :self.num_visible] = False

        mask = torch.gather(
            mask, 1, ids_restore
        )

        return (
            visible,
            mask,
            ids_keep,
            ids_restore,
        )

    def forward(
        self,
        images,
        valid_mask=None,
        *,
        generator=None,
    ):
        if images.ndim != 4 or images.shape[1:] != (
            self.in_channels,
            self.image_size,
            self.image_size,
        ):
            raise ValueError(
                "Expected [B,6,128,128] input"
            )

        b = images.shape[0]

        if valid_mask is None:
            valid = torch.ones(
                b, 1,
                self.image_size,
                self.image_size,
                dtype=torch.bool,
                device=images.device,
            )
        else:
            expected_shape = (
                b, 1,
                self.image_size,
                self.image_size,
            )

            if valid_mask.shape != expected_shape:
                raise ValueError(
                    "Incorrect validity-mask shape"
                )

            valid = valid_mask.bool()

        expanded_valid = valid.expand_as(images)

        if not torch.isfinite(
            images[expanded_valid]
        ).all():
            raise ValueError(
                "Non-finite values inside valid pixels"
            )

        clean = torch.where(
            expanded_valid,
            images,
            torch.zeros_like(images),
        )

        tokens = self.patch_embed(
            clean
        ).flatten(2).transpose(1, 2)

        tokens = tokens + self.encoder_pos

        (
            visible,
            patch_mask,
            ids_keep,
            ids_restore,
        ) = self.random_masking(
            tokens,
            generator=generator,
        )

        encoded = self.encoder(visible)
        encoded = self.encoder_norm(encoded)

        embedding = encoded.mean(dim=1)

        decoder_visible = self.decoder_embed(
            encoded
        )

        n_masked = (
            self.num_patches
            - self.num_visible
        )

        mask_tokens = self.mask_token.expand(
            b, n_masked, -1
        )

        shuffled = torch.cat(
            [decoder_visible, mask_tokens],
            dim=1,
        )

        restored = torch.gather(
            shuffled,
            dim=1,
            index=ids_restore.unsqueeze(-1).expand(
                -1, -1, shuffled.shape[-1]
            ),
        )

        restored = restored + self.decoder_pos

        decoded = self.decoder(restored)
        decoded = self.decoder_norm(decoded)

        predicted_patches = (
            self.decoder_prediction(decoded)
        )

        reconstruction = self.unpatchify(
            predicted_patches
        )

        return {
            "reconstruction": reconstruction,
            "predicted_patches": predicted_patches,
            "patch_mask": patch_mask,
            "ids_keep": ids_keep,
            "ids_restore": ids_restore,
            "embedding": embedding,
            "valid_mask": valid,
        }

    def masked_loss(
        self,
        images,
        valid_mask,
        output,
        *,
        delta=1.0,
    ):
        """Masked valid-pixel Huber reconstruction loss."""
        valid = valid_mask.bool()
        expanded_valid = valid.expand_as(images)

        target = torch.where(
            expanded_valid,
            images,
            torch.zeros_like(images),
        )

        target_patches = self.patchify(
            target.float()
        )

        valid_patches = self.patchify(
            expanded_valid.float()
        ) > 0.5

        masked = output[
            "patch_mask"
        ].unsqueeze(-1)

        scored = masked & valid_patches

        count = scored.sum()

        if not bool(count > 0):
            raise ValueError(
                "No valid masked pixels for loss"
            )

        pred = output[
            "predicted_patches"
        ].float()

        pixel_loss = F.huber_loss(
            pred,
            target_patches,
            reduction="none",
            delta=delta,
        )

        loss = (
            (pixel_loss * scored).sum()
            / count.clamp_min(1)
        )

        return loss
