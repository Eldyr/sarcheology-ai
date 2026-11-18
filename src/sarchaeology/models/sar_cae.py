
"""Phase 2.4 — Sentinel-1 SAR Convolutional Autoencoder."""

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    """Two convolutions with GroupNorm and SiLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        groups: int = 8,
    ):
        super().__init__()

        if out_channels % groups != 0:
            raise ValueError(
                "out_channels must be divisible "
                "by normalization groups"
            )

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class DownBlock(nn.Module):
    """Halve spatial resolution and increase channels."""

    def __init__(
        self,
        in_channels,
        out_channels,
        groups=8,
    ):
        super().__init__()

        self.down = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.features = ConvBlock(
            out_channels,
            out_channels,
            groups=groups,
        )

    def forward(self, x):
        return self.features(self.down(x))


class UpBlock(nn.Module):
    """Upsample by 2 without transposed convolutions."""

    def __init__(
        self,
        in_channels,
        out_channels,
        groups=8,
    ):
        super().__init__()

        self.features = ConvBlock(
            in_channels,
            out_channels,
            groups=groups,
        )

    def forward(self, x):
        x = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )
        return self.features(x)


class SARCAE(nn.Module):
    """
    Convolutional Autoencoder for normalized
    Sentinel-1 VV/VH images.

    Input:
        image: [B, 2, H, W]
        valid_mask: optional [B, 1, H, W]

    H and W must be divisible by 8.

    Outputs:
        reconstruction: [B, 2, H, W]
        latent_map: [B, latent_channels, H/8, W/8]
        embedding: [B, embedding_dim]

    No reconstruction loss is calculated here.
    """

    def __init__(
        self,
        in_channels=2,
        base_channels=32,
        latent_channels=128,
        embedding_dim=128,
        group_norm_groups=8,
    ):
        super().__init__()

        if in_channels != 2:
            raise ValueError(
                "SARCAE expects exactly 2 channels: VV, VH"
            )

        if min(
            base_channels,
            latent_channels,
            embedding_dim,
            group_norm_groups,
        ) <= 0:
            raise ValueError(
                "Model dimensions must be positive"
            )

        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.embedding_dim = embedding_dim

        b = base_channels
        g = group_norm_groups

        # Encoder: 128 -> 64 -> 32 -> 16 pixels.
        self.encoder = nn.Sequential(
            ConvBlock(2, b, groups=g),
            DownBlock(b, b * 2, groups=g),
            DownBlock(b * 2, b * 4, groups=g),
            DownBlock(b * 4, b * 8, groups=g),
            nn.Conv2d(
                b * 8,
                latent_channels,
                kernel_size=1,
            ),
        )

        # Global representation for downstream
        # embedding-based anomaly detection.
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.embedding_projection = nn.Linear(
            latent_channels,
            embedding_dim,
        )

        # Decoder: 16 -> 32 -> 64 -> 128 pixels.
        # No encoder-to-decoder skip connections:
        # reconstruction must pass through the bottleneck.
        self.decoder = nn.Sequential(
            nn.Conv2d(
                latent_channels,
                b * 8,
                kernel_size=1,
            ),
            UpBlock(b * 8, b * 4, groups=g),
            UpBlock(b * 4, b * 2, groups=g),
            UpBlock(b * 2, b, groups=g),
            nn.Conv2d(
                b,
                in_channels,
                kernel_size=1,
            ),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Linear):
                nn.init.trunc_normal_(
                    module.weight,
                    std=0.02,
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode(self, image, valid_mask=None):
        """Return spatial latent map and global embedding."""

        if image.ndim != 4 or image.shape[1] != 2:
            raise ValueError(
                "Expected SAR image [B,2,H,W]"
            )

        height, width = image.shape[-2:]

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                "Image H and W must be divisible by 8"
            )

        if valid_mask is not None:
            expected = (
                image.shape[0],
                1,
                height,
                width,
            )

            if tuple(valid_mask.shape) != expected:
                raise ValueError(
                    f"Expected valid_mask {expected}"
                )

            # Zero-fill invalid input pixels.
            image = image * (
                valid_mask > 0
            ).to(dtype=image.dtype)

        latent = self.encoder(image)

        pooled = self.global_pool(
            latent
        ).flatten(1)

        embedding = self.embedding_projection(
            pooled
        )

        return latent, embedding

    def decode(self, latent):
        """Decode a spatial latent map into VV/VH."""
        return self.decoder(latent)

    def forward(self, image, valid_mask=None):
        latent, embedding = self.encode(
            image,
            valid_mask=valid_mask,
        )

        reconstruction = self.decode(
            latent
        )

        if reconstruction.shape != image.shape:
            raise RuntimeError(
                "Reconstruction shape mismatch: "
                f"{reconstruction.shape} != {image.shape}"
            )

        return {
            "reconstruction": reconstruction,
            "latent_map": latent,
            "embedding": embedding,
        }


def build_sar_cae(config):
    """Build the model from the YAML model section."""

    model_cfg = config["model"]

    if model_cfg["architecture"] != "sar_cae":
        raise ValueError(
            "Unsupported model architecture"
        )

    return SARCAE(
        in_channels=model_cfg["in_channels"],
        base_channels=model_cfg["base_channels"],
        latent_channels=model_cfg["latent_channels"],
        embedding_dim=model_cfg["embedding_dim"],
        group_norm_groups=model_cfg[
            "group_norm_groups"
        ],
    )


def count_parameters(model):
    """Count trainable model parameters."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
