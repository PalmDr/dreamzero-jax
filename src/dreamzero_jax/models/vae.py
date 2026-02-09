"""WAN Video VAE encoder/decoder for DreamZero.

Compresses video frames into a latent space with 8x spatial / 4x temporal
downsampling. Uses channels-last (B, T, H, W, C) layout throughout.
"""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
from flax import nnx

from dreamzero_jax.nn.attention import Attention


# ---------------------------------------------------------------------------
# CausalConv3d
# ---------------------------------------------------------------------------


class CausalConv3d(nnx.Module):
    """3D convolution with causal (left-only) temporal padding.

    Spatial axes use symmetric padding. The time axis pads only on the left
    so future frames never influence past outputs.

    Input/output layout: ``(B, T, H, W, C)`` (channels-last).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int] = 3,
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] = 0,
        *,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)

        self.kernel_size = tuple(kernel_size)
        self.stride = tuple(stride)
        self.padding = tuple(padding)

        # Causal temporal padding: all on the left, none on the right.
        # For stride=1: need (kernel_t - 1) total temporal context.
        # With `padding_t` from caller (PyTorch convention): 2 * padding_t on left.
        # For stride>1: same formula — the stride handles subsampling.
        self._pad_t = 2 * self.padding[0]
        self._pad_h = self.padding[1]
        self._pad_w = self.padding[2]

        # Internal conv uses VALID padding — we handle all padding manually.
        self.conv = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=self.kernel_size,
            strides=self.stride,
            padding="VALID",
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass.

        Args:
            x: ``(B, T, H, W, C)``

        Returns:
            Convolved tensor with causal temporal padding.
        """
        # Pad: (dim, (before, after)) for each axis.
        # Axes: B=0, T=1, H=2, W=3, C=4
        pad_widths = (
            (0, 0),                    # batch
            (self._pad_t, 0),          # time: causal (left-only)
            (self._pad_h, self._pad_h),  # height: symmetric
            (self._pad_w, self._pad_w),  # width: symmetric
            (0, 0),                    # channels
        )
        x = jnp.pad(x, pad_widths, mode="constant")
        return self.conv(x)


# ---------------------------------------------------------------------------
# ResidualBlock
# ---------------------------------------------------------------------------


class ResidualBlock(nnx.Module):
    """Pre-norm residual block using CausalConv3d.

    ``x → RMSNorm → SiLU → CausalConv3d(in, out, 3) → RMSNorm → SiLU →
    CausalConv3d(out, out, 3) → (+shortcut)``

    When ``in_channels != out_channels``, a 1×1×1 CausalConv3d is used as
    the shortcut projection.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.norm1 = nnx.RMSNorm(in_channels, rngs=rngs)
        self.conv1 = CausalConv3d(
            in_channels, out_channels, kernel_size=3, padding=1,
            dtype=dtype, param_dtype=param_dtype, rngs=rngs,
        )
        self.norm2 = nnx.RMSNorm(out_channels, rngs=rngs)
        self.conv2 = CausalConv3d(
            out_channels, out_channels, kernel_size=3, padding=1,
            dtype=dtype, param_dtype=param_dtype, rngs=rngs,
        )

        if in_channels != out_channels:
            self.shortcut = CausalConv3d(
                in_channels, out_channels, kernel_size=1,
                dtype=dtype, param_dtype=param_dtype, rngs=rngs,
            )
        else:
            self.shortcut = None

    def __call__(self, x: jax.Array) -> jax.Array:
        h = jax.nn.silu(self.norm1(x))
        h = self.conv1(h)
        h = jax.nn.silu(self.norm2(h))
        h = self.conv2(h)

        if self.shortcut is not None:
            x = self.shortcut(x)
        return x + h


# ---------------------------------------------------------------------------
# AttentionBlock
# ---------------------------------------------------------------------------


class AttentionBlock(nnx.Module):
    """Spatial-only single-head self-attention block.

    Operates per-frame: reshapes ``(B, T, H, W, C)`` to ``(B*T, H*W, C)``,
    applies single-head attention, reshapes back. The output projection is
    zero-initialized so the block starts as an identity.
    """

    def __init__(
        self,
        dim: int,
        *,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.norm = nnx.RMSNorm(dim, rngs=rngs)
        self.attn = Attention(
            dim=dim,
            num_heads=1,
            qkv_bias=True,
            out_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        # Zero-init the output projection so the block starts as identity.
        self.attn.out_proj.kernel[...] = jnp.zeros_like(self.attn.out_proj.kernel[...])
        if self.attn.out_proj.bias is not None:
            self.attn.out_proj.bias[...] = jnp.zeros_like(self.attn.out_proj.bias[...])

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass.

        Args:
            x: ``(B, T, H, W, C)``

        Returns:
            Same shape with spatial attention applied per frame.
        """
        B, T, H, W, C = x.shape
        h = self.norm(x)
        # Flatten to (B*T, H*W, C) for attention
        h = h.reshape(B * T, H * W, C)
        h = self.attn(h)
        h = h.reshape(B, T, H, W, C)
        return x + h


# ---------------------------------------------------------------------------
# Resample (up/down in spatial and temporal)
# ---------------------------------------------------------------------------


class SpatialDownsample(nnx.Module):
    """Spatial 2x downsampling via asymmetric-padded stride-2 conv, per frame."""

    def __init__(
        self,
        dim: int,
        *,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        # Per-frame 2D conv with stride 2. We use a 3×3 kernel.
        # Asymmetric pad: (0,1) on H and W before stride-2 conv (like PyTorch).
        self.conv = nnx.Conv(
            in_features=dim,
            out_features=dim,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="VALID",
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Downsample spatial dims by 2.

        Args:
            x: ``(B, T, H, W, C)``

        Returns:
            ``(B, T, H//2, W//2, C)``
        """
        B, T, H, W, C = x.shape
        x = x.reshape(B * T, H, W, C)
        # Asymmetric pad: 0 left, 1 right on both H and W
        x = jnp.pad(x, ((0, 0), (0, 1), (0, 1), (0, 0)), mode="constant")
        x = self.conv(x)
        _, H2, W2, _ = x.shape
        return x.reshape(B, T, H2, W2, C)


class TemporalDownsample(nnx.Module):
    """Temporal 2x downsampling via causal stride-2 conv."""

    def __init__(
        self,
        dim: int,
        *,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.conv = CausalConv3d(
            dim, dim, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0),
            dtype=dtype, param_dtype=param_dtype, rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Downsample temporal dim by 2.

        Args:
            x: ``(B, T, H, W, C)``

        Returns:
            ``(B, T//2, H, W, C)``
        """
        return self.conv(x)


class SpatialUpsample(nnx.Module):
    """Spatial 2x upsampling via nearest-neighbor resize + conv, per frame."""

    def __init__(
        self,
        dim: int,
        *,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.conv = nnx.Conv(
            in_features=dim,
            out_features=dim,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Upsample spatial dims by 2.

        Args:
            x: ``(B, T, H, W, C)``

        Returns:
            ``(B, T, 2*H, 2*W, C)``
        """
        B, T, H, W, C = x.shape
        x = x.reshape(B * T, H, W, C)
        x = jax.image.resize(x, (B * T, H * 2, W * 2, C), method="nearest")
        x = self.conv(x)
        _, H2, W2, _ = x.shape
        return x.reshape(B, T, H2, W2, C)


class TemporalUpsample(nnx.Module):
    """Temporal 2x upsampling via channel-doubling CausalConv3d + interleave.

    Uses a CausalConv3d that doubles the channel dim, then reshapes to
    interleave the extra channels as new frames.
    """

    def __init__(
        self,
        dim: int,
        *,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.conv = CausalConv3d(
            dim, dim * 2, kernel_size=(3, 1, 1), padding=(1, 0, 0),
            dtype=dtype, param_dtype=param_dtype, rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Upsample temporal dim by 2.

        Args:
            x: ``(B, T, H, W, C)``

        Returns:
            ``(B, 2*T, H, W, C)``
        """
        B, T, H, W, C = x.shape
        x = self.conv(x)  # (B, T, H, W, 2*C)
        # Reshape: split last dim into (2, C), move the 2 next to T, then merge.
        x = x.reshape(B, T, H, W, 2, C)
        x = x.transpose(0, 1, 4, 2, 3, 5)  # (B, T, 2, H, W, C)
        x = x.reshape(B, T * 2, H, W, C)
        return x


# ---------------------------------------------------------------------------
# Encoder3d
# ---------------------------------------------------------------------------


class Encoder3d(nnx.Module):
    """Video encoder: compresses ``(B, T, H, W, 3)`` to latent ``(B, T', H', W', 2*z_dim)``.

    Architecture:
        stem → stages (ResBlocks + Resample) → middle (ResBlock + Attn + ResBlock) → head
    """

    def __init__(
        self,
        z_dim: int = 16,
        base_dim: int = 96,
        dim_multipliers: Sequence[int] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        temporal_downsample: Sequence[bool] = (False, True, True),
        *,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        dims = [base_dim * m for m in dim_multipliers]
        kw = dict(dtype=dtype, param_dtype=param_dtype, rngs=rngs)

        # Stem
        self.stem = CausalConv3d(3, dims[0], kernel_size=3, padding=1, **kw)

        # Encoder stages
        self.stages = nnx.List([])
        for i in range(len(dims)):
            in_ch = dims[i - 1] if i > 0 else dims[0]
            out_ch = dims[i]

            # ResBlocks: first block may change channels, rest preserve
            blocks = []
            for j in range(num_res_blocks):
                blocks.append(ResidualBlock(
                    in_ch if j == 0 else out_ch, out_ch, **kw,
                ))
            stage = nnx.Dict({
                "blocks": nnx.List(blocks),
            })

            # Downsample (all stages except the last)
            if i < len(dims) - 1:
                resample_layers = nnx.List([SpatialDownsample(out_ch, **kw)])
                if temporal_downsample[i]:
                    resample_layers.append(TemporalDownsample(out_ch, **kw))
                stage["resample"] = resample_layers
            else:
                stage["resample"] = nnx.List([])

            self.stages.append(stage)

        # Middle
        last_dim = dims[-1]
        self.mid_block1 = ResidualBlock(last_dim, last_dim, **kw)
        self.mid_attn = AttentionBlock(last_dim, **kw)
        self.mid_block2 = ResidualBlock(last_dim, last_dim, **kw)

        # Head
        self.head_norm = nnx.RMSNorm(last_dim, rngs=rngs)
        self.head_conv = CausalConv3d(last_dim, 2 * z_dim, kernel_size=3, padding=1, **kw)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Encode video to latent distribution parameters.

        Args:
            x: ``(B, T, H, W, 3)`` pixel-space video.

        Returns:
            ``(B, T', H', W', 2*z_dim)`` — mean and logvar concatenated.
        """
        x = self.stem(x)

        for stage in self.stages:
            for block in stage["blocks"]:
                x = block(x)
            for layer in stage["resample"]:
                x = layer(x)

        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x)

        x = jax.nn.silu(self.head_norm(x))
        x = self.head_conv(x)
        return x


# ---------------------------------------------------------------------------
# Decoder3d
# ---------------------------------------------------------------------------


class Decoder3d(nnx.Module):
    """Video decoder: reconstructs ``(B, T, H, W, 3)`` from latent ``(B, T', H', W', z_dim)``.

    Architecture:
        stem → middle (ResBlock + Attn + ResBlock) → stages (ResBlocks + Resample) → head
    """

    def __init__(
        self,
        z_dim: int = 16,
        base_dim: int = 96,
        dim_multipliers: Sequence[int] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        temporal_upsample: Sequence[bool] = (False, True, True),
        *,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        dims = [base_dim * m for m in dim_multipliers]
        # Decoder processes in reverse order of encoder dims.
        rev_dims = list(reversed(dims))
        kw = dict(dtype=dtype, param_dtype=param_dtype, rngs=rngs)

        # Stem
        self.stem = CausalConv3d(z_dim, rev_dims[0], kernel_size=3, padding=1, **kw)

        # Middle
        self.mid_block1 = ResidualBlock(rev_dims[0], rev_dims[0], **kw)
        self.mid_attn = AttentionBlock(rev_dims[0], **kw)
        self.mid_block2 = ResidualBlock(rev_dims[0], rev_dims[0], **kw)

        # Decoder stages (num_res_blocks + 1 blocks each)
        # The temporal_upsample schedule is reversed relative to encoder.
        rev_temporal = list(reversed(temporal_upsample))
        num_dec_blocks = num_res_blocks + 1

        self.stages = nnx.List([])
        for i in range(len(rev_dims)):
            in_ch = rev_dims[i - 1] if i > 0 else rev_dims[0]
            out_ch = rev_dims[i]

            blocks = []
            for j in range(num_dec_blocks):
                blocks.append(ResidualBlock(
                    in_ch if j == 0 else out_ch, out_ch, **kw,
                ))
            stage = nnx.Dict({
                "blocks": nnx.List(blocks),
            })

            # Upsample (all stages except the last)
            if i < len(rev_dims) - 1:
                resample_layers = nnx.List([])
                if rev_temporal[i]:
                    resample_layers.append(TemporalUpsample(out_ch, **kw))
                resample_layers.append(SpatialUpsample(out_ch, **kw))
                stage["resample"] = resample_layers
            else:
                stage["resample"] = nnx.List([])

            self.stages.append(stage)

        # Head
        last_dim = rev_dims[-1]
        self.head_norm = nnx.RMSNorm(last_dim, rngs=rngs)
        self.head_conv = CausalConv3d(last_dim, 3, kernel_size=3, padding=1, **kw)

    def __call__(self, z: jax.Array) -> jax.Array:
        """Decode latent to pixel-space video.

        Args:
            z: ``(B, T', H', W', z_dim)`` latent tensor.

        Returns:
            ``(B, T, H, W, 3)`` reconstructed video.
        """
        x = self.stem(z)

        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x)

        for stage in self.stages:
            for block in stage["blocks"]:
                x = block(x)
            for layer in stage["resample"]:
                x = layer(x)

        x = jax.nn.silu(self.head_norm(x))
        x = self.head_conv(x)
        return x


# ---------------------------------------------------------------------------
# WanVideoVAE — top-level module
# ---------------------------------------------------------------------------

# Pre-computed latent normalization vectors (z_dim=16).
# These are derived from the DreamZero DROID checkpoint and are used to
# normalize latents to approximately zero mean and unit variance.
_LATENT_MEAN = jnp.array([
    -0.7571, -0.7089, -0.9113, 0.0778, -0.5240, -0.8652, -0.5261, -0.0275,
    -0.3474, 0.0429, -0.2780, 0.0377, -0.2498, 0.0203, 0.0737, 0.5765,
], dtype=jnp.float32)

_LATENT_STD = jnp.array([
    7.4687, 7.1546, 6.8892, 6.4774, 7.2596, 7.0694, 7.5413, 5.7266,
    7.2956, 6.6528, 7.0108, 5.6451, 6.4170, 5.3875, 5.7530, 6.5995,
], dtype=jnp.float32)


class WanVideoVAE(nnx.Module):
    """WAN Video VAE with 8x spatial / 4x temporal compression.

    Encodes pixel-space video ``(B, T, H, W, 3)`` to latent ``(B, T/4, H/8, W/8, z_dim)``
    and decodes back. Latent normalization is applied so the latent space has
    approximately zero mean and unit variance.
    """

    def __init__(
        self,
        z_dim: int = 16,
        base_dim: int = 96,
        dim_multipliers: Sequence[int] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        temporal_downsample: Sequence[bool] = (False, True, True),
        mean: jax.Array | None = None,
        std: jax.Array | None = None,
        *,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.z_dim = z_dim
        self.mean = mean if mean is not None else _LATENT_MEAN[:z_dim]
        self.std = std if std is not None else _LATENT_STD[:z_dim]

        self.encoder = Encoder3d(
            z_dim=z_dim,
            base_dim=base_dim,
            dim_multipliers=dim_multipliers,
            num_res_blocks=num_res_blocks,
            temporal_downsample=temporal_downsample,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.decoder = Decoder3d(
            z_dim=z_dim,
            base_dim=base_dim,
            dim_multipliers=dim_multipliers,
            num_res_blocks=num_res_blocks,
            temporal_upsample=temporal_downsample,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def encode(
        self,
        x: jax.Array,
        *,
        sample: bool = False,
        key: jax.Array | None = None,
    ) -> jax.Array:
        """Encode pixel-space video to normalized latent.

        Args:
            x: ``(B, T, H, W, 3)``
            sample: If True, sample from the latent distribution (requires ``key``).
                    If False, return the mean (deterministic).
            key: PRNG key for sampling. Required when ``sample=True``.

        Returns:
            ``(B, T/4, H/8, W/8, z_dim)`` normalized latent.
        """
        h = self.encoder(x)  # (B, T', H', W', 2*z_dim)
        mean, logvar = jnp.split(h, 2, axis=-1)

        if sample:
            assert key is not None, "key required for sampling"
            std = jnp.exp(0.5 * logvar)
            z = mean + std * jax.random.normal(key, mean.shape)
        else:
            z = mean

        # Normalize latent
        z = (z - self.mean) / self.std
        return z

    def decode(self, z: jax.Array) -> jax.Array:
        """Decode normalized latent to pixel-space video.

        Args:
            z: ``(B, T', H', W', z_dim)`` normalized latent.

        Returns:
            ``(B, T, H, W, 3)`` reconstructed video.
        """
        # Denormalize latent
        z = z * self.std + self.mean
        return self.decoder(z)
