"""Embedding layers for DreamZero."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from dreamzero_jax.nn.mlp import MLP


def sinusoidal_embedding(timesteps: jax.Array, dim: int, max_period: float = 10000.0) -> jax.Array:
    """Sinusoidal timestep embeddings.

    Args:
        timesteps: 1D array of timesteps, shape (B,)
        dim: Embedding dimension (must be even)
        max_period: Maximum period for the sinusoids

    Returns:
        Embeddings of shape (B, dim)
    """
    half_dim = dim // 2
    freqs = jnp.exp(-jnp.log(max_period) * jnp.arange(half_dim) / half_dim)
    args = timesteps[:, None] * freqs[None, :]
    return jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)


class TimestepEmbedding(nnx.Module):
    """Timestep embedding with MLP projection.

    Converts scalar timesteps to dense embeddings via sinusoidal encoding + MLP.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        *,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        hidden_dim = hidden_dim or dim * 4
        self.dim = dim

        self.mlp = MLP(
            in_features=dim,
            hidden_features=hidden_dim,
            out_features=dim,
            activation=jax.nn.silu,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(self, timesteps: jax.Array) -> jax.Array:
        emb = sinusoidal_embedding(timesteps, self.dim)
        return self.mlp(emb)


class PatchEmbed(nnx.Module):
    """Convert images/video frames to patch tokens.

    For images: (B, H, W, C) -> (B, num_patches, embed_dim)
    For video: (B, T, H, W, C) -> (B, T * num_patches, embed_dim)
    """

    def __init__(
        self,
        patch_size: int | tuple[int, int] = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        bias: bool = True,
        *,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.proj = nnx.Conv(
            in_features=in_channels,
            out_features=embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            use_bias=bias,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        # x: (B, H, W, C) or (B, T, H, W, C)
        is_video = x.ndim == 5

        if is_video:
            B, T, H, W, C = x.shape
            x = x.reshape(B * T, H, W, C)

        x = self.proj(x)  # (B, H', W', embed_dim)
        B_or_BT, H_p, W_p, D = x.shape
        x = x.reshape(B_or_BT, H_p * W_p, D)  # (B, num_patches, embed_dim)

        if is_video:
            x = x.reshape(B, T * H_p * W_p, D)

        return x


class PatchEmbed3D(nnx.Module):
    """3D patch embedding for video.

    (B, T, H, W, C) -> (B, T', H', W', embed_dim) -> (B, num_patches, embed_dim)
    """

    def __init__(
        self,
        patch_size: tuple[int, int, int] = (2, 16, 16),  # (T, H, W)
        in_channels: int = 3,
        embed_dim: int = 768,
        bias: bool = True,
        *,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Use a 3D convolution to extract patches
        # Flax Conv expects (batch, *spatial, channels)
        # For 3D: (B, T, H, W, C)
        self.proj = nnx.Conv(
            in_features=in_channels,
            out_features=embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            use_bias=bias,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        # x: (B, T, H, W, C)
        x = self.proj(x)  # (B, T', H', W', embed_dim)
        B, T_p, H_p, W_p, D = x.shape
        x = x.reshape(B, T_p * H_p * W_p, D)
        return x


def get_1d_sincos_pos_embed(embed_dim: int, length: int) -> jax.Array:
    """1D sinusoidal positional embeddings.

    Args:
        embed_dim: Embedding dimension
        length: Sequence length

    Returns:
        Positional embeddings of shape (length, embed_dim)
    """
    positions = jnp.arange(length)
    return sinusoidal_embedding(positions, embed_dim)


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: tuple[int, int]) -> jax.Array:
    """2D sinusoidal positional embeddings for images.

    Args:
        embed_dim: Embedding dimension (must be divisible by 2)
        grid_size: (H, W) grid dimensions

    Returns:
        Positional embeddings of shape (H * W, embed_dim)
    """
    h, w = grid_size
    grid_h = jnp.arange(h)
    grid_w = jnp.arange(w)
    grid = jnp.meshgrid(grid_w, grid_h, indexing="xy")
    grid = jnp.stack(grid, axis=0).reshape(2, -1).T  # (H*W, 2)

    half_dim = embed_dim // 2
    emb_h = sinusoidal_embedding(grid[:, 1], half_dim)
    emb_w = sinusoidal_embedding(grid[:, 0], half_dim)
    return jnp.concatenate([emb_h, emb_w], axis=-1)


def get_3d_sincos_pos_embed(
    embed_dim: int, grid_size: tuple[int, int, int]
) -> jax.Array:
    """3D sinusoidal positional embeddings for video.

    Args:
        embed_dim: Embedding dimension (must be divisible by 3, or uses 2:1:1 split)
        grid_size: (T, H, W) grid dimensions

    Returns:
        Positional embeddings of shape (T * H * W, embed_dim)
    """
    t, h, w = grid_size

    # Split dimensions: temporal gets 1/2, spatial each get 1/4
    dim_t = embed_dim // 2
    dim_h = embed_dim // 4
    dim_w = embed_dim - dim_t - dim_h  # Handle non-divisible case

    grid_t = jnp.arange(t)
    grid_h = jnp.arange(h)
    grid_w = jnp.arange(w)

    # Create 3D grid
    mesh_t, mesh_h, mesh_w = jnp.meshgrid(grid_t, grid_h, grid_w, indexing="ij")
    mesh_t = mesh_t.reshape(-1)
    mesh_h = mesh_h.reshape(-1)
    mesh_w = mesh_w.reshape(-1)

    emb_t = sinusoidal_embedding(mesh_t, dim_t)
    emb_h = sinusoidal_embedding(mesh_h, dim_h)
    emb_w = sinusoidal_embedding(mesh_w, dim_w)

    return jnp.concatenate([emb_t, emb_h, emb_w], axis=-1)


# ============================================================================
# Rotary Position Embeddings (RoPE)
# ============================================================================


def precompute_freqs_cis(dim: int, max_len: int, theta: float = 10000.0) -> jax.Array:
    """Precompute rotary embedding frequencies.

    Args:
        dim: Head dimension (must be even)
        max_len: Maximum sequence length
        theta: Base for frequency computation

    Returns:
        Complex frequencies of shape (max_len, dim // 2)
    """
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2) / dim))
    t = jnp.arange(max_len)
    freqs = jnp.outer(t, freqs)  # (max_len, dim // 2)
    return jnp.exp(1j * freqs)


def apply_rotary_emb(
    x: jax.Array,
    freqs_cis: jax.Array,
) -> jax.Array:
    """Apply rotary embeddings to input tensor.

    Args:
        x: Input tensor of shape (..., seq_len, head_dim)
        freqs_cis: Precomputed frequencies of shape (seq_len, head_dim // 2)

    Returns:
        Tensor with rotary embeddings applied
    """
    # Reshape x to complex: (..., seq_len, head_dim // 2, 2) -> (..., seq_len, head_dim // 2)
    x_shape = x.shape
    x = x.reshape(*x_shape[:-1], -1, 2)
    x_complex = x[..., 0] + 1j * x[..., 1]

    # Broadcast freqs_cis to match x_complex shape
    # freqs_cis: (seq_len, head_dim // 2)
    # x_complex: (..., seq_len, head_dim // 2)
    freqs_cis = jnp.broadcast_to(freqs_cis, x_complex.shape)

    # Apply rotation
    x_rotated = x_complex * freqs_cis

    # Convert back to real
    x_out = jnp.stack([x_rotated.real, x_rotated.imag], axis=-1)
    return x_out.reshape(x_shape)


def precompute_freqs_cis_3d(
    dim: int,
    grid_size: tuple[int, int, int],
    theta: float = 10000.0,
) -> jax.Array:
    """Precompute 3D rotary embedding frequencies for video.

    Splits head dimension across temporal and spatial axes.

    Args:
        dim: Head dimension (must be even)
        grid_size: (T, H, W) grid dimensions
        theta: Base for frequency computation

    Returns:
        Complex frequencies of shape (T * H * W, dim // 2)
    """
    t, h, w = grid_size

    # Split dimensions: temporal gets 1/2, spatial each get 1/4
    dim_t = dim // 4
    dim_h = dim // 4
    dim_w = dim // 2 - dim_t - dim_h  # Handle remainder

    # Compute frequencies for each axis
    freqs_t = 1.0 / (theta ** (jnp.arange(0, dim_t * 2, 2) / (dim_t * 2)))
    freqs_h = 1.0 / (theta ** (jnp.arange(0, dim_h * 2, 2) / (dim_h * 2)))
    freqs_w = 1.0 / (theta ** (jnp.arange(0, dim_w * 2, 2) / (dim_w * 2)))

    # Create position grids
    grid_t = jnp.arange(t)
    grid_h = jnp.arange(h)
    grid_w = jnp.arange(w)

    # Compute outer products
    mesh_t, mesh_h, mesh_w = jnp.meshgrid(grid_t, grid_h, grid_w, indexing="ij")
    mesh_t = mesh_t.reshape(-1)
    mesh_h = mesh_h.reshape(-1)
    mesh_w = mesh_w.reshape(-1)

    # Compute frequencies for each position
    freqs_t = jnp.outer(mesh_t, freqs_t)  # (T*H*W, dim_t)
    freqs_h = jnp.outer(mesh_h, freqs_h)  # (T*H*W, dim_h)
    freqs_w = jnp.outer(mesh_w, freqs_w)  # (T*H*W, dim_w)

    # Concatenate and convert to complex
    freqs = jnp.concatenate([freqs_t, freqs_h, freqs_w], axis=-1)
    return jnp.exp(1j * freqs)


class WanRoPE3D:
    """3D rotary embeddings with WAN-style dimension split.

    Precomputes per-axis frequency tables at init.  Call with ``(f, h, w)``
    to assemble grid-specific complex frequencies.

    Dimension split (over complex pairs, i.e. ``d = head_dim // 2``):
        dim_h = d // 3
        dim_w = d // 3
        dim_f = d - 2 * (d // 3)   # absorbs any remainder
    """

    def __init__(self, head_dim: int, max_len: int = 1024, theta: float = 10000.0):
        self.head_dim = head_dim
        d = head_dim // 2  # number of complex pairs

        dim_h = d // 3
        dim_w = d // 3
        dim_f = d - 2 * dim_h
        self.dim_f = dim_f
        self.dim_h = dim_h
        self.dim_w = dim_w

        # Per-axis base frequencies — shape (dim_axis,)
        self._freqs_f = 1.0 / (theta ** (jnp.arange(0, dim_f * 2, 2, dtype=jnp.float32) / (dim_f * 2)))
        self._freqs_h = 1.0 / (theta ** (jnp.arange(0, dim_h * 2, 2, dtype=jnp.float32) / (dim_h * 2)))
        self._freqs_w = 1.0 / (theta ** (jnp.arange(0, dim_w * 2, 2, dtype=jnp.float32) / (dim_w * 2)))

        # Precompute position tables up to max_len — shape (max_len, dim_axis)
        positions = jnp.arange(max_len, dtype=jnp.float32)
        self._table_f = jnp.outer(positions, self._freqs_f)  # (max_len, dim_f)
        self._table_h = jnp.outer(positions, self._freqs_h)  # (max_len, dim_h)
        self._table_w = jnp.outer(positions, self._freqs_w)  # (max_len, dim_w)

    def __call__(self, f: int, h: int, w: int) -> jax.Array:
        """Assemble RoPE frequencies for a (f, h, w) grid.

        Returns:
            Complex array of shape ``(f * h * w, head_dim // 2)``.
        """
        # Slice precomputed tables to required lengths
        freqs_f = self._table_f[:f]  # (f, dim_f)
        freqs_h = self._table_h[:h]  # (h, dim_h)
        freqs_w = self._table_w[:w]  # (w, dim_w)

        # Broadcast each axis across the full grid using zero-copy views
        # Target: (f, h, w, dim_axis) for each, then flatten to (f*h*w, dim_axis)
        freqs_f = jnp.broadcast_to(freqs_f[:, None, None, :], (f, h, w, self.dim_f))
        freqs_h = jnp.broadcast_to(freqs_h[None, :, None, :], (f, h, w, self.dim_h))
        freqs_w = jnp.broadcast_to(freqs_w[None, None, :, :], (f, h, w, self.dim_w))

        # Flatten spatial dims and concatenate axis frequencies
        freqs_f = freqs_f.reshape(f * h * w, self.dim_f)
        freqs_h = freqs_h.reshape(f * h * w, self.dim_h)
        freqs_w = freqs_w.reshape(f * h * w, self.dim_w)

        freqs = jnp.concatenate([freqs_f, freqs_h, freqs_w], axis=-1)  # (f*h*w, d)
        return jnp.exp(1j * freqs)
