"""WAN Image Encoder (CLIP ViT-H/14) for DreamZero.

Provides visual observation encoding via a Vision Transformer. The model
outputs per-patch token features (not pooled) for cross-attention in the
DiT backbone. Uses ``use_31_block=True`` mode which returns features from
all layers except the last.

Layout: ``(B, L, C)`` for intermediate tensors, images are ``(B, H, W, C)``
channels-last to match JAX convention.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import nnx


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class QuickGELU(nnx.Module):
    """QuickGELU activation: x * sigmoid(1.702 * x)."""

    def __call__(self, x: jax.Array) -> jax.Array:
        return x * jax.nn.sigmoid(1.702 * x)


class VitSelfAttention(nnx.Module):
    """Self-attention with fused QKV projection."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        kw = dict(dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.to_qkv = nnx.Linear(dim, dim * 3, **kw)
        self.proj = nnx.Linear(dim, dim, **kw)

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Args:
            x: ``(B, L, C)``

        Returns:
            ``(B, L, C)``
        """
        B, L, C = x.shape
        n, d = self.num_heads, self.head_dim

        qkv = self.to_qkv(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        q = q.reshape(B, L, n, d)
        k = k.reshape(B, L, n, d)
        v = v.reshape(B, L, n, d)

        out = jax.nn.dot_product_attention(q, k, v)
        out = out.reshape(B, L, C)
        return self.proj(out)


class VitAttentionBlock(nnx.Module):
    """ViT transformer block with pre-norm or post-norm."""

    def __init__(
        self,
        dim: int,
        mlp_ratio: int,
        num_heads: int,
        post_norm: bool = False,
        activation: str = "gelu",
        eps: float = 1e-5,
        *,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.post_norm = post_norm

        kw = dict(dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.norm1 = nnx.LayerNorm(dim, epsilon=eps, rngs=rngs)
        self.attn = VitSelfAttention(dim, num_heads, **kw)
        self.norm2 = nnx.LayerNorm(dim, epsilon=eps, rngs=rngs)

        mid_dim = int(dim * mlp_ratio)
        if activation == "quick_gelu":
            act = QuickGELU()
        else:
            act = None  # we'll use jax.nn.gelu inline

        self.fc1 = nnx.Linear(dim, mid_dim, **kw)
        self.fc2 = nnx.Linear(mid_dim, dim, **kw)
        self._use_quick_gelu = activation == "quick_gelu"
        self._quick_gelu = act

    def _mlp(self, x: jax.Array) -> jax.Array:
        x = self.fc1(x)
        if self._use_quick_gelu:
            x = self._quick_gelu(x)
        else:
            x = jax.nn.gelu(x)
        x = self.fc2(x)
        return x

    def __call__(self, x: jax.Array) -> jax.Array:
        if self.post_norm:
            x = x + self.norm1(self.attn(x))
            x = x + self.norm2(self._mlp(x))
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self._mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Vision Transformer
# ---------------------------------------------------------------------------


class VisionTransformer(nnx.Module):
    """CLIP-style Vision Transformer.

    Processes images via patch embedding, positional embedding, N transformer
    blocks, and optional output projection.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 14,
        dim: int = 1280,
        mlp_ratio: int = 4,
        out_dim: int = 1024,
        num_heads: int = 16,
        num_layers: int = 32,
        pool_type: str = "token",
        pre_norm: bool = True,
        post_norm: bool = False,
        activation: str = "gelu",
        eps: float = 1e-5,
        *,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        assert pool_type in ("token", "token_fc"), f"Unsupported pool_type: {pool_type}"

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.dim = dim
        self.pool_type = pool_type

        gain = 1.0 / math.sqrt(dim)
        kw = dict(dtype=dtype, param_dtype=param_dtype, rngs=rngs)

        # Patch embedding — channels-last: (B, H, W, 3) -> (B, H', W', dim)
        self.patch_embedding = nnx.Conv(
            in_features=3,
            out_features=dim,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            use_bias=not pre_norm,
            **kw,
        )

        # CLS token
        self.cls_embedding = nnx.Param(
            gain * jax.random.normal(rngs.params(), (1, 1, dim))
        )

        # Positional embedding: 1 CLS + num_patches
        num_pos = self.num_patches + 1
        self.pos_embedding = nnx.Param(
            gain * jax.random.normal(rngs.params(), (1, num_pos, dim))
        )

        # Pre-norm (before transformer blocks)
        self.pre_norm = nnx.LayerNorm(dim, epsilon=eps, rngs=rngs) if pre_norm else None

        # Transformer blocks
        self.transformer = nnx.List([
            VitAttentionBlock(
                dim, mlp_ratio, num_heads,
                post_norm=post_norm, activation=activation, eps=eps, **kw,
            )
            for _ in range(num_layers)
        ])

        # Post-norm
        self.post_norm_layer = nnx.LayerNorm(dim, epsilon=eps, rngs=rngs)

        # Output head
        if pool_type == "token":
            self.head = nnx.Param(
                gain * jax.random.normal(rngs.params(), (dim, out_dim))
            )
        else:
            self.head = nnx.Linear(dim, out_dim, **kw)

    def __call__(
        self,
        x: jax.Array,
        use_31_block: bool = False,
    ) -> jax.Array:
        """Forward pass.

        Args:
            x: Images ``(B, H, W, 3)`` channels-last, float, normalized.
            use_31_block: If True, return features from all-but-last block
                          (used by DreamZero for cross-attention features).

        Returns:
            If ``use_31_block=True``: per-token features ``(B, 1 + num_patches, dim)``.
            Else: pooled output ``(B, out_dim)``.
        """
        B = x.shape[0]

        # Patch embed: (B, H, W, 3) -> (B, H', W', dim) -> (B, num_patches, dim)
        x = self.patch_embedding(x)
        x = x.reshape(B, -1, self.dim)

        # Prepend CLS token
        cls = jnp.broadcast_to(self.cls_embedding[...], (B, 1, self.dim))
        x = jnp.concatenate([cls, x], axis=1)  # (B, 1 + num_patches, dim)

        # Add positional embedding
        x = x + self.pos_embedding[...]

        if self.pre_norm is not None:
            x = self.pre_norm(x)

        # Transformer blocks
        if use_31_block:
            # Run all blocks except the last one
            for block in self.transformer[:-1]:
                x = block(x)
            return x
        else:
            for block in self.transformer:
                x = block(x)
            return x


# ---------------------------------------------------------------------------
# WanImageEncoder — top-level module
# ---------------------------------------------------------------------------


class WanImageEncoder(nnx.Module):
    """Image encoder for DreamZero, wrapping a CLIP ViT-H/14.

    Encodes observation images to per-token features for cross-attention
    in the DiT backbone. Uses ``use_31_block=True`` (all layers except the
    last) as in the original PyTorch implementation.

    Input images should be ``(B, H, W, 3)`` channels-last with pixel values
    in ``[-1, 1]``. Normalization to CLIP mean/std is applied internally.
    """

    # CLIP normalization constants
    MEAN = jnp.array([0.48145466, 0.4578275, 0.40821073])
    STD = jnp.array([0.26862954, 0.26130258, 0.27577711])

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 14,
        dim: int = 1280,
        mlp_ratio: int = 4,
        out_dim: int = 1024,
        num_heads: int = 16,
        num_layers: int = 32,
        activation: str = "gelu",
        *,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.visual = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            dim=dim,
            mlp_ratio=mlp_ratio,
            out_dim=out_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            pool_type="token",
            pre_norm=True,
            post_norm=False,
            activation=activation,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.image_size = image_size

    def encode_image(self, images: jax.Array) -> jax.Array:
        """Encode images to per-token features.

        Args:
            images: ``(B, H, W, 3)`` in ``[-1, 1]`` range, channels-last.

        Returns:
            ``(B, 1 + num_patches, dim)`` per-token features (CLS + patch tokens).
        """
        # Rescale from [-1, 1] to [0, 1]
        images = images * 0.5 + 0.5
        # Apply CLIP normalization
        images = (images - self.MEAN) / self.STD

        # Resize to model's expected size if needed
        B, H, W, C = images.shape
        if H != self.image_size or W != self.image_size:
            images = jax.image.resize(
                images,
                (B, self.image_size, self.image_size, C),
                method="bicubic",
            )

        return self.visual(images, use_31_block=True)
