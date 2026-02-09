"""WAN Text Encoder (T5-style) for DreamZero.

Full re-implementation of the T5 encoder used in DreamZero, ported from
the original PyTorch code. Uses T5-style relative position embeddings,
gated feed-forward, and RMSNorm.

Layout: ``(B, L, C)`` for all intermediate tensors.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import nnx


# ---------------------------------------------------------------------------
# T5 Relative Position Embedding
# ---------------------------------------------------------------------------


class T5RelativeEmbedding(nnx.Module):
    """T5-style relative position bias.

    Computes a per-head attention bias from the relative distance between
    query and key positions, using learned bucketed embeddings.
    """

    def __init__(
        self,
        num_buckets: int,
        num_heads: int,
        bidirectional: bool = True,
        max_dist: int = 128,
        *,
        rngs: nnx.Rngs,
    ):
        self.num_buckets = num_buckets
        self.num_heads = num_heads
        self.bidirectional = bidirectional
        self.max_dist = max_dist

        std = (2 * num_buckets * num_heads) ** -0.5
        self.embedding = nnx.Embed(
            num_embeddings=num_buckets, features=num_heads, rngs=rngs,
        )
        # Re-init with the T5-specific std
        self.embedding.embedding[...] = (
            jax.random.normal(rngs.params(), (num_buckets, num_heads)) * std
        )

    def _relative_position_bucket(self, rel_pos: jax.Array) -> jax.Array:
        """Map relative positions to bucket indices."""
        if self.bidirectional:
            num_buckets = self.num_buckets // 2
            rel_buckets = (rel_pos > 0).astype(jnp.int32) * num_buckets
            rel_pos = jnp.abs(rel_pos)
        else:
            num_buckets = self.num_buckets
            rel_buckets = jnp.zeros_like(rel_pos, dtype=jnp.int32)
            rel_pos = -jnp.minimum(rel_pos, jnp.zeros_like(rel_pos))

        max_exact = num_buckets // 2
        is_small = rel_pos < max_exact

        rel_pos_large = max_exact + (
            jnp.log(rel_pos.astype(jnp.float32) / max_exact)
            / math.log(self.max_dist / max_exact)
            * (num_buckets - max_exact)
        ).astype(jnp.int32)
        rel_pos_large = jnp.minimum(rel_pos_large, num_buckets - 1)

        rel_buckets += jnp.where(is_small, rel_pos.astype(jnp.int32), rel_pos_large)
        return rel_buckets

    def __call__(self, lq: int, lk: int) -> jax.Array:
        """Compute relative position bias.

        Args:
            lq: Query sequence length.
            lk: Key sequence length.

        Returns:
            Bias of shape ``(1, num_heads, lq, lk)``.
        """
        rel_pos = jnp.arange(lk)[None, :] - jnp.arange(lq)[:, None]
        rel_pos = self._relative_position_bucket(rel_pos)
        embeds = self.embedding(rel_pos)  # (lq, lk, num_heads)
        return embeds.transpose(2, 0, 1)[None, ...]  # (1, num_heads, lq, lk)


# ---------------------------------------------------------------------------
# T5 Attention
# ---------------------------------------------------------------------------


class T5Attention(nnx.Module):
    """T5-style multi-head attention (no scaling, bias-based position encoding)."""

    def __init__(
        self,
        dim: int,
        dim_attn: int,
        num_heads: int,
        *,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.dim = dim
        self.dim_attn = dim_attn
        self.num_heads = num_heads
        self.head_dim = dim_attn // num_heads

        kw = dict(use_bias=False, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.q = nnx.Linear(dim, dim_attn, **kw)
        self.k = nnx.Linear(dim, dim_attn, **kw)
        self.v = nnx.Linear(dim, dim_attn, **kw)
        self.o = nnx.Linear(dim_attn, dim, **kw)

    def __call__(
        self,
        x: jax.Array,
        context: jax.Array | None = None,
        mask: jax.Array | None = None,
        pos_bias: jax.Array | None = None,
    ) -> jax.Array:
        """
        Args:
            x: ``(B, L1, C)``
            context: ``(B, L2, C)`` or None (self-attention).
            mask: ``(B, L2)`` or ``(B, L1, L2)`` or None.
            pos_bias: ``(1, num_heads, L1, L2)`` or None.

        Returns:
            ``(B, L1, C)``
        """
        context = x if context is None else context
        B, _, _ = x.shape
        n, c = self.num_heads, self.head_dim

        q = self.q(x).reshape(B, -1, n, c)        # (B, L1, n, c)
        k = self.k(context).reshape(B, -1, n, c)  # (B, L2, n, c)
        v = self.v(context).reshape(B, -1, n, c)

        # Build attention bias (T5 does NOT scale by 1/sqrt(d))
        attn_bias = jnp.zeros((B, n, q.shape[1], k.shape[1]), dtype=x.dtype)
        if pos_bias is not None:
            attn_bias = attn_bias + pos_bias
        if mask is not None:
            if mask.ndim == 2:
                # (B, L2) -> (B, 1, 1, L2)
                mask_expanded = mask[:, None, None, :]
            else:
                # (B, L1, L2) -> (B, 1, L1, L2)
                mask_expanded = mask[:, None, :, :]
            attn_bias = jnp.where(mask_expanded > 0, attn_bias, jnp.finfo(x.dtype).min)

        # Compute attention (no scaling — T5 convention)
        # q/k/v: (B, L, n, c) — need (B, n, L, c) for einsum
        attn = jnp.einsum("binc,bjnc->bnij", q, k) + attn_bias
        attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(x.dtype)
        out = jnp.einsum("bnij,bjnc->binc", attn, v)

        out = out.reshape(B, -1, n * c)
        return self.o(out)


# ---------------------------------------------------------------------------
# T5 Feed-Forward (Gated GELU)
# ---------------------------------------------------------------------------


class T5FeedForward(nnx.Module):
    """T5-style gated feed-forward with GELU activation."""

    def __init__(
        self,
        dim: int,
        dim_ffn: int,
        *,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        kw = dict(use_bias=False, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.gate = nnx.Linear(dim, dim_ffn, **kw)
        self.fc1 = nnx.Linear(dim, dim_ffn, **kw)
        self.fc2 = nnx.Linear(dim_ffn, dim, **kw)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.fc2(self.fc1(x) * jax.nn.gelu(self.gate(x), approximate=True))


# ---------------------------------------------------------------------------
# T5 Layer Norm (RMSNorm)
# ---------------------------------------------------------------------------

# We use nnx.RMSNorm directly — it matches T5LayerNorm behavior.

# ---------------------------------------------------------------------------
# T5 Self-Attention Block
# ---------------------------------------------------------------------------


class T5SelfAttention(nnx.Module):
    """T5 self-attention block: norm -> attn -> residual -> norm -> ffn -> residual."""

    def __init__(
        self,
        dim: int,
        dim_attn: int,
        dim_ffn: int,
        num_heads: int,
        num_buckets: int,
        shared_pos: bool = True,
        *,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.shared_pos = shared_pos

        self.norm1 = nnx.RMSNorm(dim, rngs=rngs)
        self.attn = T5Attention(
            dim, dim_attn, num_heads, dtype=dtype, param_dtype=param_dtype, rngs=rngs,
        )
        self.norm2 = nnx.RMSNorm(dim, rngs=rngs)
        self.ffn = T5FeedForward(dim, dim_ffn, dtype=dtype, param_dtype=param_dtype, rngs=rngs)

        if not shared_pos:
            self.pos_embedding = T5RelativeEmbedding(
                num_buckets, num_heads, bidirectional=True, rngs=rngs,
            )

    def __call__(
        self,
        x: jax.Array,
        mask: jax.Array | None = None,
        pos_bias: jax.Array | None = None,
    ) -> jax.Array:
        if self.shared_pos:
            e = pos_bias
        else:
            e = self.pos_embedding(x.shape[1], x.shape[1])
        x = x + self.attn(self.norm1(x), mask=mask, pos_bias=e)
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# WanTextEncoder — top-level module
# ---------------------------------------------------------------------------


class WanTextEncoder(nnx.Module):
    """T5-style text encoder used in DreamZero.

    Processes token IDs through learned embeddings, N transformer blocks
    with relative position bias, and a final RMSNorm.
    """

    def __init__(
        self,
        vocab: int = 256384,
        dim: int = 4096,
        dim_attn: int = 4096,
        dim_ffn: int = 10240,
        num_heads: int = 64,
        num_layers: int = 24,
        num_buckets: int = 32,
        shared_pos: bool = False,
        *,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.dim = dim
        self.num_layers = num_layers

        self.token_embedding = nnx.Embed(
            num_embeddings=vocab, features=dim, rngs=rngs,
        )
        self.shared_pos = shared_pos
        if shared_pos:
            self.pos_embedding = T5RelativeEmbedding(
                num_buckets, num_heads, bidirectional=True, rngs=rngs,
            )
        else:
            self.pos_embedding = None

        self.blocks = nnx.List([
            T5SelfAttention(
                dim, dim_attn, dim_ffn, num_heads, num_buckets, shared_pos,
                dtype=dtype, param_dtype=param_dtype, rngs=rngs,
            )
            for _ in range(num_layers)
        ])

        self.norm = nnx.RMSNorm(dim, rngs=rngs)

    def __call__(
        self,
        ids: jax.Array,
        mask: jax.Array | None = None,
    ) -> jax.Array:
        """Encode token IDs to contextualized embeddings.

        Args:
            ids: Token IDs ``(B, L)`` of int32.
            mask: Attention mask ``(B, L)`` where 1 = attend, 0 = pad.

        Returns:
            Embeddings ``(B, L, dim)``.
        """
        x = self.token_embedding(ids)

        if self.shared_pos:
            e = self.pos_embedding(x.shape[1], x.shape[1])
        else:
            e = None

        for block in self.blocks:
            x = block(x, mask=mask, pos_bias=e)

        x = self.norm(x)
        return x
