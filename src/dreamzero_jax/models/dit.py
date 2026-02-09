"""WAN Diffusion Transformer backbone for DreamZero."""

from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp
from flax import nnx

from dreamzero_jax.nn.attention import Attention
from dreamzero_jax.nn.embed import (
    PatchEmbed3D,
    WanRoPE3D,
    sinusoidal_embedding,
)
from dreamzero_jax.nn.mlp import MLP

# Match PyTorch's GELU(approximate='tanh')
_gelu_approx = functools.partial(jax.nn.gelu, approximate=True)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def unpatchify(
    x: jax.Array,
    grid_size: tuple[int, int, int],
    patch_size: tuple[int, int, int],
    out_channels: int,
) -> jax.Array:
    """Reverse the patchify operation.

    Args:
        x: Flattened patch tokens ``(B, f*h*w, p_t*p_h*p_w*C)``.
        grid_size: ``(f, h, w)`` number of patches per axis.
        patch_size: ``(p_t, p_h, p_w)`` patch dimensions.
        out_channels: Number of output channels ``C``.

    Returns:
        Reconstructed volume ``(B, f*p_t, h*p_h, w*p_w, C)`` channels-last.
    """
    f, h, w = grid_size
    p_t, p_h, p_w = patch_size
    B = x.shape[0]

    # (B, f*h*w, p_t*p_h*p_w*C) -> (B, f, h, w, p_t, p_h, p_w, C)
    x = x.reshape(B, f, h, w, p_t, p_h, p_w, out_channels)

    # Interleave patch dims with grid dims:
    # (B, f, p_t, h, p_h, w, p_w, C) -> (B, f*p_t, h*p_h, w*p_w, C)
    x = x.transpose(0, 1, 4, 2, 5, 3, 6, 7)
    x = x.reshape(B, f * p_t, h * p_h, w * p_w, out_channels)
    return x


# ---------------------------------------------------------------------------
# MLPProj — image embedding projection for I2V
# ---------------------------------------------------------------------------


class MLPProj(nnx.Module):
    """Image embedding projection: LayerNorm -> Linear -> GELU -> Linear -> LayerNorm."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.norm_in = nnx.LayerNorm(in_dim, rngs=rngs)
        self.linear1 = nnx.Linear(
            in_dim, out_dim, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )
        self.linear2 = nnx.Linear(
            out_dim, out_dim, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )
        self.norm_out = nnx.LayerNorm(out_dim, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.norm_in(x)
        x = _gelu_approx(self.linear1(x))
        x = self.linear2(x)
        return self.norm_out(x)


# ---------------------------------------------------------------------------
# WanI2VCrossAttention — dual text + image cross-attention
# ---------------------------------------------------------------------------


class WanI2VCrossAttention(nnx.Module):
    """Cross-attention with separate K/V projections for image tokens.

    Expects ``context = [image_tokens || text_tokens]`` where the first
    ``num_image_tokens`` (default 257 for CLIP/SigLIP) are image tokens.

    Two parallel attention computations (shared Q), outputs summed before
    the shared output projection.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int | None = None,
        num_image_tokens: int = 257,
        qk_norm: bool = True,
        eps: float = 1e-6,
        *,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.num_heads = num_heads
        self.head_dim = head_dim or dim // num_heads
        self.num_image_tokens = num_image_tokens
        inner_dim = num_heads * self.head_dim

        # Shared Q projection
        self.q_proj = nnx.Linear(
            dim, inner_dim, use_bias=True, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )

        # Text K/V
        self.k_proj = nnx.Linear(
            dim, inner_dim, use_bias=True, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )
        self.v_proj = nnx.Linear(
            dim, inner_dim, use_bias=True, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )

        # Image K/V (separate projections)
        self.k_img = nnx.Linear(
            dim, inner_dim, use_bias=True, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )
        self.v_img = nnx.Linear(
            dim, inner_dim, use_bias=True, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )

        # Shared output projection
        self.out_proj = nnx.Linear(
            inner_dim, dim, use_bias=True, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )

        # QK norms
        self.qk_norm = qk_norm
        if qk_norm:
            self.norm_q = nnx.RMSNorm(inner_dim, epsilon=eps, rngs=rngs)
            self.norm_k_text = nnx.RMSNorm(inner_dim, epsilon=eps, rngs=rngs)
            self.norm_k_img = nnx.RMSNorm(inner_dim, epsilon=eps, rngs=rngs)

    def __call__(self, x: jax.Array, context: jax.Array) -> jax.Array:
        """
        Args:
            x: Query input ``(B, S, dim)``.
            context: ``(B, L, dim)`` where first ``num_image_tokens`` are image.

        Returns:
            ``(B, S, dim)``
        """
        B, S, _ = x.shape
        n_img = self.num_image_tokens

        img_ctx = context[:, :n_img]   # (B, n_img, dim)
        txt_ctx = context[:, n_img:]   # (B, L - n_img, dim)

        # Shared Q
        q = self.q_proj(x)
        if self.qk_norm:
            q = self.norm_q(q)
        q = q.reshape(B, S, self.num_heads, self.head_dim)

        # Text attention
        k_txt = self.k_proj(txt_ctx)
        v_txt = self.v_proj(txt_ctx)
        if self.qk_norm:
            k_txt = self.norm_k_text(k_txt)
        k_txt = k_txt.reshape(B, -1, self.num_heads, self.head_dim)
        v_txt = v_txt.reshape(B, -1, self.num_heads, self.head_dim)
        out_txt = jax.nn.dot_product_attention(q, k_txt, v_txt)

        # Image attention
        k_img = self.k_img(img_ctx)
        v_img = self.v_img(img_ctx)
        if self.qk_norm:
            k_img = self.norm_k_img(k_img)
        k_img = k_img.reshape(B, -1, self.num_heads, self.head_dim)
        v_img = v_img.reshape(B, -1, self.num_heads, self.head_dim)
        out_img = jax.nn.dot_product_attention(q, k_img, v_img)

        # Sum text + image attention, then project
        out = out_txt + out_img  # (B, S, num_heads, head_dim)
        out = out.reshape(B, S, self.num_heads * self.head_dim)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# WanDiTBlock — core transformer block with 6-parameter modulation
# ---------------------------------------------------------------------------


class WanDiTBlock(nnx.Module):
    """DiT transformer block with adaptive time modulation.

    Uses 6-parameter modulation: shift/scale/gate for self-attention and FFN.
    Cross-attention has no adaptive modulation (optional norm only).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        has_image_input: bool = False,
        qk_norm: bool = True,
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        *,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.has_image_input = has_image_input

        # Norms for self-attention and FFN (no affine — modulated externally)
        self.norm1 = nnx.LayerNorm(
            dim, use_bias=False, use_scale=False, rngs=rngs
        )
        self.norm2 = nnx.LayerNorm(
            dim, use_bias=False, use_scale=False, rngs=rngs
        )

        # Optional norm before cross-attention
        self.cross_attn_norm = cross_attn_norm
        if cross_attn_norm:
            self.norm3 = nnx.LayerNorm(dim, rngs=rngs)

        # Self-attention
        self.self_attn = Attention(
            dim, num_heads, qk_norm=qk_norm, eps=eps,
            dtype=dtype, param_dtype=param_dtype, rngs=rngs,
        )

        # Cross-attention
        if has_image_input:
            self.cross_attn = WanI2VCrossAttention(
                dim, num_heads, qk_norm=qk_norm, eps=eps,
                dtype=dtype, param_dtype=param_dtype, rngs=rngs,
            )
        else:
            self.cross_attn = Attention(
                dim, num_heads, qk_norm=qk_norm, eps=eps,
                dtype=dtype, param_dtype=param_dtype, rngs=rngs,
            )

        # FFN
        self.ffn = MLP(
            dim, ffn_dim, dim, activation=_gelu_approx,
            dtype=dtype, param_dtype=param_dtype, rngs=rngs,
        )

        # 6-parameter learnable modulation bias: (1, 6, dim)
        self.modulation = nnx.Param(
            jax.random.normal(rngs.params(), (1, 6, dim)) / math.sqrt(dim)
        )

    def __call__(
        self,
        x: jax.Array,
        e: jax.Array,
        context: jax.Array,
        freqs_cis: jax.Array,
        mask: jax.Array | None = None,
    ) -> jax.Array:
        """
        Args:
            x: Token sequence ``(B, S, dim)``.
            e: Time modulation ``(B, 6, dim)``.
            context: Cross-attention context ``(B, L, dim)``.
            freqs_cis: RoPE frequencies ``(S, head_dim // 2)`` complex.
            mask: Optional boolean self-attention mask ``(S, S)`` or
                ``(B, S, S)``.  True means attend.

        Returns:
            ``(B, S, dim)``
        """
        mod = self.modulation[...] + e  # (B, 6, dim)
        shift_msa = mod[:, 0]   # (B, dim)
        scale_msa = mod[:, 1]
        gate_msa = mod[:, 2]
        shift_mlp = mod[:, 3]
        scale_mlp = mod[:, 4]
        gate_mlp = mod[:, 5]

        # Self-attention: norm -> modulate -> attn -> gated residual
        h = self.norm1(x) * (1 + scale_msa[:, None, :]) + shift_msa[:, None, :]
        h = self.self_attn(h, freqs_cis=freqs_cis, mask=mask)
        x = x + h * gate_msa[:, None, :]

        # Cross-attention: optional norm -> attn -> residual (no modulation)
        h = self.norm3(x) if self.cross_attn_norm else x
        if self.has_image_input:
            x = x + self.cross_attn(h, context=context)
        else:
            x = x + self.cross_attn(h, context=context)

        # FFN: norm -> modulate -> ffn -> gated residual
        h = self.norm2(x) * (1 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]
        h = self.ffn(h)
        x = x + h * gate_mlp[:, None, :]

        return x


# ---------------------------------------------------------------------------
# WanDiTHead — output projection head
# ---------------------------------------------------------------------------


class WanDiTHead(nnx.Module):
    """Output projection: 2-parameter modulation -> LayerNorm -> Linear."""

    def __init__(
        self,
        dim: int,
        out_channels: int,
        patch_size: tuple[int, int, int],
        eps: float = 1e-6,
        *,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        patch_vol = patch_size[0] * patch_size[1] * patch_size[2]
        self.norm = nnx.LayerNorm(
            dim, use_bias=False, use_scale=False, epsilon=eps, rngs=rngs
        )
        self.linear = nnx.Linear(
            dim, patch_vol * out_channels,
            dtype=dtype, param_dtype=param_dtype, rngs=rngs,
        )
        # 2-parameter learnable modulation bias: (1, 2, dim)
        self.modulation = nnx.Param(
            jax.random.normal(rngs.params(), (1, 2, dim)) / math.sqrt(dim)
        )

    def __call__(self, x: jax.Array, t: jax.Array) -> jax.Array:
        """
        Args:
            x: ``(B, S, dim)``
            t: Raw time embedding ``(B, dim)``

        Returns:
            ``(B, S, patch_vol * out_channels)``
        """
        mod = self.modulation[...] + t[:, None, :]  # (B, 2, dim)
        shift = mod[:, 0]  # (B, dim)
        scale = mod[:, 1]

        x = self.norm(x) * (1 + scale[:, None, :]) + shift[:, None, :]
        return self.linear(x)


# ---------------------------------------------------------------------------
# WanDiT — full model
# ---------------------------------------------------------------------------


class WanDiT(nnx.Module):
    """WAN Diffusion Transformer backbone.

    Combines patch embedding, sinusoidal time embedding, text/image
    conditioning, N transformer blocks with RoPE, and an output head
    that unpatchifies back to video space.
    """

    def __init__(
        self,
        dim: int = 1536,
        in_channels: int = 16,
        out_channels: int = 16,
        ffn_dim: int = 8960,
        freq_dim: int = 256,
        text_dim: int = 4096,
        num_heads: int = 12,
        num_layers: int = 30,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        has_image_input: bool = False,
        qk_norm: bool = True,
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        *,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.dim = dim
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.has_image_input = has_image_input
        head_dim = dim // num_heads

        # Patch embedding (3D conv)
        self.patch_embedding = PatchEmbed3D(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=dim,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        # Time embeddings: sinusoidal -> MLP
        self.time_embedding = MLP(
            freq_dim, dim, dim, activation=jax.nn.silu,
            dtype=dtype, param_dtype=param_dtype, rngs=rngs,
        )
        self.freq_dim = freq_dim

        # Time -> 6-param modulation per block
        self.time_projection = nnx.Linear(
            dim, dim * 6, dtype=dtype, param_dtype=param_dtype, rngs=rngs,
        )

        # Text embedding projection
        self.text_embedding = MLP(
            text_dim, dim, dim, activation=_gelu_approx,
            dtype=dtype, param_dtype=param_dtype, rngs=rngs,
        )

        # Optional image embedding projection (for I2V)
        if has_image_input:
            self.img_emb = MLPProj(
                1280, dim, dtype=dtype, param_dtype=param_dtype, rngs=rngs
            )

        # Transformer blocks
        self.blocks = nnx.List([
            WanDiTBlock(
                dim=dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                has_image_input=has_image_input,
                qk_norm=qk_norm,
                cross_attn_norm=cross_attn_norm,
                eps=eps,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )
            for _ in range(num_layers)
        ])

        # Output head
        self.head = WanDiTHead(
            dim, out_channels, patch_size, eps=eps,
            dtype=dtype, param_dtype=param_dtype, rngs=rngs,
        )

        # RoPE (not an nnx.Module — no learnable params)
        self.rope = WanRoPE3D(head_dim)

    def __call__(
        self,
        x: jax.Array,
        timestep: jax.Array,
        context: jax.Array,
        clip_emb: jax.Array | None = None,
    ) -> jax.Array:
        """
        Args:
            x: Video latent ``(B, T, H, W, C)`` channels-last.
            timestep: Diffusion timestep ``(B,)``.
            context: Text embeddings ``(B, L, text_dim)``.
            clip_emb: Optional CLIP/SigLIP image embeddings ``(B, 257, 1280)``
                      for image-to-video generation.

        Returns:
            Noise prediction ``(B, T, H, W, out_channels)`` channels-last.
        """
        B = x.shape[0]

        # --- Time conditioning ---
        t = sinusoidal_embedding(timestep, self.freq_dim)  # (B, freq_dim)
        t = self.time_embedding(t)  # (B, dim)
        e = jax.nn.silu(t)
        e = self.time_projection(e)  # (B, dim * 6)
        e = e.reshape(B, 6, self.dim)  # (B, 6, dim)

        # --- Text conditioning ---
        ctx = self.text_embedding(context)  # (B, L, dim)

        # --- Optional image conditioning ---
        if self.has_image_input and clip_emb is not None:
            img_ctx = self.img_emb(clip_emb)  # (B, 257, dim)
            ctx = jnp.concatenate([img_ctx, ctx], axis=1)  # (B, 257 + L, dim)

        # --- Patchify ---
        # Use the 3D conv directly to get grid dims for RoPE and unpatchify
        x_patched = self.patch_embedding.proj(x)  # (B, f, h, w, dim)
        _, f, h, w, _ = x_patched.shape
        x_flat = x_patched.reshape(B, f * h * w, self.dim)  # (B, S, dim)

        # --- RoPE frequencies ---
        freqs_cis = self.rope(f, h, w)  # (S, head_dim // 2)

        # --- Transformer blocks ---
        for block in self.blocks:
            x_flat = block(x_flat, e, ctx, freqs_cis)

        # --- Output head ---
        x_out = self.head(x_flat, t)  # (B, S, patch_vol * out_channels)

        # --- Unpatchify ---
        return unpatchify(x_out, (f, h, w), self.patch_size, self.out_channels)
