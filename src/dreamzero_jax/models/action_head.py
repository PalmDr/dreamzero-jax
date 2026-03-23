"""Action head modules for DreamZero.

Contains multi-embodiment action encoders/decoders and the causal
action-aware DiT backbone (CausalWanDiT) for joint video + action
prediction.

Sequence layout (appended, matching original DreamZero)::

    [video_tokens | action_tokens | state_tokens]

With teacher forcing::

    [clean_video | noisy_video | action_tokens | state_tokens]
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from flax import nnx

from dreamzero_jax.nn.embed import (
    PatchEmbed3D,
    WanRoPE3D,
    rope_params_polar_1d,
    sinusoidal_embedding,
)
from dreamzero_jax.nn.mlp import MLP
from dreamzero_jax.models.dit import (
    MLPProj,
    WanDiTBlock,
    WanDiTHead,
    unpatchify,
)
from dreamzero_jax.models.causal_ops import (
    causal_block_forward,
    causal_remat_blocks,
    causal_scan_blocks,
    head_forward_per_token,
    per_token_time_conditioning,
)

_gelu_approx = functools.partial(jax.nn.gelu, approximate=True)


# ---------------------------------------------------------------------------
# Category-specific (multi-embodiment) layers
# ---------------------------------------------------------------------------


class CategorySpecificLinear(nnx.Module):
    """Linear layer with per-category (embodiment) weight matrices."""

    def __init__(
        self,
        num_categories: int,
        input_dim: int,
        output_dim: int,
        *,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.weight = nnx.Param(
            (jax.random.normal(rngs.params(), (num_categories, input_dim, output_dim)) * 0.02).astype(param_dtype)
        )
        self.bias = nnx.Param(jnp.zeros((num_categories, output_dim), dtype=param_dtype))

    def __call__(self, x: jax.Array, category_ids: jax.Array) -> jax.Array:
        W = self.weight[...][category_ids]
        b = self.bias[...][category_ids]
        if x.ndim == 3:
            return jnp.einsum("bli,bio->blo", x, W) + b[:, None, :]
        return jnp.einsum("bi,bio->bo", x, W) + b


class CategorySpecificMLP(nnx.Module):
    """Two-layer MLP with per-category weights: ReLU(Linear1) -> Linear2."""

    def __init__(
        self,
        num_categories: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.linear1 = CategorySpecificLinear(
            num_categories, input_dim, hidden_dim,
            param_dtype=param_dtype, rngs=rngs,
        )
        self.linear2 = CategorySpecificLinear(
            num_categories, hidden_dim, output_dim,
            param_dtype=param_dtype, rngs=rngs,
        )

    def __call__(self, x: jax.Array, category_ids: jax.Array) -> jax.Array:
        return self.linear2(jax.nn.relu(self.linear1(x, category_ids)), category_ids)


class MultiEmbodimentActionEncoder(nnx.Module):
    """Encode noisy actions + diffusion timestep with embodiment-specific weights."""

    def __init__(
        self,
        action_dim: int,
        hidden_size: int,
        num_embodiments: int = 32,
        *,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.hidden_size = hidden_size
        self.W1 = CategorySpecificLinear(
            num_embodiments, action_dim, hidden_size,
            param_dtype=param_dtype, rngs=rngs,
        )
        self.W2 = CategorySpecificLinear(
            num_embodiments, 2 * hidden_size, hidden_size,
            param_dtype=param_dtype, rngs=rngs,
        )
        self.W3 = CategorySpecificLinear(
            num_embodiments, hidden_size, hidden_size,
            param_dtype=param_dtype, rngs=rngs,
        )

    def __call__(
        self,
        actions: jax.Array,
        timesteps: jax.Array,
        category_ids: jax.Array,
    ) -> jax.Array:
        B, T, _ = actions.shape
        a_emb = self.W1(actions, category_ids)
        ts_expanded = jnp.broadcast_to(timesteps[:, None], (B, T))
        half = self.hidden_size // 2
        exponent = -jnp.log(10000.0) * jnp.arange(half) / half
        freqs = ts_expanded[..., None] * jnp.exp(exponent)
        tau_emb = jnp.concatenate([jnp.sin(freqs), jnp.cos(freqs)], axis=-1)
        tau_emb = tau_emb.astype(a_emb.dtype)
        x = jnp.concatenate([a_emb, tau_emb], axis=-1)
        x = jax.nn.silu(self.W2(x, category_ids))
        return self.W3(x, category_ids)


# ---------------------------------------------------------------------------
# Legacy mask (kept for backward compatibility with tests)
# ---------------------------------------------------------------------------


def make_action_causal_mask(
    num_blocks: int,
    block_video_tokens: int,
    num_action_per_block: int,
    num_state_per_block: int,
    has_clean: bool = True,
) -> jax.Array:
    """Build block-causal attention mask for the OLD interleaved layout.

    .. deprecated::
        CausalWanDiT now uses procedural blockwise attention instead of
        materialized masks.  Retained only for tests.
    """
    TYPE_CLEAN, TYPE_VIDEO, TYPE_ACTION, TYPE_STATE = 0, 1, 2, 3

    clean_len = num_blocks * block_video_tokens if has_clean else 0
    block_noisy = block_video_tokens + num_action_per_block + num_state_per_block

    parts_block: list[jax.Array] = []
    parts_type: list[jax.Array] = []

    if has_clean:
        parts_block.append(jnp.repeat(jnp.arange(num_blocks), block_video_tokens))
        parts_type.append(jnp.full(clean_len, TYPE_CLEAN, dtype=jnp.int32))

    block_type_pattern = jnp.concatenate([
        jnp.full(block_video_tokens, TYPE_VIDEO, dtype=jnp.int32),
        jnp.full(num_action_per_block, TYPE_ACTION, dtype=jnp.int32),
        jnp.full(num_state_per_block, TYPE_STATE, dtype=jnp.int32),
    ])
    parts_type.append(jnp.tile(block_type_pattern, num_blocks))
    parts_block.append(jnp.repeat(jnp.arange(num_blocks), block_noisy))

    block_ids = jnp.concatenate(parts_block)
    type_ids = jnp.concatenate(parts_type)

    q_block, k_block = block_ids[:, None], block_ids[None, :]
    q_type, k_type = type_ids[:, None], type_ids[None, :]

    same_block = q_block == k_block
    k_earlier_or_same = q_block >= k_block

    is_clean_k = k_type == TYPE_CLEAN
    is_video_k = k_type == TYPE_VIDEO
    is_action_k = k_type == TYPE_ACTION
    is_state_k = k_type == TYPE_STATE

    clean_rule = is_clean_k & k_earlier_or_same
    noisy_rule = (is_clean_k & k_earlier_or_same) | (
        (is_video_k | is_action_k) & same_block
    )
    state_rule = is_state_k & same_block

    return jnp.where(
        q_type == TYPE_CLEAN,
        clean_rule,
        jnp.where(q_type != TYPE_STATE, noisy_rule, state_rule),
    )


# ---------------------------------------------------------------------------
# CausalWanDiT — action-aware causal DiT backbone
# ---------------------------------------------------------------------------


class CausalWanDiT(nnx.Module):
    """Action-aware causal Diffusion Transformer.

    Sequence layout (appended, matching original DreamZero)::

        [video | action | state]

    With teacher forcing::

        [clean_video | noisy_video | action | state]

    Per-token time conditioning: each token gets its own timestep
    (t_video, t_action, or t_state; clean tokens get t=0).
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
        image_dim: int = 1280,
        qk_norm: bool = True,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        use_scan: bool = False,
        use_remat: bool = False,
        action_dim: int = 7,
        state_dim: int = 14,
        action_hidden_size: int = 1024,
        num_action_per_block: int = 32,
        num_state_per_block: int = 1,
        num_frames_per_block: int = 1,
        max_num_embodiments: int = 32,
        *,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.dim = dim
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.has_image_input = has_image_input
        self.use_scan = use_scan
        self.use_remat = use_remat
        self.num_action_per_block = num_action_per_block
        self.num_state_per_block = num_state_per_block
        self.num_frames_per_block = num_frames_per_block
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.patch_embedding = PatchEmbed3D(
            patch_size=patch_size, in_channels=in_channels, embed_dim=dim,
            dtype=dtype, param_dtype=param_dtype, rngs=rngs,
        )
        self.freq_dim = freq_dim
        self.time_embedding = MLP(
            freq_dim, dim, dim, activation=jax.nn.silu,
            dtype=dtype, param_dtype=param_dtype, rngs=rngs,
        )
        self.time_projection = nnx.Linear(
            dim, dim * 6, dtype=dtype, param_dtype=param_dtype, rngs=rngs,
        )
        self.text_embedding = MLP(
            text_dim, dim, dim, activation=_gelu_approx,
            dtype=dtype, param_dtype=param_dtype, rngs=rngs,
        )
        if has_image_input:
            self.img_emb = MLPProj(
                image_dim, dim, dtype=dtype, param_dtype=param_dtype, rngs=rngs,
            )

        self.blocks = nnx.List([
            WanDiTBlock(
                dim=dim, num_heads=num_heads, ffn_dim=ffn_dim,
                has_image_input=has_image_input, qk_norm=qk_norm,
                cross_attn_norm=cross_attn_norm, eps=eps,
                dtype=dtype, param_dtype=param_dtype, rngs=rngs,
            )
            for _ in range(num_layers)
        ])

        self.head = WanDiTHead(
            dim, out_channels, patch_size, eps=eps,
            dtype=dtype, param_dtype=param_dtype, rngs=rngs,
        )

        self.rope = WanRoPE3D(self.head_dim)

        self.state_encoder = CategorySpecificMLP(
            max_num_embodiments, state_dim, action_hidden_size, dim,
            param_dtype=param_dtype, rngs=rngs,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim, dim, max_num_embodiments,
            param_dtype=param_dtype, rngs=rngs,
        )
        self.action_decoder = CategorySpecificMLP(
            max_num_embodiments, dim, action_hidden_size, action_dim,
            param_dtype=param_dtype, rngs=rngs,
        )

    def __call__(
        self,
        x: jax.Array,
        timestep: jax.Array,
        context: jax.Array,
        state: jax.Array,
        embodiment_id: jax.Array,
        actions: jax.Array,
        timestep_action: jax.Array | None = None,
        clean_x: jax.Array | None = None,
        clip_emb: jax.Array | None = None,
        y: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Joint video + action forward pass."""
        if timestep_action is None:
            timestep_action = timestep
        has_clean = clean_x is not None
        B = x.shape[0]

        if y is not None:
            x = jnp.concatenate([x, y], axis=-1)

        x_patched = self.patch_embedding.proj(x)
        _, f, h, w, _ = x_patched.shape
        seq_len = f * h * w
        frame_seqlen = h * w
        x_flat = x_patched.reshape(B, seq_len, self.dim)

        video_freqs = self.rope(f, h, w)[:, None, :]
        action_freqs = rope_params_polar_1d(1024 * 10, self.head_dim)
        state_freqs = rope_params_polar_1d(1024, self.head_dim)

        action_emb = self.action_encoder(actions, timestep_action, embodiment_id)
        state_emb = self.state_encoder(state, embodiment_id)

        action_length = action_emb.shape[1]
        action_register = jnp.concatenate([action_emb, state_emb], axis=1)
        action_register_length = action_register.shape[1]

        x_seq = jnp.concatenate([x_flat, action_register], axis=1)

        if has_clean:
            if y is not None:
                clean_x = jnp.concatenate([clean_x, y], axis=-1)
            clean_patched = self.patch_embedding.proj(clean_x)
            clean_flat = clean_patched.reshape(B, seq_len, self.dim)
            full_seq = jnp.concatenate([clean_flat, x_seq], axis=1)
        else:
            full_seq = x_seq

        ts_video = jnp.broadcast_to(timestep[:, None], (B, seq_len))
        ts_action = (
            jnp.broadcast_to(timestep_action[:, None], (B, action_length))
            if timestep_action.ndim == 1
            else timestep_action
        )
        stride = ts_action.shape[1] // state_emb.shape[1]
        ts_state = ts_action[:, ::stride]
        ts_full = jnp.concatenate([ts_video, ts_action, ts_state], axis=1)

        if has_clean:
            ts_clean = jnp.zeros((B, seq_len), dtype=timestep.dtype)
            ts_full = jnp.concatenate([ts_clean, ts_full], axis=1)

        total_L = full_seq.shape[1]
        e_flat, e0_flat = per_token_time_conditioning(
            ts_full.reshape(-1), self.freq_dim,
            self.time_embedding, self.time_projection, self.dim,
        )
        e_tokens = e_flat.reshape(B, total_L, self.dim)
        e0_tokens = e0_flat.reshape(B, total_L, 6, self.dim)

        ctx = self.text_embedding(context)
        if self.has_image_input and clip_emb is not None:
            img_ctx = self.img_emb(clip_emb)
            ctx = jnp.concatenate([img_ctx, ctx], axis=1)

        is_tf = has_clean

        if self.use_scan:
            full_seq = causal_scan_blocks(
                list(self.blocks), full_seq, e0_tokens, ctx,
                video_freqs, action_freqs, state_freqs,
                action_register_length, frame_seqlen,
                self.num_frames_per_block, self.num_action_per_block,
                self.num_state_per_block, is_tf,
                use_remat=self.use_remat,
            )
        elif self.use_remat:
            full_seq = causal_remat_blocks(
                list(self.blocks), full_seq, e0_tokens, ctx,
                video_freqs, action_freqs, state_freqs,
                action_register_length, frame_seqlen,
                self.num_frames_per_block, self.num_action_per_block,
                self.num_state_per_block, is_tf,
            )
        else:
            for block in self.blocks:
                full_seq = causal_block_forward(
                    block, full_seq, e0_tokens, ctx,
                    video_freqs, action_freqs, state_freqs,
                    action_register_length, frame_seqlen,
                    self.num_frames_per_block, self.num_action_per_block,
                    self.num_state_per_block, is_tf,
                )

        if has_clean:
            full_seq = full_seq[:, seq_len:]

        video_pred = full_seq[:, :seq_len]
        action_pred = full_seq[:, seq_len:seq_len + action_length]

        offset = seq_len if has_clean else 0
        e_video = e_tokens[:, offset:offset + seq_len]
        video_noise_pred = head_forward_per_token(self.head, video_pred, e_video)
        video_noise_pred = unpatchify(
            video_noise_pred, (f, h, w), self.patch_size, self.out_channels,
        )

        action_noise_pred = self.action_decoder(action_pred, embodiment_id)

        return video_noise_pred, action_noise_pred
