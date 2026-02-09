"""Action head modules for DreamZero.

Contains multi-embodiment action encoders/decoders and the causal
action-aware DiT backbone (CausalWanDiT) for joint video + action
prediction.

Layout: ``(B, L, C)`` for all intermediate tensors.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from flax import nnx

from dreamzero_jax.nn.embed import (
    PatchEmbed3D,
    WanRoPE3D,
    sinusoidal_embedding,
)
from dreamzero_jax.nn.mlp import MLP
from dreamzero_jax.models.dit import (
    MLPProj,
    WanDiTBlock,
    WanDiTHead,
    unpatchify,
)

_gelu_approx = functools.partial(jax.nn.gelu, approximate=True)


# ---------------------------------------------------------------------------
# Category-specific (multi-embodiment) layers
# ---------------------------------------------------------------------------


class CategorySpecificLinear(nnx.Module):
    """Linear layer with per-category (embodiment) weight matrices.

    Stores ``num_categories`` separate ``(in, out)`` weight matrices and
    selects per-sample based on category IDs via index gathering.
    """

    def __init__(
        self,
        num_categories: int,
        input_dim: int,
        output_dim: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.weight = nnx.Param(
            jax.random.normal(rngs.params(), (num_categories, input_dim, output_dim)) * 0.02
        )
        self.bias = nnx.Param(jnp.zeros((num_categories, output_dim)))

    def __call__(self, x: jax.Array, category_ids: jax.Array) -> jax.Array:
        """
        Args:
            x: ``(B, L, in)`` or ``(B, in)``.
            category_ids: ``(B,)`` int.

        Returns:
            ``(B, L, out)`` or ``(B, out)``.
        """
        W = self.weight[...][category_ids]  # (B, in, out)
        b = self.bias[...][category_ids]    # (B, out)
        if x.ndim == 3:
            return jnp.einsum("bli,bio->blo", x, W) + b[:, None, :]
        return jnp.einsum("bi,bio->bo", x, W) + b


class CategorySpecificMLP(nnx.Module):
    """Two-layer MLP with per-category weights: SiLU(Linear1) -> Linear2."""

    def __init__(
        self,
        num_categories: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.linear1 = CategorySpecificLinear(
            num_categories, input_dim, hidden_dim, rngs=rngs,
        )
        self.linear2 = CategorySpecificLinear(
            num_categories, hidden_dim, output_dim, rngs=rngs,
        )

    def __call__(self, x: jax.Array, category_ids: jax.Array) -> jax.Array:
        return self.linear2(jax.nn.silu(self.linear1(x, category_ids)), category_ids)


class MultiEmbodimentActionEncoder(nnx.Module):
    """Encode noisy actions + diffusion timestep with embodiment-specific weights.

    ::

        a_emb = W1(actions, cat_ids)
        tau_emb = sinusoidal(timestep)
        x = SiLU(W2([a_emb || tau_emb], cat_ids))
        x = W3(x, cat_ids)
    """

    def __init__(
        self,
        action_dim: int,
        hidden_size: int,
        num_embodiments: int = 32,
        *,
        rngs: nnx.Rngs,
    ):
        self.hidden_size = hidden_size
        self.W1 = CategorySpecificLinear(
            num_embodiments, action_dim, hidden_size, rngs=rngs,
        )
        self.W2 = CategorySpecificLinear(
            num_embodiments, 2 * hidden_size, hidden_size, rngs=rngs,
        )
        self.W3 = CategorySpecificLinear(
            num_embodiments, hidden_size, hidden_size, rngs=rngs,
        )

    def __call__(
        self,
        actions: jax.Array,
        timesteps: jax.Array,
        category_ids: jax.Array,
    ) -> jax.Array:
        """
        Args:
            actions: ``(B, T, action_dim)`` noisy action sequence.
            timesteps: ``(B,)`` diffusion timestep.
            category_ids: ``(B,)`` embodiment IDs.

        Returns:
            ``(B, T, hidden_size)`` action embeddings.
        """
        B, T, _ = actions.shape
        a_emb = self.W1(actions, category_ids)  # (B, T, H)
        tau_emb = sinusoidal_embedding(timesteps, self.hidden_size)  # (B, H)
        tau_emb = jnp.broadcast_to(tau_emb[:, None, :], (B, T, self.hidden_size))
        x = jnp.concatenate([a_emb, tau_emb], axis=-1)  # (B, T, 2H)
        x = jax.nn.silu(self.W2(x, category_ids))  # (B, T, H)
        return self.W3(x, category_ids)  # (B, T, H)


# ---------------------------------------------------------------------------
# Block-causal mask for action-aware DiT
# ---------------------------------------------------------------------------


def make_action_causal_mask(
    num_blocks: int,
    block_video_tokens: int,
    num_action_per_block: int,
    num_state_per_block: int,
    has_clean: bool = True,
) -> jax.Array:
    """Build block-causal attention mask for joint video + action training.

    Sequence layout::

        [clean_video | noisy_video_0, action_0, state_0, ...,
                       noisy_video_N, action_N, state_N]

    Attention rules:

    * **clean_video_i** -> all clean_video_j where j <= i.
    * **noisy_video_i** -> clean_video_j (j <= i), own noisy_video_i,
      own action_i.
    * **action_i** -> clean_video_j (j <= i), own noisy_video_i, own action_i.
    * **state_i** -> own state_i only.

    Args:
        num_blocks: Number of temporal blocks.
        block_video_tokens: Video tokens per block
            (``frame_seqlen * num_frames_per_block``).
        num_action_per_block: Action tokens per block.
        num_state_per_block: State tokens per block.
        has_clean: Whether teacher-forcing clean video is prepended.

    Returns:
        Boolean mask ``(total_seq, total_seq)`` where True = attend.
    """
    TYPE_CLEAN = 0
    TYPE_VIDEO = 1
    TYPE_ACTION = 2
    TYPE_STATE = 3

    clean_len = num_blocks * block_video_tokens if has_clean else 0
    block_noisy = block_video_tokens + num_action_per_block + num_state_per_block

    # --- Build block-id and type-id arrays ---
    parts_block = []
    parts_type = []

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

    # --- Broadcast comparisons ---
    q_block = block_ids[:, None]
    k_block = block_ids[None, :]
    q_type = type_ids[:, None]
    k_type = type_ids[None, :]

    same_block = q_block == k_block
    k_earlier_or_same = q_block >= k_block

    is_clean_k = k_type == TYPE_CLEAN
    is_video_k = k_type == TYPE_VIDEO
    is_action_k = k_type == TYPE_ACTION
    is_state_k = k_type == TYPE_STATE

    # Clean video query: attend to earlier/same clean blocks
    clean_rule = is_clean_k & k_earlier_or_same

    # Noisy video / action query: attend to earlier clean + own video/action
    noisy_rule = (is_clean_k & k_earlier_or_same) | (
        (is_video_k | is_action_k) & same_block
    )

    # State query: attend to own state only
    state_rule = is_state_k & same_block

    is_clean_q = q_type == TYPE_CLEAN
    is_state_q = q_type == TYPE_STATE

    mask = jnp.where(
        is_clean_q,
        clean_rule,
        jnp.where(~is_state_q, noisy_rule, state_rule),
    )
    return mask


# ---------------------------------------------------------------------------
# CausalWanDiT — action-aware causal DiT backbone
# ---------------------------------------------------------------------------


class CausalWanDiT(nnx.Module):
    """Action-aware causal Diffusion Transformer.

    Extends the standard WanDiT with:

    * Multi-embodiment action encoder / decoder
    * State encoder
    * Block-causal attention for training with teacher forcing
    * Joint video + action noise prediction

    Training sequence layout::

        [clean_video | noisy_video_0 action_0 state_0 ...
                       noisy_video_N action_N state_N]

    Each temporal block ``i`` contains video tokens for one or more frames,
    the corresponding action tokens, and a state token.
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
        # Action-specific
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
        self.num_action_per_block = num_action_per_block
        self.num_state_per_block = num_state_per_block
        self.num_frames_per_block = num_frames_per_block
        head_dim = dim // num_heads

        # --- Shared with WanDiT ---
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
                1280, dim, dtype=dtype, param_dtype=param_dtype, rngs=rngs,
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

        # RoPE for video tokens (plain class, not nnx.Module)
        self.rope = WanRoPE3D(head_dim)

        # --- Action-specific ---
        self.state_encoder = CategorySpecificMLP(
            max_num_embodiments, state_dim, action_hidden_size, dim, rngs=rngs,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim, dim, max_num_embodiments, rngs=rngs,
        )
        self.action_decoder = CategorySpecificMLP(
            max_num_embodiments, dim, action_hidden_size, action_dim, rngs=rngs,
        )

    def _build_combined_rope(
        self,
        f: int,
        h: int,
        w: int,
        num_blocks: int,
        has_clean: bool,
    ) -> jax.Array:
        """Assemble RoPE frequencies for the combined interleaved sequence.

        Video tokens use 3D RoPE; action and state tokens use 1D RoPE
        indexed by block number.

        Returns:
            Complex array ``(total_seq, head_dim // 2)``.
        """
        video_freqs = self.rope(f, h, w)  # (f*h*w, head_dim//2) complex
        block_video_tokens = self.num_frames_per_block * h * w

        # 1D base frequencies for action/state tokens
        d = self.rope.head_dim // 2
        base_freqs = 1.0 / (
            10000.0 ** (jnp.arange(0, d * 2, 2, dtype=jnp.float32) / (d * 2))
        )

        parts = []
        if has_clean:
            parts.append(video_freqs)

        for i in range(num_blocks):
            # Video tokens: slice from 3D RoPE
            vs = i * block_video_tokens
            parts.append(video_freqs[vs : vs + block_video_tokens])

            # Action tokens: 1D RoPE at position i
            action_angles = jnp.outer(
                jnp.full(self.num_action_per_block, float(i)), base_freqs,
            )
            parts.append(jnp.exp(1j * action_angles))

            # State tokens: 1D RoPE at position i
            state_angles = jnp.outer(
                jnp.full(self.num_state_per_block, float(i)), base_freqs,
            )
            parts.append(jnp.exp(1j * state_angles))

        return jnp.concatenate(parts, axis=0)

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
    ) -> tuple[jax.Array, jax.Array]:
        """Joint video + action forward pass.

        Args:
            x: Noisy video latent ``(B, T, H, W, C)``.
            timestep: Video diffusion timestep ``(B,)``.
            context: Text embeddings ``(B, L, text_dim)``.
            state: Robot state ``(B, num_blocks, state_dim)``.
            embodiment_id: Embodiment IDs ``(B,)`` int.
            actions: Noisy actions
                ``(B, num_blocks * num_action_per_block, action_dim)``.
            timestep_action: Action diffusion timestep ``(B,)``.
                Defaults to ``timestep`` (coupled noise).
            clean_x: Clean video for teacher forcing ``(B, T, H, W, C)``
                or ``None``.
            clip_emb: CLIP image features ``(B, 257, 1280)`` or ``None``.

        Returns:
            ``(video_noise_pred, action_noise_pred)`` where
            ``video_noise_pred`` has shape ``(B, T, H, W, out_channels)`` and
            ``action_noise_pred`` has shape
            ``(B, total_actions, action_dim)``.
        """
        if timestep_action is None:
            timestep_action = timestep
        has_clean = clean_x is not None
        B = x.shape[0]

        # --- Patch embed noisy video ---
        x_patched = self.patch_embedding.proj(x)  # (B, f, h, w, dim)
        _, f, h, w, _ = x_patched.shape
        x_flat = x_patched.reshape(B, f * h * w, self.dim)

        block_video_tokens = self.num_frames_per_block * h * w
        num_blocks = f // self.num_frames_per_block
        block_noisy = (
            block_video_tokens + self.num_action_per_block + self.num_state_per_block
        )

        # --- Encode actions & state ---
        action_emb = self.action_encoder(
            actions, timestep_action, embodiment_id,
        )  # (B, total_actions, dim)
        state_emb = self.state_encoder(
            state, embodiment_id,
        )  # (B, num_blocks, dim)

        # --- Interleave noisy section ---
        noisy_parts: list[jax.Array] = []
        for i in range(num_blocks):
            vs = i * block_video_tokens
            noisy_parts.append(x_flat[:, vs : vs + block_video_tokens])
            as_ = i * self.num_action_per_block
            noisy_parts.append(
                action_emb[:, as_ : as_ + self.num_action_per_block]
            )
            noisy_parts.append(state_emb[:, i : i + 1])
        noisy_seq = jnp.concatenate(noisy_parts, axis=1)

        # --- Teacher forcing: prepend clean video ---
        if has_clean:
            clean_patched = self.patch_embedding.proj(clean_x)
            clean_flat = clean_patched.reshape(B, f * h * w, self.dim)
            full_seq = jnp.concatenate([clean_flat, noisy_seq], axis=1)
        else:
            full_seq = noisy_seq

        # --- Block-causal mask ---
        mask = make_action_causal_mask(
            num_blocks, block_video_tokens,
            self.num_action_per_block, self.num_state_per_block,
            has_clean=has_clean,
        )

        # --- RoPE ---
        freqs_cis = self._build_combined_rope(f, h, w, num_blocks, has_clean)

        # --- Time conditioning ---
        t = sinusoidal_embedding(timestep, self.freq_dim)
        t = self.time_embedding(t)
        e = jax.nn.silu(t)
        e = self.time_projection(e).reshape(B, 6, self.dim)

        # --- Text conditioning ---
        ctx = self.text_embedding(context)
        if self.has_image_input and clip_emb is not None:
            img_ctx = self.img_emb(clip_emb)
            ctx = jnp.concatenate([img_ctx, ctx], axis=1)

        # --- Transformer blocks ---
        for block in self.blocks:
            full_seq = block(full_seq, e, ctx, freqs_cis, mask=mask)

        # --- Extract video and action predictions ---
        clean_len = f * h * w if has_clean else 0
        noisy_section = full_seq[:, clean_len:]

        # Index arrays for video and action tokens within the noisy section
        block_offsets = jnp.arange(num_blocks) * block_noisy
        video_within = jnp.arange(block_video_tokens)
        video_indices = (block_offsets[:, None] + video_within[None, :]).ravel()

        action_within = jnp.arange(self.num_action_per_block) + block_video_tokens
        action_indices = (block_offsets[:, None] + action_within[None, :]).ravel()

        video_pred = noisy_section[:, video_indices]    # (B, f*h*w, dim)
        action_pred = noisy_section[:, action_indices]  # (B, total_actions, dim)

        # Video output head
        video_noise_pred = self.head(video_pred, t)
        video_noise_pred = unpatchify(
            video_noise_pred, (f, h, w), self.patch_size, self.out_channels,
        )

        # Action output
        action_noise_pred = self.action_decoder(action_pred, embodiment_id)

        return video_noise_pred, action_noise_pred
