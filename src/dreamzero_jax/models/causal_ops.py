"""Causal operation helpers for CausalWanDiT.

Contains RoPE application, blockwise self-attention wiring, per-token
block/head forward functions, and scan/remat variants.  All of these
are internal to the CausalWanDiT forward pass and not part of the
public API.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from flax import nnx

from dreamzero_jax.nn.attention import (
    Attention,
    blockwise_causal_attn,
    blockwise_causal_attn_tf,
)
from dreamzero_jax.nn.mlp import MLP
from dreamzero_jax.models.dit import WanDiTBlock, WanDiTHead
from dreamzero_jax.nn.embed import sinusoidal_embedding


# ---------------------------------------------------------------------------
# RoPE application (separate video + action + state freqs)
# ---------------------------------------------------------------------------


def apply_rope_causal(
    x: jax.Array,
    video_freqs: jax.Array,
    action_freqs: jax.Array,
    state_freqs: jax.Array,
    action_register_length: int | None,
    num_action_per_block: int = 32,
    num_state_per_block: int = 1,
) -> jax.Array:
    """Apply RoPE to ``(B, seq, num_heads, head_dim)`` with separate freq tables.

    Matches the original ``rope_action_apply_polar``: concatenates video,
    action, and state frequency tables then applies the rotation in
    complex128 (float64 precision).
    """
    B, seq_len, n, hd = x.shape
    x_f64 = x.astype(jnp.float64).reshape(B, seq_len, n, hd // 2, 2)
    x_complex = x_f64[..., 0] + 1j * x_f64[..., 1]

    freqs = video_freqs  # (video_len, 1, head_dim//2)

    if action_register_length is not None:
        num_blocks = action_register_length // (num_action_per_block + num_state_per_block)
        total_act = num_blocks * num_action_per_block
        total_state = num_blocks * num_state_per_block
        f1d_a = action_freqs[:total_act].reshape(total_act, 1, -1)
        f1d_s = state_freqs[:total_state].reshape(total_state, 1, -1)
        freqs = jnp.concatenate([freqs, f1d_a, f1d_s], axis=0)

    freqs = freqs[None, :, :, :]  # (1, seq, 1, head_dim//2)
    x_rotated = x_complex * freqs
    x_out = jnp.stack([x_rotated.real, x_rotated.imag], axis=-1)
    return x_out.reshape(B, seq_len, n, hd)


# ---------------------------------------------------------------------------
# Causal self-attention (RoPE inside, blockwise pattern)
# ---------------------------------------------------------------------------


def causal_self_attn(
    h: jax.Array,
    self_attn: Attention,
    video_freqs: jax.Array,
    action_freqs: jax.Array,
    state_freqs: jax.Array,
    action_register_length: int | None,
    frame_seqlen: int,
    num_frame_per_block: int,
    num_action_per_block: int,
    num_state_per_block: int,
    is_tf: bool,
) -> jax.Array:
    """Self-attention with blockwise causal pattern and internal RoPE.

    Reuses the projection weights and QK-norm from an existing
    ``Attention`` module, but applies RoPE and blockwise attention
    internally (matching the original CausalWanSelfAttention).
    """
    B, S, D = h.shape
    num_heads = self_attn.num_heads
    head_dim = self_attn.head_dim

    q = self_attn.q_proj(h)
    k = self_attn.k_proj(h)
    v = self_attn.v_proj(h)

    if self_attn.qk_norm:
        q = self_attn.norm_q(q)
        k = self_attn.norm_k(k)

    q = q.reshape(B, S, num_heads, head_dim)
    k = k.reshape(B, S, num_heads, head_dim)
    v = v.reshape(B, S, num_heads, head_dim)

    arl = action_register_length if action_register_length is not None else 0

    if is_tf:
        return _causal_self_attn_tf(
            q, k, v, video_freqs, action_freqs, state_freqs,
            action_register_length, frame_seqlen,
            num_frame_per_block, num_action_per_block,
            num_state_per_block, arl, self_attn,
        )

    return _causal_self_attn_infer(
        q, k, v, video_freqs, action_freqs, state_freqs,
        action_register_length, frame_seqlen,
        num_frame_per_block, num_action_per_block,
        num_state_per_block, self_attn,
    )


def _causal_self_attn_tf(
    q, k, v, video_freqs, action_freqs, state_freqs,
    action_register_length, frame_seqlen,
    num_frame_per_block, num_action_per_block,
    num_state_per_block, arl, self_attn,
):
    B, S = q.shape[0], q.shape[1]
    num_heads, head_dim = self_attn.num_heads, self_attn.head_dim
    half = (S - arl) // 2

    rq_ctx = apply_rope_causal(
        q[:, :half], video_freqs, action_freqs, state_freqs,
        action_register_length=None,
    ).astype(v.dtype)
    rk_ctx = apply_rope_causal(
        k[:, :half], video_freqs, action_freqs, state_freqs,
        action_register_length=None,
    ).astype(v.dtype)
    rq_noisy = apply_rope_causal(
        q[:, half:], video_freqs, action_freqs, state_freqs,
        action_register_length=action_register_length,
        num_action_per_block=num_action_per_block,
        num_state_per_block=num_state_per_block,
    ).astype(v.dtype)
    rk_noisy = apply_rope_causal(
        k[:, half:], video_freqs, action_freqs, state_freqs,
        action_register_length=action_register_length,
        num_action_per_block=num_action_per_block,
        num_state_per_block=num_state_per_block,
    ).astype(v.dtype)

    roped_q = jnp.concatenate([rq_ctx, rq_noisy], axis=1)
    roped_k = jnp.concatenate([rk_ctx, rk_noisy], axis=1)

    noisy_image_seq_len = half
    noisy_frames = noisy_image_seq_len // frame_seqlen
    num_blocks = (noisy_frames - 1) // num_frame_per_block
    action_horizon = num_blocks * num_action_per_block
    state_horizon = num_blocks * num_state_per_block

    x = blockwise_causal_attn_tf(
        roped_q, roped_k, v, frame_seqlen, num_frame_per_block,
        num_action_per_block, num_state_per_block,
        half, noisy_image_seq_len,
        action_horizon, state_horizon,
    )
    x = x.reshape(B, S, num_heads * head_dim)
    return self_attn.out_proj(x)


def _causal_self_attn_infer(
    q, k, v, video_freqs, action_freqs, state_freqs,
    action_register_length, frame_seqlen,
    num_frame_per_block, num_action_per_block,
    num_state_per_block, self_attn,
):
    B, S = q.shape[0], q.shape[1]
    num_heads, head_dim = self_attn.num_heads, self_attn.head_dim

    roped_q = apply_rope_causal(
        q, video_freqs, action_freqs, state_freqs,
        action_register_length=action_register_length,
        num_action_per_block=num_action_per_block,
        num_state_per_block=num_state_per_block,
    ).astype(v.dtype)
    roped_k = apply_rope_causal(
        k, video_freqs, action_freqs, state_freqs,
        action_register_length=action_register_length,
        num_action_per_block=num_action_per_block,
        num_state_per_block=num_state_per_block,
    ).astype(v.dtype)

    if action_register_length is not None:
        num_blocks = action_register_length // (num_action_per_block + num_state_per_block)
        action_horizon = num_blocks * num_action_per_block
        state_horizon = num_blocks * num_state_per_block
    else:
        action_horizon = 0
        state_horizon = 0

    x = blockwise_causal_attn(
        roped_q, roped_k, v, frame_seqlen, num_frame_per_block,
        action_horizon, state_horizon,
        num_action_per_block, num_state_per_block,
    )
    x = x.reshape(B, S, num_heads * head_dim)
    return self_attn.out_proj(x)


# ---------------------------------------------------------------------------
# Per-token causal block forward
# ---------------------------------------------------------------------------


def causal_block_forward(
    block: WanDiTBlock,
    x: jax.Array,
    e0: jax.Array,
    context: jax.Array,
    video_freqs: jax.Array,
    action_freqs: jax.Array,
    state_freqs: jax.Array,
    action_register_length: int | None,
    frame_seqlen: int,
    num_frame_per_block: int,
    num_action_per_block: int,
    num_state_per_block: int,
    is_tf: bool,
) -> jax.Array:
    """Run one WanDiTBlock with per-token modulation and causal attention.

    ``e0`` is ``(B, L, 6, D)`` per-token modulation parameters.
    ``block.modulation`` is ``(1, 6, D)`` and broadcasts to per-token.
    """
    # (1, 1, 6, D) + (B, L, 6, D) -> (B, L, 6, D)
    mod = block.modulation[...][None, :, :] + e0
    shift_msa = mod[:, :, 0, :]
    scale_msa = mod[:, :, 1, :]
    gate_msa = mod[:, :, 2, :]
    shift_mlp = mod[:, :, 3, :]
    scale_mlp = mod[:, :, 4, :]
    gate_mlp = mod[:, :, 5, :]

    h = block.norm1(x) * (1 + scale_msa) + shift_msa
    sa = causal_self_attn(
        h, block.self_attn, video_freqs, action_freqs, state_freqs,
        action_register_length, frame_seqlen,
        num_frame_per_block, num_action_per_block, num_state_per_block,
        is_tf,
    )
    x = x + sa * gate_msa

    h = block.norm3(x) if block.cross_attn_norm else x
    x = x + block.cross_attn(h, context=context)

    h = block.norm2(x) * (1 + scale_mlp) + shift_mlp
    h = block.ffn(h)
    x = x + h * gate_mlp

    return x


def causal_remat_blocks(
    blocks: list,
    x: jax.Array,
    e0: jax.Array,
    context: jax.Array,
    video_freqs: jax.Array,
    action_freqs: jax.Array,
    state_freqs: jax.Array,
    action_register_length: int | None,
    frame_seqlen: int,
    num_frame_per_block: int,
    num_action_per_block: int,
    num_state_per_block: int,
    is_tf: bool,
) -> jax.Array:
    """Run causal blocks with per-block activation checkpointing."""
    for block in blocks:
        graphdef, state = nnx.split(block)

        @functools.partial(
            jax.checkpoint,
            policy=jax.checkpoint_policies.nothing_saveable,
        )
        def _run(state, x_in):
            b = nnx.merge(graphdef, state)
            return causal_block_forward(
                b, x_in, e0, context, video_freqs, action_freqs,
                state_freqs, action_register_length, frame_seqlen,
                num_frame_per_block, num_action_per_block,
                num_state_per_block, is_tf,
            )

        x = _run(state, x)
    return x


def causal_scan_blocks(
    blocks: list,
    x: jax.Array,
    e0: jax.Array,
    context: jax.Array,
    video_freqs: jax.Array,
    action_freqs: jax.Array,
    state_freqs: jax.Array,
    action_register_length: int | None,
    frame_seqlen: int,
    num_frame_per_block: int,
    num_action_per_block: int,
    num_state_per_block: int,
    is_tf: bool,
    use_remat: bool = False,
) -> jax.Array:
    """Run causal blocks via ``jax.lax.scan`` over stacked parameters."""
    splits = [nnx.split(block) for block in blocks]
    graphdef = splits[0][0]
    all_states = [s for _, s in splits]
    stacked_state = jax.tree.map(lambda *leaves: jnp.stack(leaves), *all_states)

    input_dtype = x.dtype

    def _block_forward(layer_state, x_carry):
        block = nnx.merge(graphdef, layer_state)
        return causal_block_forward(
            block, x_carry, e0, context, video_freqs, action_freqs,
            state_freqs, action_register_length, frame_seqlen,
            num_frame_per_block, num_action_per_block,
            num_state_per_block, is_tf,
        )

    if use_remat:
        _block_forward = jax.checkpoint(
            _block_forward,
            policy=jax.checkpoint_policies.nothing_saveable,
        )

    def _body(carry, layer_state):
        x_carry = _block_forward(layer_state, carry)
        return x_carry.astype(input_dtype), None

    x_out, _ = jax.lax.scan(_body, x, stacked_state)
    return x_out


# ---------------------------------------------------------------------------
# Per-token head forward
# ---------------------------------------------------------------------------


def head_forward_per_token(
    head: WanDiTHead,
    x: jax.Array,
    e_video: jax.Array,
) -> jax.Array:
    """WanDiTHead with per-token modulation via ``e_video: (B, L, D)``.

    Original: ``mod_param(1,2,D).unsqueeze(1) + e.unsqueeze(2) -> (B,L,2,D)``
    """
    mod = head.modulation[...][None, :, :] + e_video[:, :, None, :]  # (B, L, 2, D)
    shift = mod[:, :, 0, :]
    scale = mod[:, :, 1, :]

    x = head.norm(x) * (1 + scale) + shift
    return head.linear(x)


# ---------------------------------------------------------------------------
# Per-token time conditioning
# ---------------------------------------------------------------------------


def per_token_time_conditioning(
    timestep_flat: jax.Array,
    freq_dim: int,
    time_embedding: MLP,
    time_projection: nnx.Linear,
    dim: int,
) -> tuple[jax.Array, jax.Array]:
    """Compute per-token time embeddings and modulation parameters.

    Args:
        timestep_flat: ``(B * L,)`` flattened per-token timesteps.

    Returns:
        ``(e, e0)`` where ``e`` is ``(B*L, D)`` raw embeddings and
        ``e0`` is ``(B*L, 6, D)`` modulation parameters.
    """
    sin_emb = sinusoidal_embedding(timestep_flat, freq_dim)
    e = time_embedding(sin_emb)
    e0 = jax.nn.silu(e)
    e0 = time_projection(e0).reshape(-1, 6, dim)
    return e, e0
