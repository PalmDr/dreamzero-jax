#!/usr/bin/env python3
"""Standalone PyTorch full CausalWanDiT forward pass.

NOTE: On macOS with multiple OpenMP runtimes, set:
    OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE

Loads weights from safetensors checkpoint and runs the complete
action-aware DiT forward (patch embed -> N blocks -> output head +
action decoder). No GR00T, no Hydra, no diffusers -- only torch,
numpy, safetensors.

Usage
-----
Full model (40 layers, needs all shards)::

    python scripts/pt_full_dit_forward.py \
        --checkpoint-dir /path/to/DreamZero-DROID \
        --num-layers 40 --output full_dit_outputs.npz

Quick 2-layer test on partial checkpoint::

    python scripts/pt_full_dit_forward.py \
        --checkpoint-dir ~/checkpoints/DreamZero-DROID-partial \
        --num-layers 2 --output test_dit_2layer.npz
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pt_standalone_ops import (
    DROID_DIM,
    DROID_FREQ_DIM,
    DROID_HEADS,
    DROID_HEAD_DIM,
    DROID_TEXT_DIM,
    SEED,
    attention_forward,
    gelu_approx,
    linear_forward,
    rmsnorm_forward,
    sinusoidal_embedding_pt,
)


# ---------------------------------------------------------------------------
# Selective weight loading (memory-efficient for partial checkpoints)
# ---------------------------------------------------------------------------


def load_selective_weights(ckpt_dir: Path, needed_prefixes: list[str]) -> dict[str, np.ndarray]:
    """Load only weights matching needed_prefixes from safetensors shards."""
    import json
    from safetensors import safe_open

    index_path = ckpt_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        key_to_shard = index["weight_map"]
    else:
        key_to_shard = {}
        for p in sorted(ckpt_dir.glob("*.safetensors")):
            with safe_open(str(p), framework="pt") as f:
                for k in f.keys():
                    key_to_shard[k] = p.name

    needed_keys: dict[str, str] = {}
    for key, shard in key_to_shard.items():
        for prefix in needed_prefixes:
            if key.startswith(prefix):
                needed_keys[key] = shard
                break

    shard_to_keys: dict[str, list[str]] = {}
    for key, shard in needed_keys.items():
        shard_to_keys.setdefault(shard, []).append(key)

    weights: dict[str, np.ndarray] = {}
    for shard_name in sorted(shard_to_keys.keys()):
        shard_path = ckpt_dir / shard_name
        if not shard_path.exists():
            print(f"  Skipping {shard_name} (not found, {len(shard_to_keys[shard_name])} keys)")
            continue
        keys_in_shard = shard_to_keys[shard_name]
        print(f"  Loading {shard_name} ({len(keys_in_shard)} keys)...")
        with safe_open(str(shard_path), framework="pt") as f:
            available = set(f.keys())
            for key in keys_in_shard:
                if key in available:
                    weights[key] = f.get_tensor(key).float().numpy()
    return weights


def compute_needed_prefixes(num_layers: int, has_i2v: bool) -> list[str]:
    """Return prefixes for all weights needed by the forward pass."""
    prefixes = [
        "action_head.model.patch_embedding.",
        "action_head.model.time_embedding.",
        "action_head.model.time_projection.",
        "action_head.model.text_embedding.",
        "action_head.model.action_encoder.",
        "action_head.model.action_decoder.",
        "action_head.model.state_encoder.",
        "action_head.model.head.",
    ]
    if has_i2v:
        prefixes.append("action_head.model.img_emb.")
    for i in range(num_layers):
        prefixes.append(f"action_head.model.blocks.{i}.")
    return prefixes


def strip_prefix(weights: dict[str, np.ndarray], prefix: str) -> dict[str, np.ndarray]:
    return {
        (k[len(prefix):] if k.startswith(prefix) else k): v
        for k, v in weights.items()
    }


def get_weight(weights: dict[str, np.ndarray], key: str):
    if key not in weights:
        raise KeyError(f"Weight '{key}' not found in checkpoint. "
                       f"The shard containing this weight may be missing.")
    val = weights[key]
    if isinstance(val, np.ndarray):
        return torch.from_numpy(val.copy()).float()
    return val.float()


# ---------------------------------------------------------------------------
# Model constants (DROID 14B / CausalWanDiT)
# ---------------------------------------------------------------------------

IN_CHANNELS = 16
OUT_CHANNELS = 16
I2V_CHANNELS = 20
PATCH_SIZE = (1, 2, 2)
IMAGE_DIM = 1280
NUM_IMAGE_TOKENS = 257
ACTION_DIM = 32
STATE_DIM = 64
ACTION_HIDDEN = 1024
NUM_ACTION_PER_BLOCK = 32
NUM_STATE_PER_BLOCK = 1
NUM_FRAMES_PER_BLOCK = 1


# ---------------------------------------------------------------------------
# Weight helpers
# ---------------------------------------------------------------------------


def get_weight_or_rand(weights, key, shape, label=""):
    """Load weight if present, else create random init and warn."""
    if key in weights:
        return get_weight(weights, key)
    print(f"  WARN: missing {key}{f' ({label})' if label else ''}, using random init {shape}")
    return torch.randn(shape) * 0.02


def has_weight(weights, key):
    return key in weights


# ---------------------------------------------------------------------------
# RoPE (3D + 1D for action/state)
# ---------------------------------------------------------------------------


def rope_params_polar(max_seq_len, dim, theta=10000.0):
    """Precompute 1D polar RoPE table. Returns complex (max_seq_len, dim//2)."""
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len, dtype=torch.float64),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2, dtype=torch.float64) / dim))
    return torch.polar(torch.ones_like(freqs), freqs)


def create_video_freqs(head_dim, f, h, w, start_frame=0):
    """Build 3D video RoPE using the ORIGINAL dimension split.

    Original uses 3 separate rope tables:
        dim_f = d - 4*(d//6)   (temporal, largest share)
        dim_h = 2*(d//6)       (height)
        dim_w = 2*(d//6)       (width)
    where d = head_dim (full head dim, not half).
    Each table stores complex values of shape (max_len, dim_axis//2).
    They are expanded over the (f,h,w) grid then concatenated.

    Returns complex (f*h*w, 1, head_dim//2) matching original shape.
    """
    d = head_dim
    dim_f = d - 4 * (d // 6)
    dim_h = 2 * (d // 6)
    dim_w = 2 * (d // 6)

    table_f = rope_params_polar(1024, dim_f)
    table_h = rope_params_polar(1024, dim_h)
    table_w = rope_params_polar(1024, dim_w)

    # Slice to grid sizes (temporal offset for inference)
    tf = table_f[start_frame:start_frame + f]  # (f, dim_f//2)
    th = table_h[:h]                             # (h, dim_h//2)
    tw = table_w[:w]                             # (w, dim_w//2)

    # Expand over grid: (f,h,w, dim_axis//2) then flatten to (f*h*w, dim_axis//2)
    tf = tf[:, None, None, :].expand(f, h, w, -1).reshape(f * h * w, -1)
    th = th[None, :, None, :].expand(f, h, w, -1).reshape(f * h * w, -1)
    tw = tw[None, None, :, :].expand(f, h, w, -1).reshape(f * h * w, -1)

    freqs = torch.cat([tf, th, tw], dim=-1)  # (f*h*w, head_dim//2)
    return freqs.unsqueeze(1)  # (f*h*w, 1, head_dim//2)


def rope_action_apply_polar(x, freqs, freqs_action, freqs_state,
                            action_register_length, num_action_per_block=32,
                            num_state_per_block=1):
    """Apply RoPE to (B, seq, num_heads, head_dim) tensor.

    Matches original ``rope_action_apply_polar``: concatenates 1D action/state
    freqs to the video freqs before applying to the full sequence.
    """
    B, seq_len, n, _ = x.shape
    x = torch.view_as_complex(
        x.to(torch.float64).reshape(B, seq_len, n, -1, 2))

    if action_register_length is not None:
        chunk_size = action_register_length // (num_action_per_block + num_state_per_block)
        f1d_a = freqs_action[:chunk_size * num_action_per_block].view(
            chunk_size * num_action_per_block, 1, -1)
        f1d_s = freqs_state[:chunk_size * num_state_per_block].view(
            chunk_size * num_state_per_block, 1, -1)
        freqs = torch.cat([freqs, f1d_a, f1d_s], dim=0)

    freqs = freqs.unsqueeze(0)  # (1, seq, 1, d//2)
    x = torch.view_as_real(x * freqs).flatten(3)
    return x


# ---------------------------------------------------------------------------
# Block-causal mask
# ---------------------------------------------------------------------------


def _blockwise_causal_attn(q, k, v, num_heads, frame_seqlen,
                           num_frame_per_block, action_horizon, state_horizon,
                           num_action_per_block, num_state_per_block):
    """Procedural block-wise causal attention matching the original.

    Sequence layout: [first_image | image_blocks | action_tokens | state_tokens]

    Attention rules (from original _blockwise_causal_flash_attn):
      - First image: self-attention only
      - Image block i: first_image + earlier/same image blocks + action[i] + state[i]
      - Action block i: first_image + earlier/same image blocks + action[i] + state[i]
      - State block i: self-attention only
    """
    b, total_len, n_h, d = q.shape
    head_dim = d

    first_image_len = frame_seqlen
    action_len = action_horizon
    state_len = state_horizon
    image_blocks_len = total_len - first_image_len - action_len - state_len
    num_image_blocks = image_blocks_len // (num_frame_per_block * frame_seqlen)

    first_image_start = 0
    first_image_end = first_image_len
    image_blocks_start = first_image_end
    image_blocks_end = image_blocks_start + image_blocks_len
    action_start = image_blocks_end
    action_end = action_start + action_len
    state_start = action_end

    output = torch.empty_like(q)

    def _attn(q_block, k_block, v_block):
        """Standard scaled dot-product attention on (B, L, H, D) tensors."""
        qt = q_block.transpose(1, 2).float()
        kt = k_block.transpose(1, 2).float()
        vt = v_block.transpose(1, 2).float()
        scale = head_dim ** -0.5
        scores = (qt @ kt.transpose(-2, -1)) * scale
        weights = torch.softmax(scores, dim=-1)
        out = weights @ vt
        return out.transpose(1, 2).to(q_block.dtype)

    # First image: self-attention only
    output[:, first_image_start:first_image_end] = _attn(
        q[:, first_image_start:first_image_end],
        k[:, first_image_start:first_image_end],
        v[:, first_image_start:first_image_end])

    block_size = num_frame_per_block * frame_seqlen
    for bi in range(num_image_blocks):
        blk_s = image_blocks_start + bi * block_size
        blk_e = image_blocks_start + (bi + 1) * block_size
        a_s = action_start + bi * num_action_per_block
        a_e = action_start + (bi + 1) * num_action_per_block
        s_s = state_start + bi * num_state_per_block
        s_e = state_start + (bi + 1) * num_state_per_block

        k_ctx = torch.cat([
            k[:, first_image_start:first_image_end],
            k[:, image_blocks_start:blk_e],
            k[:, a_s:a_e],
            k[:, s_s:s_e],
        ], dim=1)
        v_ctx = torch.cat([
            v[:, first_image_start:first_image_end],
            v[:, image_blocks_start:blk_e],
            v[:, a_s:a_e],
            v[:, s_s:s_e],
        ], dim=1)
        output[:, blk_s:blk_e] = _attn(q[:, blk_s:blk_e], k_ctx, v_ctx)

    for bi in range(num_image_blocks):
        a_s = action_start + bi * num_action_per_block
        a_e = action_start + (bi + 1) * num_action_per_block
        img_end = image_blocks_start + (bi + 1) * block_size
        s_s = state_start + bi * num_state_per_block
        s_e = state_start + (bi + 1) * num_state_per_block

        k_ctx = torch.cat([
            k[:, first_image_start:first_image_end],
            k[:, image_blocks_start:img_end],
            k[:, a_s:a_e],
            k[:, s_s:s_e],
        ], dim=1)
        v_ctx = torch.cat([
            v[:, first_image_start:first_image_end],
            v[:, image_blocks_start:img_end],
            v[:, a_s:a_e],
            v[:, s_s:s_e],
        ], dim=1)
        output[:, a_s:a_e] = _attn(q[:, a_s:a_e], k_ctx, v_ctx)

    for bi in range(state_horizon // num_state_per_block):
        s_s = state_start + bi * num_state_per_block
        s_e = state_start + (bi + 1) * num_state_per_block
        output[:, s_s:s_e] = _attn(
            q[:, s_s:s_e], k[:, s_s:s_e], v[:, s_s:s_e])

    return output


def _blockwise_causal_attn_tf(
    roped_query, roped_key, v, frame_seqlen, num_frame_per_block,
    num_action_per_block, num_state_per_block,
    clean_image_seq_len, noisy_image_seq_len,
    action_horizon, state_horizon):
    """Teacher-forcing attention: separate clean/noisy halves.

    Matches the original ``CausalWanSelfAttention`` teacher-forcing path.
    Layout of roped_query/roped_key/v:
      [clean_video | noisy_video | action_tokens | state_tokens]
    """
    head_dim = roped_query.shape[-1]

    def _attn(q_block, k_block, v_block):
        qt = q_block.transpose(1, 2).float()
        kt = k_block.transpose(1, 2).float()
        vt = v_block.transpose(1, 2).float()
        scale = head_dim ** -0.5
        scores = (qt @ kt.transpose(-2, -1)) * scale
        weights = torch.softmax(scores, dim=-1)
        out = weights @ vt
        return out.transpose(1, 2).to(q_block.dtype)

    def _causal_attn(q_block, k_block, v_block):
        """Causal (lower-triangular) attention."""
        qt = q_block.transpose(1, 2).float()
        kt = k_block.transpose(1, 2).float()
        vt = v_block.transpose(1, 2).float()
        scale = head_dim ** -0.5
        scores = (qt @ kt.transpose(-2, -1)) * scale
        L_q, L_k = scores.shape[-2], scores.shape[-1]
        causal = torch.tril(torch.ones(L_q, L_k, device=scores.device, dtype=torch.bool),
                            diagonal=L_k - L_q)
        scores = scores.masked_fill(~causal, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        out = weights @ vt
        return out.transpose(1, 2).to(q_block.dtype)

    half = clean_image_seq_len
    clean_q = roped_query[:, :half]
    clean_k = roped_key[:, :half]
    clean_v = v[:, :half]

    noisy_img_q = roped_query[:, half:half + noisy_image_seq_len]
    noisy_act_q = roped_query[:, half + noisy_image_seq_len:half + noisy_image_seq_len + action_horizon]
    noisy_state_q = roped_query[:, half + noisy_image_seq_len + action_horizon:]

    noisy_img_k = roped_key[:, half:half + noisy_image_seq_len]
    noisy_act_k = roped_key[:, half + noisy_image_seq_len:half + noisy_image_seq_len + action_horizon]
    noisy_state_k = roped_key[:, half + noisy_image_seq_len + action_horizon:]

    noisy_img_v = v[:, half:half + noisy_image_seq_len]
    noisy_act_v = v[:, half + noisy_image_seq_len:half + noisy_image_seq_len + action_horizon]
    noisy_state_v = v[:, half + noisy_image_seq_len + action_horizon:]

    # Clean image: blockwise causal (first frame self-attn, rest causal over all clean)
    clean_out = _causal_attn(clean_q, clean_k, clean_v)

    clean_frames = clean_image_seq_len // frame_seqlen
    noisy_frames = noisy_image_seq_len // frame_seqlen
    num_blocks = (noisy_frames - 1) // num_frame_per_block
    block_size = frame_seqlen * num_frame_per_block

    # Noisy image blocks
    noisy_img_out = torch.empty_like(noisy_img_q)
    noisy_img_out[:, :frame_seqlen] = _attn(
        noisy_img_q[:, :frame_seqlen],
        noisy_img_k[:, :frame_seqlen],
        noisy_img_v[:, :frame_seqlen])

    for bi in range(num_blocks):
        ns = frame_seqlen + bi * block_size
        ne = frame_seqlen + (bi + 1) * block_size
        clean_end = frame_seqlen + bi * block_size
        a_s = bi * num_action_per_block
        a_e = (bi + 1) * num_action_per_block
        s_s = bi * num_state_per_block
        s_e = (bi + 1) * num_state_per_block

        k_ctx = torch.cat([
            clean_k[:, :clean_end],
            noisy_img_k[:, ns:ne],
            noisy_act_k[:, a_s:a_e],
            noisy_state_k[:, s_s:s_e],
        ], dim=1)
        v_ctx = torch.cat([
            clean_v[:, :clean_end],
            noisy_img_v[:, ns:ne],
            noisy_act_v[:, a_s:a_e],
            noisy_state_v[:, s_s:s_e],
        ], dim=1)
        noisy_img_out[:, ns:ne] = _attn(noisy_img_q[:, ns:ne], k_ctx, v_ctx)

    # Noisy action blocks
    noisy_act_out = torch.empty_like(noisy_act_q)
    for bi in range(num_blocks):
        a_s = bi * num_action_per_block
        a_e = (bi + 1) * num_action_per_block
        clean_end = frame_seqlen + bi * block_size
        ni_s = frame_seqlen + bi * block_size
        ni_e = frame_seqlen + (bi + 1) * block_size
        s_s = bi * num_state_per_block
        s_e = (bi + 1) * num_state_per_block

        k_ctx = torch.cat([
            clean_k[:, :clean_end],
            noisy_img_k[:, ni_s:ni_e],
            noisy_act_k[:, a_s:a_e],
            noisy_state_k[:, s_s:s_e],
        ], dim=1)
        v_ctx = torch.cat([
            clean_v[:, :clean_end],
            noisy_img_v[:, ni_s:ni_e],
            noisy_act_v[:, a_s:a_e],
            noisy_state_v[:, s_s:s_e],
        ], dim=1)
        noisy_act_out[:, a_s:a_e] = _attn(noisy_act_q[:, a_s:a_e], k_ctx, v_ctx)

    # Noisy state blocks: self-attention only
    noisy_state_out = torch.empty_like(noisy_state_q)
    for bi in range(state_horizon // num_state_per_block):
        s_s = bi * num_state_per_block
        s_e = (bi + 1) * num_state_per_block
        noisy_state_out[:, s_s:s_e] = _attn(
            noisy_state_q[:, s_s:s_e],
            noisy_state_k[:, s_s:s_e],
            noisy_state_v[:, s_s:s_e])

    return torch.cat([clean_out, noisy_img_out, noisy_act_out, noisy_state_out], dim=1)


# ---------------------------------------------------------------------------
# Attention with RoPE + mask (for DiT blocks)
# ---------------------------------------------------------------------------


def dit_self_attn_original(
    h, weights, prefix, freqs, freqs_action, freqs_state,
    action_register_length, frame_seqlen, num_frame_per_block,
    num_action_per_block, num_state_per_block, is_tf):
    """Self-attention matching original CausalWanSelfAttention.forward.

    Applies QK-norm, computes Q/K/V, applies RoPE with proper
    clean/noisy splitting, then runs procedural block-wise attention.
    """
    B, S, D = h.shape
    num_heads = DROID_HEADS
    head_dim = D // num_heads

    q = rmsnorm_forward(
        linear_forward(h, get_weight(weights, f"{prefix}.q.weight"),
                       get_weight(weights, f"{prefix}.q.bias")),
        get_weight(weights, f"{prefix}.norm_q.weight"))
    k = rmsnorm_forward(
        linear_forward(h, get_weight(weights, f"{prefix}.k.weight"),
                       get_weight(weights, f"{prefix}.k.bias")),
        get_weight(weights, f"{prefix}.norm_k.weight"))
    v = linear_forward(h, get_weight(weights, f"{prefix}.v.weight"),
                       get_weight(weights, f"{prefix}.v.bias"))

    q = q.reshape(B, S, num_heads, head_dim)
    k = k.reshape(B, S, num_heads, head_dim)
    v = v.reshape(B, S, num_heads, head_dim)

    arl = action_register_length if action_register_length is not None else 0

    if is_tf:
        # Teacher forcing: split clean/noisy, apply rope separately
        half = (S - arl) // 2
        q_ctx = q[:, :half]
        k_ctx = k[:, :half]
        q_noisy = q[:, half:]
        k_noisy = k[:, half:]

        rq_ctx = rope_action_apply_polar(
            q_ctx, freqs, freqs_action, freqs_state,
            action_register_length=None).type_as(v)
        rk_ctx = rope_action_apply_polar(
            k_ctx, freqs, freqs_action, freqs_state,
            action_register_length=None).type_as(v)
        rq_noisy = rope_action_apply_polar(
            q_noisy, freqs, freqs_action, freqs_state,
            action_register_length=action_register_length,
            num_action_per_block=num_action_per_block,
            num_state_per_block=num_state_per_block).type_as(v)
        rk_noisy = rope_action_apply_polar(
            k_noisy, freqs, freqs_action, freqs_state,
            action_register_length=action_register_length,
            num_action_per_block=num_action_per_block,
            num_state_per_block=num_state_per_block).type_as(v)

        roped_q = torch.cat([rq_ctx, rq_noisy], dim=1)
        roped_k = torch.cat([rk_ctx, rk_noisy], dim=1)

        noisy_image_seq_len = half
        noisy_frames = noisy_image_seq_len // frame_seqlen
        num_blocks = (noisy_frames - 1) // num_frame_per_block
        action_horizon = num_blocks * num_action_per_block
        state_horizon = num_blocks * num_state_per_block

        x = _blockwise_causal_attn_tf(
            roped_q, roped_k, v, frame_seqlen, num_frame_per_block,
            num_action_per_block, num_state_per_block,
            half, noisy_image_seq_len,
            action_horizon, state_horizon)
    else:
        roped_q = rope_action_apply_polar(
            q, freqs, freqs_action, freqs_state,
            action_register_length=action_register_length,
            num_action_per_block=num_action_per_block,
            num_state_per_block=num_state_per_block).type_as(v)
        roped_k = rope_action_apply_polar(
            k, freqs, freqs_action, freqs_state,
            action_register_length=action_register_length,
            num_action_per_block=num_action_per_block,
            num_state_per_block=num_state_per_block).type_as(v)

        if action_register_length is not None:
            chunk_size = action_register_length // (num_action_per_block + num_state_per_block)
            action_horizon = chunk_size * num_action_per_block
            state_horizon = chunk_size * num_state_per_block
        else:
            action_horizon = 0
            state_horizon = 0

        x = _blockwise_causal_attn(
            roped_q, roped_k, v, num_heads, frame_seqlen,
            num_frame_per_block, action_horizon, state_horizon,
            num_action_per_block, num_state_per_block)

    x = x.flatten(2)
    out = linear_forward(x, get_weight(weights, f"{prefix}.o.weight"),
                         get_weight(weights, f"{prefix}.o.bias"))
    return out


# ---------------------------------------------------------------------------
# Cross attention (I2V dual: text + image)
# ---------------------------------------------------------------------------


def text_only_cross_attention(x, context, weights, prefix, num_heads):
    """Standard cross-attention (no image branch)."""
    q = rmsnorm_forward(
        linear_forward(x, get_weight(weights, f"{prefix}.q.weight"),
                       get_weight(weights, f"{prefix}.q.bias")),
        get_weight(weights, f"{prefix}.norm_q.weight"))
    k = rmsnorm_forward(
        linear_forward(context, get_weight(weights, f"{prefix}.k.weight"),
                       get_weight(weights, f"{prefix}.k.bias")),
        get_weight(weights, f"{prefix}.norm_k.weight"))
    v = linear_forward(context, get_weight(weights, f"{prefix}.v.weight"),
                       get_weight(weights, f"{prefix}.v.bias"))
    out = attention_forward(q, k, v, num_heads)
    return linear_forward(out, get_weight(weights, f"{prefix}.o.weight"),
                          get_weight(weights, f"{prefix}.o.bias"))


def i2v_cross_attention(x, context, weights, prefix, num_heads):
    """Dual text + image cross-attention (WanI2VCrossAttention)."""
    B, S, D = x.shape
    head_dim = D // num_heads
    n_img = NUM_IMAGE_TOKENS

    img_ctx = context[:, :n_img]
    txt_ctx = context[:, n_img:]

    q = linear_forward(x, get_weight(weights, f"{prefix}.q.weight"),
                       get_weight(weights, f"{prefix}.q.bias"))
    q = rmsnorm_forward(q, get_weight(weights, f"{prefix}.norm_q.weight"))
    q = q.reshape(B, S, num_heads, head_dim)

    k_txt = linear_forward(txt_ctx, get_weight(weights, f"{prefix}.k.weight"),
                           get_weight(weights, f"{prefix}.k.bias"))
    k_txt = rmsnorm_forward(k_txt, get_weight(weights, f"{prefix}.norm_k.weight"))
    v_txt = linear_forward(txt_ctx, get_weight(weights, f"{prefix}.v.weight"),
                           get_weight(weights, f"{prefix}.v.bias"))
    L_txt = txt_ctx.shape[1]
    k_txt = k_txt.reshape(B, L_txt, num_heads, head_dim)
    v_txt = v_txt.reshape(B, L_txt, num_heads, head_dim).to(q.dtype)

    out_txt = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k_txt.transpose(1, 2), v_txt.transpose(1, 2)
    ).transpose(1, 2)

    k_img = linear_forward(img_ctx, get_weight(weights, f"{prefix}.k_img.weight"),
                           get_weight(weights, f"{prefix}.k_img.bias"))
    k_img = rmsnorm_forward(k_img, get_weight(weights, f"{prefix}.norm_k_img.weight"))
    v_img = linear_forward(img_ctx, get_weight(weights, f"{prefix}.v_img.weight"),
                           get_weight(weights, f"{prefix}.v_img.bias"))
    k_img = k_img.reshape(B, n_img, num_heads, head_dim)
    v_img = v_img.reshape(B, n_img, num_heads, head_dim).to(q.dtype)

    out_img = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k_img.transpose(1, 2), v_img.transpose(1, 2)
    ).transpose(1, 2)

    out = (out_txt + out_img).reshape(B, S, D)
    return linear_forward(out, get_weight(weights, f"{prefix}.o.weight"),
                          get_weight(weights, f"{prefix}.o.bias"))


# ---------------------------------------------------------------------------
# Self-attention sub-block (with RoPE + mask)
# ---------------------------------------------------------------------------






# ---------------------------------------------------------------------------
# Single DiT block
# ---------------------------------------------------------------------------


def dit_block_forward(x, e0, context, weights, blk_prefix, freqs, freqs_action,
                      freqs_state, action_register_length, frame_seqlen,
                      use_i2v_ca, is_tf):
    """One CausalWanAttentionBlock with per-token modulation."""
    D = x.shape[-1]

    mod_param = get_weight(weights, f"{blk_prefix}.modulation")
    mod = mod_param.unsqueeze(1) + e0  # (1,1,6,D) + (B,L,6,D) -> (B,L,6,D)
    shift_msa = mod[:, :, 0, :]  # (B, L, D)
    scale_msa = mod[:, :, 1, :]
    gate_msa = mod[:, :, 2, :]
    shift_mlp = mod[:, :, 3, :]
    scale_mlp = mod[:, :, 4, :]
    gate_mlp = mod[:, :, 5, :]

    h = torch.layer_norm(x, [D]) * (1 + scale_msa) + shift_msa
    sa = dit_self_attn_original(
        h, weights, f"{blk_prefix}.self_attn", freqs, freqs_action, freqs_state,
        action_register_length, frame_seqlen,
        NUM_FRAMES_PER_BLOCK, NUM_ACTION_PER_BLOCK, NUM_STATE_PER_BLOCK, is_tf)
    x = x + sa * gate_msa

    n3_w = get_weight(weights, f"{blk_prefix}.norm3.weight")
    n3_b = get_weight(weights, f"{blk_prefix}.norm3.bias")
    h = torch.nn.functional.layer_norm(x, [D], n3_w, n3_b)
    if use_i2v_ca:
        ca = i2v_cross_attention(h, context, weights, f"{blk_prefix}.cross_attn", DROID_HEADS)
    else:
        ca = text_only_cross_attention(h, context, weights, f"{blk_prefix}.cross_attn", DROID_HEADS)
    x = x + ca

    h = torch.layer_norm(x, [D]) * (1 + scale_mlp) + shift_mlp
    h = gelu_approx(linear_forward(h, get_weight(weights, f"{blk_prefix}.ffn.0.weight"),
                                   get_weight(weights, f"{blk_prefix}.ffn.0.bias")))
    h = linear_forward(h, get_weight(weights, f"{blk_prefix}.ffn.2.weight"),
                       get_weight(weights, f"{blk_prefix}.ffn.2.bias"))
    x = x + h * gate_mlp

    return x


# ---------------------------------------------------------------------------
# Category-specific linear (multi-embodiment)
# ---------------------------------------------------------------------------


def category_linear(x, weights, prefix, cat_ids):
    """CategorySpecificLinear: gather per-category W,b and matmul."""
    W = get_weight(weights, f"{prefix}.W")
    b = get_weight(weights, f"{prefix}.b")
    W_sel = W[cat_ids]
    b_sel = b[cat_ids]
    if x.ndim == 3:
        return torch.einsum("bli,bio->blo", x, W_sel) + b_sel[:, None, :]
    return torch.einsum("bi,bio->bo", x, W_sel) + b_sel


# ---------------------------------------------------------------------------
# Patch embedding (Conv3d)
# ---------------------------------------------------------------------------


def patch_embed_forward(x, weights, prefix="model.patch_embedding"):
    """Conv3d patch embedding. x: (B, T, H, W, C) channels-last."""
    w = get_weight(weights, f"{prefix}.weight")
    b = get_weight(weights, f"{prefix}.bias")
    B, T, H, W, C = x.shape
    x_pt = x.permute(0, 4, 1, 2, 3)
    out = torch.nn.functional.conv3d(x_pt, w, b, stride=PATCH_SIZE)
    _, D, f, h, w_out = out.shape
    return out.permute(0, 2, 3, 4, 1), f, h, w_out


# ---------------------------------------------------------------------------
# Output head
# ---------------------------------------------------------------------------


def head_forward_per_token(x, e_video, weights, prefix="model.head"):
    """CausalHead with per-token modulation matching original.

    e_video: (B, L, D) raw per-token time embeddings for video tokens.
    Original does: modulation (1, 2, D) unsqueeze(1) + e.unsqueeze(2) (B, L, 1, D)
    -> chunk(2, dim=2) -> squeeze(2) -> per-token shift/scale.
    """
    D = x.shape[-1]
    mod_param = get_weight_or_rand(weights, f"{prefix}.modulation", (1, 2, D), "head.modulation")
    # (1, 1, 2, D) + (B, L, 1, D) -> (B, L, 2, D) -> chunk -> (B, L, 1, D) each
    mod = mod_param.unsqueeze(1) + e_video.unsqueeze(2)  # (B, L, 2, D)
    shift = mod[:, :, 0, :]  # (B, L, D)
    scale = mod[:, :, 1, :]

    x = torch.layer_norm(x, [D]) * (1 + scale) + shift
    w = get_weight_or_rand(weights, f"{prefix}.head.weight",
                           (PATCH_SIZE[0] * PATCH_SIZE[1] * PATCH_SIZE[2] * OUT_CHANNELS, D),
                           "head.head.weight")
    b = get_weight_or_rand(weights, f"{prefix}.head.bias",
                           (PATCH_SIZE[0] * PATCH_SIZE[1] * PATCH_SIZE[2] * OUT_CHANNELS,),
                           "head.head.bias")
    return linear_forward(x, w, b)


# ---------------------------------------------------------------------------
# Unpatchify
# ---------------------------------------------------------------------------


def unpatchify_pt(x, grid_size, patch_size, out_channels):
    """(B, f*h*w, p_t*p_h*p_w*C) -> (B, f*p_t, h*p_h, w*p_w, C)."""
    f, h, w = grid_size
    p_t, p_h, p_w = patch_size
    B = x.shape[0]
    x = x.reshape(B, f, h, w, p_t, p_h, p_w, out_channels)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7)
    return x.reshape(B, f * p_t, h * p_h, w * p_w, out_channels)


# ---------------------------------------------------------------------------
# MLPProj (image embedding projection)
# ---------------------------------------------------------------------------


def img_emb_forward(x, weights, prefix="model.img_emb.proj"):
    """MLPProj: LayerNorm -> Linear -> GELU -> Linear -> LayerNorm."""
    D_in = x.shape[-1]
    w0 = get_weight(weights, f"{prefix}.0.weight")
    b0 = get_weight(weights, f"{prefix}.0.bias")
    x = torch.nn.functional.layer_norm(x, [D_in], w0, b0)
    w1 = get_weight(weights, f"{prefix}.1.weight")
    b1 = get_weight(weights, f"{prefix}.1.bias")
    x = torch.nn.functional.gelu(linear_forward(x, w1, b1))
    w3 = get_weight(weights, f"{prefix}.3.weight")
    b3 = get_weight(weights, f"{prefix}.3.bias")
    x = linear_forward(x, w3, b3)
    D_out = x.shape[-1]
    w4 = get_weight(weights, f"{prefix}.4.weight")
    b4 = get_weight(weights, f"{prefix}.4.bias")
    return torch.nn.functional.layer_norm(x, [D_out], w4, b4)


# ---------------------------------------------------------------------------
# Action encoder
# ---------------------------------------------------------------------------


def _sinusoidal_pos_enc(timesteps, dim):
    """SinusoidalPositionalEncoding matching the original: [sin, cos] order.

    Args:
        timesteps: (B, T) float tensor
        dim: embedding dimension

    Returns:
        (B, T, dim) with [sin(freqs), cos(freqs)] concatenated
    """
    half_dim = dim // 2
    exponent = -torch.arange(half_dim, dtype=torch.float, device=timesteps.device) * (
        math.log(10000.0) / half_dim)
    freqs = timesteps.unsqueeze(-1).float() * exponent.exp()
    return torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)


def action_encoder_forward(actions, timestep_action, cat_ids, weights, prefix="model.action_encoder"):
    """MultiEmbodimentActionEncoder forward matching original.

    Key differences from old code:
    - Uses SinusoidalPositionalEncoding: [sin, cos] order (not [cos, sin])
    - Timestep expanded to (B, T) before encoding
    - Uses swish (=silu) activation
    """
    B, T, _ = actions.shape
    a_emb = category_linear(actions, weights, f"{prefix}.W1", cat_ids)
    # Expand scalar timestep to (B, T) then compute sinusoidal encoding
    ts_expanded = timestep_action[:, None].expand(B, T)
    tau_emb = _sinusoidal_pos_enc(ts_expanded, DROID_DIM).to(dtype=a_emb.dtype)
    x = torch.cat([a_emb, tau_emb], dim=-1)
    x = torch.nn.functional.silu(category_linear(x, weights, f"{prefix}.W2", cat_ids))
    return category_linear(x, weights, f"{prefix}.W3", cat_ids)


# ---------------------------------------------------------------------------
# State encoder
# ---------------------------------------------------------------------------


def state_encoder_forward(state, cat_ids, weights, prefix="model.state_encoder"):
    """CategorySpecificMLP: ReLU(Linear1) -> Linear2 (original uses F.relu)."""
    h = torch.nn.functional.relu(category_linear(state, weights, f"{prefix}.layer1", cat_ids))
    return category_linear(h, weights, f"{prefix}.layer2", cat_ids)


# ---------------------------------------------------------------------------
# Action decoder
# ---------------------------------------------------------------------------


def action_decoder_forward(x, cat_ids, weights, prefix="model.action_decoder"):
    """CategorySpecificMLP: ReLU(Linear1) -> Linear2 (original uses F.relu)."""
    h = torch.nn.functional.relu(category_linear(x, weights, f"{prefix}.layer1", cat_ids))
    return category_linear(h, weights, f"{prefix}.layer2", cat_ids)


# ---------------------------------------------------------------------------
# Time conditioning
# ---------------------------------------------------------------------------


def time_conditioning_per_token(timestep_flat, weights):
    """Per-token time conditioning matching original.

    Args:
        timestep_flat: (B * L,) flattened timestep values (one per token)
        weights: model weights dict

    Returns:
        (e, e0) where:
            e = (B*L, D) raw time embeddings
            e0 = (B*L, 6, D) modulation parameters
    """
    sin_emb = sinusoidal_embedding_1d_original(DROID_FREQ_DIM, timestep_flat)
    sin_emb = sin_emb.float()
    h = torch.nn.functional.silu(
        linear_forward(sin_emb, get_weight(weights, "model.time_embedding.0.weight"),
                       get_weight(weights, "model.time_embedding.0.bias")))
    e = linear_forward(h, get_weight(weights, "model.time_embedding.2.weight"),
                       get_weight(weights, "model.time_embedding.2.bias"))
    e0 = torch.nn.functional.silu(e)
    e0 = linear_forward(e0, get_weight(weights, "model.time_projection.1.weight"),
                        get_weight(weights, "model.time_projection.1.bias"))
    e0 = e0.reshape(-1, 6, DROID_DIM)
    return e, e0


def sinusoidal_embedding_1d_original(dim, position):
    """Matches original sinusoidal_embedding_1d: [cos, sin] order, float64."""
    assert dim % 2 == 0
    half = dim // 2
    position = position.to(torch.float64)
    sinusoid = torch.outer(
        position,
        torch.pow(10000.0, -torch.arange(half, dtype=torch.float64, device=position.device) / half))
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)


# ---------------------------------------------------------------------------
# Text conditioning
# ---------------------------------------------------------------------------


def text_conditioning(context, weights):
    """Text embedding: GELU(Linear) -> Linear."""
    h = gelu_approx(linear_forward(context,
                                   get_weight(weights, "model.text_embedding.0.weight"),
                                   get_weight(weights, "model.text_embedding.0.bias")))
    return linear_forward(h, get_weight(weights, "model.text_embedding.2.weight"),
                          get_weight(weights, "model.text_embedding.2.bias"))


# ---------------------------------------------------------------------------
# Full CausalWanDiT forward
# ---------------------------------------------------------------------------


def _nan_check(label, tensor):
    """Print NaN diagnostic for a tensor."""
    n = torch.isnan(tensor).sum().item()
    if n > 0:
        print(f"  NaN CHECK [{label}]: {n}/{tensor.numel()} NaN  "
              f"shape={tuple(tensor.shape)}")
    else:
        print(f"  NaN CHECK [{label}]: OK  "
              f"mean={tensor.mean():.4f} std={tensor.std():.4f}")


def causal_wan_dit_forward(
    x, timestep, context, state, embodiment_id, actions,
    timestep_action, clean_x, clip_emb, y, weights, num_layers,
):
    """Full CausalWanDiT forward pass matching the ORIGINAL DreamZero.

    Sequence layout: [video | action | state] (appended, not interleaved).
    With teacher forcing: [clean_video | noisy_video | action | state].
    Per-token time conditioning.

    Returns (video_noise_pred, action_noise_pred, intermediates_dict).
    """
    intermediates = {}
    has_clean = clean_x is not None
    B = x.shape[0]

    if y is not None:
        x = torch.cat([x, y], dim=-1)

    x_patched, f, h, w = patch_embed_forward(x, weights)
    seq_len = f * h * w
    frame_seqlen = h * w
    x_flat = x_patched.reshape(B, seq_len, DROID_DIM)
    intermediates["inter_patched_video"] = x_flat.numpy().copy()

    freqs = create_video_freqs(DROID_HEAD_DIM, f, h, w)
    freqs_action = rope_params_polar(1024 * 10, DROID_HEAD_DIM)
    freqs_state = rope_params_polar(1024, DROID_HEAD_DIM)

    action_emb = action_encoder_forward(actions, timestep_action, embodiment_id, weights)
    _nan_check("action_emb", action_emb)
    state_emb = state_encoder_forward(state, embodiment_id, weights)
    _nan_check("state_emb", state_emb)

    action_register = torch.cat([action_emb, state_emb], dim=1)
    action_length = action_emb.shape[1]
    action_register_length = action_register.shape[1]
    x_seq = torch.cat([x_flat, action_register], dim=1)
    _nan_check("x_seq (video+action+state)", x_seq)

    if has_clean:
        if y is not None:
            clean_x = torch.cat([clean_x, y], dim=-1)
        clean_patched, _, _, _ = patch_embed_forward(clean_x, weights)
        clean_flat = clean_patched.reshape(B, seq_len, DROID_DIM)
        full_seq = torch.cat([clean_flat, x_seq], dim=1)
    else:
        full_seq = x_seq

    ts_video = timestep[:, None].expand(B, seq_len)
    if timestep_action.ndim == 1:
        ts_action = timestep_action[:, None].expand(B, action_length)
    else:
        ts_action = timestep_action
    stride = ts_action.shape[1] // state_emb.shape[1]
    ts_state = ts_action[:, ::stride]
    ts_full = torch.cat([ts_video, ts_action, ts_state], dim=1)

    if has_clean:
        ts_clean = torch.zeros(B, seq_len, device=timestep.device, dtype=timestep.dtype)
        ts_full = torch.cat([ts_clean, ts_full], dim=1)

    e_flat, e0_flat = time_conditioning_per_token(ts_full.flatten(), weights)
    total_L = full_seq.shape[1]
    e_tokens = e_flat.reshape(B, total_L, DROID_DIM)
    e0_tokens = e0_flat.reshape(B, total_L, 6, DROID_DIM)
    intermediates["inter_e_tokens"] = e_tokens.numpy().copy()

    ctx = text_conditioning(context, weights)
    has_img_weights = has_weight(weights, "model.img_emb.proj.0.weight")
    if clip_emb is not None and has_img_weights:
        ctx = torch.cat([img_emb_forward(clip_emb, weights), ctx], dim=1)

    intermediates["inter_seq_pre_block"] = full_seq.numpy().copy()

    is_tf = has_clean

    for i in range(num_layers):
        print(f"    block {i}/{num_layers}...", end="", flush=True)
        t0 = time.time()
        full_seq = dit_block_forward(
            full_seq, e0_tokens, ctx, weights, f"model.blocks.{i}",
            freqs, freqs_action, freqs_state,
            action_register_length, frame_seqlen,
            use_i2v_ca=has_img_weights, is_tf=is_tf)
        print(f" {time.time() - t0:.1f}s")
        if i == 0:
            intermediates["inter_seq_post_block0"] = full_seq.numpy().copy()
        offset_blk = seq_len if has_clean else 0
        act_slice = full_seq[:, offset_blk + seq_len:offset_blk + seq_len + action_length]
        _nan_check(f"block {i} action tokens", act_slice)

    if has_clean:
        full_seq = full_seq[:, seq_len:]

    video_pred = full_seq[:, :seq_len]
    action_pred = full_seq[:, seq_len:seq_len + action_length]
    _nan_check("action_pred (extracted)", action_pred)
    intermediates["inter_action_pre_decode"] = action_pred.numpy().copy()

    offset = seq_len if has_clean else 0
    e_video = e_tokens[:, offset:offset + seq_len]
    video_out = head_forward_per_token(video_pred, e_video, weights)
    video_noise_pred = unpatchify_pt(video_out, (f, h, w), PATCH_SIZE, OUT_CHANNELS)

    action_noise_pred = action_decoder_forward(action_pred, embodiment_id, weights)
    _nan_check("action_noise_pred (decoded)", action_noise_pred)
    return video_noise_pred, action_noise_pred, intermediates


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def make_test_inputs(num_blocks=2, h_patches=4, w_patches=4, text_len=16):
    """Create deterministic test inputs for the forward pass."""
    torch.manual_seed(SEED)

    # +1 because the first frame is the context/conditioning frame
    # (self-attention only, no associated action/state tokens).
    # The remaining num_blocks frames each pair with action/state tokens.
    f = (num_blocks + 1) * NUM_FRAMES_PER_BLOCK
    H = h_patches * PATCH_SIZE[1]
    W = w_patches * PATCH_SIZE[2]
    B = 1

    x = torch.randn(B, f, H, W, IN_CHANNELS)
    clean_x = torch.randn(B, f, H, W, IN_CHANNELS)
    y = torch.randn(B, f, H, W, I2V_CHANNELS)
    timestep = torch.tensor([500.0])
    timestep_action = torch.tensor([300.0])
    context = torch.randn(B, text_len, DROID_TEXT_DIM)
    clip_emb = torch.randn(B, NUM_IMAGE_TOKENS, IMAGE_DIM)
    state = torch.randn(B, num_blocks, STATE_DIM)
    embodiment_id = torch.zeros(B, dtype=torch.long)
    total_actions = num_blocks * NUM_ACTION_PER_BLOCK
    actions = torch.randn(B, total_actions, ACTION_DIM)

    return dict(
        x=x, clean_x=clean_x, y=y,
        timestep=timestep, timestep_action=timestep_action,
        context=context, clip_emb=clip_emb,
        state=state, embodiment_id=embodiment_id, actions=actions,
    )


def parse_args():
    p = argparse.ArgumentParser(description="Standalone CausalWanDiT forward pass")
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--num-layers", type=int, default=8)
    p.add_argument("--num-blocks", type=int, default=2,
                   help="Temporal blocks (each = 1 frame)")
    p.add_argument("--h-patches", type=int, default=4)
    p.add_argument("--w-patches", type=int, default=4)
    p.add_argument("--text-len", type=int, default=16)
    p.add_argument("--output", type=str, default="full_dit_outputs.npz")
    p.add_argument("--no-clean", action="store_true",
                   help="Disable teacher-forcing clean video")
    p.add_argument("--no-i2v", action="store_true",
                   help="Disable I2V conditioning (no y, no clip_emb)")
    return p.parse_args()


def validate_block_availability(weights, num_layers):
    """Check which blocks have all required weights, return max usable count."""
    required = [
        "modulation", "self_attn.q.weight", "self_attn.k.weight",
        "self_attn.v.weight", "self_attn.o.weight", "cross_attn.q.weight",
        "cross_attn.k.weight", "cross_attn.v.weight", "cross_attn.o.weight",
        "ffn.0.weight", "ffn.2.weight", "norm3.weight",
    ]
    max_complete = 0
    for i in range(num_layers):
        missing = [s for s in required if f"model.blocks.{i}.{s}" not in weights]
        if not missing:
            max_complete = i + 1
        else:
            print(f"  Block {i}: INCOMPLETE (missing {len(missing)} keys, e.g. {missing[0]})")
            break
    return max_complete


def print_input_shapes(inputs):
    """Print shapes of all non-None inputs."""
    for name in ["x", "clean_x", "y", "context", "clip_emb", "actions", "state"]:
        val = inputs.get(name)
        if val is not None and hasattr(val, "shape"):
            print(f"  {name:<14s} {tuple(val.shape)}")
    print(f"  timestep:     {inputs['timestep'].item()}")


def save_results(video_pred, action_pred, inputs, intermediates, output_path):
    """Save forward pass results + all inputs + intermediates to .npz file."""
    results = {
        "video_noise_pred": video_pred.numpy(),
        "action_noise_pred": action_pred.numpy(),
        "input_x": inputs["x"].numpy(),
        "input_timestep": inputs["timestep"].numpy(),
        "input_actions": inputs["actions"].numpy(),
        "input_state": inputs["state"].numpy(),
        "input_timestep_action": inputs["timestep_action"].numpy(),
        "input_embodiment_id": inputs["embodiment_id"].numpy(),
    }
    if inputs.get("context") is not None:
        results["input_context"] = inputs["context"].numpy()
    if inputs.get("clip_emb") is not None:
        results["input_clip_emb"] = inputs["clip_emb"].numpy()
    if inputs.get("clean_x") is not None:
        results["input_clean_x"] = inputs["clean_x"].numpy()
    if inputs.get("y") is not None:
        results["input_y"] = inputs["y"].numpy()
    results.update(intermediates)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(output_path), **results)
    return results


def load_and_validate_weights(args):
    """Load weights, validate block availability, return (weights, num_layers)."""
    has_i2v = not args.no_i2v
    print("Loading checkpoint (selective)...")
    prefixes = compute_needed_prefixes(args.num_layers, has_i2v)
    raw_weights = load_selective_weights(args.checkpoint_dir, prefixes)
    weights = strip_prefix(raw_weights, "action_head.")

    max_complete = validate_block_availability(weights, args.num_layers)
    if max_complete < args.num_layers:
        print(f"\n  WARNING: Only {max_complete}/{args.num_layers} blocks available")
        print(f"  Reducing num_layers to {max_complete}")
    if max_complete == 0:
        print("  ERROR: No complete blocks available.")
        sys.exit(1)
    return weights, max_complete


def run_forward(inputs, weights, num_layers):
    """Run the forward pass and return (video_pred, action_pred, intermediates, elapsed)."""
    print(f"\nRunning forward pass ({num_layers} layers)...")
    t1 = time.time()
    with torch.no_grad():
        video_pred, action_pred, intermediates = causal_wan_dit_forward(
            x=inputs["x"], timestep=inputs["timestep"], context=inputs["context"],
            state=inputs["state"], embodiment_id=inputs["embodiment_id"],
            actions=inputs["actions"], timestep_action=inputs["timestep_action"],
            clean_x=inputs["clean_x"], clip_emb=inputs["clip_emb"],
            y=inputs["y"], weights=weights, num_layers=num_layers,
        )
    return video_pred, action_pred, intermediates, time.time() - t1


def main():
    args = parse_args()
    print(f"\n{'=' * 60}")
    print(f"  CausalWanDiT Full Forward Pass (PyTorch standalone)")
    print(f"  checkpoint: {args.checkpoint_dir}")
    print(f"  layers: {args.num_layers}  blocks: {args.num_blocks}  "
          f"patches: {args.h_patches}x{args.w_patches}")
    print(f"{'=' * 60}\n")

    t0 = time.time()
    weights, num_layers = load_and_validate_weights(args)
    print(f"  {len(weights)} tensors loaded ({time.time() - t0:.1f}s)")

    inputs = make_test_inputs(
        num_blocks=args.num_blocks, h_patches=args.h_patches,
        w_patches=args.w_patches, text_len=args.text_len,
    )
    if args.no_clean:
        inputs["clean_x"] = None
    if args.no_i2v:
        inputs["y"] = None
        inputs["clip_emb"] = None
    print_input_shapes(inputs)

    video_pred, action_pred, intermediates, elapsed = run_forward(inputs, weights, num_layers)

    print(f"\n  video_noise_pred:  {tuple(video_pred.shape)}  "
          f"mean={video_pred.mean():.6f}  std={video_pred.std():.6f}")
    print(f"  action_noise_pred: {tuple(action_pred.shape)}  "
          f"mean={action_pred.mean():.6f}  std={action_pred.std():.6f}")
    has_nan = torch.isnan(video_pred).any() or torch.isnan(action_pred).any()
    if has_nan:
        print("  WARNING: NaN detected!")

    results = save_results(video_pred, action_pred, inputs, intermediates, Path(args.output))
    print(f"\n  Saved {len(results)} arrays to {args.output}")
    print(f"  Intermediates: {[k for k in results if k.startswith('inter_')]}")
    print(f"  Forward: {elapsed:.1f}s  Total: {time.time() - t0:.1f}s  NaN: {'YES' if has_nan else 'no'}")


if __name__ == "__main__":
    main()
