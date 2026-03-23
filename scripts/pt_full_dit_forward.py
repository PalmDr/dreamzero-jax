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
    return torch.from_numpy(weights[key].copy()).float()


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


def wan_rope_3d(head_dim, f, h, w, theta=10000.0):
    """Compute WAN-style 3D RoPE frequencies. Returns complex (f*h*w, d)."""
    d = head_dim // 2
    dim_h = d // 3
    dim_w = d // 3
    dim_f = d - 2 * dim_h

    freqs_f = 1.0 / (theta ** (torch.arange(0, dim_f * 2, 2, dtype=torch.float32) / (dim_f * 2)))
    freqs_h = 1.0 / (theta ** (torch.arange(0, dim_h * 2, 2, dtype=torch.float32) / (dim_h * 2)))
    freqs_w = 1.0 / (theta ** (torch.arange(0, dim_w * 2, 2, dtype=torch.float32) / (dim_w * 2)))

    pos_f = torch.arange(f, dtype=torch.float32)
    pos_h = torch.arange(h, dtype=torch.float32)
    pos_w = torch.arange(w, dtype=torch.float32)

    table_f = torch.outer(pos_f, freqs_f)
    table_h = torch.outer(pos_h, freqs_h)
    table_w = torch.outer(pos_w, freqs_w)

    tf = table_f[:, None, None, :].expand(f, h, w, dim_f).reshape(f * h * w, dim_f)
    th = table_h[None, :, None, :].expand(f, h, w, dim_h).reshape(f * h * w, dim_h)
    tw = table_w[None, None, :, :].expand(f, h, w, dim_w).reshape(f * h * w, dim_w)

    freqs = torch.cat([tf, th, tw], dim=-1)
    return torch.polar(torch.ones_like(freqs), freqs)


def build_combined_rope(head_dim, f, h, w, num_blocks, has_clean):
    """Assemble RoPE for the interleaved sequence layout."""
    video_freqs = wan_rope_3d(head_dim, f, h, w)
    block_video_tokens = NUM_FRAMES_PER_BLOCK * h * w

    d = head_dim // 2
    base_freqs = 1.0 / (10000.0 ** (torch.arange(0, d * 2, 2, dtype=torch.float32) / (d * 2)))

    parts = []
    if has_clean:
        parts.append(video_freqs)

    for i in range(num_blocks):
        vs = i * block_video_tokens
        parts.append(video_freqs[vs: vs + block_video_tokens])
        action_angles = torch.outer(
            torch.full((NUM_ACTION_PER_BLOCK,), float(i)), base_freqs
        )
        parts.append(torch.polar(torch.ones_like(action_angles), action_angles))
        state_angles = torch.outer(
            torch.full((NUM_STATE_PER_BLOCK,), float(i)), base_freqs
        )
        parts.append(torch.polar(torch.ones_like(state_angles), state_angles))

    return torch.cat(parts, dim=0)


def apply_rotary_emb_pt(x, freqs_cis):
    """Apply RoPE to (B, num_heads, seq, head_dim) tensor."""
    shape = x.shape
    x = x.reshape(*shape[:-1], -1, 2)
    x_complex = torch.view_as_complex(x.float().contiguous())
    freqs = freqs_cis.unsqueeze(0).unsqueeze(0)
    x_rotated = x_complex * freqs
    x_out = torch.view_as_real(x_rotated).reshape(shape)
    return x_out


# ---------------------------------------------------------------------------
# Block-causal mask
# ---------------------------------------------------------------------------


def make_action_causal_mask(num_blocks, block_video_tokens, has_clean):
    """Build block-causal attention mask."""
    TYPE_CLEAN, TYPE_VIDEO, TYPE_ACTION, TYPE_STATE = 0, 1, 2, 3

    clean_len = num_blocks * block_video_tokens if has_clean else 0
    block_noisy = block_video_tokens + NUM_ACTION_PER_BLOCK + NUM_STATE_PER_BLOCK

    parts_block = []
    parts_type = []

    if has_clean:
        parts_block.append(torch.arange(num_blocks).repeat_interleave(block_video_tokens))
        parts_type.append(torch.full((clean_len,), TYPE_CLEAN, dtype=torch.int32))

    block_type = torch.cat([
        torch.full((block_video_tokens,), TYPE_VIDEO, dtype=torch.int32),
        torch.full((NUM_ACTION_PER_BLOCK,), TYPE_ACTION, dtype=torch.int32),
        torch.full((NUM_STATE_PER_BLOCK,), TYPE_STATE, dtype=torch.int32),
    ])
    parts_type.append(block_type.repeat(num_blocks))
    parts_block.append(torch.arange(num_blocks).repeat_interleave(block_noisy))

    block_ids = torch.cat(parts_block)
    type_ids = torch.cat(parts_type)

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

    clean_rule = is_clean_k & k_earlier_or_same
    noisy_rule = (is_clean_k & k_earlier_or_same) | ((is_video_k | is_action_k) & same_block)
    state_rule = is_state_k & same_block

    is_clean_q = q_type == TYPE_CLEAN
    is_state_q = q_type == TYPE_STATE

    mask = torch.where(is_clean_q, clean_rule, torch.where(~is_state_q, noisy_rule, state_rule))
    return mask


# ---------------------------------------------------------------------------
# Attention with RoPE + mask (for DiT blocks)
# ---------------------------------------------------------------------------


def attention_rope_mask(q, k, v, num_heads, freqs_cis, mask=None):
    """MHA with RoPE on Q/K and optional boolean mask."""
    B, L_q, D = q.shape
    L_kv = k.shape[1]
    head_dim = D // num_heads

    q = q.reshape(B, L_q, num_heads, head_dim).transpose(1, 2)
    k = k.reshape(B, L_kv, num_heads, head_dim).transpose(1, 2)
    v = v.reshape(B, L_kv, num_heads, head_dim).transpose(1, 2)

    if freqs_cis is not None:
        q = apply_rotary_emb_pt(q, freqs_cis[:L_q])
        k = apply_rotary_emb_pt(k, freqs_cis[:L_kv])
        v = v.float()

    scale = head_dim ** -0.5
    attn = (q @ k.transpose(-2, -1)) * scale

    if mask is not None:
        attn = attn.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    attn = torch.softmax(attn.float(), dim=-1).to(q.dtype)
    out = attn @ v
    return out.transpose(1, 2).reshape(B, L_q, D)


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


def dit_self_attn(h, weights, prefix, freqs_cis, mask):
    """Self-attention with QK-norm, RoPE, and mask."""
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
    out = attention_rope_mask(q, k, v, DROID_HEADS, freqs_cis, mask)
    return linear_forward(out, get_weight(weights, f"{prefix}.o.weight"),
                          get_weight(weights, f"{prefix}.o.bias"))


# ---------------------------------------------------------------------------
# Single DiT block
# ---------------------------------------------------------------------------


def dit_block_forward(x, e, context, weights, blk_prefix, freqs_cis, mask, use_i2v_ca):
    """One CausalWanDiT block: AdaLN SA + norm3 cross-attn + AdaLN FFN."""
    D = x.shape[-1]

    mod = get_weight(weights, f"{blk_prefix}.modulation") + e
    shift_msa, scale_msa, gate_msa = mod[:, 0], mod[:, 1], mod[:, 2]
    shift_mlp, scale_mlp, gate_mlp = mod[:, 3], mod[:, 4], mod[:, 5]

    h = torch.layer_norm(x, [D]) * (1 + scale_msa[:, None, :]) + shift_msa[:, None, :]
    sa = dit_self_attn(h, weights, f"{blk_prefix}.self_attn", freqs_cis, mask)
    x = x + sa * gate_msa[:, None, :]

    n3_w = get_weight(weights, f"{blk_prefix}.norm3.weight")
    n3_b = get_weight(weights, f"{blk_prefix}.norm3.bias")
    h = torch.nn.functional.layer_norm(x, [D], n3_w, n3_b)
    if use_i2v_ca:
        ca = i2v_cross_attention(h, context, weights, f"{blk_prefix}.cross_attn", DROID_HEADS)
    else:
        ca = text_only_cross_attention(h, context, weights, f"{blk_prefix}.cross_attn", DROID_HEADS)
    x = x + ca

    h = torch.layer_norm(x, [D]) * (1 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]
    h = gelu_approx(linear_forward(h, get_weight(weights, f"{blk_prefix}.ffn.0.weight"),
                                   get_weight(weights, f"{blk_prefix}.ffn.0.bias")))
    h = linear_forward(h, get_weight(weights, f"{blk_prefix}.ffn.2.weight"),
                       get_weight(weights, f"{blk_prefix}.ffn.2.bias"))
    x = x + h * gate_mlp[:, None, :]

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


def head_forward(x, t_emb, weights, prefix="model.head"):
    """WanDiTHead: 2-param modulation -> LayerNorm -> Linear."""
    D = x.shape[-1]
    mod = get_weight_or_rand(weights, f"{prefix}.modulation", (1, 2, D), "head.modulation")
    mod = mod + t_emb[:, None, :]
    shift, scale = mod[:, 0], mod[:, 1]
    x = torch.layer_norm(x, [D]) * (1 + scale[:, None, :]) + shift[:, None, :]
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
    x = gelu_approx(linear_forward(x, w1, b1))
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


def action_encoder_forward(actions, timestep_action, cat_ids, weights, prefix="model.action_encoder"):
    """MultiEmbodimentActionEncoder forward."""
    B, T, _ = actions.shape
    a_emb = category_linear(actions, weights, f"{prefix}.W1", cat_ids)
    tau_emb = sinusoidal_embedding_pt(timestep_action, DROID_DIM)
    tau_emb = tau_emb[:, None, :].expand(B, T, DROID_DIM)
    x = torch.cat([a_emb, tau_emb], dim=-1)
    x = torch.nn.functional.silu(category_linear(x, weights, f"{prefix}.W2", cat_ids))
    return category_linear(x, weights, f"{prefix}.W3", cat_ids)


# ---------------------------------------------------------------------------
# State encoder
# ---------------------------------------------------------------------------


def state_encoder_forward(state, cat_ids, weights, prefix="model.state_encoder"):
    """CategorySpecificMLP: SiLU(Linear1) -> Linear2."""
    h = torch.nn.functional.silu(category_linear(state, weights, f"{prefix}.layer1", cat_ids))
    return category_linear(h, weights, f"{prefix}.layer2", cat_ids)


# ---------------------------------------------------------------------------
# Action decoder
# ---------------------------------------------------------------------------


def action_decoder_forward(x, cat_ids, weights, prefix="model.action_decoder"):
    """CategorySpecificMLP: SiLU(Linear1) -> Linear2."""
    h = torch.nn.functional.silu(category_linear(x, weights, f"{prefix}.layer1", cat_ids))
    return category_linear(h, weights, f"{prefix}.layer2", cat_ids)


# ---------------------------------------------------------------------------
# Time conditioning
# ---------------------------------------------------------------------------


def time_conditioning(timestep, weights):
    """sinusoidal -> MLP(SiLU) -> time_projection -> reshape(B, 6, D)."""
    sin_emb = sinusoidal_embedding_pt(timestep, DROID_FREQ_DIM)
    h = torch.nn.functional.silu(
        linear_forward(sin_emb, get_weight(weights, "model.time_embedding.0.weight"),
                       get_weight(weights, "model.time_embedding.0.bias")))
    t_emb = linear_forward(h, get_weight(weights, "model.time_embedding.2.weight"),
                           get_weight(weights, "model.time_embedding.2.bias"))
    e = torch.nn.functional.silu(t_emb)
    e = linear_forward(e, get_weight(weights, "model.time_projection.1.weight"),
                       get_weight(weights, "model.time_projection.1.bias"))
    B = timestep.shape[0]
    return t_emb, e.reshape(B, 6, DROID_DIM)


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


def _concat_i2v_conditioning(x, y):
    """Concatenate I2V conditioning channels (or zeros if y is None)."""
    if y is not None:
        return torch.cat([x, y], dim=-1)
    return torch.cat([x, torch.zeros(*x.shape[:-1], I2V_CHANNELS)], dim=-1)


def _interleave_noisy_seq(x_flat, action_emb, state_emb, num_blocks, block_video_tokens):
    """Interleave video, action, state tokens into noisy sequence."""
    parts = []
    for i in range(num_blocks):
        vs = i * block_video_tokens
        parts.append(x_flat[:, vs: vs + block_video_tokens])
        as_ = i * NUM_ACTION_PER_BLOCK
        parts.append(action_emb[:, as_: as_ + NUM_ACTION_PER_BLOCK])
        parts.append(state_emb[:, i: i + 1])
    return torch.cat(parts, dim=1)


def _extract_predictions(full_seq, clean_len, num_blocks, block_video_tokens, block_noisy):
    """Extract video and action tokens from the noisy section of the sequence."""
    noisy_section = full_seq[:, clean_len:]
    block_offsets = torch.arange(num_blocks) * block_noisy
    video_within = torch.arange(block_video_tokens)
    video_indices = (block_offsets[:, None] + video_within[None, :]).reshape(-1)
    action_within = torch.arange(NUM_ACTION_PER_BLOCK) + block_video_tokens
    action_indices = (block_offsets[:, None] + action_within[None, :]).reshape(-1)
    return noisy_section[:, video_indices], noisy_section[:, action_indices]


def causal_wan_dit_forward(
    x, timestep, context, state, embodiment_id, actions,
    timestep_action, clean_x, clip_emb, y, weights, num_layers,
):
    """Full CausalWanDiT forward pass in pure PyTorch."""
    has_clean = clean_x is not None
    B = x.shape[0]

    x = _concat_i2v_conditioning(x, y)
    x_patched, f, h, w = patch_embed_forward(x, weights)
    x_flat = x_patched.reshape(B, f * h * w, DROID_DIM)

    block_video_tokens = NUM_FRAMES_PER_BLOCK * h * w
    num_blocks = f // NUM_FRAMES_PER_BLOCK
    block_noisy = block_video_tokens + NUM_ACTION_PER_BLOCK + NUM_STATE_PER_BLOCK

    action_emb = action_encoder_forward(actions, timestep_action, embodiment_id, weights)
    state_emb = state_encoder_forward(state, embodiment_id, weights)
    noisy_seq = _interleave_noisy_seq(x_flat, action_emb, state_emb, num_blocks, block_video_tokens)

    if has_clean:
        clean_x = _concat_i2v_conditioning(clean_x, y)
        clean_patched, _, _, _ = patch_embed_forward(clean_x, weights)
        full_seq = torch.cat([clean_patched.reshape(B, f * h * w, DROID_DIM), noisy_seq], dim=1)
    else:
        full_seq = noisy_seq

    mask = make_action_causal_mask(num_blocks, block_video_tokens, has_clean)
    freqs_cis = build_combined_rope(DROID_HEAD_DIM, f, h, w, num_blocks, has_clean)
    t_emb, e = time_conditioning(timestep, weights)

    ctx = text_conditioning(context, weights)
    has_img_weights = has_weight(weights, "model.img_emb.proj.0.weight")
    if clip_emb is not None and has_img_weights:
        ctx = torch.cat([img_emb_forward(clip_emb, weights), ctx], dim=1)

    for i in range(num_layers):
        print(f"    block {i}/{num_layers}...", end="", flush=True)
        t0 = time.time()
        full_seq = dit_block_forward(
            full_seq, e, ctx, weights, f"model.blocks.{i}", freqs_cis, mask, has_img_weights)
        print(f" {time.time() - t0:.1f}s")

    video_pred, action_pred = _extract_predictions(
        full_seq, f * h * w if has_clean else 0, num_blocks, block_video_tokens, block_noisy)
    video_out = head_forward(video_pred, t_emb, weights)
    video_noise_pred = unpatchify_pt(video_out, (f, h, w), PATCH_SIZE, OUT_CHANNELS)
    action_noise_pred = action_decoder_forward(action_pred, embodiment_id, weights)
    return video_noise_pred, action_noise_pred


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def make_test_inputs(num_blocks=2, h_patches=4, w_patches=4, text_len=16):
    """Create deterministic test inputs for the forward pass."""
    torch.manual_seed(SEED)

    f = num_blocks * NUM_FRAMES_PER_BLOCK
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


def save_results(video_pred, action_pred, inputs, output_path):
    """Save forward pass results to .npz file."""
    results = {
        "video_noise_pred": video_pred.numpy(),
        "action_noise_pred": action_pred.numpy(),
        "input_x": inputs["x"].numpy(),
        "input_timestep": inputs["timestep"].numpy(),
        "input_actions": inputs["actions"].numpy(),
        "input_state": inputs["state"].numpy(),
    }
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
    """Run the forward pass and return (video_pred, action_pred, elapsed)."""
    print(f"\nRunning forward pass ({num_layers} layers)...")
    t1 = time.time()
    with torch.no_grad():
        video_pred, action_pred = causal_wan_dit_forward(
            x=inputs["x"], timestep=inputs["timestep"], context=inputs["context"],
            state=inputs["state"], embodiment_id=inputs["embodiment_id"],
            actions=inputs["actions"], timestep_action=inputs["timestep_action"],
            clean_x=inputs["clean_x"], clip_emb=inputs["clip_emb"],
            y=inputs["y"], weights=weights, num_layers=num_layers,
        )
    return video_pred, action_pred, time.time() - t1


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

    video_pred, action_pred, elapsed = run_forward(inputs, weights, num_layers)

    print(f"\n  video_noise_pred:  {tuple(video_pred.shape)}  "
          f"mean={video_pred.mean():.6f}  std={video_pred.std():.6f}")
    print(f"  action_noise_pred: {tuple(action_pred.shape)}  "
          f"mean={action_pred.mean():.6f}  std={action_pred.std():.6f}")
    has_nan = torch.isnan(video_pred).any() or torch.isnan(action_pred).any()
    if has_nan:
        print("  WARNING: NaN detected!")

    results = save_results(video_pred, action_pred, inputs, Path(args.output))
    print(f"\n  Saved {len(results)} arrays to {args.output}")
    print(f"  Forward: {elapsed:.1f}s  Total: {time.time() - t0:.1f}s  NaN: {'YES' if has_nan else 'no'}")


if __name__ == "__main__":
    main()
