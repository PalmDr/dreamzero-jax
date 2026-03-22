"""Standalone PyTorch ops matching DreamZero's architecture.

Pure-function implementations of RMSNorm, attention, AdaLN, FFN, etc.
using only torch -- no GR00T or model framework dependencies.

Used by ``pytorch_standalone_forward.py`` for weight verification.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# DROID 14B constants
# ---------------------------------------------------------------------------

SEED = 42
DROID_DIM = 5120
DROID_HEADS = 40
DROID_FFN_DIM = 13824
DROID_HEAD_DIM = DROID_DIM // DROID_HEADS
DROID_TEXT_DIM = 4096
DROID_TEXT_FFN_DIM = 10240
DROID_TEXT_HEADS = 64
DROID_TEXT_HEAD_DIM = DROID_TEXT_DIM // DROID_TEXT_HEADS
DROID_FREQ_DIM = 256


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------


def load_sharded_safetensors(ckpt_dir: Path) -> dict[str, np.ndarray]:
    """Load all shards from a safetensors index into numpy arrays."""
    from safetensors import safe_open

    index_path = ckpt_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
    else:
        shard_files = sorted(p.name for p in ckpt_dir.glob("*.safetensors"))

    import torch
    weights: dict[str, np.ndarray] = {}
    for shard_name in shard_files:
        shard_path = ckpt_dir / shard_name
        if not shard_path.exists():
            print(f"  Skipping {shard_name} (not found)")
            continue
        print(f"  Loading {shard_name}...")
        with safe_open(str(shard_path), framework="pt") as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key).float().numpy()
    return weights


def strip_prefix(weights: dict[str, np.ndarray], prefix: str) -> dict[str, np.ndarray]:
    """Strip a common prefix from all weight keys."""
    return {
        (k[len(prefix):] if k.startswith(prefix) else k): v
        for k, v in weights.items()
    }


def get_weight(weights: dict[str, np.ndarray], key: str):
    """Fetch a weight by key and convert to torch float32 tensor."""
    import torch
    return torch.from_numpy(weights[key].copy()).float()


# ---------------------------------------------------------------------------
# Primitive ops
# ---------------------------------------------------------------------------


def sinusoidal_embedding_pt(timesteps, dim, max_period=10000.0):
    """Sinusoidal timestep embedding matching the JAX implementation."""
    import torch
    half_dim = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half_dim, dtype=torch.float32) / half_dim
    )
    args = timesteps[:, None].float() * freqs[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def rmsnorm_forward(x, scale, eps=1e-6):
    """RMSNorm: x * scale / sqrt(mean(x^2) + eps)."""
    import torch
    rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + eps)
    return (x.float() / rms * scale.float()).to(x.dtype)


def linear_forward(x, weight, bias=None):
    """Linear: x @ weight.T + bias (PyTorch convention: weight is [out, in])."""
    out = x @ weight.T
    if bias is not None:
        out = out + bias
    return out


def gelu_approx(x):
    """GELU with tanh approximation."""
    import torch
    return torch.nn.functional.gelu(x, approximate="tanh")


def attention_forward(q, k, v, num_heads, scale=None):
    """Standard multi-head attention with 1/sqrt(d) scaling.

    Args:
        q, k, v: (B, L, dim) already-projected tensors.
        num_heads: Number of attention heads.
        scale: Override for attention scale (default: 1/sqrt(head_dim)).

    Returns:
        (B, L_q, dim) attention output.
    """
    import torch
    B, L_q, D = q.shape
    L_kv = k.shape[1]
    head_dim = D // num_heads

    q = q.reshape(B, L_q, num_heads, head_dim).transpose(1, 2)
    k = k.reshape(B, L_kv, num_heads, head_dim).transpose(1, 2)
    v = v.reshape(B, L_kv, num_heads, head_dim).transpose(1, 2)

    if scale is None:
        scale = head_dim ** -0.5

    attn = (q @ k.transpose(-2, -1)) * scale
    attn = torch.softmax(attn.float(), dim=-1).to(q.dtype)
    out = attn @ v
    return out.transpose(1, 2).reshape(B, L_q, D)


def t5_attention(q, k, v, num_heads):
    """T5-style attention WITHOUT 1/sqrt(d) scaling."""
    import torch
    B, L_q, D = q.shape
    L_kv = k.shape[1]
    head_dim = D // num_heads

    q = q.reshape(B, L_q, num_heads, head_dim).transpose(1, 2)
    k = k.reshape(B, L_kv, num_heads, head_dim).transpose(1, 2)
    v = v.reshape(B, L_kv, num_heads, head_dim).transpose(1, 2)

    attn = q @ k.transpose(-2, -1)
    attn = torch.softmax(attn.float(), dim=-1).to(q.dtype)
    out = attn @ v
    return out.transpose(1, 2).reshape(B, L_q, D)


def load_gate_weight(weights: dict[str, np.ndarray], prefix: str):
    """Load T5 gate weight -- may be gate.0.weight (Sequential) or gate.weight."""
    key_seq = f"{prefix}.0.weight"
    key_flat = f"{prefix}.weight"
    if key_seq in weights:
        return get_weight(weights, key_seq)
    return get_weight(weights, key_flat)
