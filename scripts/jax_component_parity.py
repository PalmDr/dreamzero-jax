#!/usr/bin/env python3
"""Standalone JAX forward pass for DreamZero weight verification.

Loads safetensors checkpoint shards and runs individual model components
through JAX reimplementations that match the PyTorch standalone forward pass
exactly. Uses identical deterministic inputs (seed=42) so outputs can be
compared directly.

Saves per-component outputs as .npz for comparison against PyTorch via
``scripts/compare_outputs.py``.

Usage
-----
Run all components::

    python scripts/jax_component_parity.py \
        --checkpoint-dir /path/to/DreamZero-DROID \
        --output jax_standalone_outputs.npz

Run specific components::

    python scripts/jax_component_parity.py \
        --checkpoint-dir /path/to/DreamZero-DROID \
        --components rmsnorm,linear,attention \
        --output jax_standalone_outputs.npz
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_default_matmul_precision", "float32")

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))
sys.path.insert(0, str(_project_root / "scripts"))

from pt_standalone_ops import (
    DROID_DIM,
    DROID_FFN_DIM,
    DROID_FREQ_DIM,
    DROID_HEAD_DIM,
    DROID_HEADS,
    DROID_TEXT_DIM,
    DROID_TEXT_FFN_DIM,
    DROID_TEXT_HEAD_DIM,
    DROID_TEXT_HEADS,
    SEED,
    load_sharded_safetensors,
    strip_prefix,
)


# ---------------------------------------------------------------------------
# Torch-compatible random input generation
# ---------------------------------------------------------------------------
# PyTorch and numpy use different RNG algorithms, so we must use PyTorch
# to guarantee identical inputs across both scripts.


def torch_randn(*shape: int, seed: int = SEED) -> np.ndarray:
    """Generate random normal values matching torch.manual_seed(seed)+randn."""
    import torch
    torch.manual_seed(seed)
    return torch.randn(*shape).numpy()


def torch_randn_sequence(shapes: list[tuple[int, ...]], seed: int = SEED) -> list[np.ndarray]:
    """Generate a sequence of randn tensors from the same seeded generator."""
    import torch
    torch.manual_seed(seed)
    return [torch.randn(*s).numpy() for s in shapes]


# ---------------------------------------------------------------------------
# Weight accessors
# ---------------------------------------------------------------------------


def get_jax_weight(weights: dict[str, np.ndarray], key: str) -> jax.Array:
    return jnp.array(weights[key], dtype=jnp.float32)


def get_jax_weight_t(weights: dict[str, np.ndarray], key: str) -> jax.Array:
    """Transpose PT (out,in) -> JAX (in,out) for dense matmul."""
    return jnp.array(weights[key].T, dtype=jnp.float32)


# ---------------------------------------------------------------------------
# Primitive ops — matching PyTorch exactly
# ---------------------------------------------------------------------------


def rmsnorm_jax(x: jax.Array, scale: jax.Array, eps: float = 1e-6) -> jax.Array:
    x_f32 = x.astype(jnp.float32)
    rms = jnp.sqrt(jnp.mean(x_f32 ** 2, axis=-1, keepdims=True) + eps)
    return (x_f32 / rms * scale.astype(jnp.float32)).astype(x.dtype)


def linear_jax(x: jax.Array, weight_t: jax.Array, bias: jax.Array | None = None) -> jax.Array:
    """x @ weight_t + bias, where weight_t is already (in, out)."""
    out = x @ weight_t
    if bias is not None:
        out = out + bias
    return out


def gelu_approx_jax(x: jax.Array) -> jax.Array:
    return jax.nn.gelu(x, approximate=True)


def attention_jax(
    q: jax.Array, k: jax.Array, v: jax.Array,
    num_heads: int, scale: float | None = None,
) -> jax.Array:
    """Standard multi-head attention with 1/sqrt(d) scaling.

    q, k, v: (B, L, dim) already-projected tensors.
    """
    B, L_q, D = q.shape
    L_kv = k.shape[1]
    head_dim = D // num_heads

    q = q.reshape(B, L_q, num_heads, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(B, L_kv, num_heads, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(B, L_kv, num_heads, head_dim).transpose(0, 2, 1, 3)

    if scale is None:
        scale = head_dim ** -0.5

    attn = (q @ k.transpose(0, 1, 3, 2)) * scale
    attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(q.dtype)
    out = attn @ v
    return out.transpose(0, 2, 1, 3).reshape(B, L_q, D)


def t5_attention_jax(
    q: jax.Array, k: jax.Array, v: jax.Array, num_heads: int,
) -> jax.Array:
    """T5-style attention WITHOUT 1/sqrt(d) scaling."""
    B, L_q, D = q.shape
    L_kv = k.shape[1]
    head_dim = D // num_heads

    q = q.reshape(B, L_q, num_heads, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(B, L_kv, num_heads, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(B, L_kv, num_heads, head_dim).transpose(0, 2, 1, 3)

    attn = q @ k.transpose(0, 1, 3, 2)
    attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(q.dtype)
    out = attn @ v
    return out.transpose(0, 2, 1, 3).reshape(B, L_q, D)


def sinusoidal_embedding_jax(timesteps: jax.Array, dim: int, max_period: float = 10000.0) -> jax.Array:
    half_dim = dim // 2
    freqs = jnp.exp(
        -jnp.log(max_period) * jnp.arange(half_dim, dtype=jnp.float32) / half_dim
    )
    args = timesteps[:, None].astype(jnp.float32) * freqs[None, :]
    return jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)


def load_gate_weight_jax(weights: dict[str, np.ndarray], prefix: str) -> jax.Array:
    """Load T5 gate weight — transposed to (in, out) for JAX."""
    key_seq = f"{prefix}.0.weight"
    key_flat = f"{prefix}.weight"
    if key_seq in weights:
        return get_jax_weight_t(weights, key_seq)
    return get_jax_weight_t(weights, key_flat)


def _layer_norm_jax(x: jax.Array, D: int) -> jax.Array:
    """Match torch.layer_norm(x, [D]) — no learnable params."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + 1e-5)


def _layer_norm_affine_jax(x: jax.Array, weight: jax.Array, bias: jax.Array) -> jax.Array:
    """Match F.layer_norm(x, [D], weight, bias)."""
    return _layer_norm_jax(x, x.shape[-1]) * weight + bias


# ---------------------------------------------------------------------------
# Component test runners
# ---------------------------------------------------------------------------


def run_rmsnorm_test(weights: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    B, L, D = 1, 16, DROID_TEXT_DIM
    scale = get_jax_weight(weights, "text_encoder.norm.weight")
    x_np = torch_randn(B, L, D)
    x = jnp.array(x_np, dtype=jnp.float32)
    out = rmsnorm_jax(x, scale)
    return {
        "rmsnorm_input": x_np,
        "rmsnorm_scale": np.array(scale),
        "rmsnorm_output": np.array(out),
    }


def run_linear_test(weights: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    B, L, D = 1, 16, DROID_DIM
    w_t = get_jax_weight_t(weights, "model.blocks.0.self_attn.q.weight")
    b = get_jax_weight(weights, "model.blocks.0.self_attn.q.bias")
    x_np = torch_randn(B, L, D)
    x = jnp.array(x_np, dtype=jnp.float32)
    out = linear_jax(x, w_t, b)
    return {
        "linear_input": x_np,
        "linear_weight": weights["model.blocks.0.self_attn.q.weight"].astype(np.float32),
        "linear_bias": np.array(b),
        "linear_output": np.array(out),
    }


def run_attention_test(weights: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    B, L, D = 1, 16, DROID_DIM
    prefix = "model.blocks.0.self_attn"

    x_np = torch_randn(B, L, D)
    x = jnp.array(x_np, dtype=jnp.float32)

    q = linear_jax(x, get_jax_weight_t(weights, f"{prefix}.q.weight"),
                   get_jax_weight(weights, f"{prefix}.q.bias"))
    k = linear_jax(x, get_jax_weight_t(weights, f"{prefix}.k.weight"),
                   get_jax_weight(weights, f"{prefix}.k.bias"))
    v = linear_jax(x, get_jax_weight_t(weights, f"{prefix}.v.weight"),
                   get_jax_weight(weights, f"{prefix}.v.bias"))
    q = rmsnorm_jax(q, get_jax_weight(weights, f"{prefix}.norm_q.weight"))
    k = rmsnorm_jax(k, get_jax_weight(weights, f"{prefix}.norm_k.weight"))
    attn_out = attention_jax(q, k, v, DROID_HEADS)
    out = linear_jax(attn_out, get_jax_weight_t(weights, f"{prefix}.o.weight"),
                     get_jax_weight(weights, f"{prefix}.o.bias"))
    return {
        "attn_input": x_np,
        "attn_q_proj": np.array(q),
        "attn_k_proj": np.array(k),
        "attn_v_proj": np.array(v),
        "attn_raw_output": np.array(attn_out),
        "attn_output": np.array(out),
    }


def run_adaln_test(weights: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    B, L, D = 1, 16, DROID_DIM
    mod_param = get_jax_weight(weights, "model.blocks.0.modulation")

    inputs = torch_randn_sequence([(B, L, D), (B, 6, D)])
    x_np, e_np = inputs
    x = jnp.array(x_np, dtype=jnp.float32)
    e = jnp.array(e_np, dtype=jnp.float32)

    mod = mod_param + e
    shift_msa, scale_msa = mod[:, 0], mod[:, 1]

    x_normed = _layer_norm_jax(x, D)
    x_modulated = x_normed * (1 + scale_msa[:, None, :]) + shift_msa[:, None, :]
    return {
        "adaln_input": x_np,
        "adaln_e": e_np,
        "adaln_modulation_param": np.array(mod_param),
        "adaln_normed": np.array(x_normed),
        "adaln_output": np.array(x_modulated),
    }


def run_ffn_test(weights: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    B, L, D = 1, 16, DROID_DIM
    prefix = "model.blocks.0.ffn"
    x_np = torch_randn(B, L, D)
    x = jnp.array(x_np, dtype=jnp.float32)
    h = gelu_approx_jax(linear_jax(x, get_jax_weight_t(weights, f"{prefix}.0.weight"),
                                    get_jax_weight(weights, f"{prefix}.0.bias")))
    out = linear_jax(h, get_jax_weight_t(weights, f"{prefix}.2.weight"),
                     get_jax_weight(weights, f"{prefix}.2.bias"))
    return {
        "ffn_input": x_np,
        "ffn_hidden": np.array(h),
        "ffn_output": np.array(out),
    }


def run_cross_attention_test(weights: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    B, S, D, L_ctx = 1, 16, DROID_DIM, 32
    prefix = "model.blocks.0.cross_attn"

    inputs = torch_randn_sequence([(B, S, D), (B, L_ctx, D)])
    x_np, context_np = inputs
    x = jnp.array(x_np, dtype=jnp.float32)
    context = jnp.array(context_np, dtype=jnp.float32)

    q = rmsnorm_jax(
        linear_jax(x, get_jax_weight_t(weights, f"{prefix}.q.weight"),
                   get_jax_weight(weights, f"{prefix}.q.bias")),
        get_jax_weight(weights, f"{prefix}.norm_q.weight"))
    k = rmsnorm_jax(
        linear_jax(context, get_jax_weight_t(weights, f"{prefix}.k.weight"),
                   get_jax_weight(weights, f"{prefix}.k.bias")),
        get_jax_weight(weights, f"{prefix}.norm_k.weight"))
    v = linear_jax(context, get_jax_weight_t(weights, f"{prefix}.v.weight"),
                   get_jax_weight(weights, f"{prefix}.v.bias"))
    attn_out = attention_jax(q, k, v, DROID_HEADS)
    out = linear_jax(attn_out, get_jax_weight_t(weights, f"{prefix}.o.weight"),
                     get_jax_weight(weights, f"{prefix}.o.bias"))
    return {
        "cross_attn_x": x_np,
        "cross_attn_context": context_np,
        "cross_attn_output": np.array(out),
    }


def run_time_embedding_test(weights: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    timestep = jnp.array([500.0])
    sin_emb = sinusoidal_embedding_jax(timestep, DROID_FREQ_DIM)
    h = jax.nn.silu(
        linear_jax(sin_emb, get_jax_weight_t(weights, "model.time_embedding.0.weight"),
                   get_jax_weight(weights, "model.time_embedding.0.bias")))
    t_emb = linear_jax(h, get_jax_weight_t(weights, "model.time_embedding.2.weight"),
                       get_jax_weight(weights, "model.time_embedding.2.bias"))
    e = jax.nn.silu(t_emb)
    e_proj = linear_jax(e, get_jax_weight_t(weights, "model.time_projection.1.weight"),
                        get_jax_weight(weights, "model.time_projection.1.bias"))
    return {
        "time_sinusoidal": np.array(sin_emb),
        "time_mlp_output": np.array(t_emb),
        "time_modulation_6x": np.array(e_proj.reshape(1, 6, DROID_DIM)),
    }


def run_text_encoder_block_test(weights: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    B, L, D = 1, 16, DROID_TEXT_DIM
    blk = "text_encoder.blocks.0"

    x_np = torch_randn(B, L, D)
    x = jnp.array(x_np, dtype=jnp.float32)

    h = rmsnorm_jax(x, get_jax_weight(weights, f"{blk}.norm1.weight"))
    q = linear_jax(h, get_jax_weight_t(weights, f"{blk}.attn.q.weight"))
    k = linear_jax(h, get_jax_weight_t(weights, f"{blk}.attn.k.weight"))
    v = linear_jax(h, get_jax_weight_t(weights, f"{blk}.attn.v.weight"))
    sa_out = linear_jax(
        t5_attention_jax(q, k, v, DROID_TEXT_HEADS),
        get_jax_weight_t(weights, f"{blk}.attn.o.weight"))
    x = x + sa_out

    h2 = rmsnorm_jax(x, get_jax_weight(weights, f"{blk}.norm2.weight"))
    gate_val = gelu_approx_jax(linear_jax(h2, load_gate_weight_jax(weights, f"{blk}.ffn.gate")))
    fc1_val = linear_jax(h2, get_jax_weight_t(weights, f"{blk}.ffn.fc1.weight"))
    ffn_out = linear_jax(fc1_val * gate_val,
                         get_jax_weight_t(weights, f"{blk}.ffn.fc2.weight"))
    x = x + ffn_out
    return {
        "text_block_sa_output": np.array(sa_out),
        "text_block_ffn_output": np.array(ffn_out),
        "text_block_output": np.array(x),
    }


def run_text_embedding_test(weights: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    B, L = 1, 16
    x_np = torch_randn(B, L, DROID_TEXT_DIM)
    x = jnp.array(x_np, dtype=jnp.float32)
    h = gelu_approx_jax(linear_jax(x, get_jax_weight_t(weights, "model.text_embedding.0.weight"),
                                    get_jax_weight(weights, "model.text_embedding.0.bias")))
    out = linear_jax(h, get_jax_weight_t(weights, "model.text_embedding.2.weight"),
                     get_jax_weight(weights, "model.text_embedding.2.bias"))
    return {
        "text_emb_proj_input": x_np,
        "text_emb_proj_output": np.array(out),
    }


def run_dit_block_test(weights: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Full DiT block: AdaLN self-attn + cross-attn + FFN (no RoPE)."""
    B, S, D, L_ctx = 1, 16, DROID_DIM, 8
    blk = "model.blocks.0"

    inputs = torch_randn_sequence([(B, S, D), (B, 6, D), (B, L_ctx, D)])
    x_np, e_np, context_np = inputs
    x = jnp.array(x_np, dtype=jnp.float32)
    e = jnp.array(e_np, dtype=jnp.float32)
    context = jnp.array(context_np, dtype=jnp.float32)

    mod = get_jax_weight(weights, f"{blk}.modulation") + e
    shift_msa, scale_msa, gate_msa = mod[:, 0], mod[:, 1], mod[:, 2]
    shift_mlp, scale_mlp, gate_mlp = mod[:, 3], mod[:, 4], mod[:, 5]

    h = _layer_norm_jax(x, D) * (1 + scale_msa[:, None, :]) + shift_msa[:, None, :]
    sa = _dit_self_attn_jax(h, weights, f"{blk}.self_attn")
    x = x + sa * gate_msa[:, None, :]

    n3_w = get_jax_weight(weights, f"{blk}.norm3.weight")
    n3_b = get_jax_weight(weights, f"{blk}.norm3.bias")
    h = _layer_norm_affine_jax(x, n3_w, n3_b)
    ca = _dit_cross_attn_jax(h, context, weights, f"{blk}.cross_attn")
    x = x + ca

    h = _layer_norm_jax(x, D) * (1 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]
    h = gelu_approx_jax(linear_jax(h, get_jax_weight_t(weights, f"{blk}.ffn.0.weight"),
                                    get_jax_weight(weights, f"{blk}.ffn.0.bias")))
    h = linear_jax(h, get_jax_weight_t(weights, f"{blk}.ffn.2.weight"),
                   get_jax_weight(weights, f"{blk}.ffn.2.bias"))
    x = x + h * gate_mlp[:, None, :]

    return {"dit_block_output": np.array(x)}


def _dit_self_attn_jax(
    h: jax.Array, weights: dict[str, np.ndarray], prefix: str,
) -> jax.Array:
    q = rmsnorm_jax(
        linear_jax(h, get_jax_weight_t(weights, f"{prefix}.q.weight"),
                   get_jax_weight(weights, f"{prefix}.q.bias")),
        get_jax_weight(weights, f"{prefix}.norm_q.weight"))
    k = rmsnorm_jax(
        linear_jax(h, get_jax_weight_t(weights, f"{prefix}.k.weight"),
                   get_jax_weight(weights, f"{prefix}.k.bias")),
        get_jax_weight(weights, f"{prefix}.norm_k.weight"))
    v = linear_jax(h, get_jax_weight_t(weights, f"{prefix}.v.weight"),
                   get_jax_weight(weights, f"{prefix}.v.bias"))
    out = attention_jax(q, k, v, DROID_HEADS)
    return linear_jax(out, get_jax_weight_t(weights, f"{prefix}.o.weight"),
                      get_jax_weight(weights, f"{prefix}.o.bias"))


def _dit_cross_attn_jax(
    h: jax.Array, context: jax.Array,
    weights: dict[str, np.ndarray], prefix: str,
) -> jax.Array:
    q = rmsnorm_jax(
        linear_jax(h, get_jax_weight_t(weights, f"{prefix}.q.weight"),
                   get_jax_weight(weights, f"{prefix}.q.bias")),
        get_jax_weight(weights, f"{prefix}.norm_q.weight"))
    k = rmsnorm_jax(
        linear_jax(context, get_jax_weight_t(weights, f"{prefix}.k.weight"),
                   get_jax_weight(weights, f"{prefix}.k.bias")),
        get_jax_weight(weights, f"{prefix}.norm_k.weight"))
    v = linear_jax(context, get_jax_weight_t(weights, f"{prefix}.v.weight"),
                   get_jax_weight(weights, f"{prefix}.v.bias"))
    out = attention_jax(q, k, v, DROID_HEADS)
    return linear_jax(out, get_jax_weight_t(weights, f"{prefix}.o.weight"),
                      get_jax_weight(weights, f"{prefix}.o.bias"))


# ---------------------------------------------------------------------------
# Component registry
# ---------------------------------------------------------------------------

COMPONENTS = {
    "rmsnorm": run_rmsnorm_test,
    "linear": run_linear_test,
    "attention": run_attention_test,
    "adaln": run_adaln_test,
    "ffn": run_ffn_test,
    "cross_attention": run_cross_attention_test,
    "time_embedding": run_time_embedding_test,
    "text_encoder_block": run_text_encoder_block_test,
    "text_embedding": run_text_embedding_test,
    "dit_block": run_dit_block_test,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Standalone JAX forward pass for DreamZero verification.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--checkpoint-dir", type=Path, default=None)
    p.add_argument("--output", type=str, default="jax_standalone_outputs.npz")
    p.add_argument("--components", type=str, default=None,
                   help="Comma-separated subset of components to run.")
    p.add_argument("--list-components", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def _resolve_components(spec: str | None) -> list[str]:
    if not spec:
        return list(COMPONENTS.keys())
    names = [c.strip() for c in spec.split(",")]
    for c in names:
        if c not in COMPONENTS:
            print(f"ERROR: Unknown component '{c}'. "
                  f"Available: {', '.join(COMPONENTS.keys())}")
            sys.exit(1)
    return names


def _run_component(
    name: str, weights: dict[str, np.ndarray], verbose: bool,
) -> tuple[dict[str, np.ndarray], float, str | None]:
    t1 = time.time()
    try:
        outputs = COMPONENTS[name](weights)
        elapsed = time.time() - t1
        has_nan = any(np.any(np.isnan(v)) for v in outputs.values())
        print(f"  OK ({elapsed:.3f}s, {len(outputs)} arrays)")
        if has_nan:
            print("  WARNING: NaN detected in outputs!")
        if verbose:
            for arr_name, arr in outputs.items():
                print(f"    {arr_name:<35} shape={str(arr.shape):<20} "
                      f"mean={arr.mean():>10.6f}  std={arr.std():>10.6f}")
        return outputs, elapsed, None
    except KeyError as e:
        return {}, time.time() - t1, f"missing weight: {e}"
    except Exception as e:
        return {}, time.time() - t1, str(e)


def main():
    args = parse_args()

    if args.list_components:
        print("Available components:")
        for name in COMPONENTS:
            print(f"  {name}")
        return

    if args.checkpoint_dir is None:
        print("ERROR: --checkpoint-dir is required (unless using --list-components)")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"  Standalone JAX Forward Pass")
    print(f"  checkpoint: {args.checkpoint_dir}")
    print(f"  output:     {args.output}")
    print(f"  seed:       {SEED}")
    print(f"  backend:    {jax.default_backend()}")
    print(f"  matmul precision: float32")
    print(f"{'=' * 60}\n")

    t0 = time.time()
    print("Loading checkpoint...")
    weights = load_sharded_safetensors(args.checkpoint_dir)
    weights = strip_prefix(weights, "action_head.")
    print(f"  {len(weights)} tensors loaded ({time.time() - t0:.1f}s)")

    comp_names = _resolve_components(args.components)
    all_outputs: dict[str, np.ndarray] = {}
    passed, failed = 0, 0

    for name in comp_names:
        print(f"\n--- {name} ---")
        outputs, elapsed, err = _run_component(name, weights, args.verbose)
        if err:
            failed += 1
            print(f"  SKIPPED - {err} ({elapsed:.3f}s)")
        else:
            passed += 1
            all_outputs.update(outputs)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(output_path), **all_outputs)

    print(f"\n{'=' * 60}")
    print(f"  Saved {len(all_outputs)} arrays to {output_path}")
    print(f"  Components: {passed} passed, {failed} failed")
    print(f"  Total time: {time.time() - t0:.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
