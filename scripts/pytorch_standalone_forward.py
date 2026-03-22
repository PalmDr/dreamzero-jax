#!/usr/bin/env python3
"""Standalone PyTorch forward pass for DreamZero weight verification.

Loads safetensors checkpoint shards and runs individual model components
(text encoder, DiT block, attention, RMSNorm, AdaLN) through pure PyTorch
reimplementations. No GR00T framework needed -- only torch + safetensors.

Saves per-component outputs as .npz for comparison against JAX.

Usage
-----
Run all components::

    python scripts/pytorch_standalone_forward.py \
        --checkpoint-dir /path/to/DreamZero-DROID \
        --output pt_standalone_outputs.npz

Run specific components::

    python scripts/pytorch_standalone_forward.py \
        --checkpoint-dir /path/to/DreamZero-DROID \
        --components rmsnorm,linear,attention \
        --output pt_standalone_outputs.npz
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from pt_standalone_ops import (
    DROID_DIM,
    DROID_FREQ_DIM,
    DROID_HEADS,
    DROID_TEXT_DIM,
    DROID_TEXT_HEADS,
    SEED,
    attention_forward,
    gelu_approx,
    get_weight,
    linear_forward,
    load_gate_weight,
    load_sharded_safetensors,
    rmsnorm_forward,
    sinusoidal_embedding_pt,
    strip_prefix,
    t5_attention,
)


# ---------------------------------------------------------------------------
# Component test runners
# ---------------------------------------------------------------------------


def run_rmsnorm_test(weights):
    """Test RMSNorm with real text encoder norm weights."""
    import torch
    torch.manual_seed(SEED)
    B, L, D = 1, 16, DROID_TEXT_DIM
    scale = get_weight(weights, "text_encoder.norm.weight")
    x = torch.randn(B, L, D)
    out = rmsnorm_forward(x, scale)
    return {
        "rmsnorm_input": x.numpy(),
        "rmsnorm_scale": scale.numpy(),
        "rmsnorm_output": out.numpy(),
    }


def run_linear_test(weights):
    """Test linear projection with real DiT block weights."""
    import torch
    torch.manual_seed(SEED)
    B, L, D = 1, 16, DROID_DIM
    w = get_weight(weights, "model.blocks.0.self_attn.q.weight")
    b = get_weight(weights, "model.blocks.0.self_attn.q.bias")
    x = torch.randn(B, L, D)
    out = linear_forward(x, w, b)
    return {
        "linear_input": x.numpy(),
        "linear_weight": w.numpy(),
        "linear_bias": b.numpy(),
        "linear_output": out.numpy(),
    }


def run_attention_test(weights):
    """Test full self-attention (QKV project + QK norm + softmax + output)."""
    import torch
    torch.manual_seed(SEED)
    B, L, D = 1, 16, DROID_DIM
    prefix = "model.blocks.0.self_attn"

    x = torch.randn(B, L, D)
    q = linear_forward(x, get_weight(weights, f"{prefix}.q.weight"),
                        get_weight(weights, f"{prefix}.q.bias"))
    k = linear_forward(x, get_weight(weights, f"{prefix}.k.weight"),
                        get_weight(weights, f"{prefix}.k.bias"))
    v = linear_forward(x, get_weight(weights, f"{prefix}.v.weight"),
                        get_weight(weights, f"{prefix}.v.bias"))
    q = rmsnorm_forward(q, get_weight(weights, f"{prefix}.norm_q.weight"))
    k = rmsnorm_forward(k, get_weight(weights, f"{prefix}.norm_k.weight"))
    attn_out = attention_forward(q, k, v, DROID_HEADS)
    out = linear_forward(attn_out, get_weight(weights, f"{prefix}.o.weight"),
                          get_weight(weights, f"{prefix}.o.bias"))
    return {
        "attn_input": x.numpy(),
        "attn_q_proj": q.numpy(),
        "attn_k_proj": k.numpy(),
        "attn_v_proj": v.numpy(),
        "attn_raw_output": attn_out.numpy(),
        "attn_output": out.numpy(),
    }


def run_adaln_test(weights):
    """Test AdaLN modulation: norm(x) * (1 + scale) + shift."""
    import torch
    torch.manual_seed(SEED)
    B, L, D = 1, 16, DROID_DIM
    mod_param = get_weight(weights, "model.blocks.0.modulation")
    x = torch.randn(B, L, D)
    e = torch.randn(B, 6, D)

    mod = mod_param + e
    shift_msa, scale_msa = mod[:, 0], mod[:, 1]
    x_normed = torch.layer_norm(x, [D])
    x_modulated = x_normed * (1 + scale_msa[:, None, :]) + shift_msa[:, None, :]
    return {
        "adaln_input": x.numpy(),
        "adaln_e": e.numpy(),
        "adaln_modulation_param": mod_param.numpy(),
        "adaln_normed": x_normed.numpy(),
        "adaln_output": x_modulated.numpy(),
    }


def run_ffn_test(weights):
    """Test FFN: Linear(dim->ffn_dim) -> GELU(tanh) -> Linear(ffn_dim->dim)."""
    import torch
    torch.manual_seed(SEED)
    B, L, D = 1, 16, DROID_DIM
    prefix = "model.blocks.0.ffn"
    x = torch.randn(B, L, D)
    h = gelu_approx(linear_forward(x, get_weight(weights, f"{prefix}.0.weight"),
                                     get_weight(weights, f"{prefix}.0.bias")))
    out = linear_forward(h, get_weight(weights, f"{prefix}.2.weight"),
                          get_weight(weights, f"{prefix}.2.bias"))
    return {"ffn_input": x.numpy(), "ffn_hidden": h.numpy(), "ffn_output": out.numpy()}


def run_cross_attention_test(weights):
    """Test cross-attention with text context (no image branch)."""
    import torch
    torch.manual_seed(SEED)
    B, S, D, L_ctx = 1, 16, DROID_DIM, 32
    prefix = "model.blocks.0.cross_attn"

    x = torch.randn(B, S, D)
    context = torch.randn(B, L_ctx, D)
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
    attn_out = attention_forward(q, k, v, DROID_HEADS)
    out = linear_forward(attn_out, get_weight(weights, f"{prefix}.o.weight"),
                          get_weight(weights, f"{prefix}.o.bias"))
    return {
        "cross_attn_x": x.numpy(),
        "cross_attn_context": context.numpy(),
        "cross_attn_output": out.numpy(),
    }


def run_time_embedding_test(weights):
    """Test time embedding: sinusoidal -> MLP -> SiLU -> linear projection."""
    import torch
    torch.manual_seed(SEED)

    timestep = torch.tensor([500.0])
    sin_emb = sinusoidal_embedding_pt(timestep, DROID_FREQ_DIM)
    h = torch.nn.functional.silu(
        linear_forward(sin_emb, get_weight(weights, "model.time_embedding.0.weight"),
                        get_weight(weights, "model.time_embedding.0.bias")))
    t_emb = linear_forward(h, get_weight(weights, "model.time_embedding.2.weight"),
                            get_weight(weights, "model.time_embedding.2.bias"))
    e = torch.nn.functional.silu(t_emb)
    e_proj = linear_forward(e, get_weight(weights, "model.time_projection.1.weight"),
                             get_weight(weights, "model.time_projection.1.bias"))
    return {
        "time_sinusoidal": sin_emb.numpy(),
        "time_mlp_output": t_emb.numpy(),
        "time_modulation_6x": e_proj.reshape(1, 6, DROID_DIM).numpy(),
    }


def run_text_encoder_block_test(weights):
    """Test one T5 self-attention block from the text encoder."""
    import torch
    torch.manual_seed(SEED)
    B, L, D = 1, 16, DROID_TEXT_DIM
    blk = "text_encoder.blocks.0"

    x = torch.randn(B, L, D)

    h = rmsnorm_forward(x, get_weight(weights, f"{blk}.norm1.weight"))
    q = linear_forward(h, get_weight(weights, f"{blk}.attn.q.weight"))
    k = linear_forward(h, get_weight(weights, f"{blk}.attn.k.weight"))
    v = linear_forward(h, get_weight(weights, f"{blk}.attn.v.weight"))
    sa_out = linear_forward(t5_attention(q, k, v, DROID_TEXT_HEADS),
                             get_weight(weights, f"{blk}.attn.o.weight"))
    x = x + sa_out

    h2 = rmsnorm_forward(x, get_weight(weights, f"{blk}.norm2.weight"))
    gate_val = gelu_approx(linear_forward(h2, load_gate_weight(weights, f"{blk}.ffn.gate")))
    fc1_val = linear_forward(h2, get_weight(weights, f"{blk}.ffn.fc1.weight"))
    ffn_out = linear_forward(fc1_val * gate_val,
                              get_weight(weights, f"{blk}.ffn.fc2.weight"))
    x = x + ffn_out
    return {
        "text_block_sa_output": sa_out.numpy(),
        "text_block_ffn_output": ffn_out.numpy(),
        "text_block_output": x.numpy(),
    }


def run_text_embedding_test(weights):
    """Test text embedding projection: Linear -> GELU -> Linear."""
    import torch
    torch.manual_seed(SEED)
    B, L = 1, 16
    x = torch.randn(B, L, DROID_TEXT_DIM)
    h = gelu_approx(linear_forward(x, get_weight(weights, "model.text_embedding.0.weight"),
                                     get_weight(weights, "model.text_embedding.0.bias")))
    out = linear_forward(h, get_weight(weights, "model.text_embedding.2.weight"),
                          get_weight(weights, "model.text_embedding.2.bias"))
    return {"text_emb_proj_input": x.numpy(), "text_emb_proj_output": out.numpy()}


def run_dit_block_test(weights):
    """Test a full DiT block: AdaLN self-attn + cross-attn + FFN.

    Skips RoPE to isolate the algebraic operations for parity checking.
    """
    import torch
    torch.manual_seed(SEED)
    B, S, D, L_ctx = 1, 16, DROID_DIM, 8
    blk = "model.blocks.0"

    x = torch.randn(B, S, D)
    e = torch.randn(B, 6, D)
    context = torch.randn(B, L_ctx, D)

    mod = get_weight(weights, f"{blk}.modulation") + e
    shift_msa, scale_msa, gate_msa = mod[:, 0], mod[:, 1], mod[:, 2]
    shift_mlp, scale_mlp, gate_mlp = mod[:, 3], mod[:, 4], mod[:, 5]

    h = torch.layer_norm(x, [D]) * (1 + scale_msa[:, None, :]) + shift_msa[:, None, :]
    sa = _dit_self_attn(h, weights, f"{blk}.self_attn")
    x = x + sa * gate_msa[:, None, :]

    n3_w = get_weight(weights, f"{blk}.norm3.weight")
    n3_b = get_weight(weights, f"{blk}.norm3.bias")
    h = torch.nn.functional.layer_norm(x, [D], n3_w, n3_b)
    ca = _dit_cross_attn(h, context, weights, f"{blk}.cross_attn")
    x = x + ca

    h = torch.layer_norm(x, [D]) * (1 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]
    h = gelu_approx(linear_forward(h, get_weight(weights, f"{blk}.ffn.0.weight"),
                                     get_weight(weights, f"{blk}.ffn.0.bias")))
    h = linear_forward(h, get_weight(weights, f"{blk}.ffn.2.weight"),
                        get_weight(weights, f"{blk}.ffn.2.bias"))
    x = x + h * gate_mlp[:, None, :]

    return {"dit_block_output": x.numpy()}


def _dit_self_attn(h, weights, prefix):
    """Self-attention sub-block of DiT (no RoPE)."""
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
    out = attention_forward(q, k, v, DROID_HEADS)
    return linear_forward(out, get_weight(weights, f"{prefix}.o.weight"),
                           get_weight(weights, f"{prefix}.o.bias"))


def _dit_cross_attn(h, context, weights, prefix):
    """Cross-attention sub-block of DiT (text-only path, no image)."""
    q = rmsnorm_forward(
        linear_forward(h, get_weight(weights, f"{prefix}.q.weight"),
                        get_weight(weights, f"{prefix}.q.bias")),
        get_weight(weights, f"{prefix}.norm_q.weight"))
    k = rmsnorm_forward(
        linear_forward(context, get_weight(weights, f"{prefix}.k.weight"),
                        get_weight(weights, f"{prefix}.k.bias")),
        get_weight(weights, f"{prefix}.norm_k.weight"))
    v = linear_forward(context, get_weight(weights, f"{prefix}.v.weight"),
                        get_weight(weights, f"{prefix}.v.bias"))
    out = attention_forward(q, k, v, DROID_HEADS)
    return linear_forward(out, get_weight(weights, f"{prefix}.o.weight"),
                           get_weight(weights, f"{prefix}.o.bias"))


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
        description="Standalone PyTorch forward pass for DreamZero verification.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--checkpoint-dir", type=Path, default=None)
    p.add_argument("--output", type=str, default="pt_standalone_outputs.npz")
    p.add_argument("--components", type=str, default=None,
                   help="Comma-separated subset of components to run.")
    p.add_argument("--list-components", action="store_true")
    p.add_argument("--list-keys", action="store_true",
                   help="Print checkpoint key structure and exit.")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def _print_checkpoint_structure(weights):
    """Print checkpoint key hierarchy with shapes."""
    prefixes: dict[str, list[str]] = {}
    for key in sorted(weights.keys()):
        top = key.split(".")[0]
        prefixes.setdefault(top, []).append(key)

    total = sum(w.size for w in weights.values())
    print(f"\nCheckpoint: {len(weights)} tensors, {total / 1e9:.2f}B params")
    for prefix in sorted(prefixes.keys()):
        keys = prefixes[prefix]
        params = sum(weights[k].size for k in keys)
        print(f"  {prefix}: {len(keys)} tensors, {params / 1e6:.1f}M params")
        for k in keys[:3]:
            print(f"    {k}: {weights[k].shape} {weights[k].dtype}")
        if len(keys) > 3:
            print(f"    ... ({len(keys) - 3} more)")


def _run_component(name, weights, verbose):
    """Run one component and return (outputs_dict, elapsed, error_str)."""
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
    print(f"  Standalone PyTorch Forward Pass")
    print(f"  checkpoint: {args.checkpoint_dir}")
    print(f"  output:     {args.output}")
    print(f"  seed:       {SEED}")
    print(f"{'=' * 60}\n")

    t0 = time.time()
    print("Loading checkpoint...")
    weights = load_sharded_safetensors(args.checkpoint_dir)
    weights = strip_prefix(weights, "action_head.")
    print(f"  {len(weights)} tensors loaded ({time.time() - t0:.1f}s)")

    if args.list_keys:
        _print_checkpoint_structure(weights)
        return

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


def _resolve_components(spec: str | None) -> list[str]:
    """Parse --components flag or return all."""
    if not spec:
        return list(COMPONENTS.keys())
    names = [c.strip() for c in spec.split(",")]
    for c in names:
        if c not in COMPONENTS:
            print(f"ERROR: Unknown component '{c}'. "
                  f"Available: {', '.join(COMPONENTS.keys())}")
            sys.exit(1)
    return names


if __name__ == "__main__":
    main()
