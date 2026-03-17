"""Fused AdaLN Pallas kernel for TPU.

Fuses LayerNorm + affine modulation into a single HBM pass:
    y = (1 + scale) * LayerNorm(x) + shift

Adapted from the proven kernel in tpu_startup/kernels/adaln_large.py (1.81x speedup).

Usage:
    from dreamzero_jax.nn.pallas_ops import fused_adaln_modulate

    # Replaces: norm(x) * (1 + scale[:, None, :]) + shift[:, None, :]
    h = fused_adaln_modulate(x, scale, shift)
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

# Block size for sequence dimension tiling (MXU-aligned)
BLOCK_SEQ = 128


# ---------------------------------------------------------------------------
# Pallas kernel: fused LayerNorm + (1 + scale) * x_norm + shift
# ---------------------------------------------------------------------------

def _adaln_kernel(
    x_ref,      # [1, BLOCK_SEQ, d_model]
    scale_ref,  # [1, 1, d_model]
    shift_ref,  # [1, 1, d_model]
    y_ref,      # [1, BLOCK_SEQ, d_model]  (output)
    *,
    eps: float,
):
    """Per-tile kernel: LayerNorm in f32, modulate, write bf16."""
    x     = x_ref[0].astype(jnp.float32)          # [BLOCK_SEQ, d_model]
    scale = scale_ref[0, 0].astype(jnp.float32)   # [d_model]
    shift = shift_ref[0, 0].astype(jnp.float32)   # [d_model]

    mean   = x.mean(axis=-1, keepdims=True)
    var    = x.var(axis=-1, keepdims=True)
    x_norm = (x - mean) / jnp.sqrt(var + eps)

    y = (1.0 + scale[None, :]) * x_norm + shift[None, :]
    y_ref[0] = y.astype(jnp.bfloat16)


def _fused_adaln_pallas(
    x: jax.Array,      # [batch, seq_len, d_model]  (seq_len must be % BLOCK_SEQ == 0)
    scale: jax.Array,  # [batch, d_model]
    shift: jax.Array,  # [batch, d_model]
    *,
    eps: float = 1e-6,
) -> jax.Array:
    """Dispatch Pallas kernel over (batch, seq_blocks) grid."""
    import jax.experimental.pallas as pl

    batch, seq_len, d_model = x.shape
    assert seq_len % BLOCK_SEQ == 0

    scale3 = scale[:, None, :]  # [batch, 1, d_model]
    shift3 = shift[:, None, :]

    return pl.pallas_call(
        functools.partial(_adaln_kernel, eps=eps),
        grid=(batch, seq_len // BLOCK_SEQ),
        in_specs=[
            pl.BlockSpec((1, BLOCK_SEQ, d_model), lambda b, s: (b, s, 0)),
            pl.BlockSpec((1, 1, d_model),          lambda b, s: (b, 0, 0)),
            pl.BlockSpec((1, 1, d_model),          lambda b, s: (b, 0, 0)),
        ],
        out_specs=pl.BlockSpec((1, BLOCK_SEQ, d_model), lambda b, s: (b, s, 0)),
        out_shape=jax.ShapeDtypeStruct((batch, seq_len, d_model), jnp.bfloat16),
        interpret=False,
    )(x, scale3, shift3)


# ---------------------------------------------------------------------------
# Naive (unfused) reference implementation
# ---------------------------------------------------------------------------

def _naive_adaln_modulate(
    x: jax.Array,
    scale: jax.Array,
    shift: jax.Array,
    *,
    eps: float = 1e-6,
) -> jax.Array:
    """Pure JAX fallback: LayerNorm then modulate (two HBM passes)."""
    x_f32 = x.astype(jnp.float32)
    mean  = x_f32.mean(axis=-1, keepdims=True)
    var   = x_f32.var(axis=-1, keepdims=True)
    x_norm = (x_f32 - mean) / jnp.sqrt(var + eps)
    return ((1.0 + scale[:, None, :]) * x_norm + shift[:, None, :]).astype(jnp.bfloat16)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fused_adaln_modulate(
    x: jax.Array,       # [B, S, D]
    scale: jax.Array,   # [B, D]
    shift: jax.Array,   # [B, D]
    eps: float = 1e-6,
    use_pallas: bool = True,
) -> jax.Array:
    """Fused LayerNorm + AdaLN modulation.

    Computes: ``(1 + scale) * LayerNorm(x) + shift``

    When ``use_pallas=True``, uses a Pallas kernel that fuses the operation
    into a single HBM pass (requires TPU). Handles padding seq_len to the
    nearest multiple of 128 automatically.

    When ``use_pallas=False``, uses a pure JAX fallback (works on any backend).

    Args:
        x: Input tensor ``(B, S, D)``.
        scale: Scale modulation ``(B, D)``.
        shift: Shift modulation ``(B, D)``.
        eps: LayerNorm epsilon.
        use_pallas: Whether to use the Pallas kernel.

    Returns:
        Modulated tensor ``(B, S, D)`` in bf16.
    """
    if not use_pallas:
        return _naive_adaln_modulate(x, scale, shift, eps=eps)

    batch, seq_len, d_model = x.shape

    # Pad seq_len to nearest multiple of BLOCK_SEQ if needed
    remainder = seq_len % BLOCK_SEQ
    if remainder != 0:
        pad_len = BLOCK_SEQ - remainder
        x = jnp.pad(x, ((0, 0), (0, pad_len), (0, 0)))

    y = _fused_adaln_pallas(x, scale, shift, eps=eps)

    # Unpad back to original seq_len
    if remainder != 0:
        y = y[:, :seq_len, :]

    return y


# ---------------------------------------------------------------------------
# Correctness test
# ---------------------------------------------------------------------------

def test_correctness(
    batch: int = 2,
    seq_len: int = 300,   # intentionally not divisible by 128
    d_model: int = 256,
    eps: float = 1e-6,
) -> None:
    """Verify Pallas kernel matches naive implementation."""
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)

    x     = jax.random.normal(k1, (batch, seq_len, d_model), dtype=jnp.bfloat16)
    scale = jax.random.normal(k2, (batch, d_model), dtype=jnp.bfloat16)
    shift = jax.random.normal(k3, (batch, d_model), dtype=jnp.bfloat16)

    naive_out  = fused_adaln_modulate(x, scale, shift, eps=eps, use_pallas=False)
    pallas_out = fused_adaln_modulate(x, scale, shift, eps=eps, use_pallas=True)

    max_err = float(jnp.abs(
        naive_out.astype(jnp.float32) - pallas_out.astype(jnp.float32)
    ).max())
    print(f"Correctness test: batch={batch}, seq_len={seq_len}, d_model={d_model}")
    print(f"  Output shape: {pallas_out.shape}, dtype: {pallas_out.dtype}")
    print(f"  Max abs error: {max_err:.6f}")
    assert max_err < 0.5, f"Correctness check FAILED: max_err={max_err}"
    print("  PASSED.")


if __name__ == "__main__":
    test_correctness()
    # Also test with seq_len divisible by 128
    test_correctness(seq_len=512)
    # Test at Wan2.1 scale dims
    test_correctness(batch=1, seq_len=128, d_model=5120)
