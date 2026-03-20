"""INT8 weight-only quantization for inference.

Post-training quantization (PTQ) that replaces nnx.Linear weight matrices
with INT8 weights + per-channel bf16 scales. Activations remain in bf16.

Memory savings: bf16 weights (2 bytes/param) -> int8 + scale (~1.0 bytes/param),
roughly halving weight memory for large matmuls.

Usage::

    model = WanDiT(...)
    model = quantize_model(model)  # replaces Linear weights in-place
"""

from __future__ import annotations

import logging
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

logger = logging.getLogger(__name__)


class QuantizedLinear(nnx.Module):
    """Drop-in replacement for nnx.Linear with INT8 weight storage.

    Stores weights as ``(int8_kernel, scales)`` and dequantizes on the fly
    during the forward pass.  Per-channel (per-output-feature) symmetric
    quantization: ``float_weight = int8_weight * scale``.
    """

    def __init__(
        self,
        kernel_i8: jax.Array,
        scales: jax.Array,
        bias: jax.Array | None,
        dtype: jnp.dtype,
    ):
        self.kernel_i8 = nnx.Param(kernel_i8)
        self.scales = nnx.Param(scales)
        self.bias = nnx.Param(bias) if bias is not None else None
        self.dtype = dtype

    def __call__(self, x: jax.Array) -> jax.Array:
        w = self.kernel_i8.value.astype(self.dtype) * self.scales.value
        y = (x @ w).astype(self.dtype)
        if self.bias is not None:
            y = y + self.bias.value.astype(self.dtype)
        return y


def _quantize_kernel(
    kernel: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Symmetric per-channel INT8 quantization of a 2D weight matrix.

    Args:
        kernel: Weight matrix of shape ``(in_features, out_features)``.

    Returns:
        ``(kernel_int8, scales)`` where ``kernel_int8`` has dtype int8 and
        ``scales`` has shape ``(1, out_features)`` in the original dtype.
    """
    abs_max = jnp.max(jnp.abs(kernel), axis=0, keepdims=True)
    # Avoid division by zero for zero columns
    abs_max = jnp.maximum(abs_max, 1e-12)
    scale = abs_max / 127.0
    kernel_i8 = jnp.round(kernel / scale).astype(jnp.int8)
    return kernel_i8, scale.astype(kernel.dtype)


def _quantize_linear(linear: nnx.Linear) -> QuantizedLinear:
    """Convert a single nnx.Linear to a QuantizedLinear."""
    kernel = linear.kernel.value
    kernel_i8, scales = _quantize_kernel(kernel)

    bias = None
    if hasattr(linear, "bias") and linear.bias is not None:
        bias = linear.bias.value

    return QuantizedLinear(
        kernel_i8=kernel_i8,
        scales=scales,
        bias=bias,
        dtype=kernel.dtype,
    )


# Minimum parameter count to quantize (skip tiny projections)
_MIN_PARAMS_FOR_QUANTIZE = 4096


def _should_quantize(name: str, linear: nnx.Linear) -> bool:
    """Decide whether a Linear layer should be quantized.

    Skip embeddings, norms, conv projections, and very small layers.
    """
    kernel = linear.kernel.value
    if kernel.ndim != 2:
        return False
    if kernel.shape[0] * kernel.shape[1] < _MIN_PARAMS_FOR_QUANTIZE:
        return False

    skip_patterns = ("embed", "norm", "conv", "patch_embedding")
    name_lower = name.lower()
    return not any(pat in name_lower for pat in skip_patterns)


def _walk_and_quantize(
    module: nnx.Module,
    prefix: str = "",
) -> tuple[int, int]:
    """Recursively walk module tree and replace Linear layers in-place.

    Returns (num_quantized, num_skipped) counts.
    """
    quantized = 0
    skipped = 0

    for attr_name in list(vars(module)):
        child = getattr(module, attr_name)
        full_name = f"{prefix}.{attr_name}" if prefix else attr_name

        if isinstance(child, nnx.Linear):
            if _should_quantize(full_name, child):
                setattr(module, attr_name, _quantize_linear(child))
                quantized += 1
            else:
                skipped += 1

        elif isinstance(child, nnx.List):
            for i, item in enumerate(child):
                if isinstance(item, nnx.Module):
                    q, s = _walk_and_quantize(item, f"{full_name}.{i}")
                    quantized += q
                    skipped += s

        elif isinstance(child, nnx.Module):
            q, s = _walk_and_quantize(child, full_name)
            quantized += q
            skipped += s

    return quantized, skipped


def quantize_model(model: nnx.Module) -> nnx.Module:
    """Apply INT8 weight quantization to all eligible Linear layers.

    Walks the model tree and replaces each ``nnx.Linear`` whose weight
    matrix exceeds ``_MIN_PARAMS_FOR_QUANTIZE`` with a
    :class:`QuantizedLinear` that stores int8 weights + per-channel scales.

    Layers that are skipped: embeddings, norms, conv projections, and
    any Linear with fewer than 4096 parameters.

    Args:
        model: A Flax NNX model (modified in-place).

    Returns:
        The same model reference (modified in-place).
    """
    import gc
    quantized, skipped = _walk_and_quantize(model)
    gc.collect()
    logger.info(
        "INT8 quantization: %d layers quantized, %d skipped",
        quantized, skipped,
    )
    return model


def estimate_memory_savings(model: nnx.Module) -> dict[str, Any]:
    """Estimate memory before/after quantization without modifying the model.

    Returns dict with ``original_bytes``, ``quantized_bytes``, ``savings_bytes``,
    ``savings_pct``, ``num_eligible``, ``num_ineligible``.
    """
    original_bytes = 0
    quantized_bytes = 0
    eligible = 0
    ineligible = 0

    def _walk(mod: nnx.Module, prefix: str = "") -> None:
        nonlocal original_bytes, quantized_bytes, eligible, ineligible
        for attr_name in list(vars(mod)):
            child = getattr(mod, attr_name)
            full_name = f"{prefix}.{attr_name}" if prefix else attr_name

            if isinstance(child, nnx.Linear):
                kernel = child.kernel.value
                nbytes = kernel.size * kernel.dtype.itemsize
                original_bytes += nbytes

                if _should_quantize(full_name, child):
                    # int8 weights + bf16 scales (1 per output channel)
                    int8_bytes = kernel.size * 1
                    scale_bytes = kernel.shape[-1] * 2
                    quantized_bytes += int8_bytes + scale_bytes
                    eligible += 1
                else:
                    quantized_bytes += nbytes
                    ineligible += 1

            elif isinstance(child, nnx.List):
                for i, item in enumerate(child):
                    if isinstance(item, nnx.Module):
                        _walk(item, f"{full_name}.{i}")

            elif isinstance(child, nnx.Module):
                _walk(child, full_name)

    _walk(model)
    savings = original_bytes - quantized_bytes
    pct = 100.0 * savings / max(original_bytes, 1)

    return {
        "original_bytes": original_bytes,
        "quantized_bytes": quantized_bytes,
        "savings_bytes": savings,
        "savings_pct": pct,
        "num_eligible": eligible,
        "num_ineligible": ineligible,
    }
