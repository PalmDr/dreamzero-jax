"""TPU/GPU mesh and sharding utilities for DreamZero.

Provides utilities for distributed inference and training of the 14B parameter
DreamZero model across TPU pods or multi-GPU setups.

Key concepts:
    - **Mesh**: 2D device grid with axes ``('data', 'model')``
    - **Data parallelism** (``'data'`` axis): replicates model, shards batch
    - **Tensor parallelism** (``'model'`` axis): shards model weights

Sharding strategy for the DiT backbone:
    - Attention Q/K/V/out projections: shard heads across ``'model'``
    - FFN up-projection: shard output dim across ``'model'``
    - FFN down-projection: shard input dim across ``'model'``
    - Embeddings, norms, biases, convolutions: replicated
    - Batch dimension: sharded across ``'data'``
"""

from __future__ import annotations

import logging
import re
from typing import Any

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from flax import nnx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mesh creation
# ---------------------------------------------------------------------------

# Standard axis names used throughout the project.
DATA_AXIS = "data"
MODEL_AXIS = "model"
AXIS_NAMES = (DATA_AXIS, MODEL_AXIS)


def _infer_mesh_shape(num_devices: int) -> tuple[int, int]:
    """Infer a (data, model) mesh shape from the number of available devices.

    Heuristic: prefer more model parallelism up to 8 (typical tensor-parallel
    degree for transformer models), then scale data parallelism.

    Examples:
        1 device  -> (1, 1)
        4 devices -> (1, 4)
        8 devices -> (1, 8)
        16 devices -> (2, 8)
        32 devices -> (4, 8)
        64 devices -> (8, 8)

    For non-power-of-2 counts, find the largest model-parallel degree that
    evenly divides the device count (up to 8).

    Args:
        num_devices: Total number of available devices.

    Returns:
        ``(data_parallel, model_parallel)`` tuple.
    """
    if num_devices <= 0:
        raise ValueError(f"num_devices must be positive, got {num_devices}")

    if num_devices == 1:
        return (1, 1)

    # Find largest model-parallel degree <= 8 that divides device count
    max_mp = min(8, num_devices)
    model_parallel = 1
    for mp in range(max_mp, 0, -1):
        if num_devices % mp == 0:
            model_parallel = mp
            break

    data_parallel = num_devices // model_parallel
    return (data_parallel, model_parallel)


def create_mesh(
    mesh_shape: tuple[int, int] | None = None,
    axis_names: tuple[str, str] = AXIS_NAMES,
) -> Mesh:
    """Create a device mesh for distributed execution.

    Auto-detects the available devices and creates a 2D mesh with data and
    model parallelism axes.

    Args:
        mesh_shape: Explicit ``(data, model)`` shape. If ``None``, inferred
            from the device count using :func:`_infer_mesh_shape`.
        axis_names: Axis names for the mesh. Defaults to ``('data', 'model')``.

    Returns:
        A :class:`jax.sharding.Mesh` instance.

    Raises:
        ValueError: If the mesh shape doesn't match the device count.
    """
    devices = jax.devices()
    num_devices = len(devices)

    if mesh_shape is None:
        mesh_shape = _infer_mesh_shape(num_devices)

    dp, mp = mesh_shape
    if dp * mp != num_devices:
        raise ValueError(
            f"Mesh shape {mesh_shape} requires {dp * mp} devices, "
            f"but {num_devices} device(s) available."
        )

    device_array = np.array(devices).reshape(mesh_shape)
    mesh = Mesh(device_array, axis_names=axis_names)

    logger.info(
        "Created mesh: shape=%s, axes=%s, platform=%s, devices=%d",
        mesh_shape,
        axis_names,
        jax.default_backend(),
        num_devices,
    )
    return mesh


# ---------------------------------------------------------------------------
# PartitionSpec helpers
# ---------------------------------------------------------------------------

# Replicated across all devices.
REPLICATED = P()

# Sharded on first axis (batch) across data parallelism.
DATA_PARALLEL = P(DATA_AXIS)

# Common weight sharding specs.
# Dense kernel sharded on output dim (for Q/K/V/FFN-up): (in, out) -> (None, 'model')
SHARD_OUTPUT = P(None, MODEL_AXIS)

# Dense kernel sharded on input dim (for out-proj/FFN-down): (in, out) -> ('model', None)
SHARD_INPUT = P(MODEL_AXIS, None)


# ---------------------------------------------------------------------------
# Name-based partition spec inference
# ---------------------------------------------------------------------------

# Patterns that indicate a parameter should be sharded on its output (last) dim.
# These are Q/K/V projections and FFN up-projections.
_SHARD_OUTPUT_PATTERNS = (
    r"\.q_proj\.kernel$",
    r"\.k_proj\.kernel$",
    r"\.v_proj\.kernel$",
    r"\.k_img\.kernel$",
    r"\.v_img\.kernel$",
    r"\.w_up\.kernel$",
    r"\.w_gate\.kernel$",
    r"\.linear1\.kernel$",     # MLPProj, MLP in time_embedding/text_embedding
    r"\.time_projection\.kernel$",
)

# Patterns that indicate a parameter should be sharded on its input (first) dim.
# These are output projections and FFN down-projections.
_SHARD_INPUT_PATTERNS = (
    r"\.out_proj\.kernel$",
    r"\.w_down\.kernel$",
    r"\.linear2\.kernel$",     # MLPProj
)

# Patterns that should always be replicated regardless of shape.
_REPLICATE_PATTERNS = (
    r"\.bias$",
    r"\.scale$",
    r"norm",                   # LayerNorm, RMSNorm
    r"\.modulation$",          # DiT modulation parameters
    r"embed",                  # Embeddings (patch_embedding, text_embedding, etc.)
    r"conv",                   # Conv kernels (relatively small)
    r"CategorySpecific",       # Multi-embodiment weights (small)
    r"action_encoder",         # Action encoder weights (small)
    r"state_encoder",          # State encoder weights (small)
    r"action_decoder",         # Action decoder weights (small)
)


def _matches_any(name: str, patterns: tuple[str, ...]) -> bool:
    """Check if a parameter path matches any of the given regex patterns."""
    return any(re.search(pat, name) for pat in patterns)


def get_partition_spec(
    param_path: str,
    param_shape: tuple[int, ...],
    mesh: Mesh,
) -> P:
    """Determine the partition spec for a single parameter.

    Uses a combination of name-based pattern matching and shape heuristics:

    1. Parameters matching known replicate patterns are replicated.
    2. 2D parameters matching Q/K/V/FFN-up patterns are sharded on last dim.
    3. 2D parameters matching out-proj/FFN-down patterns are sharded on first dim.
    4. All other parameters are replicated (safe default).

    Additionally, sharding is only applied if the sharded dimension is evenly
    divisible by the model-parallel degree. If not, the parameter is replicated
    with a logged warning.

    Args:
        param_path: Dot-separated path to the parameter (e.g.,
            ``'blocks.0.self_attn.q_proj.kernel'``).
        param_shape: Shape of the parameter array.
        mesh: The device mesh to check divisibility against.

    Returns:
        A :class:`PartitionSpec` for the parameter.
    """
    model_size = mesh.shape[MODEL_AXIS]

    # 1. Always replicate certain patterns (norms, biases, embeddings, etc.)
    if _matches_any(param_path, _REPLICATE_PATTERNS):
        return REPLICATED

    # Only shard 2D weight matrices (Dense kernels).
    if len(param_shape) != 2:
        return REPLICATED

    # 2. Shard output dim for Q/K/V and FFN up-projection
    if _matches_any(param_path, _SHARD_OUTPUT_PATTERNS):
        if param_shape[-1] % model_size == 0:
            return P(None, MODEL_AXIS)
        logger.warning(
            "Cannot shard %s (shape %s) on output dim: not divisible by model_size=%d. "
            "Replicating.",
            param_path, param_shape, model_size,
        )
        return REPLICATED

    # 3. Shard input dim for output projections and FFN down-projection
    if _matches_any(param_path, _SHARD_INPUT_PATTERNS):
        if param_shape[0] % model_size == 0:
            return P(MODEL_AXIS, None)
        logger.warning(
            "Cannot shard %s (shape %s) on input dim: not divisible by model_size=%d. "
            "Replicating.",
            param_path, param_shape, model_size,
        )
        return REPLICATED

    # 4. Fallback: large 2D matrices not matched by name patterns.
    # Use shape heuristic: shard if both dims are >= 1024 (typical for
    # transformer weight matrices).
    if param_shape[0] >= 1024 and param_shape[-1] >= 1024:
        # Default to sharding on the larger dimension.
        if param_shape[-1] >= param_shape[0] and param_shape[-1] % model_size == 0:
            return P(None, MODEL_AXIS)
        elif param_shape[0] % model_size == 0:
            return P(MODEL_AXIS, None)

    # 5. Default: replicate
    return REPLICATED


# ---------------------------------------------------------------------------
# Model parameter sharding
# ---------------------------------------------------------------------------


def _flatten_state_paths(state: nnx.State) -> dict[str, jax.Array]:
    """Flatten an NNX state dict into dot-separated path -> array pairs.

    Args:
        state: Flax NNX state (nested dict-like).

    Returns:
        Flat dict mapping ``'blocks.0.self_attn.q_proj.kernel'`` style
        paths to parameter arrays.
    """
    flat: dict[str, jax.Array] = {}
    raw = state.flat_state()
    for key_tuple, value in raw.items():
        path = ".".join(str(k) for k in key_tuple)
        if hasattr(value, "value"):
            flat[path] = value.value
        else:
            flat[path] = value
    return flat


def compute_param_shardings(
    model: nnx.Module,
    mesh: Mesh,
) -> dict[str, NamedSharding]:
    """Compute sharding specs for all parameters in a Flax NNX model.

    Traverses the model's parameter state and assigns a
    :class:`NamedSharding` to each parameter based on its name and shape.

    Args:
        model: A Flax NNX model instance.
        mesh: The device mesh.

    Returns:
        Dict mapping parameter path strings to :class:`NamedSharding` objects.
    """
    graphdef, state = nnx.split(model)
    flat = _flatten_state_paths(state)

    shardings: dict[str, NamedSharding] = {}
    for path, array in flat.items():
        pspec = get_partition_spec(path, array.shape, mesh)
        shardings[path] = NamedSharding(mesh, pspec)

    return shardings


def shard_params(
    model: nnx.Module,
    mesh: Mesh,
) -> nnx.Module:
    """Reshard a model's parameters according to the computed partition specs.

    Extracts the model state, constrains each parameter to its target sharding
    via :func:`jax.device_put`, and merges the sharded state back into the
    model.

    This is typically called once after model initialization or checkpoint
    loading, before entering the ``jit``-compiled train/inference loop.

    Args:
        model: A Flax NNX model instance (may be on a single device).
        mesh: The device mesh.

    Returns:
        The same model with parameters placed on the correct shards.
    """
    graphdef, state = nnx.split(model)
    flat = state.flat_state()

    sharded_flat: dict[tuple, Any] = {}
    for key_tuple, value in flat.items():
        path = ".".join(str(k) for k in key_tuple)
        if hasattr(value, "value"):
            arr = value.value
        else:
            arr = value

        pspec = get_partition_spec(path, arr.shape, mesh)
        sharding = NamedSharding(mesh, pspec)
        sharded_arr = jax.device_put(arr, sharding)

        if hasattr(value, "value"):
            value = value.replace(sharded_arr)
        else:
            value = sharded_arr
        sharded_flat[key_tuple] = value

    state = nnx.State(sharded_flat)
    model = nnx.merge(graphdef, state)
    return model


# ---------------------------------------------------------------------------
# Data (batch) sharding
# ---------------------------------------------------------------------------


def shard_batch(
    batch: dict[str, jax.Array],
    mesh: Mesh,
) -> dict[str, jax.Array]:
    """Shard a data batch across the ``'data'`` axis of the mesh.

    Each array in the batch dict is expected to have a batch dimension as
    its first axis, which gets sharded across the data-parallel devices.

    Args:
        batch: Dictionary mapping field names to arrays, each with a leading
            batch dimension.
        mesh: The device mesh.

    Returns:
        A new dict with the same keys, but arrays placed on the mesh with
        batch sharding.

    Raises:
        ValueError: If any array's batch dimension is not divisible by the
            data-parallel degree.
    """
    data_size = mesh.shape[DATA_AXIS]
    sharded: dict[str, jax.Array] = {}

    for name, arr in batch.items():
        if arr.ndim == 0:
            # Scalar: replicate
            sharded[name] = jax.device_put(arr, NamedSharding(mesh, REPLICATED))
            continue

        if arr.shape[0] % data_size != 0:
            raise ValueError(
                f"Batch field '{name}' has batch size {arr.shape[0]} which is "
                f"not divisible by data-parallel degree {data_size}."
            )

        # Shard first axis across 'data', replicate the rest
        ndim = arr.ndim
        pspec = P(DATA_AXIS, *([None] * (ndim - 1)))
        sharded[name] = jax.device_put(arr, NamedSharding(mesh, pspec))

    return sharded


def shard_array(
    arr: jax.Array,
    mesh: Mesh,
    pspec: P | None = None,
) -> jax.Array:
    """Place a single array on the mesh with a given partition spec.

    Convenience wrapper around :func:`jax.device_put` with
    :class:`NamedSharding`.

    Args:
        arr: The array to shard.
        mesh: The device mesh.
        pspec: Partition spec. Defaults to batch-sharded (first axis on
            ``'data'``).

    Returns:
        The array placed on the mesh according to the partition spec.
    """
    if pspec is None:
        if arr.ndim == 0:
            pspec = REPLICATED
        else:
            pspec = P(DATA_AXIS, *([None] * (arr.ndim - 1)))
    return jax.device_put(arr, NamedSharding(mesh, pspec))


# ---------------------------------------------------------------------------
# Training partition specs (optimizer state, gradients)
# ---------------------------------------------------------------------------


def get_train_pspecs(
    model: nnx.Module,
    mesh: Mesh,
) -> dict[str, dict[str, P]]:
    """Compute partition specs for training state: params, gradients, optimizer.

    The returned dict can be used with ``jax.jit``'s ``in_shardings`` /
    ``out_shardings`` or with ``nnx.Optimizer`` sharding.

    Params and gradients share the same sharding. Optimizer state (e.g.,
    Adam's ``mu`` and ``nu``) matches the parameter sharding since these
    are element-wise accumulators.

    Args:
        model: A Flax NNX model instance.
        mesh: The device mesh.

    Returns:
        Dict with keys ``'params'``, ``'grads'``, ``'opt_state'``, each
        mapping parameter paths to :class:`PartitionSpec` objects.
    """
    graphdef, state = nnx.split(model)
    flat = _flatten_state_paths(state)

    param_pspecs: dict[str, P] = {}
    for path, array in flat.items():
        param_pspecs[path] = get_partition_spec(path, array.shape, mesh)

    # Gradients have the same shape and sharding as parameters.
    # Optimizer state (mu, nu for Adam) is element-wise, so same sharding.
    return {
        "params": param_pspecs,
        "grads": param_pspecs,
        "opt_state": param_pspecs,
    }


# ---------------------------------------------------------------------------
# Sharding constraint for use inside jit-compiled functions
# ---------------------------------------------------------------------------


def with_sharding_constraint(
    x: jax.Array,
    mesh: Mesh,
    pspec: P,
) -> jax.Array:
    """Apply a sharding constraint inside a jit-compiled function.

    This is a thin wrapper around ``jax.lax.with_sharding_constraint`` that
    constructs the :class:`NamedSharding` from a mesh and partition spec.

    Use this to enforce intermediate activation sharding within ``jit``-
    compiled training or inference functions (e.g., to reshard activations
    between data-parallel and model-parallel phases).

    Args:
        x: The array to constrain.
        mesh: The device mesh.
        pspec: Desired partition spec.

    Returns:
        The input array with a sharding constraint applied (identity in
        eager mode, sharding annotation in ``jit``).
    """
    return jax.lax.with_sharding_constraint(x, NamedSharding(mesh, pspec))


# ---------------------------------------------------------------------------
# Summary / debugging
# ---------------------------------------------------------------------------


def log_sharding_plan(model: nnx.Module, mesh: Mesh) -> str:
    """Generate a human-readable summary of the sharding plan for a model.

    Useful for debugging and verifying that the sharding heuristics produce
    the expected partition specs.

    Args:
        model: A Flax NNX model instance.
        mesh: The device mesh.

    Returns:
        Multi-line string summarizing each parameter's path, shape, and
        partition spec.
    """
    graphdef, state = nnx.split(model)
    flat = _flatten_state_paths(state)

    lines = [
        f"Sharding plan for mesh shape={mesh.shape} axes={mesh.axis_names}",
        f"{'Parameter Path':<70} {'Shape':<25} {'PartitionSpec':<20}",
        "-" * 115,
    ]

    total_params = 0
    sharded_params = 0

    for path, array in sorted(flat.items()):
        pspec = get_partition_spec(path, array.shape, mesh)
        size = 1
        for d in array.shape:
            size *= d
        total_params += size
        if pspec != REPLICATED:
            sharded_params += size

        shape_str = str(array.shape)
        lines.append(f"{path:<70} {shape_str:<25} {str(pspec):<20}")

    lines.append("-" * 115)
    lines.append(
        f"Total params: {total_params:,} | "
        f"Sharded params: {sharded_params:,} "
        f"({100 * sharded_params / max(total_params, 1):.1f}%)"
    )

    summary = "\n".join(lines)
    logger.info(summary)
    return summary
