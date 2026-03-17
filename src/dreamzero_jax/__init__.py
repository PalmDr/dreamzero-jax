"""DreamZero-JAX: JAX/Flax NNX implementation of DreamZero."""

__version__ = "0.1.0"

# Flax 0.10.x compat: patch nnx.List if missing
from flax import nnx as _nnx

if not hasattr(_nnx, "List"):
    from dreamzero_jax.nn.compat import List as _List
    _nnx.List = _List
