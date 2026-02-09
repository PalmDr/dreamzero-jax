"""Tests for MLP layers."""

import jax
import jax.numpy as jnp
from flax import nnx

from dreamzero_jax.nn.mlp import MLP, SwiGLU, GeGLU


def test_mlp_shape():
    """MLP preserves batch and sequence dims, transforms features."""
    rngs = nnx.Rngs(0)
    mlp = MLP(in_features=64, hidden_features=256, rngs=rngs)
    x = jax.random.normal(jax.random.key(0), (2, 16, 64))
    y = mlp(x)
    assert y.shape == (2, 16, 64)


def test_mlp_with_silu():
    """MLP works with SiLU activation."""
    rngs = nnx.Rngs(0)
    mlp = MLP(in_features=64, hidden_features=256, activation=jax.nn.silu, rngs=rngs)
    x = jax.random.normal(jax.random.key(0), (2, 16, 64))
    y = mlp(x)
    assert y.shape == (2, 16, 64)


def test_swiglu_shape():
    """SwiGLU preserves batch and sequence dims, transforms features."""
    rngs = nnx.Rngs(0)
    mlp = SwiGLU(in_features=64, hidden_features=256, rngs=rngs)
    x = jax.random.normal(jax.random.key(0), (2, 16, 64))
    y = mlp(x)
    assert y.shape == (2, 16, 64)


def test_swiglu_different_out_features():
    """SwiGLU can project to different output dimension."""
    rngs = nnx.Rngs(0)
    mlp = SwiGLU(in_features=64, hidden_features=256, out_features=128, rngs=rngs)
    x = jax.random.normal(jax.random.key(0), (2, 16, 64))
    y = mlp(x)
    assert y.shape == (2, 16, 128)


def test_geglu_shape():
    """GeGLU preserves batch and sequence dims, transforms features."""
    rngs = nnx.Rngs(0)
    mlp = GeGLU(in_features=64, hidden_features=256, rngs=rngs)
    x = jax.random.normal(jax.random.key(0), (2, 16, 64))
    y = mlp(x)
    assert y.shape == (2, 16, 64)


def test_swiglu_no_bias():
    """SwiGLU works without bias."""
    rngs = nnx.Rngs(0)
    mlp = SwiGLU(in_features=64, hidden_features=256, bias=False, rngs=rngs)
    x = jax.random.normal(jax.random.key(0), (2, 16, 64))
    y = mlp(x)
    assert y.shape == (2, 16, 64)


if __name__ == "__main__":
    test_mlp_shape()
    test_mlp_with_silu()
    test_swiglu_shape()
    test_swiglu_different_out_features()
    test_geglu_shape()
    test_swiglu_no_bias()
    print("All tests passed!")
