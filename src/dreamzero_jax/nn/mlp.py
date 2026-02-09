"""MLP variants for DreamZero."""

from typing import Callable

import jax
import jax.numpy as jnp
from flax import nnx


class MLP(nnx.Module):
    """Simple 2-layer MLP with configurable activation.

    MLP(x) = (activation(x @ W_up) @ W_down)
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int | None = None,
        activation: Callable[[jax.Array], jax.Array] = jax.nn.gelu,
        bias: bool = True,
        *,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        out_features = out_features or in_features
        self.activation = activation

        self.w_up = nnx.Linear(
            in_features,
            hidden_features,
            use_bias=bias,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.w_down = nnx.Linear(
            hidden_features,
            out_features,
            use_bias=bias,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.w_down(self.activation(self.w_up(x)))


class SwiGLU(nnx.Module):
    """SwiGLU feed-forward network.

    SwiGLU(x) = (Swish(x @ W_gate) * (x @ W_up)) @ W_down

    Reference: https://arxiv.org/abs/2002.05202
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int | None = None,
        bias: bool = True,
        *,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        out_features = out_features or in_features

        self.w_gate = nnx.Linear(
            in_features,
            hidden_features,
            use_bias=bias,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.w_up = nnx.Linear(
            in_features,
            hidden_features,
            use_bias=bias,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.w_down = nnx.Linear(
            hidden_features,
            out_features,
            use_bias=bias,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.w_down(jax.nn.silu(self.w_gate(x)) * self.w_up(x))


class GeGLU(nnx.Module):
    """GeGLU feed-forward network.

    GeGLU(x) = (GELU(x @ W_gate) * (x @ W_up)) @ W_down

    Reference: https://arxiv.org/abs/2002.05202
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int | None = None,
        bias: bool = True,
        *,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        out_features = out_features or in_features

        self.w_gate = nnx.Linear(
            in_features,
            hidden_features,
            use_bias=bias,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.w_up = nnx.Linear(
            in_features,
            hidden_features,
            use_bias=bias,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.w_down = nnx.Linear(
            hidden_features,
            out_features,
            use_bias=bias,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.w_down(jax.nn.gelu(self.w_gate(x)) * self.w_up(x))
