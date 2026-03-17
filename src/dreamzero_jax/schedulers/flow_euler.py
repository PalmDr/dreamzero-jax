"""Scan-compatible Flow Euler scheduler for DreamZero.

A minimal Euler ODE integrator for flow matching that is fully compatible
with ``jax.lax.scan``. Unlike :class:`FlowUniPCMultistepScheduler`, this
scheduler has **no mutable Python state** -- all information needed for a
step is passed explicitly via arrays, making it safe for tracing.

The denoising update is identical to :class:`FlowMatchScheduler`:

    prev_sample = sample + model_output * (sigma_next - sigma_current)

The key difference is the API: instead of looking up the timestep index
via ``jnp.argmin``, the caller passes pre-computed ``(sigma, sigma_next)``
pairs directly, enabling efficient use inside ``jax.lax.scan``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class FlowEulerSchedule(NamedTuple):
    """Pre-computed schedule arrays for scan-based Euler denoising.

    All arrays have shape ``(num_steps,)`` and are indexed by step ``i``.
    """

    sigmas: jax.Array        # (num_steps,) sigma at each step
    sigmas_next: jax.Array   # (num_steps,) sigma at the *next* step (0 for last)
    timesteps: jax.Array     # (num_steps,) timestep = sigma * num_train_timesteps


def make_flow_euler_schedule(
    num_inference_steps: int,
    num_train_timesteps: int = 1000,
    shift: float = 5.0,
    sigma_max: float = 1.0,
    sigma_min: float = 0.0,
) -> FlowEulerSchedule:
    """Build a flow-matching Euler schedule.

    Produces the same sigma sequence as :class:`FlowMatchScheduler` with
    ``extra_one_step=True``, which is what DreamZero uses at inference time.

    Args:
        num_inference_steps: Number of denoising steps.
        num_train_timesteps: Total training timesteps (for timestep scaling).
        shift: Sigma shift factor.
        sigma_max: Maximum sigma (typically 1.0).
        sigma_min: Minimum sigma (typically 0.0).

    Returns:
        A :class:`FlowEulerSchedule` NamedTuple ready for ``euler_step``
        and ``jax.lax.scan``.
    """
    # Linearly spaced sigmas with one extra (then drop last), matching
    # FlowMatchScheduler(extra_one_step=True).
    sigmas_raw = np.linspace(sigma_max, sigma_min, num_inference_steps + 1)[:-1]

    # Apply shift: sigma_shifted = shift * sigma / (1 + (shift - 1) * sigma)
    sigmas_shifted = shift * sigmas_raw / (1.0 + (shift - 1.0) * sigmas_raw)

    # Next-step sigmas: shift by one, with final sigma = 0 (clean sample).
    sigmas_next = np.zeros(num_inference_steps, dtype=np.float32)
    sigmas_next[:-1] = sigmas_shifted[1:]
    # Last step goes to 0 (matching FlowMatchScheduler behaviour).

    sigmas = jnp.array(sigmas_shifted, dtype=jnp.float32)
    sigmas_next = jnp.array(sigmas_next, dtype=jnp.float32)
    timesteps = sigmas * num_train_timesteps

    return FlowEulerSchedule(
        sigmas=sigmas,
        sigmas_next=sigmas_next,
        timesteps=timesteps,
    )


def euler_step(
    model_output: jax.Array,
    sample: jax.Array,
    sigma: jax.Array,
    sigma_next: jax.Array,
) -> jax.Array:
    """Single Euler denoising step for flow matching.

    Computes::

        prev_sample = sample + model_output * (sigma_next - sigma)

    All arguments may be traced JAX values (scan-safe).

    Args:
        model_output: Predicted velocity field, same shape as ``sample``.
        sample: Current noisy sample.
        sigma: Current sigma (scalar).
        sigma_next: Next sigma (scalar), 0 for the final step.

    Returns:
        Denoised sample (same shape and dtype as ``sample``).
    """
    # Upcast to float32 for numerical stability, then cast back.
    sample_f32 = sample.astype(jnp.float32)
    output_f32 = model_output.astype(jnp.float32)
    dt = sigma_next - sigma  # negative (denoising)
    prev = sample_f32 + output_f32 * dt
    return prev.astype(sample.dtype)
