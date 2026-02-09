"""Flow matching scheduler for DreamZero.

Implements the shifted flow matching noise schedule used for both training
and inference. This is a plain Python class (not an nnx.Module) since it
has no learnable parameters.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


class FlowMatchScheduler:
    """Flow matching scheduler with shifted sigma schedule.

    The shift formula biases the noise schedule towards higher noise levels:
        sigma_shifted = shift * sigma / (1 + (shift - 1) * sigma)

    Denoising step (Euler):
        prev_sample = sample + model_output * (sigma_next - sigma_current)

    Training target:
        target = noise - sample  (the velocity field)

    Forward diffusion:
        noisy = (1 - sigma) * sample + sigma * noise
    """

    def __init__(
        self,
        num_inference_steps: int = 100,
        num_train_timesteps: int = 1000,
        shift: float = 3.0,
        sigma_max: float = 1.0,
        sigma_min: float = 0.003 / 1.002,
        inverse_timesteps: bool = False,
        extra_one_step: bool = False,
        reverse_sigmas: bool = False,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas

        self.sigmas: jax.Array = jnp.empty(0)
        self.timesteps: jax.Array = jnp.empty(0)
        self.linear_timesteps_weights: jax.Array | None = None
        self.training: bool = False

        self.set_timesteps(num_inference_steps)

    def set_timesteps(
        self,
        num_inference_steps: int = 100,
        denoising_strength: float = 1.0,
        training: bool = False,
        shift: float | None = None,
    ) -> None:
        """Compute the sigma/timestep schedule.

        Args:
            num_inference_steps: Number of denoising steps.
            denoising_strength: Fraction of the full schedule to use (1.0 = full).
            training: If True, compute per-timestep training weights.
            shift: Override the shift parameter.
        """
        if shift is not None:
            self.shift = shift

        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength

        if self.extra_one_step:
            sigmas = jnp.linspace(sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]
        else:
            sigmas = jnp.linspace(sigma_start, self.sigma_min, num_inference_steps)

        if self.inverse_timesteps:
            sigmas = jnp.flip(sigmas)

        sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)

        if self.reverse_sigmas:
            sigmas = 1 - sigmas

        self.sigmas = sigmas
        self.timesteps = sigmas * self.num_train_timesteps

        if training:
            x = self.timesteps
            y = jnp.exp(-2 * ((x - num_inference_steps / 2) / num_inference_steps) ** 2)
            y_shifted = y - y.min()
            self.linear_timesteps_weights = y_shifted * (num_inference_steps / y_shifted.sum())
            self.training = True
        else:
            self.training = False

    def step(
        self,
        model_output: jax.Array,
        timestep: jax.Array | float,
        sample: jax.Array,
        to_final: bool = False,
    ) -> jax.Array:
        """Single Euler denoising step.

        Args:
            model_output: Predicted velocity field.
            timestep: Current timestep (scalar or array).
            sample: Current noisy sample.
            to_final: If True, step to the terminal sigma.

        Returns:
            Denoised sample.
        """
        timestep = jnp.asarray(timestep, dtype=jnp.float32)
        timestep_id = jnp.argmin(jnp.abs(self.timesteps - timestep))
        sigma = self.sigmas[timestep_id]

        if to_final:
            sigma_ = 1.0 if (self.inverse_timesteps or self.reverse_sigmas) else 0.0
        else:
            sigma_ = jnp.where(
                timestep_id + 1 >= len(self.timesteps),
                jnp.where(self.inverse_timesteps | self.reverse_sigmas, 1.0, 0.0),
                self.sigmas[jnp.minimum(timestep_id + 1, len(self.timesteps) - 1)],
            )

        prev_sample = sample + model_output * (sigma_ - sigma)
        return prev_sample

    def add_noise(
        self,
        original_samples: jax.Array,
        noise: jax.Array,
        timestep: jax.Array,
    ) -> jax.Array:
        """Forward diffusion: add noise to clean samples.

        Args:
            original_samples: Clean samples, shape ``(B, ...)``.
            noise: Noise of the same shape.
            timestep: Per-sample timesteps, shape ``(B,)``.

        Returns:
            Noisy samples.
        """
        timestep_id = jnp.argmin(
            jnp.abs(self.timesteps[..., None] - timestep[None, ...]), axis=0,
        )
        sigma = self.sigmas[timestep_id].astype(original_samples.dtype)
        # Broadcast sigma to match sample dims
        while sigma.ndim < original_samples.ndim:
            sigma = sigma[..., None]
        return (1 - sigma) * original_samples + sigma * noise

    def training_target(
        self,
        sample: jax.Array,
        noise: jax.Array,
        timestep: jax.Array,
    ) -> jax.Array:
        """Compute the training target (velocity field).

        Args:
            sample: Clean samples.
            noise: Noise.
            timestep: Timesteps (unused, included for API symmetry).

        Returns:
            Target = noise - sample.
        """
        del timestep
        return noise - sample

    def training_weight(self, timestep: jax.Array) -> jax.Array:
        """Get per-sample training loss weights.

        Args:
            timestep: Per-sample timesteps, shape ``(B,)``.

        Returns:
            Weights of shape ``(B,)``.
        """
        assert self.linear_timesteps_weights is not None, (
            "Must call set_timesteps(training=True) before training_weight"
        )
        timestep_id = jnp.argmin(
            jnp.abs(self.timesteps[..., None] - timestep[None, ...]), axis=0,
        )
        return self.linear_timesteps_weights[timestep_id]
