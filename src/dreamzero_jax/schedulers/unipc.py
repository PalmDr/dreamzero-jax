"""UniPC multistep scheduler for flow matching (DreamZero).

Higher-order ODE solver for faster inference. Implements the predictor-
corrector algorithm from the UniPC paper, adapted for flow matching.

This is a stateful class — ``step()`` mutates internal buffers (model output
history, last sample). It is NOT meant to be jit-compiled as a whole;
individual math operations are JAX-friendly but the class manages mutable
Python-level state for the multi-step history.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class SchedulerOutput:
    """Container for scheduler step output."""
    prev_sample: jax.Array


class FlowUniPCMultistepScheduler:
    """UniPC multistep scheduler for flow matching.

    Args:
        num_train_timesteps: Total training timesteps.
        solver_order: Order of the multistep solver (default 2).
        shift: Sigma shift factor.
        use_dynamic_shifting: Whether to use dynamic sigma shifting.
        predict_x0: If True, convert model output to x0 prediction internally.
        solver_type: Solver variant — ``'bh1'`` or ``'bh2'``.
        lower_order_final: Use lower-order solver near the end of the schedule.
        final_sigmas_type: How to handle the final sigma — ``'zero'`` or ``'sigma_min'``.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        solver_order: int = 2,
        shift: float | None = 1.0,
        use_dynamic_shifting: bool = False,
        predict_x0: bool = True,
        solver_type: str = "bh2",
        lower_order_final: bool = True,
        disable_corrector: Sequence[int] = (),
        final_sigmas_type: str = "zero",
    ):
        assert solver_type in ("bh1", "bh2"), f"Unsupported solver_type: {solver_type}"

        self.num_train_timesteps = num_train_timesteps
        self.solver_order = solver_order
        self.shift = shift
        self.use_dynamic_shifting = use_dynamic_shifting
        self.predict_x0 = predict_x0
        self.solver_type = solver_type
        self.lower_order_final = lower_order_final
        self.disable_corrector = set(disable_corrector)
        self.final_sigmas_type = final_sigmas_type

        # Build initial sigma schedule
        alphas = np.linspace(1, 1 / num_train_timesteps, num_train_timesteps)[::-1].copy()
        sigmas = 1.0 - alphas

        if not use_dynamic_shifting:
            assert shift is not None
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        self.sigmas: jax.Array = jnp.array(sigmas, dtype=jnp.float32)
        self.timesteps: jax.Array = self.sigmas * num_train_timesteps

        self.sigma_min = float(self.sigmas[-1])
        self.sigma_max = float(self.sigmas[0])

        # Mutable state for multi-step solver
        self.num_inference_steps: int | None = None
        self.model_outputs: list[jax.Array | None] = [None] * solver_order
        self.timestep_list: list[jax.Array | None] = [None] * solver_order
        self.lower_order_nums: int = 0
        self.last_sample: jax.Array | None = None
        self.this_order: int = 1

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: str | None = None,
        sigmas: np.ndarray | None = None,
        mu: float | None = None,
        shift: float | None = None,
    ) -> None:
        """Compute the inference sigma/timestep schedule."""
        self.num_inference_steps = num_inference_steps

        if sigmas is None:
            sigmas = np.linspace(self.sigma_max, self.sigma_min, num_inference_steps + 1).copy()[:-1]

        if self.use_dynamic_shifting:
            assert mu is not None
            sigmas = np.exp(mu) / (np.exp(mu) + (1 / sigmas - 1))
        else:
            s = shift if shift is not None else self.shift
            assert isinstance(s, float)
            sigmas = s * sigmas / (1 + (s - 1) * sigmas)

        if self.final_sigmas_type == "zero":
            sigma_last = 0.0
        else:
            raise ValueError(f"final_sigmas_type must be 'zero', got {self.final_sigmas_type}")

        timesteps = sigmas * self.num_train_timesteps
        sigmas_full = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)

        self.sigmas = jnp.array(sigmas_full)
        self.timesteps = jnp.array(timesteps, dtype=jnp.int64)

        # Reset mutable state
        self.model_outputs = [None] * self.solver_order
        self.timestep_list = [None] * self.solver_order
        self.lower_order_nums = 0
        self.last_sample = None

    @staticmethod
    def _sigma_to_alpha_sigma_t(sigma: jax.Array):
        """Convert sigma to (alpha_t, sigma_t) for flow matching."""
        return 1 - sigma, sigma

    def convert_model_output(
        self,
        model_output: jax.Array,
        sample: jax.Array,
        step_index: int,
    ) -> jax.Array:
        """Convert raw model output to x0 or epsilon prediction."""
        sigma_t = self.sigmas[step_index]
        if self.predict_x0:
            # flow_prediction: x0 = sample - sigma * model_output
            return sample - sigma_t * model_output
        else:
            # epsilon = sample - (1 - sigma) * model_output
            return sample - (1 - sigma_t) * model_output

    def _multistep_uni_p_bh_update(
        self,
        model_output: jax.Array,
        sample: jax.Array,
        order: int,
        step_index: int,
    ) -> jax.Array:
        """Predictor step of the UniPC algorithm."""
        model_output_list = self.model_outputs
        m0 = model_output_list[-1]
        x = sample

        sigma_t, sigma_s0 = self.sigmas[step_index + 1], self.sigmas[step_index]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)

        lambda_t = jnp.log(alpha_t) - jnp.log(sigma_t)
        lambda_s0 = jnp.log(alpha_s0) - jnp.log(sigma_s0)
        h = lambda_t - lambda_s0

        rks = []
        D1s = []
        for i in range(1, order):
            si = step_index - i
            mi = model_output_list[-(i + 1)]
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si])
            lambda_si = jnp.log(alpha_si) - jnp.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            D1s.append((mi - m0) / rk)

        rks.append(jnp.ones(()))
        rks = jnp.stack(rks, axis=0)

        R = []
        b = []
        hh = -h if self.predict_x0 else h
        h_phi_1 = jnp.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1
        factorial_i = 1

        B_h = jnp.expm1(hh) if self.solver_type == "bh2" else hh

        for i in range(1, order + 1):
            R.append(jnp.power(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        R = jnp.stack(R, axis=0)
        b = jnp.stack(b, axis=0)

        if len(D1s) > 0:
            D1s = jnp.stack(D1s, axis=1)
            if order == 2:
                rhos_p = jnp.array([0.5], dtype=x.dtype)
            else:
                rhos_p = jnp.linalg.solve(R[:-1, :-1], b[:-1]).astype(x.dtype)
        else:
            D1s = None
            rhos_p = None

        if self.predict_x0:
            x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
            if D1s is not None:
                pred_res = jnp.einsum("k,bk...->b...", rhos_p, D1s)
            else:
                pred_res = 0
            x_t = x_t_ - alpha_t * B_h * pred_res
        else:
            x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
            if D1s is not None:
                pred_res = jnp.einsum("k,bk...->b...", rhos_p, D1s)
            else:
                pred_res = 0
            x_t = x_t_ - sigma_t * B_h * pred_res

        return x_t.astype(x.dtype)

    def _multistep_uni_c_bh_update(
        self,
        this_model_output: jax.Array,
        last_sample: jax.Array,
        this_sample: jax.Array,
        order: int,
        step_index: int,
    ) -> jax.Array:
        """Corrector step of the UniPC algorithm."""
        model_output_list = self.model_outputs
        m0 = model_output_list[-1]
        x = last_sample
        model_t = this_model_output

        sigma_t, sigma_s0 = self.sigmas[step_index], self.sigmas[step_index - 1]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)

        lambda_t = jnp.log(alpha_t) - jnp.log(sigma_t)
        lambda_s0 = jnp.log(alpha_s0) - jnp.log(sigma_s0)
        h = lambda_t - lambda_s0

        rks = []
        D1s = []
        for i in range(1, order):
            si = step_index - (i + 1)
            mi = model_output_list[-(i + 1)]
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si])
            lambda_si = jnp.log(alpha_si) - jnp.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            D1s.append((mi - m0) / rk)

        rks.append(jnp.ones(()))
        rks = jnp.stack(rks, axis=0)

        R = []
        b = []
        hh = -h if self.predict_x0 else h
        h_phi_1 = jnp.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1
        factorial_i = 1

        B_h = jnp.expm1(hh) if self.solver_type == "bh2" else hh

        for i in range(1, order + 1):
            R.append(jnp.power(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        R = jnp.stack(R, axis=0)
        b = jnp.stack(b, axis=0)

        if len(D1s) > 0:
            D1s = jnp.stack(D1s, axis=1)
        else:
            D1s = None

        if order == 1:
            rhos_c = jnp.array([0.5], dtype=x.dtype)
        else:
            rhos_c = jnp.linalg.solve(R, b).astype(x.dtype)

        if self.predict_x0:
            x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
            if D1s is not None:
                corr_res = jnp.einsum("k,bk...->b...", rhos_c[:-1], D1s)
            else:
                corr_res = 0
            D1_t = model_t - m0
            x_t = x_t_ - alpha_t * B_h * (corr_res + rhos_c[-1] * D1_t)
        else:
            x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
            if D1s is not None:
                corr_res = jnp.einsum("k,bk...->b...", rhos_c[:-1], D1s)
            else:
                corr_res = 0
            D1_t = model_t - m0
            x_t = x_t_ - sigma_t * B_h * (corr_res + rhos_c[-1] * D1_t)

        return x_t.astype(x.dtype)

    def step(
        self,
        model_output: jax.Array,
        timestep: jax.Array | float | int,
        sample: jax.Array,
        step_index: int,
        return_dict: bool = True,
    ) -> SchedulerOutput | tuple[jax.Array]:
        """Single multistep denoising step with optional corrector.

        Args:
            model_output: Raw model prediction (velocity/flow).
            timestep: Current timestep.
            sample: Current noisy sample.
            step_index: Index into the timestep schedule.
            return_dict: If True, return SchedulerOutput; else return tuple.

        Returns:
            Denoised sample.
        """
        assert self.num_inference_steps is not None, "Call set_timesteps first"

        use_corrector = (
            step_index > 0
            and step_index - 1 not in self.disable_corrector
            and self.last_sample is not None
        )

        model_output_convert = self.convert_model_output(
            model_output=model_output, sample=sample, step_index=step_index,
        )

        if use_corrector:
            sample = self._multistep_uni_c_bh_update(
                this_model_output=model_output_convert,
                last_sample=self.last_sample,
                this_sample=sample,
                order=self.this_order,
                step_index=step_index,
            )

        # Shift history buffers
        for i in range(self.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
            self.timestep_list[i] = self.timestep_list[i + 1]

        self.model_outputs[-1] = model_output_convert
        self.timestep_list[-1] = timestep

        if self.lower_order_final:
            this_order = min(self.solver_order, len(self.timesteps) - step_index)
        else:
            this_order = self.solver_order

        self.this_order = min(this_order, self.lower_order_nums + 1)
        assert self.this_order > 0

        self.last_sample = sample
        prev_sample = self._multistep_uni_p_bh_update(
            model_output=model_output,
            sample=sample,
            order=self.this_order,
            step_index=step_index,
        )

        if self.lower_order_nums < self.solver_order:
            self.lower_order_nums += 1

        if not return_dict:
            return (prev_sample,)
        return SchedulerOutput(prev_sample=prev_sample)

    def add_noise(
        self,
        original_samples: jax.Array,
        noise: jax.Array,
        timesteps: jax.Array,
    ) -> jax.Array:
        """Forward diffusion: add noise to clean samples.

        Args:
            original_samples: Clean samples.
            noise: Noise of the same shape.
            timesteps: Per-sample timesteps.

        Returns:
            Noisy samples.
        """
        step_indices = jnp.array([
            jnp.argmin(jnp.abs(self.timesteps - t)) for t in timesteps
        ])
        sigma = self.sigmas[step_indices].astype(original_samples.dtype)
        while sigma.ndim < original_samples.ndim:
            sigma = sigma[..., None]

        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
        return alpha_t * original_samples + sigma_t * noise

    def scale_model_input(self, sample: jax.Array, *args, **kwargs) -> jax.Array:
        """No-op for flow matching (model input doesn't need scaling)."""
        return sample
