"""Tests for diffusion schedulers."""

import jax
import jax.numpy as jnp

from dreamzero_jax.schedulers.flow_matching import FlowMatchScheduler
from dreamzero_jax.schedulers.unipc import FlowUniPCMultistepScheduler, SchedulerOutput


# ---------------------------------------------------------------------------
# FlowMatchScheduler
# ---------------------------------------------------------------------------


def test_flow_match_set_timesteps():
    """set_timesteps produces arrays of correct length."""
    sched = FlowMatchScheduler(num_inference_steps=50)
    assert sched.sigmas.shape == (50,)
    assert sched.timesteps.shape == (50,)


def test_flow_match_sigma_shift():
    """Shift formula transforms sigma values away from identity."""
    sched = FlowMatchScheduler(num_inference_steps=10, shift=3.0)
    # With shift > 1, sigmas should be biased towards higher values
    unshifted = jnp.linspace(
        sched.sigma_min + (sched.sigma_max - sched.sigma_min),
        sched.sigma_min,
        10,
    )
    # Shifted sigmas should differ from unshifted
    assert not jnp.allclose(sched.sigmas, unshifted)
    # First sigma should be larger than last
    assert sched.sigmas[0] > sched.sigmas[-1]


def test_flow_match_add_noise():
    """add_noise interpolates between sample and noise."""
    sched = FlowMatchScheduler(num_inference_steps=100)
    sample = jnp.ones((2, 4))
    noise = jnp.zeros((2, 4))
    # At high sigma (early timestep), output should be closer to noise
    t_high = sched.timesteps[0] * jnp.ones(2)
    noisy = sched.add_noise(sample, noise, t_high)
    assert noisy.shape == (2, 4)
    # At low sigma (late timestep), output should be closer to sample
    t_low = sched.timesteps[-1] * jnp.ones(2)
    noisy_low = sched.add_noise(sample, noise, t_low)
    # Low sigma means more of the original sample
    assert float(jnp.mean(jnp.abs(noisy_low - sample))) < float(jnp.mean(jnp.abs(noisy - sample)))


def test_flow_match_training_target():
    """Training target is noise - sample."""
    sched = FlowMatchScheduler()
    sample = jnp.ones((2, 4))
    noise = jnp.full((2, 4), 3.0)
    target = sched.training_target(sample, noise, jnp.zeros(2))
    assert jnp.allclose(target, noise - sample)


def test_flow_match_step():
    """step produces output of correct shape."""
    sched = FlowMatchScheduler(num_inference_steps=10)
    model_output = jax.random.normal(jax.random.key(0), (2, 4, 4))
    sample = jax.random.normal(jax.random.key(1), (2, 4, 4))
    timestep = sched.timesteps[0]
    out = sched.step(model_output, timestep, sample)
    assert out.shape == sample.shape


def test_flow_match_training_weights():
    """Training weights are computed when training=True."""
    sched = FlowMatchScheduler(num_inference_steps=100)
    sched.set_timesteps(100, training=True)
    assert sched.linear_timesteps_weights is not None
    assert sched.linear_timesteps_weights.shape == (100,)
    # Weights should be positive
    assert float(sched.linear_timesteps_weights.min()) >= 0


def test_flow_match_extra_one_step():
    """extra_one_step mode produces correct number of sigmas."""
    sched = FlowMatchScheduler(num_inference_steps=10, extra_one_step=True)
    assert sched.sigmas.shape == (10,)


# ---------------------------------------------------------------------------
# FlowUniPCMultistepScheduler
# ---------------------------------------------------------------------------


def test_unipc_set_timesteps():
    """set_timesteps produces correct schedule lengths."""
    sched = FlowUniPCMultistepScheduler(shift=5.0)
    sched.set_timesteps(16)
    assert sched.num_inference_steps == 16
    # sigmas has one extra entry (final sigma)
    assert sched.sigmas.shape == (17,)
    assert sched.timesteps.shape == (16,)


def test_unipc_step_shape():
    """step produces output of correct shape via predictor."""
    sched = FlowUniPCMultistepScheduler(shift=5.0)
    sched.set_timesteps(16)
    model_output = jax.random.normal(jax.random.key(0), (1, 4, 4))
    sample = jax.random.normal(jax.random.key(1), (1, 4, 4))

    result = sched.step(model_output, sched.timesteps[0], sample, step_index=0)
    assert isinstance(result, SchedulerOutput)
    assert result.prev_sample.shape == (1, 4, 4)


def test_unipc_multistep_convergence():
    """Multiple steps should produce different outputs (solver is stepping)."""
    sched = FlowUniPCMultistepScheduler(shift=5.0)
    sched.set_timesteps(4)
    key = jax.random.key(0)
    sample = jax.random.normal(key, (1, 4, 4))

    outputs = [sample]
    for i, t in enumerate(sched.timesteps):
        model_output = jax.random.normal(jax.random.fold_in(key, i), (1, 4, 4))
        result = sched.step(model_output, t, outputs[-1], step_index=i)
        outputs.append(result.prev_sample)

    # Each step should change the sample
    for i in range(1, len(outputs)):
        assert not jnp.allclose(outputs[i], outputs[i - 1])


def test_unipc_return_tuple():
    """return_dict=False returns a tuple."""
    sched = FlowUniPCMultistepScheduler(shift=5.0)
    sched.set_timesteps(4)
    model_output = jax.random.normal(jax.random.key(0), (1, 4, 4))
    sample = jax.random.normal(jax.random.key(1), (1, 4, 4))

    result = sched.step(model_output, sched.timesteps[0], sample, step_index=0, return_dict=False)
    assert isinstance(result, tuple)
    assert result[0].shape == (1, 4, 4)


def test_unipc_add_noise():
    """add_noise produces correct shape."""
    sched = FlowUniPCMultistepScheduler(shift=5.0)
    sched.set_timesteps(16)
    sample = jnp.ones((2, 4))
    noise = jnp.zeros((2, 4))
    timesteps = sched.timesteps[:2]
    noisy = sched.add_noise(sample, noise, timesteps)
    assert noisy.shape == (2, 4)


def test_unipc_scale_model_input():
    """scale_model_input is a no-op."""
    sched = FlowUniPCMultistepScheduler()
    x = jnp.ones((2, 4))
    assert jnp.array_equal(sched.scale_model_input(x), x)
