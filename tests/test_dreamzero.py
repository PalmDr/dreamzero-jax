"""Tests for the full DreamZero model assembly."""

import jax
import jax.numpy as jnp
from flax import nnx

from dreamzero_jax.models.dreamzero import (
    DreamZero,
    DreamZeroConfig,
    InferenceOutput,
    TrainOutput,
)


def _small_config() -> DreamZeroConfig:
    """Return a minimal config for fast testing."""
    return DreamZeroConfig(
        # DiT (in/out channels must match VAE z_dim)
        dim=64,
        in_channels=4,
        out_channels=4,
        ffn_dim=128,
        freq_dim=32,
        num_heads=4,
        num_layers=2,
        patch_size=(1, 2, 2),
        qk_norm=True,
        cross_attn_norm=False,
        # Text encoder
        text_vocab=256,
        text_dim=64,
        text_attn_dim=64,
        text_ffn_dim=128,
        text_num_heads=4,
        text_num_layers=2,
        text_num_buckets=32,
        # Image encoder
        image_size=28,
        image_patch_size=14,
        image_dim=64,
        image_mlp_ratio=2,
        image_out_dim=32,
        image_num_heads=4,
        image_num_layers=2,
        # VAE
        vae_z_dim=4,
        vae_base_dim=32,
        # Action
        action_dim=7,
        state_dim=14,
        action_hidden_size=32,
        num_action_per_block=4,
        num_state_per_block=1,
        num_frames_per_block=1,
        max_num_embodiments=4,
        action_horizon=None,
        # Scheduler
        scheduler_shift=5.0,
        num_train_timesteps=100,
        num_inference_steps=4,
        cfg_scale=2.0,
        # I2V
        has_image_input=False,  # Simplify for tests
    )


# Sizes derived from the small config
BATCH = 1
# VAE with z_dim=4, base_dim=32: 8x spatial, 4x temporal compression
# For minimal test: use 4 frames, 16x16 spatial
NUM_FRAMES = 4
IMAGE_H = 16
IMAGE_W = 16
TEXT_LEN = 8


def _make_model(config=None, rngs=None):
    config = config or _small_config()
    rngs = rngs or nnx.Rngs(42)
    return DreamZero(config, rngs=rngs)


def test_dreamzero_config():
    """DreamZeroConfig creates with sensible defaults."""
    cfg = _small_config()
    assert cfg.dim == 64
    assert cfg.action_dim == 7
    assert cfg.num_inference_steps == 4


def test_dreamzero_encode_prompt():
    """encode_prompt produces correct shape."""
    model = _make_model()
    ids = jnp.zeros((BATCH, TEXT_LEN), dtype=jnp.int32)
    mask = jnp.ones((BATCH, TEXT_LEN))
    emb = model.encode_prompt(ids, mask)
    assert emb.shape == (BATCH, TEXT_LEN, 64)  # text_dim=64


def test_dreamzero_encode_prompt_masked():
    """encode_prompt zeros out padding positions."""
    model = _make_model()
    ids = jnp.zeros((BATCH, TEXT_LEN), dtype=jnp.int32)
    mask = jnp.ones((BATCH, TEXT_LEN))
    mask = mask.at[:, 4:].set(0)  # Pad last 4 positions
    emb = model.encode_prompt(ids, mask)
    # Padded positions should be zero
    assert jnp.allclose(emb[:, 4:], 0.0)


def test_dreamzero_encode_image():
    """encode_image produces CLIP features."""
    model = _make_model()
    image = jax.random.uniform(
        jax.random.key(0), (BATCH, 28, 28, 3), minval=-1, maxval=1,
    )
    features = model.encode_image(image)
    # CLIP ViT with image_size=28, patch=14 → 4 patches + CLS = 5 tokens
    assert features.shape == (BATCH, 5, 64)


def test_dreamzero_train_output_type():
    """train_step returns a TrainOutput NamedTuple."""
    cfg = _small_config()
    model = _make_model(cfg)

    # Need video that matches VAE input requirements
    # VAE: 8x spatial, 4x temporal → minimum 5 frames (1 + 4k), 16x16 spatial
    video = jax.random.uniform(
        jax.random.key(0), (BATCH, 5, IMAGE_H, IMAGE_W, 3), minval=-1, maxval=1,
    )
    ids = jnp.zeros((BATCH, TEXT_LEN), dtype=jnp.int32)

    # After VAE encoding: T'= ceil((5-1)/4)+1 = 2 frames, H'=2, W'=2
    # With num_frames_per_block=1 and patch (1,2,2): f=2, h=1, w=1
    # num_blocks = 2
    num_blocks = 2
    total_actions = num_blocks * cfg.num_action_per_block
    actions = jax.random.uniform(
        jax.random.key(1), (BATCH, total_actions, cfg.action_dim), minval=-1, maxval=1,
    )
    state = jax.random.normal(jax.random.key(2), (BATCH, num_blocks, cfg.state_dim))
    embodiment_id = jnp.array([0])

    result = model.train_step(
        video, ids, actions, state, embodiment_id, key=jax.random.key(3),
    )

    assert isinstance(result, TrainOutput)
    assert result.loss.shape == ()
    assert result.dynamics_loss.shape == ()
    assert result.action_loss.shape == ()
    # Loss should be finite
    assert jnp.isfinite(result.loss)


def test_dreamzero_generate_output_type():
    """generate returns an InferenceOutput NamedTuple."""
    cfg = _small_config()
    model = _make_model(cfg)

    video = jax.random.uniform(
        jax.random.key(0), (BATCH, 5, IMAGE_H, IMAGE_W, 3), minval=-1, maxval=1,
    )
    ids = jnp.zeros((BATCH, TEXT_LEN), dtype=jnp.int32)

    # num_blocks derived from VAE output
    num_blocks = 2
    state = jax.random.normal(jax.random.key(2), (BATCH, num_blocks, cfg.state_dim))
    embodiment_id = jnp.array([0])

    result = model.generate(
        video, ids, state, embodiment_id, key=jax.random.key(4),
    )

    assert isinstance(result, InferenceOutput)
    assert jnp.isfinite(result.action_pred).all()
    assert jnp.isfinite(result.video_pred).all()
