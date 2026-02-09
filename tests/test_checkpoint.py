"""Tests for checkpoint conversion utilities."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from dreamzero_jax.models.dreamzero import DreamZero, DreamZeroConfig
from dreamzero_jax.utils.checkpoint import (
    _transpose_dense,
    _transpose_conv2d,
    _transpose_conv3d,
    _transpose_conv_auto,
    _identity,
    build_key_mapping,
    convert_checkpoint,
    apply_to_model,
)


# Use a small config for fast tests
def _small_config():
    return DreamZeroConfig(
        dim=64,
        in_channels=4,
        out_channels=4,
        ffn_dim=128,
        freq_dim=32,
        num_heads=4,
        num_layers=2,
        patch_size=(1, 2, 2),
        text_vocab=128,
        text_dim=64,
        text_attn_dim=64,
        text_ffn_dim=128,
        text_num_heads=4,
        text_num_layers=2,
        text_num_buckets=8,
        image_size=28,
        image_patch_size=14,
        image_dim=64,
        image_mlp_ratio=2,
        image_out_dim=32,
        image_num_heads=4,
        image_num_layers=2,
        vae_z_dim=4,
        vae_base_dim=16,
        action_dim=7,
        state_dim=14,
        action_hidden_size=32,
        num_action_per_block=4,
        num_state_per_block=1,
        num_frames_per_block=1,
        max_num_embodiments=4,
        scheduler_shift=3.0,
        num_train_timesteps=100,
        num_inference_steps=4,
    )


def test_transpose_dense():
    """Linear weight transposition: (out, in) -> (in, out)."""
    pt_weight = np.random.randn(64, 32).astype(np.float32)
    flax_kernel = _transpose_dense(pt_weight)
    assert flax_kernel.shape == (32, 64)
    np.testing.assert_allclose(flax_kernel, pt_weight.T)


def test_transpose_conv2d():
    """Conv2d weight transposition: (out, in, kH, kW) -> (kH, kW, in, out)."""
    pt_weight = np.random.randn(64, 32, 3, 3).astype(np.float32)
    flax_kernel = _transpose_conv2d(pt_weight)
    assert flax_kernel.shape == (3, 3, 32, 64)


def test_transpose_conv3d():
    """Conv3d weight transposition: (out, in, kT, kH, kW) -> (kT, kH, kW, in, out)."""
    pt_weight = np.random.randn(64, 32, 3, 3, 3).astype(np.float32)
    flax_kernel = _transpose_conv3d(pt_weight)
    assert flax_kernel.shape == (3, 3, 3, 32, 64)


def test_transpose_conv_auto():
    """Auto-detect conv dimensionality."""
    # 2D conv
    pt_2d = np.random.randn(64, 32, 3, 3).astype(np.float32)
    assert _transpose_conv_auto(pt_2d).shape == (3, 3, 32, 64)

    # 3D conv
    pt_3d = np.random.randn(64, 32, 3, 3, 3).astype(np.float32)
    assert _transpose_conv_auto(pt_3d).shape == (3, 3, 3, 32, 64)

    # Dense fallback
    pt_dense = np.random.randn(64, 32).astype(np.float32)
    assert _transpose_conv_auto(pt_dense).shape == (32, 64)


def test_identity():
    """Identity transform should return input unchanged."""
    x = np.random.randn(64).astype(np.float32)
    result = _identity(x)
    np.testing.assert_array_equal(result, x)


def test_build_key_mapping():
    """Key mapping should have many rules for a full model config."""
    config = _small_config()
    builder = build_key_mapping(config)
    assert len(builder._rules) > 50


def test_key_mapping_dit_block():
    """DiT block keys should map to expected flax paths."""
    config = _small_config()
    builder = build_key_mapping(config)

    # PyTorch keys are prefixed with "model.dit."
    mapping = builder.map_key("model.dit.blocks.0.self_attn.q_proj.weight")
    assert mapping is not None
    assert "self_attn" in ".".join(mapping.flax_path)
    assert "kernel" in ".".join(mapping.flax_path)


def test_key_mapping_text_encoder():
    """Text encoder keys should map correctly."""
    config = _small_config()
    builder = build_key_mapping(config)

    mapping = builder.map_key("model.text_encoder.token_embedding.weight")
    assert mapping is not None
    assert "text_encoder" in ".".join(mapping.flax_path)


def test_key_mapping_unmapped():
    """Completely unknown keys should return None."""
    config = _small_config()
    builder = build_key_mapping(config)
    assert builder.map_key("totally.unknown.parameter.weight") is None


def test_convert_checkpoint_empty():
    """Convert with empty state dict should produce no outputs."""
    config = _small_config()
    result = convert_checkpoint({}, config)
    assert len(result) == 0


def test_convert_checkpoint_maps_params():
    """Convert should map known PyTorch keys to Flax paths."""
    config = _small_config()
    # Create a fake PyTorch weight for a known key
    fake_state = {
        "model.dit.blocks.0.self_attn.q_proj.weight": np.random.randn(64, 64).astype(np.float32),
    }
    result = convert_checkpoint(fake_state, config)
    assert len(result) == 1
    # The converted key should be a tuple with "self_attn" in it
    key = list(result.keys())[0]
    assert "self_attn" in key
    assert "kernel" in key
    # Shape should be transposed
    assert list(result.values())[0].shape == (64, 64)  # square, so same shape after transpose
