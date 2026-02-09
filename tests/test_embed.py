"""Tests for embedding layers."""

import jax
import jax.numpy as jnp
from flax import nnx

from dreamzero_jax.nn.embed import (
    sinusoidal_embedding,
    TimestepEmbedding,
    PatchEmbed,
    PatchEmbed3D,
    get_1d_sincos_pos_embed,
    get_2d_sincos_pos_embed,
    get_3d_sincos_pos_embed,
    precompute_freqs_cis,
    precompute_freqs_cis_3d,
    apply_rotary_emb,
)


def test_sinusoidal_embedding_shape():
    """Sinusoidal embedding has correct shape."""
    timesteps = jnp.array([0, 100, 500, 999])
    emb = sinusoidal_embedding(timesteps, dim=256)
    assert emb.shape == (4, 256)


def test_sinusoidal_embedding_different_timesteps():
    """Different timesteps produce different embeddings."""
    t1 = jnp.array([0])
    t2 = jnp.array([500])
    emb1 = sinusoidal_embedding(t1, dim=256)
    emb2 = sinusoidal_embedding(t2, dim=256)
    assert not jnp.allclose(emb1, emb2)


def test_timestep_embedding_shape():
    """TimestepEmbedding has correct output shape."""
    rngs = nnx.Rngs(0)
    embed = TimestepEmbedding(dim=256, rngs=rngs)
    timesteps = jnp.array([0, 100, 500, 999])
    emb = embed(timesteps)
    assert emb.shape == (4, 256)


def test_patch_embed_image():
    """PatchEmbed works on images."""
    rngs = nnx.Rngs(0)
    embed = PatchEmbed(patch_size=16, in_channels=3, embed_dim=768, rngs=rngs)
    x = jax.random.normal(jax.random.key(0), (2, 224, 224, 3))
    out = embed(x)
    # 224 / 16 = 14, so 14 * 14 = 196 patches
    assert out.shape == (2, 196, 768)


def test_patch_embed_video():
    """PatchEmbed works on video (processes each frame)."""
    rngs = nnx.Rngs(0)
    embed = PatchEmbed(patch_size=16, in_channels=3, embed_dim=768, rngs=rngs)
    x = jax.random.normal(jax.random.key(0), (2, 8, 224, 224, 3))  # 8 frames
    out = embed(x)
    # 8 frames * 196 patches = 1568
    assert out.shape == (2, 1568, 768)


def test_patch_embed_3d():
    """PatchEmbed3D works on video with temporal patches."""
    rngs = nnx.Rngs(0)
    embed = PatchEmbed3D(
        patch_size=(2, 16, 16), in_channels=3, embed_dim=768, rngs=rngs
    )
    x = jax.random.normal(jax.random.key(0), (2, 8, 224, 224, 3))
    out = embed(x)
    # T: 8/2=4, H: 224/16=14, W: 224/16=14 -> 4*14*14 = 784 patches
    assert out.shape == (2, 784, 768)


def test_1d_pos_embed():
    """1D positional embeddings have correct shape."""
    emb = get_1d_sincos_pos_embed(embed_dim=256, length=100)
    assert emb.shape == (100, 256)


def test_2d_pos_embed():
    """2D positional embeddings have correct shape."""
    emb = get_2d_sincos_pos_embed(embed_dim=256, grid_size=(14, 14))
    assert emb.shape == (196, 256)


def test_3d_pos_embed():
    """3D positional embeddings have correct shape."""
    emb = get_3d_sincos_pos_embed(embed_dim=256, grid_size=(4, 14, 14))
    assert emb.shape == (784, 256)


def test_rope_freqs_shape():
    """RoPE frequencies have correct shape."""
    freqs = precompute_freqs_cis(dim=64, max_len=100)
    assert freqs.shape == (100, 32)  # dim // 2
    assert freqs.dtype == jnp.complex64


def test_rope_apply():
    """RoPE can be applied to query/key tensors."""
    freqs = precompute_freqs_cis(dim=64, max_len=100)
    x = jax.random.normal(jax.random.key(0), (2, 8, 100, 64))  # (B, heads, seq, dim)
    x_rope = apply_rotary_emb(x, freqs)
    assert x_rope.shape == x.shape
    assert x_rope.dtype == jnp.float32


def test_rope_different_positions():
    """RoPE produces different embeddings for different positions."""
    freqs = precompute_freqs_cis(dim=64, max_len=100)
    x = jnp.ones((1, 1, 100, 64))
    x_rope = apply_rotary_emb(x, freqs)
    # Different positions should have different values
    assert not jnp.allclose(x_rope[0, 0, 0], x_rope[0, 0, 50])


def test_rope_3d_freqs_shape():
    """3D RoPE frequencies have correct shape."""
    freqs = precompute_freqs_cis_3d(dim=64, grid_size=(4, 14, 14))
    # 4 * 14 * 14 = 784 positions, dim // 2 = 32 frequencies
    assert freqs.shape == (784, 32)
    assert freqs.dtype == jnp.complex64


if __name__ == "__main__":
    test_sinusoidal_embedding_shape()
    test_sinusoidal_embedding_different_timesteps()
    test_timestep_embedding_shape()
    test_patch_embed_image()
    test_patch_embed_video()
    test_patch_embed_3d()
    test_1d_pos_embed()
    test_2d_pos_embed()
    test_3d_pos_embed()
    test_rope_freqs_shape()
    test_rope_apply()
    test_rope_different_positions()
    test_rope_3d_freqs_shape()
    print("All tests passed!")
