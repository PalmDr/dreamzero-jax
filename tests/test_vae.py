"""Tests for WAN Video VAE."""

import jax
import jax.numpy as jnp
from flax import nnx

from dreamzero_jax.models.vae import (
    AttentionBlock,
    CausalConv3d,
    Decoder3d,
    Encoder3d,
    ResidualBlock,
    SpatialDownsample,
    SpatialUpsample,
    TemporalDownsample,
    TemporalUpsample,
    WanVideoVAE,
)

# Small dims for fast tests
BATCH = 2
BASE_DIM = 32
Z_DIM = 4


def test_causal_conv3d_shape():
    """CausalConv3d produces correct output shapes with various configs."""
    conv = CausalConv3d(8, 16, kernel_size=3, padding=1, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(0), (BATCH, 4, 8, 8, 8))
    out = conv(x)
    assert out.shape == (BATCH, 4, 8, 8, 16)

    # Stride-2 in time
    conv_t = CausalConv3d(8, 16, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0), rngs=nnx.Rngs(0))
    out_t = conv_t(x)
    assert out_t.shape == (BATCH, 2, 8, 8, 16)


def test_causal_conv3d_causality():
    """Future frames don't affect past outputs in CausalConv3d."""
    conv = CausalConv3d(4, 4, kernel_size=3, padding=1, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(0), (1, 4, 4, 4, 4))

    out_full = conv(x)

    # Zero out frames 2 and 3 (future)
    x_masked = x.at[:, 2:].set(0.0)
    out_masked = conv(x_masked)

    # Frames 0 and 1 in the output should be identical
    assert jnp.allclose(out_full[:, :2], out_masked[:, :2], atol=1e-6)


def test_residual_block_shape():
    """ResidualBlock preserves spatial dims and handles channel change."""
    block = ResidualBlock(16, 16, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(0), (BATCH, 4, 8, 8, 16))
    out = block(x)
    assert out.shape == (BATCH, 4, 8, 8, 16)

    # Channel change
    block2 = ResidualBlock(16, 32, rngs=nnx.Rngs(0))
    out2 = block2(x)
    assert out2.shape == (BATCH, 4, 8, 8, 32)


def test_residual_block_shortcut():
    """1x1 shortcut conv is created when in_dim != out_dim."""
    block_same = ResidualBlock(16, 16, rngs=nnx.Rngs(0))
    assert block_same.shortcut is None

    block_diff = ResidualBlock(16, 32, rngs=nnx.Rngs(0))
    assert block_diff.shortcut is not None


def test_attention_block_shape():
    """AttentionBlock preserves shape and starts near identity."""
    block = AttentionBlock(32, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(0), (BATCH, 2, 4, 4, 32))
    out = block(x)
    assert out.shape == (BATCH, 2, 4, 4, 32)
    # Zero-init means output should be close to input initially
    assert jnp.allclose(x, out, atol=1e-5)


def test_resample_downsample2d():
    """SpatialDownsample halves H and W."""
    ds = SpatialDownsample(16, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(0), (BATCH, 4, 8, 8, 16))
    out = ds(x)
    assert out.shape == (BATCH, 4, 4, 4, 16)


def test_resample_downsample3d():
    """TemporalDownsample halves T."""
    ds = TemporalDownsample(16, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(0), (BATCH, 4, 8, 8, 16))
    out = ds(x)
    assert out.shape == (BATCH, 2, 8, 8, 16)


def test_resample_upsample2d():
    """SpatialUpsample doubles H and W."""
    us = SpatialUpsample(16, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(0), (BATCH, 4, 4, 4, 16))
    out = us(x)
    assert out.shape == (BATCH, 4, 8, 8, 16)


def test_resample_upsample3d():
    """TemporalUpsample doubles T."""
    us = TemporalUpsample(16, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(0), (BATCH, 4, 8, 8, 16))
    out = us(x)
    assert out.shape == (BATCH, 8, 8, 8, 16)


def test_encoder_output_shape():
    """Encoder produces correct latent shape from full encoder."""
    enc = Encoder3d(
        z_dim=Z_DIM, base_dim=BASE_DIM, dim_multipliers=(1, 2, 4, 4),
        num_res_blocks=2, rngs=nnx.Rngs(0),
    )
    # T=8 (4x temporal down → T'=2), H=W=16 (8x spatial down → 2)
    x = jax.random.normal(jax.random.key(0), (1, 8, 16, 16, 3))
    out = enc(x)
    # Encoder outputs 2*z_dim channels (mean + logvar)
    assert out.shape == (1, 2, 2, 2, 2 * Z_DIM)


def test_decoder_output_shape():
    """Decoder reconstructs correct shape from latent."""
    dec = Decoder3d(
        z_dim=Z_DIM, base_dim=BASE_DIM, dim_multipliers=(1, 2, 4, 4),
        num_res_blocks=2, rngs=nnx.Rngs(0),
    )
    # Latent: (1, 2, 2, 2, z_dim) → should decode to (1, 8, 16, 16, 3)
    z = jax.random.normal(jax.random.key(0), (1, 2, 2, 2, Z_DIM))
    out = dec(z)
    assert out.shape == (1, 8, 16, 16, 3)


def test_vae_roundtrip_shape():
    """Encode → decode preserves spatial/temporal shape."""
    vae = WanVideoVAE(
        z_dim=Z_DIM, base_dim=BASE_DIM, dim_multipliers=(1, 2, 4, 4),
        num_res_blocks=2,
        mean=jnp.zeros(Z_DIM),
        std=jnp.ones(Z_DIM),
        rngs=nnx.Rngs(0),
    )
    x = jax.random.normal(jax.random.key(0), (1, 8, 16, 16, 3))
    z = vae.encode(x)
    assert z.shape == (1, 2, 2, 2, Z_DIM)
    recon = vae.decode(z)
    assert recon.shape == (1, 8, 16, 16, 3)
