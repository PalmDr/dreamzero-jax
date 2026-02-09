"""Tests for WAN DiT backbone."""

import jax
import jax.numpy as jnp
from flax import nnx

from dreamzero_jax.nn.attention import Attention
from dreamzero_jax.nn.embed import WanRoPE3D
from dreamzero_jax.models.dit import (
    MLPProj,
    WanDiT,
    WanDiTBlock,
    WanDiTHead,
    WanI2VCrossAttention,
    unpatchify,
)

# Small dims for fast tests
DIM = 192
NUM_HEADS = 6
HEAD_DIM = DIM // NUM_HEADS  # 32
FFN_DIM = 512
NUM_LAYERS = 2
BATCH = 2


def test_dit_block_shape():
    """WanDiTBlock preserves input sequence shape."""
    block = WanDiTBlock(
        dim=DIM, num_heads=NUM_HEADS, ffn_dim=FFN_DIM, rngs=nnx.Rngs(0),
    )
    S = 24
    x = jax.random.normal(jax.random.key(0), (BATCH, S, DIM))
    e = jax.random.normal(jax.random.key(1), (BATCH, 6, DIM))
    ctx = jax.random.normal(jax.random.key(2), (BATCH, 10, DIM))
    rope = WanRoPE3D(HEAD_DIM)
    freqs = rope(2, 3, 4)  # 2*3*4 = 24 = S
    out = block(x, e, ctx, freqs)
    assert out.shape == (BATCH, S, DIM)


def test_dit_block_with_image_input():
    """WanDiTBlock with I2V cross-attention variant."""
    block = WanDiTBlock(
        dim=DIM, num_heads=NUM_HEADS, ffn_dim=FFN_DIM,
        has_image_input=True, rngs=nnx.Rngs(0),
    )
    S = 24
    x = jax.random.normal(jax.random.key(0), (BATCH, S, DIM))
    e = jax.random.normal(jax.random.key(1), (BATCH, 6, DIM))
    # Context: 257 image tokens + 10 text tokens
    ctx = jax.random.normal(jax.random.key(2), (BATCH, 267, DIM))
    rope = WanRoPE3D(HEAD_DIM)
    freqs = rope(2, 3, 4)
    out = block(x, e, ctx, freqs)
    assert out.shape == (BATCH, S, DIM)


def test_dit_head_shape():
    """WanDiTHead projects to correct output dimension."""
    patch_size = (1, 2, 2)
    out_channels = 16
    head = WanDiTHead(
        DIM, out_channels, patch_size, rngs=nnx.Rngs(0),
    )
    S = 48
    x = jax.random.normal(jax.random.key(0), (BATCH, S, DIM))
    t = jax.random.normal(jax.random.key(1), (BATCH, DIM))
    out = head(x, t)
    patch_vol = 1 * 2 * 2
    assert out.shape == (BATCH, S, patch_vol * out_channels)


def test_unpatchify_shape():
    """unpatchify reconstructs correct video shape."""
    grid_size = (4, 8, 8)
    patch_size = (1, 2, 2)
    out_channels = 16
    patch_vol = 1 * 2 * 2
    S = 4 * 8 * 8
    x = jax.random.normal(jax.random.key(0), (BATCH, S, patch_vol * out_channels))
    out = unpatchify(x, grid_size, patch_size, out_channels)
    assert out.shape == (BATCH, 4 * 1, 8 * 2, 8 * 2, out_channels)


def test_wan_dit_forward():
    """Full WanDiT model end-to-end with small config."""
    model = WanDiT(
        dim=DIM,
        in_channels=4,
        out_channels=4,
        ffn_dim=FFN_DIM,
        freq_dim=64,
        text_dim=128,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        patch_size=(1, 2, 2),
        rngs=nnx.Rngs(0),
    )
    # Input: (B, T, H, W, C) channels-last
    x = jax.random.normal(jax.random.key(0), (BATCH, 4, 8, 8, 4))
    timestep = jax.random.uniform(jax.random.key(1), (BATCH,))
    context = jax.random.normal(jax.random.key(2), (BATCH, 10, 128))
    out = model(x, timestep, context)
    assert out.shape == (BATCH, 4, 8, 8, 4)


def test_wan_dit_with_image():
    """Full WanDiT model with image conditioning (I2V variant)."""
    model = WanDiT(
        dim=DIM,
        in_channels=4,
        out_channels=4,
        ffn_dim=FFN_DIM,
        freq_dim=64,
        text_dim=128,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        patch_size=(1, 2, 2),
        has_image_input=True,
        rngs=nnx.Rngs(0),
    )
    x = jax.random.normal(jax.random.key(0), (BATCH, 4, 8, 8, 4))
    timestep = jax.random.uniform(jax.random.key(1), (BATCH,))
    context = jax.random.normal(jax.random.key(2), (BATCH, 10, 128))
    clip_emb = jax.random.normal(jax.random.key(3), (BATCH, 257, 1280))
    out = model(x, timestep, context, clip_emb=clip_emb)
    assert out.shape == (BATCH, 4, 8, 8, 4)


def test_wan_rope_3d_shape():
    """WanRoPE3D returns correct frequency tensor shape."""
    rope = WanRoPE3D(HEAD_DIM)
    freqs = rope(4, 8, 8)
    assert freqs.shape == (4 * 8 * 8, HEAD_DIM // 2)
    assert jnp.iscomplexobj(freqs)


def test_wan_rope_3d_different_grids():
    """Different grid sizes produce different-shaped outputs."""
    rope = WanRoPE3D(HEAD_DIM)
    freqs1 = rope(2, 4, 4)
    freqs2 = rope(4, 8, 8)
    assert freqs1.shape == (2 * 4 * 4, HEAD_DIM // 2)
    assert freqs2.shape == (4 * 8 * 8, HEAD_DIM // 2)
    assert freqs1.shape != freqs2.shape


def test_modulation_init_scale():
    """Modulation parameters are initialized with std ~ 1/sqrt(dim)."""
    block = WanDiTBlock(
        dim=DIM, num_heads=NUM_HEADS, ffn_dim=FFN_DIM, rngs=nnx.Rngs(42),
    )
    mod = block.modulation[...]
    assert mod.shape == (1, 6, DIM)
    # std should be approximately 1/sqrt(DIM); allow generous tolerance
    expected_std = 1.0 / DIM**0.5
    actual_std = float(jnp.std(mod))
    assert actual_std < expected_std * 3.0, f"std={actual_std}, expected ~{expected_std}"


def test_qk_norm_in_attention():
    """Attention with qk_norm creates RMSNorm layers and runs correctly."""
    attn = Attention(dim=DIM, num_heads=NUM_HEADS, qk_norm=True, rngs=nnx.Rngs(0))
    assert hasattr(attn, "norm_q")
    assert hasattr(attn, "norm_k")

    x = jax.random.normal(jax.random.key(0), (BATCH, 16, DIM))
    out = attn(x)
    assert out.shape == (BATCH, 16, DIM)


def test_cross_attn_no_modulation():
    """Cross-attention path has no adaptive modulation (shift/scale/gate)."""
    block = WanDiTBlock(
        dim=DIM, num_heads=NUM_HEADS, ffn_dim=FFN_DIM, rngs=nnx.Rngs(0),
    )
    # The cross_attn is a plain Attention module, not modulated
    assert isinstance(block.cross_attn, Attention)
    # norm3 should not exist when cross_attn_norm=False (default)
    assert not block.cross_attn_norm
    assert not hasattr(block, "norm3")


def test_deterministic_dit_block():
    """Same inputs produce identical outputs (deterministic)."""
    block = WanDiTBlock(
        dim=DIM, num_heads=NUM_HEADS, ffn_dim=FFN_DIM, rngs=nnx.Rngs(0),
    )
    S = 24
    x = jax.random.normal(jax.random.key(0), (BATCH, S, DIM))
    e = jax.random.normal(jax.random.key(1), (BATCH, 6, DIM))
    ctx = jax.random.normal(jax.random.key(2), (BATCH, 10, DIM))
    rope = WanRoPE3D(HEAD_DIM)
    freqs = rope(2, 3, 4)

    out1 = block(x, e, ctx, freqs)
    out2 = block(x, e, ctx, freqs)
    assert jnp.allclose(out1, out2)
