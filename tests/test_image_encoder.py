"""Tests for WAN Image Encoder (CLIP ViT)."""

import jax
import jax.numpy as jnp
from flax import nnx

from dreamzero_jax.models.image_encoder import (
    QuickGELU,
    VisionTransformer,
    VitAttentionBlock,
    VitSelfAttention,
    WanImageEncoder,
)

# Small dims for fast tests
DIM = 64
NUM_HEADS = 4
BATCH = 2
IMAGE_SIZE = 56
PATCH_SIZE = 14
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2  # 16


def test_quick_gelu():
    """QuickGELU produces correct output shape and is non-linear."""
    act = QuickGELU()
    x = jax.random.normal(jax.random.key(0), (2, 4))
    out = act(x)
    assert out.shape == (2, 4)
    # Should not be identity
    assert not jnp.allclose(out, x)


def test_vit_self_attention_shape():
    """VitSelfAttention preserves shape."""
    attn = VitSelfAttention(DIM, NUM_HEADS, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(0), (BATCH, 16, DIM))
    out = attn(x)
    assert out.shape == (BATCH, 16, DIM)


def test_vit_attention_block_prenorm():
    """VitAttentionBlock in pre-norm mode preserves shape."""
    block = VitAttentionBlock(
        DIM, mlp_ratio=4, num_heads=NUM_HEADS, post_norm=False,
        activation="gelu", rngs=nnx.Rngs(0),
    )
    x = jax.random.normal(jax.random.key(0), (BATCH, 16, DIM))
    out = block(x)
    assert out.shape == (BATCH, 16, DIM)


def test_vit_attention_block_postnorm():
    """VitAttentionBlock in post-norm mode preserves shape."""
    block = VitAttentionBlock(
        DIM, mlp_ratio=4, num_heads=NUM_HEADS, post_norm=True,
        activation="quick_gelu", rngs=nnx.Rngs(0),
    )
    x = jax.random.normal(jax.random.key(0), (BATCH, 16, DIM))
    out = block(x)
    assert out.shape == (BATCH, 16, DIM)


def test_vision_transformer_full():
    """VisionTransformer runs all blocks and returns patch+CLS tokens."""
    vit = VisionTransformer(
        image_size=IMAGE_SIZE, patch_size=PATCH_SIZE, dim=DIM,
        mlp_ratio=4, out_dim=32, num_heads=NUM_HEADS, num_layers=2,
        pool_type="token", pre_norm=True, post_norm=False,
        activation="gelu", rngs=nnx.Rngs(0),
    )
    # Channels-last input
    x = jax.random.normal(jax.random.key(0), (BATCH, IMAGE_SIZE, IMAGE_SIZE, 3))
    out = vit(x, use_31_block=False)
    # Full forward with all blocks — returns all tokens
    assert out.shape == (BATCH, 1 + NUM_PATCHES, DIM)


def test_vision_transformer_31_block():
    """VisionTransformer with use_31_block returns features without last block."""
    vit = VisionTransformer(
        image_size=IMAGE_SIZE, patch_size=PATCH_SIZE, dim=DIM,
        mlp_ratio=4, out_dim=32, num_heads=NUM_HEADS, num_layers=4,
        pool_type="token", pre_norm=True, post_norm=False,
        activation="gelu", rngs=nnx.Rngs(0),
    )
    x = jax.random.normal(jax.random.key(0), (BATCH, IMAGE_SIZE, IMAGE_SIZE, 3))
    out_31 = vit(x, use_31_block=True)
    out_full = vit(x, use_31_block=False)
    # Both should have the same token shape
    assert out_31.shape == (BATCH, 1 + NUM_PATCHES, DIM)
    assert out_full.shape == (BATCH, 1 + NUM_PATCHES, DIM)
    # But different values (one fewer block)
    assert not jnp.allclose(out_31, out_full)


def test_wan_image_encoder_encode():
    """WanImageEncoder.encode_image produces correct token shape."""
    enc = WanImageEncoder(
        image_size=IMAGE_SIZE, patch_size=PATCH_SIZE, dim=DIM,
        mlp_ratio=4, out_dim=32, num_heads=NUM_HEADS, num_layers=2,
        activation="gelu", rngs=nnx.Rngs(0),
    )
    # Images in [-1, 1] range, channels-last
    images = jax.random.uniform(jax.random.key(0), (BATCH, IMAGE_SIZE, IMAGE_SIZE, 3), minval=-1, maxval=1)
    out = enc.encode_image(images)
    assert out.shape == (BATCH, 1 + NUM_PATCHES, DIM)


def test_wan_image_encoder_resize():
    """WanImageEncoder handles images that don't match the model size."""
    enc = WanImageEncoder(
        image_size=IMAGE_SIZE, patch_size=PATCH_SIZE, dim=DIM,
        mlp_ratio=4, out_dim=32, num_heads=NUM_HEADS, num_layers=2,
        activation="gelu", rngs=nnx.Rngs(0),
    )
    # Different input resolution
    images = jax.random.uniform(jax.random.key(0), (1, 112, 112, 3), minval=-1, maxval=1)
    out = enc.encode_image(images)
    # Should still produce the correct number of tokens
    assert out.shape == (1, 1 + NUM_PATCHES, DIM)
