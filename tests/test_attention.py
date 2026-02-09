"""Tests for attention mechanisms."""

import jax
import jax.numpy as jnp
from flax import nnx

from dreamzero_jax.nn.attention import Attention, make_causal_mask, make_causal_chunk_mask
from dreamzero_jax.nn.embed import precompute_freqs_cis


def test_self_attention_shape():
    """Self-attention preserves input shape."""
    attn = Attention(dim=256, num_heads=8, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(0), (2, 16, 256))
    out = attn(x)
    assert out.shape == (2, 16, 256)


def test_self_attention_with_rope():
    """Self-attention works with rotary position embeddings."""
    attn = Attention(dim=256, num_heads=8, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(0), (2, 16, 256))
    head_dim = 256 // 8
    freqs = precompute_freqs_cis(dim=head_dim, max_len=16)
    out = attn(x, freqs_cis=freqs)
    assert out.shape == (2, 16, 256)


def test_cross_attention_context_dim():
    """Cross-attention with different context dimension."""
    attn = Attention(dim=256, num_heads=8, context_dim=512, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(0), (2, 16, 256))
    ctx = jax.random.normal(jax.random.key(1), (2, 32, 512))
    out = attn(x, context=ctx)
    assert out.shape == (2, 16, 256)


def test_cross_attention_different_lengths():
    """Cross-attention with different Q/KV sequence lengths."""
    attn = Attention(dim=256, num_heads=8, context_dim=256, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(0), (2, 10, 256))
    ctx = jax.random.normal(jax.random.key(1), (2, 50, 256))
    out = attn(x, context=ctx)
    assert out.shape == (2, 10, 256)


def test_attention_with_causal_mask():
    """Attention applies a causal mask correctly."""
    attn = Attention(dim=64, num_heads=4, rngs=nnx.Rngs(0))
    seq_len = 8
    x = jax.random.normal(jax.random.key(0), (1, seq_len, 64))
    mask = make_causal_mask(seq_len)
    out = attn(x, mask=mask)
    assert out.shape == (1, seq_len, 64)


def test_explicit_head_dim():
    """Attention with explicit head_dim independent of dim // num_heads."""
    attn = Attention(dim=1536, num_heads=12, head_dim=128, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(0), (1, 8, 1536))
    out = attn(x)
    assert out.shape == (1, 8, 1536)
    assert attn.head_dim == 128
    assert attn.num_heads == 12


def test_no_bias():
    """Attention without bias in projections."""
    attn = Attention(dim=128, num_heads=4, qkv_bias=False, out_bias=False, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(0), (1, 8, 128))
    out = attn(x)
    assert out.shape == (1, 8, 128)
    assert not attn.q_proj.use_bias
    assert not attn.out_proj.use_bias


def test_causal_mask_shape_and_structure():
    """Causal mask is lower-triangular with correct shape."""
    mask = make_causal_mask(6)
    assert mask.shape == (6, 6)
    assert mask.dtype == jnp.bool_
    # Upper triangle (excluding diagonal) should be False
    assert not jnp.any(jnp.triu(mask, k=1))
    # Diagonal should be True
    assert jnp.all(jnp.diag(mask))
    # Lower triangle (including diagonal) should be True
    assert jnp.all(mask[jnp.tril_indices(6)])


def test_chunk_mask_shape():
    """Chunk mask has correct shape."""
    mask = make_causal_chunk_mask(seq_len=24, frame_seqlen=4)
    assert mask.shape == (24, 24)
    assert mask.dtype == jnp.bool_


def test_chunk_mask_single_frame_blocks():
    """Chunk mask with 1-frame blocks is block-causal."""
    frame_seqlen = 4
    num_frames = 3
    seq_len = frame_seqlen * num_frames  # 12
    mask = make_causal_chunk_mask(seq_len, frame_seqlen, num_frames_per_block=1)

    # Tokens in frame 0 (positions 0-3) should attend to each other
    assert jnp.all(mask[:4, :4])
    # Tokens in frame 1 (positions 4-7) should attend to frames 0 and 1
    assert jnp.all(mask[4:8, :8])
    # Tokens in frame 0 should NOT attend to frame 1
    assert not jnp.any(mask[:4, 4:8])


def test_chunk_mask_multi_frame_blocks():
    """Chunk mask groups multiple frames into one causal block."""
    frame_seqlen = 4
    num_frames = 4
    seq_len = frame_seqlen * num_frames  # 16
    mask = make_causal_chunk_mask(seq_len, frame_seqlen, num_frames_per_block=2)

    # Block 0 = frames 0-1 (positions 0-7), Block 1 = frames 2-3 (positions 8-15)
    # Within block 0: full attention
    assert jnp.all(mask[:8, :8])
    # Within block 1: full attention
    assert jnp.all(mask[8:16, 8:16])
    # Block 1 attends to block 0
    assert jnp.all(mask[8:16, :8])
    # Block 0 does NOT attend to block 1
    assert not jnp.any(mask[:8, 8:16])


def test_deterministic_output():
    """Same inputs produce identical outputs (no dropout or stochasticity)."""
    attn = Attention(dim=128, num_heads=4, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(42), (1, 8, 128))
    out1 = attn(x)
    out2 = attn(x)
    assert jnp.allclose(out1, out2)


if __name__ == "__main__":
    test_self_attention_shape()
    test_self_attention_with_rope()
    test_cross_attention_context_dim()
    test_cross_attention_different_lengths()
    test_attention_with_causal_mask()
    test_explicit_head_dim()
    test_no_bias()
    test_causal_mask_shape_and_structure()
    test_chunk_mask_shape()
    test_chunk_mask_single_frame_blocks()
    test_chunk_mask_multi_frame_blocks()
    test_deterministic_output()
    print("All tests passed!")
