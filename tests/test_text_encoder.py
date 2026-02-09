"""Tests for WAN Text Encoder (T5-style)."""

import jax
import jax.numpy as jnp
from flax import nnx

from dreamzero_jax.models.text_encoder import (
    T5Attention,
    T5FeedForward,
    T5RelativeEmbedding,
    T5SelfAttention,
    WanTextEncoder,
)

# Small dims for fast tests
DIM = 64
DIM_ATTN = 64
DIM_FFN = 128
NUM_HEADS = 4
BATCH = 2
SEQ_LEN = 16


def test_t5_relative_embedding_shape():
    """T5RelativeEmbedding produces correct bias shape."""
    emb = T5RelativeEmbedding(num_buckets=32, num_heads=NUM_HEADS, rngs=nnx.Rngs(0))
    bias = emb(SEQ_LEN, SEQ_LEN)
    assert bias.shape == (1, NUM_HEADS, SEQ_LEN, SEQ_LEN)


def test_t5_relative_embedding_asymmetric():
    """Asymmetric query/key lengths are handled correctly."""
    emb = T5RelativeEmbedding(num_buckets=32, num_heads=NUM_HEADS, rngs=nnx.Rngs(0))
    bias = emb(8, 16)
    assert bias.shape == (1, NUM_HEADS, 8, 16)


def test_t5_attention_shape():
    """T5Attention preserves sequence shape."""
    attn = T5Attention(DIM, DIM_ATTN, NUM_HEADS, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(0), (BATCH, SEQ_LEN, DIM))
    out = attn(x)
    assert out.shape == (BATCH, SEQ_LEN, DIM)


def test_t5_attention_with_pos_bias():
    """T5Attention works with relative position bias."""
    attn = T5Attention(DIM, DIM_ATTN, NUM_HEADS, rngs=nnx.Rngs(0))
    emb = T5RelativeEmbedding(num_buckets=32, num_heads=NUM_HEADS, rngs=nnx.Rngs(1))
    x = jax.random.normal(jax.random.key(0), (BATCH, SEQ_LEN, DIM))
    bias = emb(SEQ_LEN, SEQ_LEN)
    out = attn(x, pos_bias=bias)
    assert out.shape == (BATCH, SEQ_LEN, DIM)


def test_t5_feedforward_shape():
    """T5FeedForward preserves shape."""
    ffn = T5FeedForward(DIM, DIM_FFN, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(0), (BATCH, SEQ_LEN, DIM))
    out = ffn(x)
    assert out.shape == (BATCH, SEQ_LEN, DIM)


def test_t5_self_attention_block():
    """T5SelfAttention block preserves shape."""
    block = T5SelfAttention(
        DIM, DIM_ATTN, DIM_FFN, NUM_HEADS, num_buckets=32, shared_pos=True,
        rngs=nnx.Rngs(0),
    )
    x = jax.random.normal(jax.random.key(0), (BATCH, SEQ_LEN, DIM))
    emb = T5RelativeEmbedding(num_buckets=32, num_heads=NUM_HEADS, rngs=nnx.Rngs(1))
    bias = emb(SEQ_LEN, SEQ_LEN)
    out = block(x, pos_bias=bias)
    assert out.shape == (BATCH, SEQ_LEN, DIM)


def test_wan_text_encoder_forward():
    """Full WanTextEncoder forward pass with small config."""
    enc = WanTextEncoder(
        vocab=1000, dim=DIM, dim_attn=DIM_ATTN, dim_ffn=DIM_FFN,
        num_heads=NUM_HEADS, num_layers=2, num_buckets=32, shared_pos=True,
        rngs=nnx.Rngs(0),
    )
    ids = jnp.zeros((BATCH, SEQ_LEN), dtype=jnp.int32)
    out = enc(ids)
    assert out.shape == (BATCH, SEQ_LEN, DIM)


def test_wan_text_encoder_with_mask():
    """WanTextEncoder works with attention mask."""
    enc = WanTextEncoder(
        vocab=1000, dim=DIM, dim_attn=DIM_ATTN, dim_ffn=DIM_FFN,
        num_heads=NUM_HEADS, num_layers=2, num_buckets=32, shared_pos=True,
        rngs=nnx.Rngs(0),
    )
    ids = jnp.zeros((BATCH, SEQ_LEN), dtype=jnp.int32)
    mask = jnp.ones((BATCH, SEQ_LEN))
    mask = mask.at[:, 10:].set(0)
    out = enc(ids, mask=mask)
    assert out.shape == (BATCH, SEQ_LEN, DIM)


def test_wan_text_encoder_non_shared_pos():
    """WanTextEncoder with per-layer position embeddings."""
    enc = WanTextEncoder(
        vocab=1000, dim=DIM, dim_attn=DIM_ATTN, dim_ffn=DIM_FFN,
        num_heads=NUM_HEADS, num_layers=2, num_buckets=32, shared_pos=False,
        rngs=nnx.Rngs(0),
    )
    ids = jnp.zeros((BATCH, SEQ_LEN), dtype=jnp.int32)
    out = enc(ids)
    assert out.shape == (BATCH, SEQ_LEN, DIM)
