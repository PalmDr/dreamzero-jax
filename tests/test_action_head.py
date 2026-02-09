"""Tests for action head modules."""

import jax
import jax.numpy as jnp
from flax import nnx

from dreamzero_jax.models.action_head import (
    CategorySpecificLinear,
    CategorySpecificMLP,
    CausalWanDiT,
    MultiEmbodimentActionEncoder,
    make_action_causal_mask,
)

# Small dims for fast tests
DIM = 64
FFN_DIM = 128
NUM_HEADS = 4
FREQ_DIM = 32
TEXT_DIM = 64
ACTION_DIM = 7
STATE_DIM = 14
ACTION_HIDDEN = 64
BATCH = 2
NUM_CATS = 4  # small number of embodiments
# Video: 2 frames, 8x8 spatial, 16 latent channels
# Patch (1,2,2) -> f=2, h=4, w=4, frame_seqlen=16
NUM_FRAMES = 2
IMAGE_H = 8
IMAGE_W = 8
IN_CHANNELS = 16
PATCH_SIZE = (1, 2, 2)
FRAMES_PER_BLOCK = 1
NUM_BLOCKS = NUM_FRAMES // FRAMES_PER_BLOCK  # 2
NUM_ACTION_PER_BLOCK = 4
NUM_STATE_PER_BLOCK = 1
FRAME_SEQLEN = (IMAGE_H // PATCH_SIZE[1]) * (IMAGE_W // PATCH_SIZE[2])  # 16
BLOCK_VIDEO_TOKENS = FRAMES_PER_BLOCK * FRAME_SEQLEN  # 16
TOTAL_ACTIONS = NUM_BLOCKS * NUM_ACTION_PER_BLOCK  # 8
TEXT_LEN = 8


# ---------------------------------------------------------------------------
# CategorySpecificLinear
# ---------------------------------------------------------------------------


def test_category_specific_linear_shape():
    """CategorySpecificLinear produces correct output shape."""
    layer = CategorySpecificLinear(NUM_CATS, 10, 20, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(0), (BATCH, 5, 10))
    cat_ids = jnp.array([0, 2])
    out = layer(x, cat_ids)
    assert out.shape == (BATCH, 5, 20)


def test_category_specific_linear_2d():
    """CategorySpecificLinear works with 2D input (no sequence dim)."""
    layer = CategorySpecificLinear(NUM_CATS, 10, 20, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(0), (BATCH, 10))
    cat_ids = jnp.array([0, 1])
    out = layer(x, cat_ids)
    assert out.shape == (BATCH, 20)


def test_category_specific_linear_different_categories():
    """Different category IDs produce different outputs."""
    layer = CategorySpecificLinear(NUM_CATS, 10, 20, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(0), (1, 5, 10))
    x_dup = jnp.concatenate([x, x], axis=0)  # same input
    cat_ids = jnp.array([0, 1])  # different embodiments
    out = layer(x_dup, cat_ids)
    assert not jnp.allclose(out[0], out[1])


# ---------------------------------------------------------------------------
# CategorySpecificMLP
# ---------------------------------------------------------------------------


def test_category_specific_mlp_shape():
    """CategorySpecificMLP produces correct output shape."""
    mlp = CategorySpecificMLP(NUM_CATS, 10, 32, 20, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(0), (BATCH, 5, 10))
    cat_ids = jnp.array([0, 3])
    out = mlp(x, cat_ids)
    assert out.shape == (BATCH, 5, 20)


# ---------------------------------------------------------------------------
# MultiEmbodimentActionEncoder
# ---------------------------------------------------------------------------


def test_action_encoder_shape():
    """MultiEmbodimentActionEncoder produces correct output shape."""
    enc = MultiEmbodimentActionEncoder(
        ACTION_DIM, DIM, NUM_CATS, rngs=nnx.Rngs(0),
    )
    actions = jax.random.normal(jax.random.key(0), (BATCH, TOTAL_ACTIONS, ACTION_DIM))
    timesteps = jnp.array([500.0, 200.0])
    cat_ids = jnp.array([0, 1])
    out = enc(actions, timesteps, cat_ids)
    assert out.shape == (BATCH, TOTAL_ACTIONS, DIM)


# ---------------------------------------------------------------------------
# make_action_causal_mask
# ---------------------------------------------------------------------------


def test_mask_shape_with_clean():
    """Mask has correct shape with teacher forcing."""
    clean_len = NUM_BLOCKS * BLOCK_VIDEO_TOKENS
    block_noisy = BLOCK_VIDEO_TOKENS + NUM_ACTION_PER_BLOCK + NUM_STATE_PER_BLOCK
    noisy_len = NUM_BLOCKS * block_noisy
    total = clean_len + noisy_len

    mask = make_action_causal_mask(
        NUM_BLOCKS, BLOCK_VIDEO_TOKENS,
        NUM_ACTION_PER_BLOCK, NUM_STATE_PER_BLOCK,
        has_clean=True,
    )
    assert mask.shape == (total, total)
    assert mask.dtype == jnp.bool_


def test_mask_shape_without_clean():
    """Mask has correct shape without teacher forcing."""
    block_noisy = BLOCK_VIDEO_TOKENS + NUM_ACTION_PER_BLOCK + NUM_STATE_PER_BLOCK
    total = NUM_BLOCKS * block_noisy

    mask = make_action_causal_mask(
        NUM_BLOCKS, BLOCK_VIDEO_TOKENS,
        NUM_ACTION_PER_BLOCK, NUM_STATE_PER_BLOCK,
        has_clean=False,
    )
    assert mask.shape == (total, total)


def test_mask_clean_is_block_causal():
    """Clean video section is block-causal."""
    mask = make_action_causal_mask(
        NUM_BLOCKS, BLOCK_VIDEO_TOKENS,
        NUM_ACTION_PER_BLOCK, NUM_STATE_PER_BLOCK,
        has_clean=True,
    )
    clean_len = NUM_BLOCKS * BLOCK_VIDEO_TOKENS
    clean_section = mask[:clean_len, :clean_len]

    # Block 0 should not attend to block 1
    b0_end = BLOCK_VIDEO_TOKENS
    b1_start = BLOCK_VIDEO_TOKENS
    assert not clean_section[0, b1_start].item()

    # Block 1 should attend to block 0
    assert clean_section[b1_start, 0].item()

    # Block 0 attends to itself
    assert clean_section[0, 0].item()


def test_mask_state_is_local():
    """State tokens only attend to own state (not video/action)."""
    mask = make_action_causal_mask(
        NUM_BLOCKS, BLOCK_VIDEO_TOKENS,
        NUM_ACTION_PER_BLOCK, NUM_STATE_PER_BLOCK,
        has_clean=False,
    )
    block_noisy = BLOCK_VIDEO_TOKENS + NUM_ACTION_PER_BLOCK + NUM_STATE_PER_BLOCK

    for i in range(NUM_BLOCKS):
        state_pos = i * block_noisy + BLOCK_VIDEO_TOKENS + NUM_ACTION_PER_BLOCK
        # State attends to itself
        assert mask[state_pos, state_pos].item()
        # State does NOT attend to own video tokens
        video_pos = i * block_noisy
        assert not mask[state_pos, video_pos].item()


# ---------------------------------------------------------------------------
# CausalWanDiT
# ---------------------------------------------------------------------------


def _make_small_model(rngs=None) -> CausalWanDiT:
    """Create a small CausalWanDiT for testing."""
    if rngs is None:
        rngs = nnx.Rngs(0)
    return CausalWanDiT(
        dim=DIM,
        in_channels=IN_CHANNELS,
        out_channels=IN_CHANNELS,
        ffn_dim=FFN_DIM,
        freq_dim=FREQ_DIM,
        text_dim=TEXT_DIM,
        num_heads=NUM_HEADS,
        num_layers=2,
        patch_size=PATCH_SIZE,
        has_image_input=False,
        qk_norm=True,
        action_dim=ACTION_DIM,
        state_dim=STATE_DIM,
        action_hidden_size=ACTION_HIDDEN,
        num_action_per_block=NUM_ACTION_PER_BLOCK,
        num_state_per_block=NUM_STATE_PER_BLOCK,
        num_frames_per_block=FRAMES_PER_BLOCK,
        max_num_embodiments=NUM_CATS,
        rngs=rngs,
    )


def test_causal_wan_dit_forward_with_teacher_forcing():
    """CausalWanDiT produces correct shapes with teacher forcing."""
    model = _make_small_model()
    key = jax.random.key(0)
    x = jax.random.normal(key, (BATCH, NUM_FRAMES, IMAGE_H, IMAGE_W, IN_CHANNELS))
    clean_x = jax.random.normal(jax.random.key(1), x.shape)
    timestep = jnp.array([500.0, 200.0])
    context = jax.random.normal(jax.random.key(2), (BATCH, TEXT_LEN, TEXT_DIM))
    state = jax.random.normal(jax.random.key(3), (BATCH, NUM_BLOCKS, STATE_DIM))
    embodiment_id = jnp.array([0, 1])
    actions = jax.random.normal(jax.random.key(4), (BATCH, TOTAL_ACTIONS, ACTION_DIM))

    video_pred, action_pred = model(
        x, timestep, context, state, embodiment_id, actions,
        clean_x=clean_x,
    )

    assert video_pred.shape == (BATCH, NUM_FRAMES, IMAGE_H, IMAGE_W, IN_CHANNELS)
    assert action_pred.shape == (BATCH, TOTAL_ACTIONS, ACTION_DIM)


def test_causal_wan_dit_forward_without_teacher_forcing():
    """CausalWanDiT produces correct shapes without teacher forcing."""
    model = _make_small_model()
    key = jax.random.key(0)
    x = jax.random.normal(key, (BATCH, NUM_FRAMES, IMAGE_H, IMAGE_W, IN_CHANNELS))
    timestep = jnp.array([500.0, 200.0])
    context = jax.random.normal(jax.random.key(2), (BATCH, TEXT_LEN, TEXT_DIM))
    state = jax.random.normal(jax.random.key(3), (BATCH, NUM_BLOCKS, STATE_DIM))
    embodiment_id = jnp.array([0, 1])
    actions = jax.random.normal(jax.random.key(4), (BATCH, TOTAL_ACTIONS, ACTION_DIM))

    video_pred, action_pred = model(
        x, timestep, context, state, embodiment_id, actions,
    )

    assert video_pred.shape == (BATCH, NUM_FRAMES, IMAGE_H, IMAGE_W, IN_CHANNELS)
    assert action_pred.shape == (BATCH, TOTAL_ACTIONS, ACTION_DIM)


def test_causal_wan_dit_different_embodiments():
    """Different embodiment IDs produce different action predictions."""
    model = _make_small_model()
    key = jax.random.key(0)
    x = jax.random.normal(key, (BATCH, NUM_FRAMES, IMAGE_H, IMAGE_W, IN_CHANNELS))
    timestep = jnp.array([500.0, 500.0])
    context = jax.random.normal(jax.random.key(2), (BATCH, TEXT_LEN, TEXT_DIM))
    state = jax.random.normal(jax.random.key(3), (BATCH, NUM_BLOCKS, STATE_DIM))
    actions = jax.random.normal(jax.random.key(4), (BATCH, TOTAL_ACTIONS, ACTION_DIM))

    # Same inputs, different embodiments
    _, action_pred_0 = model(
        x, timestep, context, state, jnp.array([0, 0]), actions,
    )
    _, action_pred_1 = model(
        x, timestep, context, state, jnp.array([1, 1]), actions,
    )

    assert not jnp.allclose(action_pred_0, action_pred_1)
