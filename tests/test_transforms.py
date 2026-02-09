"""Tests for data transforms."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dreamzero_jax.data.transforms import (
    ActionStats,
    normalize_video,
    denormalize_video,
    resize_video,
    normalize_actions,
    denormalize_actions,
    center_crop,
)


def test_normalize_video_range():
    """uint8 [0, 255] -> float32 [-1, 1]."""
    video = np.random.randint(0, 256, (2, 4, 32, 32, 3), dtype=np.uint8)
    normed = normalize_video(video)
    assert normed.dtype == np.float32
    assert normed.min() >= -1.0
    assert normed.max() <= 1.0


def test_denormalize_video_roundtrip():
    """Normalize then denormalize should recover original (approximately).

    Float rounding means some values can be off by 1, so we allow atol=1.
    """
    video = np.random.randint(0, 256, (1, 2, 16, 16, 3), dtype=np.uint8)
    normed = normalize_video(video)
    recovered = denormalize_video(normed)
    np.testing.assert_allclose(recovered.astype(np.float32), video.astype(np.float32), atol=1.0)


def test_resize_video():
    """Spatial resize should change H and W."""
    video = jnp.ones((1, 2, 32, 32, 3))
    resized = resize_video(video, (16, 16))
    assert resized.shape == (1, 2, 16, 16, 3)


def test_normalize_actions():
    """Action normalization to [-1, 1]."""
    stats = ActionStats(
        min=np.array([0.0, -1.0, -2.0]),
        max=np.array([1.0, 1.0, 2.0]),
        mean=np.array([0.5, 0.0, 0.0]),
        std=np.array([0.3, 0.5, 1.0]),
    )
    actions = np.array([[0.0, -1.0, -2.0], [1.0, 1.0, 2.0]])
    normed = normalize_actions(actions, stats)
    np.testing.assert_allclose(normed[0], [-1.0, -1.0, -1.0], atol=1e-6)
    np.testing.assert_allclose(normed[1], [1.0, 1.0, 1.0], atol=1e-6)


def test_denormalize_actions_roundtrip():
    """Normalize then denormalize should recover original."""
    stats = ActionStats(
        min=np.array([0.0, -1.0]),
        max=np.array([1.0, 1.0]),
        mean=np.zeros(2),
        std=np.ones(2),
    )
    actions = np.array([[0.3, -0.5], [0.8, 0.2]])
    normed = normalize_actions(actions, stats)
    recovered = denormalize_actions(normed, stats)
    np.testing.assert_allclose(recovered, actions, atol=1e-6)


def test_center_crop():
    """Center crop should produce expected spatial dims."""
    video = jnp.ones((1, 2, 64, 48, 3))
    cropped = center_crop(video, (32, 32))
    assert cropped.shape == (1, 2, 32, 32, 3)


def test_center_crop_preserves_center():
    """Center crop should preserve center pixels."""
    video = jnp.zeros((1, 1, 8, 8, 1))
    # Put a marker at center
    video = video.at[0, 0, 3, 3, 0].set(1.0)
    video = video.at[0, 0, 4, 4, 0].set(1.0)
    cropped = center_crop(video, (4, 4))
    assert cropped.shape == (1, 1, 4, 4, 1)
    # Center pixels should be preserved
    assert cropped[0, 0, 1, 1, 0] == 1.0 or cropped[0, 0, 2, 2, 0] == 1.0
