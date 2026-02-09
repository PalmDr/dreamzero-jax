"""Video and action transforms for the DreamZero training pipeline.

Provides functions for normalizing, resizing, cropping, and batching
data into the formats expected by the DreamZero model:

* **Video**: ``(B, T, H, W, 3)`` channels-last, float32, values in ``[-1, 1]``
* **Actions**: ``(B, num_blocks * num_action_per_block, action_dim)`` float32, ``[-1, 1]``
* **State**: ``(B, num_blocks, state_dim)`` float32
* **Token IDs**: ``(B, L)`` int32
* **Attention mask**: ``(B, L)`` float32
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Action statistics
# ---------------------------------------------------------------------------


@dataclass
class ActionStats:
    """Per-dimension action statistics for normalization.

    All arrays have shape ``(action_dim,)``.
    """

    min: jax.Array
    max: jax.Array
    mean: jax.Array
    std: jax.Array


# ---------------------------------------------------------------------------
# Video transforms
# ---------------------------------------------------------------------------


def normalize_video(video: jax.Array) -> jax.Array:
    """Convert uint8 ``[0, 255]`` video to float32 ``[-1, 1]``.

    Args:
        video: ``(*, H, W, C)`` uint8 array.

    Returns:
        ``(*, H, W, C)`` float32 array in ``[-1, 1]``.
    """
    return video.astype(jnp.float32) / 127.5 - 1.0


def denormalize_video(video: jax.Array) -> jax.Array:
    """Convert float32 ``[-1, 1]`` video back to uint8 ``[0, 255]``.

    Clips values to ``[0, 255]`` before casting.

    Args:
        video: ``(*, H, W, C)`` float32 array in ``[-1, 1]``.

    Returns:
        ``(*, H, W, C)`` uint8 array.
    """
    out = (video + 1.0) * 127.5
    return jnp.clip(out, 0, 255).astype(jnp.uint8)


def resize_video(video: jax.Array, size: tuple[int, int]) -> jax.Array:
    """Resize the spatial dimensions of a video using bilinear interpolation.

    Args:
        video: ``(T, H, W, C)`` or ``(B, T, H, W, C)`` float32 array.
        size: Target ``(H, W)`` spatial dimensions.

    Returns:
        Resized video with the same number of leading dimensions.
    """
    target_h, target_w = size
    if video.ndim == 5:
        B, T, _, _, C = video.shape
        # Flatten batch and time for resize, then reshape back
        flat = video.reshape(B * T, video.shape[2], video.shape[3], C)
        resized = jax.image.resize(flat, (B * T, target_h, target_w, C), method="bilinear")
        return resized.reshape(B, T, target_h, target_w, C)
    elif video.ndim == 4:
        T, _, _, C = video.shape
        return jax.image.resize(video, (T, target_h, target_w, C), method="bilinear")
    else:
        raise ValueError(
            f"Expected video with 4 or 5 dimensions, got {video.ndim}"
        )


def center_crop(video: jax.Array, size: tuple[int, int]) -> jax.Array:
    """Center crop the spatial dimensions of a video.

    Args:
        video: ``(T, H, W, C)`` or ``(B, T, H, W, C)`` array.
        size: Target ``(crop_h, crop_w)``.

    Returns:
        Center-cropped video.

    Raises:
        ValueError: If the crop size exceeds the video spatial dimensions.
    """
    crop_h, crop_w = size

    if video.ndim == 5:
        _, _, H, W, _ = video.shape
    elif video.ndim == 4:
        _, H, W, _ = video.shape
    else:
        raise ValueError(
            f"Expected video with 4 or 5 dimensions, got {video.ndim}"
        )

    if crop_h > H or crop_w > W:
        raise ValueError(
            f"Crop size ({crop_h}, {crop_w}) exceeds video size ({H}, {W})"
        )

    start_h = (H - crop_h) // 2
    start_w = (W - crop_w) // 2

    if video.ndim == 5:
        return jax.lax.dynamic_slice(
            video,
            (0, 0, start_h, start_w, 0),
            (video.shape[0], video.shape[1], crop_h, crop_w, video.shape[4]),
        )
    return jax.lax.dynamic_slice(
        video,
        (0, start_h, start_w, 0),
        (video.shape[0], crop_h, crop_w, video.shape[3]),
    )


def random_crop(
    video: jax.Array,
    size: tuple[int, int],
    key: jax.Array,
) -> jax.Array:
    """Random crop the spatial dimensions of a video (training augmentation).

    Args:
        video: ``(T, H, W, C)`` or ``(B, T, H, W, C)`` array.
        size: Target ``(crop_h, crop_w)``.
        key: PRNG key for random offset generation.

    Returns:
        Randomly cropped video. The same crop window is applied to all
        frames (and all batch elements if batched).

    Raises:
        ValueError: If the crop size exceeds the video spatial dimensions.
    """
    crop_h, crop_w = size

    if video.ndim == 5:
        _, _, H, W, _ = video.shape
    elif video.ndim == 4:
        _, H, W, _ = video.shape
    else:
        raise ValueError(
            f"Expected video with 4 or 5 dimensions, got {video.ndim}"
        )

    if crop_h > H or crop_w > W:
        raise ValueError(
            f"Crop size ({crop_h}, {crop_w}) exceeds video size ({H}, {W})"
        )

    key_h, key_w = jax.random.split(key)
    start_h = jax.random.randint(key_h, (), 0, H - crop_h + 1)
    start_w = jax.random.randint(key_w, (), 0, W - crop_w + 1)

    if video.ndim == 5:
        return jax.lax.dynamic_slice(
            video,
            (0, 0, start_h, start_w, 0),
            (video.shape[0], video.shape[1], crop_h, crop_w, video.shape[4]),
        )
    return jax.lax.dynamic_slice(
        video,
        (0, start_h, start_w, 0),
        (video.shape[0], crop_h, crop_w, video.shape[3]),
    )


# ---------------------------------------------------------------------------
# Action transforms
# ---------------------------------------------------------------------------


def normalize_actions(
    actions: jax.Array,
    action_stats: ActionStats,
    *,
    eps: float = 1e-8,
) -> jax.Array:
    """Normalize actions to ``[-1, 1]`` using per-dimension min/max stats.

    Applies the affine transformation::

        normalized = 2 * (actions - min) / (max - min + eps) - 1

    Args:
        actions: ``(*, action_dim)`` raw action values.
        action_stats: Per-dimension min/max statistics.
        eps: Small value to avoid division by zero.

    Returns:
        ``(*, action_dim)`` normalized actions in ``[-1, 1]``.
    """
    range_ = action_stats.max - action_stats.min + eps
    return 2.0 * (actions - action_stats.min) / range_ - 1.0


def denormalize_actions(
    actions: jax.Array,
    action_stats: ActionStats,
    *,
    eps: float = 1e-8,
) -> jax.Array:
    """Denormalize actions from ``[-1, 1]`` back to original scale.

    Inverse of :func:`normalize_actions`::

        raw = (actions + 1) / 2 * (max - min + eps) + min

    Args:
        actions: ``(*, action_dim)`` normalized actions in ``[-1, 1]``.
        action_stats: Per-dimension min/max statistics.
        eps: Small value matching the normalization epsilon.

    Returns:
        ``(*, action_dim)`` denormalized actions.
    """
    range_ = action_stats.max - action_stats.min + eps
    return (actions + 1.0) / 2.0 * range_ + action_stats.min


# ---------------------------------------------------------------------------
# Batch preparation
# ---------------------------------------------------------------------------


@dataclass
class DataConfig:
    """Configuration for data transforms applied in :func:`prepare_batch`.

    Attributes:
        video_size: Target ``(H, W)`` for video spatial dimensions.
            If ``None``, no resizing is performed.
        num_blocks: Number of temporal blocks for action chunking.
        num_action_per_block: Actions per temporal block.
        action_dim: Dimensionality of the action vector.
        state_dim: Dimensionality of the state vector.
        max_token_length: Maximum text token sequence length for padding.
        action_stats: Optional normalization statistics. If provided,
            raw actions are normalized to ``[-1, 1]``.
    """

    video_size: tuple[int, int] | None = None
    num_blocks: int = 1
    num_action_per_block: int = 32
    action_dim: int = 7
    state_dim: int = 14
    max_token_length: int = 512
    action_stats: ActionStats | None = None


def prepare_batch(
    batch_dict: dict[str, Any],
    config: DataConfig,
) -> dict[str, jax.Array]:
    """Convert a raw data dictionary to model-ready tensors.

    Expects ``batch_dict`` to contain some or all of the following keys:

    * ``"video"`` — uint8 ``(B, T, H, W, 3)`` or float32 ``(B, T, H, W, 3)``.
    * ``"actions"`` — float32 ``(B, total_actions, action_dim)``.
    * ``"state"`` — float32 ``(B, num_blocks, state_dim)``.
    * ``"token_ids"`` — int32 ``(B, L)``.
    * ``"attention_mask"`` — float32 or int ``(B, L)``.
    * ``"embodiment_id"`` — int32 ``(B,)``.

    Processing applied:

    1. **Video**: Normalized to ``[-1, 1]`` if uint8; optionally resized.
    2. **Actions**: Optionally normalized with ``config.action_stats``.
    3. **Token IDs / attention mask**: Padded or truncated to
       ``config.max_token_length``.
    4. **Embodiment ID**: Passed through as int32.

    Args:
        batch_dict: Raw data dictionary.
        config: Transform configuration.

    Returns:
        Dictionary with model-ready JAX arrays.
    """
    result: dict[str, jax.Array] = {}

    # --- Video ---
    if "video" in batch_dict:
        video = jnp.asarray(batch_dict["video"])
        # Normalize uint8 to [-1, 1]
        if video.dtype == jnp.uint8:
            video = normalize_video(video)
        # Resize if requested
        if config.video_size is not None:
            video = resize_video(video, config.video_size)
        result["video"] = video

    # --- Actions ---
    if "actions" in batch_dict:
        actions = jnp.asarray(batch_dict["actions"], dtype=jnp.float32)
        if config.action_stats is not None:
            actions = normalize_actions(actions, config.action_stats)
        result["actions"] = actions

    # --- State ---
    if "state" in batch_dict:
        result["state"] = jnp.asarray(batch_dict["state"], dtype=jnp.float32)

    # --- Token IDs ---
    if "token_ids" in batch_dict:
        token_ids = jnp.asarray(batch_dict["token_ids"], dtype=jnp.int32)
        B, L = token_ids.shape
        max_len = config.max_token_length
        if L > max_len:
            token_ids = token_ids[:, :max_len]
        elif L < max_len:
            pad_width = ((0, 0), (0, max_len - L))
            token_ids = jnp.pad(token_ids, pad_width, constant_values=0)
        result["token_ids"] = token_ids

    # --- Attention mask ---
    if "attention_mask" in batch_dict:
        mask = jnp.asarray(batch_dict["attention_mask"], dtype=jnp.float32)
        B, L = mask.shape
        max_len = config.max_token_length
        if L > max_len:
            mask = mask[:, :max_len]
        elif L < max_len:
            pad_width = ((0, 0), (0, max_len - L))
            mask = jnp.pad(mask, pad_width, constant_values=0.0)
        result["attention_mask"] = mask

    # --- Embodiment ID ---
    if "embodiment_id" in batch_dict:
        result["embodiment_id"] = jnp.asarray(
            batch_dict["embodiment_id"], dtype=jnp.int32,
        )

    return result
