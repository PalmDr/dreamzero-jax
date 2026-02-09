"""Dataset loader for the LeRobot/HuggingFace format.

LeRobot datasets are stored as HuggingFace datasets with columns such as:

* ``observation.images.top`` — per-frame images (PIL or path)
* ``action`` — per-frame action vectors
* ``observation.state`` — per-frame robot state
* ``episode_index`` — integer identifying which episode each row belongs to
* ``frame_index`` (or ``index``) — frame position within the episode
* ``timestamp`` — (optional) time in seconds

This module loads episodes, stacks frames into video sequences, and
aligns action horizons for the DreamZero training pipeline.

.. note::

    This module uses ``datasets`` (HuggingFace) directly and does **not**
    depend on the ``lerobot`` package.  Install with::

        pip install datasets Pillow

    If ``datasets`` is not available, the module can still be imported but
    :class:`LeRobotDataset` will raise an ``ImportError`` on instantiation.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

import numpy as np

try:
    from datasets import load_dataset as _hf_load_dataset
except ImportError:
    _hf_load_dataset = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class LeRobotDatasetConfig:
    """Configuration for :class:`LeRobotDataset`.

    Attributes:
        video_length: Number of frames per training sample.
        num_blocks: Number of temporal blocks for action chunking.
        num_action_per_block: Actions per temporal block.
        num_frames_per_block: Video frames per temporal block.
        image_key: Column name for the observation image.
        action_key: Column name for the action vector.
        state_key: Column name for the robot state.
        episode_key: Column name identifying episodes.
        frame_key: Column name for the frame index within an episode.
        image_size: If set, resize images to ``(H, W)`` during loading.
        selected_episodes: If set, only use these episode indices.
    """

    video_length: int = 16
    num_blocks: int = 1
    num_action_per_block: int = 32
    num_frames_per_block: int = 1
    image_key: str = "observation.images.top"
    action_key: str = "action"
    state_key: str = "observation.state"
    episode_key: str = "episode_index"
    frame_key: str = "frame_index"
    image_size: tuple[int, int] | None = None
    selected_episodes: list[int] | None = None


# ---------------------------------------------------------------------------
# Episode index
# ---------------------------------------------------------------------------


@dataclass
class _EpisodeInfo:
    """Internal bookkeeping for a single episode."""

    episode_id: int
    start_row: int
    length: int


# ---------------------------------------------------------------------------
# LeRobotDataset
# ---------------------------------------------------------------------------


class LeRobotDataset:
    """Dataset loader for LeRobot-format HuggingFace datasets.

    Loads a HuggingFace dataset, indexes episodes, and provides
    ``__getitem__`` / ``__len__`` for sampling fixed-length video + action
    sequences suitable for DreamZero training.

    Each sample is a dictionary with:

    * ``"video"`` — ``(T, H, W, 3)`` uint8 numpy array
    * ``"actions"`` — ``(total_actions, action_dim)`` float32 numpy array
    * ``"state"`` — ``(num_blocks, state_dim)`` float32 numpy array

    where ``total_actions = num_blocks * num_action_per_block``.

    Args:
        repo_id: HuggingFace dataset repository ID
            (e.g. ``"lerobot/pusht"``).
        split: Dataset split (``"train"``, ``"test"``, etc.).
        config: Dataset configuration. See :class:`LeRobotDatasetConfig`.
        hf_kwargs: Extra keyword arguments forwarded to
            ``datasets.load_dataset``.

    Raises:
        ImportError: If the ``datasets`` library is not installed.
    """

    def __init__(
        self,
        repo_id: str,
        split: str = "train",
        config: LeRobotDatasetConfig | None = None,
        **hf_kwargs: Any,
    ):
        if _hf_load_dataset is None:
            raise ImportError(
                "The 'datasets' package is required for LeRobotDataset. "
                "Install with: pip install datasets"
            )

        self.config = config or LeRobotDatasetConfig()
        self.repo_id = repo_id
        self.split = split

        # Load the dataset
        self._dataset = _hf_load_dataset(repo_id, split=split, **hf_kwargs)

        # Build episode index
        self._episodes: list[_EpisodeInfo] = []
        self._sample_indices: list[tuple[int, int]] = []  # (episode_idx, start_frame)
        self._build_episode_index()

    def _build_episode_index(self) -> None:
        """Scan the dataset to identify episode boundaries and valid start frames."""
        cfg = self.config
        ep_key = cfg.episode_key

        # Group rows by episode
        episode_starts: dict[int, int] = {}
        episode_lengths: dict[int, int] = {}

        for i, row in enumerate(self._dataset):
            ep_id = int(row[ep_key])
            if ep_id not in episode_starts:
                episode_starts[ep_id] = i
                episode_lengths[ep_id] = 0
            episode_lengths[ep_id] += 1

        # Build episode info
        for ep_id in sorted(episode_starts.keys()):
            # Skip if not in selected episodes
            if (
                cfg.selected_episodes is not None
                and ep_id not in cfg.selected_episodes
            ):
                continue

            info = _EpisodeInfo(
                episode_id=ep_id,
                start_row=episode_starts[ep_id],
                length=episode_lengths[ep_id],
            )
            self._episodes.append(info)

        # Build sample indices: (episode_list_index, start_frame_within_episode)
        # We need video_length frames + enough future actions for the horizon
        total_actions = cfg.num_blocks * cfg.num_action_per_block
        min_episode_len = cfg.video_length + total_actions

        for ep_idx, ep_info in enumerate(self._episodes):
            if ep_info.length < min_episode_len:
                # Episode too short — allow partial sequences by using what
                # we have, but still need at least video_length frames
                if ep_info.length < cfg.video_length:
                    continue
            # Valid start frames
            max_start = max(0, ep_info.length - min_episode_len)
            for start in range(max_start + 1):
                self._sample_indices.append((ep_idx, start))

    def __len__(self) -> int:
        return len(self._sample_indices)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        """Get a single training sample.

        Args:
            idx: Sample index in ``[0, len(self))``.

        Returns:
            Dictionary with ``"video"``, ``"actions"``, ``"state"`` arrays.
        """
        cfg = self.config
        ep_idx, start_frame = self._sample_indices[idx]
        ep_info = self._episodes[ep_idx]

        # Row indices in the HuggingFace dataset
        row_start = ep_info.start_row + start_frame

        # --- Load video frames ---
        video_frames = []
        for t in range(cfg.video_length):
            row_idx = row_start + t
            row = self._dataset[row_idx]
            img = self._load_image(row)
            video_frames.append(img)
        video = np.stack(video_frames, axis=0)  # (T, H, W, 3)

        # --- Load actions ---
        total_actions = cfg.num_blocks * cfg.num_action_per_block
        actions = []
        for t in range(total_actions):
            row_idx = row_start + t
            # Clamp to episode boundary
            row_idx = min(row_idx, ep_info.start_row + ep_info.length - 1)
            row = self._dataset[row_idx]
            action = np.array(row[cfg.action_key], dtype=np.float32)
            actions.append(action)
        actions_arr = np.stack(actions, axis=0)  # (total_actions, action_dim)

        # --- Load state ---
        # Sample one state per block at the block's starting frame
        states = []
        frames_per_block = max(1, cfg.video_length // cfg.num_blocks)
        for b in range(cfg.num_blocks):
            frame_idx = b * frames_per_block
            row_idx = row_start + frame_idx
            row_idx = min(row_idx, ep_info.start_row + ep_info.length - 1)
            row = self._dataset[row_idx]
            state_vec = np.array(row[cfg.state_key], dtype=np.float32)
            states.append(state_vec)
        state_arr = np.stack(states, axis=0)  # (num_blocks, state_dim)

        return {
            "video": video,
            "actions": actions_arr,
            "state": state_arr,
        }

    def _load_image(self, row: dict[str, Any]) -> np.ndarray:
        """Load and optionally resize an image from a dataset row.

        Handles PIL images, file paths, and numpy arrays.

        Args:
            row: A single row from the HuggingFace dataset.

        Returns:
            ``(H, W, 3)`` uint8 numpy array.
        """
        cfg = self.config
        img_data = row[cfg.image_key]

        # Handle PIL Image
        try:
            from PIL import Image as PILImage

            if isinstance(img_data, PILImage.Image):
                if cfg.image_size is not None:
                    # PIL resize uses (W, H) convention
                    img_data = img_data.resize(
                        (cfg.image_size[1], cfg.image_size[0]),
                        PILImage.BILINEAR,
                    )
                img_arr = np.array(img_data, dtype=np.uint8)
                if img_arr.ndim == 2:
                    img_arr = np.stack([img_arr] * 3, axis=-1)
                elif img_arr.shape[-1] == 4:
                    img_arr = img_arr[:, :, :3]
                return img_arr
        except ImportError:
            pass

        # Handle numpy array
        if isinstance(img_data, np.ndarray):
            img_arr = img_data.astype(np.uint8)
            if img_arr.ndim == 2:
                img_arr = np.stack([img_arr] * 3, axis=-1)
            return img_arr

        # Handle dict with 'bytes' key (HuggingFace image feature)
        if isinstance(img_data, dict) and "bytes" in img_data:
            import io

            from PIL import Image as PILImage

            img = PILImage.open(io.BytesIO(img_data["bytes"]))
            if cfg.image_size is not None:
                img = img.resize(
                    (cfg.image_size[1], cfg.image_size[0]),
                    PILImage.BILINEAR,
                )
            return np.array(img.convert("RGB"), dtype=np.uint8)

        raise TypeError(
            f"Unsupported image type: {type(img_data)}. "
            "Expected PIL Image, numpy array, or dict with 'bytes' key."
        )

    def get_episode_ids(self) -> list[int]:
        """Return the list of episode IDs available in this dataset."""
        return [ep.episode_id for ep in self._episodes]

    def get_episode_length(self, episode_id: int) -> int:
        """Return the number of frames in the given episode.

        Args:
            episode_id: The episode index.

        Raises:
            KeyError: If the episode ID is not found.
        """
        for ep in self._episodes:
            if ep.episode_id == episode_id:
                return ep.length
        raise KeyError(f"Episode {episode_id} not found in dataset")


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------


def collate_fn(batch: Sequence[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Stack a list of sample dicts into a batched dict.

    Each value in the input dicts must be a numpy array. Arrays are
    stacked along a new leading batch dimension.

    Args:
        batch: List of sample dictionaries from
            :meth:`LeRobotDataset.__getitem__`.

    Returns:
        Dictionary with batched numpy arrays.
    """
    if not batch:
        return {}

    keys = batch[0].keys()
    result: dict[str, np.ndarray] = {}
    for key in keys:
        result[key] = np.stack([sample[key] for sample in batch], axis=0)
    return result


# ---------------------------------------------------------------------------
# Data iterators
# ---------------------------------------------------------------------------


def create_train_dataloader(
    dataset: LeRobotDataset,
    batch_size: int,
    *,
    shuffle: bool = True,
    seed: int = 0,
    drop_last: bool = True,
    num_epochs: int | None = None,
) -> Iterator[dict[str, np.ndarray]]:
    """Create a training data iterator with shuffling and batching.

    Yields batched dictionaries of numpy arrays. Each batch is collated
    via :func:`collate_fn`.

    Args:
        dataset: The :class:`LeRobotDataset` to iterate over.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the dataset each epoch.
        seed: Random seed for shuffling.
        drop_last: Whether to drop the last incomplete batch.
        num_epochs: Number of epochs to iterate. ``None`` means infinite.

    Yields:
        Batched dictionaries with numpy arrays.
    """
    rng = random.Random(seed)
    n = len(dataset)
    indices = list(range(n))

    epoch = 0
    while num_epochs is None or epoch < num_epochs:
        if shuffle:
            rng.shuffle(indices)

        num_batches = n // batch_size if drop_last else math.ceil(n / batch_size)

        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, n)
            batch_indices = indices[start:end]

            batch = [dataset[i] for i in batch_indices]
            yield collate_fn(batch)

        epoch += 1


def create_eval_dataloader(
    dataset: LeRobotDataset,
    batch_size: int,
    *,
    drop_last: bool = False,
) -> Iterator[dict[str, np.ndarray]]:
    """Create a non-shuffled evaluation data iterator.

    Yields batched dictionaries of numpy arrays in sequential order.

    Args:
        dataset: The :class:`LeRobotDataset` to iterate over.
        batch_size: Number of samples per batch.
        drop_last: Whether to drop the last incomplete batch.

    Yields:
        Batched dictionaries with numpy arrays.
    """
    n = len(dataset)
    num_batches = n // batch_size if drop_last else math.ceil(n / batch_size)

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, n)

        batch = [dataset[i] for i in range(start, end)]
        yield collate_fn(batch)
