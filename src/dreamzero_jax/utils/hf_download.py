"""HuggingFace model downloading utilities."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def download_from_hf(
    repo_id: str,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
) -> Path:
    """Download model files from a HuggingFace repo.

    Prefers safetensors, falls back to .bin, then .pt files.

    Returns:
        Path to the downloaded checkpoint file.
    """
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for --hf downloads. "
            "Install with: uv pip install huggingface-hub"
        )

    logger.info("Listing files in HuggingFace repo: %s", repo_id)
    files = list_repo_files(repo_id, revision=revision)

    ckpt_file = _pick_checkpoint_file(files)
    if ckpt_file is None:
        raise FileNotFoundError(
            f"No checkpoint file found in {repo_id}. "
            f"Available files: {files[:20]}"
        )

    logger.info("Downloading %s from %s ...", ckpt_file, repo_id)
    local_path = hf_hub_download(
        repo_id, ckpt_file, revision=revision, cache_dir=cache_dir,
    )
    return Path(local_path)


def _pick_checkpoint_file(files: list[str]) -> str | None:
    """Pick the best checkpoint file from a list of repo files."""
    for suffix in (".safetensors", ".bin", ".pt", ".pth"):
        matches = [f for f in files if f.endswith(suffix)]
        if not matches:
            continue

        if suffix == ".safetensors":
            index = [f for f in matches if "index" in f.lower()]
            if index:
                return index[0]

        if len(matches) == 1:
            return matches[0]
        model_files = [f for f in matches if "model" in f.lower()]
        return model_files[0] if model_files else matches[0]

    return None


def load_sharded_safetensors(index_path: Path) -> dict[str, np.ndarray]:
    """Load a sharded safetensors checkpoint from its index file."""
    with open(index_path) as f:
        index = json.load(f)

    weight_map = index.get("weight_map", {})
    shard_files = set(weight_map.values())
    parent = index_path.parent

    state_dict: dict[str, np.ndarray] = {}
    for shard_name in sorted(shard_files):
        shard_path = parent / shard_name
        if not shard_path.exists():
            logger.warning("Shard not found: %s", shard_path)
            continue
        shard_data = load_single_safetensors(shard_path)
        state_dict.update(shard_data)

    logger.info(
        "Loaded %d parameters from %d shards",
        len(state_dict), len(shard_files),
    )
    return state_dict


def load_single_safetensors(path: Path) -> dict[str, np.ndarray]:
    """Load a single safetensors file as numpy arrays."""
    try:
        from safetensors.numpy import load_file
        return load_file(str(path))
    except ImportError:
        from safetensors import safe_open
        state: dict[str, np.ndarray] = {}
        with safe_open(str(path), framework="np") as f:
            for key in f.keys():
                state[key] = f.get_tensor(key)
        return state


def load_checkpoint_auto(path: Path) -> dict[str, np.ndarray]:
    """Load a checkpoint, handling directories, sharded safetensors, and files."""
    from dreamzero_jax.utils.checkpoint import load_pytorch_checkpoint

    path = Path(path)
    if path.is_dir():
        index = path / "model.safetensors.index.json"
        if index.exists():
            return load_sharded_safetensors(index)
        single = path / "model.safetensors"
        if single.exists():
            return dict(load_single_safetensors(single))
        found = find_checkpoint_file(path)
        if found:
            return load_checkpoint_auto(found)
        raise FileNotFoundError(f"No checkpoint found in {path}")

    if path.name.endswith(".index.json") or (
        path.suffix == ".json" and "safetensors" in path.name
    ):
        return load_sharded_safetensors(path)

    return load_pytorch_checkpoint(path)
