#!/usr/bin/env python3
"""DreamZero inference pipeline.

Runs the full DreamZero model on an input video and text prompt, producing
predicted actions and (optionally) decoded video frames.

Usage::

    uv run python scripts/inference.py \
        --checkpoint /path/to/flax_ckpt \
        --input-video /path/to/video.mp4 \
        --prompt "pick up the red block" \
        --output-dir ./outputs \
        --num-steps 16 \
        --cfg-scale 5.0

See ``--help`` for all options.
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import time
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from dreamzero_jax.models.dreamzero import DreamZero, DreamZeroConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Video I/O helpers
# ---------------------------------------------------------------------------


def _load_video_from_frames(frames_dir: pathlib.Path, max_frames: int | None = None) -> np.ndarray:
    """Load video frames from a directory of image files.

    Reads all ``.png`` / ``.jpg`` / ``.jpeg`` images sorted by filename.

    Args:
        frames_dir: Directory containing ordered frame images.
        max_frames: If set, only load the first *max_frames* frames.

    Returns:
        ``(T, H, W, 3)`` uint8 numpy array.
    """
    from PIL import Image

    exts = {".png", ".jpg", ".jpeg"}
    paths = sorted(
        p for p in frames_dir.iterdir()
        if p.suffix.lower() in exts
    )
    if max_frames is not None:
        paths = paths[:max_frames]

    if not paths:
        raise FileNotFoundError(f"No image files found in {frames_dir}")

    frames = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        frames.append(np.array(img))

    return np.stack(frames, axis=0)


def _load_video_from_mp4(video_path: pathlib.Path, max_frames: int | None = None) -> np.ndarray:
    """Load video frames from an mp4 file using PIL/imageio-like approach.

    Falls back to decoding frame-by-frame with a simple cv2-free method
    using the ``imageio`` library if available, otherwise attempts ``cv2``.

    Args:
        video_path: Path to ``.mp4`` video file.
        max_frames: If set, only load the first *max_frames* frames.

    Returns:
        ``(T, H, W, 3)`` uint8 numpy array.
    """
    try:
        import imageio.v3 as iio

        frames_iter = iio.imread(str(video_path), plugin="pyav")
        # imageio returns (T, H, W, C) directly for video
        frames = np.asarray(frames_iter)
        if max_frames is not None:
            frames = frames[:max_frames]
        return frames
    except ImportError:
        pass

    try:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # cv2 loads BGR, convert to RGB
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if max_frames is not None and len(frames) >= max_frames:
                break
        cap.release()
        if not frames:
            raise RuntimeError(f"Could not read frames from {video_path}")
        return np.stack(frames, axis=0)
    except ImportError:
        raise ImportError(
            "Loading mp4 videos requires either 'imageio[pyav]' or 'opencv-python'. "
            "Install one of them or provide a directory of frame images instead."
        )


def load_video(path: pathlib.Path, max_frames: int | None = None) -> np.ndarray:
    """Load video from either an mp4 file or a directory of frames.

    Args:
        path: Path to an ``.mp4`` file or a directory of frame images.
        max_frames: Optional limit on the number of frames to load.

    Returns:
        ``(T, H, W, 3)`` uint8 numpy array.
    """
    if path.is_dir():
        return _load_video_from_frames(path, max_frames=max_frames)
    elif path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
        return _load_video_from_mp4(path, max_frames=max_frames)
    else:
        raise ValueError(
            f"Unsupported video input: {path}. "
            "Provide an mp4 file or a directory of frame images."
        )


def normalize_video(video: np.ndarray) -> np.ndarray:
    """Normalize uint8 video frames to ``[-1, 1]`` float32.

    Args:
        video: ``(T, H, W, 3)`` or ``(B, T, H, W, 3)`` uint8.

    Returns:
        Same shape, float32 in ``[-1, 1]``.
    """
    return (video.astype(np.float32) / 127.5) - 1.0


def denormalize_video(video: np.ndarray) -> np.ndarray:
    """Denormalize float32 ``[-1, 1]`` video to uint8 ``[0, 255]``.

    Args:
        video: Float32 array in ``[-1, 1]``.

    Returns:
        Same shape, uint8 in ``[0, 255]``.
    """
    return np.clip((video + 1.0) * 127.5, 0, 255).astype(np.uint8)


def save_video_frames(
    video: np.ndarray,
    output_dir: pathlib.Path,
    prefix: str = "frame",
) -> list[pathlib.Path]:
    """Save video frames as individual PNG images.

    Args:
        video: ``(T, H, W, 3)`` uint8 array.
        output_dir: Directory to save frames into (created if needed).
        prefix: Filename prefix for each frame.

    Returns:
        List of saved file paths.
    """
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(video.shape[0]):
        path = output_dir / f"{prefix}_{i:04d}.png"
        Image.fromarray(video[i]).save(path)
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------


def tokenize_prompt(
    prompt: str,
    tokenizer_name: str = "google/t5-v1_1-xxl",
    max_length: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Tokenize a text prompt using a HuggingFace T5 tokenizer.

    Args:
        prompt: Text instruction string.
        tokenizer_name: HuggingFace model ID for the tokenizer.
        max_length: Maximum sequence length (padded/truncated).

    Returns:
        Tuple of ``(token_ids, attention_mask)``, each ``(1, L)`` int32/float32.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    encoding = tokenizer(
        prompt,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )

    token_ids = encoding["input_ids"].astype(np.int32)  # (1, L)
    attention_mask = encoding["attention_mask"].astype(np.float32)  # (1, L)

    return token_ids, attention_mask


# ---------------------------------------------------------------------------
# Mesh / sharding helpers
# ---------------------------------------------------------------------------


def create_mesh(mesh_shape: Sequence[int]) -> jax.sharding.Mesh:
    """Create a JAX device mesh for distributed inference.

    Args:
        mesh_shape: Tuple of ``(data_parallel, model_parallel)`` sizes.
            Product must equal the total number of available devices.

    Returns:
        A ``jax.sharding.Mesh`` with axes ``('data', 'model')``.
    """
    devices = jax.devices()
    num_devices = len(devices)
    dp, mp = mesh_shape
    if dp * mp != num_devices:
        raise ValueError(
            f"Mesh shape {mesh_shape} requires {dp * mp} devices, "
            f"but only {num_devices} are available."
        )
    device_grid = np.array(devices).reshape(mesh_shape)
    return jax.sharding.Mesh(device_grid, axis_names=("data", "model"))


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def load_checkpoint(checkpoint_path: pathlib.Path, model: DreamZero) -> DreamZero:
    """Load an orbax checkpoint into the DreamZero model.

    Attempts to use the project's own checkpoint utilities. Falls back to
    direct orbax loading if the utilities are not yet implemented.

    Args:
        checkpoint_path: Path to the orbax checkpoint directory.
        model: An initialized ``DreamZero`` model (provides the tree structure).

    Returns:
        The model with loaded weights.
    """
    try:
        from dreamzero_jax.utils.checkpoint import apply_to_model, load_flax_checkpoint

        state = load_flax_checkpoint(checkpoint_path)
        model = apply_to_model(model, state)
        logger.info("Loaded checkpoint via dreamzero_jax.utils.checkpoint")
        return model
    except (ImportError, AttributeError, NotImplementedError):
        logger.info(
            "dreamzero_jax.utils.checkpoint not available, "
            "falling back to orbax directly"
        )

    import orbax.checkpoint as ocp

    checkpointer = ocp.PyTreeCheckpointer()
    # Extract the abstract state tree from the model
    graphdef, state = nnx.split(model)
    restored_state = checkpointer.restore(str(checkpoint_path), item=state)
    model = nnx.merge(graphdef, restored_state)
    logger.info("Loaded checkpoint via orbax.checkpoint.PyTreeCheckpointer")
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def run_inference(
    model: DreamZero,
    video: jax.Array,
    token_ids: jax.Array,
    attention_mask: jax.Array,
    state: jax.Array,
    embodiment_id: jax.Array,
    num_inference_steps: int,
    cfg_scale: float,
    seed: int,
    use_jit: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """Run the DreamZero generation pipeline.

    Optionally JIT-compiles ``model.generate`` for better performance.

    Args:
        model: Initialized DreamZero model (with loaded weights).
        video: Input video ``(B, T, H, W, 3)`` in ``[-1, 1]``.
        token_ids: Text token IDs ``(B, L)`` int32.
        attention_mask: Text attention mask ``(B, L)`` float32.
        state: Robot state ``(B, num_blocks, state_dim)`` float32.
        embodiment_id: ``(B,)`` int32 embodiment IDs.
        num_inference_steps: Number of denoising steps.
        cfg_scale: Classifier-free guidance scale.
        seed: Random seed for noise initialization.
        use_jit: Whether to JIT-compile the generate call.

    Returns:
        Tuple of ``(action_pred, video_pred)`` JAX arrays.
    """
    key = jax.random.key(seed)

    if use_jit:

        @jax.jit
        def _generate(
            video: jax.Array,
            token_ids: jax.Array,
            attention_mask: jax.Array,
            state: jax.Array,
            embodiment_id: jax.Array,
            key: jax.Array,
        ):
            return model.generate(
                video=video,
                token_ids=token_ids,
                state=state,
                embodiment_id=embodiment_id,
                attention_mask=attention_mask,
                num_inference_steps=num_inference_steps,
                cfg_scale=cfg_scale,
                key=key,
            )

        logger.info("JIT-compiling model.generate (first call will be slow)...")
        t0 = time.perf_counter()
        output = _generate(video, token_ids, attention_mask, state, embodiment_id, key)
        # Block until computation finishes (for accurate timing)
        jax.block_until_ready(output)
        t1 = time.perf_counter()
        logger.info("First call (includes JIT compilation): %.2fs", t1 - t0)

        # Run again to get steady-state timing
        t0 = time.perf_counter()
        output = _generate(video, token_ids, attention_mask, state, embodiment_id, key)
        jax.block_until_ready(output)
        t1 = time.perf_counter()
        logger.info("Second call (cached): %.2fs", t1 - t0)
    else:
        logger.info("Running model.generate (no JIT)...")
        t0 = time.perf_counter()
        output = model.generate(
            video=video,
            token_ids=token_ids,
            state=state,
            embodiment_id=embodiment_id,
            attention_mask=attention_mask,
            num_inference_steps=num_inference_steps,
            cfg_scale=cfg_scale,
            key=key,
        )
        jax.block_until_ready(output)
        t1 = time.perf_counter()
        logger.info("Inference time: %.2fs", t1 - t0)

    return output.action_pred, output.video_pred


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------


def decode_video_latents(model: DreamZero, video_latents: jax.Array) -> np.ndarray:
    """Decode video latents through the VAE decoder to pixel-space.

    Args:
        model: DreamZero model (for access to the VAE).
        video_latents: ``(B, T', H', W', z_dim)`` latent tensor from generation.

    Returns:
        ``(B, T, H, W, 3)`` uint8 video frames.
    """
    logger.info("Decoding video latents through VAE (shape: %s)...", video_latents.shape)
    t0 = time.perf_counter()
    decoded = model.vae.decode(video_latents)
    jax.block_until_ready(decoded)
    t1 = time.perf_counter()
    logger.info("VAE decode time: %.2fs", t1 - t0)

    # Convert to numpy and denormalize: [-1, 1] -> [0, 255] uint8
    decoded_np = np.asarray(decoded)
    decoded_uint8 = denormalize_video(decoded_np)
    return decoded_uint8


def save_results(
    output_dir: pathlib.Path,
    action_pred: np.ndarray,
    video_frames: np.ndarray | None = None,
) -> None:
    """Save inference results to disk.

    Args:
        output_dir: Directory to write outputs into.
        action_pred: ``(B, total_actions, action_dim)`` predicted actions.
        video_frames: Optional ``(B, T, H, W, 3)`` uint8 decoded video.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save actions as numpy file
    actions_path = output_dir / "actions.npy"
    np.save(actions_path, action_pred)
    logger.info("Saved predicted actions to %s (shape: %s)", actions_path, action_pred.shape)

    # Also save as human-readable CSV for the first batch element
    actions_csv_path = output_dir / "actions.csv"
    np.savetxt(
        actions_csv_path,
        action_pred[0],
        delimiter=",",
        header="Actions for batch element 0 (each row = one timestep)",
    )
    logger.info("Saved actions CSV to %s", actions_csv_path)

    # Save video frames if provided
    if video_frames is not None:
        for b in range(video_frames.shape[0]):
            frames_dir = output_dir / f"video_batch_{b:02d}"
            saved = save_video_frames(video_frames[b], frames_dir, prefix="pred")
            logger.info(
                "Saved %d predicted video frames to %s",
                len(saved),
                frames_dir,
            )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="DreamZero inference: generate actions and video from observation + prompt.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument(
        "--checkpoint",
        type=pathlib.Path,
        required=True,
        help="Path to Flax/orbax checkpoint directory.",
    )
    parser.add_argument(
        "--input-video",
        type=pathlib.Path,
        required=True,
        help="Path to input video (mp4 file or directory of frame images).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text instruction for the task.",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("outputs"),
        help="Directory to save output files.",
    )

    # Inference params
    parser.add_argument(
        "--num-steps",
        type=int,
        default=16,
        help="Number of denoising inference steps.",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=5.0,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for noise initialization.",
    )
    parser.add_argument(
        "--embodiment-id",
        type=int,
        default=0,
        help="Embodiment ID for the robot.",
    )

    # Model / precision
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "bfloat16"],
        default="float32",
        help="Compute dtype for model parameters.",
    )
    parser.add_argument(
        "--mesh-shape",
        type=str,
        default=None,
        help=(
            "Device mesh shape for distributed inference as 'dp,mp' "
            "(e.g., '1,4'). If not set, uses a single device."
        ),
    )

    # Optional overrides
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="google/t5-v1_1-xxl",
        help="HuggingFace tokenizer model ID.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=512,
        help="Maximum text sequence length for tokenization.",
    )
    parser.add_argument(
        "--no-jit",
        action="store_true",
        help="Disable JIT compilation (useful for debugging).",
    )
    parser.add_argument(
        "--skip-video-decode",
        action="store_true",
        help="Skip VAE decoding of predicted video latents (saves time/memory).",
    )
    parser.add_argument(
        "--max-input-frames",
        type=int,
        default=None,
        help="Limit the number of input video frames loaded.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG-level) logging.",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> None:
    """Main entry point for the DreamZero inference pipeline."""
    args = parse_args(argv)

    # --- Logging setup ---
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("DreamZero Inference Pipeline")
    logger.info("=" * 60)
    logger.info("JAX devices: %s", jax.devices())
    logger.info("JAX default backend: %s", jax.default_backend())
    logger.info("Checkpoint: %s", args.checkpoint)
    logger.info("Input video: %s", args.input_video)
    logger.info("Prompt: %s", args.prompt)
    logger.info("Output dir: %s", args.output_dir)
    logger.info("Num inference steps: %d", args.num_steps)
    logger.info("CFG scale: %.1f", args.cfg_scale)
    logger.info("Seed: %d", args.seed)
    logger.info("Embodiment ID: %d", args.embodiment_id)
    logger.info("Dtype: %s", args.dtype)
    logger.info("Mesh shape: %s", args.mesh_shape or "single device")
    logger.info("=" * 60)

    # --- Parse dtype ---
    dtype = jnp.bfloat16 if args.dtype == "bfloat16" else jnp.float32

    # --- Create mesh if multi-device ---
    mesh = None
    if args.mesh_shape is not None:
        dp, mp = (int(x) for x in args.mesh_shape.split(","))
        mesh = create_mesh((dp, mp))
        logger.info("Created device mesh: %s", mesh)

    # --- Initialize model ---
    logger.info("Initializing DreamZero model...")
    t0 = time.perf_counter()
    config = DreamZeroConfig()
    model = DreamZero(config, rngs=nnx.Rngs(args.seed))
    t1 = time.perf_counter()
    logger.info("Model initialized in %.2fs", t1 - t0)

    # --- Load checkpoint ---
    logger.info("Loading checkpoint from %s ...", args.checkpoint)
    t0 = time.perf_counter()
    model = load_checkpoint(args.checkpoint, model)
    t1 = time.perf_counter()
    logger.info("Checkpoint loaded in %.2fs", t1 - t0)

    # --- Cast parameters to target dtype if bfloat16 ---
    if dtype == jnp.bfloat16:
        logger.info("Casting model parameters to bfloat16...")

        def cast_to_bf16(x):
            if isinstance(x, jax.Array) and jnp.issubdtype(x.dtype, jnp.floating):
                return x.astype(jnp.bfloat16)
            return x

        graphdef, state = nnx.split(model)
        state = jax.tree.map(cast_to_bf16, state)
        model = nnx.merge(graphdef, state)

    # --- Load and preprocess input video ---
    logger.info("Loading input video from %s ...", args.input_video)
    video_np = load_video(args.input_video, max_frames=args.max_input_frames)
    logger.info("Loaded video: shape=%s, dtype=%s", video_np.shape, video_np.dtype)

    # Normalize to [-1, 1]
    video_normalized = normalize_video(video_np)
    # Add batch dimension: (T, H, W, 3) -> (1, T, H, W, 3)
    video_batch = video_normalized[np.newaxis, ...]
    logger.info("Preprocessed video: shape=%s", video_batch.shape)

    # --- Tokenize text prompt ---
    logger.info("Tokenizing prompt: '%s'", args.prompt)
    token_ids, attention_mask = tokenize_prompt(
        args.prompt,
        tokenizer_name=args.tokenizer,
        max_length=args.max_seq_len,
    )
    logger.info("Token IDs shape: %s, Attention mask shape: %s", token_ids.shape, attention_mask.shape)

    # --- Prepare dummy state and embodiment ID ---
    # State: (B, num_blocks, state_dim) - zeros as placeholder
    # Number of blocks is determined by temporal latent frames / num_frames_per_block
    # For a rough estimate, we compute from video shape:
    #   T_latent = T_input / 4 (VAE temporal compression)
    #   num_blocks = T_latent / num_frames_per_block
    T_input = video_batch.shape[1]
    # VAE does 4x temporal compression (with 2 temporal downsamples each 2x)
    T_latent = max(1, T_input // 4)
    num_blocks = max(1, T_latent // config.num_frames_per_block)
    state = np.zeros((1, num_blocks, config.state_dim), dtype=np.float32)
    logger.info("State shape: %s (num_blocks=%d)", state.shape, num_blocks)

    embodiment_id = np.array([args.embodiment_id], dtype=np.int32)

    # --- Convert inputs to JAX arrays ---
    video_jax = jnp.array(video_batch, dtype=dtype)
    token_ids_jax = jnp.array(token_ids, dtype=jnp.int32)
    attention_mask_jax = jnp.array(attention_mask, dtype=jnp.float32)
    state_jax = jnp.array(state, dtype=dtype)
    embodiment_id_jax = jnp.array(embodiment_id, dtype=jnp.int32)

    logger.info("Input shapes:")
    logger.info("  video:          %s (dtype=%s)", video_jax.shape, video_jax.dtype)
    logger.info("  token_ids:      %s (dtype=%s)", token_ids_jax.shape, token_ids_jax.dtype)
    logger.info("  attention_mask: %s (dtype=%s)", attention_mask_jax.shape, attention_mask_jax.dtype)
    logger.info("  state:          %s (dtype=%s)", state_jax.shape, state_jax.dtype)
    logger.info("  embodiment_id:  %s (dtype=%s)", embodiment_id_jax.shape, embodiment_id_jax.dtype)

    # --- Run inference ---
    logger.info("Starting inference...")
    action_pred, video_pred = run_inference(
        model=model,
        video=video_jax,
        token_ids=token_ids_jax,
        attention_mask=attention_mask_jax,
        state=state_jax,
        embodiment_id=embodiment_id_jax,
        num_inference_steps=args.num_steps,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
        use_jit=not args.no_jit,
    )

    logger.info("Inference complete.")
    logger.info("  action_pred shape: %s", action_pred.shape)
    logger.info("  video_pred shape:  %s", video_pred.shape)

    # --- Post-process actions ---
    action_pred_np = np.asarray(action_pred)

    # --- Decode video latents (optional) ---
    video_frames_np = None
    if not args.skip_video_decode:
        video_frames_np = decode_video_latents(model, video_pred)
        logger.info("Decoded video shape: %s", video_frames_np.shape)
    else:
        logger.info("Skipping video decode (--skip-video-decode).")
        # Save raw latents instead
        latents_path = args.output_dir / "video_latents.npy"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        np.save(latents_path, np.asarray(video_pred))
        logger.info("Saved raw video latents to %s", latents_path)

    # --- Save results ---
    save_results(
        output_dir=args.output_dir,
        action_pred=action_pred_np,
        video_frames=video_frames_np,
    )

    logger.info("All outputs saved to %s", args.output_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
