#!/usr/bin/env python3
"""DreamZero-JAX inference — one-click demo entry point.

Runs the full 14B DreamZero World Action Model on a text prompt and
optional video input, producing predicted actions and video latents.

Auto-detects hardware (TPU / GPU / CPU), downloads weights from
HuggingFace if needed, and selects the optimal inference strategy:
  - TPU: staged inference (encoders and DiT never coexist in HBM)
  - GPU/CPU: direct inference (all weights loaded at once)

Examples
--------
Minimal (downloads weights, uses zeros as video input)::

    uv run python scripts/inference.py \\
        --checkpoint GEAR-Dreams/DreamZero-DROID

With a real video and custom prompt::

    uv run python scripts/inference.py \\
        --checkpoint /data/DreamZero-DROID \\
        --video /data/observation.mp4 \\
        --prompt "pick up the red block" \\
        --output output/predictions.npz

Small model on CPU for quick testing::

    uv run python scripts/inference.py \\
        --checkpoint /data/DreamZero-DROID \\
        --num-layers 8 --device cpu --dtype f32
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

VIDEO_SHAPE = (1, 33, 320, 176, 3)
TOKEN_SHAPE = (1, 512)
STATE_SHAPE_BLOCKS = 9
STATE_DIM = 64
SEED = 42

DROID_CONFIG = dict(
    dim=5120,
    ffn_dim=13824,
    num_heads=40,
    freq_dim=256,
    text_dim=4096,
    patch_size=(1, 2, 2),
    in_channels=16,
    out_channels=16,
    has_image_input=True,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DreamZero-JAX inference: generate actions and video from a text prompt.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--checkpoint", "--checkpoint-dir",
        type=str, default=None, dest="checkpoint",
        help="Path to DROID checkpoint directory, or HuggingFace repo ID. "
             "If omitted, uses random weights (quick test mode).",
    )
    p.add_argument(
        "--prompt", type=str, default="pick up the red block",
        help="Text instruction for the robot.",
    )
    p.add_argument(
        "--video", type=str, default=None,
        help="Input video path (mp4/dir of frames). Uses zeros if omitted.",
    )
    p.add_argument(
        "--output", type=str, default="output/predictions.npz",
        help="Save predictions (.npz) to this path.",
    )
    p.add_argument(
        "--num-layers", type=int, default=None,
        help="DiT layers (default: 8 for v5e-4, 40 for v5e-8+).",
    )
    p.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "tpu", "gpu", "cpu"],
        help="Device to run on. 'auto' picks TPU > GPU > CPU.",
    )
    p.add_argument(
        "--dtype", type=str, default="bf16",
        choices=["bf16", "f32"],
        help="Compute dtype. bf16 is native on TPU and halves memory.",
    )
    p.add_argument(
        "--num-steps", type=int, default=16,
        help="Number of denoising steps.",
    )
    p.add_argument(
        "--cfg-scale", type=float, default=5.0,
        help="Classifier-free guidance scale.",
    )
    p.add_argument(
        "--no-cfg", action="store_true",
        help="Disable CFG (halves activation memory, needed for 40L on v5e-8).",
    )
    p.add_argument(
        "--seed", type=int, default=SEED,
        help="Random seed for reproducibility.",
    )
    p.add_argument(
        "--quantize-int8", action="store_true",
        help="Quantize DiT to INT8 before inference (saves HBM).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------


def detect_device(requested: str) -> tuple[str, int, str]:
    """Return (platform, num_devices, description)."""
    import jax

    if requested == "auto":
        backend = jax.default_backend()
    else:
        backend = requested

    devices = jax.devices(backend) if backend != "auto" else jax.devices()
    n = len(devices)

    if backend == "tpu":
        try:
            kind = devices[0].device_kind
        except Exception:
            kind = "TPU"
        desc = f"{kind} ({n} chips)"
    elif backend == "gpu":
        try:
            kind = devices[0].device_kind
        except Exception:
            kind = "GPU"
        desc = f"{kind} ({n} devices)"
    else:
        desc = "CPU"

    return backend, n, desc


def pick_num_layers(num_devices: int, backend: str, explicit: int | None) -> int:
    if explicit is not None:
        return explicit
    if backend == "cpu":
        return 2
    if num_devices <= 4:
        return 8
    return 40


# ---------------------------------------------------------------------------
# Checkpoint resolution
# ---------------------------------------------------------------------------


def resolve_checkpoint(checkpoint: str) -> Path:
    """If checkpoint looks like an HF repo ID, download it; otherwise treat as local path."""
    local = Path(checkpoint)
    if local.exists():
        return local

    if "/" in checkpoint and not checkpoint.startswith("/"):
        print(f"  Downloading from HuggingFace: {checkpoint}")
        from dreamzero_jax.utils.hf_download import download_from_hf
        return download_from_hf(checkpoint).parent
    raise FileNotFoundError(
        f"Checkpoint not found: {checkpoint}\n"
        "Provide a local path or a HuggingFace repo ID (e.g. GEAR-Dreams/DreamZero-DROID)."
    )


# ---------------------------------------------------------------------------
# Video loading
# ---------------------------------------------------------------------------


def load_video_input(video_path: str | None) -> np.ndarray:
    """Load video from file or create zeros for demo mode."""
    if video_path is None:
        return np.zeros(VIDEO_SHAPE, dtype=np.float32)

    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if path.is_dir():
        return _load_frames_dir(path)
    return _load_video_file(path)


def _load_frames_dir(frames_dir: Path) -> np.ndarray:
    from PIL import Image

    exts = {".png", ".jpg", ".jpeg"}
    paths = sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in exts)
    if not paths:
        raise FileNotFoundError(f"No image files in {frames_dir}")

    frames = [np.array(Image.open(p).convert("RGB")) for p in paths]
    video = np.stack(frames, axis=0)
    video = (video.astype(np.float32) / 127.5) - 1.0
    return video[np.newaxis]


def _load_video_file(video_path: Path) -> np.ndarray:
    try:
        import imageio.v3 as iio
        frames = np.asarray(iio.imread(str(video_path), plugin="pyav"))
    except ImportError:
        try:
            import cv2
            cap = cv2.VideoCapture(str(video_path))
            raw = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                raw.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            if not raw:
                raise RuntimeError(f"Could not read frames from {video_path}")
            frames = np.stack(raw, axis=0)
        except ImportError:
            raise ImportError(
                "Loading mp4 requires 'imageio[pyav]' or 'opencv-python'. "
                "Install one, or pass a directory of frame images."
            )

    video = (frames.astype(np.float32) / 127.5) - 1.0
    return video[np.newaxis]


# ---------------------------------------------------------------------------
# Inference strategies
# ---------------------------------------------------------------------------


def run_staged(config, video, token_ids, state, embodiment_id, mask, args, mesh):
    """Staged inference: encoders and DiT never coexist in HBM. Best for TPU."""
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P

    from dreamzero_jax.models.staged_inference import generate_staged

    rep = NamedSharding(mesh, P())
    import jax.numpy as jnp

    video_j = jax.device_put(jnp.array(video, dtype=config.dtype), rep)
    tokens_j = jax.device_put(jnp.array(token_ids, dtype=jnp.int32), rep)
    mask_j = jax.device_put(jnp.array(mask, dtype=jnp.int32), rep)
    state_j = jax.device_put(jnp.array(state, dtype=jnp.float32), rep)
    emb_j = jax.device_put(jnp.zeros((1,), dtype=jnp.int32), rep)
    key = jax.device_put(jax.random.PRNGKey(args.seed), rep)

    return generate_staged(
        config,
        video_j, tokens_j, state_j, emb_j,
        attention_mask=mask_j,
        num_inference_steps=args.num_steps,
        cfg_scale=args.cfg_scale,
        use_cfg=not args.no_cfg,
        key=key,
        mesh=mesh,
        checkpoint_dir=str(args.checkpoint_path),
        quantize_int8=args.quantize_int8,
    )


def run_direct(config, video, token_ids, state, embodiment_id, mask, args):
    """Direct inference: load full model, run generate_scan. For GPU/CPU."""
    import jax
    import jax.numpy as jnp
    from flax import nnx

    from dreamzero_jax.models.dreamzero import DreamZero
    from dreamzero_jax.utils.checkpoint import apply_to_model, convert_checkpoint
    from dreamzero_jax.utils.hf_download import load_checkpoint_auto

    cpu = jax.devices("cpu")[0]

    print("  Creating model on CPU...", end=" ", flush=True)
    t0 = time.time()
    with jax.default_device(cpu):
        model = DreamZero(config, rngs=nnx.Rngs(0))
    print(f"done ({time.time() - t0:.1f}s)")

    if args.checkpoint_path is None:
        print("  Using random weights (no checkpoint)")
    else:
        print("  Loading checkpoint...", end=" ", flush=True)
        t0 = time.time()
        pt_state = load_checkpoint_auto(args.checkpoint_path)
        print(f"{len(pt_state)} params ({time.time() - t0:.1f}s)")

        print("  Converting weights...", end=" ", flush=True)
        t0 = time.time()
        with jax.default_device(cpu):
            converted = convert_checkpoint(pt_state, config)
            _cast_to_dtype(converted, config.param_dtype)
            applied, missing, extra = apply_to_model(model, converted)
        print(f"{applied} applied, {len(missing)} missing ({time.time() - t0:.1f}s)")
        del pt_state, converted
        gc.collect()

    video_j = jnp.array(video, dtype=config.dtype)
    tokens_j = jnp.array(token_ids, dtype=jnp.int32)
    mask_j = jnp.array(mask, dtype=jnp.int32)
    state_j = jnp.array(state, dtype=jnp.float32)
    emb_j = jnp.zeros((1,), dtype=jnp.int32)
    key = jax.random.PRNGKey(args.seed)

    return model.generate_scan(
        video_j, tokens_j, state_j, emb_j,
        attention_mask=mask_j,
        num_inference_steps=args.num_steps,
        cfg_scale=args.cfg_scale,
        use_cfg=not args.no_cfg,
        key=key,
    )


def _cast_to_dtype(converted: dict, dtype) -> None:
    """Cast float32/float64 values to target dtype in-place."""
    import ml_dtypes

    target = np.dtype(ml_dtypes.bfloat16) if "bfloat16" in str(dtype) else np.float32
    for k in list(converted.keys()):
        if converted[k].dtype in (np.float32, np.float64):
            converted[k] = converted[k].astype(target)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_config(args, dtype):
    """Build DreamZeroConfig with DROID hyperparameters."""
    from dreamzero_jax.models.dreamzero import DreamZeroConfig

    return DreamZeroConfig(
        **DROID_CONFIG,
        num_layers=args.num_layers,
        dtype=dtype,
        param_dtype=dtype,
    )


def print_banner(args, backend, desc, num_layers):
    print()
    print("DreamZero-JAX Inference")
    print("=" * 40)
    print(f"  Checkpoint:  {args.checkpoint}")
    print(f"  Device:      {desc}")
    print(f"  Layers:      {num_layers}")
    print(f"  Dtype:       {args.dtype}")
    print(f"  Prompt:      {args.prompt!r}")
    print(f"  Video:       {args.video or '(zeros — demo mode)'}")
    print(f"  Steps:       {args.num_steps}")
    print(f"  CFG:         {'off' if args.no_cfg else args.cfg_scale}")
    print(f"  Seed:        {args.seed}")
    print(f"  Output:      {args.output}")
    print("=" * 40)
    print()


def print_results(output, elapsed, output_path):
    action = np.asarray(output.action_pred, dtype=np.float32)
    video = np.asarray(output.video_pred, dtype=np.float32)

    print()
    print(f"  Inference time: {elapsed:.1f}s")
    print()
    a_shape = action.shape
    print(
        f"  Action predictions:  {a_shape} "
        f"-- {a_shape[1]} timesteps, {a_shape[2]}-dim actions"
    )
    v_shape = video.shape
    print(
        f"  Video predictions:   {v_shape} "
        f"-- {v_shape[1]} latent frames"
    )
    print()

    has_nan = np.any(np.isnan(action)) or np.any(np.isnan(video))
    if has_nan:
        print("  WARNING: NaN detected in outputs")

    print(f"  Action stats: mean={action.mean():.6f}  std={action.std():.6f}")
    print(f"  Video stats:  mean={video.mean():.6f}  std={video.std():.6f}")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out), action_pred=action, video_pred=video)
    print(f"\n  Saved to: {out}")


def main():
    args = parse_args()

    import jax

    backend, n_devices, desc = detect_device(args.device)
    args.num_layers = pick_num_layers(n_devices, backend, args.num_layers)
    print_banner(args, backend, desc, args.num_layers)

    if backend == "cpu" and args.num_layers > 2:
        print(f"\n  WARNING: CPU with {args.num_layers} layers will use ~{args.num_layers * 0.75:.0f} GB RAM.")
        print("  Consider: --num-layers 2 (quick test) or --device tpu (full model)")
        print("  Continuing anyway...\n")

    if args.checkpoint is not None:
        print("Resolving checkpoint...")
        args.checkpoint_path = resolve_checkpoint(args.checkpoint)
        print(f"  Using: {args.checkpoint_path}")
    else:
        args.checkpoint_path = None
        print("  No checkpoint — using random weights (quick test mode)")

    import jax.numpy as jnp
    dtype = jnp.bfloat16 if args.dtype == "bf16" else jnp.float32
    config = build_config(args, dtype)

    print("Loading video...")
    video = load_video_input(args.video)
    print(f"  Video shape: {video.shape}")

    token_ids = np.ones(TOKEN_SHAPE, dtype=np.int32)
    mask = np.ones(TOKEN_SHAPE, dtype=np.int32)
    state = np.zeros((1, STATE_SHAPE_BLOCKS, STATE_DIM), dtype=np.float32)
    embodiment_id = np.zeros((1,), dtype=np.int32)

    use_staged = backend == "tpu"

    print(f"\nRunning inference ({'staged' if use_staged else 'direct'})...")
    t0 = time.time()

    if use_staged:
        from dreamzero_jax.utils.sharding import create_mesh
        mesh = create_mesh()
        output = run_staged(
            config, video, token_ids, state, embodiment_id, mask, args, mesh,
        )
    else:
        output = run_direct(
            config, video, token_ids, state, embodiment_id, mask, args,
        )

    jax.block_until_ready(output)
    elapsed = time.time() - t0

    print_results(output, elapsed, args.output)


if __name__ == "__main__":
    main()
