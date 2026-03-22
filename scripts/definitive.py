#!/usr/bin/env python3
"""Definitive JAX/TPU inference with deterministic inputs and .npz output.

Runs the DreamZero-JAX model with the EXACT same deterministic inputs used
by ``gpu_reference.py``, saving all outputs as .npz for cross-framework
comparison via ``compare_outputs.py``.

This is the TPU-side counterpart of ``gpu_reference.py``.

Usage
-----
On a TPU VM with the DROID checkpoint::

    python scripts/definitive.py \
        --checkpoint-dir /path/to/DreamZero-DROID \
        --output tpu_output.npz

With custom layer count (e.g. 24L to fit on v5e-8)::

    python scripts/definitive.py \
        --checkpoint-dir /path/to/DreamZero-DROID \
        --num-layers 24 \
        --output tpu_output.npz

Then compare::

    python scripts/compare_outputs.py gpu_reference.npz tpu_output.npz
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np

SEED = 42
VIDEO_SHAPE = (1, 33, 320, 176, 3)
TOKEN_SHAPE = (1, 512)
STATE_SHAPE = (1, 9, 64)

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))


def make_deterministic_inputs():
    """Create the same fixed inputs as gpu_reference.py (numpy-based RNG)."""
    gen = np.random.RandomState(SEED)
    video = gen.randn(*VIDEO_SHAPE).astype(np.float32) * 0.1
    tokens = np.ones(TOKEN_SHAPE, dtype=np.int32)
    state = gen.randn(*STATE_SHAPE).astype(np.float32) * 0.01
    return video, tokens, state


def load_model(args: argparse.Namespace):
    """Load the JAX model with checkpoint weights."""
    import jax
    import jax.numpy as jnp
    from flax import nnx

    from dreamzero_jax.models.dreamzero import DreamZero, DreamZeroConfig
    from dreamzero_jax.utils.checkpoint import apply_to_model, convert_checkpoint
    from dreamzero_jax.utils.hf_download import load_checkpoint_auto

    print(f"JAX devices: {jax.devices()}")

    print(f"Loading PyTorch checkpoint from {args.checkpoint_dir}...")
    t0 = time.time()
    pt_state = load_checkpoint_auto(args.checkpoint_dir)
    print(f"  {len(pt_state)} params ({time.time() - t0:.1f}s)")

    cfg = DreamZeroConfig(
        dim=5120,
        ffn_dim=13824,
        num_heads=40,
        num_layers=args.num_layers,
        freq_dim=256,
        text_dim=4096,
        patch_size=(1, 2, 2),
        in_channels=16,
        out_channels=16,
        has_image_input=True,
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
    )

    cpu = jax.devices("cpu")[0]
    print(f"Creating {args.num_layers}L model on CPU...")
    with jax.default_device(cpu):
        model = DreamZero(cfg, rngs=nnx.Rngs(0))

        print("Converting weights...")
        t0 = time.time()
        converted = convert_checkpoint(pt_state, cfg)

        import ml_dtypes
        for k in list(converted.keys()):
            if converted[k].dtype in (np.float32, np.float64):
                converted[k] = converted[k].astype(ml_dtypes.bfloat16)

        applied, missing, extra = apply_to_model(model, converted)
        print(f"  {applied} applied, {len(missing)} missing, "
              f"{len(extra)} extra ({time.time() - t0:.1f}s)")

    leaves = jax.tree.leaves(nnx.state(model, nnx.Param))
    nan_count = sum(1 for leaf in leaves if np.any(np.isnan(np.asarray(leaf))))
    print(f"  NaN params: {nan_count}/{len(leaves)}")

    del pt_state, converted
    gc.collect()

    return model, cfg


def shard_model(model):
    """Shard model parameters to TPU mesh."""
    import jax
    import jax.numpy as jnp

    from dreamzero_jax.utils.sharding import create_mesh, shard_params

    mesh = create_mesh()
    print("Sharding to TPU...")
    t0 = time.time()
    model = shard_params(model, mesh, param_dtype=jnp.bfloat16)
    bu = jax.devices()[0].memory_stats().get("bytes_in_use", 0)
    print(f"  HBM: {bu / 1e9:.2f} GB ({time.time() - t0:.1f}s)")
    return model, mesh


def run_inference(model, mesh, args: argparse.Namespace) -> dict[str, np.ndarray]:
    """Run deterministic inference and collect outputs."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P

    video_np, tokens_np, state_np = make_deterministic_inputs()

    rep = NamedSharding(mesh, P())
    key = jax.device_put(jax.random.PRNGKey(SEED), rep)
    video = jax.device_put(jnp.array(video_np, dtype=jnp.bfloat16), rep)
    token_ids = jax.device_put(jnp.array(tokens_np, dtype=jnp.int32), rep)
    mask = jax.device_put(jnp.ones(TOKEN_SHAPE, dtype=jnp.int32), rep)
    state = jax.device_put(jnp.array(state_np, dtype=jnp.float32), rep)
    embodiment = jax.device_put(jnp.zeros((1,), dtype=jnp.int32), rep)

    print(f"\nRunning generate_scan ({args.num_layers}L, real weights)...")
    t0 = time.time()
    out = model.generate_scan(
        video, token_ids, state, embodiment,
        attention_mask=mask, key=key,
    )
    jax.block_until_ready(out)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    action_pred = np.asarray(out.action_pred, dtype=np.float32)
    video_pred = np.asarray(out.video_pred, dtype=np.float32)

    print(f"  action_pred: shape={action_pred.shape} "
          f"mean={action_pred.mean():.6f} std={action_pred.std():.6f}")
    print(f"  video_pred:  shape={video_pred.shape} "
          f"mean={video_pred.mean():.6f} std={video_pred.std():.6f}")

    has_nan = np.any(np.isnan(action_pred)) or np.any(np.isnan(video_pred))
    print(f"  NaN in output: {has_nan}")

    results = {
        "action_pred": action_pred,
        "video_pred": video_pred,
        "input_video": video_np,
        "input_tokens": tokens_np,
        "input_state": state_np,
    }

    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Definitive JAX/TPU inference with deterministic inputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--checkpoint-dir", type=Path, required=True,
        help="Path to DreamZero-DROID checkpoint directory.",
    )
    p.add_argument(
        "--num-layers", type=int, default=24,
        help="Number of DiT layers (24 for v5e-8, 40 for v5e-16).",
    )
    p.add_argument(
        "--output", type=str, default="tpu_output.npz",
        help="Output .npz file path.",
    )
    p.add_argument(
        "--skip-shard", action="store_true",
        help="Skip sharding (run on CPU only, for testing).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    print(f"\n{'=' * 60}")
    print(f"  Definitive JAX/TPU Inference")
    print(f"  checkpoint: {args.checkpoint_dir}")
    print(f"  layers:     {args.num_layers}")
    print(f"  output:     {args.output}")
    print(f"  seed:       {SEED}")
    print(f"{'=' * 60}\n")

    model, cfg = load_model(args)

    if args.skip_shard:
        import jax
        from dreamzero_jax.utils.sharding import create_mesh
        mesh = create_mesh()
    else:
        model, mesh = shard_model(model)

    results = run_inference(model, mesh, args)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(output_path), **results)

    print(f"\nSaved {len(results)} arrays to {output_path}:")
    for key, arr in sorted(results.items()):
        print(f"  {key:<20} shape={str(arr.shape):<25} "
              f"mean={arr.mean():.6f}  std={arr.std():.6f}")

    has_nan = (
        np.any(np.isnan(results["action_pred"]))
        or np.any(np.isnan(results["video_pred"]))
    )
    status = "PASSED" if not has_nan else "FAILED"
    print(f"\n=== DEFINITIVE INFERENCE {status} ===")


if __name__ == "__main__":
    main()
