#!/usr/bin/env python3
"""
DreamZero 14B Component Benchmarks — TPU
=========================================
Measures latency and peak HBM for each component independently:

  1. Single WanDiTBlock  (d=5120, heads=40, ffn=13824)
  2. Full WanDiT         (40 layers, full 14B config)
  3. VAE encode          (33 frames @ 320x176)
  4. VAE decode          (latents -> 33 frames)
  5. Full inference      (DreamZero.generate, 16 steps, CFG)

Protocol: 3 warmup + 10 timed iterations, jax.block_until_ready().

Usage (TPU):
  python scripts/benchmark_components.py --component all
  python scripts/benchmark_components.py --component single_block --num-layers 8

Usage (CPU, smoke-test only):
  JAX_PLATFORM_NAME=cpu python scripts/benchmark_components.py --component single_block --dtype f32
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


# ---------------------------------------------------------------------------
# Ensure the project is importable
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import dreamzero_jax  # noqa: F401 — triggers nnx.List compat patch

from dreamzero_jax.models.dit import WanDiT, WanDiTBlock
from dreamzero_jax.models.vae import WanVideoVAE
from dreamzero_jax.models.action_head import CausalWanDiT
from dreamzero_jax.models.dreamzero import DreamZero, DreamZeroConfig
from dreamzero_jax.utils.sharding import create_mesh, shard_params, REPLICATED

from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


# ---------------------------------------------------------------------------
# 14B configuration constants
# ---------------------------------------------------------------------------

CONFIG_14B = dict(
    dim=5120,
    ffn_dim=13824,
    num_heads=40,
    num_layers=40,
    freq_dim=256,
    text_dim=4096,
    patch_size=(1, 2, 2),
    in_channels=16,
    out_channels=16,
    has_image_input=True,
)

# Video shape: 33 frames @ 320x176 pixel space
# VAE compresses: 8x spatial, ~4x temporal -> (1, 9, 40, 22, 16) latent
# After patching with (1,2,2): grid is (9, 20, 11) -> 1980 tokens
VIDEO_FRAMES = 33
VIDEO_H = 320
VIDEO_W = 176
LATENT_T = 9   # ceil(33/4) with causal temporal compression
LATENT_H = 40  # 320 / 8
LATENT_W = 22  # 176 / 8 = 22
LATENT_C = 16
# After patch_size=(1,2,2): grid = (9, 20, 11), seq_len = 1980
GRID_F = 9
GRID_H = 20  # 40 / 2
GRID_W = 11  # 22 / 2
SEQ_LEN = GRID_F * GRID_H * GRID_W  # 1980

# Text context length
TEXT_SEQ_LEN = 512


# ---------------------------------------------------------------------------
# Benchmark result container
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    name: str
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    hbm_gb: float | None = None
    note: str = ""


# ---------------------------------------------------------------------------
# Timing + HBM utilities
# ---------------------------------------------------------------------------


def get_hbm_usage_gb() -> float | None:
    """Return peak HBM usage in GB, or None if unavailable."""
    try:
        dev = jax.local_devices()[0]
        stats = dev.memory_stats()
        if stats and "peak_bytes_in_use" in stats:
            return stats["peak_bytes_in_use"] / (1024 ** 3)
    except Exception:
        pass
    return None


def reset_hbm_peak() -> None:
    """Try to reset peak memory tracking (not always supported)."""
    try:
        dev = jax.local_devices()[0]
        # JAX >= 0.4.x supports this on some backends
        if hasattr(dev, "clear_memory_stats"):
            dev.clear_memory_stats()
    except Exception:
        pass


def benchmark_fn(
    fn,
    *args,
    warmup: int = 3,
    iters: int = 10,
    name: str = "unnamed",
) -> BenchmarkResult:
    """Benchmark a callable with warmup and timing.

    Returns a BenchmarkResult with mean/std/min/max ms and peak HBM.
    """
    # Warmup
    for _ in range(warmup):
        result = fn(*args)
        jax.block_until_ready(result)

    # Reset peak before timed runs
    reset_hbm_peak()

    timings = []
    for _ in range(iters):
        t0 = time.perf_counter()
        result = fn(*args)
        jax.block_until_ready(result)
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000.0)

    hbm_gb = get_hbm_usage_gb()
    arr = np.array(timings)

    return BenchmarkResult(
        name=name,
        mean_ms=float(np.mean(arr)),
        std_ms=float(np.std(arr)),
        min_ms=float(np.min(arr)),
        max_ms=float(np.max(arr)),
        hbm_gb=hbm_gb,
    )


# ---------------------------------------------------------------------------
# Component benchmarks
# ---------------------------------------------------------------------------


def resolve_dtype(dtype_str: str) -> jnp.dtype:
    return jnp.bfloat16 if dtype_str == "bf16" else jnp.float32


def bench_single_block(
    batch_size: int = 1,
    num_layers: int = 40,  # unused, kept for API consistency
    dtype_str: str = "bf16",
    warmup: int = 3,
    iters: int = 10,
    use_pallas: bool = False,
    mesh: Mesh | None = None,
) -> BenchmarkResult:
    """Benchmark a single WanDiTBlock at 14B config."""
    dtype = resolve_dtype(dtype_str)
    dim = CONFIG_14B["dim"]
    rngs = nnx.Rngs(params=jax.random.PRNGKey(0))

    pallas_str = " [Pallas]" if use_pallas else ""
    print(f"  Initializing single WanDiTBlock (d={dim}, heads=40, ffn=13824){pallas_str}...")
    block = WanDiTBlock(
        dim=dim,
        num_heads=CONFIG_14B["num_heads"],
        ffn_dim=CONFIG_14B["ffn_dim"],
        has_image_input=True,
        qk_norm=True,
        use_pallas=use_pallas,
        dtype=dtype,
        param_dtype=dtype,
        rngs=rngs,
    )

    # Shard model weights across devices if mesh is provided
    if mesh is not None:
        print("  Sharding model weights across devices...")
        block = shard_params(block, mesh)

    # Create dummy inputs
    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    x = jax.random.normal(k1, (batch_size, SEQ_LEN, dim), dtype=dtype)
    e = jax.random.normal(k2, (batch_size, 6, dim), dtype=dtype)
    # Context: 257 image tokens + TEXT_SEQ_LEN text tokens
    ctx_len = 257 + TEXT_SEQ_LEN
    context = jax.random.normal(k3, (batch_size, ctx_len, dim), dtype=dtype)
    # RoPE freqs: complex64, shape (SEQ_LEN, head_dim//2)
    head_dim = dim // CONFIG_14B["num_heads"]
    freqs_cis = jnp.ones((SEQ_LEN, head_dim // 2), dtype=jnp.complex64)

    # Replicate inputs across all devices when sharding
    if mesh is not None:
        replicated = NamedSharding(mesh, P())
        x = jax.device_put(x, replicated)
        e = jax.device_put(e, replicated)
        context = jax.device_put(context, replicated)
        freqs_cis = jax.device_put(freqs_cis, replicated)

    @nnx.jit
    def run(model, x, e, context, freqs_cis):
        return model(x, e, context, freqs_cis)

    print(f"  Running benchmark ({warmup} warmup + {iters} timed)...")
    return benchmark_fn(
        run, block, x, e, context, freqs_cis,
        warmup=warmup, iters=iters,
        name=f"Single DiT block (d={dim})",
    )


def bench_full_dit(
    batch_size: int = 1,
    num_layers: int = 40,
    dtype_str: str = "bf16",
    warmup: int = 3,
    iters: int = 10,
    use_pallas: bool = False,
    use_scan: bool = False,
    mesh: Mesh | None = None,
) -> BenchmarkResult:
    """Benchmark the full WanDiT backbone."""
    dtype = resolve_dtype(dtype_str)
    dim = CONFIG_14B["dim"]
    rngs = nnx.Rngs(params=jax.random.PRNGKey(0))

    flags = []
    if use_pallas: flags.append("Pallas")
    if use_scan: flags.append("Scan")
    flag_str = f" [{', '.join(flags)}]" if flags else ""
    print(f"  Initializing WanDiT ({num_layers} layers, d={dim}){flag_str}...")
    try:
        model = WanDiT(
            dim=dim,
            in_channels=CONFIG_14B["in_channels"],
            out_channels=CONFIG_14B["out_channels"],
            ffn_dim=CONFIG_14B["ffn_dim"],
            freq_dim=CONFIG_14B["freq_dim"],
            text_dim=CONFIG_14B["text_dim"],
            num_heads=CONFIG_14B["num_heads"],
            num_layers=num_layers,
            patch_size=CONFIG_14B["patch_size"],
            has_image_input=True,
            qk_norm=True,
            use_pallas=use_pallas,
            use_scan=use_scan,
            dtype=dtype,
            param_dtype=dtype,
            rngs=rngs,
        )
    except Exception as e:
        if "out of memory" in str(e).lower() or "OOM" in str(e):
            return BenchmarkResult(
                name=f"Full DiT ({num_layers} layers)",
                mean_ms=0, std_ms=0, min_ms=0, max_ms=0,
                note=f"OOM at {num_layers} layers",
            )
        raise

    # Shard model weights across devices if mesh is provided
    if mesh is not None:
        print("  Sharding model weights across devices...")
        model = shard_params(model, mesh)

    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    # Input: latent video (B, T, H, W, C)
    x = jax.random.normal(k1, (batch_size, LATENT_T, LATENT_H, LATENT_W, LATENT_C), dtype=dtype)
    timestep = jnp.array([500.0] * batch_size, dtype=jnp.float32)
    context = jax.random.normal(k2, (batch_size, TEXT_SEQ_LEN, CONFIG_14B["text_dim"]), dtype=dtype)
    clip_emb = jax.random.normal(k3, (batch_size, 257, 1280), dtype=dtype)

    # Replicate inputs across all devices when sharding
    if mesh is not None:
        replicated = NamedSharding(mesh, P())
        x = jax.device_put(x, replicated)
        timestep = jax.device_put(timestep, replicated)
        context = jax.device_put(context, replicated)
        clip_emb = jax.device_put(clip_emb, replicated)

    @nnx.jit
    def run(model, x, timestep, context, clip_emb):
        return model(x, timestep, context, clip_emb=clip_emb)

    print(f"  Running benchmark ({warmup} warmup + {iters} timed)...")
    return benchmark_fn(
        run, model, x, timestep, context, clip_emb,
        warmup=warmup, iters=iters,
        name=f"Full DiT ({num_layers} layers)",
    )


def bench_vae_encode(
    batch_size: int = 1,
    num_layers: int = 40,  # unused
    dtype_str: str = "bf16",
    warmup: int = 3,
    iters: int = 10,
    use_pallas: bool = False,  # unused, kept for API consistency
    mesh: Mesh | None = None,
) -> BenchmarkResult:
    """Benchmark VAE encode: 33 frames @ 320x176."""
    dtype = resolve_dtype(dtype_str)
    rngs = nnx.Rngs(params=jax.random.PRNGKey(0))

    print(f"  Initializing WanVideoVAE encoder...")
    vae = WanVideoVAE(
        z_dim=LATENT_C,
        base_dim=96,
        dtype=dtype,
        param_dtype=dtype,
        rngs=rngs,
    )

    # Shard model weights across devices if mesh is provided
    if mesh is not None:
        print("  Sharding VAE weights across devices...")
        vae = shard_params(vae, mesh)

    key = jax.random.PRNGKey(42)
    video = jax.random.normal(key, (batch_size, VIDEO_FRAMES, VIDEO_H, VIDEO_W, 3), dtype=dtype)

    # Replicate inputs across all devices when sharding
    if mesh is not None:
        video = jax.device_put(video, NamedSharding(mesh, P()))

    @nnx.jit
    def run(model, video):
        return model.encode(video)

    print(f"  Running VAE encode benchmark ({warmup} warmup + {iters} timed)...")
    return benchmark_fn(
        run, vae, video,
        warmup=warmup, iters=iters,
        name=f"VAE encode ({VIDEO_FRAMES}f @ {VIDEO_H}x{VIDEO_W})",
    )


def bench_vae_decode(
    batch_size: int = 1,
    num_layers: int = 40,  # unused
    dtype_str: str = "bf16",
    warmup: int = 3,
    iters: int = 10,
    use_pallas: bool = False,  # unused, kept for API consistency
    mesh: Mesh | None = None,
) -> BenchmarkResult:
    """Benchmark VAE decode: latents -> 33 frames."""
    dtype = resolve_dtype(dtype_str)
    rngs = nnx.Rngs(params=jax.random.PRNGKey(0))

    print(f"  Initializing WanVideoVAE decoder...")
    vae = WanVideoVAE(
        z_dim=LATENT_C,
        base_dim=96,
        dtype=dtype,
        param_dtype=dtype,
        rngs=rngs,
    )

    # Shard model weights across devices if mesh is provided
    if mesh is not None:
        print("  Sharding VAE weights across devices...")
        vae = shard_params(vae, mesh)

    key = jax.random.PRNGKey(42)
    latents = jax.random.normal(
        key, (batch_size, LATENT_T, LATENT_H, LATENT_W, LATENT_C), dtype=dtype,
    )

    # Replicate inputs across all devices when sharding
    if mesh is not None:
        latents = jax.device_put(latents, NamedSharding(mesh, P()))

    @nnx.jit
    def run(model, latents):
        return model.decode(latents)

    print(f"  Running VAE decode benchmark ({warmup} warmup + {iters} timed)...")
    return benchmark_fn(
        run, vae, latents,
        warmup=warmup, iters=iters,
        name=f"VAE decode -> {VIDEO_FRAMES}f @ {VIDEO_H}x{VIDEO_W}",
    )


def bench_full_inference(
    batch_size: int = 1,
    num_layers: int = 40,
    dtype_str: str = "bf16",
    warmup: int = 3,
    iters: int = 10,
    use_pallas: bool = False,
    mesh: Mesh | None = None,
) -> BenchmarkResult:
    """Benchmark full DreamZero.generate (16 steps, CFG)."""
    dtype = resolve_dtype(dtype_str)
    rngs = nnx.Rngs(params=jax.random.PRNGKey(0))

    config = DreamZeroConfig(
        dim=CONFIG_14B["dim"],
        ffn_dim=CONFIG_14B["ffn_dim"],
        num_heads=CONFIG_14B["num_heads"],
        num_layers=num_layers,
        freq_dim=CONFIG_14B["freq_dim"],
        text_dim=CONFIG_14B["text_dim"],
        patch_size=CONFIG_14B["patch_size"],
        in_channels=CONFIG_14B["in_channels"],
        out_channels=CONFIG_14B["out_channels"],
        has_image_input=True,
        num_inference_steps=16,
        cfg_scale=5.0,
    )

    print(f"  Initializing full DreamZero model ({num_layers} layers)...")
    try:
        model = DreamZero(config, rngs=rngs)
    except Exception as e:
        if "out of memory" in str(e).lower() or "OOM" in str(e):
            return BenchmarkResult(
                name=f"Full inference ({num_layers}L, 16 steps, CFG)",
                mean_ms=0, std_ms=0, min_ms=0, max_ms=0,
                note=f"OOM at {num_layers} layers",
            )
        raise

    # Shard model weights across devices if mesh is provided
    if mesh is not None:
        print("  Sharding DreamZero model weights across devices...")
        model = shard_params(model, mesh)

    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    # Conditioning video (single frame repeated)
    video = jax.random.normal(k1, (batch_size, VIDEO_FRAMES, VIDEO_H, VIDEO_W, 3), dtype=dtype)
    token_ids = jnp.ones((batch_size, TEXT_SEQ_LEN), dtype=jnp.int32)
    attention_mask = jnp.ones((batch_size, TEXT_SEQ_LEN), dtype=jnp.int32)
    num_blocks = LATENT_T // config.num_frames_per_block
    state = jax.random.normal(k2, (batch_size, num_blocks, config.state_dim), dtype=jnp.float32)
    embodiment_id = jnp.zeros((batch_size,), dtype=jnp.int32)

    # Replicate inputs across all devices when sharding
    if mesh is not None:
        replicated = NamedSharding(mesh, P())
        video = jax.device_put(video, replicated)
        token_ids = jax.device_put(token_ids, replicated)
        attention_mask = jax.device_put(attention_mask, replicated)
        state = jax.device_put(state, replicated)
        embodiment_id = jax.device_put(embodiment_id, replicated)

    # Note: generate() traces through the Python for-loop via XLA.
    # Use nnx.jit so model weights are traced as state, not captured as constants.
    @nnx.jit
    def run(model, video, token_ids, state, embodiment_id, attention_mask, key):
        return model.generate(
            video, token_ids, state, embodiment_id,
            attention_mask=attention_mask,
            key=key,
        )

    gen_key = jax.random.PRNGKey(99)
    if mesh is not None:
        gen_key = jax.device_put(gen_key, NamedSharding(mesh, P()))
    print(f"  Running full inference benchmark ({warmup} warmup + {iters} timed)...")
    return benchmark_fn(
        run, model, video, token_ids, state, embodiment_id, attention_mask, gen_key,
        warmup=warmup, iters=iters,
        name=f"Full inference ({num_layers}L, 16 steps, CFG)",
    )


# ---------------------------------------------------------------------------
# Component registry
# ---------------------------------------------------------------------------

COMPONENTS = {
    "single_block": bench_single_block,
    "full_dit": bench_full_dit,
    "vae_encode": bench_vae_encode,
    "vae_decode": bench_vae_decode,
    "full_inference": bench_full_inference,
}


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def print_results_table(results: list[BenchmarkResult], device_info: str) -> None:
    """Print the formatted results table."""
    print()
    print("=" * 96)
    print(f"DreamZero 14B Component Benchmarks -- {device_info}")
    print("=" * 96)
    header = (
        f"{'Component':<40} | {'Mean (ms)':>10} | {'Std (ms)':>9} | "
        f"{'Min (ms)':>9} | {'Max (ms)':>9} | {'HBM (GB)':>9}"
    )
    print(header)
    print("-" * 96)

    for r in results:
        hbm_str = f"{r.hbm_gb:>8.1f}" if r.hbm_gb is not None else "     N/A"
        if r.note:
            print(f"{r.name:<40} | {'':>10} | {'':>9} | {'':>9} | {'':>9} | {hbm_str}  [{r.note}]")
        else:
            print(
                f"{r.name:<40} | {r.mean_ms:>10.2f} | {r.std_ms:>9.2f} | "
                f"{r.min_ms:>9.2f} | {r.max_ms:>9.2f} | {hbm_str}"
            )

    print("=" * 96)


def save_results_json(results: list[BenchmarkResult], path: str, device_info: str) -> None:
    """Save results to a JSON file."""
    data = {
        "device": device_info,
        "jax_version": jax.__version__,
        "results": [asdict(r) for r in results],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {path}")


# ---------------------------------------------------------------------------
# OOM fallback logic
# ---------------------------------------------------------------------------


def run_with_oom_fallback(
    bench_fn,
    component_name: str,
    batch_size: int,
    num_layers: int,
    dtype_str: str,
    warmup: int,
    iters: int,
    use_pallas: bool = False,
    use_scan: bool = False,
    mesh: Mesh | None = None,
) -> BenchmarkResult:
    """Run a benchmark, falling back to reduced layers on OOM."""
    fallback_layers = [num_layers, 16, 8]

    for nl in fallback_layers:
        try:
            print(f"\n[{component_name}] Attempting with {nl} layers...")
            result = bench_fn(
                batch_size=batch_size,
                num_layers=nl,
                dtype_str=dtype_str,
                warmup=warmup,
                iters=iters,
                use_pallas=use_pallas,
                use_scan=use_scan,
                mesh=mesh,
            )
            if result.note and "OOM" in result.note:
                print(f"  OOM at {nl} layers, trying fewer...")
                continue
            if nl != num_layers:
                result.note = f"reduced from {num_layers} to {nl} layers (OOM)"
                result.name += f" [fallback {nl}L]"
            return result
        except Exception as e:
            err = str(e).lower()
            if "out of memory" in err or "oom" in err or "resource" in err:
                print(f"  OOM at {nl} layers, trying fewer...")
                continue
            raise

    return BenchmarkResult(
        name=f"{component_name} (OOM all configs)",
        mean_ms=0, std_ms=0, min_ms=0, max_ms=0,
        note="OOM at all attempted layer counts",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="DreamZero 14B Component Benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Components:
  single_block   Single WanDiTBlock (d=5120, heads=40, ffn=13824)
  full_dit       Full WanDiT backbone (40 layers by default)
  vae_encode     VAE encode: 33 frames @ 320x176
  vae_decode     VAE decode: latents -> 33 frames
  full_inference Full DreamZero.generate (16 steps, CFG)
  all            Run all of the above
""",
    )
    parser.add_argument(
        "--component", type=str, default="all",
        choices=["single_block", "full_dit", "vae_encode", "vae_decode", "full_inference", "all"],
        help="Which component to benchmark (default: all)",
    )
    parser.add_argument(
        "--num-layers", type=int, default=40,
        help="Number of DiT layers (default: 40 for 14B; reduce if OOM)",
    )
    parser.add_argument(
        "--dtype", type=str, default="bf16", choices=["bf16", "f32"],
        help="Compute/param dtype (default: bf16)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Batch size (default: 1)",
    )
    parser.add_argument(
        "--warmup", type=int, default=3,
        help="Number of warmup iterations (default: 3)",
    )
    parser.add_argument(
        "--iters", type=int, default=10,
        help="Number of timed iterations (default: 10)",
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="Path to save JSON results",
    )
    parser.add_argument(
        "--use-pallas", action="store_true",
        help="Enable Pallas kernel optimizations (fused AdaLN)",
    )
    parser.add_argument(
        "--use-scan", action="store_true",
        help="Enable jax.lax.scan over transformer layers (reduces memory)",
    )
    parser.add_argument(
        "--shard", action="store_true",
        help="Enable tensor parallelism: shard model weights across all TPU chips",
    )
    args = parser.parse_args()

    # --- Device info ---
    devices = jax.devices()
    device_kind = devices[0].device_kind if devices else "unknown"
    device_info = f"{device_kind} ({len(devices)} device(s))"

    print(f"JAX version: {jax.__version__}")
    print(f"Device: {device_info}")
    print(f"dtype: {args.dtype}, batch_size: {args.batch_size}, num_layers: {args.num_layers}")
    print(f"Protocol: {args.warmup} warmup + {args.iters} timed iterations")

    # Estimate weight memory
    dim = CONFIG_14B["dim"]
    approx_params_per_layer = (
        4 * dim * dim          # self-attn Q/K/V/O
        + 6 * dim * dim        # cross-attn (I2V has separate img K/V)
        + 2 * dim * CONFIG_14B["ffn_dim"]  # FFN
        + 6 * dim              # modulation
    )
    total_params = approx_params_per_layer * args.num_layers
    bytes_per_param = 2 if args.dtype == "bf16" else 4
    weight_gb = total_params * bytes_per_param / (1024 ** 3)
    print(f"Estimated DiT weight memory: ~{weight_gb:.1f} GB ({args.dtype})")
    if args.use_pallas:
        print("Pallas optimizations: ENABLED (fused AdaLN)")
    if args.use_scan:
        print("Layer scanning: ENABLED (jax.lax.scan over layers)")

    # --- Tensor parallelism mesh ---
    mesh = None
    if args.shard:
        mesh = create_mesh(mesh_shape=(1, len(devices)))
        print(f"Tensor parallelism: ENABLED (mesh shape (1, {len(devices)}), "
              f"{weight_gb / len(devices):.1f} GB/chip estimated)")
    else:
        print("Tensor parallelism: DISABLED (single chip)")

    # --- Select components ---
    if args.component == "all":
        component_names = list(COMPONENTS.keys())
    else:
        component_names = [args.component]

    results: list[BenchmarkResult] = []

    for name in component_names:
        bench_fn = COMPONENTS[name]

        # Components that may OOM: full_dit, full_inference
        if name in ("full_dit", "full_inference"):
            result = run_with_oom_fallback(
                bench_fn, name,
                batch_size=args.batch_size,
                num_layers=args.num_layers,
                dtype_str=args.dtype,
                warmup=args.warmup,
                iters=args.iters,
                use_pallas=args.use_pallas,
                use_scan=args.use_scan,
                mesh=mesh,
            )
        else:
            print(f"\n[{name}]")
            result = bench_fn(
                batch_size=args.batch_size,
                num_layers=args.num_layers,
                dtype_str=args.dtype,
                warmup=args.warmup,
                iters=args.iters,
                use_pallas=args.use_pallas,
                mesh=mesh,
            )

        results.append(result)
        if not result.note:
            print(f"  -> {result.mean_ms:.2f} ms (std={result.std_ms:.2f})")

    # --- Print table ---
    print_results_table(results, device_info)

    # --- Save JSON ---
    if args.save:
        save_results_json(results, args.save, device_info)


if __name__ == "__main__":
    main()
