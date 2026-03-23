#!/usr/bin/env python3
"""Comprehensive performance benchmark for DreamZero-JAX on TPU/GPU.

Measures inference latency, throughput, memory, hardware utilization,
per-component breakdown, and cost analysis. Designed to produce numbers
suitable for a LinkedIn post or paper table.

Protocol: 3 warmup + 10 timed iterations (configurable).

Usage:
    python scripts/benchmark_full.py \\
        --checkpoint-dir checkpoints/DreamZero-DROID \\
        --num-layers 40 \\
        --batch-sizes 1,2,4,8,16 \\
        --output results/benchmark.json
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import dreamzero_jax  # noqa: F401 — triggers nnx.List compat patch

from dreamzero_jax.models.dit import WanDiT, WanDiTBlock, WanDiTHead
from dreamzero_jax.models.action_head import CausalWanDiT
from dreamzero_jax.models.dreamzero import DreamZeroConfig
from dreamzero_jax.nn.embed import PatchEmbed3D, WanRoPE3D, sinusoidal_embedding
from dreamzero_jax.nn.mlp import MLP
from dreamzero_jax.schedulers.flow_euler import make_flow_euler_schedule, euler_step

# ---------------------------------------------------------------------------
# Hardware constants
# ---------------------------------------------------------------------------

V5E_PEAK_TFLOPS_BF16 = 197.0
V5E_PEAK_BW_GB_S = 819.0
V5E_4_SPOT_COST_HR = 4.80
V5E_8_SPOT_COST_HR = 9.60
H100_COST_HR = 2.50

# ---------------------------------------------------------------------------
# 14B model configuration
# ---------------------------------------------------------------------------

CONFIG_14B = dict(
    dim=5120,
    ffn_dim=13824,
    num_heads=40,
    freq_dim=256,
    text_dim=4096,
    patch_size=(1, 2, 2),
    in_channels=36,
    out_channels=16,
    has_image_input=True,
)

LATENT_T = 9
LATENT_H = 40
LATENT_W = 22
LATENT_C = 16
GRID_F = 9
GRID_H = 20
GRID_W = 11
SEQ_LEN = GRID_F * GRID_H * GRID_W  # 1980
TEXT_SEQ_LEN = 512


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class LatencyStats:
    mean_ms: float
    std_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float


@dataclass
class ThroughputResult:
    batch_size: int
    samples_per_sec: float
    latency: LatencyStats
    peak_hbm_gb: float | None


@dataclass
class MemoryBreakdown:
    peak_hbm_gb: float | None
    weight_memory_gb: float
    num_params: int
    bytes_per_param: float


@dataclass
class UtilizationEstimate:
    measured_tflops: float
    theoretical_peak_tflops: float
    mxu_utilization_pct: float
    measured_bw_gb_s: float | None
    bw_utilization_pct: float | None


@dataclass
class ComponentTiming:
    name: str
    mean_ms: float
    std_ms: float
    pct_of_total: float


@dataclass
class CostEntry:
    batch_size: int
    samples_per_sec: float
    cost_per_sample_v5e4: float
    cost_per_sample_v5e8: float
    cost_per_sample_h100: float | None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DreamZero-JAX full benchmark")
    p.add_argument(
        "--checkpoint-dir", type=str, default=None,
        help="Checkpoint directory (uses random weights if omitted)",
    )
    p.add_argument("--num-layers", type=int, default=40)
    p.add_argument("--batch-sizes", type=str, default="1,2,4,8,16")
    p.add_argument("--num-steps", type=int, default=16)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "f32"])
    p.add_argument("--output", type=str, default="results/benchmark.json")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "tpu", "gpu", "cpu"])
    p.add_argument("--skip-components", action="store_true")
    p.add_argument("--skip-throughput", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# HBM utilities
# ---------------------------------------------------------------------------


def get_hbm_gb() -> float | None:
    try:
        stats = jax.local_devices()[0].memory_stats()
        if stats and "bytes_in_use" in stats:
            return stats["bytes_in_use"] / (1024 ** 3)
    except Exception:
        pass
    return None


def get_peak_hbm_gb() -> float | None:
    try:
        stats = jax.local_devices()[0].memory_stats()
        if stats and "peak_bytes_in_use" in stats:
            return stats["peak_bytes_in_use"] / (1024 ** 3)
    except Exception:
        pass
    return None


def reset_peak_hbm() -> None:
    try:
        dev = jax.local_devices()[0]
        if hasattr(dev, "clear_memory_stats"):
            dev.clear_memory_stats()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def compute_latency_stats(timings_ms: np.ndarray) -> LatencyStats:
    return LatencyStats(
        mean_ms=float(np.mean(timings_ms)),
        std_ms=float(np.std(timings_ms)),
        p50_ms=float(np.percentile(timings_ms, 50)),
        p95_ms=float(np.percentile(timings_ms, 95)),
        p99_ms=float(np.percentile(timings_ms, 99)),
        min_ms=float(np.min(timings_ms)),
        max_ms=float(np.max(timings_ms)),
    )


def time_fn(fn, warmup: int, iters: int) -> tuple[np.ndarray, float | None]:
    """Run fn with warmup, return (timings_ms, peak_hbm_gb)."""
    for _ in range(warmup):
        result = fn()
        jax.block_until_ready(result)

    reset_peak_hbm()
    timings = []
    for _ in range(iters):
        t0 = time.perf_counter()
        result = fn()
        jax.block_until_ready(result)
        elapsed = (time.perf_counter() - t0) * 1000.0
        timings.append(elapsed)

    return np.array(timings), get_peak_hbm_gb()


def time_compilation(fn) -> float:
    """Measure first-call (compilation) time in ms."""
    t0 = time.perf_counter()
    result = fn()
    jax.block_until_ready(result)
    return (time.perf_counter() - t0) * 1000.0


# ---------------------------------------------------------------------------
# Model creation
# ---------------------------------------------------------------------------


def resolve_dtype(s: str) -> jnp.dtype:
    return jnp.bfloat16 if s == "bf16" else jnp.float32


def make_dit(num_layers: int, dtype: jnp.dtype) -> CausalWanDiT:
    rngs = nnx.Rngs(params=jax.random.PRNGKey(0))
    return CausalWanDiT(
        **CONFIG_14B,
        num_layers=num_layers,
        use_scan=True,
        use_remat=True,
        dtype=dtype,
        param_dtype=dtype,
        rngs=rngs,
    )


def count_params(model: nnx.Module) -> int:
    _, state = nnx.split(model)
    total = 0
    for _, v in state.flat_state():
        arr = v.value if hasattr(v, "value") else v
        total += int(np.prod(arr.shape))
    return total


def weight_memory_gb(model: nnx.Module) -> float:
    _, state = nnx.split(model)
    total_bytes = 0
    for _, v in state.flat_state():
        arr = v.value if hasattr(v, "value") else v
        total_bytes += int(np.prod(arr.shape)) * arr.dtype.itemsize
    return total_bytes / (1024 ** 3)


# ---------------------------------------------------------------------------
# Input builders
# ---------------------------------------------------------------------------


def make_dit_inputs(
    batch_size: int, dtype: jnp.dtype, num_layers: int,
) -> dict[str, jax.Array]:
    """Build synthetic inputs for CausalWanDiT forward pass."""
    B = batch_size
    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)

    x = jax.random.normal(k1, (B, LATENT_T, LATENT_H, LATENT_W, LATENT_C), dtype=dtype)
    timestep = jnp.full((B,), 500.0)
    context = jax.random.normal(k2, (B, TEXT_SEQ_LEN, CONFIG_14B["text_dim"]), dtype=dtype)

    num_blocks = GRID_F
    state = jax.random.normal(k3, (B, num_blocks, 64), dtype=jnp.float32)
    embodiment_id = jnp.zeros((B,), dtype=jnp.int32)

    total_actions = num_blocks * 32
    actions = jax.random.normal(k4, (B, total_actions, 32), dtype=dtype)
    timestep_action = jnp.full((B,), 500.0)
    clip_emb = jax.random.normal(k5, (B, 257, 1280), dtype=dtype)
    y = jax.random.normal(k6, (B, LATENT_T, LATENT_H, LATENT_W, 20), dtype=dtype)

    return dict(
        x=x, timestep=timestep, context=context, state=state,
        embodiment_id=embodiment_id, actions=actions,
        timestep_action=timestep_action, clip_emb=clip_emb, y=y,
    )


def make_denoise_inputs(
    batch_size: int, dtype: jnp.dtype, num_steps: int,
) -> dict[str, Any]:
    """Build inputs for the full denoising loop benchmark."""
    B = batch_size
    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)

    num_blocks = GRID_F
    total_actions = num_blocks * 32
    config = DreamZeroConfig(
        **CONFIG_14B,
        num_layers=40,
        dtype=dtype,
        param_dtype=dtype,
        num_inference_steps=num_steps,
    )

    noisy_video = jax.random.normal(k1, (B, LATENT_T, LATENT_H, LATENT_W, LATENT_C), dtype=dtype)
    noisy_actions = jax.random.normal(k2, (B, total_actions, 32), dtype=dtype)
    prompt_emb = jax.random.normal(k3, (B, TEXT_SEQ_LEN, CONFIG_14B["text_dim"]), dtype=dtype)
    state = jax.random.normal(k4, (B, num_blocks, 64), dtype=jnp.float32)
    embodiment_id = jnp.zeros((B,), dtype=jnp.int32)
    clip_emb = jax.random.normal(k5, (B, 257, 1280), dtype=dtype)
    y = jax.random.normal(k6, (B, LATENT_T, LATENT_H, LATENT_W, 20), dtype=dtype)

    schedule = make_flow_euler_schedule(
        num_inference_steps=num_steps,
        num_train_timesteps=config.num_train_timesteps,
        shift=config.scheduler_shift,
    )

    return dict(
        noisy_video=noisy_video,
        noisy_actions=noisy_actions,
        prompt_emb=prompt_emb,
        state=state,
        embodiment_id=embodiment_id,
        clip_emb=clip_emb,
        y=y,
        schedule=schedule,
        config=config,
    )


# ---------------------------------------------------------------------------
# 1. Inference latency benchmarks
# ---------------------------------------------------------------------------


def bench_single_forward(
    dit: CausalWanDiT, inputs: dict, warmup: int, iters: int,
) -> tuple[LatencyStats, float]:
    """Single denoising step latency + compilation time."""

    @nnx.jit
    def _step(model, x, t, ctx, st, eid, act, ta, ce, y_cond):
        return model(x, t, ctx, st, eid, act,
                     timestep_action=ta, clip_emb=ce, y=y_cond)

    args = (
        dit, inputs["x"], inputs["timestep"], inputs["context"],
        inputs["state"], inputs["embodiment_id"], inputs["actions"],
        inputs["timestep_action"], inputs["clip_emb"], inputs["y"],
    )

    compile_ms = time_compilation(lambda: _step(*args))

    timings, _ = time_fn(lambda: _step(*args), warmup, iters)
    return compute_latency_stats(timings), compile_ms


def bench_full_generation(
    dit: CausalWanDiT, denoise_inputs: dict, warmup: int, iters: int,
) -> LatencyStats:
    """Full N-step denoising loop latency."""

    sched = denoise_inputs["schedule"]
    config = denoise_inputs["config"]
    B = denoise_inputs["noisy_video"].shape[0]

    y = denoise_inputs["y"]

    @nnx.jit
    def _denoise_loop(
        model, noisy_vid, noisy_act, p_emb, st, eid, c_emb, y_cond,
    ):
        scan_xs = (
            sched.timesteps, sched.timesteps,
            sched.sigmas, sched.sigmas_next,
            sched.sigmas, sched.sigmas_next,
        )

        def _step(carry, xs):
            nv, na = carry
            tv, ta, sv, svn, sa, san = xs
            tv_b = jnp.broadcast_to(tv, (B,))
            ta_b = jnp.broadcast_to(ta, (B,))

            vp, ap = model(
                nv, tv_b, p_emb, st, eid, na,
                timestep_action=ta_b, clip_emb=c_emb, y=y_cond,
            )
            nv_next = euler_step(vp, nv, sv, svn)
            na_next = euler_step(ap, na, sa, san)
            return (nv_next, na_next), None

        (fv, fa), _ = jax.lax.scan(_step, (noisy_vid, noisy_act), scan_xs)
        return fv, fa

    fn_args = (
        dit,
        denoise_inputs["noisy_video"],
        denoise_inputs["noisy_actions"],
        denoise_inputs["prompt_emb"],
        denoise_inputs["state"],
        denoise_inputs["embodiment_id"],
        denoise_inputs["clip_emb"],
        y,
    )

    timings, _ = time_fn(lambda: _denoise_loop(*fn_args), warmup, iters)
    return compute_latency_stats(timings)


# ---------------------------------------------------------------------------
# 2. Throughput sweep
# ---------------------------------------------------------------------------


def bench_throughput(
    num_layers: int, dtype: jnp.dtype,
    batch_sizes: list[int], warmup: int, iters: int,
) -> list[ThroughputResult]:
    results = []
    for bs in batch_sizes:
        print(f"    batch_size={bs} ...", end=" ", flush=True)
        try:
            jax.clear_caches()
            gc.collect()

            cpu = jax.devices("cpu")[0]
            with jax.default_device(cpu):
                dit = make_dit(num_layers, dtype)
            inputs = make_dit_inputs(bs, dtype, num_layers)

            @nnx.jit
            def _fwd(m, x, t, ctx, st, eid, act, ta, ce, y_cond):
                return m(x, t, ctx, st, eid, act,
                         timestep_action=ta, clip_emb=ce, y=y_cond)

            args = (
                dit, inputs["x"], inputs["timestep"], inputs["context"],
                inputs["state"], inputs["embodiment_id"], inputs["actions"],
                inputs["timestep_action"], inputs["clip_emb"], inputs["y"],
            )

            timings, peak_hbm = time_fn(lambda: _fwd(*args), warmup, iters)
            stats = compute_latency_stats(timings)
            sps = bs / (stats.mean_ms / 1000.0)

            results.append(ThroughputResult(
                batch_size=bs,
                samples_per_sec=sps,
                latency=stats,
                peak_hbm_gb=peak_hbm,
            ))
            print(f"{sps:.2f} samples/s, {stats.mean_ms:.1f} ms")

            del dit, inputs
            gc.collect()
            jax.clear_caches()

        except Exception as e:
            print(f"OOM or error: {e}")
            break

    return results


# ---------------------------------------------------------------------------
# 3. Memory analysis
# ---------------------------------------------------------------------------


def analyze_memory(dit: CausalWanDiT) -> MemoryBreakdown:
    n_params = count_params(dit)
    w_gb = weight_memory_gb(dit)
    peak = get_peak_hbm_gb()
    _, state = nnx.split(dit)
    flat = list(state.flat_state())
    if flat:
        sample_arr = flat[0][1].value if hasattr(flat[0][1], "value") else flat[0][1]
        bpp = sample_arr.dtype.itemsize
    else:
        bpp = 2.0
    return MemoryBreakdown(
        peak_hbm_gb=peak,
        weight_memory_gb=w_gb,
        num_params=n_params,
        bytes_per_param=bpp,
    )


# ---------------------------------------------------------------------------
# 4. Hardware utilization estimate
# ---------------------------------------------------------------------------


def estimate_utilization(
    num_params: int, seq_len: int, latency_ms: float,
    num_devices: int,
) -> UtilizationEstimate:
    # 2 * num_params * seq_len FLOPs per forward pass (approximate)
    flops_per_fwd = 2.0 * num_params * seq_len
    measured_tflops = (flops_per_fwd / (latency_ms / 1000.0)) / 1e12

    peak_tflops = V5E_PEAK_TFLOPS_BF16 * num_devices
    mxu_pct = (measured_tflops / peak_tflops) * 100.0

    # Bandwidth: weight bytes read per forward / time
    weight_bytes = num_params * 2  # bf16
    bw_gb_s = (weight_bytes / (latency_ms / 1000.0)) / 1e9
    peak_bw = V5E_PEAK_BW_GB_S * num_devices
    bw_pct = (bw_gb_s / peak_bw) * 100.0

    return UtilizationEstimate(
        measured_tflops=measured_tflops,
        theoretical_peak_tflops=peak_tflops,
        mxu_utilization_pct=mxu_pct,
        measured_bw_gb_s=bw_gb_s,
        bw_utilization_pct=bw_pct,
    )


# ---------------------------------------------------------------------------
# 5. Per-component breakdown
# ---------------------------------------------------------------------------


def bench_component(
    name: str, fn, warmup: int, iters: int,
) -> ComponentTiming:
    timings, _ = time_fn(fn, warmup, iters)
    stats = compute_latency_stats(timings)
    return ComponentTiming(
        name=name, mean_ms=stats.mean_ms, std_ms=stats.std_ms, pct_of_total=0.0,
    )


def bench_components_breakdown(
    dit: CausalWanDiT, inputs: dict, warmup: int, iters: int,
) -> list[ComponentTiming]:
    """Time individual components of the DiT forward pass."""
    B = inputs["x"].shape[0]
    dtype = inputs["x"].dtype
    dim = CONFIG_14B["dim"]

    x_cat = jnp.concatenate([inputs["x"], inputs["y"]], axis=-1)

    def _patch_embed():
        return dit.patch_embedding.proj(x_cat)

    def _time_embed():
        t = sinusoidal_embedding(inputs["timestep"], dit.freq_dim)
        t = dit.time_embedding(t)
        e = jax.nn.silu(t)
        return dit.time_projection(e)

    def _text_embed():
        return dit.text_embedding(inputs["context"])

    x_patched = dit.patch_embedding.proj(x_cat)
    _, f, h, w, _ = x_patched.shape
    x_flat = x_patched.reshape(B, f * h * w, dim)

    t_raw = sinusoidal_embedding(inputs["timestep"], dit.freq_dim)
    t_emb = dit.time_embedding(t_raw)
    e_silu = jax.nn.silu(t_emb)
    e_proj = dit.time_projection(e_silu).reshape(B, 6, dim)

    ctx = dit.text_embedding(inputs["context"])
    if dit.has_image_input and inputs.get("clip_emb") is not None:
        img_ctx = dit.img_emb(inputs["clip_emb"])
        ctx = jnp.concatenate([img_ctx, ctx], axis=1)

    freqs = dit.rope(f, h, w)

    block0 = dit.blocks[0]

    def _single_block():
        return block0(x_flat, e_proj, ctx, freqs)

    # Head
    def _head():
        return dit.head(x_flat, t_emb)

    components = [
        ("patch_embed", _patch_embed),
        ("time_embed", _time_embed),
        ("text_embed", _text_embed),
        ("single_block", _single_block),
        ("head", _head),
    ]

    results = []
    for name, fn in components:
        print(f"    {name} ...", end=" ", flush=True)
        ct = bench_component(name, fn, warmup, iters)
        print(f"{ct.mean_ms:.2f} ms")
        results.append(ct)

    total = sum(c.mean_ms for c in results)
    for c in results:
        c.pct_of_total = (c.mean_ms / total * 100.0) if total > 0 else 0.0

    return results


# ---------------------------------------------------------------------------
# 7. Cost analysis
# ---------------------------------------------------------------------------


def compute_costs(throughput_results: list[ThroughputResult]) -> list[CostEntry]:
    costs = []
    for tr in throughput_results:
        sps = tr.samples_per_sec
        if sps <= 0:
            continue
        costs.append(CostEntry(
            batch_size=tr.batch_size,
            samples_per_sec=sps,
            cost_per_sample_v5e4=V5E_4_SPOT_COST_HR / (sps * 3600),
            cost_per_sample_v5e8=V5E_8_SPOT_COST_HR / (sps * 3600),
            cost_per_sample_h100=H100_COST_HR / (sps * 3600),
        ))
    return costs


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def print_header(title: str) -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def print_latency_stats(label: str, stats: LatencyStats) -> None:
    print(f"  {label}:")
    print(f"    mean={stats.mean_ms:.2f} ms  std={stats.std_ms:.2f} ms")
    print(f"    p50={stats.p50_ms:.2f}  p95={stats.p95_ms:.2f}  p99={stats.p99_ms:.2f} ms")
    print(f"    min={stats.min_ms:.2f}  max={stats.max_ms:.2f} ms")


def print_throughput_table(results: list[ThroughputResult]) -> None:
    print(f"  {'Batch':>6} {'Samples/s':>10} {'Latency(ms)':>12} {'Peak HBM':>10}")
    print(f"  {'-'*6} {'-'*10} {'-'*12} {'-'*10}")
    for r in results:
        hbm_str = f"{r.peak_hbm_gb:.2f} GB" if r.peak_hbm_gb else "N/A"
        print(
            f"  {r.batch_size:>6} "
            f"{r.samples_per_sec:>10.2f} "
            f"{r.latency.mean_ms:>12.1f} "
            f"{hbm_str:>10}"
        )


def print_cost_table(costs: list[CostEntry]) -> None:
    print(f"  {'Batch':>6} {'Samp/s':>8} {'$/samp v5e-4':>13} {'$/samp v5e-8':>13} {'$/samp H100':>12}")
    print(f"  {'-'*6} {'-'*8} {'-'*13} {'-'*13} {'-'*12}")
    for c in costs:
        print(
            f"  {c.batch_size:>6} "
            f"{c.samples_per_sec:>8.2f} "
            f"${c.cost_per_sample_v5e4:>12.6f} "
            f"${c.cost_per_sample_v5e8:>12.6f} "
            f"${c.cost_per_sample_h100:>11.6f}"
        )


def print_utilization(u: UtilizationEstimate) -> None:
    print(f"  Measured TFLOPS:     {u.measured_tflops:.2f}")
    print(f"  Theoretical peak:    {u.theoretical_peak_tflops:.1f} TFLOPS")
    print(f"  MXU utilization:     {u.mxu_utilization_pct:.1f}%")
    if u.measured_bw_gb_s is not None:
        print(f"  Memory BW:           {u.measured_bw_gb_s:.1f} GB/s")
    if u.bw_utilization_pct is not None:
        print(f"  BW utilization:      {u.bw_utilization_pct:.1f}%")


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------


def to_serializable(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "__dataclass_fields__"):
        return {k: to_serializable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    return obj


# ---------------------------------------------------------------------------
# Main benchmark orchestration
# ---------------------------------------------------------------------------


def run_all(args: argparse.Namespace) -> dict[str, Any]:
    dtype = resolve_dtype(args.dtype)
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    num_layers = args.num_layers
    backend = jax.default_backend() if args.device == "auto" else args.device
    num_devices = len(jax.devices())

    print_header("DreamZero-JAX Full Benchmark")
    print(f"  Backend:     {backend} ({num_devices} devices)")
    print(f"  Layers:      {num_layers}")
    print(f"  Dtype:       {args.dtype}")
    print(f"  Batch sizes: {batch_sizes}")
    print(f"  Steps:       {args.num_steps}")
    print(f"  Warmup:      {args.warmup}  Iters: {args.iters}")
    print(f"  Checkpoint:  {args.checkpoint_dir or '(random weights)'}")

    all_results: dict[str, Any] = {
        "metadata": {
            "backend": backend,
            "num_devices": num_devices,
            "num_layers": num_layers,
            "dtype": args.dtype,
            "num_steps": args.num_steps,
            "seq_len": SEQ_LEN,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        },
    }

    # --- Create model ---
    print("\n  Creating CausalWanDiT on CPU...")
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        dit = make_dit(num_layers, dtype)

    n_params = count_params(dit)
    w_mem = weight_memory_gb(dit)
    print(f"  Parameters:  {n_params:,} ({w_mem:.2f} GB)")

    # --- 3. Memory ---
    print_header("3. Memory Analysis")
    mem = analyze_memory(dit)
    all_results["memory"] = mem
    print(f"  Parameters:       {mem.num_params:,}")
    print(f"  Weight memory:    {mem.weight_memory_gb:.2f} GB")
    print(f"  Bytes/param:      {mem.bytes_per_param}")
    if mem.peak_hbm_gb is not None:
        print(f"  Peak HBM:         {mem.peak_hbm_gb:.2f} GB")

    # --- 1. Inference latency ---
    print_header("1. Inference Latency")
    inputs_b1 = make_dit_inputs(1, dtype, num_layers)

    print("  Single forward pass (compilation)...")
    single_stats, compile_ms = bench_single_forward(
        dit, inputs_b1, args.warmup, args.iters,
    )
    print_latency_stats("Per-forward-pass (1 denoising step)", single_stats)
    print(f"  First-call compilation: {compile_ms:.0f} ms")
    print(f"  Warm (post-compile):    {single_stats.mean_ms:.2f} ms")

    all_results["latency"] = {
        "single_step": single_stats,
        "compilation_ms": compile_ms,
    }

    # Full generation
    print("\n  Full generation (N denoising steps)...")
    denoise_inputs = make_denoise_inputs(1, dtype, args.num_steps)
    gen_stats = bench_full_generation(
        dit, denoise_inputs, args.warmup, args.iters,
    )
    print_latency_stats(
        f"Full generation ({args.num_steps} steps)", gen_stats,
    )
    all_results["latency"]["full_generation"] = gen_stats

    # --- 4. Hardware utilization ---
    print_header("4. Hardware Utilization (Estimated)")
    util = estimate_utilization(
        n_params, SEQ_LEN, single_stats.mean_ms, num_devices,
    )
    print_utilization(util)
    all_results["utilization"] = util

    # --- 5. Component breakdown ---
    if not args.skip_components:
        print_header("5. Per-Component Breakdown")
        comp_results = bench_components_breakdown(
            dit, inputs_b1, args.warmup, args.iters,
        )
        print(f"\n  {'Component':<20} {'Mean(ms)':>10} {'Std(ms)':>10} {'% Total':>8}")
        print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*8}")
        for c in comp_results:
            print(
                f"  {c.name:<20} {c.mean_ms:>10.2f} {c.std_ms:>10.2f} {c.pct_of_total:>7.1f}%"
            )
        all_results["components"] = comp_results

    # Free model before throughput sweep to avoid double memory
    del dit, inputs_b1
    gc.collect()
    jax.clear_caches()

    # --- 2. Throughput sweep ---
    if not args.skip_throughput:
        print_header("2. Throughput Sweep")
        tp_results = bench_throughput(
            num_layers, dtype, batch_sizes, args.warmup, args.iters,
        )
        print()
        print_throughput_table(tp_results)
        all_results["throughput"] = tp_results

        if tp_results:
            best = max(tp_results, key=lambda r: r.samples_per_sec)
            print(f"\n  Optimal batch size: {best.batch_size} ({best.samples_per_sec:.2f} samples/s)")
            print(f"  Max feasible batch: {tp_results[-1].batch_size}")

        # --- 7. Cost analysis ---
        print_header("7. Cost Analysis")
        costs = compute_costs(tp_results)
        print_cost_table(costs)
        all_results["cost"] = costs

    return all_results


def save_results(results: dict[str, Any], output_path: str) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(to_serializable(results), f, indent=2, default=str)
    print(f"\n  Results saved to: {out}")


def main() -> None:
    args = parse_args()
    results = run_all(args)
    save_results(results, args.output)
    print("\n  Benchmark complete.")


if __name__ == "__main__":
    main()
