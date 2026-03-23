#!/usr/bin/env python3
"""GPU performance benchmark for the PyTorch standalone CausalWanDiT forward pass.

Measures inference latency, throughput, peak memory, estimated FLOPS,
and per-component timing on CUDA using the same forward pass as
pt_full_dit_forward.py.

Usage::

    python scripts/benchmark_gpu.py \
        --checkpoint-dir checkpoints/DreamZero-DROID \
        --num-layers 8 \
        --batch-sizes 1,2,4,8 \
        --output results/benchmark_gpu.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pt_full_dit_forward import (
    ACTION_DIM,
    ACTION_HIDDEN,
    DROID_DIM,
    DROID_HEAD_DIM,
    DROID_HEADS,
    DROID_TEXT_DIM,
    I2V_CHANNELS,
    IMAGE_DIM,
    IN_CHANNELS,
    NUM_ACTION_PER_BLOCK,
    NUM_FRAMES_PER_BLOCK,
    NUM_IMAGE_TOKENS,
    NUM_STATE_PER_BLOCK,
    OUT_CHANNELS,
    PATCH_SIZE,
    STATE_DIM,
    action_decoder_forward,
    action_encoder_forward,
    causal_wan_dit_forward,
    compute_needed_prefixes,
    create_video_freqs,
    dit_block_forward,
    head_forward_per_token,
    load_selective_weights,
    make_test_inputs,
    patch_embed_forward,
    rope_params_polar,
    state_encoder_forward,
    strip_prefix,
    text_conditioning,
    time_conditioning_per_token,
    validate_block_availability,
)
from pt_standalone_ops import DROID_FREQ_DIM, SEED


# ---------------------------------------------------------------------------
# GPU info via nvidia-smi
# ---------------------------------------------------------------------------


def get_gpu_info() -> dict:
    """Query GPU name and total VRAM from nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        parts = out.split("\n")[0].split(", ")
        return {"gpu_name": parts[0], "vram_mb": int(parts[1])}
    except Exception:
        return {"gpu_name": "unknown", "vram_mb": 0}


# ---------------------------------------------------------------------------
# Input creation (GPU)
# ---------------------------------------------------------------------------


def make_gpu_inputs(batch_size: int, num_blocks: int, h_patches: int,
                    w_patches: int, text_len: int, device: torch.device) -> dict:
    """Create deterministic test inputs on the given CUDA device."""
    torch.manual_seed(SEED)
    f = (num_blocks + 1) * NUM_FRAMES_PER_BLOCK
    H = h_patches * PATCH_SIZE[1]
    W = w_patches * PATCH_SIZE[2]
    B = batch_size

    return dict(
        x=torch.randn(B, f, H, W, IN_CHANNELS, device=device),
        clean_x=torch.randn(B, f, H, W, IN_CHANNELS, device=device),
        y=torch.randn(B, f, H, W, I2V_CHANNELS, device=device),
        timestep=torch.full((B,), 500.0, device=device),
        timestep_action=torch.full((B,), 300.0, device=device),
        context=torch.randn(B, text_len, DROID_TEXT_DIM, device=device),
        clip_emb=torch.randn(B, NUM_IMAGE_TOKENS, IMAGE_DIM, device=device),
        state=torch.randn(B, num_blocks, STATE_DIM, device=device),
        embodiment_id=torch.zeros(B, dtype=torch.long, device=device),
        actions=torch.randn(B, num_blocks * NUM_ACTION_PER_BLOCK, ACTION_DIM, device=device),
    )


# ---------------------------------------------------------------------------
# Latency benchmark
# ---------------------------------------------------------------------------


def benchmark_latency(weights, num_layers: int, device: torch.device,
                      num_blocks: int, h_patches: int, w_patches: int,
                      text_len: int, warmup: int = 3, trials: int = 10) -> dict:
    """Measure per-forward-pass latency with warmup."""
    inputs = make_gpu_inputs(1, num_blocks, h_patches, w_patches, text_len, device)
    move_weights_to_device(weights, device)

    for _ in range(warmup):
        _run_forward(inputs, weights, num_layers)
    torch.cuda.synchronize()

    times_ms = []
    for _ in range(trials):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _run_forward(inputs, weights, num_layers)
        torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(times_ms)
    return {
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "all_ms": [round(t, 3) for t in times_ms],
    }


# ---------------------------------------------------------------------------
# Throughput benchmark
# ---------------------------------------------------------------------------


def benchmark_throughput(weights, num_layers: int, device: torch.device,
                         batch_sizes: list[int], num_blocks: int,
                         h_patches: int, w_patches: int,
                         text_len: int) -> list[dict]:
    """Measure samples/sec at each batch size, stopping at OOM."""
    move_weights_to_device(weights, device)
    results = []

    for bs in batch_sizes:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        try:
            inputs = make_gpu_inputs(bs, num_blocks, h_patches, w_patches, text_len, device)
            _run_forward(inputs, weights, num_layers)
            torch.cuda.synchronize()

            t0 = time.perf_counter()
            _run_forward(inputs, weights, num_layers)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            results.append({
                "batch_size": bs,
                "elapsed_s": round(elapsed, 4),
                "samples_per_sec": round(bs / elapsed, 2),
                "peak_memory_mb": round(peak_mb, 1),
            })
            print(f"  batch={bs:>3d}  {bs / elapsed:>8.2f} samples/s  "
                  f"peak={peak_mb:>8.1f} MB  time={elapsed:.3f}s")
        except torch.cuda.OutOfMemoryError:
            print(f"  batch={bs:>3d}  OOM")
            results.append({"batch_size": bs, "status": "OOM"})
            torch.cuda.empty_cache()
            break

    return results


# ---------------------------------------------------------------------------
# Peak memory benchmark
# ---------------------------------------------------------------------------


def benchmark_memory(weights, num_layers: int, device: torch.device,
                     num_blocks: int, h_patches: int, w_patches: int,
                     text_len: int) -> dict:
    """Measure peak GPU memory for a single forward pass."""
    move_weights_to_device(weights, device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    inputs = make_gpu_inputs(1, num_blocks, h_patches, w_patches, text_len, device)
    _run_forward(inputs, weights, num_layers)
    torch.cuda.synchronize()

    peak_bytes = torch.cuda.max_memory_allocated()
    return {
        "peak_memory_mb": round(peak_bytes / (1024 ** 2), 1),
        "peak_memory_gb": round(peak_bytes / (1024 ** 3), 3),
    }


# ---------------------------------------------------------------------------
# FLOPS estimate
# ---------------------------------------------------------------------------


def estimate_flops(num_params: int, seq_len: int, latency_s: float) -> dict:
    """Rough FLOPS estimate: 2 * params * seq_len / time."""
    flops = 2 * num_params * seq_len / latency_s
    return {
        "estimated_flops": flops,
        "estimated_tflops": round(flops / 1e12, 2),
        "num_params": num_params,
        "seq_len": seq_len,
        "latency_s": round(latency_s, 4),
    }


# ---------------------------------------------------------------------------
# Per-component timing
# ---------------------------------------------------------------------------


def benchmark_components(weights, num_layers: int, device: torch.device,
                         num_blocks: int, h_patches: int, w_patches: int,
                         text_len: int) -> dict:
    """Time individual components: patch_embed, self_attn, cross_attn, ffn, head."""
    move_weights_to_device(weights, device)
    inputs = make_gpu_inputs(1, num_blocks, h_patches, w_patches, text_len, device)
    B = 1
    f = (num_blocks + 1) * NUM_FRAMES_PER_BLOCK
    seq_len = f * h_patches * w_patches
    frame_seqlen = h_patches * w_patches

    timings = {}

    def timed(name, fn):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = fn()
        torch.cuda.synchronize()
        timings[name] = round((time.perf_counter() - t0) * 1000.0, 3)
        return result

    x_in = inputs["x"]
    clean_x = inputs["clean_x"]
    y_in = inputs["y"]
    x_cat = torch.cat([x_in, y_in], dim=-1)
    clean_cat = torch.cat([clean_x, y_in], dim=-1)

    x_patched, f_g, h_g, w_g = timed(
        "patch_embed_ms",
        lambda: patch_embed_forward(x_cat, weights, prefix="model.patch_embedding"),
    )
    x_flat = x_patched.reshape(B, seq_len, DROID_DIM)

    clean_patched, _, _, _ = patch_embed_forward(clean_cat, weights, prefix="model.patch_embedding")
    clean_flat = clean_patched.reshape(B, seq_len, DROID_DIM)

    action_emb = action_encoder_forward(
        inputs["actions"], inputs["timestep_action"],
        inputs["embodiment_id"], weights)
    state_emb = state_encoder_forward(
        inputs["state"], inputs["embodiment_id"], weights)
    action_register = torch.cat([action_emb, state_emb], dim=1)
    action_register_length = action_register.shape[1]
    action_length = action_emb.shape[1]
    x_seq = torch.cat([x_flat, action_register], dim=1)
    full_seq = torch.cat([clean_flat, x_seq], dim=1)

    ts_video = inputs["timestep"][:, None].expand(B, seq_len)
    ts_action = inputs["timestep_action"][:, None].expand(B, action_length)
    stride = ts_action.shape[1] // state_emb.shape[1]
    ts_state = ts_action[:, ::stride]
    ts_full = torch.cat([ts_video, ts_action, ts_state], dim=1)
    ts_clean = torch.zeros(B, seq_len, device=device)
    ts_full = torch.cat([ts_clean, ts_full], dim=1)

    e_flat, e0_flat = timed(
        "time_cond_ms",
        lambda: time_conditioning_per_token(ts_full.flatten(), weights),
    )
    total_L = full_seq.shape[1]
    e_tokens = e_flat.reshape(B, total_L, DROID_DIM)
    e0_tokens = e0_flat.reshape(B, total_L, 6, DROID_DIM)

    ctx = timed(
        "text_cond_ms",
        lambda: text_conditioning(inputs["context"], weights),
    )

    from pt_full_dit_forward import has_weight, img_emb_forward
    if has_weight(weights, "model.img_emb.proj.0.weight"):
        img_ctx = img_emb_forward(inputs["clip_emb"], weights)
        ctx = torch.cat([img_ctx, ctx], dim=1)
    use_i2v_ca = has_weight(weights, "model.img_emb.proj.0.weight")

    freqs = create_video_freqs(DROID_HEAD_DIM, f_g, h_g, w_g)
    freqs_action = rope_params_polar(1024 * 10, DROID_HEAD_DIM)
    freqs_state = rope_params_polar(1024, DROID_HEAD_DIM)

    sa_times = []
    ca_times = []
    ffn_times = []
    block_times = []

    for i in range(min(num_layers, 4)):
        blk = f"model.blocks.{i}"
        D = DROID_DIM

        torch.cuda.synchronize()
        t_blk_start = time.perf_counter()

        from pt_full_dit_forward import (
            dit_self_attn_original,
            get_weight,
            i2v_cross_attention,
            text_only_cross_attention,
        )
        from pt_standalone_ops import gelu_approx, linear_forward, rmsnorm_forward

        mod_param = get_weight(weights, f"{blk}.modulation")
        mod = mod_param.unsqueeze(1) + e0_tokens
        shift_msa, scale_msa, gate_msa = mod[:, :, 0], mod[:, :, 1], mod[:, :, 2]
        shift_mlp, scale_mlp, gate_mlp = mod[:, :, 3], mod[:, :, 4], mod[:, :, 5]

        h_norm = torch.layer_norm(full_seq, [D]) * (1 + scale_msa) + shift_msa

        torch.cuda.synchronize()
        t_sa0 = time.perf_counter()
        sa = dit_self_attn_original(
            h_norm, weights, f"{blk}.self_attn", freqs, freqs_action, freqs_state,
            action_register_length, frame_seqlen,
            NUM_FRAMES_PER_BLOCK, NUM_ACTION_PER_BLOCK, NUM_STATE_PER_BLOCK,
            is_tf=True)
        torch.cuda.synchronize()
        sa_times.append((time.perf_counter() - t_sa0) * 1000.0)

        x_after_sa = full_seq + sa * gate_msa

        n3_w = get_weight(weights, f"{blk}.norm3.weight")
        n3_b = get_weight(weights, f"{blk}.norm3.bias")
        h_ca = torch.nn.functional.layer_norm(x_after_sa, [D], n3_w, n3_b)

        torch.cuda.synchronize()
        t_ca0 = time.perf_counter()
        if use_i2v_ca:
            ca = i2v_cross_attention(h_ca, ctx, weights, f"{blk}.cross_attn", DROID_HEADS)
        else:
            ca = text_only_cross_attention(h_ca, ctx, weights, f"{blk}.cross_attn", DROID_HEADS)
        torch.cuda.synchronize()
        ca_times.append((time.perf_counter() - t_ca0) * 1000.0)

        x_after_ca = x_after_sa + ca

        h_ffn = torch.layer_norm(x_after_ca, [D]) * (1 + scale_mlp) + shift_mlp
        torch.cuda.synchronize()
        t_ffn0 = time.perf_counter()
        h_ffn = gelu_approx(linear_forward(
            h_ffn, get_weight(weights, f"{blk}.ffn.0.weight"),
            get_weight(weights, f"{blk}.ffn.0.bias")))
        h_ffn = linear_forward(
            h_ffn, get_weight(weights, f"{blk}.ffn.2.weight"),
            get_weight(weights, f"{blk}.ffn.2.bias"))
        torch.cuda.synchronize()
        ffn_times.append((time.perf_counter() - t_ffn0) * 1000.0)

        full_seq = x_after_ca + h_ffn * gate_mlp

        torch.cuda.synchronize()
        block_times.append((time.perf_counter() - t_blk_start) * 1000.0)

    video_pred = full_seq[:, seq_len:2 * seq_len]
    e_video = e_tokens[:, seq_len:2 * seq_len]
    timed("head_ms", lambda: head_forward_per_token(video_pred, e_video, weights))

    action_pred = full_seq[:, 2 * seq_len:2 * seq_len + action_length]
    timed("action_decode_ms", lambda: action_decoder_forward(
        action_pred, inputs["embodiment_id"], weights))

    timings["self_attn_mean_ms"] = round(float(np.mean(sa_times)), 3)
    timings["cross_attn_mean_ms"] = round(float(np.mean(ca_times)), 3)
    timings["ffn_mean_ms"] = round(float(np.mean(ffn_times)), 3)
    timings["block_mean_ms"] = round(float(np.mean(block_times)), 3)
    timings["blocks_profiled"] = len(block_times)

    return timings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def move_weights_to_device(weights: dict, device: torch.device):
    """Move numpy weight arrays to torch tensors on device, in-place."""
    for key in weights:
        val = weights[key]
        if isinstance(val, np.ndarray):
            weights[key] = torch.from_numpy(val.copy()).float().to(device)


def _run_forward(inputs: dict, weights: dict, num_layers: int):
    """Run the full forward pass, suppressing print output."""
    import io
    import contextlib

    with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
        causal_wan_dit_forward(
            x=inputs["x"], timestep=inputs["timestep"], context=inputs["context"],
            state=inputs["state"], embodiment_id=inputs["embodiment_id"],
            actions=inputs["actions"], timestep_action=inputs["timestep_action"],
            clean_x=inputs["clean_x"], clip_emb=inputs["clip_emb"],
            y=inputs["y"], weights=weights, num_layers=num_layers,
        )


def count_params(weights: dict) -> int:
    """Count total parameters across all weight tensors."""
    total = 0
    for v in weights.values():
        if isinstance(v, np.ndarray):
            total += v.size
        elif isinstance(v, torch.Tensor):
            total += v.numel()
    return total


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def print_results_table(results: dict):
    """Print a formatted summary table."""
    sep = "=" * 64
    print(f"\n{sep}")
    print("  GPU Benchmark Results")
    print(sep)

    gpu = results.get("gpu_info", {})
    print(f"  GPU:  {gpu.get('gpu_name', '?')}  ({gpu.get('vram_mb', '?')} MB)")
    print(f"  Layers: {results['config']['num_layers']}  "
          f"Params: {results['num_params']:,}")
    print(sep)

    lat = results.get("latency", {})
    if lat:
        print("\n  Latency (batch=1)")
        print(f"    mean:  {lat['mean_ms']:>8.2f} ms")
        print(f"    std:   {lat['std_ms']:>8.2f} ms")
        print(f"    p50:   {lat['p50_ms']:>8.2f} ms")
        print(f"    p95:   {lat['p95_ms']:>8.2f} ms")

    thr = results.get("throughput", [])
    if thr:
        print("\n  Throughput")
        print(f"    {'batch':>6s}  {'samples/s':>10s}  {'peak MB':>10s}")
        for row in thr:
            if "status" in row:
                print(f"    {row['batch_size']:>6d}  {'OOM':>10s}")
            else:
                print(f"    {row['batch_size']:>6d}  "
                      f"{row['samples_per_sec']:>10.2f}  "
                      f"{row['peak_memory_mb']:>10.1f}")

    mem = results.get("memory", {})
    if mem:
        print(f"\n  Peak Memory (batch=1): {mem['peak_memory_gb']:.3f} GB")

    flops = results.get("flops_estimate", {})
    if flops:
        print(f"\n  FLOPS Estimate: {flops['estimated_tflops']:.2f} TFLOPS")

    comp = results.get("components", {})
    if comp:
        print("\n  Per-Component (ms)")
        for k in ["patch_embed_ms", "time_cond_ms", "text_cond_ms",
                   "self_attn_mean_ms", "cross_attn_mean_ms",
                   "ffn_mean_ms", "head_ms", "action_decode_ms"]:
            if k in comp:
                label = k.replace("_ms", "").replace("_mean", "")
                print(f"    {label:<20s} {comp[k]:>8.3f}")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="GPU benchmark for PT CausalWanDiT")
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--num-layers", type=int, default=8)
    p.add_argument("--num-blocks", type=int, default=2)
    p.add_argument("--h-patches", type=int, default=4)
    p.add_argument("--w-patches", type=int, default=4)
    p.add_argument("--text-len", type=int, default=16)
    p.add_argument("--batch-sizes", type=str, default="1,2,4,8",
                   help="Comma-separated batch sizes for throughput test")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--output", type=str, default="results/benchmark_gpu.json")
    p.add_argument("--skip-components", action="store_true",
                   help="Skip per-component profiling")
    return p.parse_args()


def main():
    args = parse_args()
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This benchmark requires a GPU.")
        sys.exit(1)

    device = torch.device("cuda")
    gpu_info = get_gpu_info()
    print(f"\n{'=' * 64}")
    print(f"  GPU Benchmark: CausalWanDiT Forward Pass")
    print(f"  GPU: {gpu_info['gpu_name']}  VRAM: {gpu_info['vram_mb']} MB")
    print(f"  Checkpoint: {args.checkpoint_dir}")
    print(f"  Layers: {args.num_layers}  Batches: {batch_sizes}")
    print(f"{'=' * 64}\n")

    config = dict(
        num_layers=args.num_layers, num_blocks=args.num_blocks,
        h_patches=args.h_patches, w_patches=args.w_patches,
        text_len=args.text_len, batch_sizes=batch_sizes,
        warmup=args.warmup, trials=args.trials,
    )

    print("[1/6] Loading checkpoint...")
    has_i2v = True
    prefixes = compute_needed_prefixes(args.num_layers, has_i2v)
    raw_weights = load_selective_weights(args.checkpoint_dir, prefixes)
    weights = strip_prefix(raw_weights, "action_head.")
    max_layers = validate_block_availability(weights, args.num_layers)
    if max_layers < args.num_layers:
        print(f"  Only {max_layers}/{args.num_layers} blocks complete, reducing.")
        args.num_layers = max_layers
        config["num_layers"] = max_layers
    num_params = count_params(weights)
    print(f"  {len(weights)} tensors, {num_params:,} parameters\n")

    print("[2/6] Latency benchmark...")
    latency = benchmark_latency(
        weights, args.num_layers, device,
        args.num_blocks, args.h_patches, args.w_patches, args.text_len,
        warmup=args.warmup, trials=args.trials)
    print(f"  mean={latency['mean_ms']:.2f} ms  std={latency['std_ms']:.2f} ms\n")

    print("[3/6] Throughput benchmark...")
    throughput = benchmark_throughput(
        weights, args.num_layers, device, batch_sizes,
        args.num_blocks, args.h_patches, args.w_patches, args.text_len)
    print()

    print("[4/6] Memory benchmark...")
    memory = benchmark_memory(
        weights, args.num_layers, device,
        args.num_blocks, args.h_patches, args.w_patches, args.text_len)
    print(f"  peak={memory['peak_memory_gb']:.3f} GB\n")

    print("[5/6] FLOPS estimate...")
    f = (args.num_blocks + 1) * NUM_FRAMES_PER_BLOCK
    seq_len = f * args.h_patches * args.w_patches
    flops_est = estimate_flops(num_params, seq_len, latency["mean_ms"] / 1000.0)
    print(f"  ~{flops_est['estimated_tflops']:.2f} TFLOPS\n")

    components = {}
    if not args.skip_components:
        print("[6/6] Per-component profiling...")
        weights_copy = {k: (v.cpu().numpy() if isinstance(v, torch.Tensor) else v)
                        for k, v in weights.items()}
        components = benchmark_components(
            weights_copy, args.num_layers, device,
            args.num_blocks, args.h_patches, args.w_patches, args.text_len)
        print(f"  done\n")
    else:
        print("[6/6] Skipping per-component profiling\n")

    results = {
        "gpu_info": gpu_info,
        "config": config,
        "num_params": num_params,
        "latency": latency,
        "throughput": throughput,
        "memory": memory,
        "flops_estimate": flops_est,
        "components": components,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    print_results_table(results)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
