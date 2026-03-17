# DreamZero-JAX

**JAX/Flax port of NVIDIA's DreamZero 14B World Action Model, optimized for Google TPUs with custom Pallas kernels.**

DreamZero is a World Action Model that jointly predicts actions and videos from language instructions and visual observations, achieving zero-shot generalization to unseen robotics tasks. This repo is a from-scratch JAX/Flax NNX reimplementation targeting TPU v5e pods, with custom Pallas kernels for the performance-critical DiT backbone.

> **Status**: Core model ported and running on TPU. Active optimization phase.

---

## Benchmark Results

Measured on TPU v5litepod-4 (4 chips, 64GB HBM), JAX 0.6.2, bf16, batch=1.

| Component | TPU v5e (ms) | H100 (ms) | Notes |
|---|---|---|---|
| Single DiT block (d=5120) | **12.17** | — | std=0.01ms |
| Full DiT (8 layers) | **140.78** | — | 64GB limits to 8L |
| Full DiT (40 layers, est.) | **~530** | — | extrapolated from per-layer |
| VAE encode (33f@320x176) | **167.41** | — | std=0.02ms |
| VAE decode (33f@320x176) | **351.20** | — | std=0.05ms |
| Full inference (16 steps) | ~17,000 (est.) | ~3,000 | needs v5e-8 (128GB) |

Reproduce with:
```bash
uv run python scripts/benchmark_components.py --component all
```

---

## TPU Optimizations

### Fused AdaLN Pallas Kernel — 1.81x per layer

The DiT block's adaptive layer norm (shift, scale, gate) is fused into a single Pallas kernel that eliminates intermediate materializations. This is the single largest speedup and applies to every one of the 40 transformer layers.

### Chunked Attention for Large Sequences

Video tokens (880 per frame x 33 frames) exceed TPU HBM for standard attention. We use block-causal chunked attention that processes frame groups while maintaining the autoregressive mask required for action prediction.

### Scan-Compiled Denoising Loop

The full denoising loop (16 flow-matching steps) is wrapped in `jax.lax.scan` so XLA compiles it as a single fused program rather than 16 separate dispatches. This eliminates per-step host overhead.

### Coming Soon

- **KV caching** for DiT — skip recomputation of clean-image keys/values across denoising steps
- **Fused RoPE** — Pallas kernel to apply 3D rotary embeddings in-place
- **Multi-host sharding** — tensor-parallel across TPU v5e pod slices

---

## Quick Start

```bash
# Clone and install (requires uv)
git clone https://github.com/IronleafAI/dreamzero-jax.git
cd dreamzero-jax
uv sync

# Convert PyTorch checkpoint to Flax format
uv run python scripts/convert_checkpoint.py --input <pytorch_ckpt> --output <flax_ckpt>

# Run inference on TPU VM
uv run python scripts/inference.py --checkpoint <flax_ckpt> --input <observation>

# Run component benchmarks
uv run python scripts/benchmark_components.py --component all

# Validate outputs match PyTorch reference
uv run python scripts/validate_against_pytorch.py
```

---

## Architecture

DreamZero is a DiT-based diffusion model for joint video and action prediction:

```
Text Instruction ──> Text Encoder (T5) ──────────────────────────┐
                                                                  │
Multi-view Images ──> Image Encoder (CLIP ViT-H/14) ─────────────┤
                                                                  ├──> DiT Backbone (40L, 5120d) ──> Video + Actions
Video Frames ──> VAE Encoder ──> Latents ────────────────────────┤
                                                                  │
Timestep ──> Sinusoidal Embed ──> Time MLP ──────────────────────┘
```

- **14B parameters** (DROID checkpoint)
- **Flow matching** scheduler with shifted sigma schedule
- **Causal chunked attention** for autoregressive video generation with action conditioning

See **[ARCHITECTURE.md](ARCHITECTURE.md)** for full component specs, dimension tables, and PyTorch-to-JAX mapping.

---

## Project Structure

```
dreamzero-jax/
├── src/dreamzero_jax/
│   ├── nn/                  # Core building blocks
│   │   ├── attention.py     # Causal/chunked attention
│   │   ├── embed.py         # Sinusoidal, RoPE, patch embed
│   │   ├── mlp.py           # SwiGLU, GeGLU
│   │   └── pallas_ops.py    # Custom Pallas kernels (fused AdaLN, etc.)
│   │
│   ├── models/              # High-level architectures
│   │   ├── dit.py           # DiT blocks + full backbone
│   │   ├── vae.py           # Video VAE encoder/decoder
│   │   ├── text_encoder.py  # T5 wrapper
│   │   ├── image_encoder.py # CLIP ViT-H/14 wrapper
│   │   ├── action_head.py   # Flow matching action head
│   │   └── dreamzero.py     # Full model assembly
│   │
│   ├── schedulers/          # Diffusion schedulers
│   │   ├── flow_matching.py
│   │   ├── flow_euler.py
│   │   └── unipc.py
│   │
│   ├── data/                # Data loading & transforms
│   └── utils/               # Checkpoint conversion, sharding, validation
│
├── scripts/
│   ├── benchmark_components.py
│   ├── convert_checkpoint.py
│   ├── inference.py
│   ├── validate_against_pytorch.py
│   └── train.py
│
└── tests/
```

---

## Code Style

- **Flax NNX** (not Linen) for all neural network modules
- Prefer `nnx.Module` subclasses with mutable state
- Use `nnx.Rngs` for PRNG management

## Key Dependencies

```
jax[tpu]
flax
optax
orbax-checkpoint
grain
transformers
```

This project uses **uv** for package management.

---

## Development Roadmap

1. **Phase 1** — Core model architecture (done)
2. **Phase 2** — Inference pipeline + checkpoint conversion (in progress)
3. **Phase 3** — Training pipeline with distributed data loading
4. **Phase 4** — Optimization (Pallas kernels, DiT caching, multi-host)

## References

- [DreamZero Paper](https://dreamzero0.github.io/DreamZero.pdf)
- [DreamZero GitHub](https://github.com/dreamzero0/dreamzero)
- [JAX Documentation](https://jax.readthedocs.io/)
- [Flax NNX Documentation](https://flax.readthedocs.io/en/latest/nnx/)
- [TPU Best Practices](https://cloud.google.com/tpu/docs/best-practices)
