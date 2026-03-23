# DreamZero-JAX: Run NVIDIA's 14B World-Action Model on Google TPU

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.4.35+-green.svg)](https://github.com/jax-ml/jax)

A from-scratch JAX/Flax NNX port of [NVIDIA's DreamZero](https://github.com/dreamzero0/dreamzero), the 14B World Action Model that jointly predicts actions and videos from language instructions and visual observations. **Validated: 26/26 component parity with PyTorch, 100% weight loading, 130 tests pass.**

---

## Quickstart

Five commands from zero to inference:

```bash
git clone https://github.com/PalmDr/dreamzero-jax
cd dreamzero-jax
uv sync

# Quick test (random weights, no download needed)
JAX_PLATFORMS=cpu uv run python scripts/inference.py --device cpu --num-layers 2

# Full inference with real weights (requires TPU — see below)
# Download: uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('GEAR-Dreams/DreamZero-DROID', local_dir='checkpoints/DreamZero-DROID')"
# Run: uv run python scripts/inference.py --checkpoint checkpoints/DreamZero-DROID
```

---

## TPU Inference

### Create a TPU VM

```bash
gcloud compute tpus tpu-vm create dreamzero \
    --zone=us-central2-b \
    --accelerator-type=v5litepod-8 \
    --version=v2-alpha-tpuv5-lite
```

### Install

```bash
git clone https://github.com/PalmDr/dreamzero-jax
cd dreamzero-jax
pip install uv && uv pip install -e '.[tpu]'
```

### Run staged inference with real weights

The full 14B model (40 layers) exceeds single-phase HBM on v5e-8. Staged inference solves this by loading encoders and DiT sequentially -- peak memory is `max(encoders, DiT)` rather than the sum.

```bash
uv run python scripts/inference.py \
    --checkpoint checkpoints/DreamZero-DROID \
    --input-video /path/to/video.mp4 \
    --prompt "pick up the red block" \
    --dtype bfloat16 \
    --num-steps 16 \
    --cfg-scale 5.0
```

> **Precision note:** JAX on TPU defaults to reduced-precision matmuls. For exact PyTorch parity during validation, set `jax.config.update("jax_default_matmul_precision", "float32")`. For production inference, the default bf16 precision is fine and faster.

### TPU sizing guide

| Configuration | HBM/chip | Fits 40L DiT | Full pipeline |
|---|---|---|---|
| v5e-4 (4 chips, 64 GB) | 16 GB | No (8L max) | No |
| v5e-8 (8 chips, 128 GB) | 16 GB | Yes (staged) | Yes (24L direct, 40L staged) |
| v5e-16 (16 chips, 256 GB) | 16 GB | Yes | Yes (40L direct) |

---

## Validation Results

26/26 components pass numerical parity against the PyTorch reference, using real DROID checkpoint weights and identical seeded inputs (`seed=42`, `jax_default_matmul_precision=float32`).

| Component | Max Abs Diff | Cosine Sim | Status |
|---|---|---|---|
| RMSNorm | < 1e-6 | 1.000000 | PASS |
| Linear | < 1e-6 | 1.000000 | PASS |
| Self-Attention | < 1e-5 | 1.000000 | PASS |
| Cross-Attention | < 1e-5 | 1.000000 | PASS |
| AdaLN Modulation | < 1e-5 | 1.000000 | PASS |
| FFN (GELU) | < 1e-5 | 1.000000 | PASS |
| Time Embedding | < 1e-5 | 1.000000 | PASS |
| 3D RoPE | < 1e-6 | 1.000000 | PASS |
| Patch Embed | < 1e-6 | 1.000000 | PASS |
| Text Embedding | < 1e-4 | 1.000000 | PASS |
| Text Encoder Block | < 1e-4 | 1.000000 | PASS |
| Text Encoder (full) | < 1e-3 | 0.999999 | PASS |
| Image Encoder (CLIP) | < 1e-3 | 0.999999 | PASS |
| VAE Encoder | < 1e-4 | 1.000000 | PASS |
| VAE Decoder | < 1e-4 | 1.000000 | PASS |
| DiT Block (single) | < 1e-5 | 1.000000 | PASS |
| DiT Backbone (8L) | < 1e-3 | 0.999998 | PASS |
| DiT Backbone (24L) | < 1e-3 | 0.999997 | PASS |
| DiT Backbone (40L) | < 1e-3 | 0.999995 | PASS |
| Action Encoder | < 1e-5 | 1.000000 | PASS |
| State Encoder | < 1e-5 | 1.000000 | PASS |
| Category-Specific MLP | < 1e-5 | 1.000000 | PASS |
| Causal Chunked Attention | < 1e-5 | 1.000000 | PASS |
| CausalWanDiT (Action Head) | < 1e-4 | 0.999999 | PASS |
| Flow Matching Scheduler | < 1e-6 | 1.000000 | PASS |
| Full Generate (24L, 16 steps) | < 1e-2 | 0.999990 | PASS |

Reproduce with:
```bash
# Generate PyTorch fixtures
uv run python scripts/pytorch_standalone_forward.py \
    --checkpoint-dir checkpoints/DreamZero-DROID \
    --output pt_outputs.npz

# Run JAX parity check
uv run python scripts/jax_component_parity.py \
    --checkpoint-dir checkpoints/DreamZero-DROID \
    --output jax_outputs.npz

# Compare
uv run python scripts/compare_outputs.py pt_outputs.npz jax_outputs.npz
```

---

## Architecture

DreamZero is a DiT-based diffusion model for joint video and action prediction:

```
Text Instruction --> Text Encoder (T5, 24L)  ----------------+
                                                              |
Multi-view Images --> Image Encoder (CLIP ViT-H/14, 32L) ----+
                                                              +--> DiT Backbone (40L, d=5120) --> Video + Actions
Video Frames --> VAE Encoder --> Latents ---------------------+
                                                              |
Timestep --> Sinusoidal Embed --> Time MLP -------------------+
```

| Component | Parameters | Key Dimensions |
|---|---|---|
| DiT Backbone | ~12B | 40 layers, dim=5120, 40 heads, ffn=13824 |
| Video VAE | ~200M | z_dim=16, 8x spatial compression |
| Text Encoder | ~1B | T5-style, 24 layers, 64 heads |
| Image Encoder | ~600M | CLIP ViT-H/14, 32 layers |
| Action Head | ~200M | CausalWanDiT, flow matching |
| **Total** | **~14B** | |

Inference uses **flow matching** with a shifted sigma schedule (16 denoising steps default). The DiT uses **causal chunked attention** over `[clean_images][noisy_images][actions][states]` tokens, enabling autoregressive video generation with action conditioning.

See [ARCHITECTURE.md](ARCHITECTURE.md) for full component specs, dimension tables, and the PyTorch-to-JAX mapping.

---

## Weight Conversion

Convert a PyTorch checkpoint to Flax format:

```bash
# From a local safetensors checkpoint
uv run python scripts/convert_weights.py \
    --input path/to/pytorch_model.safetensors \
    --output flax_checkpoint

# From HuggingFace
uv run python scripts/convert_weights.py \
    --input dreamzero0/dreamzero-droid \
    --output flax_checkpoint --hf

# With bfloat16 casting (recommended for TPU)
uv run python scripts/convert_weights.py \
    --input path/to/pytorch_model.safetensors \
    --output flax_checkpoint --dtype bfloat16

# Verify numerical parity after conversion
uv run python scripts/convert_weights.py \
    --input path/to/pytorch_model.safetensors \
    --output flax_checkpoint --verify
```

The converter handles all weight transpositions (PyTorch OIHW -> JAX HWIO for convolutions, transposed linear layers) and key remapping automatically. 100% of parameters load with zero missing/extra keys.

---

## Benchmarks

Measured on TPU v5e-8 (8 chips, 128 GB HBM), JAX 0.6.2, bf16, batch=1.

| Component | Latency (ms) | Notes |
|---|---|---|
| Single DiT block (d=5120) | **12.17** | std=0.01ms |
| Full DiT (40 layers) | **~242** | v5e-8, sharded |
| VAE encode (33f @ 320x176) | **167** | std=0.02ms |
| VAE decode (33f @ 320x176) | **351** | std=0.05ms |
| Full inference (16 steps) | **~17s** | 40L, staged, v5e-8 |

Protocol: 3 warmup + 10 timed iterations, `jax.block_until_ready()`.

Reproduce:
```bash
uv run python scripts/benchmark_components.py --component all
```

---

## Project Structure

```
dreamzero-jax/
├── src/dreamzero_jax/
│   ├── nn/                  # Core building blocks
│   │   ├── attention.py     # Causal/chunked attention
│   │   ├── embed.py         # Sinusoidal, RoPE, patch embed
│   │   └── mlp.py           # SwiGLU, GeGLU
│   │
│   ├── models/              # High-level architectures
│   │   ├── dit.py           # DiT blocks + full backbone
│   │   ├── vae.py           # Video VAE encoder/decoder
│   │   ├── text_encoder.py  # T5 wrapper
│   │   ├── image_encoder.py # CLIP ViT-H/14 wrapper
│   │   ├── action_head.py   # Flow matching action head
│   │   ├── dreamzero.py     # Full model assembly
│   │   └── staged_inference.py  # Memory-efficient staged pipeline
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
│   ├── inference.py              # End-to-end inference
│   ├── convert_weights.py        # PyTorch -> Flax conversion
│   ├── convert_checkpoint.py     # Checkpoint conversion (orbax)
│   ├── benchmark_components.py   # Component benchmarks
│   ├── jax_component_parity.py   # JAX-side parity validation
│   ├── compare_outputs.py        # Cross-framework comparison
│   ├── validate_parity.py        # Full parity validation suite
│   └── validate_real_weights.py  # Real-weight inference validation
│
└── tests/                   # 130+ pytest tests
```

---

## Development

```bash
# Install dev dependencies
uv sync --extra dev

# Run tests
uv run pytest tests/ -v

# Run a specific test file
uv run pytest tests/test_dit.py -v

# Lint
uv run ruff check src/ tests/
```

### Code style

- **Flax NNX** (not Linen) for all neural network modules
- `nnx.Module` subclasses with mutable state
- `nnx.Rngs` for PRNG management
- `ruff` for formatting and linting (line length 100)

---

## For AI Agents

If you're an AI agent (Claude Code, Cursor, Copilot), read **[AGENTS.md](AGENTS.md)** instead of this README. It has the codebase map, key patterns, gotchas, and common tasks you need to navigate and contribute.

---

## Validation Note

Parity testing uses **deterministic synthetic inputs** (not real robot data):
- Video: `torch.randn(1, 33, 320, 176, 3) * 0.1` with `torch.manual_seed(42)`
- Tokens: `ones(1, 512)`
- State: `torch.randn(1, 9, 64) * 0.01` with same seed
- Both PyTorch and JAX use identical seeded inputs for exact reproducibility

---

## Roadmap

### Done
- [x] Full 14B JAX/Flax port with 26/26 component parity
- [x] Weight conversion (100% coverage) and inference on TPU v5e
- [x] 130 unit tests, Apache 2.0

### Next
- [ ] **Training consistency** — benchmark training loss curves GPU (PyTorch) vs TPU (JAX) on the same data, verify convergence matches
- [ ] **Latency / throughput / TCO** — head-to-head GPU (H100, A100) vs TPU (v5e, v6e) benchmarks for inference and training, with full cost-of-ownership analysis
- [ ] **Multi-slice training & inference** — lightweight multi-host support for v5e-16+ pods using JAX's distributed runtime
- [ ] **Custom Pallas kernels** — fused attention, AdaLN, and RoPE kernels to further optimize latency, throughput, and memory profile on TPU

---

## References

- [DreamZero Paper](https://dreamzero0.github.io/DreamZero.pdf) — Wu et al., 2025
- [DreamZero GitHub (PyTorch)](https://github.com/dreamzero0/dreamzero)
- [DROID Checkpoint on HuggingFace](https://huggingface.co/GEAR-Dreams/DreamZero-DROID)
- [JAX Documentation](https://jax.readthedocs.io/)
- [Flax NNX Documentation](https://flax.readthedocs.io/en/latest/nnx/)
- [TPU Best Practices](https://cloud.google.com/tpu/docs/best-practices)

## License

Apache License 2.0 -- see [LICENSE](LICENSE).
