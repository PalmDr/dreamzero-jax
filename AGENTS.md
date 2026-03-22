# AGENTS.md — DreamZero-JAX

Context file for AI agents working in this codebase.

## Project Overview

DreamZero-JAX is a JAX/Flax NNX port of [DreamZero](https://github.com/dreamzero0/dreamzero), a 14B-parameter World Action Model that jointly predicts actions and video for zero-shot robotic manipulation. The original PyTorch model runs at ~0.6s on GB200 / ~3s on H100. This port targets TPU inference (v5e pods) with tensor-parallel sharding.

The model uses a Diffusion Transformer (DiT) backbone with flow matching for joint video generation and action prediction, conditioned on text instructions and visual observations.

## Architecture Map

```
DreamZero (dreamzero.py)
├── WanTextEncoder (text_encoder.py)     — T5-style, 24 layers, 4096 dim
├── WanImageEncoder (image_encoder.py)   — CLIP ViT-H/14, 32 layers, 1280 dim
├── WanVideoVAE (vae.py)                 — CausalConv3d encoder/decoder, z_dim=16
├── CausalWanDiT (action_head.py)        — Joint video+action DiT
│   ├── PatchEmbed3D (embed.py)          — 3D conv patchification
│   ├── WanDiTBlock[] (dit.py)           — N transformer blocks with 6-param AdaLN
│   │   ├── Attention (attention.py)     — Self-attn with RoPE, optional chunking
│   │   ├── WanI2VCrossAttention (dit.py)— Dual text+image cross-attn
│   │   └── MLP (mlp.py)                — GELU FFN
│   ├── WanDiTHead (dit.py)             — 2-param modulation + linear output
│   ├── MultiEmbodimentActionEncoder     — Per-embodiment action encoding
│   ├── CategorySpecificMLP              — Per-embodiment state encoding
│   └── CategorySpecificMLP              — Per-embodiment action decoding
└── FlowMatchScheduler (flow_matching.py)— Shifted sigma schedule
```

### File Map

```
src/dreamzero_jax/
├── __init__.py              # nnx.List compat patch for Flax 0.10.x
├── nn/
│   ├── attention.py         # Attention, make_causal_mask, make_causal_chunk_mask, _chunked_attention
│   ├── embed.py             # sinusoidal_embedding, TimestepEmbedding, PatchEmbed3D, WanRoPE3D, apply_rotary_emb
│   ├── mlp.py               # MLP, SwiGLU, GeGLU
│   ├── pallas_ops.py        # fused_adaln_modulate (Pallas kernel for TPU, naive fallback for CPU/GPU)
│   └── compat.py            # nnx.List shim for Flax 0.10.x
├── models/
│   ├── dit.py               # WanDiT, WanDiTBlock, WanDiTHead, WanI2VCrossAttention, MLPProj, unpatchify
│   │                        # Also: _scan_blocks, _remat_blocks (scan/checkpoint iteration strategies)
│   ├── action_head.py       # CausalWanDiT, CategorySpecificLinear/MLP, MultiEmbodimentActionEncoder, make_action_causal_mask
│   ├── dreamzero.py         # DreamZero, DreamZeroConfig, generate/generate_scan/generate_offload
│   ├── staged_inference.py  # generate_staged — two-phase init (encoders then DiT) for tight HBM budgets
│   ├── vae.py               # WanVideoVAE, CausalConv3d, Encoder3d, Decoder3d
│   ├── text_encoder.py      # WanTextEncoder (T5-style)
│   └── image_encoder.py     # WanImageEncoder, VisionTransformer (CLIP ViT-H/14)
├── schedulers/
│   ├── flow_matching.py     # FlowMatchScheduler (plain class, no learnable params)
│   ├── flow_euler.py        # FlowEulerSchedule, euler_step, make_flow_euler_schedule
│   └── unipc.py             # FlowUniPCMultistepScheduler (higher-order ODE solver)
├── data/
│   ├── dataset.py           # LeRobot dataset loading
│   └── transforms.py        # Video/action transforms
└── utils/
    ├── checkpoint.py        # convert_checkpoint, convert_and_apply, load/save_flax_checkpoint
    ├── sharding.py          # create_mesh, shard_params, shard_batch, get_partition_spec, log_sharding_plan
    ├── quantize.py          # QuantizedLinear, quantize_model, estimate_memory_savings (INT8 weight-only)
    ├── validation.py        # compare_arrays, ComponentResult, load/save_fixture, format_report
    ├── hf_download.py       # HuggingFace checkpoint download
    └── parity_runners.py    # PyTorch vs JAX parity comparison runners

scripts/
├── convert_weights.py       # Download + convert PyTorch -> Flax (supports HF, local, --verify)
├── convert_checkpoint.py    # Lower-level checkpoint conversion
├── inference.py             # Full inference pipeline (video + text -> actions)
├── benchmark_components.py  # Per-component TPU benchmarks (block, DiT, VAE, full)
├── train.py                 # Training pipeline
├── validate_parity.py       # Validate JAX output matches PyTorch reference
├── validate_real_weights.py # Validate with real DROID checkpoint weights
├── validate_against_pytorch.py
├── generate_pt_fixtures.py  # Generate PyTorch reference fixtures for parity tests
├── gpu_reference.py         # GPU reference inference
├── debug_per_phase.py       # Debug per-phase outputs
├── compare_outputs.py       # Compare JAX vs PyTorch output arrays
├── jax_component_parity.py  # Per-component parity checks
├── pt_standalone_ops.py     # Standalone PyTorch op reference
├── pytorch_standalone_forward.py
└── definitive.py            # Definitive end-to-end validation

tests/ (140 tests)
├── test_dit.py, test_attention.py, test_embed.py, test_mlp.py     # nn/ and DiT blocks
├── test_vae.py, test_text_encoder.py, test_image_encoder.py       # Encoder/decoder models
├── test_action_head.py, test_dreamzero.py                          # Action head + full model
├── test_schedulers.py                                              # All three schedulers
├── test_checkpoint.py, test_sharding.py                            # Utils
├── test_numerical_parity.py                                        # Cross-framework parity
└── test_transforms.py                                              # Data transforms
```

## Key Patterns

### Module Convention

All neural network modules use **Flax NNX** (not Linen). Pattern:

```python
class MyModule(nnx.Module):
    def __init__(self, dim: int, *, dtype: jnp.dtype = jnp.float32, param_dtype: jnp.dtype = jnp.float32, rngs: nnx.Rngs):
        self.linear = nnx.Linear(dim, dim, dtype=dtype, param_dtype=param_dtype, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.linear(x)
```

Every module takes `dtype`, `param_dtype`, and `rngs` keyword arguments. `dtype` controls compute dtype, `param_dtype` controls storage dtype.

### Adding a New Model Component

1. Create the module in the appropriate directory (`nn/` for building blocks, `models/` for high-level models).
2. Follow the `(dim, ..., *, dtype, param_dtype, rngs)` constructor pattern.
3. Export it from the package `__init__.py`.
4. Add tests in `tests/test_<component>.py` — tests run on CPU with `JAX_PLATFORMS=cpu`.
5. If it has PyTorch weights to convert, add key-mapping rules in `utils/checkpoint.py`.

### Sharding Strategy

Two-axis mesh: `('data', 'model')`. Auto-inferred from device count (up to 8-way model parallelism).

Sharding is **name-based pattern matching** in `utils/sharding.py`:
- `q_proj/k_proj/v_proj/w_up/w_gate/linear1` kernels: shard output dim across `'model'`
- `out_proj/w_down/linear2` kernels: shard input dim across `'model'`
- Norms, biases, embeddings, convolutions, category-specific weights: replicated
- Fallback: 2D matrices with both dims >= 1024 are sharded on the larger dim

Apply sharding after init/checkpoint load:
```python
mesh = create_mesh()  # auto-detects devices
model = shard_params(model, mesh, param_dtype=jnp.bfloat16)
```

### Checkpoint Conversion (PyTorch -> Flax)

Registry-based: transform functions registered per weight-name regex pattern.

Key transforms:
- Dense weights: transpose `(out, in)` -> `(in, out)`
- Conv2d weights: transpose `(out, in, kH, kW)` -> `(kH, kW, in, out)`
- Conv3d weights: transpose `(out, in, kT, kH, kW)` -> `(kT, kH, kW, in, out)`
- Biases, embeddings, norms: identity (no transform needed)

```python
from dreamzero_jax.utils.checkpoint import convert_and_apply
model = DreamZero(config, rngs=nnx.Rngs(0))
convert_and_apply("pytorch_model.pt", model, config)
```

### Block Execution Strategies

The DiT blocks support three execution modes (controlled by `use_scan` and `use_remat` in config):

1. **Default (unrolled)**: Each block executes sequentially. Simplest, highest memory.
2. **`use_remat=True`**: Each block wrapped with `jax.checkpoint(policy=nothing_saveable)`. Recomputes activations. Lower memory.
3. **`use_scan=True`**: All blocks stacked and run via `jax.lax.scan`. XLA compiles one block body and iterates. Lowest graph size. Can combine with `use_remat`.

### Inference Strategies (HBM Management)

From most to least memory-hungry:

1. **`generate()`**: All weights loaded, UniPC multistep scheduler, Python loop over timesteps.
2. **`generate_scan()`**: All weights loaded, Euler scheduler via `jax.lax.scan`. Better XLA compilation.
3. **`generate_offload()`**: Encodes first, then deletes encoder weights before denoising. Destructive.
4. **`generate_staged()`** (staged_inference.py): Encoders and DiT never coexist in HBM. Creates encoders -> encodes -> deletes -> creates DiT -> denoises. Lowest peak memory. Required for 14B on v5e-8.

All methods support `use_cfg=False` to skip the unconditional pass (halves activation memory per step).

## Common Tasks

### "I want to add a new Pallas kernel"

1. Add kernel function and public API in `src/dreamzero_jax/nn/pallas_ops.py`.
2. Follow the existing pattern: Pallas kernel function + naive fallback + public API with `use_pallas` toggle.
3. Pallas kernels tile over `(batch, seq_blocks)` grid. BLOCK_SEQ=128 is MXU-aligned.
4. The kernel computes in f32 internally and writes bf16 output.
5. Handle seq_len padding to nearest BLOCK_SEQ multiple in the public API.
6. Add a `test_correctness()` function comparing Pallas output to naive output.
7. Export from `nn/__init__.py`.
8. Pallas only works on TPU — CPU tests must use the naive fallback (`use_pallas=False`).

### "I want to benchmark on TPU"

Use `scripts/benchmark_components.py`:
```bash
# All components
uv run python scripts/benchmark_components.py --component all

# Single block
uv run python scripts/benchmark_components.py --component single_block --num-layers 8

# CPU smoke test
JAX_PLATFORM_NAME=cpu uv run python scripts/benchmark_components.py --component single_block --dtype f32
```

Protocol: 3 warmup + 10 timed iterations with `jax.block_until_ready()`.

### "I want to convert weights from a new checkpoint"

```bash
# From local file
uv run python scripts/convert_weights.py --input model.safetensors --output flax_ckpt

# From HuggingFace
uv run python scripts/convert_weights.py --input dreamzero0/dreamzero-droid --output flax_ckpt --hf

# With dtype conversion and verification
uv run python scripts/convert_weights.py --input model.safetensors --output flax_ckpt --dtype bfloat16 --verify
```

If the new checkpoint has different key names, extend the key-mapping rules in `utils/checkpoint.py`.

### "I want to run inference"

```bash
uv run python scripts/inference.py \
    --checkpoint /path/to/flax_ckpt \
    --input-video /path/to/video.mp4 \
    --prompt "pick up the red block" \
    --num-steps 16 --cfg-scale 5.0
```

### "I want to validate JAX matches PyTorch"

```bash
# Generate PyTorch reference fixtures
uv run python scripts/generate_pt_fixtures.py

# Run parity validation
uv run python scripts/validate_parity.py
```

## Gotchas and Anti-patterns

### bf16 matmul precision
JAX bf16 matmuls default to reduced precision on TPU. For numerical parity with PyTorch (which uses f32 accumulation), either:
- Compute attention/norms in f32 (the Pallas kernel and RoPE already do this)
- Use `jax.default_matmul_precision('float32')` or set `JAX_DEFAULT_MATMUL_PRECISION=float32`

### CPU init required for large models
The 14B model does not fit in single-device memory. Initialize on CPU first, then shard:
```python
with jax.default_device(jax.devices("cpu")[0]):
    model = DreamZero(config, rngs=nnx.Rngs(0))
mesh = create_mesh()
model = shard_params(model, mesh)
```
For the tightest HBM budgets, use `generate_staged()` which never loads all weights simultaneously.

### Never SCP .venv to TPU
Python virtual environments are not portable across machines. Always `uv sync` on the target TPU VM.

### nnx.List compatibility
Flax 0.10.x does not have `nnx.List`. The package `__init__.py` patches it in automatically via `nn/compat.py`. Always `import dreamzero_jax` before using `nnx.List`.

### RoPE promotes to float32
`apply_rotary_emb` uses complex64 which promotes Q/K to float32. V is explicitly cast to match. When using `jax.lax.scan` over blocks, the carry dtype must be cast back to the input dtype each iteration (already handled in `_scan_blocks`).

### Chunked attention for long sequences
Video sequences can be very long (thousands of tokens). The `Attention` class supports `chunk_size` to split Q into chunks and process against full K/V sequentially via `lax.map`. This avoids materializing the full attention matrix.

### INT8 quantization preserves dtype
`shard_params` skips dtype casting for int8 weights (checks `arr.dtype != jnp.int8`). The sharding patterns in `sharding.py` include `kernel_i8` and `scales` variants for quantized layers.

### Scheduler is a plain class
`FlowMatchScheduler`, `FlowEulerSchedule`, and `FlowUniPCMultistepScheduler` are plain Python classes, not `nnx.Module`. They have no learnable parameters and are not part of the model state.

### Convention: channels-last throughout
All spatial tensors use channels-last layout: `(B, T, H, W, C)` for video, `(B, H, W, C)` for images. This matches JAX/XLA conventions and avoids transposes.

### generate_offload is destructive
`generate_offload()` deletes encoder weights after encoding. The model cannot encode again without re-initialization.

## Testing

```bash
# Run all tests on CPU
JAX_PLATFORMS=cpu uv run pytest tests/ -v

# Run a specific test file
JAX_PLATFORMS=cpu uv run pytest tests/test_dit.py -v

# Run a single test
JAX_PLATFORMS=cpu uv run pytest tests/test_dit.py::test_dit_block_shape -v
```

Tests use small dimensions (dim=192, num_heads=6, num_layers=2) for speed. All 140 tests run on CPU via `JAX_PLATFORMS=cpu`.

Test pattern: each test file creates models with small configs, runs a forward pass, and asserts output shapes and basic numerical properties.

## Configuration

`DreamZeroConfig` defaults match the DROID checkpoint:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `dim` | 1536 | DiT hidden dim |
| `in_channels` | 36 | 16 (VAE latent) + 20 (I2V cond: 4 mask + 16 latent) |
| `out_channels` | 16 | VAE latent channels |
| `ffn_dim` | 8960 | FFN intermediate dim |
| `num_heads` | 12 | Attention heads |
| `num_layers` | 30 | Transformer blocks |
| `patch_size` | (1, 2, 2) | Temporal, height, width patch dims |
| `action_dim` | 32 | Per-step action dimensionality |
| `state_dim` | 64 | Robot state dimensionality |
| `action_hidden_size` | 1024 | Action encoder/decoder hidden dim |
| `num_action_per_block` | 32 | Action tokens per temporal block |
| `max_num_embodiments` | 32 | Multi-embodiment weight banks |
| `text_dim` | 4096 | T5 text encoder output dim |
| `image_dim` | 1280 | CLIP ViT-H/14 output dim |
| `vae_z_dim` | 16 | VAE latent channels |
| `scheduler_shift` | 5.0 | Flow matching sigma shift |
| `num_inference_steps` | 16 | Default denoising steps |
| `cfg_scale` | 5.0 | Classifier-free guidance scale |
| `dtype` / `param_dtype` | float32 | Set to bfloat16 for TPU inference |

## Dependencies

- Python >= 3.11
- `jax >= 0.4.35`, `flax >= 0.10.0`, `optax`, `orbax-checkpoint`
- `transformers` (tokenizer), `einops`, `grain` (data loading)
- Package manager: **uv** (`uv sync` to install, `uv run` to execute)
- Dev: `pytest`, `ruff` (line-length=100, target py311)
