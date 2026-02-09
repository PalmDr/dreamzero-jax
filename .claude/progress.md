# Progress Tracker

## Phase 1: Core Model Architecture

### nn/ — Building Blocks

| Component | File | Status | Notes |
|-----------|------|--------|-------|
| MLP, SwiGLU, GeGLU | `nn/mlp.py` | done | Tests in `tests/test_mlp.py` |
| Sinusoidal embed, PatchEmbed, PatchEmbed3D, RoPE | `nn/embed.py` | done | Tests in `tests/test_embed.py` |
| Causal/chunked attention | `nn/attention.py` | done | Tests in `tests/test_attention.py`. Single `Attention` class for self/cross-attn, `make_causal_mask`, `make_causal_chunk_mask`. Now supports `qk_norm`. |
| WanRoPE3D | `nn/embed.py` | done | WAN-style 3D RoPE with dim split: dim_f = d - 2*(d//3), dim_h = d//3, dim_w = d//3. Plain class (not nnx.Module). |

### models/ — High-Level Architectures

| Component | File | Status | Notes |
|-----------|------|--------|-------|
| DiT blocks & backbone | `models/dit.py` | done | Tests in `tests/test_dit.py` (12 tests). WanDiTBlock, WanDiTHead, WanDiT, WanI2VCrossAttention, MLPProj, unpatchify. |
| Video VAE | `models/vae.py` | done | Tests in `tests/test_vae.py` (12 tests). CausalConv3d, ResidualBlock, AttentionBlock, Spatial/TemporalDown/Upsample, Encoder3d, Decoder3d, WanVideoVAE. Non-chunked/non-tiled v1. |
| Text encoder (T5) | `models/text_encoder.py` | done | Tests in `tests/test_text_encoder.py` (9 tests). T5RelativeEmbedding, T5Attention, T5FeedForward, T5SelfAttention, WanTextEncoder. |
| Image encoder (CLIP) | `models/image_encoder.py` | done | Tests in `tests/test_image_encoder.py` (8 tests). QuickGELU, VitSelfAttention, VitAttentionBlock, VisionTransformer, WanImageEncoder. |
| Action head | `models/action_head.py` | done | Tests in `tests/test_action_head.py` (12 tests). CategorySpecificLinear, CategorySpecificMLP, MultiEmbodimentActionEncoder, make_action_causal_mask, CausalWanDiT. Training forward only; KV cache/LoRA/inference deferred. |
| Full model assembly | `models/dreamzero.py` | done | Tests in `tests/test_dreamzero.py` (6 tests). DreamZeroConfig, DreamZero, TrainOutput, InferenceOutput. Training forward + basic inference with UniPC + CFG. 103 total tests passing. |

### schedulers/

| Component | File | Status | Notes |
|-----------|------|--------|-------|
| Flow matching | `schedulers/flow_matching.py` | done | Tests in `tests/test_schedulers.py` (7 tests). FlowMatchScheduler with shifted sigma schedule, training weights. |
| UniPC multistep | `schedulers/unipc.py` | done | Tests in `tests/test_schedulers.py` (6 tests). FlowUniPCMultistepScheduler with predictor-corrector, bh2 solver. |

## Phase 2: Inference Pipeline

| Component | File | Status | Notes |
|-----------|------|--------|-------|
| Checkpoint conversion | `utils/checkpoint.py` | done | Registry-based PyTorch->Flax key mapping with auto transposition for Dense/Conv/Embed/Norm. Handles all sub-models (DiT, VAE, T5 text enc, CLIP image enc, action head). orbax save/load. Diagnostic tools (shape comparison, dry-run). |
| Sharding utilities | `utils/sharding.py` | done | Mesh creation, name-based partition specs, model/batch sharding, training pspecs, sharding constraints, debug logging. |
| Inference script | `scripts/inference.py` | done | CLI with argparse, video I/O (mp4/frames), T5 tokenization, orbax checkpoint loading, JIT-compiled generate, VAE decode, saves actions (.npy/.csv) and video frames (.png). |
| Checkpoint script | `scripts/convert_checkpoint.py` | done | CLI with --input, --output, --config, --strict, --dry-run, --compare-shapes, --print-mapping. Handles DDP prefix stripping, safetensors/torch loading. |

## Phase 3: Training Pipeline

| Component | File | Status | Notes |
|-----------|------|--------|-------|
| Dataset loader | `data/dataset.py` | done | LeRobotDataset (HuggingFace datasets), LeRobotDatasetConfig, collate_fn, create_train_dataloader, create_eval_dataloader. No lerobot dependency. |
| Data transforms | `data/transforms.py` | done | ActionStats, DataConfig, normalize/denormalize video & actions, resize, center/random crop, prepare_batch. |
| Training script | `scripts/train.py` | done | Full distributed training with AdamW + warmup cosine schedule, gradient accumulation, orbax checkpointing, wandb integration, dummy data fallback. ~748 lines. |

## Numerical Validation

| Component | File | Status | Notes |
|-----------|------|--------|-------|
| Validation utilities | `utils/validation.py` | done | ComponentResult, compare_arrays, load/save_fixture, format_report, DEFAULT_TOLERANCES for 10 components. |
| PT fixture generator | `scripts/generate_pt_fixtures.py` | done | CLI to generate .npz reference outputs from PyTorch models. Supports --small, --components, --seed. 10 component generators. |
| JAX validator script | `scripts/validate_against_pytorch.py` | done | CLI to compare JAX outputs against fixtures. Text/JSON output, --strict mode, per-component tolerance overrides. |
| Parity tests | `tests/test_numerical_parity.py` | done | 10 pytest tests (one per component), graceful skip when no fixtures. Uses module-level skipif on manifest.json. |

## Phase 4: Optimization

| Component | Status | Notes |
|-----------|--------|-------|
| Profiling | not started | |
| Pallas kernels | not started | If needed |
| DiT caching | not started | |

## Test Summary

**128 tests passing** across 11 test files + parity tests (skipped without fixtures):
- Phase 1 core: test_mlp, test_embed, test_attention, test_dit, test_vae, test_text_encoder, test_image_encoder, test_action_head, test_dreamzero, test_schedulers (103 tests)
- Phase 2/3: test_checkpoint (12 tests), test_sharding (7 tests), test_transforms (7 tests)
- Validation: test_numerical_parity (10 tests, skip when no fixtures)

## Next Up

- **Phase 1 complete!** All core model components implemented and tested.
- **Phase 2 complete!** Checkpoint conversion, sharding, and inference all done. Needs validation with real checkpoints.
- **Phase 3 complete!** Data loading, transforms, and training script done.
- Phase 4: Optimization (profiling, Pallas kernels, DiT caching) not yet started.
