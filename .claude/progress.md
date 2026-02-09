# Progress Tracker

## Phase 1: Core Model Architecture

### nn/ — Building Blocks

| Component | File | Status | Notes |
|-----------|------|--------|-------|
| MLP, SwiGLU, GeGLU | `nn/mlp.py` | done | Tests in `tests/test_mlp.py` |
| Sinusoidal embed, PatchEmbed, PatchEmbed3D, RoPE | `nn/embed.py` | done | Tests in `tests/test_embed.py` |
| Causal/chunked attention | `nn/attention.py` | not started | Prerequisite for DiT and action head |

### models/ — High-Level Architectures

| Component | File | Status | Notes |
|-----------|------|--------|-------|
| DiT blocks & backbone | `models/dit.py` | not started | Depends on attention, MLP, embeddings |
| Video VAE | `models/vae.py` | not started | |
| Text encoder (T5) | `models/text_encoder.py` | not started | |
| Image encoder (CLIP) | `models/image_encoder.py` | not started | |
| Action head | `models/action_head.py` | not started | Depends on attention, flow matching scheduler |
| Full model assembly | `models/dreamzero.py` | not started | Depends on all above |

### schedulers/

| Component | File | Status | Notes |
|-----------|------|--------|-------|
| Flow matching | `schedulers/flow_matching.py` | not started | |
| UniPC multistep | `schedulers/unipc.py` | not started | |

## Phase 2: Inference Pipeline

| Component | File | Status | Notes |
|-----------|------|--------|-------|
| Checkpoint conversion | `utils/checkpoint.py` | not started | PyTorch -> Flax weight mapping |
| Sharding utilities | `utils/sharding.py` | not started | TPU mesh setup |
| Inference script | `scripts/inference.py` | not started | |
| Checkpoint script | `scripts/convert_checkpoint.py` | not started | |

## Phase 3: Training Pipeline

| Component | File | Status | Notes |
|-----------|------|--------|-------|
| Dataset loader | `data/dataset.py` | not started | LeRobot format |
| Data transforms | `data/transforms.py` | not started | |
| Training script | `scripts/train.py` | not started | |

## Phase 4: Optimization

| Component | Status | Notes |
|-----------|--------|-------|
| Profiling | not started | |
| Pallas kernels | not started | If needed |
| DiT caching | not started | |

## Next Up

- `nn/attention.py` — last building block before model-level modules
