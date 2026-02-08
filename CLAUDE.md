# DreamZero-JAX

JAX/Flax NNX implementation of [DreamZero](https://github.com/dreamzero0/dreamzero) for TPU/GPU inference and training.

## Code Style

- **Flax NNX** (not Linen) for all neural network modules
- Prefer `nnx.Module` subclasses with mutable state
- Use `nnx.Rngs` for PRNG management

## Project Overview

DreamZero is a **World Action Model** that jointly predicts actions and videos, achieving zero-shot performance on unseen robotics tasks. This project ports the original PyTorch implementation to JAX for TPU acceleration.

### Original Model Specs
- **Architecture**: DiT (Diffusion Transformer) based video generation + action prediction
- **Size**: 14B parameters (DROID checkpoint)
- **Inference**: ~0.6s on GB200, ~3s on H100 with DiT caching

## Architecture Components to Port

### Core Modules (`groot/vla/model/dreamzero/`)

1. **WAN Video DiT** (`modules/wan_video_dit.py`, `wan_video_dit_action_casual_chunk.py`)
   - Main diffusion transformer backbone
   - Causal chunked attention for video generation

2. **VAE** (`modules/wan_video_vae.py`)
   - Video encoder/decoder for latent space

3. **Text Encoder** (`modules/wan_video_text_encoder.py`)
   - Language conditioning for task instructions

4. **Image Encoder** (`modules/wan_video_image_encoder.py`)
   - Visual observation encoding

5. **Attention Mechanisms** (`modules/attention.py`, `wan2_1_attention.py`)
   - Custom attention implementations
   - Need TPU-optimized alternatives (Pallas kernels or `jax.nn.dot_product_attention`)

6. **Action Head** (`action_head/wan_flow_matching_action_tf_efficient.py`)
   - Flow matching for action prediction
   - Transformer-based action decoder

7. **Schedulers** (`modules/flow_match_scheduler.py`, `flow_unipc_multistep_scheduler.py`)
   - Diffusion sampling schedules

## Directory Structure

```
dreamzero-jax/
├── CLAUDE.md                 # This file
├── dreamzero_jax/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── dit.py            # DiT backbone in Flax
│   │   ├── vae.py            # Video VAE
│   │   ├── text_encoder.py   # Text encoder
│   │   ├── image_encoder.py  # Image encoder
│   │   ├── action_head.py    # Flow matching action head
│   │   └── attention.py      # TPU-optimized attention
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── layers.py         # Common layers (norms, embeddings)
│   │   └── schedulers.py     # Flow matching schedulers
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py        # LeRobot dataset loader
│   │   └── transforms.py     # Data transforms
│   └── utils/
│       ├── __init__.py
│       ├── checkpoint.py     # PyTorch -> Flax weight conversion
│       └── sharding.py       # TPU mesh sharding utilities
├── configs/
│   └── base.yaml             # Model configs
├── scripts/
│   ├── convert_checkpoint.py # Weight conversion script
│   ├── inference.py          # TPU inference
│   └── train.py              # TPU training
└── tests/
    └── test_models.py        # Model unit tests
```

## JAX/TPU Considerations

### Sharding Strategy
- **Tensor Parallelism**: Shard attention heads and FFN across TPU cores
- **Pipeline Parallelism**: For large models, partition layers across pods
- **Data Parallelism**: Replicate model, shard batch across hosts
- Use `jax.sharding.NamedSharding` with mesh axis names: `('data', 'model')`

### Attention Implementation
- Use `jax.nn.dot_product_attention` with `implementation='cudnn'` for GPU fallback
- For TPU: standard JAX attention or Pallas kernels for custom patterns
- Consider Flash Attention patterns for memory efficiency

### Checkpoint Conversion
- Map PyTorch `state_dict` keys to Flax `FrozenDict` structure
- Handle shape transpositions (PyTorch: `[out, in]` → Flax: `[in, out]` for Dense)
- Use `orbax` for checkpoint management

### Mixed Precision
- Use `jax.dtypes.bfloat16` for TPU-native format
- Keep master weights in float32 for training stability

## Key Dependencies

```
jax[tpu]
flax
optax
orbax-checkpoint
grain              # For data loading
transformers       # For tokenizer
```

## Package Manager

This project uses **uv** for fast, reliable Python package management.

## Development Workflow

1. **Phase 1**: Core model architecture
   - Port DiT, VAE, encoders to Flax
   - Implement attention with TPU optimization
   - Unit test each component

2. **Phase 2**: Inference pipeline
   - Checkpoint conversion from PyTorch
   - Distributed inference on TPU pod
   - Validate outputs match original

3. **Phase 3**: Training pipeline
   - Data loading with grain/tf.data
   - Distributed training with pjit
   - Gradient checkpointing for memory

4. **Phase 4**: Optimization
   - Profile and optimize bottlenecks
   - Pallas kernels for custom ops if needed
   - DiT caching implementation

## Commands

```bash
# Create environment and install dependencies
uv sync

# Convert checkpoint
uv run python scripts/convert_checkpoint.py --input <pytorch_ckpt> --output <flax_ckpt>

# Run inference
uv run python scripts/inference.py --checkpoint <flax_ckpt> --input <observation>

# Training
uv run python scripts/train.py --config configs/base.yaml
```

## References

- [DreamZero Paper](https://dreamzero0.github.io/DreamZero.pdf)
- [DreamZero GitHub](https://github.com/dreamzero0/dreamzero)
- [JAX Documentation](https://jax.readthedocs.io/)
- [Flax NNX Documentation](https://flax.readthedocs.io/en/latest/nnx/)
- [TPU Best Practices](https://cloud.google.com/tpu/docs/best-practices)
