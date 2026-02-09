# DreamZero Architecture Reference

This document summarizes the original PyTorch DreamZero architecture for porting to JAX/Flax NNX.

## Overview

- **Size**: 14B parameters
- **Inference**: ~0.6s on GB200, ~3s on H100 (with DiT caching)
- **Architecture**: DiT (Diffusion Transformer) for joint video + action prediction

## Data Flow

```
Text Instruction ──► Text Encoder (T5) ──────────────────────────┐
                                                                  │
Multi-view Images ──► Image Encoder (CLIP ViT-H/14) ─────────────┤
                                                                  ├──► DiT Backbone ──► Video + Action
Video Frames ──► VAE Encoder ──► Latents ────────────────────────┤
                                                                  │
Timestep ──► Sinusoidal Embed ──► Time MLP ──────────────────────┘
```

## Component Details

### 1. DiT Block (Diffusion Transformer)

**File**: `modules/wan_video_dit.py`

```
Input
  │
  ├──► LayerNorm ──► Self-Attention ──► Gate ──► (+) ───┐
  │                                                      │
  ├──► Cross-Attention (text/image conditioning) ──────(+)
  │                                                      │
  └──► LayerNorm ──► FFN ──► Gate ──► (+) ──────────────┘
                                                         │
                                              Modulated by Time Embed
```

**Key params**:
| Param | Small | Large |
|-------|-------|-------|
| dim | 1536 | 5120 |
| num_heads | 12 | 40 |
| ffn_dim | 8960 | 13824 |
| num_layers | 32 | 32 |

**Attention**: Flash Attention with 3D RoPE (rotary position embeddings)

**FFN**: `Linear(dim, ffn_dim) → GELU(approximate='tanh') → Linear(ffn_dim, dim)`

**Time modulation**: 6 parameters per block (shift/scale for attn input, attn output, ffn)
```python
self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
```

### 2. Causal Chunked Attention

**File**: `modules/wan_video_dit_action_casual_chunk.py`

For memory-efficient video generation with action conditioning:

- `frame_seqlen`: Tokens per frame (220 or 880)
- `num_frame_per_block`: Frames grouped per attention block
- Block-wise causal masking for autoregressive generation

**Sequence layout**: `[clean_images][noisy_images][actions][states]`

### 3. Video VAE

**File**: `modules/wan_video_vae.py`

| Param | WanVideoVAE | WanVideoVAE38 |
|-------|-------------|---------------|
| z_dim (latent) | 16 | 48 |
| base_dim | 96 | 160/256 |
| dim_multipliers | [1,2,4,4] | [1,2,4,4] |
| upsampling | 8x | 16x |

**Encoder**: CausalConv3d → ResBlocks → Downsample → Middle Attn → Latent proj

**Decoder**: Mirror of encoder with upsampling

**Key modules**:
- `CausalConv3d`: Preserves temporal causality
- `RMS_norm`: RMSNorm with learnable scale
- `AttentionBlock`: Single-head self-attention

### 4. Text Encoder (T5-style)

**File**: `modules/wan_video_text_encoder.py`

| Param | Value |
|-------|-------|
| vocab_size | 256,384 |
| num_layers | 24 |
| num_heads | 64 |
| activation | GELU |

Standard T5 transformer with relative position embeddings.

### 5. Image Encoder (CLIP ViT-H/14)

**File**: `modules/wan_video_image_encoder.py`

| Param | Value |
|-------|-------|
| patch_size | 14 |
| embedding_dim | 1024 |
| num_layers | 32 (uses 31) |
| num_heads | 16 |
| mlp_ratio | 4x |

**Normalization**:
- Mean: [0.48145466, 0.4578275, 0.40821073]
- Std: [0.26862954, 0.26130258, 0.27577711]

### 6. Action Head

**File**: `action_head/wan_flow_matching_action_tf_efficient.py`

**Class**: `WANPolicyHead`

| Param | Value |
|-------|-------|
| input_embedding_dim | 1536 |
| hidden_size | 1024 |
| action_dim | per-embodiment |
| max_embodiments | 32 |

- Flow matching for action prediction
- LoRA fine-tuning (rank-4 on q, k, v, o, ffn)
- Beta distribution noise sampling

### 7. Flow Matching Scheduler

**File**: `modules/flow_match_scheduler.py`

| Param | Default |
|-------|---------|
| num_inference_steps | 100 |
| num_train_timesteps | 1000 |
| shift | 3.0 |

**Sigma schedule**:
```python
sigma_shifted = (shift * sigma) / (1 + (shift - 1) * sigma)
```

**Denoising step**:
```python
prev_sample = sample + model_output * (sigma_next - sigma_current)
```

### 8. UniPC Multistep Scheduler

**File**: `modules/flow_unipc_multistep_scheduler.py`

Higher-order ODE solver for faster inference:
- `solver_order`: 2 (default)
- `solver_type`: "bh2"
- Predictor-corrector algorithm

## Attention Backend Priority

1. Flash Attention 3
2. Flash Attention 2
3. SageAttn
4. `torch.nn.functional.scaled_dot_product_attention`

## JAX/Flax Mapping

| PyTorch | JAX/Flax NNX |
|---------|--------------|
| `nn.Linear` | `nnx.Linear` |
| `nn.LayerNorm` | `nnx.LayerNorm` |
| `nn.RMSNorm` | `nnx.RMSNorm` |
| `nn.GELU(approximate='tanh')` | `jax.nn.gelu` (approximate=True is default) |
| `nn.SiLU` | `jax.nn.silu` |
| `nn.Embedding` | `nnx.Embed` |
| `nn.Conv2d` | `nnx.Conv` |
| `nn.MultiheadAttention` | `nnx.MultiHeadAttention` |
| Flash Attention | `jax.nn.dot_product_attention` |

## Implementation Priority

1. **nn/** - Building blocks (MLP ✓, embed, attention)
2. **models/dit.py** - DiT blocks and backbone
3. **models/vae.py** - Video VAE
4. **models/text_encoder.py** - T5 wrapper
5. **models/image_encoder.py** - CLIP wrapper
6. **models/action_head.py** - Flow matching action head
7. **schedulers/** - Flow matching, UniPC
8. **models/dreamzero.py** - Full model assembly
