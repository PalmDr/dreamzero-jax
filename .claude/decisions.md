# Decisions Log

Append-only record of architectural and implementation choices.

---

### 001 — Flax NNX over Linen

**Date**: project inception
**Choice**: Use Flax NNX (not Linen) for all modules.
**Reason**: NNX has mutable state, simpler API, and is the future direction of Flax. Better ergonomics for large models with complex state.

---

### 002 — Gated MLP variants as separate classes

**Date**: initial nn/ implementation
**Choice**: Implement `SwiGLU` and `GeGLU` as standalone `nnx.Module` subclasses rather than parameterizing a single MLP class.
**Reason**: Gated variants have a fundamentally different forward pass (split + gate), so separate classes are clearer than flag-based branching.

---

### 003 — Pure functions for positional embeddings

**Date**: initial nn/ implementation
**Choice**: `sinusoidal_embedding`, `get_*d_sincos_pos_embed`, `precompute_freqs_cis`, and `apply_rotary_emb` are pure functions, not modules.
**Reason**: These are stateless computations with no learnable parameters. Keeping them as functions avoids unnecessary module overhead and makes them composable.

---

### 005 — Single Attention class, no nnx.MultiHeadAttention subclass

**Date**: 2026-02-09
**Choice**: Build `Attention` from `nnx.Linear` projections rather than subclassing `nnx.MultiHeadAttention`.
**Reason**: `nnx.MultiHeadAttention` uses `LinearGeneral` with different weight shapes and has 30+ params. Building from `nnx.Linear` is simpler, matches the project's MLP style, and makes separate Q/K/V projections (needed for cross-attention) straightforward.

---

### 006 — Masks as pure functions, not class flags

**Date**: 2026-02-09
**Choice**: `make_causal_mask` and `make_causal_chunk_mask` are standalone functions; the `Attention` class receives masks via its `__call__` signature.
**Reason**: Stateless mask generation is more flexible — callers can compose, cache, or modify masks. Keeps the Attention class generic.

---

### 007 — jax.nn.dot_product_attention with (B, T, N, H) layout

**Date**: 2026-02-09
**Choice**: Use `jax.nn.dot_product_attention` as the core attention primitive with `(B, seq, heads, head_dim)` tensor layout.
**Reason**: This function handles scaling, softmax, and TPU/GPU dispatch automatically. The BTNH layout is its native format (avoids transposes for Q/K/V). Bias uses (B, N, T, S) convention internally.

---

### 008 — WanRoPE3D as plain class, not nnx.Module

**Date**: 2026-02-09
**Choice**: Implement `WanRoPE3D` as a plain Python class with precomputed frequency tables, not an `nnx.Module`.
**Reason**: No learnable parameters. Precomputing per-axis tables up to `max_len` and slicing at call time avoids recomputation. Uses `jnp.broadcast_to` for zero-copy expansion across the 3D grid.

---

### 009 — I2V cross-attention as standalone module

**Date**: 2026-02-09
**Choice**: `WanI2VCrossAttention` is a separate `nnx.Module` with its own Q/K/V projections rather than subclassing `Attention`.
**Reason**: It has a fundamentally different architecture — separate K/V projections for image vs text tokens, two parallel dot-product attentions with summed outputs. Composition over inheritance keeps things clear.

---

### 010 — 6-parameter modulation via nnx.Param with [...] access

**Date**: 2026-02-09
**Choice**: Store the learnable modulation bias as `nnx.Param` and access via `self.modulation[...]` (not `.value`).
**Reason**: `.value` is deprecated in Flax NNX ≥0.12. The `[...]` indexing returns the underlying array while being compatible with the NNX variable tracking system.

---

### 011 — Channels-last layout (B, T, H, W, C) for video

**Date**: 2026-02-09
**Choice**: Use channels-last layout throughout the DiT pipeline.
**Reason**: JAX/TPU-preferred format. Avoids transposes at model boundaries. PatchEmbed3D and unpatchify both work natively in this layout.

---

### 012 — VAE building blocks co-located in models/vae.py

**Date**: 2026-02-09
**Choice**: Put CausalConv3d, ResidualBlock, AttentionBlock, and resample modules directly in `models/vae.py` rather than in `nn/`.
**Reason**: These components are only used by the VAE. Co-locating them avoids polluting the shared `nn/` namespace and keeps the VAE self-contained.

---

### 013 — CausalConv3d wraps nnx.Conv with manual padding

**Date**: 2026-02-09
**Choice**: Implement CausalConv3d as a wrapper that uses `jnp.pad` for asymmetric causal temporal padding then calls `nnx.Conv` with `padding='VALID'`.
**Reason**: JAX/Flax Conv doesn't support asymmetric per-axis padding natively. Manual padding is explicit, simple, and matches the PyTorch reference behavior exactly.

---

### 014 — VAE AttentionBlock reuses nn.Attention with zero-init output

**Date**: 2026-02-09
**Choice**: Reuse the existing `Attention` class with `num_heads=1` for the VAE's spatial attention block, and zero-initialize the output projection.
**Reason**: Avoids duplicating attention logic. Zero-init makes the block start as an identity function, matching the PyTorch reference initialization.

---

### 015 — Temporal upsample via channel-doubling + reshape interleave

**Date**: 2026-02-09
**Choice**: TemporalUpsample uses a CausalConv3d that doubles channels, then reshapes `(B, T, H, W, 2, C)` → `(B, 2*T, H, W, C)` by interleaving.
**Reason**: Matches the PyTorch reference approach. Avoids transposed convolutions which can cause checkerboard artifacts.

---

### 016 — T5 attention without scaling (T5 convention)

**Date**: 2026-02-09
**Choice**: T5Attention uses raw dot-product attention without 1/sqrt(d) scaling.
**Reason**: This matches the T5 paper and PyTorch reference. T5 learned to work without scaling during pretraining. We must match this to load pretrained weights correctly.

---

### 017 — Flax NNX: avoid setting module attributes to None before conditional assignment

**Date**: 2026-02-09
**Choice**: When a module attribute may or may not hold an `nnx.Module`, use `hasattr` checks or a boolean flag instead of initializing to `None`.
**Reason**: Flax NNX treats `None` assignments as "static" attributes. Later assigning an `nnx.Module` (a "data" value) to the same attribute raises `ValueError`. Discovered when `T5SelfAttention.pos_embedding` was set to `None` then conditionally replaced.

---

### 018 — Schedulers as plain Python classes, not nnx.Module

**Date**: 2026-02-09
**Choice**: `FlowMatchScheduler` and `FlowUniPCMultistepScheduler` are plain Python classes holding JAX arrays.
**Reason**: Schedulers have no learnable parameters. They hold computed schedules (sigmas, timesteps) and mutable step-tracking state. Plain classes avoid nnx.Module overhead and match the diffusers convention.

---

### 019 — CLIP ViT with jax.nn.dot_product_attention

**Date**: 2026-02-09
**Choice**: `VitSelfAttention` uses `jax.nn.dot_product_attention` with fused QKV projection split into Q/K/V.
**Reason**: Consistent with the project's attention pattern (decision 007). Fused QKV is slightly more efficient than separate projections.

---

### 020 — CategorySpecificLinear via index gathering + einsum

**Date**: 2026-02-09
**Choice**: Implement multi-embodiment linear layers by storing `(num_categories, in, out)` weight tensors and gathering per-sample via `W[category_ids]`, then batch-matmul via `jnp.einsum`.
**Reason**: Simple, correct, and avoids custom CUDA/Pallas kernels. The gather + einsum pattern works well with JAX's XLA compiler. Number of embodiments is small (≤32) so memory overhead is minimal.

---

### 021 — Block-causal mask via broadcast type/block-id arrays

**Date**: 2026-02-09
**Choice**: Construct the action training mask using vectorized JAX operations: assign each token a `(block_id, type_id)` tuple, then build the mask via broadcasting comparisons and `jnp.where`.
**Reason**: Avoids Python loops over sequence positions (slow for large sequences). The mask is O(seq²) memory which is fine for tests; production use would need block-sparse attention (Pallas).

---

### 022 — CausalWanDiT reuses WanDiTBlock with added mask parameter

**Date**: 2026-02-09
**Choice**: Added an optional `mask` parameter to `WanDiTBlock.__call__` (backward-compatible default `None`) and built `CausalWanDiT` around the same blocks rather than creating new block classes.
**Reason**: Maximizes code reuse between standard WanDiT and CausalWanDiT. The only structural difference is the self-attention mask and the interleaved sequence layout — the block computations are identical.

---

### 023 — Mixed 3D/1D RoPE for video vs action tokens

**Date**: 2026-02-09
**Choice**: Video tokens use WanRoPE3D (frame/height/width split), action/state tokens use standard 1D RoPE indexed by block number. Combined into a single frequency array for the interleaved sequence.
**Reason**: Matches the PyTorch reference. The model learns to handle mixed positional encodings through training. 1D RoPE for actions is appropriate since they lack spatial structure.

---

### 024 — Training-only CausalWanDiT, defer KV cache and LoRA

**Date**: 2026-02-09
**Choice**: V1 implements only the training forward pass. KV caching for autoregressive inference, LoRA fine-tuning, and classifier-free guidance are deferred.
**Reason**: These are optimization/inference features that add significant complexity. Getting the core architecture correct first allows validation via checkpoint loading and loss computation.

---

### 025 — DreamZeroConfig as flat dataclass

**Date**: 2026-02-09
**Choice**: Use a single flat `@dataclass` with all hyperparameters (DiT, text encoder, image encoder, VAE, action, scheduler) rather than nested sub-configs.
**Reason**: Simpler construction for tests and configs. The PyTorch reference uses a flat argparse namespace. Sub-configs add indirection without meaningful benefit since the model is always constructed as a whole.

---

### 026 — Basic inference without KV cache

**Date**: 2026-02-09
**Choice**: `DreamZero.generate()` runs the full DiT forward pass per denoising step (no KV caching). Uses two forward passes per step for classifier-free guidance (conditional + unconditional with zero text embedding).
**Reason**: Correctness-first approach. KV-cached autoregressive generation is a significant optimization that can be added once the basic inference path is validated against the PyTorch reference.

---

### 027 — Coupled noise schedule for video and actions

**Date**: 2026-02-09
**Choice**: Training uses the same timestep for both video and action noise injection (same sigma schedule). Actions use a separate noise sample but the same scheduler timestep.
**Reason**: Matches the PyTorch reference. Joint denoising with coupled schedules is fundamental to the DreamZero approach — the model learns to jointly predict video and action velocity fields at the same noise level.

---

### 028 — Name-based sharding with shape heuristic fallback

**Date**: 2026-02-09
**Choice**: Use regex pattern matching on parameter paths (e.g., `.q_proj.kernel`, `.w_down.kernel`) to assign partition specs, with a shape-based heuristic fallback for unrecognized 2D matrices >= 1024x1024.
**Reason**: Name-based rules are explicit and auditable — you can see exactly why each parameter is sharded. Shape-only heuristics are fragile (could shard the wrong dimension). The fallback catches any large matrices missed by name patterns.

---

### 029 — Replicate small/specialized weights (CategorySpecific, action/state encoders)

**Date**: 2026-02-09
**Choice**: Always replicate multi-embodiment weights (CategorySpecificLinear), action encoder, state encoder, and action decoder parameters.
**Reason**: These are small (32 x dim x dim at most) and have 3D shapes (num_categories, in, out) that don't map cleanly to 2D tensor-parallel sharding. Replication avoids complex gather/scatter patterns for negligible memory savings.

---

### 030 — Mesh shape heuristic: up to 8-way model parallelism

**Date**: 2026-02-09
**Choice**: Auto-inferred mesh shape caps model parallelism at 8 devices, scaling data parallelism beyond that.
**Reason**: 8-way tensor parallelism is the standard for large transformer models (all-reduce costs grow with degree). For 14B params with dim=5120 and 40 heads, 8-way gives 5 heads per shard which divides evenly. Beyond 8-way, communication overhead dominates.

---

### 031 — HuggingFace datasets directly, no lerobot dependency

**Date**: 2026-02-09
**Choice**: `LeRobotDataset` loads data via `datasets.load_dataset` directly rather than depending on the `lerobot` Python package. Column names are configurable via `LeRobotDatasetConfig`.
**Reason**: Avoids a heavy optional dependency. LeRobot datasets are standard HuggingFace datasets under the hood — we only need the column naming convention, which is configurable. The `lerobot` package adds download utilities and policy wrappers we don't need.

---

### 032 — Pure-Python data iteration, not grain

**Date**: 2026-02-09
**Choice**: Use simple Python iteration with `random.shuffle` for training data loading rather than `grain` or `tf.data`.
**Reason**: Keeps the data pipeline dependency-light and debuggable. `grain` is powerful for distributed/multiprocess loading but adds complexity we don't need yet. The pipeline can be swapped to `grain` later when profiling reveals data loading as a bottleneck.

---

### 033 — Min/max action normalization to [-1, 1]

**Date**: 2026-02-09
**Choice**: Normalize actions to `[-1, 1]` using per-dimension min/max statistics rather than mean/std z-score normalization.
**Reason**: The DreamZero model uses flow matching where the action space is bounded. Min/max normalization maps to a known range that matches the model's output activation. The `ActionStats` dataclass also stores mean/std for future use (e.g., z-score normalization as an alternative).

---

### 034 — Registry-based key mapping for checkpoint conversion

**Date**: 2026-02-09
**Choice**: Use a `KeyMappingBuilder` with ordered regex rules rather than a static dictionary mapping for PyTorch -> Flax key conversion.
**Reason**: A static mapping would require enumerating every possible key (thousands for 30+ blocks). Regex rules compose naturally with helper functions (`_add_linear_rules`, `_add_dit_block_rules`, etc.) and handle variable block indices. First-match-wins semantics let more specific patterns override general ones.

---

### 035 — CategorySpecificLinear weights: no transpose

**Date**: 2026-02-09
**Choice**: Copy CategorySpecificLinear/MLP weights without transposing, unlike standard Linear layers.
**Reason**: The Flax implementation uses `jnp.einsum("bli,bio->blo", x, W)` with weight shape `(num_cats, in, out)` -- matching PyTorch's stored shape. No transpose is needed since the forward pass is einsum-based rather than matmul-based.

---

### 036 — Multiple naming conventions for VAE/ViT checkpoint keys

**Date**: 2026-02-09
**Choice**: Register alternative PyTorch naming patterns (e.g., `conv_in`/`stem`, `ln_pre`/`pre_norm`, `resblocks`/`transformer`, `in_proj_weight`/`to_qkv.weight`) alongside the primary names.
**Reason**: Different PyTorch DreamZero/WAN checkpoints may use different naming conventions depending on the codebase version. Supporting multiple conventions makes the converter robust without requiring users to know which convention their checkpoint uses.

---

### 037 — load_pytorch_checkpoint: safetensors > torch > numpy fallback

**Date**: 2026-02-09
**Choice**: Try safetensors first (no torch dependency), then torch.load, then numpy .npz.
**Reason**: safetensors is the preferred modern format (fast, safe, no pickle). torch.load is the most common legacy format. The fallback chain lets users avoid installing torch entirely if their checkpoint is in safetensors format.

---

### 004 — .claude/ folder for progress tracking

**Date**: 2026-02-09
**Choice**: Use a `.claude/` directory with `progress.md`, `decisions.md`, and `context/` for per-component notes.
**Reason**: Provides persistent cross-session context for Claude Code without cluttering CLAUDE.md. Avoids stale session logs; keeps focused, structured reference material.
