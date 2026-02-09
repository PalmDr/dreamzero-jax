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

### 004 — .claude/ folder for progress tracking

**Date**: 2026-02-09
**Choice**: Use a `.claude/` directory with `progress.md`, `decisions.md`, and `context/` for per-component notes.
**Reason**: Provides persistent cross-session context for Claude Code without cluttering CLAUDE.md. Avoids stale session logs; keeps focused, structured reference material.
