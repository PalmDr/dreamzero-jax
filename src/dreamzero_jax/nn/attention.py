"""Attention mechanisms for DreamZero."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.lax as lax
from flax import nnx

from dreamzero_jax.nn.embed import apply_rotary_emb


def _chunked_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    bias: jax.Array | None,
    chunk_size: int,
) -> jax.Array:
    """Process attention in chunks along the query dimension.

    Avoids materializing the full (q_len x kv_len) attention matrix by
    splitting Q into chunks and running ``jax.nn.dot_product_attention``
    independently for each chunk against the full K/V.

    Args:
        q: Query tensor ``(B, q_len, num_heads, head_dim)``.
        k: Key tensor ``(B, kv_len, num_heads, head_dim)``.
        v: Value tensor ``(B, kv_len, num_heads, head_dim)``.
        bias: Optional additive attention bias ``(1|B, 1|N, q_len, kv_len)``.
              Each chunk will receive its corresponding slice along the query dim.
              If the query dim of the bias is 1 (broadcast), it is used as-is.
        chunk_size: Number of query tokens per chunk.

    Returns:
        Output tensor ``(B, q_len, num_heads, head_dim)``.
    """
    B, q_len, num_heads, head_dim = q.shape

    # Pad q_len to be divisible by chunk_size
    pad_len = (chunk_size - q_len % chunk_size) % chunk_size
    if pad_len > 0:
        q = jnp.pad(q, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
        if bias is not None:
            # bias shape: (1|B, 1|N, q_len, kv_len)
            # Only pad query dim if it isn't broadcast (size 1)
            if bias.shape[-2] > 1:
                bias = jnp.pad(
                    bias,
                    ((0, 0), (0, 0), (0, pad_len), (0, 0)),
                    constant_values=jnp.finfo(q.dtype).min,
                )

    padded_q_len = q_len + pad_len
    num_chunks = padded_q_len // chunk_size

    # Reshape Q into chunks: (B, num_chunks, chunk_size, num_heads, head_dim)
    q_chunked = q.reshape(B, num_chunks, chunk_size, num_heads, head_dim)

    # Prepare per-chunk bias slices (or None)
    if bias is not None and bias.shape[-2] > 1:
        # bias: (1|B, 1|N, padded_q_len, kv_len)
        # -> (1|B, 1|N, num_chunks, chunk_size, kv_len)
        bias_chunked = bias.reshape(
            bias.shape[0], bias.shape[1], num_chunks, chunk_size, bias.shape[-1]
        )
    else:
        bias_chunked = None

    def _attend_one_chunk(i: int) -> jax.Array:
        q_chunk = q_chunked[:, i]  # (B, chunk_size, num_heads, head_dim)
        if bias_chunked is not None:
            b_chunk = bias_chunked[:, :, i]  # (1|B, 1|N, chunk_size, kv_len)
        elif bias is not None:
            # Broadcast bias (query dim == 1), use as-is
            b_chunk = bias
        else:
            b_chunk = None
        return jax.nn.dot_product_attention(q_chunk, k, v, bias=b_chunk)

    # Use lax.map to process chunks sequentially (avoids materializing all at once)
    out_chunks = lax.map(
        _attend_one_chunk,
        jnp.arange(num_chunks),
    )  # (num_chunks, B, chunk_size, num_heads, head_dim)

    # Reassemble: (num_chunks, B, chunk_size, N, D) -> (B, padded_q_len, N, D)
    out = out_chunks.transpose(1, 0, 2, 3, 4).reshape(
        B, padded_q_len, num_heads, head_dim
    )

    # Remove padding
    if pad_len > 0:
        out = out[:, :q_len]
    return out


def make_causal_mask(seq_len: int, dtype: jnp.dtype = jnp.bool_) -> jax.Array:
    """Create a causal (lower-triangular) attention mask.

    Args:
        seq_len: Sequence length.
        dtype: Output dtype.

    Returns:
        Boolean mask of shape (seq_len, seq_len) where True means "attend".
    """
    return jnp.tril(jnp.ones((seq_len, seq_len), dtype=dtype))


def make_causal_chunk_mask(
    seq_len: int,
    frame_seqlen: int,
    num_frames_per_block: int = 1,
    dtype: jnp.dtype = jnp.bool_,
) -> jax.Array:
    """Create a block-causal mask for chunked attention.

    Tokens are grouped into blocks of ``num_frames_per_block * frame_seqlen``
    tokens.  Within a block, all tokens attend to each other; across blocks,
    only causal (earlier-to-later) attention is allowed.

    Args:
        seq_len: Total sequence length (must be divisible by frame_seqlen).
        frame_seqlen: Number of tokens per frame.
        num_frames_per_block: How many frames form one causal block.
        dtype: Output dtype.

    Returns:
        Boolean mask of shape (seq_len, seq_len) where True means "attend".
    """
    block_size = frame_seqlen * num_frames_per_block
    # Assign each position to its block index
    positions = jnp.arange(seq_len)
    block_ids = positions // block_size  # (seq_len,)
    # q attends to k iff block_ids[q] >= block_ids[k]
    mask = block_ids[:, None] >= block_ids[None, :]  # (seq_len, seq_len)
    return mask.astype(dtype)


class Attention(nnx.Module):
    """Multi-head attention supporting both self- and cross-attention.

    When ``context_dim`` is provided, K and V are projected from a context
    tensor of that dimension (cross-attention).  Otherwise Q, K, V all come
    from the same input (self-attention).

    Masks and rotary embeddings are supplied by the caller, keeping this class
    stateless with respect to position encoding strategy.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int | None = None,
        context_dim: int | None = None,
        qkv_bias: bool = True,
        out_bias: bool = True,
        qk_norm: bool = False,
        eps: float = 1e-6,
        chunk_size: int | None = None,
        use_splash: bool = False,
        *,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.num_heads = num_heads
        self.head_dim = head_dim or dim // num_heads
        self.qk_norm = qk_norm
        self.chunk_size = chunk_size
        self.use_splash = use_splash

        inner_dim = num_heads * self.head_dim
        kv_input_dim = context_dim if context_dim is not None else dim

        self.q_proj = nnx.Linear(
            dim,
            inner_dim,
            use_bias=qkv_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.k_proj = nnx.Linear(
            kv_input_dim,
            inner_dim,
            use_bias=qkv_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.v_proj = nnx.Linear(
            kv_input_dim,
            inner_dim,
            use_bias=qkv_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.out_proj = nnx.Linear(
            inner_dim,
            dim,
            use_bias=out_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        if qk_norm:
            self.norm_q = nnx.RMSNorm(inner_dim, epsilon=eps, param_dtype=param_dtype, rngs=rngs)
            self.norm_k = nnx.RMSNorm(inner_dim, epsilon=eps, param_dtype=param_dtype, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,
        context: jax.Array | None = None,
        freqs_cis: jax.Array | None = None,
        mask: jax.Array | None = None,
    ) -> jax.Array:
        """Run attention.

        Args:
            x: Query input of shape (B, q_len, dim).
            context: Optional context for K/V of shape (B, kv_len, context_dim).
                     If None, self-attention is performed.
            freqs_cis: Optional RoPE frequencies of shape (seq_len, head_dim // 2).
                       Applied to Q and K before the dot product.
            mask: Optional boolean mask of shape (q_len, kv_len),
                  (B, q_len, kv_len), or (B, num_heads, q_len, kv_len).
                  True means "attend", False means "mask out".

        Returns:
            Output of shape (B, q_len, dim).
        """
        B, q_len, _ = x.shape
        kv_input = context if context is not None else x
        kv_len = kv_input.shape[1]

        # Project Q/K/V
        q = self.q_proj(x)
        k = self.k_proj(kv_input)
        v = self.v_proj(kv_input)

        # Apply QK norm before reshaping to multi-head layout
        if self.qk_norm:
            q = self.norm_q(q)
            k = self.norm_k(k)

        # Reshape to multi-head: (B, seq, num_heads, head_dim)
        q = q.reshape(B, q_len, self.num_heads, self.head_dim)
        k = k.reshape(B, kv_len, self.num_heads, self.head_dim)
        v = v.reshape(B, kv_len, self.num_heads, self.head_dim)

        # Apply rotary embeddings — freqs_cis is (seq_len, head_dim // 2).
        # apply_rotary_emb expects (..., seq_len, head_dim), so we transpose
        # to (B, num_heads, seq, head_dim) and back.
        # Note: RoPE uses complex64 which promotes Q/K to float32.
        # We cast V to match so dot_product_attention sees uniform dtypes.
        if freqs_cis is not None:
            q = apply_rotary_emb(q.transpose(0, 2, 1, 3), freqs_cis).transpose(0, 2, 1, 3)
            k = apply_rotary_emb(k.transpose(0, 2, 1, 3), freqs_cis).transpose(0, 2, 1, 3)
            v = v.astype(q.dtype)

        # Convert boolean mask to additive bias for jax.nn.dot_product_attention.
        # Despite Q/K/V being (B, T, N, H), the bias is (B, N, T, S) internally.
        # Accepted mask shapes:
        #   (q_len, kv_len)              -> broadcast over batch and heads
        #   (B, q_len, kv_len)           -> broadcast over heads
        #   (B, num_heads, q_len, kv_len) -> no broadcast needed
        if mask is not None:
            if mask.ndim == 2:
                # (q_len, kv_len) -> (1, 1, q_len, kv_len)
                bias = jnp.where(mask, 0.0, jnp.finfo(q.dtype).min)
                bias = bias[None, None, ...]
            elif mask.ndim == 3:
                # (B, q_len, kv_len) -> (B, 1, q_len, kv_len)
                bias = jnp.where(mask, 0.0, jnp.finfo(q.dtype).min)
                bias = bias[:, None, :, :]
            else:
                # (B, num_heads, q_len, kv_len) — used as-is
                bias = jnp.where(mask, 0.0, jnp.finfo(q.dtype).min)
        else:
            bias = None

        # Dispatch to chunked or standard attention.
        # Chunked attention splits Q into smaller chunks to avoid materializing
        # the full (q_len x kv_len) attention matrix — critical for long
        # video sequences that would otherwise OOM on TPU.
        if self.chunk_size is not None and q_len > self.chunk_size:
            out = _chunked_attention(q, k, v, bias, self.chunk_size)
        else:
            # jax.nn.dot_product_attention expects (B, T, N, H) layout
            out = jax.nn.dot_product_attention(q, k, v, bias=bias)

        # Merge heads: (B, q_len, num_heads, head_dim) -> (B, q_len, inner_dim)
        out = out.reshape(B, q_len, self.num_heads * self.head_dim)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Blockwise causal attention for CausalWanDiT
# ---------------------------------------------------------------------------


def _sdpa(q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
    """Scaled dot-product attention on (B, L, H, D) tensors."""
    return jax.nn.dot_product_attention(q, k, v)


def _causal_sdpa(q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
    """Causal (lower-triangular) scaled dot-product attention.

    Supports asymmetric Q/K lengths: mask is lower-triangular with
    diagonal offset ``L_k - L_q`` so the last query attends to all keys.
    """
    L_q = q.shape[1]
    L_k = k.shape[1]
    offset = L_k - L_q
    mask = jnp.tril(jnp.ones((L_q, L_k), dtype=jnp.bool_), k=offset)
    bias = jnp.where(mask, 0.0, jnp.finfo(q.dtype).min)[None, None, :, :]
    return jax.nn.dot_product_attention(q, k, v, bias=bias)


def blockwise_causal_attn(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    frame_seqlen: int,
    num_frame_per_block: int,
    action_horizon: int,
    state_horizon: int,
    num_action_per_block: int,
    num_state_per_block: int,
) -> jax.Array:
    """Procedural block-wise causal attention (inference / no teacher forcing).

    Sequence layout: ``[first_image | image_blocks | action_tokens | state_tokens]``

    All inputs are ``(B, total_len, num_heads, head_dim)`` with RoPE already
    applied to Q and K.
    """
    total_len = q.shape[1]
    first_image_len = frame_seqlen
    image_blocks_len = total_len - first_image_len - action_horizon - state_horizon
    num_image_blocks = image_blocks_len // (num_frame_per_block * frame_seqlen)
    block_size = num_frame_per_block * frame_seqlen

    image_blocks_start = first_image_len
    action_start = image_blocks_start + image_blocks_len
    state_start = action_start + action_horizon

    parts_out = []

    parts_out.append(
        _sdpa(q[:, :first_image_len], k[:, :first_image_len], v[:, :first_image_len])
    )

    img_block_parts = []
    for bi in range(num_image_blocks):
        blk_s = image_blocks_start + bi * block_size
        blk_e = image_blocks_start + (bi + 1) * block_size
        a_s = action_start + bi * num_action_per_block
        a_e = action_start + (bi + 1) * num_action_per_block
        s_s = state_start + bi * num_state_per_block
        s_e = state_start + (bi + 1) * num_state_per_block

        k_ctx = jnp.concatenate([
            k[:, :first_image_len],
            k[:, image_blocks_start:blk_e],
            k[:, a_s:a_e],
            k[:, s_s:s_e],
        ], axis=1)
        v_ctx = jnp.concatenate([
            v[:, :first_image_len],
            v[:, image_blocks_start:blk_e],
            v[:, a_s:a_e],
            v[:, s_s:s_e],
        ], axis=1)
        img_block_parts.append(_sdpa(q[:, blk_s:blk_e], k_ctx, v_ctx))
    if img_block_parts:
        parts_out.append(jnp.concatenate(img_block_parts, axis=1))

    act_parts = []
    for bi in range(num_image_blocks):
        a_s = action_start + bi * num_action_per_block
        a_e = action_start + (bi + 1) * num_action_per_block
        img_end = image_blocks_start + (bi + 1) * block_size
        s_s = state_start + bi * num_state_per_block
        s_e = state_start + (bi + 1) * num_state_per_block

        k_ctx = jnp.concatenate([
            k[:, :first_image_len],
            k[:, image_blocks_start:img_end],
            k[:, a_s:a_e],
            k[:, s_s:s_e],
        ], axis=1)
        v_ctx = jnp.concatenate([
            v[:, :first_image_len],
            v[:, image_blocks_start:img_end],
            v[:, a_s:a_e],
            v[:, s_s:s_e],
        ], axis=1)
        act_parts.append(_sdpa(q[:, a_s:a_e], k_ctx, v_ctx))
    if act_parts:
        parts_out.append(jnp.concatenate(act_parts, axis=1))

    state_parts = []
    for bi in range(state_horizon // max(num_state_per_block, 1)):
        s_s = state_start + bi * num_state_per_block
        s_e = state_start + (bi + 1) * num_state_per_block
        state_parts.append(_sdpa(q[:, s_s:s_e], k[:, s_s:s_e], v[:, s_s:s_e]))
    if state_parts:
        parts_out.append(jnp.concatenate(state_parts, axis=1))

    return jnp.concatenate(parts_out, axis=1)


def blockwise_causal_attn_tf(
    roped_q: jax.Array,
    roped_k: jax.Array,
    v: jax.Array,
    frame_seqlen: int,
    num_frame_per_block: int,
    num_action_per_block: int,
    num_state_per_block: int,
    clean_image_seq_len: int,
    noisy_image_seq_len: int,
    action_horizon: int,
    state_horizon: int,
) -> jax.Array:
    """Teacher-forcing blockwise causal attention.

    Layout: ``[clean_video | noisy_video | action_tokens | state_tokens]``

    All Q/K have RoPE already applied.  V is unmodified.
    """
    half = clean_image_seq_len

    clean_q = roped_q[:, :half]
    clean_k = roped_k[:, :half]
    clean_v = v[:, :half]

    noisy_img_q = roped_q[:, half:half + noisy_image_seq_len]
    noisy_act_q = roped_q[:, half + noisy_image_seq_len:half + noisy_image_seq_len + action_horizon]
    noisy_state_q = roped_q[:, half + noisy_image_seq_len + action_horizon:]

    noisy_img_k = roped_k[:, half:half + noisy_image_seq_len]
    noisy_act_k = roped_k[:, half + noisy_image_seq_len:half + noisy_image_seq_len + action_horizon]
    noisy_state_k = roped_k[:, half + noisy_image_seq_len + action_horizon:]

    noisy_img_v = v[:, half:half + noisy_image_seq_len]
    noisy_act_v = v[:, half + noisy_image_seq_len:half + noisy_image_seq_len + action_horizon]
    noisy_state_v = v[:, half + noisy_image_seq_len + action_horizon:]

    clean_out = _causal_sdpa(clean_q, clean_k, clean_v)

    noisy_frames = noisy_image_seq_len // frame_seqlen
    num_blocks = (noisy_frames - 1) // num_frame_per_block
    block_size = frame_seqlen * num_frame_per_block

    noisy_img_parts = []
    noisy_img_parts.append(
        _sdpa(noisy_img_q[:, :frame_seqlen],
              noisy_img_k[:, :frame_seqlen],
              noisy_img_v[:, :frame_seqlen])
    )

    for bi in range(num_blocks):
        ns = frame_seqlen + bi * block_size
        ne = frame_seqlen + (bi + 1) * block_size
        clean_end = frame_seqlen + bi * block_size
        a_s = bi * num_action_per_block
        a_e = (bi + 1) * num_action_per_block
        s_s = bi * num_state_per_block
        s_e = (bi + 1) * num_state_per_block

        k_ctx = jnp.concatenate([
            clean_k[:, :clean_end],
            noisy_img_k[:, ns:ne],
            noisy_act_k[:, a_s:a_e],
            noisy_state_k[:, s_s:s_e],
        ], axis=1)
        v_ctx = jnp.concatenate([
            clean_v[:, :clean_end],
            noisy_img_v[:, ns:ne],
            noisy_act_v[:, a_s:a_e],
            noisy_state_v[:, s_s:s_e],
        ], axis=1)
        noisy_img_parts.append(_sdpa(noisy_img_q[:, ns:ne], k_ctx, v_ctx))
    noisy_img_out = jnp.concatenate(noisy_img_parts, axis=1)

    noisy_act_parts = []
    for bi in range(num_blocks):
        a_s = bi * num_action_per_block
        a_e = (bi + 1) * num_action_per_block
        clean_end = frame_seqlen + bi * block_size
        ni_s = frame_seqlen + bi * block_size
        ni_e = frame_seqlen + (bi + 1) * block_size
        s_s = bi * num_state_per_block
        s_e = (bi + 1) * num_state_per_block

        k_ctx = jnp.concatenate([
            clean_k[:, :clean_end],
            noisy_img_k[:, ni_s:ni_e],
            noisy_act_k[:, a_s:a_e],
            noisy_state_k[:, s_s:s_e],
        ], axis=1)
        v_ctx = jnp.concatenate([
            clean_v[:, :clean_end],
            noisy_img_v[:, ni_s:ni_e],
            noisy_act_v[:, a_s:a_e],
            noisy_state_v[:, s_s:s_e],
        ], axis=1)
        noisy_act_parts.append(_sdpa(noisy_act_q[:, a_s:a_e], k_ctx, v_ctx))
    noisy_act_out = (
        jnp.concatenate(noisy_act_parts, axis=1)
        if noisy_act_parts
        else jnp.zeros_like(noisy_act_q[:, :0])
    )

    noisy_state_parts = []
    for bi in range(state_horizon // max(num_state_per_block, 1)):
        s_s = bi * num_state_per_block
        s_e = (bi + 1) * num_state_per_block
        noisy_state_parts.append(
            _sdpa(noisy_state_q[:, s_s:s_e],
                  noisy_state_k[:, s_s:s_e],
                  noisy_state_v[:, s_s:s_e])
        )
    noisy_state_out = (
        jnp.concatenate(noisy_state_parts, axis=1)
        if noisy_state_parts
        else jnp.zeros_like(noisy_state_q[:, :0])
    )

    return jnp.concatenate(
        [clean_out, noisy_img_out, noisy_act_out, noisy_state_out], axis=1
    )
