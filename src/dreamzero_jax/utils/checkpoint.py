"""Checkpoint conversion: PyTorch DreamZero -> Flax NNX.

Converts PyTorch state dicts to the flat pytree format used by Flax NNX
models, handling weight transpositions, key renaming, and parameter
reshaping for all DreamZero sub-models.

The conversion follows a registry-based approach: transform functions are
registered per weight-name pattern.  The ``build_key_mapping`` function
assembles a complete mapping from PyTorch keys to (Flax pytree path,
transform function) tuples for a given model configuration.

Usage
-----
Programmatic::

    from dreamzero_jax.utils.checkpoint import convert_and_apply
    model = DreamZero(config, rngs=nnx.Rngs(0))
    convert_and_apply("path/to/pytorch.pt", model, config)
    save_flax_checkpoint(nnx.state(model), "path/to/flax_ckpt")

CLI::

    uv run python scripts/convert_checkpoint.py \\
        --input path/to/pytorch.pt --output path/to/flax_ckpt
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Array transform helpers
# ---------------------------------------------------------------------------


def _identity(x: np.ndarray) -> np.ndarray:
    """No-op transform (bias, embedding, etc.)."""
    return x


def _transpose_dense(x: np.ndarray) -> np.ndarray:
    """PyTorch Linear weight (out, in) -> Flax kernel (in, out)."""
    assert x.ndim == 2, f"Expected 2D weight, got shape {x.shape}"
    return x.T


def _transpose_conv2d(x: np.ndarray) -> np.ndarray:
    """PyTorch Conv2d weight (out, in, kH, kW) -> Flax kernel (kH, kW, in, out)."""
    assert x.ndim == 4, f"Expected 4D weight, got shape {x.shape}"
    return np.transpose(x, (2, 3, 1, 0))


def _transpose_conv3d(x: np.ndarray) -> np.ndarray:
    """PyTorch Conv3d weight (out, in, kT, kH, kW) -> Flax kernel (kT, kH, kW, in, out)."""
    assert x.ndim == 5, f"Expected 5D weight, got shape {x.shape}"
    return np.transpose(x, (2, 3, 4, 1, 0))


def _transpose_conv_auto(x: np.ndarray) -> np.ndarray:
    """Auto-detect conv dimensionality and transpose."""
    if x.ndim == 4:
        return _transpose_conv2d(x)
    elif x.ndim == 5:
        return _transpose_conv3d(x)
    elif x.ndim == 2:
        return _transpose_dense(x)
    else:
        raise ValueError(f"Cannot auto-detect conv transpose for shape {x.shape}")


# ---------------------------------------------------------------------------
# Loading PyTorch checkpoints
# ---------------------------------------------------------------------------


def load_pytorch_checkpoint(
    path: str | Path,
    *,
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    """Load a PyTorch checkpoint and return the state dict as numpy arrays.

    Attempts to load without requiring PyTorch by using ``safetensors`` or
    numpy-based loading.  Falls back to ``torch.load`` if needed.

    Args:
        path: Path to the checkpoint file (``.pt``, ``.pth``, ``.bin``,
            or ``.safetensors``).
        device: Device string for ``torch.load`` (default ``"cpu"``).

    Returns:
        Dictionary mapping parameter names to numpy arrays.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        ImportError: If neither safetensors nor torch is available.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    state_dict: dict[str, np.ndarray] = {}

    # --- Try safetensors first (no torch dependency) ---
    if path.suffix == ".safetensors":
        try:
            from safetensors.numpy import load_file
            state_dict = load_file(str(path))
            logger.info("Loaded checkpoint via safetensors: %s", path)
            return state_dict
        except ImportError:
            logger.warning("safetensors not available, falling back to torch")

    # --- Try loading with torch ---
    try:
        import torch
        raw = torch.load(str(path), map_location=device, weights_only=True)

        # Handle common checkpoint wrappings
        if isinstance(raw, dict):
            if "state_dict" in raw:
                raw = raw["state_dict"]
            elif "model_state_dict" in raw:
                raw = raw["model_state_dict"]
            elif "model" in raw:
                raw = raw["model"]

        for k, v in raw.items():
            if isinstance(v, torch.Tensor):
                state_dict[k] = v.detach().cpu().numpy()
            elif isinstance(v, np.ndarray):
                state_dict[k] = v
            # Skip non-tensor entries (e.g. config, optimizer state)

        logger.info(
            "Loaded checkpoint via torch: %s (%d parameters)", path, len(state_dict)
        )
        return state_dict

    except ImportError:
        pass

    # --- Last resort: numpy .npz ---
    if path.suffix == ".npz":
        data = np.load(str(path), allow_pickle=True)
        state_dict = dict(data)
        logger.info("Loaded checkpoint via numpy: %s", path)
        return state_dict

    raise ImportError(
        "Cannot load checkpoint: install either 'safetensors' or 'torch'. "
        f"File: {path}"
    )


# ---------------------------------------------------------------------------
# Key mapping registry
# ---------------------------------------------------------------------------


class ParamMapping(NamedTuple):
    """Mapping from a PyTorch key to a Flax pytree path + transform."""

    flax_path: tuple[str, ...]
    transform: Callable[[np.ndarray], np.ndarray]


class KeyMappingBuilder:
    """Builds a mapping from PyTorch state-dict keys to Flax parameter paths.

    Uses a list of ``(pattern, replacement, transform)`` rules.  Patterns
    use Python ``re`` syntax (applied via ``re.sub``).  Rules are tried in
    order; the first match wins.
    """

    def __init__(self) -> None:
        self._rules: list[tuple[re.Pattern, str, Callable]] = []

    def add_rule(
        self,
        pattern: str,
        replacement: str,
        transform: Callable[[np.ndarray], np.ndarray] = _identity,
    ) -> None:
        """Register a mapping rule.

        Args:
            pattern: Regex pattern matching PyTorch key names.
            replacement: Replacement string (can use regex groups).
            transform: Array transform to apply (default: identity).
        """
        self._rules.append((re.compile(pattern), replacement, transform))

    def map_key(self, pt_key: str) -> ParamMapping | None:
        """Map a single PyTorch key.

        Returns:
            A ParamMapping if a rule matches, else None.
        """
        for regex, replacement, transform in self._rules:
            new_key, n = regex.subn(replacement, pt_key)
            if n > 0:
                # Convert dot-separated path to tuple
                flax_path = tuple(new_key.split("."))
                return ParamMapping(flax_path=flax_path, transform=transform)
        return None


# ---------------------------------------------------------------------------
# Standard weight name conversions
# ---------------------------------------------------------------------------


def _add_linear_rules(
    builder: KeyMappingBuilder,
    pt_prefix: str,
    flax_prefix: str,
    use_bias: bool = True,
) -> None:
    """Add rules for a nn.Linear -> nnx.Linear conversion."""
    builder.add_rule(
        rf"^{re.escape(pt_prefix)}\.weight$",
        f"{flax_prefix}.kernel",
        _transpose_dense,
    )
    if use_bias:
        builder.add_rule(
            rf"^{re.escape(pt_prefix)}\.bias$",
            f"{flax_prefix}.bias",
        )


def _add_conv_rules(
    builder: KeyMappingBuilder,
    pt_prefix: str,
    flax_prefix: str,
    use_bias: bool = True,
) -> None:
    """Add rules for a Conv -> nnx.Conv conversion."""
    builder.add_rule(
        rf"^{re.escape(pt_prefix)}\.weight$",
        f"{flax_prefix}.kernel",
        _transpose_conv_auto,
    )
    if use_bias:
        builder.add_rule(
            rf"^{re.escape(pt_prefix)}\.bias$",
            f"{flax_prefix}.bias",
        )


def _add_layernorm_rules(
    builder: KeyMappingBuilder,
    pt_prefix: str,
    flax_prefix: str,
    has_scale: bool = True,
    has_bias: bool = True,
) -> None:
    """Add rules for LayerNorm / RMSNorm."""
    if has_scale:
        builder.add_rule(
            rf"^{re.escape(pt_prefix)}\.weight$",
            f"{flax_prefix}.scale",
        )
    if has_bias:
        builder.add_rule(
            rf"^{re.escape(pt_prefix)}\.bias$",
            f"{flax_prefix}.bias",
        )


def _add_embed_rules(
    builder: KeyMappingBuilder,
    pt_prefix: str,
    flax_prefix: str,
) -> None:
    """Add rules for nn.Embedding -> nnx.Embed (same shape)."""
    builder.add_rule(
        rf"^{re.escape(pt_prefix)}\.weight$",
        f"{flax_prefix}.embedding",
    )


# ---------------------------------------------------------------------------
# MLP rules (nn/mlp.py: MLP uses w_up and w_down)
# ---------------------------------------------------------------------------


def _add_mlp_rules(
    builder: KeyMappingBuilder,
    pt_prefix: str,
    flax_prefix: str,
    use_bias: bool = True,
) -> None:
    """Add rules for a 2-layer MLP.

    PyTorch nn.Sequential convention: {prefix}.0.weight / {prefix}.2.weight
    Flax MLP: {prefix}.w_up.kernel / {prefix}.w_down.kernel
    """
    # Sequential(Linear, act, Linear) -> MLP(w_up, w_down)
    _add_linear_rules(builder, f"{pt_prefix}.0", f"{flax_prefix}.w_up", use_bias)
    _add_linear_rules(builder, f"{pt_prefix}.2", f"{flax_prefix}.w_down", use_bias)


def _add_mlp_direct_rules(
    builder: KeyMappingBuilder,
    pt_prefix: str,
    flax_prefix: str,
    use_bias: bool = True,
) -> None:
    """Add rules for MLP when PyTorch uses named attributes (linear1/linear2)."""
    _add_linear_rules(builder, f"{pt_prefix}.linear1", f"{flax_prefix}.w_up", use_bias)
    _add_linear_rules(builder, f"{pt_prefix}.linear2", f"{flax_prefix}.w_down", use_bias)


# ---------------------------------------------------------------------------
# Component-level mapping builders
# ---------------------------------------------------------------------------


def _add_dit_block_rules(
    builder: KeyMappingBuilder,
    pt_prefix: str,
    flax_prefix: str,
    has_image_input: bool = False,
    qk_norm: bool = True,
    cross_attn_norm: bool = False,
) -> None:
    """Add rules for a WanDiTBlock."""
    # Norms (no scale/bias -- elementwise_affine=False in PyTorch)
    # PyTorch LayerNorm with elementwise_affine=False has no weight/bias
    # so no rules needed for norm1, norm2

    # Modulation: direct copy of the learnable bias
    builder.add_rule(
        rf"^{re.escape(pt_prefix)}\.modulation$",
        f"{flax_prefix}.modulation.value",
    )

    # Self-attention
    sa_pt = f"{pt_prefix}.self_attn"
    sa_fl = f"{flax_prefix}.self_attn"
    _add_linear_rules(builder, f"{sa_pt}.q_proj", f"{sa_fl}.q_proj")
    _add_linear_rules(builder, f"{sa_pt}.k_proj", f"{sa_fl}.k_proj")
    _add_linear_rules(builder, f"{sa_pt}.v_proj", f"{sa_fl}.v_proj")
    _add_linear_rules(builder, f"{sa_pt}.out_proj", f"{sa_fl}.out_proj")
    if qk_norm:
        _add_layernorm_rules(builder, f"{sa_pt}.norm_q", f"{sa_fl}.norm_q", has_bias=False)
        _add_layernorm_rules(builder, f"{sa_pt}.norm_k", f"{sa_fl}.norm_k", has_bias=False)

    # Cross-attention
    ca_pt = f"{pt_prefix}.cross_attn"
    ca_fl = f"{flax_prefix}.cross_attn"
    _add_linear_rules(builder, f"{ca_pt}.q_proj", f"{ca_fl}.q_proj")
    _add_linear_rules(builder, f"{ca_pt}.k_proj", f"{ca_fl}.k_proj")
    _add_linear_rules(builder, f"{ca_pt}.v_proj", f"{ca_fl}.v_proj")
    _add_linear_rules(builder, f"{ca_pt}.out_proj", f"{ca_fl}.out_proj")

    if has_image_input:
        # WanI2VCrossAttention has separate image K/V
        _add_linear_rules(builder, f"{ca_pt}.k_img", f"{ca_fl}.k_img")
        _add_linear_rules(builder, f"{ca_pt}.v_img", f"{ca_fl}.v_img")
        if qk_norm:
            _add_layernorm_rules(
                builder, f"{ca_pt}.norm_q", f"{ca_fl}.norm_q", has_bias=False,
            )
            _add_layernorm_rules(
                builder, f"{ca_pt}.norm_k_text", f"{ca_fl}.norm_k_text", has_bias=False,
            )
            _add_layernorm_rules(
                builder, f"{ca_pt}.norm_k_img", f"{ca_fl}.norm_k_img", has_bias=False,
            )
    else:
        if qk_norm:
            _add_layernorm_rules(
                builder, f"{ca_pt}.norm_q", f"{ca_fl}.norm_q", has_bias=False,
            )
            _add_layernorm_rules(
                builder, f"{ca_pt}.norm_k", f"{ca_fl}.norm_k", has_bias=False,
            )

    # Optional cross-attention norm
    if cross_attn_norm:
        _add_layernorm_rules(builder, f"{pt_prefix}.norm3", f"{flax_prefix}.norm3")

    # FFN: PyTorch uses nn.Sequential(Linear, act, Linear) or named (ffn.0, ffn.2)
    _add_mlp_rules(builder, f"{pt_prefix}.ffn", f"{flax_prefix}.ffn")


def _add_dit_head_rules(
    builder: KeyMappingBuilder,
    pt_prefix: str,
    flax_prefix: str,
) -> None:
    """Add rules for WanDiTHead."""
    # LayerNorm (no scale/bias)
    # Modulation
    builder.add_rule(
        rf"^{re.escape(pt_prefix)}\.modulation$",
        f"{flax_prefix}.modulation.value",
    )
    # Linear
    _add_linear_rules(builder, f"{pt_prefix}.linear", f"{flax_prefix}.linear")


def _add_mlp_proj_rules(
    builder: KeyMappingBuilder,
    pt_prefix: str,
    flax_prefix: str,
) -> None:
    """Add rules for MLPProj (norm_in, linear1, linear2, norm_out)."""
    _add_layernorm_rules(builder, f"{pt_prefix}.norm_in", f"{flax_prefix}.norm_in")
    _add_linear_rules(builder, f"{pt_prefix}.linear1", f"{flax_prefix}.linear1")
    _add_linear_rules(builder, f"{pt_prefix}.linear2", f"{flax_prefix}.linear2")
    _add_layernorm_rules(builder, f"{pt_prefix}.norm_out", f"{flax_prefix}.norm_out")


def _add_category_specific_linear_rules(
    builder: KeyMappingBuilder,
    pt_prefix: str,
    flax_prefix: str,
) -> None:
    """Add rules for CategorySpecificLinear.

    PyTorch stores weight as (num_cats, in, out) and bias as (num_cats, out).
    Flax stores them as nnx.Param (same shapes -- no transpose needed since
    the forward pass uses einsum, not matmul).
    """
    builder.add_rule(
        rf"^{re.escape(pt_prefix)}\.weight$",
        f"{flax_prefix}.weight.value",
    )
    builder.add_rule(
        rf"^{re.escape(pt_prefix)}\.bias$",
        f"{flax_prefix}.bias.value",
    )


def _add_category_specific_mlp_rules(
    builder: KeyMappingBuilder,
    pt_prefix: str,
    flax_prefix: str,
) -> None:
    """Add rules for CategorySpecificMLP (linear1, linear2)."""
    _add_category_specific_linear_rules(
        builder, f"{pt_prefix}.linear1", f"{flax_prefix}.linear1",
    )
    _add_category_specific_linear_rules(
        builder, f"{pt_prefix}.linear2", f"{flax_prefix}.linear2",
    )


# ---------------------------------------------------------------------------
# Full model mapping
# ---------------------------------------------------------------------------


def build_key_mapping(config: Any) -> KeyMappingBuilder:
    """Build a complete PyTorch -> Flax key mapping for DreamZero.

    Args:
        config: A ``DreamZeroConfig`` instance (or compatible object with
            the same attributes).

    Returns:
        A ``KeyMappingBuilder`` with all rules registered.
    """
    b = KeyMappingBuilder()
    has_image = getattr(config, "has_image_input", True)
    qk_norm = getattr(config, "qk_norm", True)
    cross_attn_norm = getattr(config, "cross_attn_norm", False)
    num_layers = getattr(config, "num_layers", 30)

    # ---------------------------------------------------------------
    # model.dit (CausalWanDiT)
    # ---------------------------------------------------------------
    dit_pt = "model.dit"
    dit_fl = "dit"

    # Patch embedding (PatchEmbed3D -> proj is a Conv)
    _add_conv_rules(b, f"{dit_pt}.patch_embedding.proj", f"{dit_fl}.patch_embedding.proj")

    # Time embedding (MLP: Sequential with .0 and .2)
    _add_mlp_rules(b, f"{dit_pt}.time_embedding", f"{dit_fl}.time_embedding")

    # Time projection (single linear, applied after SiLU)
    _add_linear_rules(b, f"{dit_pt}.time_projection", f"{dit_fl}.time_projection")
    # Handle the case where PyTorch wraps it in Sequential(SiLU, Linear)
    # time_projection.1.weight -> time_projection.kernel
    b.add_rule(
        rf"^{re.escape(dit_pt)}\.time_projection\.1\.weight$",
        f"{dit_fl}.time_projection.kernel",
        _transpose_dense,
    )
    b.add_rule(
        rf"^{re.escape(dit_pt)}\.time_projection\.1\.bias$",
        f"{dit_fl}.time_projection.bias",
    )

    # Text embedding (MLP)
    _add_mlp_rules(b, f"{dit_pt}.text_embedding", f"{dit_fl}.text_embedding")

    # Image embedding projection (MLPProj)
    if has_image:
        _add_mlp_proj_rules(b, f"{dit_pt}.img_emb", f"{dit_fl}.img_emb")

    # DiT blocks
    for i in range(num_layers):
        _add_dit_block_rules(
            b,
            f"{dit_pt}.blocks.{i}",
            f"{dit_fl}.blocks.{i}",
            has_image_input=has_image,
            qk_norm=qk_norm,
            cross_attn_norm=cross_attn_norm,
        )

    # Output head
    _add_dit_head_rules(b, f"{dit_pt}.head", f"{dit_fl}.head")

    # Action-specific: state_encoder, action_encoder, action_decoder
    _add_category_specific_mlp_rules(
        b, f"{dit_pt}.state_encoder", f"{dit_fl}.state_encoder",
    )

    # MultiEmbodimentActionEncoder: W1, W2, W3 (CategorySpecificLinear)
    for w_name in ("W1", "W2", "W3"):
        _add_category_specific_linear_rules(
            b, f"{dit_pt}.action_encoder.{w_name}",
            f"{dit_fl}.action_encoder.{w_name}",
        )

    _add_category_specific_mlp_rules(
        b, f"{dit_pt}.action_decoder", f"{dit_fl}.action_decoder",
    )

    # ---------------------------------------------------------------
    # model.text_encoder (WanTextEncoder)
    # ---------------------------------------------------------------
    te_pt = "model.text_encoder"
    te_fl = "text_encoder"

    _add_embed_rules(b, f"{te_pt}.token_embedding", f"{te_fl}.token_embedding")

    text_num_layers = getattr(config, "text_num_layers", 24)
    for i in range(text_num_layers):
        blk_pt = f"{te_pt}.blocks.{i}"
        blk_fl = f"{te_fl}.blocks.{i}"

        # RMSNorm (norm1, norm2)
        _add_layernorm_rules(b, f"{blk_pt}.norm1", f"{blk_fl}.norm1", has_bias=False)
        _add_layernorm_rules(b, f"{blk_pt}.norm2", f"{blk_fl}.norm2", has_bias=False)

        # T5Attention: q, k, v, o (no bias)
        for proj in ("q", "k", "v", "o"):
            _add_linear_rules(
                b, f"{blk_pt}.attn.{proj}", f"{blk_fl}.attn.{proj}", use_bias=False,
            )

        # T5FeedForward: gate, fc1, fc2 (no bias)
        for fc_name in ("gate", "fc1", "fc2"):
            _add_linear_rules(
                b, f"{blk_pt}.ffn.{fc_name}", f"{blk_fl}.ffn.{fc_name}", use_bias=False,
            )

        # Per-layer relative position embedding (if not shared)
        _add_embed_rules(
            b, f"{blk_pt}.pos_embedding.embedding",
            f"{blk_fl}.pos_embedding.embedding",
        )
        # Alternative naming: direct embedding attribute
        _add_embed_rules(
            b, f"{blk_pt}.pos_embedding",
            f"{blk_fl}.pos_embedding.embedding",
        )

    # Shared position embedding (if used)
    _add_embed_rules(
        b, f"{te_pt}.pos_embedding.embedding",
        f"{te_fl}.pos_embedding.embedding",
    )

    # Final norm
    _add_layernorm_rules(b, f"{te_pt}.norm", f"{te_fl}.norm", has_bias=False)

    # ---------------------------------------------------------------
    # model.image_encoder (WanImageEncoder -> visual -> VisionTransformer)
    # ---------------------------------------------------------------
    ie_pt = "model.image_encoder.visual"
    ie_fl = "image_encoder.visual"

    # Patch embedding conv
    _add_conv_rules(b, f"{ie_pt}.patch_embedding", f"{ie_fl}.patch_embedding")
    # Alternative: some checkpoints name it conv1
    _add_conv_rules(b, f"{ie_pt}.conv1", f"{ie_fl}.patch_embedding")

    # CLS token
    b.add_rule(
        rf"^{re.escape(ie_pt)}\.cls_embedding$",
        f"{ie_fl}.cls_embedding.value",
    )
    b.add_rule(
        rf"^{re.escape(ie_pt)}\.class_embedding$",
        f"{ie_fl}.cls_embedding.value",
    )

    # Positional embedding
    b.add_rule(
        rf"^{re.escape(ie_pt)}\.pos_embedding$",
        f"{ie_fl}.pos_embedding.value",
    )
    b.add_rule(
        rf"^{re.escape(ie_pt)}\.positional_embedding$",
        f"{ie_fl}.pos_embedding.value",
    )

    # Pre-norm
    _add_layernorm_rules(b, f"{ie_pt}.pre_norm", f"{ie_fl}.pre_norm")
    _add_layernorm_rules(b, f"{ie_pt}.ln_pre", f"{ie_fl}.pre_norm")

    # Post-norm
    _add_layernorm_rules(
        b, f"{ie_pt}.post_norm_layer", f"{ie_fl}.post_norm_layer",
    )
    _add_layernorm_rules(b, f"{ie_pt}.ln_post", f"{ie_fl}.post_norm_layer")

    # Output head (projection matrix as nnx.Param)
    b.add_rule(
        rf"^{re.escape(ie_pt)}\.head$",
        f"{ie_fl}.head.value",
    )
    b.add_rule(
        rf"^{re.escape(ie_pt)}\.proj$",
        f"{ie_fl}.head.value",
    )

    # ViT transformer blocks
    image_num_layers = getattr(config, "image_num_layers", 32)
    for i in range(image_num_layers):
        vblk_pt = f"{ie_pt}.transformer.{i}"
        vblk_fl = f"{ie_fl}.transformer.{i}"

        # Alternative naming: resblocks
        vblk_pt_alt = f"{ie_pt}.transformer.resblocks.{i}"

        for prefix_pt in (vblk_pt, vblk_pt_alt):
            # LayerNorm
            _add_layernorm_rules(b, f"{prefix_pt}.norm1", f"{vblk_fl}.norm1")
            _add_layernorm_rules(b, f"{prefix_pt}.norm2", f"{vblk_fl}.norm2")
            _add_layernorm_rules(b, f"{prefix_pt}.ln_1", f"{vblk_fl}.norm1")
            _add_layernorm_rules(b, f"{prefix_pt}.ln_2", f"{vblk_fl}.norm2")

            # Self-attention: fused QKV (to_qkv) and output projection (proj)
            _add_linear_rules(
                b, f"{prefix_pt}.attn.to_qkv", f"{vblk_fl}.attn.to_qkv",
            )
            _add_linear_rules(
                b, f"{prefix_pt}.attn.proj", f"{vblk_fl}.attn.proj",
            )
            # Alternative naming: in_proj_weight/bias, out_proj
            b.add_rule(
                rf"^{re.escape(prefix_pt)}\.attn\.in_proj_weight$",
                f"{vblk_fl}.attn.to_qkv.kernel",
                _transpose_dense,
            )
            b.add_rule(
                rf"^{re.escape(prefix_pt)}\.attn\.in_proj_bias$",
                f"{vblk_fl}.attn.to_qkv.bias",
            )
            _add_linear_rules(
                b, f"{prefix_pt}.attn.out_proj", f"{vblk_fl}.attn.proj",
            )

            # MLP: fc1, fc2
            _add_linear_rules(b, f"{prefix_pt}.fc1", f"{vblk_fl}.fc1")
            _add_linear_rules(b, f"{prefix_pt}.fc2", f"{vblk_fl}.fc2")
            # Alternative naming: mlp.c_fc, mlp.c_proj (OpenAI CLIP)
            _add_linear_rules(b, f"{prefix_pt}.mlp.c_fc", f"{vblk_fl}.fc1")
            _add_linear_rules(b, f"{prefix_pt}.mlp.c_proj", f"{vblk_fl}.fc2")

    # ---------------------------------------------------------------
    # model.vae (WanVideoVAE -> encoder, decoder)
    # ---------------------------------------------------------------
    _add_vae_rules(b, "model.vae", "vae", config)

    return b


def _add_vae_rules(
    builder: KeyMappingBuilder,
    pt_prefix: str,
    flax_prefix: str,
    config: Any,
) -> None:
    """Add rules for WanVideoVAE (encoder + decoder)."""
    _add_vae_encoder_rules(builder, f"{pt_prefix}.encoder", f"{flax_prefix}.encoder", config)
    _add_vae_decoder_rules(builder, f"{pt_prefix}.decoder", f"{flax_prefix}.decoder", config)


def _add_causal_conv3d_rules(
    builder: KeyMappingBuilder,
    pt_prefix: str,
    flax_prefix: str,
) -> None:
    """Add rules for CausalConv3d (wraps a single conv)."""
    _add_conv_rules(builder, f"{pt_prefix}.conv", f"{flax_prefix}.conv")
    # Some checkpoints may name the inner conv directly
    _add_conv_rules(builder, pt_prefix, f"{flax_prefix}.conv")


def _add_residual_block_rules(
    builder: KeyMappingBuilder,
    pt_prefix: str,
    flax_prefix: str,
) -> None:
    """Add rules for ResidualBlock."""
    _add_layernorm_rules(builder, f"{pt_prefix}.norm1", f"{flax_prefix}.norm1", has_bias=False)
    _add_causal_conv3d_rules(builder, f"{pt_prefix}.conv1", f"{flax_prefix}.conv1")
    _add_layernorm_rules(builder, f"{pt_prefix}.norm2", f"{flax_prefix}.norm2", has_bias=False)
    _add_causal_conv3d_rules(builder, f"{pt_prefix}.conv2", f"{flax_prefix}.conv2")
    # Shortcut (optional)
    _add_causal_conv3d_rules(builder, f"{pt_prefix}.shortcut", f"{flax_prefix}.shortcut")
    # Alternative: skip_conv / nin_shortcut
    _add_causal_conv3d_rules(builder, f"{pt_prefix}.skip_conv", f"{flax_prefix}.shortcut")
    _add_causal_conv3d_rules(
        builder, f"{pt_prefix}.nin_shortcut", f"{flax_prefix}.shortcut",
    )


def _add_attention_block_rules(
    builder: KeyMappingBuilder,
    pt_prefix: str,
    flax_prefix: str,
) -> None:
    """Add rules for VAE AttentionBlock."""
    _add_layernorm_rules(builder, f"{pt_prefix}.norm", f"{flax_prefix}.norm", has_bias=False)
    attn_pt = f"{pt_prefix}.attn"
    attn_fl = f"{flax_prefix}.attn"
    _add_linear_rules(builder, f"{attn_pt}.q_proj", f"{attn_fl}.q_proj")
    _add_linear_rules(builder, f"{attn_pt}.k_proj", f"{attn_fl}.k_proj")
    _add_linear_rules(builder, f"{attn_pt}.v_proj", f"{attn_fl}.v_proj")
    _add_linear_rules(builder, f"{attn_pt}.out_proj", f"{attn_fl}.out_proj")


def _add_vae_encoder_rules(
    builder: KeyMappingBuilder,
    pt_prefix: str,
    flax_prefix: str,
    config: Any,
) -> None:
    """Add rules for Encoder3d."""
    # Stem
    _add_causal_conv3d_rules(builder, f"{pt_prefix}.stem", f"{flax_prefix}.stem")

    # Stages (dynamic number based on dim_multipliers)
    num_stages = 4  # default: (1, 2, 4, 4)
    num_res_blocks = 2
    for s in range(num_stages):
        for j in range(num_res_blocks):
            _add_residual_block_rules(
                builder,
                f"{pt_prefix}.stages.{s}.blocks.{j}",
                f"{flax_prefix}.stages.{s}.blocks.{j}",
            )
            # Alternative PyTorch naming: encoder.down.{s}.block.{j}
            _add_residual_block_rules(
                builder,
                f"{pt_prefix}.down.{s}.block.{j}",
                f"{flax_prefix}.stages.{s}.blocks.{j}",
            )

        # Resample layers
        for r in range(3):  # up to 3 resample layers per stage
            resample_pt = f"{pt_prefix}.stages.{s}.resample.{r}"
            resample_fl = f"{flax_prefix}.stages.{s}.resample.{r}"
            # SpatialDownsample
            _add_conv_rules(builder, f"{resample_pt}.conv", f"{resample_fl}.conv")
            # TemporalDownsample (CausalConv3d)
            _add_causal_conv3d_rules(builder, f"{resample_pt}", f"{resample_fl}")
            # Alternative naming: encoder.down.{s}.downsample
            _add_conv_rules(
                builder,
                f"{pt_prefix}.down.{s}.downsample.conv",
                f"{resample_fl}.conv",
            )

    # Middle blocks
    _add_residual_block_rules(
        builder, f"{pt_prefix}.mid_block1", f"{flax_prefix}.mid_block1",
    )
    _add_attention_block_rules(
        builder, f"{pt_prefix}.mid_attn", f"{flax_prefix}.mid_attn",
    )
    _add_residual_block_rules(
        builder, f"{pt_prefix}.mid_block2", f"{flax_prefix}.mid_block2",
    )
    # Alternative naming: mid.block_1, mid.attn_1, mid.block_2
    _add_residual_block_rules(
        builder, f"{pt_prefix}.mid.block_1", f"{flax_prefix}.mid_block1",
    )
    _add_attention_block_rules(
        builder, f"{pt_prefix}.mid.attn_1", f"{flax_prefix}.mid_attn",
    )
    _add_residual_block_rules(
        builder, f"{pt_prefix}.mid.block_2", f"{flax_prefix}.mid_block2",
    )

    # Head
    _add_layernorm_rules(
        builder, f"{pt_prefix}.head_norm", f"{flax_prefix}.head_norm", has_bias=False,
    )
    _add_causal_conv3d_rules(
        builder, f"{pt_prefix}.head_conv", f"{flax_prefix}.head_conv",
    )
    # Alternative: norm_out, conv_out
    _add_layernorm_rules(
        builder, f"{pt_prefix}.norm_out", f"{flax_prefix}.head_norm", has_bias=False,
    )
    _add_causal_conv3d_rules(
        builder, f"{pt_prefix}.conv_out", f"{flax_prefix}.head_conv",
    )


def _add_vae_decoder_rules(
    builder: KeyMappingBuilder,
    pt_prefix: str,
    flax_prefix: str,
    config: Any,
) -> None:
    """Add rules for Decoder3d."""
    # Stem
    _add_causal_conv3d_rules(builder, f"{pt_prefix}.stem", f"{flax_prefix}.stem")
    # Alternative naming: conv_in
    _add_causal_conv3d_rules(builder, f"{pt_prefix}.conv_in", f"{flax_prefix}.stem")

    # Middle blocks
    _add_residual_block_rules(
        builder, f"{pt_prefix}.mid_block1", f"{flax_prefix}.mid_block1",
    )
    _add_attention_block_rules(
        builder, f"{pt_prefix}.mid_attn", f"{flax_prefix}.mid_attn",
    )
    _add_residual_block_rules(
        builder, f"{pt_prefix}.mid_block2", f"{flax_prefix}.mid_block2",
    )
    _add_residual_block_rules(
        builder, f"{pt_prefix}.mid.block_1", f"{flax_prefix}.mid_block1",
    )
    _add_attention_block_rules(
        builder, f"{pt_prefix}.mid.attn_1", f"{flax_prefix}.mid_attn",
    )
    _add_residual_block_rules(
        builder, f"{pt_prefix}.mid.block_2", f"{flax_prefix}.mid_block2",
    )

    # Stages
    num_stages = 4
    num_dec_blocks = 3  # num_res_blocks + 1
    for s in range(num_stages):
        for j in range(num_dec_blocks):
            _add_residual_block_rules(
                builder,
                f"{pt_prefix}.stages.{s}.blocks.{j}",
                f"{flax_prefix}.stages.{s}.blocks.{j}",
            )
            _add_residual_block_rules(
                builder,
                f"{pt_prefix}.up.{s}.block.{j}",
                f"{flax_prefix}.stages.{s}.blocks.{j}",
            )

        # Resample layers
        for r in range(3):
            resample_pt = f"{pt_prefix}.stages.{s}.resample.{r}"
            resample_fl = f"{flax_prefix}.stages.{s}.resample.{r}"
            _add_conv_rules(builder, f"{resample_pt}.conv", f"{resample_fl}.conv")
            _add_causal_conv3d_rules(builder, f"{resample_pt}", f"{resample_fl}")
            _add_conv_rules(
                builder,
                f"{pt_prefix}.up.{s}.upsample.conv",
                f"{resample_fl}.conv",
            )

    # Head
    _add_layernorm_rules(
        builder, f"{pt_prefix}.head_norm", f"{flax_prefix}.head_norm", has_bias=False,
    )
    _add_causal_conv3d_rules(
        builder, f"{pt_prefix}.head_conv", f"{flax_prefix}.head_conv",
    )
    _add_layernorm_rules(
        builder, f"{pt_prefix}.norm_out", f"{flax_prefix}.head_norm", has_bias=False,
    )
    _add_causal_conv3d_rules(
        builder, f"{pt_prefix}.conv_out", f"{flax_prefix}.head_conv",
    )


# ---------------------------------------------------------------------------
# Conversion engine
# ---------------------------------------------------------------------------


def convert_checkpoint(
    pt_state_dict: dict[str, np.ndarray],
    config: Any,
    *,
    strict: bool = False,
    prefix_strip: str | None = None,
) -> dict[tuple[str, ...], jax.Array]:
    """Convert a PyTorch state dict to a flat dict of Flax arrays.

    Args:
        pt_state_dict: PyTorch parameter name -> numpy array.
        config: A ``DreamZeroConfig`` instance.
        strict: If True, raise on unmapped PyTorch keys.
        prefix_strip: Optional prefix to strip from all PyTorch keys
            before mapping (e.g. ``"module."`` for DDP checkpoints).

    Returns:
        Dictionary mapping Flax pytree path tuples to JAX arrays.
    """
    builder = build_key_mapping(config)
    converted: dict[tuple[str, ...], jax.Array] = {}
    unmapped: list[str] = []
    mapped_count = 0

    for pt_key, pt_value in pt_state_dict.items():
        # Strip common prefixes
        key = pt_key
        if prefix_strip and key.startswith(prefix_strip):
            key = key[len(prefix_strip):]

        # Also strip "module." from DDP wrapping
        if key.startswith("module."):
            key = key[len("module."):]

        mapping = builder.map_key(key)
        if mapping is None:
            unmapped.append(pt_key)
            continue

        arr = mapping.transform(pt_value)
        converted[mapping.flax_path] = jnp.array(arr)
        mapped_count += 1

    logger.info(
        "Converted %d/%d parameters (%d unmapped)",
        mapped_count,
        len(pt_state_dict),
        len(unmapped),
    )

    if unmapped:
        logger.warning("Unmapped PyTorch keys:\n  %s", "\n  ".join(sorted(unmapped)))

    if strict and unmapped:
        raise ValueError(
            f"Strict mode: {len(unmapped)} PyTorch keys could not be mapped:\n"
            + "\n".join(f"  {k}" for k in sorted(unmapped))
        )

    return converted


# ---------------------------------------------------------------------------
# Apply converted params to Flax NNX model
# ---------------------------------------------------------------------------


def _flatten_state(state: Any) -> dict[tuple[str, ...], Any]:
    """Flatten an nnx.State into a dict of path tuples -> nnx.Variable.

    Works by traversing the nested structure returned by ``nnx.state(model)``.
    """
    from flax import nnx

    flat: dict[tuple[str, ...], Any] = {}

    def _recurse(obj: Any, path: tuple[str, ...]) -> None:
        if isinstance(obj, nnx.VariableState):
            flat[path] = obj
        elif isinstance(obj, dict):
            for k, v in obj.items():
                _recurse(v, path + (str(k),))
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                _recurse(v, path + (str(i),))
        elif hasattr(obj, "__dict__"):
            for k, v in vars(obj).items():
                if not k.startswith("_"):
                    _recurse(v, path + (k,))
        # For nnx.State which acts like a mapping
        elif hasattr(obj, "flat_state"):
            for k, v in obj.flat_state().items():
                flat_key = path + tuple(k.split(".")) if isinstance(k, str) else path + (str(k),)
                flat[flat_key] = v

    _recurse(state, ())
    return flat


def apply_to_model(
    model: Any,
    converted_params: dict[tuple[str, ...], jax.Array],
    *,
    strict: bool = False,
) -> tuple[int, list[str], list[tuple[str, ...]]]:
    """Apply converted parameters to a Flax NNX model instance.

    Traverses the model's state using ``nnx.state`` and sets matching
    parameters from ``converted_params``.

    Args:
        model: A Flax NNX model instance (e.g. ``DreamZero``).
        converted_params: Output of :func:`convert_checkpoint`.
        strict: If True, raise on missing or extra keys.

    Returns:
        Tuple of (num_applied, missing_flax_keys, extra_converted_keys).
    """
    from flax import nnx

    graphdef, state = nnx.split(model)

    # Flatten the state to get all variable paths
    flat_state = _flatten_state(state)

    applied = 0
    missing_in_converted: list[str] = []
    used_converted_keys: set[tuple[str, ...]] = set()

    for flax_path, var_state in flat_state.items():
        if not isinstance(var_state, nnx.VariableState):
            continue

        # Try direct match
        if flax_path in converted_params:
            new_value = converted_params[flax_path]
            old_shape = var_state.value.shape
            new_shape = new_value.shape

            if old_shape != new_shape:
                logger.warning(
                    "Shape mismatch for %s: model=%s, checkpoint=%s",
                    ".".join(flax_path),
                    old_shape,
                    new_shape,
                )
                if strict:
                    raise ValueError(
                        f"Shape mismatch for {'.'.join(flax_path)}: "
                        f"model={old_shape}, checkpoint={new_shape}"
                    )
                continue

            var_state.value = new_value
            used_converted_keys.add(flax_path)
            applied += 1
        else:
            # Try matching without the .value suffix (for nnx.Param)
            # The flax_path from nnx.state might include 'value' at the end
            # while our mapping might or might not include it
            alt_path = flax_path
            if flax_path[-1] == "value":
                alt_path = flax_path[:-1]
            elif flax_path[-1] != "value":
                alt_path = flax_path + ("value",)

            if alt_path in converted_params:
                new_value = converted_params[alt_path]
                old_shape = var_state.value.shape
                new_shape = new_value.shape

                if old_shape != new_shape:
                    logger.warning(
                        "Shape mismatch for %s: model=%s, checkpoint=%s",
                        ".".join(flax_path),
                        old_shape,
                        new_shape,
                    )
                    if strict:
                        raise ValueError(
                            f"Shape mismatch for {'.'.join(flax_path)}: "
                            f"model={old_shape}, checkpoint={new_shape}"
                        )
                    continue

                var_state.value = new_value
                used_converted_keys.add(alt_path)
                applied += 1
            else:
                missing_in_converted.append(".".join(flax_path))

    # Reconstruct the model from the updated state
    nnx.update(model, nnx.merge(graphdef, state))

    extra_keys = [
        k for k in converted_params
        if k not in used_converted_keys
    ]

    if missing_in_converted:
        logger.info(
            "%d model parameters not found in checkpoint", len(missing_in_converted),
        )
        if len(missing_in_converted) <= 20:
            for k in sorted(missing_in_converted):
                logger.debug("  Missing: %s", k)

    if extra_keys:
        logger.info(
            "%d converted parameters not found in model", len(extra_keys),
        )
        if len(extra_keys) <= 20:
            for k in sorted(extra_keys):
                logger.debug("  Extra: %s", ".".join(k))

    if strict and (missing_in_converted or extra_keys):
        raise ValueError(
            f"Strict mode: {len(missing_in_converted)} missing, "
            f"{len(extra_keys)} extra parameters"
        )

    logger.info("Applied %d parameters to model", applied)
    return applied, missing_in_converted, extra_keys


# ---------------------------------------------------------------------------
# Orbax checkpoint save / load
# ---------------------------------------------------------------------------


def save_flax_checkpoint(
    state: Any,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    """Save Flax NNX model state to an orbax checkpoint.

    Args:
        state: An ``nnx.State`` object (from ``nnx.state(model)``).
        path: Directory path for the checkpoint.
        overwrite: If True, overwrite existing checkpoint.
    """
    import orbax.checkpoint as ocp

    path = Path(path)

    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(
        path,
        state,
        force=overwrite,
    )
    logger.info("Saved Flax checkpoint to %s", path)


def load_flax_checkpoint(
    path: str | Path,
    model: Any | None = None,
) -> Any:
    """Load a Flax checkpoint from an orbax directory.

    Args:
        path: Directory path to the checkpoint.
        model: Optional model instance. If provided, the checkpoint is
            restored into the model's state structure (enabling shape
            and dtype validation). If None, returns raw restored state.

    Returns:
        The restored state (an ``nnx.State``-compatible object).
    """
    import orbax.checkpoint as ocp
    from flax import nnx

    path = Path(path)
    checkpointer = ocp.StandardCheckpointer()

    if model is not None:
        abstract_state = jax.eval_shape(lambda: nnx.state(model))
        state = checkpointer.restore(path, target=abstract_state)
    else:
        state = checkpointer.restore(path)

    logger.info("Loaded Flax checkpoint from %s", path)
    return state


# ---------------------------------------------------------------------------
# Convenience: convert_and_apply
# ---------------------------------------------------------------------------


def convert_and_apply(
    pt_checkpoint_path: str | Path,
    model: Any,
    config: Any,
    *,
    strict: bool = False,
    prefix_strip: str | None = None,
) -> tuple[int, list[str], list[tuple[str, ...]]]:
    """End-to-end: load PyTorch checkpoint, convert, and apply to model.

    Args:
        pt_checkpoint_path: Path to PyTorch checkpoint.
        model: A Flax NNX model instance.
        config: A ``DreamZeroConfig`` instance.
        strict: If True, raise on unmapped/missing keys.
        prefix_strip: Optional prefix to strip from PyTorch keys.

    Returns:
        Tuple of (num_applied, missing_flax_keys, extra_converted_keys).
    """
    pt_state = load_pytorch_checkpoint(pt_checkpoint_path)
    converted = convert_checkpoint(
        pt_state, config, strict=strict, prefix_strip=prefix_strip,
    )
    return apply_to_model(model, converted, strict=strict)


# ---------------------------------------------------------------------------
# Diagnostic utilities
# ---------------------------------------------------------------------------


def print_key_mapping(config: Any, *, max_lines: int = 0) -> None:
    """Print the key mapping rules for debugging.

    Creates a representative set of PyTorch key names and shows what
    each maps to in Flax.  Useful for verifying the mapping before
    running a full conversion.

    Args:
        config: A ``DreamZeroConfig`` instance.
        max_lines: Maximum number of lines to print (0 = unlimited).
    """
    builder = build_key_mapping(config)
    count = 0
    for regex, replacement, transform in builder._rules:
        transform_name = transform.__name__
        print(f"  {regex.pattern}  ->  {replacement}  [{transform_name}]")
        count += 1
        if max_lines and count >= max_lines:
            print(f"  ... ({len(builder._rules) - count} more rules)")
            break


def compare_param_shapes(
    pt_state_dict: dict[str, np.ndarray],
    model: Any,
    config: Any,
    *,
    prefix_strip: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Compare shapes between PyTorch checkpoint and Flax model.

    Useful for diagnosing conversion issues before running the full
    conversion.

    Args:
        pt_state_dict: PyTorch state dict.
        model: Flax NNX model instance.
        config: DreamZeroConfig instance.
        prefix_strip: Optional prefix to strip from PyTorch keys.

    Returns:
        Dict with entries for each mapped parameter containing:
        ``pt_key``, ``pt_shape``, ``flax_path``, ``flax_shape``,
        ``expected_shape`` (after transform), ``match`` (bool).
    """
    from flax import nnx

    builder = build_key_mapping(config)
    _, state = nnx.split(model)
    flat_state = _flatten_state(state)

    report: dict[str, dict[str, Any]] = {}

    for pt_key, pt_value in pt_state_dict.items():
        key = pt_key
        if prefix_strip and key.startswith(prefix_strip):
            key = key[len(prefix_strip):]
        if key.startswith("module."):
            key = key[len("module."):]

        mapping = builder.map_key(key)
        if mapping is None:
            report[pt_key] = {
                "pt_key": pt_key,
                "pt_shape": pt_value.shape,
                "flax_path": None,
                "flax_shape": None,
                "expected_shape": None,
                "match": None,
            }
            continue

        transformed = mapping.transform(pt_value)
        flax_path = mapping.flax_path

        flax_shape = None
        if flax_path in flat_state:
            var = flat_state[flax_path]
            if hasattr(var, "value"):
                flax_shape = var.value.shape

        report[pt_key] = {
            "pt_key": pt_key,
            "pt_shape": pt_value.shape,
            "flax_path": ".".join(flax_path),
            "flax_shape": flax_shape,
            "expected_shape": transformed.shape,
            "match": flax_shape == transformed.shape if flax_shape else None,
        }

    return report
