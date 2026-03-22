"""Staged inference for DreamZero — encoders and DiT never coexist in HBM.

The ``generate_offload`` approach loads ALL weights (encoders + DiT) before
encoding starts, consuming ~13.65 GB total — leaving only ~2.35 GB/chip for
activations on v5e-8, which is not enough.

This module solves the problem with **staged initialization**:

  Phase 1: Create encoders on CPU -> shard to TPU -> encode -> delete encoders
  Phase 2: Create DiT on CPU -> shard to TPU -> denoise -> return

Peak weight memory is max(encoders, DiT) ~ 4.82 GB/chip instead of the sum.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from dreamzero_jax.models.action_head import CausalWanDiT
from dreamzero_jax.models.image_encoder import WanImageEncoder
from dreamzero_jax.models.text_encoder import WanTextEncoder
from dreamzero_jax.models.vae import WanVideoVAE
from dreamzero_jax.schedulers.flow_euler import (
    euler_step,
    make_flow_euler_schedule,
)

logger = logging.getLogger(__name__)


def _hbm_usage_str() -> str:
    """Return a formatted string of current HBM usage, or empty if unavailable."""
    try:
        dev = jax.local_devices()[0]
        stats = dev.memory_stats()
        if stats and "bytes_in_use" in stats:
            used = stats["bytes_in_use"] / (1024 ** 3)
            peak = stats.get("peak_bytes_in_use", 0) / (1024 ** 3)
            return f"HBM: {used:.2f} GB used, {peak:.2f} GB peak"
    except Exception:
        pass
    return "HBM: N/A"


def _create_encoders(config, rngs):
    """Instantiate only the encoder sub-models from a DreamZeroConfig."""
    dtype = config.dtype
    param_dtype = config.param_dtype
    text_enc = WanTextEncoder(
        vocab=config.text_vocab,
        dim=config.text_dim,
        dim_attn=config.text_attn_dim,
        dim_ffn=config.text_ffn_dim,
        num_heads=config.text_num_heads,
        num_layers=config.text_num_layers,
        num_buckets=config.text_num_buckets,
        shared_pos=False,
        dtype=dtype,
        param_dtype=param_dtype,
        rngs=rngs,
    )
    img_enc = WanImageEncoder(
        image_size=config.image_size,
        patch_size=config.image_patch_size,
        dim=config.image_dim,
        mlp_ratio=config.image_mlp_ratio,
        out_dim=config.image_out_dim,
        num_heads=config.image_num_heads,
        num_layers=config.image_num_layers,
        activation="gelu",
        dtype=dtype,
        param_dtype=param_dtype,
        rngs=rngs,
    )
    vae = WanVideoVAE(
        z_dim=config.vae_z_dim,
        base_dim=config.vae_base_dim,
        dtype=dtype,
        param_dtype=param_dtype,
        rngs=rngs,
    )
    return text_enc, img_enc, vae


def _create_dit(config, rngs):
    """Instantiate only the DiT backbone from a DreamZeroConfig."""
    return CausalWanDiT(
        dim=config.dim,
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        ffn_dim=config.ffn_dim,
        freq_dim=config.freq_dim,
        text_dim=config.text_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        patch_size=config.patch_size,
        has_image_input=config.has_image_input,
        qk_norm=config.qk_norm,
        cross_attn_norm=config.cross_attn_norm,
        use_scan=config.use_scan,
        use_remat=config.use_remat,
        action_dim=config.action_dim,
        state_dim=config.state_dim,
        action_hidden_size=config.action_hidden_size,
        num_action_per_block=config.num_action_per_block,
        num_state_per_block=config.num_state_per_block,
        num_frames_per_block=config.num_frames_per_block,
        max_num_embodiments=config.max_num_embodiments,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
        rngs=rngs,
    )


def _build_i2v_cond(vae, video, latents):
    """Build 20-channel I2V conditioning: 4ch mask + 16ch first-frame latent."""
    B, T, H, W, _ = video.shape
    T_lat = latents.shape[1]
    H_lat, W_lat = latents.shape[2], latents.shape[3]

    zeros_pad = jnp.zeros((B, T - 1, H, W, 3), dtype=video.dtype)
    first_frame = video[:, :1]
    padded = jnp.concatenate([first_frame, zeros_pad], axis=1)
    image_latent = vae.encode(padded)

    mask = jnp.zeros((B, T_lat, H_lat, W_lat, 4), dtype=latents.dtype)
    mask = mask.at[:, :1, :, :, :].set(1.0)

    return jnp.concatenate([mask, image_latent], axis=-1)


def _run_encoding(text_enc, img_enc, vae, video, token_ids, attention_mask, has_image_input):
    """Run all encoder forward passes under nnx.jit."""

    @nnx.jit
    def _encode(te, ie, va, vid, tids, mask):
        prompt_emb = te(tids, mask=mask)
        if mask is not None:
            prompt_emb = prompt_emb * mask[:, :, None]
        latents = va.encode(vid)
        clip_emb = ie.encode_image(vid[:, 0]) if has_image_input else None
        i2v_cond = _build_i2v_cond(va, vid, latents) if has_image_input else None
        return prompt_emb, latents, clip_emb, i2v_cond

    return _encode(text_enc, img_enc, vae, video, token_ids, attention_mask)


def _run_denoise_scan(dit, config, latents, prompt_emb, clip_emb,
                      state, embodiment_id, num_steps, cfg, *, key,
                      i2v_cond=None, use_cfg=True):
    """Run the Euler-scan denoising loop with a standalone DiT.

    When ``use_cfg=False``, runs only the conditional pass per step,
    halving activation memory.
    """
    B = latents.shape[0]
    T_lat = latents.shape[1]
    f = T_lat // config.patch_size[0]
    num_blocks = f // config.num_frames_per_block

    actual = state.shape[1]
    if actual < num_blocks:
        state = jnp.pad(state, ((0, 0), (0, num_blocks - actual), (0, 0)))
    elif actual > num_blocks:
        state = state[:, :num_blocks]

    key_vid, key_act = jax.random.split(key)
    noisy_video = jax.random.normal(key_vid, latents.shape)
    total_actions = (
        num_blocks
        * (config.action_horizon or config.num_action_per_block)
    )
    noisy_actions = jax.random.normal(
        key_act, (B, total_actions, config.action_dim),
    )

    sched_video = make_flow_euler_schedule(
        num_inference_steps=num_steps,
        num_train_timesteps=config.num_train_timesteps,
        shift=config.scheduler_shift,
    )
    sched_action = make_flow_euler_schedule(
        num_inference_steps=num_steps,
        num_train_timesteps=config.num_train_timesteps,
        shift=config.scheduler_shift,
    )

    scan_xs = (
        sched_video.timesteps,
        sched_action.timesteps,
        sched_video.sigmas,
        sched_video.sigmas_next,
        sched_action.sigmas,
        sched_action.sigmas_next,
    )

    if use_cfg:
        null_prompt = jnp.zeros_like(prompt_emb)

        @nnx.jit
        def _run_scan(model, noisy_vid, noisy_act, st, emb_id, p_emb, n_prompt, c_emb, y_cond):
            def _step(carry, xs):
                nv, na = carry
                (tv, ta, sv, svn, sa, san) = xs
                tv_b = jnp.broadcast_to(tv, (B,))
                ta_b = jnp.broadcast_to(ta, (B,))

                vc, ac = model(
                    nv, tv_b, p_emb, st, emb_id, na,
                    timestep_action=ta_b, clip_emb=c_emb, y=y_cond,
                )
                vu, au = model(
                    nv, tv_b, n_prompt, st, emb_id, na,
                    timestep_action=ta_b, clip_emb=c_emb, y=y_cond,
                )
                vp = vu + cfg * (vc - vu)
                nv_next = euler_step(vp, nv, sv, svn)
                na_next = euler_step(ac, na, sa, san)
                return (nv_next, na_next), None

            (fv, fa), _ = jax.lax.scan(_step, (noisy_vid, noisy_act), scan_xs)
            return fv, fa

        final_video, final_actions = _run_scan(
            dit, noisy_video, noisy_actions,
            state, embodiment_id, prompt_emb, null_prompt, clip_emb, i2v_cond,
        )
    else:
        @nnx.jit
        def _run_scan_no_cfg(model, noisy_vid, noisy_act, st, emb_id, p_emb, c_emb, y_cond):
            def _step(carry, xs):
                nv, na = carry
                (tv, ta, sv, svn, sa, san) = xs
                tv_b = jnp.broadcast_to(tv, (B,))
                ta_b = jnp.broadcast_to(ta, (B,))

                vp, ap = model(
                    nv, tv_b, p_emb, st, emb_id, na,
                    timestep_action=ta_b, clip_emb=c_emb, y=y_cond,
                )
                nv_next = euler_step(vp, nv, sv, svn)
                na_next = euler_step(ap, na, sa, san)
                return (nv_next, na_next), None

            (fv, fa), _ = jax.lax.scan(_step, (noisy_vid, noisy_act), scan_xs)
            return fv, fa

        final_video, final_actions = _run_scan_no_cfg(
            dit, noisy_video, noisy_actions,
            state, embodiment_id, prompt_emb, clip_emb, i2v_cond,
        )

    from dreamzero_jax.models.dreamzero import InferenceOutput
    return InferenceOutput(action_pred=final_actions, video_pred=final_video)


PT_PREFIX_MAP = {
    "text_encoder": "action_head.text_encoder.",
    "image_encoder": "action_head.image_encoder.",
    "vae": "action_head.vae.",
    "dit": "action_head.model.",
}

FLAX_PREFIX_MAP = {
    "text_encoder": "text_encoder",
    "image_encoder": "image_encoder",
    "vae": "vae",
    "dit": "dit",
}


def _filter_converted_for_component(
    converted: dict[tuple[str, ...], Any],
    component: str,
) -> dict[tuple[str, ...], Any]:
    """Filter converted params by Flax prefix and strip it for sub-model application.

    The full conversion produces paths like ("text_encoder", "layers", "0", ...),
    but a sub-model's flat_state has relative paths like ("layers", "0", ...).
    This strips the leading component prefix so apply_to_model can match.
    """
    flax_prefix = FLAX_PREFIX_MAP[component]
    result: dict[tuple[str, ...], Any] = {}
    for path, value in converted.items():
        if path and path[0] == flax_prefix:
            result[path[1:]] = value
    return result


def _detect_checkpoint_format(checkpoint_dir: Path) -> str:
    """Return 'orbax', 'pytorch', or 'npy' based on directory contents."""
    if (checkpoint_dir / "model.safetensors.index.json").exists():
        return "pytorch"
    if (checkpoint_dir / "model.safetensors").exists():
        return "pytorch"
    npy_files = list(checkpoint_dir.glob("*.npy"))
    if npy_files:
        return "npy"
    if (checkpoint_dir / "_METADATA").exists() or (
        checkpoint_dir / "default"
    ).is_dir():
        return "orbax"
    return "pytorch"


def _load_pt_state_filtered(
    checkpoint_dir: Path,
    pt_prefix: str,
) -> dict[str, np.ndarray]:
    """Load a PT checkpoint and return only keys matching the given prefix."""
    from dreamzero_jax.utils.hf_download import load_checkpoint_auto

    full_state = load_checkpoint_auto(checkpoint_dir)
    return {
        k: v for k, v in full_state.items()
        if k.startswith(pt_prefix)
    }


def _load_npy_state_filtered(
    checkpoint_dir: Path,
    flax_prefix: str,
) -> dict[tuple[str, ...], jax.Array]:
    """Load .npy files whose path starts with the flax sub-model prefix."""
    result: dict[tuple[str, ...], jax.Array] = {}
    for npy_path in sorted(checkpoint_dir.glob("*.npy")):
        key_str = npy_path.stem
        if not key_str.startswith(flax_prefix):
            continue
        relative_key = key_str[len(flax_prefix):]
        if relative_key.startswith("."):
            relative_key = relative_key[1:]
        flax_path = tuple(relative_key.split("."))
        result[flax_path] = jnp.array(np.load(npy_path))
    return result


def _load_weights_for_submodel(
    model: Any,
    config: Any,
    checkpoint_dir: Path,
    component: str,
    ckpt_format: str,
    verbose: bool = True,
) -> None:
    """Load checkpoint weights into a single sub-model (in-place).

    Args:
        model: The Flax NNX sub-model instance (on CPU).
        config: DreamZeroConfig for key mapping rules.
        checkpoint_dir: Path to the checkpoint directory.
        component: One of 'text_encoder', 'image_encoder', 'vae', 'dit'.
        ckpt_format: 'pytorch' or 'npy'.
        verbose: Log progress.
    """
    from dreamzero_jax.utils.checkpoint import (
        apply_to_model,
        convert_checkpoint,
    )

    if ckpt_format == "pytorch":
        from dreamzero_jax.utils.hf_download import load_checkpoint_auto
        full_state = load_checkpoint_auto(checkpoint_dir)
        if verbose:
            logger.info(
                "Converting %d PT params (full) for %s", len(full_state), component,
            )
        full_converted = convert_checkpoint(full_state, config, prefix_strip="action_head.")
        subset = _filter_converted_for_component(full_converted, component)
        if not subset:
            logger.warning(
                "No converted params for component %r", component,
            )
            return
        n_applied, missing, extra = apply_to_model(model, subset)
        if verbose:
            logger.info(
                "%s: applied %d params (%d missing, %d extra)",
                component, n_applied, len(missing), len(extra),
            )

    elif ckpt_format == "npy":
        flax_prefix = component
        params = _load_npy_state_filtered(checkpoint_dir, flax_prefix)
        if not params:
            logger.warning(
                "No .npy files with prefix %r found for %s",
                flax_prefix, component,
            )
            return
        if verbose:
            logger.info(
                "Applying %d .npy params for %s", len(params), component,
            )
        from dreamzero_jax.utils.checkpoint import apply_to_model
        n_applied, missing, extra = apply_to_model(model, params)
        if verbose:
            logger.info(
                "%s: applied %d params (%d missing, %d extra)",
                component, n_applied, len(missing), len(extra),
            )

    elif ckpt_format == "orbax":
        from dreamzero_jax.utils.checkpoint import load_flax_checkpoint
        state = load_flax_checkpoint(checkpoint_dir, model)
        nnx.update(model, state)
        if verbose:
            logger.info("%s: loaded orbax checkpoint", component)

    else:
        raise ValueError(f"Unknown checkpoint format: {ckpt_format}")


def generate_staged(
    config,
    video: jax.Array,
    token_ids: jax.Array,
    state: jax.Array,
    embodiment_id: jax.Array,
    attention_mask: jax.Array | None = None,
    num_inference_steps: int | None = None,
    cfg_scale: float | None = None,
    use_cfg: bool = True,
    *,
    key: jax.Array,
    mesh: jax.sharding.Mesh | None = None,
    verbose: bool = True,
    quantize_int8: bool = False,
    checkpoint_dir: str | Path | None = None,
):
    """Staged inference: encoders and DiT are never loaded simultaneously.

    Phase 1: Create encoders on CPU, shard to TPU, encode, delete encoders.
    Phase 2: Create DiT on CPU, shard to TPU, denoise, return results.

    Peak weight memory is max(encoders, DiT) ~ 4.82 GB/chip instead of
    the ~13.65 GB sum that ``generate_offload`` requires.

    When ``use_cfg=False``, the denoising loop runs only the conditional
    pass per step, halving activation memory. Critical for fitting 40L
    14B on v5e-8.

    Args:
        config: DreamZeroConfig instance.
        video: Conditioning video ``(B, T, H, W, 3)`` in ``[-1, 1]``.
        token_ids: Text token IDs ``(B, L)`` int32.
        state: Robot state ``(B, num_blocks, state_dim)``.
        embodiment_id: ``(B,)`` int embodiment IDs.
        attention_mask: ``(B, L)`` text attention mask.
        num_inference_steps: Override number of denoising steps.
        cfg_scale: Override classifier-free guidance scale.
        use_cfg: When False, skip the unconditional pass and CFG
            combination. The DiT runs once per step instead of twice.
        key: PRNG key for noise initialization.
        mesh: Device mesh for sharding. If None, no sharding is applied.
        verbose: Print HBM usage at each phase boundary.
        quantize_int8: Quantize DiT weights to INT8 before sharding.
        checkpoint_dir: Path to a checkpoint directory. Supports:
            - PyTorch safetensors (sharded or single) — converted on the fly
            - Directory of ``.npy`` files keyed by flax path
            - Orbax checkpoint directory
            When None, uses random initialization (for testing only).

    Returns:
        InferenceOutput with ``action_pred`` and ``video_pred``.
    """
    from dreamzero_jax.utils.sharding import shard_params

    num_steps = num_inference_steps or config.num_inference_steps
    cfg = cfg_scale or config.cfg_scale

    cpu_device = jax.devices("cpu")[0]
    cpu_ctx = jax.default_device(cpu_device)

    ckpt_path = Path(checkpoint_dir) if checkpoint_dir is not None else None
    ckpt_fmt = _detect_checkpoint_format(ckpt_path) if ckpt_path else None

    def _log(msg: str) -> None:
        if verbose:
            print(f"  [staged] {msg}  ({_hbm_usage_str()})")

    _cache: dict[str, Any] = {"pt_state": None, "converted": None}

    def _get_converted() -> dict[tuple[str, ...], Any]:
        """Convert full PT state dict once, cache for all sub-models."""
        if _cache["converted"] is None:
            from dreamzero_jax.utils.checkpoint import convert_checkpoint
            from dreamzero_jax.utils.hf_download import load_checkpoint_auto
            assert ckpt_path is not None
            pt_state = load_checkpoint_auto(ckpt_path)
            _log(f"converting {len(pt_state)} PT params (full checkpoint)")
            _cache["converted"] = convert_checkpoint(
                pt_state, config, prefix_strip="action_head.",
            )
            _cache["pt_state"] = pt_state
        return _cache["converted"]

    def _load_component(model: Any, component: str) -> None:
        """Load weights for a sub-model from the checkpoint."""
        if ckpt_path is None:
            return
        assert ckpt_fmt is not None
        if ckpt_fmt == "pytorch":
            from dreamzero_jax.utils.checkpoint import apply_to_model
            full_converted = _get_converted()
            subset = _filter_converted_for_component(full_converted, component)
            if not subset:
                logger.warning(
                    "No converted params for component %r (flax prefix %r)",
                    component, FLAX_PREFIX_MAP[component],
                )
                return
            _log(f"applying {len(subset)} converted params -> {component}")
            n_applied, missing, extra = apply_to_model(model, subset)
            _log(
                f"{component}: {n_applied} applied, "
                f"{len(missing)} missing, {len(extra)} extra"
            )
        elif ckpt_fmt == "npy":
            _load_weights_for_submodel(
                model, config, ckpt_path, component, "npy", verbose,
            )
        elif ckpt_fmt == "orbax":
            _load_weights_for_submodel(
                model, config, ckpt_path, component, "orbax", verbose,
            )

    # ---- Phase 1: Encoders ----
    _log("Phase 1: creating encoders on CPU")
    rngs = nnx.Rngs(params=jax.random.PRNGKey(0))
    with cpu_ctx:
        text_enc, img_enc, vae = _create_encoders(config, rngs)

    if ckpt_path is not None:
        _log("Phase 1: loading encoder weights from checkpoint")
        with cpu_ctx:
            _load_component(text_enc, "text_encoder")
            _load_component(img_enc, "image_encoder")
            _load_component(vae, "vae")

    if mesh is not None:
        _log("Phase 1: sharding encoder weights to TPU")
        text_enc = shard_params(text_enc, mesh, param_dtype=config.param_dtype)
        img_enc = shard_params(img_enc, mesh, param_dtype=config.param_dtype)
        vae = shard_params(vae, mesh, param_dtype=config.param_dtype)

    _log("Phase 1: running encoding")
    prompt_emb, latents, clip_emb, i2v_cond = _run_encoding(
        text_enc, img_enc, vae,
        video, token_ids, attention_mask,
        config.has_image_input,
    )
    ready = [prompt_emb, latents]
    if clip_emb is not None:
        ready.append(clip_emb)
    if i2v_cond is not None:
        ready.append(i2v_cond)
    jax.block_until_ready(tuple(ready))
    _log("Phase 1: encoding complete, deleting encoder weights")

    del text_enc, img_enc, vae
    gc.collect()
    jax.clear_caches()
    gc.collect()
    _log("Phase 1: encoders deleted (caches cleared)")

    # Free the cached PT state if encoder keys are no longer needed
    # and DiT keys will be re-read from the same cache.
    # (Keep cache alive — DiT loading will use it next.)

    # ---- Phase 2: DiT ----
    _log("Phase 2: creating DiT on CPU")
    rngs = nnx.Rngs(params=jax.random.PRNGKey(0))
    with cpu_ctx:
        dit = _create_dit(config, rngs)

        if ckpt_path is not None:
            _log("Phase 2: loading DiT weights from checkpoint")
            _load_component(dit, "dit")

        _cache["pt_state"] = None
        _cache["converted"] = None

        if quantize_int8:
            from dreamzero_jax.utils.quantize import quantize_model
            _log("Phase 2: quantizing DiT to INT8 (on CPU)")
            quantize_model(dit)

    if mesh is not None:
        _log("Phase 2: sharding DiT weights to TPU")
        dit = shard_params(dit, mesh, param_dtype=config.param_dtype)

    cfg_str = "CFG" if use_cfg else "no-CFG"
    _log(f"Phase 2: running denoising ({cfg_str})")
    result = _run_denoise_scan(
        dit, config,
        latents, prompt_emb, clip_emb,
        state, embodiment_id,
        num_steps, cfg,
        key=key,
        i2v_cond=i2v_cond,
        use_cfg=use_cfg,
    )
    jax.block_until_ready(result)
    _log("Phase 2: denoising complete")

    del dit
    gc.collect()

    return result
