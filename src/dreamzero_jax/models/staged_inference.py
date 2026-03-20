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

import jax
import jax.numpy as jnp
from flax import nnx

from dreamzero_jax.models.action_head import CausalWanDiT
from dreamzero_jax.models.image_encoder import WanImageEncoder
from dreamzero_jax.models.text_encoder import WanTextEncoder
from dreamzero_jax.models.vae import WanVideoVAE
from dreamzero_jax.schedulers.flow_euler import (
    euler_step,
    make_flow_euler_schedule,
)


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


def _run_encoding(text_enc, img_enc, vae, video, token_ids, attention_mask, has_image_input):
    """Run all encoder forward passes under nnx.jit."""

    @nnx.jit
    def _encode(te, ie, va, vid, tids, mask):
        prompt_emb = te(tids, mask=mask)
        if mask is not None:
            prompt_emb = prompt_emb * mask[:, :, None]
        latents = va.encode(vid)
        clip_emb = ie.encode_image(vid[:, 0]) if has_image_input else None
        return prompt_emb, latents, clip_emb

    return _encode(text_enc, img_enc, vae, video, token_ids, attention_mask)


def _run_denoise_scan(dit, config, latents, prompt_emb, clip_emb,
                      state, embodiment_id, num_steps, cfg, *, key):
    """Run the Euler-scan denoising loop with a standalone DiT."""
    B = latents.shape[0]
    T_lat = latents.shape[1]
    f = T_lat // config.patch_size[0]
    num_blocks = f // config.num_frames_per_block

    actual = state.shape[1]
    if actual < num_blocks:
        state = jnp.pad(state, ((0, 0), (0, num_blocks - actual), (0, 0)))
    elif actual > num_blocks:
        state = state[:, :num_blocks]

    null_prompt = jnp.zeros_like(prompt_emb)

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

    @nnx.jit
    def _run_scan(model, noisy_vid, noisy_act, st, emb_id, p_emb, n_prompt, c_emb):
        def _step(carry, xs):
            nv, na = carry
            (tv, ta, sv, svn, sa, san) = xs
            tv_b = jnp.broadcast_to(tv, (B,))
            ta_b = jnp.broadcast_to(ta, (B,))

            vc, ac = model(
                nv, tv_b, p_emb, st, emb_id, na,
                timestep_action=ta_b, clip_emb=c_emb,
            )
            vu, au = model(
                nv, tv_b, n_prompt, st, emb_id, na,
                timestep_action=ta_b, clip_emb=c_emb,
            )
            vp = vu + cfg * (vc - vu)
            nv_next = euler_step(vp, nv, sv, svn)
            na_next = euler_step(ac, na, sa, san)
            return (nv_next, na_next), None

        (fv, fa), _ = jax.lax.scan(_step, (noisy_vid, noisy_act), scan_xs)
        return fv, fa

    final_video, final_actions = _run_scan(
        dit, noisy_video, noisy_actions,
        state, embodiment_id, prompt_emb, null_prompt, clip_emb,
    )

    from dreamzero_jax.models.dreamzero import InferenceOutput
    return InferenceOutput(action_pred=final_actions, video_pred=final_video)


def generate_staged(
    config,
    video: jax.Array,
    token_ids: jax.Array,
    state: jax.Array,
    embodiment_id: jax.Array,
    attention_mask: jax.Array | None = None,
    num_inference_steps: int | None = None,
    cfg_scale: float | None = None,
    *,
    key: jax.Array,
    mesh: jax.sharding.Mesh | None = None,
    verbose: bool = True,
    quantize_int8: bool = False,
):
    """Staged inference: encoders and DiT are never loaded simultaneously.

    Phase 1: Create encoders on CPU, shard to TPU, encode, delete encoders.
    Phase 2: Create DiT on CPU, shard to TPU, denoise, return results.

    Peak weight memory is max(encoders, DiT) ~ 4.82 GB/chip instead of
    the ~13.65 GB sum that ``generate_offload`` requires.

    Args:
        config: DreamZeroConfig instance.
        video: Conditioning video ``(B, T, H, W, 3)`` in ``[-1, 1]``.
        token_ids: Text token IDs ``(B, L)`` int32.
        state: Robot state ``(B, num_blocks, state_dim)``.
        embodiment_id: ``(B,)`` int embodiment IDs.
        attention_mask: ``(B, L)`` text attention mask.
        num_inference_steps: Override number of denoising steps.
        cfg_scale: Override classifier-free guidance scale.
        key: PRNG key for noise initialization.
        mesh: Device mesh for sharding. If None, no sharding is applied.
        verbose: Print HBM usage at each phase boundary.

    Returns:
        InferenceOutput with ``action_pred`` and ``video_pred``.
    """
    from dreamzero_jax.utils.sharding import shard_params

    num_steps = num_inference_steps or config.num_inference_steps
    cfg = cfg_scale or config.cfg_scale

    cpu_device = jax.devices("cpu")[0]
    cpu_ctx = jax.default_device(cpu_device)

    def _log(msg: str) -> None:
        if verbose:
            print(f"  [staged] {msg}  ({_hbm_usage_str()})")

    # ---- Phase 1: Encoders ----
    _log("Phase 1: creating encoders on CPU")
    rngs = nnx.Rngs(params=jax.random.PRNGKey(0))
    with cpu_ctx:
        text_enc, img_enc, vae = _create_encoders(config, rngs)

    if mesh is not None:
        _log("Phase 1: sharding encoder weights to TPU")
        text_enc = shard_params(text_enc, mesh, param_dtype=config.param_dtype)
        img_enc = shard_params(img_enc, mesh, param_dtype=config.param_dtype)
        vae = shard_params(vae, mesh, param_dtype=config.param_dtype)

    _log("Phase 1: running encoding")
    prompt_emb, latents, clip_emb = _run_encoding(
        text_enc, img_enc, vae,
        video, token_ids, attention_mask,
        config.has_image_input,
    )
    jax.block_until_ready((prompt_emb, latents, clip_emb))
    _log("Phase 1: encoding complete, deleting encoder weights")

    del text_enc, img_enc, vae
    gc.collect()
    _log("Phase 1: encoders deleted")

    # ---- Phase 2: DiT ----
    _log("Phase 2: creating DiT on CPU")
    rngs = nnx.Rngs(params=jax.random.PRNGKey(0))
    with cpu_ctx:
        dit = _create_dit(config, rngs)

    if quantize_int8:
        from dreamzero_jax.utils.quantize import quantize_model
        _log("Phase 2: quantizing DiT to INT8")
        quantize_model(dit)

    if mesh is not None:
        _log("Phase 2: sharding DiT weights to TPU")
        dit = shard_params(dit, mesh, param_dtype=config.param_dtype)

    _log("Phase 2: running denoising")
    result = _run_denoise_scan(
        dit, config,
        latents, prompt_emb, clip_emb,
        state, embodiment_id,
        num_steps, cfg,
        key=key,
    )
    jax.block_until_ready(result)
    _log("Phase 2: denoising complete")

    del dit
    gc.collect()

    return result
