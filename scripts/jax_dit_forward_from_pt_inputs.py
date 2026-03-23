#!/usr/bin/env python3
"""Run JAX DiT forward using exact inputs from PyTorch standalone script.

Loads the PT output .npz (which contains the inputs + intermediates),
converts weights, runs the same forward pass with intermediate captures,
and compares each stage to find where divergence starts.

Usage:
    uv run python scripts/jax_dit_forward_from_pt_inputs.py \
        --pt-output pt_output.npz \
        --checkpoint-dir checkpoints/DreamZero-DROID \
        --output jax_dit_output.npz
"""
import argparse
import time
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

jax.config.update("jax_default_matmul_precision", "float32")


def compare_arrays(name, jax_arr, pt_arr):
    """Print per-stage comparison metrics."""
    if jax_arr.shape != pt_arr.shape:
        print(f"  {name}: SHAPE MISMATCH jax={jax_arr.shape} pt={pt_arr.shape}")
        return
    j = jax_arr.astype(np.float32).ravel()
    p = pt_arr.astype(np.float32).ravel()
    max_diff = float(np.max(np.abs(j - p)))
    j_norm = np.linalg.norm(j)
    p_norm = np.linalg.norm(p)
    cos = float(np.dot(j, p) / (j_norm * p_norm + 1e-12))
    rel_diff = max_diff / (np.mean(np.abs(p)) + 1e-12)
    print(f"  {name}: cos={cos:.6f}  max_diff={max_diff:.4f}  "
          f"rel_diff={rel_diff:.4f}  shapes={jax_arr.shape}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pt-output", required=True)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--output", default="jax_dit_output.npz")
    p.add_argument("--num-layers", type=int, default=8)
    args = p.parse_args()

    print(f"{len(jax.devices())} devices")

    pt = np.load(args.pt_output)
    print("PT npz keys:", sorted(pt.files))

    from dreamzero_jax.models.action_head import CausalWanDiT
    from dreamzero_jax.models.causal_ops import (
        causal_block_forward,
        head_forward_per_token,
        per_token_time_conditioning,
    )
    from dreamzero_jax.models.dit import unpatchify
    from dreamzero_jax.nn.embed import rope_params_polar_1d
    from dreamzero_jax.models.dreamzero import DreamZeroConfig
    from dreamzero_jax.utils.checkpoint import convert_checkpoint, apply_to_model
    from dreamzero_jax.utils.hf_download import load_checkpoint_auto
    import ml_dtypes

    cfg = DreamZeroConfig(
        dim=5120, ffn_dim=13824, num_heads=40, num_layers=args.num_layers,
        freq_dim=256, text_dim=4096, patch_size=(1, 2, 2),
        in_channels=36, out_channels=16, has_image_input=True,
        image_dim=1280, image_out_dim=1024, cross_attn_norm=True,
        action_dim=32, state_dim=64, max_num_embodiments=1,
        dtype=jnp.bfloat16, param_dtype=jnp.bfloat16,
    )

    cpu = jax.devices("cpu")[0]
    print("Loading checkpoint + creating DiT on CPU...")
    with jax.default_device(cpu):
        raw = load_checkpoint_auto(args.checkpoint_dir)
        converted = convert_checkpoint(raw, cfg)
        del raw
        dit = CausalWanDiT(
            dim=cfg.dim, in_channels=cfg.in_channels, out_channels=cfg.out_channels,
            ffn_dim=cfg.ffn_dim, freq_dim=cfg.freq_dim, text_dim=cfg.text_dim,
            num_heads=cfg.num_heads, num_layers=cfg.num_layers,
            patch_size=cfg.patch_size, has_image_input=True,
            image_dim=cfg.image_dim, qk_norm=True, cross_attn_norm=True,
            action_dim=cfg.action_dim, state_dim=cfg.state_dim,
            action_hidden_size=cfg.action_hidden_size,
            num_action_per_block=cfg.num_action_per_block,
            num_state_per_block=cfg.num_state_per_block,
            num_frames_per_block=cfg.num_frames_per_block,
            max_num_embodiments=cfg.max_num_embodiments,
            dtype=cfg.dtype, param_dtype=cfg.param_dtype,
            rngs=nnx.Rngs(0),
        )
        dit_keys = {k[1:]: v for k, v in converted.items() if k[0] == "dit"}
        for k in list(dit_keys.keys()):
            arr = np.asarray(dit_keys[k])
            if arr.dtype in (np.float32, np.float64):
                dit_keys[k] = jnp.array(arr.astype(ml_dtypes.bfloat16))
            else:
                dit_keys[k] = jnp.array(arr)
        applied, missing, extra = apply_to_model(dit, dit_keys)
        print(f"  {applied} applied, {len(missing)} missing")
    del converted, dit_keys

    rep = jax.sharding.SingleDeviceSharding(jax.devices()[0])

    # -----------------------------------------------------------------------
    # Load ALL inputs from PT npz -- no torch.randn reconstruction needed.
    # This guarantees bit-exact same inputs regardless of RNG order.
    # -----------------------------------------------------------------------
    x_np = pt["input_x"]
    timestep_np = pt["input_timestep"]
    actions_np = pt["input_actions"]
    state_np = pt["input_state"]
    timestep_action_np = pt["input_timestep_action"]
    embodiment_id_np = pt["input_embodiment_id"]

    has_clean = "input_clean_x" in pt.files
    has_y = "input_y" in pt.files
    has_context = "input_context" in pt.files
    has_clip = "input_clip_emb" in pt.files

    x_jax = jax.device_put(jnp.array(x_np, dtype=jnp.bfloat16), rep)
    t_jax = jax.device_put(jnp.array(timestep_np, dtype=jnp.bfloat16), rep)
    a_jax = jax.device_put(jnp.array(actions_np, dtype=jnp.bfloat16), rep)
    s_jax = jax.device_put(jnp.array(state_np, dtype=jnp.float32), rep)
    ta_jax = jax.device_put(jnp.array(timestep_action_np, dtype=jnp.bfloat16), rep)
    e_jax = jax.device_put(jnp.array(embodiment_id_np, dtype=jnp.int32), rep)

    clean_jax = None
    if has_clean:
        clean_jax = jax.device_put(
            jnp.array(pt["input_clean_x"], dtype=jnp.bfloat16), rep)

    y_jax = None
    if has_y:
        y_jax = jax.device_put(
            jnp.array(pt["input_y"], dtype=jnp.bfloat16), rep)

    ctx_jax = None
    if has_context:
        ctx_jax = jax.device_put(
            jnp.array(pt["input_context"], dtype=jnp.bfloat16), rep)

    clip_jax = None
    if has_clip:
        clip_jax = jax.device_put(
            jnp.array(pt["input_clip_emb"], dtype=jnp.bfloat16), rep)

    B = x_np.shape[0]
    print(f"\nInputs loaded from PT npz:")
    print(f"  x={x_np.shape} t={timestep_np.shape} actions={actions_np.shape}")
    print(f"  state={state_np.shape} clean_x={has_clean} y={has_y}")
    print(f"  context={has_context} clip_emb={has_clip}")

    # -----------------------------------------------------------------------
    # Run forward with intermediate captures (manual, matching CausalWanDiT.__call__)
    # -----------------------------------------------------------------------
    print("\nRunning JAX DiT forward with intermediate captures...")
    t0 = time.time()

    intermediates = {}

    if y_jax is not None:
        x_in = jnp.concatenate([x_jax, y_jax], axis=-1)
    else:
        x_in = x_jax

    x_patched = dit.patch_embedding.proj(x_in)
    _, f, h, w, _ = x_patched.shape
    seq_len = f * h * w
    frame_seqlen = h * w
    x_flat = x_patched.reshape(B, seq_len, dit.dim)
    intermediates["inter_patched_video"] = np.array(x_flat)

    video_freqs = dit.rope(f, h, w)[:, None, :]
    action_freqs = rope_params_polar_1d(1024 * 10, dit.head_dim)
    state_freqs = rope_params_polar_1d(1024, dit.head_dim)

    action_emb = dit.action_encoder(a_jax, ta_jax, e_jax)
    state_emb = dit.state_encoder(s_jax, e_jax)

    action_length = action_emb.shape[1]
    action_register = jnp.concatenate([action_emb, state_emb], axis=1)
    action_register_length = action_register.shape[1]

    x_seq = jnp.concatenate([x_flat, action_register], axis=1)

    if has_clean:
        if y_jax is not None:
            clean_in = jnp.concatenate([clean_jax, y_jax], axis=-1)
        else:
            clean_in = clean_jax
        clean_patched = dit.patch_embedding.proj(clean_in)
        clean_flat = clean_patched.reshape(B, seq_len, dit.dim)
        full_seq = jnp.concatenate([clean_flat, x_seq], axis=1)
    else:
        full_seq = x_seq

    ts_video = jnp.broadcast_to(t_jax[:, None], (B, seq_len))
    ts_action = (
        jnp.broadcast_to(ta_jax[:, None], (B, action_length))
        if ta_jax.ndim == 1
        else ta_jax
    )
    stride = ts_action.shape[1] // state_emb.shape[1]
    ts_state = ts_action[:, ::stride]
    ts_full = jnp.concatenate([ts_video, ts_action, ts_state], axis=1)

    if has_clean:
        ts_clean = jnp.zeros((B, seq_len), dtype=t_jax.dtype)
        ts_full = jnp.concatenate([ts_clean, ts_full], axis=1)

    total_L = full_seq.shape[1]
    e_flat, e0_flat = per_token_time_conditioning(
        ts_full.reshape(-1), dit.freq_dim,
        dit.time_embedding, dit.time_projection, dit.dim,
    )
    e_tokens = e_flat.reshape(B, total_L, dit.dim)
    e0_tokens = e0_flat.reshape(B, total_L, 6, dit.dim)
    intermediates["inter_e_tokens"] = np.array(e_tokens)

    ctx = dit.text_embedding(ctx_jax)
    if dit.has_image_input and clip_jax is not None:
        img_ctx = dit.img_emb(clip_jax)
        ctx = jnp.concatenate([img_ctx, ctx], axis=1)

    intermediates["inter_seq_pre_block"] = np.array(full_seq)

    is_tf = has_clean

    for i, block in enumerate(dit.blocks):
        print(f"    block {i}/{len(dit.blocks)}...", end="", flush=True)
        tb = time.time()
        full_seq = causal_block_forward(
            block, full_seq, e0_tokens, ctx,
            video_freqs, action_freqs, state_freqs,
            action_register_length, frame_seqlen,
            dit.num_frames_per_block, dit.num_action_per_block,
            dit.num_state_per_block, is_tf,
        )
        jax.block_until_ready(full_seq)
        print(f" {time.time() - tb:.1f}s")
        if i == 0:
            intermediates["inter_seq_post_block0"] = np.array(full_seq)

    if has_clean:
        full_seq = full_seq[:, seq_len:]

    video_pred = full_seq[:, :seq_len]
    action_pred = full_seq[:, seq_len:seq_len + action_length]
    intermediates["inter_action_pre_decode"] = np.array(action_pred)

    offset = seq_len if has_clean else 0
    e_video = e_tokens[:, offset:offset + seq_len]
    video_noise_pred = head_forward_per_token(dit.head, video_pred, e_video)
    video_noise_pred = unpatchify(
        video_noise_pred, (f, h, w), dit.patch_size, dit.out_channels,
    )
    action_noise_pred = dit.action_decoder(action_pred, e_jax)

    jax.block_until_ready((video_noise_pred, action_noise_pred))
    elapsed = time.time() - t0

    vid_np = np.array(video_noise_pred)
    act_np = np.array(action_noise_pred)
    has_nan = np.any(np.isnan(vid_np)) or np.any(np.isnan(act_np))

    print(f"\nDone in {elapsed:.1f}s")
    print(f"  video: {vid_np.shape} mean={float(np.nanmean(vid_np)):.6f} "
          f"std={float(np.nanstd(vid_np)):.6f}")
    print(f"  action: {act_np.shape} mean={float(np.nanmean(act_np)):.6f} "
          f"std={float(np.nanstd(act_np)):.6f}")
    print(f"  NaN: {has_nan}")

    # -----------------------------------------------------------------------
    # Compare final outputs
    # -----------------------------------------------------------------------
    print("\n--- Final output comparison ---")
    pt_vid = pt["video_noise_pred"]
    pt_act = pt["action_noise_pred"]
    compare_arrays("VIDEO", vid_np, pt_vid)
    compare_arrays("ACTION", act_np, pt_act)

    # -----------------------------------------------------------------------
    # Compare intermediates stage by stage
    # -----------------------------------------------------------------------
    inter_keys = [
        "inter_patched_video",
        "inter_e_tokens",
        "inter_seq_pre_block",
        "inter_seq_post_block0",
        "inter_action_pre_decode",
    ]
    print("\n--- Intermediate comparison (stage by stage) ---")
    for key in inter_keys:
        if key in pt.files and key in intermediates:
            compare_arrays(key, intermediates[key], pt[key])
        elif key not in pt.files:
            print(f"  {key}: MISSING from PT npz")
        else:
            print(f"  {key}: MISSING from JAX intermediates")

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    save_dict = {
        "video_noise_pred": vid_np,
        "action_noise_pred": act_np,
    }
    save_dict.update(intermediates)
    np.savez(args.output, **save_dict)
    print(f"\nSaved {args.output} ({len(save_dict)} arrays)")


if __name__ == "__main__":
    main()
