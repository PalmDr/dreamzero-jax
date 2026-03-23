#!/usr/bin/env python3
"""Run JAX DiT forward using exact inputs from PyTorch standalone script.

Loads the PT output .npz (which contains the inputs), converts weights,
runs the same forward pass, and saves outputs for comparison.

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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pt-output", required=True)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--output", default="jax_dit_output.npz")
    p.add_argument("--num-layers", type=int, default=8)
    args = p.parse_args()

    print(f"{len(jax.devices())} devices")

    pt = np.load(args.pt_output)
    print("PT inputs:", {k: pt[k].shape for k in pt.files})

    x_pt = pt["input_x"]
    timestep_pt = pt["input_timestep"]
    actions_pt = pt["input_actions"]
    state_pt = pt["input_state"]
    B = x_pt.shape[0]
    num_blocks = x_pt.shape[1]

    from dreamzero_jax.models.action_head import CausalWanDiT
    from dreamzero_jax.models.dreamzero import DreamZeroConfig
    from dreamzero_jax.utils.checkpoint import convert_checkpoint, apply_to_model
    from dreamzero_jax.utils.hf_download import load_checkpoint_auto
    from dreamzero_jax.utils.sharding import shard_params, create_mesh
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
    print("Loading checkpoint...")
    raw = load_checkpoint_auto(args.checkpoint_dir)
    converted = convert_checkpoint(raw, cfg)
    del raw

    print("Creating DiT on CPU...")
    with jax.default_device(cpu):
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

    mesh = create_mesh()
    dit = shard_params(dit, mesh, param_dtype=cfg.param_dtype)

    from jax.sharding import NamedSharding, PartitionSpec as P
    rep = NamedSharding(mesh, P())

    # PT standalone uses torch.manual_seed(42) to generate inputs sequentially.
    # Reconstruct the EXACT same sequence here.
    import torch
    torch.manual_seed(42)
    h_patches = x_pt.shape[2]
    w_patches = x_pt.shape[3]

    x_full = torch.randn(B, num_blocks, h_patches, w_patches, 36).numpy() * 0.1
    x_jax = jax.device_put(jnp.array(x_full, dtype=jnp.bfloat16), rep)
    t_jax = jax.device_put(jnp.array(timestep_pt, dtype=jnp.bfloat16), rep)

    ctx = torch.randn(B, 512, 4096).numpy() * 0.01
    ctx_jax = jax.device_put(jnp.array(ctx, dtype=jnp.bfloat16), rep)

    clip = torch.randn(B, 257, 1280).numpy() * 0.01
    clip_jax = jax.device_put(jnp.array(clip, dtype=jnp.bfloat16), rep)

    s_jax = jax.device_put(jnp.array(state_pt, dtype=jnp.float32), rep)
    a_jax = jax.device_put(jnp.array(actions_pt, dtype=jnp.bfloat16), rep)
    e_jax = jax.device_put(jnp.zeros((B,), dtype=jnp.int32), rep)

    print("Running JAX DiT forward...")
    t0 = time.time()
    vid_pred, act_pred = dit(
        x_jax, t_jax, ctx_jax,
        state=s_jax, embodiment_id=e_jax, actions=a_jax,
        clip_emb=clip_jax,
    )
    jax.block_until_ready((vid_pred, act_pred))
    elapsed = time.time() - t0

    vid_np = np.array(vid_pred)
    act_np = np.array(act_pred)
    has_nan = np.any(np.isnan(vid_np)) or np.any(np.isnan(act_np))

    print(f"Done in {elapsed:.1f}s")
    print(f"  video: {vid_np.shape} mean={vid_np.mean():.6f} std={vid_np.std():.6f}")
    print(f"  action: {act_np.shape} mean={act_np.mean():.6f} std={act_np.std():.6f}")
    print(f"  NaN: {has_nan}")

    pt_vid = pt["video_noise_pred"]
    pt_act = pt["action_noise_pred"]
    if vid_np.shape == pt_vid.shape:
        vid_diff = float(np.max(np.abs(vid_np - pt_vid)))
        vid_cos = float(np.sum(vid_np * pt_vid) / (np.linalg.norm(vid_np) * np.linalg.norm(pt_vid) + 1e-12))
        print(f"\n  VIDEO: max_diff={vid_diff:.4f} cos_sim={vid_cos:.6f}")
    else:
        print(f"\n  VIDEO: shape mismatch PT={pt_vid.shape} JAX={vid_np.shape}")

    if act_np.shape == pt_act.shape:
        act_diff = float(np.max(np.abs(act_np - pt_act)))
        act_cos = float(np.sum(act_np * pt_act) / (np.linalg.norm(act_np) * np.linalg.norm(pt_act) + 1e-12))
        print(f"  ACTION: max_diff={act_diff:.4f} cos_sim={act_cos:.6f}")
    else:
        print(f"  ACTION: shape mismatch PT={pt_act.shape} JAX={act_np.shape}")

    np.savez(args.output, video_noise_pred=vid_np, action_noise_pred=act_np)
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
