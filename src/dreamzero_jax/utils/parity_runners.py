"""Per-component JAX runners for parity validation.

Each runner takes a model, config, and numpy RandomState, executes
one model component, and returns a dict with inputs, output, and timing.
"""

from __future__ import annotations

import time

import numpy as np


def run_text_encoder(model, config, rng):
    """Run text encoder and return output."""
    import jax.numpy as jnp

    B, L = 2, 32
    token_ids = jnp.array(rng.randint(0, config.text_vocab, (B, L)), dtype=jnp.int32)
    mask = jnp.ones((B, L), dtype=jnp.float32)

    t0 = time.monotonic()
    out = model.encode_prompt(token_ids, mask)
    elapsed = (time.monotonic() - t0) * 1000

    return {
        "inputs": {"token_ids": np.asarray(token_ids), "mask": np.asarray(mask)},
        "output": np.asarray(out),
        "elapsed_ms": elapsed,
    }


def run_image_encoder(model, config, rng):
    """Run image encoder and return output."""
    import jax.numpy as jnp

    B = 2
    images = jnp.array(
        rng.randn(B, config.image_size, config.image_size, 3).astype(np.float32)
    )

    t0 = time.monotonic()
    out = model.encode_image(images)
    elapsed = (time.monotonic() - t0) * 1000

    return {
        "inputs": {"images": np.asarray(images)},
        "output": np.asarray(out),
        "elapsed_ms": elapsed,
    }


def run_vae_encoder(model, config, rng):
    """Run VAE encoder and return output."""
    import jax.numpy as jnp

    B, T, H, W = 1, 5, 32, 32
    video = jnp.array(rng.randn(B, T, H, W, 3).astype(np.float32))

    t0 = time.monotonic()
    out = model.encode_video(video)
    elapsed = (time.monotonic() - t0) * 1000

    return {
        "inputs": {"video": np.asarray(video)},
        "output": np.asarray(out),
        "elapsed_ms": elapsed,
    }


def run_vae_decoder(model, config, rng):
    """Run VAE decoder and return output."""
    import jax.numpy as jnp

    B = 1
    latents = jnp.array(
        rng.randn(B, 2, 4, 4, config.vae_z_dim).astype(np.float32)
    )

    t0 = time.monotonic()
    out = model.vae.decode(latents)
    elapsed = (time.monotonic() - t0) * 1000

    return {
        "inputs": {"latents": np.asarray(latents)},
        "output": np.asarray(out),
        "elapsed_ms": elapsed,
    }


def run_dit_block(model, config, rng):
    """Run a single DiT block and return output."""
    import jax.numpy as jnp
    from dreamzero_jax.nn.embed import WanRoPE3D

    B, S = 2, 16
    dim = config.dim
    head_dim = dim // config.num_heads

    x = jnp.array(rng.randn(B, S, dim).astype(np.float32))
    e = jnp.array(rng.randn(B, 6, dim).astype(np.float32))
    ctx = jnp.array(rng.randn(B, 8, dim).astype(np.float32))

    rope = WanRoPE3D(head_dim)
    freqs = rope(2, 2, 4)

    block = model.dit.blocks[0]

    t0 = time.monotonic()
    out = block(x, e, ctx, freqs)
    elapsed = (time.monotonic() - t0) * 1000

    return {
        "inputs": {"x": np.asarray(x), "e": np.asarray(e), "ctx": np.asarray(ctx)},
        "output": np.asarray(out),
        "elapsed_ms": elapsed,
    }


def run_dit_full(model, config, rng):
    """Run the full CausalWanDiT and return output."""
    import jax.numpy as jnp

    B = 1
    T, H, W = 2, 8, 8
    C = config.in_channels
    text_dim = config.text_dim
    action_dim = config.action_dim
    state_dim = config.state_dim

    x = jnp.array(rng.randn(B, T, H, W, C).astype(np.float32))
    timestep = jnp.array([500.0])
    context = jnp.array(rng.randn(B, 8, text_dim).astype(np.float32))

    patch_size = config.patch_size
    f = T // patch_size[0]
    num_blocks = f // config.num_frames_per_block
    total_actions = num_blocks * config.num_action_per_block

    state = jnp.array(rng.randn(B, num_blocks, state_dim).astype(np.float32))
    actions = jnp.array(rng.randn(B, total_actions, action_dim).astype(np.float32))
    embodiment_id = jnp.array([0], dtype=jnp.int32)
    clean_x = jnp.array(rng.randn(B, T, H, W, C).astype(np.float32))

    t0 = time.monotonic()
    vid_pred, act_pred = model.dit(
        x, timestep, context, state, embodiment_id, actions,
        timestep_action=timestep, clean_x=clean_x,
    )
    elapsed = (time.monotonic() - t0) * 1000

    return {
        "inputs": {
            "x": np.asarray(x),
            "timestep": np.asarray(timestep),
            "context": np.asarray(context),
            "state": np.asarray(state),
            "actions": np.asarray(actions),
            "embodiment_id": np.asarray(embodiment_id),
            "clean_x": np.asarray(clean_x),
        },
        "output": np.asarray(vid_pred),
        "action_output": np.asarray(act_pred),
        "elapsed_ms": elapsed,
    }


def run_action_encoder(model, config, rng):
    """Run action encoder and return output."""
    import jax.numpy as jnp

    B = 2
    A = config.num_action_per_block * 2
    action_dim = config.action_dim

    actions = jnp.array(rng.randn(B, A, action_dim).astype(np.float32))
    timesteps = jnp.array([200.0, 800.0])
    category_ids = jnp.array([0, 1], dtype=jnp.int32)

    t0 = time.monotonic()
    out = model.dit.action_encoder(actions, timesteps, category_ids)
    elapsed = (time.monotonic() - t0) * 1000

    return {
        "inputs": {
            "actions": np.asarray(actions),
            "timesteps": np.asarray(timesteps),
            "category_ids": np.asarray(category_ids),
        },
        "output": np.asarray(out),
        "elapsed_ms": elapsed,
    }


def run_state_encoder(model, config, rng):
    """Run state encoder and return output."""
    import jax.numpy as jnp

    B, S = 2, 4
    state_dim = config.state_dim

    x = jnp.array(rng.randn(B, S, state_dim).astype(np.float32))
    category_ids = jnp.array([0, 1], dtype=jnp.int32)

    t0 = time.monotonic()
    out = model.dit.state_encoder(x, category_ids)
    elapsed = (time.monotonic() - t0) * 1000

    return {
        "inputs": {"x": np.asarray(x), "category_ids": np.asarray(category_ids)},
        "output": np.asarray(out),
        "elapsed_ms": elapsed,
    }


COMPONENT_RUNNERS = {
    "text_encoder": run_text_encoder,
    "image_encoder": run_image_encoder,
    "vae_encoder": run_vae_encoder,
    "vae_decoder": run_vae_decoder,
    "dit_block": run_dit_block,
    "dit_full": run_dit_full,
    "action_encoder": run_action_encoder,
    "state_encoder": run_state_encoder,
}

COMPONENT_ORDER = [
    "text_encoder",
    "image_encoder",
    "vae_encoder",
    "vae_decoder",
    "dit_block",
    "dit_full",
    "action_encoder",
    "state_encoder",
]


def _pt_text_encoder(pt_model, inputs):
    """PyTorch text encoder forward."""
    import torch
    token_ids = torch.from_numpy(inputs["token_ids"]).long()
    mask = torch.from_numpy(inputs["mask"]).float()
    with torch.no_grad():
        out = pt_model.text_encoder(token_ids, mask=mask)
    return (out * mask[:, :, None]).cpu().numpy()


def _pt_image_encoder(pt_model, inputs):
    """PyTorch image encoder forward (channels-first conversion)."""
    import torch
    images = np.transpose(inputs["images"], (0, 3, 1, 2))
    with torch.no_grad():
        out = pt_model.image_encoder.encode_image(
            torch.from_numpy(images).float()
        )
    return out.cpu().numpy()


def _pt_vae_encoder(pt_model, inputs):
    """PyTorch VAE encoder (BTHWC -> BCTHW -> BTHWC)."""
    import torch
    video = np.transpose(inputs["video"], (0, 4, 1, 2, 3))
    with torch.no_grad():
        out = pt_model.vae.encode(torch.from_numpy(video).float())
    return np.transpose(out.cpu().numpy(), (0, 2, 3, 4, 1))


def _pt_vae_decoder(pt_model, inputs):
    """PyTorch VAE decoder (BTHWC -> BCTHW -> BTHWC)."""
    import torch
    latents = np.transpose(inputs["latents"], (0, 4, 1, 2, 3))
    with torch.no_grad():
        out = pt_model.vae.decode(torch.from_numpy(latents).float())
    return np.transpose(out.cpu().numpy(), (0, 2, 3, 4, 1))


def _pt_dit_block(pt_model, inputs):
    """PyTorch single DiT block forward."""
    import torch
    with torch.no_grad():
        out = pt_model.dit.blocks[0](
            torch.from_numpy(inputs["x"]).float(),
            torch.from_numpy(inputs["e"]).float(),
            torch.from_numpy(inputs["ctx"]).float(),
        )
    return out.cpu().numpy()


def _pt_action_encoder(pt_model, inputs):
    """PyTorch action encoder forward."""
    import torch
    with torch.no_grad():
        out = pt_model.dit.action_encoder(
            torch.from_numpy(inputs["actions"]).float(),
            torch.from_numpy(inputs["timesteps"]).float(),
            torch.from_numpy(inputs["category_ids"]).long(),
        )
    return out.cpu().numpy()


def _pt_state_encoder(pt_model, inputs):
    """PyTorch state encoder forward."""
    import torch
    with torch.no_grad():
        out = pt_model.dit.state_encoder(
            torch.from_numpy(inputs["x"]).float(),
            torch.from_numpy(inputs["category_ids"]).long(),
        )
    return out.cpu().numpy()


_PT_DISPATCH = {
    "text_encoder": _pt_text_encoder,
    "image_encoder": _pt_image_encoder,
    "vae_encoder": _pt_vae_encoder,
    "vae_decoder": _pt_vae_decoder,
    "dit_block": _pt_dit_block,
    "action_encoder": _pt_action_encoder,
    "state_encoder": _pt_state_encoder,
}


def run_pytorch_component(pt_model, comp_name, inputs, config):
    """Run a single PyTorch component and return numpy output."""
    fn = _PT_DISPATCH.get(comp_name)
    if fn is None:
        return None
    return fn(pt_model, inputs)
