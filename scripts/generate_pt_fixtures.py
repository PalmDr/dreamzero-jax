#!/usr/bin/env python3
"""Generate PyTorch reference fixtures for numerical validation.

Runs PyTorch model components and saves deterministic input/output pairs
as ``.npz`` files that can later be compared against JAX model outputs.

Usage
-----
With a real checkpoint::

    python scripts/generate_pt_fixtures.py \\
        --pytorch-source /path/to/dreamzero \\
        --checkpoint /path/to/model.pt \\
        --output-dir fixtures/pt_reference/

Small mode (random weights, no checkpoint needed)::

    python scripts/generate_pt_fixtures.py \\
        --pytorch-source /path/to/dreamzero \\
        --output-dir fixtures/pt_reference/ \\
        --small

Subset of components::

    python scripts/generate_pt_fixtures.py \\
        --pytorch-source /path/to/dreamzero \\
        --checkpoint /path/to/model.pt \\
        --components text_encoder,image_encoder
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# Ensure project source is importable
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

from dreamzero_jax.utils.validation import save_fixture, save_manifest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Small-mode config (matches test_dreamzero._small_config roughly)
# ---------------------------------------------------------------------------

SMALL_CONFIG = {
    "dim": 64,
    "in_channels": 4,
    "out_channels": 4,
    "ffn_dim": 128,
    "freq_dim": 32,
    "num_heads": 4,
    "num_layers": 2,
    "patch_size": (1, 2, 2),
    "qk_norm": True,
    "cross_attn_norm": False,
    "text_vocab": 256,
    "text_dim": 64,
    "text_attn_dim": 64,
    "text_ffn_dim": 128,
    "text_num_heads": 4,
    "text_num_layers": 2,
    "text_num_buckets": 32,
    "image_size": 28,
    "image_patch_size": 14,
    "image_dim": 64,
    "image_mlp_ratio": 2,
    "image_out_dim": 32,
    "image_num_heads": 4,
    "image_num_layers": 2,
    "vae_z_dim": 4,
    "vae_base_dim": 32,
    "action_dim": 7,
    "state_dim": 14,
    "action_hidden_size": 32,
    "num_action_per_block": 4,
    "num_state_per_block": 1,
    "num_frames_per_block": 1,
    "max_num_embodiments": 4,
    "scheduler_shift": 5.0,
    "num_train_timesteps": 100,
    "has_image_input": False,
}

ALL_COMPONENTS = [
    "text_encoder",
    "image_encoder",
    "vae_encoder",
    "vae_decoder",
    "dit_block",
    "dit_backbone",
    "category_specific",
    "action_encoder",
    "causal_dit",
    "flow_matching",
]


# ---------------------------------------------------------------------------
# Per-component fixture generators
# ---------------------------------------------------------------------------
# Each function takes (rng, config, model_or_none, output_dir) and saves
# input/output .npz files. The ``model`` argument is the full PyTorch model
# (None when --small and the component needs a sub-module to be separately
# instantiated).
#
# NOTE: These functions require PyTorch and the original DreamZero source
# to be importable. They will fail with clear errors if not available.
# ---------------------------------------------------------------------------


def _ensure_torch():
    """Import and return torch, raising a clear error if unavailable."""
    try:
        import torch
        return torch
    except ImportError:
        raise RuntimeError(
            "PyTorch is required to generate fixtures. "
            "Install it in the PyTorch environment."
        )


def generate_flow_matching(rng: np.random.RandomState, config: dict, output_dir: Path):
    """Flow matching scheduler: pure math, no PyTorch model needed."""
    B = 2
    C = config.get("in_channels", 4)
    spatial = (4, 4)

    sample = rng.randn(B, C, *spatial).astype(np.float32)
    noise = rng.randn(B, C, *spatial).astype(np.float32)
    timesteps = np.array([100.0, 500.0], dtype=np.float32)

    # Flow matching: noisy = (1-sigma) * sample + sigma * noise
    # target = noise - sample
    # These are pure math — we save the inputs and expected outputs
    # computed by numpy so JAX can verify its implementation matches.
    shift = config.get("scheduler_shift", 5.0)
    num_train = config.get("num_train_timesteps", 1000)

    # Compute sigma schedule (matching FlowMatchScheduler logic)
    sigmas = np.linspace(1.0, 0.0, num_train + 1)[:-1]
    sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

    # Map timesteps to sigma
    ts = sigmas * num_train
    sigma_vals = []
    for t in timesteps:
        idx = int(np.argmin(np.abs(ts - t)))
        sigma_vals.append(sigmas[idx])
    sigma_arr = np.array(sigma_vals, dtype=np.float32)

    # Broadcast sigma for add_noise
    sigma_bc = sigma_arr.reshape(B, 1, 1, 1)
    noisy = (1 - sigma_bc) * sample + sigma_bc * noise
    target = noise - sample

    # Transpose to channels-last for JAX convention
    save_fixture(
        output_dir / "flow_matching.npz",
        sample=np.transpose(sample, (0, 2, 3, 1)),
        noise=np.transpose(noise, (0, 2, 3, 1)),
        timesteps=timesteps,
        noisy=np.transpose(noisy, (0, 2, 3, 1)),
        target=np.transpose(target, (0, 2, 3, 1)),
    )
    logger.info("Generated flow_matching fixture")


def generate_text_encoder(
    rng: np.random.RandomState, config: dict, model, output_dir: Path,
):
    """Text encoder: token_ids + mask -> embeddings."""
    torch = _ensure_torch()
    B, L = 2, 16
    text_dim = config["text_dim"]

    token_ids = rng.randint(0, config["text_vocab"], (B, L)).astype(np.int64)
    attention_mask = np.ones((B, L), dtype=np.float32)
    attention_mask[0, L // 2 :] = 0  # Mask out second half for first sample

    with torch.no_grad():
        token_ids_t = torch.from_numpy(token_ids).long()
        mask_t = torch.from_numpy(attention_mask).float()
        # The text encoder interface may vary by PyTorch source version;
        # adjust the call signature as needed.
        embeddings = model.text_encoder(token_ids_t, mask=mask_t)
        embeddings_np = embeddings.cpu().numpy()
        # Apply mask (matching JAX encode_prompt behavior)
        embeddings_np = embeddings_np * attention_mask[:, :, None]

    save_fixture(
        output_dir / "text_encoder.npz",
        token_ids=token_ids.astype(np.int32),
        attention_mask=attention_mask,
        embeddings=embeddings_np,
    )
    logger.info("Generated text_encoder fixture (shape=%s)", embeddings_np.shape)


def generate_image_encoder(
    rng: np.random.RandomState, config: dict, model, output_dir: Path,
):
    """Image encoder: images -> features."""
    torch = _ensure_torch()
    B = 2
    image_size = config["image_size"]

    # channels-last input, will transpose to channels-first for PyTorch
    images_np = rng.randn(B, image_size, image_size, 3).astype(np.float32)
    images_chw = np.transpose(images_np, (0, 3, 1, 2))

    with torch.no_grad():
        images_t = torch.from_numpy(images_chw).float()
        features = model.image_encoder.encode_image(images_t)
        features_np = features.cpu().numpy()

    save_fixture(
        output_dir / "image_encoder.npz",
        images=images_np,
        features=features_np,
    )
    logger.info("Generated image_encoder fixture (shape=%s)", features_np.shape)


def generate_vae_encoder(
    rng: np.random.RandomState, config: dict, model, output_dir: Path,
):
    """VAE encoder: video -> latents."""
    torch = _ensure_torch()
    B, T, H, W = 1, 5, 32, 32

    # channels-last
    video_np = rng.randn(B, T, H, W, 3).astype(np.float32)
    # PyTorch expects (B, C, T, H, W)
    video_chw = np.transpose(video_np, (0, 4, 1, 2, 3))

    with torch.no_grad():
        video_t = torch.from_numpy(video_chw).float()
        latents = model.vae.encode(video_t)
        latents_np = latents.cpu().numpy()
        # Convert back to channels-last
        latents_cl = np.transpose(latents_np, (0, 2, 3, 4, 1))

    save_fixture(
        output_dir / "vae_encoder.npz",
        video=video_np,
        latents=latents_cl,
    )
    logger.info("Generated vae_encoder fixture (shape=%s)", latents_cl.shape)


def generate_vae_decoder(
    rng: np.random.RandomState, config: dict, model, output_dir: Path,
):
    """VAE decoder: latents -> video."""
    torch = _ensure_torch()
    B = 1
    z_dim = config.get("vae_z_dim", 16)
    # Small latent: (B, T', H', W', z_dim)
    latents_np = rng.randn(B, 2, 4, 4, z_dim).astype(np.float32)
    # PyTorch: (B, z_dim, T', H', W')
    latents_chw = np.transpose(latents_np, (0, 4, 1, 2, 3))

    with torch.no_grad():
        latents_t = torch.from_numpy(latents_chw).float()
        video = model.vae.decode(latents_t)
        video_np = video.cpu().numpy()
        # Convert to channels-last
        video_cl = np.transpose(video_np, (0, 2, 3, 4, 1))

    save_fixture(
        output_dir / "vae_decoder.npz",
        latents=latents_np,
        video=video_cl,
    )
    logger.info("Generated vae_decoder fixture (shape=%s)", video_cl.shape)


def generate_dit_block(
    rng: np.random.RandomState, config: dict, model, output_dir: Path,
):
    """Single DiT block forward pass."""
    torch = _ensure_torch()
    B, S = 2, 16
    dim = config["dim"]
    L = 8

    x_np = rng.randn(B, S, dim).astype(np.float32)
    e_np = rng.randn(B, 6, dim).astype(np.float32)
    context_np = rng.randn(B, L, dim).astype(np.float32)

    with torch.no_grad():
        x_t = torch.from_numpy(x_np).float()
        e_t = torch.from_numpy(e_np).float()
        ctx_t = torch.from_numpy(context_np).float()
        # Use first DiT block
        block = model.dit.blocks[0]
        out = block(x_t, e_t, ctx_t)
        out_np = out.cpu().numpy()

    save_fixture(
        output_dir / "dit_block.npz",
        x=x_np,
        e=e_np,
        context=context_np,
        output=out_np,
    )
    logger.info("Generated dit_block fixture (shape=%s)", out_np.shape)


def generate_dit_backbone(
    rng: np.random.RandomState, config: dict, model, output_dir: Path,
):
    """Full DiT backbone (video-only, no action head)."""
    torch = _ensure_torch()
    B = 1
    T = 2
    H, W = 8, 8
    C = config["in_channels"]
    text_dim = config["text_dim"]
    L = 8

    # channels-last
    x_np = rng.randn(B, T, H, W, C).astype(np.float32)
    timestep_np = np.array([500.0], dtype=np.float32)
    context_np = rng.randn(B, L, text_dim).astype(np.float32)

    # PyTorch: (B, C, T, H, W) for video input
    x_chw = np.transpose(x_np, (0, 4, 1, 2, 3))

    with torch.no_grad():
        x_t = torch.from_numpy(x_chw).float()
        t_t = torch.from_numpy(timestep_np).float()
        ctx_t = torch.from_numpy(context_np).float()
        # This call depends on the PyTorch model's interface
        noise_pred = model.dit.forward_video_only(x_t, t_t, ctx_t)
        pred_np = noise_pred.cpu().numpy()
        pred_cl = np.transpose(pred_np, (0, 2, 3, 4, 1))

    save_fixture(
        output_dir / "dit_backbone.npz",
        x=x_np,
        timestep=timestep_np,
        context=context_np,
        noise_pred=pred_cl,
    )
    logger.info("Generated dit_backbone fixture (shape=%s)", pred_cl.shape)


def generate_category_specific(
    rng: np.random.RandomState, config: dict, model, output_dir: Path,
):
    """CategorySpecificLinear layer."""
    torch = _ensure_torch()
    B, S = 2, 8
    in_dim = config.get("state_dim", 14)
    hidden_dim = config.get("action_hidden_size", 32)
    out_dim = config["dim"]

    x_np = rng.randn(B, S, in_dim).astype(np.float32)
    category_ids = np.array([0, 1], dtype=np.int64)

    with torch.no_grad():
        x_t = torch.from_numpy(x_np).float()
        cat_t = torch.from_numpy(category_ids).long()
        # Use the state encoder (CategorySpecificMLP)
        out = model.dit.state_encoder(x_t, cat_t)
        out_np = out.cpu().numpy()

    save_fixture(
        output_dir / "category_specific.npz",
        x=x_np,
        category_ids=category_ids.astype(np.int32),
        output=out_np,
    )
    logger.info("Generated category_specific fixture (shape=%s)", out_np.shape)


def generate_action_encoder(
    rng: np.random.RandomState, config: dict, model, output_dir: Path,
):
    """MultiEmbodimentActionEncoder."""
    torch = _ensure_torch()
    B = 2
    action_dim = config["action_dim"]
    A = config.get("num_action_per_block", 4) * 2  # 2 blocks worth

    actions_np = rng.randn(B, A, action_dim).astype(np.float32)
    timesteps_np = np.array([200.0, 800.0], dtype=np.float32)
    category_ids = np.array([0, 1], dtype=np.int64)

    with torch.no_grad():
        act_t = torch.from_numpy(actions_np).float()
        ts_t = torch.from_numpy(timesteps_np).float()
        cat_t = torch.from_numpy(category_ids).long()
        encoded = model.dit.action_encoder(act_t, ts_t, cat_t)
        encoded_np = encoded.cpu().numpy()

    save_fixture(
        output_dir / "action_encoder.npz",
        actions=actions_np,
        timesteps=timesteps_np,
        category_ids=category_ids.astype(np.int32),
        encoded=encoded_np,
    )
    logger.info("Generated action_encoder fixture (shape=%s)", encoded_np.shape)


def generate_causal_dit(
    rng: np.random.RandomState, config: dict, model, output_dir: Path,
):
    """Full CausalWanDiT forward pass."""
    torch = _ensure_torch()
    B = 1
    T = 2
    H, W = 8, 8
    C = config["in_channels"]
    text_dim = config["text_dim"]
    action_dim = config["action_dim"]
    state_dim = config.get("state_dim", 14)
    L = 8

    # Inputs (channels-last for saving)
    x_np = rng.randn(B, T, H, W, C).astype(np.float32)
    timestep_np = np.array([500.0], dtype=np.float32)
    context_np = rng.randn(B, L, text_dim).astype(np.float32)

    # Compute num_blocks from spatial/patch config
    patch_size = config.get("patch_size", (1, 2, 2))
    f = T // patch_size[0]
    num_blocks = f // config.get("num_frames_per_block", 1)
    n_actions_per_block = config.get("num_action_per_block", 4)
    total_actions = num_blocks * n_actions_per_block

    state_np = rng.randn(B, num_blocks, state_dim).astype(np.float32)
    actions_np = rng.randn(B, total_actions, action_dim).astype(np.float32)
    embodiment_id = np.array([0], dtype=np.int64)

    # Clean video for teacher forcing
    clean_x_np = rng.randn(B, T, H, W, C).astype(np.float32)

    # PyTorch: (B, C, T, H, W)
    x_chw = np.transpose(x_np, (0, 4, 1, 2, 3))
    clean_chw = np.transpose(clean_x_np, (0, 4, 1, 2, 3))

    with torch.no_grad():
        x_t = torch.from_numpy(x_chw).float()
        t_t = torch.from_numpy(timestep_np).float()
        ctx_t = torch.from_numpy(context_np).float()
        state_t = torch.from_numpy(state_np).float()
        act_t = torch.from_numpy(actions_np).float()
        emb_t = torch.from_numpy(embodiment_id).long()
        clean_t = torch.from_numpy(clean_chw).float()

        vid_pred, act_pred = model.dit(
            x_t, t_t, ctx_t, state_t, emb_t, act_t,
            timestep_action=t_t, clean_x=clean_t,
        )
        vid_np = vid_pred.cpu().numpy()
        act_np = act_pred.cpu().numpy()
        # Video: convert to channels-last
        vid_cl = np.transpose(vid_np, (0, 2, 3, 4, 1))

    save_fixture(
        output_dir / "causal_dit.npz",
        x=x_np,
        timestep=timestep_np,
        context=context_np,
        state=state_np,
        actions=actions_np,
        embodiment_id=embodiment_id.astype(np.int32),
        clean_x=clean_x_np,
        video_pred=vid_cl,
        action_pred=act_np,
    )
    logger.info(
        "Generated causal_dit fixture (video=%s, action=%s)",
        vid_cl.shape,
        act_np.shape,
    )


# ---------------------------------------------------------------------------
# Generator registry
# ---------------------------------------------------------------------------

# Components that need a PyTorch model
MODEL_COMPONENTS = {
    "text_encoder": generate_text_encoder,
    "image_encoder": generate_image_encoder,
    "vae_encoder": generate_vae_encoder,
    "vae_decoder": generate_vae_decoder,
    "dit_block": generate_dit_block,
    "dit_backbone": generate_dit_backbone,
    "category_specific": generate_category_specific,
    "action_encoder": generate_action_encoder,
    "causal_dit": generate_causal_dit,
}

# Components that are pure math (no model needed)
PURE_COMPONENTS = {
    "flow_matching": generate_flow_matching,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _save_small_weights(model, output_dir: Path):
    """Save PyTorch model weights as numpy arrays for later JAX loading."""
    torch = _ensure_torch()
    weights_dir = output_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    state_dict = model.state_dict()
    for key, tensor in state_dict.items():
        arr = tensor.cpu().numpy()
        safe_key = key.replace(".", "__")
        np.save(weights_dir / f"{safe_key}.npy", arr)

    logger.info("Saved %d weight tensors to %s", len(state_dict), weights_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate PyTorch reference fixtures for numerical validation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--pytorch-source",
        type=Path,
        help="Path to the original DreamZero repo (added to sys.path).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="PyTorch checkpoint file (.pt or .safetensors).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("fixtures/pt_reference"),
        help="Fixture output directory (default: fixtures/pt_reference/).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config overrides as a JSON string.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Numpy random seed (default: 42).",
    )
    parser.add_argument(
        "--small",
        action="store_true",
        help="Use small config with random weights (no checkpoint needed).",
    )
    parser.add_argument(
        "--components",
        type=str,
        default=None,
        help="Comma-separated subset of components to generate.",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Determine config
    if args.small:
        config = dict(SMALL_CONFIG)
    else:
        config = {}

    if args.config:
        overrides = json.loads(args.config)
        config.update(overrides)

    # Determine which components to generate
    if args.components:
        components = [c.strip() for c in args.components.split(",")]
        for c in components:
            if c not in ALL_COMPONENTS:
                parser.error(f"Unknown component: {c}. Valid: {ALL_COMPONENTS}")
    else:
        components = list(ALL_COMPONENTS)

    # Setup
    rng = np.random.RandomState(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate pure-math components (no model needed)
    generated = []
    for comp in components:
        if comp in PURE_COMPONENTS:
            PURE_COMPONENTS[comp](rng, config, args.output_dir)
            generated.append(comp)

    # Generate model-dependent components
    model_comps = [c for c in components if c in MODEL_COMPONENTS]
    if model_comps:
        if args.pytorch_source:
            sys.path.insert(0, str(args.pytorch_source))

        if args.small:
            logger.info(
                "Small mode: PyTorch model must be instantiated with matching config. "
                "This requires the original DreamZero source to be importable."
            )
            # In small mode, the user must still have the PyTorch source available.
            # We just use random weights instead of loading a checkpoint.
            logger.info(
                "Attempting to import PyTorch DreamZero model from source..."
            )
            try:
                _ensure_torch()
                # The import path depends on the original repo structure.
                # Users may need to adjust this based on their setup.
                logger.warning(
                    "Small mode model instantiation depends on the original "
                    "DreamZero PyTorch source. Please ensure --pytorch-source "
                    "points to the correct location."
                )
                # Placeholder: users will need to adapt this block to their
                # specific PyTorch source structure.
                logger.error(
                    "Automatic small-mode model instantiation is not yet "
                    "implemented. Please provide --checkpoint with the PyTorch "
                    "model, or manually instantiate the model and call the "
                    "individual generate_* functions."
                )
                sys.exit(1)
            except RuntimeError as e:
                logger.error(str(e))
                sys.exit(1)
        else:
            if not args.checkpoint:
                parser.error(
                    "--checkpoint is required for model-dependent components "
                    "(or use --small with PyTorch source)."
                )

            logger.info("Loading PyTorch checkpoint: %s", args.checkpoint)
            try:
                import torch
                # Load the full PyTorch model.
                # The exact loading mechanism depends on how the checkpoint
                # was saved. Common patterns:
                checkpoint = torch.load(
                    str(args.checkpoint), map_location="cpu", weights_only=False,
                )
                if isinstance(checkpoint, dict) and "model" in checkpoint:
                    model = checkpoint["model"]
                elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    # Need to instantiate model and load state dict.
                    # This requires the PyTorch source to be importable.
                    logger.error(
                        "state_dict-only checkpoints require model instantiation. "
                        "Please provide the --pytorch-source path and ensure "
                        "the model class is importable."
                    )
                    sys.exit(1)
                else:
                    model = checkpoint

                model.eval()
                logger.info("Model loaded successfully.")
            except Exception as e:
                logger.error("Failed to load checkpoint: %s", e)
                sys.exit(1)

            for comp in model_comps:
                try:
                    MODEL_COMPONENTS[comp](rng, config, model, args.output_dir)
                    generated.append(comp)
                except Exception as e:
                    logger.error("Failed to generate %s: %s", comp, e)

            if args.small:
                _save_small_weights(model, args.output_dir)

    # Collect version info
    versions: dict[str, str] = {"numpy": np.__version__}
    try:
        import torch
        versions["torch"] = torch.__version__
    except ImportError:
        pass

    # Write manifest
    save_manifest(
        args.output_dir,
        config=config,
        fixtures=generated,
        seed=args.seed,
        versions=versions,
    )

    logger.info(
        "Done. Generated %d/%d fixtures in %s",
        len(generated),
        len(components),
        args.output_dir,
    )


if __name__ == "__main__":
    main()
