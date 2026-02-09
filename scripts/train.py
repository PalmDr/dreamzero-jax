#!/usr/bin/env python3
"""Distributed training script for DreamZero on TPU/GPU.

Usage:
    uv run python scripts/train.py --dataset lerobot/aloha --output-dir runs/exp01
    uv run python scripts/train.py --config configs/base.yaml --checkpoint runs/exp01/ckpt
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import nnx

from dreamzero_jax.models.dreamzero import DreamZero, DreamZeroConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train DreamZero on TPU/GPU with distributed data parallelism.",
    )

    # Data / IO
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML or JSON config file. Values override CLI defaults.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="HuggingFace dataset repo ID (e.g. lerobot/aloha).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for checkpoints, logs, and metrics.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume from.",
    )

    # Training hyper-parameters
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Peak learning rate.")
    parser.add_argument("--warmup-steps", type=int, default=1000, help="LR warmup steps.")
    parser.add_argument("--max-steps", type=int, default=100_000, help="Maximum training steps.")
    parser.add_argument("--log-every", type=int, default=10, help="Log metrics every N steps.")
    parser.add_argument(
        "--save-every", type=int, default=1000, help="Save checkpoint every N steps."
    )
    parser.add_argument(
        "--eval-every", type=int, default=500, help="Run evaluation every N steps."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "bfloat16"],
        help="Computation dtype.",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="AdamW weight decay.",
    )

    # Distributed
    parser.add_argument(
        "--mesh-shape",
        type=str,
        default=None,
        help=(
            "Mesh shape as 'data,model' (e.g. '4,1'). "
            "Defaults to (num_devices, 1) for pure data parallelism."
        ),
    )

    return parser.parse_args()


def load_config_file(path: str) -> dict:
    """Load a YAML or JSON config file and return a dict."""
    path = Path(path)
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required for YAML config files: pip install pyyaml")
        return yaml.safe_load(text)
    elif path.suffix == ".json":
        return json.loads(text)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}. Use .yaml or .json.")


def build_model_config(overrides: dict | None = None) -> DreamZeroConfig:
    """Build a DreamZeroConfig, optionally applying overrides from a config file."""
    kwargs = {}
    if overrides and "model" in overrides:
        model_cfg = overrides["model"]
        # Map config keys to DreamZeroConfig fields
        for key, value in model_cfg.items():
            if hasattr(DreamZeroConfig, key):
                kwargs[key] = value
    return DreamZeroConfig(**kwargs)


# ---------------------------------------------------------------------------
# Mesh / sharding
# ---------------------------------------------------------------------------


def create_mesh(mesh_shape: str | None = None) -> jax.sharding.Mesh:
    """Create a device mesh for distributed training.

    Args:
        mesh_shape: Comma-separated string 'data,model' (e.g. '4,1').
            Defaults to (num_devices, 1) for pure data parallelism.

    Returns:
        A ``jax.sharding.Mesh`` with axes ``('data', 'model')``.
    """
    num_devices = jax.device_count()
    if mesh_shape is not None:
        parts = [int(x) for x in mesh_shape.split(",")]
        if len(parts) != 2:
            raise ValueError(f"mesh-shape must be 'data,model', got '{mesh_shape}'")
        dp, mp = parts
    else:
        dp, mp = num_devices, 1

    if dp * mp != num_devices:
        raise ValueError(
            f"Mesh shape ({dp}, {mp}) = {dp * mp} does not match "
            f"device count {num_devices}."
        )

    devices = np.array(jax.devices()).reshape(dp, mp)
    return jax.sharding.Mesh(devices, axis_names=("data", "model"))


def shard_batch(batch: dict, mesh: jax.sharding.Mesh) -> dict:
    """Shard a batch dict across the 'data' axis of the mesh.

    Each leaf array in *batch* is replicated along 'model' and sharded
    along the first (batch) axis over 'data'.
    """
    data_sharding = jax.sharding.NamedSharding(
        mesh,
        jax.sharding.PartitionSpec("data"),
    )
    return jax.tree.map(lambda x: jax.device_put(x, data_sharding), batch)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


def create_optimizer(
    learning_rate: float,
    warmup_steps: int,
    max_steps: int,
    weight_decay: float,
    max_grad_norm: float,
) -> optax.GradientTransformation:
    """Create AdamW optimizer with warmup cosine schedule and gradient clipping.

    Schedule:
        1. Linear warmup from 0 to ``learning_rate`` over ``warmup_steps``.
        2. Cosine decay from ``learning_rate`` to 0 over remaining steps.
    """
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=max_steps,
        end_value=0.0,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adamw(learning_rate=schedule, weight_decay=weight_decay),
    )

    return optimizer


def get_current_lr(step: int, args: argparse.Namespace) -> float:
    """Compute the current learning rate for logging purposes."""
    if step < args.warmup_steps:
        # Linear warmup
        return args.learning_rate * step / max(args.warmup_steps, 1)
    else:
        # Cosine decay
        progress = (step - args.warmup_steps) / max(args.max_steps - args.warmup_steps, 1)
        progress = min(progress, 1.0)
        return args.learning_rate * 0.5 * (1.0 + np.cos(np.pi * progress))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def create_dummy_batch(
    batch_size: int,
    config: DreamZeroConfig,
    dtype: jnp.dtype = jnp.float32,
) -> dict:
    """Create a dummy batch for testing / when data pipeline is not yet implemented.

    Returns a dict with the same keys and shapes expected by ``DreamZero.train_step``.
    """
    num_frames = config.num_frames_per_block * 4  # e.g. 4 blocks
    H, W = 128, 128  # placeholder spatial resolution
    seq_len = 64  # placeholder text sequence length
    num_blocks = num_frames // config.num_frames_per_block
    total_actions = num_blocks * config.num_action_per_block

    return {
        "video": jnp.zeros((batch_size, num_frames, H, W, 3), dtype=dtype),
        "token_ids": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
        "actions": jnp.zeros((batch_size, total_actions, config.action_dim), dtype=dtype),
        "state": jnp.zeros((batch_size, num_blocks, config.state_dim), dtype=dtype),
        "embodiment_id": jnp.zeros((batch_size,), dtype=jnp.int32),
        "attention_mask": jnp.ones((batch_size, seq_len), dtype=dtype),
    }


def load_dataset(args: argparse.Namespace, config: DreamZeroConfig):
    """Load training dataset.

    Attempts to import the project's data pipeline. Falls back to a dummy
    data iterator if the data modules are not yet implemented.

    Returns:
        An iterable that yields dicts with keys matching ``DreamZero.train_step``
        arguments.
    """
    try:
        from dreamzero_jax.data.dataset import LeRobotDataset, create_train_dataloader
        from dreamzero_jax.data.transforms import prepare_batch

        dataset = LeRobotDataset(args.dataset)
        dataloader = create_train_dataloader(
            dataset,
            batch_size=args.batch_size * jax.device_count(),
            seed=args.seed,
        )
        logger.info("Loaded dataset '%s' via LeRobotDataset.", args.dataset)

        def transform_iter():
            for raw_batch in dataloader:
                yield prepare_batch(raw_batch)

        return transform_iter()
    except (ImportError, AttributeError) as exc:
        logger.warning(
            "Data pipeline not available (%s). Using dummy data for development.", exc
        )
        global_batch_size = args.batch_size * jax.device_count()
        compute_dtype = jnp.bfloat16 if args.dtype == "bfloat16" else jnp.float32
        dummy = create_dummy_batch(global_batch_size, config, dtype=compute_dtype)

        def dummy_iter():
            while True:
                yield dummy

        return dummy_iter()


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------


def create_train_step(grad_accum_steps: int = 1):
    """Create the jitted training step function.

    If ``grad_accum_steps > 1``, the function accumulates gradients over
    micro-batches before applying the optimizer update.

    Returns:
        A function ``train_step(model, optimizer, batch, key) -> metrics``.
    """

    @nnx.jit
    def train_step_single(
        model: DreamZero,
        optimizer: nnx.Optimizer,
        batch: dict,
        key: jax.Array,
    ) -> dict:
        """Single-step training (no gradient accumulation)."""

        def loss_fn(model):
            output = model.train_step(
                video=batch["video"],
                token_ids=batch["token_ids"],
                actions=batch["actions"],
                state=batch["state"],
                embodiment_id=batch["embodiment_id"],
                attention_mask=batch.get("attention_mask"),
                action_mask=batch.get("action_mask"),
                key=key,
            )
            return output.loss, output

        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (loss, output), grads = grad_fn(model)

        # Apply optimizer update (includes gradient clipping via the optax chain)
        optimizer.update(grads)

        metrics = {
            "loss": output.loss,
            "dynamics_loss": output.dynamics_loss,
            "action_loss": output.action_loss,
        }
        return metrics

    @nnx.jit
    def train_step_accum(
        model: DreamZero,
        optimizer: nnx.Optimizer,
        batch: dict,
        key: jax.Array,
    ) -> dict:
        """Training step with gradient accumulation over micro-batches.

        Splits the batch along the leading axis into ``grad_accum_steps``
        micro-batches, accumulates gradients, and applies a single optimizer
        update.
        """
        micro_batch_size = batch["video"].shape[0] // grad_accum_steps

        # Split batch into micro-batches
        def split_micro(arr):
            return arr.reshape(grad_accum_steps, micro_batch_size, *arr.shape[1:])

        micro_batches = jax.tree.map(split_micro, batch)
        keys = jax.random.split(key, grad_accum_steps)

        # Accumulate gradients
        def accum_step(carry, xs):
            total_grads, total_metrics = carry
            micro_batch_i, key_i = xs

            def loss_fn(model):
                output = model.train_step(
                    video=micro_batch_i["video"],
                    token_ids=micro_batch_i["token_ids"],
                    actions=micro_batch_i["actions"],
                    state=micro_batch_i["state"],
                    embodiment_id=micro_batch_i["embodiment_id"],
                    attention_mask=micro_batch_i.get("attention_mask"),
                    action_mask=micro_batch_i.get("action_mask"),
                    key=key_i,
                )
                return output.loss, output

            grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
            (loss, output), grads = grad_fn(model)

            # Accumulate
            new_grads = jax.tree.map(jnp.add, total_grads, grads)
            new_metrics = {
                "loss": total_metrics["loss"] + output.loss,
                "dynamics_loss": total_metrics["dynamics_loss"] + output.dynamics_loss,
                "action_loss": total_metrics["action_loss"] + output.action_loss,
            }
            return (new_grads, new_metrics), None

        # Initialize accumulators with zeros matching the graph structure
        graphdef, state = nnx.split(model)
        zero_grads = jax.tree.map(jnp.zeros_like, state)
        zero_metrics = {
            "loss": jnp.float32(0.0),
            "dynamics_loss": jnp.float32(0.0),
            "action_loss": jnp.float32(0.0),
        }

        # Extract per-step micro batches by indexing
        # Use lax.scan for efficient accumulation
        (acc_grads, acc_metrics), _ = jax.lax.scan(
            accum_step,
            (zero_grads, zero_metrics),
            (micro_batches, keys),
        )

        # Average gradients and metrics
        avg_grads = jax.tree.map(lambda g: g / grad_accum_steps, acc_grads)
        avg_metrics = jax.tree.map(lambda m: m / grad_accum_steps, acc_metrics)

        # Apply optimizer update with averaged gradients
        optimizer.update(avg_grads)

        return avg_metrics

    if grad_accum_steps <= 1:
        return train_step_single
    else:
        return train_step_accum


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def save_checkpoint(
    output_dir: Path,
    step: int,
    model: DreamZero,
    optimizer: nnx.Optimizer,
) -> Path:
    """Save model + optimizer state and step count with orbax.

    Saves to ``output_dir / checkpoints / step_NNNNNN``.

    Returns:
        Path to the saved checkpoint directory.
    """
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Extract serializable state from NNX modules
    _, model_state = nnx.split(model)
    _, opt_state = nnx.split(optimizer)

    state_to_save = {
        "model": model_state,
        "optimizer": opt_state,
        "step": np.array(step, dtype=np.int32),
    }

    checkpointer = ocp.StandardCheckpointer()
    save_path = ckpt_dir / f"step_{step:07d}"
    checkpointer.save(save_path, state_to_save)
    logger.info("Saved checkpoint at step %d to %s", step, save_path)
    return save_path


def load_checkpoint(
    ckpt_path: str | Path,
    model: DreamZero,
    optimizer: nnx.Optimizer,
) -> int:
    """Restore model + optimizer state from an orbax checkpoint.

    Args:
        ckpt_path: Path to the checkpoint directory.
        model: The model to restore into (modified in place).
        optimizer: The optimizer to restore into (modified in place).

    Returns:
        The training step to resume from.
    """
    ckpt_path = Path(ckpt_path)

    # Build target structure for restoration
    _, model_state = nnx.split(model)
    _, opt_state = nnx.split(optimizer)

    target = {
        "model": model_state,
        "optimizer": opt_state,
        "step": np.array(0, dtype=np.int32),
    }

    checkpointer = ocp.StandardCheckpointer()
    restored = checkpointer.restore(ckpt_path, target)

    # Update model and optimizer in-place
    nnx.update(model, restored["model"])
    nnx.update(optimizer, restored["optimizer"])

    step = int(restored["step"])
    logger.info("Restored checkpoint from %s at step %d", ckpt_path, step)
    return step


# ---------------------------------------------------------------------------
# Wandb integration (optional)
# ---------------------------------------------------------------------------


def init_wandb(args: argparse.Namespace) -> bool:
    """Attempt to initialize wandb. Returns True if successful."""
    try:
        import wandb

        wandb.init(
            project="dreamzero-jax",
            config=vars(args),
            dir=args.output_dir,
        )
        logger.info("Weights & Biases logging enabled.")
        return True
    except ImportError:
        logger.info("wandb not installed. Skipping W&B logging.")
        return False
    except Exception as exc:
        logger.warning("Failed to initialize wandb: %s. Continuing without it.", exc)
        return False


def log_wandb(metrics: dict, step: int) -> None:
    """Log metrics to wandb if available."""
    try:
        import wandb

        if wandb.run is not None:
            wandb.log(metrics, step=step)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    """Main training function."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Log environment info ----
    logger.info("JAX devices: %s", jax.devices())
    logger.info("Device count: %d", jax.device_count())
    logger.info("Process count: %d", jax.process_count())
    logger.info("Process index: %d", jax.process_index())

    # ---- Load config ----
    file_config = None
    if args.config is not None:
        file_config = load_config_file(args.config)
        logger.info("Loaded config from %s", args.config)

    model_config = build_model_config(file_config)
    logger.info("Model config: %s", model_config)

    # Save effective config for reproducibility
    config_save_path = output_dir / "config.json"
    with open(config_save_path, "w") as f:
        import dataclasses

        json.dump(
            {
                "model": dataclasses.asdict(model_config),
                "training": {k: v for k, v in vars(args).items() if k != "config"},
            },
            f,
            indent=2,
            default=str,
        )
    logger.info("Saved effective config to %s", config_save_path)

    # ---- Create mesh ----
    mesh = create_mesh(args.mesh_shape)
    logger.info("Mesh shape: %s", mesh.shape)

    # ---- Initialize model ----
    logger.info("Initializing model...")
    rngs = nnx.Rngs(args.seed)
    with mesh:
        model = DreamZero(model_config, rngs=rngs)

    num_params = sum(p.size for p in jax.tree.leaves(nnx.state(model)))
    logger.info("Model parameters: %s (%.2fB)", f"{num_params:,}", num_params / 1e9)

    # ---- Create optimizer ----
    tx = create_optimizer(
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
    )
    optimizer = nnx.Optimizer(model, tx)
    logger.info("Created AdamW optimizer (lr=%.2e, wd=%.4f)", args.learning_rate, args.weight_decay)

    # ---- Resume from checkpoint ----
    start_step = 0
    if args.checkpoint is not None:
        start_step = load_checkpoint(args.checkpoint, model, optimizer)
        logger.info("Resuming from step %d", start_step)

    # ---- Data ----
    logger.info("Loading dataset '%s'...", args.dataset)
    train_iter = load_dataset(args, model_config)

    # ---- Wandb ----
    use_wandb = False
    if jax.process_index() == 0:
        use_wandb = init_wandb(args)

    # ---- Compile training step ----
    train_step_fn = create_train_step(grad_accum_steps=args.grad_accum_steps)
    logger.info(
        "Training with grad_accum_steps=%d, effective batch size=%d",
        args.grad_accum_steps,
        args.batch_size * jax.device_count() * args.grad_accum_steps,
    )

    # ---- Training loop ----
    logger.info("Starting training from step %d to %d...", start_step, args.max_steps)
    rng = jax.random.key(args.seed)

    step_times = []
    for step in range(start_step, args.max_steps):
        step_start = time.time()

        # Get batch and shard across devices
        batch = next(train_iter)
        batch = shard_batch(batch, mesh)

        # Advance PRNG
        rng, step_key = jax.random.split(rng)

        # Forward + backward + update
        metrics = train_step_fn(model, optimizer, batch, step_key)

        # Block on computation for timing (only when logging)
        if (step + 1) % args.log_every == 0 or step == start_step:
            # Force synchronization
            jax.block_until_ready(metrics)
            step_time = time.time() - step_start
            step_times.append(step_time)

            # Extract scalar values
            loss_val = float(metrics["loss"])
            dyn_loss_val = float(metrics["dynamics_loss"])
            act_loss_val = float(metrics["action_loss"])
            lr_val = get_current_lr(step, args)
            avg_step_time = np.mean(step_times[-args.log_every :])

            if jax.process_index() == 0:
                logger.info(
                    "step %7d | loss %.4f | dyn_loss %.4f | act_loss %.4f | "
                    "lr %.2e | step_time %.3fs",
                    step,
                    loss_val,
                    dyn_loss_val,
                    act_loss_val,
                    lr_val,
                    avg_step_time,
                )

                if use_wandb:
                    log_wandb(
                        {
                            "train/loss": loss_val,
                            "train/dynamics_loss": dyn_loss_val,
                            "train/action_loss": act_loss_val,
                            "train/learning_rate": lr_val,
                            "train/step_time": step_time,
                        },
                        step=step,
                    )

        # ---- Evaluation ----
        if (step + 1) % args.eval_every == 0:
            # Placeholder for evaluation logic.
            # When the eval data pipeline is implemented, add validation loss
            # computation here.
            if jax.process_index() == 0:
                logger.info("step %7d | eval checkpoint (eval pipeline not yet implemented)", step)

        # ---- Save checkpoint ----
        if (step + 1) % args.save_every == 0:
            if jax.process_index() == 0:
                save_checkpoint(output_dir, step + 1, model, optimizer)

    # ---- Final checkpoint ----
    if jax.process_index() == 0:
        save_checkpoint(output_dir, args.max_steps, model, optimizer)
        logger.info("Training complete. Final checkpoint saved.")

        if use_wandb:
            try:
                import wandb

                wandb.finish()
            except Exception:
                pass

    logger.info("Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
