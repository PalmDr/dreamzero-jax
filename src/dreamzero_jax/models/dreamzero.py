"""Full DreamZero model assembly.

Wraps the text encoder, image encoder, VAE, causal DiT, and schedulers
into a unified model for training and inference.

This corresponds to ``WANPolicyHead`` in the PyTorch reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx

from dreamzero_jax.models.action_head import CausalWanDiT
from dreamzero_jax.models.image_encoder import WanImageEncoder
from dreamzero_jax.models.text_encoder import WanTextEncoder
from dreamzero_jax.models.vae import WanVideoVAE
from dreamzero_jax.schedulers.flow_matching import FlowMatchScheduler
from dreamzero_jax.schedulers.unipc import FlowUniPCMultistepScheduler


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DreamZeroConfig:
    """Configuration for the DreamZero model.

    Groups all sub-model hyperparameters in one place.
    """

    # DiT
    dim: int = 1536
    in_channels: int = 16
    out_channels: int = 16
    ffn_dim: int = 8960
    freq_dim: int = 256
    num_heads: int = 12
    num_layers: int = 30
    patch_size: tuple[int, int, int] = (1, 2, 2)
    qk_norm: bool = True
    cross_attn_norm: bool = False

    # Text encoder
    text_vocab: int = 256384
    text_dim: int = 4096
    text_attn_dim: int = 4096
    text_ffn_dim: int = 10240
    text_num_heads: int = 64
    text_num_layers: int = 24
    text_num_buckets: int = 32

    # Image encoder
    image_size: int = 224
    image_patch_size: int = 14
    image_dim: int = 1280
    image_mlp_ratio: int = 4
    image_out_dim: int = 512
    image_num_heads: int = 16
    image_num_layers: int = 32

    # VAE
    vae_z_dim: int = 16
    vae_base_dim: int = 96

    # Action
    action_dim: int = 7
    state_dim: int = 14
    action_hidden_size: int = 1024
    num_action_per_block: int = 32
    num_state_per_block: int = 1
    num_frames_per_block: int = 1
    max_num_embodiments: int = 32
    action_horizon: int | None = None  # defaults to num_action_per_block

    # Scheduler
    scheduler_shift: float = 5.0
    num_train_timesteps: int = 1000
    num_inference_steps: int = 16
    cfg_scale: float = 5.0

    # Image-to-video
    has_image_input: bool = True


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


class TrainOutput(NamedTuple):
    """Training forward output."""

    loss: jax.Array
    dynamics_loss: jax.Array
    action_loss: jax.Array


class InferenceOutput(NamedTuple):
    """Inference output."""

    action_pred: jax.Array
    video_pred: jax.Array


# ---------------------------------------------------------------------------
# DreamZero model
# ---------------------------------------------------------------------------


class DreamZero(nnx.Module):
    """Full DreamZero model: joint video + action generation.

    Combines:

    * **Text encoder** (T5-style) for language conditioning
    * **Image encoder** (CLIP ViT) for visual conditioning
    * **Video VAE** for latent space compression
    * **CausalWanDiT** for joint video + action denoising
    * **FlowMatchScheduler** for noise schedules

    Training:
        Call :meth:`train_step` with raw video, text, actions, and state.
        Handles encoding, noise injection, DiT forward, and loss computation.

    Inference:
        Call :meth:`generate` with a conditioning frame, text prompt,
        state, and embodiment ID. Returns predicted actions and video latents.
    """

    def __init__(self, config: DreamZeroConfig, *, rngs: nnx.Rngs):
        self.config = config

        # --- Sub-models ---
        self.text_encoder = WanTextEncoder(
            vocab=config.text_vocab,
            dim=config.text_dim,
            dim_attn=config.text_attn_dim,
            dim_ffn=config.text_ffn_dim,
            num_heads=config.text_num_heads,
            num_layers=config.text_num_layers,
            num_buckets=config.text_num_buckets,
            shared_pos=False,
            rngs=rngs,
        )

        self.image_encoder = WanImageEncoder(
            image_size=config.image_size,
            patch_size=config.image_patch_size,
            dim=config.image_dim,
            mlp_ratio=config.image_mlp_ratio,
            out_dim=config.image_out_dim,
            num_heads=config.image_num_heads,
            num_layers=config.image_num_layers,
            activation="gelu",
            rngs=rngs,
        )

        self.vae = WanVideoVAE(
            z_dim=config.vae_z_dim,
            base_dim=config.vae_base_dim,
            rngs=rngs,
        )

        self.dit = CausalWanDiT(
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
            action_dim=config.action_dim,
            state_dim=config.state_dim,
            action_hidden_size=config.action_hidden_size,
            num_action_per_block=config.num_action_per_block,
            num_state_per_block=config.num_state_per_block,
            num_frames_per_block=config.num_frames_per_block,
            max_num_embodiments=config.max_num_embodiments,
            rngs=rngs,
        )

        # --- Scheduler ---
        self.scheduler = FlowMatchScheduler(
            shift=config.scheduler_shift,
            sigma_min=0.0,
            extra_one_step=True,
            num_train_timesteps=config.num_train_timesteps,
        )
        self.scheduler.set_timesteps(config.num_train_timesteps, training=True)

    # -----------------------------------------------------------------
    # Encoding helpers
    # -----------------------------------------------------------------

    def encode_prompt(
        self,
        token_ids: jax.Array,
        attention_mask: jax.Array | None = None,
    ) -> jax.Array:
        """Encode text tokens to embeddings.

        Args:
            token_ids: ``(B, L)`` int32 token IDs.
            attention_mask: ``(B, L)`` where 1 = attend, 0 = pad.

        Returns:
            ``(B, L, text_dim)`` text embeddings.
        """
        emb = self.text_encoder(token_ids, mask=attention_mask)
        if attention_mask is not None:
            emb = emb * attention_mask[:, :, None]
        return emb

    def encode_video(self, video: jax.Array) -> jax.Array:
        """Encode video to VAE latents.

        Args:
            video: ``(B, T, H, W, C)`` channels-last, values in ``[-1, 1]``.

        Returns:
            Latent ``(B, T', H', W', z_dim)`` channels-last.
        """
        return self.vae.encode(video)

    def encode_image(
        self,
        image: jax.Array,
    ) -> jax.Array:
        """Encode image with CLIP for cross-attention.

        Args:
            image: ``(B, H, W, 3)`` channels-last, values in ``[-1, 1]``.

        Returns:
            CLIP features ``(B, num_tokens, image_dim)``.
        """
        return self.image_encoder.encode_image(image)

    # -----------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------

    def train_step(
        self,
        video: jax.Array,
        token_ids: jax.Array,
        actions: jax.Array,
        state: jax.Array,
        embodiment_id: jax.Array,
        attention_mask: jax.Array | None = None,
        action_mask: jax.Array | None = None,
        *,
        key: jax.Array,
    ) -> TrainOutput:
        """Full training forward pass.

        Encodes inputs, samples noise and timesteps, runs the DiT,
        and computes MSE losses weighted by the scheduler.

        Args:
            video: Raw video ``(B, T, H, W, 3)`` in ``[-1, 1]``.
            token_ids: Text token IDs ``(B, L)`` int32.
            actions: Ground truth actions
                ``(B, num_blocks * num_action_per_block, action_dim)``
                in ``[-1, 1]``.
            state: Robot state ``(B, num_blocks, state_dim)``.
            embodiment_id: ``(B,)`` int embodiment IDs.
            attention_mask: ``(B, L)`` text attention mask (optional).
            action_mask: ``(B, total_actions, action_dim)`` per-element
                action mask (optional, defaults to all ones).
            key: PRNG key for noise sampling.

        Returns:
            :class:`TrainOutput` with ``loss``, ``dynamics_loss``,
            ``action_loss``.
        """
        B = video.shape[0]
        key_noise, key_t = jax.random.split(key)

        # --- Encode ---
        prompt_emb = self.encode_prompt(token_ids, attention_mask)
        latents = self.encode_video(video)
        # First frame for CLIP conditioning
        clip_emb = self.encode_image(video[:, 0])

        # --- Sample timesteps (uniform) ---
        timestep_ids = jax.random.randint(
            key_t, (B,), 0, self.scheduler.num_train_timesteps,
        )
        timesteps = self.scheduler.timesteps[timestep_ids]

        # --- Add noise ---
        key_vid, key_act = jax.random.split(key_noise)
        noise = jax.random.normal(key_vid, latents.shape)
        # Flatten batch*time for add_noise (scheduler expects (N, ...) with per-N timestep)
        B_lat, T_lat = latents.shape[0], latents.shape[1]
        # Broadcast timestep across temporal dimension for per-frame noise
        t_expanded = jnp.broadcast_to(timesteps[:, None], (B, T_lat)).reshape(-1)
        noisy_latents = self.scheduler.add_noise(
            latents.reshape(-1, *latents.shape[2:]),
            noise.reshape(-1, *noise.shape[2:]),
            t_expanded,
        ).reshape(latents.shape)

        noise_action = jax.random.normal(key_act, actions.shape)
        # For coupled noise, use the same timestep for actions
        total_actions = actions.shape[1]
        t_action = jnp.broadcast_to(timesteps[:, None], (B, total_actions)).reshape(-1)
        noisy_actions = self.scheduler.add_noise(
            actions.reshape(-1, actions.shape[-1]),
            noise_action.reshape(-1, noise_action.shape[-1]),
            t_action,
        ).reshape(actions.shape)

        # --- DiT forward ---
        video_pred, action_pred = self.dit(
            noisy_latents,
            timesteps,
            prompt_emb,
            state,
            embodiment_id,
            noisy_actions,
            timestep_action=timesteps,
            clean_x=latents,
            clip_emb=clip_emb if self.config.has_image_input else None,
        )

        # --- Compute loss ---
        # Training target: velocity field (noise - sample)
        video_target = self.scheduler.training_target(latents, noise, timesteps)
        action_target = self.scheduler.training_target(actions, noise_action, timesteps)

        # Per-sample video loss
        dynamics_loss = jnp.mean((video_pred - video_target) ** 2)

        # Per-sample action loss with optional masking
        if action_mask is not None:
            action_loss = jnp.mean((action_pred - action_target) ** 2 * action_mask)
        else:
            action_loss = jnp.mean((action_pred - action_target) ** 2)

        loss = dynamics_loss + action_loss

        return TrainOutput(loss=loss, dynamics_loss=dynamics_loss, action_loss=action_loss)

    # -----------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------

    def generate(
        self,
        video: jax.Array,
        token_ids: jax.Array,
        state: jax.Array,
        embodiment_id: jax.Array,
        attention_mask: jax.Array | None = None,
        num_inference_steps: int | None = None,
        cfg_scale: float | None = None,
        *,
        key: jax.Array,
    ) -> InferenceOutput:
        """Generate actions and video predictions.

        Uses the UniPC multistep scheduler for denoising with
        classifier-free guidance (unconditional = zero text embedding).

        Note: This is a basic implementation without KV caching.
        For production inference, use KV-cached autoregressive generation.

        Args:
            video: Conditioning video ``(B, T, H, W, 3)`` in ``[-1, 1]``.
            token_ids: Text token IDs ``(B, L)`` int32.
            state: Robot state ``(B, num_blocks, state_dim)``.
            embodiment_id: ``(B,)`` int embodiment IDs.
            attention_mask: ``(B, L)`` text attention mask.
            num_inference_steps: Override number of denoising steps.
            cfg_scale: Override classifier-free guidance scale.
            key: PRNG key for noise initialization.

        Returns:
            :class:`InferenceOutput` with ``action_pred`` and ``video_pred``.
        """
        num_steps = num_inference_steps or self.config.num_inference_steps
        cfg = cfg_scale or self.config.cfg_scale
        B = video.shape[0]

        # --- Encode conditioning ---
        prompt_emb = self.encode_prompt(token_ids, attention_mask)
        latents = self.encode_video(video)
        clip_emb = self.encode_image(video[:, 0])

        # Null prompt for unconditional branch (zeros)
        null_prompt = jnp.zeros_like(prompt_emb)

        # --- Initialize noise ---
        # Video noise shape matches latent spatial dims
        key_vid, key_act = jax.random.split(key)
        noisy_video = jax.random.normal(key_vid, latents.shape)

        total_actions = (
            latents.shape[1]
            // self.config.num_frames_per_block
            * (self.config.action_horizon or self.config.num_action_per_block)
        )
        noisy_actions = jax.random.normal(
            key_act, (B, total_actions, self.config.action_dim),
        )

        # --- Scheduler setup ---
        sched = FlowUniPCMultistepScheduler(
            shift=self.config.scheduler_shift,
            num_train_timesteps=self.config.num_train_timesteps,
        )
        sched.set_timesteps(num_steps)

        sched_action = FlowUniPCMultistepScheduler(
            shift=self.config.scheduler_shift,
            num_train_timesteps=self.config.num_train_timesteps,
        )
        sched_action.set_timesteps(num_steps)

        # --- Denoising loop ---
        for i, t in enumerate(sched.timesteps):
            t_action = sched_action.timesteps[i]
            t_video = jnp.broadcast_to(jnp.asarray(t, dtype=jnp.float32), (B,))
            t_act = jnp.broadcast_to(jnp.asarray(t_action, dtype=jnp.float32), (B,))

            # Conditional prediction
            vid_cond, act_cond = self.dit(
                noisy_video, t_video, prompt_emb,
                state, embodiment_id, noisy_actions,
                timestep_action=t_act,
                clip_emb=clip_emb if self.config.has_image_input else None,
            )

            # Unconditional prediction (null text)
            vid_uncond, act_uncond = self.dit(
                noisy_video, t_video, null_prompt,
                state, embodiment_id, noisy_actions,
                timestep_action=t_act,
                clip_emb=clip_emb if self.config.has_image_input else None,
            )

            # Classifier-free guidance
            vid_pred = vid_uncond + cfg * (vid_cond - vid_uncond)

            # Step schedulers
            vid_result = sched.step(vid_pred, t, noisy_video, step_index=i)
            noisy_video = vid_result.prev_sample

            act_result = sched_action.step(
                act_cond, t_action, noisy_actions, step_index=i,
            )
            noisy_actions = act_result.prev_sample

        return InferenceOutput(action_pred=noisy_actions, video_pred=noisy_video)
