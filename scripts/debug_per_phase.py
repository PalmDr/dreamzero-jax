#!/usr/bin/env python3
"""Debug: check NaN after each phase and each encoder output."""
import jax, jax.numpy as jnp, numpy as np, time, os
from flax import nnx

print(len(jax.devices()), "chips")

from dreamzero_jax.models.dreamzero import DreamZeroConfig
from dreamzero_jax.utils.checkpoint import convert_checkpoint, apply_to_model
from dreamzero_jax.utils.hf_download import load_checkpoint_auto
from dreamzero_jax.utils.sharding import shard_params, create_mesh
from jax.sharding import NamedSharding, PartitionSpec as P
import ml_dtypes

ckpt = "/home/Jiafan/checkpoints/DreamZero-DROID"
cfg = DreamZeroConfig(
    dim=5120, ffn_dim=13824, num_heads=40, num_layers=8,
    freq_dim=256, text_dim=4096, patch_size=(1, 2, 2),
    in_channels=36, out_channels=16, has_image_input=True,
    image_dim=1280, image_out_dim=1024, cross_attn_norm=True,
    action_dim=32, state_dim=64, max_num_embodiments=1,
    dtype=jnp.bfloat16, param_dtype=jnp.bfloat16,
)

print("Loading PT checkpoint...")
pt = load_checkpoint_auto(ckpt)
with jax.default_device(cpu):
    converted = convert_checkpoint(pt, cfg)
    for k in list(converted.keys()):
        arr = np.asarray(converted[k])
        if arr.dtype in (np.float32, np.float64):
            converted[k] = jnp.array(arr.astype(ml_dtypes.bfloat16))
        else:
            converted[k] = jnp.array(arr)
print(f"  {len(converted)} converted params")

cpu = jax.devices("cpu")[0]
mesh = create_mesh()
rep = NamedSharding(mesh, P())
key = jax.random.PRNGKey(42)
k1, k2 = jax.random.split(key)
v = jax.device_put(jax.random.normal(k1, (1, 33, 320, 176, 3), dtype=jnp.bfloat16) * 0.1, rep)
tokens = jax.device_put(jnp.ones((1, 512), dtype=jnp.int32), rep)
mask = jax.device_put(jnp.ones((1, 512), dtype=jnp.int32), rep)

# === Test each encoder separately ===

# Text encoder
print("\n1. TEXT ENCODER")
from dreamzero_jax.models.text_encoder import WanTextEncoder
with jax.default_device(cpu):
    te = WanTextEncoder(
        vocab=cfg.text_vocab, dim=cfg.text_dim, dim_attn=cfg.text_attn_dim,
        dim_ffn=cfg.text_ffn_dim, num_heads=cfg.text_num_heads,
        num_layers=cfg.text_num_layers, num_buckets=cfg.text_num_buckets,
        dtype=cfg.dtype, param_dtype=cfg.param_dtype, rngs=nnx.Rngs(0),
    )
    te_keys = {k[1:]: v for k, v in converted.items() if k[0] == "text_encoder"}
    applied, missing, _ = apply_to_model(te, te_keys)
    print(f"  {applied} applied, {len(missing)} missing")
te = shard_params(te, mesh, param_dtype=cfg.param_dtype)
text_emb = te(tokens, mask=mask)
jax.block_until_ready(text_emb)
te_nan = bool(jnp.any(jnp.isnan(text_emb)))
te_max = float(jnp.max(jnp.abs(text_emb)))
print(f"  output: shape={text_emb.shape} NaN={te_nan} max={te_max:.4f}")
del te; jax.clear_caches()

# Image encoder
print("\n2. IMAGE ENCODER")
from dreamzero_jax.models.image_encoder import WanImageEncoder
with jax.default_device(cpu):
    ie = WanImageEncoder(
        image_size=cfg.image_size, patch_size=cfg.image_patch_size,
        dim=cfg.image_dim, mlp_ratio=cfg.image_mlp_ratio,
        out_dim=cfg.image_out_dim, num_heads=cfg.image_num_heads,
        num_layers=cfg.image_num_layers,
        dtype=cfg.dtype, param_dtype=cfg.param_dtype, rngs=nnx.Rngs(0),
    )
    ie_keys = {k[1:]: v for k, v in converted.items() if k[0] == "image_encoder"}
    applied, missing, _ = apply_to_model(ie, ie_keys)
    print(f"  {applied} applied, {len(missing)} missing")
ie = shard_params(ie, mesh, param_dtype=cfg.param_dtype)
clip_emb = ie.encode_image(v[:, 0])
jax.block_until_ready(clip_emb)
ie_nan = bool(jnp.any(jnp.isnan(clip_emb)))
ie_max = float(jnp.max(jnp.abs(clip_emb)))
print(f"  output: shape={clip_emb.shape} NaN={ie_nan} max={ie_max:.4f}")
del ie; jax.clear_caches()

# VAE
print("\n3. VAE ENCODER")
from dreamzero_jax.models.vae import WanVideoVAE
with jax.default_device(cpu):
    vae = WanVideoVAE(
        z_dim=cfg.vae_z_dim, base_dim=cfg.vae_base_dim,
        dtype=cfg.dtype, param_dtype=cfg.param_dtype, rngs=nnx.Rngs(0),
    )
    vae_keys = {k[1:]: v for k, v in converted.items() if k[0] == "vae"}
    applied, missing, _ = apply_to_model(vae, vae_keys)
    print(f"  {applied} applied, {len(missing)} missing")
vae = shard_params(vae, mesh, param_dtype=cfg.param_dtype)
latents = vae.encode(v)
jax.block_until_ready(latents)
vae_nan = bool(jnp.any(jnp.isnan(latents)))
vae_max = float(jnp.max(jnp.abs(latents)))
print(f"  output: shape={latents.shape} NaN={vae_nan} max={vae_max:.4f}")

print(f"\n=== SUMMARY ===")
print(f"text_encoder: NaN={te_nan}")
print(f"image_encoder: NaN={ie_nan}")
print(f"vae: NaN={vae_nan}")
if te_nan or ie_nan or vae_nan:
    print("ENCODER NaN FOUND — DiT input is already corrupted")
else:
    print("ALL ENCODERS CLEAN — NaN must originate in DiT forward pass")
