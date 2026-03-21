#!/usr/bin/env python3
"""Validate DreamZero-JAX with real DROID checkpoint weights."""
import jax
import jax.numpy as jnp
import numpy as np
import time
from flax import nnx

print(f"{len(jax.devices())} TPU chips")

from dreamzero_jax.utils.hf_download import load_checkpoint_auto
print("Loading PyTorch checkpoint...")
t0 = time.time()
pt_state = load_checkpoint_auto("/home/Jiafan/checkpoints/DreamZero-DROID")
print(f"  {len(pt_state)} params ({time.time()-t0:.1f}s)")

from dreamzero_jax.models.dreamzero import DreamZero, DreamZeroConfig
from dreamzero_jax.utils.checkpoint import convert_checkpoint, apply_to_model

# Use 24L for full-pipeline inference (40L OOMs on v5e-8 full pipeline)
cfg = DreamZeroConfig(
    dim=5120, ffn_dim=13824, num_heads=40, num_layers=24,
    freq_dim=256, text_dim=4096, patch_size=(1, 2, 2),
    in_channels=16, out_channels=16, has_image_input=True,
    dtype=jnp.bfloat16, param_dtype=jnp.bfloat16,
)

cpu = jax.devices("cpu")[0]
print("Creating 24L model on CPU...")
with jax.default_device(cpu):
    model = DreamZero(cfg, rngs=nnx.Rngs(0))

    print("Converting weights...")
    t0 = time.time()
    converted = convert_checkpoint(pt_state, cfg)
    import ml_dtypes
    for k in list(converted.keys()):
        if converted[k].dtype in (np.float32, np.float64):
            converted[k] = converted[k].astype(ml_dtypes.bfloat16)
    applied, missing, extra = apply_to_model(model, converted)
    print(f"  {applied} applied, {len(missing)} missing, {len(extra)} extra ({time.time()-t0:.1f}s)")

    if missing:
        for m in sorted(missing)[:5]:
            print(f"    missing: {m}")
    if extra:
        print(f"    extra: {len(extra)} (blocks 24-39 from 40L checkpoint)")

leaves = jax.tree.leaves(nnx.state(model, nnx.Param))
nan_count = sum(1 for l in leaves if np.any(np.isnan(np.asarray(l))))
print(f"  NaN params: {nan_count}/{len(leaves)}")

del pt_state, converted
import gc; gc.collect()

from dreamzero_jax.utils.sharding import shard_params, create_mesh
mesh = create_mesh()
print("Sharding to TPU...")
t0 = time.time()
model = shard_params(model, mesh, param_dtype=jnp.bfloat16)
bu = jax.devices()[0].memory_stats().get("bytes_in_use", 0)
print(f"  HBM: {bu/1e9:.2f} GB ({time.time()-t0:.1f}s)")

from jax.sharding import NamedSharding, PartitionSpec as P
rep = NamedSharding(mesh, P())
key = jax.device_put(jax.random.PRNGKey(42), rep)
v = jax.device_put(jnp.zeros((1, 33, 320, 176, 3), dtype=jnp.bfloat16), rep)
t_ids = jax.device_put(jnp.ones((1, 512), dtype=jnp.int32), rep)
mk = jax.device_put(jnp.ones((1, 512), dtype=jnp.int32), rep)
s = jax.device_put(jnp.zeros((1, 9, 14), dtype=jnp.float32), rep)
e = jax.device_put(jnp.zeros((1,), dtype=jnp.int32), rep)

print("Running generate_scan (24L, real weights)...")
t0 = time.time()
out = model.generate_scan(v, t_ids, s, e, attention_mask=mk, key=key)
jax.block_until_ready(out)
elapsed = time.time() - t0
print(f"  DONE in {elapsed:.1f}s")
print(f"  action={out.action_pred.shape} video={out.video_pred.shape}")
a_mean = float(jnp.mean(out.action_pred))
a_std = float(jnp.std(out.action_pred))
v_mean = float(jnp.mean(out.video_pred))
v_std = float(jnp.std(out.video_pred))
print(f"  action: mean={a_mean:.4f} std={a_std:.4f}")
print(f"  video:  mean={v_mean:.4f} std={v_std:.4f}")

has_nan = bool(jnp.any(jnp.isnan(out.action_pred))) or bool(jnp.any(jnp.isnan(out.video_pred)))
print(f"  NaN in output: {has_nan}")
status = "PASSED" if not has_nan and applied > 0 else "FAILED"
reason = ""
if applied == 0:
    reason = " (0 weights applied — key mapping failed)"
elif has_nan:
    reason = " (NaN in outputs)"
print(f"\n=== REAL WEIGHT INFERENCE {status}{reason} ===")
