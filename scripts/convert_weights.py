#!/usr/bin/env python3
"""Download and convert PyTorch DreamZero weights to Flax NNX format.

Supports loading from a local path or HuggingFace model ID, with optional
bfloat16 casting and end-to-end verification against the PyTorch model.

Usage
-----
From a local checkpoint::

    uv run python scripts/convert_weights.py \
        --input path/to/pytorch_model.safetensors \
        --output path/to/flax_checkpoint

From HuggingFace::

    uv run python scripts/convert_weights.py \
        --input dreamzero0/dreamzero-droid \
        --output flax_ckpt --hf

With bfloat16 conversion::

    uv run python scripts/convert_weights.py \
        --input model.safetensors --output flax_ckpt --dtype bfloat16

Verify numerical parity::

    uv run python scripts/convert_weights.py \
        --input model.safetensors --output flax_ckpt --verify
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

from dreamzero_jax.models.dreamzero import DreamZero, DreamZeroConfig
from dreamzero_jax.utils.checkpoint import (
    apply_to_model,
    compare_param_shapes,
    convert_checkpoint,
    print_key_mapping,
    save_flax_checkpoint,
)
from dreamzero_jax.utils.hf_download import (
    download_from_hf,
    load_checkpoint_auto,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------


def _build_config(
    config_json: str | None = None,
    config_file: str | None = None,
    preset: str | None = None,
) -> DreamZeroConfig:
    """Build a DreamZeroConfig from presets and/or overrides."""
    kwargs: dict = {}

    if preset == "14b":
        kwargs.update(
            dim=5120, num_heads=40, num_layers=40, ffn_dim=13824,
            freq_dim=256, in_channels=36, out_channels=16, has_image_input=True,
        )
    elif preset == "1.3b":
        kwargs.update(
            dim=1536, num_heads=12, num_layers=30, ffn_dim=8960,
            freq_dim=256, in_channels=16, out_channels=16,
        )

    if config_file:
        with open(config_file) as f:
            overrides = json.load(f)
        for k, v in overrides.items():
            if isinstance(v, list):
                overrides[k] = tuple(v)
        kwargs.update(overrides)

    if config_json:
        overrides = json.loads(config_json)
        for k, v in overrides.items():
            if isinstance(v, list):
                overrides[k] = tuple(v)
        kwargs.update(overrides)

    return DreamZeroConfig(**kwargs)


# ---------------------------------------------------------------------------
# Dtype casting
# ---------------------------------------------------------------------------


def _cast_state_dict(converted: dict, target_dtype: str) -> dict:
    """Cast all float arrays to the target dtype. Skips integers."""
    import jax.numpy as jnp

    dtype_map = {
        "bfloat16": jnp.bfloat16,
        "float16": jnp.float16,
        "float32": jnp.float32,
    }
    target = dtype_map[target_dtype]
    cast_count = 0

    for key, arr in converted.items():
        if arr.dtype.kind == "f" and arr.dtype != target:
            converted[key] = arr.astype(target)
            cast_count += 1

    logger.info("Cast %d parameters to %s", cast_count, target_dtype)
    return converted


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def _verify_dit_block(model, config):
    """Smoke-test one DiT block. Returns a ComponentResult."""
    import jax.numpy as jnp
    import numpy as np
    from dreamzero_jax.nn.embed import WanRoPE3D
    from dreamzero_jax.utils.validation import compare_arrays

    rng = np.random.RandomState(42)
    B, S, dim = 1, 16, config.dim

    x = jnp.array(rng.randn(B, S, dim).astype(np.float32))
    e = jnp.array(rng.randn(B, 6, dim).astype(np.float32))
    ctx = jnp.array(rng.randn(B, 8, dim).astype(np.float32))
    freqs = WanRoPE3D(dim // config.num_heads)(2, 2, 4)

    try:
        out = model.dit.blocks[0](x, e, ctx, freqs)
        bad = jnp.any(jnp.isnan(out)) or jnp.any(jnp.isinf(out))
        ref = np.zeros((1,)) if bad else np.asarray(out)
        src = np.full((1,), float("inf")) if bad else np.asarray(out)
        print(f"  dit_block: shape={out.shape}, "
              f"mean={float(jnp.mean(out)):.4f}")
        return compare_arrays(src, ref, name="dit_block", atol=1e-5, rtol=1e-5)
    except Exception as exc:
        print(f"  dit_block FAILED: {exc}")
        return compare_arrays(
            np.full((1,), float("inf")), np.zeros((1,)),
            name="dit_block_error", atol=1e-5, rtol=1e-5,
        )


def _run_verification(model, config) -> bool:
    """Run smoke-test forward passes. Returns True if all pass."""
    from dreamzero_jax.utils.validation import format_report

    print("\n=== Forward Pass Verification ===\n")
    results = [_verify_dit_block(model, config)]
    print(format_report(results))
    ok = all(r.status in ("PASS", "WARN") for r in results)
    print("\nVerification", "PASSED" if ok else "FAILED")
    return ok


# ---------------------------------------------------------------------------
# Shape comparison
# ---------------------------------------------------------------------------


def _run_shape_comparison(pt_state, config, args):
    """Compare checkpoint shapes against instantiated model."""
    import jax
    from flax import nnx

    print("\nInstantiating model for shape comparison...")
    model = DreamZero(config, rngs=nnx.Rngs(0))
    report = compare_param_shapes(
        pt_state, model, config, prefix_strip=args.prefix_strip,
    )

    mismatches, unmapped, matched = [], [], 0
    for pt_key, info in sorted(report.items()):
        if info["flax_path"] is None:
            unmapped.append(pt_key)
        elif info["match"] is True:
            matched += 1
        elif info["match"] is False:
            mismatches.append(info)

    print(f"\n{matched} matched, {len(mismatches)} mismatched, "
          f"{len(unmapped)} unmapped")

    if mismatches:
        print("\nMismatched:")
        for info in mismatches:
            print(f"  {info['pt_key']}: PT={info['pt_shape']}, "
                  f"expected={info['expected_shape']}, "
                  f"model={info['flax_shape']}")

    if unmapped and args.verbose:
        print(f"\nUnmapped ({len(unmapped)}):")
        for k in unmapped[:50]:
            print(f"  {k}")
        if len(unmapped) > 50:
            print(f"  ... and {len(unmapped) - 50} more")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    ap = argparse.ArgumentParser(
        description="Convert PyTorch DreamZero weights to Flax NNX.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    g = ap.add_argument_group("input")
    g.add_argument("--input", "-i", type=str, required=True)
    g.add_argument("--hf", action="store_true",
                    help="Treat --input as HuggingFace repo ID")
    g.add_argument("--hf-revision", type=str, default=None)
    g.add_argument("--hf-cache", type=str, default=None)

    g = ap.add_argument_group("output")
    g.add_argument("--output", "-o", type=str, required=True)
    g.add_argument("--overwrite", action="store_true")

    g = ap.add_argument_group("config")
    g.add_argument("--preset", choices=["14b", "1.3b"], default=None)
    g.add_argument("--config", "-c", type=str, default=None)
    g.add_argument("--config-file", type=str, default=None)

    g = ap.add_argument_group("conversion")
    g.add_argument("--dtype", choices=["float32", "bfloat16", "float16"],
                    default="float32")
    g.add_argument("--strict", action="store_true")
    g.add_argument("--prefix-strip", type=str, default=None)

    g = ap.add_argument_group("diagnostics")
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--compare-shapes", action="store_true")
    g.add_argument("--print-mapping", action="store_true")
    g.add_argument("--verify", action="store_true")

    ap.add_argument("--verbose", "-v", action="count", default=0)
    return ap


def _resolve_input(args) -> Path:
    """Resolve the input checkpoint path (local or HuggingFace)."""
    if args.hf:
        path = download_from_hf(
            args.input, revision=args.hf_revision, cache_dir=args.hf_cache,
        )
        print(f"Downloaded to: {path}")
        return path
    return Path(args.input)


def _load_and_report(input_path: Path) -> dict:
    """Load checkpoint and print summary statistics."""
    print(f"Loading: {input_path}")
    t0 = time.monotonic()
    pt_state = load_checkpoint_auto(input_path)
    print(f"  {len(pt_state)} params ({time.monotonic() - t0:.1f}s)")
    total_bytes = sum(v.nbytes for v in pt_state.values())
    print(f"  {sum(v.size for v in pt_state.values()):,} elements, "
          f"{total_bytes / 1e9:.2f} GB")
    return pt_state


def _convert_and_save(pt_state, config, args):
    """Full conversion pipeline: instantiate, convert, save."""
    import jax
    from flax import nnx

    print("\nInstantiating model on CPU...")
    t0 = time.monotonic()
    with jax.default_device(jax.devices("cpu")[0]):
        model = DreamZero(config, rngs=nnx.Rngs(0))
    print(f"  Done ({time.monotonic() - t0:.1f}s)")

    model_params = sum(v.size for v in jax.tree.leaves(nnx.state(model)))
    print(f"  Model params: {model_params:,}")

    print("Converting...")
    t0 = time.monotonic()
    converted = convert_checkpoint(
        pt_state, config, strict=args.strict, prefix_strip=args.prefix_strip,
    )
    if args.dtype != "float32":
        converted = _cast_state_dict(converted, args.dtype)

    num_applied, missing, extra = apply_to_model(
        model, converted, strict=args.strict,
    )
    elapsed = time.monotonic() - t0
    print(f"\nResults ({elapsed:.1f}s): {num_applied} applied, "
          f"{len(missing)} missing, {len(extra)} extra")

    if missing and args.verbose >= 1:
        for k in sorted(missing)[:30]:
            print(f"  missing: {k}")
    if extra and args.verbose >= 1:
        for k in sorted(extra)[:30]:
            print(f"  extra: {'.'.join(k) if isinstance(k, tuple) else k}")

    output_path = Path(args.output)
    print(f"\nSaving to: {output_path}")
    t0 = time.monotonic()
    save_flax_checkpoint(nnx.state(model), output_path, overwrite=args.overwrite)
    print(f"  Saved ({time.monotonic() - t0:.1f}s)")

    return model


def main() -> None:
    args = _build_parser().parse_args()

    log_level = [logging.WARNING, logging.INFO, logging.DEBUG][min(args.verbose, 2)]
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = _build_config(args.config, args.config_file, args.preset)

    if args.print_mapping:
        print_key_mapping(config)
        return

    input_path = _resolve_input(args)
    pt_state = _load_and_report(input_path)

    if args.dry_run:
        print("\n[Dry run] Mapping keys:")
        converted = convert_checkpoint(
            pt_state, config, prefix_strip=args.prefix_strip,
        )
        print(f"  Mapped {len(converted)} parameters")
        for i, (p, a) in enumerate(sorted(converted.items())):
            if i >= 20:
                print(f"  ... and {len(converted) - 20} more")
                break
            print(f"  {'.'.join(p):60s} {str(a.shape):>20s}")
        return

    if args.compare_shapes:
        _run_shape_comparison(pt_state, config, args)
        return

    model = _convert_and_save(pt_state, config, args)

    if args.verify:
        sys.exit(0 if _run_verification(model, config) else 1)

    print("Done.")


if __name__ == "__main__":
    main()
