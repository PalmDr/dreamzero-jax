#!/usr/bin/env python3
"""Convert a PyTorch DreamZero checkpoint to Flax NNX format.

Usage
-----
Basic conversion::

    uv run python scripts/convert_checkpoint.py \\
        --input path/to/pytorch_model.pt \\
        --output path/to/flax_checkpoint

With custom config overrides::

    uv run python scripts/convert_checkpoint.py \\
        --input model.pt --output flax_ckpt \\
        --config '{"dim": 5120, "num_heads": 40, "num_layers": 32}'

Diagnostics only (no model instantiation)::

    uv run python scripts/convert_checkpoint.py \\
        --input model.pt --output flax_ckpt --dry-run

Compare shapes::

    uv run python scripts/convert_checkpoint.py \\
        --input model.pt --output flax_ckpt --compare-shapes
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure the project source is importable when running from the scripts/ dir
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

from dreamzero_jax.models.dreamzero import DreamZero, DreamZeroConfig
from dreamzero_jax.utils.checkpoint import (
    compare_param_shapes,
    convert_and_apply,
    convert_checkpoint,
    load_pytorch_checkpoint,
    print_key_mapping,
    save_flax_checkpoint,
)

logger = logging.getLogger(__name__)


def _build_config(config_overrides: str | None = None) -> DreamZeroConfig:
    """Build a DreamZeroConfig with optional JSON overrides.

    Args:
        config_overrides: JSON string of field overrides, e.g.
            ``'{"dim": 5120, "num_heads": 40}'``.

    Returns:
        A DreamZeroConfig instance.
    """
    kwargs: dict = {}
    if config_overrides:
        overrides = json.loads(config_overrides)
        # Handle tuple fields that get deserialized as lists
        for k, v in overrides.items():
            if isinstance(v, list):
                overrides[k] = tuple(v)
        kwargs.update(overrides)
    return DreamZeroConfig(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert PyTorch DreamZero checkpoint to Flax NNX format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to PyTorch checkpoint (.pt, .pth, .bin, or .safetensors)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output path for Flax checkpoint directory",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="JSON string with DreamZeroConfig overrides",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=None,
        help="Path to JSON file with DreamZeroConfig overrides",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any PyTorch keys cannot be mapped or model params are missing",
    )
    parser.add_argument(
        "--prefix-strip",
        type=str,
        default=None,
        help="Prefix to strip from PyTorch keys (e.g. 'module.' for DDP checkpoints)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output checkpoint",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only load and map keys; do not create a model or save",
    )
    parser.add_argument(
        "--compare-shapes",
        action="store_true",
        help="Compare checkpoint shapes against model (requires model instantiation)",
    )
    parser.add_argument(
        "--print-mapping",
        action="store_true",
        help="Print the key mapping rules and exit",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="count",
        default=0,
        help="Increase verbosity (-v for INFO, -vv for DEBUG)",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.WARNING
    if args.verbose >= 2:
        log_level = logging.DEBUG
    elif args.verbose >= 1:
        log_level = logging.INFO

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Build config
    config_json = args.config
    if args.config_file:
        config_json = Path(args.config_file).read_text()
    config = _build_config(config_json)

    # Print mapping and exit
    if args.print_mapping:
        print("Key mapping rules for DreamZeroConfig:")
        print_key_mapping(config)
        return

    # Load checkpoint
    input_path = Path(args.input)
    output_path = Path(args.output)

    print(f"Loading PyTorch checkpoint: {input_path}")
    pt_state = load_pytorch_checkpoint(input_path)
    print(f"  Loaded {len(pt_state)} parameters")

    # Dry run: just show mapping results
    if args.dry_run:
        print("\n[Dry run] Mapping PyTorch keys to Flax paths:")
        converted = convert_checkpoint(
            pt_state, config, prefix_strip=args.prefix_strip,
        )
        print(f"  Mapped {len(converted)} parameters")
        print("\nSample mappings (first 20):")
        for i, (flax_path, arr) in enumerate(sorted(converted.items())):
            if i >= 20:
                print(f"  ... and {len(converted) - 20} more")
                break
            print(f"  {'.'.join(flax_path):60s} {str(arr.shape):>20s}")
        return

    # Compare shapes
    if args.compare_shapes:
        print("\nInstantiating model for shape comparison...")
        import jax
        from flax import nnx
        model = DreamZero(config, rngs=nnx.Rngs(0))
        report = compare_param_shapes(
            pt_state, model, config, prefix_strip=args.prefix_strip,
        )

        mismatches = []
        unmapped = []
        matched = 0
        for pt_key, info in sorted(report.items()):
            if info["flax_path"] is None:
                unmapped.append(pt_key)
            elif info["match"] is True:
                matched += 1
            elif info["match"] is False:
                mismatches.append(info)

        print(f"\nShape comparison: {matched} matched, "
              f"{len(mismatches)} mismatched, {len(unmapped)} unmapped")

        if mismatches:
            print("\nMismatched parameters:")
            for info in mismatches:
                print(f"  {info['pt_key']}")
                print(f"    PT shape:     {info['pt_shape']}")
                print(f"    Expected:     {info['expected_shape']}")
                print(f"    Flax model:   {info['flax_shape']}")

        if unmapped and args.verbose:
            print(f"\nUnmapped PyTorch keys ({len(unmapped)}):")
            for k in unmapped[:50]:
                print(f"  {k}")
            if len(unmapped) > 50:
                print(f"  ... and {len(unmapped) - 50} more")
        return

    # Full conversion
    print("\nInstantiating Flax model...")
    import jax
    from flax import nnx

    model = DreamZero(config, rngs=nnx.Rngs(0))

    print("Converting and applying checkpoint...")
    num_applied, missing, extra = convert_and_apply(
        input_path,
        model,
        config,
        strict=args.strict,
        prefix_strip=args.prefix_strip,
    )

    print(f"\nConversion results:")
    print(f"  Applied:  {num_applied}")
    print(f"  Missing:  {len(missing)} (model params not in checkpoint)")
    print(f"  Extra:    {len(extra)} (checkpoint params not in model)")

    if missing and args.verbose:
        print("\n  Missing model parameters:")
        for k in sorted(missing)[:30]:
            print(f"    {k}")
        if len(missing) > 30:
            print(f"    ... and {len(missing) - 30} more")

    if extra and args.verbose:
        print("\n  Extra checkpoint parameters:")
        for k in sorted(extra)[:30]:
            print(f"    {'.'.join(k)}")
        if len(extra) > 30:
            print(f"    ... and {len(extra) - 30} more")

    # Save
    print(f"\nSaving Flax checkpoint to: {output_path}")
    state = nnx.state(model)
    save_flax_checkpoint(state, output_path, overwrite=args.overwrite)
    print("Done.")


if __name__ == "__main__":
    main()
