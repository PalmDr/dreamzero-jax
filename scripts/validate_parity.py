#!/usr/bin/env python3
"""Validate converted JAX weights against PyTorch reference outputs.

Loads a converted Flax checkpoint, runs inference on each model component,
and reports per-component numerical differences. Optionally compares against
a PyTorch model if a PyTorch checkpoint is provided.

Usage
-----
Smoke test (random weights, checks shapes and NaN)::

    uv run python scripts/validate_parity.py

Compare against PyTorch checkpoint::

    uv run python scripts/validate_parity.py \
        --checkpoint path/to/flax_ckpt \
        --pytorch-checkpoint path/to/pytorch_model.pt

Compare against pre-generated fixtures::

    uv run python scripts/validate_parity.py \
        --checkpoint path/to/flax_ckpt \
        --fixtures-dir fixtures/pt_reference/

Per-component validation only::

    uv run python scripts/validate_parity.py \
        --checkpoint path/to/flax_ckpt \
        --components dit_block,text_encoder
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

from dreamzero_jax.models.dreamzero import DreamZero, DreamZeroConfig
from dreamzero_jax.utils.parity_runners import (
    COMPONENT_ORDER,
    COMPONENT_RUNNERS,
    run_pytorch_component,
)

logger = logging.getLogger(__name__)

TOLERANCES = {
    "text_encoder": (1e-4, 1e-3),
    "image_encoder": (1e-4, 1e-3),
    "vae_encoder": (5e-5, 1e-4),
    "vae_decoder": (5e-5, 1e-4),
    "dit_block": (1e-5, 1e-5),
    "dit_full": (5e-4, 1e-3),
    "action_encoder": (1e-5, 1e-5),
    "state_encoder": (1e-5, 1e-5),
}


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

    if config_file and config_file != "None":
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
# Result type and analysis
# ---------------------------------------------------------------------------


@dataclass
class ParityResult:
    """Numerical comparison result for one component."""

    name: str
    status: str
    max_abs_diff: float
    mean_abs_diff: float
    jax_mean: float
    jax_std: float
    pt_mean: float
    pt_std: float
    shape: tuple[int, ...]
    elapsed_ms: float


def _analyze_result(
    name: str,
    jax_out: np.ndarray,
    pt_out: np.ndarray | None,
    elapsed_ms: float,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> ParityResult:
    """Analyze numerical differences between JAX and reference outputs."""
    jax_f64 = jax_out.astype(np.float64)
    jax_mean = float(np.mean(jax_f64))
    jax_std = float(np.std(jax_f64))

    if pt_out is not None:
        pt_f64 = pt_out.astype(np.float64)
        abs_diff = np.abs(jax_f64 - pt_f64)
        max_abs = float(np.max(abs_diff))
        mean_abs = float(np.mean(abs_diff))
        pt_mean = float(np.mean(pt_f64))
        pt_std = float(np.std(pt_f64))

        passes = np.allclose(jax_f64, pt_f64, atol=atol, rtol=rtol)
        warn = np.allclose(jax_f64, pt_f64, atol=atol * 10, rtol=rtol * 10)
        status = "PASS" if passes else ("WARN" if warn else "FAIL")
    else:
        pt_mean, pt_std, max_abs, mean_abs = 0.0, 0.0, 0.0, 0.0
        has_bad = np.any(np.isnan(jax_f64)) or np.any(np.isinf(jax_f64))
        status = "FAIL" if has_bad else "OK"

    return ParityResult(
        name=name, status=status,
        max_abs_diff=max_abs, mean_abs_diff=mean_abs,
        jax_mean=jax_mean, jax_std=jax_std,
        pt_mean=pt_mean, pt_std=pt_std,
        shape=tuple(jax_out.shape), elapsed_ms=elapsed_ms,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _format_report(results: list[ParityResult], has_pt: bool) -> str:
    """Format results as an aligned text table."""
    if has_pt:
        header = (
            f"{'Component':<20} {'Status':<7} {'MaxAbsDiff':>12} "
            f"{'MeanAbsDiff':>12} {'JAX mean':>10} {'PT mean':>10} {'ms':>7}"
        )
    else:
        header = (
            f"{'Component':<20} {'Status':<7} {'Shape':>20} "
            f"{'JAX mean':>10} {'JAX std':>10} {'ms':>7}"
        )

    sep = "-" * len(header)
    lines = [header, sep]

    for r in results:
        if has_pt:
            lines.append(
                f"{r.name:<20} {r.status:<7} {r.max_abs_diff:>12.2e} "
                f"{r.mean_abs_diff:>12.2e} {r.jax_mean:>10.4f} "
                f"{r.pt_mean:>10.4f} {r.elapsed_ms:>7.1f}"
            )
        else:
            lines.append(
                f"{r.name:<20} {r.status:<7} {str(r.shape):>20} "
                f"{r.jax_mean:>10.4f} {r.jax_std:>10.4f} {r.elapsed_ms:>7.1f}"
            )

    lines.append(sep)
    p = sum(1 for r in results if r.status in ("PASS", "OK"))
    f = sum(1 for r in results if r.status == "FAIL")
    w = sum(1 for r in results if r.status == "WARN")
    lines.append(f"SUMMARY: {p} passed, {w} warnings, {f} failures / {len(results)}")
    return "\n".join(lines)


def _format_json(results: list[ParityResult]) -> dict:
    """Format results as a JSON-serializable dict."""
    comps = []
    for r in results:
        comps.append({
            "name": r.name, "status": r.status,
            "max_abs_diff": r.max_abs_diff, "mean_abs_diff": r.mean_abs_diff,
            "jax_mean": r.jax_mean, "jax_std": r.jax_std,
            "pt_mean": r.pt_mean, "pt_std": r.pt_std,
            "shape": list(r.shape), "elapsed_ms": r.elapsed_ms,
        })
    p = sum(1 for r in results if r.status in ("PASS", "OK"))
    return {"components": comps, "summary": {"total": len(results), "passed": p}}


# ---------------------------------------------------------------------------
# PyTorch comparison
# ---------------------------------------------------------------------------


def _run_pytorch_comparison(
    pt_checkpoint: Path,
    jax_results: dict[str, dict],
    config,
    pytorch_source: str | None = None,
) -> dict[str, np.ndarray]:
    """Run PyTorch model on the same inputs and return outputs."""
    try:
        import torch
    except ImportError:
        logger.error("PyTorch required for --pytorch-checkpoint")
        return {}

    if pytorch_source:
        sys.path.insert(0, pytorch_source)

    logger.info("Loading PyTorch model: %s", pt_checkpoint)
    raw = torch.load(str(pt_checkpoint), map_location="cpu", weights_only=False)
    if isinstance(raw, dict):
        pt_model = raw.get("model", raw.get("state_dict", raw))
        if isinstance(pt_model, dict):
            logger.error("state_dict-only checkpoint needs --pytorch-source")
            return {}
    else:
        pt_model = raw

    pt_model.eval()
    pt_outputs = {}

    for comp_name, jax_result in jax_results.items():
        try:
            pt_out = run_pytorch_component(
                pt_model, comp_name, jax_result["inputs"], config,
            )
            if pt_out is not None:
                pt_outputs[comp_name] = pt_out
        except Exception as exc:
            logger.warning("PyTorch %s failed: %s", comp_name, exc)

    return pt_outputs


def _load_fixture_references(
    fixtures_dir: Path,
    components: list[str],
) -> dict[str, np.ndarray]:
    """Load pre-generated fixture outputs as reference arrays."""
    from dreamzero_jax.utils.validation import load_fixture

    fixture_map = {
        "text_encoder": ("text_encoder.npz", "embeddings"),
        "image_encoder": ("image_encoder.npz", "features"),
        "vae_encoder": ("vae_encoder.npz", "latents"),
        "vae_decoder": ("vae_decoder.npz", "video"),
        "dit_block": ("dit_block.npz", "output"),
        "action_encoder": ("action_encoder.npz", "encoded"),
        "state_encoder": ("category_specific.npz", "output"),
    }

    refs = {}
    for comp in components:
        if comp not in fixture_map:
            continue
        fixture_file, output_key = fixture_map[comp]
        path = fixtures_dir / fixture_file
        if not path.exists():
            logger.warning("Fixture not found: %s", path)
            continue
        data = load_fixture(path)
        if output_key in data:
            refs[comp] = data[output_key]
    return refs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _load_model_with_checkpoint(config, checkpoint_path):
    """Instantiate model and optionally load a checkpoint."""
    from flax import nnx

    logger.info("Instantiating DreamZero model...")
    model = DreamZero(config, rngs=nnx.Rngs(0))

    if checkpoint_path is None:
        logger.warning("No checkpoint -- random weights (shape/NaN checks only)")
        return model

    logger.info("Loading checkpoint: %s", checkpoint_path)
    if checkpoint_path.is_dir():
        from dreamzero_jax.utils.checkpoint import (
            apply_to_model, load_flax_checkpoint,
        )
        state = load_flax_checkpoint(str(checkpoint_path))
        apply_to_model(model, state)
    else:
        from dreamzero_jax.utils.checkpoint import convert_and_apply
        convert_and_apply(str(checkpoint_path), model, config)
    return model


def _run_all_components(model, config, components, rng, verbose):
    """Run all JAX component runners and collect results."""
    jax_results: dict[str, dict] = {}
    for comp in components:
        runner = COMPONENT_RUNNERS[comp]
        logger.info("Running JAX %s...", comp)
        try:
            result = runner(model, config, rng)
            jax_results[comp] = result
            if verbose:
                out = result["output"]
                logger.info("  shape=%s, mean=%.4f, std=%.4f",
                            out.shape, np.mean(out), np.std(out))
        except Exception as exc:
            logger.error("  %s FAILED: %s", comp, exc)
            jax_results[comp] = {
                "inputs": {}, "output": np.array([float("nan")]),
                "elapsed_ms": 0.0,
            }
    return jax_results


def _collect_analysis(jax_results, pt_outputs, components):
    """Build ParityResult list from JAX and optional PT outputs."""
    analysis: list[ParityResult] = []
    for comp in components:
        if comp not in jax_results:
            continue
        jax_out = jax_results[comp]["output"]
        pt_out = pt_outputs.get(comp)
        elapsed = jax_results[comp].get("elapsed_ms", 0.0)
        atol, rtol = TOLERANCES.get(comp, (1e-5, 1e-5))
        analysis.append(_analyze_result(comp, jax_out, pt_out, elapsed, atol, rtol))

        if "action_output" in jax_results[comp]:
            analysis.append(_analyze_result(
                f"{comp}_action", jax_results[comp]["action_output"],
                None, 0.0, atol, rtol,
            ))
    return analysis


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    p = argparse.ArgumentParser(
        description="Validate converted JAX weights for numerical parity.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--pytorch-checkpoint", type=Path, default=None)
    p.add_argument("--pytorch-source", type=str, default=None)
    p.add_argument("--fixtures-dir", type=Path, default=None)
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--config-file", type=Path, default=None)
    p.add_argument("--preset", choices=["14b", "1.3b"], default=None)
    p.add_argument("--components", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    p.add_argument("--json", action="store_true")
    p.add_argument("--strict", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    config = _build_config(
        args.config,
        str(args.config_file) if args.config_file else None,
        args.preset,
    )
    if args.dtype == "bfloat16":
        import jax.numpy as jnp
        config.dtype = jnp.bfloat16
        config.param_dtype = jnp.bfloat16

    components = (
        [c.strip() for c in args.components.split(",")]
        if args.components
        else [c for c in COMPONENT_ORDER if c in COMPONENT_RUNNERS]
    )

    model = _load_model_with_checkpoint(config, args.checkpoint)
    jax_results = _run_all_components(
        model, config, components,
        np.random.RandomState(args.seed), args.verbose,
    )

    pt_outputs: dict[str, np.ndarray] = {}
    if args.pytorch_checkpoint:
        pt_outputs = _run_pytorch_comparison(
            args.pytorch_checkpoint, jax_results, config, args.pytorch_source,
        )
    if args.fixtures_dir:
        pt_outputs.update(_load_fixture_references(args.fixtures_dir, components))

    analysis = _collect_analysis(jax_results, pt_outputs, components)

    if args.json:
        print(json.dumps(_format_json(analysis), indent=2))
    else:
        print("\n" + _format_report(analysis, bool(pt_outputs)) + "\n")

    if args.strict:
        sys.exit(1 if any(r.status == "FAIL" for r in analysis) else 0)


if __name__ == "__main__":
    main()
