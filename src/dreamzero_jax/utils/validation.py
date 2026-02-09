"""Shared infrastructure for PyTorch-to-JAX numerical validation.

Provides comparison utilities, fixture I/O, and report formatting for
validating that JAX model outputs match PyTorch reference outputs.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Default per-component tolerances
# ---------------------------------------------------------------------------

DEFAULT_TOLERANCES: dict[str, tuple[float, float]] = {
    # (atol, rtol)
    "text_encoder": (1e-4, 1e-3),
    "image_encoder": (1e-4, 1e-3),
    "vae_encoder": (5e-5, 1e-4),
    "vae_decoder": (5e-5, 1e-4),
    "dit_block": (1e-5, 1e-5),
    "dit_backbone": (5e-4, 1e-3),
    "category_specific": (1e-5, 1e-5),
    "action_encoder": (1e-5, 1e-5),
    "causal_dit": (5e-4, 1e-3),
    "flow_matching": (1e-6, 1e-6),
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ComponentResult:
    """Result of comparing one component's outputs."""

    name: str
    status: str  # "PASS", "WARN", or "FAIL"
    max_abs_diff: float
    mean_abs_diff: float
    max_rel_diff: float
    shape: tuple[int, ...]
    atol: float
    rtol: float


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def compare_arrays(
    jax_out: np.ndarray,
    ref_out: np.ndarray,
    *,
    name: str = "unnamed",
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> ComponentResult:
    """Compare JAX output against a reference array.

    Args:
        jax_out: Output from the JAX model (numpy array).
        ref_out: Reference output from the fixture (numpy array).
        name: Component name for the result.
        atol: Absolute tolerance. Values within atol are considered matching.
        rtol: Relative tolerance. Values within rtol * |ref| are considered matching.

    Returns:
        A :class:`ComponentResult` summarising the comparison.
    """
    jax_out = np.asarray(jax_out, dtype=np.float64)
    ref_out = np.asarray(ref_out, dtype=np.float64)

    abs_diff = np.abs(jax_out - ref_out)
    max_abs = float(np.max(abs_diff)) if abs_diff.size > 0 else 0.0
    mean_abs = float(np.mean(abs_diff)) if abs_diff.size > 0 else 0.0

    # Relative diff with epsilon to avoid division by zero
    denom = np.maximum(np.abs(ref_out), 1e-12)
    rel_diff = abs_diff / denom
    max_rel = float(np.max(rel_diff)) if rel_diff.size > 0 else 0.0

    # Determine status using numpy's allclose semantics
    passes = np.allclose(jax_out, ref_out, atol=atol, rtol=rtol)
    # Warn if within 10x tolerance but not passing
    warn_threshold = np.allclose(jax_out, ref_out, atol=atol * 10, rtol=rtol * 10)

    if passes:
        status = "PASS"
    elif warn_threshold:
        status = "WARN"
    else:
        status = "FAIL"

    return ComponentResult(
        name=name,
        status=status,
        max_abs_diff=max_abs,
        mean_abs_diff=mean_abs,
        max_rel_diff=max_rel,
        shape=tuple(ref_out.shape),
        atol=atol,
        rtol=rtol,
    )


# ---------------------------------------------------------------------------
# Fixture I/O
# ---------------------------------------------------------------------------


def save_fixture(path: str | Path, **arrays: np.ndarray) -> None:
    """Save arrays to an ``.npz`` file, splitting complex arrays.

    Complex arrays are stored as ``{key}_real`` and ``{key}_imag`` pairs.

    Args:
        path: Output file path.
        **arrays: Named arrays to save.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    save_dict: dict[str, np.ndarray] = {}
    for key, arr in arrays.items():
        arr = np.asarray(arr)
        if np.iscomplexobj(arr):
            save_dict[f"{key}_real"] = arr.real
            save_dict[f"{key}_imag"] = arr.imag
        else:
            save_dict[key] = arr

    np.savez(str(path), **save_dict)


def load_fixture(path: str | Path) -> dict[str, np.ndarray]:
    """Load arrays from an ``.npz`` file, reconstructing complex arrays.

    Args:
        path: Input file path.

    Returns:
        Dictionary mapping names to numpy arrays.
    """
    data = dict(np.load(str(path)))

    # Reconstruct complex arrays from real/imag pairs
    real_keys = [k for k in data if k.endswith("_real")]
    for rk in real_keys:
        base = rk[: -len("_real")]
        ik = f"{base}_imag"
        if ik in data:
            data[base] = data.pop(rk) + 1j * data.pop(ik)
        # else keep as-is (just happens to end with _real)

    return data


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_report(results: list[ComponentResult]) -> str:
    """Format comparison results as an aligned text table.

    Args:
        results: List of component results.

    Returns:
        Multi-line string with header and rows.
    """
    header = (
        f"{'Component':<28} {'Status':<8} {'Max Abs Diff':>14} "
        f"{'Mean Abs Diff':>14} {'Max Rel Diff':>14}"
    )
    sep = "-" * len(header)
    lines = [header, sep]

    pass_count = 0
    for r in results:
        lines.append(
            f"{r.name:<28} {r.status:<8} {r.max_abs_diff:>14.2e} "
            f"{r.mean_abs_diff:>14.2e} {r.max_rel_diff:>14.2e}"
        )
        if r.status == "PASS":
            pass_count += 1

    lines.append(sep)
    lines.append(f"SUMMARY: {pass_count}/{len(results)} passed")
    return "\n".join(lines)


def format_report_json(results: list[ComponentResult]) -> dict[str, Any]:
    """Format comparison results as a JSON-serializable dict.

    Args:
        results: List of component results.

    Returns:
        Dict with ``components`` list and ``summary``.
    """
    components = []
    pass_count = 0
    for r in results:
        d = asdict(r)
        # Convert shape tuple to list for JSON serialization
        d["shape"] = list(r.shape)
        components.append(d)
        if r.status == "PASS":
            pass_count += 1

    return {
        "components": components,
        "summary": {
            "total": len(results),
            "passed": pass_count,
            "failed": len(results) - pass_count,
        },
    }


def save_manifest(
    output_dir: str | Path,
    *,
    config: dict[str, Any],
    fixtures: list[str],
    seed: int,
    versions: dict[str, str] | None = None,
) -> None:
    """Write a manifest.json describing the generated fixtures.

    Args:
        output_dir: Directory to write manifest into.
        config: Model config as a dict.
        fixtures: List of fixture component names.
        seed: Random seed used for generation.
        versions: Optional dict of library versions.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "config": config,
        "fixtures": fixtures,
        "seed": seed,
        "versions": versions or {},
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load a manifest.json file.

    Args:
        path: Path to manifest.json.

    Returns:
        Parsed manifest dict.
    """
    with open(path) as f:
        return json.load(f)
