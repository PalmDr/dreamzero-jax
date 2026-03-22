#!/usr/bin/env python3
"""Compare GPU reference outputs against JAX/TPU outputs.

Loads two .npz files and computes per-tensor metrics: max absolute diff,
mean absolute diff, cosine similarity, and relative error. Reports
PASS/FAIL per component with configurable tolerance.

Usage
-----
Basic comparison::

    python scripts/compare_outputs.py \
        gpu_reference.npz tpu_output.npz

With custom tolerance (e.g. for bf16)::

    python scripts/compare_outputs.py \
        gpu_reference.npz tpu_output.npz \
        --atol 1e-2 --rtol 1e-2

JSON output::

    python scripts/compare_outputs.py \
        gpu_reference.npz tpu_output.npz --json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

BF16_ATOL = 1e-2
BF16_RTOL = 1e-2
FP32_ATOL = 1e-5
FP32_RTOL = 1e-5


@dataclass
class TensorComparison:
    """Comparison metrics for a single tensor."""

    name: str
    shape_a: tuple[int, ...]
    shape_b: tuple[int, ...]
    max_abs_diff: float
    mean_abs_diff: float
    median_abs_diff: float
    cosine_sim: float
    relative_error: float
    a_mean: float
    b_mean: float
    a_std: float
    b_std: float
    has_nan_a: bool
    has_nan_b: bool
    status: str


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two flattened arrays."""
    a_flat = a.ravel().astype(np.float64)
    b_flat = b.ravel().astype(np.float64)

    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)

    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0 if (norm_a < 1e-12) != (norm_b < 1e-12) else 1.0

    return float(np.dot(a_flat, b_flat) / (norm_a * norm_b))


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    """Compute relative L2 error: ||a - b|| / max(||a||, ||b||, eps)."""
    a64 = a.ravel().astype(np.float64)
    b64 = b.ravel().astype(np.float64)
    diff_norm = np.linalg.norm(a64 - b64)
    scale = max(np.linalg.norm(a64), np.linalg.norm(b64), 1e-12)
    return float(diff_norm / scale)


def compare_tensor(
    name: str,
    a: np.ndarray,
    b: np.ndarray,
    atol: float,
    rtol: float,
) -> TensorComparison:
    """Compare two arrays and return detailed metrics."""
    a64 = a.astype(np.float64)
    b64 = b.astype(np.float64)

    has_nan_a = bool(np.any(np.isnan(a64)))
    has_nan_b = bool(np.any(np.isnan(b64)))

    if has_nan_a or has_nan_b:
        return TensorComparison(
            name=name,
            shape_a=tuple(a.shape),
            shape_b=tuple(b.shape),
            max_abs_diff=float("inf"),
            mean_abs_diff=float("inf"),
            median_abs_diff=float("inf"),
            cosine_sim=0.0,
            relative_error=float("inf"),
            a_mean=float(np.nanmean(a64)),
            b_mean=float(np.nanmean(b64)),
            a_std=float(np.nanstd(a64)),
            b_std=float(np.nanstd(b64)),
            has_nan_a=has_nan_a,
            has_nan_b=has_nan_b,
            status="FAIL (NaN)",
        )

    if a.shape != b.shape:
        return TensorComparison(
            name=name,
            shape_a=tuple(a.shape),
            shape_b=tuple(b.shape),
            max_abs_diff=float("inf"),
            mean_abs_diff=float("inf"),
            median_abs_diff=float("inf"),
            cosine_sim=0.0,
            relative_error=float("inf"),
            a_mean=float(np.mean(a64)),
            b_mean=float(np.mean(b64)),
            a_std=float(np.std(a64)),
            b_std=float(np.std(b64)),
            has_nan_a=has_nan_a,
            has_nan_b=has_nan_b,
            status=f"FAIL (shape mismatch)",
        )

    abs_diff = np.abs(a64 - b64)
    max_abs = float(np.max(abs_diff))
    mean_abs = float(np.mean(abs_diff))
    median_abs = float(np.median(abs_diff))
    cos_sim = cosine_similarity(a, b)
    rel_err = relative_error(a, b)

    passes = np.allclose(a64, b64, atol=atol, rtol=rtol)
    loose_pass = np.allclose(a64, b64, atol=atol * 10, rtol=rtol * 10)

    if passes:
        status = "PASS"
    elif loose_pass:
        status = "WARN"
    else:
        status = "FAIL"

    return TensorComparison(
        name=name,
        shape_a=tuple(a.shape),
        shape_b=tuple(b.shape),
        max_abs_diff=max_abs,
        mean_abs_diff=mean_abs,
        median_abs_diff=median_abs,
        cosine_sim=cos_sim,
        relative_error=rel_err,
        a_mean=float(np.mean(a64)),
        b_mean=float(np.mean(b64)),
        a_std=float(np.std(a64)),
        b_std=float(np.std(b64)),
        has_nan_a=has_nan_a,
        has_nan_b=has_nan_b,
        status=status,
    )


def format_report(results: list[TensorComparison]) -> str:
    """Format comparison results as an aligned text table."""
    header = (
        f"{'Tensor':<25} {'Status':<15} {'MaxAbs':>10} {'MeanAbs':>10} "
        f"{'CosSim':>8} {'RelErr':>10} {'Shape':<20}"
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]

    for r in results:
        lines.append(
            f"{r.name:<25} {r.status:<15} {r.max_abs_diff:>10.2e} "
            f"{r.mean_abs_diff:>10.2e} {r.cosine_sim:>8.6f} "
            f"{r.relative_error:>10.2e} {str(r.shape_a):<20}"
        )

    lines.append(sep)

    n_pass = sum(1 for r in results if r.status == "PASS")
    n_warn = sum(1 for r in results if r.status == "WARN")
    n_fail = sum(1 for r in results if "FAIL" in r.status)
    total = len(results)

    lines.append(
        f"SUMMARY: {n_pass}/{total} PASS, "
        f"{n_warn} WARN, {n_fail} FAIL"
    )

    overall = "PASS" if n_fail == 0 else "FAIL"
    lines.append(f"OVERALL: {overall}")
    lines.append(sep)

    return "\n".join(lines)


def format_json(results: list[TensorComparison]) -> dict:
    """Format results as JSON-serializable dict."""
    comps = []
    for r in results:
        d = asdict(r)
        d["shape_a"] = list(r.shape_a)
        d["shape_b"] = list(r.shape_b)
        comps.append(d)

    n_pass = sum(1 for r in results if r.status == "PASS")
    n_fail = sum(1 for r in results if "FAIL" in r.status)

    return {
        "comparisons": comps,
        "summary": {
            "total": len(results),
            "pass": n_pass,
            "fail": n_fail,
            "overall": "PASS" if n_fail == 0 else "FAIL",
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare GPU reference outputs against JAX/TPU outputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("file_a", type=Path, help="First .npz file (e.g. gpu_reference.npz)")
    p.add_argument("file_b", type=Path, help="Second .npz file (e.g. tpu_output.npz)")
    p.add_argument(
        "--atol", type=float, default=BF16_ATOL,
        help=f"Absolute tolerance (default: {BF16_ATOL} for bf16).",
    )
    p.add_argument(
        "--rtol", type=float, default=BF16_RTOL,
        help=f"Relative tolerance (default: {BF16_RTOL} for bf16).",
    )
    p.add_argument(
        "--keys", type=str, default=None,
        help="Comma-separated list of keys to compare (default: all shared keys).",
    )
    p.add_argument(
        "--json", action="store_true",
        help="Output results as JSON.",
    )
    p.add_argument(
        "--strict", action="store_true",
        help="Exit with code 1 on any failure.",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print per-tensor statistics (mean, std) for both files.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    data_a = dict(np.load(str(args.file_a), allow_pickle=False))
    data_b = dict(np.load(str(args.file_b), allow_pickle=False))

    print(f"File A: {args.file_a} ({len(data_a)} arrays)")
    print(f"File B: {args.file_b} ({len(data_b)} arrays)")

    if args.keys:
        keys = [k.strip() for k in args.keys.split(",")]
    else:
        keys = sorted(set(data_a.keys()) & set(data_b.keys()))

    only_a = set(data_a.keys()) - set(data_b.keys())
    only_b = set(data_b.keys()) - set(data_a.keys())

    if only_a:
        print(f"\nOnly in A: {sorted(only_a)}")
    if only_b:
        print(f"\nOnly in B: {sorted(only_b)}")

    if not keys:
        print("\nERROR: No shared keys to compare.")
        sys.exit(1)

    print(f"\nComparing {len(keys)} shared tensors "
          f"(atol={args.atol:.0e}, rtol={args.rtol:.0e}):\n")

    results: list[TensorComparison] = []
    for key in keys:
        result = compare_tensor(key, data_a[key], data_b[key], args.atol, args.rtol)
        results.append(result)

        if args.verbose:
            print(f"  {key}:")
            print(f"    A: mean={result.a_mean:.6f} std={result.a_std:.6f} "
                  f"shape={result.shape_a}")
            print(f"    B: mean={result.b_mean:.6f} std={result.b_std:.6f} "
                  f"shape={result.shape_b}")

    if args.json:
        print(json.dumps(format_json(results), indent=2))
    else:
        print(format_report(results))

    if args.strict:
        any_fail = any("FAIL" in r.status for r in results)
        sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
