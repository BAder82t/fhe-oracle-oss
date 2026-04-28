# Copyright (C) 2026 Bader Alissaei
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Item 17 measurement probe: level-proportional proxy vs Lattigo
decrypt-based ground-truth precision.

Runs N uniform random inputs through both:

1. The level-proportional proxy currently used by the Oracle's
   ``InstrumentedFitness`` (``depth_term = min(1.0, max|x| / 3)``).
   This is the fitness component the regime router in
   ``autoconfig.py`` relies on when ``w_depth > 0``.
2. The Lattigo CKKS probe (``benchmarks/lattigo_probe/lattigo_probe``),
   which encrypts each input, evaluates ``(w·x + b)^2``, decrypts,
   and reports bits-of-precision per input via the Lattigo
   ``GetPrecisionStats`` path.

Computes Spearman correlation between the proxy and the
ground-truth error magnitude (``error = 2^(-bits_of_precision)``).
A correlation $\\rho \\geq 0.7$ validates the proxy for the regime
router; lower values indicate the proxy needs replacing (Item 14
autoconfig v2 follow-up).

Usage
-----
    python benchmarks/item17_lattigo_correlation.py \\
        --n 100 --seed 42 \\
        --out benchmarks/results/item17_lattigo_correlation.csv

The Lattigo probe binary must be built first:

    cd benchmarks/lattigo_probe && go build .
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.stats import spearmanr

# Make `fhe_oracle` importable when running from repo root or from the
# benchmarks/ directory.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fhe_oracle.adapters.lattigo import LattigoProbe, LattigoProbeError


# ----------------------------------------------------------------------
# Proxies
# ----------------------------------------------------------------------


def level_proportional_proxy(x: np.ndarray) -> float:
    """Current Oracle proxy: ``min(1.0, max|x| / 3)``.

    Source: ``fhe_oracle/diagnostics.py:120`` (``InstrumentedFitness``).
    Used by the regime router when ``w_depth > 0``.
    """
    return float(min(1.0, np.max(np.abs(x)) / 3.0))


def l2_norm_proxy(x: np.ndarray) -> float:
    """Alternative: ``||x||_2``. Predicts noise growth in
    ``(w·x + b)^2``-style circuits where the result magnitude
    drives quantisation error.
    """
    return float(np.linalg.norm(x, 2))


def squared_norm_proxy(x: np.ndarray) -> float:
    """Alternative: ``||x||_2^2``. Matches the depth-2 multiplicative
    structure of the Lattigo probe circuit.
    """
    return float(np.dot(x, x))


# ----------------------------------------------------------------------
# Experiment
# ----------------------------------------------------------------------


def generate_inputs(
    n: int,
    dim: int,
    bound: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-bound, bound, size=(n, dim))


def run_correlation(
    n: int,
    dim: int,
    bound: float,
    seed: int,
    w: Sequence[float] | None = None,
    b: float = 0.5,
    out_csv: Path | None = None,
    binary: Path | None = None,
) -> dict[str, float]:
    """Run the probe + proxies and return Spearman rho per proxy."""

    inputs = generate_inputs(n=n, dim=dim, bound=bound, seed=seed)

    if w is None:
        rng = np.random.default_rng(seed + 1)
        w = rng.normal(0, 0.5, size=dim).tolist()

    probe = LattigoProbe(binary=binary)
    rows = probe.precision_per_input(
        inputs=[list(map(float, x)) for x in inputs],
        w=list(map(float, w)),
        b=float(b),
    )

    bits = np.array([r.mean_bits for r in rows], dtype=np.float64)
    error = np.power(2.0, -bits)  # ground truth: smaller bits => larger error

    proxies = {
        "level_proportional": np.array(
            [level_proportional_proxy(x) for x in inputs],
            dtype=np.float64,
        ),
        "l2_norm": np.array(
            [l2_norm_proxy(x) for x in inputs],
            dtype=np.float64,
        ),
        "squared_norm": np.array(
            [squared_norm_proxy(x) for x in inputs],
            dtype=np.float64,
        ),
    }

    rhos: dict[str, float] = {}
    pvals: dict[str, float] = {}
    for name, vec in proxies.items():
        # We expect proxy and error to be positively correlated:
        # higher proxy => higher predicted noise => higher error.
        result = spearmanr(vec, error)
        rhos[name] = float(result.statistic)
        pvals[name] = float(result.pvalue)

    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                "idx",
                *(f"x_{i}" for i in range(dim)),
                "bits_of_precision",
                "error_2to_neg_bits",
                "level_proportional",
                "l2_norm",
                "squared_norm",
                "plaintext_value",
                "decrypted_real",
            ])
            for i, (x, row) in enumerate(zip(inputs, rows)):
                writer.writerow([
                    i,
                    *(f"{xi:.6f}" for xi in x),
                    f"{row.mean_bits:.4f}",
                    f"{error[i]:.6e}",
                    f"{proxies['level_proportional'][i]:.6f}",
                    f"{proxies['l2_norm'][i]:.6f}",
                    f"{proxies['squared_norm'][i]:.6f}",
                    f"{row.plaintext_value:.6e}",
                    f"{row.decrypted_real:.6e}",
                ])

    return {
        "n": n,
        "dim": dim,
        "bound": bound,
        "seed": seed,
        **{f"rho_{name}": rho for name, rho in rhos.items()},
        **{f"pval_{name}": pv for name, pv in pvals.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100, help="number of inputs")
    parser.add_argument("--dim", type=int, default=1, help="input dimension")
    parser.add_argument("--bound", type=float, default=3.0, help="box half-width")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("benchmarks/results/item17_lattigo_correlation.csv"))
    parser.add_argument("--binary", type=Path, default=None)
    args = parser.parse_args()

    try:
        summary = run_correlation(
            n=args.n,
            dim=args.dim,
            bound=args.bound,
            seed=args.seed,
            out_csv=args.out,
            binary=args.binary,
        )
    except LattigoProbeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(
            "Build the Go binary first: cd benchmarks/lattigo_probe && go build .",
            file=sys.stderr,
        )
        return 2

    print("Item 17 correlation summary")
    print("---------------------------")
    print(f"n          = {summary['n']}")
    print(f"dim        = {summary['dim']}")
    print(f"bound      = {summary['bound']}")
    print(f"seed       = {summary['seed']}")
    print()
    print("Spearman rho between proxy and error magnitude:")
    for name in ("level_proportional", "l2_norm", "squared_norm"):
        rho = summary[f"rho_{name}"]
        pv = summary[f"pval_{name}"]
        verdict = "VALIDATED" if rho >= 0.7 else "WEAK" if rho >= 0.4 else "INVALID"
        print(f"  {name:<22s} rho = {rho:+.4f}  p = {pv:.2e}  [{verdict}]")
    print()
    print(f"CSV: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
