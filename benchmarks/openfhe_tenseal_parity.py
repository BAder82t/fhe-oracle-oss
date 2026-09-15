# Copyright (C) 2026 Bader Alissaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""OpenFHE vs TenSEAL witness-error parity on LR d=8 Taylor-3 through OpenFHEAdapter.

Finds a witness with FHEOracle on OpenFHE, then re-measures |sigmoid - T3_FHE| on both
libraries at that witness and both box corners. Exits 1 if any relative gap exceeds 1e-3.

    python benchmarks/openfhe_tenseal_parity.py [--seed 41] [--n-trials 100]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(__file__))

from fhe_oracle import FHEOracle  # noqa: E402
from fhe_oracle.adapters.openfhe import OpenFHEAdapter  # noqa: E402
from fhe_oracle.adapters.tenseal_adapter import (  # noqa: E402
    TenSEALContext,
    make_tenseal_taylor3_fhe_fn,
)
from library_comparison import plaintext_lr_d8  # noqa: E402
from tenseal_circuits import _fit_lr_synthetic  # noqa: E402

TOL = 1e-3


def openfhe_lr_program(weights, bias):
    """Same ops as library_comparison.openfhe_lr_d8, as an adapter ``fhe_fn``."""
    w = np.asarray(weights, dtype=np.float64).tolist()

    def program(cc, ct):
        prod = cc.EvalMult(ct, cc.MakeCKKSPackedPlaintext(w))
        z = cc.EvalAdd(cc.EvalSum(prod, len(w)), float(bias))
        z3 = cc.EvalMult(cc.EvalMult(z, z), z)
        return cc.EvalAdd(cc.EvalAdd(cc.EvalMult(z, 0.25), cc.EvalMult(z3, -1.0 / 48.0)), 0.5)

    return program


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--n-trials", type=int, default=100)
    args = parser.parse_args()

    weights, bias = _fit_lr_synthetic(d=8, seed=42)
    plain = plaintext_lr_d8(weights, bias)
    adapter = OpenFHEAdapter(
        fhe_fn=openfhe_lr_program(weights, bias),
        n_features=8,
        mult_depth=6,
        scale_mod_size=40,
        output_length=1,
    )
    tenseal_fn = make_tenseal_taylor3_fhe_fn(weights, bias, TenSEALContext())

    result = FHEOracle(
        plaintext_fn=plain,
        adapter=adapter,
        input_dim=8,
        input_bounds=[(-3.0, 3.0)] * 8,
        seed=args.seed,
    ).run(n_trials=args.n_trials)
    print(f"FHEOracle on OpenFHE: verdict={result.verdict} max_error={result.max_error:.6f}")

    witnesses = {
        "oracle": list(result.worst_input),
        "corner+": (3.0 * np.sign(weights)).tolist(),
        "corner-": (-3.0 * np.sign(weights)).tolist(),
    }
    worst_gap = 0.0
    print(f"{'witness':8s} {'openfhe_err':>16s} {'tenseal_err':>16s} {'rel_diff':>10s}")
    for name, x in witnesses.items():
        openfhe_err = abs(plain(x) - adapter.evaluate(x)[0])
        tenseal_err = abs(plain(x) - tenseal_fn(x))
        gap = abs(openfhe_err - tenseal_err) / max(tenseal_err, 1e-12)
        worst_gap = max(worst_gap, gap)
        print(f"{name:8s} {openfhe_err:16.8f} {tenseal_err:16.8f} {gap:10.2e}")
    return 0 if worst_gap <= TOL else 1


if __name__ == "__main__":
    sys.exit(main())
