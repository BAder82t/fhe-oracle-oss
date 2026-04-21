# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
# Patent pending: PCT/IB2026/053378
"""Empirical verification of the Chernoff coverage bound (A4).

For each of the three mock circuits, at τ = median(max_error) from the
ρ=0.0 baseline in benchmarks/results/hybrid_warmstart.csv:

1. Draw B=100,000 uniform samples, measure divergence δ(x) for each.
2. Compute empirical μ_τ = fraction with δ ≥ τ.
3. Compute certificate's `p_disc_lower_bound(μ_τ)` at B_rand ∈ {50, 150, 250}.
4. Run 1000 bootstrap trials: draw B_rand uniform samples, check if
   max(δ) ≥ τ. Empirical discovery rate = fraction of trials that hit.
5. Assert: empirical discovery rate ≥ certificate bound - 0.05 (5 pp
   tolerance).

Output: prints a table to stdout; exits 0 if all assertions hold.
"""

from __future__ import annotations

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from fhe_oracle.guarantees import CoverageCertificate
from ablation_heuristics import make_circuit1, make_circuit2, make_circuit3


HYBRID_CSV = os.path.join(
    os.path.dirname(__file__), "results", "hybrid_warmstart.csv"
)

N_MU_ESTIMATE = 100_000  # uniform samples for μ_τ estimate
N_BOOTSTRAP = 1_000       # bootstrap trials per (circuit, B_rand)
B_RANDS = [50, 150, 250]
TOLERANCE_PP = 0.05       # 5 percentage points


def tau_from_hybrid_csv(circuit_name: str) -> float | None:
    """Extract τ = median(max_error) at ρ=0.0 for this circuit."""
    if not os.path.exists(HYBRID_CSV):
        return None
    vals = []
    with open(HYBRID_CSV) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if float(row["rho"]) == 0.0 and row["circuit"] == circuit_name:
                vals.append(float(row["max_error"]))
    return float(np.median(vals)) if vals else None


def evaluate_divergence(circuit: dict, x: np.ndarray) -> float:
    """Compute |plain - fhe| at x (max over vector output)."""
    p = circuit["plain"](x.tolist())
    f = circuit["fhe"](x.tolist())
    p_arr = np.atleast_1d(np.asarray(p, dtype=np.float64)).ravel()
    f_arr = np.atleast_1d(np.asarray(f, dtype=np.float64)).ravel()
    n = min(p_arr.size, f_arr.size)
    return float(np.max(np.abs(p_arr[:n] - f_arr[:n]))) if n > 0 else 0.0


def main() -> int:
    circuits = [make_circuit1(), make_circuit2(), make_circuit3()]

    print("Coverage Certificate Empirical Verification (A4)")
    print("=" * 90)
    print(
        f"{'circuit':<16s}  {'τ':>10s}  {'μ̂_τ':>10s}  "
        f"{'B_rand':>7s}  {'bound':>8s}  {'empirical':>10s}  "
        f"{'Δ(emp−bd)':>10s}  {'status':<6s}"
    )
    print("-" * 90)

    rng_top = np.random.default_rng(99999)
    all_pass = True
    rows_out = []

    for circuit in circuits:
        name = circuit["name"]
        bounds = circuit["bounds"]
        d = circuit["d"]
        lows = np.array([lo for lo, _ in bounds])
        highs = np.array([hi for _, hi in bounds])

        # τ from the baseline run
        tau = tau_from_hybrid_csv(name)
        if tau is None:
            print(f"  Skipping {name}: no baseline data at ρ=0.0 in hybrid_warmstart.csv")
            continue

        # Step 1-2: μ̂_τ from N_MU_ESTIMATE uniform samples.
        rng_mu = np.random.default_rng(rng_top.integers(0, 2**31))
        big_sample = rng_mu.uniform(lows, highs, size=(N_MU_ESTIMATE, d))
        divs = np.array([evaluate_divergence(circuit, x) for x in big_sample])
        hits_big = int(np.sum(divs >= tau))
        mu_hat = hits_big / N_MU_ESTIMATE

        # Step 3-5: for each B_rand, bound + bootstrap.
        for B in B_RANDS:
            cert = CoverageCertificate(
                budget_rand=B,
                threshold=tau,
                hits=max(1, int(round(B * mu_hat))),
                mu_hat=max(1, int(round(B * mu_hat))) / B,
            )
            # Use the bound at the empirical μ_τ (not the cert's own μ̂).
            bound = cert.p_disc_lower_bound(mu_hat)

            # Bootstrap empirical discovery rate.
            rng_boot = np.random.default_rng(rng_top.integers(0, 2**31))
            hits_boot = 0
            for _ in range(N_BOOTSTRAP):
                sub = rng_boot.uniform(lows, highs, size=(B, d))
                max_div = max(evaluate_divergence(circuit, x) for x in sub)
                if max_div >= tau:
                    hits_boot += 1
            empirical = hits_boot / N_BOOTSTRAP
            delta = empirical - bound
            passes = empirical >= bound - TOLERANCE_PP
            status = "OK" if passes else "FAIL"
            if not passes:
                all_pass = False

            print(
                f"{name:<16s}  {tau:>10.4e}  {mu_hat:>10.4e}  "
                f"{B:>7d}  {bound:>8.4f}  {empirical:>10.4f}  "
                f"{delta:>+10.4f}  {status:<6s}"
            )
            rows_out.append({
                "circuit": name,
                "tau": tau,
                "mu_hat": mu_hat,
                "B_rand": B,
                "bound": bound,
                "empirical": empirical,
                "delta": delta,
                "status": status,
            })
        print("-" * 90)

    out_path = os.path.join(
        os.path.dirname(__file__), "results", "coverage_certificate_verify.csv"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["circuit", "tau", "mu_hat", "B_rand", "bound",
                       "empirical", "delta", "status"],
        )
        writer.writeheader()
        for row in rows_out:
            writer.writerow(row)
    print(f"\nWrote {len(rows_out)} rows to {out_path}")
    print()
    print("All assertions pass." if all_pass else "Some assertions FAILED.")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
