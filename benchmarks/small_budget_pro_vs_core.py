# Copyright (C) 2026 Bader Alissaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Real-CKKS small-budget Pro-vs-Core benchmark.

Implements the protocol in
``research/future-work/real-ckks-small-budget-benchmark.md``. Key
question: does Pro beat Core by >=1.15x on >=3 of the 5 circuits at
B <= 100, Wilcoxon p_FDR < 0.05?

Scope for this harness (v1):
  - Circuits: C1 (Chebyshev d=10) and C3 (WDBC d=30), the two that
    already have TenSEAL builders. C2, C4, C5 are NEW and left as
    TODO stubs.
  - Methods: Random / Core / Pro (full heuristic seeds). Lesion
    variants (Pro-MM, Pro-DS, Pro-NTE) are TODO.
  - Budgets: B in {60, 100, 500} for v1. Dossier also asks for
    B in {80, 200} which are trivial to add via CLI.
  - Metric: matched-domain max divergence per seed; Wilcoxon
    signed-rank on paired seeds.

Open Q 7.1 finding (see top-of-file context): TenSEAL's
``get_noise_budget`` returns a constant ``ct.scale()``. So the
``NoiseBudgetFitness`` Pro ships has no signal variance on TenSEAL.
The real Pro differentiator exercised here is the three named
heuristic seed generators (Claim 5 of PCT/IB2026/053378).

Usage::

    pip install fhe-oracle tenseal
    pip install -e fhe-oracle-pro   # adds Pro heuristics + lifts cap
    python benchmarks/small_budget_pro_vs_core.py
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass, asdict
from statistics import median
from typing import Callable

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(__file__))

from fhe_oracle import FHEOracle
from fhe_oracle.adapters.tenseal_adapter import HAVE_TENSEAL, TenSEALContext

try:
    from scipy.stats import wilcoxon
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False


@dataclass
class Cell:
    """(circuit, method, budget, seed) outcome."""
    circuit: str
    method: str
    budget: int
    seed: int
    max_error: float
    wall_seconds: float
    verdict: str


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------

def _run_oracle(plain_fn, fhe_fn, d, bounds, seed, budget,
                use_heuristic_seeds=False, heuristic_which=("mm", "ds", "nt"),
                threshold=1e-2, restarts=0, bipop=False):
    """Wrap an FHEOracle call, return (max_err, verdict)."""
    oracle = FHEOracle(
        plaintext_fn=plain_fn,
        fhe_fn=fhe_fn,
        input_dim=d,
        input_bounds=bounds,
        seed=seed,
        sigma0=None,  # Core's shipped default for robust basin behaviour
        use_heuristic_seeds=use_heuristic_seeds,
        heuristic_which=heuristic_which,
        heuristic_k=10,
        restarts=restarts,
        bipop=bipop,
    )
    result = oracle.run(n_trials=budget, threshold=threshold)
    return float(result.max_error), result.verdict


def _run_qg_pro(plain_fn, fhe_fn, d, bounds, seed, budget,
                theta=0.8, probe_frac=0.25, threshold=1e-2):
    """QG-Pro prototype: Quantile-Gated Pro with Ensemble Fallback.

    Stage 0: run Pro (all 3 heuristics) and Core in parallel with
             budget=B_probe/2 each.
    Stage 1: gate on b_pro >= theta * b_core.
    Stage 2a: if Pro wins probe, escalate to K=3 ensemble (one scorer each).
    Stage 2b: if Pro loses probe, revert — give full remaining budget to Core.

    Simplified from Item 20 §Problem 1 Phase 5 spec: external orchestration
    via FHEOracle API (no CMA-ES warm-start). Tests whether the gating
    logic alone recovers per-seed reliability.
    """
    B = budget
    B_probe = max(8, int(B * probe_frac))
    B_main = B - B_probe
    half = B_probe // 2

    # Stage 0: parallel probe
    e_pro, _ = _run_oracle(plain_fn, fhe_fn, d, bounds, seed, half,
                           use_heuristic_seeds=True, heuristic_which=("mm","ds","nt"),
                           threshold=threshold)
    e_core, _ = _run_oracle(plain_fn, fhe_fn, d, bounds, seed ^ 1, B_probe - half,
                            use_heuristic_seeds=False, threshold=threshold)

    best = max(e_pro, e_core)

    # Stage 1: gate
    if e_pro >= theta * max(e_core, 1e-12):
        # Stage 2a: escalate to K=3 ensemble, single-scorer each
        per_member = max(10, B_main // 3)
        for k, scorer in enumerate(("mm", "ds", "nt"), start=1):
            s_k = (seed * 0xC0FFEE + k * 0x9E37) & 0xFFFFFFFF
            e_k, _ = _run_oracle(plain_fn, fhe_fn, d, bounds, s_k, per_member,
                                 use_heuristic_seeds=True, heuristic_which=(scorer,),
                                 threshold=threshold)
            if e_k > best:
                best = e_k
        verdict = "FAIL" if best >= threshold else "PASS"
        return best, verdict
    else:
        # Stage 2b: revert to Core with main budget
        e_revert, _ = _run_oracle(plain_fn, fhe_fn, d, bounds, seed ^ 2, B_main,
                                  use_heuristic_seeds=False, threshold=threshold)
        best = max(best, e_revert)
        verdict = "FAIL" if best >= threshold else "PASS"
        return best, verdict


def _run_qg_gate_only(plain_fn, fhe_fn, d, bounds, seed, budget,
                       theta=0.8, probe_evals=10, threshold=1e-2):
    """Gate-only diagnostic: is the gate itself sound?

    No ensemble, no cold-start fragmentation, no seed remapping.
    Runs a 10-eval Pro probe and a 10-eval Core probe at seed=S,
    then gates a single FULL-budget run at seed=S of the winner.
    Cost: B + 2*probe_evals = 100 evals when B=80.

    Diagnostic value: discriminates two failure modes.
      - PASS: gate is sound; ensemble fragmentation was the QG-Pro FAIL cause.
      - FAIL: gate is broken; 10-eval probe cannot distinguish good/bad seeds
              → QG-Pro architecture is wrong, warm-start won't fix it.
    """
    # Stage 0: two probes at the SAME seed (not seed^1/^2 — that was the bug)
    e_pro_probe, _ = _run_oracle(plain_fn, fhe_fn, d, bounds, seed, probe_evals,
                                  use_heuristic_seeds=True,
                                  heuristic_which=("mm", "ds", "nt"),
                                  threshold=threshold)
    e_core_probe, _ = _run_oracle(plain_fn, fhe_fn, d, bounds, seed, probe_evals,
                                   use_heuristic_seeds=False, threshold=threshold)

    # Stage 1: gate
    escalate = e_pro_probe >= theta * max(e_core_probe, 1e-12)

    # Stage 2: single FULL-budget run of the winner at same seed
    if escalate:
        e_full, verdict = _run_oracle(plain_fn, fhe_fn, d, bounds, seed, budget,
                                       use_heuristic_seeds=True,
                                       heuristic_which=("mm", "ds", "nt"),
                                       threshold=threshold)
    else:
        e_full, verdict = _run_oracle(plain_fn, fhe_fn, d, bounds, seed, budget,
                                       use_heuristic_seeds=False, threshold=threshold)

    # Final = max over probe + full (probe may have found larger error by luck)
    best = max(e_pro_probe, e_core_probe, e_full)
    verdict = "FAIL" if best >= threshold else "PASS"
    return best, verdict


def _run_random(plain_fn, fhe_fn, bounds, seed, budget):
    """Uniform-random baseline at matched budget."""
    rng = np.random.default_rng(seed)
    lows = np.array([lo for lo, _ in bounds])
    highs = np.array([hi for _, hi in bounds])
    best = 0.0
    for _ in range(budget):
        x = rng.uniform(lows, highs)
        try:
            err = abs(float(plain_fn(x)) - float(fhe_fn(x)))
        except Exception:
            err = 0.0
        if err > best:
            best = err
    verdict = "FAIL" if best >= 1e-2 else "PASS"
    return best, verdict


METHODS: dict[str, Callable] = {
    "random":       lambda pl, fh, d, b, s, B: _run_random(pl, fh, b, s, B),
    "core":         lambda pl, fh, d, b, s, B: _run_oracle(pl, fh, d, b, s, B, use_heuristic_seeds=False),
    "core_ipop":    lambda pl, fh, d, b, s, B: _run_oracle(pl, fh, d, b, s, B, use_heuristic_seeds=False, restarts=3),
    "pro":          lambda pl, fh, d, b, s, B: _run_oracle(pl, fh, d, b, s, B, use_heuristic_seeds=True),
    "pro_ipop":     lambda pl, fh, d, b, s, B: _run_oracle(pl, fh, d, b, s, B, use_heuristic_seeds=True, restarts=3),
    "qg_pro":       lambda pl, fh, d, b, s, B: _run_qg_pro(pl, fh, d, b, s, B),
    "qg_gate_only": lambda pl, fh, d, b, s, B: _run_qg_gate_only(pl, fh, d, b, s, B),
    # Lesion rows (Pro-minus-X) -- resolve open Q 7.4 (which seed
    # matters). Drop one of ("mm","ds","nt") at a time.
    "pro_minus_mm": lambda pl, fh, d, b, s, B: _run_oracle(pl, fh, d, b, s, B, use_heuristic_seeds=True, heuristic_which=("ds","nt")),
    "pro_minus_ds": lambda pl, fh, d, b, s, B: _run_oracle(pl, fh, d, b, s, B, use_heuristic_seeds=True, heuristic_which=("mm","nt")),
    "pro_minus_nt": lambda pl, fh, d, b, s, B: _run_oracle(pl, fh, d, b, s, B, use_heuristic_seeds=True, heuristic_which=("mm","ds")),
}


# ---------------------------------------------------------------------------
# Circuit zoo
# ---------------------------------------------------------------------------

def build_circuits(which: list[str]):
    """Load only the requested circuits. Returns dict name->info."""
    if not HAVE_TENSEAL:
        raise SystemExit(
            "TenSEAL is required (`pip install tenseal`); none of the "
            "dossier's C1-C5 circuits can run without it."
        )

    from tenseal_circuits import (
        build_tenseal_chebyshev_d10,
        build_tenseal_wdbc,
        build_tenseal_inner_product_d16,
        build_tenseal_near_cliff_d8,
        build_tenseal_depth_chain_d8,
    )

    ctx = TenSEALContext()
    circuits = {}
    if "C1" in which:
        c = build_tenseal_chebyshev_d10(ctx)
        circuits["C1_cheb_d10"] = {**c, "_multi_output": True}
    if "C2" in which:
        c = build_tenseal_inner_product_d16(ctx)
        circuits["C2_inner_product_d16"] = {**c, "_multi_output": False}
    if "C3" in which:
        c = build_tenseal_wdbc(ctx)
        circuits["C3_wdbc_d30"] = {**c, "_multi_output": False}
    if "C4" in which:
        c = build_tenseal_near_cliff_d8(ctx)
        circuits["C4_near_cliff_d8"] = {**c, "_multi_output": False}
    if "C5" in which:
        c = build_tenseal_depth_chain_d8(ctx)
        circuits["C5_depth_chain_d8"] = {**c, "_multi_output": False}
    return circuits


def _max_abs_div(plain_fn, fhe_fn, multi_output: bool):
    """Wrap plaintext+fhe into divergence-friendly scalar callables for
    the oracle. For multi-output circuits, reduce via max-abs diff."""

    def plain_wrapped(x):
        res = plain_fn(x)
        if multi_output:
            return float(np.max(np.abs(np.asarray(res))))
        return float(res)

    def fhe_wrapped(x):
        res = fhe_fn(x)
        if multi_output:
            return float(np.max(np.abs(np.asarray(res))))
        return float(res)

    return plain_wrapped, fhe_wrapped


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

def run_matrix(circuits, methods, budgets, seeds, output_csv):
    cells: list[Cell] = []
    total = len(circuits) * len(methods) * len(budgets) * len(seeds)
    idx = 0

    for cname, cinfo in circuits.items():
        plain_r, fhe_r = _max_abs_div(
            cinfo["plain"], cinfo["fhe"], cinfo.get("_multi_output", False)
        )
        d = cinfo["d"]
        bounds = cinfo["bounds"]
        for B in budgets:
            for method in methods:
                fn = METHODS[method]
                for seed in seeds:
                    idx += 1
                    print(f"[{idx:3d}/{total}] {cname:15s} {method:15s} B={B:3d} s={seed:2d}", end=" ", flush=True)
                    t0 = time.perf_counter()
                    try:
                        max_err, verdict = fn(plain_r, fhe_r, d, bounds, seed, B)
                    except Exception as exc:
                        print(f"ERROR: {exc}")
                        cells.append(Cell(cname, method, B, seed, 0.0, 0.0, f"ERROR: {str(exc)[:50]}"))
                        continue
                    elapsed = time.perf_counter() - t0
                    print(f"max_err={max_err:.3e} verdict={verdict} t={elapsed:.1f}s")
                    cells.append(Cell(cname, method, B, seed, max_err, elapsed, verdict))

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(cells[0]).keys()))
        w.writeheader()
        for c in cells:
            w.writerow(asdict(c))
    print()
    print(f"Wrote {len(cells)} rows → {output_csv}")
    return cells


def summary_table(cells: list[Cell]):
    """Pro/Core ratio, Wilcoxon p per (circuit, B) cell."""
    print()
    print("Summary table: max_err medians + Pro/Core ratio")
    print("-" * 95)
    print(f"{'circuit':16s} {'B':>4s} {'random_med':>12s} {'core_med':>11s} "
          f"{'pro_med':>10s} {'pro/core':>10s} {'wilcoxon_p':>11s} {'n_seeds':>8s}")

    by_cell: dict[tuple[str, int], dict[str, list[float]]] = {}
    for c in cells:
        if c.verdict.startswith("ERROR"):
            continue
        key = (c.circuit, c.budget)
        by_cell.setdefault(key, {}).setdefault(c.method, []).append(c.max_error)

    rows = []
    for (circuit, B) in sorted(by_cell.keys()):
        d = by_cell[(circuit, B)]
        r = d.get("random", [])
        co = d.get("core", [])
        pr = d.get("pro", [])
        med_r = median(r) if r else 0.0
        med_c = median(co) if co else 0.0
        med_p = median(pr) if pr else 0.0
        ratio = (med_p / med_c) if med_c > 0 else float("inf")
        if _HAVE_SCIPY and len(co) >= 3 and len(pr) == len(co):
            paired = sorted(zip(range(len(co)), co, pr))
            cc = [x for _, x, _ in paired]
            pp = [y for _, _, y in paired]
            try:
                _, pval = wilcoxon(
                    np.log(np.maximum(pp, 1e-30)) - np.log(np.maximum(cc, 1e-30))
                )
                pstr = f"{pval:.3f}"
            except Exception:
                pstr = "-"
        else:
            pstr = "-"
        n = len(co)
        print(f"{circuit:16s} {B:>4d} {med_r:>12.3e} {med_c:>11.3e} "
              f"{med_p:>10.3e} {ratio:>10.3f}x {pstr:>11s} {n:>8d}")
        rows.append((circuit, B, med_r, med_c, med_p, ratio, pstr, n))

    # Lesion summary
    print()
    print("Heuristic lesion: pro_minus_X vs pro (how much X contributes)")
    print("-" * 75)
    print(f"{'circuit':16s} {'B':>4s} {'pro':>10s} {'-MM':>10s} "
          f"{'-DS':>10s} {'-NT':>10s}")
    for (circuit, B), d in sorted(by_cell.items()):
        if "pro" not in d:
            continue
        pm = {k: (median(v) if v else 0.0) for k, v in d.items()}
        print(f"{circuit:16s} {B:>4d} "
              f"{pm.get('pro', 0):>10.3e} "
              f"{pm.get('pro_minus_mm', 0):>10.3e} "
              f"{pm.get('pro_minus_ds', 0):>10.3e} "
              f"{pm.get('pro_minus_nt', 0):>10.3e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--circuits", nargs="+", default=["C1", "C3"],
                        help="Which circuits to run (C1, C3 implemented; C2/C4/C5 TODO)")
    parser.add_argument("--methods", nargs="+",
                        default=["random", "core", "pro"])
    parser.add_argument("--budgets", nargs="+", type=int,
                        default=[60, 100, 500])
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[41, 42, 43, 44, 45])
    parser.add_argument("--output",
                        default="benchmarks/results/small_budget_pro_vs_core.csv")
    parser.add_argument("--lesion", action="store_true",
                        help="Add pro_minus_mm/ds/nt lesion rows")
    args = parser.parse_args()

    methods = list(args.methods)
    if args.lesion:
        for m in ("pro_minus_mm", "pro_minus_ds", "pro_minus_nt"):
            if m not in methods:
                methods.append(m)

    print(f"Building circuits: {args.circuits}")
    circuits = build_circuits(args.circuits)
    print(f"Built {len(circuits)}: {list(circuits.keys())}")
    print()
    print(f"Methods: {methods}")
    print(f"Budgets: {args.budgets}")
    print(f"Seeds:   {args.seeds}")
    print()

    cells = run_matrix(circuits, methods, args.budgets, args.seeds, args.output)
    summary_table(cells)
    return 0


if __name__ == "__main__":
    sys.exit(main())
