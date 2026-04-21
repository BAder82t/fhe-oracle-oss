# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""n=20 real-CKKS lesion audit (resolves paper Limitation 9).

Two parts:

Part A — Extends the existing LR d=8 TenSEAL lesion audit from
n=10 to n=20 seeds. Appends the new rows to
benchmarks/results/tenseal_ablation.csv (keeping seeds 0..9
identical for back-compat).

Part B — Runs the full 9-config lesion audit on TenSEAL Circuit 2
(depth-4 polynomial, d=6) at n=20 seeds. This is a second real-CKKS
lesion data point on a pure-CKKS-noise circuit (no polynomial
approximation confound). Writes a fresh CSV.

After both runs finish, produces a consolidated summary with:
  - ratio (FULL/lesion mean)
  - wins out of 20
  - p_uncorrected (paired Wilcoxon signed-rank, FULL vs lesion max_error)
  - p_Holm (Holm-Bonferroni over the 8 non-baseline lesions)
  - verdict: LOAD_BEARING / INCONCLUSIVE / INERT

Outputs:
  benchmarks/results/tenseal_ablation.csv                   (appended)
  benchmarks/results/tenseal_ablation_c2_n20.csv            (new)
  benchmarks/results/tenseal_ablation_n20.csv               (consolidated)
  benchmarks/results/lesion_audit_n20_summary.csv           (verdicts)
"""

from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np
from scipy.stats import wilcoxon

THIS_DIR = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, THIS_DIR)

from fhe_oracle import FHEOracle  # noqa: E402
from fhe_oracle.adapters.tenseal_adapter import HAVE_TENSEAL, TenSEALContext  # noqa: E402

from tenseal_circuits import (  # noqa: E402
    build_tenseal_circuit2,
    build_tenseal_lr_d8,
)
from tenseal_ablation import CONFIGS, run_cell as run_cell_lr  # noqa: E402


LR_NEW_SEEDS = list(range(10, 20))
ALL_SEEDS = list(range(20))
RESULTS = os.path.join(THIS_DIR, "results")
BUDGET = 60


# --- Circuit 2 lesion cell (mirrors tenseal_ablation.run_cell but for C2)

def run_cell_c2(
    config_name: str, cfg: dict, circuit: dict, seed: int, budget: int,
) -> dict:
    """Run one (config, C2, seed) cell with CKKS-native proxies.

    Circuit 2 has no (W, b) — it's a pure polynomial. The noise
    proxy is max|x_i|^2 * |x_{i+1}| (intermediate magnitude), which
    monotonically tracks CKKS multiplication noise growth on this
    circuit. Depth proxy is ||x||_inf, same as LR.
    """
    w_div, w_noise, w_depth = cfg["w"]
    seed_heuristics = cfg["seeds"]
    plain = circuit["plain"]
    fhe = circuit["fhe"]
    d = circuit["d"]
    bounds = circuit["bounds"]

    class C2CfgFitness:
        def score(self, x):
            xa = np.asarray(x, dtype=np.float64)
            try:
                p = plain(x)
                f = fhe(x)
            except Exception:
                return 0.0
            divergence = abs(float(p) - float(f))
            inter = xa[:-1] ** 2 * xa[1:]
            max_mag = float(np.max(np.abs(inter))) if inter.size else 0.0
            noise_term = min(1.0, max_mag / 8.0)  # [-2,2]^6 => max ~ 8
            depth_term = min(1.0, float(np.max(np.abs(xa))) / 2.0)
            return (
                w_div * divergence
                + w_noise * noise_term
                + w_depth * depth_term
            )

    k = 10 if seed_heuristics else 0
    oracle = FHEOracle(
        plaintext_fn=plain, fhe_fn=fhe,
        input_dim=d, input_bounds=bounds,
        fitness=C2CfgFitness(),
        seed=seed,
        use_heuristic_seeds=bool(seed_heuristics),
        heuristic_which=tuple(seed_heuristics) if seed_heuristics else ("mm", "ds", "nt"),
        heuristic_k=k,
    )
    t0 = time.perf_counter()
    res = oracle.run(n_trials=budget, threshold=0.0)
    wall = time.perf_counter() - t0
    return {
        "config": config_name,
        "circuit": "circuit2_tenseal",
        "seed": seed,
        "weights": f"({w_div},{w_noise},{w_depth})",
        "seeds_used": ",".join(seed_heuristics) if seed_heuristics else "",
        "max_error": res.max_error,
        "wall_clock_s": wall,
        "n_trials": res.n_trials,
        "verdict": res.verdict,
    }


# --- Append helpers ---------------------------------------------------------

def _append_rows(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    exists = os.path.exists(path)
    with open(path, "a" if exists else "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def _read_rows(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh))


# --- Part A: LR n=20 (append seeds 10..19 to tenseal_ablation.csv) --------

def run_lr_n20() -> None:
    if not HAVE_TENSEAL:
        print("[A] TenSEAL unavailable; skipping")
        return
    print("=" * 70)
    print(f"[A] LR d=8 lesion audit seeds {LR_NEW_SEEDS} @ B={BUDGET}")
    print("=" * 70)
    ctx = TenSEALContext()
    lr = build_tenseal_lr_d8(ctx)
    rows: list[dict] = []
    total = len(CONFIGS) * len(LR_NEW_SEEDS)
    k = 0
    t_start = time.perf_counter()
    for cfg_name, cfg in CONFIGS.items():
        for seed in LR_NEW_SEEDS:
            k += 1
            row = run_cell_lr(cfg_name, cfg, lr, seed, BUDGET)
            rows.append(row)
            print(
                f"  [{k:3d}/{total}] {cfg_name:>5s} seed={seed:2d} "
                f"err={row['max_error']:.4e} t={row['wall_clock_s']:.1f}s"
            )
    print(f"  wall-clock: {(time.perf_counter()-t_start):.1f}s")
    _append_rows(
        os.path.join(RESULTS, "tenseal_ablation.csv"),
        rows,
        [
            "config", "circuit", "seed", "weights", "seeds_used",
            "max_error", "wall_clock_s", "n_trials", "verdict",
        ],
    )


# --- Part B: Circuit 2 full n=20 lesion audit -----------------------------

def run_c2_n20() -> None:
    if not HAVE_TENSEAL:
        print("[B] TenSEAL unavailable; skipping")
        return
    print("=" * 70)
    print(f"[B] Circuit 2 lesion audit all configs × seeds 0..19 @ B={BUDGET}")
    print("=" * 70)
    ctx = TenSEALContext()
    circuit = build_tenseal_circuit2(ctx)
    rows: list[dict] = []
    total = len(CONFIGS) * len(ALL_SEEDS)
    k = 0
    t_start = time.perf_counter()
    for cfg_name, cfg in CONFIGS.items():
        for seed in ALL_SEEDS:
            k += 1
            row = run_cell_c2(cfg_name, cfg, circuit, seed, BUDGET)
            rows.append(row)
            if k % 10 == 0 or k == total:
                print(
                    f"  [{k:3d}/{total}] {cfg_name:>5s} seed={seed:2d} "
                    f"err={row['max_error']:.4e} t={row['wall_clock_s']:.1f}s"
                )
    print(f"  wall-clock: {(time.perf_counter()-t_start):.1f}s")
    out = os.path.join(RESULTS, "tenseal_ablation_c2_n20.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "config", "circuit", "seed", "weights", "seeds_used",
            "max_error", "wall_clock_s", "n_trials", "verdict",
        ])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  wrote {len(rows)} rows to {out}")


# --- Summary: verdict per (config, circuit) ---------------------------------

def _holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni adjustment. Input/output in original order."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        raw = pvalues[idx] * (m - rank)
        running = max(running, min(1.0, raw))
        adj[idx] = running
    return adj


def build_summary() -> None:
    print("=" * 70)
    print("[summary] consolidating LR + C2 n=20 lesion audit")
    print("=" * 70)
    lr_rows = _read_rows(os.path.join(RESULTS, "tenseal_ablation.csv"))
    c2_rows = _read_rows(os.path.join(RESULTS, "tenseal_ablation_c2_n20.csv"))
    consolidated = lr_rows + c2_rows

    # Write consolidated
    out = os.path.join(RESULTS, "tenseal_ablation_n20.csv")
    fields = [
        "config", "circuit", "seed",
        "oracle_max_error", "ratio", "wins",
    ]
    # Row-per-(config, circuit, seed) with FULL reference.
    by_circ_seed: dict[tuple[str, int], dict[str, float]] = {}
    for r in consolidated:
        circ = r["circuit"]
        seed = int(r["seed"])
        by_circ_seed.setdefault((circ, seed), {})[r["config"]] = float(r["max_error"])

    consolidated_rows = []
    for (circ, seed), cfgs in sorted(by_circ_seed.items()):
        full = cfgs.get("FULL", float("nan"))
        for cfg_name in CONFIGS.keys():
            v = cfgs.get(cfg_name)
            if v is None:
                continue
            ratio = (full / v) if v > 0 else float("inf")
            consolidated_rows.append({
                "config": cfg_name,
                "circuit": circ,
                "seed": seed,
                "oracle_max_error": v,
                "ratio": ratio,
                "wins": int(full > v),
            })
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in consolidated_rows:
            w.writerow(r)
    print(f"  wrote {len(consolidated_rows)} rows to {out}")

    # Summary per (circuit): FULL is baseline, the 8 non-baseline lesions
    # (DIV, -N, -D, -ND, -S, -MM, -DS, -NT) get Holm correction.
    summary_path = os.path.join(RESULTS, "lesion_audit_n20_summary.csv")
    summary_rows: list[dict] = []
    non_baseline = [c for c in CONFIGS.keys() if c != "FULL"]

    for circ in ("lr_d8_tenseal", "circuit2_tenseal"):
        seeds_present = sorted(set(
            s for (c, s) in by_circ_seed.keys() if c == circ
        ))
        if not seeds_present:
            continue
        full_values = [
            by_circ_seed[(circ, s)].get("FULL", float("nan"))
            for s in seeds_present
        ]
        p_vals: list[float] = []
        per_config: list[dict] = []
        for cfg in non_baseline:
            lesion = [
                by_circ_seed[(circ, s)].get(cfg, float("nan"))
                for s in seeds_present
            ]
            paired = [
                (f, l) for f, l in zip(full_values, lesion)
                if np.isfinite(f) and np.isfinite(l)
            ]
            if not paired:
                continue
            f_arr = np.array([p[0] for p in paired])
            l_arr = np.array([p[1] for p in paired])
            ratios = [(f / l) if l > 0 else float("inf") for f, l in paired]
            ratios_fin = [r for r in ratios if np.isfinite(r)]
            wins = int(np.sum(f_arr > l_arr))
            n = len(paired)
            mean_ratio = float(np.mean(ratios_fin)) if ratios_fin else float("nan")
            if np.allclose(f_arr, l_arr):
                p_unc = 1.0
            else:
                try:
                    p_unc = float(wilcoxon(f_arr, l_arr).pvalue)
                except Exception:
                    p_unc = float("nan")
            p_vals.append(p_unc)
            per_config.append({
                "circuit": circ,
                "config": cfg,
                "n": n,
                "mean_full": float(np.mean(f_arr)),
                "mean_lesion": float(np.mean(l_arr)),
                "mean_ratio": mean_ratio,
                "wins": wins,
                "p_uncorrected": p_unc,
            })
        adj = _holm(p_vals)
        for row, p_adj in zip(per_config, adj):
            # Verdict rules per dossier §3.2 / S0 audit:
            # LOAD_BEARING : R ≥ 1.10 AND wins ≥ 0.7·n AND p_Holm < 0.05
            # INERT        : |R − 1| < 0.10 AND 0.4n ≤ wins ≤ 0.6n AND p_Holm ≥ 0.05
            # INCONCLUSIVE : otherwise
            R = row["mean_ratio"]
            n = row["n"]
            wins_frac = row["wins"] / n if n else 0.0
            if (
                np.isfinite(R) and R >= 1.10 and wins_frac >= 0.70
                and p_adj < 0.05
            ):
                verdict = "LOAD_BEARING"
            elif (
                np.isfinite(R) and abs(R - 1.0) < 0.10
                and 0.40 <= wins_frac <= 0.60 and p_adj >= 0.05
            ):
                verdict = "INERT"
            else:
                verdict = "INCONCLUSIVE"
            row["p_holm"] = p_adj
            row["verdict"] = verdict
            summary_rows.append(row)
            print(
                f"  {circ:>18s}  {row['config']:>5s}  "
                f"R={R:.3f}  wins={row['wins']}/{n}  "
                f"p_unc={row['p_uncorrected']:.4f}  p_Holm={p_adj:.4f}  "
                f"=> {verdict}"
            )

    with open(summary_path, "w", newline="") as fh:
        fieldnames = [
            "circuit", "config", "n",
            "mean_full", "mean_lesion", "mean_ratio", "wins",
            "p_uncorrected", "p_holm", "verdict",
        ]
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)
    print(f"  wrote {len(summary_rows)} rows to {summary_path}")


def main() -> int:
    steps = set(sys.argv[1:]) if len(sys.argv) > 1 else {"A", "B", "summary"}
    if "A" in steps:
        run_lr_n20()
    if "B" in steps:
        run_c2_n20()
    if "summary" in steps:
        build_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
