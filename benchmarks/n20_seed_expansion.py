# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""n=20 seed expansion wrapper (resolves paper Limitation 8).

Extends all benchmark sweeps from n=10 to n=20 seeds by re-invoking
the existing benchmark functions on seeds 10..19 and APPENDING rows
to existing CSVs. Also computes matched uniform-random baselines on
the 3 mock circuits of Table 1 (seeds 0..19) since those baselines
are not stored in ablation_heuristics.csv.

No existing script is modified — this file imports the already-tested
per-cell primitives (run_one_cell, run_oracle, run_random_baseline,
run_cell) and drives them at the new seed range.

Outputs:
  benchmarks/results/ablation_heuristics.csv                (appended)
  benchmarks/results/tenseal_validation.csv                 (appended)
  benchmarks/results/tenseal_circuit2_validation.csv        (appended)
  benchmarks/results/external_sweep.csv                     (appended)
  benchmarks/results/table1_random_baselines_n20.csv        (new)
  benchmarks/results/n20_expansion_summary.csv              (new)
"""

from __future__ import annotations

import csv
import os
import sys
import time
from typing import Any, Iterable

import numpy as np
from scipy.stats import wilcoxon

THIS_DIR = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, THIS_DIR)

from fhe_oracle import FHEOracle  # noqa: E402
from fhe_oracle.adapters.tenseal_adapter import HAVE_TENSEAL, TenSEALContext  # noqa: E402
from fhe_oracle.empirical import EmpiricalSearch  # noqa: E402
from fhe_oracle.hybrid import _default_divergence_fn, run_hybrid  # noqa: E402

# Re-use primitives from existing benchmark modules.
from ablation_heuristics import (  # noqa: E402
    CONFIGS as ABLATION_CONFIGS,
    make_circuit1,
    make_circuit2,
    make_circuit3,
    run_one_cell as run_ablation_cell,
)
from tenseal_circuits import (  # noqa: E402
    build_tenseal_chebyshev_d10,
    build_tenseal_circuit2,
    build_tenseal_lr_d8,
    build_tenseal_wdbc,
)
from tenseal_validation import (  # noqa: E402
    B_TENSEAL as B_TENSEAL_VALIDATION,
    run_empirical_on_wdbc,
    run_oracle as run_tenseal_oracle,
    run_random_baseline as run_tenseal_random,
)
from tenseal_circuit2_validation import (  # noqa: E402
    ADVERSARIAL_BOUNDS as C2_ADV,
    B_TENSEAL as B_TENSEAL_C2,
    OPERATIONAL_BOUNDS as C2_OP,
)
from external_sweep import (  # noqa: E402
    BUDGET as EXT_BUDGET,
    JITTER as EXT_JITTER,
    THRESHOLD as EXT_THRESHOLD,
    _bounds_for,
    _random_baseline_result,
    run_circuit_cells,  # handles oracle_only, random_only, empirical_only, hybrid_union per seed
)
from wdbc_mock import build_wdbc_circuit  # noqa: E402
from mnist_mock import build_mnist_circuit  # noqa: E402


NEW_SEEDS = list(range(10, 20))
ALL_SEEDS = list(range(20))
RESULTS = os.path.join(THIS_DIR, "results")


# --- CSV append helper ------------------------------------------------------

def _append_rows(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    exists = os.path.exists(path)
    mode = "a" if exists else "w"
    with open(path, mode, newline="") as fh:
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


# --- Exp 1a: ablation_heuristics seeds 10..19 ------------------------------

def exp_1a_ablation(n_trials: int = 500) -> None:
    print("\n" + "=" * 70)
    print(f"[1a] ablation_heuristics seeds {NEW_SEEDS} @ B={n_trials}")
    print("=" * 70)
    circuits = [make_circuit1(), make_circuit2(), make_circuit3()]
    rows: list[dict] = []
    total = len(ABLATION_CONFIGS) * len(circuits) * len(NEW_SEEDS)
    k = 0
    t0 = time.perf_counter()
    for cfg_name, cfg in ABLATION_CONFIGS.items():
        for circuit in circuits:
            for seed in NEW_SEEDS:
                k += 1
                row = run_ablation_cell(cfg_name, cfg, circuit, seed, n_trials)
                rows.append(row)
                if k % 25 == 0 or k == total:
                    print(
                        f"  [{k:3d}/{total}] {cfg_name:>5s} {circuit['name']:<16s} "
                        f"seed={seed} err={row['max_error']:.3e} "
                        f"t={row['wall_clock_s']:.2f}s"
                    )
    elapsed = time.perf_counter() - t0
    print(f"  wall-clock: {elapsed:.1f}s")
    _append_rows(
        os.path.join(RESULTS, "ablation_heuristics.csv"),
        rows,
        [
            "config", "circuit", "seed", "weights", "seeds_used",
            "max_error", "worst_input", "wall_clock_s", "n_trials", "verdict",
        ],
    )
    print(f"  appended {len(rows)} rows")


# --- Exp 1a-supplement: Table-1 paired random baselines (seeds 0..19) ------
#
# Paper Table 1 compares FULL oracle vs uniform random at B=500 on the three
# mock circuits. The existing ablation_heuristics.csv only stores the oracle
# legs — regenerate the random legs for all seeds so the win-rate at n=20 is
# computable without recourse to external rerun.

def _uniform_random_on_circuit(circuit: dict, budget: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    lows = np.array([lo for lo, _ in circuit["bounds"]])
    highs = np.array([hi for _, hi in circuit["bounds"]])
    best = 0.0
    for _ in range(budget):
        x = rng.uniform(lows, highs)
        try:
            p = circuit["plain"](x.tolist())
            f = circuit["fhe"](x.tolist())
        except Exception:
            continue
        p_arr = np.atleast_1d(np.asarray(p, dtype=np.float64)).ravel()
        f_arr = np.atleast_1d(np.asarray(f, dtype=np.float64)).ravel()
        n = min(p_arr.size, f_arr.size)
        if n == 0:
            continue
        err = float(np.max(np.abs(p_arr[:n] - f_arr[:n])))
        if err > best:
            best = err
    return best


def exp_1a_random_baselines(n_trials: int = 500) -> None:
    print("\n" + "=" * 70)
    print(f"[1a.random] Table-1 uniform-random baselines on 3 mocks, seeds 0..19")
    print("=" * 70)
    circuits = [make_circuit1(), make_circuit2(), make_circuit3()]
    rows: list[dict] = []
    t0 = time.perf_counter()
    for circuit in circuits:
        for seed in ALL_SEEDS:
            err = _uniform_random_on_circuit(circuit, n_trials, seed)
            rows.append({
                "circuit": circuit["name"],
                "seed": seed,
                "budget": n_trials,
                "random_max_error": err,
            })
    print(f"  wall-clock: {time.perf_counter() - t0:.1f}s")
    path = os.path.join(RESULTS, "table1_random_baselines_n20.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["circuit", "seed", "budget", "random_max_error"]
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  wrote {len(rows)} rows to {path}")


# --- Exp 1b: tenseal_validation seeds 10..19 -------------------------------

def exp_1b_tenseal_validation() -> None:
    if not HAVE_TENSEAL:
        print("[1b] TenSEAL not available, skipping")
        return
    print("\n" + "=" * 70)
    print(f"[1b] tenseal_validation seeds {NEW_SEEDS} @ B={B_TENSEAL_VALIDATION}")
    print("=" * 70)
    ctx_lr = TenSEALContext()
    lr = build_tenseal_lr_d8(ctx_lr)
    ctx_ch = TenSEALContext()
    ch = build_tenseal_chebyshev_d10(ctx_ch)
    ctx_wdbc = TenSEALContext()
    wdbc = build_tenseal_wdbc(ctx_wdbc)

    rows: list[dict] = []
    t_total = time.perf_counter()

    for label, circuit, random_floor in [
        ("lr_matched", lr, 0.0),
        ("cheb_matched", ch, 0.0),
        ("cheb_warm", ch, 0.3),
        ("wdbc_matched", wdbc, 0.0),
    ]:
        print(f"  [{label}]")
        for seed in NEW_SEEDS:
            t0 = time.perf_counter()
            ores = run_tenseal_oracle(
                circuit, seed, B_TENSEAL_VALIDATION,
                random_floor=random_floor,
            )
            rnd_err = run_tenseal_random(
                circuit["plain"], circuit["fhe"], circuit["bounds"],
                B_TENSEAL_VALIDATION, seed,
            )
            ratio = ores.max_error / rnd_err if rnd_err > 0 else float("inf")
            print(
                f"    seed={seed:2d} ora={ores.max_error:.3e} "
                f"rnd={rnd_err:.3e} R={ratio:.2f}x t={time.perf_counter()-t0:.1f}s"
            )
            rows.append({
                "experiment": label, "circuit": circuit["name"], "seed": seed,
                "oracle_max_error": ores.max_error, "random_max_error": rnd_err,
                "ratio_oracle_over_random": ratio,
                "oracle_wins_random": int(ores.max_error > rnd_err),
                "empirical_max_error": 0.0,
                "wall_clock_s": time.perf_counter() - t0,
            })

    # wdbc_asymmetric (oracle vs empirical data distribution)
    print("  [wdbc_asymmetric]")
    for seed in NEW_SEEDS:
        t0 = time.perf_counter()
        ores = run_tenseal_oracle(wdbc, seed, B_TENSEAL_VALIDATION)
        eres = run_empirical_on_wdbc(wdbc, seed, B_TENSEAL_VALIDATION)
        print(
            f"    seed={seed:2d} ora={ores.max_error:.3e} "
            f"emp={eres.max_error:.3e} t={time.perf_counter()-t0:.1f}s"
        )
        rows.append({
            "experiment": "wdbc_asymmetric", "circuit": "wdbc_tenseal", "seed": seed,
            "oracle_max_error": ores.max_error, "random_max_error": 0.0,
            "ratio_oracle_over_random": 0.0,
            "oracle_wins_random": int(ores.max_error > eres.max_error),
            "empirical_max_error": eres.max_error,
            "wall_clock_s": time.perf_counter() - t0,
        })

    # wdbc_hybrid
    print("  [wdbc_hybrid]")
    for seed in NEW_SEEDS:
        t0 = time.perf_counter()
        hres = run_hybrid(
            plaintext_fn=wdbc["plain"], fhe_fn=wdbc["fhe"],
            input_dim=wdbc["d"], input_bounds=wdbc["bounds"],
            threshold=0.0,
            oracle_budget=B_TENSEAL_VALIDATION, oracle_seed=seed, random_floor=0.3,
            data=wdbc["data"], empirical_budget=B_TENSEAL_VALIDATION,
            jitter_std=0.1, empirical_seed=seed + 100,
        )
        print(
            f"    seed={seed:2d} source={hres.source} "
            f"ora={hres.oracle_result.max_error:.3e} emp={hres.empirical_result.max_error:.3e} "
            f"t={time.perf_counter()-t0:.1f}s"
        )
        rows.append({
            "experiment": "wdbc_hybrid", "circuit": "wdbc_tenseal", "seed": seed,
            "oracle_max_error": hres.oracle_result.max_error,
            "random_max_error": 0.0,
            "ratio_oracle_over_random": 0.0,
            "oracle_wins_random": int(hres.source == "oracle"),
            "empirical_max_error": hres.empirical_result.max_error,
            "wall_clock_s": time.perf_counter() - t0,
        })

    elapsed = time.perf_counter() - t_total
    print(f"  wall-clock: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    _append_rows(
        os.path.join(RESULTS, "tenseal_validation.csv"),
        rows,
        [
            "experiment", "circuit", "seed",
            "oracle_max_error", "random_max_error",
            "ratio_oracle_over_random", "oracle_wins_random",
            "empirical_max_error", "wall_clock_s",
        ],
    )
    print(f"  appended {len(rows)} rows")


# --- Exp 1b-bis: tenseal_circuit2 seeds 10..19 -----------------------------

def exp_1b_circuit2() -> None:
    if not HAVE_TENSEAL:
        print("[1b-c2] TenSEAL not available, skipping")
        return
    print("\n" + "=" * 70)
    print(f"[1b-c2] tenseal_circuit2_validation seeds {NEW_SEEDS} @ B={B_TENSEAL_C2}")
    print("=" * 70)

    ctx = TenSEALContext()
    circuit = build_tenseal_circuit2(ctx)
    rows: list[dict] = []
    t_total = time.perf_counter()

    for setting, bounds, random_floor in [
        ("matched", C2_ADV, 0.0),
        ("asymmetric", C2_OP, 0.0),
        ("matched_warm", C2_ADV, 0.3),
    ]:
        print(f"  [{setting}]")
        for seed in NEW_SEEDS:
            t0 = time.perf_counter()
            ores = run_tenseal_oracle(
                circuit, seed, B_TENSEAL_C2, random_floor=random_floor,
            )
            rnd = run_tenseal_random(
                circuit["plain"], circuit["fhe"], bounds, B_TENSEAL_C2, seed,
            )
            ratio = ores.max_error / rnd if rnd > 0 else float("inf")
            wins = int(ores.max_error > rnd)
            wall = time.perf_counter() - t0
            print(
                f"    seed={seed:2d} ora={ores.max_error:.3e} rnd={rnd:.3e} "
                f"R={ratio:.2f}x t={wall:.1f}s"
            )
            rows.append({
                "setting": setting, "seed": seed,
                "oracle_max_error": ores.max_error,
                "random_max_error": rnd, "ratio": ratio,
                "oracle_wins": wins, "wall_clock_s": wall,
            })

    elapsed = time.perf_counter() - t_total
    print(f"  wall-clock: {elapsed:.1f}s")
    _append_rows(
        os.path.join(RESULTS, "tenseal_circuit2_validation.csv"),
        rows,
        [
            "setting", "seed",
            "oracle_max_error", "random_max_error",
            "ratio", "oracle_wins", "wall_clock_s",
        ],
    )
    print(f"  appended {len(rows)} rows")


# --- Exp 1e: external_sweep seeds 10..19 (WDBC + MNIST) --------------------

def exp_1e_external() -> None:
    print("\n" + "=" * 70)
    print(f"[1e] external_sweep WDBC+MNIST seeds {NEW_SEEDS} @ B={EXT_BUDGET}")
    print("=" * 70)

    # Patch SEEDS inside external_sweep for this call only.
    import external_sweep as ext
    original_seeds = ext.SEEDS
    ext.SEEDS = NEW_SEEDS
    try:
        all_rows: list[dict] = []
        plain_w, fhe_w, data_w, _, _, d_w = build_wdbc_circuit(random_state=42)
        print("  [WDBC d=30]")
        t0 = time.perf_counter()
        wdbc_rows = run_circuit_cells("wdbc_lr", plain_w, fhe_w, data_w, d_w)
        all_rows.extend(wdbc_rows)
        print(f"    {len(wdbc_rows)} rows, {time.perf_counter()-t0:.1f}s")

        plain_m, fhe_m, data_m, _, _, d_m = build_mnist_circuit(random_state=42)
        print("  [MNIST d=64]")
        t0 = time.perf_counter()
        mnist_rows = run_circuit_cells("mnist_d64", plain_m, fhe_m, data_m, d_m)
        all_rows.extend(mnist_rows)
        print(f"    {len(mnist_rows)} rows, {time.perf_counter()-t0:.1f}s")
    finally:
        ext.SEEDS = original_seeds

    _append_rows(
        os.path.join(RESULTS, "external_sweep.csv"),
        all_rows,
        [
            "config", "circuit", "d", "seed",
            "max_error", "verdict", "source",
            "oracle_max_error", "empirical_max_error", "wall_clock_s",
        ],
    )
    print(f"  appended {len(all_rows)} rows")


# --- Summary CSV ------------------------------------------------------------

def _p_wilcoxon(paired_a: list[float], paired_b: list[float]) -> float:
    """Signed-rank p-value for H0: a == b. Returns 1.0 if all diffs zero."""
    diffs = [a - b for a, b in zip(paired_a, paired_b)]
    if all(abs(d) < 1e-18 for d in diffs):
        return 1.0
    try:
        return float(wilcoxon(paired_a, paired_b).pvalue)
    except Exception:
        return float("nan")


def _summarise_paired(
    rows_oracle: list[float], rows_random: list[float],
) -> dict[str, float]:
    ratios = [
        (o / r) if r > 0 else float("nan")
        for o, r in zip(rows_oracle, rows_random)
    ]
    ratios_finite = [r for r in ratios if np.isfinite(r)]
    wins = sum(1 for o, r in zip(rows_oracle, rows_random) if o > r)
    return {
        "n": len(rows_oracle),
        "mean_oracle": float(np.mean(rows_oracle)) if rows_oracle else float("nan"),
        "mean_random": float(np.mean(rows_random)) if rows_random else float("nan"),
        "mean_ratio": float(np.mean(ratios_finite)) if ratios_finite else float("nan"),
        "median_ratio": float(np.median(ratios_finite)) if ratios_finite else float("nan"),
        "max_ratio": float(np.max(ratios_finite)) if ratios_finite else float("nan"),
        "wins": wins,
        "wins_pct": wins / len(rows_oracle) if rows_oracle else 0.0,
        "p_value": _p_wilcoxon(rows_oracle, rows_random),
    }


def build_summary() -> None:
    print("\n" + "=" * 70)
    print("[summary] n20_expansion_summary.csv")
    print("=" * 70)

    summary_rows: list[dict] = []

    # --- Table 1: mock FULL vs uniform-random ---
    abl = _read_rows(os.path.join(RESULTS, "ablation_heuristics.csv"))
    rnd = _read_rows(os.path.join(RESULTS, "table1_random_baselines_n20.csv"))
    rnd_map = {
        (r["circuit"], int(r["seed"])): float(r["random_max_error"])
        for r in rnd
    }
    for circuit_name in ("circuit1_lr", "circuit2_poly", "circuit3_cheb"):
        pairs_o, pairs_r = [], []
        for r in abl:
            if r["config"] != "FULL" or r["circuit"] != circuit_name:
                continue
            seed = int(r["seed"])
            o = float(r["max_error"])
            rand = rnd_map.get((circuit_name, seed))
            if rand is None:
                continue
            pairs_o.append(o)
            pairs_r.append(rand)
        s = _summarise_paired(pairs_o, pairs_r)
        summary_rows.append({
            "table": "Table 1 (Main mocks n=20)",
            "circuit": circuit_name,
            "setting": "FULL vs uniform-random B=500",
            **s,
        })

    # --- Table 5: tenseal_validation paired ---
    tv = _read_rows(os.path.join(RESULTS, "tenseal_validation.csv"))
    for label in ("lr_matched", "cheb_matched", "cheb_warm",
                  "wdbc_matched", "wdbc_asymmetric", "wdbc_hybrid"):
        pairs_o, pairs_r = [], []
        for r in tv:
            if r["experiment"] != label:
                continue
            o = float(r["oracle_max_error"])
            if label == "wdbc_asymmetric":
                rand = float(r["empirical_max_error"])
            elif label == "wdbc_hybrid":
                # skip Wilcoxon for hybrid (different metric)
                rand = float(r["empirical_max_error"])
            else:
                rand = float(r["random_max_error"])
            pairs_o.append(o)
            pairs_r.append(rand)
        s = _summarise_paired(pairs_o, pairs_r)
        summary_rows.append({
            "table": "Table 5 (TenSEAL real-CKKS)",
            "circuit": label, "setting": "B=60", **s,
        })

    # --- Circuit 2 real-CKKS ---
    c2 = _read_rows(os.path.join(RESULTS, "tenseal_circuit2_validation.csv"))
    for setting in ("matched", "asymmetric", "matched_warm"):
        pairs_o, pairs_r = [], []
        for r in c2:
            if r["setting"] != setting:
                continue
            pairs_o.append(float(r["oracle_max_error"]))
            pairs_r.append(float(r["random_max_error"]))
        s = _summarise_paired(pairs_o, pairs_r)
        summary_rows.append({
            "table": "Circuit 2 real-CKKS",
            "circuit": "circuit2_tenseal", "setting": setting, **s,
        })

    # --- External sweep (Table 6/7) ---
    ext = _read_rows(os.path.join(RESULTS, "external_sweep.csv"))
    for circ in ("wdbc_lr", "mnist_d64"):
        for cfg in ("oracle_only", "random_only", "empirical_only", "hybrid_union"):
            sub = [r for r in ext if r["circuit"] == circ and r["config"] == cfg]
            errs = [float(r["max_error"]) for r in sub]
            summary_rows.append({
                "table": f"External sweep ({circ})",
                "circuit": circ,
                "setting": cfg,
                "n": len(errs),
                "mean_oracle": float(np.mean(errs)) if errs else float("nan"),
                "mean_random": float("nan"),
                "mean_ratio": float("nan"),
                "median_ratio": float("nan"),
                "max_ratio": float("nan"),
                "wins": -1,
                "wins_pct": float("nan"),
                "p_value": float("nan"),
            })
        # paired oracle vs random for win-rate
        pairs_o, pairs_r = [], []
        for seed in ALL_SEEDS:
            o_rows = [r for r in ext
                      if r["circuit"] == circ and r["config"] == "oracle_only"
                      and int(r["seed"]) == seed]
            r_rows = [r for r in ext
                      if r["circuit"] == circ and r["config"] == "random_only"
                      and int(r["seed"]) == seed]
            if o_rows and r_rows:
                pairs_o.append(float(o_rows[0]["max_error"]))
                pairs_r.append(float(r_rows[0]["max_error"]))
        if pairs_o:
            s = _summarise_paired(pairs_o, pairs_r)
            summary_rows.append({
                "table": f"External sweep ({circ})",
                "circuit": circ,
                "setting": "oracle_only vs random_only paired",
                **s,
            })

    # Write
    out = os.path.join(RESULTS, "n20_expansion_summary.csv")
    fieldnames = [
        "table", "circuit", "setting", "n",
        "mean_oracle", "mean_random",
        "mean_ratio", "median_ratio", "max_ratio",
        "wins", "wins_pct", "p_value",
    ]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)
    print(f"  wrote {len(summary_rows)} rows to {out}")


# --- CLI --------------------------------------------------------------------

def main() -> int:
    steps = set(sys.argv[1:]) if len(sys.argv) > 1 else {
        "1a", "1a-rnd", "1b", "1b-c2", "1e", "summary",
    }
    t_all = time.perf_counter()
    if "1a" in steps:
        exp_1a_ablation()
    if "1a-rnd" in steps:
        exp_1a_random_baselines()
    if "1b" in steps:
        exp_1b_tenseal_validation()
    if "1b-c2" in steps:
        exp_1b_circuit2()
    if "1e" in steps:
        exp_1e_external()
    if "summary" in steps:
        build_summary()
    print(f"\nTOTAL wall-clock: {time.perf_counter() - t_all:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
