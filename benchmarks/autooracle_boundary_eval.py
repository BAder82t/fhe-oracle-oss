# Copyright (C) 2026 Bader Alissaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""AutoOracle boundary probe: pre-registered BEFORE/AFTER evaluation.

PRE-REGISTRATION (written 2026-09-15, before the first run of this script)

Question. Does the AutoOracle boundary probe stop AutoOracle losing to a
corner/boundary test set when the error supremum is at a box vertex, without
regressing interior worst cases and without exceeding the evaluation budget?

Circuits. Same domain and threshold (0.01) for every method; n_trials = 200.
  lr_d8     TenSEAL build_tenseal_lr_d8, [-3,3]^8. Supremum location: exact z-range.
  cheb_d10  TenSEAL build_tenseal_chebyshev_d10, [-3,3]^10. Per-unit exact h-range.
  poly_d6   TenSEAL build_tenseal_circuit2 (depth-4 polynomial), [-2,2]^6. Plaintext
            and FHE compute the same polynomial (pure CKKS noise): location unknown.
  bump_d8   mock, exp(-|x-c|^2/4.5), c = linspace(-1.2,1.2,8), [-3,3]^8. Interior.
  decoy_d6  mock, exp(-|x-0.7|^2/1.28) + 0.15*(mean(x)/2+1), [-2,2]^6. Interior.
  wdbc_d30 (TenSEAL) is supported but EXCLUDED on compute: a plaintext dry run showed
  BEFORE spending up to 6,225 evaluations per run (d >= 16 structure diagnostic), about
  6 min per run at 60.5 ms/eval, which alone exceeds the 30 min cap.
Seeds 1..10 (seed 0 excluded: pycma treats seed 0 as time-seeded). Seed 1 is run first
as the timing seed and kept. If the projected total exceeds 30 min, drop in this order:
the after_nb arm on TenSEAL circuits (poly_d6, cheb_d10, lr_d8), then poly_d6.

Methods.
  before    fhe_oracle/autoconfig.py at git BEFORE_SHA (loaded via git show), defaults.
  after     working-tree fhe_oracle/autoconfig.py, AutoOracle defaults.
  random    n_trials uniform draws, default_rng([seed, 2]).
  corner    pool = 2^d vertices + 2d face centres + centre, default_rng([seed, 1]) order,
            first n_trials (pool smaller than n_trials is used whole and reported).
  after_nb  SECONDARY, diagnostic only: after with _BOUNDARY_SHARE = 0 (budget accounting,
            no boundary probe). Not part of any criterion.

Metric. Per seed, max |plain(x) - fhe(x)| over every counted FHE call of the method
(AutoOracle: probes, boundary probe, search, final re-measurement). AutoOracle's own
reported max_error is also recorded.

Tie tolerance tau (criterion a): 2 x (max - min) of 10 repeated FHE evaluations at the
seed-1 corner witness (0 for deterministic mocks). Strict counts are also reported.

Criteria.
  (a) Circuits whose supremum is at a vertex: after >= corner - tau on >= 6 of 10 seeds.
  (b) Every circuit: median per-seed ratio after/before >= 0.95, and NOT (paired Wilcoxon
      two-sided, zero_method='wilcox', p < 0.05 with more losses than wins).
  (c) Every after run: FHE calls <= n_trials.
  PASS iff (a), (b) and (c) all hold. FAIL -> revert the library change and report.

Known before running (prototype on plaintext mocks, seeds 1..10): budget accounting
alone lost 7/10 paired seeds on bump_d8 (median ratio 0.994, p = 0.016), because BEFORE
spends 221 evaluations at n_trials = 200. (b) is therefore expected to fail on bump_d8
for any design that respects (c). The design was prototyped on plaintext Taylor-3 and
Chebyshev stand-ins, including stand-ins of lr_d8, cheb_d10 and wdbc_d30; both mocks
were written before any AFTER result was seen.

POST-HOC ANALYSIS (NOT pre-registered; added 2026-09-15 after the bump_d8 (b) failure
had been seen, at the reviewing coordinator's request)

Reason: BEFORE itself exceeds the budget (221 FHE calls at n_trials = 200), so (b)
mixes "stop overspending" with "boundary probe regresses".
Arms: ACCOUNTING-ONLY = the after_nb arm above (probe evaluations charged, boundary
probe off; checked identical to a standalone accounting-only autoconfig on both mocks)
vs AFTER, same seeds and budget. poly_d6 seeds 2..10 lacked after_nb (dropped for
compute as pre-registered); it is filled afterwards with --fill-nb in a separate process.
Criteria: P1 median AFTER/ACCOUNTING-ONLY >= 0.95 and NOT (paired two-sided Wilcoxon
p < 0.05 with more losses than wins); P2 = (a); P3 FHE calls <= n_trials on every seed
for both arms.

REVISION CHECK (post-hoc; written 2026-09-15 after the tables above and BEFORE running
arm after_v3)

Reason: review feedback changed AFTER after it was evaluated: a pre-search witness (best
landscape-probe or boundary evaluation) is re-measured through FHEOracle's measurement
path and merged under the core verdict rule (commit c0db686); one extra evaluation is
held back for that; the d >= 16 structure diagnostic is charged; vertices use exact bounds;
the probe is skipped for adapter/fitness/multi_output. Not adopted, from plaintext screens:
best vertex as CMA-ES x0 (no difference on 11 of 12 cases; poly stand-in 5/0/5, p = 0.0625)
and face centres (never helped on these circuits; best corner-set point was a vertex on
all 40 seeds of lr_d8, cheb_d10, poly_d6 and decoy_d6).
Arm: after_v3 = working-tree autoconfig.py, added to existing seeds 1..10 with --fill-v3
in a later process (core now includes c0db686; CKKS noise unseeded).
Criteria: R1 median after_v3/ACCOUNTING-ONLY >= 0.95 and NOT (Wilcoxon p < 0.05 with more
losses); R2 after_v3 >= corner - tau on a majority of seeds where the supremum is at a
vertex; R3 FHE calls <= n_trials; R4 reported max_error >= observed max and verdict ==
FAIL iff reported max_error >= threshold, every seed. after_v3/after (v1) is descriptive.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import importlib.util
import itertools
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone

import numpy as np
from scipy.optimize import minimize
from scipy.stats import wilcoxon

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, THIS_DIR)

import fhe_oracle.autoconfig as after_mod  # noqa: E402
from fhe_oracle.fitness import absolute_error  # noqa: E402

warnings.filterwarnings("ignore", message="Could not import matplotlib")

BEFORE_SHA = "6ec1b78eea66ff67ea80d602e281debcc7e8b4ca"
SCRIPT_REL = "benchmarks/autooracle_boundary_eval.py"
THRESHOLD = 0.01
TENSEAL = ("lr_d8", "cheb_d10", "poly_d6", "wdbc_d30")
MOCKS = ("bump_d8", "decoy_d6")
PREREGISTERED = ("lr_d8", "cheb_d10", "poly_d6", "bump_d8", "decoy_d6")


# --- Circuits ----------------------------------------------------------------

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))


def _taylor3(z):
    return 0.5 + z / 4.0 - z ** 3 / 48.0


def _cheb3(h):
    return 0.5 + 0.15 * h - h ** 3 / 500.0


def _affine_sup(w, b, bounds, surrogate):
    """Sup of |sigmoid - surrogate| over the exact range of w.x+b; (value, at_vertex)."""
    zmin = b + sum(min(wi * lo, wi * hi) for wi, (lo, hi) in zip(w, bounds))
    zmax = b + sum(max(wi * lo, wi * hi) for wi, (lo, hi) in zip(w, bounds))
    z = np.linspace(zmin, zmax, 1_000_001)
    err = np.abs(_sigmoid(z) - surrogate(z))
    i = int(np.argmax(err))
    return float(err[i]), i in (0, z.size - 1)


def build(name):
    if name in TENSEAL:
        import tenseal_circuits as tc
        from fhe_oracle.adapters.tenseal_adapter import TenSEALContext

        ctx = TenSEALContext()
        if name == "lr_d8":
            c = tc.build_tenseal_lr_d8(ctx)
            sup, vert = _affine_sup(c["weights"], c["bias"], c["bounds"], _taylor3)
        elif name == "wdbc_d30":
            c = tc.build_tenseal_wdbc(ctx)
            sup, vert = _affine_sup(c["weights"], c["bias"], c["bounds"], _taylor3)
        elif name == "cheb_d10":
            c = tc.build_tenseal_chebyshev_d10(ctx)
            units = [_affine_sup(c["weights"][j], float(c["bias"][j]), c["bounds"], _cheb3)
                     for j in range(len(c["bias"]))]
            sup, vert = max(units, key=lambda u: u[0])
        else:
            c = tc.build_tenseal_circuit2(ctx)
            sup, vert = None, None
        c["ctx"] = ctx
    elif name == "bump_d8":
        cen = np.linspace(-1.2, 1.2, 8)

        def fhe(x):
            return float(np.exp(-np.sum((np.asarray(x, dtype=float) - cen) ** 2) / 4.5))

        c = {"name": name, "plain": lambda x: 0.0, "fhe": fhe, "d": 8,
             "bounds": [(-3.0, 3.0)] * 8}
        sup, vert = 1.0, False
    elif name == "decoy_d6":
        cen = np.full(6, 0.7)

        def fhe(x):
            xa = np.asarray(x, dtype=float)
            return float(np.exp(-np.sum((xa - cen) ** 2) / 1.28) + 0.15 * (np.mean(xa) / 2 + 1))

        c = {"name": name, "plain": lambda x: 0.0, "fhe": fhe, "d": 6,
             "bounds": [(-2.0, 2.0)] * 6}
        opt = minimize(lambda x: -fhe(x), cen, bounds=c["bounds"])
        sup, vert = float(-opt.fun), False
    else:
        raise ValueError(f"unknown circuit {name!r}")
    c["sup"], c["at_vertex"] = sup, vert
    return c


class Instrumented:
    """Counts FHE calls and tracks the largest error; plaintext recomputed for logging."""

    def __init__(self, c):
        self._plain, self._fhe = c["plain"], c["fhe"]
        self.lo = np.array([lo for lo, _ in c["bounds"]])
        self.hi = np.array([hi for _, hi in c["bounds"]])
        self.reset()

    def reset(self):
        self.n, self.oob, self.t_fhe = 0, 0, 0.0
        self.max_err, self.witness = -np.inf, None

    def fhe(self, x):
        xa = np.asarray(x, dtype=np.float64)
        if np.any(xa < self.lo) or np.any(xa > self.hi):
            self.oob += 1
        t0 = time.perf_counter()
        y = self._fhe(x)
        self.t_fhe += time.perf_counter() - t0
        self.n += 1
        err = float(np.max(absolute_error(self._plain(x), y)))
        if err > self.max_err:
            self.max_err, self.witness = err, xa.tolist()
        return y


# --- Methods -----------------------------------------------------------------

def load_before():
    src = subprocess.run(["git", "show", f"{BEFORE_SHA}:fhe_oracle/autoconfig.py"],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout
    name = "fhe_oracle._autoconfig_before"
    spec = importlib.util.spec_from_loader(name, loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "fhe_oracle"
    sys.modules[name] = mod
    exec(compile(src, f"{BEFORE_SHA}:fhe_oracle/autoconfig.py", "exec"), mod.__dict__)
    return mod, hashlib.sha256(src.encode()).hexdigest()


def run_auto(mod, inst, c, seed, budget, share=None):
    inst.reset()
    old = getattr(mod, "_BOUNDARY_SHARE", None)
    if share is not None:
        mod._BOUNDARY_SHARE = share
    try:
        t0 = time.perf_counter()
        ao = mod.AutoOracle(c["plain"], inst.fhe, c["bounds"])
        res = ao.run(n_trials=budget, seed=seed, threshold=THRESHOLD)
        wall = time.perf_counter() - t0
    finally:
        if share is not None:
            mod._BOUNDARY_SHARE = old
    return {
        "max_error": inst.max_err, "witness": inst.witness, "fhe_calls": inst.n,
        "oob": inst.oob, "reported_max_error": float(res.max_error),
        "reported_witness": [float(v) for v in res.worst_input], "verdict": res.verdict,
        "regime": res.regime, "strategy": res.strategy_used,
        "n_trials_reported": int(res.n_trials),
        "search_max_error": getattr(res, "search_max_error", None),
        "remeasured_error": getattr(res, "remeasured_error", None),
        "probe_n_evals": getattr(ao.probe_result, "n_evals", None),
        "wall_s": wall, "fhe_s": inst.t_fhe,
    }


def run_points(inst, pts):
    inst.reset()
    t0 = time.perf_counter()
    for x in pts:
        inst.fhe(x.tolist())
    return {"max_error": inst.max_err, "witness": inst.witness, "fhe_calls": inst.n,
            "oob": inst.oob, "wall_s": time.perf_counter() - t0, "fhe_s": inst.t_fhe}


def random_points(c, seed, budget):
    lo = np.array([a for a, _ in c["bounds"]])
    hi = np.array([b for _, b in c["bounds"]])
    return np.random.default_rng([seed, 2]).uniform(lo, hi, size=(budget, c["d"]))


def corner_points(c, seed, budget):
    d = c["d"]
    lo = np.array([a for a, _ in c["bounds"]])
    hi = np.array([b for _, b in c["bounds"]])
    mid = (lo + hi) / 2.0
    rng = np.random.default_rng([seed, 1])
    extra = [mid.copy()]
    for i in range(d):
        for v in (lo[i], hi[i]):
            p = mid.copy()
            p[i] = v
            extra.append(p)
    if d <= 12:
        verts = [np.array(v) for v in itertools.product(*zip(lo, hi))]
    else:
        seen, verts = set(), []
        while len(verts) < budget:
            bits = rng.integers(0, 2, d)
            if bits.tobytes() not in seen:
                seen.add(bits.tobytes())
                verts.append(np.where(bits == 1, hi, lo))
    pool = np.array(verts + extra)
    return pool[rng.permutation(len(pool))][:budget]


# --- Run ---------------------------------------------------------------------

def _sha256_file(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return None


def _write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=1)
    os.replace(tmp, path)


DEPENDENCIES = ("core", "fitness", "diagnostics", "preactivation", "seeds", "registry",
                "adaptive", "diversity", "guarantees", "multi_output")


def run_circuit(name, seeds, budget, out_dir, diagnostic, fill_nb=False,
                allow_code_change=False, fill_v3=False):
    path = os.path.join(out_dir, f"{name}.json")
    if os.path.exists(path):
        with open(path) as fh:
            data = json.load(fh)
        if data["budget"] != budget:
            raise RuntimeError(f"{path} was run with budget {data['budget']}")
    else:
        data = {"circuit": name, "budget": budget, "threshold": THRESHOLD, "seeds": {},
                "runs": []}
    c = build(name)
    inst = Instrumented(c)
    before_mod, before_sha256 = load_before()
    data.update({"sup": c["sup"], "at_vertex": c["at_vertex"], "d": c["d"],
                 "bounds": c["bounds"]})
    after_sha = _sha256_file(after_mod.__file__)
    prev = {r["after_autoconfig_sha256"] for r in data["runs"]}
    if prev and after_sha not in prev and not allow_code_change:
        raise RuntimeError(f"{path}: autoconfig.py changed since earlier runs; "
                           "pass --allow-code-change to record a mixed-code result")
    pkg = os.path.join(ROOT, "fhe_oracle")
    data["runs"].append({
        "dependency_sha256": {m: _sha256_file(os.path.join(pkg, f"{m}.py")) for m in DEPENDENCIES},
        "dependencies_changed_since_before_sha": (_git(
            "diff", "--name-only", BEFORE_SHA, "--",
            *[f"fhe_oracle/{m}.py" for m in DEPENDENCIES]) or "").splitlines(),
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_head": _git("rev-parse", "HEAD"), "before_sha": BEFORE_SHA,
        "before_autoconfig_sha256": before_sha256,
        "after_autoconfig_sha256": after_sha,
        "script_sha256": _sha256_file(os.path.join(ROOT, SCRIPT_REL)),
        "diagnostic_arm": diagnostic, "fill_nb": fill_nb, "fill_v3": fill_v3,
        "seeds_requested": list(seeds),
        "packages": {p: _safe_ver(p) for p in ("tenseal", "cma", "numpy", "scipy")},
        "python": sys.version.split()[0], "platform": platform.platform(),
        "loadavg_start": list(os.getloadavg()),
    })
    for seed in seeds:
        row = data["seeds"].get(str(seed))
        if row is not None:
            if fill_nb and "after_nb" not in row:
                t0 = time.perf_counter()
                row["after_nb"] = run_auto(after_mod, inst, c, seed, budget, share=0.0)
                row["after_nb"]["filled_later"] = True
                row["after_nb"]["run_index"] = len(data["runs"]) - 1
                row["fill_wall_s"] = time.perf_counter() - t0
                _write_json(path, data)
                print(f"{name} seed {seed}: filled after_nb={row['after_nb']['max_error']:.6g}"
                      f"[{row['after_nb']['fhe_calls']}] ({row['fill_wall_s']:.1f}s)", flush=True)
            if fill_v3 and "after_v3" not in row:
                t0 = time.perf_counter()
                row["after_v3"] = run_auto(after_mod, inst, c, seed, budget)
                row["after_v3"]["run_index"] = len(data["runs"]) - 1
                row["fill_v3_wall_s"] = time.perf_counter() - t0
                _write_json(path, data)
                print(f"{name} seed {seed}: after_v3={row['after_v3']['max_error']:.6g}"
                      f"[{row['after_v3']['fhe_calls']}] reported="
                      f"{row['after_v3']['reported_max_error']:.6g} {row['after_v3']['verdict']} "
                      f"({row['fill_v3_wall_s']:.1f}s)", flush=True)
            continue
        t0 = time.perf_counter()
        row = {"before": run_auto(before_mod, inst, c, seed, budget),
               "after": run_auto(after_mod, inst, c, seed, budget)}
        if diagnostic:
            row["after_nb"] = run_auto(after_mod, inst, c, seed, budget, share=0.0)
        row["random"] = run_points(inst, random_points(c, seed, budget))
        row["corner"] = run_points(inst, corner_points(c, seed, budget))
        row["seed_wall_s"] = time.perf_counter() - t0
        row["run_index"] = len(data["runs"]) - 1
        data["seeds"][str(seed)] = row
        if "tau" not in data:
            w = row["corner"]["witness"]
            reps = [float(np.max(absolute_error(c["plain"](w), c["fhe"](w))))
                    for _ in range(10)]
            data["tau"] = 2.0 * (max(reps) - min(reps))
            data["tau_repeats"] = reps
        data["runs"][-1]["ended_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        data["runs"][-1]["loadavg_end"] = list(os.getloadavg())
        _write_json(path, data)
        print(f"{name} seed {seed}: " + " ".join(
            f"{m}={row[m]['max_error']:.6g}[{row[m]['fhe_calls']}]"
            for m in ("before", "after", "after_nb", "random", "corner") if m in row)
            + f" ({row['seed_wall_s']:.1f}s)", flush=True)
    return data


def _safe_ver(pkg):
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


# --- Report ------------------------------------------------------------------

def paired(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ratios = np.where(y > 0, x / np.where(y > 0, y, 1.0), np.where(x > 0, np.inf, 1.0))
    wins, losses = int(np.sum(x > y)), int(np.sum(x < y))
    p = (float(wilcoxon(x, y, zero_method="wilcox", alternative="two-sided").pvalue)
         if np.any(x != y) else 1.0)
    return {"median_ratio": float(np.median(ratios)), "min_ratio": float(np.min(ratios)),
            "max_ratio": float(np.max(ratios)), "wins": wins, "losses": losses,
            "ties": int(x.size - wins - losses), "p": p, "n": int(x.size)}


def summarize(data):
    seeds = sorted(int(s) for s in data["seeds"])
    rows = [data["seeds"][str(s)] for s in seeds]
    budget = data["budget"]

    def col(m, k="max_error"):
        return np.array([r[m][k] for r in rows if m in r], dtype=float)

    after, before, corner, rand = col("after"), col("before"), col("corner"), col("random")
    s = {"circuit": data["circuit"], "seeds": seeds, "n": len(seeds), "sup": data["sup"],
         "at_vertex": data["at_vertex"], "tau": data.get("tau"), "budget": budget}
    s["b"] = paired(after, before)
    s["b_pass"] = bool(s["b"]["median_ratio"] >= 0.95
                       and not (s["b"]["p"] < 0.05 and s["b"]["losses"] > s["b"]["wins"]))
    s["after_calls_max"] = int(col("after", "fhe_calls").max())
    s["before_calls"] = [int(v) for v in col("before", "fhe_calls")]
    s["c_pass"] = s["after_calls_max"] <= budget
    tau = data.get("tau") or 0.0
    s["a_applies"] = bool(data["at_vertex"])
    s["a_not_smaller_tau"] = int(np.sum(after >= corner - tau))
    s["a_not_smaller_strict"] = int(np.sum(after >= corner))
    s["a_pass"] = (s["a_not_smaller_tau"] >= len(seeds) // 2 + 1) if s["a_applies"] else None
    s["after_vs_corner"] = paired(after, corner)
    s["after_vs_random"] = paired(after, rand)
    s["before_vs_corner"] = paired(before, corner)
    s["calls"] = {m: [int(col(m, "fhe_calls").min()), int(col(m, "fhe_calls").max())]
                  for m in ("before", "after", "after_nb", "after_v3", "random", "corner")
                  if all(m in r for r in rows)}
    s["nb_complete"] = all("after_nb" in r for r in rows)
    s["nb_filled_later"] = [seed for seed, r in zip(seeds, rows)
                            if r.get("after_nb", {}).get("filled_later")]
    if s["nb_complete"]:
        nb = col("after_nb")
        s["after_nb_vs_before"] = paired(nb, before)
        s["after_vs_after_nb"] = paired(after, nb)
        ph = s["after_vs_after_nb"]
        s["p1_pass"] = bool(ph["median_ratio"] >= 0.95
                            and not (ph["p"] < 0.05 and ph["losses"] > ph["wins"]))
        s["p3_pass"] = bool(s["calls"]["after"][1] <= budget
                            and s["calls"]["after_nb"][1] <= budget)
        s["median_max_error_nb"] = float(np.median(nb))
    s["v3_complete"] = bool(rows) and all("after_v3" in r for r in rows)
    if s["v3_complete"] and s["nb_complete"]:
        v3 = col("after_v3")
        s["v3_vs_acct"] = paired(v3, col("after_nb"))
        s["v3_vs_after"] = paired(v3, after)
        s["v3_vs_corner"] = paired(v3, corner)
        r1 = s["v3_vs_acct"]
        s["r1_pass"] = bool(r1["median_ratio"] >= 0.95
                            and not (r1["p"] < 0.05 and r1["losses"] > r1["wins"]))
        s["r2_not_smaller_tau"] = int(np.sum(v3 >= corner - tau))
        s["r2_pass"] = ((s["r2_not_smaller_tau"] >= len(seeds) // 2 + 1)
                        if s["a_applies"] else None)
        s["r3_pass"] = bool(s["calls"]["after_v3"][1] <= budget)
        rep = col("after_v3", "reported_max_error")
        verdict_ok = [(r["after_v3"]["verdict"] == "FAIL") == (r["after_v3"]["reported_max_error"]
                                                               >= THRESHOLD) for r in rows]
        s["r4_reported_ge_observed"] = int(np.sum(rep >= v3 * (1 - 1e-12)))
        s["r4_verdict_consistent"] = int(sum(verdict_ok))
        s["r4_pass"] = bool(s["r4_reported_ge_observed"] == len(seeds)
                            and s["r4_verdict_consistent"] == len(seeds))
        s["median_max_error_v3"] = float(np.median(v3))
        s["v3_fail_count"] = int(sum(r["after_v3"]["verdict"] == "FAIL" for r in rows))
    med = {m: float(np.median(col(m))) for m in ("before", "after", "random", "corner")}
    s["median_max_error"] = med
    s["median_reported"] = {m: float(np.median(col(m, "reported_max_error")))
                            for m in ("before", "after")}
    s["reported_vs_observed_max_gap_after"] = float(
        np.max(col("after") - col("after", "reported_max_error")))
    if data["sup"]:
        s["median_frac_sup"] = {m: v / data["sup"] for m, v in med.items()}
    s["corner_calls"] = sorted({int(v) for v in col("corner", "fhe_calls")})
    s["oob_total"] = int(sum(r[m]["oob"] for r in rows for m in r if isinstance(r[m], dict)))
    s["regimes"] = {m: sorted({r[m]["strategy"] for r in rows}) for m in ("before", "after")}
    s["compute_s"] = float(sum(r["seed_wall_s"] + r.get("fill_wall_s", 0.0)
                               + r.get("fill_v3_wall_s", 0.0) for r in rows))
    return s


def _fmt(v):
    return f"{v:.6g}"


def build_report(summaries, datas):
    L = ["# AutoOracle boundary probe: pre-registered evaluation", "",
         f"Generated by `{SCRIPT_REL}`. BEFORE = `fhe_oracle/autoconfig.py` at `{BEFORE_SHA}`; "
         "AFTER = working tree. Pre-registration: the script's module docstring.", ""]
    L += ["## Criteria", "",
          "| Circuit | Sup at vertex | (a) after >= corner - tau | (b) median after/before | "
          "(b) W/L/T, p | (b) | (c) max after calls | overall |",
          "|---|---|---|---:|---|---|---:|---|"]
    all_pass = True
    for s in summaries:
        a = ("n/a" if not s["a_applies"] else
             f"{s['a_not_smaller_tau']}/{s['n']} (strict {s['a_not_smaller_strict']}) "
             f"{'PASS' if s['a_pass'] else 'FAIL'}")
        ok = s["b_pass"] and s["c_pass"] and s["a_pass"] is not False
        all_pass &= ok
        b = s["b"]
        L.append(f"| {s['circuit']} | {s['at_vertex']} | {a} | {b['median_ratio']:.4f} | "
                 f"{b['wins']}/{b['losses']}/{b['ties']}, p={b['p']:.3g} | "
                 f"{'PASS' if s['b_pass'] else 'FAIL'} | {s['after_calls_max']} of "
                 f"{s['budget']} {'PASS' if s['c_pass'] else 'FAIL'} | "
                 f"{'PASS' if ok else 'FAIL'} |")
    L += ["", f"**Overall: {'PASS' if all_pass else 'FAIL'}**", ""]
    L += ["## POST-HOC (not pre-registered): AFTER vs ACCOUNTING-ONLY, budget-matched", "",
          "ACCOUNTING-ONLY = `after_nb` arm (probe evaluations charged, boundary probe off).", "",
          "| Circuit | P1 median after/acct (min-max) | P1 W/L/T, p | P1 | P2 after >= corner - tau "
          "| P3 max calls after / acct | overall |", "|---|---|---|---|---|---|---|"]
    ph_all = True
    for s in summaries:
        if not s["nb_complete"]:
            L.append(f"| {s['circuit']} | ACCOUNTING-ONLY arm incomplete | | | | | not evaluated |")
            ph_all = False
            continue
        ph = s["after_vs_after_nb"]
        p2 = ("n/a" if not s["a_applies"] else
              f"{s['a_not_smaller_tau']}/{s['n']} {'PASS' if s['a_pass'] else 'FAIL'}")
        ok = s["p1_pass"] and s["p3_pass"] and s["a_pass"] is not False
        ph_all &= ok
        filled = f" (acct seeds {s['nb_filled_later']} run later)" if s["nb_filled_later"] else ""
        L.append(f"| {s['circuit']}{filled} | {ph['median_ratio']:.4f} ({ph['min_ratio']:.4f}-"
                 f"{ph['max_ratio']:.4f}) | {ph['wins']}/{ph['losses']}/{ph['ties']}, "
                 f"p={ph['p']:.3g} | {'PASS' if s['p1_pass'] else 'FAIL'} | {p2} | "
                 f"{s['calls']['after'][1]} / {s['calls']['after_nb'][1]} of {s['budget']} "
                 f"{'PASS' if s['p3_pass'] else 'FAIL'} | {'PASS' if ok else 'FAIL'} |")
    L += ["", f"**Post-hoc overall: {'PASS' if ph_all else 'FAIL'}**", ""]
    L += ["## REVISION CHECK (post-hoc): AFTER v3 vs ACCOUNTING-ONLY", "",
          "| Circuit | R1 median v3/acct (min-max) | R1 W/L/T, p | R1 | R2 v3 >= corner - tau | "
          "R3 max calls | R4 reported>=observed, verdict consistent | overall | "
          "v3/after(v1) median, W/L/T | FAIL verdicts |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    rv_all = True
    for s in summaries:
        if not s.get("v3_complete") or "r1_pass" not in s:
            L.append(f"| {s['circuit']} | after_v3 arm incomplete | | | | | | not evaluated | | |")
            rv_all = False
            continue
        r1, va = s["v3_vs_acct"], s["v3_vs_after"]
        r2 = ("n/a" if not s["a_applies"] else
              f"{s['r2_not_smaller_tau']}/{s['n']} {'PASS' if s['r2_pass'] else 'FAIL'}")
        ok = s["r1_pass"] and s["r3_pass"] and s["r4_pass"] and s["r2_pass"] is not False
        rv_all &= ok
        L.append(f"| {s['circuit']} | {r1['median_ratio']:.4f} ({r1['min_ratio']:.4f}-"
                 f"{r1['max_ratio']:.4f}) | {r1['wins']}/{r1['losses']}/{r1['ties']}, "
                 f"p={r1['p']:.3g} | {'PASS' if s['r1_pass'] else 'FAIL'} | {r2} | "
                 f"{s['calls']['after_v3'][1]} of {s['budget']} "
                 f"{'PASS' if s['r3_pass'] else 'FAIL'} | {s['r4_reported_ge_observed']}/{s['n']}, "
                 f"{s['r4_verdict_consistent']}/{s['n']} {'PASS' if s['r4_pass'] else 'FAIL'} | "
                 f"{'PASS' if ok else 'FAIL'} | {va['median_ratio']:.4f}, "
                 f"{va['wins']}/{va['losses']}/{va['ties']} | {s['v3_fail_count']}/{s['n']} |")
    L += ["", f"**Revision check overall: {'PASS' if rv_all else 'FAIL'}**", ""]
    L += ["## Counted FHE calls per arm (min-max over seeds)", "",
          "| Circuit | budget | before | after | accounting-only | after_v3 | random | corner |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in summaries:
        cells = [(f"{s['calls'][m][0]}-{s['calls'][m][1]}" if m in s["calls"] else "-")
                 for m in ("before", "after", "after_nb", "after_v3", "random", "corner")]
        L.append(f"| {s['circuit']} | {s['budget']} | " + " | ".join(cells) + " |")
    L += [""]
    L += ["## Median per-seed max error", "",
          "| Circuit | Supremum | before | after | accounting-only | after_v3 | random | corner | "
          "after reported | tau | corner evals |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in summaries:
        m = s["median_max_error"]
        nb_med = _fmt(s["median_max_error_nb"]) if "median_max_error_nb" in s else "-"
        v3_med = _fmt(s["median_max_error_v3"]) if "median_max_error_v3" in s else "-"
        L.append(f"| {s['circuit']} | {_fmt(s['sup']) if s['sup'] else 'unknown'} | "
                 f"{_fmt(m['before'])} | {_fmt(m['after'])} | {nb_med} | {v3_med} | "
                 f"{_fmt(m['random'])} | "
                 f"{_fmt(m['corner'])} | {_fmt(s['median_reported']['after'])} | "
                 f"{_fmt(s['tau'] or 0.0)} | {s['corner_calls']} |")
    L += ["", "## Paired comparisons (ratio = first / second, per seed)", "",
          "| Circuit | Comparison | median ratio (min-max) | W/L/T | Wilcoxon p |",
          "|---|---|---|---|---:|"]
    for s in summaries:
        for key, label in [("b", "after / before"), ("after_vs_corner", "after / corner"),
                           ("after_vs_random", "after / random"),
                           ("before_vs_corner", "before / corner"),
                           ("after_nb_vs_before", "accounting-only / before (secondary)"),
                           ("after_vs_after_nb", "after / accounting-only (post-hoc P1)"),
                           ("v3_vs_acct", "after_v3 / accounting-only (revision R1)"),
                           ("v3_vs_after", "after_v3 / after (descriptive)"),
                           ("v3_vs_corner", "after_v3 / corner")]:
            if key in s:
                c = s[key]
                L.append(f"| {s['circuit']} | {label} | {c['median_ratio']:.4f} "
                         f"({c['min_ratio']:.4f}-{c['max_ratio']:.4f}) | "
                         f"{c['wins']}/{c['losses']}/{c['ties']} | {c['p']:.3g} |")
    L += ["", "## Provenance (one row per script invocation)", "",
          "| Circuit | run | started (UTC) | after autoconfig sha256 | fill_nb | "
          "deps changed since BEFORE_SHA |", "|---|---:|---|---|---|---|"]
    for data in datas:
        for i, r in enumerate(data["runs"]):
            deps = r.get("dependencies_changed_since_before_sha")
            L.append(f"| {data['circuit']} | {i} | {r['started_utc']} | "
                     f"`{r['after_autoconfig_sha256'][:16]}` | {r.get('fill_nb', False)} | "
                     f"{'not recorded' if deps is None else (', '.join(deps) or 'none')} |")
    L += ["", "## Per-seed values", ""]
    for s, data in zip(summaries, datas):
        L += [f"### {s['circuit']}", "",
              "| Seed | before [calls] | after [calls] | accounting-only [calls] | "
              "after_v3 [calls] reported, verdict | random | corner | after strategy |",
              "|---:|---:|---:|---:|---:|---:|---:|---|"]
        for seed in s["seeds"]:
            r = data["seeds"][str(seed)]
            nb = (f"{_fmt(r['after_nb']['max_error'])} [{r['after_nb']['fhe_calls']}]"
                  if "after_nb" in r else "-")
            v3 = (f"{_fmt(r['after_v3']['max_error'])} [{r['after_v3']['fhe_calls']}] "
                  f"{_fmt(r['after_v3']['reported_max_error'])}, {r['after_v3']['verdict']}"
                  if "after_v3" in r else "-")
            L.append(f"| {seed} | {_fmt(r['before']['max_error'])} [{r['before']['fhe_calls']}] | "
                     f"{_fmt(r['after']['max_error'])} [{r['after']['fhe_calls']}] | {nb} | {v3} | "
                     f"{_fmt(r['random']['max_error'])} | {_fmt(r['corner']['max_error'])} | "
                     f"{r['after']['strategy']} |")
        L += ["", f"Strategies: before {s['regimes']['before']}, after {s['regimes']['after']}. "
              f"Out-of-bounds evaluations: {s['oob_total']}. Compute: {s['compute_s']:.0f} s.", ""]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description="AutoOracle boundary probe evaluation")
    ap.add_argument("--circuits", default=",".join(PREREGISTERED))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seed-start", type=int, default=1)
    ap.add_argument("--budget", type=int, default=200)
    ap.add_argument("--no-diagnostic", action="store_true", help="skip the after_nb arm")
    ap.add_argument("--fill-nb", action="store_true",
                    help="add the after_nb arm to seeds already recorded without it")
    ap.add_argument("--fill-v3", action="store_true",
                    help="add the after_v3 revision arm to seeds already recorded")
    ap.add_argument("--allow-code-change", action="store_true",
                    help="allow adding results after autoconfig.py changed (recorded per run)")
    ap.add_argument("--report", action="store_true", help="summarise existing JSON only")
    ap.add_argument("--out", default=os.path.join(THIS_DIR, "results", "autooracle_boundary"))
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    names = [n for n in args.circuits.split(",") if n]
    if not args.report:
        seeds = range(args.seed_start, args.seed_start + args.seeds)
        for name in names:
            run_circuit(name, seeds, args.budget, args.out, not args.no_diagnostic,
                        fill_nb=args.fill_nb, allow_code_change=args.allow_code_change,
                        fill_v3=args.fill_v3)
    datas = []
    for name in names:
        path = os.path.join(args.out, f"{name}.json")
        if os.path.exists(path):
            with open(path) as fh:
                datas.append(json.load(fh))
    summaries = [summarize(d) for d in datas]
    report = build_report(summaries, datas)
    with open(os.path.join(args.out, "report.md"), "w") as fh:
        fh.write(report)
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summaries, fh, indent=1)
    print(report.split("## Median")[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
