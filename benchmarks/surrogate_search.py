# Copyright (C) 2026 Bader Alissaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pre-registered study B: plaintext-surrogate search, then CKKS verification.

Protocol: benchmarks/preregistration_2026-09-15.md, section B (committed at 962aab3).
Writes per_seed.csv, summary.json and report.md into --out.

Choices where the pre-registration is not explicit, fixed before the run:
- Circuit: tenseal_circuits.build_tenseal_lr_d8, domain [-3, 3]^8, default TenSEALContext
  (N=16384, [60,40,40,40,40,60], scale 2^40), one context and key set for the whole run.
  Error at x: |sigmoid(w.x + b) - CKKS output|, fresh encryption per call.
- Wall budget per seed ("median wall time of AutoOracle at n_trials = 200 on that seed"):
  AutoOracle(sigmoid, CKKS T3, bounds).run(n_trials=200, seed=s, threshold=0.01) is run 3
  times on seed s; the budget is the median of the three wall times (construction + run).
  The AutoOracle arm is the first of the three runs; runs 2-3 are timing replicates.
- SURROGATE, plaintext phase: FHEOracle(plaintext_fn=sigmoid, fhe_fn=T3 in float64, input_dim=8,
  input_bounds, seed=s).run(n_trials=5000, threshold=0.01) with library defaults (pycma CMA-ES,
  sigma0 = 1.0, x0 = box centre, default population 10, pycma bound handling, tolx 1e-12,
  tolfun 1e-15, no restarts); pycma may stop before 5000. It is interrupted if the elapsed time
  passes budget - reserve, reserve = min(1.5 * K * median CKKS time of that seed's first
  AutoOracle run, 0.5 * budget), so the plaintext phase always gets at least half the budget.
- SURROGATE, candidates: every plaintext-evaluated point, sorted by |sigmoid - T3| (descending,
  stable). Greedy pick: a point is distinct if its L-infinity distance to every picked point
  exceeds 0.01 * box width (0.06). K = 20, or all distinct points if fewer.
- SURROGATE, CKKS phase: candidates in that order, at most K. An evaluation starts only while
  elapsed < budget and counts only if it completes within the budget. Leftover time is unused
  (reported); it never buys CKKS evaluations beyond K.
- Corner set: all 2^8 vertices, 16 face centres and the centre (273 points) in the order
  default_rng([s, 1]).permutation. Same start/count rule; pool exhaustion is reported.
- Per-seed metric: max error over the arm's counted CKKS evaluations (AutoOracle: every CKKS call
  it makes, including re-measurements).
- Counting per arm: sigmoid calls, T3 plaintext calls, CKKS calls started and counted.
- Pass (both): C1 SURROGATE >= corner - 1e-5 on >= 18 of 20 seeds; C2 median SURROGATE >=
  0.99 * 140.9204. The supremum is recomputed from the exact z-range; a mismatch > 5e-5 aborts.
- Statistics: paired two-sided Wilcoxon (zero_method='wilcox'), SURROGATE vs corner and SURROGATE
  vs AutoOracle, Holm across these two.
- Uncounted references: each arm's witness re-encrypted 5 times; the z-minimising and
  z-maximising vertices re-encrypted 20 times.
- Order per seed: AutoOracle x3, SURROGATE, corner, re-encryptions. One process.
- Compute cap (--max-minutes, default 30): projected from 3 uncounted timing evaluations before
  any seed, re-checked after each seed; overrun aborts.
- Provenance: git HEAD and SHA-256 of fhe_oracle/core.py, fhe_oracle/autoconfig.py and this script,
  checked after every seed and at the end; the pre-registration must hash as at 962aab3.
- Exit codes: 0 complete, 2 TenSEAL not installed (nothing written), 3 provenance changed,
  4 compute cap, 5 evaluation error, 6 report rendering failed, 7 reference mismatch
  (pre-registration hash or supremum). CSV and JSON are written for codes 0 and 3-7.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata as md
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
from scipy.stats import rankdata, wilcoxon

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, THIS_DIR)
warnings.filterwarnings("ignore", message="Could not import matplotlib")

from fhe_oracle import FHEOracle  # noqa: E402
from fhe_oracle import __version__ as FHE_ORACLE_VERSION  # noqa: E402
from fhe_oracle.adapters import tenseal_adapter as tsa  # noqa: E402
from fhe_oracle.autoconfig import AutoOracle  # noqa: E402
from fhe_oracle.fitness import EvaluationError, absolute_error  # noqa: E402

SCRIPT_REL = "benchmarks/surrogate_search.py"
CORE_REL = "fhe_oracle/core.py"
AUTO_REL = "fhe_oracle/autoconfig.py"
PREREG_REL = "benchmarks/preregistration_2026-09-15.md"
PREREG_COMMIT = "962aab31e4239d074130a8f80394dac3e2731c1d"
PREREG_SEEDS = list(range(11, 31))
PREREG_N_TRIALS = 200
PREREG_K = 20
SUPREMUM = 140.9204
SUP_TOL = 5e-5
CORNER_TOL = 1e-5
PASS_SEEDS = 18
MEDIAN_FRAC = 0.99
THRESHOLD = 0.01
RESERVE_FACTOR = 1.5
RESERVE_MAX_SHARE = 0.5  # plaintext phase keeps at least half the budget
DEFAULTS = {"ao_repeats": 3, "plain_budget": 5000, "distinct_frac": 0.01}
ARMS = ["surrogate", "autooracle", "corner"]
LABELS = {"surrogate": "SURROGATE", "autooracle": "AutoOracle", "corner": "Corner/boundary set"}


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def t3(z):
    # Same operation order as the CKKS program.
    return z * 0.25 - z ** 3 * (1.0 / 48.0) + 0.5


def build_circuit():
    from tenseal_circuits import build_tenseal_lr_d8
    ctx = tsa.TenSEALContext()
    c = build_tenseal_lr_d8(ctx)
    return {"name": "lr_d8", "source": "benchmarks/tenseal_circuits.py::build_tenseal_lr_d8",
            "d": int(c["d"]), "bounds": [(float(a), float(b)) for a, b in c["bounds"]],
            "w": np.asarray(c["weights"], dtype=np.float64), "b": float(c["bias"]),
            "plain": c["plain"], "fhe": c["fhe"],
            "ckks": {"poly_modulus_degree": tsa.CKKS_POLY_MODULUS_DEGREE,
                     "coeff_mod_bit_sizes": list(tsa.CKKS_COEFF_MOD_BIT_SIZES),
                     "scale_bits": int(round(np.log2(tsa.CKKS_GLOBAL_SCALE)))},
            "ctx": ctx}


BUILD = {"circuit": build_circuit}  # replaceable in tests


class _Deadline(Exception):
    """Raised inside the plaintext search when its time share is spent."""


def _is_deadline(exc):
    seen = set()
    while exc is not None and id(exc) not in seen:
        if isinstance(exc, _Deadline):
            return True
        seen.add(id(exc))
        exc = exc.__cause__ or exc.__context__
    return False


class Counter:
    """Counts sigmoid, T3-plaintext and CKKS calls; logs every T3 score and CKKS evaluation."""

    def __init__(self, circ):
        self.w, self.b = circ["w"], circ["b"]
        self._plain, self._fhe = circ["plain"], circ["fhe"]
        self.lo = np.array([a for a, _ in circ["bounds"]])
        self.hi = np.array([c for _, c in circ["bounds"]])
        self.deadline = None
        self.reset()

    def reset(self):
        self.n_sigma = 0
        self.n_t3 = 0
        self.n_fhe = 0
        self.n_oob = 0
        self.t3_records: list[dict] = []
        self.fhe_records: list[dict] = []

    def _check(self, xa):
        if np.any(xa < self.lo) or np.any(xa > self.hi):
            self.n_oob += 1

    def sigma(self, x):
        self.n_sigma += 1
        return self._plain(x)

    def t3_plain(self, x):
        if self.deadline is not None and time.perf_counter() > self.deadline:
            raise _Deadline()
        xa = np.asarray(x, dtype=np.float64)
        self._check(xa)
        self.n_t3 += 1
        z = float(np.dot(self.w, xa) + self.b)
        s = t3(z)
        # Instrumentation: sigmoid recomputed for the candidate score, not counted.
        self.t3_records.append({"x": xa.tolist(), "z": z, "score": abs(float(self._plain(x)) - s)})
        return s

    def fhe(self, x):
        xa = np.asarray(x, dtype=np.float64)
        self._check(xa)
        t0 = time.perf_counter()
        y = float(self._fhe(x))
        t_end = time.perf_counter()
        self.n_fhe += 1
        z = float(np.dot(self.w, xa) + self.b)
        model = float(self._plain(x))  # instrumentation, not counted
        surr = float(t3(z))
        self.fhe_records.append({"x": xa.tolist(), "z": z, "model": model, "surrogate": surr,
                                 "fhe": y, "total": abs(model - y), "approx": abs(model - surr),
                                 "ckks": abs(surr - y), "t": t_end - t0, "t_end": t_end})
        return y


def summarize(recs):
    tot = np.array([r["total"] for r in recs])
    if tot.size == 0 or not np.all(np.isfinite(tot)):
        raise RuntimeError("empty or non-finite errors")
    i = int(np.argmax(tot))
    ts = [r["t"] for r in recs]
    return {"max_error": float(tot[i]), "witness": recs[i], "fhe_counted": len(recs),
            "max_ckks_component": float(max(r["ckks"] for r in recs)),
            "median_eval_ms": 1e3 * float(np.median(ts))}


# --- Arms --------------------------------------------------------------------

def run_autooracle(cc, circ, seed, n_trials):
    cc.reset()
    t0 = time.perf_counter()
    ao = AutoOracle(cc.sigma, cc.fhe, circ["bounds"])
    res = ao.run(n_trials=n_trials, seed=seed, threshold=THRESHOLD)
    wall = time.perf_counter() - t0
    if cc.n_oob:
        raise RuntimeError(f"AutoOracle evaluated {cc.n_oob} out-of-domain inputs")
    s = summarize(list(cc.fhe_records))
    rem = getattr(res, "remeasured_error", None)
    s.update({"wall_s": wall, "sigma_calls": cc.n_sigma, "t3_calls": cc.n_t3,
              "fhe_calls": cc.n_fhe, "n_trials_reported": int(res.n_trials),
              "reported_max_error": float(res.max_error),
              "remeasured_error": None if rem is None else float(rem),
              "reported_minus_observed": float(res.max_error) - s["max_error"],
              "regime": getattr(res, "regime", None), "strategy": getattr(res, "strategy_used", None),
              "verdict": res.verdict, "fhe_calls_within_n_trials": cc.n_fhe <= n_trials})
    return s


def top_k_distinct(cands, k, tol):
    order = sorted(range(len(cands)), key=lambda i: -cands[i]["score"])
    X = np.array([c["x"] for c in cands], dtype=np.float64)
    picked: list[int] = []
    for i in order:
        if picked and not np.all(np.max(np.abs(X[picked] - X[i]), axis=1) > tol):
            continue
        picked.append(i)
        if len(picked) == k:
            break
    return [cands[i] for i in picked]


def timed_stream(cc, points, t0, budget_s):
    """Start evaluations while elapsed < budget; count those completed within it."""
    started, counted, exhausted = 0, [], True
    for x in points:
        if time.perf_counter() - t0 >= budget_s:
            exhausted = False
            break
        started += 1
        absolute_error(cc.sigma(x), cc.fhe(x))  # raises on invalid output
        rec = cc.fhe_records[-1]
        if rec["t_end"] - t0 <= budget_s:
            counted.append(rec)
    return started, counted, exhausted


def run_surrogate(cc, circ, seed, budget_s, reserve_s, args):
    cc.reset()
    t0 = time.perf_counter()
    cc.deadline = t0 + budget_s - reserve_s
    deadline_hit, plain_trials = False, None
    try:
        oracle = FHEOracle(plaintext_fn=cc.sigma, fhe_fn=cc.t3_plain, input_dim=circ["d"],
                           input_bounds=circ["bounds"], seed=seed)
        plain_trials = int(oracle.run(n_trials=args.plain_budget, threshold=THRESHOLD).n_trials)
    except EvaluationError as exc:
        if not _is_deadline(exc):
            raise
        deadline_hit = True
    finally:
        cc.deadline = None
    t_plain = time.perf_counter() - t0
    if cc.n_oob:
        raise RuntimeError(f"plaintext search evaluated {cc.n_oob} out-of-domain inputs")
    cands = list(cc.t3_records)
    if not cands:
        raise RuntimeError("plaintext search produced no candidate before its deadline")
    sigma_plain = cc.n_sigma
    tol = args.distinct_frac * float(np.min(cc.hi - cc.lo))
    picked = top_k_distinct(cands, args.k, tol)
    t_select = time.perf_counter() - t0
    started, counted, _ = timed_stream(cc, [c["x"] for c in picked], t0, budget_s)
    wall = time.perf_counter() - t0
    if cc.n_oob:
        raise RuntimeError(f"SURROGATE evaluated {cc.n_oob} out-of-domain inputs")
    if not counted:
        raise RuntimeError("SURROGATE completed no CKKS evaluation within its wall budget")
    s = summarize(counted)
    best = max(cands, key=lambda c: c["score"])
    s.update({"wall_s": wall, "budget_s": budget_s, "unused_s": max(0.0, budget_s - wall),
              "plain_phase_s": t_plain, "select_s": t_select - t_plain, "deadline_s": budget_s - reserve_s,
              "deadline_hit": deadline_hit, "plain_trials_reported": plain_trials,
              "plain_evals": len(cands), "sigma_calls_plain": sigma_plain,
              "sigma_calls_ckks": cc.n_sigma - sigma_plain, "sigma_calls": cc.n_sigma,
              "t3_calls": cc.n_t3, "fhe_calls": cc.n_fhe, "fhe_started": started,
              "n_distinct": len(picked), "distinct_tol": tol,
              "plain_best_score": best["score"], "plain_best_z": best["z"],
              "top1_ckks_total": counted[0]["total"],
              "picked_scores": [c["score"] for c in picked]})
    return s


def corner_pool(bounds):
    lo = np.array([a for a, _ in bounds])
    hi = np.array([b for _, b in bounds])
    mid = (lo + hi) / 2.0
    pts = [list(v) for v in itertools.product(*bounds)]
    for i in range(len(bounds)):
        for v in (lo[i], hi[i]):
            p = mid.copy()
            p[i] = v
            pts.append(p.tolist())
    pts.append(mid.tolist())
    return np.array(pts, dtype=np.float64)


def run_corner(cc, pool, seed, budget_s):
    cc.reset()
    order = np.random.default_rng([seed, 1]).permutation(len(pool))
    t0 = time.perf_counter()
    started, counted, exhausted = timed_stream(cc, (pool[i].tolist() for i in order), t0, budget_s)
    wall = time.perf_counter() - t0
    if cc.n_oob:
        raise RuntimeError(f"corner set evaluated {cc.n_oob} out-of-domain inputs")
    if not counted:
        raise RuntimeError("corner set completed no CKKS evaluation within its wall budget")
    s = summarize(counted)
    s.update({"wall_s": wall, "budget_s": budget_s, "sigma_calls": cc.n_sigma, "t3_calls": cc.n_t3,
              "fhe_calls": cc.n_fhe, "fhe_started": started, "pool_size": len(pool),
              "pool_exhausted": exhausted})
    return s


def reencrypt(cc, x, k):
    cc.reset()
    for _ in range(k):
        y = cc.fhe(x)
        absolute_error(cc.fhe_records[-1]["model"], y)  # NaN/Inf raises
    tot = [r["total"] for r in cc.fhe_records]
    ck = [r["ckks"] for r in cc.fhe_records]
    return {"k": k, "mean": float(np.mean(tot)), "min": float(np.min(tot)), "max": float(np.max(tot)),
            "ckks_mean": float(np.mean(ck))}


def run_seed(cc, circ, pool, seed, args):
    t_seed = time.perf_counter()
    ao_runs = [run_autooracle(cc, circ, seed, args.n_trials) for _ in range(args.ao_repeats)]
    budget = float(np.median([r["wall_s"] for r in ao_runs]))
    reserve_raw = RESERVE_FACTOR * args.k * ao_runs[0]["median_eval_ms"] / 1e3
    reserve = min(reserve_raw, RESERVE_MAX_SHARE * budget)
    arms = {"surrogate": run_surrogate(cc, circ, seed, budget, reserve, args),
            "autooracle": ao_runs[0],
            "corner": run_corner(cc, pool, seed, budget)}
    for m in ARMS:
        arms[m]["witness_reenc"] = reencrypt(cc, arms[m]["witness"]["x"], args.repeats)
    # Bind by key; ARMS order is for display only.
    sur = arms["surrogate"]["max_error"]
    ao = arms["autooracle"]["max_error"]
    cor = arms["corner"]["max_error"]
    return {"seed": seed, "budget_s": budget, "reserve_s": reserve, "reserve_uncapped_s": reserve_raw,
            "reserve_clamped": bool(reserve < reserve_raw),
            "ao_replicates": [{k: r[k] for k in ("wall_s", "max_error", "fhe_calls", "regime",
                                                 "strategy", "n_trials_reported")} for r in ao_runs],
            **arms, "sur_minus_corner": sur - cor, "sur_minus_autooracle": sur - ao,
            "c1_seed_ok": bool(sur >= cor - CORNER_TOL), "seed_wall_s": time.perf_counter() - t_seed}


def approx_supremum(w, b, bounds, n_grid=2_000_001):
    """Max of |sigmoid - T3| over the box via the exact z-range of w.x + b."""
    zmin = b + sum(min(wi * lo, wi * hi) for wi, (lo, hi) in zip(w, bounds))
    zmax = b + sum(max(wi * lo, wi * hi) for wi, (lo, hi) in zip(w, bounds))
    z = np.linspace(zmin, zmax, n_grid)
    err = np.abs(sigmoid(z) - t3(z))
    i = int(np.argmax(err))
    return {"z_min": float(zmin), "z_max": float(zmax), "sup": float(err[i]), "z_at_sup": float(z[i]),
            "at_endpoint": i in (0, n_grid - 1)}


# --- Statistics --------------------------------------------------------------

def _stat(f, v):
    return float(f(v)) if len(v) else None


def compare(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = a - b
    nz = d[d != 0]
    if nz.size == 0:
        p, rbc = 1.0, 0.0
    else:
        p = float(wilcoxon(a, b, alternative="two-sided", zero_method="wilcox").pvalue)
        ranks = rankdata(np.abs(nz))
        wp, wm = float(ranks[nz > 0].sum()), float(ranks[nz < 0].sum())
        rbc = (wp - wm) / (wp + wm)
    ok = b > 0
    ratios = a[ok] / b[ok]
    return {"n": int(a.size), "median_a": float(np.median(a)), "median_b": float(np.median(b)),
            "median_diff": float(np.median(d)), "q1_diff": float(np.percentile(d, 25)),
            "q3_diff": float(np.percentile(d, 75)), "median_ratio": _stat(np.median, ratios),
            "n_ratio": int(ratios.size),
            "rank_biserial": rbc, "a_larger": int(np.sum(d > 0)), "a_smaller": int(np.sum(d < 0)),
            "ties": int(np.sum(d == 0)), "p_value": p,
            "test": "scipy.stats.wilcoxon two-sided, zero_method='wilcox', method='auto'"}


def holm(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


# --- Provenance --------------------------------------------------------------

def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return None


def _sha256(rel):
    with open(os.path.join(ROOT, rel), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def provenance():
    return {"git_head": _git("rev-parse", "HEAD"), "core_sha256": _sha256(CORE_REL),
            "autoconfig_sha256": _sha256(AUTO_REL), "script_sha256": _sha256(SCRIPT_REL)}


def prereg_check():
    try:
        blob = subprocess.run(["git", "show", f"{PREREG_COMMIT}:{PREREG_REL}"], cwd=ROOT,
                              capture_output=True, check=True).stdout
        committed = hashlib.sha256(blob).hexdigest()
    except Exception:
        committed = None
    current = _sha256(PREREG_REL)
    return {"path": PREREG_REL, "commit": PREREG_COMMIT, "sha256_committed": committed,
            "sha256_worktree": current, "match": committed is not None and committed == current}


def module_table():
    rows, seen = [], set()
    for name, mod in sorted(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if not f:
            continue
        rel = os.path.relpath(os.path.abspath(f), ROOT)
        if (rel in seen or rel.startswith("..")
                or not (rel.startswith("fhe_oracle") or rel.startswith("benchmarks"))):
            continue
        seen.add(rel)
        tracked = _git("ls-files", "--error-unmatch", rel) is not None
        same = (subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=ROOT).returncode == 0
                if tracked else False)
        rows.append({"module": name, "path": rel, "sha256": _sha256(rel), "tracked": tracked,
                     "matches_head": same})
    return rows


def _ver(pkg):
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


def _repo_rel(arg):
    prefix, sep, val = arg.partition("=") if arg.startswith("--") else ("", "", arg)
    if not os.path.isabs(val):
        return arg
    rel = os.path.relpath(val, ROOT)
    rel = f"<outside-repo>/{os.path.basename(val)}" if rel.startswith("..") else rel
    return f"{prefix}{sep}{rel}"


def environment():
    cpu = None
    if platform.system() == "Darwin":
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                             text=True).stdout.strip() or None
    return {"fhe_oracle_version": FHE_ORACLE_VERSION,
            "packages": {p: _ver(p) for p in ("tenseal", "cma", "numpy", "scipy")},
            "python": sys.version.split()[0], "python_executable": os.path.basename(sys.executable),
            "platform": platform.platform(), "cpu": cpu or platform.processor(),
            "cpu_count": os.cpu_count(), "argv": [_repo_rel(a) for a in sys.argv]}


# --- Output ------------------------------------------------------------------

def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items() if k not in ("ctx", "fhe", "plain")}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if np.isfinite(v) else str(v)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    return obj


CSV_ARM_FIELDS = ["max_error", "fhe_calls", "fhe_counted", "sigma_calls", "t3_calls", "wall_s",
                  "median_eval_ms", "witness_z", "witness_approx", "witness_ckks", "reenc_mean",
                  "reenc_min", "reenc_max"]


def write_csv(path, rows):
    fields = ["seed", "budget_s", "reserve_s", "reserve_clamped", "seed_wall_s", "ao_wall_1", "ao_wall_2", "ao_wall_3",
              "ao_replicate_max_errors"]
    fields += [f"{m}_{f}" for m in ARMS for f in CSV_ARM_FIELDS]
    fields += ["surrogate_plain_evals", "surrogate_plain_trials_reported", "surrogate_plain_phase_s",
               "surrogate_deadline_hit", "surrogate_n_distinct", "surrogate_fhe_started",
               "surrogate_unused_s", "surrogate_plain_best_score", "surrogate_plain_best_z",
               "surrogate_top1_ckks_total", "autooracle_n_trials_reported",
               "autooracle_reported_max_error", "autooracle_regime", "autooracle_strategy",
               "autooracle_verdict", "corner_fhe_started", "corner_pool_exhausted",
               "sur_minus_corner", "sur_minus_autooracle", "c1_seed_ok"]
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        for r in rows:
            walls = [x["wall_s"] for x in r["ao_replicates"]] + [None] * 3
            row = {"seed": r["seed"], "budget_s": r["budget_s"], "reserve_s": r["reserve_s"],
                   "reserve_clamped": r["reserve_clamped"],
                   "seed_wall_s": r["seed_wall_s"], "ao_wall_1": walls[0], "ao_wall_2": walls[1],
                   "ao_wall_3": walls[2],
                   "ao_replicate_max_errors": ";".join(f"{x['max_error']:.9g}" for x in r["ao_replicates"])}
            for m in ARMS:
                s = r[m]
                row.update({f"{m}_max_error": s["max_error"], f"{m}_fhe_calls": s["fhe_calls"],
                            f"{m}_fhe_counted": s["fhe_counted"], f"{m}_sigma_calls": s["sigma_calls"],
                            f"{m}_t3_calls": s["t3_calls"], f"{m}_wall_s": s["wall_s"],
                            f"{m}_median_eval_ms": s["median_eval_ms"],
                            f"{m}_witness_z": s["witness"]["z"],
                            f"{m}_witness_approx": s["witness"]["approx"],
                            f"{m}_witness_ckks": s["witness"]["ckks"],
                            f"{m}_reenc_mean": s["witness_reenc"]["mean"],
                            f"{m}_reenc_min": s["witness_reenc"]["min"],
                            f"{m}_reenc_max": s["witness_reenc"]["max"]})
            su, ao, co = r["surrogate"], r["autooracle"], r["corner"]
            row.update({"surrogate_plain_evals": su["plain_evals"],
                        "surrogate_plain_trials_reported": su["plain_trials_reported"],
                        "surrogate_plain_phase_s": su["plain_phase_s"],
                        "surrogate_deadline_hit": su["deadline_hit"], "surrogate_n_distinct": su["n_distinct"],
                        "surrogate_fhe_started": su["fhe_started"], "surrogate_unused_s": su["unused_s"],
                        "surrogate_plain_best_score": su["plain_best_score"],
                        "surrogate_plain_best_z": su["plain_best_z"],
                        "surrogate_top1_ckks_total": su["top1_ckks_total"],
                        "autooracle_n_trials_reported": ao["n_trials_reported"],
                        "autooracle_reported_max_error": ao["reported_max_error"],
                        "autooracle_regime": ao["regime"], "autooracle_strategy": ao["strategy"],
                        "autooracle_verdict": ao["verdict"], "corner_fhe_started": co["fhe_started"],
                        "corner_pool_exhausted": co["pool_exhausted"],
                        "sur_minus_corner": r["sur_minus_corner"],
                        "sur_minus_autooracle": r["sur_minus_autooracle"], "c1_seed_ok": r["c1_seed_ok"]})
            wr.writerow(row)


def _e(v):
    return f"{v:.3e}"


def _f(v, spec):
    return "undefined" if v is None or not np.isfinite(v) else format(v, spec)


def _mr(vals, spec=".0f"):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "—"
    lo, med, hi = min(vals), float(np.median(vals)), max(vals)
    return format(med, spec) if lo == hi else f"{format(med, spec)} ({format(lo, spec)}–{format(hi, spec)})"


def build_markdown(R):
    cfg, rows, stats, V = R["config"], R["per_seed"], R["statistics"], R["verdicts"]
    thr2 = MEDIAN_FRAC * SUPREMUM
    L = []
    a = L.append
    a("# Study B: plaintext-surrogate search, then CKKS verification")
    a("")
    a(f"Pre-registration: `{PREREG_REL}`, section B (committed at `{PREREG_COMMIT[:7]}`). "
      f"Generated by `{SCRIPT_REL}`; every number below is written by that script.")
    a("")
    if not cfg["preregistered_configuration"]:
        a("**This run does NOT use the pre-registered configuration** (seeds, n_trials, K or the "
          "fixed choices differ); it is not a result for study B.")
        a("")
    a(f"- Run (UTC): {R['started_utc']} to {R['finished_utc']}; total runtime {R['runtime_s'] / 60:.1f} min")
    a(f"- Run status: **{R['run_status']}**" + (f" ({R['error']})" if R.get("error") else ""))
    a(f"- Seeds completed: {len(rows)} of {len(cfg['seeds'])}")
    a("")
    a("## Outcome")
    a("")
    a("| Criterion | Required | Observed | Result |")
    a("|---|---|---|---|")
    c1, c2 = V["c1"], V["c2"]
    c1_obs = "—" if c1["seeds_ok"] is None else f"{c1['seeds_ok']} of {c1['n_run']}"
    a(f"| C1: SURROGATE ≥ corner set − {CORNER_TOL:g} | ≥ {PASS_SEEDS} of {len(PREREG_SEEDS)} seeds | "
      f"{c1_obs} | {c1['result']} |")
    a(f"| C2: median SURROGATE error | ≥ {thr2:.4f} (0.99 × {SUPREMUM}) | "
      f"{_f(c2['median'], '.4f')} | {c2['result']} |")
    a("")
    a(f"**Study B: {V['overall']}.** {V['reason']}")
    a("")

    a("## 1. Setup")
    a("")
    c, k = R["circuit"], R["circuit"]["ckks"]
    a(f"- **Circuit:** `{c['source']}`; intended model σ(w·x + b), surrogate T3(z) = 0.5 + z/4 − z³/48, "
      f"CKKS program evaluates T3; domain [{c['bounds'][0][0]}, {c['bounds'][0][1]}]^{c['d']}; "
      f"N = {k['poly_modulus_degree']}, coeff_mod_bit_sizes {k['coeff_mod_bit_sizes']}, scale "
      f"2^{k['scale_bits']}; one context and key set for the run.")
    a("- **Error:** |σ(z) − CKKS output| per evaluation, fresh encryption per call.")
    sup = R["references"]["supremum"]
    a(f"- **Plaintext supremum** of |σ − T3| over the box (exact z-range [{sup['z_min']:.3f}, "
      f"{sup['z_max']:.3f}]): {sup['sup']:.6f} at z = {sup['z_at_sup']:.3f}; pre-registered value "
      f"{SUPREMUM}, match within {SUP_TOL:g}: {R['references']['supremum_match']}.")
    a(f"- **Wall budget per seed:** median wall time of {cfg['ao_repeats']} AutoOracle runs "
      f"(`AutoOracle(σ, CKKS, bounds).run(n_trials={cfg['n_trials']}, seed=s, threshold={THRESHOLD})`) "
      "on that seed; the first run is the AutoOracle arm.")
    a(f"- **SURROGATE:** `FHEOracle(σ, T3 in float64, seed=s).run(n_trials={cfg['plain_budget']})` with "
      "library defaults (CMA-ES, σ0 = 1.0, box centre, no restarts); candidates sorted by plaintext "
      f"|σ − T3|; greedy distinct pick with L∞ distance > {cfg['distinct_frac']:g} × box width; top "
      f"K = {cfg['k']} evaluated under CKKS while the budget lasts. The plaintext phase is interrupted "
      f"at budget − min({RESERVE_FACTOR:g} × K × median CKKS time, {RESERVE_MAX_SHARE:g} × budget). "
      "Unused time is not spent.")
    a("- **Corner/boundary set:** 2^8 vertices, 16 face centres and the centre in a seeded order "
      "(`default_rng([s, 1])`), evaluated until the budget is spent.")
    a("- **Counting:** evaluations start only while elapsed < budget and count only if they finish "
      "within it. Every σ, T3-plaintext and CKKS call is counted per arm; AutoOracle's count includes "
      "its re-measurements.")
    a(f"- **Criteria:** C1 SURROGATE ≥ corner − {CORNER_TOL:g} on ≥ {PASS_SEEDS} of 20 seeds; C2 median "
      f"SURROGATE ≥ {thr2:.4f}. Both required. Decision: build a `surrogate_fn` API only if study B passes.")
    a("- **Statistics:** paired two-sided Wilcoxon signed-rank (scipy, `zero_method='wilcox'`), "
      "SURROGATE vs corner and SURROGATE vs AutoOracle, Holm across the two. Effect sizes: median "
      "paired difference (IQR), median ratio, matched-pairs rank-biserial correlation.")
    a(f"- **Seeds:** {cfg['seeds']}.")
    a("")
    a("**Choices fixed in the script header before the run:**")
    a("")
    for ch in R["choices"]:
        a(f"- {ch}")
    a("")

    a("## 2. Results")
    a("")
    if not rows:
        a("No seed completed.")
        a("")
    else:
        a("| Seed | Budget s | SURROGATE | AutoOracle | Corner set | SURROGATE − corner | C1 | "
          "CKKS evals S / A / C |")
        a("|---:|---:|---:|---:|---:|---:|:---:|---:|")
        for r in rows:
            a(f"| {r['seed']} | {r['budget_s']:.2f} | {r['surrogate']['max_error']:.6f} | "
              f"{r['autooracle']['max_error']:.6f} | {r['corner']['max_error']:.6f} | "
              f"{r['sur_minus_corner']:+.3e} | {'ok' if r['c1_seed_ok'] else 'no'} | "
              f"{r['surrogate']['fhe_counted']} / {r['autooracle']['fhe_calls']} / "
              f"{r['corner']['fhe_counted']} |")
        a("")
        if stats.get("comparisons"):
            a("| Comparison | Median A | Median B | Median A − B (IQR) | Median ratio | Rank-biserial | "
              "A larger / smaller / tie | Wilcoxon p | Holm p |")
            a("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
            for key, lab in (("surrogate_vs_corner", "SURROGATE (A) vs corner (B)"),
                             ("surrogate_vs_autooracle", "SURROGATE (A) vs AutoOracle (B)")):
                s = stats["comparisons"][key]
                ratio = _f(s["median_ratio"], ".6f")
                if s["n_ratio"] < s["n"]:
                    ratio += f" [n = {s['n_ratio']} of {s['n']}]"
                a(f"| {lab} | {s['median_a']:.6f} | {s['median_b']:.6f} | {s['median_diff']:+.3e} "
                  f"({s['q1_diff']:+.2e} to {s['q3_diff']:+.2e}) | {ratio} | "
                  f"{s['rank_biserial']:+.2f} | {s['a_larger']} / {s['a_smaller']} / {s['ties']} | "
                  f"{s['p_value']:.3g} | {s['p_holm']:.3g} |")
            a("")
        su = [r["surrogate"] for r in rows]
        a("**SURROGATE internals** (median, range over seeds).")
        a("")
        a("| Item | Value |")
        a("|---|---|")
        a(f"| Plaintext candidates evaluated (T3 calls) | {_mr([s['plain_evals'] for s in su])} |")
        a(f"| σ calls: plaintext phase / CKKS phase | {_mr([s['sigma_calls_plain'] for s in su])} / "
          f"{_mr([s['sigma_calls_ckks'] for s in su])} |")
        a(f"| Plaintext phase time, s | {_mr([s['plain_phase_s'] for s in su], '.3f')} |")
        a(f"| Plaintext deadline hit | {sum(s['deadline_hit'] for s in su)} of {len(su)} seeds |")
        a(f"| Distinct candidates picked | {_mr([s['n_distinct'] for s in su])} |")
        a(f"| CKKS evaluations started / counted | {_mr([s['fhe_started'] for s in su])} / "
          f"{_mr([s['fhe_counted'] for s in su])} |")
        a(f"| Wall time used / budget, s | {_mr([s['wall_s'] for s in su], '.2f')} / "
          f"{_mr([s['budget_s'] for s in su], '.2f')} |")
        a(f"| Unused budget, s | {_mr([s['unused_s'] for s in su], '.2f')} |")
        a(f"| CKKS reserve capped at {RESERVE_MAX_SHARE:g} × budget | "
          f"{sum(r['reserve_clamped'] for r in rows)} of {len(rows)} seeds |")
        a(f"| Best plaintext \\|σ − T3\\| | {_mr([s['plain_best_score'] for s in su], '.6f')} |")
        a(f"| z at best plaintext candidate | {_mr([s['plain_best_z'] for s in su], '.4f')} |")
        a(f"| CKKS error at top-ranked candidate | {_mr([s['top1_ckks_total'] for s in su], '.6f')} |")
        a("")
        ao = [r["autooracle"] for r in rows]
        co = [r["corner"] for r in rows]
        regimes = sorted({f"{x['regime']}/{x['strategy']}" for x in ao})
        a("**AutoOracle and corner set.**")
        a("")
        a("| Item | Value |")
        a("|---|---|")
        a(f"| AutoOracle regime/strategy (arm run) | {', '.join(regimes)} |")
        a(f"| AutoOracle CKKS calls / `result.n_trials` | {_mr([x['fhe_calls'] for x in ao])} / "
          f"{_mr([x['n_trials_reported'] for x in ao])} |")
        a(f"| AutoOracle runs with CKKS calls ≤ n_trials | {sum(x['fhe_calls_within_n_trials'] for x in ao)} of {len(ao)} |")
        a(f"| AutoOracle reported max_error − observed max | {_mr([x['reported_minus_observed'] for x in ao], '+.2e')} |")
        spread = [max(q["max_error"] for q in r["ao_replicates"]) - min(q["max_error"] for q in r["ao_replicates"])
                  for r in rows]
        a(f"| AutoOracle replicate wall times, s (all runs) | "
          f"{_mr([q['wall_s'] for r in rows for q in r['ao_replicates']], '.2f')} |")
        a(f"| Spread of max error across the {cfg['ao_repeats']} AutoOracle runs per seed | {_mr(spread, '.2e')} |")
        a(f"| Corner CKKS evaluations started / counted | {_mr([x['fhe_started'] for x in co])} / "
          f"{_mr([x['fhe_counted'] for x in co])} |")
        a(f"| Corner pool (273) exhausted before the budget | {sum(x['pool_exhausted'] for x in co)} of {len(co)} seeds |")
        a("")
        a("**Error at each arm's witness** (median over seeds; re-encryptions uncounted).")
        a("")
        a("| Arm | Max error | \\|z\\| at witness | Approximation \\|σ − T3\\| | CKKS execution | "
          "Re-encrypted mean (min–max over seeds) |")
        a("|---|---:|---:|---:|---:|---|")
        for m in ARMS:
            a(f"| {LABELS[m]} | {np.median([r[m]['max_error'] for r in rows]):.6f} | "
              f"{np.median([abs(r[m]['witness']['z']) for r in rows]):.4f} | "
              f"{np.median([r[m]['witness']['approx'] for r in rows]):.6f} | "
              f"{_e(np.median([r[m]['witness']['ckks'] for r in rows]))} | "
              f"{np.median([r[m]['witness_reenc']['mean'] for r in rows]):.6f} "
              f"({min(r[m]['witness_reenc']['min'] for r in rows):.6f}–"
              f"{max(r[m]['witness_reenc']['max'] for r in rows):.6f}) |")
        a("")
    ref = R["references"]
    if ref.get("vertices"):
        a("| Reference vertex | z | Re-encryptions | Mean total error | Min | Max |")
        a("|---|---:|---:|---:|---:|---:|")
        for key, lab in (("vertex_zmin", "Minimises z"), ("vertex_zmax", "Maximises z")):
            v = ref["vertices"][key]
            a(f"| {lab} | {v['z']:.3f} | {v['k']} | {v['mean']:.6f} | {v['min']:.6f} | {v['max']:.6f} |")
        a("")

    a("## 3. Deviations and caveats")
    a("")
    for d in R["caveats"]:
        a(f"- {d}")
    a("")
    a("## 4. Provenance")
    a("")
    ps, pe = R["provenance"]["start"], R["provenance"]["end"]
    a("| Item | Start | End |")
    a("|---|---|---|")
    a(f"| git HEAD | `{ps['git_head']}` | `{pe['git_head']}` |")
    for key, rel in (("core_sha256", CORE_REL), ("autoconfig_sha256", AUTO_REL), ("script_sha256", SCRIPT_REL)):
        a(f"| SHA-256 `{rel}` | `{ps[key]}` | `{pe[key]}` |")
    pr = R["provenance"]["preregistration"]
    a(f"| Pre-registration `{pr['path']}` | commit `{pr['commit']}`, SHA-256 `{pr['sha256_committed']}` | "
      f"working tree `{pr['sha256_worktree']}` (match: {pr['match']}) |")
    a("")
    a(f"Provenance unchanged during the run: **{R['provenance']['unchanged']}**. Script tracked in git "
      f"at run time: {R['provenance']['script_tracked']}.")
    mods = R["provenance"]["modules"]
    diff = [m["path"] for m in mods if not m["matches_head"]]
    a("")
    a(f"Loaded fhe_oracle and benchmark modules: {len(mods)}; differing from HEAD or untracked: "
      f"{', '.join(f'`{p}`' for p in diff) if diff else 'none'} (SHA-256 of each in summary.json).")
    a("")
    env = R["environment"]
    pk = env["packages"]
    ls, le = R["loadavg_start"], R["loadavg_end"]
    a(f"fhe-oracle {env['fhe_oracle_version']}; tenseal {pk['tenseal']}, cma {pk['cma']}, numpy "
      f"{pk['numpy']}, scipy {pk['scipy']}; Python {env['python']}; {env['platform']}; {env['cpu']} "
      f"({env['cpu_count']} logical cores). Load average (1/5/15 min): {ls[0]:.1f}/{ls[1]:.1f}/{ls[2]:.1f} "
      f"at start, {le[0]:.1f}/{le[1]:.1f}/{le[2]:.1f} at end; the machine was shared.")
    a("")
    a("## 5. Reproduce")
    a("")
    a("```bash")
    a(R["reproduce_command"])
    a("```")
    a("")
    a(f"Command used: `{' '.join([env['python_executable']] + env['argv'])}`")
    a("")
    return "\n".join(L)


# --- Main --------------------------------------------------------------------

def _parse_seeds(text):
    out = []
    for part in text.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out += list(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Pre-registered study B (surrogate search).")
    ap.add_argument("--seeds", default="11-30")
    ap.add_argument("--n-trials", type=int, default=PREREG_N_TRIALS)
    ap.add_argument("--k", type=int, default=PREREG_K)
    ap.add_argument("--ao-repeats", type=int, default=DEFAULTS["ao_repeats"])
    ap.add_argument("--plain-budget", type=int, default=DEFAULTS["plain_budget"])
    ap.add_argument("--distinct-frac", type=float, default=DEFAULTS["distinct_frac"])
    ap.add_argument("--repeats", type=int, default=5, help="re-encryptions per witness")
    ap.add_argument("--ref-k", type=int, default=20, help="re-encryptions per reference vertex")
    ap.add_argument("--timing-evals", type=int, default=3)
    ap.add_argument("--max-minutes", type=float, default=30.0)
    ap.add_argument("--out", default=os.path.join(THIS_DIR, "results", "surrogate_search"))
    args = ap.parse_args(argv)
    if not tsa.HAVE_TENSEAL and BUILD["circuit"] is build_circuit:
        print("TenSEAL is not installed; this study requires real CKKS.")
        return 2

    seeds = _parse_seeds(args.seeds)
    cap_s = 60.0 * args.max_minutes
    t_start = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    load_start = list(os.getloadavg())
    prov_start = provenance()
    prereg = prereg_check()
    run_status, error = "COMPLETE", None
    rows: list[dict] = []
    refs: dict = {"supremum": None, "supremum_match": None}
    preflight: dict = {}

    circ = BUILD["circuit"]()
    sup = approx_supremum(circ["w"], circ["b"], circ["bounds"])
    refs["supremum"] = sup
    refs["supremum_match"] = bool(abs(sup["sup"] - SUPREMUM) <= SUP_TOL)
    if not prereg["match"]:
        run_status, error = "ABORTED_REFERENCE_MISMATCH", "pre-registration hash differs from its commit"
    elif not refs["supremum_match"]:
        run_status, error = ("ABORTED_REFERENCE_MISMATCH",
                             f"plaintext supremum {sup['sup']:.6f} differs from {SUPREMUM}")
    cc = Counter(circ)
    pool = corner_pool(circ["bounds"])

    if run_status == "COMPLETE":
        rng = np.random.default_rng(12345)
        ts = []
        for _ in range(args.timing_evals):
            x = rng.uniform(cc.lo, cc.hi).tolist()
            t1 = time.perf_counter()
            y = circ["fhe"](x)
            ts.append(time.perf_counter() - t1)
            absolute_error(circ["plain"](x), y)
        med = float(np.median(ts))
        n_evals = len(seeds) * ((args.ao_repeats + 1) * args.n_trials + args.k + 3 * args.repeats) + 2 * args.ref_k
        preflight = {"eval_s": ts, "median_eval_s": med, "projected_evals": n_evals,
                     "projected_s": n_evals * med}
        print(f"median CKKS eval {med * 1e3:.1f} ms; projected {n_evals * med / 60:.1f} min "
              f"for {n_evals} evaluations", flush=True)
        if time.perf_counter() - t_start + n_evals * med > cap_s:
            run_status = "ABORTED_COMPUTE_CAP"
            error = f"projected {n_evals * med / 60:.1f} min exceeds the {args.max_minutes:g}-min cap"

    if run_status == "COMPLETE":
        for seed in seeds:
            try:
                row = run_seed(cc, circ, pool, seed, args)
            except Exception as exc:  # keep completed seeds
                run_status, error = "ABORTED_ERROR", f"seed {seed}: {type(exc).__name__}: {exc}"
                print(error, flush=True)
                break
            rows.append(row)
            print(f"seed {seed}: budget {row['budget_s']:.2f}s SURROGATE {row['surrogate']['max_error']:.6f} "
                  f"[{row['surrogate']['fhe_counted']} CKKS, {row['surrogate']['plain_evals']} plain] "
                  f"AutoOracle {row['autooracle']['max_error']:.6f} [{row['autooracle']['fhe_calls']}] "
                  f"corner {row['corner']['max_error']:.6f} [{row['corner']['fhe_counted']}] "
                  f"C1 {'ok' if row['c1_seed_ok'] else 'no'} ({row['seed_wall_s']:.1f}s)", flush=True)
            if provenance() != prov_start:
                run_status, error = "ABORTED_PROVENANCE_CHANGED", f"after seed {seed}"
                break
            elapsed = time.perf_counter() - t_start
            remaining = float(np.median([r["seed_wall_s"] for r in rows])) * (len(seeds) - len(rows))
            if remaining > 0 and elapsed + remaining > cap_s:
                run_status, error = "ABORTED_COMPUTE_CAP", f"projection exceeded the cap after seed {seed}"
                break
        if run_status == "COMPLETE":
            try:
                w, lo, hi = circ["w"], cc.lo, cc.hi
                refs["vertices"] = {}
                for key, x in (("vertex_zmin", np.where(w > 0, lo, hi)), ("vertex_zmax", np.where(w > 0, hi, lo))):
                    refs["vertices"][key] = {"x": x.tolist(), "z": float(np.dot(w, x) + circ["b"]),
                                             **reencrypt(cc, x.tolist(), args.ref_k)}
            except Exception as exc:
                run_status, error = "ABORTED_ERROR", f"references: {type(exc).__name__}: {exc}"

    # Statistics and verdicts.
    prov_end = provenance()
    unchanged = prov_end == prov_start and run_status != "ABORTED_PROVENANCE_CHANGED"
    if prov_end != prov_start and not run_status.startswith("ABORTED"):
        run_status, error = "ABORTED_PROVENANCE_CHANGED", "at end of run"
    stats: dict = {}
    complete = run_status == "COMPLETE" and len(rows) == len(seeds)
    if complete and len(rows) > 0:
        sur = [r["surrogate"]["max_error"] for r in rows]
        stats["comparisons"] = {
            "surrogate_vs_corner": compare(sur, [r["corner"]["max_error"] for r in rows]),
            "surrogate_vs_autooracle": compare(sur, [r["autooracle"]["max_error"] for r in rows])}
        adj = holm([s["p_value"] for s in stats["comparisons"].values()])
        for s, p in zip(stats["comparisons"].values(), adj):
            s["p_holm"] = p
    prereg_cfg = (seeds == PREREG_SEEDS and args.n_trials == PREREG_N_TRIALS and args.k == PREREG_K
                  and args.ao_repeats == DEFAULTS["ao_repeats"]
                  and args.plain_budget == DEFAULTS["plain_budget"]
                  and args.distinct_frac == DEFAULTS["distinct_frac"])
    n_ok = int(sum(r["c1_seed_ok"] for r in rows)) if rows else None
    med = float(np.median([r["surrogate"]["max_error"] for r in rows])) if rows else None
    if not unchanged:
        label = "INVALID (provenance changed)"
    elif run_status != "COMPLETE":
        label = f"UNDETERMINED ({run_status.lower()})"
    elif not prereg_cfg:
        label = "NOT A PRE-REGISTERED RESULT"
    else:
        label = None
    c1 = {"seeds_ok": n_ok, "n_run": len(rows),
          "result": label or ("PASS" if n_ok >= PASS_SEEDS else "FAIL")}
    c2 = {"median": med, "required": MEDIAN_FRAC * SUPREMUM,
          "result": label or ("PASS" if med >= MEDIAN_FRAC * SUPREMUM else "FAIL")}
    if not unchanged:
        overall, reason = "INVALID", "git HEAD, core.py, autoconfig.py or this script changed during the run."
    elif run_status != "COMPLETE":
        overall, reason = "UNDETERMINED", f"The run stopped: {run_status} ({error})."
    elif not prereg_cfg:
        overall, reason = "NOT A STUDY RESULT", "The configuration differs from the pre-registration."
    elif c1["result"] == "PASS" and c2["result"] == "PASS":
        overall, reason = "PASS", "Both criteria met. Pre-registered decision: build a `surrogate_fn` API."
    else:
        failed = [n for n, c in (("C1", c1), ("C2", c2)) if c["result"] != "PASS"]
        overall = "FAIL"
        reason = (f"Criterion {' and '.join(failed)} not met. Pre-registered decision: do not build a "
                  "`surrogate_fn` API.")

    choices = [
        f"Wall budget: median of {args.ao_repeats} AutoOracle wall times on the seed (the pre-registration's "
        "\"median wall time ... on that seed\"); the first run is the AutoOracle arm.",
        f"SURROGATE plaintext search: FHEOracle defaults on σ vs T3 in float64, n_trials = {args.plain_budget}, "
        "seed = s; pycma may stop early.",
        f"Plaintext deadline: budget − min({RESERVE_FACTOR:g} × K × median CKKS time of the seed's first "
        f"AutoOracle run, {RESERVE_MAX_SHARE:g} × budget); the cap keeps the plaintext phase from starting "
        "past its deadline when K or the CKKS time is large.",
        f"Distinct candidates: L∞ distance > {args.distinct_frac:g} × box width from every picked point.",
        "CKKS phase and corner set: start only while elapsed < budget, count only if completed within it; "
        "SURROGATE leftover time is unused and never buys CKKS evaluations beyond K.",
        "AutoOracle metric includes every CKKS call it makes, re-measurements included.",
        f"Holm across the two SURROGATE comparisons; C1 tolerance {CORNER_TOL:g}; supremum check tolerance {SUP_TOL:g}.",
    ]
    caveats = []
    if run_status != "COMPLETE":
        caveats.append(f"**Run stopped: {run_status}.** {error}")
    caveats += [
        "One circuit, one key set, one machine; seeds vary search and sampling randomness plus unseeded "
        "CKKS encryption noise.",
        "Equal wall-clock budgets depend on machine load; the machine was shared (see load averages). "
        "Budget, per-arm wall time and counted evaluations are reported per seed.",
        "The σ recomputation used to log candidate scores and error decompositions is instrumentation, "
        "identical in kind for all arms and not counted as search effort; it does consume wall time.",
        "Nearly all of this circuit's error is approximation error visible in plaintext, which is what "
        "the study tests; CKKS execution error at the witnesses is reported separately.",
        "Not independently replicated; produced by the tool author's own script.",
    ]

    runtime = time.perf_counter() - t_start
    out_rel = _repo_rel(os.path.abspath(args.out))
    R = {
        "study": "B. Plaintext-surrogate search, then FHE verification",
        "config": {"seeds": seeds, "n_trials": args.n_trials, "k": args.k, "ao_repeats": args.ao_repeats,
                   "plain_budget": args.plain_budget, "distinct_frac": args.distinct_frac,
                   "repeats": args.repeats, "ref_k": args.ref_k, "threshold": THRESHOLD,
                   "max_minutes": args.max_minutes, "preregistered_configuration": prereg_cfg},
        "choices": choices, "caveats": caveats, "run_status": run_status, "error": error,
        "verdicts": {"c1": c1, "c2": c2, "overall": overall, "reason": reason},
        "circuit": {k: v for k, v in circ.items() if k not in ("fhe", "plain", "ctx")},
        "references": refs, "preflight": preflight, "statistics": stats, "per_seed": rows,
        "provenance": {"start": prov_start, "end": prov_end, "unchanged": unchanged,
                       "preregistration": prereg,
                       "script_tracked": _git("ls-files", "--error-unmatch", SCRIPT_REL) is not None,
                       "tracked_changes_at_end": _git("status", "--porcelain", "--untracked-files=no"),
                       "modules": module_table()},
        "environment": environment(),
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "loadavg_start": load_start, "loadavg_end": list(os.getloadavg()), "runtime_s": runtime,
        "reproduce_command": (f"python {SCRIPT_REL} --seeds {args.seeds} --n-trials {args.n_trials} "
                              f"--k {args.k} --ao-repeats {args.ao_repeats} --plain-budget "
                              f"{args.plain_budget} --distinct-frac {args.distinct_frac:g} --repeats "
                              f"{args.repeats} --ref-k {args.ref_k} --max-minutes {args.max_minutes:g} "
                              f"--out {out_rel}"),
    }
    os.makedirs(args.out, exist_ok=True)
    write_csv(os.path.join(args.out, "per_seed.csv"), rows)
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(_clean(R), fh, indent=2)
    code = {"ABORTED_PROVENANCE_CHANGED": 3, "ABORTED_COMPUTE_CAP": 4, "ABORTED_ERROR": 5,
            "ABORTED_REFERENCE_MISMATCH": 7}.get(run_status, 0)
    try:
        text = build_markdown(R)
    except Exception as exc:  # CSV and JSON are already written
        text = (f"# Study B\n\nReport rendering failed ({type(exc).__name__}: {exc}); "
                "see summary.json and per_seed.csv.\n")
        print(f"report.md rendering failed: {type(exc).__name__}: {exc}")
        code = code or 6
    with open(os.path.join(args.out, "report.md"), "w") as fh:
        fh.write(text)
    print(f"status {run_status}; study B {overall}; wrote {out_rel} in {runtime:.0f}s; exit {code}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
