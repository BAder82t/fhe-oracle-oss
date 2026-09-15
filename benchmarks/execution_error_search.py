# Copyright (C) 2026 Bader Alissaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pre-registered study C: does FHEOracle locate CKKS execution error better than sampling?

Protocol: benchmarks/preregistration_2026-09-15.md, section C (committed at 962aab3).
Writes per_seed.csv, summary.json and report.md into --out.

Choices where the pre-registration is not explicit, fixed before the run:
- lr_d8: tenseal_circuits.build_tenseal_lr_d8, domain [-3, 3]^8, default TenSEALContext
  (N=16384, [60,40,40,40,40,60], scale 2^40). Surrogate T3(z) = 0.5 + z/4 - z^3/48.
- cheb15_s30: the only circuit in cheb15_cross_circuit.py. WDBC d=30 logistic regression
  (StandardScaler + LogisticRegression(random_state=42)), Cheb-15 sigmoid fit on [-5, 5],
  build_tenseal_context(degree=15, scale_bits=30) (N=32768, [60]+[30]*16+[60]),
  domain [-0.3, 0.3]^30 (as in that script). "T3" in the fitness is read as this
  circuit's own surrogate polynomial P15(w.x + b).
- Execution error at x: |surrogate(w.x + b) in float64 - CKKS output|, fresh encryption per call.
- Tool: FHEOracle(plaintext_fn=surrogate, fhe_fn=CKKS surrogate, input_dim, input_bounds,
  seed=s).run(n_trials=200, threshold=0.01); everything else library defaults.
- Counting: every plaintext and FHE call. The tool's count includes run()'s re-measurement.
  Each baseline gets the tool's counted FHE calls for that seed (1 plaintext + 1 FHE per point).
- Uniform random: default_rng(s). Sobol: qmc.Sobol(d, scramble=True, rng=default_rng([s, 2])),
  first n points of a 2^ceil(log2 n) block, scaled to the box. Corner set: pool of all vertices,
  2d face centres and the centre; n points drawn without replacement with default_rng([s, 1]);
  the whole pool if it is smaller than n (disclosed).
- Per-seed metric: max execution error over all counted evaluations of that method.
- Win: tool >= max(random, Sobol, corner) on the seed (exact comparison; ties go to the tool).
  A circuit passes with >= 8 of 10 wins; the study passes if any circuit passes.
- Wilcoxon: scipy two-sided, zero_method='wilcox', tool vs each baseline per circuit; Holm
  across all comparisons run (also reported against the planned family of 6).
- Order per seed: tool, random, Sobol, corner. One process; one context and key set per circuit.
- Compute cap (--max-minutes, default 30): 3 uncounted timing evaluations per circuit before any
  seed; a circuit whose projected cost would push the cumulative projection over the cap is not
  run and is recorded BLOCKED. The projection is re-checked after each seed; overrun aborts.
- Uncounted references, excluded from comparisons: each method's worst witness re-encrypted
  --repeats times per seed; the two extreme-|z| vertices re-encrypted --ref-k times.
- Provenance: git HEAD and SHA-256 of fhe_oracle/core.py and this script, checked after every
  seed and at the end; any change aborts the run.
- Exit codes: 0 complete, 3 provenance changed, 4 compute cap, 5 evaluation error,
  6 report rendering failed. CSV and JSON are written in every case.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata as md
import json
import math
import os
import platform
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone

import numpy as np
from scipy.stats import qmc, rankdata, wilcoxon

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, THIS_DIR)
warnings.filterwarnings("ignore", message="Could not import matplotlib")

from fhe_oracle import FHEOracle  # noqa: E402
from fhe_oracle import __version__ as FHE_ORACLE_VERSION  # noqa: E402
from fhe_oracle.adapters import tenseal_adapter as tsa  # noqa: E402
from fhe_oracle.fitness import absolute_error  # noqa: E402

SCRIPT_REL = "benchmarks/execution_error_search.py"
CORE_REL = "fhe_oracle/core.py"
PREREG_REL = "benchmarks/preregistration_2026-09-15.md"
PREREG_COMMIT = "962aab31e4239d074130a8f80394dac3e2731c1d"
PREREG_SEEDS = list(range(11, 21))
PREREG_N_TRIALS = 200
PREREG_CIRCUITS = ["lr_d8", "cheb15_s30"]
THRESHOLD = 0.01
PASS_WINS = 8
BASELINES = ["random", "sobol", "corner"]
METHODS = ["tool"] + BASELINES
LABELS = {"tool": "FHE Oracle", "random": "Uniform random", "sobol": "Sobol (scrambled)",
          "corner": "Corner/boundary set"}


# --- Circuits ----------------------------------------------------------------

def build_lr_d8():
    from tenseal_circuits import build_tenseal_lr_d8
    ctx = tsa.TenSEALContext()
    c = build_tenseal_lr_d8(ctx)
    return {
        "name": "lr_d8", "source": "benchmarks/tenseal_circuits.py::build_tenseal_lr_d8",
        "d": int(c["d"]), "bounds": [(float(a), float(b)) for a, b in c["bounds"]],
        "w": np.asarray(c["weights"], dtype=np.float64), "b": float(c["bias"]),
        # Same operation order as the CKKS program.
        "surrogate": lambda z: z * 0.25 - z ** 3 * (1.0 / 48.0) + 0.5,
        "surrogate_text": "T3(z) = 0.5 + z/4 - z^3/48, z = w.x + b",
        "fhe": c["fhe"],
        "ckks": {"poly_modulus_degree": tsa.CKKS_POLY_MODULUS_DEGREE,
                 "coeff_mod_bit_sizes": list(tsa.CKKS_COEFF_MOD_BIT_SIZES),
                 "scale_bits": int(round(math.log2(tsa.CKKS_GLOBAL_SCALE)))},
        "ctx": ctx,
    }


def build_cheb15_s30():
    from cheb15_cross_circuit import DEGREE, _fit_wdbc_model
    from chebyshev_polynomials import (build_tenseal_context, eval_poly_plaintext,
                                       fit_cheb_sigmoid, make_tenseal_poly_lr_fhe_fn)
    w, b, _ = _fit_wdbc_model()
    approx = fit_cheb_sigmoid(DEGREE)
    ctx = build_tenseal_context(degree=DEGREE, scale_bits=30)
    return {
        "name": "cheb15_s30",
        "source": ("benchmarks/cheb15_cross_circuit.py (WDBC d=30, Cheb-15, scale 2^30, "
                   "bounds [-0.3, 0.3])"),
        "d": int(w.shape[0]), "bounds": [(-0.3, 0.3)] * int(w.shape[0]),
        "w": np.asarray(w, dtype=np.float64), "b": float(b),
        "surrogate": lambda z: eval_poly_plaintext(z, approx),
        "surrogate_text": (f"P15(z): degree-{DEGREE} Chebyshev fit of the sigmoid on "
                           f"{list(approx.domain)}, power basis, z = w.x + b"),
        "fhe": make_tenseal_poly_lr_fhe_fn(w, b, approx, ctx),
        "ckks": {"poly_modulus_degree": int(ctx.N), "coeff_mod_bit_sizes": list(ctx.chain),
                 "scale_bits": int(ctx.scale_bits)},
        "fit_error_on_fit_interval": float(approx.fit_error),
        "ctx": ctx,
    }


BUILDERS = {"lr_d8": build_lr_d8, "cheb15_s30": build_cheb15_s30}


class Counter:
    """Counts model calls; logs the execution error of every FHE call."""

    def __init__(self, circ):
        self.w, self.b = circ["w"], circ["b"]
        self.sur, self._fhe = circ["surrogate"], circ["fhe"]
        self.lo = np.array([a for a, _ in circ["bounds"]])
        self.hi = np.array([c for _, c in circ["bounds"]])
        self.reset()

    def reset(self):
        self.n_plain = 0
        self.n_fhe = 0
        self.n_oob = 0
        self.records: list[dict] = []

    def plain(self, x):
        self.n_plain += 1
        return float(self.sur(float(np.dot(self.w, np.asarray(x, dtype=np.float64)) + self.b)))

    def fhe(self, x):
        xa = np.asarray(x, dtype=np.float64)
        if np.any(xa < self.lo) or np.any(xa > self.hi):
            self.n_oob += 1
        t0 = time.perf_counter()
        y = float(self._fhe(x))
        dt = time.perf_counter() - t0
        self.n_fhe += 1
        # Instrumentation: surrogate recomputed for the log, not counted.
        z = float(np.dot(self.w, xa) + self.b)
        s = float(self.sur(z))
        self.records.append({"x": xa.tolist(), "z": z, "surrogate": s, "ckks": y,
                             "err": abs(s - y), "t": dt})
        return y


def summarize(recs):
    errs = np.array([r["err"] for r in recs])
    if errs.size == 0 or not np.all(np.isfinite(errs)):
        raise RuntimeError("empty or non-finite execution errors")
    i = int(np.argmax(errs))
    ts = [r["t"] for r in recs]
    return {"max_error": float(errs[i]), "witness": recs[i], "witness_index": i,
            "fhe_evals": len(recs), "n_ge_threshold": int(np.sum(errs >= THRESHOLD)),
            "max_abs_z": float(max(abs(r["z"]) for r in recs)),
            "median_eval_ms": 1e3 * float(np.median(ts)), "fhe_time_s": float(sum(ts))}


# --- Methods -----------------------------------------------------------------

def run_tool(cc, circ, seed, n_trials):
    cc.reset()
    t0 = time.perf_counter()
    oracle = FHEOracle(plaintext_fn=cc.plain, fhe_fn=cc.fhe, input_dim=circ["d"],
                       input_bounds=circ["bounds"], seed=seed)
    res = oracle.run(n_trials=n_trials, threshold=THRESHOLD)
    wall = time.perf_counter() - t0
    if cc.n_oob:
        raise RuntimeError(f"tool evaluated {cc.n_oob} out-of-domain inputs")
    recs = list(cc.records)
    if cc.n_plain != cc.n_fhe or cc.n_fhe != res.n_trials + 1:
        raise RuntimeError(f"unexpected counts: plain {cc.n_plain}, fhe {cc.n_fhe}, "
                           f"n_trials {res.n_trials}")
    last = recs[-1]
    tol = 1e-12 * max(1.0, abs(res.remeasured_error))
    if (not np.allclose(last["x"], res.worst_input, rtol=0, atol=1e-12)
            or abs(last["err"] - res.remeasured_error) > tol):
        raise RuntimeError("tool re-measurement does not match the logged evaluation")
    search_max = max(r["err"] for r in recs[:-1])
    if abs(search_max - res.search_max_error) > 1e-12 * max(1.0, search_max):
        raise RuntimeError("tool search_max_error does not match the logged evaluations")
    s = summarize(recs)
    s.update({"plain_evals": cc.n_plain, "wall_s": wall, "n_trials_reported": res.n_trials,
              "reported_max_error": res.max_error, "remeasured_error": res.remeasured_error,
              "search_max_error": res.search_max_error, "verdict": res.verdict,
              "pool_exhausted": False})
    return s


def run_stream(cc, points, n):
    cc.reset()
    t0 = time.perf_counter()
    k = 0
    for x in points:
        if k >= n:
            break
        absolute_error(cc.plain(x), cc.fhe(x))  # raises on invalid output
        k += 1
    wall = time.perf_counter() - t0
    if cc.n_oob:
        raise RuntimeError(f"baseline evaluated {cc.n_oob} out-of-domain inputs")
    s = summarize(list(cc.records))
    s.update({"plain_evals": cc.n_plain, "wall_s": wall, "pool_exhausted": k < n})
    return s


def random_points(lo, hi, seed):
    rng = np.random.default_rng(seed)
    while True:
        yield rng.uniform(lo, hi).tolist()


def sobol_points(lo, hi, seed, n):
    m = max(0, math.ceil(math.log2(n)))
    u = qmc.Sobol(d=len(lo), scramble=True, rng=np.random.default_rng([seed, 2])).random_base2(m)
    for row in qmc.scale(u[:n], lo, hi):
        yield row.tolist()


def corner_pool_size(d):
    return 2 ** d + 2 * d + 1


def corner_point(i, lo, hi):
    """Pool index -> point: vertices, then face centres (lo, hi per axis), then centre."""
    d = len(lo)
    nv = 2 ** d
    if i < nv:
        bits = (i >> np.arange(d)) & 1
        return np.where(bits == 1, hi, lo)
    mid = (lo + hi) / 2.0
    j = i - nv
    if j < 2 * d:
        p = mid.copy()
        k, side = divmod(j, 2)
        p[k] = hi[k] if side else lo[k]
        return p
    return mid


def corner_points(lo, hi, seed, n):
    size = corner_pool_size(len(lo))
    idx = np.random.default_rng([seed, 1]).choice(size, size=min(n, size), replace=False)
    for i in idx:
        yield corner_point(int(i), lo, hi).tolist()


def reencrypt(cc, x, k):
    cc.reset()
    for _ in range(k):
        y = cc.fhe(x)
        absolute_error(cc.records[-1]["surrogate"], y)  # NaN/Inf raises
    errs = [r["err"] for r in cc.records]
    return {"k": k, "mean": float(np.mean(errs)), "min": float(np.min(errs)),
            "max": float(np.max(errs))}


def run_seed(cc, circ, seed, args):
    """Tool, then the baselines at the tool's counted FHE calls, then witness re-encryptions."""
    t_seed = time.perf_counter()
    tool = run_tool(cc, circ, seed, args.n_trials)
    n = tool["fhe_evals"]
    res = {"tool": tool,
           "random": run_stream(cc, random_points(cc.lo, cc.hi, seed), n),
           "sobol": run_stream(cc, sobol_points(cc.lo, cc.hi, seed, n), n),
           "corner": run_stream(cc, corner_points(cc.lo, cc.hi, seed, n), n)}
    for m in METHODS:
        res[m]["witness_reenc"] = reencrypt(cc, res[m]["witness"]["x"], args.repeats)
    best = max(BASELINES, key=lambda m: res[m]["max_error"])
    bmax = res[best]["max_error"]
    return {"seed": seed, **res, "best_baseline": best, "best_baseline_max_error": bmax,
            "tool_ge_best_baseline": bool(tool["max_error"] >= bmax),
            "ratio_tool_over_best_baseline": tool["max_error"] / bmax if bmax > 0 else None,
            "seed_wall_s": time.perf_counter() - t_seed}


def references(cc, circ, k):
    """Re-encrypt the two extreme-z vertices and the box centre (uncounted)."""
    lo, hi, w, b = cc.lo, cc.hi, circ["w"], circ["b"]
    vmax = np.where(w > 0, hi, lo)
    vmin = np.where(w > 0, lo, hi)
    refs = {"z_abs_max_box": float(max(abs(np.dot(w, vmax) + b), abs(np.dot(w, vmin) + b)))}
    for key, x in (("vertex_zmax", vmax), ("vertex_zmin", vmin), ("centre", (lo + hi) / 2)):
        refs[key] = {"x": x.tolist(), "z": float(np.dot(w, x) + b), **reencrypt(cc, x.tolist(), k)}
    return refs


# --- Statistics --------------------------------------------------------------

def _stat(f, v):
    return float(f(v)) if len(v) else None


def compare(t, b):
    t = np.asarray(t, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = t - b
    nz = d[d != 0]
    if nz.size == 0:
        p, rbc = 1.0, 0.0
    else:
        p = float(wilcoxon(t, b, alternative="two-sided", zero_method="wilcox").pvalue)
        ranks = rankdata(np.abs(nz))
        wp, wm = float(ranks[nz > 0].sum()), float(ranks[nz < 0].sum())
        rbc = (wp - wm) / (wp + wm)
    ok = b > 0
    ratios = t[ok] / b[ok]  # undefined where the baseline max error is 0
    return {
        "n": int(t.size), "median_tool": float(np.median(t)), "median_baseline": float(np.median(b)),
        "median_ratio": _stat(np.median, ratios),
        "q1_ratio": _stat(lambda r: np.percentile(r, 25), ratios),
        "q3_ratio": _stat(lambda r: np.percentile(r, 75), ratios), "n_ratios": int(ratios.size),
        "median_diff": float(np.median(d)), "rank_biserial": rbc,
        "tool_larger": int(np.sum(d > 0)), "tool_smaller": int(np.sum(d < 0)),
        "ties": int(np.sum(d == 0)), "p_value": p,
        "test": "scipy.stats.wilcoxon two-sided, zero_method='wilcox', method='auto'",
    }


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
            "script_sha256": _sha256(SCRIPT_REL)}


def module_table():
    """SHA-256 and HEAD match for every loaded fhe_oracle and benchmark module."""
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
    return {
        "fhe_oracle_version": FHE_ORACLE_VERSION,
        "packages": {p: _ver(p) for p in ("tenseal", "cma", "numpy", "scipy", "scikit-learn")},
        "python": sys.version.split()[0], "python_executable": os.path.basename(sys.executable),
        "platform": platform.platform(), "cpu": cpu or platform.processor(),
        "cpu_count": os.cpu_count(), "argv": [_repo_rel(a) for a in sys.argv],
    }


# --- Output ------------------------------------------------------------------

def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items() if k != "ctx"}
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


def write_csv(path, circuits):
    fields = ["circuit", "seed", "seed_wall_s", "tool_n_trials_reported", "tool_verdict",
              "tool_remeasured_error", "tool_search_max_error"]
    for m in METHODS:
        fields += [f"{m}_max_error", f"{m}_fhe_evals", f"{m}_plain_evals", f"{m}_wall_s",
                   f"{m}_median_eval_ms", f"{m}_witness_z", f"{m}_max_abs_z", f"{m}_n_ge_threshold",
                   f"{m}_pool_exhausted", f"{m}_reenc_mean", f"{m}_reenc_min", f"{m}_reenc_max"]
    fields += ["best_baseline", "best_baseline_max_error", "tool_ge_best_baseline",
               "ratio_tool_over_best_baseline"]
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        for c in circuits.values():
            for r in c.get("per_seed", []):
                row = {"circuit": c["name"], "seed": r["seed"], "seed_wall_s": r["seed_wall_s"],
                       "tool_n_trials_reported": r["tool"]["n_trials_reported"],
                       "tool_verdict": r["tool"]["verdict"],
                       "tool_remeasured_error": r["tool"]["remeasured_error"],
                       "tool_search_max_error": r["tool"]["search_max_error"]}
                for m in METHODS:
                    s = r[m]
                    row.update({
                        f"{m}_max_error": s["max_error"], f"{m}_fhe_evals": s["fhe_evals"],
                        f"{m}_plain_evals": s["plain_evals"], f"{m}_wall_s": s["wall_s"],
                        f"{m}_median_eval_ms": s["median_eval_ms"],
                        f"{m}_witness_z": s["witness"]["z"], f"{m}_max_abs_z": s["max_abs_z"],
                        f"{m}_n_ge_threshold": s["n_ge_threshold"],
                        f"{m}_pool_exhausted": s["pool_exhausted"],
                        f"{m}_reenc_mean": s["witness_reenc"]["mean"],
                        f"{m}_reenc_min": s["witness_reenc"]["min"],
                        f"{m}_reenc_max": s["witness_reenc"]["max"]})
                row.update({"best_baseline": r["best_baseline"],
                            "best_baseline_max_error": r["best_baseline_max_error"],
                            "tool_ge_best_baseline": r["tool_ge_best_baseline"],
                            "ratio_tool_over_best_baseline": r["ratio_tool_over_best_baseline"]})
                wr.writerow(row)


def _e(v):
    return f"{v:.3e}"


def _f(v, spec):
    return "undefined" if v is None or not np.isfinite(v) else format(v, spec)


def build_markdown(R):
    cfg, circuits, stats = R["config"], R["circuits"], R["statistics"]
    L = []
    a = L.append
    a("# Study C: search that targets CKKS execution error")
    a("")
    a(f"Pre-registration: `{PREREG_REL}`, section C (committed at `{PREREG_COMMIT[:7]}`). "
      f"Generated by `{SCRIPT_REL}`; every number below is written by that script.")
    a("")
    if not cfg["preregistered_configuration"]:
        a("**This run does NOT use the pre-registered configuration** (seeds, n_trials or "
          "circuits differ); it is not a result for study C.")
        a("")
    a(f"- Run (UTC): {R['started_utc']} to {R['finished_utc']}; total runtime "
      f"{R['runtime_s'] / 60:.1f} min")
    a(f"- Run status: **{R['run_status']}**")
    a("")
    a("## Outcome")
    a("")
    a("| Circuit | Status | Tool ≥ best baseline (seeds run) | Criterion (≥ 8 of 10) |")
    a("|---|---|---:|---|")
    for c in circuits.values():
        v = R["verdicts"]["per_circuit"][c["name"]]
        wins = "—" if v["wins"] is None else f"{v['wins']} / {v['n_run']}"
        a(f"| {c['name']} | {c['status']} | {wins} | {v['verdict']} |")
    a("")
    a(f"**Study C overall: {R['verdicts']['overall']}.** {R['verdicts']['overall_reason']}")
    a("")

    a("## 1. Setup")
    a("")
    a("Execution error at input x is |surrogate(x) − CKKS(x)|: the circuit's polynomial surrogate "
      "evaluated in float64 plaintext against the same polynomial evaluated under CKKS (fresh "
      "encryption per call). It excludes approximation error against the sigmoid, so it is the "
      "error class that plaintext testing cannot reveal.")
    a("")
    a("| Circuit | Source | d | Domain | Surrogate | N | coeff_mod_bit_sizes | Scale |")
    a("|---|---|---:|---|---|---:|---|---:|")
    for c in circuits.values():
        k = c["ckks"]
        a(f"| {c['name']} | {c['source']} | {c['d']} | [{c['bounds'][0][0]}, {c['bounds'][0][1]}]^{c['d']} "
          f"| {c['surrogate_text']} | {k['poly_modulus_degree']} | {k['coeff_mod_bit_sizes']} | "
          f"2^{k['scale_bits']} |")
    a("")
    a(f"- **Tool:** `FHEOracle(plaintext_fn=surrogate, fhe_fn=CKKS surrogate, input_dim, "
      f"input_bounds, seed=s).run(n_trials={cfg['n_trials']}, threshold={THRESHOLD})`, all other "
      "arguments library defaults (CMA-ES, σ0 = 1.0, start at box centre, no restarts, no random "
      "floor, no plugins).")
    a("- **Uniform random:** NumPy `default_rng(s)` draws from the box.")
    a("- **Sobol:** `scipy.stats.qmc.Sobol(d, scramble=True, rng=default_rng([s, 2]))`, first n "
      "points of a 2^⌈log2 n⌉ block, scaled to the box.")
    a("- **Corner/boundary set:** pool of all 2^d vertices, 2d face centres and the centre; n "
      "points drawn without replacement in a seeded order (`default_rng([s, 1])`).")
    a("- **Counting:** every plaintext and FHE call. The tool's count includes the "
      "re-measurement `run()` performs at its witness. Each baseline receives exactly the tool's "
      "counted FHE calls for that seed (one plaintext and one FHE call per point).")
    a("- **Metric:** per-seed maximum execution error over all counted evaluations of the method.")
    a("- **Criterion:** tool ≥ max(random, Sobol, corner) on a seed counts as a win (exact "
      "comparison, ties to the tool); a circuit passes with ≥ 8 of 10 wins; the study passes if "
      "any circuit passes.")
    a("- **Statistics:** paired two-sided Wilcoxon signed-rank (scipy, `zero_method='wilcox'`) of "
      "the tool against each baseline per circuit, Holm-adjusted across the comparisons run; "
      "also adjusted against the planned family of 6 (2 circuits × 3 baselines) with unrun "
      "comparisons set to p = 1. Effect sizes: median per-seed ratio tool/baseline (IQR), median "
      "paired difference, matched-pairs rank-biserial correlation (+1 = tool larger on every seed).")
    a(f"- **Seeds:** {cfg['seeds']}. Per seed the order is tool, random, Sobol, corner, in one "
      "process with one context and key set per circuit.")
    a("")
    a("**Choices made where the pre-registration is not explicit (fixed in the script header "
      "before the run):**")
    a("")
    for ch in R["choices"]:
        a(f"- {ch}")
    a("")

    a("## 2. Results")
    a("")
    for c in circuits.values():
        name = c["name"]
        a(f"### {name}")
        a("")
        if c["status"] == "BLOCKED":
            p = c["preflight"]
            a(f"**Not run.** Median time per CKKS evaluation in the pre-run timing probe: "
              f"{p['median_eval_s']:.2f} s ({len(p['eval_s'])} uncounted evaluations; context "
              f"build {p['build_s']:.1f} s). Projected cost of the full protocol "
              f"({p['projected_evals']} evaluations): {p['projected_s'] / 60:.0f} min, against a "
              f"cap of {cfg['max_minutes']:.0f} min for the whole invocation. Nothing was "
              "reduced; the circuit was left out and its criterion is undetermined.")
            a("")
            continue
        rows = c.get("per_seed", [])
        if c["status"] != "COMPLETE":
            err = f" Error: `{c['error']}`." if c.get("error") else ""
            a(f"**{c['status']}: {len(rows)} of {len(cfg['seeds'])} seeds run.{err} The criterion "
              "is not evaluated and statistics are omitted; any rows below are partial.**")
            a("")
        if not rows:
            continue
        a("Per-seed maximum execution error (counted FHE evaluations per arm in the last column).")
        a("")
        a("| Seed | Tool | Uniform random | Sobol | Corner set | Best baseline | Tool ≥ best | "
          "Tool / best | FHE evals |")
        a("|---:|---:|---:|---:|---:|---|:---:|---:|---:|")
        for r in rows:
            a(f"| {r['seed']} | {_e(r['tool']['max_error'])} | {_e(r['random']['max_error'])} | "
              f"{_e(r['sobol']['max_error'])} | {_e(r['corner']['max_error'])} | "
              f"{LABELS[r['best_baseline']]} | {'yes' if r['tool_ge_best_baseline'] else 'no'} | "
              f"{_f(r['ratio_tool_over_best_baseline'], '.3f')} | {r['tool']['fhe_evals']} |")
        a("")
        ex = [f"{m} (seeds {[r['seed'] for r in rows if r[m]['pool_exhausted']]})"
              for m in BASELINES if any(r[m]["pool_exhausted"] for r in rows)]
        pool = corner_pool_size(c["d"])
        a(f"Corner pool size: {pool}. " + ("Pool smaller than the budget: " + ", ".join(ex) + "."
                                            if ex else "No baseline ran out of points."))
        a("")
        if name in stats["comparisons"]:
            a("| Tool vs | Median tool | Median baseline | Median ratio (IQR) | Median difference | "
              "Rank-biserial | Tool larger / smaller / tie | Wilcoxon p | Holm p (run) | "
              "Holm p (planned 6) |")
            a("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
            for m in BASELINES:
                s = stats["comparisons"][name][m]
                if s["median_ratio"] is None:
                    ratio = "undefined (baseline max error 0 on every seed)"
                else:
                    ratio = f"{s['median_ratio']:.3f} ({s['q1_ratio']:.3f}–{s['q3_ratio']:.3f})"
                    if s["n_ratios"] < s["n"]:
                        ratio += f" [{s['n_ratios']} of {s['n']} seeds]"
                a(f"| {LABELS[m]} | {_e(s['median_tool'])} | {_e(s['median_baseline'])} | {ratio} | "
                  f"{s['median_diff']:+.3e} | {s['rank_biserial']:+.2f} | {s['tool_larger']} / "
                  f"{s['tool_smaller']} / {s['ties']} | {s['p_value']:.3g} | {s['p_holm']:.3g} | "
                  f"{s['p_holm_planned6']:.3g} |")
            a("")
            rb = stats["tool_vs_best_baseline"][name]
            a(f"Tool / per-seed best baseline: median ratio {_f(rb['median_ratio'], '.3f')} (range "
              f"{_f(rb['min_ratio'], '.3f')}–{_f(rb['max_ratio'], '.3f')}); best baseline by seed: "
              + ", ".join(f"{k} {v}" for k, v in rb["best_counts"].items()) + ".")
            a("")
        a("**Where the error sits (uncounted references).**")
        a("")
        a("| Method | Median max error | Median re-encrypted mean at witness | Median max / "
          "re-encrypted mean | Median \\|z\\| at witness | Median max \\|z\\| evaluated |")
        a("|---|---:|---:|---:|---:|---:|")
        for m in METHODS:
            q = [r[m]["max_error"] / r[m]["witness_reenc"]["mean"] for r in rows
                 if r[m]["witness_reenc"]["mean"] > 0]
            a(f"| {LABELS[m]} | {_e(np.median([r[m]['max_error'] for r in rows]))} | "
              f"{_e(np.median([r[m]['witness_reenc']['mean'] for r in rows]))} | "
              f"{_f(float(np.median(q)) if q else None, '.2f')} | "
              f"{np.median([abs(r[m]['witness']['z']) for r in rows]):.3f} | "
              f"{np.median([r[m]['max_abs_z'] for r in rows]):.3f} |")
        a("")
        if "references" in c:
            ref = c["references"]
            a(f"Largest |z| = |w·x + b| on the box: {ref['z_abs_max_box']:.3f}. Each method's witness "
              f"was re-encrypted {cfg['repeats']} times per seed; the ratio column shows how far the "
              "reported maximum sits above the typical error at the same input (fresh CKKS noise).")
            a("")
            a("| Reference input | z | Re-encryptions | Mean error | Min | Max |")
            a("|---|---:|---:|---:|---:|---:|")
            for key, lab in (("vertex_zmax", "Vertex maximising z"),
                             ("vertex_zmin", "Vertex minimising z"), ("centre", "Box centre")):
                v = ref[key]
                a(f"| {lab} | {v['z']:.3f} | {v['k']} | {_e(v['mean'])} | {_e(v['min'])} | "
                  f"{_e(v['max'])} |")
        else:
            a(f"Each method's witness was re-encrypted {cfg['repeats']} times per seed. The "
              "extreme-|z| reference inputs were not measured because the run stopped.")
        a("")
        med_t = np.median([r["tool"]["median_eval_ms"] for r in rows])
        a(f"Timing: median {med_t:.1f} ms per CKKS evaluation (tool); median seed wall time "
          f"{np.median([r['seed_wall_s'] for r in rows]):.1f} s. Tool verdicts at threshold "
          f"{THRESHOLD}: {sum(r['tool']['verdict'] == 'FAIL' for r in rows)} FAIL of {len(rows)}; "
          f"evaluations with execution error ≥ {THRESHOLD}: tool "
          f"{sum(r['tool']['n_ge_threshold'] for r in rows)}, random "
          f"{sum(r['random']['n_ge_threshold'] for r in rows)}, Sobol "
          f"{sum(r['sobol']['n_ge_threshold'] for r in rows)}, corner "
          f"{sum(r['corner']['n_ge_threshold'] for r in rows)}.")
        a("")

    a("## 3. Deviations and caveats")
    a("")
    for d in R["caveats"]:
        a(f"- {d}")
    a("")

    a("## 4. Provenance")
    a("")
    ps, pe = R["provenance"]["start"], R["provenance"]["end"]
    env = R["environment"]
    a("| Item | Start | End |")
    a("|---|---|---|")
    a(f"| git HEAD | `{ps['git_head']}` | `{pe['git_head']}` |")
    a(f"| SHA-256 `{CORE_REL}` | `{ps['core_sha256']}` | `{pe['core_sha256']}` |")
    a(f"| SHA-256 `{SCRIPT_REL}` | `{ps['script_sha256']}` | `{pe['script_sha256']}` |")
    a("")
    a(f"Provenance unchanged during the run: **{R['provenance']['unchanged']}**. "
      f"Script tracked in git at run time: {R['provenance']['script_tracked']}.")
    a("")
    mods = R["provenance"]["modules"]
    diff = [m["path"] for m in mods if not m["matches_head"]]
    a(f"Loaded fhe_oracle and benchmark modules: {len(mods)}; differing from HEAD or untracked: "
      f"{', '.join(f'`{p}`' for p in diff) if diff else 'none'} (SHA-256 of each in summary.json).")
    a("")
    pk = env["packages"]
    a(f"fhe-oracle {env['fhe_oracle_version']}; tenseal {pk['tenseal']}, cma {pk['cma']}, numpy "
      f"{pk['numpy']}, scipy {pk['scipy']}, scikit-learn {pk['scikit-learn']}; Python "
      f"{env['python']}; {env['platform']}; {env['cpu']} ({env['cpu_count']} logical cores). "
      f"Load average (1/5/15 min): {R['loadavg_start'][0]:.1f}/{R['loadavg_start'][1]:.1f}/"
      f"{R['loadavg_start'][2]:.1f} at start, {R['loadavg_end'][0]:.1f}/{R['loadavg_end'][1]:.1f}/"
      f"{R['loadavg_end'][2]:.1f} at end; the machine was shared.")
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
    ap = argparse.ArgumentParser(description="Pre-registered study C (execution-error search).")
    ap.add_argument("--circuits", default=",".join(PREREG_CIRCUITS))
    ap.add_argument("--seeds", default="11-20")
    ap.add_argument("--n-trials", type=int, default=PREREG_N_TRIALS)
    ap.add_argument("--repeats", type=int, default=5, help="re-encryptions per witness")
    ap.add_argument("--ref-k", type=int, default=20, help="re-encryptions per reference input")
    ap.add_argument("--timing-evals", type=int, default=3)
    ap.add_argument("--max-minutes", type=float, default=30.0)
    ap.add_argument("--out", default=os.path.join(THIS_DIR, "results", "execution_error_search"))
    args = ap.parse_args(argv)
    if not tsa.HAVE_TENSEAL:
        print("TenSEAL is not installed; this study requires real CKKS.")
        return 2

    names = [n for n in args.circuits.split(",") if n]
    seeds = _parse_seeds(args.seeds)
    cap_s = 60.0 * args.max_minutes
    t_start = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    load_start = list(os.getloadavg())
    prov_start = provenance()
    run_status = "COMPLETE"

    # Build every circuit and time a few uncounted evaluations before any seed.
    circuits, planned, projected_total = {}, [], 0.0
    for name in names:
        t0 = time.perf_counter()
        circ = BUILDERS[name]()
        build_s = time.perf_counter() - t0
        lo = np.array([x for x, _ in circ["bounds"]])
        hi = np.array([x for _, x in circ["bounds"]])
        rng = np.random.default_rng(12345)
        ts = []
        for _ in range(args.timing_evals):
            x = rng.uniform(lo, hi).tolist()
            t1 = time.perf_counter()
            y = circ["fhe"](x)
            ts.append(time.perf_counter() - t1)
            absolute_error(circ["surrogate"](float(np.dot(circ["w"], x) + circ["b"])), y)
        med = float(np.median(ts))
        n_evals = (len(seeds) * (len(METHODS) * (args.n_trials + 1 + args.repeats))
                   + 3 * args.ref_k)
        proj = n_evals * med
        circ["preflight"] = {"build_s": build_s, "eval_s": ts, "median_eval_s": med,
                             "projected_evals": n_evals, "projected_s": proj}
        circuits[name] = circ
        print(f"{name}: build {build_s:.1f}s, median eval {med * 1e3:.1f} ms, "
              f"projected {proj / 60:.1f} min for {n_evals} evaluations")
    elapsed = time.perf_counter() - t_start
    for name in names:
        proj = circuits[name]["preflight"]["projected_s"]
        if elapsed + projected_total + proj <= cap_s:
            planned.append(name)
            projected_total += proj
            circuits[name]["status"] = "PLANNED"
        else:
            circuits[name]["status"] = "BLOCKED"
            circuits[name]["fhe"] = None
            circuits[name]["ctx"] = None
            print(f"{name}: BLOCKED (projection exceeds the {args.max_minutes:.0f}-min cap)")

    stop = False
    for ci, name in enumerate(planned):
        circ = circuits[name]
        cc = Counter(circ)
        rows = []
        circ["per_seed"] = rows
        circ["status"] = "RUNNING"
        for seed in seeds:
            try:
                row = run_seed(cc, circ, seed, args)
            except Exception as exc:  # keep completed seeds; exit code 5
                circ["error"] = f"seed {seed}: {type(exc).__name__}: {exc}"
                print(f"{name}: {circ['error']}", flush=True)
                run_status, stop = "ABORTED_ERROR", True
                break
            rows.append(row)
            print(f"{name} seed {seed}: tool {_e(row['tool']['max_error'])} "
                  f"[{row['tool']['fhe_evals']}] random {_e(row['random']['max_error'])} sobol "
                  f"{_e(row['sobol']['max_error'])} corner {_e(row['corner']['max_error'])} -> "
                  f"{'WIN' if row['tool_ge_best_baseline'] else 'loss'} ({row['seed_wall_s']:.1f}s)",
                  flush=True)
            if provenance() != prov_start:
                run_status, stop = "ABORTED_PROVENANCE_CHANGED", True
                break
            elapsed = time.perf_counter() - t_start
            per_seed = float(np.median([r["seed_wall_s"] for r in rows]))
            later = sum(circuits[nm]["preflight"]["projected_s"] for nm in planned[ci + 1:])
            remaining = per_seed * (len(seeds) - len(rows)) + later
            if remaining > 0 and elapsed + remaining > cap_s:
                run_status, stop = "ABORTED_COMPUTE_CAP", True
                break
        circ["status"] = "COMPLETE" if len(rows) == len(seeds) else "INCOMPLETE"
        if circ["status"] == "COMPLETE" and not stop:
            try:
                circ["references"] = references(cc, circ, args.ref_k)
            except Exception as exc:
                circ["error"] = f"references: {type(exc).__name__}: {exc}"
                print(f"{name}: {circ['error']}", flush=True)
                run_status, stop = "ABORTED_ERROR", True
        if stop:
            break
    for name in names:
        if circuits[name]["status"] == "PLANNED":
            circuits[name]["status"] = "NOT_STARTED"

    # Statistics on completed circuits.
    stats = {"comparisons": {}, "tool_vs_best_baseline": {}}
    flat = []
    for name in names:
        c = circuits[name]
        if c["status"] != "COMPLETE":
            continue
        rows = c["per_seed"]
        tool = [r["tool"]["max_error"] for r in rows]
        stats["comparisons"][name] = {}
        for m in BASELINES:
            s = compare(tool, [r[m]["max_error"] for r in rows])
            stats["comparisons"][name][m] = s
            flat.append(s)
        ratios = [r["ratio_tool_over_best_baseline"] for r in rows
                  if r["ratio_tool_over_best_baseline"] is not None]
        stats["tool_vs_best_baseline"][name] = {
            "wins": int(sum(r["tool_ge_best_baseline"] for r in rows)),
            "median_ratio": _stat(np.median, ratios), "min_ratio": _stat(np.min, ratios),
            "max_ratio": _stat(np.max, ratios), "n_ratios": len(ratios),
            "best_counts": {LABELS[m]: sum(r["best_baseline"] == m for r in rows) for m in BASELINES}}
    pv = [s["p_value"] for s in flat]
    for s, adj in zip(flat, holm(pv) if pv else []):
        s["p_holm"] = adj
    planned6 = pv + [1.0] * (len(PREREG_CIRCUITS) * len(BASELINES) - len(pv))
    for s, adj in zip(flat, holm(planned6)[: len(pv)] if pv else []):
        s["p_holm_planned6"] = adj

    # Verdicts. Provenance is checked first so a changed library cannot yield PASS/FAIL.
    prov_end = provenance()
    unchanged = prov_end == prov_start and run_status != "ABORTED_PROVENANCE_CHANGED"
    if prov_end != prov_start and not run_status.startswith("ABORTED"):
        run_status = "ABORTED_PROVENANCE_CHANGED"
    prereg_cfg = (seeds == PREREG_SEEDS and args.n_trials == PREREG_N_TRIALS
                  and names == PREREG_CIRCUITS)
    per_circuit = {}
    for name in names:
        c = circuits[name]
        rows = c.get("per_seed", [])
        wins = int(sum(r["tool_ge_best_baseline"] for r in rows)) if rows else None
        if not unchanged:
            verdict = "INVALID (provenance changed)"
        elif c["status"] != "COMPLETE":
            verdict = f"UNDETERMINED ({c['status'].lower()})"
        elif not prereg_cfg:
            verdict = "NOT A PRE-REGISTERED RESULT"
        else:
            verdict = "PASS" if wins >= PASS_WINS else "FAIL"
        per_circuit[name] = {"wins": wins, "n_run": len(rows), "n_seeds": len(seeds),
                             "verdict": verdict}
    v = [per_circuit[n]["verdict"] for n in names]
    if not unchanged:
        overall = "INVALID"
        reason = "git HEAD, fhe_oracle/core.py or this script changed during the run."
    elif not prereg_cfg:
        overall = "NOT A STUDY RESULT"
        reason = ("The configuration (seeds, n_trials or circuits) differs from the "
                  "pre-registration.")
    elif "PASS" in v:
        overall = "PASS"
        reason = ("The pre-registered criterion requires one passing circuit: "
                  + ", ".join(n for n in names if per_circuit[n]["verdict"] == "PASS") + ".")
    elif all(x == "FAIL" for x in v):
        overall = "FAIL"
        reason = ("The tool did not reach 8 of 10 seeds on either circuit. Pre-registered "
                  "decision: state in the README that the tool does not outperform simple "
                  "sampling at locating CKKS execution error.")
    else:
        overall = "UNDETERMINED"
        reason = ("No completed circuit passed, and not every circuit was completed, so the "
                  "criterion (≥ 8 of 10 on at least one circuit) can still be met by an unrun "
                  "circuit. The README decision (fail on both circuits) is not triggered.")

    choices = [
        "lr_d8: `build_tenseal_lr_d8`, domain [-3, 3]^8, default TenSEAL context; surrogate T3.",
        "Chebyshev circuit: the only circuit defined in `cheb15_cross_circuit.py` (WDBC d = 30 "
        "logistic regression, Cheb-15 sigmoid fitted on [-5, 5], scale 2^30, 16 interior 30-bit "
        "primes, N = 32768, domain [-0.3, 0.3]^30), not the d = 8 variant in "
        "`cheb15_tight_domain.py`.",
        "\"T3\" in the fitness is read as each circuit's own surrogate polynomial (P15 for the "
        "Chebyshev circuit).",
        f"Threshold {THRESHOLD} is passed to `run()`; it sets only the verdict, not the search.",
        "Sobol uses the first n points of a scrambled 2^m block; corner order is a seeded draw "
        "without replacement from the full pool (d = 30 pool not enumerated).",
        "Win comparison is exact (≥); CKKS noise is unseeded, so near-ties are decided by noise.",
        "Holm is applied across the comparisons actually run and, separately, against the planned "
        "family of 6 with unrun comparisons at p = 1.",
        f"Compute cap {args.max_minutes:.0f} min for the invocation, decided from a pre-run timing "
        "probe (uncounted evaluations at fixed uniform points, default_rng(12345)).",
    ]
    caveats = []
    for name in names:
        if circuits[name]["status"] == "BLOCKED":
            p = circuits[name]["preflight"]
            caveats.append(
                f"**Deviation: {name} not run.** Projected {p['projected_s'] / 60:.0f} min at "
                f"{p['median_eval_s']:.2f} s per CKKS evaluation, over the "
                f"{args.max_minutes:.0f}-min cap. The study verdict therefore rests on the "
                "completed circuit(s) only; nothing else was reduced.")
    if run_status != "COMPLETE":
        errs = "; ".join(f"{n}: {circuits[n]['error']}" for n in names if circuits[n].get("error"))
        caveats.append(f"**Run stopped early: {run_status}.**" + (f" {errs}" if errs else ""))
    caveats += [
        "One fixed circuit per row, one key set, one machine; seeds vary only search and sampling "
        "randomness plus unseeded CKKS encryption noise.",
        "CKKS encryption randomness is not seeded (`TenSEALContext` ignores its seed), so a "
        "rerun differs in the reported maxima; the re-encryption columns show the noise scale at "
        "each witness.",
        "Execution error is a random quantity at a fixed input. A method's maximum over ~200 "
        "evaluations mixes where it evaluates with luck in the noise draws; compare the maximum "
        "with the re-encrypted mean at the same witness.",
        "Tool arm uses library defaults only (σ0 = 1.0); `AutoOracle`, restarts and random-floor "
        "options were not evaluated.",
        "The machine was shared with other jobs (see load averages); timings are indicative.",
        "Not independently replicated; produced by the tool author's own script.",
    ]

    runtime = time.perf_counter() - t_start
    out_rel = _repo_rel(os.path.abspath(args.out))
    R = {
        "study": "C. Search that targets CKKS execution error",
        "preregistration": {"path": PREREG_REL, "commit": PREREG_COMMIT,
                            "sha256": _sha256(PREREG_REL)},
        "config": {"circuits": names, "seeds": seeds, "n_trials": args.n_trials,
                   "threshold": THRESHOLD, "repeats": args.repeats, "ref_k": args.ref_k,
                   "timing_evals": args.timing_evals, "max_minutes": args.max_minutes,
                   "pass_wins": PASS_WINS, "preregistered_configuration": prereg_cfg,
                   "tool_kwargs": {"seed": "s", "all_other": "library defaults"}},
        "choices": choices, "caveats": caveats,
        "run_status": run_status, "verdicts": {"per_circuit": per_circuit, "overall": overall,
                                               "overall_reason": reason},
        "circuits": {n: {k: v for k, v in circuits[n].items()
                         if k not in ("fhe", "surrogate", "ctx")} for n in names},
        "statistics": stats,
        "provenance": {"start": prov_start, "end": prov_end, "unchanged": unchanged,
                       "script_tracked": _git("ls-files", "--error-unmatch", SCRIPT_REL) is not None,
                       "tracked_changes_at_end": _git("status", "--porcelain",
                                                      "--untracked-files=no"),
                       "modules": module_table()},
        "environment": environment(),
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "loadavg_start": load_start, "loadavg_end": list(os.getloadavg()),
        "runtime_s": runtime,
        "reproduce_command": (f"python {SCRIPT_REL} --circuits {','.join(names)} --seeds "
                              f"{args.seeds} --n-trials {args.n_trials} --repeats {args.repeats} "
                              f"--ref-k {args.ref_k} --max-minutes {args.max_minutes:g} "
                              f"--out {out_rel}"),
    }
    os.makedirs(args.out, exist_ok=True)
    write_csv(os.path.join(args.out, "per_seed.csv"), R["circuits"])
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(_clean(R), fh, indent=2)
    code = {"ABORTED_PROVENANCE_CHANGED": 3, "ABORTED_COMPUTE_CAP": 4,
            "ABORTED_ERROR": 5}.get(run_status, 0)
    try:
        text = build_markdown(R)
    except Exception as exc:  # CSV and JSON are already written; keep the failure visible
        text = (f"# Study C\n\nReport rendering failed ({type(exc).__name__}: {exc}); "
                "see summary.json and per_seed.csv.\n")
        print(f"report.md rendering failed: {type(exc).__name__}: {exc}")
        code = code or 6
    with open(os.path.join(args.out, "report.md"), "w") as fh:
        fh.write(text)
    print(f"status {run_status}; overall {overall}; wrote {out_rel} in {runtime:.0f}s; exit {code}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
