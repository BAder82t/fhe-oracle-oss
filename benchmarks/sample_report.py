# Copyright (C) 2026 Bader Alissaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Sample precision report: FHEOracle and AutoOracle vs random and corner baselines on real TenSEAL CKKS.

Writes report.md, report.json and per_seed.csv into --out.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import importlib.metadata as md
import inspect
import itertools
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from collections import Counter
from datetime import datetime, timezone

import numpy as np
from scipy.stats import wilcoxon

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, THIS_DIR)

from fhe_oracle import FHEOracle, registry  # noqa: E402
from fhe_oracle import __version__ as FHE_ORACLE_VERSION  # noqa: E402
from fhe_oracle.adapters import tenseal_adapter as tsa  # noqa: E402
from fhe_oracle import autoconfig as _autoconfig  # noqa: E402
from fhe_oracle.autoconfig import AutoOracle  # noqa: E402
from fhe_oracle.evallog import JsonlEvaluationLog  # noqa: E402
from fhe_oracle.fitness import absolute_error  # noqa: E402
from fhe_oracle.report import to_json, to_markdown  # noqa: E402
from tenseal_circuits import build_tenseal_lr_d8  # noqa: E402

warnings.filterwarnings("ignore", message="Could not import matplotlib")

SCRIPT_REL = "benchmarks/sample_report.py"
LIB_FILES = ("fhe_oracle/core.py", "fhe_oracle/autoconfig.py")
AUTO_N_PROBES = inspect.signature(AutoOracle.__init__).parameters["n_probes"].default
# AutoOracle needs n_trials >= n_probes + reserve without W, b; 3 if the library has no constant.
AUTO_RESERVE = int(getattr(_autoconfig, "_RESERVE", 3))
# Paired per-seed differences within this are ties (CKKS noise at a fixed input).
TIE_TOL = 1e-5
TOOL_NAMES = {"oracle": "FHEOracle", "auto": "AutoOracle"}
METHOD_LABELS = {
    "oracle": "FHEOracle", "auto": "AutoOracle",
    "random_eqeval": "uniform random, equal evaluations",
    "random_eqtime": "uniform random, equal wall-clock",
    "corner_eqeval": "corner set, equal evaluations",
    "corner_eqtime": "corner set, equal wall-clock",
    "random_auto_eqeval": "uniform random, AutoOracle's evaluation count",
    "corner_auto_eqeval": "corner set, AutoOracle's evaluation count",
}
BASELINE_CUTS = ("random_eqeval", "random_eqtime", "corner_eqeval", "corner_eqtime",
                 "random_auto_eqeval", "corner_auto_eqeval")
# (summary key, tool, reference method, label, budget basis)
COMPARISONS = [
    ("random_eqeval", "oracle", "random_eqeval", "FHEOracle vs uniform random",
     "equal evaluations"),
    ("random_eqtime", "oracle", "random_eqtime", "FHEOracle vs uniform random",
     "equal wall-clock"),
    ("corner_eqeval", "oracle", "corner_eqeval", "FHEOracle vs corner/boundary set",
     "equal evaluations"),
    ("corner_eqtime", "oracle", "corner_eqtime", "FHEOracle vs corner/boundary set",
     "equal wall-clock"),
    ("auto_vs_random_eqeval", "auto", "random_auto_eqeval", "AutoOracle vs uniform random",
     "equal evaluations"),
    ("auto_vs_corner_eqeval", "auto", "corner_auto_eqeval",
     "AutoOracle vs corner/boundary set", "equal evaluations"),
    ("auto_vs_fheoracle", "auto", "oracle", "AutoOracle vs FHEOracle defaults",
     "same `n_trials` (counted evaluations differ)"),
]
STRATEGY_TEXT = {
    "cma_es": "default CMA-ES, the same settings as the FHEOracle column",
    "robust_cma_es": "CMA-ES with σ0 = mean box width / 4 and 10 injected heuristic seeds",
    "warm_start": "30% uniform random sampling, then CMA-ES warm-started at the best sample",
    "random_only": "uniform random sampling for the whole search budget",
    "separable_cma_es": "diagonal-covariance CMA-ES",
}


# --- Circuit instrumentation -------------------------------------------------

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def taylor3(z):
    # Same formula the CKKS circuit evaluates: 0.5 + z/4 - z^3/48.
    return 0.5 + z * 0.25 - z ** 3 * (1.0 / 48.0)


class CountingCircuit:
    """Counts every model call and logs the error decomposition of each FHE call."""

    def __init__(self, circuit):
        self.w = np.asarray(circuit["weights"], dtype=np.float64)
        self.b = float(circuit["bias"])
        self._plain = circuit["plain"]
        self._fhe = circuit["fhe"]
        self.lo = np.array([lo for lo, _ in circuit["bounds"]])
        self.hi = np.array([hi for _, hi in circuit["bounds"]])
        self.reset()

    def reset(self):
        self.n_plain = 0
        self.n_fhe = 0
        self.n_oob = 0
        self.records: list[dict] = []

    def plain(self, x):
        self.n_plain += 1
        return self._plain(x)

    def fhe(self, x):
        xa = np.asarray(x, dtype=np.float64)
        if np.any(xa < self.lo) or np.any(xa > self.hi):
            self.n_oob += 1
        t0 = time.perf_counter()
        y = self._fhe(x)
        dt = time.perf_counter() - t0
        self.n_fhe += 1
        # Instrumentation only: plaintext recomputed for logging, not counted.
        z = float(np.dot(self.w, xa) + self.b)
        model = float(self._plain(x))
        surr = float(taylor3(z))
        fhe_val = float(y)
        self.records.append({
            "x": xa.tolist(), "z": z, "model": model, "surrogate": surr,
            "fhe": fhe_val, "total": abs(model - fhe_val),
            "approx": abs(model - surr), "ckks": abs(surr - fhe_val), "t": dt,
        })
        return y


def summarize(records, n, threshold):
    sub = records[:n]
    totals = np.array([r["total"] for r in sub])
    comps = np.array([[r["approx"], r["ckks"]] for r in sub])
    if not (np.all(np.isfinite(totals)) and np.all(np.isfinite(comps))):
        raise RuntimeError("non-finite error value in logged evaluations")
    i = int(np.argmax(totals))
    return {
        "fhe_evals": n,
        "plain_evals": n,
        "max_error": float(totals[i]),
        "witness": sub[i],
        "frac_ge_threshold": float(np.mean(totals >= threshold)),
        "max_ckks_component": float(max(r["ckks"] for r in sub)),
        "fhe_time_s": float(sum(r["t"] for r in sub)),
        "median_fhe_eval_ms": float(1e3 * np.median([r["t"] for r in sub])),
    }


def _tol(v):
    return 1e-12 * max(1.0, abs(v))


class EventTap:
    """on_evaluation sink for one tool run: JSONL log plus in-memory events; closed after run()."""

    def __init__(self, path):
        self._log = JsonlEvaluationLog(path)
        self.events: list[dict] = []
        self.sha256 = None

    def __call__(self, event):
        if self.sha256 is None:  # later events (e.g. shrink) are not part of this run
            self._log(event)
            self.events.append(event)

    def close(self):
        self.sha256 = self._log.sha256()
        self._log.close()
        return self.sha256

    def kinds(self):
        return dict(sorted(Counter(e["kind"] for e in self.events).items()))


def _check_events(tap, recs, label):
    """Logged events must match the FHE calls one-to-one, in order, by input and error."""
    if [e["index"] for e in tap.events] != list(range(len(recs))):
        raise RuntimeError(f"{label} emitted {len(tap.events)} events for {len(recs)} FHE calls")
    for e, r in zip(tap.events, recs):
        v = e["error"] if e["kind"] == "remeasure" else e["score"]
        if not np.allclose(e["x"], r["x"], rtol=0, atol=1e-12) or abs(v - r["total"]) > _tol(v):
            raise RuntimeError(f"{label} event {e['index']} ({e['kind']}) does not match its FHE call")


def _check_remeasurement(cc, res, label):
    """Match remeasured_error, search_max_error and max_error to the logged evaluations."""
    if cc.n_oob:
        raise RuntimeError(f"{label} evaluated {cc.n_oob} out-of-domain inputs")
    if cc.n_plain != cc.n_fhe:
        raise RuntimeError(f"{label} plaintext and FHE call counts differ")
    rem, smax = getattr(res, "remeasured_error", None), getattr(res, "search_max_error", None)
    if rem is None or smax is None:
        raise RuntimeError(f"{label} result lacks remeasured_error/search_max_error")
    last, search = cc.records[-1], cc.records[:-1]
    # run() ends with one re-measurement at worst_input.
    if (not np.allclose(last["x"], res.worst_input, rtol=0, atol=1e-12)
            or abs(last["total"] - rem) > _tol(rem)):
        raise RuntimeError(f"{label} remeasured_error does not match its last logged evaluation")
    at_witness = [r for r in search if np.allclose(r["x"], res.worst_input, rtol=0, atol=1e-12)
                  and abs(r["total"] - smax) <= _tol(smax)]
    if not at_witness or abs(max(r["total"] for r in search) - smax) > _tol(smax):
        raise RuntimeError(f"{label} search_max_error does not match the logged search evaluations")
    if abs(res.max_error - max(rem, smax)) > _tol(res.max_error):
        raise RuntimeError(f"{label} max_error is not max(remeasured_error, search_max_error)")
    if res.verdict != ("FAIL" if res.max_error >= res.threshold else "PASS"):
        raise RuntimeError(f"{label} verdict inconsistent with max_error and threshold")
    return last, at_witness[-1]


# --- Methods -----------------------------------------------------------------

def run_oracle(cc, circuit, seed, budget, threshold, log_dir):
    cc.reset()
    name = f"fheoracle_seed{seed}.jsonl"
    tap = EventTap(os.path.join(log_dir, name))
    t0 = time.perf_counter()
    oracle = FHEOracle(
        plaintext_fn=cc.plain, fhe_fn=cc.fhe, input_dim=circuit["d"],
        input_bounds=circuit["bounds"], seed=seed, on_evaluation=tap,
    )
    res = oracle.run(n_trials=budget, threshold=threshold)
    wall = time.perf_counter() - t0
    sha = tap.close()
    last, search_rec = _check_remeasurement(cc, res, "FHEOracle")
    # Library contract: at most n_trials search evaluations plus one re-measurement.
    if res.n_trials > budget or cc.n_fhe != res.n_trials + 1:
        raise RuntimeError(
            f"FHEOracle made {cc.n_fhe} FHE calls for n_trials={res.n_trials} (budget {budget})")
    recs = list(cc.records)
    _check_events(tap, recs, "FHEOracle")
    kinds = tap.kinds()
    if kinds != {"remeasure": 1, "search": res.n_trials}:
        raise RuntimeError(f"FHEOracle event kinds {kinds} for n_trials={res.n_trials}")
    s = summarize(recs, len(recs), threshold)
    s.update({
        "plain_evals": cc.n_plain,
        "wall_s": wall,
        "n_trials_reported": res.n_trials,
        "reported_max_error": res.max_error,
        "search_max_error": res.search_max_error,
        "remeasured_error": res.remeasured_error,
        "remeasure_confirms": bool(res.remeasured_error >= threshold),
        "unconfirmed_fail": res.verdict == "FAIL" and res.remeasured_error < threshold,
        "verdict": res.verdict,
        "scheme": res.scheme,
        "overhead_frac": (wall - s["fhe_time_s"]) / wall,
        "reported_record": last,
        "search_record": search_rec,
        "worst_input": list(res.worst_input),
        "log": {"path": f"evaluation_logs/{name}", "sha256": sha, "kinds": kinds},
    })
    return oracle, res, s


def run_auto(cc, circuit, seed, budget, threshold, log_dir):
    """AutoOracle with defaults; every evaluation is counted and logged."""
    cc.reset()
    name = f"autooracle_seed{seed}.jsonl"
    tap = EventTap(os.path.join(log_dir, name))
    t0 = time.perf_counter()
    auto = AutoOracle(cc.plain, cc.fhe, circuit["bounds"], on_evaluation=tap)
    res = auto.run(n_trials=budget, threshold=threshold, seed=seed)
    wall = time.perf_counter() - t0
    sha = tap.close()
    strategy = getattr(res, "strategy_used", None)
    if auto.last_oracle is None:
        raise RuntimeError(f"AutoOracle dispatched to {strategy!r}, which this report does not handle")
    if cc.n_oob:
        raise RuntimeError(f"AutoOracle evaluated {cc.n_oob} out-of-domain inputs")
    if cc.n_plain != cc.n_fhe:
        raise RuntimeError("AutoOracle plaintext and FHE call counts differ")
    recs = list(cc.records)
    _check_events(tap, recs, "AutoOracle")
    rem, smax = getattr(res, "remeasured_error", None), getattr(res, "search_max_error", None)
    if rem is None or smax is None:
        raise RuntimeError("AutoOracle result lacks remeasured_error/search_max_error")
    kinds = tap.kinds()
    n_rem = kinds.get("remeasure", 0)
    # n_trials counts every evaluation except re-measurements; all calls fit in n_trials.
    if cc.n_fhe > budget or res.n_trials != cc.n_fhe - n_rem:
        raise RuntimeError(f"AutoOracle made {cc.n_fhe} FHE calls with {n_rem} re-measurements "
                           f"for n_trials={res.n_trials} (budget {budget})")
    # One re-measurement after the inner search, plus one when a probe or boundary witness wins.
    counted = [e for e in tap.events if e["kind"] != "remeasure"
               and np.allclose(e["x"], res.worst_input, rtol=0, atol=1e-12)
               and abs(e["score"] - smax) <= _tol(smax)]
    if not counted:
        raise RuntimeError("AutoOracle search_max_error matches no counted evaluation at worst_input")
    witness_kind = counted[-1]["kind"]
    if n_rem != (1 if witness_kind == "search" else 2):
        raise RuntimeError(f"AutoOracle made {n_rem} re-measurements for a {witness_kind} witness")
    last_rm = [e for e in tap.events if e["kind"] == "remeasure"][-1]
    if (not np.allclose(last_rm["x"], res.worst_input, rtol=0, atol=1e-12)
            or abs(last_rm["error"] - rem) > _tol(rem)):
        raise RuntimeError("AutoOracle remeasured_error is not its last re-measurement at worst_input")
    if abs(res.max_error - max(rem, smax)) > _tol(res.max_error):
        raise RuntimeError("AutoOracle max_error is not max(remeasured_error, search_max_error)")
    if res.verdict != ("FAIL" if res.max_error >= res.threshold else "PASS"):
        raise RuntimeError("AutoOracle verdict inconsistent with max_error and threshold")
    # Displayed witness: the logged call at worst_input whose error equals max_error.
    idx = [i for i in range(len(recs) - 1, -1, -1)
           if np.allclose(recs[i]["x"], res.worst_input, rtol=0, atol=1e-12)
           and abs(recs[i]["total"] - res.max_error) <= _tol(res.max_error)]
    if not idx:
        raise RuntimeError("AutoOracle max_error does not match any logged evaluation")
    remeas, rem_at_witness = n_rem, True
    s = summarize(recs, len(recs), threshold)
    s.update({
        "plain_evals": cc.n_plain,
        "wall_s": wall,
        "n_trials_reported": res.n_trials,
        "reported_max_error": res.max_error,
        "verdict": res.verdict,
        "scheme": res.scheme,
        "regime": getattr(res, "regime", None),
        "strategy": strategy,
        "probe_reason": auto.probe_result.recommendation.get("reason"),
        "n_probes_configured": auto.n_probes,
        "probe_evals_reported": getattr(auto.probe_result, "n_evals", None),
        "probe_divergences_len": int(len(auto.probe_result.probe_divergences)),
        "remeasurements": remeas,
        "search_max_error": smax,
        "remeasured_error": rem,
        "remeasured_at_witness": rem_at_witness,
        "remeasure_confirms": bool(rem_at_witness and rem >= threshold),
        "unconfirmed_fail": res.verdict == "FAIL" and not (rem_at_witness and rem >= threshold),
        "reported_record_index": idx[0],
        "reported_is_final_call": idx[0] == len(recs) - 1,
        "overhead_frac": (wall - s["fhe_time_s"]) / wall,
        "reported_record": recs[idx[0]],
        "witness_kind": witness_kind,
        "worst_input": list(res.worst_input),
        "log": {"path": f"evaluation_logs/{name}", "sha256": sha, "kinds": kinds},
    })
    return auto, res, s


def run_stream(cc, make_points, eval_cuts, t_target, threshold, cap):
    """Evaluate points in order; summarise at each evaluation cut and at the time target."""
    cc.reset()
    done: list[float] = []
    exhausted = True
    target = max(eval_cuts.values())
    t0 = time.perf_counter()
    for x in make_points():
        n = len(done)
        if n >= cap or (n >= target and time.perf_counter() - t0 >= t_target):
            exhausted = False
            break
        p = cc.plain(x)
        f = cc.fhe(x)
        absolute_error(p, f)  # raises EvaluationError on invalid output
        done.append(time.perf_counter() - t0)
    if cc.n_oob:
        raise RuntimeError(f"baseline evaluated {cc.n_oob} out-of-domain inputs")
    recs = list(cc.records)
    out = {}
    for name, k in eval_cuts.items():
        n_eval = min(k, len(done))
        s = summarize(recs, n_eval, threshold)
        s["wall_s"] = done[n_eval - 1]
        s["pool_exhausted"] = exhausted and len(done) < k
        out[name] = s
    n_time = int(np.searchsorted(done, t_target, side="right"))
    if n_time == 0:
        raise RuntimeError("baseline completed no evaluation within the time budget")
    s = summarize(recs, n_time, threshold)
    s["wall_s"] = done[n_time - 1]
    s["pool_exhausted"] = exhausted and done[-1] < t_target
    out["eqtime"] = s
    return out


def random_points(lo, hi, seed):
    def gen():
        rng = np.random.default_rng(seed)
        while True:
            yield rng.uniform(lo, hi).tolist()
    return gen


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


def corner_points(pool, seed):
    def gen():
        order = np.random.default_rng([seed, 1]).permutation(len(pool))
        for i in order:
            yield pool[i].tolist()
    return gen


# --- Plaintext references ----------------------------------------------------

def approx_supremum(w, b, bounds, n_grid=2_000_001):
    """Max of |sigmoid - Taylor3| over the box, via the exact z-range of w.x+b."""
    zmin = b + sum(min(wi * lo, wi * hi) for wi, (lo, hi) in zip(w, bounds))
    zmax = b + sum(max(wi * lo, wi * hi) for wi, (lo, hi) in zip(w, bounds))
    z = np.linspace(zmin, zmax, n_grid)
    err = np.abs(sigmoid(z) - taylor3(z))
    i = int(np.argmax(err))
    out = {"z_min": zmin, "z_max": zmax, "sup": float(err[i]), "z_at_sup": float(z[i])}
    if i in (0, n_grid - 1):
        top = i == n_grid - 1
        out["x_at_sup"] = [
            (hi if (wi > 0) == top else lo) for wi, (lo, hi) in zip(w, bounds)
        ]
        out["at_vertex"] = True
    else:
        out["at_vertex"] = False
    return out


def violating_fraction(w, b, bounds, threshold, n=1_000_000, seed=0):
    rng = np.random.default_rng(seed)
    lo = np.array([a for a, _ in bounds])
    hi = np.array([c for _, c in bounds])
    hits = 0
    for start in range(0, n, 200_000):
        m = min(200_000, n - start)
        z = rng.uniform(lo, hi, size=(m, len(bounds))) @ w + b
        hits += int(np.sum(np.abs(sigmoid(z) - taylor3(z)) >= threshold))
    p = hits / n
    zc = 1.959964
    den = 1 + zc ** 2 / n
    centre = (p + zc ** 2 / (2 * n)) / den
    half = zc * np.sqrt(p * (1 - p) / n + zc ** 2 / (4 * n ** 2)) / den
    return {"n_samples": n, "seed": seed, "fraction": p,
            "wilson95": [centre - half, centre + half]}


# --- Statistics --------------------------------------------------------------

def compare(o, b):
    o = np.asarray(o, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if not (np.all(np.isfinite(o)) and np.all(np.isfinite(b)) and np.all(b > 0)):
        raise RuntimeError("max errors must be finite with positive references to form ratios")
    ratios = o / b
    diff = o - b
    tie = np.abs(diff) <= TIE_TOL
    n_zero = int(np.sum(tie))
    if n_zero == o.size:
        p, method = 1.0, f"all differences within TIE_TOL={TIE_TOL:g}"
    else:
        r = wilcoxon(np.where(tie, 0.0, diff), alternative="two-sided", zero_method="wilcox")
        p = float(r.pvalue)
        method = (f"scipy.stats.wilcoxon two-sided on differences with |d| <= {TIE_TOL:g} set to "
                  f"0, zero_method='wilcox' ({n_zero} tied pairs dropped)")
    return {
        "n": int(o.size),
        "median_tool": float(np.median(o)),
        "median_reference": float(np.median(b)),
        "median_ratio": float(np.median(ratios)),
        "q1_ratio": float(np.percentile(ratios, 25)),
        "q3_ratio": float(np.percentile(ratios, 75)),
        "min_ratio": float(np.min(ratios)),
        "max_ratio": float(np.max(ratios)),
        "tool_larger": int(np.sum(diff > TIE_TOL)),
        "tool_smaller": int(np.sum(diff < -TIE_TOL)),
        "ties": n_zero,
        "tie_tol": TIE_TOL,
        "n_used_in_test": int(o.size - n_zero),
        "p_value": p,
        "test": method,
    }


def build_comparisons(rows, sup, bounds):
    """Paired comparisons with ties and Holm adjustment; corner references get worst-vertex coverage."""
    out = {"tie_tol": TIE_TOL, "holm_family_size": len(COMPARISONS)}
    pool = corner_pool(bounds)
    vx = sup.get("x_at_sup")
    vi = next((i for i, p in enumerate(pool) if vx is not None and np.allclose(p, vx)), None)
    for key, tool, ref_m, label, basis in COMPARISONS:
        s = compare([r[tool]["max_error"] for r in rows], [r[ref_m]["max_error"] for r in rows])
        s.update({"tool": tool, "reference": ref_m, "label": label, "basis": basis})
        if ref_m.startswith("corner") and vi is not None:
            # Same seeded order as corner_points(): did this cut evaluate the worst vertex?
            missed = [r["seed"] for r in rows if vi not in np.random.default_rng(
                [r["seed"], 1]).permutation(len(pool))[:r[ref_m]["fhe_evals"]]]
            larger = [r["seed"] for r in rows
                      if r[tool]["max_error"] - r[ref_m]["max_error"] > TIE_TOL]
            s.update({"reference_missed_worst_vertex_seeds": missed, "tool_larger_seeds": larger,
                      "tool_larger_where_reference_missed": len(set(larger) & set(missed))})
        out[key] = s
    for (key, *_), adj in zip(COMPARISONS, holm([out[k]["p_value"] for k, *_ in COMPARISONS])):
        out[key]["p_holm"] = adj
    return out


def holm(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


# --- Metadata ----------------------------------------------------------------

def _run(cmd):
    try:
        return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return None


def _ver(pkg):
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


def _repo_rel(arg):
    """Repo-relative form of an absolute path argument; no home paths in artifacts."""
    prefix, sep, val = arg.partition("=") if arg.startswith("--") else ("", "", arg)
    if not os.path.isabs(val):
        return arg
    rel = os.path.relpath(val, ROOT)
    rel = f"<outside-repo>/{os.path.basename(val)}" if rel.startswith("..") else rel
    return f"{prefix}{sep}{rel}"


def _sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def collect_metadata():
    cpu = None
    if platform.system() == "Darwin":
        cpu = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    elif os.path.exists("/proc/cpuinfo"):
        with open("/proc/cpuinfo") as fh:
            cpu = next((ln.split(":", 1)[1].strip() for ln in fh
                        if ln.startswith("model name")), None)
    script_sha = _sha256(os.path.join(ROOT, SCRIPT_REL))
    tracked = _run(["git", "ls-files", "--error-unmatch", SCRIPT_REL]) is not None
    return {
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fhe_oracle_version": FHE_ORACLE_VERSION,
        "git_sha": _run(["git", "rev-parse", "HEAD"]),
        "git_tracked_changes": _run(["git", "status", "--porcelain", "--untracked-files=no"]),
        "script_tracked_in_git": tracked,
        "script_sha256": script_sha,
        "library_sha256": {p: _sha256(os.path.join(ROOT, p)) for p in LIB_FILES},
        "packages": {p: _ver(p) for p in ("fhe-oracle", "tenseal", "cma", "numpy", "scipy",
                                          "fhe-oracle-pro")},
        "registered_plugins": {"fitness": registry.list_fitness(),
                               "heuristics": registry.list_heuristics()},
        "python": sys.version.split()[0],
        "python_executable": os.path.basename(sys.executable),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": cpu or platform.processor(),
        "cpu_count": os.cpu_count(),
        "argv": [_repo_rel(a) for a in sys.argv],
    }


# --- Output ------------------------------------------------------------------

def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        if not np.isfinite(v):
            raise ValueError("non-finite value in report data")
        return v
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _vec(x, nd=3):
    # Values that round to zero print as 0.000, never -0.000.
    return "[" + ", ".join(f"{(0.0 if abs(v) < 0.5 * 10 ** -nd else v):.{nd}f}" for v in x) + "]"


def _med_range(vals, fmt="{:.0f}"):
    vals = np.asarray(vals)
    lo, med, hi = np.min(vals), np.median(vals), np.max(vals)
    if lo == hi:
        return fmt.format(med)
    return f"{fmt.format(med)} ({fmt.format(lo)}–{fmt.format(hi)})"


def write_csv(path, rows):
    methods = ["oracle", "auto", *BASELINE_CUTS]
    fields = ["seed", "oracle_reported_max_error", "oracle_n_trials_reported",
              "oracle_verdict", "oracle_overhead_frac", "auto_regime", "auto_strategy",
              "auto_reported_max_error", "auto_n_trials_reported", "auto_remeasurements",
              "auto_probe_evals_reported",
              "auto_reported_is_final_call", "auto_verdict", "oracle_search_max_error",
              "oracle_remeasured_error", "oracle_remeasure_confirms", "auto_search_max_error",
              "auto_remeasured_error", "auto_remeasured_at_witness", "auto_remeasure_confirms",
              "auto_witness_kind", "oracle_log_sha256", "auto_log_sha256",
              "oracle_events_search", "oracle_events_remeasure", "auto_events_probe",
              "auto_events_structure", "auto_events_boundary", "auto_events_search",
              "auto_events_remeasure"]
    for m in methods:
        fields += [f"{m}_max_error", f"{m}_fhe_evals", f"{m}_plain_evals", f"{m}_wall_s",
                   f"{m}_median_fhe_eval_ms", f"{m}_frac_ge_threshold",
                   f"{m}_max_ckks_component", f"{m}_witness_approx", f"{m}_witness_ckks"]
    fields += [f"ratio_{tool}_over_{ref}" for _, tool, ref, _, _ in COMPARISONS]
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        for r in rows:
            o, au = r["oracle"], r["auto"]
            row = {
                "seed": r["seed"],
                "oracle_reported_max_error": o["reported_max_error"],
                "oracle_n_trials_reported": o["n_trials_reported"],
                "oracle_verdict": o["verdict"],
                "oracle_overhead_frac": o["overhead_frac"],
                "auto_regime": au["regime"], "auto_strategy": au["strategy"],
                "auto_reported_max_error": au["reported_max_error"],
                "auto_n_trials_reported": au["n_trials_reported"],
                "auto_remeasurements": au["remeasurements"],
                "auto_probe_evals_reported": au["probe_evals_reported"],
                "auto_reported_is_final_call": au["reported_is_final_call"],
                "auto_verdict": au["verdict"],
                "oracle_search_max_error": o["search_max_error"],
                "oracle_remeasured_error": o["remeasured_error"],
                "oracle_remeasure_confirms": o["remeasure_confirms"],
                "auto_search_max_error": au["search_max_error"],
                "auto_remeasured_error": au["remeasured_error"],
                "auto_remeasured_at_witness": au["remeasured_at_witness"],
                "auto_remeasure_confirms": au["remeasure_confirms"],
                "auto_witness_kind": au["witness_kind"],
                "oracle_log_sha256": o["log"]["sha256"],
                "auto_log_sha256": au["log"]["sha256"],
            }
            for tool, tk in (("oracle", ("search", "remeasure")),
                             ("auto", ("probe", "structure", "boundary", "search", "remeasure"))):
                for kind in tk:
                    row[f"{tool}_events_{kind}"] = r[tool]["log"]["kinds"].get(kind, 0)
            for m in methods:
                s = r[m]
                row.update({
                    f"{m}_max_error": s["max_error"], f"{m}_fhe_evals": s["fhe_evals"],
                    f"{m}_plain_evals": s["plain_evals"], f"{m}_wall_s": s["wall_s"],
                    f"{m}_median_fhe_eval_ms": s["median_fhe_eval_ms"],
                    f"{m}_frac_ge_threshold": s["frac_ge_threshold"],
                    f"{m}_max_ckks_component": s["max_ckks_component"],
                    f"{m}_witness_approx": s["witness"]["approx"],
                    f"{m}_witness_ckks": s["witness"]["ckks"],
                })
            for _, tool, ref, _, _ in COMPARISONS:
                row[f"ratio_{tool}_over_{ref}"] = r[tool]["max_error"] / r[ref]["max_error"]
            wr.writerow(row)


def witness_row(label, rec):
    return (f"| {label} | `{_vec(rec['x'])}` | {rec['z']:.4f} | {rec['model']:.8f} | "
            f"{rec['surrogate']:.8f} | {rec['fhe']:.8f} | {rec['total']:.9g} | "
            f"{rec['approx']:.9g} | {rec['ckks']:.3e} |")


def build_markdown(R):
    cfg, meta, ref = R["config"], R["metadata"], R["references"]
    rows, summ = R["per_seed"], R["summary"]
    n, B, thr = cfg["n_seeds"], cfg["budget"], cfg["threshold"]
    ow = R["witnesses"]["oracle_worst"]
    au_rows = [r["auto"] for r in rows]
    n_tests = len(COMPARISONS)
    # Largest spread between repeated measurements of one input in this run.
    spreads = [abs(r[t]["search_max_error"] - r[t]["remeasured_error"])
               for r in rows for t in ("oracle", "auto")]
    spreads.append(R["witnesses"]["repeat_encryptions"]["total_spread"])
    if R["shrink"].get("repeat_encryptions"):
        spreads.append(R["shrink"]["repeat_encryptions"]["total_max"]
                       - R["shrink"]["repeat_encryptions"]["total_min"])
    spread = max(spreads)
    L = []
    a = L.append

    a("# FHE Oracle sample precision report")
    a("")
    a("Logistic regression (d = 8, Taylor-3 sigmoid) executed under real CKKS via TenSEAL. "
      "Generated by `benchmarks/sample_report.py`; every number below is written by that script.")
    a("")
    a(f"- Run (UTC): {meta['started_utc']} to {meta['date_utc']}  ")
    a(f"- fhe-oracle {meta['fhe_oracle_version']}, git `{meta['git_sha_start']}` at start"
      + ("" if not meta["git_changed_during_run"] else
         f" (HEAD moved to `{meta['git_sha']}` during the run; see Environment)") + "  ")
    a(f"- Seeds: {n} (paired), `n_trials` = {B} for FHEOracle and AutoOracle, threshold: {thr}  ")
    a(f"- Total runtime: {R['runtime_s'] / 60:.1f} min")
    a("")

    # Summary
    a("## Summary")
    a("")
    a("Metric: largest absolute error |intended model(x) − CKKS output(x)| over all counted "
      "evaluations of each method, per seed. Ratios are tool / reference for that seed.")
    a("")
    a("| Comparison | Budget basis | Reference FHE evals, median (range) | Median max error: tool | "
      f"Median max error: reference | Median ratio (IQR) | Tool larger / smaller / tie "
      f"(\\|difference\\| ≤ {TIE_TOL:g}) | Wilcoxon p (ties dropped) | Holm p |")
    a("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for key, tool, ref_m, label, basis in COMPARISONS:
        s = summ[key]
        evals = _med_range([r[ref_m]["fhe_evals"] for r in rows])
        a(f"| {label} | {basis} | {evals} | {s['median_tool']:.4g} | "
          f"{s['median_reference']:.4g} | {s['median_ratio']:.3f} ({s['q1_ratio']:.3f}–"
          f"{s['q3_ratio']:.3f}) | {s['tool_larger']} / {s['tool_smaller']} / {s['ties']} | "
          f"{s['p_value']:.3g} | {s['p_holm']:.3g} |")
    a("")
    a(f"FHEOracle FHE evaluations per seed: {_med_range([r['oracle']['fhe_evals'] for r in rows])} "
      f"({_med_range([r['oracle']['n_trials_reported'] for r in rows])} search evaluations "
      f"reported by `run()` for `n_trials` = {B}, plus one re-measurement of the reported "
      f"witness; checked on every seed). FHEOracle wall-clock per seed: "
      f"{_med_range([r['oracle']['wall_s'] for r in rows], '{:.2f}')} s. Its "
      "`search_max_error` (largest counted search evaluation) and `remeasured_error` (the "
      "re-measurement at `worst_input`) differ by at most "
      f"{summ['oracle_search_vs_remeasured_max_abs_diff']:.2e}; `max_error` is the larger of "
      "the two and equals the largest counted error on every seed (checked). Its evaluation "
      "log has exactly `n_trials` search events and one remeasure event per seed (checked).")
    a("")
    atxt = (f"AutoOracle FHE evaluations per seed: {_med_range([x['fhe_evals'] for x in au_rows])} "
            f"for `n_trials` = {B}: {_med_range([x['n_trials_reported'] for x in au_rows])} "
            "counted evaluations reported as `n_trials` by `run()`, plus "
            f"{_med_range([x['remeasurements'] for x in au_rows])} re-measurement(s); checked on "
            "every seed, with total FHE calls never above `n_trials`.")
    kinds_all = sorted({k for x in au_rows for k in x["log"]["kinds"]})
    atxt += (" Logged events per seed by kind: " + ", ".join(
        f"{k} {_med_range([x['log']['kinds'].get(k, 0) for x in au_rows])}" for k in kinds_all)
        + " (matched one-to-one against the FHE calls).")
    wk = Counter(x["witness_kind"] for x in au_rows)
    atxt += (" The reported witness came from " + "; ".join(
        f"a {k} evaluation on {c}/{n} seeds" for k, c in wk.most_common())
        + ". A search witness is re-measured once; a probe or boundary witness that beats the "
        "search is re-measured once more (checked on every seed).")
    atxt += (" AutoOracle wall-clock per seed: "
             f"{_med_range([x['wall_s'] for x in au_rows], '{:.2f}')} s. Baselines compared with "
             "AutoOracle receive exactly its counted FHE evaluations.")
    nf = sum(not x["reported_is_final_call"] for x in au_rows)
    if nf:
        atxt += (f" On {nf}/{n} seeds AutoOracle's reported witness is an earlier counted "
                 "evaluation rather than its last call; it was matched to the log by input and "
                 "error.")
    nr = sum(not x["remeasured_at_witness"] for x in au_rows)
    if nr:
        atxt += (f" On {nr}/{n} seeds AutoOracle's `remeasured_error` was not measured at its "
                 "reported witness (the witness was replaced after the inner search, or the "
                 "field is absent), so a FAIL there is not confirmed by a re-measurement.")
    a(atxt)
    a("")
    gap = summ["auto_reported_gap"]
    if gap["seeds"]:
        a(f"On {len(gap['seeds'])}/{n} seeds (seeds {gap['seeds']}) AutoOracle's reported "
          f"`max_error` is more than 0.1% below the largest error among its own counted "
          f"evaluations (largest gap {gap['max_gap']:.4g}). The tables use the largest counted "
          "error for every method.")
    else:
        a(f"AutoOracle's reported `max_error` is within 0.1% of the largest error among its "
          f"counted evaluations on every seed (largest absolute gap {gap['max_abs_gap']:.2e}).")
    a("")
    for key, tool, ref_m, label, basis in COMPARISONS:
        s = summ[key]
        det = ("detectable" if s["p_holm"] < 0.05 else "not detectable")
        parts = []
        if s["ties"]:
            parts.append(f"matched within noise (|difference| ≤ {TIE_TOL:g}) on {s['ties']}/{n} seeds")
        if s["tool_larger"]:
            txt = f"larger on {s['tool_larger']}/{n} seeds"
            k = s.get("tool_larger_where_reference_missed")
            if k is not None and k == s["tool_larger"]:
                txt += (", all of them seeds where the corner set's evaluation budget did not "
                        "reach the worst vertex")
            elif k:
                txt += (f", {k} of them seeds where the corner set's evaluation budget did not "
                        "reach the worst vertex")
            parts.append(txt)
        if s["tool_smaller"]:
            parts.append(f"smaller on {s['tool_smaller']}/{n} seeds")
        test = ("Every difference is within noise, so no paired test applies."
                if s["n_used_in_test"] == 0 else
                f"On the {s['n_used_in_test']} non-tied seeds the paired difference is {det} at "
                f"the 0.05 level after Holm adjustment across {n_tests} tests "
                f"(p = {s['p_holm']:.3g}).")
        a(f"- **{label}, {basis}:** {TOOL_NAMES[tool]}'s maximum error "
          + ("was " if s["tool_larger"] or s["tool_smaller"] else "")
          + "; ".join(parts) + f"; median ratio {s['median_ratio']:.3f}. {test}")
    a("")
    rc = Counter((x["regime"], x["strategy"]) for x in au_rows)
    a("- **AutoOracle routing:** " + "; ".join(
        f"regime `{rg}` → strategy `{st}` on {c}/{n} seeds" for (rg, st), c in rc.most_common())
      + ".")
    fails = {"oracle": sum(r["oracle"]["verdict"] == "FAIL" for r in rows),
             "auto": sum(x["verdict"] == "FAIL" for x in au_rows)}
    fails.update({m: sum(r[m]["max_error"] >= thr for r in rows)
                  for m in ["random_eqeval", "corner_eqeval"]})
    unconf = {t: sum(r[t]["unconfirmed_fail"] for r in rows) for t in ("oracle", "auto")}
    detect = (f"- **Detection at threshold {thr}:** FAIL on {fails['oracle']}/{n} seeds for "
              f"FHEOracle and {fails['auto']}/{n} for AutoOracle (verdicts returned by `run()`; "
              f"FAILs not confirmed by a re-measurement at the reported witness: "
              f"{unconf['oracle']} for FHEOracle, {unconf['auto']} for AutoOracle), "
              f"{fails['random_eqeval']}/{n} for uniform random and {fails['corner_eqeval']}/{n} "
              "for the corner set (any counted evaluation at or above the threshold, FHEOracle's "
              f"evaluation count). About {100 * ref['violating_fraction']['fraction']:.1f}% of the "
              "input box violates the threshold (plaintext estimate).")
    if all(v == n for v in fails.values()):
        detect += (" At this tolerance the methods differ in the size of the worst error they "
                   "report, not in whether a violation is found.")
    a(detect)
    a(f"- **Dominant error source:** at FHEOracle's worst witness the approximation component is "
      f"{ow['approx']:.6g} and the CKKS execution component is {ow['ckks']:.3e}.")
    a("")

    # Scope and setup
    a("## 1. Scope and setup")
    a("")
    a("**Question.** On one benchmark circuit, under the same input domain, threshold, CKKS "
      "parameters and key set, do FHEOracle (default settings) and AutoOracle (the recommended "
      "entry point) report a larger worst-case output error than uniform random sampling or a "
      "corner/boundary test set? FHEOracle is compared at equal evaluation budget and at equal "
      "wall-clock budget; AutoOracle at equal evaluation budget and against FHEOracle.")
    a("")
    c = R["circuit"]
    a("**Circuit** (`benchmarks/tenseal_circuits.py::build_tenseal_lr_d8`).")
    a("")
    a("- Intended model: σ(z) = 1 / (1 + e^(−z)), z = w·x + b, evaluated in float64 plaintext.")
    a("- Polynomial surrogate: T3(z) = 0.5 + z/4 − z³/48 (Taylor-3 sigmoid).")
    a("- FHE program: x encrypted as a CKKS vector; z = ⟨ct_x, w⟩ + b; T3(z) evaluated "
      "homomorphically; slot 0 decrypted.")
    a(f"- Weights: 200 steps of gradient descent on synthetic data (seed 42). "
      f"w = `{_vec(c['weights'], 4)}`, b = {c['bias']:.6f}.")
    a(f"- Input domain (all methods): [{c['bounds'][0][0]}, {c['bounds'][0][1]}]^{c['d']}. "
      f"Every evaluated input was checked to lie inside it.")
    a(f"- Threshold (all methods): {thr} absolute error on the probability output.")
    a("")
    k = R["ckks"]
    a("**CKKS parameters** (one context and one key set shared by every method).")
    a("")
    a("| Parameter | Value |")
    a("|---|---|")
    a(f"| Backend | TenSEAL {meta['packages']['tenseal']} (Microsoft SEAL-based) |")
    a(f"| poly_modulus_degree N | {k['poly_modulus_degree']} |")
    a(f"| coeff_mod_bit_sizes | {k['coeff_mod_bit_sizes']} (total {sum(k['coeff_mod_bit_sizes'])} bits) |")
    a(f"| global scale | 2^{int(np.log2(k['global_scale']))} |")
    a("| Galois keys | generated |")
    a("| Security level | not assessed in this report |")
    a("")
    a("**Environment.**")
    a("")
    a("| Item | Value |")
    a("|---|---|")
    a(f"| fhe-oracle | {meta['fhe_oracle_version']} |")
    a(f"| git SHA at start | `{meta['git_sha_start']}` |")
    if meta["git_changed_during_run"]:
        py = [p for p in meta["git_changed_during_run"] if p.endswith(".py")]
        a(f"| git SHA at end | `{meta['git_sha']}`; files changed: "
          f"{', '.join(meta['git_changed_during_run'])}. `fhe_oracle` and the circuit modules "
          f"are imported when the run starts; later edits are not reloaded by the running "
          f"process{' (Python files changed: ' + ', '.join(py) + ')' if py else ''} |")
    changed = [ln.split(None, 1)[1] for ln in (meta["git_tracked_changes"] or "").splitlines()
               if ln.strip()]
    a(f"| Uncommitted tracked-file changes at end | {', '.join(changed) if changed else 'none'} |")
    for p, h in meta["library_sha256_start"].items():
        end = meta["library_sha256"][p]
        a(f"| SHA-256 of `{p}` | `{h}`" + ("" if h == end else f" at start; `{end}` at end") + " |")
    a(f"| Script committed at run time | {'yes' if meta['script_tracked_in_git'] else 'no (untracked file)'} |")
    sha_note = ("" if meta["script_sha256"] == meta["script_sha256_start"]
                else f" at start; `{meta['script_sha256']}` at end (file changed during run)")
    a(f"| Script SHA-256 | `{meta['script_sha256_start']}`{sha_note} |")
    a(f"| Registered plugins | fitness: {meta['registered_plugins']['fitness'] or 'none'}; "
      f"heuristics: {meta['registered_plugins']['heuristics'] or 'none'} |")
    pk = meta["packages"]
    a(f"| tenseal / cma / numpy / scipy | {pk['tenseal']} / {pk['cma']} / {pk['numpy']} / {pk['scipy']} |")
    a(f"| Python | {meta['python']} |")
    a(f"| Platform | {meta['platform']} |")
    a(f"| CPU | {meta['cpu']} ({meta['cpu_count']} logical cores) |")
    a(f"| Run (UTC) | {meta['started_utc']} to {meta['date_utc']} |")
    a(f"| Total runtime | {R['runtime_s']:.0f} s |")
    a("")

    # Method
    a("## 2. Method")
    a("")
    a(f"**FHEOracle.** `FHEOracle(plaintext_fn, fhe_fn, input_dim=8, input_bounds, seed=s)"
      f".run(n_trials={B}, threshold={thr})` with library defaults (CMA-ES, σ0 = 1.0, start at "
      "the box centre, pure divergence fitness, no random floor, no restarts, no heuristic "
      "seeds, no plugins). `run()` evaluates at most `n_trials` search points (a partial final "
      "CMA-ES generation is evaluated but not told to CMA-ES), then re-measures its reported "
      "witness once. This is the configuration behind the README `lr_matched` row; it was not "
      "tuned for this report.")
    a("")
    a(f"**AutoOracle.** `AutoOracle(plaintext_fn, fhe_fn, bounds).run(n_trials={B}, "
      f"threshold={thr}, seed=s)` with library defaults (configured `n_probes` = {cfg['auto_n_probes']}; no "
      "`W, b`, so the preactivation test is off). It probes the landscape, classifies a regime "
      "and dispatches to an inner `FHEOracle` configuration. The exact probe steps are those of "
      "`fhe_oracle/autoconfig.py` with the SHA-256 recorded in section 1. Strategies used in "
      "this run:")
    a("")
    plugins_none = not meta["registered_plugins"]["heuristics"]
    for st in sorted({x["strategy"] for x in au_rows}):
        desc = STRATEGY_TEXT.get(st, "see `fhe_oracle/autoconfig.py`")
        if st == "robust_cma_es":
            desc += ("; with no heuristic plugin registered these come from Core's "
                     "`fallback_corner_seeds` (5 random box vertices and 5 uniform points)"
                     if plugins_none else "; seeds come from the registered heuristic plugin")
        a(f"- `{st}`: {desc}.")
    a("")
    a("**Uniform random.** Independent uniform draws from the same box (NumPy `default_rng(s)`).")
    a("")
    a(f"**Corner/boundary set.** A fixed pool of {R['corner_pool_size']} points: all 2^8 vertices, "
      "the 16 face centres and the box centre. Each seed evaluates the pool in a seeded random "
      "order until its budget is spent; the pool is not re-used.")
    a("")
    a("**Counting.** Every call to the plaintext model and to the CKKS program is counted for every "
      "method, including every AutoOracle evaluation and FHEOracle's re-measurement of its "
      "reported witness. For logging, the plaintext model and surrogate are "
      "recomputed at each FHE call; that instrumentation is identical for all methods and is not "
      "counted as search effort. Both tools' `on_evaluation` events are written to JSON Lines "
      "logs (`evaluation_logs/`, one per tool per seed, via `fhe_oracle.evallog."
      "JsonlEvaluationLog`) and matched one-to-one, in order, against the counted FHE calls; "
      "their SHA-256 digests are listed in the appendix.")
    a("")
    ex = summ["pool_exhausted_seeds"]
    eq_text = ("**Equal evaluation budget.** Each baseline is cut from one seeded stream per "
               "seed at exactly FHEOracle's counted FHE evaluations, and again at AutoOracle's.")
    for m in ("corner_eqeval", "corner_auto_eqeval"):
        if ex[m]:
            eq_text += (f" Exception ({METHOD_LABELS[m]}): the corner pool has "
                        f"{R['corner_pool_size']} points, so on seeds {ex[m]} it ran out and "
                        "used fewer evaluations.")
    a(eq_text)
    if ex["corner_eqtime"]:
        a("")
        a(f"On seeds {ex['corner_eqtime']} the corner pool ran out before the wall-clock budget "
          "was spent.")
    a("")
    a("**Equal wall-clock budget (FHEOracle only).** Each baseline receives FHEOracle's measured "
      "wall time for that seed (construction plus `run()`), and counts only evaluations that "
      "completed within it, cut from the same seeded stream.")
    a("")
    a("**AutoOracle vs FHEOracle.** Paired by seed at the same `n_trials`; the two tools' counted "
      "evaluations differ as reported above.")
    a("")
    a(f"**Statistics.** Per-seed max-error ratio (tool / reference), counts of seeds where the "
      "tool's value is larger, smaller or tied, and a paired two-sided Wilcoxon signed-rank test "
      "(scipy, `zero_method='wilcox'`) on the per-seed differences. A difference with absolute "
      f"value at most TIE_TOL = {TIE_TOL:g} is a tie: it is set to zero, so the test drops it, "
      f"and it is counted in the tie column. {n_tests} tests are reported; Holm adjustment is "
      "applied across all of them. Seeds vary the search randomness only; the circuit and keys "
      "are fixed.")
    a("")
    a("**Why TIE_TOL.** CKKS adds fresh noise to every encryption, so the same input measures "
      "slightly differently each time, and a maximum over evaluations gives a method that "
      "re-measures its witness an extra noise draw. The largest spread between repeated "
      f"measurements of the same input in this run is {spread:.2e}, so TIE_TOL is about "
      f"{TIE_TOL / spread:.0f} times that spread. The value is the margin that "
      "`benchmarks/preregistration_2026-09-15.md` uses for comparisons against the corner set. "
      "This tolerance was adopted after the first 20-seed run of this report showed noise-level "
      "differences counted as wins for AutoOracle over the corner set.")
    a("")

    # Results
    a("## 3. Results")
    a("")
    a("### 3.1 FHEOracle per-seed maximum error")
    a("")
    a("FHE evaluation counts in brackets for the wall-clock columns.")
    a("")
    a("| Seed | FHEOracle | Random, equal evals | Random, equal time [evals] | Corner, equal evals | "
      "Corner, equal time [evals] | FHEOracle wall s | Verdict [re-measured ≥ threshold] |")
    a("|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in rows:
        a(f"| {r['seed']} | {r['oracle']['max_error']:.4f} | {r['random_eqeval']['max_error']:.4f} | "
          f"{r['random_eqtime']['max_error']:.4f} [{r['random_eqtime']['fhe_evals']}] | "
          f"{r['corner_eqeval']['max_error']:.4f} | {r['corner_eqtime']['max_error']:.4f} "
          f"[{r['corner_eqtime']['fhe_evals']}] | {r['oracle']['wall_s']:.2f} | "
          f"{r['oracle']['verdict']} [{'yes' if r['oracle']['remeasure_confirms'] else 'no'}] |")
    a("")
    a("### 3.2 AutoOracle per-seed results")
    a("")
    a("| Seed | Regime | Strategy | FHE evals [re-measurements] | AutoOracle | AutoOracle reported "
      "`max_error` | Random, same evals | Corner, same evals | FHEOracle | "
      "Verdict [re-measured at witness ≥ threshold] |")
    a("|---:|---|---|---:|---:|---:|---:|---:|---:|---|")
    for r in rows:
        x = r["auto"]
        a(f"| {r['seed']} | {x['regime']} | {x['strategy']} | {x['fhe_evals']} "
          f"[{x['remeasurements']}] | {x['max_error']:.4f} | {x['reported_max_error']:.4f} | "
          f"{r['random_auto_eqeval']['max_error']:.4f} | {r['corner_auto_eqeval']['max_error']:.4f} | "
          f"{r['oracle']['max_error']:.4f} | "
          f"{x['verdict']} [{'yes' if x['remeasure_confirms'] else 'no'}] |")
    a("")
    a("Full per-seed data, including witnesses and probe reasons: `per_seed.csv` and `report.json`.")
    a("")
    a("### 3.3 Reference points (not paired comparisons)")
    a("")
    sup = ref["approx_supremum"]
    full = ref["corner_pool_full"]
    vf = ref["violating_fraction"]
    a("| Reference | Value |")
    a("|---|---|")
    a(f"| Supremum of approximation error \\|σ − T3\\| over the box (plaintext, from the exact "
      f"range of z ∈ [{sup['z_min']:.3f}, {sup['z_max']:.3f}]) | {sup['sup']:.4f} at z = "
      f"{sup['z_at_sup']:.3f}{', a box vertex' if sup['at_vertex'] else ''} |")
    a(f"| Entire corner/boundary pool evaluated once under CKKS ({full['fhe_evals']} evaluations) "
      f"| max error {full['max_error']:.4f} |")
    a(f"| Share of the box with \\|σ − T3\\| ≥ {thr} (plaintext Monte Carlo, "
      f"{vf['n_samples']:,} samples) | {100 * vf['fraction']:.2f}% (95% CI "
      f"{100 * vf['wilson95'][0]:.2f}–{100 * vf['wilson95'][1]:.2f}%) |")
    for m, label in [("oracle", "FHEOracle"), ("auto", "AutoOracle"),
                     ("random_eqeval", "Random, FHEOracle's evals"),
                     ("corner_eqeval", "Corner, FHEOracle's evals")]:
        frac = np.median([r[m]["max_error"] / sup["sup"] for r in rows])
        a(f"| {label}: median per-seed max error as a fraction of the supremum | {frac:.3f} |")
    a("")
    a("Fractions can slightly exceed 1 because the CKKS execution component adds to the "
      "approximation error.")
    a("")
    a("### 3.4 Timing")
    a("")
    oh = float(np.median([r["oracle"]["overhead_frac"] for r in rows]))
    tm = R["timing"]
    ld0, ld1 = meta["loadavg_1_5_15_start"], meta["loadavg_1_5_15_end"]
    a(f"Median time per FHE evaluation: FHEOracle {tm['oracle_median_fhe_ms']:.2f} ms, "
      f"AutoOracle {tm['auto_median_fhe_ms']:.2f} ms, random {tm['random_median_fhe_ms']:.2f} ms, "
      f"corner {tm['corner_median_fhe_ms']:.2f} ms. Share of FHEOracle wall time spent outside "
      f"FHE calls (CMA-ES, plaintext model, logging): median {100 * oh:.2f}%.")
    a("")
    counts = (f"Within FHEOracle's wall time, uniform random completed "
              f"{_med_range([r['random_eqtime']['fhe_evals'] for r in rows])} evaluations and "
              f"the corner set {_med_range([r['corner_eqtime']['fhe_evals'] for r in rows])}, "
              f"against FHEOracle's {_med_range([r['oracle']['fhe_evals'] for r in rows])}.")
    if oh < 0.05:
        counts += (" With search overhead this small, differences between these counts mostly "
                   "reflect run-to-run variation in CKKS evaluation time rather than the cost of "
                   "CMA-ES. For a cheaper model, where overhead is a larger share, the two budgets "
                   "would diverge.")
    a(counts)
    a("")
    a(f"System load average (1/5/15 min): {ld0[0]:.1f}/{ld0[1]:.1f}/{ld0[2]:.1f} at start, "
      f"{ld1[0]:.1f}/{ld1[1]:.1f}/{ld1[2]:.1f} at end. The machine was not isolated.")
    a("")
    a("### 3.5 CKKS execution component seen during search (incidental)")
    a("")
    a("None of the methods targeted this quantity; each searched for total error. Median over "
      "seeds of the largest |T3(z) − CKKS output| among each method's counted evaluations:")
    a("")
    a("| FHEOracle | AutoOracle | Random, FHEOracle's evals | Corner, FHEOracle's evals |")
    a("|---:|---:|---:|---:|")
    a(f"| {np.median([r['oracle']['max_ckks_component'] for r in rows]):.3e} | "
      f"{np.median([r['auto']['max_ckks_component'] for r in rows]):.3e} | "
      f"{np.median([r['random_eqeval']['max_ckks_component'] for r in rows]):.3e} | "
      f"{np.median([r['corner_eqeval']['max_ckks_component'] for r in rows]):.3e} |")
    a("")

    # Witness
    W = R["witnesses"]
    a("## 4. Worst witness and error decomposition")
    a("")
    a("The circuit exposes its surrogate, so the error at any input splits into approximation "
      "error |σ(z) − T3(z)| (intended model vs polynomial, both plaintext) and CKKS execution "
      "error |T3(z) − CKKS output| (plaintext polynomial vs encrypted evaluation of the same "
      "polynomial). The total is bounded by their sum. For a compiled customer model this split "
      "requires the surrogate plaintext to be available separately.")
    a("")
    a("| Witness | x | z | Intended model σ(z) | Surrogate plaintext T3(z) | CKKS output | "
      "Total error | Approximation | CKKS execution |")
    a("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    a(witness_row(f"FHEOracle, seed {W['oracle_worst_seed']} (largest `remeasured_error`), "
                  "re-measured witness", ow))
    a(witness_row(f"AutoOracle, seed {W['auto_worst_seed']} (largest reported `max_error`), "
                  "reported witness", W["auto_worst"]))
    a(witness_row(f"Largest over all methods and seeds "
                  f"({METHOD_LABELS[W['overall_worst_method']]}, seed "
                  f"{W['overall_worst_seed']})", W["overall_worst"]))
    if R["shrink"].get("record"):
        a(witness_row("FHEOracle witness after `FHEOracle.shrink`", R["shrink"]["record"]))
    a("")
    rep = W["repeat_encryptions"]
    a(f"Re-encrypting FHEOracle's witness {rep['k']} times (fresh CKKS randomness each time) gives "
      f"a CKKS execution component between {rep['ckks_min']:.3e} and {rep['ckks_max']:.3e}; the "
      f"total error varies by at most {rep['total_spread']:.3e}.")
    a("")
    if ow["ckks"] < 1e-3 * ow["approx"]:
        a(f"On this circuit the measured error is almost entirely approximation error: the Taylor-3 "
          f"surrogate departs from the sigmoid as |z| grows, and the domain allows |z| up to "
          f"{max(abs(sup['z_min']), abs(sup['z_max'])):.1f}. That defect is visible in plaintext, "
          "without encryption. The CKKS execution error at the same inputs is "
          f"{ow['ckks'] / ow['approx']:.1e} of the approximation error.")
    else:
        a("At this witness the CKKS execution component is not negligible relative to the "
          "approximation component; see the table above.")
    a("")
    sh = R["shrink"]
    a("### Shrunk witness")
    a("")
    if sh.get("record"):
        if sh["returned_original"]:
            a(f"`FHEOracle.shrink(result, max_evals={sh['max_evals']})` could not confirm a "
              "smaller failing input and returned the original witness.")
        else:
            a(f"`FHEOracle.shrink(result, max_evals={sh['max_evals']})` moved the witness toward "
              f"the box centre while keeping its error at or above {thr}. Distance from the "
              f"centre: {sh['original_norm']:.3f} → {sh['shrunk_norm']:.3f} "
              f"({100 * (1 - sh['shrunk_norm'] / sh['original_norm']):.1f}% smaller).")
        a("")
        match = "matches" if sh["n_evals"] == sh["fhe_calls_total"] else "does not match"
        a(f"`ShrinkResult.max_error` = {sh['max_error']:.12g}; `ShrinkResult.n_evals` = "
          f"{sh['n_evals']}; FHE calls measured during shrink: {sh['fhe_calls_total']} ({match} "
          "`n_evals`). These evaluations are not part of the comparisons above.")
        rs = sh["repeat_encryptions"]
        below = sum(t < thr for t in rs["totals"])
        a("")
        a(f"Re-encrypting the shrunk witness {rs['k']} times afterwards gives total error between "
          f"{rs['total_min']:.9g} and {rs['total_max']:.9g} (spread "
          f"{rs['total_max'] - rs['total_min']:.2e}); {below}/{rs['k']} re-encryptions are below "
          f"the threshold.")
        if sh["max_error"] < thr:
            a("")
            a("**Shrink caveat.** The installed `FHEOracle.shrink` returned a witness whose own "
              "measured error is below the threshold. Treat the shrunk input as a pointer to the "
              "failure boundary, not as a reproducible FAIL; the unshrunk witness above is the "
              "reproducible one.")
        elif below:
            a("")
            a(f"**Shrink caveat.** `ShrinkResult.max_error` meets the threshold, but the shrunk "
              f"input sits near the failure boundary, and {below}/{rs['k']} later re-encryptions "
              "fell below it. Treat it as a pointer to the boundary; the unshrunk witness above "
              "is the reproducible FAIL.")
    else:
        a(f"Shrink not run: {sh.get('reason')}.")
    a("")

    # PASS / FAIL
    a("## 5. What PASS and FAIL mean")
    a("")
    wrep = W["repeat_encryptions"]
    w_ok = sum(r["total"] >= thr for r in wrep["records"])
    conf_o = sum(r["oracle"]["remeasure_confirms"] for r in rows
                 if r["oracle"]["verdict"] == "FAIL")
    a(f"- **FAIL** means the tool evaluated a specific input, inside the stated domain, whose "
      f"output differed from the intended model by at least the threshold ({thr}); in this "
      "library version any counted search evaluation can set it. The final re-measurement at "
      f"FHEOracle's reported witness also met the threshold on {conf_o}/{fails['oracle']} "
      "FHEOracle FAIL seeds. Re-running "
      f"the model on FHEOracle's reported witness met the threshold on {w_ok}/{wrep['k']} fresh "
      "encryptions here. A witness whose error sits within CKKS noise of the threshold can "
      "re-evaluate below it (see the shrunk witness in section 4).")
    a("- **PASS** means no threshold violation was observed during the specified search, on the "
      "stated domain, budget, parameters and backend version. It is not proof of correctness, "
      "cryptographic security or regulatory compliance, and says nothing about inputs outside "
      "the domain.")
    a("- A method that reports a larger maximum error has found a worse input; it has not "
      "necessarily found more distinct defects.")
    a("")

    # Limitations
    a("## 6. Limitations")
    a("")
    a("- **One circuit.** A d = 8 logistic regression with synthetic weights. It is a benchmark "
      "circuit, not a customer model; results depend on circuit, domain, parameters and search "
      "strategy.")
    if ow["ckks"] < 1e-3 * ow["approx"]:
        a("- **Mostly approximation error.** The comparisons above measure how well each method "
          "locates a surrogate-design defect that is visible without encryption. A search "
          "targeting CKKS execution error alone was not run.")
    if sup["at_vertex"]:
        a("- **Worst case at a vertex.** For this circuit the error grows with |w·x + b|, so the "
          "maximum sits at a box vertex, which favours a corner/boundary test set and any "
          "method that evaluates box vertices (for example injected corner seeds or a boundary "
          "probe). Circuits with interior worst cases may rank the methods differently.")
    a(f"- **One budget and two configurations.** `n_trials` = {B}; FHEOracle and AutoOracle with "
      "library defaults. `check()`, explicit restarts and random-floor options were not "
      "evaluated separately. The README's historical `lr_matched` row uses B = 60, does not count "
      "the oracle's re-measurement and has no wall-clock comparison, so it is not directly "
      "comparable.")
    a("- **Timing.** One machine, sequential runs in one process (FHEOracle, then AutoOracle, then "
      "the baseline streams, for each seed), no isolation from other system load. Equal-time "
      "evaluation counts will differ on other hardware.")
    a("- **Not bit-reproducible.** CKKS encryption randomness is not seeded (the "
      "`TenSEALContext` seed argument is unused), so reruns differ in low-order digits and "
      "searches can follow different paths.")
    if any(ex[m] for m in ("corner_eqeval", "corner_eqtime", "corner_auto_eqeval")):
        a(f"- **Corner pool exhausted.** The {R['corner_pool_size']}-point pool was smaller than "
          "the budget on some seeds (section 2), so that baseline did not receive the full "
          "budget.")
    if sh.get("record") and sh["max_error"] < thr:
        a("- **Shrunk witness below threshold.** The installed `FHEOracle.shrink` returned a "
          "witness that measures below the threshold (section 4).")
    elif sh.get("record") and any(t < thr for t in sh["repeat_encryptions"]["totals"]):
        a("- **Shrunk witness near the boundary.** Some later re-encryptions of the shrunk input "
          "fall below the threshold (section 4).")
    a("- **Not independently replicated.** Produced by the tool's author's own script.")
    a("- **No operational-profile replay.** Inputs from a real deployment distribution were not "
      "tested; the uniform box is an adversarial domain, not a usage profile.")
    a(f"- **Statistics.** {n} paired seeds on one fixed circuit; p-values describe search "
      "randomness, not variation across models.")
    ac = summ["auto_vs_corner_eqeval"]
    missed = ac.get("reference_missed_worst_vertex_seeds", [])
    lim = ("- **Tie tolerance chosen after the first run.** TIE_TOL was adopted after the first "
           "20-seed run of this report counted CKKS-noise-level differences as wins. With it, "
           f"AutoOracle and the corner set match within noise on {ac['ties']}/{n} seeds. The "
           f"corner set's evaluation budget did not reach the worst vertex on {len(missed)}/{n} "
           "seeds")
    if ac["tool_larger"] and ac.get("tool_larger_where_reference_missed") == ac["tool_larger"]:
        lim += (f", and those are the only seeds where AutoOracle is larger ({ac['tool_larger']}/"
                f"{n}).")
    else:
        lim += (f"; AutoOracle is larger on {ac['tool_larger']}/{n} seeds and smaller on "
                f"{ac['tool_smaller']}/{n}.")
    a(lim)
    a("")

    # Reproduce
    a("## 7. Reproduce")
    a("")
    a("From the repository root, with `tenseal`, `cma`, `numpy` and `scipy` installed (versions "
      "above):")
    a("")
    a("```bash")
    a(R["reproduce_command"])
    a("```")
    a("")
    a(f"Command used for this run: `{' '.join([meta['python_executable']] + meta['argv'])}`")
    a("")

    a("## Appendix: evaluation logs")
    a("")
    a("| Seed | FHEOracle log SHA-256 | AutoOracle log SHA-256 |")
    a("|---:|---|---|")
    for r in rows:
        a(f"| {r['seed']} | `{r['oracle']['log']['sha256']}` | `{r['auto']['log']['sha256']}` |")
    a("")
    a("Files: `evaluation_logs/fheoracle_seed<N>.jsonl` and `evaluation_logs/autooracle_seed<N>.jsonl`.")
    a("")
    a("## Appendix: tool output for FHEOracle's worst seed")
    a("")
    scheme = R["oracle_result_json"]["scheme"]
    if scheme == "fhe_fn":
        scheme_txt = ("`Scheme: fhe_fn` means the model under test was supplied as a callable "
                      "rather than an adapter object; the callable here runs real TenSEAL CKKS.")
    elif scheme == "plaintext-diff":
        scheme_txt = ("`Scheme: plaintext-diff` is this library version's label for callable "
                      "mode; the callable here runs real TenSEAL CKKS.")
    else:
        scheme_txt = f"`Scheme: {scheme}` is the label reported by the library."
    a("Rendered by `fhe_oracle.report.to_markdown` (headings demoted one level). "
      f"{scheme_txt} `Trials` excludes the final re-measurement.")
    a("")
    for ln in R["tool_markdown"].splitlines():
        a("#" + ln if ln.startswith("#") else ln)
    a("")
    return "\n".join(L)


# --- Main --------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=20, help="number of paired seeds")
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--budget", type=int, default=200, help="n_trials for both tools")
    ap.add_argument("--threshold", type=float, default=0.01)
    ap.add_argument("--shrink-evals", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=5, help="re-encryptions at witnesses")
    ap.add_argument("--out", default=os.path.join(THIS_DIR, "results", "sample_report"))
    args = ap.parse_args(argv)
    if args.budget < AUTO_N_PROBES + AUTO_RESERVE:
        ap.error(f"--budget must be at least AutoOracle's n_probes + {AUTO_RESERVE} "
                 f"({AUTO_N_PROBES + AUTO_RESERVE})")
    if not tsa.HAVE_TENSEAL:
        print("TenSEAL is not installed; this report requires a real CKKS backend.")
        return 2

    t_start = time.perf_counter()
    load_start = list(os.getloadavg())
    git_start = _run(["git", "rev-parse", "HEAD"])
    started_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    script_sha_start = _sha256(os.path.join(ROOT, SCRIPT_REL))
    lib_sha_start = {p: _sha256(os.path.join(ROOT, p)) for p in LIB_FILES}
    seeds =list(range(args.seed_offset, args.seed_offset + args.seeds))
    thr, B = args.threshold, args.budget
    log_dir = os.path.join(args.out, "evaluation_logs")
    os.makedirs(log_dir, exist_ok=True)
    for old in glob.glob(os.path.join(log_dir, "*_seed*.jsonl")):
        os.remove(old)  # stale logs from an earlier run into the same --out

    ctx = tsa.TenSEALContext()
    circuit = build_tenseal_lr_d8(ctx)
    cc = CountingCircuit(circuit)
    bounds = circuit["bounds"]
    lo, hi = cc.lo, cc.hi
    pool = corner_pool(bounds)

    sup = approx_supremum(cc.w, cc.b, bounds)
    vf = violating_fraction(cc.w, cc.b, bounds, thr)
    print(f"sup|sigma-T3| = {sup['sup']:.4f}; violating fraction = {vf['fraction']:.4f}")

    rows, oracles = [], {}
    for seed in seeds:
        t0 = time.perf_counter()
        oracle, res, o = run_oracle(cc, circuit, seed, B, thr, log_dir)
        oracles[seed] = (oracle, res)
        _, _, au = run_auto(cc, circuit, seed, B, thr, log_dir)
        cuts = {"eqeval": o["fhe_evals"], "auto_eqeval": au["fhe_evals"]}
        cap = 20 * max(cuts.values())
        row = {"seed": seed, "oracle": o, "auto": au}
        for prefix, make in (("random", random_points(lo, hi, seed)),
                             ("corner", corner_points(pool, seed))):
            st = run_stream(cc, make, cuts, o["wall_s"], thr, cap)
            row[f"{prefix}_eqeval"] = st["eqeval"]
            row[f"{prefix}_eqtime"] = st["eqtime"]
            row[f"{prefix}_auto_eqeval"] = st["auto_eqeval"]
        rows.append(row)
        print(f"seed {seed:2d}: FHEOracle {o['max_error']:.4f} [{o['fhe_evals']}, {o['wall_s']:.2f}s] "
              f"AutoOracle {au['max_error']:.4f} [{au['fhe_evals']}, {au['regime']}/{au['strategy']}] "
              f"random {row['random_eqeval']['max_error']:.4f}/{row['random_eqtime']['max_error']:.4f}"
              f"/{row['random_auto_eqeval']['max_error']:.4f} "
              f"corner {row['corner_eqeval']['max_error']:.4f}/{row['corner_eqtime']['max_error']:.4f}"
              f"/{row['corner_auto_eqeval']['max_error']:.4f} ({time.perf_counter() - t0:.1f}s)")

    summary = build_comparisons(rows, sup, bounds)
    summary["pool_exhausted_seeds"] = {
        m: [r["seed"] for r in rows if r[m]["pool_exhausted"]] for m in BASELINE_CUTS
    }
    summary["detection_counts"] = {
        "oracle_verdict_fail": sum(r["oracle"]["verdict"] == "FAIL" for r in rows),
        "oracle_observed_max_ge_threshold": sum(r["oracle"]["max_error"] >= thr for r in rows),
        "auto_verdict_fail": sum(r["auto"]["verdict"] == "FAIL" for r in rows),
        "auto_observed_max_ge_threshold": sum(r["auto"]["max_error"] >= thr for r in rows),
    }
    if any(abs(r["oracle"]["reported_max_error"] - r["oracle"]["max_error"])
           > _tol(r["oracle"]["max_error"]) for r in rows):
        raise RuntimeError("FHEOracle max_error differs from the largest counted error")
    summary["oracle_search_vs_remeasured_max_abs_diff"] = float(max(
        abs(r["oracle"]["search_max_error"] - r["oracle"]["remeasured_error"]) for r in rows))
    summary["detection_counts"].update({
        "oracle_unconfirmed_fail": sum(r["oracle"]["unconfirmed_fail"] for r in rows),
        "auto_unconfirmed_fail": sum(r["auto"]["unconfirmed_fail"] for r in rows),
        "auto_remeasured_not_at_witness": sum(not r["auto"]["remeasured_at_witness"] for r in rows),
    })
    gaps = [(r["seed"], r["auto"]["max_error"] - r["auto"]["reported_max_error"], r["auto"])
            for r in rows]
    big = [(s, g, x) for s, g, x in gaps if g > 1e-3 * x["reported_max_error"]]
    summary["auto_reported_gap"] = {
        "seeds": [s for s, _, _ in big],
        "max_gap": max((g for _, g, _ in big), default=0.0),
        "max_abs_gap": float(max(abs(g) for _, g, _ in gaps)),
    }
    summary["auto_routing"] = {f"{rg}->{st}": c for (rg, st), c in
                               Counter((r["auto"]["regime"], r["auto"]["strategy"])
                                       for r in rows).items()}
    summary["auto_witness_kinds"] = dict(Counter(r["auto"]["witness_kind"] for r in rows))

    # Witnesses
    # Select by the same quantity the witness row displays.
    worst = max(rows, key=lambda r: r["oracle"]["remeasured_error"])
    w_seed = worst["seed"]
    oracle, res = oracles[w_seed]
    ow = worst["oracle"]["reported_record"]
    auto_worst = max(rows, key=lambda r: r["auto"]["reported_max_error"])
    cc.reset()
    for _ in range(args.repeats):
        cc.fhe(res.worst_input)
    reps = list(cc.records)
    cands = []
    for m in ("oracle", "auto", *BASELINE_CUTS):
        cands += [(r[m]["max_error"], m, r["seed"], r[m]["witness"]) for r in rows]
    best = max(cands, key=lambda t: t[0])

    shrink = {"max_evals": args.shrink_evals}
    if res.verdict == "FAIL":
        cc.reset()
        sr = oracle.shrink(res, max_evals=args.shrink_evals)
        tol = 1e-12 * max(1.0, sr.max_error)
        returned_original = np.array_equal(sr.shrunk_input, sr.original_input)
        match = [r for r in reversed(cc.records)
                 if np.allclose(r["x"], sr.shrunk_input, rtol=0, atol=1e-12)
                 and abs(r["total"] - sr.max_error) <= tol]
        if match:
            srec = match[0]
        else:
            # Original witness returned with run()'s max_error: search or re-measured record.
            fb = [rec for rec in (worst["oracle"]["search_record"], ow)
                  if returned_original and abs(rec["total"] - sr.max_error) <= tol]
            if not fb:
                raise RuntimeError("ShrinkResult does not match any logged evaluation")
            srec = fb[0]
        n_shrink_fhe = cc.n_fhe
        if sr.n_evals != n_shrink_fhe:
            raise RuntimeError(
                f"ShrinkResult.n_evals={sr.n_evals} but shrink made {n_shrink_fhe} FHE calls")
        cc.reset()
        for _ in range(args.repeats):
            cc.fhe(sr.shrunk_input)
        srep = [r["total"] for r in cc.records]
        shrink.update({
            "original_input": sr.original_input, "shrunk_input": sr.shrunk_input,
            "original_norm": sr.original_norm, "shrunk_norm": sr.shrunk_norm,
            "max_error": sr.max_error, "threshold": sr.threshold, "n_evals": sr.n_evals,
            "fhe_calls_total": n_shrink_fhe, "n_evals_matches_fhe_calls": sr.n_evals == n_shrink_fhe,
            "record": srec, "repr": repr(sr),
            "returned_original": bool(returned_original),
            "repeat_encryptions": {"k": args.repeats, "total_min": min(srep),
                                   "total_max": max(srep), "totals": srep},
        })
    else:
        shrink["reason"] = "worst FHEOracle seed returned PASS"

    # Full corner pool reference
    cc.reset()
    for x in pool:
        absolute_error(cc.plain(x.tolist()), cc.fhe(x.tolist()))
    full = summarize(list(cc.records), len(cc.records), thr)

    diagnostics = {
        "approximation_error": ow["approx"], "ckks_execution_error": ow["ckks"],
        "intended_model_output": ow["model"], "surrogate_plaintext_output": ow["surrogate"],
        "ckks_output": ow["fhe"],
    }
    runtime = time.perf_counter() - t_start
    out_rel = _repo_rel(os.path.abspath(args.out))
    med_ms = lambda m: float(np.median([r[m]["median_fhe_eval_ms"] for r in rows]))  # noqa: E731
    R = {
        "config": {"n_seeds": len(seeds), "seeds": seeds, "budget": B, "threshold": thr,
                   "shrink_evals": args.shrink_evals, "repeats": args.repeats,
                   "auto_n_probes": AUTO_N_PROBES,
                   "oracle_kwargs": {"seed": "s", "all_other": "library defaults"},
                   "auto_kwargs": {"seed": "s", "all_other": "library defaults"}},
        "metadata": collect_metadata(),
        "ckks": {"poly_modulus_degree": tsa.CKKS_POLY_MODULUS_DEGREE,
                 "coeff_mod_bit_sizes": list(tsa.CKKS_COEFF_MOD_BIT_SIZES),
                 "global_scale": tsa.CKKS_GLOBAL_SCALE, "galois_keys": True,
                 "shared_context": True},
        "circuit": {"name": circuit["name"], "d": circuit["d"], "bounds": bounds,
                    "weights": cc.w.tolist(), "bias": cc.b,
                    "intended_model": "sigmoid(w.x + b)",
                    "surrogate": "0.5 + z/4 - z^3/48, z = w.x + b"},
        "corner_pool_size": len(pool),
        "references": {"approx_supremum": sup, "violating_fraction": vf,
                       "corner_pool_full": full},
        "timing": {
            "oracle_median_fhe_ms": med_ms("oracle"),
            "auto_median_fhe_ms": med_ms("auto"),
            "random_median_fhe_ms": med_ms("random_eqeval"),
            "corner_median_fhe_ms": med_ms("corner_eqeval"),
        },
        "summary": summary,
        "per_seed": rows,
        "witnesses": {
            "oracle_worst_seed": w_seed, "oracle_worst": ow,
            "auto_worst_seed": auto_worst["seed"],
            "auto_worst": auto_worst["auto"]["reported_record"],
            "overall_worst_method": best[1], "overall_worst_seed": best[2],
            "overall_worst": best[3],
            "repeat_encryptions": {
                "k": args.repeats,
                "ckks_min": min(r["ckks"] for r in reps), "ckks_max": max(r["ckks"] for r in reps),
                "total_spread": max(r["total"] for r in reps) - min(r["total"] for r in reps),
                "records": reps,
            },
        },
        "shrink": shrink,
        "oracle_result_json": json.loads(to_json(res, diagnostics=diagnostics)),
        "tool_markdown": to_markdown(res, diagnostics=diagnostics),
        "runtime_s": runtime,
        "reproduce_command": (f"python {SCRIPT_REL} --seeds {len(seeds)} --seed-offset "
                              f"{args.seed_offset} --budget {B} --threshold {thr} "
                              f"--shrink-evals {args.shrink_evals} --repeats {args.repeats} "
                              f"--out {out_rel}"),
    }
    meta = R["metadata"]
    meta["loadavg_1_5_15_start"] = load_start
    meta["loadavg_1_5_15_end"] = list(os.getloadavg())
    meta["started_utc"] = started_utc
    meta["git_sha_start"] = git_start
    meta["git_changed_during_run"] = (
        (_run(["git", "diff", "--name-only", git_start, meta["git_sha"]]) or "").splitlines()
        if git_start != meta["git_sha"] else []
    )
    meta["script_sha256_start"] = script_sha_start
    meta["library_sha256_start"] = lib_sha_start
    os.makedirs(args.out, exist_ok=True)
    write_csv(os.path.join(args.out, "per_seed.csv"), rows)
    with open(os.path.join(args.out, "report.json"), "w") as fh:
        json.dump(_clean(R), fh, indent=2)
    with open(os.path.join(args.out, "report.md"), "w") as fh:
        fh.write(build_markdown(R))
    print(f"wrote {args.out} in {runtime:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
