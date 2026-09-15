# Copyright (C) 2026 Bader Alissaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Study A: AutoOracle boundary-probe confirmation (pre-registered).

PRE-REGISTRATION: benchmarks/preregistration_2026-09-15.md, section "A. AutoOracle boundary-probe
confirmation", committed at 962aab31e4239d074130a8f80394dac3e2731c1d (file SHA-256
912da47075386a07163dba517dfc06b4dfa9d24461fbb587b30e1c924a694386). This script does not edit it.

ARMS (per circuit, per seed s; n_trials = 200, threshold 0.01, seeds 11-30):
  after       committed AutoOracle(plaintext_fn, fhe_fn, bounds, on_evaluation=log).run(n_trials, seed=s)
  accounting  ACCOUNTING-ONLY: the same committed code with the boundary probe disabled (method below)
  random      N_s uniform draws, numpy default_rng([s, 2])
  corner      corner/boundary pool (2^d vertices, 2d face centres, centre), default_rng([s, 1]) order,
              first N_s points; a pool smaller than N_s is used whole (d = 6: 77 points). The pool is
              enumerated in full, so d <= 12 (larger d raises).
  N_s = the AFTER arm's counted evaluations (result.n_trials) on seed s.

ACCOUNTING-ONLY METHOD (benchmark-side; no library edit). The committed AutoOracle.run() skips the
whole pre-search path (boundary evaluations, witness merge and the evaluation reserved for its
re-measurement) when ``self.oracle_kwargs.get(k)`` is truthy for a k in ``_OTHER_OBJECTIVES``. For this
arm only, the script appends a sentinel name to ``fhe_oracle.autoconfig._OTHER_OBJECTIVES``, replaces
the instance's ``oracle_kwargs`` with a dict whose ``get(sentinel)`` is True (the sentinel is never
stored, so it is never forwarded to FHEOracle), and sets ``_RESERVE = 2`` (the accounting-only
minimum n_probes + 2). Both module globals are restored afterwards. Probe evaluations stay counted.

EQUIVALENCE CHECK: ``--check-equivalence PATH`` compares this arm with a saved accounting-only
autoconfig.py on plaintext stand-ins (evaluated inputs, results and counts), writing each comparison
row to equivalence_check.json as it completes.
  Reference: autoconfig_accounting_only_v5.py, SHA-256
  006f9ca2ec7b8bb8a21a43b921d5887c518de0138000db9f5f829eb320584c72, the accounting-only autoconfig
  saved (outside the repo) before the probe was committed.
  Run 2026-09-15 before any study seed; HEAD, guarded SHA-256s (including this script's) and every
  run row are in the check's equivalence_check.json. Stand-ins: plaintext surrogates of lr_d8,
  cheb_d10 and poly_d6, plus bump_d8, decoy_d6, saturated d=3, plateau d=5, distant-defect shell d=2,
  low-rank d=16 and preactivation k=1, k=2. (n_probes, n_trials) in {(50, 200), (50, 52), (2, 4)} for
  the first five, regime-specific budgets for the rest (low-rank (30, 6600)); seeds 0-4.
  Result: 120/120 runs identical in evaluated inputs, result fields (max_error, worst_input, verdict,
  n_trials, search_max_error, remeasured_error, strategy, regime) and FHE call counts; the event log
  matched every FHE call in 120/120; 0 boundary events; 1 re-measurement per run; globals restored.
  Unpatched AFTER differed from the reference in 45 of the 60 runs whose budget allows AFTER (every
  non-preactivation run; the preactivation route never runs the probe), so the patch is not a no-op.

ARCHIVED EQUIVALENCE INPUTS: ``--study`` requires ``--reference`` (the saved accounting-only module)
and ``--equivalence-json`` (the check's output). Before any circuit runs, both are refused unless the
check completed and passed, its reference SHA-256 equals the given file, and fhe_oracle/core.py,
fhe_oracle/autoconfig.py and this script hash as they did during the check. Both are then copied to
<out>/equivalence/ with manifest.json holding their SHA-256; a resumed run must present the same pair.

CHOICES WHERE THE PRE-REGISTRATION IS NOT EXPLICIT (fixed before any study seed):
- Counted evaluations = the evaluations counted in result.n_trials (event kinds probe, structure,
  boundary, search). The documented re-measurement is kind "remeasure": 1 for ACCOUNTING-ONLY; 1 or 2
  for AFTER (2 when a pre-search witness replaces the search witness).
- Metric = max |plaintext(x) - fhe(x)| over counted evaluations, recomputed from each logged call. The
  max including re-measurements and AutoOracle's reported max_error are also recorded, not tested.
- Criterion 1: per-seed ratio AFTER / ACCOUNTING-ONLY, median over the 20 seeds, pass if >= 0.95. Paired
  Wilcoxon signed-rank (scipy, two-sided, zero_method="wilcox"; p = 1 when every pair ties). Holm
  correction over the five criterion-1 tests (the study's family). "AFTER lower" = the rank sum of
  negative differences exceeds that of positive ones. Fails if Holm p < 0.05 and AFTER lower.
- Criterion 2 (lr_d8, cheb_d10 only): AFTER metric >= corner metric - 1e-5 on >= 16 of 20 seeds.
- Criterion 3, every seed and both arms: FHE calls == result.n_trials + remeasure events, calls <=
  n_trials + remeasure events, and remeasure events within the documented count above. Whether calls
  <= n_trials (the library's stricter invariant) is recorded too.
- Criteria, Holm correction and the decision are evaluated only for the full pre-registered set: all
  five circuits, seeds 11-30, n_trials 200, threshold 0.01, every circuit file complete, start and end
  provenance equal within each file and across files, and the pre-registration check passing at start
  and end. Holm is then over the five criterion-1 tests. Otherwise every pass field, Holm p and the
  decision are null, and summary.json records study false and Holm family size 0.
- Arm order within a seed: after, accounting, random, corner. One TenSEAL context and key set per
  circuit (per process), shared by every arm; CKKS encryption noise is not seeded.
- Mock definitions (bump_d8, decoy_d6) are identical to benchmarks/autooracle_boundary_eval.py.
- Provenance at start, on resume, before every seed and at end: git HEAD and the SHA-256 of
  fhe_oracle/core.py, fhe_oracle/autoconfig.py and this script must equal their start values, and the
  pre-registration file and its blob at 962aab3 must both hash to the fixed SHA-256 above. Any failure
  aborts; the reason is recorded in the circuit file and results so far are kept. At start
  fhe_oracle/ must have no uncommitted changes, HEAD must contain 3705b7f, and the package must be
  imported from this checkout.
- Resume: a circuit file is extended only when its config, recorded start provenance and archived
  equivalence inputs equal the current ones and it was not aborted.
- ``--study`` fixes seeds 11-30, n_trials 200 and threshold 0.01 and writes to
  benchmarks/results/study_a/ by default (circuits may be split across processes). Any other
  invocation is labelled SMOKE and no criterion or decision is evaluated.
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
import shutil
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata, wilcoxon

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, THIS_DIR)

import fhe_oracle  # noqa: E402
import fhe_oracle.autoconfig as autoconfig  # noqa: E402
from fhe_oracle.fitness import absolute_error  # noqa: E402

warnings.filterwarnings("ignore", message="Could not import matplotlib")

SCRIPT_REL = "benchmarks/study_a_boundary_confirmation.py"
PREREG_REL = "benchmarks/preregistration_2026-09-15.md"
PREREG_PATH = os.path.join(ROOT, PREREG_REL)
PREREG_COMMIT = "962aab31e4239d074130a8f80394dac3e2731c1d"
PREREG_SHA256 = "912da47075386a07163dba517dfc06b4dfa9d24461fbb587b30e1c924a694386"
FEATURE_COMMIT = "3705b7f7da6264ad7c936641d93728430b367589"
GUARDED = ("fhe_oracle/core.py", "fhe_oracle/autoconfig.py", SCRIPT_REL)
RECORDED = ("benchmarks/tenseal_circuits.py", "fhe_oracle/adapters/tenseal_adapter.py",
            "fhe_oracle/fitness.py", "fhe_oracle/preactivation.py", "fhe_oracle/diagnostics.py")
CIRCUITS = ("lr_d8", "cheb_d10", "poly_d6", "bump_d8", "decoy_d6")
TENSEAL = ("lr_d8", "cheb_d10", "poly_d6")
VERTEX_SUP = ("lr_d8", "cheb_d10")
STUDY_SEEDS = tuple(range(11, 31))
N_TRIALS = 200
THRESHOLD = 0.01
MARGIN = 1e-5
MIN_NOT_SMALLER = 16
RATIO_MIN = 0.95
ALPHA = 0.05
MAX_CORNER_DIM = 12
SENTINEL = "__study_a_accounting_only__"
DEFAULT_STUDY_OUT = os.path.join(THIS_DIR, "results", "study_a")


# --- ACCOUNTING-ONLY arm ------------------------------------------------------

class _ProbeOffKwargs(dict):
    """oracle_kwargs whose get() reports the sentinel as set; the sentinel is never stored."""

    def get(self, key: Any, default: Any = None) -> Any:
        if key == SENTINEL:
            return True
        return super().get(key, default)


class probe_disabled:
    """Context manager: run ``ao`` down the committed no-pre-search branch (see header)."""

    def __init__(self, mod: Any, ao: Any, active: bool) -> None:
        self.mod, self.ao, self.active = mod, ao, active

    def __enter__(self) -> None:
        if not self.active:
            return
        self.saved = (self.mod._OTHER_OBJECTIVES, self.mod._RESERVE)
        self.mod._OTHER_OBJECTIVES = self.saved[0] + (SENTINEL,)
        self.mod._RESERVE = 2
        self.ao.oracle_kwargs = _ProbeOffKwargs(self.ao.oracle_kwargs)

    def __exit__(self, *exc: object) -> None:
        if self.active:
            self.mod._OTHER_OBJECTIVES, self.mod._RESERVE = self.saved


# --- Circuits -----------------------------------------------------------------

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))


def _taylor3(z):
    return 0.5 + z / 4.0 - z ** 3 / 48.0


def _cheb3(h):
    return 0.5 + 0.15 * h - h ** 3 / 500.0


def _affine_sup(w, b, bounds, surrogate):
    zmin = b + sum(min(wi * lo, wi * hi) for wi, (lo, hi) in zip(w, bounds))
    zmax = b + sum(max(wi * lo, wi * hi) for wi, (lo, hi) in zip(w, bounds))
    z = np.linspace(zmin, zmax, 1_000_001)
    err = np.abs(_sigmoid(z) - surrogate(z))
    return float(err.max())


def _bump_fhe():
    cen = np.linspace(-1.2, 1.2, 8)
    return lambda x: float(np.exp(-np.sum((np.asarray(x, dtype=float) - cen) ** 2) / 4.5))


def _decoy_fhe():
    cen = np.full(6, 0.7)

    def fhe(x):
        xa = np.asarray(x, dtype=float)
        return float(np.exp(-np.sum((xa - cen) ** 2) / 1.28) + 0.15 * (np.mean(xa) / 2 + 1))

    return fhe


def build(name: str) -> dict[str, Any]:
    if name in TENSEAL:
        import tenseal_circuits as tc

        from fhe_oracle.adapters.tenseal_adapter import TenSEALContext

        ctx = TenSEALContext()
        if name == "lr_d8":
            c = tc.build_tenseal_lr_d8(ctx)
            c["sup"] = _affine_sup(c["weights"], c["bias"], c["bounds"], _taylor3)
        elif name == "cheb_d10":
            c = tc.build_tenseal_chebyshev_d10(ctx)
            c["sup"] = max(_affine_sup(c["weights"][j], float(c["bias"][j]), c["bounds"], _cheb3)
                           for j in range(len(c["bias"])))
        else:
            c = tc.build_tenseal_circuit2(ctx)
            c["sup"] = None
        c["ctx"] = ctx
        return c
    if name == "bump_d8":
        return {"name": name, "plain": lambda x: 0.0, "fhe": _bump_fhe(), "d": 8,
                "bounds": [(-3.0, 3.0)] * 8, "sup": 1.0}
    if name == "decoy_d6":
        fhe = _decoy_fhe()
        opt = minimize(lambda x: -fhe(x), np.full(6, 0.7), bounds=[(-2.0, 2.0)] * 6)
        return {"name": name, "plain": lambda x: 0.0, "fhe": fhe, "d": 6,
                "bounds": [(-2.0, 2.0)] * 6, "sup": float(-opt.fun)}
    raise ValueError(f"unknown circuit {name!r}")


class Instrumented:
    """Records every FHE call: input, error (plaintext recomputed, not counted) and time."""

    def __init__(self, c: dict[str, Any]) -> None:
        self._plain, self._fhe = c["plain"], c["fhe"]
        self.lo = np.array([lo for lo, _ in c["bounds"]])
        self.hi = np.array([hi for _, hi in c["bounds"]])
        self.reset()

    def reset(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.oob = 0

    def fhe(self, x: Any) -> Any:
        xa = np.asarray(x, dtype=np.float64)
        if np.any(xa < self.lo) or np.any(xa > self.hi):
            self.oob += 1
        t0 = time.perf_counter()
        y = self._fhe(x)
        dt = time.perf_counter() - t0
        err = float(np.max(absolute_error(self._plain(x), y)))
        self.calls.append({"x": xa.ravel().tolist(), "err": err, "t": dt})
        return y


# --- Arms ---------------------------------------------------------------------

def run_auto(inst: Instrumented, c: dict[str, Any], seed: int, budget: int,
             accounting_only: bool) -> dict[str, Any]:
    inst.reset()
    events: list[dict[str, Any]] = []
    ao = autoconfig.AutoOracle(c["plain"], inst.fhe, c["bounds"], on_evaluation=events.append)
    t0 = time.perf_counter()
    with probe_disabled(autoconfig, ao, accounting_only):
        res = ao.run(n_trials=budget, seed=seed, threshold=THRESHOLD)
    wall = time.perf_counter() - t0
    calls = inst.calls
    if len(events) != len(calls) or any(e["x"] != k["x"] for e, k in zip(events, calls)):
        raise RuntimeError("evaluation log does not match the FHE call sequence")
    kinds = [e["kind"] for e in events]
    counted = [k["err"] for k, kind in zip(calls, kinds) if kind != "remeasure"]
    remeasure = kinds.count("remeasure")
    documented = remeasure == 1 if accounting_only else remeasure in (1, 2)
    if accounting_only and "boundary" in kinds:
        raise RuntimeError("ACCOUNTING-ONLY arm ran the boundary probe")
    return {
        "metric": max(counted), "max_error_all_calls": max(k["err"] for k in calls),
        "reported_max_error": float(res.max_error), "verdict": res.verdict,
        "worst_input": [float(v) for v in res.worst_input], "strategy": res.strategy_used,
        "regime": res.regime, "n_trials_reported": int(res.n_trials), "fhe_calls": len(calls),
        "remeasure_events": remeasure, "kind_counts": {k: kinds.count(k) for k in sorted(set(kinds))},
        "identity_ok": len(calls) == res.n_trials + remeasure,
        "within_prereg_bound": len(calls) <= budget + remeasure,
        "within_n_trials": len(calls) <= budget, "remeasure_documented": documented,
        "search_max_error": getattr(res, "search_max_error", None),
        "remeasured_error": getattr(res, "remeasured_error", None),
        "oob": inst.oob, "wall_s": wall, "fhe_s": float(sum(k["t"] for k in calls)),
    }


def random_points(c: dict[str, Any], seed: int, n: int) -> np.ndarray:
    lo = np.array([a for a, _ in c["bounds"]])
    hi = np.array([b for _, b in c["bounds"]])
    return np.random.default_rng([seed, 2]).uniform(lo, hi, size=(n, c["d"]))


def corner_points(c: dict[str, Any], seed: int, n: int) -> np.ndarray:
    d = c["d"]
    if d > MAX_CORNER_DIM:
        raise ValueError(f"corner pool enumerates all 2^d vertices; d={d} exceeds the supported "
                         f"d <= {MAX_CORNER_DIM}")
    lo = np.array([a for a, _ in c["bounds"]])
    hi = np.array([b for _, b in c["bounds"]])
    mid = (lo + hi) / 2.0
    extra = [mid.copy()]
    for i in range(d):
        for v in (lo[i], hi[i]):
            p = mid.copy()
            p[i] = v
            extra.append(p)
    verts = [np.array(v) for v in itertools.product(*zip(lo, hi))]
    pool = np.array(verts + extra)
    return pool[np.random.default_rng([seed, 1]).permutation(len(pool))][:n]


def run_points(inst: Instrumented, pts: np.ndarray) -> dict[str, Any]:
    inst.reset()
    t0 = time.perf_counter()
    for x in pts:
        inst.fhe(x.tolist())
    return {"metric": max(k["err"] for k in inst.calls), "fhe_calls": len(inst.calls),
            "oob": inst.oob, "wall_s": time.perf_counter() - t0,
            "fhe_s": float(sum(k["t"] for k in inst.calls))}


# --- Provenance ---------------------------------------------------------------

def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


def _git_bytes(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=True).stdout


def _sha256(rel: str) -> str:
    return _sha256_file(os.path.join(ROOT, rel))


def _sha256_file(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def guarded_state() -> dict[str, Any]:
    return {"git_head": _git("rev-parse", "HEAD"), "sha256": {r: _sha256(r) for r in GUARDED}}


def prereg_state() -> dict[str, Any]:
    """Hash the pre-registration file and its committed blob against the fixed constant."""
    try:
        file_sha: str | None = _sha256_file(PREREG_PATH)
    except OSError:
        file_sha = None
    try:
        committed: str | None = hashlib.sha256(
            _git_bytes("show", f"{PREREG_COMMIT}:{PREREG_REL}")).hexdigest()
    except subprocess.CalledProcessError:
        committed = None
    return {"path": PREREG_REL, "commit": PREREG_COMMIT, "expected_sha256": PREREG_SHA256,
            "file_sha256": file_sha, "committed_sha256": committed,
            "ok": file_sha == PREREG_SHA256 and committed == PREREG_SHA256}


def repo_checks() -> dict[str, Any]:
    pkg_dir = os.path.dirname(os.path.abspath(fhe_oracle.__file__))
    if pkg_dir != os.path.join(ROOT, "fhe_oracle"):
        raise RuntimeError(f"fhe_oracle imported from {pkg_dir}, not this checkout")
    if _git("status", "--porcelain", "--", "fhe_oracle"):
        raise RuntimeError("fhe_oracle/ has uncommitted changes; AFTER must be the committed code")
    if subprocess.run(["git", "merge-base", "--is-ancestor", FEATURE_COMMIT, "HEAD"],
                      cwd=ROOT).returncode != 0:
        raise RuntimeError(f"HEAD does not contain {FEATURE_COMMIT}")
    return {"feature_commit_in_head": True, "fhe_oracle_clean": True, "package_dir_ok": True}


def environment() -> dict[str, Any]:
    def ver(p: str) -> str | None:
        try:
            return md.version(p)
        except md.PackageNotFoundError:
            return None
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "cpu_count": os.cpu_count(), "loadavg": list(os.getloadavg()),
            "packages": {p: ver(p) for p in ("tenseal", "cma", "numpy", "scipy")},
            "recorded_sha256": {r: _sha256(r) for r in RECORDED},
            "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def _write_json(path: str, data: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=1)
    os.replace(tmp, path)


def _abort(data: dict[str, Any], path: str, status: str, **record: Any) -> None:
    data["status"] = status
    data.update(record)
    _write_json(path, data)
    raise RuntimeError(status)


def _require_prereg(data: dict[str, Any], path: str, when: str) -> dict[str, Any]:
    state = prereg_state()
    if not state["ok"]:
        _abort(data, path, f"aborted: pre-registration changed {when}", abort_prereg=state)
    return state


def archive_equivalence(ref_path: str, eq_path: str, out: str) -> dict[str, Any]:
    """Validate the equivalence inputs against the current code, then copy them with hashes."""
    with open(eq_path) as fh:
        eq = json.load(fh)
    ref_sha, eq_sha = _sha256_file(ref_path), _sha256_file(eq_path)
    if eq.get("status") != "complete" or eq.get("equivalent") is not True:
        raise RuntimeError(f"{eq_path}: the equivalence check did not complete and pass")
    if eq.get("reference_sha256") != ref_sha:
        raise RuntimeError(f"reference {ref_path} differs from the one the equivalence check used")
    current = guarded_state()["sha256"]
    for rel in GUARDED:
        if eq.get("guarded", {}).get("sha256", {}).get(rel) != current[rel]:
            raise RuntimeError(f"{rel} changed since the equivalence check")
    manifest = {"reference": {"file": os.path.basename(ref_path), "sha256": ref_sha},
                "equivalence_check": {"file": "equivalence_check.json", "sha256": eq_sha}}
    dest = os.path.join(out, "equivalence")
    man_path = os.path.join(dest, "manifest.json")
    if os.path.exists(man_path):
        with open(man_path) as fh:
            if json.load(fh) != manifest:
                raise RuntimeError(f"{man_path}: archived equivalence inputs differ from these")
    os.makedirs(dest, exist_ok=True)
    for key, src in (("reference", ref_path), ("equivalence_check", eq_path)):
        target = os.path.join(dest, manifest[key]["file"])
        shutil.copyfile(src, target)
        if _sha256_file(target) != manifest[key]["sha256"]:
            raise RuntimeError(f"copy of {src} does not match its SHA-256")
    _write_json(man_path, manifest)
    return manifest


# --- Run ----------------------------------------------------------------------

def run_circuit(name: str, seeds: list[int], budget: int, out: str, study: bool,
                archive: dict[str, Any] | None = None) -> dict[str, Any]:
    start = guarded_state()
    path = os.path.join(out, f"{name}.json")
    config = {"circuit": name, "budget": budget, "threshold": THRESHOLD, "study": study,
              "seeds": list(seeds)}
    if os.path.exists(path):
        with open(path) as fh:
            data = json.load(fh)
        if (data["config"] != config or data["start"]["guarded"] != start
                or data["start"].get("equivalence_archive") != archive):
            raise RuntimeError(f"{path}: config, provenance or equivalence inputs differ; "
                               "refusing to extend it")
        if str(data.get("status", "")).startswith("aborted"):
            raise RuntimeError(f"{path}: previous run aborted ({data['status']}); refusing to extend it")
        repo_checks()
        when = "on resume"
    else:
        data = {"config": config, "start": {"guarded": start, "checks": repo_checks(),
                                            "prereg": prereg_state(), "equivalence_archive": archive,
                                            "environment": environment()},
                "seeds": {}, "status": "running"}
        when = "at start"
    _require_prereg(data, path, when)
    c = build(name)
    inst = Instrumented(c)
    data["sup"] = c["sup"]
    for seed in seeds:
        if str(seed) in data["seeds"]:
            continue
        if guarded_state() != start:
            _abort(data, path, f"aborted: provenance changed before seed {seed}",
                   abort_state=guarded_state())
        _require_prereg(data, path, f"before seed {seed}")
        t0 = time.perf_counter()
        row: dict[str, Any] = {"after": run_auto(inst, c, seed, budget, False),
                               "accounting": run_auto(inst, c, seed, budget, True)}
        n_base = row["after"]["n_trials_reported"]
        row["baseline_n"] = n_base
        row["random"] = run_points(inst, random_points(c, seed, n_base))
        row["corner"] = run_points(inst, corner_points(c, seed, n_base))
        row["seed_wall_s"] = time.perf_counter() - t0
        data["seeds"][str(seed)] = row
        _write_json(path, data)
        print(f"{name} seed {seed}: after={row['after']['metric']:.6g}[{row['after']['fhe_calls']}] "
              f"acct={row['accounting']['metric']:.6g}[{row['accounting']['fhe_calls']}] "
              f"random={row['random']['metric']:.6g} corner={row['corner']['metric']:.6g}"
              f"[{row['corner']['fhe_calls']}] ({row['seed_wall_s']:.1f}s)", flush=True)
    end, end_prereg = guarded_state(), prereg_state()
    data["end"] = {"guarded": end, "prereg": end_prereg, "environment": environment()}
    if end != start:
        _abort(data, path, "aborted: provenance changed during run")
    if not end_prereg["ok"]:
        _abort(data, path, "aborted: pre-registration changed during run", abort_prereg=end_prereg)
    data["status"] = "complete"
    _write_json(path, data)
    return data


# --- Statistics and report ----------------------------------------------------

def paired(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    d = a - b
    ratios = np.divide(a, b, out=np.where(a > 0, np.inf, 1.0), where=b > 0)
    nz = d != 0
    out = {"median_ratio": float(np.median(ratios)), "min_ratio": float(ratios.min()),
           "max_ratio": float(ratios.max()), "wins": int(np.sum(d > 0)),
           "losses": int(np.sum(d < 0)), "ties": int(np.sum(~nz)), "p": 1.0,
           "w_pos": 0.0, "w_neg": 0.0}
    if nz.any():
        out["p"] = float(wilcoxon(a, b, zero_method="wilcox", alternative="two-sided").pvalue)
        ranks = rankdata(np.abs(d[nz]))
        out["w_pos"] = float(ranks[d[nz] > 0].sum())
        out["w_neg"] = float(ranks[d[nz] < 0].sum())
    out["after_lower"] = out["w_neg"] > out["w_pos"]
    return out


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    adj = [0.0] * len(pvals)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def is_study(datas: list[dict[str, Any]]) -> bool:
    """True only for the complete pre-registered set under one unchanged provenance."""
    names = [d.get("config", {}).get("circuit") for d in datas]
    if sorted(names) != sorted(CIRCUITS):
        return False
    starts = [d.get("start", {}).get("guarded") for d in datas]
    if any(s is None or s != starts[0] for s in starts):
        return False
    for d in datas:
        cfg, start, end = d.get("config", {}), d.get("start", {}), d.get("end", {})
        if not (cfg.get("study") is True and cfg.get("budget") == N_TRIALS
                and cfg.get("threshold") == THRESHOLD and cfg.get("seeds") == list(STUDY_SEEDS)
                and sorted(int(s) for s in d.get("seeds", {})) == list(STUDY_SEEDS)
                and d.get("status") == "complete" and end.get("guarded") == start.get("guarded")
                and start.get("prereg", {}).get("ok") is True and end.get("prereg", {}).get("ok") is True):
            return False
    return True


def _describe(data: dict[str, Any]) -> dict[str, Any]:
    name = data["config"]["circuit"]
    seeds = sorted(int(s) for s in data["seeds"])
    s: dict[str, Any] = {"circuit": name, "seeds": seeds, "n": len(seeds), "sup": data.get("sup"),
                         "budget": data["config"]["budget"], "status": data.get("status"),
                         "c1": {"p_holm": None}, "c2_applies": name in VERTEX_SUP,
                         "c1_pass": None, "c2_pass": None, "c3_pass": None, "pass": None}
    if not seeds:
        return s
    rows = [data["seeds"][str(x)] for x in seeds]

    def col(arm: str, key: str = "metric") -> np.ndarray:
        return np.array([r[arm][key] for r in rows], dtype=float)

    after, acct, corner = col("after"), col("accounting"), col("corner")
    s["c1"] = dict(paired(after, acct), p_holm=None)
    s["c2_not_smaller"] = int(np.sum(after >= corner - MARGIN))
    s["c2_not_smaller_strict"] = int(np.sum(after >= corner))
    s["c3"] = {arm: {
        "identity_ok": all(r[arm]["identity_ok"] for r in rows),
        "within_prereg_bound": all(r[arm]["within_prereg_bound"] for r in rows),
        "remeasure_documented": all(r[arm]["remeasure_documented"] for r in rows),
        "within_n_trials": all(r[arm]["within_n_trials"] for r in rows),
        "calls": [int(col(arm, "fhe_calls").min()), int(col(arm, "fhe_calls").max())],
        "remeasure_events": sorted({r[arm]["remeasure_events"] for r in rows}),
    } for arm in ("after", "accounting")}
    arms = ("after", "accounting", "random", "corner")
    s["median_metric"] = {a: float(np.median(col(a))) for a in arms}
    s["calls"] = {a: [int(col(a, "fhe_calls").min()), int(col(a, "fhe_calls").max())] for a in arms}
    s["after_vs_corner"] = paired(after, corner)
    s["after_vs_random"] = paired(after, col("random"))
    s["strategies"] = {a: sorted({r[a]["strategy"] for r in rows}) for a in ("after", "accounting")}
    s["oob"] = int(sum(r[a]["oob"] for r in rows for a in arms))
    s["compute_s"] = float(sum(r["seed_wall_s"] for r in rows))
    s["fhe_ms_median"] = float(1e3 * np.median([r[a]["fhe_s"] / r[a]["fhe_calls"]
                                                for r in rows for a in arms]))
    return s


def summarize(datas: list[dict[str, Any]]) -> dict[str, Any]:
    study = is_study(datas)
    circuits = [_describe(d) for d in datas]
    summary: dict[str, Any] = {
        "study": study, "holm_family": [], "holm_family_size": 0, "decision": None,
        "prereg": {"path": PREREG_REL, "commit": PREREG_COMMIT, "sha256": PREREG_SHA256},
        "circuits": circuits,
    }
    if not study:
        return summary
    by = {s["circuit"]: s for s in circuits}
    for name, p in zip(CIRCUITS, holm([by[n]["c1"]["p"] for n in CIRCUITS])):
        s = by[name]
        s["c1"]["p_holm"] = p
        s["c1_pass"] = bool(s["c1"]["median_ratio"] >= RATIO_MIN
                            and not (p < ALPHA and s["c1"]["after_lower"]))
        s["c2_pass"] = bool(s["c2_not_smaller"] >= MIN_NOT_SMALLER) if s["c2_applies"] else None
        s["c3_pass"] = all(v["identity_ok"] and v["within_prereg_bound"] and v["remeasure_documented"]
                           for v in s["c3"].values())
        s["pass"] = bool(s["c1_pass"] and s["c3_pass"] and s["c2_pass"] is not False)
    summary.update(holm_family=list(CIRCUITS), holm_family_size=len(CIRCUITS),
                   decision="keep" if all(by[n]["pass"] for n in CIRCUITS) else "revert")
    return summary


def _choices_text() -> str:
    doc = __doc__ or ""
    marker = "CHOICES WHERE THE PRE-REGISTRATION IS NOT EXPLICIT"
    return doc.split(marker, 1)[1].split("\n", 1)[1].strip("\n") if marker in doc else ""


def report(datas: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    study = summary["study"]

    def mark(v: Any) -> str:
        return "-" if v is None else ("PASS" if v else "FAIL")

    L = ["# Study A: AutoOracle boundary-probe confirmation", ""]
    if not study:
        L += ["**SMOKE / INCOMPLETE: not study A results. No criterion, Holm correction or decision "
              f"is evaluated ({len(datas)} circuit file(s) present).**", ""]
    L += [f"Script `{SCRIPT_REL}`; pre-registration `{PREREG_REL}` at `{PREREG_COMMIT}` "
          f"(SHA-256 `{PREREG_SHA256}`).", "",
          f"Holm family: {', '.join(summary['holm_family']) or 'not computed'} "
          f"(size {summary['holm_family_size']}).", "",
          "| Circuit | seeds | C1 median after/acct (range) | W/L/T | p | Holm p | AFTER lower | C1 | "
          "C2 after >= corner - 1e-5 | C2 | C3 calls after / acct (remeasure events) | C3 | circuit |",
          "|---|---:|---|---|---:|---:|---|---|---|---|---|---|---|"]
    for s in summary["circuits"]:
        if s["n"] == 0:
            L.append(f"| {s['circuit']} | 0 | no seeds ({s['status']}) | | | | | - | | - | | - | - |")
            continue
        c1, c3 = s["c1"], s["c3"]
        holm_p = "-" if c1["p_holm"] is None else f"{c1['p_holm']:.3g}"
        c2 = (f"{s['c2_not_smaller']}/{s['n']} (strict {s['c2_not_smaller_strict']})"
              if s["c2_applies"] else "n/a")
        L.append(f"| {s['circuit']} | {s['n']} | {c1['median_ratio']:.4f} ({c1['min_ratio']:.4f}-"
                 f"{c1['max_ratio']:.4f}) | {c1['wins']}/{c1['losses']}/{c1['ties']} | {c1['p']:.3g} | "
                 f"{holm_p} | {c1['after_lower']} | {mark(s['c1_pass'])} | {c2} | {mark(s['c2_pass'])} | "
                 f"{c3['after']['calls']} / {c3['accounting']['calls']} "
                 f"({c3['after']['remeasure_events']} / {c3['accounting']['remeasure_events']}) | "
                 f"{mark(s['c3_pass'])} | {mark(s['pass'])} |")
    L.append("")
    if study:
        keep = summary["decision"] == "keep"
        L += [f"**Decision (pre-registered rule): {'KEEP the boundary probe' if keep else 'REVERT the boundary probe, keep the accounting fix'}.**", ""]
    L += ["## Median per-seed metric and counted evaluations", "",
          "| Circuit | supremum | after | accounting | random | corner | calls after | calls acct | "
          "random evals | corner evals | median ms/FHE eval | compute s |",
          "|---|---:|---:|---:|---:|---:|---|---|---|---|---:|---:|"]
    for s in summary["circuits"]:
        if s["n"] == 0:
            continue
        m, k = s["median_metric"], s["calls"]
        L.append(f"| {s['circuit']} | {s['sup'] if s['sup'] is not None else 'unknown'} | "
                 f"{m['after']:.6g} | {m['accounting']:.6g} | {m['random']:.6g} | {m['corner']:.6g} | "
                 f"{k['after']} | {k['accounting']} | {k['random']} | {k['corner']} | "
                 f"{s['fhe_ms_median']:.2f} | {s['compute_s']:.0f} |")
    L += ["", "## Per-seed values", ""]
    for data, s in zip(datas, summary["circuits"]):
        if s["n"] == 0:
            continue
        L += [f"### {s['circuit']} (status: {s['status']}; strategies after {s['strategies']['after']}, "
              f"acct {s['strategies']['accounting']}; out-of-bounds {s['oob']})", "",
              "| seed | after [calls, remeasure] | accounting [calls, remeasure] | random [n] | corner [n] |",
              "|---:|---:|---:|---:|---:|"]
        for seed in s["seeds"]:
            r = data["seeds"][str(seed)]
            L.append(f"| {seed} | {r['after']['metric']:.6g} [{r['after']['fhe_calls']}, "
                     f"{r['after']['remeasure_events']}] | {r['accounting']['metric']:.6g} "
                     f"[{r['accounting']['fhe_calls']}, {r['accounting']['remeasure_events']}] | "
                     f"{r['random']['metric']:.6g} [{r['random']['fhe_calls']}] | "
                     f"{r['corner']['metric']:.6g} [{r['corner']['fhe_calls']}] |")
        L.append("")
    L += ["## Provenance", "",
          "| Circuit | status | start HEAD | end HEAD | core.py | autoconfig.py | script | "
          "pre-registration ok start / end | equivalence reference |",
          "|---|---|---|---|---|---|---|---|---|"]
    for data in datas:
        st, en = data["start"], data.get("end", {})
        arch = st.get("equivalence_archive") or {}
        ref = arch.get("reference", {}).get("sha256", "")[:12] or "not archived"
        L.append(f"| {data['config']['circuit']} | {data.get('status')} | `{st['guarded']['git_head'][:12]}` | "
                 f"`{en.get('guarded', {}).get('git_head', 'n/a')[:12]}` | "
                 + " | ".join(f"`{st['guarded']['sha256'][r][:12]}`" for r in GUARDED)
                 + f" | {st.get('prereg', {}).get('ok')} / {en.get('prereg', {}).get('ok')} | {ref} |")
    L += ["", "## Choices where the pre-registration is not explicit", "", _choices_text(), ""]
    return "\n".join(L)


# --- Equivalence check ----------------------------------------------------------

def _standins() -> dict[str, tuple]:
    from tenseal_circuits import _circuit2_plaintext_fn, _fit_lr_synthetic

    w, b = _fit_lr_synthetic(8, 42)
    rng = np.random.default_rng(123)
    Wc, bc = rng.standard_normal((4, 10)) * 0.5, rng.standard_normal(4) * 0.1
    k6 = np.arange(1, 7)
    Wr = np.random.default_rng(3).standard_normal((2, 16))
    pre = {k: (np.random.default_rng(0).normal(size=(k, 8)), np.random.default_rng(1).normal(size=k))
           for k in (1, 2)}

    def plateau():
        vals = np.concatenate([np.full(47, 0.098), [0.30, 0.34, 0.36]])
        np.random.RandomState(7).shuffle(vals)
        state = {"i": 0}

        def fhe(x):
            state["i"] += 1
            return float(vals[(state["i"] - 1) % vals.size])
        return fhe

    def preact(k):
        W, bb = pre[k]
        return lambda x: float(np.max(np.abs(W @ np.asarray(x) + bb)) ** 3 / 10.0)

    zero = lambda x: 0.0  # noqa: E731
    small = [(50, 200), (50, 52), (2, 4)]
    return {
        # name: (plain, fhe factory, bounds, AutoOracle kwargs, [(n_probes, n_trials)])
        "lr_d8": (lambda x: float(_sigmoid(w @ np.asarray(x) + b)),
                  lambda: (lambda x: float(_taylor3(w @ np.asarray(x) + b))), [(-3.0, 3.0)] * 8, {}, small),
        "cheb_d10": (lambda x: _sigmoid(Wc @ np.asarray(x) + bc).tolist(),
                     lambda: (lambda x: _cheb3(Wc @ np.asarray(x) + bc).tolist()), [(-3.0, 3.0)] * 10, {},
                     small),
        "poly_d6": (_circuit2_plaintext_fn,
                    lambda: (lambda x: _circuit2_plaintext_fn(x) + 1e-6 * (1 + abs(_circuit2_plaintext_fn(x)))
                             * np.sin(1e4 * float(k6 @ np.asarray(x)))), [(-2.0, 2.0)] * 6, {}, small),
        "bump_d8": (zero, _bump_fhe, [(-3.0, 3.0)] * 8, {}, small),
        "decoy_d6": (zero, _decoy_fhe, [(-2.0, 2.0)] * 6, {}, small),
        "saturated_d3": (zero, lambda: (lambda x: 2.0), [(-1.0, 1.0)] * 3, {}, [(20, 60), (2, 4)]),
        "plateau_d5": (zero, plateau, [(-3.0, 3.0)] * 5, {}, [(50, 100)]),
        "shell_d2": (zero, lambda: (lambda x: 5.0 if abs(x[0]) > 15.0 else 0.0), [(-20.0, 20.0)] * 2, {},
                     [(30, 80)]),
        "low_rank_d16": (zero, lambda: (lambda x: 0.01 * float(np.sum(np.sin(Wr @ np.asarray(x))))),
                         [(-1.0, 1.0)] * 16, {}, [(30, 6600)]),
        "preact_k1": (zero, lambda: preact(1), [(-3.0, 3.0)] * 8, {"W": pre[1][0], "b": pre[1][1]},
                      [(50, 200), (50, 60)]),
        "preact_k2": (zero, lambda: preact(2), [(-3.0, 3.0)] * 8, {"W": pre[2][0], "b": pre[2][1]},
                      [(50, 200), (50, 52)]),
    }


def _trace(mod: Any, plain: Any, make_fhe: Any, bounds: Any, kw: dict, n_probes: int, n_trials: int,
           seed: int, accounting_only: bool, log: bool) -> dict[str, Any]:
    fhe_fn = make_fhe()
    xs: list[list[float]] = []

    def fhe(x):
        xs.append(np.asarray(x, dtype=np.float64).ravel().tolist())
        return fhe_fn(x)

    events: list[dict[str, Any]] = []
    extra = {"on_evaluation": events.append} if log else {}
    ao = mod.AutoOracle(plain, fhe, bounds, n_probes=n_probes, **kw, **extra)
    with probe_disabled(mod, ao, accounting_only):
        res = ao.run(n_trials=n_trials, seed=seed, threshold=THRESHOLD)
    key = (float(res.max_error), tuple(float(v) for v in res.worst_input), res.verdict, int(res.n_trials),
           getattr(res, "search_max_error", None), getattr(res, "remeasured_error", None),
           getattr(res, "strategy_used", None), getattr(res, "regime", None))
    kinds = [e["kind"] for e in events]
    return {"xs": xs, "key": key, "events_match": (not log) or (
        len(events) == len(xs) and [e["x"] for e in events] == xs
        and len(kinds) - kinds.count("remeasure") == res.n_trials), "kinds": kinds}


def equivalence_check(ref_path: str, out: str, seeds: list[int]) -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location("fhe_oracle._study_a_reference", ref_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load the reference autoconfig module from {ref_path!r}")
    ref_sha = _sha256_file(ref_path)
    ref = importlib.util.module_from_spec(spec)
    ref.__package__ = "fhe_oracle"
    sys.modules[spec.name] = ref
    spec.loader.exec_module(ref)
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "equivalence_check.json")
    result: dict[str, Any] = {
        "reference_path": os.path.basename(ref_path), "reference_sha256": ref_sha,
        "guarded": guarded_state(), "seeds": seeds, "status": "running", "equivalent": None,
        "rows": [], "utc_start": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    _write_json(path, result)
    saved = (autoconfig._OTHER_OBJECTIVES, autoconfig._RESERVE)
    for name, (plain, make_fhe, bounds, kw, budgets) in _standins().items():
        for n_probes, n_trials in budgets:
            for seed in seeds:
                a = _trace(ref, plain, make_fhe, bounds, kw, n_probes, n_trials, seed, False, False)
                b = _trace(autoconfig, plain, make_fhe, bounds, kw, n_probes, n_trials, seed, True, True)
                row = {"case": name, "n_probes": n_probes, "n_trials": n_trials, "seed": seed,
                       "same_inputs": a["xs"] == b["xs"], "same_result_and_counts": a["key"] == b["key"],
                       "n_calls": len(b["xs"]), "events_match": b["events_match"],
                       "boundary_events": b["kinds"].count("boundary"),
                       "remeasure_events": b["kinds"].count("remeasure")}
                if n_trials >= n_probes + autoconfig._RESERVE_1D_PREACT:
                    after = _trace(autoconfig, plain, make_fhe, bounds, kw, n_probes, n_trials, seed,
                                   False, False)
                    row["after_differs"] = after["xs"] != a["xs"] or after["key"] != a["key"]
                result["rows"].append(row)
                _write_json(path, result)
    rows = result["rows"]
    restored = (autoconfig._OTHER_OBJECTIVES, autoconfig._RESERVE) == saved
    result.update({
        "runs": len(rows),
        "identical_runs": sum(r["same_inputs"] and r["same_result_and_counts"] for r in rows),
        "events_match_runs": sum(r["events_match"] for r in rows),
        "runs_where_after_differs": sum(bool(r.get("after_differs")) for r in rows),
        "runs_with_after": sum("after_differs" in r for r in rows),
        "globals_restored": restored,
        "equivalent": all(r["same_inputs"] and r["same_result_and_counts"] and r["events_match"]
                          and r["boundary_events"] == 0 for r in rows) and restored,
        "status": "complete", "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    _write_json(path, result)
    return result


# --- Main -----------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Study A: AutoOracle boundary-probe confirmation")
    ap.add_argument("--study", action="store_true", help="pre-registered run: seeds 11-30, n_trials 200")
    ap.add_argument("--circuits", default=",".join(CIRCUITS))
    ap.add_argument("--seeds", default=None, help="comma-separated (smoke only)")
    ap.add_argument("--budget", type=int, default=None, help="n_trials (smoke only)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--report", action="store_true", help="summarise existing JSON only")
    ap.add_argument("--check-equivalence", metavar="REFERENCE_AUTOCONFIG", default=None)
    ap.add_argument("--reference", default=None, help="saved accounting-only module (archived)")
    ap.add_argument("--equivalence-json", default=None, help="equivalence_check.json (archived)")
    args = ap.parse_args(argv)
    names = [n for n in args.circuits.split(",") if n]
    if any(n not in CIRCUITS for n in names):
        ap.error(f"circuits must be among {CIRCUITS}")
    if bool(args.reference) != bool(args.equivalence_json):
        ap.error("--reference and --equivalence-json go together")
    if args.study:
        if args.seeds is not None or args.budget is not None:
            ap.error("--study fixes seeds and budget")
        if not args.report and not args.reference:
            ap.error("--study needs --reference and --equivalence-json (archived with the results)")
        seeds, budget = list(STUDY_SEEDS), N_TRIALS
        out = args.out or DEFAULT_STUDY_OUT
    else:
        seeds = [int(s) for s in (args.seeds or "0,1").split(",")]
        budget = args.budget or 60
        if args.out is None:
            ap.error("smoke runs need --out")
        if set(seeds) & set(STUDY_SEEDS) and not args.report:
            ap.error("seeds 11-30 are reserved for --study")
        out = args.out
    os.makedirs(out, exist_ok=True)
    if args.check_equivalence:
        res = equivalence_check(args.check_equivalence, out, seeds)
        print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))
        return 0 if res["equivalent"] else 1
    if not args.report:
        archive = (archive_equivalence(args.reference, args.equivalence_json, out)
                   if args.reference else None)
        for name in names:
            run_circuit(name, seeds, budget, out, args.study, archive)
    datas = []
    for name in CIRCUITS:
        p = os.path.join(out, f"{name}.json")
        if os.path.exists(p):
            with open(p) as fh:
                datas.append(json.load(fh))
    summary = summarize(datas)
    text = report(datas, summary)
    with open(os.path.join(out, "report.md"), "w") as fh:
        fh.write(text)
    _write_json(os.path.join(out, "summary.json"), summary)
    print(text.split("## Median")[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
