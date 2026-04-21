# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Empirical estimation of FHE divergence landscape properties (P1-P3, Q).

Supports the theory document at
``research/theory/cma-es-conditional-guarantee.md``.

For each benchmark circuit we sample plaintext-only inputs and estimate:

- **L** : 95th/99th-percentile Lipschitz slope on random pairs
  outside an empirical bug region, proxy for the Lipschitz
  constant of Property P1.
- **rho_A** : fraction of uniform samples x at which the numerical
  gradient ``grad delta(x)`` points toward an empirical bug witness
  (cosine similarity > 0), proxy for attractor-basin measure P2.
- **kappa** : median finite-difference Hessian condition number
  inside the attractor basin, proxy for P3.
- **Q_frac** : fraction of uniform samples where ``||grad delta||``
  is below a noise floor (plateau region). Large Q_frac implies
  Property Q (misleading gradient).

Everything is plaintext - no TenSEAL required. A single run across
four circuits finishes in under two minutes on a laptop. Output:
``benchmarks/results/landscape_properties.csv``.
"""

from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from logistic_regression import _fit_logistic  # noqa: E402


def _mock_lr_fhe_factory(plaintext_fn):
    """Mock CKKS-style noise with adversarial hot zone.

    Copied verbatim from ``logistic_regression._build_fhe_fn`` so the
    landscape estimator does not fall back to the concrete-ml stub
    (which returns plain identically when concrete-ml is installed,
    making delta = 0 and defeating the measurement).
    """

    def mock_fhe_fn(x):
        arr = np.asarray(x, dtype=np.float64)
        plain = plaintext_fn(x)
        seed = int(abs(hash(tuple(round(v, 9) for v in arr))) % (2**31))
        local = np.random.default_rng(seed)
        noise = float(local.normal(0.0, 1e-4))
        z_proxy = float(np.dot(arr, arr))
        if z_proxy > 4.0 and abs(plain - 0.5) < 0.25:
            amp = 1.0 + 50.0 * (z_proxy - 4.0)
            noise *= amp
        return plain + noise

    return mock_fhe_fn
from polynomial_eval import mock_fhe_poly, plaintext_poly  # noqa: E402
from chebyshev_polynomials import (  # noqa: E402
    eval_poly_plaintext,
    fit_cheb_sigmoid,
    taylor3_approx,
)


# ----------------------------- circuit builders -------------------------------


@dataclass(frozen=True)
class CircuitSpec:
    name: str
    d: int
    bounds: list[tuple[float, float]]
    plaintext_fn: Callable[[list[float]], float]
    fhe_fn: Callable[[list[float]], float]


def _lr_circuit() -> CircuitSpec:
    rng = np.random.default_rng(42)
    d = 8
    n = 200
    X = rng.normal(0.0, 1.0, size=(n, d))
    w_true = rng.normal(0.0, 1.0, size=d)
    y = (X @ w_true > 0).astype(int)
    w, b = _fit_logistic(X, y, rng)

    def plaintext_fn(x):
        z = float(np.dot(w, x) + b)
        return 1.0 / (1.0 + np.exp(-z))

    fhe_fn = _mock_lr_fhe_factory(plaintext_fn)
    return CircuitSpec(
        name="LR_d8_mock",
        d=d,
        bounds=[(-3.0, 3.0)] * d,
        plaintext_fn=plaintext_fn,
        fhe_fn=fhe_fn,
    )


def _poly_circuit() -> CircuitSpec:
    return CircuitSpec(
        name="Poly_d6_mock",
        d=6,
        bounds=[(-2.0, 2.0)] * 6,
        plaintext_fn=plaintext_poly,
        fhe_fn=mock_fhe_poly,
    )


def _cheb_circuit() -> CircuitSpec:
    """Chebyshev plateau-then-cliff analogue (plaintext only).

    Reproduces the paper's Cheb-15 mechanism without TenSEAL: the
    CKKS noise is replaced by the Cheb-15-vs-sigmoid extrapolation
    error outside the [-5, 5] fit domain. Inside the fit interval,
    delta ~1e-6; just outside, delta grows as z^15. That is the
    plateau-then-cliff geometry that causes 0/10 wins.
    """
    rng = np.random.default_rng(11)
    d = 10
    w = rng.normal(0.0, 1.0, size=d)
    w = w / float(np.linalg.norm(w))  # unit weight vector
    # Bounds [-3, 3]^10 -> |z| <= 3*sqrt(10) ~ 9.5, so we reach the cliff.
    bounds = [(-3.0, 3.0)] * d
    approx = fit_cheb_sigmoid(15, domain=(-5.0, 5.0))

    def plaintext_fn(x):
        z = float(np.dot(w, x))
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def fhe_fn(x):
        z = float(np.dot(w, x))
        return eval_poly_plaintext(z, approx)

    return CircuitSpec(
        name="Cheb15_d10_plateau",
        d=d,
        bounds=bounds,
        plaintext_fn=plaintext_fn,
        fhe_fn=fhe_fn,
    )


def _wdbc_circuit() -> CircuitSpec:
    """Taylor-3 WDBC analogue. Skip if sklearn absent; returns None."""
    try:
        from wdbc_mock import build_wdbc_circuit
    except Exception:
        return None  # type: ignore
    try:
        plaintext_fn, fhe_fn, data, _w, _b, d = build_wdbc_circuit(42)
    except Exception:
        return None  # type: ignore
    # Use the data's per-coordinate min/max as box bounds.
    lows = data.min(axis=0).astype(float)
    highs = data.max(axis=0).astype(float)
    bounds = [(float(lo), float(hi)) for lo, hi in zip(lows, highs)]
    return CircuitSpec(
        name="WDBC_d30_taylor3",
        d=d,
        bounds=bounds,
        plaintext_fn=plaintext_fn,
        fhe_fn=fhe_fn,
    )


# ----------------------------- landscape estimators ---------------------------


def divergence(circuit: CircuitSpec, x: np.ndarray) -> float:
    return abs(
        float(circuit.plaintext_fn(x.tolist()))
        - float(circuit.fhe_fn(x.tolist()))
    )


def _sample_uniform(
    circuit: CircuitSpec, n: int, rng: np.random.Generator
) -> np.ndarray:
    lows = np.array([lo for lo, _ in circuit.bounds])
    highs = np.array([hi for _, hi in circuit.bounds])
    return rng.uniform(lows, highs, size=(n, circuit.d))


def _locate_bug_witness(
    circuit: CircuitSpec, n_seed: int, rng: np.random.Generator
) -> tuple[np.ndarray, float]:
    """Best-of-n-seed uniform sample; returns (witness, delta(witness))."""
    samples = _sample_uniform(circuit, n_seed, rng)
    best_x = samples[0]
    best_d = divergence(circuit, best_x)
    for x in samples[1:]:
        d = divergence(circuit, x)
        if d > best_d:
            best_d = d
            best_x = x
    return best_x, best_d


def _grad_fd(
    circuit: CircuitSpec, x: np.ndarray, h: float = 1e-3
) -> np.ndarray:
    """Central-difference gradient of delta(.) at x."""
    d = x.size
    g = np.zeros(d)
    for i in range(d):
        xp = x.copy()
        xm = x.copy()
        xp[i] += h
        xm[i] -= h
        g[i] = (divergence(circuit, xp) - divergence(circuit, xm)) / (2.0 * h)
    return g


def _hessian_fd(
    circuit: CircuitSpec, x: np.ndarray, h: float = 5e-3
) -> np.ndarray:
    """Central-difference Hessian of delta at x (d x d, symmetric)."""
    d = x.size
    H = np.zeros((d, d))
    d0 = divergence(circuit, x)
    for i in range(d):
        for j in range(i, d):
            if i == j:
                xp = x.copy(); xp[i] += h
                xm = x.copy(); xm[i] -= h
                H[i, i] = (
                    divergence(circuit, xp) - 2.0 * d0 + divergence(circuit, xm)
                ) / (h * h)
            else:
                xpp = x.copy(); xpp[i] += h; xpp[j] += h
                xpm = x.copy(); xpm[i] += h; xpm[j] -= h
                xmp = x.copy(); xmp[i] -= h; xmp[j] += h
                xmm = x.copy(); xmm[i] -= h; xmm[j] -= h
                H[i, j] = (
                    divergence(circuit, xpp)
                    - divergence(circuit, xpm)
                    - divergence(circuit, xmp)
                    + divergence(circuit, xmm)
                ) / (4.0 * h * h)
                H[j, i] = H[i, j]
    return H


def estimate_landscape(
    circuit: CircuitSpec,
    n_witness_seed: int = 2000,
    n_pairs: int = 1000,
    n_basin_probe: int = 400,
    n_hessian_probe: int = 20,
    grad_norm_floor_quantile: float = 0.20,
    seed: int = 0,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)

    witness, delta_witness = _locate_bug_witness(circuit, n_witness_seed, rng)

    # P1: Lipschitz slope on random pairs (both endpoints outside bug region).
    pair_a = _sample_uniform(circuit, n_pairs, rng)
    pair_b = _sample_uniform(circuit, n_pairs, rng)
    lipschitz_ratios: list[float] = []
    tau = 0.5 * delta_witness  # treat top-half-of-max as bug region
    for xa, xb in zip(pair_a, pair_b):
        da = divergence(circuit, xa)
        db = divergence(circuit, xb)
        if da >= tau or db >= tau:
            continue
        dist = float(np.linalg.norm(xa - xb))
        if dist < 1e-9:
            continue
        lipschitz_ratios.append(abs(da - db) / dist)
    if lipschitz_ratios:
        L_p95 = float(np.percentile(lipschitz_ratios, 95))
        L_p99 = float(np.percentile(lipschitz_ratios, 99))
        L_max = float(np.max(lipschitz_ratios))
    else:
        L_p95 = L_p99 = L_max = float("nan")

    # P2: attractor basin measure - cos similarity of grad delta with
    # direction to witness, plus a gradient-norm floor so we do not
    # count plateau noise.
    probes = _sample_uniform(circuit, n_basin_probe, rng)
    cos_pos = 0
    grad_norms: list[float] = []
    for x in probes:
        g = _grad_fd(circuit, x)
        grad_norms.append(float(np.linalg.norm(g)))
    grad_norms_arr = np.asarray(grad_norms)
    # Floor used only to filter basin probes (avoid counting plateau noise
    # as valid direction signal). Defaults to 20th-percentile of grad norm.
    norm_floor = float(np.quantile(grad_norms_arr, grad_norm_floor_quantile))
    # Q diagnostic: fraction of probes whose gradient norm, rescaled to
    # a witness-scaled "meaningful signal" floor, is negligible. The
    # floor is ``0.01 * witness_delta / domain_radius`` so a gradient
    # that would take >= 100 domain-radii-worth of steps to deliver the
    # witness magnitude counts as plateau. Independent of rho_A.
    lows_arr = np.array([lo for lo, _ in circuit.bounds])
    highs_arr = np.array([hi for _, hi in circuit.bounds])
    domain_radius = float(np.linalg.norm(highs_arr - lows_arr))
    witness_floor = 0.01 * delta_witness / max(domain_radius, 1e-12)
    plateau_frac = float(np.mean(grad_norms_arr <= witness_floor))
    grad_norm_median = float(np.median(grad_norms_arr))

    basin_points: list[np.ndarray] = []
    for x, gn in zip(probes, grad_norms):
        direction = witness - x
        nrm = float(np.linalg.norm(direction))
        if nrm < 1e-9 or gn < norm_floor:
            continue
        g = _grad_fd(circuit, x)  # re-evaluate - cheap at d up to 30
        direction /= nrm
        g_nrm = float(np.linalg.norm(g))
        if g_nrm < 1e-15:
            continue
        cos = float(np.dot(g, direction) / g_nrm)
        if cos > 0.0:
            cos_pos += 1
            basin_points.append(x)
    rho_A = cos_pos / max(1, n_basin_probe)

    # P3: Hessian condition number on a subset of basin points.
    kappas: list[float] = []
    if basin_points:
        subset = basin_points[:n_hessian_probe]
        for x in subset:
            try:
                H = _hessian_fd(circuit, x)
                eigvals = np.linalg.eigvalsh(H)
                absvals = np.abs(eigvals)
                mx = float(absvals.max()) if absvals.size else 0.0
                mn = float(absvals.min()) if absvals.size else 0.0
                if mx > 0 and mn > 1e-15:
                    kappas.append(mx / mn)
            except np.linalg.LinAlgError:
                continue
    if kappas:
        kappa_median = float(np.median(kappas))
        kappa_p90 = float(np.percentile(kappas, 90))
    else:
        kappa_median = kappa_p90 = float("nan")

    # cliff_ratio: how much larger is witness_delta than what a globally
    # Lipschitz landscape with slope L_p95 would predict across the
    # domain. Ratio >> 1 indicates plateau-then-cliff geometry.
    if L_p95 == L_p95 and L_p95 > 0:  # not NaN
        cliff_ratio = delta_witness / max(L_p95 * domain_radius, 1e-300)
    else:
        cliff_ratio = float("nan")

    return {
        "circuit": circuit.name,
        "d": circuit.d,
        "witness_delta": delta_witness,
        "L_p95": L_p95,
        "L_p99": L_p99,
        "L_max": L_max,
        "rho_A": rho_A,
        "Q_plateau_frac": plateau_frac,
        "cliff_ratio": cliff_ratio,
        "grad_norm_median": grad_norm_median,
        "kappa_median": kappa_median,
        "kappa_p90": kappa_p90,
        "n_pairs_used": len(lipschitz_ratios),
        "n_basin_probe": n_basin_probe,
        "n_kappas_used": len(kappas),
    }


# ----------------------------- main driver -----------------------------------


def main() -> int:
    circuits = [c for c in (_lr_circuit(), _poly_circuit(), _cheb_circuit(), _wdbc_circuit()) if c is not None]

    rows: list[dict] = []
    for c in circuits:
        print(f"[{c.name}] estimating landscape properties (d={c.d})...")
        row = estimate_landscape(
            c,
            n_witness_seed=2000,
            n_pairs=800,
            n_basin_probe=300,
            n_hessian_probe=15,
        )
        rows.append(row)
        print(
            f"  witness_delta={row['witness_delta']:.3e}  "
            f"L_p95={row['L_p95']:.3e}  "
            f"rho_A={row['rho_A']:.3f}  "
            f"cliff_ratio={row['cliff_ratio']:.3e}  "
            f"kappa_median={row['kappa_median']:.3e}"
        )

    out_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "results")
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "landscape_properties.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
