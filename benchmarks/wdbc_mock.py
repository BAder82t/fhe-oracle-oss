# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
# Patent pending: PCT/IB2026/053378
"""WDBC logistic regression mock — Taylor-3 vs true sigmoid (A3).

Reconstructs §6.7 of the paper. Isolates the polynomial-approximation
component of the WDBC LR circuit:

- Dataset: `sklearn.datasets.load_breast_cancer` (569, 30).
- Preprocessing: StandardScaler.
- Model: sklearn LogisticRegression (default params,
  `random_state=42`, `max_iter=1000`).
- Plaintext: true sigmoid σ(z) = 1/(1+exp(-z)).
- FHE mock: Taylor-3 σ_T3(z) = 0.5 + z/4 − z³/48.
- Divergence: δ(x) = |σ(W·x + b) − σ_T3(W·x + b)|.

This is faithful to the mechanism the paper's real-CKKS result is
driven by (Taylor-3 cubic blow-up at large |z|) but does NOT model
CKKS residual noise. See §6.6 of the paper: "the LR real-CKKS
experiment primarily stresses the polynomial-approximation
component".
"""

from __future__ import annotations

import numpy as np

try:
    from sklearn.datasets import load_breast_cancer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))


def _taylor3_sigmoid(z: float) -> float:
    return 0.5 + z / 4.0 - z ** 3 / 48.0


def build_wdbc_circuit(random_state: int = 42):
    """Build WDBC plaintext + mock-FHE functions and return training data.

    Returns
    -------
    plaintext_fn : callable(list[float]) -> float
    fhe_fn       : callable(list[float]) -> float
    data         : np.ndarray, shape (569, 30)  (standardised features)
    weights      : np.ndarray, shape (30,)
    bias         : float
    dim          : int (30)
    """
    if not _HAS_SKLEARN:
        raise RuntimeError(
            "scikit-learn is required for WDBC benchmark. "
            "Install with: pip install scikit-learn"
        )
    X, y = load_breast_cancer(return_X_y=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=1000, random_state=random_state)
    clf.fit(X_scaled, y)

    weights = clf.coef_[0].astype(np.float64)
    bias = float(clf.intercept_[0])

    def plaintext_fn(x) -> float:
        xa = np.asarray(x, dtype=np.float64)
        z = float(np.dot(weights, xa) + bias)
        return float(_sigmoid(z))

    def fhe_fn(x) -> float:
        xa = np.asarray(x, dtype=np.float64)
        z = float(np.dot(weights, xa) + bias)
        return float(_taylor3_sigmoid(z))

    return plaintext_fn, fhe_fn, X_scaled, weights, bias, 30


def sanity_check() -> None:
    """Print sanity-check values matching paper §6.7."""
    plain, fhe, data, w, b, d = build_wdbc_circuit(random_state=42)

    w_norm = float(np.linalg.norm(w))
    acc = _training_accuracy(plain, data, w, b)
    print("=" * 60)
    print("WDBC mock sanity check")
    print("=" * 60)
    print(f"  ‖w‖₂ = {w_norm:.4f}    (paper: ≈ 3.84)")
    print(f"  b    = {b:.4f}         (paper: ≈ 0.22)")
    print(f"  training accuracy = {acc:.4f}  (paper: ≈ 0.988)")

    # Sample extreme preactivations and measure divergence.
    z_vals = []
    deltas = []
    for row in data:
        z = float(np.dot(w, row) + b)
        z_vals.append(z)
        deltas.append(abs(plain(row) - fhe(row)))
    z_vals = np.array(z_vals)
    deltas = np.array(deltas)

    print(f"  z range across training: [{z_vals.min():.2f}, {z_vals.max():.2f}]")
    print(f"  |z| > 5 samples: {int((np.abs(z_vals) > 5).sum())}")
    print(f"  max δ on training data: {deltas.max():.4e}")
    print(f"  median δ on training data: {np.median(deltas):.4e}")

    # Show a single large-|z| sample's divergence.
    idx = int(np.argmax(np.abs(z_vals)))
    print(
        f"  worst training sample: idx={idx}, "
        f"z={z_vals[idx]:.2f}, "
        f"σ={plain(data[idx]):.4f}, "
        f"σ_T3={fhe(data[idx]):.4f}, "
        f"δ={deltas[idx]:.4e}"
    )


def _training_accuracy(plain_fn, data, w, b) -> float:
    """Compute training accuracy using the plaintext sigmoid."""
    probs = np.array([plain_fn(row) for row in data])
    # Reload labels to compute accuracy.
    X, y = load_breast_cancer(return_X_y=True)
    preds = (probs >= 0.5).astype(int)
    return float(np.mean(preds == y))


if __name__ == "__main__":
    sanity_check()
