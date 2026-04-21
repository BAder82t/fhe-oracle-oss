# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
# Patent pending: PCT/IB2026/053378
"""MNIST logistic regression mock (B3) — Taylor-3 vs true sigmoid.

Extends the WDBC mock pattern (paper §6.7) to a higher-dimensional
external benchmark. Because running TenSEAL at d=784 at B=60 is
infeasible (each FHE evaluation would take seconds and the polynomial
modulus degree required would be much larger), B3 stays in MOCK
territory: the Taylor-3 sigmoid approximation error is computed in
plaintext and treated as the FHE divergence.

Default dataset: sklearn.datasets.load_digits (d=64, 8x8 images).
Optional: full MNIST (d=784) via sklearn.datasets.fetch_openml if
network is available; falls back to load_digits on failure.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from sklearn.datasets import load_digits, fetch_openml
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))


def _taylor3_sigmoid(z: float) -> float:
    return 0.5 + z / 4.0 - z ** 3 / 48.0


def build_mnist_circuit(
    use_full_mnist: bool = False, random_state: int = 42
) -> tuple[Any, Any, np.ndarray, np.ndarray, float, int]:
    """Build MNIST plaintext + mock-FHE functions and return training data.

    Parameters
    ----------
    use_full_mnist : bool
        If True, try to fetch 28x28 MNIST (d=784) from openml. Falls
        back to `load_digits` (d=64) on any fetch failure.
    random_state : int
        Seed for LogisticRegression (determinism).

    Returns
    -------
    plaintext_fn : callable(list[float]) -> float
    fhe_fn       : callable(list[float]) -> float
    data         : np.ndarray (n, d)  standardised features
    weights      : np.ndarray (d,)
    bias         : float
    dim          : int
    """
    if not _HAS_SKLEARN:
        raise RuntimeError(
            "scikit-learn is required for MNIST benchmark. "
            "Install with: pip install scikit-learn"
        )

    X: np.ndarray
    y: np.ndarray
    if use_full_mnist:
        try:
            X_full, y_full = fetch_openml(
                "mnist_784", return_X_y=True, as_frame=False, parser="auto"
            )
            X = np.asarray(X_full, dtype=np.float64) / 255.0
            y = np.asarray([int(v) for v in y_full])
            # Subsample to keep training + sweep tractable.
            rng = np.random.default_rng(random_state)
            idx = rng.choice(len(y), size=min(5000, len(y)), replace=False)
            X = X[idx]
            y = y[idx]
            print(f"[mnist_mock] Loaded MNIST-784, n={len(y)}, d={X.shape[1]}")
        except Exception as exc:
            print(
                f"[mnist_mock] fetch_openml failed ({exc!r}); "
                "falling back to load_digits (d=64)."
            )
            X, y = load_digits(return_X_y=True)
    else:
        X, y = load_digits(return_X_y=True)

    y_binary = (y == 0).astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Guard against constant-feature zero variance (plain MNIST has lots
    # of always-black border pixels after standardisation -> nan).
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    clf = LogisticRegression(max_iter=2000, random_state=random_state)
    clf.fit(X_scaled, y_binary)

    weights = clf.coef_[0].astype(np.float64)
    bias = float(clf.intercept_[0])
    d = int(len(weights))

    def plaintext_fn(x) -> float:
        arr = np.asarray(x, dtype=np.float64)
        z = float(np.dot(weights, arr) + bias)
        return float(_sigmoid(z))

    def fhe_fn(x) -> float:
        arr = np.asarray(x, dtype=np.float64)
        z = float(np.dot(weights, arr) + bias)
        return float(_taylor3_sigmoid(z))

    return plaintext_fn, fhe_fn, X_scaled, weights, bias, d


def _training_accuracy(
    plain_fn, data: np.ndarray, y_binary: np.ndarray
) -> float:
    probs = np.array([plain_fn(row) for row in data])
    preds = (probs >= 0.5).astype(int)
    return float(np.mean(preds == y_binary))


def sanity_check(use_full_mnist: bool = False) -> None:
    """Print sanity-check values for the MNIST mock.

    Compares to WDBC (paper §6.7):
      WDBC:  d=30, ||w||2 ~ 3.84, accuracy ~ 0.988.
      MNIST 0-vs-rest on load_digits (d=64) is a ~99% task with
      relatively large ||w||2 because the class is well-separated.
    """
    plain, fhe, data, w, b, d = build_mnist_circuit(
        use_full_mnist=use_full_mnist
    )

    # Reconstruct y_binary to compute accuracy.
    if use_full_mnist:
        # Accuracy is fetched against a subsample; skip to keep things
        # simple and print just the model stats.
        y_binary = None
    else:
        _X_raw, y = load_digits(return_X_y=True)
        y_binary = (y == 0).astype(int)

    w_norm = float(np.linalg.norm(w))
    print("=" * 60)
    print("MNIST mock sanity check (B3)")
    print(f"  dataset   : {'mnist_784' if use_full_mnist else 'load_digits'}")
    print(f"  dim       : {d}")
    print("=" * 60)
    print(f"  ||w||_2   = {w_norm:.4f}")
    print(f"  bias      = {b:.4f}")
    if y_binary is not None:
        acc = _training_accuracy(plain, data, y_binary)
        print(f"  training accuracy (plaintext sigmoid) = {acc:.4f}")

    # z-range on training data.
    z_vals = np.array([float(np.dot(w, row) + b) for row in data])
    deltas = np.array([abs(plain(row) - fhe(row)) for row in data])
    print(f"  z range   : [{z_vals.min():.2f}, {z_vals.max():.2f}]")
    print(f"  |z| > 5   : {int((np.abs(z_vals) > 5).sum())} samples")
    print(f"  max δ on training      : {deltas.max():.4e}")
    print(f"  median δ on training   : {np.median(deltas):.4e}")
    # Taylor-3 diverges rapidly at large |z| because the cubic term
    # dominates. This is the same mechanism the paper's WDBC result
    # is driven by.
    idx = int(np.argmax(np.abs(z_vals)))
    print(
        f"  worst training sample: idx={idx}, "
        f"z={z_vals[idx]:.2f}, "
        f"sigma={plain(data[idx]):.4f}, "
        f"sigma_T3={fhe(data[idx]):.4f}, "
        f"delta={deltas[idx]:.4e}"
    )


if __name__ == "__main__":
    import sys

    use_full = "--full" in sys.argv
    sanity_check(use_full_mnist=use_full)
