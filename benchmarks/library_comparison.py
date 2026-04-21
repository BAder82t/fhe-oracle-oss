# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""FHE library comparison harness.

Runs the same mathematical circuit (logistic regression with a
Taylor-3 sigmoid, d=8, standard weights) through each available FHE
library's adapter and reports precision bugs found by AutoOracle.

Output format: CSV + console table. One row per (library, circuit).

Usage::

    python benchmarks/library_comparison.py          # all available libraries
    python benchmarks/library_comparison.py --libs tenseal openfhe

Adding a new library:

1. Implement ``FHEAdapter`` in ``fhe_oracle/adapters/<name>.py``
2. Register its availability flag (``HAVE_<NAME>``)
3. Add a branch to ``build_lr_d8_circuit`` below

Methodology notes (make this reproducible):
- Same LR weights (Circuit 1 from the paper draft)
- Same Taylor-3 sigmoid: sigma_T3(z) = 0.5 + z/4 - z^3/48
- Bounds: [-3, 3]^8
- AutoOracle with default config; probe budget 50, search budget 450 per run
- Seeds 41..45 (median reported; raw values preserved in CSV)
- Threshold: 1e-2 (paper default)
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Any, Callable

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(__file__))

from fhe_oracle import AutoOracle
from tenseal_circuits import _fit_lr_synthetic


# ---------------------------------------------------------------------------
# Library registry
# ---------------------------------------------------------------------------

@dataclass
class LibraryInfo:
    name: str
    version: str | None
    available: bool
    note: str


def inspect_libraries() -> list[LibraryInfo]:
    """Probe each supported library's install state without importing
    eagerly; libraries that fail to import are marked unavailable with
    their error as the note."""
    libs: list[LibraryInfo] = []

    # TenSEAL
    try:
        import tenseal as ts
        libs.append(LibraryInfo("tenseal", ts.__version__, True, ""))
    except Exception as e:
        libs.append(LibraryInfo("tenseal", None, False, str(e)[:80]))

    # OpenFHE
    try:
        import openfhe  # noqa: F401
        libs.append(LibraryInfo("openfhe",
                                getattr(openfhe, "__version__", "unknown"),
                                True, ""))
    except Exception as e:
        libs.append(LibraryInfo("openfhe", None, False, str(e)[:80]))

    # Microsoft SEAL (via seal / seal-python-py bindings)
    try:
        import seal  # noqa: F401
        libs.append(LibraryInfo("seal", getattr(seal, "__version__", "unknown"),
                                True, ""))
    except Exception as e:
        libs.append(LibraryInfo("seal", None, False, str(e)[:80]))

    # Concrete ML
    try:
        import concrete.ml as cml  # noqa: F401
        libs.append(LibraryInfo("concrete-ml",
                                getattr(cml, "__version__", "unknown"),
                                True, ""))
    except Exception as e:
        libs.append(LibraryInfo("concrete-ml", None, False, str(e)[:80]))

    # Concrete (lower-level TFHE, used for unified-squared-dot circuit)
    try:
        from concrete import fhe as _cf  # noqa: F401
        libs.append(LibraryInfo("concrete",
                                getattr(_cf, "__version__", "unknown"),
                                True, ""))
    except Exception as e:
        libs.append(LibraryInfo("concrete", None, False, str(e)[:80]))

    # Pyfhel (separate SEAL wrapper, complements TenSEAL)
    try:
        import Pyfhel  # noqa: F401
        libs.append(LibraryInfo("pyfhel",
                                getattr(Pyfhel, "__version__", "unknown"),
                                True, ""))
    except Exception as e:
        libs.append(LibraryInfo("pyfhel", None, False, str(e)[:80]))

    return libs


# ---------------------------------------------------------------------------
# Circuit 1: LR d=8 with Taylor-3 sigmoid
# ---------------------------------------------------------------------------

def plaintext_lr_d8(weights: np.ndarray, bias: float):
    """Exact double-precision reference implementation."""
    def plain(x):
        z = float(np.dot(weights, np.asarray(x, dtype=np.float64)) + bias)
        z = float(np.clip(z, -500, 500))
        return float(1.0 / (1.0 + np.exp(-z)))
    return plain


def tenseal_lr_d8(weights: np.ndarray, bias: float):
    """TenSEAL CKKS implementation -- same semantic as plaintext_lr_d8
    but evaluated via homomorphic circuit + Taylor-3 sigmoid."""
    from fhe_oracle.adapters.tenseal_adapter import (
        HAVE_TENSEAL, TenSEALContext, make_tenseal_taylor3_fhe_fn,
    )
    if not HAVE_TENSEAL:
        return None
    ctx = TenSEALContext()
    return make_tenseal_taylor3_fhe_fn(weights, bias, ctx)


def openfhe_lr_d8(weights: np.ndarray, bias: float):
    """OpenFHE CKKS implementation of LR d=8 Taylor-3 sigmoid.

    Matches the TenSEAL CKKS parameters as closely as possible:
    scaling mod size 40, multiplicative depth 3 (enough for z^3),
    ring dimension 16384 (paper §6.5-6.7).
    """
    try:
        import openfhe as ofh
    except ImportError:
        return None

    w_list = np.asarray(weights, dtype=np.float64).tolist()
    b_val = float(bias)

    # --- One-time crypto context setup (shared across eval calls) ---
    params = ofh.CCParamsCKKSRNS()
    params.SetMultiplicativeDepth(6)              # EvalSum + z^3 + ct*pt scalar mults ~ depth 5-6
    params.SetScalingModSize(40)
    params.SetBatchSize(8)                        # slot-width >= len(weights)
    cc = ofh.GenCryptoContext(params)
    cc.Enable(ofh.PKESchemeFeature.PKE)
    cc.Enable(ofh.PKESchemeFeature.LEVELEDSHE)
    cc.Enable(ofh.PKESchemeFeature.ADVANCEDSHE)
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    # EvalSum requires rotation keys for 1,2,4 (log2(batch_size) rotations)
    cc.EvalSumKeyGen(keys.secretKey)
    w_plain = cc.MakeCKKSPackedPlaintext(w_list)  # fixed weight vector

    def fhe_fn(x):
        xa = np.asarray(x, dtype=np.float64).ravel().tolist()
        # Pad to batch size 8 if needed (weights are len-8 already).
        x_ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(xa))
        # Element-wise mult with plaintext weights then sum via EvalSum
        prod_ct = cc.EvalMult(x_ct, w_plain)
        dot_ct = cc.EvalSum(prod_ct, 8)           # reduces to slot 0
        # Add bias as a plaintext scalar (encoded as length-1 vector broadcast)
        z_ct = cc.EvalAdd(dot_ct, b_val)
        # Taylor-3: 0.5 + z/4 - z^3/48
        z2_ct = cc.EvalMult(z_ct, z_ct)
        z3_ct = cc.EvalMult(z2_ct, z_ct)
        t_ct = cc.EvalAdd(
            cc.EvalAdd(
                cc.EvalMult(z_ct, 0.25),
                cc.EvalMult(z3_ct, -1.0 / 48.0),
            ),
            0.5,
        )
        pt = cc.Decrypt(keys.secretKey, t_ct)
        pt.SetLength(1)
        return float(pt.GetRealPackedValue()[0])

    return fhe_fn


def seal_lr_d8(weights: np.ndarray, bias: float):
    """Placeholder: Microsoft SEAL CKKS implementation.

    TODO: same as openfhe_lr_d8 -- wire once SEAL python bindings
    can be installed on the host.
    """
    return None


def concrete_ml_lr_d8(weights: np.ndarray, bias: float):
    """Placeholder: Concrete ML implementation.

    TODO: Concrete ML is TFHE-based, not CKKS; the closest analogue is
    a quantised LR compiled via Concrete ML. This is a different
    precision regime (integer + bootstrapping-driven error) and will
    produce a qualitatively different number than CKKS -- worth a
    separate row explicitly labelled TFHE.
    """
    return None


# ===========================================================================
# Integer-arithmetic circuit family (BGV, BFV, TFHE)
# ===========================================================================
#
# LR-inspired integer inner product: y = sum(w_i * x_i) mod p, with
# depth-2 squaring: z = y^2 mod p. Same weight layout as the CKKS LR
# circuit (scaled to int16 range), so cross-family comparisons are
# sane even though the schemes are different.

INT_D = 8
INT_SCALE = 100        # scale floats -> int16-safe integers
INT_MODULUS = 65537    # prime plaintext modulus (fits depth-2 squared dot)


def _int_weights_bias(weights: np.ndarray, bias: float):
    """Quantise CKKS LR weights to integer-scheme range."""
    w_int = np.round(weights * INT_SCALE).astype(np.int64).tolist()
    b_int = int(round(bias * INT_SCALE))
    return w_int, b_int


def plaintext_int_lr_d8(weights: np.ndarray, bias: float):
    """Plaintext int-LR: z = (w . x + b)^2 mod p, exact integer arithmetic."""
    w_int, b_int = _int_weights_bias(weights, bias)

    def plain(x):
        xi = np.round(np.asarray(x) * INT_SCALE).astype(np.int64)
        y = int(sum(w * int(xv) for w, xv in zip(w_int, xi))) + b_int
        z = (y * y) % INT_MODULUS
        return float(z)

    return plain


def openfhe_bgv_int_lr_d8(weights: np.ndarray, bias: float):
    """OpenFHE BGV implementation of int-LR squared inner product."""
    try:
        import openfhe as ofh
    except ImportError:
        return None

    w_int, b_int = _int_weights_bias(weights, bias)

    params = ofh.CCParamsBGVRNS()
    params.SetMultiplicativeDepth(2)
    params.SetPlaintextModulus(INT_MODULUS)
    params.SetBatchSize(INT_D)
    cc = ofh.GenCryptoContext(params)
    cc.Enable(ofh.PKESchemeFeature.PKE)
    cc.Enable(ofh.PKESchemeFeature.LEVELEDSHE)
    cc.Enable(ofh.PKESchemeFeature.ADVANCEDSHE)
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalSumKeyGen(keys.secretKey)

    w_plain = cc.MakePackedPlaintext(w_int)

    def fhe_fn(x):
        xi = np.round(np.asarray(x) * INT_SCALE).astype(np.int64).tolist()
        x_pt = cc.MakePackedPlaintext(xi)
        x_ct = cc.Encrypt(keys.publicKey, x_pt)
        prod_ct = cc.EvalMult(x_ct, w_plain)
        dot_ct = cc.EvalSum(prod_ct, INT_D)       # sum to slot 0
        b_plain = cc.MakePackedPlaintext([b_int] * INT_D)
        y_ct = cc.EvalAdd(dot_ct, b_plain)
        z_ct = cc.EvalMult(y_ct, y_ct)            # depth 2
        pt = cc.Decrypt(keys.secretKey, z_ct)
        pt.SetLength(1)
        # BGV decrypt returns ints in [0, p); match plaintext semantics
        return float(pt.GetPackedValue()[0] % INT_MODULUS)

    return fhe_fn


def openfhe_bfv_int_lr_d8(weights: np.ndarray, bias: float):
    """OpenFHE BFV implementation of int-LR squared inner product."""
    try:
        import openfhe as ofh
    except ImportError:
        return None

    w_int, b_int = _int_weights_bias(weights, bias)

    params = ofh.CCParamsBFVRNS()
    params.SetMultiplicativeDepth(2)
    params.SetPlaintextModulus(INT_MODULUS)
    params.SetBatchSize(INT_D)
    cc = ofh.GenCryptoContext(params)
    cc.Enable(ofh.PKESchemeFeature.PKE)
    cc.Enable(ofh.PKESchemeFeature.LEVELEDSHE)
    cc.Enable(ofh.PKESchemeFeature.ADVANCEDSHE)
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalSumKeyGen(keys.secretKey)

    w_plain = cc.MakePackedPlaintext(w_int)

    def fhe_fn(x):
        xi = np.round(np.asarray(x) * INT_SCALE).astype(np.int64).tolist()
        x_pt = cc.MakePackedPlaintext(xi)
        x_ct = cc.Encrypt(keys.publicKey, x_pt)
        prod_ct = cc.EvalMult(x_ct, w_plain)
        dot_ct = cc.EvalSum(prod_ct, INT_D)
        b_plain = cc.MakePackedPlaintext([b_int] * INT_D)
        y_ct = cc.EvalAdd(dot_ct, b_plain)
        z_ct = cc.EvalMult(y_ct, y_ct)
        pt = cc.Decrypt(keys.secretKey, z_ct)
        pt.SetLength(1)
        return float(pt.GetPackedValue()[0] % INT_MODULUS)

    return fhe_fn


INT_LIBRARY_BUILDERS: dict[str, Callable[[np.ndarray, float], Callable | None]] = {
    "openfhe-bgv": openfhe_bgv_int_lr_d8,
    "openfhe-bfv": openfhe_bfv_int_lr_d8,
}


# ===========================================================================
# UNIFIED CIRCUIT: (w . x + b)^2 — runs in every library
# ===========================================================================
#
# Real-valued plaintext reference for fairness: y_true = (w . x + b)^2
# computed in float64. Each library's FHE path computes the same math in
# its native scheme. Divergence = |y_true - y_fhe|. Integer schemes
# quantise and un-scale their output so the comparison is in the same
# units for all libraries.

UNIFIED_INT_SCALE = 5.0            # quantisation factor for int-scheme libs
UNIFIED_INT_PLAINTEXT_MOD = 786433  # NTT-friendly prime (24*32768+1) -- compatible with ring=16384


def unified_plaintext(weights: np.ndarray, bias: float):
    """Reference: y = (w . x + b)^2 in float64."""
    w = np.asarray(weights, dtype=np.float64)
    b_val = float(bias)

    def plain(x):
        z = float(np.dot(w, np.asarray(x, dtype=np.float64).ravel())) + b_val
        return z * z
    return plain


def tenseal_squared_dot(weights: np.ndarray, bias: float):
    """TenSEAL CKKS (w . x + b)^2."""
    from fhe_oracle.adapters.tenseal_adapter import HAVE_TENSEAL, TenSEALContext
    if not HAVE_TENSEAL:
        return None
    ctx = TenSEALContext()
    w_list = np.asarray(weights, dtype=np.float64).tolist()
    b_val = float(bias)

    def fhe_fn(x):
        xa = np.asarray(x, dtype=np.float64)
        ct_x = ctx.encrypt(xa)
        ct_z = ct_x.dot(w_list) + b_val
        ct_z2 = ct_z * ct_z
        return float(ctx.decrypt(ct_z2)[0])
    return fhe_fn


def openfhe_squared_dot(weights: np.ndarray, bias: float):
    """OpenFHE CKKS (w . x + b)^2."""
    try:
        import openfhe as ofh
    except ImportError:
        return None
    w_list = np.asarray(weights, dtype=np.float64).tolist()
    b_val = float(bias)
    params = ofh.CCParamsCKKSRNS()
    params.SetMultiplicativeDepth(4)
    params.SetScalingModSize(40)
    params.SetBatchSize(8)
    cc = ofh.GenCryptoContext(params)
    cc.Enable(ofh.PKESchemeFeature.PKE)
    cc.Enable(ofh.PKESchemeFeature.LEVELEDSHE)
    cc.Enable(ofh.PKESchemeFeature.ADVANCEDSHE)
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalSumKeyGen(keys.secretKey)
    w_plain = cc.MakeCKKSPackedPlaintext(w_list)

    def fhe_fn(x):
        xa = np.asarray(x, dtype=np.float64).ravel().tolist()
        x_ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(xa))
        prod_ct = cc.EvalMult(x_ct, w_plain)
        dot_ct = cc.EvalSum(prod_ct, 8)
        z_ct = cc.EvalAdd(dot_ct, b_val)
        z2_ct = cc.EvalMult(z_ct, z_ct)
        pt = cc.Decrypt(keys.secretKey, z2_ct)
        pt.SetLength(1)
        return float(pt.GetRealPackedValue()[0])
    return fhe_fn


def pyfhel_squared_dot(weights: np.ndarray, bias: float):
    """Pyfhel CKKS (w . x + b)^2 — depth 2 fits Pyfhel's manual mgmt."""
    try:
        from Pyfhel import Pyfhel
    except ImportError:
        return None
    w = np.asarray(weights, dtype=np.float64)
    b_val = float(bias)
    HE = Pyfhel()
    HE.contextGen(scheme='ckks', n=16384, scale=2**40,
                  qi_sizes=[60, 40, 40, 40, 40, 60])
    HE.keyGen(); HE.rotateKeyGen(); HE.relinKeyGen()

    def fhe_fn(x):
        xa = np.asarray(x, dtype=np.float64).ravel()
        ct_x = HE.encryptFrac(xa)
        pt_w = HE.encodeFrac(w)
        ct_prod = HE.multiply_plain(ct_x, pt_w)
        HE.relinearize(ct_prod); HE.rescale_to_next(ct_prod)
        ct_sum = HE.cumul_add(ct_prod, in_new_ctxt=True)
        ct_z = ct_sum + b_val
        ct_z2 = HE.multiply(ct_z, ct_z)
        HE.relinearize(ct_z2); HE.rescale_to_next(ct_z2)
        out = HE.decryptFrac(ct_z2)
        return float(out[0])
    return fhe_fn


def openfhe_bgv_squared_dot(weights: np.ndarray, bias: float):
    """OpenFHE BGV (w . x + b)^2 — quantise inputs, return un-scaled float.

    Scale bookkeeping: w and x are quantised at scale ``s`` each,
    so w_int·x_int has scale s^2. To add b at the same scale,
    b_int must be quantised at s^2 (not s).
    """
    try:
        import openfhe as ofh
    except ImportError:
        return None
    s = UNIFIED_INT_SCALE
    w_int = [int(round(float(v) * s)) for v in weights]
    b_int = int(round(float(bias) * s * s))       # scale s^2 to match w·x
    params = ofh.CCParamsBGVRNS()
    params.SetMultiplicativeDepth(2)
    params.SetPlaintextModulus(UNIFIED_INT_PLAINTEXT_MOD)
    params.SetBatchSize(8)
    cc = ofh.GenCryptoContext(params)
    cc.Enable(ofh.PKESchemeFeature.PKE)
    cc.Enable(ofh.PKESchemeFeature.LEVELEDSHE)
    cc.Enable(ofh.PKESchemeFeature.ADVANCEDSHE)
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalSumKeyGen(keys.secretKey)
    w_pt = cc.MakePackedPlaintext(w_int)

    def fhe_fn(x):
        xi = [int(round(float(v) * s)) for v in np.asarray(x).ravel()]
        x_pt = cc.MakePackedPlaintext(xi)
        x_ct = cc.Encrypt(keys.publicKey, x_pt)
        prod_ct = cc.EvalMult(x_ct, w_pt)
        dot_ct = cc.EvalSum(prod_ct, 8)
        b_pt = cc.MakePackedPlaintext([b_int] * 8)
        z_ct = cc.EvalAdd(dot_ct, b_pt)
        z2_ct = cc.EvalMult(z_ct, z_ct)
        pt = cc.Decrypt(keys.secretKey, z2_ct)
        pt.SetLength(1)
        # Un-scale: we squared a s-scaled value, so output is in units of s^4
        val = int(pt.GetPackedValue()[0])
        # Cast back to signed; BGV returns modular residue
        p = UNIFIED_INT_PLAINTEXT_MOD
        if val >= p // 2:
            val -= p
        return float(val) / (s ** 4)
    return fhe_fn


def openfhe_bfv_squared_dot(weights: np.ndarray, bias: float):
    """OpenFHE BFV (w . x + b)^2 — same protocol as BGV variant."""
    try:
        import openfhe as ofh
    except ImportError:
        return None
    s = UNIFIED_INT_SCALE
    w_int = [int(round(float(v) * s)) for v in weights]
    b_int = int(round(float(bias) * s * s))       # scale s^2 to match w·x
    params = ofh.CCParamsBFVRNS()
    params.SetMultiplicativeDepth(2)
    params.SetPlaintextModulus(UNIFIED_INT_PLAINTEXT_MOD)
    params.SetBatchSize(8)
    cc = ofh.GenCryptoContext(params)
    cc.Enable(ofh.PKESchemeFeature.PKE)
    cc.Enable(ofh.PKESchemeFeature.LEVELEDSHE)
    cc.Enable(ofh.PKESchemeFeature.ADVANCEDSHE)
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalSumKeyGen(keys.secretKey)
    w_pt = cc.MakePackedPlaintext(w_int)

    def fhe_fn(x):
        xi = [int(round(float(v) * s)) for v in np.asarray(x).ravel()]
        x_pt = cc.MakePackedPlaintext(xi)
        x_ct = cc.Encrypt(keys.publicKey, x_pt)
        prod_ct = cc.EvalMult(x_ct, w_pt)
        dot_ct = cc.EvalSum(prod_ct, 8)
        b_pt = cc.MakePackedPlaintext([b_int] * 8)
        z_ct = cc.EvalAdd(dot_ct, b_pt)
        z2_ct = cc.EvalMult(z_ct, z_ct)
        pt = cc.Decrypt(keys.secretKey, z2_ct)
        pt.SetLength(1)
        val = int(pt.GetPackedValue()[0])
        p = UNIFIED_INT_PLAINTEXT_MOD
        if val >= p // 2:
            val -= p
        return float(val) / (s ** 4)
    return fhe_fn


def concrete_squared_dot(weights: np.ndarray, bias: float):
    """Concrete (TFHE) (w . x + b)^2 via concrete.fhe lower-level API."""
    try:
        from concrete import fhe
    except ImportError:
        return None
    s = UNIFIED_INT_SCALE
    w_int_np = np.round(np.asarray(weights, dtype=np.float64) * s).astype(np.int64)
    b_int = int(round(float(bias) * s * s))     # scale s^2 to match w·x

    @fhe.compiler({"x": "encrypted"})
    def circuit(x):
        z = np.sum(w_int_np * x) + b_int
        return z * z

    # Compile on a representative input set
    rng = np.random.default_rng(0)
    input_set = [np.round(rng.uniform(-3.0, 3.0, 8) * s).astype(np.int64)
                 for _ in range(50)]
    try:
        compiled = circuit.compile(input_set)
    except Exception:
        return None

    def fhe_fn(x):
        xi = np.round(np.asarray(x, dtype=np.float64).ravel() * s).astype(np.int64)
        try:
            val = int(compiled.encrypt_run_decrypt(xi))
        except Exception:
            return 0.0
        return float(val) / (s ** 4)
    return fhe_fn


UNIFIED_LIBRARY_BUILDERS: dict[str, Callable[[np.ndarray, float], Callable | None]] = {
    "tenseal":     tenseal_squared_dot,
    "openfhe":     openfhe_squared_dot,
    "pyfhel":      pyfhel_squared_dot,
    "openfhe-bgv": openfhe_bgv_squared_dot,
    "openfhe-bfv": openfhe_bfv_squared_dot,
    "concrete":    concrete_squared_dot,
}


def pyfhel_lr_d8(weights: np.ndarray, bias: float):
    """Pyfhel CKKS implementation of LR d=8 Taylor-1 sigmoid (partial).

    Pyfhel wraps Microsoft SEAL and exposes a lower-level API than
    TenSEAL: every ciphertext-ciphertext multiplication requires the
    caller to manually manage scale and modulus-switching via
    ``rescale_to_next`` + ``mod_switch_to_next``. A full Taylor-3
    matching the TenSEAL row would need per-operation level-alignment
    that is a separate engineering exercise. This adapter therefore
    implements a Taylor-1 truncation (0.5 + z/4) so the Pyfhel row is
    honest about what works out-of-the-box.

    Result: the Pyfhel row is NOT directly comparable to TenSEAL /
    OpenFHE on the Taylor-3 circuit; they compute different
    polynomials. Flagged as partial in the README.
    """
    try:
        from Pyfhel import Pyfhel
    except ImportError:
        return None

    w = np.asarray(weights, dtype=np.float64)
    b_val = float(bias)

    HE = Pyfhel()
    HE.contextGen(scheme='ckks', n=16384, scale=2**40,
                  qi_sizes=[60, 40, 40, 40, 40, 60])
    HE.keyGen()
    HE.rotateKeyGen()
    HE.relinKeyGen()

    def fhe_fn(x):
        xa = np.asarray(x, dtype=np.float64).ravel()
        ct_x = HE.encryptFrac(xa)
        pt_w = HE.encodeFrac(w)
        ct_prod = HE.multiply_plain(ct_x, pt_w)
        HE.relinearize(ct_prod)
        HE.rescale_to_next(ct_prod)           # z at level 1 after rescale
        ct_sum = HE.cumul_add(ct_prod, in_new_ctxt=True)
        ct_z = ct_sum + b_val
        # Taylor-3: 0.5 + z/4 - z^3/48, all aligned to z^3's level (3)
        # Compute z^2 then z^3 with rescales; then scale the linear term
        # to match via mod-switches.
        ct_z2 = HE.multiply(ct_z, ct_z)
        HE.relinearize(ct_z2)
        HE.rescale_to_next(ct_z2)              # z^2 at level 2
        # For z^3 = z^2 * z, align z down to level 2 first
        ct_z_at_lvl2 = HE.mod_switch_to_next(ct_z)   # z at level 2
        ct_z3 = HE.multiply(ct_z2, ct_z_at_lvl2)
        HE.relinearize(ct_z3)
        HE.rescale_to_next(ct_z3)              # z^3 at level 3
        # Scalar mult by -1/48 (Pyfhel operator overload)
        ct_cubic_term = ct_z3 * (-1.0 / 48.0)
        # Align z/4 to z^3's level for final add
        ct_lin = ct_z * 0.25
        ct_lin = HE.mod_switch_to_next(ct_lin)
        ct_lin = HE.mod_switch_to_next(ct_lin)
        ct_lin = HE.mod_switch_to_next(ct_lin)
        ct_result = ct_lin + ct_cubic_term + 0.5
        out = HE.decryptFrac(ct_result)
        return float(out[0])

    return fhe_fn


LIBRARY_BUILDERS: dict[str, Callable[[np.ndarray, float], Callable | None]] = {
    "tenseal": tenseal_lr_d8,
    "openfhe": openfhe_lr_d8,
    "seal": seal_lr_d8,
    "pyfhel": pyfhel_lr_d8,
    "concrete-ml": concrete_ml_lr_d8,
}


CIRCUIT_FAMILIES = {
    "ckks-taylor3": {
        "label": "LR d=8 Taylor-3 sigmoid (CKKS real-valued)",
        "plain": plaintext_lr_d8,
        "builders": LIBRARY_BUILDERS,
        "bounds": [(-3.0, 3.0)] * 8,
        "threshold": 1e-2,
    },
    "int-lr-squared": {
        "label": "Int-LR (w.x+b)^2 mod p=65537 (BGV/BFV exact integer, d=8)",
        "plain": plaintext_int_lr_d8,
        "builders": {
            "openfhe-bgv": openfhe_bgv_int_lr_d8,
            "openfhe-bfv": openfhe_bfv_int_lr_d8,
        },
        "bounds": [(-3.0, 3.0)] * 8,
        "threshold": 1e-2,
    },
    "unified-squared-dot": {
        "label": "Unified (w.x+b)^2 on [-3,3]^8 — runs in every library, real-valued plaintext reference",
        "plain": unified_plaintext,
        "builders": UNIFIED_LIBRARY_BUILDERS,
        "bounds": [(-3.0, 3.0)] * 8,
        "threshold": 1e-6,
    },
}


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    library: str
    library_version: str
    circuit: str
    seed: int
    verdict: str
    max_error: float
    regime: str
    strategy_used: str
    evals: int
    wall_seconds: float


def run_one(library_name: str, library_version: str,
            plain_fn: Callable, fhe_fn: Callable,
            circuit_name: str, seed: int,
            n_trials: int, threshold: float) -> BenchmarkResult:
    """Single-seed AutoOracle run. Returns metrics row."""
    counter = {"n": 0}

    def fhe_counted(x):
        counter["n"] += 1
        return fhe_fn(x)

    auto = AutoOracle(
        plaintext_fn=plain_fn,
        fhe_fn=fhe_counted,
        bounds=[(-3.0, 3.0)] * 8,
        n_probes=50,
    )
    t0 = time.perf_counter()
    try:
        result = auto.run(n_trials=n_trials, seed=seed, threshold=threshold)
    except Exception as exc:
        return BenchmarkResult(
            library=library_name,
            library_version=library_version,
            circuit=circuit_name,
            seed=seed,
            verdict=f"ERROR: {str(exc)[:60]}",
            max_error=0.0,
            regime="-",
            strategy_used="-",
            evals=counter["n"],
            wall_seconds=time.perf_counter() - t0,
        )
    elapsed = time.perf_counter() - t0

    return BenchmarkResult(
        library=library_name,
        library_version=library_version,
        circuit=circuit_name,
        seed=seed,
        verdict=result.verdict,
        max_error=float(result.max_error),
        regime=getattr(result, "regime", "-"),
        strategy_used=getattr(result, "strategy_used", "-"),
        evals=counter["n"],
        wall_seconds=elapsed,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--libs", nargs="*",
                        help="Libraries to benchmark (default: all available)")
    parser.add_argument("--circuit", default="ckks-taylor3",
                        choices=sorted(CIRCUIT_FAMILIES.keys()),
                        help="Circuit family: CKKS real-valued or integer-BGV/BFV/TFHE")
    parser.add_argument("--seeds", nargs="*", type=int,
                        default=[41, 42, 43, 44, 45])
    parser.add_argument("--n-trials", type=int, default=500)
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override circuit-family default threshold")
    parser.add_argument("--output", default=None,
                        help="Output CSV; defaults to benchmarks/results/library_comparison_<circuit>.csv")
    args = parser.parse_args()

    circuit = CIRCUIT_FAMILIES[args.circuit]
    threshold = args.threshold if args.threshold is not None else circuit["threshold"]
    output = args.output if args.output else f"benchmarks/results/library_comparison_{args.circuit}.csv"

    all_libs = {li.name: li for li in inspect_libraries()}
    builders = circuit["builders"]

    if args.libs:
        selected = args.libs
    else:
        # Default: only libraries that are both (a) installed and (b)
        # have a builder for the chosen circuit family.
        selected = [name for name in builders.keys()
                    if name in all_libs and all_libs[name].available]
        if not selected and "openfhe" in [li.name for li in all_libs.values()
                                          if li.available] and any(
            b.startswith("openfhe-") for b in builders):
            # Integer schemes use the same openfhe import for BGV/BFV.
            selected = [b for b in builders if b.startswith("openfhe-")]

    print(f"Circuit family: {args.circuit} -- {circuit['label']}")
    print(f"Library availability:")
    for li in all_libs.values():
        marker = "✓" if li.available else "✗"
        print(f"  {marker} {li.name:15s} {str(li.version or '-'):15s} "
              f"{li.note}")
    print(f"Running benchmark on: {selected}")
    print()

    weights, bias = _fit_lr_synthetic(d=8, seed=42)
    plain_fn = circuit["plain"](weights, bias)

    results: list[BenchmarkResult] = []

    for lib_name in selected:
        # Integer-scheme libs (openfhe-bgv, openfhe-bfv) share the
        # openfhe install; their "availability" is gated by openfhe.
        probe_name = "openfhe" if lib_name.startswith("openfhe-") else lib_name
        li = all_libs.get(probe_name) or LibraryInfo(lib_name, None, False, "unknown library")

        if not li.available:
            print(f"[skip] {lib_name} not installed: {li.note}")
            continue

        build = builders.get(lib_name)
        if build is None:
            print(f"[skip] no {args.circuit} builder for {lib_name}")
            continue

        fhe_fn = build(weights, bias)
        if fhe_fn is None:
            print(f"[skip] {lib_name} builder not yet implemented (PR welcome)")
            continue

        print(f"[run ] {lib_name} {li.version} — {args.circuit} — {len(args.seeds)} seeds")
        for seed in args.seeds:
            result = run_one(
                library_name=lib_name,
                library_version=str(li.version),
                plain_fn=plain_fn,
                fhe_fn=fhe_fn,
                circuit_name=args.circuit,
                seed=seed,
                n_trials=args.n_trials,
                threshold=threshold,
            )
            results.append(result)
            print(f"       seed={seed} verdict={result.verdict} "
                  f"max_err={result.max_error:.3e} "
                  f"regime={result.regime} "
                  f"t={result.wall_seconds:.1f}s")

    if not results:
        print()
        print("No benchmarks ran. Install at least one FHE library:")
        print("  pip install tenseal")
        print("  pip install openfhe          # Linux only as of v1.5")
        return 1

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
    print()
    print(f"Wrote {len(results)} rows → {output}")

    # Summary table: median max_error per library
    by_lib: dict[str, list[BenchmarkResult]] = {}
    for r in results:
        by_lib.setdefault(r.library, []).append(r)

    print()
    print(f"{'library':15s} {'version':12s} {'fail_rate':>10s} "
          f"{'median_max_err':>18s} {'wall_median':>12s}")
    print("-" * 75)
    for lib, rs in by_lib.items():
        fail_rate = sum(1 for r in rs if r.verdict == "FAIL") / len(rs)
        med_max = float(np.median([r.max_error for r in rs]))
        med_t = float(np.median([r.wall_seconds for r in rs]))
        ver = rs[0].library_version
        print(f"{lib:15s} {ver:12s} {fail_rate*100:>9.0f}% "
              f"{med_max:>18.3e} {med_t:>11.1f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
