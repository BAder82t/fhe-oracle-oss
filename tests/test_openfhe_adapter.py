# Copyright (C) 2026 Bader Alissaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""OpenFHEAdapter against a fake ``openfhe`` module mimicking the calls it uses.

Exact float arithmetic stands in for CKKS; this checks plumbing, not OpenFHE numerics.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from fhe_oracle import FHEOracle
from fhe_oracle.adapters.openfhe import OpenFHEAdapter
from fhe_oracle.adapters.tenseal_adapter import HAVE_TENSEAL

# LR d=8 weights from benchmarks/tenseal_circuits._fit_lr_synthetic(d=8, seed=42).
W8 = np.array([
    -0.11717413872017622, 1.2575599788291338, 1.3731959507427856, -1.3650751039982105,
    -0.2172268057454441, 1.2281916499933732, 0.3565020287019052, 0.4254103572823009,
])
B8 = -0.08038431808593909


class _Plaintext:
    def __init__(self, values):
        self.values = [float(v) for v in values]

    def SetLength(self, n):  # noqa: N802 - OpenFHE method name
        self.values = self.values[:n]

    def GetRealPackedValue(self):  # noqa: N802
        return list(self.values)


class _Ciphertext:
    def __init__(self, slots, level=0):
        self.slots = np.asarray(slots, dtype=np.float64)
        self.level = level

    def GetLevel(self):  # noqa: N802
        return self.level


class _CryptoContext:
    def __init__(self, params):
        self.batch = params.batch
        self.enabled: set[str] = set()

    def Enable(self, feature):  # noqa: N802
        self.enabled.add(feature)

    def KeyGen(self):  # noqa: N802
        return types.SimpleNamespace(publicKey="pk", secretKey="sk")

    def EvalMultKeyGen(self, secret_key):  # noqa: N802
        pass

    def EvalSumKeyGen(self, secret_key):  # noqa: N802
        # Mirrors SchemeBase::EvalSumKeyGen, which throws unless ADVANCEDSHE is enabled.
        if "ADVANCEDSHE" not in self.enabled:
            raise RuntimeError(
                "EvalSumKeyGen operation has not been enabled. "
                "Enable(ADVANCEDSHE) must be called to enable it."
            )

    def MakeCKKSPackedPlaintext(self, values):  # noqa: N802
        if len(values) > self.batch:
            raise RuntimeError("more values than slots")
        return _Plaintext(values)

    def Encrypt(self, public_key, pt):  # noqa: N802
        slots = np.zeros(self.batch)
        slots[: len(pt.values)] = pt.values
        return _Ciphertext(slots)

    def Decrypt(self, secret_key, ct):  # noqa: N802
        # All batch slots come back; callers trim with SetLength.
        return _Plaintext(ct.slots)

    def _operand(self, value):
        if isinstance(value, _Ciphertext):
            return value.slots, value.level
        if isinstance(value, _Plaintext):
            slots = np.zeros(self.batch)
            slots[: len(value.values)] = value.values
            return slots, 0
        return float(value), 0

    def EvalMult(self, a, b):  # noqa: N802
        (va, la), (vb, lb) = self._operand(a), self._operand(b)
        return _Ciphertext(va * vb, max(la, lb) + 1)

    def EvalAdd(self, a, b):  # noqa: N802
        (va, la), (vb, lb) = self._operand(a), self._operand(b)
        return _Ciphertext(va + vb, max(la, lb))

    def EvalSum(self, ct, batch_size):  # noqa: N802
        return _Ciphertext(np.full(self.batch, ct.slots[:batch_size].sum()), ct.level)


class _Params:
    def SetMultiplicativeDepth(self, depth):  # noqa: N802
        self.depth = depth

    def SetScalingModSize(self, bits):  # noqa: N802
        self.scale_bits = bits

    def SetBatchSize(self, batch):  # noqa: N802
        self.batch = batch

    def SetSecurityLevel(self, level):  # noqa: N802
        self.security = level


@pytest.fixture
def fake_openfhe(monkeypatch):
    mod = types.ModuleType("openfhe")
    mod.CCParamsCKKSRNS = _Params
    mod.GenCryptoContext = _CryptoContext
    mod.SecurityLevel = types.SimpleNamespace(
        HEStd_128_classic=128, HEStd_192_classic=192, HEStd_256_classic=256
    )
    mod.PKESchemeFeature = types.SimpleNamespace(
        PKE="PKE", LEVELEDSHE="LEVELEDSHE", ADVANCEDSHE="ADVANCEDSHE"
    )
    monkeypatch.setitem(sys.modules, "openfhe", mod)
    return mod


def _lr_taylor3_program(cc, ct):
    """Same ops as benchmarks/library_comparison.openfhe_lr_d8; result in slot 0."""
    prod = cc.EvalMult(ct, cc.MakeCKKSPackedPlaintext(W8.tolist()))
    z = cc.EvalAdd(cc.EvalSum(prod, 8), B8)
    z3 = cc.EvalMult(cc.EvalMult(z, z), z)
    return cc.EvalAdd(cc.EvalAdd(cc.EvalMult(z, 0.25), cc.EvalMult(z3, -1.0 / 48.0)), 0.5)


def _z(x):
    return float(W8 @ np.asarray(x, dtype=np.float64) + B8)


def _taylor3(x):
    z = _z(x)
    return 0.5 + z / 4 - z**3 / 48


def _sigmoid(x):
    return float(1.0 / (1.0 + np.exp(-_z(x))))


def _lr_adapter(**kwargs):
    return OpenFHEAdapter(
        fhe_fn=_lr_taylor3_program, n_features=8, mult_depth=6, scale_mod_size=40, **kwargs
    )


def test_context_enables_advanced_she_for_eval_sum(fake_openfhe):
    _lr_adapter()


def test_oracle_runs_scalar_output_circuit(fake_openfhe):
    adapter = _lr_adapter(output_length=1)
    assert len(adapter.evaluate([0.5] * 8)) == 1
    result = FHEOracle(
        plaintext_fn=_taylor3,
        adapter=adapter,
        input_dim=8,
        input_bounds=[(-3.0, 3.0)] * 8,
        seed=0,
    ).run(n_trials=40, threshold=1e-6)
    assert result.verdict == "PASS"
    assert result.max_error < 1e-6


def test_default_output_length_keeps_elementwise_programs(fake_openfhe):
    adapter = OpenFHEAdapter(fhe_fn=lambda cc, ct: cc.EvalMult(ct, ct), n_features=3)
    assert adapter.evaluate([1.0, 2.0, 3.0]) == [1.0, 4.0, 9.0]


@pytest.mark.parametrize("output_length", [0, 9])
def test_output_length_outside_slots_rejected(fake_openfhe, output_length):
    with pytest.raises(ValueError, match="output_length"):
        _lr_adapter(output_length=output_length)


def test_noise_budget_uses_scale_mod_size(fake_openfhe):
    adapter = OpenFHEAdapter(
        fhe_fn=lambda cc, ct: ct, n_features=2, mult_depth=3, scale_mod_size=40
    )
    assert adapter.get_noise_budget(_Ciphertext([0.0, 0.0], level=1)) == 80.0


def _assert_witness_matches_tenseal(sign):
    from fhe_oracle.adapters.tenseal_adapter import (
        TenSEALContext,
        make_tenseal_taylor3_fhe_fn,
    )

    witness = (sign * 3.0 * np.sign(W8)).tolist()
    plain = _sigmoid(witness)
    openfhe_err = abs(plain - _lr_adapter(output_length=1).evaluate(witness)[0])
    tenseal_err = abs(plain - make_tenseal_taylor3_fhe_fn(W8, B8, TenSEALContext())(witness))
    assert tenseal_err > 100.0
    assert abs(openfhe_err - tenseal_err) / tenseal_err <= 1e-3


@pytest.mark.skipif(not HAVE_TENSEAL, reason="TenSEAL not installed")
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_witness_error_matches_tenseal(fake_openfhe, sign):
    _assert_witness_matches_tenseal(sign)


def _have_openfhe() -> bool:
    try:
        import openfhe  # noqa: F401
    except ImportError:
        return False
    return True


real_openfhe = pytest.mark.skipif(not _have_openfhe(), reason="openfhe not importable")


@real_openfhe
def test_real_openfhe_scalar_output_through_oracle():
    result = FHEOracle(
        plaintext_fn=_taylor3,
        adapter=_lr_adapter(output_length=1),
        input_dim=8,
        input_bounds=[(-3.0, 3.0)] * 8,
        seed=0,
    ).run(n_trials=30, threshold=1e-2)
    assert result.verdict == "PASS"
    assert result.max_error < 1e-3


@real_openfhe
@pytest.mark.skipif(not HAVE_TENSEAL, reason="TenSEAL not installed")
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_real_openfhe_witness_error_matches_tenseal(sign):
    _assert_witness_matches_tenseal(sign)
