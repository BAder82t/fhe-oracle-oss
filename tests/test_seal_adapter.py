# Copyright (C) 2026 Bader Alissaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""SealAdapter against a fake ``seal`` module mimicking the SEAL-Python calls it uses.

No SEAL binding is importable here; exact float arithmetic stands in for CKKS.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from fhe_oracle import FHEOracle
from fhe_oracle.adapters.seal import SealAdapter

W4 = np.array([0.5, -0.3, 0.8, 0.1])
B4 = 0.1


class _Ciphertext:
    def __init__(self, slots, chain_index):
        self.slots = np.asarray(slots, dtype=np.float64)
        self.chain_index = chain_index

    def parms_id(self):
        return self.chain_index


class _ContextData:
    def __init__(self, chain_index):
        self._chain_index = chain_index

    def chain_index(self):
        return self._chain_index


class _SEALContext:
    def __init__(self, parms):
        self.slots = parms.poly_modulus_degree // 2
        self.top = len(parms.coeff_modulus) - 2

    def get_context_data(self, parms_id):
        return _ContextData(parms_id)

    def first_parms_id(self):
        return self.top


class _EncryptionParameters:
    def __init__(self, scheme):
        self.scheme = scheme

    def set_poly_modulus_degree(self, degree):
        self.poly_modulus_degree = degree

    def set_coeff_modulus(self, coeff_modulus):
        self.coeff_modulus = coeff_modulus


class _KeyGenerator:
    def __init__(self, context):
        pass

    def secret_key(self):
        return "sk"

    def create_public_key(self, destination):
        pass

    def create_relin_keys(self, destination):
        pass


class _CKKSEncoder:
    def __init__(self, context):
        self._slots = context.slots

    def slot_count(self):
        return self._slots

    def encode(self, values, scale):
        if len(values) != self._slots:
            raise ValueError("encode expects one value per slot")
        return np.asarray(values, dtype=np.float64)

    def decode(self, plain):
        return np.asarray(plain, dtype=np.float64)


class _Encryptor:
    def __init__(self, context, public_key):
        self._top = context.top

    def encrypt(self, plain):
        return _Ciphertext(plain, self._top)


class _Decryptor:
    def __init__(self, context, secret_key):
        pass

    def decrypt(self, ct):
        return ct.slots  # every slot comes back

    def invariant_noise_budget(self, ct):
        raise RuntimeError("unsupported scheme")  # SEAL rejects this for CKKS


class _Evaluator:
    def __init__(self, context):
        pass


@pytest.fixture
def fake_seal(monkeypatch):
    mod = types.ModuleType("seal")
    mod.scheme_type = types.SimpleNamespace(ckks="ckks")
    mod.EncryptionParameters = _EncryptionParameters
    mod.CoeffModulus = types.SimpleNamespace(Create=lambda degree, bits: list(bits))
    mod.SEALContext = _SEALContext
    mod.KeyGenerator = _KeyGenerator
    mod.PublicKey = object
    mod.RelinKeys = object
    mod.Encryptor = _Encryptor
    mod.Decryptor = _Decryptor
    mod.Evaluator = _Evaluator
    mod.CKKSEncoder = _CKKSEncoder
    monkeypatch.setitem(sys.modules, "seal", mod)
    return mod


def _dot_program(evaluator, relin_keys, ct):
    """z = w . x + b in slot 0; the other slots keep backend-defined values."""
    out = ct.slots.copy()
    out[0] = float(W4 @ ct.slots[: len(W4)] + B4)
    return _Ciphertext(out, ct.chain_index - 1)


def _plain_dot(x):
    return float(W4 @ np.asarray(x, dtype=np.float64) + B4)


def test_oracle_runs_scalar_output_circuit(fake_seal):
    adapter = SealAdapter(fhe_fn=_dot_program, n_features=4, output_length=1)
    assert adapter.evaluate([1.0, -0.5, 0.3, 0.7]) == [_plain_dot([1.0, -0.5, 0.3, 0.7])]
    result = FHEOracle(
        plaintext_fn=_plain_dot,
        adapter=adapter,
        input_dim=4,
        input_bounds=[(-3.0, 3.0)] * 4,
        seed=0,
    ).run(n_trials=40, threshold=1e-6)
    assert result.verdict == "PASS"
    assert result.max_error < 1e-9


def test_default_output_length_keeps_elementwise_programs(fake_seal):
    square = lambda ev, rk, ct: _Ciphertext(ct.slots**2, ct.chain_index - 1)  # noqa: E731
    adapter = SealAdapter(fhe_fn=square, n_features=3)
    assert adapter.evaluate([1.0, 2.0, 3.0]) == [1.0, 4.0, 9.0]


@pytest.mark.parametrize("output_length", [0, 4097])
def test_output_length_outside_slots_rejected(fake_seal, output_length):
    with pytest.raises(ValueError, match="output_length"):
        SealAdapter(fhe_fn=_dot_program, n_features=4, output_length=output_length)


def test_metadata_fallbacks_still_work(fake_seal):
    adapter = SealAdapter(fhe_fn=_dot_program, n_features=4, mult_depth=2, output_length=1)
    ct_out = adapter.run_fhe_program(adapter.encrypt([0.0] * 4))
    assert adapter.get_noise_budget(ct_out) == 40.0
    assert adapter.get_mult_depth_used(ct_out) == 1
