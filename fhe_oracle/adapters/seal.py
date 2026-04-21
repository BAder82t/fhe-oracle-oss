# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
# Patent pending: PCT/IB2026/053378
"""Adapter for Microsoft SEAL (Python bindings: seal-python).

Installation
------------
SEAL's Python bindings are not on PyPI. Build them from source:

    git clone https://github.com/Huelse/SEAL-Python.git
    cd SEAL-Python
    git submodule update --init --recursive
    pip install -r requirements.txt
    python setup.py build_ext -i
    python setup.py install

After build, the adapter below should work. The ``seal`` module name is
assumed — if your build produces a different name (e.g. ``pyseal``),
edit the import lines in this file.

Example
-------
    from fhe_oracle.adapters.seal import SealAdapter

    def square_fn(evaluator, relin_keys, ct):
        result = evaluator.square(ct)
        evaluator.relinearize_inplace(result, relin_keys)
        evaluator.rescale_to_next_inplace(result)
        return result

    adapter = SealAdapter(fhe_fn=square_fn, n_features=4, mult_depth=2)
"""

from __future__ import annotations

from typing import Any, Callable

from .base import FHEAdapter


class SealAdapter(FHEAdapter):
    """FHEAdapter backed by Microsoft SEAL CKKS."""

    def __init__(
        self,
        fhe_fn: Callable[[Any, Any, Any], Any],
        n_features: int,
        mult_depth: int = 2,
        scale_bits: int = 40,
        poly_modulus_degree: int = 8192,
    ) -> None:
        try:
            import seal  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "seal-python is not installed. See this module's docstring "
                "for build instructions."
            )

        self._fhe_fn = fhe_fn
        self._n_features = n_features
        self._mult_depth = mult_depth
        self._scale = 2.0 ** scale_bits
        self._scale_bits = scale_bits

        (
            self._context,
            self._encryptor,
            self._decryptor,
            self._evaluator,
            self._encoder,
            self._relin_keys,
        ) = self._setup_context(poly_modulus_degree, mult_depth, scale_bits)

    def encrypt(self, x: list[float]) -> Any:
        padded = list(x) + [0.0] * max(0, self._slot_count - len(x))
        plain = self._encoder.encode(padded, self._scale)
        return self._encryptor.encrypt(plain)

    def decrypt(self, ciphertext: Any) -> list[float]:
        plain = self._decryptor.decrypt(ciphertext)
        values = self._encoder.decode(plain)
        return [v.real for v in values[: self._n_features]]

    def run_fhe_program(self, ciphertext: Any) -> Any:
        return self._fhe_fn(self._evaluator, self._relin_keys, ciphertext)

    def get_noise_budget(self, ciphertext: Any) -> float:
        try:
            budget = self._decryptor.invariant_noise_budget(ciphertext)
            return float(budget)
        except Exception:
            pass
        try:
            ctx_data = self._context.get_context_data(ciphertext.parms_id())
            chain_index = ctx_data.chain_index()
            return float(chain_index * self._scale_bits)
        except Exception:
            pass
        return float(self._mult_depth * self._scale_bits)

    def get_mult_depth_used(self, ciphertext: Any) -> int:
        try:
            ctx_data = self._context.get_context_data(ciphertext.parms_id())
            total = self._context.get_context_data(
                self._context.first_parms_id()
            ).chain_index()
            remaining = ctx_data.chain_index()
            return int(total - remaining)
        except Exception:
            return 0

    def get_scheme_name(self) -> str:
        return "CKKS-SEAL"

    def _setup_context(
        self,
        poly_modulus_degree: int,
        mult_depth: int,
        scale_bits: int,
    ) -> tuple:
        import seal

        parms = seal.EncryptionParameters(seal.scheme_type.ckks)
        parms.set_poly_modulus_degree(poly_modulus_degree)

        middle = [scale_bits] * max(1, mult_depth)
        bit_sizes = [60] + middle + [60]
        parms.set_coeff_modulus(
            seal.CoeffModulus.Create(poly_modulus_degree, bit_sizes)
        )

        context = seal.SEALContext(parms)
        keygen = seal.KeyGenerator(context)
        secret_key = keygen.secret_key()

        public_key = seal.PublicKey()
        keygen.create_public_key(public_key)

        relin_keys = seal.RelinKeys()
        keygen.create_relin_keys(relin_keys)

        encryptor = seal.Encryptor(context, public_key)
        decryptor = seal.Decryptor(context, secret_key)
        evaluator = seal.Evaluator(context)
        encoder = seal.CKKSEncoder(context)

        self._slot_count = encoder.slot_count()

        return context, encryptor, decryptor, evaluator, encoder, relin_keys
