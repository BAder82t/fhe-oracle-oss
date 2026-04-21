# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
# Patent pending: PCT/IB2026/053378
"""Adapter for OpenFHE (Python bindings: openfhe).

Installation
------------
    pip install openfhe

The adapter sets up a CKKS crypto context with configurable security,
generates keys, and wraps a user-supplied homomorphic function.

Example
-------
    from fhe_oracle.adapters.openfhe import OpenFHEAdapter

    def my_fhe_fn(cc, ct):
        return cc.EvalMult(ct, ct)  # element-wise square

    adapter = OpenFHEAdapter(
        fhe_fn=my_fhe_fn, n_features=4, mult_depth=2
    )
"""

from __future__ import annotations

from typing import Any, Callable

from .base import FHEAdapter


class OpenFHEAdapter(FHEAdapter):
    """FHEAdapter backed by OpenFHE CKKS."""

    def __init__(
        self,
        fhe_fn: Callable[[Any, Any], Any],
        n_features: int,
        mult_depth: int = 2,
        scale_mod_size: int = 50,
        security_level: int = 128,
    ) -> None:
        try:
            import openfhe  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "OpenFHE Python bindings are not installed. "
                "Install with: pip install openfhe"
            )

        self._fhe_fn = fhe_fn
        self._n_features = n_features
        self._mult_depth = mult_depth
        self._cc, self._kp = self._setup_context(
            mult_depth, scale_mod_size, n_features, security_level
        )

    def encrypt(self, x: list[float]) -> Any:
        padded = list(x) + [0.0] * max(0, self._batch_size - len(x))
        pt = self._cc.MakeCKKSPackedPlaintext(padded)
        return self._cc.Encrypt(self._kp.publicKey, pt)

    def decrypt(self, ciphertext: Any) -> list[float]:
        pt = self._cc.Decrypt(self._kp.secretKey, ciphertext)
        values = pt.GetRealPackedValue()
        return list(values[: self._n_features])

    def run_fhe_program(self, ciphertext: Any) -> Any:
        return self._fhe_fn(self._cc, ciphertext)

    def get_noise_budget(self, ciphertext: Any) -> float:
        try:
            level_used = ciphertext.GetLevel()
            total_bits = self._mult_depth * 50
            consumed_bits = level_used * 50
            return max(0.0, float(total_bits - consumed_bits))
        except Exception:
            return max(0.0, float(self._mult_depth * 50))

    def get_mult_depth_used(self, ciphertext: Any) -> int:
        try:
            return int(ciphertext.GetLevel())
        except Exception:
            return 0

    def get_scheme_name(self) -> str:
        return "CKKS-OpenFHE"

    def _setup_context(
        self,
        mult_depth: int,
        scale_mod_size: int,
        n_features: int,
        security_level: int,
    ) -> tuple[Any, Any]:
        import openfhe

        batch = 1
        while batch < n_features:
            batch <<= 1
        self._batch_size = batch

        params = openfhe.CCParamsCKKSRNS()
        params.SetMultiplicativeDepth(mult_depth)
        params.SetScalingModSize(scale_mod_size)
        params.SetBatchSize(batch)

        sec_map = {
            128: openfhe.SecurityLevel.HEStd_128_classic,
            192: openfhe.SecurityLevel.HEStd_192_classic,
            256: openfhe.SecurityLevel.HEStd_256_classic,
        }
        params.SetSecurityLevel(
            sec_map.get(security_level, openfhe.SecurityLevel.HEStd_128_classic)
        )

        cc = openfhe.GenCryptoContext(params)
        cc.Enable(openfhe.PKESchemeFeature.PKE)
        cc.Enable(openfhe.PKESchemeFeature.LEVELEDSHE)

        kp = cc.KeyGen()
        cc.EvalMultKeyGen(kp.secretKey)
        cc.EvalSumKeyGen(kp.secretKey)

        return cc, kp
