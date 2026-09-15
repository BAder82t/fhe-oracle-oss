# Copyright (C) 2026 Bader Alissaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Adapter for OpenFHE (Python bindings: openfhe).

Installation
------------
    pip install openfhe

The adapter sets up a CKKS crypto context with configurable security,
generates keys, and wraps a user-supplied homomorphic function.

Example
-------
    from fhe_oracle.adapters.openfhe import OpenFHEAdapter

    def lr_program(cc, ct):  # z = w . x + b lands in slot 0
        prod = cc.EvalMult(ct, cc.MakeCKKSPackedPlaintext(weights))
        return cc.EvalAdd(cc.EvalSum(prod, 8), bias)

    adapter = OpenFHEAdapter(
        fhe_fn=lr_program, n_features=8, mult_depth=2, output_length=1
    )

``output_length`` is the number of leading slots holding the program's result;
it defaults to ``n_features`` for element-wise programs such as ``cc.EvalMult(ct, ct)``.
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
        output_length: int | None = None,
    ) -> None:
        try:
            import openfhe  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                f"OpenFHE Python bindings are not importable ({exc}). "
                "Install with: pip install openfhe"
            ) from exc

        self._fhe_fn = fhe_fn
        self._n_features = n_features
        self._mult_depth = mult_depth
        self._scale_mod_size = scale_mod_size
        self._batch_size = 1 << max(0, n_features - 1).bit_length()
        self._output_length = n_features if output_length is None else output_length
        if not 1 <= self._output_length <= self._batch_size:
            raise ValueError(
                f"output_length must be in [1, {self._batch_size}], got {self._output_length}"
            )
        self._cc, self._kp = self._setup_context(
            mult_depth, scale_mod_size, n_features, security_level
        )

    def encrypt(self, x: list[float]) -> Any:
        padded = list(x) + [0.0] * max(0, self._batch_size - len(x))
        pt = self._cc.MakeCKKSPackedPlaintext(padded)
        return self._cc.Encrypt(self._kp.publicKey, pt)

    def decrypt(self, ciphertext: Any) -> list[float]:
        pt = self._cc.Decrypt(self._kp.secretKey, ciphertext)
        pt.SetLength(self._output_length)
        return list(pt.GetRealPackedValue()[: self._output_length])

    def run_fhe_program(self, ciphertext: Any) -> Any:
        return self._fhe_fn(self._cc, ciphertext)

    def get_noise_budget(self, ciphertext: Any) -> float:
        try:
            level_used = ciphertext.GetLevel()
            total_bits = self._mult_depth * self._scale_mod_size
            consumed_bits = level_used * self._scale_mod_size
            return max(0.0, float(total_bits - consumed_bits))
        except Exception:  # noqa: BLE001 - metadata-only fallback; max_error and verdict unaffected
            return max(0.0, float(self._mult_depth * self._scale_mod_size))

    def get_mult_depth_used(self, ciphertext: Any) -> int:
        try:
            return int(ciphertext.GetLevel())
        except Exception:  # noqa: BLE001 - metadata-only fallback; max_error and verdict unaffected
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

        params = openfhe.CCParamsCKKSRNS()
        params.SetMultiplicativeDepth(mult_depth)
        params.SetScalingModSize(scale_mod_size)
        params.SetBatchSize(self._batch_size)

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
        cc.Enable(openfhe.PKESchemeFeature.ADVANCEDSHE)  # EvalSumKeyGen throws without it

        kp = cc.KeyGen()
        cc.EvalMultKeyGen(kp.secretKey)
        cc.EvalSumKeyGen(kp.secretKey)

        return cc, kp
