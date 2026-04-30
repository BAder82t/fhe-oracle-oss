# Copyright (C) 2026 Bader Alissaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Abstract base class for FHE library adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class FHEAdapter(ABC):
    """Uniform interface for FHE backends used by the oracle."""

    @abstractmethod
    def encrypt(self, x: list[float]) -> Any: ...

    @abstractmethod
    def decrypt(self, ciphertext: Any) -> list[float]: ...

    @abstractmethod
    def run_fhe_program(self, ciphertext: Any) -> Any: ...

    @abstractmethod
    def get_noise_budget(self, ciphertext: Any) -> float: ...

    @abstractmethod
    def get_mult_depth_used(self, ciphertext: Any) -> int: ...

    @abstractmethod
    def get_scheme_name(self) -> str: ...

    def evaluate(self, x: list[float]) -> list[float]:
        """Run the full encrypt -> compute -> decrypt path on ``x``.

        Default implementation chains the abstract methods. Subclasses
        may override for efficiency. Returns the decrypted output of
        ``run_fhe_program`` evaluated on the encryption of ``x``.
        """
        ct = self.encrypt(x)
        ct_out = self.run_fhe_program(ct)
        return self.decrypt(ct_out)
