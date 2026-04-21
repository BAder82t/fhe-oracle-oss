# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
# Patent pending: PCT/IB2026/053378
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
