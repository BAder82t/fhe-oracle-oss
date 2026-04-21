# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
# Patent pending: PCT/IB2026/053378
"""FHE library adapters for noise-guided oracle search.

Each adapter wraps a specific FHE backend (OpenFHE, Concrete, SEAL)
with a uniform interface:

    encrypt(x)                -> ciphertext
    decrypt(ct)               -> list[float]
    run_fhe_program(ct)       -> ciphertext
    get_noise_budget(ct)      -> float
    get_mult_depth_used(ct)   -> int
    get_scheme_name()         -> str

Adapters are optional — the pure-divergence API works without any FHE
library installed. Adapters unlock noise-guided search, which finds
more bugs per evaluation than divergence alone.
"""

from .base import FHEAdapter

__all__ = ["FHEAdapter"]
