# Copyright (C) 2026 Bader Alissaei
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Release verification: imports + API surface + __init__ exports.

Run:

    python scripts/verify_release.py

Exits 0 on success, 1 on any failure.
"""

from __future__ import annotations

import importlib
import inspect
import sys
from typing import Iterable


MODULES: list[str] = [
    "fhe_oracle",
    "fhe_oracle.core",
    "fhe_oracle.empirical",
    "fhe_oracle.hybrid",
    "fhe_oracle.guarantees",
    "fhe_oracle.diagnostics",
    "fhe_oracle.preactivation",
    "fhe_oracle.cascade",
    "fhe_oracle.heuristics",
    "fhe_oracle.adapters",
    "fhe_oracle.adapters.tenseal_adapter",
]

EXPECTED_INIT_PARAMS: list[str] = [
    "random_floor",
    "warm_start",
    "warm_sigma_scale",
    "restarts",
    "bipop",
    "separable",
    "seed",
]

EXPECTED_PUBLIC: list[str] = [
    "FHEOracle",
    "OracleResult",
    "EmpiricalSearch",
    "EmpiricalResult",
    "run_hybrid",
    "HybridResult",
    "CoverageCertificate",
    "confidence_adjusted_pass",
    "per_op_trace",
    "ComponentLog",
    "InstrumentedFitness",
    "OperationTrace",
    "OperationStep",
    "TracingTenSEALFn",
    "PreactivationOracle",
    "PreactivationResult",
    "CascadeSearch",
    "CascadeResult",
    "evaluate_correlation",
    "DivergenceFitness",
    "NoiseGuidedFitness",
]


def _check_imports(names: Iterable[str]) -> list[str]:
    failures: list[str] = []
    for name in names:
        try:
            importlib.import_module(name)
            print(f"OK  import {name}")
        except Exception as exc:  # pragma: no cover - defensive
            failures.append(f"{name}: {exc}")
            print(f"FAIL import {name}: {exc}")
    return failures


def _check_api() -> list[str]:
    failures: list[str] = []
    from fhe_oracle.core import FHEOracle  # noqa: WPS433

    sig = inspect.signature(FHEOracle.__init__)
    for param in EXPECTED_INIT_PARAMS:
        if param not in sig.parameters:
            failures.append(f"FHEOracle.__init__ missing kwarg: {param}")
            print(f"FAIL FHEOracle.__init__ missing: {param}")
        else:
            print(f"OK  FHEOracle.__init__ has {param}")

    import fhe_oracle

    for name in EXPECTED_PUBLIC:
        if not hasattr(fhe_oracle, name):
            failures.append(f"fhe_oracle.{name} missing")
            print(f"FAIL fhe_oracle.{name} missing")
        else:
            print(f"OK  fhe_oracle.{name}")

    if not hasattr(fhe_oracle, "__version__"):
        failures.append("fhe_oracle.__version__ missing")
        print("FAIL __version__ missing")
    else:
        print(f"OK  version = {fhe_oracle.__version__}")

    return failures


def main() -> int:
    print("== imports ==")
    failures = _check_imports(MODULES)
    print("\n== API surface ==")
    failures += _check_api()
    print()
    if failures:
        print(f"{len(failures)} failures:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
