# Copyright (C) 2026 Bader Issaei / VaultBytes Innovations Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
# Patent pending: PCT/IB2026/053378
"""FHE Oracle: adversarial precision testing for FHE programs.

Public API
----------
    from fhe_oracle import FHEOracle

    oracle = FHEOracle(
        plaintext_fn=my_model.predict,
        fhe_fn=my_fhe_model.predict,
        input_dim=10,
        input_bounds=[(-3.0, 3.0)] * 10,
    )
    result = oracle.run(n_trials=500, threshold=0.01)
    print(result.verdict)      # "PASS" or "FAIL"
    print(result.max_error)    # largest divergence found
    print(result.worst_input)  # input that triggered it
"""

from .adaptive import AdaptiveBudget, AdaptiveConfig
from .autoconfig import AutoOracle, ProbeResult, Regime, classify_landscape
from .cascade import CascadeResult, CascadeSearch, evaluate_correlation
from .core import FHEOracle, OracleResult
from .diagnostics import (
    ComponentLog,
    InstrumentedFitness,
    OperationStep,
    OperationTrace,
    TracingTenSEALFn,
    per_op_trace,
)
from .diversity import DiversityInjector, InjectionStrategy
from .empirical import EmpiricalResult, EmpiricalSearch
from .fitness import DivergenceFitness
from .guarantees import CoverageCertificate, confidence_adjusted_pass
from .hybrid import HybridResult, run_hybrid
from .multi_output import MultiOutputFitness, MultiOutputMode
from .preactivation import PreactivationOracle, PreactivationResult
from .subspace import SubspaceOracle

__all__ = [
    "AdaptiveBudget",
    "AdaptiveConfig",
    "AutoOracle",
    "CascadeResult",
    "CascadeSearch",
    "ComponentLog",
    "CoverageCertificate",
    "DivergenceFitness",
    "DiversityInjector",
    "EmpiricalResult",
    "EmpiricalSearch",
    "FHEOracle",
    "HybridResult",
    "InjectionStrategy",
    "InstrumentedFitness",
    "MultiOutputFitness",
    "MultiOutputMode",
    "OperationStep",
    "OperationTrace",
    "OracleResult",
    "PreactivationOracle",
    "PreactivationResult",
    "ProbeResult",
    "Regime",
    "SubspaceOracle",
    "TracingTenSEALFn",
    "classify_landscape",
    "confidence_adjusted_pass",
    "evaluate_correlation",
    "per_op_trace",
    "run_hybrid",
]
__version__ = "0.5.0"
