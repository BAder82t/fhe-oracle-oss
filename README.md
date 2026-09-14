# FHE Oracle

[![CI](https://github.com/BAder82t/fhe-oracle/actions/workflows/ci.yml/badge.svg)](https://github.com/BAder82t/fhe-oracle/actions/workflows/ci.yml)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

Adversarial precision testing for Fully Homomorphic Encryption.
Finds CKKS bugs that random testing misses.

## Install

```bash
pip install fhe-oracle
```

Optional adapters:

```bash
pip install "fhe-oracle[tenseal]"     # CKKS via TenSEAL
pip install "fhe-oracle[openfhe]"     # CKKS / BGV / BFV via OpenFHE (Linux)
pip install "fhe-oracle[concrete]"    # TFHE via Concrete ML
```

## 30-second example

```python
import numpy as np
from fhe_oracle import FHEOracle

def plaintext_fn(x):
    return float(np.sum(np.asarray(x) ** 2))

def fhe_fn(x):
    # Stand-in for your FHE-compiled predict function.
    # Here: noise scales with input norm^2 (a CKKS depth-noise pattern),
    # with a hot zone that inflates the error 100x when |x|^2 > 8.
    v = float(np.sum(np.asarray(x) ** 2))
    base = 1e-5 * v
    amp = 100.0 if v > 8.0 else 1.0
    return plaintext_fn(x) + base * amp

oracle = FHEOracle(
    plaintext_fn=plaintext_fn,
    fhe_fn=fhe_fn,
    input_dim=4,
    input_bounds=[(-3.0, 3.0)] * 4,
    seed=0,
)
result = oracle.run(n_trials=300, threshold=1e-3)
print(result.verdict)      # "FAIL"
print(result.max_error)    # ~3.6e-2
print(result.worst_input)  # ~[3.0, 3.0, 3.0, -3.0]
```

Output:

```
OracleResult(verdict='FAIL', max_error=3.593336e-02, trials=304, elapsed=0.05s)
```

Swap the fixture for a real `fhe_fn` (e.g. `concrete-ml`'s
`predict_proba(x, fhe="execute")`) and the oracle will search
adversarially for inputs that break precision.

## Why this exists

FHE precision bugs are **input-localised**. A CKKS circuit that passes
on 99,999 random inputs in a row can return garbage on the 100,000th.
The inputs that trigger failure sit in narrow regions of the input
space — regions that scale with multiplicative depth and the
magnitude of intermediate ciphertexts — and those regions are vanishingly
unlikely to be hit by uniform random sampling.

Random testing wastes evaluations in safe parts of the input space.
An adversarial optimiser (CMA-ES, guided by a noise-budget-aware
fitness function) spends its budget climbing toward the failure
region instead, and finds bugs orders of magnitude larger than random
sampling in the same wall-clock budget.

The reference logistic-regression example illustrates a polynomial
approximation defect in a **synthetic CKKS-like circuit**. Random testing
uses the operational range `[-0.3, 0.3]^5`; the oracle searches
`[-5, 5]^5`. Its large error ratio reflects both the different domains
and the search methods, and is not a matched comparison or speedup.
Reproduce this illustration with:

```bash
pip install cma numpy
python benchmarks/sigmoid_defect_benchmark.py --seed 42
```

## How it works

- **CMA-ES search** over the input domain, guided by plaintext/FHE
  output divergence by default, with optional custom fitness plugins.
- **Adapters** for OpenFHE, Concrete ML, and TenSEAL connect supported
  circuits to divergence search. Optional plugins can supply additional
  fitness functions. Synthetic checks can run without native FHE libraries.
- **Output**: PASS/FAIL verdict, worst input, sensitivity map, and a
  structured JSON/Markdown report for artefact upload.

## Benchmarks

See [benchmarks/](./benchmarks/README.md) for reproducible circuits.
Historical results below are recorded in
[the 20-seed summary](benchmarks/results/n20_expansion_summary.csv).
They are not a fresh validation of v0.6.0. Ratios measure the maximum
error discovered, not runtime or number of bugs found.

| Real TenSEAL circuit / setting | Seeds | Median oracle/random max-error ratio | Oracle wins |
|---|---:|---:|---:|
| LR, matched (`lr_matched`, B=60) | 20 | 2.04× | 15/20 |
| Depth-4 polynomial, matched | 20 | 1.41× | 16/20 |
| Chebyshev, matched (`cheb_matched`, B=60) | 20 | 0.38× | 3/20 |

Results depend on circuit, domain, parameters and strategy. The synthetic
reference's historical 4,259× ratio compares different domains and is
excluded from this matched-results table. Approximation error between
an intended model and its polynomial surrogate must be distinguished
from the error introduced by encrypted execution of that surrogate.

For a customer evaluation, pin backend versions, use the same input domain
and tolerance for each method, count all model evaluations, and compare
both equal evaluation budgets and equal wall-clock budgets across seeds.

## Verdicts and evaluation errors

`PASS` means no threshold violation was observed during the specified
search; it is not proof of correctness, cryptographic security or
regulatory compliance. Coverage confidence is conditional on the
caller-supplied minimum failure-region measure.

Invalid evaluations abort the run instead of producing a PASS. The built-in
precision comparisons reject backend exceptions, non-finite values, empty
outputs and mismatched output shapes. Scalars and one-element vectors
are compatible; higher-dimensional shapes must match. `EvaluationError`
is exported for callers to catch. Custom fitness implementations must
propagate backend failures and return finite scores.

The CLI exits **0** for PASS, **1** for a measured precision FAIL and
**2** for a model, configuration or evaluation error. In CI, treat every
nonzero exit as a blocked check. Some backend/property callbacks may
propagate their original exception; they never imply a successful check.

## Supported integrations

Core includes pure divergence search. Noise-budget fitness and named
heuristic implementations require separately installed plugins; supplying
an adapter alone does not install them. Broken plugin entry points emit
warnings and fall back to available Core functionality.

TenSEAL, OpenFHE and Concrete are optional integrations with backend-specific
requirements. The Lattigo module is a restricted subprocess precision probe,
not a general Core adapter, and its Go binary must be built separately.
The package remains Alpha while backend/version validation expands.

## CI/CD integration

Drop `oracle_check.py` at your repo root:

```python
import os
from fhe_oracle import FHEOracle
from my_model import plaintext_fn, fhe_fn

oracle = FHEOracle(
    plaintext_fn=plaintext_fn,
    fhe_fn=fhe_fn,
    input_dim=10,
    input_bounds=[(-3.0, 3.0)] * 10,
)
result = oracle.run(
    n_trials=int(os.environ.get("ORACLE_N_TRIALS", "500")),
    threshold=float(os.environ.get("ORACLE_THRESHOLD", "0.01")),
)
print(result)
raise SystemExit(0 if result.verdict == "PASS" else 1)
```

Add `.github/workflows/fhe-precision.yml`:

```yaml
name: FHE Precision Test
on: [push, pull_request]

jobs:
  fhe-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install fhe-oracle
      - run: python oracle_check.py
        env:
          ORACLE_THRESHOLD: "0.01"
          ORACLE_N_TRIALS: "500"
```

Full template: [examples/github_action.yml](./examples/github_action.yml).

## Features (v0.6)

- **One-call check** — `check(plaintext_fn, fhe_fn, input_bounds)` runs
  `AutoOracle` and, on FAIL, automatically shrinks the witness and
  localizes the fault (if the circuit supports tracing), returning a
  `CheckResult` with a ready-to-print report. Replaces the run →
  check-verdict → shrink → trace → render sequence with one call.
- **CLI** — `fhe-oracle check model.py` runs the same flow against a
  Python file defining `plaintext_fn`/`fhe_fn`/`input_bounds` at module
  level (optionally `n_trials`/`threshold`/`seed` too, overridable via
  `--n-trials`/`--threshold`/`--seed`/`--format`/`--no-shrink`). Exits
  0 on PASS, 1 on FAIL, 2 on a model, configuration or evaluation error.
- **Witness shrinking** — `FHEOracle.shrink(result)` reduces a FAIL
  witness toward a reference point (default: box centre) via
  per-coordinate binary search, while divergence keeps meeting the
  original threshold. Returns a `ShrinkResult` with the minimised
  input and the shrink ratio achieved.
- **Fault localization** — `localize_fault(trace)` reads a
  `per_op_trace()` result and returns the `OperationStep` most likely
  responsible for the divergence, using the already-decrypted
  per-step values (no perturbation/guessing required).
- **Circuit structure diagnostic** — `characterize_structure(fn, dim,
  bounds)` estimates whether a target function has exploitable
  low-rank structure before you reach for `separable=True` or a
  `SubspaceOracle` `subspace_dim`. Returns a `StructureReport` with an
  `effective_rank` estimate and a plain-English recommendation.
- **Structure-aware `AutoOracle` routing** — a new `LOW_RANK_STRUCTURE`
  regime measures the divergence surface's actual rank via
  `characterize_structure` and dispatches to `separable=True` CMA-ES
  only when real low-rank structure is found (dimension alone is not
  used, avoiding the earlier `d>100` heuristic's regression on
  isotropic circuits).
- **Multi-library differential testing** — `differential_test(adapter_a,
  adapter_b, input_dim, ...)` searches for inputs where two FHE
  adapters' decrypted outputs disagree, using only `encrypt`/`decrypt`
  (no noise-budget API, no reference plaintext function needed).
- **Property-based fitness functions** — `AdditivityFitness` and
  `ScalarLinearityFitness` search for algebraic-property violations
  (`f(a+b) != f(a)+f(b)`, `f(c*x) != c*f(x)`) as drop-in `FHEOracle`
  fitness objects.
- **CI diagnostics** — `report.to_markdown`/`to_json` accept an
  optional `diagnostics` dict, rendered under a `## Diagnostics`
  section on FAIL. `examples/oracle_check.py` is a real, runnable CI
  script demonstrating the full shrink-and-report flow.
- **Adapter-agnostic tracing** — `TracingCircuit` generalises
  `TracingTenSEALFn`'s per-operation tracing pattern to any `FHEAdapter`
  for a declared sequence of named steps.

## Features (v0.5)

- **Cross-library benchmark harness** — `benchmarks/library_comparison.py`
  drives the same `(w·x+b)²` circuit through every installed adapter
  and emits a single CSV per family (CKKS / integer).
- **`sigma0=None` auto-scale** + **`DISTANT_DEFECT` regime** in
  `AutoOracle` — handles landscapes where the failure region sits
  outside the initial search ball.

## Features (v0.4)

- **Periodic diversity injection** — `FHEOracle(...,
  diversity_injection=True, inject_every=5, inject_count=3)` injects
  diverse candidates (corner / uniform / best-neighbour) into the
  CMA-ES population every N generations, preventing the covariance
  collapse that strands vanilla CMA-ES on plateau landscapes. See
  `fhe_oracle.diversity.DiversityInjector`.
- **Adaptive budget allocation** — `FHEOracle(..., adaptive=True)`
  enables three behaviours simultaneously: early stop on a
  definitive FAIL, auto-extension when divergence is still climbing
  at budget exhaustion, and strategy-switch to uniform random when
  CMA-ES's step size collapses on a plateau. Configure via
  `AdaptiveConfig`.
- **Multi-output rank-aware fitness** — `FHEOracle(...,
  multi_output=True, multi_output_mode="combined")` wraps the
  user's vector-valued plaintext/FHE pair in a `MultiOutputFitness`
  that targets decision-altering precision failures (argmax flips,
  near-margin inputs) on top of max-absolute error. Use
  `MultiOutputFitness.detailed_report(x)` to inspect a witness.
- **All three default OFF** — backward-compatible with v0.3.x.
  Opt in per-call or via `AutoOracle(..., adaptive=True,
  diversity_injection=True)` (kwargs are forwarded to the inner
  `FHEOracle`).

## Features (v0.3)

- **Auto-configuration probe** — `AutoOracle(...)` runs a 50-eval
  probe to classify the divergence landscape
  (`FULL_DOMAIN_SATURATION`, `PLATEAU_THEN_CLIFF`,
  `PREACTIVATION_DOMINATED`, `STANDARD`) and dispatches to the
  best search strategy automatically. No prior paper reading
  required.
- **Random subspace embedding** (experimental) —
  `SubspaceOracle(...)` projects `d >> 100` inputs into `k`-dim
  random subspaces and searches with CMA-ES. Currently benefits
  only low-rank hidden-layer quantisation bugs; for dense
  directional / corner-region bugs prefer `PreactivationOracle`
  (when `W, b` are available) or uniform random sampling. See
  `research/release/v030-benchmark-report.md` for the evaluation.
- **Pure-divergence defaults** — `w_noise` and `w_depth` now
  defaulted to `0.0` in v0.3 (paper §6.15 empirical evidence).
  Those arguments were removed in v0.5.1; current Core uses divergence
  fitness and optional plugin providers.

## Features (v0.2)

- **Pure-divergence mode** — CI-friendly, no FHE library required.
- **Hybrid random + CMA-ES with warm-start** — `random_floor=0.3`
  on `FHEOracle` reserves a fraction of the budget for uniform
  sampling, then warm-starts CMA-ES at the best random point.
- **IPOP / BIPOP restarts** — `restarts=N, bipop=True` on
  `FHEOracle` for multi-basin landscapes.
- **Separable CMA-ES** — `separable=True` for axis-aligned
  landscapes (high-dim settings).
- **Union verdict (oracle + empirical)** — `run_hybrid(...)`
  returns a `HybridResult`; PASS iff both the adversarial and
  training-distribution legs pass.
- **Coverage certificate** — the random-floor phase produces a
  `CoverageCertificate` attached to `OracleResult`; pair with
  `budget_for(eta, p)` or `pass_confidence(eta)` for a
  probabilistic PASS statement.
- **Preactivation search** — `PreactivationOracle(W, b, ...)`
  searches in preactivation z-space, collapsing d=784 affine
  front-ends to a rank-k subproblem.
- **Cascade (multi-fidelity) search** — `CascadeSearch(...)`
  runs cheap-fidelity search then re-scores the top-K under an
  expensive fidelity.
- **Per-operation trace diagnostic** — `per_op_trace(x, plain,
  fhe)` and `TracingTenSEALFn` localise where error accumulates
  in a CKKS circuit.
- **TenSEAL adapter** — `pip install fhe-oracle[tenseal]`
  enables noise-guided search on CKKS.

## Licensing

AGPL-3.0-or-later. See [LICENSE](./LICENSE).
