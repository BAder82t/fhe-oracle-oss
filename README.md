# FHE Oracle

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Commercial license available](https://img.shields.io/badge/commercial-available-brightgreen.svg)](./COMMERCIAL.md)

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

On the reference logistic-regression benchmark in this repo (a CKKS
circuit with a polynomial sigmoid approximation defect), the oracle
finds divergence **4,259× larger** than random sampling at an equal
500-evaluation budget. Reproduce with:

```bash
pip install cma numpy
python benchmarks/patent_logistic_regression.py --seed 42
```

## How it works

- **CMA-ES search** over the input domain, guided by a noise-aware
  fitness that combines plaintext/FHE divergence with ciphertext
  noise-budget consumption and multiplicative-depth utilisation.
- **Adapters** for OpenFHE, Concrete ML, and TenSEAL turn on
  noise-guided search. A pure divergence fallback works without any
  native FHE library — useful for CI.
- **Output**: PASS/FAIL verdict, worst input, sensitivity map, and a
  structured JSON/Markdown report for artefact upload.

## Benchmarks

See [benchmarks/](./benchmarks/README.md) for reproducible circuits.
Numbers below are from live runs on this repo (500-evaluation budget,
deterministic seed 42):

| Circuit                                     | Dim | Random max error | Oracle max error | Ratio       |
|---------------------------------------------|-----|------------------|------------------|-------------|
| Logistic regression (reference)             | 5   | 3.5e-4           | 1.50             | **4,259×**  |
| Logistic regression (input-amplified mock)  | 8   | 2.7e-1           | 6.8e-1           | 2.5×        |
| Polynomial (depth 4)                        | 6   | 1.7e-2           | 1.9e-2           | 1.1×        |
| Dense + Chebyshev sigmoid                   | 10  | —                | 1.0e-1           | —           |

Pure-Python divergence-only benchmarks run in under one second on a
2020-era laptop. Library-comparison benchmarks (TenSEAL / OpenFHE /
Pyfhel) take 15–45 s/seed; reach the unified-circuit numbers via
`benchmarks/library_comparison.py`.

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

## Open-core split

This repo ships **two packages**:

- **`fhe-oracle`** — Core (AGPL-3.0). Public on PyPI. CMA-ES search,
  AutoOracle landscape probe, adapters, divergence fitness, hybrid
  random + CMA, coverage certificate. Caps `n_trials ≤ 1000`.
- **`fhe-oracle-pro`** — Pro. Commercial-only, registers via Python
  entry points (`fhe_oracle.heuristics`, `fhe_oracle.fitness`). When
  installed alongside Core, the public `AutoOracle(...)` API
  transparently uses Pro's noise-budget-aware fitness and named
  heuristic seed generators (Multiplication Magnifier, Depth Seeker,
  Near-Threshold Explorer). No code changes needed for migration.

See [fhe-oracle-pro/README.md](./fhe-oracle-pro/README.md) and
[COMMERCIAL.md](./COMMERCIAL.md) for the licence boundary.

## Features (v0.5)

- **Open-core split** — Pro extracted into `fhe-oracle-pro` package
  with Python entry-point registration. Core remains AGPL; Pro is
  commercial.
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
  default to `0.0` (paper §6.15 empirical evidence); pass
  `w_noise=0.5, w_depth=0.3` to restore v0.2 shaping behaviour.

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

Dual-licensed:

- **AGPL-3.0-or-later** — free for research, personal use, and
  AGPL-compatible open-source projects. See [LICENSE](./LICENSE).
- **Commercial** — for closed-source products, SaaS, or any use
  that cannot comply with AGPL's copyleft. See
  [COMMERCIAL.md](./COMMERCIAL.md) for scope and contact.

Quick rule of thumb: if you cannot release your product's source
under AGPL-3.0, you need a commercial licence. Contact
**b@vaultbytes.com**.

## Related work

- **[CipherExplain](https://vaultbytes.com/cipherexplain)** — full
  encrypted SHAP suite, homomorphic SHAP, DP privacy, EU AI Act
  compliance tooling

## Contact

b@vaultbytes.com
