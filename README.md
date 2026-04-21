# FHE Oracle

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Commercial license available](https://img.shields.io/badge/commercial-available-brightgreen.svg)](./COMMERCIAL.md)
[![Patent pending](https://img.shields.io/badge/patent-PCT%2FIB2026%2F053378-orange.svg)](./NOTICE)

Adversarial precision testing for Fully Homomorphic Encryption.
Finds CKKS bugs that random testing misses.

## FHE library leaderboard

### 🎯 Picking a library — TL;DR

- Integer / PIR / encrypted SQL → **OpenFHE BFV**
- Quantised ML inference → **Concrete ML** (TFHE)
- Real-valued CKKS (LR, neural net, SHAP) → **TenSEAL**

### A note on "same test for all"

FHE schemes split into two families that **cannot share a single
mathematical circuit without bias**:

- **CKKS** is real-valued with bounded approximation error. Natural test:
  `(w·x + b)²` on real `[-3, 3]^8`.
- **BGV / BFV / TFHE** are exact integer/quantised arithmetic. Natural test:
  `(w_int·x_int + b_int)² mod p` on quantised integer inputs.

Forcing integer schemes to approximate real-valued `(w·x + b)²` injects
quantisation error (order 1/s², where s is the quantisation scale) that
dominates any library-level precision signal. Conversely, running CKKS on a
modular-integer circuit wastes its real-valued precision.

**Our compromise:** CKKS libraries are benchmarked on the **identical CKKS
circuit** (`(w·x + b)²` on real inputs, depth 2) — their comparison is
directly apples-to-apples. Integer-scheme libraries are benchmarked on the
**identical integer circuit** (`(w·x_int + b)² mod p` at depth 2) — also
apples-to-apples within their family. The two tables below report each.

### Leaderboard — Integer / quantised schemes

Circuit: `(w_int·x_int + b_int)² mod p = 65537` on quantised-int16 `[-3, 3]^8`.
Every library in this table runs **the same integer circuit**.

| Rank  | Library              | Scheme | Fail rate    | Median max-err | Wall      |
|-------|----------------------|--------|--------------|----------------|-----------|
| 🥇 1  | **OpenFHE BFV 1.5+** | BFV    | **0 %** (0/5) | **0**         | **12.7 s**|
| 🥈 2  | **OpenFHE BGV 1.5+** | BGV    | **0 %** (0/5) | **0**         | 15.6 s    |
| 🥉 3  | **Concrete ML 1.9.0**| TFHE   | 33 % (1/3)    | 0             | 0.6 s     |

BFV and BGV are bit-exact — zero divergence on every seed. Concrete ML's 33% FAIL is a quantisation-boundary crossing effect, not an algorithmic error. TFHE wins on wall-clock by 20× (0.6 s vs 12.7 s) — the tradeoff: TFHE operates on small integers with fixed bit-widths, BGV/BFV handle larger integer ranges.

### Leaderboard — CKKS (real-valued)

Circuit: `(w·x + b)²` on real-valued `[-3, 3]^8`, depth 2. Every library in
this table runs **the exact same circuit**. `AutoOracle(B=500)`, 5 seeds,
threshold = 1e-2.

| Rank  | Library                                          | Scheme | Fail rate       | Median max-err | Wall     |
|-------|--------------------------------------------------|--------|-----------------|----------------|----------|
| 🥇 1  | **OpenFHE 1.5+**                                 | CKKS   | **0 % (0/5)** ✅ | **1.57e-08**   | 21.2 s   |
| 🥈 2  | **Pyfhel 3.5.0**                                 | CKKS   | 100 % (5/5)     | 1.30e-03       | 43.9 s   |
| 🥉 3  | **TenSEAL 0.3.16** (Microsoft SEAL CKKS wrapper) | CKKS   | 100 % (5/5)     | 2.20e-03       | **17.7 s** |

**OpenFHE is the only CKKS library that passes.** AutoOracle's 500-eval
adversarial search couldn't push OpenFHE's max-err above 2.7e-08 across any
of 5 seeds; TenSEAL and Pyfhel both hit ~1–2e-3. On identical math, OpenFHE
is **~140,000× more precise than TenSEAL** and **~80,000× more precise than
Pyfhel**.

This is a genuine library-level split. All three run `(w·x + b)²` with CKKS
ring dimension 16384 and coefficient-modulus chain [60, 40, 40, 40, 40, 60].
Hypothesised cause: OpenFHE rescales after every ciphertext multiplication
by default; TenSEAL defers rescales. Ring-level correctness seems fine in
all three (smoke tests agree on scalar outputs to 6 decimal places); the
adversarial oracle is what exposes the difference on worst-case inputs.

**Trade-off for CKKS users:**

- **Need precision** → OpenFHE (100× less error than TenSEAL/Pyfhel).
- **Need speed** → TenSEAL (1.2× faster wall-clock than OpenFHE, 2.5× faster than Pyfhel). You accept ~140,000× worse precision in return.
- **Avoid for multi-level circuits** → Pyfhel. Slowest + requires manual level management for depth > 2.

### Pyfhel caveat

Pyfhel exposes a lower-level SEAL API than TenSEAL — every
ciphertext × ciphertext multiplication requires the caller to manage
`rescale_to_next` and `mod_switch_to_next` manually. On the depth-2
`(w·x+b)²` circuit Pyfhel works (shown above). On higher-degree
polynomials (Taylor-3 with a `z³` term) Pyfhel's scale management fails
with `ValueError: scale out of bounds`. Its middle-of-the-pack result on
the unified benchmark suggests the SEAL implementation is reasonable —
the Pyfhel API wrapper is the real cost.

### Lattigo (Go) — deferred

Go-only; requires a subprocess wrapper to call from Python. Lattigo ships a
built-in noise estimator that could serve as an Item 07 probe path; tracked
in `research/future-work/` as future work.

### Three things to read

1. **OpenFHE is the only CKKS library that passes.** On the identical depth-2 `(w·x+b)²` circuit, AutoOracle's 500-evaluation adversarial search cannot drive OpenFHE above max-err 2.7e-08 across 5 seeds — it passes the 1e-2 threshold on every seed. TenSEAL and Pyfhel both FAIL with max-err ~1–2e-3. That's a **~140,000× gap** between OpenFHE and TenSEAL on the same math.
2. **Exact schemes (BGV, BFV, TFHE) are bit-exact by design** — their rows reward library correctness. BFV/BGV at 0% FAIL means OpenFHE's implementation is correct. Concrete ML's 33% reflects a TFHE quantisation-boundary effect, not implementation error.
3. **FHE Oracle is scheme-agnostic.** Same `AutoOracle(...)` drives every row above. Add a library by writing one adapter; leaderboard updates on re-run. PRs welcome.

### Reproduce

```bash
# CKKS unified (w·x+b)^2 benchmark, runs in any Py venv with TenSEAL
pip install fhe-oracle tenseal
python benchmarks/library_comparison.py --circuit unified-squared-dot --libs tenseal

# Full leaderboard (OpenFHE BGV/BFV/CKKS) via Linux/amd64 Docker
docker build --platform linux/amd64 -t fhe-oracle-bench .
docker run --rm --platform linux/amd64 fhe-oracle-bench \
    python benchmarks/library_comparison.py --circuit unified-squared-dot
```

## Install

```bash
pip install fhe-oracle
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

On the patent-reference benchmark in this repo (a CKKS logistic
regression circuit with a polynomial sigmoid approximation defect),
the oracle finds divergence **4,259× larger** than random sampling
at an equal 500-evaluation budget. Reproduce with:

```bash
pip install cma numpy
python benchmarks/patent_logistic_regression.py --seed 42
```

## How it works

- **CMA-ES search** over the input domain, guided by a noise-aware
  fitness that combines plaintext/FHE divergence with ciphertext
  noise-budget consumption and multiplicative-depth utilisation.
- **Adapters** for OpenFHE, Concrete ML, and SEAL turn on
  noise-guided search. A pure divergence fallback works without any
  native FHE library — useful for CI.
- **Output**: PASS/FAIL verdict, worst input, sensitivity map, and a
  structured JSON/Markdown report for artefact upload.

## Benchmarks

See [benchmarks/](./benchmarks/README.md) for reproducible circuits:

Numbers below are from live runs on this repo (500-evaluation budget,
deterministic seed 42):

| Circuit | Dim | Random max error | Oracle max error | Ratio |
|---------|-----|------------------|------------------|-------|
| Logistic regression (patent reference) | 5 | 3.5e-4 | 1.50 | **4,259×** |
| Logistic regression (input-amplified mock) | 8 | 2.7e-1 | 6.8e-1 | 2.5× |
| Polynomial (depth 4) | 6 | 1.7e-2 | 1.9e-2 | 1.1× |
| Dense + Chebyshev sigmoid | 10 | — | 1.0e-1 | — |

Each benchmark runs in under one second on a 2020-era laptop.

The `concrete-ml` path in `logistic_regression.py` is a stub wrapper;
replace it with a compiled `predict_proba(x, fhe="execute")` call to
exercise a real FHE backend.

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

## Patent notice

A patent application covering this method has been filed
(**PCT/IB2026/053378**, "System and Method for Adversarial
Noise-Guided Differential Testing of Fully Homomorphic Encryption
Programs"). The open-source code here is licensed under **AGPL-3.0**.

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

- **PCT/IB2026/053378** — FHE Differential Testing Oracle (this
  project's patent application)
- **[CipherExplain](https://vaultbytes.com/cipherexplain)** — full
  encrypted SHAP suite, homomorphic SHAP, DP privacy, EU AI Act
  compliance tooling

## Contact

b@vaultbytes.com
