# FHE Oracle Benchmarks

Reproducible circuits that compare FHE Oracle (CMA-ES adversarial
search) against random sampling in the same evaluation budget.

## Run

```bash
pip install cma numpy
python benchmarks/patent_logistic_regression.py   # headline 4,259x ratio
python benchmarks/logistic_regression.py
python benchmarks/polynomial_eval.py
python benchmarks/neural_layer.py
```

Each benchmark completes in under 60 seconds on a 2020-era laptop.

## What each benchmark measures

| File | Circuit | FHE backend |
|------|---------|-------------|
| `patent_logistic_regression.py` | w·x + b → polynomial sigmoid (defect) | simulated CKKS adapter |
| `logistic_regression.py` | w·x + b → sigmoid | concrete-ml (if installed) or calibrated mock |
| `polynomial_eval.py` | depth-4 polynomial over ℝ⁶ | calibrated mock |
| `neural_layer.py` | dense layer + Chebyshev sigmoid | deterministic approximation |

## Why a mock FHE function?

A useful benchmark has to run on any machine. concrete-ml and openfhe
require native builds that take tens of minutes to install and fail on
several common CI images. The mocks inject noise that matches the
structure of real CKKS precision bugs: dense baseline noise (~1e-4 to
1e-5) plus input-dependent amplification in the regions where real
circuits exhaust noise budget.

If concrete-ml is installed, `logistic_regression.py` auto-detects it
and uses the real FHE path. Otherwise it falls back to the mock.

## Expected result

On `patent_logistic_regression.py` (seed 42) the oracle finds
**4,259× larger divergence** than random sampling — the headline
figure from the patent evaluation. The smaller "mock CKKS" and
polynomial circuits shown in the top-level README produce more
modest 1–3× ratios; they exercise different noise regimes and serve
as faster sanity checks.

Swap the simulated adapter for a real compiled FHE circuit to
benchmark your own backend.
