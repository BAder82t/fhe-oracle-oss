# FHE Oracle Benchmarks

Reproducible circuits that compare FHE Oracle (CMA-ES adversarial
search) against random sampling in the same evaluation budget.

## Run

```bash
pip install cma numpy
python benchmarks/sigmoid_defect_benchmark.py   # synthetic reference (asymmetric domains)
python benchmarks/logistic_regression.py
python benchmarks/polynomial_eval.py
python benchmarks/neural_layer.py
```

Each benchmark completes in under 60 seconds on a 2020-era laptop.

## What each benchmark measures

| File | Circuit | FHE backend |
|------|---------|-------------|
| `sigmoid_defect_benchmark.py` | w·x + b → polynomial sigmoid (defect) | simulated CKKS adapter |
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

`sigmoid_defect_benchmark.py` is an illustration, not a matched
comparison: random sampling uses the operational range `[-0.3, 0.3]^5`
while the oracle searches `[-5, 5]^5`. Its large error ratio (about
4,259× at seed 42) reflects both the wider domain and the search
method, so it is not a speedup or evidence of algorithmic superiority.
The mock circuits exercise different noise regimes and serve as fast
sanity checks.

Matched real-CKKS results (TenSEAL, same domain and budget for both
methods, 20 seeds) are in
[`results/n20_expansion_summary.csv`](results/n20_expansion_summary.csv).
The oracle finds larger errors on some circuits (logistic regression,
depth-4 polynomial) and smaller ones on others (Chebyshev).

Swap the simulated adapter for a real compiled FHE circuit to
benchmark your own backend.
