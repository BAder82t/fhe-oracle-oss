# Pre-registration, 15 September 2026

Written and committed before any of the runs below. Results that deviate from
this plan must be labelled post hoc.

Common settings: real TenSEAL CKKS (`benchmarks/tenseal_circuits.py`, one
context and key set per circuit), threshold 0.01, `n_trials` = 200, every
model evaluation counted, paired two-sided Wilcoxon tests with Holm correction
within each study. Seeds 11–30 have not been used in any earlier run.
The library commit and script SHA-256 are recorded with every result.

## A. AutoOracle boundary-probe confirmation

The 15 September evaluation compared the boundary probe against an
`AutoOracle` that exceeded its budget (221 evaluations at `n_trials` = 200),
so its budget-matched analysis was post hoc. This study repeats it with the
correct comparator on fresh seeds.

- Circuits: lr_d8, cheb_d10, poly_d6 (TenSEAL); bump_d8, decoy_d6 (interior
  worst-case mocks).
- Arms: ACCOUNTING-ONLY (probe evaluations counted, no boundary probe) and
  AFTER (committed `AutoOracle`), seeds 11–30. Corner/boundary set and uniform
  random at the same counted evaluations as reference baselines.
- Metric: per-seed maximum error over all counted evaluations.
- Pass, per circuit:
  1. AFTER / ACCOUNTING-ONLY median ratio ≥ 0.95 and not significantly lower
     (Holm-adjusted p < 0.05 with AFTER lower);
  2. on circuits whose supremum is at a box vertex (lr_d8, cheb_d10), AFTER ≥
     corner set minus 1e-5 on at least 16 of 20 seeds;
  3. neither arm exceeds `n_trials` plus its documented re-measurement.
- Decision: keep the boundary probe only if all circuits pass. Otherwise
  revert it and keep the accounting fix.

## B. Plaintext-surrogate search, then FHE verification

Value unproven. On the sample-report circuit almost all error is
approximation error that is visible without encryption.

- Circuit: lr_d8, domain [-3, 3]^8, 20 seeds (11–30).
- Arms at equal wall-clock (each seed gets the median wall time of AutoOracle
  at `n_trials` = 200 on that seed):
  1. SURROGATE: CMA-ES on |σ − T3| in plaintext, then the top-K distinct
     candidates (K = 20) evaluated under CKKS; reported error is the CKKS
     measurement.
  2. AutoOracle at `n_trials` = 200.
  3. Corner/boundary set.
- Pass: SURROGATE ≥ corner set minus 1e-5 on at least 18 of 20 seeds, and
  median SURROGATE error ≥ 0.99 × the plaintext supremum 140.9204.
- Decision: build a `surrogate_fn` API only if it passes.

## C. Search that targets CKKS execution error

Value unproven. This is the error class plaintext testing cannot reveal.

- Circuits: lr_d8 and the Chebyshev degree-15 circuit at a 30-bit scale
  (`benchmarks/cheb15_cross_circuit.py`), where execution error dominates.
- Fitness for the tool arm: |T3(x) − CKKS(x)| (surrogate plaintext vs
  encrypted surrogate), FHEOracle defaults, `n_trials` = 200, 10 seeds (11–20).
- Baselines at the same counted evaluations: uniform random, Sobol sequence,
  corner/boundary set.
- Metric: per-seed maximum execution error.
- Pass: the tool arm ≥ the best baseline on at least 8 of 10 seeds for at least
  one circuit.
- Decision: if it fails on both circuits, state in the README that the tool
  does not outperform simple sampling at locating CKKS execution error.
