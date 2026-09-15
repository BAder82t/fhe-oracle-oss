# Changelog

## 0.7.0

- `OpenFHEAdapter` construction failed on real OpenFHE (`EvalSumKeyGen` needs
  `ADVANCEDSHE`) and decrypted every slot; it now enables the feature and takes
  `output_length` (default `n_features`; use 1 for scalar outputs). Checked
  against TenSEAL on LR d=8 (relative witness-error gap 1.6e-5).
- `SealAdapter` takes `output_length` the same way (tested with a fake backend
  only). `ConcreteAdapter` sent float inputs to integer circuits; it now takes
  `quantize_fn`/`dequantize_fn` and returns flat float outputs. Added
  `adapters.concrete.predict_proba_fhe_fn(model, fhe=...)`.
- `benchmarks/library_comparison.py` turned Concrete backend and compile
  failures into 0.0 outputs or silent skips; errors now propagate, errored
  runs record NaN and the script exits 2.
- `AutoOracle` counts every probe evaluation (landscape probe, second pass,
  centre ball, structure diagnostic) against `n_trials`; it could spend 221
  evaluations at `n_trials=200`. The minimum budget is `n_probes + 3`
  (`n_probes + 10` with single-row `W, b`). The d ≥ 16 structure diagnostic
  runs only when the budget covers it (about 400·d + 50 evaluations), so
  low-rank routing needs a large budget. `AutoOracle`'s `result.n_trials` now
  counts probe evaluations too (every counted evaluation except
  re-measurements), matching `FHEOracle`; it previously reported 128 for a
  run that made 199 calls.
- `AutoOracle` probes box vertices within min(15% of `n_trials`, 4 + 2d)
  evaluations: it snaps the best probe points to their nearest vertices,
  tries the mirror, and climbs by single-coordinate flips when a vertex leads.
  A winning vertex is re-measured and follows the same verdict rule. In a
  pre-registered comparison with the same `AutoOracle` without the probe
  (`benchmarks/preregistration_2026-09-15.md`, study A: real TenSEAL, 20 fresh
  seeds, both arms within `n_trials=200`) the median maximum error rose 1.74×
  (LR d=8), 3.14× (Chebyshev d=10) and 1.52× (polynomial d=6), higher on every
  seed, and matched the corner set within 1e-5 on 20/20 (LR) and 16/20
  (Chebyshev) seeds. On interior-worst-case mocks the median was unchanged,
  but the probe's budget lowered the result on 3 and 6 of 20 seeds (Holm
  p = 0.109 and 0.055). Results: `benchmarks/results/study_a/`.
- `AutoOracle`'s preactivation route passes the threshold and returns a verdict.
- `AutoOracle` and `check()` send every counted evaluation and re-measurement to
  `on_evaluation` (`probe`, `structure`, `boundary`, `search`, `remeasure`, and
  shrink events), numbered consecutively across the run, so a JSONL audit of an
  `AutoOracle` run is complete and replays with `replay()`. `classify_landscape`
  also accepts `on_evaluation`.
- `PreactivationOracle(on_evaluation=...)` logs every model evaluation
  (`search`, `remeasure`) with input-space `x`, so its logs replay with
  `replay()`. Passing `on_evaluation` to `run()` raises `TypeError` (the inner
  search would have logged z-space points). `AutoOracle` uses it instead of
  wrapping the model functions.
- `run_hybrid` could report FAIL with an empirical witness outside
  `input_bounds`; `EmpiricalSearch` accepts `bounds` and clips samples to it,
  and `run_hybrid` passes its bounds. `run_hybrid` now rejects data whose
  column count differs from `input_dim`.
- `CascadeSearch` de-duplicates top-K candidates before the expensive stage
  (`dedupe_tol`, default 1e-3 of the box width), no longer evaluates the cheap
  model twice per point in preactivation mode, and rejects `top_k < 1`.
- `check(..., W=, b=)` crashed; `PreactivationResult` now carries `verdict`,
  `threshold`, `worst_input`, `scheme` and `noise_state`. Added
  `PreactivationOracle.run(threshold=...)`, `CascadeSearch.run(threshold=...)`
  with `CascadeResult.verdict`, and `HybridResult.verdict`.
- Add an evaluation log: `FHEOracle(on_evaluation=...)` receives every counted
  evaluation (search, re-measurement, shrink), with `JsonlEvaluationLog`,
  `read_log` and `replay` for audit trails on backends that are not
  bit-reproducible.
- Add `CoverageCertificate.violating_fraction_upper_bound(confidence)`, a
  Clopper–Pearson upper bound on the violating share of the domain from the
  random-floor sample, with no assumed minimum failure-region size.
- Add optional `batch_fhe_fn` to `FHEOracle` to evaluate each CMA-ES
  generation and the random-floor sample in one call (for example with a
  process pool). Outputs must be returned in input order; the search, budget
  and validation are unchanged. Requires `fhe_fn` and the built-in fitness.
- Fix a false PASS on noisy backends: the verdict came only from one final
  re-measurement, so a violation observed during search could be reported
  as PASS. `run()` now fails when any counted evaluation meets the threshold
  (for error-measuring fitness) and reports `search_max_error` and
  `remeasured_error`.
- Multi-output rank modes now fail on an argmax flip at the witness or during
  search (`class_flip`), even when the absolute error is below threshold.
- `seed=0` is reproducible; pycma treated 0 as a clock-based seed.
- Fix `FHEOracle.shrink` returning a witness that no longer fails on noisy
  backends. The final point is re-measured up to five times the way `run()`
  measures its verdict; a noisy point must clear the threshold by the
  observed spread, otherwise shrink steps back toward the original witness.
  `ShrinkResult.max_error` is the lowest confirming measurement and always
  meets the threshold. `n_evals` includes verification (up to 30 evaluations
  of `max_evals` are reserved for it) and `max_evals` must be positive.
- `FHEOracle.run()` no longer evaluates more than `n_trials`. Budgets smaller
  than one CMA-ES generation, IPOP/BIPOP restarts, heuristic seed injection
  and random-floor runs could overshoot (by up to 40 evaluations). The final
  partial generation is evaluated but not told to CMA-ES. Adaptive mode, which
  may extend its budget by design, is unchanged.
- Adaptive mode no longer reports a budget extension that pycma rejected.
  A failed `maxfevals` update now emits a `RuntimeWarning` and is not counted
  in `adaptive_extensions_used`.
- Justify the remaining broad exception handlers (none are on the verdict or
  `max_error` path); the CLI error message now includes the exception type.
- CI tests Python 3.13.
- `OracleResult.scheme` is `"fhe_fn"` for callable models and
  `"custom-fitness"` without a model, instead of `"plaintext-diff"`, which
  mislabelled real encrypted backends.
- Document that `TenSEALContext` does not seed CKKS encryption randomness.
- Benchmarks README no longer presents the synthetic 4,259x ratio as a
  headline result.

## 0.6.0

### Reliability and migration

- Invalid model evaluations abort precision testing instead of returning zero
  divergence. Built-in comparisons reject missing or mismatched outputs,
  empty outputs, non-finite numbers and unsupported complex outputs.
- `EvaluationError` is public. Callers that previously depended on invalid
  candidates being silently ignored must handle errors explicitly. CLI exit
  code 2 denotes model/configuration/evaluation errors; 0 and 1 remain PASS
  and measured precision FAIL. Backend/property callbacks can also propagate
  their original exceptions.
- Validation is shared by direct, automatic, differential, multi-output,
  empirical, preactivation, subspace and cascade search paths. Final output
  measurement independently validates the model, including custom-fitness runs.
- Custom/plugin fitness scores must be finite. Custom implementations remain
  responsible for propagating any backend errors they encounter.
- Fix bounded one-dimensional CMA-ES initialization, including restart runs.
  Bound transforms remain enabled.

### Features and verification

- Includes the one-call `check()` API and CLI, witness shrinking, fault
  localization, structure diagnostics, differential testing, property fitness,
  CI diagnostics and adapter-agnostic tracing developed since 0.5.2.
- Align package version with the documented feature set. Update release API
  verification to reflect Core's plugin architecture.
- Add native TenSEAL CI coverage and regression tests for invalid evaluations.
- Point project URLs, CI badge and citation metadata at the public
  repository; pin ruff in CI.
- Correct benchmark descriptions: the synthetic reference uses asymmetric
  domains; the matched historical results include both wins and losses.
- Clarify current AGPL licensing, unverified patent status, optional plugin
  requirements and the limits of PASS. This release changes no license terms.

The package remains Alpha. Tests validate supported exercised configurations;
PASS is a bounded empirical result, not a proof or a security certification.
