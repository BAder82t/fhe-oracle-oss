# Study A: AutoOracle boundary-probe confirmation

Script `benchmarks/study_a_boundary_confirmation.py`; pre-registration `benchmarks/preregistration_2026-09-15.md` at `962aab31e4239d074130a8f80394dac3e2731c1d` (SHA-256 `912da47075386a07163dba517dfc06b4dfa9d24461fbb587b30e1c924a694386`).

Holm family: lr_d8, cheb_d10, poly_d6, bump_d8, decoy_d6 (size 5).

| Circuit | seeds | C1 median after/acct (range) | W/L/T | p | Holm p | AFTER lower | C1 | C2 after >= corner - 1e-5 | C2 | C3 calls after / acct (remeasure events) | C3 | circuit |
|---|---:|---|---|---:|---:|---|---|---|---|---|---|---|
| lr_d8 | 20 | 1.7435 (1.0666-3.0359) | 20/0/0 | 1.91e-06 | 9.54e-06 | False | PASS | 20/20 (strict 14) | PASS | [200, 200] / [200, 200] ([2] / [1]) | PASS | PASS |
| cheb_d10 | 20 | 3.1419 (1.9688-27.8503) | 20/0/0 | 1.91e-06 | 9.54e-06 | False | PASS | 16/20 (strict 15) | PASS | [200, 200] / [200, 200] ([2] / [1]) | PASS | PASS |
| poly_d6 | 20 | 1.5156 (1.1812-3.3283) | 20/0/0 | 1.91e-06 | 9.54e-06 | False | PASS | n/a | - | [200, 200] / [200, 200] ([2] / [1]) | PASS | PASS |
| bump_d8 | 20 | 1.0000 (0.9170-1.0000) | 0/3/17 | 0.109 | 0.109 | True | PASS | n/a | - | [199, 200] / [200, 200] ([1, 2] / [1]) | PASS | PASS |
| decoy_d6 | 20 | 1.0000 (0.6655-1.0000) | 0/6/14 | 0.0277 | 0.0554 | True | PASS | n/a | - | [199, 199] / [200, 200] ([1] / [1]) | PASS | PASS |

**Decision (pre-registered rule): KEEP the boundary probe.**

## Median per-seed metric and counted evaluations

| Circuit | supremum | after | accounting | random | corner | calls after | calls acct | random evals | corner evals | median ms/FHE eval | compute s |
|---|---:|---:|---:|---:|---:|---|---|---|---|---:|---:|
| lr_d8 | 140.9203800670085 | 140.923 | 80.8333 | 38.8837 | 140.923 | [200, 200] | [200, 200] | [198, 198] | [198, 198] | 45.29 | 761 |
| cheb_d10 | 4.075794756584004 | 4.07587 | 1.29137 | 0.717952 | 3.90235 | [200, 200] | [200, 200] | [198, 198] | [198, 198] | 64.79 | 1102 |
| poly_d6 | unknown | 0.000242019 | 0.000159772 | 0.000114431 | 0.00024206 | [200, 200] | [200, 200] | [198, 198] | [77, 77] | 107.66 | 1708 |
| bump_d8 | 1.0 | 0.902702 | 0.909934 | 0.250014 | 0.333824 | [199, 200] | [200, 200] | [198, 198] | [198, 198] | 0.00 | 1 |
| decoy_d6 | 1.2028000449666623 | 1.10418 | 1.11309 | 0.526179 | 0.300363 | [199, 199] | [200, 200] | [198, 198] | [77, 77] | 0.00 | 1 |

## Per-seed values

### lr_d8 (status: complete; strategies after ['cma_es', 'robust_cma_es'], acct ['cma_es', 'robust_cma_es']; out-of-bounds 0)

| seed | after [calls, remeasure] | accounting [calls, remeasure] | random [n] | corner [n] |
|---:|---:|---:|---:|---:|
| 11 | 140.923 [200, 2] | 85.5844 [200, 1] | 29.1374 [198] | 137.327 [198] |
| 12 | 140.923 [200, 2] | 73.1287 [200, 1] | 68.171 [198] | 140.923 [198] |
| 13 | 140.923 [200, 2] | 86.1316 [200, 1] | 75.2699 [198] | 140.923 [198] |
| 14 | 140.923 [200, 2] | 69.3725 [200, 1] | 40.1866 [198] | 140.923 [198] |
| 15 | 140.923 [200, 2] | 108.149 [200, 1] | 62.3749 [198] | 140.923 [198] |
| 16 | 140.923 [200, 2] | 76.6929 [200, 1] | 28.472 [198] | 137.327 [198] |
| 17 | 140.923 [200, 2] | 81.4542 [200, 1] | 37.5809 [198] | 140.923 [198] |
| 18 | 140.923 [200, 2] | 69.1242 [200, 1] | 42.3457 [198] | 125.649 [198] |
| 19 | 140.923 [200, 2] | 72.1698 [200, 1] | 36.4182 [198] | 137.327 [198] |
| 20 | 140.923 [200, 2] | 94.2421 [200, 1] | 68.7734 [198] | 137.327 [198] |
| 21 | 140.923 [200, 2] | 80.2125 [200, 1] | 42.8302 [198] | 125.649 [198] |
| 22 | 140.923 [200, 2] | 81.7653 [200, 1] | 30.1764 [198] | 137.327 [198] |
| 23 | 140.923 [200, 2] | 73.3002 [200, 1] | 52.2734 [198] | 137.327 [198] |
| 24 | 140.923 [200, 2] | 132.122 [200, 1] | 34.2265 [198] | 140.923 [198] |
| 25 | 140.923 [200, 2] | 89.3433 [200, 1] | 60.6516 [198] | 140.923 [198] |
| 26 | 140.923 [200, 2] | 88.4844 [200, 1] | 46.4794 [198] | 140.923 [198] |
| 27 | 140.923 [200, 2] | 46.4184 [200, 1] | 34.8305 [198] | 140.923 [198] |
| 28 | 140.923 [200, 2] | 95.8692 [200, 1] | 34.5917 [198] | 140.923 [198] |
| 29 | 140.923 [200, 2] | 65.3426 [200, 1] | 23.3054 [198] | 140.923 [198] |
| 30 | 140.923 [200, 2] | 68.1641 [200, 1] | 28.9189 [198] | 140.923 [198] |

### cheb_d10 (status: complete; strategies after ['cma_es', 'random_only', 'warm_start'], acct ['cma_es', 'random_only', 'warm_start']; out-of-bounds 0)

| seed | after [calls, remeasure] | accounting [calls, remeasure] | random [n] | corner [n] |
|---:|---:|---:|---:|---:|
| 11 | 4.07587 [200, 2] | 1.82593 [200, 1] | 0.700523 [198] | 3.78064 [198] |
| 12 | 4.07587 [200, 2] | 1.9532 [200, 1] | 1.31786 [198] | 4.07587 [198] |
| 13 | 4.07587 [200, 2] | 0.146349 [200, 1] | 0.726956 [198] | 4.05968 [198] |
| 14 | 4.07587 [200, 2] | 1.17833 [200, 1] | 1.32692 [198] | 3.78064 [198] |
| 15 | 4.05695 [200, 2] | 2.06066 [200, 1] | 0.574895 [198] | 4.05968 [198] |
| 16 | 4.05695 [200, 2] | 1.17327 [200, 1] | 0.588179 [198] | 4.07587 [198] |
| 17 | 4.05695 [200, 2] | 1.56317 [200, 1] | 0.758743 [198] | 4.02407 [198] |
| 18 | 4.07587 [200, 2] | 1.88479 [200, 1] | 0.9932 [198] | 4.05695 [198] |
| 19 | 4.05695 [200, 2] | 0.863183 [200, 1] | 0.363716 [198] | 4.07587 [198] |
| 20 | 4.05695 [200, 2] | 0.18106 [200, 1] | 0.735639 [198] | 3.78064 [198] |
| 21 | 4.07587 [200, 2] | 1.03467 [200, 1] | 0.379369 [198] | 2.98919 [198] |
| 22 | 4.05695 [200, 2] | 1.30454 [200, 1] | 0.816492 [198] | 2.98919 [198] |
| 23 | 4.05695 [200, 2] | 1.2782 [200, 1] | 0.55202 [198] | 4.07587 [198] |
| 24 | 4.07587 [200, 2] | 0.164235 [200, 1] | 0.998494 [198] | 3.55787 [198] |
| 25 | 4.07587 [200, 2] | 1.95206 [200, 1] | 0.840595 [198] | 4.02407 [198] |
| 26 | 4.07587 [200, 2] | 1.95854 [200, 1] | 0.38862 [198] | 3.76518 [198] |
| 27 | 4.05695 [200, 2] | 1.51424 [200, 1] | 0.810884 [198] | 3.78064 [198] |
| 28 | 4.07587 [200, 2] | 1.07418 [200, 1] | 0.613446 [198] | 3.19929 [198] |
| 29 | 4.05695 [200, 2] | 1.34758 [200, 1] | 0.557684 [198] | 4.02407 [198] |
| 30 | 4.07587 [200, 2] | 0.531294 [200, 1] | 0.708948 [198] | 3.78064 [198] |

### poly_d6 (status: complete; strategies after ['cma_es'], acct ['cma_es']; out-of-bounds 0)

| seed | after [calls, remeasure] | accounting [calls, remeasure] | random [n] | corner [n] |
|---:|---:|---:|---:|---:|
| 11 | 0.000241952 [200, 2] | 0.00016389 [200, 1] | 0.00013941 [198] | 0.000242086 [77] |
| 12 | 0.000242032 [200, 2] | 0.000180514 [200, 1] | 0.000124827 [198] | 0.000242036 [77] |
| 13 | 0.000241943 [200, 2] | 0.000106383 [200, 1] | 0.000102356 [198] | 0.000242015 [77] |
| 14 | 0.000241965 [200, 2] | 0.000180245 [200, 1] | 0.000130411 [198] | 0.00024207 [77] |
| 15 | 0.000242037 [200, 2] | 0.000113941 [200, 1] | 0.000131031 [198] | 0.000242062 [77] |
| 16 | 0.000242066 [200, 2] | 0.000191876 [200, 1] | 0.000111148 [198] | 0.000242024 [77] |
| 17 | 0.000242028 [200, 2] | 0.000182869 [200, 1] | 0.000145318 [198] | 0.000242063 [77] |
| 18 | 0.000242071 [200, 2] | 0.000144667 [200, 1] | 0.00011936 [198] | 0.000242094 [77] |
| 19 | 0.000241998 [200, 2] | 0.000181679 [200, 1] | 9.49434e-05 [198] | 0.00024206 [77] |
| 20 | 0.000242019 [200, 2] | 0.000192103 [200, 1] | 0.000112705 [198] | 0.000242056 [77] |
| 21 | 0.000242019 [200, 2] | 8.87002e-05 [200, 1] | 7.62698e-05 [198] | 0.000242049 [77] |
| 22 | 0.000242031 [200, 2] | 0.000155654 [200, 1] | 8.64352e-05 [198] | 0.000242009 [77] |
| 23 | 0.000242003 [200, 2] | 0.00015181 [200, 1] | 0.000116157 [198] | 0.000242102 [77] |
| 24 | 0.000242076 [200, 2] | 8.57405e-05 [200, 1] | 9.83849e-05 [198] | 0.000242091 [77] |
| 25 | 0.000242015 [200, 2] | 0.000187888 [200, 1] | 0.000104812 [198] | 0.00024207 [77] |
| 26 | 0.000242089 [200, 2] | 7.2736e-05 [200, 1] | 0.000126468 [198] | 0.000242056 [77] |
| 27 | 0.000242018 [200, 2] | 0.000204885 [200, 1] | 0.000145056 [198] | 0.000242035 [77] |
| 28 | 0.000242058 [200, 2] | 0.00017138 [200, 1] | 0.000107307 [198] | 0.000242045 [77] |
| 29 | 0.000242008 [200, 2] | 0.000113462 [200, 1] | 8.84507e-05 [198] | 0.000242059 [77] |
| 30 | 0.000242 [200, 2] | 0.000135345 [200, 1] | 0.000141056 [198] | 0.000242097 [77] |

### bump_d8 (status: complete; strategies after ['cma_es'], acct ['cma_es']; out-of-bounds 0)

| seed | after [calls, remeasure] | accounting [calls, remeasure] | random [n] | corner [n] |
|---:|---:|---:|---:|---:|
| 11 | 0.921698 [199, 1] | 0.921698 [200, 1] | 0.306115 [198] | 0.333824 [198] |
| 12 | 0.939538 [199, 1] | 0.939538 [200, 1] | 0.344296 [198] | 0.333824 [198] |
| 13 | 0.90259 [199, 1] | 0.914533 [200, 1] | 0.396456 [198] | 0.333824 [198] |
| 14 | 0.9067 [199, 1] | 0.9067 [200, 1] | 0.255946 [198] | 0.223769 [198] |
| 15 | 0.92843 [199, 1] | 0.92843 [200, 1] | 0.34725 [198] | 0.333824 [198] |
| 16 | 0.906422 [199, 1] | 0.906422 [200, 1] | 0.18327 [198] | 0.333824 [198] |
| 17 | 0.950798 [199, 1] | 0.950798 [200, 1] | 0.213055 [198] | 0.223769 [198] |
| 18 | 0.797393 [199, 1] | 0.797393 [200, 1] | 0.112144 [198] | 0.333824 [198] |
| 19 | 0.94041 [199, 1] | 0.94041 [200, 1] | 0.246091 [198] | 0.333824 [198] |
| 20 | 0.897159 [199, 1] | 0.897159 [200, 1] | 0.119806 [198] | 0.333824 [198] |
| 21 | 0.902813 [199, 1] | 0.902813 [200, 1] | 0.213542 [198] | 0.333824 [198] |
| 22 | 0.869121 [199, 1] | 0.947767 [200, 1] | 0.248395 [198] | 0.333824 [198] |
| 23 | 0.901229 [199, 1] | 0.901229 [200, 1] | 0.195967 [198] | 0.333824 [198] |
| 24 | 0.870229 [199, 1] | 0.870229 [200, 1] | 0.276669 [198] | 0.223769 [198] |
| 25 | 0.901165 [199, 1] | 0.901165 [200, 1] | 0.470776 [198] | 0.333824 [198] |
| 26 | 0.951349 [200, 2] | 0.951349 [200, 1] | 0.251633 [198] | 0.333824 [198] |
| 27 | 0.873277 [199, 1] | 0.873277 [200, 1] | 0.221833 [198] | 0.223769 [198] |
| 28 | 0.95007 [199, 1] | 0.95007 [200, 1] | 0.276175 [198] | 0.333824 [198] |
| 29 | 0.902419 [199, 1] | 0.913169 [200, 1] | 0.305628 [198] | 0.333824 [198] |
| 30 | 0.804274 [199, 1] | 0.804274 [200, 1] | 0.235124 [198] | 0.141666 [198] |

### decoy_d6 (status: complete; strategies after ['cma_es', 'warm_start'], acct ['cma_es', 'warm_start']; out-of-bounds 0)

| seed | after [calls, remeasure] | accounting [calls, remeasure] | random [n] | corner [n] |
|---:|---:|---:|---:|---:|
| 11 | 1.03271 [199, 1] | 1.05097 [200, 1] | 0.371292 [198] | 0.300363 [77] |
| 12 | 0.487642 [199, 1] | 0.732718 [200, 1] | 0.63919 [198] | 0.300363 [77] |
| 13 | 1.00749 [199, 1] | 1.18487 [200, 1] | 0.551933 [198] | 0.300363 [77] |
| 14 | 1.12859 [199, 1] | 1.12859 [200, 1] | 0.490361 [198] | 0.300363 [77] |
| 15 | 0.834813 [199, 1] | 0.834813 [200, 1] | 0.94974 [198] | 0.300363 [77] |
| 16 | 1.07918 [199, 1] | 1.07918 [200, 1] | 0.633461 [198] | 0.300363 [77] |
| 17 | 1.07617 [199, 1] | 1.07617 [200, 1] | 0.693233 [198] | 0.300363 [77] |
| 18 | 1.1277 [199, 1] | 1.1277 [200, 1] | 0.601041 [198] | 0.300363 [77] |
| 19 | 1.0529 [199, 1] | 1.0529 [200, 1] | 0.523003 [198] | 0.300363 [77] |
| 20 | 1.15056 [199, 1] | 1.15056 [200, 1] | 0.529355 [198] | 0.300363 [77] |
| 21 | 1.13596 [199, 1] | 1.17438 [200, 1] | 0.716368 [198] | 0.300363 [77] |
| 22 | 1.08995 [199, 1] | 1.08995 [200, 1] | 0.367295 [198] | 0.300363 [77] |
| 23 | 1.11846 [199, 1] | 1.11846 [200, 1] | 0.50112 [198] | 0.300363 [77] |
| 24 | 1.10063 [199, 1] | 1.10063 [200, 1] | 0.601005 [198] | 0.300363 [77] |
| 25 | 1.10772 [199, 1] | 1.10772 [200, 1] | 0.894594 [198] | 0.300363 [77] |
| 26 | 1.11992 [199, 1] | 1.1305 [200, 1] | 0.477401 [198] | 0.300363 [77] |
| 27 | 1.12533 [199, 1] | 1.15879 [200, 1] | 0.490549 [198] | 0.300363 [77] |
| 28 | 1.15456 [199, 1] | 1.15456 [200, 1] | 0.376396 [198] | 0.300363 [77] |
| 29 | 1.14512 [199, 1] | 1.14512 [200, 1] | 0.501384 [198] | 0.300363 [77] |
| 30 | 1.05251 [199, 1] | 1.05251 [200, 1] | 0.463891 [198] | 0.300363 [77] |

## Provenance

| Circuit | status | start HEAD | end HEAD | core.py | autoconfig.py | script | pre-registration ok start / end | equivalence reference |
|---|---|---|---|---|---|---|---|---|
| lr_d8 | complete | `56b6d4149765` | `56b6d4149765` | `24979925df7d` | `853c5b1fc158` | `18b64e61e2c9` | True / True | 006f9ca2ec7b |
| cheb_d10 | complete | `56b6d4149765` | `56b6d4149765` | `24979925df7d` | `853c5b1fc158` | `18b64e61e2c9` | True / True | 006f9ca2ec7b |
| poly_d6 | complete | `56b6d4149765` | `56b6d4149765` | `24979925df7d` | `853c5b1fc158` | `18b64e61e2c9` | True / True | 006f9ca2ec7b |
| bump_d8 | complete | `56b6d4149765` | `56b6d4149765` | `24979925df7d` | `853c5b1fc158` | `18b64e61e2c9` | True / True | 006f9ca2ec7b |
| decoy_d6 | complete | `56b6d4149765` | `56b6d4149765` | `24979925df7d` | `853c5b1fc158` | `18b64e61e2c9` | True / True | 006f9ca2ec7b |

## Choices where the pre-registration is not explicit

- Counted evaluations = the evaluations counted in result.n_trials (event kinds probe, structure,
  boundary, search). The documented re-measurement is kind "remeasure": 1 for ACCOUNTING-ONLY; 1 or 2
  for AFTER (2 when a pre-search witness replaces the search witness).
- Metric = max |plaintext(x) - fhe(x)| over counted evaluations, recomputed from each logged call. The
  max including re-measurements and AutoOracle's reported max_error are also recorded, not tested.
- Criterion 1: per-seed ratio AFTER / ACCOUNTING-ONLY, median over the 20 seeds, pass if >= 0.95. Paired
  Wilcoxon signed-rank (scipy, two-sided, zero_method="wilcox"; p = 1 when every pair ties). Holm
  correction over the five criterion-1 tests (the study's family). "AFTER lower" = the rank sum of
  negative differences exceeds that of positive ones. Fails if Holm p < 0.05 and AFTER lower.
- Criterion 2 (lr_d8, cheb_d10 only): AFTER metric >= corner metric - 1e-5 on >= 16 of 20 seeds.
- Criterion 3, every seed and both arms: FHE calls == result.n_trials + remeasure events, calls <=
  n_trials + remeasure events, and remeasure events within the documented count above. Whether calls
  <= n_trials (the library's stricter invariant) is recorded too.
- Criteria, Holm correction and the decision are evaluated only for the full pre-registered set: all
  five circuits, seeds 11-30, n_trials 200, threshold 0.01, every circuit file complete, start and end
  provenance equal within each file and across files, and the pre-registration check passing at start
  and end. Holm is then over the five criterion-1 tests. Otherwise every pass field, Holm p and the
  decision are null, and summary.json records study false and Holm family size 0.
- Arm order within a seed: after, accounting, random, corner. One TenSEAL context and key set per
  circuit (per process), shared by every arm; CKKS encryption noise is not seeded.
- Mock definitions (bump_d8, decoy_d6) are identical to benchmarks/autooracle_boundary_eval.py.
- Provenance at start, on resume, before every seed and at end: git HEAD and the SHA-256 of
  fhe_oracle/core.py, fhe_oracle/autoconfig.py and this script must equal their start values, and the
  pre-registration file and its blob at 962aab3 must both hash to the fixed SHA-256 above. Any failure
  aborts; the reason is recorded in the circuit file and results so far are kept. At start
  fhe_oracle/ must have no uncommitted changes, HEAD must contain 3705b7f, and the package must be
  imported from this checkout.
- Resume: a circuit file is extended only when its config, recorded start provenance and archived
  equivalence inputs equal the current ones and it was not aborted.
- ``--study`` fixes seeds 11-30, n_trials 200 and threshold 0.01 and writes to
  benchmarks/results/study_a/ by default (circuits may be split across processes). Any other
  invocation is labelled SMOKE and no criterion or decision is evaluated.
