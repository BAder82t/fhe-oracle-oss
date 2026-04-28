# Benchmark Results

Measured data from FHE Oracle experiments. Each CSV is paired with a
description and a plain-English summary of what the numbers mean.

## S0 — Heuristic Lesion Audit (2026-04-18)

**Gate decision:** `research/experiment-plan/results/S0-gate-decision.md`
**Experiment spec:** `research/experiment-plan/S0-heuristic-lesion-audit.md`
**Dossier:** `research/future-work/13-heuristic-marginal-value.md`

### What the experiment tested

The paper claims (`fhe_oracle_draft.tex:161-165, 522-556, 567-569`) that
three fitness-shaping weights (`w_div`, `w_noise`, `w_depth`) and three
noise-guided seed heuristics (`s_mm` multiplication magnifier, `s_ds`
depth seeker, `s_nt` near-threshold explorer) are load-bearing for the
1.80× / 2.28× headline oracle-vs-random ratios. Until this experiment
ran, neither the seeds nor any per-heuristic lesion was implemented —
[fhe_oracle/core.py:196-213](../../fhe_oracle/core.py#L196-L213) called
bare `ask/tell`, never `es.inject(...)`.

S0 closes both gaps:

1. **Code gap** — `fhe_oracle/heuristics.py` implements the three seed
   scorers + `generate_seeds()`; `fhe_oracle/core.py` gains 4 new kwargs
   (`use_heuristic_seeds`, `heuristic_which`, `heuristic_k`,
   `heuristic_tau`) and invokes `es.inject(seeds, force=True)` before
   the first generation.
2. **Measurement gap** — `benchmarks/ablation_heuristics.py` runs the
   9-config × N-circuit × 10-seed lesion matrix. 9 configs: FULL, DIV,
   -N, -D, -ND, -S, -MM, -DS, -NT (each removes one component).

### Files

- **`ablation_heuristics.csv`** — raw per-cell data. 270 rows =
  9 configs × 3 circuits × 10 seeds. Columns:
  `config, circuit, seed, weights, seeds_used, max_error,
  worst_input, wall_clock_s, n_trials, verdict`.
- **`ablation_summary.csv`** — aggregated per (config, circuit). 24
  rows = 8 lesion configs × 3 circuits (DIV is a lesion leg as well).
  Columns: `config, circuit, n_seeds, full_median_err,
  lesion_median_err, median_ratio_full_over_lesion,
  median_ratio_lesion_over_div, wins_full_over_lesion_of_10,
  wins_lesion_over_div_of_10, p_uncorrected, p_holm, verdict`.

### Reproduction

```bash
python benchmarks/ablation_heuristics.py 500          # 11.6s on M1
python benchmarks/analysis/ablation_summary.py        # <1s
```

Seeds fixed `[0..9]`, budget B=500 per cell. pycma ≥ 3.0 required for
`force=True` on `es.inject()`.

### Results in one sentence

**0 of 24 (config × circuit) comparisons clear the LOAD_BEARING bar
(median ratio ≥ 1.10×, wins ≥ 7/10, `p_holm < 0.05`)** — on the mock
circuits evaluated, the three shaping weights and three seed heuristics
are not empirically supported as load-bearing beyond pure divergence
fitness.

### Verdict breakdown

| Config | LOAD_BEARING | INERT | INCONCLUSIVE |
|---|---:|---:|---:|
| DIV (pure divergence)        | 0 | 1 | 2 |
| -N  (no noise term)          | 0 | 0 | 3 |
| -D  (no depth term)          | 0 | 3 | 0 |
| -ND (neither noise nor depth)| 0 | 1 | 2 |
| -S  (no seed injection)      | 0 | 2 | 1 |
| -MM (no multiplier seed)     | 0 | 2 | 1 |
| -DS (no depth-seeker seed)   | 0 | 2 | 1 |
| -NT (no near-threshold seed) | 0 | 0 | 3 |

Verdict rule (per dossier §4.4):
- **LOAD_BEARING** = FULL/lesion ratio ≥ 1.10× AND wins ≥ 7/10 AND p_holm < 0.05
- **INERT** = ratio ∈ [0.95, 1.05] AND wins ∈ [3, 7]
- **INCONCLUSIVE** = anything else (needs 20-seed replication)

### Per-circuit description

**Circuit 1 — Logistic regression (d=8, bounds [-3, 3]^8).** A
hand-fit LR classifier with a hot-zone mock where `z_proxy > 4` and
`|plain - 0.5| < 0.25` amplifies Gaussian noise by up to 50×. FULL
median max_error = 6.12e-1. Pure-divergence DIV gives 5.17e-1 (ratio
R=1.14× in FULL's favour, 6/10 wins) — close but not significant.
Noise-lesion `-N` scores R=1.29× with 7/10 wins but `p_holm=1.0` — fails
LOAD_BEARING on the statistical criterion. All other lesions are
INCONCLUSIVE. Best single hit: seed=4 LR find of max_error ~1.7
(a real divergence in the hot zone, same class of bug the paper
describes at `tex:1053-1070`).

**Circuit 2 — Depth-4 polynomial (d=6, bounds [-2, 2]^6).** The paper's
`p(x) = Σ c_i x_i² x_{i+1}` with noise amplification at
`max|x_i² x_{i+1}| > 1`. Every single lesion produces a ratio in
[0.95, 1.05] vs FULL. Pure-divergence DIV is marginally stronger
(6/10 wins over FULL). **The shaping terms add zero measurable signal
on this circuit** — this is the cleanest null result of the three
circuits.

**Circuit 3 — Dense + Chebyshev sigmoid (d=10, bounds [-3, 3]^10).**
Hidden width 4, Chebyshev-3 sigmoid approximation. All ratios at
1.000× ± 0.02. Wins/10 split at 0-7. The paper already notes
(`tex:1142-1168`) that CMA-ES loses to random on Chebyshev plateaus
0/10; FULL-vs-lesion shows the same plateau geometry — every config
converges to roughly the same max_error (~0.10) because the landscape
is flat until the box corner, and none of the shaping or seeding
targets the corner. This is the item-04 / item-05 territory
(hybrid-random warm-start + IPOP/BIPOP restarts), not what shaping
can fix.

### What this means for the paper

Per dossier §4.4 this is the **outcome-3 scenario ("shaping inert")**:

> "shaping inert: materially weakens the method; recommend demoting
> the three-heuristic contribution claim (`tex:161-165`) or relegating
> it to appendix."

Paper edits now required:

1. **`tex:161-165`** (three-heuristic contribution claim) — narrow to
   "divergence fitness plus K=10 noise-guided seeding, with shaping
   terms unablated on real CKKS".
2. **`tex:553-556`** (fitness weights disclaimer) — add forward
   pointer to this lesion audit and its null mock outcome.
3. **`tex:567-569`** (K=10 seed injection claim) — now backed by code
   (`fhe_oracle/core.py:198`) but also backed by measured null on
   mocks. Flag as "seeding effect pending real-CKKS measurement".
4. **New §Appendix A.1** — the 24-cell lesion table from
   `ablation_summary.csv` as the primary appendix figure.

### Honest caveats (what this result does NOT prove)

1. **Mock noise proxies are collinear with coordinate norms.** The
   harness' `noise_term = ||x||/√d·3` and `depth_term = max|x|/3` are
   monotone in the same quantities as `mult_magnifier` and
   `depth_seeker` scorers — so lesioning `w_noise` or `w_depth`
   removes signal that on mocks has no independent information.
   **Real conclusions about `w_noise` vs `w_depth` independence
   require the TenSEAL LR row and the item-B1 real-CKKS port**, both
   pending (no TenSEAL adapter in repo as of 2026-04-18).
2. **Chebyshev circuit's flat ratios are expected.** The paper's own
   analysis (`tex:1140-1155`) says CMA-ES can't move on this
   landscape; lesioning shaping terms can't fix a plateau trap that
   item 04 (hybrid-random warm-start) is supposed to address.
3. **n=10 may be underpowered.** At `p_holm = 1.0` across the board,
   either the heuristics are truly inert on mocks OR 10 seeds is
   insufficient to detect a real effect. The dossier's 20-seed
   replication clause applies; that's a cheap follow-up (~30s
   additional wall-clock).
4. **WDBC and TenSEAL LR rows deferred.** Both are named in the S0
   protocol (dossier §4.1, experiment §4 point 4) but neither has an
   adapter in the current repo — TenSEAL adapter is missing entirely
   per dossier 02 §2.3; WDBC reproducibility script is missing per
   S0 Risk 3. A full gate decision on real CKKS requires them.

### What this does prove

- **The code gap is closed.** `es.inject()` now runs. The paper's
  §4.5 claim of K=10 gen-0 seeding is now a fact about the open-source
  implementation, not just the paper.
- **The measurement protocol works.** 9×3×10 in 12 seconds with paired
  Wilcoxon + Holm correction is a reproducible, drop-in appendix table.
- **Pure divergence fitness is strong on mocks.** DIV vs FULL is a
  near-tie on all three circuits, with DIV marginally winning on
  Circuit 2. This is consistent with the paper's own honest disclosure
  at `fhe_oracle_draft.md:793-797` that "the main search in the
  benchmarks above relies on pure divergence fitness".

### Downstream implications

- **Paper A spine (A4+A1+A2+A3): unblocked.** These compose on top of
  `DivergenceFitness` and do not depend on shaping being load-bearing.
  Proceed.
- **Paper B B1 real-CKKS port: now the highest-value next step.** It
  is the first circuit where `-N` / `-D` lesions have independent
  signal. If the TenSEAL row shows LOAD_BEARING on real CKKS, the
  paper's shaping claims are recovered; if not, the outcome-3 demote
  becomes permanent.
- **Backlog C5 (adaptive weights) is now more interesting, not less.**
  If static weights are inert, adaptive weights have no static baseline
  to beat — the null strengthens the measurement story.
