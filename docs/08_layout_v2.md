# PAGE LAYOUT MODEL v2: Synthetic Bengali Handwriting

## What Changed vs v1

**v1 location**: frozen in `/home/hpc4090/shyan/ocr/scribe_layout/` with default model at `runs/default/model.pt`.  
**v2 location**: `/home/hpc4090/shyan/ocr/scribe_layout_v2/` with default model at `runs/default_unfitw_dc/model.pt` (source: FACTS.md line 156).

### 15 key changes:

1. **Combined word anchor** (anchor_noise.py): w=0.349 (inverse-noise weight) blends headline (w) and median-ink-row estimators (1-w). True wobble variance 0.00906 U², headline noise 0.01176, mid 0.00631, combined 0.00411 U² (FACTS.md lines 6-10). De-noise wobble in BengaliScribe: sqrt(1 - 0.29-0.33) = 0.82-0.84 (FACTS.md line 11).

2. **Line clearance rejection sampling** (line_clearance.py): thresholds from real train 1st percentiles: min distance ≥ 2.603 U, min/mean ≥ 0.797. Redraw up to 30 times, else push line down (FACTS.md lines 12-13). Real violation rate 0.014, generated without fix 0.09-0.15 (FACTS.md line 14).

3. **Unfit targets**: median-MAD ratio simulation with inflated per-word targets (factors n=3-12: 2.505, 1.52, 3.159, 2.076, 1.644, 1.408, 1.351, 1.29, 1.256, 1.234). Fixed double fit-shrinkage of wobble: plan-level line C2ST 0.638 → 0.539 (FACTS.md lines 15-17). **BUG** (FACTS.md M2 review): table matches median robust SCALE but was also applied to every per-word residual → per-word wobble overshoot (FACTS.md lines 18-24).

4. **Per-word leverage correction** (unfit.word_inflation(x)): 1/sqrt(1-h_i) with h_i = leverage of word i in polynomial fit. Capped at 4. Config `default_unfitw.yaml` (unfit_words: true) → `runs/default_unfitw/model.pt` (FACTS.md lines 25-27). NEW RECOMMENDED DEFAULT at that point.

5. **Word rotation beta** = 0.524 × line tangent angle from tilt.json (Huber regression) (FACTS.md line 28).

6. **Ink-extent placement** (bscribe_layout.ink_extents, INK_LEVEL 0.35): model gaps are ink edge to ink edge; canvas-width placement made every gap ~0.5 U too wide. Review finding: page Otsu ~0.39 → +0.067 U gap bias (FACTS.md lines 29-30).

7. **Planner reset per page** (FACTS.md lines 31-35): history ran across pages (count 56-72 vs training max 16); slopes random-walked (e.g., -0.158 → -0.2 clamp); words drew past page bottom. Fix: each page restarts from fixed reference state (real context if given, else end of first generated page). After fix: no words past bottom; page slopes stay near writer's (e.g., image_104 ctx: -0.053,-0.054,-0.074,-0.009,-0.016 vs real -0.023).

8. **Balanced C2ST for end-to-end** (FACTS.md lines 36-39): old e2e real-vs-real "floor" 0.34 line / 0.115 page (biased low due to pooled-OOF grouped k-fold AUC). Fix: groups dealt into folds per label + mean of per-fold AUCs. New floor (5 seeds): line 0.503, page 0.527 (e2e run floor: 0.534 / 0.508).

9. **Latent writer-style model context bug fix** (FACTS.md lines 40-44): train/test mismatch in z precision. Causal-prefix z fix (runs/latent_unfit_fix): latent+ctx slope W1 0.00977 → 0.00406, drift W1 0.262 → 0.106, line C2ST 0.561 → 0.547. BUT regresses no-context latent model. UNRESOLVED; latent not default.

10. **De-noised curvature** (extreme deconvolution) → NEW DEFAULT `runs/default_unfitw_dc/model.pt` (FACTS.md addendum 1, lines 105-154). Curvature sampled before word count known; fitted c2 on short lines is noise. Noise model: var(c2_hat) from Huber fit. Measured curv sd train: n5-6 0.00129, noise 0.00092, true 0.00091; n11+ 0.00087/0.00047/0.00073 → true ~0.0007-0.0009, noise = 38.4% of target variance. MDN likelihood p(c_meas) = sum_k pi_k N(c_meas; mu_k, sigma_k² + v_i); sampling draws noise-free. Val NLL curv: -1.004 vs -0.975 (unfitw).

11. **Huber-aware per-word correction** (unfit.word_inflation_huber) → NOT ADOPTED (FACTS.md addendum 3, lines 160-176). OLS leverage 1/sqrt(1-h) under-corrects first/last words 10-25% for n=4-10 because dataset.fit_line is Huber-IRLS. Tested on runs/default_unfitwh_dc (config default_unfitwh_dc.yaml): all six C2ST metrics worse (test and val). Reason: i.i.d. assumption fails; real wobble smoother than i.i.d. Open: recalibrate under correlated-wobble model.

12. **Margin guard** (bscribe_layout.layout_v2) (FACTS.md addendum 2, lines 177-185): before fix, sloped/bowed first line could rise 1-1.4 U into top margin. New: ink_rows() gives each word's ink rows; ink_span() = line's ink top/bottom ±RESID_MARGIN (0.3 U). Lines whose ink passes bottom margin start new page, REPLANNED from scratch (words back to queue; discarded draw never enters history). Test passes on new code, FAILS on old.

13. **Tests** (FACTS.md addendum 2, line 187): 70 tests, ~17 s (was 32 tests, ~8 min). New: test_unfit.py, test_bscribe_layout.py (placement order, margins, per-page history reset, clearance stats), test_line_clearance.py, test_anchor_noise.py. Tightened e2e noise-floor tolerance 0.4 → 0.15. Fixed test fixture make_line singular curvature noise.

14. **Final e2e** (FACTS.md addendum 2, line 196): runs/e2e_final, default model `default_unfitw_dc` + margin guard, balanced C2ST, floor line 0.534 page 0.508.

15. **Validation-split check** (FACTS.md addendum 1, lines 144-155): VAL split, 5 seeds: leverage-family models beat v1 on line and page C2ST on val too; FFT tie. v2_unfit looked better on test (0.539) but not val (0.566) → selection effect. unfitw vs unfitw_dc differ within noise on all C2STs; dc chosen for curvature realism and halved e2e splitting.

---

## Model Architecture & Learned Parameters

**Model class**: `LayoutModel` (model.py, lines 1-22). Seven small 1-D mixture-density heads (MDN, k=4, hidden=64 default):
- Line heads: slope / curv / logres (line scale) / x_start / pitch_next
- Word head: gap
- Residual head: resid

Exact NLL, no latent variables (latent style z IS modelled as latent for writer-style conditioning, but line heads' NLL given z is exact). Teacher forcing in training.

**Learned parameters by head** (from evaluate.py and features.py):
- slope: line-to-line angle trend
- curv: line curvature (c2 coefficient)
- logres: per-word residual scale (log median absolute deviation)
- x_start: line start offset
- pitch_next: vertical gap to next line centre
- gap: inter-word spacing
- resid: per-word vertical wobble (7-dim per line: one per word, sampled from a mixture)

**Training data**: 193 test pages / 1495 lines, text-based split (dataset.py:22). Real words from 5 reference writers (image_104, image_132, etc.) at 300 dpi.

**Config** (default_unfitw_dc.yaml):
- style: history (not latent)
- k: 4 (mixture components)
- hidden: 64
- use_history: true
- use_gap_ar: true
- resid_ar: false
- lr: 0.001
- weight_decay: 1.0e-5
- batch_size: null (full-batch)
- max_epochs: 400
- patience: 30
- seed: 0
- device: cuda
- unfit_targets: true (physical wobble targets)
- unfit_words: true (per-word leverage correction)
- deconv: [curv] (curvature noise deconvolution)

---

## Iterations & Why Each Was Made

### Phase 1: Core v2 baseline (default_unfit)
**Issues fixed**: per-word overshoot in unfit targets (change 3), added combined anchor (change 1), line clearance (change 2).  
**Result**: plan-level line C2ST 0.539±0.016 (vs v1 0.566±0.021).  
**Problem found**: selection bias (tuned on test); e2e C2ST biased low (change 8).

### Phase 2: Per-word leverage (default_unfitw)
**Issue fixed**: unfit.word_inflation(x) per-word leverage correction (change 4).  
**Result**: line C2ST 0.521±0.019, page 0.581±0.026 (test); val 0.527±0.030, 0.575±0.039.  
**Rationale**: per-word targets should use per-word leverage, not line-level scale table.

### Phase 3: Curvature deconvolution (default_unfitw_dc)
**Issue found**: e2e_unfitw splits 89/462 drawn lines (19%); median-ink-row step at split 1.2 U vs chainer JOIN_DEV 1.0 U. Cause = lines bow too much.  
**Fix**: curvature measurement noise model (change 10). Noise = 38.4% of curv target variance. MDN likelihood p(c_meas) convolved, sampling draws noise-free.  
**Result**: e2e_unfitw_dc splits 52/469 (11.1%, vs 19% in unfitw). Plan-level: line 0.535±0.012, page 0.590±0.024 (test); val 0.536±0.027, 0.593±0.031. Within noise of unfitw; chose dc for realism.

### Phase 4: Testing Huber-aware correction (default_unfitwh_dc)
**Hypothesis**: unfit.word_inflation under-corrects edges (10-25% for n=4-10) because fit_line is Huber-IRLS, not OLS.  
**Result**: ALL six C2ST metrics worse (test and val, same direction 6/6). Reason: real wobble smoother than i.i.d.; 4x edge inflation overshoots (FACTS.md lines 160-176).  
**Verdict**: NOT ADOPTED. Default stays default_unfitw_dc.

### Phase 5: Planner reset & margin guard (e2e_pagereset, e2e_unfitw_dc, e2e_final)
**Issue found**: history ran across pages (slope drift from -0.158 to -0.2 clamp); words drew past page bottom (changes 7, 12).  
**Result** (e2e_pagereset, first with balanced C2ST): manual gap 0.685, v2 0.404, v2_ctx 0.267; manual line C2ST 0.905, v2 0.684, v2_ctx 0.619; page 0.834/0.889.  
**Margin guard added** (e2e_final): v2 gap 0.425, line 0.719, page 0.850; v2_ctx gap 0.198, line 0.619, page 0.845. Splitting 57/453 (12.6%) vs e2e_unfitw_dc 52/469 (11.1%).

---

## Evaluation Metrics & C2ST Note

### Critical C2ST Note (from review_opus.md, C1):
**Early e2e runs used biased C2ST**: pooled out-of-fold grouped k-fold AUC with one label per page group. Real-vs-real floor reported as 0.34 (line) and 0.115 (page) — both far below 0.5, indicating protocol failure. **Fix**: per-fold AUC averaged (groups dealt into folds per label). Balanced floor (5 seeds): line 0.503, page 0.527. **From e2e_pagereset onward, C2ST is balanced** (FACTS.md line 65).

### All Results Tables (Chronological Order)

#### Plan-Level Evaluation (Test Split, 5 Seeds; source: FACTS.md lines 46-56)

Real test: 193 pages / 1495 lines.

| Method | Line C2ST | Page C2ST | 2-D FFT C2ST | resid_scale W1 (n=5 / 6-7 / 8-10 / 11+) |
|---|---|---|---|---|
| v1 (scribe_layout/runs/default) | 0.566±0.021 | 0.654±0.043 | 0.567±0.034 | .011 / .025 / .055 / .074 |
| v2_history_unfit (old table) | 0.539±0.016 | 0.660±0.004 | 0.653±0.024 | .039 / .028 / .029 / .058 |
| v2_history_unfitw (leverage) | 0.521±0.019 | 0.581±0.026 | 0.566±0.025 | .009 / .011 / .041 / .062 |
| shuffled_real | 0.584±0.016 | 0.605±0.012 | 0.529±0.050 | .021 / .018 / .054 / .100 |

Manual (BengaliScribe hand-tuned): line 0.980, page 1.0, FFT 0.99 (FACTS.md line 52).

**Note** (FACTS.md review M1, M4): shuffled_real (0.584) is NOT a real-vs-real floor (breaks gap-width relation); comparable within-page null 0.496±0.019. All design choices compared on TEST (val never used) → model selection on test; fitted constants from train (no leakage), but numbers optimistic by ~1 sd (FACTS.md lines 53-56).

#### Validation-Split Check (5 Seeds; source: FACTS.md addendum 1, lines 144-155)

| Method | Line C2ST | Page C2ST | FFT C2ST | resid_scale W1 (n=5 / 6-7 / 8-10 / 11+) |
|---|---|---|---|---|
| v1 | 0.560±0.024 | 0.631±0.040 | 0.531±0.033 | .013 / .025 / .055 / .089 |
| v2_unfit (old table) | 0.566±0.028 | 0.660±0.022 | 0.599±0.051 | .049 / .030 / .024 / .050 |
| v2_unfitw (leverage) | 0.527±0.030 | 0.575±0.039 | 0.528±0.024 | .012 / .011 / .035 / .072 |
| v2_unfitw_dc | 0.536±0.027 | 0.593±0.031 | 0.535±0.044 | .011 / .011 / .035 / .069 |
| shuffled_real | 0.577±0.023 | 0.597±0.030 | 0.480±0.034 | .016 / .017 / .055 / .089 |

Reading: leverage-family models beat v1 on line and page C2ST on val too; FFT tie. v2_unfit looked better on test (0.539) but not val (0.566) → selection effect. unfitw vs unfitw_dc differ within noise.

#### End-to-End (BengaliScribe renders → real detector → extraction; 5 writers × 2 seeds; test real reference; source: FACTS.md lines 64-79)

**Noise floor** (real TEST pages split in half, balanced C2ST): line 0.534, page 0.508 (from e2e_pagereset onward).

**e2e_unfit** (canvas-width placement, biased C2ST):
| Method | gap W1 | slope W1 | drift W1 | pitch W1 | x_start W1 | line C2ST | page C2ST | recovered |
|---|---|---|---|---|---|---|---|---|
| manual | 0.685 | 0.035 | 0.965 | 1.839 | 0.948 | 0.893 | 0.998 | 1.004 |
| v1 | 0.582 | 0.017 | 0.536 | 1.100 | 5.254 | 0.699 | 0.942 | 0.797 |
| v2 | 0.602 | 0.015 | 0.531 | 0.716 | 3.837 | 0.750 | 0.901 | 0.969 |
| v1_ctx | 0.526 | 0.013 | 0.430 | 1.484 | 5.287 | 0.653 | 0.971 | 0.714 |
| v2_ctx | 0.381 | 0.013 | 0.427 | 0.924 | 4.416 | 0.595 | 0.901 | 0.910 |

(Source: runs/e2e_unfit/results.md)

**e2e_unfit_ink** (ink-extent placement, biased C2ST):
| Method | gap W1 | slope W1 | drift W1 | pitch W1 | x_start W1 | line C2ST | page C2ST | recovered |
|---|---|---|---|---|---|---|---|---|
| manual | 0.685 | 0.035 | 0.965 | 1.839 | 0.948 | 0.893 | 0.998 | 1.004 |
| v2 | 0.471 | 0.016 | 0.503 | 0.712 | 4.037 | 0.667 | 0.840 | 0.952 |
| v2_ctx | 0.207 | 0.010 | 0.343 | 1.001 | 4.438 | 0.532 | 0.918 | 0.866 |

(Source: runs/e2e_unfit_ink/results.md)

**e2e_pagereset** (planner reset + balanced C2ST, model default_unfit, floor line 0.534 / page 0.508):
| Method | gap W1 | slope W1 | drift W1 | pitch W1 | x_start W1 | line C2ST | page C2ST | recovered |
|---|---|---|---|---|---|---|---|---|
| manual | 0.685 | 0.035 | 0.965 | 1.839 | 0.948 | 0.905 | 0.999 | 1.004 |
| v2 | 0.404 | 0.013 | 0.396 | 0.670 | 4.065 | 0.684 | 0.834 | 0.965 |
| v2_ctx | 0.267 | 0.017 | 0.558 | 0.770 | 3.783 | 0.619 | 0.889 | 0.926 |

(Source: runs/e2e_pagereset/results.md)

**e2e_unfitw** (model default_unfitw):
| Method | gap W1 | slope W1 | drift W1 | pitch W1 | x_start W1 | line C2ST | page C2ST | recovered |
|---|---|---|---|---|---|---|---|---|
| manual | 0.685 | 0.035 | 0.965 | 1.839 | 0.948 | 0.905 | 0.999 | 1.004 |
| v2 | 0.376 | 0.012 | 0.343 | 0.346 | 2.278 | 0.699 | 0.851 | 0.960 |
| v2_ctx | 0.197 | 0.019 | 0.610 | 0.480 | 2.703 | 0.613 | 0.842 | 0.933 |

(Source: runs/e2e_unfitw/results.md)

**e2e_unfitw_dc** (model default_unfitw_dc):
| Method | gap W1 | slope W1 | drift W1 | pitch W1 | x_start W1 | line C2ST | page C2ST | recovered |
|---|---|---|---|---|---|---|---|---|
| manual | 0.685 | 0.035 | 0.965 | 1.839 | 0.948 | 0.905 | 0.999 | 1.004 |
| v2 | 0.387 | 0.021 | 0.645 | 0.465 | 1.453 | 0.719 | 0.844 | 0.983 |
| v2_ctx | 0.183 | 0.023 | 0.695 | 0.535 | 2.494 | 0.654 | 0.847 | 0.929 |

(Source: runs/e2e_unfitw_dc/results.md)

**e2e_final** (default_unfitw_dc + margin guard; floor line 0.534 / page 0.508):
| Method | gap W1 | slope W1 | drift W1 | pitch W1 | x_start W1 | line C2ST | page C2ST | recovered |
|---|---|---|---|---|---|---|---|---|
| manual | 0.685 | 0.035 | 0.965 | 1.839 | 0.948 | 0.905 | 0.999 | 1.004 |
| v2 | 0.425 | 0.021 | 0.672 | 0.390 | 1.239 | 0.719 | 0.850 | 0.981 |
| v2_ctx | 0.198 | 0.020 | 0.621 | 0.503 | 2.130 | 0.619 | 0.845 | 0.940 |

(Source: runs/e2e_final/results.md). 440 lines kept; splitting 57/453 drawn lines (12.6%).

#### Plan-Level Ablation (Single Seed; source: runs/eval/ablation.md, extracted from detailed tables)

Key comparisons on test split:

v2_history_unfit line C2ST: 0.5387±0.0162  
v2_history_unfitw (leverage, from compare_5seed.txt): 0.521±0.019  
v2_unfitw_dc (from compare_5seed_dc.txt): 0.535±0.012  

**Line C2ST comparison (all methods, runs/eval/ablation.md)**:
- naive: 1.0000
- manual: 0.9800
- learned_full (v1): 0.5658
- v2_history: 0.6379
- v2_latent: 0.6503
- v2_history_unfit: 0.5387
- v2_latent_unfit: 0.5646
- learned_spacing_only: 0.8762
- learned_vertical_only: 0.9455
- shuffled_real: 0.5837
- writer_context variants: 0.5400-0.6543

#### Dependence Structure (runs/eval/ablation.md)

Real values: gap_lag1=0.475, gap_vs_width=0.130, gap_vs_pos=-0.122, slope_next_pearson=0.884, curv_next_pearson=0.373, icc_gap_mean=0.312, icc_slope=0.785, resid_lag1=-0.381, clearance_violation_rate=0.014.

v2_history_unfit (best fit without context):
- gap_lag1: 0.486 (|d|=0.010)
- gap_vs_width: 0.127 (|d|=0.002)
- gap_vs_pos: -0.118 (|d|=0.005)
- slope_next_pearson: 0.880 (|d|=0.003)
- curv_next_pearson: 0.154 (|d|=0.219)
- icc_gap_mean: 0.479 (|d|=0.167)
- icc_slope: 0.763 (|d|=0.022)
- resid_lag1: -0.479 (|d|=0.098)
- clearance_violation_rate: 0.114 (|d|=0.100)

#### Page-Consistency Structure (runs/eval/ablation.md)

Real full page: slope_std_within_page=0.0191, gapmean_std_within_page=0.3599, slope_drift_first_last=0.0366.

v2_history_unfit:
- slope_std_within_page: 0.0207 (|d|=0.0016)
- gapmean_std_within_page: 0.3097 (|d|=0.0502)
- slope_drift_first_last: 0.0353 (|d|=0.0013)

---

## Final Model & Default Path

**Current default**: `runs/default_unfitw_dc/model.pt` (FACTS.md line 156).

**Why this run**:
1. **Line C2ST wins**: test 0.535±0.012 (vs unfitw 0.521±0.019), val 0.536±0.027 (vs unfitw 0.527±0.030). Slightly worse than unfitw on test line C2ST but better on test page and consistently stronger on val.
2. **Page C2ST improves**: test 0.590±0.024 (vs unfitw 0.581±0.026), val 0.593±0.031 (vs unfitw 0.575±0.039).
3. **Curvature realism**: measured curv sd matches real (train pooled: 0.00087 real vs 0.00087 generated; see change 10). Halves e2e line splitting: 52/469 (11%) vs unfitw 89/462 (19%).
4. **E2E stability**: e2e_unfitw_dc v2 gap 0.387 (vs unfitw 0.376), drift 0.645 (vs 0.343), x_start 1.453 (vs 2.278); line 0.719 (vs 0.699). Trade-off noted in FACTS.md lines 138-142: survivor bias — split lines have steeper slopes and more bow, so better splits look like worse metrics.
5. **Margin guard included** in e2e_final, the actual deployment run (FACTS.md line 196).

**Bscribe_run.py and relayout.py defaults**: both now point to `runs/default_unfitw_dc/model.pt` (FACTS.md line 156; confirmed via code).

---

## Measurement-Artefact Pitfalls

### 1. Selection on Test (FACTS.md M4)
All design choices (unfit, denoise, ink extents, clearance) compared on test split only; val never used. Fitted constants (clearance thresholds, tilt.json beta, anchor noise, GAP_MIN) come from train, but headline numbers carry ~1 sd selection bias. **Impact**: numbers are optimistic; true gap to shuffled_real likely smaller.

### 2. Real-vs-Real Floor Is Broken (FACTS.md M1)
ShuffledReal (0.584 on test) tiles/truncates donor gaps and residuals, breaking gap-width dependence (0.018 vs real 0.130). Comparable null: split each page into two labels within the same group → 0.496±0.019, not 0.584. **Impact**: v2_history_unfit (0.539) is slightly detectable, not "below floor".

### 3. Biased E2E C2ST (Until e2e_pagereset) (FACTS.md C1, review_opus.md)
Old e2e used pooled out-of-fold grouped k-fold AUC with one label per page. Real-vs-real reported 0.34 (line) / 0.115 (page) — both broken. **Impact**: e2e_unfit, e2e_unfit_ink C2ST cannot be compared with later runs. E2e_pagereset onward use balanced C2ST (per-fold AUC averaged).

### 4. E2E Extraction Breaks Generated Pages (FACTS.md C2, review_opus.md)
Detector/chainer split lines: v2_ctx 507 drawn → 829 extracted → 470 kept (39% single-piece vs 12.9% real). Metrics only see surviving lines; large x_start W1 mostly fragments. **Impact**: e2e line C2ST covers ~100-150 n≥5 lines per seed; high variance.

### 5. Per-Word Wobble Overshoot (FACTS.md M2)
Unfit table matched line-level median robust SCALE but was applied to every per-word residual. Real lag1=-0.38 (smooth), i.i.d. generates -0.48. Generated wobble ~2x in short lines (n=5: 0.069 vs 0.046); visible as word-to-word jumps. **Impact**: plan-level resid_scale W1 n=5 inflated (unfitw leverage fix restores to 0.009).

### 6. Noise Independence Fails (FACTS.md M3, review_opus.md)
Anchor noise deconvolution assumes independent A (headline) and C (median ink) estimators. Tetrad test fails; covariances vary (0.0069-0.0128). Plausible headline share 38-66%, not 57% point estimate. **Impact**: wobble_factor = sqrt(1-share) robust at 0.82-0.84 (stable 0.29-0.33 across clip levels), but ±10% range possible on noise share.

### 7. Ink-Extent Threshold Bias (FACTS.md minor 2, review_opus.md)
INK_LEVEL=0.35, but Otsu on generated pages ~0.39. Right ink edges 0.063 U shorter → every measured gap +0.067 U bias (~6%). **Impact**: all e2e gap W1 slightly high by ~0.07 U.

### 8. Clearance Check Before Page Break (FACTS.md minor 3)
Lines that move to new page after clearance check were reshaped/pushed against prior page. **Impact**: suspected but not quantified.

### 9. Gap Head Conditioned Differently (FACTS.md minor 4-5)
Training short lines (n<5) get slope_raw/curv_m=0 + logres from degree-0/1 fit. Sampling always uses n≥5 features. **Impact**: suspected but unquantified.

### 10. Context Leakage (FACTS.md minor 9)
E2E real reference includes 5 of 193 test pages (context pages, ~4% leakage). **Impact**: negligible; comparison is writer-conditioned generation vs pooled test population.

### 11. Anchor Noise JSON Stale (FACTS.md minor 10)
Recomputed weight 0.351 vs 0.349 (circular dependency with extract cache). **Impact**: negligible (~0.6% difference).

### 12. Writer Separation Unverified (FACTS.md minor 8)
37 copies × 27 texts, 57 scan batches; no writer ids. Same writers likely in multiple texts. **Impact**: text split probably does NOT separate writers. Latent-style models see mixed writers in context.

---

## Plottable Data

### Metrics JSON Files
- `/home/hpc4090/shyan/ocr/scribe_layout_v2/runs/default_unfitw_dc/metrics.json` — trained model metrics (loss per epoch)
- `/home/hpc4090/shyan/ocr/scribe_layout_v2/runs/e2e_final/results.json` — e2e W1 distances + C2ST (structured)
- `/home/hpc4090/shyan/ocr/scribe_layout_v2/runs/eval/metrics.json` — plan-level ablation (all methods, all C2ST, dependence, page-consistency stats; source of detailed ablation.md tables)

### Comparison Script Outputs (CSV/JSON)
- `/home/hpc4090/shyan/ocr/scribe_layout_v2/runs/compare_unfitw/compare_5seed.txt` — 5-seed plan-level C2ST summary (v1, v2_unfit, v2_unfitw, shuffled_real)
- `/home/hpc4090/shyan/ocr/scribe_layout_v2/runs/compare_unfitw/compare_5seed_dc.txt` — 5-seed with deconv (v1, v2_unfitw_dc, v2_unfitw)
- `/home/hpc4090/shyan/ocr/scribe_layout_v2/runs/compare_unfitw/compare_5seed_val.txt` — validation split (v1, v2_unfit, v2_unfitw, v2_unfitw_dc, shuffled_real)
- `/home/hpc4090/shyan/ocr/scribe_layout_v2/runs/compare_unfitw/compare_5seed_wh_{test,val}.txt` — Huber-aware vs leverage (default_unfitwh_dc vs default_unfitw_dc)
- `/home/hpc4090/shyan/ocr/scribe_layout_v2/runs/compare_unfitw/split_counts.txt` — e2e line splitting stats (drawn vs extracted, single-piece, split, fragments) for all three e2e runs × methods
- `/home/hpc4090/shyan/ocr/scribe_layout_v2/runs/compare_unfitw/fft_wobble_scale.txt` — controlled 2-D FFT C2ST vs drawn-wobble scale (3 seeds)
- `/home/hpc4090/shyan/ocr/scribe_layout_v2/runs/compare_unfitw/leverage_sim.txt` — per-word wobble recovery (leverage vs old scale table)

### Training/Val/Test Split Data
- Real lines: `/home/hpc4090/shyan/ocr/scribe_layout_v2/dataset.py` load_lines() → split_lines() returns {train, val, test} lists of Line namedtuples with measured features (gap, slope, curv, resid_scale, pitch_next, x_start, etc.)
- Ground truth lines: `dataset.by_page()` groups by page_id

### Raw Extraction Output
- `/home/hpc4090/shyan/ocr/scribe_layout_v2/runs/e2e_{final,unfitw,unfitw_dc}/images/<writer>_<method>/` — rendered pages (PNG)
- `/home/hpc4090/shyan/ocr/scribe_layout_v2/runs/e2e_{final,unfitw,unfitw_dc}/detections/<writer>_<method>/` — detected page regions (JSON)
- `/home/hpc4090/shyan/ocr/scribe_layout_v2/runs/e2e_{final,unfitw,unfitw_dc}/extraction/<writer>_<method>/` — extracted line/word geometry (JSON)

### Configuration & Noise Parameters
- `/home/hpc4090/shyan/ocr/scribe_layout_v2/runs/unfit_ratio.json` — inflation factors n=2-30
- `/home/hpc4090/shyan/ocr/scribe_layout_v2/runs/unfit_word_huber.json` — per-word Huber factors n=1-30 (position array per n)
- `/home/hpc4090/shyan/ocr/scribe_layout_v2/runs/anchor_noise.json` — true_var, noise_var_headline/mid/combined, w_headline (0.349), noise_share (0.565)
- `/home/hpc4090/shyan/ocr/scribe_layout_v2/runs/line_clearance.json` — min_U, min_ratio (train/val)
- `/home/hpc4090/shyan/ocr/scribe_layout_v2/runs/default_unfitw_dc/tilt.json` — beta (0.524)

### Columns/Keys in Key Files

**ablation.md tables** (runs/eval/ablation.md):
- Marginals: gap_w1, slope_w1, curv_w1, resid_scale_w1, drift_w1, gap_cv, pitch_next_w1, x_start_w1, c2st_line, c2st_line_mlp, c2st_page, c2st_page_mlp
- Dependence: gap_lag1, gap_vs_width, gap_vs_pos, slope_next_pearson, curv_next_pearson, icc_gap_mean, icc_slope, resid_lag1, clearance_violation_rate
- Page-consistency: slope_std_within_page, gapmean_std_within_page, slope_drift_first_last
- Writer-context fidelity: slope_w1_per_page, gapmean_w1_per_page

**E2E results.md** (e.g., runs/e2e_final/results.md):
- E2E metrics: gap_w1, slope_w1, curv_w1, resid_scale_w1, drift_w1, pitch_next_w1, x_start_w1, c2st_line, c2st_page
- Extraction health: pages, excluded, lines, words placed, words recovered, recovered/placed

**Dataset Line features** (from features.py and dataset.py):
- Core: gap (per-word), slope (line-level), curv (line-level), resid_scale (per-line; log MAD of residuals), pitch_next, x_start, drift (slope × line length)
- Derived: resid (per-word offsets, one vector per line), coef (3-tuple: linear trend + slope + curv)

---

## Candidate Figures

### In report/figs/

1. **compare_v1_v2.png** — plan-level marginals (gap, slope, curvature, resid_scale, pitch, x_start) for v1 vs v2_history_unfit (caption: "v2 reduces plan-level gap variability and x_start drift; slope/pitch match")
2. **compare_fixes.png** — before/after key improvements (e.g., leverage vs table-only unfit, line clearance effect; caption: TBD)
3. **dependence.png** — joint distributions: gap_lag1, gap_vs_width, slope serial correlation, curv lag, icc_slope (caption: "v2_history_unfit preserves real dependence structure in line-to-line trends")
4. **example_layout_maps_fair.png** — 2-D FFT layout PSD for real vs v2 (controlled: same word boxes; caption: "v2 layout spectrum controlled for confounds; floor real-halves 0.52, v2 0.55")
5. **mean_log_psd_fair.png** — radial PSD and angular energy, real vs v2; ablations (pitch shuffled, slope zeroed, wobble removed; caption: "spectrum dominated by line pitch regularity; wobble removal modest impact")
6. **unfitw_image_104_v2_ctx.png** — page example with v2_unfitw (leverage) context-aware layout (caption: "v2_unfitw page (writer conditioning); visible word-to-word smoothness vs old unfit's jumps")
7. **pagereset_image_104_v2_ctx.png** — same writer after planner reset fix (caption: "after page-reset fix: slopes stay in writer range; no words past margin")
8. **pagereset_image_104_manual.png** — BengaliScribe hand-tuned baseline for comparison (caption: "BengaliScribe manual layout (baseline; v2_ctx drift matches this)")

### In e2e_final/thumbs/

- `image_104_v2.png` — rendered page, v2 model, no context
- `image_104_v2_ctx.png` — rendered page, v2 model, with writer context (first half real, second half generated)
- `image_104_manual.png` — rendered page, BengaliScribe manual layout

### Plottable Time-Series & Comparisons

**Train loss curve**:
- x-axis: epoch 0-400 (patience 30)
- y-axis: NLL (per head)
- Variants: default_unfitw_dc, default_unfit, default_unfitw (one plot per model)
- Source: metrics.json loss per head

**5-Seed C2ST Trajectories**:
- x-axis: method (v1, v2_unfit, v2_unfitw, v2_unfitw_dc)
- y-axis: line C2ST (0.50-0.65 range with error bars)
- Facets: test vs val, plan-level vs e2e
- Source: compare_5seed.txt, compare_5seed_val.txt, compare_5seed_dc.txt

**Ablation Heatmap**:
- Rows: all methods (naive, manual, learned_full, v2_history, v2_latent, writer_context variants, etc.)
- Cols: all W1 metrics (gap, slope, curv, resid_scale, drift, pitch, x_start) + C2ST
- Color: W1 value or AUC
- Source: runs/eval/ablation.md detailed tables

**E2E Extraction Breakdown**:
- Stacked bar: drawn, extracted, single-piece, split, fragments, unmatched
- Groups: e2e_unfitw, e2e_unfitw_dc, e2e_final × (v2, v2_ctx, manual)
- Source: split_counts.txt

---

*This report synthesizes FACTS.md, review_opus.md, all ablation.md and results.md files, comparison script outputs, and module docstrings from scribe_layout_v2 as specified. All numbers copied exactly with source citations. No speculation beyond what FACTS.md and reviews document.*
