# Verified facts for the v2 report (compiled by the coordinator; all numbers measured)

Lower AUC = closer to real; 0.5 = indistinguishable. "U" = writer unit = median word core height (~3.5 mm).

## v2 changes over v1 (code in scribe_layout_v2/, v1 frozen in scribe_layout/)
1. Combined word anchor (extract.py, anchor_noise.py): anchor = w*headline + (1-w)*(median ink row - page offset), w = 0.349,
   inverse-noise weight. Noise deconvolution from two estimators: cov(rA, rC) = true wobble var 0.00906 U^2; noise var
   headline 0.01176, mid 0.00631, combined 0.00411. Headline-only noise share 0.565 (Opus review M3: independence assumption
   fails a tetrad test with 2 extra estimators; plausible range 38-66%; winsorising level moves it 0.61 -> 0.526; the
   COMBINED-anchor share 0.29-0.33 is stable, so the generation-time de-noise factor 0.82-0.84 is robust).
2. De-noised wobble in BengaliScribe (bscribe_layout.wobble_factor): sqrt(1 - share_combined) ~ 0.83.
3. Line clearance rejection sampling (line_clearance.py, bscribe_layout.py): thresholds from real train 1st percentiles:
   min distance >= 2.603 U, min/mean >= 0.797. Redraw slope/curv/logres up to 30 times, else push line down.
   Plan-level (no rejection) violation rate: generated 0.09-0.15 vs real 0.014.
4. Unfit targets v1 attempt (unfit.inflation(n)): median-MAD ratio table from simulation; factors n=3..12:
   2.505, 1.52, 3.159, 2.076, 1.644, 1.408, 1.351, 1.29, 1.256, 1.234. Fixed double fit-shrinkage of wobble
   (plan-level line C2ST v2_history 0.638 -> v2_history_unfit 0.539).
   BUG (Opus review M2, verified by coordinator simulation): the table matches the median robust SCALE of a line, but was
   also applied to every per-word residual target -> per-word wobble overshoot. Simulation (iid wobble sd 0.150 on real
   train-line x positions, refit with dataset.fit_line):
   n:            3     4     5     6     7     8     9     10    11    12
   raw fit sd  .087  .107  .098  .109  .115  .121  .124  .126  .129  .132
   leverage    .151  .149  .142  .147  .148  .150  .150  .149  .149  .149
   old table   .219  .163  .311  .227  .190  .170  .168  .163  .162  .159
5. FIX: per-word leverage correction unfit.word_inflation(x) = 1/sqrt(1-h_i), h_i = leverage of word i in the same
   polynomial fit (deg 2 if n>=5, 1 if n>=3, else 0; centred x), capped at 4. logres (line scale) keeps the table.
   Config configs/default_unfitw.yaml (unfit_words: true) -> runs/default_unfitw/model.pt. NEW RECOMMENDED DEFAULT.
6. Word rotation beta = 0.524 x line tangent angle (tilt.json, Huber regression of headline tilt on tangent).
7. Ink-extent placement (bscribe_layout.ink_extents, INK_LEVEL 0.35): model gaps are ink edge to ink edge; placing by
   canvas width made every gap ~0.5 U too wide. (Review minor: page Otsu ~0.39 -> residual +0.067 U gap bias.)
8. Planner reset per page (Opus review C3, verified): the planner history ran across pages (history count reached 56-72
   lines vs max 16 in training; slopes random-walked, e.g. page mean -0.158 hitting the -0.2 clamp; 2 words past page
   bottom). Fix: each new page restarts from a fixed reference state (real context page if given, else end of the first
   generated page). After fix: no words past bottom; page slope means stay near the writer's (e.g. image_104 ctx:
   -0.053,-0.054,-0.074,-0.009,-0.016 vs real -0.023).
9. Balanced C2ST for end-to-end (Opus review C1, verified): with one label per page group, pooled-OOF grouped k-fold AUC is
   biased low (old e2e real-vs-real "floor" 0.34 line / 0.115 page). Fix: groups dealt into folds per label + mean of
   per-fold AUCs (evaluate.grouped_kfold_auc(balanced=True), e2e only). New floor over 5 seeds: line 0.503, page 0.527
   (e2e run floor: 0.534 / 0.508). Plan-level C2ST unchanged (groups there hold both labels).
10. Latent writer-style model: context bug = train/test mismatch in z precision (training z = leave-one-out over ~whole page;
    context z = half page). Causal-prefix z fix (runs/latent_unfit_fix): latent+ctx slope W1 0.00977 -> 0.00406,
    drift W1 0.262 -> 0.106, line C2ST 0.561 -> 0.547 (history+ctx: 0.00332 / 0.110 / 0.540). BUT it regresses the
    no-context latent model (slope W1 0.00533 -> 0.01283, page C2ST 0.684 -> 0.829): one fixed z reused at all line
    positions mismatches position-dependent z precision. UNRESOLVED; latent is not the default.

## Plan-level evaluation (test split by text, 193 pages / 1495 lines; 5 seeds; mean +- sd)
| method | line C2ST | page C2ST | 2-D FFT C2ST (controlled) | resid_scale W1 n=5 / 6-7 / 8-10 / 11+ |
| v1 (scribe_layout/runs/default) | 0.566+-0.021 | 0.654+-0.043 | 0.567+-0.034 | .011 .025 .055 .074 |
| v2_history_unfit (old table) | 0.539+-0.016 | 0.660+-0.004 | 0.653+-0.024 | .039 .028 .029 .058 |
| v2_history_unfitw (leverage) | 0.521+-0.019 | 0.581+-0.026 | 0.566+-0.025 | .009 .011 .041 .062 |
| shuffled_real | 0.584+-0.016 | 0.605+-0.012 | 0.529+-0.050 | .021 .018 .054 .100 |
manual (BengaliScribe hand-tuned): line 0.980, page 1.0, FFT 0.99.
Review M1: shuffled_real (0.584) is NOT a real-vs-real floor (it breaks the gap-width relation); the comparable
within-page null is 0.496+-0.019. Review M4: all design choices compared on TEST (val never used) -> model selection on
test; fitted constants come from train (no leakage), but numbers are optimistic by selection. Writer overlap across texts
likely but unverifiable (no writer ids).
Older 1-seed plan-level table (runs/eval/metrics.json): v2_history 0.638/0.752, v2_latent 0.650/0.749, v2_latent_unfit
0.565/0.684 (line/page).

## FFT wobble-scale diagnostic (controlled 2-D FFT C2ST, 3 seeds, v2_history_unfit with drawn resid scaled)
v1 0.612; v2 fitted-target model (runs/default) 0.628; v2_unfit x1.0 0.683; x0.83 0.658; x0.5 0.578; x0.0 0.601.
-> the excess per-word wobble is what the page spectrum detected; motivated fix 5.

## End-to-end (BengaliScribe renders -> real detector -> same extraction; 5 writers x 2 seeds; test real reference)
Balanced C2ST from e2e_pagereset onward; earlier runs used the biased pooled AUC (do not compare C2ST across that line).
e2e_unfit (canvas-width placement, biased C2ST):  v1 gap W1 0.582 line 0.699 page 0.942 recovered 0.797;
   v2 0.602 / 0.750 / 0.901 / 0.969; v1_ctx 0.526/0.653/0.971/0.714; v2_ctx 0.381/0.595/0.901/0.910; manual 0.685/0.893/0.998/1.004
e2e_unfit_ink (ink-extent placement, biased C2ST): v2 gap 0.471 line 0.667 page 0.840 rec 0.952; v2_ctx 0.207/0.532/0.918/0.866
e2e_pagereset (planner reset + balanced C2ST, model default_unfit):
| method | gap W1 | slope W1 | drift W1 | pitch W1 | x_start W1 | line C2ST | page C2ST | recovered |
| manual | 0.685 | 0.035 | 0.965 | 1.839 | 0.948 | 0.905 | 0.999 | 1.004 |
| v2 | 0.404 | 0.013 | 0.396 | 0.670 | 4.065 | 0.684 | 0.834 | 0.965 |
| v2_ctx | 0.267 | 0.017 | 0.558 | 0.770 | 3.783 | 0.619 | 0.889 | 0.926 |
e2e_unfitw (model default_unfitw): see scribe_layout_v2/runs/e2e_unfitw/results.md (coordinator fills in if not present).
Known e2e caveats (review C2): the detector/chainer splits generated lines (v2_ctx 507 drawn -> 829 extracted -> 470
kept; 39% single-piece extracted lines vs 12.9% real) -> metrics only see surviving lines; large x_start W1 is mostly
split-line fragments (fragment starts mid-page), partly real margin spread. Line C2ST features omit x_start and pitch.
BengaliScribe adds paragraph gaps/indents that hcr pages lack. Only ~50-60 n>=5 generated lines per seed.

## 2-D FFT layout-map experiment (scribe_layout_fft2d/, results.md)
Layout map = filled word boxes at 8 px/U, Hann window, 2-D PSD -> radial PSD (32 bins), angular energy (18 bins),
vertical line-pitch peak. Uncontrolled version (detector pages) was CONFOUNDED: single non-layout features separate
real vs generated (text block height AUC 0.985, median word width 0.90-0.93, box height ~0.82) -> AUC 0.96-1.0 meant
text/page-fill/glyphs, not layout. Controlled version: generators run on each real test page's own word widths; real and
generated rendered with the same fixed box (anchor -0.25..+0.85 U); window 32x18 U inside the real 10th-pct text block;
confound AUCs all 0.5 by construction. Floor (real halves) 0.483 logreg / 0.522 mlp. Single-seed AUCs move up to +-0.09
(use the multi-seed numbers above). Ablations on real pages (population-pooled): shuffling line pitch moves the mean
log-PSD 6x more (L2 0.080) than zeroing slope (0.013) or removing wobble (0.013) -> the spectrum is dominated by line
pitch regularity; peak-prominence statistic is unreliable (rises after destroying periodicity).

## 1-D Fourier experiment (scribe_layout_fourier/, runs/results.md, fair_dense_psd.json)
Dense headline traces, Lomb-Scargle PSD ~ 1/f^alpha, alpha = 1.06; spectral synthesis generator. Fair comparison on
layout-realisable traces (rigid rotated words), band wavelength >= 4 U, distance to real: manual 3.786, v1 0.702,
fourier 0.668, fourier continuous (not realisable) 0.244. Agent's earlier "17x better" claim was a floor artifact; its
"6x x_start" claim came from including indents.

## Code review (runs/review_opus.md) summary
Claims after review: plan-level line C2ST holds (null ~0.50, selection on test); old e2e line 0.532 did NOT hold;
gap W1 real but on surviving lines with +0.07 U bias; "57% noise" needs range 38-66% and refers to headline-only anchor.
Tests that passed vacuously: noise-floor test tolerance |AUC-0.5|<0.4; TestC2ST single-sample groups; latent slope test on
a fake model; no tests for bscribe_layout, unfit, anchor_noise, line_clearance.

# ADDENDUM (round 2; coordinator-verified)

## 11. De-noised curvature (extreme deconvolution) -> NEW DEFAULT runs/default_unfitw_dc/model.pt
Config configs/default_unfitw_dc.yaml = default_unfitw + `deconv: [curv]`.
Motivation (e2e line splitting): in e2e_unfitw, matching extracted lines to drawn GT lines (annotation polygons;
script runs/compare_unfitw/split_diag.py): v2 89 of 462 drawn lines split (19%), 127 single-piece lines;
manual 6 of 355 (1.7%). At split points the median-ink-row step between neighbouring words is 1.2 U median
(q10 0.72, q90 1.83) vs chainer JOIN_DEV 1.0 U; 74% exceed it; merge residual finite but >= 0.6 U for 86%.
Not a glyph effect: (mid - headline) offset IQR generated 0.19 U vs real 0.30 U. Cause = lines that bow too much.
Line sag = c2 * (half line length)^2 (U), plan-level, test split, seed 0 (runs/compare_unfitw/sag.py):
              |sag|>0.5U   sag sd n5-6  n7-8   measured curv sd n5-6 / n7-8 / n9-10 / n11+
 real          0.108       0.265  0.322        0.00125 / 0.00083 / 0.00112 / 0.00086
 v1            0.170
 v2_unfitw     0.175       0.358  0.509        0.00146 / 0.00113 / 0.00105 / 0.00113
 v2_unfitw_dc  0.093       0.260  0.314        0.00117 / 0.00077 / 0.00073 / 0.00071
Adjacent-word max step > 1 U: real 0.099, v1 0.173, unfitw 0.169, unfitw_dc 0.138.
Cause: curvature is sampled before the word count is known; fitted c2 on 5-6-word lines is mostly noise.
Noise model: var(c2_hat) = s^2 [(V'V)^-1]_22 (centred x, OLS approx of the Huber fit), s^2 = per-word residual
variance, estimated per line as RSS/(n-3) and POOLED by word count n (n>=11 pooled). Train split
(runs/compare_unfitw/curvnoise2.py): measured curv sd n5-6 0.00129, noise sd 0.00092, implied true 0.00091;
n7-8 0.00092/0.00053/0.00075; n9-10 0.00084/0.00053/0.00066; n11+ 0.00087/0.00047/0.00073 -> true curvature
sd ~0.0007-0.0009 at every n (consistent habit); measurement noise = 38.4% of curv target variance (train tables).
Method: MDN curv head likelihood p(c_meas) = sum_k pi_k N(c_meas; mu_k, sigma_k^2 + v_i) (extreme deconvolution,
Bovy, Hogg & Roweis 2011, Ann. Appl. Stat.); sampling draws from sum_k pi_k N(mu_k, sigma_k^2) (noise-free).
Code: nn_utils.MDN.nll(noise_var), model.head_nll (cfg "deconv"), features.curv_noise_var.
Val NLL curv head (scored on measured curvature under the convolved likelihood): -1.004 vs -0.975 (unfitw);
all other heads bit-identical (same seed).
Plan-level 5 seeds, TEST: unfitw_dc line 0.535+-0.012 page 0.590+-0.024 FFT 0.550+-0.020; resid_scale W1
.007/.011/.042/.061 (n=5/6-7/8-10/11+). Plan-level slope/drift/curv W1 (5 seeds, runs/compare_unfitw/drift.py):
unfitw slope 0.0045 drift 0.158 curv 0.0002 pitch 0.109; unfitw_dc slope 0.0044 drift 0.155 curv 0.0001 pitch 0.114.
E2E e2e_unfitw_dc (balanced C2ST): manual as before; v2 gap 0.387 slope 0.021 drift 0.645 pitch 0.465 x_start 1.453
line 0.719 page 0.844 recovered 0.983 (473 lines kept); v2_ctx gap 0.183 slope 0.023 drift 0.695 pitch 0.535
x_start 2.494 line 0.654 page 0.847 recovered 0.929.
Splitting vs drawn GT lines: unfitw 89/462 split, 127 single-piece, 113 extra fragments; unfitw_dc 52/469 split,
64 single-piece, 61 extra fragments.
SURVIVOR BIAS (explains higher e2e slope/drift W1 for dc): drawn slope from GT line polygons
(runs/compare_unfitw/surv.py). e2e_unfitw: split lines median |slope| 0.046 (q90 0.096), bow 0.66 U; kept 0.026
(0.070), bow 0.32 U -> the old e2e metrics silently excluded steep/bowed lines. e2e_unfitw_dc: split 0.052,
kept 0.025; ALL drawn lines median |slope| 0.027 q90 0.071 vs real test measured 0.029 / 0.086. So drawn slopes
match real; e2e slope/drift W1 are not comparable across runs with different split rates.

## 12. Validation-split check (selection-free; runs/compare_unfitw/compare_5seed_val.py -> compare_5seed_val.txt)
VAL split, 5 seeds:      line C2ST      page C2ST      FFT C2ST       resid_scale W1 n5 / 6-7 / 8-10 / 11+
 v1                     0.560+-0.024   0.631+-0.040   0.531+-0.033   .013 .025 .055 .089
 v2_unfit (old table)   0.566+-0.028   0.660+-0.022   0.599+-0.051   .049 .030 .024 .050
 v2_unfitw (leverage)   0.527+-0.030   0.575+-0.039   0.528+-0.024   .012 .011 .035 .072
 v2_unfitw_dc           0.536+-0.027   0.593+-0.031   0.535+-0.044   .011 .011 .035 .069
 shuffled_real          0.577+-0.023   0.597+-0.030   0.480+-0.034   .016 .017 .055 .089
Reading: leverage-family models beat v1 on line and page C2ST on val too; FFT tie. v2_unfit (old table) looked
better than v1 on test line C2ST (0.539) but not on val (0.566) -> that earlier claim was a selection effect.
unfitw vs unfitw_dc differ within noise on all C2STs (test and val). dc chosen as default for the curvature
realism and halved e2e splitting, not for C2ST.

## Default model now: scribe_layout_v2/runs/default_unfitw_dc/model.pt (bscribe_run.py and relayout.py defaults).

# ADDENDUM (round 3; coordinator-verified)

## 13. Huber-aware per-word correction: NEGATIVE result (not adopted)
Test agent found unfit.word_inflation (OLS leverage 1/sqrt(1-h)) under-corrects first/last words by ~10-25% for
n=4..10, because dataset.fit_line is Huber-IRLS (tests/test_unfit.py documents it). Implemented
unfit.word_inflation_huber: per (n, word position) factor sd(offset)/sd(Huber-fit residual), 4000 i.i.d. N(0,1)
sims, evenly spaced words, cached runs/unfit_word_huber.json. Factors (edge ... edge), Huber vs leverage:
 n=5: [4.0(cap) 1.14 1.46 1.14 4.0] vs [2.96 1.26 1.39 1.26 2.96]; n=6: [2.91 1.13 1.2 1.21 1.13 2.89] vs [2.37 1.2 1.26 ...];
 n=8 edges 2.11/2.13 vs 1.85; n=10 edges 1.76/1.74 vs 1.62; n=12 edges 1.55/1.57 vs 1.49.
Model runs/default_unfitwh_dc (config default_unfitwh_dc.yaml, unfit_words: huber, deconv: [curv]).
5 seeds (runs/compare_unfitw/compare_5seed_wh_{test,val}.txt):
 TEST: unfitw_dc line 0.535+-0.012 page 0.590+-0.024 FFT 0.550+-0.020 | unfitwh_dc 0.544+-0.012 0.614+-0.011 0.583+-0.029
 VAL : unfitw_dc line 0.536+-0.027 page 0.593+-0.031 FFT 0.535+-0.044 | unfitwh_dc 0.551+-0.022 0.619+-0.043 0.551+-0.049
Huber-aware is worse on all three C2STs on both splits (same direction 6/6). Interpretation (not verified further):
the exact correction is exact only for i.i.d. wobble; real wobble is smoother than i.i.d. (review M2: lag-1 residual
correlation -0.38 real vs -0.48 i.i.d.), and edge residuals also carry line-end detection noise, so a 4x edge
inflation overshoots. Default stays runs/default_unfitw_dc. Proper fix (open): calibrate the correction under a
correlated-wobble model fitted to the real lag correlations.

## 14. Margin guard (bscribe_layout.layout_v2) -- test agent finding, fixed
Before: only `center + 1.6*body > H - mb` was checked; a sloped/bowed first line could rise ~1-1.4 U (2.5-5 mm at
300 dpi) into the top margin, and slopes could carry ink below the bottom margin. Now: ink_rows() gives each word's
ink rows; ink_span() = the line's ink top/bottom along its current trajectory +-RESID_MARGIN (0.3 U, ~2 sd of the
drawn wobble). A line whose ink would pass the bottom margin starts a new page and is REPLANNED from scratch
(words back to the queue; the discarded draw never enters the page history), and a page's first line is pushed
down until its ink clears the top margin. Test tests/test_bscribe_layout.py::TestWordsStayInsideMargins asserts
ink within margins +-0.5 U over 2 seeds x (no context, context); it passes on the new code and FAILS on the old
code (ink top 178 px vs limit 216 px).

## 15. Tests (scribe_layout_v2/tests): 70 tests, ~17 s (was 32 tests, ~8 min)
New: test_unfit.py, test_bscribe_layout.py (placement order across pages, margins, exact per-page history reset with
and without context, clearance stats), test_line_clearance.py, test_anchor_noise.py (replicates main() formulas on
synthetic data: anchor_noise.main is not importable without real data). Added TestPhysWords, TestUnfitModeMapping,
balanced grouped-k-fold AUC tests (real-vs-real ~0.5 with balanced=True; balanced=False biased low on single-label
groups; shifted data > 0.8). Tightened e2e noise-floor tolerance 0.4 -> 0.15 (measured max deviation ~0.08).
Fixed test fixture make_line (all-zero x made the curvature noise term singular; library now uses pinv).
TestLatentReducesSlopeDrift kept (fake model) with a docstring explaining why.

## 16. Final e2e (runs/e2e_final; default model runs/default_unfitw_dc + margin guard; balanced C2ST; floor line
0.534 page 0.508):
| manual | gap 0.685 | slope 0.035 | drift 0.965 | pitch 1.839 | x_start 0.948 | line 0.905 | page 0.999 | rec 1.004 |
| v2     | 0.425 | 0.021 | 0.672 | 0.390 | 1.239 | 0.719 | 0.850 | 0.981 (440 lines kept) |
| v2_ctx | 0.198 | 0.020 | 0.621 | 0.503 | 2.130 | 0.619 | 0.845 | 0.940 (432 lines kept) |
Splitting vs drawn GT (v2): 57 of 453 drawn lines split (12.6%), 80 single-piece, 69 extra fragments
(e2e_unfitw_dc: 52/469 = 11.1%) -> same within noise.
