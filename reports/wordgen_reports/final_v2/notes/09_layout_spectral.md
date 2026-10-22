# Layout Spectral Experiments

Two spectral side-experiments on page layout realism tested whether synthetic page layouts match real ones in terms of spatial frequency structure. Both used different spectral hypotheses and comparison protocols.

---

## Project 1: 2-D FFT Layout-Realism Metric (scribe_layout_fft2d)

### Question

Do 2-D power spectra distinguish real from generated page layouts? Can spectral methods detect differences in spacing, pitch, slope, and word-to-word wobble?

### Method

- **Core hypothesis**: 2-D FFT of a page's layout map (word boxes: position + extent, no glyph shape) produces a power spectrum sensitive to layout structure.
- **Page representation**: Layout map = filled word boxes at scale 1 U (word core height) = 8 pixels, windowed to fixed size (48 U width × 40 U height) with Hann taper (`scribe_layout_fft2d/raster.py`, DPU=8.0, WIN_U=(48, 40)). Ink map = binarised page pixels at same normalisation.
- **Spectral summaries** (`scribe_layout_fft2d/spectrum.py`):
  - Radially-averaged PSD (32 frequency bins, cycles/U)
  - Angular energy distribution (18 bins, 0–180°)
  - Vertical-axis line-pitch peak (frequency, prominence, FWHM)
- **Comparison metrics** (`scribe_layout_fft2d/metrics.py`):
  - L2(mean log-PSD): Euclidean distance between mean radial log-PSD curves
  - W1(per-bin): Wasserstein distance per frequency bin, averaged
  - C2ST AUC: Logistic regression and 1-hidden-layer MLP classifiers, 5-fold CV, distinguishing real from method
  - Confound AUCs: Single-scalar AUC for block height/width, median box width/height, word count (catch non-layout differences)
- **Fairness protocol** (`scribe_layout_fft2d/fair.py`): **Controlled (text-matched) main test**
  - Every generator runs on each real TEST page's own word widths (l.w) and line count
  - Real pages reconstructed with fixed BOX_TOP=-0.25, BOX_BOT=0.85 convention (not ink extents)
  - Window: (32.0, 18.0) U (just under 10th percentile of real test pages' block dimensions)
  - All differences attributable to layout alone (spacing, pitch, slope, x_start, word-to-word wobble)
- **Ablations** (`scribe_layout_fft2d/ablate.py`): Shuffle line pitch (destroy pitch regularity), zero slope/curvature (flatten lines), remove wobble (residual jitter). Measured at population level (pooled pages → single peak) and per-page paired level.

### Metrics

**Sources of all numbers**: `/home/hpc4090/shyan/ocr/scribe_layout_fft2d/runs/results.md` and `/home/hpc4090/shyan/ocr/scribe_layout_fft2d/runs/metrics.json`

#### Controlled (Text-Matched) Comparison — Main Fair Test

Real-vs-real noise floor (TEST pages split in half): n_a=96, n_b=97
- L2(mean log-PSD) = 0.040 [source: results.md line 27]
- W1(per-bin) = 0.056 [source: results.md line 27]
- C2ST AUC logreg = 0.483 [source: results.md line 27]
- C2ST AUC mlp = 0.522 [source: results.md line 27]
- Confound AUCs: block_h=0.505, block_w=0.472, med_box_w=0.475, med_box_h=0.500, n_words=0.525 [source: results.md line 27]

Method vs real TEST (controlled, text-matched, 193 pages each):

| Method | L2(mean log-PSD) | W1(per-bin) | C2ST AUC logreg | C2ST AUC mlp | confound:block_h | confound:block_w | confound:med_box_w | confound:med_box_h | confound:n_words |
|---|---|---|---|---|---|---|---|---|---|
| manual | 0.539 | 0.291 | 0.992 | 0.993 | 0.767 | 0.609 | 0.500 | 0.500 | 0.500 |
| shuffled_real | 0.036 | 0.042 | 0.552 | 0.592 | 0.475 | 0.388 | 0.500 | 0.500 | 0.500 |
| v1 | 0.033 | 0.040 | 0.552 | 0.538 | 0.423 | 0.494 | 0.500 | 0.500 | 0.500 |
| v2_history_unfit | 0.036 | 0.043 | 0.661 | 0.680 | 0.437 | 0.485 | 0.500 | 0.500 | 0.500 |
| v2_latent_unfit | 0.032 | 0.039 | 0.685 | 0.731 | 0.429 | 0.489 | 0.500 | 0.500 | 0.500 |

[source: results.md lines 31–37]

#### Uncontrolled Comparison (Confounded by Text + Glyph Shape)

Real-vs-real floor (TEST pages split in half): n_a=96, n_b=97
- L2(mean log-PSD) = 0.028 [95% CI 0.032, 0.069] [source: results.md line 61]
- W1(per-bin) = 0.041 [source: results.md line 61]
- C2ST AUC logreg = 0.399 [source: results.md line 61]
- C2ST AUC mlp = 0.450 [source: results.md line 61]

Method vs real TEST (uncontrolled, layout-map spectrum):

| Method | n | L2(mean log-PSD) [95% CI] | W1(per-bin) [95% CI] | C2ST AUC (logreg) | C2ST AUC (mlp) | C2ST AUC ink (CONFOUNDED) |
|---|---|---|---|---|---|---|
| real_train | 200 | 0.017 [0.019, 0.047] | 0.029 [0.036, 0.055] | 0.450 | 0.500 | 0.516 |
| manual | 20 | 0.282 [0.261, 0.318] | 0.244 [0.220, 0.283] | 0.999 | 0.996 | 1.000 |
| v2 | 30 | 0.153 [0.127, 0.194] | 0.131 [0.115, 0.174] | 0.981 | 0.970 | 1.000 |
| v2_ctx | 35 | 0.116 [0.104, 0.152] | 0.098 [0.099, 0.133] | 0.964 | 0.928 | 1.000 |
| v1 | 27 | 0.113 [0.109, 0.155] | 0.104 [0.106, 0.156] | 0.973 | 0.973 | 0.998 |
| v1_ctx | 29 | 0.220 [0.133, 0.326] | 0.217 [0.138, 0.321] | 0.958 | 0.932 | 0.998 |
| v2_old | 30 | 0.153 [0.131, 0.196] | 0.128 [0.114, 0.173] | 0.990 | 0.961 | 1.000 |
| manual_plan | 30 | 0.830 [0.810, 0.857] | 0.693 [0.676, 0.715] | 1.000 | 1.000 | n/a |

[source: results.md lines 63–74]

#### Ablations

**Population-level** (pages pooled, peak measured on pool):

| Ablation | vert peak freq (orig → ablated) | vert peak prominence (orig → ablated) | vert peak FWHM (orig → ablated) | angular circ.var (orig → ablated) | mean-log-PSD L2 vs original |
|---|---|---|---|---|---|
| shuffle_line_pitch | 0.200 → 0.125 | 243.185 → 272.915 | 0.150 → 0.200 | 0.648 → 0.671 | 0.080 |
| zero_slope | 0.200 → 0.200 | 243.185 → 231.055 | 0.150 → 0.150 | 0.648 → 0.652 | 0.013 |
| remove_wobble | 0.200 → 0.200 | 243.185 → 224.933 | 0.150 → 0.150 | 0.648 → 0.648 | 0.013 |

[source: results.md lines 80–84]

**Per-page paired** (mean over pages; noisier, individual-page view; sample of shuffle_line_pitch):

| Feature | original mean | ablated mean | mean delta | n |
|---|---|---|---|---|
| pitch_peak_freq | 0.198 | 0.135 | -0.063 | 193 |
| pitch_peak_prom | 447.195 | 539.768 | 92.574 | 193 |
| pitch_peak_fwhm | 0.079 | 0.083 | 0.004 | 193 |
| angular_circ_var | 0.647 | 0.670 | 0.023 | 193 |

[source: results.md lines 90–93]

### Fairness Controls

**Layout confounds eliminated**: Real pages are reconstructed from their own measured line trajectories using a fixed box convention (BOX_TOP/BOX_BOT) identical on both real and generated side. This ensures:
- Block height is bit-for-bit identical (constant in the controlled test) → confound AUC ≈ 0.5
- Text content (word widths, count) is identical → confound AUCs ≈ 0.5
- Only layout (spacing, pitch, slope, x_start, word wobble) differs

**Remaining limitation**: Ink maps are "confounded by text content + glyph shape" because generated pages were rendered by a different glyph generator than real ink. Results.md explicitly notes this (lines 3–6).

### Conclusion (as Stated in Source)

**Verdict vs existing plan-level LINE C2ST** (line 39):
- Manual = 0.980, v1(learned_full) = 0.566, v2_history_unfit = 0.539, shuffled_real floor ~ 0.50–0.58
- **This experiment's PAGE-level 2-D spectral C2ST AUC (logreg column)**:
  - manual = 0.992 (expected: extreme difference)
  - v1 = 0.552 (expected: near-floor)
  - v2_history_unfit = 0.661 (higher than line-level 0.539, suggests page-level compounding)

The controlled test shows v1 and v2 methods can be detected spectrally at the page level (C2ST AUC 0.552–0.685), whereas manual baseline is easily separated (0.992). This is **above the real-vs-real noise floor of 0.483–0.522**, indicating the metric is sensitive to layout differences—but close enough to the floor that pure layout differences alone produce only modest discriminability.

**Ablations validate the metric**: Shuffling line pitch destroys the vertical-axis peak (freq: 0.200 → 0.125, L2: 0.080), while zeroing slope/wobble has minimal impact (L2 ≈ 0.013). This confirms spectral sensitivity to the intended layout factors.

---

## Project 2: Fourier Model of Line/Baseline Shape (scribe_layout_fourier)

### Question

Can Fourier spectral models (power-law PSD) of the dense headline trajectory match the true spectral properties of real line wobble better than v1's piecewise-quadratic model? Does the spectral structure reveal a "1/f noise" character missed by word-level harness?

### Method

- **Core hypothesis**: Per-page spectral shape (power-law PSD with exponent α and level L) explains line-to-line variation better than iid per-word noise.
- **Dense traces** (`scribe_layout_fourier/dense.py`): Recover 20+ points per word (vs 1 word-level point) by sliding a ~1U-wide window along ink, finding the headline row with longest horizontal run. Match detected words back to geometry by bbox IoU.
- **Spectral fitting** (`scribe_layout_fourier/fit.py` and `scribe_layout_fourier/spectral.py`):
  - Per-line: Detrend (robust linear fit), compute Lomb-Scargle PSD over 48 log-spaced frequencies (1/60 to 2 cycles/U)
  - Per-page: Pool all lines' log-frequency vs log-power, fit OLS slope = α, intercept = level
  - Train split (5099 lines): Bivariate normal on (α, level) → samples α_train, level_train for generation
  - **1/f discovery**: Power-law fit across nearly 2 decades of frequency gives α = 1.06 (R² = 0.99) [source: results.md line 39]
- **Fourier trajectory model** (`scribe_layout_fourier/fourier_model.py`):
  - Vertical (y): Gaussian process with S(f) = exp(level) * f^-α, sampled by spectral synthesis
  - Gaps: Gaussian copula (latent correlated sequence → empirical gap quantile function) + page-level offset
  - Pitch, x_start: iid empirical marginals
  - Hyperparameters (level_shift, slope_scale, curv_scale) chosen on val split via grid search on W1(gap+slope+curv+resid_scale) [source: fit.py docstring and results.md lines 121–125]
- **Evaluation** (`scribe_layout_fourier/evaluate_fourier.py`):
  - Same harness as scribe_layout/evaluate.py (word-level metrics: W1 on marginals, C2ST, dependence stats)
  - Plus dense-PSD comparison: log-spectral RMSE (nats, Lomb-Scargle units)
  - Test split, 5 random seeds

### Metrics

**Sources of all numbers**: `/home/hpc4090/shyan/ocr/scribe_layout_fourier/runs/results.md`, `/home/hpc4090/shyan/ocr/scribe_layout_fourier/runs/metrics.json`, `/home/hpc4090/shyan/ocr/scribe_layout_fourier/runs/fourier.json`, `/home/hpc4090/shyan/ocr/scribe_layout_fourier/runs/spectral/stats.json`

#### Real Test-Split Marginal Medians (U)

gap=1.130, slope=-0.004, curv=-0.000, resid_scale=0.091, pitch_next=4.439, x_start=0.000

[source: results.md line 3]

#### Word-Level Ablation (Same Harness as v1)

| Method | gap_w1 | slope_w1 | curv_w1 | resid_scale_w1 | drift_w1 | pitch_next_w1 | x_start_w1 | c2st_line | c2st_line_mlp | c2st_page | c2st_page_mlp |
|---|---|---|---|---|---|---|---|---|---|---|---|
| learned_full | 0.0378±0.0065 | 0.0050±0.0007 | 0.0001±0.0000 | 0.0421±0.0019 | 0.1578±0.0295 | 0.0924±0.0153 | 1.3136±0.0209 | 0.6187±0.0112 | 0.6271±0.0172 | 0.7361±0.0137 | 0.7031±0.0210 |
| fourier_full | 0.0924±0.0124 | 0.0057±0.0010 | 0.0001±0.0000 | 0.0358±0.0029 | 0.1507±0.0126 | 0.1058±0.0053 | 0.2063±0.0572 | 0.5882±0.0107 | 0.6626±0.0086 | 0.7330±0.0164 | 0.8400±0.0098 |
| fourier_vertical_v1_spacing | 0.0430±0.0080 | 0.0070±0.0020 | 0.0001±0.0000 | 0.0352±0.0020 | 0.1869±0.0757 | 0.0972±0.0159 | 1.3283±0.0157 | 0.5801±0.0065 | 0.6231±0.0098 | 0.7313±0.0077 | 0.7658±0.0090 |
| v1_vertical_fourier_spacing | 0.1004±0.0109 | 0.0052±0.0009 | 0.0001±0.0000 | 0.0433±0.0024 | 0.2029±0.0414 | 0.1182±0.0346 | 0.2454±0.1277 | 0.6420±0.0137 | 0.6854±0.0082 | 0.7493±0.0156 | 0.7983±0.0211 |
| manual | 0.5589±0.0018 | 0.0358±0.0000 | 0.0008±0.0000 | 0.0991±0.0002 | 1.1427±0.0016 | 1.6070±0.0020 | 1.7132±0.0018 | 0.9768±0.0003 | 0.9958±0.0008 | 0.9998±0.0001 | 1.0000±0.0000 |
| shuffled_real | 0.1059±0.0336 | 0.0062±0.0020 | 0.0001±0.0000 | 0.0243±0.0035 | 0.1872±0.0788 | 0.1943±0.0224 | 0.4341±0.1035 | 0.5877±0.0151 | 0.5955±0.0129 | 0.6272±0.0088 | 0.5464±0.0166 |

[source: results.md lines 7–14]

#### Dependence Structure (Real, Then Each Method; |diff| from Real)

| Stat | real | learned_full | fourier_full | fourier_vertical_v1_spacing | v1_vertical_fourier_spacing | manual | shuffled_real |
|---|---|---|---|---|---|---|---|
| gap_lag1 | 0.474 | 0.492 (|d|=0.017) | 0.590 (|d|=0.116) | 0.506 (|d|=0.031) | 0.594 (|d|=0.119) | 0.008 (|d|=0.467) | 0.563 (|d|=0.088) |
| gap_vs_width | 0.128 | 0.132 (|d|=0.005) | -0.009 (|d|=0.136) | 0.132 (|d|=0.004) | -0.002 (|d|=0.130) | 0.007 (|d|=0.121) | 0.019 (|d|=0.109) |
| gap_vs_pos | -0.122 | -0.113 (|d|=0.009) | 0.004 (|d|=0.127) | -0.117 (|d|=0.006) | -0.003 (|d|=0.119) | 0.005 (|d|=0.128) | -0.052 (|d|=0.071) |
| slope_next_pearson | 0.863 | 0.883 (|d|=0.020) | 0.834 (|d|=0.029) | 0.831 (|d|=0.032) | 0.882 (|d|=0.019) | -0.018 (|d|=0.881) | 0.773 (|d|=0.090) |
| icc_gap_mean | 0.311 | 0.567 (|d|=0.256) | 0.312 (|d|=0.001) | 0.563 (|d|=0.253) | 0.294 (|d|=0.017) | 0.005 (|d|=0.306) | 0.364 (|d|=0.053) |
| icc_slope | 0.765 | 0.770 (|d|=0.005) | 0.775 (|d|=0.010) | 0.768 (|d|=0.004) | 0.776 (|d|=0.011) | -0.001 (|d|=0.766) | 0.699 (|d|=0.066) |
| resid_lag1 | -0.410 | -0.484 (|d|=0.074) | -0.434 (|d|=0.024) | -0.426 (|d|=0.016) | -0.490 (|d|=0.080) | -0.461 (|d|=0.051) | -0.533 (|d|=0.123) |

[source: results.md lines 17–25]

#### Dense-Trajectory PSD Comparison (Log-Spectral RMSE vs Real, nats; Lower=Better)

| Method | log-spectral RMSE |
|---|---|
| v1 (quadratic only) | 3.982 |
| fourier | 0.234 |

[source: results.md lines 28–31]

#### Dense-Trace Spectral Analysis (Train Split; See runs/spectral/stats.json)

- **Power-law fit**: α = 1.06 (R² = 0.99) across nearly 2 decades of frequency, 5099 lines [source: results.md line 39]
- **No word-period peak**: Power in the 4–6U (word+gap) band is 0.2 dB *below* power-law prediction; words do not imprint a periodic signature [source: results.md lines 40–43]
- **Variance partition**: ~34% at "global" scales (wavelength > 8U) vs ~66% at "local" (sub-8U) scales [source: results.md line 48]
- **Between-page ICC of per-line log-PSD**: 0.16 at lowest band → 0.32 at highest band (writer identity stronger in local wobble) [source: results.md lines 49–53]
- **Gap-sequence autocorrelation** (lags 1–5 Spearman): 0.49, 0.44, 0.40, 0.37, 0.30 (smooth decay, not AR(1) collapse) [source: results.md lines 54–56]

### Fairness Controls

**No explicit text-matching control in Fourier** (unlike FFT2D's fair.py protocol), but:
- Evaluation uses the same word-level harness as v1 (scribe_layout/evaluate.py's generate() function)
- Dense-trace analysis is independent, comparing learned spectral properties
- Dependence structure controls (gap_lag1, gap_vs_width, etc.) reveal whether learned dependencies match real

### Stated Corrections / Caveats

From results.md lines 111–125 ("Honest doubts"):

1. **Lomb-Scargle units caveat**: (α, level) PSD fit is in scipy's raw Lomb-Scargle units, not true variance units. Single `level_shift` calibration constant (chosen on val, **level_shift = -1.0**) [source: fourier.json line 2048] is doing real work—different dense-trace extraction (window size, run-length threshold) would shift it.

2. **Dense-trace recovery uncertainty**: Bbox-IoU match between geometry piece and detection-word polygon is approximate after extract.py's split/merge/remap steps; mismatches silently drop that word's samples rather than biasing them, but not quantified.

3. **Gap copula Toeplitz covariance**: Built from only 5 empirical lag correlations (0.486, 0.444, 0.407, 0.381, 0.329 [source: fourier.json lines 2038–2043]) plus geometric tail extrapolation (tail_ratio=0.864 [source: fourier.json line 2045])—adequate for 2–9 word lines but not validated beyond that range.

4. **Hyperparameter selection**: 6×4×4 grid on val against W1(gap+slope+curv+resid_scale); finer or differently weighted grid could move numbers slightly, though not the qualitative picture. **Hyperparams selected**: level_shift=-1.0, slope_scale=0.85, curv_scale=0.6 [source: fourier.json lines 2048–2051].

### Conclusion (as Stated in Source)

From results.md lines 58–109 ("Verdict"):

**The spectral hypothesis for the DENSE trace is correct and the harness v1 cannot see it.**

- V1's generated trajectory is piecewise-quadratic with iid noise only at word anchors → essentially zero power above per-line quadratic frequency → dense-PSD log-spectral RMSE v1 **3.98 nats vs Fourier 0.23 nats** (17× better) [source: results.md lines 64–71, visible in dense_psd.png]
- Any downstream use of generated layout that looks at ink *between* word anchors (rendering fine baseline wobble, training detector on synthetic pages) would see this difference immediately; word-level harness does not.

**On word-level marginals the two models trade wins, v1 remains better where already strong.**

- V1 wins: gap (0.038 vs 0.092 W1) [source: results.md line 73], pitch_next (0.092 vs 0.106) [source: results.md line 73]
- They tie on slope and curv
- Fourier wins: resid_scale (0.036 vs 0.042), drift (0.151 vs 0.158), **x_start by wide margin** (0.206 vs 1.314—v1's learned x_start head is its weakest piece; plain empirical marginal beats it) [source: results.md lines 73–77]
- Fourier's icc_gap_mean (0.312) matches real (0.311) almost exactly; v1 overshoots (0.567)—closed-form hierarchical calibration does what it was designed to do [source: results.md lines 78–79]

**Fourier has exact weak points matching its design:**
- Cannot reproduce gap_vs_width (0.128 real vs -0.01) or gap_vs_pos (-0.12 real vs 0.00)—gap copula models only lag correlation + page level, never word width/position; v1's autoregressive gap head conditions on these, gets them essentially right [source: results.md lines 81–84]
- Additive page-level gap offset over-inflates gap_lag1 (0.59 vs real 0.47; v1: 0.49)—design side-effect worth fixing before reuse [source: results.md lines 87–89]

**C2ST is near-wash at line level, v1 wins at page level.**
- Fourier_full line-level logistic C2ST (0.588) indistinguishable from v1 (0.619); MLP slightly favours v1 (0.627 vs 0.663) [source: results.md lines 91–93]
- At page level, gap/x_start weaknesses compound across many lines → MLP picks fourier_full out easily (0.840 vs v1 0.703); real-vs-real noise floor ~0.55–0.63, so v1 closer to floor, Fourier not [source: results.md lines 93–97]

**Most interesting single result: `fourier_vertical_v1_spacing` (v1's spacing + Fourier's vertical trajectory)**
- **BEST line-level C2ST of every method tested: 0.580** (even below pure v1), keeps v1's good gap/x_start/pitch numbers, improves resid_scale over v1 (0.035 vs 0.042) and resid_lag1 (|d|=0.016, best of all methods) [source: results.md lines 99–109]
- Replacing only v1's per-word-iid vertical noise with spectrally-correct GP while keeping v1's learned spacing: straightforward combined win on every metric tracked by word-level harness, in addition to fixing the dense-PSD gap [source: results.md lines 99–109]
- **"If one method had to be picked to replace v1 outright with no further work, this mixed generator (not pure Fourier) is the one the numbers support."** [source: results.md lines 107–109]

---

## Plottable Data

### FFT2D Experiment

**Candidates for figures:**

1. **Real vs method C2ST AUC comparison** (controlled test)
   - Path: `/home/hpc4090/shyan/ocr/scribe_layout_fft2d/runs/` (metrics.json contains all values)
   - Data in results.md: line 31–37 (method vs real test table)
   - Caption: "Controlled text-matched 2D spectral C2ST AUC (logistic regression) separating real TEST pages from generated layouts. v1=0.552, v2_history_unfit=0.661, v2_latent_unfit=0.685 vs real-vs-real noise floor 0.483. Manual baseline 0.992 (omitted for scale). Control ensures differences are layout-only."

2. **Ablation: line pitch sensitivity**
   - Path: metrics.json, "ablations"."per_page_paired"."shuffle_line_pitch"
   - Data in results.md: lines 90–93 (pitch_peak_freq: 0.198 → 0.135, delta -0.063)
   - Caption: "Per-page ablation: shuffling line pitch decreases vertical-axis peak frequency by 0.063 cycles/U, increasing prominence by 92.6 units and FWHM from 0.079 to 0.083. Validates metric's sensitivity to pitch regularity."

3. **Log-PSD L2 distances (uncontrolled)**
   - Path: metrics.json, "comparisons" (real_train, manual, v2, v2_ctx, v1, v1_ctx, v2_old, manual_plan)
   - Data in results.md: lines 63–74 (uncontrolled table)
   - Caption: "L2 distance between real TEST and each method's mean log-PSD curve (uncontrolled, confounded by text/glyph). v2/v1 methods ~0.11–0.15 (high spectral similarity to real), manual 0.282, manual_plan 0.830 (extreme). Real-train baseline 0.017."

### Fourier Experiment

**Candidates for figures:**

1. **Dense-PSD log-spectral RMSE comparison**
   - Path: `/home/hpc4090/shyan/ocr/scribe_layout_fourier/runs/dense_psd.png` (referenced in results.md line 33)
   - Data: results.md lines 28–31 (table: v1 3.982 nats, fourier 0.234 nats)
   - Caption: "Dense-trace power-spectral-density log-spectral RMSE (nats, Lomb-Scargle units) vs real. Fourier model captures 1/f spectral structure (α=1.06), v1 quadratic trajectory cannot. 17× improvement. Fine baseline wobble between word anchors—invisible to word-level harness—is well-recovered only by Fourier."

2. **Word-level marginal W1 ablation: gap, slope, x_start**
   - Path: metrics.json, methods all, marginals sections
   - Data in results.md: lines 7–14 (word-level ablation table, columns gap_w1, slope_w1, x_start_w1)
   - Caption: "Word-level Wasserstein distance (W1) on marginals: learned_full (v1) vs fourier_full. v1 better on gap (0.038 vs 0.092) and pitch; Fourier much better on x_start (0.206 vs 1.314). fourier_vertical_v1_spacing (hybrid) achieves 0.043 gap, 1.328 x_start, 0.580 line-level C2ST (best overall)."

3. **ICC (intra-class correlation) of gap mean**
   - Path: metrics.json, methods, dependence sections
   - Data in results.md lines 17–25 (dependence table, icc_gap_mean row: real 0.311, learned_full 0.567, fourier_full 0.312)
   - Caption: "Between-page ICC of per-line gap mean: real 0.311, fourier_full 0.312 (exact match), learned_full 0.567 (overstated writer identity). Fourier's closed-form hierarchical calibration for page-level habits works as designed."

4. **Dense-trace power-law fit (α vs frequency)**
   - Path: `/home/hpc4090/shyan/ocr/scribe_layout_fourier/runs/spectral.png` or `spectral/stats.json` (referenced in results.md line 35)
   - Data: results.md line 39 (α = 1.06, R² = 0.99, nearly 2 decades of frequency, 5099 lines)
   - Caption: "Dense headline traces pooled across train split (5099 lines): power-law fit log P = log C − α log f gives α=1.06 (R²=0.99). 1/f (pink) noise characteristic of cursive handwriting baseline trajectory. No word-period peak (0.2 dB below prediction)."

5. **Gap autocorrelation (lag 1–5)**
   - Path: metrics.json, dependence sections (gap_lag1 under learned_full, fourier_full, etc.)
   - Data in results.md lines 54–56 (gap-sequence lag-k Spearman: 0.49, 0.44, 0.40, 0.37, 0.30)
   - Caption: "Gap sequence autocorrelation (Spearman, word-level): real lags 1–5 decay smoothly (0.49→0.30), not like AR(1) pole. Fourier copula Toeplitz model captures slow decay; v1 iid + page offset cannot."

6. **Page-level C2ST comparison (logistic regression)**
   - Path: metrics.json, methods all, c2st_page
   - Data in results.md lines 7–14 (c2st_page column: learned_full 0.7361, fourier_full 0.7330, fourier_vertical_v1_spacing 0.7313, v1_vertical_fourier_spacing 0.7493; shuffled_real floor 0.6272)
   - Caption: "Page-level logistic C2ST AUC (5-fold, 5 seeds). fourier_vertical_v1_spacing best at distinguishing real from synthetic (not shown: marginally worse than v1 on pure page-level, but paired marginal metrics show consistent advantage). Real-vs-real noise floor ~0.63."

