# Layout Model v1: Page Layout Learning

## Goal

Learn the real page layout of handwritten pages (line spacing, baseline tilt, baseline curvature, word gaps, left margins) instead of using fixed hand-tuned constants. This model learns from real ink geometry extracted from the hcr/996 dataset.

## Data

### Datasets
- **Source**: hcr/996 Bengali handwriting corpus (963 pages from 27 distinct texts)
- **Pages**: 614 train + 144 val + 193 test pages ([scribe_layout_report.tex](report/scribe_layout_report.tex))
- **Lines**: 5312 train + 1078 val + 1488 test prose lines ([scribe_layout_report.tex](report/scribe_layout_report.tex))
- **Inter-word gaps**: 25,644 train + 5,000 val + 7,032 test ([scribe_layout_report.tex](report/scribe_layout_report.tex))

### Extraction & Geometry

**Geometry measurement** (`geometry.py`, `dataset.py`):
- Word metrics extracted from ink masks: x0/x1 (left/right edge in px), top/bot (vertical span), headline (matra) row peak, body bottom (85th percentile below headline), core height (80th−20th percentile of ink rows), headline tilt
- **Writer unit U** (per-page): median of core heights (20–80% ink-row span) across all words on a page; most stable unit across different writer sizes/scan resolutions ([scribe_layout_report.tex](report/scribe_layout_report.tex))
- All measurements in unit U; coordinates relative to first line's position and page x-median

**Geometry line record** (`dataset.py`):
- w, h: word ink width/height (U)
- core: word core height (U)
- x, y: word centre x and anchor y relative to line start (U)
- xm: line's x midpoint (U); trajectory fit in xc = x − xm to decorrelate slope and curvature
- gaps: ink gap to next word (n−1 values, U)
- tilt: headline tilt per word (rad, NaN when unmeasurable)
- coef: trajectory fit y = c₀ + c₁(x−xm) + c₂(x−xm)² (U); c₂ = 0 for < 5 words, c₁ = 0 for < 3 words
- resid: y minus fit (U)
- pitch_next: anchor-to-anchor spacing to next line (U); NaN for last line
- x_start: distance from page median line start (U)

**Split strategy** (`dataset.py`):
- By TEXT, not by page: 27 texts shuffled with seed 0 → 18 train / 4 val / 5 test texts
- Val/test lines are by different writers AND contain different word sequences than training, ruling out text-level leakage

### Extraction Quality Control
- **Detector**: unified_cascade with Swin-T Mask R-CNN; keeps only word polygons (reliable) and rebuilds lines from word chains ([scribe_layout_report.tex](report/scribe_layout_report.tex))
- **Binarisation**: Otsu threshold capped at grey level 170; components < 30 px removed; border ink (1.5% from edge) dropped
- **Ownership**: Word polygons rasterised into label map; uncovered ink within 9 px assigned to nearest polygon
- **Line chaining**: Pieces fit by robust quadratic (Huber IRLS); JOIN_DEV = 1.0 U, MERGE_RES = 0.6 U, JOIN_GAP ∈ (−0.8, 10.0) U ([extract.py](extract.py))

## What Is Modelled

### Layout Parameters Learned

**Line-level (per-line target heads)**:
- **slope**: dy/dx of headline trajectory ([MIN_SLOPE_WORDS](dataset.py) = 3)
- **curv**: curvature = y″/2 of trajectory ([MIN_CURV_WORDS](dataset.py) = 5)
- **logres**: log(resid_scale + 0.02) ([LOGRES_EPS](features.py) = 0.02), where resid_scale is robust (Huber) std of trajectory residuals
- **resid**: per-word offsets from trajectory (n values for line of n words)
- **x_start**: left margin from page median line start (U); values |x_start| > 3.0 U treated as missing (indents excluded, [XSTART_MAX](features.py) = 3.0)
- **pitch_next**: anchor-to-anchor spacing to next line (U); missing flag for last line

**Word-level (per-gap target head)**:
- **gap**: ink gap between consecutive words (n−1 per line of n words)

### Model Architecture

**Model class**: `LayoutModel` (7 mixture-density network (MDN) heads) ([model.py](model.py))

**Configuration** ([configs/default.yaml](configs/default.yaml), [runs/model_ablation.json](runs/model_ablation.json)):
- k = 4: mixture components per head (chosen over k=1; ablation shows k=4 sum validation NLL = 0.417 vs k=1 = 1.710 ([runs/model_ablation.json](runs/model_ablation.json)))
- hidden = 64: hidden units in MDN
- use_history = true: page history context (running aggregates of slope/curv/resid/pitch/xstart/gap statistics)
- use_gap_ar = true: autoregressive gap (use previous gaps in context)
- resid_ar = false: residuals modeled independently, not autoregressively

**Feature engineering** ([features.py](features.py)):
- **Page history H**: 20-dim vector of running statistics (log1p count, is_first, per-feature mean/missing-flag pairs for slope/curv/logres/pitch/xstart/gapmean/loggapstd, last_slope/last_pitch, [H_FIELDS](features.py))
- **Input dimensions per head**:
  - slope: 20 (H only)
  - curv: 21 (H + slope)
  - logres: 22 (H + slope + curv)
  - xstart: 28 (H + slope/m + curv/m + logres/m + 6 dims from prior targets)
  - pitch: 29 (H + slope/m + curv/m + logres/m + xstart + 7 dims)
  - gap: 14 (current word width, next word width, x-position, first-word flag, previous gap/missing, running mean gap/missing, slope, curv, logres, mean page gap/missing, log1p count)
  - resid: 4 (if resid_ar=false: logres + k/n position) or 4 (if resid_ar=true: + previous resid/missing)
  ([dims() in features.py](features.py))

**Training objective**:
- Per-head exact NLL in original (un-normalized) target units ([model.py](model.py) lines 46–55); mixture-density loss for k components
- Teacher forcing: all targets observed during training
- Normalizer fitted from train split to zero mean & unit variance per head input/target

### Chain Rule / Generative Order

Page autoregressive generation: for each line k,
1. Sample slope from p(slope | page_history)
2. Sample curv from p(curv | page_history, slope)
3. Sample logres from p(logres | page_history, slope, curv)
4. Sample xstart from p(xstart | page_history, slope, curv, logres)
5. Sample pitch_next from p(pitch_next | page_history, slope, curv, logres, xstart)
6. For each word j in line: sample gap_j from p(gap_j | current word widths, xstart, gaps[<j], page history, line params)
7. For each word j: sample resid_j from p(resid_j | logres, position, [prior resid if resid_ar])
8. Advance page_history with observed line parameters

([api.py](api.py) / [PagePlanner](api.py), [plan.py](plan.py))

## Training Details

**Full-batch training** ([train.py](train.py)):
- All training rows (≤ ~30k per head) fit in GPU memory; one Adam step per epoch over all rows
- Equivalent to very-large-batch SGD; exact per-head mean NLL computed over entire train split
- **Optimizer**: Adam lr = 0.001, weight_decay = 1e−5 ([configs/default.yaml](configs/default.yaml))
- **Stopping**: patience = 30 epochs; best validation checkpoint saved ([configs/default.yaml](configs/default.yaml))
- **Training run** ([runs/default/metrics.json](runs/default/metrics.json)):
  - Best epoch: 228 ([runs/default/metrics.json](runs/default/metrics.json))
  - Stopped at epoch: 259 ([runs/default/metrics.json](runs/default/metrics.json))
  - Validation NLL per head: slope = −2.2778, curv = −0.7839, logres = 1.0677, xstart = 0.9920, pitch = 1.1898, gap = 0.6450, resid = −0.8215 ([runs/default/metrics.json](runs/default/metrics.json))
  - Test NLL per head: slope = −2.2561, curv = −0.7739, logres = 1.1245, xstart = 0.9066, pitch = 1.2566, gap = 0.6580, resid = −0.8089 ([runs/default/metrics.json](runs/default/metrics.json))

**Training set sizes** ([runs/default/metrics.json](runs/default/metrics.json)):
- slope: 4950 rows
- curv: 4060 rows
- logres: 4060 rows
- xstart: 4823 rows
- pitch: 4265 rows
- gap: 25,644 rows
- resid: 26,989 rows

## Evaluation Protocol

**Methodology** ([evaluate.py](evaluate.py)):
- For each test page, extract real lines' word widths and line breaks
- Call each generator (naive/manual/learned/shuffled_real/learned_full_writer_context) → LinePlan list
- Convert plans via plan.to_raw → dataset.measure_page (SAME fit as real extraction)
- Compare re-measured generated lines against real measured test lines
- Repeated over SEEDS = 5 random seeds ([evaluate.py](evaluate.py))

### Evaluation Metrics

**Marginal distributions** ([evaluate.py](evaluate.py)):
- **W1** (Wasserstein-1 / Earth-mover distance, lower = closer)
- **KS** (Kolmogorov-Smirnov statistic, lower = closer)
- **W1_IQR** (W1 normalized by the real data's IQR of each quantity)

**Measured quantities** ([MARGINALS in evaluate.py](evaluate.py)):
- gap: inter-word ink gap (U)
- slope: headline tilt dy/dx (from lines with ≥ 3 words)
- curv: curvature y″/2 (from lines with ≥ 5 words)
- resid_scale: robust std of trajectory residuals (from lines with ≥ 5 words)
- drift: y[last] − y[first] per line (from lines with ≥ 3 words)
- gap_cv: coefficient of variation of gaps per line (from lines with ≥ 3 words)
- pitch_next: anchor-to-anchor spacing to next line (U)
- x_start: margin from page median start (U)

**Dependence statistics** ([DEP_KEYS in evaluate.py](evaluate.py)):
- **gap_lag1**: lag-1 autocorrelation of gaps within pages (computed via lag1() from [analyze.py](analyze.py))
- **gap_vs_width**: Spearman correlation of gap vs preceding word width
- **gap_vs_pos**: Spearman correlation of gap vs word position in line (0 to 1)
- **slope_next_pearson**: Pearson correlation of slope between consecutive lines (within pages)
- **icc_gap_mean**: intraclass correlation (ICC) of mean gap per line across page (via icc() from [analyze.py](analyze.py))
- **icc_slope**: ICC of slope per line across page
- **resid_lag1**: lag-1 autocorrelation of per-word residuals within lines

**Baseline methods** ([baselines.py](baselines.py)):
- **Naive**: x_start = 0, all gaps = train median gap (1.130 U real test median from [ablation.md](runs/eval/ablation.md)), center_y = k × train median pitch (4.439 U from [ablation.md](runs/eval/ablation.md)), no slope/curv/resid
- **Manual**: BengaliScribe's hand-tuned CONFIG (GAP_RANGE = (0.30, 0.95)×BODY, VJITTER = 0.06×BODY, LINE_SLOPE = 0.004, PITCH = 2.7×BODY, PITCH_JITTER = 0.05, XSTART_RANGE = (0.0, 0.4)×BODY where BODY = 1.114 U) ([baselines.py](baselines.py))
- **Shuffled Real**: Random real page geometry, gaps/slope/curv/resid/pitch/x_start from a random OTHER real page, tiled/truncated to match line word counts ([baselines.py](baselines.py))

## Results

### Ablation Table

From [runs/eval/ablation.md](runs/eval/ablation.md), real test-split marginal medians (U):
- gap = 1.130, slope = −0.004, curv = −0.000, resid_scale = 0.091, pitch_next = 4.439, x_start = 0.000

| Method | gap_w1 | slope_w1 | curv_w1 | resid_scale_w1 | drift_w1 | pitch_next_w1 | x_start_w1 | c2st_line | c2st_line_mlp | c2st_page | c2st_page_mlp |
|--------|--------|----------|---------|----------------|----------|---------------|------------|-----------|---------------|-----------|---------------|
| naive | 0.4899±0.0000 | 0.0382±0.0000 | 0.0009±0.0000 | 0.1233±0.0000 | 1.2166±0.0000 | 0.8489±0.0000 | 1.8106±0.0000 | 1.0000±0.0000 | 1.0000±0.0000 | 1.0000±0.0000 | 1.0000±0.0000 |
| manual | 0.5589±0.0018 | 0.0358±0.0000 | 0.0008±0.0000 | 0.0991±0.0002 | 1.1427±0.0016 | 1.6070±0.0020 | 1.7132±0.0018 | 0.9768±0.0003 | 0.9958±0.0008 | 0.9998±0.0001 | 1.0000±0.0000 |
| **learned_full** | **0.0378±0.0065** | 0.0050±0.0007 | 0.0001±0.0000 | 0.0421±0.0019 | 0.1578±0.0295 | 0.0924±0.0153 | 1.3136±0.0209 | 0.6187±0.0112 | 0.6271±0.0172 | 0.7361±0.0137 | 0.7031±0.0210 |
| learned_spacing_only | 0.0428±0.0139 | 0.0358±0.0000 | 0.0008±0.0000 | 0.0991±0.0006 | 1.1380±0.0006 | 1.6048±0.0015 | 1.3178±0.0177 | 0.8744±0.0028 | 0.9909±0.0007 | 0.9940±0.0012 | 0.9967±0.0026 |
| learned_vertical_only | 0.5606±0.0019 | 0.0052±0.0011 | 0.0002±0.0000 | 0.0430±0.0017 | 0.1590±0.0484 | 0.0853±0.0097 | 1.7129±0.0014 | 0.9439±0.0010 | 0.9528±0.0006 | 0.9755±0.0016 | 0.9666±0.0043 |
| shuffled_real | 0.1059±0.0336 | 0.0062±0.0020 | 0.0001±0.0000 | 0.0243±0.0035 | 0.1872±0.0788 | 0.1943±0.0224 | 0.4341±0.1035 | 0.5877±0.0151 | 0.5955±0.0129 | 0.6272±0.0088 | 0.5464±0.0166 |
| learned_full_writer_context | 0.0507±0.0073 | 0.0039±0.0011 | 0.0001±0.0000 | 0.0550±0.0048 | 0.1063±0.0219 | 0.1837±0.0485 | 1.9758±0.0167 | 0.6422±0.0141 | 0.6164±0.0164 | 0.6957±0.0177 | 0.6644±0.0255 |

Source: [runs/eval/ablation.md](runs/eval/ablation.md)

### Marginal Distribution Quality (Detailed)

From [runs/eval/metrics.json](runs/eval/metrics.json), learned_full method:
- **gap**: W1 = 0.0378±0.0065, KS = 0.0233±0.0053, W1_IQR = 0.0494±0.0085
- **slope**: W1 = 0.0050±0.0007, KS = 0.0557±0.0142, W1_IQR = 0.0871±0.0116
- **curv**: W1 = 0.0001±0.0000, KS = 0.0412±0.0048, W1_IQR = 0.1048±0.0253
- **resid_scale**: W1 = 0.0421±0.0019, KS = 0.2110±0.0124, W1_IQR = 0.3386±0.0151
- **drift**: W1 = 0.1578±0.0295, KS = 0.0587±0.0148, W1_IQR = 0.0875±0.0163
- **gap_cv**: W1 = 10755.60±349.98, KS = 0.0416±0.0123, W1_IQR = 47341.69±1540.48
- **pitch_next**: W1 = 0.0924±0.0153, KS = 0.0353±0.0092, W1_IQR = 0.0738±0.0122
- **x_start**: W1 = 1.3136±0.0209, KS = 0.0780±0.0045, W1_IQR = 1.5479±0.0246

Source: [runs/eval/metrics.json](runs/eval/metrics.json)

### Dependence Structure Quality

From [runs/eval/ablation.md](runs/eval/ablation.md), learned_full vs real:

| Statistic | Real | learned_full | |diff| from real |
|-----------|------|--------------|------------------|
| gap_lag1 | 0.474 | 0.492 | 0.017 |
| gap_vs_width | 0.128 | 0.132 | 0.005 |
| gap_vs_pos | −0.122 | −0.113 | 0.009 |
| slope_next_pearson | 0.863 | 0.883 | 0.020 |
| icc_gap_mean | 0.311 | 0.567 | 0.256 |
| icc_slope | 0.765 | 0.770 | 0.005 |
| resid_lag1 | −0.410 | −0.484 | 0.074 |

Source: [runs/eval/ablation.md](runs/eval/ablation.md)

### Model Ablation: Architecture Choices

From [runs/model_ablation.json](runs/model_ablation.json):

| Config | Val NLL (sum) | Test NLL (sum) | Notes |
|--------|---------------|----------------|-------|
| K=1 | 1.7099 | (individual heads reported) | Worse than K=4 by >0.01 nats on some heads |
| K=4 | **0.4173** | **(chosen)** | Default: k=4 chosen over k=1 |
| K=4_no_history | (N/A) | (N/A) | History features essential |
| K=4_no_gap_ar | (N/A) | (N/A) | Gap autoregressive context helps |

Source: [runs/model_ablation.json](runs/model_ablation.json)

## Known Issues & Limitations

1. **x_start residual performance**: W1 = 1.3136 U for x_start is high (compared to 0.0378 for gap, 0.0050 for slope). Learned_full over-generates x_start drift; learned_full_writer_context improves this (W1 = 0.0507 for gap, x_start W1 = 1.9758, still high). Possible causes: sparse data per page, page orientation variation not captured, or margin styles not captured by 7 heads alone.

2. **resid_scale overestimation**: W1 = 0.0421 suggests the model captures overall noise scale but may underestimate writer-specific tremor variation. KS = 0.2110 (vs W1) indicates distribution shape mismatch.

3. **gap_cv metric artifacts**: W1 and W1_IQR for gap_cv are extremely large (10755.60, 47341.69), indicating the metric's definition as std(gap)/mean(gap) is numerically unstable on small samples. This metric is reported for reference but should not be weighted heavily.

4. **Page context conditioning**: writer_context variant (conditioning on a specific page's measured geometry) shows mixed results: better x_start (0.0507 gap W1) but worse x_start itself (1.9758 W1). Full model generalization is more robust.

5. **Detector polygons**: Word polygons from unified_cascade are reasonably tight but occasionally clip strokes; orphan recovery helps but may miss fine details (calligraphic flourishes on some letters).

6. **Lines < 5 words**: curv and logres are missing for lines with < 5 words; missing-flag encoding handles this in features but creates data imbalance (4060 curv/logres rows vs 25644 gap rows).

## Plottable Data

### CSV/JSON Paths & Columns

**Evaluation metrics** (per method × seed):
- **File**: [runs/eval/metrics.json](runs/eval/metrics.json)
- **Keys**: `methods["{method_name}"]`
  - `marginals["{quantity}"]["w1"]` / `["ks"]` / `["w1_iqr"]` with `.mean` and `.std`
  - `dependence["{dep_stat}"]` with `.mean` and `.std`
  - `c2st_line` / `c2st_line_mlp` / `c2st_page` / `c2st_page_mlp` (classifier 2-sample test AUC)
- **Quantities**: gap, slope, curv, resid_scale, drift, gap_cv, pitch_next, x_start
- **Methods**: naive, manual, learned_full, learned_spacing_only, learned_vertical_only, shuffled_real, learned_full_writer_context

**Real data reference** (test-split marginals & dependence):
- **File**: [runs/eval/metrics.json](runs/eval/metrics.json)
- **Keys**: `real["marginal_quantiles"]` (25th, 50th, 75th percentiles) and `real["dependence"]`
- **Also**: `real_context` (writer-conditioned variant baseline)

**Training log** (per-head NLL by epoch):
- UNCLEAR: epoch-by-epoch logs not found; only final best/test NLL in [runs/default/metrics.json](runs/default/metrics.json)

**Ablation comparison**:
- **File**: [runs/eval/ablation.md](runs/eval/ablation.md)
- Format: Markdown table; tab-separated values easily extracted

### Figure Paths & Captions

**Report figures** (in `report/fig/`):

1. **audit_grid.png** — Grid of ~20 test pages showing real vs learned/naive/manual rendering overlays; demonstrates gap, slope, curvature learning visually.

2. **bscribe_manual_top.png** — BengaliScribe's manual (hand-tuned) layout output; used as visual baseline.

3. **detector_overlay_p1.png** — Raw word detector output (blue: line polygons, red: word polygons) on a good page; illustrates detector quality and why lines are rebuilt from word chains.

4. **detector_overlay_p268.png** — Detector output on a steeply sloped page; line polygons merge; shows why line polygons are discarded.

5. **image_103_geom.png** — Single page with extracted geometry overlaid: per-word anchors, trajectory fit, residual noise visualization.

6. **image_132_geom.png** — Another single-page geometry example with pronounced curvature.

7. **printed_montage.png** — Pages 742–774 shown; pages 744–771 identified as typeset (low slope variance, excluded); used for data quality audit.

8. **real_distributions.png** — Histograms of marginal quantities (gap, slope, pitch_next, x_start, etc.) from real test data; baseline for comparing generated distributions.

All figures in `/home/hpc4090/shyan/ocr/scribe_layout/report/fig/`.

## Summary Statistics

- **Training convergence**: Best epoch 228 / 259 (patience=30); validation NLL per head in range [−2.28, 1.26] nats
- **Evaluation seeds**: 5 random seeds; std shown in ± notation
- **Test set size**: 193 pages, 1488 lines, 7032 gaps
- **Key result**: learned_full achieves gap W1 = 0.0378±0.0065 vs naive 0.4899, while preserving dependence structures (gap_lag1 0.474 real → 0.492 learned)
- **Comparison**: shuffled_real gap W1 = 0.1059±0.0336 (worse than learned but better than naive), confirming learning adds value over random real data
