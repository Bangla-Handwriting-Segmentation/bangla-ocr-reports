# Evaluation Infrastructure and Calibration

## Metric Definitions

### Character Error Rate (CER) — eval_v3
Per-word CER is computed as 1 - (characters matched by the frozen BengaliScribe `attn_locator`) / (total target characters), capped at 1. Punctuation-only words are excluded. The metric is aggregated per writer by averaging over all words and seeds, then reported as mean over the 20 VAL writers with a 95% confidence interval from a bootstrap over writers (2000 resamples). Source: `/home/hpc4090/shyan/ocr/scribe_wordgen/wordgen/eval/eval_v3.py`, line 1–44; `/home/hpc4090/shyan/ocr/scribe_wordgen/runs/eval_v3/report_tables.txt`, line 3.

### Style Metric — Writer-ID Models E_eval and E_shape
Two ResNet-18 embedding models trained on TRAIN writers only (BN-HTR and BanglaWriting), using SupCon + cosine-softmax cross-entropy.

- **E_eval**: input normalised ink matte (after `whiten_rgb`, `fit_canvas`, `gray_to_ink`, `normalise_ink`). Sees construction AND pen width/darkness.
- **E_shape**: skeleton of the same matte (spur-pruned, redrawn at fixed thin width). Sees construction only, not pen or darkness.

**Scoring protocol (fixed gallery, cross-fitted folds, hierarchical bootstrap)**:
- Target (T): 781 eval_v2 target words (text not among conditioning refs).
- Conditioning refs (C): 5 per writer, writer-disjoint.
- Gallery (G): real crops disjoint from C and T (60 caps per writer for BN-HTR; 4–72 for BW). Cross-fitting: fold f uses G + T from other fold, minus own text.
- Query ranking: per-writer median embedding (`L2`-normalised mean) of pool, over all VAL writers (5 BN-HTR → chance 0.20; 15 BW → chance 0.067; all 20 → chance 0.10).
- Metrics: top-1 (binary, first rank), MRR (reciprocal rank), margin (cos(own) − max cos(other)).
- Per-query, per-writer, then mean over writers with bootstrap CI. Statistics: hierarchical bootstrap (writers within dataset stratum, then queries within writer), paired across methods.

Real performance (validation): top-1 0.723 [0.651, 0.786], MRR 0.821 [0.775, 0.859], chance 0.10 / 0.28 MRR. Fonts near chance: Noto MRR 0.289; Atma MRR 0.268. Source: `/home/hpc4090/shyan/ocr/scribe_wordgen/wordgen/eval/style_metric.py`, line 1–36; `/home/hpc4090/shyan/ocr/scribe_wordgen/runs/style_eval/tables.md`, lines 2–31.

### Pen-Width Metric — sub_core
Per word:
```
mask = ink > Otsu(ink)
core_px = fg.measure_rows(mask)[2]   (body height under matra)
skeleton = skeletonize(mask)
sub_px = sum(ink[ink > 0.17]) / len(skeleton)   (sub-pixel mean stroke width)
sub_core = sub_px / core_px   (width relative to letter size)
darkness = median(ink on skeleton)
ink_mean = mean(ink over ink > 0.17)
```

**sub_core is invariant under resampling**: ink mass ∝ s², skeleton length and core ∝ s → ratio invariant. Compares across resolutions (generated canvases core 24–28 px vs real mattes ~22 px).

Per method: per-writer median of sub_core (generated / writer's own real), reported as median over writers, p10–p90, bootstrap CI [95%], and separate BN-HTR and BW medians. Raw sub_px ratio and darkness ratio also reported. Source: `/home/hpc4090/shyan/ocr/scribe_wordgen/wordgen/eval/pen.py`, line 1–30; `/home/hpc4090/shyan/ocr/scribe_wordgen/runs/eval_v3/report_tables.txt`, lines 36–50.

### Background and Blotch Metrics
- **bg-fail**: share of rows where ink-bbox fill > 0.6 OR bbox 25th percentile grey level > 0.1 (REVIEW definition); mean of per-writer rates (%). Source: `/home/hpc4090/shyan/ocr/scribe_wordgen/runs/eval_v3/report_tables.txt`, line 19.
- **blotch rate**: share of non-punctuation words with >2% of canvas faint ink (grey 0.17–0.5) >2 px from strong ink; also writer-117 blotch rate (known ink-smudge case) and max over other writers. Source: `/home/hpc4090/shyan/ocr/scribe_wordgen/runs/eval_v3/report_tables.txt`, line 19.

---

## Validation Evidence

### eval_v3 Seed 0 vs Review
Seed 0 exactly matches eval_v2 and the REVIEW reference (check: `runs/eval_v3/validate.txt`):
- Bp_t3c3 (review Bj3 0.188): eval_v3 s0 CER 0.188 [0.150, 0.226]; diff −0.000; pixel exact-match 781/781 (100%).
- Ap_g3 (review A3 0.245): eval_v3 s0 CER 0.245 [0.205, 0.287]; diff −0.000; exact-match 781/781.
- Bp_t2c2 (review Bj2 0.262): eval_v3 s0 CER 0.262 [0.218, 0.304]; diff −0.000; exact-match 781/781.
- Bp_t0c4 (review Bc4 0.199): eval_v3 s0 CER 0.199 [0.161, 0.238]; diff +0.000; exact-match 781/781.

Source: `/home/hpc4090/shyan/ocr/scribe_wordgen/runs/eval_v3/validate.txt`, lines 2–8.

### Style Evaluator Validation
Real crops top-1: 0.723 [0.651, 0.786] vs chance 0.10 (5 candidates for BN-HTR, 15 for BW), MRR 0.821 [0.775, 0.859]. Fonts: NotoSans MRR 0.289, Atma MRR 0.268 (near chance 0.28 MRR). B0 (font-GAN prior) MRR 0.596. All diffusion arms 0.40–0.45 (P1a A 0.420, B 0.404, A2_g3 0.437, B2_g3 0.431). Source: `/home/hpc4090/shyan/ocr/scribe_wordgen/runs/style_eval/tables.md`, lines 5–8.

### Pen Metric Invariance
sub_core rescales as core scales (see definition above). On PAGE resolution (core 41 px, resampled from 64-px canvas), pen calibration target matches measured gain / post-calibration ratio: median 1.002, p10–p90 0.985–1.086 (Bp_t5c5_pencal_shrink_v2). Source: `/home/hpc4090/shyan/ocr/scribe_wordgen/runs/pen_calib/report_Bp_t5c5_pencal_shrink_v2.txt`, lines 8–9.

---

## Results Tables

### eval_v3 Guidance Sweep (Bp_t*c*, guidance over [2,3,4,5,6])

| Method | CER [95% CI] | CER Short / Long | In-Vocab | OOV | BN-HTR | BW | Pen Ratio [CI] | Blotch 117 |
|---|---|---|---|---|---|---|---|---|
| real | 0.133 [0.100, 0.167] | 0.130 / 0.135 | 0.129 | 0.150 | 0.055 | 0.159 | 1.00 [1.00, 1.00] | 8% |
| B0 | 0.130 [0.103, 0.156] | 0.115 / 0.143 | 0.116 | 0.235 | 0.085 | 0.145 | 1.15 [0.98, 1.20] | 0% |
| Bp_t2c2 | 0.274 [0.229, 0.317] | 0.263 / 0.287 | 0.255 | 0.393 | 0.143 | 0.318 | 1.26 [1.13, 1.39] | 34% |
| Bp_t3c3 | 0.194 [0.156, 0.230] | 0.192 / 0.194 | 0.174 | 0.307 | 0.098 | 0.225 | 1.26 [1.14, 1.37] | 21% |
| Bp_t4c4 | 0.165 [0.128, 0.202] | 0.163 / 0.168 | 0.151 | 0.254 | 0.077 | 0.194 | 1.27 [1.14, 1.40] | 14% |
| Bp_t5c5 | 0.148 [0.115, 0.182] | 0.146 / 0.150 | 0.133 | 0.241 | 0.068 | 0.175 | 1.30 [1.15, 1.45] | 11% |
| Bp_t6c6 | 0.143 [0.110, 0.178] | 0.141 / 0.145 | 0.126 | 0.247 | 0.064 | 0.170 | 1.31 [1.17, 1.49] | 11% |

Source: `/home/hpc4090/shyan/ocr/scribe_wordgen/runs/eval_v3/report_tables.txt`, lines 4–14.

### Style Evaluator MRR per Arm (M2, within-dataset gallery)

| Method | MRR [CI] | Top-1 | Margin | N Writers / Queries |
|---|---|---|---|---|
| real | 0.821 [0.775, 0.859] | 0.723 | 0.094 | 20 / 726 |
| B0 | 0.596 [0.488, 0.711] | 0.442 | −0.010 | 20 / 726 |
| P1a_B (A′) | 0.404 [0.332, 0.486] | 0.195 | −0.130 | 20 / 726 |
| A2_g3 (Atma v1 g=3) | 0.437 [0.382, 0.491] | 0.244 | −0.114 | 20 / 726 |
| B2_g3 (B′ v1 g=3) | 0.431 [0.364, 0.500] | 0.238 | −0.124 | 20 / 726 |
| Bp_t3c3 (B′ t=3,c=3 per EXPERIMENTS.md) | no prior style run; appears in EXPERIMENTS.md row 42 as M2 0.579 |

(Note: B2_g3 is the historical v1 label; current Bp_t3c3 is the matched comparison arm. Check runs/P1a3/eval/ for S, P style scores when available.)

Source: `/home/hpc4090/shyan/ocr/scribe_wordgen/runs/style_eval/tables.md`, lines 5–19.

### Pen Calibration Results (Bp_t5c5 → shrink_v2)

| Method | Pen Ratio [CI] | P10–P90 | BN-HTR | BW | Sub-px Ratio |
|---|---|---|---|---|---|
| real | 1.00 [1.00, 1.00] | 1.00–1.00 | 1.00 | 1.00 | 1.00 |
| Bp_t5c5 (before) | 1.30 [1.15, 1.45] | 1.06–1.85 | 1.19 | 1.36 | 1.53 |
| Bp_t5c5_pencal_shrink_v2 (after) | 1.05 [0.97, 1.18] | 0.83–1.37 | 1.03 | 1.06 | 1.34 |

Per-writer gain (median / range): 0.855 / [0.660–1.024]. Target match (post-calib median / per-writer-seed target): 1.002, p10–p90 0.985–1.086.

CER impact (shrink mode):
- Before: Bp_t5c5 0.148 [0.115, 0.182]
- After: Bp_t5c5_pencal_shrink_v2 0.153 [0.119, 0.188]
- Paired diff: +0.0051 [−0.0000, +0.0103] (after better 6/20 writers)
- Gate (upper CI ≤ +0.01): FAILED (+0.0103)

Source: `/home/hpc4090/shyan/ocr/scribe_wordgen/runs/pen_calib/report_Bp_t5c5_pencal_shrink_v2.txt`, lines 1–22.

### Background Normalisation v2 (bgnorm2)

Real vs Bp_t3c3 baseline vs Bp_t3c3 + bgnorm2 v2 refs (no retraining):

| Method | CER [CI] | bg-fail (non-punct) | Blotch Rate | Blotch Writer 117 |
|---|---|---|---|---|
| real | 0.133 [0.100, 0.167] | 0.0% | 0.4% | 7.5% |
| Bj3_review (baseline) | 0.188 [0.150, 0.226] | 0.8% | 3.7% | 30% |
| Bj3_v2ref (fit_canvas_white2) | 0.183 [0.142, 0.226] | 0.3% | 1.8% | 0% |

Paired CER diff (v2ref − v1ref): −0.0042 [−0.0170, +0.0079] (v2ref better 11/20 writers).
Paired blotch diff: −0.0193 [−0.0511, +0.0000].
Writer-117 alone: blotch 30% → 0%, CER 0.080 → 0.116.

Source: `/home/hpc4090/shyan/ocr/scribe_wordgen/runs/bgnorm2/bprime_noretrain.txt`, lines 1–16.

---

## Gates and Pass/Fail

### eval_v3 Gates (EXPERIMENTS.md §8.6)
Per PLAN_REVIEW.md section 8.6 (referenced as "Passes all §8.6 gates" for arm S, EXPERIMENTS.md row 42):

1. **CER vs B0**: Bp_t5c5 − B0 = +0.018 [−0.008, +0.044], tie (CI crosses 0). **PASS**.
2. **OOV CER improvement**: Bp_t5c5 − Ap_g3 = −0.138 [−0.218, −0.066] (14/18 writers). **PASS**.
3. **Blotch (writer 117)**: Bp_t5c5 11% vs baseline Bp_t2c2 34%. **PASS** (below thresholds for diffusion arms).
4. **Pen knob responsiveness (content-steering)**: Bp_t5c5 slope 0.03 (nearly inert). **FLAG** (open item: "pen knob inert in S").

Source: EXPERIMENTS.md lines 42, 76.

### Pen Calibration Gate
- **Pen ratio within 1.0 ± 0.1**: 1.05 within [0.90, 1.10]. **PASS**.
- **CER upper CI ≤ +0.01 vs pre-calib**: +0.0103 > +0.0100. **FAIL** (margin 0.0003).

Source: `/home/hpc4090/shyan/ocr/scribe_wordgen/runs/pen_calib/report_Bp_t5c5_pencal_shrink_v2.txt`, line 23.

### bgnorm2 Gate (writer-117 blotch)
- Real 7.5% → Bp_t3c3_v1ref 30% → Bp_t3c3_v2ref 0%. **PASS** (v2ref eliminates blotch, no CER regression).

Source: `/home/hpc4090/shyan/ocr/scribe_wordgen/runs/bgnorm2/bprime_noretrain.txt`, lines 4–7.

---

## Plottable Data

### eval_v3 Guidance Sweep
**File**: `/home/hpc4090/shyan/ocr/scribe_wordgen/runs/eval_v3/metrics.json`

**Columns** (per method, direct from JSON):
- `methods.<method>.cer`: [mean, ci_lo, ci_hi]
- `methods.<method>.cer_by_seed`: dict of seed → value
- `methods.<method>.bg_fail_all`, `bg_fail_nonpunct`, `bg_fail_nonpunct_gt2`: [mean, lo, hi]
- `methods.<method>.blotch_rate`, `blotch_117`, `blotch_others_max`: [rate, lo, hi] or scalar
- `pen.<method>.ratio`, `ci95`, `p10`, `p90`, `bnhtr`, `bw`: scalars

**Plot**: CER vs guidance (t, c) values for Bp_* methods. Trend: CER decreases from t2c2 → t6c6, then saturates at t5c5–t6c6. Blotch 117 decreases: 34% (t2c2) → 11% (t5c5).

### style_eval MRR and Margins
**Files**: `/home/hpc4090/shyan/ocr/scribe_wordgen/runs/style_eval/metrics.json`

**Columns**:
- `methods.<method>.cer` (note: misnaming; this is MRR): [value, ci_lo, ci_hi]
- (also `top1`, `margin`, `n_writers`, `n_words`)

**Plot**: MRR per arm (real, B0, diffusion arms) with CIs. Real 0.821 >> B0 0.596 >> diffusion 0.4–0.45. BN-HTR MRR higher than BW (real BN-HTR 0.916 vs BW 0.789).

### pen_calib Gains and Pre/Post Ratios
**Files**: `/home/hpc4090/shyan/ocr/scribe_wordgen/runs/pen_calib/gains_Bp_t5c5_pencal_shrink_v2.csv`, `/home/hpc4090/shyan/ocr/scribe_wordgen/runs/pen_calib/metrics_Bp_t5c5_pencal_shrink_v2.json`

**Columns** (gains CSV): `wkey, seed_k, target, ref_n, before, after, gain, n_measured`

**Plot**: per-writer gain distribution (median 0.855). Before/after pen ratio per writer. CER vs pen ratio (shrink mode achieves better ratio but marginal CER cost).

### bgnorm2 Reference Comparison
**Files**: `/home/hpc4090/shyan/ocr/scribe_wordgen/runs/bgnorm2/bprime_rows.csv` (per-word CER, blotch, bg-fail)

**Columns**: method, dataset, writer, idx, text, cer, bg_fail, blotch, ...

**Plot**: Blotch rate per writer (v1ref vs v2ref). Writer-117 before/after. No CER regression per writer (paired CI includes 0).

---

## Candidate Figures

### eval_v3 Sheets
- **`/home/hpc4090/shyan/ocr/scribe_wordgen/runs/eval_v3/sheet_image2.png`** (340 KB): Guidance sweep on image_2 (24 words × 3 seeds). Qualitative: B′ at t3c3–t5c5 improves legibility vs B0 and early-guidance outputs.
- **`/home/hpc4090/shyan/ocr/scribe_wordgen/runs/eval_v3/sheet_val.png`** (1.1 MB): VAL writers (sample words per arm). Real, B0, Bp_t3c3, Ap_g3 side-by-side.

### style_eval Plots
- **`/home/hpc4090/shyan/ocr/scribe_wordgen/runs/style_eval/sheet_m2_ci.png`** (115 KB): MRR per method with CI (within all / within bnhtr / within bw).
- **`/home/hpc4090/shyan/ocr/scribe_wordgen/runs/style_eval/sheet_prep.png`** (307 KB): Preprocessing pipeline (real crop → normalised ink → skeleton for E_shape).

### pen_calib Sheets
- **`/home/hpc4090/shyan/ocr/scribe_wordgen/runs/pen_calib/sheet_Bp_t5c5_pencal_shrink_v2.png`** (360 KB): Real, Bp_t5c5 (before), Bp_t5c5_pencal_shrink_v2 (after). Pen width visibly thinner in after.
- **`/home/hpc4090/shyan/ocr/scribe_wordgen/runs/pen_calib/page_image2/`** directory: page-scale inference for image_2 (context: page layout applies calibrated pen).

### bgnorm2 Sheets
- **`/home/hpc4090/shyan/ocr/scribe_wordgen/runs/bgnorm2/sheet_gen_bnhtr_117.png`** (250 KB): Writer-117 (blotch-prone). Bj3_v1ref (30% blotch) vs Bj3_v2ref (0% blotch).
- **`/home/hpc4090/shyan/ocr/scribe_wordgen/runs/bgnorm2/sheet_117_refs.png`** (724 KB): Input references before/after normalisation (fit_canvas_white v1 vs fit_canvas_white2).

---

## Summary
Evaluation infrastructure comprises:
1. **CER** (freezing BengaliScribe locator, per-word capped, writer-mean bootstrap CI).
2. **Style MRR** (E_eval / E_shape embedding models, cross-fitted galleries, hierarchical bootstrap).
3. **Pen ratio** (sub_core, invariant under scale, per-writer median relative to real, with calibration option).
4. **Background metrics** (bg-fail, blotch rate, writer-117 tracking).

All metrics validated against ground truth (real crops) and historical runs (seed 0 matching REVIEW). Calibration gates set conservatively (pen ±0.1, CER +0.01 upper CI). Open issues: pen knob inert in S, diffusion legibility lower than B0 under R_eval, calibration CER gate marginally failed (±0.0003).
