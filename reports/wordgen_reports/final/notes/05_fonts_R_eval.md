# GLYPHSCRIBE FONTS AND THE R_eval INDEPENDENT RECOGNISER

Font-pretraining experiment + rescoring of generator arms (BengaliScribe B0, diffusion arms S/P).

## Recogniser Architecture

**CRNN-CTC (4.9M params)** with codepoint-level CTC loss (EXPERIMENTS.md, 2026-10-19). Input: 64×256 (H×W) ink matte, ink high on 0. 

Conv stack (wordgen/recog/model.py):
- Stem: 1 → 32 channels, pool (2,2) → 32×128
- Stage 1: 32 → 64 + ResBlock, pool (2,2) → 16×64  
- Stage 2: 64 → 128 + ResBlock, pool (2,1) → 8×64
- Stage 3: 128 → 256 + ResBlock, pool (2,1) → 4×64
- Collapse: (4,1) kernel → 1×64 feature map

Sequence model: 2-layer BiLSTM (256 hidden), output projection to charset (219 classes + CTC blank = 0). Decoding: greedy CTC collapse + blank removal.

## Input Representation

**64×256 ink-matte** (wordgen/recog/inkrep.py): 
- Source: real crops (JPEG/PNG) or cached generator renders (PNG saved as 255×(1−ink))
- Conversion: 85th percentile = paper level, 5th percentile = floor
- Stretch: (paper − ink) / (paper − floor), clipped to [0,1], threshold 0.18
- Canvas: left-aligned crop, ~10% of B0 renders wider than 256 px → squashed to 256 via LANCZOS
- Word codepoint CER: per-word errors capped at 1, corpus-aggregated (micro-averaged)

## Synthetic Data Pipeline

**Fonts** (runs/font_search/fonts_glyphscribe/; wordgen/recog/synth.py):
- 37 handwriting-style fonts (hw/*.ttf) + 11 Google OFL Bengali fonts (fonts/*/*.ttf)  
- 38 total (1 printed/Nilima excluded)
- Decorative downweighting: {Kobiguru, Fizz Unicod, Shreya Unicode, TB Mainak} at 0.3× frequency
- All Unicode Bengali with GSUB; shaping via fontTools + PIL RAQM

**Vocab** (runs/R_eval/vocab_synth.tsv): 
- 203,372 rows (Bengali words + digits)
- Columns: word, freq, source, weight (freq^0.5 with digit downweighting × 0.05)
- Sampling: weighted by cumulative weights

**Augmentation** (wordgen/recog/synth.py):
- Stroke thickness: variable via opacity jitter
- Slant/rotation: ±6° shear, ±3° rotation (affine + BILINEAR)
- Baseline wave: small sinusoidal offset
- Blur: post-render Gaussian + high-pass texture from backgrounds
- Noise: Gaussian (σ 1–6 px)
- Crop margins: loose + occasional neighbour-glyph fragment
- JPEG recompression
- Horizontal squash/stretch

**Real augmentation** (wordgen/recog/data.py, _augment_real):
- Horizontal shift into empty columns (±12% cap)
- Affine shear/rotation (30% + 30% probability each)
- Thickness via max/avg filter (25% probability)
- Noise: Gaussian (σ 1–6 px, 30% probability)

## Training: 6 Arms, 30k–60k Steps

**Protocol** (runs/R_eval/full_chain.sh): Lock-based single GPU, num_workers=20, batch_size=128 (default), lr=3e-4, warmup=500, eval/save every 1000 steps.

| Step | Arm | Mode | Init | Max Steps | Data | VAL CER | Note |
|---|---|---|---|---|---|---|---|
| 1 | R_font0 | pretrain | — | 60,000 | Synthetic (vocab_synth.tsv) | 0.456 [0.326, 0.534] | Fonts alone don't transfer |
| 2 | R_real | real | — | 30,000 | Real TRAIN (BN-HTR + BW) | 0.037 [0.028, 0.051] | Baseline |
| 3 | R_font2real | finetune | R_font0 | 30,000 | Real TRAIN | **0.033** [0.025, 0.044] | **Selected** |
| 4a | R_real_10 | real | — | 6,000 | 10% of TRAIN writers | 0.178 [0.140, 0.208] | Scarce-data baseline |
| 4b | R_font2real_10 | finetune | R_font0 | 6,000 | 10% of TRAIN writers | 0.115 [0.080, 0.137] | Scarce-data + fonts |
| 5 | R_mix | mix | R_font0 | 30,000 | 25% synthetic + 75% real | 0.037 [0.027, 0.049] | Mixing ≈ real only |

Paired gains (R_font2real − R_real):
- Overall: −0.0042 [−0.0080, −0.0022], significant (runs/R_eval/RESULTS.md, line 137)
- BW: −0.0107 [−0.0167, −0.0037], larger (line 136)
- 10% writers: −0.0634 [−0.0804, −0.0538], substantial (line 140)

## VAL Results: Per-Model CER (Both Datasets, 4,299 Words)

### Model Selection (runs/R_eval/RESULTS.md, lines 5–13)

| Model | n | CER | CER_lo | CER_hi |
|---|---|---|---|---|
| R_font2real | 4,299 | 0.0329 | 0.0245 | 0.0443 |
| R_real | 4,299 | 0.0370 | 0.0281 | 0.0511 |
| R_mix | 4,299 | 0.0370 | 0.0267 | 0.0486 |
| R_font2real_10 | 4,299 | 0.1147 | 0.0797 | 0.1366 |
| R_real_10 | 4,299 | 0.1781 | 0.1404 | 0.2082 |
| R_font0 | 4,299 | 0.4563 | 0.3265 | 0.5338 |

Source: runs/R_eval/RESULTS.md, lines 6–13 (via runs/R_eval/selection.json, lines 3–45).

### Subset Breakdown: R_font2real

All words (lines 56–72):

| Dataset | Subset | n | CER | CER_lo | CER_hi |
|---|---|---|---|---|---|
| BN-HTR | all | 3,253 | 0.0283 | 0.0178 | 0.0350 |
| BN-HTR | oov | 313 | 0.0642 | 0.0407 | 0.0813 |
| BW | all | 1,046 | 0.0507 | 0.0372 | 0.0647 |
| BW | oov | 143 | 0.0673 | 0.0466 | 0.0876 |
| overall | all | 4,299 | 0.0329 | 0.0245 | 0.0443 |

Words with ≥3 codepoints (lines 56–72):

| Dataset | n | CER | CER_lo | CER_hi |
|---|---|---|---|---|
| BN-HTR | 2,940 | 0.0225 | 0.0137 | 0.0278 |
| BW | 903 | 0.0463 | 0.0330 | 0.0613 |
| overall | 3,843 | 0.0270 | 0.0201 | 0.0378 |

Source: runs/R_eval/RESULTS.md, lines 55–72.

## TEST: R_real and R_font2real (Evaluated Once)

Overall CER (7,665 words):

| Model | n | CER | CER_lo | CER_hi |
|---|---|---|---|---|
| R_real | 7,665 | 0.0429 | 0.0293 | 0.0606 |
| R_font2real | 7,665 | 0.0342 | 0.0228 | 0.0492 |
| Gain (font2real − real) | | −0.0087 [−0.0172, −0.0024] | | |

Paired gain at oov_not_synth (clean OOV, 237 words): 0.0963 [0.0646, 0.1299] (R_real) → 0.0854 [0.0556, 0.1129] (R_font2real), diff −0.0109 [−0.0250, +0.0045] (line 180 vs 162).

Source: runs/R_eval/RESULTS.md, lines 142–180.

## Generator-Arm Rescoring: R_font2real vs R_real

**Note**: Cached PNGs are ink mattes (255×(1−ink)), not scans (EXPERIMENTS.md, Known pitfalls). They use legacy img_to_ink (no min_span floor), so short words (< 3 codepoints) are inflated (S: 0.089 all vs 0.055 ≥3cp). Paired merges on (dataset, writer, idx, seed_k) for S/P; B0/real on (dataset, writer, idx).

### Scored with R_real

All words (runs/R_eval/RESULTS.md, lines 194–202):

| Method | n | CER | CER_lo | CER_hi |
|---|---|---|---|---|
| B0 | 781 | 0.0446 | 0.0364 | 0.0536 |
| real | 781 | 0.0515 | 0.0389 | 0.0649 |
| P_t5c5 | 1,562 | 0.1028 | 0.0742 | 0.1357 |
| S_t5c5 | 1,562 | 0.1169 | 0.0815 | 0.1573 |
| Bp_t5c5 | 1,562 | 0.1450 | 0.1137 | 0.1796 |

≥3 codepoints (lines 204–212):

| Method | n | CER | CER_lo | CER_hi |
|---|---|---|---|---|
| B0 | 666 | 0.0396 | 0.0312 | 0.0484 |
| real | 666 | 0.0504 | 0.0376 | 0.0639 |
| P_t5c5 | 1,332 | 0.0536 | 0.0366 | 0.0737 |
| S_t5c5 | 1,332 | 0.0548 | 0.0380 | 0.0750 |
| Bp_t5c5 | 1,332 | 0.1016 | 0.0801 | 0.1265 |

### Scored with R_font2real (Selected)

All words (lines 235–243):

| Method | n | CER | CER_lo | CER_hi |
|---|---|---|---|---|
| B0 | 781 | 0.0302 | 0.0248 | 0.0358 |
| real | 781 | 0.0438 | 0.0310 | 0.0586 |
| P_t5c5 | 1,562 | 0.0905 | 0.0664 | 0.1179 |
| S_t5c5 | 1,562 | 0.0889 | 0.0652 | 0.1153 |
| Bp_t5c5 | 1,562 | 0.1395 | 0.1117 | 0.1713 |

≥3 codepoints (lines 245–253):

| Method | n | CER | CER_lo | CER_hi |
|---|---|---|---|---|
| B0 | 666 | 0.0226 | 0.0162 | 0.0287 |
| real | 666 | 0.0424 | 0.0294 | 0.0568 |
| P_t5c5 | 1,332 | 0.0577 | 0.0402 | 0.0786 |
| S_t5c5 | 1,332 | 0.0552 | 0.0381 | 0.0753 |
| Bp_t5c5 | 1,332 | 0.1021 | 0.0810 | 0.1264 |

### Key Paired Diffs (R_font2real)

All words (lines 255–263):

| a | b | n | Merge | Diff_CER | Diff_lo | Diff_hi |
|---|---|---|---|---|---|---|
| S_t5c5 | B0 | 1,562 | dataset+writer+idx | 0.0587 | 0.0361 | 0.0833 |
| S_t5c5 | P_t5c5 | 1,562 | +seed_k | −0.0016 | −0.0111 | 0.0076 |
| P_t5c5 | B0 | 1,562 | dataset+writer+idx | 0.0603 | 0.0375 | 0.0860 |
| B0 | real | 781 | dataset+writer+idx | −0.0136 | −0.0268 | −0.0021 |
| S_t3c3 | B0 | 1,562 | dataset+writer+idx | 0.0782 | 0.0519 | 0.1081 |

≥3 codepoints (lines 265–273):

| a | b | n | Diff_CER | Diff_lo | Diff_hi |
|---|---|---|---|---|---|
| S_t5c5 | B0 | 1,332 | 0.0326 | 0.0145 | 0.0523 |
| S_t5c5 | P_t5c5 | 1,332 | −0.0025 | −0.0099 | 0.0050 |
| P_t5c5 | B0 | 1,332 | 0.0351 | 0.0174 | 0.0567 |
| B0 | real | 666 | −0.0198 | −0.0347 | −0.0067 |
| S_t3c3 | B0 | 1,332 | 0.0453 | 0.0262 | 0.0658 |

Interpretation (EXPERIMENTS.md, lines 30–34): B0 scores better than real crops (+0.005 codepoint CER all-word). S and P are ~1.3–1.5× real's R_eval CER on ≥3cp. Legibility favours font-like renders; CER is a floor check, not fidelity. S − P is a tie (−0.0016 [−0.0111, 0.0076]); S is less legible than B0 by ~0.033 codepoint CER on ≥3cp, but the short-word img_to_ink artifact inflates the all-word gap.

Source: runs/R_eval/RESULTS.md, lines 131–273; runs/R_eval/rescore.csv.

## Bugs Fixed by Opus Review (EXPERIMENTS.md, 2026-10-19)

Three blockers in rescore.py and data.py, fixed before launch:

1. **Wrong word pairing in rescoring**: Fixed merge key logic to pair (dataset, writer, idx, seed_k) correctly for diffusion arms vs (dataset, writer, idx) for B0/real (seeds_differ_across_arms pitfall, line 62).

2. **Double crop_ink on cached mattes**: Cached generator-arm PNGs are already ink mattes (255×(1−ink), e.g., runs/eval_v3/cache/*); calling crop_ink again re-stretched and inflated ink ~18% on real crops, biasing R_real CER downward. Fixed: inkrep.ink_from_matte_png loads without re-stretching (line 61).

3. **Augmentation cutting words**: _augment_real's symmetric horizontal roll (±12%) deleted word edges on ~22% of samples, introducing label noise. Fixed: roll only into EMPTY columns detected per-sample (data.py, ~line 150).

Source: runs/R_eval/full_chain.log (line: "Opus review fixed 3 blockers before launch").

## GlyphScribe Font Scores

**37 handwriting + 11 Google fonts**, scored vs writer image_2 (12 words, runs/font_search/glyphscribe/scores.csv). Metrics: chamfer distance (skeleton), pen width/core ratio, corner density, slant, matra thickness/continuity.

### Top 10 by Shape (Skeleton Chamfer)

| Rank | Font | Category | Chamfer_mean | Chamfer_lo95 | Chamfer_hi95 | Gap_vs_real |
|---|---|---|---|---|---|---|
| 1 | Atma-Medium | handwriting | 0.1616 | 0.1399 | 0.1841 | — |
| 2 | Atma-Regular | handwriting | 0.1643 | 0.1444 | 0.1854 | — |
| 3 | Atma-Light | handwriting | 0.1644 | 0.1441 | 0.1854 | — |
| 4 | Junudullah Zami | handwriting | 0.1662 | 0.1366 | 0.2002 | — |
| 5 | HindSiliguri-Medium | handwriting | 0.1678 | 0.1384 | 0.1975 | — |
| 6 | Tagore | handwriting | 0.1689 | 0.1465 | 0.1915 | — |
| 7 | HindSiliguri-Regular | handwriting | 0.1694 | 0.1422 | 0.1990 | — |
| 8 | BenSenHandwriting | handwriting | 0.1703 | 0.1494 | 0.1921 | — |
| 9 | HindSiliguri-Light | handwriting | 0.1705 | 0.1437 | 0.1990 | — |
| 10 | Qayyum Smriti Unicode | handwriting | 0.1714 | 0.1483 | 0.1963 | — |
| — | NotoSans (baseline) | — | 0.1972 | 0.1586 | 0.2441 | +0.0256 [+0.0186, +0.0400] |

All CIs overlap (12 words); top ~10 are statistically indistinguishable.

### Pen Width / Core Ratio

Writer image_2 real: width/core = 0.075. Thin fonts match better:

| Font | Width_over_core | Gap_vs_real |
|---|---|---|
| Qayyum Thin | 0.068 | −0.007 |
| Junudullah Zami / Tagore | ~0.094 | +0.019 |
| Google fonts (avg) | 0.11–0.23 | +0.035–+0.155 |
| NotoSans (baseline) | 0.177 | +0.102 |

Source: runs/font_search/glyphscribe/scores.csv, lines 2–40 (chamfer), width_over_core_*, width_over_core_gap_vs_real columns.

### Matra Continuity

No font replicates the writer's broken matra (EXPERIMENTS.md, line 41). Continuity scores near-chance; theme fonts (Kobiguru 0.820, Fizz 0.756) are highest; Shreya 0.763, Bolokal 0.917.

### GlyphScribe Metadata

- Repository: github.com/tahsinchoudhury/GlyphScribe (public)
- Licence: Code only; font licences unknown → internal research only
- Unicode coverage: All 38 have correct Bengali GSUB shaping (verified via fontTools + PIL RAQM)
- Backgrounds: 4 paper textures (fonts_glyphscribe/backgrounds/)
- Total codepoints (union): ~95% of vocab_synth.tsv

Source: runs/font_search/fonts_glyphscribe/PROVENANCE.txt, SOURCE_COMMIT.txt, LICENSE; EXPERIMENTS.md line 41.

## Plottable Data

### Learning Curves

**Files**: runs/R_eval/{R_font0, R_real, R_font2real, R_real_10, R_font2real_10, R_mix}/train.log

**Line format (train loss)**:
```
[HH:MM:SS] step N/MAX loss=X.XXXX lr=Y.YYe-ZZ THROUGHPUT img/s
```
Extract: regex `step (\d+)/\d+ loss=([\d.]+)` for (step, loss).

**Line format (val eval)**:
```
[HH:MM:SS]   [eval@STEP] VAL(sample n=SAMPLE_SIZE) CER=X.XXXX WER=X.XXXX
```
Extract: regex `eval@(\d+).*CER=([\d.]+)` for (eval_step, cer).

**Cadence**: Training every 50 steps (log_every=50); eval every 1,000 steps (eval_every=1000). One epoch per arm ~1–2 min.

### Results CSVs

**Files**: runs/R_eval/results_R_*_{val,test}.csv

**Columns**: text, pred, writer, dataset, row_idx, oov, in_synth (per runs/R_eval/full_chain.sh, lines 51–70).

CER per row is computed post-hoc via runs/R_eval/make_results_md.py; aggregate CER via runs/R_eval/metrics.py (bootstrap_cer_ci, corpus CER).

### Rescore CSV

**File**: runs/R_eval/rescore.csv

**Columns**: method, dataset, writer, idx, text, seed_k, seed, g_text, g_content, path, source_csv, g, knob, pred_R_real, cer_R_real, pred_R_eval, cer_R_eval

**Aggregation**: Per-row CER computed via edit_distance(pred, text); aggregated via bootstrap_cer_ci per writer. Paired diffs via bootstrap_paired_diff_ci (per-writer resample, 2000 iterations).

Source: runs/R_eval/rescore.py, lines 47–120; runs/R_eval/metrics.py, lines 50–100.

## Candidate Figures

### Synthetics Sheet
- **Path**: runs/R_eval/synth_sheet.png (239 KB)
- **Content**: Grid of synthetic word renders from 48 fonts (vocab_synth.tsv sample), showcasing diversity in skeleton, thickness, slant, noise, and background texture.
- **Caption**: Synthetic handwriting-style Bengali words used in R_font0 pretraining (60k steps). Diversity from 37 GlyphScribe handwriting fonts + 11 Google OFL fonts, with variable stroke thickness, slant, rotation, and paper texture augmentation.

### Error Sheets (per model, VAL)
- **Paths**: runs/R_eval/errors_R_*_val.png (35–48 KB each)
  - errors_R_font0_val.png: Fonts-only model failures (many misreadings)
  - errors_R_real_val.png: Real-only baseline errors
  - errors_R_font2real_val.png: Selected model errors (fewer, mostly punctuation/digits)
  - errors_R_font2real_10_val.png: 10%-writer scarce-data errors
  - errors_R_mix_val.png: Mixed-training errors
- **Caption** (R_font2real): Top misreadings from the font→real model on VAL (4,299 words). Remaining errors predominantly lone punctuation (, । ') and Latin/Bengali digit confusion (per EXPERIMENTS.md, line 40).

### Font Shaping Check
- **Path**: runs/font_search/glyphscribe/shaping_check.png (1.6 MB)
- **Content**: Grid of all 48 fonts rendering a test word (e.g., "যাওয়া"), verifying GSUB shaping correctness.
- **Caption**: Unicode Bengali GSUB shaping verification for 37 GlyphScribe handwriting fonts + 11 Google OFL + 1 printed. All fonts correctly shape complex conjuncts and diacritics.

### Font Score Sheet
- **Path**: runs/font_search/glyphscribe/sheet.png (1.1 MB)
- **Content**: Skeleton overlays (red=writer, blue/green=fonts) for top fonts vs writer image_2.
- **Caption**: Skeleton chamfer distance: Atma variants (chamfer 0.162–0.164) closest to real handwriting; NotoSans (0.197, baseline) ranks 28th of 39. Top ~10 fonts have overlapping 95% CIs.

### TEST Error Sheets
- **Paths**: runs/R_eval/errors_R_real_test.png, errors_R_font2real_test.png (35–37 KB each)
- **Caption** (for R_font2real TEST): Independent test-set errors (7,665 words, 20 writers not used in model development). Font pretraining reduces OOV errors from 0.096 (R_real) to 0.085 (R_font2real, clean OOV subset, n=237).

---

## Summary

R_eval (font-pretraining CRNN-CTC) validates that synthetic renders from 48 Bengali fonts improve legibility on real handwriting, especially when real data is scarce (10%-writer scenario: −0.063 [−0.080, −0.054] CER gain). On full real TRAIN (30k steps), the gain is modest (−0.004 codepoint CER overall) but significant on BW (−0.011). Fonts alone do not transfer (R_font0: 0.456 CER); mixing during fine-tuning does not beat plain fine-tuning (R_mix = R_real). Rescoring of diffusion arms S and P reveals ~1.3× real's legibility gap vs B0 font-GAN, partly due to short-word img_to_ink artifact. GlyphScribe fonts are viable for pretraining; Atma variants best match skeleton geometry; thin fonts (Qayyum, Tagore) best match pen width.

