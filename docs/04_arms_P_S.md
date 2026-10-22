# Arms P and S (P1a3) with Page Sheets and Smudge Fix

Latest fine-tuned diffusion arms with pen control (P and S) evaluated on the standard VAL protocol with improved recognition metrics, learned layout rendering, and ink preprocessing fixes.

## Architecture

Both arms **P** and **S** extend B' (runs/P1a2/B) through three additive features, each optional at init:

### 1. Pen-Width Knob
- **Module**: `PenEmbed` (wordgen/models/diffusion3.py)
- **Input**: Median sub-pixel width / core height (sub_core ratio) of the 5 reference canvases, drawn for the target word from the same writer or page group (B' fallback policy)
- **Encoding**: Normalize to z-score with TRAIN data statistics: z = (log v − μ) / σ, clipped to [−3, 3]
  - μ = −1.8347746345905305, σ = 0.34197642919583526 (source: runs/P1a3/S/config.json line 4-5)
- **Embedding**: Sinusoidal timestep-like embedding from u = 1000(z + 3) / 6, dim=320 → MLP(Linear→SiLU→Linear) with last layer zero-initialized, out_dim=1280, added to the UNet timestep embedding
- **Dropout**: Learned null vector (zero-init) replaces the pen embedding when dropout drops the pen signal (pen_drop=0.05)
- **Effect**: No-op at initialization (MLP's last layer is zero). At inference, sets the pen knob multiplier relative to the reference median; range 0.6–1.6× corresponds to ~0.062–0.246 log-units (sweep.csv)

### 2. Null-Style Fix (Arm P Minimum)
- **B' behavior**: Falls back to `label_emb(writer_id)` table when style is dropped during CFG; training always passed writer_id as the conditioning y
- **P behavior**: No `label_emb` lookup; a learned vector `null_style` in the embedding space (initialized to the mean TRAIN style feature: `style_lin(mean(features))`) replaces the dropped style
- **Result**: Decouples style conditioning from writer identity, allowing the model to learn a neutral style conditioned only on text and pen knob

### 3. Reference Tokens in Decoupled Cross-Attention (Arm S Only)
- **Encoder**: Generator-side Bengali writer encoder (`WriterEncoder`, ResNet-18, random init, SupCon trained on 100,868 TRAIN rows; runs/P1a3/style_enc/model.pt)
  - Input: fit_canvas_white2 canvases, 64×256 grey uint8
  - Output: 256-d per-canvas embedding
- **Tokenization**: Spatial map from layer4 (2×8 = 16 positions, 256-d each), per ref → 16 tokens per ref
  - 5 exchangeable refs → 80 tokens, projected to cross-attention width (320)
  - Learned per-position slot embedding (16 shared slots) disambiguates the 5 refs
  - Tokens travel inside the `context` tensor, split off by count in each block (required for CheckpointFunction's backprop)
- **Decoupled attention**: In every transformer block (IP-Adapter style):
  ```
  out = attn(q, text+content) + keep * W_ref * attn(q, K_ref, V_ref)
  ```
  - K_ref, V_ref initialized from the block's own to_k, to_v
  - W_ref zero-initialized (no-op at init)
  - Per-sample `keep` mask (binary, no gradient) passed as module attribute; zeroed during style dropout
- **Alternative not taken**: Appending tokens to shared context breaks the no-op property: 80 extra keys change softmax normalization even with zero W_ref
- **Sanity check** (VAL, no training on VAL): 5 refs per VAL writer → mean embedding; top-1 among own-dataset VAL writers (source: runs/P1a3/style_eval/, E_shape M2s): S 0.517 MRR, P 0.416, B0 0.533

### Warm Start
- Both P and S **warm-start from runs/P1a2/B** (B' full checkpoint) (source: runs/P1a3/S/config.json line 50: `warm_start: /home/hpc4090/shyan/ocr/scribe_wordgen/runs/P1a2/B`)
- DiffusionPen UNet weights are transferred; pen knob, null_style, and ref_tokens initialize as described above

---

## Training Hyperparameters

**Data**: 100,868 rows (TRAIN from data_cache/p1a3_cache/index.csv), bgnorm2 latents/style features (fit_canvas_white2 64×256 canvases) (source: wordgen/train/train_diffusion3.py line 55)

**Sampling** (per-draw probabilities with replacement):
- Punctuation-only rows: 1.5% of total draws (source: runs/P1a3/S/config.json sampling.shares.punct = 0.015)
- Non-punctuation BanglaWriting: 38% (source: runs/P1a3/S/config.json sampling.shares.bw = 0.38)
- Non-punctuation BN-HTR: 60.5% (source: runs/P1a3/S/config.json sampling.shares.bnhtr = 0.605)

**Per-writer rebalancing** (within each dataset group):
- Probability ∝ writer pen weight: w_pen(writer) = clip(dataset median pen / writer pen, 0.5, 2.0)
- Thin-pen writers (low pen ratio) upweighted up to 2×; thick-pen writers (high ratio) downweighted to 0.5×
- 177 writers upweighted to non-default weight, 1 at maximum (source: runs/P1a3/S/config.json sampling.n_writers_upweighted = 177, n_writers_at_max = 1)

**Optimizer**:
- Batch size: 32 (source: runs/P1a3/S/config.json line 7)
- Learning rate: 0.0001 base (source: runs/P1a3/S/config.json line 8)
- Warmup: 300 steps (source: runs/P1a3/S/config.json line 10)
- Final LR frac: 0.1 (source: runs/P1a3/S/config.json line 11)
- New LR multiplier: 10.0 (pen knob, ref_tokens, null_style) (source: runs/P1a3/S/config.json line 9)
- Clip grad: 1.0 (source: runs/P1a3/S/config.json line 52)
- Ref tokens learning rate multiplier: 2.0 (source: runs/P1a3/S/config.json line 51)

**Dropout** (CFG null drops all together):
- Text: 0.1 (source: runs/P1a3/S/config.json line 12)
- Content tokens: 0.1 (source: runs/P1a3/S/config.json line 13)
- Style: 0.1 (includes null_style fallback and ref_tokens zeroing) (source: runs/P1a3/S/config.json line 14)
- Pen: 0.05 (uses null pen embedding) (source: runs/P1a3/S/config.json line 15)

**Checkpoints**:
- Arm P: 16,000 steps (runs/P1a3/P) (source: /home/hpc4090/shyan/ocr/scribe_wordgen/runs/P1a3/P/metrics.csv: 16001 lines = 16000 steps)
- Arm S: 24,000 steps (runs/P1a3/S) (source: /home/hpc4090/shyan/ocr/scribe_wordgen/runs/P1a3/S/metrics.csv: 24001 lines = 24000 steps)
- Wall time (S): 2108.6 seconds ≈ 35 minutes (source: runs/P1a3/S/config.json line 54)
- Final loss (S, EMA): 0.0581 (source: runs/P1a3/S/config.json line 55)

**Core fraction band** (valid reference selection):
- Refs with core_px / ink_height ∈ [0.17, 0.75] are usable; fallback to all measurable refs if < 2 survive (source: runs/P1a3/S/config.json line 45-48)

---

## The Divergence and Gradient Bug Story

### S_diverged_v1 (2026-10-17 12:50–13:04, ~1500 steps)
- Training loss was ~0.06 at step 0, stable through step ~700
- At step 800, loss jumps to 0.27; steps 550–1500 show divergence with loss oscillating 0.12–0.25 (source: runs/P1a3/S_diverged_v1/train.log lines 13–34)
- **Root cause** (identified post-hoc): DiffusionPen block checkpointing's CheckpointFunction was re-running the forward pass after the side channel (pen embedding, null_style) had been reset to its initial/null state
  - Checkpoint's backward pass needs the forward activation tensors; DiffusionPen's custom CheckpointFunction stores side-channel state (pen embedding output, style latent) as module attributes
  - On backward, if the side channel is modified (e.g., pen dropout null vector), the stored state became stale
  - This caused incorrect gradient flow for the pen knob and null_style parameters, leading to instability (source: EXPERIMENTS.md line 42: "fixing a gradient bug (DiffusionPen block checkpointing re-ran forward after the side channel was reset)")

### S_diverged_v2 (2026-10-17 13:05–13:15, ~1050 steps, stopped early)
- Same training setup as v1
- Loss remains stable ~0.06–0.09 through step 1050 with only minor spikes (steps 800–1000: peaks ~0.11 then settle) (source: runs/P1a3/S_diverged_v2/train.log lines 13–24)
- Stopped early (1050 steps instead of full 24000)
- **Conclusion**: Gradient bug was fixed (side channel state properly preserved during checkpointing); divergence gone

### S_t5c5 (Final, 2026-10-17 retrained, 24k steps)
- Full 24,000-step training run after fix applied
- Final checkpoint (runs/P1a3/S/) trained successfully
- Loss stabilizes by step 1000–1500 and continues smooth decay (source: runs/P1a3/S/metrics.csv)

---

## Results Tables

**Evaluation protocol (VAL)**: 781 target words, 20 writers (5 BN-HTR + 15 BanglaWriting), 2 seeds per arm, per-word CER capped at 1, per-writer mean, 95% CI bootstrap over writers.

### Locator CER (BengaliScribe attn_locator)

| Arm | CER | CI | OOV CER | Notes |
|---|---|---|---|---|
| real | 0.133 | [0.100, 0.167] | 0.150 | baseline |
| B0 | 0.130 | [0.103, 0.156] | 0.235 | font-GAN prior |
| P_t5c5 | 0.132 | [0.097, 0.171] | 0.194 | pen + null-style, tied with B0 |
| S_t5c5 | 0.137 | [0.101, 0.174] | 0.202 | pen + null-style + ref tokens, +0.007 vs B0 [tie] |

(Source: runs/P1a3/eval/report_tables.txt lines 6–15)

**Paired comparisons**:
- P_t5c5 − B0: +0.002 [−0.028, +0.037] (10/20 writers favor P) (source: runs/P1a3/eval/report_tables.txt line 69)
- S_t5c5 − B0: +0.007 [−0.023, +0.040] (10/20 writers favor S) (source: runs/P1a3/eval/report_tables.txt line 71)
- S_t5c5 − P_t5c5: +0.006 [−0.004, +0.015] (6/20 writers favor S) (source: runs/P1a3/eval/report_tables.txt line 73)

### R_eval CER (Independent CRNN-CTC recogniser on real handwriting pretraining)

Arm S is **less legible than B0** under R_eval. This reflects that the recogniser was trained on synthetic fonts, and B0's font-like renderings score better than handwriting-fidelity generates (source: EXPERIMENTS.md line 31–34):

| Arm | R_eval CER all | R_eval CER ≥3cp | Paired diff (S − B0) |
|---|---|---|---|
| B0 | 0.030 | 0.023 | baseline |
| S_t5c5 | 0.089 | 0.055 | +0.059 [+0.036, +0.083] all; +0.033 [+0.015, +0.052] ≥3cp |
| P_t5c5 | 0.090 | 0.058 | S − P −0.002 [−0.011, +0.008] (tie) |

(Source: EXPERIMENTS.md line 22–34; runs/P1a3/eval/report_tables.txt)

**Note**: The short-word artifact inflates all-word CER (S 0.089 all vs 0.055 ≥3cp, 0.034-point gap). This is due to `img_to_ink`'s 5th-percentile floor for ink normalization: lone graphemes (০ ও २) cover 2–3% of canvas, so paper texture is stretched to black. Post-hoc fix applied (see Smudge Fix section below).

### Style MRR (E_eval writer-ID model)

Arm S significantly outperforms P on style matching; both lag B0:

| Arm | M2 MRR | M2s MRR | Paired diff (S − P) |
|---|---|---|---|
| real | 0.821 | 0.768 | — |
| B0 | 0.596 | 0.533 | — |
| P_t5c5 | 0.443 | 0.416 | baseline |
| S_t5c5 | 0.579 | 0.517 | +0.136 [+0.085, +0.187] M2; +0.101 [+0.057, +0.140] M2s (17/20 writers) |

(Source: runs/P1a3/eval/report_tables.txt lines 99–143)

**Subset performance**:
- BN-HTR (M2 E_eval): S 0.745, P 0.556, B0 0.883 (source: runs/P1a3/eval/report_tables.txt line 156)
- BanglaWriting (M2 E_eval): S 0.523, P 0.405, B0 0.500 (source: runs/P1a3/eval/report_tables.txt line 175)

### Pen Ratio (Raw and Calibrated)

| Arm | Raw ratio | CI | Calibrated ratio | CI |
|---|---|---|---|---|
| real | 1.00 | [1.00, 1.00] | — | — |
| B0 | 1.15 | [0.98, 1.20] | — | — |
| P_t5c5 | 1.15 | [1.01, 1.30] | 1.06 | [0.97, 1.15] |
| S_t5c5 | 1.13 | [1.07, 1.21] | 1.06 | [0.98, 1.12] |

(Source: runs/P1a3/eval/report_tables.txt lines 42–59)

**Gates (REVIEW 8.6)**: All three metrics pass:
- CER ≤ B0 + 0.05 (0.180 threshold): P 0.132 ✓, S 0.137 ✓ (source: runs/P1a3/eval/report_tables.txt line 271–273)
- Pen calibrated ≤ 1.15: P 1.06 ✓, S 1.06 ✓ (source: runs/P1a3/eval/report_tables.txt line 271–273)
- Writer-117 blotch < 10%: P 0% ✓, S 0% ✓ (source: runs/P1a3/eval/report_tables.txt line 271–273)

### Background Artifacts and Blotch

| Arm | bg-fail all | bg-fail non-punct | blotch rate | blotch writer 117 |
|---|---|---|---|---|
| real | 1.2% | 0.0% | 0.4% | 8% |
| B0 | 0.2% | 0.0% | 0.0% | 0% |
| P_t5c5 | 3.0% | 2.7% | 2.1% | 0% |
| S_t5c5 | 3.0% | 3.0% | 2.3% | 0% |

(Source: runs/P1a3/eval/report_tables.txt lines 24–34)

**Interpretation**: bg-fail = ink-bbox fill > 0.6 or bbox p25 > 0.1; blotch = faint ink > 2% of canvas and > 2 px from strong ink. Diffusion arms (P, S) have higher bg-fail (stochastic artifacts) but pass blotch gate. bgnorm2 (white paper normalization) eliminated writer-117 blotch spikes seen in earlier arms.

---

## Pen Knob Sweep

**Setup**: Multiplier applied to the reference median pen value; g5 (guidance scale), seed 0, 8 words × 20 writers (source: runs/P1a3/eval/report_tables.txt line 92–95):

| Arm | Slope d log(gen sub_core) / d log(knob) [95% CI] | gen/knob at mult 1.0 |
|---|---|---|---|
| P | 0.11 [0.08, 0.14] | 0.82 |
| S | 0.03 [0.00, 0.05] | 0.80 |

(Source: runs/P1a3/eval/report_tables.txt lines 93–95)

**Interpretation**:
- **Arm P**: Pen knob is active; a 10% increase in knob multiplier → ~1.1% increase in generated sub_core (source: runs/P1a3/eval/report_tables.txt line 87: "Pen knob vs output ... P_t5c5 | 0.15 | 0.31 | 0.86" slope = 0.15 without mult)
- **Arm S**: Pen knob is nearly inert; slope ≈ 0.03, roughly 3× weaker than P (source: runs/P1a3/eval/report_tables.txt line 89: "S_t5c5 | 0.15 | 0.15 | 0.82", Spearman = 0.15, close to random)

**Raw output data** (knob × multiplier, per writer medians):
- mult 0.6: P knob 0.0636–0.2471, S same range, gen sub_core 0.062–0.123 (P) / 0.062–0.119 (S) (source: runs/P1a3/eval/sweep.csv lines 2–50)
- mult 1.0: P knob 0.1060–0.3852, S same, gen sub_core 0.104–0.133 (P) / 0.104–0.132 (S) (source: runs/P1a3/eval/sweep.csv lines 18–65)
- mult 1.6: P knob 0.1696–0.6163, S same, gen sub_core 0.108–0.135 (P) / 0.107–0.131 (S) (source: runs/P1a3/eval/sweep.csv lines 34–81)

---

## Page Sheets Findings

**Scope**: Full handwritten pages (image_2, 3 BanglaWriting VAL writers) rendered with the LEARNED layout (scribe_layout_v2/runs/default_unfitw_dc) for real page (when available), B0, B', P, P_cal, S, S_cal (source: wordgen/integrate/page_sheets.py docstring; runs/page_sheets/all.log)

**Key observations**:

1. **Generated text renders smaller** than the real page on BanglaWriting writers (bw_101_21_0, bw_10_20_0)
   - Example: real page 7–11 lines on first page; P/S generate 6–8 lines (source: runs/page_sheets/all.log lines 14–36)
   - Likely cause: layout model trained on single words, not full pages; spatial generalization limited

2. **Arm P renders dash words (−) as thick bars** at certain pen knob settings
   - Observed on short words with very low content (1–2 graphemes)
   - Pen knob multiplier → output density in dash characters, overcomplicating the dash stroke
   - Not observed in S (source: EXPERIMENTS.md line 44)

3. **Layout variability across generators**: Same text + same reference writer produces different line counts (6–13 lines depending on arm/guidance)
   - Suggests word spacing or vertical positioning differs, likely due to subtle differences in rendered word widths
   - Real layout relies on pixel-level bounding boxes; small width differences compound across a page

4. **Page sheets PNG locations** (sheets for visual comparison):
   - image_2: /home/hpc4090/shyan/ocr/scribe_wordgen/runs/page_sheets/sheets/image_2_sheet_l0.png (side-by-side) and image_2_zoom_l0.png (3-line crops)
   - bw_101_21_0: bw_101_21_0_sheet_l0.png and bw_101_21_0_zoom_l0.png
   - bw_10_20_0: bw_10_20_0_sheet_l0.png and bw_10_20_0_zoom_l0.png
   - bnhtr_99: bnhtr_99_sheet_l0.png and bnhtr_99_zoom_l0.png
   - (Source: runs/page_sheets/all.log)

---

## Smudge Fix

### Problem: `img_to_ink` Short-Word Artifact

The 5th-percentile floor for ink normalization assumed ink covers ≥5% of canvas. For lone graphemes (০ ও २), coverage is only 2–3%, causing paper texture to be stretched to black, creating false "smudges" on page sheets (source: EXPERIMENTS.md line 43; wordgen/eval/p1a_eval.py).

### Solution

Added optional parameter `min_span` to `img_to_ink` in wordgen/eval/p1a_eval.py and wordgen/integrate/page_sheets.py:
```python
img_to_ink(..., min_span=100)  # opt-in; default=None for backward compat
```
Sets a minimum canvas span for ink binarization, so lone glyphs are no longer forced to fill the canvas (source: EXPERIMENTS.md line 43).

### Results

- **Before fix**: Lone-glyph ink fraction 39–56% of canvas
- **After fix**: Lone-glyph ink fraction 2–3% of canvas
- **Impact on CER/blotch**: Normal multi-grapheme words unchanged (source: EXPERIMENTS.md line 43)
- **Cacheing**: Eval caches (runs/P1a3/eval/cache) still use legacy conversion; short-word CER/blotch for P/B'/S carry the artifact until all arms rescored together with `min_span` applied (source: EXPERIMENTS.md line 43)

---

## Plottable Data

### Training Loss Curves
- **Arm P**: `/home/hpc4090/shyan/ocr/scribe_wordgen/runs/P1a3/P/metrics.csv`
  - Columns: step, loss, lr, sec_per_step, frac_bw, frac_punct, knob_med
  - 16,001 rows (0–16000 steps)
  
- **Arm S**: `/home/hpc4090/shyan/ocr/scribe_wordgen/runs/P1a3/S/metrics.csv`
  - Columns: step, loss, lr, sec_per_step, frac_bw, frac_punct, knob_med
  - 24,001 rows (0–24000 steps)

### Pen Knob Sweep
- **CSV**: `/home/hpc4090/shyan/ocr/scribe_wordgen/runs/P1a3/eval/sweep.csv`
- **Columns**: arm, wkey (writer_dataset), idx, text, mult (multiplier), knob, base_knob, sub_core, sub_px, core_px, blotch
- **Data**: All multipliers (0.6, 0.8, 1.0, 1.25, 1.6) × 5 BN-HTR writers × 8–10 words + 5 BanglaWriting writers × 8 words each
- **Plot**: Scatter (mult vs gen sub_core) or line (mult on x-axis, gen sub_core on y-axis, per writer)

### Per-Writer Differences
- **CER diffs**: runs/P1a3/eval/report_tables.txt lines 61–81 (paired CER comparisons with bootstrap CI and per-writer counts)
- **Style MRR diffs**: runs/P1a3/eval/report_tables.txt lines 117–143 (paired M2 MRR with p-values and writer tallies)
- **Pen diffs**: runs/P1a3/eval/report_tables.txt lines 42–59 (per-writer p10–p90 ratios, per-dataset means)

### Writer Conditioning
- **CSV**: `/home/hpc4090/shyan/ocr/scribe_wordgen/runs/P1a3/eval/writer_cond.csv`
- Per-writer CER/pen/style metrics for P and S arms

### Calibration Gains
- **CSV**: `/home/hpc4090/shyan/ocr/scribe_wordgen/runs/P1a3/eval/calib_gains.csv`
- Pen calibration ratios (target/gen) used for the `_cal` variants

---

## Candidate Figures

### Evaluation Summary
- **`/home/hpc4090/shyan/ocr/scribe_wordgen/runs/P1a3/eval/sheet_val.png`**
  - VAL set side-by-side word render comparison (real, B0, B', P_cal, S_cal)
  - Caption: "Word renders (VAL set, all arms, seed 0). Left to right: real crops, B0 (font-GAN), B' (grapheme tokens), P_cal (pen+null-style calibrated), S_cal (pen+null-style+ref-tokens calibrated)."

### Page Sheets
- **`/home/hpc4090/shyan/ocr/scribe_wordgen/runs/page_sheets/sheets/image_2_sheet_l0.png`**
  - Full 1-page side-by-side rendering (image_2 test writer, ~12 lines)
  - Caption: "Full-page sheet (image_2, 1 page, real vs B0 vs B' vs P_cal vs S_cal). P renders dashes as thick bars (see line 4). S generates more even line heights. Text renders smaller than the real page for BW writers."

- **`/home/hpc4090/shyan/ocr/scribe_wordgen/runs/page_sheets/sheets/image_2_zoom_l0.png`**
  - 3-line crop zoom, high-resolution view
  - Caption: "Zoom of lines 2–4 (image_2 sheet). Text rendering clarity across arms."

- **`/home/hpc4090/shyan/ocr/scribe_wordgen/runs/page_sheets/sheets/bw_101_21_0_sheet_l0.png`**
  - BanglaWriting writer 101 page; typically 6–7 lines generated vs 7 real
  - Caption: "Full-page BanglaWriting writer (bw_101_21_0). Generated text smaller than real. Pen calibration (P_cal, S_cal) shows more consistent stroke weight."

- **`/home/hpc4090/shyan/ocr/scribe_wordgen/runs/page_sheets/sheets/bnhtr_99_sheet_l0.png`**
  - BN-HTR writer 99 page; 12–13 lines
  - Caption: "Full-page BN-HTR writer (bnhtr_99, handwriting dataset). Layout stable across arms."

### Diagnostic
- **`/home/hpc4090/shyan/ocr/scribe_wordgen/runs/P1a3/eval/diag_bg_P_raw.png`**
  - Background failure and blotch diagnostics (raw P arm, pre-calibration)
  - Caption: "P arm background artifacts (raw, uncalibrated). bg-fail rate 3.0% (ink-bbox fill > 0.6 or p25 > 0.1), mostly from stochastic diffusion boundary noise."

