# Early Diffusion Experiments: probe_day1, P1a, P1a′, P1a3

## Summary

This section documents the initial diffusion model experiments starting from the frozen DiffusionPen checkpoint (trained on IAM), moving to Bengali data (BN-HTR + BanglaWriting), and progressively addressing legibility and styling issues through a series of experimental arms.

---

## 1. probe_day1: Proof of Concept (2026-10-12)

**What changed:** First attempt to sample from DiffusionPen (`dp_words_b64`, IAM-trained) on the target writer's handwriting style. Fine-tuning with 120 steps on image_2's 17 confident alignment crops (test set: 6 held-out crops). Budget: <1.5 GPU-hours.

**Architecture:** DiffusionPen U-Net from `Diffusion_model_dataset_building/dp_words_b64/unet_final.pt` (64×256 canvas, DDIM-50 sampling). Style conditioning via 5-crop style references. No content conditioning. Text-only cross-attention (CANINE).

**Training details:**
- Method: TTA (test-time adaptation) fine-tune on held-out real crops, 120 steps, batch 4, lr 5e-5, text-drop 0.1
- Training crops: 17 confident-alignment crops from image_2 (25% held out from 23 total)
- GPU time: ~2.5 min productive + ~4 min wasted on over-long fine-tune checkpoint explosion

**Results (24 probe words, locator CER):**

| method | n | CER (mean) | pen width/core | darkness |
|---|---|---|---|---|
| real (confident crops only) | 9 | 0.00 (circular) | 0.064 | 1.00 |
| B0 (BengaliScribe font-GAN) | 24 | 0.184 | 0.063 | 0.97 |
| DP, image_2 refs | 24 | 0.973 | 0.091 | 1.00 |
| DP, writer_99 refs | 24 | 0.931 | 0.110 | 1.00 |
| DP, writer_117 refs | 24 | 0.899 | 0.104 | 1.00 |
| DP, fine-tuned on image_2 | 24 | 0.651 | 0.100 | 1.00 |

Source: `runs/probe_day1/REPORT.md` section "Numbers"

**What was learned:**
- Diffusion texture is qualitatively rounder and more flowing than B0's angular, re-inked font skeleton. Visual evidence of handwriting-like character (`runs/probe_day1/line.png`).
- DP is NOT thinner than B0 at 64×256 canvas; both miss image_2's very thin pen (0.064 w/core vs DP's 0.091-0.110).
- Legibility is the dominant failure mode: conjuncts and long words are frequently misspelled or garbled (mean CER 0.89–0.97 for three reference sets).
- Style pathway does work: different reference sets produce visibly different slant/letter-width/joining style.
- No sign of memorization: fine-tuned model NCC vs nearest training crop ranges 0.14–0.42 (no 0.7–0.9+ copying).

**Known issues:**
- Probe metrics are small-scale (9–24 words, single day, indicative only).
- Locator CER is out-of-domain for DP's 64×256 native distribution.
- Darknesss metric saturates to ~1.0 by construction (uninformative).

Source: `runs/probe_day1/REPORT.md`

---

## 2. P1a: Content Map Experiment (2026-10-12)

**What changed:** Full training of DiffusionPen on BN-HTR+BanglaWriting TRAIN writers (40/writer cap: 14,155 crops, 354 writers), 20k steps at 64×256. Arm A = text-only control (CANINE context only). Arm B = 8-channel grapheme content map painted into extra input channels.

**Architecture:**
- Base: DiffusionPen U-Net from IAM checkpoint, 2-level model (8×32, 4×16 latent blocks)
- Arm A: Text via CANINE (frozen encoder), text-drop 0.1, no content
- Arm B: Text + 8-ch painted content map (grapheme embedding, zero-init first-conv slice), content-drop 0.1
- Canvas: 64×256 (fit_canvas convention)
- Style: Frozen IAM MobileNetV2, global vector pooled from 5 references

**Training details:**
- Steps: 20,000 (40 epochs over capped ~14k crops)
- Batch: 32
- Learning rate: 1e-4
- Training data: 14,155 crops from 354 writers (40/writer cap), 5,033 unique texts
- Wall time A: 3288 sec; Arm B: measured (no train_info for B in REVIEW era)

Source: `runs/P1a/train_info.json`, `runs/P1a/REVIEW.md` section 0

**Results (VAL, 20 writers, corrected CER table):**

| method | CER cap1, left-aligned | punctuation excluded | BN-HTR excl. 117 | BW | median CER | bg-failure % | sub-px width/real |
|---|---|---|---|---|---|---|---|
| real | 0.192 | 0.133 | 0.049 | 0.237 | 0.00 | 1.2 | 1.00 |
| B0 | 0.189 | 0.130 | 0.084 | 0.224 | 0.00 | 0.2 | ≈1.3 |
| B3 (baseline) | 0.460 | 0.422 | 0.427 | 0.476 | 0.50 | 3.8 | 1.32 |
| A | 0.516 | 0.482 | 0.423 | 0.525 | 0.60 | 8.7 | 1.40 |
| B | 0.494 | 0.457 | 0.361 | 0.516 | 0.60 | 8.8 | 1.40 |

Source: `runs/P1a/REVIEW.md` section 1, "Corrected table"

Paired differences (bootstrap over writers, left-aligned cap1, punctuation excluded):

| comparison | all 20 writers | BN-HTR | BW |
|---|---|---|---|
| B − A | −0.024 [−0.045, −0.002] (14/20 writers) | −0.062 [−0.084, −0.040] | −0.011 [−0.034, 0.012] |
| A − B3 | +0.060 [0.015, 0.114] | | |
| B − B3 | +0.036 [−0.011, 0.084] | −0.066 [−0.111, −0.021] | +0.043 [0.000, 0.080] |

Source: `runs/P1a/REVIEW.md` section 1

**Verdict:** NO-GO for content map. Content map is close to unused: painting another word's map does not steer output (CER vs painted word 0.96). Map-only (no text) gives near-unconditional output (0.95 vs 0.97 unconditional). B's 2-point gain over A is only on in-vocabulary words and is negligible vs the 0.33 CER gap to B0.

**Bugs and metric issues found:**
1. CER is unbounded on 1-character targets (punctuation: CER 23–29 for garbage reads); raw CER > 1 on 15% of words.
2. Grey-box failures on BN-HTR writer 117: grey background refs produce grey-filled target (w/core explodes to ~100 for 95–98% of words).
3. Eval asymmetry: real/B0 crops are tight, left-aligned; A/B/B3 are full 256-wide canvas. B0 wider than 256 px (~10%) were truncated, raising CER by 0.04–0.05.
4. Per-writer bootstrap uses unbounded values; CI width 1.1 CER for arm A.

**Key findings:**
- Content embedding barely moved during training: row-norm median 0.078 vs init ≈0.057; content channel contribution ≈2% of first-conv output vs 9.7% for latent channels.
- In-vocabulary vs OOV: A CER 0.415 (in-vocab) vs 0.568 (OOV); B 0.390 vs 0.574. B's gain is entirely in-vocabulary.
- Thick pen: both arms generate 1.4× the writer's pen width, but rank order preserved (Spearman ρ 0.88–0.90).
- Guidance (CFG g=1/2/3): arm A shows g=1 gives 1.31× pen, g=2/3 give 1.48×; CER improves from 0.78 to 0.46–0.50.

Source: `runs/P1a/REVIEW.md` sections 0, 2, 3

**Root causes identified:**
- Grey background: 6.4% of BN-HTR TRAIN crops have paper < 250; model learned "grey refs → grey target" correctly (faithful conditional behavior on un-normalized data). VAL writer 117 refs paper 151–188.
- Short-word blobs: 1–2-character words have low signal; `fit_canvas` puts them in 150+ px padding; content map too weak to help.
- Thick pen: CFG guidance contributes some thickening; training-target pen distribution skews thick (38% BN-HTR, w/core 0.171); weak IAM style encoder (real-vs-real MRR 0.41).
- Data scale: 14k crops, 5k unique texts, ~13% of available TRAIN; OOV CER vs in-vocab gap suggests word-shape memorization through CANINE, not compositional grapheme drawing.

Source: `runs/P1a/REVIEW.md` section 3

---

## 3. P1a′ (A′, B′): Full Data + White-Paper Norm (2026-10-13)

**What changed:** Retrained with (a) full no-cap TRAIN subset (98,856 crops, 354 writers, vs P1a's 14,155), (b) white-paper normalization on every crop and reference, (c) redesigned content conditioning as grapheme-token cross-attention (not painted map). Arm A′ = text-only control. Arm B′ = grapheme tokens + column-position encoding in cross-attention. 40k steps.

**Architecture:**
- Base: Same DiffusionPen U-Net from IAM checkpoint
- Arm A′: Text (CANINE) + white-paper-norm refs, text-drop 0.1, no content
- Arm B′: Text + grapheme-token embeddings (768-dim, matching CANINE hidden size) + column-centre and span positional encodings, appended to cross-attention context. Content-drop 0.1. Optional 8-ch spatial map (rescaled embedding, TARGET_MAP_NORM=4.0).
- Canvas: 64×256 with `fit_canvas_white` normalization

**Training details:**
- Steps: 40,000 (40 epochs over full ~99k crops)
- Batch: 32
- Learning rate: 1e-4
- Training data: 98,856 crops from 354 writers (no per-writer cap), 354 unique writers
- Arm A′ wall time: 1344 sec (~22 min)
- Arm B′ wall time: 3378 sec (~56 min)
- Vocabulary: 354 writers, ~1300 grapheme clusters (saved as grapheme_vocab.json)

Source: `runs/P1a2/A/train_info.json`, `runs/P1a2/B/train_info.json`

**REPORT verdict (before review fix):** "B′ is a NO-GO, worse than A′."

**REVIEW verdict (bug fix, correct CFG null):** B′ is the best diffusion model so far. Bug found in eval_v2: content map/tokens put into BOTH CFG branches (no guidance of content conditioning). Fixed null = text "" + zero map + masked tokens.

**Results (VAL, 20 writers, CER capped at 1, punctuation excluded):**

| method | CER (all) | 1–2 graph | >2 graph | in-vocab | OOV | bg-fail % | pen/writer-real |
|---|---|---|---|---|---|---|---|
| real | 0.133 [0.100, 0.167] | 0.130 | 0.135 | 0.129 | 0.150 | 1.2 | 1.00 |
| B0 | 0.130 [0.103, 0.156] | 0.115 | 0.143 | 0.116 | 0.235 | 0.2 | 1.15 |
| B3 | 0.421 [0.381, 0.463] | 0.397 | 0.433 | 0.400 | 0.508 | 3.8 | 1.13 |
| **A′ g2 (reported)** | 0.334 [0.288, 0.382] | 0.328 | 0.341 | 0.312 | 0.496 | 5.4 | 1.87 |
| **A′ g3 (fixed)** | 0.245 [0.205, 0.287] | 0.241 | 0.247 | 0.228 | 0.384 | 5.2 | 1.22 |
| **B′ g2 (fixed CFG)** | 0.262 [0.218, 0.304] | 0.249 | 0.274 | 0.239 | 0.385 | 1.3 | 1.27 |
| **B′ g3 (fixed CFG)** | **0.188 [0.150, 0.226]** | 0.191 | 0.188 | 0.169 | 0.301 | 1.8 | 1.29 |
| B′ content-only g4 (M) | 0.199 [0.161, 0.238] | 0.206 | 0.195 | 0.186 | 0.303 | 1.7 | – |

Source: `runs/P1a2/REVIEW.md` section 2, "Full VAL with the fixed sampler"

By dataset (overall CER):

| dataset | real | B0 | A′ g3 | B′ g3 |
|---|---|---|---|---|
| BN-HTR (5 writers) | 0.055 [0.036, 0.075] | 0.085 [0.054, 0.124] | 0.160 [0.128, 0.188] | **0.108 [0.077, 0.139]** |
| BanglaWriting (15 writers) | 0.159 [0.126, 0.197] | 0.145 [0.118, 0.173] | 0.273 [0.219, 0.325] | 0.214 [0.170, 0.256] |

Source: `runs/P1a2/REPORT.md` section 4 (note: REVIEW corrected these with fixed CFG)

**Paired differences (fixed CFG, bootstrapped over writers):**

| comparison | difference | writers favouring the first |
|---|---|---|
| B′g3 − A′g3 | −0.057 [−0.089, −0.032] | 18/20 |
| B′ content-only − A′g3 | −0.045 [−0.067, −0.026] | 17/20 |

Source: `runs/P1a2/REVIEW.md` section 2

**Shuffled-content gate (content steering test):**

| condition | CER vs intended word |
|---|---|
| B′ with correct CFG (g3) | 0.188 |
| B′ full (text + tokens, g3) | 0.188 |
| B′ text only (tokens nulled, g3) | 0.435 |
| B′ content only (no text, g=2) | 0.383 |
| B′ content only, g=4 | **0.248** |
| B′ shuffled-content (another word's tokens, text for intended word) | **0.432** (vs painted word 0.558 at g2, 0.400 at g4) |

Source: `runs/P1a2/REVIEW.md` section 2 ("Shuffled-content gate")

Content path ablation (content-only g4):
- tokens + map: 0.248
- **map alone: 0.902** (painted map is inert, same as P1a)
- tokens alone: 0.338

Conclusion: Word identity carried by grapheme tokens, not the spatial map.

Source: `runs/P1a2/REVIEW.md` section 2

**Bugs found and fixed:**
1. **Eval v2 validation against P1a:** CER numbers matched within bootstrap noise (P1a A 0.481 → eval_v2 A 0.481; B 0.457 → 0.457).
2. **Locator canvas:** Requires fixed 64×256 canvas (frozen pos parameter shape `(1, C, 16, 64)`). Fixed by tight-crop + left-align, pad to 64×256, squeeze via resize if native > 256 px.
3. **Off-by-one vocab_size:** Stage_gen and shuffled_content_test called `vocab_size=len(vocab)` but training reserves +1 for pad/OOV. Fixed to `len(vocab) + 1`.
4. **Sign-inverted OOV gate:** `gate_oov_pass = bool((-oov_pt) >= 0.05)` instead of `bool(oov_pt >= 0.05)`. Fixed and re-run from scratch.

Source: `runs/P1a2/REPORT.md` sections 1, 4

**White-paper normalization caveat:** Single-scalar normalization (90th-percentile paper → 255, linear scale) does not flatten illumination or neutralise sepia tint. Residual tan/yellow tint in 0.8% of crops (chroma > 15, mostly BN-HTR). Does not separate gradient/texture from dense ink (14% of BN-HTR have 30th-percentile L < 230).

Source: `runs/P1a2/REPORT.md` section 2

**bg-failure metric re-examined (REVIEW section 3):**

| method | bg-fail % (all rows) | punctuation-only (% fail) | non-punct | 117 faint-blotch % |
|---|---|---|---|---|
| real | 1.2 | 9.9 | 0.0 | 8 |
| A′ | 5.4 | **45.3** | 2.5 | **45** |
| B′ fixed g3 | 1.8 | – | 0.4 | **48** |

A′'s excess bg-fail comes from untrained punctuation (dropped from P1a′ training) and 1–2-grapheme collapse. B′'s grapheme tokens give short words concrete layout. On real words (>2 graphemes), both are ≈0.2%. Writer 117's grey blotches (faint ink 0.18–0.5) are not caught by the bg-failure metric (fill > 0.6 or p25 > 0.1), but faint-blotch score (>2% canvas covered by faint ink >2px from stroke) flags 45–48% of 117's words.

Source: `runs/P1a2/REVIEW.md` section 3

**Pen ratio re-measured (REVIEW section 1):**

eval_v2's EDT-based measurement (integer-quantized: 1, 1.83, 3, 3.47, …) is an artefact. Real crops sit at 1.0 for 52% of words; generators sit at 3.0 for 46–58% of words. Sub-pixel width / core (like-for-like):

| method | sub-px width/core |
|---|---|
| real | 1.00 (2.41 px at core 22 px) |
| B0 | 1.15 [0.86–1.27] |
| B3 | 1.13 [0.84–1.54] |
| A′ g3 | 1.22 |
| B′ g3 fixed CFG | 1.29 [0.93–1.80] |

Conclusion: A′ and B′ are ≈1.3× the writer's pen relative to letter size, thicker than B0/B3 (≈1.15). Likely cause: P1a′ training is 84% BN-HTR (vs P1a 38%), and BN-HTR pens are thick (w/core 0.171 at canvas). Guidance thickens only slightly (sub-px 1.40 → 1.45–1.50 from g2 to g3/g4).

Source: `runs/P1a2/REVIEW.md` section 1

**What was learned:**
- A′'s gain over P1a A is real (−0.147 CER, 20/20 writers) from full data + white-paper norm. Mostly in-vocabulary.
- Grapheme-token cross-attention is the working content pathway; the spatial painted map remains inert.
- Content steering gate passes: shuffled-content CER vs painted word 0.432 at g3 (better than P1a's 0.959).
- With fixed CFG and equal guidance (g3), B′ beats A′ by 0.057 CER (18/20 writers).
- Evaluator affinity: locator is BN-HTR-trained; A′ training is 84% BN-HTR, so larger BN-HTR gain (−0.25 vs −0.11 on BW) may partly reflect this.

Source: `runs/P1a2/REVIEW.md` sections 2, 0

---

## 4. Side-by-Side Comparison (2026-10-13)

**What changed:** Direct comparison of all generators (B0, B3, P1a A, P1a′ A′, P1a′ B′) on image_2 text and VAL writers.

**Methods compared:**

| method | VAL CER | image_2 CER (24 words × 3 seeds) | image_2 pen vs writer | checkpoint/notes |
|---|---|---|---|---|
| real | 0.133 [0.100, 0.167] | 0.00 (9 crops only) | 1.00 | handwritten baseline |
| B0 (BengaliScribe) | 0.130 [0.103, 0.156] | 0.17 | 1.11 | font-GAN baseline |
| B3 (DiffusionPen IAM) | 0.421 [0.381, 0.463] | 0.62 | 2.51 | external DiffusionPen |
| P1a A | 0.481 (P1a review) | 0.64 (0.60/0.65/0.68) | 1.68 | runs/P1a/A, 40/writer cap |
| P1a′ A′ g3 | **0.245** [0.205, 0.287] | **0.40** (0.45/0.34/0.43) | 1.70 | runs/P1a2/A, full data + white-norm |
| P1a′ B′ g3 fixed CFG | **0.188** [0.150, 0.226] | **0.34** (0.40/0.23/0.39) | 1.98 | runs/P1a2/B, tokens + positions |
| Atma font (plain) | – | 0.08 | 1.50 | no learning (out-of-dist) |

Source: `runs/compare_inference/README.md` and `image2_scores.txt`

**Visual assessment (from sheets):**
- Real (image_2 page line 1): very thin, sparse, rounded cursive pen.
- B0: bold, uniform-width, angular strokes; re-inked font skeleton.
- B3: some texture but very thick and misspelled.
- P1a A: rounder flow than B0; still thick and garbled.
- P1a′ A′: crisper, more legible, still generic upright thick hand.
- P1a′ B′: spells showcase words correctly where A′ produces pseudo-words (e.g. కార్తిక, రాత్ర, ঘटিকায়). Pen and slant still generic, not image_2's thin rounded hand.

Source: `runs/compare_inference/README.md` and `runs/P1a2/REVIEW.md` section 2 ("image2_fix.png")

---

## 5. Font Search: Ranking Candidate Fonts (2026-10-13)

**What changed:** Measured 12 Bengali Google Fonts against image_2's handwriting (12 words, hand-verified matches against `hcr/996/Annotations/4.txt`). Used explainable measures (no learned model): chamfer distance (primary), stroke width, roundness, proportions, slant, matra properties.

**Candidate fonts (11 Google OFL Bengali fonts downloaded; 12 = Noto Sans Bengali already in repo):**
Alkatra, Anek Bangla, Atma, Baloo Da 2, Galada, Google Sans, Hind Siliguri, Mina, Noto Sans Bengali (current + GF copy), Noto Serif Bengali, Tiro Bangla.

All 12 fonts verified for correct shaping (PIL+raqm harfbuzz, conjuncts, reph, nukta, vocalic-r, visarga, candrabindu, ya-phala, dotted-circle sanity check).

Source: `runs/font_search/REPORT.md` sections 1, 4 ("Shaping verification")

**Ranked result (chamfer distance, primary metric; 95% CI bootstrap over 12 words):**

| # | Font | chamfer [CI] | chamfer@pen | width/core | corner/mm@pen | slant deg |
|---|---|---|---|---|---|---|
| 1 | **Atma** | 0.164 [0.144, 0.186] | 0.178 | 0.138 | 0.308 | −0.6 |
| 2 | **Hind Siliguri** | 0.169 [0.142, 0.199] | 0.177 | 0.163 | 0.286 | −0.2 |
| 3 | **Noto Serif Bengali** | 0.176 [0.146, 0.209] | 0.187 | 0.138 | 0.297 | +0.2 |
| 4 | Galada | 0.176 [0.152, 0.204] | 0.192 | 0.232 | 0.325 | −12.8 |
| 5 | Mina | 0.177 [0.148, 0.209] | 0.187 | 0.155 | 0.297 | −0.1 |
| 6 | Google Sans | 0.179 [0.147, 0.214] | 0.195 | 0.173 | 0.238 | −0.1 |
| 7 | Alkatra | 0.183 [0.157, 0.212] | 0.197 | 0.232 | 0.237 | −3.9 |
| 8 | Anek Bangla | 0.187 [0.152, 0.227] | 0.200 | 0.194 | 0.254 | −0.2 |
| 9 | Baloo Da 2 | 0.190 [0.158, 0.230] | 0.201 | 0.149 | 0.275 | +0.3 |
| 10 | Tiro Bangla | 0.193 [0.160, 0.233] | 0.199 | 0.177 | 0.291 | +0.7 |
| 11 | Noto Sans Bengali (current) | 0.197 [0.157, 0.243] | 0.213 | 0.177 | 0.290 | +0.3 |
| 12 | Noto Sans Bengali (GF copy) | 0.203 [0.164, 0.249] | 0.214 | 0.179 | 0.304 | +3.3 |

Real writer (image_2) target: width/core = 0.075, width = 0.294–0.307 mm, corner/mm = 0.393, slant = −4.2 deg, matra thickness 1.31 mm, matra continuity 0.73, word-width/core 5.20.

Source: `runs/font_search/REPORT.md` section 3

**Top 3 assessment:**
1. **Atma** (OFL): lowest chamfer, both raw and pen-matched. Stroke-width/core 0.138 closest to writer's among top group. Upright (−0.6° vs real −4.2°). Rounded display design matches "rounded, looped" qualitatively and numerically.
2. **Hind Siliguri** (OFL): very close second. Lowest corner-density gap at matched pen (0.286 vs real 0.393), least angular outline. Humanist sans, strong shape match despite not strictly "handwriting" style.
3. **Noto Serif Bengali** (OFL): third. Same width/core as Atma (0.138), near-zero slant (+0.2°). Serif design stylistically odd for "handwriting," but numbers competitive. Flagged as judgement call.

**Caveat:** Top 6 fonts sit within ~1 point of each other's chamfer CI (12 words insufficient to cleanly separate). Treat as "these 3–5 meaningfully better than NotoSansBengali," not precise total order.

Source: `runs/font_search/REPORT.md` sections 4, 3

**Key measurements:**
- All fonts render strokes 1.8×–3× heavier than writer: width/core 0.138–0.232 vs 0.075 real (measured as 0.59–0.99 mm vs 0.294–0.307 mm at these word sizes).
- No font reproduces broken matra/headline: font continuity 0.93–0.98 vs real 0.73.
- Corner-density appears "backwards" (real 0.393 > fonts 0.238–0.325) because thin ink's outline wiggles show up as corners. Pen-matched comparison (width-controlled) is fairer and shows fonts still < real.

Source: `runs/font_search/REPORT.md` section 5

---

## 6. P1a3: Pen Knob and Style (2026-10-19)

**What changed:** Added pen-width conditioning (scalar embedding from refs' measured sub-px width/core) and null-style fix to B′. Arm S = B′ + pen knob + ref tokens in decoupled cross-attn. Arm P = B′ + pen knob + null-style fix. Rebalancing to address BN-HTR/BW imbalance. Checkpoint was gradient bug fix (DiffusionPen block checkpointing re-ran forward after side channel reset).

**Training details:**
- Steps: 24,000 (estimate from "35 min" vs P1a2 benchmark)
- Arm S: pen knob + ref tokens in cross-attn decoupled
- Arm P: pen knob + null-style fix
- Wall time: ~35 min on RTX 5090

Source: EXPERIMENTS.md log row "Arm S" (2026-10-19)

**Results (from Current state table, EXPERIMENTS.md):**

| arm | CER (locator) | R_eval CER all / ≥3cp | style M2 MRR | pen | checkpoint |
|---|---|---|---|---|---|
| **S_t5c5** | 0.137 [0.101, 0.174] | 0.089 / 0.055 | **0.579** | 1.13 (1.06 cal) | runs/P1a3/S/ |
| **P_t5c5** | 0.132 [0.097, 0.171] | 0.090 / 0.058 | 0.443 | 1.15 (1.06 cal) | runs/P1a3/P/ |
| B0 | 0.130 [0.103, 0.156] | 0.030 / 0.023 | 0.596 | 1.15 | BengaliScribe `models/v2` |
| Bv2_t5c5 | 0.151 [0.117, 0.186] | 0.141 / 0.102 | 0.418 | 1.29 | runs/P1a2/B/ |

Source: EXPERIMENTS.md "Current state (2026-10-19)" table

**Paired comparisons:**
- S − B0 style: −0.017 [−0.116, +0.088], statistical tie.
- S − P style: +0.136 [+0.085, +0.187], 17/20 writers.
- S − B0 CER: +0.007 [−0.023, +0.040], tie.
- S OOV CER: 0.202 vs B0 0.235; paired CI crosses 0.

Source: EXPERIMENTS.md "Current state" section

**R_eval rescoring picture (independent CRNN-CTC recogniser):**
- S − B0: +0.059 [+0.036, +0.083] on all words; +0.033 [+0.015, +0.052] on ≥3-codepoint words. S is less legible than B0.
- S − P: tie (−0.002 [−0.011, +0.008]).
- B0 scores better than real crops (−0.014 [−0.027, −0.002]); legibility rewards font-like renders.

Source: EXPERIMENTS.md "Current state" section

**Gates status (from P1a3 eval report):**
- Passes all §8.6 gates
- M2 shape MRR: S 0.517, P 0.416, B0 0.533
- BN-HTR M2: 0.745
- Writer-117 blotch: S 0%, B0/P unspecified
- Pen knob nearly inert in S (sweep slope 0.03 vs P 0.11)

Source: EXPERIMENTS.md "Arm S" row / `runs/P1a3/eval/report_tables.txt`

**What was learned:**
- Style M2 MRR: S (0.579) maintains strong style capability vs P (0.443) and close to B0 (0.596). Ref tokens help style representation.
- Pen knob in S is nearly inert (slope 0.03 vs P 0.11), suggesting the knob or its initialization may need adjustment in S.
- R_eval reveals S is worse than B0 on legibility despite similar or tied locator CER — locator was too weak to separate arms.
- B0 scores better than real crops with R_eval, indicating legibility evaluation favours font-like renders over handwriting.

Source: EXPERIMENTS.md notes and P1a3 eval data

---

## 7. Bugs Found and Fixes

### CFG-Null Sampling Bug (P1a2 REVIEW, 2026-10-13)

**The bug:** In `eval_v2.sample_arm2` and `sample_arm2_shuffled`, the content map and tokens were put into **both** the conditional AND unconditional CFG branches. `ctx_null.update(content_ids=ids, ...)` and the same `content` map were passed to the unconditional pass. This meant guidance only extrapolated the text direction; content conditioning was never guided.

**The fix:** Correct joint null = text "" + zero map + masked tokens, matching what training used (10% content drop with zero map and masked tokens in `train_diffusion2.py`).

**Impact on results:**
- Before fix: B′ reported as 0.550 CER (REPORT.md)
- After fix: B′ g3 = 0.188 CER (REVIEW.md, measured), −0.057 [−0.089, −0.032] better than A′ g3
- Content-steering gate: shuffled CER vs painted word 0.793 → 0.432 (g3) / 0.400 (g4)
- Content-only g4: 0.199 CER (writes the word without text)

Source: `runs/P1a2/REVIEW.md` section 0 ("The bug")

### Eval v2 Bugs (P1a2 REPORT, sections 1, 4)

1. **Locator canvas mismatch:** Docstring promised "tight-crop + left-align at native width, no truncation," but locator's frozen pos parameter is fixed shape `(1, C, 16, 64)`, so only 256-px width (→64 feature width after x4 stride) is supported. No way to score native-width canvas. Fixed by tight-crop + left-align, pad to 64×256, squeeze via resize if native > 256.

2. **Off-by-one vocab_size:** `stage_gen` and `shuffled_content_test` called `load_arm2("B", vocab_size=len(vocab))` but training saved checkpoint's `text_encoder.tok_embed.weight` as `(1301, 768)` with `len(vocab)==1300` (reserves +1 for OOV/pad). Both call sites fixed to `len(vocab) + 1`.

3. **Sign-inverted OOV gate:** `gate_oov_pass = bool((-oov_pt) >= 0.05)` instead of `bool(oov_pt >= 0.05)`. Fixed and full run re-done from scratch.

Source: `runs/P1a2/REPORT.md` sections 1, 4

### Gradient Bug (P1a3, 2026-10-19)

**The bug:** DiffusionPen block checkpointing re-ran forward pass after the side channel was reset, causing incorrect gradients.

**The fix:** Retrain arm S after fix. 24k steps, 35 min.

**Impact:** Arm S results reported with fix applied.

Source: EXPERIMENTS.md "Arm S" row

---

## Plottable Data

### Training Loss Curves

**P1a2 Arm A′ (full data, text-only control):**
- File: `/home/hpc4090/shyan/ocr/scribe_wordgen/runs/P1a2/A/metrics.csv`
- Columns: `step` (0–40000), `loss` (floats), `sec_per_step`, `img_per_sec`
- 40,000 steps recorded; final loss ~0.0755 (from REVIEW text "last 1k steps")
- Can plot loss vs step; typical denoising schedule decay visible in raw loss

**P1a2 Arm B′ (full data, grapheme tokens + positions):**
- File: `/home/hpc4090/shyan/ocr/scribe_wordgen/runs/P1a2/B/metrics.csv`
- Columns: same as A′
- 40,000 steps; final loss ~0.0745 (0.5% below A′)
- Token/position learning embedded in the same loss; no ablated losses recorded

**P1a Arm A (40/writer cap, text-only):**
- File: `/home/hpc4090/shyan/ocr/scribe_wordgen/runs/P1a/A/metrics.csv`
- 20,000 steps (roughly double the training rate vs P1a2 due to smaller data)
- Loss trajectory available for comparison of data-scale effect on convergence

**Guidance sweep data (P1a):**
- File: `runs/P1a/REVIEW.md` section 3 ("Thick pen" analysis)
- Table: "arm A on 24 words: g=1 gives 1.31×, g=2 gives 1.48×, g=3 gives 1.48×. CER is 0.78, 0.50 and 0.46."
- Can plot CFG scale (g) vs CER and vs pen width; shows non-monotonic pen response

### Evaluation Metrics CSVs

**P1a2 full VAL metrics:**
- File: `/home/hpc4090/shyan/ocr/scribe_wordgen/runs/P1a2/eval_v2_rows.csv`
- Per-word CER, pen ratios, bg-failure flags for all 781 words × 20 writers × methods (real, B0, B3, A′, B′)
- Can aggregate by writer, by dataset (BN-HTR vs BW), by word length (1–2 vs >2 graphemes), by in-vocab/OOV

**P1a2 shuffled-content ablation:**
- File: `/home/hpc4090/shyan/ocr/scribe_wordgen/runs/P1a2/shuffled_content.csv`
- 48 VAL words (2 seeds each), conditions: full B′, zero map, shuffled map, map-only, unconditional, content-only
- CER for each (column CER labels); can plot content-steering progression

**P1a image_2 side-by-side scores:**
- File: `runs/compare_inference/image2_scores.csv`
- 24 words × methods (real, B0, B3, P1a A, P1a′ A′, P1a′ B′)
- Columns: method, words (n), CER (pooled over seeds), per-seed CER (e.g. "0.598 0.645 0.684"), pen sub/core, pen vs writer
- Can plot CER by method; seed variation visible per row

**Font search measurements:**
- File: `runs/font_search/scores_long.csv`
- Raw per-word, per-font measurements: 144 rows (12 words × 12 fonts)
- Columns: word, font, chamfer, chamfer@pen, width/core, corner/mm, slant, matra_thickness, matra_continuity, word_width_core
- Can plot by font (Atma vs Hind vs Noto) or aggregate to bootstrap CIs (per-font means in `scores.csv`)

### Known Step Ranges and Axis Labels

**P1a2 loss curves:**
- x-axis: step ∈ [0, 40000]
- y-axis: loss (starts ~0.3, ends ~0.07–0.08)
- Rolling 1k-step averages: A′ final 0.0755, B′ final 0.0745

**Image_2 guidance sweeps:**
- x-axis: guidance scale g ∈ {1, 2, 3}
- y-axis (left): CER ∈ [0.46, 0.78]
- y-axis (right): pen width ratio ∈ [1.31, 1.48]

**Font chamfer CI plot:**
- x-axis: 12 fonts (ranked 1–12, top 6 within ~1 CI)
- y-axis: chamfer distance [0.15–0.21]
- Error bars: 95% bootstrap CI (12 words each)

---

## Candidate Figures

### Existing PNG Paths

**probe_day1:**
- `runs/probe_day1/sheet.png` — 24 probe words × [real | B0 | DP image_2 | DP writer_99 | DP writer_117 | DP fine-tuned]; each row labelled with word, CER, held-out flag. **Caption:** "Probe word comparison: DiffusionPen early sampling variants vs B0 and real reference. DP methods show rounder, more flowing texture but high misspelling rate (CER 0.65–0.97) and thick pen vs real."
- `runs/probe_day1/line.png` — page line 1 (9 words) comparison across methods. **Caption:** "Page line 1 visual assessment: real vs B0 font-GAN vs DP variants. Diffusion texture visibly rounder and more cursive; neither approaches real's thinness."

**P1a2 evaluation:**
- `runs/P1a2/sheet_val.png` — VAL writers' 4 words each × [real | B0 | B3 | A′ | B′]. **Caption:** "VAL cohort comparison: A′ crisper and more legible than B′ under buggy eval_v2 sampling. Visual shows B′ sometimes mangles words more. (Note: sampled with incorrect CFG null; fixed results in REVIEW show B′ superior.)"
- `runs/P1a2/sheet_image2.png` — image_2 showcase words with [real | B0 | B3 | A′ | B′]. **Caption:** "Image_2 showcase with white-paper-normalized refs (P1a2 default). Neither arm reliably reproduces intended words (e.g. row 00 'ঊনিশে' → different pseudo-word in both). Consistent with OOV CER ~0.50–0.65 and content-steering gate failure."
- `runs/P1a2/bgnorm_sheet.png` — 20 darkest-paper BN-HTR crops before/after white-paper normalization. **Caption:** "White-paper normalization (fit_canvas_white): background lightens substantially but residual tan/yellow tint remains in some crops (sepia not corrected by single-scalar normalization)."

**P1a2 REVIEW supplementary:**
- `scratch/review_p1a2_out/bnhtr117_refs_raw_vs_white.png` (if saved during review) — writer 117 references before/after whitening. **Caption:** "Writer 117 reference whitening: 90th-percentile paper → 255 scaling leaves residual tan box with illumination gradient on pure-white padding, explaining 45–48% faint-blotch rate on 117's VAL words."
- `scratch/review_p1a2_out/bprime_cfg_sheet.png` (if saved) — B′ sampled 6 ways (CFG bug vs fixed, various guidance scales). **Caption:** "B′ sampling variants with correct joint CFG null: g=2 (0.262 CER) vs g=3 (0.188 CER) vs content-only g=4 (0.199 CER). Fixed null enables effective content guidance."
- `scratch/review_p1a2_out/image2_fix.png` — image_2 showcase with whitened refs and fixed B′ sampler. **Caption:** "Image_2 with fixed CFG null and whitened refs: B′ g3 spells showcase words correctly where A′ g3 produces pseudo-words (e.g. കാര്തിക, രാത്ര, ഘടികായ). Pen and slant still generic, not matching writer's thin rounded hand."

**compare_inference:**
- `runs/compare_inference/image2_side_by_side.png` — 24 image_2 words × [real (9 only) | B0 | B3 | P1a A | P1a′ A′ g3 | P1a′ B′ g3 fixed | Atma font]. Under each cell: locator read + CER. **Caption:** "Side-by-side generator comparison on image_2 (probe writer, 24 words). B′ g3 fixed is best diffusion arm (0.34 CER), but still 2× B0 (0.17). Top Atma font baseline 0.08 (no learning, OOD distribution)."
- `runs/compare_inference/val_side_by_side.png` — 6 VAL writers × 4 words each × methods. **Caption:** "VAL writer subset (6 writers, 4 words each): shows per-writer legibility variation. Diffusion arms more consistent per-writer than OOV probe words."

**font_search:**
- `runs/font_search/sheet.png` — 12 words × [REAL | NotoSansBengali (raw & @pen) | Atma (raw & @pen) | HindSiliguri (raw & @pen) | NotoSerifBengali (raw & @pen)]. **Caption:** "Font candidate comparison: Atma and Hind Siliguri have lowest chamfer distance to real writer's skeleton (raw render vs pen-redrawn at writer's width 0.294 mm). NotoSansBengali, current choice, ranks 11th of 12."
- `runs/font_search/sheets/shaping_check.png` — 9 conjunct strings (ক্ষ, জ্ঞ, ন্ত্র, etc.) + invalid isolated-phala across 12 fonts. **Caption:** "Shaping verification: all 12 candidate fonts render conjuncts, reph, nukta, vocalic-r, visarga, candrabindu, and ya-phala correctly. Dotted circles appear only for deliberately invalid input, confirming harfbuzz/GSUB compliance."

---

## Summary of Findings

**Trajectory:** probe_day1 showed diffusion-generated texture is qualitatively handwriting-like (rounder, flowing) vs font-GAN but with high misspelling and thick pen. P1a revealed the first content-map design (painted 8-ch embedding) was inert (<2% signal). P1a′ with full data + white-paper norm reduced legibility gap dramatically (A′ 0.334 → 0.245 CER with fixed CFG), and grapheme-token cross-attention re-design succeeded in carrying word identity (shuffled gate 0.793 → 0.432). P1a3 added pen knob (nearly inert in arm S) and null-style fix. Side-by-side and font-search experiments showed B′ is best diffusion arm measured to date (0.188 VAL CER with fixed CFG, R_eval shows gap vs B0 legibility remains, style M2 MRR near B0).

**Open blockers for P1b:**
- Legibility: B′ R_eval CER still ~2× B0 on all words (~1.3× on ≥3-codepoint); locator-only evaluation was too weak.
- Pen: A′/B′ generate ~1.3× writer's pen (sub-px width/core); calibration post-hoc or wider canvas (128×512) proposed.
- Background: 45–48% of writer-117 words have faint blotches; white-paper norm (single scalar) insufficient.
- Content: Grapheme tokens working; spatial painted map inert.

**Bugs fixed:** CFG-null sampling (content in both branches), locator canvas architecture, vocab-size off-by-one, OOV gate sign inversion, gradient checkpointing.

---

**Report compiled from:** EXPERIMENTS.md, runs/probe_day1/REPORT.md, runs/P1a/REVIEW.md, runs/P1a2/REPORT.md, runs/P1a2/REVIEW.md, runs/compare_inference/README.md, runs/font_search/REPORT.md, wordgen/models/diffusion.py, wordgen/models/diffusion2.py, train_info.json files, metrics.csv files (2026-10-13 and 2026-10-19).
