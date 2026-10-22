# Project Goal, Motivation, Diagnosis, Data and Overall Methodology

## 1. Problem Statement and Motivation

### What is B0 (BengaliScribe Generator)?
- B0 is the baseline generator, the current implementation in BengaliScribe (`BengaliScribe/code/font_generator_pilot_sal.py`)
- A 4.6 M-param U-Net over a font image (PLAN.md §1.1) that generates Bengali handwritten words by transforming a font skeleton with learned style warping and style conditioning
- Trained on BN-HTR with `TRAIN_SUBSET=20000` crops for 20 epochs (PLAN.md §1.6, §6-T0)

### The Font-Skeleton Look Problem
B0's output visually looks like **thick, angular font text** rather than natural handwriting for three interconnected architectural reasons:

1. **Architecture preserves the font:** The U-Net keeps 1/4- and 1/2-resolution skip connections (`SKIP_SCALES=(1,1,0)`, PLAN.md §1.1) that pin construction to the font skeleton with 2-px precision. The only warp available is `StyleWarp` (`WARP_MAX=0.12`, PLAN.md §1.1), which gives at most **1.4 px of vertical and 5.8 px of horizontal displacement** on the 64×256 canvas (PLAN_REVIEW.md 0). Cross-attention to references is disabled (`XATTN=False`, PLAN.md §1.1).

2. **Losses reward an averaged font skeleton:** The loss (4·L1 + 1·LSGAN + 0.3·style-CE + 0.3·CTC, PLAN.md §1.2) is ink-weighted with salience on the matra and loops of **both the target and the font** (PLAN.md §1.2). When prior and target are misaligned beyond the warp's range, L1 optimization converges to a blurred stroke along the prior's skeleton — exactly the observed "font skeleton re-inked thick" (PLAN.md §1.2). The PatchGAN is also conditioned on the font prior (PLAN.md §1.2), so "looks real given this font image" further favours font-aligned strokes.

3. **Resolution and pen are dictated by training data, not by the writer:** The canvas is 64×256 px (PLAN.md §1.4). Measured stroke width / core height ratios (PLAN.md §1.4 table):
   - BN-HTR (generator's training data): 0.106 median (IQR 0.077–0.185), native h 74 px, darkness 0.98
   - image_2 (reference writer): **0.053 median** (0.048–0.067), native h 107 px, darkness 0.79
   - Image_2's pen is **below BN-HTR's 10th percentile**: out of distribution (PLAN.md §1.4)
   - At 64 px canvas, image_2's stroke is ~1.6 px, unrenderable with sharp edges (PLAN.md §1.4)
   - At page output (300 dpi, 0.39 mm generated vs 0.25 mm real, PLAN_REVIEW.md M10), **most visible thickness is added in integration** (upsampling, thresholding, pasting)

### Measured Evidence of the Problem
- Generated stroke width at page level: **0.39 mm vs 0.25 mm real** — **+55% thickness** (PLAN.md §1.4, notation [O])
- Generated darkness saturates: **0.93 generated vs 0.80 real** (PLAN.md §1.4, notation [O]), because `crop_ink` stretches every target so its 5th-percentile pixel equals ink 1.0 (PLAN.md §1.4)

---

## 2. Datasets

### BN-HTR (BN-HTR Dataset)
- **Total crops:** 92,380 labelled and filtered (PLAN.md §2.1, verified in runs/T0/REPORT.md section 2)
- **Total writers:** 149 (runs/T0/REPORT.md section 2)
- **Labels:** XLSX via xlsx reader; filtered by `label_filter.json` (PLAN.md §2.1)
- **Maximum grapheme constraint:** ≤16 graphemes, 93,758 → 92,380 after filtering (PLAN.md §1.6)
- **Native resolution:** median crop height 74 px (PLAN.md §1.4 table), median core ~28 px (runs/T0/REPORT.md section 5: p50=30 px)
- **Writer-disjoint split** (hardcoded from PLAN.md §2.2):
  - **TEST:** {15, 22, 24, 28, 30, 48, 51, 54, 69, 136} — **10 writers, 5,768 crops** (runs/T0/REPORT.md section 3). These writers are also in BengaliScribe's generator/locator held-out set (PLAN.md §2.2).
  - **VAL:** {99, 117, 127, 142, 149} — **5 writers** (runs/T0/REPORT.md section 3). Unseen by generator/locator, probably seen by bnstyle-v4 (PLAN.md §2.2, flagged [A]).
  - **TRAIN:** remaining 134 writers (92,350 ok crops, 83,329 after excluding measure_rows failures, runs/T0/REPORT.md section 3)

### BanglaWriting
- **Total crops:** 20,523 (PLAN.md §2.1, verified in runs/T0/REPORT.md section 2)
- **Total writers:** 260 (`style_id` = page = person `id_age_gender`, PLAN.md §2.1)
- **Labels:** manual, high confidence, from `Diffusion_model_dataset_building/dataset/labels.csv` (PLAN.md §2.1)
- **Native resolution:** median crop height 163 px (PLAN.md §1.4 table), median core ~49 px (runs/T0/REPORT.md section 5: p50=49 px)
- **Writer-disjoint split** (hash-sorted `style_id`, PLAN.md §2.2):
  - **TRAIN:** 220 writers (runs/T0/REPORT.md section 3)
  - **VAL:** 15 writers (runs/T0/REPORT.md section 3)
  - **TEST:** 25 writers (runs/T0/REPORT.md section 3)
  - Hash function: `key = int(hashlib.sha256(style_id.encode('utf-8')).hexdigest(), 16)`, sorted ascending (runs/T0/REPORT.md section 3)

### hcr/996 (Held-out Evaluation Reference)
- **Source:** `/home/hpc4090/shyan/ocr/hcr/996` — 995 pages, 27 texts (PLAN.md §2.1)
- **Scope:** eval-only few-shot references and targets; never used in training (PLAN.md §2.1)
- **Word pieces:** ~48.7k extracted from `scribe_layout_v2/cache/geometry` (PLAN.md §2.1)
- **Writer information:** unknown; page ≠ proven writer (flagged [A] in PLAN.md §2.1). **Open check (T1):** cluster pages with E_eval to determine one writer per page (PLAN.md §2.2).
- **Labels:** text only (no ground truth); needs alignment via T1 (PLAN.md §6-T1)
- **Native resolution:** A4 at 300 dpi, median core 41 px (PLAN.md §2.1)

### Fonts Available
- On disk locally: NotoSansBengali.ttf and free-serif.ttf only (PLAN.md §2.1)
- External fonts in `/home/hpc4090/nuhan/data/hcr_fp/GlyphScribe/bangla_fonts/`: 37 Bengali TTFs + 1 printed (Noto Serif Bengali, Hind Siliguri, Baloo Da 2, Atma, Tiro Bangla, etc.), but licences mixed and require user OK before use (PLAN_REVIEW.md M8)

---

## 3. Evaluation Methodology

### Normalisation (PLAN.md §2.3, runs/T0/REPORT.md §4.1)

**Core-normalised canvas:** parameterised by target `core_px` (default 32, measured at 48 pending review decision on thin pens):
- Height: 5 × core_px (160 px at core 32, 240 px at core 48, runs/T0/REPORT.md §4.1)
- Matra row: 2.25 × core_px (72 at core 32, 108 at core 48, runs/T0/REPORT.md §4.1)
- Base row: matra_row + core_px (104 at core 32, 156 at core 48, runs/T0/REPORT.md §4.1)
- Width: natural aspect ratio, scaled uniformly
- Ink: background-subtracted grey (not stretched to 1.0), so darkness remains a writer property (runs/T0/REPORT.md §4.1)

**Clipping (known issue, PLAN_REVIEW.md B1):**
- Core 32: BN-HTR TRAIN **22.62% clipped**, BanglaWriting **21.10% clipped** (runs/T0/REPORT.md §4.3)
- Core 48: BN-HTR TRAIN **22.67% clipped**, BanglaWriting **21.24% clipped** (runs/T0/REPORT.md §4.3)
- Clipping is overwhelmingly at the top (above matra): BN-HTR top-only 19.2% vs bottom-only 0.38% (runs/T0/REPORT.md §4.3)
- Root cause: `measure_rows` produces unreliable small-core estimates on ~20–25% of crops (runs/T0/REPORT.md §4.4). Mitigation (T0b): writer-prior fallback reduced BN-HTR TRAIN clip to **18.98%**, BanglaWriting to **14.57%** (runs/T0b/REPORT.md §3), still short of <1% target (flagged for Checkpoint A decision)

### Evaluator Models (PLAN.md §3)
Trained once on TRAIN writers (BN-HTR + BanglaWriting) and **never used as training loss**:

- **R_eval:** CRNN-CTC word recogniser at 64 px, with thickness/blur augmentation (PLAN.md §3)
  - VAL: real crops 0.037–0.043 CER depending on arms; fonts→real transfer better than fonts alone (runs/R_eval/RESULTS.md)
  - TEST: confirmed 0.034–0.043 (runs/R_eval/RESULTS.md)

- **E_eval:** ResNet-18 writer-ID model (SupCon) on core-normalised ink mattes (PLAN.md §3)
  - Real-vs-real top-1: 0.821 MRR on test writers (EXPERIMENTS.md current state, notation M2)
  - Chance: 0.28 (over 20 VAL writers in EXPERIMENTS.md protocol)

- **E_shape:** ResNet-18 on skeletons drawn at 1 px, construction-only (PLAN.md §3)
  - Untested for pen invariance; needs augmentation and spur pruning (PLAN_REVIEW.md M2c)

### Metrics (PLAN.md §3, EXPERIMENTS.md protocol)

**VAL Protocol (fixed, writer-disjoint):**
- **20 writers total:** 5 from BN-HTR (VAL set {99, 117, 127, 142, 149}) + 15 from BanglaWriting (data_cache/splits.json)
- **781 target words** total across both datasets (EXPERIMENTS.md protocol)
- **5 style references per writer**, sampled from the same writer's other pages, excluding the target crop (PLAN.md §3, EXPERIMENTS.md protocol)
- **Bootstrap for 95% CIs:** per-writer bootstrap (1000×) over the 20 VAL writers (PLAN.md §3)

**Primary metrics:**

| ID | Metric | Definition | Null | Ceiling |
|---|---|---|---|---|
| M1 | Readability | CER and WER from R_eval on generated words (PLAN.md §3) | font rendering's CER | real T crops (same writers) |
| M2 | Style ID (pen+construction) | top-1 writer-ID via E_eval: query = generated word, gallery = mean E_eval of G over N eval writers (PLAN.md §3) | chance 1/N (0.28 for N=20 in VAL protocol) | real-vs-real |
| M2s | Construction ID (pen-invariant) | M2 with E_shape (pen-stripped) (PLAN.md §3) | chance | real-vs-real |
| M3 | Stroke statistics | Per writer: W1 distance (generated vs real T) on width/core, darkness, corner rate (turn >60° within ±0.3 core), curvature, loops above matra per word (PLAN.md §3) | ratio 1 = as close as the writer's own other crops | — |
| M4 | Distribution | KID on E_eval features; generated-T vs real-T pooled over eval writers (PLAN.md §3) | real-vs-real split-half KID | — |
| M5 | Content-matched retrieval (hcr only) | For word w on page p: is generated w closer (E_eval/E_shape) to real w on page p than to real w on other ~36 pages of same text? (PLAN.md §3) | chance ≈1/37 | real w with different crop augmentation |
| M6 | Copy check | NCC of each generated word to nearest C crop vs same quantity for real T (PLAN.md §3) | real T distribution | flag if gen closer to C than T |
| M7 | Visual sheet | 6 writers × 8 words: refs, font, B0, B1, B3, candidate, real (PLAN.md §3) | — | — |

**Baselines:**
- **B0:** BengaliScribe font-GAN (NotoSans prior), VAL CER 0.130 [0.103, 0.156], M2 0.596 (EXPERIMENTS.md current state)
- **B1:** B0 + pen re-render with width/darkness estimated from C crops (PLAN.md §3)
- **B2:** real-vs-real (PLAN.md §3)
- **B3:** `dp_words_b64` DiffusionPen baseline (saw all 260 BW writers; BN-HTR only, hcr only, PLAN.md §3)
- **B4:** font rendering (PLAN.md §3)

---

## 4. Planned Phases, Arms and Decision Gates

### Phase Timeline (PLAN.md §6)

**T-1. Visible probe (~1 GPU-h, day 1 — PLAN_REVIEW.md M10)**
- B1 page for image_2: page-resolution pen re-render with width/darkness from C
- `dp_words_b64` sampled with image_2 and VAL-writer refs
- Per-writer fine-tune of that model on one VAL writer's C crops (TTA)

**T0 / T0b. Data index and splits** ✓ **complete**
- Output: `data_cache/index_{bnhtr,bw}.csv` with path, writer, page, text, native h/w, matra, base, core and clip flag; plus `data_cache/splits.json` (runs/T0/REPORT.md §1)
- Accept: counts 92,380/149 and 20,523/260 match, TEST lists verified, writer disjointness asserted, <1% clip rate **NOT met** (flagged for Checkpoint A, PLAN_REVIEW.md B1)

**T1. hcr alignment and writer clustering** *pending*
- Output: `data_cache/hcr_aligned.csv` (page, piece ids, token, confidence)
- Accept: ≥97% precision, E_eval page clustering to determine one writer per page (PLAN.md §2.2, §6-T1)

**T2. Evaluators** *pending*
- R_eval: CRNN-CTC on 64 px (PLAN.md §6-T2, runs/R_eval/RESULTS.md confirmed)
- E_eval: ResNet-18 SupCon on core-normalised mattes
- E_shape: ResNet-18 on skeletons at 1 px
- Training: R_eval ~1.5 GPU-h, E_eval/E_shape ~1 GPU-h each (PLAN.md §6-T2)

**T3. Metric suite and baselines** *pending*
- Code: `wordgen/eval/run_eval.py`, `wordgen/eval/strokes.py`, `wordgen/baselines/bscribe_words.py`
- Accept: identical runs (seeded), nulls table, M7 sheet (PLAN.md §6-T3)

**◆ Checkpoint A (Opus)** *pending*
- Writer leakage audit across all models (PLAN.md §6)
- Evaluator validity checks: font ≈ chance on M2, real ≫ chance, R_eval real CER reported (PLAN.md §6)
- Core-px decision (32 vs 48, VAE IoU gate ≥0.93 on thin-pen writers PLAN_REVIEW.md B1)
- Clip rate acceptance flagged for decision (PLAN_REVIEW.md B1)

**T4. Content map and writer width factor** *pending*
- Output: learned grapheme embeddings, writer width factor computed leave-target-out (PLAN.md §6-T4)
- Accept: VAE IoU ≥0.93 on VAL crops at chosen canvas, width factor correlation reported (PLAN.md §6-T4)

**T5. Diffusion pilots (P1a)** *in progress*
- **P1a (A, B):** 64×256, IAM warm start (NOT `dp_words_b64`, due to PLAN_REVIEW.md B2)
  - Arm A: control, no content map (configs/p1a_A.yaml)
  - Arm B: grapheme-token content map (configs/p1a_B.yaml)
  - **20000 steps, batch 32** (configs/p1a_A.yaml, configs/p1a_B.yaml)
- **P1a′ (A′, B′):** full no-cap TRAIN (98,856 crops), white-paper norm, 40k steps (runs/P1a2/REPORT.md)
- **P1a3 (P, S):** latest iteration with pen knob and ref tokens (EXPERIMENTS.md current state)
- Each logs VAL M1/M2/M3 snapshot per epoch (PLAN.md §6-T5)

**◆ Checkpoint B (Opus)** *pending*
- GO/NO-GO per PLAN.md §4 (revised in PLAN_REVIEW.md M5):
  - (i) M1 CER ≤ B0 CER + 0.05 absolute **OR** relative to real-T CER (PLAN_REVIEW.md M5)
  - (ii) M2 **AND** M2s top-1 > B1, bootstrap CI excluding 0 (PLAN.md §4, PLAN_REVIEW.md M5)
  - (iii) M3 corner-rate and width ratios beat B1 on **construction metrics only** (PLAN_REVIEW.md M5)
- M6 checked to rule out copying (PLAN_REVIEW.md M6)
- P1a NO-GO → redesign before P1b; P1b NO-GO → fallback

**T6. Full diffusion (or fallback GAN)** *pending*
- **On GO:** ~10 GPU-h full run at chosen core canvas (PLAN.md §6-T6)
- **On NO-GO:** T6′ fallback, approach (a)+(b): retrained font-prior GAN at core canvas (~5 GPU-h) plus approach (a) ablation alone (PLAN.md §4, §6-T6)

**T7. Pen renderer (e)** *pending*
- Post-process per writer: stroke-width quantile function, darkness, edge profile from C crops (PLAN.md §4 approach (e))
- Accept: M3 width/darkness ratio on VAL ≤ B1's, M1 not worse (PLAN.md §6-T7)

**T8. Test-time adaptation (d)** *pending*
- LoRA on attention/last blocks, fitted to target writer's C crops (PLAN.md §4 approach (d))
- Accept: leave-crop-out protocol, M6 checked, M2s improves without CER rising >0.03 (PLAN.md §6-T8)

**T9. Integration** *pending*
- Output: 300-dpi pages via `wordgen/integrate/run_page.py`, items contract (PLAN.md §5, §6-T9)
- Accept: comparison sheet for image_2 and 3 hcr TEST pages, BengaliScribe tree unchanged (PLAN.md §6-T9)

**◆ Checkpoint C (Opus)** *pending*
- Single TEST evaluation: final configuration against B0–B4 with bootstrap CIs (PLAN.md §6)
- `reports/RESULTS.md`: every number with its command (PLAN.md §6)

### Budget
- Total: ~24 GPU-h (PLAN_REVIEW.md M9: T2 ~3.5h + T5 ~4h + T6 ~10–5h + T8 ~1–2h)

---

## 5. Candidate Figures (Existing PNG Paths)

| Path | Caption |
|---|---|
| `/home/hpc4090/shyan/ocr/scribe_wordgen/runs/T0/sheet.png` | Core-normalised canvas on 50 BN-HTR and BanglaWriting crops at core 32 px: original crop (left) vs normalised canvas (right) with matra (red) and baseline (blue) overlays; shows clipping failures and measure_rows robustness. |
| `/home/hpc4090/shyan/ocr/scribe_wordgen/runs/T0b/sheet.png` | T0b robustness pass: 60 crops showing original \| v1 canvas \| v2 canvas (with writer-prior fallback), demonstrating clipping reduction on degenerate-core crops. |
| `/home/hpc4090/shyan/ocr/scribe_wordgen/runs/probe_day1/sheet.png` | Probe result: DiffusionPen (`dp_words_b64`) on image_2 text (24 words × 3 seeds) vs real pages; shows handwriting-like texture but conjunct misspelling. |
| `/home/hpc4090/shyan/ocr/scribe_wordgen/runs/probe_day1/line.png` | Single-writer fine-tune result (TTA) for image_2 with probe model; side-by-side real vs generated line. |
| `/home/hpc4090/shyan/ocr/scribe_wordgen/runs/P1a2/sheet_val.png` | P1a′ arm A′ (control, no content map) and B′ (grapheme tokens in cross-attn) on 20 VAL writers; shows content map improves CER (0.245 → 0.188). |
| `/home/hpc4090/shyan/ocr/scribe_wordgen/runs/P1a2/sheet_image2.png` | P1a′ on image_2: A′, B′, real crops side-by-side; image_2 CER 0.17 (B0) vs 0.34–0.40 (arms). |
| `/home/hpc4090/shyan/ocr/scribe_wordgen/runs/eval_v3/sheet_val.png` | Latest arms (P, S, B0, real) on 20 VAL writers; M1 CER tied S/B0, R_eval shows S less legible (0.089 vs 0.030). |
| `/home/hpc4090/shyan/ocr/scribe_wordgen/runs/eval_v3/sheet_image2.png` | Arms comparison on image_2; S pen closer to real than B0 but legibility gap under R_eval. |
| `/home/hpc4090/shyan/ocr/scribe_wordgen/runs/style_eval/sheet_prep.png` | E_eval and E_shape evaluator prep: training samples (real crops) for the two ResNet-18 models. |
| `/home/hpc4090/shyan/ocr/scribe_wordgen/runs/style_eval/sheet_m2_ci.png` | M2 MRR results: real-vs-real 0.821, B0 0.596, all diffusion arms 0.40–0.45; with 95% CIs per writer. |
| `/home/hpc4090/shyan/ocr/scribe_wordgen/runs/font_search/sheet.png` | Google OFL Bengali fonts scored against image_2 skeleton (12 words); Atma/Hind Siliguri best, thin fonts (Qayyum Thin 0.068) closer to writer pen than NotoSans (0.197). |
| `/home/hpc4090/shyan/ocr/scribe_wordgen/runs/R_eval/smoke_errors.png` | R_eval recogniser error examples on synthetic (fonts-only), real, and fonts→real models; shows failure modes (punctuation, digit confusion). |
| `/home/hpc4090/shyan/ocr/scribe_wordgen/runs/R_eval/synth_sheet.png` | R_eval synthetic training data: rendered words from 48 Bengali fonts (37 GlyphScribe + 11 Google) used as pretraining. |
| `/home/hpc4090/shyan/ocr/scribe_wordgen/runs/pen_calib/sheet_Bp_t5c5_pencal_shrink_v2.png` | Pen calibration test on arm B′: post-hoc page-scale pen re-render (shrink mode) reducing width from 1.30 to 1.06 relative to real. |
| `/home/hpc4090/shyan/ocr/scribe_wordgen/runs/compare_inference/image2_side_by_side.png` | Side-by-side generators on image_2 (24 words × 3 seeds): B0, B′, A′, plain Atma font; CER 0.17 / 0.34 / 0.40 / 0.08. |
| `/home/hpc4090/shyan/ocr/scribe_wordgen/runs/bgnorm2/sheet_thinpen.png` | Background normalisation (bgnorm2) test on thin-pen writers (lowest width/core quartile); white-paper removal fixes writer-117 blotch artefact (30% → 0%). |

---

## Appendix: Key Definitions and Terminology

- **Core (px):** Measured matra bar height in pixels, a writer's stable pen size; normalized to core_px=32 (or 48) for canvas scaling (PLAN.md §2.3, runs/T0/REPORT.md §4.1)
- **Matra/baseline:** Rows separating ascending strokes (loop region), main body, and descending strokes (runs/T0/REPORT.md §4.1)
- **Clip rate:** Percentage of crops where ink extends beyond the canvas headroom budget (PLAN.md §2.3, runs/T0/REPORT.md §4.3)
- **measure_rows:** BengaliScribe function identifying matra peak and core from ink histogram per crop (PLAN.md §2.3, runs/T0/REPORT.md §4.4)
- **Writer-disjoint:** TRAIN/VAL/TEST splits assigned by writer, not by crop, ensuring no writer appears in two splits (PLAN.md §2.2, runs/T0/REPORT.md §3)
- **Content map:** Learned grapheme-cluster embedding painted into each grapheme's column span at latent resolution, providing spatial identity without forcing font-skeleton alignment (PLAN.md §4 approach (c), configs/p1a_B.yaml)
- **Pen knob:** Multiplicative scale on generated stroke width, tested as post-hoc control (EXPERIMENTS.md current state)
- **Ref tokens:** Reference crop spatial feature tokens added to cross-attention, enabling spatial style conditioning (EXPERIMENTS.md current state, PLAN_REVIEW.md M1)
- **TTA (test-time adaptation):** LoRA fine-tuning on target writer's reference crops at inference time (PLAN.md §4 approach (d), PLAN.md §6-T8)
