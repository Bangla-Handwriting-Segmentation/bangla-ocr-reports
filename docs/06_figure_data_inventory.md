# Figure and Data Inventory: scribe_wordgen/runs/

**Scope:** Complete inventory of summary figures and plottable data across all experiment folders in `runs/`, excluding cache/, eval_cache/, and per-word image caches. Excludes: rescore_v2, S_bestofN, S_ocrloss (in-progress experiments).

**Generated:** 2026-10-19

---

## Summary Statistics

- **Total experiment folders:** 15
- **Summary images (PNG/JPG/PDF):** ~200 files
- **Data files (CSV/JSON/TSV):** ~500+ files
- **Large images flagged for downscaling:** 12 files (>6000px dimension or >5MB)
- **Key chart-ready datasets:** R_eval/, style_eval/, pen_calib/, P1a*/

---

## 1. bgnorm2 (14 files) — Brightness Normalization Study

### Images (6 files)

| Path | Dimensions | Size | Description |
|------|-----------|------|-------------|
| `sheet_117_refs.png` | 942x1078 | 0.69 MB | Real handwritten word samples from writer 117 reference set |
| `sheet_darkest_train.png` | 942x1428 | 0.66 MB | Generated words with darkest pixel intensity normalization |
| `sheet_gen_bnhtr_117.png` | 786x1632 | 0.24 MB | Generated words for writer 117 with brightness normalization applied |
| `sheet_gen_bw_74_13_0.png` | 786x1632 | 0.14 MB | Generated words for writer 74 entry 13 with normalization |
| `sheet_thinpen.png` | 942x1428 | 0.25 MB | Generated words with thin pen intensity profile |
| `sheet_worst_mass.png` | 942x1148 | 0.59 MB | Generated words showing worst-case pixel mass distribution |

### Data Files (8 files)

| Path | Rows/Entries | Key Columns | Measures |
|------|--------------|-------------|----------|
| `measure_rows.csv` | 3,938 | method, writer, metric_name, value, ... | Pixel mass, intensity, stroke properties across methods |
| `measure.json` | 10 entries | Keys: min_mass, max_mass, darkest_intensity, ... | Measurement statistics summary |
| `bprime_rows.csv` | 3,124 | method, writer, idx, text, norm_value | Normalized brightness values per word |
| `bprime_rows_raw.csv` | 1,562 | method, writer, raw_value | Raw brightness before normalization |
| `bprime_noretrain.json` | 8 entries | Skip retraining analysis | VAE model state without retraining |
| `vae_rows.csv` | 1,490 | method, epoch, loss, reconstruction_error | Training loss curves for VAE models |
| `vae.json` | 5 entries | Model hyperparameters | VAE configuration summary |
| `metrics.json` | 5 entries | cer, wer, ncc_score | Evaluation metrics summary |

---

## 2. compare_inference (3 files) — Inference Speed / Quality Comparison

### Images (2 files)

| Path | Dimensions | Size | Description |
|------|-----------|------|-------------|
| `image2_side_by_side.png` | 1942x2110 | 0.98 MB | Side-by-side inference results on image_2 test set |
| `val_side_by_side.png` | 1726x2046 | 0.99 MB | Side-by-side inference results on validation split |

### Data Files (1 file)

| Path | Rows | Key Columns | Measures |
|------|------|-------------|----------|
| `image2_scores.csv` | 297 | method, writer, idx, cer, inference_time_ms | Comparison of inference time vs. accuracy |

---

## 3. eval_v3 (5 files) — Comprehensive Multi-Method Evaluation

### Images (2 files)

| Path | Dimensions | Size | Description |
|------|-----------|------|-------------|
| `sheet_val.png` | 2238x3096 | 1.02 MB | **Chart-ready:** Validation sheet comparing 8 model variants (B0, B3, Ap_g3, Bp_t3c3, etc.) across 50+ validation words with real handwritten reference |
| `sheet_image2.png` | 1656x1750 | 0.33 MB | Test set comparison on image_2 document |

### Data Files (3 files)

| Path | Rows/Entries | Key Columns | Measures |
|------|--------------|-------------|----------|
| `gen_rows.csv` | 17,182 | writer, idx, text, method, cer, ncc_score | **Chart-ready:** Per-word evaluation results; supports CER/NCC plots by writer/method |
| `metrics.json` | 4 entries | cer, wer, ncc_per_method | Summary metrics; good for table |
| `train_vocab_v2.json` | 16,034 entries | word→id mapping | Character and word vocabulary |

---

## 4. font_search (30 files) — Font Selection via Real/Synthetic Matching

### Images (15 files)

| Path | Dimensions | Size | Description |
|------|-----------|------|-------------|
| `sheet.png` | 2440x1142 | 0.75 MB | Comparison of 12 font candidates vs. real handwriting |
| `sheets/shaping_check.png` | 2600x1440 | 0.39 MB | Glyph shaping validation grid |
| `glyphscribe/sheet.png` | 3360x1142 | 1.09 MB | GlyphScribe font candidates comparison |
| `glyphscribe/shaping_check.png` | 2600x4680 | 1.53 MB | **FLAGGED:** Large dimension (2600x4680); consider 50% downscale for PDF |
| `crops/*.png` (12 files) | 200-400 px | 0.01-0.02 MB each | Real word crops (ঊনিশে, কার্তিক, ব্যাংকের, etc.) used for matching |
| `fonts_glyphscribe/backgrounds/*.{png,jpg}` (3 files) | 200-300 px | 0.01-0.02 MB | Background textures for synthetic generation |

### Data Files (15 files)

| Path | Rows | Key Columns | Measures |
|------|------|-------------|----------|
| `scores.csv` | 12 | font_name, cer, ncc_score, writer | **Chart-ready:** Font ranking by CER/NCC |
| `scores_long.csv` | 144 | font_name, writer, idx, text, cer | Per-word per-font scores; supports writer breakdown charts |
| `glyphscribe/scores.csv` | 39 | font_name, variant, cer, ncc | Alternative font set evaluation |
| `glyphscribe/scores_long.csv` | 468 | font_name, variant, writer, idx, cer | Per-word variant scores |
| `fonts.csv` | 36 | font_name, family, style, available | Font metadata and availability |
| `real_crops_measure.csv` | 12 | word_idx, width, height, pixel_area | Real word dimensions |
| `glyphscribe/real_crops_measure.csv` | 12 | Same structure | Measurement comparison |
| `writer_pen.json` | 2 | pen_width, pen_angle | Pen configuration for writer |
| `glyphscribe/writer_pen.json` | 2 | Same | Alternative pen config |
| `crops/words.json` | 12 | word_id→{text, writer} | Crop metadata |

---

## 5. P1a (9 files) — Phase 1a Baseline Model Training

### Images (2 files)

| Path | Dimensions | Size | Description |
|------|-----------|------|-------------|
| `sheet_val.png` | 1420x2738 | 1.30 MB | **Chart-ready:** Validation comparison of methods B0, B3, A, B on 50+ words with real reference |
| `sheet_image2.png` | 1390x2200 | 0.69 MB | Test set results on separate document |

### Data Files (7 files)

| Path | Rows/Entries | Key Columns | Measures |
|------|--------------|-------------|----------|
| `metrics_raw.csv` | 3,905 | dataset, writer, method, idx, text, seed, cer, ncc | **Chart-ready:** Per-word evaluation; writer/method breakdowns |
| `metrics.json` | 3 | n_val_writers, n_words_total, methods | Summary counts |
| `A/metrics.csv` | 20,000 | epoch, loss, val_loss, cer | **Chart-ready:** Training curve for method A |
| `B/metrics.csv` | 20,000 | epoch, loss, val_loss, cer | Training curve for method B |
| `A/train_info.json` | 12 | Hyperparameters, config | Training configuration |
| `B/train_info.json` | 12 | Hyperparameters, config | Training configuration |
| `A/grapheme_vocab.json` | 914 | grapheme→id | Grapheme vocabulary (Bengali script) |
| `B/grapheme_vocab.json` | 914 | grapheme→id | Grapheme vocabulary |

---

## 6. P1a2 (13 files) — Phase 1a2 Extended Training

### Images (2 files)

| Path | Dimensions | Size | Description |
|------|-----------|------|-------------|
| `sheet_val.png` | 1420x2738 | 1.22 MB | Validation sheet with real vs. generated word comparisons |
| `sheet_image2.png` | 1390x2200 | 0.60 MB | Test split results |
| `bgnorm_sheet.png` | 700x1584 | 0.65 MB | Brightness normalization impact visualization |

### Data Files (10 files)

| Path | Rows/Entries | Key Columns | Measures |
|------|--------------|-------------|----------|
| `eval_v2_rows.csv` | 3,905 | dataset, writer, idx, text, cer, ncc | Per-word evaluation v2 |
| `eval_v2_validate_rows.csv` | 3,905 | Same | Validation split v2 |
| `metrics_raw.csv` | 3,905 | Same as P1a | Per-word metrics |
| `metrics.json` | 5 | cer, wer, ncc_per_method | Summary |
| `shuffled_content.csv` | 144 | train/val/test split info | Dataset shuffle metadata |
| `A/metrics.csv` | 40,000 | epoch, loss, val_loss | Longer training curves than P1a |
| `B/metrics.csv` | 40,000 | epoch, loss, val_loss | Extended training |
| `A/train_info.json` | 13 | Hyperparameters | Config A |
| `B/train_info.json` | 13 | Hyperparameters | Config B |
| `A/grapheme_vocab.json` | 1,300 | grapheme→id | Extended grapheme set |
| `B/grapheme_vocab.json` | 1,300 | grapheme→id | Extended grapheme set |

---

## 7. P1a3 (113 files) — Phase 1a3 Advanced Modeling & Calibration

### Summary Images (3 files)

| Path | Dimensions | Size | Description |
|------|-----------|------|-------------|
| `eval/sheet_val.png` | 1992x3096 | 0.63 MB | Comprehensive validation sheet across S, P, and variants |
| `eval/diag_bg_P_raw.png` | 1536x384 | 0.15 MB | Background diagnostic for P method |

### Evaluation Sweep Images (80+ files)

| Path | Dimensions | Size | Type |
|------|-----------|------|------|
| `eval/sweep_png/S/bnhtr_117_00_কোন_x0.6.png` | 256x64 | 0.01 MB | Scale sweep 0.6x (14 writers × 2 scales × 2 dirs = 56 files) |
| `eval/sweep_png/P/bnhtr_117_00_কোন_x1.6.png` | 256x64 | 0.01 MB | Scale sweep 1.6x (similar structure) |
| Subtotal | — | — | Systematic scale variation (0.6x, 1.0x, 1.6x) for robustness analysis |

### Data Files (20+ files)

| Path | Rows/Entries | Key Columns | Measures |
|------|--------------|-------------|----------|
| `eval/gen_rows.csv` | 18,744 | method, scale, writer, idx, text, cer, ncc | **Chart-ready:** Scale sweep results; CER vs. scale factor plot |
| `eval/sweep.csv` | 1,600 | scale_factor, method, metric | Summary sweep results |
| `eval/calib_gains.csv` | 240 | method, gain_type, writer, value | Calibration gain factors |
| `eval/calib_global_ratio.json` | 6 | ratio statistics | Global normalization ratio |
| `eval/writer_cond.csv` | 60 | writer, condition, parameter | Per-writer conditions |
| `eval/metrics.json` | 9 entries | cer, wer, ncc, per_method_stats | Summary evaluation metrics |
| `eval/bgfail_diag.csv` | 7,260 | method, writer, failure_mode, count | Background failure diagnostics |
| `S/metrics.csv` | 24,000 | epoch, loss, validation metrics | **Chart-ready:** Training curve for S variant (largest) |
| `P/metrics.csv` | 16,000 | epoch, loss, validation metrics | Training curve for P variant |
| `S_diverged_v1/metrics.csv` | 1,501 | epoch, loss | Diverged training run v1 |
| `S_diverged_v2/metrics.csv` | 1,051 | epoch, loss | Diverged training run v2 |
| `style_enc/metrics.csv` | 8,000 | epoch, loss | Style encoder training |
| `*/writer_weights.csv` (7 files) | 354 rows each | writer_id, weight | Per-writer importance weights across model variants |
| `*/config.json` (7 files) | 19-23 entries | Hyperparameters | Model configurations for S, P, S_diverged variants |

---

## 8. page_inference (7 files) — Full-Page Generation

### Images (4 files)

| Path | Dimensions | Size | Description |
|------|-----------|------|-------------|
| `B0_learned_w0_l0_page01.png` | 1240x1754 | 0.15 MB | Generated full page (B0 baseline model) |
| `Bp_learned_w0_l0_page01.png` | 1240x1754 | 0.17 MB | Generated full page (Bp tuned model) |
| `Bp_learned_w0_l1_page01.png` | 1240x1754 | 0.17 MB | Generated full page (Bp with line embedding) |
| `compare_w0_l0.png` | 4020x1450 | 0.59 MB | Side-by-side page comparison (3 variants) |

### Data Files (3 files)

| Path | Entries | Measures |
|------|---------|----------|
| `B0_learned_w0_l0_page01.json` | 4 | Page layout, line positions, inference log |
| `Bp_learned_w0_l0_page01.json` | 4 | Page layout, line positions, inference log |
| `Bp_learned_w0_l1_page01.json` | 4 | Page layout, line embedding info |

---

## 9. page_sheets (138 files) — Page-Level Synthesis Evaluation

### Page Renderings (40+ files)

| Path | Dimensions | Size | Description |
|------|-----------|------|-------------|
| `pages/image_2/B0_l0_page01.png` | 2480x3508 | 0.42 MB | Full-page synthesis for image_2 test |
| `pages/bw_10_20_0/Bp_t5c5_l0_page01.png` | 2480x3508 | 0.40 MB | Full-page synthesis for writer bw_10_20_0 |
| `pages/bnhtr_99/S_t5c5_cal_l0_page01.png` | 2480x3508 | 0.50 MB | Full-page synthesis with calibration |
| `dryrun/pages/*/*.png` (10 files) | 1654x2339 | 0.12-0.24 MB | Lower-res dryrun versions |
| Subtotal | — | — | 40 page-level PNG files × 4-5 model variants |

### Composited Sheet Views (12 files) — **FLAGGED FOR DOWNSCALING**

| Path | Dimensions | Size | Flag | Recommendation |
|------|-----------|------|------|-----------------|
| `sheets/bnhtr_99_sheet_l0.png` | **12100x1490** | 1.69 MB | DIMENSION | Downscale to 50% width (6050x745) for PDF |
| `sheets/bw_10_20_0_sheet_l0.png` | **13153x1490** | 1.64 MB | DIMENSION | Downscale to 50% width (6577x745) for PDF |
| `sheets/image_2_sheet_l0.png` | **12100x1490** | 1.34 MB | DIMENSION | Downscale to 50% width (6050x745) for PDF |
| `sheets/bw_101_21_0_sheet_l0.png` | **12739x1490** | 1.08 MB | DIMENSION | Downscale to 50% width (6370x745) for PDF |
| `sheets/bnhtr_99_zoom_l0.png` | **2074x7457** | 1.46 MB | DIMENSION | Downscale to 65% (1348x4847) for PDF |
| `sheets/bw_10_20_0_zoom_l0.png` | **2078x8022** | 1.54 MB | DIMENSION | Downscale to 65% (1351x5214) for PDF |
| `sheets/bw_101_21_0_zoom_l0.png` | **2088x7960** | 1.50 MB | DIMENSION | Downscale to 65% (1357x5174) for PDF |
| `sheets/image_2_zoom_l0.png` | **2100x7150** | 1.47 MB | DIMENSION | Downscale to 65% (1365x4648) for PDF |
| `dryrun/sheets/bw_101_21_0_sheet_l0.png` | **10719x1490** | 0.75 MB | DIMENSION | Downscale to 50% width (5360x745) for PDF |
| `dryrun/sheets/image_2_sheet_l0.png` | **10080x1490** | 1.17 MB | DIMENSION | Downscale to 50% width (5040x745) for PDF |
| `dryrun/sheets/bw_101_21_0_zoom_l0.png` | **1422x4341** | 0.72 MB | DIMENSION | Downscale to 65% (925x2822) for PDF |
| `dryrun/sheets/image_2_zoom_l0.png` | **1358x3993** | 0.76 MB | DIMENSION | Downscale to 65% (883x2595) for PDF |

### Page Layout Data (32+ files)

| Path | Entries | Type | Measures |
|------|---------|------|----------|
| `pages/*/B0_l0_page01.json` (8 files) | 4 | JSON | Page layout, line positions, bounding boxes |
| `pages/*/Bp_t3c3_l0_page01.json` (8 files) | 4 | JSON | Tuned model layout |
| `pages/*/Bp_t5c5_l0_page01.json` (8 files) | 4 | JSON | Extended tuning layout |
| `pages/*/S_t5c5_cal_l0_page01.json` (8 files) | 4 | JSON | Style-calibrated layout |
| `dryrun/pages/*/config.json` (4 files) | Variable | JSON | Dryrun configuration |

---

## 10. pen_calib (31 files) — Pen Parameter Calibration

### Calibration Study Images (7 files)

| Path | Dimensions | Size | Description |
|------|-----------|------|-------------|
| `sheet_Bp_t5c5_pencal_writer.png` | 968x3010 | 0.35 MB | Writer-level calibration results sheet |
| `sheet_Bp_t5c5_pencal_writer_v2.png` | 968x3010 | 0.35 MB | Writer calibration v2 |
| `sheet_Bp_t5c5_pencal_global_v2.png` | 968x3010 | 0.34 MB | Global pen parameter calibration |
| `sheet_Bp_t5c5_pencal_shrink_v2.png` | 968x3010 | 0.34 MB | Shrink variant pen calibration |
| `sheet_Bp_t5c5_pencal_word_v2.png` | 968x3010 | 0.34 MB | Per-word pen calibration |
| `page_image2/compare_w0_l0.png` | 3010x1450 | 0.47 MB | Page comparison (3-way) |
| `page_image2/compare_w0_l0_shrink.png` | 3010x1450 | 0.47 MB | Page comparison (shrink variant) |
| `page_image2/compare_w0_l0_writer.png` | 3010x1450 | 0.47 MB | Page comparison (writer variant) |

### Calibration Metrics (12 files)

| Path | Rows | Key Columns | Type |
|------|------|-------------|------|
| `scores_Bp_t5c5_pencal_*.csv` (5 files) | 3,905 each | writer, idx, text, cer, ncc, pen_params | **Chart-ready:** CER by calibration method; per-writer breakdown |
| `gains_Bp_t5c5_pencal_*.csv` (5 files) | 40 each | gain_type, value, writer | Gain improvement factors |
| `gain_targets_Bp_t5c5_pencal_*.csv` (2 files) | 20 each | target_param, value | Target calibration parameters |
| `metrics_Bp_t5c5_pencal_*.json` (5 files) | 4 entries each | cer, wer, ncc, per_writer_stats | Summary metrics |

### Page Synthesis Data (12 files)

| Path | Entries | Type | Measures |
|------|---------|------|----------|
| `page_image2/Bp_pencal_*_page01.json` (6 files) | 4 | JSON | Page layout with pen calibration |
| (PNG counterparts) | — | PNG | Page renderings |

---

## 11. probe_day1 (108 files) — Daily Probe / Real Data Annotation

### Summary Sheet (2 files)

| Path | Dimensions | Size | Description |
|------|-----------|------|-------------|
| `sheet.png` | 1390x2200 | 1.66 MB | Real handwritten page from day 1 probe |
| `line.png` | 2476x406 | 0.62 MB | Single line extraction and comparison |

### Per-Writer Generated Samples (96 files)

| Path | Dimensions | Count | Description |
|------|-----------|-------|-------------|
| `dp_raw/writer_99/*.png` | 256x64 | 24 | Generated words for writer 99 |
| `dp_raw/writer_117/*.png` | 256x64 | 24 | Generated words for writer 117 |
| `dp_raw/finetuned/*.png` | 256x64 | 24 | Finetuned model outputs |
| `dp_raw/image_2/*.png` | 256x64 | 24 | Alternative image set |

### Metadata & Metrics (8 files)

| Path | Rows | Columns | Measures |
|------|------|---------|----------|
| `align.json` | 8 entries | line_id, top, bottom, confidence | Line detection and alignment |
| `metrics.json` | 6 entries | cer, ncc, writer_count | Summary metrics |
| `tta_image_2/metrics.csv` | 648 | writer, idx, text, cer, prediction | Test-time augmentation results |
| `tta_image_2/labels.csv` | 17 | line_id, text, writer | Line labels |
| `tta_image_2/tta_timing.json` | 3 | total_time, per_word_time | Timing stats |
| `dp_raw/finetuned/copy_check.json` | 24 | entry_id, match_status | Data integrity verification |
| `dp_raw/*/timing.json` (3 files) | 5 entries each | generation_time, per_word | Timing for each writer |
| `dp_raw/*/_refs_used.json` (3 files) | 5 entries | ref_text, writer, weight | Reference word selection |

---

## 12. R_eval (24 files) — Recognizer Evaluation (OCR Performance)

### Error Diagnostics (8 files)

| Path | Dimensions | Size | Description |
|------|-----------|------|-------------|
| `errors_R_real_val.png` | 700x1040 | 0.03 MB | OCR errors on real handwritten validation set |
| `errors_R_font0_val.png` | 700x1040 | 0.05 MB | OCR errors on font0 synthetic validation |
| `errors_R_font2real_val.png` | 700x1040 | 0.04 MB | OCR errors on font2real synthetic validation |
| `errors_R_mix_val.png` | 700x1040 | 0.04 MB | OCR errors on mixed real/synthetic validation |
| `errors_R_real_10_val.png` | 700x1040 | 0.04 MB | OCR errors on real training subset (10 words) |
| `errors_R_font2real_10_val.png` | 700x1040 | 0.04 MB | OCR errors on font2real subset |
| `errors_R_font2real_test.png` | 700x1040 | 0.03 MB | OCR errors on font2real test set |
| `errors_R_real_test.png` | 700x1040 | 0.03 MB | OCR errors on real test set |
| `synth_sheet.png` | 532x4224 | 0.23 MB | Synthetic word generation examples |
| `smoke_errors.png` | 700x1040 | 0.04 MB | Smoke test error subset |

### Results CSVs (8 files) — **Chart-Ready**

| Path | Rows | Key Columns | Measures |
|------|------|-------------|----------|
| `results_R_real_val.csv` | 4,299 | text, pred, writer, dataset, cer, ncc | **Chart:** Real validation CER by writer |
| `results_R_font0_val.csv` | 4,299 | text, pred, writer, dataset, cer, ncc | Font0 synthetic validation results |
| `results_R_font2real_val.csv` | 4,299 | text, pred, writer, dataset, cer, ncc | Font2Real synthetic validation results |
| `results_R_mix_val.csv` | 4,299 | text, pred, writer, dataset, cer, ncc | Mixed training results |
| `results_R_real_test.csv` | 7,665 | text, pred, writer, dataset, cer, ncc | Real test set (larger; final evaluation) |
| `results_R_font2real_test.csv` | 7,665 | text, pred, writer, dataset, cer, ncc | Font2Real test set results |
| `results_R_real_10_val.csv` | 4,299 | Same | Limited real word subset |
| `results_R_font2real_10_val.csv` | 4,299 | Same | Limited font2real subset |

### Summary Data (6 files)

| Path | Rows | Key Columns | Measures |
|------|------|-------------|----------|
| `rescore.csv` | 35,926 | method, dataset, writer, idx, cer_old, cer_new | Rescoring before/after optimization |
| `smoke_eval_results.csv` | 600 | writer, idx, text, cer | Smoke test subset (quick validation) |
| `smoke_eval_results2.csv` | 20 | writer, text, cer | Minimal smoke test |
| `smoke_rescore.csv` | 50 | Same as rescore | Rescore smoke test |
| `selection.json` | 2 | train_set, test_set | Dataset selection log |
| `vocab_synth.tsv` | 202,561 | word_id, text, frequency | Synthetic vocabulary (large; supports frequency analysis) |
| `font_cmap_cache.json` | Variable | font_id→cmap | Font character mapping cache |

---

## 13. style_eval (6 files) — Style Consistency Evaluation

### Summary Images (2 files)

| Path | Dimensions | Size | Description |
|------|-----------|------|-------------|
| `sheet_prep.png` | 1730x1606 | 0.29 MB | Preprocessing pipeline visualization |
| `sheet_m2_ci.png` | 1320x770 | 0.11 MB | Method comparison with confidence intervals (m2) |

### Evaluation Data (4 files)

| Path | Rows | Key Columns | Measures |
|------|------|-------------|----------|
| `m6_ncc.csv` | 9,572 | method, writer, idx, text, ncc_score, ci_lower, ci_upper | **Chart-ready:** NCC scores by writer with confidence intervals |
| `protocol_items.csv` | 1,570 | writer, item_id, style_attribute, value | Protocol items for style assessment |
| `metrics.json` | 8 entries | m1_cer, m2_cer_ci, m6_ncc_mean | Per-method summary statistics |
| `E_shape_v1_nopenaug/metrics_with_v1.json` | 8 entries | Same | Alternative configuration results |

---

## 14. T0 (1 file) — Baseline Full-Page Generation

### Image (1 file)

| Path | Dimensions | Size | Description |
|------|-----------|------|-------------|
| `sheet.png` | **942x8408** | 1.45 MB | **FLAGGED:** Very tall page (~8400px); downscale to 50% height for PDF (942x4204) |

---

## 15. T0b (1 file) — Extended Baseline Full-Page

### Image (1 file)

| Path | Dimensions | Size | Description |
|------|-----------|------|-------------|
| `sheet.png` | **1162x10118** | 2.20 MB | **FLAGGED:** Very tall page (~10100px); downscale to 50% height for PDF (1162x5059) |

---

## Summary: Images Flagged for PDF Downscaling

**Total flagged: 12 files**

### Strategy:
- **Wide sheets (>6000px width):** Downscale to 50% width; preserve height
- **Tall pages (>8000px height):** Downscale to 50% or 65% height; preserve width
- **Large file size (>5MB):** Currently none exceed 5MB; largest is 2.2MB

### Specific recommendations:

| File | Current | Recommended | Ratio |
|------|---------|-------------|-------|
| T0b/sheet.png | 1162×10118 | 1162×5059 | 50% height |
| T0/sheet.png | 942×8408 | 942×4204 | 50% height |
| page_sheets/sheets/bw_10_20_0_sheet_l0.png | 13153×1490 | 6577×745 | 50% width |
| page_sheets/sheets/bnhtr_99_sheet_l0.png | 12100×1490 | 6050×745 | 50% width |
| page_sheets/sheets/image_2_sheet_l0.png | 12100×1490 | 6050×745 | 50% width |
| page_sheets/sheets/bw_101_21_0_sheet_l0.png | 12739×1490 | 6370×745 | 50% width |
| page_sheets/sheets/bw_10_20_0_zoom_l0.png | 2078×8022 | 1351×5214 | 65% |
| page_sheets/sheets/bnhtr_99_zoom_l0.png | 2074×7457 | 1348×4847 | 65% |
| page_sheets/sheets/image_2_zoom_l0.png | 2100×7150 | 1365×4648 | 65% |
| page_sheets/sheets/bw_101_21_0_zoom_l0.png | 2088×7960 | 1357×5174 | 65% |
| page_sheets/dryrun/sheets/bw_101_21_0_sheet_l0.png | 10719×1490 | 5360×745 | 50% width |
| page_sheets/dryrun/sheets/image_2_sheet_l0.png | 10080×1490 | 5040×745 | 50% width |
| font_search/glyphscribe/shaping_check.png | 2600×4680 | 1800×3224 | 69% |

---

## Summary: Data Files Recommended for Charts

### Training Curves (Loss/CER over epochs)
- `P1a/A/metrics.csv` (20k rows) — Method A training
- `P1a/B/metrics.csv` (20k rows) — Method B training
- `P1a2/A/metrics.csv` (40k rows) — Extended A training
- `P1a2/B/metrics.csv` (40k rows) — Extended B training
- `P1a3/S/metrics.csv` (24k rows) — Style model training
- `P1a3/P/metrics.csv` (16k rows) — Phonetic model training
- `bgnorm2/vae_rows.csv` (1.5k rows) — VAE training

### Per-Writer CER Breakdowns
- `R_eval/results_R_real_val.csv` (4.3k rows) — Real validation; plot CER histogram by writer
- `R_eval/results_R_real_test.csv` (7.7k rows) — Real test set (final)
- `pen_calib/scores_Bp_t5c5_pencal_*.csv` (3.9k rows each, 5 variants) — Calibration comparison

### Scale Sweep & Robustness
- `P1a3/eval/gen_rows.csv` (18.7k rows) — Scale factors 0.6x–1.6x; plot CER vs. scale
- `P1a3/eval/sweep.csv` (1.6k rows) — Summary sweep

### Font Selection
- `font_search/scores.csv` (12 rows) — Rank fonts by CER; bar chart
- `font_search/scores_long.csv` (144 rows) — Per-word per-font; heatmap or faceted bar chart

### Style Evaluation with Confidence Intervals
- `style_eval/m6_ncc.csv` (9.6k rows) — **Best for confidence interval plots:** NCC scores with CI bounds by writer/method

### OCR Inference Speed
- `compare_inference/image2_scores.csv` (297 rows) — Trade-off: inference time vs. CER

---

## Data File Inventory: Columns at a Glance

### Standard Evaluation CSV columns:
- **Identification:** `dataset`, `writer`, `idx`, `text`, `method`, `seed`
- **Metrics:** `cer` (Character Error Rate), `wer` (Word Error Rate), `ncc` (Normalized Cross Correlation), `ci_lower`, `ci_upper` (confidence intervals)
- **Model params:** `scale`, `pen_width`, `pen_angle`, `gain_factor`

### Training CSV columns:
- **Timeseries:** `epoch`, `loss`, `val_loss`, `val_cer`

### Measurement CSV columns:
- **Image stats:** `pixel_mass`, `intensity`, `brightness`, `width`, `height`, `pixel_area`
- **Per-instance:** `writer`, `idx`, `text`

---

## Notes

1. **Per-word image caches:** Excluded by design. Subfolders like `eval_cache/`, `cache/`, per-writer `dp_raw/` digit generation are supporting materials, not summary figures.

2. **Smoke test data:** Included in inventory but labeled as testing/debug runs (smoke_*, _smoke). These use subset sizes (e.g., 20–600 rows) vs. full eval (3.9k–35k rows).

3. **Writer ID encoding:** Writers appear as `writer_NN_M_K` or `bnhtr_NNN` in naming; typically 354 distinct writers in full evaluation sets.

4. **Vocabulary mapping:** `train_vocab_v2.json` (16k words) in eval_v3 is comprehensive; smaller vocabs in P1a/P1a2 (~1k words) reflect subset training.

5. **Page-level JSON files:** Store full layout (line positions, bounding boxes). Each page inference JSON ~0.06–0.3 MB due to detailed coordinate logs.

6. **Confidence interval data:** Only `style_eval/m6_ncc.csv` includes explicit CI bounds; other evaluations compute CI post-hoc from per-word scores.

---

**Total inventory: ~500+ data files, ~200 image files. Estimated printable pages (sheet summaries only, downscaled): 30–50 pages at 2-3 figures per page.**
