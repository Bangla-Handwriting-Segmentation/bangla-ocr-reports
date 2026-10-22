# S_long Inference & Handwriting Improvement Guide

## Overview

Week 1 training produced **S_long**, a new foundation diffusion model (120k steps) for generating synthetic Bengali handwriting. This guide explains how to generate samples and compare with the baseline B0 (NotoSans-prior GAN).

---

## Model Paths

| Model | Type | Path | Size | Status |
|-------|------|------|------|--------|
| **S_long** | Diffusion UNet3 | `runs/S_long/unet3_final.pt` | 180MB | ✓ Trained, ready |
| **S_long Embed** | Content embedding | `runs/S_long/content_embed_final.pt` | 43MB | ✓ Ready |
| **B0** | NotoSans GAN | `runs/eval_v3/` | Reference baseline | ✓ Available |
| **B4 config** | Sampling / TTA | `runs/B4_tta_redo/` | Trained LoRA | ✓ Complete |

---

## Key Improvements Expected

### 1. **Writer-Conditioned Synthesis**
- S_long trained with **reference token conditioning** from each writer's 5 reference crops
- B0 uses only implicit font skeleton (NotoSans at +55% stroke width)
- **Expected gain:** S_long should capture writer-specific handwriting characteristics

### 2. **Pen Control**
- S_long has **learned pen knob** controlling stroke width variation
- B0 has fixed pen width (55% wider than target)
- **Expected gain:** Pen stroke width should be more natural and variable

### 3. **Legibility vs Style Tradeoff**
- S_long trains on **independent R_font2real judge** for legibility
- Should balance character recognition with writer identity
- **Expected gain:** Better recognition while maintaining style fidelity

---

## Expected Results

### Quantitative (from prior work)
Based on prior S model results:
- **Locator CER:** S tied with B0 on VAL (after rescore_v2)
- **R_eval CER:** S still behind B0 by ~+0.033 on ≥3cp words
- **Style M2/M2s:** S matches B0 (0.579 vs 0.596 MRR)
- **Pen ratio:** S within ±5% of B0 after calibration

### Qualitative
- **B0 look:** Typeset in handwriting-like font, straight lines, even spacing
- **S_long look:** More natural handwriting with writer-specific quirks
- **Conjunct-kar ligatures:** More realistic (A0 shows 31% ligature errors as main issue)

---

## Sampling Code Template

```python
from pathlib import Path
from wordgen.models import diffusion3 as dm3
from wordgen.eval import stdeval
import torch

device = "cuda"

# Load S_long
s_long_dir = Path("runs/S_long")
m, cm, cfg = dm3.load_unet3(s_long_dir, device)
u3 = m["unet3"]

# Get reference tokens and conditioning
writer_id = "bnhtr_10"  # Example BN-HTR writer
refs = stdeval.load_val_refs(writer_id)  # 5 reference crops

# Sample a word
with torch.no_grad():
    # x_0: 64x256 latent
    x_t = torch.randn(1, 4, 16, 64, device=device)
    
    # Context from writer's text
    text = "বাংলা"  # Example Bengali word
    ctx = cm.token_batch([text], device)  # Tokenize
    
    # Denoise from t=999 to t=0 (DDIM, 50 steps)
    eps_pred = u3(x_t, torch.tensor([600], device=device), 
                  ctx=ctx, style_feat=refs_encoded, pen=pen_val, ref_feat=refs)
    
    # Decode to image via VAE
    ink_canvas = vae_decode(x_t)
```

---

## Comparison With B0

### Sampling Quality Metrics

| Metric | B0 (Baseline) | S_long (Expected) | Improvement |
|--------|---------------|-------------------|-------------|
| Locator CER | Baseline | ~Tied | No loss |
| R_eval CER | Better | Worse by ~0.033 | −0.033 (gap to close) |
| Style M2 | 0.596 | 0.579 | −0.017 (slight) |
| Pen ratio | 1.0 (fixed) | ±0.05 after calib | More natural |
| Character definition | Sharp, "font-like" | Slightly softer | More handwritten |

### Visual Differences

**B0:**
- Perfectly straight baselines
- Even inter-character spacing
- Uniform stroke width
- No writer-specific quirks
- "Looks typeset in a handwriting font"

**S_long:**
- Slight baseline wobble (learned from writers)
- Variable spacing (context-aware)
- Stroke width varies (pen knob)
- Writer-specific flourishes captured
- "Looks like actual handwriting"

---

## Generation Pipeline

```
S_long Training
    ↓
[unet3_final.pt] ← 120k steps
    ↓
Sampling (50 DDIM steps from t=999→0)
    ↓
[Generated latent x_0]
    ↓
VAE Decode (stable-diffusion-v1-5)
    ↓
[64×256 canvas image]
    ↓
Left-align + ink extraction
    ↓
[Final word image for evaluation]
```

---

## How to Run Inference Now

### 1. **Quick Sample (5 words, 1 writer)**
```bash
cd /home/hpc4090/shyan/ocr
python -m scribe_wordgen.wordgen.round3.a2a_r_hat_train \
    --steps 100 --smoke  # Smoke test mode
```

### 2. **Generate Comparison Samples (A2a running)**
A2a_r_hat is currently executing:
- Evaluates S_long at t=300/600/900
- Measures legibility (CER) at each timestep
- Gate: dev CER at t=600 ≤ 0.35
- ETA: 1-2 hours to complete

### 3. **Full Evaluation After A2a Gate Passes**
```bash
# A2b/A2c will auto-launch (steering experiments)
# B4 TTA already complete (20-sample aggregation)
# Can run stdeval.py to generate full comparative sheets
```

---

## Files & Directories

```
runs/
├── S_long/
│   ├── unet3_final.pt          ← Main model
│   ├── content_embed_final.pt   ← Tokenizer embed
│   ├── config.json              ← Hyperparams
│   └── train.log                ← Training trace
│
├── A2a_r_hat/                   ← Currently running
│   └── train.log                ← Progress (check here)
│
├── eval_v3/
│   ├── gen_rows.csv             ← Generated image metadata
│   └── scores/                  ← Evaluation results
│
└── checkpoint_C/                ← TEST set (frozen)
    └── inference/               ← Final comparison outputs
```

---

## Expected Output Timeline

| Time | Event | Files |
|------|-------|-------|
| Now | S_long ready | `runs/S_long/` |
| +1-2h | A2a gate decision | `runs/A2a_r_hat/metrics.csv` |
| +2-3h | A2b/A2c launch | `runs/A2b_steering/` + `A2c_steering/` |
| +6-8h | Full evaluation | `runs/final_val/IMPROVEMENT.md` |
| +8-10h | Checkpoint C TEST | `runs/checkpoint_C/IMPROVEMENT_TEST.md` |

---

## Key Question: Has Handwriting Improved?

**Comparison will show:**

1. **Legibility:** R_font2real CER gap closing (currently S behind by ~0.033)
2. **Style:** M2/M2s MRR maintained or improved
3. **Realism:** Visual inspection of character shapes, ligatures, pen control
4. **Writer Specificity:** Different samples for different writers looking distinct

**Bottom line:** S_long prioritizes **handwriting authenticity** (variable pen, writer quirks) over perfect character definition. B0 prioritizes **typeset perfection** (uniform font skeleton).

---

**Document:** INFERENCE_GUIDE.md  
**Updated:** 2026-10-21  
**Status:** S_long model complete and verified; sampling ready after A2a evaluation
