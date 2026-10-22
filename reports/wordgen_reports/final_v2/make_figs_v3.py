#!/usr/bin/env python
"""Build the charts for the new synth_pretrain section (report.tex sec:synthpretrain).

Run (CPU only, after make_figs.py and make_figs_v2.py):
    CUDA_VISIBLE_DEVICES="" /home/hpc4090/shyan/ocr/venv/bin/python make_figs_v3.py

Reads only:
  - scribe_wordgen/runs/synth_pretrain/{summary_val,summary_test,paired_val,paired_test}.csv
  - scribe_wordgen/runs/synth_pretrain/word_list.csv
  - scribe_wordgen/runs/synth_pretrain/cache/{F,B0,S1,S8}.npy  (read-only, mmap)
  - scribe_wordgen/runs/synth_pretrain/inference/compare_val_seed11.png (copied as-is)
Writes only into ./figs/. No GPU, no training/eval is launched.
"""
import random
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
OUT = HERE / "figs"
OUT.mkdir(exist_ok=True)
ROOT = Path("/home/hpc4090/shyan/ocr")
WG = ROOT / "scribe_wordgen"
SP = WG / "runs" / "synth_pretrain"

BN_FONT = ROOT / "scribe_layout_v2" / "bscribe_shim" / "NotoSansBengali.ttf"
fm.fontManager.addfont(str(BN_FONT))
FONT_HW = "/home/hpc4090/nuhan/data/hcr_fp/GlyphScribe/bangla_fonts/hw/NotoSansBengali-VariableFont_wdth,wght.ttf"

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e4e3df"
CP = {"F": "#d98c00", "B0": "#2a78d6", "S1": "#9b8fd1", "S8": "#4a3a80", "Rf10": "#52514e"}
plt.rcParams.update({
    "font.family": ["DejaVu Sans", "Noto Sans Bengali"],
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 8, "savefig.bbox": "tight",
    "savefig.dpi": 200, "pdf.fonttype": 42,
})

MADE = []


def save(fig, name):
    p = OUT / name
    fig.savefig(p)
    plt.close(fig)
    MADE.append(name)


# =============================================================================
# 1) synth_pretrain_cer.pdf: (a) VAL CER per arm with 95% CI, (b) key paired diffs
#    on VAL and TEST (forest-style)
# =============================================================================
summary_val = pd.read_csv(SP / "summary_val.csv")
paired_val = pd.read_csv(SP / "paired_val.csv")
paired_test = pd.read_csv(SP / "paired_test.csv")

arms = ["F", "B0", "S1", "S8", "Rf10"]
disp = ["F (fonts)", "B0", "S1", "S8", "Rf10 (ref)"]
sub = summary_val[summary_val.subset == "all"].set_index("arm").loc[arms]

fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0))
ax = axs[0]
x = np.arange(len(arms))
colours = [CP[a] for a in arms]
v, lo, hi = sub["cer"].values, sub["cer_lo"].values, sub["cer_hi"].values
err = np.array([v - lo, hi - v])
ax.bar(x, v, color=colours, width=0.62, edgecolor="white", linewidth=1)
ax.errorbar(x, v, yerr=err, fmt="none", ecolor=INK, elinewidth=0.9, capsize=2.5)
for xi, vv, hh in zip(x, v, hi):
    ax.text(xi, hh, f"{vv:.3f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
ax.set_xticks(x, disp, rotation=15)
ax.set_ylabel("VAL CER (codepoint, all words)")
ax.set_title("(a) VAL CER per pretraining arm", loc="left", fontsize=8.5)

ax = axs[1]
pairs = [("S8", "F"), ("S8", "B0"), ("S8", "S1"), ("B0", "F")]
labels = ["S8−F", "S8−B0", "S8−S1", "B0−F"]
y = np.arange(len(pairs))[::-1]
for i, (a, b) in enumerate(pairs):
    rv = paired_val[(paired_val.a == a) & (paired_val.b == b) & (paired_val.subset == "all")].iloc[0]
    rt = paired_test[(paired_test.a == a) & (paired_test.b == b) & (paired_test.subset == "all")].iloc[0]
    ax.plot([rv["lo"], rv["hi"]], [y[i] + 0.12, y[i] + 0.12], color=INK, lw=1.2)
    ax.plot(rv["diff"], y[i] + 0.12, "o", color=INK, ms=5, label="VAL" if i == 0 else None)
    ax.plot([rt["lo"], rt["hi"]], [y[i] - 0.12, y[i] - 0.12], color=MUTED, lw=1.2)
    ax.plot(rt["diff"], y[i] - 0.12, "^", color=MUTED, ms=5, label="TEST" if i == 0 else None)
ax.axvline(0, color=INK2, lw=0.8, ls="--")
ax.set_yticks(np.arange(len(pairs))[::-1], labels)
ax.set_xlabel("paired CER diff (negative = first arm more legible)")
ax.legend(loc="lower right", fontsize=7.5)
ax.set_title("(b) key paired diffs, VAL vs TEST", loc="left", fontsize=8.5)
fig.suptitle("synth_pretrain: CRNN pretrained on 6,000 matched words, fine-tuned on 10% of TRAIN writers (3 seeds)",
             fontsize=8.2, y=1.04)
fig.tight_layout()
save(fig, "synth_pretrain_cer.pdf")

# =============================================================================
# 2) synth_pretrain_corpus_examples.png: same ~8 random words rendered by
#    F / B0 / S1 / S8, read directly from the fixed (6000,64,256) uint8
#    ink-high caches (invert for display).
# =============================================================================
word_list = pd.read_csv(SP / "word_list.csv")
N_WORDS = len(word_list)
PICK_SEED = 7
N_PICK = 8
rng = random.Random(PICK_SEED)
idxs = sorted(rng.sample(range(N_WORDS), N_PICK))

ARM_COLS = ["F", "B0", "S1", "S8"]
arrs = {a: np.load(SP / "cache" / f"{a}.npy", mmap_mode="r") for a in ARM_COLS}

IMW, IMH = 256, 64
SCALE = 1.4
CW, CH = int(IMW * SCALE), int(IMH * SCALE)
LABEL_W = 230
HEAD = 40
ROW_PAD = 10
RH = CH + ROW_PAD
W = LABEL_W + CW * len(ARM_COLS) + 20
H = HEAD + RH * N_PICK + 10

f_hdr = ImageFont.truetype(FONT_HW, 15, layout_engine=ImageFont.Layout.RAQM)
f_word = ImageFont.truetype(FONT_HW, 20, layout_engine=ImageFont.Layout.RAQM)

sheet = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(sheet)
xs = [10] + [LABEL_W + CW * j for j in range(len(ARM_COLS))]
headers = ["word (VAL pretrain corpus)"] + ARM_COLS
for xx, h in zip(xs, headers):
    d.text((xx, 12), h, fill=(60, 70, 80), font=f_hdr)
d.line([(0, HEAD - 4), (W, HEAD - 4)], fill=(180, 180, 180))

for row, idx in enumerate(idxs):
    y = HEAD + row * RH
    if row % 2:
        d.rectangle([0, y, W, y + RH], fill=(246, 247, 249))
    text = word_list.iloc[idx]["text"]
    writer = word_list.iloc[idx]["writer_key"]
    d.text((10, y + RH // 2 - 14), str(text), fill=(20, 20, 20), font=f_word)
    d.text((10, y + RH // 2 + 10), f"idx {idx} · {writer}", fill=(130, 130, 130),
            font=ImageFont.truetype(FONT_HW, 11, layout_engine=ImageFont.Layout.RAQM))
    for j, a in enumerate(ARM_COLS):
        im_arr = 255 - np.asarray(arrs[a][idx])
        im = Image.fromarray(im_arr).convert("RGB").resize((CW, CH))
        sheet.paste(im, (LABEL_W + CW * j, y + ROW_PAD // 2))

sheet.save(OUT / "synth_pretrain_corpus_examples.png")
MADE.append(f"synth_pretrain_corpus_examples.png (8 random words, seed={PICK_SEED})")

# =============================================================================
# 3) Copy the inference comparison sheet (illustrative, 20 random VAL words)
# =============================================================================
shutil.copyfile(SP / "inference" / "compare_val_seed11.png", OUT / "synth_pretrain_infer_compare.png")
MADE.append("synth_pretrain_infer_compare.png (copied)")

print("wrote", len(MADE), "files:")
for n in MADE:
    print("  figs/" + n)
