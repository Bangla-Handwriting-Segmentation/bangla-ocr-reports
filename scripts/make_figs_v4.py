#!/usr/bin/env python
"""Build the charts for the new mix_pretrain section (report.tex sec:mixpretrain).

Run (CPU only, after make_figs.py / make_figs_v2.py / make_figs_v3.py):
    CUDA_VISIBLE_DEVICES="" /home/hpc4090/shyan/ocr/venv/bin/python make_figs_v4.py

Reads only:
  - scribe_wordgen/runs/mix_pretrain/{summary_val,summary_test,paired_val,paired_test}.csv
  - scribe_wordgen/runs/mix_pretrain/contact_sheet.png                (copied as-is)
  - scribe_wordgen/runs/mix_pretrain/inference/compare_val_seed11.png (copied as-is)
Writes only into ./figs/. No GPU, no training/eval is launched.
"""
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "figs"
OUT.mkdir(exist_ok=True)
ROOT = Path("/home/hpc4090/shyan/ocr")
WG = ROOT / "scribe_wordgen"
MP = WG / "runs" / "mix_pretrain"

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e4e3df"
CP = {"Fs": "#d98c00", "FsS": "#4a3a80", "FsB0": "#2a78d6", "Rf10": "#52514e"}
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
# 1) mix_pretrain_cer.pdf: (a) VAL CER per arm with 95% CI, (b) key paired diffs
#    on VAL and TEST (forest-style)
# =============================================================================
summary_val = pd.read_csv(MP / "summary_val.csv")
paired_val = pd.read_csv(MP / "paired_val.csv")
paired_test = pd.read_csv(MP / "paired_test.csv")

arms = ["Fs", "FsS", "FsB0", "Rf10"]
disp = ["Fs (font stream)", "Fs+S (25%)", "Fs+B0 (25%)", "Rf10 (ref)"]
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
pairs = [("FsS", "Fs"), ("FsS", "FsB0"), ("FsB0", "Fs"), ("FsS", "Rf10")]
labels = ["FsS−Fs", "FsS−FsB0", "FsB0−Fs", "FsS−Rf10"]
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
fig.suptitle("mix_pretrain: CRNN pretrained on the font stream + 25% generated words, fine-tuned on 10% of TRAIN writers (3 seeds)",
             fontsize=8.0, y=1.04)
fig.tight_layout()
save(fig, "mix_pretrain_cer.pdf")

# =============================================================================
# 2) Copy the corpus contact sheet (S1 vs B0, 8 words each) built by chain.sh
#    stage 6 (wordgen.synthgen.contact_sheet), unmodified.
# =============================================================================
shutil.copyfile(MP / "contact_sheet.png", OUT / "mix_pretrain_contact_sheet.png")
MADE.append("mix_pretrain_contact_sheet.png (copied)")

# =============================================================================
# 3) Copy the inference comparison sheet (illustrative, 16 random VAL words)
# =============================================================================
shutil.copyfile(MP / "inference" / "compare_val_seed11.png", OUT / "mix_pretrain_infer_compare.png")
MADE.append("mix_pretrain_infer_compare.png (copied)")

print("wrote", len(MADE), "files:")
for n in MADE:
    print("  figs/" + n)
