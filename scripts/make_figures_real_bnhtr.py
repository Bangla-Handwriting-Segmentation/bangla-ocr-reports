"""Figures for the real-BN-HTRd cross-dataset validation section."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "assets"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "figure.dpi": 160,
    "savefig.dpi": 160,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})

COLOR_BND = "#d62728"
COLOR_SWIN = "#2ca02c"


def savefig(fig, name):
    path = OUT / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


real = json.loads((ROOT / "bench_results" / "swin_vs_bn_drishti_bnhtr_real.json").read_text())
synth = json.loads((ROOT / "bench_results" / "swin_vs_bn_drishti_full.json").read_text())

# ---------------------------------------------------------------------------
# Figure 14: real-BN-HTRd P/R/F1
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for ax, task in zip(axes, ["line", "word"]):
    bnd = real["bn_drishti"][task]["match"]
    swin = real["swin"][task]["match"]
    vals_bnd = [bnd["precision"], bnd["recall"], bnd["f1"]]
    vals_swin = [swin["precision"], swin["recall"], swin["f1"]]
    xlabels = ["P", "R", "F1"]
    xx = np.arange(3)
    w = 0.35
    ax.bar(xx - w / 2, vals_bnd, w, label="BN-DRISHTI (rect.)", color=COLOR_BND)
    ax.bar(xx + w / 2, vals_swin, w, label="Swin (ours, polygon)", color=COLOR_SWIN)
    ax.set_xticks(xx)
    ax.set_xticklabels(xlabels)
    ax.set_title(task.capitalize() + " task")
    ax.set_ylim(0, 1.15)
    for i, v in enumerate(vals_bnd):
        ax.text(i - w / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    for i, v in enumerate(vals_swin):
        ax.text(i + w / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
axes[0].set_ylabel("Score")
axes[0].legend(fontsize=9, loc="lower left")
fig.suptitle("Precision / Recall / F1 on REAL BN-HTRd handwriting (84-page test split)\nsame protocol as Table 3, different (real, not synthetic) dataset")
fig.tight_layout()
savefig(fig, "fig14_real_prf1.png")

# ---------------------------------------------------------------------------
# Figure 15: real-BN-HTRd mAP
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for ax, task in zip(axes, ["line", "word"]):
    bnd = real["bn_drishti"][task]["coco_map"]
    swin = real["swin"][task]["coco_map"]
    vals_bnd = [bnd["map_50"], bnd["map_75"], bnd["map"]]
    vals_swin = [swin["map_50"], swin["map_75"], swin["map"]]
    xlabels = ["mAP@50", "mAP@75", "mAP"]
    xx = np.arange(3)
    w = 0.35
    ax.bar(xx - w / 2, vals_bnd, w, label="BN-DRISHTI (rect.)", color=COLOR_BND)
    ax.bar(xx + w / 2, vals_swin, w, label="Swin (ours, polygon)", color=COLOR_SWIN)
    ax.set_xticks(xx)
    ax.set_xticklabels(xlabels)
    ax.set_title(task.capitalize() + " task")
    ax.set_ylim(0, 1.1)
    for i, v in enumerate(vals_bnd):
        ax.text(i - w / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=8)
    for i, v in enumerate(vals_swin):
        ax.text(i + w / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=8)
axes[0].set_ylabel("mAP")
axes[0].legend(fontsize=9, loc="upper right")
fig.suptitle("Mask-mode COCO-style mAP on REAL BN-HTRd handwriting (84-page test split)")
fig.tight_layout()
savefig(fig, "fig15_real_map.png")

# ---------------------------------------------------------------------------
# Figure 16: the domain-gap reversal -- F1 on synthetic vs. real, per system
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
for ax, task in zip(axes, ["line", "word"]):
    bnd_synth = synth["bn_drishti"][task]["match"]["f1"]
    swin_synth = synth["swin"][task]["match"]["f1"]
    bnd_real = real["bn_drishti"][task]["match"]["f1"]
    swin_real = real["swin"][task]["match"]["f1"]
    xx = np.arange(2)
    w = 0.35
    ax.bar(xx - w / 2, [bnd_synth, bnd_real], w, label="BN-DRISHTI (rect.)", color=COLOR_BND)
    ax.bar(xx + w / 2, [swin_synth, swin_real], w, label="Swin (ours, polygon)", color=COLOR_SWIN)
    ax.set_xticks(xx)
    ax.set_xticklabels(["Synthetic\n(syn-hw, 6,045 pg)", "Real\n(BN-HTRd, 84 pg)"])
    ax.set_title(task.capitalize() + " task — F1")
    ax.set_ylim(0, 1.22)
    for i, v in enumerate([bnd_synth, bnd_real]):
        ax.text(i - w / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    for i, v in enumerate([swin_synth, swin_real]):
        ax.text(i + w / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
axes[0].set_ylabel("F1")
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, 1.14))
fig.suptitle("Line-task ranking reverses between datasets; word task narrows to near-parity\nSwin dominates on synthetic data it was trained on, BN-DRISHTI leads on real handwriting it was trained on", y=1.22)
fig.tight_layout()
savefig(fig, "fig16_domain_gap_reversal.png")

print("done")
