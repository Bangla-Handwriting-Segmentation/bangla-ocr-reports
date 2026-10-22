"""Figures for the neutral-turf BanglaWriting benchmark (Part IV of the report)."""
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


bw = json.loads((ROOT / "bench_results" / "swin_vs_bn_drishti_banglawriting.json").read_text())
bw_box = json.loads((ROOT / "bench_results" / "swin_vs_bn_drishti_banglawriting_boxmode.json").read_text())
synth = json.loads((ROOT / "bench_results" / "swin_vs_bn_drishti_full.json").read_text())
real = json.loads((ROOT / "bench_results" / "swin_vs_bn_drishti_bnhtr_real.json").read_text())

# ---------------------------------------------------------------------------
# Figure 17: P/R/F1 on BanglaWriting, word task
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.6, 3.8))
bnd = bw["bn_drishti"]["word"]["match"]
swin = bw["swin"]["word"]["match"]
vals_bnd = [bnd["precision"], bnd["recall"], bnd["f1"]]
vals_swin = [swin["precision"], swin["recall"], swin["f1"]]
xlabels = ["Precision", "Recall", "F1"]
xx = np.arange(3)
w = 0.35
ax.bar(xx - w / 2, vals_bnd, w, label="BN-DRISHTI (rect.)", color=COLOR_BND)
ax.bar(xx + w / 2, vals_swin, w, label="Swin (ours, polygon)", color=COLOR_SWIN)
ax.set_xticks(xx)
ax.set_xticklabels(xlabels)
ax.set_ylim(0, 1.15)
for i, v in enumerate(vals_bnd):
    ax.text(i - w / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
for i, v in enumerate(vals_swin):
    ax.text(i + w / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
ax.set_ylabel("Score")
ax.legend(fontsize=9, loc="upper center")
ax.set_title("Word task, BanglaWriting (260 pages, neutral for both systems)")
fig.tight_layout()
savefig(fig, "fig17_banglawriting_prf1.png")

# ---------------------------------------------------------------------------
# Figure 18: box-mode vs mask-mode mAP on BanglaWriting -- the shape-format collapse
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
for ax, mode, data, title in zip(axes, ["box", "mask"], [bw_box, bw], ["Box-mode mAP", "Mask-mode mAP"]):
    bnd = data["bn_drishti"]["word"]["coco_map"]
    swin = data["swin"]["word"]["coco_map"]
    vals_bnd = [bnd["map_50"], bnd["map_75"], bnd["map"]]
    vals_swin = [swin["map_50"], swin["map_75"], swin["map"]]
    xlabels = ["mAP@50", "mAP@75", "mAP"]
    xx = np.arange(3)
    w = 0.35
    ax.bar(xx - w / 2, vals_bnd, w, label="BN-DRISHTI (rect.)", color=COLOR_BND)
    ax.bar(xx + w / 2, vals_swin, w, label="Swin (ours, polygon)", color=COLOR_SWIN)
    ax.set_xticks(xx)
    ax.set_xticklabels(xlabels)
    ax.set_title(title)
    ax.set_ylim(0, 0.62)
    for i, v in enumerate(vals_bnd):
        ax.text(i - w / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    for i, v in enumerate(vals_swin):
        ax.text(i + w / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
axes[0].set_ylabel("mAP")
axes[0].legend(fontsize=9, loc="upper right")
fig.suptitle("On truly foreign-turf data, mask-mode collapses Swin's mAP far harder than box-mode\n"
             "(manually-drawn GT boxes are looser than syn-hw's ink-derived boxes, widening the polygon/rectangle gap)")
fig.tight_layout()
savefig(fig, "fig18_banglawriting_boxvsmask.png")

# ---------------------------------------------------------------------------
# Figure 19: three-dataset F1 comparison -- the actual answer to "who generalizes better"
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4.2))
datasets = ["Synthetic\n(syn-hw)\nSwin's home turf", "Real\n(BN-HTRd)\nBN-DRISHTI's home turf", "Neutral\n(BanglaWriting)\nforeign to both"]
bnd_f1 = [synth["bn_drishti"]["word"]["match"]["f1"], real["bn_drishti"]["word"]["match"]["f1"], bw["bn_drishti"]["word"]["match"]["f1"]]
swin_f1 = [synth["swin"]["word"]["match"]["f1"], real["swin"]["word"]["match"]["f1"], bw["swin"]["word"]["match"]["f1"]]
xx = np.arange(3)
w = 0.35
ax.bar(xx - w / 2, bnd_f1, w, label="BN-DRISHTI (rect.)", color=COLOR_BND)
ax.bar(xx + w / 2, swin_f1, w, label="Swin (ours, polygon)", color=COLOR_SWIN)
ax.set_xticks(xx)
ax.set_xticklabels(datasets, fontsize=9)
ax.set_ylim(0, 1.05)
for i, v in enumerate(bnd_f1):
    ax.text(i - w / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
for i, v in enumerate(swin_f1):
    ax.text(i + w / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
ax.set_ylabel("Word-task F1 (split-merge, IoU$\\geq$0.5)")
ax.legend(fontsize=9, loc="upper right")
ax.set_title("Word-task F1 across all three datasets\nOn neutral ground, Swin's detection F1 is still higher than BN-DRISHTI's")
fig.tight_layout()
savefig(fig, "fig19_three_dataset_f1.png")

print("done")
