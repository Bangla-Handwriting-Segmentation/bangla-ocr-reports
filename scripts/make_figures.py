"""Generate all report figures from bench_results/*.json. Run once, offline, no GPU needed."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BR = ROOT / "bench_results"
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

COLOR_BND = "#d62728"   # red (matches overlay convention)
COLOR_SWIN = "#2ca02c"  # green
COLOR_GT = "#1f77b4"    # blue (unused directly in bars, kept for legend consistency)


def load(name):
    return json.loads((BR / name).read_text())


def savefig(fig, name):
    path = OUT / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------------------------------
# Figure 1: matching-strategy effect on BN-DRISHTI-only P/R/F1 (strict vs split-merge)
# ---------------------------------------------------------------------------
strict_box = load("strict_box.json")
splitmerge_mask = load("splitmerge_mask.json")

tasks = ["Line", "Word"]
strict_f1 = [strict_box["line"]["match"]["f1"], strict_box["word"]["match"]["f1"]]
sm_f1 = [splitmerge_mask["line"]["match"]["f1"], splitmerge_mask["word"]["match"]["f1"]]
strict_p = [strict_box["line"]["match"]["precision"], strict_box["word"]["match"]["precision"]]
sm_p = [splitmerge_mask["line"]["match"]["precision"], splitmerge_mask["word"]["match"]["precision"]]
strict_r = [strict_box["line"]["match"]["recall"], strict_box["word"]["match"]["recall"]]
sm_r = [splitmerge_mask["line"]["match"]["recall"], splitmerge_mask["word"]["match"]["recall"]]

fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
metrics = [("Precision", strict_p, sm_p), ("Recall", strict_r, sm_r), ("F1", strict_f1, sm_f1)]
x = np.arange(len(tasks))
width = 0.32
for ax, (label, strict_vals, sm_vals) in zip(axes, metrics):
    ax.bar(x - width / 2, strict_vals, width, label="Strict 1:1", color="#7f7f7f")
    ax.bar(x + width / 2, sm_vals, width, label="Split-merge", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_title(label)
    ax.set_ylim(0, 1.0)
    for i, v in enumerate(strict_vals):
        ax.text(i - width / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    for i, v in enumerate(sm_vals):
        ax.text(i + width / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
axes[0].set_ylabel("Score")
axes[-1].legend(loc="lower right", fontsize=9)
fig.suptitle("Effect of Wolf & Jolion split-merge matching on BN-DRISHTI scores\n(full 6,045-page syn-hw validation split, IoU$\\geq$0.5)")
fig.tight_layout()
savefig(fig, "fig1_matching_strategy.png")

# ---------------------------------------------------------------------------
# Figure 2: box-mode vs mask-mode mAP sanity check (BN-DRISHTI only, rectangles)
# ---------------------------------------------------------------------------
strict_mask = load("strict_mask.json")
labels = ["Line mAP@50", "Line mAP@75", "Word mAP@50", "Word mAP@75"]
box_vals = [strict_box["line"]["coco_map"]["map_50"], strict_box["line"]["coco_map"]["map_75"],
            strict_box["word"]["coco_map"]["map_50"], strict_box["word"]["coco_map"]["map_75"]]
mask_vals = [strict_mask["line"]["coco_map"]["map_50"], strict_mask["line"]["coco_map"]["map_75"],
             strict_mask["word"]["coco_map"]["map_50"], strict_mask["word"]["coco_map"]["map_75"]]

fig, ax = plt.subplots(figsize=(7, 3.6))
x = np.arange(len(labels))
width = 0.35
ax.bar(x - width / 2, box_vals, width, label="Vector box IoU", color="#9467bd")
ax.bar(x + width / 2, mask_vals, width, label="Rasterized mask IoU (768px canvas)", color="#8c564b")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=10)
ax.set_ylabel("mAP")
ax.set_title("Sanity check: box-mode vs. mask-mode scoring agree\nfor a pure rectangle-vs-rectangle system (BN-DRISHTI)")
ax.legend(fontsize=9)
for i, (b, m) in enumerate(zip(box_vals, mask_vals)):
    ax.text(i - width / 2, b + 0.01, f"{b:.3f}", ha="center", fontsize=8)
    ax.text(i + width / 2, m + 0.01, f"{m:.3f}", ha="center", fontsize=8)
fig.tight_layout()
savefig(fig, "fig2_box_vs_mask_sanity.png")

# ---------------------------------------------------------------------------
# Figure 3 & 4: three-way full-scale (6,045 page) P/R/F1 and mAP comparison
# ---------------------------------------------------------------------------
full = load("swin_vs_bn_drishti_full.json")

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for ax, task in zip(axes, ["line", "word"]):
    bnd = full["bn_drishti"][task]["match"]
    swin = full["swin"][task]["match"]
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
    ax.set_ylim(0, 1.05)
    for i, v in enumerate(vals_bnd):
        ax.text(i - w / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    for i, v in enumerate(vals_swin):
        ax.text(i + w / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
axes[0].set_ylabel("Score")
axes[0].legend(fontsize=9, loc="lower right")
fig.suptitle("Precision / Recall / F1 at IoU$\\geq$0.5, split-merge matching\nfull 6,045-page held-out syn-hw split")
fig.tight_layout()
savefig(fig, "fig3_full_prf1.png")

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for ax, task in zip(axes, ["line", "word"]):
    bnd = full["bn_drishti"][task]["coco_map"]
    swin = full["swin"][task]["coco_map"]
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
    ax.set_ylim(0, 0.95)
    for i, v in enumerate(vals_bnd):
        ax.text(i - w / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=8)
    for i, v in enumerate(vals_swin):
        ax.text(i + w / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=8)
axes[0].set_ylabel("mAP")
axes[0].legend(fontsize=9, loc="upper right")
fig.suptitle("Mask-mode COCO-style mAP, full 6,045-page held-out syn-hw split\n(note the mAP@75 inversion — investigated in Section VI)")
fig.tight_layout()
savefig(fig, "fig4_full_map.png")

# ---------------------------------------------------------------------------
# Figure 5: resolution sweep (mAP@75 investigation) -- real matched pairs
# ---------------------------------------------------------------------------
inv = load("mask_resolution_investigation.json")
rows = inv["real"]["by_resolution"]
xlab = [str(r["max_side"]) for r in rows]
swin_iou = [r["swin_mean_iou"] for r in rows]
bnd_iou = [r["bnd_mean_iou"] for r in rows]

fig, ax = plt.subplots(figsize=(6.5, 3.6))
xx = np.arange(len(xlab))
ax.plot(xx, swin_iou, "o-", color=COLOR_SWIN, label="Swin (ours, polygon)")
ax.plot(xx, bnd_iou, "o-", color=COLOR_BND, label="BN-DRISHTI (rectangle)")
ax.set_xticks(xx)
ax.set_xticklabels(xlab)
ax.set_xlabel("Mask canvas longest side (px)")
ax.set_ylabel("Mean matched-pair IoU")
ax.set_ylim(0.5, 0.85)
ax.set_title("Mean IoU vs. rasterization resolution\n(701 Swin / 547 BN-DRISHTI matched line pairs, 40 pages)")
ax.legend(fontsize=9)
fig.tight_layout()
savefig(fig, "fig5_resolution_sweep.png")

# ---------------------------------------------------------------------------
# Figure 6: box-mode vs mask-mode mAP@75 for Swin vs BN-DRISHTI (150-page corroboration)
# ---------------------------------------------------------------------------
boxmode = load("swin_vs_bn_drishti_boxmode.json")
maskmode150 = load("swin_vs_bn_drishti_res2048.json")  # 150-page, 2048px canvas -- use as mask-mode reference at matched sample size

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for ax, task in zip(axes, ["line", "word"]):
    box_bnd = boxmode["bn_drishti"][task]["coco_map"]["map_75"]
    box_swin = boxmode["swin"][task]["coco_map"]["map_75"]
    mask_bnd = maskmode150["bn_drishti"][task]["coco_map"]["map_75"]
    mask_swin = maskmode150["swin"][task]["coco_map"]["map_75"]
    xx = np.arange(2)
    w = 0.35
    ax.bar(xx - w / 2, [mask_bnd, mask_swin], w, label="Mask-mode (2048px)", color="#17becf")
    ax.bar(xx + w / 2, [box_bnd, box_swin], w, label="Box-mode (vector)", color="#bcbd22")
    ax.set_xticks(xx)
    ax.set_xticklabels(["BN-DRISHTI", "Swin (ours)"])
    ax.set_title(task.capitalize() + " task — mAP@75")
    ax.set_ylim(0, 0.85)
    for i, v in enumerate([mask_bnd, mask_swin]):
        ax.text(i - w / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=8)
    for i, v in enumerate([box_bnd, box_swin]):
        ax.text(i + w / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=8)
axes[0].set_ylabel("mAP@75")
axes[0].legend(fontsize=9, loc="upper left")
fig.suptitle("mAP@75 flips once Swin's polygon is reduced to its own bounding box\n(150-page sample, split-merge matching)")
fig.tight_layout()
savefig(fig, "fig6_boxmode_vs_maskmode_map75.png")

# ---------------------------------------------------------------------------
# Figure 7: polygon area / own-bbox area ratio (single annotated summary bar)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.2, 3.2))
mean_ratio = 0.734
lo, hi = 0.559, 0.964
ax.set_ylim(-0.5, 0.6)
ax.barh([0], [mean_ratio], color=COLOR_SWIN, height=0.5)
ax.barh([0], [1.0], color="none", edgecolor="black", height=0.5)
ax.errorbar([mean_ratio], [0], xerr=[[mean_ratio - lo], [hi - mean_ratio]], fmt="none",
            ecolor="black", capsize=6, linewidth=1.5)
ax.axvline(1.0, color="black", linewidth=0.8, linestyle="--")
ax.set_xlim(0, 1.05)
ax.set_yticks([])
ax.set_xlabel("Polygon area / own bounding-box area")
ax.set_title("Real Swin line predictions: how much of its own bbox\na tight polygon actually covers (n=32; range shown)")
ax.text(mean_ratio, 0.42, f"mean = {mean_ratio:.3f}", ha="center", fontsize=9)
fig.tight_layout()
savefig(fig, "fig7_polygon_bbox_ratio.png")

print("done")
