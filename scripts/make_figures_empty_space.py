"""Build the ink/background-coverage bar chart and the geometric-encapsulation example
figure from bench_results/empty_space_analysis.json + report/assets/examples/examples.npy."""
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


# ---------------------------------------------------------------------------
# Figure 8: ink recall (comparable) vs background/margin coverage (very different)
# ---------------------------------------------------------------------------
data = json.loads((ROOT / "bench_results" / "empty_space_analysis.json").read_text())

fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
for ax, task in zip(axes, ["line", "word"]):
    bnd, swin = data[task]["bnd"], data[task]["swin"]
    labels = ["Ink recall\n(true text captured)", "Background coverage\n(empty margin also claimed)"]
    xx = np.arange(2)
    w = 0.35
    bnd_vals = [bnd["mean_ink_recall"], bnd["mean_background_coverage"]]
    swin_vals = [swin["mean_ink_recall"], swin["mean_background_coverage"]]
    ax.bar(xx - w / 2, bnd_vals, w, label="BN-DRISHTI (rect.)", color=COLOR_BND)
    ax.bar(xx + w / 2, swin_vals, w, label="Swin (ours, polygon)", color=COLOR_SWIN)
    ax.set_xticks(xx)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title(f"{task.capitalize()} task  (n={bnd['n']}/{swin['n']} matched instances)", fontsize=10)
    ax.set_ylim(0, 1.15)
    for i, v in enumerate(bnd_vals):
        ax.text(i - w / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    for i, v in enumerate(swin_vals):
        ax.text(i + w / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
axes[0].set_ylabel("Fraction")
axes[0].legend(fontsize=9, loc="upper center")
fig.suptitle("Both systems recall real ink equally well; only BN-DRISHTI's rectangle\n"
             "also claims most of the GT box's empty margin — the mechanical cause of its mAP@75 edge")
fig.tight_layout()
savefig(fig, "fig8_ink_vs_background_coverage.png")

# ---------------------------------------------------------------------------
# Figure 9: geometric encapsulation examples (raw crop / ink mask / shape overlay)
# ---------------------------------------------------------------------------
examples = np.load(ROOT / "report" / "assets" / "examples" / "examples.npy", allow_pickle=True)
by_tag = {}
for ex in examples:
    by_tag.setdefault(ex["tag"], []).append(ex)

# pick the largest (most legible) example per tag
wanted_tags = ["line_bnd", "line_swin", "word_bnd", "word_swin"]
chosen = {}
for tag in wanted_tags:
    cands = by_tag.get(tag, [])
    if not cands:
        continue
    chosen[tag] = max(cands, key=lambda e: e["gray_crop"].size)

titles = {
    "line_bnd": "Line — BN-DRISHTI rectangle",
    "line_swin": "Line — Swin polygon (ours)",
    "word_bnd": "Word — BN-DRISHTI rectangle",
    "word_swin": "Word — Swin polygon (ours)",
}
out_names = {
    "line_bnd": "fig9a_geo_line_bnd.png",
    "line_swin": "fig9b_geo_line_swin.png",
    "word_bnd": "fig9c_geo_word_bnd.png",
    "word_swin": "fig9d_geo_word_swin.png",
}
col_titles = ["Source crop (GT box)", "Ink vs. background\n(Otsu threshold, independent)", "Prediction overlaid on empty-space map"]

import cv2

for tag in wanted_tags:
    if tag not in chosen:
        continue
    ex = chosen[tag]
    gray = ex["gray_crop"]
    ink = ex["ink_mask"]
    pred_mask = ex["pred_mask"]
    h, w = gray.shape
    aspect = h / w
    fig_w = 10.5
    panel_h = max(1.3, fig_w / 3 * aspect)

    fig, axes = plt.subplots(1, 3, figsize=(fig_w, panel_h + 0.9))

    axes[0].imshow(gray, cmap="gray", vmin=0, vmax=255)

    ink_vis = np.zeros((h, w, 3), dtype=np.uint8)
    ink_vis[..., :] = 255
    ink_vis[ink] = (30, 30, 30)
    ink_vis[~ink] = (255, 235, 205)
    axes[1].imshow(ink_vis)

    overlay = ink_vis.copy()
    covered_bg = (~ink) & (pred_mask.astype(bool))
    overlay[covered_bg] = (255, 140, 0)
    axes[2].imshow(overlay)
    color = COLOR_BND if "bnd" in tag else COLOR_SWIN
    mask_u8 = (pred_mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        pts = cnt.reshape(-1, 2)
        axes[2].plot(np.append(pts[:, 0], pts[0, 0]), np.append(pts[:, 1], pts[0, 1]),
                     color=color, linewidth=2.5)

    for col in range(3):
        axes[col].set_xticks([])
        axes[col].set_yticks([])
        axes[col].set_title(col_titles[col], fontsize=10)
    fig.suptitle(titles[tag], fontsize=12, y=1.02)
    fig.tight_layout()
    savefig(fig, out_names[tag])

print("done")
