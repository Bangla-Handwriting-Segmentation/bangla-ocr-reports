"""Test-split word F1 per system, raw vs +x, from unified_cascade/results/all_stats.json."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
h = json.loads((ROOT / "unified_cascade/results/all_stats.json").read_text())["headline"]
rows = [("BN-DRISHTI", "bn_drishti", None),
        ("Naive Swin cascade", "baseline", "baseline+x"),
        ("B1: lines→words", "p1_best", "p1_best+x"),
        ("B2: + graphemes", "g_v3", "g_v3+x")]
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
fig, ax = plt.subplots(figsize=(6.4, 2.7), dpi=200)
bar_h = 0.36
for i, (label, raw, x) in enumerate(rows):
    y = len(rows) - 1 - i
    v = h[raw]["strict@0.5"]["f1"]
    ax.barh(y + bar_h / 2 + 0.01, v, height=bar_h, color=BLUE, edgecolor="#fcfcfb", linewidth=1)
    ax.text(v + 0.008, y + bar_h / 2, f"{v:.3f}", va="center", fontsize=7.5, color=INK)
    if x:
        v2 = h[x]["strict@0.5"]["f1"]
        ax.barh(y - bar_h / 2 - 0.01, v2, height=bar_h, color=ORANGE, edgecolor="#fcfcfb", linewidth=1)
        ax.text(v2 + 0.008, y - bar_h / 2, f"{v2:.3f}", va="center", fontsize=7.5, color=INK)
ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1], fontsize=8, color=INK)
ax.set_xlim(0, 0.9)
ax.set_xlabel("strict word F1 @ IoU 0.5 (195 test pages)", fontsize=8, color=MUTED)
ax.tick_params(axis="x", labelsize=7, colors=MUTED, length=0)
ax.tick_params(axis="y", length=0)
ax.grid(axis="x", color=GRID, linewidth=0.6)
ax.set_axisbelow(True)
for s in ax.spines.values():
    s.set_visible(False)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=BLUE, label="raw boxes"), Patch(color=ORANGE, label="+x (box convention calibrated)")],
          fontsize=7, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
fig.tight_layout()
fig.savefig(Path(__file__).with_name("test_f1.pdf"))
fig.savefig(Path(__file__).with_name("test_f1.png"))
