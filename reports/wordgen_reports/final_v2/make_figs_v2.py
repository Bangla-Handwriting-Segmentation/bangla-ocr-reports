#!/usr/bin/env python
"""Build the charts for the three new v2 sections (rescore_v2, S_bestofN, S_ocrloss)
and the EXPERIMENTS.md "Current state" summary bar chart.

Run:  /home/hpc4090/shyan/ocr/venv/bin/python make_figs_v2.py   (after make_figs.py)
CPU only. Reads only source result files from scribe_wordgen/runs and
scribe_wordgen/EXPERIMENTS.md; writes only into ./figs/. No GPU, no training/eval
is launched. S_ocrloss has finished (both S_ocr and S_ft0 ran their full 6,000
steps); its figure plots the full-run, smoothed loss curves for both arms.
"""
import json
import re
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "figs"
OUT.mkdir(exist_ok=True)
ROOT = Path("/home/hpc4090/shyan/ocr")
W = ROOT / "scribe_wordgen" / "runs"
WG = ROOT / "scribe_wordgen"

BN_FONT = ROOT / "scribe_layout_v2" / "bscribe_shim" / "NotoSansBengali.ttf"
fm.fontManager.addfont(str(BN_FONT))

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e4e3df"
C = {
    "real": "#52514e", "B0": "#2a78d6", "Bp": "#eb6834", "Bv2": "#eb6834",
    "P": "#1baf7a", "S": "#4a3aa7", "Sbest": "#8a3aa7", "B3": "#eda100", "A": "#e87ba4",
    "pend": "#b23a2e",
}
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


def bar_ci(ax, labels, vals, lo, hi, colours, fmt="{:.3f}", label_vals=True):
    x = np.arange(len(labels))
    err = np.array([np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)])
    ax.bar(x, vals, color=colours, width=0.62, edgecolor="white", linewidth=1)
    ax.errorbar(x, vals, yerr=err, fmt="none", ecolor=INK, elinewidth=0.9, capsize=2.5)
    ax.set_xticks(x, labels)
    ax.grid(axis="x", visible=False)
    if label_vals:
        for xi, v, h in zip(x, vals, hi):
            ax.text(xi, h, fmt.format(v), ha="center", va="bottom", fontsize=7, color=INK2)


def parse_md_tables(text):
    """Return list of (heading_context, DataFrame) for every markdown table."""
    out, lines, ctx = [], text.splitlines(), []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("#") or (ln.strip() and not ln.startswith("|") and len(ln) < 80):
            ctx.append(ln.strip())
        if ln.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[-| :]+\|$", lines[i + 1].strip()):
            hdr = [h.strip() for h in ln.strip().strip("|").split("|")]
            rows = []
            j = i + 2
            while j < len(lines) and lines[j].startswith("|"):
                rows.append([c.strip() for c in lines[j].strip().strip("|").split("|")])
                j += 1
            out.append((list(ctx[-4:]), pd.DataFrame(rows, columns=hdr)))
            i = j
            continue
        i += 1
    return out


def find_table(tables, *needles):
    for ctx, df in tables:
        s = " ".join(ctx)
        if all(n in s for n in needles):
            return df
    raise KeyError(needles)


def parse_kv_tables(text):
    """S_bestofN/RESULTS.md uses plain 'col | col | col' blocks (no leading '|',
    no '---' separator row), each introduced by the last '#'-heading seen (which
    may be followed by a free-text description line before the table starts).
    Returns a list of (heading, DataFrame)."""
    lines = text.splitlines()
    out, heading, i = [], "", 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("#"):
            heading = ln.strip()
            i += 1
            continue
        if " | " in ln and i + 1 < len(lines) and " | " in lines[i + 1]:
            hdr = [c.strip() for c in ln.split(" | ")]
            rows, j = [], i + 1
            while j < len(lines) and " | " in lines[j]:
                parts = [c.strip() for c in lines[j].split(" | ")]
                if len(parts) != len(hdr):
                    break
                rows.append(parts)
                j += 1
            if rows:
                out.append((heading, pd.DataFrame(rows, columns=hdr)))
            i = j
            continue
        i += 1
    return out


def find_kv_table(tables, *needles):
    for heading, df in tables:
        if all(n in heading for n in needles):
            return df
    raise KeyError(needles)


# =============================================================================
# 1) rescore_v2: smudge-fix (img_to_ink min_span=100) rescoring of every arm
# =============================================================================
rv2_txt = (W / "rescore_v2" / "RESULTS.md").read_text()
rv2_tabs = parse_md_tables(rv2_txt)
rv2_summary = pd.read_csv(W / "rescore_v2" / "results_summary.csv")
verify = pd.read_csv(W / "rescore_v2" / "verify_pixel_agreement.csv")
n_verify, n_exact = len(verify), int((verify["exact"] == True).sum())

loc_tab = find_table(rv2_tabs, "attn_locator CER (read-only reference")
loc_tab = loc_tab.set_index("method")


def ci_from_bracket(s):
    m = re.match(r"([\d.]+)\s*\[([\d.]+),\s*([\d.]+)\]", s.strip())
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


loc_pair_legacy = find_table(rv2_tabs, "legacy vs new (min_span=100), paired per arm")
loc_pair_legacy = loc_pair_legacy[loc_pair_legacy["dataset"] == "all"]
loc_pair_legacy = loc_pair_legacy.set_index("a")

arms_loc = ["B0", "Bp_t5c5", "Bv2_t5c5", "P_t5c5", "S_t5c5"]
new_cer = [ci_from_bracket(loc_tab.loc[a, "cer"]) for a in arms_loc]
# legacy = new - diff(new-legacy); diff table gives new-legacy, so legacy = new - diff
legacy_cer = []
for a in arms_loc:
    if a == "B0":
        legacy_cer.append(new_cer[0])  # B0 unaffected by the smudge fix
        continue
    d = float(loc_pair_legacy.loc[a, "cer_diff"])
    legacy_cer.append((new_cer[arms_loc.index(a)][0] - d, np.nan, np.nan))

fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
ax = axs[0]
x = np.arange(len(arms_loc))
wdt = 0.36
labels = ["B0", "B′ t5c5", "B′v2 t5c5", "P t5c5", "S t5c5"]
colours = [C["B0"], C["Bp"], C["Bv2"], C["P"], C["S"]]
for i, (lab, col) in enumerate(zip(labels, colours)):
    lv, nv = legacy_cer[i][0], new_cer[i]
    ax.bar(x[i] - wdt / 2, lv, width=wdt, color=col, alpha=0.4, edgecolor="white")
    ax.bar(x[i] + wdt / 2, nv[0], width=wdt, color=col, alpha=1.0, edgecolor="white")
    ax.errorbar(x[i] + wdt / 2, nv[0], yerr=[[nv[0] - nv[1]], [nv[2] - nv[0]]], fmt="none",
                ecolor=INK, elinewidth=0.9, capsize=2.5)
real_cer = ci_from_bracket(loc_tab.loc["real", "cer"])[0]
ax.axhline(real_cer, color=C["real"], lw=0.9, ls="--")
ax.text(len(arms_loc) - 0.5, real_cer, "real", color=INK2, fontsize=7, va="bottom", ha="right")
ax.set_xticks(x, labels, rotation=15)
ax.set_ylabel("locator CER (letter-unit)")
ax.legend(handles=[Patch(color=INK2, alpha=0.4, label="legacy ink"), Patch(color=INK2, alpha=1.0, label="min_span=100 (new)")],
          loc="upper left")
ax.set_title("(a) Locator CER, legacy vs smudge-fixed ink", loc="left", fontsize=8.5)

ax = axs[1]
arms_rv = ["B0", "P_t5c5", "S_t5c5"]
sub = rv2_summary[(rv2_summary.ckpt == "R_font2real") & (rv2_summary.subset == "all") & (rv2_summary.method.isin(arms_rv))]
sub = sub.set_index("method")
left_sub = find_table(rv2_tabs, "LEFT-ALIGNED renders", "all words").set_index("method")
x2 = np.arange(len(arms_rv))
for i, a in enumerate(arms_rv):
    v, lo, hi = sub.loc[a, ["cer", "cer_lo", "cer_hi"]]
    la = float(left_sub.loc[a, "cer"])
    col = {"B0": C["B0"], "P_t5c5": C["P"], "S_t5c5": C["S"]}[a]
    ax.bar(x2[i] - wdt / 2, v, width=wdt, color=col, alpha=1.0, edgecolor="white")
    ax.errorbar(x2[i] - wdt / 2, v, yerr=[[v - lo], [hi - v]], fmt="none", ecolor=INK, elinewidth=0.9, capsize=2.5)
    ax.bar(x2[i] + wdt / 2, la, width=wdt, color=col, alpha=0.4, edgecolor="white", hatch="//")
real_v = float(sub.loc["B0", "cer"]) if False else None
real_row = rv2_summary[(rv2_summary.ckpt == "R_font2real") & (rv2_summary.subset == "all") & (rv2_summary.method == "real")]
ax.axhline(float(real_row.cer.iloc[0]), color=C["real"], lw=0.9, ls="--")
ax.text(len(arms_rv) - 0.5, float(real_row.cer.iloc[0]), "real", color=INK2, fontsize=7, va="bottom", ha="right")
ax.set_xticks(x2, ["B0", "P t5c5", "S t5c5"])
ax.set_ylabel("R_eval (R_font2real) CER, all words")
ax.legend(handles=[Patch(color=INK2, alpha=1.0, label="as rendered (centred)"),
                    Patch(color=INK2, alpha=0.4, hatch="//", label="left-aligned")], loc="upper left")
ax.set_title("(b) Centring effect: as-is vs left-aligned", loc="left", fontsize=8.5)
fig.suptitle(f"rescore_v2: {n_exact}/{n_verify} regenerated renders pixel-exact vs the old cache", fontsize=8.5, y=1.02)
fig.tight_layout()
save(fig, "rescore_v2_panels.pdf")

# =============================================================================
# 2) S_bestofN: best-of-N legibility selection
# =============================================================================
bn_txt = (W / "S_bestofN" / "RESULTS.md").read_text()
bn_tabs = parse_kv_tables(bn_txt)
headline = json.loads((W / "S_bestofN" / "headline.json").read_text())

r3 = find_kv_table(bn_tabs, "CER, R_font2real (different checkpoint", ">=3 codepoints").set_index("method")
Ns = [1, 2, 4, 8]
r3_vals = [float(r3.loc[f"bestofN{n}", "CER [95% CI] (n)"].split()[0]) for n in Ns]
r3_ci = [ci_from_bracket(r3.loc[f"bestofN{n}", "CER [95% CI] (n)"]) for n in Ns]

loc_bn = find_kv_table(bn_tabs, "BengaliScribe attn_locator (genuinely independent")
loc_bn = loc_bn[loc_bn["method"].isin(["bestofN1", "bestofN8", "B0"])].set_index("method")

fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8))
ax = axs[0]
ax.plot(Ns, r3_vals, color=C["S"], marker="o", ms=5, lw=1.8)
ax.errorbar(Ns, r3_vals, yerr=[[v - c[1] for v, c in zip(r3_vals, r3_ci)], [c[2] - v for v, c in zip(r3_vals, r3_ci)]],
            fmt="none", ecolor=C["S"], elinewidth=0.9, capsize=2.5)
b0_r3 = float(r3.loc["B0", "CER [95% CI] (n)"].split()[0])
real_r3 = float(rv2_summary[(rv2_summary.ckpt == "R_font2real") & (rv2_summary.subset == ">=3cp") &
                             (rv2_summary.method == "real")]["cer"].iloc[0])
ax.axhline(b0_r3, color=C["B0"], lw=0.9, ls="--")
ax.text(8.1, b0_r3, "B0", color=INK2, fontsize=7, va="center")
ax.axhline(real_r3, color=C["real"], lw=0.9, ls="--")
ax.text(8.1, real_r3, "real", color=INK2, fontsize=7, va="center")
ax.set_xscale("log", base=2)
ax.set_xticks(Ns, [str(n) for n in Ns])
ax.minorticks_off()
ax.set_xlabel("N (seeds sampled, select min CTC-NLL under R_real)")
ax.set_ylabel("R_font2real CER, ≥3 codepoints")
ax.set_title("(a) R_font2real (shares real training\ndata with the selector)", loc="left", fontsize=8)

ax = axs[1]
xs = [1, 8]
yv = [ci_from_bracket(loc_bn.loc[f"bestofN{n}", "CER [95% CI]"]) for n in xs]
ax.bar([0, 1], [v[0] for v in yv], color=[C["S"], C["Sbest"]], width=0.55, edgecolor="white")
ax.errorbar([0, 1], [v[0] for v in yv],
            yerr=[[v[0] - v[1] for v in yv], [v[2] - v[0] for v in yv]], fmt="none", ecolor=INK, elinewidth=0.9, capsize=2.5)
for i, v in enumerate(yv):
    ax.text(i, v[2], f"{v[0]:.3f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
b0_loc = ci_from_bracket(loc_bn.loc["B0", "CER [95% CI]"])[0]
ax.axhline(b0_loc, color=C["B0"], lw=0.9, ls="--")
ax.text(1.4, b0_loc, "B0", color=INK2, fontsize=7, va="bottom")
ax.set_xticks([0, 1], ["bestof1\n(= S t5c5)", "bestof8"])
ax.set_ylabel("attn_locator CER (letter-unit)")
ax.set_title("(b) locator (independent of the selector)", loc="left", fontsize=8)
fig.suptitle("Best-of-N selection for Arm S: headline R_font2real ≥3cp bestof8−bestof1 = "
             f"{headline['paired']['diff_cer']:+.3f} [{headline['paired']['lo']:+.3f},{headline['paired']['hi']:+.3f}], "
             f"{headline['n_writers_better']}/{headline['n_writers_total']} writers better", fontsize=8, y=1.06)
fig.tight_layout()
save(fig, "bestofn_cer_vs_n.pdf")

# =============================================================================
# 3) S_ocrloss: full-run training curves, both arms (finished: 6,000/6,000 steps)
# =============================================================================
m_ocr = pd.read_csv(WG / "runs" / "S_ocrloss" / "S_ocr" / "metrics.csv")
m_ft0 = pd.read_csv(WG / "runs" / "S_ocrloss" / "S_ft0" / "metrics.csv")
WIN = 101
for _m in (m_ocr, m_ft0):
    _m["diff_smooth"] = _m["loss_diff"].rolling(WIN, min_periods=10, center=True).mean()
    _m["ctc_smooth"] = _m["loss_ctc_raw"].rolling(WIN, min_periods=10, center=True).mean()
total_steps = 6000
C["Sft0"] = "#c98a2b"

fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8))
ax = axs[0]
ax.plot(m_ocr["step"], m_ocr["diff_smooth"], color=C["S"], lw=1.3, label="S_ocr ($\\lambda$=0.1)")
ax.plot(m_ft0["step"], m_ft0["diff_smooth"], color=C["Sft0"], lw=1.3, label="S_ft0 (control, $\\lambda$=0)")
ax.set_xlabel("step")
ax.set_ylabel("diffusion loss (101-step rolling mean)")
ax.set_xlim(0, total_steps)
ax.legend(loc="upper right", fontsize=7)
ax.set_title("(a) diffusion loss", loc="left", fontsize=8.5)

ax = axs[1]
ax.plot(m_ocr["step"], m_ocr["ctc_smooth"], color=C["S"], lw=1.3, label="S_ocr")
ax.plot(m_ft0["step"], m_ft0["ctc_smooth"], color=C["Sft0"], lw=1.3, label="S_ft0")
ax.set_xlabel("step")
ax.set_ylabel("aux CTC loss, raw (101-step rolling mean)")
ax.set_xlim(0, total_steps)
ax.legend(loc="upper right", fontsize=7)
ax.set_title("(b) auxiliary OCR-CTC loss ($t$<200)", loc="left", fontsize=8.5)
fig.suptitle("S_ocrloss: full run, both arms, 6,000/6,000 steps each — finished, verdict NOT A WIN",
             fontsize=8.2, y=1.04)
fig.tight_layout()
save(fig, "ocrloss_training_curve.pdf")

# Contact sheet is a static rendered artefact (not derived from a results table);
# copy it as-is into figs/ so the report figure stays reproducible from this script.
shutil.copyfile(W / "S_ocrloss" / "contact_sheet.png", OUT / "ocrloss_contact_sheet.png")
MADE.append("ocrloss_contact_sheet.png (copied)")

# =============================================================================
# 4) Current-state summary bar chart (EXPERIMENTS.md "Current state (2026-10-19)")
# =============================================================================
exp_txt = (WG / "EXPERIMENTS.md").read_text()
sect = exp_txt.split("## Current state")[1].split("\n## Log")[0]
cur_tabs = parse_md_tables("## Current state" + sect)
cur = cur_tabs[0][1]
cur = cur[cur["arm"] != ""].reset_index(drop=True)


def clean_arm(a):
    return a.replace("**", "").strip()


def parse_cer_bracket(s):
    m = re.match(r"([\d.]+)\s*\[([\d.]+),\s*([\d.]+)\]", s.strip())
    return (float(m.group(1)), float(m.group(2)), float(m.group(3))) if m else (float(s.strip()), np.nan, np.nan)


cur["arm_clean"] = cur["arm"].apply(clean_arm)
cur["loc"] = cur["CER (locator)"].apply(parse_cer_bracket)
cur["reval_all"] = cur["R_eval CER all / ≥cp".replace("≥cp", "≥3cp") if False else [c for c in cur.columns if c.startswith("R_eval")][0]].apply(lambda s: float(s.split("/")[0].strip()))
reval_col = [c for c in cur.columns if c.startswith("R_eval")][0]
cur["reval_3"] = cur[reval_col].apply(lambda s: float(s.split("/")[1].strip()))
cur["style"] = cur["style M2 MRR"].apply(lambda s: float(s.replace("**", "")))

order = ["real", "B0", "S_t5c5, best of 8", "S_t5c5", "P_t5c5", "Bv2_t5c5"]
cur = cur.set_index("arm_clean").loc[order].reset_index()
disp = ["real", "B0", "S best-8", "S", "P", "B′v2"]
colz = [C["real"], C["B0"], C["Sbest"], C["S"], C["P"], C["Bv2"]]

fig, axs = plt.subplots(1, 3, figsize=(8.2, 2.9))
ax = axs[0]
v = [r[0] for r in cur["loc"]]
lo = [r[1] for r in cur["loc"]]
hi = [r[2] for r in cur["loc"]]
bar_ci(ax, disp, v, lo, hi, colz, fmt="{:.3f}")
ax.set_ylabel("locator CER (letter-unit)")
ax.set_title("(a) locator CER", loc="left", fontsize=8.5)
ax.tick_params(axis="x", labelrotation=35)
ax.set_xticklabels(ax.get_xticklabels(), ha="right")

ax = axs[1]
x = np.arange(len(disp))
wdt = 0.38
ax.bar(x - wdt / 2, cur["reval_all"], width=wdt, color=colz, alpha=1.0, edgecolor="white", label="all words")
ax.bar(x + wdt / 2, cur["reval_3"], width=wdt, color=colz, alpha=0.45, edgecolor="white", label="≥3 codepoints")
ax.set_xticks(x, disp, rotation=35, ha="right")
ax.set_ylabel("R_eval (R_font2real) CER, codepoint")
ax.legend(loc="upper right", fontsize=7)
ax.set_title("(b) R_eval CER", loc="left", fontsize=8.5)

ax = axs[2]
ax.bar(x, cur["style"], color=colz, width=0.62, edgecolor="white")
for xi, v2 in zip(x, cur["style"]):
    ax.text(xi, v2, f"{v2:.3f}", ha="center", va="bottom", fontsize=7, color=INK2)
ax.set_xticks(x, disp, rotation=35, ha="right")
ax.set_ylabel("style M2 MRR (eval_p1a3 protocol)")
ax.set_title("(c) style MRR", loc="left", fontsize=8.5)
fig.suptitle("Current state, VAL, 20 writers, 781 targets (smudge-fixed ink; min_span=100)", fontsize=8.5, y=1.03)
fig.tight_layout()
save(fig, "current_state_bars.pdf")

print("wrote", len(MADE), "files:")
for n in MADE:
    print("  figs/" + n)
