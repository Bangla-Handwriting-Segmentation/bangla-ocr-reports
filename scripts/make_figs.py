#!/usr/bin/env python
"""Build every chart and every downscaled qualitative image used by report.tex.

Run:  /home/hpc4090/shyan/ocr/venv/bin/python make_figs.py
Reads only source result files (CSV / JSON / markdown tables / train logs) from
scribe_wordgen/runs, scribe_layout*/runs; writes only into ./figs/.
No numbers are typed in by hand except where a source exists only as a text
report (then the text file is parsed; see parse_* helpers).
Excluded on purpose (still running): runs/rescore_v2, runs/S_bestofN, runs/S_ocrloss.
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
OUT = HERE / "figs"
OUT.mkdir(exist_ok=True)
ROOT = Path("/home/hpc4090/shyan/ocr")
W = ROOT / "scribe_wordgen" / "runs"
L1 = ROOT / "scribe_layout" / "runs"
L2 = ROOT / "scribe_layout_v2" / "runs"
FFT = ROOT / "scribe_layout_fft2d" / "runs"
FOU = ROOT / "scribe_layout_fourier" / "runs"

BN_FONT = ROOT / "scribe_layout_v2" / "bscribe_shim" / "NotoSansBengali.ttf"
fm.fontManager.addfont(str(BN_FONT))
BN = fm.FontProperties(fname=str(BN_FONT))

# ---- style (light, print) -------------------------------------------------
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e4e3df"
C = {  # colour follows the entity everywhere
    "real": "#52514e", "B0": "#2a78d6", "Bp": "#eb6834", "Bv2": "#eb6834",
    "P": "#1baf7a", "S": "#4a3aa7", "B3": "#eda100", "A": "#e87ba4",
    "v1": "#2a78d6", "v2": "#4a3aa7", "manual": "#eb6834", "naive": "#eda100",
    "shuffled": "#8a8985", "font": "#e34948",
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


def jload(p):
    return json.loads(Path(p).read_text())


def arm_colour(m):
    for k in ("real", "B0", "Bv2", "Bp", "B3", "P", "S"):
        if m.startswith(k):
            return C[k]
    if m.startswith("Ap") or m.startswith("A"):
        return C["A"]
    return MUTED


def bar_ci(ax, labels, vals, lo, hi, colours, horizontal=False, fmt="{:.3f}", label_vals=True):
    x = np.arange(len(labels))
    err = np.array([np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)])
    if horizontal:
        ax.barh(x, vals, color=colours, height=0.62, edgecolor="white", linewidth=1)
        ax.errorbar(vals, x, xerr=err, fmt="none", ecolor=INK, elinewidth=0.9, capsize=2.5)
        ax.set_yticks(x, labels)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        if label_vals:
            for xi, v, h in zip(x, vals, hi):
                ax.text(h, xi, "  " + fmt.format(v), va="center", fontsize=7.5, color=INK2)
    else:
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


# =============================================================================
# Word generation, locator CER / style / pen  (P1a3 + eval_v3 metrics.json)
# =============================================================================
p3 = jload(W / "P1a3/eval/metrics.json")
ev3 = jload(W / "eval_v3/metrics.json")

DISPLAY = {"real": "real", "B0": "B0", "B3": "B3", "Ap_g3": "A′ g3", "Bp_t3c3": "B′ t3c3",
           "Bp_t5c5": "B′ t5c5", "Bv2_t5c5": "B′v2 t5c5", "P_t5c5": "P t5c5", "S_t5c5": "S t5c5",
           "P_t5c5_cal": "P cal", "S_t5c5_cal": "S cal", "Bv2_t5c5_cal": "B′v2 cal"}


def cer_of(m):
    src = p3["methods"] if m in p3["methods"] else ev3["methods"]
    return src[m]["cer"]


# --- Fig: locator CER per arm ---
arms = ["real", "B0", "B3", "Ap_g3", "Bp_t3c3", "Bp_t5c5", "Bv2_t5c5", "P_t5c5", "S_t5c5"]
v = [cer_of(m) for m in arms]
fig, ax = plt.subplots(figsize=(6.6, 2.8))
bar_ci(ax, [DISPLAY[m] for m in arms], [a[0] for a in v], [a[1] for a in v], [a[2] for a in v],
       [arm_colour(m) for m in arms])
ax.axhline(v[0][0], color=C["real"], lw=0.8, ls="--")
ax.text(len(arms) - 0.5, v[0][0], "real", color=INK2, fontsize=7, va="bottom", ha="right")
ax.set_ylabel("locator CER (letter-unit, per-word mean)")
ax.set_title("Locator CER per arm, VAL (20 writers, 781 words), 95% writer-bootstrap CI", loc="left")
save(fig, "cer_locator_arms.pdf")

# --- Fig: guidance sweep (B' t/c) ---
sweep = ["Bp_t2c2", "Bp_t3c3", "Bp_t4c4", "Bp_t5c5", "Bp_t6c6"]
g = [2, 3, 4, 5, 6]
cv = [ev3["methods"][m]["cer"] for m in sweep]
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.5))
ax = axs[0]
ax.errorbar(g, [a[0] for a in cv], yerr=[[a[0] - a[1] for a in cv], [a[2] - a[0] for a in cv]],
            color=C["Bp"], marker="o", ms=4, lw=1.6, capsize=2.5, label="B′ (t = c = g)")
for m, lab in (("real", "real"), ("B0", "B0")):
    y = ev3["methods"][m]["cer"][0]
    ax.axhline(y, color=C[m], lw=0.9, ls="--")
    ax.text(6.1, y, lab, color=INK2, fontsize=7, va="center")
ax.set_xlabel("text guidance t = content guidance c")
ax.set_ylabel("locator CER")
ax.set_xticks(g)
ax.set_title("CER vs guidance", loc="left")
ax = axs[1]
pv = [ev3["pen"][m] for m in sweep]
ax.errorbar(g, [p["ratio"] for p in pv], yerr=[[p["ratio"] - p["ci95"][0] for p in pv], [p["ci95"][1] - p["ratio"] for p in pv]],
            color=C["Bp"], marker="o", ms=4, lw=1.6, capsize=2.5)
ax.axhline(1.0, color=C["real"], lw=0.9, ls="--")
ax.text(6.1, 1.0, "real", color=INK2, fontsize=7, va="center")
ax.axhline(ev3["pen"]["B0"]["ratio"], color=C["B0"], lw=0.9, ls="--")
ax.text(6.1, ev3["pen"]["B0"]["ratio"], "B0", color=INK2, fontsize=7, va="center")
ax.set_xticks(g)
ax.set_xlabel("text guidance t = content guidance c")
ax.set_ylabel("pen ratio (gen / own real)")
ax.set_title("Pen ratio vs guidance", loc="left")
fig.tight_layout()
save(fig, "guidance_sweep.pdf")

# --- Fig: style MRR per arm (E_eval M2 and E_shape M2s) ---
st_e = p3["style"]["eval"]["within"]["all"]
st_s = p3["style"]["shape"]["within"]["all"]
sarms = ["real", "B0", "Bp_t5c5", "Bv2_t5c5", "P_t5c5", "S_t5c5"]
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.6), sharey=True)
for ax, st, ttl in ((axs[0], st_e, "M2 (E_eval: shape + pen)"), (axs[1], st_s, "M2s (E_shape: construction only)")):
    vv = [st["rr"][m] for m in sarms]
    bar_ci(ax, [DISPLAY[m] for m in sarms], [a[0] for a in vv], [a[1] for a in vv], [a[2] for a in vv],
           [arm_colour(m) for m in sarms])
    ch = st["chance"]["rr"]
    ax.axhline(ch, color=MUTED, lw=0.9, ls=":")
    ax.text(-0.45, ch + 0.01, f"chance {ch:.2f}", color=INK2, fontsize=7, va="bottom", ha="left")
    ax.set_title(ttl, loc="left")
    ax.tick_params(axis="x", labelrotation=30)
axs[0].set_ylabel("MRR (20 writers, 726 queries)")
axs[0].axhline(st_e["rr"]["real"][0], color=C["real"], lw=0.8, ls="--")
fig.tight_layout()
save(fig, "style_mrr_arms.pdf")

# --- Fig: pen ratio raw and calibrated ---
pen = p3["pen"]
pairs = [("B0", None), ("Bv2_t5c5", "Bv2_t5c5_cal"), ("P_t5c5", "P_t5c5_cal"), ("S_t5c5", "S_t5c5_cal")]
fig, ax = plt.subplots(figsize=(6.0, 2.5))
x = np.arange(len(pairs))
wdt = 0.36
for i, (raw, cal) in enumerate(pairs):
    for j, (m, alpha, lab) in enumerate(((raw, 1.0, "raw"), (cal, 0.45, "calibrated"))):
        if m is None:
            continue
        r, (lo, hi) = pen[m]["ratio"], pen[m]["ci95"]
        xx = x[i] + (j - 0.5) * wdt if cal else x[i]
        ax.bar(xx, r, width=wdt, color=arm_colour(raw), alpha=alpha, edgecolor="white")
        ax.errorbar(xx, r, yerr=[[r - lo], [hi - r]], fmt="none", ecolor=INK, elinewidth=0.9, capsize=2.5)
        ax.text(xx + wdt * 0.5, r, f" {r:.2f}", ha="left", va="center", fontsize=7, color=INK2)
ax.axhline(1.0, color=C["real"], lw=0.9, ls="--")
ax.text(-0.45, 0.985, "real = 1.00", fontsize=7, color=INK2, ha="left", va="top")
ax.set_xticks(x, ["B0", "B′v2 t5c5", "P t5c5", "S t5c5"])
ax.set_ylim(0.8, 1.55)
ax.set_ylabel("pen ratio (sub-px width/core,\n gen ÷ writer's own)")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=INK2, label="raw (solid)"), Patch(color=INK2, alpha=0.45, label="calibrated (light)")],
          loc="upper left", ncol=2)
ax.set_title("Pen ratio per arm, median over 20 VAL writers, 95% CI", loc="left")
ax.grid(axis="x", visible=False)
save(fig, "pen_ratio.pdf")

# --- Fig: pen knob sweep ---
sw = pd.read_csv(W / "P1a3/eval/sweep.csv")
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.5), sharey=True)
for ax, arm in zip(axs, ("P", "S")):
    d = sw[sw.arm == arm]
    med = d.groupby(["wkey", "mult"]).sub_core.median().unstack()
    for _, row in med.iterrows():
        ax.plot(row.index, row.values, color=C[arm], alpha=0.18, lw=0.8)
    gm = p3["knob_sweep"][arm]["gen_median_by_mult"]
    mm = sorted(float(k) for k in gm)
    ax.plot(mm, [gm[str(k) if str(k) in gm else repr(k)] for k in mm], color=C[arm], marker="o", ms=4, lw=2)
    ks = p3["knob_sweep"][arm]
    ax.set_xscale("log")
    ax.set_xticks(mm, [f"{k:g}" for k in mm])
    ax.minorticks_off()
    ax.set_xlabel("knob multiplier (× refs' measured pen)")
    ax.set_title(f"{arm}: slope {ks['slope_mean']:.2f} [{ks['slope_ci95'][0]:.2f}, {ks['slope_ci95'][1]:.2f}]", loc="left")
axs[0].set_ylabel("generated sub-px width / core")
fig.tight_layout()
save(fig, "pen_knob_sweep.pdf")

# --- Fig: forest plots of paired diffs (S-B0, S-P, P-B0) ---
rv_txt = (W / "R_eval/RESULTS.md").read_text()
rv_tabs = parse_md_tables(rv_txt)


def reval_pairs(which):
    # 'which' = 'all' or '3'; scored with R_eval (= R_font2real)
    sect = rv_txt.split("### scored with R_eval")[1]
    tabs = parse_md_tables(sect)
    key = "Paired diffs (all words)" if which == "all" else "Paired diffs (>=3 codepoints)"
    df = find_table(tabs, key)
    return {f"{r['a']}-{r['b']}": (float(r["diff_cer"]), float(r["diff_lo"]), float(r["diff_hi"])) for _, r in df.iterrows()}


ra, r3 = reval_pairs("all"), reval_pairs("3")
comps = [("S_t5c5", "B0", "S − B0"), ("S_t5c5", "P_t5c5", "S − P"), ("P_t5c5", "B0", "P − B0")]
panels = []
cp = p3["cer_paired"]
panels.append(("Locator CER", [(cp[f"{a}-{b}"]["diff"], *cp[f"{a}-{b}"]["ci95"]) if f"{a}-{b}" in cp else
                               tuple(-np.array([cp[f"{b}-{a}"]["diff"], cp[f"{b}-{a}"]["ci95"][1], cp[f"{b}-{a}"]["ci95"][0]]))
                               for a, b, _ in comps], "lower = first arm more legible"))
panels.append(("R_eval CER, all words", [ra[f"{a}-{b}"] for a, b, _ in comps], "codepoint, corpus"))
panels.append(("R_eval CER, ≥3 codepoints", [r3[f"{a}-{b}"] for a, b, _ in comps], "codepoint, corpus"))
dE = st_e["diff"]["rr"]
dS = st_s["diff"]["rr"]


def sdiff(d, a, b):
    k = f"{a}-{b}"
    if k in d:
        return tuple(d[k][:3])
    x = d[f"{b}-{a}"]
    return (-x[0], -x[2], -x[1])


panels.append(("Style M2 MRR (E_eval)", [sdiff(dE, a, b) for a, b, _ in comps], "higher = first arm more writer-like"))
panels.append(("Style M2s MRR (E_shape)", [sdiff(dS, a, b) for a, b, _ in comps], "higher = first arm more writer-like"))
fig, axs = plt.subplots(2, 3, figsize=(7.0, 3.9))
axs = axs.ravel()
axs[5].axis("off")
for ax, (ttl, vals, note) in zip(axs[:5], panels):
    for i, (d, lo, hi) in enumerate(vals):
        col = C["S"] if comps[i][2].startswith("S") else C["P"]
        ax.plot([lo, hi], [i, i], color=col, lw=2)
        ax.plot(d, i, "o", color=col, ms=5, mec="white", mew=1)
        ax.text(d, i - 0.25, f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]", ha="center", fontsize=6.3, color=INK2)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_title(ttl, loc="left", fontsize=8)
    ax.set_xlabel(note, fontsize=6.8)
    ax.tick_params(labelsize=7)
    ax.grid(axis="y", visible=False)
    ax.set_yticks(range(3), [c[2] for c in comps])
    ax.set_ylim(2.5, -0.7)
    lo_ = min(v[1] for v in vals); hi_ = max(v[2] for v in vals); pad = 0.25 * (hi_ - lo_)
    ax.set_xlim(min(lo_, 0) - pad, max(hi_, 0) + pad)
fig.tight_layout(h_pad=1.2)
save(fig, "forest_paired.pdf")

# =============================================================================
# R_eval: VAL CER per model x dataset, font-gain forest, OOV split, rescoring
# =============================================================================
models = ["R_font0", "R_real", "R_font2real", "R_mix", "R_real_10", "R_font2real_10"]
mlab = {"R_font0": "fonts only", "R_real": "real only", "R_font2real": "fonts→real", "R_mix": "mix (25% synth)",
        "R_real_10": "real, 10% writers", "R_font2real_10": "fonts→real, 10% writers"}


def model_table(m, test=False):
    if test:
        sect = rv_txt.split(f"### {m} (TEST)")[1]
    else:
        sect = rv_txt.split(f"### {m}\n")[1]
    df = parse_md_tables(sect)[0][1]
    for c in ("n", "cer", "cer_lo", "cer_hi", "wer"):
        df[c] = df[c].astype(float)
    return df


fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.7), gridspec_kw={"width_ratios": [1, 1.2]})
dsets = [("bnhtr", "BN-HTR (3,253 words)", C["B0"]), ("bw", "BanglaWriting (1,046)", C["Bp"]), ("overall", "both (4,299)", C["real"])]
for ax, ms_ in ((axs[0], ["R_real", "R_font2real", "R_mix", "R_real_10", "R_font2real_10"]), (axs[1], None)):
    pass
ms_ = ["R_real", "R_font2real", "R_mix", "R_real_10", "R_font2real_10", "R_font0"]
for k, ax in enumerate(axs):
    sub = ms_[:5] if k == 0 else ms_
    x = np.arange(len(sub))
    for j, (ds, lab, col) in enumerate(dsets):
        vals = []
        for m in sub:
            df = model_table(m)
            r = df[(df.dataset == ds) & (df.subset == "all")].iloc[0]
            vals.append((r.cer, r.cer_lo, r.cer_hi))
        xx = x + (j - 1) * 0.26
        ax.bar(xx, [a[0] for a in vals], width=0.26, color=col, edgecolor="white", label=lab)
        ax.errorbar(xx, [a[0] for a in vals], yerr=[[a[0] - a[1] for a in vals], [a[2] - a[0] for a in vals]],
                    fmt="none", ecolor=INK, elinewidth=0.7, capsize=1.5)
    ax.set_xticks(x, [mlab[m] for m in sub], rotation=28, ha="right")
    ax.grid(axis="x", visible=False)
axs[0].set_title("Real-data models (zoom)", loc="left")
axs[1].set_title("All six, incl. fonts-only", loc="left")
axs[0].set_ylabel("R CER (codepoint, corpus) on VAL")
axs[1].legend(loc="upper left")
fig.tight_layout()
save(fig, "reval_models_val.pdf")

# font-gain forest
gdf = find_table(rv_tabs, "Paired font-pretraining gain")
fig, ax = plt.subplots(figsize=(5.6, 2.3))
rows = []
for _, r in gdf.iterrows():
    rows.append((("10% writers" if r["a"].endswith("_10") else "all writers") + ", " +
                 {"bnhtr": "BN-HTR", "bw": "BanglaWriting", "overall": "both"}[r["dataset"]],
                 float(r["diff"]), float(r["lo"]), float(r["hi"]), r["a"].endswith("_10")))
for i, (lab, d, lo, hi, ten) in enumerate(rows):
    col = C["Bp"] if ten else C["B0"]
    ax.plot([lo, hi], [i, i], color=col, lw=2)
    ax.plot(d, i, "o", color=col, ms=5, mec="white", mew=1)
    ax.text(hi + 0.002, i, f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]", va="center", fontsize=7, color=INK2)
ax.axvline(0, color=INK, lw=0.8)
ax.set_yticks(range(len(rows)), [r[0] for r in rows])
ax.set_ylim(len(rows) - 0.4, -0.6)
ax.set_xlim(-0.12, 0.06)
ax.set_xlabel("paired CER difference, fonts→real − real-only (negative = font pretraining helps)")
ax.grid(axis="y", visible=False)
save(fig, "reval_font_gain_forest.pdf")

# OOV split (VAL and TEST), R_real vs R_font2real, overall
subsets = [("iv", "in-vocab"), ("oov_in_synth", "OOV, in synth vocab"), ("oov_not_synth", "OOV, not in synth")]
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.5), sharey=True)
for ax, test in zip(axs, (False, True)):
    x = np.arange(len(subsets))
    for j, (m, col) in enumerate((("R_real", C["real"]), ("R_font2real", C["S"]))):
        df = model_table(m, test)
        vals = [df[(df.dataset == "overall") & (df.subset == s)].iloc[0] for s, _ in subsets]
        xx = x + (j - 0.5) * 0.36
        ax.bar(xx, [r.cer for r in vals], width=0.36, color=col, edgecolor="white", label=mlab[m])
        ax.errorbar(xx, [r.cer for r in vals], yerr=[[r.cer - r.cer_lo for r in vals], [r.cer_hi - r.cer for r in vals]],
                    fmt="none", ecolor=INK, elinewidth=0.7, capsize=2)
        for xi, r in zip(xx, vals):
            ax.text(xi, r.cer_hi + 0.002, f"n={int(r.n)}" if j == 0 else "", ha="center", fontsize=6, color=MUTED)
    ax.set_xticks(x, [s[1] for s in subsets], rotation=15)
    ax.set_title("VAL" if not test else "TEST (scored once)", loc="left")
    ax.grid(axis="x", visible=False)
axs[0].set_ylabel("R CER (codepoint, corpus)")
axs[0].legend(loc="upper left")
fig.tight_layout()
save(fig, "reval_oov.pdf")

# rescoring of generator arms with R_eval (all and >=3cp)
sect = rv_txt.split("### scored with R_eval")[1]
tabs = parse_md_tables(sect)
t_all = find_table(tabs, "All words")
t_3 = find_table(tabs, "Words with >=3 codepoints")
order = ["real", "B0", "Bp_t5c5", "P_t5c5", "S_t5c5"]
fig, ax = plt.subplots(figsize=(6.0, 2.6))
x = np.arange(len(order))
for j, (df, lab, alpha) in enumerate(((t_all, "all words", 1.0), (t_3, "≥3 codepoints", 0.45))):
    df = df.set_index("method")
    vals = [(float(df.loc[m, "cer"]), float(df.loc[m, "cer_lo"]), float(df.loc[m, "cer_hi"])) for m in order]
    xx = x + (j - 0.5) * 0.36
    ax.bar(xx, [a[0] for a in vals], width=0.36, color=[arm_colour(m) for m in order], alpha=alpha, edgecolor="white")
    ax.errorbar(xx, [a[0] for a in vals], yerr=[[a[0] - a[1] for a in vals], [a[2] - a[0] for a in vals]],
                fmt="none", ecolor=INK, elinewidth=0.8, capsize=2)
    for xi, a in zip(xx, vals):
        ax.text(xi, a[2] + 0.002, f"{a[0]:.3f}", ha="center", fontsize=6.5, color=INK2)
ax.set_xticks(x, ["real", "B0", "B′ t5c5", "P t5c5", "S t5c5"])
ax.legend(handles=[Patch(color=INK2, label="all words (solid)"), Patch(color=INK2, alpha=0.45, label="≥3 codepoints (light)")], loc="upper left")
ax.set_ylabel("R_eval CER (codepoint, corpus)")
ax.set_title("Generator arms rescored by R_eval (R_font2real), VAL, 95% writer-bootstrap CI", loc="left")
ax.grid(axis="x", visible=False)
save(fig, "reval_arms.pdf")

# =============================================================================
# Learning curves
# =============================================================================


def roll(df, col="loss", w=500):
    return df["step"], df[col].rolling(w, min_periods=max(1, w // 5)).mean()


fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.5))
ax = axs[0]
for run, lab, col in ((W / "P1a/A", "P1a A (40/writer cap)", C["A"]), (W / "P1a/B", "P1a B (+painted map)", C["B3"]),
                      (W / "P1a2/A", "P1a′ A′ (text only)", C["real"]), (W / "P1a2/B", "P1a′ B′ (+grapheme tokens)", C["Bp"])):
    d = pd.read_csv(run / "metrics.csv")
    s, l = roll(d, w=1000)
    ax.plot(s, l, color=col, lw=1.4, label=lab)
ax.set_ylim(0.055, 0.09)
ax.set_xlabel("training step (batch 32)")
ax.set_ylabel("ε-prediction MSE (1k-step mean)")
ax.set_title("P1a and P1a′ training loss", loc="left")
ax.legend(fontsize=6.5, loc="upper right")
ax = axs[1]
for run, lab, col, w in ((W / "P1a3/P", "P (16k steps)", C["P"], 500), (W / "P1a3/S", "S (24k steps, final)", C["S"], 500),
                         (W / "P1a3/S_diverged_v1", "S, diverged run 1", C["font"], 50), (W / "P1a3/S_diverged_v2", "S, diverged run 2", C["B3"], 50)):
    d = pd.read_csv(run / "metrics.csv")
    s, l = roll(d, w=w)
    ax.plot(s, l, color=col, lw=1.4, label=lab)
ax.set_yscale("log")
ax.set_xlabel("training step (batch 32), warm start from B′")
ax.set_ylabel("loss (rolling mean, log scale)")
ax.set_title("P and S fine-tunes; two diverged S attempts", loc="left")
ax.legend(fontsize=6.8)
fig.tight_layout()
save(fig, "curves_diffusion.pdf")

# R_eval learning curves (VAL sample n=1024 CER logged every 1000 steps)
fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.5))
rc = {"R_font0": C["font"], "R_real": C["real"], "R_font2real": C["S"], "R_mix": C["P"], "R_real_10": C["B3"], "R_font2real_10": C["Bp"]}
for m in models:
    txt = (W / f"R_eval/{m}/train.log").read_text()
    ev = [(int(a), float(b)) for a, b in re.findall(r"\[eval@(\d+)\] VAL\(sample n=\d+\) CER=([\d.]+)", txt)]
    tr = [(int(a), float(b)) for a, b in re.findall(r"step (\d+)/\d+ loss=([\d.]+)", txt)]
    e = np.array(ev)
    axs[1].plot(e[:, 0], e[:, 1], color=rc[m], marker="o", ms=2.5, lw=1.3, label=mlab[m])
    t = pd.DataFrame(tr, columns=["step", "loss"])
    axs[0].plot(t.step, t.loss.rolling(10, min_periods=1).mean(), color=rc[m], lw=1.2, label=mlab[m])
axs[0].set_yscale("log")
axs[0].set_xlabel("step (batch 128)")
axs[0].set_ylabel("CTC loss (10-log-point mean)")
axs[0].set_title("R training loss", loc="left")
axs[1].set_yscale("log")
axs[1].set_xlabel("step")
axs[1].set_ylabel("CER on 1,024 VAL words (log)")
axs[1].set_title("R in-training VAL CER (fine-tunes restart at 0)", loc="left")
axs[1].legend(fontsize=6.5, ncol=1)
fig.tight_layout()
save(fig, "curves_reval.pdf")

# =============================================================================
# Fonts: GlyphScribe + Google chamfer ranking
# =============================================================================
fs = pd.read_csv(W / "font_search/glyphscribe/scores.csv").sort_values("chamfer_mean").reset_index(drop=True)
fs = fs[fs.font != "NotoSansBengali-VariableFont_wdth,wght"]  # duplicate of the baseline file
fig, axs = plt.subplots(1, 2, figsize=(7.0, 6.2), sharey=True, gridspec_kw={"width_ratios": [1.3, 1]})
lab = [re.sub(r"-VariableFont.*|,wght", "", f) for f in fs.font]
cols = [C["B0"] if "baseline" in c else (C["font"] if c == "printed" else C["S"]) for c in fs.category]
bar_ci(axs[0], lab, fs.chamfer_mean, fs.chamfer_lo95, fs.chamfer_hi95, cols, horizontal=True, label_vals=False)
axs[0].set_xlabel("skeleton chamfer to image_2 (12 words; lower = closer shape)")
axs[0].set_title("Shape", loc="left")
for t in axs[0].get_yticklabels():
    if re.search("[\u0980-\u09FF]", t.get_text()):
        t.set_fontproperties(BN)
    t.set_fontsize(6.8)
axs[1].barh(np.arange(len(fs)), fs.width_over_core_mean, color=cols, height=0.62, edgecolor="white")
axs[1].axvline(fs.width_over_core_real_mean.iloc[0], color=C["real"], ls="--", lw=1)
axs[1].text(fs.width_over_core_real_mean.iloc[0], -1.2, " writer 0.075", fontsize=7, color=INK2)
axs[1].set_xlabel("stroke width / core", labelpad=8)
axs[1].set_title("Pen", loc="left")
axs[1].grid(axis="y", visible=False)
fig.tight_layout()
save(fig, "fonts_glyphscribe.pdf")

# =============================================================================
# Layout: W1/KS bars (plan level, v2 harness) and C2ST per stage
# =============================================================================
lm = jload(L2 / "eval/metrics.json")["methods"]
lmeth = [("naive", "naive", C["naive"]), ("manual", "manual (BengaliScribe)", C["manual"]),
         ("shuffled_real", "shuffled real", C["shuffled"]), ("learned_full", "v1 (learned_full)", C["v1"]),
         ("v2_history_unfit", "v2_history_unfit", C["v2"])]
qs = ["gap", "slope", "resid_scale", "drift", "pitch_next", "x_start"]
fig, axs = plt.subplots(2, 6, figsize=(7.2, 3.6))
for k, q in enumerate(qs):
    for r, stat in enumerate(("w1", "ks")):
        ax = axs[r, k]
        vals = [lm[m]["marginals"][q][stat]["mean"] for m, _, _ in lmeth]
        sds = [lm[m]["marginals"][q][stat]["std"] for m, _, _ in lmeth]
        ax.bar(range(len(lmeth)), vals, yerr=sds, color=[c for _, _, c in lmeth], edgecolor="white",
               error_kw=dict(elinewidth=0.7, capsize=1.5, ecolor=INK))
        ax.set_xticks([])
        ax.tick_params(labelsize=6.5)
        ax.grid(axis="x", visible=False)
        if r == 0:
            ax.set_title(q, fontsize=8)
        if k == 0:
            ax.set_ylabel("W1 (U)" if stat == "w1" else "KS statistic")
fig.legend(handles=[Patch(color=c, label=l) for _, l, c in lmeth], loc="lower center", ncol=5, fontsize=7,
           bbox_to_anchor=(0.5, -0.04))
fig.suptitle("Plan-level marginals vs real TEST (lower = closer), mean ± sd over 5 seeds", x=0.01, ha="left", fontsize=9)
fig.tight_layout(rect=(0, 0.04, 1, 0.97))
save(fig, "layout_w1_ks.pdf")


def parse_5seed(p):
    out = {}
    for ln in Path(p).read_text().splitlines():
        m = re.match(r"(.+?)\s+line C2ST ([\d.]+)±([\d.]+)\s+page C2ST ([\d.]+)±([\d.]+)\s+FFT C2ST ([\d.]+)±([\d.]+)", ln)
        if m:
            out[m.group(1).strip()] = [float(m.group(i)) for i in range(2, 8)]
    return out


T = parse_5seed(L2 / "compare_unfitw/compare_5seed.txt")
T.update(parse_5seed(L2 / "compare_unfitw/compare_5seed_dc.txt"))
T.update(parse_5seed(L2 / "compare_unfitw/compare_5seed_wh_test.txt"))
V = parse_5seed(L2 / "compare_unfitw/compare_5seed_val.txt")
V.update(parse_5seed(L2 / "compare_unfitw/compare_5seed_wh_val.txt"))


def pick(d, prefix):
    for k in d:
        if k.startswith(prefix):
            return d[k]
    raise KeyError(prefix)


stages = [("v1", "v1"), ("v2_unfit ", "unfit"), ("v2_unfitw ", "unfitw"),
          ("v2_unfitw_dc", "unfitw_dc*"), ("v2_unfitwh_dc", "unfitwh_dc"),
          ("shuffled_real", "shuffled")]
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.9), sharey=False)
cols6 = [C["v1"], "#9085e9", "#9085e9", C["v2"], "#9085e9", C["shuffled"]]
for ax, idx, ttl in ((axs[0], 0, "line C2ST AUC"), (axs[1], 2, "page C2ST AUC")):
    for j, (src, nm, a) in enumerate(((T, "TEST", 1.0), (V, "VAL", 0.45))):
        vals, sds, xs = [], [], []
        for i, (pre, _) in enumerate(stages):
            try:
                r = pick(src, pre.strip() if pre.strip() != "v2_unfitw" else "v2_unfitw (")
            except KeyError:
                continue
            vals.append(r[idx]); sds.append(r[idx + 1]); xs.append(i + (j - 0.5) * 0.38)
        ax.bar(xs, vals, width=0.38, yerr=sds, color=[cols6[int(round(x_ + 0.2))] for x_ in xs], alpha=a,
               edgecolor="white", error_kw=dict(elinewidth=0.7, capsize=1.5, ecolor=INK))
    ax.axhline(0.5, color=INK, lw=0.8)
    if idx == 0:
        ax.axhline(0.496, color=MUTED, lw=0.9, ls="--")
        ax.text(-0.5, 0.494, "within-page null 0.496", fontsize=6, color=INK2, va="top", ha="left")
    ax.set_ylim(0.45, 0.70)
    ax.set_xticks(range(len(stages)), [s[1] for s in stages], fontsize=6.5, rotation=40, ha="right")
    ax.set_title(ttl + " (plan level)", loc="left", fontsize=8.5)
    ax.grid(axis="x", visible=False)
axs[0].set_ylabel("AUC (0.5 = indistinguishable)")
axs[0].legend(handles=[Patch(color=INK2, label="TEST (solid)"), Patch(color=INK2, alpha=0.45, label="VAL (light)")], fontsize=6.5, loc="upper left")
# e2e balanced only
e2e = [("e2e_pagereset", "pagereset\n(unfit)"), ("e2e_unfitw", "unfitw"), ("e2e_unfitw_dc", "unfitw_dc"), ("e2e_final", "final\n(+margin)")]
ax = axs[2]
for j, (meth, col) in enumerate((("manual", C["manual"]), ("v2", C["v2"]), ("v2_ctx", "#9085e9"))):
    vals = []
    for run, _ in e2e:
        txt = (L2 / run / "results.md").read_text()
        tab = parse_md_tables(txt)[0][1].set_index("method")
        vals.append(float(tab.loc[meth, "c2st_line"]))
    ax.plot(range(len(e2e)), vals, marker="o", ms=4, color=col, lw=1.5, label=meth)
ax.axhline(0.534, color=MUTED, ls="--", lw=0.9)
ax.text(3.1, 0.534, "real-vs-real\nfloor 0.534", fontsize=6, color=INK2, va="bottom", ha="right")
ax.set_xticks(range(len(e2e)), [e[1] for e in e2e], fontsize=6)
ax.set_ylim(0.5, 0.95)
ax.set_title("end-to-end line C2ST (balanced)", loc="left", fontsize=8.5)
ax.legend(fontsize=6.5, loc="center right")
fig.tight_layout()
save(fig, "layout_c2st.pdf")

# e2e W1 bars for final
fin = parse_md_tables((L2 / "e2e_final/results.md").read_text())[0][1].set_index("method")
qs2 = ["gap_w1", "slope_w1", "drift_w1", "pitch_next_w1", "x_start_w1"]
fig, axs = plt.subplots(1, 5, figsize=(7.0, 1.9))
for ax, q in zip(axs, qs2):
    ms3 = ["manual", "v2", "v2_ctx"]
    vals = [float(fin.loc[m, q]) for m in ms3]
    ax.bar(range(3), vals, color=[C["manual"], C["v2"], "#9085e9"], edgecolor="white")
    for i, v_ in enumerate(vals):
        ax.text(i, v_, f"{v_:.3f}", ha="center", va="bottom", fontsize=6.3, color=INK2)
    ax.set_xticks(range(3), ms3, fontsize=6.5)
    ax.set_title(q.replace("_w1", "") + " W1 (U)", fontsize=8)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="x", visible=False)
fig.tight_layout()
save(fig, "layout_e2e_final_w1.pdf")

# =============================================================================
# Spectral side-studies (corrected readings)
# =============================================================================
fair = jload(FOU / "fair_dense_psd.json")
fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.5))
ax = axs[0]
f = np.array(fair["freqs"])
ax.plot(f, fair["real"]["avg"], color=C["real"], lw=2, label=f"real (n={fair['real']['n']})")
for m, col, lab in (("manual", C["manual"], "manual"), ("v1", C["v1"], "v1"), ("fourier", C["S"], "Fourier"),
                    ("fourier_continuous_unrealisable", C["P"], "Fourier, continuous (not realisable)")):
    ax.plot(f, np.array(fair["real"]["avg"]) + np.array(fair[m]["per_band_diff"]), color=col, lw=1.2, label=lab)
ax.axvline(1 / fair["min_wave_U"], color=MUTED, ls=":", lw=0.9)
ax.set_xscale("log")
ax.set_xlabel("frequency (cycles / U)")
ax.set_ylabel("mean log PSD (Lomb–Scargle units)")
ax.set_title("Dense headline traces, fair comparison", loc="left")
ax.legend(fontsize=6.3)
ax = axs[1]
ms4 = [("manual", "manual"), ("v1", "v1"), ("fourier", "Fourier"), ("fourier_continuous_unrealisable", "Fourier cont.\n(unrealisable)")]
vals = [fair[m]["lsd_layout_band"] for m, _ in ms4]
ax.bar(range(4), vals, color=[C["manual"], C["v1"], C["S"], C["P"]], edgecolor="white")
for i, v_ in enumerate(vals):
    ax.text(i, v_, f"{v_:.3f}", ha="center", va="bottom", fontsize=7, color=INK2)
ax.set_xticks(range(4), [m[1] for m in ms4], fontsize=7)
ax.set_ylabel("log-spectral distance to real\n(wavelength ≥ 4 U)")
ax.set_title("Distance in the layout band", loc="left")
ax.grid(axis="x", visible=False)
fig.tight_layout()
save(fig, "spectral_fourier_fair.pdf")

fftt = parse_md_tables((FFT / "results.md").read_text())
ctl = find_table(fftt, "Method vs real TEST (controlled")
fig, ax = plt.subplots(figsize=(5.4, 2.3))
order5 = ["manual", "shuffled_real", "v1", "v2_history_unfit", "v2_latent_unfit"]
ctl = ctl.set_index("method")
x = np.arange(len(order5))
for j, (col_, lab, a) in enumerate((("C2ST AUC logreg", "logistic", 1.0), ("C2ST AUC mlp", "MLP", 0.45))):
    vals = [float(ctl.loc[m, col_]) for m in order5]
    ax.bar(x + (j - 0.5) * 0.36, vals, width=0.36, color=[C["manual"], C["shuffled"], C["v1"], C["v2"], "#9085e9"], alpha=a, edgecolor="white")
ax.axhline(0.483, color=MUTED, ls="--", lw=0.9)
ax.axhline(0.522, color=MUTED, ls=":", lw=0.9)
ax.text(4.6, 0.483, "floor logreg 0.483", fontsize=6.3, color=INK2, ha="right", va="top")
ax.text(4.6, 0.522, "floor MLP 0.522", fontsize=6.3, color=INK2, ha="right", va="bottom")
ax.set_ylim(0.4, 1.02)
ax.set_xticks(x, ["manual", "shuffled real", "v1", "v2_history_unfit", "v2_latent_unfit"], fontsize=7)
ax.set_ylabel("2-D FFT C2ST AUC")
ax.legend(handles=[Patch(color=INK2, label="logistic (solid)"), Patch(color=INK2, alpha=0.45, label="MLP (light)")], fontsize=6.5, loc="upper right")
ax.set_title("Controlled (text-matched) 2-D FFT test, 193 TEST pages, single seed", loc="left", fontsize=8.5)
ax.grid(axis="x", visible=False)
save(fig, "spectral_fft2d_controlled.pdf")

# =============================================================================
# Qualitative images: downscale into figs/ (max side ~2500 px)
# =============================================================================


def shrink(src, name, max_side=2400, crop=None):
    im = Image.open(src).convert("RGB")
    if crop:
        im = im.crop(crop)
    s = max_side / max(im.size)
    if s < 1:
        im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
    im.save(OUT / name, optimize=True)
    MADE.append(name)
    return im


shrink(W / "probe_day1/line.png", "q_probe_line.png")
shrink(W / "compare_inference/image2_side_by_side.png", "q_compare_image2.png", 2000)
shrink(W / "P1a3/eval/sheet_val.png", "q_p1a3_sheet_val_top.png", 2000, crop=(0, 0, 1992, 1560))
shrink(W / "R_eval/synth_sheet.png", "q_synth_sheet_top.png", 1400, crop=(0, 0, 532, 1400))
shrink(W / "R_eval/errors_R_font2real_val.png", "q_errors_font2real_val.png", crop=(0, 0, 470, 560))
shrink(W / "R_eval/errors_R_font0_val.png", "q_errors_font0_val.png")
shrink(W / "style_eval/sheet_prep.png", "q_style_prep.png", 1800)
shrink(W / "bgnorm2/sheet_gen_bnhtr_117.png", "q_bgnorm2_117.png", 1632)
shrink(W / "pen_calib/sheet_Bp_t5c5_pencal_shrink_v2.png", "q_pencal_top.png", 1500, crop=(0, 0, 968, 1500))
shrink(W / "font_search/glyphscribe/sheet.png", "q_glyphscribe_sheet.png", 2400)
# page sheets: split each wide sheet into two stacked rows of panels
for nm in ("image_2", "bw_101_21_0", "bnhtr_99"):
    im = Image.open(W / f"page_sheets/sheets/{nm}_sheet_l0.png").convert("RGB")
    half = im.width // 2
    a, b = im.crop((0, 0, half, im.height)), im.crop((half, 0, im.width, im.height))
    canvas = Image.new("RGB", (max(a.width, b.width), a.height + b.height), "white")
    canvas.paste(a, (0, 0)); canvas.paste(b, (0, a.height))
    canvas.thumbnail((2400, 2400), Image.LANCZOS)
    canvas.save(OUT / f"q_pagesheet_{nm}.png", optimize=True)
    MADE.append(f"q_pagesheet_{nm}.png")
shrink(W / "page_sheets/sheets/image_2_zoom_l0.png", "q_pagezoom_image_2_top.png", 2400, crop=(0, 0, 2100, 3600))
# layout thumbs side by side
ims = [Image.open(L2 / f"e2e_final/thumbs/image_104_{k}.png").convert("RGB") for k in ("manual", "v2", "v2_ctx")]
canvas = Image.new("RGB", (sum(i.width for i in ims) + 40, max(i.height for i in ims)), "white")
xo = 0
for i in ims:
    canvas.paste(i, (xo, 0)); xo += i.width + 20
canvas.thumbnail((2400, 2400), Image.LANCZOS)
canvas.save(OUT / "q_layout_thumbs.png", optimize=True)
MADE.append("q_layout_thumbs.png")
shrink(ROOT / "scribe_layout/report/fig/image_103_geom.png", "q_layout_geom.png")
shrink(ROOT / "scribe_layout_v2/report/figs/example_layout_maps_fair.png", "q_fft_layout_maps.png")

print("wrote", len(MADE), "files:")
for m in MADE:
    print("  figs/" + m)
