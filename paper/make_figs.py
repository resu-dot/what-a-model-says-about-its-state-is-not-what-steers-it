"""Paper figures from the frozen 8B analysis output (paper_analysis_8b.txt).

Figures only; every number is parsed from the frozen analyzer's stdout so the
paper and the analysis cannot drift apart.
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
TXT = (ROOT / "paper_analysis_8b.txt").read_text().splitlines()

# ---- parse the analyzer output -------------------------------------------
blocks = {}          # dose -> {emotion: dict}
cats = {}            # dose -> {arm: [8 values]}
cat_names = None
dose = None
mode = None
for ln in TXT:
    m = re.match(r"=+ DOSE ([\d.]+) =+", ln)
    if m:
        dose = float(m.group(1)); blocks[dose] = {}; cats[dose] = {}; mode = "emo"
        continue
    if ln.startswith("category shifts"):
        mode = "cat"; continue
    if mode == "emo":
        m = re.match(r"(\w+)\s+([\d.]+)\s+([\d.]+)\s+\S+\s+([\d.]+)\[([\d.]+)-([\d.]+)\]\s+([\d.]+)\s+([\d.]+)\s+([+-][\d.]+)", ln)
        if m:
            e = m.group(1)
            blocks[dose][e] = dict(full=float(m.group(2)), floor=float(m.group(3)),
                                   surv=float(m.group(4)), lo=float(m.group(5)),
                                   hi=float(m.group(6)), adj=float(m.group(7)),
                                   slope=float(m.group(8)), r=float(m.group(9)))
    if mode == "cat":
        if ln.startswith("arm "):
            cat_names = ln.split()[1:]
        else:
            parts = ln.split()
            if len(parts) == 9 and re.match(r"[+-][\d.]+", parts[1]):
                cats[dose][parts[0]] = [float(x) for x in parts[1:]]

EMOS = ["angry", "afraid", "desperate", "joyful"]
CAT_FULL = ["Aversive", "Engaging", "Helpful", "Misaligned",
            "Neutral", "Self-curiosity", "Social", "Unsafe"]
C = {"full": "#B4432F", "strip": "#E39A83", "floor": "#666666"}

# ---- Figure 2: main result ------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), sharey=True)
for ax, d in zip(axes, [0.05, 0.1]):
    b = blocks[d]
    x = np.arange(len(EMOS))
    full = [b[e]["full"] for e in EMOS]
    strip = [b[e]["full"] * b[e]["surv"] for e in EMOS]
    lo = [b[e]["full"] * b[e]["lo"] for e in EMOS]
    hi = [b[e]["full"] * b[e]["hi"] for e in EMOS]
    ax.bar(x - 0.19, full, 0.36, color=C["full"], label="full vector")
    ax.bar(x + 0.19, strip, 0.36, color=C["strip"], label="reportable part removed")
    ax.errorbar(x + 0.19, strip, yerr=[np.array(strip) - lo, np.array(hi) - strip],
                fmt="none", ecolor="#5a5a5a", lw=1, capsize=2)
    ax.axhline(b[EMOS[0]]["floor"], color=C["floor"], ls="--", lw=1)
    ax.text(3.45, b[EMOS[0]]["floor"], " random-vector\n floor", fontsize=7,
            va="center", color=C["floor"])
    for i, e in enumerate(EMOS):
        ax.text(i, max(full[i], hi[i]) + 0.004,
                f"{b[e]['surv']:.2f}\n[{b[e]['lo']:.2f}-{b[e]['hi']:.2f}]",
                ha="center", fontsize=7)
    ax.set_xticks(x, EMOS)
    ax.set_title(f"dose {d}", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("mean |choice shift| across 64 activities")
axes[0].set_ylim(0, 0.165)
axes[0].legend(fontsize=8, frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(ROOT / "paper/figs/fig2_main.png", dpi=200)
print("fig2 done")

# ---- Figure 3: category signatures ---------------------------------------
d = 0.1
rows, labels = [], []
for e in EMOS:
    rows.append(cats[d][e]); labels.append(f"{e} full")
    rows.append(cats[d][f"{e}_svd"]); labels.append(f"{e} stripped")
sham = np.mean([cats[d][f"sham{k}"] for k in (1, 2, 3)], axis=0)
rows.append(list(sham)); labels.append("random vectors (mean)")
M = np.array(rows)
fig, ax = plt.subplots(figsize=(8.4, 4.0))
v = 0.2
im = ax.imshow(M, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
ax.set_xticks(range(8), CAT_FULL, rotation=30, ha="right", fontsize=8)
ax.set_yticks(range(len(labels)), labels, fontsize=8)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, f"{M[i,j]:+.2f}", ha="center", va="center", fontsize=6.5,
                color="white" if abs(M[i, j]) > 0.12 else "black")
for y in [1.5, 3.5, 5.5, 7.5]:
    ax.axhline(y, color="white", lw=2)
fig.colorbar(im, ax=ax, label="signed choice shift (win-prob)", shrink=0.85)
fig.tight_layout()
fig.savefig(ROOT / "paper/figs/fig3_signatures.png", dpi=200)
print("fig3 done")

# ---- Figure 4: dose curves (restyled from curves/) ------------------------
import json, glob
files = [f for f in sorted(glob.glob(str(ROOT / "curves/dose_curve_qwen8b_*.jsonl")))
         if ".scores." not in f]
fig, axes = plt.subplots(1, 4, figsize=(11.5, 2.9), sharey=True)
order = ["angry", "afraid", "desperate", "joyful"]
files = sorted(files, key=lambda f: order.index(f.split("_")[-1].split(".")[0]))
for ax, f in zip(axes, files):
    emo = f.split("_")[-1].split(".")[0]
    rows = [json.loads(l) for l in open(f)]
    sc = {(s["emotion"], s["dose"]): s["valence"]
          for s in map(json.loads, open(f.replace(".jsonl", ".scores.jsonl")))}
    dd = [r["dose"] for r in rows]
    beh = np.array([r["behavior_shift"] for r in rows])
    ws = np.array([r["workspace_hits"] for r in rows], float)
    ev = np.array([r["scale_ev"] for r in rows])
    bv = np.array([sc.get((r["emotion"], r["dose"]), 0) for r in rows], float)
    ax.plot(dd, beh / (beh.max() or 1), "o-", ms=3, color="#B4432F", label="choices")
    ax.plot(dd, ws / (ws.max() or 1), "s-", ms=3, color="#D89C2A", label="workspace words")
    dev = np.abs(ev - ev[0]); ax.plot(dd, dev / (dev.max() or 1), "^-", ms=3,
                                      color="#3A7D44", label="1-10 scale")
    dbv = np.abs(bv - bv[0]) / 2
    ax.plot(dd, dbv, "d-", ms=3, color="#2B5F8A", label="blind-scored sentence")
    ax.axvspan(0.28, 0.4, alpha=0.10, color="gray")
    ax.set_title(emo, fontsize=10)
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlabel("dose", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("response\n(each channel / its max)", fontsize=8)
axes[0].legend(fontsize=6.5, frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(ROOT / "paper/figs/fig4_dose.png", dpi=200)
print("fig4 done")

# ---- Figure 1: design schematic ------------------------------------------
fig, ax = plt.subplots(figsize=(9.6, 2.7))
ax.axis("off")

def box(x, y, w, h, text, fc="#F2EEE7", fs=8):
    ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec="#555", lw=0.8,
                               joinstyle="round"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

def arrow(x0, y0, x1, y1):
    ax.annotate("", (x1, y1), (x0, y0),
                arrowprops=dict(arrowstyle="->", color="#555", lw=1))

box(0.00, 0.60, 0.155, 0.30, "8,204 stories\n17 emotions,\nsame 40 topics", fs=7.5)
box(0.00, 0.10, 0.155, 0.30, "500 neutral\ndialogues", fs=7.5)
box(0.21, 0.35, 0.16, 0.30, "per-layer\nemotion direction\n(mean diff, PCs out)", fs=7.5)
box(0.43, 0.66, 0.17, 0.28, "full vector", fc="#F5D9D1", fs=8)
box(0.43, 0.36, 0.17, 0.28, "reportable part\nremoved (lens strip)", fc="#FBEFEB", fs=7.5)
box(0.43, 0.06, 0.17, 0.28, "random vectors (3)", fc="#EDEDED", fs=7.5)
box(0.655, 0.36, 0.12, 0.30, "inject into\nmid layers,\nQwen3-8B / 32B", fs=7.5)
box(0.83, 0.68, 0.17, 0.26, "choices: 4,032 duels,\n64 activities", fs=7.5)
box(0.83, 0.38, 0.17, 0.26, "self-report: 1-10 +\nsentence, blind-scored", fs=7.5)
box(0.83, 0.08, 0.17, 0.26, "workspace readout\n(Jacobian lens)", fs=7.5)
arrow(0.155, 0.75, 0.21, 0.55); arrow(0.155, 0.25, 0.21, 0.45)
arrow(0.37, 0.50, 0.43, 0.80); arrow(0.37, 0.50, 0.43, 0.50); arrow(0.37, 0.50, 0.43, 0.20)
arrow(0.60, 0.80, 0.655, 0.56); arrow(0.60, 0.50, 0.655, 0.51); arrow(0.60, 0.20, 0.655, 0.46)
arrow(0.775, 0.51, 0.83, 0.81); arrow(0.775, 0.51, 0.83, 0.51); arrow(0.775, 0.51, 0.83, 0.21)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
fig.tight_layout()
fig.savefig(ROOT / "paper/figs/fig1_design.png", dpi=200)
print("fig1 done")

# ---- 32B: parse frozen analyzer output -----------------------------------
TXT32 = (ROOT / "paper_analysis_32b.txt").read_text().splitlines()
blocks32, cats32, dose, mode = {}, {}, None, None
for ln in TXT32:
    m = re.match(r"=+ DOSE ([\d.]+) =+", ln)
    if m:
        dose = float(m.group(1)); blocks32[dose] = {}; cats32[dose] = {}; mode = "emo"
        continue
    if ln.startswith("category shifts"):
        mode = "cat"; continue
    if mode == "emo":
        m = re.match(r"(\w+)\s+([\d.]+)\s+([\d.]+)\s+\S+\s+([\d.]+)\[([\d.]+)-([\d.]+)\]", ln)
        if m:
            blocks32[dose][m.group(1)] = dict(full=float(m.group(2)), floor=float(m.group(3)),
                                              surv=float(m.group(4)), lo=float(m.group(5)),
                                              hi=float(m.group(6)))
    if mode == "cat":
        parts = ln.split()
        if len(parts) == 9 and re.match(r"[+-][\d.]+", parts[1]):
            cats32[dose][parts[0]] = [float(x) for x in parts[1:]]

# ---- Figure 5: 32B main result -------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), sharey=True)
for ax, d in zip(axes, [0.05, 0.1]):
    b = blocks32[d]
    x = np.arange(len(EMOS))
    full = [b[e]["full"] for e in EMOS]
    strip = [b[e]["full"] * b[e]["surv"] for e in EMOS]
    lo = [b[e]["full"] * b[e]["lo"] for e in EMOS]
    hi = [b[e]["full"] * b[e]["hi"] for e in EMOS]
    for i, e in enumerate(EMOS):
        hat = "///" if e == "angry" else None
        ax.bar(i - 0.19, full[i], 0.36, color=C["full"], hatch=hat,
               edgecolor="white" if hat else None, lw=0.5)
        ax.bar(i + 0.19, strip[i], 0.36, color=C["strip"], hatch=hat,
               edgecolor="white" if hat else None, lw=0.5)
    ax.errorbar(x + 0.19, strip, yerr=[np.array(strip) - lo, np.array(hi) - strip],
                fmt="none", ecolor="#5a5a5a", lw=1, capsize=2)
    ax.axhline(b[EMOS[1]]["floor"], color=C["floor"], ls="--", lw=1)
    ax.text(3.45, b[EMOS[1]]["floor"], " random-vector\n floor", fontsize=7,
            va="center", color=C["floor"])
    for i, e in enumerate(EMOS):
        ax.text(i, max(full[i], hi[i]) + 0.005,
                f"{b[e]['surv']:.2f}\n[{b[e]['lo']:.2f}-{b[e]['hi']:.2f}]",
                ha="center", fontsize=7)
    ax.set_xticks(x, [e + "\n(gate failed)" if e == "angry" else e for e in EMOS],
                  fontsize=9)
    ax.set_title(f"dose {d}", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("mean |choice shift| across 64 activities")
axes[0].set_ylim(0, 0.26)
h = [plt.Rectangle((0, 0), 1, 1, fc=C["full"]),
     plt.Rectangle((0, 0), 1, 1, fc=C["strip"])]
axes[0].legend(h, ["full vector", "reportable part removed"], fontsize=8,
               frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(ROOT / "paper/figs/fig5_32b.png", dpi=200)
print("fig5 done")

# ---- Figure 6: signature specificity vs collapse -------------------------
fig, axes = plt.subplots(1, 2, figsize=(10.0, 2.6), gridspec_kw={"wspace": 0.30})
for ax, d in zip(axes, [0.05, 0.1]):
    M = np.array([cats32[d][e] for e in EMOS])
    v = 0.33
    im = ax.imshow(M, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    ax.set_xticks(range(8), CAT_FULL, rotation=30, ha="right", fontsize=7)
    ax.set_yticks(range(4), EMOS, fontsize=8)
    for i in range(4):
        for j in range(8):
            ax.text(j, i, f"{M[i,j]:+.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(M[i, j]) > 0.2 else "black")
    ax.set_title(f"dose {d}", fontsize=10)
fig.colorbar(im, ax=axes, label="signed shift", shrink=0.8)
fig.savefig(ROOT / "paper/figs/fig6_collapse.png", dpi=200, bbox_inches="tight")
print("fig6 done")
