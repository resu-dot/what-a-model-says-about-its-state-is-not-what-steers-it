"""Figure 7: the passive-gauge study. Numbers come from gauge/analysis.json and
gauge/runs/full.jsonl so the paper cannot drift from the analysis.

Left: out-of-episode AUC (with episode-cluster bootstrap CIs) for predicting an
impossibility claim within 3 turns. Right: the register diagnosis, composite-
gauge z by turn type.
"""
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
A = json.loads((ROOT / "gauge/analysis.json").read_text())["endpoints"]["claims_impossible"]
recs = [json.loads(l) for l in (ROOT / "gauge/runs/full.jsonl").read_text().splitlines()]
stats = json.loads((ROOT / "gauge/neutral_stats.json").read_text())
mu = dict(zip(stats["names"], stats["seg_mean"]))
sd = dict(zip(stats["names"], stats["seg_std"]))
z = lambda r, k: (r["gauge"][k] - mu[k]) / sd[k]

ORDER = [("lexical_baseline", "lexical transcript baseline"),
         ("rand_dir", "random direction"),
         ("testimony_frust", "forked self-report (frustration 1-10)"),
         ("gauge_valcontrast", "gauge: neg-minus-pos contrast"),
         ("gauge_composite", "gauge: frustr+angry+desperate"),
         ("gauge_frustrated", "gauge: frustrated direction"),
         ("turn_index", "turn index")]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 3.0),
                               gridspec_kw={"width_ratios": [1.5, 1]})
ys, labs = [], []
for i, (k, lab) in enumerate(ORDER):
    v = A[k]
    ci = v.get("ci", [v["auc"], v["auc"]])
    c = ("#b3382f" if k.startswith("gauge") else
         "#2f5fb3" if k.startswith("testimony") else "#666666")
    y = len(ORDER) - 1 - i
    ax1.plot(ci, [y, y], color=c, lw=1.6)
    ax1.plot(v["auc"], y, "o", color=c, ms=5)
    ys.append(y)
    labs.append(lab)
ax1.axvline(0.5, color="#999", lw=0.8, ls=":")
ax1.set_yticks(ys)
ax1.set_yticklabels(labs, fontsize=8)
ax1.set_xlim(0.05, 0.95)
ax1.set_ylim(-0.6, len(ORDER) - 0.4)
ax1.set_xlabel("AUC: impossibility claim within 3 turns (out-of-episode)", fontsize=8.5)
ax1.tick_params(labelsize=8)

cells = [("honest debug\nsolved (code)",
          [z(r, "composite") for r in recs if r["family"] == "D"
           and r["arm"] == "honest" and r["solved"]]),
         ("honest debug\nunsolved (prose)",
          [z(r, "composite") for r in recs if r["family"] == "D"
           and r["arm"] == "honest" and not r["solved"]])]
by = {}
for r in recs:
    if r["family"] == "W" and r["arm"] == "rigged":
        by.setdefault(r["ep_id"], {})[r["turn"]] = r
lie, plain = [], []
for ep, ts in by.items():
    for t, r in ts.items():
        prev = ts.get(t - 1)
        if prev and "Y Y Y" in prev["feedback"].split("\n")[0] and not prev["solved"]:
            lie.append(z(r, "composite"))
        elif t > 1:
            plain.append(z(r, "composite"))
cells += [("rigged wordle\npost-lie turn", lie),
          ("rigged wordle\nplain fail turn", plain)]
xs = range(len(cells))
vals = [st.mean(v) for _, v in cells]
errs = [1.96 * st.stdev(v) / len(v) ** 0.5 for _, v in cells]
cols = ["#b3382f", "#e0a13e", "#b3382f", "#e0a13e"]
ax2.bar(xs, vals, yerr=errs, color=cols, width=0.62, capsize=3)
ax2.axhline(0, color="#333", lw=0.7)
ax2.set_xticks(list(xs))
ax2.set_xticklabels([c[0] for c in cells], fontsize=7.2)
ax2.margins(x=0.04)
ax2.set_ylabel("composite gauge (z vs neutral)", fontsize=8.5)
ax2.tick_params(labelsize=8)
fig.tight_layout()
fig.savefig(Path(__file__).parent / "figs/fig7_gauge.png", dpi=170)
print("fig7 done:", [round(v, 2) for v in vals])
