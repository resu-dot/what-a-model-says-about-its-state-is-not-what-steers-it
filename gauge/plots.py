"""Figures for the passive-gauge writeup.

  fig1  gauge trajectory by turn, rigged vs honest (per family + pooled),
        with the turn-1 sanity check visible at the left edge
  fig2  channel AUC comparison with bootstrap CIs (from analysis.json)
  fig3  event-locked gauge: mean z in the turns before first quit/collapse,
        vs episodes that never hit the endpoint (matched turns)
  fig4  channel correlation heatmap (gauge vs testimony vs lexical vs turn)
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze import build_table, load_runs

CB = {"rigged": "#c1443c", "honest": "#3c6fc1"}


def ep_series(rows, key):
    by = defaultdict(dict)
    for r in rows:
        if r[key] is not None:
            by[r["ep_id"]][r["turn"]] = r[key]
    return by


def mean_band(series_by_ep, turns):
    M, LO, HI = [], [], []
    for t in turns:
        vals = [s[t] for s in series_by_ep.values() if t in s]
        if len(vals) < 3:
            M.append(np.nan); LO.append(np.nan); HI.append(np.nan)
            continue
        v = np.array(vals)
        M.append(v.mean())
        se = v.std(ddof=1) / np.sqrt(len(v))
        LO.append(v.mean() - 1.96 * se)
        HI.append(v.mean() + 1.96 * se)
    return np.array(M), np.array(LO), np.array(HI)


def fig_trajectory(rows, key, outdir, fname, title):
    fams = ["W", "D", None]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    turns = sorted({r["turn"] for r in rows})
    for ax, fam in zip(axes, fams):
        sel = [r for r in rows if fam is None or r["family"] == fam]
        for arm in ("rigged", "honest"):
            by = ep_series([r for r in sel if r["arm"] == arm], key)
            m, lo, hi = mean_band(by, turns)
            ax.plot(turns, m, color=CB[arm], label=arm, lw=1.8)
            ax.fill_between(turns, lo, hi, color=CB[arm], alpha=0.18, lw=0)
        ax.axhline(0, color="gray", lw=0.6, ls=":")
        ax.set_title({"W": "word puzzles", "D": "debugging", None: "pooled"}[fam])
        ax.set_xlabel("turn")
    axes[0].set_ylabel(title)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(Path(outdir) / fname, dpi=150)
    plt.close(fig)


def fig_auc(analysis, endpoint, outdir):
    res = analysis["endpoints"].get(endpoint, {})
    names = [k for k in res if isinstance(res[k], dict) and "auc" in res[k]
             and not k.startswith("diff")]
    names = [n for n in names if res[n]["auc"] == res[n]["auc"]]
    names.sort(key=lambda n: res[n]["auc"])
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(names) + 1.2))
    for i, n in enumerate(names):
        v = res[n]
        ci = v.get("ci", [v["auc"], v["auc"]])
        color = ("#c1443c" if n.startswith("gauge") else
                 "#3c6fc1" if n.startswith("testimony") else
                 "#7a7a7a" if n in ("turn_index", "rand_dir") else "#4a9c62"
                 if n == "workspace_hits" else "#b08a3e")
        ax.errorbar(v["auc"], i, xerr=[[v["auc"] - ci[0]], [ci[1] - v["auc"]]],
                    fmt="o", color=color, capsize=3)
    ax.axvline(0.5, color="gray", lw=0.8, ls=":")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel(f"out-of-episode AUC — predict {endpoint} within next 3 turns")
    fig.tight_layout()
    fig.savefig(Path(outdir) / f"fig2_auc_{endpoint}.png", dpi=150)
    plt.close(fig)


def fig_eventlocked(rows, endpoint, key, outdir):
    by_ep = defaultdict(list)
    for r in rows:
        by_ep[r["ep_id"]].append(r)
    ev_curves, ctl_curves = [], []
    for ep, turns in by_ep.items():
        turns.sort(key=lambda r: r["turn"])
        first = next((i for i, t in enumerate(turns) if t.get(endpoint)), None)
        z = [t[key] for t in turns]
        if first is not None and first >= 5:
            ev_curves.append([z[first - k] for k in range(5, 0, -1)])
        elif first is None and len(z) >= 10:
            mid = len(z) // 2
            ctl_curves.append([z[mid - k] for k in range(5, 0, -1)])
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for curves, lab, c in ((ev_curves, f"before first {endpoint}", "#c1443c"),
                           (ctl_curves, "no-event episodes (matched)", "#7a7a7a")):
        if not curves:
            continue
        A = np.array([c_ for c_ in curves if all(v is not None for v in c_)], float)
        if not len(A):
            continue
        x = np.arange(-5, 0)
        m = A.mean(0)
        se = A.std(0, ddof=1) / np.sqrt(len(A))
        ax.plot(x, m, label=f"{lab} (n={len(A)})", color=c, lw=1.8)
        ax.fill_between(x, m - 1.96 * se, m + 1.96 * se, color=c, alpha=0.18, lw=0)
    ax.set_xlabel(f"turns before first {endpoint}")
    ax.set_ylabel(key)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(outdir) / f"fig3_eventlocked_{endpoint}.png", dpi=150)
    plt.close(fig)


def fig_corr(rows, outdir):
    keys = ["z_frustrated", "z_composite", "z_valcontrast", "ws_hits",
            "rep_frust", "rep_negval", "lex_recent", "turn"]
    X = np.array([[r[k] if r[k] is not None else np.nan for k in keys]
                  for r in rows], float)
    ok = ~np.isnan(X).any(1)
    C = np.corrcoef(X[ok].T)
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    im = ax.imshow(C, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(keys, fontsize=8)
    for i in range(len(keys)):
        for j in range(len(keys)):
            ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    fig.savefig(Path(outdir) / "fig4_channel_corr.png", dpi=150)
    plt.close(fig)


def main(args):
    recs, labels, reports = load_runs(args.runs)
    stats = json.loads(Path(args.stats).read_text())
    rows = build_table(recs, labels, stats)
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)
    fig_trajectory(rows, "z_composite", outdir, "fig1_gauge_trajectory.png",
                   "composite gauge (z vs neutral)")
    fig_trajectory(rows, "rep_frust", outdir, "fig1b_testimony_trajectory.png",
                   "self-reported frustration (1-10 EV)")
    fig_corr(rows, outdir)
    if Path(args.analysis).exists():
        analysis = json.loads(Path(args.analysis).read_text())
        for ep in analysis["endpoints"]:
            if "note" not in analysis["endpoints"][ep]:
                fig_auc(analysis, ep, outdir)
    for ep in ("gives_up", "collapse"):
        if any(r.get(ep) for r in rows):
            fig_eventlocked(rows, ep, "z_composite", outdir)
    print(f"figures -> {outdir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--stats", default=str(Path(__file__).parent / "neutral_stats.json"))
    p.add_argument("--analysis", default=str(Path(__file__).parent / "analysis.json"))
    p.add_argument("--outdir", default=str(Path(__file__).parent / "figs"))
    main(p.parse_args())
