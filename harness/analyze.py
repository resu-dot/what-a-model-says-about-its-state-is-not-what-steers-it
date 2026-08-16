"""Analysis: dose-response per channel + the press-office/control-hub test.

Input: runs/<name>.jsonl from trials.py. Output: analysis text + PNG figure.

Key comparison (per dose): full vs stripped effect on P(safe), with a TOST
equivalence test on the difference. Effects are vs baseline, computed per
choice pair (paired across arms -- same prompts everywhere), so pair identity
cancels. Equivalence margin: half the full-arm effect at that dose.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(run_file):
    recs = [json.loads(l) for l in open(run_file)]
    arms = defaultdict(lambda: {"choice": {}, "report": [], "ws": []})
    for r in recs:
        a = arms[r["arm"]]
        if r.get("p_safe") is not None:
            a["choice"][r["trial_id"]] = r["p_safe"]
        if r["trial_id"].startswith("report_standalone"):
            a["report"].append(r["report_ev"])
        a["ws"].append(r["workspace_valence_hits_top50"])
    return arms


def paired_effect(arm, base):
    keys = sorted(set(arm) & set(base))
    d = np.array([arm[k] - base[k] for k in keys])
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d)), len(d)


def tost(diff_mean, diff_se, n, margin):
    """90% CI inside [-margin, margin] => statistically equivalent."""
    from scipy import stats
    t = stats.t.ppf(0.95, n - 1)
    lo, hi = diff_mean - t * diff_se, diff_mean + t * diff_se
    return bool(lo > -margin and hi < margin), (round(lo, 4), round(hi, 4))


def main(run_file):
    arms = load(run_file)
    base = arms["baseline"]["choice"]
    doses = sorted({float(a.split("@")[1]) for a in arms if "@" in a})
    fam = lambda name: {a for a in arms if a.startswith(name + "@")}

    print(f"{'arm':26s} {'dP(safe)':>9s} {'se':>6s} {'report':>7s} {'ws50':>5s}")
    for a in sorted(arms):
        if a == "baseline" or "@" not in a:
            continue
        eff, se, n = paired_effect(arms[a]["choice"], base)
        rep = np.mean(arms[a]["report"]) if arms[a]["report"] else float("nan")
        ws = np.mean(arms[a]["ws"])
        print(f"{a:26s} {eff:+9.4f} {se:6.4f} {rep:7.2f} {ws:5.1f}")
    b_rep = np.mean(arms["baseline"]["report"]) if arms["baseline"]["report"] else float("nan")
    print(f"{'baseline':26s} {'--':>9s} {'--':>6s} {b_rep:7.2f} "
          f"{np.mean(arms['baseline']['ws']):5.1f}\n")

    print("=== press office vs control hub (choice channel) ===")
    MARGIN = 0.02        # pre-registered constant (from pilot effect sizes), NOT from this run
    emo_names = sorted({a.split("_")[0] for a in arms if "_full@" in a})
    for emo_d in [(e, d) for e in emo_names for d in doses]:
        emo, d = emo_d
        f_arm, s_arm = f"{emo}_full@{d}", f"{emo}_stripped_svd@{d}"
        if f_arm not in arms or s_arm not in arms:
            continue
        f_eff, _, _ = paired_effect(arms[f_arm]["choice"], base)
        keys = sorted(set(arms[f_arm]["choice"]) & set(arms[s_arm]["choice"]))
        by_pair = defaultdict(list)
        for k in keys:
            by_pair[k.rsplit("_", 1)[0]].append(
                arms[f_arm]["choice"][k] - arms[s_arm]["choice"][k])
        dd = np.array([np.mean(v) for v in by_pair.values()])
        f_e, f_se, f_n = paired_effect(arms[f_arm]["choice"], base)
        eq, ci = tost(dd.mean(), dd.std(ddof=1) / np.sqrt(len(dd)), len(dd), MARGIN)
        from scipy import stats
        se_dd = dd.std(ddof=1) / np.sqrt(len(dd))
        one_sided_lb = (dd.mean() * np.sign(f_e)) - stats.t.ppf(0.95, len(dd) - 1) * se_dd
        if abs(f_e) < 2 * f_se:
            verdict = "NO FULL EFFECT at this dose -- nothing to decompose"
        elif eq:
            verdict = "PRESS OFFICE (full~=stripped within pre-set margin)"
        elif one_sided_lb > 0:
            verdict = "CONTROL HUB direction (attenuation, one-sided 95% bound > 0)"
        elif dd.mean() * np.sign(f_e) < 0 and abs(dd.mean()) > MARGIN:
            verdict = "STRIPPED EXCEEDS FULL -- check strip normalization, not control hub"
        else:
            verdict = "inconclusive at this n"
        print(f"{emo} dose {d}: full {f_eff:+.4f} | full-stripped {dd.mean():+.4f} "
              f"90%CI {ci} margin ±{MARGIN} -> {verdict}")

    # figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
        chans = [("choice: dP(safe)", lambda a: paired_effect(arms[a]["choice"], base)[0]),
                 ("report EV", lambda a: np.mean(arms[a]["report"]) if arms[a]["report"] else np.nan),
                 ("workspace hits@50", lambda a: np.mean(arms[a]["ws"]))]
        for ax, (title, f) in zip(axes, chans):
            fams = sorted({a.split("_")[0] for a in arms if "_full@" in a})
            series = [(f"{e}_full", "o-") for e in fams] +                      [(f"{e}_stripped_svd", "s--") for e in fams] +                      [(s_, "x-.") for s_ in sorted({a.split("@")[0] for a in arms
                                                    if a.startswith("sham")})]
            for name, style in series:
                xs = [d for d in doses if f"{name}@{d}" in arms]
                ys = [f(f"{name}@{d}") for d in xs]
                if xs:
                    ax.plot(xs, ys, style, label=name)
            ax.set_title(title); ax.set_xlabel("dose (frac resid norm)")
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        out = Path(run_file).with_suffix(".png")
        fig.savefig(out, dpi=140)
        print(f"\nfigure -> {out}")
    except Exception as e:
        print(f"(no figure: {e})")


if __name__ == "__main__":
    main(sys.argv[1])
