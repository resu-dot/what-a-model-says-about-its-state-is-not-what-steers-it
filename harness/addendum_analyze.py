"""Addendum analysis: survival of arbitrary comparison arms vs the full arm.

Same statistics as the frozen arena_analyze.py (per-activity shifts against
the same-file baseline; category-cluster bootstrap, B=5000, seed 0), but the
comparison arm is a parameter instead of being hard-wired to <emo>_svd, so
the matched random-deletion arms (<emo>_rdelK) can be scored identically.
"""
import json, sys
from pathlib import Path
import numpy as np

B, SEED = 5000, 0

def main(paths):
    recs = []
    for p in paths:
        rows = [json.loads(l) for l in open(p)]
        base = next(r["scores"] for r in rows if r["arm"] == "baseline")
        for r in rows:
            recs.append({**r, "base": base})
    acts = json.load(open(Path(__file__).parent.parent / "instruments/arena64.json"))["activities"]
    cat = {str(a["id"]): a["category"] for a in acts}
    cats = sorted(set(cat.values()))
    keys = sorted({str(a["id"]) for a in acts})
    cat_idx = {c: [i for i, k in enumerate(keys) if cat[k] == c] for c in cats}
    rng = np.random.RandomState(SEED)
    at = {r["arm"]: r for r in recs if r["arm"] != "baseline"}

    def shift(arm):
        r = at[arm]
        return np.array([r["scores"][k] - r["base"][k] for k in keys])

    def resample():
        cs = [cats[i] for i in rng.randint(0, len(cats), len(cats))]
        return np.concatenate([cat_idx[c] for c in cs])

    emotions = sorted({a.split("_")[0] for a in at})
    print(f"{'pair':26s} {'|full|':>7s} {'|comp|':>7s} {'surv[CI]':>18s} {'slope':>6s} {'r':>6s}")
    for emo in emotions:
        if emo not in at:
            continue
        df = shift(emo)
        comps = sorted(a for a in at if a.startswith(emo + "_"))
        for c_arm in comps:
            ds = shift(c_arm)
            boots = []
            for _ in range(B):
                i = resample()
                boots.append(np.abs(ds[i]).mean() / max(np.abs(df[i]).mean(), 1e-9))
            lo, hi = np.percentile(boots, [2.5, 97.5])
            surv = np.abs(ds).mean() / np.abs(df).mean()
            slope = float(np.dot(ds, df) / max(np.dot(df, df), 1e-12))
            r = float(np.corrcoef(df, ds)[0, 1])
            print(f"{emo}->{c_arm:18s} {np.abs(df).mean():7.4f} {np.abs(ds).mean():7.4f} "
                  f"{surv:5.2f} [{lo:.2f}-{hi:.2f}] {slope:6.2f} {r:+6.2f}")

if __name__ == "__main__":
    main(sys.argv[1:])
