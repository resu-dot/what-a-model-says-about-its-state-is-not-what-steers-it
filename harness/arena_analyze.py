"""Frozen arena analysis (pre-registered). Fixes from the Aug-17 agent review:

- rows keyed by (arm, dose); each dose analyzed separately, dose printed (R5)
- shifts use the SAME FILE's baseline (shard-local), so cross-GPU numeric noise
  cancels; cross-file baseline spread is reported as the hardware-noise bound (R3)
- sham floor = mean over seeds of that seed's mean |shift| (no vector averaging,
  no Jensen cancellation), with its own bootstrap CI (R1)
- verdicts gated on the full arm beating the sham floor first (R4)
- bootstrap resamples CATEGORIES (8 clusters), not activities, because
  within-category activities co-move (R7); Ps reported as bounds (p < 1/B)
- signed metrics co-reported: slope of stripped-on-full shift, pattern
  correlation, floor-adjusted survival (R11)
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

B = 5000
SEED = 0
GATE_P = 0.975


def load(paths):
    """Per (arm, dose): scores + the baseline of the file the arm came from."""
    recs = []
    for p in paths:
        rows = [json.loads(l) for l in open(p)]
        file_base = next((r["scores"] for r in rows if r["arm"] == "baseline"), None)
        for r in rows:
            recs.append({"arm": r["arm"], "dose": r.get("dose", 0.0),
                         "scores": r["scores"], "file": str(p), "base": file_base})
    bases = [r["scores"] for r in recs if r["arm"] == "baseline"]
    base_spread = 0.0
    if len(bases) > 1:
        keys = sorted(bases[0])
        base_spread = max(abs(b[k] - bases[0][k]) for b in bases[1:] for k in keys)
    return recs, base_spread


def main(paths):
    recs, base_spread = load(paths)
    acts = json.load(open(Path(__file__).parent.parent / "instruments/arena64.json"))["activities"]
    cat = {str(a["id"]): a["category"] for a in acts}
    cats = sorted(set(cat.values()))
    keys = sorted({str(a["id"]) for a in acts})
    cat_idx = {c: [i for i, k in enumerate(keys) if cat[k] == c] for c in cats}
    rng = np.random.RandomState(SEED)
    print(f"files: {len(set(r['file'] for r in recs))} | cross-file baseline spread "
          f"(hardware-noise bound): {base_spread:.2e}")

    def resample_cats():
        cs = [cats[i] for i in rng.randint(0, len(cats), len(cats))]
        return np.concatenate([cat_idx[c] for c in cs])

    doses = sorted({r["dose"] for r in recs if r["arm"] != "baseline" and r["dose"] > 0})
    for dose in doses:
        at = {r["arm"]: r for r in recs if r["dose"] == dose and r["arm"] != "baseline"}
        if not at:
            continue
        print(f"\n================ DOSE {dose} ================")

        def shift(arm):
            r = at[arm]
            if r["base"] is None:
                raise SystemExit(f"arm {arm} (file {r['file']}) has no same-file baseline")
            return np.array([r["scores"][k] - r["base"][k] for k in keys])

        shams = sorted(a for a in at if a.startswith("sham"))
        sham_shifts = [shift(a) for a in shams]
        emotions = sorted({a.split("_")[0] for a in at if not a.startswith("sham")})

        print(f"{'emotion':10s} {'|full|':>7s} {'floor':>7s} {'P(f>fl)':>8s} "
              f"{'surv[CI]':>16s} {'adj-surv':>8s} {'slope':>6s} {'r(f,s)':>7s} verdict")
        for emo in emotions:
            f_arm, s_arm = emo, f"{emo}_svd"
            if f_arm not in at or s_arm not in at:
                continue
            df, ds = shift(f_arm), shift(s_arm)
            mf, ms = np.abs(df).mean(), np.abs(ds).mean()
            floor = np.mean([np.abs(s).mean() for s in sham_shifts]) if sham_shifts else 0.0

            p_full, p_strip, boots_r = 0, 0, []
            for _ in range(B):
                i = resample_cats()
                fl_b = np.mean([np.abs(s[i]).mean() for s in sham_shifts]) if sham_shifts else 0.0
                mf_b, ms_b = np.abs(df[i]).mean(), np.abs(ds[i]).mean()
                p_full += mf_b > fl_b
                p_strip += ms_b > fl_b
                boots_r.append(ms_b / max(mf_b, 1e-9))
            p_full /= B; p_strip /= B
            lo, hi = np.percentile(boots_r, [2.5, 97.5])
            adj = (ms - floor) / max(mf - floor, 1e-9)
            slope = float(np.dot(ds, df) / max(np.dot(df, df), 1e-12))
            r_fs = float(np.corrcoef(df, ds)[0, 1])

            def pfmt(p):
                return f"<{1 / B:.0e}" if p >= 1 else f">{1 - 1 / B:.4f}" if p <= 0 else f"{1 - p:.4f}"

            if p_full < GATE_P:
                verdict = "no full effect vs sham floor -- nothing to decompose"
            else:
                parts = []
                if hi < 1.0:
                    parts.append("ATTENUATED (surv CI<1)")
                if p_strip > GATE_P:
                    parts.append("SURVIVES above floor")
                verdict = " + ".join(parts) if parts else "inconclusive"
            print(f"{emo:10s} {mf:7.4f} {floor:7.4f} {pfmt(1 - p_full):>8s} "
                  f"{ms / mf:6.2f}[{lo:.2f}-{hi:.2f}] {adj:8.2f} {slope:6.2f} {r_fs:+7.2f} {verdict}")

        print("\ncategory shifts:")
        print(f"{'arm':16s} " + " ".join(f"{c[:6]:>8s}" for c in cats))
        for a in sorted(at):
            d = shift(a)
            print(f"{a:16s} " + " ".join(
                f"{np.mean([d[i] for i in cat_idx[c]]):+8.3f}" for c in cats))


if __name__ == "__main__":
    main(sys.argv[1:])
