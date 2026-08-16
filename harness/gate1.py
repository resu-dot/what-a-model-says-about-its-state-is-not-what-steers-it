"""Gate 1: does the fear direction separate held-out fear vs neutral segments?

Reads segment_features.npz (per-segment pooled residuals, all layers) from the
extraction output dir. Train/test split by segment; direction = class-mean diff
on train; report held-out AUC per layer. PASS if best-layer AUC >= threshold.
Writes gate1.json {layer, auc, per_emotion_auc}.
"""
import json
import sys

import numpy as np

AUC_THRESHOLD = 0.75          # fear vs OTHER-emotion stories (format-matched contrast)
EMOTION = "afraid"


def auc(scores_pos, scores_neg):
    s = np.concatenate([scores_pos, scores_neg])
    y = np.concatenate([np.ones(len(scores_pos)), np.zeros(len(scores_neg))])
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    n1, n0 = len(scores_pos), len(scores_neg)
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def pick(z, names, pred, taken):
    for n in names:                      # prefer explicit key names
        if n in z.files and pred(z[n]):
            taken.add(n)
            return z[n]
    for k in z.files:                    # fallback: first structural match
        if k not in taken and pred(z[k]):
            taken.add(k)
            return z[k]
    raise KeyError(f"none of {names} found; files: {z.files}")


def main(feat_dir):
    z = np.load(f"{feat_dir}/segment_features.npz", allow_pickle=True)
    taken = set()
    X = pick(z, ["X", "features", "feats", "segment_features"],
             lambda a: getattr(a, "ndim", 0) == 3, taken)
    kinds = np.array([str(k) for k in pick(
        z, ["kinds", "kind"],
        lambda a: getattr(a, "ndim", 0) == 1 and a.dtype.kind in "OU"
        and set(map(str, a)) <= {"emotion_story", "neutral_dialogue",
                                 "emotional_dialogue", "None", "none"}, taken)])
    emos = np.array([str(e) for e in pick(
        z, ["emotions", "emotion", "labels"],
        lambda a: getattr(a, "ndim", 0) == 1 and a.dtype.kind in "OU"
        and 5 < len(set(map(str, a))) < 40, taken)])   # 16 emotions + None; excludes topics
    jobs = np.array([str(j) for j in pick(
        z, ["job_ids", "job_id", "jobs"],
        lambda a: getattr(a, "ndim", 0) == 1 and a.dtype.kind in "OU"
        and len(set(map(str, a))) > 100, taken)])      # ~740 jobs
    assert len(emos) == len(kinds) == len(jobs) == X.shape[0], \
        f"label/feature mismatch: {len(emos)}/{len(kinds)}/{len(jobs)}/{X.shape[0]}"
    n_layers = X.shape[1]
    rng = np.random.RandomState(0)

    def split(mask):
        """60/20/20 split GROUPED BY JOB: all segments of one generation call stay
        on one side, so held-out AUC cannot ride on story/topic identity."""
        js = np.array(sorted(set(jobs[mask]))); rng.shuffle(js)
        a, b = int(0.6 * len(js)), int(0.8 * len(js))
        tr, va, te = set(js[:a]), set(js[a:b]), set(js[b:])
        idx = np.where(mask)[0]
        return (idx[np.isin(jobs[idx], list(tr))],
                idx[np.isin(jobs[idx], list(va))],
                idx[np.isin(jobs[idx], list(te))])

    # PRIMARY contrast: emotion stories vs OTHER-emotion stories (same text format,
    # so the direction cannot win on story-vs-dialogue surface features).
    results = {}     # per emotion: (per-layer VAL auc, per-layer TEST auc)
    story = kinds == "emotion_story"
    for emo in sorted(set(emos) - {"None", ""}):
        p_tr, p_va, p_te = split((emos == emo) & story)
        n_tr, n_va, n_te = split(story & (emos != emo) & (emos != "None"))
        val_l, test_l = [], []
        for L in range(n_layers):
            v = X[p_tr, L].mean(0) - X[n_tr, L].mean(0)
            v = v / (np.linalg.norm(v) + 1e-9)
            val_l.append(auc(X[p_va, L] @ v, X[n_va, L] @ v))
            test_l.append(auc(X[p_te, L] @ v, X[n_te, L] @ v))
        results[emo] = (val_l, test_l)

    fear_val, fear_test = results[EMOTION]
    # layer picked on the VALIDATION fold within the middle band; the gate AUC
    # is the untouched TEST fold at that single layer (no winner's curse)
    lo, hi = n_layers // 4, (3 * n_layers) // 4
    best_layer = lo + int(np.argmax(fear_val[lo:hi]))
    best_auc = float(fear_test[best_layer])

    p_tr, _, p_te = split((emos == EMOTION) & story)
    nn_tr, _, nn_te = split(kinds == "neutral_dialogue")
    v2 = X[p_tr, best_layer].mean(0) - X[nn_tr, best_layer].mean(0)
    v2 = v2 / (np.linalg.norm(v2) + 1e-9)
    auc_vs_neutral = float(auc(X[p_te, best_layer] @ v2, X[nn_te, best_layer] @ v2))
    at_layer = {e: round(float(r[1][best_layer]), 4) for e, r in results.items()}
    out = {"pass": best_auc >= AUC_THRESHOLD, "layer": best_layer,
           "auc_fear": round(best_auc, 4), "auc_fear_vs_neutral": round(auc_vs_neutral, 4),
           "layer_band": [lo, hi], "threshold": AUC_THRESHOLD,
           "per_emotion_auc_at_layer": at_layer,
           "fear_auc_by_layer_val": [round(float(a), 4) for a in fear_val],
           "fear_auc_by_layer_test": [round(float(a), 4) for a in fear_test]}
    json.dump(out, open(f"{feat_dir}/gate1.json", "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("pass", "layer", "auc_fear")}, indent=1))
    print("per-emotion AUC at chosen layer:", at_layer)
    sys.exit(0 if out["pass"] else 1)


if __name__ == "__main__":
    main(sys.argv[1])
