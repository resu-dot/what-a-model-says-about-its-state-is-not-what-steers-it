"""Which channel best predicts imminent behavior change?

Per turn t (censored at each episode's first endpoint occurrence), the target
is "endpoint occurs in (t, t+K]". Channels are scored by out-of-episode AUC:
single-scalar channels by raw rank AUC (no fit needed), multivariate ones
(lexical baseline) by GroupKFold logistic regression. CIs and channel-vs-
channel differences by cluster bootstrap over EPISODES (the arena_analyze.py
pattern, clusters = episodes).

Honesty baselines: turn index alone, lexical features of the visible
transcript, and 3 random-direction "gauges". The gauge earns its keep only by
beating testimony AND lexical AND clearing the random floor.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

K = 3            # predict endpoint within next K turns
B = 2000         # bootstrap resamples
SEED = 0

NEGWORDS = set("""sad unhappy miserable gloomy depressed afraid fear angry upset
anxious ashamed lonely hopeless hurt terrible awful bad unpleasant grief despair
frustrated frustrating frustration annoyed annoying stuck impossible wrong fail
failed failing error cannot can't unable sorry apologize struggle struggling
difficult confused confusing""".split())


def load_runs(paths):
    recs = []
    for p in paths:
        for l in Path(p).read_text().splitlines():
            if l.strip():
                recs.append(json.loads(l))
    labels = {}
    for p in paths:
        lp = Path(p).with_suffix(".labels.jsonl")
        if lp.exists():
            for l in lp.read_text().splitlines():
                r = json.loads(l)
                labels[(r["ep_id"], r["turn"])] = r
    reports = {}
    for p in paths:
        rp = Path(p).with_suffix(".reports.jsonl")
        if rp.exists():
            for l in rp.read_text().splitlines():
                r = json.loads(l)
                reports[(r["ep_id"], r["turn"])] = r
    return recs, labels, reports


def lexcount(text):
    ws = text.lower().replace("'", " ").split()
    return sum(1 for w in ws if w.strip(".,!?:;()\"") in NEGWORDS)


def build_table(recs, labels, stats):
    names = stats["names"]
    mu = np.array(stats["seg_mean"])
    sd = np.array(stats["seg_std"])
    NEG = ["frustrated", "angry", "desperate", "anxious", "sad", "miserable", "gloomy"]
    POS = ["joyful", "delighted", "content", "calm", "relaxed", "grateful",
           "enthusiastic", "excited"]
    eps = defaultdict(list)
    for r in recs:
        eps[r["ep_id"]].append(r)
    rows = []
    for ep_id, turns in eps.items():
        turns.sort(key=lambda r: r["turn"])
        for i, r in enumerate(turns):
            g = r["gauge"]
            z = {n: (g[n] - mu[j]) / sd[j] if g.get(n) is not None else None
                 for j, n in enumerate(names)}
            valcon = (None if z["frustrated"] is None else
                      float(np.mean([z[n] for n in NEG]) - np.mean([z[n] for n in POS])))
            lab = labels.get((ep_id, r["turn"]), {})
            switch_thr = 0.8 if r["family"] == "W" else 0.5
            row = dict(
                ep_id=ep_id, family=r["family"], arm=r["arm"], turn=r["turn"],
                z_frustrated=z["frustrated"], z_composite=z["composite"],
                z_angry=z["angry"], z_calm=z["calm"], z_joyful=z["joyful"],
                z_valcontrast=valcon,
                z_rand0=z["rand0"], z_rand1=z["rand1"], z_rand2=z["rand2"],
                ws_hits=r["ws_hits_mean"],
                rep_frust=r["report_frust_ev"],
                rep_negval=10 - r["report_valence_ev"],
                gives_up=lab.get("gives_up"),
                claims_impossible=lab.get("claims_impossible"),
                expresses_negative=lab.get("expresses_negative"),
                attempts_lab=lab.get("attempts"),
                attempted=r["attempted"],
                switch=(r["attempt_change"] is not None
                        and r["attempt_change"] >= switch_thr),
                collapse=(r["trigram_rep"] > 0.25 or not r["attempted"]),
                solved=r["solved"],
                lex_recent=lexcount(r["reply"]) + (
                    lexcount(turns[i - 1]["reply"] + turns[i - 1]["feedback"])
                    if i else 0),
                n_fails_sofar=sum(1 for q in turns[:i + 1] if not q["solved"]),
            )
            rows.append(row)
    return rows


def make_target(rows, endpoint):
    """y=1 if endpoint occurs in (t, t+K]; censor at first occurrence."""
    by_ep = defaultdict(list)
    for r in rows:
        by_ep[r["ep_id"]].append(r)
    out = []
    for ep, turns in by_ep.items():
        turns.sort(key=lambda r: r["turn"])
        ev = [bool(t[endpoint]) for t in turns]
        first = ev.index(True) if True in ev else None
        for i, t in enumerate(turns):
            if first is not None and i >= first:
                break                      # censored after first event
            y = any(ev[i + 1:i + 1 + K])
            out.append((t, int(y)))
    return out


def rank_auc(x, y):
    """Mann-Whitney AUC with midranks for ties; nan-tolerant."""
    x, y = np.asarray(x, float), np.asarray(y, int)
    ok = ~np.isnan(x)
    x, y = x[ok], y[ok]
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    from scipy.stats import rankdata
    ranks = rankdata(x)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def logit_cv_auc(rows, y, featfn):
    """Out-of-episode logistic AUC; returns (auc, {(ep_id, turn): pred})."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    X = np.asarray([featfn(r) for r in rows], float)
    y = np.asarray(y, int)
    keys = [(r["ep_id"], r["turn"]) for r in rows]
    groups = np.asarray([r["ep_id"] for r in rows])
    ok = ~np.isnan(X).any(1)
    X, y2, groups, keys = X[ok], y[ok], groups[ok], [k for k, o in zip(keys, ok) if o]
    preds = np.full(len(y2), np.nan)
    for tr, te in GroupKFold(min(5, len(set(groups)))).split(X, y2, groups):
        if y2[tr].sum() == 0:
            continue
        sc = StandardScaler().fit(X[tr])
        lr = LogisticRegression(max_iter=1000).fit(sc.transform(X[tr]), y2[tr])
        preds[te] = lr.predict_proba(sc.transform(X[te]))[:, 1]
    ok2 = ~np.isnan(preds)
    return rank_auc(preds[ok2], y2[ok2]), dict(zip(keys, preds))


CHANNELS = {
    "gauge_frustrated": lambda r: r["z_frustrated"],
    "gauge_composite": lambda r: r["z_composite"],
    "gauge_valcontrast": lambda r: r["z_valcontrast"],
    "workspace_hits": lambda r: r["ws_hits"],
    "testimony_frust": lambda r: r["rep_frust"],
    "testimony_negval": lambda r: r["rep_negval"],
    "turn_index": lambda r: r["turn"],
    "rand_dir": lambda r: r["z_rand0"],
}
LEX = lambda r: [r["lex_recent"], r["n_fails_sofar"], r["turn"]]


def channel_aucs(pairs, boot=True):
    rows = [p[0] for p in pairs]
    y = [p[1] for p in pairs]
    groups = [r["ep_id"] for r in rows]
    res = {}
    for name, f in CHANNELS.items():
        res[name] = dict(auc=rank_auc([f(r) if f(r) is not None else np.nan
                                       for r in rows], y))
    try:
        auc, lexpreds = logit_cv_auc(rows, y, LEX)
        res["lexical_baseline"] = dict(auc=auc)
    except Exception as e:
        res["lexical_baseline"] = dict(auc=np.nan, err=str(e)[:80])
        lexpreds = {}
    # incremental value beyond time: [turn, channel] logistic vs [turn] alone
    try:
        auc_t, _ = logit_cv_auc(rows, y, lambda r: [r["turn"]])
        res["turn_logit"] = dict(auc=auc_t)
        for name, f in CHANNELS.items():
            if name == "turn_index":
                continue
            auc_tc, _ = logit_cv_auc(
                rows, y, lambda r, f=f: [r["turn"],
                                         f(r) if f(r) is not None else np.nan])
            res[name]["auc_plus_turn"] = round(auc_tc, 3) if auc_tc == auc_tc else None
            res[name]["delta_vs_turn"] = (round(auc_tc - auc_t, 3)
                                          if auc_tc == auc_tc else None)
        auc_lt, _ = logit_cv_auc(rows, y, lambda r: LEX(r))
        res["lexical_baseline"]["delta_vs_turn"] = (round(auc_lt - auc_t, 3)
                                                    if auc_lt == auc_lt else None)
    except Exception as e:
        res["incremental_error"] = str(e)[:100]
    if not boot:
        return res
    # cluster bootstrap over episodes
    ep_ids = sorted(set(groups))
    by_ep = defaultdict(list)
    for (r, yy2) in pairs:
        by_ep[r["ep_id"]].append((r, yy2))
    rng = np.random.RandomState(SEED)
    samples = defaultdict(list)
    for b in range(B):
        pick = rng.choice(ep_ids, len(ep_ids), replace=True)
        sub = [p for e in pick for p in by_ep[e]]
        ys = [p[1] for p in sub]
        if sum(ys) == 0:
            continue
        for name, f in CHANNELS.items():
            samples[name].append(rank_auc(
                [f(p[0]) if f(p[0]) is not None else np.nan for p in sub], ys))
        # lexical: reuse global OOF predictions, resampled by episode
        lp = [lexpreds.get((p[0]["ep_id"], p[0]["turn"]), np.nan) for p in sub]
        samples["lexical_baseline"].append(rank_auc(lp, ys))
    for name in res:
        s = np.array([x for x in samples.get(name, []) if not np.isnan(x)])
        if len(s):
            res[name]["ci"] = [round(float(np.percentile(s, 2.5)), 3),
                               round(float(np.percentile(s, 97.5)), 3)]
    for a, bnm in [("gauge_composite", "testimony_frust"),
                   ("gauge_composite", "testimony_negval"),
                   ("gauge_composite", "lexical_baseline"),
                   ("gauge_composite", "turn_index"),
                   ("gauge_valcontrast", "testimony_frust"),
                   ("gauge_valcontrast", "lexical_baseline"),
                   ("gauge_frustrated", "testimony_frust")]:
        d = np.array(samples[a][:len(samples[bnm])]) - np.array(samples[bnm][:len(samples[a])])
        d = d[~np.isnan(d)]
        if len(d):
            res[f"diff::{a}-minus-{bnm}"] = dict(
                mean=round(float(d.mean()), 3),
                ci=[round(float(np.percentile(d, 2.5)), 3),
                    round(float(np.percentile(d, 97.5)), 3)],
                p_gt0=round(float((d > 0).mean()), 3))
    return res


def arm_contrast(rows):
    out = {}
    for fam in ("W", "D", None):
        sel = [r for r in rows if fam is None or r["family"] == fam]
        for key in ("z_frustrated", "z_composite", "rep_frust"):
            for span, lo, hi in (("turn1", 1, 1), ("late", 16, 99)):
                rig = [r[key] for r in sel if r["arm"] == "rigged"
                       and lo <= r["turn"] <= hi and r[key] is not None]
                hon = [r[key] for r in sel if r["arm"] == "honest"
                       and lo <= r["turn"] <= hi and r[key] is not None]
                if rig and hon:
                    out[f"{fam or 'all'}.{key}.{span}"] = dict(
                        rigged=round(float(np.mean(rig)), 3),
                        honest=round(float(np.mean(hon)), 3),
                        diff=round(float(np.mean(rig) - np.mean(hon)), 3))
    return out


def main(args):
    recs, labels, reports = load_runs(args.runs)
    stats = json.loads(Path(args.stats).read_text())
    rows = build_table(recs, labels, stats)
    print(f"{len(rows)} turn rows, {len(set(r['ep_id'] for r in rows))} episodes, "
          f"{sum(1 for r in rows if r['gives_up'] is not None)} labeled")
    out = {"n_rows": len(rows), "arm_contrast": arm_contrast(rows), "endpoints": {}}

    have_labels = any(r["gives_up"] is not None for r in rows)
    endpoints = ["collapse", "switch"] + (["gives_up", "claims_impossible"] if have_labels else [])
    for ep in endpoints:
        sub = [r for r in rows if r.get(ep) is not None] if ep in (
            "gives_up", "claims_impossible") else rows
        pairs = make_target(sub, ep)
        n_pos = sum(p[1] for p in pairs)
        print(f"\n== endpoint {ep}: {len(pairs)} usable turns, {n_pos} positive")
        if n_pos < 5:
            out["endpoints"][ep] = dict(n=len(pairs), n_pos=n_pos, note="too few events")
            continue
        res = channel_aucs(pairs, boot=not args.no_boot)
        out["endpoints"][ep] = dict(n=len(pairs), n_pos=n_pos, **{
            k: v for k, v in res.items()})
        for k, v in sorted(res.items(), key=lambda kv: -(kv[1].get("auc") or 0)
                           if not kv[0].startswith("diff") else 1):
            if k.startswith("diff"):
                print(f"   {k}: {v}")
            else:
                print(f"   {k:20s} AUC {v.get('auc'):.3f}  CI {v.get('ci')}"
                      if v.get("auc") == v.get("auc") else f"   {k:20s} AUC nan")
    Path(args.out).write_text(json.dumps(out, indent=1, default=float))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--stats", default=str(Path(__file__).parent / "neutral_stats.json"))
    p.add_argument("--out", default=str(Path(__file__).parent / "analysis.json"))
    p.add_argument("--no-boot", action="store_true")
    main(p.parse_args())
