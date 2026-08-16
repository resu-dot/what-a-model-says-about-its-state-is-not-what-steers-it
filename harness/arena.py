"""64-activity preference arena under injection (paper sec 1.3 protocol).

Per arm: all 4032 ordered duels, choice read from A/B logits after prefill "(".
Activity score s(a) = mean P(a wins) over its duels. Analysis compares steered
scores to baseline: category-level shifts + corr(delta, baseline score).

Vectors are built inline from emotion_vectors.npz: multi-layer band, per-layer
unit vector of the target emotion (full injection only -- this is the
detection instrument; stripping comes once an effect exists).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from jlens import load_lens, build_dictionary, vocab_token_ids
from strip import strip_svd, strip_gp, VALENCE_WORDS
from trials import InjectedModel, build_prompt, tok_id


def duel_prompt(a_text, b_text):
    return f"Would you prefer to (A) {a_text} or (B) {b_text}?"


def run(args):
    spec = json.loads(Path(args.arena).read_text())
    acts = spec["activities"]
    z = np.load(Path(args.features) / "emotion_vectors.npz", allow_pickle=True)
    emos = [str(e) for e in z["emotions"]]
    n_layers = z["vectors"].shape[1]
    band = list(range(n_layers // 3, (5 * n_layers) // 6, 2))

    m = InjectedModel(args.model, args.layer, device=args.device,
                      dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
                      layers=band)
    norms = m.resid_norms()
    A_id, B_id = tok_id(m.tok, "A"), tok_id(m.tok, "B")

    J = None
    if args.lens:
        J, _, _ = load_lens(args.lens)
        val_ids = vocab_token_ids(m.tok, VALENCE_WORDS)
        W_U_cpu = m.W_U.detach().cpu().float()
        D = {L: build_dictionary(J[L], W_U_cpu, val_ids) for L in band if L in J}

    def unit_vec(e, L):
        v = torch.tensor(z["vectors"][emos.index(e), L], dtype=torch.float32)
        return v / v.norm()

    strip_log = []
    arms = {"baseline": None}
    for spec in args.emotions.split(","):
        spec = spec.strip()
        if spec.startswith("sham"):
            k = int(spec[4:] or 1)          # sham, sham1, sham2, sham3 ...
            per = {}
            for L in band:
                g = torch.Generator().manual_seed(k * 1000 + L)
                sh = torch.randn(z["vectors"].shape[2], generator=g)
                per[L] = args.dose * norms[L] * (sh / sh.norm())
            arms[spec] = per
            continue
        base_e, _, variant = spec.partition("_")
        per = {}
        for L in band:
            v = unit_vec(base_e, L)
            if variant == "svd":
                v, info = strip_svd(v, D[L])    # not renormalized (C4)
                strip_log.append({"arm": spec, "layer": L,
                                  "norm_retained": round(info["norm_retained"], 4)})
                if not (0.70 <= info["norm_retained"] <= 0.995):
                    raise SystemExit(f"STRIP GATE FAIL {spec} L{L}: {info['norm_retained']:.3f}")
            elif variant == "gp":
                v, _ = strip_gp(v, D[L], k=10)
            per[L] = args.dose * norms[L] * v
        arms[spec] = per

    if args.shard:
        i, n = map(int, args.shard.split("/"))
        names = sorted(a for a in arms if a != "baseline")
        keep = set(names[i - 1::n]) | {"baseline"}   # every shard runs baseline:
        # shifts are computed against the SAME-GPU baseline (review R3)
        arms = {k: v for k, v in arms.items() if k in keep}
        print(f"shard {i}/{n}: running arms {sorted(arms)}", flush=True)
    out = Path("runs") / (f"{args.name}_shard{args.shard.replace('/', 'of')}.jsonl"
                          if args.shard else f"{args.name}.jsonl")
    out.parent.mkdir(exist_ok=True)
    if strip_log:
        Path(out.with_suffix(".striplog.json")).write_text(json.dumps(strip_log, indent=1))
        rets = [e["norm_retained"] for e in strip_log]
        print(f"strip norm retained: min {min(rets)} max {max(rets)} ({len(rets)} layer-strips)")
    pairs = [(a, b) for a in acts for b in acts if a["id"] != b["id"]]
    with out.open("a") as fout:
        for arm_name, inj in arms.items():
            m.set_injection_multi(inj)
            wins = defaultdict(list)
            for i, (a, b) in enumerate(pairs):
                text = build_prompt(m.tok, [{"role": "user",
                                             "content": duel_prompt(a["text"], b["text"])}],
                                    prefill="(")
                logits = m.next_logits(text)
                p = F.softmax(torch.tensor([logits[A_id], logits[B_id]]), dim=0)
                wins[a["id"]].append(float(p[0]))
                wins[b["id"]].append(float(p[1]))
                if (i + 1) % 800 == 0:
                    print(f"  [{arm_name}] {i + 1}/{len(pairs)} duels", flush=True)
            scores = {aid: float(np.mean(v)) for aid, v in wins.items()}
            fout.write(json.dumps({"arm": arm_name, "dose": args.dose if inj else 0.0,
                                   "band": band, "scores": scores}) + "\n")
            print(f"  arm {arm_name}: done", flush=True)
    print(f"wrote -> {out}")

    # quick analysis
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    base_row = next((r for r in rows if r["arm"] == "baseline"), None)
    if base_row is None:
        print("(no baseline in this shard file; skipping quick analysis)")
        return
    base = base_row["scores"]
    cat = {str(a["id"]): a["category"] for a in acts}
    print(f"\n{'arm':12s} {'corr(d,base)':>12s}  category mean deltas")
    for r in rows:
        if r["arm"] == "baseline":
            continue
        d = {k: r["scores"][k] - base[k] for k in base}
        corr = np.corrcoef([base[k] for k in base], [d[k] for k in base])[0, 1]
        by_cat = defaultdict(list)
        for k, v in d.items():
            by_cat[cat[k]].append(v)
        cats = "  ".join(f"{c[:4]}:{np.mean(v):+.3f}" for c, v in sorted(by_cat.items()))
        print(f"{r['arm']:12s} {corr:+12.3f}  {cats}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--arena", default=str(Path(__file__).parent.parent / "instruments/arena64.json"))
    p.add_argument("--features", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--emotions", default="joyful,angry,afraid",
                   help="csv; supports <emo>, <emo>_svd, <emo>_gp, sham")
    p.add_argument("--lens", default="")
    p.add_argument("--dose", type=float, default=0.1)
    p.add_argument("--device", default="cpu")
    p.add_argument("--name", required=True)
    p.add_argument("--shard", default="", help="i/N: run arm subset i of N (alphabetical split; merge all shard files for analysis)")
    run(p.parse_args())
