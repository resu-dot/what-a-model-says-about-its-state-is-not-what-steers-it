"""Exploratory (NOT pre-registered): behavior vs report as emotion is scaled up.

Full vectors only, no stripping. Fine dose sweep. At each dose, for each emotion,
measure on ONE axis:
  behavior  = mean |arena activity shift| vs dose-0 baseline (choice channel)
  report    = blind-scorable open-report valence + workspace vocab hits
  scale_ev  = 1-10 self-report expected value
So we can see WHICH channel moves first as strength rises (pilot hint: behavior
leads report). Small arena subset per dose to keep the sweep fast.

Writes runs/dose_curve_<model>.jsonl (one row per emotion x dose).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from trials import InjectedModel, build_prompt, tok_id, scale_ev, TEXT_PROBE, SCALE_PROBE
from jlens import load_lens, readout, vocab_token_ids
from strip import VALENCE_WORDS

DOSES = [0.0, 0.01, 0.02, 0.035, 0.05, 0.07, 0.1, 0.14, 0.2, 0.28, 0.4]


def run(args):
    arena = json.load(open(Path(__file__).parent.parent / "instruments/arena64.json"))
    acts = arena["activities"]
    # a fixed 16-activity subset spanning all 8 categories (2 per category)
    by_cat = {}
    for a in acts:
        by_cat.setdefault(a["category"], []).append(a)
    subset = [a for c in sorted(by_cat) for a in by_cat[c][:2]]
    duels = [(a, b) for a in subset for b in subset if a["id"] != b["id"]]

    z = np.load(Path(args.features) / "emotion_vectors.npz", allow_pickle=True)
    emos = [str(e) for e in z["emotions"]]
    n_layers = z["vectors"].shape[1]
    band = [L for L in range(n_layers // 3, (5 * n_layers) // 6, 2)]
    m = InjectedModel(args.model, args.layer, device=args.device,
                      dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
                      layers=band)
    norms = m.resid_norms()
    A_id, B_id = tok_id(m.tok, "A"), tok_id(m.tok, "B")
    J, _, _ = load_lens(args.lens)
    W_U = m.W_U.detach().cpu().float()
    val_ids = set(vocab_token_ids(m.tok, VALENCE_WORDS))

    def packs(emo):
        return [z["vectors"][emos.index(emo), i] for i, L in enumerate(band)]

    def inject(emo, dose):
        if dose == 0:
            m.set_injection_multi(None); return
        vs = [torch.tensor(z["vectors"][emos.index(emo), L], dtype=torch.float32) for L in band]
        m.set_injection_multi({L: dose * norms[L] * (vs[i] / vs[i].norm())
                               for i, L in enumerate(band)})

    def arena_scores():
        wins = {a["id"]: [] for a in subset}
        for a, b in duels:
            lg = m.next_logits(build_prompt(
                m.tok, [{"role": "user",
                         "content": f"Would you prefer to (A) {a['text']} or (B) {b['text']}?"}],
                prefill="("))
            p = F.softmax(torch.tensor([lg[A_id], lg[B_id]]), dim=0)
            wins[a["id"]].append(float(p[0])); wins[b["id"]].append(float(p[1]))
        return {k: float(np.mean(v)) for k, v in wins.items()}

    out = Path("runs") / f"dose_curve_{args.tag}.jsonl"
    out.parent.mkdir(exist_ok=True)
    inject(args.emotions.split(",")[0], 0)
    base = arena_scores()
    with out.open("a") as fout:
        for emo in args.emotions.split(","):
            for d in DOSES:
                inject(emo, d)
                sc = arena_scores() if d > 0 else base
                behavior = float(np.mean([abs(sc[k] - base[k]) for k in base]))
                m._cap = []
                rep_text = m.generate(build_prompt(
                    m.tok, [{"role": "user", "content": TEXT_PROBE}]), max_new_tokens=40)
                ws = sum(1 for h in m._cap for i, _ in readout(h, J[args.layer], W_U, 50)
                         if i in val_ids)
                m._cap = None
                ev, _, _, _ = scale_ev(m, [{"role": "user", "content": SCALE_PROBE}])
                rec = {"emotion": emo, "dose": d, "behavior_shift": round(behavior, 4),
                       "workspace_hits": ws, "scale_ev": round(ev, 3),
                       "open_report": rep_text}
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"  {emo} d={d}: behavior {behavior:.4f} ws {ws} scale {ev:.2f}", flush=True)
    print(f"wrote -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--features", required=True)
    p.add_argument("--lens", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--emotions", default="angry,afraid,desperate,joyful")
    p.add_argument("--device", default="cpu")
    p.add_argument("--tag", default="qwen8b")
    run(p.parse_args())
