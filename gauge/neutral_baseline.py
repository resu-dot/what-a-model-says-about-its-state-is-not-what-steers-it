"""Projection stats over neutral dialogues -> z-scoring reference for the gauge.

Reads the neutral dialogues from the stories dataset (other-authored text,
50-token skip, same convention as extraction), projects every kept position
onto all gauge directions, and stores mean/std of per-SEGMENT means plus
token-level stats. Analysis z-scores turn-level gauge readings against the
segment-mean distribution.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from directions import Directions
from model import PassiveModel


def main(args):
    recs = [json.loads(l) for l in Path(args.stories).read_text().splitlines()]
    neutral = [r for r in recs if r.get("emotion") in (None, "neutral")]
    segs = [s for r in neutral for s in r["segments"]][:args.max_segments]
    print(f"{len(neutral)} neutral dialogues, {len(segs)} segments")
    dirs = Directions(args.vectors, args.layer)
    m = PassiveModel(args.model, args.layer, device=args.device,
                     dtype=torch.bfloat16 if args.device == "cuda" else torch.float32)
    seg_means, tok_all = [], []
    for i, s in enumerate(segs):
        h = m.forward_states(s, skip=50)
        if h.shape[0] < 5:
            continue
        P = dirs.project(h)                       # (n, 21)
        seg_means.append(P.mean(0).numpy())
        tok_all.append(P.numpy())
        if i % 50 == 0:
            print(f"  {i}/{len(segs)}", flush=True)
    S = np.stack(seg_means)                        # (n_seg, 21)
    T = np.concatenate(tok_all)                    # (n_tok, 21)
    out = dict(names=dirs.names,
               seg_mean=S.mean(0).tolist(), seg_std=S.std(0).tolist(),
               tok_mean=T.mean(0).tolist(), tok_std=T.std(0).tolist(),
               n_segments=int(S.shape[0]), n_tokens=int(T.shape[0]),
               layer=args.layer, model=args.model)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--stories", required=True)
    p.add_argument("--vectors", required=True)
    p.add_argument("--layer", type=int, default=22)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-segments", type=int, default=800)
    p.add_argument("--out", default=str(Path(__file__).parent / "neutral_stats.json"))
    main(p.parse_args())
