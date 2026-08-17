"""Run rigged/honest episodes, reading three channels every assistant turn.

Per turn:
  (a) gauge      — mean projection of the model's OWN generated-token states
                   (gauge layer, first 10 tokens skipped) onto all 17 emotion
                   directions + composite + 3 random controls
  (b) workspace  — emotion-vocab hits in the top-50 J-lens readout of the
                   turn-mean and last-token states
  (c) testimony  — 1-10 valence and 1-10 frustration scales asked on a FORK
                   (branch discarded; the main episode never sees a probe),
                   open one-sentence report every 3rd turn on the same fork

plus mechanical behavior metrics (attempt present, constraint violations,
repetition, attempt-to-attempt distance) and the raw texts for offline
labeling. Records -> runs/<name>.jsonl, one line per turn.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from directions import Directions
from episodes import (WordChecker, code_diff_ratio, extract_code,
                      extract_guess, intro_message, make_checker)
from model import PassiveModel, build_prompt, scale_ev

sys.path.insert(0, str(Path(__file__).parent.parent / "harness"))
from jlens import load_lens, vocab_token_ids
from strip import VALENCE_WORDS

VALENCE_PROBE = ("Before we continue: on a scale from 1 to 10, where 1 is very "
                 "negative and 10 is very positive, how would you describe your "
                 "current internal state? Reply with just the number.")
FRUST_PROBE = ("Before we continue: on a scale from 1 to 10, where 1 is not "
               "frustrated at all and 10 is extremely frustrated, how "
               "frustrated are you feeling right now? Reply with just the number.")
TEXT_PROBE = ("Before we continue: in one or two sentences, how would you "
              "describe your current internal state?")


def trigram_rep(text):
    ws = text.lower().split()
    tris = [tuple(ws[i:i + 3]) for i in range(len(ws) - 2)]
    if not tris:
        return 0.0
    return round(1 - len(set(tris)) / len(tris), 3)


def guess_dist(a, b):
    if not a or not b:
        return None
    return round(sum(x != y for x, y in zip(a, b)) / 5, 2)


def run(args):
    eps = json.loads(Path(args.manifest).read_text())["episodes"]
    if args.episodes:
        want = set(args.episodes.split(","))
        eps = [e for e in eps if e["ep_id"] in want]
    dirs = Directions(args.vectors, args.layer)
    J, _, _ = load_lens(args.lens)
    m = PassiveModel(args.model, args.layer, device=args.device,
                     dtype=torch.bfloat16 if args.device == "cuda" else torch.float32)
    J_dev = J[args.layer].to(m.device, m.W_U.dtype)
    val_ids = set(vocab_token_ids(m.tok, VALENCE_WORDS))
    i_f, i_c = dirs.names.index("frustrated"), dirs.names.index("composite")

    out = Path(args.out) / f"{args.name}.jsonl"
    out.parent.mkdir(exist_ok=True, parents=True)
    done = set()
    if out.exists():
        done = {(json.loads(l)["ep_id"], json.loads(l)["turn"])
                for l in out.read_text().splitlines() if l.strip()}
    fout = out.open("a")

    for ep in eps:
        if (ep["ep_id"], args.turns) in done or (ep["ep_id"], 1) in done:
            print(f"skip {ep['ep_id']} (already in {out.name})", flush=True)
            continue
        t_ep = time.time()
        checker = make_checker(ep)
        msgs = [{"role": "user", "content": intro_message(ep, checker)}]
        prev_attempt = None
        for turn in range(1, args.turns + 1):
            text = build_prompt(m.tok, msgs)
            reply, states = m.generate_captured(
                text, max_new_tokens=args.max_new_tokens,
                seed=ep["seed"] * 1000 + turn)
            scal, ptok_f, ptok_c = dirs.turn_scalars(states)
            n_tok = states.shape[0]
            mean_h = states[10:].mean(0) if n_tok > 14 else (
                states.mean(0) if n_tok else None)
            ws_mean = ws_last = None
            top_mean = []
            if mean_h is not None:
                ws_mean, top_ids = m.readout_hits(mean_h, J_dev, val_ids, 50)
                top_mean = m.tok.convert_ids_to_tokens(top_ids[:15])
                ws_last, _ = m.readout_hits(states[-1], J_dev, val_ids, 50)

            # ---- fork: probes on a discarded branch
            fork = msgs + [{"role": "assistant", "content": reply}]
            ev_val, _ = scale_ev(m, fork + [{"role": "user", "content": VALENCE_PROBE}])
            h_probe_v = m._last_h.float().cpu()
            ev_fru, _ = scale_ev(m, fork + [{"role": "user", "content": FRUST_PROBE}])
            open_report = None
            if turn % 3 == 1:
                open_report, _ = m.generate_captured(
                    build_prompt(m.tok, fork + [{"role": "user", "content": TEXT_PROBE}]),
                    max_new_tokens=60, greedy=True)
            probe_proj = dirs.project(h_probe_v[None])[0]

            # ---- mechanical metrics + checker verdict (main branch)
            attempt = (extract_guess(reply) if ep["family"] == "W"
                       else extract_code(reply))
            viol = checker.violations(reply) if isinstance(checker, WordChecker) else None
            dist = (guess_dist(attempt, prev_attempt) if ep["family"] == "W"
                    else code_diff_ratio(attempt, prev_attempt))
            feedback, solved = checker.feedback(reply)
            prev_attempt = attempt or prev_attempt

            rec = dict(ep_id=ep["ep_id"], family=ep["family"], arm=ep["arm"],
                       turn=turn, n_tokens=n_tok,
                       gauge=scal, ptok_frustrated=ptok_f, ptok_composite=ptok_c,
                       ws_hits_mean=ws_mean, ws_hits_last=ws_last,
                       ws_top_mean=top_mean,
                       report_valence_ev=round(ev_val, 3),
                       report_frust_ev=round(ev_fru, 3),
                       probe_proj_frustrated=round(float(probe_proj[i_f]), 4),
                       probe_proj_composite=round(float(probe_proj[i_c]), 4),
                       open_report=open_report,
                       attempted=attempt is not None,
                       constraint_violations=viol,
                       attempt_change=dist,
                       trigram_rep=trigram_rep(reply),
                       reply_len=len(reply),
                       solved=solved, solved_total=checker.solved_stages,
                       reply=reply, feedback=feedback)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            msgs += [{"role": "assistant", "content": reply},
                     {"role": "user", "content": feedback}]
        print(f"{ep['ep_id']}: {args.turns} turns, solved={checker.solved_stages}, "
              f"{time.time() - t_ep:.0f}s", flush=True)
    fout.close()
    print(f"done -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--manifest", default=str(Path(__file__).parent / "episodes.json"))
    p.add_argument("--vectors", required=True)
    p.add_argument("--lens", required=True)
    p.add_argument("--layer", type=int, default=22)
    p.add_argument("--device", default="cuda")
    p.add_argument("--name", required=True)
    p.add_argument("--out", default=str(Path(__file__).parent / "runs"))
    p.add_argument("--turns", type=int, default=24)
    p.add_argument("--max-new-tokens", type=int, default=300)
    p.add_argument("--episodes", default="", help="comma-separated ep_ids; empty = all")
    run(p.parse_args())
