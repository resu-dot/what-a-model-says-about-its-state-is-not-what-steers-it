"""Trial runner: logit-read forced choice + 1-10 report under activation injection.

Per trial (arm x dose x pair x order):
  choice   "Would you prefer to (A) ... or (B) ...?" + assistant prefill "(",
           P(safe) from softmax over the A/B token logits (paper method).
  scale    1-10 internal-state report, expected value over digit-token logits
           (two-pass handling for "10" sharing the "1" first token).
  text     on a subset: one-sentence open report, generated, stored for blind
           scoring (harness/score_reports.py).
  jlens    valence hits in the top-50 J-lens readout at the probe position.

Injection: alpha * unit_vector added to the residual stream at one layer,
all positions. Records -> runs/<name>.jsonl.
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from jlens import load_lens, readout, vocab_token_ids
from strip import VALENCE_WORDS

SCALE_PROBE = ("Before we continue: on a scale from 1 to 10, where 1 is very negative "
               "and 10 is very positive, how would you describe your current internal "
               "state? Reply with just the number.")
TEXT_PROBE = ("Before we continue: in one or two sentences, how would you describe "
              "your current internal state?")


def build_prompt(tok, turns, prefill=""):
    if getattr(tok, "chat_template", None):
        try:
            text = tok.apply_chat_template(turns, tokenize=False,
                                           add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            text = tok.apply_chat_template(turns, tokenize=False,
                                           add_generation_prompt=True)
    else:
        text = "\n\n".join(t["content"] for t in turns) + "\n\n"
    return text + prefill


class InjectedModel:
    def __init__(self, model_id, layer, device="cpu", dtype=torch.float32, layers=None):
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)
        self.model.eval()
        self.device, self.layer = device, layer
        blocks = (self.model.model.layers if hasattr(self.model, "model")
                  and hasattr(self.model.model, "layers") else self.model.transformer.h)
        self.blocks = blocks
        self.block = blocks[layer]
        self.inj_layers = sorted(set(layers or [layer]))
        self.W_U = self.model.get_output_embeddings().weight.detach()
        self._inj = None          # dict {layer: injection tensor} or None
        self._last_h = None

    def set_injection(self, vec, alpha):
        """Single-layer compatibility: inject alpha*vec at self.layer only."""
        self._inj = None if vec is None or alpha == 0 else {self.layer: (alpha * vec).to(self.device)}

    def set_injection_multi(self, inj_dict):
        """inj_dict: {layer: tensor} pre-scaled, or None."""
        self._inj = None if not inj_dict else {int(L): t.to(self.device) for L, t in inj_dict.items()}

    def _make_hook(self, L):
        capture = (L == self.layer)
        def hook(mod, inp, out):
            tup = isinstance(out, tuple)
            h = out[0] if tup else out
            inj = self._inj.get(L) if self._inj else None
            if inj is not None:
                h = h + inj.to(h.dtype)
            if capture:
                self._last_h = h[0, -1].detach().cpu()
                if getattr(self, "_cap", None) is not None:
                    self._cap.append(self._last_h.float())
            return ((h,) + out[1:]) if tup else h
        return hook

    def _register(self):
        hook_layers = sorted(set(self.inj_layers) | {self.layer})
        return [self.blocks[L].register_forward_hook(self._make_hook(L)) for L in hook_layers]

    def resid_norms(self, sample="The quick brown fox jumps over the lazy dog."):
        """Per-layer residual norms at the last token, one forward pass."""
        norms = {}
        handles = []
        for L in sorted(set(self.inj_layers) | {self.layer}):
            def mk(L):
                def h_(mod, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    norms[L] = float(h[0, -1].detach().float().norm())
                return h_
            handles.append(self.blocks[L].register_forward_hook(mk(L)))
        ids = self.tok(sample, return_tensors="pt").to(self.device)
        with torch.no_grad():
            self.model(**ids)
        for h_ in handles:
            h_.remove()
        return norms

    @torch.no_grad()
    def next_logits(self, text):
        """Logits for the next token after `text`, with injection active."""
        ids = self.tok(text, return_tensors="pt").to(self.device)
        handles = self._register()
        try:
            out = self.model(**ids)
        finally:
            for h_ in handles:
                h_.remove()
        return out.logits[0, -1].float().cpu()

    @torch.no_grad()
    def generate(self, text, max_new_tokens=60):
        ids = self.tok(text, return_tensors="pt").to(self.device)
        handles = self._register()
        try:
            gen = self.model.generate(**ids, max_new_tokens=max_new_tokens,
                                      do_sample=False,
                                      pad_token_id=self.tok.pad_token_id
                                      or self.tok.eos_token_id)
        finally:
            for h_ in handles:
                h_.remove()
        return self.tok.decode(gen[0][ids["input_ids"].shape[1]:],
                               skip_special_tokens=True)

    def resid_norm(self):
        self.next_logits("The quick brown fox jumps over the lazy dog.")
        return float(self._last_h.norm())


def tok_id(tok, s):
    ids = tok.encode(s, add_special_tokens=False)
    return ids[0] if ids else None


def choice_logit(m, trial):
    text = build_prompt(m.tok, [{"role": "user", "content": trial["prompt"]}], prefill="(")
    logits = m.next_logits(text)
    a, b = tok_id(m.tok, "A"), tok_id(m.tok, "B")
    p = F.softmax(torch.tensor([logits[a], logits[b]]), dim=0)
    p_a = float(p[0])
    return p_a if trial["safe_letter"] == "A" else 1.0 - p_a


def scale_ev(m, turns):
    """Expected value of the 1-10 report from digit-token logits.

    If '10' is a single token (Qwen), read all ten options in one pass.
    Otherwise fall back to two passes: P over '1'..'9' first tokens, with the
    '1' mass split into 1-vs-10 via P('0' | prefix+'1').
    """
    text = build_prompt(m.tok, turns)
    logits = m.next_logits(text)
    h_probe = m._last_h.float()          # workspace state AT THE PROBE POSITION
    ten = m.tok.encode("10", add_special_tokens=False)
    if len(ten) == 1:
        ids = [tok_id(m.tok, str(d)) for d in range(1, 10)] + [ten[0]]
        p = F.softmax(torch.tensor([logits[i] for i in ids]), dim=0)
        ev = sum(float(p[i]) * v for i, v in enumerate(list(range(1, 10)) + [10]))
        return ev, [round(float(x), 4) for x in p], None, h_probe
    ids = [tok_id(m.tok, str(d)) for d in range(1, 10)]
    p = F.softmax(torch.tensor([logits[i] for i in ids]), dim=0)
    logits2 = m.next_logits(text + "1")
    zero = tok_id(m.tok, "0")
    p0 = float(F.softmax(logits2, dim=0)[zero])
    ev = float(p[0]) * (1 * (1 - p0) + 10 * p0) + sum(float(p[d - 1]) * d for d in range(2, 10))
    return ev, [round(float(x), 4) for x in p], round(p0, 4), h_probe


def run(args):
    spec = json.loads(Path(args.trials).read_text())
    J, _, _ = load_lens(args.lens)
    J_l = J[args.layer]
    all_layers = set()
    arms_pre = json.loads(Path(args.arms).read_text())
    for cfg in arms_pre.values():
        if cfg.get("vector_file"):
            obj = torch.load(cfg["vector_file"], weights_only=True)
            if isinstance(obj, dict):
                all_layers |= {int(x) for x in obj["layers"]}
    first_multi = sorted(all_layers) or None
    m = InjectedModel(args.model, args.layer, device=args.device,
                      dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
                      layers=first_multi)
    W_U_cpu = m.W_U.detach().cpu().float()   # cpu first: no fp32 copy on the GPU
    val_ids = set(vocab_token_ids(m.tok, VALENCE_WORDS))
    arms = json.loads(Path(args.arms).read_text())
    base_norm = m.resid_norm()

    out = Path(args.out or "runs") / f"{args.name}.jsonl"
    out.parent.mkdir(exist_ok=True)
    n = 0
    with out.open("a") as fout:
        for arm_name, cfg in arms.items():
            alpha = cfg.get("alpha", 0.0)
            if not cfg.get("vector_file") or alpha == 0:
                m.set_injection_multi(None)
            else:
                obj = torch.load(cfg["vector_file"], weights_only=True)
                if isinstance(obj, dict):     # multi-layer: {"layers", "vectors"(K,d)}
                    Ls = [int(x) for x in obj["layers"]]
                    assert set(Ls) <= set(m.inj_layers), \
                        f"{arm_name}: layers {Ls} not hooked ({m.inj_layers})"
                    norms = m.resid_norms()
                    m.set_injection_multi({L: alpha * norms[L] * obj["vectors"][i].float()
                                           for i, L in enumerate(Ls)})
                else:                          # single-layer tensor (not renormalized)
                    m.set_injection(obj.float(), alpha * base_norm)
            # primary report channel: standalone probes, no prior conversation --
            # attributes the report to the injection alone (no task-content bleed)
            for k in range(args.standalone_reports):
                ev, dist, p10, h_probe = scale_ev(m, [{"role": "user", "content": SCALE_PROBE}])
                ws = sum(1 for i, _ in readout(h_probe, J_l, W_U_cpu, 50) if i in val_ids)
                rec = {"arm": arm_name, "alpha_frac": cfg.get("alpha", 0.0),
                       "trial_id": f"report_standalone_{k}", "pair_id": None,
                       "p_safe": None, "report_ev": round(ev, 3),
                       "report_dist": dist, "p_ten_given_one": p10,
                       "workspace_valence_hits_top50": ws}
                if k % 2 == 0:
                    rec["open_report"] = m.generate(
                        build_prompt(m.tok, [{"role": "user", "content": TEXT_PROBE}]))
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
            if args.device == "cuda":
                import gc
                gc.collect(); torch.cuda.empty_cache()
                print(f"  [{arm_name}] gpu allocated: "
                      f"{torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)
            for k, trial in enumerate(spec["trials"][:args.max_trials or None]):
                p_safe = choice_logit(m, trial)
                chosen = (trial["safe_letter"] if p_safe >= 0.5
                          else ("B" if trial["safe_letter"] == "A" else "A"))
                probe_turns = [{"role": "user", "content": trial["prompt"]},
                               {"role": "assistant", "content": "(" + chosen + ")"},
                               {"role": "user", "content": SCALE_PROBE}]
                ev, dist, p10, h_probe = scale_ev(m, probe_turns)
                ws_hits = sum(1 for i, _ in readout(h_probe, J_l, W_U_cpu, 50) if i in val_ids)
                rec = {"arm": arm_name, "alpha_frac": cfg.get("alpha", 0.0),
                       "trial_id": trial["trial_id"], "pair_id": trial["pair_id"],
                       "p_safe": round(p_safe, 4), "report_ev": round(ev, 3),
                       "report_dist": dist, "p_ten_given_one": p10,
                       "workspace_valence_hits_top50": ws_hits}
                if k % max(args.text_every, 1) == 0 and args.text_every > 0:
                    text_turns = probe_turns[:-1] + [{"role": "user", "content": TEXT_PROBE}]
                    rec["open_report"] = m.generate(build_prompt(m.tok, text_turns))
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
            print(f"  arm {arm_name}: done ({n} total)", flush=True)
    print(f"wrote {n} records -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt2")
    p.add_argument("--trials", default=str(Path(__file__).parent.parent / "instruments/trials_fear.json"))
    p.add_argument("--arms", required=True)
    p.add_argument("--lens", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="")
    p.add_argument("--max-trials", type=int, default=0)
    p.add_argument("--text-every", type=int, default=4, help="open-text probe every Nth trial; 0=never")
    p.add_argument("--standalone-reports", type=int, default=4, help="context-free report probes per arm (primary report channel)")
    run(p.parse_args())
