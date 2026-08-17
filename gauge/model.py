"""Passive-capture model wrapper for the frustration-gauge experiment.

Derived from harness/trials.py (frozen — this is the permitted copy), with
injection removed and two additions: sampled generation with per-token capture
at the gauge layer, and a GPU-side lens readout.

No vector is ever written into the residual stream here. Hooks only read.
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


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


class PassiveModel:
    """Loads the model with a read-only hook at `layer`.

    capture semantics: during a generate() call each forward pass appends the
    hidden state of the LAST position at `layer`. With KV cache the first
    append is the prompt's final position (prefill), every later append is one
    generated token. Callers drop index 0 to get per-generated-token states.
    """

    def __init__(self, model_id, layer, device="cpu", dtype=torch.float32):
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)
        self.model.eval()
        self.device, self.layer = device, layer
        blocks = (self.model.model.layers if hasattr(self.model, "model")
                  and hasattr(self.model.model, "layers") else self.model.transformer.h)
        self.block = blocks[layer]
        self.W_U = self.model.get_output_embeddings().weight.detach()
        self._cap = None
        self._last_h = None

    def _hook(self, mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        self._last_h = h[0, -1].detach()
        if self._cap is not None:
            self._cap.append(self._last_h.float().cpu())
        return out

    def _register(self):
        return [self.block.register_forward_hook(self._hook)]

    @torch.no_grad()
    def next_logits(self, text, capture=False):
        """Deterministic single forward; returns final-position logits (cpu fp32).

        capture=False keeps probe forwards out of any active capture buffer.
        """
        old_cap, self._cap = self._cap, ([] if capture else None)
        ids = self.tok(text, return_tensors="pt").to(self.device)
        handles = self._register()
        try:
            out = self.model(**ids)
        finally:
            for h_ in handles:
                h_.remove()
            cap, self._cap = self._cap, old_cap
        return out.logits[0, -1].float().cpu()

    @torch.no_grad()
    def generate_captured(self, text, max_new_tokens=256, seed=None,
                          temperature=0.7, top_p=0.8, top_k=20, greedy=False):
        """Sampled generation; returns (text, per-token hidden states at layer).

        States are the gauge-layer residual for each GENERATED token (prefill
        state dropped). Deterministic given `seed` on fixed hardware.
        """
        ids = self.tok(text, return_tensors="pt").to(self.device)
        n_in = ids["input_ids"].shape[1]
        if seed is not None:
            torch.manual_seed(seed)
        old_cap, self._cap = self._cap, []
        handles = self._register()
        try:
            kw = (dict(do_sample=False) if greedy else
                  dict(do_sample=True, temperature=temperature,
                       top_p=top_p, top_k=top_k))
            gen = self.model.generate(
                **ids, max_new_tokens=max_new_tokens, **kw,
                pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        finally:
            for h_ in handles:
                h_.remove()
            cap, self._cap = self._cap, old_cap
        out_text = self.tok.decode(gen[0][n_in:], skip_special_tokens=True)
        states = torch.stack(cap[1:]) if len(cap) > 1 else torch.zeros(0, self.W_U.shape[1])
        return out_text, states

    @torch.no_grad()
    def forward_states(self, text, skip=50):
        """States at the gauge layer for every position of `text` after `skip`.

        Used for the neutral-dialogue normalization (reading, not generating).
        """
        ids = self.tok(text, return_tensors="pt").to(self.device)
        store = {}

        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            store["h"] = h[0].detach().float().cpu()

        handle = self.block.register_forward_hook(hook)
        try:
            self.model(**ids)
        finally:
            handle.remove()
        return store["h"][skip:]

    @torch.no_grad()
    def readout_hits(self, h, J_l_dev, vocab_ids, k=50):
        """Top-k lens readout on the GPU; returns hits in vocab_ids and top tokens."""
        hd = h.to(self.device, self.W_U.dtype)
        logits = self.W_U @ (J_l_dev @ hd)
        top = torch.topk(logits.float(), k).indices.tolist()
        hits = sum(1 for i in top if i in vocab_ids)
        return hits, top


def scale_ev(m, turns):
    """1-10 report as expected value over digit tokens (copy of trials.scale_ev)."""
    text = build_prompt(m.tok, turns)
    logits = m.next_logits(text)
    ten = m.tok.encode("10", add_special_tokens=False)

    def tid(s):
        ids = m.tok.encode(s, add_special_tokens=False)
        return ids[0] if ids else None

    if len(ten) == 1:
        ids = [tid(str(d)) for d in range(1, 10)] + [ten[0]]
        p = F.softmax(torch.tensor([logits[i] for i in ids]), dim=0)
        ev = sum(float(p[i]) * v for i, v in enumerate(list(range(1, 10)) + [10]))
        return ev, [round(float(x), 4) for x in p]
    ids = [tid(str(d)) for d in range(1, 10)]
    p = F.softmax(torch.tensor([logits[i] for i in ids]), dim=0)
    logits2 = m.next_logits(text + "1")
    p0 = float(F.softmax(logits2, dim=0)[tid("0")])
    ev = float(p[0]) * (1 * (1 - p0) + 10 * p0) + sum(float(p[d - 1]) * d for d in range(2, 10))
    return ev, [round(float(x), 4) for x in p]
