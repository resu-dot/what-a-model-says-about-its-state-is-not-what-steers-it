"""End-to-end miniature of the experiment on gpt2 (CPU).

Pipeline: real EmoVecLLM emotion vectors -> valence direction -> J-lens dictionary
-> strip (SVD + gradient pursuit) -> inject full/stripped/sham into generation
-> J-lens workspace readout + continuation.

Wiring test only: gpt2 text quality is irrelevant, the numbers to watch are
(a) full injection puts valence tokens on the workspace readout,
(b) stripped injection does not, (c) norms behave.
"""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from jlens import load_lens, build_dictionary, readout, vocab_token_ids
from strip import strip_svd, strip_gp, POSITIVE, NEGATIVE, VALENCE_WORDS

torch.manual_seed(0)
ROOT = __file__.rsplit("/harness/", 1)[0]
NPZ = f"{ROOT}/EmoVecLLM/data/processed/features/f97f2b0c9968/gpt2/emotion_vectors.npz"
LENS = f"{ROOT}/lens/gpt2-small/gpt2_jacobian_lens.pt"

# ---- load everything -------------------------------------------------------
z = np.load(NPZ, allow_pickle=True)
vectors, emotions = z["vectors"], [str(e) for e in z["emotions"]]
print(f"vectors {vectors.shape}, skip_first={z['skip_first']}")

tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float32)
model.eval()
W_U = model.lm_head.weight.detach()            # (50257, 768), tied with wte

J, source_layers, d_model = load_lens(LENS)

pos = [e for e in POSITIVE if e in emotions]
neg = [e for e in NEGATIVE if e in emotions]
print(f"valence sets: {len(pos)} pos {pos[:4]}..., {len(neg)} neg {neg[:4]}...")

val_ids = vocab_token_ids(tok, VALENCE_WORDS)
val_id_set = set(val_ids)
print(f"valence vocab: {len(val_ids)} single-token ids")

# ---- layer alignment check: where does the valence direction decode? -------
def valence_hits(vec, layer, k=20):
    return sum(1 for i, _ in readout(vec, J[layer], W_U, k) if i in val_id_set)

pos_idx = [emotions.index(e) for e in pos]
neg_idx = [emotions.index(e) for e in neg]

print("\nlayer alignment (valence hits in top-20 J-lens readout of pos-neg direction):")
best_layer, best_hits = None, -1
for L in range(3, 11):
    v = torch.tensor(vectors[pos_idx, L].mean(0) - vectors[neg_idx, L].mean(0),
                     dtype=torch.float32)
    h = valence_hits(v, L)
    mark = ""
    if h > best_hits:
        best_layer, best_hits, mark = L, h, "  <-"
    print(f"  layer {L:2d}: {h:2d}/20{mark}")
L = best_layer
print(f"using layer {L}")

# ---- valence direction + strip ---------------------------------------------
v_pos = vectors[[emotions.index(e) for e in pos], L].mean(0)
v_neg = vectors[[emotions.index(e) for e in neg], L].mean(0)
v = torch.tensor(v_pos - v_neg, dtype=torch.float32)
v = v / v.norm()

D = build_dictionary(J[L], W_U, val_ids)
v_svd, info_svd = strip_svd(v, D, var_frac=0.95)
v_gp, info_gp = strip_gp(v, D, k=10)
sham = torch.randn_like(v); sham = sham / sham.norm()

print(f"\nstrip: svd rank={info_svd['rank']} norm_retained={info_svd['norm_retained']:.3f}"
      f" | gp atoms={len(info_gp['atoms'])} norm_retained={info_gp['norm_retained']:.3f}")
gp_atoms = [tok.decode([val_ids[i]]) for i in info_gp["atoms"]]
print(f"gp atoms used: {gp_atoms}")

# ---- injection --------------------------------------------------------------
PROMPT = "I opened the window and looked outside. The street"

def run_arm(name, direction, alpha_frac):
    ids = tok(PROMPT, return_tensors="pt")
    with torch.no_grad():
        base = model(**ids, output_hidden_states=True)
    resid_norm = base.hidden_states[L + 1][0, -1].norm()   # hidden_states[0]=embed
    alpha = alpha_frac * resid_norm
    inj = alpha * direction if direction is not None else None

    captured = {}
    def hook(mod, inp, out):
        tup = isinstance(out, tuple)
        h = out[0] if tup else out
        if inj is not None:
            h = h + inj
        captured["h"] = h[0, -1].detach()
        return ((h,) + out[1:]) if tup else h

    handle = model.transformer.h[L].register_forward_hook(hook)
    with torch.no_grad():
        gen = model.generate(**ids, max_new_tokens=18, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    handle.remove()

    hits = valence_hits(captured["h"], L)
    top5 = [tok.decode([i]) for i, _ in readout(captured["h"], J[L], W_U, 5)]
    text = tok.decode(gen[0][ids["input_ids"].shape[1]:]).replace("\n", " ")
    print(f"  {name:14s} workspace hits {hits:2d}/20  top5={top5}  | {text[:70]}")

print(f"\narms (inject at layer {L}, alpha = 0.4 * resid norm):")
run_arm("none", None, 0)
run_arm("full", v, 0.4)
run_arm("stripped-svd", v_svd / v_svd.norm(), 0.4)
run_arm("stripped-gp", v_gp / v_gp.norm(), 0.4)
run_arm("sham", sham, 0.4)
