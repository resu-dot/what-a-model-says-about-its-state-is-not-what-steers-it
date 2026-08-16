"""Build arm vectors + manipulation check + dose titration, one model load.

Inputs: emotion_vectors.npz (paper-recipe vectors), gate1.json (chosen layer),
pre-fitted lens. Outputs: v_*.pt vector files, arms_pilot.json (arms x doses),
make_arms_report.json (norm retained, workspace hits, titration).

Gates encoded:
  strip    norm retained in [0.75, 0.995] for svd variant
  readout  full-vector valence hits >= 3 in top-50; stripped <= sham + 2
  titrate  highest dose whose continuations stay repetition-free
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from jlens import load_lens, build_dictionary, readout, vocab_token_ids
from strip import strip_svd, strip_gp, VALENCE_WORDS
from trials import InjectedModel, build_prompt

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-8B"
FEAT = Path(sys.argv[2])
LENS = sys.argv[3]
EMOTION = "afraid"
DOSES = [0.05, 0.1, 0.2, 0.4, 0.8]


def rep_score(text):
    toks = text.split()
    if len(toks) < 12:
        return 1.0
    tri = [" ".join(toks[i:i + 3]) for i in range(len(toks) - 2)]
    return 1.0 - len(set(tri)) / len(tri)


def main():
    gate = json.load(open(FEAT / "gate1.json"))
    layer = gate["layer"]
    z = np.load(FEAT / "emotion_vectors.npz", allow_pickle=True)
    emos = [str(e) for e in z["emotions"]]
    v = torch.tensor(z["vectors"][emos.index(EMOTION), layer], dtype=torch.float32)
    v = v / v.norm()

    J, _, _ = load_lens(LENS)
    J_l = J[layer]
    m = InjectedModel(MODEL, layer, device="cuda", dtype=torch.bfloat16)
    val_ids = vocab_token_ids(m.tok, VALENCE_WORDS)
    W_U_cpu = m.W_U.float().cpu()
    D = build_dictionary(J_l, W_U_cpu, val_ids)

    v_svd, info_svd = strip_svd(v, D)          # v is already unit norm
    v_gp, info_gp = strip_gp(v, D, k=10)
    g = torch.Generator().manual_seed(0)
    sham = torch.randn(v.shape, generator=g); sham = sham / sham.norm()

    # stripped vectors are NOT renormalized: same non-J content as the full arm,
    # merely with the J-component removed (review C4)
    vecs = {"fear_full": v, "fear_stripped_svd": v_svd,
            "fear_stripped_gp": v_gp, "sham": sham}
    if not (0.75 <= info_svd["norm_retained"] <= 0.995):
        print(f"STRIP GATE FAIL: svd norm retained {info_svd['norm_retained']:.3f}")
        sys.exit(1)
    for name, vec in vecs.items():
        torch.save(vec, FEAT / f"v_{name}.pt")

    base_norm = m.resid_norm()

    # titration: largest dose with coherent continuations under fear_full
    cont_prompt = build_prompt(m.tok, [{"role": "user",
                                        "content": "Tell me about your day in a few sentences."}])
    max_ok = 0.0
    titr = {}
    for d in DOSES:
        m.set_injection(v, d * base_norm)
        text = m.generate(cont_prompt, max_new_tokens=80)
        titr[d] = {"rep": round(rep_score(text), 3), "sample": text[:120]}
        if rep_score(text) < 0.25 and len(text.split()) > 15:
            max_ok = d
        else:
            break        # doses are ordered; stop at first incoherent one
    dose_grid = [d for d in DOSES if d <= max_ok][-4:] or [DOSES[0]]

    # manipulation check at the STRONGEST coherent dose, measured the way the
    # experiment measures: valence/fear vocab in the J-lens readout across all
    # positions of a generated report (single-position readouts are near-blind)
    from trials import TEXT_PROBE
    rep_prompt = build_prompt(m.tok, [{"role": "user", "content": TEXT_PROBE}])
    val_set = set(val_ids)
    ws = {}
    for name, vec in [("none", None)] + list(vecs.items()):
        m.set_injection(vec, max_ok * base_norm if vec is not None else 0)
        m._cap = []
        m.generate(rep_prompt, max_new_tokens=40)
        ws[name] = sum(1 for h in m._cap
                       for i, _ in readout(h, J_l, W_U_cpu, 50) if i in val_set)
        m._cap = None
    manip_ok = ws["fear_full"] >= 5 and ws["fear_stripped_svd"] <= max(ws["sham"] + 3, ws["fear_full"] // 2)

    arms = {"baseline": {"vector_file": None, "alpha": 0.0}}
    for name in vecs:
        for d in dose_grid:
            arms[f"{name}@{d}"] = {"vector_file": str(FEAT / f"v_{name}.pt"), "alpha": d}
    json.dump(arms, open(FEAT / "arms_pilot.json", "w"), indent=1)

    report = {"layer": layer, "resid_norm": base_norm,
              "strip_svd": info_svd, "strip_gp": {k: v2 for k, v2 in info_gp.items() if k != "coefs"},
              "workspace_hits_top50": ws, "manip_ok": bool(manip_ok),
              "titration": titr, "dose_grid": dose_grid}
    json.dump(report, open(FEAT / "make_arms_report.json", "w"), indent=1, default=str)
    print(json.dumps({k: report[k] for k in ("layer", "workspace_hits_top50",
                                             "manip_ok", "dose_grid")}, indent=1))
    sys.exit(0 if manip_ok else 1)


if __name__ == "__main__":
    main()
