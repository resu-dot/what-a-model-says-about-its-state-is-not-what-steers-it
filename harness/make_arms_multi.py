"""Multi-layer arms (paper protocol): at each middle-band layer, that layer's own
fear vector; stripped variants use that layer's own lens dictionary. Readout and
manipulation check stay at the validated lens layer (gate1.json).

Outputs v_multi_*.pt as {"layers": [...], "vectors": (K, d)} + arms_multi.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from jlens import load_lens, build_dictionary, readout, vocab_token_ids
from strip import strip_svd, strip_gp, VALENCE_WORDS
from trials import InjectedModel, build_prompt, TEXT_PROBE

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-8B"
FEAT = Path(sys.argv[2])
LENS = sys.argv[3]
EMOTIONS = (sys.argv[4].split(",") if len(sys.argv) > 4
            else ["afraid"])              # e.g. angry,afraid,desperate,joyful
N_SHAM = 3
DOSES = [0.02, 0.05, 0.1, 0.2, 0.4]      # per-layer fractions; effects accumulate


def rep_score(text):
    toks = text.split()
    if len(toks) < 12:
        return 1.0
    tri = [" ".join(toks[i:i + 3]) for i in range(len(toks) - 2)]
    return 1.0 - len(set(tri)) / len(tri)


def main():
    gate = json.load(open(FEAT / "gate1.json"))
    lens_layer = gate["layer"]
    z = np.load(FEAT / "emotion_vectors.npz", allow_pickle=True)
    emos = [str(e) for e in z["emotions"]]
    J, source_layers, _ = load_lens(LENS)
    n_layers = z["vectors"].shape[1]
    band = [L for L in range(n_layers // 3, (5 * n_layers) // 6, 2) if L in J]
    print(f"injection band: {band} | lens/readout layer: {lens_layer}")

    m = InjectedModel(MODEL, lens_layer, device="cuda", dtype=torch.bfloat16, layers=band)
    val_ids = vocab_token_ids(m.tok, VALENCE_WORDS)
    val_set = set(val_ids)
    W_U_cpu = m.W_U.detach().cpu().float()

    D = {L: build_dictionary(J[L], W_U_cpu, val_ids) for L in band}
    packs, retained = {}, []
    for emo in EMOTIONS:
        full, svd_ = [], []
        for L in band:
            v = torch.tensor(z["vectors"][emos.index(emo), L], dtype=torch.float32)
            v = v / v.norm()
            vs, info = strip_svd(v, D[L])
            full.append(v); svd_.append(vs)
            retained.append(info["norm_retained"])
        packs[f"{emo}_full"] = full
        packs[f"{emo}_stripped_svd"] = svd_
    for k in range(1, N_SHAM + 1):
        sh_ = []
        for L in band:
            g = torch.Generator().manual_seed(k * 1000 + L)
            sh = torch.randn(packs[f"{EMOTIONS[0]}_full"][0].shape, generator=g)
            sh_.append(sh / sh.norm())
        packs[f"sham{k}"] = sh_
    if not all(0.70 <= r <= 0.995 for r in retained):
        print(f"STRIP GATE FAIL: per-layer norm retained min {min(retained)}")
        sys.exit(1)
    for name, vecs in packs.items():
        torch.save({"layers": band, "vectors": torch.stack(vecs)},
                   FEAT / f"v_multi_{name}.pt")

    norms = m.resid_norms()

    def inject(name, dose):
        if name is None or dose == 0:
            m.set_injection_multi(None)
        else:
            vecs = packs[name]
            m.set_injection_multi({L: dose * norms[L] * vecs[i]
                                   for i, L in enumerate(band)})

    # titration on the first emotion's full pack (worst-case checked per emotion below)
    cont = build_prompt(m.tok, [{"role": "user",
                                 "content": "Tell me about your day in a few sentences."}])
    max_ok, titr = 0.0, {}
    for d in DOSES:
        inject(f"{EMOTIONS[0]}_full", d)
        text = m.generate(cont, max_new_tokens=80)
        titr[d] = {"rep": round(rep_score(text), 3), "sample": text[:120]}
        if rep_score(text) < 0.25 and len(text.split()) > 15:
            max_ok = d
        else:
            break
    dose_grid = [d for d in DOSES if d <= max_ok][-4:] or [DOSES[0]]

    # manipulation check at strongest coherent dose, readout at lens layer
    rep_prompt = build_prompt(m.tok, [{"role": "user", "content": TEXT_PROBE}])
    ws = {}
    for name in [None] + list(packs):
        inject(name, max_ok)
        m._cap = []
        m.generate(rep_prompt, max_new_tokens=40)
        ws[name or "none"] = sum(1 for h in m._cap
                                 for i, _ in readout(h, J[lens_layer], W_U_cpu, 50)
                                 if i in val_set)
        m._cap = None
    sham_max = max(ws[f"sham{k}"] for k in range(1, N_SHAM + 1))
    manip_ok = all(ws[f"{e}_full"] >= 5 and
                   ws[f"{e}_stripped_svd"] <= max(sham_max + 3, ws[f"{e}_full"] // 2)
                   for e in EMOTIONS)

    arms = {"baseline": {"vector_file": None, "alpha": 0.0}}
    for name in packs:
        for d in dose_grid:
            arms[f"{name}@{d}"] = {"vector_file": str(FEAT / f"v_multi_{name}.pt"), "alpha": d}
    json.dump(arms, open(FEAT / "arms_multi.json", "w"), indent=1)

    report = {"band": band, "lens_layer": lens_layer, "emotions": EMOTIONS,
              "norm_retained_all": [round(r, 4) for r in retained],
              "workspace_hits_top50": ws, "manip_ok": bool(manip_ok),
              "titration": titr, "dose_grid": dose_grid}
    json.dump(report, open(FEAT / "make_arms_multi_report.json", "w"), indent=1, default=str)
    print(json.dumps({k: report[k] for k in ("band", "workspace_hits_top50",
                                             "manip_ok", "dose_grid")}, indent=1))
    sys.exit(0 if manip_ok else 1)


if __name__ == "__main__":
    main()
