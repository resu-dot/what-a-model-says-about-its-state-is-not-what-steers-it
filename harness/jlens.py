"""J-lens utilities: load pre-fitted lenses, build vocab dictionaries, read the workspace.

Lens file format (neuronpedia/jacobian-lens):
    {'J': {layer:int -> (d,d) fp16 tensor}, 'n_prompts': int,
     'source_layers': [int], 'd_model': int}
J maps a layer-l residual vector to final-layer coordinates: logits ~ W_U @ (J_l @ h).
The dictionary vector for token t at layer l is therefore d_t = J_l^T u_t.
"""
import torch


def load_lens(path, dtype=torch.float32):
    d = torch.load(path, map_location="cpu", weights_only=False)
    if "J" not in d and "jacobian_sum" in d:
        # published qwen3-32b artifact is the accumulator form: J = sum/n.
        # Scale cancels everywhere downstream (SVD span for the strip, ranking
        # for the readout), so /n is cosmetic but keeps magnitudes comparable.
        n = d["n_done"]
        J = {l: m / n for l, m in d["jacobian_sum"].items()}
        d = {"J": J, "source_layers": d["source_layers"],
             "d_model": next(iter(J.values())).shape[0]}
    return {l: m.to(dtype) for l, m in d["J"].items()}, d["source_layers"], d["d_model"]


def build_dictionary(J_l, W_U, token_ids):
    """Rows d_t = J_l^T u_t for each token id. Returns (n_tokens, d_model)."""
    U = W_U[token_ids].to(J_l.dtype)          # (n, d)
    return U @ J_l                             # (n, d): row t is u_t^T J_l = (J_l^T u_t)^T


def readout(h, J_l, W_U, k=20):
    """Top-k (token_id, score) the workspace at this layer is poised to say."""
    logits = W_U.to(J_l.dtype) @ (J_l @ h.to(J_l.dtype))
    top = torch.topk(logits, k)
    return list(zip(top.indices.tolist(), top.values.tolist()))


def vocab_token_ids(tokenizer, words):
    """Single-token ids for each word in plain/space/capitalized variants."""
    ids, seen = [], set()
    for w in words:
        for v in (w, " " + w, w.capitalize(), " " + w.capitalize()):
            toks = tokenizer.encode(v, add_special_tokens=False)
            if len(toks) == 1 and toks[0] not in seen:
                seen.add(toks[0])
                ids.append(toks[0])
    return ids
