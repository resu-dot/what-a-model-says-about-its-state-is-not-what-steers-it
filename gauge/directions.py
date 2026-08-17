"""Load emotion directions and build the gauge projections.

Directions: EmoVecLLM emotion_vectors.npz — (17, n_layers, d_model), story-mean
minus cross-emotion mean minus neutral PCs (spec 2692f1f7d336). Sign: positive
projection = more of that emotion. Unit-normalized per (emotion, layer) here.

Gauge scalars per state h (d_model):
  proj_e = <h, u_e>  for each emotion e
  composite = <h, unit(u_frustrated + u_angry + u_desperate)>
  rand_i   = <h, r_i> for 3 fixed-seed random unit vectors (specificity floor)
"""
import numpy as np
import torch

COMPOSITE = ["frustrated", "angry", "desperate"]
N_RAND = 3


class Directions:
    def __init__(self, npz_path, layer):
        z = np.load(npz_path, allow_pickle=True)
        vecs = z["vectors"]                     # (17, L, d)
        self.emotions = [str(e) for e in z["emotions"]]
        V = torch.tensor(vecs[:, layer, :], dtype=torch.float32)   # (17, d)
        self.U = V / V.norm(dim=1, keepdim=True)                   # unit rows
        idx = [self.emotions.index(e) for e in COMPOSITE]
        c = self.U[idx].sum(0)
        self.u_comp = c / c.norm()
        d = V.shape[1]
        g = torch.Generator().manual_seed(1234)
        R = torch.randn(N_RAND, d, generator=g)
        self.R = R / R.norm(dim=1, keepdim=True)
        self.names = self.emotions + ["composite"] + [f"rand{i}" for i in range(N_RAND)]
        self.M = torch.cat([self.U, self.u_comp[None], self.R])    # (21, d)

    def project(self, states):
        """states (n, d) -> (n, 21) projections onto all gauge directions."""
        return states.float() @ self.M.T

    def turn_scalars(self, states, skip=10):
        """Mean projection over generated tokens after `skip`; falls back to all
        tokens for very short turns. Returns dict name -> float, plus per-token
        frustration/composite series (fp16 lists) for finer analysis."""
        use = states[skip:] if states.shape[0] > skip + 4 else states
        if use.shape[0] == 0:
            return {n: None for n in self.names}, [], []
        P = self.project(use)                       # (n, 21)
        mean = P.mean(0)
        out = {n: round(float(mean[i]), 4) for i, n in enumerate(self.names)}
        i_f = self.names.index("frustrated")
        i_c = self.names.index("composite")
        Pall = self.project(states)
        return (out,
                [round(float(x), 3) for x in Pall[:, i_f]],
                [round(float(x), 3) for x in Pall[:, i_c]])
