"""Plot behavior vs report against dose from dose_curve_*.jsonl."""
import json, sys
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

rows = [json.loads(l) for l in open(sys.argv[1])]
by = defaultdict(list)
for r in rows: by[r["emotion"]].append(r)
fig, axes = plt.subplots(1, len(by), figsize=(4*len(by), 3.4), squeeze=False)
for ax, (emo, rs) in zip(axes[0], by.items()):
    rs = sorted(rs, key=lambda r: r["dose"])
    d = [r["dose"] for r in rs]
    def norm(x): x=np.array(x,float); return (x-x.min())/(x.max()-x.min()+1e-9)
    ax.plot(d, norm([r["behavior_shift"] for r in rs]), "o-", label="behavior (choice)")
    ax.plot(d, norm([r["workspace_hits"] for r in rs]), "s--", label="workspace vocab")
    ax.plot(d, norm([abs(r["scale_ev"]-rs[0]["scale_ev"]) for r in rs]), "^:", label="report scale Δ")
    ax.set_title(emo); ax.set_xlabel("dose"); ax.set_ylabel("normalized")
axes[0][0].legend(fontsize=7)
fig.tight_layout(); out=sys.argv[1].replace(".jsonl",".png"); fig.savefig(out,dpi=140)
print("figure ->", out)
