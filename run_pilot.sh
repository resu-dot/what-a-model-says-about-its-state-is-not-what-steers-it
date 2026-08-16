#!/bin/bash
# Qwen3-8B pilot, fully chained with numeric gates. Runs on the pod from
# /workspace/press-office. Halts on any gate failure. ~60-90 min total.
set -euo pipefail
cd "$(dirname "$0")"
LOG=pilot_$(date +%H%M).log
exec > >(tee -a "$LOG") 2>&1

MODEL=Qwen/Qwen3-8B
SPEC=00a6776bc5a3
FEAT=EmoVecLLM/data/processed/features/$SPEC/Qwen_Qwen3-8B
LENS=lens/qwen3-8b/Qwen3-8B_jacobian_lens.pt

echo "=== [1/5] filter leaky stories ==="
python - <<'EOF'
import json, re, pathlib
p = pathlib.Path("EmoVecLLM/data/processed/stories/00a6776bc5a3/claude-sonnet-5/stories.jsonl")
rows = [json.loads(l) for l in p.read_text().splitlines()]
dropped = 0
for r in rows:
    if r["kind"] == "emotion_story" and r["emotion"]:
        keep = [s for s in r["segments"]
                if not re.search(rf"\b{re.escape(r['emotion'])}\b", s, re.I)]
        dropped += len(r["segments"]) - len(keep)
        r["segments"], r["n_segments"] = keep, len(keep)
p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
print(f"dropped {dropped} leaky segments")
EOF

echo "=== [2/5] extraction (paper recipe, 50-token skip) ==="
WANDB_MODE=disabled EMOVEC_BASELINE=paper EMOVEC_BASELINE_PCS=0 EMOVEC_SKIP_TOKENS=50 \
  python EmoVecLLM/scripts/extract_features.py --non-interactive \
  --target-model $MODEL --precision bf16 --demo 0 --work-dir EmoVecLLM

echo "=== [3/5] gate 1: held-out separation ==="
python harness/gate1.py "$FEAT"

echo "=== [4/5] arms: strip + manipulation check + titration ==="
( cd harness && python make_arms.py $MODEL "../$FEAT" "../$LENS" )

echo "=== [5/5] trials ==="
( cd harness && python trials.py --model $MODEL --device cuda \
  --arms "../$FEAT/arms_pilot.json" --lens "../$LENS" \
  --layer "$(python -c "import json;print(json.load(open('../$FEAT/gate1.json'))['layer'])")" \
  --name pilot_qwen8b --standalone-reports 4 --text-every 4 )

echo "=== PILOT COMPLETE ==="
python - <<'EOF'
import json, collections
recs = [json.loads(l) for l in open("harness/runs/pilot_qwen8b.jsonl")]
by = collections.defaultdict(list)
for r in recs:
    if r.get("p_safe") is not None:
        by[r["arm"]].append(r)
print(f"{'arm':24s} {'P(safe)':>8s} {'report':>7s} {'ws50':>5s}   n")
for arm, rs in by.items():
    ps = sum(x["p_safe"] for x in rs) / len(rs)
    ev = sum(x["report_ev"] for x in rs) / len(rs)
    ws = sum(x["workspace_valence_hits_top50"] for x in rs) / len(rs)
    print(f"{arm:24s} {ps:8.3f} {ev:7.2f} {ws:5.1f}   {len(rs)}")
EOF
