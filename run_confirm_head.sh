#!/bin/bash
# Confirmation serial head (run on ONE pod per tier). Usage: ./run_confirm_head.sh 8b|32b
set -euo pipefail
cd "$(dirname "$0")"
export HF_HOME=/workspace/hf
TIER=${1:?8b or 32b}
if [ "$TIER" = "8b" ]; then MODEL=Qwen/Qwen3-8B; LENSDIR=qwen3-8b; LENSFILE=Qwen3-8B_jacobian_lens.pt
else MODEL=Qwen/Qwen3-32B; LENSDIR=qwen3-32b; LENSFILE=Qwen3-32B_jacobian_lens.pt; fi
SPEC=2692f1f7d336
FEAT=EmoVecLLM/data/processed/features/$SPEC/$(echo $MODEL | tr / _)
LENS=lens/$LENSDIR/$LENSFILE
EMOTIONS=angry,afraid,desperate,joyful
exec > >(tee -a confirm_head_$TIER.log) 2>&1

echo "=== HEAD [1/4] leakage filter (lemma-aware, idempotent) ==="
python harness/filter_stories.py EmoVecLLM/data/processed/stories/$SPEC/claude-sonnet-5/stories.jsonl

echo "=== HEAD [2/4] extraction ==="
WANDB_MODE=disabled EMOVEC_BASELINE=paper EMOVEC_BASELINE_PCS=0 EMOVEC_SKIP_TOKENS=50 \
  EMOVEC_STORIES_PATH=EmoVecLLM/data/processed/stories/$SPEC/claude-sonnet-5/stories.jsonl \
  python EmoVecLLM/scripts/extract_features.py --non-interactive \
  --target-model $MODEL --precision bf16 --demo 0 --work-dir EmoVecLLM
python - "$FEAT" "$SPEC" <<'PYCHK'
import json, sys
mf = json.load(open(f"{sys.argv[1]}/features_manifest.json"))
assert sys.argv[2] in json.dumps(mf), f"extraction did not use spec {sys.argv[2]}: {mf}"
print("extraction spec check OK")
PYCHK

echo "=== HEAD [3/4] gate 1 (job-grouped) ==="
python harness/gate1.py "$FEAT"

echo "=== HEAD [4/4] arms: 4 emotions x full/stripped + 3 shams, titration ==="
( cd harness && python make_arms_multi.py $MODEL "../$FEAT" "../$LENS" $EMOTIONS )
python - "$FEAT" <<'PYEOF'
import json, sys
r = json.load(open(f"{sys.argv[1]}/make_arms_multi_report.json"))
g = json.load(open(f"{sys.argv[1]}/gate1.json"))
doses = r["dose_grid"][-2:]                 # frozen rule: two highest coherent doses
json.dump({"layer": g["layer"], "doses": doses, "band": r["band"]},
          open("confirm_env.json", "w"))
print("confirm_env.json:", {"layer": g["layer"], "doses": doses})
PYEOF
echo "=== HEAD DONE — launch shards ==="
