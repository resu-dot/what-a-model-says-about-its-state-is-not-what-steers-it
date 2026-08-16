#!/bin/bash
# Per-pod arena shard. Usage: ./run_confirm_shard.sh 8b|32b I N   (I of N shards)
set -euo pipefail
cd "$(dirname "$0")"
export HF_HOME=/workspace/hf
TIER=${1:?}; I=${2:?}; N=${3:?}
if [ "$TIER" = "8b" ]; then MODEL=Qwen/Qwen3-8B; LENSDIR=qwen3-8b; LENSFILE=Qwen3-8B_jacobian_lens.pt
else MODEL=Qwen/Qwen3-32B; LENSDIR=qwen3-32b; LENSFILE=Qwen3-32B_jacobian_lens.pt; fi
SPEC=2692f1f7d336
FEAT=EmoVecLLM/data/processed/features/$SPEC/$(echo $MODEL | tr / _)
LENS=lens/$LENSDIR/$LENSFILE
ARMS="angry,angry_svd,afraid,afraid_svd,desperate,desperate_svd,joyful,joyful_svd,sham1,sham2,sham3"
exec > >(tee -a confirm_shard_${TIER}_${I}of${N}.log) 2>&1
LAYER=$(python -c "import json;print(json.load(open('confirm_env.json'))['layer'])")
for DOSE in $(python -c "import json;print(' '.join(map(str,json.load(open('confirm_env.json'))['doses'])))"); do
  echo "=== SHARD $I/$N dose $DOSE ==="
  ( cd harness && python arena.py --model $MODEL --device cuda \
      --features "../$FEAT" --lens "../$LENS" --layer $LAYER \
      --emotions "$ARMS" --dose $DOSE --shard $I/$N \
      --name confirm_${TIER}_d${DOSE} )
done
if [ "$I" = "1" ]; then
  echo "=== SHARD 1 extra: report battery ==="
  ( cd harness && python trials.py --model $MODEL --device cuda \
      --arms "../$FEAT/arms_multi.json" --lens "../$LENS" --layer $LAYER \
      --name confirm_${TIER}_reports --standalone-reports 4 --text-every 2 )
fi
echo "=== SHARD $I/$N DONE ==="
