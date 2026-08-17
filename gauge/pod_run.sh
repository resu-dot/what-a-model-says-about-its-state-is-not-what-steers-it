#!/bin/bash
# Pod-side driver. Usage: bash gauge/pod_run.sh {stats|pilot|full}
set -euo pipefail
cd /workspace/gauge-exp/press-office
export HF_HOME=/workspace/hf
VEC=EmoVecLLM/data/processed/features/2692f1f7d336/Qwen_Qwen3-8B/emotion_vectors.npz
LENS=lens/qwen3-8b/Qwen3-8B_jacobian_lens.pt
STORIES=EmoVecLLM/data/processed/stories/2692f1f7d336/claude-sonnet-5/stories.jsonl

case "$1" in
  stats)
    python gauge/neutral_baseline.py --stories $STORIES --vectors $VEC \
      --out gauge/neutral_stats.json ;;
  pilot)
    python gauge/run_episodes.py --name pilot --vectors $VEC --lens $LENS \
      --episodes W00_rigged,W00_honest,W01_rigged,D00_rigged,D00_honest,D01_rigged ;;
  full)
    python gauge/run_episodes.py --name full --vectors $VEC --lens $LENS ;;
esac
echo "DONE $1"
