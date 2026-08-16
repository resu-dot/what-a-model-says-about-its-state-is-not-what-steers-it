#!/bin/bash
# One-time pod bootstrap. Run from /workspace/press-office.
set -euo pipefail
pip -q install "transformers>=4.55" accelerate scikit-learn h5py pandas einops pyyaml wandb tqdm scipy
huggingface-cli download Qwen/Qwen3-8B --quiet &   # pull weights while anything else runs
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
wait
echo "SETUP DONE"
