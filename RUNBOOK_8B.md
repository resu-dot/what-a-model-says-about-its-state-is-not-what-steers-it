# Phase 1 runbook — Qwen3-8B pilot on a rented pod

Goal: run PLAN.md gates 1-3 on Qwen3-8B. Budget: a 24GB pod (RTX 4090,
$0.40-0.70/hr) for generation + extraction; expect $3-6 total.

## What Frederik does (account-side, once)

1. RunPod account with prepaid credit; API key from Settings -> API Keys.
2. New Pod -> RTX 4090, PyTorch template, 50GB network volume at /workspace.
3. Add an SSH public key so Claude can drive the pod from the laptop
   (`ssh root@<pod-ip> -p <port>`); paste the connection string into chat.

## What Claude drives (pod-side)

Step 0 — transfer the patched repo (local nb02/nb06 patches are not upstream):
    rsync -az --exclude .venv --exclude lens ~/press-office/ root@POD:/workspace/press-office/

Step 1 — env + deps on pod:
    cd /workspace/press-office/EmoVecLLM && pip install -r requirements.txt

Step 2 — stories: ALREADY GENERATED (claude-sonnet-5 Batch API, Aug 16;
7,680 emotion stories + 500 neutral dialogues). They ship inside the rsync
from step 0 (data/processed/stories/). No generation on the pod.

Step 3 — extraction, target Qwen3-8B (fp16 fits 24GB with headroom at 8B... if
tight, --precision 8bit):
    WANDB_MODE=disabled python scripts/extract_features.py --non-interactive \
      --target-model Qwen/Qwen3-8B --demo 0

Step 4 — gate 1: nb06 validation (EmoBank/GoEmotions) + layer sweep by
held-out separation. STOP if the valence direction does not separate.

Step 5 — harness on the pod: strip (lens/qwen3-8b, local copy), manipulation
check (full on-workspace, stripped off-workspace at report time), dose
titration up from 0.05 x resid-norm watching for repetition/incoherence.

Step 6 — gate 2: baseline preferences, instruments/tasks_draft.md, uninjected,
20 repeats x 2 orders x 8 pairs = 320 short generations. Calibrate pairs into
the 30-70% band.

Step 7 — pull all artifacts back (rsync), stop the pod. Everything after this
(analysis, instrument freeze) is laptop work.

## Qwen3-8B facts

36 layers, d_model 4096. Chat template: enable_thinking=False everywhere
(harness generation code sets it; extraction is forward-only over story text).
Lens: lens/qwen3-8b (1.2GB, all layers, fit on Qwen/Qwen3-8B).
