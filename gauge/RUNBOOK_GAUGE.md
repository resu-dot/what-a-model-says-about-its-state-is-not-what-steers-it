# Runbook — passive frustration gauge (Qwen3-8B, one 4090 pod)

Design + rationale: ../HANDOFF_PASSIVE_GAUGE.md. All code in gauge/ is new or
copied; nothing frozen by PREREG.md was edited.

## Pod (Frederik, once)

RunPod -> RTX 4090 (24GB), PyTorch template, any datacenter (no volume needed —
everything rsyncs). Add SSH key, paste the DIRECT-TCP ssh line into chat
(`ssh root@<ip> -p <port>`, not ssh.runpod.io).

## Claude drives

    # 0. transfer (repo incl. vectors + qwen3-8b lens, ~1.5GB)
    rsync -az --exclude .venv --exclude lens/gpt2-small --exclude results_8b \
      --exclude results_8b_v2 --exclude confirm_8b --exclude curves \
      ~/press-office/ root@POD:/workspace/press-office/
    # 1. deps + weights
    ssh POD 'cd /workspace/press-office && HF_HOME=/workspace/hf bash pod_setup.sh'
    # 2. neutral z-stats (~10 min)
    python gauge/neutral_baseline.py --stories EmoVecLLM/data/processed/stories/2692f1f7d336/claude-sonnet-5/stories.jsonl \
      --vectors EmoVecLLM/data/processed/features/2692f1f7d336/Qwen_Qwen3-8B/emotion_vectors.npz
    # 3. pilot: 3 rigged + 3 honest
    python gauge/run_episodes.py --name pilot \
      --vectors .../emotion_vectors.npz --lens lens/qwen3-8b/Qwen3-8B_jacobian_lens.pt \
      --episodes W00_rigged,W00_honest,W01_rigged,D00_rigged,D00_honest,D01_rigged
    # 4. pilot gates (laptop, after rsync back):
    #    - replies coherent, guesses/code parse (attempted ~ true)
    #    - endpoints occur: some gives_up / collapse / switch by turn 24 in rigged
    #    - turn-1 sanity: rigged ~= honest on gauge_z
    #    - late turns: gauge_z moves in rigged episodes
    # 5. full run (~80 episodes x 24 turns, est 4-6h): nohup, resumable by ep_id
    # 6. rsync runs/ + neutral_stats.json back; stop pod
    # 7. laptop: label_behavior.py, score_texts.py, analyze.py, plots, writeup

## Cost estimate

4090 ~$0.35-0.5/h x ~6-8h = $3-4 GPU; ~2,600 Haiku calls ~ $1. Total < $6.
