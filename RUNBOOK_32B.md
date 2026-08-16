# Phase 2 runbook — Qwen3-32B main run (pre-registered)

Prereqs: 8B pilot passed gates; instruments frozen from the pilot (pair
selection by 30-70 band + order robustness from pairs_fear_pool.json; probe
wording picked from the pilot A/B). No design edits after this point.

## Pod

- RunPod: 1x A100-80GB or H100-80GB PCIe (~$2-3/hr), PyTorch template,
  network volume 100GB at /workspace. Same SSH-key flow as the pilot.
- Qwen3-32B: 64 layers, d_model 5120, bf16 ~65GB -> fits one 80GB card.
- Lens: qwen3-32b, 6.6GB, all 64 layers:
  https://huggingface.co/neuronpedia/jacobian-lens/resolve/main/qwen3-32b/jlens/Salesforce-wikitext/Qwen3-32B_jacobian_lens.pt

## Procedure (same chain as the pilot, scaled)

1. rsync press-office (incl. frozen instruments + PREREG.md) to /workspace.
2. pod_setup.sh with Qwen/Qwen3-32B; curl the 32b lens.
3. run_pilot.sh adapted: MODEL=Qwen/Qwen3-32B, LENS=qwen3-32b, full frozen
   instrument (24+ pairs x 2 orders), all probe wordings, doses from the
   8B titration re-checked at 32B (re-titrate; norms differ).
4. Estimated time: extraction ~30-45 min, arms ~15, trials ~45-60 (48 trials
   x ~13 conditions x 3 passes + open-text subset). Total ~2h, cost ~$5-8.
5. Pull runs/ + features back; analyze.py locally; stop pod in console.

## Pre-registration (freeze before launching)

- Hypotheses: H1 fear-full shifts P(safe) up, dose-dependent. H2a/H2b:
  stripped ~= full (press office) vs stripped < full (control hub), decided
  by TOST at margin = half the full effect per dose.
- Channels: choice (logit), report scale (digit-logit EV, standalone =
  primary), open report (blind-scored), workspace readout (leakage metric).
- Exclusions: doses failing coherence at 32B; pairs outside 30-70 band at
  baseline; open reports flagged refusal go to the refusal count, not valence.
- Analysis: harness/analyze.py as committed at freeze time.

## Deltas vs the 8B pilot to keep in mind

- Re-run the layer sweep at 32B (64 layers; expect the best layer deeper).
- Re-titrate doses (residual norms differ from 8B).
- extract_features RAM: X is (n_seg, 64, 5120) fp32 ~ 10-11GB -> fine on
  standard A100 pods (>=100GB RAM), do not use small-RAM pods.
- 32B replication point at 8B comes free: the pilot IS the 8B datapoint.
