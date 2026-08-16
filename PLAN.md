# Press office or control hub?

Does an emotion's influence on choices run through the model's reportable (J-space) workspace?

Apart Research digital minds hackathon. Plan frozen from the Aug 16 2026 design chat.

## The experiment

Inject an emotion direction into an open model and measure three channels at once: the verbal workspace (J-lens readout), self-report (valence 1-10 + one sentence), and forced choice (tasks the model must complete). Then inject a stripped version of the same vector, with the J-space (reportable) component projected out.

- Choices shift as much as with the full vector: the workspace is a press office. Reports describe, they do not steer.
- Choices shift less: the workspace is a control hub. Becoming reportable adds behavioral force.

Arms: full / stripped / matched non-valence control / sham (random, norm-matched). Doses: 0 + 4 strengths, titrated to just below collapse. Equivalence testing (TOST or Bayesian), not "no significant difference".

## Decisions log

- Model: Qwen3-32B (chat model, the only variant released) for the main run. Qwen3-8B for the pilot. Rationale: injection-to-self-report is evidenced only at 32B and up (arXiv 2602.20031); pre-fitted J-lenses exist for the whole Qwen3 ladder.
- Emotion directions: EmoVecLLM pipeline (difference of means, emotion vs neutral stories). Stimuli reused as fixed text; generator does not need to equal target.
- Strip: J-lens dictionary d_t = J_l^T u_t for a valence vocabulary; subtract the J-space component (SVD projection, gradient-pursuit variant as robustness check). Published procedure from the workspace paper.
- Lens: pre-fitted, from huggingface.co/neuronpedia/jacobian-lens (qwen3-32b: all 64 layers, fit on Qwen/Qwen3-32B, wikitext-103). Caveat: prose corpus; transfer to chat formatting is tested by the manipulation check.
- Choice measure: forced choice with direction-appropriate options only (Frederik, Aug 16: pleasant-vs-tedious scrapped, was copied in from another project). Each injected direction gets an instrument matched to what it carries: fear -> safe-vs-risky, anger -> confront-vs-accommodate (drafted in instruments/emotion_matched_drafts.md); if a valence composite is injected, an appropriate valence-matched choice set must be designed first.
- DECIDED (Frederik, Aug 16 eve): choice is read by the Anthropic paper's logit method ("Would you prefer to (A) ... or (B) ...?", assistant prefill "(", softmax over A/B token logits; counterbalanced orders). Self-report is BOTH the 1-10 scale (expected value over digit-token logits, two-pass split of the 1/10 first-token ambiguity) AND a one-or-two-sentence open report on a trial subset, scored by a blind model (Haiku, rubric: valence/names_emotion/refusal; scorer sees only the text). Implemented in harness/trials.py + harness/score_reports.py. Probe wording variants still get the pilot A/B before freezing.
- OPEN DECISION (settled empirically in the pilot): which direction(s) carry the main experiment — discrete emotions with matched instruments, or the fear-vs-anger dissociation (both negative valence, opposite action tendencies, each the other's control). Pilot probes all candidate instruments uninjected + full-injection at small N; primary = stable baseline + largest injection effect. All 16 emotions are extracted individually regardless, so extraction serves every candidate.
- Same layer for extraction, lens, and injection. Layer picked by held-out separation sweep.
- Qwen3 thinking mode disabled everywhere (enable_thinking=False), frozen before the main run.
- Report probe wording is high leverage (0.3% to 39% sensitivity swing in the 32B literature). Freeze after piloting.

## Gates, in order

1. Extraction gate: valence direction separates held-out pos/neg contexts, mild steering shifts sentiment. No pass, stop.
2. Baseline preference gate: stable forced choices across repeats and option order without injection. Noise baseline is itself a publishable null.
3. Manipulation check: full injection puts valence on the workspace, stripped injection does not, verified at report time (leakage check).
4. Freeze instruments (probes, prompts, coding rules, analysis) before the main run.

## Phases

- Phase 0 (local, Mac, free): end-to-end miniature on gpt2. EmoVecLLM lite run, gpt2-small lens, strip module, injection harness, toy readout. Proves every line of custom code before renting.
- Phase 1 (rented GPU, ~24GB): Qwen3-8B pilot. Real extraction, layer sweep, manipulation check, dose titration, draft instruments.
- Phase 2 (rented H100/A100-80GB): Qwen3-32B main run, pre-registered. Also the 8B replication point for the scale story.

## Verified (Aug 16)

- EmoVecLLM has a plain HF fallback backend (AutoModelForCausalLM, output_hidden_states) if TransformerLens rejects a model. Layer handling is dynamic. nb06 validation is model-agnostic.
- Shipped stimuli are lite: 171 emotions x 3 topics + 100 neutral dialogues (generator Qwen2.5-7B). To do: expand nb02 manifest to ~100 topics for the chosen emotions.
- qwen3-32b lens: one 6.6GB .pt, all 64 layers, fit on the exact chat model we run.
- Qwen3-32B: 64 layers, d_model 5120, vocab 151936, untied embeddings (use lm_head as W_U).
