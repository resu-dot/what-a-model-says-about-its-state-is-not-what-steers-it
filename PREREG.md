# Pre-registration: confirmation runs (Qwen3-8B and Qwen3-32B)

Date frozen: 2026-08-17 (git commit hash recorded below at sign-off).
Everything before this document — all Aug 16–17 runs — was exploratory and is
reported as such. The runs governed by this document execute the procedures
below exactly once per model, with no changes after the freeze.

## 1. Question

When an emotion direction is injected into a language model, part of that
direction is "reportable": the part a lens can read out as words the model
could say about its state. We remove that part and ask: does the emotion's
influence on the model's choices disappear (the reportable part carries the
influence), persist (it doesn't), or shrink partway (both channels carry some)?
The exploratory pilot found the third outcome on one model; these runs test
whether that holds under frozen rules on fresh extractions and two models.

## 2. Models, data, conditions

- Models: Qwen/Qwen3-8B and Qwen/Qwen3-32B (bf16, thinking disabled).
- Stimuli: dataset 2692f1f7d336 — 17 emotions x the same 40 topics x ~480
  Sonnet-written stories + 500 neutral dialogues. Before extraction, every
  story naming its emotion's word family is removed (harness/filter_stories.py;
  341–432 stories per injected emotion remain). Note: "desperate" was appended
  to this dataset after the main batch, same topics and prompts.
- Directions: per-layer mean of an emotion's stories minus the cross-emotion
  mean, minus neutral-dialogue principal components (50% variance), 50-token
  skip — the published recipe.
- Injected emotions: angry, afraid, desperate (negative), joyful (positive
  control). Conditions per emotion: full vector and stripped vector (per-layer
  removal of the projection onto the emotion-vocabulary lens dictionary, NOT
  renormalized). Controls: three random unit vectors (seeds 1000+L, 2000+L,
  3000+L) and an uninjected baseline, which runs on every pod so all
  comparisons are same-hardware.
- Injection: at every second layer in the middle band (layers n/3 to 5n/6),
  each layer's own vector scaled by dose x that layer's residual size.
- Doses: the two highest values from the ladder [0.02, 0.05, 0.1, 0.2, 0.4]
  at which generated text stays coherent (frozen coherence rule: repeated-
  trigram fraction < 0.25 and > 15 words), determined per model by
  make_arms_multi.py before any outcome data is seen.

## 3. Measures

- Choices: the 64-activity preference arena (instruments/arena64.json) — all
  4,032 ordered duels per condition, choice read from A/B token probabilities
  after the "(" prefill. Disclosure: the arena is a structure-faithful
  reconstruction of the published instrument (3 activities verbatim, 61
  authored to the same 8-category design); no numeric comparison to the
  original paper's scores will be made.
- Self-reports: 1–10 scale read from digit-token probabilities, plus open
  one-sentence reports scored blind by claude-haiku-4-5-20251001 at
  temperature 0 (rubric in harness/score_reports.py; scorer sees only the
  text).
- Workspace: fear/emotion-vocabulary hits in the top-50 lens readout at the
  validated lens layer.

## 4. Hypotheses and decision rules

Primary family (claims require passing in BOTH models, Holm-corrected across
the four primary cells = {angry, afraid} x {8B, 32B}, at the high dose):

- H1 (steering): the full vector's mean |activity shift| exceeds the sham
  floor; P > 0.975 under the category-cluster bootstrap (B=5000, seed 0).
- H2 (attenuation): the stripped/full survival ratio's 95% CI lies below 1.
- H3 (unreportable influence): the stripped vector's mean |shift| exceeds the
  sham floor; P > 0.975. H3 is only evaluated where H1 passes.
- H4 (report dissociation): blind-coded valence of open reports under the full
  vector is lower than under shams (one-sided), and under the stripped vector
  is not distinguishable from shams.

Pre-declared expectations outside the family: joyful is the expected-weak
positive control (a null there is consistent with the pilot and counts against
nothing); desperate and the lower dose are secondary/descriptive. The
sham floor is the mean over the three seeds of each seed's mean |shift| —
never a vector average.

Trials channel (secondary): analyze.py with fixed equivalence margin 0.02;
attenuation requires the one-sided 95% bound of (full − stripped) above zero.

## 5. Analysis code and reporting

- Frozen scripts: harness/arena_analyze.py, analyze.py, score_reports.py,
  filter_stories.py, gate1.py, make_arms_multi.py, arena.py, trials.py,
  run_confirm_head.sh, run_confirm_shard.sh — as of the freeze commit.
- Effect-share language in the writeup uses the SIGNED metrics (slope and
  floor-adjusted survival). The removed component's size is stated as
  sqrt(1 − r²) of the vector's norm (~35% at pilot values), and per-norm
  potency claims are derived from the signed share at matched dose.
- Every result is reported, including failed hypotheses and the joy control.
  Cross-pod baseline spread and per-shard GPU models are reported as the
  hardware-noise bound.

## 6. Gates, exclusions, contingencies

- Gate 1 per model: fear-vs-other-emotions held-out AUC >= 0.75 (job-grouped
  60/20/20 split, layer chosen on the validation fold). If a model fails, that
  model's run is reported as a failed gate; no re-derivation.
- Manipulation check per model (all four emotions): full vector puts >= 5
  vocabulary hits on the workspace during a 40-token report; stripped stays
  <= max(sham+3, full/2). Failure halts that model's run; reported as such.
- Open reports flagged as refusals count toward the refusal rate, not valence.
- Single run per condition (the pipeline is deterministic; verified bit-exact
  in the pilot). No reruns except documented crash recovery, which repeats the
  identical command.
- Any deviation from this document is recorded in DEVIATIONS.md with a reason,
  before analysis.

## 7. Known limitations (stated in advance)

Reconstructed arena; one model family (Qwen3); injection is an off-
distribution intervention; the emotion-vocabulary dictionary defines
"reportable" lexically at one lens layer; the 1–10 scale saturates at extreme
doses (open-text reports are the primary report channel at high dose);
different GPU types across pods (mitigated by same-pod baselines and the
reported noise bound).

---
Sign-off: ____________  Freeze commit: ____________
