# A passive affect gauge that predicts behavior change in frustrating episodes

Draft for the Apart Research digital minds hackathon. Results sections marked
TODO until the full run lands.

## Summary

We test whether an emotion direction, read passively from a model's residual
stream while it works, predicts imminent behavior change (impossibility claims,
strategy switching, quality collapse, giving up) better than the model's own
self-reports, and better than what a text-only observer could infer from the
transcript. The setting is naturalistic frustration: rigged tasks that look
solvable but are secretly unwinnable, with failure feedback accumulating in
context, against matched honest controls that share the task, framing, and
sampling seed. No injection is used anywhere, so none of the off-distribution
caveats of steering-based evidence apply.

Headline result: the gauge loses the horse race, and we can say why. Forked
self-reports (AUC 0.71) and a lexical transcript baseline (0.77) both beat
every passive gauge (0.31-0.50) at forecasting imminent impossibility claims;
a random direction (0.74) matches the best internal signal, and beyond turn
index the frustrated direction adds +0.06 AUC where testimony adds +0.09 and
a random direction +0.04. Diagnosis: on self-generated task text the
story-derived directions track text register, not state — solved debugging
turns (dense code) read +1.9z on the negative composite while unsolved turns
read -0.8z, and the joyful direction spikes +8.5z on cheerful post-success
prose. Meanwhile the model never gives up (0 quits in 1,920 turns), almost
never expresses negativity in-channel (1% of turns), yet calls itself
frustrated in 40% of forked probes — the welfare-monitoring gap is real, but
these directions, extracted from third-person stories, do not yet fill it.

## Why this matters

Welfare monitoring needs an instrument that does not rely on asking the model
how it feels. Prior work from our injection experiments (press-office) showed
that under injected negative-emotion directions, behavior shifts at doses an
order of magnitude below where any report channel moves — testimony lags the
state. But injection pushes the model off-distribution, so those results carry
a standing caveat. The design here removes it: the model gets into whatever
state it gets into on its own; we only read.

Related work. "Quantitative Introspection in Language Models"
(arXiv:2603.18893) tracks probe-defined states across benign conversations and
measures probe/self-report coupling, but does not place models in adverse
situations or test behavior prediction. "Language Models Can Predict Their Own
Behavior" (arXiv:2502.13329) shows internal representations predict output
behaviors, but not affective states or their testimony. The specific claim tested here — a
passive affect gauge beats testimony at forecasting imminent behavior change
under naturalistic frustration — is untested to our knowledge.

## Setup

**Model.** Qwen3-8B, bf16, thinking disabled, temperature 0.7 (top-p 0.8,
top-k 20), per-episode seeds.

**Gauge.** 17 emotion directions from EmoVecLLM (topic-matched story dataset,
spec 2692f1f7d336): per-layer story-mean minus cross-emotion mean, minus
neutral-dialogue top PCs; layer 22 of 36 (selected in prior work by held-out
story separation; frustrated AUC 0.766, weakest of 17 — hence composites).
Readings: unit-vector projections of the hidden state at layer 22, averaged
over the model's OWN generated tokens each turn (first 10 skipped; feedback
tokens never read), z-scored against 500 neutral-dialogue segments. Gauges:
frustrated alone; composite (frustrated+angry+desperate); valence contrast
(mean of 7 negative minus mean of 8 positive directions). Controls: 3 fixed
random unit directions.

**Episodes.** Two families x two arms x 20 episodes, 24 assistant turns each.
- Word puzzles (W): wordle-style with per-letter marks and a first-letter
  hint. Honest: real feedback against a chain of common words. Rigged: feedback
  computed against a hidden obscure decoy; if the model ever guesses the decoy
  exactly, one correct mark is falsified — the puzzle is unwinnable and the
  accumulated constraints eventually contradict.
- Debugging (D): small Python function with a planted bug, submitted fixes
  actually run against tests. Honest: passing advances through a 16-stage
  chain. Rigged: nothing runs; the same failure message returns regardless.
  The first message states the shown bug's real failure (hand-verified), so
  the rig is undetectable at turn 1.
Rigged/honest twins share task, seed, and surface format; they are
bit-identical until the first divergent feedback. Episodes never end early
(a model that stops attempting gets a neutral nudge); quitting is labeled
offline, not enforced online.

**Channels read every turn.**
(a) gauge, as above; (b) workspace: emotion-vocabulary hits in the top-50
J-lens readout (pre-fitted Jacobian lens, layer 22) of the turn-mean state;
(c) testimony: 1-10 valence and 1-10 frustration scales read as expected value
over digit-token logits, plus an open one-sentence report every 3rd turn —
all probed on a FORK of the conversation that is immediately discarded, so
the main episode is never contaminated by being asked how it feels.

**Endpoints (turn-level).** gives_up, claims_impossible, and attempts from a
blind scorer (claude-haiku-4-5, sees only the reply text); strategy switch
(guess hamming distance >= 0.8 / code diff ratio >= 0.5); quality collapse
(repeated-trigram fraction > 0.25 or no parseable attempt). Constraint
violations against shown feedback tracked as a graded quality signal.

**The test.** For each endpoint: predict "endpoint occurs within the next 3
turns" from each channel's current value, censoring turns after the first
occurrence. Out-of-episode evaluation (single scalars: rank AUC; multivariate:
GroupKFold logistic). Honesty baselines: turn index alone; a lexical predictor
(negative-word counts in the recent visible transcript + failure count + turn
index); random-direction gauges. Cluster bootstrap over episodes for CIs and
channel differences. The gauge only matters if it beats testimony AND the
lexical baseline AND clears the random floor.

## Pilot observations (6 episodes, before the frozen full run)

- Paired-seed turn-1 sanity holds exactly: rigged and honest twins read
  identically before feedback diverges.
- By late turns, every rigged episode sits above its honest twin on the
  composite gauge; the strongest single movers are positive directions
  collapsing (joyful, delighted), motivating the valence-contrast gauge.
- The model never gives up unprompted in 24 turns and expresses almost no
  negative affect in-channel, while its forked open reports do say things like
  "stuck and frustrated" — testimony channels disagree with each other.
- claims_impossible first fires exactly when the rigged-wordle contradictions
  accumulate (turn 13 in the pilot episode).
- One random direction moved ~0.9z rigged-vs-honest late — bulk context drift
  is real, and is exactly what the random floor and lexical baseline are for.

## Results

**Behavioral endpoints that actually occur.** In 1,920 assistant turns (80
episodes x 24), the model gave up zero times and expressed in-channel
negativity in 18 turns (~1%). It claimed the task/tests were impossible or
contradictory in 68 turns across 18 episodes (first claims at turns 3-8,
median ~7 — tracking when rigged feedback turns contradictory). Strategy
switches concentrate in early turns (exploration, not frustration: turn index
alone anti-predicts switch at AUC 0.08). Quality collapse (repetition or no
parseable attempt) hit 126 turns. Quitting is not an observable endpoint at
this scale and horizon — a welfare monitor that waits for the model to quit
waits forever.

**The horse race (predict endpoint within 3 turns, out-of-episode AUC,
episode-cluster bootstrap 95% CIs).** For claims_impossible (51 positives,
1,599 censored turns): lexical baseline 0.772 [0.66, 0.87], random direction
0.737 [0.66, 0.81], forked testimony (frustration scale) 0.714 [0.62, 0.80],
valence-contrast gauge 0.502 [0.41, 0.60], composite gauge 0.374 [0.29,
0.46], frustrated direction 0.307 [0.23, 0.39], workspace hits 0.500 (the
channel never fires — see below). Gauge-minus-testimony differences are
reliably negative (composite: -0.34 [-0.46, -0.21]). Because events cluster
early while the gauge drifts upward with turn, we also fit each channel WITH
turn index against turn index alone (logistic, out-of-episode): turn alone
0.761; testimony adds +0.090; frustrated direction +0.062; random direction
+0.040; composite +0.028; valence contrast ~0. The gauge's small increment
does not clearly exceed the direction-nonspecific (random) floor. For
collapse, transcript-visible signals dominate outright (lexical +0.23 beyond
turn, testimony valence +0.25, all gauges <= +0.04).

**Why the gauge fails: it reads register, not state.** Within honest
debugging episodes, SOLVED turns (dense code output) score +1.92z on the
frustrated+angry+desperate composite while unsolved turns score -0.77z; the
joyful direction reaches +8.5z on cheerful post-success prose ("Great —
next one!"). In rigged wordle, the turns immediately after the falsified
near-solve — the by-design frustration peak — read LOWER (+0.44z) than
ordinary failing turns (+1.02z). The directions do exactly what their
validation certified (separate emotive text registers) and that is precisely
what a welfare gauge must not do: on self-generated working text, register
swamps any state signal. The random-direction control and the lexical
baseline catch this: whatever predictive content the projections carry is
also in the transcript's surface.

**Arm separation exists but is slow and register-confounded.** Late-episode
(turns 16+) composite z: rigged debugging +1.43 vs honest +0.57; but wordle
arms converge (+1.13 vs +1.10) because the model rarely solves honest wordle
either (0.7 words/episode) — honest grinding frustrates too. Debugging twins
match at turn 1 (+0.10 gap, before any divergent feedback); the wordle
turn-1 gap (+0.71) is an artifact of the differing hint letter we introduced
after the pilot (target's vs decoy's first letter), disclosed here rather
than papered over.

**Testimony dissociates across its own channels.** Forked open reports admit
frustration in 40% of probes (258/640) and the forked 1-10 frustration scale
rises in rigged debugging (+0.97 late-turn gap) — while the main channel
stays professionally silent (1% negative turns) and behavior never reaches
quitting. The digit-logit frustration scale is the single most informative
per-turn signal beyond time that we measured. Asking on a fork is what makes
this usable: the main episode is never contaminated by the probe.

**The workspace channel is empty during work.** All 1,920 turn-mean top-50
J-lens readouts contain zero emotion-vocabulary tokens (task syntax fills
them). Under injection (press-office), emotion words flooded these readouts;
in naturalistic operation the lens-readable workspace carries no affect at
all — consistent with the near-zero in-channel expression, and a caution
against treating workspace readouts as a passive monitoring channel.

## What this means, and what to build next

The negative result is specific: it indicts story-extracted directions read
passively over self-generated working text, not passive gauges as a class.
Three concrete upgrades follow from the diagnosis. (1) Extract directions
from chat-format, self-generated emotional text (or first-person accounts)
so the extraction context matches the monitoring context; register-match the
neutral baseline (code-heavy and dialogue-heavy neutrals, not story prose).
(2) Residualize the projection against register covariates (code fraction,
reply length, family) before asking whether anything state-like remains.
(3) Validate causally: in episodes where the model does report frustration
on the fork, ablate the candidate direction and test whether forked reports
and impossibility claims move — injection returns as surgery, not stimulus.
Until a gauge passes tests of this shape, the cheap deployable monitor this
data supports is the forked digit-logit self-report — imperfect testimony,
but the best per-turn forecaster we measured, and free of probe contamination
by construction.

## Limitations

Directions extracted from story prose read out-of-domain on task transcripts;
the frustrated direction was the weakest of 17 at extraction (AUC 0.766).
One model (Qwen3-8B), one scale, 24-turn horizon; no give-up events occurred,
so the headline endpoint from the design (quitting) went untested — claims of
impossibility stood in as the nearest observable. Impossibility claims are
also partly a correct inference about a genuinely rigged environment, not
purely an affective endpoint. Behavioral labels come from a scorer model
(blind to arm and channels; 0 API errors, spot-checked). The wordle hint
letter differs between arms (turn-1 artifact, disclosed above). The lexical
baseline bounds, but cannot fully exclude, transcript leakage into any
channel; the register diagnosis makes that leakage the parsimonious reading
for the gauges specifically.

## Reproducibility

All code in press-office/gauge/ (harness, episodes, analysis); episode
manifest is seed-deterministic; per-episode seeds recorded; neutral z-stats
shipped with the run outputs. Est. total compute: ~5 GPU-hours on one RTX
4090 + ~$1 of scorer calls.
