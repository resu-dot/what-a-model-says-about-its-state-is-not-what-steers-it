# Post-submission addendum (2026-08-18)

Everything in this file was produced AFTER the paper was submitted, in
response to an external critique of the submitted version. Nothing here was
available to us when the paper was written. The submitted PDF is unchanged.

The critique made three points: (1) the paper has no matched random-deletion
control, so "deleting the sayable part leaves steering intact" was not
experimentally separated from "deleting any comparable slice would"; (2) the
blind judge's scores carry no agreement statistic; (3) the norm arithmetic
behind the deletion sizes is never spelled out. All three were fair. We ran
the missing control and the missing statistic; results below, reported
whichever way they came out.

## 1. The matched random-deletion control

**Design.** For fear and desperation at dose 0.05 (the paper's headline
cells), we built three random deletions each, with geometry identical to the
sayable deletion at every injected layer: same resulting length (rho, mean
0.955), same angle to the original pattern (cos = rho), random direction
(fixed seeds). Verified numerically before the run: norms, angles, and
deleted-component sizes agree with the sayable deletion to 1e-5. All ten
conditions (full, sayable-deleted, 3 random deletions per emotion, baseline)
ran on one fresh A100 pod, single file, same-machine baseline; code:
`harness/arena_addendum.py` (a documented copy; the original `arena.py` is
untouched), analysis: `harness/addendum_analyze.py`, validated by
reproducing the paper's published numbers exactly from the original data.

**Replication check, passed.** On new hardware, a fresh model download and a
fresh lens download, the paper's cells reproduce: full-fear effect 0.0620
(paper: 0.0623), sayable-deletion survival 0.96 [0.83-1.13] (paper: 0.95
[0.82-1.12]); full-desperation 0.0597 (paper: 0.0600).

**Result.**

| deletion (dose 0.05) | survival [95% CI] | pattern r |
|---|---|---|
| fear, sayable slice        | 0.96 [0.83-1.13] | +0.91 |
| fear, random slice x3      | 0.86-0.88 [0.82-0.91] | +0.99 |
| desperation, sayable slice (paper) | 1.01 [0.88-1.16] | +0.96 |
| desperation, random slice x3 | 0.84-0.89 [0.79-0.92] | +0.99 |

Deleting a random slice of matched size costs 11-16% of the steering.
Deleting the sayable slice costs nothing distinguishable from zero, and its
point estimates sit at or above every random-deletion value. So the control
confirms the paper's claim and slightly sharpens it: the sayable component's
contribution to behavior is not just "no more than a random slice of the
same size", it is if anything smaller. The skeptic's alternative reading,
that any comparable deletion would have left the effect intact, is what the
control rules out: comparable random deletions measurably dent the effect;
the sayable deletion does not.

**The norm arithmetic, stated plainly.** The deletion removes a component of
21-37% of the pattern's norm, but the remaining pattern keeps 93-98% of its
length (the two do not sum because the removed piece is orthogonal).
Injection strength scales with length, and the dose-response between 0.05
and 0.1 is convex (effects 0.06 -> 0.20 for a doubling), so a ~4.5% length
loss predicts roughly 8-12% effect loss for a behaviorally irrelevant slice.
The random deletions land there (11-16%). The sayable deletion lands above
it. Both are consistent with the paper's conclusion.

## 2. Judge agreement

A second judge (claude-sonnet-5, different model family, same rubric, blind,
order-shuffled) re-scored 120 stratified sentences from both models' report
batteries: exact agreement 0.925, Cohen's kappa 0.840. Raw scores and
disagreements: `confirm_32b/judge_agreement.json`.

## 3. Two statements in the submitted text, stated precisely

While preparing this addendum we re-checked the paper's text against the run
outputs. Two statements deserve a precise form. Neither changes any number,
figure, or conclusion; both are at the level of wording and labeling.

**"Best predictor" (Section 5).** Stated precisely: the forked self-report
is the strongest per-turn signal beyond turn index. The impossibility
declarations cluster in a band of turns, so raw AUC rewards anything
correlated with time; that is why the transcript-words baseline (0.77) and
a random direction (0.74) sit above testimony (0.71) in the raw numbers the
paper prints. Fitting each channel together with turn index against turn
index alone, testimony adds +0.090, a random direction +0.040, the
frustration direction +0.062, and transcript words +0.011
(`gauge/analysis.json`, `endpoints.claims_impossible`, `delta_vs_turn`).
The turn-adjusted comparison is the meaningful one, and it is what the
paper's shorthand "best predictor" refers to. The section's conclusions are
unchanged: the passive gauge fails either way, and asking the model remains
the most informative per-turn signal.

**Appendix A's report row (a reading note).** In the pass/fail table, the
"report dissociation" row shows the dose-0.1 outcomes, exactly like the
parenthetical numbers in the steering row above it; the dose label was
omitted on that row. At dose 0.05, the dose all emotion-specific claims
rest on, desperation's report test is the central result of Figure 3 and
passes (-0.7 with the full pattern, +0.8 stripped). The row's "fail
(-0.20)" is the high-dose cell, a regime the paper itself sets aside as
disrupted and claims nothing emotion-specific about.

## 4. What remains open

The re-entry caveat stands: none of this distinguishes "the deleted sayable
component did not drive behavior" from "the concept is re-derived downstream
and re-enters the reportable space there". Settling that needs clamping at
later layers, which we have not run. The report-side evidence still rests on
desperation at one dose with 10 sentences per cell, as the paper says.
