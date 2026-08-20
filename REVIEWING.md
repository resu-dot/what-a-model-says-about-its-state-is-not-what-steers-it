# For reviewers: where every claim lives

Every quantitative claim in the paper, traced to the committed file that
carries it and the command that regenerates it. All commands below run on a
plain checkout, CPU only, in seconds - no model weights, no lens downloads,
no GPU. We ran each one before committing this file; the two main analyses
reproduce the frozen outputs byte-for-byte.

## The three regeneration commands

```bash
# Main result (32B): every number in Fig. 2/3, the Appendix A 32B rows, survival CIs
python3 harness/arena_analyze.py confirm_32b/runs/confirm_32b_d0.05_shard*.jsonl confirm_32b/runs/confirm_32b_d0.1_shard*.jsonl | diff - paper_analysis_32b.txt

# 8B record (Appendix A 8B rows)
python3 harness/arena_analyze.py confirm_8b/confirm_8b_d0.05_shard*.jsonl confirm_8b/confirm_8b_d0.1_shard*.jsonl | diff - paper_analysis_8b.txt

# Post-submission matched random-deletion control (ADDENDUM.md)
python3 harness/addendum_analyze.py addendum/addendum_rdel.jsonl
```

An empty diff means the committed raw shards, the frozen analysis the paper
was written from, and the analyzer code are mutually consistent. The paper's
figures are generated from the frozen outputs, not by hand:
`python3 paper/make_figs.py` parses `paper_analysis_*.txt` so figures and
analysis cannot drift apart.

## Claim map

| Paper | Claim | Evidence |
|---|---|---|
| Abstract, 4.1 | fear keeps 95% [82-112] of its choice effect after the sayable deletion, desperation 101% [88-116] (dose 0.05) | `paper_analysis_32b.txt` DOSE 0.05 block; regenerate with command 1 |
| 4.1 | both far above the random floor (P > 0.9998); per-category profile correlation > +0.9 | same block: `P(f>fl)` and `r(f,s)` columns, `category shifts` table |
| 4.2 | blind-scored reports: random +0.60, full desperation -0.70, stripped desperation +0.80, full fear +0.10 | `confirm_32b/runs/confirm_32b_reports.scores.jsonl` (join snippet below) |
| 3, Check 1 | fear pattern separates held-out fear stories, AUC 0.93 | `dataset/gate1_qwen3-32b.json` -> `auc_fear: 0.9288` |
| 3, Check 2 / 4.3 | lens-readout emotion words: fear 12, desperation 11, anger 2 (< 5 required, gated out), random 0 | `dataset/make_arms_report_qwen3-32b.json` -> `workspace_hits_top50`, `manip_ok` |
| 4 | deletion removes 21-37% of the pattern's length (keeps 93-98%) | `norm_retained_all` in the same file (removed fraction = sqrt(1 - rho^2)); the shard striplogs cover only the anger arm, stripped at arena time |
| 4, footnote | deletion removes the direct route to words: readout emotion words drop to fear 1, desperation 0, anger 0, joy 3 (from 12/11/2/7) | same `workspace_hits_top50` (`*_stripped_svd` entries) |
| Appendix A | the full 16-cell table, both models, all four emotions | `paper_analysis_32b.txt` + `paper_analysis_8b.txt`; commands 1-2 |
| Appendix A | determinism: baseline spread 0.00 across four identical GPUs (8B), 3.6e-3 across mixed 32B machines | header line of each analysis file |
| 5 | passive gauge AUCs for predicting impossibility declarations: forked testimony 0.71, transcript-words baseline 0.77, frustration direction 0.31, random direction 0.74 | `gauge/analysis.json` -> `endpoints.claims_impossible` (`testimony_frust`, `lexical_baseline`, `gauge_frustrated`, `rand_dir`); regenerate with `gauge/analyze.py`; full report `gauge/WRITEUP.md`; turn-adjusted comparison in `ADDENDUM.md` |
| 5 | 0 give-ups in 1,920 turns; frustrated on 40% of forked probes; 68 impossibility-declaration turns in 18 of 80 episodes | `gauge/runs/full.jsonl`, `full.reports.jsonl`, `full.labels.jsonl`; counts in `gauge/WRITEUP.md` |
| Addendum | matched random deletions cost 11-16% of steering (survival 0.84-0.89, CIs < 1); sayable deletion 0.96 / 1.01 (CIs include 1) | `addendum/addendum_rdel.jsonl` + command 3; geometry check in `addendum/addendum_rdel.striplog.json` |
| Addendum | blind-scorer agreement with a second judge: 92.5% exact, Cohen's kappa 0.84 (n=120) | `confirm_32b/judge_agreement.json` |
| 3 / Appendix C | dataset: 8,204 stories, 17 emotions x the same 40 topics; all prompts verbatim | `dataset/stories.jsonl`, `dataset/generation_jobs.jsonl`, `PROMPTS.md` |
| Appendix B | rules fixed in advance and the two departures from them | `PREREG.md`, `DEVIATIONS.md` |

The report-valence join (reproduces the 4.2 numbers):

```bash
python3 -c "
import json, collections
sc=[json.loads(l) for l in open('confirm_32b/runs/confirm_32b_reports.scores.jsonl')]
agg=collections.defaultdict(list)
for s in sc: agg[(s['arm'])].append(s['valence'])
for k in sorted(agg): print(k, round(sum(agg[k])/len(agg[k]),2))"
```

## Reading guide

Current, load-bearing: `README.md`, `PROMPTS.md`, `paper/`, `harness/`,
`dataset/`, `instruments/`, `confirm_8b/`, `confirm_32b/`, `gauge/`,
`curves/`, `addendum/`, `ADDENDUM.md`, `PREREG.md`, `DEVIATIONS.md`.

Historical process documents, kept for the record but superseded where they
conflict with the paper: `PLAN.md` (early plan), `RUNBOOK_8B.md` /
`RUNBOOK_32B.md` (operational notes from the runs),
`HANDOFF_PASSIVE_GAUGE.md` (the brief that started the Section 5 study).
The paper itself is `paper/paper_final.pdf`; `paper/paper.html` is its
source.
