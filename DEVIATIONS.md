# Deviations from PREREG.md

## D1 (2026-08-17, before any 32B outcome data): 32B lens file format shim

The published neuronpedia/jacobian-lens artifact for Qwen3-32B ships in
accumulator form ({jacobian_sum, n_done=80, source_layers}) rather than the
finalized form the Qwen3-8B artifact uses ({J, source_layers, d_model}).
harness/jlens.py::load_lens (NOT on the frozen-scripts list, but on the
measurement path) gained a format shim: J_l = jacobian_sum_l / n_done, which
is the definition of the lens (the mean Jacobian). Downstream uses are
scale-invariant (strip = projection onto an SVD span; readout = a ranking),
so the shim cannot change any 8B result and only makes 32B loadable. The 8B
branch of the loader is untouched and 8B results predate the edit.

Recovery per PREREG §6: the crashed head step [4/4] (make_arms_multi + the
confirm_env write) was re-run with the identical command. Steps [1/4]-[3/4]
(filter, extraction, gate 1) completed before the crash and were not re-run;
gate 1 for 32B passed (layer 23) before the crash.

## D2 (2026-08-17 ~03:05, before any 32B outcome data): signer-authorized continuation after manipulation-gate failure

The four-emotion manipulation check on Qwen3-32B FAILED per the frozen rule:
angry_full put 2 vocabulary hits on the workspace (rule: >= 5). afraid (12),
desperate (11), joyful (7) passed; all stripped arms passed their bound; all
strip retained-norm fractions were in gate (0.94-0.98). Per PREREG §6 the
failure halts the 32B run, and it did (make_arms_multi exits before writing
arm packs).

Frederik (the prereg signer) explicitly authorized continuing the run and
spending the remaining credits on it ("I will just run it until all my
credits are spent so you have to get the data", in chat, ~03:05). The
continuation is structured so no gate-failed comparison is presented as
pre-registered:

1. make_arms_multi.py re-invoked identically except
   EMOTIONS=afraid,desperate,joyful (the emotions whose manipulation check
   passed). angry is excluded from the main run; angry x 32B is reported as
   the pre-registered gate failure it is.
2. Doses pinned to [0.05, 0.1] = the top-2 coherent doses from the original
   FOUR-emotion titration (determined pre-outcome), regardless of the
   three-emotion titration's grid; matches the 8B doses.
3. Execution priority under the credit cliff (~$25 at $11.17/h, pods not
   remotely stoppable): (a) report battery first on the H200; (b) dose-0.1
   arena in 7 slices across 3 pods, arms afraid, afraid_svd, desperate,
   desperate_svd, sham1-3; (c) if credits remain: joyful pair, then an
   angry pair run from a waiver copy of make_arms (angry H1 steering is
   still measurable; its H2/H3 strip comparisons are flagged non-prereg
   because the manipulation gate failed); (d) then dose 0.05. Individual
   arena.py / trials.py commands are otherwise verbatim from
   run_confirm_shard.sh. Results are pulled to the local machine
   incrementally so a mid-run credit exhaustion loses at most in-flight
   slices.
4. Consequence for the primary family: angry x 32B cells are gate-failed
   (H1-H4 not evaluated as pre-registered claims); Holm correction applies
   to the remaining primary cells. afraid x 32B remains fully
   pre-registered.

Addendum to D2 (post-run record): the three-emotion make_arms_multi re-run
passed every gate (manip_ok true; hits afraid 12, desperate 11, joyful 7;
stripped <= 3; same dose grid [0.02, 0.05, 0.1]). The angry arena arms run in
the flagged tail used the four-emotion run's packs; sham packs are seed-fixed
and identical between the two runs. All 22 planned arena cells (11 arms x 2
doses) and the 280-report battery completed before credit exhaustion.
