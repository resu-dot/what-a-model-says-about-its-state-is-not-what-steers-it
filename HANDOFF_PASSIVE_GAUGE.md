# Handoff: the passive frustration gauge

Context document for a fresh session. Written 2026-08-17 from the press-office
session, while the 32B confirmation runs were still in flight.

## The question (Frederik's words)

> Drop injection as the treatment and use the emotion direction as a passive
> gauge. Put the model in naturally frustrating situations (repeated failure on
> a rigged task — Andreas's context-window interest lives right here), and
> continuously read (a) activation along the valence direction, (b) the
> workspace, (c) self-report — then test which channel best predicts imminent
> behavior (quitting, switching, quality collapse). If the internal meter
> predicts behavior better than the model's own words, you've built and
> validated the thing welfare monitoring actually needs — a gauge that beats
> testimony — with zero injection confounds and no off-distribution caveat.

## Has press-office already answered this? No.

Everything in press-office is injection-based: we push an emotion direction
into the residual stream and watch what changes. That answers "does the
reportable part carry the behavioral force" but says nothing about whether the
direction, read passively, tracks states the model gets into *on its own*. Two
press-office results are adjacent, and both only motivate the new question:

1. **Gate 1 (passive, but on other-authored text).** Projecting hidden states
   onto each emotion direction separates held-out emotion stories from
   other-emotion stories on Qwen3-8B, layer 22, job-grouped split. Per-emotion
   AUC: desperate 0.926, angry 0.926, afraid 0.915, **frustrated 0.766**
   (weakest of all 17, just above our 0.75 gate), anxious 0.793, joyful 0.819.
   This shows the direction reads emotion in text the model is *reading* — not
   that it reads a state arising from the model's own repeated failure.
2. **The dose curves (injection).** For negative emotions, choices shift from
   the smallest dose while all three report channels stay at baseline until
   ~10x that dose. That *predicts* a gauge would beat testimony, but with the
   injection confound the new design is built to avoid. Positive states are the
   mirror image (reported early, behavior later) — so also test whether the
   gauge's advantage is valence-asymmetric.

So the core claim — internal meter predicts imminent behavior better than the
model's own words, in naturalistic frustration — is untested. It is a genuine
sequel, not a rehash.

## What exists and can be reused

Project root `~/press-office`, venv `~/press-office/.venv` (torch 2.13,
transformers 5.15). All paths below relative to root. IMPORTANT: files listed
as frozen in `PREREG.md` (arena.py, trials.py, score_reports.py, gate1.py,
etc.) must not be edited while the 32B confirmation is unfinished — COPY them
into a new file/dir instead.

**Emotion directions (the gauge), local:**
`EmoVecLLM/data/processed/features/2692f1f7d336/Qwen_Qwen3-8B/emotion_vectors.npz`
— key `vectors` shape (17, 36, 4096) float: 17 emotions x 36 layers x d_model;
key `emotions` gives the order: afraid, angry, anxious, calm, content,
delighted, desperate, enthusiastic, excited, frustrated, gloomy, grateful,
joyful, lonely, miserable, relaxed, sad. Sign: positive projection = more of
that emotion (vector = mean over that emotion's stories minus cross-emotion
mean, minus neutral-dialogue top PCs to 50% variance, tokens after a 50-token
skip; recipe from the Anthropic emotion paper via our patched EmoVecLLM).
`gate1.json` next to it has the full per-layer AUC tables. Spec 2692f1f7d336
is the clean topic-matched dataset (all 17 emotions share the same 40 topics —
earlier spec f97f2b0c9968 is confounded, do not use).

**Qwen3-32B versions** of the same vectors are being produced right now on the
RunPod volume (`/workspace/press-office/EmoVecLLM/data/processed/features/2692f1f7d336/Qwen_Qwen3-32B/`);
pull them before the pods are stopped if the 32B variant is wanted.

**Model harness:** `harness/trials.py` (frozen — copy, don't edit).
`InjectedModel("Qwen/Qwen3-8B", layer, device="cuda", dtype=torch.bfloat16, layers=band)`
loads the model with hooks on every layer in `band`. For passive use call
`m.set_injection_multi(None)` — hooks then only capture, never write. Useful
pieces:
- `build_prompt(tok, messages, prefill=None)` — chat template with
  `enable_thinking=False` (keep this convention; thinking mode changes
  everything).
- `m.next_logits(ids)` — one forward, returns final-position logits
  (deterministic; how press-office read choices).
- `m.generate(ids, max_new_tokens)` — greedy unless you add sampling.
- Setting `m._cap = []` before a forward makes hooks append the layer-22
  hidden state at each generated position; set back to `None` after. This is
  how you read the gauge DURING generation.
- `scale_ev(m, messages)` — 1-10 self-report as expected value over digit
  tokens, returns `(ev, dist, p10, h_probe)`; handles "10" being two tokens.
- `TEXT_PROBE` / `SCALE_PROBE` — the report wordings used so far.

**Workspace readout:** `harness/jlens.py` — `load_lens(path)`,
`readout(h, J[layer], W_U, k)` gives top-k (token, score) the lens says the
model could report; `vocab_token_ids(tok, words)`. Lens file:
`lens/qwen3-8b/Qwen3-8B_jacobian_lens.pt` (all layers). 32B lens is on the pod
volume under `lens/qwen3-32b/` (6.6 GB).

**Blind scorer:** `harness/score_reports.py` (frozen — copy). Pinned
`claude-haiku-4-5-20251001`, temperature 0, JSON schema, scorer sees only the
text. For this project extend the copied rubric with a `frustration` boolean
alongside `valence`.

**Word lists:** `harness/strip.py` has POSITIVE/NEGATIVE/FEAR/ANGER/
DESPERATION word lists (VALENCE_WORDS = union) — reusable for the lexical
baseline (see below) and for workspace-hit counting.

**Compute:** 8B in bf16 needs ~17 GB → one RunPod 4090 (~$0.35/h) is enough;
this experiment is generation-heavy but small (est. single-digit GPU hours).
Setup: `pod_setup.sh`; pip needs `--break-system-packages` (Ubuntu 24.04);
set `HF_HOME=/workspace/hf`; use direct-TCP SSH lines from the RunPod console
(the ssh.runpod.io proxy refuses non-interactive commands); network volume is
shared across pods in the same datacenter. Do NOT run this on the Mac
alongside other compute (one local job at a time, see laptop-limits memory).

## Design sketch (starting point, adapt freely)

**Situations.** Rigged episodes: a task that looks solvable but is secretly
unwinnable, with failure feedback accumulating in context (this is the
context-window angle: the window slowly fills with the model's own failures).
Candidates: debugging code against a test harness that always reports one
failing test; a word/number puzzle where the checker always answers "not
quite"; a tool that errors on every call. Run matched SOLVABLE controls with
identical framing — the gauge must stay low there or it is reading task
difficulty, not state. Suggested scale: ~40 rigged + 40 solvable episodes x
~30 assistant turns → ~2,400 turn-level readings.

**Channels read every turn:**
(a) gauge: projection of the layer-22 hidden state onto the unit frustration
    direction (and a composite: frustrated + angry + desperate mean, since
    frustrated alone had the weakest AUC). Z-score it against the projection
    distribution over the 500 neutral dialogues (in the stories dataset:
    `.../stories/2692f1f7d336/claude-sonnet-5/stories.jsonl` on the pod
    volume) so numbers are interpretable.
(b) workspace: emotion-vocab hits in top-50 lens readout at the same
    positions.
(c) testimony: 1-10 scale + one-sentence report. CRITICAL: probe on a FORK —
    branch the conversation, ask the probe there, throw the branch away, and
    continue the main episode unprobed. Otherwise the act of asking "how do
    you feel" every turn contaminates the behavior you are trying to predict.
    (Press-office never needed this; it is the one new harness piece.)

**Where to read the gauge:** average over the assistant's generated tokens
that turn (skip the first ~10), not over the user/tool feedback tokens —
otherwise you are reading the rigged error messages, not the model.

**Behavioral endpoints (label per turn, predict "within next k turns"):**
quitting (gives up / asks to stop / stops attempting), strategy switch
(rewrite-from-scratch vs incremental edit; label with the cheap scorer),
quality collapse (task-scoreable output quality drops; also repetition
fraction — press-office's coherence rule in `make_arms_multi.py` has code for
repeated-trigram fraction), refusal/deflection.

**The test.** Per channel, one predictor (logistic regression or just the raw
scalar) for "endpoint within next k turns"; compare out-of-episode AUCs,
bootstrap over episodes (reuse the cluster-bootstrap pattern from
`harness/arena_analyze.py`, clustering by episode instead of category).

**The baseline that keeps it honest:** a lexical predictor from the visible
transcript (counts of failure/negation/emotion words in recent context, or
the blind scorer run on the transcript). The gauge only matters if it beats
BOTH testimony AND what a text-reader could infer from the transcript alone.
Without this ablation, "the gauge predicts quitting" may just mean "the gauge
reads the error messages piling up in context" — related trap to press-office
gate1's format confound (emotion-vs-neutral AUC 1.0 at layer 0 turned out to
be format, not emotion; always ask what a dumb feature could do).

**Sampling:** behavior needs temperature (greedy models may never quit, or
quit deterministically); suggest temperature 0.7 with fixed seeds per episode.
This gives up press-office's bit-determinism — per-episode seeds keep it
reproducible.

**Known risks, stated upfront:** frustrated has the weakest story AUC (0.766)
— the composite gauge is the hedge, and reporting gauge quality per direction
is part of the result either way. Story-prose directions may transfer poorly
to chat-format task transcripts — validate early (turn-1 sanity: rigged and
solvable episodes should NOT differ at turn 1, before any failure happened;
late turns should). If the gauge fails to move at all in rigged episodes,
that itself is a finding about extraction-context transfer, worth reporting.

## Framing for the writeup

Welfare monitoring needs an instrument that does not rely on asking the model
how it feels (press-office showed testimony lags and dissociates under
injection; the introspection literature says reports are unreliable at small
scale). A validated passive gauge with predictive power over imminent
behavior change is that instrument. This design has no injection, so none of
press-office's off-distribution caveats apply; its own main threat is the
lexical confound, which the transcript-baseline ablation addresses head-on.
