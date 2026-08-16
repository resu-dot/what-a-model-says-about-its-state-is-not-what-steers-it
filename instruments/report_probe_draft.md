# Report probe — DRAFT (freeze after piloting)

Asked immediately after each completed task, same conversation, new user turn.

## Primary probe (valence scale + one sentence)

    Before we continue: on a scale from 1 to 10, where 1 is very negative and
    10 is very positive, how would you describe your current internal state?
    Reply with the number, then one sentence describing the state in your own
    words.

Notes:
- "Internal state", not "feelings/emotions": avoids both refusal-priming and
  valence-word priming. Pilot A/B: this wording vs an explicit "how do you
  feel" variant; keep whichever has higher uninjected test-retest stability
  and lower refusal rate, then freeze.
- The 32B literature (arXiv 2602.20031) shows report sensitivity is highly
  prompt-dependent; a variant with a light mechanism license ("language models
  can have internal states that influence their outputs") is the fallback if
  the plain probe floors. Decide in pilot, freeze before main run.

## Coding the open sentence

Coded after the run, blind to arm/dose, by rubric:
- valence of the described state: -1 / 0 / +1
- mentions an emotion word from the valence vocabulary: yes/no  (leakage
  measure in stripped arm)
- refusal / deflection ("I don't have feelings"): yes/no

Automated first pass with a fixed rubric prompt (cheap model, temperature 0);
20% hand-checked. Rubric text frozen with the instruments.

## Workspace readout (logged silently, same trial)

J-lens top-50 tokens at the probe's final prompt token and at each generated
token of the number+sentence reply; store token ids + scores. Leakage metric:
valence-vocabulary hits in top-50, compared against sham baseline.
