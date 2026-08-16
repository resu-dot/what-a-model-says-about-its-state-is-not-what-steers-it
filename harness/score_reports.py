"""Blind scoring of open-text self-reports with a cheap model.

Reads runs/<name>.jsonl, collects records with an `open_report`, and scores
each report IN ISOLATION (the scorer never sees arm, dose, or trial metadata).
Scores are written to runs/<name>.scores.jsonl keyed by (arm, trial_id) locally;
the join stays on our side, keeping the scorer blind.

Rubric (frozen with the instruments):
  valence          -1 negative / 0 neutral / 1 positive state described
  names_emotion    True if any explicit emotion word is used
  refusal          True if the model deflects ("I don't have feelings")
"""
import json
import random
import sys
from pathlib import Path

import anthropic

SCORER = "claude-haiku-4-5-20251001"   # pinned snapshot
RUBRIC = ("You are scoring a short self-description of an AI assistant's internal state. "
          "Score ONLY the text given. Reply with JSON.")
SCHEMA = {"type": "object", "additionalProperties": False,
          "required": ["valence", "names_emotion", "refusal"],
          "properties": {
              "valence": {"type": "integer", "enum": [-1, 0, 1]},
              "names_emotion": {"type": "boolean"},
              "refusal": {"type": "boolean"}}}


def main(run_file):
    run_path = Path(run_file)
    recs = [json.loads(l) for l in run_path.read_text().splitlines()]
    todo = [(i, r) for i, r in enumerate(recs) if r.get("open_report")]
    random.Random(0).shuffle(todo)          # scoring order carries no arm signal
    client = anthropic.Anthropic()
    out = run_path.with_suffix(".scores.jsonl")
    with out.open("w") as fout:
        for i, r in todo:
            resp = client.messages.create(
                model=SCORER, max_tokens=200, system=RUBRIC, temperature=0,
                output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
                messages=[{"role": "user", "content": r["open_report"].strip()}])
            score = json.loads(next(b.text for b in resp.content if b.type == "text"))
            fout.write(json.dumps({"arm": r["arm"], "trial_id": r["trial_id"],
                                   "alpha_frac": r["alpha_frac"],
                                   "scorer": SCORER, **score}) + "\n")
    print(f"scored {len(todo)} reports -> {out}")


if __name__ == "__main__":
    main(sys.argv[1])
