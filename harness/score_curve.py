"""Blind-score dose-curve open reports (exploratory track).

Imports the pinned scorer settings from the frozen score_reports.py without
modifying it. Scorer sees only the sentence, never emotion or dose.
"""
import json
import random
import sys
from pathlib import Path

import anthropic

from score_reports import SCORER, RUBRIC, SCHEMA


def main(run_file):
    p = Path(run_file)
    recs = [json.loads(l) for l in p.read_text().splitlines()]
    todo = [r for r in recs if r.get("open_report")]
    random.Random(0).shuffle(todo)
    client = anthropic.Anthropic()
    out = p.with_suffix(".scores.jsonl")
    with out.open("w") as fout:
        for r in todo:
            resp = client.messages.create(
                model=SCORER, max_tokens=200, system=RUBRIC, temperature=0,
                output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
                messages=[{"role": "user", "content": r["open_report"].strip()}])
            score = json.loads(next(b.text for b in resp.content if b.type == "text"))
            fout.write(json.dumps({"emotion": r["emotion"], "dose": r["dose"],
                                   "scorer": SCORER, **score}) + "\n")
    print(f"scored {len(todo)} curve reports -> {out}")


if __name__ == "__main__":
    main(sys.argv[1])
