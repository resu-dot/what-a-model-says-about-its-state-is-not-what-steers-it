"""Drop story segments that name their emotion (lemma + word-family aware).

For the four INJECTED emotions the whole word family from strip.py's report
vocabulary is filtered (review R10: Sonnet obeyed the literal-word ban but used
synonyms in 8-27% of segments; filtering must be symmetric across the compared
arms). Other emotions keep the exact-word/lemma rule. In-place, idempotent.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from strip import FEAR_WORDS, ANGER_WORDS, DESPERATION_WORDS, POSITIVE

LEMMAS = {"desperate": r"\bdesperat\w*", "angry": r"\b(?:angry|anger|angrily)\b",
          "joyful": r"\b(?:joy|joyful|joyfully|joyous)\b",
          "anxious": r"\b(?:anxious|anxiously|anxiety)\b",
          # 'content' is an ordinary noun ("contents of the box") -- only filter
          # clear emotional uses
          "content": r"\b(?:contented|contentment|contentedly)\b|\bfe(?:el|els|elt|eling)\s+content\b"}

JOY_FAMILY = ["joy", "joyful", "joyfully", "joyous", "happy", "happiness", "delighted",
              "delight", "elated", "gleeful", "cheerful", "ecstatic", "thrilled"]
FAMILIES = {"afraid": FEAR_WORDS + ["afraid", "fear", "fearing"],
            "angry": ANGER_WORDS + ["angry", "anger", "angrily"],
            "desperate": DESPERATION_WORDS,
            "joyful": JOY_FAMILY}

def pattern(emo):
    if emo in FAMILIES:
        alts = "|".join(re.escape(w) for w in FAMILIES[emo])
        return rf"\b(?:{alts})\b"
    return LEMMAS.get(emo, rf"\b{re.escape(emo)}\w*")

def main(path):
    rows = [json.loads(l) for l in open(path)]
    dropped = 0
    for r in rows:
        if r["kind"] == "emotion_story" and r.get("emotion"):
            keep = [s for s in r["segments"] if not re.search(pattern(r["emotion"]), s, re.I)]
            dropped += len(r["segments"]) - len(keep)
            r["segments"], r["n_segments"] = keep, len(keep)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"dropped {dropped} leaky segments")

if __name__ == "__main__":
    main(sys.argv[1])
