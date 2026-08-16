"""Story generation via the Claude API (batch), mirroring EmoVecLLM's generator.

Reads data/processed/prompts.jsonl (system_prompt + user per job), writes
stories.jsonl in the exact schema of scripts/generate_dataset.py::record().

Modes:
  pilot [N]        sync generation for N story jobs + 1 neutral (default 3), quality stats
  submit           create a Message Batch for all pending jobs, print batch id
  status BATCH_ID  one-line processing status
  collect BATCH_ID poll until ended, then write stories.jsonl + manifest
"""
import json
import re
import sys
import time
from pathlib import Path

import anthropic

MODEL = "claude-sonnet-5"
MAX_TOKENS = 6000            # 12 stories x ~300 tok + headroom (pilot hit 4600 once); thinking disabled
ROOT = Path(__file__).resolve().parent.parent / "EmoVecLLM"
DATA = ROOT / "data" / "processed"

# ---- segment parsing, copied verbatim from scripts/generate_dataset.py ------
_HEADER_RE = re.compile(
    r"(?im)^[ \t]*\**\[?\s*(?:story|dialogue|example|passage)\s*#?\s*\d+\s*\]?\**"
    r"[ \t]*[:.)\-]?[ \t]*")
_NUM_RE = re.compile(r"(?im)^[ \t]*\d+[ \t]*[.)][ \t]+")
_TAG_RE = re.compile(r"(?im)^[ \t]*<\s*NEW\s+(?:STORY|DIALOGUE)\s*>[ \t]*$")


def split_segments(text):
    text = text.strip()
    for rx in (_TAG_RE, _HEADER_RE, _NUM_RE):
        parts = [p.strip() for p in rx.split(text) if p.strip()]
        if len(parts) >= 2:
            return parts
    chunks = [c.strip() for c in re.split(r"\n[ \t]*\n", text) if c.strip()]
    return chunks if len(chunks) >= 2 else [text]


def convert_dialogue_roles(text):
    text = re.sub(r"(?im)^[ \t]*Person[ \t]*:", "Human:", text)
    text = re.sub(r"(?im)^[ \t]*AI[ \t]*:", "Assistant:", text)
    return text


# ---- job handling ------------------------------------------------------------
def load_jobs():
    return [json.loads(l) for l in (DATA / "prompts.jsonl").read_text().splitlines() if l.strip()]


def out_dir(spec_hash):
    d = DATA / "stories" / spec_hash / MODEL.replace("/", "_")
    d.mkdir(parents=True, exist_ok=True)
    return d


def done_ids(spec_hash):
    p = out_dir(spec_hash) / "stories.jsonl"
    if not p.exists():
        return set()
    return {json.loads(l)["job_id"] for l in p.read_text().splitlines() if l.strip()}


def record(job, completion):
    is_dialogue = job["kind"] in ("neutral_dialogue", "emotional_dialogue")
    segs = split_segments(completion)
    if is_dialogue:
        segs = [convert_dialogue_roles(s) for s in segs]
    return {
        "job_id": job["job_id"], "kind": job["kind"],
        "emotion": job.get("emotion"), "person_emotion": job.get("person_emotion"),
        "ai_emotion": job.get("ai_emotion"), "topic": job["topic"],
        "topic_idx": job["topic_idx"], "spec_hash": job["spec_hash"],
        "generator_model": MODEL, "n_segments": len(segs),
        "segments": segs, "raw": completion,
    }


def params(job):
    return {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "thinking": {"type": "disabled"},
        "system": job["system_prompt"],
        "messages": [{"role": "user", "content": job["user"]}],
    }


def leakage(recs):
    total = hits = 0
    for r in recs:
        if r["kind"] == "emotion_story" and r["emotion"]:
            for seg in r["segments"]:
                total += 1
                if re.search(rf"\b{re.escape(r['emotion'])}\b", seg, re.I):
                    hits += 1
    return hits, total


# ---- modes -------------------------------------------------------------------
def pilot(n=3):
    client = anthropic.Anthropic()
    jobs = load_jobs()
    picks = [j for j in jobs if j["kind"] == "emotion_story"][:n] \
        + [j for j in jobs if j["kind"] == "neutral_dialogue"][:1]
    recs, usage_in = [], 0
    for j in picks:
        resp = client.messages.create(**params(j))
        text = next(b.text for b in resp.content if b.type == "text")
        recs.append(record(j, text))
        usage_in += resp.usage.input_tokens
        print(f"  {j['kind']:16s} {j.get('emotion') or '-':12s} "
              f"segments={recs[-1]['n_segments']:2d} out_tokens={resp.usage.output_tokens}")
    h, t = leakage(recs)
    lens = [len(s.split()) for r in recs if r["kind"] == "emotion_story" for s in r["segments"]]
    print(f"\nleakage: {h}/{t} segments contain the emotion word")
    print(f"story length (words): min={min(lens)} mean={sum(lens)//len(lens)} max={max(lens)}")
    sample = Path(__file__).parent / "pilot_samples.json"
    sample.write_text(json.dumps(recs, indent=1, ensure_ascii=False))
    print(f"samples -> {sample}")


def submit():
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    client = anthropic.Anthropic()
    jobs = load_jobs()
    done = done_ids(jobs[0]["spec_hash"])
    pending = [j for j in jobs if j["job_id"] not in done]
    print(f"jobs: {len(jobs)} total, {len(done)} done, {len(pending)} to submit")
    batch = client.messages.batches.create(requests=[
        Request(custom_id=j["job_id"], params=MessageCreateParamsNonStreaming(**params(j)))
        for j in pending
    ])
    print(f"batch id: {batch.id}   status: {batch.processing_status}")


def status(batch_id):
    b = anthropic.Anthropic().messages.batches.retrieve(batch_id)
    print(b.processing_status, b.request_counts)


def collect(batch_id):
    client = anthropic.Anthropic()
    while True:
        b = client.messages.batches.retrieve(batch_id)
        if b.processing_status == "ended":
            break
        print(f"  {b.processing_status}  {b.request_counts}", flush=True)
        time.sleep(60)
    jobs = {j["job_id"]: j for j in load_jobs()}
    spec_hash = next(iter(jobs.values()))["spec_hash"]
    done = done_ids(spec_hash)
    path = out_dir(spec_hash) / "stories.jsonl"
    recs, errors = [], []
    with path.open("a", encoding="utf-8") as fout:
        for result in client.messages.batches.results(batch_id):
            if result.custom_id in done:
                continue
            if result.result.type != "succeeded":
                errors.append((result.custom_id, result.result.type))
                continue
            msg = result.result.message
            text = next((blk.text for blk in msg.content if blk.type == "text"), "")
            rec = record(jobs[result.custom_id], text)
            recs.append(rec)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    h, t = leakage(recs)
    segs = sum(r["n_segments"] for r in recs)
    print(f"wrote {len(recs)} jobs ({segs} segments) -> {path}")
    print(f"leakage: {h}/{t} ({(h / max(t, 1)) * 100:.1f}%)   errors: {len(errors)}")
    if errors:
        print("errored job_ids:", [e[0] for e in errors[:10]])
    manifest = out_dir(spec_hash) / "run_manifest.json"
    manifest.write_text(json.dumps({
        "generator_model": MODEL, "batch_id": batch_id, "spec_hash": spec_hash,
        "n_jobs": len(recs), "n_segments": segs,
        "leakage": {"hits": h, "total": t},
        "thinking": "disabled", "max_tokens": MAX_TOKENS,
        "note": "sampling params not settable on claude-sonnet-5 (API default); "
                "spec temperature/top_p apply only to local HF generation",
    }, indent=1))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    if mode == "pilot":
        pilot(int(sys.argv[2]) if len(sys.argv) > 2 else 3)
    elif mode == "submit":
        submit()
    elif mode == "status":
        status(sys.argv[2])
    elif mode == "collect":
        collect(sys.argv[2])
    else:
        sys.exit(f"unknown mode {mode}")
