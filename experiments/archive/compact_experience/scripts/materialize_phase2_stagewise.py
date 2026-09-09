#!/usr/bin/env python3
"""research: materialize no-waste stagewise JSONL for Phase-2 sequence curriculum.

The old Phase-2 trainer fed 160-word examples at seq_len 64/128 and still counted
all 160 words as exposure. This script rewrites the exact official_100M and
mix25_100M schedules into stage-specific chunks so counted words are visible:
  stage1 (0-30M):  40-word chunks for seq_len 64
  stage2 (30-60M): 80-word chunks for seq_len 128
  stage3 (60-100M):160-word chunks for seq_len 256
Boundaries align with 160-word rows: 30M=187500 rows, 60M=375000 rows.
"""
from __future__ import annotations
import json, hashlib, time
from pathlib import Path

ROOT = Path("experiments/archive/compact_experience")
OUT = ROOT / "data/phase2_stagewise"
INPUTS = {
    "official": ROOT / "data/fixedinit_replication/official_100M.jsonl",
    "mix25": ROOT / "data/mixture/training_files/mix_25pct_100M.jsonl",
}
STAGES = [
    {"stage": 1, "row_start": 0, "row_end": 187_500, "chunk_words": 40, "seq_len": 64, "batch_size": 512, "target_words": 30_000_000},
    {"stage": 2, "row_start": 187_500, "row_end": 375_000, "chunk_words": 80, "seq_len": 128, "batch_size": 256, "target_words": 30_000_000},
    {"stage": 3, "row_start": 375_000, "row_end": 625_000, "chunk_words": 160, "seq_len": 256, "batch_size": 128, "target_words": 40_000_000},
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def chunks(words: list[str], n: int):
    assert len(words) == 160, len(words)
    assert 160 % n == 0, n
    for i in range(0, 160, n):
        yield i // n, words[i:i+n]


def materialize_one(name: str, src: Path) -> dict:
    stage_out = {s["stage"]: OUT / f"{name}_stage{s['stage']}_seq{s['seq_len']}_{s['target_words']//1_000_000}M.jsonl" for s in STAGES}
    handles = {k: p.open("w", encoding="utf-8") for k,p in stage_out.items()}
    stats = {s["stage"]: {"rows":0, "words":0, "seq_len":s["seq_len"], "batch_size":s["batch_size"], "chunk_words":s["chunk_words"], "target_words":s["target_words"], "path":str(stage_out[s["stage"]])} for s in STAGES}
    try:
        with src.open(encoding="utf-8") as f:
            for row_idx, line in enumerate(f):
                r = json.loads(line)
                w = r["text"].split()
                if int(r.get("words", len(w))) != 160 or len(w) != 160:
                    raise RuntimeError(f"{src} row {row_idx} expected 160 words got field={r.get('words')} actual={len(w)}")
                stage = None
                for s in STAGES:
                    if s["row_start"] <= row_idx < s["row_end"]:
                        stage = s
                        break
                if stage is None:
                    raise RuntimeError(f"row {row_idx} not assigned to a stage")
                st = stage["stage"]
                for chunk_id, ww in chunks(w, stage["chunk_words"]):
                    out = {
                        "text": " ".join(ww),
                        "words": len(ww),
                        "example_id": stats[st]["rows"],
                        "source": f"{r.get('source','')}::phase2_stage{st}::row{row_idx}::chunk{chunk_id}",
                        "orig_example_id": r.get("example_id"),
                        "orig_row_index": row_idx,
                        "chunk_id": chunk_id,
                    }
                    handles[st].write(json.dumps(out, ensure_ascii=False) + "\n")
                    stats[st]["rows"] += 1; stats[st]["words"] += len(ww)
        total_words = sum(v["words"] for v in stats.values())
        if total_words != 100_000_000:
            raise RuntimeError(f"{name}: total words {total_words}")
        for s in STAGES:
            st=s["stage"]
            if stats[st]["words"] != s["target_words"]:
                raise RuntimeError(f"{name} stage {st}: {stats[st]['words']} != {s['target_words']}")
    finally:
        for h in handles.values(): h.close()
    for st,p in stage_out.items():
        stats[st]["sha256"] = sha256_file(p)
    return {"input": str(src), "input_sha256": sha256_file(src), "stages": stats, "total_words": sum(v["words"] for v in stats.values())}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0=time.time(); payload={"status":"PHASE2_STAGEWISE_MATERIALIZED", "design":"no-waste sequence curriculum chunks: 40w->64, 80w->128, 160w->256; dynamic batches 512/256/128", "inputs":{}, "stages":STAGES}
    for name,src in INPUTS.items():
        print("materializing", name, src)
        payload["inputs"][name] = materialize_one(name, src)
    payload["elapsed_sec"] = round(time.time()-t0,1)
    out = OUT / "stagewise_summary.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))

if __name__ == "__main__": main()
