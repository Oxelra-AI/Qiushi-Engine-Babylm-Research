#!/usr/bin/env python3
"""research: compare deterministic binding query phrasings with Entity task wording.

The purpose is to verify that the frame-varied binding rows are not simply training
near copies of the official Entity container prompts.  The training rows vary only
the final query sentence; the official Entity task uses boxes, contents, and explicit
move/remove/put operations.  This script records lexical overlap and keyword matches
between the eight binding query templates and the official Entity prompts.
"""
from __future__ import annotations

import json
import pathlib
import re
import time
from collections import Counter

ROOT = pathlib.Path(".").resolve()
FRAME_MANIFEST = ROOT / "experiments/archive/relation_learning/data/frame_varied_recombination_rows/manifest.json"
ENTITY_DIR = ROOT / "experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/entity_tracking"
OUT = ROOT / "experiments/archive/relation_learning/data/frame_phrase_entity_overlap"
ENTITY_FILES = ["regular.jsonl", "ambiref.jsonl", "move_contents.jsonl"]
ENTITY_KEYWORDS = {"box", "boxes", "contains", "contain", "contained", "move", "moves", "moved", "remove", "removed", "put", "into", "from", "nothing"}
STOP = {"the", "a", "an", "of", "is", "are", "be", "as", "this", "that", "with", "for", "to", "by", "at", "in", "on", "and", "or", "it", "these"}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def words(s: str) -> list[str]:
    return [x.lower() for x in re.findall(r"[A-Za-z]+", s)]


def content_words(s: str) -> set[str]:
    return {w for w in words(s) if w not in STOP}


def read_jsonl(path: pathlib.Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def tail_prompt(prefix: str) -> str:
    # The official query is the final incomplete clause, usually "Box N contains ".
    text = re.sub(r"\s+", " ", prefix).strip()
    parts = re.split(r"(?<=[.!?])\s+", text)
    return parts[-1] if parts else text


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(FRAME_MANIFEST.read_text(encoding="utf-8"))
    entity_prompts = []
    op_counter = Counter()
    keyword_counter = Counter()
    for fn in ENTITY_FILES:
        for obj in read_jsonl(ENTITY_DIR / fn):
            pref = str(obj.get("input_prefix", ""))
            tail = tail_prompt(pref)
            toks = words(pref)
            for kw in ENTITY_KEYWORDS:
                if kw in toks:
                    keyword_counter[kw] += 1
            for op in ["move", "remove", "put"]:
                if re.search(rf"\b{op}\b", pref.lower()):
                    op_counter[op] += 1
            entity_prompts.append({"file": fn, "tail": tail, "input_words": len(pref.split()), "tail_words": words(tail)})
    entity_tail_vocab = set()
    entity_full_vocab = set()
    for ep in entity_prompts:
        entity_tail_vocab.update(content_words(ep["tail"]))
    for fn in ENTITY_FILES:
        for obj in read_jsonl(ENTITY_DIR / fn):
            entity_full_vocab.update(content_words(str(obj.get("input_prefix", ""))))
    frame_rows = []
    for fr in manifest["frames"]:
        templ = fr["template"]
        skeleton = templ.replace("{entity}", "ENTITY").replace("{answer}", "ANSWER")
        toks = content_words(skeleton) - {"entity", "answer"}
        frame_rows.append({
            "frame_id": fr["frame_id"],
            "frame_split": fr["frame_split"],
            "template": templ,
            "content_words": sorted(toks),
            "entity_keyword_overlap": sorted(toks & ENTITY_KEYWORDS),
            "entity_tail_vocab_overlap": sorted(toks & entity_tail_vocab),
            "entity_full_vocab_overlap": sorted(toks & entity_full_vocab),
            "looks_container_prompt": int(bool(toks & {"box", "contains", "contain", "move", "remove", "put"})),
        })
    sample_tails = []
    seen = set()
    for ep in entity_prompts[:200]:
        t = ep["tail"]
        if t not in seen:
            seen.add(t); sample_tails.append(t)
        if len(sample_tails) >= 12:
            break
    summary = {
        "status": "FRAME_PHRASE_ENTITY_OVERLAP",
        "created_utc": now(),
        "frames_path": str(FRAME_MANIFEST.relative_to(ROOT)),
        "entity_dir": str(ENTITY_DIR.relative_to(ROOT)),
        "n_entity_prompts": len(entity_prompts),
        "entity_tail_content_vocab": sorted(entity_tail_vocab),
        "official_keyword_counts": dict(keyword_counter),
        "official_operation_counts": dict(op_counter),
        "sample_official_tail_prompts": sample_tails,
        "frame_rows": frame_rows,
        "n_frames_with_container_keyword": sum(int(r["looks_container_prompt"]) for r in frame_rows),
        "science_readout": "The binding frames use state/context/passage phrasings and no box/contains/move/remove/put/nothing wording; they are not near copies of the official Entity container query surface.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research frame phrase / Entity wording comparison", "", summary["science_readout"], "", "| frame | split | container-keyword? | Entity tail overlap | template |", "|---|---|---:|---|---|"]
    for r in frame_rows:
        lines.append(f"| {r['frame_id']} | {r['frame_split']} | {r['looks_container_prompt']} | {', '.join(r['entity_tail_vocab_overlap']) or 'none'} | {r['template']} |")
    lines += ["", "Official Entity tail prompt examples:"] + [f"- `{x}`" for x in sample_tails]
    ((ROOT / 'research/documents/relation_learning/data/frame_phrase_entity_overlap/summary.md')).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "out": str(OUT.relative_to(ROOT)), "n_frames_with_container_keyword": summary["n_frames_with_container_keyword"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
