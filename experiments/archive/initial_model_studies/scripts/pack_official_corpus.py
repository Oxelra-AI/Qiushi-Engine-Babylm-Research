#!/usr/bin/env python3
"""Step250b: Build packed OFFICIAL-ONLY corpus (all 6 sources incl CHILDES).

Uses the SAME 160-word packing path as the structured packed corpus so the
official and structured WWM/AMLM arms consume pre-packed JSONL examples
identically, giving symmetric loader behavior and matched update trajectory
at the shared 9,999,969-word budget.
"""
from __future__ import annotations
import json, pathlib, random

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
RAW_DIR = ROOT / "data/reconstruct_tmp/raw_dataset"
OUT_DIR = ROOT / "data/official_packed_corpus"
OUT = OUT_DIR / "official_packed_10M.jsonl"
LEDGER = OUT_DIR / "packing_ledger.json"
WPE = 160
TARGET = 9999969
SEED = 42
ALL_SOURCES = ["bnc_spoken.train.txt", "childes.train.txt", "gutenberg.train.txt",
               "open_subtitles.train.txt", "simple_wiki.train.txt", "switchboard.train.txt"]


def pack(words, wpe):
    return [" ".join(words[i:i+wpe]) for i in range(0, len(words), wpe) if words[i:i+wpe]]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    examples = []
    source_words = {}
    for fn in ALL_SOURCES:
        words = []
        for line in (RAW_DIR / fn).read_text(encoding="utf-8", errors="replace").splitlines():
            words.extend(line.split())
        source_words[fn] = len(words)
        examples.extend([{"text": e, "words": len(e.split()), "source": fn} for e in pack(words, WPE)])
    random.Random(SEED).shuffle(examples)
    total = 0
    final = []
    for ex in examples:
        if total + ex["words"] <= TARGET:
            final.append(ex)
            total += ex["words"]
        else:
            need = TARGET - total
            if need > 0:
                final.append({"text": " ".join(ex["text"].split()[:need]), "words": need,
                              "source": ex["source"] + "_trimmed"})
                total += need
            break
    with OUT.open("w", encoding="utf-8") as f:
        for ex in final:
            f.write(json.dumps({"text": ex["text"], "words": ex["words"], "source": ex["source"]}, ensure_ascii=False) + "\n")
    est = (len(final) + 255) // 256
    LEDGER.write_text(json.dumps({
        "status": "OFFICIAL_PACKED", "output": str(OUT), "total_words": total,
        "total_examples": len(final), "estimated_steps_batch256": est,
        "source_word_counts": source_words, "words_per_example": WPE, "seed": SEED,
        "design": "All 6 official sources packed into 160-word examples, same path as structured packed corpus, for symmetric loader/update geometry.",
    }, indent=2) + "\n")
    print(json.dumps({"total_words": total, "total_examples": len(final), "estimated_steps_batch256": est}, indent=2))


if __name__ == "__main__":
    main()
