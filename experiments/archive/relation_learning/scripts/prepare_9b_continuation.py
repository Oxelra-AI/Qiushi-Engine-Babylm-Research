#!/usr/bin/env python3
"""research: prepare continuation prompts for Qwen3.5-9B state-use generation.

The initial batch-generation output records local `index` fields
0..N-1 for the original prompt file. Continuation runs restart their local
indices, so this script writes a remaining-prompt JSONL whose rows include
`original_prompt_index`; later validation can map each continuation output by its
local index into this continuation file and then back to the original prompt
metadata.  Do not infer coverage from exact prompt strings because distinct
selected rows can yield identical prompt strings.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import time

ROOT = pathlib.Path.cwd()
DATA = ROOT / "experiments/archive/relation_learning/data/state_use_generation"
PROMPTS = DATA / "state_use_prompts_9b.jsonl"
RAW_MAIN = DATA / "raw_outputs_9b.jsonl"
OUT_PROMPTS = DATA / "state_use_prompts_9b_remaining_after_step043.jsonl"
META = DATA / "continuation_prompt_metadata_revision_044.json"


def load_jsonl(path: pathlib.Path):
    rows = []
    if not path.exists():
        return rows
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    prompts = load_jsonl(PROMPTS)
    main_outputs = load_jsonl(RAW_MAIN)

    covered = set()
    index_out_of_range = 0
    non_integer_index = 0
    for r in main_outputs:
        idx = r.get("index")
        if not isinstance(idx, int):
            non_integer_index += 1
            continue
        if 0 <= idx < len(prompts):
            covered.add(idx)
        else:
            index_out_of_range += 1

    remaining_indices = [i for i in range(len(prompts)) if i not in covered]
    with OUT_PROMPTS.open("w") as f:
        for i in remaining_indices:
            rec = dict(prompts[i])
            rec["original_prompt_index"] = i
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    sequential_prefix = 0
    for i in range(len(prompts)):
        if i in covered:
            sequential_prefix += 1
        else:
            break

    meta = {
        "status": "CONTINUATION_PROMPTS_READY",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "all_prompts": len(prompts),
        "main_outputs": len(main_outputs),
        "covered_prompts": len(covered),
        "remaining_prompts": len(remaining_indices),
        "sequential_prefix_covered": sequential_prefix,
        "duplicate_output_indices": len(main_outputs) - len(covered) - index_out_of_range - non_integer_index,
        "index_out_of_range": index_out_of_range,
        "non_integer_index": non_integer_index,
        "remaining_prompt_path": str(OUT_PROMPTS),
        "sha256_remaining_prompts": sha256_file(OUT_PROMPTS) if OUT_PROMPTS.exists() else "",
        "min_remaining_index": min(remaining_indices) if remaining_indices else None,
        "max_remaining_index": max(remaining_indices) if remaining_indices else None,
        "note": "Continuation outputs restart local indices; validate a continuation file against this remaining-prompt JSONL, then use original_prompt_index for global identity.",
    }
    META.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
