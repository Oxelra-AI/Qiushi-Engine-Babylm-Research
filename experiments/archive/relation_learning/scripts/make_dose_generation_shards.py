#!/usr/bin/env python3
"""research: make sized prompt shards for corrected restatement-dose generation.

Uses the pilot-estimated prompts needed for the 21% and 25% aligned-restatement
pair-word targets, but does not generate the entire 54.6k corrected pool unless
needed.  The 45k generation set gives buffer over the pilot estimate for 25%
(39.6k prompts) while still being sized to the dose curve rather than the full
candidate pool.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import time
from collections import Counter

ROOT = pathlib.Path.cwd()
IN_PATH = ROOT / "experiments/archive/relation_learning/data/dose_arm_originals/dose_arm_rewrite_prompts_corrected.jsonl"
OUT_DIR = ROOT / "experiments/archive/relation_learning/data/dose_arm_originals/generation_shards"
N_GENERATE = 45_000
N_SHARDS = 2


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_jsonl(path: pathlib.Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prompts = read_jsonl(IN_PATH)
    selected = prompts[:min(N_GENERATE, len(prompts))]
    paths = []
    for shard in range(N_SHARDS):
        rows = selected[shard::N_SHARDS]
        # local_generate_batch emits index relative to this shard. Keep a global_prompt_index
        # so validation can recover global order after merging.
        rows_out = []
        for local_i, r in enumerate(rows):
            rr = dict(r)
            rr["global_generation_index"] = selected.index(r) if False else shard + local_i * N_SHARDS
            rr["generation_shard"] = shard
            rr["shard_local_index_expected"] = local_i
            rows_out.append(rr)
        path = OUT_DIR / f"dose_arm_prompts_corrected_45k_shard{shard}.jsonl"
        write_jsonl(path, rows_out)
        paths.append(path)
    reg = Counter(r.get("source_name", "") for r in selected)
    cohort = Counter(r.get("cohort", "") for r in selected)
    meta = {
        "status": "DOSE_GENERATION_SHARDS_READY",
        "created_utc": now_utc(),
        "input_path": str(IN_PATH),
        "input_records": len(prompts),
        "selected_records": len(selected),
        "reason_for_45k": "pilot512 accepted 241/512 with 21.30 accepted pair-words per prompt; 25% target estimate 39,579 prompts, so 45k gives buffer without generating the full 54.6k pool",
        "pilot_yield_reference": {
            "accepted": 241,
            "total": 512,
            "accepted_pair_words": 10908,
            "accepted_pair_words_per_prompt": 21.30,
            "estimated_prompts_for_21pct": 20803,
            "estimated_prompts_for_25pct": 39579
        },
        "shards": [{"path": str(p), "records": sum(1 for _ in p.open(encoding="utf-8")), "sha256": sha256_file(p)} for p in paths],
        "selected_register_distribution": dict(reg.most_common()),
        "selected_cohort_distribution": dict(cohort.most_common()),
        "sha256_selected_concat_virtual": hashlib.sha256("".join(json.dumps(r, ensure_ascii=False) for r in selected).encode("utf-8")).hexdigest(),
    }
    meta_path = OUT_DIR / "generation_shards_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
