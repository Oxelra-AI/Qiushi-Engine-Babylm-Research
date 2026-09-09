#!/usr/bin/env python3
"""research: Extract salvageable rejected originals from COMPACT_EXPERIENCE for restatement dose arm.

Reads the COMPACT_EXPERIENCE clean pairs and generation outputs to identify source sentences
whose rewrites were rejected for remediable reasons (copy_overlap, entity_recall,
etc.) but whose source text is adequate (not source_incomplete_or_fragment).

These originals can be re-prompted with Qwen3.5-9B and improved anti-copy prompts
to produce additional aligned restatement pairs for the dose arm.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import time

ROOT = pathlib.Path.cwd()
COMPACT_EXPERIENCE_DATA = ROOT / "experiments/archive/compact_experience/data"
CLEAN_PAIRS = COMPACT_EXPERIENCE_DATA / "qwen_clean_aligned/clean_pairs.jsonl"
SOURCE_SENTENCES = COMPACT_EXPERIENCE_DATA / "qwen_aligned/source_sentences.jsonl"
PROMPTS = COMPACT_EXPERIENCE_DATA / "qwen_aligned/rewrite_prompts.jsonl"
EXTRA_PROMPTS = COMPACT_EXPERIENCE_DATA / "qwen_clean_aligned/extra_rewrite_prompts.jsonl"
OUTPUTS = ROOT / "experiments/archive/compact_experience/training/runs/qwen_rewrites_full/outputs.jsonl"

OUT_DIR = ROOT / "experiments/archive/relation_learning/data/dose_arm_originals"


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


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()

    # 1. Load all clean (accepted) pairs to get their source sentence IDs
    print("[LOAD] Reading clean pairs...", flush=True)
    clean_pairs = read_jsonl(CLEAN_PAIRS)
    accepted_ids = set()
    for p in clean_pairs:
        rid = p.get("id") or p.get("pair_id") or ""
        accepted_ids.add(rid)
        # Also track the source sentence to match later
    print(f"  Clean pairs: {len(clean_pairs)}, unique IDs: {len(accepted_ids)}", flush=True)

    # 2. Load all generation prompt files to get the full list of originals
    all_prompts = []
    for pfile in [PROMPTS, EXTRA_PROMPTS]:
        if pfile.exists():
            rows = read_jsonl(pfile)
            all_prompts.extend(rows)
            print(f"  Loaded {len(rows)} prompts from {pfile.name}", flush=True)
        else:
            print(f"  [SKIP] {pfile} not found", flush=True)

    # Build prompt ID -> prompt mapping
    prompt_by_id = {p.get("id", ""): p for p in all_prompts}
    print(f"  Total prompts: {len(all_prompts)}, unique IDs: {len(prompt_by_id)}", flush=True)

    # 3. Identify rejected originals (prompts whose IDs are NOT in accepted set)
    rejected_prompts = []
    for pid, p in prompt_by_id.items():
        if pid not in accepted_ids:
            rejected_prompts.append(p)

    print(f"  Rejected originals: {len(rejected_prompts)}", flush=True)

    # 4. Filter out source_incomplete_or_fragment candidates
    # We can't directly know the rejection reason from the prompts alone,
    # but we can apply basic quality checks to the source sentence
    salvageable = []
    excluded_reasons = {}
    for p in rejected_prompts:
        src = p.get("source_sentence", "")
        src_words = len(src.split())
        src_name = p.get("source_name", "")

        # Quality checks on source sentence
        reason = None
        if src_words < 8:
            reason = "too_short"
        elif src_words > 60:
            reason = "too_long"
        elif src.count("xxx") > 2:
            reason = "too_many_placeholder"
        elif src.strip().startswith("@") or src.strip().startswith("#"):
            reason = "meta_prefix"
        elif len(set(src.split())) / max(1, src_words) < 0.3:
            reason = "low_diversity"
        elif not any(c.isalpha() for c in src):
            reason = "no_alpha"

        if reason:
            excluded_reasons[reason] = excluded_reasons.get(reason, 0) + 1
        else:
            salvageable.append({
                "original_id": p.get("id", ""),
                "source_sentence": src,
                "source_words": src_words,
                "source_name": src_name,
                "source_example_id": p.get("source_example_id"),
                "source_sentence_idx": p.get("source_sentence_idx"),
                "source_sha256": sha256_str(src),
            })

    print(f"  Excluded from salvageable: {excluded_reasons}", flush=True)
    print(f"  Salvageable originals: {len(salvageable)}", flush=True)

    # 5. Source register distribution
    from collections import Counter
    register_dist = Counter(s["source_name"] for s in salvageable)
    word_dist = {}
    for name in sorted(register_dist):
        ws = [s["source_words"] for s in salvageable if s["source_name"] == name]
        word_dist[name] = {
            "count": len(ws),
            "mean_words": round(sum(ws) / len(ws), 1) if ws else 0,
            "total_words": sum(ws),
        }

    # 6. Write outputs
    salvageable_path = OUT_DIR / "salvageable_rejected_originals.jsonl"
    write_jsonl(salvageable_path, salvageable)

    # 7. Build improved rewrite prompts
    PROMPT_TEMPLATE = (
        "Rewrite the following sentence to mean exactly the same thing, using "
        "different words and sentence structure.\n\n"
        "Rules:\n"
        "- Keep every named entity, proper name, speaker label, and number unchanged.\n"
        "- Do not add, remove, or change any facts.\n"
        "- Use different vocabulary and phrasing — do not copy more than 3 consecutive "
        "words from the original.\n"
        "- Output one complete rewritten sentence only. No explanation.\n\n"
        "Original: {source_sentence}\n\n"
        "Rewritten:"
    )

    prompts = []
    for s in salvageable:
        prompts.append({
            "id": f"dose_{s['original_id']}",
            "prompt": PROMPT_TEMPLATE.format(source_sentence=s["source_sentence"]),
            "source_sentence": s["source_sentence"],
            "source_words": s["source_words"],
            "source_name": s["source_name"],
            "source_example_id": s.get("source_example_id"),
        })

    prompts_path = OUT_DIR / "dose_arm_rewrite_prompts.jsonl"
    write_jsonl(prompts_path, prompts)

    # 8. Also write a 256-prompt pilot subset
    pilot = prompts[:256]
    pilot_path = OUT_DIR / "dose_arm_rewrite_prompts_pilot256.jsonl"
    write_jsonl(pilot_path, pilot)

    elapsed = round(time.time() - started, 1)
    meta = {
        "status": "DOSE_ARM_ORIGINALS_EXTRACTED",
        "created_utc": now_utc(),
        "elapsed_sec": elapsed,
        "clean_pairs_loaded": len(clean_pairs),
        "total_prompts_loaded": len(all_prompts),
        "rejected_originals": len(rejected_prompts),
        "excluded_reasons": excluded_reasons,
        "salvageable_originals": len(salvageable),
        "register_distribution": word_dist,
        "outputs": {
            "salvageable_originals": str(salvageable_path),
            "rewrite_prompts": str(prompts_path),
            "pilot_prompts": str(pilot_path),
        },
        "prompt_template": PROMPT_TEMPLATE.replace("{source_sentence}", "<source>"),
    }
    meta_path = OUT_DIR / "extraction_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(meta, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
