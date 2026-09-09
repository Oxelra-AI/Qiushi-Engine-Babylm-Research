#!/usr/bin/env python3
"""research — Convert exact 10M text corpora to trainer-compatible JSONL.

Validation constraints before training:
  - independently recount final file words (authority, not internal builder counters)
  - deterministic chunking/packing with no partial selected examples
  - comparable example length distribution, update count, source mixing, LR schedule
  - record source/style provenance from the final emitted files

The BabyLM trainer's --example_jsonl path consumes examples in exact file order
with DataLoader shuffle=False. Therefore this converter performs the ordering:
fixed-size chunks are deterministically shuffled so both arms receive comparable
sequence packing/order randomness while preserving exact 10M words.
"""
from __future__ import annotations
import argparse, json, pathlib, random, statistics, hashlib
from collections import Counter

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
CORPUS_DIR = ROOT / "data/custom_corpus"
OUT_DIR = ROOT / "data/screen_jsonl"


def sha256_file(path: pathlib.Path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_words(path: pathlib.Path) -> list[str]:
    words: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            words.extend(line.split())
    return words


def chunk_words(words: list[str], words_per_example: int, source: str) -> list[dict]:
    if len(words) % words_per_example != 0:
        raise RuntimeError(f"word count {len(words)} not divisible by words_per_example={words_per_example}")
    rows = []
    for i in range(0, len(words), words_per_example):
        chunk = words[i:i + words_per_example]
        rows.append({
            "text": " ".join(chunk),
            "words": len(chunk),
            "source": source,
            "original_chunk_id": i // words_per_example,
        })
    return rows


def assign_custom_sources(rows: list[dict], manifest: dict, words_per_example: int) -> None:
    """Label custom_mix chunks by source boundaries from final emitted word offsets."""
    off_w = int(manifest["composition"]["official_retained"]["words"])
    wiki_w = int(manifest["composition"]["simple_wiki_added"]["words"])
    # Boundary labels are approximate if exact top-up added a few words; top-up is tiny and legal.
    for r in rows:
        start = r["original_chunk_id"] * words_per_example
        end = start + words_per_example
        if end <= off_w:
            src = "custom_mix::official_retained"
        elif start >= off_w and end <= off_w + wiki_w:
            src = "custom_mix::simple_wiki_added"
        elif start >= off_w + wiki_w:
            src = "custom_mix::gem_simplified_added"
        else:
            src = "custom_mix::boundary_mixed"
        r["source"] = src


def write_jsonl(path: pathlib.Path, rows: list[dict]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    lens = []
    src_words = Counter()
    with path.open("w", encoding="utf-8") as f:
        for ex_id, r in enumerate(rows):
            text = r["text"]
            wc = len(text.split())
            if wc != r["words"]:
                raise RuntimeError(f"row {ex_id} word mismatch {wc} != {r['words']}")
            obj = dict(r)
            obj["example_id"] = ex_id
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            total += wc
            lens.append(wc)
            src_words[obj["source"]] += wc
    return {
        "path": str(path),
        "rows": len(rows),
        "words": total,
        "sha256": sha256_file(path),
        "length_min": min(lens) if lens else 0,
        "length_max": max(lens) if lens else 0,
        "length_mean": statistics.mean(lens) if lens else 0,
        "source_words": dict(src_words),
        "sample_rows_without_text": [{k:v for k,v in rows[i].items() if k != "text"} | {"example_id": i} for i in range(min(5, len(rows)))]
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--words_per_example", type=int, default=160)
    p.add_argument("--target_words", type=int, default=10_000_000)
    p.add_argument("--seed", type=int, default=219)
    p.add_argument("--output_dir", default=str(OUT_DIR))
    args = p.parse_args()

    out = pathlib.Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((CORPUS_DIR / "corpus_manifest.json").read_text(encoding="utf-8"))
    arms = {
        "official_only": CORPUS_DIR / "official_only_10M.txt",
        "custom_mix": CORPUS_DIR / "custom_mix_10M.txt",
    }
    payload = {
        "status": "TEXT_TO_JSONL_SCREEN",
        "words_per_example": args.words_per_example,
        "target_words": args.target_words,
        "seed": args.seed,
        "training_comparability": {
            "jsonl_order": "deterministically shuffled chunks, consumed with trainer shuffle=False",
            "selected_words_per_arm": args.target_words,
            "rows_per_arm_expected": args.target_words // args.words_per_example,
            "batch_size_planned": 256,
            "micro_batch_size_planned": 128,
            "optimizer_steps_expected": "ceil(rows_per_arm / 256) = 245 with final partial batch",
            "lr_schedule": "identical because selected words, row count, batch size, warmup_fraction and seeds are fixed",
        },
        "arms": {},
        "source_manifest_path": str(CORPUS_DIR / "corpus_manifest.json"),
        "characterization_path": str(CORPUS_DIR / "characterization.json"),
    }

    for arm, path in arms.items():
        words = read_words(path)
        independent_count = len(words)
        if independent_count != args.target_words:
            raise RuntimeError(f"{arm}: independent word count {independent_count} != {args.target_words}")
        rows = chunk_words(words, args.words_per_example, f"{arm}::text")
        if arm == "official_only":
            for r in rows:
                r["source"] = "official_only::all_sources_mixed_text"
        else:
            assign_custom_sources(rows, manifest, args.words_per_example)
        rng = random.Random(args.seed)
        rng.shuffle(rows)
        jsonl_path = out / f"{arm}_w{args.words_per_example}_shuf_seed{args.seed}.jsonl"
        stats = write_jsonl(jsonl_path, rows)
        payload["arms"][arm] = {
            "input_text": str(path),
            "input_text_sha256": sha256_file(path),
            "independent_text_words": independent_count,
            "jsonl": stats,
        }

    # Cross-arm comparability checks
    a = payload["arms"]["official_only"]["jsonl"]
    b = payload["arms"]["custom_mix"]["jsonl"]
    payload["checks"] = {
        "both_exact_target_words": a["words"] == b["words"] == args.target_words,
        "same_rows": a["rows"] == b["rows"],
        "same_example_length": a["length_min"] == b["length_min"] == a["length_max"] == b["length_max"] == args.words_per_example,
    }
    if not all(payload["checks"].values()):
        raise RuntimeError(f"comparability check failed: {payload['checks']}")

    meta_path = out / "jsonl_manifest.json"
    meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "manifest": str(meta_path),
        "checks": payload["checks"],
        "official_jsonl": payload["arms"]["official_only"]["jsonl"],
        "custom_jsonl": payload["arms"]["custom_mix"]["jsonl"],
    }, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
