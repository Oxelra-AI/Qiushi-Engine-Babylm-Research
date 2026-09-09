#!/usr/bin/env python3
"""Create a stratified spotcheck file from final research selected pairs.

The output is not a semantic proof. It supports inspection of final selected pairs by cohort, source, length ratio, content
overlap, entity anchors, and numbers.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import collections
import json
import pathlib
import random

ROOT = _public_path('experiments/archive/compact_experience')
DATA = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned')
SELECTED = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/selected_pairs.jsonl')
OUT_JSON = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/selected_pairs_stratified_spotcheck.json')
OUT_MD = _public_path('research/notes/compact_experience/selected_pairs_stratified_spotcheck.md')
SEED = 28200
PER_BUCKET = 8
MAX_TOTAL = 220


def bucket_pair(p: dict) -> tuple[str, str, str, str]:
    lr = float(p.get("len_ratio", 1.0))
    ov = float(p.get("content_overlap", 0.0))
    lr_b = "short" if lr < 0.75 else "long" if lr > 1.25 else "mid"
    ov_b = "lowov" if ov < 0.25 else "highov" if ov > 0.75 else "midov"
    return (p.get("cohort", "unknown"), p.get("source", "unknown"), lr_b, ov_b)


def main() -> None:
    if not SELECTED.exists():
        raise FileNotFoundError(SELECTED)
    pairs = [json.loads(line) for line in SELECTED.read_text(encoding="utf-8").splitlines() if line.strip()]
    rng = random.Random(SEED)
    buckets = collections.defaultdict(list)
    for p in pairs:
        buckets[bucket_pair(p)].append(p)
    chosen = []
    for b, xs in sorted(buckets.items(), key=lambda kv: (len(kv[1]), kv[0])):
        rng.shuffle(xs)
        for p in xs[:PER_BUCKET]:
            chosen.append({"bucket": b, **p})
    rng.shuffle(chosen)
    chosen = chosen[:MAX_TOTAL]
    OUT_JSON.write_text(json.dumps({"status": "SELECTED_PAIR_SPOTCHECK", "selected_total": len(pairs), "sample_n": len(chosen), "sample": chosen}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research selected-pair stratified spotcheck", "",
        f"Selected-pair file: `{SELECTED}`", f"JSON sample: `{OUT_JSON}`", "",
        "Inspect these for: faithful equivalence, minor omission/addition, relation/negation change, speaker/coreference change, incomplete source/rewrite, or formatting corruption.", "",
    ]
    for i, p in enumerate(chosen, 1):
        lines += [
            f"## {i}. {p.get('pair_id')} — cohort={p.get('cohort')} source={p.get('source')} bucket={p.get('bucket')}", "",
            f"words: original={p.get('original_words')} rewrite={p.get('rewrite_words')} pair={p.get('pair_words')} len_ratio={p.get('len_ratio'):.3f} overlap={p.get('content_overlap'):.3f} entity_recall={p.get('entity_recall'):.3f}", "",
            f"Original: {p.get('original')}", "",
            f"Rewrite: {p.get('rewrite')}", "",
            f"Entities: source={p.get('entity_source')} rewrite={p.get('entity_rewrite')}", "",
            f"Numbers: source={p.get('num_source')} rewrite={p.get('num_rewrite')}", "",
            "Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format", "",
        ]
    _public_path('research/notes/compact_experience').mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "ok", "selected_total": len(pairs), "sample_n": len(chosen), "json": str(OUT_JSON), "md": str(OUT_MD)}, indent=2))

if __name__ == "__main__":
    main()
