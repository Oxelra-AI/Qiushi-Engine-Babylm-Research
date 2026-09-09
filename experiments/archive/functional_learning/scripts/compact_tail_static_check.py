#!/usr/bin/env python3
"""research: static checks for compact-tail materializations before learner runs."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
SCRIPTS = _public_path('experiments/archive/functional_learning/scripts')
sys.path.insert(0, str(SCRIPTS))

import corrected_bridge_trainer as trainer  # noqa: E402

PARENT_PATH = _public_path('models/frontier')
DEFAULT_REF = _public_path('experiments/archive/functional_learning/data/reference_tail/reference_tail_ordinary_wordpaced.jsonl')
DEFAULT_ARMS = [
    _public_path('experiments/archive/functional_learning/data/compact_tail_materialization/pilot_labeled/compact_unspent/compact_unspent_reference_tail_wordpaced.jsonl'),
    _public_path('experiments/archive/functional_learning/data/compact_tail_materialization/pilot_labeled/compact_reinvest/compact_reinvest_reference_tail_wordpaced.jsonl'),
    _public_path('experiments/archive/functional_learning/data/compact_tail_materialization/pilot_labeled/compact_neutral/compact_neutral_reference_tail_wordpaced.jsonl'),
]
OUT = _public_path('experiments/archive/functional_learning/data/compact_tail_materialization/pilot_labeled/static_check_summary.json')


def rel(p: Path | str) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def row_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
    return trainer.row_key(row)


def summarize_rows(path: Path) -> Dict[str, Any]:
    rows = load_jsonl(path)
    keys = [row_key(r) for r in rows]
    key_counts = Counter(keys)
    duplicate_keys = [k for k, v in key_counts.items() if v > 1]
    modified = [r for r in rows if r.get("compact_modified_pair_ids")]
    topup = [r for r in rows if r.get("semantic_compact_topup")]
    word_mismatches = []
    for i, r in enumerate(rows):
        w = len(str(r.get("text", "")).split())
        if int(r.get("words", w)) != w:
            word_mismatches.append({"line": i, "declared": r.get("words"), "actual": w, "source": r.get("source")})
            if len(word_mismatches) >= 20:
                break
    return {
        "path": rel(path),
        "rows": len(rows),
        "words": sum(int(r.get("words", len(str(r.get("text", "")).split()))) for r in rows),
        "source_rows": dict(Counter(r.get("source", "") for r in rows)),
        "duplicate_row_keys": len(duplicate_keys),
        "duplicate_row_key_examples": [list(map(str, k)) for k in duplicate_keys[:5]],
        "word_count_mismatches": len(word_mismatches),
        "word_count_mismatch_examples": word_mismatches,
        "modified_rows": len(modified),
        "modified_pair_occurrences": sum(len(r.get("compact_modified_pair_ids", [])) for r in modified),
        "modified_saved_words": sum(int(r.get("compact_saved_words", 0)) for r in modified),
        "topup_rows": len(topup),
        "topup_words": sum(int(r.get("words", len(str(r.get("text", "")).split()))) for r in topup),
    }


def macro_prep_stats(path: Path, n_updates: int, words_per_update: int, seq_length: int, train_seed: int) -> List[Dict[str, Any]]:
    tokenizer = trainer.AutoTokenizer.from_pretrained(str(PARENT_PATH), local_files_only=True, use_fast=True)
    wgb = trainer.WordGroupBuilder(tokenizer)
    rows = load_jsonl(path)
    out = []
    cursor = 0
    for update_i in range(n_updates):
        batch = []
        words = 0
        while cursor < len(rows) and words < words_per_update:
            r = rows[cursor]
            batch.append(r)
            words += int(r.get("words", len(str(r.get("text", "")).split())))
            cursor += 1
        if not batch:
            break
        examples, stats = trainer.prepare_targets_for_macro(batch, tokenizer, seq_length, wgb, "ordinary_wwm", train_seed, 0.15)
        out.append({
            "update": update_i + 1,
            "rows": len(batch),
            "words": words,
            "ordinary_rows": stats["ordinary_rows"],
            "ordinary_words": stats["ordinary_words"],
            "ordinary_target_tokens": stats["ordinary_target_tokens"],
            "ordinary_zero_label_rows": stats["ordinary_zero_label_rows"],
            "relation_rows": stats["relation_rows"],
            "relation_target_tokens": stats["relation_target_tokens"],
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default=str(DEFAULT_REF))
    ap.add_argument("--arms", nargs="*", default=[str(p) for p in DEFAULT_ARMS])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--prep-updates", type=int, default=2)
    ap.add_argument("--words-per-update", type=int, default=39533)
    ap.add_argument("--seq-length", type=int, default=512)
    ap.add_argument("--train-seed", type=int, default=47047)
    args = ap.parse_args()

    paths = [Path(args.reference)] + [Path(p) for p in args.arms]
    summaries = []
    for p in paths:
        summaries.append(summarize_rows(p))
    prep = {}
    for p in paths:
        prep[rel(p)] = macro_prep_stats(p, int(args.prep_updates), int(args.words_per_update), int(args.seq_length), int(args.train_seed))
    result = {
        "status": "COMPACT_TAIL_STATIC_CHECK_DONE",
        "row_summaries": summaries,
        "prep_first_updates": prep,
        "interpretation": "Static compatibility check only; it verifies word/key accounting and target preparation before any learner comparison.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
