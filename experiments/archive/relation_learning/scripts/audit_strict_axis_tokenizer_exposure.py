#!/usr/bin/env python3
"""research: audit tokenizer-fitting text exposure for the Strict-complement axis.

The research axis was sampled from BabyLM-community/BabyLM-2026-Strict after
same-source 16-token exclusion against the pinned Strict-Small raw files and an
exact stream-row scan.  This script checks the remaining provenance question for
ordinary-fit use: whether the axis is directly visible to the shared tokenizer
fitting pool (`compliant_tokenizer`), which was trained on the compact-
view-reinvest 10M stream.  It reports exact normalized row equality and 16-token
phrase overlap between the axis rows and tokenizer-fitting rows.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import csv
import hashlib
import json
import pathlib
import re
import time
from collections import Counter, defaultdict
from typing import Any

ROOT0 = _public_path('experiments/archive/relation_learning/scripts/audit_strict_axis_tokenizer_exposure.py')
ROOT = _PUBLIC_ROOT

AXIS = ROOT / "experiments/archive/relation_learning/data/strict_complement_ngram_axis/strict_complement_ngram_axis_3000_rows.jsonl"
TOKENIZER_META = ROOT / "experiments/archive/frontier_consolidation/data/compliant_tokenizer/tokenizer_metadata.json"
TOKENIZER_POOL = ROOT / "experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"
OUT_DIR = ROOT / "experiments/archive/relation_learning/data/strict_axis_tokenizer_exposure"
N = 16
WORD_RE = re.compile(r"\S+")


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def norm_text(s: str) -> str:
    return " ".join(str(s).lower().split())


def tokens(s: str) -> list[str]:
    return WORD_RE.findall(norm_text(s))


def h_ngram(toks: list[str], i: int, n: int) -> str:
    return hashlib.blake2b("\x1f".join(toks[i:i+n]).encode("utf-8"), digest_size=12).hexdigest()


def read_jsonl(path: pathlib.Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    axis_rows = []
    axis_exact: dict[str, list[int]] = defaultdict(list)
    axis_ngram_to_rows: dict[str, set[int]] = defaultdict(set)
    axis_ngram_examples: dict[str, str] = {}
    for idx, obj in enumerate(read_jsonl(AXIS)):
        exid = int(obj.get("example_id", idx))
        src = str(obj.get("source", ""))
        text = str(obj["text"])
        nt = norm_text(text)
        toks = tokens(text)
        axis_rows.append({"axis_index": idx, "example_id": exid, "source": src, "norm_text": nt, "tokens": toks})
        axis_exact[nt].append(idx)
        for i in range(0, max(0, len(toks) - N + 1)):
            hh = h_ngram(toks, i, N)
            axis_ngram_to_rows[hh].add(idx)
            if hh not in axis_ngram_examples:
                axis_ngram_examples[hh] = " ".join(toks[i:i+N])

    exact_hits = []
    ngram_hit_rows: dict[int, dict[str, Any]] = {}
    ngram_occurrences = 0
    tokenizer_rows = 0
    tokenizer_words = 0
    source_rows = Counter()
    tokenizer_pool_path = TOKENIZER_POOL
    meta = json.loads(TOKENIZER_META.read_text(encoding="utf-8"))
    meta_pool = pathlib.Path(str(meta.get("training_data", {}).get("pool", "")))
    if str(meta_pool):
        candidate = ROOT / meta_pool if not meta_pool.is_absolute() else meta_pool
        if candidate.exists():
            tokenizer_pool_path = candidate

    for j, obj in enumerate(read_jsonl(tokenizer_pool_path)):
        text = str(obj.get("text", ""))
        src = str(obj.get("source", obj.get("origin_source", "")))
        nt = norm_text(text)
        toks = tokens(text)
        tokenizer_rows += 1
        tokenizer_words += int(obj.get("words", len(toks)))
        source_rows[src] += 1
        if nt in axis_exact:
            for ai in axis_exact[nt]:
                exact_hits.append({
                    "axis_index": ai,
                    "axis_example_id": axis_rows[ai]["example_id"],
                    "axis_source": axis_rows[ai]["source"],
                    "tokenizer_pool_row_index": j,
                    "tokenizer_pool_source": src,
                    "hit_type": "exact_normalized_row_equality",
                })
        seen_in_row: set[tuple[int, str]] = set()
        for i in range(0, max(0, len(toks) - N + 1)):
            hh = h_ngram(toks, i, N)
            if hh not in axis_ngram_to_rows:
                continue
            phrase = " ".join(toks[i:i+N])
            # verify phrase equality because hashes are short by design
            if phrase != axis_ngram_examples.get(hh):
                continue
            for ai in axis_ngram_to_rows[hh]:
                key = (ai, hh)
                if key in seen_in_row:
                    continue
                seen_in_row.add(key)
                ngram_occurrences += 1
                rec = ngram_hit_rows.setdefault(ai, {
                    "axis_index": ai,
                    "axis_example_id": axis_rows[ai]["example_id"],
                    "axis_source": axis_rows[ai]["source"],
                    "axis_text_prefix": axis_rows[ai]["norm_text"][:220],
                    "n_matching_ngram_hashes": 0,
                    "n_tokenizer_pool_rows_hit": 0,
                    "tokenizer_pool_sources": Counter(),
                    "example_phrases": [],
                })
                rec["n_matching_ngram_hashes"] += 1
                rec["n_tokenizer_pool_rows_hit"] += 1
                rec["tokenizer_pool_sources"][src] += 1
                if len(rec["example_phrases"]) < 5:
                    rec["example_phrases"].append({
                        "phrase": phrase,
                        "tokenizer_pool_row_index": j,
                        "tokenizer_pool_source": src,
                    })

    ngram_rows = []
    for rec in ngram_hit_rows.values():
        rec = dict(rec)
        rec["tokenizer_pool_sources"] = json.dumps(dict(rec["tokenizer_pool_sources"]), sort_keys=True)
        rec["example_phrases"] = json.dumps(rec["example_phrases"], ensure_ascii=False)
        ngram_rows.append(rec)
    ngram_rows.sort(key=lambda r: (-int(r["n_matching_ngram_hashes"]), int(r["axis_index"])))

    with (OUT_DIR / "exact_row_hits.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["axis_index", "axis_example_id", "axis_source", "tokenizer_pool_row_index", "tokenizer_pool_source", "hit_type"])
        w.writeheader(); w.writerows(exact_hits)
    with (OUT_DIR / "ngram16_hits_by_axis_row.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["axis_index", "axis_example_id", "axis_source", "axis_text_prefix", "n_matching_ngram_hashes", "n_tokenizer_pool_rows_hit", "tokenizer_pool_sources", "example_phrases"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(ngram_rows)

    summary = {
        "status": "STRICT_AXIS_TOKENIZER_EXPOSURE_AUDIT_DONE",
        "created_utc": now(),
        "elapsed_sec": round(time.time() - t0, 2),
        "axis": {"path": rel(AXIS), "rows": len(axis_rows), "ngram_n": N},
        "tokenizer_metadata": rel(TOKENIZER_META),
        "tokenizer_pool": {"path": rel(tokenizer_pool_path), "rows": tokenizer_rows, "words": tokenizer_words, "source_rows": dict(source_rows)},
        "exact_normalized_row_equality": {"hit_occurrences": len(exact_hits), "unique_axis_rows_hit": len({r["axis_index"] for r in exact_hits})},
        "ngram16_overlap": {"hit_occurrences": ngram_occurrences, "unique_axis_rows_hit": len(ngram_hit_rows)},
        "outputs": {
            "exact_row_hits_csv": rel(OUT_DIR / "exact_row_hits.csv"),
            "ngram16_hits_by_axis_row_csv": rel(OUT_DIR / "ngram16_hits_by_axis_row.csv"),
            "summary_json": rel(OUT_DIR / "summary.json"),
        },
        "scientific_interpretation": (
            "Exact row hits would make the Strict-complement axis trained text for the shared tokenizer. "
            "Sixteen-token phrase hits indicate residual phrase exposure through tokenizer fitting/generation; they affect absolute fit wording, "
            "but within-family model deltas remain interpretable because all compared arms in these families use the same tokenizer."
        ),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
