#!/usr/bin/env python3
"""research: materialize probe-clean nested restatement-dose streams.

research's leakage audit showed that the research selected dose pairs included many
source sentences that were exact substrings of the ordinary heldout rows.  The
validated shards still contain enough clean material in aggregate, but clean
shard1 alone is short of the 443,200-word dose21 target.  This repaired
materializer therefore makes the smallest change that preserves the dose law:

  * filter source pool is the research probe-clean accepted shards;
  * dose21 uses all clean shard1 material plus an exact, minimal word top-up from
    clean shard0 to reach 443,200 pair words/pass;
  * dose25 is a strict superset of dose21 and adds exactly 400,000 further clean
    shard0 pair words/pass;
  * packing, row assignment, row count, word count, pair visibility, and stream
    construction reuse the token-aware research implementation.

The resulting streams supersede research for training.  research artifacts are kept
as the leakage-finding record and must not be used for GPU training.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import collections
import json
import pathlib
import sys
import time
from typing import Any

SCRIPT_DIR = _public_path('experiments/archive/relation_learning/scripts')
sys.path.insert(0, str(SCRIPT_DIR))
import pack_materialize_dose_arm as base  # noqa: E402
import materialize_nested_restatement_doses as mat59  # noqa: E402

ROOT = pathlib.Path.cwd()
STUDY = ROOT / "experiments/archive/relation_learning"
CLEAN1 = STUDY / "data/dose_leakage_audit/accepted_dose_arm_pairs_shard1_probe_clean.jsonl"
CLEAN0 = STUDY / "data/dose_leakage_audit/accepted_dose_arm_pairs_shard0_probe_clean.jsonl"
FILTER_SUMMARY = STUDY / "data/dose_leakage_audit/accepted_shard_probe_filter_summary.json"
OUT_DEFAULT = STUDY / "data/probe_clean_nested_dose_streams"
DOSE21_ADDED = 443_200
DOSE25_TOPUP = 400_000
DOSE25_ADDED = DOSE21_ADDED + DOSE25_TOPUP


def write_json(path: pathlib.Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_jsonl_rows(rows: list[dict[str, Any]]) -> str:
    import hashlib
    h = hashlib.sha256()
    for r in rows:
        h.update(json.dumps(r, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def source_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: collections.Counter[str] = collections.Counter()
    words: collections.Counter[str] = collections.Counter()
    for r in rows:
        src = str(r.get("source", ""))
        counts[src] += 1
        words[src] += int(r.get("pair_words", 0))
    return {"counts": dict(counts), "words": dict(words)}


def exact_select_labeled(pool: list[dict[str, Any]], target_words: int, seed: int, label: str) -> list[dict[str, Any]]:
    selected = base.exact_select(pool, target_words, seed)
    got = sum(int(p["pair_words"]) for p in selected)
    if got != target_words:
        raise RuntimeError(f"{label} exact selection produced {got} != {target_words}")
    return selected


def annotate(rows: list[dict[str, Any]], selection_role: str) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        rr = dict(r)
        rr["selection_role"] = selection_role
        out.append(rr)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clean-shard1", default=str(CLEAN1))
    ap.add_argument("--clean-shard0", default=str(CLEAN0))
    ap.add_argument("--filter-summary", default=str(FILTER_SUMMARY))
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--selection-seed", type=int, default=60000)
    ap.add_argument("--write-stream", action="store_true")
    ap.add_argument("--skip-base-sha", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clean1 = base.load_accepted([pathlib.Path(args.clean_shard1)])
    clean0 = base.load_accepted([pathlib.Path(args.clean_shard0)])
    clean1_words = sum(int(p["pair_words"]) for p in clean1)
    clean0_words = sum(int(p["pair_words"]) for p in clean0)
    if clean1_words >= DOSE21_ADDED:
        dose21_core = exact_select_labeled(clean1, DOSE21_ADDED, args.selection_seed + 21, "dose21_from_clean1")
        dose21_topup0: list[dict[str, Any]] = []
    else:
        deficit = DOSE21_ADDED - clean1_words
        dose21_core = clean1
        dose21_topup0 = exact_select_labeled(clean0, deficit, args.selection_seed + 210, "dose21_minimal_clean0_topup")
    used_ids = {str(p["pair_id"]) for p in dose21_topup0}
    remaining0 = [p for p in clean0 if str(p["pair_id"]) not in used_ids]
    if sum(int(p["pair_words"]) for p in remaining0) < DOSE25_TOPUP:
        raise RuntimeError("clean shard0 remaining material is insufficient for dose25 topup")
    dose25_extra = exact_select_labeled(remaining0, DOSE25_TOPUP, args.selection_seed + 25, "dose25_extra_clean0")

    dose21_selected = annotate(dose21_core, "dose21_clean_shard1_all_or_exact") + annotate(dose21_topup0, "dose21_clean_shard0_minimal_topup")
    dose25_extra = annotate(dose25_extra, "dose25_clean_shard0_extra_topup")
    dose25_selected = dose21_selected + dose25_extra

    if sum(int(p["pair_words"]) for p in dose21_selected) != DOSE21_ADDED:
        raise RuntimeError("dose21 word target mismatch")
    if sum(int(p["pair_words"]) for p in dose25_selected) != DOSE25_ADDED:
        raise RuntimeError("dose25 word target mismatch")
    if len({str(p["pair_id"]) for p in dose25_selected}) != len(dose25_selected):
        raise RuntimeError("duplicate pair_id in dose25 selected set")
    if not {str(p["pair_id"]) for p in dose21_selected}.issubset({str(p["pair_id"]) for p in dose25_selected}):
        raise RuntimeError("dose21 not subset of dose25")

    tokenizer = base.load_tokenizer(base.TOKENIZER)
    if tokenizer is None:
        raise RuntimeError("Tokenizer is required for probe-clean materialization")

    packed21 = mat59.pack_pairs_token_aware(dose21_selected, tokenizer)
    packed_extra = mat59.pack_pairs_token_aware(dose25_extra, tokenizer)

    selected21_path = out_dir / "dose21_selected_pairs_probe_clean.jsonl"
    selected25_extra_path = out_dir / "dose25_extra_selected_pairs_probe_clean_from_shard0.jsonl"
    selected25_path = out_dir / "dose25_selected_pairs_probe_clean_superset.jsonl"
    packed21_path = out_dir / "dose21_packed_pair_rows_probe_clean.jsonl"
    packed_extra_path = out_dir / "dose25_extra_packed_pair_rows_probe_clean.jsonl"
    base.write_jsonl(selected21_path, dose21_selected)
    base.write_jsonl(selected25_extra_path, dose25_extra)
    base.write_jsonl(selected25_path, dose25_selected)
    base.write_jsonl(packed21_path, packed21)
    base.write_jsonl(packed_extra_path, packed_extra)

    base_sha = "skipped" if args.skip_base_sha else base.sha256_file(base.BASE_STREAM)
    dose21_meta = mat59.materialize_mode("dose21", packed21, [], out_dir, tokenizer, args.write_stream)
    dose25_meta = mat59.materialize_mode("dose25", packed21, packed_extra, out_dir, tokenizer, args.write_stream)

    core21_ids = [str(r["pair_ids"]) for r in []]  # placeholder to keep metadata schema simple
    rows21 = base.read_jsonl(pathlib.Path(dose21_meta["row_meta_path"])) if args.write_stream else []
    rows25 = base.read_jsonl(pathlib.Path(dose25_meta["row_meta_path"])) if args.write_stream else []
    if rows21 and rows25:
        d21 = {(int(r["pass"]), int(r.get("row_index", r.get("row_index_in_pass")))): r for r in rows21}
        d25 = {(int(r["pass"]), int(r.get("row_index", r.get("row_index_in_pass")))): r for r in rows25}
        same_core = all(k in d25 and d25[k].get("pair_ids") == r.get("pair_ids") and int(d25[k].get("pair_words", -1)) == int(r.get("pair_words", -2)) for k, r in d21.items())
    else:
        same_core = None

    filter_summary = json.loads(pathlib.Path(args.filter_summary).read_text(encoding="utf-8")) if pathlib.Path(args.filter_summary).exists() else None
    clean1_used_words = sum(int(p["pair_words"]) for p in dose21_core)
    clean0_dose21_words = sum(int(p["pair_words"]) for p in dose21_topup0)
    clean0_extra_words = sum(int(p["pair_words"]) for p in dose25_extra)
    meta = {
        "status": "PROBE_CLEAN_NESTED_RESTATEMENT_DOSES_MATERIALIZED" if args.write_stream else "PROBE_CLEAN_NESTED_RESTATEMENT_DOSES_DRYRUN",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "supersedes": "research streams; research selection had heldout/probe text hits and must not be used for training.",
        "base_stream": str(base.BASE_STREAM),
        "base_sha256": base_sha,
        "write_stream": bool(args.write_stream),
        "tokenizer": str(base.TOKENIZER),
        "max_tokens_no_special_for_replacement_rows": mat59.MAX_TOKENS,
        "clean_shard1": str(args.clean_shard1),
        "clean_shard0": str(args.clean_shard0),
        "clean_shard1_pairs": len(clean1),
        "clean_shard1_pair_words": clean1_words,
        "clean_shard0_pairs": len(clean0),
        "clean_shard0_pair_words": clean0_words,
        "selection_seed": args.selection_seed,
        "targets": {
            "dose21_added_pair_words_per_10M": DOSE21_ADDED,
            "dose25_added_pair_words_per_10M": DOSE25_ADDED,
            "dose25_extra_pair_words_from_clean_shard0_per_10M": DOSE25_TOPUP,
            "inherited_pair_words_per_10M": base.INHERITED_PAIR_WORDS,
            "dose21_total_pair_fraction": (base.INHERITED_PAIR_WORDS + DOSE21_ADDED) / base.TOTAL_WORDS_PER_PASS,
            "dose25_total_pair_fraction": (base.INHERITED_PAIR_WORDS + DOSE25_ADDED) / base.TOTAL_WORDS_PER_PASS,
        },
        "probe_leakage_filter_summary": filter_summary,
        "dose21_selection": {
            "source": "probe-clean shard1 plus minimal exact probe-clean shard0 top-up because shard1 clean material was short after leakage removal",
            "selected_pairs": len(dose21_selected),
            "selected_pair_words": sum(int(p["pair_words"]) for p in dose21_selected),
            "clean_shard1_used_pairs": len(dose21_core),
            "clean_shard1_used_pair_words": clean1_used_words,
            "clean_shard0_topup_pairs": len(dose21_topup0),
            "clean_shard0_topup_pair_words": clean0_dose21_words,
            "selected_sha256_virtual": sha256_jsonl_rows(dose21_selected),
            "packed_rows_per_pass": len(packed21),
            "pair_word_stats": base.stat([float(p["pair_words"]) for p in dose21_selected]),
            "pair_source": source_counts(dose21_selected),
            "packed_pair_word_stats": base.stat([float(r["pair_words"]) for r in packed21]),
            "packed_pair_token_stats": base.stat([float(r["pair_segment_tokens_no_special"]) for r in packed21]),
            "outputs": {"selected_pairs": str(selected21_path), "packed_pair_rows": str(packed21_path)},
        },
        "dose25_selection": {
            "source": "strict superset of probe-clean dose21 plus exact 400000-word extra from remaining clean shard0",
            "selected_pairs": len(dose25_selected),
            "selected_pair_words": sum(int(p["pair_words"]) for p in dose25_selected),
            "extra_pairs_from_clean_shard0": len(dose25_extra),
            "extra_pair_words_from_clean_shard0": clean0_extra_words,
            "selected_sha256_virtual": sha256_jsonl_rows(dose25_selected),
            "packed_core_rows_per_pass": len(packed21),
            "packed_extra_rows_per_pass": len(packed_extra),
            "extra_pair_word_stats": base.stat([float(p["pair_words"]) for p in dose25_extra]),
            "extra_pair_source": source_counts(dose25_extra),
            "packed_extra_pair_word_stats": base.stat([float(r["pair_words"]) for r in packed_extra]),
            "packed_extra_pair_token_stats": base.stat([float(r["pair_segment_tokens_no_special"]) for r in packed_extra]),
            "outputs": {"extra_selected_pairs": str(selected25_extra_path), "selected_pairs_superset": str(selected25_path), "extra_packed_pair_rows": str(packed_extra_path)},
        },
        "nesting_checks": {
            "content_superset": {str(p["pair_id"]) for p in dose21_selected}.issubset({str(p["pair_id"]) for p in dose25_selected}),
            "dose21_core_rows_identical_in_dose25_metadata": same_core,
        },
        "dose21_materialization": dose21_meta,
        "dose25_materialization": dose25_meta,
        "artifact_sha256": {
            "dose21_selected_pairs_probe_clean": base.sha256_file(selected21_path),
            "dose25_extra_selected_pairs_probe_clean_from_shard0": base.sha256_file(selected25_extra_path),
            "dose25_selected_pairs_probe_clean_superset": base.sha256_file(selected25_path),
            "dose21_packed_pair_rows_probe_clean": base.sha256_file(packed21_path),
            "dose25_extra_packed_pair_rows_probe_clean": base.sha256_file(packed_extra_path),
            "clean_shard1": base.sha256_file(pathlib.Path(args.clean_shard1)),
            "clean_shard0": base.sha256_file(pathlib.Path(args.clean_shard0)),
        },
        "scientific_note": "Probe-clean repair removes selected pairs overlapping ordinary heldout/probe text before training. The exact 21% and 25% total pair fractions and dose25 superset relation are preserved; the only change from research is that dose21 needs a small clean shard0 top-up because clean shard1 is short after leakage removal.",
        "elapsed_sec": round(time.time() - t0, 2),
    }
    meta_path = out_dir / "probe_clean_nested_dose_materialization_metadata.json"
    write_json(meta_path, meta)
    print(json.dumps({
        "status": meta["status"],
        "write_stream": meta["write_stream"],
        "dose21_selected_pairs": len(dose21_selected),
        "dose21_words": sum(int(p["pair_words"]) for p in dose21_selected),
        "dose21_clean_shard0_topup_words": clean0_dose21_words,
        "dose21_packed_rows": len(packed21),
        "dose25_selected_pairs": len(dose25_selected),
        "dose25_words": sum(int(p["pair_words"]) for p in dose25_selected),
        "dose25_extra_words": clean0_extra_words,
        "dose25_extra_packed_rows": len(packed_extra),
        "dose21_stream_sha256": dose21_meta.get("stream_sha256"),
        "dose25_stream_sha256": dose25_meta.get("stream_sha256"),
        "metadata": str(meta_path),
        "elapsed_sec": meta["elapsed_sec"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
