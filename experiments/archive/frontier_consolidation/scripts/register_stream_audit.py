#!/usr/bin/env python3
"""research: file-only stream audit for register-displacement pair.

Audits actual 10M/100M JSONL streams for the primary position-matched direct
register pair.  Checks row counts, word counts, 10x repetition, clean equality on
unselected rows, changed row positions, identical admitted FineWeb text multiset,
source mixtures, tokenizer fertility on changed rows, and replacement-row mapping.

No model loading, training, evaluation, GPU, GlobalPIQA, SuperGLUE, AoA, upload,
or leaderboard.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import collections, csv, hashlib, json, math, pathlib, statistics, time
from typing import Any


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT

ROOT = find_user_root()
WS = ROOT / "experiments/archive/frontier_consolidation"
OUT = WS / "data/register_stream_audit"
POOL = WS / "data/register_position_matched_direct_pools"
CLEAN10 = WS / "data/dose_2p64x_rowholdout_pools/cleanqwen_lengthmatched_dose2p64x_10M.jsonl"
TOKENIZER_DIR = WS / "data/compliant_tokenizer"
TOKENIZER_JSON = TOKENIZER_DIR / "tokenizer.json"
META = POOL / "register_position_matched_direct_pools_metadata.json"
ARMS = {
    "childsub_posmatched": {
        "file10": POOL / "regpos_childsub_posmatched_samefw_quarter_10M.jsonl",
        "file100": POOL / "regpos_childsub_posmatched_samefw_quarter_100M.jsonl",
    },
    "adult_posmatched": {
        "file10": POOL / "regpos_adult_posmatched_samefw_quarter_10M.jsonl",
        "file100": POOL / "regpos_adult_posmatched_samefw_quarter_100M.jsonl",
    },
}
EXPECTED_ROWS10 = 65_313
EXPECTED_WORDS10 = 10_000_000
EXPECTED_ROWS100 = 653_130
EXPECTED_WORDS100 = 100_000_000
EXPECTED_TOKENIZER_SHA = "91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def row_words(row: dict[str, Any]) -> int:
    return int(row.get("words", len(str(row.get("text", "")).split())))


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def multiset_hash(items: list[str]) -> str:
    h = hashlib.sha256()
    for item in sorted(items):
        h.update(item.encode("utf-8")); h.update(b"\0")
    return h.hexdigest()


def stream100_check(path100: pathlib.Path, base10: list[dict[str, Any]]) -> dict[str, Any]:
    rows = 0; words = 0; mismatches = []
    n = len(base10)
    with path100.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            rows += 1; words += row_words(r)
            if i < n * 10:
                b = base10[i % n]
                if r.get("text") != b.get("text") or row_words(r) != row_words(b):
                    if len(mismatches) < 20:
                        mismatches.append({"row100": i, "row10": i % n, "words100": row_words(r), "words10": row_words(b), "text100_sha": text_hash(str(r.get("text", ""))), "text10_sha": text_hash(str(b.get("text", "")))})
    return {"rows": rows, "words": words, "rows_match": rows == EXPECTED_ROWS100, "words_match": words == EXPECTED_WORDS100, "tenfold_ordered_repetition": rows == n * 10 and not mismatches, "mismatch_examples": mismatches}


def source_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    c = collections.Counter()
    for r in rows:
        src = str(r.get("source", ""))
        c[src] += row_words(r)
    return dict(sorted(c.items()))


def component_counts_from_meta(arm_meta: dict[str, Any]) -> dict[str, int]:
    return dict((arm_meta.get("displaced_clean_summary") or {}).get("component_words") or {})


def load_tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(TOKENIZER_JSON))


def token_len(tok: Any, text: str) -> int:
    return len(tok.encode(text).ids)


def summarize_numeric(xs: list[float]) -> dict[str, Any]:
    return {
        "n": len(xs),
        "mean": statistics.mean(xs) if xs else None,
        "median": statistics.median(xs) if xs else None,
        "min": min(xs) if xs else None,
        "max": max(xs) if xs else None,
        "stdev": statistics.stdev(xs) if len(xs) >= 2 else 0.0 if len(xs) == 1 else None,
    }


def changed_indices(rows: list[dict[str, Any]]) -> list[int]:
    out = []
    for i, r in enumerate(rows):
        if r.get("regpos_row_kind") == "position_matched_direct_fineweb_replacement":
            out.append(i)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.loads(META.read_text(encoding="utf-8"))
    arm_meta = {a["arm_key"]: a for a in meta.get("arm_results", [])}
    tok_sha = sha256_file(TOKENIZER_JSON)
    tok = load_tokenizer()
    clean = read_jsonl(CLEAN10)
    clean_word_total = sum(row_words(r) for r in clean)

    arm_rows10 = {name: read_jsonl(cfg["file10"]) for name, cfg in ARMS.items()}
    audit_rows: list[dict[str, Any]] = []
    changed_sets: dict[str, set[int]] = {}
    fineweb_hashes: dict[str, list[str]] = {}

    for name, rows in arm_rows10.items():
        indices = changed_indices(rows)
        changed_sets[name] = set(indices)
        changed = [rows[i] for i in indices]
        unchanged_mismatch = []
        length_mismatch = []
        clean_replacement_mismatch = []
        token_diffs = []
        fertility_fine = []
        fertility_clean_replaced = []
        for i, (r, c) in enumerate(zip(rows, clean)):
            if i in changed_sets[name]:
                if row_words(r) != row_words(c):
                    length_mismatch.append(i)
                repl = r.get("regpos_replaced_clean_row_index")
                if repl != i:
                    clean_replacement_mismatch.append({"row": i, "replaced_clean_row_index": repl})
                tl_f = token_len(tok, str(r.get("text", "")))
                tl_c = token_len(tok, str(c.get("text", "")))
                token_diffs.append(tl_f - tl_c)
                if row_words(r):
                    fertility_fine.append(tl_f / row_words(r))
                if row_words(c):
                    fertility_clean_replaced.append(tl_c / row_words(c))
            else:
                if r.get("text") != c.get("text") or row_words(r) != row_words(c):
                    if len(unchanged_mismatch) < 20:
                        unchanged_mismatch.append({"row": i, "arm_text_sha": text_hash(str(r.get("text", ""))), "clean_text_sha": text_hash(str(c.get("text", ""))), "arm_words": row_words(r), "clean_words": row_words(c), "arm_source": r.get("source"), "clean_source": c.get("source")})
        fine_hash = [text_hash(str(r.get("text", ""))) for r in changed]
        fineweb_hashes[name] = fine_hash
        am = arm_meta[name]
        ds = am.get("displaced_clean_summary") or {}
        audit_rows.append({
            "arm": name,
            "file10": rel(ARMS[name]["file10"]),
            "file100": rel(ARMS[name]["file100"]),
            "sha10_actual": sha256_file(ARMS[name]["file10"]),
            "sha10_metadata": am.get("sha_10m"),
            "sha100_actual": sha256_file(ARMS[name]["file100"]),
            "sha100_metadata": am.get("sha_100m"),
            "rows10": len(rows),
            "words10": sum(row_words(r) for r in rows),
            "rows10_match": len(rows) == EXPECTED_ROWS10,
            "words10_match": sum(row_words(r) for r in rows) == EXPECTED_WORDS10,
            "changed_rows": len(indices),
            "changed_words": sum(row_words(r) for r in changed),
            "rho": am.get("rho"),
            "row_index_mean": ds.get("row_index_mean"),
            "row_index_sd": ds.get("row_index_sd"),
            "displaced_child_sub_fraction": ds.get("child_sub_fraction"),
            "displaced_adult_fraction": ds.get("adult_gut_simple_fraction"),
            "unchanged_mismatch_count_visible": len(unchanged_mismatch),
            "unchanged_mismatch_examples_json": json.dumps(unchanged_mismatch, ensure_ascii=False),
            "changed_word_length_mismatch_count": len(length_mismatch),
            "changed_word_length_mismatch_examples": json.dumps(length_mismatch[:20]),
            "replaced_clean_row_index_mismatch_count": len(clean_replacement_mismatch),
            "replaced_clean_row_index_mismatch_examples": json.dumps(clean_replacement_mismatch[:20]),
            "fineweb_text_multiset_hash_actual": multiset_hash(fine_hash),
            "fineweb_text_multiset_hash_metadata": am.get("fineweb_text_multiset_hash"),
            "token_diff_fine_minus_replaced_clean_mean": summarize_numeric(token_diffs)["mean"],
            "token_diff_fine_minus_replaced_clean_min": summarize_numeric(token_diffs)["min"],
            "token_diff_fine_minus_replaced_clean_max": summarize_numeric(token_diffs)["max"],
            "fineweb_changed_token_fertility_mean": summarize_numeric(fertility_fine)["mean"],
            "replaced_clean_token_fertility_mean": summarize_numeric(fertility_clean_replaced)["mean"],
            "displaced_component_words_json": json.dumps(component_counts_from_meta(am), ensure_ascii=False),
        })

    # Pair comparisons.
    child_rows = arm_rows10["childsub_posmatched"]
    adult_rows = arm_rows10["adult_posmatched"]
    child_idx = changed_sets["childsub_posmatched"]
    adult_idx = changed_sets["adult_posmatched"]
    pair = {
        "changed_row_count_child": len(child_idx),
        "changed_row_count_adult": len(adult_idx),
        "same_changed_positions_count": len(child_idx & adult_idx),
        "union_changed_positions_count": len(child_idx | adult_idx),
        "fineweb_multiset_identical_by_text_hash": collections.Counter(fineweb_hashes["childsub_posmatched"]) == collections.Counter(fineweb_hashes["adult_posmatched"]),
        "changed_word_count_sequence_identical_after_sort": sorted(row_words(child_rows[i]) for i in child_idx) == sorted(row_words(adult_rows[i]) for i in adult_idx),
        "all_rows_word_count_sequence_identical_between_arms": [row_words(r) for r in child_rows] == [row_words(r) for r in adult_rows],
        "all_rows_word_count_sequence_identical_to_clean": [row_words(r) for r in child_rows] == [row_words(r) for r in clean] == [row_words(r) for r in adult_rows],
    }
    # Text equality outside union of changed rows: rows not changed by either arm should both equal clean.
    outside_union_mismatch = []
    for i in range(len(clean)):
        if i in child_idx or i in adult_idx:
            continue
        if child_rows[i].get("text") != clean[i].get("text") or adult_rows[i].get("text") != clean[i].get("text"):
            if len(outside_union_mismatch) < 20:
                outside_union_mismatch.append(i)
    pair["outside_union_equal_clean"] = not outside_union_mismatch
    pair["outside_union_mismatch_examples"] = outside_union_mismatch

    stream100 = {name: stream100_check(cfg["file100"], arm_rows10[name]) for name, cfg in ARMS.items()}
    # Cross-arm file diff count at 10M.
    cross_diffs = []
    for i, (cr, ar) in enumerate(zip(child_rows, adult_rows)):
        if cr.get("text") != ar.get("text"):
            if len(cross_diffs) < 20:
                cross_diffs.append({"row": i, "child_kind": cr.get("regpos_row_kind"), "adult_kind": ar.get("regpos_row_kind"), "child_words": row_words(cr), "adult_words": row_words(ar)})
    pair["cross_arm_text_diff_rows_expected_union"] = len(child_idx | adult_idx)
    pair["cross_arm_text_diff_examples"] = cross_diffs

    # Write replacement maps for external inspection.
    map_rows = []
    for name, rows in arm_rows10.items():
        for i in sorted(changed_sets[name]):
            r = rows[i]
            c = clean[i]
            map_rows.append({
                "arm": name,
                "output_row_index": i,
                "words": row_words(r),
                "fineweb_example_id": r.get("example_id"),
                "fineweb_original_row_index": r.get("regpos_original_fineweb_row_index"),
                "replaced_clean_row_index": r.get("regpos_replaced_clean_row_index"),
                "fineweb_text_sha": text_hash(str(r.get("text", ""))),
                "clean_text_sha": text_hash(str(c.get("text", ""))),
                "clean_source": c.get("source"),
                "clean_example_id": c.get("example_id"),
                "fineweb_token_len": token_len(tok, str(r.get("text", ""))),
                "clean_token_len": token_len(tok, str(c.get("text", ""))),
            })

    def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
        fields = sorted({k for r in rows for k in r}) if rows else []
        with path.open("w", encoding="utf-8", newline="") as f:
            if fields:
                w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

    write_csv(OUT / "arm_stream_audit_rows.csv", audit_rows)
    write_csv(OUT / "replacement_map_rows.csv", map_rows)

    result = {
        "status": "REGISTER_STREAM_AUDIT_COMPLETE",
        "created_utc": now(),
        "boundary": "File-only stream audit; no model loading, training, evaluation, GPU, GlobalPIQA, SuperGLUE, AoA, upload, or leaderboard.",
        "tokenizer_sha": tok_sha,
        "tokenizer_sha_match": tok_sha == EXPECTED_TOKENIZER_SHA,
        "clean10": {"path": rel(CLEAN10), "rows": len(clean), "words": clean_word_total, "rows_match": len(clean) == EXPECTED_ROWS10, "words_match": clean_word_total == EXPECTED_WORDS10, "source_counts": source_counts(clean)},
        "arm_audit_rows": audit_rows,
        "pair_audit": pair,
        "stream100_checks": stream100,
        "files": {"arm_audit_csv": rel(OUT / "arm_stream_audit_rows.csv"), "replacement_map_csv": rel(OUT / "replacement_map_rows.csv"), "summary_json": rel(OUT / "register_stream_audit_summary.json"), "summary_md": rel(OUT / "register_stream_audit_summary.md")},
    }
    (OUT / "register_stream_audit_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")

    def fmt(x: Any) -> str:
        try:
            if isinstance(x, bool):
                return str(x)
            return f"{float(x):.4f}"
        except Exception:
            return str(x)
    lines = [
        "# research register stream audit",
        "",
        result["boundary"],
        "",
        f"Tokenizer SHA match: {result['tokenizer_sha_match']} (`{tok_sha}`).",
        f"Clean rows/words match expected: {result['clean10']['rows_match']} / {result['clean10']['words_match']}.",
        "",
        "## Arm checks",
        "",
        "| arm | rows10 | words10 | changed rows | changed words | unchanged mismatches | length mismatches | replacement-index mismatches | fineweb multiset hash ok | mean token diff FineWeb-clean | FineWeb fertility | clean fertility |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for r in audit_rows:
        lines.append(f"| {r['arm']} | {r['rows10']} | {r['words10']} | {r['changed_rows']} | {r['changed_words']} | {r['unchanged_mismatch_count_visible']} | {r['changed_word_length_mismatch_count']} | {r['replaced_clean_row_index_mismatch_count']} | {r['fineweb_text_multiset_hash_actual']==r['fineweb_text_multiset_hash_metadata']} | {fmt(r['token_diff_fine_minus_replaced_clean_mean'])} | {fmt(r['fineweb_changed_token_fertility_mean'])} | {fmt(r['replaced_clean_token_fertility_mean'])} |")
    lines += [
        "",
        "## Pair checks",
        "",
        f"- FineWeb text multiset identical by text hash: {pair['fineweb_multiset_identical_by_text_hash']}.",
        f"- Changed row count child/adult: {pair['changed_row_count_child']} / {pair['changed_row_count_adult']}.",
        f"- Same changed positions across arms: {pair['same_changed_positions_count']}; union changed positions: {pair['union_changed_positions_count']}.",
        f"- All rows word-count sequence identical between arms: {pair['all_rows_word_count_sequence_identical_between_arms']}; identical to clean: {pair['all_rows_word_count_sequence_identical_to_clean']}.",
        f"- Rows outside union of changed positions equal clean in both arms: {pair['outside_union_equal_clean']}.",
        "",
        "## 100M repetition",
        "",
        "| arm | rows | words | ordered 10x repetition | mismatches |",
        "|---|---:|---:|---|---:|",
    ]
    for name, rec in stream100.items():
        lines.append(f"| {name} | {rec['rows']} | {rec['words']} | {rec['tenfold_ordered_repetition']} | {len(rec['mismatch_examples'])} |")
    lines += [
        "",
        "## Interpretation",
        "",
        "The actual stream files support the intended pair if all checks above are true: the contrast varies the directly removed clean register while keeping the admitted FineWeb multiset and word-count/row-count geometry fixed. Token fertility differences between admitted FineWeb and replaced clean rows remain part of the substitution mechanism and should be reported with score results.",
        "",
        f"Replacement map: `{rel(OUT / 'replacement_map_rows.csv')}`",
        f"JSON: `{rel(OUT / 'register_stream_audit_summary.json')}`",
    ]
    (OUT / "register_stream_audit_summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "tokenizer_sha_match": result["tokenizer_sha_match"],
        "pair_audit": pair,
        "stream100_ordered_repetition": {k: v["tenfold_ordered_repetition"] for k, v in stream100.items()},
        "summary_md": rel(OUT / "register_stream_audit_summary.md"),
    }, indent=2, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
