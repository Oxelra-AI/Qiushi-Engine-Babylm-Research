#!/usr/bin/env python3
"""research: audit whether added dose-pair originals were already present in the base stream.

The nested dose intervention inserts packed (original, rewrite) relation rows into the
compact-view-reinvest 100M stream while preserving inherited qwen_pair_packed rows.
A scientific interpretation question is whether the inserted originals are new
ordinary source exposure or duplicates of source sentences already present elsewhere
in the compact-view base stream.  This script checks the selected dose pairs against
the base 10M stream by source/example_id and normalized sentence containment, and
checks whether any selected source rows were themselves among the ordinary rows
replaced by the dose materializer.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import csv
import json
import pathlib
import re
import statistics
import time
from collections import Counter, defaultdict
from typing import Any

ROOT0 = _public_path('experiments/archive/relation_learning/scripts/audit_dose_original_base_presence.py')
ROOT = _PUBLIC_ROOT

WS = ROOT / "experiments/archive/relation_learning"
BASE10 = ROOT / "experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"
BASE100 = ROOT / "experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl"
DOSE21 = WS / "data/probe_clean_nested_dose_streams/dose21_selected_pairs_probe_clean.jsonl"
DOSE25_EXTRA = WS / "data/probe_clean_nested_dose_streams/dose25_extra_selected_pairs_probe_clean_from_shard0.jsonl"
DOSE25_SUPER = WS / "data/probe_clean_nested_dose_streams/dose25_selected_pairs_probe_clean_superset.jsonl"
META21 = WS / "data/probe_clean_nested_dose_streams/dose21/dose21_dose_rows_meta.jsonl"
META25 = WS / "data/probe_clean_nested_dose_streams/dose25/dose25_dose_rows_meta.jsonl"
OUT = WS / "data/dose_original_base_presence"
WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def norm_words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def norm_text(text: str) -> str:
    return " ".join(norm_words(text))


def contains_word_sequence(hay_words: list[str], needle_words: list[str]) -> bool:
    if not needle_words:
        return False
    n = len(needle_words)
    if n > len(hay_words):
        return False
    first = needle_words[0]
    for i, w in enumerate(hay_words[: len(hay_words) - n + 1]):
        if w == first and hay_words[i : i + n] == needle_words:
            return True
    return False


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                fields.append(k); seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def load_base_index(path: pathlib.Path) -> tuple[dict[tuple[str, int], list[dict[str, Any]]], dict[str, Any]]:
    idx: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    source_words = Counter()
    n = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            n += 1
            src = str(r.get("source", ""))
            try:
                eid = int(r.get("example_id"))
            except Exception:
                continue
            text = str(r.get("text", ""))
            nw = norm_words(text)
            rec = {
                "example_id": eid,
                "source": src,
                "words": int(r.get("words", 0)),
                "norm_words": nw,
                "text_prefix": text[:220],
            }
            idx[(src, eid)].append(rec)
            source_words[src] += int(r.get("words", 0))
    meta = {"path": rel(path), "rows": n, "indexed_source_example_keys": len(idx), "source_words": dict(source_words)}
    return idx, meta


def selected_sets() -> dict[str, list[dict[str, Any]]]:
    d21 = read_jsonl(DOSE21)
    d25e = read_jsonl(DOSE25_EXTRA)
    d25s = read_jsonl(DOSE25_SUPER)
    return {"dose21": d21, "dose25_extra": d25e, "dose25_superset": d25s}


def load_replaced_example_ids(path: pathlib.Path) -> set[tuple[str, int]]:
    out: set[tuple[str, int]] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            try:
                out.add((str(r.get("old_source", "")), int(r.get("old_example_id"))))
            except Exception:
                pass
    return out


def pair_presence_rows(name: str, pairs: list[dict[str, Any]], base_idx: dict[tuple[str, int], list[dict[str, Any]]], replaced21: set[tuple[str, int]], replaced25: set[tuple[str, int]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in pairs:
        src = str(p.get("source", ""))
        eid = p.get("example_id")
        try:
            eid_i = int(eid)
        except Exception:
            eid_i = -1
        key = (src, eid_i)
        original = str(p.get("original", ""))
        ow = norm_words(original)
        candidates = base_idx.get(key, [])
        contains = [c for c in candidates if contains_word_sequence(c["norm_words"], ow)]
        rows.append({
            "set": name,
            "pair_id": str(p.get("pair_id", "")),
            "original_id": str(p.get("original_id", "")),
            "source": src,
            "example_id": eid_i,
            "original_words_field": int(p.get("original_words", len(original.split()) or 0)),
            "rewrite_words_field": int(p.get("rewrite_words", 0)),
            "pair_words_field": int(p.get("pair_words", 0)),
            "base_same_source_example_rows": len(candidates),
            "base_contains_exact_normalized_original": bool(contains),
            "base_contains_count": len(contains),
            "base_match_sources": ";".join(sorted({c["source"] for c in contains})),
            "base_match_example_ids": ";".join(str(c["example_id"]) for c in contains[:5]),
            "replaced_in_dose21_by_same_source_example": key in replaced21,
            "replaced_in_dose25_by_same_source_example": key in replaced25,
            "selection_role": str(p.get("selection_role", "")),
        })
    return rows


def summarize(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"set": label, "pairs": 0}
    cnt = Counter()
    by_source: dict[str, Counter] = defaultdict(Counter)
    word = Counter()
    by_source_words: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        src = str(r["source"])
        present = bool(r["base_contains_exact_normalized_original"])
        same_key = int(r["base_same_source_example_rows"]) > 0
        rep21 = bool(r["replaced_in_dose21_by_same_source_example"])
        rep25 = bool(r["replaced_in_dose25_by_same_source_example"])
        ow = int(r["original_words_field"]); rw = int(r["rewrite_words_field"]); pw = int(r["pair_words_field"])
        cnt["pairs"] += 1
        cnt["same_source_example_key_in_base"] += int(same_key)
        cnt["exact_original_text_in_base_row"] += int(present)
        cnt["present_and_not_replaced_dose21"] += int(present and not rep21)
        cnt["present_and_not_replaced_dose25"] += int(present and not rep25)
        cnt["replaced_by_dose21_same_key"] += int(rep21)
        cnt["replaced_by_dose25_same_key"] += int(rep25)
        word["pair_words"] += pw
        word["original_words"] += ow
        word["rewrite_words"] += rw
        word["original_words_present_in_base"] += ow if present else 0
        word["rewrite_words_where_original_present"] += rw if present else 0
        word["pair_words_where_original_present"] += pw if present else 0
        word["pair_words_where_original_absent"] += pw if not present else 0
        for k, v in [("pairs", 1), ("present", int(present)), ("same_key", int(same_key))]:
            by_source[src][k] += v
        for k, v in [("pair_words", pw), ("original_words", ow), ("rewrite_words", rw), ("original_words_present", ow if present else 0), ("pair_words_present", pw if present else 0)]:
            by_source_words[src][k] += v
    source_rows = []
    for src in sorted(by_source):
        c = by_source[src]; w = by_source_words[src]
        source_rows.append({
            "source": src,
            "pairs": int(c["pairs"]),
            "present_pairs": int(c["present"]),
            "present_pair_frac": float(c["present"] / c["pairs"]) if c["pairs"] else None,
            "same_key_pairs": int(c["same_key"]),
            "pair_words": int(w["pair_words"]),
            "original_words": int(w["original_words"]),
            "rewrite_words": int(w["rewrite_words"]),
            "original_words_present": int(w["original_words_present"]),
            "pair_words_present": int(w["pair_words_present"]),
        })
    return {
        "set": label,
        "pairs": n,
        "pair_counts": dict(cnt),
        "word_counts": dict(word),
        "fractions": {
            "pairs_with_same_source_example_key_in_base": cnt["same_source_example_key_in_base"] / n,
            "pairs_with_exact_original_text_in_base_row": cnt["exact_original_text_in_base_row"] / n,
            "original_word_fraction_already_in_base": word["original_words_present_in_base"] / max(1, word["original_words"]),
            "pair_word_fraction_with_original_already_in_base": word["pair_words_where_original_present"] / max(1, word["pair_words"]),
        },
        "by_source": source_rows,
    }


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    base_idx, base_meta = load_base_index(BASE10)
    sets = selected_sets()
    replaced21 = load_replaced_example_ids(META21)
    replaced25 = load_replaced_example_ids(META25)
    all_rows: list[dict[str, Any]] = []
    summaries = []
    for name, pairs in sets.items():
        rows = pair_presence_rows(name, pairs, base_idx, replaced21, replaced25)
        write_csv(OUT / f"{name}_original_presence_rows.csv", rows)
        all_rows.extend(rows)
        summaries.append(summarize(rows, name))
    write_csv(OUT / "all_original_presence_rows.csv", all_rows)
    result = {
        "status": "DOSE_ORIGINAL_BASE_PRESENCE_DONE",
        "created_utc": now(),
        "base_index": base_meta,
        "base100_reference": rel(BASE100),
        "selected_inputs": {"dose21": rel(DOSE21), "dose25_extra": rel(DOSE25_EXTRA), "dose25_superset": rel(DOSE25_SUPER)},
        "replacement_meta_inputs": {"dose21": rel(META21), "dose25": rel(META25)},
        "replaced_source_example_keys": {"dose21": len(replaced21), "dose25": len(replaced25)},
        "set_summaries": summaries,
        "outputs": {
            "all_rows": rel(OUT / "all_original_presence_rows.csv"),
            "dose21_rows": rel(OUT / "dose21_original_presence_rows.csv"),
            "dose25_extra_rows": rel(OUT / "dose25_extra_original_presence_rows.csv"),
            "dose25_superset_rows": rel(OUT / "dose25_superset_original_presence_rows.csv"),
            "summary": rel(OUT / "summary.json"),
            "summary_md": rel((_PUBLIC_ROOT / 'research/documents/relation_learning/data/dose_original_base_presence/summary.md')),
        },
        "interpretation_note": "Presence means the selected pair's normalized original sentence is contained in a base 10M ordinary row with the same source and example_id. Because the 100M stream repeats the base pool ten times, such originals are cross-row duplicates when inserted in dose rows unless their same source/example_id row was also replaced. This does not test semantic novelty of the rewrite.",
        "elapsed_sec": round(time.time() - t0, 2),
    }
    (OUT / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research dose original base-presence audit", "", f"Base 10M: `{base_meta['path']}`", ""]
    for s in summaries:
        fr = s.get("fractions", {})
        wc = s.get("word_counts", {})
        pc = s.get("pair_counts", {})
        lines.append(f"## {s['set']}")
        lines.append(f"- pairs: {s['pairs']}")
        lines.append(f"- exact original text already in same-source/example base row: {pc.get('exact_original_text_in_base_row', 0)} pairs ({fr.get('pairs_with_exact_original_text_in_base_row', 0):.3f})")
        lines.append(f"- original words already in base rows: {wc.get('original_words_present_in_base', 0)} / {wc.get('original_words', 0)} ({fr.get('original_word_fraction_already_in_base', 0):.3f})")
        lines.append(f"- pair words whose original was already in base: {wc.get('pair_words_where_original_present', 0)} / {wc.get('pair_words', 0)} ({fr.get('pair_word_fraction_with_original_already_in_base', 0):.3f})")
        lines.append(f"- present originals not themselves replaced by dose25 same-key row: {pc.get('present_and_not_replaced_dose25', 0)}")
        lines.append("")
    lines.append(result["interpretation_note"])
    ((_PUBLIC_ROOT / 'research/documents/relation_learning/data/dose_original_base_presence/summary.md')).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "summary": result["outputs"]["summary"], "md": result["outputs"]["summary_md"], "elapsed_sec": result["elapsed_sec"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
