#!/usr/bin/env python3
"""research refinement: exposure multiplicity of dose originals already in the base pool.

The first base-presence audit was binary: did a selected dose original occur in the
base 10M stream, and was its same source/example row ever replaced by the dose
materializer?  For exposure accounting the pass multiplicity matters because the
100M stream contains ten shuffled passes and the dose replacement slots hit a given
source/example row in only some passes.  This script combines the selected-pair
presence rows with replacement metadata counters to estimate, per 10M pass and over
100M exposure, how much of each dose is (a) duplicate originals already in the base,
(b) newly exposed original text, and (c) new rewrite text.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import csv
import json
import pathlib
import statistics
import time
from collections import Counter, defaultdict
from typing import Any

ROOT0 = _public_path('experiments/archive/relation_learning/scripts/refine_dose_original_exposure.py')
ROOT = _PUBLIC_ROOT
WS = ROOT / "experiments/archive/relation_learning"
PRES = WS / "data/dose_original_base_presence/all_original_presence_rows.csv"
META21 = WS / "data/probe_clean_nested_dose_streams/dose21/dose21_dose_rows_meta.jsonl"
META25 = WS / "data/probe_clean_nested_dose_streams/dose25/dose25_dose_rows_meta.jsonl"
OUT = WS / "data/dose_original_base_presence"
PASSES = 10


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8"); return
    fields=[]; seen=set()
    for r in rows:
        for k in r:
            if k not in seen:
                fields.append(k); seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def replacement_counter(path: pathlib.Path) -> Counter[tuple[str,int]]:
    c: Counter[tuple[str,int]] = Counter()
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            r=json.loads(line)
            try:
                c[(str(r.get("old_source","")), int(r.get("old_example_id")))] += 1
            except Exception:
                pass
    return c


def as_bool(x: str) -> bool:
    return str(x).strip().lower() in {"true","1","yes"}


def summarize(rows: list[dict[str, Any]], set_name: str, mode: str) -> dict[str, Any]:
    n=len(rows)
    words=Counter(); counts=Counter(); repl_dist=Counter(); by_source=defaultdict(Counter)
    for r in rows:
        src=r["source"]; present=bool(r["present"]); rc=int(r[f"replacement_count_{mode}"])
        ow=int(r["original_words"]); rw=int(r["rewrite_words"]); pw=int(r["pair_words"])
        counts["pairs"] += 1
        counts["present_pairs"] += int(present)
        counts["absent_original_pairs"] += int(not present)
        counts["present_pairs_replaced_zero_passes"] += int(present and rc == 0)
        counts["present_pairs_replaced_some_passes"] += int(present and 0 < rc < PASSES)
        counts["present_pairs_replaced_all_passes"] += int(present and rc >= PASSES)
        repl_dist[str(rc)] += int(present)
        # Per-pass selected material.
        words["pair_words_per_pass"] += pw
        words["original_words_per_pass"] += ow
        words["rewrite_words_per_pass"] += rw
        words["present_original_words_per_pass"] += ow if present else 0
        words["absent_original_words_per_pass"] += ow if not present else 0
        words["pair_words_with_present_original_per_pass"] += pw if present else 0
        words["pair_words_with_absent_original_per_pass"] += pw if not present else 0
        # Over 100M. Each selected relation pair is inserted once per pass.
        words["inserted_original_word_occurrences_100M"] += ow * PASSES
        words["inserted_rewrite_word_occurrences_100M"] += rw * PASSES
        words["inserted_pair_word_occurrences_100M"] += pw * PASSES
        if present:
            # The base ordinary occurrence would be present once/pass before intervention.
            words["base_original_word_occurrences_before_100M"] += ow * PASSES
            words["same_original_base_occurrences_replaced_100M"] += ow * min(PASSES, rc)
            words["same_original_base_occurrences_remaining_100M"] += ow * max(0, PASSES - rc)
            words["net_extra_original_word_occurrences_vs_base_100M"] += ow * max(0, PASSES - rc)
        else:
            words["new_absent_original_word_occurrences_100M"] += ow * PASSES
            words["net_extra_original_word_occurrences_vs_base_100M"] += ow * PASSES
        # Rewrite text is counted as new exposure relative to the base unless another audit later proves otherwise.
        words["new_rewrite_word_occurrences_100M"] += rw * PASSES
        b=by_source[src]
        b["pairs"] += 1; b["present_pairs"] += int(present); b["pair_words"] += pw; b["original_words"] += ow; b["rewrite_words"] += rw
        b["present_original_words"] += ow if present else 0; b["replacement_passes_present_pairs"] += rc if present else 0
    pair_words=words["pair_words_per_pass"]
    orig_words=words["original_words_per_pass"]
    present_orig=words["present_original_words_per_pass"]
    new_text_per_pass = words["absent_original_words_per_pass"] + words["rewrite_words_per_pass"]
    duplicate_original_per_pass = present_orig
    source_rows=[]
    for src,c in sorted(by_source.items()):
        source_rows.append({
            "set": set_name, "mode": mode, "source": src,
            "pairs": int(c["pairs"]), "present_pairs": int(c["present_pairs"]),
            "present_pair_frac": c["present_pairs"] / c["pairs"] if c["pairs"] else None,
            "pair_words_per_pass": int(c["pair_words"]),
            "original_words_per_pass": int(c["original_words"]),
            "rewrite_words_per_pass": int(c["rewrite_words"]),
            "present_original_words_per_pass": int(c["present_original_words"]),
            "mean_replacement_passes_among_present_pairs": c["replacement_passes_present_pairs"] / c["present_pairs"] if c["present_pairs"] else None,
        })
    return {
        "set": set_name,
        "mode": mode,
        "pairs": n,
        "counts": dict(counts),
        "replacement_pass_distribution_among_present_pairs": dict(sorted(repl_dist.items(), key=lambda kv:int(kv[0]))),
        "mean_replacement_passes_among_present_pairs": (sum(int(k)*v for k,v in repl_dist.items()) / max(1, sum(repl_dist.values()))),
        "word_counts": dict(words),
        "per_pass_decomposition": {
            "pair_words": int(pair_words),
            "original_words": int(orig_words),
            "rewrite_words": int(words["rewrite_words_per_pass"]),
            "duplicate_original_words_already_in_base": int(duplicate_original_per_pass),
            "absent_original_words_new_to_base": int(words["absent_original_words_per_pass"]),
            "rewrite_words_new_to_base_assuming_no_rewrite_duplicate": int(words["rewrite_words_per_pass"]),
            "new_or_relocated_relation_text_words": int(new_text_per_pass),
            "duplicate_original_fraction_of_pair_words": duplicate_original_per_pass / max(1, pair_words),
            "new_or_relocated_text_fraction_of_pair_words": new_text_per_pass / max(1, pair_words),
        },
        "exposure_100M": {
            "inserted_pair_word_occurrences": int(words["inserted_pair_word_occurrences_100M"]),
            "inserted_original_word_occurrences": int(words["inserted_original_word_occurrences_100M"]),
            "inserted_rewrite_word_occurrences": int(words["inserted_rewrite_word_occurrences_100M"]),
            "base_present_original_word_occurrences_before": int(words["base_original_word_occurrences_before_100M"]),
            "same_original_base_occurrences_replaced": int(words["same_original_base_occurrences_replaced_100M"]),
            "same_original_base_occurrences_remaining": int(words["same_original_base_occurrences_remaining_100M"]),
            "net_extra_original_word_occurrences_vs_base": int(words["net_extra_original_word_occurrences_vs_base_100M"]),
            "new_absent_original_word_occurrences": int(words["new_absent_original_word_occurrences_100M"]),
            "new_rewrite_word_occurrences_assuming_no_rewrite_duplicate": int(words["new_rewrite_word_occurrences_100M"]),
        },
        "by_source": source_rows,
    }


def main() -> None:
    rows0 = read_csv(PRES)
    rc21 = replacement_counter(META21); rc25 = replacement_counter(META25)
    enriched=[]
    # Drop duplicate dose25_superset rows if any? We keep the three set views separate.
    for r in rows0:
        key=(r["source"], int(r["example_id"]))
        rr={
            "set": r["set"], "pair_id": r["pair_id"], "source": r["source"], "example_id": int(r["example_id"]),
            "present": as_bool(r["base_contains_exact_normalized_original"]),
            "original_words": int(r["original_words_field"]), "rewrite_words": int(r["rewrite_words_field"]), "pair_words": int(r["pair_words_field"]),
            "replacement_count_dose21": int(rc21.get(key,0)),
            "replacement_count_dose25": int(rc25.get(key,0)),
        }
        enriched.append(rr)
    write_csv(OUT / "all_original_presence_rows_with_replacement_counts.csv", enriched)
    summaries=[]
    for set_name, mode in [("dose21", "dose21"), ("dose21", "dose25"), ("dose25_extra", "dose25"), ("dose25_superset", "dose25")]:
        rows=[r for r in enriched if r["set"]==set_name]
        summaries.append(summarize(rows, set_name, mode))
    # Flatten source table.
    source_rows=[]
    for s in summaries:
        source_rows.extend(s["by_source"])
    write_csv(OUT / "exposure_refined_by_source.csv", source_rows)
    result={
        "status": "DOSE_ORIGINAL_EXPOSURE_REFINED",
        "created_utc": now(),
        "input_presence_rows": rel(PRES),
        "replacement_meta": {"dose21": rel(META21), "dose25": rel(META25)},
        "passes": PASSES,
        "summaries": summaries,
        "outputs": {
            "row_counts": rel(OUT / "all_original_presence_rows_with_replacement_counts.csv"),
            "by_source": rel(OUT / "exposure_refined_by_source.csv"),
            "summary": rel(OUT / "exposure_refined_summary.json"),
            "summary_md": rel((_PUBLIC_ROOT / 'research/documents/relation_learning/data/dose_original_base_presence/exposure_refined_summary.md')),
        },
        "interpretation_note": "About three quarters of selected dose original words were already present in the base 10M stream. Replacement slots hit some of those same source/example rows in only a subset of shuffled passes, so the dose usually adds duplicate original exposure plus new rewrite exposure and local co-location while displacing other ordinary text. The exact relation effect must therefore be read as duplicate-original/new-rewrite/local-pair substitution rather than as wholly new source content.",
    }
    (OUT / "exposure_refined_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    lines=["# research refined dose original exposure", "", result["interpretation_note"], ""]
    for s in summaries:
        d=s["per_pass_decomposition"]; e=s["exposure_100M"]
        lines += [f"## {s['set']} scored as {s['mode']} stream", f"- pair words/pass: {d['pair_words']}", f"- duplicate original words/pass already in base: {d['duplicate_original_words_already_in_base']} ({d['duplicate_original_fraction_of_pair_words']:.3f} of pair words)", f"- absent-original + rewrite words/pass: {d['new_or_relocated_relation_text_words']} ({d['new_or_relocated_text_fraction_of_pair_words']:.3f} of pair words)", f"- mean same-source/example replacement passes among present originals: {s['mean_replacement_passes_among_present_pairs']:.3f} / 10", f"- 100M inserted rewrite word occurrences: {e['inserted_rewrite_word_occurrences']}", f"- 100M net extra original word occurrences vs base after same-row replacements: {e['net_extra_original_word_occurrences_vs_base']}", ""]
    ((_PUBLIC_ROOT / 'research/documents/relation_learning/data/dose_original_base_presence/exposure_refined_summary.md')).write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "summary": result["outputs"]["summary"], "md": result["outputs"]["summary_md"]}, indent=2), flush=True)

if __name__ == "__main__":
    main()
