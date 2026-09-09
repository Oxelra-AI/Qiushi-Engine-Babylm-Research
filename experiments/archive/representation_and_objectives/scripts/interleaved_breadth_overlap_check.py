#!/usr/bin/env python3
"""research: official score-text n-gram overlap check for the interleaved
whole-sentence source-breadth arm.

This uses the same post-construction provenance scanner as research. It never
uses official text for source selection; it only measures whether the repaired
preferred breadth companion set has suspicious direct overlap before training.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import time
from typing import Any

ROOT = pathlib.Path(".").resolve()
WS = ROOT / "experiments/archive/representation_and_objectives"
AUDIT = WS / "scripts/fw_comparison_mechanical_audit.py"
BREADTH_BUILDER_PATH = WS / "scripts/fw_source_breadth_arm.py"
BREADTH_SOURCES = WS / "data/fw_source_breadth_wholesentence_arm/source_breadth_wholesentence_companion_sources.jsonl"
OUT_DIR = WS / "data/fw_interleaved_breadth_overlap_check"
NOTE = (ROOT / 'research/notes/representation_and_objectives/fw_interleaved_breadth_overlap_check.md')


def import_module(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audit = import_module(AUDIT, "audit_for_interleaved_overlap")
    breadth_builder = import_module(BREADTH_BUILDER_PATH, "build_for_interleaved_overlap")
    pairs = breadth_builder.load_pairs()
    breadth_sources = read_jsonl(BREADTH_SOURCES)
    records, ng_to_ids, eval_counts = audit.build_eval_index()
    pair_sources = [{"norm_hash": p.get("norm_hash"), "text": p.get("source_text"), "words": p.get("source_words")} for p in pairs]
    compact_rewrites = [{"norm_hash": p.get("norm_hash"), "text": p.get("rewrite_text"), "words": p.get("rewrite_words")} for p in pairs]
    breadth_companions = [{"norm_hash": r.get("norm_hash"), "text": r.get("text"), "words": r.get("words")} for r in breadth_sources]
    comps = [
        audit.component_overlap("common_fineweb_selected_sources", pair_sources, "text", "words", records, ng_to_ids),
        audit.component_overlap("compact_rewrite_companions", compact_rewrites, "text", "words", records, ng_to_ids),
        audit.component_overlap("source_breadth_interleaved_wholesentence_companions", breadth_companions, "text", "words", records, ng_to_ids),
    ]
    result = {
        "status": "FW_INTERLEAVED_BREADTH_OVERLAP_CHECK",
        "strict_eval_record_counts_by_top": eval_counts,
        "strict_eval_records_total": len(records),
        "strict_eval_unique_ngrams": {str(n): len(ng_to_ids[n]) for n in audit.NS},
        "component_kind_summary": comps,
        "scientific_reading": "Post-construction overlap measurement only. No exact 10-gram matches in a differing component is the key leakage-rejection signal; short 7/8-gram generic overlaps should be stratified if a future result is concentrated in the affected task families.",
        "elapsed_sec": round(time.time() - t0, 2),
    }
    out_json = OUT_DIR / "fw_interleaved_breadth_overlap_check.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def line(name: str) -> str:
        rec = next(x for x in comps if x["component_kind"] == name)
        return f"n7 {rec['components_with_any_match_n7']} comps/{rec['unique_matched_ngrams_n7']} unique, n8 {rec['components_with_any_match_n8']} comps/{rec['unique_matched_ngrams_n8']} unique, n10 {rec['components_with_any_match_n10']} comps/{rec['unique_matched_ngrams_n10']} unique"
    NOTE.write_text(
        "# research — interleaved breadth official-text overlap check\n\n"
        f"- Common selected FineWeb sources: {line('common_fineweb_selected_sources')}.\n"
        f"- Compact rewrite companions: {line('compact_rewrite_companions')}.\n"
        f"- Interleaved whole-sentence breadth companions: {line('source_breadth_interleaved_wholesentence_companions')}.\n\n"
        "The overlap scanner is post-construction provenance checking only. The preferred interleaved breadth companion set has no exact 10-gram matches to the audited official score text.\n\n"
        f"JSON: `{out_json}`\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": result["status"],
        "common_n10": next(x for x in comps if x["component_kind"] == "common_fineweb_selected_sources")["components_with_any_match_n10"],
        "compact_rewrite_n10": next(x for x in comps if x["component_kind"] == "compact_rewrite_companions")["components_with_any_match_n10"],
        "breadth_n10": next(x for x in comps if x["component_kind"] == "source_breadth_interleaved_wholesentence_companions")["components_with_any_match_n10"],
        "breadth_n7": next(x for x in comps if x["component_kind"] == "source_breadth_interleaved_wholesentence_companions")["components_with_any_match_n7"],
        "json": str(out_json),
        "note": str(NOTE),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
