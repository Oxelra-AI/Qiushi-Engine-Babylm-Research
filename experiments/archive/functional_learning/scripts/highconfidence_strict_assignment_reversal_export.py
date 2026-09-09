#!/usr/bin/env python3
"""research: high-confidence strict assignment-reversal export.

This is a quality-focused companion to the larger research/065 exports.  It removes
filtered triples whose extracted entity is not a whole literal substring of the exact
source sentence or is far from the extracted value.  This catches failures such as
extracting "Hare" from "O'Hare" in a heading-like birth-year sentence.  It then builds
source/entity-disjoint held maps and preserves the A/B answer-slot schema.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import importlib.util
import json
import pathlib
import re
import time
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Tuple

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
SCRIPT = _public_path('experiments/archive/functional_learning/scripts/export_larger_assignment_reversal_pool.py')
SOURCE_SCRIPT = _public_path('experiments/archive/functional_learning/scripts/source_disjoint_assignment_reversal_export.py')
STRICT_SCRIPT = _public_path('experiments/archive/functional_learning/scripts/strict_disjoint_assignment_reversal_export.py')
DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/highconfidence_strict_assignment_reversal_export')

HIGHCONF_CAPS = {
    "death_place": 190,
    "birthplace": 165,
    "birth_year": 150,
    "located_in": 20,
    "founded_year": 20,
    "nationality": 16,
    "occupation": 16,
}


def import_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def rel(p: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_jsonl(path: pathlib.Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def entity_whole_occurrence(sentence: str, entity: str) -> Tuple[int, int] | None:
    if not entity:
        return None
    for m in re.finditer(re.escape(entity), sentence, flags=re.IGNORECASE):
        before = sentence[m.start() - 1] if m.start() > 0 else " "
        after = sentence[m.end()] if m.end() < len(sentence) else " "
        # Treat alphanumerics and apostrophe/hyphen as within-token continuations.
        if before.isalnum() or after.isalnum() or before in "'-’" or after in "'-’":
            continue
        return (m.start(), m.end())
    return None


def relation_evidence_ok(relation: str, sentence: str, value_pos: int) -> bool:
    pre = sentence[max(0, value_pos - 150): value_pos].lower()
    if relation in {"birth_year", "birthplace"}:
        return "born" in pre
    if relation == "death_place":
        return "died" in pre
    if relation == "founded_year":
        return bool(re.search(r"founded|established|created|formed|started|opened", pre))
    if relation == "located_in":
        return bool(re.search(r"\b(city|town|village|district|municipality|province|region|state|county|island|area|borough|suburb|commune)\b", pre))
    if relation in {"nationality", "occupation"}:
        return bool(re.search(r"\b(is|was)\s+(a|an)\s+", pre))
    return True


def high_confidence_triples(s64) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    triples, base_summary = s64.load_triples()
    kept, dropped = [], []
    for t in triples:
        sent = str(t.get("sentence", ""))
        ent = str(t.get("entity", ""))
        val = str(t.get("value", ""))
        relation = str(t.get("relation", ""))
        vi = sent.find(val)
        occ = entity_whole_occurrence(sent, ent)
        reason = None
        if vi < 0:
            reason = "value_not_literal"
        elif occ is None:
            reason = "entity_not_whole_literal"
        elif abs(vi - occ[0]) > 150 and relation in {"birth_year", "birthplace", "death_place", "founded_year", "nationality", "occupation"}:
            reason = "entity_value_far_apart"
        elif not relation_evidence_ok(relation, sent, vi):
            reason = "relation_cue_not_near_value"
        if reason is None:
            nt = dict(t)
            nt["highconfidence_filter"] = "whole_entity_literal_and_relation_cue_near_value"
            kept.append(nt)
        else:
            dd = dict(t)
            dd["drop_reason"] = reason
            dropped.append(dd)
    # Deduplicate after filtering, keeping shortest sentence per entity/relation/value/source id.
    seen = set()
    uniq = []
    for t in sorted(kept, key=lambda x: (x["relation"], x["entity"].lower(), x["value"].lower(), len(x["sentence"]), str(x.get("sentence_id")))):
        key = (t["relation"], t["entity"].lower(), t["value"].lower(), t.get("sentence_id"))
        if key not in seen:
            seen.add(key)
            uniq.append(t)
    summary = {
        "base_summary": base_summary,
        "kept_after_highconfidence_filter": len(uniq),
        "kept_by_relation": dict(Counter(t["relation"] for t in uniq)),
        "dropped": len(dropped),
        "dropped_by_reason": dict(Counter(t["drop_reason"] for t in dropped)),
        "dropped_by_relation": dict(Counter(t["relation"] for t in dropped)),
        "dropped_examples": dropped[:20],
    }
    return uniq, summary


def filter_rows(rows: List[Dict[str, Any]], split: str | None = None, frame_split: str | None = None, original_only: bool = False) -> List[Dict[str, Any]]:
    out = rows
    if split is not None:
        out = [r for r in out if r.get("split") == split]
    if original_only:
        out = [r for r in out if r.get("frame_id") == "f00_original_a01"]
    if frame_split is not None:
        out = [r for r in out if r.get("frame_split") == frame_split]
    return out


def pair_records_for(rows: List[Dict[str, Any]], pair_meta: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [pair_meta[pid] for pid in sorted({r["pair_id"] for r in rows})]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=65066)
    ap.add_argument("--min-large-relation-triples", type=int, default=20)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    s64 = import_module("assignment_export", SCRIPT)
    s65src = import_module("source_assignment_export", SOURCE_SCRIPT)
    s65strict = import_module("strict_assignment_export", STRICT_SCRIPT)
    old_caps = dict(s64.RELATION_CAPS)
    s64.RELATION_CAPS.clear(); s64.RELATION_CAPS.update(HIGHCONF_CAPS)
    triples, triple_summary = high_confidence_triples(s64)
    raw_maps, split_notes = s65src.make_maps(s64, triples, args.seed, args.min_large_relation_triples)
    maps, removed_maps, filter_info = s65strict.strict_filter_train_against_held(raw_maps)
    maps.sort(key=lambda x: (x["split"], x["relation"], x["base_pair_id"]))
    s64.RELATION_CAPS.clear(); s64.RELATION_CAPS.update(old_caps)

    all_rows: List[Dict[str, Any]] = []
    pair_meta: Dict[str, Dict[str, Any]] = {}
    for m in maps:
        for spec in s64.assignment_specs(m):
            for frame in s64.FRAMES:
                rows, bp = s64.make_two_rows(m, spec, frame)
                for r in rows:
                    r["packet_type"] = "A01_HIGHCONF_STRICT_ASSIGNMENT_REVERSAL_USE"
                    r["map_policy"] = "highconfidence_strict_source_entity_disjoint_literals"
                    r["training_contract"] = "high-confidence strict assignment-reversal A/B answer-slot: same source_context and shared_new value; two rows per assignment; mask/supervise answer_text tokens only."
                bp["map_policy"] = "highconfidence_strict_source_entity_disjoint_literals"
                all_rows.extend(rows)
                pair_meta[bp["pair_id"]] = bp

    files: Dict[str, Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]] = {}
    for split, label in [("train", "train"), ("heldout", "heldout")]:
        rows_all = filter_rows(all_rows, split=split)
        rows_seen = filter_rows(all_rows, split=split, frame_split="train_seen")
        rows_unseen = filter_rows(all_rows, split=split, frame_split="eval_unseen")
        rows_original = filter_rows(all_rows, split=split, original_only=True)
        files[f"{label}_frame_all"] = (rows_all, pair_records_for(rows_all, pair_meta))
        files[f"{label}_frame_seen"] = (rows_seen, pair_records_for(rows_seen, pair_meta))
        files[f"{label}_frame_unseen"] = (rows_unseen, pair_records_for(rows_unseen, pair_meta))
        files[f"{label}_original"] = (rows_original, pair_records_for(rows_original, pair_meta))

    outputs, checks = {}, {}
    for name, (rows, pairs) in files.items():
        row_path = args.out_dir / f"a01_highconf_strict_assignment_reversal_{name}.jsonl"
        pair_path = args.out_dir / f"binding_pairs_{name}.jsonl"
        write_jsonl(row_path, rows)
        write_jsonl(pair_path, pairs)
        chk = s64.static_check(rows, pairs)
        checks[name] = chk
        outputs[name] = {"rows_path": rel(row_path), "binding_pairs_path": rel(pair_path), "rows": len(rows), "pairs": len(pairs)}

    op_path = args.out_dir / "a01_highconf_strict_assignment_reversal_operation_maps.jsonl"
    write_jsonl(op_path, maps)
    removed_path = args.out_dir / "removed_train_overlap_maps.jsonl"
    write_jsonl(removed_path, removed_maps)
    sample = []
    for relation in sorted(set(m["relation"] for m in maps)):
        sample.extend([m for m in maps if m["relation"] == relation][:2])
    sample_path = args.out_dir / "operation_map_sample.json"
    sample_path.write_text(json.dumps(sample, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    fatal = sum(ch["duplicate_row_ids"] + ch["duplicate_pair_ids"] + ch["n_span_errors"] + ch["n_bad_pair_groups"] for ch in checks.values())
    overlap = s65strict.split_overlap(maps)
    status = "HIGHCONF_STRICT_ASSIGNMENT_REVERSAL_EXPORT_READY" if fatal == 0 and overlap["source_ids"]["n_overlap"] == 0 and overlap["entities"]["n_overlap"] == 0 else "HIGHCONF_STRICT_ASSIGNMENT_REVERSAL_EXPORT_HAS_WARNINGS"
    summary = {
        "status": status,
        "created_utc": now(),
        "scientific_purpose": "Cleaner large literal-value assignment-reversal pool: whole-entity literal filter, source/entity-disjoint held maps, same A/B answer-slot row contract.",
        "base_scripts": {"generator": rel(SCRIPT), "source_splitter": rel(SOURCE_SCRIPT), "strict_filter": rel(STRICT_SCRIPT)},
        "triple_summary": triple_summary,
        "highconfidence_relation_caps": HIGHCONF_CAPS,
        "split_notes_by_relation": split_notes,
        "raw_maps_before_strict_filter": len(raw_maps),
        "strict_filter_info": {**filter_info, "removed_maps_path": rel(removed_path)},
        "n_base_maps": len(maps),
        "base_split_counts": dict(Counter(m["split"] for m in maps)),
        "base_relation_counts": dict(Counter(m["relation"] for m in maps)),
        "base_split_relation_counts": {sp: dict(Counter(m["relation"] for m in maps if m["split"] == sp)) for sp in ["train", "heldout"]},
        "outputs": outputs,
        "operation_maps": rel(op_path),
        "sample": rel(sample_path),
        "static_check_fatal_counts": {name: chk["duplicate_row_ids"] + chk["duplicate_pair_ids"] + chk["n_span_errors"] + chk["n_bad_pair_groups"] for name, chk in checks.items()},
        "static_checks": checks,
        "split_overlap": overlap,
        "reuse_profile": s65strict.map_reuse_profile(maps),
        "contract": {
            "row_schema": "A/B answer-slot: two rows per assignment-specific pair_id; pair_half A is unchanged/source_state, pair_half B is updated/new_state.",
            "quality_filter": "Every extracted entity is a whole literal substring of its exact source sentence; the value is literal and relation cue is near the value.",
            "strict_split": "No held source id or held entity is used in train maps after filtering; source values may overlap.",
            "scope": "Controlled counterfactual maps from exact BabyLM source sentences; update sentences and replacement values are constructed; not direct broad BabyLM improvement evidence.",
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if status.endswith("HAS_WARNINGS") or fatal != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
