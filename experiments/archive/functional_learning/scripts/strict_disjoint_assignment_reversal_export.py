#!/usr/bin/env python3
"""Stricter large assignment-reversal export.

The research larger export reached >500 maps but its random split reused source
sentences/entities across train and held.  The research source-disjoint companion
reserved source triples before pairing, but a few multi-relation sentences/entities still
crossed splits and the default caps left less margin after filtering.  This script uses
larger caps, then removes train maps sharing any source sentence id or entity with held
maps.  It preserves the A/B answer-slot row schema and writes a cleaner large substrate.
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
import time
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Tuple

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
SCRIPT = _public_path('experiments/archive/functional_learning/scripts/export_larger_assignment_reversal_pool.py')
SOURCE_SCRIPT = _public_path('experiments/archive/functional_learning/scripts/source_disjoint_assignment_reversal_export.py')
DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/strict_disjoint_assignment_reversal_export')

STRICT_CAPS = {
    "death_place": 190,
    "birthplace": 165,
    "birth_year": 150,
    "located_in": 20,
    "founded_year": 25,
    "nationality": 18,
    "occupation": 18,
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


def ids_entities(m: Dict[str, Any]) -> Tuple[set, set]:
    sids = {m.get("source_a_id"), m.get("source_b_id")}
    ents = {str(m.get("entity_a", "")).lower(), str(m.get("entity_b", "")).lower()}
    return {x for x in sids if x}, {x for x in ents if x}


def strict_filter_train_against_held(maps: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    held_sources, held_entities = set(), set()
    for m in maps:
        if m["split"] == "heldout":
            sids, ents = ids_entities(m)
            held_sources.update(sids)
            held_entities.update(ents)
    kept, removed = [], []
    for m in maps:
        if m["split"] != "train":
            kept.append(m)
            continue
        sids, ents = ids_entities(m)
        if (sids & held_sources) or (ents & held_entities):
            removed.append(m)
        else:
            kept.append(m)
    info = {
        "removed_train_maps_sharing_held_source_or_entity": len(removed),
        "removed_by_relation": dict(Counter(m["relation"] for m in removed)),
    }
    return kept, removed, info


def split_overlap(maps: List[Dict[str, Any]]) -> Dict[str, Any]:
    per_split = {"train": defaultdict(set), "heldout": defaultdict(set)}
    for m in maps:
        sp = m["split"]
        per_split[sp]["source_ids"].update([m.get("source_a_id"), m.get("source_b_id")])
        per_split[sp]["entities"].update([str(m.get("entity_a", "")).lower(), str(m.get("entity_b", "")).lower()])
        per_split[sp]["source_values"].update([str(m.get("value_a", "")).lower(), str(m.get("value_b", "")).lower()])
    out = {}
    for key in ["source_ids", "entities", "source_values"]:
        tr = {x for x in per_split["train"][key] if x}
        he = {x for x in per_split["heldout"][key] if x}
        inter = sorted(tr & he)
        out[key] = {"n_overlap": len(inter), "examples": inter[:20]}
    return out


def map_reuse_profile(maps: List[Dict[str, Any]]) -> Dict[str, Any]:
    source_counter = Counter()
    entity_counter = Counter()
    value_counter = Counter()
    for m in maps:
        for sid in [m.get("source_a_id"), m.get("source_b_id")]:
            if sid:
                source_counter[sid] += 1
        for ent in [m.get("entity_a"), m.get("entity_b")]:
            if ent:
                entity_counter[str(ent).lower()] += 1
        for val in [m.get("value_a"), m.get("value_b")]:
            if val:
                value_counter[str(val).lower()] += 1
    return {
        "unique_source_ids": len(source_counter),
        "unique_entities_lower": len(entity_counter),
        "unique_source_values_lower": len(value_counter),
        "max_maps_per_source_id": max(source_counter.values(), default=0),
        "max_maps_per_entity_lower": max(entity_counter.values(), default=0),
        "top_source_ids": [{"key": k, "count": v} for k, v in source_counter.most_common(12)],
        "top_entities_lower": [{"key": k, "count": v} for k, v in entity_counter.most_common(12)],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=65065)
    ap.add_argument("--min-large-relation-triples", type=int, default=20)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    s64 = import_module("assignment_export", SCRIPT)
    s65 = import_module("source_assignment_export", SOURCE_SCRIPT)
    old_caps = dict(s64.RELATION_CAPS)
    s64.RELATION_CAPS.clear(); s64.RELATION_CAPS.update(STRICT_CAPS)
    triples, triple_summary = s64.load_triples()
    raw_maps, split_notes = s65.make_maps(s64, triples, args.seed, args.min_large_relation_triples)
    maps, removed_maps, filter_info = strict_filter_train_against_held(raw_maps)
    maps.sort(key=lambda x: (x["split"], x["relation"], x["base_pair_id"]))
    # Restore imported module global just in case another import reuses it in-process.
    s64.RELATION_CAPS.clear(); s64.RELATION_CAPS.update(old_caps)

    all_rows: List[Dict[str, Any]] = []
    pair_meta: Dict[str, Dict[str, Any]] = {}
    for m in maps:
        for spec in s64.assignment_specs(m):
            for frame in s64.FRAMES:
                rows, bp = s64.make_two_rows(m, spec, frame)
                for r in rows:
                    r["packet_type"] = "A01_STRICT_DISJOINT_ASSIGNMENT_REVERSAL_USE"
                    r["map_policy"] = "strict_source_entity_disjoint_literals"
                    r["training_contract"] = "strict-disjoint assignment-reversal A/B answer-slot: same source_context and shared_new value; two rows per assignment; mask/supervise answer_text tokens only."
                bp["map_policy"] = "strict_source_entity_disjoint_literals"
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
        row_path = args.out_dir / f"a01_strict_disjoint_assignment_reversal_{name}.jsonl"
        pair_path = args.out_dir / f"binding_pairs_{name}.jsonl"
        write_jsonl(row_path, rows)
        write_jsonl(pair_path, pairs)
        chk = s64.static_check(rows, pairs)
        checks[name] = chk
        outputs[name] = {"rows_path": rel(row_path), "binding_pairs_path": rel(pair_path), "rows": len(rows), "pairs": len(pairs)}

    op_path = args.out_dir / "a01_strict_disjoint_assignment_reversal_operation_maps.jsonl"
    write_jsonl(op_path, maps)
    removed_path = args.out_dir / "removed_train_overlap_maps.jsonl"
    write_jsonl(removed_path, removed_maps)
    sample = []
    for relation in sorted(set(m["relation"] for m in maps)):
        sample.extend([m for m in maps if m["relation"] == relation][:2])
    sample_path = args.out_dir / "operation_map_sample.json"
    sample_path.write_text(json.dumps(sample, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    fatal = sum(ch["duplicate_row_ids"] + ch["duplicate_pair_ids"] + ch["n_span_errors"] + ch["n_bad_pair_groups"] for ch in checks.values())
    overlap = split_overlap(maps)
    status = "STRICT_DISJOINT_ASSIGNMENT_REVERSAL_EXPORT_READY" if fatal == 0 and overlap["source_ids"]["n_overlap"] == 0 and overlap["entities"]["n_overlap"] == 0 else "STRICT_DISJOINT_ASSIGNMENT_REVERSAL_EXPORT_HAS_WARNINGS"
    summary = {
        "status": status,
        "created_utc": now(),
        "scientific_purpose": "Large literal-value assignment-reversal pool with strict no train-held source-id/entity overlap after filtering, while retaining small relation types as train-only type variation.",
        "base_scripts": {"generator": rel(SCRIPT), "source_splitter": rel(SOURCE_SCRIPT)},
        "triple_summary_after_filter": triple_summary,
        "strict_relation_caps": STRICT_CAPS,
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
        "reuse_profile": map_reuse_profile(maps),
        "contract": {
            "row_schema": "A/B answer-slot: two rows per assignment-specific pair_id; pair_half A is unchanged/source_state, pair_half B is updated/new_state.",
            "assignment_reversal": "Each base map produces update_a and update_b packets with the same source_context and shared_new_value but opposite updated entity.",
            "strict_split": "No held source id or held entity is used in train maps after filtering; source values may overlap because years and common places recur.",
            "scope": "Controlled counterfactual maps from exact BabyLM source sentences; this is a mechanistic substrate, not direct broad BabyLM improvement evidence.",
            "small_relation_boundary": "founded_year, located_in, nationality, and occupation are train-only because the filtered triple pool is too small for source-disjoint held pairs; held relation coverage is birth_year, birthplace, and death_place.",
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if status.endswith("HAS_WARNINGS") or fatal != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
