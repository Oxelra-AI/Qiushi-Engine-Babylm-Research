#!/usr/bin/env python3
"""research: build a stricter assignment-reversal export with source-disjoint held maps.

This is a companion to the research larger export.  The research pool reaches the
requested 500+ scale but creates many pair combinations from a finite triple set,
so its train/held partitions reuse source sentences and entities.  This script keeps
that scale while reserving held source triples before pairing for the large relations
where enough triples exist.  Small relation types are retained as train-only variety.
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
import random
import time
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Tuple

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
BASE_SCRIPT = _public_path('experiments/archive/functional_learning/scripts/export_larger_assignment_reversal_pool.py')
DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/source_disjoint_assignment_reversal_export')


def import_step064():
    spec = importlib.util.spec_from_file_location("larger_assignment_export", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {BASE_SCRIPT}")
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


def split_triples_for_held(group: List[Dict[str, Any]], relation: str, seed: int, min_large: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Return train_triples, held_triples.  Only large relations get held triples."""
    rng = random.Random((seed * 1000003) ^ sum(ord(c) for c in relation))
    group = list(group)
    group.sort(key=lambda x: (str(x.get("entity", "")).lower(), str(x.get("value", "")).lower(), str(x.get("sentence_id", ""))))
    if len(group) < min_large:
        return group, [], {"relation": relation, "n_triples": len(group), "mode": "train_only_small_relation"}
    shuffled = group[:]
    rng.shuffle(shuffled)
    held_n = max(2, int(round(len(shuffled) * 0.20)))
    held_n = min(held_n, len(shuffled) - 2)
    held = shuffled[:held_n]
    train = shuffled[held_n:]
    return train, held, {"relation": relation, "n_triples": len(group), "mode": "source_triple_disjoint", "train_triples": len(train), "held_triples": len(held)}


def pair_candidates(mod, group: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    group = sorted(group, key=lambda x: (x["entity"].lower(), x["value"].lower(), str(x.get("sentence_id"))))
    out: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for i, a in enumerate(group):
        for b in group[i + 1:]:
            if a["value"].lower() == b["value"].lower():
                continue
            if a["entity"].lower() == b["entity"].lower():
                continue
            if a["entity"].lower() in b["sentence"].lower() or b["entity"].lower() in a["sentence"].lower():
                continue
            if a["value"].lower() in b["sentence"].lower() or b["value"].lower() in a["sentence"].lower():
                continue
            if mod.wc(a["sentence"]) + mod.wc(b["sentence"]) > 120:
                continue
            out.append((a, b))
    return out


def make_maps(mod, triples: List[Dict[str, Any]], seed: int, min_large_relation_triples: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rng = random.Random(seed)
    # Deduplicate to one shortest sentence per entity/relation before split.
    best: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for t in triples:
        key = (t["relation"], t["entity"].lower())
        old = best.get(key)
        if old is None or mod.wc(t["sentence"]) < mod.wc(old["sentence"]):
            best[key] = t
    by_rel: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for t in best.values():
        by_rel[t["relation"]].append(t)

    maps: List[Dict[str, Any]] = []
    split_notes: Dict[str, Any] = {}
    for relation in sorted(by_rel):
        cap = int(mod.RELATION_CAPS.get(relation, 0))
        if cap <= 0:
            continue
        train_triples, held_triples, note = split_triples_for_held(by_rel[relation], relation, seed, min_large_relation_triples)
        split_notes[relation] = note
        split_groups = [("train", train_triples), ("heldout", held_triples)]
        selected_by_split: Dict[str, List[Tuple[Dict[str, Any], Dict[str, Any]]]] = {"train": [], "heldout": []}
        if held_triples:
            held_target = max(1, int(round(cap * 0.20)))
        else:
            held_target = 0
        for split, grp in split_groups:
            cands = pair_candidates(mod, grp)
            rng.shuffle(cands)
            if split == "heldout":
                selected_by_split[split] = cands[:held_target]
            else:
                # Preserve total cap.  If held could not fill its nominal target, train receives the remainder.
                train_target = cap - len(selected_by_split["heldout"])
                selected_by_split[split] = cands[:train_target]
        # Because train is selected before held in the loop above, recompute train after held selection if needed.
        if held_triples:
            held_cands = pair_candidates(mod, held_triples)
            rng.shuffle(held_cands)
            selected_by_split["heldout"] = held_cands[:held_target]
            train_cands = pair_candidates(mod, train_triples)
            rng.shuffle(train_cands)
            selected_by_split["train"] = train_cands[: max(0, cap - len(selected_by_split["heldout"]))]
        for split in ["heldout", "train"]:
            for k, (a, b) in enumerate(selected_by_split[split]):
                source_context = a["sentence"].rstrip() + " " + b["sentence"].rstrip()
                shared = mod.choose_shared_new(relation, [a["value"], b["value"]], source_context, rng)
                if shared is None:
                    continue
                tmpl = mod.RELATION_TEMPLATES[relation]["update"]
                map_id = f"xsrf_{relation}_{split}_{k:04d}"
                maps.append({
                    "base_pair_id": map_id,
                    "relation": relation,
                    "source_context": source_context,
                    "entity_a": a["entity"],
                    "entity_b": b["entity"],
                    "value_a": a["value"],
                    "value_b": b["value"],
                    "shared_new_value": shared,
                    "update_a_sentence": tmpl.format(ENTITY=a["entity"], NEW_VALUE=shared),
                    "update_b_sentence": tmpl.format(ENTITY=b["entity"], NEW_VALUE=shared),
                    "source_a_sentence": a["sentence"],
                    "source_b_sentence": b["sentence"],
                    "source_a_id": a.get("sentence_id"),
                    "source_b_id": b.get("sentence_id"),
                    "source_a_value_span": a.get("value_span"),
                    "source_b_value_span": b.get("value_span"),
                    "triple_provenance_a": a.get("triple_provenance"),
                    "triple_provenance_b": b.get("triple_provenance"),
                    "split": split,
                    "augmentation_note": "Update sentences are controlled counterfactual augmentation; source sentences are exact BabyLM text from extracted relation triples.",
                    "operation_contract": "update_a and update_b are assignment reversals with the same source_context and shared_new_value; retained source answer depends on queried entity.",
                    "split_policy": "source_triple_disjoint_held_for_large_relations; small relations are train-only type variety",
                })
    maps.sort(key=lambda x: (x["split"], x["relation"], x["base_pair_id"]))
    return maps, split_notes


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=65065)
    ap.add_argument("--min-large-relation-triples", type=int, default=20)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    mod = import_step064()
    triples, triple_summary = mod.load_triples()
    maps, split_notes = make_maps(mod, triples, args.seed, args.min_large_relation_triples)

    all_rows: List[Dict[str, Any]] = []
    pair_meta: Dict[str, Dict[str, Any]] = {}
    for m in maps:
        for spec in mod.assignment_specs(m):
            for frame in mod.FRAMES:
                rows, bp = mod.make_two_rows(m, spec, frame)
                # Preserve the assignment-reversal row contract fields.
                for r in rows:
                    r["packet_type"] = "A01_SOURCE_DISJOINT_ASSIGNMENT_REVERSAL_USE"
                    r["map_policy"] = "source_triple_disjoint_literals"
                    r["training_contract"] = "source-disjoint assignment-reversal A/B answer-slot: same source_context and shared_new value; two rows per assignment; mask/supervise answer_text tokens only."
                bp["map_policy"] = "source_triple_disjoint_literals"
                all_rows.extend(rows)
                pair_meta[bp["pair_id"]] = bp

    files: Dict[str, Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]] = {}
    for split, split_label in [("train", "train"), ("heldout", "heldout")]:
        rows_all = filter_rows(all_rows, split=split)
        rows_seen = filter_rows(all_rows, split=split, frame_split="train_seen")
        rows_unseen = filter_rows(all_rows, split=split, frame_split="eval_unseen")
        rows_original = filter_rows(all_rows, split=split, original_only=True)
        files[f"{split_label}_frame_all"] = (rows_all, pair_records_for(rows_all, pair_meta))
        files[f"{split_label}_frame_seen"] = (rows_seen, pair_records_for(rows_seen, pair_meta))
        files[f"{split_label}_frame_unseen"] = (rows_unseen, pair_records_for(rows_unseen, pair_meta))
        files[f"{split_label}_original"] = (rows_original, pair_records_for(rows_original, pair_meta))

    outputs = {}
    checks = {}
    for name, (rows, pairs) in files.items():
        row_path = args.out_dir / f"a01_source_disjoint_assignment_reversal_{name}.jsonl"
        pair_path = args.out_dir / f"binding_pairs_{name}.jsonl"
        write_jsonl(row_path, rows)
        write_jsonl(pair_path, pairs)
        chk = mod.static_check(rows, pairs)
        checks[name] = chk
        outputs[name] = {"rows_path": rel(row_path), "binding_pairs_path": rel(pair_path), "rows": len(rows), "pairs": len(pairs)}

    op_path = args.out_dir / "a01_source_disjoint_assignment_reversal_operation_maps.jsonl"
    write_jsonl(op_path, maps)
    sample = []
    for relation in sorted(set(m["relation"] for m in maps)):
        sample.extend([m for m in maps if m["relation"] == relation][:2])
    (args.out_dir / "operation_map_sample.json").write_text(json.dumps(sample, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    fatal = sum(ch["duplicate_row_ids"] + ch["duplicate_pair_ids"] + ch["n_span_errors"] + ch["n_bad_pair_groups"] for ch in checks.values())
    summary = {
        "status": "SOURCE_DISJOINT_ASSIGNMENT_REVERSAL_EXPORT_READY" if fatal == 0 else "SOURCE_DISJOINT_ASSIGNMENT_REVERSAL_EXPORT_HAS_ERRORS",
        "created_utc": now(),
        "scientific_purpose": "Provide a large literal-value assignment-reversal pool whose held rows use source triples reserved before pairing for the large relations, avoiding the train-held source/entity reuse found in the research larger export.",
        "base_step064_generator": rel(BASE_SCRIPT),
        "triple_sources": {"narrow": rel(mod.NARROW_TRIPLES), "broad": rel(mod.BROAD_TRIPLES)},
        "triple_summary_after_filter": triple_summary,
        "split_notes_by_relation": split_notes,
        "relation_caps": dict(mod.RELATION_CAPS),
        "n_base_maps": len(maps),
        "base_split_counts": dict(Counter(m["split"] for m in maps)),
        "base_relation_counts": dict(Counter(m["relation"] for m in maps)),
        "base_split_relation_counts": {sp: dict(Counter(m["relation"] for m in maps if m["split"] == sp)) for sp in ["train", "heldout"]},
        "outputs": outputs,
        "operation_maps": rel(op_path),
        "sample": rel(args.out_dir / "operation_map_sample.json"),
        "static_checks": checks,
        "split_overlap": split_overlap(maps),
        "contract": {
            "row_schema": "A/B answer-slot: exactly two rows per assignment-specific pair_id; pair_half A is unchanged/source_state, pair_half B is updated/new_state.",
            "assignment_reversal": "Each base map produces update_a and update_b packets with the same source_context and shared_new_value but opposite updated entity.",
            "scope": "Controlled counterfactual relation-literal maps from exact BabyLM source sentences; update sentences and replacement values are constructed.",
            "boundary": "Held source/entity disjointness is provided for large relations only; small relation types are train-only variety because the filtered triple pool is too small for source-disjoint held pairs.",
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if fatal != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
