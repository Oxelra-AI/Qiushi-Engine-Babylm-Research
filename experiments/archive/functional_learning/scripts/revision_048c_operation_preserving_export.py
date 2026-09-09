#!/usr/bin/env python3
"""Export relation-first state maps for independent expression-transfer work.

The export provides the exact semantic state maps and the
operation-preserving scoring contract, without requiring reuse of the exact
surface wording.  The crucial invariant is the recipient-only reversal:
for a fixed query and candidate set, changing only which entity receives the shared
new value flips whether the queried entity's source value or shared_new is correct.
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
from typing import Any, Dict, Iterable, List

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
FIXED_PAIRS = _public_path('experiments/archive/functional_learning/data/saved_state_replicate_neutral_threeentity/fixed_repaired_pairs.jsonl')


def rel(path: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def read_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: pathlib.Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def overlap_counts(pairs: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    train = [p for p in pairs if p.get("split") == "train"]
    train_entities = {str(x).lower() for p in train for x in [p["entity_a"], p["entity_b"]]}
    train_values = {str(x).lower() for p in train for x in [p["value_a"], p["value_b"], p["shared_new_value"]]}
    out = {}
    for p in pairs:
        if p.get("split") != "held":
            continue
        ent_overlap = sum(1 for x in [p["entity_a"], p["entity_b"]] if str(x).lower() in train_entities)
        source_overlap = sum(1 for x in [p["value_a"], p["value_b"]] if str(x).lower() in train_values)
        new_overlap = 1 if str(p["shared_new_value"]).lower() in train_values else 0
        out[p["pair_id"]] = {
            "held_entity_overlap_count_with_train": ent_overlap,
            "held_source_value_overlap_count_with_train": source_overlap,
            "held_shared_new_value_overlap_with_train": new_overlap,
        }
    return out


def candidate_contract(pair: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "pair_id": pair["pair_id"],
        "split": pair.get("split"),
        "relation": pair["relation"],
        "entity_a": pair["entity_a"],
        "entity_b": pair["entity_b"],
        "state_a_source_value": pair["value_a"],
        "state_b_source_value": pair["value_b"],
        "shared_new_value": pair["shared_new_value"],
        "source_a_sentence": pair.get("source_a_sentence"),
        "source_b_sentence": pair.get("source_b_sentence"),
        "source_context": pair["source_context"],
        "original_update_a_sentence": pair["update_a_sentence"],
        "original_update_b_sentence": pair["update_b_sentence"],
        "original_use_frame_a": pair["use_frame_a"],
        "original_use_frame_b": pair["use_frame_b"],
        "candidate_set_for_query_a": {
            "source_correct_when_b_updated": pair["value_a"],
            "wrong_other_source": pair["value_b"],
            "new_correct_when_a_updated": pair["shared_new_value"],
        },
        "candidate_set_for_query_b": {
            "source_correct_when_a_updated": pair["value_b"],
            "wrong_other_source": pair["value_a"],
            "new_correct_when_b_updated": pair["shared_new_value"],
        },
        "required_contexts": {
            "neutral_query_a": "source_context + rephrased query frame for entity_a; correct source value_a must outrank value_b and shared_new",
            "neutral_query_b": "source_context + rephrased query frame for entity_b; correct source value_b must outrank value_a and shared_new",
            "self_update_a_query_a": "source_context + update_a(rephrased if desired) + query_a; shared_new must outrank value_a and value_b",
            "distractor_update_b_query_a": "source_context + update_b + query_a; value_a must outrank value_b and shared_new",
            "self_update_b_query_b": "source_context + update_b + query_b; shared_new must outrank value_b and value_a",
            "distractor_update_a_query_b": "source_context + update_a + query_b; value_b must outrank value_a and shared_new",
        },
        "rendering_constraints_for_reexpression": [
            "Do not change source_context, entity_a/entity_b, value_a/value_b, or shared_new_value.",
            "Rephrase update and/or query wording only if the relation type and candidate slot remain clear.",
            "Both entities must use the same relation type and the same candidate set; candidates should remain plausible for either entity under that relation.",
            "The final answer span must be a contiguous complete candidate phrase and all candidate phrase tokens should be masked simultaneously in scoring.",
            "Do not use a candidate-type or world-knowledge cue that makes one candidate uniquely appropriate for one entity independent of the presented context.",
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pairs", default=str(FIXED_PAIRS))
    ap.add_argument("--include-train", action="store_true")
    args = ap.parse_args()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = read_jsonl(pathlib.Path(args.pairs))
    overlaps = overlap_counts(pairs)
    held = [p for p in pairs if p.get("split") == "held"]
    exported = []
    for p in held:
        rec = candidate_contract(p)
        rec.update(overlaps.get(p["pair_id"], {}))
        exported.append(rec)
    exported.sort(key=lambda r: (
        r.get("held_entity_overlap_count_with_train", 99),
        r.get("held_source_value_overlap_count_with_train", 99),
        r.get("relation", ""),
        r.get("pair_id", ""),
    ))
    write_jsonl(out_dir / "a01_operation_preserving_held_maps.jsonl", exported)
    low_overlap = [r for r in exported if r.get("held_entity_overlap_count_with_train") == 0]
    write_jsonl(out_dir / "a01_low_entity_overlap_examples.jsonl", low_overlap)
    summary = {
        "status": "A01_OPERATION_PRESERVING_EXPORT_READY",
        "source_pairs": rel(args.pairs),
        "n_held_maps": len(exported),
        "relations": dict(collections.Counter(r["relation"] for r in exported)),
        "entity_overlap_counts": dict(collections.Counter(str(r.get("held_entity_overlap_count_with_train")) for r in exported)),
        "source_value_overlap_counts": dict(collections.Counter(str(r.get("held_source_value_overlap_count_with_train")) for r in exported)),
        "n_low_entity_overlap_examples": len(low_overlap),
        "contract": {
            "central_operation": "contextual state selection under recipient-only reversal",
            "success_rule": "For each query orientation, self-update chooses shared_new; distractor-update retains the queried entity's source value over both wrong source and shared_new; neutral retrieves the queried source value over both alternatives.",
            "scoring": "Simultaneously mask the full candidate span for each candidate in the same rendered final frame; compare mean log probability or a predeclared length-normalized score consistently across candidates.",
            "why_this_avoids_the_Step74_shortcut": "Candidates are value_a/value_b/shared_new for the same typed relation (mostly place names), and both query orientations share the same candidate set, so entity type alone should not determine the answer without reading the contextual assignment.",
        },
        "outputs": {
            "held_maps": rel(out_dir / "a01_operation_preserving_held_maps.jsonl"),
            "low_entity_overlap_examples": rel(out_dir / "a01_low_entity_overlap_examples.jsonl"),
            "summary": rel(out_dir / "summary.json"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
