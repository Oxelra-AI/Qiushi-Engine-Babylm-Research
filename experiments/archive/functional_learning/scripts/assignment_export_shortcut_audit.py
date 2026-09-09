#!/usr/bin/env python3
"""research: static shortcut-feature audit for assignment-reversal exports.

Earlier binding packets were found to contain cheap answer features. This audit
measures similar text-level regularities in the larger literal-value assignment pools.
The point is not to reject the pool: the task intentionally requires using the update
recipient.  The point is to make explicit which readouts would show only surface/recency
rules and which require stronger controls.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import pathlib
import statistics
from collections import Counter, defaultdict
from typing import Any, Dict, List

ROOT = _public_path('.')
DEFAULT_EXPORTS = [
    _public_path('experiments/archive/functional_learning/data/larger_assignment_reversal_export'),
    _public_path('experiments/archive/functional_learning/data/strict_disjoint_assignment_reversal_export'),
]


def rel(p: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def load_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def contains(hay: str, needle: str) -> bool:
    return str(needle).lower() in str(hay).lower()


def count_contains(hay: str, needle: str) -> int:
    h, n = str(hay).lower(), str(needle).lower()
    if not n:
        return 0
    c = start = 0
    while True:
        i = h.find(n, start)
        if i < 0:
            return c
        c += 1
        start = i + max(1, len(n))


def mean(xs: List[float]) -> float | None:
    return float(sum(xs) / len(xs)) if xs else None


def quantiles(xs: List[float]) -> Dict[str, float | None]:
    if not xs:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {"mean": mean(xs), "median": float(statistics.median(xs)), "min": float(min(xs)), "max": float(max(xs))}


def summarize_bool(rows: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    n = len(rows)
    c = sum(1 for r in rows if r.get(key))
    return {"true": c, "false": n - c, "true_frac": c / n if n else None}


def find_row_file(export_dir: pathlib.Path, kind: str) -> pathlib.Path:
    manifest = json.loads((export_dir / "manifest.json").read_text(encoding="utf-8"))
    path = pathlib.Path(manifest["outputs"][kind]["rows_path"])
    return ROOT / path if not path.is_absolute() else path


def audit_export(export_dir: pathlib.Path, row_kind: str) -> Dict[str, Any]:
    manifest = json.loads((export_dir / "manifest.json").read_text(encoding="utf-8"))
    row_path = find_row_file(export_dir, row_kind)
    rows = load_jsonl(row_path)
    feats = []
    for r in rows:
        ans = r["answer_text"]
        query = r["query_entity"]
        update_sentence = r.get("update_sentence", "")
        source_sentence = r.get("source_sentence", "")
        # Excluding the final answer span, because training/eval should mask that location.
        a, b = int(r["answer_char_start"]), int(r["answer_char_end"])
        context_without_answer = r["context_text"][:a] + r["context_text"][b:]
        f = {
            "row_id": r["row_id"],
            "pair_id": r["pair_id"],
            "split": r["split"],
            "relation": r["relation"],
            "role": r["role"],
            "answer_kind": r["answer_kind"],
            "answer_len_words": len(str(ans).split()),
            "context_words": len(str(r["context_text"]).split()),
            "answer_in_update_sentence": contains(update_sentence, ans),
            "answer_in_source_sentence": contains(source_sentence, ans),
            "query_entity_in_update_sentence": contains(update_sentence, query),
            "query_entity_in_source_sentence": contains(source_sentence, query),
            "wrong_other_source_in_update_sentence": contains(update_sentence, r.get("wrong_other_source", "")),
            "shared_new_value_in_source_sentence": contains(source_sentence, r.get("shared_new_value", "")),
            "answer_occurrences_excluding_final_answer": count_contains(context_without_answer, ans),
            "query_entity_occurrences_in_update": count_contains(update_sentence, query),
            "query_entity_occurrences_in_source": count_contains(source_sentence, query),
        }
        f["role_pred_update_answer_overlap"] = "updated_entity" if f["answer_in_update_sentence"] else "unchanged_entity"
        f["role_pred_query_update_overlap"] = "updated_entity" if f["query_entity_in_update_sentence"] else "unchanged_entity"
        feats.append(f)

    by_role = {}
    for role in sorted(set(f["role"] for f in feats)):
        sub = [f for f in feats if f["role"] == role]
        by_role[role] = {
            "n": len(sub),
            "answer_in_update_sentence": summarize_bool(sub, "answer_in_update_sentence"),
            "answer_in_source_sentence": summarize_bool(sub, "answer_in_source_sentence"),
            "query_entity_in_update_sentence": summarize_bool(sub, "query_entity_in_update_sentence"),
            "query_entity_in_source_sentence": summarize_bool(sub, "query_entity_in_source_sentence"),
            "wrong_other_source_in_update_sentence": summarize_bool(sub, "wrong_other_source_in_update_sentence"),
            "shared_new_value_in_source_sentence": summarize_bool(sub, "shared_new_value_in_source_sentence"),
            "answer_occurrences_excluding_final_answer": quantiles([float(f["answer_occurrences_excluding_final_answer"]) for f in sub]),
            "context_words": quantiles([float(f["context_words"]) for f in sub]),
        }

    shortcut_acc = {}
    for pred_key in ["role_pred_update_answer_overlap", "role_pred_query_update_overlap"]:
        acc = sum(1 for f in feats if f[pred_key] == f["role"]) / len(feats) if feats else None
        shortcut_acc[pred_key] = acc
    by_relation_role = {}
    for relation in sorted(set(f["relation"] for f in feats)):
        by_relation_role[relation] = {}
        for role in sorted(set(f["role"] for f in feats)):
            sub = [f for f in feats if f["relation"] == relation and f["role"] == role]
            if not sub:
                continue
            by_relation_role[relation][role] = {
                "n": len(sub),
                "answer_in_update_frac": sum(f["answer_in_update_sentence"] for f in sub) / len(sub),
                "query_in_update_frac": sum(f["query_entity_in_update_sentence"] for f in sub) / len(sub),
                "answer_in_source_frac": sum(f["answer_in_source_sentence"] for f in sub) / len(sub),
                "answer_occ_excl_final_mean": mean([float(f["answer_occurrences_excluding_final_answer"]) for f in sub]),
            }

    surprising = [f for f in feats if f["role_pred_update_answer_overlap"] != f["role"] or f["role_pred_query_update_overlap"] != f["role"]]
    summary = {
        "export_dir": rel(export_dir),
        "manifest_status": manifest.get("status"),
        "row_kind": row_kind,
        "row_path": rel(row_path),
        "n_rows": len(rows),
        "roles": dict(Counter(f["role"] for f in feats)),
        "relations": dict(Counter(f["relation"] for f in feats)),
        "by_role": by_role,
        "shortcut_role_prediction_accuracy": shortcut_acc,
        "by_relation_role": by_relation_role,
        "surprising_feature_rows_count": len(surprising),
        "surprising_feature_rows_examples": surprising[:20],
        "scientific_reading": (
            "The A/B rows intentionally make the updated row's answer appear in the update sentence and the unchanged row's answer appear in the source sentence. "
            "A model can succeed on in-format rows by learning the recipient/recency rule: if the query entity is the updated entity named in the update sentence, use the new value; otherwise retrieve the source value for the other entity. "
            "This is a legitimate controlled state-update operation, but it is not by itself evidence of broad natural state tracking. Strong interpretation requires wrong-recipient, no-update, source/entity-disjoint held rows, candidate-type controls, and broad BabyLM preservation."
        ),
    }
    out_path = export_dir / f"shortcut_feature_audit_{row_kind}.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary["audit_path"] = rel(out_path)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export-dir", type=pathlib.Path, action="append", default=[])
    ap.add_argument("--row-kind", default="train_frame_all")
    args = ap.parse_args()
    dirs = args.export_dir or DEFAULT_EXPORTS
    results = [audit_export(d, args.row_kind) for d in dirs]
    print(json.dumps({"status": "ASSIGNMENT_EXPORT_SHORTCUT_AUDIT_DONE", "results": results}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
