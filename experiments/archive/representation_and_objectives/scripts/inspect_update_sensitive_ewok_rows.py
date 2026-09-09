#!/usr/bin/env python3
"""Inspect official EWoK rows used as update-sensitive surfaces in research."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import collections
import csv
import json
import re
import time
from pathlib import Path
from typing import Any

USER_ROOT = _public_path('.')
A01_WS = _public_path('experiments/archive/representation_and_objectives')
OUT_DIR = _public_path('experiments/archive/representation_and_objectives/data/update_sensitive_ewok_inspection')
OUT_JSON = _public_path('experiments/archive/representation_and_objectives/data/update_sensitive_ewok_inspection/update_sensitive_ewok_inspection.json')
OUT_MD = _public_path('research/notes/representation_and_objectives/update_sensitive_ewok_inspection.md')
PATHS = {
    "legal16k_step155": _public_path('experiments/archive/representation_and_objectives/data/scale1p75_matched_ewok_interaction/legal16k_100M/ewok_interaction_records.csv'),
    "legal40_step092": _public_path('experiments/archive/representation_and_objectives/data/full_ewok_interaction_specificity/ewok_full_interaction_specificity_records_fullcpu.csv'),
}
UPDATE_DOMAINS = {"material-dynamics", "physical-dynamics"}
TERMS = [
    "open", "closed", "empty", "full", "fill", "filled", "emptied", "shut", "wet", "dry",
    "broke", "broken", "melt", "melted", "cooked", "raw", "freeze", "frozen", "burn",
    "clean", "dirty", "ripe", "unripe", "old", "new", "fall", "fell", "drop", "dropped",
]


def rel(p: Path) -> str:
    return str(p.resolve().relative_to(USER_ROOT))


def row_text(row: dict[str, str]) -> str:
    keys = [
        "ContextType", "ContextDiff", "TargetDiff", "ConceptA", "ConceptB",
        "context_diff_c1_texts_joined", "context_diff_c2_texts_joined", "target1", "target2",
        "context1_deleted_diff", "context2_deleted_diff", "context1_swapped_diff", "context2_swapped_diff",
    ]
    return " ".join(row.get(k, "") for k in keys)


def inspect_one(name: str, path: Path) -> dict[str, Any]:
    domains = collections.Counter()
    update_context_types = collections.Counter()
    update_context_diffs = collections.Counter()
    update_target_diffs = collections.Counter()
    update_concept_pairs = collections.Counter()
    term_hits = collections.Counter()
    examples = []
    update_rows = 0
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            # research contains two models in one row file; inspect one copy only for content.
            if row.get("model") and row.get("model") != "legal40_8x480_43022":
                continue
            d = row.get("domain", "")
            domains[d] += 1
            if d not in UPDATE_DOMAINS:
                continue
            update_rows += 1
            update_context_types[row.get("ContextType", "")] += 1
            update_context_diffs[row.get("ContextDiff", "")] += 1
            update_target_diffs[row.get("TargetDiff", "")] += 1
            update_concept_pairs[(row.get("ConceptA", ""), row.get("ConceptB", ""))] += 1
            text = row_text(row).lower()
            for term in TERMS:
                if re.search(r"\b" + re.escape(term) + r"\b", text):
                    term_hits[term] += 1
            if len(examples) < 18:
                examples.append({
                    "global_index": row.get("global_index"),
                    "domain": row.get("domain"),
                    "ContextType": row.get("ContextType"),
                    "ContextDiff": row.get("ContextDiff"),
                    "TargetDiff": row.get("TargetDiff"),
                    "ConceptA": row.get("ConceptA"),
                    "ConceptB": row.get("ConceptB"),
                    "context_diff_c1_texts_joined": row.get("context_diff_c1_texts_joined"),
                    "context_diff_c2_texts_joined": row.get("context_diff_c2_texts_joined"),
                    "target1": row.get("target1"),
                    "target2": row.get("target2"),
                })
    return {
        "path": rel(path),
        "domains": dict(domains),
        "update_rows": update_rows,
        "update_context_types": dict(update_context_types.most_common()),
        "update_context_diffs": dict(update_context_diffs.most_common(30)),
        "update_target_diffs": dict(update_target_diffs.most_common(30)),
        "update_concept_pairs_top": [{"concept_a": a, "concept_b": b, "n": n} for (a, b), n in update_concept_pairs.most_common(30)],
        "term_hits": dict(term_hits.most_common()),
        "examples": examples,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "update_domains": sorted(UPDATE_DOMAINS),
        "inspections": {name: inspect_one(name, path) for name, path in PATHS.items()},
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# research update-sensitive EWoK surface inspection",
        "",
        "This file inspects the official EWoK rows used as the research update-sensitive surface link. It is content inspection only; no model inference.",
        "",
    ]
    for name, info in out["inspections"].items():
        lines.extend([
            f"## {name}",
            f"Path: `{info['path']}`",
            f"Update-domain rows: {info['update_rows']}",
            "",
            "Context types: " + json.dumps(info["update_context_types"], ensure_ascii=False),
            "",
            "Top ContextDiff: " + json.dumps(info["update_context_diffs"], ensure_ascii=False),
            "",
            "Top TargetDiff: " + json.dumps(info["update_target_diffs"], ensure_ascii=False),
            "",
            "Term hits: " + json.dumps(info["term_hits"], ensure_ascii=False),
            "",
            "Representative rows:",
            "",
        ])
        for ex in info["examples"][:10]:
            lines.append("```json")
            lines.append(json.dumps(ex, ensure_ascii=False, sort_keys=True))
            lines.append("```")
        lines.append("")
    lines.append(f"JSON: `{rel(OUT_JSON)}`")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "UPDATE_EWOK_INSPECTION_DONE", "out_json": rel(OUT_JSON), "out_md": rel(OUT_MD)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
