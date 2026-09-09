#!/usr/bin/env python3
"""research: build a focused EWoK row list for official-compatible margin scoring.

Uses the same four official prediction files as research, but selects a balanced
subset rather than the first 80 negative rows.  The subset concentrates on the
relation domains where compact-view treatment-by-seed interaction was worst,
while retaining stable/control rows for interpreting margin scale.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import csv
import json
from pathlib import Path
from typing import Any


def find_user_root() -> Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
A01 = USER_ROOT / "experiments/archive/representation_and_objectives"
A02 = USER_ROOT / "experiments/archive/frontier_consolidation"
OUT_DIR = A01 / "data/official_ewok_margin_subset"
GOLD_DIR = A01 / "data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/ewok_filtered"
PRED_PATHS = {
    "clean430": A02 / "data/official_ewok_clean_qwen/official_outputs/clean_qwen_seed43022/EWoK/chck_100M/official_ewok_clean_qwen_seed43022/zero_shot/mlm/ewok/ewok_filtered/predictions.json",
    "clean431": A02 / "data/official_ewok_clean_qwen/official_outputs/clean_qwen_seed43122/EWoK/chck_100M/official_ewok_clean_qwen_seed43122/zero_shot/mlm/ewok/ewok_filtered/predictions.json",
    "reinv430": A01 / "data/official_ewok_reeval/official_outputs/EWoK/chck_100M/official_ewok_reinvest/zero_shot/mlm/ewok/ewok_filtered/predictions.json",
    "reinv431": A01 / "data/official_ewok_reeval_seed43122/official_outputs/EWoK/chck_100M/official_ewok_seed43122/zero_shot/mlm/ewok/ewok_filtered/predictions.json",
}

WORST_RELATION_DOMAINS = [
    "material-dynamics",
    "physical-dynamics",
    "spatial-relations",
    "physical-interactions",
    "social-relations",
]
CONTROL_DOMAINS = [
    "agent-properties",
    "material-properties",
    "social-interactions",
]


def load_items() -> list[dict[str, Any]]:
    preds = {k: json.loads(p.read_text(encoding="utf-8")) for k, p in PRED_PATHS.items()}
    items: list[dict[str, Any]] = []
    for gold_path in sorted(GOLD_DIR.glob("*.jsonl")):
        domain = gold_path.stem
        gold_lines = [json.loads(line) for line in gold_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        pred_lists = {k: preds[k][domain]["predictions"] for k in preds}
        for idx, data in enumerate(gold_lines):
            correct_sentence = " ".join([data["Context1"], data["Target1"]]).strip()
            row: dict[str, Any] = {
                "domain": domain,
                "idx": idx,
                "uid": f"{domain}_{idx}",
                "ConceptA": data.get("ConceptA"),
                "ConceptB": data.get("ConceptB"),
                "ContextType": data.get("ContextType"),
                "ContextDiff": data.get("ContextDiff"),
                "TargetDiff": data.get("TargetDiff"),
            }
            for cell, plist in pred_lists.items():
                pr = plist[idx]["pred"].strip()
                row[cell + "_correct"] = int(pr == correct_sentence)
            row["TE430"] = row["reinv430_correct"] - row["clean430_correct"]
            row["TE431"] = row["reinv431_correct"] - row["clean431_correct"]
            row["DiD_item"] = row["TE431"] - row["TE430"]
            row["pattern"] = "".join(str(row[c + "_correct"]) for c in ["clean430", "reinv430", "clean431", "reinv431"])
            items.append(row)
    return items


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    items = load_items()
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()

    def add_rows(rows: list[dict[str, Any]], reason: str, limit: int | None = None) -> int:
        n = 0
        for r in rows:
            key = (r["domain"], int(r["idx"]))
            if key in seen:
                continue
            rr = dict(r)
            rr["selection_reason"] = reason
            selected.append(rr)
            seen.add(key)
            n += 1
            if limit is not None and n >= limit:
                break
        return n

    counts: dict[str, dict[str, int]] = {}
    for dom in WORST_RELATION_DOMAINS:
        dom_rows = [r for r in items if r["domain"] == dom]
        counts[dom] = {}
        # Include all strongest negative treatment-by-seed conflicts.
        counts[dom]["all_DiD_neg2"] = add_rows(
            sorted([r for r in dom_rows if r["DiD_item"] == -2], key=lambda r: (r["ContextType"] or "", r["ContextDiff"] or "", r["TargetDiff"] or "", r["idx"])),
            f"{dom}:all_DiD_neg2",
            None,
        )
        # Add a bounded slice of weaker negative conflicts from each worst domain.
        counts[dom]["sample_DiD_neg1"] = add_rows(
            sorted([r for r in dom_rows if r["DiD_item"] == -1], key=lambda r: (r["ContextType"] or "", r["ContextDiff"] or "", r["TargetDiff"] or "", r["idx"])),
            f"{dom}:sample_DiD_neg1",
            30,
        )
        # Stable easy/hard rows define margin scale in the same relation domains.
        counts[dom]["stable_1111"] = add_rows(
            sorted([r for r in dom_rows if r["pattern"] == "1111"], key=lambda r: r["idx"]),
            f"{dom}:stable_1111_control",
            10,
        )
        counts[dom]["stable_0000"] = add_rows(
            sorted([r for r in dom_rows if r["pattern"] == "0000"], key=lambda r: r["idx"]),
            f"{dom}:stable_0000_control",
            10,
        )

    for dom in CONTROL_DOMAINS:
        dom_rows = [r for r in items if r["domain"] == dom]
        counts[dom] = {}
        counts[dom]["stable_1111"] = add_rows(
            sorted([r for r in dom_rows if r["pattern"] == "1111"], key=lambda r: r["idx"]),
            f"{dom}:stable_1111_crossdomain_control",
            15,
        )
        counts[dom]["stable_0000"] = add_rows(
            sorted([r for r in dom_rows if r["pattern"] == "0000"], key=lambda r: r["idx"]),
            f"{dom}:stable_0000_crossdomain_control",
            15,
        )
        counts[dom]["positive_or_neutral_DiD"] = add_rows(
            sorted([r for r in dom_rows if r["DiD_item"] >= 1], key=lambda r: (-r["DiD_item"], r["idx"])),
            f"{dom}:positive_or_neutral_DiD_control",
            15,
        )

    out_csv = OUT_DIR / "ewok_margin_focus_selection.csv"
    fields = [
        "uid", "domain", "idx", "selection_reason", "ConceptA", "ConceptB", "ContextType", "ContextDiff", "TargetDiff",
        "clean430_correct", "reinv430_correct", "clean431_correct", "reinv431_correct", "TE430", "TE431", "DiD_item", "pattern",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in selected:
            w.writerow({k: r.get(k) for k in fields})

    by_domain: dict[str, int] = {}
    by_reason: dict[str, int] = {}
    by_pattern: dict[str, int] = {}
    for r in selected:
        by_domain[r["domain"]] = by_domain.get(r["domain"], 0) + 1
        by_reason[r["selection_reason"]] = by_reason.get(r["selection_reason"], 0) + 1
        by_pattern[r["pattern"]] = by_pattern.get(r["pattern"], 0) + 1
    payload = {
        "status": "EWOK_MARGIN_FOCUS_SELECTION",
        "selection_csv": str(out_csv),
        "n_selected": len(selected),
        "worst_relation_domains": WORST_RELATION_DOMAINS,
        "control_domains": CONTROL_DOMAINS,
        "selection_counts": counts,
        "selected_by_domain": dict(sorted(by_domain.items())),
        "selected_by_reason": dict(sorted(by_reason.items())),
        "selected_by_pattern": dict(sorted(by_pattern.items())),
        "purpose": "Target official-compatible margin scoring at material/physical/spatial/social relation rows where treatment-by-seed interaction was strongest, with within-domain and cross-domain stable controls.",
    }
    out_json = OUT_DIR / "ewok_margin_focus_selection_summary.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
