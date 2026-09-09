#!/usr/bin/env python3
"""Analyze research concentrated selective compact learner test.

Reads the summary and per-arm eval scores and writes a compact scientific table.  The
metric convention is positive gain = parent NLL - child NLL on the same masked targets.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import json
import pathlib
from typing import Any, Dict, List, Optional

ROOT = _public_path('.')
DEFAULT_RUN = _public_path('experiments/archive/functional_learning/data/concentrated_compact_learning_rerun')


def rel(p: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def load_json(path: pathlib.Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric(summary: Dict[str, Any], view: str, cond: str, field: str = "mean_nll") -> Optional[float]:
    obj = summary.get("by_view_condition", {}).get(f"{view}/{cond}")
    if not obj:
        return None
    v = obj.get(field)
    return float(v) if v is not None else None


def help_metric(summary: Dict[str, Any], view: str) -> Optional[float]:
    obj = summary.get("by_view_source_help", {}).get(view)
    if not obj:
        return None
    v = obj.get("mean_source_help")
    return float(v) if v is not None else None


def safe_div(a: Optional[float], b: float) -> Optional[float]:
    if a is None or b == 0:
        return None
    return a / b


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=pathlib.Path, default=DEFAULT_RUN)
    args = ap.parse_args()
    run_dir = args.run_dir
    summ = load_json(run_dir / "summary.json")
    plan = load_json(run_dir / "plan.json")
    parent = summ["parent_eval_summary"]

    rows: List[Dict[str, Any]] = []
    for arm in summ["arm_summaries"]:
        name = arm["arm"]["name"]
        train_view = arm["arm"]["train_view"]
        es = arm["eval_summary"]
        final_log = arm.get("final_train_log") or {}
        charged = int(final_log.get("charged_row_words_cum", 0))
        view_words = int(final_log.get("view_words_cum", 0))
        row: Dict[str, Any] = {
            "arm": name,
            "train_view": train_view,
            "epochs": arm["completed_epochs"],
            "charged_row_words": charged,
            "view_words": view_words,
        }
        for view in ["current", "compact"]:
            for cond in ["with_source", "view_only"]:
                p = metric(parent, view, cond)
                c = metric(es, view, cond)
                gain = None if p is None or c is None else p - c
                row[f"{view}_{cond}_parent_nll"] = p
                row[f"{view}_{cond}_child_nll"] = c
                row[f"{view}_{cond}_gain"] = gain
        for view in ["current", "compact"]:
            ph = help_metric(parent, view)
            ch = help_metric(es, view)
            row[f"{view}_source_help_parent"] = ph
            row[f"{view}_source_help_child"] = ch
            row[f"{view}_source_help_shift"] = None if ph is None or ch is None else ch - ph
        trained_gain = row.get(f"{train_view}_with_source_gain")
        trained_gain_vo = row.get(f"{train_view}_view_only_gain")
        cross_view = "compact" if train_view == "current" else "current"
        cross_gain = row.get(f"{cross_view}_with_source_gain")
        row["trained_view_with_source_gain"] = trained_gain
        row["trained_view_view_only_gain"] = trained_gain_vo
        row["cross_view_with_source_gain"] = cross_gain
        row["trained_gain_per_10k_charged_words"] = safe_div(trained_gain, charged / 10000.0)
        row["trained_gain_per_10k_view_words"] = safe_div(trained_gain, view_words / 10000.0)
        rows.append(row)

    out_csv = run_dir / "concentrated_gain_table.csv"
    fields = list(rows[0].keys()) if rows else []
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

    # Compact comparison statements.
    by_name = {r["arm"]: r for r in rows}
    current = by_name.get("current_equal_epoch", {})
    compact_eq = by_name.get("compact_equal_epoch", {})
    compact_wm = by_name.get("compact_wordmatched", {})
    comparison = {
        "status": "CONCENTRATED_COMPACT_ANALYZED",
        "run_dir": rel(run_dir),
        "table_csv": rel(out_csv),
        "sample": {
            "reviewed_label_counts": plan.get("reviewed_label_counts"),
            "n_faithful_reviewed": plan.get("n_faithful_reviewed"),
            "n_common_train_pairs": plan.get("n_common_train_pairs"),
            "current_row_words_per_epoch": plan.get("current_row_words_per_epoch_common"),
            "compact_row_words_per_epoch": plan.get("compact_row_words_per_epoch_common"),
            "saved_words_per_epoch": plan.get("realized_saved_words_per_epoch_common"),
            "eval_tasks": plan.get("eval_stats", {}).get("n_tasks"),
        },
        "key_metrics": {
            "current_equal_current_with_source_gain": current.get("current_with_source_gain"),
            "current_equal_compact_with_source_cross_gain": current.get("compact_with_source_gain"),
            "compact_equal_compact_with_source_gain": compact_eq.get("compact_with_source_gain"),
            "compact_equal_current_with_source_cross_gain": compact_eq.get("current_with_source_gain"),
            "compact_wordmatched_compact_with_source_gain": compact_wm.get("compact_with_source_gain"),
            "compact_wordmatched_current_with_source_cross_gain": compact_wm.get("current_with_source_gain"),
            "current_equal_current_source_help_shift": current.get("current_source_help_shift"),
            "compact_equal_compact_source_help_shift": compact_eq.get("compact_source_help_shift"),
            "compact_wordmatched_compact_source_help_shift": compact_wm.get("compact_source_help_shift"),
            "compact_wordmatched_minus_compact_equal_compact_gain": None if not compact_wm or not compact_eq else compact_wm.get("compact_with_source_gain") - compact_eq.get("compact_with_source_gain"),
        },
        "interpretation": [
            "Each arm mainly improves the view it trained on; cross-view gains are much smaller, so the first signal is view-specific acquisition rather than broad reusable semantic transfer.",
            "The compact view has stronger parent source-conditioned reconstruction than the inherited current view on this reviewed subset; compact training preserves and slightly increases that source help.",
            "At roughly matched charged words, compact_wordmatched improves compact-with-source NLL slightly more than current_equal improves current-with-source NLL, but target sets differ and the sample is only 17 pairs, so this is a promising bounded signal rather than a model-improvement result.",
            "The result supports running a larger reviewed subset and a more diagnostic held-source/content-probe before committing to full legal tail training; it does not certify unreviewed automatic rows.",
        ],
    }
    out_json = run_dir / "concentrated_analysis_summary.json"
    out_json.write_text(json.dumps(comparison, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(comparison, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
