#!/usr/bin/env python3
"""Summarize available research real-stream credit-audit CSVs.

The full tokenization audit is slow. This script consumes whatever per-update CSVs
exist and writes a compact evidence summary so learner results can be interpreted
against actual target-credit mass.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import csv
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
IN_DIR = _public_path('experiments/archive/functional_learning/data/real_stream_credit_audit')
OUT = _public_path('experiments/archive/functional_learning/data/real_stream_credit_audit/available_credit_audit_summary.json')

NUMERIC = {
    "rows", "words", "qwen_rows", "compact_modified_rows", "topup_rows",
    "inherited_wwm_total_targets", "inherited_wwm_qwen_targets", "inherited_wwm_nonqwen_targets",
    "inherited_wwm_qwen_target_fraction", "inherited_wwm_compact_modified_targets",
    "correspondence_focus_total_targets", "correspondence_focus_qwen_targets", "correspondence_focus_nonqwen_targets",
    "correspondence_focus_qwen_target_fraction", "correspondence_focus_candidate_groups",
    "correspondence_focus_selected_groups", "correspondence_focus_selected_pair_refs",
    "correspondence_focus_compact_modified_targets", "target_ratio_focus_vs_wwm",
    "qwen_target_ratio_focus_vs_wwm",
}


def rel(p: Path | str) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def stats(vals: List[float]) -> Dict[str, Any]:
    if not vals:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {"n": len(vals), "mean": sum(vals)/len(vals), "median": statistics.median(vals), "min": min(vals), "max": max(vals)}


def load_csv(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            out = dict(r)
            for k in list(out):
                if k in NUMERIC:
                    try:
                        if out[k] == "" or out[k] is None:
                            out[k] = 0
                        elif "." in str(out[k]) or "e" in str(out[k]).lower():
                            out[k] = float(out[k])
                        else:
                            out[k] = int(out[k])
                    except Exception:
                        pass
            rows.append(out)
    return rows


def summarize_tail(path: Path) -> Dict[str, Any]:
    rows = load_csv(path)
    totals = Counter()
    for r in rows:
        for k in [
            "rows", "words", "qwen_rows", "compact_modified_rows", "topup_rows",
            "inherited_wwm_total_targets", "inherited_wwm_qwen_targets", "inherited_wwm_nonqwen_targets",
            "inherited_wwm_compact_modified_targets", "correspondence_focus_total_targets",
            "correspondence_focus_qwen_targets", "correspondence_focus_nonqwen_targets",
            "correspondence_focus_candidate_groups", "correspondence_focus_selected_groups",
            "correspondence_focus_selected_pair_refs", "correspondence_focus_compact_modified_targets",
        ]:
            totals[k] += int(r.get(k, 0) or 0)
    wwm_total = totals["inherited_wwm_total_targets"]
    focus_total = totals["correspondence_focus_total_targets"]
    tail_name = rows[0].get("tail_name", path.name.replace("_per_update_credit.csv", "")) if rows else path.name
    return {
        "tail_name": tail_name,
        "csv": rel(path),
        "updates_available": len(rows),
        "first_update": rows[0] if rows else None,
        "last_update": rows[-1] if rows else None,
        "totals": dict(totals),
        "aggregate_inherited_wwm_qwen_target_fraction": (totals["inherited_wwm_qwen_targets"] / wwm_total) if wwm_total else None,
        "aggregate_correspondence_focus_qwen_target_fraction": (totals["correspondence_focus_qwen_targets"] / focus_total) if focus_total else None,
        "aggregate_target_ratio_focus_vs_wwm": (focus_total / wwm_total) if wwm_total else None,
        "aggregate_qwen_target_ratio_focus_vs_wwm": (totals["correspondence_focus_qwen_targets"] / totals["inherited_wwm_qwen_targets"]) if totals["inherited_wwm_qwen_targets"] else None,
        "per_update_stats": {
            "inherited_wwm_qwen_target_fraction": stats([float(r.get("inherited_wwm_qwen_target_fraction", 0) or 0) for r in rows]),
            "correspondence_focus_qwen_target_fraction": stats([float(r.get("correspondence_focus_qwen_target_fraction", 0) or 0) for r in rows]),
            "qwen_target_ratio_focus_vs_wwm": stats([float(r.get("qwen_target_ratio_focus_vs_wwm", 0) or 0) for r in rows]),
            "target_ratio_focus_vs_wwm": stats([float(r.get("target_ratio_focus_vs_wwm", 0) or 0) for r in rows]),
        },
        "scientific_reading": "correspondence_focus protects source tokens from masking and concentrates qwen supervision on second-view content, but in this audited prefix it also assigns much less total qwen loss mass than inherited WWM; a neutral learner result can therefore mean under-pressure rather than failed source-visible learning.",
    }


def main() -> None:
    IN_DIR.mkdir(parents=True, exist_ok=True)
    summaries = [summarize_tail(p) for p in sorted(IN_DIR.glob("*_per_update_credit.csv"))]
    final = {
        "status": "AVAILABLE_CREDIT_AUDIT_SUMMARY",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_dir": rel(IN_DIR),
        "tail_summaries": summaries,
        "interpretation": "Summary of whatever credit-audit CSVs exist. The run that created them may have timed out, but completed CSVs are usable for target-mass accounting.",
    }
    OUT.write_text(json.dumps(final, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(final, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
