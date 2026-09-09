#!/usr/bin/env python3
"""Rank research/036 no-AoA checkpoint trajectory summaries.

Reads *_trajectory_summary.json files produced by eval_checkpoint_trajectory_fullzeroshot.py,
compares clean-Qwen and developmental-order families across checkpoints, and emits a compact
candidate list for expensive corrected full nine-column evaluation. This script does not read or
use AoA/CDI words, AoA predictions, child curves, or AoA scores; it ranks only by seven full
zero-shot/Reading columns.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import argparse
import json
import pathlib
from statistics import mean
from typing import Any, Dict, List

DEFAULT_ROOT = _public_path('experiments/archive/compact_experience/data/trajectory_screen')
OUT = _public_path('experiments/archive/compact_experience/data/trajectory_candidate_ranking.json')
NOTE = _public_path('research/notes/compact_experience/trajectory_screen_candidate_ranking.md')
KEYS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
FAMILIES = {
    "clean_qwen_seed43022": "clean_qwen",
    "clean_qwen_seed43122": "clean_qwen",
    "devcurr_seed43022": "devcurr_firstpass",
    "devcurr_seed43122": "devcurr_firstpass",
}


def load(path: pathlib.Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def row_record(target: str, ckpt: str, row: Dict[str, Any]) -> Dict[str, Any]:
    rec: Dict[str, Any] = {
        "target": target,
        "family": FAMILIES.get(target, target),
        "checkpoint": ckpt,
        "equal7_full_eval": row.get("equal7_full_eval"),
    }
    for k in KEYS:
        rec[k] = row.get(k)
    rec["complete"] = all(rec.get(k) is not None for k in KEYS) and rec.get("equal7_full_eval") is not None
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--top_k", type=int, default=12)
    args = ap.parse_args()
    root = pathlib.Path(args.root)
    rows: List[Dict[str, Any]] = []
    summaries = sorted(root.glob("*_trajectory_summary.json"))
    for p in summaries:
        data = load(p)
        target = str(data.get("target") or p.name.replace("_trajectory_summary.json", ""))
        table = data.get("table") or {}
        for ckpt, row in table.items():
            if isinstance(row, dict):
                rows.append(row_record(target, ckpt, row))
    complete = [r for r in rows if r.get("complete")]
    ranked = sorted(complete, key=lambda r: float(r["equal7_full_eval"]), reverse=True)

    # Per-target and per-family views.
    by_target: Dict[str, Dict[str, Any]] = {}
    for target in sorted({r["target"] for r in complete}):
        rr = [r for r in complete if r["target"] == target]
        if rr:
            by_target[target] = max(rr, key=lambda r: float(r["equal7_full_eval"]))
    by_family: Dict[str, Dict[str, Any]] = {}
    for fam in sorted({r["family"] for r in complete}):
        rr = [r for r in complete if r["family"] == fam]
        if rr:
            by_family[fam] = max(rr, key=lambda r: float(r["equal7_full_eval"]))

    # Seed-paired checkpoint means where both seeds exist for a family/checkpoint.
    paired: List[Dict[str, Any]] = []
    for fam in sorted({r["family"] for r in complete}):
        fam_rows = [r for r in complete if r["family"] == fam]
        ckpts = sorted({r["checkpoint"] for r in fam_rows})
        for ckpt in ckpts:
            rr = [r for r in fam_rows if r["checkpoint"] == ckpt]
            if len(rr) >= 2:
                paired.append({
                    "family": fam,
                    "checkpoint": ckpt,
                    "n": len(rr),
                    "mean_equal7_full_eval": mean(float(x["equal7_full_eval"]) for x in rr),
                    "targets": [{"target": x["target"], "equal7_full_eval": x["equal7_full_eval"]} for x in sorted(rr, key=lambda x: x["target"])],
                })
    paired = sorted(paired, key=lambda r: float(r["mean_equal7_full_eval"]), reverse=True)

    payload = {
        "status": "TRAJECTORY_CANDIDATE_RANKING",
        "non_leakage_statement": "Ranks only BLiMP/Supplement/EWoK/Entity/COMPS/GlobalPIQA/Reading trajectory summaries; no AoA/CDI words, child curves, AoA predictions, AoA outputs, or SuperGLUE are used.",
        "root": str(root),
        "summary_files": [str(p) for p in summaries],
        "num_rows": len(rows),
        "num_complete_rows": len(complete),
        "top_rows": ranked[: args.top_k],
        "best_by_target": by_target,
        "best_by_family": by_family,
        "paired_seed_means": paired[: args.top_k],
        "missing_or_incomplete": [r for r in rows if not r.get("complete")],
        "recommendation_rule": "Run corrected full nine-column eval on the best no-AoA rows, prioritizing any row whose equal7_full_eval exceeds the known clean-Qwen 100M no-AoA row and any seed-paired family/checkpoint mean that improves over baseline; use AoA only inside the final full evaluation.",
    }
    _public_path('experiments/archive/compact_experience/data').mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# research trajectory candidate ranking",
        "",
        payload["non_leakage_statement"],
        "",
        f"Complete rows: {len(complete)} / {len(rows)}",
        "",
        "## Top no-AoA rows",
    ]
    for i, r in enumerate(ranked[: args.top_k], 1):
        lines.append(f"{i}. `{r['target']}` `{r['checkpoint']}` equal7={float(r['equal7_full_eval']):.6f} BLiMP={r.get('BLiMP')} Supplement={r.get('Supplement')} EWoK={r.get('EWoK')} Entity={r.get('Entity')} COMPS={r.get('COMPS')} GlobalPIQA={r.get('GlobalPIQA')} Reading={r.get('Reading')}")
    lines += ["", "## Best by target"]
    for target, r in by_target.items():
        lines.append(f"- `{target}` best `{r['checkpoint']}` equal7={float(r['equal7_full_eval']):.6f}")
    lines += ["", "## Paired seed means"]
    for r in paired[: args.top_k]:
        lines.append(f"- `{r['family']}` `{r['checkpoint']}` mean_equal7={float(r['mean_equal7_full_eval']):.6f} n={r['n']} targets={r['targets']}")
    if payload["missing_or_incomplete"]:
        lines += ["", "## Incomplete rows"]
        for r in payload["missing_or_incomplete"][:50]:
            lines.append(f"- `{r['target']}` `{r['checkpoint']}` keys_present={[k for k in KEYS if r.get(k) is not None]} equal7={r.get('equal7_full_eval')}")
    _public_path('research/notes/compact_experience').mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT), "note": str(NOTE), "top": ranked[: min(5, len(ranked))]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
