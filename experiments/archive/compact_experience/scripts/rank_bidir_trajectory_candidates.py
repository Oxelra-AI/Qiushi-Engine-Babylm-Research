#!/usr/bin/env python3
"""Rank bidirectional pair-order no-AoA trajectory screen against clean-Qwen references."""
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

WORKSPACE = _public_path('experiments/archive/compact_experience')
DEFAULT_ROOT = _public_path('experiments/archive/compact_experience/data/bidir_trajectory_screen')
CLEAN_ROOT = _public_path('experiments/archive/compact_experience/data/trajectory_screen')
OUT = _public_path('experiments/archive/compact_experience/data/bidir_trajectory_ranking.json')
NOTE = _public_path('research/notes/compact_experience/bidir_trajectory_ranking.md')
KEYS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]


def load(path: pathlib.Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rows_from_summary(path: pathlib.Path, family: str) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    d = load(path)
    target = str(d.get("target") or path.name.replace("_trajectory_summary.json", ""))
    out = []
    for ckpt, row in (d.get("table") or {}).items():
        if not isinstance(row, dict):
            continue
        rec = {"target": target, "family": family, "checkpoint": ckpt, "equal7_full_eval": row.get("equal7_full_eval")}
        for k in KEYS:
            rec[k] = row.get(k)
        rec["complete"] = all(rec.get(k) is not None for k in KEYS) and rec.get("equal7_full_eval") is not None
        out.append(rec)
    return out


def seed_mean(rows: List[Dict[str, Any]], family: str) -> List[Dict[str, Any]]:
    rr = [r for r in rows if r.get("family") == family and r.get("complete")]
    out = []
    for ckpt in sorted({r["checkpoint"] for r in rr}):
        xs = [r for r in rr if r["checkpoint"] == ckpt]
        if len(xs) >= 2:
            out.append({
                "family": family,
                "checkpoint": ckpt,
                "n": len(xs),
                "mean_equal7_full_eval": mean(float(x["equal7_full_eval"]) for x in xs),
                "targets": [{"target": x["target"], "equal7_full_eval": x["equal7_full_eval"]} for x in sorted(xs, key=lambda x: x["target"])],
            })
    return sorted(out, key=lambda r: float(r["mean_equal7_full_eval"]), reverse=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--clean_root", default=str(CLEAN_ROOT))
    ap.add_argument("--top_k", type=int, default=12)
    args = ap.parse_args()
    root = pathlib.Path(args.root)
    clean_root = pathlib.Path(args.clean_root)
    rows: List[Dict[str, Any]] = []
    for seed in ["43022", "43122"]:
        rows += rows_from_summary(root / f"bidir_seed{seed}_trajectory_summary.json", "bidir_pair_order")
        rows += rows_from_summary(clean_root / f"clean_qwen_seed{seed}_trajectory_summary.json", "clean_qwen")
    complete = [r for r in rows if r.get("complete")]
    ranked = sorted(complete, key=lambda r: float(r["equal7_full_eval"]), reverse=True)
    best_by_target: Dict[str, Dict[str, Any]] = {}
    for target in sorted({r["target"] for r in complete}):
        rr = [r for r in complete if r["target"] == target]
        if rr:
            best_by_target[target] = max(rr, key=lambda x: float(x["equal7_full_eval"]))
    paired = seed_mean(complete, "bidir_pair_order") + seed_mean(complete, "clean_qwen")
    paired = sorted(paired, key=lambda r: float(r["mean_equal7_full_eval"]), reverse=True)
    best_clean = max([r for r in complete if r["family"] == "clean_qwen"], key=lambda x: float(x["equal7_full_eval"]), default=None)
    best_bidir = max([r for r in complete if r["family"] == "bidir_pair_order"], key=lambda x: float(x["equal7_full_eval"]), default=None)
    payload = {
        "status": "BIDIR_TRAJECTORY_RANKING",
        "non_leakage_statement": "Ranks only BLiMP/Supplement/EWoK/Entity/COMPS/GlobalPIQA/Reading; AoA/CDI words, child curves, AoA predictions, AoA outputs, and SuperGLUE are not used.",
        "root": str(root),
        "clean_reference_root": str(clean_root),
        "num_rows": len(rows),
        "num_complete_rows": len(complete),
        "best_bidir": best_bidir,
        "best_clean_reference": best_clean,
        "top_rows": ranked[: args.top_k],
        "best_by_target": best_by_target,
        "paired_seed_means": paired[: args.top_k],
        "recommendation_rule": "Promote best bidirectional checkpoint(s) for full nine-column eval only if no-AoA surface is competitive with clean-Qwen or specifically repairs COMPS/Reading/BLiMP/EWoK without large Supplement/Entity/SuperGLUE loss; AoA only inside frozen full eval.",
    }
    _public_path('experiments/archive/compact_experience/data').mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research bidirectional trajectory ranking", "", payload["non_leakage_statement"], "", f"Complete rows: {len(complete)} / {len(rows)}"]
    if best_bidir:
        lines.append(f"Best bidir: `{best_bidir['target']}` `{best_bidir['checkpoint']}` equal7={float(best_bidir['equal7_full_eval']):.6f}")
    if best_clean:
        lines.append(f"Best clean reference: `{best_clean['target']}` `{best_clean['checkpoint']}` equal7={float(best_clean['equal7_full_eval']):.6f}")
    lines += ["", "## Top rows"]
    for i, r in enumerate(ranked[: args.top_k], 1):
        lines.append(f"{i}. `{r['family']}` `{r['target']}` `{r['checkpoint']}` equal7={float(r['equal7_full_eval']):.6f} BLiMP={r.get('BLiMP')} Supplement={r.get('Supplement')} EWoK={r.get('EWoK')} Entity={r.get('Entity')} COMPS={r.get('COMPS')} GlobalPIQA={r.get('GlobalPIQA')} Reading={r.get('Reading')}")
    lines += ["", "## Paired seed means"]
    for r in paired[: args.top_k]:
        lines.append(f"- `{r['family']}` `{r['checkpoint']}` mean_equal7={float(r['mean_equal7_full_eval']):.6f} n={r['n']} targets={r['targets']}")
    _public_path('research/notes/compact_experience').mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT), "note": str(NOTE), "best_bidir": best_bidir, "best_clean_reference": best_clean, "top": ranked[: min(args.top_k, len(ranked))]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
