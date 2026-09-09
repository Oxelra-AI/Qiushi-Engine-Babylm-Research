#!/usr/bin/env python3
"""Parse official report files for research vs adapter scale1.75 at 20/30/40/50M.

Focuses on the columns that explain the score tradeoff: BLiMP, Supplement, EWoK,
Entity, COMPS, and GlobalPIQA split scores. This is CPU-only and reads already
produced official-compatible reports.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

USER_ROOT = _public_path('.')
DATA = _public_path('experiments/archive/frontier_consolidation/data')
OUT = _public_path('experiments/archive/frontier_consolidation/data/subtask_report_trajectory')
EXPOSURES = ["20M", "30M", "40M", "50M"]
ARMS = ["research", "scale1p75"]
COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]

# Per-target JSON locations by arm/exposure.
PER_TARGETS = {
    ("research", "20M"): _public_path('experiments/archive/frontier_consolidation/data/adapter_matched_horizon_eval/per_target/adapter128_disabled_h100M20M_seed43022.json'),
    ("scale1p75", "20M"): _public_path('experiments/archive/frontier_consolidation/data/scaled_train_20M_eval_s1p75/eval/per_target/adapter128_scale1p75_h100M20M_seed43022.json'),
    ("research", "30M"): _public_path('experiments/archive/frontier_consolidation/data/trajectory_eval_30_40/chck_30M/eval/per_target/legal_chck_30M.json'),
    ("scale1p75", "30M"): _public_path('experiments/archive/frontier_consolidation/data/trajectory_eval_scale1p75_30_40/chck_30M/eval/per_target/scale1p75_chck_30M.json'),
    ("research", "40M"): _public_path('experiments/archive/frontier_consolidation/data/trajectory_eval_30_40/chck_40M/eval/per_target/legal_chck_40M.json'),
    ("scale1p75", "40M"): _public_path('experiments/archive/frontier_consolidation/data/trajectory_eval_scale1p75_30_40/chck_40M/eval/per_target/scale1p75_chck_40M.json'),
    ("research", "50M"): _public_path('experiments/archive/frontier_consolidation/data/50M_eval/eval/per_target/legal_chck50M.json'),
    ("scale1p75", "50M"): _public_path('experiments/archive/frontier_consolidation/data/scale1p75_50M_eval/eval/per_target/adapter128_scale1p75_h100M50M_seed43022.json'),
}


def load(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def parse_report(path: Path) -> Dict[str, float]:
    txt = path.read_text(encoding="utf-8", errors="replace")
    out: Dict[str, float] = {}
    for line in txt.splitlines():
        line = line.strip()
        m = re.match(r"^([^:#]+):\s*(-?\d+(?:\.\d+)?)\s*$", line)
        if not m:
            continue
        key = m.group(1).strip()
        if key.upper() in {"TEMPERATURE"}:
            continue
        out[key] = float(m.group(2))
    return out


def column_reports(payload: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    tasks = payload.get("tasks", {})
    for col in COLUMNS:
        rec = tasks.get(col)
        if not rec:
            continue
        if col.startswith("GlobalPIQA"):
            # GlobalPIQA report often has only the aggregate score; keep explicit score.
            out[col] = {"score": float(rec.get("score")) if rec.get("score") is not None else None}
            continue
        rp = rec.get("report")
        if rp and Path(rp).exists():
            out[col] = parse_report(Path(rp))
        elif rec.get("score") is not None:
            out[col] = {"score": float(rec["score"])}
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reports: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {arm: {} for arm in ARMS}
    missing = []
    for arm in ARMS:
        for exp in EXPOSURES:
            payload = load(PER_TARGETS[(arm, exp)])
            if payload is None:
                missing.append(str(PER_TARGETS[(arm, exp)]))
                continue
            reports[arm][exp] = column_reports(payload)

    deltas: Dict[str, Dict[str, Dict[str, float]]] = {}
    for exp in EXPOSURES:
        a = reports["research"].get(exp)
        b = reports["scale1p75"].get(exp)
        if not a or not b:
            continue
        deltas[exp] = {}
        for col in sorted(set(a) & set(b)):
            keys = sorted(set(a[col]) & set(b[col]))
            deltas[exp][col] = {k: b[col][k] - a[col][k] for k in keys if a[col][k] is not None and b[col][k] is not None}

    # Summarize the strongest positive/negative subtask changes at 50M and trajectories for key subtasks.
    top50 = []
    if "50M" in deltas:
        for col, vals in deltas["50M"].items():
            for k, v in vals.items():
                if k.lower().startswith("average") or k == "score":
                    continue
                top50.append({"column": col, "subtask": k, "delta": v})
        top50.sort(key=lambda x: x["delta"])
    tracked_keys = [
        ("EWoK", "physical-dynamics"), ("EWoK", "material-properties"), ("EWoK", "quantitative-properties"),
        ("EWoK", "active-passive"), ("EWoK", "number"), ("Entity", "regular_4_ops"), ("Entity", "regular_5_ops"),
        ("Entity", "move_contents_5_ops"), ("Entity", "ambiref_4_ops"), ("Entity", "ambiref_5_ops"),
        ("GlobalPIQA_parallel", "score"), ("GlobalPIQA_nonparallel", "score"),
    ]
    tracked = []
    for col, sub in tracked_keys:
        row = {"column": col, "subtask": sub}
        for exp in EXPOSURES:
            row[exp] = deltas.get(exp, {}).get(col, {}).get(sub)
        tracked.append(row)

    out = {"status": "SUBTASK_REPORT_TRAJECTORY", "missing": missing, "reports": reports, "deltas": deltas, "top50_negative": top50[:20], "top50_positive": top50[-20:][::-1], "tracked": tracked}
    out_json = _public_path('experiments/archive/frontier_consolidation/data/subtask_report_trajectory/subtask_report_trajectory.json')
    out_md = _public_path('experiments/archive/frontier_consolidation/data/subtask_report_trajectory/subtask_report_trajectory.md')
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = ["# research subtask report trajectory", "", f"Missing per-target files: {len(missing)}", "", "## 50M largest negative subtask deltas", "", "| column | subtask | delta |", "|---|---|---:|"]
    for r in top50[:15]:
        lines.append(f"| {r['column']} | {r['subtask']} | {r['delta']:+.2f} |")
    lines += ["", "## 50M largest positive subtask deltas", "", "| column | subtask | delta |", "|---|---|---:|"]
    for r in top50[-15:][::-1]:
        lines.append(f"| {r['column']} | {r['subtask']} | {r['delta']:+.2f} |")
    lines += ["", "## Selected delta trajectories", "", "| column | subtask | 20M | 30M | 40M | 50M |", "|---|---|---:|---:|---:|---:|"]
    for r in tracked:
        def fmt(x): return "" if x is None else f"{x:+.2f}"
        lines.append(f"| {r['column']} | {r['subtask']} | {fmt(r['20M'])} | {fmt(r['30M'])} | {fmt(r['40M'])} | {fmt(r['50M'])} |")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "out_json": str(out_json), "out_md": str(out_md), "missing": len(missing), "n_50m_subtasks": len(top50)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
