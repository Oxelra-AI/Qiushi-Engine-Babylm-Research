#!/usr/bin/env python3
"""Parse 20/30/40/50/70/80M subtask report trajectories for scale1.75 vs research.

Uses official evaluator best_temperature_report.txt/report.txt files already written by
research/104/105 evaluations. If 70/80M adapter reports are not present yet, writes a
pending partial trajectory rather than failing.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

USER_ROOT = _public_path('.')
W = _public_path('experiments/archive/frontier_consolidation')
DATA = _public_path('experiments/archive/frontier_consolidation/data')
OUT = _public_path('experiments/archive/frontier_consolidation/data/subtask_late_trajectory')
REPORT_BLOCK_RE = re.compile(r"^###\s+(.+?)\s*$")
LINE_RE = re.compile(r"^(.+?):\s*(-?\d+(?:\.\d+)?)\s*$")

SELECTED = [
    ("EWoK", "physical-dynamics"),
    ("EWoK", "material-properties"),
    ("EWoK", "quantitative-properties"),
    ("EWoK", "active-passive"),
    ("EWoK", "number"),
    ("EWoK", "game"),
    ("EWoK", "social-interactions"),
    ("Entity", "regular_4_ops"),
    ("Entity", "regular_5_ops"),
    ("Entity", "move_contents_5_ops"),
    ("Entity", "ambiref_4_ops"),
    ("Entity", "ambiref_5_ops"),
    ("BLiMP", "superlative_quantifiers_1"),
    ("BLiMP", "wh_questions_object_gap"),
    ("BLiMP", "superlative_quantifiers_2"),
    ("BLiMP", "quantifiers"),
    ("Supplement", "qa_congruence_tricky"),
    ("Supplement", "subject_aux_inversion"),
    ("COMPS", "wugs_dist_in_between"),
    ("GlobalPIQA_parallel", "score"),
    ("GlobalPIQA_nonparallel", "score"),
]


def parse_report(path: Path) -> Optional[Dict[str, float]]:
    if not path.exists():
        return None
    out: Dict[str, float] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("TEMPERATURE"):
            continue
        m = LINE_RE.match(line)
        if not m:
            continue
        key = m.group(1).strip()
        val = float(m.group(2))
        if key == "AVERAGE ACCURACY":
            key = "score"
        out[key] = val
    return out


def report_path(arm: str, exposure: str, col: str) -> Optional[Path]:
    ep = f"chck_{exposure}"
    if arm == "research":
        if exposure in {"20M", "30M", "40M"}:
            # research trajectory evaluator names.
            base = _public_path('experiments/archive/frontier_consolidation/data/trajectory_eval_30_40')
            if exposure == "20M":
                # 20M came from research matched-horizon disabled/research eval.
                candidates = list((_public_path('experiments/archive/frontier_consolidation/data/adapter_matched_horizon_eval')).glob(f"**/{col}/chck_20M/**/best_temperature_report.txt"))
                if col == "Reading":
                    candidates = list((_public_path('experiments/archive/frontier_consolidation/data/adapter_matched_horizon_eval')).glob("**/Reading/chck_20M/**/report.txt"))
                if candidates:
                    return candidates[0]
                return None
            candidates = list((base / ep).glob(f"**/{col}/{ep}/**/best_temperature_report.txt"))
            if col == "Reading":
                candidates = list((base / ep).glob(f"**/{col}/{ep}/**/report.txt"))
            return candidates[0] if candidates else None
        if exposure == "50M":
            root = _public_path('experiments/archive/frontier_consolidation/data/50M_eval/eval/official_outputs/legal_chck50M')
        elif exposure in {"70M", "80M"}:
            root = _public_path('experiments/archive/frontier_consolidation/data/legal_mature_treatment_effect_eval/official_outputs') / f"complianttok_reinvest_seed43022_{exposure}"
        else:
            return None
    else:
        if exposure == "20M":
            root = _public_path('experiments/archive/frontier_consolidation/data/scaled_train_20M_eval_s1p75/eval/official_outputs/adapter128_scale1p75_h100M20M_seed43022')
        elif exposure in {"30M", "40M"}:
            base = _public_path('experiments/archive/frontier_consolidation/data/trajectory_eval_scale1p75_30_40')
            candidates = list((base / ep).glob(f"**/{col}/{ep}/**/best_temperature_report.txt"))
            if col == "Reading":
                candidates = list((base / ep).glob(f"**/{col}/{ep}/**/report.txt"))
            return candidates[0] if candidates else None
        elif exposure == "50M":
            root = _public_path('experiments/archive/frontier_consolidation/data/scale1p75_50M_eval/eval/official_outputs/adapter128_scale1p75_h100M50M_seed43022')
        elif exposure in {"70M", "80M"}:
            root = _public_path('experiments/archive/frontier_consolidation/data/scale1p75_70_80_eval') / ep / "eval/official_outputs" / f"adapter128_scale1p75_{ep}"
        else:
            return None
    candidates = list(root.glob(f"{col}/{ep}/**/best_temperature_report.txt"))
    if col == "Reading":
        candidates = list(root.glob(f"{col}/{ep}/**/report.txt"))
    return candidates[0] if candidates else None


def read_reports() -> Tuple[Dict[str, Dict[str, Dict[str, Dict[str, float]]]], list[dict]]:
    reports: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {"research": {}, "scale1p75": {}}
    missing = []
    cols = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]
    exposures = ["20M", "30M", "40M", "50M", "70M", "80M"]
    for arm in ["research", "scale1p75"]:
        for exp in exposures:
            reports[arm].setdefault(exp, {})
            for col in cols:
                p = report_path(arm, exp, col)
                rec = parse_report(p) if p else None
                if rec is None:
                    missing.append({"arm": arm, "exposure": exp, "column": col, "path": None if p is None else str(p)})
                else:
                    reports[arm][exp][col] = rec
    return reports, missing


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reports, missing = read_reports()
    deltas: Dict[str, Dict[str, Dict[str, float]]] = {}
    largest_pos = {}
    largest_neg = {}
    for exp in ["20M", "30M", "40M", "50M", "70M", "80M"]:
        deltas[exp] = {}
        items = []
        for col, ref_report in reports["research"].get(exp, {}).items():
            ad_report = reports["scale1p75"].get(exp, {}).get(col)
            if not ad_report:
                continue
            common = sorted(set(ref_report) & set(ad_report))
            dd = {k: ad_report[k] - ref_report[k] for k in common}
            deltas[exp][col] = dd
            items.extend([{"column": col, "subtask": k, "delta": v} for k, v in dd.items()])
        largest_pos[exp] = sorted(items, key=lambda r: r["delta"], reverse=True)[:20]
        largest_neg[exp] = sorted(items, key=lambda r: r["delta"])[:20]
    selected_rows = []
    for col, sub in SELECTED:
        row = {"column": col, "subtask": sub}
        for exp in ["20M", "30M", "40M", "50M", "70M", "80M"]:
            row[exp] = deltas.get(exp, {}).get(col, {}).get(sub)
        selected_rows.append(row)
    out = {"status": "SUBTASK_LATE_TRAJECTORY", "reports": reports, "deltas": deltas, "missing": missing, "largest_positive": largest_pos, "largest_negative": largest_neg, "selected_rows": selected_rows}
    out_json = _public_path('experiments/archive/frontier_consolidation/data/subtask_late_trajectory/subtask_late_trajectory.json')
    out_md = _public_path('research/documents/frontier_consolidation/data/subtask_late_trajectory/subtask_late_trajectory.md')
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research subtask late trajectory", "", f"Missing report records: {len(missing)}", "", "## Selected delta trajectories", "", "| column | subtask | 20M | 30M | 40M | 50M | 70M | 80M |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in selected_rows:
        def f(exp: str) -> str:
            v = r.get(exp)
            return "" if v is None else f"{float(v):+.2f}"
        lines.append(f"| {r['column']} | {r['subtask']} | {f('20M')} | {f('30M')} | {f('40M')} | {f('50M')} | {f('70M')} | {f('80M')} |")
    for exp in ["70M", "80M"]:
        lines += ["", f"## {exp} largest negative subtask deltas", "", "| column | subtask | delta |", "|---|---|---:|"]
        for r in largest_neg.get(exp, [])[:15]:
            lines.append(f"| {r['column']} | {r['subtask']} | {r['delta']:+.2f} |")
        lines += ["", f"## {exp} largest positive subtask deltas", "", "| column | subtask | delta |", "|---|---|---:|"]
        for r in largest_pos.get(exp, [])[:15]:
            lines.append(f"| {r['column']} | {r['subtask']} | {r['delta']:+.2f} |")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "out_json": str(out_json), "out_md": str(out_md), "missing": len(missing), "has_70M_adapter": bool(reports["scale1p75"].get("70M")), "has_80M_adapter": bool(reports["scale1p75"].get("80M"))}, indent=2), flush=True)


if __name__ == "__main__":
    main()
