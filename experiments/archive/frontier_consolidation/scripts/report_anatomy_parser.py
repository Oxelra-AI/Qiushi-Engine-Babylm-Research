#!/usr/bin/env python3
"""research: parse official best-temperature reports for mature Muon anatomy.

Compares continuous matched-decay Muon 80M against research 80M on the damaged
and improved cheap-column surfaces. This script operates only on saved official
reports and prediction artifacts; it runs no model evaluation.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import re
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/frontier_consolidation')
OUT = _public_path('experiments/archive/frontier_consolidation/data/continuous_muon_anatomy')

REPORTS = {
    "reference_80M": {
        "Entity": _public_path('experiments/archive/frontier_consolidation/data/legal_mature_treatment_effect_eval/official_outputs/complianttok_reinvest_seed43022_80M/Entity/chck_80M/full_complianttok_reinvest_seed43022_80M_Entity/zero_shot/mlm/entity_tracking/entity_tracking/best_temperature_report.txt'),
        "EWoK": _public_path('experiments/archive/frontier_consolidation/data/legal_mature_treatment_effect_eval/official_outputs/complianttok_reinvest_seed43022_80M/EWoK/chck_80M/full_complianttok_reinvest_seed43022_80M_EWoK/zero_shot/mlm/ewok/ewok_filtered/best_temperature_report.txt'),
        "GlobalPIQA_parallel": _public_path('experiments/archive/frontier_consolidation/data/legal_mature_treatment_effect_eval/official_outputs/complianttok_reinvest_seed43022_80M/GlobalPIQA_parallel/chck_80M/full_complianttok_reinvest_seed43022_80M_GlobalPIQA_parallel/zero_shot/mlm/global_piqa_parallel/global_piqa_parallel/best_temperature_report.txt'),
        "GlobalPIQA_nonparallel": _public_path('experiments/archive/frontier_consolidation/data/legal_mature_treatment_effect_eval/official_outputs/complianttok_reinvest_seed43022_80M/GlobalPIQA_nonparallel/chck_80M/full_complianttok_reinvest_seed43022_80M_GlobalPIQA_nonparallel/zero_shot/mlm/global_piqa_nonparallel/global_piqa_nonparallel/best_temperature_report.txt'),
        "Reading": _public_path('experiments/archive/frontier_consolidation/data/legal_mature_treatment_effect_eval/official_outputs/complianttok_reinvest_seed43022_80M/Reading/chck_80M/full_complianttok_reinvest_seed43022_80M_Reading/zero_shot/mlm/reading/report.txt'),
    },
    "continuous_muon_80M": {
        "Entity": _public_path('experiments/archive/frontier_consolidation/data/muon_wdmatched_mature_eval/official_outputs/muon_lr008_wd00125_80M/Entity/chck_80M/full_muon_lr008_wd00125_80M_Entity/zero_shot/mlm/entity_tracking/entity_tracking/best_temperature_report.txt'),
        "EWoK": _public_path('experiments/archive/frontier_consolidation/data/muon_wdmatched_mature_eval/official_outputs/muon_lr008_wd00125_80M/EWoK/chck_80M/full_muon_lr008_wd00125_80M_EWoK/zero_shot/mlm/ewok/ewok_filtered/best_temperature_report.txt'),
        "GlobalPIQA_parallel": _public_path('experiments/archive/frontier_consolidation/data/muon_wdmatched_mature_eval/official_outputs/muon_lr008_wd00125_80M/GlobalPIQA_parallel/chck_80M/full_muon_lr008_wd00125_80M_GlobalPIQA_parallel/zero_shot/mlm/global_piqa_parallel/global_piqa_parallel/best_temperature_report.txt'),
        "GlobalPIQA_nonparallel": _public_path('experiments/archive/frontier_consolidation/data/muon_wdmatched_mature_eval/official_outputs/muon_lr008_wd00125_80M/GlobalPIQA_nonparallel/chck_80M/full_muon_lr008_wd00125_80M_GlobalPIQA_nonparallel/zero_shot/mlm/global_piqa_nonparallel/global_piqa_nonparallel/best_temperature_report.txt'),
        "Reading": _public_path('experiments/archive/frontier_consolidation/data/muon_wdmatched_mature_eval/official_outputs/muon_lr008_wd00125_80M/Reading/chck_80M/full_muon_lr008_wd00125_80M_Reading/zero_shot/mlm/reading/report.txt'),
    },
}

SECTION_RE = re.compile(r"^###\s+(.+?)\s*$")
KV_RE = re.compile(r"^(.+?):\s*(-?\d+(?:\.\d+)?)\s*$")
AVG_RE = re.compile(r"^AVERAGE ACCURACY\s*$")
READ_RE = re.compile(r"^(EYE TRACKING SCORE|SELF-PACED READING SCORE):\s*(-?\d+(?:\.\d+)?)")


def parse_report(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    sections: dict[str, dict[str, float]] = {}
    cur = "header"
    last_avg = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = SECTION_RE.match(line)
        if m:
            cur = m.group(1)
            sections.setdefault(cur, {})
            continue
        rm = READ_RE.match(line)
        if rm:
            sections.setdefault("Reading", {})[rm.group(1).replace(" SCORE", "")] = float(rm.group(2))
            continue
        km = KV_RE.match(line)
        if km:
            sections.setdefault(cur, {})[km.group(1)] = float(km.group(2))
            continue
        if last_avg == "waiting":
            try:
                sections.setdefault("AVERAGE", {})["score"] = float(line)
            except ValueError:
                pass
            last_avg = None
        elif AVG_RE.match(line):
            last_avg = "waiting"
    return {"path": str(path), "sections": sections}


def flatten(parsed: dict[str, Any]) -> dict[str, float]:
    out = {}
    for section, vals in parsed["sections"].items():
        for k, v in vals.items():
            out[f"{section}/{k}"] = float(v)
    return out


def compare(a: dict[str, float], b: dict[str, float]) -> dict[str, Any]:
    keys = sorted(set(a) & set(b))
    rows = []
    for k in keys:
        rows.append({"key": k, "research": a[k], "muon": b[k], "delta": b[k] - a[k]})
    rows.sort(key=lambda r: r["delta"])
    return {
        "n_common": len(rows),
        "largest_losses": rows[:20],
        "largest_gains": list(reversed(rows[-20:])),
        "mean_delta_all_report_entries": mean(r["delta"] for r in rows) if rows else None,
        "all_rows": rows,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    parsed = {model: {task: parse_report(path) for task, path in tasks.items()} for model, tasks in REPORTS.items()}
    flat = {model: {task: flatten(p) for task, p in tasks.items()} for model, tasks in parsed.items()}
    comparisons = {}
    for task in REPORTS["reference_80M"]:
        comparisons[task] = compare(flat["reference_80M"][task], flat["continuous_muon_80M"][task])
    out = {"status": "CONTINUOUS_MUON_REPORT_ANATOMY", "parsed": parsed, "comparisons": comparisons}
    (_public_path('experiments/archive/frontier_consolidation/data/continuous_muon_anatomy/continuous_muon_report_anatomy.json')).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# research continuous matched-decay Muon 80M report anatomy",
        "",
        "Compared against research legal reinvest 80M using saved official report files only.",
        "",
    ]
    for task, comp in comparisons.items():
        lines += [f"## {task}", "", "### Largest losses (Muon - research)", "", "| entry | research | Muon | Δ |", "|---|---:|---:|---:|"]
        for r in comp["largest_losses"][:12]:
            lines.append(f"| {r['key']} | {r['research']:.3f} | {r['muon']:.3f} | {r['delta']:+.3f} |")
        lines += ["", "### Largest gains (Muon - research)", "", "| entry | research | Muon | Δ |", "|---|---:|---:|---:|"]
        for r in comp["largest_gains"][:12]:
            lines.append(f"| {r['key']} | {r['research']:.3f} | {r['muon']:.3f} | {r['delta']:+.3f} |")
        lines.append("")
    (_public_path('experiments/archive/frontier_consolidation/data/continuous_muon_anatomy/continuous_muon_report_anatomy.md')).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": out["status"],
        "out": str(OUT),
        "entity_largest_losses": comparisons["Entity"]["largest_losses"][:6],
        "ewok_largest_losses": comparisons["EWoK"]["largest_losses"][:6],
        "globalpiqa_parallel_largest_gains": comparisons["GlobalPIQA_parallel"]["largest_gains"][:6],
        "reading": comparisons["Reading"],
    }, indent=2))


if __name__ == "__main__":
    main()
