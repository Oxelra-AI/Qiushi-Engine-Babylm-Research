#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import pathlib
from statistics import mean

ROOT = _public_path('.')
WORKSPACE = _public_path('experiments/archive/frontier_consolidation')
OUT = _public_path('experiments/archive/frontier_consolidation/data/private_scale_thresholds')
PANEL = _public_path('experiments/archive/frontier_consolidation/data/private_scale_panel_analysis/private_scale_panel_analysis.json')
COH_SG = _public_path('experiments/archive/frontier_consolidation/data/fastpath4M_coherent_superglue_summary/fastpath4M_coherent_superglue_summary.json')
CHCK82 = _public_path('experiments/archive/frontier_consolidation/data/chck82_independent_verification/chck82_independent_verification.json')
CHEAP = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]

def rel(p: pathlib.Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)

def read(p: pathlib.Path):
    return json.loads(p.read_text())

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panel = read(PANEL)
    coh_sg = read(COH_SG)["superglue"]
    protected = read(CHCK82)
    ch_scores = {k: float(v) for k, v in protected["score_arithmetic"]["scores"].items() if v is not None}
    ch_overall = float(protected["score_arithmetic"]["overall_reported"])
    ch_cheap = mean(ch_scores[c] for c in CHEAP)
    arms = panel["score_table"]
    rows = {}
    for arm in ["coherent86_private_alpha0p5", "coherent86_private_alpha0p75", "coherent86_private_alpha1"]:
        cheap = float(arms[arm]["cheap7"])
        overall_if_coh_sg = mean([*([float(arms[arm]["scores"][c]) for c in CHEAP]), coh_sg, 0.0])
        sg_to_match_chck82 = 9 * ch_overall - 7 * cheap
        alpha1_overall = mean([*([float(arms["coherent86_private_alpha1"]["scores"][c]) for c in CHEAP]), coh_sg, 0.0])
        sg_to_match_alpha1 = 9 * alpha1_overall - 7 * cheap
        rows[arm] = {
            "cheap7": cheap,
            "delta_cheap7_vs_chck82": cheap - ch_cheap,
            "overall_if_superglue_equals_alpha1_first_sg": overall_if_coh_sg,
            "delta_overall_if_alpha1_sg_vs_chck82": overall_if_coh_sg - ch_overall,
            "superglue_needed_to_match_chck82_overall_aoa0": sg_to_match_chck82,
            "superglue_slack_below_chck82_sg": ch_scores["SuperGLUE"] - sg_to_match_chck82,
            "superglue_needed_to_match_alpha1_overall_using_alpha1_first_sg": sg_to_match_alpha1,
            "superglue_slack_below_alpha1_first_sg": coh_sg - sg_to_match_alpha1,
        }
    out = {
        "status": "COMPLETE",
        "sources": {"panel": rel(PANEL), "coherent_alpha1_sg_summary": rel(COH_SG), "chck82": rel(CHCK82)},
        "protected_chck82": {"cheap7": ch_cheap, "superglue": ch_scores["SuperGLUE"], "overall": ch_overall},
        "coherent_alpha1_first_superglue": coh_sg,
        "rows": rows,
        "scientific_reading": "If alpha0.5/0.75 SuperGLUE stays near the already observed coherent/chck82 range, their higher cheap7 makes them stronger no-training materialized endpoints; mechanism remains amplitude-controlled redistribution.",
    }
    js = _public_path('experiments/archive/frontier_consolidation/data/private_scale_thresholds/private_scale_thresholds.json')
    md = _public_path('research/documents/frontier_consolidation/data/private_scale_thresholds/private_scale_thresholds.md')
    js.write_text(json.dumps(out, indent=2) + "\n")
    lines = ["# research private-scale SuperGLUE thresholds", "", f"Protected chck82 Overall: `{ch_overall}`; SuperGLUE `{ch_scores['SuperGLUE']}`; cheap7 `{ch_cheap}`.", f"Alpha1 first SuperGLUE: `{coh_sg}`.", "", "| arm | cheap7 | SG needed to match chck82 | slack below chck82 SG | SG needed to match alpha1 | slack below alpha1 SG | Overall if alpha1 SG |", "|---|---:|---:|---:|---:|---:|---:|"]
    for arm, r in rows.items():
        lines.append(f"| {arm} | {r['cheap7']} | {r['superglue_needed_to_match_chck82_overall_aoa0']} | {r['superglue_slack_below_chck82_sg']} | {r['superglue_needed_to_match_alpha1_overall_using_alpha1_first_sg']} | {r['superglue_slack_below_alpha1_first_sg']} | {r['overall_if_superglue_equals_alpha1_first_sg']} |")
    lines += ["", out["scientific_reading"], "", f"JSON: `{rel(js)}`"]
    md.write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": out["status"], "out_json": rel(js), "out_md": rel(md)}, indent=2))

if __name__ == "__main__":
    main()
