#!/usr/bin/env python3
"""research: no-training synthesis for ordinary chck_84M as a practical endpoint candidate.

Consumes saved official-compatible cheap7 payloads and compares ordinary84 against
protected chck82, coherent86, and ordinary86 using the validated research item parser.
No model inference and no submission-side action.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import json, math, pathlib, sys, time
from statistics import mean
from typing import Any

ROOT = _public_path('.')
A02_SCRIPTS = _public_path('experiments/archive/frontier_consolidation/scripts')
if str(A02_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(A02_SCRIPTS))
import pairwise_item_flip_analysis as s107  # noqa: E402
s107.ROOT = ROOT

CHEAP = list(s107.CHEAP_COLS)
DISCRETE = list(s107.DISCRETE_COLUMNS)
PATHS = {
    "chck82": _public_path('experiments/archive/representation_and_objectives/data/scale1p75_chck82_full_eval_reproduction/staged_full_eval/per_target/scale1p75_chck82_independent.json'),
    "coherent86": _public_path('experiments/archive/representation_and_objectives/data/fastpath4M_coherent_cheap_eval/per_target/fastpath4M_coherent.json'),
    "ordinary84": _public_path('experiments/archive/representation_and_objectives/data/frozen80_eval/per_target/ordinary84.json'),
    "ordinary86": _public_path('experiments/archive/representation_and_objectives/data/scale1p75_chck86_cheap7_eval/per_target/scale1p75_chck86_cheap7.json'),
    "anchor80": _public_path('experiments/archive/representation_and_objectives/data/frozen80_eval/per_target/anchor80.json'),
}
OUT = _public_path('experiments/archive/representation_and_objectives/data/ordinary84_candidate_synthesis')
CHCK82_FULL = _public_path('experiments/archive/frontier_consolidation/data/chck82_independent_verification/chck82_independent_verification.json')
COH86_SG_A02 = _public_path('experiments/archive/frontier_consolidation/data/fastpath4M_coherent_repeat_superglue_summary/fastpath4M_coherent_repeat_sg_superglue_summary.json')
COH86_SG_A01 = _public_path('experiments/archive/representation_and_objectives/data/fastpath4M_coherent_superglue_summary/fastpath4M_coherent_superglue_summary.json')

def now(): return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def rel(p: pathlib.Path|str) -> str:
    p=pathlib.Path(p)
    try: return str(p.resolve().relative_to(ROOT))
    except Exception: return str(p)

def read_json(p: pathlib.Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))

def scores_from_payload(payload: dict[str,Any]) -> dict[str,float|None]:
    sc=payload.get("official_overall",{}).get("scores",{})
    return {c:(None if sc.get(c) is None else float(sc.get(c))) for c in CHEAP+["SuperGLUE","AoA"]}

def cheap7(sc: dict[str,float|None]) -> float:
    return mean(float(sc[c]) for c in CHEAP)

def pair(base: str, cand: str) -> dict[str,Any]:
    bl=s107.PayloadLoader(PATHS[base]); cl=s107.PayloadLoader(PATHS[cand])
    cols={}
    for col in DISCRETE:
        cols[col]=s107.compare_column(bl, cl, col)
    aggregate={
        "total_gain": sum(cols[c]["flip_counts"].get("gain",0) for c in DISCRETE),
        "total_loss": sum(cols[c]["flip_counts"].get("loss",0) for c in DISCRETE),
        "total_common": sum(cols[c]["n_common"] for c in DISCRETE),
        "discrete_payload_mean_delta": mean(float(cols[c]["delta_score_payload"]) for c in DISCRETE if cols[c]["delta_score_payload"] is not None),
        "discrete_reconstructed_mean_delta": mean(float(cols[c]["delta_score_reconstructed_common"]) for c in DISCRETE),
    }
    aggregate["total_net"] = aggregate["total_gain"] - aggregate["total_loss"]
    aggregate["total_net_pct"] = 100.0 * aggregate["total_net"] / aggregate["total_common"]
    return {"base":base,"candidate":cand,"columns":cols,"aggregate":aggregate}

def sg_from_summary(path:pathlib.Path)->float|None:
    if not path.exists(): return None
    j=read_json(path)
    for k in ["superglue","SuperGLUE"]:
        if j.get(k) is not None: return float(j[k])
    if j.get("score") is not None: return float(j["score"])
    if isinstance(j.get("official_overall"),dict):
        v=j["official_overall"].get("scores",{}).get("SuperGLUE")
        if v is not None: return float(v)
    return None

def overall_from(cheap_scores:dict[str,float|None], sg:float, aoa:float=0.0)->float:
    return mean([float(cheap_scores[c]) for c in CHEAP]+[float(sg), float(aoa)])

OUT.mkdir(parents=True, exist_ok=True)
payloads={k:read_json(v) for k,v in PATHS.items()}
scores={k:scores_from_payload(v) for k,v in payloads.items()}
cheap={k:cheap7(scores[k]) for k in scores}
# Reference full overall for chck82.
chck82_full=read_json(CHCK82_FULL)
chck82_overall=float(chck82_full["score_arithmetic"]["overall_reported"])
chck82_sg=float(chck82_full["score_arithmetic"]["scores"]["SuperGLUE"])
coh_sgs=[x for x in [sg_from_summary(COH86_SG_A02), sg_from_summary(COH86_SG_A01)] if x is not None]
coh_min_sg=min(coh_sgs) if coh_sgs else None
coh_max_sg=max(coh_sgs) if coh_sgs else None
coh_a02_min_overall=overall_from(scores["coherent86"], coh_min_sg, 0.0) if coh_min_sg is not None else None
coh_a01_max_overall=overall_from(scores["coherent86"], coh_max_sg, 0.0) if coh_max_sg is not None else None
sum84=sum(float(scores["ordinary84"][c]) for c in CHEAP)
thresholds={
    "ordinary84_sg_needed_to_tie_chck82_with_aoa0": chck82_overall*9.0 - sum84,
    "ordinary84_sg_needed_to_tie_coherent86_min_projected_with_aoa0": None if coh_a02_min_overall is None else coh_a02_min_overall*9.0 - sum84,
    "ordinary84_sg_needed_to_tie_coherent86_max_projected_with_aoa0": None if coh_a01_max_overall is None else coh_a01_max_overall*9.0 - sum84,
    "ordinary84_projected_overall_if_sg_equals_chck82_and_aoa0": overall_from(scores["ordinary84"], chck82_sg, 0.0),
}
comparisons={
    "ordinary84_vs_chck82": pair("chck82","ordinary84"),
    "ordinary84_vs_coherent86": pair("coherent86","ordinary84"),
    "ordinary84_vs_ordinary86": pair("ordinary86","ordinary84"),
    "ordinary84_vs_anchor80": pair("anchor80","ordinary84"),
}
summary={
    "status":"ORDINARY84_CANDIDATE_SYNTHESIS",
    "created_utc":now(),
    "payload_paths":{k:rel(v) for k,v in PATHS.items()},
    "scores":scores,
    "cheap7":cheap,
    "chck82_full_reference":{"overall":chck82_overall,"superglue":chck82_sg,"path":rel(CHCK82_FULL)},
    "coherent86_superglue_refs":{"values":coh_sgs,"min_projected_overall_aoa0":coh_a02_min_overall,"max_projected_overall_aoa0":coh_a01_max_overall},
    "projection_thresholds":thresholds,
    "comparisons":comparisons,
    "scientific_reading":[
        "ordinary84 is not a new mechanism; it is an existing same-seed scale1.75 trajectory checkpoint found while constructing the fast-path control.",
        "Its cheap7 exceeds protected chck82 and coherent86, so SuperGLUE-only is the lowest-cost decision-changing endpoint check before any AoA/materialization.",
        "The item-transition comparisons should be read as trajectory endpoint selection and redistribution, not a data-efficient learning principle or binding-deficit repair."
    ]
}
(_public_path('experiments/archive/representation_and_objectives/data/ordinary84_candidate_synthesis/ordinary84_candidate_synthesis.json')).write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
lines=[]
lines.append("# research ordinary84 candidate synthesis")
lines.append("")
lines.append("No-training analysis of saved official-compatible payloads. ordinary84 is the same seed43022 scale1.75 trajectory, not a new training route.")
lines.append("")
lines.append("## Cheap7 scores")
lines.append("")
lines.append("| arm | BLiMP | Supplement | EWoK | Entity | COMPS | GlobalPIQA | Reading | cheap7 |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for arm in ["anchor80","chck82","ordinary84","ordinary86","coherent86"]:
    sc=scores[arm]
    lines.append(f"| {arm} | {sc['BLiMP']:.3f} | {sc['Supplement']:.3f} | {sc['EWoK']:.3f} | {sc['Entity']:.3f} | {sc['COMPS']:.3f} | {sc['GlobalPIQA']:.3f} | {sc['Reading']:.3f} | {cheap[arm]:.6f} |")
lines.append("")
lines.append("## SuperGLUE projection thresholds for ordinary84")
lines.append("")
for k,v in thresholds.items():
    lines.append(f"- {k}: `{v}`")
lines.append("")
lines.append("## Item-transition comparisons")
for cname, comp in comparisons.items():
    ag=comp["aggregate"]
    lines.append("")
    lines.append(f"### {cname}")
    lines.append(f"- total gain/loss/net: {ag['total_gain']}/{ag['total_loss']}/{ag['total_net']:+d} over {ag['total_common']} common discrete items")
    lines.append(f"- discrete payload mean delta: {ag['discrete_payload_mean_delta']:+.6f}; reconstructed mean delta: {ag['discrete_reconstructed_mean_delta']:+.6f}")
    lines.append("| column | payload Δ | recon Δ | gains | losses | net | best groups | worst groups |")
    lines.append("|---|---:|---:|---:|---:|---:|---|---|")
    for col in DISCRETE:
        c=comp["columns"][col]
        best=", ".join(f"{g['group']} {g['net_gain_minus_loss']:+d}/{g['n']}" for g in c["best_groups_by_item_net"][:3])
        worst=", ".join(f"{g['group']} {g['net_gain_minus_loss']:+d}/{g['n']}" for g in c["worst_groups_by_item_net"][:3])
        pd=c["delta_score_payload"]
        lines.append(f"| {col} | {pd:+.3f} | {c['delta_score_reconstructed_common']:+.3f} | {c['flip_counts'].get('gain',0)} | {c['flip_counts'].get('loss',0)} | {c['gain_minus_loss_items']:+d} | {best} | {worst} |")
lines.append("")
lines.append("## Scientific reading")
for x in summary["scientific_reading"]:
    lines.append(f"- {x}")
lines.append("")
lines.append(f"JSON: `{rel(_public_path('experiments/archive/representation_and_objectives/data/ordinary84_candidate_synthesis/ordinary84_candidate_synthesis.json'))}`")
(_public_path('research/documents/representation_and_objectives/data/ordinary84_candidate_synthesis/ordinary84_candidate_synthesis.md')).write_text("\n".join(lines)+"\n", encoding="utf-8")
print(json.dumps({"status":summary["status"],"cheap7":cheap,"thresholds":thresholds,"ordinary84_vs_chck82_net":comparisons["ordinary84_vs_chck82"]["aggregate"]["total_net"],"ordinary84_vs_coherent86_net":comparisons["ordinary84_vs_coherent86"]["aggregate"]["total_net"],"out_json":rel(_public_path('experiments/archive/representation_and_objectives/data/ordinary84_candidate_synthesis/ordinary84_candidate_synthesis.json')),"out_md":rel(_public_path('research/documents/representation_and_objectives/data/ordinary84_candidate_synthesis/ordinary84_candidate_synthesis.md'))}, indent=2))
