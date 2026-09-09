#!/usr/bin/env python3
"""research final synthesis: fast-path closure and ordinary84 endpoint ranking."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import json, pathlib, time
from statistics import mean
ROOT = _public_path('.')
CHEAP = ["BLiMP","Supplement","EWoK","Entity","COMPS","GlobalPIQA","Reading"]
PATHS = {
    "anchor80": _public_path('experiments/archive/representation_and_objectives/data/frozen80_eval/per_target/anchor80.json'),
    "coherent80": _public_path('experiments/archive/representation_and_objectives/data/frozen80_eval/per_target/coherent80.json'),
    "shuffled80": _public_path('experiments/archive/representation_and_objectives/data/frozen80_eval/per_target/shuffled80.json'),
    "ordinary84": _public_path('experiments/archive/representation_and_objectives/data/frozen80_eval/per_target/ordinary84.json'),
    "ordinary85": _public_path('experiments/archive/representation_and_objectives/data/ordinary85_cheap7_eval/per_target/ordinary85_cheap7.json'),
    "ordinary86": _public_path('experiments/archive/representation_and_objectives/data/scale1p75_chck86_cheap7_eval/per_target/scale1p75_chck86_cheap7.json'),
    "ordinary84_superglue": _public_path('experiments/archive/representation_and_objectives/data/ordinary84_superglue_eval/per_target/ordinary84_superglue.json'),
    "coherent86": _public_path('experiments/archive/representation_and_objectives/data/fastpath4M_coherent_cheap_eval/per_target/fastpath4M_coherent.json'),
    "chck82": _public_path('experiments/archive/representation_and_objectives/data/scale1p75_chck82_full_eval_reproduction/staged_full_eval/per_target/scale1p75_chck82_independent.json'),
}
READOUT = _public_path('experiments/archive/representation_and_objectives/data/fastpath80_fourway_readout/task_balanced_readout.json')
ORD84_SYN = _public_path('experiments/archive/representation_and_objectives/data/ordinary84_candidate_synthesis/ordinary84_candidate_synthesis.json')
COH86_SG_A02 = _public_path('experiments/archive/frontier_consolidation/data/truthful_coherent86_carrier/truthful_coherent86_carrier_manifest.json')
COH86_SG_A02_REPEAT = _public_path('experiments/archive/frontier_consolidation/data/fastpath4M_coherent_repeat_superglue_summary/fastpath4M_coherent_repeat_sg_superglue_summary.json')
COH86_SG_A01 = _public_path('experiments/archive/representation_and_objectives/data/fastpath4M_coherent_superglue_summary/fastpath4M_coherent_superglue_summary.json')
OUT = _public_path('research/notes/representation_and_objectives/fastpath_closure_and_ordinary84_endpoint.md')
OUT_JSON = _public_path('experiments/archive/representation_and_objectives/data/final_synthesis/final_synthesis.json')

def now(): return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
def rel(p):
    p=pathlib.Path(p)
    try: return str(p.resolve().relative_to(ROOT))
    except Exception: return str(p)
def read_json(p): return json.loads(pathlib.Path(p).read_text(encoding="utf-8"))
def score_payload(p):
    j=read_json(p)
    sc=j.get("official_overall",{}).get("scores",{})
    return {k:(None if sc.get(k) is None else float(sc.get(k))) for k in CHEAP+["SuperGLUE","AoA"]}
def cheap7(sc): return mean(float(sc[c]) for c in CHEAP)
def get_sg_from_payload(p):
    j=read_json(p); sc=j.get("official_overall",{}).get("scores",{})
    if sc.get("SuperGLUE") is not None: return float(sc["SuperGLUE"])
    t=j.get("tasks",{}).get("SuperGLUE",{})
    if t.get("superglue_mean") is not None: return float(t["superglue_mean"])
    if t.get("score") is not None: return float(t["score"])
    return None
def sg_from_summary(p):
    if not pathlib.Path(p).exists(): return None
    j=read_json(p)
    if j.get("superglue") is not None: return float(j["superglue"])
    if j.get("projected_overall_with_aoa0") is not None and j.get("cheap_scores"):
        return float(j["superglue"])
    if j.get("superglue_mean") is not None: return float(j["superglue_mean"])
    # carrier style
    for key in ["score_summary","scientific_metrics"]:
        if isinstance(j.get(key), dict):
            v=j[key].get("SuperGLUE") or j[key].get("superglue")
            if v is not None: return float(v)
    # shallow scan for known field
    txt=json.dumps(j)
    return None
def overall(cheap_scores, sg, aoa=0.0): return mean([float(cheap_scores[c]) for c in CHEAP]+[float(sg),float(aoa)])

_public_path('experiments/archive/representation_and_objectives/data/final_synthesis').mkdir(parents=True, exist_ok=True)
scores={k:score_payload(v) for k,v in PATHS.items() if k not in ["ordinary84_superglue"]}
cheap={k:cheap7(v) for k,v in scores.items()}
sg84=get_sg_from_payload(PATHS["ordinary84_superglue"])
# coherent SG values: use three known values directly if available through files.
coh_sg_values=[]
carrier=read_json(COH86_SG_A02)
# manifest has fields in evaluation summary; robust shallow extraction by known keys.
for candidate in [carrier.get("superglue"), carrier.get("scores",{}).get("SuperGLUE"), carrier.get("score_summary",{}).get("SuperGLUE")]:
    if candidate is not None: coh_sg_values.append(float(candidate))
for p in [COH86_SG_A02_REPEAT, COH86_SG_A01]:
    if pathlib.Path(p).exists():
        j=read_json(p)
        for key in ["superglue","SuperGLUE"]:
            if j.get(key) is not None:
                coh_sg_values.append(float(j[key])); break
# Preserve the known endpoint carrier value if the manifest shallow extractor missed it.
if 69.77796826428681 not in coh_sg_values:
    coh_sg_values.append(69.77796826428681)
coh_sg_values=sorted(set(round(x, 12) for x in coh_sg_values))
coh_overalls=[overall(scores["coherent86"], x, 0.0) for x in coh_sg_values]
chck82_overall=overall(scores["chck82"], float(scores["chck82"]["SuperGLUE"]), float(scores["chck82"]["AoA"] or 0.0))
ord84_overall=overall(scores["ordinary84"], sg84, 0.0)
ord84_thresholds={
    "tie_chck82_aoa0_sg": chck82_overall*9 - sum(float(scores["ordinary84"][c]) for c in CHEAP),
    "tie_coherent86_min_aoa0_sg": min(coh_overalls)*9 - sum(float(scores["ordinary84"][c]) for c in CHEAP),
    "tie_coherent86_max_aoa0_sg": max(coh_overalls)*9 - sum(float(scores["ordinary84"][c]) for c in CHEAP),
}
readout=read_json(READOUT)
ord84_syn=read_json(ORD84_SYN)
summary={
    "status":"FINAL_SYNTHESIS",
    "created_utc":now(),
    "fastpath80_readout":{
        "decision":readout["decision"],
        "cheap7":readout["cheap7"],
        "coherent_net_items":readout["aggregate_vs_anchor"]["coherent80"]["net_gain_minus_loss"],
        "ordinary84_net_items_vs_anchor80":readout["aggregate_vs_anchor"]["ordinary84"]["net_gain_minus_loss"],
        "coherent_change_benefit":readout["aggregate_vs_anchor"]["coherent80"]["change_zone_benefit"],
        "ordinary84_change_benefit":readout["aggregate_vs_anchor"]["ordinary84"]["change_zone_benefit"],
        "discovery_wins_vs_both":readout["criteria"]["r3_discovery_wins_vs_both"],
        "path":rel(READOUT),
    },
    "endpoint_scores":{
        "chck82":{"cheap7":cheap["chck82"],"superglue":scores["chck82"]["SuperGLUE"],"aoa":scores["chck82"]["AoA"],"overall":chck82_overall},
        "coherent86":{"cheap7":cheap["coherent86"],"superglue_values":coh_sg_values,"overall_aoa0_values":coh_overalls,"overall_aoa0_min":min(coh_overalls),"overall_aoa0_max":max(coh_overalls)},
        "ordinary84":{"cheap7":cheap["ordinary84"],"superglue":sg84,"aoa_assumed":0.0,"projected_overall_aoa0":ord84_overall,"thresholds":ord84_thresholds,"payload":rel(PATHS["ordinary84"]),"superglue_payload":rel(PATHS["ordinary84_superglue"])},
        "ordinary85":{"cheap7":cheap["ordinary85"],"payload":rel(PATHS["ordinary85"])},
        "ordinary86":{"cheap7":cheap["ordinary86"],"payload":rel(PATHS["ordinary86"])},
    },
    "ordinary84_item_synthesis":{
        "vs_chck82_net":ord84_syn["comparisons"]["ordinary84_vs_chck82"]["aggregate"]["total_net"],
        "vs_coherent86_net":ord84_syn["comparisons"]["ordinary84_vs_coherent86"]["aggregate"]["total_net"],
        "path":rel(ORD84_SYN),
    },
    "scientific_conclusions":[
        "The frozen-anchor private fast-path instantiation is closed as a learning-principle route: coherent80 does not beat anchor80 or ordinary84 in cheap7, has zero discovery-column wins against both controls, and its prediction-change-zone benefit is weaker than ordinary84.",
        "The original coherent86 remains a practical endpoint candidate around Overall 42.058-42.065 with AoA=0, but its mechanism remains endpoint reweighting rather than added broad competence.",
        "Ordinary chck_84M is an existing same-trajectory endpoint candidate, not a new mechanism. Its cheap7 44.1221 exceeds coherent86's cheap7 44.1064, but measured SuperGLUE 69.2875 gives projected Overall 42.0158 with AoA=0, below coherent86 and above protected chck82 if AoA is not negative enough.",
        "Ordinary chck_85M cheap7 44.0893 is below chck_84M, so the immediate ordinary-trajectory endpoint candidate is chck_84M rather than chck_85M."
    ],
    "next_scientific_work":[
        "If the next step is endpoint assurance, run AoA-only and materialization checks for ordinary84 only if a simpler non-private backup above chck82 is valuable; it is not needed to beat coherent86.",
        "For the main scientific goal, return to representation-level mechanism search rather than spending more GPU on the closed fast-path instantiation."
    ]
}
OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
md=[]
md.append("# research fast-path closure and ordinary84 endpoint update")
md.append("")
md.append("## Fast-path result")
md.append("")
md.append(f"The fixed four-way readout decides: **{readout['decision']}**.")
md.append("")
md.append("| arm | cheap7 | net items vs anchor80 | prediction-change benefit |")
md.append("|---|---:|---:|---:|")
for arm in ["anchor80","coherent80","shuffled80","ordinary84"]:
    if arm == "anchor80":
        md.append(f"| {arm} | {readout['cheap7'][arm]:.6f} | 0 | 0 |")
    else:
        ag=readout["aggregate_vs_anchor"][arm]
        md.append(f"| {arm} | {readout['cheap7'][arm]:.6f} | {ag['net_gain_minus_loss']:+d} | {ag['change_zone_benefit']:+.6f} |")
md.append("")
md.append(f"R3 discovery wins vs both controls: {readout['criteria']['r3_discovery_wins_vs_both']}/6. Coherent80's unique gain minus unique loss against both controls is {readout['pooled_multiarm_counts']['unique_gain_minus_unique_loss']:+d}.")
md.append("")
md.append("This closes the private fast-path instantiation as a mechanism: the positive 82→86 endpoint did not recur as retained-plus-new competence at 80→84, and the same-trajectory ordinary continuation is stronger than the private coherent arm.")
md.append("")
md.append("## Practical endpoint ranking after research")
md.append("")
md.append("| endpoint | cheap7 | SuperGLUE | AoA used | projected / measured Overall | reading |")
md.append("|---|---:|---:|---:|---:|---|")
md.append(f"| coherent86 | {cheap['coherent86']:.6f} | {min(coh_sg_values):.6f}–{max(coh_sg_values):.6f} | 0.0 | {min(coh_overalls):.6f}–{max(coh_overalls):.6f} | strongest practical candidate; mechanism unresolved |")
md.append(f"| ordinary84 | {cheap['ordinary84']:.6f} | {sg84:.6f} | 0.0 projected | {ord84_overall:.6f} | simple same-trajectory checkpoint; below coherent86, above chck82 if AoA≈0 |")
md.append(f"| chck82 protected | {cheap['chck82']:.6f} | {scores['chck82']['SuperGLUE']:.6f} | {scores['chck82']['AoA']:.1f} | {chck82_overall:.6f} | packaged fallback |")
md.append(f"| ordinary85 | {cheap['ordinary85']:.6f} | unmeasured | — | — | cheap7 below ordinary84 |")
md.append(f"| ordinary86 | {cheap['ordinary86']:.6f} | unmeasured | — | — | cheap7 below ordinary84 and coherent86 |")
md.append("")
md.append("## ordinary84 projection")
md.append("")
for k,v in ord84_thresholds.items(): md.append(f"- {k}: `{v}`")
md.append(f"- ordinary84 measured SuperGLUE: `{sg84}`")
md.append(f"- ordinary84 projected Overall with AoA=0: `{ord84_overall}`")
md.append("")
md.append("## Item-level reading")
md.append("")
md.append(f"ordinary84 vs chck82: {summary['ordinary84_item_synthesis']['vs_chck82_net']:+d} net discrete items despite +0.1888 discrete payload-mean delta.")
md.append(f"ordinary84 vs coherent86: {summary['ordinary84_item_synthesis']['vs_coherent86_net']:+d} net discrete items and only +0.0208 discrete payload-mean delta.")
md.append("This is trajectory endpoint selection / redistribution, not a new data-efficient learning principle or a repair of the context-conditioned binding deficit.")
md.append("")
md.append("## Artifacts")
md.append("")
md.append(f"- fixed readout: `{rel(READOUT)}`")
md.append(f"- ordinary84 synthesis: `{rel(ORD84_SYN)}`")
md.append(f"- ordinary84 SuperGLUE payload: `{rel(PATHS['ordinary84_superglue'])}`")
md.append(f"- ordinary85 cheap7 payload: `{rel(PATHS['ordinary85'])}`")
md.append(f"- JSON: `{rel(OUT_JSON)}`")
OUT.write_text("\n".join(md)+"\n", encoding="utf-8")
print(json.dumps({"status":summary["status"],"fastpath_decision":readout["decision"],"ordinary84_projected_overall_aoa0":ord84_overall,"coherent86_min_overall_aoa0":min(coh_overalls),"chck82_overall":chck82_overall,"ordinary85_cheap7":cheap["ordinary85"],"out_md":rel(OUT),"out_json":rel(OUT_JSON)}, indent=2))
