#!/usr/bin/env python3
"""research trusted comparison using repaired SuperGLUE and research measured AoA manifests.

This mirrors the research comparison but points to the completed research AoA manifests.
It preserves the distinction between measured zero and missing-checkpoint placeholder zero.
It does not establish a submission endpoint; it assembles the corrected coordinate evidence
currently available.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

OUT_ROOT = Path("experiments/archive/functional_learning/data/trusted_comparison_with_measured_aoa")
OUT_ROOT.mkdir(parents=True, exist_ok=True)
SUPERGLUE_REQUIRED_TASKS = ["boolq", "multirc", "rte", "wsc", "mrpc", "qqp", "mnli"]
MODEL_PATHS = {
    "coherent86": {
        "zero_reading": "experiments/archive/frontier_consolidation/data/private_scale_sentinel_eval/coherent86_private_scale_0p75/per_target/coherent86_private_scale_0p75.json",
        "superglue": "experiments/archive/functional_learning/data/repaired_coherent86_eval/per_target/repaired_coherent86_alpha075.json",
        "aoa_manifest": "experiments/archive/functional_learning/data/batched_aoa_measured/full/coherent86/full/aoa_manifest.json",
        "expected_aoa_steps": 18,
    },
    "dense_seed62064": {
        "zero_reading": "experiments/archive/functional_learning/data/dense_focus_official_eval_seed62064/per_target/dense_focus_seed62064_u0080.json",
        "superglue": "experiments/archive/functional_learning/data/repaired_dense_eval_seed62064/per_target/repaired_dense_seed62064_u0080.json",
        "aoa_manifest": "experiments/archive/functional_learning/data/batched_aoa_measured/full/dense_seed62064/full/aoa_manifest.json",
        "expected_aoa_steps": 18,
    },
    "dense_seed62065": {
        "zero_reading": "experiments/archive/functional_learning/data/dense_focus_official_eval_seed62065/per_target/dense_focus_seed62065_u0080.json",
        "superglue": "experiments/archive/functional_learning/data/repaired_dense_eval_seed62065/per_target/repaired_dense_seed62065_u0080.json",
        "aoa_manifest": "experiments/archive/functional_learning/data/batched_aoa_measured/full/dense_seed62065/full/aoa_manifest.json",
        "expected_aoa_steps": 18,
    },
}


def load_json(path: str | Path) -> dict[str, Any] | None:
    p=Path(path)
    if not p.is_file(): return None
    return json.loads(p.read_text())


def validate_superglue(payload: dict[str,Any] | None) -> dict[str,Any]:
    if payload is None: return {"valid":False,"reason":"payload_missing"}
    sg=payload.get("tasks",{}).get("SuperGLUE",{})
    if not sg: return {"valid":False,"reason":"no_superglue_task"}
    details=sg.get("superglue_primary_metric_details",[])
    detail_tasks={}
    for d in details if isinstance(details,list) else []:
        if isinstance(d,dict) and d.get("task") and d.get("metric") and d.get("score") is not None:
            detail_tasks[d["task"]]={"score":float(d["score"]),"metric":d["metric"],"results_txt":d.get("results_txt")}
    missing=[t for t in SUPERGLUE_REQUIRED_TASKS if t not in detail_tasks]
    if missing: return {"valid":False,"reason":"missing_tasks","missing":missing,"present":sorted(detail_tasks)}
    task_records=sg.get("tasks",[])
    bad=[r.get("task") for r in task_records if isinstance(r,dict) and r.get("returncode") not in (0,None)]
    if bad: return {"valid":False,"reason":"nonzero_subtask_returncode","tasks":bad}
    mean=sg.get("superglue_mean")
    if mean is None: mean=sum(detail_tasks[t]["score"] for t in SUPERGLUE_REQUIRED_TASKS)/7.0
    return {"valid":True,"superglue_mean":float(mean),"per_task":detail_tasks,"n_tasks":7}


def extract_zero_reading_scores(payload: dict[str,Any] | None) -> dict[str,float]:
    if payload is None: return {}
    tasks=payload.get("tasks",{})
    scores={}
    for col in ["BLiMP","Supplement","EWoK","Entity","COMPS"]:
        t=tasks.get(col,{})
        if t.get("returncode")==0 and t.get("score") is not None:
            scores[col]=float(t["score"])
    gp_par=tasks.get("GlobalPIQA_parallel",{}).get("score")
    gp_non=tasks.get("GlobalPIQA_nonparallel",{}).get("score")
    if gp_par is not None and gp_non is not None:
        scores["GlobalPIQA_parallel"]=float(gp_par); scores["GlobalPIQA_nonparallel"]=float(gp_non); scores["GlobalPIQA"]=(float(gp_par)+float(gp_non))/2
    reading=tasks.get("Reading",{}).get("scores",{}).get("Reading")
    if reading is not None: scores["Reading"]=float(reading)
    return scores


def validate_aoa(path: str, expected_steps:int=18)->dict[str,Any]:
    man=load_json(path)
    if man is None: return {"status":"missing","measured":False,"score":None,"source":path}
    payload=man.get("aoa_payload_for_comparison",{}); score=man.get("score",{}); assembled=man.get("assembled",{}); summary=assembled.get("summary",{})
    complete=bool(man.get("complete_measured_evidence")); measured=bool(payload.get("measured") or score.get("measured"))
    n_steps=int(payload.get("n_steps") or len(assembled.get("steps",[])) or 0)
    if not complete or not measured:
        return {"status":"present_not_measured","measured":False,"score":None,"source":path,"manifest_status":man.get("status")}
    if n_steps!=expected_steps:
        return {"status":"wrong_step_count","measured":False,"score":None,"source":path,"n_steps":n_steps,"expected_steps":expected_steps}
    if summary.get("missing_steps") or summary.get("steps_with_wrong_counts"):
        return {"status":"incomplete_steps","measured":False,"score":None,"source":path,"missing_steps":summary.get("missing_steps"),"wrong_counts":summary.get("steps_with_wrong_counts")}
    estimator=man.get("aoa_estimator") or score.get("aoa_estimator") or {}
    return {"status":"measured","measured":True,"score":float(payload.get("aoa_leaderboard_score", score.get("aoa_leaderboard_score"))),"raw_correlation":payload.get("aoa_raw_correlation",score.get("aoa_raw_correlation")),"legitimate_zero":bool(payload.get("legitimate_zero",score.get("legitimate_zero"))),"estimator_id":payload.get("estimator_id") or estimator.get("estimator_id"),"estimator_variant":payload.get("estimator_variant") or estimator.get("variant_name"),"source":path,"n_results":payload.get("n_results",summary.get("n_results")),"n_steps":n_steps,"n_contexts_evaluated":payload.get("n_contexts_evaluated"),"n_valid_words":score.get("n_valid_words"),"fitness_summary":score.get("fitness_summary")}


def compute_overall(scores:dict[str,float], aoa_score:float|None, aoa_type:str)->dict[str,Any]:
    cols=["BLiMP","Supplement","EWoK","Entity","COMPS","SuperGLUE","GlobalPIQA","Reading"]
    vals={k:scores.get(k) for k in cols}; vals["AoA"]=aoa_score
    missing=[k for k,v in vals.items() if v is None]
    if missing: return {"complete":False,"missing":missing,"components":vals,"aoa_type":aoa_type}
    return {"complete":True,"overall":sum(float(v) for v in vals.values())/9.0,"components":vals,"aoa_type":aoa_type}


def build_record(label:str,cfg:dict[str,Any])->dict[str,Any]:
    zr=extract_zero_reading_scores(load_json(cfg["zero_reading"]))
    sg=validate_superglue(load_json(cfg["superglue"]))
    scores=dict(zr)
    if sg.get("valid"): scores["SuperGLUE"]=sg["superglue_mean"]
    aoa=validate_aoa(cfg["aoa_manifest"],cfg.get("expected_aoa_steps",18))
    rec={"label":label,"sources":cfg,"zero_reading_scores":zr,"superglue_validation":sg,"aoa_validation":aoa,"scores_without_aoa":scores,"projected_overall_aoa0":compute_overall(scores,0.0,"projected_zero"),"measured_overall":compute_overall(scores,aoa["score"] if aoa.get("measured") else None,"measured_step087_aoa")}
    rec["complete_coordinate_available"]=bool(sg.get("valid") and aoa.get("measured") and rec["measured_overall"].get("complete"))
    return rec


def compare(parent:dict[str,Any], child:dict[str,Any])->dict[str,Any]:
    out={"parent_label":parent["label"],"child_label":child["label"]}
    if not(parent["superglue_validation"].get("valid") and child["superglue_validation"].get("valid")):
        out.update({"status":"incomplete_superglue","parent_superglue":parent["superglue_validation"],"child_superglue":child["superglue_validation"]}); return out
    comp={}
    for k in ["BLiMP","Supplement","EWoK","Entity","COMPS","SuperGLUE","GlobalPIQA","Reading"]:
        if parent["scores_without_aoa"].get(k) is not None and child["scores_without_aoa"].get(k) is not None:
            comp[k]=child["scores_without_aoa"][k]-parent["scores_without_aoa"][k]
    sg_delta={}
    for t in SUPERGLUE_REQUIRED_TASKS:
        p=parent["superglue_validation"].get("per_task",{}).get(t,{}).get("score"); c=child["superglue_validation"].get("per_task",{}).get(t,{}).get("score")
        if p is not None and c is not None: sg_delta[t]={"parent":p,"child":c,"delta":c-p}
    out["component_deltas_without_aoa"]=comp; out["superglue_per_task"]=sg_delta
    for key in ["projected_overall_aoa0","measured_overall"]:
        p=parent[key]; c=child[key]
        out[key]={"complete":bool(p.get("complete") and c.get("complete"))}
        if out[key]["complete"]: out[key].update({"parent_overall":p["overall"],"child_overall":c["overall"],"delta":c["overall"]-p["overall"],"parent_components":p["components"],"child_components":c["components"]})
    if out["measured_overall"].get("complete"):
        out["comparison_state"]="measured_positive" if out["measured_overall"]["delta"]>0 else "measured_not_positive"
    elif out["projected_overall_aoa0"].get("complete"):
        out["comparison_state"]="projected_positive_no_measured" if out["projected_overall_aoa0"]["delta"]>0 else "projected_not_positive_no_measured"
    else: out["comparison_state"]="incomplete_coordinate"
    out["status"]="complete"
    return out


def main():
    records={label:build_record(label,cfg) for label,cfg in MODEL_PATHS.items()}
    comparisons=[]
    for child in ["dense_seed62064","dense_seed62065"]:
        comparisons.append(compare(records["coherent86"],records[child]))
    result={"status":"TRUSTED_COMPARISON_WITH_MEASURED_AOA","created_utc":datetime.now(timezone.utc).isoformat(),"note":"Uses repaired AutoModel SuperGLUE payloads and research complete measured AoA manifests. A measured AoA zero is completed scorer output, not missing evidence. This assembles coordinate evidence only; submission readiness and scientific value remain separate empirical questions.","models":records,"comparisons":comparisons}
    out_json=OUT_ROOT/"trusted_comparison_with_measured_aoa.json"; out_md=(OUT_ROOT.parents[4] / 'research/documents/functional_learning/data/trusted_comparison_with_measured_aoa/trusted_comparison_with_measured_aoa.md')
    out_json.write_text(json.dumps(result,indent=2,ensure_ascii=False,default=str)+"\n")
    lines=["# research trusted comparison with measured AoA\n\n",result["note"]+"\n\n"]
    for label,rec in records.items():
        lines.append(f"## {label}\n- SuperGLUE valid: `{rec['superglue_validation'].get('valid')}` {rec['superglue_validation'].get('reason','')}\n- AoA: `{rec['aoa_validation'].get('status')}` score `{rec['aoa_validation'].get('score')}` estimator `{rec['aoa_validation'].get('estimator_id')}`\n- Projected Overall(AoA0): `{rec['projected_overall_aoa0'].get('overall') if rec['projected_overall_aoa0'].get('complete') else None}`\n- Measured Overall: `{rec['measured_overall'].get('overall') if rec['measured_overall'].get('complete') else None}`\n\n")
    for c in comparisons:
        lines.append(f"## {c.get('parent_label')} vs {c.get('child_label')}\n- Status: `{c.get('status')}`; state `{c.get('comparison_state')}`\n")
        if c.get("measured_overall",{}).get("complete"): lines.append(f"- Measured delta: `{c['measured_overall']['delta']}`\n")
        if c.get("projected_overall_aoa0",{}).get("complete"): lines.append(f"- Projected AoA0 delta: `{c['projected_overall_aoa0']['delta']}`\n")
        lines.append("\n")
    out_md.write_text("".join(lines))
    print(json.dumps({"status":result["status"],"out_json":str(out_json),"out_md":str(out_md),"comparison_states":[c.get("comparison_state") for c in comparisons]},indent=2))

if __name__=="__main__":
    main()
