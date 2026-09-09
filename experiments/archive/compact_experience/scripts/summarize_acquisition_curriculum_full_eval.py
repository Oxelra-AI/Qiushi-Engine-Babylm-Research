#!/usr/bin/env python3
"""Summarize corrected full-eval results for acquisition-curriculum candidates."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import json
import pathlib
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional

ROOT = _public_path('experiments/archive/compact_experience')
PER_TARGET = _public_path('experiments/archive/compact_experience/data/acquisition_curriculum_full_eval_candidates/per_target')
TARGETS_JSON = _public_path('experiments/archive/compact_experience/data/acquisition_curriculum_full_eval_targets.json')
RANKING = _public_path('experiments/archive/compact_experience/data/acquisition_curriculum_trajectory_ranking.json')
CLEAN_SUMMARY = _public_path('experiments/archive/compact_experience/data/full_eval/full_eval_summary.json')
OUT = _public_path('experiments/archive/compact_experience/data/acquisition_curriculum_full_eval_summary.json')
NOTE = _public_path('research/notes/compact_experience/acquisition_curriculum_full_eval_interpretation.md')
LEADER_OVERALL = 41.8
KEYS9 = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "SuperGLUE", "Reading", "AoA"]
KEYS7 = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]


def read_json(p: pathlib.Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def normalize_scores(scores: Dict[str, Any]) -> Dict[str, Optional[float]]:
    return {k: (None if scores.get(k) is None else float(scores.get(k))) for k in KEYS9}


def overall_block(d: Dict[str, Any]) -> Dict[str, Any]:
    return d.get("official_overall") or {}


def scores(d: Dict[str, Any]) -> Dict[str, Optional[float]]:
    o=overall_block(d)
    return normalize_scores(o.get("scores") or o.get("task_scores") or d.get("scores") or d.get("task_scores") or {})


def overall(d: Dict[str, Any]) -> Optional[float]:
    o=overall_block(d)
    v=o.get("Overall")
    if v is None: v=o.get("official_style_overall")
    if v is None: v=d.get("Overall") or d.get("overall") or d.get("official_style_overall")
    return None if v is None else float(v)


def mean_available(sc: Dict[str, Optional[float]], keys: Iterable[str]) -> Optional[float]:
    vals=[sc.get(k) for k in keys]
    if any(v is None for v in vals): return None
    return mean(float(v) for v in vals)  # type: ignore[arg-type]


def clean() -> Dict[str, Any]:
    d=read_json(CLEAN_SUMMARY) if CLEAN_SUMMARY.exists() else {}
    rec=(d.get("targets") or {}).get("qwen_clean_aligned", {}) if isinstance(d.get("targets"), dict) else {}
    if not rec: return {"target":"qwen_clean_aligned","overall":41.34429066479573,"scores":{}}
    return {"target":"qwen_clean_aligned","overall":float(rec.get("Overall", rec.get("overall", 41.34429066479573))),"scores":normalize_scores(rec.get("scores") or rec)}


def row_from_file(p: pathlib.Path) -> Dict[str, Any]:
    d=read_json(p); sc=scores(d); ov=overall(d)
    row={"target":str(d.get("target") or p.stem),"endpoint":str(d.get("endpoint") or ""),"family":str(d.get("family") or ""),"overall":ov,"scores":sc,"equal7_from_full_eval":mean_available(sc,KEYS7),"complete_for_provisional_overall":overall_block(d).get("complete_for_provisional_overall"),"submit_ready_aoa":overall_block(d).get("submit_ready_aoa"),"aoa_raw_correlation":overall_block(d).get("aoa_raw_correlation"),"aoa_leaderboard_score":overall_block(d).get("aoa_leaderboard_score")}
    if row["equal7_from_full_eval"] is not None:
        row["required_superglue_plus_aoa_to_exceed_41p8_from_equal7"] = 9*LEADER_OVERALL - 7*float(row["equal7_from_full_eval"])
    if ov is not None:
        row["delta_vs_visible_41p8_leader"] = ov - LEADER_OVERALL
    return row


def diff(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out={"a":a.get("target"),"b":b.get("target"),"a_minus_b_overall":None}
    if a.get("overall") is not None and b.get("overall") is not None: out["a_minus_b_overall"]=float(a["overall"])-float(b["overall"])
    for k in ["equal7_from_full_eval"]+KEYS9:
        av=(a.get("scores") or {}).get(k) if k in KEYS9 else a.get(k)
        bv=(b.get("scores") or {}).get(k) if k in KEYS9 else b.get(k)
        if av is not None and bv is not None: out[f"delta_{k}"]=float(av)-float(bv)
    return out


def main() -> None:
    rows=[row_from_file(p) for p in sorted(PER_TARGET.glob("*.json"))] if PER_TARGET.exists() else []
    ref=clean(); by={r["target"]:r for r in rows}
    ranked=sorted([r for r in rows if r.get("overall") is not None], key=lambda r: float(r["overall"]), reverse=True)
    contrasts=[]
    for suf in ["10M","20M","30M","40M","50M","60M","70M","75M","80M","85M","90M","95M","100M"]:
        a2=by.get(f"acq_A2_full_{suf}")
        for fam in ["acq_A0_baseline_order","acq_A0p_repeated_random","acq_A1_frequency_only","acq_A3_random_label","acq_A4_reverse"]:
            b=by.get(f"{fam}_{suf}")
            if a2 and b: contrasts.append(diff(a2,b))
    payload={"status":"ACQUISITION_CURRICULUM_FULL_EVAL_SUMMARY","interpretation_scope":"Reads completed full-eval files for no-AoA frozen acquisition-curriculum checkpoints. AoA is final measurement only.","visible_leader_overall":LEADER_OVERALL,"per_target_dir":str(PER_TARGET),"ranking_source":str(RANKING),"selected_targets_source":str(TARGETS_JSON),"selected_targets":read_json(TARGETS_JSON) if TARGETS_JSON.exists() else {},"clean_reference":ref,"rows":rows,"ranked_all_by_overall":ranked,"candidate_minus_clean_reference":[diff(r,ref) for r in rows if r.get("overall") is not None],"a2_minus_controls_same_checkpoint":contrasts}
    _public_path('experiments/archive/compact_experience/data').mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    lines=["# research acquisition-curriculum full-eval interpretation","",payload["interpretation_scope"],"",f"Visible leader reference: {LEADER_OVERALL:.4f}",f"Clean-Qwen reference: {ref.get('overall')}","","## Ranked targets"]
    if not ranked: lines.append("No completed acquisition full-eval files found yet.")
    for i,r in enumerate(ranked,1):
        s=r["scores"]
        lines.append(f"{i}. `{r['target']}` `{r['endpoint']}` family={r['family']} Overall={float(r['overall']):.6f} delta_vs_41.8={float(r.get('delta_vs_visible_41p8_leader',0.0)):.6f} equal7={r.get('equal7_from_full_eval')} SuperGLUE={s.get('SuperGLUE')} AoA={s.get('AoA')} need_SG_plus_AoA>{r.get('required_superglue_plus_aoa_to_exceed_41p8_from_equal7')}")
    lines += ["", "## Candidate minus clean-Qwen"]
    for d in payload["candidate_minus_clean_reference"]: lines.append("- "+json.dumps(d, ensure_ascii=False, sort_keys=True))
    lines += ["", "## A2 minus controls same checkpoint"]
    for d in contrasts: lines.append("- "+json.dumps(d, ensure_ascii=False, sort_keys=True))
    _public_path('research/notes/compact_experience').mkdir(parents=True, exist_ok=True); NOTE.write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"out":str(OUT),"note":str(NOTE),"best":ranked[:1]}, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
