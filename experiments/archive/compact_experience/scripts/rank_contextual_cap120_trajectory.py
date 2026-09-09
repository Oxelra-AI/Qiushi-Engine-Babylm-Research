#!/usr/bin/env python3
"""Rank contextual cap-120 no-AoA trajectory screen against clean-Qwen and matched control.

The ranker uses only BLiMP/Supplement/EWoK/Entity/COMPS/GlobalPIQA/Reading.  It also
computes two interpretable profiles: R = recovery columns (BLiMP, EWoK, COMPS,
Reading) and P = preserve columns (Supplement, Entity, SuperGLUE is unavailable here
until full eval, so P_no_sglue uses Supplement+Entity).  SuperGLUE and AoA are never
used here.
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

WORKSPACE = _public_path('experiments/archive/compact_experience')
DEFAULT_ROOT = _public_path('experiments/archive/compact_experience/data/contextual_cap120_trajectory_screen')
CLEAN_ROOT = _public_path('experiments/archive/compact_experience/data/trajectory_screen')
OUT = _public_path('experiments/archive/compact_experience/data/contextual_cap120_trajectory_ranking.json')
NOTE = _public_path('research/notes/compact_experience/contextual_cap120_trajectory_ranking.md')
KEYS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
RECOVERY = ["BLiMP", "EWoK", "COMPS", "Reading"]
PRESERVE_NO_SGLUE = ["Supplement", "Entity"]


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
        rec: Dict[str, Any] = {"target": target, "family": family, "checkpoint": ckpt, "equal7_full_eval": row.get("equal7_full_eval")}
        for k in KEYS:
            rec[k] = row.get(k)
        rec["complete"] = all(rec.get(k) is not None for k in KEYS) and rec.get("equal7_full_eval") is not None
        if rec["complete"]:
            rec["recovery_R"] = mean(float(rec[k]) for k in RECOVERY)
            rec["preserve_no_superglue_P"] = mean(float(rec[k]) for k in PRESERVE_NO_SGLUE)
        out.append(rec)
    return out


def best_for(rows: List[Dict[str, Any]], target: str, key: str = "equal7_full_eval") -> Dict[str, Any] | None:
    xs = [r for r in rows if r.get("target") == target and r.get("complete") and r.get(key) is not None]
    return max(xs, key=lambda r: float(r[key]), default=None)


def paired_treat_control(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    treat = [r for r in rows if r.get("target") == "context_cap120_treat_seed43022" and r.get("complete")]
    ctrl = [r for r in rows if r.get("target") == "context_cap120_control_seed43022" and r.get("complete")]
    by_ctrl = {r["checkpoint"]: r for r in ctrl}
    out = []
    for t in treat:
        c = by_ctrl.get(t["checkpoint"])
        if not c:
            continue
        rec: Dict[str, Any] = {
            "checkpoint": t["checkpoint"],
            "treat_equal7": t["equal7_full_eval"],
            "control_equal7": c["equal7_full_eval"],
            "delta_equal7": float(t["equal7_full_eval"]) - float(c["equal7_full_eval"]),
            "treat_recovery_R": t.get("recovery_R"),
            "control_recovery_R": c.get("recovery_R"),
            "delta_recovery_R": float(t.get("recovery_R")) - float(c.get("recovery_R")),
            "treat_preserve_no_superglue_P": t.get("preserve_no_superglue_P"),
            "control_preserve_no_superglue_P": c.get("preserve_no_superglue_P"),
            "delta_preserve_no_superglue_P": float(t.get("preserve_no_superglue_P")) - float(c.get("preserve_no_superglue_P")),
        }
        for k in KEYS:
            rec[f"delta_{k}"] = float(t[k]) - float(c[k])
            rec[f"treat_{k}"] = t[k]
            rec[f"control_{k}"] = c[k]
        out.append(rec)
    return sorted(out, key=lambda r: (float(r["delta_equal7"]), float(r["treat_equal7"])), reverse=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--clean_root", default=str(CLEAN_ROOT))
    ap.add_argument("--top_k", type=int, default=12)
    args = ap.parse_args()
    root = pathlib.Path(args.root)
    clean_root = pathlib.Path(args.clean_root)
    rows: List[Dict[str, Any]] = []
    rows += rows_from_summary(root / "context_cap120_treat_seed43022_trajectory_summary.json", "context_cap120_treatment")
    rows += rows_from_summary(root / "context_cap120_control_seed43022_trajectory_summary.json", "context_cap120_control")
    rows += rows_from_summary(clean_root / "clean_qwen_seed43022_trajectory_summary.json", "clean_qwen_reference")
    complete = [r for r in rows if r.get("complete")]
    ranked = sorted(complete, key=lambda r: float(r["equal7_full_eval"]), reverse=True)
    paired = paired_treat_control(complete)
    payload = {
        "status": "CONTEXTUAL_CAP120_TRAJECTORY_RANKING",
        "non_leakage_statement": "Ranks only BLiMP/Supplement/EWoK/Entity/COMPS/GlobalPIQA/Reading. AoA/CDI words, child curves, AoA predictions, AoA outputs, and SuperGLUE are not used.",
        "root": str(root),
        "clean_reference_root": str(clean_root),
        "num_rows": len(rows),
        "num_complete_rows": len(complete),
        "best_treatment_equal7": best_for(complete, "context_cap120_treat_seed43022", "equal7_full_eval"),
        "best_treatment_recovery_R": best_for(complete, "context_cap120_treat_seed43022", "recovery_R"),
        "best_control_equal7": best_for(complete, "context_cap120_control_seed43022", "equal7_full_eval"),
        "best_clean_reference_equal7": best_for(complete, "clean_qwen_seed43022", "equal7_full_eval"),
        "top_rows": ranked[: args.top_k],
        "paired_treat_control_by_checkpoint": paired,
        "recommendation_rule": "Promote treatment checkpoint(s) for full nine-column eval if equal7 is competitive with clean-Qwen or if R={BLiMP,EWoK,COMPS,Reading} recovers while P_no_sglue={Supplement,Entity} does not collapse; evaluate matched control at the same checkpoint or nearest no-AoA control optimum. AoA is final measurement only.",
    }
    _public_path('experiments/archive/compact_experience/data').mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research contextual cap-120 trajectory ranking", "", payload["non_leakage_statement"], "", f"Complete rows: {len(complete)} / {len(rows)}"]
    bt = payload["best_treatment_equal7"]
    bc = payload["best_clean_reference_equal7"]
    if bt:
        lines.append(f"Best treatment equal7: `{bt['checkpoint']}` {float(bt['equal7_full_eval']):.6f} R={float(bt['recovery_R']):.6f} P_no_sglue={float(bt['preserve_no_superglue_P']):.6f}")
    if bc:
        lines.append(f"Best clean reference equal7: `{bc['checkpoint']}` {float(bc['equal7_full_eval']):.6f} R={float(bc['recovery_R']):.6f} P_no_sglue={float(bc['preserve_no_superglue_P']):.6f}")
    lines += ["", "## Top rows"]
    for i, r in enumerate(ranked[: args.top_k], 1):
        lines.append(f"{i}. `{r['family']}` `{r['target']}` `{r['checkpoint']}` equal7={float(r['equal7_full_eval']):.6f} R={float(r['recovery_R']):.6f} P_no_sglue={float(r['preserve_no_superglue_P']):.6f} BLiMP={r.get('BLiMP')} Supplement={r.get('Supplement')} EWoK={r.get('EWoK')} Entity={r.get('Entity')} COMPS={r.get('COMPS')} GlobalPIQA={r.get('GlobalPIQA')} Reading={r.get('Reading')}")
    lines += ["", "## Treatment minus matched control by checkpoint"]
    for r in paired[: args.top_k]:
        lines.append(f"- `{r['checkpoint']}` delta_equal7={r['delta_equal7']:.6f} treat={float(r['treat_equal7']):.6f} control={float(r['control_equal7']):.6f} delta_R={r['delta_recovery_R']:.6f} delta_P_no_sglue={r['delta_preserve_no_superglue_P']:.6f} delta_BLiMP={r['delta_BLiMP']:.3f} delta_Supplement={r['delta_Supplement']:.3f} delta_EWoK={r['delta_EWoK']:.3f} delta_Entity={r['delta_Entity']:.3f} delta_COMPS={r['delta_COMPS']:.3f} delta_GlobalPIQA={r['delta_GlobalPIQA']:.3f} delta_Reading={r['delta_Reading']:.3f}")
    _public_path('research/notes/compact_experience').mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT), "note": str(NOTE), "best_treatment_equal7": payload["best_treatment_equal7"], "best_clean_reference_equal7": payload["best_clean_reference_equal7"], "top_paired": paired[: min(args.top_k, len(paired))]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
