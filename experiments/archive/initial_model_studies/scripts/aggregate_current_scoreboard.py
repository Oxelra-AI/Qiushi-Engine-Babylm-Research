#!/usr/bin/env python3
"""research current BabyLM coordinate table.

Combines the trusted protected 9/9 coordinate, S1/S2 available columns, research
S1/S2 EWoK, research SuperGLUE if available, Route B 10M, and the public leader
reference. S1/S2 AoA is explicitly marked unavailable from existing artifacts because
`chck_1M` through `chck_9M` are absent.
"""
from __future__ import annotations
import json, pathlib
from typing import Any

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
DATA = ROOT / "data"
OUT_JSON = DATA / "current_scoreboard_status.json"
OUT_NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/current_scoreboard_status.md')

PROTECTED = DATA / "debertav2_b256_true_9of9_coordinate.json"
S1_AVAIL = DATA / "s1_100m_available_coordinate.json"
S2_AVAIL = DATA / "true_s2_100m_available_coordinate.json"
EWOK = DATA / "s1_s2_100m_ewok_scores.json"
SUPER = DATA / "s1_s2_100m_superglue_results.json"
ROUTE_B = DATA / "route_b_threearm_scores.json"
LEADER = DATA / "public_leader_available_coordinate.json"

S1_ROOT = ROOT / "training/runs/babylm_leadershape_s1_100M_aligned_micro128/hf_model"
S2_ROOT = ROOT / "training/runs/babylm_true_s2_100M_wordclock_wwm70_len64256/hf_model"
EXPECTED_AOA_STEPS = [f"chck_{i}M" for i in range(1, 10)] + [f"chck_{i}M" for i in range(10, 101, 10)]


def read_json(p: pathlib.Path) -> dict[str, Any] | None:
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def aoa_artifact_state(root: pathlib.Path) -> dict[str, Any]:
    present = [s for s in EXPECTED_AOA_STEPS if (root / s).exists()]
    missing = [s for s in EXPECTED_AOA_STEPS if not (root / s).exists()]
    return {
        "model_root": str(root),
        "expected_steps": EXPECTED_AOA_STEPS,
        "present_steps": present,
        "missing_steps": missing,
        "true_official_style_aoa_from_existing_artifacts": len(missing) == 0,
        "interpretation": "AoA needs the strict-small checkpoint sequence. Existing S1/S2 roots start at chck_10M, so a true AoA trajectory is unavailable without retraining or accepted nonstandard interpolation.",
    }


def nlp_average(scores: dict[str, float]) -> float | None:
    keys = ["BLiMP", "BLiMP Supplement", "EWoK", "Entity Tracking", "COMPS", "(Super)GLUE", "GlobalPIQA"]
    if not all(k in scores and scores[k] is not None for k in keys):
        return None
    return sum(float(scores[k]) for k in keys) / len(keys)


def human_average(scores: dict[str, float]) -> float | None:
    if not (scores.get("Reading") is not None and scores.get("AoA") is not None):
        return None
    return (float(scores["Reading"]) + float(scores["AoA"])) / 2.0


def overall(scores: dict[str, float]) -> float | None:
    keys = ["BLiMP", "BLiMP Supplement", "EWoK", "Entity Tracking", "COMPS", "(Super)GLUE", "GlobalPIQA", "Reading", "AoA"]
    if not all(k in scores and scores[k] is not None for k in keys):
        return None
    return sum(float(scores[k]) for k in keys) / len(keys)


def avail_to_scores(av: dict[str, Any], ewok: float | None, superglue: float | None, aoa: float | None) -> dict[str, float | None]:
    scores = av["scores"]
    derived = av.get("derived_columns", {})
    gp = derived.get("GlobalPIQA_mean_parallel_nonparallel")
    rd = derived.get("Reading_mean_eye_self_paced") or derived.get("Reading_mean_eye_selfpaced")
    return {
        "BLiMP": scores.get("blimp"),
        "BLiMP Supplement": scores.get("supplement"),
        "EWoK": ewok,
        "Entity Tracking": scores.get("entity_tracking"),
        "COMPS": scores.get("comps"),
        "(Super)GLUE": superglue,
        "GlobalPIQA": gp,
        "Reading": rd,
        "AoA": aoa,
    }


def superglue_value(super_data: dict[str, Any] | None, name: str) -> float | None:
    if not super_data:
        return None
    rec = super_data.get("models", {}).get(name)
    if not rec:
        return None
    if rec.get("num_completed_tasks") == 7 and rec.get("superglue") is not None:
        return float(rec["superglue"])
    return None


def route_b_summary(route_b: dict[str, Any] | None) -> dict[str, Any] | None:
    if not route_b:
        return None
    out = {}
    for name in ["baseline", "surface", "token_id"]:
        sc = route_b.get("models", {}).get(name, {}).get("scores", {})
        if sc:
            out[name] = {
                "BLiMP": sc.get("blimp"),
                "BLiMP Supplement": sc.get("supplement"),
                "EWoK": sc.get("ewok"),
                "Entity Tracking": sc.get("entity_tracking"),
                "COMPS": sc.get("comps"),
                "GlobalPIQA": sc.get("GlobalPIQA_mean"),
                "Reading": sc.get("Reading_mean"),
            }
    return {"scores_10M_available": out, "deltas": route_b.get("deltas")}


def main() -> None:
    protected = read_json(PROTECTED)
    s1_av = read_json(S1_AVAIL)
    s2_av = read_json(S2_AVAIL)
    ewok = read_json(EWOK)
    super_data = read_json(SUPER)
    route_b = read_json(ROUTE_B)
    leader = read_json(LEADER)
    if not protected or not s1_av or not s2_av or not ewok:
        missing = [str(p) for p, v in [(PROTECTED, protected), (S1_AVAIL, s1_av), (S2_AVAIL, s2_av), (EWOK, ewok)] if not v]
        raise FileNotFoundError(f"missing required inputs: {missing}")

    s1_scores = avail_to_scores(s1_av, ewok["models"]["s1"]["ewok"], superglue_value(super_data, "s1"), None)
    s2_scores = avail_to_scores(s2_av, ewok["models"]["s2"]["ewok"], superglue_value(super_data, "s2"), None)
    protected_scores = protected["scores_official_columns"]

    payload = {
        "status": "CURRENT_SCOREBOARD_STATUS",
        "protected_true_9of9": {
            "model": protected.get("model"),
            "scores": protected_scores,
            "NLP Average": protected.get("NLP Average"),
            "Human-like Average": protected.get("Human-like Average"),
            "Overall Average": protected.get("Overall Average"),
            "evidence": str(PROTECTED),
        },
        "s1_100m_coordinate_status": {
            "model_path": s1_av.get("model_path_direct"),
            "scores": s1_scores,
            "NLP Average if SuperGLUE present": nlp_average(s1_scores),
            "Human-like Average": human_average(s1_scores),
            "Overall Average": overall(s1_scores),
            "missing_for_true_9of9": [k for k, v in s1_scores.items() if v is None],
            "aoa_artifact_state": aoa_artifact_state(S1_ROOT),
            "evidence": {"available": str(S1_AVAIL), "ewok": str(EWOK), "superglue": str(SUPER) if SUPER.exists() else None},
        },
        "s2_100m_coordinate_status": {
            "model_path": s2_av.get("model_path_direct"),
            "scores": s2_scores,
            "NLP Average if SuperGLUE present": nlp_average(s2_scores),
            "Human-like Average": human_average(s2_scores),
            "Overall Average": overall(s2_scores),
            "missing_for_true_9of9": [k for k, v in s2_scores.items() if v is None],
            "aoa_artifact_state": aoa_artifact_state(S2_ROOT),
            "evidence": {"available": str(S2_AVAIL), "ewok": str(EWOK), "superglue": str(SUPER) if SUPER.exists() else None},
        },
        "public_leader_reference": leader,
        "route_b_10m": route_b_summary(route_b),
        "scientific_readout": {
            "current_complete_internal_best": "protected_true_9of9",
            "s1_s2_main_target_columns": {
                "s1_minus_protected": {"Entity Tracking": s1_scores["Entity Tracking"] - protected_scores["Entity Tracking"], "EWoK": s1_scores["EWoK"] - protected_scores["EWoK"], "GlobalPIQA": s1_scores["GlobalPIQA"] - protected_scores["GlobalPIQA"]},
                "s2_minus_protected": {"Entity Tracking": s2_scores["Entity Tracking"] - protected_scores["Entity Tracking"], "EWoK": s2_scores["EWoK"] - protected_scores["EWoK"], "GlobalPIQA": s2_scores["GlobalPIQA"] - protected_scores["GlobalPIQA"]},
            },
            "interpretation": "S1 and S2 are not complete 9/9 coordinates from existing artifacts. They are already behind protected on Entity and EWoK; S1/S2 closure mainly documents the model-shape/curriculum result rather than revealing a likely hidden SOTA candidate.",
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# research — Current BabyLM score-coordinate state",
        "",
        f"Evidence JSON: `{OUT_JSON}`",
        "",
        "## Current complete internal coordinate",
        "",
        f"Protected DeBERTa-v2 8×480 WWM remains the only true complete internal 9/9 coordinate: Overall {protected.get('Overall Average'):.4f}.",
        "",
        "## S1/S2 100M coordinate state",
        "",
        "| model | BLiMP | Supp | EWoK | Entity | COMPS | GlobalPIQA | Reading | SuperGLUE | AoA | true 9/9? |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for label, rec in [("S1", payload["s1_100m_coordinate_status"]), ("S2", payload["s2_100m_coordinate_status"])]:
        sc = rec["scores"]
        def fmt(x): return "—" if x is None else f"{float(x):.2f}"
        lines.append(f"| {label} | {fmt(sc['BLiMP'])} | {fmt(sc['BLiMP Supplement'])} | {fmt(sc['EWoK'])} | {fmt(sc['Entity Tracking'])} | {fmt(sc['COMPS'])} | {fmt(sc['GlobalPIQA'])} | {fmt(sc['Reading'])} | {fmt(sc['(Super)GLUE'])} | {fmt(sc['AoA'])} | no |")
    lines += [
        "",
        "S1/S2 AoA cannot be computed in the same way as the trusted protected coordinate from current files because the early checkpoints `chck_1M`–`chck_9M` were not saved. SuperGLUE is running in research and this table should be regenerated after it completes.",
        "",
        "## Route B 10M readout",
        "",
        "Route B surface adapter is not scale-worthy: surface minus token-ID gives Entity +0.02, COMPS +0.36, EWoK -2.67, GlobalPIQA +2.47, Reading 0.00.",
    ]
    OUT_NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT_JSON), "note": str(OUT_NOTE), "s1_missing": payload["s1_100m_coordinate_status"]["missing_for_true_9of9"], "s2_missing": payload["s2_100m_coordinate_status"]["missing_for_true_9of9"]}, indent=2))


if __name__ == "__main__":
    main()
