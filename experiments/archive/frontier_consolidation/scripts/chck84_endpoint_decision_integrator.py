#!/usr/bin/env python3
"""research: integrate chck_84M endpoint evidence when SuperGLUE arrives.

This CPU/file-only tool reads the cheap7 selected-grid row, protected chck_82M
verification, chck_84M identity/item/phase analyses, and (if present) the research
SuperGLUE-only summary.  It produces a single endpoint-branch decision record.
It does not run evaluation, upload to Hugging Face, or submit to the leaderboard.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import math
import pathlib
import time
from statistics import mean
from typing import Any

CHEAP_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
ALPHA075_PROJECTED = 42.1210247099666
ALPHA075_CHEAP7 = 44.18142857142857
ALPHA075_SUPERGLUE = 69.81922238969935


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
WORKSPACE = USER_ROOT / "experiments/archive/frontier_consolidation"
DEFAULT_CHEAP = WORKSPACE / "data/selected_trajectory_eval_scale1p75_seed43022_reference_common2M/selected_trajectory.json"
DEFAULT_SUPERGLUE = WORKSPACE / "data/reference_chck84_superglue_summary/scale1p75_seed43022_reference_chck84_superglue_superglue_summary.json"
DEFAULT_CHCK82 = WORKSPACE / "data/chck82_independent_verification/chck82_independent_verification.json"
DEFAULT_IDENTITY = WORKSPACE / "data/chck84_identity/chck_84M_identity_audit.json"
DEFAULT_ITEM = WORKSPACE / "data/chck84_item_movement_analysis/chck84_item_movement_summary.json"
DEFAULT_PHASE = WORKSPACE / "data/phase_lag_score_analysis/phase_lag_score_analysis.json"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: pathlib.Path | str) -> str:
    p = pathlib.Path(path)
    try:
        return str(p.resolve().relative_to(USER_ROOT))
    except Exception:
        return str(p)


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: pathlib.Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def fnum(x: Any) -> float | None:
    if x is None or x == "":
        return None
    try:
        y = float(x)
    except Exception:
        return None
    if math.isnan(y):
        return None
    return y


def load_cheap_row(path: pathlib.Path, endpoint: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    data = read_json(path)
    if isinstance(data, dict) and isinstance(data.get("rows"), list):
        data = data["rows"]
    if not isinstance(data, list):
        return None
    for raw in data:
        if isinstance(raw, dict) and raw.get("endpoint") == endpoint:
            row = dict(raw)
            scores = {c: fnum(row.get(c)) for c in CHEAP_COLUMNS}
            if all(v is not None for v in scores.values()):
                row["scores"] = {c: float(scores[c]) for c in CHEAP_COLUMNS}  # type: ignore[index]
                row["cheap7_recomputed"] = float(mean(row["scores"].values()))
                row["cheap7"] = fnum(row.get("cheap7")) or row["cheap7_recomputed"]
            return row
    return None


def load_chck82(path: pathlib.Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    j = read_json(path)
    scores = j.get("score_arithmetic", {}).get("scores", {})
    if not isinstance(scores, dict):
        return None
    out = {
        "overall": fnum(j.get("score_arithmetic", {}).get("overall_reported")),
        "cheap7": fnum(j.get("score_arithmetic", {}).get("cheap7_reported")),
        "superglue": fnum(scores.get("SuperGLUE")),
        "aoa": fnum(scores.get("AoA")) or 0.0,
        "scores": {k: fnum(v) for k, v in scores.items()},
        "source": rel(path),
    }
    return out


def compute_overall(cheap_scores: dict[str, float], superglue: float, aoa: float = 0.0) -> float:
    return float(mean([cheap_scores[c] for c in CHEAP_COLUMNS] + [superglue, aoa]))


def read_optional(path: pathlib.Path) -> tuple[bool, Any | None, str | None]:
    if not path.exists():
        return False, None, None
    try:
        return True, read_json(path), None
    except Exception as e:
        return True, None, repr(e)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="chck_84M")
    ap.add_argument("--cheap-trajectory", type=pathlib.Path, default=DEFAULT_CHEAP)
    ap.add_argument("--superglue-summary", type=pathlib.Path, default=DEFAULT_SUPERGLUE)
    ap.add_argument("--chck82", type=pathlib.Path, default=DEFAULT_CHCK82)
    ap.add_argument("--identity", type=pathlib.Path, default=DEFAULT_IDENTITY)
    ap.add_argument("--item-summary", type=pathlib.Path, default=DEFAULT_ITEM)
    ap.add_argument("--phase-summary", type=pathlib.Path, default=DEFAULT_PHASE)
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cheap = load_cheap_row(args.cheap_trajectory, args.endpoint)
    chck82 = load_chck82(args.chck82)
    sg_exists, sg, sg_error = read_optional(args.superglue_summary)
    id_exists, ident, id_error = read_optional(args.identity)
    item_exists, item, item_error = read_optional(args.item_summary)
    phase_exists, phase, phase_error = read_optional(args.phase_summary)

    status = "PENDING_SUPERGLUE" if sg is None else "COMPLETE_WITH_SUPERGLUE"
    if cheap is None or chck82 is None:
        status = "MISSING_REQUIRED_CHEAP_OR_REFERENCE"

    decision: dict[str, Any] = {
        "status": status,
        "created_utc": now(),
        "endpoint": args.endpoint,
        "inputs": {
            "cheap_trajectory": {"path": rel(args.cheap_trajectory), "exists": args.cheap_trajectory.exists()},
            "superglue_summary": {"path": rel(args.superglue_summary), "exists": sg_exists, "error": sg_error},
            "chck82_reference": {"path": rel(args.chck82), "exists": args.chck82.exists()},
            "identity_audit": {"path": rel(args.identity), "exists": id_exists, "error": id_error},
            "item_summary": {"path": rel(args.item_summary), "exists": item_exists, "error": item_error},
            "phase_summary": {"path": rel(args.phase_summary), "exists": phase_exists, "error": phase_error},
        },
        "cheap_row": cheap,
        "protected_chck82": chck82,
        "coherent86_alpha075_reference": {
            "projected_overall_aoa0": ALPHA075_PROJECTED,
            "cheap7": ALPHA075_CHEAP7,
            "superglue": ALPHA075_SUPERGLUE,
            "classification": "separate stronger local endpoint; mechanism classified as amplitude-controlled redistribution",
        },
        "superglue_result": None,
        "endpoint_arithmetic": None,
        "evidence_reading": {},
        "next_actions": [],
        "leaderboard_submission_allowed_for_agent": False,
    }

    if cheap is not None and chck82 is not None:
        thresholds = {
            "superglue_needed_to_beat_chck82_overall_with_aoa0": float(9.0 * float(chck82["overall"]) - 7.0 * float(cheap["cheap7"])),
            "superglue_needed_for_overall_42p0_with_aoa0": float(9.0 * 42.0 - 7.0 * float(cheap["cheap7"])),
            "superglue_needed_to_match_alpha075_projected_42p1210247099666_with_aoa0": float(9.0 * ALPHA075_PROJECTED - 7.0 * float(cheap["cheap7"])),
        }
        decision["thresholds"] = thresholds

    if isinstance(sg, dict):
        superglue = fnum(sg.get("superglue"))
        overall = fnum(sg.get("projected_overall_with_aoa0"))
        if superglue is not None and cheap is not None and cheap.get("scores") and overall is None:
            overall = compute_overall(cheap["scores"], superglue, 0.0)
        decision["superglue_result"] = {
            "superglue": superglue,
            "projected_overall_with_aoa0": overall,
            "summary_status": sg.get("status"),
            "source": rel(args.superglue_summary),
            "payload_path": sg.get("payload_path"),
            "elapsed_sec": sg.get("elapsed_sec"),
        }
        if superglue is not None and overall is not None and cheap is not None and chck82 is not None:
            decision["endpoint_arithmetic"] = {
                "cheap7": float(cheap["cheap7"]),
                "superglue": superglue,
                "aoa_assumed": 0.0,
                "projected_overall_with_aoa0": overall,
                "delta_vs_chck82_overall": float(overall - float(chck82["overall"])),
                "delta_vs_42p0": float(overall - 42.0),
                "delta_vs_alpha075_projected": float(overall - ALPHA075_PROJECTED),
                "beats_chck82": bool(overall > float(chck82["overall"])),
                "beats_42p0": bool(overall >= 42.0),
                "matches_or_beats_alpha075_projected": bool(overall >= ALPHA075_PROJECTED),
            }
            if overall > float(chck82["overall"]):
                decision["next_actions"].append("If identity/integrity checks remain clean, build and locally validate a faithful trusted-code HF bundle for chck_84M; do not submit to the leaderboard.")
            else:
                decision["next_actions"].append("Do not package chck_84M as successor; keep item/phase evidence as trajectory characterization.")
    else:
        decision["next_actions"].append("Wait for managed SuperGLUE task delivery; do not infer Overall before the summary exists.")

    if isinstance(ident, dict):
        decision["evidence_reading"]["identity"] = {
            "within_10_epoch_limit": ident.get("within_10_epoch_limit"),
            "model_sha256": ident.get("endpoint_files", {}).get("model.safetensors", {}).get("sha256"),
            "static_same_as_reference": ident.get("all_static_files_same_as_reference_except_model"),
            "cpu_loadability": ident.get("cpu_loadability"),
            "source": rel(args.identity),
        }
    if isinstance(item, dict):
        cols = item.get("official_like_column_scores_from_predictions", {})
        # Some fields are nested by column; keep main deltas from known summary if present.
        deltas = {}
        try:
            for col, block in cols.items():
                if isinstance(block, dict) and block.get("chck_84M") is not None and block.get("chck_82M") is not None:
                    deltas[col] = float(block["chck_84M"] - block["chck_82M"])
        except Exception:
            deltas = {}
        decision["evidence_reading"]["item_movement"] = {
            "n_classification_items": item.get("n_classification_items"),
            "churn_by_column_raw_items": item.get("churn_by_column_raw_items"),
            "deltas_84_minus_82_from_recomputed_predictions": deltas,
            "source": rel(args.item_summary),
            "reading": "research found 5/6 classification columns positive and cheap6 without GlobalPIQA positive; 84M is broad-but-shallow, not GlobalPIQA-only.",
        }
    if isinstance(phase, dict):
        decision["evidence_reading"]["phase_lag"] = {
            "n_scored_intervals": phase.get("n_scored_intervals"),
            "local_78_to_92M_sequence": phase.get("local_78_to_92M_sequence"),
            "reading": phase.get("scientific_reading"),
            "source": rel(args.phase_summary),
        }

    if sg is None:
        decision["endpoint_arithmetic"] = {
            "status": "pending_superglue",
            "known_cheap7": cheap.get("cheap7") if cheap else None,
            "thresholds": decision.get("thresholds"),
        }

    # Always preserve the mechanism boundary.
    decision["mechanism_boundary"] = (
        "chck_84M is an endpoint branch inside one scale1.75 seed43022 trajectory. Even if it improves Overall, it does not by itself establish a transferable law; seed43122 common-grid scoring and A01 directional-fork evidence remain prior before any new A02 ordering H100 screen."
    )

    out_json = args.out_dir / "chck84_endpoint_decision_integrated.json"
    out_md = args.out_dir / "chck84_endpoint_decision_integrated.md"
    write_json(out_json, decision)
    lines = ["# research chck_84M endpoint decision integrator\n\n"]
    lines.append(f"Status: **{decision['status']}**. Endpoint `{args.endpoint}`.\n\n")
    if decision.get("thresholds"):
        th = decision["thresholds"]
        lines.append("## SuperGLUE thresholds (AoA=0)\n\n")
        lines.append(f"- Beat chck82: `{th['superglue_needed_to_beat_chck82_overall_with_aoa0']:.6f}`.\n")
        lines.append(f"- Reach Overall 42.0: `{th['superglue_needed_for_overall_42p0_with_aoa0']:.6f}`.\n")
        lines.append(f"- Match alpha0.75 projected: `{th['superglue_needed_to_match_alpha075_projected_42p1210247099666_with_aoa0']:.6f}`.\n\n")
    if decision.get("endpoint_arithmetic"):
        lines.append("## Endpoint arithmetic\n\n")
        lines.append("```json\n" + json.dumps(decision["endpoint_arithmetic"], indent=2, ensure_ascii=False) + "\n```\n\n")
    lines.append("## Scientific boundary\n\n" + decision["mechanism_boundary"] + "\n\n")
    if decision["next_actions"]:
        lines.append("## Next actions\n\n")
        for a in decision["next_actions"]:
            lines.append(f"- {a}\n")
        lines.append("\n")
    lines.append(f"JSON: `{rel(out_json)}`\n")
    out_md.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": decision["status"], "endpoint": args.endpoint, "superglue_present": isinstance(sg, dict), "out_json": rel(out_json), "out_md": rel(out_md)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
