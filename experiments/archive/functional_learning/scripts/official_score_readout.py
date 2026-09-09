#!/usr/bin/env python3
"""research: score-level readout for dense-focus official-compatible evaluation.

This CPU-only helper compares the dense official-compatible payload, even while it
is incomplete, against the true coherent86 alpha0.75 reference coordinate.  It does
not run evaluation.  When the dense payload is complete, re-run this script and the
research transition comparator with --include-superglue.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

ROOT = pathlib.Path(".").resolve()
COH_ZERO = pathlib.Path("experiments/archive/frontier_consolidation/data/private_scale_sentinel_eval/coherent86_private_scale_0p75/per_target/coherent86_private_scale_0p75.json")
COH_SG = pathlib.Path("experiments/archive/frontier_consolidation/data/private_scale_superglue_eval/coherent86_private_scale_0p75/per_target/coherent86_private_scale_0p75_sg_retry.json")
DENSE_DEFAULT = pathlib.Path("experiments/archive/functional_learning/data/dense_focus_official_eval_seed62064/per_target/dense_focus_seed62064_u0080.json")
OFFICIAL_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "SuperGLUE", "GlobalPIQA", "Reading", "AoA"]
NON_AOA_COLUMNS = [c for c in OFFICIAL_COLUMNS if c != "AoA"]
CHEAP7_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
ZERO_FAMILY_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]
SG_TASKS = ["boolq", "mnli", "mrpc", "multirc", "qqp", "rte", "wsc"]


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def task_present(payload: dict[str, Any], col: str) -> bool:
    tasks = payload.get("tasks", {})
    if col in tasks:
        return True
    if col == "GlobalPIQA":
        return "GlobalPIQA_parallel" in tasks and "GlobalPIQA_nonparallel" in tasks
    return False


def task_complete(payload: dict[str, Any], col: str) -> bool:
    tasks = payload.get("tasks", {})
    if col == "GlobalPIQA":
        return task_complete(payload, "GlobalPIQA_parallel") and task_complete(payload, "GlobalPIQA_nonparallel")
    t = tasks.get(col)
    if not isinstance(t, dict):
        return False
    if col == "SuperGLUE":
        sub = t.get("tasks") or []
        return len(sub) == len(SG_TASKS) and t.get("superglue_mean") is not None and all(isinstance(x, dict) and x.get("returncode") == 0 for x in sub)
    if col == "AoA":
        # For single-endpoint evaluation, AoA may be recorded as not available.
        return True
    if col == "Reading":
        return t.get("returncode") == 0 and isinstance(t.get("scores"), dict) and t["scores"].get("Reading") is not None
    return t.get("returncode") == 0 and t.get("score") is not None


def score(payload: dict[str, Any], col: str) -> float | None:
    overall = payload.get("official_overall", {}).get("scores", {})
    if isinstance(overall, dict) and overall.get(col) is not None:
        return float(overall[col])
    tasks = payload.get("tasks", {})
    if col == "GlobalPIQA":
        a = score(payload, "GlobalPIQA_parallel")
        b = score(payload, "GlobalPIQA_nonparallel")
        if a is not None and b is not None:
            return (a + b) / 2.0
        return None
    if col == "Reading":
        t = tasks.get("Reading")
        if isinstance(t, dict):
            s = t.get("scores", {})
            if isinstance(s, dict) and s.get("Reading") is not None:
                return float(s["Reading"])
    if col == "SuperGLUE":
        t = tasks.get("SuperGLUE")
        if isinstance(t, dict) and t.get("superglue_mean") is not None:
            return float(t["superglue_mean"])
    t = tasks.get(col)
    if isinstance(t, dict) and t.get("score") is not None:
        return float(t["score"])
    return None


def coherent_scores() -> dict[str, float | None]:
    z = load_json(COH_ZERO)
    sg = load_json(COH_SG)
    out = {c: score(z, c) for c in OFFICIAL_COLUMNS}
    out["SuperGLUE"] = score(sg, "SuperGLUE")
    out["AoA"] = 0.0
    return out


def payload_scores(payload: dict[str, Any]) -> dict[str, float | None]:
    out = {c: score(payload, c) for c in OFFICIAL_COLUMNS}
    # Single checkpoint AoA is unavailable; use zero only in projected arithmetic, not as a real measured score.
    return out


def mean(vals: list[float]) -> float:
    return sum(vals) / len(vals)


def overall_with_aoa0(scores: dict[str, float | None]) -> float | None:
    vals = []
    for c in OFFICIAL_COLUMNS:
        v = scores.get(c)
        if c == "AoA" and v is None:
            v = 0.0
        if v is None:
            return None
        vals.append(float(v))
    return mean(vals)


def cheap7(scores: dict[str, float | None]) -> float | None:
    vals = [scores.get(c) for c in CHEAP7_COLUMNS]
    if any(v is None for v in vals):
        return None
    return mean([float(v) for v in vals if v is not None])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dense-payload", default=str(DENSE_DEFAULT))
    ap.add_argument("--out-root", default="experiments/archive/functional_learning/data/dense_official_score_readout")
    args = ap.parse_args()
    dense_path = pathlib.Path(args.dense_payload)
    out_root = pathlib.Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    ref = coherent_scores()
    dense_payload = load_json(dense_path)
    den = payload_scores(dense_payload)
    den_aoa0 = dict(den)
    if den_aoa0.get("AoA") is None:
        den_aoa0["AoA"] = 0.0

    columns = []
    known_delta_sum = 0.0
    known_non_aoa = []
    unknown_non_aoa = []
    for c in OFFICIAL_COLUMNS:
        r = ref.get(c)
        d = den.get(c)
        delta = None if r is None or d is None else d - r
        if c != "AoA":
            if delta is None:
                unknown_non_aoa.append(c)
            else:
                known_delta_sum += float(delta)
                known_non_aoa.append(c)
        columns.append({
            "column": c,
            "reference": r,
            "dense": d,
            "delta_dense_minus_reference": delta,
            "present_in_dense_payload": task_present(dense_payload, c),
            "complete_in_dense_payload": task_complete(dense_payload, c),
        })

    ref_overall = overall_with_aoa0(ref)
    dense_overall = overall_with_aoa0(den_aoa0)
    full_delta_needed_remaining = -known_delta_sum
    mean_needed_per_unknown_non_aoa = None if not unknown_non_aoa else full_delta_needed_remaining / len(unknown_non_aoa)
    result = {
        "status": "DENSE_OFFICIAL_SCORE_READOUT",
        "dense_payload": rel(dense_path),
        "reference_payloads": {"zero_reading": rel(COH_ZERO), "superglue": rel(COH_SG)},
        "reference_scores": ref,
        "dense_scores_seen": den,
        "columns": columns,
        "known_non_aoa_columns": known_non_aoa,
        "unknown_non_aoa_columns": unknown_non_aoa,
        "known_non_aoa_delta_sum": known_delta_sum,
        "remaining_sum_delta_needed_to_match_reference_overall_aoa0": full_delta_needed_remaining,
        "mean_delta_needed_per_unknown_non_aoa_column": mean_needed_per_unknown_non_aoa,
        "reference_cheap7": cheap7(ref),
        "dense_cheap7_if_all_seen": cheap7(den),
        "reference_projected_overall_aoa0": ref_overall,
        "dense_projected_overall_aoa0_if_all_seen": dense_overall,
        "projected_delta_if_all_seen": None if ref_overall is None or dense_overall is None else dense_overall - ref_overall,
        "interpretation": "This is a score readout from completed entries in the dense payload. Unknown columns must not be inferred from fast-screen results; the remaining-delta values only state what the unknown official columns would need to contribute for dense to equal or exceed coherent86 under AoA=0 arithmetic.",
    }

    out_json = out_root / "dense_official_score_readout.json"
    out_md = out_root / "dense_official_score_readout.md"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = []
    lines.append("# research dense official score readout\n")
    lines.append(f"Dense payload: `{rel(dense_path)}`")
    lines.append(f"Reference zero/Reading: `{rel(COH_ZERO)}`")
    lines.append(f"Reference SuperGLUE: `{rel(COH_SG)}`\n")
    lines.append("| column | coherent86 | dense seen | dense-reference | present | complete |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in columns:
        def fmt(x: Any) -> str:
            return "" if x is None else f"{float(x):.6f}"
        def fmtd(x: Any) -> str:
            return "" if x is None else f"{float(x):+.6f}"
        lines.append(f"| {row['column']} | {fmt(row['reference'])} | {fmt(row['dense'])} | {fmtd(row['delta_dense_minus_reference'])} | {row['present_in_dense_payload']} | {row['complete_in_dense_payload']} |")
    lines.append("\n## Arithmetic state\n")
    lines.append(f"- Known non-AoA delta sum: {known_delta_sum:+.6f} over {known_non_aoa}.")
    lines.append(f"- Unknown non-AoA columns: {unknown_non_aoa}.")
    if mean_needed_per_unknown_non_aoa is not None:
        lines.append(f"- Remaining total delta needed to equal coherent86 Overall(AoA0): {full_delta_needed_remaining:+.6f}, mean {mean_needed_per_unknown_non_aoa:+.6f} over each unknown non-AoA column.")
    if dense_overall is not None:
        lines.append(f"- Dense projected Overall(AoA0): {dense_overall:.12f}; delta {dense_overall - ref_overall:+.12f}.")
    else:
        lines.append("- Dense projected Overall(AoA0) is not available until all non-AoA columns are present.")
    lines.append("\nUnknown columns are not estimated here. Use this file to read the official payload as it actually exists.")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "out_json": rel(out_json), "out_md": rel(out_md), "known_delta_sum": known_delta_sum, "unknown_non_aoa_columns": unknown_non_aoa}, indent=2), flush=True)


if __name__ == "__main__":
    main()
