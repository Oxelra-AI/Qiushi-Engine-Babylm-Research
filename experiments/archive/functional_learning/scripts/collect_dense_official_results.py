#!/usr/bin/env python3
"""research/continuation: collect dense official-compatible results when ready.

This script is intentionally conservative.  It reads the seed62064/seed62065
per-target payloads, checks which official columns are present, and only computes
complete official-coordinate transition comparisons when the required columns are
available.  It never substitutes fast-screen numbers for missing official values.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import pathlib
import subprocess
import sys
import time
from typing import Any, Dict, List

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
COMPARE = _public_path('experiments/archive/functional_learning/scripts/official_transition_compare.py')
PARENT_ZERO = "experiments/archive/frontier_consolidation/data/private_scale_sentinel_eval/coherent86_private_scale_0p75/per_target/coherent86_private_scale_0p75.json"
PARENT_SG = "experiments/archive/frontier_consolidation/data/private_scale_superglue_eval/coherent86_private_scale_0p75/per_target/coherent86_private_scale_0p75_sg_retry.json"
DENSE64 = "experiments/archive/functional_learning/data/dense_focus_official_eval_seed62064/per_target/dense_focus_seed62064_u0080.json"
DENSE65 = "experiments/archive/functional_learning/data/dense_focus_official_eval_seed62065/per_target/dense_focus_seed62065_u0080.json"
OUT = _public_path('experiments/archive/functional_learning/data/dense_official_collection')
OFFICIAL_REQUIRED = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading", "SuperGLUE", "AoA"]
OFFICIAL9 = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "SuperGLUE", "GlobalPIQA", "Reading", "AoA"]
SUPERGLUE_TASKS = ["boolq", "mnli", "mrpc", "multirc", "qqp", "rte", "wsc"]


def rel(path: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def as_path(path: str | pathlib.Path) -> pathlib.Path:
    p = pathlib.Path(path)
    return p if p.is_absolute() else ROOT / p


def load(path: str | pathlib.Path) -> Dict[str, Any] | None:
    p = as_path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def score_from_payload(payload: Dict[str, Any], col: str) -> float | None:
    tasks = payload.get("tasks") or {}
    if col == "GlobalPIQA":
        vals = []
        for k in ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]:
            rec = tasks.get(k) or {}
            if rec.get("score") is not None and rec.get("predictions"):
                vals.append(float(rec["score"]))
        return sum(vals) / 2.0 if len(vals) == 2 else None
    if col == "Reading":
        rec = tasks.get("Reading") or {}
        scores = rec.get("scores") or {}
        return None if scores.get("Reading") is None or not rec.get("predictions") else float(scores["Reading"])
    if col == "SuperGLUE":
        rec = tasks.get("SuperGLUE") or {}
        entries = rec.get("tasks") or []
        names = {str(x.get("task")) for x in entries if isinstance(x, dict) and x.get("returncode") == 0 and x.get("predictions")}
        if names != set(SUPERGLUE_TASKS):
            return None
        # The primary-metric patch is present only after eval_superglue has finished
        # all seven tasks and patch_superglue_primary_metric has rewritten the payload.
        if len(rec.get("superglue_primary_metric_details") or []) != len(SUPERGLUE_TASKS):
            return None
        return None if rec.get("superglue_mean") is None else float(rec["superglue_mean"])
    if col == "AoA":
        rec = tasks.get("AoA") or {}
        # Accept the arbitrary-single-checkpoint AoA=0 record only after the evaluator
        # has explicitly written an AoA task, not while the background job is still
        # between SuperGLUE and AoA.
        if not rec:
            return None
        for key in ["aoa_leaderboard_score", "aoa_for_provisional_overall"]:
            if rec.get(key) is not None:
                return float(rec[key])
        return None
    rec = tasks.get(col) or {}
    return None if rec.get("score") is None or not rec.get("predictions") else float(rec["score"])


def payload_status(label: str, path: str) -> Dict[str, Any]:
    payload = load(path)
    if payload is None:
        return {"label": label, "path": path, "exists": False, "complete": False, "present": [], "missing": OFFICIAL_REQUIRED, "scores": {}}
    present = []
    missing = []
    scores = {}
    for col in OFFICIAL_REQUIRED:
        val = score_from_payload(payload, col)
        scores[col] = val
        if val is None:
            missing.append(col)
        else:
            present.append(col)
    if scores.get("GlobalPIQA_parallel") is not None and scores.get("GlobalPIQA_nonparallel") is not None:
        scores["GlobalPIQA"] = (scores["GlobalPIQA_parallel"] + scores["GlobalPIQA_nonparallel"]) / 2.0
    if all(scores.get(k) is not None for k in OFFICIAL9):
        scores["projected_overall_aoa0"] = sum(float(scores[k]) for k in OFFICIAL9) / len(OFFICIAL9)
    if all(scores.get(k) is not None for k in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]):
        scores["cheap7_mean"] = sum(float(scores[k]) for k in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]) / 7.0
    return {"label": label, "path": path, "exists": True, "complete": len(missing) == 0, "present": present, "missing": missing, "scores": scores}


def run_compare(label: str, dense_path: str, out_root: pathlib.Path, tag: str) -> Dict[str, Any]:
    cmd = [
        sys.executable, rel(COMPARE),
        "--a-label", "coherent86_official_alpha075",
        "--a-payload", PARENT_ZERO, PARENT_SG,
        "--b-label", label,
        "--b-payload", dense_path,
        "--out-root", rel(out_root),
        "--tag", tag,
        "--include-superglue",
        "--max-examples", "20",
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return {"cmd": cmd, "returncode": proc.returncode, "elapsed_sec": round(time.time() - t0, 1), "stdout_tail": proc.stdout[-2000:], "stderr_tail": proc.stderr[-4000:]}


def transition_score(path: pathlib.Path) -> Dict[str, float] | None:
    if not path.exists():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    return d.get("score_summary", {}).get("b_computed_scores") or d.get("score_summary", {}).get("b_payload_scores")


def transition_deltas(path: pathlib.Path) -> Dict[str, float] | None:
    if not path.exists():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for r in d.get("score_summary", {}).get("rows", []):
        if r.get("computed_delta_b_minus_a") is not None:
            out[r["column"]] = float(r["computed_delta_b_minus_a"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", type=pathlib.Path, default=OUT)
    ap.add_argument("--dry-run", action="store_true", help="Only report payload completeness; do not run comparisons even if complete.")
    args = ap.parse_args()
    out = as_path(args.out_root)
    out.mkdir(parents=True, exist_ok=True)

    statuses = {
        "seed62064": payload_status("dense_seed62064", DENSE64),
        "seed62065": payload_status("dense_seed62065", DENSE65),
        "coherent86_zero": payload_status("coherent86_zero", PARENT_ZERO),
        "coherent86_sg": payload_status("coherent86_sg", PARENT_SG),
    }
    compare_results = {}
    if not args.dry_run:
        if statuses["seed62064"]["complete"]:
            compare_results["seed62064"] = run_compare("dense_seed62064_official", DENSE64, out / "seed62064_vs_coherent86", "dense62064_vs_coherent86_official_full")
        if statuses["seed62065"]["complete"]:
            compare_results["seed62065"] = run_compare("dense_seed62065_official", DENSE65, out / "seed62065_vs_coherent86", "dense62065_vs_coherent86_official_full")

    transition_paths = {
        "seed62064": out / "seed62064_vs_coherent86/dense62064_vs_coherent86_official_full_official_transition.json",
        "seed62065": out / "seed62065_vs_coherent86/dense62065_vs_coherent86_official_full_official_transition.json",
    }
    scores = {k: transition_score(p) for k, p in transition_paths.items()}
    deltas = {k: transition_deltas(p) for k, p in transition_paths.items()}
    seed_delta_diff = None
    if deltas.get("seed62064") and deltas.get("seed62065"):
        shared = sorted(set(deltas["seed62064"]) & set(deltas["seed62065"]))
        seed_delta_diff = {k: deltas["seed62065"][k] - deltas["seed62064"][k] for k in shared}

    result = {
        "status": "DENSE_OFFICIAL_COLLECTION",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "payload_statuses": statuses,
        "dry_run": bool(args.dry_run),
        "compare_results": compare_results,
        "transition_paths": {k: rel(p) for k, p in transition_paths.items()},
        "computed_scores_from_complete_transitions": scores,
        "computed_deltas_vs_coherent86": deltas,
        "seed62065_minus_seed62064_delta_difference": seed_delta_diff,
        "interpretation": "If payloads are incomplete, missing official columns remain unmeasured and must not be inferred. If complete, the transition files provide the exact official-coordinate comparison including SuperGLUE and AoA0 arithmetic.",
    }
    out_json = out / "dense_official_collection.json"
    out_md = out / "dense_official_collection.md"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research dense official collection\n"]
    for key, st in statuses.items():
        lines.append(f"- {key}: exists `{st['exists']}`, complete `{st['complete']}`, present `{st['present']}`, missing `{st['missing']}`.")
        if st.get("scores"):
            lines.append(f"  scores: `{json.dumps(st['scores'], ensure_ascii=False)}`")
    lines.append("")
    if compare_results:
        lines.append("## Comparisons run\n")
        for key, cr in compare_results.items():
            lines.append(f"- {key}: returncode `{cr['returncode']}`, path `{rel(transition_paths[key])}`.")
    else:
        lines.append("No complete dense official comparison was run in this invocation.\n")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "out_json": rel(out_json), "out_md": rel(out_md), "complete_seeds": [k for k in ["seed62064", "seed62065"] if statuses[k]["complete"]]}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
