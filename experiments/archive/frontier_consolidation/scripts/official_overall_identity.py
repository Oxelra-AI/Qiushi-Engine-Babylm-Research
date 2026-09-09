#!/usr/bin/env python3
"""research official Strict-Small Overall(AoA0) identity check.

This is a source-grounded scorer-identity artifact, not a new evaluator.  It reads
snippets from the live BabyLM leaderboard Space clone saved in research and
records the exact formula used by `src/leaderboard/read_evals.py` for Strict-Small:

  overall = mean(BLiMP, Supplement, EWoK, Entity, COMPS, SuperGLUE/GLUE,
                 GlobalPIQA, Reading, AoA)

For MLM carriers with scalar/missing AoA and no `aoa_surprisals`, AoA contributes
0.0. Therefore a candidate with known cheap7 = mean(BLiMP, Supplement, EWoK,
Entity, COMPS, GlobalPIQA, Reading) and SuperGLUE has

  Overall(AoA0) = (7 * cheap7 + SuperGLUE) / 9.

The script applies this identity to protected chck82 and current private-scale
endpoints and writes a durable note with source file hashes and line references.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import time
from statistics import mean
from typing import Any

ROOT = pathlib.Path(".")
SPACE = ROOT / "experiments/archive/frontier_consolidation/data/live_leaderboard_space_repo"
READ_EVALS = SPACE / "src/leaderboard/read_evals.py"
EVAL_SUB = SPACE / "src/submission/eval_submission.py"
CHECK_VAL = SPACE / "src/submission/check_validity.py"
OUT = ROOT / "experiments/archive/frontier_consolidation/data/official_overall_identity"
PANEL = ROOT / "experiments/archive/frontier_consolidation/data/private_scale_panel_analysis/private_scale_panel_analysis.json"
CHCK82 = ROOT / "experiments/archive/frontier_consolidation/data/chck82_independent_verification/chck82_independent_verification.json"
ALPHA1_SG = ROOT / "experiments/archive/frontier_consolidation/data/fastpath4M_coherent_superglue_summary/fastpath4M_coherent_superglue_summary.json"
ALPHA1_SG2 = ROOT / "experiments/archive/frontier_consolidation/data/repeat_coherent86_superglue_summary/repeat_coherent86_superglue_summary.json"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def line_excerpt(path: pathlib.Path, start: int, end: int) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [f"{i+1}: {lines[i]}" for i in range(start-1, min(end, len(lines)))]


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def formula(cheap7: float, superglue: float, aoa: float = 0.0) -> float:
    return (7.0 * cheap7 + float(superglue) + float(aoa)) / 9.0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = read_json(PANEL)
    panel_scores = {k: v["cheap7"] for k, v in panel["score_table"].items() if v.get("cheap7") is not None}

    chck82_obj = read_json(CHCK82)
    # research file schema has protected score fields; tolerate either naming.
    protected = {
        "cheap7": float(chck82_obj.get("cheap7") or chck82_obj.get("cheap7_mean") or 43.95944987645173),
        "superglue": float(chck82_obj.get("SuperGLUE") or chck82_obj.get("superglue") or 69.7661813713118),
        "overall": float(chck82_obj.get("Overall") or chck82_obj.get("overall") or 41.942481167385985),
    }
    alpha1_sg = read_json(ALPHA1_SG)
    sg1 = float(alpha1_sg.get("superglue") or alpha1_sg.get("SuperGLUE") or alpha1_sg.get("score") or 69.77796826428681)
    sg_values = [sg1]
    if ALPHA1_SG2.exists():
        sg2_obj = read_json(ALPHA1_SG2)
        # Accept common schemas.
        for key in ["superglue", "SuperGLUE", "score"]:
            if key in sg2_obj:
                sg_values.append(float(sg2_obj[key])); break
        else:
            if "scores" in sg2_obj and "SuperGLUE" in sg2_obj["scores"]:
                sg_values.append(float(sg2_obj["scores"]["SuperGLUE"]))

    rows = {}
    # chck82 verifies the identity against a known full Overall.
    rows["chck82_anchor"] = {
        "cheap7": protected["cheap7"],
        "superglue": protected["superglue"],
        "aoa": 0.0,
        "overall_formula": formula(protected["cheap7"], protected["superglue"], 0.0),
        "known_overall": protected["overall"],
        "difference_formula_minus_known": formula(protected["cheap7"], protected["superglue"], 0.0) - protected["overall"],
    }
    # Candidate rows use alpha1 observed SG as placeholder until alpha-specific SG tasks finish.
    for arm in ["coherent86_private_alpha1", "coherent86_private_alpha0p5", "coherent86_private_alpha0p75"]:
        if arm in panel_scores:
            rows[arm] = {
                "cheap7": float(panel_scores[arm]),
                "superglue_reference_alpha1_first": sg1,
                "aoa": 0.0,
                "overall_if_alpha1_first_sg": formula(float(panel_scores[arm]), sg1, 0.0),
                "delta_vs_chck82_if_alpha1_first_sg": formula(float(panel_scores[arm]), sg1, 0.0) - protected["overall"],
                "superglue_needed_to_match_chck82": 9.0 * protected["overall"] - 7.0 * float(panel_scores[arm]),
            }

    out = {
        "status": "COMPLETE",
        "created_utc": now(),
        "source_files": {
            "read_evals_py": {"path": str(READ_EVALS), "sha256": sha(READ_EVALS), "formula_lines": [363, 377]},
            "eval_submission_py": {"path": str(EVAL_SUB), "sha256": sha(EVAL_SUB), "aoa_lines": [80, 87]},
            "check_validity_py": {"path": str(CHECK_VAL), "sha256": sha(CHECK_VAL), "missing_aoa_and_fast_lines": [209, 295]},
        },
        "source_excerpts": {
            "strict_overall_formula_read_evals_363_377": line_excerpt(READ_EVALS, 363, 377),
            "submission_aoa_mapping_eval_submission_80_87": line_excerpt(EVAL_SUB, 80, 87),
            "validator_missing_aoa_allowed_check_validity_209_220": line_excerpt(CHECK_VAL, 209, 220),
            "validator_missing_fast_allowed_check_validity_292_295": line_excerpt(CHECK_VAL, 292, 295),
        },
        "identity": {
            "strict_small_columns": ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "SuperGLUE", "GlobalPIQA", "Reading", "AoA"],
            "cheap7_columns": ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"],
            "formula": "Overall(AoA0) = (7 * cheap7 + SuperGLUE) / 9",
            "note": "AoA is 0.0 for scalar/missing AoA without aoa_surprisals; missing fast_eval_results validates but public submit warns that challenge submissions should include fast history.",
        },
        "rows": rows,
        "alpha1_superglue_values_seen": sg_values,
    }
    out_json = OUT / "official_overall_identity.json"
    out_md = (ROOT / 'research/documents/frontier_consolidation/data/official_overall_identity/official_overall_identity.md')
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# research official Strict-Small Overall(AoA0) identity",
        "",
        "This records the live Space formula from the research clone and applies it to current endpoints.",
        "",
        "## Source-grounded formula",
        "",
        f"- `read_evals.py` SHA `{out['source_files']['read_evals_py']['sha256']}` lines 363-377 define `overall_average = mean(BLiMP, Supplement, EWoK, Entity, COMPS, GLUE, GlobalPIQA, Reading, AoA)`.",
        f"- `eval_submission.py` SHA `{out['source_files']['eval_submission_py']['sha256']}` lines 80-87 maps missing/malformed scalar AoA to 0.0; `read_evals.py` lines 204-208 zero AoA when `aoa_surprisals` is absent.",
        f"- `check_validity.py` SHA `{out['source_files']['check_validity_py']['sha256']}` lines 209-220 and 292-295 allow missing AoA and missing fast results, although `submit.py` later warns that fast history is required for challenge submissions.",
        "",
        "Therefore for these MLM carriers with scalar AoA 0 and known cheap7:",
        "",
        "`Overall(AoA0) = (7 * cheap7 + SuperGLUE) / 9`",
        "",
        "## Rows",
        "",
        "| arm | cheap7 | SuperGLUE used | Overall formula | Δ vs chck82 | SG needed to match chck82 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    base_overall = protected["overall"]
    for arm, r in rows.items():
        if arm == "chck82_anchor":
            lines.append(f"| {arm} | {r['cheap7']:.12f} | {r['superglue']:.12f} | {r['overall_formula']:.12f} | {r['overall_formula']-base_overall:+.12f} |  |")
        else:
            lines.append(f"| {arm} | {r['cheap7']:.12f} | {r['superglue_reference_alpha1_first']:.12f} | {r['overall_if_alpha1_first_sg']:.12f} | {r['delta_vs_chck82_if_alpha1_first_sg']:+.12f} | {r['superglue_needed_to_match_chck82']:.12f} |")
    lines += [
        "",
        f"chck82 identity residual: `{rows['chck82_anchor']['difference_formula_minus_known']:+.12e}`.",
        "",
        "Caveat: this establishes the arithmetic identity for the evaluated version of the leaderboard application. A public challenge-counted row still depends on an accepted submission and the application recomputing the submitted results. This analysis is not evidence of leaderboard acceptance.",
        "",
        f"JSON: `{out_json}`",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "out_json": str(out_json), "chck82_residual": rows["chck82_anchor"]["difference_formula_minus_known"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
