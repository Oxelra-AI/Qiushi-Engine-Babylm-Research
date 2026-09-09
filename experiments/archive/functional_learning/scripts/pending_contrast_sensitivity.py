#!/usr/bin/env python3
"""research: sensitivity algebra for paired-seed closing comparisons.

Reads the latest strict split-evaluation admission file and computes exact affine
forms for the still-missing seed62065 contrasts.  This allows conversion of
newly delivered component results into scientific contrasts without mixing score
surfaces or guessing from partial tables.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import pathlib
import time
from typing import Any, Dict, List

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
DEFAULT_ADMISSION = _public_path('experiments/archive/functional_learning/data/strict_split_eval_admission_after_o_boolq/strict_split_eval_admission.json')
OUT = _public_path('experiments/archive/functional_learning/data/pending_contrast_sensitivity')
SG_TASKS = ["boolq", "multirc", "rte", "wsc", "mrpc", "qqp", "mnli"]
OVERALL = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "SuperGLUE", "GlobalPIQA", "Reading", "AoA"]


def rel(p: pathlib.Path | str | None) -> str | None:
    if p is None:
        return None
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def finite(v) -> float | None:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def sum_known(vals: List[float | None]) -> tuple[float, List[int]]:
    s = 0.0
    missing = []
    for i, v in enumerate(vals):
        if v is None:
            missing.append(i)
        else:
            s += float(v)
    return s, missing


def solve_multirc_for_ms(target_overall: float, non_sg_sum: float, known_sg_sum6: float) -> float:
    # target_overall = (non_sg_sum + (known_sg_sum6 + m)/7)/9
    return 7.0 * (9.0 * target_overall - non_sg_sum) - known_sg_sum6


def main() -> None:
    adm = load(DEFAULT_ADMISSION)
    endpoints = adm["endpoints"]
    cp = adm["seed62064_counterparts"]
    out: Dict[str, Any] = {
        "status": "PENDING_CONTRAST_SENSITIVITY",
        "created_utc": now(),
        "admission_source": rel(DEFAULT_ADMISSION),
        "interpretation": "Affine forms and thresholds for still-missing seed62065 comparisons; not a final Overall while components are missing.",
    }

    # MS62065: only MultiRC is missing in strict admission.
    ms = endpoints["ms62065"]
    ms_scores = ms["scores"]
    ms_non_sg_sum = sum(float(ms_scores[k]) for k in OVERALL if k != "SuperGLUE" and ms_scores[k] is not None)
    ms_sg_scores = {t: finite(ms["superglue_task_scores"].get(t)) for t in SG_TASKS}
    ms_known_sg_sum = sum(v for t, v in ms_sg_scores.items() if t != "multirc" and v is not None)
    ms_known_missing = [t for t, v in ms_sg_scores.items() if v is None]
    ms64 = float(cp["densemask_sparselabel_seed62064"]["Overall"])
    clean65 = float(cp["clean_pres_lambda1_eval_seed62065"]["Overall"])
    clean64 = float(cp["clean_pres_lambda1_eval_seed62064"]["Overall"])
    ms64_resid = clean64 - ms64
    target_ms_for_same_resid = clean65 - ms64_resid
    out["ms62065"] = {
        "known_non_superglue_sum": ms_non_sg_sum,
        "known_superglue_sum_excluding_missing": ms_known_sg_sum,
        "missing_superglue_tasks": ms_known_missing,
        "affine_form": "if MultiRC=m, SG=(known_superglue_sum_excluding_missing+m)/7 and Overall=(known_non_superglue_sum+SG)/9",
        "multiRC_threshold_for_equal_MS64_overall": solve_multirc_for_ms(ms64, ms_non_sg_sum, ms_known_sg_sum),
        "multiRC_threshold_for_seed65_clean_minus_ms_equals_seed64_residual": solve_multirc_for_ms(target_ms_for_same_resid, ms_non_sg_sum, ms_known_sg_sum),
        "seed62064_ms_overall": ms64,
        "seed62065_clean_overall": clean65,
        "seed62064_clean_minus_ms": ms64_resid,
        "target_ms65_overall_for_same_clean_minus_ms_residual": target_ms_for_same_resid,
    }

    # O62065: BLiMP, COMPS, and MultiRC missing after AoA repair.
    o = endpoints["o62065"]
    o_scores = o["scores"]
    known_non = sum(float(o_scores[k]) for k in OVERALL if k not in {"BLiMP", "COMPS", "SuperGLUE"} and o_scores[k] is not None)
    o_sg_scores = {t: finite(o["superglue_task_scores"].get(t)) for t in SG_TASKS}
    o_known_sg_sum = sum(v for t, v in o_sg_scores.items() if t != "multirc" and v is not None)
    o64 = float(cp["ordinary_inherited_wwm_seed62064"]["Overall"])
    out["o62065"] = {
        "known_non_BLiMP_COMPS_superglue_sum": known_non,
        "known_superglue_sum_excluding_missing": o_known_sg_sum,
        "missing_components": o["missing_scores"],
        "missing_superglue_tasks": [t for t, v in o_sg_scores.items() if v is None],
        "affine_form": "if BLiMP=b, COMPS=c, MultiRC=m, SG=(known_superglue_sum_excluding_missing+m)/7 and Overall=(known_non_BLiMP_COMPS_superglue_sum+b+c+SG)/9",
        "seed62064_ordinary_overall": o64,
        "required_total_blimp_plus_comps_plus_multirc_over7_for_equal_o64": 9.0 * o64 - known_non - o_known_sg_sum / 7.0,
        "equivalently_threshold_for_multirc_given_blimp_comps": "m = 7*(9*O64 - known_non - b - c) - known_superglue_sum_excluding_missing",
    }

    OUT.mkdir(parents=True, exist_ok=True)
    out_json = _public_path('experiments/archive/functional_learning/data/pending_contrast_sensitivity/pending_contrast_sensitivity.json')
    out_md = _public_path('research/documents/functional_learning/data/pending_contrast_sensitivity/pending_contrast_sensitivity.md')
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research pending paired-contrast sensitivity\n\n", f"Created: `{out['created_utc']}`\n\n", f"Admission source: `{out['admission_source']}`\n\n"]
    lines.append("## MS62065\n\n")
    x = out["ms62065"]
    lines.append(f"Known non-SuperGLUE sum: `{x['known_non_superglue_sum']}`; known SG six-task sum: `{x['known_superglue_sum_excluding_missing']}`; missing: `{x['missing_superglue_tasks']}`.\n\n")
    lines.append(f"If MultiRC=m, Overall=( `{x['known_non_superglue_sum']}` + (`{x['known_superglue_sum_excluding_missing']}` + m)/7 )/9.\n\n")
    lines.append(f"MultiRC threshold for MS65 to equal MS64 Overall `{x['seed62064_ms_overall']}`: `{x['multiRC_threshold_for_equal_MS64_overall']}`.\n\n")
    lines.append(f"MultiRC threshold for clean65-MS65 to equal the seed64 residual `{x['seed62064_clean_minus_ms']}`: `{x['multiRC_threshold_for_seed65_clean_minus_ms_equals_seed64_residual']}`.\n\n")
    lines.append("## O62065\n\n")
    y = out["o62065"]
    lines.append(f"Known non-BLiMP/COMPS/SuperGLUE sum: `{y['known_non_BLiMP_COMPS_superglue_sum']}`; known SG six-task sum: `{y['known_superglue_sum_excluding_missing']}`; missing: `{y['missing_components']}`.\n\n")
    lines.append(f"If BLiMP=b, COMPS=c, MultiRC=m, Overall=( `{y['known_non_BLiMP_COMPS_superglue_sum']}` + b + c + (`{y['known_superglue_sum_excluding_missing']}` + m)/7 )/9.\n\n")
    lines.append(f"For equality with O64 Overall `{y['seed62064_ordinary_overall']}`, b+c+m/7 must equal `{y['required_total_blimp_plus_comps_plus_multirc_over7_for_equal_o64']}`.\n")
    out_md.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": out["status"], "out_json": rel(out_json), "out_md": rel(out_md)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
