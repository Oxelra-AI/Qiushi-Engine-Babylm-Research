#!/usr/bin/env python3
"""research: consolidate corrected-tokenizer full-eval coordinate readiness.

Reads the research dataset/code audits plus research GlobalPIQA lineage and the
patched wrapper files. Writes a compact JSON/note explaining which coordinates
are identical, which are repaired, and which wrapper patches are now load-bearing
for the research full official evaluations.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import re
import time
from pathlib import Path
from typing import Any

HERE = _public_path('experiments/archive/representation_and_objectives/scripts/eval_coordinate_readiness_summary.py')
A01 = _public_path('experiments/archive/representation_and_objectives')
USER_ROOT = _public_path('.')
WORKSPACE = _public_path('experiments/archive/representation_and_objectives')
OUT = _public_path('experiments/archive/representation_and_objectives/data/eval_coordinate_readiness/eval_coordinate_readiness_summary.json')
NOTE = _public_path('research/notes/representation_and_objectives/eval_coordinate_readiness_summary.md')

DATA_AUDIT = _public_path('experiments/archive/representation_and_objectives/data/eval_dataset_coordinate_audit/eval_dataset_coordinate_audit.json')
CODE_AUDIT = _public_path('experiments/archive/representation_and_objectives/data/eval_code_coordinate_audit/eval_code_coordinate_audit.json')
GPIQA = _public_path('experiments/archive/representation_and_objectives/data/globalpiqa_official_lineage/globalpiqa_official_lineage.json')
WRAPPERS = {
    "43022": _public_path('experiments/archive/representation_and_objectives/training/scripts/full_eval_strictsmalltok_seed43022.py'),
    "43122": _public_path('experiments/archive/representation_and_objectives/training/scripts/full_eval_strictsmalltok_seed43122.py'),
}
FULL_CONTROLLER = _public_path('experiments/archive/representation_and_objectives/scripts/strictsmalltok_posttrain_eval_controller.py')
PLAN = _public_path('experiments/archive/representation_and_objectives/data/two_seed_full_eval_plan/two_seed_full_eval_plan.json')
POLICY_REGRESSION = _public_path('experiments/archive/representation_and_objectives/data/projection_policy_regression/projection_policy_regression.json')


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(USER_ROOT))
    except Exception:
        return str(p)


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def wrapper_record(seed: str, path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    return {
        "path": rel(path),
        "exists": path.exists(),
        "uses_pristine_strict_override": "base.STRICT = PRISTINE_STRICT" in text,
        "uses_step038_globalpiqa_generated_paths": "GLOBALPIQA_GENERATED" in text and "global_piqa_parallel" in text and "global_piqa_nonparallel" in text,
        "still_mentions_initial_model_studies_script_import": "COMPACT_EXPERIENCE_SCRIPTS" in text,
        "note": "The wrapper still imports COMPACT_EXPERIENCE's robust runner module, but overrides base.STRICT to the pristine current checkout and redirects GlobalPIQA data to research official-generator files before calling base.main().",
    }


def main() -> None:
    data = load(DATA_AUDIT) or {}
    code = load(CODE_AUDIT) or {}
    gpiqa = load(GPIQA) or {}
    plan = load(PLAN) or {}
    reg = load(POLICY_REGRESSION) or {}
    wrappers = {seed: wrapper_record(seed, p) for seed, p in WRAPPERS.items()}
    non_repaired_data_identical = {}
    for task, rec in (data.get("tasks") or {}).items():
        if task in {"EWoK", "AoA", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel"}:
            continue
        non_repaired_data_identical[task] = bool(rec.get("identical_inventory_and_hashes"))
    gcheck = gpiqa.get("checks") or {}
    globalpiqa_resolved = bool(
        gcheck.get("all_generated_match_inherited_bytes")
        and gcheck.get("all_generated_counts_match_collator_expectation")
        and gcheck.get("initial_model_studies_dl_identical_to_pristine_dl")
    )
    code_groups = code.get("groups") or {}
    non_finetune_runner_identical = all((code_groups.get(k) or {}).get("identical_py_hashes") for k in ["sentence_zero_shot", "reading", "global_piqa"])
    finetune_diff = "finetune" in (code.get("runner_relevant_diffs") or {})
    wrapper_patch_ok = all(r["uses_pristine_strict_override"] and r["uses_step038_globalpiqa_generated_paths"] for r in wrappers.values())
    policy_ok = bool((reg.get("pass") is True) and (plan.get("policy") or {}).get("must_full_evaluate_both_completed_retrains") is True)
    payload = {
        "status": "EVAL_COORDINATE_READINESS_SUMMARY",
        "created_utc": now_utc(),
        "audits": {
            "dataset_audit": rel(DATA_AUDIT),
            "code_audit": rel(CODE_AUDIT),
            "globalpiqa_lineage": rel(GPIQA),
            "two_seed_plan": rel(PLAN),
            "projection_policy_regression": rel(POLICY_REGRESSION),
        },
        "non_repaired_data_identical_for": non_repaired_data_identical,
        "globalpiqa_resolved_by_step038_official_generator_byte_equivalence": globalpiqa_resolved,
        "ewok_repaired_by_step048_official_7618_row_script": True,
        "aoa_repaired_by_step048_min_context0_8005_context_script": True,
        "runner_code": {
            "sentence_zero_shot_reading_globalpiqa_identical_between_initial_model_studies_and_pristine": non_finetune_runner_identical,
            "finetune_classifier_diff_found": finetune_diff,
            "finetune_diff_resolution": "Patched both research A01 wrappers to set base.STRICT to the pristine current checkout before SuperGLUE finetune subprocesses run.",
            "wrappers": wrappers,
        },
        "policy": {
            "projection_policy_regression_pass": reg.get("pass"),
            "two_seed_plan_ready_seeds": plan.get("ready_seeds"),
            "both_completed_retrains_require_full_pristine_coordinate": True,
            "surface_first_role": "scheduling only, not endpoint selection or omission of second seed",
        },
        "ready_for_full_official_evaluation_after_patch": bool(all(non_repaired_data_identical.values()) and globalpiqa_resolved and non_finetune_runner_identical and finetune_diff and wrapper_patch_ok and policy_ok),
        "load_bearing_warning": "Do not run old wrappers without the pristine STRICT override; otherwise SuperGLUE uses the patched INITIAL_MODEL_STUDIES classifier_model.py rather than the current official checkout.",
    }
    _public_path('experiments/archive/representation_and_objectives/data/eval_coordinate_readiness').mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# research evaluation-coordinate readiness summary",
        "",
        f"Ready after patch: `{payload['ready_for_full_official_evaluation_after_patch']}`",
        "",
        "## What is identical without repair",
        "",
    ]
    for task, ok in non_repaired_data_identical.items():
        lines.append(f"- {task}: `{ok}`")
    lines.extend([
        "",
        "## What is intentionally repaired or supplied",
        "",
        f"- EWoK: research current official 7,618-row re-evaluation (`{payload['ewok_repaired_by_step048_official_7618_row_script']}`).",
        f"- AoA: research min_context=0, 8,005 contexts/checkpoint (`{payload['aoa_repaired_by_step048_min_context0_8005_context_script']}`).",
        f"- GlobalPIQA: research current official `dl.py` generated files are byte-identical to inherited files (`{globalpiqa_resolved}`).",
        "- SuperGLUE: wrapper subprocess root patched to the pristine current checkout because INITIAL_MODEL_STUDIES `finetune/classifier_model.py` differs from the official file.",
        "",
        "## Wrapper patch status",
        "",
    ])
    for seed, rec in wrappers.items():
        lines.append(f"- seed{seed}: pristine STRICT override `{rec['uses_pristine_strict_override']}`, research GlobalPIQA paths `{rec['uses_step038_globalpiqa_generated_paths']}` — `{rec['path']}`")
    lines.extend([
        "",
        "## Evaluation policy",
        "",
        "Both completed corrected-tokenizer retrains should receive the same full official coordinate. The seven-column surface can order work but not replace SuperGLUE/AoA/pristine collation or remove the second seed.",
    ])
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "out_json": rel(OUT),
        "note": rel(NOTE),
        "ready_for_full_official_evaluation_after_patch": payload["ready_for_full_official_evaluation_after_patch"],
        "globalpiqa_resolved": globalpiqa_resolved,
        "wrapper_patch_ok": wrapper_patch_ok,
    }, indent=2), flush=True)
    if not payload["ready_for_full_official_evaluation_after_patch"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
